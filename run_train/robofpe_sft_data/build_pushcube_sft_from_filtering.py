#!/usr/bin/env python3
"""Build PushCube wrist-camera SFT data from Robometer filtering selections.

The filtering pool stores Robometer/render-camera videos plus actions and flat
ManiSkill states.  PushCube SFT needs base_camera + hand_camera videos, so this
script restores each saved env state and renders those two training cameras.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from mani_skill.utils import common
from tqdm import tqdm

FPS = 30
IMAGE_SIZE = 224
CHUNK_SIZE = 1000
TASK_ID = "PushCube-v1"
PROMPT = "Push the cube across the tabletop until its center lies inside the target region."
WORK_DIR = Path("/data/yingxi/robometer/failure_detection_5ind_10ood")
PRED_JSONL = WORK_DIR / "data_filtering_progress30" / "IND" / "ours_224" / "PushCube-v1.jsonl"
OUT_DIR = Path("/data/yingxi/datasets/robofpe_sft/PushCube-v1_wrist_filtered_top100_robometer/lerobot")
QUALITY_DIRS = ("successful_labeld", "successful_labeled", "suboptimal_labeled", "failure_labeled")
SUCCESS_LABELS = {"successful_labeld", "successful_labeled"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_arrow(path: Path) -> list[dict[str, Any]]:
    with pa.memory_map(str(path), "r") as source:
        try:
            table = ipc.open_stream(source).read_all()
        except Exception:
            source.seek(0)
            table = ipc.open_file(source).read_all()
    return table.to_pylist()


def quality_name(label: str | None) -> str:
    if label in SUCCESS_LABELS:
        return "success"
    if label == "suboptimal_labeled":
        return "suboptimal"
    if label == "failure_labeled":
        return "failure"
    return str(label or "unknown")


def score_progress(progress: list[Any], method: str) -> float:
    values = [float(v) for v in progress[:-1]]
    n = len(values)
    if not values:
        return -1.0
    if method == "final":
        return values[-1]
    if method == "delta":
        return values[-1] - values[0]
    if method == "late_mean_delta":
        late_count = max(1, math.ceil(0.3 * n))
        return 0.5 * (sum(values[-late_count:]) / late_count) + 0.5 * (values[-1] - values[0])
    if method == "slope":
        if n < 2:
            return -1.0
        mean_x = (n - 1) / 2.0
        mean_y = sum(values) / n
        var_x = sum((i - mean_x) ** 2 for i in range(n))
        if var_x < 1e-12:
            return -1.0
        return sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values)) / var_x
    if method == "pearson":
        if n < 2:
            return -1.0
        mean_x = (n - 1) / 2.0
        mean_y = sum(values) / n
        cov = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
        var_x = sum((i - mean_x) ** 2 for i in range(n))
        var_y = sum((v - mean_y) ** 2 for v in values)
        denom = math.sqrt(var_x * var_y)
        return cov / denom if denom >= 1e-12 else -1.0
    raise ValueError(f"unknown score method: {method}")


def selected_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(args.pred_jsonl)
    query_rows = [r for r in rows if r.get("query_task") == args.query_task]
    if args.domain:
        query_rows = [r for r in query_rows if str(r.get("domain", "")).upper() == args.domain.upper()]
    if not query_rows:
        raise ValueError(f"no rows for query_task={args.query_task} in {args.pred_jsonl}")

    scored: list[dict[str, Any]] = []
    for row in query_rows:
        item = dict(row)
        item["quality"] = item.get("quality") or quality_name(item.get("quality_label"))
        item["_filter_score"] = score_progress(item.get("progress_pred") or [], args.score_method)
        scored.append(item)
    ranked = sorted(scored, key=lambda item: (-float(item["_filter_score"]), str(item.get("row_key") or item.get("id") or "")))

    skipped = []
    selected = []
    for rank, row in enumerate(ranked, start=1):
        if args.source_task_id and row.get("task_id") != args.source_task_id:
            if rank <= args.top_k:
                skipped.append({"rank": rank, "task_id": row.get("task_id"), "quality": row.get("quality"), "id": row.get("id")})
            continue
        row["_rank_all_candidates"] = rank
        selected.append(row)
        if len(selected) >= args.top_k:
            break
    if len(selected) < args.top_k:
        raise RuntimeError(f"only selected {len(selected)} rows, requested top_k={args.top_k}")
    if args.limit is not None:
        selected = selected[: args.limit]

    meta = {
        "pred_jsonl": str(args.pred_jsonl),
        "domain": args.domain,
        "query_task": args.query_task,
        "source_task_id": args.source_task_id,
        "score_method": args.score_method,
        "requested_top_k": args.top_k,
        "written_episodes": len(selected),
        "skipped_non_source_in_first_top_k": skipped,
        "selection_note": "Rows are ranked by the finalized filtering score; non-PushCube rows are skipped because PushCube SFT replay requires PushCube env states.",
    }
    return selected, meta


def load_source_rows(work_dir: Path, task_id: str) -> dict[tuple[str, str], dict[str, Any]]:
    task_dir = work_dir / task_id
    if not task_dir.exists():
        raise FileNotFoundError(task_dir)
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for dirname in QUALITY_DIRS:
        dataset_dir = task_dir / dirname / "hf_dataset"
        for arrow in sorted(dataset_dir.glob("*.arrow")):
            for row in read_arrow(arrow):
                row = dict(row)
                row["_dataset_dir"] = str(dataset_dir)
                row["_arrow_path"] = str(arrow)
                index[(str(row.get("id")), str(row.get("quality_label")))] = row
    if not index:
        raise FileNotFoundError(f"no source rows found below {task_dir}")
    return index


def np_leaf(value: Any) -> np.ndarray:
    arr = common.to_numpy(value)
    arr = np.asarray(arr)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def camera_rgb(obs: dict[str, Any], camera_uid: str, image_size: int) -> np.ndarray:
    frame = np_leaf(obs["sensor_data"][camera_uid]["rgb"])
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[:2] != (image_size, image_size):
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return frame


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"no frames for {path}")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def array_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def schema(state_dim: int, action_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("observation.state", pa.list_(pa.float32(), state_dim)),
            pa.field("actions", pa.list_(pa.float32(), action_dim)),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("task", pa.string()),
            pa.field("prompt", pa.string()),
        ]
    )


def write_episode_parquet(path: Path, rows: list[dict[str, Any]], state_dim: int, action_dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, schema=schema(state_dim, action_dim), preserve_index=False)
    pq.write_table(table, path)


def features(state_dim: int, action_dim: int, image_size: int, total_videos: int) -> dict[str, Any]:
    base = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None, "fps": FPS},
        "actions": {"dtype": "float32", "shape": [action_dim], "names": None, "fps": FPS},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": FPS},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "task": {"dtype": "string", "shape": [1], "names": None, "fps": FPS},
        "prompt": {"dtype": "string", "shape": [1], "names": None, "fps": FPS},
    }
    for key in ("observation.images.top", "observation.images.wrist"):
        base[key] = {
            "dtype": "video",
            "shape": [image_size, image_size, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(FPS),
                "video.height": image_size,
                "video.width": image_size,
                "video.channels": 3,
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return base


def make_env(args: argparse.Namespace):
    return gym.make(
        TASK_ID,
        obs_mode="rgb",
        robot_uids="panda_wristcam",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        sensor_configs={
            "shader_pack": args.shader,
            "base_camera": {"width": args.image_size, "height": args.image_size, "shader_pack": args.shader},
            "hand_camera": {"width": args.image_size, "height": args.image_size, "shader_pack": args.shader},
        },
        sim_backend=args.sim_backend,
    )


def render_episode(env, source_row: dict[str, Any], selection: dict[str, Any], episode_index: int, global_index: int, args: argparse.Namespace):
    actions = np.asarray(source_row["actions"], dtype=np.float32)
    states = np.asarray(source_row["states"], dtype=np.float32)
    if actions.ndim != 2 or states.ndim != 2:
        raise ValueError(f"bad action/state rank for {selection.get('id')}: actions={actions.shape}, states={states.shape}")
    length = min(len(actions), len(states), args.max_episode_frames or len(actions))
    if length <= 0:
        raise ValueError(f"empty episode for {selection.get('id')}")

    env.reset(seed=args.seed + episode_index)
    rows = []
    top_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    for t in range(length):
        env.unwrapped.set_state(common.to_tensor(states[t : t + 1], device=env.unwrapped.device))
        obs = common.to_numpy(env.unwrapped.get_obs())
        state = np_leaf(obs["agent"]["qpos"]).astype(np.float32)
        action = actions[t].astype(np.float32)
        top_frames.append(camera_rgb(obs, "base_camera", args.image_size))
        wrist_frames.append(camera_rgb(obs, "hand_camera", args.image_size))
        rows.append(
            {
                "observation.state": state.tolist(),
                "actions": action.tolist(),
                "timestamp": float(t / FPS),
                "frame_index": int(t),
                "episode_index": int(episode_index),
                "index": int(global_index),
                "task_index": 0,
                "task": PROMPT,
                "prompt": PROMPT,
            }
        )
        global_index += 1
    return rows, top_frames, wrist_frames, global_index


def write_dataset(args: argparse.Namespace) -> None:
    selections, selection_meta = selected_records(args)
    source_index = load_source_rows(args.work_dir, TASK_ID)
    out = args.out_dir
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    rows_all: list[dict[str, Any]] = []
    episodes_meta: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    global_index = 0
    state_dim = None
    action_dim = None
    try:
        for episode_index, selection in enumerate(tqdm(selections, desc="render PushCube SFT")):
            key = (str(selection.get("id")), str(selection.get("quality_label")))
            if key not in source_index and selection.get("quality_label") in SUCCESS_LABELS:
                key = (str(selection.get("id")), "successful_labeled")
            source_row = source_index.get(key)
            if source_row is None:
                raise KeyError(f"source row not found for {key}")
            rows, top_frames, wrist_frames, global_index = render_episode(env, source_row, selection, episode_index, global_index, args)
            state_dim = len(rows[0]["observation.state"])
            action_dim = len(rows[0]["actions"])
            chunk = episode_index // CHUNK_SIZE
            write_episode_parquet(out / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet", rows, state_dim, action_dim)
            write_video(out / "videos" / f"chunk-{chunk:03d}" / "observation.images.top" / f"episode_{episode_index:06d}.mp4", top_frames, FPS)
            write_video(out / "videos" / f"chunk-{chunk:03d}" / "observation.images.wrist" / f"episode_{episode_index:06d}.mp4", wrist_frames, FPS)
            rows_all.extend(rows)
            source_meta = source_row.get("metadata") or {}
            episode_meta = {
                "episode_index": episode_index,
                "tasks": [PROMPT],
                "length": len(rows),
                "source_id": selection.get("id"),
                "quality": selection.get("quality"),
                "quality_label": selection.get("quality_label"),
                "filter_rank_all_candidates": selection.get("_rank_all_candidates"),
                "filter_score": selection.get("_filter_score"),
                "filter_model": args.model_name,
                "filter_score_method": args.score_method,
                "source_h5": source_meta.get("source_h5"),
                "source_traj_key": source_meta.get("source_traj_key"),
                "source_arrow_path": source_row.get("_arrow_path"),
                "metadata": source_meta,
            }
            episodes_meta.append(episode_meta)
            manifest_rows.append({k: episode_meta.get(k) for k in episode_meta if k != "metadata"})
    finally:
        env.close()

    if state_dim is None or action_dim is None:
        raise RuntimeError("no episodes written")

    meta_dir = out / "meta"
    meta_dir.mkdir(exist_ok=True)
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for item in episodes_meta:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"task_index": 0, "task": PROMPT}, ensure_ascii=False) + "\n")
    pd.DataFrame({"task_index": [0], "task": [PROMPT]}).to_parquet(meta_dir / "tasks.parquet", index=False)

    df_all = pd.DataFrame(rows_all)
    actions = np.stack(df_all["actions"].map(np.asarray).to_numpy()).astype(np.float32)
    states = np.stack(df_all["observation.state"].map(np.asarray).to_numpy()).astype(np.float32)
    (meta_dir / "stats.json").write_text(
        json.dumps({"actions": array_stats(actions), "observation.state": array_stats(states)}, indent=2) + "\n",
        encoding="utf-8",
    )
    data_size = sum(path.stat().st_size for path in out.glob("data/**/*.parquet"))
    info = {
        "codebase_version": "v2.0",
        "dataset_variant": "robofpe-wrist-sft-from-robometer-filtering-v1",
        "task_id": TASK_ID,
        "robot_type": "panda_wristcam",
        "total_episodes": len(episodes_meta),
        "total_frames": len(rows_all),
        "total_tasks": 1,
        "total_videos": len(episodes_meta) * 2,
        "total_chunks": int((len(episodes_meta) + CHUNK_SIZE - 1) // CHUNK_SIZE),
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "data_files_size_in_mb": int(data_size / (1024 * 1024)),
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features(state_dim, action_dim, args.image_size, len(episodes_meta) * 2),
        "training_cameras": ["observation.images.top", "observation.images.wrist"],
        "camera_uids": {"top": "base_camera", "wrist": "hand_camera"},
        "selection": selection_meta,
        "quality_counts": dict(Counter(item["quality"] for item in manifest_rows)),
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "filtering_selection_manifest.json").write_text(
        json.dumps({"selection": selection_meta, "episodes": manifest_rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out / "filtering_selection_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "episode_index",
            "source_id",
            "quality",
            "quality_label",
            "length",
            "filter_rank_all_candidates",
            "filter_score",
            "filter_model",
            "filter_score_method",
            "source_h5",
            "source_traj_key",
            "source_arrow_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{name: row.get(name, "") for name in fieldnames} for row in manifest_rows])
    print(f"Wrote {len(episodes_meta)} episodes / {len(rows_all)} frames to {out}")
    print(f"quality_counts={info['quality_counts']}")
    print(f"state_dim={state_dim}, action_dim={action_dim}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--pred-jsonl", type=Path, default=PRED_JSONL)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--domain", default="IND")
    parser.add_argument("--model-name", default="ours_224")
    parser.add_argument("--query-task", default=TASK_ID)
    parser.add_argument("--source-task-id", default=TASK_ID)
    parser.add_argument("--score-method", default="final", choices=["pearson", "final", "delta", "slope", "late_mean_delta"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit", type=int, help="debug: only write the first N selected episodes")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--shader", default="default")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    write_dataset(parse_args())
