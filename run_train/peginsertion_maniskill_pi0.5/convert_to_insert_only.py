#!/usr/bin/env python3
"""Convert the full pick-up-and-insert SFT dataset to insert-only.

Each source episode is cropped to the move-and-insert segment (transfer +
pre_insert + correction + insert), starting at the lift-end frame where the
peg is grasped, lifted, and stationary above the pickup point -- matching the
insert-only eval start state produced by PegInsertionLiftPlanner. The reach +
grasp + close_gripper + lift prefix is dropped.

Task descriptions are rewritten to insert-only wording (single-stage
"insert the peg into the hole" variants; two-stage "pick up ... insert"
prompts are excluded). The source dataset is read-only; all output goes to a
new dataset directory.

Crop criterion (per episode parquet):
  t_close   = first frame where actions[:,6] < 0   (gripper starts closing)
  t_lift_end= first frame after t_close where TCP z has plateaued (lift done)
              and lateral xy motion has not yet started (transfer onset).
              Uses observation.state_tcp[:,2] (z) and [:,0:2] (xy).

Outputs a LeRobot v2.0 dataset: parquets (all columns preserved, frame_index/
index rebased), re-cut mp4 videos, and rewritten meta/ (info.json,
episodes.jsonl, tasks.jsonl/parquet, stats.json).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from insert_only_crop import (
    EVAL_PROMPT,
    find_lift_end,
    generate_insert_only_prompts,
)


# --- video re-cut -----------------------------------------------------------

def recut_video(src_mp4: Path, dst_mp4: Path, start_frame: int, fps: int) -> int:
    """Decode src mp4 from start_frame, re-encode to dst. Returns frames written."""
    cap = cv2.VideoCapture(str(src_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {src_mp4}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_mp4), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open writer {dst_mp4}")
    # skip to start_frame
    for _ in range(start_frame):
        if not cap.grab():
            cap.release()
            writer.release()
            raise RuntimeError(f"video shorter than start_frame={start_frame}: {src_mp4}")
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    return written


# --- per-episode conversion -------------------------------------------------

def iter_episode_parquets(data_dir: Path):
    for chunk in sorted(data_dir.glob("chunk-*")):
        for ep in sorted(chunk.glob("episode_*.parquet")):
            yield ep


def convert_episode(
    src_parquet: Path,
    src_video_root: Path,
    dst_data: Path,
    dst_video_root: Path,
    new_ei: int,
    chunks_size: int,
    global_idx: int,
    task_index: int,
    prompt: str,
    fps: int,
    video_keys: list[str],
) -> tuple[int, int]:
    """Crop one episode to consecutive new_ei. Returns (src_len, dst_len).

    dst parquet/videos are named by new_ei and placed in chunk new_ei//chunks_size,
    so skipped source episodes leave no gap in the output indexing.
    """
    table = pq.read_table(src_parquet)
    schema = table.schema
    n_src = table.num_rows

    actions = np.stack(table.column("actions").to_pandas().to_numpy())
    state_tcp = np.stack(table.column("observation.state_tcp").to_pandas().to_numpy())
    t_lift = find_lift_end(actions, state_tcp)
    if t_lift is None or n_src - t_lift < 10:
        raise RuntimeError(f"no reliable lift-end (t_lift={t_lift}, len={n_src})")

    dst_len = n_src - t_lift
    new_chunk = f"chunk-{new_ei // chunks_size:03d}"
    new_stem = f"episode_{new_ei:06d}"
    dst_parquet = dst_data / new_chunk / f"{new_stem}.parquet"

    new_cols = {}
    for name in schema.names:
        col = table.column(name).to_pandas().to_numpy()
        if name == "task":
            new_cols[name] = pa.array([prompt] * dst_len, type=pa.string())
        elif name == "task_index":
            new_cols[name] = pa.array([task_index] * dst_len, type=pa.int64())
        elif name == "frame_index":
            new_cols[name] = pa.array(np.arange(dst_len, dtype=np.int64), type=pa.int64())
        elif name == "index":
            new_cols[name] = pa.array(np.arange(global_idx, global_idx + dst_len, dtype=np.int64), type=pa.int64())
        elif name == "timestamp":
            new_cols[name] = pa.array(np.arange(dst_len, dtype=np.float32) / float(fps), type=pa.float32())
        elif name == "episode_index":
            new_cols[name] = pa.array([new_ei] * dst_len, type=pa.int64())
        else:
            new_cols[name] = pa.array(list(col[t_lift:]))
    new_table = pa.table(new_cols, schema=schema)
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_parquet)

    # Re-cut videos (src named by source ei, dst named by new_ei).
    src_chunk = src_parquet.parent.name
    src_stem = src_parquet.stem
    for vk in video_keys:
        src_mp4 = src_video_root / src_chunk / vk / f"{src_stem}.mp4"
        dst_mp4 = dst_video_root / new_chunk / vk / f"{new_stem}.mp4"
        if not src_mp4.exists():
            raise RuntimeError(f"missing video {src_mp4}")
        written = recut_video(src_mp4, dst_mp4, t_lift, fps)
        if written != dst_len:
            raise RuntimeError(
                f"video frame mismatch for {vk}: wrote {written} vs parquet {dst_len} (src {src_mp4})"
            )
    return n_src, dst_len


# --- meta rebuild -----------------------------------------------------------

def rebuild_meta(
    dst_meta: Path,
    src_info: dict,
    episode_lengths: list[int],
    episode_task_index: dict[int, int],
    prompts: list[str],
) -> None:
    dst_meta.mkdir(parents=True, exist_ok=True)
    # info.json: clone, update total_frames/total_tasks.
    info = json.loads(json.dumps(src_info))
    info["total_episodes"] = len(episode_lengths)
    info["total_frames"] = int(sum(episode_lengths))
    info["total_tasks"] = len(prompts)
    info["splits"] = {"train": f"0:{len(episode_lengths)}"}
    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # tasks.jsonl + tasks.parquet
    with open(dst_meta / "tasks.jsonl", "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"task_index": i, "task": p}) + "\n")
    pq.write_table(
        pa.table({
            "task_index": pa.array(range(len(prompts)), type=pa.int64()),
            "task": pa.array(prompts, type=pa.string()),
        }),
        dst_meta / "tasks.parquet",
    )

    # episodes.jsonl: rebase from/to_index, length, tasks.
    from_idx = 0
    with open(dst_meta / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ei, length in enumerate(episode_lengths):
            ti = episode_task_index[ei]
            rec = {
                "episode_index": ei,
                "dataset_from_index": from_idx,
                "dataset_to_index": from_idx + length,
                "tasks": [prompts[ti]],
                "length": length,
                "success": True,
            }
            f.write(json.dumps(rec) + "\n")
            from_idx += length

    # stats.json: recompute state/action/task_index stats from cropped parquets (done in finalize).


def compute_stats(dst_data_dir: Path, dst_meta: Path, fps: int) -> None:
    """Recompute stats.json from cropped parquets (state_tcp, actions, task_index)."""
    sums = {}
    squares = {}
    counts = {}
    mins = {}
    maxs = {}
    task_idx_vals = []
    for ep in iter_episode_parquets(dst_data_dir):
        t = pq.read_table(ep)
        for name in ["observation.state_tcp", "actions", "observation.state",
                     "debug.env_action", "debug.ref_next_tcp", "debug.tcp_before",
                     "debug.tcp_after", "episode_reset_state"]:
            if name not in t.schema.names:
                continue
            arr = np.stack(t.column(name).to_pandas().to_numpy()).astype(np.float64)
            if name not in sums:
                sums[name] = arr.sum(axis=0)
                squares[name] = (arr ** 2).sum(axis=0)
                counts[name] = arr.shape[0]
                mins[name] = arr.min(axis=0)
                maxs[name] = arr.max(axis=0)
            else:
                sums[name] += arr.sum(axis=0)
                squares[name] += (arr ** 2).sum(axis=0)
                counts[name] += arr.shape[0]
                mins[name] = np.minimum(mins[name], arr.min(axis=0))
                maxs[name] = np.maximum(maxs[name], arr.max(axis=0))
        ti = np.asarray(t.column("task_index").to_pandas().to_numpy())
        task_idx_vals.extend(ti.tolist())
    stats = {}
    for name in sums:
        c = counts[name]
        mean = sums[name] / c
        std = np.sqrt(squares[name] / c - mean ** 2)
        stats[name] = {
            "mean": mean.tolist(), "std": std.tolist(),
            "min": mins[name].tolist(), "max": maxs[name].tolist(),
            "count": [int(c)],
        }
    if task_idx_vals:
        arr = np.asarray(task_idx_vals, dtype=np.float64)
        stats["task_index"] = {
            "mean": [float(arr.mean())], "std": [float(arr.std())],
            "min": [int(arr.min())], "max": [int(arr.max())],
            "count": [len(task_idx_vals)],
        }
    with open(dst_meta / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# --- main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--dst-dir", required=True)
    ap.add_argument("--num-prompts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=None, help="limit (small-batch test)")
    ap.add_argument("--skip-stats", action="store_true", help="skip stats.json recompute")
    args = ap.parse_args()

    src = Path(args.src_dir).resolve()
    dst = Path(args.dst_dir).resolve()
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"dst dir not empty: {dst}")
    if not (src / "meta" / "info.json").exists():
        raise SystemExit(f"src meta/info.json not found: {src}")

    with open(src / "meta" / "info.json") as f:
        src_info = json.load(f)
    fps = int(src_info.get("fps", 20))
    video_keys = [k for k, v in src_info.get("features", {}).items()
                  if isinstance(v, dict) and v.get("dtype") == "video"]

    prompts = generate_insert_only_prompts(args.num_prompts, args.seed)
    assert prompts[0] == EVAL_PROMPT
    print(f"generated {len(prompts)} insert-only prompts (idx0={EVAL_PROMPT!r})")

    src_data = src / "data"
    src_videos = src / "videos"
    dst_data = dst / "data"
    dst_videos = dst / "videos"
    dst_meta = dst / "meta"

    ep_paths = list(iter_episode_parquets(src_data))
    if args.max_episodes is not None:
        ep_paths = ep_paths[: args.max_episodes]
    chunks_size = int(src_info.get("chunks_size", 1000))
    print(f"converting {len(ep_paths)} episodes -> {dst} (chunks_size={chunks_size})")

    # Deterministic source-episode->prompt assignment (covers all prompts first).
    rng = random.Random(args.seed)
    pool = list(range(len(prompts)))
    rng.shuffle(pool)
    src_task_index = {}
    for offset, ep in enumerate(ep_paths):
        ei = int(ep.stem.split("_")[-1])
        src_task_index[ei] = pool[offset] if offset < len(pool) else rng.choice(pool)

    new_lengths = []
    new_task_index = {}
    skipped = []
    global_idx = 0
    new_ei = 0
    done = 0
    for ep_path in ep_paths:
        src_ei = int(ep_path.stem.split("_")[-1])
        ti = src_task_index[src_ei]
        try:
            n_src, n_dst = convert_episode(
                ep_path, src_videos, dst_data, dst_videos,
                new_ei, chunks_size, global_idx,
                ti, prompts[ti], fps, video_keys,
            )
        except Exception as exc:
            print(f"  SKIP src_ep{src_ei}: {exc}")
            skipped.append((src_ei, str(exc)))
            continue
        new_lengths.append(n_dst)
        new_task_index[new_ei] = ti
        global_idx += n_dst
        new_ei += 1
        done += 1
        if done % 200 == 0:
            print(f"  converted {done}/{len(ep_paths)} (src_ep{src_ei} -> new_ep{new_ei-1}: {n_src}->{n_dst})")

    rebuild_meta(dst_meta, src_info, new_lengths, new_task_index, prompts)
    if not args.skip_stats:
        print("computing stats.json ...")
        compute_stats(dst_data, dst_meta, fps)

    # progress / conversion log
    with open(dst_meta / "collection_progress.json", "w") as f:
        json.dump({
            "converted_from": str(src),
            "total_episodes": len(new_lengths),
            "total_frames": int(sum(new_lengths)),
            "skipped": skipped,
            "prompts": len(prompts),
        }, f, indent=2)

    print(f"\ndone. episodes={len(new_lengths)} frames={sum(new_lengths)} skipped={len(skipped)}")
    print(f"output: {dst}")
    if skipped:
        print(f"skipped episodes: {[s[0] for s in skipped][:20]}{' ...' if len(skipped)>20 else ''}")


if __name__ == "__main__":
    main()
