#!/usr/bin/env python3
"""Filter LeRobot per-episode dataset for peg-insertion state policy training."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REQUIRED_COLUMNS = ["observation.peg_head_pose", "observation.hole_pose", "actions"]
DEFAULT_TASK = "insert the blue peg vertically into the orange hole"
MAD_SCALE = 0.67448975
MAD_EPS = 1e-12


@dataclass
class EpisodeRecord:
    source_episode_index: int
    source_path: str
    source_chunk: str
    length: int | None = None
    features: dict[str, float] | None = None
    feature_detail: dict[str, Any] | None = None
    reasons: list[str] = field(default_factory=list)
    robust_outliers: dict[str, float] = field(default_factory=dict)
    robust_zscores: dict[str, float] = field(default_factory=dict)
    accepted_stage2: bool = False
    accepted_final: bool = False
    output_episode_index: int | None = None
    output_file: str | None = None


@dataclass
class SourceMeta:
    info: dict[str, Any]
    tasks_map: dict[int, str]
    episode_tasks: dict[int, str]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _list_episode_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not paths:
        paths = sorted(data_dir.glob("episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet under {data_dir}")
    return paths

def _episode_id_from_path(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if not match:
        raise ValueError(f"Cannot parse episode id from {path}")
    return int(match.group(1))


def _episode_id_from_table(table: pa.Table, path: Path) -> int:
    if "episode_index" in table.schema.names and table.num_rows > 0:
        try:
            return int(table.column("episode_index")[0].as_py())
        except Exception:
            pass
    return _episode_id_from_path(path)


def _load_source_meta(dataset_root: Path) -> SourceMeta:
    info = _read_json(dataset_root / "meta" / "info.json")
    tasks_map: dict[int, str] = {}
    episode_tasks: dict[int, str] = {}

    for row in _read_jsonl(dataset_root / "meta" / "tasks.jsonl"):
        if "task_index" in row and "task" in row:
            tasks_map[int(row["task_index"])] = str(row["task"])

    for row in _read_jsonl(dataset_root / "meta" / "episodes.jsonl"):
        if "episode_index" not in row:
            continue
        ep = int(row["episode_index"])
        tasks = row.get("tasks")
        if isinstance(tasks, list) and tasks:
            episode_tasks[ep] = str(tasks[0])

    return SourceMeta(info=info, tasks_map=tasks_map, episode_tasks=episode_tasks)


def _xyz3(value: Any, column_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        raise ValueError(f"{column_name} shape invalid: dim={arr.size} < 3")
    if not np.isfinite(arr).all():
        raise ValueError(f"{column_name} contains NaN/Inf")
    return arr[:3]


def _compute_feature(source_id: int, source_path: Path, source_chunk: str, df: pd.DataFrame) -> tuple[dict[str, float], dict[str, Any]]:
    peg = np.stack([_xyz3(v, "observation.peg_head_pose") for v in df["observation.peg_head_pose"].to_numpy()]).astype(np.float32)
    hole = np.stack([_xyz3(v, "observation.hole_pose") for v in df["observation.hole_pose"].to_numpy()]).astype(np.float32)
    action = np.stack([_xyz3(v, "actions") for v in df["actions"].to_numpy()]).astype(np.float32)

    rel = peg - hole
    jumps = np.linalg.norm(np.diff(peg, axis=0), axis=1) if peg.shape[0] > 1 else np.zeros(0, dtype=np.float32)
    action_abs = np.abs(action)

    scalars: dict[str, float] = {
        "length": float(len(df)),
        "max_pos_jump": float(jumps.max()) if jumps.size > 0 else 0.0,
        "max_abs_coord": float(max(np.abs(peg).max(), np.abs(hole).max())),
        "max_abs_action_xyz": float(action_abs.max()),
    }
    detail = {
        "source_episode_index": int(source_id),
        "source_path": str(source_path),
        "source_chunk": str(source_chunk),
        "length": int(len(df)),
        "init_rel_xyz": rel[0].astype(float).tolist(),
        "final_rel_xyz": rel[-1].astype(float).tolist(),
        "traj_min_xyz": rel.min(axis=0).astype(float).tolist(),
        "traj_max_xyz": rel.max(axis=0).astype(float).tolist(),
        "traj_range_xyz": (rel.max(axis=0) - rel.min(axis=0)).astype(float).tolist(),
        "action_abs_mean_xyz": action_abs.mean(axis=0).astype(float).tolist(),
        "action_abs_max_xyz": action_abs.max(axis=0).astype(float).tolist(),
        "action_rms_xyz": np.sqrt(np.square(action).mean(axis=0)).astype(float).tolist(),
        "max_pos_jump": scalars["max_pos_jump"],
        "max_abs_coord": scalars["max_abs_coord"],
        "max_abs_action_xyz": scalars["max_abs_action_xyz"],
    }
    for i, axis in enumerate(["x", "y", "z"]):
        scalars[f"init_rel_{axis}"] = float(detail["init_rel_xyz"][i])
        scalars[f"final_rel_{axis}"] = float(detail["final_rel_xyz"][i])
        scalars[f"traj_min_{axis}"] = float(detail["traj_min_xyz"][i])
        scalars[f"traj_max_{axis}"] = float(detail["traj_max_xyz"][i])
        scalars[f"traj_range_{axis}"] = float(detail["traj_range_xyz"][i])
        scalars[f"action_abs_mean_{axis}"] = float(detail["action_abs_mean_xyz"][i])
        scalars[f"action_abs_max_{axis}"] = float(detail["action_abs_max_xyz"][i])
        scalars[f"action_rms_{axis}"] = float(detail["action_rms_xyz"][i])
    return scalars, detail


def _robust_stats(values_by_key: dict[str, np.ndarray]) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    stats: dict[str, dict[str, float]] = {}
    zscores: dict[str, np.ndarray] = {}
    for key, values in values_by_key.items():
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        if mad < MAD_EPS:
            z = np.where(np.isclose(values, med, atol=1e-12), 0.0, np.inf)
        else:
            z = np.abs(MAD_SCALE * (values - med) / mad)
        stats[key] = {"median": med, "mad": mad, "mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max())}
        zscores[key] = z.astype(np.float64)
    return stats, zscores


def _ensure_output_dir(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("output-dir must be different from input-dir")
    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(f"Output dir is not empty: {output_dir}")
        if overwrite:
            shutil.rmtree(output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)


def _discover_video_keys(input_dir: Path, info: dict[str, Any]) -> list[str]:
    keys: set[str] = set()
    features = info.get("features", {})
    if isinstance(features, dict):
        for key, spec in features.items():
            if isinstance(spec, dict) and spec.get("dtype") == "video":
                keys.add(str(key))
    videos_dir = input_dir / "videos"
    if videos_dir.exists():
        for chunk_dir in videos_dir.glob("chunk-*"):
            if not chunk_dir.is_dir():
                continue
            for key_dir in chunk_dir.iterdir():
                if key_dir.is_dir():
                    keys.add(key_dir.name)
    return sorted(keys)


def _hardlink_or_copy(src: Path, dst: Path, prefer_hardlink: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if prefer_hardlink:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def _stats_init(dim: int) -> dict[str, Any]:
    return {"sum": np.zeros(dim, dtype=np.float64), "sum_sq": np.zeros(dim, dtype=np.float64), "min": np.full(dim, np.inf, dtype=np.float64), "max": np.full(dim, -np.inf, dtype=np.float64), "count": 0}


def _stats_update(acc: dict[str, Any], arr: np.ndarray) -> None:
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] == 0:
        return
    acc["sum"] += values.sum(axis=0)
    acc["sum_sq"] += np.square(values).sum(axis=0)
    acc["min"] = np.minimum(acc["min"], values.min(axis=0))
    acc["max"] = np.maximum(acc["max"], values.max(axis=0))
    acc["count"] += int(values.shape[0])


def _stats_finalize(acc: dict[str, Any]) -> dict[str, Any]:
    count = int(acc["count"])
    if count <= 0:
        return {"mean": [], "std": [], "min": [], "max": [], "count": [0]}
    mean = acc["sum"] / count
    var = np.maximum(acc["sum_sq"] / count - np.square(mean), 0.0)
    return {"mean": mean.tolist(), "std": np.sqrt(var).tolist(), "min": acc["min"].tolist(), "max": acc["max"].tolist(), "count": [count]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter LeRobot state dataset episodes")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-episode-len", type=int, default=10)
    parser.add_argument("--robust-z-threshold", type=float, default=6.0)
    parser.add_argument(
        "--enable-robust-filter",
        action="store_true",
        help=(
            "Reject an episode when any diagnostic feature exceeds the robust-Z threshold. "
            "Disabled by default because task-dependent trajectory features are multimodal."
        ),
    )
    parser.add_argument("--hard-abs-coordinate-max", type=float, default=2.0)
    parser.add_argument("--hard-pos-jump-max", type=float, default=0.2)
    parser.add_argument("--hard-action-abs-max", type=float, default=0.1)
    parser.add_argument("--chunks-size", type=int, default=-1)
    parser.add_argument("--fps", type=float, default=-1.0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--no-hardlink", action="store_true")
    parser.add_argument("--fallback-task", default=DEFAULT_TASK)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    _ensure_output_dir(input_dir, output_dir, overwrite=bool(args.overwrite))

    data_dir = input_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Input dataset missing data dir: {data_dir}")

    source_meta = _load_source_meta(input_dir)
    source_info = copy.deepcopy(source_meta.info)
    source_chunks_size = int(source_info.get("chunks_size", 1000))
    chunks_size = int(args.chunks_size) if int(args.chunks_size) > 0 else source_chunks_size
    fps = float(args.fps) if float(args.fps) > 0 else float(source_info.get("fps", 20.0))

    episode_paths = _list_episode_paths(data_dir)
    if int(args.max_episodes) > 0:
        episode_paths = episode_paths[: int(args.max_episodes)]

    records: dict[int, EpisodeRecord] = {}
    for episode_path in episode_paths:
        table = pq.read_table(episode_path)
        source_id = _episode_id_from_table(table, episode_path)
        source_chunk = episode_path.parent.name if episode_path.parent.name.startswith("chunk-") else f"chunk-{source_id // max(source_chunks_size, 1):03d}"

        record = EpisodeRecord(source_episode_index=source_id, source_path=str(episode_path), source_chunk=source_chunk)

        missing = [name for name in REQUIRED_COLUMNS if name not in table.schema.names]
        if missing:
            record.reasons.append(f"missing_fields:{','.join(missing)}")
            records[source_id] = record
            continue

        df = table.select(REQUIRED_COLUMNS).to_pandas()
        record.length = int(len(df))
        if record.length < int(args.min_episode_len):
            record.reasons.append(f"episode_too_short:<{int(args.min_episode_len)}")
            records[source_id] = record
            continue

        try:
            scalars, detail = _compute_feature(source_id, episode_path, source_chunk, df)
            record.features = scalars
            record.feature_detail = detail
            record.length = int(detail["length"])
        except ValueError as exc:
            msg = str(exc)
            if "NaN/Inf" in msg:
                record.reasons.append(f"nan_or_inf:{msg}")
            else:
                record.reasons.append(f"shape_error:{msg}")

        records[source_id] = record

    all_ids = sorted(records.keys())
    stage1_ids = [eid for eid in all_ids if records[eid].features is not None and not records[eid].reasons]

    robust_stats: dict[str, dict[str, float]] = {}
    robust_zscores: dict[str, np.ndarray] = {}
    if stage1_ids:
        scalar_values: dict[str, list[float]] = {}
        for eid in stage1_ids:
            assert records[eid].features is not None
            for key, value in records[eid].features.items():
                scalar_values.setdefault(key, []).append(float(value))
        robust_stats, robust_zscores = _robust_stats({k: np.asarray(v, dtype=np.float64) for k, v in scalar_values.items()})

    z_th = float(args.robust_z_threshold)
    hard_coord = float(args.hard_abs_coordinate_max)
    hard_jump = float(args.hard_pos_jump_max)
    hard_action = float(args.hard_action_abs_max)

    for idx, eid in enumerate(stage1_ids):
        rec = records[eid]
        assert rec.features is not None

        if rec.features["max_abs_coord"] > hard_coord:
            rec.reasons.append(f"hard_abs_coordinate>{hard_coord}")
        if rec.features["max_pos_jump"] > hard_jump:
            rec.reasons.append(f"hard_pos_jump>{hard_jump}")
        if rec.features["max_abs_action_xyz"] > hard_action:
            rec.reasons.append(f"hard_action_abs>{hard_action}")

        over: dict[str, float] = {}
        for key, zarr in robust_zscores.items():
            z = float(zarr[idx])
            rec.robust_zscores[key] = z
            if math.isinf(z) or z > z_th:
                over[key] = z
        rec.robust_outliers = over
        if args.enable_robust_filter and over:
            top = sorted(over.items(), key=lambda x: x[1], reverse=True)[:8]
            msg = ",".join([f"{k}:{v:.3f}" if math.isfinite(v) else f"{k}:inf" for k, v in top])
            rec.reasons.append(f"robust_outlier>{z_th}:{msg}")

        rec.accepted_stage2 = not rec.reasons

    stage2_ids = [eid for eid in stage1_ids if records[eid].accepted_stage2]
    p95_length: float | None = None
    if stage2_ids:
        lengths = np.asarray([int(records[eid].length or 0) for eid in stage2_ids], dtype=np.float64)
        p95_length = float(np.percentile(lengths, 95.0))

    final_ids: list[int] = []
    for eid in stage2_ids:
        rec = records[eid]
        length = int(rec.length or 0)
        if p95_length is not None and float(length) > p95_length:
            rec.reasons.append(f"length_above_p95>{p95_length:.6f}")
            rec.accepted_final = False
        else:
            rec.accepted_final = True
            final_ids.append(eid)

    final_ids = sorted(final_ids)

    video_keys = _discover_video_keys(input_dir, source_info)
    copied_videos = 0
    copy_mode_count = {"hardlink": 0, "copy": 0}

    task_to_index: dict[str, int] = {}
    task_rows: list[dict[str, Any]] = []

    def _task_index(task_text: str) -> int:
        if task_text not in task_to_index:
            idx = len(task_to_index)
            task_to_index[task_text] = idx
            task_rows.append({"task_index": idx, "task": task_text})
        return task_to_index[task_text]

    stats_acc: dict[str, dict[str, Any]] = {}
    episodes_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    global_index = 0
    for new_ep_idx, source_id in enumerate(final_ids):
        rec = records[source_id]
        src_path = Path(rec.source_path)
        src_table = pq.read_table(src_path)
        df = src_table.to_pandas()

        task_text = source_meta.episode_tasks.get(source_id)
        if not task_text and "task" in df.columns and len(df) > 0 and isinstance(df["task"].iloc[0], str):
            task_text = str(df["task"].iloc[0])
        if not task_text and "task_index" in df.columns and len(df) > 0:
            try:
                task_text = source_meta.tasks_map.get(int(df["task_index"].iloc[0]))
            except Exception:
                task_text = None
        if not task_text:
            task_text = str(args.fallback_task)

        task_idx = _task_index(task_text)

        length = int(len(df))
        frame_index = np.arange(length, dtype=np.int64)
        frame_global = np.arange(global_index, global_index + length, dtype=np.int64)
        timestamp = frame_index.astype(np.float32) / np.float32(fps)

        df["episode_index"] = np.full(length, new_ep_idx, dtype=np.int64)
        df["frame_index"] = frame_index
        df["index"] = frame_global
        df["timestamp"] = timestamp
        df["task_index"] = np.full(length, task_idx, dtype=np.int64)
        df["task"] = np.asarray([task_text] * length, dtype=object)

        dst_chunk = f"chunk-{new_ep_idx // max(chunks_size, 1):03d}"
        dst_path = output_dir / "data" / dst_chunk / f"episode_{new_ep_idx:06d}.parquet"
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        out_table = pa.Table.from_pandas(df, preserve_index=False)
        if src_table.schema.metadata:
            out_schema = out_table.schema.with_metadata(src_table.schema.metadata)
            out_table = out_table.cast(out_schema)
        pq.write_table(out_table, dst_path)

        rec.output_episode_index = new_ep_idx
        rec.output_file = str(dst_path.relative_to(output_dir))

        for key in ["observation.peg_head_pose", "observation.hole_pose", "actions"]:
            if key in df.columns and length > 0:
                arr = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in df[key].to_numpy()]).astype(np.float64)
                if key not in stats_acc:
                    stats_acc[key] = _stats_init(arr.shape[1])
                _stats_update(stats_acc[key], arr)

        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if key in df.columns and length > 0:
                arr = np.asarray(df[key].to_numpy(), dtype=np.float64)
                if key not in stats_acc:
                    stats_acc[key] = _stats_init(1)
                _stats_update(stats_acc[key], arr)

        for video_key in video_keys:
            src_video = input_dir / "videos" / rec.source_chunk / video_key / f"episode_{source_id:06d}.mp4"
            if not src_video.exists():
                continue
            dst_video = output_dir / "videos" / dst_chunk / video_key / f"episode_{new_ep_idx:06d}.mp4"
            mode = _hardlink_or_copy(src_video, dst_video, prefer_hardlink=not bool(args.no_hardlink))
            copied_videos += 1
            copy_mode_count[mode] = int(copy_mode_count.get(mode, 0)) + 1

        episodes_rows.append({"episode_index": new_ep_idx, "source_episode_index": source_id, "dataset_from_index": global_index, "dataset_to_index": global_index + length, "tasks": [task_text], "length": length, "success": True})
        manifest_rows.append({"compact_episode_index": new_ep_idx, "source_episode_index": source_id, "compact_num_frames": length, "accepted": True, "output_file": str(dst_path.relative_to(output_dir)), "error": None})

        global_index += length

    for source_id in all_ids:
        rec = records[source_id]
        if rec.accepted_final:
            continue
        manifest_rows.append({"compact_episode_index": -1, "source_episode_index": source_id, "compact_num_frames": int(rec.length or 0), "accepted": False, "output_file": None, "error": "; ".join(rec.reasons) if rec.reasons else "filtered"})

    stats_json = {key: _stats_finalize(acc) for key, acc in stats_acc.items()}
    with (output_dir / "meta" / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2)

    with (output_dir / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes_rows:
            f.write(json.dumps(row) + "\n")

    with (output_dir / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for row in task_rows:
            f.write(json.dumps(row) + "\n")

    if task_rows:
        pq.write_table(pa.table({"task_index": pa.array([int(x["task_index"]) for x in task_rows], type=pa.int64()), "task": pa.array([str(x["task"]) for x in task_rows], type=pa.string())}), output_dir / "meta" / "tasks.parquet")

    info = copy.deepcopy(source_info) if source_info else {}
    info["codebase_version"] = info.get("codebase_version", "v2.0")
    info["total_episodes"] = len(final_ids)
    info["total_frames"] = int(global_index)
    info["total_tasks"] = len(task_rows)
    info["total_chunks"] = int(math.ceil(len(final_ids) / max(chunks_size, 1))) if final_ids else 0
    info["chunks_size"] = int(chunks_size)
    info["fps"] = float(fps)
    info["splits"] = {"train": f"0:{len(final_ids)}"}
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["total_videos"] = int(copied_videos)
    if copied_videos > 0:
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    else:
        info.pop("video_path", None)

    with (output_dir / "meta" / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    metadata = {"source_dir": str(input_dir), "output_dir": str(output_dir), "summary": {"num_episodes": len(all_ids), "num_accepted": len(final_ids), "num_rejected": len(all_ids) - len(final_ids), "accepted_rate": float(len(final_ids) / len(all_ids)) if all_ids else 0.0}, "manifest": manifest_rows}
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    reason_ids: dict[str, list[int]] = {}
    removed_ids: list[int] = []
    removed_details: list[dict[str, Any]] = []
    for source_id in all_ids:
        rec = records[source_id]
        if rec.accepted_final:
            continue
        removed_ids.append(source_id)
        removed_details.append({"source_episode_index": source_id, "source_path": rec.source_path, "length": rec.length, "reasons": rec.reasons, "robust_outliers": rec.robust_outliers, "feature_detail": rec.feature_detail})
        for reason in set(rec.reasons):
            category = reason.split(":", 1)[0].split(">", 1)[0]
            reason_ids.setdefault(category, []).append(source_id)

    report = {"input_dir": str(input_dir), "output_dir": str(output_dir), "thresholds": {"min_episode_len": int(args.min_episode_len), "robust_filter_enabled": bool(args.enable_robust_filter), "robust_z_threshold": z_th, "hard_abs_coordinate_max": hard_coord, "hard_pos_jump_max": hard_jump, "hard_action_abs_max": hard_action}, "counts": {"source_total": len(all_ids), "integrity_pass": len(stage1_ids), "hard_filter_pass": len(stage2_ids), "outlier_removed": len(all_ids) - len(stage2_ids), "p95_removed": len(stage2_ids) - len(final_ids), "kept_final": len(final_ids), "removed_total": len(all_ids) - len(final_ids)}, "p95_length": p95_length, "removed_source_episode_ids": sorted(removed_ids), "removed_details": removed_details, "reason_counts": {k: len(v) for k, v in reason_ids.items()}, "reason_episode_ids": {k: sorted(v) for k, v in reason_ids.items()}, "feature_stats": robust_stats, "video_copy": {"video_keys": video_keys, "copied_videos": copied_videos, "modes": copy_mode_count}}

    with (output_dir / "filter_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["counts"], indent=2))
    print(f"Wrote filtered dataset: {output_dir}")
    print(f"Wrote filter report: {output_dir / 'filter_report.json'}")


if __name__ == "__main__":
    main()
