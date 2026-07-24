#!/usr/bin/env python3
"""Merge the original insert-only dataset with HG-DAgger shards into ONE LeRobot-v2 dir.

The pi0.5 SFT path's ``resolve_lerobot_repo_id`` takes only ``data_paths[0]``, so
there is no native multi-dataset merge. This script concatenates N input LeRobot-v2
dirs (the original insert-only 3200 + the per-GPU HG-DAgger shards) into one dir,
offsetting ``episode_index`` / ``index`` / ``frame_index`` / ``timestamp`` so they
stay globally unique, and adds a per-frame ``source`` column
(0 = historical / original, 1 = HG-new) that the WeightedRandomSampler in
``fsdp_vla_sft_worker.build_dataloader`` keys on for the controllable new-data
fraction.

NO on-disk duplication: the per-batch new-data fraction is enforced by the sampler,
not by repeating trajectories (avoids exact-trajectory overfitting). Videos are
copied (not re-encoded) with offset episode ids; parquet is re-chunked per the
LeRobot layout (one parquet per episode, CHUNK_SIZE episodes per chunk). Meta is
rebuilt from scratch via the fork's ``_write_dataset_metadata`` (so the merged dir
has NO ``meta/openpi/<config>/norm_stats.json`` -- ``sft_finetune_pi05base.sh``
auto-recomputes it over the merged distribution).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent  # .../hg_dagger
_REPO = _HERE.parents[2]  # .../RLinf_RoboFAPE
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import collect_hg_dagger_data as ctl  # noqa: E402  (fork helper lib)

SOURCE_HISTORICAL = 0
SOURCE_HG_NEW = 1

# Fixed-length list fields + their lengths. The original insert-only dataset was
# collected by an older controller collector that lacked the per-frame peg/hole
# poses, so a concat leaves those columns NaN for historical rows. They are
# diagnostics (NOT SFT model inputs -- the dataconfig only maps
# observation.state_tcp + images + actions), so filling missing ones with zero
# lists is SFT-safe. episode_reset_state is variable-length + present in all
# current inputs, so not filled here.
_LIST_FIELD_LENGTHS = {
    "actions": 7,
    "debug.env_action": 7,
    "debug.ref_next_tcp": 16,
    "debug.tcp_before": 16,
    "debug.tcp_after": 16,
    "observation.state": 8,
    "observation.state_tcp": 8,
    "observation.peg_pose": 7,
    "observation.peg_head_pose": 7,
    "observation.hole_pose": 7,
}


def _ensure_list_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Fill any missing fixed-length list field with zero lists (SFT-safe)."""
    for field, length in _LIST_FIELD_LENGTHS.items():
        if field not in df.columns:
            df[field] = [np.zeros(length, dtype=np.float32).tolist()] * len(df)
    return df


# The fork's parquet schema extended with the per-frame provenance column. The
# OpenPI dataconfig ignores unknown columns, so ``source`` is SFT-safe; the
# WeightedRandomSampler reads it directly from the parquet.
_VIDEO_KEYS = [
    "observation.images.top",
    "observation.images.wrist",
    "observation.images.wrist_back",
    "observation.images.render",
]


def _merged_schema() -> pa.Schema:
    fields = list(ctl._dataset_schema())
    fields.append(pa.field("source", pa.int8()))
    return pa.schema(fields)


def _read_episodes_jsonl(d: str) -> list[dict[str, Any]]:
    path = os.path.join(d, "meta", "episodes.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _read_tasks_jsonl(d: str) -> list[tuple[int, str]]:
    """Return [(task_index, task), ...] from the dir's tasks.jsonl."""
    path = os.path.join(d, "meta", "tasks.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append((int(rec["task_index"]), str(rec["task"])))
    return out


def _read_dataset_parquet(d: str) -> pd.DataFrame:
    parquets = sorted(Path(d).rglob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)


def _discover_hg_shards(hg_dir: str) -> list[str]:
    """A collection outdir holds per-GPU shards under shard_gpu<id>/."""
    shards = sorted(glob.glob(os.path.join(hg_dir, "shard_gpu*")))
    return [s for s in shards if os.path.isdir(s)]


def _write_merged_episode(outdir: str, episode_df: pd.DataFrame, episode_index: int) -> None:
    episode_df = ctl._prepare_dataset_df(episode_df.reset_index(drop=True))
    chunk = episode_index // ctl.CHUNK_SIZE
    chunk_dir = os.path.join(outdir, "data", f"chunk-{chunk:03d}")
    os.makedirs(chunk_dir, exist_ok=True)
    # Build the table column-by-column with explicit types so list elements cast
    # to float32 (from_pandas does not cast list<item> float64->float32, which
    # breaks on re-read data where Python floats are float64).
    schema = _merged_schema()
    data = {}
    for field in schema:
        col = episode_df[field.name].tolist()
        try:
            data[field.name] = pa.array(col, type=field.type)
        except (pa.ArrowNotImplementedError, pa.ArrowInvalid, pa.ArrowTypeError):
            # Fall back: coerce scalars/objects to the field type via numpy.
            data[field.name] = pa.array(
                [np.asarray(v, dtype=np.float32).tolist() if field.type.id == pa.list_(pa.float32()).id else v
                 for v in col],
                type=field.type,
            )
    table = pa.table(data, schema=schema)
    pq.write_table(table, os.path.join(chunk_dir, f"episode_{episode_index:06d}.parquet"))


def _link_episode_videos(
    src_dir: str, dst_dir: str, old_ep: int, new_ep: int
) -> None:
    """Symlink each episode's videos into the merged dir (offset episode id).

    Symlink (not copy) so merging the 3200-episode original does not duplicate
    ~12k videos / dozens of GB -- the LeRobot loader follows symlinks. Source
    paths are made absolute so the links survive regardless of the merged dir's
    location.
    """
    old_chunk = old_ep // ctl.CHUNK_SIZE
    new_chunk = new_ep // ctl.CHUNK_SIZE
    for key in _VIDEO_KEYS:
        src = os.path.join(
            src_dir, "videos", f"chunk-{old_chunk:03d}", key, f"episode_{old_ep:06d}.mp4"
        )
        if not os.path.exists(src):
            continue
        src = os.path.abspath(src)
        dst = os.path.join(
            dst_dir, "videos", f"chunk-{new_chunk:03d}", key, f"episode_{new_ep:06d}.mp4"
        )
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        try:
            os.symlink(src, dst)
        except OSError:
            # Fall back to a copy if symlinks are unavailable (rare on Linux).
            shutil.copy2(src, dst)


def merge(
    orig_dirs: list[str],
    hg_dirs: list[str],
    out_dir: str,
) -> dict[str, Any]:
    # (dir, source) pairs; historical first so its episode ids are the low range.
    inputs: list[tuple[str, int]] = [(d, SOURCE_HISTORICAL) for d in orig_dirs]
    for d in hg_dirs:
        shards = _discover_hg_shards(d)
        if shards:
            for s in shards:
                inputs.append((s, SOURCE_HG_NEW))
        else:
            inputs.append((d, SOURCE_HG_NEW))

    if os.path.exists(out_dir):
        raise FileExistsError(
            f"Output dir exists (merge refuses to clobber): {out_dir}. "
            "Remove it first."
        )
    os.makedirs(out_dir, exist_ok=True)

    merged_parts: list[pd.DataFrame] = []
    merged_episodes: list[dict[str, Any]] = []
    merged_tasks: list[str] = []
    task_to_idx: dict[str, int] = {}
    ep_offset = 0
    idx_offset = 0
    video_copies: list[tuple[str, int, int]] = []  # (src_dir, old_ep, new_ep)
    source_counts = {SOURCE_HISTORICAL: 0, SOURCE_HG_NEW: 0}

    for src_dir, source in inputs:
        df = _read_dataset_parquet(src_dir)
        if df.empty:
            print(f"[merge] skip empty input: {src_dir}", flush=True)
            continue
        df = _ensure_list_fields(df)
        old_tasks = _read_tasks_jsonl(src_dir)
        # Map old task_index -> merged task_index (union, stable order).
        tmap: dict[int, int] = {}
        for old_ti, task in sorted(old_tasks):
            if task not in task_to_idx:
                task_to_idx[task] = len(merged_tasks)
                merged_tasks.append(task)
            tmap[old_ti] = task_to_idx[task]
        if tmap:
            df["task_index"] = df["task_index"].map(tmap).astype("int64")
        df["task"] = df["task"].astype("string")

        # Offset episode_index + global index so they stay unique across inputs.
        df["episode_index"] = (df["episode_index"].astype("int64") + ep_offset)
        df["index"] = (df["index"].astype("int64") + idx_offset)
        # Re-derive per-episode frame_index + timestamp (0..len-1) so the merged
        # dataset is internally consistent regardless of the source's indexing.
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        df["frame_index"] = (
            df.groupby("episode_index").cumcount().astype("int64")
        )
        df["timestamp"] = (df["frame_index"].astype("float32") / float(ctl.FPS)).astype("float32")
        df["source"] = np.int8(source)

        old_to_new: dict[int, int] = {}
        for old_ep in sorted(df["episode_index"].unique()):
            new_ep = int(old_ep)  # already offset
            old_to_new[int(old_ep)] = new_ep
            ep_df = df[df["episode_index"] == new_ep]
            length = len(ep_df)
            merged_episodes.append(
                {
                    "episode_index": new_ep,
                    "dataset_from_index": int(ep_df["index"].min()),
                    "dataset_to_index": int(ep_df["index"].max()) + 1,
                    "tasks": [str(t) for t in ep_df["task"].unique()],
                    "length": int(length),
                    "success": True,
                }
            )

        # Videos: old_ep (pre-offset within THIS input) -> new_ep. The df above is
        # already offset, so new_ep == current episode_index. For the video source
        # path we need the ORIGINAL (pre-offset) episode id == new_ep - ep_offset.
        for new_ep in sorted(df["episode_index"].unique()):
            old_ep_in_input = int(new_ep) - ep_offset
            video_copies.append((src_dir, old_ep_in_input, int(new_ep)))

        merged_parts.append(df)
        source_counts[source] += int(len(df))
        ep_offset = int(df["episode_index"].max()) + 1
        idx_offset = int(df["index"].max()) + 1
        print(
            f"[merge] {src_dir} source={source}: +{len(df)} frames "
            f"(cumulative episodes={ep_offset}, frames={idx_offset})",
            flush=True,
        )

    if not merged_parts:
        raise RuntimeError("All inputs were empty -- nothing to merge.")

    merged_df = pd.concat(merged_parts, ignore_index=True)

    # Write per-episode parquet (extended schema with source).
    for episode_index in sorted(merged_df["episode_index"].unique()):
        ep_df = merged_df[merged_df["episode_index"] == episode_index]
        _write_merged_episode(out_dir, ep_df, int(episode_index))

    # Link videos (symlink) with offset episode ids.
    for src_dir, old_ep, new_ep in video_copies:
        _link_episode_videos(src_dir, out_dir, old_ep, new_ep)

    # Meta (episodes.jsonl, tasks.{jsonl,parquet}, stats.json, info.json). Uses the
    # fork's writer so the layout matches what SFT/eval expect.
    ctl._write_dataset_metadata(
        out_dir,
        merged_df,
        merged_episodes,
        merged_tasks,
        write_stats=True,
        collect_mode="insert_only",
    )

    # Copy the historical dataset's openpi norm_stats into the merged dir so
    # sft_finetune_pi05base.sh uses it (skipping calculate_norm_stats.py, which
    # mishandles local-path LeRobot datasets in this lerobot/openpi version --
    # LeRobotDatasetMetadata(repo_id, root=None) treats a local path as an HF
    # repo id). The HG data is a small fraction of the merge, so the historical
    # norm_stats is a valid approximation; the `source` column + WeightedRandomSampler
    # still control the per-batch new-data fraction regardless.
    for src_dir, source in inputs:
        if source != SOURCE_HISTORICAL:
            continue
        src_openpi = os.path.join(src_dir, "meta", "openpi")
        if not os.path.isdir(src_openpi):
            continue
        dst_openpi = os.path.join(out_dir, "meta", "openpi")
        for root, _dirs, files in os.walk(src_openpi):
            rel = os.path.relpath(root, src_openpi)
            target = os.path.join(dst_openpi, rel) if rel != "." else dst_openpi
            os.makedirs(target, exist_ok=True)
            for fname in files:
                shutil.copy2(os.path.join(root, fname), os.path.join(target, fname))
        print(f"[merge] copied openpi norm_stats from {src_dir} -> {dst_openpi}", flush=True)
        break

    # Drop video keys whose videos are absent for some episodes. The original
    # insert-only dataset was collected WITHOUT a wrist_back camera (only
    # top/wrist/render), but the fork's _features() declares all four -> the
    # LeRobotDataset file-existence assertion would fail (3197 missing
    # wrist_back videos). The wrist SFT config uses base+wrist only (not
    # wrist_back), so dropping the absent key from `features` is SFT-safe.
    _info_path = os.path.join(out_dir, "meta", "info.json")
    with open(_info_path, encoding="utf-8") as f:
        _info = json.load(f)
    _feats = _info.get("features", {})
    _chunk0 = os.path.join(out_dir, "videos", "chunk-000")
    _dropped = []
    for _key in list(_feats.keys()):
        _ft = _feats[_key]
        if isinstance(_ft, dict) and _ft.get("dtype") == "video":
            _ep0 = os.path.join(_chunk0, _key, "episode_000000.mp4")
            if not os.path.exists(_ep0):
                del _feats[_key]
                _dropped.append(_key)
    if _dropped:
        _info["features"] = _feats
        _n_eps = int(_info.get("total_episodes", 0))
        _remaining_vid = sum(
            1 for v in _feats.values()
            if isinstance(v, dict) and v.get("dtype") == "video"
        )
        _info["total_videos"] = _n_eps * _remaining_vid
        with open(_info_path, "w", encoding="utf-8") as f:
            json.dump(_info, f, indent=2)
        print(f"[merge] dropped absent video keys from features: {_dropped}", flush=True)

    summary = {
        "total_frames": int(len(merged_df)),
        "total_episodes": int(merged_df["episode_index"].nunique()),
        "total_tasks": len(merged_tasks),
        "frames_by_source": {
            "historical": source_counts[SOURCE_HISTORICAL],
            "hg_new": source_counts[SOURCE_HG_NEW],
        },
        "hg_fraction": (
            round(source_counts[SOURCE_HG_NEW] / max(1, len(merged_df)), 4)
        ),
    }
    summary_path = os.path.join(out_dir, "meta", "merge_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[merge] wrote {summary}", flush=True)
    print(f"[merge] summary: {summary_path}", flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge original insert-only + HG-DAgger shards into one LeRobot-v2 dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--orig", action="append", default=[], metavar="DIR",
        help="historical dataset dir (source=0). Repeatable.",
    )
    parser.add_argument(
        "--hg", action="append", default=[], metavar="DIR",
        help="HG-DAgger collection outdir (source=1); shard_gpu* subdirs are "
        "auto-expanded. Repeatable.",
    )
    parser.add_argument("--out", required=True, help="output merged LeRobot-v2 dir")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.orig and not args.hg:
        raise ValueError("Provide at least one --orig and/or --hg input dir.")
    merge(args.orig, args.hg, args.out)


if __name__ == "__main__":
    main()
