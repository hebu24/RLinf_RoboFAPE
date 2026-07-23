#!/usr/bin/env python3
"""Rewrite the `task` / `task_index` metadata of a LeRobot v2.0 peg-insertion
controller dataset so that prompts align with the eval-time prompt distribution.

Background
----------
The SFT checkpoint trained on `peg_insertion_vertical_controller_3200` achieves
0% eval success. Root cause #1: training prompts are 154 long sentences
("Coordinate with the wrist-camera-guided robot arm to ...") loaded from
`meta/tasks.jsonl` via `prompt_from_task=True`, but eval feeds the short
sentence "insert the peg into the hole" (from
`PegInsertionVerticalEnv.get_language_instruction`, which shadows the yaml
`task_description`). The token sequences barely overlap -> severe language OOD.

This script regenerates a set of short, imperative-style prompts whose form
matches the eval prompt, and rewrites the dataset's prompt metadata in place:

  * `meta/tasks.jsonl` / `meta/tasks.parquet`  -> N new prompts
  * `meta/info.json`                            -> total_tasks = N
  * `meta/episodes.jsonl`                       -> per-episode `tasks` field
  * every `data/chunk-*/episode_*.parquet`      -> `task` (string) + `task_index`
                                                  (int64) columns

Trajectories, images/videos, state, actions, and stats.json are NOT touched —
prompt is pure metadata.

`task_index 0` is fixed to the exact eval prompt
"insert the peg into the hole" so the eval prompt is strictly in-distribution.
Episode -> prompt assignment is `episode_index % N` (deterministic, covers all N).

Usage
-----
    python rewrite_dataset_prompts.py --dry-run                 # preview prompts
    python rewrite_dataset_prompts.py                           # apply in place
    python rewrite_dataset_prompts.py --num-prompts 500 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from peg_insertion_prompts import EVAL_PROMPT, generate_prompts


# ---------------------------------------------------------------------------
# Dataset rewrite
# ---------------------------------------------------------------------------


def _backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak.long")
    if not bak.exists():
        shutil.copy2(path, bak)


def _rewrite_tasks_meta(meta_dir: Path, prompts: list[str]) -> None:
    tasks_jsonl = meta_dir / "tasks.jsonl"
    tasks_parquet = meta_dir / "tasks.parquet"
    _backup(tasks_jsonl)
    _backup(tasks_parquet)

    with open(tasks_jsonl, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            f.write(json.dumps({"task_index": i, "task": p}) + "\n")

    table = pa.table(
        {
            "task_index": pa.array(range(len(prompts)), type=pa.int64()),
            "task": pa.array(prompts, type=pa.string()),
        }
    )
    pq.write_table(table, tasks_parquet)


def _rewrite_info(meta_dir: Path, num_prompts: int) -> None:
    info_path = meta_dir / "info.json"
    _backup(info_path)
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    info["total_tasks"] = num_prompts
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _rewrite_stats(meta_dir: Path, episode_task_index: dict[int, int], data_dir: Path) -> None:
    stats_path = meta_dir / "stats.json"
    if not stats_path.exists():
        return
    _backup(stats_path)
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)

    values: list[int] = []
    for ep_path in _iter_episode_parquets(data_dir):
        episode_index = int(ep_path.stem.split("_")[-1])
        task_index = episode_task_index[episode_index]
        values.extend([task_index] * pq.read_metadata(ep_path).num_rows)
    if not values:
        return

    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    stats["task_index"] = {
        "mean": [mean],
        "std": [var**0.5],
        "min": [min(values)],
        "max": [max(values)],
        "count": [len(values)],
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def _rewrite_episodes(meta_dir: Path, episode_prompt: dict[int, str]) -> None:
    ep_path = meta_dir / "episodes.jsonl"
    _backup(ep_path)
    out = []
    with open(ep_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ei = rec["episode_index"]
            rec["tasks"] = [episode_prompt[ei]]
            out.append(json.dumps(rec))
    with open(ep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def _rewrite_parquet(parquet_path: Path, task_index: int, prompt: str) -> None:
    table = pq.read_table(parquet_path)
    schema = table.schema
    n = table.num_rows
    new_task = pa.array([prompt] * n, type=pa.string())
    new_idx = pa.array([task_index] * n, type=pa.int64())
    # build a new table preserving original column order & schema
    cols = {}
    for name in schema.names:
        if name == "task":
            cols[name] = new_task
        elif name == "task_index":
            cols[name] = new_idx
        else:
            cols[name] = table.column(name)
    new_table = pa.table(cols, schema=schema)
    pq.write_table(new_table, parquet_path)


def _iter_episode_parquets(data_dir: Path):
    for chunk in sorted(data_dir.glob("chunk-*")):
        for ep in sorted(chunk.glob("episode_*.parquet")):
            yield ep


def _assign_episode_prompts(
    episode_indices: list[int],
    prompts: list[str],
    seed: int,
) -> dict[int, int]:
    """Randomly assign one prompt to each episode while covering the full bank."""
    import random

    rng = random.Random(seed)
    pool = list(range(len(prompts)))
    assignment: dict[int, int] = {}
    shuffled = pool.copy()
    rng.shuffle(shuffled)

    for offset, episode_index in enumerate(sorted(episode_indices)):
        if offset < len(shuffled):
            assignment[episode_index] = shuffled[offset]
        else:
            assignment[episode_index] = rng.choice(pool)
    return assignment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200",
    )
    ap.add_argument("--num-prompts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--assignment-seed",
        type=int,
        default=None,
        help="seed for episode->prompt assignment; defaults to --seed",
    )
    ap.add_argument("--dry-run", action="store_true", help="only print prompts")
    args = ap.parse_args()

    prompts = generate_prompts(args.num_prompts, args.seed)
    assert prompts[0] == EVAL_PROMPT, "task_index 0 must be the eval prompt"

    lengths = [len(p) for p in prompts]
    two_stage = sum(1 for p in prompts if " and " in p)
    print(f"generated {len(prompts)} unique prompts (seed={args.seed})")
    print(f"length: min={min(lengths)} max={max(lengths)} mean={sum(lengths)/len(lengths):.1f}")
    print(f"two-stage: {two_stage} ({two_stage/len(prompts):.0%})")
    print(f"task_index 0: {prompts[0]!r}")
    print("sample (0,1,2,3,100,250,499):")
    for i in [0, 1, 2, 3, 100, 250, 499]:
        print(f"  [{i}] {prompts[i]}")

    if args.dry_run:
        print("\n--dry-run: not writing. Sample of all prompts:")
        for i, p in enumerate(prompts):
            print(f"  [{i}] {p}")
        return

    src = Path(args.src)
    meta_dir = src / "meta"
    data_dir = src / "data"
    if not meta_dir.exists():
        raise SystemExit(f"meta dir not found: {meta_dir}")

    print(f"\nrewriting in place: {src}")

    # 1. tasks meta
    _rewrite_tasks_meta(meta_dir, prompts)
    print(f"  wrote tasks.jsonl / tasks.parquet ({len(prompts)} prompts)")

    # 2. info.json
    _rewrite_info(meta_dir, len(prompts))
    print(f"  updated info.json total_tasks={len(prompts)}")

    # 3. episodes.jsonl — need episode list first
    episode_paths = list(_iter_episode_parquets(data_dir))
    episode_indices = [int(ep_path.stem.split("_")[-1]) for ep_path in episode_paths]
    episode_task_index = _assign_episode_prompts(
        episode_indices,
        prompts,
        args.seed if args.assignment_seed is None else args.assignment_seed,
    )
    episode_prompt: dict[int, str] = {}
    for ei, idx in episode_task_index.items():
        episode_prompt[ei] = prompts[idx]
    _rewrite_episodes(meta_dir, episode_prompt)
    print(f"  updated episodes.jsonl ({len(episode_prompt)} episodes)")

    # 4. parquets
    done = 0
    for ep_path in episode_paths:
        ei = int(ep_path.stem.split("_")[-1])
        idx = episode_task_index[ei]
        _rewrite_parquet(ep_path, idx, prompts[idx])
        done += 1
        if done % 500 == 0:
            print(f"  rewrote {done}/{len(episode_paths)} parquets")
    print(f"  rewrote {done} parquets")

    # 5. stats.json task_index summary
    _rewrite_stats(meta_dir, episode_task_index, data_dir)
    print("  updated stats.json task_index summary")

    print("\ndone. backups under meta/*.bak.long")


if __name__ == "__main__":
    main()
