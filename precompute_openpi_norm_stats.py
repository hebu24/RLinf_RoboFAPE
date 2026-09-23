#!/usr/bin/env python3
"""Precompute and cache OpenPI norm stats from LeRobot parquet columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _stats(rows: list[list[float]]) -> dict[str, list[float]]:
    values = np.asarray(rows, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).astype(np.float32).tolist(),
        "std": values.std(axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(np.float32).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="actions")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    paths = sorted((data_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files under {data_dir / 'data'}")

    states: list[list[float]] = []
    actions: list[list[float]] = []
    for path in paths:
        table = pq.read_table(path, columns=[args.state_key, args.action_key])
        states.extend(table[args.state_key].to_pylist())
        actions.extend(table[args.action_key].to_pylist())

    norm_stats = {"norm_stats": {"state": _stats(states), "actions": _stats(actions)}}
    output = data_dir / "meta" / "openpi" / args.config_name / "norm_stats.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(norm_stats, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {output}: episodes={len(paths)}, frames={len(states)}, "
        f"state_dim={len(norm_stats['norm_stats']['state']['mean'])}, "
        f"action_dim={len(norm_stats['norm_stats']['actions']['mean'])}"
    )


if __name__ == "__main__":
    main()
