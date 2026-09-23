#!/usr/bin/env python3
"""Compute OpenPI norm stats from the fields used by peg-insertion SFT."""

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
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    paths = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files under {dataset_dir / 'data'}")

    states: list[list[float]] = []
    actions: list[list[float]] = []
    for path in paths:
        table = pq.read_table(
            path, columns=["observation.state_tcp", "debug.fk_delta_action"]
        )
        states.extend(table["observation.state_tcp"].to_pylist())
        actions.extend(table["debug.fk_delta_action"].to_pylist())

    state_stats = _stats(states)
    action_stats = _stats(actions)
    if len(state_stats["mean"]) != 8 or len(action_stats["mean"]) != 7:
        raise ValueError(
            "PegInsertion SFT contract requires state=8D and debug.fk_delta_action=7D; "
            f"got state={len(state_stats['mean'])}, actions={len(action_stats['mean'])}"
        )

    output = (
        Path(args.output_dir)
        / "physical-intelligence"
        / "maniskill"
        / "norm_stats.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"norm_stats": {"state": state_stats, "actions": action_stats}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {output}: episodes={len(paths)}, frames={len(states)}, "
        f"state_dim=8, action_dim=7"
    )


if __name__ == "__main__":
    main()
