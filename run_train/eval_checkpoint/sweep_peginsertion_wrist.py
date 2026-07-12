#!/usr/bin/env python3
"""Evaluate every wrist PegInsertion checkpoint and plot metrics by step."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_PATH = Path(__file__).resolve().parents[2]
STEP_RE = re.compile(r"global_step_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run wrist-camera PegInsertion evaluation for all global_step_* "
            "actor checkpoints under a checkpoint directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help=(
            "Directory containing global_step_*/actor checkpoints, or one actor "
            "checkpoint directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for per-checkpoint eval logs, CSV, JSON, and plots.",
    )
    parser.add_argument("--venv-dir", default="/opt/kairan/envs/rlinf")
    parser.add_argument("--gpu-ids", default="0-3")
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument(
        "--reward-key",
        choices=("max_reward", "return", "reward"),
        default="max_reward",
        help=(
            "Episode metric used for the reward curve. 'max_reward' is the "
            "maximum step reward within each trajectory; 'return' is trajectory "
            "cumulative reward; 'reward' is trajectory average reward."
        ),
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-step trajectory_metrics.json files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue evaluating later checkpoints if one checkpoint fails.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N checkpoints after sorting by step.",
    )
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        default=None,
        help="Only evaluate specific global steps. Can be passed more than once.",
    )
    parser.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        help="Extra Hydra override passed through to eval_checkpoint.py.",
    )
    return parser.parse_args()


def discover_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if checkpoint_dir.name == "actor":
        match = STEP_RE.match(checkpoint_dir.parent.name)
        if match:
            return [(int(match.group(1)), checkpoint_dir)]

    checkpoints: dict[tuple[int, str], Path] = {}
    for actor_dir in checkpoint_dir.rglob("actor"):
        if not actor_dir.is_dir():
            continue
        match = STEP_RE.match(actor_dir.parent.name)
        if not match:
            continue
        step = int(match.group(1))
        checkpoints[(step, str(actor_dir.resolve()))] = actor_dir.resolve()

    return [(step, path) for (step, _), path in sorted(checkpoints.items())]


def _to_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    return [float(value) for value in values]


def summarize_episode_metrics(
    step: int,
    checkpoint_path: Path,
    trajectory_metrics_path: Path,
    reward_key: str,
) -> dict[str, Any]:
    metrics = json.loads(trajectory_metrics_path.read_text(encoding="utf-8"))
    success_values = _to_float_list(metrics.get("success_once"))
    reward_values = _to_float_list(metrics.get(reward_key))
    max_reward_values = _to_float_list(metrics.get("max_reward"))
    return_values = _to_float_list(metrics.get("return"))
    avg_reward_values = _to_float_list(metrics.get("reward"))

    if not success_values:
        raise RuntimeError(f"No success_once values in {trajectory_metrics_path}")
    if not reward_values:
        raise RuntimeError(f"No {reward_key!r} values in {trajectory_metrics_path}")

    row = {
        "step": step,
        "checkpoint_path": str(checkpoint_path),
        "num_trajectories": int(metrics.get("num_trajectories", len(success_values))),
        "success_rate": mean(success_values),
        "mean_selected_reward": mean(reward_values),
        "max_selected_reward": max(reward_values),
        "reward_key": reward_key,
        "trajectory_metrics_path": str(trajectory_metrics_path),
    }
    if max_reward_values:
        row["mean_max_reward"] = mean(max_reward_values)
        row["max_reward"] = max(max_reward_values)
    if return_values:
        row["max_return"] = max(return_values)
        row["mean_return"] = mean(return_values)
    if avg_reward_values:
        row["max_episode_avg_reward"] = max(avg_reward_values)
        row["mean_episode_avg_reward"] = mean(avg_reward_values)
    return row


def run_eval_for_checkpoint(
    *,
    checkpoint_path: Path,
    step: int,
    log_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trajectory_metrics_path = log_dir / "trajectory_metrics.json"
    if args.resume and trajectory_metrics_path.exists():
        return summarize_episode_metrics(
            step, checkpoint_path, trajectory_metrics_path, args.reward_key
        )

    env = os.environ.copy()
    env.update(
        {
            "VENV_DIR": args.venv_dir,
            "CHECKPOINT_PATH": str(checkpoint_path),
            "LOG_DIR": str(log_dir),
            "GPU_IDS": args.gpu_ids,
            "NUM_EVAL_EPISODES": str(args.num_eval_episodes),
            "NUM_ENVS": str(args.num_envs),
            "MAX_EPISODE_STEPS": str(args.max_episode_steps),
            "SEED": str(args.seed),
            "EVAL_ACTION_SCALE": str(args.action_scale),
            "SAVE_VIDEO": "true" if args.save_video else "false",
            "RAY_TMP_DIR": f"/tmp/ray_eval_wrist_{os.getpid()}_{step}",
        }
    )
    cmd = [
        "bash",
        str(REPO_PATH / "run_train/eval_checkpoint/run_peginsertion_wrist.sh"),
        "--save-episode-metrics",
        *args.hydra_override,
    ]
    print(f"[step {step}] evaluating {checkpoint_path}", flush=True)
    subprocess.run(cmd, cwd=REPO_PATH, env=env, check=True)
    return summarize_episode_metrics(
        step, checkpoint_path, trajectory_metrics_path, args.reward_key
    )


def write_rows(rows: list[dict[str, Any]], output_dir: Path) -> None:
    json_path = output_dir / "wrist_sweep_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / "wrist_sweep_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def plot_rows(rows: list[dict[str, Any]], output_dir: Path) -> None:
    valid_rows = [
        row
        for row in sorted(rows, key=lambda item: int(item["step"]))
        if "success_rate" in row and "mean_selected_reward" in row
    ]
    if not valid_rows:
        print("No successful eval rows to plot.", file=sys.stderr)
        return

    steps = [int(row["step"]) for row in valid_rows]
    success_rates = [float(row["success_rate"]) for row in valid_rows]
    mean_rewards = [float(row["mean_selected_reward"]) for row in valid_rows]
    reward_key = str(valid_rows[0].get("reward_key", "return"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(steps, success_rates, marker="o", linewidth=1.8)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Mean success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, mean_rewards, marker="o", linewidth=1.8, color="tab:orange")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel(f"Mean trajectory {reward_key}")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    combined_path = output_dir / "wrist_sweep_curves.png"
    fig.savefig(combined_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, success_rates, marker="o", linewidth=1.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    sr_path = output_dir / "success_rate_vs_step.png"
    fig.savefig(sr_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, mean_rewards, marker="o", linewidth=1.8, color="tab:orange")
    ax.set_xlabel("Training step")
    ax.set_ylabel(f"Mean trajectory {reward_key}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    reward_path = output_dir / "max_reward_vs_step.png"
    fig.savefig(reward_path, dpi=180)
    plt.close(fig)

    print(f"Wrote {combined_path}")
    print(f"Wrote {sr_path}")
    print(f"Wrote {reward_path}")


def main() -> None:
    args = parse_args()
    if args.num_eval_episodes <= 0 or args.num_envs <= 0:
        raise ValueError("--num-eval-episodes and --num-envs must be positive")
    if args.num_eval_episodes % args.num_envs != 0:
        raise ValueError("--num-eval-episodes must be divisible by --num-envs")

    checkpoints = discover_checkpoints(Path(args.checkpoint_dir))
    if args.step:
        wanted_steps = set(args.step)
        checkpoints = [(step, path) for step, path in checkpoints if step in wanted_steps]
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        raise FileNotFoundError(
            f"No global_step_*/actor checkpoints found under {args.checkpoint_dir}"
        )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_PATH / "logs" / "peginsertion_wrist_ckpt_sweep"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for step, checkpoint_path in checkpoints:
        log_dir = output_dir / f"global_step_{step}"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            row = run_eval_for_checkpoint(
                checkpoint_path=checkpoint_path,
                step=step,
                log_dir=log_dir,
                args=args,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            row = {
                "step": step,
                "checkpoint_path": str(checkpoint_path),
                "error": str(exc),
            }
            print(f"[step {step}] failed: {exc}", file=sys.stderr)
        rows.append(row)
        write_rows(rows, output_dir)
        plot_rows(rows, output_dir)


if __name__ == "__main__":
    main()
