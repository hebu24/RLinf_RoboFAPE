#!/usr/bin/env python3
# ruff: noqa: I001
"""Evaluate every wrist PegInsertion checkpoint and plot metrics by step."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import queue
import re
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_PATH = Path(__file__).resolve().parents[2]
# Matches both SFT checkpoints (global_step_<N>) and RL checkpoints
# (global_step_<N>_trainenvstep_<M>). Group 1 = PPO/SFT step, group 2 = the
# optional RL trainenvstep. The `$` anchor keeps us from matching substrings.
STEP_RE = re.compile(r"global_step_(\d+)(?:_trainenvstep_(\d+))?$")


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
        default=None,
        help=(
            "Directory containing global_step_*/actor checkpoints, or one actor "
            "checkpoint directory. Required unless --plot-only is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for per-checkpoint eval logs, CSV, JSON, and plots.",
    )
    parser.add_argument("--venv-dir", default="/data/yingxi/kairan/envs/rlinf")
    parser.add_argument("--gpu-ids", default="0-3")
    parser.add_argument(
        "--gpus-per-ckpt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Give EVERY checkpoint the full --gpu-ids set (e.g. 4,5 -> 2 env "
            "workers, ranks 0+1, env seeds 0+1) so checkpoints run SEQUENTIALLY "
            "and each reproduces training's multi-env-worker seed topology "
            "(training used 2 env workers, 4 sub-envs each = 8 seeds). Default "
            "false: one GPU per parallel checkpoint slot (1 env worker each, "
            "rank-0 seeds only). Use --num-envs=8 --num-eval-episodes as a "
            "multiple of 8 with this for exact training-seed coverage."
        ),
    )
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=None,
        help=(
            "CPUs exposed by the shared Ray head. Defaults to max(8, 4 per "
            "evaluation GPU) to avoid an oversized idle worker pool."
        ),
    )
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
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save evaluation videos for every checkpoint.",
    )
    parser.add_argument(
        "--manage-ray",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start one shared Ray head for the whole sweep.",
    )
    parser.add_argument(
        "--ray-port",
        type=int,
        default=6380,
        help=(
            "GCS port for the sweep's detached Ray head. Must differ from the SFT "
            "cluster's port (SFT uses 6379) so the two clusters never collide."
        ),
    )
    parser.add_argument(
        "--ray-object-store-memory",
        type=int,
        default=50_000_000_000,
        help=(
            "Object store bytes for the sweep head. Default 50G; /dev/shm is 1008G "
            "so this is comfortable hygiene (eval needs far less than the 200G default)."
        ),
    )
    parser.add_argument(
        "--ray-tmp-dir",
        default="/data/yingxi/ray_es",
        help=(
            "Temp dir for the sweep's Ray head. Must be on /data (NOT /tmp): "
            "xulab's root-backed /tmp fills to 100% and Ray's object store + "
            "logs would crash it with Errno 28. Keep the BASE path SHORT: Ray "
            "appends sweep_<pid>/session_<ts>_<pid>/sockets/plasma_store, and "
            "the AF_UNIX socket path must stay under 107 bytes (a long base "
            "like /data/yingxi/ray_tmp_eval_sweep overflows it)."
        ),
    )
    parser.add_argument(
        "--ray-dashboard-port",
        type=int,
        default=8266,
        help=(
            "Dashboard server port for the sweep head. Must differ from any "
            "concurrent cluster's dashboard (the SFT head uses 8265). Enabling the "
            "dashboard (instead of --include-dashboard=false) lets ray state API "
            "calls like list_actors succeed instead of raising "
            "ConnectionError: Could not read 'dashboard' from GCS."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-step trajectory_metrics.json files.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Do not evaluate checkpoints or start Ray. Rebuild metrics and plots "
            "from every evaluation_summary.json under --output-dir, including "
            "results whose original checkpoints have since been deleted."
        ),
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
    parser.add_argument(
        "--norm-stats-source",
        default=None,
        help=(
            "Source actor checkpoint dir whose physical-intelligence/ assets "
            "hold norm_stats.json. Before evaluating each checkpoint, if it "
            "lacks those assets, copy them from here. RL checkpoints do not save "
            "norm_stats.json (the assets are fixed input constants, identical to "
            "the SFT checkpoint the policy was initialized from), so point this "
            "at that SFT actor dir. Leave unset for an SFT sweep (SFT ckpts "
            "already contain the assets)."
        ),
    )
    parser.add_argument(
        "--run-script",
        default=str(REPO_PATH / "run_train/eval_checkpoint/run_peginsertion_wrist_insert_only.sh"),
        help=(
            "Launcher script to invoke per checkpoint (insert-only wrist eval)."
        ),
    )
    return parser.parse_args()


def parse_gpu_ids(raw_value: str) -> list[str]:
    gpu_ids: list[str] = []
    for piece in raw_value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start_raw, end_raw = piece.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid GPU range: {piece}")
            gpu_ids.extend(str(gpu_id) for gpu_id in range(start, end + 1))
        else:
            gpu_ids.append(piece)
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id")
    return gpu_ids


def discover_checkpoints(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if checkpoint_dir.name == "actor":
        match = STEP_RE.match(checkpoint_dir.parent.name)
        if match:
            return [(int(match.group(1)), checkpoint_dir)]

    # Key by (step, trainenvstep, path) so RL checkpoints
    # (global_step_<N>_trainenvstep_<M>) sort by PPO step first, then
    # trainenvstep — never out of order relative to one another.
    checkpoints: dict[tuple[int, int, str], Path] = {}
    for actor_dir in checkpoint_dir.rglob("actor"):
        if not actor_dir.is_dir():
            continue
        match = STEP_RE.match(actor_dir.parent.name)
        if not match:
            continue
        step = int(match.group(1))
        trainenvstep = int(match.group(2)) if match.group(2) is not None else 0
        checkpoints[(step, trainenvstep, str(actor_dir.resolve()))] = actor_dir.resolve()

    return [(step, path) for (step, _, _), path in sorted(checkpoints.items())]


def _checkpoint_name(checkpoint_path: Path) -> str:
    """The global_step_<N>[_trainenvstep_<M>] directory name holding this actor."""
    return checkpoint_path.parent.name


def _trainenvstep_of(checkpoint_path: Path) -> int | None:
    match = STEP_RE.match(checkpoint_path.parent.name)
    if match and match.group(2) is not None:
        return int(match.group(2))
    return None


def _ensure_norm_stats(checkpoint_path: Path, source_path: Path) -> None:
    """Copy physical-intelligence/ norm_stats assets into a checkpoint if missing.

    RL checkpoints do not save norm_stats.json (the assets are fixed input
    constants, identical to the SFT checkpoint the policy was initialized
    from). The OpenPI loader reads them from
    ``<checkpoint_path>/<asset_id>/norm_stats.json``, so they must live inside
    the actor dir. Idempotent: skip if already present.
    """
    dest_assets = checkpoint_path / "physical-intelligence"
    if dest_assets.exists():
        return
    src_assets = source_path.expanduser().resolve() / "physical-intelligence"
    if not src_assets.exists():
        raise FileNotFoundError(
            f"--norm-stats-source {source_path} has no physical-intelligence/ "
            f"assets dir; cannot copy norm_stats into {checkpoint_path}"
        )
    print(f"[norm_stats] copying {src_assets} -> {dest_assets}", flush=True)
    shutil.copytree(src_assets, dest_assets)


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
        "trainenvstep": _trainenvstep_of(checkpoint_path),
        "checkpoint_name": _checkpoint_name(checkpoint_path),
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


def summarize_evaluation_summary(
    summary_path: Path, reward_key: str
) -> dict[str, Any]:
    """Convert a persisted evaluation summary into one sweep-plot row."""
    checkpoint_name = summary_path.parent.name
    match = STEP_RE.match(checkpoint_name)
    if not match:
        raise RuntimeError(
            f"Evaluation summary is not under a global_step_* directory: {summary_path}"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"No metrics object in {summary_path}")
    if "success_once" not in metrics:
        raise RuntimeError(f"No success_once metric in {summary_path}")
    if reward_key not in metrics:
        raise RuntimeError(f"No {reward_key!r} metric in {summary_path}")

    step = int(match.group(1))
    selected_reward = float(metrics[reward_key])
    row = {
        "step": step,
        "trainenvstep": int(match.group(2)) if match.group(2) is not None else None,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(summary.get("checkpoint_path", "")),
        "num_trajectories": int(metrics.get("num_trajectories", 0)),
        "success_rate": float(metrics["success_once"]),
        "mean_selected_reward": selected_reward,
        "max_selected_reward": selected_reward,
        "reward_key": reward_key,
        "evaluation_summary_path": str(summary_path),
    }
    for metric_name, row_mean_key, row_max_key in (
        ("max_reward", "mean_max_reward", "max_reward"),
        ("return", "mean_return", "max_return"),
        ("reward", "mean_episode_avg_reward", "max_episode_avg_reward"),
    ):
        if metric_name in metrics:
            value = float(metrics[metric_name])
            row[row_mean_key] = value
            row[row_max_key] = value
    return row


def rebuild_plots_from_evaluation_summaries(
    output_dir: Path, reward_key: str
) -> list[dict[str, Any]]:
    """Rebuild sweep artifacts solely from persisted evaluation summaries."""
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for summary_path in sorted(output_dir.rglob("evaluation_summary.json")):
        try:
            rows.append(summarize_evaluation_summary(summary_path, reward_key))
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{summary_path}: {exc}")

    if not rows:
        raise FileNotFoundError(
            f"No usable evaluation_summary.json files found under {output_dir}"
        )
    rows.sort(
        key=lambda row: (
            int(row["step"]),
            int(row["trainenvstep"]) if row["trainenvstep"] is not None else 0,
            str(row["checkpoint_name"]),
        )
    )
    write_rows(rows, output_dir)
    plot_rows(rows, output_dir)
    for failure in failures:
        print(f"Skipping invalid evaluation summary: {failure}", file=sys.stderr)
    print(f"Rebuilt plots from {len(rows)} evaluation summaries.")
    return rows


def run_eval_for_checkpoint(
    *,
    checkpoint_path: Path,
    step: int,
    gpu_id: str,
    worker_slot: int,
    log_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trajectory_metrics_path = log_dir / "trajectory_metrics.json"
    if args.resume and trajectory_metrics_path.exists():
        row = summarize_episode_metrics(
            step, checkpoint_path, trajectory_metrics_path, args.reward_key
        )
        row["gpu_id"] = gpu_id
        row["log_dir"] = str(log_dir)
        return row

    # RL checkpoints lack norm_stats.json; copy the fixed input-normalization
    # assets from the SFT checkpoint the policy was initialized from (they are
    # identical constants, not learned during RL). Idempotent; no-op for SFT
    # sweeps that leave --norm-stats-source unset.
    if args.norm_stats_source:
        _ensure_norm_stats(checkpoint_path, Path(args.norm_stats_source))

    env = os.environ.copy()
    env.update(
        {
            "RAY_ADDRESS": f"127.0.0.1:{args.ray_port}",
            "VENV_DIR": args.venv_dir,
            "CHECKPOINT_PATH": str(checkpoint_path),
            "LOG_DIR": str(log_dir),
            "GPU_IDS": gpu_id,
            "NUM_EVAL_EPISODES": str(args.num_eval_episodes),
            "NUM_ENVS": str(args.num_envs),
            "MAX_EPISODE_STEPS": str(args.max_episode_steps),
            "SEED": str(args.seed),
            "EVAL_ACTION_SCALE": str(args.action_scale),
            "SAVE_VIDEO": "true" if args.save_video else "false",
            "MANAGE_RAY": "false",
            "RAY_TMP_DIR": f"{args.ray_tmp_dir}_{os.getpid()}_{step}",
        }
    )
    unique_suffix = f"{os.getpid()}_{worker_slot}_{step}"
    run_script = args.run_script
    if not os.path.isabs(run_script):
        run_script = str(REPO_PATH / run_script)
    cmd = [
        "bash",
        run_script,
        "--save-episode-metrics",
        f"env.group_name=EnvGroupEval{unique_suffix}",
        f"rollout.group_name=RolloutGroupEval{unique_suffix}",
        *args.hydra_override,
    ]
    print(f"[gpu {gpu_id} step {step}] evaluating {checkpoint_path}", flush=True)
    subprocess.run(cmd, cwd=REPO_PATH, env=env, check=True)
    row = summarize_episode_metrics(
        step, checkpoint_path, trajectory_metrics_path, args.reward_key
    )
    row["gpu_id"] = gpu_id
    row["log_dir"] = str(log_dir)
    return row


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


def annotate_curve_points(ax: Any, steps: list[int], values: list[float]) -> None:
    """Label every curve marker with its plotted metric value."""
    for index, (step, value) in enumerate(zip(steps, values, strict=True)):
        # Alternate offsets so adjacent points with similar values remain legible.
        vertical_offset = 7 if index % 2 == 0 else -13
        ax.annotate(
            f"{value:.3f}",
            (step, value),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if vertical_offset > 0 else "top",
            fontsize=7,
        )


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
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].grid(True, alpha=0.3)
    annotate_curve_points(axes[0], steps, success_rates)

    axes[1].plot(steps, mean_rewards, marker="o", linewidth=1.8, color="tab:orange")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel(f"Mean trajectory {reward_key}")
    axes[1].grid(True, alpha=0.3)
    annotate_curve_points(axes[1], steps, mean_rewards)

    fig.tight_layout()
    combined_path = output_dir / "wrist_sweep_curves.png"
    fig.savefig(combined_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, success_rates, marker="o", linewidth=1.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean success rate")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    annotate_curve_points(ax, steps, success_rates)
    fig.tight_layout()
    sr_path = output_dir / "success_rate_vs_step.png"
    fig.savefig(sr_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, mean_rewards, marker="o", linewidth=1.8, color="tab:orange")
    ax.set_xlabel("Training step")
    ax.set_ylabel(f"Mean trajectory {reward_key}")
    ax.grid(True, alpha=0.3)
    annotate_curve_points(ax, steps, mean_rewards)
    fig.tight_layout()
    reward_path = output_dir / "max_reward_vs_step.png"
    fig.savefig(reward_path, dpi=180)
    plt.close(fig)

    print(f"Wrote {combined_path}")
    print(f"Wrote {sr_path}")
    print(f"Wrote {reward_path}")


def raise_file_descriptor_limit(minimum: int = 65536) -> None:
    """Raise RLIMIT_NOFILE before Ray inherits the sweep process limits."""
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_limit = min(max(soft_limit, minimum), hard_limit)
    if target_limit > soft_limit:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard_limit))
    if target_limit < minimum:
        raise RuntimeError(
            f"Ray needs at least {minimum} open files for a concurrent sweep, but "
            f"this shell hard limit is {hard_limit}. Raise it before launching."
        )
    print(f"Ray file descriptor limit: {target_limit}", flush=True)


def _scoped_ray_kill(ray_port: int) -> None:
    """Kill ONLY the Ray processes bound to ray_port (gcs_server + raylet + dashboard).

    `ray stop` cannot target one cluster (it kills ALL ray on the host, incl. an SFT
    job on 6379), so we scope by the GCS port in each process' cmdline:
    gcs_server carries `--gcs_server_port=<port>`; raylet/dashboard carry
    `--gcs-address=<ip>:<port>` (verified in ray/_private/services.py). A different
    port can never be matched, so a concurrent SFT cluster is never touched.
    """
    patterns = [
        f"gcs_server.*--gcs_server_port={ray_port}",
        f"raylet.*--gcs-address=[^ ]*:{ray_port}",
        f"dashboard.*--gcs-address=[^ ]*:{ray_port}",
    ]
    for pattern in patterns:
        subprocess.run(
            ["pkill", "-9", "-f", pattern],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    # Let the raylet release GPU resources before a new head re-registers them.
    time.sleep(2)


def start_shared_ray(args: argparse.Namespace) -> None:
    raise_file_descriptor_limit()
    ray_bin = Path(args.venv_dir).expanduser().resolve() / "bin" / "ray"
    if not ray_bin.exists():
        raise FileNotFoundError(f"Ray binary does not exist: {ray_bin}")
    ray_port = int(args.ray_port)
    ray_tmp_dir = Path(args.ray_tmp_dir) / f"sweep_{os.getpid()}"
    ray_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Scoped stale cleanup: only the eval port, never SFT's 6379. Never bare `ray stop`.
    _scoped_ray_kill(ray_port)

    # Pop RAY_ADDRESS first so `ray start --head` does not try to attach to a
    # pre-existing cluster; set it back after so the driver, per-checkpoint
    # subprocesses, and worker actors all attach to THIS head.
    os.environ.pop("RAY_ADDRESS", None)
    gpu_count = len(parse_gpu_ids(args.gpu_ids))
    ray_num_cpus = args.ray_num_cpus or max(8, 4 * gpu_count)
    if ray_num_cpus <= 0:
        raise ValueError("--ray-num-cpus must be positive")
    subprocess.run(
        [
            str(ray_bin),
            "start",
            "--head",
            f"--port={ray_port}",
            f"--temp-dir={ray_tmp_dir}",
            f"--num-cpus={ray_num_cpus}",
            f"--dashboard-port={int(args.ray_dashboard_port)}",
            f"--object-store-memory={int(args.ray_object_store_memory)}",
        ],
        check=True,
    )
    # Pin driver + subprocesses + workers to this head. Workers honor RAY_ADDRESS via
    # ray.init(address="auto") (worker.py) and Manager.get_runtime_env_vars()
    # (manager.py) which copies RAY_ADDRESS into the worker runtime_env.
    os.environ["RAY_ADDRESS"] = f"127.0.0.1:{ray_port}"


def stop_shared_ray(args: argparse.Namespace) -> None:
    # Scoped: kill ONLY the eval head on this port. Never a bare `ray stop`
    # (that would kill an SFT cluster on 6379).
    _scoped_ray_kill(int(args.ray_port))
    # Clean this sweep's temp dir only.
    ray_tmp_dir = Path(args.ray_tmp_dir) / f"sweep_{os.getpid()}"
    shutil.rmtree(ray_tmp_dir, ignore_errors=True)


def run_checkpoint_sweep(
    checkpoints: list[tuple[int, Path]],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    # When --gpus-per-ckpt, every checkpoint gets the FULL gpu set (one worker,
    # checkpoints run sequentially) so a 2-GPU set yields 2 env workers (ranks
    # 0+1, env seeds 0+1) = training's multi-worker seed topology. Otherwise
    # (default) one GPU per parallel worker slot (1 env worker each).
    if args.gpus_per_ckpt:
        worker_gpu_ids = [",".join(gpu_ids)]
    else:
        worker_gpu_ids = gpu_ids
    rows: list[dict[str, Any]] = []
    completed: dict[int, dict[str, Any]] = {}
    rows_lock = threading.Lock()
    task_queue: queue.Queue[tuple[int, Path]] = queue.Queue()
    for item in checkpoints:
        task_queue.put(item)

    def gpu_worker(worker_slot: int, gpu_id: str) -> None:
        nonlocal rows
        while True:
            try:
                step, checkpoint_path = task_queue.get_nowait()
            except queue.Empty:
                return
            # Use the checkpoint's own dir name (global_step_<N> for SFT,
            # global_step_<N>_trainenvstep_<M> for RL) so RL checkpoints are
            # unambiguous and never collide; SFT behavior is unchanged.
            log_dir = output_dir / _checkpoint_name(checkpoint_path)
            log_dir.mkdir(parents=True, exist_ok=True)
            try:
                row = run_eval_for_checkpoint(
                    checkpoint_path=checkpoint_path,
                    step=step,
                    gpu_id=gpu_id,
                    worker_slot=worker_slot,
                    log_dir=log_dir,
                    args=args,
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                row = {
                    "step": step,
                    "checkpoint_path": str(checkpoint_path),
                    "gpu_id": gpu_id,
                    "error": str(exc),
                }
                print(f"[gpu {gpu_id} step {step}] failed: {exc}", file=sys.stderr)
            finally:
                task_queue.task_done()

            with rows_lock:
                completed[step] = row
                rows = [completed[item_step] for item_step in sorted(completed)]
                # Evaluation summaries persist independently of checkpoint
                # retention. Rebuild from the whole eval directory at every
                # refresh so intermediate plots do not drop historical points.
                rebuild_plots_from_evaluation_summaries(output_dir, args.reward_key)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_gpu_ids)) as executor:
        futures = [
            executor.submit(gpu_worker, worker_slot, gpu_id)
            for worker_slot, gpu_id in enumerate(worker_gpu_ids)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    return rows


def main() -> None:
    args = parse_args()
    if args.num_eval_episodes <= 0 or args.num_envs <= 0:
        raise ValueError("--num-eval-episodes and --num-envs must be positive")
    if args.num_eval_episodes % args.num_envs != 0:
        raise ValueError("--num-eval-episodes must be divisible by --num-envs")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_PATH / "logs" / "peginsertion_wrist_ckpt_sweep"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        rebuild_plots_from_evaluation_summaries(output_dir, args.reward_key)
        return

    if args.checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required unless --plot-only is used")
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

    if args.manage_ray:
        start_shared_ray(args)
    try:
        run_checkpoint_sweep(checkpoints, output_dir, args)
        # Include retained historical evaluations as well as checkpoints from
        # this invocation; the checkpoint directory may have pruned old RL ckpts.
        rebuild_plots_from_evaluation_summaries(output_dir, args.reward_key)
    finally:
        if args.manage_ray:
            stop_shared_ray(args)


if __name__ == "__main__":
    main()
