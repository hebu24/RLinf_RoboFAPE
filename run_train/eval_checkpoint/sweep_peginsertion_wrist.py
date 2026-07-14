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
        "--run-script",
        default=str(REPO_PATH / "run_train/eval_checkpoint/run_peginsertion_wrist.sh"),
        help=(
            "Launcher script to invoke per checkpoint. Point this at "
            "run_peginsertion_wrist_insert_only.sh for insert-only sweeps."
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
            "RAY_TMP_DIR": f"/tmp/ray_eval_wrist_{os.getpid()}_{step}",
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
    ray_tmp_dir = Path(f"/tmp/ray_eval_wrist_sweep_{os.getpid()}")
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
    ray_tmp_dir = Path(f"/tmp/ray_eval_wrist_sweep_{os.getpid()}")
    shutil.rmtree(ray_tmp_dir, ignore_errors=True)


def run_checkpoint_sweep(
    checkpoints: list[tuple[int, Path]],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
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
            log_dir = output_dir / f"global_step_{step}"
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
                write_rows(rows, output_dir)
                plot_rows(rows, output_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [
            executor.submit(gpu_worker, worker_slot, gpu_id)
            for worker_slot, gpu_id in enumerate(gpu_ids)
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

    if args.manage_ray:
        start_shared_ray(args)
    try:
        run_checkpoint_sweep(checkpoints, output_dir, args)
    finally:
        if args.manage_ray:
            stop_shared_ray(args)


if __name__ == "__main__":
    main()
