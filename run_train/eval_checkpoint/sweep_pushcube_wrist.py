#!/usr/bin/env python3
# ruff: noqa: I001
"""Sweep PushCube-v1 wrist checkpoints across multiple seeds; plot per-seed (transparent)
+ mean (opaque) success rate vs training step. Supports --watch to eval each new checkpoint
as it appears (serial, single GPU, no overlap)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
EXPECTED_CKPT_SUBDIRS = ("dcp_checkpoint", "model_state_dict", "physical-intelligence")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-seed PushCube-v1 wrist checkpoint sweep with sr-vs-step plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint-dir", required=True, help="Directory containing global_step_*/actor checkpoints.")
    p.add_argument("--output-dir", default=None, help="Dir for per-(step,seed) logs, CSV, JSON, plots.")
    p.add_argument("--venv-dir", default="/opt/kairan/envs/rlinf")
    p.add_argument("--run-script", default=str(REPO_PATH / "run_train/eval_checkpoint/run_pushcube_wrist.sh"))
    p.add_argument("--gpu-ids", default="7")
    p.add_argument("--seeds", default="0-7", help="Seed range or comma list, e.g. 0-7 or 0,2,4.")
    p.add_argument("--num-eval-episodes", type=int, default=50)
    p.add_argument("--num-envs", type=int, default=50)
    p.add_argument("--max-episode-steps", type=int, default=180)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--manage-ray", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ray-port", type=int, default=6380)
    p.add_argument("--ray-object-store-memory", type=int, default=50_000_000_000)
    p.add_argument("--ray-dashboard-port", type=int, default=8266)
    p.add_argument("--resume", action="store_true", help="Reuse existing per-(step,seed) trajectory_metrics.json.")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--watch", action="store_true", help="Poll for new checkpoints and eval each as it completes.")
    p.add_argument("--watch-poll-interval", type=int, default=30)
    p.add_argument("--ckpt-stable-seconds", type=int, default=30, help="Checkpoint mtime stability window before evaluating.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--step", type=int, action="append", default=None, help="Only evaluate specific global steps.")
    p.add_argument("--hydra-override", action="append", default=[])
    return p.parse_args()


def parse_seeds(raw: str) -> list[int]:
    out: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(piece))
    return sorted(set(out))


def parse_gpu_ids(raw: str) -> list[str]:
    ids: list[str] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = piece.split("-", 1)
            ids.extend(str(i) for i in range(int(a), int(b) + 1))
        else:
            ids.append(piece)
    if not ids:
        raise ValueError("--gpu-ids must contain at least one GPU id")
    return ids


def discover_checkpoints(checkpoint_dir: Path, only_steps: set[int] | None = None) -> list[tuple[int, Path]]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    found: dict[tuple[int, str], Path] = {}
    for actor_dir in checkpoint_dir.rglob("actor"):
        if not actor_dir.is_dir():
            continue
        m = STEP_RE.match(actor_dir.parent.name)
        if not m:
            continue
        step = int(m.group(1))
        if only_steps and step not in only_steps:
            continue
        found[(step, str(actor_dir.resolve()))] = actor_dir.resolve()
    return [(step, path) for (step, _), path in sorted(found.items())]


def ckpt_is_complete(actor_dir: Path) -> bool:
    """A checkpoint is complete when all expected subdirs are present."""
    return all((actor_dir / sub).exists() for sub in EXPECTED_CKPT_SUBDIRS)


def ckpt_mtime(actor_dir: Path) -> float:
    """Max mtime across the actor dir tree (write-in-progress keeps changing)."""
    latest = 0.0
    for root, _dirs, files in os.walk(actor_dir):
        for name in files:
            try:
                latest = max(latest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                pass
        try:
            latest = max(latest, os.path.getmtime(root))
        except OSError:
            pass
    return latest


def _to_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    return [float(v) for v in values]


def summarize_step_seed(step: int, seed: int, checkpoint_path: Path, traj_path: Path) -> dict[str, Any]:
    metrics = json.loads(traj_path.read_text(encoding="utf-8"))
    success_values = _to_float_list(metrics.get("success_once"))
    reward_values = _to_float_list(metrics.get("max_reward"))
    if not success_values:
        raise RuntimeError(f"No success_once values in {traj_path}")
    row = {
        "step": step,
        "seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "num_trajectories": int(metrics.get("num_trajectories", len(success_values))),
        "success_rate": mean(success_values),
        "mean_max_reward": mean(reward_values) if reward_values else None,
        "max_max_reward": max(reward_values) if reward_values else None,
        "trajectory_metrics_path": str(traj_path),
    }
    return row


def run_eval_for_step_seed(*, checkpoint_path: Path, step: int, seed: int, log_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    traj_path = log_dir / "trajectory_metrics.json"
    if args.resume and traj_path.exists():
        return summarize_step_seed(step, seed, checkpoint_path, traj_path)

    env = os.environ.copy()
    env.update(
        {
            "RAY_ADDRESS": f"127.0.0.1:{args.ray_port}",
            "VENV_DIR": args.venv_dir,
            "CHECKPOINT_PATH": str(checkpoint_path),
            "LOG_DIR": str(log_dir),
            "GPU_IDS": args.gpu_ids,
            "NUM_EVAL_EPISODES": str(args.num_eval_episodes),
            "NUM_ENVS": str(args.num_envs),
            "MAX_EPISODE_STEPS": str(args.max_episode_steps),
            "SEED": str(seed),
            "EVAL_ACTION_SCALE": str(args.action_scale),
            "SAVE_VIDEO": "true" if args.save_video else "false",
            "MANAGE_RAY": "false",  # sweep owns the shared Ray head
            "RAY_TMP_DIR": f"/tmp/ray_eval_pushcube_{os.getpid()}_{step}_{seed}",
        }
    )
    run_script = args.run_script
    if not os.path.isabs(run_script):
        run_script = str(REPO_PATH / run_script)
    cmd = ["bash", run_script, "--save-episode-metrics", *args.hydra_override]
    print(f"[step {step} seed {seed}] evaluating {checkpoint_path}", flush=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "sweep_call.log").open("w") as f:
        proc = subprocess.run(cmd, cwd=REPO_PATH, env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"eval failed (exit {proc.returncode}) for step {step} seed {seed}; see {log_dir}/sweep_call.log")
    return summarize_step_seed(step, seed, checkpoint_path, traj_path)


def write_rows(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    json_path = output_dir / "pushcube_sweep_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "pushcube_sweep_metrics.csv"
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def plot_rows(rows: list[dict[str, Any]], output_dir: Path) -> None:
    by_step: dict[int, dict[int, float]] = {}
    for r in rows:
        if "success_rate" not in r:
            continue
        by_step.setdefault(int(r["step"]), {})[int(r["seed"])] = float(r["success_rate"])
    if not by_step:
        return
    steps = sorted(by_step)
    seeds = sorted({s for sp in by_step.values() for s in sp})
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for si, sd in enumerate(seeds):
        xs, ys = [], []
        for st in steps:
            if sd in by_step[st]:
                xs.append(st)
                ys.append(by_step[st][sd])
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.0, alpha=0.25, color=cmap(si % 10), label=f"seed {sd}")
    xm, ym = [], []
    for st in steps:
        vals = list(by_step[st].values())
        if vals:
            xm.append(st)
            ym.append(mean(vals))
    ax.plot(xm, ym, marker="o", markersize=6, linewidth=2.6, alpha=1.0, color="black", label="mean SR", zorder=10)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out = output_dir / "sr_vs_step_multiseed.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote {out}")


def raise_file_descriptor_limit(minimum: int = 65536) -> None:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, minimum), hard)
    if target > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    if target < minimum:
        raise RuntimeError(f"Ray needs >= {minimum} open files; hard limit is {hard}.")
    print(f"Ray file descriptor limit: {target}", flush=True)


def _scoped_ray_kill(ray_port: int) -> None:
    for pattern in (
        f"gcs_server.*--gcs_server_port={ray_port}",
        f"raylet.*--gcs-address=[^ ]*:{ray_port}",
        f"dashboard.*--gcs-address=[^ ]*:{ray_port}",
    ):
        subprocess.run(["pkill", "-9", "-f", pattern], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)


def start_shared_ray(args: argparse.Namespace) -> None:
    raise_file_descriptor_limit()
    ray_bin = Path(args.venv_dir).expanduser().resolve() / "bin" / "ray"
    if not ray_bin.exists():
        raise FileNotFoundError(f"Ray binary not found: {ray_bin}")
    ray_tmp = Path(f"/opt/yingxi/ray_tmp_eval_sweep_{os.getpid()}")
    ray_tmp.mkdir(parents=True, exist_ok=True)
    _scoped_ray_kill(args.ray_port)
    os.environ.pop("RAY_ADDRESS", None)
    subprocess.run(
        [
            str(ray_bin), "start", "--head",
            f"--port={args.ray_port}",
            f"--temp-dir={ray_tmp}",
            f"--num-cpus=48",
            f"--dashboard-port={int(args.ray_dashboard_port)}",
            f"--object-store-memory={int(args.ray_object_store_memory)}",
        ],
        check=True,
    )
    os.environ["RAY_ADDRESS"] = f"127.0.0.1:{args.ray_port}"


def stop_shared_ray(args: argparse.Namespace) -> None:
    _scoped_ray_kill(int(args.ray_port))
    shutil.rmtree(f"/opt/yingxi/ray_tmp_eval_sweep_{os.getpid()}", ignore_errors=True)


def eval_checkpoint_all_seeds(step: int, checkpoint_path: Path, output_dir: Path, args: argparse.Namespace, rows: list, lock) -> None:
    """Evaluate all seeds for one checkpoint (serial, single GPU). Append rows + replot."""
    for seed in parse_seeds(args.seeds):
        log_dir = output_dir / f"global_step_{step}" / f"seed_{seed}"
        try:
            row = run_eval_for_step_seed(
                checkpoint_path=checkpoint_path, step=step, seed=seed, log_dir=log_dir, args=args
            )
        except Exception as exc:
            print(f"[step {step} seed {seed}] FAILED: {exc}", file=sys.stderr, flush=True)
            if not args.continue_on_error and not args.watch:
                raise
            row = {"step": step, "seed": seed, "checkpoint_path": str(checkpoint_path), "error": str(exc)}
        with lock:
            rows.append(row)
            write_rows(rows, output_dir)
            plot_rows(rows, output_dir)
            # emit a line the Monitor can catch
            if "success_rate" in row:
                print(f"[done] step {step} seed {seed} sr={row['success_rate']:.4f}", flush=True)


def discover_completed_checkpoints(checkpoint_dir: Path, seen: set[int]) -> list[tuple[int, Path]]:
    """Find completed checkpoints not yet seen."""
    new = []
    for step, path in discover_checkpoints(checkpoint_dir):
        if step in seen:
            continue
        if not ckpt_is_complete(path):
            continue
        new.append((step, path))
    return new


def watch_loop(args: argparse.Namespace, output_dir: Path, rows: list, lock) -> None:
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    # A step is "seen" only when ALL seeds have completed trajectory_metrics;
    # otherwise the watch loop must re-enter it (eval_checkpoint_all_seeds + --resume
    # will skip the already-done seeds and only redo the missing ones).
    all_seeds = parse_seeds(args.seeds)
    from collections import defaultdict
    seed_count: dict[int, int] = defaultdict(int)
    for r in rows:
        if "success_rate" in r:
            seed_count[int(r["step"])] += 1
    seen: set[int] = set(step for step, cnt in seed_count.items() if cnt >= len(all_seeds))
    print(f"[watch] start; checkpoint_dir={checkpoint_dir}; seen={sorted(seen)} (full-seed steps); "
          f"partial={[s for s in sorted(seed_count) if s not in seen]}", flush=True)
    while True:
        # check shared ray alive; restart if died
        try:
            subprocess.run([str(Path(args.venv_dir) / "bin" / "ray"), "status"],
                           env={**os.environ, "RAY_ADDRESS": f"127.0.0.1:{args.ray_port}"},
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        except Exception:
            print("[watch] eval Ray head down; restarting...", flush=True)
            start_shared_ray(args)
        # find newly-completed checkpoints (stable mtime)
        candidates = []
        for step, path in discover_checkpoints(checkpoint_dir):
            if step in seen:
                continue
            if not ckpt_is_complete(path):
                continue
            mt = ckpt_mtime(path)
            if time.time() - mt < args.ckpt_stable_seconds:
                continue  # still being written
            candidates.append((step, path))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            step, path = candidates[0]
            print(f"[watch] new completed checkpoint: global_step_{step}", flush=True)
            eval_checkpoint_all_seeds(step, path, output_dir, args, rows, lock)
            seen.add(step)
            with lock:
                completed = [r for r in rows if "success_rate" in r]
                if completed:
                    by_step: dict[int, list[float]] = {}
                    for r in completed:
                        by_step.setdefault(int(r["step"]), []).append(float(r["success_rate"]))
                    avgs = {st: mean(v) for st, v in by_step.items()}
                    print(f"[watch] step {step} done; avg_sr per step: " +
                          ", ".join(f"{s}={a:.3f}" for s, a in sorted(avgs.items())), flush=True)
        else:
            time.sleep(args.watch_poll_interval)


def main() -> None:
    args = parse_args()
    if args.num_eval_episodes <= 0 or args.num_envs <= 0:
        raise ValueError("--num-eval-episodes and --num-envs must be positive")
    if args.num_eval_episodes % args.num_envs != 0:
        raise ValueError("--num-eval-episodes must be divisible by --num-envs")

    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir
        else REPO_PATH / "logs" / "pushcube_wrist_ckpt_sweep"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    lock = threading.Lock()

    # seed rows from resume
    if args.resume:
        for step, path in discover_checkpoints(Path(args.checkpoint_dir)):
            for seed in parse_seeds(args.seeds):
                traj = output_dir / f"global_step_{step}" / f"seed_{seed}" / "trajectory_metrics.json"
                if traj.exists():
                    try:
                        rows.append(summarize_step_seed(step, seed, path, traj))
                    except Exception:
                        pass

    if args.manage_ray:
        start_shared_ray(args)
    try:
        if args.watch:
            watch_loop(args, output_dir, rows, lock)
        else:
            only_steps = set(args.step) if args.step else None
            checkpoints = discover_checkpoints(Path(args.checkpoint_dir), only_steps)
            if args.limit is not None:
                checkpoints = checkpoints[: args.limit]
            if not checkpoints:
                raise FileNotFoundError(f"No global_step_*/actor checkpoints under {args.checkpoint_dir}")
            for step, path in checkpoints:
                eval_checkpoint_all_seeds(step, path, output_dir, args, rows, lock)
    finally:
        if args.manage_ray and not args.watch:
            stop_shared_ray(args)
        # In --watch mode the loop is infinite; ray stays up until killed externally.


if __name__ == "__main__":
    main()
