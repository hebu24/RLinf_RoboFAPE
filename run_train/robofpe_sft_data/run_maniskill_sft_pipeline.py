#!/usr/bin/env python3
"""Run the RoboFPE ManiSkill collect -> convert -> q99 SFT data pipeline.

The inventory is JSON: {"hosts": [{"host": "10.0.0.1", "port": 22,
"gpus": [0, 1]}]}. All hosts must see the same /data/yingxi filesystem.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


TASKS = (
    "PegInsertionSide-v1",
    "PegInsertionVertical-v1",
    "PullCubeTool-golf",
    "UprightStack-v1",
    "PlugCharger-v1",
)
REPO = "/data/yingxi/RLinf_RoboFAPE"
ROBOFPE = "/data/yingxi/RoboFPE"
PYTHON = "/data/yingxi/robometer/failure_detection_env/bin/python"


def command(host: dict, script: str) -> list[str]:
    target = f"{host.get('user', 'root')}@{host['host']}"
    return [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-p", str(host.get("port", 22)), target, "bash", "-lc", script,
    ]


def remote(host: dict, script: str) -> None:
    result = subprocess.run(
        ["bash", "-lc", script] if host.get("local") else command(host, script),
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"remote command failed on {host['host']}:{host.get('port', 22)}")


def runtime_prefix() -> str:
    return " ".join([
        "export PYTHONPATH=/data/yingxi/RoboFPE:/data/yingxi/RoboFPE/mani_envs:/data/yingxi/RLinf_RoboFAPE:${PYTHONPATH:-};",
        "export MS_ASSET_DIR=/data/yingxi/robofac;",
        "export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json;",
        "export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json;",
        "export PYTHONUNBUFFERED=1;",
        f"cd {REPO};",
    ])


def slots(inventory: dict) -> list[tuple[dict, int]]:
    result = []
    for host in inventory["hosts"]:
        for gpu in host["gpus"]:
            result.append((host, int(gpu)))
    if not result:
        raise ValueError("inventory contains no GPU slots")
    return result


def write_status(root: Path, name: str, value: dict) -> None:
    path = root / "pipeline" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def split(total: int, count: int) -> list[int]:
    return [total // count + (index < total % count) for index in range(count)]


def collect_task(task: str, task_root: Path, gpu_slots: list[tuple[dict, int]], count: int, seed: int, resume: bool) -> float:
    started = time.monotonic()
    targets = split(count, len(gpu_slots))

    def one(index: int) -> None:
        host, gpu = gpu_slots[index]
        slot_root = task_root / "shards" / f"{host['host'].replace('.', '-')}-gpu{gpu}"
        manifest = slot_root / "collection_manifest.json"
        if manifest.exists():
            collected = json.loads(manifest.read_text(encoding="utf-8")).get("num_successes")
            if resume and collected == targets[index]:
                return
            raise RuntimeError(f"collection output already exists or is incomplete: {slot_root}")
        args = " ".join([
            f"mkdir -p /tmp/robofpe_sft_gpu{gpu};",
            f"XDG_RUNTIME_DIR=/tmp/robofpe_sft_gpu{gpu} CUDA_VISIBLE_DEVICES={gpu}", shlex.quote(PYTHON),
            "run_train/robofpe_sft_data/collect_sft_data.py collect",
            f"--task-id {shlex.quote(task)} --num-traj {targets[index]}",
            f"--output-dir {shlex.quote(str(slot_root))}",
            "--robot-uids panda_wristcam --success-only --sim-backend gpu",
            "--randomize-initial-poses --randomize-wrist-camera --randomize-render-camera --randomize-lighting",
            "--save-video --max-attempts-per-traj 100 --no-convert",
            f"--seed {seed + index * 1000000}",
        ])
        remote(host, runtime_prefix() + args)

    with ThreadPoolExecutor(max_workers=len(gpu_slots)) as pool:
        futures = [pool.submit(one, index) for index in range(len(gpu_slots)) if targets[index]]
        for future in as_completed(futures):
            future.result()
    return time.monotonic() - started


def postprocess(task: str, task_root: Path, filtered: Path, coordinator: dict, workers: int, resume: bool) -> tuple[float, float]:
    raw = task_root / "shards"
    dataset = task_root / "lerobot"
    task = task_root.name.removesuffix("_render_wrist")
    convert = " ".join([
        shlex.quote(PYTHON), "run_train/robofpe_sft_data/collect_sft_data.py convert",
        f"--task-id {shlex.quote(task)} --input {shlex.quote(str(raw))}",
        f"--dataset-dir {shlex.quote(str(dataset))} --robot-uids panda_wristcam",
        f"--num-convert-workers {workers}",
    ])
    started = time.monotonic()
    if (dataset / "meta" / "info.json").exists():
        if not resume:
            raise RuntimeError(f"conversion output already exists: {dataset}")
    else:
        remote(coordinator, runtime_prefix() + convert)
    remote(coordinator, runtime_prefix() + " ".join([
        shlex.quote(PYTHON), "run_train/robofpe_sft_data/collect_sft_data.py validate",
        f"--dataset-dir {shlex.quote(str(dataset))} --raw-input {shlex.quote(str(raw))}",
    ]))
    convert_seconds = time.monotonic() - started

    filter_cmd = " ".join([
        shlex.quote(PYTHON), shlex.quote(f"{ROBOFPE}/scripts/filter_lerobot_q99.py"),
        f"--input-dir {shlex.quote(str(dataset))} --output-dir {shlex.quote(str(filtered))}",
    ])
    started = time.monotonic()
    if (filtered / "meta" / "info.json").exists():
        if not resume:
            raise RuntimeError(f"filtered output already exists: {filtered}")
    else:
        remote(coordinator, runtime_prefix() + filter_cmd)
    remote(coordinator, runtime_prefix() + " ".join([
        shlex.quote(PYTHON), "run_train/robofpe_sft_data/collect_sft_data.py validate",
        f"--dataset-dir {shlex.quote(str(filtered))}",
    ]))
    return convert_seconds, time.monotonic() - started


def run(args: argparse.Namespace) -> None:
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    gpu_slots = slots(inventory)
    tasks = tuple(args.tasks.split(",")) if args.tasks else TASKS
    if any(task not in TASKS for task in tasks):
        raise ValueError(f"Unsupported task list: {tasks}")
    root = Path(args.output_root).resolve()
    if args.mode == "smoke":
        root = root / "smoke"
    count = args.num_trajectories or (
        args.smoke_trajectories if args.mode == "smoke" else 3200
    )
    started = time.monotonic()
    report = {
        "mode": args.mode,
        "count_per_task": count,
        "gpu_slots": len(gpu_slots),
        "convert_workers": args.convert_workers,
        "postprocess_workers": args.postprocess_workers,
        "tasks": {},
    }
    # GPU collection for the next task can proceed while CPU-only conversion
    # and filtering jobs run independently for completed tasks.
    with ThreadPoolExecutor(max_workers=args.postprocess_workers) as postprocess_pool:
        postprocess_futures = {}
        for offset, task in enumerate(tasks):
            task_root = root / f"{task}_render_wrist"
            filtered = root / f"{task}_render_wrist_filtered_q99" / "lerobot"
            collect_seconds = collect_task(task, task_root, gpu_slots, count, args.seed + offset * 10000, args.resume)
            report["tasks"][task] = {"collect_seconds": collect_seconds}
            write_status(root, "status.json", report)
            postprocess_futures[task] = postprocess_pool.submit(
                postprocess,
                task,
                task_root,
                filtered,
                gpu_slots[0][0],
                args.convert_workers,
                args.resume,
            )

        for task in tasks:
            convert_seconds, filter_seconds = postprocess_futures[task].result()
            report["tasks"][task].update(
                convert_seconds=convert_seconds,
                filter_seconds=filter_seconds,
            )
            write_status(root, "status.json", report)
    report["elapsed_seconds"] = time.monotonic() - started
    if args.mode == "smoke":
        import heapq

        scale = 3200 / count
        collect_ready = 0.0
        postprocess_workers = [0.0] * args.postprocess_workers
        heapq.heapify(postprocess_workers)
        for task in tasks:
            values = report["tasks"][task]
            collect_ready += values["collect_seconds"] * scale
            worker_ready = heapq.heappop(postprocess_workers)
            postprocess_ready = max(collect_ready, worker_ready)
            postprocess_ready += (values["convert_seconds"] + values["filter_seconds"]) * scale
            heapq.heappush(postprocess_workers, postprocess_ready)
        # GPU collection is sequential, while each completed task is scheduled
        # on the earliest available CPU postprocess worker.
        report["full_eta_seconds"] = max(collect_ready, max(postprocess_workers))
    write_status(root, "report.json", report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-root", default="/data/yingxi/datasets/robofpe_sft")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--smoke-trajectories", type=int, default=8)
    parser.add_argument(
        "--num-trajectories", type=int, default=None,
        help="Override trajectory count without changing the full pipeline path.",
    )
    parser.add_argument("--convert-workers", type=int, default=4)
    parser.add_argument("--postprocess-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--resume", action="store_true", help="Validate and skip completed stages; reject partial outputs.")
    args = parser.parse_args()
    if args.smoke_trajectories < 2:
        raise SystemExit("--smoke-trajectories must be at least 2")
    if args.num_trajectories is not None and args.num_trajectories < 2:
        raise SystemExit("--num-trajectories must be at least 2")
    if args.postprocess_workers < 1:
        raise SystemExit("--postprocess-workers must be at least 1")
    run(args)


if __name__ == "__main__":
    main()
