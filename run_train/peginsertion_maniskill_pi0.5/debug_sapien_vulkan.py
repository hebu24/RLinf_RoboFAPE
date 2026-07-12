#!/usr/bin/env python3
"""Diagnose SAPIEN/ManiSkill Vulkan render device selection.

This script intentionally does not collect data.  It checks what SAPIEN sees in
the same Python environment used by collection and then tries to create a small
PegInsertionVertical env with selected render backends.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import os.path as osp
import subprocess
import sys
import traceback


REPO_PATH = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)


def _register_peg_env() -> None:
    task_path = osp.join(
        REPO_PATH, "rlinf", "envs", "maniskill", "tasks", "peg_insertion_vertical.py"
    )
    spec = importlib.util.spec_from_file_location("_peg_insertion_vertical", task_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)


def _run_command(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("\n$ " + " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        print(result.stdout.strip())
        print(f"[exit={result.returncode}]")
    except Exception as exc:
        print(f"[command failed: {exc!r}]")


def _print_env() -> None:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "DISPLAY",
        "VK_ICD_FILENAMES",
        "SAPIEN_VULKAN_LIBRARY_PATH",
        "SAPIEN_DISABLE_RAY_TRACING",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
    ]
    print("Environment:")
    for key in keys:
        print(f"  {key}={os.environ.get(key, '')}")


def _print_sapien_devices(max_cuda_devices: int) -> None:
    import sapien

    print(f"\nsapien={sapien.__file__}")
    try:
        import mani_skill

        print(f"mani_skill={mani_skill.__file__}")
    except Exception as exc:
        print(f"mani_skill import failed: {exc!r}")

    try:
        import torch

        print(f"torch cuda available={torch.cuda.is_available()}")
        print(f"torch cuda device_count={torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"  torch cuda:{idx}={torch.cuda.get_device_name(idx)}")
    except Exception as exc:
        print(f"torch cuda probe failed: {exc!r}")

    try:
        print("\nsapien.render.get_device_summary():")
        print(sapien.render.get_device_summary())
    except Exception as exc:
        print(f"sapien.render.get_device_summary failed: {exc!r}")

    print("\nSAPIEN Device probes:")
    for name in ["cpu", "cuda", *[f"cuda:{idx}" for idx in range(max_cuda_devices)]]:
        try:
            device = sapien.Device(name)
            info = {
                "name": getattr(device, "name", None),
                "cuda_id": getattr(device, "cuda_id", None),
                "pci_string": getattr(device, "pci_string", None),
                "is_cuda": bool(getattr(device, "is_cuda", False)),
                "is_cpu": bool(getattr(device, "is_cpu", False)),
                "can_render": bool(device.can_render()),
                "can_present": bool(device.can_present()),
            }
            print(f"  {name}: {json.dumps(info, default=str)}")
        except Exception as exc:
            print(f"  {name}: ERROR {exc!r}")


def _try_env(render_backend: str, control_mode: str, sim_backend: str) -> bool:
    import gymnasium as gym

    print(
        f"\nTrying gym env: render_backend={render_backend!r}, "
        f"control_mode={control_mode!r}, sim_backend={sim_backend!r}"
    )
    env = None
    try:
        kwargs = {}
        if render_backend.lower() not in {"none", "null"}:
            kwargs["render_backend"] = render_backend
        else:
            kwargs["render_backend"] = None
        env = gym.make(
            "PegInsertionVertical-v1",
            num_envs=1,
            obs_mode="rgb",
            robot_uids="panda_wristcam",
            control_mode=control_mode,
            sim_backend=sim_backend,
            render_mode="all",
            reward_mode="normalized_dense",
            max_episode_steps=10,
            sensor_configs=dict(shader_pack="default", hand_camera=dict(width=224, height=224)),
            human_render_camera_configs=dict(shader_pack="default"),
            **kwargs,
        )
        obs, info = env.reset(seed=0, options={"randomize_initial_poses": True})
        del obs, info
        raw_obs = env.unwrapped.get_obs()
        sensor_keys = list(raw_obs.get("sensor_data", {}).keys())
        print(f"  OK: sensor_keys={sensor_keys}")
        return True
    except Exception as exc:
        print(f"  ERROR: {exc!r}")
        traceback.print_exc()
        return False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-backends",
        default="gpu:0,cuda:0,gpu:1,cuda:1,cpu",
        help="Comma-separated render backends to try.",
    )
    parser.add_argument("--control-mode", default="pd_joint_pos")
    parser.add_argument("--sim-backend", default="cpu")
    parser.add_argument("--max-cuda-devices", type=int, default=8)
    args = parser.parse_args()

    _print_env()
    _run_command(["nvidia-smi", "-L"])
    _run_command(["vulkaninfo", "--summary"])
    env = dict(os.environ)
    env["VK_ICD_FILENAMES"] = env.get(
        "VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json"
    )
    _run_command(["vulkaninfo", "--summary"], env=env)

    _register_peg_env()
    _print_sapien_devices(args.max_cuda_devices)

    results = {}
    for backend in [x.strip() for x in args.render_backends.split(",") if x.strip()]:
        results[backend] = _try_env(backend, args.control_mode, args.sim_backend)

    print("\nRender backend results:")
    print(json.dumps(results, indent=2))
    if not any(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
