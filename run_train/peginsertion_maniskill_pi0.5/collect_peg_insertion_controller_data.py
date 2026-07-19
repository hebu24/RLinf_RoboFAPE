#!/usr/bin/env python3
"""Collect PegInsertionVertical SFT data in the eval controller action domain.

This collector uses the existing RoboFPE motion-planning solver only to produce
    a successful reference trajectory.  It then dry-runs a no-image
``pd_ee_target_delta_pose`` controller tracker.  Only successful dry-runs are
replayed once more with RGB sensors enabled and written to disk.  The dataset
action labels are the raw physical target deltas that the policy should output
at eval time:

``[dx, dy, dz, droll, dpitch, dyaw, gripper]``.

The first six dimensions are split, not silently clipped, to stay inside the
Panda EE controller bounds.  ``debug.env_action`` stores the exact normalized
action sent to ManiSkill: position ``actions[:3] / 0.1``, rotation
``-actions[3:6] / 0.1``, plus the binary gripper command.  This makes training
labels, eval action conversion, and replay smoke auditable without an empirical
action scale.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _early_arg_value(name: str, default: str | None = None) -> str | None:
    prefix = f"{name}="
    for index, arg in enumerate(sys.argv[1:]):
        if arg == name and index + 2 <= len(sys.argv[1:]):
            return sys.argv[index + 2]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return default


def _early_configure_cuda_visible_devices() -> None:
    """Constrain CUDA before importing torch, ManiSkill, or SAPIEN.

    The CLI accepts physical nvidia-smi ids.  Once CUDA_VISIBLE_DEVICES is set,
    worker render backends are remapped to local ids later in main().
    """
    render_prefix = (
        _early_arg_value("--render-backend-prefix", "cuda") or "cuda"
    ).lower()
    if render_prefix not in {"", "default", "cuda", "sapien_cuda"}:
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return

    gpu_ids = _early_arg_value("--gpu-ids")
    if not gpu_ids:
        return
    visible = ",".join(value.strip() for value in gpu_ids.split(",") if value.strip())
    if not visible:
        return

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ["RLINF_COLLECTOR_SET_CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["RLINF_COLLECTOR_PHYSICAL_GPU_IDS"] = visible


_early_configure_cuda_visible_devices()

# SAPIEN only checks /usr/share/vulkan/icd.d/nvidia_icd.json before falling back
# to its bundled, older NVIDIA ICD. On this machine the real driver ICD lives in
# /etc, so pin it before importing ManiSkill/SAPIEN.
_NVIDIA_VK_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"
if "VK_ICD_FILENAMES" not in os.environ and osp.exists(_NVIDIA_VK_ICD):
    os.environ["VK_ICD_FILENAMES"] = _NVIDIA_VK_ICD
_SYSTEM_VULKAN_LOADER = "/usr/lib/x86_64-linux-gnu/libvulkan.so.1"
if "SAPIEN_VULKAN_LIBRARY_PATH" not in os.environ and osp.exists(_SYSTEM_VULKAN_LOADER):
    os.environ["SAPIEN_VULKAN_LIBRARY_PATH"] = _SYSTEM_VULKAN_LOADER

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transforms3d.quaternions import quat2mat

REPO_PATH = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

# Register the RLinf PegInsertionVertical env before importing the RoboFPE solver.
import importlib.util as _ilu

_TASK_PATH = osp.join(
    REPO_PATH, "rlinf", "envs", "maniskill", "tasks", "peg_insertion_vertical.py"
)
_spec = _ilu.spec_from_file_location("_peg_insertion_vertical", _TASK_PATH)
_task_module = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_task_module)

ROBOFPE = "/home/gpu4/yingxi/RoboFPE/mani_envs"
sys.path.insert(0, ROBOFPE)

# The solver imports the RoboFPE task class for type hints.  Keep that import
# from re-registering another environment implementation.
import types as _types

_tasks_module = _types.ModuleType("tasks")
_peg_module = _types.ModuleType("tasks.task_PegInsertionVertical")
_peg_module.PegInsertionVerticalEnv = type("PegInsertionVerticalEnv", (), {})
_tasks_module.task_PegInsertionVertical = _peg_module
sys.modules["tasks"] = _tasks_module
sys.modules["tasks.task_PegInsertionVertical"] = _peg_module

from solutions.solve_PegInsertionVertical import solve_peginsertionvertical

from rlinf.envs.maniskill.peg_insertion_pi05 import (
    PI05_ACTION_DIM,
    PI05_STATE_DIM,
    aligned_pi05_state_from_tcp_matrix,
    binary_gripper_from_solver_action,
    model_action_to_panda_env_action,
    normalized_abs_summary,
    panda_action_bound_summary,
    quantile_normalize_with_base_stats,
    split_target_delta_model_actions,
)

from peg_insertion_prompts import (
    DEFAULT_NUM_PROMPTS,
    DEFAULT_PROMPT_SEED,
    generate_prompts,
)
from insert_only_crop import (
    find_lift_end,
    generate_insert_only_prompts,
)

FPS = 20
IMAGE_SIZE = 224
RENDER_W = 640
RENDER_H = 480
CHUNK_SIZE = 1000

STATE_NAMES = ["tcp_x", "tcp_y", "tcp_z", "roll", "pitch", "yaw", "finger0", "finger1"]
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


class TrackingPlanError(RuntimeError):
    """Raised when a reference cannot be represented as controller actions."""


TASK_DESCRIPTIONS = generate_prompts(DEFAULT_NUM_PROMPTS, DEFAULT_PROMPT_SEED)


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pose_matrix_from_pq(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat2mat(np.asarray(q, dtype=np.float64))
    mat[:3, 3] = np.asarray(p, dtype=np.float64)
    return mat


def _tcp_matrix_in_robot_root(unwrapped) -> np.ndarray:
    tcp_pose_in_root = unwrapped.agent.robot.pose.inv() * unwrapped.agent.tcp.pose
    return tcp_pose_in_root.to_transformation_matrix().detach().cpu().numpy()[0]


def _qpos8(unwrapped) -> np.ndarray:
    qpos = _as_numpy(unwrapped.agent.robot.get_qpos())[0].astype(np.float32)
    state = np.zeros(8, dtype=np.float32)
    state[:7] = qpos[:7]
    state[7] = qpos[-1]
    return state


def _capture_obs(env) -> dict[str, Any]:
    unwrapped = env.unwrapped
    obs = unwrapped.get_obs()
    sensor_data = obs["sensor_data"]
    base = _as_numpy(sensor_data["base_camera"]["rgb"])[0].astype(np.uint8).copy()
    hand = _as_numpy(sensor_data["hand_camera"]["rgb"])[0].astype(np.uint8).copy()
    hand_back = (
        _as_numpy(sensor_data["hand_camera_back"]["rgb"])[0].astype(np.uint8).copy()
    )
    render = _capture_render_image(unwrapped, base)
    state_capture = _capture_state(env)
    return {
        "base_camera_rgb": base,
        "hand_camera_rgb": hand,
        "hand_camera_back_rgb": hand_back,
        "render_rgb": render,
        **state_capture,
    }


def _capture_state(env) -> dict[str, np.ndarray]:
    unwrapped = env.unwrapped
    state = _qpos8(unwrapped)
    tcp_matrix = _tcp_matrix_in_robot_root(unwrapped)
    state_tcp = aligned_pi05_state_from_tcp_matrix(tcp_matrix, [state[7], state[7]])
    return {
        "state": state,
        "state_tcp": state_tcp,
        "tcp_matrix_root": tcp_matrix,
    }


def _capture_render_image(unwrapped, fallback_base: np.ndarray) -> np.ndarray:
    render_camera = unwrapped._sensors.get("render_camera")
    if render_camera is not None:
        return _as_numpy(render_camera.get_obs()["rgb"])[0].astype(np.uint8).copy()
    return cv2.resize(fallback_base, (RENDER_W, RENDER_H))


def _images_from_obs(env, obs: dict[str, Any]) -> dict[str, np.ndarray]:
    sensor_data = obs["sensor_data"]
    base = _as_numpy(sensor_data["base_camera"]["rgb"])[0].astype(np.uint8).copy()
    hand = _as_numpy(sensor_data["hand_camera"]["rgb"])[0].astype(np.uint8).copy()
    hand_back = (
        _as_numpy(sensor_data["hand_camera_back"]["rgb"])[0].astype(np.uint8).copy()
    )
    render = _capture_render_image(env.unwrapped, base)
    return {
        "base_camera_rgb": base,
        "hand_camera_rgb": hand,
        "hand_camera_back_rgb": hand_back,
        "render_rgb": render,
    }


class ReferenceRecorder:
    """Record a successful pd_joint_pos solver trajectory."""

    def __init__(self, env):
        self.env = env
        self.records: list[dict[str, Any]] = []
        self._orig_step = env.step

    def start(self) -> None:
        self.records = []
        recorder = self

        def _step(action, **kwargs):
            pre = _capture_state(recorder.env)
            obs, reward, terminated, truncated, info = recorder._orig_step(
                action, **kwargs
            )
            post = _capture_state(recorder.env)
            recorder.records.append(
                {
                    "pre": pre,
                    "post": post,
                    "solver_action": np.asarray(action, dtype=np.float32).copy(),
                }
            )
            return obs, reward, terminated, truncated, info

        self.env.step = _step

    def stop(self) -> None:
        self.env.step = self._orig_step


def _render_backend_from_gpu_id(gpu_id: int, render_backend_prefix: str) -> str | None:
    prefix = render_backend_prefix.lower()
    if prefix in {"cpu", "sapien_cpu"}:
        return "cpu"
    if prefix in {"none", "off"}:
        return "none"
    if prefix in {"", "default", "gpu", "cuda", "sapien_cuda"}:
        return (
            f"gpu:{gpu_id}"
            if prefix in {"", "default", "gpu"}
            else f"{prefix}:{gpu_id}"
        )
    return f"{render_backend_prefix}:{gpu_id}"


def _gpu_label(gpu_id: int, render_backend_prefix: str) -> str:
    prefix = render_backend_prefix.lower()
    if prefix in {"", "default", "cuda", "sapien_cuda"}:
        visible = [
            value.strip()
            for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if value.strip()
        ]
        if 0 <= gpu_id < len(visible):
            return f"{gpu_id}/physical{visible[gpu_id]}"
    return str(gpu_id)


def _make_env(
    control_mode: str,
    gpu_id: int,
    render_backend_prefix: str,
    *,
    capture_images: bool,
):
    # Even with obs_mode="none", SAPIEN may need a renderer while parsing robot
    # visuals.  Keep one persistent renderer per env and avoid per-step RGB
    # observations for failed dry-runs.
    render_backend = _render_backend_from_gpu_id(gpu_id, render_backend_prefix)
    kwargs: dict[str, Any] = {}
    if render_backend is not None:
        kwargs["render_backend"] = render_backend
    env_kwargs: dict[str, Any] = dict(
        num_envs=1,
        obs_mode="rgb" if capture_images else "none",
        robot_uids="panda_wristcam",
        control_mode=control_mode,
        sim_backend="cpu",
        render_mode="all" if capture_images else None,
        reward_mode="normalized_dense",
        max_episode_steps=600,
        **kwargs,
    )
    if capture_images:
        env_kwargs.update(
            sensor_configs=dict(shader_pack="default"),
            human_render_camera_configs=dict(shader_pack="default"),
        )
    return gym.make("PegInsertionVertical-v1", **env_kwargs)


def _make_env_with_retry(
    control_mode: str,
    gpu_id: int,
    render_backend_prefix: str,
    *,
    capture_images: bool,
    max_retries: int = 2,
):
    last_err = None
    gpu_label = _gpu_label(gpu_id, render_backend_prefix)
    for attempt in range(max_retries):
        try:
            return _make_env(
                control_mode,
                gpu_id,
                render_backend_prefix,
                capture_images=capture_images,
            )
        except RuntimeError as exc:
            last_err = exc
            if "ErrorDeviceLost" not in str(exc) and "vk" not in str(exc).lower():
                raise
            print(
                f"[gpu{gpu_label}] Vulkan error creating {control_mode} env "
                f"attempt {attempt + 1}/{max_retries}; retrying..."
            )
            time.sleep(5.0)
    raise last_err


def _raw_action_to_env_action(raw_action: np.ndarray) -> np.ndarray:
    return model_action_to_panda_env_action(raw_action)


def _gripper_from_solver_action(action: np.ndarray) -> float:
    return binary_gripper_from_solver_action(action)


def _arm_controller(env):
    controller = env.unwrapped.agent.controller
    if hasattr(controller, "controllers"):
        return controller.controllers.get("arm", controller)
    return controller


def _sync_target_delta_pose_controller(env) -> None:
    """Reset the target-delta controller target to the current TCP pose."""
    arm_controller = _arm_controller(env)
    config = getattr(arm_controller, "config", None)
    if bool(getattr(config, "use_target", False)) and hasattr(
        arm_controller, "_target_pose"
    ):
        arm_controller._target_pose = arm_controller.ee_pose_at_base


def _base_norm_diagnostics(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    states = np.stack([row["observation.state_tcp"] for row in rows]).astype(np.float32)
    actions = np.stack([row["actions"] for row in rows]).astype(np.float32)
    return {
        "base_norm_state_abs": normalized_abs_summary(
            quantile_normalize_with_base_stats(states, "state")
        ),
        "base_norm_action_abs": normalized_abs_summary(
            quantile_normalize_with_base_stats(actions, "actions")
        ),
    }


def _build_tracking_plan(
    records: list[dict[str, Any]],
    initial_target_tcp: np.ndarray,
    max_episode_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    current_target_tcp = np.asarray(initial_target_tcp, dtype=np.float64)
    split_records = 0
    max_substeps = 1
    for ref_index, record in enumerate(records):
        target_tcp = np.asarray(record["post"]["tcp_matrix_root"], dtype=np.float64)
        gripper = _gripper_from_solver_action(record["solver_action"])
        try:
            pieces = split_target_delta_model_actions(
                current_target_tcp,
                target_tcp,
                gripper,
            )
        except ValueError as exc:
            raise TrackingPlanError(
                f"Failed to split reference step {ref_index}: {exc}"
            ) from exc
        if len(pieces) > 1:
            split_records += 1
            max_substeps = max(max_substeps, len(pieces))
        for substep_index, (raw_action, sub_target_tcp) in enumerate(pieces):
            plan.append(
                {
                    "actions": raw_action.astype(np.float32),
                    "debug.ref_next_tcp": sub_target_tcp.reshape(-1).astype(np.float32),
                    "debug.ref_source_step": int(ref_index),
                    "debug.ref_substep": int(substep_index),
                }
            )
            current_target_tcp = np.asarray(sub_target_tcp, dtype=np.float64)
            if len(plan) > max_episode_steps:
                raise TrackingPlanError(
                    "Target-delta plan exceeds max_episode_steps after splitting: "
                    f"{len(plan)} > {max_episode_steps}"
                )
    if not plan:
        raise TrackingPlanError("Reference produced no controller actions")
    actions = np.stack([item["actions"] for item in plan]).astype(np.float32)
    diagnostics = {
        "tracking_plan_len": int(len(plan)),
        "tracking_reference_len": int(len(records)),
        "tracking_split_records": int(split_records),
        "tracking_max_substeps_per_reference": int(max_substeps),
        **panda_action_bound_summary(actions),
    }
    if not diagnostics["within_controller_bounds"]:
        raise TrackingPlanError(
            f"Target-delta plan contains out-of-bound actions: {diagnostics}"
        )
    return plan, diagnostics


def _solver_result_summary(result: Any) -> dict[str, Any]:
    if isinstance(result, tuple):
        status = result[0]
        stage_records = (
            result[1] if len(result) > 1 and isinstance(result[1], list) else []
        )
    else:
        status = result
        stage_records = []
    last_stage = stage_records[-1]["stage"] if stage_records else None
    if isinstance(status, np.generic):
        status = status.item()
    if not isinstance(status, (int, float, str, bool)) and status is not None:
        status = repr(status)
    return {
        "solver_status": status,
        "num_stage_records": len(stage_records),
        "last_stage": last_stage,
        "stage_records": stage_records,
    }


def _collect_reference(
    env,
    seed: int,
    reset_options: dict[str, Any],
):
    recorder = ReferenceRecorder(env)
    try:
        recorder.start()
        result = solve_peginsertionvertical(
            env,
            seed=seed,
            debug=False,
            vis=False,
            reset_options=reset_options,
        )
        recorder.stop()
        summary = _solver_result_summary(result)
        success = False
        if not (isinstance(result, int) and result == -1):
            info = env.unwrapped.evaluate()
            success = bool(info["success"].detach().cpu().numpy()[0])
        summary["env_success"] = success
        summary["num_records"] = len(recorder.records)
        if not success or not recorder.records:
            return {"success": False, "records": recorder.records, **summary}
        return {"success": True, "records": recorder.records, **summary}
    finally:
        recorder.stop()


def _reset_for_tracking(
    env,
    seed: int,
    reset_options: dict[str, Any],
    reset_state: np.ndarray | None,
    *,
    capture_images: bool,
):
    if reset_state is None:
        reset_result = env.reset(seed=seed, options=reset_options)
        obs = reset_result[0] if capture_images else None
        reset_state = _as_numpy(env.unwrapped.get_state())[0].astype(np.float32)
        _sync_target_delta_pose_controller(env)
        return obs, reset_state

    env.reset(seed=0, options={"randomize_initial_poses": False})
    env.unwrapped.set_state(
        torch.as_tensor(reset_state, dtype=torch.float32).reshape(1, -1)
    )
    _sync_target_delta_pose_controller(env)
    obs = env.unwrapped.get_obs() if capture_images else None
    return obs, reset_state


def _track_reference_with_controller(
    env,
    records: list[dict[str, Any]],
    seed: int,
    reset_options: dict[str, Any],
    *,
    capture_images: bool,
    reset_state: np.ndarray | None = None,
    action_plan: list[dict[str, Any]] | None = None,
):
    obs, reset_state = _reset_for_tracking(
        env,
        seed,
        reset_options,
        reset_state,
        capture_images=capture_images,
    )

    plan_diagnostics = None
    if action_plan is None:
        initial_target_tcp = _capture_state(env)["tcp_matrix_root"]
        max_episode_steps = getattr(
            env.unwrapped,
            "max_episode_steps",
            getattr(env.unwrapped, "_max_episode_steps", 600),
        )
        max_episode_steps = int(max_episode_steps or 600)
        action_plan, plan_diagnostics = _build_tracking_plan(
            records,
            initial_target_tcp,
            max_episode_steps=max_episode_steps,
        )

    rows = []
    base_frames = []
    wrist_frames = []
    wrist_back_frames = []
    render_frames = []
    for plan_item in action_plan:
        pre = _capture_state(env)
        images = _images_from_obs(env, obs) if capture_images else None
        target_tcp = np.asarray(
            plan_item["debug.ref_next_tcp"], dtype=np.float32
        ).reshape(4, 4)
        raw_action = np.asarray(plan_item["actions"], dtype=np.float32)
        env_action = _raw_action_to_env_action(raw_action)
        step_obs, reward, terminated, truncated, info = env.step(
            env_action.reshape(1, -1)
        )
        del reward, terminated, truncated, info
        obs = step_obs if capture_images else None
        post = _capture_state(env)

        if capture_images:
            rows.append(
                {
                    "actions": raw_action,
                    "debug.env_action": env_action,
                    "debug.ref_next_tcp": target_tcp.reshape(-1).astype(np.float32),
                    "debug.tcp_before": pre["tcp_matrix_root"]
                    .reshape(-1)
                    .astype(np.float32),
                    "debug.tcp_after": post["tcp_matrix_root"]
                    .reshape(-1)
                    .astype(np.float32),
                    "observation.state": pre["state"],
                    "observation.state_tcp": pre["state_tcp"],
                    "episode_reset_state": reset_state,
                }
            )
            assert images is not None
            base_frames.append(images["base_camera_rgb"])
            wrist_frames.append(images["hand_camera_rgb"])
            wrist_back_frames.append(images["hand_camera_back_rgb"])
            render_frames.append(images["render_rgb"])

    eval_info = env.unwrapped.evaluate()
    success = bool(eval_info["success"].detach().cpu().numpy()[0])
    if not capture_images:
        return (
            success,
            reset_state,
            rows,
            None,
            None,
            None,
            None,
            action_plan,
            plan_diagnostics,
        )
    norm_diagnostics = _base_norm_diagnostics(rows)
    return (
        success,
        reset_state,
        rows,
        np.stack(base_frames),
        np.stack(wrist_frames),
        np.stack(wrist_back_frames),
        np.stack(render_frames),
        action_plan,
        norm_diagnostics,
    )


def _write_video(path: str, frames: np.ndarray) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


_VIS_KEYS = (
    ("observation.images.top", "base"),
    ("observation.images.wrist", "wrist_front"),
    ("observation.images.wrist_back", "wrist_back"),
)


def _create_visualization_samples(root, total, count=10, seed=0):
    count = min(max(count, 0), total)
    if not count:
        return
    ids = sorted(
        map(int, np.random.default_rng(seed).choice(total, count, replace=False))
    )
    out = osp.join(root, "visualization_samples")
    if osp.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    manifest = []
    for eid in ids:
        chunk = eid // CHUNK_SIZE
        captures = []
        sources = []
        for key, label in _VIS_KEYS:
            src = osp.join(
                root, "videos", f"chunk-{chunk:03d}", key, f"episode_{eid:06d}.mp4"
            )
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open {src}")
            captures.append((cap, label))
            sources.append(osp.relpath(src, root))
        dst = osp.join(out, f"episode_{eid:06d}_three_view.mp4")
        writer = cv2.VideoWriter(
            dst, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (IMAGE_SIZE * 3, IMAGE_SIZE)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open {dst}")
        frames = 0
        try:
            while True:
                panels = []
                for cap, label in captures:
                    ok, frame = cap.read()
                    if not ok:
                        panels = []
                        break
                    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                    cv2.putText(
                        frame,
                        label,
                        (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    panels.append(frame)
                if not panels:
                    break
                joined = np.concatenate(panels, axis=1)
                cv2.putText(
                    joined,
                    f"episode {eid}",
                    (6, IMAGE_SIZE - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(joined)
                frames += 1
        finally:
            writer.release()
            for cap, _ in captures:
                cap.release()
        manifest.append(
            {
                "episode_index": eid,
                "frames": frames,
                "video": osp.relpath(dst, root),
                "sources": sources,
            }
        )
    with open(osp.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "samples": manifest}, f, indent=2)
    print(f"Created {len(manifest)} visualization samples -> {out}")


class LeRobotControllerWriter:
    def __init__(self, outdir: str, *, resume: bool = False):
        self.outdir = osp.abspath(outdir)
        self.rows: list[pd.DataFrame] = []
        self.episodes: list[dict[str, Any]] = []
        self.global_index = 0
        self.task_to_idx: dict[str, int] = {}
        self.tasks: list[str] = []
        # Recorded into meta/info.json for traceability. Set by the collector
        # via the --collect-mode flag ("full" or "insert_only").
        self.collect_mode = "insert_only"
        os.makedirs(self.outdir, exist_ok=True)
        if resume:
            self._load_existing()

    def _load_existing(self) -> None:
        """Restore writer state from completed episode parquet files."""
        paths = sorted(Path(self.outdir).glob("data/chunk-*/episode_*.parquet"))
        for expected_episode, path in enumerate(paths):
            try:
                episode_index = int(path.stem.removeprefix("episode_"))
            except ValueError as exc:
                raise RuntimeError(f"Invalid episode parquet name: {path}") from exc
            if episode_index != expected_episode:
                raise RuntimeError(
                    f"Cannot resume non-contiguous shard {self.outdir}: expected "
                    f"episode_{expected_episode:06d}.parquet, found {path.name}"
                )
            df = pd.read_parquet(path)
            if df.empty:
                raise RuntimeError(f"Cannot resume empty episode parquet: {path}")
            task = str(df["task"].iloc[0])
            if df["task"].astype(str).nunique() != 1:
                raise RuntimeError(f"Episode has multiple tasks: {path}")
            task_index = self._task_index(task)
            length = len(df)
            df["episode_index"] = np.full(length, episode_index, dtype=np.int64)
            df["index"] = np.arange(
                self.global_index, self.global_index + length, dtype=np.int64
            )
            df["task_index"] = np.full(length, task_index, dtype=np.int64)
            df["task"] = [task] * length
            self.rows.append(df)
            self.episodes.append(
                {
                    "episode_index": episode_index,
                    "dataset_from_index": self.global_index,
                    "dataset_to_index": self.global_index + length,
                    "tasks": [task],
                    "length": length,
                    "success": True,
                }
            )
            self.global_index += length
        if paths:
            print(
                f"Resumed shard {self.outdir}: {len(self.episodes)} completed "
                f"episodes, {self.global_index} frames"
            )

    def resume_seed(self, base_seed: int) -> int:
        """Return the first seed not recorded in the append-only attempt journal."""
        path = osp.join(self.outdir, "meta", "collection_attempts.jsonl")
        if not osp.exists(path):
            return base_seed
        max_seed = base_seed - 1
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    max_seed = max(max_seed, int(json.loads(line)["seed"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        return max_seed + 1
    def _task_index(self, task: str) -> int:
        if task not in self.task_to_idx:
            self.task_to_idx[task] = len(self.tasks)
            self.tasks.append(task)
        return self.task_to_idx[task]

    def log_attempt(self, record: dict[str, Any]) -> None:
        meta_dir = osp.join(self.outdir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        path = osp.join(meta_dir, "collection_attempts.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def add_episode(
        self,
        episode_rows: list[dict[str, Any]],
        base_frames: np.ndarray,
        wrist_frames: np.ndarray,
        wrist_back_frames: np.ndarray,
        render_frames: np.ndarray,
        task: str,
    ) -> int:
        episode_index = len(self.episodes)
        task_index = self._task_index(task)
        length = len(episode_rows)
        frame_index = np.arange(length, dtype=np.int64)
        df = pd.DataFrame(episode_rows)
        df["timestamp"] = frame_index.astype(np.float32) / FPS
        df["frame_index"] = frame_index
        df["episode_index"] = np.full(length, episode_index, dtype=np.int64)
        df["index"] = np.arange(
            self.global_index, self.global_index + length, dtype=np.int64
        )
        df["task_index"] = np.full(length, task_index, dtype=np.int64)
        df["task"] = [task] * length
        self.rows.append(df)
        self.episodes.append(
            {
                "episode_index": episode_index,
                "dataset_from_index": self.global_index,
                "dataset_to_index": self.global_index + length,
                "tasks": [task],
                "length": length,
                "success": True,
            }
        )
        self.global_index += length

        chunk = episode_index // CHUNK_SIZE
        for key, frames in [
            ("observation.images.top", base_frames),
            ("observation.images.wrist", wrist_frames),
            ("observation.images.wrist_back", wrist_back_frames),
            ("observation.images.render", render_frames),
        ]:
            video_path = osp.join(
                self.outdir,
                "videos",
                f"chunk-{chunk:03d}",
                key,
                f"episode_{episode_index:06d}.mp4",
            )
            _write_video(video_path, frames)
        _write_episode_table(self.outdir, df, episode_index)
        self.flush_metadata(write_stats=False)
        return episode_index

    def flush_metadata(self, *, write_stats: bool) -> None:
        if not self.rows:
            return
        df = pd.concat(self.rows, ignore_index=True)
        df["task"] = df["task"].astype("string")
        _write_dataset_metadata(
            self.outdir,
            df,
            self.episodes,
            self.tasks,
            write_stats=write_stats,
            collect_mode=self.collect_mode,
        )

    def finalize(self) -> None:
        if not self.rows:
            print(f"No episodes collected in {self.outdir}")
            return
        self.flush_metadata(write_stats=True)


def _list_type(dtype=pa.float32()):
    return pa.list_(dtype)


def _dataset_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("actions", _list_type()),
            pa.field("debug.env_action", _list_type()),
            pa.field("debug.ref_next_tcp", _list_type()),
            pa.field("debug.tcp_before", _list_type()),
            pa.field("debug.tcp_after", _list_type()),
            pa.field("observation.state", _list_type()),
            pa.field("observation.state_tcp", _list_type()),
            pa.field("episode_reset_state", _list_type()),
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("task", pa.string()),
        ]
    )


def _array_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "p01": np.percentile(values, 1, axis=0).tolist(),
        "p99": np.percentile(values, 99, axis=0).tolist(),
        "count": [int(len(values))],
    }


def _features(num_reset_state_dims: int) -> dict[str, Any]:
    features = {
        "actions": {
            "dtype": "float32",
            "shape": [PI05_ACTION_DIM],
            "names": ACTION_NAMES,
            "fps": float(FPS),
        },
        "debug.env_action": {
            "dtype": "float32",
            "shape": [PI05_ACTION_DIM],
            "names": ACTION_NAMES,
            "fps": float(FPS),
        },
        "debug.ref_next_tcp": {
            "dtype": "float32",
            "shape": [16],
            "names": None,
            "fps": float(FPS),
        },
        "debug.tcp_before": {
            "dtype": "float32",
            "shape": [16],
            "names": None,
            "fps": float(FPS),
        },
        "debug.tcp_after": {
            "dtype": "float32",
            "shape": [16],
            "names": None,
            "fps": float(FPS),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": [f"qpos_{i}" for i in range(8)],
            "fps": float(FPS),
        },
        "observation.state_tcp": {
            "dtype": "float32",
            "shape": [PI05_STATE_DIM],
            "names": STATE_NAMES,
            "fps": float(FPS),
        },
        "episode_reset_state": {
            "dtype": "float32",
            "shape": [num_reset_state_dims],
            "names": None,
            "fps": float(FPS),
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [1],
            "names": None,
            "fps": float(FPS),
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "fps": float(FPS),
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "fps": float(FPS),
        },
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(FPS)},
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "fps": float(FPS),
        },
        "task": {"dtype": "string", "shape": [1], "names": None, "fps": float(FPS)},
    }
    for key, height, width in [
        ("observation.images.top", IMAGE_SIZE, IMAGE_SIZE),
        ("observation.images.wrist", IMAGE_SIZE, IMAGE_SIZE),
        ("observation.images.wrist_back", IMAGE_SIZE, IMAGE_SIZE),
        ("observation.images.render", RENDER_H, RENDER_W),
    ]:
        features[key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(FPS),
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return features


LIST_FIELDS = [
    "actions",
    "debug.env_action",
    "debug.ref_next_tcp",
    "debug.tcp_before",
    "debug.tcp_after",
    "observation.state",
    "observation.state_tcp",
    "episode_reset_state",
]


def _prepare_dataset_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for field in LIST_FIELDS:
        df[field] = df[field].map(
            lambda value: np.asarray(value, dtype=np.float32).tolist()
        )
    return df


def _write_episode_table(
    outdir: str, episode_df: pd.DataFrame, episode_index: int
) -> None:
    episode_df = _prepare_dataset_df(episode_df.reset_index(drop=True))
    chunk = episode_index // CHUNK_SIZE
    chunk_dir = osp.join(outdir, "data", f"chunk-{chunk:03d}")
    os.makedirs(chunk_dir, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(
            episode_df, schema=_dataset_schema(), preserve_index=False
        ),
        osp.join(chunk_dir, f"episode_{episode_index:06d}.parquet"),
    )


def _write_dataset_tables(
    outdir: str,
    df: pd.DataFrame,
    episodes: list[dict[str, Any]],
    tasks: list[str],
    collect_mode: str = "insert_only",
) -> None:
    df = _prepare_dataset_df(df)
    for episode_index in sorted(df["episode_index"].unique()):
        episode_index = int(episode_index)
        episode_df = df[df["episode_index"] == episode_index].reset_index(drop=True)
        _write_episode_table(outdir, episode_df, episode_index)

    _write_dataset_metadata(
        outdir, df, episodes, tasks, write_stats=True, collect_mode=collect_mode
    )


def _write_dataset_metadata(
    outdir: str,
    df: pd.DataFrame,
    episodes: list[dict[str, Any]],
    tasks: list[str],
    *,
    write_stats: bool,
    collect_mode: str = "insert_only",
) -> None:
    meta_dir = osp.join(outdir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(osp.join(meta_dir, "episodes.jsonl"), "w", encoding="utf-8") as f:
        for episode in episodes:
            f.write(json.dumps(episode) + "\n")
    with open(osp.join(meta_dir, "tasks.jsonl"), "w", encoding="utf-8") as f:
        for idx, task in enumerate(tasks):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
    pd.DataFrame({"task_index": list(range(len(tasks))), "task": tasks}).to_parquet(
        osp.join(meta_dir, "tasks.parquet"), index=False
    )

    if write_stats:
        stats: dict[str, Any] = {}
        for field in [
            "actions",
            "debug.env_action",
            "observation.state",
            "observation.state_tcp",
            "episode_reset_state",
        ]:
            stats[field] = _array_summary(
                np.stack(df[field].to_numpy()).astype(np.float32)
            )
        actions = np.stack(df["actions"].to_numpy()).astype(np.float32)
        states_tcp = np.stack(df["observation.state_tcp"].to_numpy()).astype(np.float32)
        stats["base_norm.actions.abs"] = normalized_abs_summary(
            quantile_normalize_with_base_stats(actions, "actions")
        )
        stats["base_norm.observation.state_tcp.abs"] = normalized_abs_summary(
            quantile_normalize_with_base_stats(states_tcp, "state")
        )
        for field in [
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]:
            values = df[field].to_numpy()
            stats[field] = {
                "mean": [float(values.mean())],
                "std": [float(values.std())],
                "min": [float(values.min())],
                "max": [float(values.max())],
                "count": [int(len(values))],
            }
        with open(osp.join(meta_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    num_reset_state_dims = len(df["episode_reset_state"].iloc[0])
    data_size = sum(
        path.stat().st_size for path in Path(outdir).rglob("data/**/*.parquet")
    )
    total_episodes = int(df["episode_index"].nunique())
    info = {
        "codebase_version": "v2.0",
        "dataset_variant": "peg-insertion-controller-domain-v1",
        "collect_mode": collect_mode,
        "robot_type": "panda_wristcam",
        "total_episodes": total_episodes,
        "total_frames": int(len(df)),
        "total_tasks": len(tasks),
        "total_videos": total_episodes * 3,
        "total_chunks": int((total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE),
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "data_files_size_in_mb": int(data_size / (1024 * 1024)),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _features(num_reset_state_dims),
        "action_semantics": {
            "policy_output": "physical target-delta [dx, dy, dz, droll, dpitch, dyaw, gripper]",
            "rotation": "Euler XYZ",
            "controller": "pd_ee_target_delta_pose",
            "env_action_mapping": (
                "debug.env_action[:3] = actions[:3] / 0.1; "
                "debug.env_action[3:6] = -actions[3:6] / 0.1; "
                "debug.env_action[6] = actions[6]"
            ),
            "gripper": "+1=open, -1=close, matching base pi0.5 ManiSkill stats",
            "required_eval_action_scale": 1.0,
            "collection_tracking": "no-image dry-run first, RGB replay only after success",
            "partial_flush": "each accepted episode writes parquet/videos immediately; stats.json is final after finalize",
        },
    }
    with open(osp.join(meta_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    with open(
        osp.join(meta_dir, "collection_progress.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "total_episodes": total_episodes,
                "total_frames": int(len(df)),
                "stats_finalized": bool(write_stats),
                "last_episode_index": int(
                    max(episode["episode_index"] for episode in episodes)
                ),
            },
            f,
            indent=2,
        )


def _collect_episodes(
    num_traj: int,
    base_seed: int,
    writer: LeRobotControllerWriter,
    gpu_id: int,
    max_attempts: int,
    render_backend_prefix: str,
    progress_prefix: str = "",
    collect_mode: str = "insert_only",
):
    reset_options = {"randomize_initial_poses": True}
    seed = base_seed
    passed = 0
    attempts = 0
    ref_fail = 0
    ctrl_dry_fail = 0
    ctrl_capture_fail = 0
    # Prompt pool: insert_only uses single-stage "insert the peg" wording (matches
    # convert_to_insert_only.py); full keeps the two-stage pick-up-and-insert set.
    if collect_mode == "insert_only":
        task_pool = generate_insert_only_prompts(
            DEFAULT_NUM_PROMPTS, DEFAULT_PROMPT_SEED
        )
    else:
        task_pool = TASK_DESCRIPTIONS
    writer.collect_mode = collect_mode
    render_backend = _render_backend_from_gpu_id(gpu_id, render_backend_prefix)
    gpu_label = _gpu_label(gpu_id, render_backend_prefix)
    pbar = tqdm(total=num_traj, desc=progress_prefix or "Collect controller data")
    reference_env = None
    controller_dry_env = None
    controller_capture_env = None
    try:
        print(
            f"{progress_prefix}Creating persistent ManiSkill envs on gpu{gpu_label} "
            f"render_backend={render_backend}: pd_joint_pos(no-image), "
            "pd_ee_target_delta_pose(no-image dry-run), "
            "pd_ee_target_delta_pose(RGB capture)"
        )
        reference_env = _make_env_with_retry(
            "pd_joint_pos",
            gpu_id,
            render_backend_prefix,
            capture_images=False,
        )
        controller_dry_env = _make_env_with_retry(
            "pd_ee_target_delta_pose",
            gpu_id,
            render_backend_prefix,
            capture_images=False,
        )
        controller_capture_env = _make_env_with_retry(
            "pd_ee_target_delta_pose",
            gpu_id,
            render_backend_prefix,
            capture_images=True,
        )
        print(
            f"{progress_prefix}Persistent envs ready on gpu{gpu_label}; "
            "subsequent attempts will use reset(seed=...) only. "
            "RGB capture runs only after no-image controller dry-run succeeds."
        )
        while passed < num_traj:
            if attempts >= max_attempts:
                pbar.close()
                raise RuntimeError(
                    f"Exceeded max attempts while collecting controller-domain data: "
                    f"passed={passed}, attempts={attempts}, ref_fail={ref_fail}, "
                    f"ctrl_dry_fail={ctrl_dry_fail}, "
                    f"ctrl_capture_fail={ctrl_capture_fail}. Check replay/controller tracking before "
                    "running large collection."
                )
            attempts += 1
            attempt_started_at = time.time()
            attempt_record: dict[str, Any] = {
                "seed": seed,
                "attempt": attempts,
                "passed_before": passed,
                "gpu_id": gpu_id,
                "gpu_label": gpu_label,
                "render_backend": render_backend,
            }
            try:
                ref_started_at = time.time()
                reference = _collect_reference(reference_env, seed, reset_options)
                attempt_record.update(
                    {
                        "ref_time_sec": time.time() - ref_started_at,
                        "ref_success": bool(reference["success"]),
                        "ref_num_records": int(reference["num_records"]),
                        "ref_solver_status": reference["solver_status"],
                        "ref_num_stage_records": int(reference["num_stage_records"]),
                        "ref_last_stage": reference["last_stage"],
                    }
                )
                if not reference["success"]:
                    ref_fail += 1
                    attempt_record.update(
                        {
                            "status": "ref_fail",
                            "passed_after": passed,
                            "ref_fail": ref_fail,
                            "ctrl_dry_fail": ctrl_dry_fail,
                            "ctrl_capture_fail": ctrl_capture_fail,
                            "total_time_sec": time.time() - attempt_started_at,
                        }
                    )
                    writer.log_attempt(attempt_record)
                    seed += 1
                    continue
                records = reference["records"]
                dry_started_at = time.time()
                try:
                    (
                        dry_success,
                        reset_state,
                        _,
                        _,
                        _,
                        _,
                        _,
                        action_plan,
                        dry_plan_diagnostics,
                    ) = _track_reference_with_controller(
                        controller_dry_env,
                        records,
                        seed,
                        reset_options,
                        capture_images=False,
                    )
                except TrackingPlanError as exc:
                    dry_success = False
                    reset_state = None
                    action_plan = None
                    dry_plan_diagnostics = {"tracking_plan_error": str(exc)}
                attempt_record.update(
                    {
                        "dry_time_sec": time.time() - dry_started_at,
                        "dry_success": bool(dry_success),
                        **(dry_plan_diagnostics or {}),
                    }
                )
                if not dry_success:
                    ctrl_dry_fail += 1
                    attempt_record.update(
                        {
                            "status": "ctrl_dry_fail",
                            "passed_after": passed,
                            "ref_fail": ref_fail,
                            "ctrl_dry_fail": ctrl_dry_fail,
                            "ctrl_capture_fail": ctrl_capture_fail,
                            "total_time_sec": time.time() - attempt_started_at,
                        }
                    )
                    writer.log_attempt(attempt_record)
                    seed += 1
                    continue
                capture_started_at = time.time()
                assert reset_state is not None and action_plan is not None
                (
                    success,
                    _,
                    rows,
                    base,
                    wrist,
                    wrist_back,
                    render,
                    _,
                    norm_diagnostics,
                ) = _track_reference_with_controller(
                    controller_capture_env,
                    records,
                    seed,
                    reset_options,
                    capture_images=True,
                    reset_state=reset_state,
                    action_plan=action_plan,
                )
                attempt_record.update(
                    {
                        "capture_time_sec": time.time() - capture_started_at,
                        "capture_success": bool(success),
                        **(norm_diagnostics or {}),
                    }
                )
                if not success:
                    ctrl_capture_fail += 1
                    attempt_record.update(
                        {
                            "status": "ctrl_capture_fail",
                            "passed_after": passed,
                            "ref_fail": ref_fail,
                            "ctrl_dry_fail": ctrl_dry_fail,
                            "ctrl_capture_fail": ctrl_capture_fail,
                            "total_time_sec": time.time() - attempt_started_at,
                        }
                    )
                    writer.log_attempt(attempt_record)
                    seed += 1
                    continue
                assert (
                    base is not None
                    and wrist is not None
                    and wrist_back is not None
                    and render is not None
                )
                # insert_only mode: crop to the move-and-insert segment starting at
                # the lift-end frame (drops reach/grasp/close/lift prefix), matching
                # convert_to_insert_only.py via the shared find_lift_end criterion.
                if collect_mode == "insert_only":
                    actions_arr = np.stack([r["actions"] for r in rows])
                    state_tcp_arr = np.stack([r["observation.state_tcp"] for r in rows])
                    t_lift = find_lift_end(actions_arr, state_tcp_arr)
                    if t_lift is None or len(rows) - t_lift < 10:
                        attempt_record.update(
                            {
                                "status": "insert_crop_skip",
                                "t_lift": t_lift,
                                "episode_len": len(rows),
                                "passed_after": passed,
                                "ref_fail": ref_fail,
                                "ctrl_dry_fail": ctrl_dry_fail,
                                "ctrl_capture_fail": ctrl_capture_fail,
                                "total_time_sec": time.time() - attempt_started_at,
                            }
                        )
                        writer.log_attempt(attempt_record)
                        seed += 1
                        continue
                    rows = rows[t_lift:]
                    base = base[t_lift:]
                    wrist = wrist[t_lift:]
                    wrist_back = wrist_back[t_lift:]
                    render = render[t_lift:]
                task = random.Random(seed).choice(task_pool)
                episode_index = writer.add_episode(
                    rows, base, wrist, wrist_back, render, task
                )
                passed += 1
                attempt_record.update(
                    {
                        "status": "success",
                        "episode_index": episode_index,
                        "episode_len": len(rows),
                        "passed_after": passed,
                        "ref_fail": ref_fail,
                        "ctrl_dry_fail": ctrl_dry_fail,
                        "ctrl_capture_fail": ctrl_capture_fail,
                        "total_time_sec": time.time() - attempt_started_at,
                    }
                )
                writer.log_attempt(attempt_record)
                pbar.update(1)
                pbar.set_postfix(
                    dict(
                        pass_rate=f"{passed / attempts:.1%}",
                        ref_fail=ref_fail,
                        dry_fail=ctrl_dry_fail,
                        cap_fail=ctrl_capture_fail,
                        seed=seed,
                    )
                )
                seed += 1
            except Exception as exc:
                message = str(exc).lower()
                if (
                    "failed to find device" in message
                    or "cuda:" in message
                    or "vulkan" in message
                    or "render_device" in message
                ):
                    pbar.close()
                    raise RuntimeError(
                        f"Fatal environment/render backend error on gpu{gpu_label}: {exc}. "
                        "Persistent-env collector will not recreate ManiSkill envs "
                        "inside the same process because repeated svulkan2 renderer "
                        "create/close can poison Vulkan state."
                    ) from exc
                attempt_record.update(
                    {
                        "status": "exception",
                        "error": repr(exc),
                        "passed_after": passed,
                        "ref_fail": ref_fail,
                        "ctrl_dry_fail": ctrl_dry_fail,
                        "ctrl_capture_fail": ctrl_capture_fail,
                        "total_time_sec": time.time() - attempt_started_at,
                    }
                )
                writer.log_attempt(attempt_record)
                print(f"{progress_prefix}Error seed={seed}: {exc}")
                traceback.print_exc()
                seed += 1
    finally:
        pbar.close()
        print(
            f"{progress_prefix}Closing persistent ManiSkill envs on gpu{gpu_label}; "
            f"passed={passed}, attempts={attempts}, ref_fail={ref_fail}, "
            f"ctrl_dry_fail={ctrl_dry_fail}, ctrl_capture_fail={ctrl_capture_fail}"
        )
        for env in [controller_capture_env, controller_dry_env, reference_env]:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    print(f"{progress_prefix}Warning: env.close() failed: {exc}")
    return passed, attempts, ref_fail, ctrl_dry_fail, ctrl_capture_fail


def _collect_worker(
    worker_id: int,
    num_traj: int,
    base_seed: int,
    shard_dir: str,
    gpu_id: int,
    startup_delay: float,
    max_attempts_multiplier: int,
    render_backend_prefix: str,
    collect_mode: str = "insert_only",
    resume: bool = False,
):
    if startup_delay > 0:
        time.sleep(startup_delay)
    gpu_label = _gpu_label(gpu_id, render_backend_prefix)
    writer = LeRobotControllerWriter(shard_dir, resume=resume)
    existing = len(writer.episodes)
    if existing > num_traj:
        raise RuntimeError(
            f"Shard {shard_dir} already has {existing} episodes, exceeding its "
            f"configured target {num_traj}."
        )
    remaining = num_traj - existing
    if remaining == 0:
        writer.finalize()
        print(f"[w{worker_id}/gpu{gpu_label}] Shard already complete: {existing}/{num_traj}")
        return (worker_id, shard_dir, 0, 0, 0, 0, 0)
    start_seed = writer.resume_seed(base_seed) if resume else base_seed
    print(
        f"[w{worker_id}/gpu{gpu_label}] Starting with existing={existing}, "
        f"remaining={remaining}, seed={start_seed}"
    )
    result = _collect_episodes(
        remaining,
        start_seed,
        writer,
        gpu_id,
        max_attempts=max(remaining * max_attempts_multiplier, remaining),
        render_backend_prefix=render_backend_prefix,
        progress_prefix=f"[w{worker_id}/gpu{gpu_label}] ",
        collect_mode=collect_mode,
    )
    writer.finalize()
    print(
        f"[w{worker_id}/gpu{gpu_label}] Worker complete: passed={result[0]}, "
        f"attempts={result[1]}, ref_fail={result[2]}, "
        f"ctrl_dry_fail={result[3]}, ctrl_capture_fail={result[4]}"
    )
    return (worker_id, shard_dir, *result)


def _merge_shards(
    final_dir: str, shard_dirs: list[str], collect_mode: str = "insert_only"
) -> None:
    final_dir = osp.abspath(final_dir)
    for subdir in ["data", "meta", "videos"]:
        path = osp.join(final_dir, subdir)
        if osp.exists(path):
            shutil.rmtree(path)
    os.makedirs(final_dir, exist_ok=True)

    all_frames = []
    episodes = []
    task_to_idx: dict[str, int] = {}
    tasks: list[str] = []
    global_episode = 0
    global_index = 0
    for shard_dir in shard_dirs:
        paths = sorted(Path(shard_dir).glob("data/chunk-*/episode_*.parquet"))
        if not paths:
            continue
        shard_df = pd.concat(
            [pd.read_parquet(path) for path in paths], ignore_index=True
        )
        for local_episode in sorted(shard_df["episode_index"].unique()):
            local_episode = int(local_episode)
            episode_df = shard_df[shard_df["episode_index"] == local_episode].copy()
            old_task = str(episode_df["task"].iloc[0])
            if old_task not in task_to_idx:
                task_to_idx[old_task] = len(tasks)
                tasks.append(old_task)
            new_task_idx = task_to_idx[old_task]
            length = len(episode_df)
            episode_df["episode_index"] = global_episode
            episode_df["index"] = np.arange(
                global_index, global_index + length, dtype=np.int64
            )
            episode_df["task_index"] = new_task_idx
            all_frames.append(episode_df)
            episodes.append(
                {
                    "episode_index": global_episode,
                    "dataset_from_index": global_index,
                    "dataset_to_index": global_index + length,
                    "tasks": [old_task],
                    "length": length,
                    "success": True,
                }
            )

            old_chunk = local_episode // CHUNK_SIZE
            new_chunk = global_episode // CHUNK_SIZE
            for key in [
                "observation.images.top",
                "observation.images.wrist",
                "observation.images.wrist_back",
                "observation.images.render",
            ]:
                src = osp.join(
                    shard_dir,
                    "videos",
                    f"chunk-{old_chunk:03d}",
                    key,
                    f"episode_{local_episode:06d}.mp4",
                )
                dst = osp.join(
                    final_dir,
                    "videos",
                    f"chunk-{new_chunk:03d}",
                    key,
                    f"episode_{global_episode:06d}.mp4",
                )
                os.makedirs(osp.dirname(dst), exist_ok=True)
                if osp.exists(src):
                    shutil.copy2(src, dst)
            global_episode += 1
            global_index += length

    if not all_frames:
        raise RuntimeError("No successful shard data to merge.")
    merged_df = pd.concat(all_frames, ignore_index=True)
    merged_df["task"] = merged_df["task"].astype("string")
    _write_dataset_tables(
        final_dir, merged_df, episodes, tasks, collect_mode=collect_mode
    )
    print(
        f"Merged controller-domain dataset: {global_episode} episodes, "
        f"{len(merged_df)} frames -> {final_dir}"
    )


def _parse_gpu_ids(gpu_ids: str) -> list[int]:
    ids = [int(value) for value in gpu_ids.split(",") if value.strip()]
    if not ids:
        raise ValueError("--gpu-ids must contain at least one GPU id")
    return ids


def _runtime_gpu_ids(
    requested_gpu_ids: list[int], render_backend_prefix: str
) -> list[int]:
    prefix = render_backend_prefix.lower()
    if prefix not in {"", "default", "cuda", "sapien_cuda"}:
        return requested_gpu_ids

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_ids = _parse_gpu_ids(visible) if visible else []
    requested_visible = visible_ids == requested_gpu_ids
    collector_set_visible = (
        os.environ.get("RLINF_COLLECTOR_SET_CUDA_VISIBLE_DEVICES") == "1"
    )
    if collector_set_visible or requested_visible:
        return list(range(len(requested_gpu_ids)))
    return requested_gpu_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-traj", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        default="/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller",
    )
    parser.add_argument(
        "--collect-mode",
        choices=("full", "insert_only"),
        default="insert_only",
        help=(
            "full = record the whole pick-up-and-insert episode. "
            "insert_only (default) = crop to the move-and-insert segment starting "
            "at the lift-end frame (drops reach/grasp/close/lift prefix), matching "
            "convert_to_insert_only.py via the shared find_lift_end criterion."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--worker-stagger", type=float, default=5.0)
    parser.add_argument("--visualization-samples", type=int, default=10)
    parser.add_argument("--visualization-seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from completed per-episode parquet files in the output shard(s). "
            "Without this flag, an existing multi-worker _shards directory is replaced."
        ),
    )
    parser.add_argument(
        "--render-backend-prefix",
        default="cuda",
        help=(
            "Render backend selector. Use 'cuda' to match CUDA/nvidia-smi GPU ids "
            "(recommended), 'gpu' for the legacy Vulkan gpu:<id> device order, "
            "or 'cpu' for render_backend=cpu."
        ),
    )
    parser.add_argument(
        "--mp-start-method",
        choices=("spawn", "fork", "forkserver"),
        default="spawn",
        help=(
            "Multiprocessing start method. Use spawn by default because forking "
            "after importing ManiSkill/Vulkan/Torch can hang in renderer or CUDA "
            "runtime state."
        ),
    )
    parser.add_argument(
        "--max-attempts-multiplier",
        type=int,
        default=20,
        help="Per-worker max attempts is num_worker_episodes multiplied by this value.",
    )
    args = parser.parse_args()

    requested_gpu_ids = _parse_gpu_ids(args.gpu_ids)
    gpu_ids = _runtime_gpu_ids(requested_gpu_ids, args.render_backend_prefix)
    print(
        "GPU selection: "
        f"requested_physical={requested_gpu_ids}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}, "
        f"render_backend_prefix={args.render_backend_prefix!r}, "
        f"runtime_backend_ids={gpu_ids}"
    )

    output_dir = osp.abspath(args.output_dir)
    if args.num_workers <= 1:
        writer = LeRobotControllerWriter(output_dir, resume=args.resume)
        existing = len(writer.episodes)
        if existing > args.num_traj:
            raise RuntimeError(
                f"Output already has {existing} episodes, exceeding target {args.num_traj}."
            )
        remaining = args.num_traj - existing
        if remaining == 0:
            writer.finalize()
            _create_visualization_samples(
                output_dir,
                args.num_traj,
                args.visualization_samples,
                args.visualization_seed,
            )
            print(f"Already complete: {existing}/{args.num_traj} episodes in {output_dir}")
            return
        start_seed = writer.resume_seed(args.seed) if args.resume else args.seed
        result = _collect_episodes(
            remaining,
            start_seed,
            writer,
            gpu_ids[0],
            max_attempts=max(
                remaining * args.max_attempts_multiplier, remaining
            ),
            render_backend_prefix=args.render_backend_prefix,
            collect_mode=args.collect_mode,
        )
        writer.finalize()
        _create_visualization_samples(
            output_dir, args.num_traj, args.visualization_samples, args.visualization_seed
        )
        print(
            f"Done: existing={existing} newly_passed={result[0]} "
            f"attempts={result[1]} total={len(writer.episodes)} out={output_dir}"
        )
        return

    counts = [args.num_traj // args.num_workers] * args.num_workers
    for i in range(args.num_traj % args.num_workers):
        counts[i] += 1

    shard_root = osp.join(output_dir, "_shards")
    if osp.exists(shard_root) and not args.resume:
        shutil.rmtree(shard_root)
    os.makedirs(shard_root, exist_ok=True)
    worker_args = []
    gpu_counts = {gpu_id: 0 for gpu_id in gpu_ids}
    for worker_id, count in enumerate(counts):
        gpu_id = gpu_ids[worker_id % len(gpu_ids)]
        delay = gpu_counts[gpu_id] * args.worker_stagger
        gpu_counts[gpu_id] += 1
        worker_args.append(
            (
                worker_id,
                count,
                args.seed + worker_id * 100000,
                osp.join(shard_root, f"shard_{worker_id:03d}"),
                gpu_id,
                delay,
                args.max_attempts_multiplier,
                args.render_backend_prefix,
                args.collect_mode,
                args.resume,
            )
        )

    print(
        f"Launching {args.num_workers} workers across GPUs {gpu_ids}; "
        f"collecting {args.num_traj} controller-domain episodes."
        f" mp_start_method={args.mp_start_method}"
    )
    ctx = mp.get_context(args.mp_start_method)
    with ctx.Pool(args.num_workers) as pool:
        results = pool.starmap(_collect_worker, worker_args)
    print(f"Worker results: {results}")
    _merge_shards(
        output_dir, [item[3] for item in worker_args], collect_mode=args.collect_mode
    )
    _create_visualization_samples(
        output_dir, args.num_traj, args.visualization_samples, args.visualization_seed
    )


if __name__ == "__main__":
    main()
