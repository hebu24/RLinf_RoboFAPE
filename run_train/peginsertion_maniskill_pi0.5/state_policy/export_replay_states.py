#!/usr/bin/env python3
"""Replay compact episodes and export aligned peg/hole states."""

from __future__ import annotations

import argparse
import glob
import importlib.util as ilu
import json
import os
import os.path as osp
import shutil
import sys
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_PATH = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

TASK_PATH = osp.join(REPO_PATH, "rlinf", "envs", "maniskill", "tasks", "peg_insertion_vertical.py")
spec = ilu.spec_from_file_location("_peg_insertion_vertical", TASK_PATH)
task_module = ilu.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(task_module)

from rlinf.envs.maniskill.peg_insertion_pi05 import model_action_to_panda_env_action


@dataclass
class ExportEpisodeReport:
    compact_episode_index: int
    source_episode_index: int | None = None
    compact_num_frames: int = 0
    replay_qpos_error_mean: float | None = None
    replay_qpos_error_max: float | None = None
    success: bool | None = None
    accepted: bool = False
    output_file: str | None = None
    error: str | None = None


def _load_episode_paths(data_dir: str) -> list[str]:
    paths = sorted(glob.glob(osp.join(data_dir, "data", "chunk-*", "episode_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No per-episode parquet files found under {data_dir}")
    return paths


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _qpos8(env) -> np.ndarray:
    qpos = _as_numpy(env.unwrapped.agent.robot.get_qpos())[0].astype(np.float32)
    state = np.zeros(8, dtype=np.float32)
    state[:7] = qpos[:7]
    state[7] = qpos[-1]
    return state


def _qpos8_to_robot_qpos9(qpos8: np.ndarray) -> np.ndarray:
    qpos8 = np.asarray(qpos8, dtype=np.float32).reshape(-1)
    if qpos8.shape[0] != 8:
        raise ValueError(f"Expected compact qpos dim 8, got shape={qpos8.shape}")
    finger = float(qpos8[7])
    return np.array([*qpos8[:7], finger, finger], dtype=np.float32)


def _make_env(render_backend: str | None, control_mode: str):
    kwargs: dict[str, Any] = {}
    if render_backend:
        kwargs["render_backend"] = render_backend
    return gym.make("PegInsertionVertical-v1", num_envs=1, obs_mode="state_dict", robot_uids="panda_wristcam", control_mode=control_mode, sim_backend="cpu", reward_mode="normalized_dense", max_episode_steps=600, **kwargs)


def _arm_controller(env):
    controller = env.unwrapped.agent.controller
    if hasattr(controller, "controllers"):
        return controller.controllers.get("arm", controller)
    return controller


def _sync_target_delta_pose_controller(env) -> None:
    arm_controller = _arm_controller(env)
    config = getattr(arm_controller, "config", None)
    if bool(getattr(config, "use_target", False)) and hasattr(arm_controller, "_target_pose"):
        arm_controller._target_pose = arm_controller.ee_pose_at_base


def _raw_to_env_action(action: np.ndarray, action_scale: float) -> np.ndarray:
    return model_action_to_panda_env_action(action, action_scale=action_scale)


def _pose_raw_1d(raw_pose: Any) -> np.ndarray:
    pose = _as_numpy(raw_pose).astype(np.float32)
    if pose.ndim == 2:
        return pose[0]
    return pose


def _refresh_articulation_kinematics_if_possible(env) -> None:
    scene = env.unwrapped.scene
    px = getattr(scene, "px", None)
    for fn_name in ["cpu_update_articulation_kinematics", "update_articulation_kinematics"]:
        for obj in [px, scene]:
            fn = getattr(obj, fn_name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    continue


def _init_compact_episode_state(env, compact_df: pd.DataFrame) -> None:
    compact_reset = np.array(compact_df["episode_reset_state"].iloc[0], dtype=np.float32, copy=True)
    compact_qpos = np.stack(compact_df["observation.state"].to_numpy()).astype(np.float32)
    env.reset(seed=0, options={"randomize_initial_poses": False})
    env.unwrapped.set_state(torch.as_tensor(compact_reset, dtype=torch.float32).reshape(1, -1))
    first_robot_qpos = _qpos8_to_robot_qpos9(compact_qpos[0])
    env.unwrapped.agent.robot.set_qpos(torch.as_tensor(first_robot_qpos, dtype=torch.float32).reshape(1, -1))
    _refresh_articulation_kinematics_if_possible(env)
    peg_pose = env.unwrapped.agent.tcp.pose * task_module.sapien.Pose([0.06, 0, 0])
    env.unwrapped.peg.set_pose(peg_pose)
    _sync_target_delta_pose_controller(env)


def _evaluate_success(env) -> bool:
    info = env.unwrapped.evaluate()
    success_raw = info.get("success", False)
    success_np = _as_numpy(success_raw).reshape(-1)
    return bool(success_np[0]) if success_np.size else bool(success_raw)



def _replay_and_collect(env, compact_df: pd.DataFrame, action_scale: float) -> tuple[pd.DataFrame, list[float], bool]:
    compact_actions = np.stack(compact_df["actions"].to_numpy()).astype(np.float32)
    compact_qpos = np.stack(compact_df["observation.state"].to_numpy()).astype(np.float32)
    _init_compact_episode_state(env, compact_df)
    rows: list[dict[str, Any]] = []
    qpos_errors: list[float] = []
    compact_episode_index = int(compact_df["episode_index"].iloc[0])
    for frame_index, action in enumerate(compact_actions):
        replay_qpos_before = _qpos8(env)
        compact_q = compact_qpos[frame_index]
        row = {
            "episode_index": compact_episode_index,
            "source_episode_index": compact_episode_index,
            "frame_index": frame_index,
            "suffix_start_index": 0,
            "actions": action.astype(np.float32),
            "compact_qpos": compact_q.astype(np.float32),
            "source_qpos": compact_q.astype(np.float32),
            "replay_qpos": replay_qpos_before.astype(np.float32),
            "replay_qpos_l2_error": np.nan,
            "peg_pose": _pose_raw_1d(env.unwrapped.peg.pose.raw_pose),
            "peg_head_pose": _pose_raw_1d(env.unwrapped.peg_head_pose.raw_pose),
            "box_hole_pose": _pose_raw_1d(env.unwrapped.box_hole_pose.raw_pose),
        }
        env_action = _raw_to_env_action(action, action_scale)
        env.step(env_action.reshape(1, -1))
        if frame_index + 1 < len(compact_qpos):
            next_qpos_error = float(np.linalg.norm(_qpos8(env) - compact_qpos[frame_index + 1]))
            row["replay_qpos_l2_error"] = next_qpos_error
            qpos_errors.append(next_qpos_error)
        rows.append(row)
    success = _evaluate_success(env)
    return pd.DataFrame(rows), qpos_errors, success


def _prepare_output_dir(output_dir: str, compact_dir: str, source_dir: str | None, overwrite: bool) -> str:
    out_abs = osp.abspath(output_dir)
    compact_abs = osp.abspath(compact_dir)
    occupied = {compact_abs}
    if source_dir:
        occupied.add(osp.abspath(source_dir))
    if out_abs in occupied:
        raise ValueError("output-dir must be independent from compact/source input dirs")
    if osp.exists(out_abs):
        if not overwrite and any(os.scandir(out_abs)):
            raise FileExistsError(f"Output dir is not empty: {out_abs}. Use --overwrite to replace it.")
        if overwrite:
            shutil.rmtree(out_abs)
    os.makedirs(osp.join(out_abs, "data"), exist_ok=True)
    return out_abs


def _aggregate_summary(reports: list[ExportEpisodeReport]) -> dict[str, Any]:
    accepted = [r for r in reports if r.accepted]
    replay_errors = [r.replay_qpos_error_max for r in accepted if r.replay_qpos_error_max is not None]
    return {
        "num_episodes": len(reports),
        "num_accepted": len(accepted),
        "num_rejected": len(reports) - len(accepted),
        "accepted_rate": float(len(accepted) / len(reports)) if reports else 0.0,
        "num_success": len([r for r in reports if r.success]),
        "success_rate": float(len([r for r in reports if r.success]) / len(reports)) if reports else 0.0,
        "replay_qpos_error_max_mean": float(np.mean(replay_errors)) if replay_errors else None,
        "replay_qpos_error_max_p90": float(np.percentile(replay_errors, 90)) if replay_errors else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-dir", default=("/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/" "peg_insertion_vertical_dualwrist_insert_only_compact_3200"))
    parser.add_argument("--source-dir", default=None, help="Deprecated compatibility argument. This script ignores source episodes.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--qpos-error-threshold", type=float, default=0.05)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--control-mode", default="pd_ee_target_delta_pose")
    parser.add_argument("--render-backend", default=None)
    args = parser.parse_args()

    output_dir = _prepare_output_dir(args.output_dir, args.compact_dir, args.source_dir, args.overwrite)

    if args.source_dir:
        print(f"[info] --source-dir is ignored: {osp.abspath(args.source_dir)}")

    compact_paths = _load_episode_paths(args.compact_dir)
    if args.num_episodes > 0:
        compact_paths = compact_paths[: args.num_episodes]

    env = _make_env(args.render_backend, args.control_mode)
    reports: list[ExportEpisodeReport] = []
    try:
        for compact_path in tqdm(compact_paths, desc="Replay export"):
            compact_df = pd.read_parquet(compact_path, columns=["episode_index", "episode_reset_state", "actions", "observation.state"]).reset_index(drop=True)
            compact_episode_index = int(compact_df["episode_index"].iloc[0])
            report = ExportEpisodeReport(compact_episode_index=compact_episode_index, compact_num_frames=len(compact_df))
            try:
                exported_df, qpos_errors, success = _replay_and_collect(env, compact_df, args.action_scale)
                replay_qpos_error_max = float(np.max(qpos_errors)) if qpos_errors else 0.0
                replay_qpos_error_mean = float(np.mean(qpos_errors)) if qpos_errors else 0.0
                report.replay_qpos_error_mean = replay_qpos_error_mean
                report.replay_qpos_error_max = replay_qpos_error_max
                report.success = success
                report.accepted = bool(replay_qpos_error_max <= args.qpos_error_threshold)
                out_rel = osp.join("data", f"episode_{compact_episode_index:06d}.parquet")
                exported_df.to_parquet(osp.join(output_dir, out_rel), index=False)
                report.output_file = out_rel
            except Exception as exc:
                report.error = repr(exc)

            reports.append(report)
    finally:
        env.close()

    metadata = {
        "compact_dir": osp.abspath(args.compact_dir),
        "source_dir": osp.abspath(args.source_dir) if args.source_dir else None,
        "source_dir_ignored": bool(args.source_dir),
        "output_dir": output_dir,
        "num_episodes_requested": args.num_episodes,
        "control_mode": args.control_mode,
        "action_scale": args.action_scale,
        "qpos_error_threshold": args.qpos_error_threshold,
        "initialization": "compact_first_qpos_rigid_grasp",
        "summary": _aggregate_summary(reports),
        "manifest": [asdict(report) for report in reports],
    }

    metadata_path = osp.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata["summary"], indent=2))
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
