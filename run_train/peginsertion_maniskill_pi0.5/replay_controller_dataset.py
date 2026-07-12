#!/usr/bin/env python3
"""Strict replay smoke for controller-domain PegInsertionVertical data."""

from __future__ import annotations

import argparse
import glob
import importlib.util as ilu
import json
import os
import os.path as osp
import sys
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_PATH = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

TASK_PATH = osp.join(
    REPO_PATH, "rlinf", "envs", "maniskill", "tasks", "peg_insertion_vertical.py"
)
spec = ilu.spec_from_file_location("_peg_insertion_vertical", TASK_PATH)
task_module = ilu.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(task_module)

from rlinf.envs.maniskill.peg_insertion_pi05 import model_action_to_panda_env_action


@dataclass
class ReplayStats:
    episode_index: int
    num_frames: int
    env_action_max_abs_error: float | None = None
    qpos_error_mean: float | None = None
    qpos_error_max: float | None = None
    tcp_error_mean: float | None = None
    tcp_error_max: float | None = None
    success: bool | None = None
    error: str | None = None


def _load_episode_paths(data_dir: str) -> list[str]:
    paths = sorted(glob.glob(osp.join(data_dir, "data", "chunk-*", "episode_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No per-episode parquet files found under {data_dir}")
    return paths


def _select_paths(paths: list[str], num_episodes: int, seed: int) -> list[str]:
    if len(paths) <= num_episodes:
        return paths
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), size=num_episodes, replace=False)
    return [paths[i] for i in sorted(idx)]


def _make_env(render_backend: str | None, control_mode: str):
    kwargs: dict[str, Any] = {}
    if render_backend:
        kwargs["render_backend"] = render_backend
    return gym.make(
        "PegInsertionVertical-v1",
        num_envs=1,
        obs_mode="rgb",
        robot_uids="panda_wristcam",
        control_mode=control_mode,
        sim_backend="cpu",
        reward_mode="normalized_dense",
        max_episode_steps=600,
        render_mode="all",
        sensor_configs=dict(shader_pack="default", hand_camera=dict(width=224, height=224)),
        **kwargs,
    )


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _qpos8(env) -> np.ndarray:
    qpos = _as_numpy(env.unwrapped.agent.robot.get_qpos())[0].astype(np.float32)
    state = np.zeros(8, dtype=np.float32)
    state[:7] = qpos[:7]
    state[7] = qpos[-1]
    return state


def _tcp_matrix(env) -> np.ndarray:
    tcp_pose_in_root = env.unwrapped.agent.robot.pose.inv() * env.unwrapped.agent.tcp.pose
    return tcp_pose_in_root.to_transformation_matrix().detach().cpu().numpy()[0]


def _arm_controller(env):
    controller = env.unwrapped.agent.controller
    if hasattr(controller, "controllers"):
        return controller.controllers.get("arm", controller)
    return controller


def _sync_target_delta_pose_controller(env) -> None:
    arm_controller = _arm_controller(env)
    config = getattr(arm_controller, "config", None)
    if bool(getattr(config, "use_target", False)) and hasattr(
        arm_controller, "_target_pose"
    ):
        arm_controller._target_pose = arm_controller.ee_pose_at_base


def _raw_to_env_action(action: np.ndarray, action_scale: float) -> np.ndarray:
    return model_action_to_panda_env_action(action, action_scale=action_scale)


def replay_episode(env, ep_df: pd.DataFrame, action_scale: float) -> ReplayStats:
    eid = int(ep_df["episode_index"].iloc[0])
    if "episode_reset_state" not in ep_df.columns:
        raise ValueError("Dataset lacks episode_reset_state; strict replay is impossible.")
    reset_state = np.asarray(ep_df["episode_reset_state"].iloc[0], dtype=np.float32)
    env.reset(seed=0, options={"randomize_initial_poses": False})
    env.unwrapped.set_state(torch.as_tensor(reset_state, dtype=torch.float32).reshape(1, -1))
    _sync_target_delta_pose_controller(env)

    actions = np.stack(ep_df["actions"].to_numpy()).astype(np.float32)
    expected_env_actions = (
        np.stack(ep_df["debug.env_action"].to_numpy()).astype(np.float32)
        if "debug.env_action" in ep_df.columns
        else None
    )
    qpos_targets = np.stack(ep_df["observation.state"].to_numpy()).astype(np.float32)
    tcp_after_targets = (
        np.stack(ep_df["debug.tcp_after"].to_numpy()).astype(np.float32).reshape(-1, 4, 4)
        if "debug.tcp_after" in ep_df.columns
        else None
    )

    env_action_errors = []
    qpos_errors = []
    tcp_errors = []
    for t, action in enumerate(actions):
        env_action = _raw_to_env_action(action, action_scale)
        if expected_env_actions is not None:
            env_action_errors.append(float(np.max(np.abs(env_action - expected_env_actions[t]))))
        env.step(env_action.reshape(1, -1))
        if t + 1 < len(qpos_targets):
            qpos_errors.append(float(np.linalg.norm(_qpos8(env) - qpos_targets[t + 1])))
        if tcp_after_targets is not None:
            tcp_errors.append(float(np.linalg.norm(_tcp_matrix(env)[:3, 3] - tcp_after_targets[t, :3, 3])))

    info = env.unwrapped.evaluate()
    success = bool(info["success"].detach().cpu().numpy()[0])
    return ReplayStats(
        episode_index=eid,
        num_frames=len(ep_df),
        env_action_max_abs_error=(float(max(env_action_errors)) if env_action_errors else None),
        qpos_error_mean=(float(np.mean(qpos_errors)) if qpos_errors else None),
        qpos_error_max=(float(np.max(qpos_errors)) if qpos_errors else None),
        tcp_error_mean=(float(np.mean(tcp_errors)) if tcp_errors else None),
        tcp_error_max=(float(np.max(tcp_errors)) if tcp_errors else None),
        success=success,
    )


def _aggregate(results: list[ReplayStats]) -> dict[str, Any]:
    ok = [result for result in results if result.error is None]
    summary: dict[str, Any] = {
        "num_episodes": len(results),
        "num_ok": len(ok),
        "num_errors": len(results) - len(ok),
    }
    for field in [
        "env_action_max_abs_error",
        "qpos_error_mean",
        "qpos_error_max",
        "tcp_error_mean",
        "tcp_error_max",
    ]:
        vals = [getattr(result, field) for result in ok if getattr(result, field) is not None]
        if vals:
            summary[field] = {
                "mean": float(np.mean(vals)),
                "max": float(np.max(vals)),
                "p90": float(np.percentile(vals, 90)),
            }
    successes = [result.success for result in ok if result.success is not None]
    if successes:
        summary["env_success_rate"] = float(np.mean(successes))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller",
    )
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-backend", default=None)
    parser.add_argument("--control-mode", default="pd_ee_target_delta_pose")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    paths = _select_paths(_load_episode_paths(args.data_dir), args.num_episodes, args.seed)
    env = _make_env(args.render_backend, args.control_mode)
    results: list[ReplayStats] = []
    try:
        for path in tqdm(paths, desc="Controller replay"):
            ep_df = pd.read_parquet(path).reset_index(drop=True)
            try:
                results.append(replay_episode(env, ep_df, args.action_scale))
            except Exception as exc:
                eid = int(ep_df["episode_index"].iloc[0]) if len(ep_df) else -1
                results.append(ReplayStats(episode_index=eid, num_frames=len(ep_df), error=repr(exc)))
    finally:
        env.close()

    report = {
        "data_dir": osp.abspath(args.data_dir),
        "action_scale": args.action_scale,
        "control_mode": args.control_mode,
        "summary": _aggregate(results),
        "episodes": [asdict(result) for result in results],
        "notes": [
            "This strict replay requires controller-domain data with episode_reset_state.",
            "For correct data, action_scale should be 1.0 and env_action_max_abs_error should be near zero.",
            "PegInsertion pi0.5 controller-domain data uses pd_ee_target_delta_pose.",
        ],
    }
    output_json = args.output_json or osp.join(args.data_dir, "meta", "controller_replay_smoke.json")
    os.makedirs(osp.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
