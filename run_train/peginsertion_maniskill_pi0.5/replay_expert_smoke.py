#!/usr/bin/env python3
"""Replay-smoke checks for converted PegInsertionVertical expert data.

The converted LeRobot dataset stores robot qpos and expert delta-pose actions,
but not the sampled peg/hole object poses.  Therefore this script provides two
checks:

1. FK consistency: deterministic, no physics stepping.  It verifies that each
   stored action matches the TCP transform change implied by adjacent stored
   qpos frames.
2. Open-loop env step: resets a fresh environment, sets the robot to the first
   qpos, executes stored actions with ``pd_ee_delta_pose``, and measures TCP
   tracking drift.  Task success is reported when available, but should not be
   treated as exact dataset replay because object poses are not stored.
"""

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
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles
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

from rlinf.envs.maniskill.peg_insertion_pi05 import qpos8_to_robot_qpos


ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


@dataclass
class EpisodeReplayStats:
    episode_index: int
    num_frames: int
    fk_action_max_abs_error: float | None = None
    fk_action_p99_abs_error: float | None = None
    fk_position_cosine_mean: float | None = None
    fk_position_error_mean: float | None = None
    fk_position_error_max: float | None = None
    open_loop_position_error_mean: float | None = None
    open_loop_position_error_max: float | None = None
    open_loop_rotation_error_mean: float | None = None
    open_loop_rotation_error_max: float | None = None
    open_loop_actual_delta_ratio_mean: float | None = None
    open_loop_actual_delta_cosine_mean: float | None = None
    env_success: bool | None = None
    env_inserted: bool | None = None
    env_grasped: bool | None = None
    error: str | None = None


def _load_episode_paths(data_dir: str) -> list[str]:
    paths = sorted(glob.glob(osp.join(data_dir, "data", "chunk-*", "episode_*.parquet")))
    single = osp.join(data_dir, "data", "chunk-000", "file-000.parquet")
    if paths:
        return paths
    if osp.exists(single):
        return [single]
    raise FileNotFoundError(f"No LeRobot parquet files found under {data_dir}")


def _select_episodes(paths: list[str], episode_ids: list[int] | None, num_episodes: int, seed: int):
    if episode_ids:
        id_set = set(episode_ids)
        selected = []
        for path in paths:
            if "episode_" in osp.basename(path):
                eid = int(osp.basename(path).split("_")[-1].split(".")[0])
                if eid in id_set:
                    selected.append(path)
            else:
                selected.append(path)
        return selected
    rng = np.random.default_rng(seed)
    if len(paths) <= num_episodes:
        return paths
    idx = rng.choice(len(paths), size=num_episodes, replace=False)
    return [paths[i] for i in sorted(idx)]


def _episode_groups(path: str, episode_ids: list[int] | None):
    df = pd.read_parquet(path)
    if episode_ids is not None:
        df = df[df["episode_index"].isin(episode_ids)]
    for eid in sorted(df["episode_index"].unique()):
        yield int(eid), df[df["episode_index"] == eid].reset_index(drop=True)


def _make_env(control_mode: str, render_backend: str | None, obs_mode: str = "state"):
    kwargs: dict[str, Any] = {}
    if render_backend:
        kwargs["render_backend"] = render_backend
    return gym.make(
        "PegInsertionVertical-v1",
        num_envs=1,
        obs_mode=obs_mode,
        robot_uids="panda_wristcam",
        control_mode=control_mode,
        sim_backend="cpu",
        reward_mode="normalized_dense",
        max_episode_steps=600,
        render_mode="all",
        sensor_configs=dict(shader_pack="default"),
        **kwargs,
    )


def _set_robot_qpos(unwrapped, qpos8: np.ndarray) -> None:
    qpos = qpos8_to_robot_qpos(qpos8)
    unwrapped.agent.robot.set_qpos(torch.as_tensor(qpos).reshape(1, -1))


def _tcp_matrix_in_robot_root(unwrapped) -> np.ndarray:
    tcp_pose_in_root = unwrapped.agent.robot.pose.inv() * unwrapped.agent.tcp.pose
    return tcp_pose_in_root.to_transformation_matrix().detach().cpu().numpy()[0]


def _tcp_mats_from_qpos(env, ep_df: pd.DataFrame) -> np.ndarray:
    uw = env.unwrapped
    mats = np.zeros((len(ep_df), 4, 4), dtype=np.float64)
    for i, qpos8 in enumerate(ep_df["observation.state"]):
        _set_robot_qpos(uw, np.asarray(qpos8, dtype=np.float32))
        mats[i] = _tcp_matrix_in_robot_root(uw)
    return mats


def _euler_xyz_delta_from_mats(tcp_mats: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    actions = np.zeros((len(tcp_mats), 7), dtype=np.float32)
    for t in range(len(tcp_mats) - 1):
        dp = tcp_mats[t + 1, :3, 3] - tcp_mats[t, :3, 3]
        r_delta = tcp_mats[t + 1, :3, :3] @ tcp_mats[t, :3, :3].T
        dr = matrix_to_euler_angles(
            torch.as_tensor(r_delta, dtype=torch.float64).unsqueeze(0), "XYZ"
        )[0].numpy()
        actions[t] = np.concatenate([dp, dr, [float(gripper[t])]])
    if len(tcp_mats) > 1:
        actions[-1] = actions[-2]
    return actions


def _rotation_angle_error(r_a: np.ndarray, r_b: np.ndarray) -> float:
    r_delta = r_a @ r_b.T
    trace = np.trace(r_delta)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos))


def _normalize_panda_dpose_action(action: np.ndarray) -> np.ndarray:
    normalized = np.asarray(action, dtype=np.float32).copy()
    normalized[:6] /= 0.1
    return normalized


def run_fk_check(env, eid: int, ep_df: pd.DataFrame) -> EpisodeReplayStats:
    actions = np.stack(ep_df["actions"].to_numpy()).astype(np.float32)
    tcp_mats = _tcp_mats_from_qpos(env, ep_df)
    recomputed = _euler_xyz_delta_from_mats(tcp_mats, actions[:, 6])
    diff = actions - recomputed
    pos_errors = np.linalg.norm(diff[:, :3], axis=1)
    pos_norm_a = np.linalg.norm(actions[:, :3], axis=1)
    pos_norm_b = np.linalg.norm(recomputed[:, :3], axis=1)
    denom = np.maximum(pos_norm_a * pos_norm_b, 1e-9)
    cosine = np.sum(actions[:, :3] * recomputed[:, :3], axis=1) / denom
    moving = (pos_norm_a > 1e-6) & (pos_norm_b > 1e-6)
    return EpisodeReplayStats(
        episode_index=eid,
        num_frames=len(ep_df),
        fk_action_max_abs_error=float(np.abs(diff).max()),
        fk_action_p99_abs_error=float(np.percentile(np.abs(diff), 99)),
        fk_position_cosine_mean=float(np.mean(cosine[moving])) if moving.any() else None,
        fk_position_error_mean=float(np.mean(pos_errors)),
        fk_position_error_max=float(np.max(pos_errors)),
    )


def run_open_loop(
    env,
    eid: int,
    ep_df: pd.DataFrame,
    max_steps: int | None,
    render_backend: str | None,
    normalize_panda_dpose: bool,
    action_scale: float,
) -> EpisodeReplayStats:
    actions = np.stack(ep_df["actions"].to_numpy()).astype(np.float32)
    steps = min(len(actions) - 1, max_steps or (len(actions) - 1))
    target_env = _make_env("pd_joint_pos", render_backend, obs_mode="state")
    try:
        target_env.reset(seed=0, options={"randomize_initial_poses": False})
        target_mats = _tcp_mats_from_qpos(target_env, ep_df.iloc[: steps + 1])
    finally:
        target_env.close()

    env.reset(seed=0, options={"randomize_initial_poses": False})
    _set_robot_qpos(env.unwrapped, np.asarray(ep_df["observation.state"].iloc[0], dtype=np.float32))

    pos_errors: list[float] = []
    rot_errors: list[float] = []
    delta_ratios: list[float] = []
    delta_cosines: list[float] = []
    prev = _tcp_matrix_in_robot_root(env.unwrapped)
    for t in range(steps):
        env_action = (
            _normalize_panda_dpose_action(actions[t])
            if normalize_panda_dpose
            else actions[t]
        )
        env_action[:6] *= action_scale
        obs, reward, terminated, truncated, info = env.step(env_action.reshape(1, -1))
        del obs, reward, terminated, truncated, info
        current = _tcp_matrix_in_robot_root(env.unwrapped)
        target = target_mats[t + 1]
        pos_errors.append(float(np.linalg.norm(current[:3, 3] - target[:3, 3])))
        rot_errors.append(_rotation_angle_error(current[:3, :3], target[:3, :3]))
        target_delta = target[:3, 3] - target_mats[t, :3, 3]
        actual_delta = current[:3, 3] - prev[:3, 3]
        target_norm = float(np.linalg.norm(target_delta))
        actual_norm = float(np.linalg.norm(actual_delta))
        if target_norm > 1e-8:
            delta_ratios.append(actual_norm / target_norm)
            delta_cosines.append(
                float(np.dot(actual_delta, target_delta) / (actual_norm * target_norm + 1e-9))
            )
        prev = current

    eval_info = env.unwrapped.evaluate()
    success = bool(eval_info["success"].detach().cpu().numpy()[0])
    inserted = success
    grasped = None
    try:
        grasped = bool(env.unwrapped.agent.is_grasping(env.unwrapped.peg, max_angle=30).detach().cpu().numpy()[0])
    except Exception:
        pass
    return EpisodeReplayStats(
        episode_index=eid,
        num_frames=len(ep_df),
        open_loop_position_error_mean=float(np.mean(pos_errors)),
        open_loop_position_error_max=float(np.max(pos_errors)),
        open_loop_rotation_error_mean=float(np.mean(rot_errors)),
        open_loop_rotation_error_max=float(np.max(rot_errors)),
        open_loop_actual_delta_ratio_mean=(
            float(np.mean(delta_ratios)) if delta_ratios else None
        ),
        open_loop_actual_delta_cosine_mean=(
            float(np.mean(delta_cosines)) if delta_cosines else None
        ),
        env_success=success,
        env_inserted=inserted,
        env_grasped=grasped,
    )


def _merge_stats(base: EpisodeReplayStats, extra: EpisodeReplayStats) -> EpisodeReplayStats:
    data = asdict(base)
    for key, value in asdict(extra).items():
        if value is not None:
            data[key] = value
    return EpisodeReplayStats(**data)


def _aggregate(results: list[EpisodeReplayStats]) -> dict[str, Any]:
    ok = [r for r in results if r.error is None]
    summary: dict[str, Any] = {
        "num_episodes": len(results),
        "num_ok": len(ok),
        "num_errors": len(results) - len(ok),
    }
    for field in [
        "fk_action_max_abs_error",
        "fk_action_p99_abs_error",
        "fk_position_cosine_mean",
        "fk_position_error_mean",
        "fk_position_error_max",
        "open_loop_position_error_mean",
        "open_loop_position_error_max",
        "open_loop_rotation_error_mean",
        "open_loop_rotation_error_max",
        "open_loop_actual_delta_ratio_mean",
        "open_loop_actual_delta_cosine_mean",
    ]:
        vals = [getattr(r, field) for r in ok if getattr(r, field) is not None]
        if vals:
            summary[field] = {
                "mean": float(np.mean(vals)),
                "max": float(np.max(vals)),
                "p50": float(np.percentile(vals, 50)),
                "p90": float(np.percentile(vals, 90)),
            }
    successes = [r.env_success for r in ok if r.env_success is not None]
    if successes:
        summary["env_success_rate"] = float(np.mean(successes))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_3200",
    )
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--episode-ids", default="", help="Comma-separated episode ids to check.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["fk", "open-loop", "both"], default="fk")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--render-backend", default=None, help="For example gpu:0.")
    parser.add_argument(
        "--control-mode",
        choices=["pd_ee_delta_pose", "pd_ee_target_delta_pose"],
        default="pd_ee_delta_pose",
        help="Controller used for open-loop replay. Target-delta accumulates commands in the controller target pose.",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help="Scale applied to the first 6 action dimensions after optional normalization.",
    )
    parser.add_argument(
        "--no-normalize-panda-dpose",
        action="store_true",
        help="Send raw deltas directly to pd_ee_delta_pose. This reproduces the old behavior and is useful only as a negative-control check.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Defaults to <data-dir>/meta/replay_expert_smoke.json.",
    )
    args = parser.parse_args()

    episode_ids = (
        [int(x) for x in args.episode_ids.split(",") if x.strip()]
        if args.episode_ids
        else None
    )
    paths = _select_episodes(
        _load_episode_paths(args.data_dir), episode_ids, args.num_episodes, args.seed
    )
    output_json = args.output_json or osp.join(
        args.data_dir, "meta", "replay_expert_smoke.json"
    )

    fk_env = None
    step_env = None
    results: list[EpisodeReplayStats] = []
    try:
        if args.mode in ["fk", "both"]:
            fk_env = _make_env("pd_joint_pos", args.render_backend, obs_mode="state")
            fk_env.reset(seed=0, options={"randomize_initial_poses": False})
        if args.mode in ["open-loop", "both"]:
            step_env = _make_env(args.control_mode, args.render_backend, obs_mode="state")

        processed = 0
        target_count = args.num_episodes if episode_ids is None else len(episode_ids)
        for path in tqdm(paths, desc="Replay smoke"):
            for eid, ep_df in _episode_groups(path, episode_ids):
                if episode_ids is None and processed >= target_count:
                    break
                try:
                    stats = EpisodeReplayStats(episode_index=eid, num_frames=len(ep_df))
                    if fk_env is not None:
                        stats = _merge_stats(stats, run_fk_check(fk_env, eid, ep_df))
                    if step_env is not None:
                        stats = _merge_stats(
                            stats,
                            run_open_loop(
                                step_env,
                                eid,
                                ep_df,
                                args.max_steps,
                                args.render_backend,
                                not args.no_normalize_panda_dpose,
                                args.action_scale,
                            ),
                        )
                    results.append(stats)
                except Exception as exc:
                    results.append(
                        EpisodeReplayStats(
                            episode_index=eid,
                            num_frames=len(ep_df),
                            error=repr(exc),
                        )
                    )
                processed += 1
            if episode_ids is None and processed >= target_count:
                break
    finally:
        if fk_env is not None:
            fk_env.close()
        if step_env is not None:
            step_env.close()

    report = {
        "data_dir": osp.abspath(args.data_dir),
        "mode": args.mode,
        "control_mode": args.control_mode,
        "action_scale": args.action_scale,
        "normalize_panda_dpose": not args.no_normalize_panda_dpose,
        "num_requested": args.num_episodes if episode_ids is None else len(episode_ids),
        "summary": _aggregate(results),
        "episodes": [asdict(r) for r in results],
        "notes": [
            "FK consistency should have near-zero action errors; non-zero errors indicate conversion/action mismatch.",
            "Open-loop uses fresh object poses because the dataset does not store peg/hole poses; use TCP tracking drift as the main signal.",
        ],
    }
    os.makedirs(osp.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
