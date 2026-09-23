#!/usr/bin/env python3
"""Collect RoboFPE/ManiSkill successful trajectories and export render/wrist SFT data.

The default path is PushCube-v1 with panda_wristcam, producing a LeRobot-style
OpenPI dataset with observation.images.top (render_camera) and
observation.images.wrist (hand_camera).
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _configure_local_maniskill_assets() -> None:
    if os.environ.get("MS_ASSET_DIR"):
        return
    candidates = (
        "/data/yingxi/robofac",
        "/data/yingxi/maniskill_assets",
    )
    for candidate in candidates:
        replica_cad = osp.join(
            candidate, "data", "scene_datasets", "replica_cad_dataset",
            "configs", "scenes", "apt_1.scene_instance.json",
        )
        if osp.exists(replica_cad):
            os.environ["MS_ASSET_DIR"] = candidate
            return


_configure_local_maniskill_assets()

import numpy as np
from tqdm import tqdm

ROBOFPE_ROOT = os.environ.get("ROBOFPE_ROOT", "/data/yingxi/RoboFPE")
ROBOFPE_MANI_ENVS = osp.join(ROBOFPE_ROOT, "mani_envs")
ROBOFPE_DC = osp.join(ROBOFPE_MANI_ENVS, "data_collection")
ROBOFPE_RUN = osp.join(ROBOFPE_DC, "run")
ROBOFPE_COLLECT = osp.join(ROBOFPE_DC, "collect")
ROBOFPE_UTILS = osp.join(ROBOFPE_DC, "utils")
for _path in [ROBOFPE_ROOT, ROBOFPE_MANI_ENVS, ROBOFPE_DC, ROBOFPE_RUN, ROBOFPE_COLLECT, ROBOFPE_UTILS]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: F401,E402
# Prefer RoboFPE's task implementation over ManiSkill's same-named default.
from tasks.task_PlugCharger import PlugChargerEnv as _RoboFPEPlugChargerEnv  # noqa: F401,E402
from mani_skill.trajectory import utils as trajectory_utils  # noqa: E402
from mani_skill.utils.wrappers.record import RecordEpisode  # noqa: E402
from mani_skill.utils import common  # noqa: E402

from collect_progress import _compatible_reward_mode  # noqa: E402
from render_domain_randomization import (  # noqa: E402
    apply_render_randomization,
    human_render_camera_overrides,
    render_randomization_metadata,
    sample_render_randomization_spec,
)
from utils.trajectory_limits import (  # noqa: E402
    CollectionEpisodeTimeout,
    CollectionStepLimit,
    resolve_max_episode_steps,
)

FPS = 30
IMAGE_SIZE = 224
CHUNK_SIZE = 1000
DEFAULT_ROBOT_UIDS = "panda_wristcam"
BASE_CAMERA = "base_camera"
WRIST_CAMERA = "hand_camera"
RENDER_CAMERA = "render_camera"
SEED_STRIDE = 1_000_000


class RecordTcpPose(gym.Wrapper):
    """Ensure every recorded observation contains the current TCP pose."""

    def _with_tcp_pose(self, obs):
        if not isinstance(obs, dict):
            raise TypeError(f"TCP recording requires dict observations, got {type(obs)!r}")
        obs = dict(obs)
        extra = dict(obs.get("extra") or {})
        extra["tcp_pose"] = self.unwrapped.agent.tcp.pose.raw_pose
        obs["extra"] = extra
        return obs

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return self._with_tcp_pose(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._with_tcp_pose(obs), reward, terminated, truncated, info


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    max_episode_steps: int
    solver_timeout: float = 120.0
    supports_insert_only: bool = False
    top_camera: str = RENDER_CAMERA
    wrist_camera: str = WRIST_CAMERA




def _load_solution(task_id: str):
    if task_id == "PushCube-v1":
        from mani_skill.examples.motionplanning.panda.solutions import solvePushCube
        return solvePushCube
    if task_id == "StackCube-v1":
        try:
            from error_solutions.StackCube_v1 import solve_with_errors
            return solve_with_errors
        except Exception:
            from mani_skill.examples.motionplanning.panda.solutions import solveStackCube
            return solveStackCube
    if task_id == "PlugCharger-v1":
        from solutions.solve_PlugCharger import solve
        return solve
    if task_id == "PegInsertionVertical-v1":
        from solutions.solve_PegInsertionVertical import solve_peginsertionvertical
        return solve_peginsertionvertical
    if task_id == "PegInsertionSide-v1":
        from mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side import solve
        return solve
    if task_id == "PullCubeTool-golf":
        from solutions.solve_PullCubeTool import solve_pullcubetool_golf
        return solve_pullcubetool_golf
    if task_id == "UprightStack-v1":
        from solutions.solve_UprightStack import solveUprightStack
        return solveUprightStack
    if task_id == "LiftPegUpright-box":
        from solutions.solve_LiftPegUpright import solve_liftpegupright_box
        return solve_liftpegupright_box
    if task_id == "PickCube-ball":
        from solutions.solve_PickCube import solve_pickcube_ball
        return solve_pickcube_ball
    if task_id == "PullCube-block":
        from solutions.solve_PullCube import solve_pullcube_block
        return solve_pullcube_block
    raise KeyError(task_id)


class SolverTimeout(RuntimeError):
    pass

def _solver_alarm_handler(signum, frame):
    raise SolverTimeout("motion planner timed out")

def run_solution(solve, env, seed, debug, vis, reset_options=None, timeout_seconds=None):
    import inspect
    kwargs = dict(seed=seed, debug=debug, vis=vis)
    if reset_options is not None:
        try:
            if "reset_options" in inspect.signature(solve).parameters:
                kwargs["reset_options"] = reset_options
        except (TypeError, ValueError):
            pass
    previous = signal.signal(signal.SIGALRM, _solver_alarm_handler)
    if timeout_seconds:
        signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    try:
        return solve(env, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def unwrap_solution_result(res):
    if isinstance(res, dict):
        return res, list(res.get("stage_records", []))
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], list):
        return res[0], res[1]
    return res, []


def solution_result_stats(res):
    if isinstance(res, dict):
        elapsed = max((r.get("elapsed_steps", 0) for r in res.get("stage_records", [])), default=0)
        return bool(res.get("final_success")), int(elapsed)
    return bool(res[-1]["success"].item()), int(res[-1]["elapsed_steps"].item())


def environment_success(env) -> bool:
    """Read the task's final success state instead of trusting planner return types."""
    result = env.unwrapped.evaluate()
    success = common.to_numpy(result["success"])
    return bool(np.all(np.asarray(success)))


def annotate_episode_metadata(h5_path, episode_metadata):
    json_path = h5_path[:-3] + ".json" if h5_path.endswith(".h5") else f"{h5_path}.json"
    if not osp.exists(json_path):
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes")
    if isinstance(episodes, list):
        for episode, metadata in zip(episodes, episode_metadata, strict=False):
            episode.update(metadata)
    else:
        data["episodes_metadata"] = episode_metadata
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


TASKS: dict[str, TaskSpec] = {
    "PushCube-v1": TaskSpec(
        task_id="PushCube-v1",
        prompt="Push the cube across the tabletop until its center lies inside the target region.",
        max_episode_steps=180,
    ),
    "StackCube-v1": TaskSpec(
        task_id="StackCube-v1",
        prompt="Stack the red cube on top of the green cube.",
        max_episode_steps=300,
    ),
    "PlugCharger-v1": TaskSpec(
        task_id="PlugCharger-v1",
        prompt="Insert the charger into the receptacle.",
        max_episode_steps=350,
    ),
    "PegInsertionVertical-v1": TaskSpec(
        task_id="PegInsertionVertical-v1",
        prompt="Insert the peg vertically into the target hole.",
        max_episode_steps=450,
        supports_insert_only=True,
    ),
    "PegInsertionSide-v1": TaskSpec(
        task_id="PegInsertionSide-v1",
        prompt="Insert the peg into the side-facing target hole.",
        max_episode_steps=350,
        supports_insert_only=True,
    ),
    "PullCubeTool-golf": TaskSpec(
        task_id="PullCubeTool-golf",
        prompt="Use the L-shaped tool to pull the golf ball into the robot's reachable target region.",
        max_episode_steps=350,
    ),
    "UprightStack-v1": TaskSpec(
        task_id="UprightStack-v1",
        prompt="Stand the brick upright and stack it on the red cube.",
        max_episode_steps=1000,
        solver_timeout=300.0,
    ),
    "LiftPegUpright-box": TaskSpec(
        task_id="LiftPegUpright-box",
        prompt="Stand the cracker box upright on the table.",
        max_episode_steps=350,
    ),
    "PickCube-ball": TaskSpec(
        task_id="PickCube-ball",
        prompt="Pick up the tennis ball and place it at the target.",
        max_episode_steps=200,
    ),
    "PullCube-block": TaskSpec(
        task_id="PullCube-block",
        prompt="Pull the wooden block into the target region.",
        max_episode_steps=200,
    ),
}


def _stable_seed(token: Any) -> int:
    import hashlib

    digest = hashlib.sha256(str(token).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _look_at_pose(eye, target):
    from mani_skill.utils import sapien_utils

    pose = sapien_utils.look_at(eye, target)
    p = np.asarray(pose.p, dtype=np.float64).reshape(-1)
    q = np.asarray(pose.q, dtype=np.float64).reshape(-1)
    return [float(x) for x in p[:3]] + [float(x) for x in q[:4]]


def _sample_camera_spec(task_id: str, seed_token: str, *, extreme: bool = False) -> dict[str, Any]:
    rng = np.random.default_rng(_stable_seed(seed_token))
    is_pull_block = task_id == "PullCube-block"
    is_plug_charger = task_id == "PlugCharger-v1"
    vertical_render_pose = None
    if task_id == "PegInsertionVertical-v1":
        # Keep the task's calibrated human view.  In particular, do not use
        # the generic positive-Y camera: this task is viewed from slightly
        # behind the table (negative Y).
        base_eye = np.array([0.35, 0.0, 0.65], dtype=np.float64)
        base_target = np.array([0.0, 0.0, 0.15], dtype=np.float64)
        vertical_render_pose = np.array(
            [0.705400, -0.086655, 0.686691, 0.025112, -0.237384, -0.033640, 0.970508],
            dtype=np.float64,
        )
        render_eye = vertical_render_pose[:3]
        render_target = None
    elif task_id == "PegInsertionSide-v1":
        base_eye = np.array([0.35, 0.0, 0.65], dtype=np.float64)
        base_target = np.array([0.0, 0.0, 0.15], dtype=np.float64)
        render_eye = np.array([0.5, -0.5, 0.8], dtype=np.float64)
        render_target = np.array([0.05, -0.1, 0.4], dtype=np.float64)
    elif task_id == "PullCubeTool-golf":
        base_eye = np.array([0.3, 0.0, 0.5], dtype=np.float64)
        base_target = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
        render_eye = np.array([0.85, -0.5, 0.6], dtype=np.float64)
        render_target = np.array([0.0, 0.0, 0.35], dtype=np.float64)
    elif task_id == "UprightStack-v1":
        base_eye = np.array([-0.3, 0.0, 0.6], dtype=np.float64)
        base_target = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
        render_eye = np.array([0.2, 0.8, 0.3], dtype=np.float64)
        render_target = np.array([0.0, 0.0, 0.15], dtype=np.float64)
    elif task_id == "PushCube-v1":
        base_eye = np.array([0.3, 0.0, 0.6], dtype=np.float64)
        base_target = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
        render_eye = np.array([0.6, 0.7, 0.6], dtype=np.float64)
        render_target = np.array([0.0, 0.0, 0.35], dtype=np.float64)
    elif is_pull_block:
        # Calibrated against all eight range corners: this ReplicaCAD task is
        # occluded by the generic camera's positive-Y placement.
        base_eye = np.array([0.3, 0.0, 0.6], dtype=np.float64)
        base_target = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
        render_eye = np.array([0.7, -0.2, 0.6], dtype=np.float64)
        render_target = np.array([0.0, 0.0, 0.35], dtype=np.float64)
    elif is_plug_charger:
        # Local pose on the receptacle, matching PlugChargerEnv's mounted camera.
        base_eye = np.array([0.35, 0.0, 0.65], dtype=np.float64)
        base_target = np.array([0.0, 0.0, 0.15], dtype=np.float64)
        render_eye = np.array([0.15, -0.5, 0.04], dtype=np.float64)
        render_target = np.array([0.0, 0.0, 0.04], dtype=np.float64)
    else:
        base_eye = np.array([0.35, 0.0, 0.65], dtype=np.float64)
        base_target = np.array([0.0, 0.0, 0.15], dtype=np.float64)
        render_eye = np.array([0.65, 0.6, 0.65], dtype=np.float64)
        render_target = np.array([0.0, 0.0, 0.25], dtype=np.float64)

    if extreme:
        signs = np.array([
            -1.0 if rng.random() < 0.5 else 1.0,
            -1.0 if rng.random() < 0.5 else 1.0,
            -1.0 if rng.random() < 0.5 else 1.0,
        ])
        top_delta = signs * np.array([0.08, 0.08, 0.05])
        wrist_delta = signs * np.array([0.01, 0.01, 0.01])
    else:
        top_delta = rng.uniform([-0.04, -0.04, -0.03], [0.04, 0.04, 0.03])
        wrist_delta = rng.uniform([-0.005, -0.005, -0.005], [0.005, 0.005, 0.005])

    if is_plug_charger:
        # Keep PlugChargerEnv's receptacle-mounted render camera fixed.
        render_delta = np.zeros(3, dtype=np.float64)
    elif is_pull_block:
        # ponytail: calibrated box is intentionally small; enlarge only after
        # rendering every new boundary on the ReplicaCAD scene.
        if extreme:
            render_delta = signs * np.array([0.025, 0.025, 0.020])
        else:
            render_delta = rng.uniform([-0.025, -0.025, -0.020], [0.025, 0.025, 0.020])
    elif extreme:
        render_delta = signs * np.array([0.02, 0.02, 0.015])
    else:
        # Camera augmentation stays within a small neighbourhood of each
        # task's calibrated default view; large scene-crossing shifts produce
        # invalid/occluded policy observations.
        render_delta = rng.uniform([-0.02, -0.02, -0.015], [0.02, 0.02, 0.015])

    if vertical_render_pose is not None:
        render_pose = vertical_render_pose.copy()
        render_pose[:3] += render_delta
        render_pose = [float(x) for x in render_pose]
    else:
        render_pose = _look_at_pose(render_eye + render_delta, render_target)

    return {
        "base": {"uid": BASE_CAMERA, "pose": _look_at_pose(base_eye + top_delta, base_target)},
        "wrist": {
            "uid": WRIST_CAMERA,
            "pose": [float(wrist_delta[0]), float(wrist_delta[1]), float(wrist_delta[2]), 1.0, 0.0, 0.0, 0.0],
        },
        "render": {"uid": RENDER_CAMERA, "pose": render_pose},
    }


def _sensor_configs(args, camera_spec: dict[str, Any] | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {"shader_pack": args.shader}
    top_cfg = {"width": args.image_size, "height": args.image_size, "shader_pack": args.shader}
    wrist_cfg = {"width": args.image_size, "height": args.image_size, "shader_pack": args.shader}
    if camera_spec and args.randomize_top_camera:
        top_cfg["pose"] = camera_spec["base"]["pose"]
    if camera_spec and args.randomize_wrist_camera:
        wrist_cfg["pose"] = camera_spec["wrist"]["pose"]
    cfg[BASE_CAMERA] = top_cfg
    cfg[WRIST_CAMERA] = wrist_cfg
    return cfg


def _human_render_configs(args, render_spec: dict[str, Any] | None, camera_spec: dict[str, Any] | None) -> dict[str, Any]:
    if args.task_id == "PlugCharger-v1":
        # Do not override the task's receptacle-mounted render camera.
        return {"shader_pack": args.shader}
    if camera_spec and args.randomize_render_camera:
        return {"shader_pack": args.shader, RENDER_CAMERA: {"pose": camera_spec["render"]["pose"]}}
    return human_render_camera_overrides(args.shader, render_spec)


def _make_env(args, traj_name: str, render_spec: dict[str, Any] | None, camera_spec: dict[str, Any] | None):
    kwargs = dict(
        obs_mode=args.obs_mode,
        robot_uids=args.robot_uids,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        reward_mode=args.reward_mode or _compatible_reward_mode(args.task_id),
        sensor_configs=_sensor_configs(args, camera_spec),
        human_render_camera_configs=_human_render_configs(args, render_spec, camera_spec),
        viewer_camera_configs={"shader_pack": args.shader},
        sim_backend=args.sim_backend,
        max_episode_steps=args.max_episode_steps or TASKS[args.task_id].max_episode_steps,
    )
    env = gym.make(args.task_id, **kwargs)
    apply_render_randomization(env, render_spec)
    env = CollectionStepLimit(env, args.max_episode_steps or TASKS[args.task_id].max_episode_steps)
    env = RecordTcpPose(env)
    env = RecordEpisode(
        env,
        output_dir=osp.join(args.output_dir, "raw", args.task_id),
        trajectory_name=traj_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="RoboFPE/ManiSkill motion-planning solution for SFT collection",
        video_fps=FPS,
        info_on_video=False,
        avoid_overwriting_video=True,
        save_on_reset=False,
    )
    return env


def _validate_args(args) -> TaskSpec:
    if args.task_id not in TASKS:
        raise SystemExit(f"Unsupported task_id={args.task_id}. Available: {sorted(TASKS)}")
    spec = TASKS[args.task_id]
    if args.collect_mode == "insert_only" and not spec.supports_insert_only:
        raise SystemExit(f"{args.task_id} only supports collect-mode=full.")
    if args.robot_uids != DEFAULT_ROBOT_UIDS and not args.allow_no_wrist_camera:
        raise SystemExit(
            f"SFT collection requires wrist camera robot_uids={DEFAULT_ROBOT_UIDS}; got {args.robot_uids}. "
            "Pass --allow-no-wrist-camera only for debugging."
        )
    return spec


def _reset_options(args) -> dict[str, Any] | None:
    opts: dict[str, Any] = {}
    if args.randomize_initial_poses:
        opts["randomize_initial_poses"] = True
    if args.collect_mode == "insert_only":
        opts["collect_mode"] = "insert_only"
    return opts or None


def _remove_record_files(h5_path: str, video_path: str | None = None) -> None:
    paths = [h5_path, h5_path.replace(".h5", ".json")]
    if video_path:
        paths.append(video_path)
    else:
        paths.append(h5_path.replace(".h5", ".mp4"))
    for path in paths:
        if osp.exists(path):
            os.remove(path)


def _collection_episode_metadata(
    args, spec: TaskSpec, stage_records, camera_spec, render_spec, seed, worker_id,
    randomization_token, construction_seed,
):
    metadata = {
        "task_id": args.task_id,
        "prompt": spec.prompt,
        "collect_mode": args.collect_mode,
        "robot_uids": args.robot_uids,
        "stage_records": stage_records,
        "collection_max_episode_steps": args.max_episode_steps or spec.max_episode_steps,
        "solver_timeout": args.solver_timeout or spec.solver_timeout,
        "randomization_token": randomization_token,
        "construction_seed": int(construction_seed),
        "randomization_flags": {
            "initial_poses": bool(args.randomize_initial_poses),
            "base_camera": bool(args.randomize_top_camera),
            "wrist_camera": bool(args.randomize_wrist_camera),
            "render_camera": bool(args.randomize_render_camera),
            "lighting": bool(args.randomize_lighting),
        },
        "camera_randomization": camera_spec,
        "camera_uids": {
            "dataset_main": RENDER_CAMERA,
            "dataset_wrist": WRIST_CAMERA,
            "raw_base": BASE_CAMERA,
        },
        "render_camera_source": "RecordEpisode env.render() / human_render_camera_configs",
        "render_video": "same stem as source_h5 with .mp4 suffix",
        "seed": int(seed),
        "worker_id": int(worker_id),
    }
    if render_spec is not None:
        metadata.update(render_randomization_metadata(render_spec, randomization_token))
    return metadata


def _collect_worker(args, target_successes: int, worker_id: int, worker_output_dir: str) -> dict[str, Any]:
    spec = _validate_args(args)
    os.makedirs(worker_output_dir, exist_ok=True)
    try:
        solve = _load_solution(args.task_id)
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import solver for {args.task_id}. Check RoboFPE/ManiSkill version compatibility. Original error: {exc}"
        ) from exc
    passed = 0
    seed = args.seed + worker_id * args.worker_seed_stride
    attempts = 0
    max_attempts = max(target_successes * args.max_attempts_per_traj, target_successes)
    successes: list[bool] = []
    raw_h5: list[str] = []
    randomization_rows: list[dict[str, Any]] = []
    pbar = tqdm(total=target_successes, desc=f"collect {args.task_id} worker={worker_id}")
    while passed < target_successes:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Worker {worker_id} collected {passed}/{target_successes} successes "
                f"after {attempts} attempts."
            )
        attempts += 1
        token = f"{args.seed}:{args.task_id}:{worker_id}:{attempts}:{passed}:{seed}"
        construction_seed = _stable_seed(token) % (2**32)
        random.seed(construction_seed)
        np.random.seed(construction_seed)
        render_spec = sample_render_randomization_spec(token) if args.randomize_lighting else None
        camera_spec = _sample_camera_spec(args.task_id, token)
        traj_name = f"{args.task_id}_worker_{worker_id:02d}_{passed:06d}_seed_{seed}"
        worker_args = argparse.Namespace(**vars(args))
        worker_args.output_dir = worker_output_dir
        env = _make_env(worker_args, traj_name, render_spec, camera_spec)
        h5_path = env._h5_file.filename
        video_path = osp.join(worker_output_dir, "raw", args.task_id, f"{traj_name}.mp4")
        stage_records: list[dict[str, Any]] = []
        try:
            try:
                res = run_solution(
                    solve,
                    env,
                    seed=seed,
                    debug=False,
                    vis=args.vis,
                    reset_options=_reset_options(args),
                    timeout_seconds=args.solver_timeout or spec.solver_timeout,
                )
                res, stage_records = unwrap_solution_result(res)
            except (CollectionEpisodeTimeout, SolverTimeout) as exc:
                print(f"solution timed out for seed={seed}: {exc}")
                print(f"solution timed out for seed={seed}")
                res = -1
            except Exception as exc:
                print(f"solution failed for seed={seed}: {exc}")
                res = -1
            success = False if res == -1 else environment_success(env)
            if not success:
                print(f"solution did not reach task success for seed={seed}: result={res!r}")
            successes.append(bool(success))
            if args.success_only and not success:
                env.flush_trajectory(save=False)
                if args.save_video:
                    env.flush_video(name=traj_name, save=False)
                env.close()
                _remove_record_files(h5_path, video_path)
                seed += 1
                continue
            env.flush_trajectory()
            if args.save_video:
                env.flush_video(name=traj_name)
            env.close()
            metadata = _collection_episode_metadata(
                args, spec, stage_records, camera_spec, render_spec, seed, worker_id,
                token, construction_seed,
            )
            annotate_episode_metadata(h5_path, [metadata])
            raw_h5.append(h5_path)
            randomization_rows.append(metadata)
            passed += 1
            pbar.update(1)
            pbar.set_postfix(dict(attempts=attempts, success_rate=float(np.mean(successes))))
            seed += 1
        except Exception:
            try:
                env.close()
            except Exception:
                pass
            raise
    pbar.close()
    manifest = {
        "task": asdict(spec),
        "worker_id": worker_id,
        "num_successes": len(raw_h5),
        "attempts": attempts,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "raw_h5": raw_h5,
        "robot_uids": args.robot_uids,
        "training_cameras": ["observation.images.top", "observation.images.wrist"],
        "camera_uids": {
            "top": RENDER_CAMERA,
            "wrist": WRIST_CAMERA,
            "raw_base": BASE_CAMERA,
        },
    }
    manifest_path = osp.join(worker_output_dir, "collection_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(osp.join(worker_output_dir, "randomization_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(randomization_rows, f, indent=2)
    return manifest


def _worker_command(args, worker_id: int, target: int, worker_output_dir: str) -> list[str]:
    sim_backend = args.sim_backend
    if sim_backend in {"gpu", "cuda", "physx_cuda"}:
        # The parent assigns CUDA_VISIBLE_DEVICES to one physical GPU per
        # worker. Inside that process, the assigned GPU is always cuda:0.
        sim_backend = "gpu"
    command = [
        sys.executable,
        osp.abspath(__file__),
        "collect",
        "--task-id", args.task_id,
        "--num-traj", str(target),
        "--output-dir", worker_output_dir,
        "--robot-uids", args.robot_uids,
        "--collect-mode", args.collect_mode,
        "--sim-backend", sim_backend,
        "--shader", args.shader,
        "--image-size", str(args.image_size),
        "--seed", str(args.seed),
        "--max-attempts-per-traj", str(args.max_attempts_per_traj),
        "--solver-timeout", str(args.solver_timeout or TASKS[args.task_id].solver_timeout),
        "--worker-seed-stride", str(args.worker_seed_stride),
        "--worker-id", str(worker_id),
        "--worker-target", str(target),
    ]
    if args.reward_mode:
        command += ["--reward-mode", args.reward_mode]
    if args.max_episode_steps is not None:
        command += ["--max-episode-steps", str(args.max_episode_steps)]
    for flag in [
        "randomize_initial_poses",
        "randomize_top_camera",
        "randomize_wrist_camera",
        "randomize_render_camera",
        "randomize_lighting",
        "save_video",
        "vis",
    ]:
        if getattr(args, flag):
            command.append("--" + flag.replace("_", "-"))
    if args.obs_mode != "rgb":
        command += ["--obs-mode", args.obs_mode]
    if args.render_mode != "rgb_array":
        command += ["--render-mode", args.render_mode]
    return command


def _collect_parallel(args) -> None:
    _validate_args(args)
    if args.num_workers < 1:
        raise SystemExit("--num-workers must be >= 1")
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise SystemExit("--gpu-ids must contain at least one GPU id")
    if args.num_workers > len(gpu_ids):
        raise SystemExit(
            "SAPIEN/Vulkan collection requires at most one worker per physical GPU; "
            f"got --num-workers={args.num_workers} for --gpu-ids={args.gpu_ids}"
        )
    import torch

    visible_gpu_count = torch.cuda.device_count()
    if len(gpu_ids) > visible_gpu_count:
        raise RuntimeError(
            f"Requested {len(gpu_ids)} GPUs ({','.join(gpu_ids)}), but only "
            f"{visible_gpu_count} CUDA device(s) are visible to this process. "
            "Set CUDA_VISIBLE_DEVICES to the available GPUs or run on a "
            "machine with four visible GPUs."
        )
    worker_targets = [args.num_traj // args.num_workers] * args.num_workers
    for worker_id in range(args.num_traj % args.num_workers):
        worker_targets[worker_id] += 1
    shard_root = Path(args.output_dir) / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen, int, Path]] = []
    for worker_id, target in enumerate(worker_targets):
        if target == 0:
            continue
        worker_dir = shard_root / f"worker-{worker_id:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        # One collector owns one Vulkan device. CUDA_VISIBLE_DEVICES makes that
        # physical assignment unambiguous inside the subprocess.
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_id]
        command = _worker_command(args, worker_id, target, str(worker_dir))
        processes.append(
            (subprocess.Popen(command, env=env), worker_id, worker_dir)
        )
    failures = []
    manifests = []
    worker_timeout = args.worker_timeout or (args.solver_timeout * 2.0 + 60.0)
    for process, worker_id, worker_dir in processes:
        try:
            return_code = process.wait(timeout=worker_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            failures.append((worker_id, "timeout"))
            continue
        manifest_path = worker_dir / "collection_manifest.json"
        if return_code != 0 or not manifest_path.exists():
            failures.append((worker_id, return_code))
            continue
        with manifest_path.open("r", encoding="utf-8") as f:
            manifests.append(json.load(f))
    if failures:
        raise RuntimeError(f"SFT collection workers failed: {failures}")
    raw_h5 = [path for manifest in manifests for path in manifest.get("raw_h5", [])]
    if len(raw_h5) != args.num_traj:
        raise RuntimeError(
            f"Expected {args.num_traj} successful trajectories, found {len(raw_h5)}"
        )
    aggregate = {
        "task": asdict(TASKS[args.task_id]),
        "num_successes": len(raw_h5),
        "attempts": sum(item.get("attempts", 0) for item in manifests),
        "workers": manifests,
        "raw_h5": raw_h5,
        "robot_uids": args.robot_uids,
        "training_cameras": ["observation.images.top", "observation.images.wrist"],
        "camera_uids": {
            "top": RENDER_CAMERA,
            "wrist": WRIST_CAMERA,
            "raw_base": BASE_CAMERA,
        },
    }
    with (Path(args.output_dir) / "collection_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    randomization_rows = []
    for manifest in manifests:
        worker_dir = shard_root / f"worker-{manifest['worker_id']:02d}"
        path = worker_dir / "randomization_manifest.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                randomization_rows.extend(json.load(f))
    with (Path(args.output_dir) / "randomization_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(randomization_rows, f, indent=2)
    if args.convert:
        cargs = argparse.Namespace(**vars(args))
        cargs.input = str(shard_root)
        cargs.dataset_dir = osp.join(args.output_dir, "lerobot")
        cargs.overwrite = True
        cargs.num_convert_workers = max(1, args.num_convert_workers or args.num_workers)
        convert(cargs)


def collect(args) -> None:
    if args.worker_id is not None:
        _collect_worker(args, args.worker_target or args.num_traj, args.worker_id, args.output_dir)
        return
    if args.num_workers > 1:
        _collect_parallel(args)
        return
    manifest = _collect_worker(args, args.num_traj, 0, args.output_dir)
    if args.convert:
        cargs = argparse.Namespace(**vars(args))
        cargs.input = osp.join(args.output_dir, "raw", args.task_id)
        cargs.dataset_dir = osp.join(args.output_dir, "lerobot")
        cargs.num_convert_workers = max(1, args.num_convert_workers or 1)
        convert(cargs)


def _read_json(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    with open(sidecar, "r", encoding="utf-8") as f:
        return json.load(f)


def _traj_keys(h5) -> list[str]:
    return sorted([key for key in h5.keys() if key.startswith("traj_")], key=lambda x: int(x.split("_")[-1]))


def _np_leaf(value) -> np.ndarray:
    arr = np.asarray(value)
    while arr.ndim >= 1 and arr.shape[0] == 1 and arr.ndim > 1:
        arr = arr[0]
    return arr


def _get_nested(group, path: list[str], index: int | None = None) -> np.ndarray | None:
    cur: Any = group
    for part in path:
        if part not in cur:
            return None
        cur = cur[part]
    return cur[()] if index is None else cur[index]


def _camera_rgb(traj, camera_uid: str, t: int) -> np.ndarray:
    candidates = [
        ["obs", "sensor_data", camera_uid, "rgb"],
        ["obs", "image", camera_uid, "rgb"],
        ["obs", camera_uid, "rgb"],
    ]
    for path in candidates:
        data = _get_nested(traj, path, t)
        if data is not None:
            frame = _np_leaf(data)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            return frame
    raise KeyError(f"Could not find RGB for camera {camera_uid}")


def _robot_state(traj: h5py.Group, t: int) -> np.ndarray:
    candidates = [
        ["obs", "agent", "qpos"],
        ["obs", "state", "agent", "qpos"],
    ]
    for path in candidates:
        data = _get_nested(traj, path, t)
        if data is not None:
            return _np_leaf(data).astype(np.float32)
    # Fallback for RecordEpisode files collected with obs_mode=none.
    if "env_states" in traj:
        states = trajectory_utils.dict_to_list_of_dicts({k: v[()] for k, v in traj["env_states"].items()})
        state = states[min(t, len(states) - 1)]
        for key in ["agent", "articulations"]:
            if key in state:
                text = np.asarray(str(state[key]).encode("utf-8"), dtype=np.uint8).astype(np.float32)
                return text[:8]
    raise KeyError("Could not find robot qpos in trajectory observations; collect with --obs-mode rgb.")


def _tcp_state(traj: h5py.Group, t: int, state: np.ndarray) -> np.ndarray | None:
    """Convert the captured world TCP pose to the aligned pi0.5 state."""
    pose = _get_nested(traj, ["obs", "extra", "tcp_pose"], t)
    if pose is None:
        return None
    pose = _np_leaf(pose).astype(np.float64)
    if pose.shape != (7,):
        return None
    from rlinf.envs.maniskill.peg_insertion_pi05 import (
        aligned_pi05_state_from_tcp_matrix,
    )
    import sapien

    articulations = traj["env_states"]["articulations"]
    root_data = next(iter(articulations.values()))
    root = np.asarray(root_data[t, :7], dtype=np.float64)
    root_pose = sapien.Pose(p=root[:3], q=root[3:])
    tcp_pose = sapien.Pose(p=pose[:3], q=pose[3:])
    matrix = np.asarray(
        (root_pose.inv() * tcp_pose).to_transformation_matrix(),
        dtype=np.float64,
    )
    gripper = float(state[7] if state.shape[0] > 7 else state[-1])
    return aligned_pi05_state_from_tcp_matrix(matrix, [gripper, gripper])


def _resize(frame: np.ndarray, size: int) -> np.ndarray:
    import cv2
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"No frames for {path}")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _array_summary(values: np.ndarray) -> dict[str, Any]:
    values = values.astype(np.float64)
    return {
        "sum": values.sum(axis=0),
        "sumsq": np.square(values).sum(axis=0),
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "count": int(values.shape[0]),
    }


def _merge_summary(acc: dict[str, Any] | None, item: dict[str, Any]) -> dict[str, Any]:
    if acc is None:
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in item.items()}
    acc["sum"] += item["sum"]
    acc["sumsq"] += item["sumsq"]
    acc["min"] = np.minimum(acc["min"], item["min"])
    acc["max"] = np.maximum(acc["max"], item["max"])
    acc["count"] += item["count"]
    return acc


def _summary_to_stats(summary: dict[str, Any]) -> dict[str, Any]:
    count = int(summary["count"])
    mean = summary["sum"] / count
    var = np.maximum(summary["sumsq"] / count - np.square(mean), 0.0)
    return {
        "mean": mean.astype(float).tolist(),
        "std": np.sqrt(var).astype(float).tolist(),
        "min": summary["min"].astype(float).tolist(),
        "max": summary["max"].astype(float).tolist(),
        "count": [count],
    }


def _read_video_frames(path: Path, expected_frames: int) -> list[np.ndarray]:
    import cv2

    if not path.exists():
        raise FileNotFoundError(
            f"Missing human render video {path}. Collect with --save-video."
        )
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while len(frames) < expected_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) < expected_frames:
        raise RuntimeError(
            f"Human render video {path} has {len(frames)} frames; "
            f"expected at least {expected_frames}."
        )
    return frames[:expected_frames]


def _convert_episode_job(job: dict[str, Any]) -> dict[str, Any]:
    import h5py

    h5_path = Path(job["h5_path"])
    out = Path(job["out"])
    episode_index = int(job["episode_index"])
    global_start = int(job["global_start"])
    image_size = int(job["image_size"])
    task_prompt = job["task_prompt"]

    with h5py.File(h5_path, "r") as h5:
        traj = h5[job["traj_key"]]
        actions = traj["actions"][()].astype(np.float32)
        if actions.ndim == 3 and actions.shape[1] == 1:
            actions = actions[:, 0]
        T = actions.shape[0]
        render_frames = _read_video_frames(h5_path.with_suffix(".mp4"), T)
        top_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []
        states: list[np.ndarray] = []
        tcp_states: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        for t in range(T):
            top = _resize(render_frames[t], image_size)
            wrist = _resize(_camera_rgb(traj, WRIST_CAMERA, t), image_size)
            state = _robot_state(traj, t).astype(np.float32)
            action = actions[t].astype(np.float32)
            tcp_state = _tcp_state(traj, t, state) if job["include_tcp"] else None
            top_frames.append(top)
            wrist_frames.append(wrist)
            states.append(state)
            row = {
                "observation.state": state.tolist(),
                "actions": action.tolist(),
                "timestamp": float(t / FPS),
                "frame_index": int(t),
                "episode_index": int(episode_index),
                "index": int(global_start + t),
                "task_index": 0,
                "task": task_prompt,
                "prompt": task_prompt,
            }
            if tcp_state is not None:
                row["observation.state_tcp"] = tcp_state.tolist()
                tcp_states.append(tcp_state)
            rows.append(row)

    state_values = np.stack(states).astype(np.float32)
    tcp_values = np.stack(tcp_states).astype(np.float32) if tcp_states else None
    if tcp_values is not None and len(tcp_values) != T:
        raise RuntimeError(f"{h5_path}: TCP state missing for some frames")
    chunk = episode_index // CHUNK_SIZE
    state_dim = int(state_values.shape[-1])
    action_dim = int(actions.shape[-1])
    tcp_dim = int(tcp_values.shape[-1]) if tcp_values is not None else None
    _write_episode_parquet(
        out / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet",
        rows,
        state_dim,
        action_dim,
        tcp_dim,
    )
    for video_key, frames in [
        ("observation.images.top", top_frames),
        ("observation.images.wrist", wrist_frames),
    ]:
        _write_video(
            out / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4",
            frames,
            FPS,
        )
    return {
        "episode_index": episode_index,
        "episode_meta": {
            "episode_index": episode_index,
            "tasks": [task_prompt],
            "length": T,
            "source_h5": str(h5_path),
            "metadata": job["metadata"],
        },
        "length": T,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "tcp_state_dim": tcp_dim,
        "actions": _array_summary(actions),
        "states": _array_summary(state_values),
        "tcp_states": _array_summary(tcp_values) if tcp_values is not None else None,
    }


def _array_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def _features(
    state_dim: int,
    action_dim: int,
    image_size: int,
    total_videos: int,
    tcp_state_dim: int | None = None,
) -> dict[str, Any]:
    features = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": None, "fps": FPS},
        "actions": {"dtype": "float32", "shape": [action_dim], "names": None, "fps": FPS},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": FPS},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": FPS},
        "task": {"dtype": "string", "shape": [1], "names": None, "fps": FPS},
        "prompt": {"dtype": "string", "shape": [1], "names": None, "fps": FPS},
    }
    for key in ["observation.images.top", "observation.images.wrist"]:
        features[key] = {
            "dtype": "video",
            "shape": [image_size, image_size, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": float(FPS),
                "video.height": image_size,
                "video.width": image_size,
                "video.channels": 3,
                "video.codec": "mp4v",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    if tcp_state_dim is not None:
        features["observation.state_tcp"] = {
            "dtype": "float32",
            "shape": [tcp_state_dim],
            "names": [
                "tcp_x",
                "tcp_y",
                "tcp_z",
                "roll",
                "pitch",
                "yaw",
                "finger0",
                "finger1",
            ],
            "fps": FPS,
        }
    return features


def _schema(state_dim: int, action_dim: int, tcp_state_dim: int | None = None):
    import pyarrow as pa
    fields = [
        pa.field("observation.state", pa.list_(pa.float32(), state_dim)),
        pa.field("actions", pa.list_(pa.float32(), action_dim)),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("task", pa.string()),
        pa.field("prompt", pa.string()),
    ]
    if tcp_state_dim is not None:
        fields.insert(1, pa.field("observation.state_tcp", pa.list_(pa.float32(), tcp_state_dim)))
    return pa.schema(fields)


def _write_episode_parquet(
    path: Path,
    rows: list[dict[str, Any]],
    state_dim: int,
    action_dim: int,
    tcp_state_dim: int | None = None,
) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    pq.write_table(
        pa.Table.from_pandas(
            df,
            schema=_schema(state_dim, action_dim, tcp_state_dim),
            preserve_index=False,
        ),
        path,
    )


def _episode_metadata(sidecar: dict[str, Any], index: int) -> dict[str, Any]:
    episodes = sidecar.get("episodes") or []
    if index < len(episodes):
        return episodes[index]
    return {}


def _iter_h5_files(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    return sorted(p.rglob("*.h5"))


def _load_episode_allowlist(path: str | None) -> set[Path] | None:
    if not path:
        return None
    accepted: set[Path] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "accepted":
                continue
            source = row.get("source_h5")
            if not source:
                raise ValueError(f"{path}:{line_number}: accepted row has no source_h5")
            accepted.add(Path(source).resolve())
    if not accepted:
        raise ValueError(f"Episode allowlist contains no accepted rows: {path}")
    return accepted


def _validated_raw_files(args) -> list[Path]:
    import h5py

    discovered = _iter_h5_files(args.input)
    allowlist = _load_episode_allowlist(getattr(args, "episode_allowlist", None))
    if allowlist is not None:
        by_path = {path.resolve(): path for path in discovered}
        missing = sorted(str(path) for path in allowlist - set(by_path))
        if missing:
            raise ValueError(f"Allowlist references H5 files outside --input: {missing[:5]}")
        discovered = [by_path[path] for path in sorted(allowlist, key=str)]
    if not discovered:
        raise SystemExit(f"No .h5 files found under {args.input}")

    seeds: set[int] = set()
    for h5_path in discovered:
        sidecar_path = h5_path.with_suffix(".json")
        video_path = h5_path.with_suffix(".mp4")
        if not sidecar_path.exists() or not video_path.exists():
            raise FileNotFoundError(f"Raw episode is not H5/JSON/MP4 paired: {h5_path}")
        sidecar = _read_json(h5_path)
        episodes = sidecar.get("episodes") or []
        with h5py.File(h5_path, "r") as h5:
            trajectory_count = len(_traj_keys(h5))
        if len(episodes) != trajectory_count:
            raise ValueError(
                f"{h5_path}: {trajectory_count} trajectories but {len(episodes)} metadata rows"
            )
        for metadata in episodes:
            if metadata.get("task_id") != args.task_id:
                raise ValueError(
                    f"{h5_path}: raw task_id={metadata.get('task_id')!r}, "
                    f"requested task_id={args.task_id!r}"
                )
            if metadata.get("robot_uids") != args.robot_uids:
                raise ValueError(
                    f"{h5_path}: raw robot_uids={metadata.get('robot_uids')!r}, "
                    f"requested robot_uids={args.robot_uids!r}"
                )
            if metadata.get("success") is not True:
                raise ValueError(f"{h5_path}: conversion requires success=true")
            seed = metadata.get("seed", metadata.get("episode_seed"))
            if seed is not None:
                seed = int(seed)
                if seed in seeds:
                    raise ValueError(f"Duplicate episode seed {seed}: {h5_path}")
                seeds.add(seed)
    return discovered


def convert(args) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import h5py
    import pandas as pd
    _validate_args(args)
    h5_files = _validated_raw_files(args)
    out = Path(args.dataset_dir)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    global_index = 0
    episode_index = 0
    state_dim = None
    action_dim = None
    task_prompt = TASKS[args.task_id].prompt
    for h5_path in tqdm(h5_files, desc="convert h5"):
        sidecar = _read_json(h5_path)
        with h5py.File(h5_path, "r") as h5:
            for local_idx, key in enumerate(_traj_keys(h5)):
                traj = h5[key]
                actions = traj["actions"][()].astype(np.float32)
                if actions.ndim == 3 and actions.shape[1] == 1:
                    actions = actions[:, 0]
                T = actions.shape[0]
                jobs.append({
                    "h5_path": str(h5_path),
                    "traj_key": key,
                    "episode_index": episode_index,
                    "global_start": global_index,
                    "task_prompt": task_prompt,
                    "image_size": args.image_size,
                    "out": str(out),
                    "metadata": _episode_metadata(sidecar, local_idx),
                    "include_tcp": args.task_id.startswith("PegInsertion"),
                })
                episode_index += 1
                global_index += T
    if not jobs:
        raise RuntimeError("No frames converted")
    num_convert_workers = max(1, int(getattr(args, "num_convert_workers", 1)))
    print(f"Converting {len(jobs)} episodes with {num_convert_workers} worker(s)")
    results = []
    actions_summary = None
    states_summary = None
    tcp_states_summary = None
    tcp_state_dim = None
    if num_convert_workers == 1:
        iterator = (_convert_episode_job(job) for job in jobs)
        for result in tqdm(iterator, total=len(jobs), desc="write episodes"):
            results.append(result)
            actions_summary = _merge_summary(actions_summary, result["actions"])
            states_summary = _merge_summary(states_summary, result["states"])
            if result["tcp_states"] is not None:
                tcp_states_summary = _merge_summary(tcp_states_summary, result["tcp_states"])
                tcp_state_dim = result["tcp_state_dim"]
    else:
        with ProcessPoolExecutor(max_workers=num_convert_workers) as pool:
            futures = [pool.submit(_convert_episode_job, job) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="write episodes"):
                result = future.result()
                results.append(result)
                actions_summary = _merge_summary(actions_summary, result["actions"])
                states_summary = _merge_summary(states_summary, result["states"])
                if result["tcp_states"] is not None:
                    tcp_states_summary = _merge_summary(tcp_states_summary, result["tcp_states"])
                    tcp_state_dim = result["tcp_state_dim"]
    results.sort(key=lambda item: item["episode_index"])
    episodes_meta = [item["episode_meta"] for item in results]
    state_dim = int(results[0]["state_dim"])
    action_dim = int(results[0]["action_dim"])
    total_frames = int(sum(item["length"] for item in results))
    meta_dir = out / "meta"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")
    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": task_prompt}) + "\n")
    pd.DataFrame({"task_index": [0], "task": [task_prompt]}).to_parquet(meta_dir / "tasks.parquet", index=False)
    stats = {
        "actions": _summary_to_stats(actions_summary),
        "observation.state": _summary_to_stats(states_summary),
    }
    if tcp_states_summary is not None:
        stats["observation.state_tcp"] = _summary_to_stats(tcp_states_summary)
    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    data_size = sum(path.stat().st_size for path in out.rglob("data/**/*.parquet"))
    info = {
        "codebase_version": "v2.0",
        "dataset_variant": "robofpe-render-wrist-sft-v2",
        "task_id": args.task_id,
        "collect_mode": args.collect_mode,
        "robot_type": args.robot_uids,
        "total_episodes": episode_index,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": episode_index * 2,
        "total_chunks": int((episode_index + CHUNK_SIZE - 1) // CHUNK_SIZE),
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "data_files_size_in_mb": int(data_size / (1024 * 1024)),
        "splits": {"train": f"0:{episode_index}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _features(
            state_dim,
            action_dim,
            args.image_size,
            episode_index * 2,
            tcp_state_dim,
        ),
        "training_cameras": ["observation.images.top", "observation.images.wrist"],
        "camera_uids": {
            "top": RENDER_CAMERA,
            "wrist": WRIST_CAMERA,
            "raw_base": BASE_CAMERA,
        },
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    validate_sft_camera_contract(out)
    print(f"Wrote {episode_index} episodes to {out}")
    print(
        f"state_dim={state_dim}, action_dim={action_dim}, "
        f"tcp_state_dim={tcp_state_dim}, "
        f"cameras=top:{RENDER_CAMERA}, wrist:{WRIST_CAMERA}"
    )


def _decoded_frame_count(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(f"Decoded an empty frame: {path}")
            count += 1
    finally:
        capture.release()
    return count


def validate_sft_camera_contract(
    dataset_dir: Path,
    raw_input: str | None = None,
    full_video_decode: bool = False,
) -> None:
    """Fail closed unless the converted dataset has the two-view ManiSkill contract."""
    import cv2

    root = Path(dataset_dir)
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    expected_features = {"observation.images.top", "observation.images.wrist"}
    video_features = {
        key for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    }
    if video_features != expected_features:
        raise RuntimeError(f"SFT image features must be {expected_features}, got {video_features}")
    if info.get("training_cameras") != sorted(expected_features):
        raise RuntimeError(f"Unexpected training_cameras: {info.get('training_cameras')}")
    camera_uids = info.get("camera_uids", {})
    if camera_uids.get("top") != RENDER_CAMERA or camera_uids.get("wrist") != WRIST_CAMERA:
        raise RuntimeError(f"Wrong SFT camera routing: {camera_uids}")
    if BASE_CAMERA in video_features or BASE_CAMERA in (set(camera_uids.values()) - {BASE_CAMERA}):
        raise RuntimeError("base_camera must not be an SFT policy image")

    episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not episodes:
        raise RuntimeError("Dataset has no episodes")
    if info.get("total_episodes") != len(episodes):
        raise RuntimeError(
            f"info total_episodes={info.get('total_episodes')}, metadata has {len(episodes)}"
        )
    task_id = info.get("task_id")
    if task_id not in TASKS:
        raise RuntimeError(f"Unknown dataset task_id: {task_id!r}")
    task_rows = [json.loads(line) for line in (meta / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    expected_prompt = TASKS[task_id].prompt
    if task_rows != [{"task_index": 0, "task": expected_prompt}]:
        raise RuntimeError(f"Wrong task metadata: {task_rows}")

    total_frames = 0
    for expected_index, episode in enumerate(episodes):
        index = int(episode["episode_index"])
        if index != expected_index:
            raise RuntimeError(f"Non-contiguous episode index: expected {expected_index}, got {index}")
        length = int(episode["length"])
        total_frames += length
        chunk = index // int(info["chunks_size"])
        parquet = root / info["data_path"].format(
            episode_chunk=chunk, episode_index=index
        )
        if not parquet.exists():
            raise FileNotFoundError(f"Missing episode parquet: {parquet}")
        import pyarrow.parquet as pq

        table = pq.read_table(parquet, columns=["episode_index", "frame_index", "task", "prompt"])
        if table.num_rows != length:
            raise RuntimeError(f"{parquet}: {table.num_rows} rows, expected {length}")
        data = table.to_pydict()
        if set(data["episode_index"]) != {index} or data["frame_index"] != list(range(length)):
            raise RuntimeError(f"Bad frame/episode indices: {parquet}")
        if set(data["task"]) != {expected_prompt} or set(data["prompt"]) != {expected_prompt}:
            raise RuntimeError(f"Wrong task prompt in {parquet}")
        for key in sorted(expected_features):
            video = root / info["video_path"].format(
                episode_chunk=chunk, episode_index=index, video_key=key
            )
            capture = cv2.VideoCapture(str(video))
            ok, frame = capture.read()
            capture.release()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError(f"Cannot decode required SFT video: {video}")
            if full_video_decode and _decoded_frame_count(video) != length:
                raise RuntimeError(f"Video frame count does not match episode length: {video}")
    if info.get("total_frames") != total_frames:
        raise RuntimeError(f"info total_frames={info.get('total_frames')}, expected {total_frames}")

    if raw_input:
        import h5py

        raw_paths = _iter_h5_files(raw_input)
        if not raw_paths:
            raise RuntimeError(f"No raw H5 found under {raw_input}")
        for raw_path in raw_paths:
            with h5py.File(raw_path, "r") as h5:
                traj = h5[sorted(_traj_keys(h5))[0]]
                sensors = traj["obs"]["sensor_data"]
                missing = {BASE_CAMERA, WRIST_CAMERA} - set(sensors.keys())
                if missing:
                    raise RuntimeError(f"{raw_path}: missing raw camera(s): {sorted(missing)}")


def validate(args) -> None:
    validate_sft_camera_contract(
        Path(args.dataset_dir), args.raw_input, args.full_video_decode
    )
    print(f"Validated ManiSkill SFT camera contract: {args.dataset_dir}")


def summarize(args) -> None:
    path = Path(args.dataset_dir)
    info_path = path / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"Missing {info_path}")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    print(json.dumps({
        "task_id": info.get("task_id"),
        "robot_type": info.get("robot_type"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "training_camera_count": len(info.get("training_cameras", [])),
        "training_cameras": info.get("training_cameras", []),
        "camera_uids": info.get("camera_uids", {}),
        "render_only_cameras": info.get("render_only_cameras", []),
        "features": {k: v.get("shape") for k, v in info.get("features", {}).items() if k in ["observation.state", "actions", "observation.images.top", "observation.images.wrist"]},
    }, indent=2))


def preview_randomization(args) -> None:
    import cv2
    _validate_args(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.num_samples):
        token = f"preview:{args.task_id}:{args.seed}:{i}:{args.extreme}"
        render_spec = sample_render_randomization_spec(token) if args.randomize_lighting else None
        camera_spec = _sample_camera_spec(args.task_id, token, extreme=args.extreme)
        env = gym.make(
            args.task_id,
            obs_mode="rgb",
            robot_uids=args.robot_uids,
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode=args.reward_mode or _compatible_reward_mode(args.task_id),
            sensor_configs=_sensor_configs(args, camera_spec),
            human_render_camera_configs=_human_render_configs(args, render_spec, camera_spec),
            sim_backend=args.sim_backend,
        )
        apply_render_randomization(env, render_spec)
        obs, _ = env.reset(seed=args.seed + i)
        obs_np = common.to_numpy(obs)
        sensor_data = obs_np.get("sensor_data", {}) if isinstance(obs_np, dict) else {}
        for label, uid in [("base", BASE_CAMERA), ("wrist", WRIST_CAMERA)]:
            frame = sensor_data[uid]["rgb"]
            frame = _np_leaf(frame)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            cv2.imwrite(str(out / f"sample_{i:02d}_{label}.png"), cv2.cvtColor(_resize(frame, args.image_size), cv2.COLOR_RGB2BGR))
        try:
            render = common.to_numpy(
                env.unwrapped.render_rgb_array(camera_name=RENDER_CAMERA)
            )
            render = _np_leaf(render)
            if render.shape[-1] == 4:
                render = render[..., :3]
            render = _resize(render, args.image_size)
            cv2.imwrite(
                str(out / f"sample_{i:02d}_render.png"),
                cv2.cvtColor(render, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                str(out / f"sample_{i:02d}_top.png"),
                cv2.cvtColor(render, cv2.COLOR_RGB2BGR),
            )
        except Exception as exc:
            print(f"render_camera preview failed for sample {i}: {exc}")
        env.close()
        rows.append({"sample": i, "camera_randomization": camera_spec, "render_randomization": render_spec})
    with open(out / "preview_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "task_id": args.task_id,
            "robot_uids": args.robot_uids,
            "training_camera_count": 2,
            "training_cameras": {"top": RENDER_CAMERA, "wrist": WRIST_CAMERA},
            "raw_observation_cameras": {"base": BASE_CAMERA, "wrist": WRIST_CAMERA},
            "samples": rows,
        }, f, indent=2)
    print(f"Wrote preview images and camera report to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, require_task_id=False):
        p.add_argument(
            "--task-id",
            required=require_task_id,
            default=None if require_task_id else "PushCube-v1",
            choices=sorted(TASKS),
        )
        p.add_argument("--robot-uids", default=DEFAULT_ROBOT_UIDS)
        p.add_argument("--allow-no-wrist-camera", action="store_true")
        p.add_argument("--collect-mode", default="full", choices=["full", "insert_only"])
        p.add_argument("--sim-backend", default="auto")
        p.add_argument("--shader", default="default")
        p.add_argument("--reward-mode", default=None)
        p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--randomize-initial-poses", action="store_true")
        p.add_argument("--randomize-top-camera", action="store_true")
        p.add_argument("--randomize-wrist-camera", action="store_true")
        p.add_argument("--randomize-render-camera", action="store_true")
        p.add_argument("--randomize-lighting", action="store_true")
        p.add_argument("--max-episode-steps", type=int, default=None)

    p = sub.add_parser("collect")
    add_common(p)
    p.add_argument("--num-traj", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--obs-mode", default="rgb")
    p.add_argument("--render-mode", default="rgb_array")
    p.add_argument("--success-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-attempts-per-traj", type=int, default=50)
    p.add_argument("--solver-timeout", type=float, default=None)
    p.add_argument(
        "--worker-timeout", type=float, default=None,
        help="Hard timeout for each parallel worker process; defaults to solver-timeout*2+60.",
    )
    p.add_argument("--worker-seed-stride", type=int, default=SEED_STRIDE)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--vis", action="store_true")
    p.add_argument("--convert", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--num-convert-workers",
        type=int,
        default=48,
        help="Parallel conversion workers (default: 48).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of independent single-environment worker processes.",
    )
    p.add_argument("--gpu-ids", default="0", help="Comma-separated physical GPU ids for worker processes.")
    p.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-target", type=int, default=None, help=argparse.SUPPRESS)
    p.set_defaults(func=collect)

    p = sub.add_parser("convert")
    add_common(p, require_task_id=True)
    p.add_argument("--input", required=True, help="Raw .h5 file or directory")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--episode-allowlist", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--num-convert-workers", type=int, default=48)
    p.set_defaults(func=convert)

    p = sub.add_parser("preview-randomization")
    add_common(p)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--extreme", action="store_true")
    p.set_defaults(func=preview_randomization)

    p = sub.add_parser("summarize")
    p.add_argument("--dataset-dir", required=True)
    p.set_defaults(func=summarize)

    p = sub.add_parser("validate")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--raw-input", default=None)
    p.add_argument("--full-video-decode", action="store_true")
    p.set_defaults(func=validate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
