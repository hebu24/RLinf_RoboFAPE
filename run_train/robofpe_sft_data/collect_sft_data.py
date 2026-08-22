#!/usr/bin/env python3
"""Collect RoboFPE/ManiSkill successful trajectories and export wrist-camera SFT data.

The default path is PushCube-v1 with panda_wristcam, producing a LeRobot-style
OpenPI dataset with observation.images.top and observation.images.wrist.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
TOP_CAMERA = "base_camera"
WRIST_CAMERA = "hand_camera"
RENDER_CAMERA = "render_camera"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    prompt: str
    max_episode_steps: int
    supports_insert_only: bool = False
    top_camera: str = TOP_CAMERA
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
        from mani_skill.examples.motionplanning.panda.solutions import solvePlugCharger
        return solvePlugCharger
    if task_id == "PegInsertionVertical-v1":
        from solutions.solve_PegInsertionVertical import solve_peginsertionvertical
        return solve_peginsertionvertical
    if task_id == "PegInsertionSide-v1":
        from solutions.solve_PegInsertionSide import solve_peginsertionside
        return solve_peginsertionside
    raise KeyError(task_id)


def run_solution(solve, env, seed, debug, vis, reset_options=None):
    import inspect
    kwargs = dict(seed=seed, debug=debug, vis=vis)
    if reset_options is not None:
        try:
            if "reset_options" in inspect.signature(solve).parameters:
                kwargs["reset_options"] = reset_options
        except (TypeError, ValueError):
            pass
    return solve(env, **kwargs)


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
        max_episode_steps=400,
        supports_insert_only=True,
    ),
    "PegInsertionSide-v1": TaskSpec(
        task_id="PegInsertionSide-v1",
        prompt="Insert the peg into the side-facing target hole.",
        max_episode_steps=350,
        supports_insert_only=True,
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
    if task_id == "PushCube-v1":
        base_eye = np.array([0.3, 0.0, 0.6], dtype=np.float64)
        base_target = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
        render_eye = np.array([0.6, 0.7, 0.6], dtype=np.float64)
        render_target = np.array([0.0, 0.0, 0.35], dtype=np.float64)
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
        render_delta = signs * np.array([0.14, 0.14, 0.08])
    else:
        top_delta = rng.uniform([-0.04, -0.04, -0.03], [0.04, 0.04, 0.03])
        wrist_delta = rng.uniform([-0.005, -0.005, -0.005], [0.005, 0.005, 0.005])
        render_delta = rng.uniform([-0.08, -0.08, -0.05], [0.08, 0.08, 0.05])

    return {
        "top": {"uid": TOP_CAMERA, "pose": _look_at_pose(base_eye + top_delta, base_target)},
        "wrist": {
            "uid": WRIST_CAMERA,
            "pose": [float(wrist_delta[0]), float(wrist_delta[1]), float(wrist_delta[2]), 1.0, 0.0, 0.0, 0.0],
        },
        "render": {"uid": RENDER_CAMERA, "pose": _look_at_pose(render_eye + render_delta, render_target)},
    }


def _sensor_configs(args, camera_spec: dict[str, Any] | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {"shader_pack": args.shader}
    top_cfg = {"width": args.image_size, "height": args.image_size, "shader_pack": args.shader}
    wrist_cfg = {"width": args.image_size, "height": args.image_size, "shader_pack": args.shader}
    if camera_spec and args.randomize_top_camera:
        top_cfg["pose"] = camera_spec["top"]["pose"]
    if camera_spec and args.randomize_wrist_camera:
        wrist_cfg["pose"] = camera_spec["wrist"]["pose"]
    cfg[TOP_CAMERA] = top_cfg
    cfg[WRIST_CAMERA] = wrist_cfg
    return cfg


def _human_render_configs(args, render_spec: dict[str, Any] | None, camera_spec: dict[str, Any] | None) -> dict[str, Any]:
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
    )
    env = gym.make(args.task_id, **kwargs)
    apply_render_randomization(env, render_spec)
    env = CollectionStepLimit(env, args.max_episode_steps or TASKS[args.task_id].max_episode_steps)
    env = RecordEpisode(
        env,
        output_dir=osp.join(args.output_dir, "raw", args.task_id),
        trajectory_name=traj_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="RoboFPE/ManiSkill motion-planning solution for SFT collection",
        video_fps=FPS,
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


def _remove_record_files(h5_path: str) -> None:
    for path in [h5_path, h5_path.replace(".h5", ".json"), h5_path.replace(".h5", ".mp4")]:
        if osp.exists(path):
            os.remove(path)


def collect(args) -> None:
    spec = _validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        solve = _load_solution(args.task_id)
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import solver for {args.task_id}. Check RoboFPE/ManiSkill version compatibility. Original error: {exc}"
        ) from exc
    passed = 0
    seed = args.seed
    attempts = 0
    max_attempts = max(args.num_traj * args.max_attempts_per_traj, args.num_traj)
    successes: list[bool] = []
    raw_h5: list[str] = []
    randomization_rows: list[dict[str, Any]] = []
    pbar = tqdm(total=args.num_traj, desc=f"collect {args.task_id}")
    while passed < args.num_traj:
        if attempts >= max_attempts:
            raise RuntimeError(f"Collected {passed}/{args.num_traj} successes after {attempts} attempts.")
        attempts += 1
        token = f"{args.seed}:{args.task_id}:{attempts}:{passed}:{seed}"
        render_spec = sample_render_randomization_spec(token) if args.randomize_lighting else None
        camera_spec = _sample_camera_spec(args.task_id, token)
        traj_name = f"{args.task_id}_{passed:06d}_seed_{seed}"
        env = _make_env(args, traj_name, render_spec, camera_spec)
        h5_path = env._h5_file.filename
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
                )
                res, stage_records = unwrap_solution_result(res)
            except CollectionEpisodeTimeout:
                res = -1
            except Exception as exc:
                print(f"solution failed for seed={seed}: {exc}")
                res = -1
            success = False if res == -1 else solution_result_stats(res)[0]
            successes.append(bool(success))
            if args.success_only and not success:
                env.flush_trajectory(save=False)
                if args.save_video:
                    env.flush_video(save=False)
                env.close()
                _remove_record_files(h5_path)
                seed += 1
                continue
            env.flush_trajectory()
            if args.save_video:
                env.flush_video()
            env.close()
            metadata = {
                "task_id": args.task_id,
                "prompt": spec.prompt,
                "collect_mode": args.collect_mode,
                "robot_uids": args.robot_uids,
                "stage_records": stage_records,
                "collection_max_episode_steps": args.max_episode_steps or spec.max_episode_steps,
                "camera_randomization": camera_spec,
            }
            if render_spec is not None:
                metadata.update(render_randomization_metadata(render_spec, token))
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
        "num_successes": len(raw_h5),
        "attempts": attempts,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "raw_h5": raw_h5,
        "robot_uids": args.robot_uids,
        "training_cameras": ["observation.images.top", "observation.images.wrist"],
        "render_only_cameras": ["observation.images.render"],
    }
    with open(osp.join(args.output_dir, "collection_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(osp.join(args.output_dir, "randomization_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(randomization_rows, f, indent=2)
    if args.convert:
        cargs = argparse.Namespace(**vars(args))
        cargs.input = osp.join(args.output_dir, "raw", args.task_id)
        cargs.dataset_dir = osp.join(args.output_dir, "lerobot")
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


def _get_nested(group, path: list[str]) -> np.ndarray | None:
    cur: Any = group
    for part in path:
        if part not in cur:
            return None
        cur = cur[part]
    return cur[()]


def _camera_rgb(traj, camera_uid: str, t: int) -> np.ndarray:
    candidates = [
        ["obs", "sensor_data", camera_uid, "rgb"],
        ["obs", "image", camera_uid, "rgb"],
        ["obs", camera_uid, "rgb"],
    ]
    for path in candidates:
        data = _get_nested(traj, path)
        if data is not None:
            frame = _np_leaf(data[t])
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            return frame
    raise KeyError(f"Could not find RGB for camera {camera_uid}. Available obs keys: {list(traj.get(obs, {}).keys())}")


def _robot_state(traj: h5py.Group, t: int) -> np.ndarray:
    candidates = [
        ["obs", "agent", "qpos"],
        ["obs", "state", "agent", "qpos"],
    ]
    for path in candidates:
        data = _get_nested(traj, path)
        if data is not None:
            return _np_leaf(data[t]).astype(np.float32)
    # Fallback for RecordEpisode files collected with obs_mode=none.
    if "env_states" in traj:
        states = trajectory_utils.dict_to_list_of_dicts({k: v[()] for k, v in traj["env_states"].items()})
        state = states[min(t, len(states) - 1)]
        for key in ["agent", "articulations"]:
            if key in state:
                text = np.asarray(str(state[key]).encode("utf-8"), dtype=np.uint8).astype(np.float32)
                return text[:8]
    raise KeyError("Could not find robot qpos in trajectory observations; collect with --obs-mode rgb.")


def _resize(frame: np.ndarray, size: int) -> np.ndarray:
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


def _array_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def _features(state_dim: int, action_dim: int, image_size: int, total_videos: int) -> dict[str, Any]:
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
    return features


def _schema(state_dim: int, action_dim: int):
    import pyarrow as pa
    return pa.schema([
        pa.field("observation.state", pa.list_(pa.float32(), state_dim)),
        pa.field("actions", pa.list_(pa.float32(), action_dim)),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("task", pa.string()),
        pa.field("prompt", pa.string()),
    ])


def _write_episode_parquet(path: Path, rows: list[dict[str, Any]], state_dim: int, action_dim: int) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df, schema=_schema(state_dim, action_dim), preserve_index=False), path)


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


def convert(args) -> None:
    import h5py
    import pandas as pd
    _validate_args(args)
    h5_files = _iter_h5_files(args.input)
    if not h5_files:
        raise SystemExit(f"No .h5 files found under {args.input}")
    out = Path(args.dataset_dir)
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rows_all: list[dict[str, Any]] = []
    episodes_meta: list[dict[str, Any]] = []
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
                top_frames: list[np.ndarray] = []
                wrist_frames: list[np.ndarray] = []
                rows: list[dict[str, Any]] = []
                for t in range(T):
                    top = _resize(_camera_rgb(traj, TOP_CAMERA, t), args.image_size)
                    wrist = _resize(_camera_rgb(traj, WRIST_CAMERA, t), args.image_size)
                    state = _robot_state(traj, t).astype(np.float32)
                    action = actions[t].astype(np.float32)
                    state_dim = int(state.shape[-1])
                    action_dim = int(action.shape[-1])
                    top_frames.append(top)
                    wrist_frames.append(wrist)
                    rows.append({
                        "observation.state": state.tolist(),
                        "actions": action.tolist(),
                        "timestamp": float(t / FPS),
                        "frame_index": int(t),
                        "episode_index": int(episode_index),
                        "index": int(global_index),
                        "task_index": 0,
                        "task": task_prompt,
                        "prompt": task_prompt,
                    })
                    global_index += 1
                chunk = episode_index // CHUNK_SIZE
                _write_episode_parquet(out / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet", rows, state_dim, action_dim)
                for video_key, frames in [("observation.images.top", top_frames), ("observation.images.wrist", wrist_frames)]:
                    _write_video(out / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4", frames, FPS)
                rows_all.extend(rows)
                meta = _episode_metadata(sidecar, local_idx)
                episodes_meta.append({
                    "episode_index": episode_index,
                    "tasks": [task_prompt],
                    "length": T,
                    "source_h5": str(h5_path),
                    "metadata": meta,
                })
                episode_index += 1
    if state_dim is None or action_dim is None:
        raise RuntimeError("No frames converted")
    meta_dir = out / "meta"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")
    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"task_index": 0, "task": task_prompt}) + "\n")
    pd.DataFrame({"task_index": [0], "task": [task_prompt]}).to_parquet(meta_dir / "tasks.parquet", index=False)
    df_all = pd.DataFrame(rows_all)
    actions = np.stack(df_all["actions"].map(np.asarray).to_numpy()).astype(np.float32)
    states = np.stack(df_all["observation.state"].map(np.asarray).to_numpy()).astype(np.float32)
    stats = {
        "actions": _array_stats(actions),
        "observation.state": _array_stats(states),
    }
    with open(meta_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    data_size = sum(path.stat().st_size for path in out.rglob("data/**/*.parquet"))
    info = {
        "codebase_version": "v2.0",
        "dataset_variant": "robofpe-wrist-sft-v1",
        "task_id": args.task_id,
        "collect_mode": args.collect_mode,
        "robot_type": args.robot_uids,
        "total_episodes": episode_index,
        "total_frames": len(rows_all),
        "total_tasks": 1,
        "total_videos": episode_index * 2,
        "total_chunks": int((episode_index + CHUNK_SIZE - 1) // CHUNK_SIZE),
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "data_files_size_in_mb": int(data_size / (1024 * 1024)),
        "splits": {"train": f"0:{episode_index}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _features(state_dim, action_dim, args.image_size, episode_index * 2),
        "training_cameras": ["observation.images.top", "observation.images.wrist"],
        "render_only_cameras": ["observation.images.render"],
        "camera_uids": {"top": TOP_CAMERA, "wrist": WRIST_CAMERA, "render": RENDER_CAMERA},
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Wrote {episode_index} episodes to {out}")
    print(f"state_dim={state_dim}, action_dim={action_dim}, cameras=top:{TOP_CAMERA}, wrist:{WRIST_CAMERA}")


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
        for label, uid in [("top", TOP_CAMERA), ("wrist", WRIST_CAMERA)]:
            frame = sensor_data[uid]["rgb"]
            frame = _np_leaf(frame)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            cv2.imwrite(str(out / f"sample_{i:02d}_{label}.png"), cv2.cvtColor(_resize(frame, args.image_size), cv2.COLOR_RGB2BGR))
        try:
            render = env.render_rgb_array()
            if isinstance(render, list):
                render = render[0]
            cv2.imwrite(str(out / f"sample_{i:02d}_render.png"), cv2.cvtColor(render[..., :3], cv2.COLOR_RGB2BGR))
        except Exception as exc:
            print(f"render_camera preview failed for sample {i}: {exc}")
        env.close()
        rows.append({"sample": i, "camera_randomization": camera_spec, "render_randomization": render_spec})
    with open(out / "preview_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "task_id": args.task_id,
            "robot_uids": args.robot_uids,
            "training_camera_count": 2,
            "training_cameras": {"top": TOP_CAMERA, "wrist": WRIST_CAMERA},
            "render_only_cameras": [RENDER_CAMERA],
            "samples": rows,
        }, f, indent=2)
    print(f"Wrote preview images and camera report to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--task-id", default="PushCube-v1", choices=sorted(TASKS))
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
    p.add_argument("--success-only", action="store_true", default=True)
    p.add_argument("--max-attempts-per-traj", type=int, default=50)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--vis", action="store_true")
    p.add_argument("--convert", action="store_true", default=True)
    p.add_argument("--num-workers", type=int, default=1, help="Reserved for shard orchestration; v1 runs one process.")
    p.set_defaults(func=collect)

    p = sub.add_parser("convert")
    add_common(p)
    p.add_argument("--input", required=True, help="Raw .h5 file or directory")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
