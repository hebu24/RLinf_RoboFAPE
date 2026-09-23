# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate an embodied checkpoint without starting a training actor."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import ray
import torch.multiprocessing as mp
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate an RLinf embodied checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--config-name", default="maniskill_ppo_openpi_pi05")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--task-id", default="PutOnPlateInScene25Main-v3")
    parser.add_argument("--obj-set", default="train")
    parser.add_argument("--task-description")
    parser.add_argument("--num-eval-episodes", type=int, default=25)
    parser.add_argument("--num-envs", type=int, default=25)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default=None, help="Comma/range seed spec, e.g. 0-7 or 0,2,4. Reuses one actor/env initialization for this checkpoint.")
    parser.add_argument(
        "--parallel-seed-blocks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Assign one distinct seed to each parallel environment per rollout epoch.",
    )
    parser.add_argument(
        "--stop-after-failures",
        type=int,
        default=0,
        help="Stop a parallel-seed-block evaluation once this many success_once=false trajectories are collected.",
    )
    parser.add_argument("--gpu-ids", default="1")
    parser.add_argument("--obs-mode")
    parser.add_argument("--control-mode")
    parser.add_argument("--sim-backend")
    parser.add_argument("--init-params-json", default="{}")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument(
        "--save-episode-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--save-video", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ignore-terminations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fixed-reset-state-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--initial-state-manifest", help="Exact-pair reset manifest from build_exact_pair_manifest.py")
    parser.add_argument("--auto-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Maximum videos per environment worker; unset keeps config value.")
    args, hydra_overrides = parser.parse_known_args()
    invalid = [value for value in hydra_overrides if value.startswith("-")]
    if invalid:
        parser.error(f"Unrecognized arguments: {' '.join(invalid)}")
    return args, hydra_overrides


def _parse_gpu_placement(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _eval_component_placement(gpu_ids: str):
    """Place rollout on one GPU and each vectorized EnvWorker on its own GPU."""
    ids = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    if len(ids) >= 3:
        env_ids = [_parse_gpu_placement(item) for item in ids[1:]]
        if env_ids != list(range(env_ids[0], env_ids[-1] + 1)):
            raise ValueError("multi-worker env GPU ids must be a contiguous range")
        return OmegaConf.create(
            {
                "rollout": str(_parse_gpu_placement(ids[0])),
                "env": f"{env_ids[0]}-{env_ids[-1]}",
            }
        )
    if len(ids) == 2:
        return OmegaConf.create(
            {
                "rollout": _parse_gpu_placement(ids[0]),
                "env": _parse_gpu_placement(ids[1]),
            }
        )
    return OmegaConf.create({"env,rollout": _parse_gpu_placement(gpu_ids)})


def _parse_seed_spec(raw_value: str | None, default_seed: int) -> list[int]:
    if not raw_value:
        return [int(default_seed)]
    seeds: list[int] = []
    for piece in raw_value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = piece.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(piece))
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate evaluation seeds are forbidden: {seeds}")
    return sorted(seeds)


def _load_init_params(raw_value: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--init-params-json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--init-params-json must contain a JSON object")
    return value


def _validate_peg_insertion_eval_cfg(cfg: DictConfig, task_id: str) -> None:
    if task_id != "PegInsertionSide-v1":
        return
    wrap_obs_mode = cfg.env.eval.get("wrap_obs_mode", None)
    if wrap_obs_mode != "simple":
        raise ValueError(
            "PegInsertionSide-v1 evaluation must use "
            "env.eval.wrap_obs_mode=simple and the existing Side TCP profile."
        )
    model_cfg = cfg.rollout.model
    openpi_cfg = model_cfg.get("openpi", {})
    config_name = openpi_cfg.get("config_name", None)
    config_profiles = {
        "pi05_maniskill_peg_insertion_wrist": (2, True, False),
    }
    if config_name not in config_profiles:
        expected_config_names = ", ".join(sorted(config_profiles))
        raise ValueError(
            "PegInsertionSide-v1 evaluation must use the peg-insertion "
            "OpenPI config used by SFT. Expected rollout.model.openpi.config_name "
            f"to be one of {expected_config_names}, got {config_name!r}. "
            "Do not evaluate a peg-insertion SFT checkpoint with generic "
            "pi05_maniskill transforms/norm stats."
        )
    expected_images, expected_wrist, expected_wrist_back = config_profiles[config_name]
    num_images = int(openpi_cfg.get("num_images_in_input", -1))
    if num_images != expected_images:
        raise ValueError(
            "PegInsertionSide-v1 evaluation image count does not match "
            f"{config_name}: expected num_images_in_input={expected_images}, "
            f"got {num_images}."
        )
    use_wrist_image = bool(cfg.env.eval.get("use_wrist_image", False))
    if use_wrist_image != expected_wrist:
        raise ValueError(
            "PegInsertionSide-v1 evaluation wrist image routing does not "
            f"match {config_name}: expected env.eval.use_wrist_image={expected_wrist}, "
            f"got {use_wrist_image}."
        )
    use_wrist_back_image = bool(cfg.env.eval.get("use_wrist_back_image", False))
    if use_wrist_back_image != expected_wrist_back:
        raise ValueError(
            "PegInsertionSide-v1 evaluation back-wrist image routing does not "
            f"match {config_name}: expected "
            f"env.eval.use_wrist_back_image={expected_wrist_back}, got "
            f"{use_wrist_back_image}."
        )
    num_action_chunks = int(model_cfg.get("num_action_chunks", -1))
    action_horizon = int(openpi_cfg.get("action_horizon", num_action_chunks))
    if num_action_chunks != 10 or action_horizon != 10:
        raise ValueError(
            "PegInsertionSide-v1 SFT uses 10-step action chunks. Expected "
            "rollout.model.num_action_chunks=10 and "
            f"rollout.model.openpi.action_horizon=10, got {num_action_chunks} "
            f"and {action_horizon}."
        )
    policy_setup = str(model_cfg.get("policy_setup", ""))
    if policy_setup != "panda-ee-target-dpose":
        raise ValueError(
            "PegInsertionSide-v1 evaluation must use "
            "rollout.model.policy_setup=panda-ee-target-dpose (use_target=True, "
            "target-delta labels) so physical TCP actions are converted for "
            "the ManiSkill pd_ee_target_delta_pose controller. "
            f"Got {policy_setup!r}."
        )
    control_mode = str(cfg.env.eval.init_params.get("control_mode", ""))
    if control_mode != "pd_ee_target_delta_pose":
        raise ValueError(
            "PegInsertionSide-v1 evaluation must use "
            "env.eval.init_params.control_mode=pd_ee_target_delta_pose. "
            f"Got {control_mode!r}."
        )


def _validate_peginsertionvertical_eval_cfg(cfg: DictConfig, task_id: str) -> None:
    if task_id != "PegInsertionVertical-v1":
        return
    env_cfg = cfg.env.eval
    model_cfg = cfg.rollout.model
    openpi_cfg = model_cfg.get("openpi", {})
    expected_config = (
        "run_train/peginsertion_maniskill_pi0.5/config/"
        "maniskill_peg_insertion_vertical_wrist_sft_eval_openpi_pi05.yaml"
    )

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(
                "PegInsertionVertical-v1 eval contract violation: "
                f"{message}. Use {expected_config}."
            )

    def require_equal(name: str, actual: Any, expected: Any) -> None:
        require(
            actual == expected,
            f"{name} expected {expected!r}, got {actual!r}",
        )

    require_equal("env.eval.wrap_obs_mode", env_cfg.get("wrap_obs_mode"), "simple")
    require_equal("env.eval.main_camera_key", env_cfg.get("main_camera_key"), "render_camera")
    require_equal("env.eval.use_wrist_image", bool(env_cfg.get("use_wrist_image", False)), True)
    require_equal("env.eval.use_wrist_back_image", bool(env_cfg.get("use_wrist_back_image", False)), False)
    require_equal("env.eval.capture_render_image_for_reward", bool(env_cfg.get("capture_render_image_for_reward", False)), False)
    require_equal("env.eval.reset_options.pre_grasped", bool(env_cfg.get("reset_options", {}).get("pre_grasped", True)), False)
    require_equal("env.eval.reset_options.randomize_initial_poses", bool(env_cfg.get("reset_options", {}).get("randomize_initial_poses", False)), True)
    require_equal("env.eval.init_params.id", env_cfg.init_params.get("id"), task_id)
    require_equal("env.eval.init_params.robot_uids", env_cfg.init_params.get("robot_uids"), "panda_wristcam")
    require_equal("env.eval.init_params.control_mode", env_cfg.init_params.get("control_mode"), "pd_joint_pos")
    require_equal("env.eval.max_episode_steps", int(env_cfg.get("max_episode_steps", -1)), 450)
    require_equal("env.eval.max_steps_per_rollout_epoch", int(env_cfg.get("max_steps_per_rollout_epoch", -1)), 450)
    require_equal("env.eval.task_description", str(env_cfg.get("task_description", "")), "Insert the peg vertically into the target hole.")
    require_equal("rollout.model.action_dim", int(model_cfg.get("action_dim", -1)), 8)
    require_equal("rollout.model.num_action_chunks", int(model_cfg.get("num_action_chunks", -1)), 10)
    require_equal("rollout.model.openpi.config_name", openpi_cfg.get("config_name"), "pi05_maniskill_wrist")
    require_equal("rollout.model.openpi.num_images_in_input", int(openpi_cfg.get("num_images_in_input", -1)), 2)
    require_equal("rollout.model.openpi.action_horizon", int(openpi_cfg.get("action_horizon", -1)), 10)
    require_equal("rollout.model.policy_setup", str(model_cfg.get("policy_setup", "")), "panda")
    require_equal("env.eval.use_pi05_proprio", bool(env_cfg.get("use_pi05_proprio", False)), False)
    render_cfg = env_cfg.init_params.get("human_render_camera_configs", {})
    pose = render_cfg.get("render_camera", {}).get("pose")
    expected_pose = [0.7054, -0.086655, 0.686691, 0.025112, -0.237384, -0.03364, 0.970508]
    require(pose is not None and len(pose) == 7, f"render_camera pose expected 7 values, got {pose!r}")
    require(all(abs(float(actual) - expected) < 1e-5 for actual, expected in zip(pose, expected_pose)), f"render_camera pose expected {expected_pose!r}, got {list(pose)!r}")


def _validate_stackcube_eval_cfg(cfg: DictConfig, task_id: str) -> None:
    if task_id != "StackCube-v1":
        return
    env_cfg = cfg.env.eval
    model_cfg = cfg.rollout.model
    if env_cfg.get("main_camera_key") != "render_camera":
        raise ValueError("StackCube eval must use main_camera_key=render_camera.")
    if not bool(env_cfg.get("use_wrist_image", False)):
        raise ValueError("StackCube eval must enable the hand_camera wrist image.")
    if env_cfg.init_params.get("control_mode") != "pd_joint_pos":
        raise ValueError("StackCube eval must use 8D pd_joint_pos control.")
    if int(model_cfg.get("action_dim", -1)) != 8 or int(model_cfg.get("num_action_chunks", -1)) != 10:
        raise ValueError("StackCube eval requires action_dim=8 and num_action_chunks=10.")
    render_cfg = env_cfg.init_params.get("human_render_camera_configs", {})
    pose = render_cfg.get("render_camera", {}).get("pose")
    try:
        valid_pose = len(pose) == 7
    except TypeError:
        valid_pose = False
    if not valid_pose:
        raise ValueError(
            "StackCube eval requires human_render_camera_configs.render_camera.pose; "
            "a top-level pose is ignored by ManiSkill."
        )


def _validate_liftpegupright_eval_cfg(cfg: DictConfig, task_id: str) -> None:
    if task_id != "LiftPegUpright-box":
        return
    env_cfg = cfg.env.eval
    model_cfg = cfg.rollout.model
    openpi_cfg = model_cfg.get("openpi", {})
    if env_cfg.get("main_camera_key") != "render_camera":
        raise ValueError("LiftPegUpright-box eval must use main_camera_key=render_camera.")
    if not bool(env_cfg.get("use_wrist_image", False)) or bool(
        env_cfg.get("use_wrist_back_image", False)
    ):
        raise ValueError(
            "LiftPegUpright-box eval requires render_camera + hand_camera only "
            "(use_wrist_image=true, use_wrist_back_image=false)."
        )
    if env_cfg.init_params.get("control_mode") != "pd_joint_pos":
        raise ValueError("LiftPegUpright-box eval requires pd_joint_pos control.")
    if int(model_cfg.get("action_dim", -1)) != 8:
        raise ValueError("LiftPegUpright-box eval requires action_dim=8.")
    if int(model_cfg.get("num_action_chunks", -1)) != 10:
        raise ValueError("LiftPegUpright-box eval requires num_action_chunks=10.")
    if openpi_cfg.get("config_name") != "pi05_maniskill_wrist":
        raise ValueError(
            "LiftPegUpright-box eval requires openpi.config_name=pi05_maniskill_wrist."
        )
    if int(openpi_cfg.get("num_images_in_input", -1)) != 2:
        raise ValueError("LiftPegUpright-box eval requires num_images_in_input=2.")
    if int(env_cfg.get("max_episode_steps", -1)) != 350:
        raise ValueError("LiftPegUpright-box eval requires max_episode_steps=350.")


def _validate_uprightstack_eval_cfg(cfg: DictConfig, task_id: str) -> None:
    if task_id != "UprightStack-v1":
        return
    env_cfg = cfg.env.eval
    model_cfg = cfg.rollout.model
    openpi_cfg = model_cfg.get("openpi", {})
    if env_cfg.get("task_description") != "Stand the brick upright and stack it on the red cube.":
        raise ValueError(
            "UprightStack-v1 eval must use the RoboFPE SFT prompt exactly: "
            "'Stand the brick upright and stack it on the red cube.'"
        )
    if env_cfg.get("main_camera_key") != "render_camera":
        raise ValueError("UprightStack-v1 eval must use main_camera_key=render_camera.")
    if not bool(env_cfg.get("use_wrist_image", False)) or bool(
        env_cfg.get("use_wrist_back_image", False)
    ):
        raise ValueError(
            "UprightStack-v1 eval requires render_camera + hand_camera only."
        )
    if env_cfg.init_params.get("control_mode") != "pd_joint_pos":
        raise ValueError("UprightStack-v1 eval requires pd_joint_pos control.")
    if env_cfg.init_params.get("robot_uids") != "panda_wristcam":
        raise ValueError("UprightStack-v1 eval requires robot_uids=panda_wristcam.")
    if env_cfg.init_params.get("obs_mode") != "rgb":
        raise ValueError("UprightStack-v1 eval requires obs_mode=rgb.")
    if env_cfg.init_params.get("sim_backend") != "gpu":
        raise ValueError("UprightStack-v1 eval requires sim_backend=gpu.")
    if int(model_cfg.get("action_dim", -1)) != 8:
        raise ValueError("UprightStack-v1 eval requires action_dim=8.")
    if int(model_cfg.get("num_action_chunks", -1)) != 10:
        raise ValueError("UprightStack-v1 eval requires num_action_chunks=10.")
    if openpi_cfg.get("config_name") != "pi05_maniskill_wrist":
        raise ValueError("UprightStack-v1 eval requires pi05_maniskill_wrist.")
    if int(openpi_cfg.get("num_images_in_input", -1)) != 2:
        raise ValueError("UprightStack-v1 eval requires num_images_in_input=2.")
    if int(env_cfg.get("max_episode_steps", -1)) != 1000:
        raise ValueError("UprightStack-v1 eval requires max_episode_steps=1000.")
    render_cfg = env_cfg.init_params.get("human_render_camera_configs", {})
    if "render_camera" not in render_cfg:
        raise ValueError("UprightStack-v1 eval requires a named render_camera config.")
    render_camera = render_cfg["render_camera"]
    if int(render_camera.get("width", -1)) != 512 or int(render_camera.get("height", -1)) != 512:
        raise ValueError("UprightStack-v1 eval requires a 512x512 render_camera.")
    sensors = env_cfg.init_params.get("sensor_configs", {})
    for camera_name in ("base_camera", "hand_camera"):
        camera = sensors.get(camera_name, {})
        if int(camera.get("width", -1)) != 224 or int(camera.get("height", -1)) != 224:
            raise ValueError(
                f"UprightStack-v1 eval requires {camera_name} to be 224x224."
            )


def _apply_uprightstack_randomization(cfg: DictConfig, args: argparse.Namespace) -> None:
    """Reuse RoboFPE's camera/light sampler for each eval seed."""
    if args.task_id != "UprightStack-v1":
        return
    if os.environ.get("UPRIGHTSTACK_RANDOMIZATION", "true").lower() in {"0", "false", "no"}:
        return
    try:
        from render_domain_randomization import sample_render_randomization_spec
        from run_train.eval_checkpoint.uprightstack_randomization import (
            sample_uprightstack_camera_spec,
        )
    except ImportError as exc:
        raise RuntimeError(
            "UprightStack randomized eval requires RoboFPE collector helpers on PYTHONPATH"
        ) from exc

    token = f"eval:{args.task_id}:{int(args.seed)}"
    camera_spec = sample_uprightstack_camera_spec(token)
    lighting_spec = sample_render_randomization_spec(token)
    with open_dict(cfg):
        sensors = cfg.env.eval.init_params.sensor_configs
        sensors.base_camera.pose = camera_spec["base"]["pose"]
        sensors.hand_camera.pose = camera_spec["wrist"]["pose"]
        cfg.env.eval.init_params.human_render_camera_configs.render_camera.pose = camera_spec["render"]["pose"]
        cfg.env.eval.render_randomization_spec = lighting_spec
        cfg.env.eval.randomization_token = token


def _validate_pullcubetool_golf_eval_cfg(cfg: DictConfig, task_id: str) -> None:
    if task_id != "PullCubeTool-golf":
        return
    env_cfg, model_cfg = cfg.env.eval, cfg.rollout.model
    openpi_cfg = model_cfg.get("openpi", {})
    expected_prompt = "Use the L-shaped tool to pull the golf ball into the robot's reachable target region."
    expected_pose = [0.85, -0.5, 0.6, 0.26071918, -0.11947089, 0.03253291, 0.95744133]
    def require(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            raise ValueError(f"PullCubeTool-golf eval {name} expected {expected!r}, got {actual!r}")
    require("main_camera_key", env_cfg.get("main_camera_key"), "render_camera")
    require("use_wrist_image", bool(env_cfg.get("use_wrist_image", False)), True)
    require("use_wrist_back_image", bool(env_cfg.get("use_wrist_back_image", False)), False)
    require("task_description", str(env_cfg.get("task_description", "")), expected_prompt)
    require("robot_uids", env_cfg.init_params.get("robot_uids"), "panda_wristcam")
    require("control_mode", env_cfg.init_params.get("control_mode"), "pd_joint_pos")
    require("max_episode_steps", int(env_cfg.get("max_episode_steps", -1)), 350)
    require("action_dim", int(model_cfg.get("action_dim", -1)), 8)
    require("num_action_chunks", int(model_cfg.get("num_action_chunks", -1)), 10)
    require("openpi.config_name", openpi_cfg.get("config_name"), "pi05_maniskill_wrist")
    require("openpi.num_images_in_input", int(openpi_cfg.get("num_images_in_input", -1)), 2)
    require("openpi.mask_gripper_loss", bool(openpi_cfg.get("mask_gripper_loss", True)), False)
    pose = env_cfg.init_params.get("human_render_camera_configs", {}).get("render_camera", {}).get("pose")
    if pose is None or len(pose) != 7 or any(abs(float(a) - b) > 1e-5 for a, b in zip(pose, expected_pose)):
        raise ValueError(f"PullCubeTool-golf eval requires named fixed render_camera pose {expected_pose!r}, got {pose!r}")


def _validate_pullcubetool_golf_placement(args: argparse.Namespace) -> None:
    if args.task_id != "PullCubeTool-golf":
        return
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if len(gpu_ids) != 2 or gpu_ids[0] == gpu_ids[1]:
        raise ValueError(
            "PullCubeTool-golf eval requires exactly two distinct --gpu-ids: "
            "rollout first and the isolated Vulkan env worker second."
        )


def build_config(args: argparse.Namespace, hydra_overrides: list[str]) -> DictConfig:
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    config_dir = Path(args.config_dir).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not config_dir.is_dir():
        raise NotADirectoryError(f"Config directory does not exist: {config_dir}")
    if args.num_eval_episodes <= 0 or args.num_envs <= 0:
        raise ValueError("num_eval_episodes and num_envs must be positive")
    if args.num_eval_episodes % args.num_envs != 0:
        raise ValueError(
            "num_eval_episodes must be divisible by num_envs because RLinf "
            "evaluates a fixed-size parallel batch each rollout epoch"
        )
    _validate_pullcubetool_golf_placement(args)

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=hydra_overrides)

    actor_model = cfg.get("actor", {}).get("model")
    if actor_model is None:
        raise ValueError(
            "The base config must define actor.model so it can be reused for rollout"
        )

    init_params = _load_init_params(args.init_params_json)
    with open_dict(cfg):
        # Training configs keep most OpenPI fields under actor.model. Evaluation
        # workers need the same complete model config under rollout.model.
        cfg.rollout.model = OmegaConf.merge(actor_model, cfg.rollout.model)
        cfg.rollout.model.model_path = str(checkpoint_path)

        cfg.runner.task_type = "embodied_eval"
        cfg.runner.only_eval = True
        cfg.runner.val_check_interval = -1
        cfg.runner.save_interval = -1
        cfg.runner.resume_dir = None
        cfg.runner.ckpt_path = None
        cfg.runner.logger.log_path = str(Path(args.log_dir).expanduser().resolve())
        cfg.runner.logger.experiment_name = f"eval-{args.task_id}"

        cfg.cluster.component_placement = _eval_component_placement(args.gpu_ids)

        cfg.env.eval.rollout_epoch = args.num_eval_episodes // args.num_envs
        cfg.env.eval.total_num_envs = args.num_envs
        cfg.env.eval.auto_reset = args.auto_reset
        cfg.env.eval.ignore_terminations = args.ignore_terminations
        cfg.env.eval.use_fixed_reset_state_ids = args.fixed_reset_state_ids
        if args.initial_state_manifest:
            manifest = Path(args.initial_state_manifest).expanduser().resolve()
            if not manifest.is_file():
                raise FileNotFoundError(f"Initial-state manifest does not exist: {manifest}")
            cfg.env.eval.initial_state_manifest = str(manifest)
            cfg.env.eval.auto_reset = False
        cfg.env.eval.is_eval = True
        cfg.env.eval.seed = args.seed
        cfg.env.eval.max_episode_steps = args.max_episode_steps
        cfg.env.eval.max_steps_per_rollout_epoch = args.max_episode_steps
        cfg.env.eval.action_scale = args.action_scale
        cfg.env.eval.video_cfg.save_video = args.save_video
        cfg.env.eval.video_cfg.video_base_dir = str(
            Path(args.log_dir).expanduser().resolve() / "video" / "eval"
        )
        if args.max_videos is not None:
            if args.max_videos < 0:
                raise ValueError("--max-videos must be non-negative")
            cfg.env.eval.video_cfg.max_videos = args.max_videos
        cfg.env.eval.init_params.id = args.task_id
        cfg.env.eval.init_params.max_episode_steps = args.max_episode_steps
        if args.obj_set:
            cfg.env.eval.init_params.obj_set = args.obj_set
        elif "obj_set" in cfg.env.eval.init_params:
            del cfg.env.eval.init_params.obj_set
        if args.task_description:
            cfg.env.eval.task_description = args.task_description
        if args.obs_mode:
            cfg.env.eval.init_params.obs_mode = args.obs_mode
        if args.control_mode:
            cfg.env.eval.init_params.control_mode = args.control_mode
        if args.sim_backend:
            cfg.env.eval.init_params.sim_backend = args.sim_backend
        cfg.env.eval.init_params = OmegaConf.merge(
            cfg.env.eval.init_params, init_params
        )
        _apply_uprightstack_randomization(cfg, args)
        if args.task_id in {"PegInsertionVertical-v1", "PegInsertionSide-v1"}:
            # RoboFPE stores observation.images.top from render_camera and
            # observation.images.wrist from hand_camera for both peg tasks.
            cfg.env.eval.main_camera_key = "render_camera"
            cfg.env.eval.execute_action_chunks = 10

        if args.task_id == "PegInsertionSide-v1":
            # Side has no hand_camera_back sensor.
            sensor_configs = cfg.env.eval.init_params.get("sensor_configs")
            if sensor_configs is not None and "hand_camera_back" in sensor_configs:
                del sensor_configs.hand_camera_back
            cfg.env.eval.reset_options.pre_grasped = False
            cfg.env.eval.capture_render_image_for_reward = False
            cfg.env.eval.ignore_terminations = False
            cfg.env.eval.use_pi05_proprio = True
            cfg.env.eval.main_camera_key = "render_camera"

    num_action_chunks = int(cfg.rollout.model.num_action_chunks)
    if args.max_episode_steps % num_action_chunks != 0:
        raise ValueError(
            f"max_episode_steps ({args.max_episode_steps}) must be divisible by "
            f"num_action_chunks ({num_action_chunks})"
        )
    cfg = validate_cfg(cfg)
    if args.control_mode:
        # validate_cfg derives a mode from policy_setup; an explicit CLI value wins.
        with open_dict(cfg):
            cfg.env.eval.init_params.control_mode = args.control_mode
    _validate_peg_insertion_eval_cfg(cfg, args.task_id)
    _validate_peginsertionvertical_eval_cfg(cfg, args.task_id)
    _validate_stackcube_eval_cfg(cfg, args.task_id)
    _validate_liftpegupright_eval_cfg(cfg, args.task_id)
    _validate_uprightstack_eval_cfg(cfg, args.task_id)
    _validate_pullcubetool_golf_eval_cfg(cfg, args.task_id)
    return cfg


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _evaluate_with_existing_runner(
    cfg: DictConfig,
    runner: EmbodiedEvalRunner,
    *,
    log_step: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env_handle = runner.env.evaluate(
        input_channel=runner.env_channel,
        rollout_channel=runner.rollout_channel,
    )
    rollout_handle = runner.rollout.evaluate(
        input_channel=runner.rollout_channel,
        output_channel=runner.env_channel,
    )
    env_results = env_handle.wait()
    env_decoupled_mode = cfg.runner.get("enable_decoupled_mode", False)
    if not env_decoupled_mode:
        rollout_handle.wait()
    eval_metrics_list = [results for results in env_results if results is not None]
    metrics = {
        key: _json_value(value)
        for key, value in compute_evaluate_metrics(eval_metrics_list).items()
    }
    prefixed_metrics = {f"eval/{key}": value for key, value in metrics.items()}
    runner.logger.info(prefixed_metrics)
    runner.metric_logger.log(step=log_step, data=prefixed_metrics)
    return metrics, eval_metrics_list


def _run_eval_many_seeds(
    cfg: DictConfig,
    seeds: list[int],
    base_log_dir: Path,
    *,
    save_episode_metrics_enabled: bool = False,
    parallel_seed_blocks: bool = False,
    stop_after_failures: int = 0,
) -> list[tuple[int, Path, dict[str, Any], list[dict[str, Any]]]]:
    rollout_group = None
    env_group = None
    runner = None
    try:
        cluster = Cluster(cluster_cfg=cfg.cluster)
        placement = HybridComponentPlacement(cfg, cluster)
        rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=placement.get_strategy("rollout"),
        )
        env_group = EnvWorker.create_group(cfg).launch(
            cluster,
            name=cfg.env.group_name,
            placement_strategy=placement.get_strategy("env"),
        )
        runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout_group, env=env_group)
        runner.init_workers()
        results: list[tuple[int, Path, dict[str, Any], list[dict[str, Any]]]] = []
        if parallel_seed_blocks and len(seeds) % int(cfg.env.eval.total_num_envs) != 0:
            raise ValueError(
                "--parallel-seed-blocks requires a seed count divisible by num-envs"
            )
        seed_blocks = (
            [seeds[index : index + int(cfg.env.eval.total_num_envs)]
             for index in range(0, len(seeds), int(cfg.env.eval.total_num_envs))]
            if parallel_seed_blocks
            else [[seed] for seed in seeds]
        )
        failures_collected = 0
        for block in seed_blocks:
            seed = block[0]
            seed_log_dir = base_log_dir / f"seed_{seed}"
            seed_log_dir.mkdir(parents=True, exist_ok=True)
            with open_dict(cfg):
                cfg.env.eval.seed = int(seed)
                cfg.runner.logger.log_path = str(seed_log_dir)
                cfg.env.eval.video_cfg.video_base_dir = str(seed_log_dir / "video" / "eval")
            if parallel_seed_blocks:
                for rank, env_seed in enumerate(block):
                    runner.env.execute_on(rank).set_eval_seed(int(env_seed)).wait()
            else:
                runner.env.set_eval_seed(int(seed)).wait()
            metrics, eval_metrics_list = _evaluate_with_existing_runner(
                cfg, runner, log_step=int(seed)
            )
            if save_episode_metrics_enabled:
                episode_metrics_path = save_episode_metrics(eval_metrics_list, seed_log_dir)
                print(f'Episode metrics seed {seed}: {episode_metrics_path}')
            results.append((int(seed), seed_log_dir, metrics, eval_metrics_list))
            if stop_after_failures:
                success_once = _serialize_episode_metrics(eval_metrics_list).get(
                    "success_once", []
                )
                failures_collected += sum(not bool(value) for value in success_once)
                if failures_collected >= stop_after_failures:
                    break
        return results
    finally:
        if runner is not None:
            runner.metric_logger.finish()
        if env_group is not None:
            env_group.close()
        if rollout_group is not None:
            rollout_group.close()
        if runner is not None:
            runner.env_channel.close()
            runner.rollout_channel.close()
        if ray.is_initialized():
            _driver_started_ray = os.environ.get("RLINF_EVAL_STARTED_RAY", "") == "1"
            if _driver_started_ray:
                ray.shutdown()


def _run_eval(cfg: DictConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rollout_group = None
    env_group = None
    runner = None
    try:
        cluster = Cluster(cluster_cfg=cfg.cluster)
        placement = HybridComponentPlacement(cfg, cluster)
        rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=placement.get_strategy("rollout"),
        )
        env_group = EnvWorker.create_group(cfg).launch(
            cluster,
            name=cfg.env.group_name,
            placement_strategy=placement.get_strategy("env"),
        )
        runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout_group, env=env_group)
        runner.init_workers()
        env_handle = runner.env.evaluate(
            input_channel=runner.env_channel,
            rollout_channel=runner.rollout_channel,
        )
        rollout_handle = runner.rollout.evaluate(
            input_channel=runner.rollout_channel,
            output_channel=runner.env_channel,
        )
        env_results = env_handle.wait()
        env_decoupled_mode = cfg.runner.get("enable_decoupled_mode", False)
        if not env_decoupled_mode:
            rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        metrics = {
            key: _json_value(value)
            for key, value in compute_evaluate_metrics(eval_metrics_list).items()
        }
        prefixed_metrics = {f"eval/{key}": value for key, value in metrics.items()}
        runner.logger.info(prefixed_metrics)
        runner.metric_logger.log(step=0, data=prefixed_metrics)
        return metrics, eval_metrics_list
    finally:
        if runner is not None:
            runner.metric_logger.finish()
        if env_group is not None:
            env_group.close()
        if rollout_group is not None:
            rollout_group.close()
        if runner is not None:
            runner.env_channel.close()
            runner.rollout_channel.close()
        if ray.is_initialized():
            # Only shut down if this process started the Ray cluster.
            # If Ray was already running before we connected (address="auto"),
            # shutting down would kill the shared training cluster.
            _driver_started_ray = os.environ.get("RLINF_EVAL_STARTED_RAY", "") == "1"
            if _driver_started_ray:
                ray.shutdown()


def _serialize_episode_metrics(
    eval_metrics_list: list[dict[str, Any]],
) -> dict[str, Any]:
    def _as_cpu(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        return value

    episode_metrics: dict[str, Any] = {}
    for key in ("success_once", "success_at_end", "return", "reward", "max_reward", "episode_len", "manifest_reference_index"):
        shards = [metrics[key] for metrics in eval_metrics_list if key in metrics]
        if not shards:
            continue
        values = []
        for shard in shards:
            if hasattr(shard, "detach"):
                shard = shard.detach().cpu()
            if hasattr(shard, "reshape"):
                shard = shard.reshape(-1)
            if hasattr(shard, "tolist"):
                values.extend(shard.tolist())
            else:
                values.append(shard)
        episode_metrics[key] = values

    for key in (
        "eef_pos", "cube_pos", "goal_pos", "push_success",
        "cube_a_pos", "cube_b_pos", "cube_a_grasped", "stack_success",
        "executed_actions",
        "left_ball_force", "right_ball_force", "left_ball_angle",
        "right_ball_angle", "left_ball_contact", "right_ball_contact",
        "gripper_width", "ball_height", "ball_to_tcp_distance",
        "agent_is_grasping", "grasp_confirmed_raw", "grasp_confirmed",
        "lift_confirmed",
        "tool_pos", "ball_velocity", "tool_grasp_raw", "tool_grasp_confirmed",
        "ball_forward_velocity", "pull_success",
    ):
        values = []
        for metrics in eval_metrics_list:
            if key not in metrics:
                continue
            shard = _as_cpu(metrics[key])
            if not hasattr(shard, "shape") or len(shard.shape) < 2:
                continue
            lengths = metrics.get("episode_len")
            lengths = _as_cpu(lengths) if lengths is not None else None
            for traj_idx in range(int(shard.shape[0])):
                if lengths is not None and hasattr(lengths, "reshape"):
                    length = int(lengths.reshape(-1)[traj_idx].item())
                    length = max(0, min(length, int(shard.shape[1])))
                else:
                    length = int(shard.shape[1])
                trace = shard[traj_idx, :length]
                if hasattr(trace, "tolist"):
                    values.append(trace.tolist())
                else:
                    values.append(trace)
        if values:
            episode_metrics[key] = values

    episode_metrics["num_trajectories"] = (
        len(episode_metrics.get("return", []))
        or len(episode_metrics.get("reward", []))
        or len(episode_metrics.get("success_once", []))
    )
    return episode_metrics


def evaluate(cfg: DictConfig) -> dict[str, Any]:
    metrics, _ = _run_eval(cfg)
    return metrics


def save_episode_metrics(
    eval_metrics_list: list[dict[str, Any]],
    log_dir: Path,
) -> Path:
    episode_metrics = _serialize_episode_metrics(eval_metrics_list)
    episode_metrics_path = log_dir / "trajectory_metrics.json"
    episode_metrics_path.write_text(
        json.dumps(episode_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return episode_metrics_path


def main() -> None:
    args, hydra_overrides = parse_args()
    cfg = build_config(args, hydra_overrides)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(resolved_cfg, indent=2))

    requested_seeds = _parse_seed_spec(args.seeds, args.seed)
    checkpoint_path = str(Path(args.checkpoint_path).expanduser().resolve())

    if args.seeds:
        base_log_dir = Path(cfg.runner.logger.log_path)
        results = _run_eval_many_seeds(
            cfg,
            requested_seeds,
            base_log_dir,
            save_episode_metrics_enabled=args.save_episode_metrics,
            parallel_seed_blocks=args.parallel_seed_blocks,
            stop_after_failures=args.stop_after_failures,
        )
        summaries = []
        for seed, seed_log_dir, metrics, eval_metrics_list in results:
            if args.save_episode_metrics:
                episode_metrics_path = save_episode_metrics(eval_metrics_list, seed_log_dir)
                print(f"Episode metrics seed {seed}: {episode_metrics_path}")
            summary = {
                "checkpoint_path": checkpoint_path,
                "task_id": args.task_id,
                "seed": seed,
                "num_eval_episodes_requested": args.num_eval_episodes,
                "metrics": metrics,
            }
            actual_episodes = metrics.get("num_trajectories")
            if actual_episodes != args.num_eval_episodes:
                print(
                    "Warning: evaluator returned "
                    f"{actual_episodes} trajectories for seed {seed}; requested {args.num_eval_episodes}."
                )
            summary_path = seed_log_dir / "evaluation_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(f"Evaluation summary seed {seed}: {summary_path}")
            summaries.append(summary)
        combined = {
            "checkpoint_path": checkpoint_path,
            "task_id": args.task_id,
            "seeds": requested_seeds,
            "summaries": summaries,
        }
        base_log_dir.mkdir(parents=True, exist_ok=True)
        combined_path = base_log_dir / "evaluation_summary_multiseed.json"
        combined_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
        print(f"Combined evaluation summary: {combined_path}")
        print(json.dumps(combined, indent=2))
        return

    metrics, eval_metrics_list = _run_eval(cfg)
    if args.save_episode_metrics:
        episode_metrics_path = save_episode_metrics(
            eval_metrics_list, Path(cfg.runner.logger.log_path)
        )
        print(f"Episode metrics: {episode_metrics_path}")
    summary = {
        "checkpoint_path": checkpoint_path,
        "task_id": args.task_id,
        "num_eval_episodes_requested": args.num_eval_episodes,
        "metrics": metrics,
    }
    actual_episodes = metrics.get("num_trajectories")
    if actual_episodes != args.num_eval_episodes:
        print(
            "Warning: evaluator returned "
            f"{actual_episodes} trajectories; requested {args.num_eval_episodes}."
        )
    log_dir = Path(cfg.runner.logger.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
