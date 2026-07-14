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
    args, hydra_overrides = parser.parse_known_args()
    invalid = [value for value in hydra_overrides if value.startswith("-")]
    if invalid:
        parser.error(f"Unrecognized arguments: {' '.join(invalid)}")
    return args, hydra_overrides


def _parse_gpu_placement(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _load_init_params(raw_value: str) -> dict[str, Any]:
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--init-params-json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--init-params-json must contain a JSON object")
    return value


def _validate_peg_insertion_eval_cfg(cfg: DictConfig, task_id: str) -> None:
    if task_id != "PegInsertionVertical-v1":
        return
    wrap_obs_mode = cfg.env.eval.get("wrap_obs_mode", None)
    if wrap_obs_mode != "simple":
        raise ValueError(
            "PegInsertionVertical-v1 evaluation must use "
            "env.eval.wrap_obs_mode=simple so the policy receives base_camera "
            "images and aligned pi0.5 proprio. Use the peg-insertion config "
            "run_train/peginsertion_maniskill_pi0.5/config/"
            "maniskill_peg_insertion_vertical_ppo_openpi_pi05.yaml instead of "
            "a generic ManiSkill config."
        )
    model_cfg = cfg.rollout.model
    openpi_cfg = model_cfg.get("openpi", {})
    config_name = openpi_cfg.get("config_name", None)
    if config_name not in {
        "pi05_maniskill_peg_insertion",
        "pi05_maniskill_peg_insertion_wrist",
        "pi05_maniskill_peg_insertion_actual_ee",
    }:
        raise ValueError(
            "PegInsertionVertical-v1 evaluation must use the peg-insertion "
            "OpenPI config used by SFT. Expected rollout.model.openpi.config_name "
            "to be pi05_maniskill_peg_insertion, "
            "pi05_maniskill_peg_insertion_wrist, or "
            "pi05_maniskill_peg_insertion_actual_ee, got "
            f"{config_name!r}. "
            "Do not evaluate a peg-insertion SFT checkpoint with generic "
            "pi05_maniskill transforms/norm stats."
        )
    num_images = int(openpi_cfg.get("num_images_in_input", -1))
    is_wrist = config_name in {
        "pi05_maniskill_peg_insertion_wrist",
        "pi05_maniskill_peg_insertion_actual_ee",
    }
    expected_images = 2 if is_wrist else 1
    if num_images != expected_images:
        raise ValueError(
            "PegInsertionVertical-v1 evaluation image count does not match "
            f"{config_name}: expected num_images_in_input={expected_images}, "
            f"got {num_images}."
        )
    use_wrist_image = bool(cfg.env.eval.get("use_wrist_image", False))
    if use_wrist_image != is_wrist:
        raise ValueError(
            "PegInsertionVertical-v1 evaluation wrist image routing does not "
            f"match {config_name}: expected env.eval.use_wrist_image={is_wrist}, "
            f"got {use_wrist_image}."
        )
    num_action_chunks = int(model_cfg.get("num_action_chunks", -1))
    action_horizon = int(openpi_cfg.get("action_horizon", num_action_chunks))
    if num_action_chunks != 10 or action_horizon != 10:
        raise ValueError(
            "PegInsertionVertical-v1 SFT uses 10-step action chunks. Expected "
            "rollout.model.num_action_chunks=10 and "
            f"rollout.model.openpi.action_horizon=10, got {num_action_chunks} "
            f"and {action_horizon}."
        )
    policy_setup = str(model_cfg.get("policy_setup", ""))
    if policy_setup not in ("panda-ee-target-dpose", "panda-ee-dpose"):
        raise ValueError(
            "PegInsertionVertical-v1 evaluation must use "
            "rollout.model.policy_setup=panda-ee-target-dpose (use_target=True, "
            "target-delta labels) or panda-ee-dpose (use_target=False, "
            "actual-EE-delta labels) so physical TCP actions are converted for "
            "the matching ManiSkill pd_ee_(target_)delta_pose controller. "
            f"Got {policy_setup!r}."
        )
    control_mode = str(cfg.env.eval.init_params.get("control_mode", ""))
    if control_mode not in ("pd_ee_target_delta_pose", "pd_ee_delta_pose"):
        raise ValueError(
            "PegInsertionVertical-v1 evaluation must use "
            "env.eval.init_params.control_mode=pd_ee_target_delta_pose or "
            "pd_ee_delta_pose. "
            f"Got {control_mode!r}."
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

    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
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

        cfg.cluster.component_placement = OmegaConf.create(
            {"env,rollout": _parse_gpu_placement(args.gpu_ids)}
        )

        cfg.env.eval.rollout_epoch = args.num_eval_episodes // args.num_envs
        cfg.env.eval.total_num_envs = args.num_envs
        cfg.env.eval.auto_reset = True
        cfg.env.eval.ignore_terminations = args.ignore_terminations
        cfg.env.eval.use_fixed_reset_state_ids = args.fixed_reset_state_ids
        cfg.env.eval.is_eval = True
        cfg.env.eval.seed = args.seed
        cfg.env.eval.max_episode_steps = args.max_episode_steps
        cfg.env.eval.max_steps_per_rollout_epoch = args.max_episode_steps
        cfg.env.eval.action_scale = args.action_scale
        cfg.env.eval.video_cfg.save_video = args.save_video
        cfg.env.eval.video_cfg.video_base_dir = str(
            Path(args.log_dir).expanduser().resolve() / "video" / "eval"
        )
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
    return cfg


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


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
    episode_metrics: dict[str, Any] = {}
    for key in ("success_once", "return", "reward", "max_reward", "episode_len"):
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

    metrics, eval_metrics_list = _run_eval(cfg)
    if args.save_episode_metrics:
        episode_metrics_path = save_episode_metrics(
            eval_metrics_list, Path(cfg.runner.logger.log_path)
        )
        print(f"Episode metrics: {episode_metrics_path}")
    summary = {
        "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()),
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
