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
from pathlib import Path
from typing import Any

import torch.multiprocessing as mp
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
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
    return cfg


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def evaluate(cfg: DictConfig) -> dict[str, Any]:
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
    metrics = {key: _json_value(value) for key, value in runner.evaluate().items()}
    prefixed_metrics = {f"eval/{key}": value for key, value in metrics.items()}
    runner.logger.info(prefixed_metrics)
    runner.metric_logger.log(step=0, data=prefixed_metrics)
    runner.metric_logger.finish()
    return metrics


def main() -> None:
    args, hydra_overrides = parse_args()
    cfg = build_config(args, hydra_overrides)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(resolved_cfg, indent=2))

    metrics = evaluate(cfg)
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
