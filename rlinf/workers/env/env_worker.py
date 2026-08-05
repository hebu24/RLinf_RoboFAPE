# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import gc
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
    Trajectory,
    convert_trajectories_to_batch,
)
from rlinf.envs import get_env_cls
from rlinf.envs.action_utils import TemporalEnsembleBuffer, prepare_actions
from rlinf.envs.utils import get_env_attr
from rlinf.envs.wrappers import RecordVideo

# Pure fns (no server) for down-sample index + progress->chunk mapping (must stay
# in sync with RobometerHistoryRewardModel.compute_reward, which uses the same).
from rlinf.models.embodiment.reward.robometer_reward_model import (
    RobometerEpisodeReward,
    _robometer_boundary_frame_indices,
    _robometer_downsample_indices,
    reconstruct_robometer_delta_reward,
    reconstruct_robometer_episode_reward,
    robometer_assignment_metric_values,
)
from rlinf.scheduler import Channel, Cluster, CommMapper, Worker
from rlinf.utils.distributed import masked_stats, normalize_from_stats
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import (
    clone_nested_to_cpu,
    copy_dict_tensor,
    split_dict_to_chunk,
    update_nested_cfg,
)
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import (
    flatten_embodied_batch,
    pack_batch,
    preprocess_embodied_batch,
)
from rlinf.workers.env.completed_episode_buffer import (
    CompletedEpisodeBuffer,
    split_trajectory_by_sizes,
)
from rlinf.workers.env.history_manager import (
    HistoryManager,
    history_obs_from_step,
    history_success_from_step,
)


@dataclass
class WindowChunkRef:
    episode_id: int
    chunk_index: int


def _rdebug_log_path() -> str:
    """Resolve the Robometer reward debug log path per-run.

    Two concurrent RL runs share the codebase; without per-run paths their debug
    writes interleave in one file. Set ``RLINF_REWARD_DEBUG_LOG`` to a distinct
    path per run (e.g. ``/tmp/robometer_rdebug_absolute.log`` vs ``_delta.log``).
    Defaults to the legacy shared path so single-run behavior is unchanged.
    """
    import os

    return os.environ.get(
        "RLINF_REWARD_DEBUG_LOG", "/tmp/robometer_rdebug.log"
    )


class EnvWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.train_video_cnt = 0
        self.eval_video_cnt = 0
        self.should_stop = False

        self.env_list = []
        self.eval_env_list = []

        self.last_obs_list = []
        self.last_intervened_info_list = []
        self._prefetched_train_bootstrap: list[EnvOutput] | None = None
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.stage_num = self.cfg.rollout.pipeline_stage_num

        self.reward_mode = self.cfg.get("reward", {}).get("reward_mode", "per_step")
        self.history_reward_assign = self.cfg.get("reward", {}).get(
            "history_reward_assign", False
        )
        self.debug_chunk_funnel = bool(
            self.cfg.get("reward", {}).get("debug_chunk_funnel", False)
        )
        self.fail_fast_on_unexpected_chunk_filter = bool(
            self.cfg.get("reward", {}).get(
                "fail_fast_on_unexpected_chunk_filter", False
            )
        )
        self.use_reward_model = self.cfg.get("reward", {}).get(
            "use_reward_model", False
        )
        self.use_realworld_reward = self.cfg.get("reward", {}).get(
            "standalone_realworld", False
        )
        self.use_external_reward_model = (
            self.use_reward_model and not self.use_realworld_reward
        )
        self.env_infos_reward_keys = ("success", "episode", "final_info")
        self.history_train_mode = self.cfg.get("reward", {}).get(
            "history_train_mode", "rollout_window"
        )
        self.use_completed_episode_buffer = (
            self.history_train_mode == "complete_episode"
            and self.reward_mode == "history_buffer"
            and self.cfg.get("reward", {}).get("model", {}).get("model_type")
            == "robometer"
        )
        if (
            self.reward_mode == "history_buffer"
            and self.cfg.get("reward", {}).get("model", {}).get("model_type")
            == "robometer"
            and self.history_train_mode not in {"rollout_window", "complete_episode"}
        ):
            raise ValueError(
                "reward.history_train_mode must be 'rollout_window' or "
                f"'complete_episode', got {self.history_train_mode!r}."
            )
        if self.use_external_reward_model:
            self.reward_weight = self.cfg.reward.get("reward_weight", 1.0)
            self.env_reward_weight = self.cfg.reward.get("env_reward_weight", 0.0)
            # Reward shaping: absolute (progress / progress-1 per low-level step) or
            # delta (per-chunk progress_{i+1}-progress_i + success_bonus on success
            # chunks). Delta mode POSTs chunk-boundary frames instead of a uniformly
            # down-sampled video; see reconstruct_robometer_delta_reward.
            self.reward_shaping = self.cfg.reward.get("shaping", "absolute")
            self.absolute_success_terminal_bonus = float(
                self.cfg.reward.get("absolute", {}).get(
                    "success_terminal_bonus", 0.0
                )
            )
            self.delta_success_bonus = float(
                self.cfg.reward.get("delta", {}).get("success_bonus", 0.1)
            )
            self.delta_failure_terminal_penalty = float(
                self.cfg.reward.get("delta", {}).get("failure_terminal_penalty", 0.0)
            )

        # Env configurations
        self.use_training_pipeline = self.cfg.runner.get("use_training_pipeline", False)
        self.only_eval = getattr(self.cfg.runner, "only_eval", False)
        self.model_cfg = (
            self.cfg.rollout.model if self.only_eval else self.cfg.actor.model
        )
        train_env_cfg = self.cfg.env.get("train", None)
        eval_env_cfg = self.cfg.env.get("eval", None)
        self.enable_train = not self.only_eval and train_env_cfg is not None
        self.enable_eval = (
            self.cfg.runner.get("val_check_interval", -1) > 0 or self.only_eval
        )
        self.rollout_epoch = (
            train_env_cfg.rollout_epoch if train_env_cfg is not None else 1
        )
        self.eval_rollout_epoch = eval_env_cfg.rollout_epoch if self.enable_eval else 1

        # Independent rollout windows (ASYNC_INDEPENDENT_ROLLOUT_WINDOW_IMPLEMENTATION.md).
        # continuous: legacy cross-window continuation (last_obs_list resumes the
        #   previous window's sim state/history into the next bootstrap_step).
        # independent: force-close unfinished episodes at each window boundary
        #   (synthetic truncation, never a task termination), settle Robometer
        #   rewards up to the boundary, then reset all train envs + clear history
        #   so the next window starts from a fresh reset observation.
        rollout_window_mode = (
            train_env_cfg.get("rollout_window_mode", "continuous")
            if train_env_cfg is not None
            else "continuous"
        )
        self.independent_rollout_windows = self._validate_rollout_window_mode(
            rollout_window_mode,
            (
                train_env_cfg.get("auto_reset", False)
                if train_env_cfg is not None
                else False
            ),
            self.history_train_mode,
            self.rollout_epoch,
        )
        # Per-stage forced-timeout mask stashed by _finalize_independent_window_boundary
        # for reward-query settlement assertions + debug. None when no window has
        # been finalized yet.
        self._independent_window_forced_timeout_masks: list[torch.Tensor | None] = [
            None
        ] * self.stage_num
        # Per-stage flag: True iff any episode in the current window succeeded
        # (any assignment.episode_success). Reset each window in
        # _reset_window_chunk_refs. Used to mask (skip) SR==0 all-fail windows.
        self._window_any_episode_success: list[bool] = [False] * self.stage_num

        self.train_enable_offload = (
            train_env_cfg.get("enable_offload", False)
            if train_env_cfg is not None
            else False
        )
        self.eval_enable_offload = (
            eval_env_cfg.get("enable_offload", False)
            if eval_env_cfg is not None
            else False
        )
        if self.enable_train:
            self.train_num_envs_per_stage = (
                self.cfg.env.train.total_num_envs // self._world_size // self.stage_num
            )
            self.train_batch_size = self.cfg.env.train.total_num_envs // self.stage_num
        if self.enable_eval:
            self.eval_num_envs_per_stage = (
                self.cfg.env.eval.total_num_envs // self._world_size // self.stage_num
            )
            self.eval_batch_size = self.cfg.env.eval.total_num_envs // self.stage_num
        self.n_train_chunk_steps = 0
        if self.enable_train:
            self.n_train_chunk_steps = (
                self.cfg.env.train.max_steps_per_rollout_epoch
                // self.model_cfg.num_action_chunks
            )
        self.n_eval_chunk_steps = 0
        if self.enable_eval:
            # Real-time chunking: execute only the first k actions of each predicted
            # chunk before re-querying the model. k defaults to num_action_chunks
            # (current open-loop behavior). With temporal_ensemble_weight m > 0,
            # overlapping predictions are time-weighted-blended (ACT) before the
            # (linear) panda action conversion. See TemporalEnsembleBuffer.
            self.eval_execute_chunks = int(
                self.cfg.env.eval.get(
                    "execute_action_chunks", self.model_cfg.num_action_chunks
                )
            )
            self.eval_ensemble_m = float(
                self.cfg.env.eval.get("temporal_ensemble_weight", 0.0)
            )
            assert 1 <= self.eval_execute_chunks <= self.model_cfg.num_action_chunks, (
                f"env.eval.execute_action_chunks must be in "
                f"[1, {self.model_cfg.num_action_chunks}], got {self.eval_execute_chunks}"
            )
            assert self.eval_ensemble_m >= 0.0, (
                f"env.eval.temporal_ensemble_weight must be >= 0, "
                f"got {self.eval_ensemble_m}"
            )
            self.n_eval_chunk_steps = (
                self.cfg.env.eval.max_steps_per_rollout_epoch
                // self.eval_execute_chunks
            )
        self.actor_split_num = (
            1 if not self.enable_train else self.get_actor_split_num()
        )
        if self.use_training_pipeline and self.enable_train:
            self._init_pipeline_params()
        if (
            not self.use_training_pipeline
            and self.enable_train
            and self.cfg.algorithm.get("actor_channel_keyed_routing", False)
        ):
            self._init_async_actor_params()

        if self.enable_train:
            self.train_prev_done: list[torch.Tensor] = [
                torch.zeros(self.train_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
        if self.enable_eval:
            self.eval_prev_done: list[torch.Tensor] = [
                torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
                for _ in range(self.stage_num)
            ]
            # Per-stage temporal-ensemble state for eval real-time chunking (m > 0).
            # Created lazily in env_evaluate_step once the env device is known, because
            # eval envs are constructed per stage after __init__.
            self.eval_ensemble_buffers: list = [None] * self.stage_num
            self.eval_ensemble_step: list[int] = [0] * self.stage_num
        self.env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)

        if self.env_decoupled_mode:
            # Init the batch_router for env decoupled mode
            # The batch_router is a dictionary that maps the tag to the list of batch_index.
            self.batch_router = {}
            assert self._component_placement.get_world_size(
                "env"
            ) >= self._component_placement.get_world_size("rollout"), (
                "the world size of env must be greater than the world size of rollout in env_decoupled_mode"
            )

    def init_worker(self):
        # This is a barrier to ensure all envs' initial setup upon import is done
        # Essential for RealWorld env to ensure initial ROS node setup is done
        self.broadcast(
            True,
            groups=[(self._group_name, list(range(self._world_size)))],
        )

        self.update_env_cfg()

        if self.enable_train:
            train_env_cls = get_env_cls(self.cfg.env.train.env_type, self.cfg.env.train)
            self.env_list = self._setup_env_and_wrappers(
                env_cls=train_env_cls,
                env_cfg=self.cfg.env.train,
                num_envs_per_stage=self.train_num_envs_per_stage,
            )
            if self.train_enable_offload:
                assert all(hasattr(env, "offload") for env in self.env_list), (
                    "train envs must have an offload method to enable offload!"
                )

        if self.enable_eval:
            eval_env_cls = get_env_cls(self.cfg.env.eval.env_type, self.cfg.env.eval)
            self.eval_env_list = self._setup_env_and_wrappers(
                env_cls=eval_env_cls,
                env_cfg=self.cfg.env.eval,
                num_envs_per_stage=self.eval_num_envs_per_stage,
            )
            if self.eval_enable_offload:
                assert all(hasattr(env, "offload") for env in self.eval_env_list), (
                    "eval envs must have an offload method to enable offload!"
                )

        if self.enable_train:
            if self.reward_mode == "history_buffer":
                self.train_history_managers = [
                    HistoryManager(self.cfg.reward, self.train_num_envs_per_stage)
                    for _ in range(self.stage_num)
                ]
                self.history_lengths = [{} for _ in range(self.stage_num)]
                self._episode_chunk_ids = [
                    torch.zeros(self.train_num_envs_per_stage, dtype=torch.long)
                    for _ in range(self.stage_num)
                ]
                self._episode_chunk_counts = [
                    torch.zeros(self.train_num_envs_per_stage, dtype=torch.long)
                    for _ in range(self.stage_num)
                ]
                self._window_chunk_refs = [[] for _ in range(self.stage_num)]
                self._window_chunk_funnel = [
                    self._new_chunk_funnel_state() for _ in range(self.stage_num)
                ]
                if self.use_completed_episode_buffer:
                    max_chunks = math.ceil(
                        self.cfg.env.train.max_episode_steps
                        / self.cfg.env.train.execute_action_chunks
                    )
                    self.completed_episode_buffers = [
                        CompletedEpisodeBuffer(
                            self.train_num_envs_per_stage,
                            max_chunks=max_chunks,
                            max_episode_steps=self.cfg.env.train.max_episode_steps,
                        )
                        for _ in range(self.stage_num)
                    ]

        self._init_env()

    def _reset_window_chunk_refs(self) -> None:
        if self.reward_mode != "history_buffer" or not self.enable_train:
            return
        self._window_chunk_refs = [[] for _ in range(self.stage_num)]
        self._window_chunk_funnel = [
            self._new_chunk_funnel_state() for _ in range(self.stage_num)
        ]
        # Reset the per-window success flag (set True by assign_history_reward
        # when any episode succeeds; used to skip SR==0 all-fail windows).
        self._window_any_episode_success = [False] * self.stage_num

    @staticmethod
    def _new_chunk_funnel_state() -> dict[str, Any]:
        return {
            "raw_window_chunks": 0,
            "query_units": 0,
            "queried_chunks": 0,
            "assigned_chunks": 0,
            "masked_in_chunks": 0,
            "query_units_by_env_episode": set(),
            "queried_chunk_refs": set(),
            "assigned_chunk_refs": set(),
            "drop_reasons": defaultdict(int),
        }

    def _chunk_debug_active(self) -> bool:
        return bool(getattr(self, "debug_chunk_funnel", False))

    def _chunk_fail_fast_active(self) -> bool:
        return bool(getattr(self, "fail_fast_on_unexpected_chunk_filter", False))

    def _log_chunk_debug(self, prefix: str, **fields: Any) -> None:
        parts = [f"{key}={fields[key]}" for key in sorted(fields)]
        msg = f"{prefix} {' '.join(parts)}".strip()
        self.log_info(msg)
        try:
            log_path = Path(_rdebug_log_path())
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    def _record_chunk_drop(
        self, stage_id: int, reason: str, count: int = 1, **ctx: Any
    ) -> None:
        state = self._window_chunk_funnel[stage_id]
        state["drop_reasons"][reason] += int(count)
        if self._chunk_debug_active():
            self._log_chunk_debug(
                "CHUNK_DROP_REASON",
                count=int(count),
                reason=reason,
                stage_id=stage_id,
                **ctx,
            )

    def _fail_unexpected_chunk(self, reason: str, **ctx: Any) -> None:
        if self._chunk_debug_active() or self._chunk_fail_fast_active():
            self._log_chunk_debug("CHUNK_FAIL_FAST", reason=reason, **ctx)
        raise RuntimeError(f"{reason}: {ctx}")

    def _ensure_expected_chunk(
        self,
        *,
        allowed: bool,
        stage_id: int,
        reason: str,
        count: int = 1,
        **ctx: Any,
    ) -> None:
        if allowed:
            return
        self._record_chunk_drop(stage_id, reason, count=count, **ctx)
        if self._chunk_fail_fast_active():
            self._fail_unexpected_chunk(reason, **ctx)

    def _window_chunk_count_for_episode(
        self, stage_id: int, env_id: int, episode_id: int
    ) -> int:
        return len(self._window_chunk_slices_for_episode(stage_id, env_id, episode_id))

    def _emit_chunk_funnel_metrics(
        self, stage_id: int, env_metrics: dict[str, list]
    ) -> None:
        state = self._window_chunk_funnel[stage_id]
        drop_reasons = dict(state["drop_reasons"])
        masked_in_chunks = int(state["masked_in_chunks"])
        filtered = int(
            state["raw_window_chunks"]
            - masked_in_chunks
            - drop_reasons.get("pickup_prefix_excluded", 0)
            - drop_reasons.get("outside_current_window", 0)
        )
        summary = {
            "raw_window_chunks": int(state["raw_window_chunks"]),
            "query_units": int(state["query_units"]),
            "queried_chunks": int(state["queried_chunks"]),
            "assigned_chunks": int(state["assigned_chunks"]),
            "masked_in_chunks": masked_in_chunks,
            "filtered_chunks": max(filtered, 0),
            "drop_reasons": drop_reasons,
        }
        if self._chunk_debug_active():
            self._log_chunk_debug("CHUNK_FUNNEL", stage_id=stage_id, **summary)
        metric_values = {
            "chunk_funnel/raw_window_chunks": float(summary["raw_window_chunks"]),
            "chunk_funnel/query_units": float(summary["query_units"]),
            "chunk_funnel/queried_chunks": float(summary["queried_chunks"]),
            "chunk_funnel/assigned_chunks": float(summary["assigned_chunks"]),
            "chunk_funnel/masked_in_chunks": float(summary["masked_in_chunks"]),
        }
        for reason, value in sorted(drop_reasons.items()):
            metric_values[f"chunk_funnel/masked_out_chunks_reason_{reason}"] = float(
                value
            )
        for key, value in metric_values.items():
            env_metrics[key].append(torch.tensor([value], dtype=torch.float32))

    @staticmethod
    def _count_masked_in_chunks(loss_mask: Any) -> int:
        if loss_mask is None:
            return 0
        if isinstance(loss_mask, list):
            if not loss_mask:
                return 0
            loss_mask = torch.stack(loss_mask, dim=0)
        if not isinstance(loss_mask, torch.Tensor):
            return 0
        return int(loss_mask.to(dtype=torch.bool).all(dim=-1).sum())

    def _sync_window_chunk_refs(self, stage_id: int) -> None:
        refs = self._window_chunk_refs[stage_id]
        reward_steps = len(self.rollout_results[stage_id].rewards)
        if len(refs) > reward_steps:
            del refs[reward_steps:]

    def _record_window_chunk_ref(
        self, stage_id: int, env_output: EnvOutput, has_rewards: bool
    ) -> None:
        if self.reward_mode != "history_buffer":
            return

        if has_rewards:
            episode_ids = self._episode_chunk_ids[stage_id].clone()
            chunk_indices = self._episode_chunk_counts[stage_id].clone()
            self._window_chunk_funnel[stage_id]["raw_window_chunks"] += int(
                self.train_num_envs_per_stage
            )
            self._window_chunk_refs[stage_id].append(
                [
                    WindowChunkRef(
                        episode_id=int(episode_ids[env_id]),
                        chunk_index=int(chunk_indices[env_id]),
                    )
                    for env_id in range(self.train_num_envs_per_stage)
                ]
            )
            self._episode_chunk_counts[stage_id] = chunk_indices + 1
            self._sync_window_chunk_refs(stage_id)

        if env_output.dones is None:
            return
        done_envs = env_output.dones.any(dim=1).to(dtype=torch.bool)
        if not bool(done_envs.any()):
            return
        self._episode_chunk_ids[stage_id][done_envs] += 1
        self._episode_chunk_counts[stage_id][done_envs] = 0

    def _window_chunk_slices_for_episode(
        self, stage_id: int, env_id: int, episode_id: int
    ) -> list[tuple[int, int]]:
        self._sync_window_chunk_refs(stage_id)
        refs = self._window_chunk_refs[stage_id]
        reward_steps = len(self.rollout_results[stage_id].rewards)
        slices: list[tuple[int, int]] = []
        for local_chunk_idx, per_env_refs in enumerate(refs):
            if local_chunk_idx >= reward_steps:
                break
            ref = per_env_refs[env_id]
            if ref.episode_id != episode_id:
                continue
            slices.append((local_chunk_idx, ref.chunk_index))
        return slices

    @staticmethod
    def _history_input_emitted_env_ids(history_input: dict[str, Any]) -> set[int]:
        """Return env ids that emitted at least one history item this query."""
        emitted_env_ids: set[int] = set()
        for buffer_data in (history_input or {}).values():
            if not isinstance(buffer_data, dict):
                continue
            for per_env_values in buffer_data.values():
                if not isinstance(per_env_values, list):
                    continue
                for env_id, values in enumerate(per_env_values):
                    if values:
                        emitted_env_ids.add(env_id)
        return emitted_env_ids

    def update_env_cfg(self):
        if self.enable_train:
            # train env
            train_override_cfgs = self.cfg.env.train.get("override_cfgs", None)
            if train_override_cfgs is not None:
                assert len(train_override_cfgs) > self._rank, (
                    f"{len(train_override_cfgs)=} > {self._rank=}"
                )

                general_train_override_cfg = OmegaConf.to_container(
                    self.cfg.env.train.get("override_cfg", {}), resolve=True
                )
                override_cfg = OmegaConf.to_container(
                    train_override_cfgs[self._rank], resolve=True
                ).copy()

                base_cfg = {}
                base_cfg = update_nested_cfg(base_cfg, general_train_override_cfg)
                base_cfg = update_nested_cfg(base_cfg, override_cfg)
                setattr(self.cfg.env.train, "override_cfg", OmegaConf.create(base_cfg))
            self._inject_realworld_reward_cfg(self.cfg.env.train)
        if self.enable_eval:
            eval_override_cfgs = self.cfg.env.eval.get("override_cfgs", None)
            if eval_override_cfgs is not None:
                assert len(eval_override_cfgs) > self._rank, (
                    f"{len(eval_override_cfgs)=} > {self._rank=}"
                )

                general_eval_override_cfg = OmegaConf.to_container(
                    self.cfg.env.eval.get("override_cfg", {}), resolve=True
                )
                eval_override_cfg = OmegaConf.to_container(
                    eval_override_cfgs[self._rank], resolve=True
                ).copy()
                base_eval_cfg = {}
                base_eval_cfg = update_nested_cfg(
                    base_eval_cfg, general_eval_override_cfg
                )
                base_eval_cfg = update_nested_cfg(base_eval_cfg, eval_override_cfg)
                setattr(
                    self.cfg.env.eval, "override_cfg", OmegaConf.create(base_eval_cfg)
                )
            self._inject_realworld_reward_cfg(self.cfg.env.eval)

    def _init_pipeline_params(self):
        actor_ws = self._component_placement.get_world_size("actor")
        logical_env_ws = self._world_size * self.stage_num
        self.shuffle_rollout = self.cfg.algorithm.get("shuffle_rollout", True)
        self.pipeline_stage_actor_splits = [
            CommMapper.get_dst_ranks(
                batch_size=self.cfg.env.train.total_num_envs,
                src_world_size=logical_env_ws,
                dst_world_size=actor_ws,
                src_rank=self._rank * self.stage_num + stage_id,
            )
            for stage_id in range(self.stage_num)
        ]
        local_actor_ranks = {
            actor_rank
            for actor_splits in self.pipeline_stage_actor_splits
            for actor_rank, _ in actor_splits
        }
        self.pipeline_actor_env_ranks = {
            actor_rank: sorted(
                {
                    logical_src_rank // self.stage_num
                    for logical_src_rank, _ in CommMapper.get_src_ranks(
                        batch_size=self.cfg.env.train.total_num_envs,
                        src_world_size=logical_env_ws,
                        dst_world_size=actor_ws,
                        dst_rank=actor_rank,
                    )
                }
            )
            for actor_rank in range(actor_ws)
        }
        self.pipeline_actor_keys = {
            actor_rank: CommMapper.build_channel_key(
                actor_rank, actor_rank, "pipeline_actor"
            )
            for actor_rank in local_actor_ranks
        }
        if self.shuffle_rollout:
            self.shuffle_generators = {
                actor_rank: torch.Generator().manual_seed(
                    self.cfg.actor.seed + actor_rank + self._rank * actor_ws
                )
                for actor_rank in local_actor_ranks
            }

    def _init_async_actor_params(self):
        """Build per-actor-rank channel keys for deterministic non-pipeline routing.

        Mirrors ``_init_pipeline_params`` but uses the ``"async_actor"`` tag so each
        env worker writes each shard to a dedicated actor-rank queue (via
        ``CommMapper.get_dst_ranks``) instead of the shared DEFAULT queue. Eliminates
        the multi-rank DEFAULT-queue competition that starves actor ranks.
        """
        actor_ws = self._component_placement.get_world_size("actor")
        logical_env_ws = self._world_size * self.stage_num
        self.async_stage_actor_splits = [
            CommMapper.get_dst_ranks(
                batch_size=self.cfg.env.train.total_num_envs,
                src_world_size=logical_env_ws,
                dst_world_size=actor_ws,
                src_rank=self._rank * self.stage_num + stage_id,
            )
            for stage_id in range(self.stage_num)
        ]
        local_actor_ranks = {
            actor_rank
            for actor_splits in self.async_stage_actor_splits
            for actor_rank, _ in actor_splits
        }
        self.async_actor_keys = {
            actor_rank: CommMapper.build_channel_key(
                actor_rank, actor_rank, "async_actor"
            )
            for actor_rank in local_actor_ranks
        }

    def _inject_realworld_reward_cfg(self, env_cfg: DictConfig):
        if not (self.use_reward_model and self.use_realworld_reward):
            return
        if env_cfg.env_type != "realworld":
            return

        reward_placements = self._component_placement.get_strategy(
            "reward"
        ).get_placement(Cluster())
        assert len(reward_placements) > 0, (
            "Reward placement must contain at least one worker."
        )
        reward_placement = reward_placements[0]
        reward_hardware_ranks = self._component_placement.get_hardware_ranks("reward")
        assert len(reward_hardware_ranks) > 0, (
            "Reward placement must contain at least one hardware rank."
        )

        override_cfg = OmegaConf.to_container(
            env_cfg.get("override_cfg", {}), resolve=True
        )
        override_cfg["use_reward_model"] = True
        override_cfg["reward_worker_cfg"] = OmegaConf.to_container(
            self.cfg.reward, resolve=True
        )
        override_cfg["reward_worker_hardware_rank"] = reward_hardware_ranks[0]
        override_cfg["reward_worker_node_rank"] = reward_placement.cluster_node_rank
        override_cfg["reward_worker_node_group"] = reward_placement.node_group_label
        override_cfg["reward_image_key"] = env_cfg.main_image_key
        setattr(env_cfg, "override_cfg", OmegaConf.create(override_cfg))

    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage: int):
        env_list = []

        for stage_id in range(self.stage_num):
            env = env_cls(
                cfg=env_cfg,
                num_envs=num_envs_per_stage,
                seed_offset=self._rank * self.stage_num + stage_id,
                total_num_processes=self._world_size * self.stage_num,
                worker_info=self.worker_info,
            )
            if env_cfg.get("reset_options", {}) and env_cfg.reset_options.get(
                "pre_grasped", False
            ):
                from rlinf.envs.wrappers import PreGraspedInitWrapper

                env = PreGraspedInitWrapper(
                    env,
                    seed=(
                        int(getattr(env_cfg, "seed", 0))
                        if bool(getattr(env_cfg, "shared_reset_seed", False))
                        else int(getattr(env_cfg, "seed", 0)) + self._rank
                    ),
                    shared_reset_seed=bool(
                        getattr(env_cfg, "shared_reset_seed", False)
                    ),
                )
            if env_cfg.video_cfg.save_video:
                env = RecordVideo(env, env_cfg.video_cfg)
            if env_cfg.get("data_collection", None) and getattr(
                env_cfg.data_collection, "enabled", False
            ):
                from rlinf.envs.wrappers import CollectEpisode

                env = CollectEpisode(
                    env,
                    save_dir=env_cfg.data_collection.save_dir,
                    rank=self._rank,
                    num_envs=num_envs_per_stage,
                    export_format=getattr(
                        env_cfg.data_collection, "export_format", "pickle"
                    ),
                    robot_type=getattr(env_cfg.data_collection, "robot_type", "panda"),
                    fps=getattr(env_cfg.data_collection, "fps", 10),
                    only_success=getattr(
                        env_cfg.data_collection, "only_success", False
                    ),
                    finalize_interval=getattr(
                        env_cfg.data_collection, "finalize_interval", 100
                    ),
                )
            env_list.append(env)
        return env_list

    def _init_env(self):
        for i in range(self.stage_num):
            if self.enable_train:
                if self.cfg.env.train.auto_reset:
                    extracted_obs, _ = self.env_list[i].reset()
                    self.last_obs_list.append(extracted_obs)
                    self.last_intervened_info_list.append((None, None))
                    # Initial reset stashed pick-up frames (pre-grasped). Prepend
                    # them to this stage's history buffer so the first episode's
                    # robometer video starts at the pick-up. History is empty
                    # here, so prepend is unconditionally safe.
                    if self.reward_mode == "history_buffer":
                        _consume = get_env_attr(
                            self.env_list[i], "consume_pickup_frames"
                        )
                        _consume = (
                            _consume() if callable(_consume) else (_consume or {})
                        )
                        _hm = self.train_history_managers[i]
                        for _e, _frs in (_consume or {}).items():
                            if _frs:
                                _hm.prepend_history_entries(int(_e), _frs)
                if self.train_enable_offload and self.cfg.env.train.get(
                    "enable_init_offload", True
                ):
                    self.env_list[i].offload()
            if self.enable_eval:
                if self.eval_enable_offload:
                    self.eval_env_list[i].offload()

    @Worker.timer("env_interact_step")
    def env_interact_step(
        self, chunk_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to interact with the environment.
        """
        chunk_actions = prepare_actions(
            raw_chunk_actions=chunk_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=self.model_cfg.num_action_chunks,
            action_dim=self.model_cfg.action_dim,
            action_scale=self.cfg.env.train.get("action_scale", 1.0),
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.env_list[stage_id].chunk_step(chunk_actions)
        )
        if self.reward_mode == "history_buffer":
            self._append_chunk_history(stage_id, obs_list, infos_list)
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )
        if not self.cfg.env.train.auto_reset:
            if self.cfg.env.train.ignore_terminations:
                if chunk_truncations[:, -1].any():
                    assert chunk_truncations[:, -1].all()
                    if "episode" in infos:
                        for key in infos["episode"]:
                            env_info[key] = infos["episode"][key].cpu()
            else:
                if "episode" in infos:
                    for key in infos["episode"]:
                        env_info[key] = infos["episode"][key].cpu()
        elif chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        intervene_actions = (
            infos["intervene_action"] if "intervene_action" in infos else None
        )
        intervene_flags = infos["intervene_flag"] if "intervene_flag" in infos else None
        if self.cfg.env.train.auto_reset and chunk_dones.any():
            if "intervene_action" in infos["final_info"]:
                intervene_actions = infos["final_info"]["intervene_action"]
                intervene_flags = infos["final_info"]["intervene_flag"]

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
            rewards=chunk_rewards,
            env_infos=infos if isinstance(infos, dict) else None,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )
        return env_output, env_info

    def _history_obs_from_step(
        self, obs: dict[str, Any] | None, infos: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        return history_obs_from_step(obs, infos)

    def _append_chunk_history(
        self,
        stage_id: int,
        obs_list: list[dict[str, Any]],
        infos_list: list[dict[str, Any]],
    ) -> None:
        history_manager = self.train_history_managers[stage_id]
        step_history = [
            history_obs_from_step(obs, infos)
            for obs, infos in zip(obs_list, infos_list, strict=True)
        ]
        step_success = [
            history_success_from_step(infos, history_manager.num_envs)
            for infos in infos_list
        ]
        history_manager.append_history_sequence(step_history, success_list=step_success)

    def _history_success_from_step(self, infos: dict[str, Any] | None) -> list[bool]:
        return history_success_from_step(infos, self.train_num_envs_per_stage)

    def env_evaluate_step(
        self, raw_actions: torch.Tensor, stage_id: int
    ) -> tuple[EnvOutput, dict[str, Any]]:
        """
        This function is used to evaluate the environment.

        Real-time chunking: when env.eval.execute_action_chunks (k) <
        num_action_chunks, only the first k actions of the predicted chunk are
        executed before the model is re-queried. When
        env.eval.temporal_ensemble_weight (m) > 0, the k executed actions are
        time-weighted blends of all overlapping buffered predictions (ACT-style
        temporal ensembling), closing the within-chunk open loop and smoothing
        chunk-boundary discontinuity. Defaults (k = num_action_chunks, m = 0)
        reproduce the original open-loop behavior.
        """
        k = self.eval_execute_chunks
        m = self.eval_ensemble_m
        env = self.eval_env_list[stage_id]

        if m > 0:
            # Temporal ensembling: blend overlapping predictions across the k
            # executed steps. Buffer holds raw (post-Unnormalize) model actions.
            if self.eval_ensemble_buffers[stage_id] is None:
                num_envs = env.num_envs
                device = env.device
                self.eval_ensemble_buffers[stage_id] = TemporalEnsembleBuffer(
                    horizon=self.model_cfg.num_action_chunks,
                    m=m,
                    num_envs=num_envs,
                    device=device,
                )
            buffer = self.eval_ensemble_buffers[stage_id]
            base_step = self.eval_ensemble_step[stage_id]
            buffer.push(raw_actions, predict_step=base_step)
            # Build the k blended raw actions [B, k, D] by aging the buffer.
            blended = []
            for s in range(k):
                blended.append(buffer.current_action(base_step + s))
            blended_chunk = torch.stack(blended, dim=1)  # [B, k, D]
            exec_raw = blended_chunk
        else:
            # Pure slicing: execute the first k raw actions directly.
            exec_raw = raw_actions[:, :k, :]

        chunk_actions = prepare_actions(
            raw_chunk_actions=exec_raw,
            env_type=self.cfg.env.eval.env_type,
            model_type=self.model_cfg.model_type,
            num_action_chunks=k,
            action_dim=self.model_cfg.action_dim,
            action_scale=self.cfg.env.eval.get("action_scale", 1.0),
            policy=self.model_cfg.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )
        env_info = {}

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = env.chunk_step(
            chunk_actions
        )
        if isinstance(obs_list, (list, tuple)):
            extracted_obs = obs_list[-1] if obs_list else None
        if isinstance(infos_list, (list, tuple)):
            infos = infos_list[-1] if infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        final_obs = (
            self._build_chunk_final_obs(obs_list, infos_list)
            if self.use_external_reward_model
            else (
                infos["final_observation"]
                if isinstance(infos, dict) and "final_observation" in infos
                else None
            )
        )

        current_dones = chunk_dones[:, -1]  # [num_envs] bool
        if self.cfg.env.eval.auto_reset:
            newly_done = current_dones
        else:
            prev = self.eval_prev_done[stage_id].to(current_dones.device)
            newly_done = current_dones & ~prev
            self.eval_prev_done[stage_id] = prev | current_dones

        if m > 0:
            # Advance the global step; drop stale predictions for reset envs so
            # the next chunk's blend does not mix in pre-reset predictions.
            self.eval_ensemble_step[stage_id] += k
            if newly_done.any():
                buffer.mark_reset(newly_done, step=self.eval_ensemble_step[stage_id])

        if newly_done.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info[key] = final_info["episode"][key][newly_done].cpu()
            elif "episode" in infos:
                for key in infos["episode"]:
                    env_info[key] = infos["episode"][key][newly_done].cpu()

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=final_obs,
        )
        return env_output, env_info

    def _build_chunk_final_obs(self, obs_list, infos_list):
        """Build per-env terminal observations for a whole chunk.

        Matches the old wrapper semantics:
        - default to the last rollout observation for each env
        - if an env terminated earlier in the chunk, replace that env's observation
          with the true `final_observation` captured at that substep
        """
        if not isinstance(obs_list, (list, tuple)) or len(obs_list) == 0:
            return None

        last_obs = obs_list[-1]
        if not isinstance(last_obs, dict):
            return None

        merged_final_obs = copy_dict_tensor(last_obs)

        if not isinstance(infos_list, (list, tuple)):
            return merged_final_obs

        for step_infos in infos_list:
            if not isinstance(step_infos, dict):
                continue
            if (
                "final_observation" not in step_infos
                or "_final_observation" not in step_infos
            ):
                continue

            final_obs = step_infos["final_observation"]
            reset_mask = step_infos["_final_observation"]
            if final_obs is None or reset_mask is None:
                continue
            reset_mask = (
                reset_mask.detach().cpu().numpy()
                if isinstance(reset_mask, torch.Tensor)
                else np.asarray(reset_mask)
            )
            done_mask = (
                reset_mask.any(axis=-1)
                if reset_mask.ndim > 1
                else reset_mask.astype(bool)
            )
            if not done_mask.any():
                continue

            for key, value in merged_final_obs.items():
                if key not in final_obs:
                    continue

                final_value = final_obs[key]
                if isinstance(value, torch.Tensor) and isinstance(
                    final_value, torch.Tensor
                ):
                    dst_mask = torch.as_tensor(done_mask, device=value.device)
                    src_mask = dst_mask.to(device=final_value.device)
                    merged_final_obs[key][dst_mask] = final_value[src_mask]
                elif isinstance(value, np.ndarray) and isinstance(
                    final_value, np.ndarray
                ):
                    merged_final_obs[key][done_mask] = final_value[done_mask]

        return merged_final_obs

    @staticmethod
    def _infer_rollout_batch_size(data: Any) -> int:
        """Infer batch dim for routed shards; supports RolloutResult and plain tensor payloads.

        When the channel carries a non-``RolloutResult`` shard (e.g. reward tensor or eval
        actions) into a rollout recv, avoid assuming dataclass fields and delegate or use
        the leading dimension of dense arrays.
        """

        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            return int(data.shape[0])
        if isinstance(data, RolloutResult):
            for field_name in (
                "actions",
                "prev_logprobs",
                "prev_values",
                "bootstrap_values",
                "versions",
            ):
                value = getattr(data, field_name, None)
                if isinstance(value, torch.Tensor):
                    return int(value.shape[0])
            forward_inputs = getattr(data, "forward_inputs", None)
            if forward_inputs:
                first_tensor = next(iter(forward_inputs.values()))
                if isinstance(first_tensor, torch.Tensor):
                    return int(first_tensor.shape[0])
            raise ValueError("Cannot infer batch size from rollout result.")
        from rlinf.scheduler import infer_batch_size

        return infer_batch_size(data)

    @Worker.timer("compute_bootstrap_rewards")
    def compute_bootstrap_rewards(
        self,
        env_output: EnvOutput,
        bootstrap_values: torch.Tensor | None,
        reward_model_output: torch.Tensor | None,
    ) -> torch.Tensor | None:
        rewards = env_output.rewards
        if rewards is None:
            return None

        if reward_model_output is not None:
            rm = reward_model_output.to(rewards.dtype)
            # Per-step / terminal reward models return a scalar-per-env [B] (or
            # [B, num_action_chunks]) tensor -> additive blend with the env
            # reward. A per-chunk history reward (e.g. robometer progress)
            # returns [B, T_history] with T_history != num_action_chunks; that
            # cannot be blended into the per-sub-step env rewards and is instead
            # scattered per-chunk by assign_history_reward -- skip the blend
            # here (env reward is zeroed via env_reward_weight=0, so this is a
            # no-op that avoids a shape broadcast error and double-counting).
            if rm.dim() <= 1 or rm.shape == rewards.shape:
                rewards = self.env_reward_weight * rewards + self.reward_weight * rm

        adjusted_rewards = rewards.clone()
        if (
            bootstrap_values is None
            or not self.cfg.env.train.auto_reset
            or env_output.dones is None
        ):
            return adjusted_rewards

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "standard":
            last_step_truncations = env_output.truncations[:, -1]
        else:
            last_step_truncations = env_output.dones[:, -1]

        if not last_step_truncations.any():
            return adjusted_rewards

        final_values = torch.zeros_like(adjusted_rewards[:, -1], dtype=torch.float32)
        final_values[last_step_truncations] = (
            bootstrap_values[last_step_truncations].reshape(-1).to(torch.float32)
        )
        adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values
        return adjusted_rewards

    def _history_reward_placeholder(
        self,
        rewards: torch.Tensor | None,
        rollout_result: RolloutResult,
    ) -> torch.Tensor | None:
        if rewards is not None:
            return rewards
        if self.reward_mode != "history_buffer" or not getattr(
            self, "history_reward_assign", False
        ):
            return None

        action = None
        if rollout_result.forward_inputs:
            action = rollout_result.forward_inputs.get("action")
        if action is None:
            action = rollout_result.actions
        if action is None:
            return None

        batch_size = int(action.shape[0])
        chunk_size = int(self.model_cfg.num_action_chunks)
        return torch.zeros((batch_size, chunk_size), dtype=torch.float32)

    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            for i in range(self.stage_num):
                if self.cfg.env.train.video_cfg.save_video:
                    flush_video = get_env_attr(self.env_list[i], "flush_video")
                    if callable(flush_video):
                        flush_video()
                self.env_list[i].update_reset_state_ids()
        elif mode == "eval":
            for i in range(self.stage_num):
                if self.cfg.env.eval.video_cfg.save_video:
                    flush_video = get_env_attr(self.eval_env_list[i], "flush_video")
                    if callable(flush_video):
                        flush_video()
                if not self.cfg.env.eval.auto_reset:
                    self.eval_env_list[i].update_reset_state_ids()

    @Worker.timer("get_reward_model_output")
    def get_reward_model_output(
        self,
        env_output: EnvOutput,
        send_channel: Channel,
        recv_channel: Channel,
        stage_id: int | None = None,
        last_run: bool = False,
    ):
        if __import__("os").environ.get("RLINF_REWARD_DEBUG"):
            try:
                with open(_rdebug_log_path(), "a") as _f:
                    _f.write(
                        f"EW get_reward_model_output ENTER rank={getattr(self, '_rank', -1)} reward_mode={self.reward_mode}\n"
                    )
            except Exception:
                pass
        if self.reward_mode in {"per_step", "history_buffer"}:
            observations = (
                env_output.final_obs
                if env_output.final_obs is not None
                else env_output.obs
            )
        elif self.reward_mode == "terminal" and env_output.final_obs is not None:
            observations = env_output.final_obs
        else:
            return None
        reward_input = dict(observations)
        if env_output.env_infos is not None:
            reward_input["env_infos"] = self._select_reward_env_infos(
                env_output.env_infos
            )

        dones = env_output.dones
        if dones is not None and getattr(dones, "ndim", 0) > 1:
            dones = dones[:, -1]
        if dones is not None:
            reward_input["dones"] = dones

        if self.reward_mode == "history_buffer":
            if stage_id is None:
                raise ValueError("stage_id is required for history-buffer reward.")
            history_manager = self.train_history_managers[stage_id]
            emit_mask = (
                dones.to(dtype=torch.bool)
                if self.use_completed_episode_buffer and dones is not None
                else torch.ones(self.train_num_envs_per_stage, dtype=torch.bool)
            )
            query_info_all = {
                env_id: (
                    int(self._episode_chunk_ids[stage_id][env_id]),
                    int(len(history_manager.history_entries[env_id])),
                    int(history_manager.pickup_counts[env_id]),
                    list(history_manager.success_history_entries[env_id]),
                )
                for env_id in range(self.train_num_envs_per_stage)
                if bool(emit_mask[env_id])
            }
            for env_id, (episode_id, *_rest) in query_info_all.items():
                if episode_id < 0:
                    self._ensure_expected_chunk(
                        allowed=False,
                        stage_id=stage_id,
                        reason="query_input_missing",
                        env_id=env_id,
                        episode_id=episode_id,
                    )
            query_emit_mask = emit_mask.clone().to(dtype=torch.bool)
            if not self.use_completed_episode_buffer:
                for env_id, (
                    episode_id,
                    history_len,
                    pickup_count,
                    _success_trace,
                ) in query_info_all.items():
                    window_chunk_count = self._window_chunk_count_for_episode(
                        stage_id, env_id, episode_id
                    )
                    if window_chunk_count > 0:
                        continue
                    query_emit_mask[env_id] = False
                    reason = (
                        "pickup_prefix_excluded"
                        if history_len <= pickup_count
                        else "outside_current_window"
                    )
                    self._record_chunk_drop(
                        stage_id,
                        reason,
                        count=0,
                        env_id=env_id,
                        episode_id=episode_id,
                        history_len=history_len,
                        pickup_count=pickup_count,
                        window_chunk_count=window_chunk_count,
                    )
            query_info = {
                env_id: info
                for env_id, info in query_info_all.items()
                if bool(query_emit_mask[env_id])
            }
            # Capture pickup_counts BEFORE build_history_input clears done envs
            # (delta mode needs the per-env pickup offset to select chunk-boundary
            # frames in RobometerHistoryRewardModel.compute_reward).
            _delta_pickup_counts = list(history_manager.pickup_counts)
            history_input, history_lengths = history_manager.build_history_input(
                dones=dones, emit_mask=query_emit_mask
            )
            reward_input["history_input"] = history_input
            self.history_lengths[stage_id] = dict(history_lengths)
            emitted_env_ids = self._history_input_emitted_env_ids(history_input)
            if not hasattr(self, "_last_history_query_info"):
                self._last_history_query_info = {}
            query_info_with_refs = {}
            funnel_state = self._window_chunk_funnel[stage_id]
            for env_id in sorted(emitted_env_ids):
                if env_id not in query_info:
                    self._ensure_expected_chunk(
                        allowed=False,
                        stage_id=stage_id,
                        reason="query_input_missing",
                        env_id=env_id,
                    )
                    continue
                episode_id, history_len, pickup_count, success_trace = query_info[env_id]
                if history_len != len(success_trace):
                    self._ensure_expected_chunk(
                        allowed=False,
                        stage_id=stage_id,
                        reason="query_input_missing",
                        env_id=env_id,
                        episode_id=episode_id,
                        history_len=history_len,
                        success_trace_len=len(success_trace),
                    )
                chunk_refs = tuple(
                    self._window_chunk_slices_for_episode(stage_id, env_id, episode_id)
                )
                window_chunk_count = len(chunk_refs)
                funnel_state["query_units"] += 1
                funnel_state["query_units_by_env_episode"].add((env_id, episode_id))
                funnel_state["queried_chunks"] += int(window_chunk_count)
                query_info_with_refs[env_id] = (
                    episode_id,
                    history_len,
                    pickup_count,
                    success_trace,
                    chunk_refs,
                )
                if window_chunk_count <= 0:
                    self._ensure_expected_chunk(
                        allowed=False,
                        stage_id=stage_id,
                        reason="query_input_missing",
                        env_id=env_id,
                        episode_id=episode_id,
                        history_len=history_len,
                        pickup_count=pickup_count,
                        window_chunk_count=window_chunk_count,
                    )
                if self._chunk_debug_active():
                    self._log_chunk_debug(
                        "CHUNK_QUERY_UNIT",
                        stage_id=stage_id,
                        env_id=env_id,
                        episode_id=episode_id,
                        is_done=bool(dones[env_id]) if dones is not None else False,
                        history_len=history_len,
                        pickup_count=pickup_count,
                        window_chunk_count=window_chunk_count,
                    )
                for local_chunk_idx, episode_chunk_idx in self._window_chunk_slices_for_episode(
                    stage_id, env_id, episode_id
                ):
                    funnel_state["queried_chunk_refs"].add(
                        (env_id, episode_id, local_chunk_idx, episode_chunk_idx)
                    )
            self._last_history_query_info[stage_id] = query_info_with_refs
            if self.reward_shaping == "delta":
                reward_input["shaping"] = "delta"
                reward_input["pickup_counts"] = _delta_pickup_counts
                # chunk_size = num_action_chunks matches assign_history_reward's
                # chunk_size (rollout_rewards[-1].shape[-1]); execute_action_chunks
                # == num_action_chunks in this config so the boundary frame interval
                # is the correct low-level step count per chunk.
                reward_input["chunk_size"] = int(self.model_cfg.num_action_chunks)
            # Auto-reset prepend: build_history_input just cleared done envs;
            # the env's reset (inside the preceding chunk_step) already stashed
            # this env's new pick-up frames via ManiskillEnv._render_pickup_frames.
            # Prepend them now so the next chunk's append lands insert frames
            # AFTER the pick-up (robometer sees the full pick-up+insert video).
            # Pick-up frames never become rollout steps -> not in RL training data.
            if dones is not None and bool(dones.any()):
                _consume = get_env_attr(
                    self.env_list[stage_id], "consume_pickup_frames"
                )
                _consume = _consume() if callable(_consume) else (_consume or {})
                _hm = self.train_history_managers[stage_id]
                for _e in dones.nonzero(as_tuple=False).reshape(-1).tolist():
                    if _consume and _e in _consume and _consume[_e]:
                        _hm.prepend_history_entries(int(_e), _consume[_e])

        if last_run and self.reward_mode != "history_buffer":
            reward_input.update(
                {
                    "last_run": torch.ones(
                        (self.train_num_envs_per_stage, 1), dtype=torch.bool
                    )
                }
            )
        if self.reward_mode == "history_buffer":
            if self.use_completed_episode_buffer:
                if dones is None or not bool(dones.any()):
                    return None
            else:
                has_done = dones is not None and bool(dones.any())
                if not last_run and not has_done:
                    return None
                if not self._last_history_query_info.get(stage_id):
                    return None
        elif not (last_run or (dones is not None and bool(dones.any()))):
            return None
        self.send_to(
            group_name=self.cfg.reward.group_name,
            channel=send_channel,
            data=reward_input,
            tag="train_reward_obs",
            async_op=True,
            decoupled_mode=self.env_decoupled_mode,
        )
        reward_output = self.recv_from(
            group_name=self.cfg.reward.group_name,
            channel=recv_channel,
            tag="train_reward_obs",
            batch_size=self.train_batch_size,
            decoupled_mode=self.env_decoupled_mode,
        )
        if self.reward_mode == "history_buffer":
            expected_queries = getattr(self, "_last_history_query_info", {}).get(
                stage_id, {}
            )
            if expected_queries and reward_output is None:
                self._ensure_expected_chunk(
                    allowed=False,
                    stage_id=stage_id,
                    reason="query_output_missing",
                    query_units=len(expected_queries),
                )
        if __import__("os").environ.get("RLINF_REWARD_DEBUG"):
            try:
                with open(_rdebug_log_path(), "a") as _f:
                    _f.write(
                        f"EW get_reward_model_output RECV rank={getattr(self, '_rank', -1)} reward_output={type(reward_output)} is_none={reward_output is None}\n"
                    )
            except Exception:
                pass
        if self.reward_mode != "terminal" or reward_output is None:
            return reward_output
        return self._scatter_terminal_reward_output(
            env_output=env_output, reward_output=reward_output
        )

    def _select_reward_env_infos(self, env_infos: dict[str, Any]) -> dict[str, Any]:
        reward_env_infos = {}
        for key in self.env_infos_reward_keys:
            if key not in env_infos:
                continue
            reward_env_infos[key] = clone_nested_to_cpu(env_infos[key])
        return reward_env_infos

    def _scatter_terminal_reward_output(
        self,
        env_output: EnvOutput,
        reward_output: torch.Tensor,
    ) -> torch.Tensor:
        if env_output.rewards is None or env_output.dones is None:
            return reward_output

        done_envs = env_output.dones.any(dim=1)
        sparse_rewards = torch.zeros_like(env_output.rewards, dtype=reward_output.dtype)
        if not done_envs.any():
            return sparse_rewards

        done_steps = env_output.dones.to(torch.int64).argmax(dim=1)
        sparse_rewards[done_envs, done_steps[done_envs]] = (
            reward_output[done_envs].reshape(-1).to(sparse_rewards.dtype)
        )
        return sparse_rewards

    def assign_history_reward(
        self, stage_id: int, reward_model_output: torch.Tensor
    ) -> dict[int, RobometerEpisodeReward]:
        rollout_rewards = self.rollout_results[stage_id].rewards
        if not rollout_rewards:
            return {}
        reward = (self.reward_weight * reward_model_output).to(
            rollout_rewards[-1].dtype
        )
        if reward_model_output.dim() != 2:
            reward_assign_lengths = [
                min(
                    history_buffer_length[env_id]
                    for history_buffer_length in self.history_lengths[stage_id].values()
                )
                for env_id in range(self.train_num_envs_per_stage)
            ]
            for env_id, reward_assign_length in enumerate(reward_assign_lengths):
                for reward_assign_step in range(
                    2, min(reward_assign_length, len(rollout_rewards)) + 1
                ):
                    rollout_rewards[-reward_assign_step][env_id] += reward[env_id]
            return {}

        stash = getattr(self, "_last_history_query_info", {}).pop(stage_id, {})
        if not stash:
            return {}
        max_frames = int(self.cfg.reward.model.get("max_robometer_frames", 60))
        fail_shift = float(self.cfg.reward.model.get("fail_shift", 1.0))
        chunk_size = int(rollout_rewards[-1].shape[-1])
        delta_mode = self.reward_shaping == "delta"
        assignments: dict[int, RobometerEpisodeReward] = {}
        funnel_state = self._window_chunk_funnel[stage_id]
        for env_id, entry in stash.items():
            if len(entry) == 4:
                episode_id, history_len, pickup_count, success_trace = entry
                chunk_refs_snapshot = tuple(
                    self._window_chunk_slices_for_episode(stage_id, env_id, episode_id)
                )
            else:
                (
                    episode_id,
                    history_len,
                    pickup_count,
                    success_trace,
                    chunk_refs_snapshot,
                ) = entry
            insert_steps = history_len - pickup_count
            total_chunks = math.ceil(insert_steps / chunk_size)
            env_progress = reward[env_id].detach().cpu().numpy().astype(np.float32)
            if delta_mode:
                # Delta shaping: progress is one value per chunk-boundary frame
                # (total_chunks + 1). max_robometer_frames is NOT applied; the
                # boundary frame count is already bounded by episode chunk count.
                expected_progress = len(
                    _robometer_boundary_frame_indices(
                        history_len, pickup_count, chunk_size
                    )
                )
                if env_progress.shape[0] < expected_progress:
                    raise ValueError(
                        "Robometer boundary progress is shorter than the completed "
                        f"episode's boundary frame count: env_id={env_id}, "
                        f"expected={expected_progress}, got={env_progress.shape[0]}."
                    )
                assignment = reconstruct_robometer_delta_reward(
                    env_progress[:expected_progress],
                    history_len=history_len,
                    pickup_count=pickup_count,
                    success_trace=success_trace,
                    chunk_size=chunk_size,
                    total_chunks=total_chunks,
                    success_bonus=self.delta_success_bonus,
                    failure_terminal_penalty=self.delta_failure_terminal_penalty,
                )
            else:
                expected_progress = len(
                    _robometer_downsample_indices(history_len, max_frames)
                )
                if env_progress.shape[0] < expected_progress:
                    raise ValueError(
                        "Robometer progress is shorter than the completed episode's "
                        f"downsampled history: env_id={env_id}, "
                        f"expected={expected_progress}, got={env_progress.shape[0]}."
                    )
                assignment = reconstruct_robometer_episode_reward(
                    env_progress[:expected_progress],
                    history_len=history_len,
                    pickup_count=pickup_count,
                    success_trace=success_trace,
                    max_frames=max_frames,
                    fail_shift=fail_shift,
                    chunk_size=chunk_size,
                    total_chunks=total_chunks,
                    success_terminal_bonus=getattr(
                        self, "absolute_success_terminal_bonus", 0.0
                    ),
                )
            if __import__("os").environ.get("RLINF_REWARD_DEBUG"):
                try:
                    with open(_rdebug_log_path(), "a") as _f:
                        _n_boundary = (
                            len(assignment.downsample_indices)
                            if delta_mode
                            else 0
                        )
                        _delta_sum = float(
                            np.asarray(assignment.chunk_reward[:, 0]).sum()
                        ) if delta_mode else 0.0
                        _success_chunks = int(
                            np.asarray(assignment.chunk_reward[:, 0] > 0).sum()
                        ) if delta_mode else 0
                        _f.write(
                            f"[assign] env_id={env_id} shaping={self.reward_shaping} "
                            f"history_len={history_len} pickup={pickup_count} "
                            f"total_chunks={total_chunks} chunk_size={chunk_size} "
                            f"n_boundary={_n_boundary} delta_sum={_delta_sum:.4f} "
                            f"success_chunks={_success_chunks} "
                            f"episode_success={assignment.episode_success}\n"
                        )
                except Exception:
                    pass
            assignments[env_id] = assignment

            if not self.use_completed_episode_buffer:
                chunk_refs = list(chunk_refs_snapshot)
                if not chunk_refs:
                    self._ensure_expected_chunk(
                        allowed=False,
                        stage_id=stage_id,
                        reason="assignment_uncovered_window_chunk",
                        env_id=env_id,
                        episode_id=episode_id,
                        history_len=history_len,
                        pickup_count=pickup_count,
                        window_chunk_count=0,
                    )
                assigned_chunk_refs: set[tuple[int, int]] = set()
                for local_chunk_idx, episode_chunk_idx in chunk_refs:
                    if episode_chunk_idx >= assignment.chunk_reward.shape[0]:
                        self._ensure_expected_chunk(
                            allowed=False,
                            stage_id=stage_id,
                            reason="assignment_oob",
                            env_id=env_id,
                            episode_id=episode_id,
                            local_chunk_idx=local_chunk_idx,
                            episode_chunk_idx=episode_chunk_idx,
                            history_len=history_len,
                            pickup_count=pickup_count,
                            window_chunk_count=len(chunk_refs),
                        )
                    ref_key = (local_chunk_idx, episode_chunk_idx)
                    if ref_key in assigned_chunk_refs:
                        self._ensure_expected_chunk(
                            allowed=False,
                            stage_id=stage_id,
                            reason="assignment_ref_mismatch",
                            env_id=env_id,
                            episode_id=episode_id,
                            local_chunk_idx=local_chunk_idx,
                            episode_chunk_idx=episode_chunk_idx,
                            history_len=history_len,
                            pickup_count=pickup_count,
                            window_chunk_count=len(chunk_refs),
                        )
                    assigned_chunk_refs.add(ref_key)
                    target = rollout_rewards[local_chunk_idx]
                    chunk_mask = np.asarray(
                        assignment.chunk_loss_mask[episode_chunk_idx], dtype=bool
                    )
                    if not bool(chunk_mask.all()):
                        self._ensure_expected_chunk(
                            allowed=False,
                            stage_id=stage_id,
                            reason="loss_mask_false_after_assignment",
                            env_id=env_id,
                            episode_id=episode_id,
                            local_chunk_idx=local_chunk_idx,
                            episode_chunk_idx=episode_chunk_idx,
                            history_len=history_len,
                            pickup_count=pickup_count,
                            window_chunk_count=len(chunk_refs),
                        )
                    target[env_id] = torch.as_tensor(
                        assignment.chunk_reward[episode_chunk_idx],
                        dtype=target.dtype,
                        device=target.device,
                    )
                    self.rollout_results[stage_id].loss_mask[local_chunk_idx][
                        env_id
                    ] = torch.as_tensor(
                        assignment.chunk_loss_mask[episode_chunk_idx],
                        dtype=torch.bool,
                        device=target.device,
                    )
                    funnel_state["assigned_chunks"] += 1
                    funnel_state["assigned_chunk_refs"].add(
                        (env_id, episode_id, local_chunk_idx, episode_chunk_idx)
                    )
                missing_refs = {
                    (env_id, episode_id, local_chunk_idx, episode_chunk_idx)
                    for local_chunk_idx, episode_chunk_idx in chunk_refs
                } - {
                    (env_id, episode_id, local_chunk_idx, episode_chunk_idx)
                    for local_chunk_idx, episode_chunk_idx in assigned_chunk_refs
                }
                if missing_refs:
                    self._ensure_expected_chunk(
                        allowed=False,
                        stage_id=stage_id,
                        reason="assignment_uncovered_window_chunk",
                        env_id=env_id,
                        episode_id=episode_id,
                        history_len=history_len,
                        pickup_count=pickup_count,
                        window_chunk_count=len(chunk_refs),
                    )

        expected_refs = funnel_state["queried_chunk_refs"]
        if expected_refs:
            missing_assignment_refs = expected_refs - funnel_state["assigned_chunk_refs"]
            if missing_assignment_refs:
                sample = sorted(missing_assignment_refs)[0]
                self._ensure_expected_chunk(
                    allowed=False,
                    stage_id=stage_id,
                    reason="assignment_uncovered_window_chunk",
                    env_id=sample[0],
                    episode_id=sample[1],
                    local_chunk_idx=sample[2],
                    episode_chunk_idx=sample[3],
                )

        if self.use_completed_episode_buffer:
            self.completed_episode_buffers[stage_id].add_rewards(assignments)
        return assignments

    @Worker.timer("env/bootstrap_step")
    def bootstrap_step(self) -> list[EnvOutput]:
        def get_zero_dones() -> torch.Tensor:
            return (
                torch.zeros((self.train_num_envs_per_stage,), dtype=bool)
                .unsqueeze(1)
                .repeat(1, self.model_cfg.num_action_chunks)
            )

        env_outputs: list[EnvOutput] = []
        if not self.cfg.env.train.auto_reset:
            for stage_id in range(self.stage_num):
                self.env_list[stage_id].is_start = True
                extracted_obs, infos = self.env_list[stage_id].reset()
                dones = get_zero_dones()
                terminations = dones.clone()
                truncations = dones.clone()

                env_output = EnvOutput(
                    obs=extracted_obs,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    final_obs=(
                        infos["final_observation"]
                        if "final_observation" in infos
                        else None
                    ),
                    intervene_actions=None,
                    intervene_flags=None,
                )
                env_outputs.append(env_output)
        else:
            dones = get_zero_dones()
            terminations = dones.clone()
            truncations = dones.clone()

            for stage_id in range(self.stage_num):
                env_output = EnvOutput(
                    obs=self.last_obs_list[stage_id],
                    rewards=None,
                    dones=dones,
                    terminations=terminations,
                    truncations=truncations,
                    intervene_actions=self.last_intervened_info_list[stage_id][0],
                    intervene_flags=self.last_intervened_info_list[stage_id][1],
                )
                env_outputs.append(env_output)

        return env_outputs

    def _send_train_bootstrap(
        self, rollout_channel: Channel, env_outputs: list[EnvOutput]
    ) -> None:
        for stage_id in range(self.stage_num):
            env_output: EnvOutput = env_outputs[stage_id]
            env_batch = env_output.to_dict()
            self.send_to(
                group_name=self.cfg.rollout.group_name,
                channel=rollout_channel,
                data={
                    "obs": env_batch["obs"],
                    "final_obs": env_batch["final_obs"],
                },
                mode="train",
                tag="rollout_results",
                decoupled_mode=self.env_decoupled_mode,
            )

    def _bootstrap_and_send_train(self, rollout_channel: Channel) -> list[EnvOutput]:
        env_outputs = self.bootstrap_step()
        self._send_train_bootstrap(rollout_channel, env_outputs)
        return env_outputs

    def prefetch_train_bootstrap(self, rollout_channel: Channel) -> None:
        """Prepare and send the first env batch for the next training rollout."""
        # Independent windows: the env+history reset happens at the END of
        # _run_interact_once (replacing store_last_obs). The runner overlaps this
        # prefetch with actor training (embodied_runner), so it could read a stale
        # last_obs_list (pre-reset) and cache the previous window's state for the
        # next window. Disable prefetch entirely; _run_interact_once then falls
        # through to _bootstrap_and_send_train on a fresh last_obs_list. Cost: a
        # small latency (no overlap of the first env batch with actor compute).
        if self.independent_rollout_windows:
            return
        if self._prefetched_train_bootstrap is not None:
            raise RuntimeError(
                "A prefetched train bootstrap already exists. "
                "Call interact() to consume it before prefetching again."
            )
        self._prefetched_train_bootstrap = self._bootstrap_and_send_train(
            rollout_channel
        )

    def record_env_metrics(
        self,
        env_metrics: dict[str, list],
        env_info: dict[str, Any],
    ):
        for key, value in env_info.items():
            env_metrics.setdefault(key, []).append(value)

    def store_last_obs_and_intervened_info(self, env_output_list: list[EnvOutput]):
        self.last_obs_list = [env_output.obs for env_output in env_output_list]
        self.last_intervened_info_list = [
            (env_output.intervene_actions, env_output.intervene_flags)
            for env_output in env_output_list
        ]

    @staticmethod
    def _validate_rollout_window_mode(
        rollout_window_mode: str,
        auto_reset: bool,
        history_train_mode: str,
        rollout_epoch: int,
    ) -> bool:
        """Validate env.train.rollout_window_mode and return whether independent.

        Pure (no self) so unit tests can exercise the validation without
        constructing a distributed EnvWorker. continuous is the legacy
        cross-window continuation; independent force-closes unfinished episodes
        at each window boundary and requires auto_reset=true,
        history_train_mode='rollout_window' (not complete_episode, which retains
        trajectories across windows), and rollout_epoch=1 (each internal epoch
        is a window; otherwise reset would cut mid-trajectory).
        """
        if rollout_window_mode not in {"continuous", "independent"}:
            raise ValueError(
                "env.train.rollout_window_mode must be 'continuous' or "
                f"'independent', got {rollout_window_mode!r}."
            )
        independent = rollout_window_mode == "independent"
        if independent:
            if not auto_reset:
                raise ValueError(
                    "env.train.rollout_window_mode='independent' requires "
                    "env.train.auto_reset=true."
                )
            if history_train_mode == "complete_episode":
                raise ValueError(
                    "Independent rollout windows require "
                    "reward.history_train_mode='rollout_window'; "
                    "'complete_episode' retains trajectories across windows."
                )
            if rollout_epoch != 1:
                raise ValueError(
                    "Independent rollout windows currently require rollout_epoch=1 "
                    f"(got {rollout_epoch}); each internal epoch would otherwise "
                    "reset mid-trajectory."
                )
        return independent

    def _finalize_independent_window_boundary(
        self,
        env_output: EnvOutput,
        stage_id: int,
        env_metrics: dict[str, list],
    ) -> EnvOutput:
        """Force unfinished env slots to end at an independent window boundary.

        Models the artificial window boundary as a *truncation* (time-limit cut),
        never a task *termination*: only the last action-chunk step ``[:, -1]`` is
        touched, naturally-done envs keep their original flags, and forced-timeout
        envs get ``dones[:, -1]=True`` + ``truncations[:, -1]|=True`` so the post-loop
        Robometer query settles them and GAE stops bootstrapping at the boundary.
        Returns a *new* EnvOutput (cloned tensors); the original tensors owned by the
        already-recorded ChunkStepResults are left untouched, so the synthetic
        terminal lands only on the trajectory's T+1 bootstrap boundary, not on the
        last action chunk.
        """
        if env_output.dones is None:
            raise RuntimeError("Cannot finalize rollout window without done flags.")

        dones = env_output.dones.clone().to(dtype=torch.bool)
        terminations = (
            env_output.terminations.clone().to(dtype=torch.bool)
            if env_output.terminations is not None
            else torch.zeros_like(dones)
        )
        truncations = (
            env_output.truncations.clone().to(dtype=torch.bool)
            if env_output.truncations is not None
            else torch.zeros_like(dones)
        )

        naturally_done = dones[:, -1].clone()
        forced_timeout = ~naturally_done

        dones[:, -1] = True
        truncations[:, -1] = torch.logical_or(truncations[:, -1], forced_timeout)

        self._independent_window_forced_timeout_masks[stage_id] = forced_timeout

        env_metrics["window/episodes"].append(
            torch.tensor([dones.shape[0]], dtype=torch.float32)
        )
        env_metrics["window/natural_terminal_episodes"].append(
            torch.tensor([naturally_done.sum().item()], dtype=torch.float32)
        )
        env_metrics["window/forced_timeout_episodes"].append(
            torch.tensor([forced_timeout.sum().item()], dtype=torch.float32)
        )
        env_metrics["window/forced_timeout_fraction"].append(
            forced_timeout.float().mean().reshape(1).cpu()
        )

        return EnvOutput(
            obs=env_output.obs,
            final_obs=env_output.final_obs,
            rewards=env_output.rewards,
            env_infos=env_output.env_infos,
            dones=dones,
            terminations=terminations,
            truncations=truncations,
            intervene_actions=env_output.intervene_actions,
            intervene_flags=env_output.intervene_flags,
        )

    def _reset_train_stage_for_next_independent_window(self, stage_id: int) -> None:
        """Reset one train stage's env + history for the next independent window.

        Order matters: this is called *after* the post-loop Robometer query and
        reward assignment (and after trajectories are sent), so settling is
        already complete. ``env.is_start=True`` + ``reset()`` re-renders the
        pick-up prefix on the GPU env; ``reset_all`` wipes the previous window's
        history so the next Robometer video never spans windows; then the fresh
        pick-up frames are prepended. ``last_obs_list`` is overwritten with the
        fresh reset observation so the next ``bootstrap_step`` does not resume.
        """
        env = self.env_list[stage_id]
        if hasattr(env, "reset_all_for_rollout_window"):
            extracted_obs, _ = env.reset_all_for_rollout_window()
        else:
            env.is_start = True
            extracted_obs, _ = env.reset()

        if self.reward_mode == "history_buffer":
            history_manager = self.train_history_managers[stage_id]
            history_manager.reset_all()

            consume_pickup_frames = get_env_attr(env, "consume_pickup_frames")
            pickup_frames = (
                consume_pickup_frames()
                if callable(consume_pickup_frames)
                else (consume_pickup_frames or {})
            )
            for env_id, frames in pickup_frames.items():
                if frames:
                    history_manager.prepend_history_entries(int(env_id), frames)

        self.last_obs_list[stage_id] = extracted_obs
        self.last_intervened_info_list[stage_id] = (None, None)

    def _assert_independent_forced_timeouts_settled(
        self,
        stage_id: int,
        forced_timeout: torch.Tensor,
        assignments: dict[int, "RobometerEpisodeReward"],
    ) -> None:
        """Verify every forced-timeout env WITH a window trajectory was settled.

        Forbids silently dropping unfinished trajectories. ``assign_history_reward``
        pops ``_last_history_query_info[stage]`` internally, so the authoritative
        record of what was settled is the returned ``assignments`` dict: each forced
        env whose current episode has window chunk refs must have an assignment, and
        that assignment must be a failure (``episode_success is False`` — a synthetic
        truncation must never look like a task success). A forced env whose current
        episode has 0 window chunk refs is SKIPPED: it auto-reset into a fresh
        episode at the boundary (its previous episode was naturally completed +
        settled mid-window), so there is no trajectory in this window to drop.
        Raises ``RuntimeError`` (never a silent skip) with enough context to debug
        the funnel.
        """
        forced_env_ids = (
            torch.nonzero(forced_timeout, as_tuple=False).flatten().tolist()
        )
        for env_id in forced_env_ids:
            episode_id = int(self._episode_chunk_ids[stage_id][env_id])
            chunk_refs = tuple(
                self._window_chunk_slices_for_episode(stage_id, env_id, episode_id)
            )
            if not chunk_refs:
                # Forced-timeout env whose current episode has 0 window chunks:
                # it auto-reset into a fresh episode right at the boundary (its
                # previous episode was naturally completed + settled mid-window).
                # There is no trajectory in this window to settle, so this is
                # NOT a dropped trajectory — skip it (keep checking the rest).
                continue
            assignment = assignments.get(env_id)
            if assignment is None:
                raise RuntimeError(
                    f"[independent-window] forced-timeout env {env_id} "
                    f"(stage {stage_id}, episode {episode_id}) has no reward "
                    "assignment; unfinished trajectory was not queried/settled "
                    "by Robometer (was reward_model_output None?)."
                )
            if assignment.episode_success:
                raise RuntimeError(
                    f"[independent-window] forced-timeout env {env_id} "
                    f"(stage {stage_id}, episode {episode_id}) settled as "
                    "success; synthetic truncation must be a failure."
                )

    def _skip_zero_success_windows(
        self, env_metrics: dict[str, list]
    ) -> None:
        """Mask (skip) SR==0 all-fail windows before sending trajectories.

        For each stage whose current window had NO successful episode
        (``_window_any_episode_success[stage_id]`` is False, i.e. window-level
        ``episode_success_rate == 0``), set the whole trajectory's per-chunk
        ``loss_mask`` to False. The actor's token-mean loss (``masked_mean``)
        then returns 0 for those chunks -> 0 gradient -> the all-fail batch is a
        no-op update (the policy does not move on noise-only failures), without
        breaking the FSDP collective (backward still runs with 0 grad). Stages
        with at least one success keep their loss_mask untouched.
        """
        for stage_id in range(self.stage_num):
            skipped = not self._window_any_episode_success[stage_id]
            if skipped:
                rollout_result = self.rollout_results[stage_id]
                for lm in rollout_result.loss_mask:
                    if lm is not None:
                        lm.fill_(False)
            env_metrics["window/skipped_zero_success"].append(
                torch.tensor([1.0 if skipped else 0.0], dtype=torch.float32)
            )

    @Worker.timer("env/send_rollout_trajectories")
    async def send_rollout_trajectories(
        self,
        rollout_result: EmbodiedRolloutResult | Trajectory,
        channel: Channel,
        stage_id: int = 0,
        keyed: bool = False,
    ):
        if keyed:
            # Deterministic per-actor-rank routing: split this env worker's batch by
            # CommMapper.get_dst_ranks shards and write each to a dedicated queue, so
            # every actor rank receives exactly its shard (no DEFAULT-queue racing).
            actor_splits = self.async_stage_actor_splits[stage_id]
            split_sizes = [size for _, size in actor_splits]
            if isinstance(rollout_result, EmbodiedRolloutResult):
                trajectories = rollout_result.to_splited_trajectories_by_sizes(
                    split_sizes
                )
                rollout_result.clear()
            else:
                trajectories = split_trajectory_by_sizes(rollout_result, split_sizes)
            for (actor_rank, _), trajectory in zip(actor_splits, trajectories):
                channel.put(
                    trajectory,
                    key=self.async_actor_keys[actor_rank],
                    async_op=True,
                )
            del trajectories
            gc.collect()
            return
        if isinstance(rollout_result, EmbodiedRolloutResult):
            trajectories = rollout_result.to_splited_trajectories(self.actor_split_num)
            rollout_result.clear()
        else:
            batch_size = int(rollout_result.rewards.shape[1])
            if batch_size % self.actor_split_num != 0:
                raise ValueError(
                    f"Completed episode batch {batch_size} is not divisible by "
                    f"actor_split_num={self.actor_split_num}."
                )
            trajectories = split_trajectory_by_sizes(
                rollout_result,
                [batch_size // self.actor_split_num] * self.actor_split_num,
            )
        for trajectory in trajectories:
            channel.put(trajectory, async_op=True)
        del trajectories
        gc.collect()

    def _completed_episode_batch(
        self, stage_id: int, rollout_result: EmbodiedRolloutResult
    ) -> Trajectory:
        buffer = self.completed_episode_buffers[stage_id]
        buffer.ingest(rollout_result.to_trajectory())
        return buffer.pop_batch(self.train_num_envs_per_stage)

    @staticmethod
    def _record_completed_episode_metrics(env_metrics, buffer) -> None:
        wait_rounds = list(buffer.completed_wait_rounds)
        metrics = {
            "reward/pending_episodes": buffer.pending_count,
            "reward/ready_episodes": buffer.ready_episodes,
            "reward/completed_episodes": buffer.completed_episodes,
            "reward/cross_rollout_episodes": buffer.cross_rollout_episodes,
            "reward/valid_chunks": buffer.last_batch_valid_chunks,
            "reward/padding_chunks": buffer.last_batch_padding_chunks,
            "reward/episode_wait_rollouts": (
                float(sum(wait_rounds) / len(wait_rounds)) if wait_rounds else 0.0
            ),
            "reward/superseded_completed_episodes": buffer.superseded_completed_episodes,
            "reward/selected_episode_version_min": buffer.selected_episode_version_min,
            "reward/selected_episode_version_max": buffer.selected_episode_version_max,
            "reward/selected_episode_wait_rollouts": buffer.selected_episode_wait_rollouts,
        }
        for key, value in metrics.items():
            env_metrics[key].append(torch.tensor([value], dtype=torch.float32))

    @Worker.timer("run_interact_once")
    async def _run_interact_once(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None,
        *,
        cooperative_yield: bool,
    ) -> dict[str, torch.Tensor]:
        if __import__("os").environ.get("RLINF_REWARD_DEBUG"):
            try:
                with open(_rdebug_log_path(), "a") as _f:
                    _f.write(
                        f"EW _run_interact_once rank={getattr(self, '_rank', -1)} "
                        f"reward_channel_none={reward_channel is None} "
                        f"use_reward_model={self.use_reward_model} "
                        f"use_external_reward_model={self.use_external_reward_model}\n"
                    )
            except Exception:
                pass
        self.rollout_results: list[EmbodiedRolloutResult] = [
            EmbodiedRolloutResult(
                max_episode_length=self.cfg.env.train.max_episode_steps,
            )
            for _ in range(self.stage_num)
        ]
        self._reset_window_chunk_refs()
        env_metrics = defaultdict(list)

        for epoch in range(self.rollout_epoch):
            env_outputs = self.bootstrap_step()
            for stage_id in range(self.stage_num):
                if epoch == 0 and self._prefetched_train_bootstrap is not None:
                    env_outputs = self._prefetched_train_bootstrap
                    self._prefetched_train_bootstrap = None
                else:
                    env_outputs = self._bootstrap_and_send_train(rollout_channel)

            for chunk_step_idx in range(self.n_train_chunk_steps):
                for stage_id in range(self.stage_num):
                    if cooperative_yield:
                        await asyncio.sleep(0)

                    env_output = env_outputs[stage_id]
                    curr_obs = env_output.obs
                    if env_output.intervene_actions is not None:
                        self.rollout_results[stage_id].update_last_actions(
                            env_output.intervene_actions,
                            env_output.intervene_flags,
                        )

                    reward_model_output = None
                    if reward_channel is not None and chunk_step_idx != 0:
                        reward_model_output = self.get_reward_model_output(
                            env_output,
                            send_channel=reward_channel,
                            recv_channel=input_channel,
                            stage_id=stage_id,
                        )
                        if reward_model_output is not None:
                            env_metrics["reward_model_output"].append(
                                reward_model_output.detach().float().reshape(-1).cpu()
                            )

                    rollout_result = self.recv_from(
                        group_name=self.cfg.rollout.group_name,
                        channel=input_channel,
                        tag="train_rollout_results",
                        batch_size=self.train_batch_size,
                        merge_fn=RolloutResult.merge_rollout_results,
                        infer_batch_size_fn=self._infer_rollout_batch_size,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    rewards = self.compute_bootstrap_rewards(
                        env_output, rollout_result.bootstrap_values, reward_model_output
                    )
                    rewards = self._history_reward_placeholder(
                        rewards, rollout_result
                    )
                    chunk_step_result = ChunkStepResult(
                        actions=rollout_result.forward_inputs.get("action", None),
                        prev_logprobs=(
                            rollout_result.prev_logprobs
                            if self.collect_prev_infos
                            else None
                        ),
                        prev_values=(
                            rollout_result.prev_values
                            if self.collect_prev_infos
                            else None
                        ),
                        forward_inputs=rollout_result.forward_inputs,
                        versions=rollout_result.versions,
                        dones=env_output.dones,
                        truncations=env_output.truncations,
                        terminations=env_output.terminations,
                        rewards=rewards,
                    )
                    self.rollout_results[stage_id].append_step_result(chunk_step_result)
                    self._record_window_chunk_ref(
                        stage_id, env_output, has_rewards=rewards is not None
                    )
                    if (
                        self.reward_mode == "history_buffer"
                        and self.history_reward_assign
                        and reward_model_output is not None
                    ):
                        assignments = self.assign_history_reward(
                            stage_id, reward_model_output
                        )
                        for key, values in robometer_assignment_metric_values(
                            assignments
                        ).items():
                            env_metrics[key].append(values)
                        if any(
                            a.episode_success for a in assignments.values()
                        ):
                            self._window_any_episode_success[stage_id] = True
                    if rollout_result.save_flags is not None:
                        self.rollout_results[stage_id].mark_last_step_with_flags(
                            rollout_result.save_flags
                        )

                    env_output, env_info = self.env_interact_step(
                        rollout_result.actions, stage_id
                    )
                    env_batch = env_output.to_dict()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data={
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="train",
                        tag="rollout_results",
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    if self.collect_transitions:
                        next_obs = (
                            env_output.final_obs
                            if env_output.dones.any() and self.cfg.env.train.auto_reset
                            else env_output.obs
                        )
                        self.rollout_results[stage_id].append_transitions(
                            curr_obs, next_obs
                        )

                    env_outputs[stage_id] = env_output
                    should_record = (
                        self.cfg.env.train.auto_reset
                        or self.cfg.env.train.ignore_terminations
                        or chunk_step_idx == self.n_train_chunk_steps - 1
                    )
                    if should_record:
                        self.record_env_metrics(env_metrics, env_info)
            # Independent rollout windows: force-close unfinished envs at the
            # window boundary BEFORE the post-loop Robometer query so their
            # done=True makes the reward worker settle them, and the T+1
            # bootstrap boundary carries the synthetic truncation (GAE stops
            # here; no cross-window bootstrap).
            if self.independent_rollout_windows:
                for stage_id in range(self.stage_num):
                    env_outputs[stage_id] = self._finalize_independent_window_boundary(
                        env_outputs[stage_id], stage_id, env_metrics
                    )
            for stage_id in range(self.stage_num):
                env_output = env_outputs[stage_id]
                if env_output.intervene_actions is not None:
                    self.rollout_results[stage_id].update_last_actions(
                        env_output.intervene_actions,
                        env_output.intervene_flags,
                    )

                reward_model_output = None
                if reward_channel is not None:
                    # In independent mode this post-loop query is the window's
                    # final settlement (last_run=True forces unfinished-prefix
                    # queries); rollout_epoch is pinned to 1 there.
                    last_run = (
                        self.independent_rollout_windows
                        or epoch == self.rollout_epoch - 1
                    )
                    reward_model_output = self.get_reward_model_output(
                        env_output,
                        send_channel=reward_channel,
                        recv_channel=input_channel,
                        stage_id=stage_id,
                        last_run=last_run,
                    )
                    if reward_model_output is not None:
                        env_metrics["reward_model_output"].append(
                            reward_model_output.detach().float().reshape(-1).cpu()
                        )
                rollout_result = self.recv_from(
                    group_name=self.cfg.rollout.group_name,
                    channel=input_channel,
                    tag="train_rollout_results",
                    batch_size=self.train_batch_size,
                    merge_fn=RolloutResult.merge_rollout_results,
                    infer_batch_size_fn=self._infer_rollout_batch_size,
                    decoupled_mode=self.env_decoupled_mode,
                )
                rewards = self.compute_bootstrap_rewards(
                    env_output, rollout_result.bootstrap_values, reward_model_output
                )
                if (
                    self.reward_mode == "history_buffer"
                    and self.history_reward_assign
                ):
                    # The post-loop recv is the bootstrap value for the next state,
                    # not an additional trainable action chunk. Robometer rewards
                    # queried here must be scattered onto the already-retained
                    # rollout window; adding a placeholder reward here would make
                    # loss_mask/rewards one step longer than versions/actions.
                    rewards = None
                    assignments: dict[int, RobometerEpisodeReward] = {}
                    if reward_model_output is not None:
                        assignments = self.assign_history_reward(
                            stage_id, reward_model_output
                        )
                        for key, values in robometer_assignment_metric_values(
                            assignments
                        ).items():
                            env_metrics[key].append(values)
                        if any(
                            a.episode_success for a in assignments.values()
                        ):
                            self._window_any_episode_success[stage_id] = True
                    # Independent windows: every forced-timeout env MUST be
                    # settled here. If reward_model_output was None (no Robometer
                    # query) assignments stays empty and this fires, surfacing
                    # dropped unfinished trajectories instead of silencing them.
                    if self.independent_rollout_windows:
                        forced_timeout = self._independent_window_forced_timeout_masks[
                            stage_id
                        ]
                        if forced_timeout is not None and bool(forced_timeout.any()):
                            self._assert_independent_forced_timeouts_settled(
                                stage_id, forced_timeout, assignments
                            )
                else:
                    rewards = self._history_reward_placeholder(rewards, rollout_result)
                chunk_step_result = ChunkStepResult(
                    prev_values=(
                        rollout_result.prev_values if self.collect_prev_infos else None
                    ),
                    dones=env_output.dones,
                    truncations=env_output.truncations,
                    terminations=env_output.terminations,
                    rewards=rewards,
                )
                self.rollout_results[stage_id].append_step_result(chunk_step_result)
                self._record_window_chunk_ref(
                    stage_id, env_output, has_rewards=rewards is not None
                )

            # Independent windows: skip SR==0 all-fail windows by masking the
            # whole trajectory's loss_mask False (no episode succeeded this
            # window). The actor's token-mean loss (masked_mean) then returns 0
            # for these chunks -> 0 grad -> no-op update (the all-fail batch
            # does not move the policy), without breaking the FSDP collective.
            if self.independent_rollout_windows:
                self._skip_zero_success_windows(env_metrics)

            if self.use_training_pipeline and actor_channel is not None:
                send_results: list[EmbodiedRolloutResult | Trajectory]
                for stage_id in range(self.stage_num):
                    rollout_result = self.rollout_results[stage_id]
                    if getattr(rollout_result, "loss_mask", None) is not None:
                        self._window_chunk_funnel[stage_id]["masked_in_chunks"] = int(
                            self._count_masked_in_chunks(rollout_result.loss_mask)
                        )
                    self._emit_chunk_funnel_metrics(stage_id, env_metrics)
                if self.use_completed_episode_buffer:
                    send_results = [
                        self._completed_episode_batch(stage_id, rollout_result)
                        for stage_id, rollout_result in enumerate(self.rollout_results)
                    ]
                    for buffer in self.completed_episode_buffers:
                        self._record_completed_episode_metrics(env_metrics, buffer)
                else:
                    send_results = self.rollout_results
                await self.send_rollout_trajectories_pipeline(
                    send_results, actor_channel
                )
                self.rollout_results: list[EmbodiedRolloutResult] = [
                    EmbodiedRolloutResult(
                        max_episode_length=self.cfg.env.train.max_episode_steps,
                    )
                    for _ in range(self.stage_num)
                ]
                self._reset_window_chunk_refs()

            # Independent windows: reset all train envs + clear history now
            # (after Robometer settlement and trajectory send) and overwrite
            # last_obs_list with the fresh reset observation. The next
            # bootstrap_step() then starts a genuinely new window instead of
            # resuming the previous sim state/history. Prefetch is disabled in
            # this mode, so no stale cached bootstrap can race the reset.
            if self.independent_rollout_windows:
                for stage_id in range(self.stage_num):
                    self._reset_train_stage_for_next_independent_window(stage_id)
            else:
                self.store_last_obs_and_intervened_info(env_outputs)
            self.finish_rollout()

        if not self.use_training_pipeline and actor_channel is not None:
            for stage_id in range(self.stage_num):
                send_result: EmbodiedRolloutResult | Trajectory
                rollout_result = self.rollout_results[stage_id]
                if getattr(rollout_result, "loss_mask", None) is not None:
                    self._window_chunk_funnel[stage_id]["masked_in_chunks"] = int(
                        self._count_masked_in_chunks(rollout_result.loss_mask)
                    )
                self._emit_chunk_funnel_metrics(stage_id, env_metrics)
                if self.use_completed_episode_buffer:
                    send_result = self._completed_episode_batch(
                        stage_id, self.rollout_results[stage_id]
                    )
                    buffer = self.completed_episode_buffers[stage_id]
                    self._record_completed_episode_metrics(env_metrics, buffer)
                else:
                    send_result = self.rollout_results[stage_id]
                await self.send_rollout_trajectories(
                    send_result,
                    actor_channel,
                    stage_id=stage_id,
                    keyed=self.cfg.algorithm.get("actor_channel_keyed_routing", False),
                )
            # reduce memory peak
            self.rollout_results = []
            self._reset_window_chunk_refs()
            gc.collect()

        for key, value in env_metrics.items():
            env_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return env_metrics

    @Worker.timer("interact")
    async def interact(
        self,
        input_channel: Channel,
        rollout_channel: Channel,
        reward_channel: Channel | None,
        actor_channel: Channel | None = None,
    ):
        env_metrics = await self._run_interact_once(
            input_channel,
            rollout_channel,
            reward_channel,
            actor_channel,
            cooperative_yield=False,
        )

        for env in self.env_list:
            if self.train_enable_offload:
                env.offload()

        return env_metrics

    def evaluate(self, input_channel: Channel, rollout_channel: Channel):
        eval_metrics = defaultdict(list)

        for eval_rollout_epoch in range(self.eval_rollout_epoch):
            if not self.cfg.env.eval.auto_reset or eval_rollout_epoch == 0:
                for stage_id in range(self.stage_num):
                    self.eval_env_list[stage_id].is_start = True
                    self.eval_prev_done[stage_id] = torch.zeros(
                        self.eval_num_envs_per_stage, dtype=torch.bool
                    )
                    extracted_obs, infos = self.eval_env_list[stage_id].reset()
                    env_output = EnvOutput(
                        obs=extracted_obs,
                        final_obs=(
                            infos["final_observation"]
                            if "final_observation" in infos
                            else None
                        ),
                    )
                    env_batch = env_output.to_dict()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data={
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                        tag="rollout_results",
                        decoupled_mode=self.env_decoupled_mode,
                    )

            for eval_step in range(self.n_eval_chunk_steps):
                for stage_id in range(self.stage_num):
                    rollout_results = self.recv_from(
                        group_name=self.cfg.rollout.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        batch_size=self.eval_batch_size,
                        infer_batch_size_fn=self._infer_rollout_batch_size
                        if self.env_decoupled_mode
                        else None,
                        decoupled_mode=self.env_decoupled_mode,
                    )
                    raw_chunk_actions = (
                        rollout_results.actions
                        if hasattr(rollout_results, "actions")
                        else rollout_results
                    )
                    if isinstance(raw_chunk_actions, torch.Tensor):
                        raw_chunk_actions = raw_chunk_actions.detach().cpu().numpy()
                    else:
                        raw_chunk_actions = np.asarray(raw_chunk_actions)
                    env_output, env_info = self.env_evaluate_step(
                        raw_chunk_actions, stage_id
                    )

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)

                    if self.cfg.env.eval.auto_reset:
                        if (
                            eval_rollout_epoch == self.eval_rollout_epoch - 1
                            and eval_step == self.n_eval_chunk_steps - 1
                        ):
                            continue
                    else:
                        if eval_step == self.n_eval_chunk_steps - 1:
                            continue
                    env_batch = env_output.to_dict()
                    self.send_to(
                        group_name=self.cfg.rollout.group_name,
                        channel=rollout_channel,
                        data={
                            "obs": env_batch["obs"],
                            "final_obs": env_batch["final_obs"],
                        },
                        mode="eval",
                        tag="rollout_results",
                        decoupled_mode=self.env_decoupled_mode,
                    )

            self.finish_rollout(mode="eval")
        for stage_id in range(self.stage_num):
            if self.eval_enable_offload:
                self.eval_env_list[stage_id].offload()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics

    def get_actor_split_num(self):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def compute_advantages_and_returns(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Advantages/returns are rollout-level quantities, so compute them before
        # splitting. After this point each channel item is an actor micro-batch that can
        # be trained directly without reconstructing the full rollout batch on actor.
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": rollout_batch["rewards"],
            "dones": rollout_batch["dones"],
            "values": rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "chunk_reward_aggregation": self.cfg.algorithm.get(
                "chunk_reward_aggregation", "sum"
            ),
            "loss_mask": rollout_batch.get("loss_mask", None),
            "loss_mask_sum": rollout_batch.get("loss_mask_sum", None),
            "normalize_advantages": self.cfg.algorithm.get("normalize_advantages", True)
            and not self.use_training_pipeline,
        }
        advantages_and_returns = calculate_adv_and_returns(**kwargs)
        rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            rollout_batch["loss_mask"] = kwargs["loss_mask"]
        if kwargs["loss_mask_sum"] is not None:
            rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]
        return rollout_batch

    def prepare_pipeline_batch(self, trajectory: Trajectory) -> dict[str, torch.Tensor]:
        batch = convert_trajectories_to_batch([trajectory])
        batch = preprocess_embodied_batch(
            batch,
            rollout_epoch=1,
            auto_reset=self.cfg.env.train.auto_reset,
            ignore_terminations=self.cfg.env.train.ignore_terminations,
            reward_type=self.cfg.algorithm.reward_type,
            filter_rewards=self.cfg.algorithm.get("filter_rewards", False),
            group_size=self.cfg.algorithm.group_size,
            rewards_lower_bound=self.cfg.algorithm.get("rewards_lower_bound", None),
            rewards_upper_bound=self.cfg.algorithm.get("rewards_upper_bound", None),
        )
        return self.compute_advantages_and_returns(batch)

    def pack_pipeline_micro_batches(
        self, batch: dict[str, torch.Tensor], actor_rank: int
    ) -> list[dict]:
        batch_size = batch["prev_logprobs"].shape[0] * batch["prev_logprobs"].shape[1]
        if self.shuffle_rollout:
            shuffle_id = torch.randperm(
                batch_size, generator=self.shuffle_generators[actor_rank]
            )
        else:
            shuffle_id = torch.arange(batch_size)

        flatten_batch = flatten_embodied_batch(batch, shuffle_id)
        micro_batch_size = self.cfg.actor.micro_batch_size
        assert batch_size % micro_batch_size == 0, (
            f"Batch size {batch_size} is not divisible by micro_batch_size {micro_batch_size}."
        )
        num_micro_batches = batch_size // micro_batch_size
        micro_batches = split_dict_to_chunk(flatten_batch, num_micro_batches, dim=0)
        return [pack_batch(micro_batch) for micro_batch in micro_batches]

    async def send_rollout_trajectories_pipeline(
        self,
        rollout_results: list[EmbodiedRolloutResult | Trajectory],
        channel: Channel,
    ) -> None:
        pending_batches: list[tuple[int, dict[str, torch.Tensor]]] = []
        batches_by_actor_rank: dict[int, list[dict[str, torch.Tensor]]] = defaultdict(
            list
        )

        with self.worker_timer("prepare_micro_batches"):
            for stage_id, rollout_result in enumerate(rollout_results):
                actor_splits = self.pipeline_stage_actor_splits[stage_id]
                split_sizes = [split_size for _, split_size in actor_splits]
                if isinstance(rollout_result, EmbodiedRolloutResult):
                    trajectories = rollout_result.to_splited_trajectories_by_sizes(
                        split_sizes
                    )
                else:
                    trajectories = split_trajectory_by_sizes(
                        rollout_result, split_sizes
                    )

                for (actor_rank, _), trajectory in zip(actor_splits, trajectories):
                    batch = self.prepare_pipeline_batch(trajectory)
                    pending_batches.append((actor_rank, batch))
                    batches_by_actor_rank[actor_rank].append(batch)

            if self.cfg.algorithm.get("normalize_advantages", True):
                for actor_rank, batches in sorted(batches_by_actor_rank.items()):
                    local_adv_stats = sum(
                        masked_stats(batch["advantages"], batch.get("loss_mask"))
                        for batch in batches
                    )
                    env_ranks = self.pipeline_actor_env_ranks[actor_rank]
                    global_adv_stats = sum(
                        self.broadcast(
                            local_adv_stats if self._rank == src_rank else None,
                            groups=[(self._group_name, env_ranks)],
                            src=(self._group_name, src_rank),
                        )
                        for src_rank in env_ranks
                    )
                    for batch in batches:
                        batch["advantages"] = normalize_from_stats(
                            batch["advantages"], global_adv_stats
                        )

            for actor_rank, batch in pending_batches:
                for micro_batch in self.pack_pipeline_micro_batches(batch, actor_rank):
                    channel.put(
                        micro_batch,
                        key=self.pipeline_actor_keys[actor_rank],
                        async_op=True,
                    )
