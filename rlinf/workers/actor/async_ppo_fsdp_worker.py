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
import os
import queue
import threading
import time
from typing import Any, Optional

import numpy as np
import torch

from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.priority_store import PriorityStore
from rlinf.data.staleness_mask import compute_staleness_mask, count_fresh_chunks
from rlinf.scheduler import CommMapper, Worker
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_embodied_reward_metrics,
    compute_rollout_metrics,
)
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory, masked_mean, reshape_entropy
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def flatten_rollout_batch_for_train(
    nested_dict: dict, shuffle_id: Optional[torch.Tensor]
) -> dict:
    """Flatten [T, B, ...] rollout tensors to [T*B, ...] for actor training."""
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            if isinstance(value, torch.Tensor):
                value = value[:-1]

        if "env_info" in key:
            raise NotImplementedError("env_info nested dict is not supported here")

        if value is None:
            ret_dict[key] = None
            continue

        if isinstance(value, torch.Tensor):
            flat = value.reshape(-1, *value.shape[2:])
            ret_dict[key] = flat[shuffle_id] if shuffle_id is not None else flat
        elif isinstance(value, dict):
            ret_dict[key] = flatten_rollout_batch_for_train(value, shuffle_id)
        else:
            raise NotImplementedError(
                f"Unsupported value type in rollout batch: key={key}, type={type(value)}"
            )

    return ret_dict


class AsyncPPOEmbodiedFSDPActor(EmbodiedFSDPActor):
    """Embodied FSDP actor worker for async PPO / decoupled actor-critic training."""

    should_stop = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_store_size = self.cfg.algorithm.get(
            "rollout_store_size_per_rank", 1
        )
        self.rollout_store = PriorityStore(maxsize=self.rollout_store_size)
        # Captured by the receive thread on a fatal error so the readiness
        # collective can sync an all-reduce abort instead of deadlocking.
        self._recv_thread_exc: Exception | None = None

    async def recv_rollout_trajectories(self, input_channel):
        # drain channel
        if getattr(self, "_recv_queue", None) is None:
            self._recv_queue = queue.Queue()
        if (
            getattr(self, "_recv_rollout_thread", None) is None
            or not self._recv_rollout_thread.is_alive()
        ):
            self._recv_thread_exc = None
            self._recv_rollout_thread = threading.Thread(
                target=self._recv_rollout_thread_main,
                args=(input_channel,),
                daemon=True,
            )
            self._recv_rollout_thread.start()

    def _recv_rollout_thread_main(self, input_channel):
        staleness_filter_mode = self.cfg.algorithm.get(
            "staleness_filter_mode", "trajectory"
        )
        staleness_threshold = self.cfg.algorithm.get("staleness_threshold", None)
        keyed_routing = self.cfg.algorithm.get("actor_channel_keyed_routing", False)
        recv_key = (
            CommMapper.build_channel_key(self._rank, self._rank, "async_actor")
            if keyed_routing
            else None
        )
        while not self.should_stop:
            try:
                trajectory: Trajectory = (
                    input_channel.get(key=recv_key)
                    if keyed_routing
                    else input_channel.get()
                )
            except Exception as exc:  # noqa: BLE001 - never let daemon die silently
                self._recv_thread_exc = exc
                self.log_info(
                    f"recv thread aborting rank={self._rank} version={self.version} "
                    f"exc={exc!r}"
                )
                return
            self.log_info(
                f"recv trajectory rank={self._rank} version={self.version} "
                f"versions.shape={trajectory.versions.shape} "
                f"v_min={float(trajectory.versions.min())} "
                f"v_mean={float(trajectory.versions.float().mean()):.2f} "
                f"v_max={float(trajectory.versions.max())} "
                f"recv_queue={self._recv_queue.qsize() if self._recv_queue else 0}"
            )
            # chunk_mask mode never drops a trajectory here: stale chunks are masked
            # later in _compute_staleness_mask so the episode stays whole for GAE.
            if (
                staleness_filter_mode != "chunk_mask"
                and staleness_threshold is not None
                and trajectory.versions.min() < (self.version - staleness_threshold)
            ):
                continue
            self._recv_queue.put(trajectory)

    @Worker.timer("drain_received_trajectories")
    def _drain_received_trajectories(self):
        staleness_filter_mode = self.cfg.algorithm.get(
            "staleness_filter_mode", "trajectory"
        )
        staleness_threshold = self.cfg.algorithm.get("staleness_threshold", None)
        while True:
            try:
                traj: Trajectory = self._recv_queue.get_nowait()
                min_v = float(traj.versions.min().item())
                mean_v = float(traj.versions.float().mean().item())
                max_v = float(traj.versions.max().item())
                self.log_info(
                    f"drain traj rank={self._rank} version={self.version} "
                    f"versions.shape={traj.versions.shape} "
                    f"v_min={min_v} v_mean={mean_v:.2f} v_max={max_v} "
                    f"recv_queue={self._recv_queue.qsize()}"
                )
                if (
                    staleness_filter_mode != "chunk_mask"
                    and staleness_threshold is not None
                    and min_v < (self.version - staleness_threshold)
                ):
                    continue
                self.rollout_store.add((min_v, mean_v), traj)
                self.log_info(f"rollout_store size={len(self.rollout_store)}")
            except queue.Empty:
                break

    @Worker.timer("wait_for_rollout_store_ready")
    async def _wait_for_rollout_store_ready(self):
        """Wait until this rank (and, in chunk_mask mode, ALL ranks) have fresh
        rollout data; returns the taken ``list[Trajectory]`` batch.

        trajectory mode (default): per-rank ``remove_below`` + ``topn`` (legacy).
        chunk_mask mode: a two-phase ``all_reduce(MIN)`` readiness collective so
        every rank enters training together -- phase 1 confirms every rank has a
        candidate, phase 2 confirms every rank has >=1 fresh chunk; on success
        ``take_topn`` consumes the batch in lockstep, on all-stale every rank
        ``discard_topn`` together. A per-iteration ``all_reduce(MAX)`` abort
        syncs all ranks out on timeout / receive-thread crash (the collective is
        the sync point, so no rank can break out alone -> no deadlock).
        """
        while getattr(self, "_recv_queue", None) is None:
            await asyncio.sleep(1)

        staleness_filter_mode = self.cfg.algorithm.get(
            "staleness_filter_mode", "trajectory"
        )
        staleness_threshold = self.cfg.algorithm.get("staleness_threshold", None)
        on_policy_min_ratio = self.cfg.algorithm.get("on_policy_min_ratio", 0.0)
        n = self.rollout_store_size

        if staleness_filter_mode != "chunk_mask":
            while True:
                self._drain_received_trajectories()
                if staleness_threshold is not None:
                    with self.worker_timer("remove_below"):
                        self.rollout_store.remove_below(
                            self.version - staleness_threshold
                        )
                if len(self.rollout_store) >= n:
                    if on_policy_min_ratio <= 0.0:
                        break
                    metrics_data = self.rollout_store.get_metric()
                    on_policy_ratio = metrics_data.get(int(self.version), {}).get(
                        "ratio", 0.0
                    )
                    self.log_info(
                        f"rollout store metrics={metrics_data} "
                        f"on_policy_ratio={on_policy_ratio:.4f} "
                        f"on_policy_min_ratio={on_policy_min_ratio}"
                    )
                    if on_policy_ratio >= on_policy_min_ratio:
                        break
                await asyncio.sleep(1)
            batch = self.rollout_store.topn(n)
            self._staleness_readiness = {
                "staleness_received_trajectories": len(batch),
                "staleness_global_retry_rounds": 0,
                "staleness_wait_seconds": 0.0,
            }
            return batch

        # ---- chunk_mask: two-phase all-reduce readiness collective ----
        timeout = float(self.cfg.algorithm.get("rollout_store_wait_timeout_s", 600))
        status_interval = float(
            self.cfg.algorithm.get("rollout_store_status_interval_s", 30)
        )
        device = (
            torch.cuda.current_device()
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        cutoff = (
            int(self.version) - int(staleness_threshold)
            if staleness_threshold is not None
            else -(10**9)
        )
        start = time.time()
        global_retry = 0
        last_status = 0.0
        received = 0

        while True:
            self._drain_received_trajectories()
            candidates = self.rollout_store.peek_topn(n)
            received = max(received, len(candidates))

            # Per-iteration abort sync (all_reduce is the sync point): any rank
            # timed out OR its recv thread crashed -> every rank raises together.
            now = time.time()
            local_abort = 1.0 if (
                (now - start) >= timeout
                or self._recv_thread_exc is not None
            ) else 0.0
            abort = torch.tensor([local_abort], device=device)
            torch.distributed.all_reduce(abort, op=torch.distributed.ReduceOp.MAX)
            if abort.item() >= 1.0:
                reason = (
                    f"recv thread crash: {self._recv_thread_exc!r}"
                    if self._recv_thread_exc is not None
                    else f"timeout after {now - start:.1f}s"
                )
                self._staleness_readiness = {
                    "staleness_received_trajectories": received,
                    "staleness_global_retry_rounds": global_retry,
                    "staleness_wait_seconds": now - start,
                }
                raise RuntimeError(
                    f"rollout store readiness abort rank={self._rank} "
                    f"version={self.version} {reason} global_retry={global_retry}"
                )

            # Phase 1: every rank has >= n candidates?
            has = torch.tensor(
                [1.0 if len(candidates) >= n else 0.0], device=device
            )
            torch.distributed.all_reduce(has, op=torch.distributed.ReduceOp.MIN)
            if has.item() < 1.0:
                if now - last_status >= status_interval:
                    self._log_wait_status(start, global_retry, candidates, cutoff, "phase1 wait-candidate")
                    last_status = now
                await asyncio.sleep(1)
                continue

            # Phase 2: every rank has >= 1 fresh chunk?
            fresh = torch.tensor(
                [float(self._count_fresh_chunks(candidates, cutoff))],
                device=device,
            )
            torch.distributed.all_reduce(fresh, op=torch.distributed.ReduceOp.MIN)
            if fresh.item() < 1.0:
                # All ranks have candidates but some rank has zero fresh chunks:
                # discard in lockstep and wait for fresh data to refill.
                self.rollout_store.discard_topn(n)
                global_retry += 1
                if now - last_status >= status_interval:
                    self._log_wait_status(start, global_retry, candidates, cutoff, "phase2 all-stale discard")
                    last_status = now
                await asyncio.sleep(1)
                continue

            batch = self.rollout_store.take_topn(n)
            self._staleness_readiness = {
                "staleness_received_trajectories": len(batch),
                "staleness_global_retry_rounds": global_retry,
                "staleness_wait_seconds": time.time() - start,
            }
            self.log_info(
                f"readiness ready rank={self._rank} version={self.version} "
                f"took={len(batch)} global_retry={global_retry} "
                f"wait_s={time.time()-start:.1f}"
            )
            return batch

    def _count_fresh_chunks(self, candidates, cutoff: int) -> int:
        """Count fresh chunk-steps across candidate trajectories (readiness phase 2)."""
        return count_fresh_chunks(candidates, cutoff)["fresh"]

    def _log_wait_status(self, start, global_retry, candidates, cutoff, stage):
        now = time.time()
        alive = (
            self._recv_rollout_thread.is_alive()
            if getattr(self, "_recv_rollout_thread", None) is not None
            else False
        )
        stats = count_fresh_chunks(candidates, cutoff)
        self.log_info(
            f"readiness {stage} rank={self._rank} version={self.version} "
            f"wait_s={now-start:.1f} global_retry={global_retry} "
            f"recv_alive={alive} candidates={len(candidates)} "
            f"recv_queue={self._recv_queue.qsize() if self._recv_queue else 0} "
            f"store={len(self.rollout_store)} "
            f"trainable_chunks={stats['trainable']} fresh_chunks={stats['fresh']} "
            f"stale_chunks={stats['stale']} "
            f"v_min={stats['version_min']} v_mean={stats['version_mean']:.3f} "
            f"v_max={stats['version_max']}"
        )

    def _compute_staleness_mask(self) -> dict:
        """AND the low-level loss_mask with a per-chunk-step freshness mask.

        Stale chunks (version < actor_version - staleness_threshold) are masked
        out of loss + reward aggregation, but the episode stays whole for GAE.
        Stores effective low-level + chunk masks on rollout_batch; see
        ``rlinf.data.staleness_mask.compute_staleness_mask`` for details.
        """
        return compute_staleness_mask(
            self.rollout_batch,
            self.version,
            self.cfg.algorithm.get("staleness_threshold", None),
        )

    @Worker.timer("construct_rollout_batch")
    async def construct_rollout_batch(self, max_trajectories: int | None = None):
        # from _recv_queue to rollout_batch
        rollout_batch = await self._wait_for_rollout_store_ready()
        if self.cfg.algorithm.get("staleness_filter_mode", "trajectory") != "chunk_mask":
            # trajectory mode: the wait is per-rank, so the barrier is the
            # cross-rank sync point before FSDP training. chunk_mask mode's
            # two-phase collective already synchronized all ranks.
            torch.distributed.barrier()

        version_metrics = self.rollout_store.get_metric()
        self.log_info(f"rollout store version metrics={version_metrics}")

        staleness_metrics: dict = {}
        for version_val, stats in version_metrics.items():
            if version_val == "discarded_unused":
                staleness_metrics["discarded_unused_trajs"] = stats
                continue
            diff = int(self.version) - int(version_val)
            staleness_metrics[f"data_staleness_{diff}/ratio"] = stats["ratio"]

        self.rollout_batch = convert_trajectories_to_batch(rollout_batch)
        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)
        if self.cfg.algorithm.get("staleness_filter_mode", "trajectory") == "chunk_mask":
            staleness_metrics.update(self._compute_staleness_mask())
        if getattr(self, "_staleness_readiness", None):
            staleness_metrics.update(self._staleness_readiness)
        self.log_info(f"staleness metrics={staleness_metrics}")
        return staleness_metrics

    @torch.inference_mode()
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        proximal_values = self.rollout_batch.get("proximal_values", None)
        prev_values = self.rollout_batch.get("prev_values", None)
        reward_metrics = compute_embodied_reward_metrics(
            self.rollout_batch["rewards"],
            self.rollout_batch.get("loss_mask", None),
            reward_type=self.cfg.algorithm.reward_type,
            chunk_reward_aggregation=self.cfg.algorithm.get(
                "chunk_reward_aggregation", "sum"
            ),
            gamma=float(self.cfg.algorithm.get("gamma", 1.0)),
        )

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": proximal_values if proximal_values is not None else prev_values,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "chunk_reward_aggregation": self.cfg.algorithm.get(
                "chunk_reward_aggregation", "sum"
            ),
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
            # Wire normalize_returns from cfg to the GAE (advantages.py:83). Default
            # False; set True in config so returns -> O(1), which keeps value_loss
            # O(1) and prevents the value_head grad from dominating the merged
            # clip_grad_norm (which starves the actor's policy grad).
            "normalize_returns": self.cfg.algorithm.get("normalize_returns", False),
        }

        adv_and_ret = calculate_adv_and_returns(**kwargs)
        self.rollout_batch.update(adv_and_ret)

        if self.cfg.algorithm.get("staleness_filter_mode", "trajectory") == "chunk_mask":
            # GAE consumed the effective low-level mask for reward aggregation;
            # expose the chunk-level effective mask to policy/value/entropy loss
            # (preprocess_loss_inputs flattens the mask, so a low-level mask would
            # mismatch the logprob target_shape).
            self.rollout_batch["loss_mask"] = self.rollout_batch[
                "staleness_chunk_loss_mask"
            ]
            self.rollout_batch["loss_mask_sum"] = self.rollout_batch[
                "staleness_chunk_loss_mask_sum"
            ]
        else:
            if kwargs["loss_mask"] is not None:
                self.rollout_batch["loss_mask"] = kwargs["loss_mask"]
            if kwargs["loss_mask_sum"] is not None:
                self.rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        rollout_metrics.update(reward_metrics)
        return rollout_metrics

    @torch.inference_mode()
    def compute_proximal_logprobs(self) -> None:
        assert not self.is_weight_offloaded, (
            "Weight offloading is not supported when recomputing proximal logprobs."
        )

        t_dim = self.rollout_batch["prev_logprobs"].shape[0]
        b_dim = self.rollout_batch["prev_logprobs"].shape[1]

        flat = flatten_rollout_batch_for_train(self.rollout_batch, shuffle_id=None)
        total = flat["prev_logprobs"].shape[0]
        micro_batch_size = self.cfg.actor.micro_batch_size
        num_splits = (total + micro_batch_size - 1) // micro_batch_size

        iterator = split_dict_to_chunk(flat, num_splits)

        self.model.eval()
        proximal_logprobs_list = []

        for micro_batch in iterator:
            micro_batch = put_tensor_device(micro_batch, self.device)
            forward_inputs = micro_batch.get("forward_inputs", None)
            if forward_inputs is None:
                raise ValueError(
                    "Missing forward_inputs in compute_proximal_logprobs. "
                    "This usually means batch splitting dropped nested dict fields."
                )

            model_kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                model_kwargs["temperature"] = (
                    self.cfg.rollout.sampling_params.temperature_train
                )
                model_kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
            elif SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.GR00T,
                SupportedModel.ABOT_M0,
            ]:
                model_kwargs["prev_logprobs"] = micro_batch["prev_logprobs"]

            out = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=False,
                use_cache=False,
                **model_kwargs,
            )
            proximal_logprobs_list.append(out["logprobs"].cpu())

        proximal_logprobs = torch.cat(proximal_logprobs_list, dim=0).view(
            t_dim,
            b_dim,
            *self.rollout_batch["prev_logprobs"].shape[2:],
        )
        self.rollout_batch["proximal_logprobs"] = proximal_logprobs

    def run_training(self) -> dict[str, Any]:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        t_dim = int(self.rollout_batch["prev_logprobs"].shape[0])
        b_dim = int(self.rollout_batch["prev_logprobs"].shape[1])
        total_samples = t_dim * b_dim

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.cfg.actor.seed) + int(self._rank))
        shuffle_id = torch.randperm(total_samples, generator=generator)

        with torch.no_grad():
            self.rollout_batch = flatten_rollout_batch_for_train(
                self.rollout_batch, shuffle_id
            )

        if self.cfg.algorithm.normalize_advantages:
            self.rollout_batch["advantages"] = masked_normalization(
                self.rollout_batch["advantages"],
                self.rollout_batch.get("loss_mask", None),
            )

        self.model.train()

        world_size = int(self._world_size)
        global_batch_size = int(self.cfg.actor.global_batch_size)
        micro_batch_size = int(self.cfg.actor.micro_batch_size)

        assert global_batch_size % (micro_batch_size * world_size) == 0, (
            f"global_batch_size {global_batch_size} must be divisible by "
            f"micro_batch_size {micro_batch_size} * world_size {world_size}"
        )

        per_rank_batch_size = global_batch_size // world_size
        micro_per_rank = per_rank_batch_size // micro_batch_size
        self.gradient_accumulation = micro_per_rank

        flattened_rollout_size = int(self.rollout_batch["prev_logprobs"].shape[0])
        assert flattened_rollout_size % per_rank_batch_size == 0, (
            f"Flattened rollout size {flattened_rollout_size} must be divisible by "
            f"per-rank batch size {per_rank_batch_size}"
        )
        num_global_batches = flattened_rollout_size // per_rank_batch_size

        metrics: dict[str, list] = {}
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))

        for _ in range(update_epoch):
            global_batch_iter = split_dict_to_chunk(
                self.rollout_batch,
                num_global_batches,
            )

            for train_global_batch in global_batch_iter:
                train_global_batch_size = int(
                    train_global_batch["prev_logprobs"].shape[0]
                )
                assert train_global_batch_size == per_rank_batch_size, (
                    f"Expected per-rank global batch size {per_rank_batch_size}, "
                    f"got {train_global_batch_size}"
                )
                assert train_global_batch_size % micro_batch_size == 0

                micro_batch_iter = split_dict_to_chunk(
                    train_global_batch,
                    micro_per_rank,
                )

                self.optimizer.zero_grad()

                for mb_idx, data in enumerate(micro_batch_iter):
                    data = put_tensor_device(
                        data,
                        f"cuda:{int(os.environ['LOCAL_RANK'])}",
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(mb_idx + 1) == self.gradient_accumulation,
                    )

                    advantages = data["advantages"]
                    old_logprobs = data["prev_logprobs"]
                    returns = data.get("returns", None)
                    prev_values = data.get("prev_values", None)
                    loss_mask = data.get("loss_mask", None)
                    loss_mask_sum = data.get("loss_mask_sum", None)

                    versions = data.get("versions", None)
                    proximal_logprobs = data.get("proximal_logprobs", None)
                    proximal_values = data.get("proximal_values", None)
                    current_version = int(self.version) + 1

                    forward_inputs = data.get("forward_inputs", None)
                    if forward_inputs is None:
                        raise ValueError(
                            "Missing forward_inputs in run_training. "
                            "This usually means batch splitting dropped nested dict fields."
                        )

                    model_kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        model_kwargs["temperature"] = (
                            self.cfg.rollout.sampling_params.temperature_train
                        )
                        model_kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
                    elif SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T,
                        SupportedModel.ABOT_M0,
                    ]:
                        model_kwargs["prev_logprobs"] = old_logprobs

                    compute_values = self.cfg.algorithm.adv_type == "gae"

                    with self.amp_context:
                        out = self.model(
                            forward_inputs=forward_inputs,
                            compute_logprobs=True,
                            compute_entropy=(self.cfg.algorithm.entropy_bonus > 0),
                            compute_values=compute_values,
                            use_cache=False,
                            **model_kwargs,
                        )

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T,
                        SupportedModel.ABOT_M0,
                    ]:
                        old_logprobs = out["prev_logprobs"]

                    loss_kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": out["logprobs"],
                        "values": out.get("values", None),
                        "old_logprobs": old_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": proximal_values
                        if proximal_values is not None
                        else prev_values,
                        "proximal_logprobs": proximal_logprobs,
                        "versions": versions,
                        "current_version": current_version,
                        "behave_weight_threshold": self.cfg.algorithm.get(
                            "behave_weight_threshold", None
                        ),
                        "clip_ratio_c": self.cfg.algorithm.get("clip_ratio_c", 3.0),
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }

                    loss, metrics_data = policy_loss(**loss_kwargs)

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not loss_kwargs["critic_warmup"]
                    ):
                        entropy = out["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=out["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                    loss = loss / self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/entropy_loss"] = float(
                        entropy_loss.detach().item()
                    )
                    metrics_data["actor/total_loss"] = float(loss.detach().item())
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                extra_metrics = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    extra_metrics["critic/lr"] = lr_list[1]
                append_to_dict(metrics, extra_metrics)

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()

        mean_metric_dict = {k: float(np.mean(v)) for k, v in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict,
            op=torch.distributed.ReduceOp.AVG,
        )
        return mean_metric_dict
