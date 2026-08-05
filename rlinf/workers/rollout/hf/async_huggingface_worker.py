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

import torch
from omegaconf.omegaconf import DictConfig

from rlinf.data.embodied_io_struct import (
    RolloutResult,
)
from rlinf.scheduler import Channel, Worker
from rlinf.scheduler.worker.routing import CommMapper
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def rollout_episode_capacity(
    *,
    staleness_threshold: int,
    version: int,
    rollout_store_size_per_rank: int,
    total_num_train_envs: int,
    rollout_epoch: int,
) -> int:
    """Return the episode production limit for an async rollout worker."""
    if rollout_store_size_per_rank <= 0:
        raise ValueError("rollout_store_size_per_rank must be positive")
    return (
        (staleness_threshold + version + 1)
        * rollout_store_size_per_rank
        * total_num_train_envs
        * rollout_epoch
    )


class AsyncMultiStepRolloutWorker(MultiStepRolloutWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._generate_task: asyncio.Task = None
        self.staleness_threshold = cfg.algorithm.get("staleness_threshold", None)
        self.rollout_store_size_per_rank = int(
            cfg.algorithm.get("rollout_store_size_per_rank", 1)
        )
        self._rollout_credits = self.rollout_store_size_per_rank
        self._rollout_credit_waits = 0
        # set the decoupled rollout worker sync weight time
        self.sync_rollout_weight_time = (
            self.num_pipeline_stages * self.n_train_chunk_steps * self.rollout_epoch
        )

        assert not self.enable_offload, (
            "Offload not supported in AsyncMultiStepRolloutWorker"
        )

        self._background_weight_sync_active = self.cfg.actor.get(
            "sync_weight_no_wait", False
        )
        self._weight_sync_requested = False
        self._weight_sync_work = None
        self._weight_sync_apply_total = 0
        self._weight_sync_coalesced_total = 0
        self._weight_sync_request_total = 0
        self.log_info(
            "async rollout capacity configured: "
            f"store_size_per_rank={self.rollout_store_size_per_rank} "
            f"staleness_threshold={self.staleness_threshold} version={self.version} "
            f"finished_episodes={self.finished_episodes} "
            f"capacity={self._episode_capacity()}"
        )

    def _episode_capacity(self) -> int | None:
        if self.staleness_threshold is None:
            return None
        return rollout_episode_capacity(
            staleness_threshold=int(self.staleness_threshold),
            version=int(self.version),
            rollout_store_size_per_rank=self.rollout_store_size_per_rank,
            total_num_train_envs=int(self.total_num_train_envs),
            rollout_epoch=int(self.rollout_epoch),
        )

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
        credit_channel: Channel | None = None,
    ):
        assert self._generate_task is None, (
            "generate task is not None but generate function is called."
        )
        self._generate_task = asyncio.create_task(
            self._generate(
                input_channel, output_channel, metric_channel, credit_channel
            )
        )
        try:
            await self._generate_task
        except asyncio.CancelledError:
            pass

    async def _generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        metric_channel: Channel,
        credit_channel: Channel | None = None,
    ):
        if self.env_decoupled_mode:
            await self.decoupled_generate_one_epoch(input_channel, output_channel)
        else:
            while True:
                if self._background_weight_sync_active:
                    await self._poll_background_weight_sync()
                if credit_channel is None:
                    await self.wait_if_stale()
                else:
                    await self._acquire_rollout_credit(credit_channel)
                for _ in range(self.rollout_epoch):
                    await self.generate_one_epoch(input_channel, output_channel)
                if self.finished_episodes is not None:
                    self.finished_episodes += (
                        self.total_num_train_envs * self.rollout_epoch
                    )
                rollout_metrics = self.pop_execution_times()
                rollout_metrics = {
                    f"time/rollout/{k}": v for k, v in rollout_metrics.items()
                }
                metric_channel.put(
                    {"rank": self._rank, "time": rollout_metrics},
                    async_op=True,
                )

    async def _acquire_rollout_credit(self, credit_channel: Channel) -> None:
        """Consume one actor-issued credit before producing a rollout window.

        Credits bound the number of windows in flight. The actor returns credits
        when trajectories are consumed or explicitly discarded, so a readiness
        retry can always trigger replacement production without waiting for the
        actor version to advance.
        """
        while self._rollout_credits <= 0:
            self._rollout_credit_waits += 1
            key = CommMapper.build_channel_key(
                self._rank, self._rank, "rollout_credit"
            )
            amount = await credit_channel.get(key=key, async_op=True).async_wait()
            amount = int(amount)
            if amount <= 0:
                raise ValueError(f"Rollout credit must be positive, got {amount}")
            self._rollout_credits += amount
            self.log_info(
                f"received rollout credits rank={self._rank} amount={amount} "
                f"available={self._rollout_credits} waits={self._rollout_credit_waits}"
            )
        self._rollout_credits -= 1

    async def wait_if_stale(self) -> None:
        if self.staleness_threshold is None:
            return
        assert self.finished_episodes is not None, (
            "finished_episodes should be initialized."
        )
        while True:
            capacity = self._episode_capacity()
            assert capacity is not None
            if (
                self.finished_episodes + self.total_num_train_envs * self.rollout_epoch
                <= capacity
            ):
                break
            await asyncio.sleep(0.01)

    def stop(self):
        if self._generate_task is not None and not self._generate_task.done():
            self._generate_task.cancel()

    async def _recv_and_apply_actor_sync(self) -> int:
        await super().sync_model_from_actor()
        return self.version

    def _start_background_weight_sync_if_needed(self):
        if (
            not self._background_weight_sync_active
            or not self._weight_sync_requested
            or self._weight_sync_work is not None
        ):
            return

        self._weight_sync_requested = False
        self._weight_sync_work = asyncio.create_task(self._recv_and_apply_actor_sync())

    @Worker.timer("rollout/poll_weight_sync")
    async def _poll_background_weight_sync(self):
        self._start_background_weight_sync_if_needed()
        if self._weight_sync_work is None:
            return

        if not self._weight_sync_work.done():
            return

        await self._weight_sync_work
        self._weight_sync_work = None
        self._weight_sync_apply_total += 1

        self._start_background_weight_sync_if_needed()

    @Worker.timer("rollout/request_weight_sync")
    async def request_actor_sync_model(self):
        self._weight_sync_request_total += 1
        if self._weight_sync_requested or self._weight_sync_work is not None:
            self._weight_sync_coalesced_total += 1
        self._weight_sync_requested = True
        self._start_background_weight_sync_if_needed()

    async def decoupled_generate_one_epoch(
        self, input_channel: Channel, output_channel: Channel
    ):
        self.update_dagger_beta()
        decoupled_generate_time = 1
        while True:
            if decoupled_generate_time % self.sync_rollout_weight_time == 0:
                self.update_dagger_beta()
                if self._background_weight_sync_active:
                    await self._poll_background_weight_sync()
                await self.wait_if_stale()
            decoupled_generate_time = decoupled_generate_time + 1
            (
                env_output,
                split_sizes,
            ) = await self.recv_from_and_record_batch_routes_with_timeout(
                group_name=self.cfg.env.group_name,
                channel=input_channel,
                tag="rollout_results",
                batch_size=self.train_batch_size,
                merge_fn=self._merge_obs_batches,
                infer_batch_size_fn=self._infer_env_batch_size,
                timeout_time=0.02,
                recv_queue_size=self.rollout_queue_size,
            )
            actions, result = self.predict(env_output["obs"])
            save_flags = None
            if result.get("expert_label_flag", False):
                save_flags = torch.full(
                    (actions.shape[0], self.cfg.actor.model.num_action_chunks),
                    True,
                    dtype=torch.bool,
                    device=actions.device,
                )
            rollout_result = RolloutResult(
                actions=actions,
                prev_logprobs=result["prev_logprobs"]
                if self.collect_prev_infos
                else None,
                prev_values=result["prev_values"] if self.collect_prev_infos else None,
                bootstrap_values=self.get_bootstrap_values(
                    env_output.get("final_obs", None)
                ),
                save_flags=save_flags,
                forward_inputs=result["forward_inputs"],
                versions=torch.full_like(
                    result["prev_logprobs"],
                    float(self.version),
                    dtype=torch.float32,
                ),
            )
            self.send_to_recorded_batch_routes(
                group_name=self.cfg.env.group_name,
                channel=output_channel,
                data=rollout_result,
                tag="rollout_results",
                split_fn=self._split_rollout_result,
                split_sizes=split_sizes,
            )
