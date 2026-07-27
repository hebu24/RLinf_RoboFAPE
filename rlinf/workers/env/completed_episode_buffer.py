"""Episode-aware buffering for terminal history rewards."""

from __future__ import annotations

from collections import deque
from dataclasses import fields
from typing import Any

import torch

from rlinf.data.embodied_io_struct import Trajectory, get_model_weights_id
from rlinf.models.embodiment.reward.robometer_reward_model import (
    RobometerEpisodeReward,
)

_STEP_ZERO_FIELDS = {"rewards", "loss_mask", "intervene_flags"}
_BOUNDARY_ZERO_FIELDS = {"prev_values"}
_BOUNDARY_FALSE_FIELDS = {"terminations", "truncations"}
_BOUNDARY_TRUE_FIELDS = {"dones"}


def _trajectory_steps(trajectory: Trajectory) -> int:
    for name in ("actions", "prev_logprobs", "versions", "loss_mask"):
        value = getattr(trajectory, name, None)
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
    if isinstance(trajectory.rewards, torch.Tensor):
        return int(trajectory.rewards.shape[0])
    raise ValueError("Trajectory has no step-aligned tensor fields.")


def normalize_rollout_trajectory(trajectory: Trajectory) -> Trajectory:
    """Convert EnvWorker rollout tensors to canonical episode-buffer alignment.

    ``EmbodiedRolloutResult`` records rewards for the initial bootstrap state and
    every post-action state, so rewards contain ``T+1`` entries while actions and
    policy fields contain ``T``. Episode reconstruction aligns action ``t`` with
    reward/done boundary ``t+1`` and therefore drops only the initial reward.
    """
    steps = _trajectory_steps(trajectory)
    result = Trajectory(max_episode_length=trajectory.max_episode_length)
    for item in fields(Trajectory):
        name = item.name
        value = getattr(trajectory, name)
        if value is None:
            continue
        if isinstance(value, (int, str)):
            setattr(result, name, value)
        elif isinstance(value, dict):
            normalized = {}
            for key, tensor in value.items():
                if tensor.shape[0] != steps:
                    raise ValueError(
                        f"Rollout dict field {name}.{key} must have T entries: "
                        f"{tensor.shape[0]} vs T={steps}."
                    )
                normalized[key] = tensor.contiguous()
            setattr(result, name, normalized)
        elif isinstance(value, torch.Tensor):
            length = int(value.shape[0])
            if name in {"rewards", "loss_mask"} and length == steps + 1:
                value = value[1:]
            elif length not in {steps, steps + 1}:
                raise ValueError(
                    f"Rollout field {name} must have T or T+1 entries: "
                    f"{length} vs T={steps}."
                )
            setattr(result, name, value.contiguous())
        else:
            raise ValueError(
                f"Unsupported trajectory field type for {name}: {type(value)}"
            )
    if result.rewards is None or result.rewards.shape[0] != steps:
        raise ValueError(
            "Normalized rollout must contain one reward tensor per action step."
        )
    return result


def _slice_dict(value: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    return {key: tensor[start:end].contiguous() for key, tensor in value.items()}


def _slice_trajectory(trajectory: Trajectory, start: int, end: int) -> Trajectory:
    total_steps = _trajectory_steps(trajectory)
    result = Trajectory(max_episode_length=trajectory.max_episode_length)
    for item in fields(Trajectory):
        name = item.name
        value = getattr(trajectory, name)
        if value is None:
            continue
        if isinstance(value, (int, str)):
            setattr(result, name, value)
        elif isinstance(value, dict):
            setattr(result, name, _slice_dict(value, start, end))
        elif isinstance(value, torch.Tensor):
            if value.shape[0] == total_steps:
                setattr(result, name, value[start:end].contiguous())
            elif value.shape[0] == total_steps + 1:
                setattr(result, name, value[start : end + 1].contiguous())
            else:
                raise ValueError(
                    f"Unsupported trajectory field length for {name}: "
                    f"{value.shape[0]} vs steps={total_steps}."
                )
        else:
            raise ValueError(
                f"Unsupported trajectory field type for {name}: {type(value)}"
            )
    return result


def _select_env(trajectory: Trajectory, env_id: int) -> Trajectory:
    result = Trajectory(max_episode_length=trajectory.max_episode_length)
    for item in fields(Trajectory):
        name = item.name
        value = getattr(trajectory, name)
        if value is None:
            continue
        if isinstance(value, (int, str)):
            setattr(result, name, value)
        elif isinstance(value, dict):
            setattr(
                result,
                name,
                {
                    key: tensor[:, env_id : env_id + 1].contiguous()
                    for key, tensor in value.items()
                },
            )
        elif isinstance(value, torch.Tensor):
            setattr(result, name, value[:, env_id : env_id + 1].contiguous())
        else:
            raise ValueError(
                f"Unsupported trajectory field type for {name}: {type(value)}"
            )
    return result


def _concat_trajectories(left: Trajectory | None, right: Trajectory) -> Trajectory:
    if left is None:
        return right
    left_steps = _trajectory_steps(left)
    right_steps = _trajectory_steps(right)
    result = Trajectory(
        max_episode_length=max(left.max_episode_length, right.max_episode_length)
    )
    for item in fields(Trajectory):
        name = item.name
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is None:
            setattr(result, name, right_value)
            continue
        if right_value is None:
            setattr(result, name, left_value)
            continue
        if isinstance(left_value, (int, str)):
            setattr(result, name, right_value)
        elif isinstance(left_value, dict):
            setattr(
                result,
                name,
                {
                    key: torch.cat(
                        [left_value[key], right_value[key]], dim=0
                    ).contiguous()
                    for key in left_value
                },
            )
        elif isinstance(left_value, torch.Tensor):
            if (
                left_value.shape[0] == left_steps
                and right_value.shape[0] == right_steps
            ):
                merged = torch.cat([left_value, right_value], dim=0)
            elif (
                left_value.shape[0] == left_steps + 1
                and right_value.shape[0] == right_steps + 1
            ):
                merged = torch.cat([left_value, right_value[1:]], dim=0)
            else:
                raise ValueError(
                    f"Cannot concatenate misaligned trajectory field {name}."
                )
            setattr(result, name, merged.contiguous())
        else:
            raise ValueError(
                f"Unsupported trajectory field type for {name}: {type(left_value)}"
            )
    return result


def _first_done_step(trajectory: Trajectory) -> int | None:
    steps = _trajectory_steps(trajectory)
    if trajectory.dones is None or trajectory.dones.shape[0] != steps + 1:
        raise ValueError("Episode buffering requires dones with T+1 entries.")
    done = trajectory.dones[1:].reshape(steps, -1).any(dim=1)
    indices = done.nonzero(as_tuple=False).reshape(-1)
    return None if indices.numel() == 0 else int(indices[0].item())


def _apply_reward(trajectory: Trajectory, reward: RobometerEpisodeReward) -> Trajectory:
    steps = _trajectory_steps(trajectory)
    reward_chunks = int(reward.chunk_reward.shape[0])
    if reward_chunks > steps:
        raise ValueError(
            f"Robometer reward has more chunks than the completed episode: {reward_chunks} > {steps}."
        )
    if trajectory.rewards is None:
        raise ValueError("Completed episode trajectory has no rewards tensor.")
    trajectory.rewards.zero_()
    if trajectory.loss_mask is None:
        trajectory.loss_mask = torch.zeros_like(trajectory.rewards, dtype=torch.bool)
    else:
        trajectory.loss_mask.zero_()
    start = steps - reward_chunks
    reward_tensor = torch.as_tensor(
        reward.chunk_reward,
        dtype=trajectory.rewards.dtype,
        device=trajectory.rewards.device,
    ).unsqueeze(1)
    mask_tensor = torch.as_tensor(
        reward.chunk_loss_mask,
        dtype=torch.bool,
        device=trajectory.loss_mask.device,
    ).unsqueeze(1)
    trajectory.rewards[start:] = reward_tensor
    trajectory.loss_mask[start:] = mask_tensor
    return trajectory


def _pad_tensor(
    name: str, value: torch.Tensor, pad: int, boundary: bool
) -> torch.Tensor:
    if pad <= 0:
        return value
    shape = (pad,) + tuple(value.shape[1:])
    if boundary and name in _BOUNDARY_TRUE_FIELDS:
        extension = torch.ones(shape, dtype=value.dtype, device=value.device)
    elif boundary and (name in _BOUNDARY_ZERO_FIELDS or name in _BOUNDARY_FALSE_FIELDS):
        extension = torch.zeros(shape, dtype=value.dtype, device=value.device)
    elif name in _STEP_ZERO_FIELDS:
        extension = torch.zeros(shape, dtype=value.dtype, device=value.device)
    else:
        extension = value[-1:].expand(shape).clone()
    return torch.cat([value, extension], dim=0).contiguous()


def pad_completed_episode(trajectory: Trajectory, max_chunks: int) -> Trajectory:
    """Right-pad one completed episode while masking all padded chunks."""
    steps = _trajectory_steps(trajectory)
    if steps > max_chunks:
        raise ValueError(f"Episode has {steps} chunks, exceeding max {max_chunks}.")
    pad = max_chunks - steps
    if pad == 0:
        return trajectory
    result = Trajectory(max_episode_length=trajectory.max_episode_length)
    for item in fields(Trajectory):
        name = item.name
        value = getattr(trajectory, name)
        if value is None:
            continue
        if isinstance(value, (int, str)):
            setattr(result, name, value)
        elif isinstance(value, dict):
            setattr(
                result,
                name,
                {
                    key: _pad_tensor(key, tensor, pad, boundary=False)
                    for key, tensor in value.items()
                },
            )
        elif isinstance(value, torch.Tensor):
            if value.shape[0] == steps:
                setattr(result, name, _pad_tensor(name, value, pad, boundary=False))
            elif value.shape[0] == steps + 1:
                setattr(result, name, _pad_tensor(name, value, pad, boundary=True))
            else:
                raise ValueError(f"Cannot pad misaligned trajectory field {name}.")
        else:
            raise ValueError(
                f"Unsupported trajectory field type for {name}: {type(value)}"
            )
    return result


def combine_completed_episodes(episodes: list[Trajectory]) -> Trajectory:
    """Combine equal-length single-env completed episodes along the batch axis."""
    if not episodes:
        raise ValueError("No completed episodes to combine.")
    result = Trajectory(
        max_episode_length=max(ep.max_episode_length for ep in episodes)
    )
    for item in fields(Trajectory):
        name = item.name
        values = [getattr(ep, name) for ep in episodes]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError(f"Completed episodes disagree on field {name}.")
        first = values[0]
        if isinstance(first, int):
            setattr(result, name, max(values))
        elif isinstance(first, str):
            setattr(result, name, first)
        elif isinstance(first, dict):
            setattr(
                result,
                name,
                {
                    key: torch.cat([value[key] for value in values], dim=1).contiguous()
                    for key in first
                },
            )
        elif isinstance(first, torch.Tensor):
            setattr(result, name, torch.cat(values, dim=1).contiguous())
        else:
            raise ValueError(
                f"Unsupported trajectory field type for {name}: {type(first)}"
            )
    if result.versions is not None:
        result.model_weights_id = get_model_weights_id(result.versions)
    return result


class CompletedEpisodeBuffer:
    """Carry incomplete env episodes across rollout boundaries."""

    def __init__(self, num_envs: int, max_chunks: int, max_episode_steps: int = 0):
        self.num_envs = num_envs
        self.max_chunks = max_chunks
        self.max_episode_steps = max_episode_steps
        self.pending: list[Trajectory | None] = [None] * num_envs
        self.rewards: list[deque[RobometerEpisodeReward]] = [
            deque() for _ in range(num_envs)
        ]
        # Per-env ready slot: only the LATEST completed episode survives a
        # collection window; older ones are superseded (counted, not sent) so a
        # single env can never flood the actor channel and every env contributes
        # exactly one episode per training batch.
        self.ready: list[Trajectory | None] = [None] * num_envs
        self.ready_valid_chunks: list[int] = [0] * num_envs
        self.ready_wait_rounds: list[int] = [0] * num_envs
        self.superseded_completed_episodes = 0
        self.selected_episode_version_min = 0
        self.selected_episode_version_max = 0
        self.selected_episode_wait_rollouts = 0.0
        self.cross_rollout_episodes = 0
        self.completed_episodes = 0
        self.ingest_round = 0
        self.pending_since_round: list[int | None] = [None] * num_envs
        self.completed_wait_rounds: deque[int] = deque(maxlen=1024)
        self.last_batch_valid_chunks = 0
        self.last_batch_padding_chunks = 0

    def add_rewards(self, assignments: dict[int, RobometerEpisodeReward]) -> None:
        for env_id, reward in assignments.items():
            self.rewards[env_id].append(reward)

    def ingest(self, trajectory: Trajectory) -> None:
        self.ingest_round += 1
        trajectory = normalize_rollout_trajectory(trajectory)
        batch_size = int(trajectory.rewards.shape[1])
        if batch_size != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} envs, got {batch_size}.")
        for env_id in range(self.num_envs):
            had_pending = self.pending[env_id] is not None
            if not had_pending:
                self.pending_since_round[env_id] = self.ingest_round
            combined = _concat_trajectories(
                self.pending[env_id], _select_env(trajectory, env_id)
            )
            while combined is not None:
                done_step = _first_done_step(combined)
                if done_step is None:
                    self.pending[env_id] = combined
                    break
                if not self.rewards[env_id]:
                    raise ValueError(
                        f"Completed env {env_id} episode has no Robometer reward assignment."
                    )
                end = done_step + 1
                episode = _slice_trajectory(combined, 0, end)
                episode = _apply_reward(episode, self.rewards[env_id].popleft())
                if self.ready[env_id] is not None:
                    # Same env completed again within this window: keep only the
                    # latest; supersede the older episode (counted, not trained).
                    self.superseded_completed_episodes += 1
                self.ready[env_id] = pad_completed_episode(episode, self.max_chunks)
                self.ready_valid_chunks[env_id] = _trajectory_steps(episode)
                started = self.pending_since_round[env_id]
                wait = 0 if started is None else self.ingest_round - started
                self.ready_wait_rounds[env_id] = wait
                self.completed_wait_rounds.append(wait)
                self.completed_episodes += 1
                if had_pending:
                    self.cross_rollout_episodes += 1
                    had_pending = False
                if end == _trajectory_steps(combined):
                    combined = None
                    self.pending[env_id] = None
                    self.pending_since_round[env_id] = None
                else:
                    combined = _slice_trajectory(
                        combined, end, _trajectory_steps(combined)
                    )
                    self.pending[env_id] = combined
                    self.pending_since_round[env_id] = self.ingest_round

    def pop_batch(self, batch_size: int) -> Trajectory:
        if batch_size > self.num_envs:
            raise ValueError(
                f"pop_batch batch_size {batch_size} > num_envs {self.num_envs}."
            )
        episodes: list[Trajectory] = []
        valid_chunks: list[int] = []
        waits: list[int] = []
        versions: list[int] = []
        for env_id in range(batch_size):
            if self.ready[env_id] is None:
                raise ValueError(
                    f"Env {env_id} has no completed episode after a full "
                    f"collection window (max_episode_steps={self.max_episode_steps}): "
                    f"pending_len={1 if self.pending[env_id] is not None else 0}, "
                    f"reward_assignments={len(self.rewards[env_id])}, "
                    f"completed_episodes={self.completed_episodes}."
                )
            ep = self.ready[env_id]
            episodes.append(ep)
            valid_chunks.append(self.ready_valid_chunks[env_id])
            waits.append(self.ready_wait_rounds[env_id])
            ev = getattr(ep, "versions", None)
            if ev is not None:
                versions.append(int(ev.min().item()))
            self.ready[env_id] = None
            self.ready_valid_chunks[env_id] = 0
            self.ready_wait_rounds[env_id] = 0
        self.last_batch_valid_chunks = sum(valid_chunks)
        self.last_batch_padding_chunks = (
            batch_size * self.max_chunks - self.last_batch_valid_chunks
        )
        self.selected_episode_version_min = min(versions) if versions else 0
        self.selected_episode_version_max = max(versions) if versions else 0
        self.selected_episode_wait_rollouts = (
            float(sum(waits) / len(waits)) if waits else 0.0
        )
        return combine_completed_episodes(episodes)

    @property
    def ready_episodes(self) -> int:
        return sum(item is not None for item in self.ready)

    @property
    def pending_count(self) -> int:
        return sum(item is not None for item in self.pending)


def split_trajectory_by_sizes(
    trajectory: Trajectory, split_sizes: list[int]
) -> list[Trajectory]:
    """Split a batched trajectory along its environment dimension."""
    results = [Trajectory() for _ in split_sizes]
    for item in fields(Trajectory):
        name = item.name
        value = getattr(trajectory, name)
        if value is None:
            continue
        if isinstance(value, (int, str)):
            for result in results:
                setattr(result, name, value)
        elif isinstance(value, torch.Tensor):
            for result, part in zip(results, torch.split(value, split_sizes, dim=1)):
                setattr(result, name, part.contiguous())
        elif isinstance(value, dict):
            split_values = {
                key: torch.split(tensor, split_sizes, dim=1)
                for key, tensor in value.items()
            }
            for idx, result in enumerate(results):
                setattr(
                    result,
                    name,
                    {
                        key: parts[idx].contiguous()
                        for key, parts in split_values.items()
                    },
                )
        else:
            raise ValueError(
                f"Unsupported trajectory field type for {name}: {type(value)}"
            )
    return results
