import numpy as np
import pytest
import torch

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.utils import preprocess_embodied_advantages_inputs
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    Trajectory,
)
from rlinf.models.embodiment.reward.robometer_reward_model import RobometerEpisodeReward
from rlinf.workers.env.completed_episode_buffer import CompletedEpisodeBuffer


def _segment(start: int, steps: int, done_at_end: bool) -> Trajectory:
    rewards = torch.zeros((steps, 1, 2), dtype=torch.float32)
    dones = torch.zeros((steps + 1, 1, 2), dtype=torch.bool)
    if done_at_end:
        dones[-1, 0, -1] = True
    values = torch.zeros((steps + 1, 1, 1), dtype=torch.float32)
    return Trajectory(
        max_episode_length=8,
        actions=torch.arange(start, start + steps, dtype=torch.float32).view(
            steps, 1, 1
        ),
        rewards=rewards,
        dones=dones,
        terminations=dones.clone(),
        truncations=torch.zeros_like(dones),
        prev_logprobs=torch.zeros((steps, 1, 1)),
        prev_values=values,
        versions=torch.zeros((steps, 1, 1)),
        loss_mask=torch.zeros_like(rewards, dtype=torch.bool),
        forward_inputs={"action": torch.zeros((steps, 1, 1))},
    )


def _reward(values=None) -> RobometerEpisodeReward:
    chunk_reward = np.asarray(
        values
        if values is not None
        else [[-0.9, -0.8], [-0.7, -0.6], [-0.5, -0.4], [-0.3, 0.4]],
        dtype=np.float32,
    )
    low_level = chunk_reward.reshape(-1)
    return RobometerEpisodeReward(
        downsample_indices=list(range(low_level.size)),
        per_step_progress=np.linspace(0.1, 0.8, low_level.size, dtype=np.float32),
        per_step_reward=low_level,
        per_step_loss_mask=np.ones(low_level.size, dtype=bool),
        chunk_reward=chunk_reward,
        chunk_loss_mask=np.ones_like(chunk_reward, dtype=bool),
    )


def test_completed_episode_buffer_preserves_cross_rollout_prefix_and_pads():
    buffer = CompletedEpisodeBuffer(num_envs=1, max_chunks=4)
    buffer.ingest(_segment(start=0, steps=2, done_at_end=False))
    assert buffer.pending_count == 1

    buffer.add_rewards({0: _reward()})
    buffer.ingest(_segment(start=2, steps=2, done_at_end=True))
    batch = buffer.pop_batch(1)

    torch.testing.assert_close(
        batch.actions[:4, 0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0])
    )
    torch.testing.assert_close(
        batch.rewards[:, 0],
        torch.tensor([[-0.9, -0.8], [-0.7, -0.6], [-0.5, -0.4], [-0.3, 0.4]]),
    )
    assert batch.loss_mask[:, 0].all()
    assert buffer.cross_rollout_episodes == 1


def test_real_rollout_result_drops_initial_bootstrap_reward_and_pads_boundaries():
    rollout = EmbodiedRolloutResult(max_episode_length=8)
    for step in range(2):
        done = torch.tensor([[False, False]])
        rollout.append_step_result(
            ChunkStepResult(
                actions=torch.tensor([[float(step)]]),
                rewards=torch.full((1, 2), float(10 + step)),
                dones=done,
                terminations=done.clone(),
                truncations=torch.zeros_like(done),
                prev_logprobs=torch.zeros((1, 1)),
                prev_values=torch.tensor([[float(step)]]),
                versions=torch.zeros((1, 1)),
                forward_inputs={"action": torch.tensor([[float(step)]])},
            )
        )
    rollout.append_step_result(
        ChunkStepResult(
            rewards=torch.full((1, 2), 12.0),
            dones=torch.tensor([[False, True]]),
            terminations=torch.tensor([[False, True]]),
            truncations=torch.tensor([[False, False]]),
            prev_values=torch.tensor([[2.0]]),
        )
    )

    buffer = CompletedEpisodeBuffer(num_envs=1, max_chunks=4)
    buffer.add_rewards({0: _reward([[0.1, 0.2], [0.3, 0.4]])})
    buffer.ingest(rollout.to_trajectory())
    batch = buffer.pop_batch(1)

    assert batch.actions.shape == (4, 1, 1)
    torch.testing.assert_close(batch.actions[:2, 0, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(
        batch.rewards[:, 0],
        torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.0, 0.0], [0.0, 0.0]]),
    )
    assert batch.loss_mask[:2, 0].all()
    assert not batch.loss_mask[2:, 0].any()
    assert batch.dones.shape == (5, 1, 2)
    assert batch.dones[2, 0].any()
    assert batch.dones[3:, 0].all()
    torch.testing.assert_close(batch.prev_values[3:, 0, 0], torch.zeros(2))

    processed = preprocess_embodied_advantages_inputs(
        rewards=batch.rewards,
        dones=batch.dones,
        values=batch.prev_values,
        loss_mask=batch.loss_mask,
        reward_type="chunk_level",
        adv_type="gae",
        gamma=0.99,
        chunk_reward_aggregation="discounted_sum",
        group_size=1,
    )
    _, returns = compute_gae_advantages_and_returns(
        rewards=processed["rewards"],
        dones=processed["dones"],
        values=processed["values"],
        loss_mask=processed["loss_mask"],
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=False,
    )
    torch.testing.assert_close(returns[2:], torch.zeros_like(returns[2:]))


def test_multiple_completed_episodes_from_one_env_keep_only_latest_and_supersede():
    """One env completes two episodes within a single collection window.

    Per the bounded completed-episode queue (plan Section 5), only the LATEST
    completed episode survives the window; the older one is superseded (counted
    in ``superseded_completed_episodes``) and never reaches the actor channel,
    so a single env cannot flood the actor. A second ``pop_batch`` must raise
    because the per-env ready slot is empty.
    """
    trajectory = _segment(start=0, steps=4, done_at_end=False)
    trajectory.dones[2, 0, -1] = True
    trajectory.terminations[2, 0, -1] = True
    trajectory.dones[4, 0, -1] = True
    trajectory.terminations[4, 0, -1] = True

    buffer = CompletedEpisodeBuffer(num_envs=1, max_chunks=2, max_episode_steps=600)
    buffer.add_rewards({0: _reward([[0.1, 0.2], [0.3, 0.4]])})
    buffer.add_rewards({0: _reward([[0.5, 0.6], [0.7, 0.8]])})
    buffer.ingest(trajectory)

    # The older episode was superseded (counted), not queued alongside the latest.
    assert buffer.superseded_completed_episodes == 1
    assert buffer.completed_episodes == 2

    # pop_batch returns the LATEST completed episode only.
    latest = buffer.pop_batch(1)
    torch.testing.assert_close(latest.actions[:, 0, 0], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(
        latest.rewards[:, 0], torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    )
    # Supersede count is unchanged by popping.
    assert buffer.superseded_completed_episodes == 1

    # The per-env ready slot is now empty: a second pop must raise (no flood).
    with pytest.raises(ValueError, match="no completed episode"):
        buffer.pop_batch(1)
