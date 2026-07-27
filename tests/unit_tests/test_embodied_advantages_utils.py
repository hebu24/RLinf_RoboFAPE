import torch

from rlinf.algorithms.utils import (
    aggregate_embodied_chunk_rewards,
    preprocess_embodied_advantages_inputs,
)
from rlinf.utils.metric_utils import embodied_reward_metric_values


def test_chunk_level_discounted_sum_aggregates_low_level_rewards_once():
    rewards = torch.tensor(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
        ]
    )
    dones = torch.zeros((3, 1, 3), dtype=torch.bool)
    values = torch.zeros((3, 1, 1), dtype=torch.float32)
    loss_mask = torch.tensor(
        [
            [[True, True, False]],
            [[True, False, False]],
        ]
    )

    processed = preprocess_embodied_advantages_inputs(
        rewards=rewards,
        dones=dones,
        values=values,
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask.clone(),
        reward_type="chunk_level",
        adv_type="gae",
        gamma=0.5,
        chunk_reward_aggregation="discounted_sum",
        group_size=1,
    )

    assert processed["num_chunk"] == 2
    assert processed["chunk_size"] == 1
    torch.testing.assert_close(
        processed["rewards"],
        torch.tensor([[2.0], [4.0]]),
    )
    torch.testing.assert_close(
        processed["loss_mask"],
        torch.tensor([[True], [True]]),
    )


def test_reward_metric_values_match_gae_aggregation_and_exclude_padding():
    rewards = torch.tensor(
        [
            [[-0.8, -0.6], [-0.9, 0.4]],
            [[-0.2, 0.0], [0.5, 0.6]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    loss_mask = torch.tensor(
        [
            [[True, True], [True, True]],
            [[True, False], [True, True]],
            [[False, False], [False, False]],
        ]
    )
    values = embodied_reward_metric_values(
        rewards,
        loss_mask,
        reward_type="chunk_level",
        chunk_reward_aggregation="discounted_sum",
        gamma=0.5,
    )

    torch.testing.assert_close(
        values["reward_low_level"],
        torch.tensor([-0.8, -0.6, -0.9, 0.4, -0.2, 0.5, 0.6]),
    )
    torch.testing.assert_close(
        values["reward_chunk"], torch.tensor([-1.1, -0.7, -0.2, 0.8])
    )
    torch.testing.assert_close(values["reward_episode_sum"], torch.tensor([-1.3, 0.1]))

    aggregated = aggregate_embodied_chunk_rewards(
        rewards,
        loss_mask,
        gamma=0.5,
        aggregation="discounted_sum",
    )
    dones = torch.zeros((4, 2, 2), dtype=torch.bool)
    prev_values = torch.zeros((4, 2, 1), dtype=torch.float32)
    processed = preprocess_embodied_advantages_inputs(
        rewards=rewards,
        dones=dones,
        values=prev_values,
        loss_mask=loss_mask,
        reward_type="chunk_level",
        adv_type="gae",
        gamma=0.5,
        chunk_reward_aggregation="discounted_sum",
        group_size=1,
    )
    torch.testing.assert_close(processed["rewards"], aggregated.squeeze(-1))


def test_failed_episode_reward_metrics_have_no_positive_values():
    progress = torch.tensor([[[0.2, 0.8]], [[1.0, 0.0]]])
    rewards = progress - 1.0
    loss_mask = torch.tensor([[[True, True]], [[True, False]]])

    values = embodied_reward_metric_values(
        rewards,
        loss_mask,
        reward_type="chunk_level",
        chunk_reward_aggregation="discounted_sum",
        gamma=0.99,
    )

    assert values["reward_low_level"].max().item() <= 0
    assert values["reward_chunk"].max().item() <= 0
    assert (values["reward_low_level"] > 0).float().mean().item() == 0
    assert (values["reward_chunk"] > 0).float().mean().item() == 0
