import torch

from rlinf.algorithms.utils import preprocess_embodied_advantages_inputs


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
