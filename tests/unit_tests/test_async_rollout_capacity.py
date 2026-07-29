import pytest

from rlinf.workers.rollout.hf.async_huggingface_worker import (
    rollout_episode_capacity,
)


@pytest.mark.parametrize("rollout_epoch", [1, 2])
def test_store_size_one_preserves_legacy_capacity(rollout_epoch):
    for version in range(4):
        capacity = rollout_episode_capacity(
            staleness_threshold=2,
            version=version,
            rollout_store_size_per_rank=1,
            total_num_train_envs=8,
            rollout_epoch=rollout_epoch,
        )
        assert capacity == (2 + version + 1) * 8 * rollout_epoch


def test_store_size_two_sustains_two_trajectories_per_actor_version():
    episodes_per_trajectory = 8
    previous_capacity = 0
    for version in range(5):
        capacity = rollout_episode_capacity(
            staleness_threshold=1,
            version=version,
            rollout_store_size_per_rank=2,
            total_num_train_envs=episodes_per_trajectory,
            rollout_epoch=1,
        )
        if version > 0:
            assert capacity - previous_capacity == 2 * episodes_per_trajectory
        previous_capacity = capacity


def test_rollout_capacity_rejects_nonpositive_store_size():
    with pytest.raises(ValueError, match="must be positive"):
        rollout_episode_capacity(
            staleness_threshold=1,
            version=0,
            rollout_store_size_per_rank=0,
            total_num_train_envs=8,
            rollout_epoch=1,
        )
