import numpy as np
import pytest

from rlinf.models.embodiment.reward.robometer_reward_model import (
    _apply_stepwise_success_shift,
    _interpolate_insert_progress_from_downsampled_frames,
    _resolve_env_success_from_infos,
)


def test_resolve_env_success_prefers_final_info_episode_success_once():
    env_infos = {
        "final_info": {"episode": {"success_once": np.array([True, False])}},
        "episode": {"success_once": np.array([False, True])},
        "success": np.array([False, True]),
    }

    assert _resolve_env_success_from_infos(env_infos, 0) is True
    assert _resolve_env_success_from_infos(env_infos, 1) is False


def test_resolve_env_success_falls_back_to_episode_success_once():
    env_infos = {
        "episode": {"success_once": np.array([False, True])},
        "success": np.array([True, False]),
    }

    assert _resolve_env_success_from_infos(env_infos, 0) is False
    assert _resolve_env_success_from_infos(env_infos, 1) is True


def test_resolve_env_success_falls_back_to_root_success():
    env_infos = {"success": np.array([0, 1])}

    assert _resolve_env_success_from_infos(env_infos, 0) is False
    assert _resolve_env_success_from_infos(env_infos, 1) is True


def test_resolve_env_success_raises_when_missing():
    with pytest.raises(ValueError, match="env-side success signal"):
        _resolve_env_success_from_infos({"episode": {}}, 0)


def test_interpolate_insert_progress_returns_empty_mask_without_labels():
    rewards, mask = _interpolate_insert_progress_from_downsampled_frames(
        progress=[],
        history_len=6,
        pickup_count=2,
        max_frames=6,
    )

    np.testing.assert_array_equal(rewards, np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(mask, np.zeros(4, dtype=bool))


def test_interpolate_insert_progress_constant_fills_single_label():
    rewards, mask = _interpolate_insert_progress_from_downsampled_frames(
        progress=[0.1, 0.7],
        history_len=10,
        pickup_count=8,
        max_frames=2,
    )

    np.testing.assert_allclose(rewards, np.array([0.7, 0.7], dtype=np.float32))
    np.testing.assert_array_equal(mask, np.array([True, True]))


def test_interpolate_insert_progress_linearly_interpolates_and_edge_fills():
    rewards, mask = _interpolate_insert_progress_from_downsampled_frames(
        progress=[0.0, 0.2, 0.8],
        history_len=8,
        pickup_count=2,
        max_frames=3,
    )

    np.testing.assert_allclose(
        rewards,
        np.array([0.2, 0.2, 0.35, 0.5, 0.65, 0.8], dtype=np.float32),
    )
    np.testing.assert_array_equal(mask, np.ones(6, dtype=bool))


def test_apply_stepwise_success_shift_only_rewards_success_steps():
    rewards = _apply_stepwise_success_shift(
        per_step_progress=np.array([0.2, 0.4, 0.7, 0.9], dtype=np.float32),
        per_step_success=np.array([False, False, True, False]),
        fail_shift=1.0,
    )

    np.testing.assert_allclose(
        rewards,
        np.array([-0.8, -0.6, 0.7, -0.1], dtype=np.float32),
        atol=1e-6,
    )
