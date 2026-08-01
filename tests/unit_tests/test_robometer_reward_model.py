import numpy as np
import pytest
import torch

from rlinf.algorithms.utils import aggregate_embodied_chunk_rewards
from rlinf.models.embodiment.reward.robometer_reward_model import (
    _apply_stepwise_success_shift,
    _interpolate_insert_progress_from_downsampled_frames,
    _resolve_env_success_from_infos,
    _robometer_boundary_frame_indices,
    reconstruct_robometer_delta_reward,
    reconstruct_robometer_episode_reward,
    robometer_assignment_metric_values,
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


def test_failed_episode_diagnostics_use_reconstruction_success_trace():
    failed = reconstruct_robometer_episode_reward(
        [0.2, 0.4],
        history_len=2,
        pickup_count=0,
        success_trace=[False, False],
        max_frames=2,
        fail_shift=1.0,
        chunk_size=2,
        total_chunks=1,
    )
    successful = reconstruct_robometer_episode_reward(
        [0.2, 0.8],
        history_len=2,
        pickup_count=0,
        success_trace=[False, True],
        max_frames=2,
        fail_shift=1.0,
        chunk_size=2,
        total_chunks=1,
    )
    metrics = robometer_assignment_metric_values({0: failed, 1: successful})

    assert failed.episode_success is False
    assert successful.episode_success is True
    np.testing.assert_allclose(
        metrics["reward/robometer_episode_success_rate"].numpy(), [0.0, 1.0]
    )
    np.testing.assert_allclose(
        metrics["reward/failed_episode_reward_sum"].numpy(), [-1.4]
    )
    np.testing.assert_allclose(
        metrics["reward/successful_episode_reward_sum"].numpy(), [0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        metrics["reward/success_minus_failure_reward_margin"].numpy(), [1.4]
    )
    np.testing.assert_allclose(
        metrics["reward/robometer_initial_progress"].numpy(), [0.2, 0.2]
    )
    np.testing.assert_allclose(
        metrics["reward/robometer_final_progress"].numpy(), [0.4, 0.8]
    )
    np.testing.assert_allclose(
        metrics["reward/failed_episode_positive_violation_rate"].numpy(), [0.0]
    )
    np.testing.assert_allclose(
        metrics["reward/failed_low_level_positive_fraction"].numpy(), [0.0, 0.0]
    )


def test_failed_episode_violation_uses_tolerance_but_positive_fraction_is_strict():
    failed = reconstruct_robometer_episode_reward(
        [1.0000005],
        history_len=1,
        pickup_count=0,
        success_trace=[False],
        max_frames=1,
        fail_shift=1.0,
        chunk_size=1,
        total_chunks=1,
    )
    metrics = robometer_assignment_metric_values({0: failed})

    np.testing.assert_allclose(
        metrics["reward/failed_episode_positive_violation_rate"].numpy(), [0.0]
    )
    np.testing.assert_allclose(
        metrics["reward/failed_low_level_positive_fraction"].numpy(), [1.0]
    )


def _robometer_cfg():
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "server_url": "http://unused",
            "task": "insert",
            "use_frame_steps": False,
            "min_history_size": 2,
            "max_robometer_frames": 60,
            "render_buffer_name": "render_buffer",
            "history_buffers": {"render_buffer": {"history_keys": ["render_images"]}},
        }
    )


def test_compute_reward_queries_all_emitted_prefixes(monkeypatch):
    from rlinf.models.embodiment.reward import robometer_reward_model as module

    captured = {}

    def fake_post(server_url, samples, timeout_s, use_frame_steps):
        captured["samples"] = samples
        return {"outputs_progress": {"progress_pred": [[0.2, 0.8], [0.1, 0.3]]}}

    monkeypatch.setattr(module, "_post_evaluate_batch_npy", fake_post)
    model = module.RobometerHistoryRewardModel(_robometer_cfg())
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    output = model.compute_reward(
        {
            "history_input": {
                "render_buffer": {"render_images": [[frame, frame], [frame, frame]]}
            },
            "env_infos": {"success": np.array([True, False])},
        }
    )

    assert len(captured["samples"]) == 2
    assert captured["samples"][0]["trajectory"]["id"] == "0"
    assert captured["samples"][1]["trajectory"]["id"] == "1"
    np.testing.assert_allclose(output[0].numpy(), np.array([0.2, 0.8]))
    np.testing.assert_allclose(output[1].numpy(), np.array([0.1, 0.3]))


def test_compute_reward_rejects_missing_progress(monkeypatch):
    from rlinf.models.embodiment.reward import robometer_reward_model as module

    monkeypatch.setattr(
        module,
        "_post_evaluate_batch_npy",
        lambda *args, **kwargs: {"outputs_progress": {"progress_pred": []}},
    )
    model = module.RobometerHistoryRewardModel(_robometer_cfg())
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="unexpected number"):
        model.compute_reward(
            {
                "history_input": {"render_buffer": {"render_images": [[frame, frame]]}},
                "env_infos": {"success": np.array([True])},
                "dones": np.array([True]),
            }
        )


def test_compute_reward_skips_empty_prefix_history(monkeypatch):
    from rlinf.models.embodiment.reward import robometer_reward_model as module

    captured = {}

    def fake_post(server_url, samples, timeout_s, use_frame_steps):
        captured["samples"] = samples
        return {"outputs_progress": {"progress_pred": [[0.4, 0.6]]}}

    monkeypatch.setattr(module, "_post_evaluate_batch_npy", fake_post)
    model = module.RobometerHistoryRewardModel(_robometer_cfg())
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    output = model.compute_reward(
        {
            "history_input": {"render_buffer": {"render_images": [[], [frame, frame]]}},
            "env_infos": {"success": np.array([False, False])},
        }
    )

    assert len(captured["samples"]) == 1
    assert captured["samples"][0]["trajectory"]["id"] == "1"
    np.testing.assert_allclose(output[0].numpy(), np.array([0.0, 0.0]))
    np.testing.assert_allclose(output[1].numpy(), np.array([0.4, 0.6]))


def test_compute_reward_skips_prefix_history_below_minimum(monkeypatch):
    from rlinf.models.embodiment.reward import robometer_reward_model as module

    captured = {}

    def fake_post(server_url, samples, timeout_s, use_frame_steps):
        captured["samples"] = samples
        return {"outputs_progress": {"progress_pred": [[0.4, 0.6]]}}

    monkeypatch.setattr(module, "_post_evaluate_batch_npy", fake_post)
    model = module.RobometerHistoryRewardModel(_robometer_cfg())
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    output = model.compute_reward(
        {
            "history_input": {
                "render_buffer": {"render_images": [[frame], [frame, frame]]}
            },
            "env_infos": {"success": np.array([False, False])},
        }
    )

    assert len(captured["samples"]) == 1
    assert captured["samples"][0]["trajectory"]["id"] == "1"
    np.testing.assert_allclose(output[0].numpy(), np.array([0.0, 0.0]))
    np.testing.assert_allclose(output[1].numpy(), np.array([0.4, 0.6]))


# ---------------------------------------------------------------------------
# Delta shaping (Option 2: chunk-boundary-frame diff)
# ---------------------------------------------------------------------------


def test_boundary_frame_indices_count_is_chunks_plus_one():
    # history_len=40, pickup=0, chunk_size=10 -> 4 chunks -> 5 boundary frames.
    idx = _robometer_boundary_frame_indices(40, 0, 10)
    assert idx == [0, 10, 20, 30, 39]  # last clamped from 40 -> 39


def test_boundary_frame_indices_with_pickup_offset():
    # pickup=5, insert=35, chunk_size=10 -> 4 chunks -> 5 boundaries from idx 5.
    idx = _robometer_boundary_frame_indices(40, 5, 10)
    assert idx == [5, 15, 25, 35, 39]


def test_boundary_frame_indices_empty_when_no_history():
    assert _robometer_boundary_frame_indices(0, 0, 10) == []


def test_delta_reward_boundary_diff_plus_success_bonus():
    # 4 chunks, pickup=0, chunk_size=10, history_len=40.
    # success sticky from chunk 1 onward (end-step of chunk 1 = step 19).
    success = [False] * 40
    for s in (19, 29, 39):
        success[s] = True
    progress = [0.0, 0.2, 0.5, 0.8, 1.0]  # 5 boundary frames
    r = reconstruct_robometer_delta_reward(
        progress,
        history_len=40,
        pickup_count=0,
        success_trace=success,
        chunk_size=10,
        total_chunks=4,
        success_bonus=0.1,
    )
    deltas = [0.2, 0.3, 0.3, 0.2]
    expected = [
        deltas[0],  # chunk 0: no success (step 9 False)
        deltas[1] + 0.1,  # chunk 1: success (step 19)
        deltas[2] + 0.1,  # chunk 2: success (step 29)
        deltas[3] + 0.1,  # chunk 3: success (step 39)
    ]
    np.testing.assert_allclose(r.chunk_reward[:, 0], expected, atol=1e-6)
    # index 1..9 are zero (reward only at index 0).
    np.testing.assert_array_equal(r.chunk_reward[:, 1:], np.zeros((4, 9)))
    # loss_mask True for ALL sub-steps of every insertion chunk (matches absolute).
    np.testing.assert_array_equal(r.chunk_loss_mask, np.ones((4, 10), dtype=bool))
    assert r.episode_success is True
    assert r.initial_progress == pytest.approx(0.0)
    assert r.final_progress == pytest.approx(1.0)
    assert r.success_bonus_sum == pytest.approx(0.3)


def test_delta_reward_failure_chunk_has_no_bonus():
    success = [False] * 40  # never succeeds
    progress = [0.0, 0.1, 0.2, 0.3, 0.4]
    r = reconstruct_robometer_delta_reward(
        progress,
        history_len=40,
        pickup_count=0,
        success_trace=success,
        chunk_size=10,
        total_chunks=4,
        success_bonus=0.1,
    )
    # Only deltas, no bonus.
    np.testing.assert_allclose(r.chunk_reward[:, 0], [0.1, 0.1, 0.1, 0.1], atol=1e-6)
    assert r.episode_success is False


def test_delta_reward_failure_terminal_penalty_is_applied_once():
    success = [False] * 40
    progress = [0.0, 0.1, 0.2, 0.3, 0.4]
    r = reconstruct_robometer_delta_reward(
        progress,
        history_len=40,
        pickup_count=0,
        success_trace=success,
        chunk_size=10,
        total_chunks=4,
        success_bonus=0.1,
        failure_terminal_penalty=-0.4,
    )

    np.testing.assert_allclose(r.chunk_reward[:, 0], [0.1, 0.1, 0.1, -0.3], atol=1e-6)
    assert r.chunk_reward[:, 0].sum() == pytest.approx(0.0)


def test_delta_reward_success_does_not_receive_failure_terminal_penalty():
    success = [False] * 40
    success[19] = True
    progress = [0.0, 0.1, 0.2, 0.3, 0.4]
    r = reconstruct_robometer_delta_reward(
        progress,
        history_len=40,
        pickup_count=0,
        success_trace=success,
        chunk_size=10,
        total_chunks=4,
        success_bonus=0.1,
        failure_terminal_penalty=-0.4,
    )

    np.testing.assert_allclose(r.chunk_reward[:, 0], [0.1, 0.2, 0.1, 0.1], atol=1e-6)
    assert r.chunk_reward[:, 0].sum() == pytest.approx(0.5)


def test_delta_reward_telescope_sum_equals_final_progress():
    # With no success bonus and no success, sum of deltas == final - initial.
    success = [False] * 40
    progress = [0.0, 0.1, 0.3, 0.6, 1.0]
    r = reconstruct_robometer_delta_reward(
        progress,
        history_len=40,
        pickup_count=0,
        success_trace=success,
        chunk_size=10,
        total_chunks=4,
        success_bonus=0.0,
    )
    np.testing.assert_allclose(
        r.chunk_reward[:, 0].sum(), progress[-1] - progress[0], atol=1e-6
    )


def test_delta_reward_discounted_sum_reproduces_scalar():
    # The [chunk_size] placement (scalar at idx 0, zeros elsewhere, mask all True)
    # must reproduce the per-chunk delta exactly under discounted_sum (gamma^0=1).
    success = [False] * 40
    progress = [0.0, 0.2, 0.5, 0.8, 1.0]
    r = reconstruct_robometer_delta_reward(
        progress,
        history_len=40,
        pickup_count=0,
        success_trace=success,
        chunk_size=10,
        total_chunks=4,
        success_bonus=0.0,
    )
    chunk_rewards = torch.as_tensor(r.chunk_reward).unsqueeze(1)  # [4, 1, 10]
    chunk_loss_mask = torch.as_tensor(r.chunk_loss_mask).unsqueeze(1)
    aggregated = aggregate_embodied_chunk_rewards(
        chunk_rewards, chunk_loss_mask, gamma=0.99, aggregation="discounted_sum"
    )  # [4, 1, 1]
    np.testing.assert_allclose(
        aggregated.squeeze().numpy(), [0.2, 0.3, 0.3, 0.2], atol=1e-6
    )


def test_delta_reward_rejects_short_boundary_progress():
    with pytest.raises(ValueError, match="boundary progress is shorter"):
        reconstruct_robometer_delta_reward(
            [0.0, 0.2, 0.5],  # 3 values, but 4 chunks -> need 5
            history_len=40,
            pickup_count=0,
            success_trace=[False] * 40,
            chunk_size=10,
            total_chunks=4,
            success_bonus=0.1,
        )


def test_delta_reward_rejects_misaligned_success_trace():
    with pytest.raises(ValueError, match="Success trace must align"):
        reconstruct_robometer_delta_reward(
            [0.0, 0.2, 0.5, 0.8, 1.0],
            history_len=40,
            pickup_count=0,
            success_trace=[False] * 10,  # wrong length
            chunk_size=10,
            total_chunks=4,
            success_bonus=0.1,
        )


def test_compute_reward_delta_selects_boundary_frames(monkeypatch):
    """Delta mode POSTs only chunk-boundary frames (5), not the full history (40)."""
    from rlinf.models.embodiment.reward import robometer_reward_model as module

    captured = {}

    def fake_post(server_url, samples, timeout_s, use_frame_steps):
        captured["samples"] = samples
        captured["n_frames"] = [
            s["trajectory"]["frames"].shape[0] for s in samples
        ]
        return {
            "outputs_progress": {
                "progress_pred": [[0.0, 0.2, 0.5, 0.8, 1.0]]
            }
        }

    monkeypatch.setattr(module, "_post_evaluate_batch_npy", fake_post)
    model = module.RobometerHistoryRewardModel(_robometer_cfg())
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frames = [frame] * 40  # 40 low-level steps -> 4 chunks -> 5 boundary frames
    output = model.compute_reward(
        {
            "history_input": {"render_buffer": {"render_images": [frames]}},
            "env_infos": {"success": np.array([True])},
            "dones": np.array([True]),
            "shaping": "delta",
            "pickup_counts": [0],
            "chunk_size": 10,
        }
    )
    # Only the 5 boundary frames were POSTed (not all 40).
    assert captured["n_frames"] == [5]
    np.testing.assert_allclose(
        output[0].numpy(), [0.0, 0.2, 0.5, 0.8, 1.0], atol=1e-6
    )


def test_compute_reward_delta_requires_pickup_counts(monkeypatch):
    from rlinf.models.embodiment.reward import robometer_reward_model as module

    monkeypatch.setattr(
        module,
        "_post_evaluate_batch_npy",
        lambda *a, **k: {"outputs_progress": {"progress_pred": [[]]}},
    )
    model = module.RobometerHistoryRewardModel(_robometer_cfg())
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="pickup_counts"):
        model.compute_reward(
            {
                "history_input": {
                    "render_buffer": {"render_images": [[frame, frame]]}
                },
                "env_infos": {"success": np.array([True])},
                "dones": np.array([True]),
                "shaping": "delta",
                "chunk_size": 10,
            }
        )
