import math

import pytest
import torch

from rlinf.algorithms.losses import compute_decoupled_ppo_actor_loss
from rlinf.workers.actor.async_ppo_fsdp_worker import (
    _ADV_LOGPROB_METRIC_REASON_TOO_FEW_SAMPLES,
    _ADV_LOGPROB_METRIC_REASON_ZERO_ADVANTAGE_VARIANCE,
    _ADV_LOGPROB_METRIC_REASON_ZERO_LOGPROB_DELTA_VARIANCE,
    compute_adv_logprob_diagnostics,
    compute_gradient_conflict_metrics,
    compute_policy_adv_logprob_diagnostics,
    compute_post_update_ppo_surrogate_metrics,
    compute_post_update_proximal_metrics,
    compute_preupdate_logprob_mismatch_metrics,
)


def test_post_update_ppo_surrogate_matches_decoupled_actor_loss():
    advantages = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float32)
    behavior = torch.tensor([-0.2, 0.0, -2.0], dtype=torch.float32)
    proximal = torch.zeros(3, dtype=torch.float32)
    post = torch.tensor([0.1, -0.1, 0.8], dtype=torch.float32)
    loss_mask = torch.ones(3, dtype=torch.bool)
    common = {
        "old_logprobs": behavior,
        "proximal_logprobs": proximal,
        "advantages": advantages,
        "clip_ratio_low": 0.1,
        "clip_ratio_high": 0.1,
        "clip_ratio_c": 3.0,
        "loss_mask": loss_mask,
        "loss_mask_sum": torch.full((3,), 2.0),
        "max_episode_steps": 2,
        "behave_weight_threshold": 2.0,
    }
    expected_pre, _ = compute_decoupled_ppo_actor_loss(logprobs=proximal, **common)
    expected_post, _ = compute_decoupled_ppo_actor_loss(logprobs=post, **common)

    metrics = compute_post_update_ppo_surrogate_metrics(
        advantages=advantages,
        old_logprobs=behavior,
        proximal_logprobs=proximal,
        post_update_logprobs=post,
        loss_mask=loss_mask,
        loss_mask_sum=torch.full((3,), 2.0),
        max_episode_steps=2,
        logprob_type="chunk_level",
        single_action_dim=1,
        reward_type="chunk_level",
        clip_ratio_low=0.1,
        clip_ratio_high=0.1,
        clip_ratio_c=3.0,
        behave_weight_threshold=2.0,
    )

    assert metrics["actor/pre_update_ppo_actor_loss"] == pytest.approx(
        expected_pre.item()
    )
    assert metrics["actor/post_update_ppo_actor_loss"] == pytest.approx(
        expected_post.item()
    )
    assert metrics["actor/post_update_ppo_surrogate_improvement"] == pytest.approx(
        (expected_pre - expected_post).item()
    )
    # The third element has behavior weight exp(2) and is excluded by threshold.
    assert metrics["actor/post_update_behavior_valid_fraction"] == pytest.approx(2 / 3)
    assert metrics["actor/post_update_ppo_improved_fraction"] == pytest.approx(1.0)
    assert metrics["actor/post_update_first_order_surrogate_gain"] > 0


def test_post_update_ppo_surrogate_detects_reverse_update():
    advantages = torch.tensor([2.0, 1.0, -1.0, -2.0], dtype=torch.float32)
    proximal = torch.zeros(4, dtype=torch.float32)
    post = torch.tensor([-0.05, -0.02, 0.02, 0.05], dtype=torch.float32)

    metrics = compute_post_update_ppo_surrogate_metrics(
        advantages=advantages,
        old_logprobs=proximal,
        proximal_logprobs=proximal,
        post_update_logprobs=post,
        loss_mask=torch.ones(4, dtype=torch.bool),
        loss_mask_sum=None,
        max_episode_steps=None,
        logprob_type="chunk_level",
        single_action_dim=1,
        reward_type="chunk_level",
        clip_ratio_low=0.1,
        clip_ratio_high=0.1,
        clip_ratio_c=3.0,
        behave_weight_threshold=2.0,
    )

    assert metrics["actor/post_update_ppo_surrogate_improvement"] < 0
    assert metrics["actor/post_update_first_order_surrogate_gain"] < 0
    assert metrics["actor/post_update_pg_weighted_logprob_corr"] < -0.99
    assert metrics["actor/post_update_ppo_improved_fraction"] == 0.0


def test_post_update_ppo_surrogate_empty_effective_mask_is_safe():
    metrics = compute_post_update_ppo_surrogate_metrics(
        advantages=torch.tensor([1.0], dtype=torch.float32),
        old_logprobs=torch.tensor([0.0], dtype=torch.float32),
        proximal_logprobs=torch.tensor([0.0], dtype=torch.float32),
        post_update_logprobs=torch.tensor([0.1], dtype=torch.float32),
        loss_mask=torch.zeros(1, dtype=torch.bool),
        loss_mask_sum=None,
        max_episode_steps=None,
        logprob_type="chunk_level",
        single_action_dim=1,
        reward_type="chunk_level",
        clip_ratio_low=0.1,
        clip_ratio_high=0.1,
        clip_ratio_c=3.0,
        behave_weight_threshold=2.0,
    )

    assert metrics["actor/pre_update_ppo_actor_loss"] == 0.0
    assert metrics["actor/post_update_ppo_actor_loss"] == 0.0
    assert math.isnan(metrics["actor/post_update_ppo_improved_fraction"])


def test_post_update_ppo_surrogate_preprocesses_chunk_logprobs_like_training():
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float32)
    proximal = torch.zeros((2, 2, 3), dtype=torch.float32)
    post = proximal.clone()
    post[0] = 0.02
    post[1] = -0.02

    metrics = compute_post_update_ppo_surrogate_metrics(
        advantages=advantages,
        old_logprobs=proximal,
        proximal_logprobs=proximal,
        post_update_logprobs=post,
        loss_mask=torch.ones(2, dtype=torch.bool),
        loss_mask_sum=torch.full((2,), 2.0),
        max_episode_steps=2,
        logprob_type="chunk_level",
        single_action_dim=3,
        reward_type="chunk_level",
        clip_ratio_low=0.1,
        clip_ratio_high=0.1,
        clip_ratio_c=3.0,
        behave_weight_threshold=2.0,
    )

    assert metrics["actor/post_update_ppo_surrogate_improvement"] > 0
    assert metrics["actor/post_update_ppo_improved_fraction"] == 1.0


def test_policy_adv_diagnostics_ignore_behavior_backend_mismatch():
    advantages = torch.tensor([2.0, 1.0, -1.0, -2.0])
    proximal = torch.full((4,), -0.2)
    behavior = proximal - 0.5
    post = proximal + torch.tensor([0.4, 0.2, -0.2, -0.4])

    metrics = compute_policy_adv_logprob_diagnostics(
        advantages=advantages,
        proximal_logprobs=proximal,
        post_update_logprobs=post,
        loss_mask=torch.ones(4, dtype=torch.bool),
        logprob_type="chunk_level",
        single_action_dim=2,
    )
    behavior_based = compute_adv_logprob_diagnostics(
        advantages=advantages,
        prev_logprobs=behavior,
        post_update_logprobs=post,
        loss_mask=torch.ones(4, dtype=torch.bool),
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert metrics["actor/policy_adv_logprob_corr"] > 0.99
    assert metrics["actor/policy_adv_direction_match_rate"] == 1.0
    assert metrics["actor/policy_top_quartile_logprob_delta"] > 0
    assert metrics["actor/adv_weighted_policy_logprob_delta"] > 0
    assert behavior_based["actor/adv_logprob_direction_match_rate"] != 1.0


def test_policy_adv_diagnostics_report_negative_alignment_and_respect_mask():
    metrics = compute_policy_adv_logprob_diagnostics(
        advantages=torch.tensor([2.0, -2.0, 100.0]),
        proximal_logprobs=torch.zeros(3),
        post_update_logprobs=torch.tensor([-0.2, 0.2, 100.0]),
        loss_mask=torch.tensor([True, True, False]),
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert metrics["actor/policy_adv_logprob_corr"] < -0.99
    assert metrics["actor/policy_adv_direction_match_rate"] == 0.0
    assert metrics["actor/adv_weighted_policy_logprob_delta"] < 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_policy_adv_diagnostics_normalize_inputs_to_advantage_device():
    metrics = compute_policy_adv_logprob_diagnostics(
        advantages=torch.tensor([2.0, -2.0]),
        proximal_logprobs=torch.zeros(2, device="cuda"),
        post_update_logprobs=torch.tensor([0.2, -0.2], device="cuda"),
        loss_mask=torch.ones(2, dtype=torch.bool, device="cuda"),
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert metrics["actor/policy_adv_direction_match_rate"] == 1.0
    assert metrics["actor/adv_weighted_policy_logprob_delta"] > 0


def test_post_update_proximal_metrics_measure_optimizer_step_and_mask():
    proximal = torch.zeros(4)
    post = torch.tensor([0.2, -0.2, 100.0, float("nan")])
    metrics = compute_post_update_proximal_metrics(
        proximal_logprobs=proximal,
        post_update_logprobs=post,
        loss_mask=torch.tensor([True, True, False, True]),
        logprob_type="chunk_level",
        single_action_dim=2,
        clip_ratio_low=0.1,
        clip_ratio_high=0.1,
    )

    assert metrics["actor/post_update_proximal_approx_kl"] == pytest.approx(0.0)
    assert metrics["actor/post_update_proximal_ratio"] == pytest.approx(
        (math.exp(0.2) + math.exp(-0.2)) / 2
    )
    assert metrics["actor/post_update_proximal_clip_fraction"] == 1.0
    assert metrics["actor/post_update_logprob_delta_mean"] == pytest.approx(0.0)
    assert metrics["actor/post_update_logprob_delta_abs_mean"] == pytest.approx(0.2)
    assert metrics["actor/post_update_logprob_delta_abs_max"] == pytest.approx(0.2)


def test_post_update_proximal_metrics_empty_mask_is_safe():
    metrics = compute_post_update_proximal_metrics(
        proximal_logprobs=torch.zeros(2),
        post_update_logprobs=torch.ones(2),
        loss_mask=torch.zeros(2, dtype=torch.bool),
        logprob_type="chunk_level",
        single_action_dim=2,
        clip_ratio_low=0.1,
        clip_ratio_high=0.1,
    )

    assert all(math.isnan(value) for value in metrics.values())


def test_gradient_conflict_metrics_from_three_norms():
    orthogonal = compute_gradient_conflict_metrics(3.0, 4.0, 5.0)
    aligned = compute_gradient_conflict_metrics(3.0, 4.0, 7.0)
    opposed = compute_gradient_conflict_metrics(3.0, 4.0, 1.0)
    invalid = compute_gradient_conflict_metrics(0.0, 4.0, 4.0)

    assert orthogonal["actor_critic/shared_grad_cosine_sampled"] == pytest.approx(0)
    assert aligned["actor_critic/shared_grad_cosine_sampled"] == pytest.approx(1)
    assert opposed["actor_critic/shared_grad_cosine_sampled"] == pytest.approx(-1)
    assert orthogonal["critic/policy_grad_norm_ratio_sampled"] == pytest.approx(4 / 3)
    assert invalid["actor/grad_diagnostics_valid"] == 0.0
    assert math.isnan(invalid["actor_critic/shared_grad_cosine_sampled"])


def test_preupdate_proximal_anchor_is_identity_but_behavior_mismatch_is_visible():
    behavior = torch.tensor([-0.3, -0.1, -0.2], dtype=torch.float32)
    proximal = behavior + torch.tensor([0.2, 0.0, -0.2])
    metrics = compute_preupdate_logprob_mismatch_metrics(
        proximal_logprobs=proximal,
        behavior_logprobs=behavior,
        loss_mask=torch.ones(3, dtype=torch.bool),
        logprob_type="chunk_level",
        single_action_dim=7,
        clip_ratio_low=0.1,
        clip_ratio_high=0.1,
    )

    assert metrics["actor/preupdate_behavior_approx_kl"] == pytest.approx(0.0)
    assert metrics["actor/preupdate_behavior_ratio"] != pytest.approx(1.0)
    assert metrics["actor/preupdate_behavior_clip_fraction"] == pytest.approx(2 / 3)
    assert metrics["actor/preupdate_behavior_logprob_delta_abs_max"] == pytest.approx(
        0.2
    )


def test_frozen_proximal_anchor_gives_unit_ratio_before_parameter_update():
    proximal = torch.tensor([-0.2, -0.4], dtype=torch.float32)
    current = proximal.clone()
    ratio = torch.exp(current - proximal)

    torch.testing.assert_close(ratio, torch.ones_like(ratio))

    _, metrics = compute_decoupled_ppo_actor_loss(
        logprobs=current,
        old_logprobs=proximal - 0.2,
        proximal_logprobs=proximal,
        advantages=torch.tensor([1.0, -1.0], dtype=torch.float32),
        clip_ratio_low=0.1,
        clip_ratio_high=0.1,
        loss_mask=torch.ones(2, dtype=torch.bool),
        behave_weight_threshold=2.0,
    )
    assert metrics["actor/proximal_ratio"].item() == pytest.approx(1.0)
    assert metrics["actor/proximal_approx_kl"].item() == pytest.approx(0.0)
    assert metrics["actor/behav_approx_kl"].item() == pytest.approx(-0.2)


def test_decoupled_actor_metrics_keep_schema_for_all_masked_rank():
    common = {
        "logprobs": torch.tensor([0.1, 0.2], dtype=torch.float32),
        "old_logprobs": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "proximal_logprobs": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "advantages": torch.tensor([1.0, -1.0], dtype=torch.float32),
        "versions": torch.tensor([8.0, 8.0], dtype=torch.float32),
        "current_version": 9,
        "clip_ratio_low": 0.1,
        "clip_ratio_high": 0.1,
        "behave_weight_threshold": 2.0,
    }
    _, trainable_metrics = compute_decoupled_ppo_actor_loss(
        loss_mask=torch.tensor([True, True]), **common
    )
    _, noop_metrics = compute_decoupled_ppo_actor_loss(
        loss_mask=torch.tensor([False, False]), **common
    )

    assert set(noop_metrics) == set(trainable_metrics)
    assert torch.isnan(noop_metrics["actor/average_version"])
    assert noop_metrics["actor/current_version"].item() == 9.0


def test_adv_logprob_diagnostics_chunk_level_positive_alignment():
    metrics = compute_adv_logprob_diagnostics(
        advantages=torch.tensor([2.0, 1.0, -1.0, -2.0], dtype=torch.float32),
        prev_logprobs=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        post_update_logprobs=torch.tensor([0.4, 0.2, -0.2, -0.4], dtype=torch.float32),
        loss_mask=torch.tensor([True, True, True, True]),
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert metrics["actor/adv_logprob_metric_valid"] == 1.0
    assert metrics["actor/adv_logprob_delta_corr"] > 0.99
    assert metrics["actor/adv_logprob_direction_match_rate"] == 1.0
    assert metrics["actor/mean_logprob_delta_pos_adv"] > 0
    assert metrics["actor/mean_logprob_delta_neg_adv"] < 0


def test_adv_logprob_diagnostics_chunk_level_negative_alignment():
    metrics = compute_adv_logprob_diagnostics(
        advantages=torch.tensor([2.0, 1.0, -1.0, -2.0], dtype=torch.float32),
        prev_logprobs=torch.zeros(4, dtype=torch.float32),
        post_update_logprobs=torch.tensor([-0.4, -0.2, 0.2, 0.4], dtype=torch.float32),
        loss_mask=torch.tensor([True, True, True, True]),
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert metrics["actor/adv_logprob_delta_corr"] < -0.99
    assert metrics["actor/adv_logprob_direction_match_rate"] == 0.0


def test_adv_logprob_diagnostics_action_level_reduces_per_item():
    metrics = compute_adv_logprob_diagnostics(
        advantages=torch.tensor([2.0, -1.0], dtype=torch.float32),
        prev_logprobs=torch.zeros((2, 4), dtype=torch.float32),
        post_update_logprobs=torch.tensor(
            [[0.2, 0.1, 0.3, 0.2], [-0.3, -0.2, -0.1, -0.4]], dtype=torch.float32
        ),
        loss_mask=torch.tensor([True, True]),
        logprob_type="action_level",
        single_action_dim=2,
    )

    assert metrics["actor/adv_logprob_metric_valid"] == 1.0
    assert metrics["actor/adv_logprob_direction_match_rate"] == 1.0
    assert metrics["actor/adv_logprob_metric_numel"] == 2.0


def test_adv_logprob_diagnostics_token_level_uses_any_mask_per_item():
    metrics = compute_adv_logprob_diagnostics(
        advantages=torch.tensor([[2.0, 2.0], [-1.0, -1.0]], dtype=torch.float32),
        prev_logprobs=torch.zeros((2, 4), dtype=torch.float32),
        post_update_logprobs=torch.tensor(
            [[0.2, 0.1, 0.3, 0.2], [-0.3, -0.2, -0.1, -0.4]], dtype=torch.float32
        ),
        loss_mask=torch.tensor([[True, False], [True, False]]),
        logprob_type="token_level",
        single_action_dim=2,
    )

    assert metrics["actor/adv_logprob_metric_valid"] == 1.0
    assert metrics["actor/adv_logprob_metric_numel"] == 2.0
    assert metrics["actor/adv_logprob_direction_match_rate"] == 1.0


def test_adv_logprob_diagnostics_too_few_samples_is_invalid():
    metrics = compute_adv_logprob_diagnostics(
        advantages=torch.tensor([1.0], dtype=torch.float32),
        prev_logprobs=torch.tensor([0.0], dtype=torch.float32),
        post_update_logprobs=torch.tensor([0.2], dtype=torch.float32),
        loss_mask=torch.tensor([True]),
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert metrics["actor/adv_logprob_metric_valid"] == 0.0
    assert (
        metrics["actor/adv_logprob_metric_invalid_reason"]
        == _ADV_LOGPROB_METRIC_REASON_TOO_FEW_SAMPLES
    )
    assert math.isnan(metrics["actor/adv_logprob_delta_corr"])


def test_adv_logprob_diagnostics_zero_advantage_variance_is_invalid():
    metrics = compute_adv_logprob_diagnostics(
        advantages=torch.tensor([1.0, 1.0], dtype=torch.float32),
        prev_logprobs=torch.tensor([0.0, 0.0], dtype=torch.float32),
        post_update_logprobs=torch.tensor([0.2, 0.5], dtype=torch.float32),
        loss_mask=torch.tensor([True, True]),
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert metrics["actor/adv_logprob_metric_valid"] == 0.0
    assert (
        metrics["actor/adv_logprob_metric_invalid_reason"]
        == _ADV_LOGPROB_METRIC_REASON_ZERO_ADVANTAGE_VARIANCE
    )


def test_adv_logprob_diagnostics_zero_delta_variance_is_invalid():
    metrics = compute_adv_logprob_diagnostics(
        advantages=torch.tensor([2.0, -1.0], dtype=torch.float32),
        prev_logprobs=torch.tensor([0.0, 0.0], dtype=torch.float32),
        post_update_logprobs=torch.tensor([0.2, 0.2], dtype=torch.float32),
        loss_mask=torch.tensor([True, True]),
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert metrics["actor/adv_logprob_metric_valid"] == 0.0
    assert (
        metrics["actor/adv_logprob_metric_invalid_reason"]
        == _ADV_LOGPROB_METRIC_REASON_ZERO_LOGPROB_DELTA_VARIANCE
    )
