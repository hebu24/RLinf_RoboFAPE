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
    compute_preupdate_logprob_mismatch_metrics,
)


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
