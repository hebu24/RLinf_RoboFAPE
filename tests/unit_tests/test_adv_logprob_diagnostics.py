import math

import torch

from rlinf.workers.actor.async_ppo_fsdp_worker import (
    _ADV_LOGPROB_METRIC_REASON_TOO_FEW_SAMPLES,
    _ADV_LOGPROB_METRIC_REASON_ZERO_ADVANTAGE_VARIANCE,
    _ADV_LOGPROB_METRIC_REASON_ZERO_LOGPROB_DELTA_VARIANCE,
    compute_adv_logprob_diagnostics,
)


def test_adv_logprob_diagnostics_chunk_level_positive_alignment():
    metrics = compute_adv_logprob_diagnostics(
        advantages=torch.tensor([2.0, 1.0, -1.0, -2.0], dtype=torch.float32),
        prev_logprobs=torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        post_update_logprobs=torch.tensor([0.4, 0.2, -0.2, -0.4], dtype=torch.float32),
        loss_mask=torch.tensor([True, True, True, True]),
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert metrics["actor/adv_logprob_metric_valid"].item() == 1.0
    assert metrics["actor/adv_logprob_delta_corr"].item() > 0.99
    assert metrics["actor/adv_logprob_direction_match_rate"].item() == 1.0
    assert metrics["actor/mean_logprob_delta_pos_adv"].item() > 0
    assert metrics["actor/mean_logprob_delta_neg_adv"].item() < 0


def test_adv_logprob_diagnostics_chunk_level_negative_alignment():
    metrics = compute_adv_logprob_diagnostics(
        advantages=torch.tensor([2.0, 1.0, -1.0, -2.0], dtype=torch.float32),
        prev_logprobs=torch.zeros(4, dtype=torch.float32),
        post_update_logprobs=torch.tensor([-0.4, -0.2, 0.2, 0.4], dtype=torch.float32),
        loss_mask=torch.tensor([True, True, True, True]),
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert metrics["actor/adv_logprob_delta_corr"].item() < -0.99
    assert metrics["actor/adv_logprob_direction_match_rate"].item() == 0.0


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

    assert metrics["actor/adv_logprob_metric_valid"].item() == 1.0
    assert metrics["actor/adv_logprob_direction_match_rate"].item() == 1.0
    assert metrics["actor/adv_logprob_metric_numel"].item() == 2.0


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

    assert metrics["actor/adv_logprob_metric_valid"].item() == 1.0
    assert metrics["actor/adv_logprob_metric_numel"].item() == 2.0
    assert metrics["actor/adv_logprob_direction_match_rate"].item() == 1.0


def test_adv_logprob_diagnostics_too_few_samples_is_invalid():
    metrics = compute_adv_logprob_diagnostics(
        advantages=torch.tensor([1.0], dtype=torch.float32),
        prev_logprobs=torch.tensor([0.0], dtype=torch.float32),
        post_update_logprobs=torch.tensor([0.2], dtype=torch.float32),
        loss_mask=torch.tensor([True]),
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert metrics["actor/adv_logprob_metric_valid"].item() == 0.0
    assert (
        metrics["actor/adv_logprob_metric_invalid_reason"].item()
        == _ADV_LOGPROB_METRIC_REASON_TOO_FEW_SAMPLES
    )
    assert math.isnan(metrics["actor/adv_logprob_delta_corr"].item())


def test_adv_logprob_diagnostics_zero_advantage_variance_is_invalid():
    metrics = compute_adv_logprob_diagnostics(
        advantages=torch.tensor([1.0, 1.0], dtype=torch.float32),
        prev_logprobs=torch.tensor([0.0, 0.0], dtype=torch.float32),
        post_update_logprobs=torch.tensor([0.2, 0.5], dtype=torch.float32),
        loss_mask=torch.tensor([True, True]),
        logprob_type="chunk_level",
        single_action_dim=2,
    )

    assert metrics["actor/adv_logprob_metric_valid"].item() == 0.0
    assert (
        metrics["actor/adv_logprob_metric_invalid_reason"].item()
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

    assert metrics["actor/adv_logprob_metric_valid"].item() == 0.0
    assert (
        metrics["actor/adv_logprob_metric_invalid_reason"].item()
        == _ADV_LOGPROB_METRIC_REASON_ZERO_LOGPROB_DELTA_VARIANCE
    )
