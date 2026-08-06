import pytest
import torch

from rlinf.algorithms.losses import (
    compute_explained_variance_statistics,
    compute_ppo_critic_loss,
    explained_variance_metrics_from_statistics,
)
from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor


def _critic_inputs(values, returns, loss_mask=None):
    prev_values = torch.zeros_like(values)
    if loss_mask is None:
        loss_mask = torch.ones_like(values, dtype=torch.bool)
    loss_mask_sum = loss_mask.sum().reshape(1).expand_as(loss_mask)
    return {
        "values": values,
        "prev_values": prev_values,
        "returns": returns,
        "value_clip": 0.2,
        "huber_delta": 10.0,
        "loss_mask": loss_mask,
        "loss_mask_sum": loss_mask_sum,
        "max_episode_steps": int(loss_mask.numel()),
    }


def test_explained_variance_reports_too_few_samples():
    kwargs = _critic_inputs(
        values=torch.tensor([0.2], dtype=torch.float32),
        returns=torch.tensor([0.7], dtype=torch.float32),
        loss_mask=torch.tensor([True]),
    )
    _, metrics = compute_ppo_critic_loss(**kwargs)
    assert torch.isnan(metrics["critic/explained_variance"])
    assert float(metrics["critic/explained_variance_valid"].item()) == 0.0
    assert float(metrics["critic/explained_variance_numel"].item()) == 1.0
    assert float(metrics["critic/explained_variance_invalid_reason"].item()) == 1.0


def test_explained_variance_reports_zero_return_variance():
    kwargs = _critic_inputs(
        values=torch.tensor([0.1, 0.2], dtype=torch.float32),
        returns=torch.tensor([1.0, 1.0], dtype=torch.float32),
        loss_mask=torch.tensor([True, True]),
    )
    _, metrics = compute_ppo_critic_loss(**kwargs)
    assert torch.isnan(metrics["critic/explained_variance"])
    assert float(metrics["critic/explained_variance_valid"].item()) == 0.0
    assert float(metrics["critic/explained_variance_numel"].item()) == 2.0
    assert float(metrics["critic/explained_variance_invalid_reason"].item()) == 2.0
    assert float(metrics["critic/explained_variance_var_returns"].item()) == 0.0


def test_explained_variance_reports_valid_stats():
    kwargs = _critic_inputs(
        values=torch.tensor([0.0, 0.5, 1.5], dtype=torch.float32),
        returns=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        loss_mask=torch.tensor([True, True, True]),
    )
    _, metrics = compute_ppo_critic_loss(**kwargs)
    assert float(metrics["critic/explained_variance_valid"].item()) == 1.0
    assert float(metrics["critic/explained_variance_numel"].item()) == 3.0
    assert float(metrics["critic/explained_variance_invalid_reason"].item()) == 0.0
    assert torch.isfinite(metrics["critic/explained_variance"])
    assert torch.isfinite(metrics["critic/explained_variance_var_returns"])


def test_summed_statistics_match_direct_concatenated_explained_variance():
    returns_parts = (
        torch.tensor([0.0]),
        torch.tensor([1.0, 2.0, 4.0]),
    )
    values_parts = (
        torch.tensor([0.2]),
        torch.tensor([0.8, 2.5, 3.0]),
    )
    summed_statistics = sum(
        (
            compute_explained_variance_statistics(returns, values)
            for returns, values in zip(returns_parts, values_parts)
        ),
        torch.zeros(6, dtype=torch.float64),
    )
    summed_metrics = explained_variance_metrics_from_statistics(summed_statistics)

    all_returns = torch.cat(returns_parts).double()
    all_values = torch.cat(values_parts).double()
    expected = 1.0 - torch.var(all_returns - all_values) / torch.var(all_returns)

    torch.testing.assert_close(summed_metrics["critic/explained_variance"], expected)
    assert summed_metrics["critic/explained_variance_valid"].item() == 1.0
    assert summed_metrics["critic/explained_variance_numel"].item() == 4.0


def test_global_explained_variance_uses_remote_rank_statistics(monkeypatch):
    actor = object.__new__(AsyncPPOEmbodiedFSDPActor)
    actor.device = torch.device("cpu")
    actor.rollout_batch = {
        "returns": torch.tensor([0.0]),
        "loss_mask": torch.tensor([True]),
    }
    local_values = torch.tensor([0.2])
    remote_returns = torch.tensor([1.0, 2.0, 4.0])
    remote_values = torch.tensor([0.8, 2.5, 3.0])
    remote_statistics = compute_explained_variance_statistics(
        remote_returns, remote_values
    )

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(statistics, op):
        assert op == torch.distributed.ReduceOp.SUM
        statistics.add_(remote_statistics.to(statistics.device))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    metrics = actor.compute_global_explained_variance_metrics(local_values)

    all_returns = torch.cat((actor.rollout_batch["returns"], remote_returns)).double()
    all_values = torch.cat((local_values, remote_values)).double()
    expected = 1.0 - torch.var(all_returns - all_values) / torch.var(all_returns)
    assert metrics["critic/explained_variance"] == pytest.approx(expected.item())
    assert metrics["critic/explained_variance_valid"] == 1.0
    assert metrics["critic/explained_variance_numel"] == 4.0


def test_global_statistics_preserve_non_finite_input_reason():
    statistics = compute_explained_variance_statistics(
        returns=torch.tensor([0.0, float("nan"), 2.0]),
        values=torch.tensor([0.1, 0.2, 1.8]),
    )
    metrics = explained_variance_metrics_from_statistics(statistics)

    assert torch.isnan(metrics["critic/explained_variance"])
    assert metrics["critic/explained_variance_valid"].item() == 0.0
    assert metrics["critic/explained_variance_numel"].item() == 3.0
    assert metrics["critic/explained_variance_invalid_reason"].item() == 3.0
