import torch

from rlinf.algorithms.losses import compute_ppo_critic_loss


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
