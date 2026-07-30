import pytest
import torch

from rlinf.algorithms import losses


@pytest.mark.parametrize(
    "loss_fn_name,actor_fn_name",
    [
        ("compute_decoupled_ppo_actor_critic_loss", "compute_decoupled_ppo_actor_loss"),
        ("compute_ppo_actor_critic_loss", "compute_ppo_actor_loss"),
    ],
)
def test_actor_critic_loss_is_unscaled_sum(monkeypatch, loss_fn_name, actor_fn_name):
    actor_loss = torch.tensor(2.0, requires_grad=True)
    critic_loss = torch.tensor(5.0, requires_grad=True)
    monkeypatch.setattr(
        losses, actor_fn_name, lambda **kwargs: (actor_loss, {"actor/test": actor_loss})
    )
    monkeypatch.setattr(
        losses,
        "compute_ppo_critic_loss",
        lambda **kwargs: (critic_loss, {"critic/value_loss": critic_loss}),
    )
    components = {}

    total, metrics = getattr(losses, loss_fn_name)(loss_components=components)

    assert total.item() == pytest.approx(7.0)
    assert metrics["critic/value_loss"].item() == pytest.approx(5.0)
    assert "critic/scaled_value_loss" not in metrics
    assert "critic/value_loss_coef" not in metrics
    assert components == {"actor_loss": actor_loss, "critic_loss": critic_loss}
