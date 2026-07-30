import torch

from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor


class _DisabledGradScaler:
    def scale(self, loss):
        return loss

    def get_scale(self):
        return 1.0


def test_sampled_gradient_measurement_does_not_change_actual_gradient():
    model = torch.nn.Linear(2, 1, bias=False)
    reference = torch.nn.Linear(2, 1, bias=False)
    reference.load_state_dict(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    actor = object.__new__(AsyncPPOEmbodiedFSDPActor)
    actor.model = model
    actor.optimizer = optimizer
    actor.grad_scaler = _DisabledGradScaler()
    actor.device = torch.device("cpu")

    inputs = torch.tensor([[1.0, -2.0]])
    output = model(inputs)
    actor_loss = output.square().mean()
    scaled_critic_loss = 0.1 * (output - 1.0).square().mean()
    combined_loss = actor_loss + scaled_critic_loss

    assert actor._measure_global_grad_norm(actor_loss) > 0
    assert actor._measure_global_grad_norm(scaled_critic_loss) > 0
    assert actor._measure_global_grad_norm(combined_loss) > 0
    combined_loss.backward()
    measured_gradient = model.weight.grad.detach().clone()

    reference_output = reference(inputs)
    reference_loss = reference_output.square().mean() + 0.1 * (
        reference_output - 1.0
    ).square().mean()
    reference_loss.backward()

    torch.testing.assert_close(measured_gradient, reference.weight.grad)
