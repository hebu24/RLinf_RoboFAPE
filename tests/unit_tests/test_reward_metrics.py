import torch

from rlinf.utils import metric_utils


def test_distributed_reward_stats_reduce_all_ranks(monkeypatch):
    from rlinf.scheduler.worker.worker import Worker

    monkeypatch.setattr(
        Worker.torch_platform, "current_device", lambda: torch.device("cpu")
    )
    calls = 0

    def fake_all_reduce(tensor, op):
        nonlocal calls
        if calls == 0:
            tensor += torch.tensor([1.0, 2.0, 1.0])
        elif calls == 1:
            tensor.copy_(torch.maximum(tensor, torch.tensor([4.0, 5.0])))
        calls += 1

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    stats = metric_utils._distributed_reward_stats(torch.tensor([-2.0, 1.0]))

    assert calls == 2
    assert stats == {
        "mean": 0.0,
        "min": -4.0,
        "max": 5.0,
        "positive_fraction": 0.5,
    }
