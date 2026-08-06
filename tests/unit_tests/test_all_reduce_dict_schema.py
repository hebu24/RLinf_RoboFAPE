import pytest
import torch

from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict


class _CPUPlatform:
    @staticmethod
    def current_device():
        return torch.device("cpu")


def test_all_reduce_dict_schema_mismatch_fails_before_value_collective(monkeypatch):
    monkeypatch.setattr(Worker, "torch_platform", _CPUPlatform)
    calls = []

    def fake_all_reduce(tensor, op, group=None):
        calls.append(op)
        if op == torch.distributed.ReduceOp.MAX:
            tensor[0] += 2

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    with pytest.raises(RuntimeError, match="metric schema differs across ranks"):
        all_reduce_dict({"metric/a": 1.0}, validate_schema=True)

    assert calls == [
        torch.distributed.ReduceOp.MIN,
        torch.distributed.ReduceOp.MAX,
    ]


def test_all_reduce_dict_matching_schema_reduces_values(monkeypatch):
    monkeypatch.setattr(Worker, "torch_platform", _CPUPlatform)
    calls = []

    def fake_all_reduce(tensor, op, group=None):
        calls.append(op)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    result = all_reduce_dict(
        {"metric/b": 2.0, "metric/a": 1.0},
        op=torch.distributed.ReduceOp.AVG,
        validate_schema=True,
    )

    assert result == {"metric/a": 1.0, "metric/b": 2.0}
    assert calls == [
        torch.distributed.ReduceOp.MIN,
        torch.distributed.ReduceOp.MAX,
        torch.distributed.ReduceOp.AVG,
    ]
