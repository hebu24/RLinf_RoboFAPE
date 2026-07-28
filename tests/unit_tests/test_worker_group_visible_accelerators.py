from rlinf.scheduler.hardware import AcceleratorType
from rlinf.scheduler.worker.worker_group import _resolve_worker_visible_accelerators


def test_resolve_worker_visible_accelerators_keeps_physical_mapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert _resolve_worker_visible_accelerators(
        AcceleratorType.NV_GPU, ["0", "1", "3"]
    ) == ["4", "5", "7"]


def test_resolve_worker_visible_accelerators_falls_back_without_outer_mask(
    monkeypatch,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _resolve_worker_visible_accelerators(
        AcceleratorType.NV_GPU, ["0", "1"]
    ) == ["0", "1"]


def test_resolve_worker_visible_accelerators_preserves_out_of_range_ids(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert _resolve_worker_visible_accelerators(
        AcceleratorType.NV_GPU, ["5"]
    ) == ["5"]


def test_resolve_worker_visible_accelerators_clamps_node_level_workers_to_outer_mask(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert _resolve_worker_visible_accelerators(
        AcceleratorType.NV_GPU, ["0", "1", "2", "3", "4", "5", "6", "7"]
    ) == ["2", "3"]
