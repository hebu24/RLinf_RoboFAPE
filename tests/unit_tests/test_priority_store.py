"""Unit tests for PriorityStore.peek_topn / take_topn / discard_topn.

These three methods back the chunk_mask readiness collective: peek inspects
candidates across ranks, take consumes a batch (remove + mark used), and
discard drops stale candidates in lockstep. The key invariant they share with
``remove_below`` is the ``_discarded_unused`` counter: an item dropped without
ever being taken (used) increments it; a taken item does not.
"""

import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.priority_store import PriorityStore


def _traj(v: int) -> Trajectory:
    """A minimal trajectory carrying a single policy version ``v``."""
    return Trajectory(versions=torch.tensor([float(v)], dtype=torch.float32))


def _store_with(*versions: int) -> PriorityStore:
    store = PriorityStore(maxsize=len(versions) + 2)
    for v in versions:
        store.add((float(v), float(v)), _traj(v))
    return store


def _versions(trajs):
    return [int(t.versions.reshape(-1)[0].item()) for t in trajs]


def test_peek_topn_does_not_mark_or_remove():
    store = _store_with(1, 2, 3)
    seen = store.peek_topn(2)
    assert _versions(seen) == [3, 2]  # highest priority first
    assert len(store) == 3  # peek did not remove
    # Peek did not mark anything used -> evicting everything counts all as discarded-unused.
    store.remove_below(999)
    assert len(store) == 0
    assert store.get_metric()["discarded_unused"] == 3


def test_take_topn_removes_and_marks_used():
    store = _store_with(1, 2, 3)
    taken = store.take_topn(2)
    assert _versions(taken) == [3, 2]  # highest priority first
    assert len(store) == 1  # taken items removed
    # Remaining item (v=1) was not taken -> not marked used.
    store.remove_below(999)
    assert len(store) == 0
    assert store.get_metric()["discarded_unused"] == 1


def test_discard_topn_removes_without_marking():
    store = _store_with(1, 2, 3)
    store.discard_topn(2)
    assert len(store) == 1  # discarded items removed
    assert store.get_metric()["discarded_unused"] == 2  # neither was used
    # The survivor (v=1) can still be taken normally.
    taken = store.take_topn(1)
    assert _versions(taken) == [1]
    assert len(store) == 0
    # Taking marks used; discarding nothing afterwards keeps the counter stable.
    assert store.get_metric()["discarded_unused"] == 2


def test_peek_discard_peek_sequence():
    store = _store_with(1, 2, 3)
    assert _versions(store.peek_topn(2)) == [3, 2]
    store.discard_topn(2)  # drops v=3, v=2
    assert _versions(store.peek_topn(1)) == [1]
    assert len(store) == 1


def test_empty_store_peek_take_discard():
    store = PriorityStore(maxsize=2)
    assert store.peek_topn(3) == []
    assert store.take_topn(3) == []
    store.discard_topn(3)  # no-op, must not raise
    assert len(store) == 0
    assert store.get_metric()["discarded_unused"] == 0


def test_n_greater_than_len_returns_all():
    store = _store_with(1, 2)
    assert _versions(store.peek_topn(5)) == [2, 1]
    taken = store.take_topn(5)
    assert _versions(taken) == [2, 1]
    assert len(store) == 0


def test_take_topn_returns_highest_priority_first():
    store = _store_with(1, 5, 3)
    assert _versions(store.take_topn(2)) == [5, 3]
    assert _versions(store.peek_topn(1)) == [1]


def test_get_metric_after_take_reflects_remaining_only():
    store = _store_with(1, 2, 3)
    store.take_topn(2)  # removes v=3, v=2
    metric = store.get_metric()
    assert metric["discarded_unused"] == 0
    assert 1 in metric and metric[1]["ratio"] == 1.0
    assert 2 not in metric and 3 not in metric
