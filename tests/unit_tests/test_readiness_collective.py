"""Unit tests for the readiness collective's pure building blocks (Change 2).

The multi-rank ``all_reduce(MIN/MAX)`` collective itself is validated by the
4-GPU smoke (it needs a real process group). These tests cover the pure,
importable pieces that drive the collective's decisions:

- ``count_fresh_chunks``: phase-2 freshness counting over Trajectory candidates
- ``PriorityStore.peek/take/discard``: the lockstep peek -> discard -> take
  sequence the collective drives on every rank
"""

import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.priority_store import PriorityStore
from rlinf.data.staleness_mask import count_fresh_chunks


def _traj(versions_1d, loss_mask_1d):
    """A Trajectory with versions [T,1,1] and a low-level mask [T,1,1]."""
    t = len(versions_1d)
    return Trajectory(
        versions=torch.tensor(versions_1d, dtype=torch.float32).view(t, 1, 1),
        loss_mask=torch.tensor(loss_mask_1d, dtype=torch.bool).view(t, 1, 1),
    )


def test_count_fresh_chunks_all_fresh():
    traj = _traj([9, 10, 10], [True, True, True])
    # actor_version=10, threshold=1 -> cutoff=9; all three are fresh.
    s = count_fresh_chunks([traj], cutoff=9)
    assert s["fresh"] == 3
    assert s["trainable"] == 3
    assert s["stale"] == 0
    assert s["version_min"] == 9
    assert s["version_max"] == 10


def test_count_fresh_chunks_stale_prefix():
    traj = _traj([8, 8, 9, 10], [True, True, True, True])
    s = count_fresh_chunks([traj], cutoff=9)
    assert s["fresh"] == 2  # only v=9,10
    assert s["stale"] == 2  # v=8,8
    assert s["trainable"] == 4
    assert s["version_min"] == 8
    assert s["version_max"] == 10


def test_count_fresh_chunks_pickup_excluded():
    # chunk 0: pickup (loss_mask False, version 0) -> excluded from trainable + stats.
    traj = _traj([0, 9, 10], [False, True, True])
    s = count_fresh_chunks([traj], cutoff=9)
    assert s["trainable"] == 2
    assert s["fresh"] == 2
    assert s["stale"] == 0
    assert s["version_min"] == 9  # pickup v=0 not counted
    assert s["version_max"] == 10


def test_count_fresh_chunks_empty_when_all_stale():
    traj = _traj([7, 7, 7], [True, True, True])
    s = count_fresh_chunks([traj], cutoff=9)
    assert s["fresh"] == 0  # phase 2 -> this rank has zero fresh -> all discard
    assert s["stale"] == 3


def test_collective_peek_discard_take_sequence_on_store():
    # Simulate the per-rank store sequence the collective drives: peek -> (all
    # stale on some rank) discard -> peek -> take. Validates the PriorityStore
    # primitives support the protocol without a real process group.
    store = PriorityStore(maxsize=4)
    # round 1: 3 stale candidates
    for v in (7, 7, 7):
        store.add((float(v), float(v)), _traj([v], [True]))
    cand = store.peek_topn(3)
    assert count_fresh_chunks(cand, cutoff=9)["fresh"] == 0  # all stale
    store.discard_topn(3)  # collective discards in lockstep
    assert len(store) == 0
    assert store.get_metric()["discarded_unused"] == 3

    # round 2: fresh candidates arrive
    for v in (9, 10):
        store.add((float(v), float(v)), _traj([v], [True]))
    cand = store.peek_topn(2)
    assert count_fresh_chunks(cand, cutoff=9)["fresh"] == 2  # all fresh
    taken = store.take_topn(2)  # collective takes in lockstep
    assert len(taken) == 2
    assert len(store) == 0
    # taken items are marked used -> not double-counted as discarded-unused
    assert store.get_metric()["discarded_unused"] == 3
