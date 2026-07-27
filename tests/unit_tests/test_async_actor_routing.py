"""Unit tests for deterministic async-actor keyed routing (Change 4).

Validates ``CommMapper.get_dst_ranks`` / ``build_channel_key`` for the
``"async_actor"`` tag used by ``EnvWorker._init_async_actor_params``: every env
worker shards its batch across actor ranks so each actor rank receives a
dedicated, balanced queue (no DEFAULT-queue competition that starves ranks).

Imports only ``rlinf.scheduler`` to avoid the pre-existing
``rlinf.utils.utils`` -> ``metric_utils`` -> ``algorithms`` circular import that
makes modules pulling ``MultiStepRolloutWorker`` uncollectable in this worktree.
"""

from rlinf.scheduler import CommMapper


def _dst_balance(batch_size: int, src_ws: int, dst_ws: int) -> dict:
    """Total items received per dst rank, summed across all src ranks."""
    per_dst = {r: 0 for r in range(dst_ws)}
    for src in range(src_ws):
        for dst, size in CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=src_ws,
            dst_world_size=dst_ws,
            src_rank=src,
        ):
            per_dst[dst] += size
    return per_dst


def test_async_actor_channel_key_is_per_rank_and_stable():
    # Each actor rank reads its own dedicated queue; keys must be distinct & stable.
    keys = [CommMapper.build_channel_key(r, r, "async_actor") for r in range(4)]
    assert keys == [f"{r}_{r}_async_actor" for r in range(4)]
    assert len(set(keys)) == 4


def test_async_actor_routing_4_to_4_one_shard_per_env_worker():
    # 4 env workers -> 4 actor ranks: env rank r sends all 4 of its envs to actor r.
    for src in range(4):
        assert CommMapper.get_dst_ranks(16, 4, 4, src) == [(src, 4)]
    assert _dst_balance(16, 4, 4) == {r: 4 for r in range(4)}


def test_async_actor_routing_2_to_4_each_actor_balanced():
    # 2 env workers feed 4 actor ranks: each actor still receives 4 envs total.
    assert CommMapper.get_dst_ranks(16, 2, 4, 0) == [(0, 4), (1, 4)]
    assert CommMapper.get_dst_ranks(16, 2, 4, 1) == [(2, 4), (3, 4)]
    assert _dst_balance(16, 2, 4) == {r: 4 for r in range(4)}


def test_async_actor_routing_4_to_2_each_actor_balanced():
    # 4 env workers feed 2 actor ranks: each actor receives 8 envs total.
    assert CommMapper.get_dst_ranks(16, 4, 2, 0) == [(0, 4)]
    assert CommMapper.get_dst_ranks(16, 4, 2, 1) == [(0, 4)]
    assert CommMapper.get_dst_ranks(16, 4, 2, 2) == [(1, 4)]
    assert CommMapper.get_dst_ranks(16, 4, 2, 3) == [(1, 4)]
    assert _dst_balance(16, 4, 2) == {0: 8, 1: 8}


def test_async_actor_keys_cover_all_actor_ranks_distinctly():
    # The env worker builds one key per actor rank it shards to; across the full
    # dst world every actor rank must be reachable by exactly one key.
    for src in range(4):
        assert {r for r, _ in CommMapper.get_dst_ranks(16, 4, 4, src)} == {src}
    assert (
        len(
            {
                CommMapper.build_channel_key(r, r, "async_actor")
                for r in range(4)
            }
        )
        == 4
    )
