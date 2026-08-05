import asyncio

from rlinf.workers.actor.async_ppo_fsdp_worker import AsyncPPOEmbodiedFSDPActor


def test_actor_returns_confirmed_credit_to_every_rollout_rank():
    awaited = []
    puts = []

    class _Work:
        def __init__(self, key):
            self.key = key

        async def async_wait(self):
            awaited.append(self.key)

    class _Channel:
        def put(self, item, *, key, async_op):
            puts.append((item, key, async_op))
            return _Work(key)

    class _Placement:
        def get_world_size(self, component):
            assert component == "rollout"
            return 2

    actor = object.__new__(AsyncPPOEmbodiedFSDPActor)
    actor._rank = 0
    actor._rollout_credit_channel = _Channel()
    actor._component_placement = _Placement()
    actor.log_info = lambda *_args, **_kwargs: None

    asyncio.run(actor._return_rollout_credits(2, reason="stale_discard"))

    assert puts == [
        (2, "0_0_rollout_credit", True),
        (2, "1_1_rollout_credit", True),
    ]
    assert awaited == ["0_0_rollout_credit", "1_1_rollout_credit"]


def test_nonzero_actor_rank_does_not_duplicate_global_credits():
    class _Channel:
        def put(self, *_args, **_kwargs):
            raise AssertionError("nonzero actor rank must not return global credit")

    actor = object.__new__(AsyncPPOEmbodiedFSDPActor)
    actor._rank = 1
    actor._rollout_credit_channel = _Channel()

    asyncio.run(actor._return_rollout_credits(2, reason="consumed"))
