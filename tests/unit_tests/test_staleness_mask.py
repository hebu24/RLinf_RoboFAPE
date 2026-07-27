"""Unit tests for chunk-level staleness masking (Change 1).

Tests ``rlinf.data.staleness_mask.compute_staleness_mask`` (pure function) +
``aggregate_embodied_chunk_rewards`` so the two-mask decoupling is validated
without instantiating the FSDP actor (which pulls the heavy ``utils.utils``
import graph). Imports only clean roots (``rlinf.data.*``, ``rlinf.algorithms.utils``).
"""

import torch

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.algorithms.utils import (
    aggregate_embodied_chunk_rewards,
    preprocess_embodied_advantages_inputs,
)
from rlinf.data.staleness_mask import compute_staleness_mask


def _batch(versions, loss_mask):
    """Build a minimal rollout_batch dict with versions [n,B,1] and a low-level mask."""
    return {
        "versions": torch.tensor(versions, dtype=torch.float32).view(
            len(versions), 1, 1
        ),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.bool).view(
            len(loss_mask), 1, 1
        ),
    }


def test_stale_prefix_masked_fresh_suffix_kept():
    # versions [8, 8, 9, 10], actor_version=10, threshold=1 -> cutoff=9.
    batch = _batch([8, 8, 9, 10], [True, True, True, True])
    diag = compute_staleness_mask(batch, actor_version=10, staleness_threshold=1)
    # chunks 0,1 (v=8) are stale; chunks 2,3 (v=9,10) are fresh.
    eff = batch["loss_mask"].view(-1).tolist()
    assert eff == [False, False, True, True]
    assert diag["staleness_masked_chunks"] == 2
    assert diag["staleness_effective_chunks"] == 2
    assert diag["staleness_masked_fraction"] == 0.5
    # version range over ALL trainable positions (stale prefix included).
    assert diag["staleness_version_min"] == 8
    assert diag["staleness_version_max"] == 10
    assert diag["staleness_version_mean"] == 8.75  # mean of [8,8,9,10]


def test_pickup_excluded_from_freshness_stats():
    # chunk 0 is pickup: version 0 but loss_mask=False -> never trainable, so it
    # neither counts as masked nor enters version statistics.
    batch = _batch([0, 9, 10], [False, True, True])
    diag = compute_staleness_mask(batch, actor_version=10, staleness_threshold=1)
    assert batch["loss_mask"].view(-1).tolist() == [False, True, True]
    assert diag["staleness_masked_chunks"] == 0  # the two trainable chunks are fresh
    assert diag["staleness_effective_chunks"] == 2
    assert diag["staleness_version_min"] == 9
    assert diag["staleness_version_max"] == 10


def test_effective_mask_shapes_low_level_vs_chunk():
    batch = {
        "versions": torch.tensor([8.0, 9.0, 10.0]).view(3, 1, 1),
        "loss_mask": torch.ones((3, 1, 4), dtype=torch.bool),  # na=4
    }
    compute_staleness_mask(batch, actor_version=10, staleness_threshold=1)
    assert batch["loss_mask"].shape == (3, 1, 4)  # effective low-level
    assert batch["staleness_chunk_loss_mask"].shape == (3, 1, 1)  # chunk-level
    # staleness_chunk_loss_mask = effective_low.any(-1)
    assert torch.equal(
        batch["staleness_chunk_loss_mask"],
        batch["loss_mask"].any(dim=-1, keepdim=True),
    )


def test_loss_mask_sum_recomputed_from_effective():
    batch = _batch([8, 8, 9, 10], [True, True, True, True])
    compute_staleness_mask(batch, actor_version=10, staleness_threshold=1)
    # 2 effective low-level chunks -> sum = 2 (expanded over all positions).
    assert int(batch["loss_mask_sum"][0, 0, 0].item()) == 2
    assert int(batch["staleness_chunk_loss_mask_sum"][0, 0, 0].item()) == 2


def test_missing_inputs_is_noop():
    # No versions / no threshold -> nothing changes, empty diagnostics.
    batch = {"loss_mask": torch.ones((2, 1, 1), dtype=torch.bool)}
    assert compute_staleness_mask(batch, 10, None) == {}
    batch2 = {"versions": torch.tensor([9.0]).view(1, 1, 1)}
    assert compute_staleness_mask(batch2, 10, 1) == {}


def test_reward_aggregation_uses_effective_low_level_mask():
    # After compute_staleness_mask, aggregate_embodied_chunk_rewards must mask at
    # the low-level (per action-chunk) resolution before summing over the chunk.
    # 1 chunk, 1 batch, 3 action-chunks; rewards all 1.0; effective mask [T,T,F].
    rewards = torch.ones((1, 1, 3))
    eff_low = torch.tensor([True, True, False]).view(1, 1, 3)
    out = aggregate_embodied_chunk_rewards(
        rewards, eff_low, gamma=1.0, aggregation="sum"
    )
    assert out.shape == (1, 1, 1)
    assert float(out.sum().item()) == 2.0  # only the two unmasked positions


def test_chunk_mask_reduces_to_any_over_action_chunks():
    # policy_loss receives the chunk mask; it must equal effective_low.any(-1).
    batch = {
        "versions": torch.tensor([10.0, 10.0]).view(2, 1, 1),
        "loss_mask": torch.tensor(
            [[True, True, False], [True, False, False]]
        ).view(2, 1, 3),  # chunk 0 has 2 valid, chunk 1 has 1 valid
    }
    compute_staleness_mask(batch, actor_version=10, staleness_threshold=1)
    chunk_mask = batch["staleness_chunk_loss_mask"].view(-1).tolist()
    assert chunk_mask == [True, True]  # both chunks have >=1 fresh valid position


def test_gae_traverses_full_episode_fresh_suffix_gets_terminal_return():
    """Plan T4: GAE must traverse the WHOLE terminal episode (stale prefix
    included) so the fresh suffix's return reflects the true terminal outcome,
    while the stale prefix is masked out of reward aggregation (=> 0 reward) and
    out of the loss (chunk mask).

    Setup: versions [8,8,9,10] / actor 10 / threshold 1 -> effective mask
    [F,F,T,T]; rewards all 1.0 -> aggregated (masked) rewards [0,0,1,1];
    terminal done only at the end; values 0 (no bootstrap). With gamma=1 and
    gae_lambda=1, G_t = sum of masked future rewards -> [2,2,2,1].
    """
    n_chunk, na = 4, 1
    batch = {
        "versions": torch.tensor([8.0, 8.0, 9.0, 10.0]).view(n_chunk, 1, na),
        "loss_mask": torch.ones((n_chunk, 1, na), dtype=torch.bool),
    }
    compute_staleness_mask(batch, actor_version=10, staleness_threshold=1)
    # effective masks: stale prefix [F,F], fresh suffix [T,T]
    assert batch["loss_mask"].view(-1).tolist() == [False, False, True, True]

    rewards = torch.ones((n_chunk, 1, na))              # r = 1 everywhere
    dones = torch.zeros((n_chunk + 1, 1, na), dtype=torch.bool)
    dones[-1] = True                                     # terminal done at the end
    values = torch.zeros((n_chunk + 1, 1, 1))           # no bootstrap value

    processed = preprocess_embodied_advantages_inputs(
        rewards=rewards,
        dones=dones,
        values=values,
        loss_mask=batch["loss_mask"],                    # effective low-level mask
        reward_type="chunk_level",
        adv_type="gae",
        gamma=1.0,
        chunk_reward_aggregation="discounted_sum",
        group_size=1,
    )
    _, returns = compute_gae_advantages_and_returns(
        rewards=processed["rewards"],
        dones=processed["dones"],
        values=processed["values"],
        loss_mask=processed["loss_mask"],
        gamma=1.0,
        gae_lambda=1.0,
        normalize_advantages=False,
    )
    # gamma=1, lambda=1, values=0 -> G_t = sum of (masked) future rewards.
    # masked rewards: [0,0,1,1]; terminal done at end -> returns = [2,2,2,1].
    torch.testing.assert_close(returns.view(-1), torch.tensor([2.0, 2.0, 2.0, 1.0]))

    # The fresh suffix carries the correct terminal return: last chunk = its
    # own reward (no future bootstrap), second-to-last = its reward + last.
    assert float(returns[-1].item()) == 1.0
    assert float(returns[-2].item()) == 2.0
    # The stale prefix's return is NONZERO: GAE traversed across the staleness
    # boundary (did not truncate the episode at the first stale chunk) -- the
    # prefix simply propagated the fresh suffix's return through zeroed rewards.
    # The prefix is excluded from LOSS via staleness_chunk_loss_mask, not from GAE.
    assert float(returns[0].item()) == 2.0
    assert not batch["staleness_chunk_loss_mask"].view(-1)[:2].any()


def test_inconsistent_action_versions_within_chunk_raises():
    # Section 1.5: if versions is ever per-action [n_chunk, B, na>1], the action
    # positions within a chunk-step must all share one version. A chunk carrying
    # two different versions means it was produced by two policies (corruption).
    batch = {
        "versions": torch.tensor([[8.0, 9.0], [10.0, 10.0]]).view(2, 1, 2),  # na=2
        "loss_mask": torch.ones((2, 1, 2), dtype=torch.bool),
    }
    import pytest

    with pytest.raises(ValueError, match="Inconsistent action versions"):
        compute_staleness_mask(batch, actor_version=10, staleness_threshold=1)


def test_consistent_per_action_versions_does_not_raise():
    # Same per-action shape but both action positions agree per chunk-step: OK.
    batch = {
        "versions": torch.tensor([[8.0, 8.0], [10.0, 10.0]]).view(2, 1, 2),  # na=2
        "loss_mask": torch.ones((2, 1, 2), dtype=torch.bool),
    }
    diag = compute_staleness_mask(batch, actor_version=10, staleness_threshold=1)
    # chunk 0 (v=8) stale, chunk 1 (v=10) fresh -> effective [F, T] on both actions
    assert batch["loss_mask"].view(-1).tolist() == [False, False, True, True]
    assert diag["staleness_effective_chunks"] == 1


def test_4d_versions_with_mismatched_loss_mask_trailing_dims():
    # The real rollout carries versions [n,B,na,n_env] whose trailing dims do NOT
    # match loss_mask's [n,B,na,na]; a direct ``trainable & (versions>=cutoff)``
    # crashes with "size of tensor a (na) must match b (n_env) at dim 3". The
    # per-chunk-step reduction must broadcast freshness cleanly across mismatched
    # trailing ranks.
    n, B, na, ne = 4, 1, 2, 3
    per_chunk = torch.tensor([8.0, 8.0, 9.0, 10.0]).view(n, B, 1, 1)
    versions = per_chunk.expand(n, B, na, ne).contiguous()  # [n,B,2,3]
    loss_mask = torch.ones((n, B, na, na), dtype=torch.bool)  # [n,B,2,2] -- differ
    batch = {"versions": versions, "loss_mask": loss_mask}
    diag = compute_staleness_mask(batch, actor_version=10, staleness_threshold=1)
    # effective low-level mask keeps loss_mask shape; stale prefix masked.
    assert batch["loss_mask"].shape == (n, B, na, na)
    expected = torch.zeros((n, B, na, na), dtype=torch.bool)
    expected[2:] = True
    assert torch.equal(batch["loss_mask"], expected)
    assert batch["staleness_chunk_loss_mask"].shape == (n, B, 1)
    assert batch["staleness_chunk_loss_mask"].view(-1).tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert diag["staleness_effective_chunks"] == 2
    assert diag["staleness_masked_chunks"] == 2
    assert diag["staleness_version_min"] == 8
    assert diag["staleness_version_max"] == 10


