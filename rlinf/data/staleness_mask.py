# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chunk-level staleness masking for async PPO completed episodes.

A completed episode may span multiple policy versions. The legacy behavior drops
the WHOLE trajectory when ``versions.min()`` is stale, which starves actor ranks
and deadlocks the FSDP collective. Instead, keep the whole episode (needed for
Robometer reward reconstruction + GAE) and only mask stale CHUNKS out of the
loss / reward aggregation.

This module is a pure function over a ``rollout_batch`` dict so it can be unit
tested without instantiating the FSDP actor (which pulls the heavy
``rlinf.utils.utils`` import graph).
"""

from typing import Any

import torch


def candidate_batch_is_usable(stats: dict, min_fresh_chunks: int) -> bool:
    """Return whether a candidate batch can safely enter the actor collective.

    An all-masked batch is an intentional no-op (for example, a skipped
    zero-success window), not stale data. A batch with trainable chunks is usable
    only when it contains the configured minimum number of fresh chunks.
    """
    return bool(
        int(stats.get("trainable", 0)) == 0
        or int(stats.get("fresh", 0)) >= int(min_fresh_chunks)
    )


def compute_staleness_mask(
    rollout_batch: dict[str, Any],
    actor_version: int,
    staleness_threshold: int | None,
) -> dict[str, float | int]:
    """AND ``loss_mask`` with a per-chunk-step freshness mask, in place.

    A chunk is fresh iff ``version >= actor_version - staleness_threshold``.
    Pickup / padding / already-masked positions (``loss_mask=False``) are
    excluded from version statistics and stay masked. Produces:

    - ``loss_mask``                     = effective low-level  ``[n_chunk, B, na]``
    - ``staleness_chunk_loss_mask``     = effective chunk      ``[n_chunk, B, 1]``
    - ``loss_mask_sum``                 = recomputed low-level count
    - ``staleness_chunk_loss_mask_sum`` = recomputed chunk count

    Returns staleness diagnostics (chunk counts + version range over trainable
    positions). No-op (returns ``{}``) if versions / loss_mask / threshold absent.
    """
    versions = rollout_batch.get("versions", None)
    loss_mask = rollout_batch.get("loss_mask", None)
    if versions is None or loss_mask is None or staleness_threshold is None:
        return {}
    if versions.ndim < 2 or loss_mask.ndim < 2:
        return {}

    cutoff = int(actor_version) - int(staleness_threshold)
    trainable = loss_mask.to(torch.bool)
    # ``versions`` is [n_chunk, B, ...] where dims 0,1 are (chunk-step, batch) and
    # every trailing dim encodes action/extra positions that must share ONE policy
    # version per chunk-step. Reduce to a per-chunk-step version (min over the
    # trailing dims) so freshness broadcasts against the loss_mask's trailing dims
    # regardless of their concrete rank (3D [n,B,na] or 4D [n,B,na,*]) -- the real
    # rollout carries versions [n,B,na,n_env] whose trailing dims do NOT match
    # loss_mask's, so a direct ``trainable & (versions >= cutoff)`` would crash.
    v_flat = versions.reshape(versions.shape[0], versions.shape[1], -1)  # [n, B, P]
    v_min_step = v_flat.amin(dim=-1, keepdim=True)  # [n, B, 1]
    v_max_step = v_flat.amax(dim=-1, keepdim=True)  # [n, B, 1]
    # Section 1.5: a policy chunk carries ONE version shared across all of its
    # action positions. If the per-chunk-step min != max, the chunk-step was
    # produced by two different policies (data corruption).
    if not torch.equal(v_min_step, v_max_step):
        raise ValueError(
            "Inconsistent action versions within a policy chunk: a chunk-step "
            "carries different versions across its action positions, which "
            "indicates data corruption (one chunk produced by two policies)."
        )
    freshness = v_min_step.to(torch.int64) >= cutoff  # [n, B, 1]
    # Broadcast freshness against the loss_mask's trailing dims (which may differ
    # from versions' trailing dims). Reshape to [n, B, 1, 1, ...] matching the
    # loss_mask's trailing rank so the AND broadcasts cleanly.
    fresh_bc = freshness.reshape(
        freshness.shape[0], freshness.shape[1], *([1] * (loss_mask.ndim - 2))
    )
    effective_low = trainable & fresh_bc  # loss_mask shape
    effective_chunk = effective_low.reshape(
        effective_low.shape[0], effective_low.shape[1], -1
    ).any(dim=-1, keepdim=True)  # [n, B, 1]

    rollout_batch["loss_mask"] = effective_low
    rollout_batch["staleness_chunk_loss_mask"] = effective_chunk
    # Per-batch count of effective LOW-LEVEL positions (sum over chunk-step +
    # trailing, keep batch). Both loss_mask_sum and staleness_chunk_loss_mask_sum
    # use this SAME low-level count: masked_mean_ratio normalizes the loss by it,
    # and using the chunk count (effective_chunk.sum) here inflated value_loss by
    # ~na (the per-chunk action factor) vs trajectory mode. staleness_chunk_loss_mask_sum
    # keeps the chunk-level [n,B,1] shape to match the chunk-level loss_mask.
    sum_dims = [0] + list(range(2, effective_low.ndim))
    low_level_count = effective_low.sum(dim=sum_dims, keepdim=True)  # [1,B,1,...]
    rollout_batch["loss_mask_sum"] = low_level_count.expand_as(effective_low)
    rollout_batch["staleness_chunk_loss_mask_sum"] = low_level_count.reshape(
        1, -1, 1
    ).expand_as(effective_chunk)

    # Version statistics over trainable chunk-steps only (pickup / padding with
    # loss_mask=False are excluded so they do not skew the range).
    step_trainable = trainable.reshape(
        trainable.shape[0], trainable.shape[1], -1
    ).any(dim=-1)  # [n, B]
    per_step_version = v_min_step.squeeze(-1).to(torch.int64)  # [n, B]
    active_versions = per_step_version[step_trainable]

    original_chunks = int(step_trainable.sum().item())
    fresh_chunks = int(effective_chunk.sum().item())
    masked_chunks = original_chunks - fresh_chunks
    if active_versions.numel() > 0:
        v_min = int(active_versions.min().item())
        v_mean = float(active_versions.float().mean().item())
        v_max = int(active_versions.max().item())
    else:
        v_min = v_mean = v_max = 0

    return {
        "staleness_masked_chunks": masked_chunks,
        "staleness_masked_fraction": (
            masked_chunks / original_chunks if original_chunks else 0.0
        ),
        "staleness_effective_chunks": fresh_chunks,
        "staleness_version_min": v_min,
        "staleness_version_mean": v_mean,
        "staleness_version_max": v_max,
    }


def count_fresh_chunks(candidates, cutoff: int) -> dict:
    """Freshness statistics over a list of candidate ``Trajectory`` objects.

    Mirrors ``compute_staleness_mask`` but operates on pre-batch Trajectory
    objects (used by the readiness collective phase 2 to decide whether this
    rank has >=1 fresh chunk without yet assembling the training batch).

    Returns ``{fresh, trainable, stale, version_min, version_mean, version_max}``
    where chunk counts are over chunk-steps (``.any(dim=-1)``) and version range
    is over trainable chunk-steps only (pickup/padding excluded).
    """
    fresh = 0
    trainable_total = 0
    v_min = 0
    v_max = 0
    v_sum = 0.0
    v_count = 0
    for traj in candidates:
        versions = getattr(traj, "versions", None)
        loss_mask = getattr(traj, "loss_mask", None)
        if versions is None or loss_mask is None or versions.ndim < 2:
            continue
        trainable = loss_mask.to(torch.bool)
        # Per-chunk-step version (dims 0,1 = n_chunk, B; trailing = action/extra),
        # reduced by min so it matches compute_staleness_mask's freshness logic and
        # is robust to versions [n,B,na,n_env] whose trailing dims != loss_mask's.
        v_flat = versions.reshape(versions.shape[0], versions.shape[1], -1)
        v_min_step = v_flat.amin(dim=-1)  # [n, B]
        step_trainable = trainable.reshape(
            trainable.shape[0], trainable.shape[1], -1
        ).any(dim=-1)  # [n, B]
        trainable_total += int(step_trainable.sum().item())
        # A chunk-step is fresh iff it is trainable AND its version >= cutoff.
        eff_step = step_trainable & (v_min_step.to(torch.int64) >= cutoff)  # [n, B]
        fresh += int(eff_step.sum().item())
        active = v_min_step[step_trainable]
        if active.numel() > 0:
            a_min = int(active.min().item())
            a_max = int(active.max().item())
            v_min = a_min if v_min == 0 else min(v_min, a_min)
            v_max = max(v_max, a_max)
            v_sum += float(active.float().sum().item())
            v_count += int(active.numel())
    return {
        "fresh": fresh,
        "trainable": trainable_total,
        "stale": trainable_total - fresh,
        "version_min": v_min,
        "version_mean": (v_sum / v_count) if v_count else 0.0,
        "version_max": v_max,
    }
