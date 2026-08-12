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

"""Robometer progress-estimator reward model (HTTP client to the robometer server).

Plugs into RLinf's ``history_buffer`` reward path: the env worker accumulates
the scene's human render-camera frames (``render_images`` -- one frame per
low-level env step) and, at trajectory done, sends them to this model (which
runs in the reward-worker Ray group). This model POSTs the per-env frame stacks
to a running robometer eval server (``POST /evaluate_batch_npy``), parses the
per-frame progress curve in ``[0, 1]``, and returns a per-frame reward tensor::

    reward[env, frame_i] = progress[i]        if the trajectory succeeded
                           progress[i] - 1.0  if it did not

``success`` must come from the env/eval side success signals; if it is missing,
we raise instead of falling back to robometer ``success_probs``. The env worker
later interpolates these down-sampled frame rewards back onto low-level
insertion steps, then aggregates them back to chunk rewards for chunk-level PPO.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.reward.base_reward_model import BaseRewardModel

# --- Robometer smoke capture (env-var gated; no-op when RLINF_ROBOMETER_SMOKE_DIR unset) ---
_SMOKE_DIR = os.environ.get("RLINF_ROBOMETER_SMOKE_DIR", "").strip() or None
_SMOKE_CAP = int(os.environ.get("RLINF_ROBOMETER_SMOKE_CAP", "10")) if _SMOKE_DIR else 0
_SMOKE_LOCAL_LOCK = threading.Lock()
_SMOKE_PREV = {}  # per-process: env_id -> {arr,prog,success,shift,len} for reset-detection

ROBOMETER_REWARD_PIPELINE_VERSION = "episode-v1"
_LOGGER = logging.getLogger(__name__)


def _robometer_downsample_indices(total: int, cap: int = 60):
    """Deterministic uniform down-sample indices over [0, total-1] to <= cap.

    Shared by ``RobometerHistoryRewardModel.compute_reward`` (down-samples the
    POSTed frames) AND ``env_worker.assign_history_reward`` (recomputes the SAME
    indices to map progress -> chunks + set loss_mask). MUST stay in sync: both
    call this with the SAME total (= history buffer length per env).
    """
    if total <= cap:
        return list(range(total))
    import numpy as np

    idx = np.linspace(0, total - 1, cap).astype(int)
    seen: set[int] = set()
    out: list[int] = []
    for i in idx:
        if int(i) not in seen:
            seen.add(int(i))
            out.append(int(i))
    return out


def _robometer_boundary_frame_indices(
    history_len: int, pickup_count: int, chunk_size: int
):
    """Deterministic chunk-boundary frame indices for delta shaping.

    Returns the frame index at the start of insertion and after each chunk:
    ``pickup_count + i * chunk_size`` for ``i = 0..n_chunks``, each clamped to
    ``[0, history_len - 1]``. ``n_chunks = ceil(insert_steps / chunk_size)`` where
    ``insert_steps = history_len - pickup_count``. The result always has exactly
    ``n_chunks + 1`` entries (duplicates from clamping are kept -- a repeated frame
    just yields a zero delta for that chunk, which is correct).

    Shared by ``RobometerHistoryRewardModel.compute_reward`` (selects the boundary
    frames to POST in delta mode) AND ``env_worker.assign_history_reward``
    (recomputes the SAME indices' count to map boundary progress -> chunks). MUST
    stay in sync, mirroring ``_robometer_downsample_indices`` for absolute mode.
    """
    if history_len <= 0:
        return []
    n_insert = max(0, history_len - pickup_count)
    if n_insert == 0 or chunk_size <= 0:
        return [max(0, min(pickup_count, history_len - 1))]
    n_chunks = (n_insert + chunk_size - 1) // chunk_size
    out: list[int] = []
    for i in range(n_chunks + 1):
        idx = pickup_count + i * chunk_size
        idx = max(0, min(idx, history_len - 1))
        out.append(idx)
    return out


def _extract_env_value(value: Any, env_id: int) -> Any:
    """Extract one env's entry from a possibly-batched success container."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, np.bool_, np.number)):
        return value
    try:
        env_value = value[env_id]
    except Exception:
        return None
    if isinstance(env_value, np.ndarray) and env_value.shape == ():
        return env_value.item()
    if hasattr(env_value, "item"):
        try:
            return env_value.item()
        except Exception:
            return env_value
    return env_value


def _resolve_env_success_from_infos(
    env_infos: dict[str, Any] | None, env_id: int
) -> bool:
    """Resolve success using the same priority as peg-insertion eval.

    Priority:
    1. final_info.episode.success_once
    2. episode.success_once
    3. root success

    Raises:
        ValueError: When no reliable env-side success signal is available.
    """
    if not isinstance(env_infos, dict):
        raise ValueError(
            f"Missing env_infos for env_id={env_id}; cannot resolve success_once."
        )

    final_info = env_infos.get("final_info")
    if isinstance(final_info, dict):
        final_episode = final_info.get("episode")
        if isinstance(final_episode, dict):
            value = _extract_env_value(final_episode.get("success_once"), env_id)
            if value is not None:
                return bool(value)

    episode = env_infos.get("episode")
    if isinstance(episode, dict):
        value = _extract_env_value(episode.get("success_once"), env_id)
        if value is not None:
            return bool(value)

    value = _extract_env_value(env_infos.get("success"), env_id)
    if value is not None:
        return bool(value)

    raise ValueError(
        "Robometer reward requires an env-side success signal but none was found. "
        f"env_id={env_id}, available env_info keys={sorted(env_infos.keys())}."
    )


def _interpolate_insert_progress_from_downsampled_frames(
    progress, history_len: int, pickup_count: int, max_frames: int = 60
):
    """Map down-sampled robometer progress -> per-step insertion reward + mask.

    ``history_len`` is the original full video length = pickup + insertion low-level
    steps. The history is uniformly downsampled before the Robometer POST; this
    function reprojects the returned frame rewards back onto insertion low-level
    steps and linearly interpolates between labeled insertion frames.

    Returns:
        ``(per_step_reward, per_step_has_reward)`` where both have length
        ``max(0, history_len - pickup_count)``.
    """
    n_insert = max(0, history_len - pickup_count)
    per_step_reward = np.zeros(n_insert, dtype=np.float32)
    per_step_has_reward = np.zeros(n_insert, dtype=bool)
    if n_insert == 0:
        return per_step_reward, per_step_has_reward

    ds = _robometer_downsample_indices(history_len, max_frames)
    prog = (
        np.asarray(progress, dtype=np.float32) if progress is not None else np.zeros(0)
    )

    labeled_steps: list[int] = []
    labeled_values: list[float] = []
    for j, orig_idx in enumerate(ds):
        if j >= prog.shape[0]:
            break
        if orig_idx < pickup_count:
            continue
        step_idx = int(orig_idx) - pickup_count
        if 0 <= step_idx < n_insert:
            labeled_steps.append(step_idx)
            labeled_values.append(float(prog[j]))

    if not labeled_steps:
        return per_step_reward, per_step_has_reward

    unique_steps: list[int] = []
    unique_values: list[float] = []
    for step_idx, value in zip(labeled_steps, labeled_values, strict=True):
        if unique_steps and step_idx == unique_steps[-1]:
            unique_values[-1] = value
            continue
        unique_steps.append(step_idx)
        unique_values.append(value)

    if len(unique_steps) == 1:
        per_step_reward[:] = unique_values[0]
        per_step_has_reward[:] = True
        return per_step_reward, per_step_has_reward

    for seg_idx in range(len(unique_steps) - 1):
        start_step = unique_steps[seg_idx]
        end_step = unique_steps[seg_idx + 1]
        start_value = unique_values[seg_idx]
        end_value = unique_values[seg_idx + 1]
        if end_step <= start_step:
            per_step_reward[start_step] = end_value
            continue
        steps = np.arange(start_step, end_step + 1, dtype=np.float32)
        alpha = (steps - float(start_step)) / float(end_step - start_step)
        per_step_reward[start_step : end_step + 1] = (
            1.0 - alpha
        ) * start_value + alpha * end_value

    first_step = unique_steps[0]
    last_step = unique_steps[-1]
    per_step_reward[:first_step] = unique_values[0]
    per_step_reward[last_step + 1 :] = unique_values[-1]
    per_step_has_reward[:] = True
    return per_step_reward, per_step_has_reward


def _apply_stepwise_success_shift(
    per_step_progress: np.ndarray,
    per_step_success: np.ndarray,
    fail_shift: float = 1.0,
) -> np.ndarray:
    """Convert interpolated progress into reward using per-step success labels."""
    progress = np.asarray(per_step_progress, dtype=np.float32)
    success = np.asarray(per_step_success, dtype=bool)
    if progress.shape != success.shape:
        raise ValueError(
            "per_step_progress and per_step_success must have identical shape: "
            f"{progress.shape=} vs {success.shape=}."
        )
    reward = progress.copy()
    reward[~success] -= float(fail_shift)
    return reward


@dataclass(frozen=True)
class RobometerEpisodeReward:
    """Shared Robometer reward reconstruction result for one completed episode."""

    downsample_indices: list[int]
    per_step_progress: np.ndarray
    per_step_reward: np.ndarray
    per_step_loss_mask: np.ndarray
    chunk_reward: np.ndarray
    chunk_loss_mask: np.ndarray
    # Per-chunk oracle value V(s_chunk) for "oracle_value" shaping (progress used
    # directly as the GAE value). Same [total_chunks, chunk_size] layout as
    # ``chunk_reward`` (value at index 0, zeros elsewhere) so env_worker can
    # scatter it into ``prev_values`` with the same indexing. None for
    # absolute/delta shaping (value head supplies prev_values).
    chunk_values: np.ndarray = None
    episode_success: bool = False
    initial_progress: float = float("nan")
    final_progress: float = float("nan")
    success_bonus_sum: float = 0.0


def robometer_assignment_metric_values(
    assignments: dict[int, RobometerEpisodeReward],
    *,
    positive_tolerance: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Build per-episode diagnostics from the rewards queued for PPO."""
    if not assignments:
        return {}
    assignment_values = list(assignments.values())
    episode_success = torch.tensor(
        [float(assignment.episode_success) for assignment in assignment_values],
        dtype=torch.float32,
    )
    metrics = {
        # This is resolved exclusively from the environment-side sticky success
        # trace. Keep the legacy name and expose an unambiguous primary metric.
        "reward/robometer_episode_success_rate": episode_success,
        "episode_success_rate": episode_success.clone(),
    }
    successful_assignments = [
        assignment for assignment in assignment_values if assignment.episode_success
    ]
    failed_assignments = [
        assignment for assignment in assignment_values if not assignment.episode_success
    ]

    def _episode_sum(assignment: RobometerEpisodeReward) -> float:
        valid_rewards = np.asarray(assignment.per_step_reward, dtype=np.float32)[
            np.asarray(assignment.per_step_loss_mask, dtype=bool)
        ]
        return float(valid_rewards.sum())

    successful_episode_sums = [
        _episode_sum(assignment) for assignment in successful_assignments
    ]
    failed_episode_sums = [_episode_sum(assignment) for assignment in failed_assignments]
    if successful_episode_sums:
        metrics["reward/successful_episode_reward_sum"] = torch.tensor(
            successful_episode_sums, dtype=torch.float32
        )
    if successful_episode_sums and failed_episode_sums:
        metrics["reward/success_minus_failure_reward_margin"] = torch.tensor(
            [np.mean(successful_episode_sums) - np.mean(failed_episode_sums)],
            dtype=torch.float32,
        )

    metrics["reward/robometer_initial_progress"] = torch.tensor(
        [assignment.initial_progress for assignment in assignment_values],
        dtype=torch.float32,
    )
    metrics["reward/robometer_final_progress"] = torch.tensor(
        [assignment.final_progress for assignment in assignment_values],
        dtype=torch.float32,
    )
    metrics["reward/success_bonus_fraction"] = torch.tensor(
        [
            assignment.success_bonus_sum
            / max(abs(_episode_sum(assignment)), positive_tolerance)
            for assignment in assignment_values
        ],
        dtype=torch.float32,
    )
    if not failed_assignments:
        return metrics

    failed_episode_violations = []
    failed_positive_steps = []
    for assignment in failed_assignments:
        valid_rewards = np.asarray(assignment.per_step_reward, dtype=np.float32)[
            np.asarray(assignment.per_step_loss_mask, dtype=bool)
        ]
        failed_episode_violations.append(
            float(np.any(valid_rewards > positive_tolerance))
        )
        failed_positive_steps.extend((valid_rewards > 0).astype(np.float32).tolist())

    metrics.update(
        {
            "reward/failed_episode_reward_sum": torch.tensor(
                failed_episode_sums, dtype=torch.float32
            ),
            "reward/failed_episode_positive_violation_rate": torch.tensor(
                failed_episode_violations, dtype=torch.float32
            ),
            "reward/failed_low_level_positive_fraction": torch.tensor(
                failed_positive_steps, dtype=torch.float32
            ),
        }
    )
    return metrics


def reconstruct_robometer_episode_reward(
    progress: Any,
    *,
    history_len: int,
    pickup_count: int,
    success_trace: Any,
    max_frames: int,
    fail_shift: float,
    chunk_size: int,
    total_chunks: int,
    success_terminal_bonus: float = 0.0,
) -> RobometerEpisodeReward:
    """Reconstruct low-level and chunk-aligned reward for a completed episode."""
    if chunk_size <= 0 or total_chunks <= 0:
        raise ValueError(
            f"chunk_size and total_chunks must be positive, got {chunk_size=} {total_chunks=}."
        )
    success = np.asarray(success_trace, dtype=bool)
    if success.shape[0] != history_len:
        raise ValueError(
            "Success trace must align with the complete Robometer history: "
            f"{success.shape[0]=} vs {history_len=}."
        )
    per_step_progress, per_step_loss_mask = (
        _interpolate_insert_progress_from_downsampled_frames(
            progress, history_len, pickup_count, max_frames
        )
    )
    insert_success = success[pickup_count:history_len]
    if insert_success.shape != per_step_progress.shape:
        raise ValueError(
            "Insertion success trace does not align with reconstructed progress: "
            f"{insert_success.shape=} vs {per_step_progress.shape=}."
        )
    per_step_reward = _apply_stepwise_success_shift(
        per_step_progress, insert_success, fail_shift=fail_shift
    )
    episode_success = bool(insert_success.any())
    if episode_success:
        # 将 bonus 加在第一次达到环境真实成功的低层 action 上，而不是
        # 任意 Robometer 高进度帧上；success_trace 来自环境 sticky success。
        first_success_step = int(np.flatnonzero(insert_success)[0])
        per_step_reward[first_success_step] += float(success_terminal_bonus)

    capacity = total_chunks * chunk_size
    if per_step_reward.shape[0] > capacity:
        raise ValueError(
            "Completed episode reward exceeds trajectory capacity: "
            f"{per_step_reward.shape[0]=} vs {capacity=}."
        )
    chunk_reward = np.zeros((total_chunks, chunk_size), dtype=np.float32)
    chunk_loss_mask = np.zeros((total_chunks, chunk_size), dtype=bool)
    start = capacity - per_step_reward.shape[0]
    chunk_reward.reshape(-1)[start:] = per_step_reward
    chunk_loss_mask.reshape(-1)[start:] = per_step_loss_mask
    return RobometerEpisodeReward(
        downsample_indices=_robometer_downsample_indices(history_len, max_frames),
        per_step_progress=per_step_progress,
        per_step_reward=per_step_reward,
        per_step_loss_mask=per_step_loss_mask,
        chunk_reward=chunk_reward,
        chunk_loss_mask=chunk_loss_mask,
        episode_success=episode_success,
        initial_progress=float(per_step_progress[0]),
        final_progress=float(per_step_progress[-1]),
        success_bonus_sum=(
            float(success_terminal_bonus) if episode_success else 0.0
        ),
    )


def reconstruct_robometer_delta_reward(
    boundary_progress: Any,
    *,
    history_len: int,
    pickup_count: int,
    success_trace: Any,
    chunk_size: int,
    total_chunks: int,
    success_bonus: float = 0.1,
    failure_terminal_penalty: float = 0.0,
) -> RobometerEpisodeReward:
    """Reconstruct per-chunk delta reward for a completed episode (delta shaping).

    Delta shaping (Option 2): Robometer is queried only at chunk-boundary frames
    (``n_chunks + 1`` frames). The per-chunk reward is::

        chunk_reward[i] = (p[i+1] - p[i]) + success_bonus * 1[success[i]]
        + failure_terminal_penalty * 1[episode_failure and i == last_chunk]

    where ``p`` is the boundary-frame progress and ``success[i]`` is whether chunk
    ``i`` ends in a success step. No interpolation, no per-step success shift -- the
    reward is already chunk-level.

    Reward tensor placement (verified against ``masked_mean_ratio`` in losses.py):
    the per-chunk scalar is placed at index 0 of the ``[chunk_size]`` sub-step
    vector (zeros elsewhere), and ``loss_mask`` is True for ALL ``chunk_size``
    sub-steps of each insertion chunk (matching absolute mode's
    ``per_step_has_reward[:] = True``). This keeps ``discounted_sum`` exact
    (``gamma^0 = 1``, the zeroed tail contributes nothing) and keeps
    ``loss_mask_ratio = loss_mask_sum / max_episode_steps`` identical to absolute
    mode (~insert_steps / max_episode_steps) -- setting loss_mask True only at
    index 0 would shrink ``loss_mask_sum`` ~10x and inflate the loss via
    ``masked_mean_ratio``.
    """
    if chunk_size <= 0 or total_chunks <= 0:
        raise ValueError(
            f"chunk_size and total_chunks must be positive, got {chunk_size=} {total_chunks=}."
        )
    success = np.asarray(success_trace, dtype=bool)
    if success.shape[0] != history_len:
        raise ValueError(
            "Success trace must align with the complete Robometer history: "
            f"{success.shape[0]=} vs {history_len=}."
        )
    # Expected boundary-frame count = total_chunks + 1 (one before each chunk + one
    # after the last). env_worker recomputes total_chunks = ceil(insert_steps /
    # chunk_size) from the same (history_len, pickup_count), so this matches the
    # frame count selected in compute_reward.
    n_expected = total_chunks + 1
    prog = np.asarray(boundary_progress, dtype=np.float32)
    if prog.shape[0] < n_expected:
        raise ValueError(
            "Robometer boundary progress is shorter than the completed episode's "
            f"boundary frame count: expected={n_expected}, got {prog.shape[0]=}."
        )
    prog = prog[:n_expected]
    if not np.isfinite(prog).all():
        raise ValueError("Robometer returned non-finite boundary progress.")

    chunk_reward = np.zeros((total_chunks, chunk_size), dtype=np.float32)
    chunk_loss_mask = np.zeros((total_chunks, chunk_size), dtype=bool)

    # Per-chunk delta + success bonus. success[i] = success at the END boundary step
    # of chunk i (the step just before the next chunk's start frame). Once inserted,
    # env success is sticky, so this marks the achieving chunk and all after it.
    for i in range(total_chunks):
        delta = float(prog[i + 1] - prog[i])
        end_step = min(pickup_count + (i + 1) * chunk_size - 1, history_len - 1)
        chunk_success = bool(success[end_step]) if end_step >= pickup_count else False
        chunk_reward[i, 0] = delta + (
            float(success_bonus) if chunk_success else 0.0
        )
        # loss_mask True for ALL sub-steps of insertion chunks (matching absolute
        # mode) so masked_mean_ratio's loss_mask_ratio matches absolute scaling.
        chunk_loss_mask[i, :] = True

    insert_success = success[pickup_count:history_len]
    episode_success = (
        bool(insert_success.any()) if insert_success.size > 0 else False
    )
    if not episode_success:
        # Keep dense progress feedback, but make an incomplete insertion
        # unambiguously worse than a successful trajectory at episode end.
        chunk_reward[-1, 0] += float(failure_terminal_penalty)

    # per_step_* diagnostics: report per-chunk delta + bonus as the low-level
    # proxy so robometer_assignment_metric_values still has arrays to summarize.
    # (In delta mode a failed-but-progressing episode legitimately has positive
    # deltas, so the absolute-mode "positive violation" metric is not meaningful
    # here -- it is diagnostic-only and does not gate training.)
    per_step_progress = prog[1:] - prog[:-1]
    per_step_reward = chunk_reward[:, 0].copy()
    per_step_loss_mask = np.ones(total_chunks, dtype=bool)

    return RobometerEpisodeReward(
        downsample_indices=_robometer_boundary_frame_indices(
            history_len, pickup_count, chunk_size
        ),
        per_step_progress=per_step_progress,
        per_step_reward=per_step_reward,
        per_step_loss_mask=per_step_loss_mask,
        chunk_reward=chunk_reward,
        chunk_loss_mask=chunk_loss_mask,
        episode_success=episode_success,
        initial_progress=float(prog[0]),
        final_progress=float(prog[-1]),
        success_bonus_sum=float(
            sum(
                float(success_bonus)
                for i in range(total_chunks)
                if success[
                    min(pickup_count + (i + 1) * chunk_size - 1, history_len - 1)
                ]
            )
        ),
    )


def reconstruct_robometer_oracle_value_reward(
    boundary_progress: Any,
    *,
    history_len: int,
    pickup_count: int,
    success_trace: Any,
    chunk_size: int,
    total_chunks: int,
    success_bonus: float = 1.0,
    failure_terminal_penalty: float = -0.4,
    fail_shift: float = 1.0,
) -> RobometerEpisodeReward:
    """Reconstruct per-chunk oracle value + terminal-only reward (oracle_value shaping).

    Oracle-value shaping: Robometer boundary-frame progress is used DIRECTLY as
    the GAE value ``V(s_chunk)``; the reward carries only terminal bonuses
    (``success_bonus`` on success / ``failure_terminal_penalty`` on failure). The
    per-chunk advantage under ``V = progress`` is the TD delta::

        delta_i = r_i + gamma * V[i+1] * (~done) - V[i]
                = gamma * prog[i+1] - prog[i]      (non-terminal; r_i = 0)

    i.e. the chunk progress increment -- dense, low-variance, and equivalent to
    potential-based reward shaping (Ng et al. 1999, Phi = progress), which is
    policy-invariant w.r.t. the optimal policy.

    Boundary progress, layout, and validation mirror
    ``reconstruct_robometer_delta_reward``: ``boundary_progress`` has
    ``total_chunks + 1`` entries; ``chunk_values[i]`` (the oracle V for chunk i)
    is placed at index 0 of the ``[chunk_size]`` sub-step vector; ``loss_mask``
    is True for all sub-steps of each insertion chunk. ``chunk_reward`` is zero
    except the terminal chunk's bonus/penalty (the value-head bootstrap path is
    overwritten by ``assign_history_reward`` for robometer episodes, so -- like
    delta mode -- we use a fixed terminal penalty rather than a gamma*V
    bootstrap).

    The per-episode success/fail shift (``V -= fail_shift`` on failure) is a
    constant offset. With ``gamma != 1`` it leaves a tiny
    ``(gamma - 1) * shift`` residual in non-terminal deltas (negligible,
    ~0.01/chunk at gamma=0.99); it only materially separates success from
    failure at the terminal chunk.
    """
    if chunk_size <= 0 or total_chunks <= 0:
        raise ValueError(
            f"chunk_size and total_chunks must be positive, got {chunk_size=} {total_chunks=}."
        )
    success = np.asarray(success_trace, dtype=bool)
    if success.shape[0] != history_len:
        raise ValueError(
            "Success trace must align with the complete Robometer history: "
            f"{success.shape[0]=} vs {history_len=}."
        )
    n_expected = total_chunks + 1
    prog = np.asarray(boundary_progress, dtype=np.float32)
    if prog.shape[0] < n_expected:
        raise ValueError(
            "Robometer boundary progress is shorter than the completed episode's "
            f"boundary frame count: expected={n_expected}, got {prog.shape[0]=}."
        )
    prog = prog[:n_expected]
    if not np.isfinite(prog).all():
        raise ValueError("Robometer returned non-finite boundary progress.")

    insert_success = success[pickup_count:history_len]
    episode_success = (
        bool(insert_success.any()) if insert_success.size > 0 else False
    )
    # Per-episode constant shift: failed trajectories get V -= fail_shift so a
    # failed episode that reached high progress still values below a successful
    # one (matches the absolute-mode reward convention).
    shift = 0.0 if episode_success else float(fail_shift)

    chunk_reward = np.zeros((total_chunks, chunk_size), dtype=np.float32)
    chunk_values = np.zeros((total_chunks, chunk_size), dtype=np.float32)
    chunk_loss_mask = np.zeros((total_chunks, chunk_size), dtype=bool)

    # Oracle V at chunk i = progress at chunk i's start boundary frame.
    for i in range(total_chunks):
        chunk_values[i, 0] = float(prog[i]) - shift
        chunk_loss_mask[i, :] = True

    # Terminal-only reward on the LAST chunk (mirrors delta's terminal handling;
    # the value-head bootstrap is overwritten by assign_history_reward, so use a
    # fixed bonus/penalty, not a gamma*V bootstrap).
    if episode_success:
        chunk_reward[-1, 0] += float(success_bonus)
    else:
        chunk_reward[-1, 0] += float(failure_terminal_penalty)

    # Diagnostics: per-chunk delta (the effective dense signal) + terminal reward.
    per_step_progress = prog[1:] - prog[:-1]
    per_step_reward = chunk_reward[:, 0].copy()
    per_step_loss_mask = np.ones(total_chunks, dtype=bool)

    return RobometerEpisodeReward(
        downsample_indices=_robometer_boundary_frame_indices(
            history_len, pickup_count, chunk_size
        ),
        per_step_progress=per_step_progress,
        per_step_reward=per_step_reward,
        per_step_loss_mask=per_step_loss_mask,
        chunk_reward=chunk_reward,
        chunk_loss_mask=chunk_loss_mask,
        chunk_values=chunk_values,
        episode_success=episode_success,
        initial_progress=float(prog[0]),
        final_progress=float(prog[-1]),
        success_bonus_sum=float(success_bonus) if episode_success else 0.0,
    )


def _smoke_acquire_slot(smoke_dir, cap):
    """Atomically acquire a dump slot index < cap across processes (3 reward shards)."""
    import fcntl

    os.makedirs(smoke_dir, exist_ok=True)
    lock_path = os.path.join(smoke_dir, ".counter.lock")
    ctr_path = os.path.join(smoke_dir, ".counter")
    with open(lock_path, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            n = 0
            if os.path.exists(ctr_path):
                try:
                    with open(ctr_path) as f:
                        n = int(f.read().strip() or "0")
                except Exception:
                    n = 0
            if n >= cap:
                return None
            with open(ctr_path, "w") as f:
                f.write(str(n + 1))
            return n
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _smoke_save_video(frames, path, fps=30):
    """Save [T, H, W, 3] uint8 as mp4 (cv2 -> imageio -> PNG fallback)."""
    h, w = frames.shape[1], frames.shape[2]
    try:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if wr.isOpened():
            for f in frames:
                wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            wr.release()
            return
    except Exception:
        pass
    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(path), [np.asarray(f) for f in frames], fps=fps)
        return
    except Exception:
        pass
    import matplotlib.pyplot as plt

    stem = os.path.splitext(os.path.basename(path))[0]
    for i, f in enumerate(frames):
        plt.imsave(
            os.path.join(os.path.dirname(path), f"{stem}_{i:03d}.png"), np.asarray(f)
        )


def _smoke_dump(
    smoke_dir, slot, env_id, arr, prog, env_success, shift, task, dones_present
):
    tdir = os.path.join(smoke_dir, f"traj_{slot:03d}")
    os.makedirs(tdir, exist_ok=True)
    _smoke_save_video(
        np.asarray(arr), os.path.join(tdir, "robometer_input.mp4"), fps=30
    )
    np.save(os.path.join(tdir, "progress.npy"), np.asarray(prog, dtype=np.float32))
    np.save(
        os.path.join(tdir, "reward.npy"),
        np.asarray(prog, dtype=np.float32) - float(shift),
    )
    with open(os.path.join(tdir, "success.txt"), "w") as f:
        f.write("1" if env_success else "0")
    with open(os.path.join(tdir, "meta.json"), "w") as f:
        json.dump(
            {
                "slot": int(slot),
                "env_id": int(env_id),
                "task": task,
                "n_chunks": int(arr.shape[0]),
                "success": bool(env_success),
                "dones_present": bool(dones_present),
            },
            f,
            indent=2,
        )


def _np_to_npy_file_tuple(arr: np.ndarray, filename: str):
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return (filename, buf, "application/octet-stream")


def _build_multipart_payload(samples: list[dict[str, Any]]):
    """Mirror robometer ``scripts/inference/example_inference.build_multipart_payload``.

    Vendored here because the robometer package lives in a separate uv venv and
    is not importable from the RLinf conda env.
    """
    files: dict[str, Any] = {}
    data: dict[str, str] = {}
    numpy_fields = ["frames", "lang_vector", "video_embeddings"]
    for i, sample in enumerate(samples):
        sample_copy = json.loads(json.dumps(sample, default=str))
        traj = sample.get("trajectory", {})
        traj_copy = sample_copy.get("trajectory", {})
        for field in numpy_fields:
            val = traj.get(field, None)
            if val is None:
                continue
            if hasattr(val, "detach") and hasattr(val, "cpu"):
                val = val.detach().cpu().numpy()
            if isinstance(val, np.ndarray):
                key = f"sample_{i}_trajectory_{field}"
                files[key] = _np_to_npy_file_tuple(val, f"{key}.npy")
                traj_copy[field] = {"__numpy_file__": key}
            else:
                traj_copy[field] = val
        if "frames_shape" in traj_copy and isinstance(
            traj_copy["frames_shape"], (tuple, list)
        ):
            traj_copy["frames_shape"] = [int(x) for x in traj_copy["frames_shape"]]
        sample_copy["trajectory"] = traj_copy
        data[f"sample_{i}"] = json.dumps(sample_copy)
    return files, data


def _post_evaluate_batch_npy(
    server_url: str,
    samples: list[dict[str, Any]],
    timeout_s: float,
    use_frame_steps: bool,
) -> dict[str, Any]:
    import requests

    files, data = _build_multipart_payload(samples)
    data["use_frame_steps"] = "true" if use_frame_steps else "false"
    url = server_url.rstrip("/") + "/evaluate_batch_npy"
    resp = requests.post(url, files=files, data=data, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


class RobometerHistoryRewardModel(BaseRewardModel):
    """Per-frame progress from a robometer eval server.

    Config (under ``reward.model``): ``server_url``, ``task`` (the robometer
    task string), ``timeout_s``, ``use_frame_steps``, ``fail_shift`` (default
    1.0), ``min_history_size`` (return None below this), and
    ``max_robometer_frames``. ``success_threshold`` is still parsed for backward
    compatibility but peg-insertion RL success now comes only from env infos.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.server_url = cfg.get("server_url", "http://127.0.0.1:8000")
        self.task = cfg.get("task", "Insert the peg vertically into the target hole.")
        self.timeout_s = float(cfg.get("timeout_s", 120.0))
        self.use_frame_steps = bool(cfg.get("use_frame_steps", True))
        self.success_threshold = float(cfg.get("success_threshold", 0.5))
        self.fail_shift = float(cfg.get("fail_shift", 1.0))
        self.min_history_size = int(cfg.get("min_history_size", 2))
        # Cap the number of frames POSTed to the robometer per env (uniform
        # down-sample of the full pick-up+insert history buffer). Keeps the
        # single-POST sequence length bounded (calibration + server memory) and
        # matches the frame count the model was probed at (60f -> success 0.84).
        # The env_worker recomputes the SAME down-sample indices (deterministic
        # linspace) to map progress -> chunks + set loss_mask.
        self.max_robometer_frames = int(cfg.get("max_robometer_frames", 60))
        buffers = cfg.get("history_buffers", {}) or {}
        self.render_buffer_name = cfg.get(
            "render_buffer_name",
            next(iter(buffers)) if buffers else None,
        )
        self.debug_memory_profile = bool(cfg.get("debug_memory_profile", False))

    def forward(self, input_data, labels=None):
        raise NotImplementedError(
            "RobometerHistoryRewardModel is an inference-only HTTP reward client."
        )

    @torch.no_grad()
    def compute_reward(self, observations: Any) -> Optional[torch.Tensor]:
        profile_start = time.perf_counter()
        history_input = observations.get("history_input", {}) or {}
        buf = history_input.get(self.render_buffer_name, {})
        frame_lists = buf.get("render_images", [])  # list[env] of list[frame]
        n_envs = len(frame_lists)
        if n_envs == 0:
            return None

        env_infos = observations.get("env_infos")

        # Delta / oracle_value shaping: env_worker passes per-env pickup_counts +
        # chunk_size so we select chunk-boundary frames (n_chunks+1) instead of
        # uniformly down-sampling. max_robometer_frames is NOT applied in these
        # modes (boundary count is already small, bounded by episode chunk count).
        # oracle_value reuses the SAME boundary-frame selection as delta; the
        # shaping-specific reconstruction happens later in env_worker.
        shaping = observations.get("shaping", "absolute")
        preselected = bool(observations.get("robometer_history_preselected", False))
        pickup_counts = observations.get("pickup_counts", None)
        chunk_size = int(observations.get("chunk_size", 0) or 0)
        boundary_mode = shaping in ("delta", "oracle_value")
        if boundary_mode:
            if pickup_counts is None:
                raise ValueError(
                    "Delta shaping requires `pickup_counts` in the reward input."
                )
            if chunk_size <= 0:
                raise ValueError(
                    f"Delta shaping requires a positive `chunk_size`, got {chunk_size}."
                )

        ready: list[tuple[int, np.ndarray, int]] = []  # (env_id, real arr, real_t)
        selected_frames = 0
        padded_frames = 0
        payload_bytes = 0
        for env_id, frames in enumerate(frame_lists):
            if not frames:
                continue
            arr = np.stack([np.asarray(f, dtype=np.uint8) for f in frames])
            if arr.shape[0] < self.min_history_size:
                continue
            if preselected:
                # Env workers select the exact same deterministic frame indices
                # before Ray serialization.  Do not apply those indices again.
                pass
            elif boundary_mode:
                # Select chunk-boundary frames (start of insertion + after each
                # chunk). env_worker recomputes the SAME indices' count to map
                # boundary progress -> chunks in reconstruct_robometer_delta_reward.
                ds = _robometer_boundary_frame_indices(
                    arr.shape[0], int(pickup_counts[env_id]), chunk_size
                )
                arr = arr[ds]
            else:
                # Uniform down-sample the full pick-up+insert buffer to <= max frames
                # before the single POST (use_frame_steps=False). env_worker recomputes
                # the SAME indices to map progress -> chunks + set loss_mask.
                ds = _robometer_downsample_indices(arr.shape[0], self.max_robometer_frames)
                if len(ds) < arr.shape[0]:
                    arr = arr[ds]
            selected_frames += int(arr.shape[0])
            payload_bytes += int(arr.nbytes)
            ready.append((env_id, arr, arr.shape[0]))
        if not ready:
            return None

        # Pad each env's frames to the batch max with repeat-last-frame so the
        # server's per-batch torch.stack(progress_list) sees EQUAL sequence
        # lengths. Without this, envs with different down-sampled lengths (e.g.
        # 60 vs 43) make the server stack fail ([60,10] vs [43,10]) -> RuntimeError
        # -> 500. real_t is kept so the reward mapping below uses only the real
        # frames' progress (the padded tail's progress is ignored).
        max_t = max(r[2] for r in ready)
        samples = []
        for env_id, arr, real_t in ready:
            if real_t < max_t:
                arr_pad = np.concatenate(
                    [arr, np.repeat(arr[-1:], max_t - real_t, axis=0)], axis=0
                )
            else:
                arr_pad = arr
            padded_frames += int(arr_pad.shape[0] - real_t)
            samples.append(
                {
                    "sample_type": "progress",
                    "trajectory": {
                        "frames": arr_pad,
                        "frames_shape": list(arr_pad.shape),
                        "task": self.task,
                        "id": str(env_id),
                        "metadata": {"subsequence_length": int(arr_pad.shape[0])},
                        "video_embeddings": None,
                    },
                }
            )
        if self.debug_memory_profile:
            _LOGGER.info(
                "robometer payload profile "
                f"envs={len(ready)} selected_frames={selected_frames} "
                f"padded_frames={padded_frames} payload_bytes={payload_bytes} "
                f"max_t={max_t} shaping={shaping} "
                f"elapsed_s={time.perf_counter() - profile_start:.3f}"
            )
        outputs = _post_evaluate_batch_npy(
            self.server_url, samples, self.timeout_s, self.use_frame_steps
        )
        prog_lists = outputs.get("outputs_progress", {}).get("progress_pred", [])
        if len(prog_lists) != len(ready):
            raise ValueError(
                "Robometer returned an unexpected number of progress sequences: "
                f"expected {len(ready)}, got {len(prog_lists)}."
            )

        out = torch.zeros((n_envs, max_t), dtype=torch.float32)
        for idx, (env_id, arr, real_t) in enumerate(ready):
            if prog_lists[idx] is None or len(prog_lists[idx]) < real_t:
                raise ValueError(
                    "Robometer returned an empty or truncated progress sequence: "
                    f"env_id={env_id}, expected at least {real_t}, got "
                    f"{0 if prog_lists[idx] is None else len(prog_lists[idx])}."
                )
            prog = np.asarray(prog_lists[idx], dtype=np.float32)[:real_t]
            if not np.isfinite(prog).all():
                raise ValueError(
                    f"Robometer returned non-finite progress for env_id={env_id}."
                )
            t = real_t
            env_success = _resolve_env_success_from_infos(env_infos, env_id)
            shift = 0.0 if env_success else self.fail_shift
            out[env_id, :t] = torch.from_numpy(prog)
            if _SMOKE_DIR:
                try:
                    _prev = _SMOKE_PREV.get(env_id)
                    _cur_len = int(arr.shape[0])
                    if _prev is not None and _cur_len < _prev["len"]:
                        _slot = None
                        with _SMOKE_LOCAL_LOCK:
                            _slot = _smoke_acquire_slot(_SMOKE_DIR, _SMOKE_CAP)
                        if _slot is not None:
                            _smoke_dump(
                                _SMOKE_DIR,
                                _slot,
                                env_id,
                                _prev["arr"],
                                _prev["prog"],
                                _prev["success"],
                                _prev["shift"],
                                self.task,
                                False,
                            )
                    _SMOKE_PREV[env_id] = {
                        "arr": arr,
                        "prog": prog,
                        "success": env_success,
                        "shift": shift,
                        "len": _cur_len,
                    }
                except Exception:
                    pass
        return out
