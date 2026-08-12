# Copyright 2025 The RLinf Authors.
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

import asyncio
import math
import os
import queue
import threading
import time
from typing import Any, Optional

import numpy as np
import torch

from rlinf.algorithms.losses import (
    compute_decoupled_ppo_actor_loss,
    compute_explained_variance_statistics,
    explained_variance_metrics_from_statistics,
)
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import kl_penalty, preprocess_loss_inputs
from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.priority_store import PriorityStore
from rlinf.data.staleness_mask import (
    candidate_batch_is_usable,
    compute_staleness_mask,
    count_fresh_chunks,
)
from rlinf.scheduler import CommMapper, Worker
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_embodied_reward_metrics,
    compute_rollout_metrics,
)
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import (
    clear_memory,
    cpu_weight_swap,
    masked_mean,
    masked_mean_ratio,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

_ADV_LOGPROB_METRIC_REASON_VALID = 0
_ADV_LOGPROB_METRIC_REASON_TOO_FEW_SAMPLES = 1
_ADV_LOGPROB_METRIC_REASON_ZERO_ADVANTAGE_VARIANCE = 2
_ADV_LOGPROB_METRIC_REASON_ZERO_LOGPROB_DELTA_VARIANCE = 3
_ADV_LOGPROB_METRIC_REASON_NON_FINITE_INPUTS = 4


def flatten_rollout_batch_for_train(
    nested_dict: dict, shuffle_id: Optional[torch.Tensor]
) -> dict:
    """Flatten [T, B, ...] rollout tensors to [T*B, ...] for actor training."""
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            if isinstance(value, torch.Tensor):
                value = value[:-1]

        if "env_info" in key:
            raise NotImplementedError("env_info nested dict is not supported here")

        if value is None:
            ret_dict[key] = None
            continue

        if isinstance(value, torch.Tensor):
            flat = value.reshape(-1, *value.shape[2:])
            ret_dict[key] = flat[shuffle_id] if shuffle_id is not None else flat
        elif isinstance(value, dict):
            ret_dict[key] = flatten_rollout_batch_for_train(value, shuffle_id)
        else:
            raise NotImplementedError(
                f"Unsupported value type in rollout batch: key={key}, type={type(value)}"
            )

    return ret_dict


def _reduce_to_item_level(
    tensor: torch.Tensor,
    logprob_type: str,
    single_action_dim: int,
    *,
    reduction: str,
) -> torch.Tensor:
    """Reduce flattened training tensors to actor item granularity."""
    if logprob_type == "chunk_level":
        if tensor.dim() == 1:
            base = tensor
        else:
            base = tensor.reshape(tensor.shape[0], -1).sum(dim=1)
    elif logprob_type == "action_level":
        if tensor.dim() == 1:
            base = tensor
        else:
            base = tensor.reshape(tensor.shape[0], -1, single_action_dim).sum(dim=-1)
    elif logprob_type == "token_level":
        if tensor.dim() == 1:
            base = tensor
        else:
            base = tensor.reshape(tensor.shape[0], -1, single_action_dim)
    else:
        raise ValueError(f"Unsupported logprob_type={logprob_type!r}.")

    if reduction == "sum":
        return base if base.dim() == 1 else base.sum(dim=tuple(range(1, base.dim())))
    if reduction == "mean":
        return (
            base
            if base.dim() == 1
            else base.float().mean(dim=tuple(range(1, base.dim())))
        )
    if reduction == "any":
        return base if base.dim() == 1 else base.any(dim=tuple(range(1, base.dim())))
    raise ValueError(f"Unsupported reduction={reduction!r}.")


def compute_adv_logprob_diagnostics(
    *,
    advantages: torch.Tensor,
    prev_logprobs: torch.Tensor,
    post_update_logprobs: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    logprob_type: str,
    single_action_dim: int,
) -> dict[str, torch.Tensor]:
    """Measure whether higher-advantage items received larger logprob increases."""
    device = advantages.device
    prev_logprobs = prev_logprobs.to(device=device)
    post_update_logprobs = post_update_logprobs.to(device=device)
    if loss_mask is not None:
        loss_mask = loss_mask.to(device=device)

    def _scalar(value: float | torch.Tensor) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    metrics = {
        "actor/adv_logprob_delta_corr": float("nan"),
        "actor/adv_logprob_delta_cov": float("nan"),
        "actor/mean_logprob_delta_pos_adv": float("nan"),
        "actor/mean_logprob_delta_neg_adv": float("nan"),
        "actor/adv_logprob_direction_match_rate": float("nan"),
        "actor/adv_top_quartile_logprob_delta_mean": float("nan"),
        "actor/adv_logprob_metric_valid": 0.0,
        "actor/adv_logprob_metric_numel": 0.0,
        "actor/adv_logprob_metric_invalid_reason": float(
            _ADV_LOGPROB_METRIC_REASON_VALID
        ),
    }

    item_advantages = _reduce_to_item_level(
        advantages.float(),
        logprob_type,
        single_action_dim,
        reduction="mean",
    )
    item_delta_logprob = _reduce_to_item_level(
        (post_update_logprobs.float() - prev_logprobs.float()),
        logprob_type,
        single_action_dim,
        reduction="sum",
    )
    if loss_mask is None:
        item_mask = torch.ones_like(item_advantages, dtype=torch.bool)
    else:
        item_mask = _reduce_to_item_level(
            loss_mask.bool(),
            logprob_type,
            single_action_dim,
            reduction="any",
        ).bool()

    finite_mask = torch.isfinite(item_advantages) & torch.isfinite(item_delta_logprob)
    valid_mask = item_mask & finite_mask
    metrics["actor/adv_logprob_metric_numel"] = float(valid_mask.count_nonzero().item())

    if bool((item_mask & ~finite_mask).any()):
        metrics["actor/adv_logprob_metric_invalid_reason"] = float(
            _ADV_LOGPROB_METRIC_REASON_NON_FINITE_INPUTS
        )
        return metrics

    if valid_mask.count_nonzero() <= 1:
        metrics["actor/adv_logprob_metric_invalid_reason"] = float(
            _ADV_LOGPROB_METRIC_REASON_TOO_FEW_SAMPLES
        )
        return metrics

    masked_advantages = item_advantages[valid_mask]
    masked_delta_logprob = item_delta_logprob[valid_mask]

    adv_mean = masked_advantages.mean()
    delta_mean = masked_delta_logprob.mean()
    centered_adv = masked_advantages - adv_mean
    centered_delta = masked_delta_logprob - delta_mean
    adv_var = torch.mean(centered_adv.square())
    delta_var = torch.mean(centered_delta.square())

    if not torch.isfinite(adv_var) or not torch.isfinite(delta_var):
        metrics["actor/adv_logprob_metric_invalid_reason"] = float(
            _ADV_LOGPROB_METRIC_REASON_NON_FINITE_INPUTS
        )
        return metrics
    if adv_var <= 0:
        metrics["actor/adv_logprob_metric_invalid_reason"] = float(
            _ADV_LOGPROB_METRIC_REASON_ZERO_ADVANTAGE_VARIANCE
        )
        return metrics
    if delta_var <= 0:
        metrics["actor/adv_logprob_metric_invalid_reason"] = float(
            _ADV_LOGPROB_METRIC_REASON_ZERO_LOGPROB_DELTA_VARIANCE
        )
        return metrics

    covariance = torch.mean(centered_adv * centered_delta)
    correlation = covariance / torch.sqrt(adv_var * delta_var)

    pos_mask = masked_advantages > 0
    neg_mask = masked_advantages < 0
    nonzero_mask = pos_mask | neg_mask

    if bool(pos_mask.any()):
        metrics["actor/mean_logprob_delta_pos_adv"] = _scalar(
            masked_delta_logprob[pos_mask].mean()
        )
    if bool(neg_mask.any()):
        metrics["actor/mean_logprob_delta_neg_adv"] = _scalar(
            masked_delta_logprob[neg_mask].mean()
        )
    if bool(nonzero_mask.any()):
        sign_match = torch.sign(masked_advantages[nonzero_mask]) == torch.sign(
            masked_delta_logprob[nonzero_mask]
        )
        metrics["actor/adv_logprob_direction_match_rate"] = _scalar(
            sign_match.float().mean()
        )

    top_count = max(1, math.ceil(masked_advantages.numel() * 0.25))
    top_values = torch.topk(masked_advantages, k=top_count).values
    top_threshold = top_values.min()
    top_mask = masked_advantages >= top_threshold
    if bool(top_mask.any()):
        metrics["actor/adv_top_quartile_logprob_delta_mean"] = _scalar(
            masked_delta_logprob[top_mask].mean()
        )

    metrics["actor/adv_logprob_delta_corr"] = _scalar(correlation)
    metrics["actor/adv_logprob_delta_cov"] = _scalar(covariance)
    metrics["actor/adv_logprob_metric_valid"] = 1.0
    return metrics


def compute_policy_adv_logprob_diagnostics(
    *,
    advantages: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    post_update_logprobs: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    logprob_type: str,
    single_action_dim: int,
) -> dict[str, float]:
    """Measure advantage alignment against the frozen actor-side policy."""
    device = advantages.device
    proximal_logprobs = proximal_logprobs.to(device=device)
    post_update_logprobs = post_update_logprobs.to(device=device)
    if loss_mask is not None:
        loss_mask = loss_mask.to(device=device)

    base = compute_adv_logprob_diagnostics(
        advantages=advantages,
        prev_logprobs=proximal_logprobs,
        post_update_logprobs=post_update_logprobs,
        loss_mask=loss_mask,
        logprob_type=logprob_type,
        single_action_dim=single_action_dim,
    )
    renamed = {
        "actor/adv_logprob_delta_corr": "actor/policy_adv_logprob_corr",
        "actor/adv_logprob_delta_cov": "actor/policy_adv_logprob_cov",
        "actor/mean_logprob_delta_pos_adv": ("actor/policy_positive_adv_logprob_delta"),
        "actor/mean_logprob_delta_neg_adv": ("actor/policy_negative_adv_logprob_delta"),
        "actor/adv_logprob_direction_match_rate": (
            "actor/policy_adv_direction_match_rate"
        ),
        "actor/adv_top_quartile_logprob_delta_mean": (
            "actor/policy_top_quartile_logprob_delta"
        ),
        "actor/adv_logprob_metric_valid": "actor/policy_adv_metric_valid",
        "actor/adv_logprob_metric_numel": "actor/policy_adv_metric_numel",
        "actor/adv_logprob_metric_invalid_reason": (
            "actor/policy_adv_metric_invalid_reason"
        ),
    }
    metrics = {renamed[key]: value for key, value in base.items()}

    item_advantages = _reduce_to_item_level(
        advantages.float(), logprob_type, single_action_dim, reduction="mean"
    )
    item_delta = _reduce_to_item_level(
        post_update_logprobs.float() - proximal_logprobs.float(),
        logprob_type,
        single_action_dim,
        reduction="sum",
    )
    if loss_mask is None:
        item_mask = torch.ones_like(item_advantages, dtype=torch.bool)
    else:
        item_mask = _reduce_to_item_level(
            loss_mask.bool(), logprob_type, single_action_dim, reduction="any"
        ).bool()
    valid = item_mask & torch.isfinite(item_advantages) & torch.isfinite(item_delta)
    metrics["actor/adv_weighted_policy_logprob_delta"] = (
        float((item_advantages[valid] * item_delta[valid]).mean().item())
        if bool(valid.any())
        else float("nan")
    )
    return metrics


def compute_gradient_conflict_metrics(
    policy_grad_norm: float,
    critic_grad_norm: float,
    combined_grad_norm: float,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Derive policy/critic gradient alignment from three global norms."""
    policy_sq = policy_grad_norm**2
    critic_sq = critic_grad_norm**2
    denominator = 2.0 * policy_grad_norm * critic_grad_norm
    valid = denominator > eps and all(
        math.isfinite(value)
        for value in (policy_grad_norm, critic_grad_norm, combined_grad_norm)
    )
    cosine = (
        (combined_grad_norm**2 - policy_sq - critic_sq) / denominator
        if valid
        else float("nan")
    )
    if valid:
        cosine = min(1.0, max(-1.0, cosine))
    return {
        "actor/grad_diagnostics_valid": float(valid),
        "actor/policy_grad_norm_sampled": policy_grad_norm,
        "critic/grad_norm_sampled": critic_grad_norm,
        "actor/combined_grad_norm_sampled": combined_grad_norm,
        "critic/policy_grad_norm_ratio_sampled": (
            critic_grad_norm / max(policy_grad_norm, eps)
        ),
        "actor_critic/shared_grad_cosine_sampled": cosine,
    }


def compute_preupdate_logprob_mismatch_metrics(
    *,
    proximal_logprobs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    logprob_type: str,
    single_action_dim: int,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> dict[str, float]:
    """Measure rollout-versus-actor logprob mismatch before optimization."""
    proximal = _reduce_to_item_level(
        proximal_logprobs.float(), logprob_type, single_action_dim, reduction="sum"
    )
    behavior = _reduce_to_item_level(
        behavior_logprobs.float(), logprob_type, single_action_dim, reduction="sum"
    )
    if loss_mask is None:
        mask = torch.ones_like(proximal, dtype=torch.bool)
    else:
        mask = _reduce_to_item_level(
            loss_mask.bool(), logprob_type, single_action_dim, reduction="any"
        ).bool()
    mask &= torch.isfinite(proximal) & torch.isfinite(behavior)
    if not bool(mask.any()):
        return {
            "actor/preupdate_behavior_approx_kl": float("nan"),
            "actor/preupdate_behavior_ratio": float("nan"),
            "actor/preupdate_behavior_clip_fraction": float("nan"),
            "actor/preupdate_behavior_logprob_delta_abs_mean": float("nan"),
            "actor/preupdate_behavior_logprob_delta_abs_max": float("nan"),
        }

    delta = proximal[mask] - behavior[mask]
    ratio = torch.exp(delta)
    return {
        "actor/preupdate_behavior_approx_kl": float((-delta.mean()).item()),
        "actor/preupdate_behavior_ratio": float(ratio.mean().item()),
        "actor/preupdate_behavior_clip_fraction": float(
            ((ratio < 1.0 - clip_ratio_low) | (ratio > 1.0 + clip_ratio_high))
            .float()
            .mean()
            .item()
        ),
        "actor/preupdate_behavior_logprob_delta_abs_mean": float(
            delta.abs().mean().item()
        ),
        "actor/preupdate_behavior_logprob_delta_abs_max": float(
            delta.abs().max().item()
        ),
    }


def compute_post_update_proximal_metrics(
    *,
    proximal_logprobs: torch.Tensor,
    post_update_logprobs: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    logprob_type: str,
    single_action_dim: int,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> dict[str, float]:
    """Measure the optimizer update against the frozen proximal policy."""
    device = proximal_logprobs.device
    post_update_logprobs = post_update_logprobs.to(device=device)
    if loss_mask is not None:
        loss_mask = loss_mask.to(device=device)
    proximal = _reduce_to_item_level(
        proximal_logprobs.float(), logprob_type, single_action_dim, reduction="sum"
    )
    post_update = _reduce_to_item_level(
        post_update_logprobs.float(), logprob_type, single_action_dim, reduction="sum"
    )
    if loss_mask is None:
        mask = torch.ones_like(proximal, dtype=torch.bool)
    else:
        mask = _reduce_to_item_level(
            loss_mask.bool(), logprob_type, single_action_dim, reduction="any"
        ).bool()
    mask &= torch.isfinite(proximal) & torch.isfinite(post_update)
    names = (
        "actor/post_update_proximal_approx_kl",
        "actor/post_update_proximal_ratio",
        "actor/post_update_proximal_clip_fraction",
        "actor/post_update_logprob_delta_mean",
        "actor/post_update_logprob_delta_abs_mean",
        "actor/post_update_logprob_delta_abs_max",
    )
    if not bool(mask.any()):
        return dict.fromkeys(names, float("nan"))

    delta = post_update[mask] - proximal[mask]
    ratio = torch.exp(delta)
    return {
        "actor/post_update_proximal_approx_kl": float((-delta.mean()).item()),
        "actor/post_update_proximal_ratio": float(ratio.mean().item()),
        "actor/post_update_proximal_clip_fraction": float(
            ((ratio < 1.0 - clip_ratio_low) | (ratio > 1.0 + clip_ratio_high))
            .float()
            .mean()
            .item()
        ),
        "actor/post_update_logprob_delta_mean": float(delta.mean().item()),
        "actor/post_update_logprob_delta_abs_mean": float(delta.abs().mean().item()),
        "actor/post_update_logprob_delta_abs_max": float(delta.abs().max().item()),
    }


def compute_post_update_ppo_surrogate_metrics(
    *,
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    post_update_logprobs: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    loss_mask_sum: Optional[torch.Tensor],
    max_episode_steps: Optional[int],
    logprob_type: str,
    single_action_dim: int,
    reward_type: str,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_c: Optional[float],
    behave_weight_threshold: Optional[float],
) -> dict[str, float]:
    """Measure the realized update using the same decoupled PPO surrogate as training.

    Unlike an unweighted advantage/logprob correlation, the primary improvement
    metric below uses the behavior importance weights, behavior threshold, PPO
    clipping, dual clipping, and embodied loss aggregation used by the actor loss.
    """
    device = proximal_logprobs.device
    advantages = advantages.to(device=device, dtype=torch.float32)
    old_logprobs = old_logprobs.to(device=device, dtype=torch.float32)
    proximal_logprobs = proximal_logprobs.to(device=device, dtype=torch.float32)
    post_update_logprobs = post_update_logprobs.to(device=device, dtype=torch.float32)
    if loss_mask is None:
        loss_mask = torch.ones_like(proximal_logprobs, dtype=torch.bool)
    else:
        loss_mask = loss_mask.to(device=device, dtype=torch.bool)
    if loss_mask_sum is not None:
        loss_mask_sum = loss_mask_sum.to(device=device)

    # Match policy_loss(task_type="embodied") exactly: chunk/action/token-level
    # logprobs and their associated loss tensors have different native shapes.
    prepared = preprocess_loss_inputs(
        logprobs=post_update_logprobs,
        old_logprobs=old_logprobs,
        proximal_logprobs=proximal_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask_sum,
        logprob_type=logprob_type,
        single_action_dim=single_action_dim,
        reward_type=reward_type,
    )
    post_update_logprobs = prepared["logprobs"]
    old_logprobs = prepared["old_logprobs"]
    proximal_logprobs = prepared["proximal_logprobs"]
    advantages = prepared["advantages"]
    loss_mask = prepared["loss_mask"]
    loss_mask_sum = prepared["loss_mask_sum"]

    loss_kwargs = {
        "old_logprobs": old_logprobs,
        "clip_ratio_low": clip_ratio_low,
        "clip_ratio_high": clip_ratio_high,
        "advantages": advantages,
        "proximal_logprobs": proximal_logprobs,
        "loss_mask": loss_mask,
        "clip_ratio_c": clip_ratio_c,
        "max_episode_steps": max_episode_steps,
        "loss_mask_sum": loss_mask_sum,
        "behave_weight_threshold": behave_weight_threshold,
    }
    with torch.no_grad():
        pre_loss, _ = compute_decoupled_ppo_actor_loss(
            logprobs=proximal_logprobs, **loss_kwargs
        )
        post_loss, _ = compute_decoupled_ppo_actor_loss(
            logprobs=post_update_logprobs, **loss_kwargs
        )

        behavior_weight = torch.exp(proximal_logprobs - old_logprobs)
        valid = (
            loss_mask
            & torch.isfinite(advantages)
            & torch.isfinite(proximal_logprobs)
            & torch.isfinite(post_update_logprobs)
            & torch.isfinite(behavior_weight)
        )
        if behave_weight_threshold is not None:
            valid &= behavior_weight <= behave_weight_threshold

        metrics = {
            "actor/post_update_ppo_actor_loss": float(post_loss.item()),
            "actor/pre_update_ppo_actor_loss": float(pre_loss.item()),
            "actor/post_update_ppo_surrogate_improvement": float(
                (pre_loss - post_loss).item()
            ),
            "actor/post_update_behavior_weight_mean": float("nan"),
            "actor/post_update_behavior_weight_max": float("nan"),
            "actor/post_update_behavior_valid_fraction": float(
                valid.count_nonzero().float().div(loss_mask.count_nonzero() or 1).item()
            ),
            "actor/post_update_ppo_improved_fraction": float("nan"),
            "actor/post_update_ppo_weighted_improved_fraction": float("nan"),
            "actor/post_update_pg_weighted_logprob_corr": float("nan"),
            "actor/post_update_first_order_surrogate_gain": float("nan"),
        }
        if not bool(valid.any()):
            return metrics

        valid_weights = behavior_weight[valid]
        metrics["actor/post_update_behavior_weight_mean"] = float(
            valid_weights.mean().item()
        )
        metrics["actor/post_update_behavior_weight_max"] = float(
            valid_weights.max().item()
        )

        def surrogate_contribution(logprobs: torch.Tensor) -> torch.Tensor:
            ratio = torch.exp(logprobs - proximal_logprobs)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
            )
            contribution = torch.maximum(
                -advantages * ratio, -advantages * clipped_ratio
            )
            if clip_ratio_c is not None:
                dual_clipped = torch.sign(advantages) * clip_ratio_c * advantages
                contribution = torch.minimum(contribution, dual_clipped)
            return contribution * behavior_weight

        improvement = surrogate_contribution(
            proximal_logprobs
        ) - surrogate_contribution(post_update_logprobs)
        improved = improvement[valid] > 0
        metrics["actor/post_update_ppo_improved_fraction"] = float(
            improved.float().mean().item()
        )
        metrics["actor/post_update_ppo_weighted_improved_fraction"] = float(
            (valid_weights * improved.float()).sum().div(valid_weights.sum()).item()
        )

        delta = post_update_logprobs[valid] - proximal_logprobs[valid]
        valid_advantages = advantages[valid]
        weighted_advantage_mean = (
            valid_weights * valid_advantages
        ).sum() / valid_weights.sum()
        weighted_delta_mean = (valid_weights * delta).sum() / valid_weights.sum()
        weighted_adv_var = (
            valid_weights * (valid_advantages - weighted_advantage_mean).square()
        ).sum() / valid_weights.sum()
        weighted_delta_var = (
            valid_weights * (delta - weighted_delta_mean).square()
        ).sum() / valid_weights.sum()
        if weighted_adv_var > 0 and weighted_delta_var > 0:
            covariance = (
                valid_weights
                * (valid_advantages - weighted_advantage_mean)
                * (delta - weighted_delta_mean)
            ).sum() / valid_weights.sum()
            metrics["actor/post_update_pg_weighted_logprob_corr"] = float(
                (covariance / torch.sqrt(weighted_adv_var * weighted_delta_var)).item()
            )

        first_order_terms = (
            behavior_weight * advantages * (post_update_logprobs - proximal_logprobs)
        )
        if loss_mask_sum is not None and max_episode_steps is not None:
            first_order_gain = masked_mean_ratio(
                first_order_terms, valid, (loss_mask_sum.float() / max_episode_steps)
            )
        else:
            first_order_gain = masked_mean(first_order_terms, valid)
        metrics["actor/post_update_first_order_surrogate_gain"] = float(
            first_order_gain.item()
        )
        return metrics


class AsyncPPOEmbodiedFSDPActor(EmbodiedFSDPActor):
    """Embodied FSDP actor worker for async PPO / decoupled actor-critic training."""

    should_stop = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # EmbodiedFSDPActor does not inherit FSDPActor's reference-policy
        # initialization. Keep the async PPO reference anchored to the SFT
        # weights loaded during init_worker(), before the first actor update.
        self.kl_beta = float(self.cfg.algorithm.get("kl_beta", 0.0))
        self.kl_penalty_type = self.cfg.algorithm.get(
            "kl_penalty_type", self.cfg.algorithm.get("kl_penalty", "low_var_kl")
        )
        self.ref_policy_state_dict = None
        self.offload_model_buffer = {}
        self.rollout_store_size = self.cfg.algorithm.get(
            "rollout_store_size_per_rank", 1
        )
        self.rollout_store = PriorityStore(maxsize=self.rollout_store_size)
        # Captured by the receive thread on a fatal error so the readiness
        # collective can sync an all-reduce abort instead of deadlocking.
        self._recv_thread_exc: Exception | None = None
        self._rollout_credit_channel = None

    def init_worker(self) -> None:
        """Initialize the actor and capture the immutable SFT KL reference."""
        super().init_worker()
        if self.kl_beta > 0:
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, load_grad=False)
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            if self.enable_offload:
                self.offload_param_and_grad()

    async def recv_rollout_trajectories(self, input_channel, credit_channel=None):
        self._rollout_credit_channel = credit_channel
        # drain channel
        if getattr(self, "_recv_queue", None) is None:
            self._recv_queue = queue.Queue()
        if (
            getattr(self, "_recv_rollout_thread", None) is None
            or not self._recv_rollout_thread.is_alive()
        ):
            self._recv_thread_exc = None
            self._recv_rollout_thread = threading.Thread(
                target=self._recv_rollout_thread_main,
                args=(input_channel,),
                daemon=True,
            )
            self._recv_rollout_thread.start()

    def _recv_rollout_thread_main(self, input_channel):
        staleness_filter_mode = self.cfg.algorithm.get(
            "staleness_filter_mode", "trajectory"
        )
        staleness_threshold = self.cfg.algorithm.get("staleness_threshold", None)
        keyed_routing = self.cfg.algorithm.get("actor_channel_keyed_routing", False)
        recv_key = (
            CommMapper.build_channel_key(self._rank, self._rank, "async_actor")
            if keyed_routing
            else None
        )
        while not self.should_stop:
            try:
                trajectory: Trajectory = (
                    input_channel.get(key=recv_key)
                    if keyed_routing
                    else input_channel.get()
                )
            except Exception as exc:  # noqa: BLE001 - never let daemon die silently
                self._recv_thread_exc = exc
                self.log_info(
                    f"recv thread aborting rank={self._rank} version={self.version} "
                    f"exc={exc!r}"
                )
                return
            self.log_info(
                f"recv trajectory rank={self._rank} version={self.version} "
                f"versions.shape={trajectory.versions.shape} "
                f"v_min={float(trajectory.versions.min())} "
                f"v_mean={float(trajectory.versions.float().mean()):.2f} "
                f"v_max={float(trajectory.versions.max())} "
                f"recv_queue={self._recv_queue.qsize() if self._recv_queue else 0}"
            )
            # chunk_mask mode never drops a trajectory here: stale chunks are masked
            # later in _compute_staleness_mask so the episode stays whole for GAE.
            if (
                staleness_filter_mode != "chunk_mask"
                and staleness_threshold is not None
                and trajectory.versions.min() < (self.version - staleness_threshold)
            ):
                continue
            self._recv_queue.put(trajectory)

    @Worker.timer("drain_received_trajectories")
    def _drain_received_trajectories(self):
        staleness_filter_mode = self.cfg.algorithm.get(
            "staleness_filter_mode", "trajectory"
        )
        staleness_threshold = self.cfg.algorithm.get("staleness_threshold", None)
        while True:
            try:
                traj: Trajectory = self._recv_queue.get_nowait()
                min_v = float(traj.versions.min().item())
                mean_v = float(traj.versions.float().mean().item())
                max_v = float(traj.versions.max().item())
                self.log_info(
                    f"drain traj rank={self._rank} version={self.version} "
                    f"versions.shape={traj.versions.shape} "
                    f"v_min={min_v} v_mean={mean_v:.2f} v_max={max_v} "
                    f"recv_queue={self._recv_queue.qsize()}"
                )
                if (
                    staleness_filter_mode != "chunk_mask"
                    and staleness_threshold is not None
                    and min_v < (self.version - staleness_threshold)
                ):
                    continue
                self.rollout_store.add((min_v, mean_v), traj)
                self.log_info(f"rollout_store size={len(self.rollout_store)}")
            except queue.Empty:
                break

    @Worker.timer("wait_for_rollout_store_ready")
    async def _wait_for_rollout_store_ready(self):
        """Wait until this rank (and, in chunk_mask mode, ALL ranks) have fresh
        rollout data; returns the taken ``list[Trajectory]`` batch.

        trajectory mode (default): per-rank ``remove_below`` + ``topn`` (legacy).
        chunk_mask mode: a two-phase ``all_reduce(MIN)`` readiness collective so
        every rank enters training together -- phase 1 confirms every rank has a
        candidate, phase 2 confirms every rank has >=1 fresh chunk; on success
        ``take_topn`` consumes the batch in lockstep, on all-stale every rank
        ``discard_topn`` together. A per-iteration ``all_reduce(MAX)`` abort
        syncs all ranks out on timeout / receive-thread crash (the collective is
        the sync point, so no rank can break out alone -> no deadlock).
        """
        while getattr(self, "_recv_queue", None) is None:
            await asyncio.sleep(1)

        staleness_filter_mode = self.cfg.algorithm.get(
            "staleness_filter_mode", "trajectory"
        )
        staleness_threshold = self.cfg.algorithm.get("staleness_threshold", None)
        on_policy_min_ratio = self.cfg.algorithm.get("on_policy_min_ratio", 0.0)
        n = self.rollout_store_size

        if staleness_filter_mode != "chunk_mask":
            while True:
                self._drain_received_trajectories()
                if staleness_threshold is not None:
                    with self.worker_timer("remove_below"):
                        self.rollout_store.remove_below(
                            self.version - staleness_threshold
                        )
                if len(self.rollout_store) >= n:
                    if on_policy_min_ratio <= 0.0:
                        break
                    metrics_data = self.rollout_store.get_metric()
                    on_policy_ratio = metrics_data.get(int(self.version), {}).get(
                        "ratio", 0.0
                    )
                    self.log_info(
                        f"rollout store metrics={metrics_data} "
                        f"on_policy_ratio={on_policy_ratio:.4f} "
                        f"on_policy_min_ratio={on_policy_min_ratio}"
                    )
                    if on_policy_ratio >= on_policy_min_ratio:
                        break
                await asyncio.sleep(1)
            batch = self.rollout_store.topn(n)
            self._staleness_readiness = {
                "staleness_received_trajectories": len(batch),
                "staleness_global_retry_rounds": 0,
                "staleness_wait_seconds": 0.0,
            }
            return batch

        # ---- chunk_mask: two-phase all-reduce readiness collective ----
        timeout = float(self.cfg.algorithm.get("rollout_store_wait_timeout_s", 600))
        initial_timeout = float(
            self.cfg.algorithm.get(
                "rollout_store_initial_wait_timeout_s",
                max(timeout, 1800.0),
            )
        )
        status_interval = float(
            self.cfg.algorithm.get("rollout_store_status_interval_s", 30)
        )
        device = (
            torch.cuda.current_device()
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        cutoff = (
            int(self.version) - int(staleness_threshold)
            if staleness_threshold is not None
            else -(10**9)
        )
        min_fresh_chunks = int(self.cfg.algorithm.get("readiness_min_fresh_chunks", 1))
        start = time.time()
        global_retry = 0
        last_status = 0.0
        received = 0

        while True:
            self._drain_received_trajectories()
            candidates = self.rollout_store.peek_topn(n)
            received = max(received, len(candidates))

            # Per-iteration abort sync (all_reduce is the sync point): any rank
            # timed out OR its recv thread crashed -> every rank raises together.
            now = time.time()
            wait_deadline = timeout if received > 0 else initial_timeout
            local_abort = (
                1.0
                if ((now - start) >= wait_deadline or self._recv_thread_exc is not None)
                else 0.0
            )
            abort = torch.tensor([local_abort], device=device)
            torch.distributed.all_reduce(abort, op=torch.distributed.ReduceOp.MAX)
            if abort.item() >= 1.0:
                reason = (
                    f"recv thread crash: {self._recv_thread_exc!r}"
                    if self._recv_thread_exc is not None
                    else (
                        f"initial-timeout after {now - start:.1f}s"
                        if received == 0
                        else f"timeout after {now - start:.1f}s"
                    )
                )
                self._staleness_readiness = {
                    "staleness_received_trajectories": received,
                    "staleness_global_retry_rounds": global_retry,
                    "staleness_wait_seconds": now - start,
                }
                raise RuntimeError(
                    f"rollout store readiness abort rank={self._rank} "
                    f"version={self.version} {reason} global_retry={global_retry}"
                )

            # Phase 1: every rank has >= n candidates?
            has = torch.tensor([1.0 if len(candidates) >= n else 0.0], device=device)
            torch.distributed.all_reduce(has, op=torch.distributed.ReduceOp.MIN)
            if has.item() < 1.0:
                if now - last_status >= status_interval:
                    self._log_wait_status(
                        start, global_retry, candidates, cutoff, "phase1 wait-candidate"
                    )
                    last_status = now
                await asyncio.sleep(1)
                continue

            # Phase 2: an intentionally all-masked no-op shard is usable. Only a
            # shard that has trainable chunks but no sufficiently fresh chunks is
            # stale and must be discarded. This keeps skip-zero-success windows
            # from being confused with stale data and deadlocking production.
            local_stats = count_fresh_chunks(candidates, cutoff)
            local_usable = float(
                candidate_batch_is_usable(local_stats, min_fresh_chunks)
            )
            usable = torch.tensor([local_usable], device=device)
            torch.distributed.all_reduce(usable, op=torch.distributed.ReduceOp.MIN)
            if usable.item() < 1.0:
                # All ranks have candidates but some rank lacks the minimum
                # fresh chunk budget: discard in lockstep and wait for fresher
                # data so critic statistics are defined on >1 sample.
                self.rollout_store.discard_topn(n)
                await self._return_rollout_credits(n, reason="stale_discard")
                global_retry += 1
                if now - last_status >= status_interval:
                    self._log_wait_status(
                        start,
                        global_retry,
                        candidates,
                        cutoff,
                        f"phase2 insufficient-fresh<{min_fresh_chunks} discard",
                    )
                    last_status = now
                await asyncio.sleep(1)
                continue

            batch = self.rollout_store.take_topn(n)
            await self._return_rollout_credits(len(batch), reason="consumed")
            self._staleness_readiness = {
                "staleness_received_trajectories": len(batch),
                "staleness_global_retry_rounds": global_retry,
                "staleness_wait_seconds": time.time() - start,
            }
            self.log_info(
                f"readiness ready rank={self._rank} version={self.version} "
                f"took={len(batch)} global_retry={global_retry} "
                f"wait_s={time.time() - start:.1f} "
                f"min_fresh_chunks={min_fresh_chunks}"
            )
            return batch

    async def _return_rollout_credits(self, count: int, *, reason: str) -> None:
        """Return global window credits after a collective store disposition."""
        if self._rank != 0 or self._rollout_credit_channel is None or count <= 0:
            return
        rollout_ws = self._component_placement.get_world_size("rollout")
        works = []
        for rollout_rank in range(rollout_ws):
            key = CommMapper.build_channel_key(
                rollout_rank, rollout_rank, "rollout_credit"
            )
            works.append(
                self._rollout_credit_channel.put(count, key=key, async_op=True)
            )
        await asyncio.gather(*(work.async_wait() for work in works))
        self.log_info(
            f"returned rollout credits count={count} reason={reason} "
            f"rollout_world_size={rollout_ws}"
        )

    def _count_fresh_chunks(self, candidates, cutoff: int) -> int:
        """Count fresh chunk-steps across candidate trajectories (readiness phase 2)."""
        return count_fresh_chunks(candidates, cutoff)["fresh"]

    def _log_wait_status(self, start, global_retry, candidates, cutoff, stage):
        now = time.time()
        alive = (
            self._recv_rollout_thread.is_alive()
            if getattr(self, "_recv_rollout_thread", None) is not None
            else False
        )
        stats = count_fresh_chunks(candidates, cutoff)
        self.log_info(
            f"readiness {stage} rank={self._rank} version={self.version} "
            f"wait_s={now - start:.1f} global_retry={global_retry} "
            f"recv_alive={alive} candidates={len(candidates)} "
            f"recv_queue={self._recv_queue.qsize() if self._recv_queue else 0} "
            f"store={len(self.rollout_store)} "
            f"trainable_chunks={stats['trainable']} fresh_chunks={stats['fresh']} "
            f"stale_chunks={stats['stale']} "
            f"v_min={stats['version_min']} v_mean={stats['version_mean']:.3f} "
            f"v_max={stats['version_max']}"
        )

    def _compute_staleness_mask(self) -> dict:
        """AND the low-level loss_mask with a per-chunk-step freshness mask.

        Stale chunks (version < actor_version - staleness_threshold) are masked
        out of loss + reward aggregation, but the episode stays whole for GAE.
        Stores effective low-level + chunk masks on rollout_batch; see
        ``rlinf.data.staleness_mask.compute_staleness_mask`` for details.
        """
        return compute_staleness_mask(
            self.rollout_batch,
            self.version,
            self.cfg.algorithm.get("staleness_threshold", None),
        )

    @Worker.timer("construct_rollout_batch")
    async def construct_rollout_batch(self, max_trajectories: int | None = None):
        # from _recv_queue to rollout_batch
        rollout_batch = await self._wait_for_rollout_store_ready()
        if (
            self.cfg.algorithm.get("staleness_filter_mode", "trajectory")
            != "chunk_mask"
        ):
            # trajectory mode: the wait is per-rank, so the barrier is the
            # cross-rank sync point before FSDP training. chunk_mask mode's
            # two-phase collective already synchronized all ranks.
            torch.distributed.barrier()

        version_metrics = self.rollout_store.get_metric()
        self.log_info(f"rollout store version metrics={version_metrics}")

        staleness_metrics: dict = {}
        for version_val, stats in version_metrics.items():
            if version_val == "discarded_unused":
                staleness_metrics["discarded_unused_trajs"] = stats
                continue
            diff = int(self.version) - int(version_val)
            staleness_metrics[f"data_staleness_{diff}/ratio"] = stats["ratio"]

        self.rollout_batch = convert_trajectories_to_batch(rollout_batch)
        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)
        if (
            self.cfg.algorithm.get("staleness_filter_mode", "trajectory")
            == "chunk_mask"
        ):
            staleness_metrics.update(self._compute_staleness_mask())
        if getattr(self, "_staleness_readiness", None):
            staleness_metrics.update(self._staleness_readiness)
        self.log_info(f"staleness metrics={staleness_metrics}")
        return staleness_metrics

    @torch.inference_mode()
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        proximal_values = self.rollout_batch.get("proximal_values", None)
        prev_values = self.rollout_batch.get("prev_values", None)
        initial_loss_mask = self.rollout_batch.get("loss_mask", None)
        masked_in_before_staleness = float("nan")
        if initial_loss_mask is not None:
            masked_in_before_staleness = float(
                initial_loss_mask.to(dtype=torch.bool).all(dim=-1).sum().item()
            )
        reward_metrics = compute_embodied_reward_metrics(
            self.rollout_batch["rewards"],
            initial_loss_mask,
            reward_type=self.cfg.algorithm.reward_type,
            chunk_reward_aggregation=self.cfg.algorithm.get(
                "chunk_reward_aggregation", "sum"
            ),
            gamma=float(self.cfg.algorithm.get("gamma", 1.0)),
        )

        # critic_free: force values=None so GAE's critic_free branch fires
        # (gamma=lambda=1, advantage = cumulative reward = REINFORCE). Used by
        # the delta+critic_free fallback. use_oracle_value: prev_values holds
        # injected robometer progress (env_worker), used as the GAE value.
        critic_free = self.cfg.algorithm.get("critic_free", False)
        if critic_free:
            gae_values = None
        else:
            gae_values = (
                proximal_values if proximal_values is not None else prev_values
            )
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": gae_values,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "chunk_reward_aggregation": self.cfg.algorithm.get(
                "chunk_reward_aggregation", "sum"
            ),
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
            # Wire normalize_returns from cfg to the GAE (advantages.py:83). Default
            # False; set True in config so returns -> O(1), which keeps value_loss
            # O(1) and prevents the value_head grad from dominating the merged
            # clip_grad_norm (which starves the actor's policy grad).
            "normalize_returns": self.cfg.algorithm.get("normalize_returns", False),
        }

        adv_and_ret = calculate_adv_and_returns(**kwargs)
        self.rollout_batch.update(adv_and_ret)

        if (
            self.cfg.algorithm.get("staleness_filter_mode", "trajectory")
            == "chunk_mask"
        ):
            # GAE consumed the effective low-level mask for reward aggregation;
            # expose the chunk-level effective mask to policy/value/entropy loss
            # (preprocess_loss_inputs flattens the mask, so a low-level mask would
            # mismatch the logprob target_shape).
            self.rollout_batch["loss_mask"] = self.rollout_batch[
                "staleness_chunk_loss_mask"
            ]
            self.rollout_batch["loss_mask_sum"] = self.rollout_batch[
                "staleness_chunk_loss_mask_sum"
            ]
        else:
            if kwargs["loss_mask"] is not None:
                self.rollout_batch["loss_mask"] = kwargs["loss_mask"]
            if kwargs["loss_mask_sum"] is not None:
                self.rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        rollout_metrics.update(reward_metrics)
        final_loss_mask = self.rollout_batch.get("loss_mask", None)
        masked_in_after_staleness = float("nan")
        if final_loss_mask is not None:
            masked_in_after_staleness = float(
                final_loss_mask.to(dtype=torch.bool).all(dim=-1).sum().item()
            )
        rollout_metrics.update(
            {
                "chunk_funnel/actor_critic_shared_loss_mask": 1.0,
                "chunk_funnel/masked_in_chunks_before_staleness": (
                    masked_in_before_staleness
                ),
                "chunk_funnel/masked_in_chunks_after_staleness": (
                    masked_in_after_staleness
                ),
                "chunk_funnel/masked_out_chunks_reason_staleness_filtered": (
                    masked_in_before_staleness - masked_in_after_staleness
                    if not math.isnan(masked_in_before_staleness)
                    and not math.isnan(masked_in_after_staleness)
                    else float("nan")
                ),
            }
        )
        return rollout_metrics

    @torch.inference_mode()
    def compute_proximal_logprobs(self) -> None:
        assert not self.is_weight_offloaded, (
            "Weight offloading is not supported when recomputing proximal logprobs."
        )

        t_dim = self.rollout_batch["prev_logprobs"].shape[0]
        b_dim = self.rollout_batch["prev_logprobs"].shape[1]

        flat = flatten_rollout_batch_for_train(self.rollout_batch, shuffle_id=None)
        total = flat["prev_logprobs"].shape[0]
        micro_batch_size = self.cfg.actor.micro_batch_size
        num_splits = (total + micro_batch_size - 1) // micro_batch_size

        iterator = split_dict_to_chunk(flat, num_splits)

        self.model.eval()
        proximal_logprobs_list = []

        for micro_batch in iterator:
            micro_batch = put_tensor_device(micro_batch, self.device)
            forward_inputs = micro_batch.get("forward_inputs", None)
            if forward_inputs is None:
                raise ValueError(
                    "Missing forward_inputs in compute_proximal_logprobs. "
                    "This usually means batch splitting dropped nested dict fields."
                )

            model_kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                model_kwargs["temperature"] = (
                    self.cfg.rollout.sampling_params.temperature_train
                )
                model_kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
            elif SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.GR00T,
                SupportedModel.ABOT_M0,
            ]:
                model_kwargs["prev_logprobs"] = micro_batch["prev_logprobs"]

            out = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=False,
                use_cache=False,
                **model_kwargs,
            )
            proximal_logprobs_list.append(out["logprobs"].cpu())

        flat_proximal_logprobs = torch.cat(proximal_logprobs_list, dim=0)
        proximal_logprobs = flat_proximal_logprobs.view(
            t_dim,
            b_dim,
            *self.rollout_batch["prev_logprobs"].shape[2:],
        )
        self.rollout_batch["proximal_logprobs"] = proximal_logprobs
        self._preupdate_logprob_metrics = compute_preupdate_logprob_mismatch_metrics(
            proximal_logprobs=flat_proximal_logprobs,
            behavior_logprobs=flat["prev_logprobs"],
            loss_mask=flat.get("loss_mask", None),
            logprob_type=self.cfg.algorithm.logprob_type,
            single_action_dim=self.cfg.actor.model.get("action_dim", 7),
            clip_ratio_low=float(self.cfg.algorithm.clip_ratio_low),
            clip_ratio_high=float(self.cfg.algorithm.clip_ratio_high),
        )

    @torch.inference_mode()
    def compute_reference_logprobs(self) -> None:
        """Compute frozen-SFT logprobs without retaining a second GPU model."""
        if self.kl_beta <= 0:
            return
        assert self.ref_policy_state_dict is not None, (
            "Reference KL is enabled but the frozen reference weights are missing."
        )
        assert not self.is_weight_offloaded, (
            "Reference-KL recomputation does not support weight offloading."
        )

        total = self.rollout_batch["prev_logprobs"].shape[0]
        micro_batch_size = self.cfg.actor.micro_batch_size
        num_splits = (total + micro_batch_size - 1) // micro_batch_size
        iterator = split_dict_to_chunk(self.rollout_batch, num_splits)
        reference_logprobs_list = []

        self.model.eval()
        with cpu_weight_swap(
            self.model, self.ref_policy_state_dict, self.offload_model_buffer
        ):
            for micro_batch in iterator:
                micro_batch = put_tensor_device(micro_batch, self.device)
                forward_inputs = micro_batch.get("forward_inputs", None)
                if forward_inputs is None:
                    raise ValueError(
                        "Missing forward_inputs in compute_reference_logprobs."
                    )
                out = self.model(
                    forward_inputs=forward_inputs,
                    compute_logprobs=True,
                    compute_entropy=False,
                    compute_values=False,
                    use_cache=False,
                )
                reference_logprobs_list.append(out["logprobs"].cpu())

        self.rollout_batch["reference_logprobs"] = torch.cat(
            reference_logprobs_list, dim=0
        )

    @torch.inference_mode()
    def compute_post_update_outputs(
        self,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Recompute policy and critic outputs at one post-update snapshot."""
        assert not self.is_weight_offloaded, (
            "Weight offloading is not supported when recomputing post-update logprobs."
        )

        flat = self.rollout_batch
        total = flat["prev_logprobs"].shape[0]
        micro_batch_size = self.cfg.actor.micro_batch_size
        num_splits = (total + micro_batch_size - 1) // micro_batch_size
        iterator = split_dict_to_chunk(flat, num_splits)

        self.model.eval()
        post_update_logprobs_list = []
        post_update_values_list = []
        # Skip the value-head forward when using an oracle value (robometer
        # progress injected as prev_values by env_worker) -- no critic to update.
        compute_values = self.cfg.algorithm.adv_type == "gae" and not (
            self.cfg.algorithm.get("use_oracle_value", False)
        )

        for micro_batch in iterator:
            micro_batch = put_tensor_device(micro_batch, self.device)
            forward_inputs = micro_batch.get("forward_inputs", None)
            if forward_inputs is None:
                raise ValueError(
                    "Missing forward_inputs in compute_post_update_outputs. "
                    "This usually means batch splitting dropped nested dict fields."
                )

            model_kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                model_kwargs["temperature"] = (
                    self.cfg.rollout.sampling_params.temperature_train
                )
                model_kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
            elif SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.GR00T,
                SupportedModel.ABOT_M0,
            ]:
                model_kwargs["prev_logprobs"] = micro_batch["prev_logprobs"]

            out = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=compute_values,
                use_cache=False,
                **model_kwargs,
            )
            post_update_logprobs_list.append(out["logprobs"].cpu())
            if compute_values:
                post_update_values_list.append(out["values"].cpu())

        post_update_values = (
            torch.cat(post_update_values_list, dim=0) if compute_values else None
        )
        return torch.cat(post_update_logprobs_list, dim=0), post_update_values

    def compute_global_explained_variance_metrics(
        self, post_update_values: torch.Tensor
    ) -> dict[str, float]:
        """Compute one EV over the complete post-update batch on all actor ranks."""
        statistics = compute_explained_variance_statistics(
            returns=self.rollout_batch["returns"],
            values=post_update_values,
            loss_mask=self.rollout_batch.get("loss_mask", None),
        ).to(self.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(statistics, op=torch.distributed.ReduceOp.SUM)
        metrics = explained_variance_metrics_from_statistics(statistics)
        return {key: float(value.item()) for key, value in metrics.items()}

    def _measure_global_grad_norm(self, loss: torch.Tensor) -> float:
        """Backward one sampled loss and return its unclipped global grad norm."""
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward(retain_graph=True)
        scale = float(self.grad_scaler.get_scale())
        squared_norm = torch.zeros((), device=self.device, dtype=torch.float64)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                squared_norm += parameter.grad.detach().double().square().sum()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                squared_norm, op=torch.distributed.ReduceOp.SUM
            )
        grad_norm = float(torch.sqrt(squared_norm).item()) / scale
        self.optimizer.zero_grad()
        return grad_norm

    def run_training(self) -> dict[str, Any]:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        t_dim = int(self.rollout_batch["prev_logprobs"].shape[0])
        b_dim = int(self.rollout_batch["prev_logprobs"].shape[1])
        total_samples = t_dim * b_dim

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.cfg.actor.seed) + int(self._rank))
        shuffle_id = torch.randperm(total_samples, generator=generator)

        with torch.no_grad():
            self.rollout_batch = flatten_rollout_batch_for_train(
                self.rollout_batch, shuffle_id
            )

        if self.cfg.algorithm.normalize_advantages:
            self.rollout_batch["advantages"] = masked_normalization(
                self.rollout_batch["advantages"],
                self.rollout_batch.get("loss_mask", None),
            )

        # The batch is shuffled before this call, so reference logprobs share the
        # exact loss-mask and sample order used by every optimizer microbatch.
        self.compute_reference_logprobs()

        self.model.train()

        world_size = int(self._world_size)
        global_batch_size = int(self.cfg.actor.global_batch_size)
        micro_batch_size = int(self.cfg.actor.micro_batch_size)

        assert global_batch_size % (micro_batch_size * world_size) == 0, (
            f"global_batch_size {global_batch_size} must be divisible by "
            f"micro_batch_size {micro_batch_size} * world_size {world_size}"
        )

        per_rank_batch_size = global_batch_size // world_size
        micro_per_rank = per_rank_batch_size // micro_batch_size
        self.gradient_accumulation = micro_per_rank

        flattened_rollout_size = int(self.rollout_batch["prev_logprobs"].shape[0])
        assert flattened_rollout_size % per_rank_batch_size == 0, (
            f"Flattened rollout size {flattened_rollout_size} must be divisible by "
            f"per-rank batch size {per_rank_batch_size}"
        )
        num_global_batches = flattened_rollout_size // per_rank_batch_size

        metrics: dict[str, list] = {}
        for key, value in getattr(self, "_preupdate_logprob_metrics", {}).items():
            append_to_dict(metrics, {key: value})
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))

        for _ in range(update_epoch):
            global_batch_iter = split_dict_to_chunk(
                self.rollout_batch,
                num_global_batches,
            )

            for global_batch_idx, train_global_batch in enumerate(global_batch_iter):
                train_global_batch_size = int(
                    train_global_batch["prev_logprobs"].shape[0]
                )
                assert train_global_batch_size == per_rank_batch_size, (
                    f"Expected per-rank global batch size {per_rank_batch_size}, "
                    f"got {train_global_batch_size}"
                )
                assert train_global_batch_size % micro_batch_size == 0

                micro_batch_iter = split_dict_to_chunk(
                    train_global_batch,
                    micro_per_rank,
                )

                self.optimizer.zero_grad()

                for mb_idx, data in enumerate(micro_batch_iter):
                    data = put_tensor_device(
                        data,
                        f"cuda:{int(os.environ['LOCAL_RANK'])}",
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(mb_idx + 1) == self.gradient_accumulation,
                    )

                    advantages = data["advantages"]
                    old_logprobs = data["prev_logprobs"]
                    returns = data.get("returns", None)
                    prev_values = data.get("prev_values", None)
                    loss_mask = data.get("loss_mask", None)
                    loss_mask_sum = data.get("loss_mask_sum", None)

                    versions = data.get("versions", None)
                    proximal_logprobs = data.get("proximal_logprobs", None)
                    reference_logprobs = data.get("reference_logprobs", None)
                    proximal_values = data.get("proximal_values", None)
                    current_version = int(self.version) + 1

                    forward_inputs = data.get("forward_inputs", None)
                    if forward_inputs is None:
                        raise ValueError(
                            "Missing forward_inputs in run_training. "
                            "This usually means batch splitting dropped nested dict fields."
                        )

                    model_kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        model_kwargs["temperature"] = (
                            self.cfg.rollout.sampling_params.temperature_train
                        )
                        model_kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
                    elif SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T,
                        SupportedModel.ABOT_M0,
                    ]:
                        model_kwargs["prev_logprobs"] = old_logprobs

                    compute_values = self.cfg.algorithm.adv_type == "gae" and not (
                        self.cfg.algorithm.get("use_oracle_value", False)
                    )

                    with self.amp_context:
                        out = self.model(
                            forward_inputs=forward_inputs,
                            compute_logprobs=True,
                            compute_entropy=(self.cfg.algorithm.entropy_bonus > 0),
                            compute_values=compute_values,
                            use_cache=False,
                            **model_kwargs,
                        )

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T,
                        SupportedModel.ABOT_M0,
                    ]:
                        old_logprobs = out["prev_logprobs"]

                    loss_kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": out["logprobs"],
                        "values": out.get("values", None),
                        "old_logprobs": old_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": proximal_values
                        if proximal_values is not None
                        else prev_values,
                        "proximal_logprobs": proximal_logprobs,
                        "versions": versions,
                        "current_version": current_version,
                        "behave_weight_threshold": self.cfg.algorithm.get(
                            "behave_weight_threshold", None
                        ),
                        "clip_ratio_c": self.cfg.algorithm.get("clip_ratio_c", 3.0),
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }

                    loss_components: dict[str, torch.Tensor] = {}
                    loss_kwargs["loss_components"] = loss_components
                    loss, metrics_data = policy_loss(**loss_kwargs)

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not loss_kwargs["critic_warmup"]
                    ):
                        entropy = out["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=out["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                    reference_kl = torch.tensor(0.0, device=out["logprobs"].device)
                    reference_kl_penalty = torch.tensor(
                        0.0, device=out["logprobs"].device
                    )
                    if self.kl_beta > 0 and not loss_kwargs["critic_warmup"]:
                        if reference_logprobs is None:
                            raise RuntimeError(
                                "Reference KL is enabled but reference logprobs "
                                "were not computed."
                            )
                        reference_logprobs = reference_logprobs.to(
                            device=out["logprobs"].device,
                            dtype=out["logprobs"].dtype,
                        )
                        # logprobs and loss_mask can carry different trailing
                        # action dimensions (e.g. joint flow logprobs versus
                        # per-action validity). Reduce each to PPO's item
                        # granularity before masking instead of relying on
                        # incompatible tensor broadcasting.
                        reference_kl_items = _reduce_to_item_level(
                            kl_penalty(
                                out["logprobs"],
                                reference_logprobs,
                                self.kl_penalty_type,
                            ),
                            self.cfg.algorithm.logprob_type,
                            self.cfg.actor.model.get("action_dim", 7),
                            reduction="mean",
                        )
                        reference_mask_items = (
                            torch.ones_like(reference_kl_items, dtype=torch.bool)
                            if loss_mask is None
                            else _reduce_to_item_level(
                                loss_mask.bool(),
                                self.cfg.algorithm.logprob_type,
                                self.cfg.actor.model.get("action_dim", 7),
                                reduction="any",
                            ).bool()
                        )
                        reference_kl = masked_mean(
                            reference_kl_items, mask=reference_mask_items
                        )
                        reference_kl_penalty = self.kl_beta * reference_kl
                        loss = loss + reference_kl_penalty

                    loss = loss / self.gradient_accumulation
                    grad_diagnostics_interval = int(
                        self.cfg.actor.get("grad_diagnostics_interval", 0)
                    )
                    run_grad_diagnostics = (
                        grad_diagnostics_interval > 0
                        and not loss_kwargs["critic_warmup"]
                        and current_version % grad_diagnostics_interval == 0
                        and global_batch_idx == 0
                        and mb_idx == 0
                    )
                    if run_grad_diagnostics:
                        raise RuntimeError(
                            "Sampled gradient conflict diagnostics are unsupported "
                            "with FSDP: repeated backward calls on a retained graph "
                            "can invalidate flat-parameter storage. Set "
                            "actor.grad_diagnostics_interval to 0."
                        )
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/entropy_loss"] = float(
                        entropy_loss.detach().item()
                    )
                    metrics_data["actor/reference_kl"] = float(
                        reference_kl.detach().item()
                    )
                    metrics_data["actor/reference_kl_penalty"] = float(
                        reference_kl_penalty.detach().item()
                    )
                    metrics_data["actor/reference_kl_beta"] = float(self.kl_beta)
                    metrics_data["actor/total_loss"] = float(loss.detach().item())
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                extra_metrics = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    extra_metrics["critic/lr"] = lr_list[1]
                append_to_dict(metrics, extra_metrics)

        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        post_update_logprobs, post_update_values = self.compute_post_update_outputs()
        post_update_metrics = compute_policy_adv_logprob_diagnostics(
            advantages=self.rollout_batch["advantages"],
            proximal_logprobs=self.rollout_batch["proximal_logprobs"],
            post_update_logprobs=post_update_logprobs,
            loss_mask=self.rollout_batch.get("loss_mask", None),
            logprob_type=self.cfg.algorithm.logprob_type,
            single_action_dim=self.cfg.actor.model.get("action_dim", 7),
        )
        post_update_metrics.update(
            compute_post_update_proximal_metrics(
                proximal_logprobs=self.rollout_batch["proximal_logprobs"],
                post_update_logprobs=post_update_logprobs,
                loss_mask=self.rollout_batch.get("loss_mask", None),
                logprob_type=self.cfg.algorithm.logprob_type,
                single_action_dim=self.cfg.actor.model.get("action_dim", 7),
                clip_ratio_low=self.cfg.algorithm.clip_ratio_low,
                clip_ratio_high=self.cfg.algorithm.clip_ratio_high,
            )
        )
        post_update_metrics.update(
            compute_post_update_ppo_surrogate_metrics(
                advantages=self.rollout_batch["advantages"],
                old_logprobs=self.rollout_batch["prev_logprobs"],
                proximal_logprobs=self.rollout_batch["proximal_logprobs"],
                post_update_logprobs=post_update_logprobs,
                loss_mask=self.rollout_batch.get("loss_mask", None),
                loss_mask_sum=self.rollout_batch.get("loss_mask_sum", None),
                max_episode_steps=self.cfg.env.train.max_episode_steps,
                logprob_type=self.cfg.algorithm.logprob_type,
                single_action_dim=self.cfg.actor.model.get("action_dim", 7),
                reward_type=self.cfg.algorithm.reward_type,
                clip_ratio_low=self.cfg.algorithm.clip_ratio_low,
                clip_ratio_high=self.cfg.algorithm.clip_ratio_high,
                clip_ratio_c=self.cfg.algorithm.get("clip_ratio_c", 3.0),
                behave_weight_threshold=self.cfg.algorithm.get(
                    "behave_weight_threshold", None
                ),
            )
        )
        append_to_dict(metrics, post_update_metrics)

        if post_update_values is not None:
            global_ev_metrics = self.compute_global_explained_variance_metrics(
                post_update_values
            )
            # EV is nonlinear and must not be averaged over optimizer
            # micro-batches. Replace all local samples with the one global value.
            for key, value in global_ev_metrics.items():
                metrics[key] = [value]

        clear_memory()

        mean_metric_dict = {k: float(np.mean(v)) for k, v in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict,
            op=torch.distributed.ReduceOp.AVG,
            validate_schema=True,
        )
        return mean_metric_dict
