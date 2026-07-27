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

import math
import os
import time

import numpy as np
import torch
import torch.distributed


def compute_split_num(num, split_num):
    return math.lcm(num, split_num) // split_num


def _normalize_metric_shard(shard: object) -> torch.Tensor:
    """One rank's metric -> 1D float tensor on CPU."""
    if shard is None:
        return torch.tensor([], dtype=torch.float32)
    if isinstance(shard, torch.Tensor):
        return shard.detach().cpu().reshape(-1).float()
    if isinstance(shard, list):
        if not shard:
            return torch.tensor([], dtype=torch.float32)
        return torch.cat([x.detach().cpu().reshape(-1).float() for x in shard], dim=0)
    return torch.as_tensor(shard, dtype=torch.float32).cpu().reshape(-1)


def count_trajectories(metrics_dict):
    """
    Count the total number of trajectories from metrics dictionary.

    Args:
        metrics_dict: Dictionary of metrics where each value is a tensor after concatenation.
                     Each tensor's first dimension represents the number of trajectories.

    Returns:
        int: Total number of trajectories. If metrics_dict is empty, returns 0.
    """
    if not metrics_dict:
        return 0

    # Use the first metric tensor to get the trajectory count
    # All metrics should have the same first dimension (number of trajectories)
    first_key = next(iter(metrics_dict.keys()))
    first_tensor = metrics_dict[first_key]

    if isinstance(first_tensor, torch.Tensor):
        return first_tensor.shape[0]
    elif isinstance(first_tensor, list):
        # If it's a list of tensors, sum up all trajectory counts
        return sum(
            t.shape[0] if isinstance(t, torch.Tensor) else len(t) for t in first_tensor
        )
    else:
        raise TypeError(f"Unsupported tensor type: {type(first_tensor)}")


def compute_evaluate_metrics(eval_metrics_list):
    """
    List of evaluate metrics, list length stands for rollout process

    Returns:
        dict: Aggregated metrics with mean values and trajectory count
    """
    if not eval_metrics_list:
        return {}

    all_eval_metrics = {}
    env_info_keys: set[str] = set()
    for eval_metrics in eval_metrics_list:
        env_info_keys.update(eval_metrics.keys())

    # Count trajectories from each process
    trajectory_counts = []
    for eval_metrics in eval_metrics_list:
        count = count_trajectories(eval_metrics)
        trajectory_counts.append(count)

    for env_info_key in env_info_keys:
        metric = [
            eval_metrics[env_info_key]
            for eval_metrics in eval_metrics_list
            if env_info_key in eval_metrics
        ]
        if metric:
            all_eval_metrics[env_info_key] = metric

    for key in all_eval_metrics:
        shards = [_normalize_metric_shard(s) for s in all_eval_metrics[key]]
        stacked = torch.concat(shards).float()
        all_eval_metrics[key] = (
            stacked.mean().detach().cpu().numpy()
            if stacked.numel() > 0
            else np.asarray(0.0, dtype=np.float64)
        )

    # Add total trajectory count to metrics
    all_eval_metrics["num_trajectories"] = sum(trajectory_counts)

    return all_eval_metrics


def embodied_reward_metric_values(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor | None,
    *,
    reward_type: str,
    chunk_reward_aggregation: str,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Build masked reward values at each level of the PPO reward pipeline."""
    low_level_mask = (
        torch.ones_like(rewards, dtype=torch.bool)
        if loss_mask is None
        else loss_mask.to(device=rewards.device, dtype=torch.bool)
    )
    if low_level_mask.shape != rewards.shape:
        low_level_mask = torch.broadcast_to(low_level_mask, rewards.shape)
    low_level_values = rewards[low_level_mask]

    if reward_type == "chunk_level":
        # Lazy import: rlinf.algorithms.utils -> algorithms/__init__ -> advantages
        # -> rlinf.utils.utils would form a circular import at module load time
        # (metric_utils is imported by utils.utils). Import at call time instead.
        from rlinf.algorithms.utils import aggregate_embodied_chunk_rewards

        chunk_rewards = aggregate_embodied_chunk_rewards(
            rewards,
            low_level_mask,
            gamma=gamma,
            aggregation=chunk_reward_aggregation,
        )
        chunk_mask = low_level_mask.any(dim=-1, keepdim=True)
    else:
        chunk_rewards = rewards
        chunk_mask = low_level_mask

    chunk_values = chunk_rewards[chunk_mask]
    episode_mask = chunk_mask.reshape(chunk_mask.shape[0], chunk_mask.shape[1], -1).any(
        dim=(0, 2)
    )
    episode_sums = (
        (chunk_rewards * chunk_mask.to(dtype=chunk_rewards.dtype))
        .reshape(chunk_rewards.shape[0], chunk_rewards.shape[1], -1)
        .sum(dim=(0, 2))
    )
    episode_sums = episode_sums[episode_mask]
    return {
        "reward_low_level": low_level_values,
        "reward_chunk": chunk_values,
        "reward_episode_sum": episode_sums,
    }


def _distributed_reward_stats(values: torch.Tensor) -> dict[str, float]:
    from rlinf.scheduler.worker.worker import Worker

    device = Worker.torch_platform.current_device()
    values = values.detach().to(device=device, dtype=torch.float32).reshape(-1)
    if values.numel() == 0:
        local_sum = torch.tensor(0.0, device=device)
        local_count = torch.tensor(0.0, device=device)
        local_positive = torch.tensor(0.0, device=device)
        local_min = torch.tensor(float("inf"), device=device)
        local_max = torch.tensor(float("-inf"), device=device)
    else:
        local_sum = values.sum()
        local_count = torch.tensor(float(values.numel()), device=device)
        local_positive = (values > 0).to(dtype=torch.float32).sum()
        local_min = values.min()
        local_max = values.max()

    totals = torch.stack([local_sum, local_count, local_positive])
    extrema = torch.stack([-local_min, local_max])
    torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(extrema, op=torch.distributed.ReduceOp.MAX)
    total_sum, total_count, total_positive = totals.tolist()
    if total_count <= 0:
        return {
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "positive_fraction": float("nan"),
        }
    return {
        "mean": total_sum / total_count,
        "min": -extrema[0].item(),
        "max": extrema[1].item(),
        "positive_fraction": total_positive / total_count,
    }


def compute_embodied_reward_metrics(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor | None,
    *,
    reward_type: str,
    chunk_reward_aggregation: str,
    gamma: float,
) -> dict[str, float]:
    """Compute distributed scalar metrics for rewards actually consumed by PPO."""
    metric_values = embodied_reward_metric_values(
        rewards,
        loss_mask,
        reward_type=reward_type,
        chunk_reward_aggregation=chunk_reward_aggregation,
        gamma=gamma,
    )
    metrics = {}
    for prefix, values in metric_values.items():
        stats = _distributed_reward_stats(values)
        metrics.update({f"{prefix}_{name}": value for name, value in stats.items()})
    return metrics


def compute_rollout_metrics(data_buffer: dict) -> dict:
    rollout_metrics = {}
    loss_mask = data_buffer.get("loss_mask", None)

    def reduce_metrics(values: torch.Tensor) -> tuple[float, float, float]:
        from rlinf.scheduler.worker.worker import Worker

        device = Worker.torch_platform.current_device()

        if values.numel() == 0:
            count = torch.tensor(0.0, device=device, dtype=torch.float32)
            values_sum = torch.tensor(0.0, device=device, dtype=torch.float32)
            min_value = float("inf")
            max_value = float("-inf")
        else:
            values = values.to(device)
            count = torch.tensor(
                values.numel(), device=values.device, dtype=torch.float32
            )
            values_sum = values.to(dtype=torch.float32).sum()
            max_value = torch.max(values).detach().item()
            min_value = torch.min(values).detach().item()

        reduce_sum_count = torch.stack([values_sum, count])
        reduce_min_max = torch.as_tensor(
            [-min_value, max_value],
            device=device,
            dtype=torch.float32,
        )
        torch.distributed.all_reduce(
            reduce_sum_count, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(reduce_min_max, op=torch.distributed.ReduceOp.MAX)
        reduced_sum, reduced_count = reduce_sum_count.tolist()
        reduced_min, reduced_max = reduce_min_max.tolist()

        if reduced_count <= 0:
            return float("nan"), float("nan"), float("nan")
        return reduced_sum / reduced_count, -reduced_min, reduced_max

    def valid_values(values: torch.Tensor) -> torch.Tensor:
        if loss_mask is None:
            return values.reshape(-1)
        mask = loss_mask.to(device=values.device, dtype=torch.bool)
        if mask.shape != values.shape:
            mask = torch.broadcast_to(mask, values.shape)
        return values[mask]

    if "rewards" in data_buffer:
        rewards = data_buffer["rewards"]
        rewards = valid_values(rewards)
        mean_rewards, _, _ = reduce_metrics(rewards)

        rewards_metrics = {
            "rewards": mean_rewards,
        }
        rollout_metrics.update(rewards_metrics)

    if "advantages" in data_buffer:
        advantages = data_buffer["advantages"]
        advantages = valid_values(advantages)
        mean_adv, min_adv, max_adv = reduce_metrics(advantages)

        advantages_metrics = {
            "advantages_mean": mean_adv,
            "advantages_max": max_adv,
            "advantages_min": min_adv,
        }
        rollout_metrics.update(advantages_metrics)

    if data_buffer.get("returns", None) is not None:
        returns = data_buffer["returns"]
        returns = valid_values(returns)
        mean_ret, min_ret, max_ret = reduce_metrics(returns)

        returns_metrics = {
            "returns_mean": mean_ret,
            "returns_max": max_ret,
            "returns_min": min_ret,
        }
        rollout_metrics.update(returns_metrics)

    # Total training-data env steps this rollout = # of loss_mask=True chunks
    # (down-sampled insert+hasReward only; pick-up + non-sampled chunks are
    # excluded from the loss). All-reduced SUM across actor ranks so every
    # rank returns the same global total (runner then sums the per-rank list,
    # which is a no-op since all ranks hold the same value).
    if loss_mask is not None:
        from rlinf.scheduler.worker.worker import Worker

        _device = Worker.torch_platform.current_device()
        _lm_sum = loss_mask.to(device=_device).bool().sum()
        torch.distributed.all_reduce(_lm_sum, op=torch.distributed.ReduceOp.SUM)
        rollout_metrics["train_env_steps"] = _lm_sum.item()

    return rollout_metrics


def append_to_dict(data, new_data):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def compute_loss_mask(dones):
    _, actual_bsz, num_action_chunks = dones.shape
    n_chunk_step = dones.shape[0] - 1
    flattened_dones = dones.transpose(1, 2).reshape(
        -1, actual_bsz
    )  # [(n_chunk_step + 1) * num_action_chunks, rollout_epoch x bsz]
    flattened_dones = flattened_dones[
        -(n_chunk_step * num_action_chunks + 1) :
    ]  # [n_steps+1, actual-bsz]
    flattened_loss_mask = (flattened_dones.cumsum(dim=0) == 0)[
        :-1
    ]  # [n_steps, actual-bsz]

    loss_mask = flattened_loss_mask.reshape(n_chunk_step, num_action_chunks, actual_bsz)
    loss_mask = loss_mask.transpose(
        1, 2
    )  # [n_chunk_step, actual_bsz, num_action_chunks]

    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True)  # [1, bsz, 1]
    loss_mask_sum = loss_mask_sum.expand_as(loss_mask)

    return loss_mask, loss_mask_sum


def resolve_loss_mask(
    rollout_batch,
    auto_reset: bool,
    ignore_terminations: bool,
    chunk_level: bool,
    staleness_filter_mode: str = "trajectory",
):
    """Resolve ``(loss_mask, loss_mask_sum)``.

    A trajectory-provided ``loss_mask`` (``rollout_batch["loss_mask"]``, set by
    env_worker.assign_history_reward for robometer-progress-labeled insert chunks)
    WINS — only those chunks train (policy + value loss masked). Else build from
    ``dones`` when ``not auto_reset and not ignore_terminations`` (the legacy
    episode-boundary mask). Else ``(None, None)`` -> downstream defaults to
    all-ones. Shared by the actor (fsdp_actor_worker) + the pipeline path
    (utils.preprocess_embodied_batch) so both honor the trajectory-provided mask.
    """
    provided = rollout_batch.get("loss_mask", None)
    if provided is not None:
        loss_mask = provided
        loss_mask_sum = rollout_batch.get("loss_mask_sum", None)
        if staleness_filter_mode == "chunk_mask":
            # Keep low-level resolution; sum is recomputed after freshness masking
            # by compute_staleness_mask on the async actor.
            return loss_mask, None
        if loss_mask_sum is None:
            loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True).expand_as(loss_mask)
        if chunk_level:
            loss_mask = loss_mask.any(dim=-1, keepdim=True)
            loss_mask_sum = loss_mask_sum[..., -1:]
        return loss_mask, loss_mask_sum
    if not auto_reset and not ignore_terminations:
        dones = rollout_batch["dones"]
        loss_mask, loss_mask_sum = compute_loss_mask(dones)
        if staleness_filter_mode == "chunk_mask":
            return loss_mask, None
        if chunk_level:
            loss_mask = loss_mask.any(dim=-1, keepdim=True)
            loss_mask_sum = loss_mask_sum[..., -1:]
        return loss_mask, loss_mask_sum
    return None, None


def print_metrics_table(
    step: int,
    total_steps: int,
    start_time: float,
    metrics: dict,
    start_step: int = 0,
    log_path: str | None = None,
):
    """Print training metrics in a simple, fast formatted table.

    The rendered table is written to stdout and, when ``log_path`` is given,
    also appended to ``<log_path>/metrics.log``.
    """
    # Accumulate the table into lines so the exact same rendering goes to both
    # stdout and the log file.
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    # Calculate progress info
    progress = (step + 1) / total_steps * 100
    elapsed_time = time.time() - start_time
    steps_done = step + 1 - start_step
    eta_seconds = (
        elapsed_time / steps_done * (total_steps - step - 1) if steps_done > 0 else 0
    )

    def format_time(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    # Format elapsed time and ETA
    elapsed_str = format_time(elapsed_time)
    eta_str = format_time(eta_seconds)

    # Create progress bar
    bar_width = 40
    filled = int(bar_width * progress / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Print header with progress
    total_width = 120

    def _fit_line(text: str, width: int) -> str:
        if len(text) <= width:
            return text + (" " * (width - len(text)))
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "…"

    def _fit_cell(text: str, width: int) -> str:
        return _fit_line(text, width)

    def _print_section_title(title: str) -> None:
        title_text = f" {title} "
        padding = total_width - 2 - len(title_text)
        left = padding // 2
        right = padding - left
        emit(f"├{'─' * left}{title_text}{'─' * right}┤")

    emit(f"\n╭{'─' * (total_width - 2)}╮")
    _print_section_title("Metric Table")

    # First line: Global Step and Progress
    step_str = f"Global Step: {step + 1:4d}/{total_steps}"
    progress_str = f"Progress: {bar} │ {progress:5.1f}%"
    line1 = f"│ {step_str} │ {progress_str}"
    line1 = _fit_line(line1, total_width - 2)
    emit(f"{line1} │")

    # Second line: Time information
    elapsed_str_formatted = f"Elapsed: {elapsed_str}"
    eta_str_formatted = f"ETA: {eta_str}"
    step_time_str = f"Step Time: {elapsed_time / steps_done:.3f}s"
    line2 = f"│ {elapsed_str_formatted} │ {eta_str_formatted} │ {step_time_str}"
    line2 = _fit_line(line2, total_width - 2)
    emit(f"{line2} │")

    # Group metrics by category
    categories = {
        "Time": {},
        "Environment": {},
        "Rollout": {},
        "Evaluation": {},
        "Replay Buffer": {},
        "Training/Actor": {},
        "Training/Critic": {},
        "Training/Other": {},
    }

    for key, value in metrics.items():
        if "/" in key:
            category, metric_name = key.split("/", 1)
            category_map = {
                "time": "Time",
                "env": "Environment",
                "rollout": "Rollout",
                "eval": "Evaluation",
                "replay_buffer": "Replay Buffer",
            }
            if category in category_map:
                categories[category_map[category]][metric_name] = value
            elif category == "train":
                if metric_name.startswith("actor/"):
                    categories["Training/Actor"][metric_name] = value
                elif metric_name.startswith("critic/"):
                    categories["Training/Critic"][metric_name] = value
                elif metric_name.startswith("replay_buffer/"):
                    categories["Replay Buffer"][
                        metric_name.replace("replay_buffer/", "")
                    ] = value
                else:
                    categories["Training/Other"][metric_name] = value

    # Print metrics by category - 3 metrics per row
    table_width = total_width  # Match header width
    base_col_width = (table_width - 4) // 3
    remainder = (table_width - 4) - (base_col_width * 3)
    col_widths = [
        base_col_width + (1 if remainder > 0 else 0),
        base_col_width + (1 if remainder > 1 else 0),
        base_col_width,
    ]

    for category_name, category_metrics in categories.items():
        if category_metrics:
            _print_section_title(category_name)
            # Blank line before metrics (except Global Step section, which is separate)
            emit(f"│{' ' * (table_width - 2)}│")

            # Sort metrics for consistent output
            sorted_metrics = sorted(category_metrics.items())

            # Print in 3-column layout
            for i in range(0, len(sorted_metrics), 3):
                # Get up to 3 metrics for this row
                row_metrics = []
                for j in range(3):
                    if i + j < len(sorted_metrics):
                        metric_name, metric_value = sorted_metrics[i + j]

                        # Format value
                        if isinstance(metric_value, float):
                            if abs(metric_value) < 0.001 and metric_value != 0:
                                formatted_value = f"{metric_value:.2e}"
                            elif abs(metric_value) < 0.01:
                                formatted_value = f"{metric_value:.4f}"
                            elif abs(metric_value) > 10000:
                                formatted_value = f"{metric_value:.2e}"
                            elif abs(metric_value) > 100:
                                formatted_value = f"{metric_value:.1f}"
                            else:
                                formatted_value = f"{metric_value:.3f}"
                        else:
                            formatted_value = str(metric_value)

                        display = f"{metric_name}={formatted_value}"
                        row_metrics.append(display)
                    else:
                        row_metrics.append("")

                # Create the line with exactly 3 columns
                line = (
                    f"│{_fit_cell(row_metrics[0], col_widths[0])}"
                    f"│{_fit_cell(row_metrics[1], col_widths[1])}"
                    f"│{_fit_cell(row_metrics[2], col_widths[2])}│"
                )
                emit(line)

            # Section separator (minimal)
            emit(f"│{' ' * (table_width - 2)}│")

    # Bottom border
    emit(f"╰{'─' * (table_width - 2)}╯")

    emit()

    table = "\n".join(lines)
    print(table)
    if log_path:
        os.makedirs(log_path, exist_ok=True)
        with open(os.path.join(log_path, "metrics.log"), "a") as metrics_file:
            metrics_file.write(table + "\n")
