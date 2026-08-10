#!/usr/bin/env python3
"""Plot two PegInsertion evaluation sweeps on one success-rate chart."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two PegInsertion evaluation sweeps. The smooth curve is a "
            "centered moving average; the translucent circles remain the original "
            "evaluation samples."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--robofape-dir",
        required=True,
        help="RoboFAPE sweep directory containing wrist_sweep_metrics.csv.",
    )
    parser.add_argument(
        "--robometer-dir",
        required=True,
        help="Robometer sweep directory containing wrist_sweep_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which to write the comparison plot.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=3,
        help="Centered moving-average window in evaluated checkpoints; use 1 for raw lines.",
    )
    parser.add_argument(
        "--exclude-step",
        type=int,
        action="append",
        default=[10],
        help="PPO step to omit from both plots; step 10 is excluded by default.",
    )
    parser.add_argument("--robofape-label", default="RoboFAPE")
    parser.add_argument("--robometer-label", default="Robometer")
    return parser.parse_args()


def load_success_rates(
    sweep_dir: Path,
) -> tuple[list[int], list[float]]:
    """Read and validate success-rate samples from a sweep CSV."""
    csv_path = sweep_dir.expanduser().resolve() / "wrist_sweep_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing sweep metrics CSV: {csv_path}")

    samples: dict[int, float] = {}
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            try:
                step = int(row["step"])
                success_rate = float(row["success_rate"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid step or success_rate in {csv_path}: {row}") from exc
            if not 0.0 <= success_rate <= 1.0:
                raise ValueError(f"success_rate must be in [0, 1] in {csv_path}: {success_rate}")
            if step in samples:
                raise ValueError(f"Duplicate PPO step {step} in {csv_path}")
            samples[step] = success_rate

    if not samples:
        raise ValueError(f"No evaluation samples in {csv_path}")
    steps = sorted(samples)
    return steps, [samples[step] for step in steps]


def exclude_steps(
    steps: Sequence[int], values: Sequence[float], excluded_steps: set[int]
) -> tuple[list[int], list[float]]:
    """Remove selected samples after any baseline-derived values are computed."""
    kept = [(step, value) for step, value in zip(steps, values, strict=True) if step not in excluded_steps]
    if not kept:
        raise ValueError("All evaluation samples were excluded")
    kept_steps, kept_values = zip(*kept, strict=True)
    return list(kept_steps), list(kept_values)


def centered_moving_average(values: Sequence[float], window: int) -> list[float]:
    """Return a centered moving average while retaining edge samples."""
    if window < 1:
        raise ValueError("--smooth must be at least 1")
    if window == 1:
        return list(values)

    radius_left = (window - 1) // 2
    radius_right = window // 2
    return [
        sum(values[max(0, index - radius_left) : min(len(values), index + radius_right + 1)])
        / (min(len(values), index + radius_right + 1) - max(0, index - radius_left))
        for index in range(len(values))
    ]


def adaptive_ylim(
    series: Sequence[Sequence[float]],
    *,
    tick_step: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    include: float | None = None,
) -> tuple[float, float]:
    """Set a compact readable y-range with optional natural metric bounds."""
    values = [value for values in series for value in values]
    if include is not None:
        values.append(include)
    lower_value, upper_value = min(values), max(values)
    span = max(upper_value - lower_value, 2 * tick_step)
    padding = 0.15 * span
    lower = math.floor((lower_value - padding) / tick_step) * tick_step
    upper = math.ceil((upper_value + padding) / tick_step) * tick_step
    if lower_bound is not None:
        lower = max(lower_bound, lower)
    if upper_bound is not None:
        upper = min(upper_bound, upper)
    if lower == upper:
        upper = upper + tick_step if upper_bound is None else min(upper_bound, upper + tick_step)
    return lower, upper


def plot_series(
    ax: plt.Axes,
    steps: Sequence[int],
    values: Sequence[float],
    label: str,
    color: str,
    smooth: int,
) -> None:
    """Draw a smooth trend plus the actual checkpoint evaluation samples."""
    smoothed = centered_moving_average(values, smooth)
    ax.scatter(
        steps,
        values,
        s=36,
        color=color,
        alpha=0.38,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.plot(steps, smoothed, color=color, linewidth=2.7, label=label, zorder=4)


def success_rate_change_percentage_points(
    steps: Sequence[int], values: Sequence[float], baseline_step: int = 15
) -> list[float]:
    """Express success-rate changes relative to one checkpoint in percentage points."""
    try:
        baseline = values[steps.index(baseline_step)]
    except ValueError as exc:
        raise ValueError(f"Missing baseline step {baseline_step}") from exc
    return [(value - baseline) * 100.0 for value in values]


def style_axis(ax: plt.Axes, suffix: str) -> None:
    """Apply the shared presentation style for comparison plots."""
    ax.grid(axis="y", color="#cfd8dc", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b0bec5")
    ax.spines["bottom"].set_color("#b0bec5")
    ax.text(0.0, 1.015, suffix, transform=ax.transAxes, color="#607d8b", fontsize=9, va="bottom")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.96))


def main() -> None:
    args = parse_args()
    if args.smooth < 1:
        raise ValueError("--smooth must be at least 1")

    excluded_steps = set(args.exclude_step)
    robofape_all_steps, robofape_all_rates = load_success_rates(Path(args.robofape_dir))
    robometer_all_steps, robometer_all_rates = load_success_rates(Path(args.robometer_dir))
    robofape_delta_all = success_rate_change_percentage_points(
        robofape_all_steps, robofape_all_rates
    )
    robometer_delta_all = success_rate_change_percentage_points(
        robometer_all_steps, robometer_all_rates
    )
    robofape_steps, robofape_rates = exclude_steps(
        robofape_all_steps, robofape_all_rates, excluded_steps
    )
    robometer_steps, robometer_rates = exclude_steps(
        robometer_all_steps, robometer_all_rates, excluded_steps
    )
    robofape_delta_steps, robofape_delta = exclude_steps(
        robofape_all_steps, robofape_delta_all, excluded_steps
    )
    robometer_delta_steps, robometer_delta = exclude_steps(
        robometer_all_steps, robometer_delta_all, excluded_steps
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": "#263238",
            "axes.titlecolor": "#15242b",
            "xtick.color": "#455a64",
            "ytick.color": "#455a64",
        }
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.5), layout="constrained")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafb")
    plot_series(ax, robofape_steps, robofape_rates, args.robofape_label, "#0072B2", args.smooth)
    plot_series(ax, robometer_steps, robometer_rates, args.robometer_label, "#D55E00", args.smooth)

    ax.set_title("PegInsertion Evaluation Success Rate", loc="left", fontsize=15, pad=12)
    ax.set_xlabel("PPO training step")
    ax.set_ylabel("Success rate")
    ax.set_ylim(
        adaptive_ylim(
            (robofape_rates, robometer_rates),
            tick_step=0.05,
            lower_bound=0.0,
            upper_bound=1.0,
        )
    )
    suffix = "raw samples" if args.smooth == 1 else f"{args.smooth}-checkpoint moving average; raw samples shown"
    style_axis(ax, suffix)

    output_path = output_dir / "robofape_vs_robometer_success_rate.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")

    fig, ax = plt.subplots(figsize=(9.2, 5.5), layout="constrained")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafb")
    plot_series(
        ax, robofape_delta_steps, robofape_delta, args.robofape_label, "#0072B2", args.smooth
    )
    plot_series(
        ax, robometer_delta_steps, robometer_delta, args.robometer_label, "#D55E00", args.smooth
    )
    ax.axhline(0.0, color="#78909c", linewidth=1.0, linestyle="--", zorder=1)
    ax.axvline(15, color="#90a4ae", linewidth=1.0, linestyle=":", zorder=1)
    ax.set_title("PegInsertion Success-Rate Change from Base Model", loc="left", fontsize=15, pad=12)
    ax.set_xlabel("PPO training step")
    ax.set_ylabel("Change from step 15 (percentage points)")
    ax.set_ylim(adaptive_ylim((robofape_delta, robometer_delta), tick_step=5.0, include=0.0))
    style_axis(ax, f"Step 15: warmup end / base model; {suffix}")
    delta_output_path = output_dir / "robofape_vs_robometer_success_rate_change_vs_step15.png"
    fig.savefig(delta_output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {delta_output_path}")


if __name__ == "__main__":
    main()
