#!/usr/bin/env python3
"""Plot paper-ready comparisons for the two PushCube training runs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path("/data/yingxi/RLinf_RoboFAPE")
OURS = ROOT / "logs/pushcube_rl_env32_abs30_debugfix2_seed0_31_20260826_112825"
BASELINE = ROOT / "logs/pushcube_rl_env32_abs30_robometer4b_parallel_threadcap_20260826_120445"
OUT = ROOT / "plots"

RED = "#C66F76"
BLUE = "#6289AE"
INK = "#263238"
GRID = "#D9DEE2"


def load_sweep(path):
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    steps = np.unique(rows["step"])
    values = [rows["success_rate"][rows["step"] == step].astype(float) for step in steps]
    return steps.astype(float), values


def smooth(values, window):
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    window = max(window, 3)
    if window % 2 == 0:
        window -= 1
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_ev(path):
    event_files = sorted(path.glob("events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file under {path}")
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    events = accumulator.Scalars("train/critic/explained_variance")
    return (
        np.array([event.step for event in events], dtype=float),
        np.array([event.value for event in events], dtype=float),
    )


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.65)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA5AB")
    ax.spines["bottom"].set_color("#9AA5AB")
    ax.tick_params(colors=INK, labelsize=10)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sr():
    ours_steps, ours_values = load_sweep(OURS / "sweep_multiseed/pushcube_sweep_metrics.csv")
    baseline_steps, baseline_values = load_sweep(
        BASELINE / "sweep_gpu3/pushcube_sweep_metrics.csv"
    )
    max_step = min(ours_steps.max(), baseline_steps.max())
    ours_mask = ours_steps <= max_step
    baseline_mask = baseline_steps <= max_step
    ours_values = [values for values, keep in zip(ours_values, ours_mask) if keep]
    baseline_values = [values for values, keep in zip(baseline_values, baseline_mask) if keep]
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for steps, values, color, label in [
        (ours_steps[ours_mask], ours_values, RED, "Ours"),
        (baseline_steps[baseline_mask], baseline_values, BLUE, "Baseline"),
    ]:
        for step, seed_values in zip(steps, values):
            ax.scatter(
                np.full(len(seed_values), step),
                seed_values,
                s=18,
                color=color,
                alpha=0.20,
                linewidths=0,
                zorder=2,
            )
        mean = np.array([seed_values.mean() for seed_values in values])
        ax.plot(
            steps,
            smooth(mean, 5),
            color=color,
            linewidth=2.6,
            solid_capstyle="round",
            label=label,
            zorder=3,
        )
    ax.set_xlim(0, max_step)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Success rate")
    ax.set_title("PushCube evaluation success rate", loc="left", color=INK, weight="bold", pad=12)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=10, handlelength=2.4)
    style_axis(ax)
    save(fig, "pushcube_ours_vs_baseline_sr")


def plot_ev():
    ours_steps, ours_values = load_ev(OURS / "tensorboard")
    baseline_steps, baseline_values = load_ev(BASELINE / "tensorboard")
    max_step = min(ours_steps.max(), baseline_steps.max())
    ours_mask = ours_steps <= max_step
    baseline_mask = baseline_steps <= max_step

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for steps, values, color, label in [
        (ours_steps[ours_mask], ours_values[ours_mask], RED, "Ours"),
        (baseline_steps[baseline_mask], baseline_values[baseline_mask], BLUE, "Baseline"),
    ]:
        ax.scatter(steps, values, s=8, color=color, alpha=0.14, linewidths=0, zorder=2)
        ax.plot(
            steps,
            smooth(values, 31),
            color=color,
            linewidth=2.4,
            solid_capstyle="round",
            label=label,
            zorder=3,
        )
    ax.set_xlim(1, max_step)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Explained variance")
    ax.set_title("Critic explained variance during training", loc="left", color=INK, weight="bold", pad=12)
    ax.legend(frameon=False, ncol=2, loc="best", fontsize=10, handlelength=2.4)
    style_axis(ax)
    save(fig, "pushcube_ours_vs_baseline_ev")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plot_sr()
    plot_ev()
    print(f"Wrote plots to {OUT}")
