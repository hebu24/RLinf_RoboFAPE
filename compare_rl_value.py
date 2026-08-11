#!/usr/bin/env python3
"""Compare RL value/critic across two runs over common training steps.

For each common checkpoint step shared by two RL runs (Robometer-4B vs
Checkpoint-400), evaluate the value_head on a fixed test set (10 success +
10 failure trajectories sampled from the progress-collection arrow dataset).

Because the VLM is frozen (train_expert_only=True) and both runs start from
the same SFT base, the VLM prefix feature is identical across all checkpoints
and both runs -- computed once, reused for every value_head.

Produces:
  1. metrics_vs_step.png  -- Spearman & MAE (min-max normalized) vs training
     step, two lines per panel (one per run), split by success/failure.
  2. metrics.json / metrics.csv -- raw per-(run,step,quality) aggregates.
  3. per_traj/*.png -- per-trajectory value-curve comparison (both runs,
     step color gradient) against GT progress.

Isolation: NO Ray. ManiSkill env uses sim_backend="cpu". Set
CUDA_VISIBLE_DEVICES to a free GPU before running.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import torch

sys.path.insert(0, "/data/yingxi/RLinf_RoboFAPE")

from omegaconf import OmegaConf

# ============================================================
# Run configuration
# ============================================================

LOG_ROOT = "/data/yingxi/RLinf_RoboFAPE/logs"

RUNS = {
    "rbm4b": {
        "label": "Robometer-4B (gpu01)",
        "log_dir": os.path.join(
            LOG_ROOT,
            "20260809-01:20:10-peg_insertion_rl_async_absolute_independent_"
            "window_bonus0_fresh_robometer4b_gpu01_notcolocate",
        ),
    },
    "ckpt400": {
        "label": "Checkpoint-400 (gpu57)",
        "log_dir": os.path.join(
            LOG_ROOT,
            "20260809-01:56:53-peg_insertion_rl_async_absolute_independent_"
            "window_bonus0_fresh_ckpt400_gpu57_notcolocate",
        ),
    },
}
for r in RUNS.values():
    r["ckpt_base"] = os.path.join(
        r["log_dir"], "peg_insertion_async_ppo_pi05_robometer", "checkpoints"
    )

# SFT base checkpoint (norm_stats source + frozen VLM base, identical for both runs)
SFT_CKPT_DIR = (
    "/data/yingxi/RLinf_RoboFAPE/logs/"
    "20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/"
    "checkpoints/global_step_40000/actor"
)

ARROW_DIR = (
    "/data/yingxi/robometer/progress_collection_randomized/"
    "PegInsertionVertical-v1/hf_dataset"
)

# Value head architecture (verified from checkpoint inspection)
VH_INPUT_DIM = 2048
VH_HIDDEN = (512, 256, 128)
VH_ACTIVATION = "gelu"
VH_BIAS_LAST = True

# Prefix token layout for pi05 (from get_value_from_vlm)
NUM_IMAGES_IN_INPUT = 2
LANG_TOKEN_LEN = 200  # pi05
ALL_TOKEN_LENGTH = 968  # pi05: 256*3 + 200

# Common steps across both runs (verified present in both)
COMMON_STEPS = [15, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]


# ============================================================
# Helper functions (adapted from visualize_rl_value.py)
# ============================================================

def sample_indices(length, num_frames):
    count = min(num_frames, length)
    return np.linspace(0, length - 1, count, dtype=int)


def load_arrow_dataset(arrow_dir):
    arrow_files = sorted(glob.glob(os.path.join(arrow_dir, "*.arrow")))
    tables = [pa.ipc.open_stream(f).read_all() for f in arrow_files]
    table = pa.concat_tables(tables)
    return table


def sample_trajectories(table, num_samples, quality_label, seed=42):
    """Sample trajectories from arrow dataset by quality_label."""
    rng = np.random.RandomState(seed)
    labels = np.array([table.column("quality_label")[i].as_py() for i in range(table.num_rows)])
    indices = np.where(labels == quality_label)[0]
    if len(indices) > num_samples:
        indices = rng.choice(indices, size=num_samples, replace=False)
    trajectories = []
    for idx in sorted(indices):
        row = {col: table.column(col)[idx].as_py() for col in table.column_names}
        trajectories.append(row)
    return trajectories


def find_checkpoint_dir(ckpt_base, global_step):
    pattern = os.path.join(ckpt_base, "global_step_%d_*" % global_step)
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    return dirs[0]


def ensure_norm_stats(ckpt_dir):
    """Copy norm_stats to top-level if missing (model_path expects it there)."""
    norm_stats_dst = os.path.join(ckpt_dir, "physical-intelligence", "maniskill", "norm_stats.json")
    if os.path.exists(norm_stats_dst):
        return
    norm_stats_src = os.path.join(SFT_CKPT_DIR, "physical-intelligence", "maniskill", "norm_stats.json")
    if not os.path.exists(norm_stats_src):
        src2 = os.path.join(ckpt_dir, "actor", "physical-intelligence", "maniskill", "norm_stats.json")
        if os.path.exists(src2):
            norm_stats_src = src2
    if not os.path.exists(norm_stats_src):
        print("WARNING: norm_stats not found: %s" % norm_stats_src)
        return
    os.makedirs(os.path.dirname(norm_stats_dst), exist_ok=True)
    shutil.copy2(norm_stats_src, norm_stats_dst)
    print("Copied norm_stats to %s" % norm_stats_dst)


def restore_env_state(env, env_states, step_idx):
    sd = env.unwrapped.get_state_dict()
    for actor_name in ["peg", "vertical_box_with_hole"]:
        if actor_name in env_states["actors"]:
            h5_data = env_states["actors"][actor_name]
            sd["actors"][actor_name] = torch.from_numpy(h5_data[step_idx:step_idx + 1]).float()
    env_art_key = "panda_wristcam" if "panda_wristcam" in sd.get("articulations", {}) else "panda"
    h5_art_key = "panda" if "panda" in env_states["articulations"] else "panda_wristcam"
    if h5_art_key in env_states["articulations"]:
        h5_data = env_states["articulations"][h5_art_key]
        sd["articulations"][env_art_key] = torch.from_numpy(h5_data[step_idx:step_idx + 1]).float()
    env.unwrapped.set_state_dict(sd)


def render_cameras(env):
    obs = env.unwrapped.get_obs()
    base_img = obs["sensor_data"]["base_camera"]["rgb"]
    wrist_img = obs["sensor_data"]["hand_camera"]["rgb"]
    return base_img, wrist_img


def compute_prefix_feature(model, base_img, wrist_img, state, task_desc, device):
    """Compute the VLM prefix feature (mean-pooled, input to value_head).

    VLM frozen (train_expert_only=True) -> identical across all checkpoints
    and both runs. Compute once, reuse everywhere.
    """
    from openpi.models import model as _model

    env_obs = {
        "main_images": base_img.permute(0, 3, 1, 2).to(device),
        "wrist_images": wrist_img.permute(0, 3, 1, 2).to(device),
        "task_descriptions": task_desc,
        "states": state.to(device),
        "extra_view_images": None,
        "wrist_back_images": None,
    }

    to_process_obs = model.obs_processor(env_obs)
    processed_obs = model.input_transform(to_process_obs, transpose=False)
    processed_obs = model.precision_processor(processed_obs)
    observation = _model.Observation.from_dict(processed_obs)

    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(
            observation, train=False
        )
        prefix_output, _, _ = model._build_prefix_cache(images, img_masks, lang_tokens, lang_masks)

        prefix_mask = (
            [True] * 256 * NUM_IMAGES_IN_INPUT
            + [False] * 256 * (3 - NUM_IMAGES_IN_INPUT)
            + [True] * LANG_TOKEN_LEN
        )
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)

    return prefix_out_value  # [1, 2048]


def load_value_head(ckpt_path, device="cuda"):
    """Load value_head weights from a checkpoint into a standalone MLP."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vh_keys = {}
    for k, v in sd.items():
        if k.startswith("value_head."):
            vh_keys[k.replace("value_head.", "", 1)] = v
    if not vh_keys:
        raise ValueError("No value_head keys in checkpoint: %s" % ckpt_path)

    layers = []
    in_dim = VH_INPUT_DIM
    for h in VH_HIDDEN:
        layers.append(torch.nn.Linear(in_dim, h))
        layers.append(torch.nn.GELU())
        in_dim = h
    layers.append(torch.nn.Linear(in_dim, 1, bias=VH_BIAS_LAST))
    vh = torch.nn.Module()
    vh.mlp = torch.nn.Sequential(*layers)
    vh.forward = lambda x: vh.mlp(x)
    vh.load_state_dict(vh_keys)
    vh = vh.to(device).float()
    vh.eval()
    return vh


def load_model(ckpt_dir, device="cuda"):
    """Load the OpenPI model (frozen VLM) from a checkpoint directory."""
    from rlinf.models.embodiment.openpi import get_model

    ensure_norm_stats(ckpt_dir)

    cfg = OmegaConf.create({
        "model_path": ckpt_dir,
        "openpi": {
            "config_name": "pi05_maniskill_peg_insertion_wrist",
            "num_images_in_input": 2,
            "train_expert_only": True,
            "add_value_head": True,
            "value_after_vlm": True,
            "value_vlm_mode": "mean_token",
            "detach_critic_input": True,
            "use_dsrl": False,
            "action_chunk": 10,
            "num_steps": 5,
            "noise_method": "flow_sde",
            "noise_level": 0.5,
            "action_env_dim": 7,
            "action_horizon": 10,
            "joint_logprob": False,
            "dsrl_state_dim": 8,
            "dsrl_action_noise_dim": 32,
            "dsrl_num_q_heads": 10,
            "dsrl_agg_q": "mean",
            "dsrl_image_latent_dim": 64,
            "dsrl_state_latent_dim": 64,
            "dsrl_hidden_dims": [128, 128, 128],
            "noise_params": [0.7, 0.3, 400],
            "bias_last": False,
        },
        "openpi_data": None,
    })

    model = get_model(cfg)
    model = model.to(device)
    model.eval()
    return model


# ============================================================
# Metrics
# ============================================================

def spearman_corr(a, b):
    """Spearman rank correlation (scale-invariant)."""
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(a, b)
        return float(r) if not np.isnan(r) else 0.0
    except Exception:
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = (np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum())) + 1e-12
        return float((ra * rb).sum() / denom)


def minmax01(a):
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < 1e-8:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def mae_minmax(a, b):
    """MAE between min-max normalized value (->[0,1]) and target_progress ([0,1])."""
    return float(np.mean(np.abs(minmax01(a) - b)))


# ============================================================
# Plotting
# ============================================================

def plot_metrics_vs_step(metrics, steps, run_labels, output_path, num_traj):
    """1x2 grid: (Spearman, MAE). 4 lines per panel = 2 runs x 2 qualities.
    Color: rbm4b=blue, ckpt400=red. Shade: success=dark+solid, failure=light+dashed."""
    qualities = ["success", "failure"]
    metric_names = [
        ("spearman", "Spearman corr (value vs progress)"),
        ("mae", "MAE (value min-max [0,1] vs progress)"),
    ]
    # Two color families: blue for rbm4b, red for ckpt400
    # success = dark solid, failure = light dashed
    style = {
        ("rbm4b", "success"):   {"color": "#08519c", "ls": "-",  "marker": "o"},
        ("rbm4b", "failure"):   {"color": "#6baed6", "ls": "--", "marker": "s"},
        ("ckpt400", "success"):  {"color": "#a50f15", "ls": "-",  "marker": "o"},
        ("ckpt400", "failure"):  {"color": "#fb6a4a", "ls": "--", "marker": "s"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for mi, (mkey, mname) in enumerate(metric_names):
        ax = axes[mi]
        for run in ["rbm4b", "ckpt400"]:
            for q in qualities:
                vals = [metrics[run][str(step)][q][mkey]["mean"] for step in steps]
                stds = [metrics[run][str(step)][q][mkey]["std"] for step in steps]
                xs = np.array(steps, dtype=float)
                s = style[(run, q)]
                lbl = "%s — %s" % (run_labels[run], q)
                ax.plot(xs, vals, marker=s["marker"], ms=5, color=s["color"],
                        linestyle=s["ls"], label=lbl, linewidth=2)
                ax.fill_between(xs, np.array(vals) - np.array(stds),
                                np.array(vals) + np.array(stds),
                                color=s["color"], alpha=0.12)
        ax.set_xlabel("Training step (global_step)", fontsize=11)
        ax.set_ylabel(mname, fontsize=11)
        ax.set_xticks(steps)
        ax.tick_params(axis="x", labelsize=8, rotation=45)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, alpha=0.3)
        if mkey == "spearman":
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color="gray", lw=0.8, ls=":")
        elif mkey == "mae":
            ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="best", ncol=2)

    fig.suptitle(
        "Value-head metrics vs training step  (n=%d trajs/quality)\n"
        "Robometer-4B (blue) vs Checkpoint-400 (red)  |  dark=success, light=failure  |  "
        "reward=normalized_dense absolute, normalize_returns=True",
        fontsize=12, y=1.01,
    )
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.13, top=0.88, wspace=0.20)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % output_path)


def plot_traj_comparison(frames, run_values, target_progress, step_indices,
                         video_id, quality, steps_to_plot, output_path):
    """Per-trajectory: top = frame strip; bottom = 2 panels (rbm4b | ckpt400)
    each showing value curves (step color gradient) + GT progress."""
    ncols = len(frames)
    fig = plt.figure(figsize=(max(14.0, ncols * 0.55), 8.5))
    # gridspec with ncols columns: top row = one cell per frame,
    # bottom row = two panels each spanning half the columns.
    gs = fig.add_gridspec(2, ncols, height_ratios=[1.0, 2.6], hspace=0.1, wspace=0.05)

    for i, frame in enumerate(frames):
        ax = fig.add_subplot(gs[0, i])
        h, w = frame.shape[:2]
        cropped = frame[:, w // 3: 2 * w // 3]
        ax.imshow(cropped)
        ax.axis("off")
        ax.set_title("s%d" % step_indices[i], fontsize=5)

    x = np.arange(ncols)
    # Two color families: rbm4b=Blues, ckpt400=Reds (light→dark = early→late step)
    cmaps = {"rbm4b": plt.cm.Blues, "ckpt400": plt.cm.Reds}
    half = max(1, ncols // 2)
    for ri, run in enumerate(["rbm4b", "ckpt400"]):
        sl = slice(0, half) if ri == 0 else slice(half, ncols)
        ax = fig.add_subplot(gs[1, sl])
        cmap = cmaps[run](np.linspace(0.35, 0.95, len(steps_to_plot)))
        for k, step in enumerate(steps_to_plot):
            vals = run_values[run].get(step)
            if vals is None:
                continue
            ax.plot(x, vals, marker="o", ms=3, color=cmap[k],
                    label="step %d" % step, linewidth=1.6, alpha=0.85)
        ax2 = ax.twinx()
        ax2.plot(x, target_progress, marker="s", ms=3, color="#2ca02c",
                 label="GT progress", ls="--", lw=1.5, alpha=0.7)
        ax2.set_ylabel("GT progress", color="#2ca02c", fontsize=9)
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis="y", labelcolor="#2ca02c", labelsize=8)
        ax.set_ylabel("Critic value", fontsize=9)
        ax.set_xlabel("Frame", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([str(si) for si in step_indices], fontsize=5, rotation=45)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(RUNS[run]["label"], fontsize=10)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=7)

    fig.suptitle("%s (%s)" % (video_id, quality), fontsize=10, y=0.99)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.1, top=0.93)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % output_path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compare RL value across two runs over steps")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Trajectories per success/failure (test set)")
    parser.add_argument("--num-frames", type=int, default=30,
                        help="Frames to downsample each traj to")
    parser.add_argument("--steps", nargs="+", type=int, default=COMMON_STEPS,
                        help="Common training steps to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="/data/yingxi/RLinf_RoboFAPE/value_compare")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--traj-plots", action="store_true", default=True,
                        help="Also produce per-trajectory comparison PNGs")
    args = parser.parse_args()

    device = args.device
    steps = sorted(args.steps)
    os.makedirs(args.output_dir, exist_ok=True)
    per_traj_dir = os.path.join(args.output_dir, "per_traj")
    if args.traj_plots:
        os.makedirs(per_traj_dir, exist_ok=True)

    # 1. Load arrow dataset
    print("Loading arrow dataset...")
    table = load_arrow_dataset(ARROW_DIR)
    print("  Loaded %d trajectories" % table.num_rows)

    # 2. Sample test set (10 success + 10 failure)
    print("Sampling %d success + %d failure (seed=%d)..." % (
        args.num_samples, args.num_samples, args.seed))
    success_trajs = sample_trajectories(table, args.num_samples, "successful_labeled", args.seed)
    failure_trajs = sample_trajectories(table, args.num_samples, "failure_labeled", args.seed)
    all_trajs = [("success", t) for t in success_trajs] + [("failure", t) for t in failure_trajs]
    print("  %d success + %d failure" % (len(success_trajs), len(failure_trajs)))

    # 3. Verify checkpoints exist for all steps in both runs
    print("\nResolving checkpoints...")
    ckpt_dirs = {}  # [run][step] = dir
    for run in RUNS:
        ckpt_dirs[run] = {}
        for step in steps:
            d = find_checkpoint_dir(RUNS[run]["ckpt_base"], step)
            if d is None:
                print("  WARNING: %s step %d not found, skipping that step for this run" % (run, step))
            else:
                ckpt_dirs[run][step] = d
                print("  %s step %d -> %s" % (run, step, os.path.basename(d)))
    # keep only steps present in BOTH runs
    common = [s for s in steps if s in ckpt_dirs["rbm4b"] and s in ckpt_dirs["ckpt400"]]
    print("  Common steps present in both runs: %s" % common)
    steps = common

    # 4. Create ManiSkill env (CPU sim, no GPU, no Ray)
    print("\nCreating ManiSkill env (CPU sim)...")
    import gymnasium as gym
    import rlinf.envs.maniskill  # registers PegInsertionVertical-v1
    env = gym.make(
        "PegInsertionVertical-v1",
        obs_mode="rgb",
        robot_uids="panda_wristcam",
        control_mode="pd_ee_target_delta_pose",
        render_mode="all",
        sim_backend="cpu",
        num_envs=1,
    )
    env.reset(seed=args.seed)

    # 5. Load frozen VLM ONCE (from ckpt400 latest step -- identical across runs)
    model_step = max(steps)
    model_ckpt = ckpt_dirs["ckpt400"][model_step]
    print("\nLoading OpenPI model (frozen VLM) from ckpt400 step %d: %s" % (model_step, model_ckpt))
    model = load_model(model_ckpt, device)

    # 6. Load value_heads for all (run, step)
    print("\nLoading value_heads...")
    value_heads = {}  # [run][step] = vh
    for run in RUNS:
        value_heads[run] = {}
        for step in steps:
            ckpt_path = os.path.join(ckpt_dirs[run][step], "actor", "model_state_dict", "full_weights.pt")
            if not os.path.exists(ckpt_path):
                print("  MISSING weights: %s" % ckpt_path)
                continue
            vh = load_value_head(ckpt_path, device)
            value_heads[run][step] = vh
            print("  loaded %s step %d" % (run, step))

    # 7. Process each trajectory: render frames + prefix features ONCE, eval all vh
    # metrics[run][step][quality] = {spearman:{list}, mae:{list}}
    metrics = {run: {s: {"success": {"spearman": [], "mae": []},
                         "failure": {"spearman": [], "mae": []}} for s in steps}
              for run in RUNS}

    # representative steps for per-traj plots (<=6 to stay readable)
    traj_steps = [s for s in [15, 100, 200, 300, 400, 500, 600] if s in steps][:6]

    for qi, (quality, traj) in enumerate(all_trajs):
        video_id = traj["id"]
        task_desc = traj["task"]
        target_progress_all = np.array(traj["target_progress"], dtype=float)

        metadata = traj["metadata"]
        source_h5 = metadata["source_h5"]
        source_traj_key = metadata["source_traj_key"]

        print("\n[%d/%d] %s (%s)..." % (qi + 1, len(all_trajs), video_id, quality))
        print("  h5: %s" % source_h5)

        if not os.path.exists(source_h5):
            print("  SKIP: h5 missing")
            continue

        with h5py.File(source_h5, "r") as f:
            traj_data = f[source_traj_key]
            num_steps = traj_data["env_states/actors/peg"].shape[0]
            env_states = {
                "actors": {
                    "peg": traj_data["env_states/actors/peg"][:],
                    "vertical_box_with_hole": traj_data["env_states/actors/vertical_box_with_hole"][:],
                },
                "articulations": {
                    "panda": traj_data["env_states/articulations/panda"][:],
                },
            }

        max_steps = min(num_steps, len(target_progress_all))
        step_indices = sample_indices(max_steps, args.num_frames)
        print("  %d steps -> %d frames" % (num_steps, len(step_indices)))

        frames = []
        prefix_features = []
        for si in step_indices:
            restore_env_state(env, env_states, si)
            base_img, wrist_img = render_cameras(env)
            state = torch.zeros(1, 8).float().to(device)
            try:
                feat = compute_prefix_feature(model, base_img, wrist_img, state, task_desc, device)
                prefix_features.append(feat)
            except Exception as e:
                print("  ERROR feature @ step %d: %s" % (si, e))
                prefix_features.append(torch.zeros(1, VH_INPUT_DIM, device=device))
            frames.append(base_img[0].cpu().numpy())
            if (si - step_indices[0]) % 10 == 0:
                print("  rendered step %d/%d" % (si, step_indices[-1]))

        prefix_features = torch.cat(prefix_features, dim=0)  # [N, 2048]
        target_progress = target_progress_all[step_indices]

        # Eval every value_head on these shared features
        run_values = {run: {} for run in RUNS}
        with torch.no_grad():
            for run in RUNS:
                for step in steps:
                    vh = value_heads[run].get(step)
                    if vh is None:
                        continue
                    vals = vh(prefix_features).squeeze(-1).cpu().numpy()
                    run_values[run][step] = vals
                    sp = spearman_corr(vals, target_progress)
                    mae = mae_minmax(vals, target_progress)
                    metrics[run][step][quality]["spearman"].append(sp)
                    metrics[run][step][quality]["mae"].append(mae)

        # per-trajectory plot (representative steps only)
        if args.traj_plots:
            out = os.path.join(per_traj_dir, "%s_%s.png" % (quality, video_id))
            plot_traj_comparison(
                frames, run_values, target_progress, step_indices,
                video_id, quality, traj_steps, out,
            )

    env.close()

    # 8. Aggregate
    print("\nAggregating metrics...")
    agg = {run: {} for run in RUNS}
    for run in RUNS:
        for step in steps:
            agg[run][step] = {}
            for q in ["success", "failure"]:
                agg[run][step][q] = {}
                for mkey in ["spearman", "mae"]:
                    arr = np.array(metrics[run][step][q][mkey])
                    if len(arr) == 0:
                        agg[run][step][q][mkey] = {"mean": float("nan"), "std": float("nan"), "n": 0}
                    else:
                        agg[run][step][q][mkey] = {
                            "mean": float(np.mean(arr)),
                            "std": float(np.std(arr)),
                            "n": int(len(arr)),
                        }
                    print("  %s step%3d %-7s %-8s mean=%.4f std=%.4f n=%d" % (
                        run, step, q, mkey,
                        agg[run][step][q][mkey]["mean"],
                        agg[run][step][q][mkey]["std"],
                        agg[run][step][q][mkey]["n"]))

    # 9. Save JSON + CSV
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"steps": steps, "runs": RUNS, "metrics": agg,
                   "num_samples": args.num_samples, "seed": args.seed}, f, indent=2)
    with open(os.path.join(args.output_dir, "metrics.csv"), "w") as f:
        f.write("run,step,quality,metric,mean,std,n\n")
        for run in RUNS:
            for step in steps:
                for q in ["success", "failure"]:
                    for mkey in ["spearman", "mae"]:
                        a = agg[run][step][q][mkey]
                        f.write("%s,%d,%s,%s,%s,%s,%d\n" % (
                            run, step, q, mkey, a["mean"], a["std"], a["n"]))
    print("Saved metrics.json + metrics.csv to %s" % args.output_dir)

    # 10. Metrics vs step plot
    plot_metrics_vs_step(
        agg, steps, {r: RUNS[r]["label"] for r in RUNS},
        os.path.join(args.output_dir, "metrics_vs_step.png"),
        args.num_samples,
    )

    print("\nDone! Output at: %s" % args.output_dir)


if __name__ == "__main__":
    main()
