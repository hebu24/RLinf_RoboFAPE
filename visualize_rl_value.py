#!/usr/bin/env python3
"""Visualize RL policy value/critic per-frame on PegInsertionVertical videos.

Renders base+wrist camera images from h5 env_states via ManiSkill,
computes VLM prefix features once (VLM frozen via train_expert_only=True),
evaluates multiple checkpoint value_heads on the same features, and produces
frame-strip + value-line-chart PNGs following plot_combined_progress.py style.
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
from rlinf.models.embodiment.modules.value_head import ValueHead

# ============================================================
# Constants
# ============================================================

RL_LOG_DIR = (
    "/data/yingxi/RLinf_RoboFAPE/logs/"
    "20260809-01:56:53-peg_insertion_rl_async_absolute_independent_window_"
    "bonus0_fresh_ckpt400_gpu57_notcolocate"
)
CKPT_BASE = os.path.join(RL_LOG_DIR, "peg_insertion_async_ppo_pi05_robometer", "checkpoints")
SFT_CKPT_DIR = (
    "/data/yingxi/RLinf_RoboFAPE/logs/"
    "20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/"
    "checkpoints/global_step_40000/actor"
)
ARROW_DIR = (
    "/data/yingxi/robometer/progress_collection_randomized/"
    "PegInsertionVertical-v1/hf_dataset"
)

# Value head architecture (from checkpoint inspection)
VH_INPUT_DIM = 2048
VH_HIDDEN = (512, 256, 128)
VH_ACTIVATION = "gelu"
VH_BIAS_LAST = True

# Prefix token layout for pi05 (from get_value_from_vlm)
NUM_IMAGES_IN_INPUT = 2
LANG_TOKEN_LEN = 200  # pi05
ALL_TOKEN_LENGTH = 968  # pi05: 256*3 + 200


# ============================================================
# Helper functions
# ============================================================

def sample_indices(length, num_frames):
    """Uniform sampling of frame indices (from plot_combined_progress.py)."""
    count = min(num_frames, length)
    return np.linspace(0, length - 1, count, dtype=int)


def load_arrow_dataset(arrow_dir):
    """Load arrow dataset using pyarrow."""
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


def find_checkpoint_dir(global_step):
    """Find the checkpoint directory for a given global_step."""
    pattern = os.path.join(CKPT_BASE, "global_step_%d_*" % global_step)
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError("No checkpoint found for global_step_%d" % global_step)
    return dirs[0]


def ensure_norm_stats(ckpt_dir):
    """Copy norm_stats from SFT checkpoint if missing in RL checkpoint."""
    norm_stats_dst = os.path.join(ckpt_dir, "physical-intelligence", "maniskill", "norm_stats.json")
    if os.path.exists(norm_stats_dst):
        return
    norm_stats_src = os.path.join(SFT_CKPT_DIR, "physical-intelligence", "maniskill", "norm_stats.json")
    if not os.path.exists(norm_stats_src):
        print("WARNING: norm_stats not found at SFT path either: %s" % norm_stats_src)
        return
    os.makedirs(os.path.dirname(norm_stats_dst), exist_ok=True)
    shutil.copy2(norm_stats_src, norm_stats_dst)
    print("Copied norm_stats to %s" % norm_stats_dst)


def restore_env_state(env, env_states, step_idx):
    """Restore ManiSkill env state from h5 env_states at step_idx."""
    sd = env.unwrapped.get_state_dict()

    # Restore actors
    for actor_name in ["peg", "vertical_box_with_hole"]:
        if actor_name in env_states["actors"]:
            h5_data = env_states["actors"][actor_name]
            sd["actors"][actor_name] = torch.from_numpy(h5_data[step_idx:step_idx + 1]).float()

    # Restore articulations (h5 uses "panda", env uses "panda_wristcam")
    env_art_key = "panda_wristcam" if "panda_wristcam" in sd.get("articulations", {}) else "panda"
    h5_art_key = "panda" if "panda" in env_states["articulations"] else "panda_wristcam"
    if h5_art_key in env_states["articulations"]:
        h5_data = env_states["articulations"][h5_art_key]
        sd["articulations"][env_art_key] = torch.from_numpy(h5_data[step_idx:step_idx + 1]).float()

    # Restore controller if present
    if "controller" in sd and "arm" in sd["controller"]:
        pass  # controller state will be derived from articulation state

    env.unwrapped.set_state_dict(sd)


def render_cameras(env):
    """Render base and wrist camera images from the env."""
    obs = env.unwrapped.get_obs()
    base_img = obs["sensor_data"]["base_camera"]["rgb"]  # [1, 224, 224, 3] uint8
    wrist_img = obs["sensor_data"]["hand_camera"]["rgb"]  # [1, 224, 224, 3] uint8
    return base_img, wrist_img


def compute_prefix_feature(model, base_img, wrist_img, state, task_desc, device):
    """Compute the VLM prefix feature (mean-pooled, input to value_head).

    Since VLM is frozen (train_expert_only=True), this feature is identical
    across all RL checkpoints. Compute once, reuse for all value_heads.
    """
    from openpi.models import model as _model

    # Build env_obs matching roboverse_env._wrap_obs format
    # Images from ManiSkill are [B, H, W, C]; model expects [B, C, H, W]
    env_obs = {
        "main_images": base_img.permute(0, 3, 1, 2).to(device),
        "wrist_images": wrist_img.permute(0, 3, 1, 2).to(device),
        "task_descriptions": task_desc,
        "states": state.to(device),
        "extra_view_images": None,
        "wrist_back_images": None,
    }

    # Process through model pipeline: obs_processor -> input_transform -> precision
    to_process_obs = model.obs_processor(env_obs)
    processed_obs = model.input_transform(to_process_obs, transpose=False)
    processed_obs = model.precision_processor(processed_obs)
    observation = _model.Observation.from_dict(processed_obs)

    # Build prefix and extract mean-pooled feature
    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(observation, train=False)
        prefix_output, _, _ = model._build_prefix_cache(images, img_masks, lang_tokens, lang_masks)

        # Replicate get_value_from_vlm mean-pooling (value_vlm_mode="mean_token")
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
    """Load value_head weights from a checkpoint into a standalone ValueHead."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vh_keys = {}
    for k, v in sd.items():
        if k.startswith("value_head."):
            vh_keys[k.replace("value_head.", "", 1)] = v

    if not vh_keys:
        raise ValueError("No value_head keys found in checkpoint: %s" % ckpt_path)

    # Build MLP directly (avoids ValueHead._init_weights bug with gelu)
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
    """Load the OpenPI model with value head from a checkpoint directory."""
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


def plot_value_visualization(frames, values_dict, target_progress,
                             step_indices, output_path, video_id, quality):
    """Create frame-strip + value-line-chart visualization.

    Layout follows plot_combined_progress.py:
    - Top row: N frame images (cropped middle third)
    - Bottom row: line chart with value curves + target_progress
    """
    ncols = len(frames)
    fig = plt.figure(figsize=(max(14.0, ncols * 0.5), 10))
    grid = fig.add_gridspec(2, ncols, height_ratios=[1.0, 2.5], hspace=0.0, wspace=0.025)

    # Top row: frames
    latest_ckpt = list(values_dict.keys())[-1]
    for i, frame in enumerate(frames):
        ax = fig.add_subplot(grid[0, i])
        h, w = frame.shape[:2]
        cropped = frame[:, w // 3: 2 * w // 3]
        ax.imshow(cropped)
        ax.axis("off")
        val = values_dict[latest_ckpt][i]
        ax.set_title("%d | %.3f" % (step_indices[i], val), fontsize=5)

    # Bottom row: line chart
    ax = fig.add_subplot(grid[1, :])
    x = np.arange(ncols)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(values_dict)))
    for (ckpt_name, values), color in zip(values_dict.items(), colors):
        ax.plot(x, values, marker="o", markersize=3, color=color,
                label="Value @ %s" % ckpt_name, linewidth=1.5)

    # Target progress on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(x, target_progress, marker="s", markersize=3, color="#d62728",
             label="GT progress", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.set_ylabel("Target Progress", color="#d62728", fontsize=9)
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=8)

    ax.set_ylabel("Critic Value", fontsize=9)
    ax.set_xlabel("Frame Position", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(si) for si in step_indices], fontsize=5, rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=7)

    fig.suptitle("%s (%s)" % (video_id, quality), fontsize=9, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.12, top=0.93)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % output_path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize RL value/critic per-frame")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[100, 400, 750],
                        help="Global step values to compare")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Number of trajectories per success/failure")
    parser.add_argument("--num-frames", type=int, default=30,
                        help="Number of frames to downsample to")
    parser.add_argument("--output-dir", type=str,
                        default="/data/yingxi/RLinf_RoboFAPE/value_plots",
                        help="Output directory for PNG files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda"

    # 1. Load arrow dataset
    print("Loading arrow dataset...")
    table = load_arrow_dataset(ARROW_DIR)
    print("  Loaded %d trajectories" % table.num_rows)

    # 2. Sample trajectories
    print("Sampling trajectories...")
    success_trajs = sample_trajectories(table, args.num_samples, "successful_labeled", args.seed)
    failure_trajs = sample_trajectories(table, args.num_samples, "failure_labeled", args.seed)
    all_trajs = [("success", t) for t in success_trajs] + [("failure", t) for t in failure_trajs]
    print("  %d success + %d failure" % (len(success_trajs), len(failure_trajs)))

    # 3. Create ManiSkill env
    print("Creating ManiSkill env...")
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

    # 4. Load OpenPI model (from latest checkpoint - VLM frozen, same across all)
    print("Loading OpenPI model (VLM frozen, load once)...")
    latest_step = max(args.checkpoints)
    latest_ckpt_dir = find_checkpoint_dir(latest_step)
    model = load_model(latest_ckpt_dir, device)

    # 5. Pre-load value_heads for all checkpoints
    print("Loading value_heads for checkpoints: %s" % args.checkpoints)
    value_heads = {}
    for step in args.checkpoints:
        ckpt_dir = find_checkpoint_dir(step)
        ckpt_path = os.path.join(ckpt_dir, "actor", "model_state_dict", "full_weights.pt")
        vh = load_value_head(ckpt_path, device)
        value_heads["step_%d" % step] = vh
        print("  Loaded value_head for step_%d" % step)

    # 6. Process each trajectory
    for quality, traj in all_trajs:
        video_id = traj["id"]
        task_desc = traj["task"]
        target_progress_all = np.array(traj["target_progress"])
        states_all = np.array(traj["states"])

        metadata = traj["metadata"]
        source_h5 = metadata["source_h5"]
        source_traj_key = metadata["source_traj_key"]

        print("\nProcessing %s (%s)..." % (video_id, quality))
        print("  h5: %s" % source_h5)

        if not os.path.exists(source_h5):
            print("  WARNING: h5 not found, skipping")
            continue

        # Load h5 env_states
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

        # Downsample to N frames (cap at target_progress length - h5 has 1 extra initial step)
        max_steps = min(num_steps, len(target_progress_all))
        step_indices = sample_indices(max_steps, args.num_frames)
        print("  %d steps -> %d frames" % (num_steps, len(step_indices)))

        # Render frames and compute prefix features
        frames = []
        prefix_features = []

        for si in step_indices:
            # Restore env state
            restore_env_state(env, env_states, si)

            # Render cameras
            base_img, wrist_img = render_cameras(env)

            # State is NOT used for VLM value (only images + language go into prefix).
            # Model expects 8-dim TCP state; arrow dataset has 57-dim full state.
            # Use zeros to pass through Normalize/Pad transforms without error.
            state = torch.zeros(1, 8).float().to(device)

            # Compute prefix feature
            try:
                feature = compute_prefix_feature(model, base_img, wrist_img, state, task_desc, device)
                prefix_features.append(feature)
            except Exception as e:
                print("  ERROR computing feature at step %d: %s" % (si, e))
                prefix_features.append(torch.zeros(1, VH_INPUT_DIM, device=device))

            # Store base camera frame for visualization
            frame_np = base_img[0].cpu().numpy()
            frames.append(frame_np)

            if (si - step_indices[0]) % 10 == 0:
                print("  Rendered step %d/%d" % (si, step_indices[-1]))

        # Stack features [N, 2048]
        prefix_features = torch.cat(prefix_features, dim=0)

        # Compute values for each checkpoint
        values_dict = {}
        with torch.no_grad():
            for ckpt_name, vh in value_heads.items():
                values = vh(prefix_features).squeeze(-1).cpu().numpy()
                values_dict[ckpt_name] = values

        # Get target_progress at sampled indices
        target_progress = target_progress_all[step_indices]

        # Plot
        output_path = os.path.join(args.output_dir, "%s_%s.png" % (quality, video_id))
        plot_value_visualization(
            frames, values_dict, target_progress, step_indices,
            output_path, video_id, quality
        )

    env.close()
    print("\nDone! Output at: %s" % args.output_dir)


if __name__ == "__main__":
    main()
