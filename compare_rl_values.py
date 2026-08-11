#!/usr/bin/env python3
"""Compare RL critic values across two training logs.

Computes MAE and Spearman correlation between RL critic value and ground truth
progress for all overlapping checkpoint steps between two training runs.
"""

import argparse, glob, json, os, shutil, sys
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "/data/yingxi/RLinf_RoboFAPE")
from omegaconf import OmegaConf

ARROW_DIR = "/data/yingxi/robometer/progress_collection_randomized/PegInsertionVertical-v1/hf_dataset"
SFT_CKPT_DIR = "/data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor"

LOGS = {
    "Robometer-4B": "/data/yingxi/RLinf_RoboFAPE/logs/20260809-01:20:10-peg_insertion_rl_async_absolute_independent_window_bonus0_fresh_robometer4b_gpu01_notcolocate",
    "Checkpoint-400": "/data/yingxi/RLinf_RoboFAPE/logs/20260809-01:56:53-peg_insertion_rl_async_absolute_independent_window_bonus0_fresh_ckpt400_gpu57_notcolocate",
}
LOG_CKPT_SUBDIR = "peg_insertion_async_ppo_pi05_robometer/checkpoints"
VH_INPUT_DIM = 2048
VH_HIDDEN = (512, 256, 128)
NUM_IMAGES_IN_INPUT = 2
LANG_TOKEN_LEN = 200


def sample_indices(length, num_frames):
    return np.linspace(0, length - 1, min(num_frames, length), dtype=int)

def load_arrow_dataset(arrow_dir):
    tables = [pa.ipc.open_stream(f).read_all() for f in sorted(glob.glob(os.path.join(arrow_dir, "*.arrow")))]
    return pa.concat_tables(tables)

def sample_trajectories(table, num_samples, quality_label, seed=42):
    rng = np.random.RandomState(seed)
    labels = np.array([table.column("quality_label")[i].as_py() for i in range(table.num_rows)])
    indices = np.where(labels == quality_label)[0]
    if len(indices) > num_samples:
        indices = rng.choice(indices, size=num_samples, replace=False)
    return [{col: table.column(col)[idx].as_py() for col in table.column_names} for idx in sorted(indices)]

def find_all_steps(log_dir):
    ckpt_base = os.path.join(log_dir, LOG_CKPT_SUBDIR)
    return sorted([int(d.split("_")[2]) for d in os.listdir(ckpt_base) if d.startswith("global_step_")])

def find_checkpoint_dir(log_dir, global_step):
    pattern = os.path.join(log_dir, LOG_CKPT_SUBDIR, "global_step_%d_*" % global_step)
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError("No checkpoint for step %d in %s" % (global_step, log_dir))
    return dirs[0]

def ensure_norm_stats(ckpt_dir):
    dst = os.path.join(ckpt_dir, "physical-intelligence", "maniskill", "norm_stats.json")
    if os.path.exists(dst):
        return
    src = os.path.join(SFT_CKPT_DIR, "physical-intelligence", "maniskill", "norm_stats.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def restore_env_state(env, env_states, step_idx):
    sd = env.unwrapped.get_state_dict()
    for actor_name in ["peg", "vertical_box_with_hole"]:
        if actor_name in env_states["actors"]:
            sd["actors"][actor_name] = torch.from_numpy(env_states["actors"][actor_name][step_idx:step_idx+1]).float()
    art_key = "panda_wristcam" if "panda_wristcam" in sd.get("articulations", {}) else "panda"
    h5_art_key = "panda" if "panda" in env_states["articulations"] else "panda_wristcam"
    if h5_art_key in env_states["articulations"]:
        sd["articulations"][art_key] = torch.from_numpy(env_states["articulations"][h5_art_key][step_idx:step_idx+1]).float()
    env.unwrapped.set_state_dict(sd)

def render_cameras(env):
    obs = env.unwrapped.get_obs()
    return obs["sensor_data"]["base_camera"]["rgb"], obs["sensor_data"]["hand_camera"]["rgb"]

def compute_prefix_feature(model, base_img, wrist_img, state, task_desc, device):
    from openpi.models import model as _model
    env_obs = {"main_images": base_img.permute(0,3,1,2).to(device), "wrist_images": wrist_img.permute(0,3,1,2).to(device),
               "task_descriptions": task_desc, "states": state.to(device), "extra_view_images": None, "wrist_back_images": None}
    to_process = model.obs_processor(env_obs)
    processed = model.input_transform(to_process, transpose=False)
    processed = model.precision_processor(processed)
    observation = _model.Observation.from_dict(processed)
    with torch.no_grad():
        images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(observation, train=False)
        prefix_output, _, _ = model._build_prefix_cache(images, img_masks, lang_tokens, lang_masks)
        prefix_mask = [True]*256*NUM_IMAGES_IN_INPUT + [False]*256*(3-NUM_IMAGES_IN_INPUT) + [True]*LANG_TOKEN_LEN
        prefix_out_value = prefix_output[:, prefix_mask, :].mean(dim=1, keepdim=False).to(dtype=torch.float32)
    return prefix_out_value

def load_value_head(ckpt_path, device="cuda"):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vh_keys = {k.replace("value_head.", "", 1): v for k, v in sd.items() if k.startswith("value_head.")}
    if not vh_keys:
        raise ValueError("No value_head keys in %s" % ckpt_path)
    layers, in_dim = [], VH_INPUT_DIM
    for h in VH_HIDDEN:
        layers += [torch.nn.Linear(in_dim, h), torch.nn.GELU()]
        in_dim = h
    layers.append(torch.nn.Linear(in_dim, 1, bias=True))
    vh = torch.nn.Module(); vh.mlp = torch.nn.Sequential(*layers)
    vh.forward = lambda x: vh.mlp(x)
    vh.load_state_dict(vh_keys)
    return vh.to(device).float().eval()

def load_model(ckpt_dir, device="cuda"):
    from rlinf.models.embodiment.openpi import get_model
    ensure_norm_stats(ckpt_dir)
    cfg = OmegaConf.create({"model_path": ckpt_dir, "openpi": {
        "config_name": "pi05_maniskill_peg_insertion_wrist", "num_images_in_input": 2,
        "train_expert_only": True, "add_value_head": True, "value_after_vlm": True,
        "value_vlm_mode": "mean_token", "detach_critic_input": True, "use_dsrl": False,
        "action_chunk": 10, "num_steps": 5, "noise_method": "flow_sde", "noise_level": 0.5,
        "action_env_dim": 7, "action_horizon": 10, "joint_logprob": False,
        "dsrl_state_dim": 8, "dsrl_action_noise_dim": 32, "dsrl_num_q_heads": 10,
        "dsrl_agg_q": "mean", "dsrl_image_latent_dim": 64, "dsrl_state_latent_dim": 64,
        "dsrl_hidden_dims": [128,128,128], "noise_params": [0.7,0.3,400], "bias_last": True},
        "openpi_data": None})
    return get_model(cfg).to(device).eval()

def compute_metrics(values, target_progress):
    """MAE (normalized value vs progress) and Spearman (raw value vs progress)."""
    v_min, v_max = values.min(), values.max()
    values_norm = (values - v_min) / (v_max - v_min) if v_max - v_min > 1e-8 else np.zeros_like(values)
    mae = float(np.mean(np.abs(values_norm - target_progress)))
    spear = float(spearmanr(values, target_progress)[0]) if len(values) > 1 else 0.0
    return mae, (spear if not np.isnan(spear) else 0.0)

def plot_metrics_vs_step(metrics, steps, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ylabel in zip(axes, ["mae", "spearman"], ["MAE (normalized)", "Spearman"]):
        for log_name in LOGS:
            means = [np.mean(metrics["all"][s][log_name][metric]) for s in steps if log_name in metrics["all"][s]]
            stds = [np.std(metrics["all"][s][log_name][metric]) for s in steps if log_name in metrics["all"][s]]
            ax.errorbar(steps[:len(means)], means, yerr=stds, marker="o", capsize=3, label="%s (all)" % log_name)
            means_s = [np.mean(metrics["success"][s][log_name][metric]) for s in steps if log_name in metrics["success"][s]]
            ax.plot(steps[:len(means_s)], means_s, marker="s", linestyle="--", label="%s (success)" % log_name, alpha=0.7)
            means_f = [np.mean(metrics["failure"][s][log_name][metric]) for s in steps if log_name in metrics["failure"][s]]
            ax.plot(steps[:len(means_f)], means_f, marker="^", linestyle="--", label="%s (failure)" % log_name, alpha=0.7)
        ax.set_xlabel("Training Step"); ax.set_ylabel(ylabel); ax.set_title(ylabel)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle("Critic Value vs Progress: Robometer-4B vs Checkpoint-400", fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "metrics_vs_step.png")
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print("Saved: %s" % path)

def plot_example(tf, values_by_log, step_indices, output_path):
    ncols = len(tf["frames"])
    fig = plt.figure(figsize=(max(14, ncols*0.5), 8))
    grid = fig.add_gridspec(2, ncols, height_ratios=[1.0, 2.5], hspace=0.0, wspace=0.025)
    for i, frame in enumerate(tf["frames"]):
        ax = fig.add_subplot(grid[0, i]); h, w = frame.shape[:2]
        ax.imshow(frame[:, w//3:2*w//3]); ax.axis("off")
        ax.set_title("%d" % step_indices[i], fontsize=5)
    ax = fig.add_subplot(grid[1, :]); x = np.arange(ncols)
    colors = {"Robometer-4B": "#1b9e77", "Checkpoint-400": "#d95f02"}
    for log_name, values in values_by_log.items():
        ax.plot(x, values, marker="o", markersize=3, color=colors.get(log_name, "blue"), label=log_name, linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(x, tf["target_progress"], marker="s", markersize=3, color="#d62728", label="GT progress", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.set_ylabel("Target Progress", color="#d62728", fontsize=9); ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=8)
    ax.set_ylabel("Critic Value", fontsize=9); ax.set_xlabel("Frame Position", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([str(si) for si in step_indices], fontsize=5, rotation=45)
    ax.tick_params(axis="y", labelsize=8); ax.grid(True, alpha=0.3)
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, loc="best", fontsize=7)
    fig.suptitle("%s (%s)" % (tf["video_id"], tf["quality"]), fontsize=9, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.12, top=0.93)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    print("Saved: %s" % output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="/data/yingxi/RLinf_RoboFAPE/value_comparison")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = "cuda"

    # 1. Find overlapping steps
    print("Finding overlapping checkpoint steps...")
    all_steps = {n: find_all_steps(d) for n, d in LOGS.items()}
    overlap = sorted(set(all_steps["Robometer-4B"]) & set(all_steps["Checkpoint-400"]))
    print("Overlapping steps: %s" % overlap)

    # 2. Load dataset and sample
    print("Loading arrow dataset...")
    table = load_arrow_dataset(ARROW_DIR)
    success = sample_trajectories(table, args.num_samples, "successful_labeled", args.seed)
    failure = sample_trajectories(table, args.num_samples, "failure_labeled", args.seed)
    all_trajs = [("success", t) for t in success] + [("failure", t) for t in failure]
    print("  %d success + %d failure" % (len(success), len(failure)))

    # 3. Load model + env
    print("Loading model (VLM frozen)...")
    model_ckpt = find_checkpoint_dir(LOGS["Checkpoint-400"], max(overlap))
    model = load_model(model_ckpt, device)
    import gymnasium as gym; import rlinf.envs.maniskill
    env = gym.make("PegInsertionVertical-v1", obs_mode="rgb", robot_uids="panda_wristcam",
                   control_mode="pd_ee_target_delta_pose", render_mode="all", sim_backend="cpu", num_envs=1)
    env.reset(seed=args.seed)

    # 4. Compute prefix features
    print("Computing prefix features for %d trajectories..." % len(all_trajs))
    traj_data = []
    for qi, (quality, traj) in enumerate(all_trajs):
        vid = traj["id"]; task = traj["task"]; tp_all = np.array(traj["target_progress"])
        meta = traj["metadata"]; sh5 = meta["source_h5"]; stk = meta["source_traj_key"]
        print("  [%d/%d] %s (%s)..." % (qi+1, len(all_trajs), vid, quality))
        if not os.path.exists(sh5):
            print("  SKIP: h5 not found"); continue
        with h5py.File(sh5, "r") as f:
            td = f[stk]; ns = td["env_states/actors/peg"].shape[0]
            es = {"actors": {"peg": td["env_states/actors/peg"][:], "vertical_box_with_hole": td["env_states/actors/vertical_box_with_hole"][:]},
                  "articulations": {"panda": td["env_states/articulations/panda"][:]}}
        max_s = min(ns, len(tp_all)); si = sample_indices(max_s, args.num_frames)
        frames, features = [], []
        for s in si:
            restore_env_state(env, es, s); bi, wi = render_cameras(env)
            st = torch.zeros(1, 8).float().to(device)
            feat = compute_prefix_feature(model, bi, wi, st, task, device)
            features.append(feat); frames.append(bi[0].cpu().numpy())
        traj_data.append({"quality": quality, "video_id": vid,
            "features": torch.cat(features, dim=0), "target_progress": tp_all[si],
            "frames": frames, "step_indices": si})
    env.close()
    print("  Computed features for %d trajectories" % len(traj_data))

    # 5. For each log x checkpoint, compute metrics
    print("Computing metrics for all checkpoints...")
    metrics = {"all": {}, "success": {}, "failure": {}}
    for step in overlap:
        print("  Step %d:" % step)
        for cat in metrics: metrics[cat][step] = {}
        for log_name, log_dir in LOGS.items():
            ckpt_dir = find_checkpoint_dir(log_dir, step)
            ckpt_path = os.path.join(ckpt_dir, "actor", "model_state_dict", "full_weights.pt")
            try: vh = load_value_head(ckpt_path, device)
            except Exception as e:
                print("    %s: FAILED: %s" % (log_name, e)); continue
            mae_a, sp_a, mae_s, sp_s, mae_f, sp_f = [], [], [], [], [], []
            with torch.no_grad():
                for tf in traj_data:
                    vals = vh(tf["features"]).squeeze(-1).cpu().numpy()
                    mae, sp = compute_metrics(vals, tf["target_progress"])
                    mae_a.append(mae); sp_a.append(sp)
                    if tf["quality"] == "success": mae_s.append(mae); sp_s.append(sp)
                    else: mae_f.append(mae); sp_f.append(sp)
            metrics["all"][step][log_name] = {"mae": mae_a, "spearman": sp_a}
            metrics["success"][step][log_name] = {"mae": mae_s, "spearman": sp_s}
            metrics["failure"][step][log_name] = {"mae": mae_f, "spearman": sp_f}
            print("    %s: MAE=%.4f+/-%.4f, Spear=%.4f+/-%.4f" % (
                log_name, np.mean(mae_a), np.std(mae_a), np.mean(sp_a), np.std(sp_a)))

    # 6. Save metrics JSON
    os.makedirs(args.output_dir, exist_ok=True)
    mj = {}
    for cat in ["all", "success", "failure"]:
        mj[cat] = {}
        for s in overlap:
            mj[cat][str(s)] = {}
            for ln in LOGS:
                if ln in metrics[cat][s]:
                    m = metrics[cat][s][ln]
                    mj[cat][str(s)][ln] = {"mae_mean": float(np.mean(m["mae"])), "mae_std": float(np.std(m["mae"])),
                        "spearman_mean": float(np.mean(m["spearman"])), "spearman_std": float(np.std(m["spearman"]))}
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(mj, f, indent=2)
    print("Saved metrics.json")

    # 7. Plot metrics vs step
    print("Plotting metrics vs step...")
    plot_metrics_vs_step(metrics, overlap, args.output_dir)

    # 8. Example visualizations (latest step, both logs)
    print("Saving example visualizations...")
    latest = max(overlap)
    for tf in traj_data[:6]:
        vals_by_log = {}
        for ln, ld in LOGS.items():
            ckpt_dir = find_checkpoint_dir(ld, latest)
            ckpt_path = os.path.join(ckpt_dir, "actor", "model_state_dict", "full_weights.pt")
            vh = load_value_head(ckpt_path, device)
            with torch.no_grad():
                vals_by_log[ln] = vh(tf["features"]).squeeze(-1).cpu().numpy()
        path = os.path.join(args.output_dir, "examples", "%s_%s.png" % (tf["quality"], tf["video_id"]))
        plot_example(tf, vals_by_log, tf["step_indices"], path)

    print("\nDone! Output at: %s" % args.output_dir)

if __name__ == "__main__":
    main()
