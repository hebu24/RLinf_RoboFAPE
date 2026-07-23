#!/usr/bin/env python3
import argparse, glob, json, os, os.path as osp, sys, importlib.util as ilu
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_PATH = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

s = ilu.spec_from_file_location("_peg", "/opt/yingxi/RLinf_RoboFAPE/rlinf/envs/maniskill/tasks/peg_insertion_vertical.py")
m = ilu.module_from_spec(s); s.loader.exec_module(m)
import gymnasium as gym

from rlinf.envs.maniskill.peg_insertion_pi05 import (
    PI05_ACTION_DIM,
    PI05_STATE_DIM,
    aligned_pi05_state_from_tcp_matrix,
    describe_action_semantics,
    euler_xyz_delta_actions_from_tcp_matrices,
    qpos8_to_robot_qpos,
)

SDIM_TCP = PI05_STATE_DIM
ADIM = PI05_ACTION_DIM
STATE_NAMES = ["tcp_x", "tcp_y", "tcp_z", "roll", "pitch", "yaw", "finger0", "finger1"]
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

def make_env(render_backend=None):
    kwargs = {}
    if render_backend:
        kwargs["render_backend"] = render_backend
    env = gym.make(
        "PegInsertionVertical-v1",
        num_envs=1,
        obs_mode="state",
        robot_uids="panda_wristcam",
        control_mode="pd_joint_pos",
        sim_backend="cpu",
        max_episode_steps=600,
        **kwargs,
    )
    env.reset(seed=0)
    return env

def fk_proprio(uw, qpos8):
    robot = uw.agent.robot
    grip = float(qpos8[7])
    robot_qpos = qpos8_to_robot_qpos(qpos8)
    robot.set_qpos(torch.tensor(robot_qpos).unsqueeze(0))
    tcp_root = uw.agent.robot.pose.inv() * uw.agent.tcp.pose
    T = tcp_root.to_transformation_matrix().detach().cpu().numpy()[0]
    proprio = aligned_pi05_state_from_tcp_matrix(T, [grip, grip])
    return proprio, T

def process_episode(uw, ep_df):
    T = len(ep_df)
    states_tcp = np.zeros((T, SDIM_TCP), dtype=np.float32)
    tcp_mats = np.zeros((T, 4, 4), dtype=np.float64)
    source_actions = np.stack(ep_df["actions"] if "actions" in ep_df.columns else ep_df["action"].values)
    gripper_actions = source_actions[:, 6]
    for t in range(T):
        qpos8 = np.asarray(ep_df["observation.state"].iloc[t], dtype=np.float32)
        proprio, Tmat = fk_proprio(uw, qpos8)
        states_tcp[t] = proprio
        tcp_mats[t] = Tmat
    actions = euler_xyz_delta_actions_from_tcp_matrices(tcp_mats, gripper_actions)
    return states_tcp, actions

def _load_dataset_frame(data_dir):
    pq_path = osp.join(data_dir, "data", "chunk-000", "file-000.parquet")
    if osp.exists(pq_path):
        return pd.read_parquet(pq_path), pq_path, "single"
    pq_files = sorted(glob.glob(osp.join(data_dir, "data", "chunk-*", "episode_*.parquet")))
    if not pq_files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {data_dir}")
    return pd.concat([pd.read_parquet(f) for f in pq_files], ignore_index=True), pq_path, "episodes"

def _array_summary(values):
    values = np.asarray(values)
    return {
        "mean": values.mean(0).tolist(),
        "std": values.std(0).tolist(),
        "min": values.min(0).tolist(),
        "max": values.max(0).tolist(),
        "p01": np.percentile(values, 1, axis=0).tolist(),
        "p50": np.percentile(values, 50, axis=0).tolist(),
        "p99": np.percentile(values, 99, axis=0).tolist(),
        "max_abs": np.abs(values).max(0).tolist(),
    }

def _summarize_array(name, values):
    summary = _array_summary(values)
    for key in ["mean", "std", "min", "max", "p01", "p50", "p99", "max_abs"]:
        print(f"{name} {key}={np.asarray(summary[key])}")
    return summary

def _load_openpi_norm_stats(path):
    if path is None:
        return None
    if not osp.exists(path):
        raise FileNotFoundError(f"norm stats json does not exist: {path}")
    data = json.load(open(path))
    return data.get("norm_stats", data)

def _quantile_normalize(values, stats):
    q01 = np.asarray(stats["q01"], dtype=np.float64)[: values.shape[-1]]
    q99 = np.asarray(stats["q99"], dtype=np.float64)[: values.shape[-1]]
    return (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

def _worst_normalized_sample(name, values, normalized, dim_names, frame_info):
    abs_norm = np.abs(normalized)
    row, dim = np.unravel_index(abs_norm.argmax(), abs_norm.shape)
    info = {
        "field": name,
        "row": int(row),
        "dim": int(dim),
        "dim_name": dim_names[dim],
        "raw": float(values[row, dim]),
        "normalized": float(normalized[row, dim]),
        "abs_normalized": float(abs_norm[row, dim]),
    }
    if frame_info is not None:
        info.update(
            {
                "episode_index": int(frame_info["episode_index"][row]),
                "frame_index": int(frame_info["frame_index"][row]),
                "index": int(frame_info["index"][row]),
            }
        )
    return info

def _check_norm_stats(
    actions,
    states_tcp,
    norm_stats,
    threshold,
    frame_info=None,
    fail_on_max=False,
):
    if norm_stats is None:
        return None
    norm_state = _quantile_normalize(states_tcp, norm_stats["state"])
    norm_action = _quantile_normalize(actions, norm_stats["actions"])
    state_summary = _summarize_array("normalized state_tcp", norm_state)
    action_summary = _summarize_array("normalized actions", norm_action)
    worst_state = _worst_normalized_sample(
        "observation.state_tcp", states_tcp, norm_state, STATE_NAMES, frame_info
    )
    worst_action = _worst_normalized_sample(
        "actions", actions, norm_action, ACTION_NAMES, frame_info
    )
    worst = max([worst_state, worst_action], key=lambda item: item["abs_normalized"])
    print(f"worst normalized sample={worst}")
    state_abs = np.abs(norm_state)
    action_abs = np.abs(norm_action)
    max_abs = max(float(state_abs.max()), float(action_abs.max()))
    p99_abs = max(
        float(np.percentile(state_abs, 99, axis=0).max()),
        float(np.percentile(action_abs, 99, axis=0).max()),
    )
    pass_value = max_abs if fail_on_max else p99_abs
    diagnostics = {
        "normalized_state_tcp": state_summary,
        "normalized_actions": action_summary,
        "worst_state_tcp": worst_state,
        "worst_actions": worst_action,
        "worst": worst,
        "max_abs": max_abs,
        "p99_abs": p99_abs,
        "threshold": float(threshold),
        "fail_on_max": bool(fail_on_max),
        "passed": bool(pass_value <= threshold),
    }
    if pass_value > threshold:
        check_name = "max_abs" if fail_on_max else "p99_abs"
        diagnostics["error_message"] = (
            f"Normalized peg-insertion data is still OOD: {check_name}={pass_value:.3f} "
            f"> threshold={threshold}. Worst field={worst['field']} "
            f"dim={worst['dim_name']} raw={worst['raw']:.6g} "
            f"normalized={worst['normalized']:.6g}. Revisit state/action "
            "alignment or use peg-insertion-specific norm stats."
        )
    elif max_abs > threshold:
        diagnostics["warning_message"] = (
            f"Normalized peg-insertion data has sparse max outliers: "
            f"max_abs={max_abs:.3f} > threshold={threshold}, but p99_abs={p99_abs:.3f} "
            f"passes. Worst field={worst['field']} dim={worst['dim_name']} "
            f"raw={worst['raw']:.6g} normalized={worst['normalized']:.6g}."
        )
        print(diagnostics["warning_message"])
    return diagnostics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical")
    p.add_argument("--norm-stats-json", default=None, help="Optional OpenPI norm_stats.json for a post-conversion quantile check.")
    p.add_argument("--norm-threshold", type=float, default=20.0)
    p.add_argument("--fail-on-norm-max", action="store_true", help="Fail when any single normalized value exceeds --norm-threshold. By default the check fails on p99_abs so sparse Euler singularities are recorded but do not block conversion.")
    p.add_argument("--allow-ood", action="store_true", help="Write converted data and diagnostics even if the optional norm check exceeds the threshold.")
    p.add_argument("--render-backend", default=None, help="Optional ManiSkill render backend, for example gpu:0.")
    p.add_argument(
        "--overwrite-actions-from-fk",
        action="store_true",
        help=(
            "Legacy mode: overwrite actions from adjacent qpos FK deltas. "
            "Do not use this for controller-domain datasets collected with "
            "collect_peg_insertion_controller_data.py."
        ),
    )
    args = p.parse_args()
    dd = osp.abspath(args.data_dir)
    df, pq_path, layout = _load_dataset_frame(dd)
    print(f"Loaded {len(df)} frames, {df[chr(101)+chr(112)+chr(105)+chr(115)+chr(111)+chr(100)+chr(101)+chr(95)+chr(105)+chr(110)+chr(100)+chr(101)+chr(120)].nunique()} episodes")
    if args.overwrite_actions_from_fk:
        print(f"Regenerating actions as {describe_action_semantics()}")
    else:
        print("Regenerating observation.state_tcp only; preserving existing actions.")
    env = make_env(render_backend=args.render_backend)
    uw = env.unwrapped
    ep_ids = sorted(df["episode_index"].unique())
    all_state_tcp = []
    all_actions = []
    for eid in tqdm(ep_ids, desc="FK convert"):
        ep_df = df[df["episode_index"] == eid].reset_index(drop=True)
        st_tcp, acts = process_episode(uw, ep_df)
        all_state_tcp.extend(st_tcp.tolist())
        all_actions.extend(acts.tolist())
    env.close()
    df["observation.state_tcp"] = all_state_tcp
    df["debug.fk_delta_action"] = all_actions
    if args.overwrite_actions_from_fk:
        df["actions"] = all_actions
    import pyarrow as pa, pyarrow.parquet as pq
    if layout == "single":
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, pq_path)
    else:
        for eid in ep_ids:
            eid=int(eid); chunk=eid//1000
            cdir=osp.join(dd,"data",f"chunk-{chunk:03d}"); os.makedirs(cdir,exist_ok=True)
            ep_df=df[df["episode_index"]==eid].reset_index(drop=True)
            pq.write_table(pa.Table.from_pandas(ep_df,preserve_index=False),osp.join(cdir,f"episode_{eid:06d}.parquet"))
    stats_path = osp.join(dd, "meta", "stats.json")
    stats = json.load(open(stats_path))
    a = np.stack(df["actions"].values)
    stats["actions"] = {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "max": a.max(0).tolist(), "min": a.min(0).tolist(), "count": [len(a)]}
    st = np.stack(df["observation.state_tcp"].values)
    stats["observation.state_tcp"] = {"mean": st.mean(0).tolist(), "std": st.std(0).tolist(), "max": st.max(0).tolist(), "min": st.min(0).tolist(), "count": [len(st)]}
    json.dump(stats, open(stats_path, "w"), indent=2)
    info_path = osp.join(dd, "meta", "info.json")
    info = json.load(open(info_path))
    info["features"]["observation.state_tcp"] = {"dtype": "float32", "shape": [SDIM_TCP], "names": STATE_NAMES, "fps": info["fps"]}
    info["features"]["actions"] = {"dtype": "float32", "shape": [ADIM], "names": ACTION_NAMES, "fps": info["fps"]}
    info["features"]["debug.fk_delta_action"] = {"dtype": "float32", "shape": [ADIM], "names": ACTION_NAMES, "fps": info["fps"]}
    json.dump(info, open(info_path, "w"), indent=2)
    diagnostics = {
        "action_semantics": describe_action_semantics(),
        "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES,
        "actions": _summarize_array("actions", a),
        "state_tcp": _summarize_array("state_tcp", st),
        "gripper_action_unique": sorted(np.unique(a[:, 6]).astype(float).tolist()),
    }
    frame_info = {
        "episode_index": df["episode_index"].to_numpy(),
        "frame_index": df["frame_index"].to_numpy(),
        "index": df["index"].to_numpy(),
    }
    norm_diagnostics = _check_norm_stats(
        a,
        st,
        _load_openpi_norm_stats(args.norm_stats_json),
        args.norm_threshold,
        frame_info=frame_info,
        fail_on_max=args.fail_on_norm_max,
    )
    if norm_diagnostics is not None:
        diagnostics["norm_check"] = norm_diagnostics
    diagnostics_path = osp.join(dd, "meta", "conversion_diagnostics.json")
    json.dump(diagnostics, open(diagnostics_path, "w"), indent=2)
    print(f"Wrote diagnostics {diagnostics_path}")
    if (
        norm_diagnostics is not None
        and not norm_diagnostics["passed"]
        and not args.allow_ood
    ):
        raise ValueError(norm_diagnostics["error_message"])
    print(f"Wrote {pq_path}")

if __name__ == "__main__":
    main()
