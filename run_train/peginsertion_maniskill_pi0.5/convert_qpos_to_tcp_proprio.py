#!/usr/bin/env python3
import argparse, json, os, os.path as osp, importlib.util as ilu
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transforms3d.euler import mat2euler
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles

s = ilu.spec_from_file_location("_peg", "/opt/yingxi/RLinf_RoboFAPE/rlinf/envs/maniskill/tasks/peg_insertion_vertical.py")
m = ilu.module_from_spec(s); s.loader.exec_module(m)
import gymnasium as gym

SDIM_TCP = 8
ADIM = 7

def make_env():
    env = gym.make("PegInsertionVertical-v1", num_envs=1, obs_mode="state", robot_uids="panda_wristcam", control_mode="pd_joint_pos", sim_backend="cpu", max_episode_steps=600)
    env.reset(seed=0)
    return env

def fk_proprio(uw, qpos8):
    robot = uw.agent.robot
    grip = float(qpos8[7])
    robot_qpos = np.array(list(qpos8[:7]) + [grip, grip], dtype=np.float32)
    robot.set_qpos(torch.tensor(robot_qpos).unsqueeze(0))
    tcp_root = uw.agent.robot.pose.inv() * uw.agent.tcp.pose
    T = tcp_root.to_transformation_matrix().detach().cpu().numpy()[0]
    pos = T[:3, 3]
    eu = mat2euler(T[:3, :3], "sxyz")
    proprio = np.concatenate([pos, eu, [grip, grip]]).astype(np.float32)
    return proprio, T

def process_episode(uw, ep_df):
    T = len(ep_df)
    states_tcp = np.zeros((T, SDIM_TCP), dtype=np.float32)
    poses_pos = np.zeros((T, 3), dtype=np.float64)
    poses_R = np.zeros((T, 3, 3), dtype=np.float64)
    orig_actions = np.stack(ep_df["action"].values)
    for t in range(T):
        qpos8 = np.asarray(ep_df["observation.state"].iloc[t], dtype=np.float32)
        proprio, Tmat = fk_proprio(uw, qpos8)
        states_tcp[t] = proprio
        poses_pos[t] = Tmat[:3, 3]
        poses_R[t] = Tmat[:3, :3]
    actions = np.zeros((T, ADIM), dtype=np.float32)
    for t in range(T - 1):
        dp = poses_pos[t + 1] - poses_pos[t]
        R_delta = poses_R[t + 1] @ poses_R[t].T
        R_t = torch.tensor(R_delta).unsqueeze(0)
        dr = matrix_to_euler_angles(R_t, "XYZ")[0].numpy()
        g = float(orig_actions[t, 6])
        actions[t] = np.concatenate([dp, dr, [g]])
    if T > 1:
        actions[-1] = actions[-2]
    return states_tcp, actions

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical")
    args = p.parse_args()
    dd = osp.abspath(args.data_dir)
    pq_path = osp.join(dd, "data", "chunk-000", "file-000.parquet")
    df = pd.read_parquet(pq_path)
    print(f"Loaded {len(df)} frames, {df[chr(101)+chr(112)+chr(105)+chr(115)+chr(111)+chr(100)+chr(101)+chr(95)+chr(105)+chr(110)+chr(100)+chr(101)+chr(120)].nunique()} episodes")
    env = make_env()
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
    df["action"] = all_actions
    import pyarrow as pa, pyarrow.parquet as pq
    schema = pa.schema([
        pa.field("action", pa.list_(pa.float32())),
        pa.field("observation.state", pa.list_(pa.float32())),
        pa.field("observation.state_tcp", pa.list_(pa.float32())),
        pa.field("timestamp", pa.float32()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("task", pa.string()),
    ])
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, pq_path)
    stats_path = osp.join(dd, "meta", "stats.json")
    stats = json.load(open(stats_path))
    a = np.stack(df["action"].values)
    stats["action"] = {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "max": a.max(0).tolist(), "min": a.min(0).tolist(), "count": [len(a)]}
    st = np.stack(df["observation.state_tcp"].values)
    stats["observation.state_tcp"] = {"mean": st.mean(0).tolist(), "std": st.std(0).tolist(), "max": st.max(0).tolist(), "min": st.min(0).tolist(), "count": [len(st)]}
    json.dump(stats, open(stats_path, "w"), indent=2)
    info_path = osp.join(dd, "meta", "info.json")
    info = json.load(open(info_path))
    info["features"]["observation.state_tcp"] = {"dtype": "float32", "shape": [SDIM_TCP], "names": ["tcp_x", "tcp_y", "tcp_z", "roll", "pitch", "yaw", "finger0", "finger1"], "fps": info["fps"]}
    json.dump(info, open(info_path, "w"), indent=2)
    print(f"Done. action mean={a.mean(0)}")
    print(f"state_tcp mean={st.mean(0)}")
    print(f"Wrote {pq_path}")

if __name__ == "__main__":
    main()
