#!/usr/bin/env python3
import argparse, json, os, os.path as osp, shutil
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm

TASK = "insert the blue peg vertically into the orange hole"
FPS = 20; SDIM = 8; SDIM_TCP = 8; ADIM = 7; CSIZE = 1000
IW = 224; IH = 224; RW = 640; RH = 480
CAMERAS = ["top", "wrist", "render"]
def build_features():
    feat = {
        "observation.state": {"dtype": "float32", "shape": [SDIM], "names": [f"joint_{i}" for i in range(SDIM)], "fps": float(FPS)},
        "observation.state_tcp": {"dtype": "float32", "shape": [SDIM_TCP], "names": ["tcp_x", "tcp_y", "tcp_z", "roll", "pitch", "yaw", "finger0", "finger1"], "fps": float(FPS)},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": float(FPS)},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(FPS)},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(FPS)},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(FPS)},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": float(FPS)},
        "task": {"dtype": "string", "shape": [1], "names": None, "fps": float(FPS)},
        "actions": {"dtype": "float32", "shape": [ADIM], "names": [f"action_{i}" for i in range(ADIM)], "fps": float(FPS)},
    }
    for c, h, w in [("top", IH, IW), ("wrist", IH, IW), ("render", RH, RW)]:
        feat[f"observation.images.{c}"] = {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"], "info": {"video.fps": float(FPS), "video.height": h, "video.width": w, "video.channels": 3, "video.codec": "mp4v", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}}
    return feat
def compute_stats(df):
    s = {}
    for col, key in [("actions", "actions"), ("observation.state", "observation.state"), ("observation.state_tcp", "observation.state_tcp")]:
        a = np.stack(df[col].values)
        s[key] = {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "max": a.max(0).tolist(), "min": a.min(0).tolist(), "count": [len(a)]}
    for fld in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        v = df[fld].values
        s[fld] = {"mean": [float(v.mean())], "std": [float(v.std())], "max": [int(v.max())] if fld != "timestamp" else [float(v.max())], "min": [int(v.min())] if fld != "timestamp" else [float(v.min())], "count": [len(v)]}
    return s
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    args = p.parse_args()
    dd = osp.abspath(args.data_dir)
    print(f"Converting {dd} from v3.0 to v2.0 format...")
    old_pq = osp.join(dd, "data", "chunk-000", "file-000.parquet")
    if not osp.exists(old_pq):
        print(f"ERROR: {old_pq} not found"); return
    df = pd.read_parquet(old_pq)
    print(f"Loaded {len(df)} frames, {df['episode_index'].nunique()} episodes")
    print(f"Columns: {list(df.columns)}")
    if "action" in df.columns and "actions" not in df.columns:
        df = df.rename(columns={"action": "actions"}); print("Renamed 'action' -> 'actions'")
    if "observation.state_tcp" not in df.columns:
        print("ERROR: observation.state_tcp missing. Run convert_qpos_to_tcp_proprio.py first!"); return
    data_dir = osp.join(dd, "data")
    if osp.exists(data_dir): shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    ep_ids = sorted(df["episode_index"].unique())
    schema = pa.schema([pa.field("actions", pa.list_(pa.float32())), pa.field("observation.state", pa.list_(pa.float32())), pa.field("observation.state_tcp", pa.list_(pa.float32())), pa.field("timestamp", pa.float32()), pa.field("frame_index", pa.int64()), pa.field("episode_index", pa.int64()), pa.field("index", pa.int64()), pa.field("task_index", pa.int64()), pa.field("task", pa.string())])
    total_chunks = 0
    for eid in tqdm(ep_ids, desc="Splitting parquets"):
        ep_df = df[df["episode_index"] == eid].reset_index(drop=True)
        chunk = eid // CSIZE
        chunk_dir = osp.join(data_dir, f"chunk-{chunk:03d}")
        os.makedirs(chunk_dir, exist_ok=True)
        if chunk + 1 > total_chunks: total_chunks = chunk + 1
        ep_path = osp.join(chunk_dir, f"episode_{eid:06d}.parquet")
        table = pa.Table.from_pandas(ep_df, schema=schema, preserve_index=False)
        pq.write_table(table, ep_path)
    print(f"Wrote {len(ep_ids)} parquets in {total_chunks} chunks")
    videos_dir = osp.join(dd, "videos")
    temp_videos = osp.join(dd, "_videos_old")
    if osp.exists(videos_dir): shutil.move(videos_dir, temp_videos)
    os.makedirs(videos_dir, exist_ok=True)
    for cam in CAMERAS:
        vkey = f"observation.images.{cam}"
        old_cam_dir = osp.join(temp_videos, vkey, "chunk-000")
        if not osp.exists(old_cam_dir):
            print(f"WARNING: {old_cam_dir} not found"); continue
        old_files = sorted(Path(old_cam_dir).glob("file-*.mp4"))
        for old_f in tqdm(old_files, desc=f"Moving {cam}"):
            ep_idx = int(old_f.stem.split("-")[1])
            chunk = ep_idx // CSIZE
            new_dir = osp.join(videos_dir, f"chunk-{chunk:03d}", vkey)
            os.makedirs(new_dir, exist_ok=True)
            shutil.move(str(old_f), osp.join(new_dir, f"episode_{ep_idx:06d}.mp4"))
    shutil.rmtree(temp_videos, ignore_errors=True)
    print("Videos moved to v2.0 layout")
    TE = int(len(ep_ids)); TF = int(len(df)); total_chunks = int(total_chunks)
    feat = build_features()
    dsz = sum(f.stat().st_size for f in Path(data_dir).rglob("*.parquet"))
    info = {"codebase_version": "v2.0", "robot_type": "panda_wristcam", "total_episodes": TE, "total_frames": TF, "total_tasks": 1, "total_videos": TE * 3, "total_chunks": total_chunks, "chunks_size": CSIZE, "fps": FPS, "data_files_size_in_mb": int(dsz / (1024 * 1024)), "splits": {"train": f"0:{TE}"}, "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", "features": feat}
    with open(osp.join(dd, "meta", "info.json"), "w") as f: json.dump(info, f, indent=2)
    print(f"Wrote info.json (v2.0, {TE} eps, {TF} frames, {total_chunks} chunks)")
    stats = compute_stats(df)
    with open(osp.join(dd, "meta", "stats.json"), "w") as f: json.dump(stats, f, indent=2)
    print("Wrote stats.json")
    with open(osp.join(dd, "meta", "tasks.jsonl"), "w") as f: f.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    print("Wrote tasks.jsonl")
    with open(osp.join(dd, "meta", "episodes.jsonl"), "w") as f:
        for eid in ep_ids:
            T = len(df[df["episode_index"] == eid])
            f.write(json.dumps({"episode_index": int(eid), "tasks": [TASK], "length": T}) + "\n")
    print(f"Wrote episodes.jsonl ({len(ep_ids)} episodes)")
    for old_artifact in [osp.join(dd, "meta", "episodes", "chunk-000", "file-000.parquet"), osp.join(dd, "meta", "tasks.parquet")]:
        if osp.exists(old_artifact): os.remove(old_artifact); print(f"Removed {osp.basename(old_artifact)}")
    print(f"\nDone! Dataset converted to v2.0 format at {dd}")

if __name__ == "__main__":
    main()
