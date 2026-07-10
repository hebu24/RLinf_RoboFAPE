#!/usr/bin/env python3
"""Randomize task descriptions for an existing v2.0 dataset."""
import argparse, json, os, os.path as osp, glob, random
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm

DESC_FILE = "/home/gpu4/yingxi/RoboFPE/mani_envs/data_collection/task_descriptions/peg_insertion_vertical.json"
SCHEMA = pa.schema([
    pa.field("actions", pa.list_(pa.float32())),
    pa.field("observation.state", pa.list_(pa.float32())),
    pa.field("observation.state_tcp", pa.list_(pa.float32())),
    pa.field("timestamp", pa.float32()),
    pa.field("frame_index", pa.int64()),
    pa.field("episode_index", pa.int64()),
    pa.field("index", pa.int64()),
    pa.field("task_index", pa.int64()),
    pa.field("task", pa.string()),
])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--desc-file", default=DESC_FILE)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dd = osp.abspath(args.data_dir)
    descriptions = json.load(open(args.desc_file))["PegInsertionVertical-v1"]
    n_desc = len(descriptions)
    task_to_idx = {d: i for i, d in enumerate(descriptions)}
    print(f"Loaded {n_desc} task descriptions")
    rng = random.Random(args.seed)
    pq_files = sorted(glob.glob(osp.join(dd, "data", "chunk-*", "episode_*.parquet")))
    print(f"Found {len(pq_files)} parquet files")
    ep_tasks = {}
    for f in tqdm(pq_files, desc="Updating parquets"):
        eid = int(osp.basename(f).split("_")[1].split(".")[0])
        desc = rng.choice(descriptions)
        tidx = task_to_idx[desc]
        df = pd.read_parquet(f)
        n = len(df)
        df["task"] = desc
        df["task_index"] = tidx
        table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
        pq.write_table(table, f)
        ep_tasks[eid] = (desc, tidx, n)
    nl = chr(10)
    with open(osp.join(dd, "meta", "tasks.jsonl"), "w") as f:
        for i, desc in enumerate(descriptions):
            f.write(json.dumps({"task_index": i, "task": desc}) + nl)
    print(f"Wrote tasks.jsonl ({n_desc} tasks)")
    with open(osp.join(dd, "meta", "episodes.jsonl"), "w") as f:
        for eid in sorted(ep_tasks.keys()):
            desc, tidx, length = ep_tasks[eid]
            f.write(json.dumps({"episode_index": eid, "tasks": [desc], "length": length}) + nl)
    print(f"Wrote episodes.jsonl ({len(ep_tasks)} episodes)")
    info = json.load(open(osp.join(dd, "meta", "info.json")))
    info["total_tasks"] = n_desc
    json.dump(info, open(osp.join(dd, "meta", "info.json"), "w"), indent=2)
    print(f"Updated info.json (total_tasks={n_desc})")
    print("Done!")

if __name__ == "__main__":
    main()
