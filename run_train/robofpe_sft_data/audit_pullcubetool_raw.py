#!/usr/bin/env python3
"""Audit PullCubeTool-golf raw episodes and build a conversion allowlist."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import h5py
import numpy as np


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz = q[:, 1:]
    return v + 2 * np.cross(xyz, np.cross(xyz, v) + q[:, :1] * v)


def video_ok(path: Path) -> bool:
    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok_first, first = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames - 1))
        ok_last, last = capture.read()
        return bool(ok_first and ok_last and first is not None and last is not None)
    finally:
        capture.release()


def audit(path: Path) -> dict:
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    metadata = sidecar["episodes"][0]
    with h5py.File(path, "r") as h5:
        traj = h5["traj_0"]
        ball = traj["env_states/actors/cube"][:, :3]
        tool_state = traj["env_states/actors/l_shape_tool"][:, :7]
        tcp = traj["obs/extra/tcp_pose"][:, :3]
        base_xy = traj["env_states/articulations/panda_wristcam"][0, :2]
        distances = np.linalg.norm(ball[:, :2] - base_xy, axis=1)
        tcp_tool = np.linalg.norm(tcp - tool_state[:, :3], axis=1)
        hook = tool_state[:, :3] + rotate(
            tool_state[:, 3:7], np.array([0.225, 0.04, 0.0])
        )
        hook_ball = np.linalg.norm(hook - ball, axis=1)
        length = int(traj["actions"].shape[0])
    progress = float(distances[0] - distances[-1])
    final_distance = float(distances[-1])
    held_q75 = float(np.quantile(tcp_tool, 0.75))
    min_hook_ball = float(hook_ball.min())
    video = path.with_suffix(".mp4")
    reasons = []
    if metadata.get("task_id") != "PullCubeTool-golf" or metadata.get("success") is not True:
        reasons.append("metadata")
    if progress < 0.05 or final_distance >= 0.6:
        reasons.append("false_success")
    if held_q75 >= 0.08:
        reasons.append("no_grasp")
    if min_hook_ball >= 0.10:
        reasons.append("no_tool_contact")
    if not video_ok(video):
        reasons.append("render_bad")
    return {
        "source_h5": str(path.resolve()),
        "render_video": str(video.resolve()),
        "seed": int(metadata.get("seed", metadata["episode_seed"])),
        "length": length,
        "ball_progress_to_base": progress,
        "final_ball_base_distance": final_distance,
        "tcp_tool_distance_q75": held_q75,
        "min_hook_ball_distance": min_hook_ball,
        "status": "rejected" if reasons else "accepted",
        "reasons": reasons,
    }


def stratified_sample(rows: list[dict], count: int) -> list[dict]:
    per_group = max(1, count // 5)
    selected = []
    selected += sorted(rows, key=lambda row: row["ball_progress_to_base"])[:per_group]
    selected += sorted(rows, key=lambda row: -row["final_ball_base_distance"])[:per_group]
    selected += sorted(rows, key=lambda row: row["length"])[:per_group]
    selected += sorted(rows, key=lambda row: -row["length"])[:per_group]
    rng = random.Random(20260921)
    selected += rng.sample(rows, min(per_group, len(rows)))
    unique = {row["source_h5"]: row for row in selected}
    if len(unique) < count:
        for row in rows:
            unique.setdefault(row["source_h5"], row)
            if len(unique) == count:
                break
    return list(unique.values())[:count]


def video_frames(path: Path, count: int = 5) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        for index in np.linspace(0, max(0, total - 1), count).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {index}: {path}")
            frames.append(cv2.resize(frame, (160, 160)))
    finally:
        capture.release()
    return frames


def write_sheets(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for page_start in range(0, len(rows), 10):
        page = rows[page_start:page_start + 10]
        strips = []
        for row in page:
            strip = np.concatenate(video_frames(Path(row["render_video"])), axis=1)
            cv2.rectangle(strip, (0, 0), (800, 22), (0, 0, 0), -1)
            label = f"seed={row['seed']} progress={row['ball_progress_to_base']:.3f} final={row['final_ball_base_distance']:.3f}"
            cv2.putText(strip, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            strips.append(strip)
        cv2.imwrite(str(output / f"review_{page_start // 10:03d}.jpg"), np.concatenate(strips))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contact-sheet-dir", required=True)
    parser.add_argument("--visual-sample-size", type=int, default=100)
    args = parser.parse_args()

    paths = sorted(Path(args.input).rglob("*.h5"))
    rows = [audit(path) for path in paths]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    sample = stratified_sample([row for row in rows if row["status"] == "accepted"], args.visual_sample_size)
    write_sheets(sample, Path(args.contact_sheet_dir))
    print(json.dumps({
        "episodes": len(rows),
        "accepted": sum(row["status"] == "accepted" for row in rows),
        "rejected": sum(row["status"] == "rejected" for row in rows),
        "visual_sample": len(sample),
    }, indent=2))


if __name__ == "__main__":
    main()
