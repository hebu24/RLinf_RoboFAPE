#!/usr/bin/env python3
"""Render 224x224 base-camera images at randomization boundaries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import os.path as osp
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from mani_skill.utils import sapien_utils
from transforms3d.euler import euler2quat

ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
TASK = osp.join(
    ROOT, "rlinf", "envs", "maniskill", "tasks", "peg_insertion_vertical.py"
)
SIZE = 224
DIRS = {
    "e": (1, 0),
    "ne": (1, 1),
    "n": (0, 1),
    "nw": (-1, 1),
    "w": (-1, 0),
    "sw": (-1, -1),
    "s": (0, -1),
    "se": (1, -1),
}


@dataclass(frozen=True)
class Case:
    name: str
    hole: np.ndarray
    peg: np.ndarray
    radius: float


def load_task():
    s = importlib.util.spec_from_file_location("_peg_extremes", TASK)
    if s is None or s.loader is None:
        raise RuntimeError(f"Cannot load {TASK}")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def array(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def ray_limit(o, d, b):
    v = []
    for a, c in enumerate(d):
        if c > 0:
            v.append((b[a, 1] - o[a]) / c)
        elif c < 0:
            v.append((b[a, 0] - o[a]) / c)
    return float(min(v))


def cases(cls):
    hb = np.asarray(cls.hole_xy_randomization_bounds)
    pb = np.asarray(cls.peg_xy_randomization_bounds)
    lo, hi = map(float, cls.peg_relative_hole_radius_bounds)
    out = []
    for xn, x in (("xmin", hb[0, 0]), ("xmax", hb[0, 1])):
        for yn, y in (("ymin", hb[1, 0]), ("ymax", hb[1, 1])):
            h = np.array([x, y], dtype=np.float32)
            for dn, raw in DIRS.items():
                d = np.asarray(raw, dtype=np.float32)
                d /= np.linalg.norm(d)
                far = min(hi, ray_limit(h, d, pb))
                if far + 1e-7 < lo:
                    continue
                for rn, r in (("rmin", lo), ("rmax", far)):
                    out.append(
                        Case(f"hole_{xn}_{yn}__{dn}__{rn}", h.copy(), h + r * d, r)
                    )
    return out


def pose(xy, z):
    return {
        "p": [float(xy[0]), float(xy[1]), float(z)],
        "q": np.asarray(euler2quat(0, np.pi / 2, 0)).tolist(),
    }


def render(env, c, seed):
    u = env.unwrapped
    obs, _ = env.reset(
        seed=seed,
        options={
            "randomize_initial_poses": False,
            "peg_pose": pose(c.peg, u.table_top_z + u.peg_half_length),
            "hole_pose": pose(c.hole, u.table_top_z + u.hole_half_depth),
            "robot_qpos": u.default_robot_qpos.copy(),
        },
    )
    rgb = array(obs["sensor_data"]["base_camera"]["rgb"])[0]
    if rgb.shape != (SIZE, SIZE, 3):
        raise RuntimeError(f"Unexpected shape {rgb.shape}")
    return rgb.astype(np.uint8, copy=False)


def sheet(items, path, cols):
    lh = 58
    rows = math.ceil(len(items) / cols)
    out = np.full((rows * (SIZE + lh), cols * SIZE, 3), 245, np.uint8)
    for i, (c, rgb) in enumerate(items):
        row, col = divmod(i, cols)
        x = col * SIZE
        y = row * (SIZE + lh)
        out[y : y + SIZE, x : x + SIZE] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        labels = (
            c.name.split("__")[0],
            "__".join(c.name.split("__")[1:]),
            f"h=({c.hole[0]:+.3f},{c.hole[1]:+.3f}) p=({c.peg[0]:+.3f},{c.peg[1]:+.3f})",
        )
        for j, label in enumerate(labels):
            cv2.putText(
                out,
                label,
                (x + 3, y + SIZE + 16 + j * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    if not cv2.imwrite(path, out):
        raise RuntimeError(f"Cannot write {path}")


def xyz(value):
    result = np.fromstring(value, sep=",", dtype=np.float32)
    if result.shape != (3,):
        raise argparse.ArgumentTypeError(f"expected x,y,z, got {value!r}")
    return result


def args():
    p = argparse.ArgumentParser(
        description="Render valid extreme XY randomization cases"
    )
    p.add_argument("--fov", type=float, default=0.6)
    p.add_argument("--camera-eye", type=xyz, default=xyz("0.60,-0.05,0.55"))
    p.add_argument("--camera-target", type=xyz, default=xyz("0.0,0.0,0.15"))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-backend", default="gpu:0")
    p.add_argument("--columns", type=int, default=4)
    return p.parse_args()


def main():
    a = args()
    if not math.isfinite(a.fov) or not 0 < a.fov < math.pi:
        raise ValueError("--fov must be in (0, pi)")
    if a.columns < 1:
        raise ValueError("--columns must be positive")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    m = load_task()
    import gymnasium as gym

    tag = f"{a.fov:.3f}".replace(".", "p")
    out = osp.abspath(a.output_dir or f"base_camera_fov_{tag}_extremes")
    frames = osp.join(out, "frames")
    os.makedirs(frames, exist_ok=True)
    cs = cases(m.PegInsertionVerticalEnv)
    camera_pose = sapien_utils.look_at(a.camera_eye, a.camera_target)
    env = gym.make(
        "PegInsertionVertical-v1",
        num_envs=1,
        obs_mode="rgb",
        robot_uids="panda_wristcam",
        control_mode="pd_joint_pos",
        sim_backend="cpu",
        render_backend=a.render_backend,
        render_mode="all",
        reward_mode="normalized_dense",
        max_episode_steps=600,
        sensor_configs={
            "shader_pack": "default",
            "base_camera": {
                "pose": camera_pose,
                "width": SIZE,
                "height": SIZE,
                "fov": a.fov,
            },
        },
        human_render_camera_configs={"shader_pack": "default"},
    )
    items = []
    meta = []
    try:
        for i, c in enumerate(cs):
            rgb = render(env, c, a.seed)
            fp = osp.join(frames, f"{i:03d}__{c.name}.png")
            if not cv2.imwrite(fp, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Cannot write {fp}")
            items.append((c, rgb))
            meta.append(
                {
                    "index": i,
                    "name": c.name,
                    "hole_xy": c.hole.tolist(),
                    "peg_xy": c.peg.tolist(),
                    "relative_peg_xy": (c.peg - c.hole).tolist(),
                    "radius": c.radius,
                    "frame": osp.relpath(fp, out),
                }
            )
        sp = osp.join(out, "contact_sheet.png")
        sheet(items, sp, a.columns)
        with open(osp.join(out, "cases.json"), "w", encoding="utf-8") as g:
            json.dump(
                {
                    "fov_rad": a.fov,
                    "fov_deg": math.degrees(a.fov),
                    "camera_eye": a.camera_eye.tolist(),
                    "camera_target": a.camera_target.tolist(),
                    "camera_quaternion": array(camera_pose.q)[0].tolist(),
                    "image_shape": [SIZE, SIZE, 3],
                    "case_count": len(cs),
                    "cases": meta,
                },
                g,
                indent=2,
            )
        print(f"Rendered {len(cs)} valid extreme cases")
        print(f"Frames: {frames}")
        print(f"Contact sheet: {sp}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
