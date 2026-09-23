"""Lightweight UprightStack camera sampling shared by eval entrypoints."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _stable_seed(token: Any) -> int:
    digest = hashlib.sha256(str(token).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _look_at_pose(eye: np.ndarray, target: np.ndarray) -> list[float]:
    from mani_skill.utils import sapien_utils

    pose = sapien_utils.look_at(eye, target)
    return [
        *np.asarray(pose.p, dtype=np.float64).reshape(-1)[:3].tolist(),
        *np.asarray(pose.q, dtype=np.float64).reshape(-1)[:4].tolist(),
    ]


def sample_uprightstack_camera_spec(seed_token: Any) -> dict[str, dict[str, list[float]]]:
    """Match RoboFPE collect_sft_data.py's UprightStack camera ranges."""
    rng = np.random.default_rng(_stable_seed(seed_token))
    top_delta = rng.uniform([-0.04, -0.04, -0.03], [0.04, 0.04, 0.03])
    wrist_delta = rng.uniform([-0.005, -0.005, -0.005], [0.005, 0.005, 0.005])
    render_delta = rng.uniform([-0.02, -0.02, -0.015], [0.02, 0.02, 0.015])

    base_eye = np.array([-0.3, 0.0, 0.6], dtype=np.float64)
    base_target = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    render_eye = np.array([0.2, 0.8, 0.3], dtype=np.float64)
    render_target = np.array([0.0, 0.0, 0.15], dtype=np.float64)
    return {
        "base": {
            "pose": _look_at_pose(base_eye + top_delta, base_target),
        },
        "wrist": {
            "pose": [
                float(wrist_delta[0]), float(wrist_delta[1]), float(wrist_delta[2]),
                1.0, 0.0, 0.0, 0.0,
            ],
        },
        "render": {
            "pose": _look_at_pose(render_eye + render_delta, render_target),
        },
    }
