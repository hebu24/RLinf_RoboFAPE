# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared pi0.5 state/action helpers for PegInsertionVertical.

The model action label is a physical target-delta TCP action:
``[dx, dy, dz, droll, dpitch, dyaw, gripper]`` with Euler XYZ rotation.
At evaluation time this is sent to ManiSkill Panda
``pd_ee_target_delta_pose`` after conversion to the controller's normalized
action space.

The proprio state is an aligned TCP pose used only as model input.  The fixed
alignment maps the peg-insertion TCP distribution into the same rough coordinate
range as the 25Main pi0.5 ManiSkill SFT checkpoint norm stats, avoiding the
extreme normalized Euler values produced by the raw TCP frame.
"""

from __future__ import annotations

import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat

PI05_STATE_DIM = 8
PI05_ACTION_DIM = 7
PANDA_EE_DELTA_POS_BOUND = 0.1
PANDA_EE_DELTA_ROT_BOUND = 0.1
PANDA_GRIPPER_OPEN = 1.0
PANDA_GRIPPER_CLOSE = -1.0

# State/action quantile stats from the base RLinf-Pi05-ManiSkill-25Main-SFT
# checkpoint.  Keep them local so data collection can reject obviously OOD
# samples before SFT.
_BASE_MANISKILL_STATE_Q01 = np.array(
    [
        -0.20252828299999237,
        -0.21026643007993698,
        -0.21924416363239288,
        -0.017773685976862907,
        0.8226010286808014,
        -1.5134326148033141,
        0.021859830990433694,
        0.021078145131468773,
    ],
    dtype=np.float64,
)
_BASE_MANISKILL_STATE_Q99 = np.array(
    [
        0.41199415922164917,
        0.48613785028457646,
        0.7463473677635193,
        0.016534310169517997,
        1.4169902801513672,
        1.8775047111511234,
        0.03700000047683716,
        0.03700000047683716,
    ],
    dtype=np.float64,
)
_BASE_MANISKILL_ACTION_Q01 = np.array(
    [
        -0.025044564604759217,
        -0.03014469236135483,
        -0.03136833757162094,
        -0.005981205031275749,
        -0.02122248336672783,
        -0.20235729217529297,
        -1.0,
    ],
    dtype=np.float64,
)
_BASE_MANISKILL_ACTION_Q99 = np.array(
    [
        0.02858110427856446,
        0.03280403457581998,
        0.03535533085465432,
        0.005980716086924076,
        0.013883259352296595,
        0.2048933058977127,
        1.0,
    ],
    dtype=np.float64,
)
_BASE_MANISKILL_STATE_POS_MEAN = np.array(
    [0.07715192926514126, 0.17562641970592635, 0.3297820340626251],
    dtype=np.float64,
)
_BASE_MANISKILL_STATE_EULER_MEAN = np.array(
    [-0.00012817750709427617, 1.0532821746532337, 0.28805883864091536],
    dtype=np.float64,
)
_BASE_MANISKILL_STATE_EULER_Q01 = _BASE_MANISKILL_STATE_Q01[3:6]
_BASE_MANISKILL_STATE_EULER_Q99 = _BASE_MANISKILL_STATE_Q99[3:6]

# Mean raw TCP state measured from the existing peg-insertion dataset before
# alignment.  Kept explicit so offline conversion and online inference apply
# the same transform.
_PEG_RAW_STATE_POS_MEAN = np.array(
    [0.5473582148551941, -0.06287267059087753, 0.26079684495925903],
    dtype=np.float64,
)
_PEG_RAW_TCP_NOMINAL_EULER = np.array([np.pi, 0.0, 0.0], dtype=np.float64)

_POS_OFFSET = _BASE_MANISKILL_STATE_POS_MEAN - _PEG_RAW_STATE_POS_MEAN
_ROT_ALIGN = euler2mat(*_BASE_MANISKILL_STATE_EULER_MEAN, axes="sxyz") @ euler2mat(
    *_PEG_RAW_TCP_NOMINAL_EULER, axes="sxyz"
).T


def binary_gripper_from_solver_action(action: np.ndarray) -> float:
    """Convert solver gripper commands to the base pi0.5 binary convention."""
    value = float(np.asarray(action, dtype=np.float32).reshape(-1)[-1])
    return PANDA_GRIPPER_OPEN if value > 0.0 else PANDA_GRIPPER_CLOSE


def model_action_to_panda_env_action(
    action: np.ndarray,
    action_scale: float = 1.0,
) -> np.ndarray:
    """Map physical pi0.5 action labels to Panda normalized env actions.

    ManiSkill Panda ``PDEEPoseController`` maps normalized position actions
    with symmetric ``[-0.1, 0.1]`` bounds, so position uses ``delta / 0.1``.
    Its rotation path clips by vector norm and then multiplies by
    ``rot_lower``; for Panda this is ``-0.1``.  Therefore the normalized
    rotation action must use the opposite sign: ``-delta / 0.1``.

    The gripper dimension is already the normalized Panda command where
    ``+1`` means open and ``-1`` means close.
    """
    env_action = np.asarray(action, dtype=np.float32).copy()
    if env_action.shape[-1] != PI05_ACTION_DIM:
        raise ValueError(
            f"Expected last action dim {PI05_ACTION_DIM}, got {env_action.shape[-1]}"
        )
    env_action[..., :3] = (
        env_action[..., :3] * float(action_scale) / PANDA_EE_DELTA_POS_BOUND
    )
    env_action[..., 3:6] = (
        -env_action[..., 3:6] * float(action_scale) / PANDA_EE_DELTA_ROT_BOUND
    )
    return env_action.astype(np.float32)


def target_delta_model_action(
    prev_target_tcp: np.ndarray,
    next_target_tcp: np.ndarray,
    gripper: float,
) -> np.ndarray:
    """Compute a physical target-delta action in the Panda root frame."""
    prev_target_tcp = np.asarray(prev_target_tcp, dtype=np.float64)
    next_target_tcp = np.asarray(next_target_tcp, dtype=np.float64)
    raw = np.zeros(PI05_ACTION_DIM, dtype=np.float32)
    raw[:3] = next_target_tcp[:3, 3] - prev_target_tcp[:3, 3]
    r_delta = next_target_tcp[:3, :3] @ prev_target_tcp[:3, :3].T
    raw[3:6] = matrix_to_euler_angles(
        torch.as_tensor(r_delta, dtype=torch.float64).unsqueeze(0), "XYZ"
    )[0].numpy()
    raw[6] = float(gripper)
    return raw


def panda_action_bound_summary(actions: np.ndarray) -> dict[str, float | bool]:
    """Summarize whether model actions fit Panda controller physical bounds."""
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, PI05_ACTION_DIM)
    pos_abs_max = float(np.max(np.abs(actions[:, :3]))) if len(actions) else 0.0
    rot_norm_max = (
        float(np.max(np.linalg.norm(actions[:, 3:6], axis=1))) if len(actions) else 0.0
    )
    env_actions = model_action_to_panda_env_action(actions)
    env_pos_abs_max = (
        float(np.max(np.abs(env_actions[:, :3]))) if len(env_actions) else 0.0
    )
    env_rot_norm_max = (
        float(np.max(np.linalg.norm(env_actions[:, 3:6], axis=1)))
        if len(env_actions)
        else 0.0
    )
    return {
        "pos_abs_max": pos_abs_max,
        "rot_norm_max": rot_norm_max,
        "env_pos_abs_max": env_pos_abs_max,
        "env_rot_norm_max": env_rot_norm_max,
        "within_controller_bounds": bool(
            env_pos_abs_max <= 1.0 + 1e-6 and env_rot_norm_max <= 1.0 + 1e-6
        ),
    }


def target_delta_step_count(action: np.ndarray) -> int:
    """Return the minimum Panda target-delta substeps needed for one action."""
    action = np.asarray(action, dtype=np.float32)
    pos_steps = int(
        np.ceil(np.max(np.abs(action[:3])) / PANDA_EE_DELTA_POS_BOUND - 1e-6)
    )
    rot_steps = int(
        np.ceil(np.linalg.norm(action[3:6]) / PANDA_EE_DELTA_ROT_BOUND - 1e-6)
    )
    return max(1, pos_steps, rot_steps)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _normalize_quat(q0 + fraction * (q1 - q0))
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * fraction
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return _normalize_quat(s0 * q0 + s1 * q1)


def interpolate_tcp_matrix(
    start_tcp: np.ndarray,
    end_tcp: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Interpolate a TCP matrix with linear position and quaternion slerp."""
    start_tcp = np.asarray(start_tcp, dtype=np.float64)
    end_tcp = np.asarray(end_tcp, dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = (1.0 - fraction) * start_tcp[:3, 3] + fraction * end_tcp[:3, 3]
    q0 = mat2quat(start_tcp[:3, :3])
    q1 = mat2quat(end_tcp[:3, :3])
    out[:3, :3] = quat2mat(_slerp_quat(q0, q1, fraction))
    return out


def split_target_delta_model_actions(
    prev_target_tcp: np.ndarray,
    next_target_tcp: np.ndarray,
    gripper: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split one target-delta into Panda-bound substeps.

    Returns ``(action, substep_target_tcp)`` pairs.  The last target is exactly
    ``next_target_tcp``.
    """
    full_action = target_delta_model_action(prev_target_tcp, next_target_tcp, gripper)
    steps = target_delta_step_count(full_action)
    while True:
        current = np.asarray(prev_target_tcp, dtype=np.float64)
        pieces: list[tuple[np.ndarray, np.ndarray]] = []
        for idx in range(1, steps + 1):
            target = interpolate_tcp_matrix(
                prev_target_tcp, next_target_tcp, idx / steps
            )
            action = target_delta_model_action(current, target, gripper)
            pieces.append((action, target))
            current = target
        if panda_action_bound_summary(np.stack([item[0] for item in pieces]))[
            "within_controller_bounds"
        ]:
            return pieces
        steps *= 2
        if steps > 64:
            raise ValueError(
                "Unable to split target-delta action within Panda controller bounds"
            )


def quantile_normalize_with_base_stats(values: np.ndarray, field: str) -> np.ndarray:
    """Normalize state/actions with base pi0.5 ManiSkill quantile stats."""
    values = np.asarray(values, dtype=np.float64)
    if field == "state":
        q01 = _BASE_MANISKILL_STATE_Q01
        q99 = _BASE_MANISKILL_STATE_Q99
    elif field == "actions":
        q01 = _BASE_MANISKILL_ACTION_Q01
        q99 = _BASE_MANISKILL_ACTION_Q99
    else:
        raise ValueError(f"Unsupported base stats field: {field}")
    q01 = q01[: values.shape[-1]]
    q99 = q99[: values.shape[-1]]
    return (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def normalized_abs_summary(values: np.ndarray) -> dict[str, float]:
    """Return compact absolute-value summary for normalized arrays."""
    values = np.asarray(values, dtype=np.float64)
    abs_values = np.abs(values.reshape(-1, values.shape[-1]))
    return {
        "max": float(abs_values.max()) if abs_values.size else 0.0,
        "p99": float(np.percentile(abs_values, 99)) if abs_values.size else 0.0,
    }


def _canonicalize_state_euler(euler: np.ndarray) -> np.ndarray:
    """Choose the Euler-equivalent state rotation closest to base pi0.5 stats.

    ``mat2euler`` can return an equivalent representation with roll near
    ``+/-pi`` for a small number of peg-insertion poses.  That is numerically
    valid, but it is far outside the base ManiSkill pi0.5 quantile stats and
    breaks normalization.  Pick the equivalent angle triplet nearest to the
    base state mean under the base quantile scale.
    """
    euler = np.asarray(euler, dtype=np.float64)
    scale = np.maximum(
        _BASE_MANISKILL_STATE_EULER_Q99 - _BASE_MANISKILL_STATE_EULER_Q01,
        1e-6,
    )
    roots = (
        euler,
        np.array([euler[0] + np.pi, np.pi - euler[1], euler[2] + np.pi]),
    )
    shifts = (-2.0 * np.pi, 0.0, 2.0 * np.pi)
    best = None
    best_score = np.inf
    for root in roots:
        for da in shifts:
            for db in shifts:
                for dc in shifts:
                    candidate = root + np.array([da, db, dc])
                    score = np.linalg.norm(
                        (candidate - _BASE_MANISKILL_STATE_EULER_MEAN) / scale
                    )
                    if score < best_score:
                        best = candidate
                        best_score = score
    return best.astype(np.float64)


def qpos8_to_robot_qpos(qpos8: np.ndarray) -> np.ndarray:
    """Convert stored 8D qpos to the Panda robot's 9D qpos."""
    qpos8 = np.asarray(qpos8, dtype=np.float32)
    grip = float(qpos8[7])
    return np.array([*qpos8[:7], grip, grip], dtype=np.float32)


def aligned_pi05_state_from_tcp_matrix(
    tcp_matrix: np.ndarray,
    gripper: np.ndarray | list[float] | tuple[float, float],
) -> np.ndarray:
    """Build the aligned 8D pi0.5 proprio state from a TCP transform matrix."""
    tcp_matrix = np.asarray(tcp_matrix, dtype=np.float64)
    pos = tcp_matrix[:3, 3] + _POS_OFFSET
    rot = _ROT_ALIGN @ tcp_matrix[:3, :3]
    euler = _canonicalize_state_euler(mat2euler(rot, "sxyz"))
    gripper = np.asarray(gripper, dtype=np.float64)
    if gripper.shape == ():
        gripper = np.repeat(gripper, 2)
    return np.concatenate([pos, euler, gripper[:2]]).astype(np.float32)


def aligned_pi05_state_from_tcp_matrices(
    tcp_matrices: np.ndarray,
    grippers: np.ndarray,
) -> np.ndarray:
    """Vectorized wrapper for aligned 8D pi0.5 proprio states."""
    return np.stack(
        [
            aligned_pi05_state_from_tcp_matrix(tcp_matrix, gripper)
            for tcp_matrix, gripper in zip(tcp_matrices, grippers, strict=True)
        ],
        axis=0,
    ).astype(np.float32)


def euler_xyz_delta_actions_from_tcp_matrices(
    tcp_matrices: np.ndarray,
    gripper_actions: np.ndarray,
) -> np.ndarray:
    """Generate target-delta Euler XYZ actions from TCP matrices.

    Rotation is Euler XYZ because ManiSkill ``PDEEPoseController`` parses the
    last three pose dimensions with ``euler_angles_to_matrix(..., "XYZ")``.
    """
    tcp_matrices = np.asarray(tcp_matrices, dtype=np.float64)
    gripper_actions = np.asarray(gripper_actions, dtype=np.float32)
    num_steps = tcp_matrices.shape[0]
    actions = np.zeros((num_steps, PI05_ACTION_DIM), dtype=np.float32)
    for t in range(num_steps - 1):
        dp = tcp_matrices[t + 1, :3, 3] - tcp_matrices[t, :3, 3]
        r_delta = tcp_matrices[t + 1, :3, :3] @ tcp_matrices[t, :3, :3].T
        dr = matrix_to_euler_angles(
            torch.as_tensor(r_delta, dtype=torch.float64).unsqueeze(0), "XYZ"
        )[0].numpy()
        gripper = (
            PANDA_GRIPPER_OPEN
            if float(gripper_actions[t]) > 0.0
            else PANDA_GRIPPER_CLOSE
        )
        actions[t] = np.concatenate([dp, dr, [gripper]])
    if num_steps > 1:
        actions[-1] = actions[-2]
    return actions


def describe_action_semantics() -> str:
    """Return a short human-readable action convention description."""
    return (
        "physical target-delta [dx, dy, dz, droll, dpitch, dyaw, gripper] "
        "with Euler XYZ rotation; Panda env rotation mapping uses -delta/0.1"
    )
