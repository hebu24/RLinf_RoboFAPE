# Copyright 2025 The RLinf Authors.
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

import numpy as np
import torch

from rlinf.config import SupportedModel
from rlinf.envs import SupportedEnvType


class TemporalEnsembleBuffer:
    """ACT-style time-weighted temporal ensemble of overlapping action chunks.

    Used for eval real-time chunking: the model predicts a full horizon-H chunk
    every k env steps (k <= H), and at each executed step we blend the
    ``age``-th action of every still-overlapping buffered prediction, weighting
    each by ``exp(-m * age)``. This closes the within-chunk open loop (each
    executed action is conditioned on a recent observation) and smooths the
    chunk-boundary discontinuity, while blending before the (linear) panda
    action conversion so the average is exact.

    Per-env reset masking drops predictions made before an env's last reset, so
    a freshly-reset env does not blend in stale pre-reset predictions.
    """

    def __init__(self, horizon: int, m: float, num_envs: int, device):
        self.horizon = int(horizon)
        self.m = float(m)
        self.num_envs = int(num_envs)
        self.device = device
        # parallel lists; chunks[i] is [num_envs, horizon, action_dim]
        self.chunks: list[torch.Tensor] = []
        self.predict_steps: list[int] = []
        # per-env global step of the most recent reset (inclusive); preds with
        # predict_step < reset_step[e] are ignored for env e.
        self.reset_step = torch.zeros(num_envs, dtype=torch.long, device=device)

    def push(self, chunk, predict_step: int) -> None:
        """Buffer a freshly predicted chunk [num_envs, horizon, action_dim].

        Accepts torch.Tensor or numpy.ndarray (the rollout worker ships model
        outputs as numpy); converted to a tensor on the buffer's device.
        """
        if not torch.is_tensor(chunk):
            chunk = torch.as_tensor(chunk, dtype=torch.float32)
        chunk = chunk.detach().float()
        if chunk.device != self.device:
            chunk = chunk.to(self.device)
        self.chunks.append(chunk)
        self.predict_steps.append(int(predict_step))
        # drop predictions that can no longer overlap any future step.
        # A prediction made at step p overlaps up to step p + horizon - 1.
        if len(self.predict_steps) > 1:
            oldest_needed = predict_step - self.horizon + 1
            while self.predict_steps and self.predict_steps[0] < oldest_needed:
                self.chunks.pop(0)
                self.predict_steps.pop(0)

    def mark_reset(self, env_mask: torch.Tensor, step: int) -> None:
        """Record that envs in ``env_mask`` reset at ``step`` (inclusive).

        Predictions made before ``step`` are ignored for those envs afterward.
        """
        if env_mask is None or not bool(env_mask.any()):
            return
        mask = env_mask.to(self.device).bool()
        self.reset_step[mask] = int(step)

    def current_action(self, current_step: int) -> torch.Tensor:
        """Return the blended action [num_envs, action_dim] for ``current_step``.

        Blends the ``age``-th action of each overlapping, non-reset prediction,
        weighted by ``exp(-m * age)``. With m == 0 returns the newest valid
        prediction's action (pure slicing).
        """
        if not self.chunks:
            raise RuntimeError("TemporalEnsembleBuffer.current_action with empty buffer")
        ages = [current_step - ps for ps in self.predict_steps]
        # keep overlapping predictions (age in [0, horizon))
        keep = [i for i, a in enumerate(ages) if 0 <= a < self.horizon]
        if not keep:
            raise RuntimeError(
                "TemporalEnsembleBuffer: no overlapping prediction for current_step "
                f"{current_step}; predict_steps={self.predict_steps}"
            )
        ages = torch.tensor(
            [ages[i] for i in keep], dtype=torch.float32, device=self.device
        )  # [J]
        # each prediction's age-th action along the horizon axis: [J, num_envs, action_dim]
        acts = torch.stack(
            [self.chunks[keep[j]][:, ages[j].long(), :] for j in range(len(keep))],
            dim=0,
        )  # [J, num_envs, action_dim]

        # per-env validity mask: prediction i valid for env e iff
        # predict_step_i >= reset_step[e]
        ps = torch.tensor(
            [self.predict_steps[i] for i in keep],
            dtype=torch.long,
            device=self.device,
        )  # [J]
        valid = ps[:, None] >= self.reset_step[None, :]  # [J, num_envs]

        if self.m > 0:
            weights = torch.exp(-self.m * ages)  # [J]
        else:
            weights = torch.ones(len(keep), device=self.device)  # uniform; argmax newest below
        weights = weights[:, None] * valid.float()  # [J, num_envs]

        if self.m == 0:
            # pure slicing: take the newest valid prediction per env.
            # keep[] is ordered oldest->newest; pick the last valid per env.
            order = torch.arange(len(keep), device=self.device)  # [J]
            valid_order = order[:, None] * valid  # [J, num_envs], 0 for invalid
            newest_idx = valid_order.argmax(dim=0)  # [num_envs]
            env_idx = torch.arange(self.num_envs, device=self.device)
            return acts[newest_idx, env_idx]

        denom = weights.sum(dim=0)  # [num_envs]
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        blended = (weights.unsqueeze(-1) * acts).sum(dim=0) / denom.unsqueeze(-1)
        return blended  # [num_envs, action_dim]



def prepare_actions_for_maniskill(
    raw_chunk_actions,
    num_action_chunks,
    action_dim,
    action_scale,
    policy,
) -> torch.Tensor:
    # TODO only suitable for action_dim = 7
    policy = policy or "widowx_bridge"
    reshaped_actions = raw_chunk_actions.reshape(-1, action_dim)
    if "panda" in policy:
        chunk_actions = reshaped_actions.copy()
        if policy in [
            "panda-ee-dpose",
            "panda-ee-target-dpos",
            "panda-ee-target-dpose",
        ]:
            from rlinf.envs.maniskill.peg_insertion_pi05 import (
                model_action_to_panda_env_action,
            )

            # The policy predicts raw controller-space deltas:
            # [meters, meters, meters, radians, radians, radians, gripper].
            # ManiSkill's Panda EE pose controllers have normalize_action=True.
            # Position uses delta / 0.1; rotation uses -delta / 0.1 because
            # PDEEPoseController multiplies normalized rotation by rot_lower.
            chunk_actions = model_action_to_panda_env_action(
                chunk_actions, action_scale=action_scale
            )
        else:
            chunk_actions[:, :6] *= action_scale
        return torch.tensor(chunk_actions, dtype=torch.float32).cuda().reshape(
            -1, num_action_chunks, action_dim
        )
    batch_size = reshaped_actions.shape[0]
    raw_actions = {
        "world_vector": np.array(reshaped_actions[:, :3]),
        "rotation_delta": np.array(reshaped_actions[:, 3:6]),
        "open_gripper": np.array(
            reshaped_actions[:, 6:7]
        ),  # range [0, 1]; 1 = open; 0 = close
    }

    # process raw_action to obtain the action to be sent to the maniskill2 environment
    actions = {}
    actions["world_vector"] = raw_actions["world_vector"] * action_scale  # [B, 3]
    actions["rot_axangle"] = raw_actions["rotation_delta"] * action_scale  # [B, 3]

    if policy == "google_robot":
        raise NotImplementedError
    elif policy == "widowx_bridge":
        actions["gripper"] = 2.0 * (raw_actions["open_gripper"] > 0.5) - 1.0  # [B, 1]
    elif policy == "panda_wristcam":
        actions["gripper"] = 2.0 * (raw_actions["open_gripper"] > 0.5) - 1.0  # [B, 1]

    actions["terminate_episode"] = np.array([0.0] * batch_size).reshape(-1, 1)  # [B, 1]

    actions = {k: torch.tensor(v, dtype=torch.float32) for k, v in actions.items()}
    actions = torch.cat(
        [actions["world_vector"], actions["rot_axangle"], actions["gripper"]], dim=1
    ).cuda()

    chunk_actions = actions.reshape(-1, num_action_chunks, action_dim)

    return chunk_actions


def prepare_actions_for_libero(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
        SupportedModel.GR00T_N1D6,
        SupportedModel.GR00T_N1D7,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = np.sign(chunk_actions[..., -1]) * -1.0
    return chunk_actions


def prepare_actions_for_isaaclab(
    raw_chunk_actions,
    model_type,
) -> torch.Tensor:
    """
    Here reture a general 7 dof action. If the action is modified, please change the output of the model
    For example, in `RLinf/rlinf/models/embodiment/gr00t/simulation_io.py`
    """
    chunk_actions = (
        torch.from_numpy(raw_chunk_actions)
        if isinstance(raw_chunk_actions, np.ndarray)
        else raw_chunk_actions
    )
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = torch.sign(chunk_actions[..., -1]) * -1.0
    return chunk_actions


def prepare_actions_for_polaris(
    raw_chunk_actions,
    model_type,
) -> torch.Tensor:
    """
    Here reture a general 7 dof action. If the action is modified, please change the output of the model
    For example, in `RLinf/rlinf/models/embodiment/gr00t/simulation_io.py`
    """
    chunk_actions = (
        torch.from_numpy(raw_chunk_actions)
        if isinstance(raw_chunk_actions, np.ndarray)
        else raw_chunk_actions
    )
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = torch.sign(chunk_actions[..., -1]) * -1.0
    elif SupportedModel(model_type) == SupportedModel.OPENPI:
        chunk_actions[..., -1] = torch.where(
            chunk_actions[..., -1] > 0.5,
            torch.ones_like(chunk_actions[..., -1]),
            torch.zeros_like(chunk_actions[..., -1]),
        )
    return chunk_actions


def prepare_actions_for_calvin(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) == SupportedModel.OPENPI:
        chunk_actions[..., -1] = np.sign(chunk_actions[..., -1])
    else:
        chunk_actions[..., -1] = np.where(chunk_actions[..., -1] > 0, 1, -1)
    return chunk_actions


def prepare_actions_for_metaworld(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        # the action dimesion of metaworld is 4-dim (x, y, z, gripper)
        # we need to extract the first 3-dim and the last dim in a 7-dim action
        if chunk_actions.shape[-1] == 7:
            chunk_actions = np.concatenate(
                [chunk_actions[..., :3], chunk_actions[..., -1:]], axis=-1
            )
    return chunk_actions


def prepare_actions_for_robocasa(
    raw_chunk_actions,
    action_dim,
    action_space,
) -> np.ndarray:
    """
    Prepare actions for robocasa environment.
    Model outputs 32D actions per chunk, and model got first N valid actions defined by action_space, but robocasa expects 12D.
    So extract the first N dimensions, fit to corresponding ids, and pad the rest to get12 dimensions (3D pos + 3D ori + 1D gripper + 5D base).
    """

    # raw_chunk_actions shape: [num_chunks, 32]
    # Extract first action_dim (<=12) dimensions as valid action chunks
    # Then pad them to default actions to get (..., 12)-shaped action chunks for RobocasaEnv.step()
    from rlinf.envs.robocasa.utils import (
        ROBOCASA_ALL_ACTION_DIM,
        ROBOCASA_DEFAULT_ACTION,
        get_action_ids,
        get_action_space,
    )

    assert action_dim <= ROBOCASA_ALL_ACTION_DIM, (
        f"Requested action_dim ({action_dim}) exceeds max dimension ({ROBOCASA_ALL_ACTION_DIM})."
    )

    valid_chunk_actions = raw_chunk_actions[..., :action_dim]

    chunk_actions = np.full(
        shape=valid_chunk_actions.shape[:-1] + (ROBOCASA_ALL_ACTION_DIM,),
        fill_value=ROBOCASA_DEFAULT_ACTION,
        dtype=valid_chunk_actions.dtype,
    )

    all_action_ids = get_action_ids(get_action_space(action_space))
    assert len(all_action_ids) == action_dim, (
        f"Mismatch between action_space ids length ({len(all_action_ids)}) and provided action_dim ({action_dim})."
    )
    chunk_actions[..., all_action_ids] = valid_chunk_actions

    return chunk_actions


def prepare_actions_for_genesis(
    raw_chunk_actions,
    model_type,
) -> torch.Tensor:
    """Prepare actions for the Genesis environment.

    For VLA models (OpenVLA / OpenVLA-OFT), transforms the gripper
    dimension from a [0, 1] continuous value to a {-1, +1} binary signal
    (matching the convention used by other embodied envs).

    For all other models the actions are returned as-is, converted to a
    torch tensor on CUDA.
    """
    if isinstance(raw_chunk_actions, np.ndarray):
        chunk_actions = torch.from_numpy(raw_chunk_actions).float()
    else:
        chunk_actions = raw_chunk_actions.clone().float()
    if SupportedModel(model_type) in [
        SupportedModel.OPENVLA,
        SupportedModel.OPENVLA_OFT,
    ]:
        chunk_actions[..., -1] = 2 * chunk_actions[..., -1] - 1
        chunk_actions[..., -1] = torch.sign(chunk_actions[..., -1]) * -1.0
    return chunk_actions


def prepare_actions_for_mujoco(raw_chunk_actions, model_type):
    if raw_chunk_actions.shape[-1] >= 7:
        chunk_actions = np.concatenate(
            [raw_chunk_actions[..., :3], raw_chunk_actions[..., 6:7]], axis=-1
        )
    else:
        chunk_actions = raw_chunk_actions[..., :4]
    if SupportedModel(model_type) == SupportedModel.OPENPI:
        chunk_actions[..., -1] = np.clip(chunk_actions[..., -1], -1.0, 1.0)
    return chunk_actions


def prepare_actions_for_d4rl(
    raw_chunk_actions,
    action_dim: int,
    model_type,
) -> np.ndarray:
    # D4RL: take first action_dim dims from policy output
    raw = np.asarray(raw_chunk_actions, dtype=np.float32)
    chunk_actions = raw[..., :action_dim].copy()
    # OPENPI: clip last dim to match continuous action space
    if SupportedModel(model_type) == SupportedModel.OPENPI:
        chunk_actions[..., -1] = np.clip(chunk_actions[..., -1], -1.0, 1.0)
    return chunk_actions


def prepare_actions_for_roboverse(
    raw_chunk_actions,
    model_type,
) -> np.ndarray:
    chunk_actions = raw_chunk_actions
    if SupportedModel(model_type) == SupportedModel.OPENPI:
        chunk_actions[..., -1] = np.where(chunk_actions[..., -1] < 0.0, 1.0, 0.0)
    return chunk_actions


def prepare_actions(
    raw_chunk_actions,
    env_type: str,
    model_type: str,
    num_action_chunks,
    action_dim,
    action_scale: float = 1.0,
    policy: str = "widowx_bridge",
    wm_env_type=None,
) -> torch.Tensor | np.ndarray:
    if isinstance(raw_chunk_actions, torch.Tensor):
        raw_chunk_actions = raw_chunk_actions.detach().cpu().contiguous()
        if raw_chunk_actions.dtype == torch.bfloat16:
            raw_chunk_actions = raw_chunk_actions.float()
        raw_chunk_actions = raw_chunk_actions.numpy()

    env_type = SupportedEnvType(env_type)
    if env_type == SupportedEnvType.LIBERO:
        chunk_actions = prepare_actions_for_libero(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.OPENSORAWM or env_type == SupportedEnvType.WANWM:
        # TODO: Implement prepare_actions_for_opensora_wm
        if wm_env_type == "libero":
            chunk_actions = prepare_actions_for_libero(
                raw_chunk_actions=raw_chunk_actions,
                model_type=model_type,
            )
        else:
            raise NotImplementedError(f"Env type {wm_env_type} not implemented")
    elif env_type == SupportedEnvType.MANISKILL:
        chunk_actions = prepare_actions_for_maniskill(
            raw_chunk_actions=raw_chunk_actions,
            num_action_chunks=num_action_chunks,
            action_dim=action_dim,
            action_scale=action_scale,
            policy=policy,
        )
    elif env_type == SupportedEnvType.ROBOTWIN:
        chunk_actions = raw_chunk_actions
    elif env_type == SupportedEnvType.EMBODICHAIN:
        chunk_actions = raw_chunk_actions
    elif env_type == SupportedEnvType.METAWORLD:
        chunk_actions = prepare_actions_for_metaworld(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.CALVIN:
        chunk_actions = prepare_actions_for_calvin(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.BEHAVIOR:
        chunk_actions = raw_chunk_actions
    elif env_type == SupportedEnvType.ISAACLAB:
        chunk_actions = prepare_actions_for_isaaclab(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.ROBOCASA:
        chunk_actions = prepare_actions_for_robocasa(
            raw_chunk_actions=raw_chunk_actions,
            action_dim=action_dim,
            action_space=policy,
        )
    elif env_type == SupportedEnvType.REALWORLD:
        chunk_actions = raw_chunk_actions
    elif env_type == SupportedEnvType.GENESIS:
        chunk_actions = prepare_actions_for_genesis(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.FRANKASIM:
        chunk_actions = prepare_actions_for_mujoco(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.D4RL:
        chunk_actions = prepare_actions_for_d4rl(
            raw_chunk_actions=raw_chunk_actions,
            action_dim=action_dim,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.ROBOVERSE:
        chunk_actions = prepare_actions_for_roboverse(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    elif env_type == SupportedEnvType.POLARIS:
        chunk_actions = prepare_actions_for_polaris(
            raw_chunk_actions=raw_chunk_actions,
            model_type=model_type,
        )
    else:
        chunk_actions = raw_chunk_actions

    return chunk_actions
