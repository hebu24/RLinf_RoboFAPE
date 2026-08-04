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

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.utils.nested_dict_process import clone_nested_to_cpu, copy_dict_tensor


def history_obs_from_step(
    obs: dict[str, Any] | None, infos: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Use the true terminal observation for auto-reset low-level steps."""
    if obs is None or not isinstance(infos, dict):
        return obs
    final_obs = infos.get("final_observation")
    reset_mask = infos.get("_final_observation")
    if final_obs is None or reset_mask is None:
        return obs
    merged = copy_dict_tensor(obs)
    reset_mask = (
        reset_mask.detach().cpu().numpy()
        if isinstance(reset_mask, torch.Tensor)
        else np.asarray(reset_mask)
    )
    done_mask = (
        reset_mask.any(axis=-1) if reset_mask.ndim > 1 else reset_mask.astype(bool)
    )
    if not done_mask.any():
        return merged
    for key, value in merged.items():
        if key not in final_obs:
            continue
        final_value = final_obs[key]
        if isinstance(value, torch.Tensor) and isinstance(final_value, torch.Tensor):
            dst_mask = torch.as_tensor(done_mask, device=value.device)
            src_mask = dst_mask.to(device=final_value.device)
            merged[key][dst_mask] = final_value[src_mask]
        elif isinstance(value, np.ndarray) and isinstance(final_value, np.ndarray):
            merged[key][done_mask] = final_value[done_mask]
    return merged


def history_success_from_step(
    infos: dict[str, Any] | None, num_envs: int
) -> list[bool]:
    """Resolve root/final success for one low-level vector-env step."""
    if not isinstance(infos, dict):
        raise ValueError(
            "Reward history requires per-step env infos to resolve success."
        )
    success = infos.get("success")
    if success is None:
        raise ValueError(
            "Reward history requires per-step root `success` in env infos."
        )
    success_values = (
        success.detach().cpu().bool().clone()
        if isinstance(success, torch.Tensor)
        else torch.as_tensor(np.asarray(success), dtype=torch.bool)
    )
    if success_values.numel() != num_envs:
        raise ValueError(
            f"Expected {num_envs} success values, got {success_values.numel()}."
        )
    final_info = infos.get("final_info")
    reset_mask = infos.get("_final_info")
    if (
        isinstance(final_info, dict)
        and reset_mask is not None
        and "success" in final_info
    ):
        final_success = final_info["success"]
        final_success_values = (
            final_success.detach().cpu().bool()
            if isinstance(final_success, torch.Tensor)
            else torch.as_tensor(np.asarray(final_success), dtype=torch.bool)
        )
        reset_mask = (
            reset_mask.detach().cpu().numpy()
            if isinstance(reset_mask, torch.Tensor)
            else np.asarray(reset_mask)
        )
        done_mask = (
            reset_mask.any(axis=-1) if reset_mask.ndim > 1 else reset_mask.astype(bool)
        )
        if done_mask.any():
            done_mask_t = torch.as_tensor(done_mask, dtype=torch.bool)
            success_values[done_mask_t] = final_success_values[done_mask_t]
    return success_values.tolist()


class HistoryManager:
    def __init__(self, reward_cfg: DictConfig, num_envs: int):
        self.num_envs = num_envs
        self.history_buffers = self.setup_history_buffers(reward_cfg)

        self.history_keys = sorted(
            {
                history_key
                for history_buffer in self.history_buffers
                for history_key in history_buffer["history_keys"]
            }
        )

        self.history_entries: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
        self.success_history_entries: list[list[bool]] = [[] for _ in range(num_envs)]

        self.history_counts = [0 for _ in range(num_envs)]

        # Per-env count of prepended pick-up frames (for assign_history_reward
        # to skip pick-up progress when scattering onto policy chunks). Reset in
        # clear_history alongside the buffer.
        self.pickup_counts = [0 for _ in range(num_envs)]

    def setup_history_buffers(self, reward_cfg: DictConfig) -> list[dict[str, Any]]:
        history_buffers = reward_cfg.get("model", {}).get("history_buffers", None)
        if history_buffers is None:
            raise ValueError(
                "HistoryManager requires 'history_buffers' in YAML under reward.model.history_buffers."
            )

        history_buffers = [
            self.setup_history_buffer(history_buffer_name, history_buffer_cfg)
            for history_buffer_name, history_buffer_cfg in history_buffers.items()
        ]
        self.validate_history_buffers(history_buffers)

        self.max_history_size = max(
            history_buffer["history_size"] for history_buffer in history_buffers
        )
        return history_buffers

    def setup_history_buffer(
        self, history_buffer_name: str, history_buffer_cfg: dict[str, Any]
    ) -> dict[str, Any]:
        history_size = history_buffer_cfg.get("history_size")
        if not history_size:
            logging.warning(
                f"Using empty history buffer {history_buffer_name} with a 0 history_size as it's not defined."
            )
            history_size = 0

        min_history_size = history_buffer_cfg.get("min_history_size", 0)

        input_interval = history_buffer_cfg.get("input_interval")
        if not input_interval:
            logging.warning(
                f"Using empty history buffer {history_buffer_name} with a history_size={history_size} as it's not defined."
            )
            input_interval = max(history_size, 1)

        history_keys = history_buffer_cfg.get("history_keys")
        if not history_keys:
            raise ValueError(
                f"History buffer '{history_buffer_cfg}' doesn't define 'history_keys'."
            )

        input_on_done = history_buffer_cfg.get("input_on_done", False)

        return {
            "name": history_buffer_name,
            "history_size": history_size,
            "min_history_size": min_history_size,
            "input_interval": input_interval,
            "history_keys": history_keys,
            "input_on_done": input_on_done,
        }

    def validate_history_buffers(self, history_buffers: list[dict[str, Any]]) -> None:
        history_names = [history_buffer["name"] for history_buffer in history_buffers]
        history_name_set = set(history_names)
        if len(history_names) != len(history_name_set):
            raise ValueError(
                "History buffer names must be unique for proper extraction."
            )

    def append_to_history_entries(
        self,
        observations: dict[str, Any] | None,
        step_success: list[bool] | torch.Tensor | None = None,
    ) -> None:
        if observations is None:
            return
        success_values: list[bool] | None = None
        if step_success is not None:
            if isinstance(step_success, torch.Tensor):
                success_values = step_success.detach().cpu().bool().tolist()
            else:
                success_values = [bool(v) for v in step_success]
            if len(success_values) != self.num_envs:
                raise ValueError(
                    "HistoryManager step_success must have one value per env: "
                    f"expected {self.num_envs}, got {len(success_values)}."
                )
        for env_id in range(self.num_envs):
            history_entry = {}
            for history_key in self.history_keys:
                history_values = observations.get(history_key, None)
                if history_values is None:
                    continue
                history_entry[history_key] = clone_nested_to_cpu(history_values[env_id])
            self.history_entries[env_id].append(history_entry)
            self.success_history_entries[env_id].append(
                False if success_values is None else bool(success_values[env_id])
            )
            self.history_counts[env_id] += 1

    def append_history_sequence(
        self,
        observations_list: list[dict[str, Any] | None],
        success_list: list[list[bool]] | list[torch.Tensor] | None = None,
    ) -> None:
        """Append a time-ordered sequence of per-step observations.

        Used by the peg-insertion Robometer reward path so the history buffer
        stores one entry per low-level env step instead of one entry per chunk.
        """
        if success_list is not None and len(success_list) != len(observations_list):
            raise ValueError(
                "HistoryManager success_list must align with observations_list: "
                f"{len(success_list)=} vs {len(observations_list)=}."
            )
        for step_idx, observations in enumerate(observations_list):
            step_success = None if success_list is None else success_list[step_idx]
            self.append_to_history_entries(observations, step_success=step_success)

    def prepend_history_entries(self, env_id: int, frames: list) -> None:
        """Prepend pick-up render frames to the FRONT of the history buffer.

        Called by env_worker (initial reset + auto-reset) so the robometer sees
        the full pick-up+insert video. Pick-up frames never enter RL training
        data (they are not rollout steps); ``pickup_counts[env_id]`` records the
        count so ``assign_history_reward`` can skip pick-up progress when
        scattering the curve onto policy chunks.
        """
        if not frames:
            return
        # The final planner frame is the lift-end reset state and is duplicated by
        # the first insertion observation. Keep this normalization here so RL and
        # smoke use the exact same pick-up prefix.
        pickup_frames = list(frames[:-1]) if len(frames) > 1 else list(frames)
        pickup_entries = [
            {"render_images": clone_nested_to_cpu(frame)} for frame in pickup_frames
        ]
        # Preserve the planner's chronological approach -> grasp -> lift order.
        self.history_entries[env_id][0:0] = pickup_entries
        self.success_history_entries[env_id][0:0] = [False] * len(pickup_entries)
        self.history_counts[env_id] += len(pickup_entries)
        self.pickup_counts[env_id] = len(pickup_entries)

    def build_history_input(
        self,
        dones: torch.Tensor,
        emit_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, Any], dict[str, list[int]]]:
        history_input: dict[str, dict[str, list[list]]] = {}
        history_length: dict[str, list[int]] = {}

        def append_to_history_input(
            history_buffer, history_range, env_idx: int
        ) -> None:
            history_buffer_name = history_buffer["name"]

            if history_buffer_name not in history_length:
                history_length[history_buffer_name] = [0 for _ in range(self.num_envs)]
            input_history_entries = self.history_entries[env_idx][history_range]
            history_length[history_buffer_name][env_idx] += len(input_history_entries)

            if history_buffer_name not in history_input:
                history_input[history_buffer_name] = {}
            for history_key in history_buffer["history_keys"]:
                if history_key not in history_input[history_buffer_name]:
                    history_input[history_buffer_name][history_key] = [
                        [] for _ in range(self.num_envs)
                    ]
                history_input[history_buffer_name][history_key][env_idx].extend(
                    [
                        entry[history_key]
                        for entry in input_history_entries
                        if history_key in entry
                    ]
                )

        if (dones.shape[0] != self.num_envs) or (dones.ndim != 1):
            raise ValueError(
                f"Expect the dones to have a shape of (self.num_envs,) = ({self.num_envs},), got {dones.shape}"
            )

        if emit_mask is None:
            emit_mask = torch.ones_like(dones, dtype=torch.bool)
        if emit_mask.shape != dones.shape:
            raise ValueError(
                "HistoryManager emit_mask must match dones: "
                f"{emit_mask.shape=} vs {dones.shape=}."
            )
        emit_mask = emit_mask.to(dtype=torch.bool, device=dones.device)

        for env_idx, done in enumerate(dones):
            for history_buffer in self.history_buffers:
                if not bool(emit_mask[env_idx]):
                    continue
                if (
                    len(self.history_entries[env_idx])
                    < history_buffer["min_history_size"]
                ):
                    history_range = slice(0, 0)
                elif (
                    self.history_counts[env_idx] % history_buffer["input_interval"] == 0
                ):
                    history_range = slice(
                        max(
                            0,
                            len(self.history_entries[env_idx])
                            - history_buffer["history_size"],
                        ),
                        len(self.history_entries[env_idx]),
                    )
                elif done and history_buffer["input_on_done"]:
                    history_range = slice(
                        max(
                            0,
                            len(self.history_entries[env_idx])
                            - self.history_counts[env_idx]
                            % history_buffer["input_interval"],
                        ),
                        len(self.history_entries[env_idx]),
                    )
                else:
                    continue
                append_to_history_input(history_buffer, history_range, env_idx)

            if done:
                self.clear_history(env_idx)
            else:
                self.trim_history(env_idx)

        return history_input, history_length

    def clear_history(self, env_id: int) -> None:
        self.history_entries[env_id].clear()
        self.success_history_entries[env_id].clear()
        self.history_counts[env_id] = 0
        self.pickup_counts[env_id] = 0

    def reset_all(self) -> None:
        """Clear all per-environment histories before a new independent rollout window.

        Mirrors ``clear_history`` for every env (history_entries,
        success_history_entries, history_counts, pickup_counts) so the next
        window's Robometer video never spans the previous window. Does not
        rebuild the HistoryManager; pickup prefixes are re-established by the
        env_worker via ``consume_pickup_frames`` + ``prepend_history_entries``.
        """
        for env_id in range(self.num_envs):
            self.clear_history(env_id)

    def trim_history(self, env_idx: int) -> None:
        cur_len = len(self.history_entries[env_idx])
        if cur_len <= self.max_history_size:
            return
        dropped = cur_len - self.max_history_size
        self.history_entries[env_idx] = self.history_entries[env_idx][
            -self.max_history_size :
        ]
        self.success_history_entries[env_idx] = self.success_history_entries[env_idx][
            -self.max_history_size :
        ]
        # Preprended pick-up frames live at the front of the buffer, so when we
        # trim from the front we may discard some or all of that prefix. Keep
        # pickup_counts aligned with the *current* retained history window.
        self.pickup_counts[env_idx] = max(0, self.pickup_counts[env_idx] - dropped)
