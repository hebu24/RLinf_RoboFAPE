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

from collections import OrderedDict
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import json
from pathlib import Path
import copy
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import put_info_on_image, tile_images
from mani_envs.data_collection.exact_state import compare_states, env_slice, state_hash
from omegaconf import open_dict
from omegaconf.omegaconf import OmegaConf

from rlinf.utils.logging import get_logger

__all__ = ["ManiskillEnv"]

_logger = get_logger()


def _arm_controller(env):
    """Return the arm sub-controller of a ManiSkill agent's combined controller."""
    controller = env.unwrapped.agent.controller
    if hasattr(controller, "controllers"):
        return controller.controllers.get("arm", controller)
    return controller


def _sync_target_delta_pose_controller(env) -> None:
    """Re-anchor the target-delta controller's ``_target_pose`` to the actual EE pose.

    With ``use_target=True`` (``pd_ee_target_delta_pose``), the controller open-loop
    integrates each action delta onto ``_target_pose`` (the previous *commanded* target)
    rather than the actual end-effector pose. Over a long eval episode this lets IK/PD
    tracking error and model error compound on the target and drift away from the real
    EE, which the policy never observes. Re-anchoring the target to the actual EE once
    per action chunk closes that loop while preserving within-chunk target chaining.

    No-op for controllers without ``use_target`` / ``_target_pose``.
    """
    arm_controller = _arm_controller(env)
    config = getattr(arm_controller, "config", None)
    if not bool(getattr(config, "use_target", False)) or not hasattr(
        arm_controller, "_target_pose"
    ):
        return
    prev_target = arm_controller._target_pose
    actual = arm_controller.ee_pose_at_base
    if prev_target is not None:
        drift = (prev_target.p - actual.p).norm(dim=-1).max().item()
        if drift > 1e-3:
            _logger.debug(
                "pd_ee_target_delta_pose: re-anchoring _target_pose to actual EE "
                "(max pos drift = %.5f m)",
                drift,
            )
    arm_controller._target_pose = actual


def extract_termination_from_info(info, num_envs, device):
    if "success" in info:
        if "fail" in info:
            terminated = torch.logical_or(info["success"], info["fail"])
        else:
            terminated = info["success"].clone()
    else:
        if "fail" in info:
            terminated = info["fail"].clone()
        else:
            terminated = torch.zeros(num_envs, dtype=bool, device=device)
    return terminated


class ManiskillEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        env_seed = cfg.seed
        self.shared_reset_seed = bool(getattr(cfg, "shared_reset_seed", False))
        # Training normally decorrelates every vector env with consecutive seeds.
        # For deterministic SFT-baseline runs, deliberately keep every worker on
        # the exact same reset seed instead.
        self.seed_offset = int(seed_offset)
        self.seed = env_seed if self.shared_reset_seed else env_seed + self.seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self.use_full_state = bool(getattr(cfg, "use_full_state", False))
        self.record_kinematic_trace = bool(
            getattr(cfg, "record_kinematic_trace", False)
        )
        # Re-anchor the pd_ee_target_delta_pose controller's _target_pose to the actual
        # EE pose once per action chunk (see _sync_target_delta_pose_controller). This
        # prevents the open-loop target from drifting away from the real EE over a long
        # eval episode. Toggle off to reproduce the legacy drifting behavior.
        self.sync_target_pose_per_chunk = bool(
            getattr(cfg, "sync_target_pose_per_chunk", True)
        )
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.initial_state_manifest = getattr(cfg, "initial_state_manifest", None)
        self._initial_state_records = []
        if self.initial_state_manifest:
            manifest = json.loads(Path(self.initial_state_manifest).read_text())
            self._initial_state_records = manifest.get("references", [])

        self.video_cfg = cfg.video_cfg

        self.cfg = cfg

        with open_dict(cfg):
            cfg.init_params.num_envs = num_envs
        env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        for camera_key in ("human_render_camera_configs",):
            camera_cfg = env_args.get(camera_key)
            if isinstance(camera_cfg, dict) and "pose" in camera_cfg:
                camera_cfg["pose"] = np.asarray(camera_cfg["pose"], dtype=np.float32)
        self.env: BaseEnv = gym.make(**env_args)
        render_randomization_spec = getattr(cfg, "render_randomization_spec", None)
        if render_randomization_spec:
            try:
                from render_domain_randomization import apply_render_randomization
            except ImportError as exc:
                raise RuntimeError(
                    "render_randomization_spec requires RoboFPE render helpers on PYTHONPATH"
                ) from exc
            apply_render_randomization(self.env, render_randomization_spec)
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )  # [B, ]
        self.record_metrics = record_metrics
        self._is_start = True
        self._init_reset_state_ids()
        self.info_logging_keys = ["is_src_obj_grasped", "consecutive_grasp", "success"]
        self._show_goal_site_visual()
        if self.record_metrics:
            self._init_metrics()

    @property
    def total_num_group_envs(self):
        if hasattr(self.env.unwrapped, "total_num_trials"):
            return self.env.unwrapped.total_num_trials
        if hasattr(self.env, "xyz_configs") and hasattr(self.env, "quat_configs"):
            return len(self.env.xyz_configs) * len(self.env.quat_configs)
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def num_envs(self):
        return self.env.unwrapped.num_envs

    @property
    def reset_seed(self):
        if self.shared_reset_seed:
            return self.seed
        base_seed = int(getattr(self.cfg, "seed", 0))
        start = base_seed + self.seed_offset * int(self.num_envs)
        return list(range(start, start + int(self.num_envs)))

    @property
    def device(self):
        return self.env.unwrapped.device

    @property
    def elapsed_steps(self):
        return self.env.unwrapped.elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def instruction(self):
        instruction_getter = getattr(
            self.env.unwrapped, "get_language_instruction", None
        )
        if callable(instruction_getter):
            return instruction_getter()
        task_description = getattr(self.cfg, "task_description", None)
        if task_description is not None:
            return [str(task_description)] * self.num_envs
        return [""] * self.num_envs

    def _base_reset_options(self):
        reset_options = getattr(self.cfg, "reset_options", None)
        if reset_options is None:
            return {}
        if OmegaConf.is_config(reset_options):
            return OmegaConf.to_container(reset_options, resolve=True) or {}
        return dict(reset_options)

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def set_eval_seed(self, seed: int):
        seed = int(seed)
        self.cfg.seed = seed
        self.seed = seed if self.shared_reset_seed else seed + self.seed_offset
        self._is_start = True
        self._init_reset_state_ids()

    def update_reset_state_ids(self):
        if self._initial_state_records:
            # Exact-pair rollouts must select a manifest state deterministically
            # from the logical evaluation seed, never from ManiSkill's random
            # trial table. This makes state identity reproducible and auditable.
            ref_idx = int(self.seed) % len(self._initial_state_records)
            reset_state_ids = torch.full(
                (self.num_group,), ref_idx, dtype=torch.long
            )
            self.reset_state_ids = reset_state_ids.repeat_interleave(
                repeats=self.group_size
            ).to(self.device)
            return
        if self.shared_reset_seed and hasattr(self, "reset_state_ids"):
            return
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(1 if self.shared_reset_seed else self.num_group,),
            generator=self._generator,
        )
        if self.shared_reset_seed:
            reset_state_ids = reset_state_ids.repeat(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    def _show_goal_site_visual(self):
        """Keep ManiSkill goal-site visualization visible for reward-model RGB input."""
        if not hasattr(self.env.unwrapped, "goal_site"):
            return

        goal_site = self.env.unwrapped.goal_site
        if hasattr(self.env.unwrapped, "_hidden_objects"):
            while goal_site in self.env.unwrapped._hidden_objects:
                self.env.unwrapped._hidden_objects.remove(goal_site)
        if hasattr(goal_site, "show_visual"):
            goal_site.show_visual()

    def _wrap_obs(self, raw_obs, infos=None):
        wrap_obs_mode = getattr(self.cfg, "wrap_obs_mode", "default")
        if wrap_obs_mode == "raw":
            assert infos is not None
            return infos["extracted_obs"]

        if wrap_obs_mode == "simple":
            if self.env.unwrapped.obs_mode == "state":
                return {"states": raw_obs}
            elif self.env.unwrapped.obs_mode == "rgb":
                sensor_data = raw_obs.pop("sensor_data")
                raw_obs.pop("sensor_param")
                if hasattr(self.env.unwrapped, "get_pi05_proprio"):
                    state = self.env.unwrapped.get_pi05_proprio()
                elif bool(getattr(self.cfg, "use_pi05_proprio", False)):
                    from rlinf.envs.maniskill.peg_insertion_pi05 import (
                        aligned_pi05_state_from_tcp_matrices,
                    )

                    tcp_pose_in_root = (
                        self.env.unwrapped.agent.robot.pose.inv()
                        * self.env.unwrapped.agent.tcp.pose
                    )
                    tcp_transform = (
                        tcp_pose_in_root.to_transformation_matrix()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    gripper = (
                        self.env.unwrapped.agent.robot.get_qpos()
                        .to(torch.float32)[:, -2:]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    state = torch.as_tensor(
                        aligned_pi05_state_from_tcp_matrices(
                            tcp_transform, gripper
                        ),
                        device=self.device,
                        dtype=torch.float32,
                    )
                elif self.use_full_state:
                    state = self._get_full_state_obs()
                else:
                    # Default proprio = robot qpos (matches collectors that store
                    # agent/qpos as observation.state). flatten_state_dict produced a
                    # mismatched dim for tasks lacking get_pi05_proprio (e.g. PushCube-v1
                    # gave 25-dim flatten vs 9-dim qpos/norm_stats).
                    _robot = getattr(getattr(self.env.unwrapped, "agent", None), "robot", None)
                    if _robot is not None:
                        state = _robot.get_qpos().to(self.device, dtype=torch.float32)
                    else:
                        state = common.flatten_state_dict(
                            raw_obs, use_torch=True, device=self.device
                        )

                main_camera_key = getattr(self.cfg, "main_camera_key", "base_camera")
                if main_camera_key == "render_camera":
                    try:
                        main_images = self.env.unwrapped.render_rgb_array(
                            camera_name=main_camera_key
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "Configured main_camera_key=render_camera, but the "
                            "environment could not render render_camera"
                        ) from exc
                    if not torch.is_tensor(main_images):
                        main_images = torch.as_tensor(
                            main_images, device=self.device
                        )
                    if main_images.ndim == 3:
                        main_images = main_images.unsqueeze(0)
                    main_images = main_images.to(torch.uint8)
                else:
                    main_images = sensor_data[main_camera_key]["rgb"]
                sorted_images = OrderedDict(sorted(sensor_data.items()))
                sorted_images.pop("base_camera", None)
                wrist_images = None
                if bool(getattr(self.cfg, "use_wrist_image", False)) and (
                    "hand_camera" in sorted_images
                ):
                    wrist_images = sorted_images.pop("hand_camera")["rgb"]
                else:
                    sorted_images.pop("hand_camera", None)
                # Back-facing wrist camera (eye-in-hand). Routed to its own named
                # channel (not the catch-all extra_view_images) so it maps cleanly
                # onto the model's right_wrist_0_rgb slot. Enabled by
                # cfg.use_wrist_back_image; absent sensors / disabled flag -> None.
                wrist_back_images = None
                if bool(getattr(self.cfg, "use_wrist_back_image", False)) and (
                    "hand_camera_back" in sorted_images
                ):
                    wrist_back_images = sorted_images.pop("hand_camera_back")["rgb"]
                else:
                    sorted_images.pop("hand_camera_back", None)
                # Human render camera (third-person external view, 640x480) for the
                # robometer progress-reward server. Gated by
                # cfg.capture_render_image_for_reward so non-RL runs are unaffected.
                # Popped from sorted_images so it is NOT bundled into
                # extra_view_images (the policy ignores extra_view anyway for
                # num_images_in_input==2). The reward path reads render_images from
                # env_output.obs/final_obs; the policy path drops it via
                # embodied_io_struct.prepare_observations (which only packs
                # main/wrist/extra/states/task).
                render_images = None
                if bool(getattr(self.cfg, "capture_render_image_for_reward", False)):
                    render_cam_key = getattr(
                        self.cfg, "render_camera_key", "reward_camera"
                    )
                    if render_cam_key in sorted_images:
                        render_images = sorted_images.pop(render_cam_key)["rgb"]
                    elif render_cam_key in sensor_data:
                        render_images = sensor_data[render_cam_key]["rgb"]
                extra_view_images = (
                    torch.stack([v["rgb"] for v in sorted_images.values()], dim=1)
                    if sorted_images
                    else None
                )
                return {
                    "main_images": main_images,
                    "extra_view_images": extra_view_images,
                    "wrist_images": wrist_images,
                    "wrist_back_images": wrist_back_images,
                    "render_images": render_images,
                    "states": state,
                    "task_descriptions": self.instruction,
                }

        # Default
        obs_image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"].to(
            torch.uint8
        )  # [B, H, W, C]
        proprioception: torch.Tensor = self.env.unwrapped.agent.robot.get_qpos().to(
            obs_image.device, dtype=torch.float32
        )
        return {
            "main_images": obs_image,
            "states": proprioception,
            "task_descriptions": self.instruction,
        }

    def _get_full_state_obs(self):
        base_env = self.env.unwrapped
        mode_attr = "_obs_mode" if hasattr(base_env, "_obs_mode") else "obs_mode"
        original_mode = getattr(base_env, mode_attr)
        setattr(base_env, mode_attr, "state")
        try:
            state_obs = base_env.get_obs()
        finally:
            setattr(base_env, mode_attr, original_mode)

        if isinstance(state_obs, dict):
            return common.flatten_state_dict(
                state_obs, use_torch=True, device=self.device
            )
        return state_obs

    def _calc_step_reward(self, reward, info):
        if getattr(self.cfg, "reward_mode", "default") == "raw":
            pass
        elif getattr(self.cfg, "reward_mode", "default") == "only_success":
            reward = info["success"] * 1.0
        else:
            reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
                self.env.unwrapped.device
            )  # [B, ]
            reward += info["is_src_obj_grasped"] * 0.1
            reward += info["consecutive_grasp"] * 0.1
            reward += (info["success"] & info["is_src_obj_grasped"]) * 1.0
        # diff
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.max_rewards = torch.full(
            (self.num_envs,), -float("inf"), device=self.device, dtype=torch.float32
        )
        if self.record_kinematic_trace:
            trace_shape = (
                self.num_envs,
                int(getattr(self.cfg, "max_episode_steps", 0)),
                3,
            )
            if trace_shape[1] <= 0:
                raise ValueError("record_kinematic_trace requires max_episode_steps > 0")
            self._trace_eef_pos = torch.full(
                trace_shape, float("nan"), device=self.device, dtype=torch.float32
            )
            self._trace_cube_pos = torch.full_like(self._trace_eef_pos, float("nan"))
            self._trace_goal_pos = torch.full_like(self._trace_eef_pos, float("nan"))
            self._trace_push_success = torch.zeros(
                trace_shape[:2], device=self.device, dtype=torch.bool
            )
            self._trace_cube_a_pos = torch.full_like(self._trace_eef_pos, float("nan"))
            self._trace_cube_b_pos = torch.full_like(self._trace_eef_pos, float("nan"))
            self._trace_cube_a_grasped = torch.zeros(trace_shape[:2], device=self.device, dtype=torch.bool)
            self._trace_stack_success = torch.zeros(trace_shape[:2], device=self.device, dtype=torch.bool)
            self._trace_left_ball_force = torch.full(trace_shape[:2], float("nan"), device=self.device)
            self._trace_right_ball_force = torch.full_like(self._trace_left_ball_force, float("nan"))
            self._trace_left_ball_angle = torch.full_like(self._trace_left_ball_force, float("nan"))
            self._trace_right_ball_angle = torch.full_like(self._trace_left_ball_force, float("nan"))
            self._trace_gripper_width = torch.full_like(self._trace_left_ball_force, float("nan"))
            self._trace_ball_height = torch.full_like(self._trace_left_ball_force, float("nan"))
            self._trace_ball_to_tcp_distance = torch.full_like(self._trace_left_ball_force, float("nan"))
            self._trace_left_ball_contact = torch.zeros(trace_shape[:2], device=self.device, dtype=torch.bool)
            self._trace_right_ball_contact = torch.zeros_like(self._trace_left_ball_contact)
            self._trace_grasp_confirmed_raw = torch.zeros_like(self._trace_left_ball_contact)
            self._trace_grasp_confirmed = torch.zeros_like(self._trace_left_ball_contact)
            self._trace_lift_confirmed = torch.zeros_like(self._trace_left_ball_contact)
            self._trace_agent_is_grasping = torch.zeros_like(self._trace_left_ball_contact)
            self._trace_tool_pos = torch.full_like(self._trace_eef_pos, float("nan"))
            self._trace_ball_velocity = torch.full_like(self._trace_eef_pos, float("nan"))
            self._trace_tool_grasp_confirmed = torch.zeros_like(self._trace_left_ball_contact)
            self._trace_tool_grasp_raw = torch.zeros_like(self._trace_left_ball_contact)
            self._trace_ball_forward_velocity = torch.full_like(self._trace_left_ball_force, float("nan"))
            self._trace_pull_success = torch.zeros_like(self._trace_left_ball_contact)
            self._tool_grasp_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
            self._grasp_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
            self._lift_baseline = torch.full((self.num_envs,), float("nan"), device=self.device)
            self._trace_actions = None

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
                self.max_rewards[mask] = -float("inf")
                self._reset_kinematic_trace(mask)
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
                self.max_rewards[:] = -float("inf")
                self._reset_kinematic_trace()

    def _reset_kinematic_trace(self, mask=None) -> None:
        if not self.record_kinematic_trace:
            return
        if mask is None:
            self._trace_eef_pos.fill_(float("nan"))
            self._trace_cube_pos.fill_(float("nan"))
            self._trace_goal_pos.fill_(float("nan"))
            self._trace_push_success.fill_(False)
            self._trace_cube_a_pos.fill_(float("nan")); self._trace_cube_b_pos.fill_(float("nan"))
            self._trace_cube_a_grasped.fill_(False); self._trace_stack_success.fill_(False)
            for name in ("_trace_left_ball_force", "_trace_right_ball_force", "_trace_left_ball_angle", "_trace_right_ball_angle", "_trace_gripper_width", "_trace_ball_height", "_trace_ball_to_tcp_distance", "_lift_baseline"):
                getattr(self, name).fill_(float("nan"))
            for name in ("_trace_left_ball_contact", "_trace_right_ball_contact", "_trace_grasp_confirmed_raw", "_trace_grasp_confirmed", "_trace_lift_confirmed", "_trace_agent_is_grasping"):
                getattr(self, name).fill_(False)
            self._trace_tool_pos.fill_(float("nan")); self._trace_ball_velocity.fill_(float("nan"))
            self._trace_tool_grasp_confirmed.fill_(False); self._trace_tool_grasp_raw.fill_(False)
            self._trace_ball_forward_velocity.fill_(float("nan")); self._trace_pull_success.fill_(False)
            self._tool_grasp_streak.zero_()
            self._grasp_streak.zero_()
            self._trace_actions = None
            return
        self._trace_eef_pos[mask] = float("nan")
        self._trace_cube_pos[mask] = float("nan")
        self._trace_goal_pos[mask] = float("nan")
        self._trace_push_success[mask] = False
        self._trace_cube_a_pos[mask] = float("nan"); self._trace_cube_b_pos[mask] = float("nan")
        self._trace_cube_a_grasped[mask] = False; self._trace_stack_success[mask] = False
        for name in ("_trace_left_ball_force", "_trace_right_ball_force", "_trace_left_ball_angle", "_trace_right_ball_angle", "_trace_gripper_width", "_trace_ball_height", "_trace_ball_to_tcp_distance", "_lift_baseline"):
            getattr(self, name)[mask] = float("nan")
        for name in ("_trace_left_ball_contact", "_trace_right_ball_contact", "_trace_grasp_confirmed_raw", "_trace_grasp_confirmed", "_trace_lift_confirmed", "_trace_agent_is_grasping"):
            getattr(self, name)[mask] = False
        self._trace_tool_pos[mask] = float("nan"); self._trace_ball_velocity[mask] = float("nan")
        self._trace_tool_grasp_confirmed[mask] = False; self._trace_tool_grasp_raw[mask] = False
        self._trace_ball_forward_velocity[mask] = float("nan"); self._trace_pull_success[mask] = False
        self._tool_grasp_streak[mask] = 0
        self._grasp_streak[mask] = 0
        if self._trace_actions is not None:
            self._trace_actions[mask] = float("nan")

    def _pose_position(self, obj, name: str) -> torch.Tensor:
        pose = getattr(obj, "pose", None)
        pos = getattr(pose, "p", None)
        if pos is None:
            raise AttributeError(f"{name}.pose.p is unavailable")
        if not isinstance(pos, torch.Tensor):
            pos = torch.as_tensor(pos, device=self.device, dtype=torch.float32)
        return pos.to(device=self.device, dtype=torch.float32)

    def _record_current_kinematic_trace(self, episode_info: dict, actions) -> None:
        if not self.record_kinematic_trace:
            return
        unw = self.env.unwrapped
        eef_pos = self._pose_position(unw.agent.tcp, "agent.tcp")
        evaluate_result = unw.evaluate()
        push_success = evaluate_result.get("success", False)
        if not isinstance(push_success, torch.Tensor):
            push_success = torch.as_tensor(
                push_success, device=self.device, dtype=torch.bool
            )
        push_success = push_success.to(device=self.device, dtype=torch.bool)

        step_idx = (self.elapsed_steps.to(self.device).long() - 1).clamp(
            min=0, max=self._trace_eef_pos.shape[1] - 1
        )
        env_idx = torch.arange(self.num_envs, device=self.device)
        self._trace_eef_pos[env_idx, step_idx] = eef_pos
        action_tensor = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if self._trace_actions is None:
            self._trace_actions = torch.full(
                (*self._trace_eef_pos.shape[:2], action_tensor.shape[-1]),
                float("nan"), device=self.device, dtype=torch.float32,
            )
        self._trace_actions[env_idx, step_idx] = action_tensor
        cube_obj = getattr(unw, "obj", getattr(unw, "cube", None))
        goal_obj = getattr(unw, "goal_region", getattr(unw, "goal_site", None))
        cube_pos = self._pose_position(cube_obj, "obj") if cube_obj is not None else torch.full_like(eef_pos, float("nan"))
        goal_pos = self._pose_position(goal_obj, "goal_region") if goal_obj is not None else torch.full_like(eef_pos, float("nan"))
        self._trace_cube_pos[env_idx, step_idx] = cube_pos
        self._trace_goal_pos[env_idx, step_idx] = goal_pos
        self._trace_push_success[env_idx, step_idx] = push_success

        # PullCubeTool-golf task-specific observables.  ``is_grasping`` is
        # backed by bilateral contact geometry in ManiSkill; combine it with
        # a closed gripper and three consecutive frames to reject fake closes.
        tool = getattr(unw, "l_shape_tool", None)
        if tool is not None:
            tool_pos = self._pose_position(tool, "l_shape_tool")
            self._trace_tool_pos[env_idx, step_idx] = tool_pos
            ball_velocity = getattr(getattr(cube_obj, "get_velocity", None), "__call__", lambda: None)()
            if ball_velocity is None:
                ball_velocity = getattr(cube_obj, "velocity", None)
            if ball_velocity is None:
                ball_velocity = torch.zeros_like(cube_pos)
            ball_velocity = torch.as_tensor(ball_velocity, device=self.device, dtype=torch.float32)
            if ball_velocity.ndim == 1:
                ball_velocity = ball_velocity.unsqueeze(0).expand(self.num_envs, -1)
            self._trace_ball_velocity[env_idx, step_idx] = ball_velocity
            raw_tool = unw.agent.is_grasping(tool, max_angle=20)
            raw_tool = torch.as_tensor(raw_tool, device=self.device, dtype=torch.bool).reshape(-1)
            qpos = unw.agent.robot.get_qpos()
            width = torch.abs(qpos[..., -2:]).sum(dim=1)
            if hasattr(unw.agent, "finger1_link") and hasattr(unw.agent, "finger2_link"):
                left_vec = unw.scene.get_pairwise_contact_forces(unw.agent.finger1_link, tool)
                right_vec = unw.scene.get_pairwise_contact_forces(unw.agent.finger2_link, tool)
                left_dir = unw.agent.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
                right_dir = -unw.agent.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
                left_force = torch.linalg.norm(left_vec, dim=1)
                right_force = torch.linalg.norm(right_vec, dim=1)
                left_angle = torch.rad2deg(common.compute_angle_between(left_dir, left_vec))
                right_angle = torch.rad2deg(common.compute_angle_between(right_dir, right_vec))
                raw_tool = raw_tool & (left_force >= 0.5) & (right_force >= 0.5) & (left_angle <= 85.0) & (right_angle <= 85.0)
            raw_tool = raw_tool & (width <= 0.04)
            self._tool_grasp_streak = torch.where(raw_tool, self._tool_grasp_streak + 1, torch.zeros_like(self._tool_grasp_streak))
            tool_grasp = self._tool_grasp_streak >= 3
            robot_base = unw.agent.robot.get_links()[0].pose.p
            axis = robot_base[:, :2] + torch.tensor([0.05, 0.0], device=self.device) - cube_pos[:, :2]
            axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp_min(1e-6)
            forward_v = (ball_velocity[:, :2] * axis).sum(dim=1)
            self._trace_tool_grasp_raw[env_idx, step_idx] = raw_tool
            self._trace_tool_grasp_confirmed[env_idx, step_idx] = tool_grasp
            self._trace_ball_forward_velocity[env_idx, step_idx] = forward_v
            self._trace_pull_success[env_idx, step_idx] = push_success

        def pos_or_none(name):
            obj = getattr(unw, name, None)
            return self._pose_position(obj, name) if obj is not None and getattr(obj, "pose", None) is not None else None
        cube_a = pos_or_none("cubeA")
        cube_b = pos_or_none("cubeB")
        if cube_a is not None and cube_b is not None:
            self._trace_cube_a_pos[env_idx, step_idx] = cube_a
            self._trace_cube_b_pos[env_idx, step_idx] = cube_b
            result = evaluate_result
            grasp = result.get("is_src_obj_grasped", result.get("is_cubeA_grasped", False))
            stack = result.get("success", False)
            self._trace_cube_a_grasped[env_idx, step_idx] = torch.as_tensor(grasp, device=self.device, dtype=torch.bool)
            self._trace_stack_success[env_idx, step_idx] = torch.as_tensor(stack, device=self.device, dtype=torch.bool)

        # Keep the exact Panda contact physics in the trajectory metrics.  This
        # is the same 0.5 N / 85 deg / 3-frame rule used by the standalone
        # PickCube-ball smoke test.
        cube = getattr(unw, "cube", None)
        if cube is not None and hasattr(unw.agent, "finger1_link"):
            left_vec = unw.scene.get_pairwise_contact_forces(unw.agent.finger1_link, cube)
            right_vec = unw.scene.get_pairwise_contact_forces(unw.agent.finger2_link, cube)
            left_dir = unw.agent.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
            right_dir = -unw.agent.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
            left_force = torch.linalg.norm(left_vec, dim=1)
            right_force = torch.linalg.norm(right_vec, dim=1)
            left_angle = torch.rad2deg(common.compute_angle_between(left_dir, left_vec))
            right_angle = torch.rad2deg(common.compute_angle_between(right_dir, right_vec))
            left_valid = (left_force >= 0.5) & (left_angle <= 85.0)
            right_valid = (right_force >= 0.5) & (right_angle <= 85.0)
            raw = left_valid & right_valid
            self._grasp_streak = torch.where(raw, self._grasp_streak + 1, torch.zeros_like(self._grasp_streak))
            confirmed = self._grasp_streak >= 3
            height = cube.pose.p[:, 2]
            qpos = unw.agent.robot.get_qpos()
            width = torch.abs(qpos[..., -2:]).sum(dim=1)
            tcp_distance = torch.linalg.norm(cube.pose.p - eef_pos, dim=1)
            self._lift_baseline = torch.where(torch.isnan(self._lift_baseline), height, self._lift_baseline)
            lifted = height >= self._lift_baseline + 0.04
            self._trace_left_ball_force[env_idx, step_idx] = left_force
            self._trace_right_ball_force[env_idx, step_idx] = right_force
            self._trace_left_ball_angle[env_idx, step_idx] = left_angle
            self._trace_right_ball_angle[env_idx, step_idx] = right_angle
            self._trace_gripper_width[env_idx, step_idx] = width
            self._trace_ball_height[env_idx, step_idx] = height
            self._trace_ball_to_tcp_distance[env_idx, step_idx] = tcp_distance
            self._trace_left_ball_contact[env_idx, step_idx] = left_force > 0
            self._trace_right_ball_contact[env_idx, step_idx] = right_force > 0
            self._trace_grasp_confirmed_raw[env_idx, step_idx] = raw
            self._trace_grasp_confirmed[env_idx, step_idx] = confirmed
            self._trace_lift_confirmed[env_idx, step_idx] = lifted
            self._trace_agent_is_grasping[env_idx, step_idx] = unw.agent.is_grasping(cube)

        episode_info["eef_pos"] = self._trace_eef_pos.clone()
        episode_info["cube_pos"] = self._trace_cube_pos.clone()
        episode_info["goal_pos"] = self._trace_goal_pos.clone()
        episode_info["push_success"] = self._trace_push_success.clone()
        episode_info["tool_pos"] = self._trace_tool_pos.clone()
        episode_info["ball_velocity"] = self._trace_ball_velocity.clone()
        episode_info["ball_velocity"] = self._trace_ball_velocity.clone()
        episode_info["tool_grasp_raw"] = self._trace_tool_grasp_raw.clone()
        episode_info["tool_grasp_confirmed"] = self._trace_tool_grasp_confirmed.clone()
        episode_info["ball_forward_velocity"] = self._trace_ball_forward_velocity.clone()
        episode_info["pull_success"] = self._trace_pull_success.clone()
        if self._trace_actions is not None:
            episode_info["executed_actions"] = self._trace_actions.clone()
        if cube_a is not None and cube_b is not None:
            episode_info["cube_a_pos"] = self._trace_cube_a_pos.clone()
            episode_info["cube_b_pos"] = self._trace_cube_b_pos.clone()
            episode_info["cube_a_grasped"] = self._trace_cube_a_grasped.clone()
            episode_info["stack_success"] = self._trace_stack_success.clone()
        for name in ("left_ball_force", "right_ball_force", "left_ball_angle", "right_ball_angle",
                     "gripper_width", "ball_height", "ball_to_tcp_distance",
                     "left_ball_contact", "right_ball_contact", "grasp_confirmed_raw",
                     "grasp_confirmed", "lift_confirmed", "agent_is_grasping"):
            episode_info[name] = getattr(self, f"_trace_{name}").clone()

    def _record_metrics(self, step_reward, infos, actions):
        episode_info = {}
        self.returns += step_reward
        self.max_rewards = torch.maximum(self.max_rewards, step_reward)
        if "success" in infos:
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
            episode_info["fail_once"] = self.fail_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["max_reward"] = self.max_rewards.clone()
        if self._initial_state_records:
            episode_info["manifest_reference_index"] = self.reset_state_ids.clone()
        self._record_current_kinematic_trace(episode_info, actions)
        infos["episode"] = episode_info
        return infos

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = None,
    ):
        if options is None:
            seed = self.reset_seed
            options = (
                {"episode_id": self.reset_state_ids}
                if self.use_fixed_reset_state_ids
                else {}
            )
        else:
            options = dict(options)
            if self.shared_reset_seed:
                seed = self.seed
        base_options = self._base_reset_options()
        if base_options:
            merged_options = dict(base_options)
            merged_options.update(options)
            options = merged_options
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        raw_obs = self._restore_manifest_states(raw_obs)
        # Pick-up replay-render for the robometer (train only). The task env's
        # _initialize_episode (above, when pre_grasped) stashed a per-env
        # pick-up state trajectory; replay-render reward_camera at each state
        # and stash the frames for env_worker / the smoke to prepend to the
        # robometer history buffer. Pick-up frames never become rollout steps,
        # so they do NOT enter RL training data. Eval (is_eval) is skipped.
        self._pending_pickup_frames = {}
        if (
            bool(getattr(self.cfg, "capture_render_image_for_reward", False))
            and not bool(getattr(self.cfg, "is_eval", False))
            and getattr(self.env.unwrapped, "_pending_pickup_trajectories", None)
        ):
            env_idx = options.get("env_idx")
            if env_idx is None:
                env_idx = torch.arange(self.num_envs, device=self.device)
            try:
                self._pending_pickup_frames = self._render_pickup_frames(env_idx)
            except Exception:  # noqa: BLE001  never break reset on a render hiccup
                self._pending_pickup_frames = {}
            self.env.unwrapped._pending_pickup_trajectories = {}
        self._show_goal_site_visual()
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        self._last_policy_obs = extracted_obs
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return extracted_obs, infos

    def _restore_manifest_states(self, raw_obs):
        if not self._initial_state_records:
            return raw_obs
        # Reset first so simulator bookkeeping exists, then replace each selected
        # vector slot with the complete saved state. The hash is checked by the
        # manifest builder/labeler; this path never silently falls back to a seed.
        current = self.env.unwrapped.get_state_dict()
        ids = self.reset_state_ids.detach().cpu().tolist() if hasattr(self.reset_state_ids, "detach") else list(range(self.num_envs))
        saved_by_env = []
        for env_idx, ref_idx in enumerate(ids):
            record = self._initial_state_records[int(ref_idx) % len(self._initial_state_records)]
            saved = torch.load(record["state_path"], map_location="cpu", weights_only=False)
            saved_by_env.append(saved)
            def merge(dst, src):
                if isinstance(dst, dict):
                    for key in dst:
                        if key in src: merge(dst[key], src[key])
                else:
                    src = torch.as_tensor(src, device=dst.device, dtype=dst.dtype)
                    if dst.ndim == src.ndim: dst[env_idx] = src
                    elif dst.ndim == src.ndim + 1: dst[env_idx] = src
            merge(current, saved)
        self.env.unwrapped.set_state_dict(current)
        self.env.unwrapped.scene.update_render()
        restored = self.env.unwrapped.get_state_dict()
        for env_idx, ref_idx in enumerate(ids):
            record = self._initial_state_records[int(ref_idx) % len(self._initial_state_records)]
            expected = record.get("initial_state_sha256")
            if expected:
                restored_state = env_slice(restored, env_idx)
                actual = state_hash(restored_state)
                difference = compare_states(saved_by_env[env_idx], restored_state)
                if actual != expected and difference is not None:
                    raise RuntimeError(
                        f"exact initial-state hash mismatch for env={env_idx}, "
                        f"reference={record.get('id')}: {actual} != {expected}; "
                        f"difference={difference}"
                    )
        return self.env.unwrapped.get_obs()

    def reset_all_for_rollout_window(self):
        """Reset all vectorized environments for an independent rollout window.

        Thin wrapper over ``reset`` signaling a fresh episode start
        (``is_start=True``) so pick-up replay-render stashes new frames for the
        robometer history prefix. This is a worker-level rollout-window semantic
        (independent windows reset between windows), NOT an environment-dynamics
        change: it does not forge done flags, modify chunk_step, or call
        update_reset_state_ids (existing shared_reset_seed /
        use_fixed_reset_state_ids reproduction semantics are preserved).
        """
        self.is_start = True
        return self.reset()

    def consume_pickup_frames(self):
        """Return and clear stashed pick-up render frames (for robometer prepend).

        Called by env_worker (initial reset + auto-reset) and the single-process
        smoke after reset. Returns ``{global_env_idx: [HxWxC uint8 np arrays]}``;
        ``{}`` when nothing was stashed (non-pre-grasped / eval / render failed).
        """
        frames = getattr(self, "_pending_pickup_frames", None) or {}
        self._pending_pickup_frames = {}
        return frames

    def _render_pickup_frames(self, env_idx):
        """Replay-render the planner's pick-up state trajectory on the GPU env.

        For each recorded pick-up state (approach -> grasp -> lift), set
        peg/box/robot state, run the GPU kinematics sequence, render
        reward_camera, and collect the frame. Non-subset envs are held at their
        current state and the full sim state is restored at the end, so this is
        safe mid-episode (auto-reset subset). Returns
        ``{global_env_idx: [HxWxC uint8 np arrays]}``.
        """
        from mani_skill.utils.structs.pose import Pose

        unw = self.env.unwrapped
        trajs = getattr(unw, "_pending_pickup_trajectories", None) or {}
        if hasattr(env_idx, "detach"):
            gi = env_idx.detach().cpu().numpy()
        else:
            gi = np.asarray(env_idx)
        gi = gi.reshape(-1).astype(np.int64).tolist()
        max_len = max(
            (len(trajs[g]) for g in gi if g in trajs and trajs[g]), default=0
        )
        if max_len == 0:
            return {}

        render_cam_key = getattr(self.cfg, "render_camera_key", "reward_camera")
        # Snapshot current full per-env state to hold non-subset envs + restore.
        peg_raw = unw.peg.pose.raw_pose.clone()  # [num_envs, 7] (p, q)
        box_raw = unw.box.pose.raw_pose.clone()
        qpos_cur = unw.agent.robot.get_qpos().clone()  # [num_envs, 9]
        saved = unw.get_state_dict()  # full sim state for final restore

        frames: dict[int, list] = {g: [] for g in gi}
        for t in range(max_len):
            peg_t = peg_raw.clone()
            box_t = box_raw.clone()
            qpos_t = qpos_cur.clone()
            for g in gi:
                traj = trajs.get(g)
                if not traj:
                    continue
                rec = traj[min(t, len(traj) - 1)]
                peg_t[g] = torch.as_tensor(
                    rec["peg_pose"], dtype=peg_raw.dtype, device=self.device
                )
                box_t[g] = torch.as_tensor(
                    rec["hole_pose"], dtype=box_raw.dtype, device=self.device
                )
                qpos_t[g] = torch.as_tensor(
                    rec["robot_qpos"], dtype=qpos_cur.dtype, device=self.device
                )
            unw.peg.set_pose(Pose.create_from_pq(peg_t[:, :3], peg_t[:, 3:]))
            unw.box.set_pose(Pose.create_from_pq(box_t[:, :3], box_t[:, 3:]))
            unw.agent.robot.set_qpos(qpos_t)
            if getattr(unw, "gpu_sim_enabled", False):
                unw.scene._gpu_apply_all()
                unw.scene.px.gpu_update_articulation_kinematics()
                unw.scene._gpu_fetch_all()
            unw.scene.update_render()
            sensor_data = unw.get_obs()["sensor_data"]
            if render_cam_key not in sensor_data:
                continue
            render = common.to_numpy(sensor_data[render_cam_key]["rgb"])
            for g in gi:
                frames[g].append(np.asarray(render[g]).astype(np.uint8))

        # Restore full sim state (non-subset envs unchanged; subset envs left at
        # the lift-end reset state the policy expects = the trajectory's last
        # frame, which _initialize_episode already set).
        try:
            unw.set_state_dict(saved)
        except Exception:  # noqa: BLE001
            pass
        return frames

    def step(
        self, actions: Union[Array, dict] = None, auto_reset=True
    ) -> tuple[Array, Array, Array, Array, dict]:
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        extracted_obs = self._wrap_obs(raw_obs, infos=infos)
        self._last_policy_obs = extracted_obs
        step_reward = self._calc_step_reward(_reward, infos)

        infos = self._record_metrics(step_reward, infos, actions)
        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
        if isinstance(truncations, bool):
            truncations = torch.tensor([truncations], device=self.device)
            truncations = truncations.repeat(self.num_envs)
        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    infos["episode"]["fail_at_end"] = infos["fail"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)
        self._last_policy_obs = extracted_obs
        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        if self.sync_target_pose_per_chunk:
            # Re-anchor the target-delta controller to the actual EE before executing a
            # fresh model-predicted chunk, so deltas integrate from the state the policy
            # was conditioned on instead of a stale commanded target.
            _sync_target_delta_pose_controller(self.env)
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = torch_clone_dict(extracted_obs)
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        options = {"env_idx": env_idx}
        final_info = torch_clone_dict(infos)
        if self.use_fixed_reset_state_ids:
            options.update(episode_id=self.reset_state_ids[env_idx])
        extracted_obs, infos = self.reset(options=options)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def run(self):
        obs, info = self.reset()
        for step in range(100):
            action = self.env.action_space.sample()
            obs, rew, terminations, truncations, infos = self.step(action)
            print(
                f"Step {step}: obs={obs.keys()}, rew={rew.mean()}, terminations={terminations.float().mean()}, truncations={truncations.float().mean()}"
            )

    # render utils
    def capture_image(self, infos=None):
        frames = []
        render_source = self.video_cfg.get("render_source", "observation")
        if render_source == "policy_input":
            policy_obs = getattr(self, "_last_policy_obs", None)
            if policy_obs is None:
                return None
            main_img = common.to_numpy(policy_obs["main_images"])
            wrist_img = policy_obs.get("wrist_images")
            wrist_img = None if wrist_img is None else common.to_numpy(wrist_img)
            if main_img.ndim == 3:
                main_img = main_img[None]
            if wrist_img is not None and wrist_img.ndim == 3:
                wrist_img = wrist_img[None]
            for i in range(len(main_img)):
                frame = main_img[i]
                if wrist_img is not None:
                    wrist = wrist_img[min(i, len(wrist_img) - 1)]
                    if frame.shape[:2] != wrist.shape[:2]:
                        # Video is a display-only montage. Keep both exact
                        # policy cameras, resizing the render view to the
                        # wrist stream's 224x224 display resolution.
                        from PIL import Image

                        frame = np.asarray(
                            Image.fromarray(frame).resize(
                                (wrist.shape[1], wrist.shape[0]), Image.Resampling.BILINEAR
                            )
                        )
                    frame = np.concatenate([frame, wrist], axis=1)
                frames.append(frame)
        elif render_source in ("human_render", "render_camera", "human"):
            camera_name = self.video_cfg.get("render_camera_name", "render_camera")
            render_img = self.env.unwrapped.render_rgb_array(camera_name=camera_name)
            if render_img is None:
                _logger.warning(
                    "Failed to capture human render camera %r; falling back to observation cameras.",
                    camera_name,
                )
                render_source = "observation"
            else:
                render_img = common.to_numpy(render_img)
                if len(render_img.shape) == 3:
                    render_img = render_img[None]
                frames = [render_img[i] for i in range(len(render_img))]

        if render_source not in ("policy_input", "human_render", "render_camera", "human"):
            raw_obs = self.env.unwrapped.get_obs()
            sensor_data = raw_obs["sensor_data"]
            base_img = common.to_numpy(sensor_data["base_camera"]["rgb"])
            wrist_img = None
            if bool(getattr(self.cfg, "use_wrist_image", False)) and "hand_camera" in sensor_data:
                wrist_img = common.to_numpy(sensor_data["hand_camera"]["rgb"])

            if len(base_img.shape) == 3:
                base_img = base_img[None]
            if wrist_img is not None and len(wrist_img.shape) == 3:
                wrist_img = wrist_img[None]

            for i in range(len(base_img)):
                frame = base_img[i]
                if wrist_img is not None:
                    if wrist_img.shape[0] > i:
                        frame = np.concatenate([frame, wrist_img[i]], axis=1)
                    else:
                        frame = np.concatenate([frame, wrist_img[0]], axis=1)
                frames.append(frame)

        if infos is not None:
            for i in range(len(frames)):
                info_item = {
                    k: v if np.size(v) == 1 else v[i] for k, v in infos.items()
                }
                frames[i] = put_info_on_image(frames[i], info_item)
        if len(frames) > 1:
            return tile_images(frames, nrows=int(np.sqrt(self.num_envs)))
        return frames[0]

    def render(self, info, rew=None):
        if self.video_cfg.info_on_video:
            scalar_info = gym_utils.extract_scalars_from_info(
                common.to_numpy(info), batch_size=self.num_envs
            )
            if rew is not None:
                scalar_info["reward"] = common.to_numpy(rew)
                if np.size(scalar_info["reward"]) > 1:
                    scalar_info["reward"] = [
                        float(rew) for rew in scalar_info["reward"]
                    ]
                else:
                    scalar_info["reward"] = float(scalar_info["reward"])
            image = self.capture_image(scalar_info)
        else:
            image = self.capture_image()
        return image

    def sample_action_space(self):
        return self.env.action_space.sample()
