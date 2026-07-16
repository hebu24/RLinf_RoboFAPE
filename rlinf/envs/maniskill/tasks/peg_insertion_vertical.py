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

from typing import Any, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from transforms3d.euler import euler2quat

from rlinf.envs.maniskill.peg_insertion_pi05 import (
    aligned_pi05_state_from_tcp_matrices,
)


def _build_vertical_box_with_hole(
    scene: ManiSkillScene,
    inner_radius: float,
    outer_radius: float,
    depth: float,
):
    """Build a square vertical tunnel whose local +x axis is the insertion axis."""
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    offset = thickness + inner_radius
    mat = sapien.render.RenderMaterial(base_color=np.array([249, 140, 54, 255]) / 255)

    half_sizes = [
        [depth, thickness, outer_radius],
        [depth, thickness, outer_radius],
        [depth, outer_radius, thickness],
        [depth, outer_radius, thickness],
    ]
    poses = [
        sapien.Pose([0, offset, 0]),
        sapien.Pose([0, -offset, 0]),
        sapien.Pose([0, 0, offset]),
        sapien.Pose([0, 0, -offset]),
    ]

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose=pose, half_size=half_size)
        builder.add_box_visual(pose=pose, half_size=half_size, material=mat)
    return builder


def _build_table_plane(
    scene: ManiSkillScene,
    half_sizes,
    color,
    altitude: float,
    name: str = "light_gray_table",
):
    builder = scene.create_actor_builder()
    mat = sapien.render.RenderMaterial(base_color=color)
    builder.add_plane_collision(
        pose=sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0])
    )
    builder.add_plane_repeated_visual(
        pose=sapien.Pose(p=[0, 0, altitude]),
        half_size=[half_sizes[0], half_sizes[1]],
        mat=mat,
    )
    builder.initial_pose = sapien.Pose()
    return builder.build_static(name=name)


def _external_camera_pose():
    return sapien.Pose(
        p=[0.705400, -0.086655, 0.686691],
        q=[0.025112, -0.237384, -0.033640, 0.970508],
    )


def _with_wrist_camera_sensor_config(sensor_configs):
    merged = dict(sensor_configs or {})
    hand_camera = dict(merged.get("hand_camera", {}))
    hand_camera.setdefault("width", 224)
    hand_camera.setdefault("height", 224)
    merged["hand_camera"] = hand_camera
    # Back-facing wrist camera shares the front wrist resolution so the model's
    # left/right_wrist_0_rgb image slots have matching shapes.
    hand_camera_back = dict(merged.get("hand_camera_back", {}))
    hand_camera_back.setdefault("width", 224)
    hand_camera_back.setdefault("height", 224)
    merged["hand_camera_back"] = hand_camera_back
    return merged


@register_env("PegInsertionVertical-v1", max_episode_steps=600)
class PegInsertionVerticalEnv(BaseEnv):
    """Coordinate the wrist-camera-guided robot arm to insert a vertical peg."""

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda"]
    agent: Union[PandaWristCam, Panda]

    peg_half_length = 0.105
    peg_half_width = 0.024
    hole_clearance = 0.004
    hole_outer_radius = 0.075
    hole_half_depth = 0.04
    table_half_size = [0.75, 0.35, 0.003]
    table_top_z = 0.0
    default_robot_qpos = np.array(
        [
            0.0,
            np.pi / 8,
            0,
            -np.pi * 5 / 8,
            0,
            np.pi * 3 / 4,
            np.pi / 4,
            0.04,
            0.04,
        ],
        dtype=np.float32,
    )
    robot_qpos_randomization_scale = np.array(
        [0.18, 0.14, 0.18, 0.12, 0.18, 0.14, 0.18],
        dtype=np.float32,
    )
    robot_qpos_limit_margin = 0.12
    robot_tcp_init_bounds = np.array(
        [
            [-0.42, -0.22],
            [-0.18, 0.18],
            [0.18, 0.42],
        ],
        dtype=np.float32,
    )
    hole_xy_randomization_bounds = np.array(
        [
            [-0.12, 0.12],
            [-0.12, 0.12],
        ],
        dtype=np.float32,
    )
    peg_relative_hole_radius_bounds = np.array([0.17, 0.30], dtype=np.float32)
    peg_xy_randomization_bounds = np.array(
        [
            [-0.30, 0.15],
            [-0.30, 0.30],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        obj_set: str | None = None,
        robot_init_qpos_noise=0.02,
        render_randomization_spec=None,
        **kwargs,
    ):
        self.obj_set = obj_set
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.render_randomization_spec = render_randomization_spec
        # Insert-only eval: a PegInsertionLiftPlanner registered via
        # set_lift_planner(). Must be set before super().__init__ since
        # BaseEnv construction may trigger an early _initialize_episode.
        self._lift_planner = None
        if robot_uids == "panda_wristcam":
            kwargs["sensor_configs"] = _with_wrist_camera_sensor_config(
                kwargs.get("sensor_configs")
            )
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        configs = [
            CameraConfig(
                "base_camera", _external_camera_pose(), 224, 224, 1.0, 0.01, 100
            )
        ]
        # Back-facing wrist camera (eye-in-hand): mounted on the same camera_link
        # as the front hand_camera but flipped 180 deg about x for a rear view,
        # complementary to the front wrist camera. Only panda_wristcam has a
        # camera_link; skipped for the base panda robot.
        links_map = getattr(getattr(self.agent, "robot", None), "links_map", {})
        if "camera_link" in links_map:
            configs.append(
                CameraConfig(
                    "hand_camera_back",
                    sapien.Pose(p=[0, 0, 0], q=[0, 1, 0, 0]),
                    224,
                    224,
                    1.0,
                    0.01,
                    100,
                    mount=links_map["camera_link"],
                )
            )
        return configs

    @property
    def _default_human_render_camera_configs(self):
        pose = _external_camera_pose()
        camera_spec = (self.render_randomization_spec or {}).get("camera")
        if camera_spec:
            pose = sapien.Pose(
                p=np.asarray(camera_spec["p"], dtype=np.float32),
                q=np.asarray(camera_spec["q"], dtype=np.float32),
            )

        return CameraConfig("render_camera", pose, 640, 480, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def get_language_instruction(self):
        return ["insert the peg into the hole"] * self.num_envs

    def _load_scene(self, options: dict):
        self.table = _build_table_plane(
            self.scene,
            half_sizes=self.table_half_size,
            color=[0.50, 0.50, 0.50, 1],
            altitude=self.table_top_z,
        )
        self._build_curtains()

        self.peg_half_sizes = torch.tensor(
            [[self.peg_half_length, self.peg_half_width, self.peg_half_width]],
            device=self.device,
        ).repeat(self.num_envs, 1)
        peg_head_offsets = torch.zeros((self.num_envs, 3), device=self.device)
        peg_head_offsets[:, 0] = self.peg_half_length
        self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)
        self.box_hole_radii = torch.full(
            (self.num_envs,),
            self.peg_half_width + self.hole_clearance,
            device=self.device,
        )
        self.box_hole_offsets = Pose.create_from_pq(
            p=torch.zeros((self.num_envs, 3), device=self.device)
        )

        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            color_1=np.array([0, 134, 214, 255]) / 255,
            color_2=np.array([0, 134, 214, 255]) / 255,
            name="peg",
            body_type="dynamic",
            initial_pose=sapien.Pose(
                p=[0, -0.18, self.table_top_z + self.peg_half_width]
            ),
        )

        builder = _build_vertical_box_with_hole(
            self.scene,
            inner_radius=self.peg_half_width + self.hole_clearance,
            outer_radius=self.hole_outer_radius,
            depth=self.hole_half_depth,
        )
        builder.initial_pose = sapien.Pose(
            p=[0.12, 0, self.table_top_z + self.hole_half_depth],
            q=euler2quat(0, np.pi / 2, 0),
        )
        self.box = builder.build_kinematic("vertical_box_with_hole")

    def _build_curtains(self):
        curtain_color = [0.01, 0.01, 0.012, 1]
        curtain_bottom_z = self.table_top_z
        curtain_height = 1.2
        curtain_center_z = curtain_bottom_z + curtain_height / 2
        actors.build_box(
            self.scene,
            half_sizes=[0.02, 0.65, curtain_height / 2],
            color=curtain_color,
            name="back_black_curtain",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[-0.75, 0, curtain_center_z]),
        )
        actors.build_box(
            self.scene,
            half_sizes=[0.85, 0.02, curtain_height / 2],
            color=curtain_color,
            name="left_black_curtain",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[-0.25, 0.35, curtain_center_z]),
        )
        actors.build_box(
            self.scene,
            half_sizes=[0.85, 0.02, curtain_height / 2],
            color=curtain_color,
            name="right_black_curtain",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[-0.25, -0.35, curtain_center_z]),
        )

    def _load_lighting(self, options: dict):
        lighting_spec = (self.render_randomization_spec or {}).get("lighting")
        for scene in self.scene.sub_scenes:
            if not lighting_spec:
                scene.ambient_light = [0.55, 0.55, 0.55]
                scene.add_directional_light(
                    [0, 0, -1], [1.8, 1.8, 1.8], shadow=True, shadow_scale=5
                )
                scene.add_directional_light([1, 1, -1], [0.55, 0.55, 0.55])
                continue
            scene.ambient_light = lighting_spec["ambient"]
            for light in lighting_spec.get("directional_lights", []):
                kwargs = {}
                if light.get("shadow"):
                    kwargs["shadow"] = True
                    if "shadow_scale" in light:
                        kwargs["shadow_scale"] = light["shadow_scale"]
                scene.add_directional_light(
                    light["direction"], light["color"], **kwargs
                )

    def _as_pose_option(self, value, batch_size):
        if value is None:
            return None
        if isinstance(value, dict):
            p = value.get("p")
            q = value.get("q", [1.0, 0.0, 0.0, 0.0])
        else:
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape[-1] != 7:
                raise ValueError(
                    "Pose options must be dicts with p/q or arrays with 7 values."
                )
            p = arr[..., :3]
            q = arr[..., 3:]
        p = torch.as_tensor(p, dtype=torch.float32, device=self.device).reshape(-1, 3)
        q = torch.as_tensor(q, dtype=torch.float32, device=self.device).reshape(-1, 4)
        if p.shape[0] == 1 and batch_size > 1:
            p = p.repeat(batch_size, 1)
            q = q.repeat(batch_size, 1)
        return Pose.create_from_pq(p, q)

    def _robot_qlimits(self):
        getter = getattr(self.agent.robot, "get_qlimits", None)
        if getter is None:
            return None
        limits = getter()
        if hasattr(limits, "detach"):
            limits = limits.detach().cpu().numpy()
        limits = np.asarray(limits, dtype=np.float32)
        if limits.ndim == 3:
            limits = limits[0]
        return limits

    def _sample_random_robot_qpos(self, env_idx, episode_rngs):
        batch_size = len(env_idx)
        qpos = np.tile(self.default_robot_qpos, (batch_size, 1))
        original_qpos = self.agent.robot.get_qpos()[env_idx]
        if hasattr(original_qpos, "detach"):
            original_qpos = original_qpos.detach().cpu().numpy()
        env_indices = env_idx.detach().cpu().numpy()
        qlimits = self._robot_qlimits()
        lower = upper = None
        if qlimits is not None and qlimits.shape[0] >= qpos.shape[1]:
            lower = qlimits[: qpos.shape[1], 0] + self.robot_qpos_limit_margin
            upper = qlimits[: qpos.shape[1], 1] - self.robot_qpos_limit_margin
            lower[-2:] = 0.04
            upper[-2:] = 0.04

        for i in range(batch_size):
            episode_rng = episode_rngs[i]
            accepted = None
            for _ in range(100):
                candidate = self.default_robot_qpos.copy()
                candidate[:7] += episode_rng.uniform(
                    -self.robot_qpos_randomization_scale,
                    self.robot_qpos_randomization_scale,
                )
                candidate[-2:] = 0.04
                if lower is not None:
                    candidate = np.clip(candidate, lower, upper)
                    if np.any(candidate[:7] <= lower[:7]) or np.any(
                        candidate[:7] >= upper[:7]
                    ):
                        continue
                self.agent.robot.set_qpos(candidate.reshape(1, -1))
                if self.gpu_sim_enabled:
                    self.scene.px.gpu_update_articulation_kinematics()
                tcp_pos = self.agent.tcp.pose.p
                if hasattr(tcp_pos, "detach"):
                    tcp_pos = tcp_pos.detach().cpu().numpy()
                tcp_pos = np.asarray(tcp_pos, dtype=np.float32).reshape(-1, 3)[
                    env_indices[0]
                ]
                if not np.all(
                    (self.robot_tcp_init_bounds[:, 0] <= tcp_pos)
                    & (tcp_pos <= self.robot_tcp_init_bounds[:, 1])
                ):
                    continue
                accepted = candidate
                break
            if accepted is None:
                accepted = self.default_robot_qpos.copy()
                accepted[:7] += episode_rng.normal(
                    0, self.robot_init_qpos_noise, size=7
                )
                if lower is not None:
                    accepted = np.clip(accepted, lower, upper)
                accepted[-2:] = 0.04
            qpos[i] = accepted
        if original_qpos is not None:
            self.agent.robot.set_qpos(original_qpos)
            if self.gpu_sim_enabled:
                self.scene.px.gpu_update_articulation_kinematics()
        return qpos

    def _sample_random_peg_xy_around_hole(self, hole_xy, episode_rngs):
        peg_xy = np.zeros_like(hole_xy, dtype=np.float32)
        for i, center_xy in enumerate(hole_xy):
            episode_rng = episode_rngs[i]
            accepted = None
            for _ in range(100):
                theta = episode_rng.uniform(-np.pi, np.pi)
                radius = episode_rng.uniform(
                    self.peg_relative_hole_radius_bounds[0],
                    self.peg_relative_hole_radius_bounds[1],
                )
                candidate = center_xy + radius * np.array(
                    [np.cos(theta), np.sin(theta)],
                    dtype=np.float32,
                )
                if np.all(
                    (self.peg_xy_randomization_bounds[:, 0] <= candidate)
                    & (candidate <= self.peg_xy_randomization_bounds[:, 1])
                ):
                    accepted = candidate
                    break
            if accepted is None:
                theta = episode_rng.uniform(-np.pi, np.pi)
                radius = self.peg_relative_hole_radius_bounds[0]
                accepted = center_xy + radius * np.array(
                    [np.cos(theta), np.sin(theta)],
                    dtype=np.float32,
                )
                accepted = np.clip(
                    accepted,
                    self.peg_xy_randomization_bounds[:, 0],
                    self.peg_xy_randomization_bounds[:, 1],
                )
            peg_xy[i] = accepted
        return peg_xy

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        options = options or {}
        # Insert-only eval: when pre_grasped is set and a lift planner is
        # registered, produce a freshly motion-planned grasped+lifted state for
        # the envs being reset (env_idx is the reset subset, incl. the
        # auto-reset done subset) and feed it through the existing override
        # path below. Reached by both the initial reset and every auto-reset.
        if (
            options.get("pre_grasped")
            and self._lift_planner is not None
            and options.get("peg_pose") is None
        ):
            options = self._inject_planned_lift_state(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            episode_rngs = self._batched_episode_rng[env_idx]
            peg_pos = torch.zeros((b, 3))
            peg_pos[:, 0] = -0.22
            peg_pos[:, 1] = -0.12
            peg_pos[:, 2] = self.table_top_z + self.peg_half_length
            peg_quat = torch.tensor(
                euler2quat(0, np.pi / 2, 0), device=self.device
            ).repeat(b, 1)
            peg_pose = Pose.create_from_pq(peg_pos, peg_quat)

            hole_pos = torch.zeros((b, 3))
            hole_pos[:, 0] = -0.04
            hole_pos[:, 1] = 0.02
            hole_pos[:, 2] = self.table_top_z + self.hole_half_depth
            hole_quat = torch.tensor(
                euler2quat(0, np.pi / 2, 0), device=self.device
            ).repeat(b, 1)
            hole_pose = Pose.create_from_pq(hole_pos, hole_quat)

            qpos = (
                np.stack(
                    [
                        episode_rngs[i].normal(
                            0,
                            self.robot_init_qpos_noise,
                            size=len(self.default_robot_qpos),
                        )
                        for i in range(b)
                    ]
                )
                + self.default_robot_qpos
            )

            if options.get("randomize_initial_poses"):
                hole_xy = episode_rngs.uniform(
                    self.hole_xy_randomization_bounds[:, 0],
                    self.hole_xy_randomization_bounds[:, 1],
                    size=(2,),
                )
                peg_xy = self._sample_random_peg_xy_around_hole(hole_xy, episode_rngs)
                hole_pos[:, :2] = torch.as_tensor(
                    hole_xy, dtype=torch.float32, device=self.device
                )
                peg_pos[:, :2] = torch.as_tensor(
                    peg_xy, dtype=torch.float32, device=self.device
                )
                peg_pose = Pose.create_from_pq(peg_pos, peg_quat)
                hole_pose = Pose.create_from_pq(hole_pos, hole_quat)
                qpos = self._sample_random_robot_qpos(env_idx, episode_rngs)

            peg_pose_override = self._as_pose_option(options.get("peg_pose"), b)
            hole_pose_override = self._as_pose_option(
                options.get("hole_pose", options.get("box_pose")),
                b,
            )
            if peg_pose_override is not None:
                peg_pose = peg_pose_override
            if hole_pose_override is not None:
                hole_pose = hole_pose_override
            if "robot_qpos" in options and options["robot_qpos"] is not None:
                robot_qpos = np.asarray(options["robot_qpos"], dtype=np.float32)
                if robot_qpos.ndim == 1:
                    robot_qpos = np.tile(robot_qpos, (b, 1))
                qpos = robot_qpos.reshape(b, -1)

            self.peg.set_pose(peg_pose)
            self.box.set_pose(hole_pose)
            # The normal pick-up setting always starts with an open gripper.
            # The insert-only eval supplies a pre-grasped robot_qpos (closed
            # fingers) via reset_options.pre_grasped, so keep those fingers
            # closed instead of forcing them open.
            if not options.get("pre_grasped"):
                qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.60, 0.06, 0]))

    def set_lift_planner(self, planner):
        """Register a PegInsertionLiftPlanner for insert-only evaluation.

        When set, ``_initialize_episode`` (gated on ``options["pre_grasped"]``)
        initializes each reset env with a motion-planned grasped+lifted peg.
        Reached by both the initial reset and every auto-reset.
        """
        self._lift_planner = planner

    def _inject_planned_lift_state(self, env_idx, options):
        import numpy as np

        if hasattr(env_idx, "detach"):
            gi = env_idx.detach().cpu().numpy()
        else:
            gi = np.asarray(env_idx)
        gi = gi.reshape(-1).astype(np.int64).tolist()
        state = self._lift_planner.plan_lifted_states(gi)
        merged = dict(options)
        merged.update(
            {
                "peg_pose": state["peg_pose"],
                "hole_pose": state["hole_pose"],
                "robot_qpos": state["robot_qpos"],
                "pre_grasped": True,
            }
        )
        return merged

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets

    @property
    def box_hole_pose(self):
        return self.box.pose * self.box_hole_offsets

    @property
    def goal_pose(self):
        return self.box_hole_pose * self.peg_head_offsets.inv()

    def _quat_angle(self, q1, q2):
        q1 = torch.nn.functional.normalize(q1, dim=1)
        q2 = torch.nn.functional.normalize(q2, dim=1)
        dot = torch.abs(torch.sum(q1 * q2, dim=1)).clamp(max=1.0)
        return 2 * torch.acos(dot)

    def _target_pose_tensors(self, target_pose):
        target_p = (
            torch.tensor(
                np.asarray(target_pose.p, dtype=np.float32),
                device=self.device,
            )
            .reshape(1, 3)
            .repeat(self.num_envs, 1)
        )
        target_q = (
            torch.tensor(
                np.asarray(target_pose.q, dtype=np.float32),
                device=self.device,
            )
            .reshape(1, 4)
            .repeat(self.num_envs, 1)
        )
        return target_p, target_q

    def peg_vertical_alignment(self):
        peg_rot = rotation_conversions.quaternion_to_matrix(self.peg.pose.q)
        peg_axis = peg_rot[:, :, 0]
        downward = torch.tensor([0, 0, -1], device=self.device, dtype=peg_axis.dtype)
        return torch.sum(peg_axis * downward, dim=1)

    def is_peg_vertical(self, max_angle_deg=10):
        return self.peg_vertical_alignment() > np.cos(np.deg2rad(max_angle_deg))

    def _tcp_pose_error(self, target_pose):
        target_p, target_q = self._target_pose_tensors(target_pose)
        tcp_pos_error = torch.linalg.norm(self.agent.tcp.pose.p - target_p, axis=1)
        tcp_rot_error = self._quat_angle(self.agent.tcp.pose.q, target_q)
        return tcp_pos_error, tcp_rot_error

    def evaluate_substage(self, stage, target_pose=None):
        """Single-env substage checks used by failure data collection.

        ``target_pose`` is accepted for backward compatibility with existing
        callers, but per-stage success is intentionally based only on task
        state rather than intended TCP poses.
        """
        peg_vertical = self.is_peg_vertical(max_angle_deg=10)
        is_grasped = self.agent.is_grasping(self.peg, max_angle=30)
        peg_lift = self.peg.pose.p[:, 2] - (self.table_top_z + self.peg_half_length)

        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
        head_lateral_dist = torch.linalg.norm(peg_head_wrt_goal.p[:, 1:], axis=1)
        peg_lateral_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

        inserted, peg_head_pos_at_hole = self.has_peg_inserted()

        if stage == "reach":
            success = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        elif stage == "grasp":
            success = is_grasped
        elif stage == "lift":
            success = is_grasped
        elif stage == "pre_insert":
            success = is_grasped | peg_vertical
        elif stage == "insert":
            success = inserted
        else:
            raise ValueError(f"Unsupported PegInsertionVertical substage: {stage}")

        diagnostics = {
            "success": success,
            "is_grasped": is_grasped,
            "peg_vertical": peg_vertical,
            "peg_vertical_alignment": self.peg_vertical_alignment(),
            "peg_lift": peg_lift,
            "head_lateral_dist": head_lateral_dist,
            "peg_lateral_dist": peg_lateral_dist,
            "peg_head_pos_at_hole": peg_head_pos_at_hole,
            "inserted": inserted,
        }
        return diagnostics

    def has_peg_inserted(self):
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        insertion_axis_flag = peg_head_pos_at_hole[:, 0] >= -0.01
        y_flag = torch.abs(peg_head_pos_at_hole[:, 1]) <= self.box_hole_radii
        z_flag = torch.abs(peg_head_pos_at_hole[:, 2]) <= self.box_hole_radii

        orientation_flag = self.is_peg_vertical(max_angle_deg=10)
        return (
            insertion_axis_flag & y_flag & z_flag & orientation_flag,
            peg_head_pos_at_hole,
        )

    def evaluate(self):
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        return {
            "success": success,
            "peg_head_pos_at_hole": peg_head_pos_at_hole,
        }

    def _get_obs_extra(self, info: dict):
        obs = {"tcp_pose": self.agent.tcp.pose.raw_pose}
        obs_mode_struct = getattr(self, "obs_mode_struct", None)
        use_state = getattr(obs_mode_struct, "use_state", False) or self._obs_mode in [
            "state",
            "state_dict",
        ]
        if use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_half_size=self.peg_half_sizes,
                box_hole_pose=self.box_hole_pose.raw_pose,
                box_hole_radius=self.box_hole_radii,
            )
        return obs

    def get_pi05_proprio(self):
        tcp_pose_in_root = self.agent.robot.pose.inv() * self.agent.tcp.pose
        tcp_transform = (
            tcp_pose_in_root.to_transformation_matrix().detach().cpu().numpy()
        )
        gripper = self.agent.robot.get_qpos().to(torch.float32)[:, -2:]
        state = aligned_pi05_state_from_tcp_matrices(
            tcp_transform, gripper.detach().cpu().numpy()
        )
        return torch.as_tensor(state, device=self.device, dtype=torch.float32)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        gripper_pos = self.agent.tcp.pose.p
        grasp_pose = self.peg.pose * sapien.Pose([-0.06, 0, 0])
        gripper_to_peg_dist = torch.linalg.norm(gripper_pos - grasp_pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)

        is_grasped = self.agent.is_grasping(self.peg, max_angle=30)
        reward = reaching_reward + is_grasped

        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
        head_lateral_dist = torch.linalg.norm(peg_head_wrt_goal.p[:, 1:], axis=1)
        peg_lateral_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)
        align_reward = 3 * (
            1
            - torch.tanh(
                0.5 * (head_lateral_dist + peg_lateral_dist)
                + 4.5 * torch.maximum(head_lateral_dist, peg_lateral_dist)
            )
        )
        reward += align_reward * is_grasped

        insertion_dist = torch.linalg.norm(
            (self.box_hole_pose.inv() * self.peg_head_pose).p, axis=1
        )
        reward += 5 * (1 - torch.tanh(5.0 * insertion_dist)) * is_grasped
        reward[info["success"]] = 10
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10
