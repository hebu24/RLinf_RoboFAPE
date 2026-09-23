from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat


class NoTableSceneBuilder(SceneBuilder):
    """Reset Panda in ReplicaCAD without adding ManiSkill's tabletop."""

    def build(self):
        self.scene_objects = []

    def initialize(self, env_idx: torch.Tensor):
        if self.env.robot_uids not in ("panda", "panda_wristcam"):
            raise NotImplementedError(self.env.robot_uids)
        qpos = np.array(
            [
                0.0,
                np.pi / 8,
                0,
                -np.pi * 5 / 8,
                0,
                np.pi * 3 / 4,
                -np.pi / 4 if self.env.robot_uids == "panda_wristcam" else np.pi / 4,
                0.04,
                0.04,
            ]
        )
        b = len(env_idx)
        if self.env._enhanced_determinism:
            qpos = self.env._batched_episode_rng[env_idx].normal(
                0, self.robot_init_qpos_noise, len(qpos)
            ) + qpos
        else:
            qpos = self.env._episode_rng.normal(
                0, self.robot_init_qpos_noise, (b, len(qpos))
            ) + qpos
        qpos[:, -2:] = 0.04
        self.env.agent.reset(qpos)
        self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))


@register_env("PullCube-block", max_episode_steps=50)
class PullCubeBlockEnv(BaseEnv):
    """RoboFPE PullCube-block task with the YCB wooden block."""

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: PandaWristCam | Panda | Fetch
    goal_radius = 0.1
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = look_at([0.7, -0.2, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = NoTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.replicaCAD_scene = ReplicaCADSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.replicaCAD_scene.build(0)
        builder = actors.get_actor_builder(self.scene, id="ycb:036_wood_block")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        self.obj = builder.build(name="obj")
        self.goal_region = actors.build_red_white_target(
            self.scene, radius=self.goal_radius, thickness=1e-5, name="goal_region",
            add_collision=False, body_type="kinematic"
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            self.obj.set_pose(Pose.create_from_pq(xyz, [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]))
            target = xyz - torch.tensor([0.1 + self.goal_radius, 0, 0], device=self.device)
            target[..., 2] = 1e-3
            self.goal_region.set_pose(Pose.create_from_pq(target, euler2quat(0, np.pi / 2, 0)))

    def evaluate(self):
        placed = torch.linalg.norm(self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1) < self.goal_radius
        return {"success": placed}

    def _get_obs_extra(self, info: dict):
        obs = {"tcp_pose": self.agent.tcp.pose.raw_pose, "goal_pos": self.goal_region.pose.p}
        if self.obs_mode_struct.use_state:
            obs["obj_pose"] = self.obj.pose.raw_pose
        return obs

    def compute_dense_reward(self, obs: Any, action, info: dict):
        pull_pos = self.obj.pose.p + torch.tensor([self.cube_half_size + 0.01, 0, 0], device=self.device)
        dist = torch.linalg.norm(pull_pos - self.agent.tcp.pose.p, axis=1)
        reward = 1 - torch.tanh(5 * dist)
        reached = dist < 0.01
        obj_dist = torch.linalg.norm(self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1)
        reward += (1 - torch.tanh(5 * obj_dist)) * reached
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 3
