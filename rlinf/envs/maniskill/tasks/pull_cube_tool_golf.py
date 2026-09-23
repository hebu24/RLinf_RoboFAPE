from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.structs.pose import Pose

from .pull_cube_block import NoTableSceneBuilder


@register_env("PullCubeTool-golf", max_episode_steps=100)
class PullCubeToolGolfEnv(BaseEnv):
    """RoboFPE golf-ball pull task, ported to RLinf's ManiSkill API."""

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: PandaWristCam | Panda | Fetch

    cube_half_size = 0.02
    handle_length = 0.25
    hook_length = 0.05
    width = 0.04
    height = 0.03
    cube_size = 0.02
    arm_reach = 0.35

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.85, -0.5, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _build_l_shaped_tool(self):
        builder = self.scene.create_actor_builder()
        material = sapien.render.RenderMaterial()
        material.set_base_color([1, 0, 0, 1])
        material.metallic = 1.0
        material.roughness = 0.0
        material.specular = 1.0
        builder.add_box_collision(sapien.Pose([self.handle_length / 2, 0, 0]), [self.handle_length / 2, self.width / 2, self.height / 2], density=500)
        builder.add_box_visual(sapien.Pose([self.handle_length / 2, 0, 0]), [self.handle_length / 2, self.width / 2, self.height / 2], material=material)
        builder.add_box_collision(sapien.Pose([self.handle_length - self.hook_length / 2, self.width, 0]), [self.hook_length / 2, self.width, self.height / 2])
        builder.add_box_visual(sapien.Pose([self.handle_length - self.hook_length / 2, self.width, 0]), [self.hook_length / 2, self.width, self.height / 2], material=material)
        return builder.build(name="l_shape_tool")

    def _load_scene(self, options: dict):
        self.scene_builder = NoTableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.scene_builder.build()
        self.replicaCAD_scene = ReplicaCADSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.replicaCAD_scene.build(1)
        builder = actors.get_actor_builder(self.scene, id="ycb:058_golf_ball")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cube = builder.build(name="cube")
        self.l_shape_tool = self._build_l_shaped_tool()

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6) for _ in range(3)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            tool_xyz = torch.zeros((b, 3), device=self.device)
            tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            tool_xyz[..., 2] = self.height / 2
            self.l_shape_tool.set_pose(Pose.create_from_pq(p=tool_xyz, q=torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)))
            cube_xyz = torch.zeros((b, 3), device=self.device)
            cube_xyz[..., 0] = self.arm_reach + torch.rand(b, device=self.device) * self.handle_length - 0.3
            cube_xyz[..., 1] = torch.rand(b, device=self.device) * 0.3 - 0.25
            cube_xyz[..., 2] = self.cube_size / 2 + 0.015
            q = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False,
                bounds=(-np.pi / 6, np.pi / 6), device=self.device,
            )
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz, q=q))

    def evaluate(self):
        cube_pos = self.cube.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p
        pulled_close = torch.linalg.norm(cube_pos[:, :2] - robot_base_pos[:, :2], dim=1) < 0.6
        workspace_center = robot_base_pos.clone()
        workspace_center[:, 0] += self.arm_reach * 0.1
        cube_distance = torch.linalg.norm(cube_pos - workspace_center, dim=1)
        return {"success": pulled_close, "success_once": pulled_close, "success_at_end": pulled_close,
                "cube_progress": (1 - torch.tanh(3.0 * cube_distance)).mean(), "cube_distance": cube_distance.mean(),
                "reward": self.compute_normalized_dense_reward(None, None, {"success": pulled_close})}

    def _get_obs_extra(self, info: dict):
        obs = {"tcp_pose": self.agent.tcp.pose.raw_pose}
        if self.obs_mode_struct.use_state:
            obs.update(cube_pose=self.cube.pose.raw_pose, tool_pose=self.l_shape_tool.pose.raw_pose)
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_pos, cube_pos, tool_pos = self.agent.tcp.pose.p, self.cube.pose.p, self.l_shape_tool.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p
        grasp_pos = tool_pos + torch.tensor([0.02, 0, 0], device=self.device)
        reward = 2.0 * (1 - torch.tanh(5 * torch.linalg.norm(tcp_pos - grasp_pos, dim=1)))
        grasping = self.agent.is_grasping(self.l_shape_tool, max_angle=20)
        reward += 2.0 * grasping
        ideal_hook = cube_pos + torch.tensor([-(self.hook_length + self.cube_half_size), -0.067, 0], device=self.device)
        positioned = torch.linalg.norm(tool_pos - ideal_hook, dim=1) < 0.05
        reward += 1.5 * (1 - torch.tanh(3 * torch.linalg.norm(tool_pos - ideal_hook, dim=1))) * grasping
        target = robot_base_pos + torch.tensor([0.05, 0, 0], device=self.device)
        initial = torch.linalg.norm(torch.tensor([self.arm_reach + 0.1, 0, self.cube_size / 2], device=self.device) - target)
        reward += 3.0 * (initial - torch.linalg.norm(cube_pos - target, dim=1)) / initial * positioned * grasping
        reward[cube_pos[:, 0] > self.arm_reach + 0.15] -= 2.0
        reward[info["success"]] += 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 5.0
