from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.ai2thor import iTHORSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array
from transforms3d.euler import euler2quat

from .pull_cube_block import NoTableSceneBuilder


@register_env("LiftPegUpright-box", max_episode_steps=50)
class LiftPegUprightBoxEnv(BaseEnv):
    """Stand a YCB cracker box upright on the table."""

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: PandaWristCam | Panda | Fetch
    peg_half_width = 0.025
    box_upright_half_height = 0.108624

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = NoTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.ai2thor_scene = iTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.ai2thor_scene.build(0)
        builder = actors.get_actor_builder(self.scene, id="ycb:003_cracker_box")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        self.peg = builder.build(name="peg")

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [0.4, 0.4, 0.4]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.peg_half_width
            q = euler2quat(np.pi / 2, 0, np.pi / 2)
            self.peg.set_pose(Pose.create_from_pq(p=xyz, q=q))

    def evaluate(self):
        qmat = rotation_conversions.quaternion_to_matrix(self.peg.pose.q)
        longitudinal_axis = qmat[..., :, 2]
        upright = torch.abs(longitudinal_axis[..., 2]) > np.cos(0.1)
        at_table = torch.abs(self.peg.pose.p[:, 2] - self.box_upright_half_height) < 0.02
        return {"success": upright & at_table}

    def _get_obs_extra(self, info: dict):
        obs = {"tcp_pose": self.agent.tcp.pose.raw_pose}
        if self.obs_mode_struct.use_state:
            obs["obj_pose"] = self.peg.pose.raw_pose
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        qmat = rotation_conversions.quaternion_to_matrix(self.peg.pose.q)
        local_z = torch.tensor([0.0, 0, 1.0], device=self.device)
        rot_rew = torch.abs((qmat @ local_z).view(-1, 3) @ local_z)
        z_dist = torch.abs(self.peg.pose.p[:, 2] - self.box_upright_half_height)
        reward = rot_rew + 1 - torch.tanh(5 * z_dist)
        to_grip_dist = torch.linalg.norm(self.peg.pose.p - self.agent.tcp.pose.p, axis=1)
        reaching = 1 - torch.tanh(5 * to_grip_dist)
        reaching[self.agent.is_grasping(self.peg)] = 1
        reward += reaching / 5
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs, action, info) / 3.0
