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
from mani_skill.utils.scene_builder.ai2thor import ArchitecTHORSceneBuilder
from mani_skill.utils.structs.pose import Pose


class _NoTableSceneBuilder:
    def __init__(self, env, robot_init_qpos_noise):
        self.env = env
        self.robot_init_qpos_noise = robot_init_qpos_noise

    def build(self):
        pass

    def initialize(self, env_idx):
        b = len(env_idx)
        if self.env.robot_uids in ("panda", "panda_wristcam"):
            qpos = np.array([0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4,
                             np.pi / 4 if self.env.robot_uids == "panda" else -np.pi / 4, 0.04, 0.04])
            qpos = self.env._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "fetch":
            qpos = np.array([0, 0, 0, 0.386, 0, 0, 0, -np.pi / 4, 0, np.pi / 4,
                             0, np.pi / 3, 0, 0.015, 0.015])
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, 0]))


@register_env("PickCube-ball", max_episode_steps=50)
class PickCubeBallEnv(BaseEnv):
    """RoboFPE PickCube-ball task with the YCB tennis ball."""

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: PandaWristCam | Panda | Fetch
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02,
                 camera_randomization_spec=None, render_randomization_spec=None,
                 obj_set=None, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.camera_randomization_spec = camera_randomization_spec
        self.render_randomization_spec = render_randomization_spec
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # Match RoboFPE collect_sft_data._sample_camera_spec(): external
        # positive-Y camera, target [0, 0, 0.25].
        pose = sapien_utils.look_at([0.65, 0.6, 0.65], [0.0, 0.0, 0.25])
        camera_spec = (self.camera_randomization_spec or {}).get("render")
        if camera_spec and camera_spec.get("pose"):
            values = np.asarray(camera_spec["pose"], dtype=np.float32)
            pose = sapien.Pose(p=values[:3], q=values[3:7])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = _NoTableSceneBuilder(self, self.robot_init_qpos_noise)
        self.table_scene.build()
        self.scene_builder = ArchitecTHORSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build([2] * self.num_envs)
        builder = actors.get_actor_builder(self.scene, id="ycb:056_tennis_ball")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cube = builder.build(name="cube")
        self.goal_site = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[0, 1, 0, 1],
            name="goal_site", body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: dict):
        obs = {"is_grasped": info["is_grasped"], "tcp_pose": self.agent.tcp.pose.raw_pose,
               "goal_pos": self.goal_site.pose.p}
        if self.obs_mode_struct.use_state:
            obs.update(obj_pose=self.cube.pose.raw_pose,
                       tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                       obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p)
        return obs

    def evaluate(self):
        placed = torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1) <= self.goal_thresh
        grasped = self.agent.is_grasping(self.cube)
        static = self.agent.is_static(0.2)
        return {"success": placed & static, "is_obj_placed": placed,
                "is_robot_static": static, "is_grasped": grasped}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reach = 1 - torch.tanh(5 * torch.linalg.norm(self.cube.pose.p - self.agent.tcp.pose.p, axis=1))
        reward = reach + info["is_grasped"]
        dist = torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
        reward += (1 - torch.tanh(5 * dist)) * info["is_grasped"]
        static = 1 - torch.tanh(5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1))
        reward += static * info["is_obj_placed"]
        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 5

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [0.4, 0.4, 0.4]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])
