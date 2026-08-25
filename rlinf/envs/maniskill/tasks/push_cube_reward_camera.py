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

import numpy as np
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


@register_env("PushCubeRewardCamera-v1", max_episode_steps=200)
class PushCubeRewardCameraEnv(PushCubeEnv):
    """PushCube with a dedicated third-person reward camera in sensor_data."""

    @property
    def _default_sensor_configs(self):
        configs = list(super()._default_sensor_configs)
        if any(getattr(cfg, "uid", None) == "reward_camera" for cfg in configs):
            return configs

        reward_pose = sapien_utils.look_at(
            eye=np.array([0.35, -0.55, 0.45]),
            target=np.array([0.0, 0.0, 0.02]),
        )
        configs.append(
            CameraConfig("reward_camera", reward_pose, 320, 240, np.pi / 3, 0.01, 100)
        )
        return configs
