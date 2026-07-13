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

"""Insert-only eval reset wrapper.

For PegInsertionVertical checkpoints trained on full pick-up-and-insert data,
the pick-up phase produces BC compound errors that corrupt the insert phase.
This wrapper initializes each eval episode with the peg already grasped and
lifted (the grasp + lift having been motion-planned by
``PegInsertionLiftPlanner``), so only move-and-insert is evaluated.

On every ``reset`` it queries the planner once per parallel environment,
stacks the per-env lifted states into batched arrays, and injects them as
``peg_pose`` / ``hole_pose`` / ``robot_qpos`` reset options together with
``pre_grasped: True``. The underlying ``ManiskillEnv`` merges its base
``reset_options`` underneath (caller wins), and
``PegInsertionVerticalEnv._initialize_episode`` applies the overrides and
skips its forced gripper-open line when ``pre_grasped`` is set. All other
methods (``step`` / ``chunk_step`` / ``capture_image`` / ``render``) are
passed through unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np


class PreGraspedInitWrapper(gym.Wrapper):
    """Reset into a motion-planned grasp+lift state for insert-only eval.

    Args:
        env: A batched ``ManiskillEnv`` (GPU). Must expose ``num_envs``.
        planner: Optional ``PegInsertionLiftPlanner``. Created lazily on the
            first reset if not supplied.
        seed: Base seed for per-env/per-episode planner seeds.
    """

    def __init__(self, env, *, planner: Any = None, seed: int = 0):
        super().__init__(env)
        self._planner = planner
        self._base_seed = int(seed)
        self._episode_counter = 0

    def _ensure_planner(self):
        if self._planner is None:
            # Imported lazily so the RoboFPE dependency is only pulled in when
            # insert-only eval is actually used.
            from rlinf.envs.maniskill.peg_insertion_lift_planner import (
                PegInsertionLiftPlanner,
            )

            self._planner = PegInsertionLiftPlanner()
        return self._planner

    def _planner_seed(self, env_index: int) -> int:
        # Derive a unique, deterministic seed per (episode, env) pair without
        # using time or RNG (which are unavailable/side-effectful here).
        return (
            self._base_seed * 1_000_003
            + self._episode_counter * 1_009
            + env_index * 97
            + 7
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        planner = self._ensure_planner()
        num_envs = int(self.num_envs)

        robot_qpos = np.zeros((num_envs, 9), dtype=np.float32)
        peg_pose = np.zeros((num_envs, 7), dtype=np.float32)
        hole_pose = np.zeros((num_envs, 7), dtype=np.float32)
        for i in range(num_envs):
            state = planner.plan_lifted_state(seed=self._planner_seed(i))
            robot_qpos[i] = state["robot_qpos"]
            peg_pose[i] = state["peg_pose"]
            hole_pose[i] = state["hole_pose"]
        self._episode_counter += 1

        merged_options: dict = dict(options) if options else {}
        merged_options.update(
            {
                "peg_pose": peg_pose,
                "hole_pose": hole_pose,
                "robot_qpos": robot_qpos,
                "pre_grasped": True,
            }
        )
        return self.env.reset(seed=seed, options=merged_options)
