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

"""Owner wrapper for the insert-only eval lift planner.

For PegInsertionVertical checkpoints trained on full pick-up-and-insert data,
the pick-up phase produces BC compound errors that corrupt the insert phase.
The insert-only setting isolates move-and-insert: every episode starts with
the peg already grasped and lifted (grasp + lift motion-planned by
``PegInsertionLiftPlanner``), so only transport + align + descend + insert is
evaluated.

This wrapper does **not** override ``reset``. Auto-reset (``ManiskillEnv.
_handle_auto_reset -> self.reset``) bypasses outer wrappers, so overriding
``reset`` here would only fire on the first episode. Instead, the planner is
registered on the underlying ``PegInsertionVerticalEnv`` via
``set_lift_planner``; ``_initialize_episode`` (reached by BOTH the initial
reset and every auto-reset, since both go through ``self.env.reset ->
_initialize_episode``) plans and injects the grasped state when
``pre_grasped`` is set. This wrapper only owns the planner lifecycle.

All step / chunk_step / capture_image calls pass through unchanged.
"""

from __future__ import annotations

import gymnasium as gym


class PreGraspedInitWrapper(gym.Wrapper):
    """Own the lift planner and register it on the underlying peg task env.

    Args:
        env: A batched ``ManiskillEnv`` (GPU). ``env.env.unwrapped`` must be a
            ``PegInsertionVerticalEnv`` exposing ``set_lift_planner``.
        seed: Base seed for per-env/per-episode planner seeds.
    """

    def __init__(self, env, *, seed: int = 0):
        super().__init__(env)
        from rlinf.envs.maniskill.peg_insertion_lift_planner import (
            PegInsertionLiftPlanner,
        )

        self._planner = PegInsertionLiftPlanner(base_seed=int(seed))
        # Register on the underlying PegInsertionVerticalEnv so
        # _initialize_episode (initial reset + auto-reset) plans a grasped
        # state. self.env = ManiskillEnv; .env = gym env; .unwrapped = task.
        self.env.env.unwrapped.set_lift_planner(self._planner)

    def close(self):
        try:
            self._planner.close()
        finally:
            super().close()
