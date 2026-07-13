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

"""Motion-plan the grasp + lift phase for insert-only PegInsertion eval.

The wrist SFT policy is trained on full pick-up-and-insert demonstrations.
To isolate the move-and-insert skill, insert-only evaluation starts every
episode with the peg already grasped and lifted. Instead of hand-constructing
a grasp pose (fragile, and ManiSkill ships no IK), this planner reuses the
existing RoboFPE motion planner -- the same one used by
``collect_peg_insertion_controller_data.py`` to generate SFT data -- to solve
grasp + lift on a warm CPU single-environment, then captures the lifted state
``(robot_qpos[9], peg_pose[7], hole_pose[7])`` at the end of the lift stage.

The captured state is self-consistent (the peg is genuinely grasped in the CPU
sim), so replaying it kinematically into the GPU eval environment reproduces a
grasped peg. The eval wrapper stacks per-env states into batched arrays and
passes them as ``peg_pose`` / ``hole_pose`` / ``robot_qpos`` reset options,
which ``PegInsertionVerticalEnv._initialize_episode`` already honors.

Process isolation
-----------------
SAPIEN/PhysX cannot host a CPU-sim environment and a GPU-sim environment in the
same process ("GPU PhysX can only be enabled once"; "single-GPU rendering"). The
eval env is GPU-batched, so this planner runs in a **separate persistent
subprocess** that owns the warm CPU env. The parent (a thin proxy used by the
eval wrapper) sends a seed over stdin and reads the lifted state back as one
JSON line. One subprocess is spawned per eval env worker and reused across all
episodes, so env construction is paid once.

Per-episode cost is one CPU solve (~seconds). A precomputed grasped-state pool
is a documented future optimization for large sweeps.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types as _types
from typing import Any


# The RoboFPE solver lives outside this repo. Mirror the collector's sys.path
# setup and the ``tasks`` module shim so that importing
# ``solutions.solve_PegInsertionVertical`` does not re-register a second
# PegInsertionVertical environment implementation.
ROBOFPE = os.environ.get("RLINF_ROBOFPE_PATH", "/home/gpu4/yingxi/RoboFPE/mani_envs")


# --------------------------------------------------------------------------- #
# Subprocess worker (runs in the child process that owns the CPU env).
# --------------------------------------------------------------------------- #
def _import_solver() -> Any:
    """Import the RoboFPE solver (idempotent within the worker process)."""
    if ROBOFPE not in sys.path:
        sys.path.insert(0, ROBOFPE)
    # The solver imports ``tasks.task_PegInsertionVertical`` for type hints.
    # Provide a dummy so that import does not register another env impl.
    if "tasks" not in sys.modules:
        tasks_module = _types.ModuleType("tasks")
        peg_module = _types.ModuleType("tasks.task_PegInsertionVertical")
        peg_module.PegInsertionVerticalEnv = type("PegInsertionVerticalEnv", (), {})
        tasks_module.task_PegInsertionVertical = peg_module
        sys.modules["tasks"] = tasks_module
        sys.modules["tasks.task_PegInsertionVertical"] = peg_module
    from solutions.solve_PegInsertionVertical import (  # type: ignore[import-not-found]
        solve_peginsertionvertical,
    )

    return solve_peginsertionvertical


def _as_numpy(value):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class _LiftStateRecorder:
    """Record per-step (robot_qpos, peg_pose, hole_pose) during a solver run.

    Mirrors ``ReferenceRecorder`` from the collector but captures the full
    9-dim robot qpos (arm7 + 2 fingers) and the peg/box world poses, which is
    what the GPU eval env needs to reproduce the grasped state.
    """

    def __init__(self, env):
        self.env = env
        self.records: list[dict] = []
        self._orig_step = None

    def start(self) -> None:
        self.records = []
        recorder = self

        def _step(action, **kwargs):
            obs, reward, terminated, truncated, info = recorder._orig_step(
                action, **kwargs
            )
            recorder.records.append(recorder._capture())
            return obs, reward, terminated, truncated, info

        self._orig_step = self.env.step
        self.env.step = _step

    def stop(self) -> None:
        if self._orig_step is not None:
            self.env.step = self._orig_step
            self._orig_step = None

    def _capture(self) -> dict:
        import numpy as np

        unwrapped = self.env.unwrapped
        robot_qpos = _as_numpy(unwrapped.agent.robot.get_qpos())[0].astype(np.float32)
        peg_p = _as_numpy(unwrapped.peg.pose.p)[0].astype(np.float32)
        peg_q = _as_numpy(unwrapped.peg.pose.q)[0].astype(np.float32)
        box_p = _as_numpy(unwrapped.box.pose.p)[0].astype(np.float32)
        box_q = _as_numpy(unwrapped.box.pose.q)[0].astype(np.float32)
        return {
            "robot_qpos": robot_qpos,
            "peg_pose": np.concatenate([peg_p, peg_q]).astype(np.float32),
            "hole_pose": np.concatenate([box_p, box_q]).astype(np.float32),
        }


# Lift-end detection thresholds (metres).
LIFT_HEIGHT_ABOVE_TABLE = 0.05
PICKUP_XY_TOLERANCE = 0.03
MAX_RETRIES = 4


def _select_lift_end(records, table_top_z, peg_half_length):
    """Pick the index of the lift-end frame.

    A frame qualifies as "lifted but not yet transported" when the peg is
    grasped (closed fingers), raised off the table, and its xy is still close
    to where it was picked up (i.e. before the pre_insert transport). Returns
    the last qualifying frame; falls back to the last grasped-and-raised frame.
    """
    import numpy as np

    if not records:
        return None
    resting_z = table_top_z + peg_half_length
    lift_z = resting_z + LIFT_HEIGHT_ABOVE_TABLE

    pickup_xy = None
    for rec in records:
        if rec["robot_qpos"][-2:].max() < 0.04 and rec["peg_pose"][2] > lift_z:
            pickup_xy = rec["peg_pose"][:2].copy()
            break

    last_qualifying = None
    last_raised_grasped = None
    for i, rec in enumerate(records):
        grasped = bool(rec["robot_qpos"][-2:].max() < 0.04)
        raised = rec["peg_pose"][2] > lift_z
        if grasped and raised:
            last_raised_grasped = i
            if pickup_xy is not None:
                xy = rec["peg_pose"][:2]
                if float(np.linalg.norm(xy - pickup_xy)) < PICKUP_XY_TOLERANCE:
                    last_qualifying = i
    if last_qualifying is not None:
        return last_qualifying
    return last_raised_grasped


def planner_worker_main() -> None:
    """Child-process entry point: own a warm CPU env and serve lifted states.

    Reads JSON requests ``{"id": <int>, "seed": <int>}`` from stdin (one per
    line) and writes JSON responses ``{"id": <int>, "state": {...}}`` (or
    ``{"id": <int>, "error": "..."}``) to stdout. Arrays are serialized as
    lists.
    """
    import gymnasium as gym
    import numpy as np  # noqa: F401  (ensure numpy is importable in worker)

    solve = _import_solver()
    render_backend = os.environ.get("RLINF_PLANNER_RENDER_BACKEND", "none")
    env = gym.make(
        "PegInsertionVertical-v1",
        num_envs=1,
        obs_mode="none",
        robot_uids="panda_wristcam",
        control_mode="pd_joint_pos",
        sim_backend="cpu",
        render_backend=render_backend,
        reward_mode="normalized_dense",
        max_episode_steps=600,
    )
    unwrapped = env.unwrapped
    table_top_z = float(unwrapped.table_top_z)
    peg_half_length = float(unwrapped.peg_half_length)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            seed = int(req["seed"])
            rid = req.get("id")
        except Exception as exc:  # noqa: BLE001
            sys.stdout.write(json.dumps({"error": f"bad request: {exc}"}) + "\n")
            sys.stdout.flush()
            continue
        try:
            state = None
            last_err = None
            for attempt in range(MAX_RETRIES):
                attempt_seed = seed + attempt
                recorder = _LiftStateRecorder(env)
                try:
                    recorder.start()
                    result = solve(
                        env,
                        seed=attempt_seed,
                        debug=False,
                        vis=False,
                        reset_options={"randomize_initial_poses": True},
                    )
                finally:
                    recorder.stop()

                if isinstance(result, int) and result == -1:
                    continue
                idx = _select_lift_end(recorder.records, table_top_z, peg_half_length)
                if idx is None:
                    continue
                rec = recorder.records[idx]
                state = {
                    "robot_qpos": rec["robot_qpos"].tolist(),
                    "peg_pose": rec["peg_pose"].tolist(),
                    "hole_pose": rec["hole_pose"].tolist(),
                }
                break
            if state is None:
                raise RuntimeError(
                    f"no lifted state for seed={seed}"
                    + (f" after {MAX_RETRIES} retries: {last_err}" if last_err else "")
                )
            sys.stdout.write(json.dumps({"id": rid, "state": state}) + "\n")
        except Exception as exc:  # noqa: BLE001
            sys.stdout.write(json.dumps({"id": rid, "error": str(exc)}) + "\n")
        sys.stdout.flush()

    try:
        env.close()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Parent-side proxy used by the eval wrapper.
# --------------------------------------------------------------------------- #
class PegInsertionLiftPlanner:
    """Parent-side proxy that fetches lifted states from a worker subprocess.

    The subprocess owns the warm CPU env (PhysX/rendering are process-global,
    so it cannot live in the GPU eval worker's process). One subprocess is
    spawned per proxy and reused across all episodes.
    """

    def __init__(self, *, python_bin: str | None = None):
        self._python_bin = python_bin or sys.executable
        self._proc: subprocess.Popen | None = None
        self._req_id = 0

    def _ensure_proc(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        env = os.environ.copy()
        # CPU-only renderer keeps the worker off the GPU the eval env uses.
        env.setdefault("RLINF_PLANNER_RENDER_BACKEND", "none")
        self._proc = subprocess.Popen(
            [
                self._python_bin,
                "-c",
                "from rlinf.envs.maniskill.peg_insertion_lift_planner import "
                "planner_worker_main; planner_worker_main()",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._req_id = 0
        return self._proc

    def plan_lifted_state(self, seed: int) -> dict:
        """Return the lifted state ``{robot_qpos, peg_pose, hole_pose}`` for one episode."""
        import numpy as np

        proc = self._ensure_proc()
        self._req_id += 1
        rid = self._req_id
        proc.stdin.write(json.dumps({"id": rid, "seed": int(seed)}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(
                f"PegInsertionLiftPlanner worker exited unexpectedly (seed={seed}). stderr: {err}"
            )
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"PegInsertionLiftPlanner error: {resp['error']}")
        state = resp["state"]
        return {
            "robot_qpos": np.asarray(state["robot_qpos"], dtype=np.float32),
            "peg_pose": np.asarray(state["peg_pose"], dtype=np.float32),
            "hole_pose": np.asarray(state["hole_pose"], dtype=np.float32),
        }

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.stdin.close()
                self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self._proc.kill()
        finally:
            self._proc = None
