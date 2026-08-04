import json
import queue
from unittest.mock import MagicMock

import pytest
import numpy as np

from rlinf.envs.maniskill import peg_insertion_lift_planner as planner_module
from rlinf.envs.maniskill.peg_insertion_lift_planner import (
    RESP_PREFIX,
    PegInsertionLiftPlanner,
)


def _planner_with_response(response):
    planner = PegInsertionLiftPlanner()
    proc = MagicMock()
    proc.stdin = MagicMock()
    planner._proc = proc
    planner._stdout_queue = queue.Queue()
    planner._stdout_queue.put(response)
    planner._ensure_proc = MagicMock(return_value=proc)
    return planner


def test_plan_lifted_state_reads_background_stdout_queue():
    state = {
        "robot_qpos": [0.0] * 9,
        "peg_pose": [0.0] * 7,
        "hole_pose": [0.0] * 7,
        "trajectory": [],
    }
    response = RESP_PREFIX + json.dumps({"id": 1, "state": state}) + "\n"
    planner = _planner_with_response(response)

    result = planner.plan_lifted_state(seed=7)

    assert result["robot_qpos"].shape == (9,)
    planner._proc.stdin.write.assert_called_once()


def test_plan_lifted_state_times_out_instead_of_blocking(monkeypatch):
    planner = _planner_with_response(None)
    planner._stdout_queue = queue.Queue()
    monkeypatch.setattr(planner_module, "PLANNER_REQUEST_TIMEOUT_S", 0.01)

    with pytest.raises(RuntimeError, match="timed out after 0.01s"):
        planner.plan_lifted_state(seed=11)


def test_shared_reset_seed_is_identical_for_all_envs_and_episodes(monkeypatch):
    planner = PegInsertionLiftPlanner(base_seed=0, shared_reset_seed=True)
    requested_seeds = []

    def fake_plan(seed):
        requested_seeds.append(seed)
        return {
            "robot_qpos": np.zeros(9, dtype=np.float32),
            "peg_pose": np.zeros(7, dtype=np.float32),
            "hole_pose": np.zeros(7, dtype=np.float32),
            "trajectory": [],
        }

    monkeypatch.setattr(planner, "plan_lifted_state", fake_plan)
    planner.plan_lifted_states([0, 3, 7])
    planner.plan_lifted_states([2])

    assert requested_seeds == [0, 0, 0, 0]
