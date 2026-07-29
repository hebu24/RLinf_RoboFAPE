from pathlib import Path

from rlinf.runners.embodied_runner import prune_rolling_checkpoints


def _checkpoint(root: Path, step: int) -> Path:
    path = root / f"global_step_{step}_trainenvstep_{step * 100}"
    path.mkdir()
    return path


def test_prune_rolling_checkpoints_keeps_milestones_and_latest(tmp_path):
    for step in (50, 60, 70, 100, 110):
        _checkpoint(tmp_path, step)
    unrelated = tmp_path / "notes"
    unrelated.mkdir()

    removed = prune_rolling_checkpoints(str(tmp_path), 110, 50)

    assert {Path(path).name for path in removed} == {
        "global_step_60_trainenvstep_6000",
        "global_step_70_trainenvstep_7000",
    }
    assert {path.name for path in tmp_path.iterdir()} == {
        "global_step_50_trainenvstep_5000",
        "global_step_100_trainenvstep_10000",
        "global_step_110_trainenvstep_11000",
        "notes",
    }


def test_milestone_save_removes_previous_rolling_checkpoint(tmp_path):
    _checkpoint(tmp_path, 90)
    _checkpoint(tmp_path, 100)

    prune_rolling_checkpoints(str(tmp_path), 100, 50)

    assert {path.name for path in tmp_path.iterdir()} == {
        "global_step_100_trainenvstep_10000"
    }
