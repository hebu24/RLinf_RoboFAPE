import numpy as np
import torch
from omegaconf import OmegaConf

import rlinf.workers.env.env_worker as env_worker_module
from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult, EnvOutput
from rlinf.models.embodiment.reward.robometer_reward_model import RobometerEpisodeReward
from rlinf.workers.env.env_worker import EnvWorker


def _assignment(values) -> RobometerEpisodeReward:
    chunk_reward = np.asarray(values, dtype=np.float32)
    return RobometerEpisodeReward(
        downsample_indices=list(range(chunk_reward.size)),
        per_step_progress=np.linspace(0.1, 0.9, chunk_reward.size, dtype=np.float32),
        per_step_reward=chunk_reward.reshape(-1),
        per_step_loss_mask=np.ones(chunk_reward.size, dtype=bool),
        chunk_reward=chunk_reward,
        chunk_loss_mask=np.ones_like(chunk_reward, dtype=bool),
    )


def _build_worker(shaping: str) -> EnvWorker:
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "reward": {"model": {"max_robometer_frames": 60, "fail_shift": 1.0}},
        }
    )
    worker.reward_weight = 1.0
    worker.reward_shaping = shaping
    worker.delta_success_bonus = 0.1
    worker.use_completed_episode_buffer = False
    worker.reward_mode = "history_buffer"
    worker.enable_train = True
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 1
    worker._episode_chunk_ids = [torch.zeros(1, dtype=torch.long)]
    worker._episode_chunk_counts = [torch.zeros(1, dtype=torch.long)]
    worker._window_chunk_refs = [[]]
    worker.rollout_results = [EmbodiedRolloutResult(max_episode_length=8)]
    worker._last_done_buffer_info = {0: {}}
    return worker


def _append_window_chunk(worker: EnvWorker, done: bool) -> None:
    result = ChunkStepResult(
        rewards=torch.zeros((1, 2), dtype=torch.float32),
        dones=torch.tensor([[False, done]], dtype=torch.bool),
        terminations=torch.tensor([[False, done]], dtype=torch.bool),
        truncations=torch.zeros((1, 2), dtype=torch.bool),
    )
    worker.rollout_results[0].append_step_result(result)
    worker._record_window_chunk_ref(
        0,
        EnvOutput(
            obs={"states": torch.zeros((1, 1), dtype=torch.float32)},
            dones=result.dones,
            terminations=result.terminations,
            truncations=result.truncations,
        ),
        has_rewards=True,
    )


def test_record_window_chunk_ref_resets_episode_index_after_done():
    worker = _build_worker("absolute")

    _append_window_chunk(worker, done=True)
    _append_window_chunk(worker, done=False)

    refs = worker._window_chunk_refs[0]
    assert refs[0][0].episode_id == 0
    assert refs[0][0].chunk_index == 0
    assert refs[1][0].episode_id == 1
    assert refs[1][0].chunk_index == 0


def test_assign_history_reward_absolute_backfills_only_current_window(monkeypatch):
    worker = _build_worker("absolute")
    worker._window_chunk_refs = [[
        [env_worker_module.WindowChunkRef(episode_id=7, chunk_index=2)],
        [env_worker_module.WindowChunkRef(episode_id=7, chunk_index=3)],
    ]]
    for _ in range(2):
        worker.rollout_results[0].append_step_result(
            ChunkStepResult(rewards=torch.zeros((1, 2), dtype=torch.float32))
        )
    worker._last_done_buffer_info[0][0] = (7, 8, 0, [False] * 8)

    monkeypatch.setattr(
        env_worker_module,
        "reconstruct_robometer_episode_reward",
        lambda *args, **kwargs: _assignment(
            [[0.1, 0.2], [0.3, 0.4], [1.1, 1.2], [2.1, 2.2]]
        ),
    )

    worker.assign_history_reward(0, torch.ones((1, 8), dtype=torch.float32))

    torch.testing.assert_close(
        worker.rollout_results[0].rewards[0][0], torch.tensor([1.1, 1.2])
    )
    torch.testing.assert_close(
        worker.rollout_results[0].rewards[1][0], torch.tensor([2.1, 2.2])
    )
    assert worker.rollout_results[0].loss_mask[0][0].all()
    assert worker.rollout_results[0].loss_mask[1][0].all()


def test_assign_history_reward_delta_backfills_only_current_window(monkeypatch):
    worker = _build_worker("delta")
    worker._window_chunk_refs = [[
        [env_worker_module.WindowChunkRef(episode_id=3, chunk_index=1)],
        [env_worker_module.WindowChunkRef(episode_id=3, chunk_index=2)],
    ]]
    for _ in range(2):
        worker.rollout_results[0].append_step_result(
            ChunkStepResult(rewards=torch.zeros((1, 2), dtype=torch.float32))
        )
    worker._last_done_buffer_info[0][0] = (3, 6, 0, [False] * 6)

    monkeypatch.setattr(
        env_worker_module,
        "reconstruct_robometer_delta_reward",
        lambda *args, **kwargs: _assignment(
            [[0.2, 0.3], [1.2, 1.3], [2.2, 2.3]]
        ),
    )

    worker.assign_history_reward(0, torch.ones((1, 4), dtype=torch.float32))

    torch.testing.assert_close(
        worker.rollout_results[0].rewards[0][0], torch.tensor([1.2, 1.3])
    )
    torch.testing.assert_close(
        worker.rollout_results[0].rewards[1][0], torch.tensor([2.2, 2.3])
    )


def test_window_chunk_refs_ignore_steps_without_rewards(monkeypatch):
    worker = _build_worker("absolute")
    worker._record_window_chunk_ref(
        0,
        EnvOutput(
            obs={"states": torch.zeros((1, 1), dtype=torch.float32)},
            dones=torch.tensor([[False, False]], dtype=torch.bool),
            terminations=torch.tensor([[False, False]], dtype=torch.bool),
            truncations=torch.tensor([[False, False]], dtype=torch.bool),
        ),
        has_rewards=False,
    )
    worker.rollout_results[0].append_step_result(
        ChunkStepResult(rewards=torch.zeros((1, 2), dtype=torch.float32))
    )
    worker._record_window_chunk_ref(
        0,
        EnvOutput(
            obs={"states": torch.zeros((1, 1), dtype=torch.float32)},
            dones=torch.tensor([[False, True]], dtype=torch.bool),
            terminations=torch.tensor([[False, True]], dtype=torch.bool),
            truncations=torch.tensor([[False, False]], dtype=torch.bool),
        ),
        has_rewards=True,
    )
    worker._last_done_buffer_info[0][0] = (0, 2, 0, [False, False])

    monkeypatch.setattr(
        env_worker_module,
        "reconstruct_robometer_episode_reward",
        lambda *args, **kwargs: _assignment([[0.4, 0.5]]),
    )

    worker.assign_history_reward(0, torch.ones((1, 2), dtype=torch.float32))

    assert len(worker._window_chunk_refs[0]) == len(worker.rollout_results[0].rewards)
    torch.testing.assert_close(
        worker.rollout_results[0].rewards[0][0], torch.tensor([0.4, 0.5])
    )


def test_get_reward_model_output_queries_unfinished_prefixes_only_at_window_end(
    monkeypatch,
):
    worker = _build_worker("absolute")
    worker.use_completed_episode_buffer = False
    worker.train_history_managers = [type("HM", (), {})()]
    history_manager = worker.train_history_managers[0]
    history_manager.history_entries = [[{"render_images": "a"}], [{"render_images": "b"}]]
    history_manager.pickup_counts = [0, 0]
    history_manager.success_history_entries = [[False], [False]]
    history_manager.build_history_input = lambda dones, emit_mask: (
        {
            "render_buffer": {
                "render_images": [
                    [np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
                    [np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
                ]
            }
        },
        {"render_buffer": [2, 2]},
    )
    worker._episode_chunk_ids = [torch.tensor([4, 9], dtype=torch.long)]
    worker._episode_chunk_counts = [torch.tensor([1, 3], dtype=torch.long)]
    worker.send_to = lambda **kwargs: None
    monkeypatch.setattr(
        worker,
        "recv_from",
        lambda **kwargs: torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
    )

    reward = worker.get_reward_model_output(
        EnvOutput(
            obs={"states": torch.zeros((2, 1), dtype=torch.float32)},
            final_obs={"states": torch.zeros((2, 1), dtype=torch.float32)},
            env_infos={"success": np.array([False, False])},
            dones=torch.tensor([False, False], dtype=torch.bool),
        ),
        send_channel=None,
        recv_channel=None,
        stage_id=0,
        last_run=False,
    )

    assert reward is None

    reward = worker.get_reward_model_output(
        EnvOutput(
            obs={"states": torch.zeros((2, 1), dtype=torch.float32)},
            final_obs={"states": torch.zeros((2, 1), dtype=torch.float32)},
            env_infos={"success": np.array([False, False])},
            dones=torch.tensor([False, False], dtype=torch.bool),
        ),
        send_channel=None,
        recv_channel=None,
        stage_id=0,
        last_run=True,
    )

    torch.testing.assert_close(
        reward, torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    )
    assert worker._last_history_query_info[0][0][0] == 4
    assert worker._last_history_query_info[0][1][0] == 9
