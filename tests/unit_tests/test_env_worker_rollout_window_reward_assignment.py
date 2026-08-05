from collections import defaultdict

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import rlinf.workers.env.env_worker as env_worker_module
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
)
from rlinf.models.embodiment.reward.robometer_reward_model import RobometerEpisodeReward
from rlinf.utils.utils import masked_mean, masked_mean_ratio
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
            "reward": {
                "group_name": "RewardGroup",
                "model": {"max_robometer_frames": 60, "fail_shift": 1.0},
            },
        }
    )
    worker.reward_weight = 1.0
    worker.reward_shaping = shaping
    worker.delta_success_bonus = 0.1
    worker.delta_failure_terminal_penalty = 0.0
    worker.model_cfg = OmegaConf.create({"num_action_chunks": 2})
    worker.use_completed_episode_buffer = False
    worker.reward_mode = "history_buffer"
    worker.history_reward_assign = True
    worker.debug_chunk_funnel = False
    worker.fail_fast_on_unexpected_chunk_filter = False
    worker.env_infos_reward_keys = ("success", "episode", "final_info")
    worker.enable_train = True
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 1
    worker.train_batch_size = 1
    worker._episode_chunk_ids = [torch.zeros(1, dtype=torch.long)]
    worker._episode_chunk_counts = [torch.zeros(1, dtype=torch.long)]
    worker._window_chunk_refs = [[]]
    worker._window_chunk_funnel = [worker._new_chunk_funnel_state()]
    worker.history_lengths = [{}]
    worker.rollout_results = [EmbodiedRolloutResult(max_episode_length=8)]
    worker._last_history_query_info = {0: {}}
    worker.log_info = lambda *args, **kwargs: None
    worker._accelerator_type = "cpu"
    worker._timer_metrics = {}
    worker.env_list = [
        type("EnvStub", (), {"consume_pickup_frames": lambda self: {}})()
    ]
    worker.env_decoupled_mode = False
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
    worker._window_chunk_refs = [
        [
            [env_worker_module.WindowChunkRef(episode_id=7, chunk_index=2)],
            [env_worker_module.WindowChunkRef(episode_id=7, chunk_index=3)],
        ]
    ]
    for _ in range(2):
        worker.rollout_results[0].append_step_result(
            ChunkStepResult(rewards=torch.zeros((1, 2), dtype=torch.float32))
        )
    worker._last_history_query_info[0][0] = (7, 8, 0, [False] * 8)

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
    worker._window_chunk_refs = [
        [
            [env_worker_module.WindowChunkRef(episode_id=3, chunk_index=1)],
            [env_worker_module.WindowChunkRef(episode_id=3, chunk_index=2)],
        ]
    ]
    for _ in range(2):
        worker.rollout_results[0].append_step_result(
            ChunkStepResult(rewards=torch.zeros((1, 2), dtype=torch.float32))
        )
    worker._last_history_query_info[0][0] = (3, 6, 0, [False] * 6)

    monkeypatch.setattr(
        env_worker_module,
        "reconstruct_robometer_delta_reward",
        lambda *args, **kwargs: _assignment([[0.2, 0.3], [1.2, 1.3], [2.2, 2.3]]),
    )

    worker.assign_history_reward(0, torch.ones((1, 4), dtype=torch.float32))

    torch.testing.assert_close(
        worker.rollout_results[0].rewards[0][0], torch.tensor([1.2, 1.3])
    )
    torch.testing.assert_close(
        worker.rollout_results[0].rewards[1][0], torch.tensor([2.2, 2.3])
    )


def test_assign_history_reward_delta_passes_failure_terminal_penalty(monkeypatch):
    worker = _build_worker("delta")
    worker.delta_failure_terminal_penalty = -0.4
    worker._window_chunk_refs = [
        [
            [env_worker_module.WindowChunkRef(episode_id=3, chunk_index=0)],
        ]
    ]
    worker.rollout_results[0].append_step_result(
        ChunkStepResult(rewards=torch.zeros((1, 2), dtype=torch.float32))
    )
    worker._last_history_query_info[0][0] = (3, 2, 0, [False, False])
    captured = {}

    def _reconstruct(*args, **kwargs):
        captured.update(kwargs)
        return _assignment([[0.2, 0.3]])

    monkeypatch.setattr(
        env_worker_module, "reconstruct_robometer_delta_reward", _reconstruct
    )
    worker.assign_history_reward(0, torch.ones((1, 2), dtype=torch.float32))

    assert captured["failure_terminal_penalty"] == pytest.approx(-0.4)


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
    worker._last_history_query_info[0][0] = (0, 2, 0, [False, False])

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


def test_history_reward_placeholder_records_unlabeled_action_chunk():
    worker = _build_worker("absolute")
    env_output = EnvOutput(
        obs={"states": torch.zeros((1, 1), dtype=torch.float32)},
        dones=torch.tensor([[False, False]], dtype=torch.bool),
        terminations=torch.tensor([[False, False]], dtype=torch.bool),
        truncations=torch.tensor([[False, False]], dtype=torch.bool),
        rewards=None,
    )
    rollout_result = RolloutResult(
        forward_inputs={"action": torch.zeros((1, 14), dtype=torch.float32)}
    )

    rewards = worker.compute_bootstrap_rewards(env_output, None, None)
    rewards = worker._history_reward_placeholder(rewards, rollout_result)
    worker.rollout_results[0].append_step_result(
        ChunkStepResult(
            actions=rollout_result.forward_inputs["action"],
            forward_inputs=rollout_result.forward_inputs,
            dones=env_output.dones,
            terminations=env_output.terminations,
            truncations=env_output.truncations,
            rewards=rewards,
        )
    )
    worker._record_window_chunk_ref(0, env_output, has_rewards=rewards is not None)

    assert len(worker.rollout_results[0].rewards) == 1
    assert len(worker._window_chunk_refs[0]) == 1
    assert worker._window_chunk_refs[0][0][0].episode_id == 0
    assert worker._window_chunk_refs[0][0][0].chunk_index == 0
    torch.testing.assert_close(
        worker.rollout_results[0].rewards[0], torch.zeros((1, 2))
    )
    assert not bool(worker.rollout_results[0].loss_mask[0].any())


def test_count_masked_in_chunks_accepts_rollout_loss_mask_list():
    mask = [
        torch.tensor([[True, True], [True, False]], dtype=torch.bool),
        torch.tensor([[False, False], [True, True]], dtype=torch.bool),
    ]

    assert EnvWorker._count_masked_in_chunks(mask) == 2


def test_get_reward_model_output_queries_finished_prefixes_before_window_end(
    monkeypatch,
):
    worker = _build_worker("absolute")
    worker.train_num_envs_per_stage = 2
    worker.use_completed_episode_buffer = False
    worker.train_history_managers = [type("HM", (), {})()]
    history_manager = worker.train_history_managers[0]
    history_manager.history_entries = [
        [{"render_images": "a"}],
        [{"render_images": "b"}],
    ]
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
    worker.send_to = lambda **kwargs: None
    worker._window_chunk_refs = [
        [
            [
                env_worker_module.WindowChunkRef(episode_id=4, chunk_index=0),
                env_worker_module.WindowChunkRef(episode_id=9, chunk_index=0),
            ]
        ]
    ]
    worker.rollout_results[0].append_step_result(
        ChunkStepResult(rewards=torch.zeros((2, 2), dtype=torch.float32))
    )

    reward = worker.get_reward_model_output(
        EnvOutput(
            obs={"states": torch.zeros((2, 1), dtype=torch.float32)},
            final_obs={"states": torch.zeros((2, 1), dtype=torch.float32)},
            env_infos={"success": np.array([False, False])},
            dones=torch.tensor([True, False], dtype=torch.bool),
        ),
        send_channel=None,
        recv_channel=None,
        stage_id=0,
        last_run=False,
    )

    torch.testing.assert_close(
        reward, torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    )
    assert worker._last_history_query_info[0][0][0] == 4
    assert worker._last_history_query_info[0][1][0] == 9


def test_get_reward_model_output_queries_unfinished_prefixes_at_window_end(
    monkeypatch,
):
    worker = _build_worker("absolute")
    worker.train_num_envs_per_stage = 2
    worker.use_completed_episode_buffer = False
    worker.train_history_managers = [type("HM", (), {})()]
    history_manager = worker.train_history_managers[0]
    history_manager.history_entries = [
        [{"render_images": "a"}],
        [{"render_images": "b"}],
    ]
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
    worker._window_chunk_refs = [
        [
            [
                env_worker_module.WindowChunkRef(episode_id=4, chunk_index=0),
                env_worker_module.WindowChunkRef(episode_id=9, chunk_index=0),
            ]
        ]
    ]
    worker.rollout_results[0].append_step_result(
        ChunkStepResult(rewards=torch.zeros((2, 2), dtype=torch.float32))
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
        last_run=True,
    )

    torch.testing.assert_close(
        reward, torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    )
    assert worker._last_history_query_info[0][0][0] == 4
    assert worker._last_history_query_info[0][1][0] == 9


def test_get_reward_model_output_skips_autoreset_tail_without_window_chunk(
    monkeypatch,
):
    worker = _build_worker("delta")
    worker.train_num_envs_per_stage = 2
    worker.fail_fast_on_unexpected_chunk_filter = True
    worker.train_history_managers = [type("HM", (), {})()]
    history_manager = worker.train_history_managers[0]
    history_manager.history_entries = [
        [{"render_images": "pickup_or_tail"}] * 39,
        [{"render_images": "trainable"}] * 49,
    ]
    history_manager.pickup_counts = [29, 29]
    history_manager.success_history_entries = [[False] * 39, [False] * 49]

    def _build_history_input(dones, emit_mask):
        assert emit_mask.tolist() == [False, True]
        return (
            {
                "render_buffer": {
                    "render_images": [
                        [],
                        [np.zeros((2, 2, 3), dtype=np.uint8)] * 49,
                    ]
                }
            },
            {"render_buffer": [0, 49]},
        )

    history_manager.build_history_input = _build_history_input
    worker._episode_chunk_ids = [torch.tensor([1, 0], dtype=torch.long)]
    worker._episode_chunk_counts = [torch.tensor([0, 1], dtype=torch.long)]
    worker._window_chunk_refs = [
        [
            [
                env_worker_module.WindowChunkRef(episode_id=0, chunk_index=0),
                env_worker_module.WindowChunkRef(episode_id=0, chunk_index=0),
            ]
        ]
    ]
    worker.rollout_results[0].append_step_result(
        ChunkStepResult(rewards=torch.zeros((2, 2), dtype=torch.float32))
    )
    worker.send_to = lambda **kwargs: None
    monkeypatch.setattr(
        worker,
        "recv_from",
        lambda **kwargs: torch.tensor([[0.0, 0.0], [0.1, 0.2]], dtype=torch.float32),
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
        last_run=True,
    )

    torch.testing.assert_close(
        reward, torch.tensor([[0.0, 0.0], [0.1, 0.2]], dtype=torch.float32)
    )
    assert set(worker._last_history_query_info[0]) == {1}
    assert worker._window_chunk_funnel[0]["drop_reasons"]["outside_current_window"] == 0


def test_assign_history_reward_fail_fast_on_mask_false(monkeypatch):
    worker = _build_worker("absolute")
    worker.fail_fast_on_unexpected_chunk_filter = True
    worker._window_chunk_refs = [
        [
            [env_worker_module.WindowChunkRef(episode_id=1, chunk_index=0)],
        ]
    ]
    worker.rollout_results[0].append_step_result(
        ChunkStepResult(rewards=torch.zeros((1, 2), dtype=torch.float32))
    )
    worker._last_history_query_info[0][0] = (1, 2, 0, [False, False])

    def _bad_assignment(*args, **kwargs):
        out = _assignment([[0.1, 0.2]])
        return RobometerEpisodeReward(
            downsample_indices=out.downsample_indices,
            per_step_progress=out.per_step_progress,
            per_step_reward=out.per_step_reward,
            per_step_loss_mask=out.per_step_loss_mask,
            chunk_reward=out.chunk_reward,
            chunk_loss_mask=np.array([[True, False]]),
            episode_success=False,
        )

    monkeypatch.setattr(
        env_worker_module,
        "reconstruct_robometer_episode_reward",
        _bad_assignment,
    )

    with pytest.raises(RuntimeError, match="loss_mask_false_after_assignment"):
        worker.assign_history_reward(0, torch.ones((1, 2), dtype=torch.float32))


def test_assign_history_reward_uses_query_time_chunk_refs_snapshot(monkeypatch):
    worker = _build_worker("delta")
    worker._window_chunk_refs = [
        [
            [env_worker_module.WindowChunkRef(episode_id=0, chunk_index=0)],
        ]
    ]
    worker.rollout_results[0].append_step_result(
        ChunkStepResult(rewards=torch.zeros((1, 2), dtype=torch.float32))
    )
    worker._last_history_query_info[0][0] = (
        0,
        31,
        29,
        [False] * 31,
        ((0, 0),),
    )
    # Simulate a new chunk being appended after reward query but before assignment.
    worker._window_chunk_refs[0].append(
        [env_worker_module.WindowChunkRef(episode_id=0, chunk_index=1)]
    )
    worker.rollout_results[0].append_step_result(
        ChunkStepResult(rewards=torch.zeros((1, 2), dtype=torch.float32))
    )

    monkeypatch.setattr(
        env_worker_module,
        "reconstruct_robometer_delta_reward",
        lambda *args, **kwargs: _assignment([[0.2, 0.3]]),
    )

    worker.assign_history_reward(0, torch.ones((1, 2), dtype=torch.float32))

    torch.testing.assert_close(
        worker.rollout_results[0].rewards[0][0], torch.tensor([0.2, 0.3])
    )
    torch.testing.assert_close(
        worker.rollout_results[0].rewards[1][0], torch.tensor([0.0, 0.0])
    )
    assert worker.rollout_results[0].loss_mask[0][0].all()
    assert not bool(worker.rollout_results[0].loss_mask[1][0].any())


# ---------------------------------------------------------------------------
# Independent rollout window tests (ASYNC_INDEPENDENT_ROLLOUT_WINDOW_IMPLEMENTATION.md §14)
# ---------------------------------------------------------------------------


def _build_independent_worker(shaping: str = "delta", num_envs: int = 1) -> EnvWorker:
    worker = _build_worker(shaping)
    worker.train_num_envs_per_stage = num_envs
    worker.independent_rollout_windows = True
    worker._independent_window_forced_timeout_masks = [None]
    worker._window_any_episode_success = [False]
    worker.last_obs_list = [None]
    worker.last_intervened_info_list = [(None, None)]
    worker._episode_chunk_ids = [torch.zeros(num_envs, dtype=torch.long)]
    worker._episode_chunk_counts = [torch.zeros(num_envs, dtype=torch.long)]
    return worker


def test_finalize_independent_window_boundary_marks_only_unfinished():
    """§14.1: forced-timeout envs get done+truncation at [:, -1]; naturally-done
    envs keep their original flags; the original env_output is not mutated."""
    worker = _build_independent_worker("delta", num_envs=2)
    env_output = EnvOutput(
        obs={"states": torch.zeros((2, 1), dtype=torch.float32)},
        dones=torch.tensor([[False, True], [False, False]], dtype=torch.bool),
        terminations=torch.tensor([[False, True], [False, False]], dtype=torch.bool),
        truncations=torch.zeros((2, 2), dtype=torch.bool),
    )
    env_metrics: defaultdict[str, list] = defaultdict(list)
    finalized = worker._finalize_independent_window_boundary(env_output, 0, env_metrics)

    # env 0 was naturally done -> flags untouched (done stays True, no truncation)
    assert finalized.dones[0, -1].item() is True
    assert finalized.truncations[0, -1].item() is False
    assert finalized.terminations[0, -1].item() is True
    # env 1 forced timeout -> done + truncation, NOT a task termination
    assert finalized.dones[1, -1].item() is True
    assert finalized.truncations[1, -1].item() is True
    assert finalized.terminations[1, -1].item() is False
    # earlier chunk step untouched
    assert finalized.dones[0, 0].item() is False
    assert finalized.dones[1, 0].item() is False
    assert finalized.truncations[1, 0].item() is False
    # original env_output tensor not mutated in place
    assert env_output.dones[1, -1].item() is False
    assert env_output.truncations[1, -1].item() is False
    # window metrics
    assert int(env_metrics["window/episodes"][0].item()) == 2
    assert int(env_metrics["window/natural_terminal_episodes"][0].item()) == 1
    assert int(env_metrics["window/forced_timeout_episodes"][0].item()) == 1
    torch.testing.assert_close(
        env_metrics["window/forced_timeout_fraction"][0],
        torch.tensor([0.5], dtype=torch.float32),
    )
    # forced-timeout mask stashed for the settlement assertion
    forced = worker._independent_window_forced_timeout_masks[0]
    assert forced[0].item() is False
    assert forced[1].item() is True


def test_gae_does_not_bootstrap_across_independent_window_boundary():
    """§14.2: a done=True at the T+1 boundary stops GAE from bootstrapping the
    next window's value (forced env); continuous (done=False) does bootstrap."""
    from rlinf.algorithms.advantages import compute_gae_advantages_and_returns

    T, bsz = 4, 2
    rewards = torch.ones((T, bsz), dtype=torch.float32)
    # env 0: independent boundary (done at T+1); env 1: continuous (no boundary done)
    dones = torch.zeros((T + 1, bsz), dtype=torch.bool)
    dones[T, 0] = True

    def _last_adv(values_T: float) -> torch.Tensor:
        values = torch.zeros((T + 1, bsz), dtype=torch.float32)
        values[T] = values_T
        adv, _ = compute_gae_advantages_and_returns(
            rewards=rewards,
            gamma=0.99,
            gae_lambda=0.95,
            values=values,
            normalize_advantages=False,
            normalize_returns=False,
            dones=dones,
        )
        return adv[-1]

    adv_low = _last_adv(10.0)
    adv_high = _last_adv(99.0)
    # env 0 (boundary done=True): last-step advantage independent of next-window value
    torch.testing.assert_close(adv_low[0], adv_high[0])
    # env 1 (continuous, done=False): last-step advantage DOES depend on next value
    assert not torch.allclose(adv_low[1], adv_high[1])


def test_get_reward_model_output_covers_forced_timeout_after_finalize(monkeypatch):
    """§14.3: after finalize sets done=True for a forced-timeout env, the
    post-loop Robometer query (last_run=True) still settles it and records a
    failure-ending success trace."""
    worker = _build_independent_worker("delta", num_envs=2)
    worker.train_history_managers = [type("HM", (), {})()]
    hm = worker.train_history_managers[0]
    hm.history_entries = [[{"render_images": "a"}], [{"render_images": "b"}]]
    hm.pickup_counts = [0, 0]
    hm.success_history_entries = [[False], [False]]
    hm.build_history_input = lambda dones, emit_mask: (
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
    worker._episode_chunk_ids = [torch.tensor([0, 0], dtype=torch.long)]
    worker._episode_chunk_counts = [torch.tensor([1, 1], dtype=torch.long)]
    worker._window_chunk_refs = [
        [
            [
                env_worker_module.WindowChunkRef(episode_id=0, chunk_index=0),
                env_worker_module.WindowChunkRef(episode_id=0, chunk_index=0),
            ]
        ]
    ]
    worker.rollout_results[0].append_step_result(
        ChunkStepResult(rewards=torch.zeros((2, 2), dtype=torch.float32))
    )
    worker.send_to = lambda **kwargs: None
    monkeypatch.setattr(
        worker,
        "recv_from",
        lambda **kwargs: torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
    )

    # Finalize: env 0 naturally done, env 1 forced timeout.
    env_output = EnvOutput(
        obs={"states": torch.zeros((2, 1), dtype=torch.float32)},
        final_obs={"states": torch.zeros((2, 1), dtype=torch.float32)},
        env_infos={"success": np.array([True, False])},
        dones=torch.tensor([[False, True], [False, False]], dtype=torch.bool),
        terminations=torch.tensor([[False, True], [False, False]], dtype=torch.bool),
        truncations=torch.zeros((2, 2), dtype=torch.bool),
    )
    env_metrics: defaultdict[str, list] = defaultdict(list)
    finalized = worker._finalize_independent_window_boundary(env_output, 0, env_metrics)

    reward = worker.get_reward_model_output(
        finalized, send_channel=None, recv_channel=None, stage_id=0, last_run=True
    )
    torch.testing.assert_close(
        reward, torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    )
    # Both envs queried; forced-timeout env 1's success trace ends in failure.
    assert set(worker._last_history_query_info[0]) == {0, 1}
    # query_info_with_refs stores a 5-tuple (episode_id, history_len,
    # pickup_count, success_trace, chunk_refs).
    _ep0, _hl0, _pc0, trace0, _refs0 = worker._last_history_query_info[0][0]
    _ep1, _hl1, _pc1, trace1, _refs1 = worker._last_history_query_info[0][1]
    assert trace0[-1] is False  # robometer-level success resolved False (prefix)
    assert trace1[-1] is False


def test_assert_independent_forced_timeouts_settled_passes_and_raises():
    """§14.4: the settlement assertion accepts a forced env settled as failure,
    but raises if it is missing, settled as success, or has no chunk refs."""
    worker = _build_independent_worker("delta", num_envs=1)
    worker._window_chunk_refs = [
        [[env_worker_module.WindowChunkRef(episode_id=0, chunk_index=0)]]
    ]
    # _window_chunk_slices_for_episode guards on len(rollout_results.rewards),
    # so a step result must exist for the chunk ref to be visible.
    worker.rollout_results[0].append_step_result(
        ChunkStepResult(rewards=torch.zeros((1, 2), dtype=torch.float32))
    )
    forced = torch.tensor([True], dtype=torch.bool)

    # Happy path: forced env has a failure assignment + chunk refs.
    ok = RobometerEpisodeReward(
        downsample_indices=[0],
        per_step_progress=np.array([0.1], dtype=np.float32),
        per_step_reward=np.array([0.1], dtype=np.float32),
        per_step_loss_mask=np.array([True]),
        chunk_reward=np.array([[0.1]], dtype=np.float32),
        chunk_loss_mask=np.array([[True]]),
        episode_success=False,
    )
    worker._assert_independent_forced_timeouts_settled(0, forced, {0: ok})

    # Missing assignment -> RuntimeError.
    with pytest.raises(RuntimeError, match="no reward assignment"):
        worker._assert_independent_forced_timeouts_settled(0, forced, {})

    # Settled as success -> RuntimeError.
    bad = RobometerEpisodeReward(
        downsample_indices=[0],
        per_step_progress=np.array([0.1], dtype=np.float32),
        per_step_reward=np.array([0.1], dtype=np.float32),
        per_step_loss_mask=np.array([True]),
        chunk_reward=np.array([[0.1]], dtype=np.float32),
        chunk_loss_mask=np.array([[True]]),
        episode_success=True,
    )
    with pytest.raises(RuntimeError, match="settled as success"):
        worker._assert_independent_forced_timeouts_settled(0, forced, {0: bad})

    # No chunk refs (env auto-reset into an empty episode at the window
    # boundary; previous episode was naturally completed + settled mid-window)
    # -> SKIPPED, not a dropped trajectory. Must NOT raise even with no
    # assignment, because there is no window trajectory to settle.
    worker._window_chunk_refs = [[]]
    worker._assert_independent_forced_timeouts_settled(0, forced, {0: ok})
    worker._assert_independent_forced_timeouts_settled(0, forced, {})


def test_assign_history_reward_delta_applies_failure_penalty_to_forced_timeout(
    monkeypatch,
):
    """§14.4: a forced-timeout env (success_trace all-False) is reconstructed with
    failure_terminal_penalty and episode_success=False (existing failure path)."""
    worker = _build_independent_worker("delta", num_envs=1)
    worker.delta_failure_terminal_penalty = -0.4
    worker._window_chunk_refs = [
        [[env_worker_module.WindowChunkRef(episode_id=0, chunk_index=0)]]
    ]
    worker.rollout_results[0].append_step_result(
        ChunkStepResult(rewards=torch.zeros((1, 2), dtype=torch.float32))
    )
    worker._last_history_query_info[0][0] = (0, 2, 0, [False, False])
    captured = {}

    def _reconstruct(*args, **kwargs):
        captured.update(kwargs)
        out = _assignment([[0.2, 0.3]])
        return RobometerEpisodeReward(
            downsample_indices=out.downsample_indices,
            per_step_progress=out.per_step_progress,
            per_step_reward=out.per_step_reward,
            per_step_loss_mask=out.per_step_loss_mask,
            chunk_reward=out.chunk_reward,
            chunk_loss_mask=out.chunk_loss_mask,
            episode_success=False,
        )

    monkeypatch.setattr(
        env_worker_module, "reconstruct_robometer_delta_reward", _reconstruct
    )
    worker.assign_history_reward(0, torch.ones((1, 2), dtype=torch.float32))
    assert captured["failure_terminal_penalty"] == pytest.approx(-0.4)


def test_reset_train_stage_for_next_independent_window_orders_reset_and_history():
    """§14.5: reset happens after settlement; history is cleared (reset_all) and
    only the fresh pickup prefix is prepended; last_obs_list holds the fresh obs."""
    worker = _build_independent_worker("delta", num_envs=1)

    class _FakeHM:
        def __init__(self):
            self.calls = []
            self.prepend_calls = []

        def reset_all(self):
            self.calls.append("reset_all")

        def prepend_history_entries(self, env_id, frames):
            self.prepend_calls.append((int(env_id), list(frames)))

    class _FakeEnv:
        def __init__(self):
            self.is_start = False
            self.reset_calls = 0
            self._obs = {"states": torch.zeros((1, 1), dtype=torch.float32)}
            self._pickup = {0: ["frame_a", "frame_b"]}

        def reset(self):
            self.reset_calls += 1
            return self._obs, {}

        def consume_pickup_frames(self):
            return self._pickup

    fake_env = _FakeEnv()
    fake_hm = _FakeHM()
    worker.env_list = [fake_env]
    worker.train_history_managers = [fake_hm]
    worker.reward_mode = "history_buffer"

    worker._reset_train_stage_for_next_independent_window(0)

    assert fake_env.reset_calls == 1
    assert fake_env.is_start is True
    assert fake_hm.calls == ["reset_all"]
    assert fake_hm.prepend_calls == [(0, ["frame_a", "frame_b"])]
    assert worker.last_obs_list[0] is fake_env._obs
    assert worker.last_intervened_info_list[0] == (None, None)


def test_prefetch_train_bootstrap_disabled_in_independent_mode():
    """§14.6: independent mode disables prefetch so no stale last_obs_list is
    cached for the next window; _prefetched_train_bootstrap stays None."""
    worker = _build_independent_worker("delta", num_envs=1)
    worker._prefetched_train_bootstrap = None

    def _fail(*args, **kwargs):
        raise AssertionError("prefetch must not bootstrap in independent mode")

    worker._bootstrap_and_send_train = _fail
    worker.prefetch_train_bootstrap(rollout_channel=None)
    assert worker._prefetched_train_bootstrap is None


def test_skip_zero_success_windows_masks_all_fail_trajectory():
    """SR==0 window (no episode succeeded) -> whole trajectory loss_mask all
    False so the actor's masked_mean loss is 0 (no-op update); metric flags it."""
    worker = _build_independent_worker("delta", num_envs=1)
    worker.rollout_results[0] = EmbodiedRolloutResult(max_episode_length=8)
    # Two chunks with some True loss_mask (would normally train).
    worker.rollout_results[0].loss_mask = [
        torch.tensor([[True, True], [True, False]]),
        torch.tensor([[False, True], [True, True]]),
    ]
    worker._window_any_episode_success[0] = False  # window SR == 0

    env_metrics: defaultdict[str, list] = defaultdict(list)
    worker._skip_zero_success_windows(env_metrics)

    # All loss_mask entries now False (trajectory excluded from the gradient).
    for lm in worker.rollout_results[0].loss_mask:
        assert not bool(lm.any())
    assert float(env_metrics["window/skipped_zero_success"][0].item()) == 1.0


def test_skip_zero_success_windows_keeps_successful_trajectory():
    """Window with at least one success -> loss_mask untouched; metric = 0."""
    worker = _build_independent_worker("delta", num_envs=1)
    worker.rollout_results[0] = EmbodiedRolloutResult(max_episode_length=8)
    original = [
        torch.tensor([[True, True], [True, False]]),
        torch.tensor([[False, True], [True, True]]),
    ]
    worker.rollout_results[0].loss_mask = [t.clone() for t in original]
    worker._window_any_episode_success[0] = True  # window had a success

    env_metrics: defaultdict[str, list] = defaultdict(list)
    worker._skip_zero_success_windows(env_metrics)

    # loss_mask unchanged.
    for got, exp in zip(worker.rollout_results[0].loss_mask, original):
        torch.testing.assert_close(got, exp)
    assert float(env_metrics["window/skipped_zero_success"][0].item()) == 0.0


def test_masked_mean_and_ratio_safe_on_all_false_mask():
    """masked_mean already returns 0 for all-False mask; masked_mean_ratio clamps
    loss_mask_ratio to avoid div-by-zero, so an all-fail (fully-masked) batch
    yields a finite 0-grad no-op update (no inf/nan)."""
    values = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([False, False, False])
    torch.testing.assert_close(masked_mean(values, mask), torch.tensor(0.0))
    # Scalar loss_mask_ratio == 0 -> clamped, mask zeroes it -> 0 (scalar mean).
    out = masked_mean_ratio(values, mask, torch.tensor(0.0))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.tensor(0.0))
    # Multi-element (per-batch) loss_mask_ratio with a zero entry -> no crash
    # (the bug that took down the delta-fresh run: float() on a multi-element
    # tensor). Masked batch contributes 0, no inf/nan.
    values2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask2d = torch.tensor([[True, True], [False, False]])
    ratio = torch.tensor([0.5, 0.0])  # batch 1 fully masked
    out2 = masked_mean_ratio(values2d, mask2d, ratio)
    assert torch.isfinite(out2).all()



@pytest.mark.parametrize(
    "mode, auto_reset, hist_mode, epoch, expect_independent, raises",
    [
        ("continuous", True, "rollout_window", 1, False, False),
        ("independent", True, "rollout_window", 1, True, False),
        ("independent", False, "rollout_window", 1, None, True),
        ("independent", True, "complete_episode", 1, None, True),
        ("independent", True, "rollout_window", 2, None, True),
        ("bogus", True, "rollout_window", 1, None, True),
    ],
)
def test_validate_rollout_window_mode(
    mode, auto_reset, hist_mode, epoch, expect_independent, raises
):
    """§14.7: config validation — continuous is legacy; independent requires
    auto_reset, rollout_window history, and rollout_epoch=1."""
    if raises:
        with pytest.raises(ValueError):
            EnvWorker._validate_rollout_window_mode(mode, auto_reset, hist_mode, epoch)
    else:
        assert (
            EnvWorker._validate_rollout_window_mode(mode, auto_reset, hist_mode, epoch)
            is expect_independent
        )
