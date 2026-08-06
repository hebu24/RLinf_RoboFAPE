"""Validate the formal 16-episode async Delta training batch (continuous and
independent rollout-window variants).

§15 of ASYNC_INDEPENDENT_ROLLOUT_WINDOW_IMPLEMENTATION.md: in independent mode
the 16 env slots each finalize as a window-local episode per window (some natural
completions, some synthetic timeouts), so ``episodes_per_update`` stays 16 but
its semantics shift from "16 completed episodes" to "16 window-local finalized
episodes".
"""

from pathlib import Path

from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parents[2] / "examples/embodiment/config"
CONTINUOUS_CONFIG_PATH = (
    CONFIG_DIR / "maniskill_async_ppo_peg_insertion_pi05_delta.yaml"
)
INDEPENDENT_CONFIG_PATH = (
    CONFIG_DIR / "maniskill_async_ppo_peg_insertion_pi05_delta_independent_window.yaml"
)


def _assert_16_episode_one_optimizer_batch(cfg) -> None:
    actor_world_size = 2
    episodes_per_trajectory_per_rank = (
        int(cfg.env.train.total_num_envs) // actor_world_size
    )
    trajectories_per_rank = int(cfg.algorithm.rollout_store_size_per_rank)
    episodes_per_update = (
        episodes_per_trajectory_per_rank * trajectories_per_rank * actor_world_size
    )
    # PPO samples model action chunks (10 actions), while execute_action_chunks
    # controls low-level environment stepping and must not change flattening.
    chunks_per_episode = int(cfg.env.train.max_episode_steps) // int(
        cfg.actor.model.num_action_chunks
    )
    flattened_samples_per_rank = (
        episodes_per_trajectory_per_rank * trajectories_per_rank * chunks_per_episode
    )
    samples_per_optimizer_step_per_rank = (
        int(cfg.actor.global_batch_size) // actor_world_size
    )

    assert episodes_per_trajectory_per_rank == 4
    assert trajectories_per_rank == 2
    assert episodes_per_update == 16
    assert chunks_per_episode == 60
    assert flattened_samples_per_rank == 480
    assert int(cfg.env.train.execute_action_chunks) == 8
    assert int(cfg.env.eval.execute_action_chunks) == 8
    assert flattened_samples_per_rank // samples_per_optimizer_step_per_rank == 1
    assert int(cfg.actor.micro_batch_size) == 8
    assert samples_per_optimizer_step_per_rank // int(cfg.actor.micro_batch_size) == 60
    assert (
        int(cfg.actor.global_batch_size)
        % (int(cfg.actor.micro_batch_size) * actor_world_size)
        == 0
    )
    assert float(cfg.actor.optim.lr) == 3e-7
    assert "value_loss_coef" not in cfg.algorithm
    assert bool(cfg.actor.model.openpi.detach_critic_input)
    assert int(cfg.actor.grad_diagnostics_interval) == 0
    assert bool(cfg.env.train.shared_reset_seed)
    assert bool(cfg.env.eval.shared_reset_seed)
    assert bool(cfg.env.train.use_fixed_reset_state_ids)
    assert bool(cfg.env.eval.use_fixed_reset_state_ids)


def test_delta_async_update_uses_16_window_local_episodes_and_one_optimizer_step():
    """Renamed from ..._16_completed_episodes_...: in independent mode the 16
    slots are window-local finalized episodes (natural + synthetic timeout),
    not necessarily 16 natural completions."""
    cfg = OmegaConf.load(CONTINUOUS_CONFIG_PATH)
    _assert_16_episode_one_optimizer_batch(cfg)
    assert int(cfg.actor.optim.critic_warmup_steps) == 50
    assert float(cfg.reward.delta.failure_terminal_penalty) == -0.4
    # Continuous (legacy) config does not opt into independent windows.
    assert cfg.env.train.get("rollout_window_mode", "continuous") == "continuous"


def test_independent_window_delta_config():
    """The W40 independent variant keeps its own rollout and batch geometry."""
    cfg = OmegaConf.load(INDEPENDENT_CONFIG_PATH)
    assert int(cfg.actor.optim.critic_warmup_steps) == 15
    assert float(cfg.reward.delta.failure_terminal_penalty) == -1.0
    assert cfg.env.train.rollout_window_mode == "independent"
    assert bool(cfg.env.train.auto_reset)
    assert cfg.reward.history_train_mode == "rollout_window"
    assert bool(cfg.reward.history_reward_assign)
    # experiment/log name must distinguish from legacy continuous-window runs.
    assert "independent_window" in cfg.runner.logger.experiment_name
    # warmup-end permanent checkpoint (step 15) for clean resume.
    assert bool(cfg.runner.get("save_critic_warmup_checkpoint"))
    assert 15 in list(cfg.runner.get("checkpoint_permanent_steps", []))


def test_400_step_independent_window_batch_matches_40_chunk_rollout():
    """The W40 variant must not inherit the W60 optimizer batch size."""
    cfg = OmegaConf.load(INDEPENDENT_CONFIG_PATH)
    actor_world_size = 2
    trajectories_per_rank = int(cfg.algorithm.rollout_store_size_per_rank)
    episodes_per_trajectory_per_rank = int(cfg.env.train.total_num_envs) // actor_world_size
    chunks_per_episode = int(cfg.env.train.max_episode_steps) // int(
        cfg.actor.model.num_action_chunks
    )
    local_rollout_samples = (
        trajectories_per_rank * episodes_per_trajectory_per_rank * chunks_per_episode
    )

    assert int(cfg.env.train.max_episode_steps) == 400
    assert int(cfg.reward.model.max_robometer_frames) == 40
    assert local_rollout_samples == 320
    assert int(cfg.actor.global_batch_size) == local_rollout_samples * actor_world_size
    assert local_rollout_samples % int(cfg.actor.micro_batch_size) == 0
