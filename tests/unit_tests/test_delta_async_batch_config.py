"""Validate the formal 16-episode async Delta training batch."""

from pathlib import Path

from omegaconf import OmegaConf

CONFIG_PATH = (
    Path(__file__).parents[2]
    / "examples/embodiment/config/maniskill_async_ppo_peg_insertion_pi05_delta.yaml"
)


def test_delta_async_update_uses_16_completed_episodes_and_one_optimizer_step():
    cfg = OmegaConf.load(CONFIG_PATH)

    actor_world_size = 2
    episodes_per_trajectory_per_rank = (
        int(cfg.env.train.total_num_envs) // actor_world_size
    )
    trajectories_per_rank = int(cfg.algorithm.rollout_store_size_per_rank)
    episodes_per_update = (
        episodes_per_trajectory_per_rank * trajectories_per_rank * actor_world_size
    )
    chunks_per_episode = int(cfg.env.train.max_episode_steps) // int(
        cfg.env.train.execute_action_chunks
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
    assert flattened_samples_per_rank == 480
    assert flattened_samples_per_rank // samples_per_optimizer_step_per_rank == 1
    assert int(cfg.actor.micro_batch_size) == 8
    assert samples_per_optimizer_step_per_rank // int(cfg.actor.micro_batch_size) == 60
    assert (
        int(cfg.actor.global_batch_size)
        % (int(cfg.actor.micro_batch_size) * actor_world_size)
        == 0
    )
    assert float(cfg.actor.optim.lr) == 3e-7
    assert int(cfg.actor.optim.critic_warmup_steps) == 50
    assert "value_loss_coef" not in cfg.algorithm
    assert bool(cfg.actor.model.openpi.detach_critic_input)
    assert int(cfg.actor.grad_diagnostics_interval) == 0
