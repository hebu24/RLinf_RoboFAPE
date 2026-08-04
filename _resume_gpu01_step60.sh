#!/bin/bash
set -x
cd /data/yingxi/RLinf_RoboFAPE
export CUDA_VISIBLE_DEVICES=0,1
export RL_RAY_PORT=6384
export RAY_DASHBOARD_PORT=8264
export RAY_DASHBOARD_AGENT_PORT=52370
export RAY_MIN_WORKER_PORT=10002
export RAY_MAX_WORKER_PORT=10399
export LOG_DIR=logs/20260803-12:39:09-peg_insertion_rl_async_absolute_lr1e6_env16_batch960_warmup30_gpu01
bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh runner.resume_dir=/data/yingxi/RLinf_RoboFAPE/logs/20260803-12:39:09-peg_insertion_rl_async_absolute_lr1e6_env16_batch960_warmup30_gpu01/peg_insertion_async_ppo_pi05_robometer/checkpoints/global_step_60_trainenvstep_56736 actor.optim.lr=1e-6 actor.global_batch_size=960 env.train.total_num_envs=16 actor.optim.critic_warmup_steps=30 reward.model.timeout_s=600 algorithm.rollout_store_wait_timeout_s=1800
rc=$?
echo ===SESSION_EXIT=$rc===
exec bash
