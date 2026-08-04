#!/usr/bin/env bash
# Ablation RL run: replicate gpu23 ("23卡训练") setting but swap robometer reward
# ckpt checkpoint-400 (RBM/20bin, :8000) -> BASE Robometer-4B (RFM/10bin, bf16)
# served on :8001/GPU4. GPUs 4,5 (CVD=4,5), Ray port 6386. Overrides match gpu23:
# lr=1e-6 envs=16 batch=960 shaping=absolute(implicit). expandable_segments on both
# RL actor + server to avoid the fragmentation OOM that killed attempt 1 (server's
# 8x60-frame RFM forward + idle RL rank0 on shared GPU4).
set -x
cd /data/yingxi/RLinf_RoboFAPE
export CUDA_VISIBLE_DEVICES=4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RL_RAY_PORT=6386
export RAY_DASHBOARD_AGENT_PORT=52372
export LOG_DIR="logs/$(date +'%Y%m%d-%H:%M:%S')-peg_insertion_rl_async_absolute_lr1e6_env16_batch960_robometer4b_gpu45"
bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
  actor.optim.lr=1e-6 \
  env.train.total_num_envs=16 \
  actor.global_batch_size=960 \
  reward.model.server_url=http://127.0.0.1:8001
echo "===EXIT=$?==="
exec bash
