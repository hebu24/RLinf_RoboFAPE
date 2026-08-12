set -x
cd /data/yingxi/RLinf_RoboFAPE
export CUDA_VISIBLE_DEVICES=4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RL_RAY_PORT=6396
export RAY_DASHBOARD_AGENT_PORT=52382
export RAY_MIN_WORKER_PORT=10402
export RAY_MAX_WORKER_PORT=10799
export TMPDIR=/data/yingxi/tmp
export HF_HOME=/data/yingxi/hf_cache
export CONFIG_NAME=maniskill_async_ppo_peg_insertion_pi05_oracle_value
export LOG_DIR="logs/$(date +'%Y%m%d-%H:%M:%S')-peg_insertion_rl_oracle_value_gpu45_smoke"
export RLINF_REWARD_DEBUG=1
bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
  reward.model.server_url=http://127.0.0.1:8000 \
  runner.max_epochs=2 \
  env.train.total_num_envs=8 \
  env.eval.total_num_envs=2 \
  actor.global_batch_size=480 \
  reward.model.timeout_s=600 \
  algorithm.rollout_store_wait_timeout_s=3600
echo "===EXIT=$?==="
exec bash
