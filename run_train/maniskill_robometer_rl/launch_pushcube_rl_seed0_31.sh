#!/usr/bin/env bash
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE

export PYTHONUNBUFFERED=1
export PYTHON_BIN=/data/yingxi/RLinf_RoboFAPE/.venv/bin/python
export RL_GPU_IDS=0,1
export ROBOMETER_GPU_ID=2
export START_ROBOMETER_SERVER=true
export ROBOMETER_STARTUP_WAIT_S=120
export ROBOMETER_PORT=8001
export ROBOMETER_CKPT=/data/yingxi/robometer/checkpoint-400-5tasks
export ROBOMETER_REPO=/data/yingxi/RoboFPE/robometer
export RL_RAY_PORT=6386
export RAY_DASHBOARD_PORT=8266
export RAY_DASHBOARD_AGENT_PORT=52372
export RAY_TMPDIR=/data/yingxi/ray_tmp_pushcube_rl_6386
export LOG_DIR=/data/yingxi/RLinf_RoboFAPE/logs/pushcube_rl_env32_abs30_seed0_31_20260825_gpu012
export RLINF_REWARD_DEBUG_LOG=${LOG_DIR}/robometer_rdebug.log

exec bash run_train/maniskill_robometer_rl/run_task_rl_async.sh \
  pushcube \
  reward.model.timeout_s=600 \
  algorithm.rollout_store_wait_timeout_s=1800
