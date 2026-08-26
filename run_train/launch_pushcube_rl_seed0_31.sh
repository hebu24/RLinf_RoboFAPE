#!/usr/bin/env bash
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE

export PYTHONUNBUFFERED=1
export PYTHON_BIN="${PYTHON_BIN:-/data/yingxi/RLinf_RoboFAPE/.venv/bin/python}"
export RL_GPU_IDS="${RL_GPU_IDS:-0,1}"
export ROBOMETER_GPU_ID="${ROBOMETER_GPU_ID:-2}"
export START_ROBOMETER_SERVER="${START_ROBOMETER_SERVER:-true}"
export ROBOMETER_STARTUP_WAIT_S="${ROBOMETER_STARTUP_WAIT_S:-120}"
export ROBOMETER_PORT="${ROBOMETER_PORT:-8001}"
export ROBOMETER_CKPT="${ROBOMETER_CKPT:-/data/yingxi/robometer/checkpoint-400-5tasks}"
export ROBOMETER_REPO="${ROBOMETER_REPO:-/data/yingxi/RoboFPE/robometer}"
export RL_RAY_PORT="${RL_RAY_PORT:-6386}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
export RAY_DASHBOARD_AGENT_PORT="${RAY_DASHBOARD_AGENT_PORT:-52372}"
export RAY_TMPDIR="${RAY_TMPDIR:-/data/yingxi/ray_tmp_pushcube_rl_${RL_RAY_PORT}}"
export LOG_DIR="${LOG_DIR:-/data/yingxi/RLinf_RoboFAPE/logs/pushcube_rl_$(date +%Y%m%d_%H%M%S)}"
export RLINF_REWARD_DEBUG_LOG="${RLINF_REWARD_DEBUG_LOG:-${LOG_DIR}/robometer_rdebug.log}"

mkdir -p "${LOG_DIR}"
printf 'RL_GPU_IDS=%s\nROBOMETER_GPU_ID=%s\nLOG_DIR=%s\nRL_RAY_PORT=%s\nROBOMETER_PORT=%s\n' \
  "${RL_GPU_IDS}" "${ROBOMETER_GPU_ID}" "${LOG_DIR}" "${RL_RAY_PORT}" "${ROBOMETER_PORT}"

exec bash run_train/maniskill_robometer_rl/run_task_rl_async.sh \
  pushcube \
  reward.model.timeout_s=600 \
  algorithm.rollout_store_wait_timeout_s=1800 \
  "$@"
