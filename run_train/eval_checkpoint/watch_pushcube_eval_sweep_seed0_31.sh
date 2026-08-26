#!/usr/bin/env bash
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE

export PYTHONUNBUFFERED=1

RL_LOG_DIR="${RL_LOG_DIR:?Set RL_LOG_DIR to the RL experiment directory}"
CKPT_DIR="${CKPT_DIR:-${RL_LOG_DIR}/pushcube_rl_robometer_absolute/checkpoints}"
OUT_DIR="${OUT_DIR:-${RL_LOG_DIR}/sweep_gpu3}"
SWEEP_GPU_IDS="${SWEEP_GPU_IDS:-3}"
SWEEP_RAY_PORT="${SWEEP_RAY_PORT:-6387}"
SWEEP_RAY_DASHBOARD_PORT="${SWEEP_RAY_DASHBOARD_PORT:-8267}"
SWEEP_RAY_DASHBOARD_AGENT_PORT="${SWEEP_RAY_DASHBOARD_AGENT_PORT:-52373}"
SWEEP_RAY_CLIENT_SERVER_PORT="${SWEEP_RAY_CLIENT_SERVER_PORT:-10041}"
SWEEP_RAY_MIN_WORKER_PORT="${SWEEP_RAY_MIN_WORKER_PORT:-13400}"
SWEEP_RAY_MAX_WORKER_PORT="${SWEEP_RAY_MAX_WORKER_PORT:-13799}"
SWEEP_RAY_TEMP_DIR="${SWEEP_RAY_TEMP_DIR:-/data/yingxi/ray_tmp_pushcube_eval_${SWEEP_RAY_PORT}}"
SWEEP_SEEDS="${SWEEP_SEEDS:-0-7}"
SWEEP_NUM_EPISODES="${SWEEP_NUM_EPISODES:-50}"
SWEEP_NUM_ENVS="${SWEEP_NUM_ENVS:-10}"
SWEEP_MAX_EPISODE_STEPS="${SWEEP_MAX_EPISODE_STEPS:-200}"
SWEEP_POLL_INTERVAL="${SWEEP_POLL_INTERVAL:-10}"

mkdir -p "${OUT_DIR}"
printf 'RL_LOG_DIR=%s\nCKPT_DIR=%s\nOUT_DIR=%s\nSWEEP_GPU_IDS=%s\nSWEEP_RAY_PORT=%s\n' \
  "${RL_LOG_DIR}" "${CKPT_DIR}" "${OUT_DIR}" "${SWEEP_GPU_IDS}" "${SWEEP_RAY_PORT}"

exec "${PYTHON_BIN:-/data/yingxi/RLinf_RoboFAPE/.venv/bin/python}" \
  run_train/eval_checkpoint/sweep_pushcube_wrist.py \
  --checkpoint-dir "${CKPT_DIR}" \
  --output-dir "${OUT_DIR}" \
  --watch \
  --resume \
  --continue-on-error \
  --gpu-ids "${SWEEP_GPU_IDS}" \
  --seeds "${SWEEP_SEEDS}" \
  --num-eval-episodes "${SWEEP_NUM_EPISODES}" \
  --num-envs "${SWEEP_NUM_ENVS}" \
  --max-episode-steps "${SWEEP_MAX_EPISODE_STEPS}" \
  --ray-port "${SWEEP_RAY_PORT}" \
  --ray-dashboard-port "${SWEEP_RAY_DASHBOARD_PORT}" \
  --ray-dashboard-agent-port "${SWEEP_RAY_DASHBOARD_AGENT_PORT}" \
  --ray-client-server-port "${SWEEP_RAY_CLIENT_SERVER_PORT}" \
  --ray-min-worker-port "${SWEEP_RAY_MIN_WORKER_PORT}" \
  --ray-max-worker-port "${SWEEP_RAY_MAX_WORKER_PORT}" \
  --ray-temp-dir "${SWEEP_RAY_TEMP_DIR}" \
  --watch-poll-interval "${SWEEP_POLL_INTERVAL}" \
  --no-save-video
