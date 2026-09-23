#!/usr/bin/env bash
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE
export PYTHONUNBUFFERED=1

CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the SFT checkpoints directory}"
OUT_DIR="${OUT_DIR:-$(dirname "${CHECKPOINT_DIR}")/uprightstack_sweep_seed0_49_eval1}"
GPU_IDS="${GPU_IDS:-0,1}"
SEEDS="${SEEDS:-0-49}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-1}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
RAY_PORT="${RAY_PORT:-6395}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8295}"
RAY_DASHBOARD_AGENT_PORT="${RAY_DASHBOARD_AGENT_PORT:-52395}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-10095}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-14500}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-14899}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/data/yingxi/ray_tmp_eval_uprightstack_${RAY_PORT}}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-false}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"
VIDEO_SEEDS="${VIDEO_SEEDS:-0,10,20,30,40}"
PYTHON_BIN="${PYTHON_BIN:-/data/yingxi/RLinf_RoboFAPE/.venv/bin/python}"

mkdir -p "${OUT_DIR}"
exec "${PYTHON_BIN}" run_train/eval_checkpoint/sweep_pushcube_wrist.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-dir "${OUT_DIR}" \
  --run-script run_train/eval_checkpoint/run_uprightstack_wrist.sh \
  --gpu-ids "${GPU_IDS}" \
  --seeds "${SEEDS}" \
  --video-seeds "${VIDEO_SEEDS}" \
  --num-eval-episodes "${NUM_EVAL_EPISODES}" \
  --num-envs "${NUM_ENVS}" \
  --max-episode-steps "${MAX_EPISODE_STEPS}" \
  --ray-port "${RAY_PORT}" \
  --ray-dashboard-port "${RAY_DASHBOARD_PORT}" \
  --ray-dashboard-agent-port "${RAY_DASHBOARD_AGENT_PORT}" \
  --ray-client-server-port "${RAY_CLIENT_SERVER_PORT}" \
  --ray-min-worker-port "${RAY_MIN_WORKER_PORT}" \
  --ray-max-worker-port "${RAY_MAX_WORKER_PORT}" \
  --ray-temp-dir "${RAY_TEMP_DIR}" \
  $(if [[ "${RAY_INCLUDE_DASHBOARD}" == "true" ]]; then printf "%s" "--ray-include-dashboard"; else printf "%s" "--no-ray-include-dashboard"; fi) \
  --watch-poll-interval "${POLL_SECONDS}" \
  --ckpt-stable-seconds "${STABLE_SECONDS}" \
  --watch \
  --resume \
  --continue-on-error \
  --pin-checkpoints \
  $(if [[ "${SAVE_VIDEO}" == "true" ]]; then printf '%s' '--save-video'; else printf '%s' '--no-save-video'; fi)
