#!/usr/bin/env bash
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE

VENV_DIR="${VENV_DIR:-/data/yingxi/RLinf_RoboFAPE/.venv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the training checkpoints directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR%/checkpoints}/pickcube_sweep_seed0_7_eval50}"
GPU_IDS="${GPU_IDS:-0,1}"
SEEDS="${SEEDS:-0-7}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
NUM_ENVS="${NUM_ENVS:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-180}"
RAY_PORT="${RAY_PORT:-6380}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/data/yingxi/ray_tmp_eval_pickcube_sweep}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-4}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-8000000000}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
POLL_SECONDS="${POLL_SECONDS:-30}"

exec "${VENV_DIR}/bin/python" \
  run_train/eval_checkpoint/sweep_pushcube_wrist.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-script run_train/eval_checkpoint/run_pickcube_wrist.sh \
  --venv-dir "${VENV_DIR}" \
  --gpu-ids "${GPU_IDS}" \
  --seeds "${SEEDS}" \
  --num-eval-episodes "${NUM_EVAL_EPISODES}" \
  --num-envs "${NUM_ENVS}" \
  --max-episode-steps "${MAX_EPISODE_STEPS}" \
  --ray-port "${RAY_PORT}" \
  --ray-num-cpus "${RAY_NUM_CPUS}" \
  --ray-object-store-memory "${RAY_OBJECT_STORE_MEMORY}" \
  --ray-temp-dir "${RAY_TEMP_DIR}" \
  --watch-poll-interval "${POLL_SECONDS}" \
  --ckpt-stable-seconds "${STABLE_SECONDS}" \
  --watch \
  --resume \
  --continue-on-error \
  --pin-checkpoints \
  --no-save-video \
  --no-ray-include-dashboard
