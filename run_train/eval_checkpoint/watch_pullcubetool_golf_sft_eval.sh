#!/usr/bin/env bash
set -euo pipefail
cd /data/yingxi/RLinf_RoboFAPE
VENV_DIR="${VENV_DIR:-/data/yingxi/RLinf_RoboFAPE/.venv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the SFT checkpoints directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_DIR%/checkpoints}/pullcubetool_golf_sweep_seed0_7_eval50_fixed_camera}"
GPU_IDS="${GPU_IDS:-0,1}"
SEEDS="${SEEDS:-0-7}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
NUM_ENVS="${NUM_ENVS:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-350}"
RAY_PORT="${RAY_PORT:-6380}"
TRAIN_PATTERN="${TRAIN_PATTERN:-train_vla_sft.py.*pullcubetool_golf_sft_pi05_wrist}"
POLL_SECONDS="${POLL_SECONDS:-30}"
while pgrep -af "${TRAIN_PATTERN}" >/dev/null 2>&1; do
  printf '[%(%F %T)T] SFT is active; delaying GPU eval.\n' -1
  sleep "${POLL_SECONDS}"
done
exec "${VENV_DIR}/bin/python" run_train/eval_checkpoint/sweep_pullcubetool_golf_wrist.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" --output-dir "${OUTPUT_DIR}" \
  --run-script run_train/eval_checkpoint/run_pullcubetool_golf_wrist.sh \
  --watch --venv-dir "${VENV_DIR}" --gpu-ids "${GPU_IDS}" --seeds "${SEEDS}" \
  --num-eval-episodes "${NUM_EVAL_EPISODES}" --num-envs "${NUM_ENVS}" --max-episode-steps "${MAX_EPISODE_STEPS}" \
  --ray-port "${RAY_PORT}" --ray-num-cpus "${RAY_NUM_CPUS:-4}" --ray-object-store-memory "${RAY_OBJECT_STORE_MEMORY:-8000000000}" \
  --ray-temp-dir "${RAY_TEMP_DIR:-/data/ray_tmp_eval_pullcubetool_golf_sweep}" --watch-poll-interval "${POLL_SECONDS}" \
  --ckpt-stable-seconds "${STABLE_SECONDS:-120}" --resume --continue-on-error --pin-checkpoints --no-save-video --no-ray-include-dashboard
