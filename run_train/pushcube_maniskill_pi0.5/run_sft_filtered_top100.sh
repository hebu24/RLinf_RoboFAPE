#!/usr/bin/env bash
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE

export PYTHONUNBUFFERED=1
export PYTHONPATH=/data/yingxi/RLinf_RoboFAPE:${PYTHONPATH:-}
export PATH=/data/yingxi/RLinf_RoboFAPE/.venv/bin:${PATH}

PYTHON_BIN="${PYTHON_BIN:-/data/yingxi/RLinf_RoboFAPE/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-/data/yingxi/datasets/robofpe_sft/PushCube-v1_wrist_filtered_top100_robometer/lerobot}"
BUILD_DATASET="${BUILD_DATASET:-1}"
OVERWRITE_DATASET="${OVERWRITE_DATASET:-0}"

if [[ "${BUILD_DATASET}" == "1" ]]; then
  build_args=(
    run_train/robofpe_sft_data/build_pushcube_sft_from_filtering.py
    --out-dir "${DATA_DIR}"
    --domain "${FILTER_DOMAIN:-IND}"
    --model-name "${FILTER_MODEL_NAME:-ours_224}"
    --query-task "${FILTER_QUERY_TASK:-PushCube-v1}"
    --source-task-id "${FILTER_SOURCE_TASK_ID:-PushCube-v1}"
    --score-method "${FILTER_SCORE_METHOD:-final}"
    --top-k "${FILTER_TOP_K:-100}"
    --seed "${FILTER_RENDER_SEED:-0}"
    --shader "${FILTER_SHADER:-default}"
    --sim-backend "${FILTER_SIM_BACKEND:-auto}"
  )
  [[ "${OVERWRITE_DATASET}" == "1" ]] && build_args+=(--overwrite)
  [[ -n "${FILTER_LIMIT:-}" ]] && build_args+=(--limit "${FILTER_LIMIT}")
  [[ -n "${FILTER_MAX_EPISODE_FRAMES:-}" ]] && build_args+=(--max-episode-frames "${FILTER_MAX_EPISODE_FRAMES}")
  "${PYTHON_BIN}" "${build_args[@]}"
fi

export DATA_DIR
export GPU_IDS="${GPU_IDS:-0,1}"
export CONFIG_NAME="${CONFIG_NAME:-pushcube_sft_openpi_pi05_wrist}"
export OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_maniskill_wrist}"
export PREPARED_BASE="${PREPARED_BASE:-/data/yingxi/RLinf_RoboFAPE/run_train/pushcube_maniskill_pi0.5/base/pi05_base_pushcube_wrist_filtered_top100}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-pushcube_sft_wrist_filtered_top100_robometer}"
export SFT_RAY_PORT="${SFT_RAY_PORT:-6379}"
export SFT_DASHBOARD_AGENT_PORT="${SFT_DASHBOARD_AGENT_PORT:-52366}"
export SFT_RAY_TMPDIR="${SFT_RAY_TMPDIR:-/tmp/ray_sft_${SFT_RAY_PORT}}"

printf 'DATA_DIR=%s\nGPU_IDS=%s\nEXPERIMENT_NAME=%s\nSFT_RAY_PORT=%s\n' \
  "${DATA_DIR}" "${GPU_IDS}" "${EXPERIMENT_NAME}" "${SFT_RAY_PORT}"

exec bash sft_finetune_pi05base.sh
