#!/usr/bin/env bash
# Single-GPU PushCube-v1 step20k SFT rollout rendering.
#
# This is for machines with only one visible GPU. It runs seeds serially so every
# trajectory remains an independent MP4 (NUM_ENVS=1) without launching multiple
# Ray heads or over-subscribing Torch compile workers.
set -euo pipefail

REPO_PATH="${REPO_PATH:-/data/yingxi/RLinf_RoboFAPE}"
cd "${REPO_PATH}"

VENV_DIR="${VENV_DIR:-${REPO_PATH}/.venv}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_PATH}/logs/pushcube_sft_wrist_filtered_conservative_20260823/train_40k_fresh_0_3/pushcube_sft_wrist_filtered_conservative_40k_fresh_0_3/checkpoints/global_step_20000/actor}"
OUT_ROOT="${OUT_ROOT:-${REPO_PATH}/logs/pushcube_sft_step20k_fail_videos_1gpu_serial}"

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
GPU_ID="${GPU_ID:-0}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-180}"
EVAL_RAY_PORT="${EVAL_RAY_PORT:-6380}"
EVAL_RAY_DASHBOARD_PORT="${EVAL_RAY_DASHBOARD_PORT:-8265}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"
MAX_VIDEOS="${MAX_VIDEOS:-50}"

if [[ "${NUM_ENVS}" != "1" ]]; then
  echo "NUM_ENVS must stay 1 so each trajectory is rendered as an independent video." >&2
  exit 1
fi
if [[ ! -x "${VENV_DIR}/bin/python" || ! -x "${VENV_DIR}/bin/ray" ]]; then
  echo "Missing Python or Ray in VENV_DIR=${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint actor dir does not exist: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "REPO_PATH=${REPO_PATH}"
echo "VENV_DIR=${VENV_DIR}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "SEEDS=${SEEDS}"
echo "GPU_ID=${GPU_ID}"
echo "NUM_EVAL_EPISODES=${NUM_EVAL_EPISODES}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}"
echo "TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS}"
echo

for seed in ${SEEDS}; do
  log_dir="${OUT_ROOT}/seed_${seed}"
  launcher_log="${OUT_ROOT}/seed_${seed}.launcher.log"
  ray_tmp_dir="/tmp/rpe_${seed}_${EVAL_RAY_PORT}"

  echo "[start] seed=${seed} gpu=${GPU_ID} log_dir=${log_dir}"
  VENV_DIR="${VENV_DIR}" \
  CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
  LOG_DIR="${log_dir}" \
  GPU_IDS="${GPU_ID}" \
  NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES}" \
  NUM_ENVS="${NUM_ENVS}" \
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
  SEED="${seed}" \
  EVAL_RAY_PORT="${EVAL_RAY_PORT}" \
  EVAL_RAY_DASHBOARD_PORT="${EVAL_RAY_DASHBOARD_PORT}" \
  RAY_TMP_DIR="${ray_tmp_dir}" \
  MANAGE_RAY=true \
  SAVE_VIDEO="${SAVE_VIDEO}" \
  bash run_train/eval_checkpoint/run_pushcube_wrist.sh \
    --save-episode-metrics \
    "env.eval.video_cfg.max_videos=${MAX_VIDEOS}" \
    "env.group_name=EnvGroupEvalSeed${seed}" \
    "rollout.group_name=RolloutGroupEvalSeed${seed}" \
    > "${launcher_log}" 2>&1
  echo "[done] seed=${seed}; log=${launcher_log}"
done

echo
echo "Outputs: ${OUT_ROOT}"
