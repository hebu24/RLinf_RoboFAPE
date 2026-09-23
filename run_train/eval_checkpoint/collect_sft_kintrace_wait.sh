#!/usr/bin/env bash
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE

THREAD_LIMIT="${THREAD_LIMIT:-8500}"
CHECK_INTERVAL_S="${CHECK_INTERVAL_S:-60}"
LOG_ROOT="${LOG_ROOT:-/data/yingxi/RLinf_RoboFAPE/logs/kintrace_sft_global_step_20000_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${LOG_ROOT}/collect.log"

mkdir -p "${LOG_ROOT}"

echo "[collect] log_root=${LOG_ROOT}" | tee -a "${RUN_LOG}"
echo "[collect] waiting for thread_count <= ${THREAD_LIMIT} and no active eval_checkpoint.py" | tee -a "${RUN_LOG}"

while true; do
  thread_count="$(ps -eLf | wc -l | tr -d ' ')"
  active_eval_count="$(
    ps -eo cmd= \
      | grep 'run_train/eval_checkpoint/eval_checkpoint.py' \
      | grep -v 'kintrace_sft_global_step_20000' \
      | grep -v grep \
      | wc -l \
      | tr -d ' '
  )"
  echo "[collect] $(date +%F_%T) thread_count=${thread_count} active_eval_count=${active_eval_count}" | tee -a "${RUN_LOG}"
  if [[ "${thread_count}" -le "${THREAD_LIMIT}" && "${active_eval_count}" -eq 0 ]]; then
    break
  fi
  sleep "${CHECK_INTERVAL_S}"
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export OPENPI_INPUT_TRANSFORM_MAX_WORKERS=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1

export VENV_DIR=/data/yingxi/RLinf_RoboFAPE/.venv
export CHECKPOINT_PATH=/data/yingxi/RLinf_RoboFAPE/logs/pushcube_sft_wrist_filtered_conservative_20260823/train_40k_fresh_0_3/pushcube_sft_wrist_filtered_conservative_40k_fresh_0_3/checkpoints/global_step_20000/actor
export LOG_DIR="${LOG_ROOT}"
export NUM_EVAL_EPISODES=50
export NUM_ENVS=10
export MAX_EPISODE_STEPS=200
export SEEDS=0-7
export GPU_IDS=2
export SAVE_VIDEO=false
export MANAGE_RAY=true
export FORCE_RESTART_RAY=true
export EVAL_RAY_PORT=6592
export EVAL_RAY_INCLUDE_DASHBOARD=false
export EVAL_RAY_NUM_CPUS=8
export EVAL_RAY_MIN_WORKER_PORT=14200
export EVAL_RAY_MAX_WORKER_PORT=14599
export RAY_TMP_DIR=/tmp/ray_eval_kintrace_sft_20000

echo "[collect] starting SFT kintrace eval" | tee -a "${RUN_LOG}"
bash run_train/eval_checkpoint/run_pushcube_wrist.sh \
  --save-episode-metrics \
  +env.eval.record_kinematic_trace=true \
  2>&1 | tee -a "${RUN_LOG}"

echo "[collect] completed ${LOG_ROOT}" | tee -a "${RUN_LOG}"
