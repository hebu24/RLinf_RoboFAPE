#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/data/yingxi/RLinf_RoboFAPE/logs/20260905-21:56:50-peg_insertion_sft_openpi_pi05_wrist-3200"
CHECKPOINT_DIR="$(find "${LOG_DIR}" -maxdepth 3 -type d -path '*/checkpoints' | sort | tail -1)"
OUT_DIR="${LOG_DIR}/peg_insertion_side_sweep_eval"
EVAL_LOG="${LOG_DIR}/peg_side_sweep_eval.log"
STATE_FILE="${OUT_DIR}/.last_seen_step"
export TASK_ID="PegInsertionSide-v1"
export TASK_DESCRIPTION="Insert the peg into the side-facing target hole."
GPU_IDS=(0 1)
NUM_ENVS=2

mkdir -p "${OUT_DIR}"
echo "[$(date '+%F %T')] watcher started" >> "${EVAL_LOG}"

while true; do
  CHECKPOINT_DIR="$(find "${LOG_DIR}" -maxdepth 3 -type d -path '*/checkpoints' | sort | tail -1)"
  if [[ -z "${CHECKPOINT_DIR}" ]]; then
    sleep 30
    continue
  fi
  latest_actor="$(find "${CHECKPOINT_DIR}" -mindepth 2 -maxdepth 2 -type d -name actor 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -n "${latest_actor}" ]]; then
    latest_step="$(basename "$(dirname "${latest_actor}")")"
    if [[ ! -e "${latest_actor}/dcp_checkpoint" || ! -e "${latest_actor}/model_state_dict" || ! -e "${latest_actor}/trainer_state.json" ]]; then
      sleep 30
      continue
    fi
    latest_mtime="$(find "${latest_actor}" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 || true)"
    now="$(date +%s)"
    stable=0
    if [[ -n "${latest_mtime}" ]] && awk -v now="${now}" -v latest="${latest_mtime}" 'BEGIN { exit !((now - latest) >= 120) }'; then
      stable=1
    fi
    previous=""
    [[ -f "${STATE_FILE}" ]] && previous="$(cat "${STATE_FILE}")"
    if [[ "${stable}" == 1 && "${latest_step}" != "${previous}" ]]; then
      echo "[$(date '+%F %T')] new stable ${latest_step}; launching 2-GPU parallel sweep" >> "${EVAL_LOG}"
      if /data/yingxi/RLinf_RoboFAPE/.venv/bin/python \
        /data/yingxi/RLinf_RoboFAPE/run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --output-dir "${OUT_DIR}" \
        --venv-dir /data/yingxi/RLinf_RoboFAPE/.venv \
        --gpu-ids "${GPU_IDS[0]},${GPU_IDS[1]}" \
        --seeds 0-7 \
        --num-eval-episodes 50 \
        --num-envs "${NUM_ENVS}" \
        --max-episode-steps 350 \
        --no-save-video \
        --resume \
        --continue-on-error \
        --ray-port 6392 \
        --ray-dashboard-port 8392 \
        --ray-tmp-dir /data/yingxi/ray_side_sweep \
        >> "${EVAL_LOG}" 2>&1; then
        printf '%s\n' "${latest_step}" > "${STATE_FILE}"
        echo "[$(date '+%F %T')] sweep finished for ${latest_step}" >> "${EVAL_LOG}"
      else
        echo "[$(date '+%F %T')] sweep failed for ${latest_step}; will retry" >> "${EVAL_LOG}"
      fi
    fi
  fi
  sleep 30
done
