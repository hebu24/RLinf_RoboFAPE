#!/usr/bin/env bash
# Monitor a Vertical SFT checkpoints directory and sweep each stable checkpoint.
set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
if [[ -z "${TRAIN_LOG_DIR}" && -z "${CHECKPOINT_DIR}" ]]; then
  echo "Set TRAIN_LOG_DIR or CHECKPOINT_DIR." >&2
  exit 2
fi

VENV_DIR="${VENV_DIR:-${REPO_PATH}/.venv}"
OUT_DIR="${OUT_DIR:-${TRAIN_LOG_DIR:-${CHECKPOINT_DIR%/checkpoints}}/peginsertion_vertical_sweep_eval}"
EVAL_LOG="${EVAL_LOG:-${OUT_DIR}/monitor.log}"
STATE_FILE="${STATE_FILE:-${OUT_DIR}/.last_seen_step}"
GPU_IDS="${GPU_IDS:-0,1}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
NUM_ENVS="${NUM_ENVS:-2}"
SEEDS="${SEEDS:-0-7}"
POLL_SECONDS="${POLL_SECONDS:-30}"
RAY_PORT="${RAY_PORT:-6394}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8394}"
RAY_TMP_DIR="${RAY_TMP_DIR:-/data/yingxi/ray_vertical_sweep}"

mkdir -p "${OUT_DIR}"
exec >>"${EVAL_LOG}" 2>&1
echo "[$(date '+%F %T')] Vertical monitor started"

while true; do
  current_checkpoint_dir="${CHECKPOINT_DIR}"
  if [[ -z "${current_checkpoint_dir}" ]]; then
    current_checkpoint_dir="$(find "${TRAIN_LOG_DIR}" -maxdepth 4 -type d -name checkpoints 2>/dev/null | sort -V | tail -1 || true)"
  fi
  if [[ -z "${current_checkpoint_dir}" || ! -d "${current_checkpoint_dir}" ]]; then
    sleep "${POLL_SECONDS}"
    continue
  fi

  latest_actor="$(find "${current_checkpoint_dir}" -mindepth 2 -maxdepth 2 -type d -name actor 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -z "${latest_actor}" ]]; then
    sleep "${POLL_SECONDS}"
    continue
  fi
  if [[ ! -e "${latest_actor}/dcp_checkpoint" || ! -e "${latest_actor}/model_state_dict" || ! -e "${latest_actor}/trainer_state.json" ]]; then
    sleep "${POLL_SECONDS}"
    continue
  fi

  latest_step="$(basename "$(dirname "${latest_actor}")")"
  latest_mtime="$(find "${latest_actor}" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 || true)"
  now="$(date +%s)"
  stable=0
  if [[ -n "${latest_mtime}" ]] && awk -v now="${now}" -v latest="${latest_mtime}" 'BEGIN { exit !((now - latest) >= 120) }'; then
    stable=1
  fi
  previous=""
  [[ -f "${STATE_FILE}" ]] && previous="$(<"${STATE_FILE}")"
  if [[ "${stable}" == 1 && "${latest_step}" != "${previous}" ]]; then
    echo "[$(date '+%F %T')] Stable ${latest_step}; starting Vertical 8-seed sweep"
    if "${VENV_DIR}/bin/python" "${REPO_PATH}/run_train/eval_checkpoint/sweep_peginsertion_wrist.py" \
      --checkpoint-dir "${current_checkpoint_dir}" \
      --output-dir "${OUT_DIR}" \
      --venv-dir "${VENV_DIR}" \
      --gpu-ids "${GPU_IDS}" \
      --seeds "${SEEDS}" \
      --num-eval-episodes "${NUM_EVAL_EPISODES}" \
      --num-envs "${NUM_ENVS}" \
      --max-episode-steps 450 \
      --run-script "${REPO_PATH}/run_train/eval_checkpoint/run_peginsertionvertical_wrist.sh" \
      --no-save-video \
      --resume \
      --continue-on-error \
      --ray-port "${RAY_PORT}" \
      --ray-dashboard-port "${RAY_DASHBOARD_PORT}" \
      --ray-tmp-dir "${RAY_TMP_DIR}"; then
      printf '%s\n' "${latest_step}" >"${STATE_FILE}"
      echo "[$(date '+%F %T')] Sweep finished for ${latest_step}"
    else
      echo "[$(date '+%F %T')] Sweep failed for ${latest_step}; will retry"
    fi
  fi
  sleep "${POLL_SECONDS}"
done
