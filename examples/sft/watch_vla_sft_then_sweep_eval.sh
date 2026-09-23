#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-${REPO_PATH}/logs}"
LOG_DIR="${LOG_DIR:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
CONFIG_NAME="${CONFIG_NAME:-}"
AUTO_EVAL_KIND="${AUTO_EVAL_KIND:-auto}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEEDS="${SEEDS:-0-7}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
TRAIN_PATTERN="${TRAIN_PATTERN:-train_vla_sft.py}"
OUT_DIR="${OUT_DIR:-}"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

latest_log_dir() {
  if [[ -n "${LOG_DIR}" ]]; then
    printf '%s\n' "${LOG_DIR}"
    return 0
  fi
  local match
  if [[ -n "${CONFIG_NAME}" ]]; then
    match="$(find "${LOG_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "*-${CONFIG_NAME}-3200" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
    if [[ -n "${match}" ]]; then
      printf '%s\n' "${match}"
      return 0
    fi
  fi
  find "${LOG_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
}

training_active() {
  if pgrep -af "${TRAIN_PATTERN}" >/dev/null 2>&1; then
    if [[ -n "${CONFIG_NAME}" ]]; then
      pgrep -af "${TRAIN_PATTERN}.*${CONFIG_NAME}" >/dev/null 2>&1 && return 0
    else
      return 0
    fi
  fi
  return 1
}

ckpt_complete() {
  local actor="$1"
  [[ -d "${actor}" ]] || return 1
  [[ -e "${actor}/dcp_checkpoint" ]] || return 1
  [[ -e "${actor}/model_state_dict" ]] || return 1
  [[ -e "${actor}/trainer_state.json" ]] || return 1
}

ckpt_stable() {
  local actor="$1"
  local latest now age
  latest="$(find "${actor}" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 || true)"
  [[ -n "${latest}" ]] || return 1
  now="$(date +%s)"
  age="$(awk -v now="${now}" -v latest="${latest}" 'BEGIN { printf "%d", now - latest }')"
  [[ "${age}" -ge "${STABLE_SECONDS}" ]]
}

latest_actor() {
  find "${CHECKPOINT_DIR}" -mindepth 2 -maxdepth 2 -type d -name actor -print 2>/dev/null | sort -V | tail -1
}

select_eval_script() {
  case "${AUTO_EVAL_KIND}" in
    none)
      return 1
      ;;
    pushcube)
      printf '%s\n' "${REPO_PATH}/run_train/eval_checkpoint/sweep_pushcube_wrist.py"
      ;;
    peginsertion|peg_insertion)
      printf '%s\n' "${REPO_PATH}/run_train/eval_checkpoint/sweep_peginsertion_wrist.py"
      ;;
    auto)
      case "${CONFIG_NAME:-${LOG_DIR}}" in
        *pushcube*)
          printf '%s\n' "${REPO_PATH}/run_train/eval_checkpoint/sweep_pushcube_wrist.py"
          ;;
        *peg_insertion*|*peginsertion*)
          printf '%s\n' "${REPO_PATH}/run_train/eval_checkpoint/sweep_peginsertion_wrist.py"
          ;;
        *)
          return 1
          ;;
      esac
      ;;
    *)
      return 1
      ;;
  esac
}

LOG_DIR="$(latest_log_dir)"
if [[ -z "${LOG_DIR}" ]]; then
  echo "Could not find a training log dir under ${LOG_ROOT}." >&2
  exit 1
fi
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${LOG_DIR}/checkpoints}"
OUT_DIR="${OUT_DIR:-${LOG_DIR}/sweep_seed0_7_eval50}"
EVAL_SCRIPT="$(select_eval_script)" || { echo "Could not infer eval script from CONFIG_NAME=${CONFIG_NAME} or LOG_DIR=${LOG_DIR}." >&2; exit 1; }

log "watching ${LOG_DIR}"
log "checkpoint dir ${CHECKPOINT_DIR}"

while true; do
  current_log_dir="$(latest_log_dir)"
  if [[ -n "${current_log_dir}" ]]; then
    LOG_DIR="${current_log_dir}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-${LOG_DIR}/checkpoints}"
    OUT_DIR="${OUT_DIR:-${LOG_DIR}/sweep_seed0_7_eval50}"
  fi

  actor="$(latest_actor || true)"
  if [[ -n "${actor}" ]] && ckpt_complete "${actor}" && ckpt_stable "${actor}" && ! training_active; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

mkdir -p "${OUT_DIR}"
log "training finished; launching eval from ${CHECKPOINT_DIR}"
python "${EVAL_SCRIPT}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-dir "${OUT_DIR}" \
  --resume \
  --continue-on-error \
  --pin-checkpoints \
  --gpu-ids "${GPU_IDS}" \
  --seeds "${SEEDS}" \
  --num-eval-episodes "${NUM_EVAL_EPISODES}" \
  --no-save-video
