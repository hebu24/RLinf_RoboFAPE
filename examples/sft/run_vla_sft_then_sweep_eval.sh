#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -gt 0 ]]; then
  CONFIG_NAME="$1"
  shift
else
  CONFIG_NAME="maniskill_ppo_openvlaoft"
fi

LOG_DIR="${LOG_DIR:-${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}-3200}"
OUT_DIR="${OUT_DIR:-${LOG_DIR}/sweep_seed0_7_eval50}"
SEEDS="${SEEDS:-0-7}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
AUTO_EVAL_KIND="${AUTO_EVAL_KIND:-auto}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TRAIN_ARGS=("$@")

export LOG_DIR

bash "${SCRIPT_DIR}/run_vla_sft.sh" "${CONFIG_NAME}" "${TRAIN_ARGS[@]}"

case "${AUTO_EVAL_KIND}" in
  none)
    exit 0
    ;;
  auto)
    case "${CONFIG_NAME}" in
      *pushcube*)
        EVAL_SCRIPT="${REPO_PATH}/run_train/eval_checkpoint/sweep_pushcube_wrist.py"
        ;;
      *peg_insertion*)
        EVAL_SCRIPT="${REPO_PATH}/run_train/eval_checkpoint/sweep_peginsertion_wrist.py"
        ;;
      *)
        printf 'No post-train sweep mapping for config %s; skipping eval.\n' "${CONFIG_NAME}" >&2
        exit 0
        ;;
    esac
    ;;
  pushcube)
    EVAL_SCRIPT="${REPO_PATH}/run_train/eval_checkpoint/sweep_pushcube_wrist.py"
    ;;
  peginsertion|peg_insertion)
    EVAL_SCRIPT="${REPO_PATH}/run_train/eval_checkpoint/sweep_peginsertion_wrist.py"
    ;;
  *)
    printf 'Unknown AUTO_EVAL_KIND=%s\n' "${AUTO_EVAL_KIND}" >&2
    exit 2
    ;;
esac

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${LOG_DIR}/checkpoints}"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  printf 'Missing checkpoints dir: %s\n' "${CHECKPOINT_DIR}" >&2
  exit 1
fi

printf 'Training finished. Launching sweep eval from %s\n' "${CHECKPOINT_DIR}"
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
