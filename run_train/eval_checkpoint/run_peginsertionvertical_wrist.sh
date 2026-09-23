#!/usr/bin/env bash
# Full-trajectory Vertical PegInsertion SFT evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CONFIG_DIR="${CONFIG_DIR:-${REPO_PATH}/run_train/peginsertion_maniskill_pi0.5/config}"
export VENV_DIR="${VENV_DIR:-${REPO_PATH}/.venv}"
export CONFIG_NAME="${CONFIG_NAME:-maniskill_peg_insertion_vertical_wrist_sft_eval_openpi_pi05}"
export TASK_ID="${TASK_ID:-PegInsertionVertical-v1}"
export TASK_DESCRIPTION="${TASK_DESCRIPTION:-Insert the peg vertically into the target hole.}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-450}"
export CONTROL_MODE="${CONTROL_MODE:-pd_joint_pos}"

if [[ "${TASK_ID}" != "PegInsertionVertical-v1" ]]; then
  echo "run_peginsertionvertical_wrist.sh only supports PegInsertionVertical-v1 (got ${TASK_ID})." >&2
  exit 2
fi

exec bash "${SCRIPT_DIR}/run_peginsertion_wrist.sh" "$@"
