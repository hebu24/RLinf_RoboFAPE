#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export VENV_DIR="${VENV_DIR:-${REPO_PATH}/.venv}"
export CONFIG_DIR="${CONFIG_DIR:-${REPO_PATH}/run_train/pushcube_maniskill_pi0.5/config}"
export CONFIG_NAME="${CONFIG_NAME:-maniskill_pickcube_wrist_sft_eval_openpi_pi05}"
export TASK_ID="${TASK_ID:-PickCube-ball}"
export TASK_DESCRIPTION="${TASK_DESCRIPTION:-Pick up the tennis ball and place it at the target.}"
export EVAL_RAY_PORT="${EVAL_RAY_PORT:-6380}"
export RAY_TMP_DIR="${RAY_TMP_DIR:-/data/yingxi/ray_tmp_eval_pickcube}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
# The sweep assigns one SEED per worker; do not recurse over its aggregate selector.
unset SEEDS

exec bash "${SCRIPT_DIR}/run_pushcube_wrist.sh" "$@"
