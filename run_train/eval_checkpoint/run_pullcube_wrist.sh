#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export VENV_DIR="${VENV_DIR:-${REPO_PATH}/.venv}"
export CONFIG_DIR="${CONFIG_DIR:-${REPO_PATH}/run_train/pushcube_maniskill_pi0.5/config}"
export CONFIG_NAME="${CONFIG_NAME:-maniskill_pickcube_wrist_sft_eval_openpi_pi05}"
export TASK_ID="${TASK_ID:-PullCube-block}"
export TASK_DESCRIPTION="${TASK_DESCRIPTION:-Pull the wooden block into the target region.}"
export EVAL_RAY_PORT="${EVAL_RAY_PORT:-6380}"
export RAY_TMP_DIR="${RAY_TMP_DIR:-/data/ray_eval_pullcube}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
# PullCube evaluation requires CUDA for both the policy and ManiSkill GPU simulation.
if ! "${VENV_DIR}/bin/python" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  echo "PullCube evaluation must be launched on a CUDA-capable node." >&2
  exit 1
fi
# Use PullCubeBlockEnv's collection-time default render camera. Set
# INIT_PARAMS_JSON only when deliberately evaluating a camera perturbation.
# The sweep assigns one SEED per worker; do not recurse over its aggregate selector.
unset SEEDS
exec bash "${SCRIPT_DIR}/run_pushcube_wrist.sh" "$@"
