#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export VENV_DIR="${VENV_DIR:-${REPO_PATH}/.venv}"
export CONFIG_DIR="${CONFIG_DIR:-${REPO_PATH}/run_train/pushcube_maniskill_pi0.5/config}"
export CONFIG_NAME="${CONFIG_NAME:-maniskill_liftpegupright_wrist_sft_eval_openpi_pi05}"
export TASK_ID="${TASK_ID:-LiftPegUpright-box}"
export TASK_DESCRIPTION="${TASK_DESCRIPTION:-Stand the cracker box upright on the table.}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-350}"
export NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
export NUM_ENVS="${NUM_ENVS:-50}"
export CONTROL_MODE="${CONTROL_MODE:-pd_joint_pos}"
export EVAL_RAY_PORT="${EVAL_RAY_PORT:-6380}"
export RAY_TMP_DIR="${RAY_TMP_DIR:-/data/ray_eval_liftpegupright}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-/data/yingxi/robofac}"
# LiftPegUpright requires CUDA for both the policy and ManiSkill GPU simulation.
if ! "${VENV_DIR}/bin/python" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  echo "LiftPegUpright evaluation must be launched on a CUDA-capable node." >&2
  exit 1
fi
unset SEEDS

exec bash "${SCRIPT_DIR}/run_pushcube_wrist.sh" "$@"
