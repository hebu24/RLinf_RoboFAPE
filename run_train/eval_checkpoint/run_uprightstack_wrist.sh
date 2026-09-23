#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export VENV_DIR="${VENV_DIR:-${REPO_PATH}/.venv}"
export CONFIG_DIR="${CONFIG_DIR:-${REPO_PATH}/run_train/pushcube_maniskill_pi0.5/config}"
export CONFIG_NAME="${CONFIG_NAME:-maniskill_uprightstack_wrist_sft_eval_openpi_pi05}"
export TASK_ID="${TASK_ID:-UprightStack-v1}"
export TASK_DESCRIPTION="${TASK_DESCRIPTION:-Stand the brick upright and stack it on the red cube.}"
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
export NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
export NUM_ENVS="${NUM_ENVS:-50}"
export CONTROL_MODE="${CONTROL_MODE:-pd_joint_pos}"
export EVAL_RAY_PORT="${EVAL_RAY_PORT:-6380}"
export RAY_TMP_DIR="${RAY_TMP_DIR:-/data/ray_eval_uprightstack}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-/data/yingxi/robofac}"
export PYTHONPATH="/data/yingxi/RoboFPE:/data/yingxi/RoboFPE/mani_envs:/data/yingxi/RoboFPE/mani_envs/data_collection/utils:${PYTHONPATH:-}"
# UprightStack requires CUDA for both the policy and ManiSkill GPU simulation.
if ! "${VENV_DIR}/bin/python" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
  echo "UprightStack evaluation must be launched on a CUDA-capable node." >&2
  exit 1
fi
exec bash "${SCRIPT_DIR}/run_pushcube_wrist.sh" "$@"
