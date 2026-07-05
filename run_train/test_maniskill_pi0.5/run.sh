#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_NAME="maniskill_ppo_openpi_pi05"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export EMBODIED_PATH="${SCRIPT_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

ray stop >/dev/null 2>&1 || true
ray start --head --num-gpus=1

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}-gpu${CUDA_VISIBLE_DEVICES}"
mkdir -p "${LOG_DIR}"

python "${REPO_PATH}/examples/embodiment/train_embodied_agent.py" \
  --config-path "${SCRIPT_DIR}/config" \
  --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOG_DIR}" \
  2>&1 | tee "${LOG_DIR}/run_embodiment.log"
