#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_NAME="${CONFIG_NAME:-maniskill_peg_insertion_vertical_ppo_openpi_pi05}"
VENV_DIR="${VENV_DIR:-/opt/kairan/envs/rlinf}"
PYTHON_BIN="${VENV_DIR}/bin/python"
RAY_BIN="${VENV_DIR}/bin/ray"

export EMBODIED_PATH="${SCRIPT_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${RAY_BIN}" ]]; then
  echo "Missing Python or Ray in ${VENV_DIR}." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import openpi" >/dev/null 2>&1; then
  echo "OpenPI is not installed in ${VENV_DIR}." >&2
  exit 1
fi

# RLinf placement uses physical GPU IDs, so Ray must discover all GPUs.
unset CUDA_VISIBLE_DEVICES
"${RAY_BIN}" stop >/dev/null 2>&1 || true
RAY_TMP_DIR="${REPO_PATH}/logs/ray_tmp"
mkdir -p "${RAY_TMP_DIR}"
"${RAY_BIN}" start --head --temp-dir="${RAY_TMP_DIR}"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"

"${PYTHON_BIN}" "${REPO_PATH}/examples/embodiment/train_embodied_agent.py" \
  --config-path "${SCRIPT_DIR}/config" \
  --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOG_DIR}" \
  "$@" \
  2>&1 | tee "${LOG_DIR}/run_embodiment.log"
