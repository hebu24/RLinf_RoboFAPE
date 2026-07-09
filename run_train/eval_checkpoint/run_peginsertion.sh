#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Edit these values directly, or override them with environment variables.
VENV_DIR="${VENV_DIR:-/opt/kairan/envs/rlinf}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/opt/yingxi/rlinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise/checkpoints/global_step_150/actor}"
CONFIG_DIR="${CONFIG_DIR:-${REPO_PATH}/run_train/peginsertion_maniskill_pi0.5/config}"
CONFIG_NAME="${CONFIG_NAME:-maniskill_peg_insertion_vertical_ppo_openpi_pi05}"
TASK_ID="${TASK_ID:-PegInsertionVertical-v1}"
OBJ_SET="${OBJ_SET:-}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-insert the blue peg vertically into the orange hole}"

# NUM_EVAL_EPISODES is the exact trajectory count and must be divisible by NUM_ENVS.
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-25}"
NUM_ENVS="${NUM_ENVS:-5}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-600}"
SEED="${SEED:-0}"
GPU_IDS="${GPU_IDS:-0}"

SAVE_VIDEO="${SAVE_VIDEO:-true}"
IGNORE_TERMINATIONS="${IGNORE_TERMINATIONS:-true}"
FIXED_RESET_STATE_IDS="${FIXED_RESET_STATE_IDS:-false}"
OBS_MODE="${OBS_MODE:-}"
CONTROL_MODE="${CONTROL_MODE:-}"
SIM_BACKEND="${SIM_BACKEND:-}"
INIT_PARAMS_JSON="${INIT_PARAMS_JSON:-{}}"
EVAL_ACTION_SCALE="${EVAL_ACTION_SCALE:-1.0}"

PYTHON_BIN="${VENV_DIR}/bin/python"
RAY_BIN="${VENV_DIR}/bin/ray"
if [[ ! -x "${PYTHON_BIN}" || ! -x "${RAY_BIN}" ]]; then
  echo "Missing Python or Ray in ${VENV_DIR}." >&2
  exit 1
fi
if [[ ! -e "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c "import openpi" >/dev/null 2>&1; then
  echo "OpenPI is not installed in ${VENV_DIR}." >&2
  exit 1
fi

export EMBODIED_PATH="${CONFIG_DIR%/config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

# Raise fd limit so raylet can manage many worker connections on multi-GPU runs.
ulimit -n 1048576 2>/dev/null || true

# RLinf placement uses physical GPU IDs, so Ray must discover all GPUs.
unset CUDA_VISIBLE_DEVICES
"${RAY_BIN}" stop >/dev/null 2>&1 || true
RAY_TMP_DIR="${REPO_PATH}/logs/ray_tmp"
mkdir -p "${RAY_TMP_DIR}"
"${RAY_BIN}" start --head --temp-dir="${RAY_TMP_DIR}"
trap '"${RAY_BIN}" stop >/dev/null 2>&1 || true' EXIT

LOG_DIR="${LOG_DIR:-${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-eval-${TASK_ID}}"
mkdir -p "${LOG_DIR}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/eval_checkpoint.py"
  --checkpoint-path "${CHECKPOINT_PATH}"
  --config-dir "${CONFIG_DIR}"
  --config-name "${CONFIG_NAME}"
  --log-dir "${LOG_DIR}"
  --task-id "${TASK_ID}"
  --obj-set "${OBJ_SET}"
  --task-description "${TASK_DESCRIPTION}"
  --num-eval-episodes "${NUM_EVAL_EPISODES}"
  --num-envs "${NUM_ENVS}"
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --seed "${SEED}"
  --gpu-ids "${GPU_IDS}"
  --init-params-json "${INIT_PARAMS_JSON}"
  --action-scale "${EVAL_ACTION_SCALE}"
)

[[ -n "${OBS_MODE}" ]] && CMD+=(--obs-mode "${OBS_MODE}")
[[ -n "${CONTROL_MODE}" ]] && CMD+=(--control-mode "${CONTROL_MODE}")
[[ -n "${SIM_BACKEND}" ]] && CMD+=(--sim-backend "${SIM_BACKEND}")
[[ "${SAVE_VIDEO}" == "true" ]] && CMD+=(--save-video) || CMD+=(--no-save-video)
[[ "${IGNORE_TERMINATIONS}" == "true" ]] && CMD+=(--ignore-terminations) || CMD+=(--no-ignore-terminations)
[[ "${FIXED_RESET_STATE_IDS}" == "true" ]] && CMD+=(--fixed-reset-state-ids) || CMD+=(--no-fixed-reset-state-ids)
CMD+=("$@")

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/eval.log"
