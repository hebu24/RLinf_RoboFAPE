#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Edit these values directly, or override them with environment variables.
VENV_DIR="${VENV_DIR:-/opt/kairan/envs/rlinf}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/opt/yingxi/rlinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise/checkpoints/global_step_150/actor}"
CONFIG_DIR="${CONFIG_DIR:-${REPO_PATH}/run_train/test_maniskill_pi0.5/config}"
CONFIG_NAME="${CONFIG_NAME:-maniskill_ppo_openpi_pi05}"
TASK_ID="${TASK_ID:-PutOnPlateInScene25Main-v3}"
# Set OBJ_SET='' for tasks whose constructor does not accept obj_set.
OBJ_SET="${OBJ_SET-train}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-}"

# NUM_EVAL_EPISODES is the exact trajectory count and must be divisible by NUM_ENVS.
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-25}"
NUM_ENVS="${NUM_ENVS:-25}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-80}"
SEED="${SEED:-0}"
GPU_IDS="${GPU_IDS:-1}"

SAVE_VIDEO="${SAVE_VIDEO:-true}"
IGNORE_TERMINATIONS="${IGNORE_TERMINATIONS:-true}"
FIXED_RESET_STATE_IDS="${FIXED_RESET_STATE_IDS:-true}"
OBS_MODE="${OBS_MODE:-}"
CONTROL_MODE="${CONTROL_MODE:-}"
SIM_BACKEND="${SIM_BACKEND:-}"
# Extra task constructor arguments, for example: {"use_multiple_plates": false}
INIT_PARAMS_JSON="${INIT_PARAMS_JSON:-{}}"

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

# RLinf placement uses physical GPU IDs, so Ray must discover all GPUs.
unset CUDA_VISIBLE_DEVICES
# Eval runs on its OWN Ray cluster on EVAL_RAY_PORT (default 6380), isolated from
# SFT training (which uses 6379). Never a bare `ray stop` (that kills ALL ray on the
# host, incl. SFT); teardown is scoped to EVAL_RAY_PORT only.
EVAL_RAY_PORT="${EVAL_RAY_PORT:-6380}"
# Pin driver + worker actors to the eval cluster. An inherited RAY_ADDRESS (e.g.
# set by sweep_peginsertion_wrist.py) takes precedence.
export RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:${EVAL_RAY_PORT}}"

_eval_scoped_ray_kill() {
  pkill -9 -f "gcs_server.*--gcs_server_port=${EVAL_RAY_PORT}"  >/dev/null 2>&1 || true
  pkill -9 -f "raylet.*--gcs-address=[^ ]*:${EVAL_RAY_PORT}"    >/dev/null 2>&1 || true
  pkill -9 -f "dashboard.*--gcs-address=[^ ]*:${EVAL_RAY_PORT}" >/dev/null 2>&1 || true
  sleep 2
}

# MANAGE_RAY=false (default): attach to the eval cluster on EVAL_RAY_PORT (e.g. one
# started by the sweep, or a head you started). MANAGE_RAY=true: start a dedicated
# eval head on EVAL_RAY_PORT. Either way, only this port is ever touched.
MANAGE_RAY="${MANAGE_RAY:-false}"
_EVAL_STARTED_RAY=false
if [[ "${MANAGE_RAY}" == "true" ]]; then
  if RAY_ADDRESS="127.0.0.1:${EVAL_RAY_PORT}" "${RAY_BIN}" status >/dev/null 2>&1; then
    echo "Reusing the eval Ray cluster on port ${EVAL_RAY_PORT}; will not stop it."
  else
    RAY_TMP_DIR="${RAY_TMP_DIR:-${REPO_PATH}/logs/ray_tmp}"
    mkdir -p "${RAY_TMP_DIR}"
    unset CUDA_VISIBLE_DEVICES   # so the head's raylet registers all GPUs
    _eval_scoped_ray_kill         # clear stale eval head on this port only
    "${RAY_BIN}" start --head --port="${EVAL_RAY_PORT}" --temp-dir="${RAY_TMP_DIR}" --include-dashboard=false
    _EVAL_STARTED_RAY=true
    export RLINF_EVAL_STARTED_RAY=1
    export RAY_ADDRESS="127.0.0.1:${EVAL_RAY_PORT}"
    trap '_eval_scoped_ray_kill' EXIT
  fi
fi

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
  --num-eval-episodes "${NUM_EVAL_EPISODES}"
  --num-envs "${NUM_ENVS}"
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --seed "${SEED}"
  --gpu-ids "${GPU_IDS}"
  --init-params-json "${INIT_PARAMS_JSON}"
)

[[ -n "${TASK_DESCRIPTION}" ]] && CMD+=(--task-description "${TASK_DESCRIPTION}")
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
