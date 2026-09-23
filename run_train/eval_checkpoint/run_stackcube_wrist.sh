#!/usr/bin/env bash
# StackCube-v1 ManiSkill wrist-camera checkpoint evaluation.
# 2-image policy input: render_camera + hand_camera; pd_joint_pos; 8D actions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_DIR="${VENV_DIR:-/data/yingxi/RLinf_RoboFAPE/.venv}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
CONFIG_DIR="${CONFIG_DIR:-${REPO_PATH}/run_train/pushcube_maniskill_pi0.5/config}"
CONFIG_NAME="${CONFIG_NAME:-maniskill_stackcube_wrist_sft_eval_openpi_pi05}"
TASK_ID="${TASK_ID:-StackCube-v1}"
OBJ_SET="${OBJ_SET:-}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-Stack the red cube on top of the green cube.}"

NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
NUM_ENVS="${NUM_ENVS:-5}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
SEED="${SEED:-0}"
SEEDS="${SEEDS:-}"
GPU_IDS="${GPU_IDS:-0}"

SAVE_VIDEO="${SAVE_VIDEO:-true}"
IGNORE_TERMINATIONS="${IGNORE_TERMINATIONS:-true}"
FIXED_RESET_STATE_IDS="${FIXED_RESET_STATE_IDS:-true}"
CONTROL_MODE="${CONTROL_MODE:-pd_joint_pos}"
OBS_MODE="${OBS_MODE:-}"
SIM_BACKEND="${SIM_BACKEND:-}"
INIT_PARAMS_JSON="${INIT_PARAMS_JSON:-{}}"
EVAL_ACTION_SCALE="${EVAL_ACTION_SCALE:-1.0}"

PYTHON_BIN="${VENV_DIR}/bin/python"
RAY_BIN="${VENV_DIR}/bin/ray"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python in ${VENV_DIR}." >&2
  exit 1
fi
if [[ -x "${RAY_BIN}" ]]; then
  RAY_CMD=("${RAY_BIN}")
elif "${PYTHON_BIN}" -c "import ray" >/dev/null 2>&1; then
  RAY_CMD=("${PYTHON_BIN}" -m ray)
else
  echo "Missing Ray executable/module in ${VENV_DIR}." >&2
  exit 1
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH to the actor checkpoint you want to evaluate." >&2
  exit 1
fi
if [[ ! -e "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ "$(basename "${CHECKPOINT_PATH}")" == "checkpoints" ]]; then
  echo "CHECKPOINT_PATH must point to one global_step_<N>/actor directory, not the checkpoints root." >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c "import openpi" >/dev/null 2>&1; then
  echo "OpenPI is not installed in ${VENV_DIR}." >&2
  exit 1
fi

MANISKILL_STATS="${CHECKPOINT_PATH}/physical-intelligence/maniskill/norm_stats.json"
ISAAC_STATS="${CHECKPOINT_PATH}/RLinf/IsaacLab-Stack-Cube-Data/norm_stats.json"
if [[ ! -f "${MANISKILL_STATS}" && -f "${ISAAC_STATS}" ]]; then
  mkdir -p "$(dirname "${MANISKILL_STATS}")"
  tmp_stats="${MANISKILL_STATS}.tmp.$$"
  cp -f "${ISAAC_STATS}" "${tmp_stats}"
  mv -f "${tmp_stats}" "${MANISKILL_STATS}"
fi

export EMBODIED_PATH="${CONFIG_DIR%/config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
[[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
[[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]] && export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export HYDRA_FULL_ERROR=1
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
ulimit -n 1048576 2>/dev/null || true

EVAL_RAY_PORT="${EVAL_RAY_PORT:-6380}"
EVAL_RAY_MIN_WORKER_PORT="${EVAL_RAY_MIN_WORKER_PORT:-}"
EVAL_RAY_MAX_WORKER_PORT="${EVAL_RAY_MAX_WORKER_PORT:-}"
EVAL_RAY_INCLUDE_DASHBOARD="${EVAL_RAY_INCLUDE_DASHBOARD:-true}"
export RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:${EVAL_RAY_PORT}}"

_eval_scoped_ray_kill() {
  local ray_tmp_dir="${RAY_TMP_DIR:-/tmp/ray_eval_stackcube}"
  pkill -9 -f "gcs_server.*--gcs_server_port=${EVAL_RAY_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "raylet.*--gcs-address=[^ ]*:${EVAL_RAY_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "dashboard.*--gcs-address=[^ ]*:${EVAL_RAY_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "ray/autoscaler/_private/monitor.py.*--logs-dir=${ray_tmp_dir}/" >/dev/null 2>&1 || true
  pkill -9 -f "ray/_private/log_monitor.py.*--session-dir=${ray_tmp_dir}/" >/dev/null 2>&1 || true
  sleep 2
}

MANAGE_RAY="${MANAGE_RAY:-true}"
FORCE_RESTART_RAY="${FORCE_RESTART_RAY:-false}"
if [[ "${MANAGE_RAY}" == "true" ]]; then
  if [[ "${FORCE_RESTART_RAY}" == "true" ]]; then
    RAY_TMP_DIR="${RAY_TMP_DIR:-/tmp/ray_eval_stackcube}"
    _eval_scoped_ray_kill
  fi
  if [[ "${FORCE_RESTART_RAY}" != "true" ]] && RAY_ADDRESS="127.0.0.1:${EVAL_RAY_PORT}" "${RAY_CMD[@]}" status >/dev/null 2>&1; then
    echo "Reusing the eval Ray cluster on port ${EVAL_RAY_PORT}; will not stop it."
  else
    RAY_TMP_DIR="${RAY_TMP_DIR:-/tmp/ray_eval_stackcube}"
    mkdir -p "${RAY_TMP_DIR}"
    # Keep caller-provided CUDA_VISIBLE_DEVICES so eval can avoid GPUs with broken Vulkan offscreen rendering.
    _eval_scoped_ray_kill
    RAY_START_ARGS=(start --head --port="${EVAL_RAY_PORT}" --temp-dir="${RAY_TMP_DIR}" --include-dashboard="${EVAL_RAY_INCLUDE_DASHBOARD}")
    [[ -n "${EVAL_RAY_NUM_CPUS:-}" ]] && RAY_START_ARGS+=(--num-cpus="${EVAL_RAY_NUM_CPUS}")
    if [[ "${EVAL_RAY_INCLUDE_DASHBOARD}" == "true" ]]; then
      RAY_START_ARGS+=(--dashboard-host=127.0.0.1 --dashboard-port="${EVAL_RAY_DASHBOARD_PORT:-8265}")
    fi
    if [[ -n "${EVAL_RAY_MIN_WORKER_PORT}" || -n "${EVAL_RAY_MAX_WORKER_PORT}" ]]; then
      if [[ -z "${EVAL_RAY_MIN_WORKER_PORT}" || -z "${EVAL_RAY_MAX_WORKER_PORT}" ]]; then
        echo "Set both EVAL_RAY_MIN_WORKER_PORT and EVAL_RAY_MAX_WORKER_PORT, or neither." >&2
        exit 1
      fi
      RAY_START_ARGS+=(--min-worker-port="${EVAL_RAY_MIN_WORKER_PORT}" --max-worker-port="${EVAL_RAY_MAX_WORKER_PORT}")
    fi
    "${RAY_CMD[@]}" "${RAY_START_ARGS[@]}"
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
  --task-description "${TASK_DESCRIPTION}"
  --num-eval-episodes "${NUM_EVAL_EPISODES}"
  --num-envs "${NUM_ENVS}"
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --seed "${SEED}"
  --gpu-ids "${GPU_IDS}"
  --init-params-json "${INIT_PARAMS_JSON}"
  --action-scale "${EVAL_ACTION_SCALE}"
  --control-mode "${CONTROL_MODE}"
)

[[ -n "${SEEDS}" ]] && CMD+=(--seeds "${SEEDS}")
[[ -n "${OBS_MODE}" ]] && CMD+=(--obs-mode "${OBS_MODE}")
[[ -n "${SIM_BACKEND}" ]] && CMD+=(--sim-backend "${SIM_BACKEND}")
[[ "${SAVE_VIDEO}" == "true" ]] && CMD+=(--save-video) || CMD+=(--no-save-video)
[[ "${IGNORE_TERMINATIONS}" == "true" ]] && CMD+=(--ignore-terminations) || CMD+=(--no-ignore-terminations)
[[ "${FIXED_RESET_STATE_IDS}" == "true" ]] && CMD+=(--fixed-reset-state-ids) || CMD+=(--no-fixed-reset-state-ids)
CMD+=("$@")

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/eval.log"
