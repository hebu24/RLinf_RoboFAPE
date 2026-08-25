#!/usr/bin/env bash
# PushCube-v1 wrist-camera checkpoint evaluation.
# 2-image wrist (base_camera + hand_camera), pd_joint_pos control, pi05_maniskill_wrist.
# Reuses the isolated eval Ray cluster (EVAL_RAY_PORT=6380) so it never touches the SFT cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_DIR="${VENV_DIR:-/opt/kairan/envs/rlinf}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
CONFIG_DIR="${CONFIG_DIR:-${REPO_PATH}/run_train/pushcube_maniskill_pi0.5/config}"
CONFIG_NAME="${CONFIG_NAME:-maniskill_pushcube_wrist_sft_eval_openpi_pi05}"
TASK_ID="${TASK_ID:-PushCube-v1}"
OBJ_SET="${OBJ_SET:-}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-Push the cube across the tabletop until its center lies inside the target region}"

# 50 episodes / 50 envs = 1 rollout epoch. max_episode_steps 180 divisible by num_action_chunks 10.
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
NUM_ENVS="${NUM_ENVS:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-180}"
SEED="${SEED:-0}"
GPU_IDS="${GPU_IDS:-7}"

SAVE_VIDEO="${SAVE_VIDEO:-true}"
IGNORE_TERMINATIONS="${IGNORE_TERMINATIONS:-true}"
FIXED_RESET_STATE_IDS="${FIXED_RESET_STATE_IDS:-true}"
# pd_joint_pos is NOT in get_robot_control_mode's mapping, so we MUST pass it explicitly
# to override validate_cfg's derivation from policy_setup.
CONTROL_MODE="${CONTROL_MODE:-pd_joint_pos}"
OBS_MODE="${OBS_MODE:-}"
SIM_BACKEND="${SIM_BACKEND:-}"
INIT_PARAMS_JSON="${INIT_PARAMS_JSON:-{}}"
EVAL_ACTION_SCALE="${EVAL_ACTION_SCALE:-1.0}"

PYTHON_BIN="${VENV_DIR}/bin/python"
RAY_BIN="${VENV_DIR}/bin/ray"
if [[ ! -x "${PYTHON_BIN}" || ! -x "${RAY_BIN}" ]]; then
  echo "Missing Python or Ray in ${VENV_DIR}." >&2; exit 1
fi
if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "Set CHECKPOINT_PATH to the actor checkpoint you want to evaluate." >&2; exit 1
fi
if [[ ! -e "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint does not exist: ${CHECKPOINT_PATH}" >&2; exit 1
fi
if [[ "$(basename "${CHECKPOINT_PATH}")" == "checkpoints" ]]; then
  echo "CHECKPOINT_PATH must point to one global_step_<N>/actor directory, not the checkpoints root." >&2; exit 1
fi
if ! "${PYTHON_BIN}" -c "import openpi" >/dev/null 2>&1; then
  echo "OpenPI is not installed in ${VENV_DIR}." >&2; exit 1
fi

export EMBODIED_PATH="${CONFIG_DIR%/config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
[[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
[[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]] && export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
ulimit -n 1048576 2>/dev/null || true

EVAL_RAY_PORT="${EVAL_RAY_PORT:-6380}"
export RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:${EVAL_RAY_PORT}}"

_eval_scoped_ray_kill() {
  local ray_tmp_dir="${RAY_TMP_DIR:-/tmp/ray_eval_pushcube}"
  pkill -9 -f "gcs_server.*--gcs_server_port=${EVAL_RAY_PORT}"  >/dev/null 2>&1 || true
  pkill -9 -f "raylet.*--gcs-address=[^ ]*:${EVAL_RAY_PORT}"    >/dev/null 2>&1 || true
  pkill -9 -f "dashboard.*--gcs-address=[^ ]*:${EVAL_RAY_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "ray/autoscaler/_private/monitor.py.*--logs-dir=${ray_tmp_dir}/" >/dev/null 2>&1 || true
  pkill -9 -f "ray/_private/log_monitor.py.*--session-dir=${ray_tmp_dir}/" >/dev/null 2>&1 || true
  sleep 2
}

MANAGE_RAY="${MANAGE_RAY:-false}"
_EVAL_STARTED_RAY=false
if [[ "${MANAGE_RAY}" == "true" ]]; then
  if RAY_ADDRESS="127.0.0.1:${EVAL_RAY_PORT}" "${RAY_BIN}" status >/dev/null 2>&1; then
    echo "Reusing the eval Ray cluster on port ${EVAL_RAY_PORT}; will not stop it."
  else
    RAY_TMP_DIR="${RAY_TMP_DIR:-/tmp/ray_eval_pushcube}"
    mkdir -p "${RAY_TMP_DIR}"
    unset CUDA_VISIBLE_DEVICES
    _eval_scoped_ray_kill
    "${RAY_BIN}" start --head --port="${EVAL_RAY_PORT}" --temp-dir="${RAY_TMP_DIR}" --include-dashboard=true --dashboard-host=127.0.0.1 --dashboard-port="${EVAL_RAY_DASHBOARD_PORT:-8265}"
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
