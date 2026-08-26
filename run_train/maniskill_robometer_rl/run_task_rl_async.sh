#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
SRC_FILE="${EMBODIED_PATH}/train_async.py"

TASK="${TASK:-${1:-pushcube}}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${TASK}" in
  pushcube)
    CONFIG_NAME="${CONFIG_NAME:-maniskill_async_ppo_pushcube_robometer_absolute}"
    MODEL_PATH="${MODEL_PATH:-/data/yingxi/RLinf_RoboFAPE/logs/pushcube_sft_wrist_filtered_conservative_20260823/train_40k_fresh_0_3/pushcube_sft_wrist_filtered_conservative_40k_fresh_0_3/checkpoints/global_step_35000/actor}"
    ;;
  peg_insertion)
    CONFIG_NAME="${CONFIG_NAME:-maniskill_async_ppo_peg_insertion_pi05}"
    MODEL_PATH="${MODEL_PATH:-/data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor}"
    ;;
  *)
    echo "Unsupported TASK=${TASK}. Add a config mapping in this launcher." >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Set PYTHON_BIN to the RLinf environment python." >&2
  exit 1
fi
RAY_BIN="${RAY_BIN:-$(dirname "${PYTHON_BIN}")/ray}"
if [[ ! -x "${RAY_BIN}" ]]; then
  echo "Set RAY_BIN to the RLinf environment ray executable." >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

export EMBODIED_PATH
export REPO_PATH
export MODEL_PATH
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
ROBOMETER_PORT="${ROBOMETER_PORT:-8001}"
ROBOMETER_SERVER_URL="${ROBOMETER_SERVER_URL:-http://127.0.0.1:${ROBOMETER_PORT}}"
LOG_DIR="${LOG_DIR:-${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${TASK}_rl_robometer_absolute_env32_step200}"
mkdir -p "${LOG_DIR}"

ROBOMETER_SERVER_PID=""

_stop_robometer_server() {
  if [[ -n "${ROBOMETER_SERVER_PID}" ]] && kill -0 "${ROBOMETER_SERVER_PID}" >/dev/null 2>&1; then
    kill "${ROBOMETER_SERVER_PID}" >/dev/null 2>&1 || true
    wait "${ROBOMETER_SERVER_PID}" >/dev/null 2>&1 || true
  fi
}

_rl_scoped_ray_kill() {
  pkill -9 -f "gcs_server.*--gcs_server_port=${RL_RAY_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "raylet.*--gcs-address=[^ ]*:${RL_RAY_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "dashboard.*--gcs-address=[^ ]*:${RL_RAY_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "ray.util.client.server.*--address=[^ ]*:${RL_RAY_PORT}" >/dev/null 2>&1 || true
  sleep 2
}

_cleanup() {
  _stop_robometer_server
  _rl_scoped_ray_kill
}

_robometer_health_check() {
  "${PYTHON_BIN}" - "${ROBOMETER_SERVER_URL}" <<'PY'
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        body = response.read().decode("utf-8", errors="replace")
except Exception as exc:
    print(f"Robometer health check failed for {url}: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)
print(body)
PY
}

if [[ "${START_ROBOMETER_SERVER:-false}" == "true" ]]; then
  ROBOMETER_GPU_ID="${ROBOMETER_GPU_ID:-2}"
  ROBOMETER_CKPT="${ROBOMETER_CKPT:-/data/yingxi/robometer/checkpoint-400-5tasks}"
  ROBOMETER_REPO="${ROBOMETER_REPO:-/data/yingxi/RoboFPE/robometer}"
  ROBOMETER_LOG="${LOG_DIR}/robometer_server.log"

  if _robometer_health_check >"/tmp/robometer_existing_health_${ROBOMETER_PORT}.json" 2>/dev/null; then
    if [[ "${ROBOMETER_REUSE_EXISTING_SERVER:-false}" == "true" ]]; then
      echo "Reusing existing Robometer server at ${ROBOMETER_SERVER_URL}: $(cat "/tmp/robometer_existing_health_${ROBOMETER_PORT}.json")"
    else
      echo "Robometer server is already healthy at ${ROBOMETER_SERVER_URL}." >&2
      echo "Refusing to start training against an ambiguous pre-existing server. Set START_ROBOMETER_SERVER=false or ROBOMETER_REUSE_EXISTING_SERVER=true to reuse it." >&2
      exit 1
    fi
  else
    (
      cd "${REPO_PATH}"
      ROBOMETER_GPU_ID="${ROBOMETER_GPU_ID}" \
      ROBOMETER_CKPT="${ROBOMETER_CKPT}" \
      ROBOMETER_REPO="${ROBOMETER_REPO}" \
      ROBOMETER_PORT="${ROBOMETER_PORT}" \
      bash run_robometer_4b_server.sh
    ) >"${ROBOMETER_LOG}" 2>&1 &
    ROBOMETER_SERVER_PID=$!
    echo "${ROBOMETER_SERVER_PID}" > "${LOG_DIR}/robometer_server.pid"
    sleep "${ROBOMETER_STARTUP_WAIT_S:-20}"

    ROBOMETER_HEALTHY=false
    for _ in $(seq 1 "${ROBOMETER_HEALTH_RETRIES:-24}"); do
      if ! kill -0 "${ROBOMETER_SERVER_PID}" >/dev/null 2>&1; then
        echo "Robometer server process exited during startup. Last log lines:" >&2
        tail -n 120 "${ROBOMETER_LOG}" >&2 || true
        exit 1
      fi
      if _robometer_health_check; then
        ROBOMETER_HEALTHY=true
        break
      fi
      sleep 5
    done
    if [[ "${ROBOMETER_HEALTHY}" != "true" ]]; then
      echo "Robometer server did not become healthy at ${ROBOMETER_SERVER_URL}. Last log lines:" >&2
      tail -n 120 "${ROBOMETER_LOG}" >&2 || true
      exit 1
    fi
  fi
elif [[ "${ROBOMETER_REQUIRE_HEALTH:-true}" == "true" ]]; then
  _robometer_health_check
fi

export CUDA_VISIBLE_DEVICES="${RL_GPU_IDS:-0,1}"
export RL_RAY_PORT="${RL_RAY_PORT:-6386}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
export RAY_DASHBOARD_AGENT_PORT="${RAY_DASHBOARD_AGENT_PORT:-52372}"
export RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-10040}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"
export RAY_TMPDIR="${RAY_TMPDIR:-/data/yingxi/ray_tmp_${TASK}_rl_${RL_RAY_PORT}}"
export RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-13000}"
export RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-13399}"
export RAY_ADDRESS="127.0.0.1:${RL_RAY_PORT}"

# Keep this launcher isolated from any concurrent Ray jobs on the host. Never use
# bare `ray stop`; it cannot target one cluster and would kill other jobs.
ulimit -n 1048576 2>/dev/null || true
mkdir -p "${RAY_TMPDIR}"
_rl_scoped_ray_kill
trap _cleanup EXIT
"${RAY_BIN}" start --head \
  --port="${RL_RAY_PORT}" \
  --ray-client-server-port="${RAY_CLIENT_SERVER_PORT}" \
  --num-cpus="${RAY_NUM_CPUS}" \
  --temp-dir="${RAY_TMPDIR}" \
  --dashboard-host=127.0.0.1 \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_PORT}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT}"

CMD=(
  "${PYTHON_BIN}" "${SRC_FILE}"
  --config-path "${EMBODIED_PATH}/config"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "runner.logger.experiment_name=${TASK}_rl_robometer_absolute"
  "actor.model.model_path=${MODEL_PATH}"
  "rollout.model.model_path=${MODEL_PATH}"
  "reward.model.server_url=${ROBOMETER_SERVER_URL}"
)
CMD+=("$@")

printf "Running:"
printf " %q" "${CMD[@]}"
printf "\n"
printf "Running:" > "${LOG_DIR}/run_task_rl_async.cmd"
printf " %q" "${CMD[@]}" >> "${LOG_DIR}/run_task_rl_async.cmd"
printf "\n" >> "${LOG_DIR}/run_task_rl_async.cmd"
"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/train_async.log"
