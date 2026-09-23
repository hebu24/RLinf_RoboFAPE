#!/usr/bin/env bash
# Parallel PushCube-v1 step20k SFT rollout rendering.
#
# Defaults:
#   - checkpoint: PushCube wrist SFT global_step_20000 actor
#   - seeds: 0-7
#   - 50 rollouts per seed
#   - NUM_ENVS=1 so every rollout is saved as an independent MP4
#   - one Ray head per seed, isolated by Ray/head/dashboard/worker ports
#
# Videos:
#   ${OUT_ROOT}/seed_<seed>/video/eval/seed_<seed>/<trajectory_index>.mp4
#
# Failures:
#   Check ${OUT_ROOT}/seed_<seed>/trajectory_metrics.json.
#   Entries with success_once[index] == false/0 correspond to failed videos.
set -euo pipefail

REPO_PATH="${REPO_PATH:-/data/yingxi/RLinf_RoboFAPE}"
cd "${REPO_PATH}"

VENV_DIR="${VENV_DIR:-${REPO_PATH}/.venv}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_PATH}/logs/pushcube_sft_wrist_filtered_conservative_20260823/train_40k_fresh_0_3/pushcube_sft_wrist_filtered_conservative_40k_fresh_0_3/checkpoints/global_step_20000/actor}"
OUT_ROOT="${OUT_ROOT:-${REPO_PATH}/logs/pushcube_sft_step20k_fail_videos_parallel}"

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
if [[ -z "${GPU_IDS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  else
    GPU_IDS="0"
  fi
fi
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-50}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-180}"
BASE_RAY_PORT="${BASE_RAY_PORT:-6380}"
BASE_DASHBOARD_PORT="${BASE_DASHBOARD_PORT:-8265}"
BASE_RAY_WORKER_PORT="${BASE_RAY_WORKER_PORT:-20000}"
RAY_WORKER_PORT_SPAN="${RAY_WORKER_PORT_SPAN:-1000}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"
MAX_VIDEOS="${MAX_VIDEOS:-50}"
EVAL_RAY_INCLUDE_DASHBOARD="${EVAL_RAY_INCLUDE_DASHBOARD:-false}"

if [[ ! -x "${VENV_DIR}/bin/python" || ! -x "${VENV_DIR}/bin/ray" ]]; then
  echo "Missing Python or Ray in VENV_DIR=${VENV_DIR}" >&2
  exit 1
fi

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint actor dir does not exist: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ "${NUM_ENVS}" != "1" ]]; then
  echo "NUM_ENVS must stay 1 so each trajectory is rendered as an independent video." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a GPU_ARRAY <<< "${GPU_IDS}"

if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-${#GPU_ARRAY[@]}}"
if ! [[ "${MAX_PARALLEL_JOBS}" =~ ^[0-9]+$ ]] || (( MAX_PARALLEL_JOBS < 1 )); then
  echo "MAX_PARALLEL_JOBS must be a positive integer." >&2
  exit 1
fi

echo "REPO_PATH=${REPO_PATH}"
echo "VENV_DIR=${VENV_DIR}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "SEEDS=${SEEDS}"
echo "GPU_IDS=${GPU_IDS}"
echo "NUM_EVAL_EPISODES=${NUM_EVAL_EPISODES}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}"
echo "MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}"
echo "BASE_RAY_WORKER_PORT=${BASE_RAY_WORKER_PORT}"
echo "RAY_WORKER_PORT_SPAN=${RAY_WORKER_PORT_SPAN}"
echo "EVAL_RAY_INCLUDE_DASHBOARD=${EVAL_RAY_INCLUDE_DASHBOARD}"
echo

pids=()
pid_seeds=()
failed=0

wait_one() {
  local pid="${pids[0]}"
  local seed="${pid_seeds[0]}"
  if wait "${pid}"; then
    echo "[done] seed=${seed}"
  else
    echo "[failed] seed=${seed}; see ${OUT_ROOT}/seed_${seed}.launcher.log" >&2
    failed=1
  fi
  pids=("${pids[@]:1}")
  pid_seeds=("${pid_seeds[@]:1}")
}

idx=0
for seed in "${SEED_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
  ray_port=$((BASE_RAY_PORT + idx))
  dashboard_port=$((BASE_DASHBOARD_PORT + idx))
  worker_min_port=$((BASE_RAY_WORKER_PORT + idx * RAY_WORKER_PORT_SPAN))
  worker_max_port=$((worker_min_port + RAY_WORKER_PORT_SPAN - 1))
  log_dir="${OUT_ROOT}/seed_${seed}"
  launcher_log="${OUT_ROOT}/seed_${seed}.launcher.log"
  ray_tmp_dir="/tmp/ray_eval_pushcube_step20k_seed_${seed}_${ray_port}"

  echo "[launch] seed=${seed} gpu=${gpu} ray_port=${ray_port} worker_ports=${worker_min_port}-${worker_max_port} log_dir=${log_dir}"
  (
    set -euo pipefail
    VENV_DIR="${VENV_DIR}" \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    LOG_DIR="${log_dir}" \
    GPU_IDS="${gpu}" \
    NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES}" \
    NUM_ENVS="${NUM_ENVS}" \
    MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS}" \
    SEED="${seed}" \
    EVAL_RAY_PORT="${ray_port}" \
    EVAL_RAY_DASHBOARD_PORT="${dashboard_port}" \
    EVAL_RAY_INCLUDE_DASHBOARD="${EVAL_RAY_INCLUDE_DASHBOARD}" \
    EVAL_RAY_MIN_WORKER_PORT="${worker_min_port}" \
    EVAL_RAY_MAX_WORKER_PORT="${worker_max_port}" \
    RAY_TMP_DIR="${ray_tmp_dir}" \
    MANAGE_RAY=true \
    FORCE_RESTART_RAY=true \
    SAVE_VIDEO="${SAVE_VIDEO}" \
    bash run_train/eval_checkpoint/run_pushcube_wrist.sh \
      --save-episode-metrics \
      "env.eval.video_cfg.max_videos=${MAX_VIDEOS}" \
      "env.group_name=EnvGroupEvalSeed${seed}" \
      "rollout.group_name=RolloutGroupEvalSeed${seed}"
  ) > "${launcher_log}" 2>&1 &

  pids+=("$!")
  pid_seeds+=("${seed}")
  idx=$((idx + 1))

  if (( ${#pids[@]} >= MAX_PARALLEL_JOBS )); then
    wait_one
  fi
done

while (( ${#pids[@]} > 0 )); do
  wait_one
done

echo
echo "Outputs:"
for seed in "${SEED_ARRAY[@]}"; do
  echo "  seed ${seed}: ${OUT_ROOT}/seed_${seed}"
done

exit "${failed}"
