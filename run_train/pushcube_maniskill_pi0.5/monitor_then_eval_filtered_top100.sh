#!/usr/bin/env bash
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE

BASELINE_CKPT_DIR="${BASELINE_CKPT_DIR:-/data/yingxi/RLinf_RoboFAPE/logs/20260829-21:25:01-pushcube_sft_openpi_pi05_wrist-3200/pushcube_sft_wrist_filtered_top100_baseline/checkpoints}"
OURS_CKPT_DIR="${OURS_CKPT_DIR:-/data/yingxi/RLinf_RoboFAPE/logs/20260829-21:25:13-pushcube_sft_openpi_pi05_wrist-3200/pushcube_sft_wrist_filtered_top100_ours/checkpoints}"
BASELINE_OUT_DIR="${BASELINE_OUT_DIR:-$(dirname "${BASELINE_CKPT_DIR}")/sweep_seed0_7_eval100}"
OURS_OUT_DIR="${OURS_OUT_DIR:-$(dirname "${OURS_CKPT_DIR}")/sweep_seed0_7_eval100}"
MONITOR_DIR="${MONITOR_DIR:-/data/yingxi/RLinf_RoboFAPE/logs/pushcube_sft_filtering_top100_eval_monitor}"

FINAL_STEP="${FINAL_STEP:-10000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABLE_SECONDS="${STABLE_SECONDS:-60}"
SWEEP_SEEDS="${SWEEP_SEEDS:-0-7}"
SWEEP_NUM_EPISODES="${SWEEP_NUM_EPISODES:-100}"
SWEEP_NUM_ENVS="${SWEEP_NUM_ENVS:-10}"
SWEEP_MAX_EPISODE_STEPS="${SWEEP_MAX_EPISODE_STEPS:-200}"
PYTHON_BIN="${PYTHON_BIN:-/data/yingxi/RLinf_RoboFAPE/.venv/bin/python}"

mkdir -p "${MONITOR_DIR}" "${BASELINE_OUT_DIR}" "${OURS_OUT_DIR}"
exec > >(tee -a "${MONITOR_DIR}/monitor.log") 2>&1

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

actor_dir() {
  local ckpt_dir="$1"
  printf '%s/global_step_%s/actor' "${ckpt_dir}" "${FINAL_STEP}"
}

ckpt_complete() {
  local actor="$1"
  [[ -d "${actor}" ]] || return 1
  [[ -e "${actor}/dcp_checkpoint" ]] || return 1
  [[ -e "${actor}/model_state_dict" ]] || return 1
  [[ -e "${actor}/trainer_state.json" ]] || return 1
}

ckpt_stable() {
  local actor="$1"
  local latest now age
  latest="$(find "${actor}" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -1 || true)"
  [[ -n "${latest}" ]] || return 1
  now="$(date +%s)"
  age="$(awk -v now="${now}" -v latest="${latest}" 'BEGIN { printf "%d", now - latest }')"
  [[ "${age}" -ge "${STABLE_SECONDS}" ]]
}

training_active() {
  pgrep -f 'train_vla_sft.py.*pushcube_sft_wrist_filtered_top100_(baseline|ours)' >/dev/null 2>&1 && return 0
  pgrep -f 'ray::FSDPVlaSftWorker.run_training' >/dev/null 2>&1 && return 0
  return 1
}

wait_for_training_done() {
  local baseline_actor ours_actor
  baseline_actor="$(actor_dir "${BASELINE_CKPT_DIR}")"
  ours_actor="$(actor_dir "${OURS_CKPT_DIR}")"
  log "waiting for final checkpoints: ${baseline_actor} and ${ours_actor}"
  while true; do
    if ckpt_complete "${baseline_actor}" && ckpt_stable "${baseline_actor}" \
      && ckpt_complete "${ours_actor}" && ckpt_stable "${ours_actor}"; then
      if training_active; then
        log "final checkpoints are complete, but training processes are still active; waiting"
      else
        log "final checkpoints complete and training processes exited"
        return 0
      fi
    else
      log "final checkpoints not ready yet"
    fi
    sleep "${POLL_SECONDS}"
  done
}

launch_eval() {
  local name="$1"
  local ckpt_dir="$2"
  local out_dir="$3"
  local gpu_ids="$4"
  local ray_port="$5"
  local dashboard_port="$6"
  local dashboard_agent_port="$7"
  local client_port="$8"
  local min_worker_port="$9"
  local max_worker_port="${10}"
  local log_path="${MONITOR_DIR}/${name}.eval.log"
  local pid_path="${MONITOR_DIR}/${name}.eval.pid"

  log "starting ${name} eval: gpu=${gpu_ids}, ckpt=${ckpt_dir}, out=${out_dir}, episodes/seed=${SWEEP_NUM_EPISODES}"
  (
    cd /data/yingxi/RLinf_RoboFAPE
    exec "${PYTHON_BIN}" run_train/eval_checkpoint/sweep_pushcube_wrist.py \
      --checkpoint-dir "${ckpt_dir}" \
      --output-dir "${out_dir}" \
      --resume \
      --continue-on-error \
      --pin-checkpoints \
      --gpu-ids "${gpu_ids}" \
      --seeds "${SWEEP_SEEDS}" \
      --num-eval-episodes "${SWEEP_NUM_EPISODES}" \
      --num-envs "${SWEEP_NUM_ENVS}" \
      --max-episode-steps "${SWEEP_MAX_EPISODE_STEPS}" \
      --ray-port "${ray_port}" \
      --ray-dashboard-port "${dashboard_port}" \
      --ray-dashboard-agent-port "${dashboard_agent_port}" \
      --ray-client-server-port "${client_port}" \
      --ray-min-worker-port "${min_worker_port}" \
      --ray-max-worker-port "${max_worker_port}" \
      --ray-object-store-memory 3000000000 \
      --ray-temp-dir "/tmp/ray_eval_${name}_${ray_port}" \
      --no-save-video
  ) >"${log_path}" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_path}"
  log "${name} eval pid=${pid}; log=${log_path}"
}

if [[ -e "${MONITOR_DIR}/eval_started.marker" ]]; then
  log "eval already started according to ${MONITOR_DIR}/eval_started.marker; exiting"
  exit 0
fi

wait_for_training_done
date > "${MONITOR_DIR}/eval_started.marker"

launch_eval baseline "${BASELINE_CKPT_DIR}" "${BASELINE_OUT_DIR}" 0,1 6387 8267 52373 10041 13400 13799
launch_eval ours "${OURS_CKPT_DIR}" "${OURS_OUT_DIR}" 2,3 6389 8268 52374 10051 13800 14199

baseline_pid="$(cat "${MONITOR_DIR}/baseline.eval.pid")"
ours_pid="$(cat "${MONITOR_DIR}/ours.eval.pid")"
baseline_status=0
ours_status=0
wait "${baseline_pid}" || baseline_status=$?
wait "${ours_pid}" || ours_status=$?
log "eval finished: baseline_status=${baseline_status}, ours_status=${ours_status}"

if [[ "${baseline_status}" -ne 0 || "${ours_status}" -ne 0 ]]; then
  exit 1
fi
