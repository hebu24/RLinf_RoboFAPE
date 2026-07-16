#!/usr/bin/env bash
set -euo pipefail

cd /opt/yingxi/RLinf_RoboFAPE

export PATH=/opt/kairan/envs/rlinf/bin:$PATH
export RAY_TMPDIR=/tmp/ray_sft_wrist
export SFT_RAY_PORT="${SFT_RAY_PORT:-6379}"
export CUDA_LAUNCH_BLOCKING=1

DATA_DIR="${DATA_DIR:-/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
if ((${#GPU_ID_ARRAY[@]} == 0)); then
  echo "GPU_IDS must contain at least one GPU id." >&2
  exit 1
fi
for gpu_id in "${GPU_ID_ARRAY[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "GPU_IDS must be a comma-separated list of physical GPU ids, got: ${GPU_IDS}" >&2
    exit 1
  fi
done

SFT_COMPONENT_PLACEMENT="${GPU_IDS}"
if ((${#GPU_ID_ARRAY[@]} == 1)); then
  SFT_COMPONENT_PLACEMENT="${GPU_ID_ARRAY[0]}"
else
  consecutive=1
  prev_gpu_id="${GPU_ID_ARRAY[0]}"
  for gpu_id in "${GPU_ID_ARRAY[@]:1}"; do
    if ((gpu_id != prev_gpu_id + 1)); then
      consecutive=0
      break
    fi
    prev_gpu_id="${gpu_id}"
  done
  if ((consecutive)); then
    last_gpu_index=$((${#GPU_ID_ARRAY[@]} - 1))
    SFT_COMPONENT_PLACEMENT="${GPU_ID_ARRAY[0]}-${GPU_ID_ARRAY[${last_gpu_index}]}"
  fi
fi

export SFT_COMPONENT_PLACEMENT
# SFT runs its OWN detached Ray head on SFT_RAY_PORT (default 6379), isolated from
# any eval sweep cluster (port 6380). Never a bare `ray stop` (that kills ALL ray on
# the host, incl. eval); teardown is scoped to SFT_RAY_PORT only.
export RAY_ADDRESS="127.0.0.1:${SFT_RAY_PORT}"
unset CUDA_VISIBLE_DEVICES

_sft_scoped_ray_kill() {
  pkill -9 -f "gcs_server.*--gcs_server_port=${SFT_RAY_PORT}"  >/dev/null 2>&1 || true
  pkill -9 -f "raylet.*--gcs-address=[^ ]*:${SFT_RAY_PORT}"    >/dev/null 2>&1 || true
  pkill -9 -f "dashboard.*--gcs-address=[^ ]*:${SFT_RAY_PORT}" >/dev/null 2>&1 || true
  sleep 2
}

echo "Physical GPU_IDS=${GPU_IDS}; RLinf cluster.component_placement=${SFT_COMPONENT_PLACEMENT}"
echo "SFT Ray head: port=${SFT_RAY_PORT}, temp_dir=${RAY_TMPDIR}, RAY_ADDRESS=${RAY_ADDRESS}"
echo "CUDA_VISIBLE_DEVICES is unset for the Ray driver; RLinf workers set it from placement."

if [[ "${SFT_STOP_RAY_BEFORE_START:-1}" == "1" ]]; then
  echo "Clearing stale SFT Ray on port ${SFT_RAY_PORT} (scoped; does not touch other clusters)."
  _sft_scoped_ray_kill
fi

# On SFT exit (incl. ray-start failure or interrupt), tear down ONLY the SFT head
# (port SFT_RAY_PORT), never other clusters. Armed before `ray start` so a failed
# start is cleaned up too.
trap '_sft_scoped_ray_kill' EXIT

# Start the SFT detached head. The driver + workers attach to it via RAY_ADDRESS.
ray start --head --port="${SFT_RAY_PORT}" --temp-dir="${RAY_TMPDIR}"

bash examples/sft/run_vla_sft.sh \
  peg_insertion_sft_openpi_pi05_wrist \
  data.train_data_paths="${DATA_DIR}" \
  cluster.component_placement="{actor\\,env\\,rollout:${SFT_COMPONENT_PLACEMENT}}" \
  "${RESUME_DIR:+runner.resume_dir=${RESUME_DIR}}"
