#!/usr/bin/env bash
set -euo pipefail

cd /opt/yingxi/RLinf_RoboFAPE

export PATH=/opt/kairan/envs/rlinf/bin:$PATH
export RAY_TMPDIR=/opt/yingxi/RLinf_RoboFAPE/ray_tmp_wrist
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
unset RAY_ADDRESS
unset CUDA_VISIBLE_DEVICES

echo "Physical GPU_IDS=${GPU_IDS}; RLinf cluster.component_placement=${SFT_COMPONENT_PLACEMENT}"
echo "CUDA_VISIBLE_DEVICES is unset for the Ray driver; RLinf workers set it from placement."

if [[ "${SFT_STOP_RAY_BEFORE_START:-1}" == "1" ]]; then
  echo "Stopping existing Ray before SFT so Ray redetects physical GPU resources."
  ray stop --force >/dev/null 2>&1 || true
fi

bash examples/sft/run_vla_sft.sh \
  peg_insertion_sft_openpi_pi05_wrist \
  data.train_data_paths="${DATA_DIR}" \
  cluster.component_placement="{actor\\,env\\,rollout:${SFT_COMPONENT_PLACEMENT}}" \
  "${RESUME_DIR:+runner.resume_dir=${RESUME_DIR}}"
