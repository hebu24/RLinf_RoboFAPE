#!/usr/bin/env bash
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE

export PYTHON_BIN=/data/yingxi/robometer/failure_detection_env/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_PYTHON_LIB_ROOT=/data/yingxi/robofac/lib/python3.10/site-packages/nvidia
export CUDA_LIB_DIRS="$(find "$CUDA_PYTHON_LIB_ROOT" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:')"
export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MS_ASSET_DIR=/data/yingxi/robofac
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export PYTHONUNBUFFERED=1

LOG_FILE=/data/yingxi/datasets/robofpe_sft/ind_raw_collect_20260902.log
mkdir -p /data/yingxi/datasets/robofpe_sft
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== RAW COLLECT PIPELINE START $(date -Is) ====="
echo "HOST $(hostname)"
nvidia-smi --query-gpu=index,name --format=csv,noheader

run_collect() {
  local task="$1"
  local seed="$2"
  local out="/data/yingxi/datasets/robofpe_sft/${task}_render_wrist"
  echo "===== COLLECT RAW $task seed=$seed out=$out $(date -Is) ====="
  "$PYTHON_BIN" run_train/robofpe_sft_data/collect_sft_data.py collect \
    --task-id "$task" \
    --num-traj 3200 \
    --output-dir "$out" \
    --robot-uids panda_wristcam \
    --success-only \
    --randomize-initial-poses \
    --randomize-wrist-camera \
    --randomize-render-camera \
    --randomize-lighting \
    --save-video \
    --sim-backend gpu \
    --num-workers 16 \
    --gpu-ids 0,1,2,3 \
    --max-attempts-per-traj 100 \
    --seed "$seed" \
    --no-convert
  echo "===== RAW DONE $task $(date -Is) ====="
  find "$out/shards" -name '*.h5' | wc -l | xargs -I{} echo "$task raw_h5={}"
}

# This batch intentionally excludes PlugCharger-v1. StackCube-v1 is collected last
# with the same raw collection settings as the preceding tasks.
run_collect PegInsertionSide-v1 30260902
run_collect StackCube-v1 50260902

echo "===== RAW COLLECT PIPELINE DONE $(date -Is) ====="
