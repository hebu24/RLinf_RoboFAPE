#!/usr/bin/env bash
set -u
set -o pipefail

PY=/data/yingxi/robometer/failure_detection_env/bin/python
ROOT=/data/yingxi/RLinf_RoboFAPE
COL=$ROOT/run_train/robofpe_sft_data/collect_sft_data.py
OUT=/data/yingxi/datasets/robofpe_sft/full_2tasks_20260920
LOG=$OUT/full_pipeline.log
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

export MS_ASSET_DIR=/data/yingxi/robofac
export CUDA_PYTHON_LIB_ROOT=/data/yingxi/robofac/lib/python3.10/site-packages/nvidia
export PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
cd "$ROOT"

collect_task() {
  local task="$1" seed="$2"
  local dir="$OUT/$task"
  mkdir -p "$dir"
  if [ -f "$dir/collection_manifest.json" ]; then
    echo "COLLECT_SKIP_EXISTING $task"
    return 0
  fi
  echo "COLLECT_START $task seed=$seed $(date -Is)"
  "$PY" "$COL" collect \
    --task-id "$task" --num-traj 3200 --output-dir "$dir" \
    --robot-uids panda_wristcam --success-only \
    --randomize-initial-poses --randomize-wrist-camera --randomize-render-camera --randomize-lighting \
    --save-video --sim-backend gpu --num-workers 4 --gpu-ids 0,1,2,3 \
    --max-attempts-per-traj 100 --solver-timeout 120 \
    --worker-timeout 172800 --seed "$seed" --no-convert
  local rc=$?
  echo "COLLECT_DONE $task rc=$rc $(date -Is)"
  return "$rc"
}

convert_task() {
  local task="$1"
  local dir="$OUT/$task"
  echo "CONVERT_START $task $(date -Is)"
  "$PY" "$COL" convert --input "$dir/shards" --dataset-dir "$dir/lerobot" \
    --overwrite --num-convert-workers 48
  local rc=$?
  echo "CONVERT_DONE $task rc=$rc $(date -Is)"
  return "$rc"
}

collect_task PegInsertionVertical-v1 2026092000 || exit $?
convert_task PegInsertionVertical-v1 &
vertical_convert_pid=$!

collect_task PullCubeTool-golf 2026093000 || exit $?
convert_task PullCubeTool-golf &
pull_convert_pid=$!

wait "$vertical_convert_pid"; vrc=$?
wait "$pull_convert_pid"; prc=$?
echo "FULL_PIPELINE_DONE vertical_rc=$vrc pull_rc=$prc $(date -Is)"
exit $((vrc || prc))
