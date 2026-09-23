#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/yingxi/RLinf_RoboFAPE
PY=/data/yingxi/robometer/failure_detection_env/bin/python
OUT=/data/yingxi/datasets/robofpe_sft/robofpe_5tasks_20260920
export MS_ASSET_DIR=/data/yingxi/robofac
export PYTHONPATH="$ROOT:$ROOT/run_train/robofpe_sft_data:/data/yingxi/RoboFPE/mani_envs${PYTHONPATH:+:$PYTHONPATH}"
GPUS=0,1,2,3
LOG_ROOT="$OUT/logs_5tasks_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
tasks=(PegInsertionVertical-v1 PlugCharger-v1 UprightStack-v1 PegInsertionSide-v1 PullCubeTool-golf)
declare -A seeds=([PegInsertionVertical-v1]=100000 [PlugCharger-v1]=200000 [UprightStack-v1]=300000 [PegInsertionSide-v1]=400000 [PullCubeTool-golf]=500000)
collect_task() {
  local task=$1 out="$OUT/${1}_render_wrist"
  "$PY" "$ROOT/run_train/robofpe_sft_data/collect_sft_data.py" collect \
    --task-id "$task" --num-traj 3200 --output-dir "$out" --robot-uids panda_wristcam \
    --success-only --randomize-initial-poses --randomize-wrist-camera --randomize-render-camera \
    --randomize-lighting --save-video --sim-backend gpu --num-workers 4 --gpu-ids "$GPUS" \
    --max-attempts-per-traj 100 --seed "${seeds[$task]}" --no-convert \
    >"$LOG_ROOT/${task}.collect.log" 2>&1
}
convert_task() {
  local task=$1 out="$OUT/${1}_render_wrist"
  "$PY" "$ROOT/run_train/robofpe_sft_data/collect_sft_data.py" convert \
    --task-id "$task" --input "$out/shards" --dataset-dir "$out/lerobot" \
    --overwrite --num-convert-workers 48 >"$LOG_ROOT/${task}.convert.log" 2>&1
  "$PY" "$ROOT/run_train/robofpe_sft_data/collect_sft_data.py" validate \
    --dataset-dir "$out/lerobot" --raw-input "$out/shards" >>"$LOG_ROOT/${task}.convert.log" 2>&1
}
collect_task "${tasks[0]}"
for ((i=0; i<${#tasks[@]}; i++)); do
  task=${tasks[$i]}
  echo "[$(date -Is)] collection complete: $task"
  convert_task "$task" & convert_pid=$!
  if (( i + 1 < ${#tasks[@]} )); then
    next=${tasks[$((i + 1))]}
    echo "[$(date -Is)] collecting $next while converting $task"
    collect_task "$next"
    wait "$convert_pid"
  else
    wait "$convert_pid"
  fi
done
