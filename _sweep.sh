#!/bin/bash
# Parameterized sweep ep50 eval. Independent run dir (no overwrite).
# usage: SUFFIX=v3 SEED=1 PORT_BASE=6542 DASH_BASE=8292 bash _sweep.sh {ckpt400|4b}
#   defaults: SUFFIX=clean SEED=0 PORT_BASE=6540 DASH_BASE=8290
CASE="$1"
SUFFIX="${SUFFIX:-clean}"
SEED="${SEED:-0}"
PORT_BASE="${PORT_BASE:-6540}"
DASH_BASE="${DASH_BASE:-8290}"
cd /data/yingxi/RLinf_RoboFAPE
source /data/yingxi/kairan/envs/rlinf/bin/activate rlinf
export RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data/yingxi/tmp HF_HOME=/data/yingxi/.cache/huggingface
NORM=/data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000
if [ "$CASE" = "ckpt400" ]; then
  RUN=/data/yingxi/RLinf_RoboFAPE/logs/20260809-01:56:53-peg_insertion_rl_async_absolute_independent_window_bonus0_fresh_ckpt400_gpu57_notcolocate
  GPUS=4,5; PORT=$PORT_BASE; DASH=$DASH_BASE
elif [ "$CASE" = "4b" ]; then
  RUN=/data/yingxi/RLinf_RoboFAPE/logs/20260809-01:20:10-peg_insertion_rl_async_absolute_independent_window_bonus0_fresh_robometer4b_gpu01_notcolocate
  GPUS=6,7; PORT=$((PORT_BASE+1)); DASH=$((DASH_BASE+1))
else echo "usage: $0 {ckpt400|4b}"; exit 1; fi
echo "[$CASE] sweep start $(date) suffix=$SUFFIX seed=$SEED gpus=$GPUS ray=$PORT"
exec /data/yingxi/kairan/envs/rlinf/bin/python run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
  --ray-port $PORT --ray-dashboard-port $DASH --ray-tmp-dir /data/yingxi/ray_es \
  --checkpoint-dir "$RUN/peg_insertion_async_ppo_pi05_robometer/checkpoints" \
  --norm-stats-source "$NORM" \
  --output-dir "$RUN/peg_insertion_async_ppo_pi05_robometer/rl_eval_sweep_ep50_$SUFFIX" \
  --num-eval-episodes 50 --num-envs 10 --gpu-ids $GPUS \
  --action-scale 1.0 --seed $SEED \
  --hydra-override env.eval.shared_reset_seed=true \
  --continue-on-error --resume
