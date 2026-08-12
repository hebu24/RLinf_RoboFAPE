#!/bin/bash
# Independent sweep ep50 eval on ckpt400 or 4b run checkpoints.
# New output dir (rl_eval_sweep_ep50_clean) — does NOT touch existing rl_eval_sweep_ep50.
# usage: bash _sweep_clean.sh {ckpt400|4b}
CASE="$1"
cd /data/yingxi/RLinf_RoboFAPE
source /data/yingxi/kairan/envs/rlinf/bin/activate rlinf
export RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/data/yingxi/tmp HF_HOME=/data/yingxi/.cache/huggingface
NORM=/data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000
if [ "$CASE" = "ckpt400" ]; then
  RUN=/data/yingxi/RLinf_RoboFAPE/logs/20260809-01:56:53-peg_insertion_rl_async_absolute_independent_window_bonus0_fresh_ckpt400_gpu57_notcolocate
  GPUS=4,5; PORT=6540; DASH=8290
elif [ "$CASE" = "4b" ]; then
  RUN=/data/yingxi/RLinf_RoboFAPE/logs/20260809-01:20:10-peg_insertion_rl_async_absolute_independent_window_bonus0_fresh_robometer4b_gpu01_notcolocate
  GPUS=6,7; PORT=6541; DASH=8291
else echo "usage: $0 {ckpt400|4b}"; exit 1; fi
echo "[$CASE] sweep start $(date) ckpt=$RUN gpus=$GPUS ray=$PORT"
exec /data/yingxi/kairan/envs/rlinf/bin/python run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
  --ray-port $PORT --ray-dashboard-port $DASH --ray-tmp-dir /data/yingxi/ray_es \
  --checkpoint-dir "$RUN/peg_insertion_async_ppo_pi05_robometer/checkpoints" \
  --norm-stats-source "$NORM" \
  --output-dir "$RUN/peg_insertion_async_ppo_pi05_robometer/rl_eval_sweep_ep50_clean" \
  --num-eval-episodes 50 --num-envs 10 --gpu-ids $GPUS \
  --action-scale 1.0 --seed 0 \
  --hydra-override env.eval.shared_reset_seed=true \
  --continue-on-error --resume
