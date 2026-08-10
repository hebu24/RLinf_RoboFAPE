#!/usr/bin/env bash
# Continuous eval loop for Run A (robometer4b) on GPU 3.
# Picks up new checkpoints every 5 min. Uses --resume to skip already-done ones.
set -euo pipefail
cd /data/yingxi/RLinf_RoboFAPE
export RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

CKPT_DIR="logs/20260809-01:20:10-peg_insertion_rl_async_absolute_independent_window_bonus0_fresh_robometer4b_gpu01_notcolocate/peg_insertion_async_ppo_pi05_robometer/checkpoints"
OUT_DIR="logs/20260809-01:20:10-peg_insertion_rl_async_absolute_independent_window_bonus0_fresh_robometer4b_gpu01_notcolocate/peg_insertion_async_ppo_pi05_robometer/rl_eval_sweep_ode_ep50"
NORM_SRC="/data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor"
PY=/data/yingxi/kairan/envs/rlinf/bin/python

echo "=== GPU3 continuous eval loop for Run A (robometer4b) ==="
echo "Starting continuous loop."

while true; do
  echo ""
  echo "=== [GPU3] Sweep run at $(date) ==="
  $PY run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
    --checkpoint-dir "$CKPT_DIR" \
    --output-dir "$OUT_DIR" \
    --gpu-ids 3 \
    --num-eval-episodes 50 \
    --num-envs 10 \
    --no-save-video \
    --norm-stats-source "$NORM_SRC" \
    --hydra-override actor.model.openpi.noise_method=flow_ode \
    --ray-port 6391 \
    --ray-dashboard-port 8271 \
    --ray-tmp-dir /data/yingxi/ray_es \
    --resume \
    --continue-on-error \
    || echo "[GPU3] Sweep exited with error, will retry in 60s"
  echo "[GPU3] Sweep done. Results so far:"
  find "$OUT_DIR" -name 'evaluation_summary.json' -exec python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
m=d['metrics']
step=sys.argv[1].split('global_step_')[1].split('_')[0]
print('  step %3d: succ=%.2f s1=%.2f r=%.3f' % (int(step), m['success_at_end'], m['success_once'], m['reward']))
" {} \; 2>/dev/null | sort -t' ' -k2 -n
  echo "[GPU3] Sleeping 300s before next check for new checkpoints..."
  sleep 300
done
