#!/bin/bash
set -uo pipefail
cd /data/yingxi/RLinf_RoboFAPE
# REQUIRED: lift planner reads this to locate mani_envs/solutions; code default is a
# wrong H100 path (/home/gpu4/...) so without this the eval fails with
# ModuleNotFoundError: No module named solutions. Propagates to the Popen worker
# via env=os.environ.copy().
export RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs
RUN=logs/20260731-11:18:41-peg_insertion_rl_async_absolute_16ep_single_step/peg_insertion_async_ppo_pi05_robometer
OUT=$RUN/rl_eval_sweep_48ep
mkdir -p "$OUT"
SFT=logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor
LOG="$OUT/resume_run2.log"
echo "=== resume sweep2 start $(date) RLINF_ROBOFPE_PATH=$RLINF_ROBOFPE_PATH ===" | tee "$LOG"
/data/yingxi/kairan/envs/rlinf/bin/python run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
  --checkpoint-dir "$RUN/checkpoints" \
  --output-dir "$OUT" \
  --resume \
  --num-eval-episodes 48 --num-envs 8 \
  --gpu-ids 1,3 \
  --ray-port 6501 \
  --ray-object-store-memory 20000000000 \
  --norm-stats-source "$SFT" \
  --continue-on-error 2>&1 | tee -a "$LOG"
echo "=== resume sweep2 exit=$? $(date) ===" | tee -a "$LOG"
