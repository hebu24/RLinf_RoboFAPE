#!/usr/bin/env bash
# HG-DAgger orchestrator for ONE round:
#   collect (student executes, teacher labels) -> merge (original + HG, +source)
#   -> compute Strategy-A MAX_STEPS = K*|D_new|/batch -> retrain (WeightedRandomSampler)
#   -> eval the new student. Default SEQUENTIAL (one Ray job at a time -> no
#   port/GPU conflict). Run 2-3 rounds; each round's new student feeds the next.
#
# Usage (round 0 from the base SFT student):
#   ROUND=0 STUDENT_CKPT=/data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor \
#     ORIG_DIR=/data/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_insert_only_3200 \
#     bash run_hg_dagger_round.sh
# Next round (the script prints the command): pass STUDENT_CKPT=<new student> ROUND=<i+1>.
#
# Ray isolation: collect is multiprocessing (no Ray); retrain uses SFT_RAY_PORT
# (default 6379); eval uses EVAL_RAY_PORT (default 6380). Run sequentially so the
# three never overlap. To overlap rounds, give each a distinct SFT_RAY_PORT +
# EVAL_RAY_PORT + disjoint GPU sets (see RAY_ISOLATION.md) -- never overlap GPUs.
set -euo pipefail
cd /data/yingxi/RLinf_RoboFAPE

# --- required ---
: "${ROUND:?ROUND (iteration index) required}"
: "${STUDENT_CKPT:?STUDENT_CKPT=<student global_step_N/actor from round i-1> required}"
: "${ORIG_DIR:?ORIG_DIR=<original insert-only LeRobot-v2 dir> required}"

# --- knobs ---
NUM_TRAJ="${NUM_TRAJ:-800}"                  # episodes collected this round
COLLECT_GPU_IDS="${COLLECT_GPU_IDS:-0,1,2,3}"
COLLECT_MAX_STEPS="${COLLECT_MAX_STEPS:-600}"
RETRAIN_GPU_IDS="${RETRAIN_GPU_IDS:-4,5,6,7}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-0,1,2,3}"
NEW_DATA_FRACTION="${NEW_DATA_FRACTION:-0.5}"
K="${K:-10}"                                 # Strategy-A multiplier (k in [5,20])
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}" # micro_batch(16) * 4 GPUs
EVAL_EPISODES="${EVAL_EPISODES:-8}"
EVAL_ENVS="${EVAL_ENVS:-8}"

REPO=/data/yingxi/RLinf_RoboFAPE
HG_DIR="$REPO/run_train/peginsertion_maniskill_pi0.5/hg_dagger"
COLLECT_OUT="/data/yingxi/hg_dagger_round${ROUND}"
MERGED_DIR="/data/yingxi/hg_dagger_merged_r${ROUND}"

export PATH=/data/yingxi/kairan/envs/rlinf/bin:$PATH
export PYTHONPATH=$REPO:${PYTHONPATH:-}
# Lift planner imports the RoboFPE solver (collect + eval both need it).
export RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs
export TMPDIR=/data/yingxi/tmp
export HF_HOME=/data/yingxi/.cache/huggingface
mkdir -p "$TMPDIR" "$COLLECT_OUT"

# --- 1. collect HG-DAgger data with the current student (student executes + teacher labels) ---
echo "[round $ROUND] 1/4 collect: $NUM_TRAJ trajs on GPU $COLLECT_GPU_IDS (student=$STUDENT_CKPT)"
rm -rf "$COLLECT_OUT"
python "$HG_DIR/collect_hg_dagger.py" \
  --student-ckpt "$STUDENT_CKPT" \
  --outdir "$COLLECT_OUT" \
  --num-traj "$NUM_TRAJ" \
  --gpu-ids "$COLLECT_GPU_IDS" \
  --max-episode-steps "$COLLECT_MAX_STEPS" \
  2>&1 | tee "logs/hg_dagger_collect_r${ROUND}.log"

# --- 2. merge original + HG-DAgger shards (adds per-frame `source` column; no duplication) ---
echo "[round $ROUND] 2/4 merge: --orig $ORIG_DIR --hg $COLLECT_OUT -> $MERGED_DIR"
rm -rf "$MERGED_DIR"
python "$HG_DIR/merge_datasets.py" \
  --orig "$ORIG_DIR" \
  --hg "$COLLECT_OUT" \
  --out "$MERGED_DIR" \
  2>&1 | tee "logs/hg_dagger_merge_r${ROUND}.log"

# --- 3. Strategy-A step budget ---
# |D_new| = HG frames collected this round (from the merge summary). Strategy-A's
# "steps this iteration = K*|D_new|/batch" is RELATIVE, but the runner's
# runner.max_steps is ABSOLUTE (it stops when global_step >= max_steps, and on
# resume global_step starts at the resumed step). So MAX_STEPS = RESUME_STEP +
# K*|D_new|/batch. actor.optim.total_training_steps is set to the same MAX_STEPS
# by run_hg_dagger_retrain.sh (cosine LR cap).
D_NEW=$(/data/yingxi/kairan/envs/rlinf/bin/python -c "
import json
s = json.load(open('$MERGED_DIR/meta/merge_summary.json'))
print(s['frames_by_source']['hg_new'])
")
RESUME_DIR_FOR_BUDGET="$(dirname "$STUDENT_CKPT")"
RESUME_STEP=$(basename "$RESUME_DIR_FOR_BUDGET" | sed 's/global_step_//')
if ! [[ "$RESUME_STEP" =~ ^[0-9]+$ ]]; then
  echo "[round $ROUND] ERROR: could not parse resume step from $RESUME_DIR_FOR_BUDGET (got '$RESUME_STEP')" >&2
  exit 1
fi
NEW_STEPS=$(( K * D_NEW / GLOBAL_BATCH_SIZE ))
MAX_STEPS=$(( RESUME_STEP + NEW_STEPS ))
echo "[round $ROUND] 3/4 Strategy-A: K=$K |D_new|=$D_NEW batch=$GLOBAL_BATCH_SIZE resume_step=$RESUME_STEP +new=$NEW_STEPS -> MAX_STEPS(absolute)=$MAX_STEPS"

# --- 4. retrain: resume student on merged data with the WeightedRandomSampler ---
# RESUME_DIR must be the global_step_<N> dir (parent of `actor`); STUDENT_CKPT is
# the .../global_step_<N>/actor dir (get_model wants `actor`).
RESUME_DIR="$(dirname "$STUDENT_CKPT")"
echo "[round $ROUND] 4/4 retrain: resume $RESUME_DIR on $MERGED_DIR (GPU $RETRAIN_GPU_IDS, fraction=$NEW_DATA_FRACTION, MAX_STEPS=$MAX_STEPS)"
ROUND="$ROUND" DATA_DIR="$MERGED_DIR" RESUME_DIR="$RESUME_DIR" \
  MAX_STEPS="$MAX_STEPS" NEW_DATA_FRACTION="$NEW_DATA_FRACTION" GPU_IDS="$RETRAIN_GPU_IDS" \
  bash "$HG_DIR/run_hg_dagger_retrain.sh"

# locate the new student checkpoint. run_vla_sft.sh names the log dir
# `logs/<timestamp>-<CONFIG_NAME>-3200` (the -3200 suffix is hardcoded there;
# EXPERIMENT_NAME only sets the tensorboard experiment_name, not the dir). The
# newest such dir is this round's retrain output (the base student's is older).
NEW_RUN_DIR=$(ls -dt "$REPO"/logs/*-peg_insertion_sft_openpi_pi05_wrist-3200 2>/dev/null | head -1)
if [[ -z "$NEW_RUN_DIR" ]]; then
  echo "[round $ROUND] ERROR: could not find retrain run dir (logs/*-peg_insertion_sft_openpi_pi05_wrist-3200)" >&2
  exit 1
fi
NEW_CKPT=$(ls -d "$NEW_RUN_DIR"/checkpoints/global_step_*/actor 2>/dev/null | sort -V | tail -1)
if [[ -z "$NEW_CKPT" ]]; then
  echo "[round $ROUND] ERROR: no global_step_*/actor under $NEW_RUN_DIR" >&2
  exit 1
fi
echo "[round $ROUND] new student: $NEW_CKPT"

# --- 5. eval the new student (insert-only, 600 steps) ---
echo "[round $ROUND] eval: $EVAL_EPISODES eps on GPU $EVAL_GPU_IDS (600 steps, insert-only)"
CHECKPOINT_PATH="$NEW_CKPT" GPU_IDS="$EVAL_GPU_IDS" \
  NUM_EVAL_EPISODES="$EVAL_EPISODES" NUM_ENVS="$EVAL_ENVS" MAX_EPISODE_STEPS=600 \
  EVAL_RAY_PORT="${EVAL_RAY_PORT:-6380}" MANAGE_RAY=true \
  RAY_TMP_DIR=/data/yingxi/ray_tmp_eval_wrist \
  TMPDIR=/data/yingxi/tmp HF_HOME=/data/yingxi/.cache/huggingface \
  bash run_train/eval_checkpoint/run_peginsertion_wrist_insert_only.sh \
  2>&1 | tee "logs/hg_dagger_eval_r${ROUND}.log"

NEXT=$((ROUND + 1))
echo "[round $ROUND] done. new student=$NEW_CKPT"
echo "[round $ROUND] next round: ROUND=$NEXT STUDENT_CKPT=$NEW_CKPT ORIG_DIR=$ORIG_DIR bash $HG_DIR/run_hg_dagger_round.sh"
