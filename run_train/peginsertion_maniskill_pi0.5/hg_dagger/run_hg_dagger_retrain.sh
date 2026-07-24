#!/usr/bin/env bash
# HG-DAgger SFT retrain (one round).
#
# Resume the current student (pi0.5 wrist VLA) on the MERGED dataset (original
# insert-only + this round's HG-DAgger shards) with:
#   * a controllable new-data fraction (data.weighted_new_data_fraction ->
#     WeightedRandomSampler in fsdp_vla_sft_worker.build_dataloader; default 0.5)
#   * a Strategy-A step budget: runner.max_steps = K * |D_new| / global_batch
#     (computed by run_hg_dagger_round.sh and passed in as MAX_STEPS), with a
#     matching cosine actor.optim.total_training_steps.
#
# Mirrors run_sft_insert_wrist_v2.sh + sft_finetune_pi05base.sh Ray isolation
# (distinct GCS port + dashboard-agent + /data temp-dir + disjoint GPUs +
# scoped teardown; never a bare `ray stop`). See RAY_ISOLATION.md.
#
# Usage (usually called by run_hg_dagger_round.sh):
#   ROUND=0 DATA_DIR=<merged> RESUME_DIR=<student global_step_N> MAX_STEPS=5000 \
#     GPU_IDS=4,5,6,7 NEW_DATA_FRACTION=0.5 bash run_hg_dagger_retrain.sh
set -euo pipefail
cd /data/yingxi/RLinf_RoboFAPE
export PYTHONUNBUFFERED=1

# --- required inputs (pass explicitly per round) ---
: "${DATA_DIR:?DATA_DIR=<merged LeRobot-v2 dir> required}"
: "${RESUME_DIR:?RESUME_DIR=<student global_step_N> dir required}"
: "${MAX_STEPS:?MAX_STEPS (Strategy-A K*|D_new|/batch) required}"
: "${ROUND:?ROUND (iteration index, for experiment name) required}"

# --- knobs ---
NEW_DATA_FRACTION="${NEW_DATA_FRACTION:-0.5}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"

# --- Ray isolation (5 rules; defaults disjoint from eval's 6380) ---
export SFT_RAY_PORT="${SFT_RAY_PORT:-6379}"
export SFT_DASHBOARD_AGENT_PORT="${SFT_DASHBOARD_AGENT_PORT:-52366}"
export SFT_RAY_TMPDIR="${SFT_RAY_TMPDIR:-/data/yingxi/ray_tmp_sft_${SFT_RAY_PORT}}"
# `/` is ~95% full on xulab -> every temp dir on /data (Errno 28 otherwise).
export TMPDIR="${TMPDIR:-/data/yingxi/tmp}"
export HF_HOME="${HF_HOME:-/data/yingxi/.cache/huggingface}"
mkdir -p "$SFT_RAY_TMPDIR" "$TMPDIR" "$HF_HOME"

# --- Hydra / openpi config (wrist insert-only) ---
export CONFIG_NAME=peg_insertion_sft_openpi_pi05_wrist
export OPENPI_CONFIG_NAME=pi05_maniskill_peg_insertion_wrist
export PREPARED_BASE=/data/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/base/pi05_base_peg_wrist_insert
export EXPERIMENT_NAME="peg_insertion_hg_dagger_wrist_r${ROUND}"
export DATA_DIR
export RESUME_DIR
export GPU_IDS

# mask gripper flow-matching loss (std=0 -> pure-noise target). For DAgger we
# use a CONSTANT low LR (lr_scheduler=constant, lr_warmup_steps=0): we resume
# from a trained student near the END of its original cosine, so a cosine
# schedule would put the resume step at ~0 LR (barely learning). A constant
# 1.5e-6 gives meaningful correction updates over the (short) Strategy-A budget.
# runner.max_steps is ABSOLUTE (resume starts global_step at the resumed step),
# so it = RESUME_STEP + K*|D_new|/batch (set by run_hg_dagger_round.sh).
# +data.weighted_new_data_fraction is a NEW key not in the base SFT config.
export EXTRA_HYDRA="+actor.model.openapi.mask_gripper_loss=True actor.optim.lr=1.5e-6 actor.optim.lr_scheduler=constant actor.optim.lr_warmup_steps=0 runner.max_steps=${MAX_STEPS} actor.optim.total_training_steps=${MAX_STEPS} +data.weighted_new_data_fraction=${NEW_DATA_FRACTION}"

bash sft_finetune_pi05base.sh 2>&1 | tee "logs/hg_dagger_retrain_r${ROUND}.log"
