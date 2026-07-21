#!/usr/bin/env bash
# Dual-wrist (front+back) insert-only SFT launcher.
#
# Wraps sft_finetune_pi05base.sh (which owns the Ray-isolation wiring: scoped
# pkill by SFT_RAY_PORT, RAY_ADDRESS pin, ulimit -n 1048576, EXIT trap, no bare
# `ray stop`). See RAY_ISOLATION.md.
#
# This uses a DISTINCT Ray port (6383) + dashboard-agent port (52369) so it can
# run concurrently with the wrist SFT (6379/52366), actual-ee SFT (6381/52367),
# and eval (6380). To run a second dual-wrist experiment at the same time, pick
# a new port pair (e.g. 6384/52370) and a distinct GPU half.
#
# Required env (override on the command line): DATA_DIR must point at a
# dual-wrist dataset collected with collect_peg_insertion_controller_data.py
# --collect-mode insert_only (contains observation.images.wrist_back).
# Resume with either `RESUME_DIR=/.../global_step_N bash sft_finetune_dualwrist.sh`
# or `bash sft_finetune_dualwrist.sh /.../global_step_N`.
set -euo pipefail
cd /opt/yingxi/RLinf_RoboFAPE

export PYTHONUNBUFFERED=1
export DATA_DIR="${DATA_DIR:-/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_dualwrist_insert_only_3200}"
export GPU_IDS="${GPU_IDS:-4,5,6,7}"
# Distinct Ray isolation from wrist(6379)/actual-ee(6381)/eval(6380).
export SFT_RAY_PORT="${SFT_RAY_PORT:-6383}"
export SFT_DASHBOARD_AGENT_PORT="${SFT_DASHBOARD_AGENT_PORT:-52369}"
export CONFIG_NAME="${CONFIG_NAME:-peg_insertion_sft_openpi_pi05_dualwrist}"
export OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_maniskill_peg_insertion_dualwrist}"
export PREPARED_BASE="${PREPARED_BASE:-/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/base/pi05_base_peg_dualwrist}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-peg_insertion_sft_dualwrist}"
export RESUME_DIR="${RESUME_DIR:-${1:-}}"

# The global_step_10000 checkpoint was created with batch 16/64 and a 40000-step
# cosine schedule. A strict resume must rebuild that same optimizer/scheduler
# configuration; only runner.max_steps changes to stop the continued run at 20000.
# DCP then restores model weights, Adam moments, scheduler position, and RNG state.
if [[ -n "$RESUME_DIR" ]]; then
  export EXTRA_HYDRA="${EXTRA_HYDRA:-+actor.model.openpi.mask_gripper_loss=True actor.micro_batch_size=16 actor.global_batch_size=64 actor.optim.lr=3e-6 actor.optim.lr_warmup_steps=1000 runner.max_steps=20000 actor.optim.total_training_steps=40000 runner.save_interval=5000 actor.optim.num_cycles=0.5 actor.optim.min_lr=2.5e-7}"
else
  # Fresh run: use a self-contained 20000-step cosine schedule.
  export EXTRA_HYDRA="${EXTRA_HYDRA:-+actor.model.openpi.mask_gripper_loss=True actor.optim.lr=3e-6 actor.optim.lr_warmup_steps=1000 runner.max_steps=20000 actor.optim.total_training_steps=20000 runner.save_interval=5000 actor.optim.num_cycles=0.5 actor.optim.min_lr=2.5e-7}"
fi

bash sft_finetune_pi05base.sh 2>&1 | tee logs/sft_dualwrist_tmux.log

