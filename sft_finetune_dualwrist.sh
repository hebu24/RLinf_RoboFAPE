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
set -euo pipefail
cd /opt/yingxi/RLinf_RoboFAPE

export PYTHONUNBUFFERED=1
export DATA_DIR="${DATA_DIR:-/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_dualwrist_insert_only_3200}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
# Distinct Ray isolation from wrist(6379)/actual-ee(6381)/eval(6380).
export SFT_RAY_PORT="${SFT_RAY_PORT:-6383}"
export SFT_DASHBOARD_AGENT_PORT="${SFT_DASHBOARD_AGENT_PORT:-52369}"
export CONFIG_NAME="${CONFIG_NAME:-peg_insertion_sft_openpi_pi05_dualwrist}"
export OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_maniskill_peg_insertion_dualwrist}"
export PREPARED_BASE="${PREPARED_BASE:-/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/base/pi05_base_peg_dualwrist}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-peg_insertion_sft_dualwrist}"
# Insert-only OOD start distribution: mask the gripper dim in the flow-matching
# loss (std=0 -> pure-noise target), lower lr + longer warmup. Same recipe as the
# insert-only wrist v2 run. Cap the run at 10000 steps (max_steps ==
# total_training_steps so the cosine LR schedule decays to min_lr over the run;
# save a checkpoint every 1000 steps).
export EXTRA_HYDRA="${EXTRA_HYDRA:-+actor.model.openapi.mask_gripper_loss=True actor.optim.lr=1e-5 actor.optim.lr_warmup_steps=1000 runner.max_steps=40000 actor.optim.total_training_steps=40000 runner.save_interval=5000 actor.optim.num_cycles=0.5 actor.optim.min_lr=2.5e-7}"

bash sft_finetune_pi05base.sh 2>&1 | tee logs/sft_dualwrist_tmux.log

