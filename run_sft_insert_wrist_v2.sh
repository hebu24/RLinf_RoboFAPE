#!/usr/bin/env bash
set -euo pipefail
cd /data/yingxi/RLinf_RoboFAPE
export PYTHONUNBUFFERED=1
export DATA_DIR=/data/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_insert_only_3200
export GPU_IDS=4,5,6,7
export SFT_RAY_PORT=6379
export SFT_DASHBOARD_AGENT_PORT=52366
export CONFIG_NAME=peg_insertion_sft_openpi_pi05_wrist
export OPENPI_CONFIG_NAME=pi05_maniskill_peg_insertion_wrist
export PREPARED_BASE=/data/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/base/pi05_base_peg_wrist_insert
export EXPERIMENT_NAME=peg_insertion_sft_insert_only_wrist_v2
# Fixes: mask gripper dim in flow-matching loss (std=0 -> pure-noise target),
# lower lr + longer warmup for the OOD insert-only start distribution.
export EXTRA_HYDRA="+actor.model.openapi.mask_gripper_loss=True actor.optim.lr=1.5e-6 actor.optim.lr_warmup_steps=1000"
bash sft_finetune_pi05base.sh 2>&1 | tee logs/sft_insert_wrist_v2_tmux.log
