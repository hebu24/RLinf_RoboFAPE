#!/usr/bin/env bash
# Actual-EE-delta insert-only wrist checkpoint evaluation (use_target=False).
#
# Same actual-EE-delta controller as the full-task actual-ee eval (panda-ee-dpose /
# pd_ee_delta_pose, use_target=False, closed-loop), but insert-only: every episode
# starts with the peg grasped + lifted (motion-planned by PegInsertionLiftPlanner),
# then the policy only transports/aligns/inserts. The model must have been trained
# on actual-EE-delta labels (the actual-ee SFT config).
set -euo pipefail

export CONFIG_NAME="maniskill_peg_insertion_vertical_sft_eval_openpi_pi05_actual_ee_insert_only"
# Transport+align+insert is short; must be divisible by num_action_chunks (10)
# and execute_action_chunks (5).
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
export TASK_DESCRIPTION="${TASK_DESCRIPTION:-transport and insert the grasped peg into the hole}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_peginsertion_wrist.sh" "$@"
