#!/usr/bin/env bash
# Insert-only wrist checkpoint evaluation.
#
# Reuses run_peginsertion_wrist.sh but points at the insert-only config, whose
# reset_options.pre_grasped=True makes every episode start with the peg already
# grasped and lifted (motion-planned by PegInsertionLiftPlanner). The policy is
# then evaluated only on transport + align + descend + insert.
set -euo pipefail

export CONFIG_NAME="maniskill_peg_insertion_vertical_wrist_sft_eval_openpi_pi05_insert_only"
# 600-step rollout horizon: gives the policy enough time to transport + align +
# insert (200 steps cut off ~half the successes -> 1/8 vs 4/8 at 600). Divisible by
# num_action_chunks (10) and execute_action_chunks (10).
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-600}"
export TASK_DESCRIPTION="${TASK_DESCRIPTION:-transport and insert the grasped peg into the hole}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_peginsertion_wrist.sh" "$@"
