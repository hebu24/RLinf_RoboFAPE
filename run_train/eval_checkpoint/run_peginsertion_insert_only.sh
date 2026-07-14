#!/usr/bin/env bash
# Insert-only base-only (no wrist camera) checkpoint evaluation.
#
# Reuses run_peginsertion.sh but points at the insert-only config, whose
# reset_options.pre_grasped=True makes every episode start with the peg already
# grasped and lifted (motion-planned by PegInsertionLiftPlanner). The policy is
# then evaluated only on transport + align + descend + insert.
set -euo pipefail

export CONFIG_NAME="maniskill_peg_insertion_vertical_sft_eval_openpi_pi05_insert_only"
# Transport+align+insert is short; must be divisible by num_action_chunks (10).
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-200}"
export TASK_DESCRIPTION="${TASK_DESCRIPTION:-transport and insert the grasped peg into the hole}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_peginsertion.sh" "$@"
