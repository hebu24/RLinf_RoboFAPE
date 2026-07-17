#!/usr/bin/env bash
# Dual-wrist (front+back) insert-only checkpoint evaluation.
#
# Reuses run_peginsertion_wrist.sh but points at the dual-wrist insert-only
# config, whose reset_options.pre_grasped=True makes every episode start with the
# peg already grasped and lifted (motion-planned by PegInsertionLiftPlanner). The
# policy is then evaluated only on transport + align + descend + insert. The
# dual-wrist config also sets use_wrist_back_image=True + num_images_in_input=3.
set -euo pipefail

export CONFIG_NAME="maniskill_peg_insertion_vertical_dualwrist_sft_eval_openpi_pi05_insert_only"
# Keep the full 600-step rollout budget; it is divisible by both
# num_action_chunks (10) and execute_action_chunks (5).
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-600}"
export TASK_DESCRIPTION="${TASK_DESCRIPTION:-transport and insert the grasped peg into the hole}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_peginsertion_wrist.sh" "$@"
