#!/usr/bin/env bash
# Actual-EE-delta wrist checkpoint evaluation (use_target=False).
#
# Reuses run_peginsertion_wrist.sh but points at the actual-EE-delta eval config.
# The model must have been trained on actual-EE-delta labels
# (collect_peg_insertion_actual_ee_delta.py / convert_controller_to_actual_ee.py),
# so eval runs with policy_setup=panda-ee-dpose -> pd_ee_delta_pose
# (use_target=False, closed-loop: each delta integrates from the actual EE).
set -euo pipefail

export CONFIG_NAME="maniskill_peg_insertion_vertical_sft_eval_openpi_pi05_actual_ee"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_peginsertion_wrist.sh" "$@"
