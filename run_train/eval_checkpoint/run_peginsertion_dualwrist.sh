#!/usr/bin/env bash
# Dual-wrist (front+back) checkpoint evaluation -- full pick-up-and-insert episodes.
#
# Reuses run_peginsertion_wrist.sh (which owns the Ray-isolation wiring:
# EVAL_RAY_PORT, scoped pkill, RAY_ADDRESS pin, MANAGE_RAY, ulimit, no bare
# `ray stop`) but points at the dual-wrist eval config, which sets
# use_wrist_image=True + use_wrist_back_image=True + num_images_in_input=3 so the
# back wrist camera feeds the model's right_wrist_0_rgb slot.
set -euo pipefail

export CONFIG_NAME="maniskill_peg_insertion_vertical_dualwrist_sft_eval_openpi_pi05"
export TASK_DESCRIPTION="${TASK_DESCRIPTION:-insert the blue peg vertically into the orange hole}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_peginsertion_wrist.sh" "$@"
