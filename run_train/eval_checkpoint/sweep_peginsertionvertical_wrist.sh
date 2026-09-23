#!/usr/bin/env bash
# Run the standard checkpoint sweep with the Vertical full-trajectory contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_PATH}/.venv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
if [[ -z "${CHECKPOINT_DIR}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Set CHECKPOINT_DIR and OUTPUT_DIR." >&2
  exit 2
fi

exec "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/sweep_peginsertion_wrist.py" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --venv-dir "${VENV_DIR}" \
  --gpu-ids "${GPU_IDS:-0,1}" \
  --seeds "${SEEDS:-0-7}" \
  --num-eval-episodes "${NUM_EVAL_EPISODES:-50}" \
  --num-envs "${NUM_ENVS:-2}" \
  --max-episode-steps 450 \
  --run-script "${SCRIPT_DIR}/run_peginsertionvertical_wrist.sh" \
  --ray-port "${RAY_PORT:-6394}" \
  --ray-dashboard-port "${RAY_DASHBOARD_PORT:-8394}" \
  --ray-tmp-dir "${RAY_TMP_DIR:-/data/yingxi/ray_vertical_sweep}" \
  "$@"
