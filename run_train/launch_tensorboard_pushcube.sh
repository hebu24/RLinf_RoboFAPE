#!/usr/bin/env bash
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE

exec "${TENSORBOARD_BIN:-/data/yingxi/RLinf_RoboFAPE/.venv/bin/tensorboard}" \
  --logdir "${TB_LOGDIR:-/data/yingxi/RLinf_RoboFAPE/logs}" \
  --host 0.0.0.0 \
  --port "${TB_PORT:-6006}" \
  --reload_interval "${TB_RELOAD_INTERVAL:-30}"
