#!/usr/bin/env bash
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE

exec /data/yingxi/RLinf_RoboFAPE/.venv/bin/tensorboard \
  --logdir /data/yingxi/RLinf_RoboFAPE/logs \
  --host 0.0.0.0 \
  --port 6006 \
  --reload_interval 30
