#!/usr/bin/env bash
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE

export PYTHONUNBUFFERED=1

RL_LOG_DIR=/data/yingxi/RLinf_RoboFAPE/logs/pushcube_rl_env32_abs30_seed0_31_20260825_gpu012
CKPT_DIR=${RL_LOG_DIR}/pushcube_rl_robometer_absolute/checkpoints
OUT_DIR=${RL_LOG_DIR}/eval_sweep_gpu3_seed0_7

exec /data/yingxi/RLinf_RoboFAPE/.venv/bin/python \
  run_train/eval_checkpoint/sweep_pushcube_wrist.py \
  --checkpoint-dir "${CKPT_DIR}" \
  --output-dir "${OUT_DIR}" \
  --watch \
  --resume \
  --continue-on-error \
  --gpu-ids 0,1,2,3 \
  --seeds 0-7 \
  --num-eval-episodes 50 \
  --num-envs 10 \
  --max-episode-steps 200 \
  --ray-port 6387 \
  --ray-dashboard-port 8267 \
  --ray-dashboard-agent-port 52373 \
  --ray-min-worker-port 13400 \
  --ray-max-worker-port 13799 \
  --ray-temp-dir /data/yingxi/ray_tmp_pushcube_eval_6387 \
  --no-save-video
