#!/usr/bin/env bash
# Robometer reward server. Run this outside the RL Ray cluster, typically on one
# dedicated GPU while RL uses two other GPUs.
set -euo pipefail
set -x

export PATH="$HOME/.local/bin:$PATH"

ROBOMETER_REPO="${ROBOMETER_REPO:-/data/yingxi/RoboFPE/robometer}"
ROBOMETER_CKPT="${ROBOMETER_CKPT:-/data/yingxi/robometer/robometer-4b}"
ROBOMETER_GPU_ID="${ROBOMETER_GPU_ID:-2}"
ROBOMETER_PORT="${ROBOMETER_PORT:-8001}"
ROBOMETER_BATCH_SIZE="${ROBOMETER_BATCH_SIZE:-4}"
ROBOMETER_NUM_GPUS="${ROBOMETER_NUM_GPUS:-1}"
ROBOMETER_PYTHON_BIN="${ROBOMETER_PYTHON_BIN:-${ROBOMETER_REPO}/.venv/bin/python}"

if [[ ! -d "${ROBOMETER_REPO}" ]]; then
  echo "ROBOMETER_REPO does not exist: ${ROBOMETER_REPO}" >&2
  exit 1
fi
if [[ ! -e "${ROBOMETER_CKPT}" ]]; then
  echo "ROBOMETER_CKPT does not exist: ${ROBOMETER_CKPT}" >&2
  exit 1
fi
if [[ ! -x "${ROBOMETER_PYTHON_BIN}" ]]; then
  echo "ROBOMETER_PYTHON_BIN is not executable: ${ROBOMETER_PYTHON_BIN}" >&2
  exit 1
fi

cd "${ROBOMETER_REPO}"
mkdir -p /data/yingxi/tmp /data/yingxi/.cache/huggingface
export CUDA_VISIBLE_DEVICES="${ROBOMETER_GPU_ID}"
export TMPDIR="${TMPDIR:-/data/yingxi/tmp}"
export HF_HOME="${HF_HOME:-/data/yingxi/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/yingxi/.cache/huggingface/datasets}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "${ROBOMETER_PYTHON_BIN}" robometer/evals/eval_server.py \
  "model_path=${ROBOMETER_CKPT}" \
  server_url=0.0.0.0 \
  "server_port=${ROBOMETER_PORT}" \
  "num_gpus=${ROBOMETER_NUM_GPUS}" \
  "batch_size=${ROBOMETER_BATCH_SIZE}"
