#!/usr/bin/env bash
# Robometer-4B (RFM, 10 bins, bf16) reward server for the ablation RL run.
# GPU 4 (co-located with RL rank0), port 8001 — DISTINCT from :8000 (checkpoint-400).
# expandable_segments:True reclaims fragmentation (RFM 8x60-frame forward peaks
# ~35GB active + 12GB fragmentation; expandable_segments drops the 12GB so it
# fits alongside the idle RL rank0 on the shared 80GB GPU4).
set -x
export PATH="$HOME/.local/bin:$PATH"
cd /home/yingxi/RoboFAC/robometer
mkdir -p /data/yingxi/tmp /data/yingxi/.cache/huggingface
export CUDA_VISIBLE_DEVICES=4
export TMPDIR=/data/yingxi/tmp
export HF_HOME=/data/yingxi/.cache/huggingface
export HF_DATASETS_CACHE=/data/yingxi/.cache/huggingface/datasets
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /home/yingxi/RoboFAC/robometer/.venv/bin/python robometer/evals/eval_server.py \
  model_path=/data/yingxi/robometer/Robometer-4B \
  server_url=0.0.0.0 server_port=8001 num_gpus=1 batch_size=4
