#!/usr/bin/env bash
# Independent SFT finetune script that starts from the generic pi05_base checkpoint
# (/opt/zhangchenyu/weights/pi05_base) and computes fresh norm_stats from the
# training data (instead of reusing the maniskill norm_stats bundled with
# RLinf-Pi05-ManiSkill-25Main-SFT).
#
# It does NOT modify the shared pi05_base dir. Instead it builds a "prepared base"
# directory that symlinks pi05_base/model.safetensors and holds a freshly-computed
# <asset_id>/norm_stats.json, then points actor.model.model_path at it. The rest of
# the SFT chain (get_model weight+norm_stats loading, FSDPVlaSftWorker.save_checkpoint
# norm_stats copy) is reused unchanged, so the saved checkpoint layout is identical to
# sft_finetune.sh.
#
# Usage:
#   GPU_IDS=0,1,2,3 bash sft_finetune_pi05base.sh
#   DATA_DIR=... GPU_IDS=0 bash sft_finetune_pi05base.sh
#   FORCE_NORM_STATS=1 bash sft_finetune_pi05base.sh          # recompute norm_stats
#   CONFIG_NAME=peg_insertion_sft_openpi_pi05_wrist \
#     OPENPI_CONFIG_NAME=pi05_maniskill_peg_insertion_wrist bash sft_finetune_pi05base.sh
set -euo pipefail

cd /opt/yingxi/RLinf_RoboFAPE

# Resolve the repo-local rlinf package (an older installed copy at
# /opt/kairan/RLinf shadows it otherwise and lacks the peg_insertion configs).
export PYTHONPATH=/opt/yingxi/RLinf_RoboFAPE:${PYTHONPATH:-}
export PATH=/opt/kairan/envs/rlinf/bin:$PATH
export RAY_TMPDIR=/tmp/ray_sft_pi05base
export CUDA_LAUNCH_BLOCKING=1

# --- inputs ---
DATA_DIR="${DATA_DIR:-/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200}"
PI05_BASE="${PI05_BASE:-/opt/zhangchenyu/weights/pi05_base}"
PREPARED_BASE="${PREPARED_BASE:-/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/base/pi05_base_peg}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

# Hydra config (examples/sft/config/<NAME>.yaml) and the matching openpi config_name
# used to compute norm_stats. Override both together for the wrist variant.
CONFIG_NAME="${CONFIG_NAME:-peg_insertion_sft_openpi_pi05}"
OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_maniskill_peg_insertion}"
NORM_STATS_ASSET="physical-intelligence/maniskill"

# --- (a) prepare base dir: symlink pi05_base weights (do not copy 16.5GB / do not touch shared dir) ---
mkdir -p "$PREPARED_BASE"
ln -sfn "$PI05_BASE/model.safetensors" "$PREPARED_BASE/model.safetensors"
[[ -f "$PI05_BASE/config.json" ]] && ln -sfn "$PI05_BASE/config.json" "$PREPARED_BASE/config.json"
echo "[pi05base] prepared base dir: $PREPARED_BASE (symlinks -> $PI05_BASE)"

# --- (b) compute norm_stats from the training data into the prepared base (cached unless FORCE_NORM_STATS=1) ---
NS_FILE="$PREPARED_BASE/$NORM_STATS_ASSET/norm_stats.json"
if [[ ! -f "$NS_FILE" || "${FORCE_NORM_STATS:-0}" == "1" ]]; then
  echo "[pi05base] computing norm_stats from $DATA_DIR -> $NS_FILE"
  # Norm_stats is a pure-statistics (CPU) pass: hide all GPUs so the data loader does
  # not allocate VRAM and compete with training on other cards. GPU visibility is
  # restored (unset CUDA_VISIBLE_DEVICES) below before launching the SFT trainer.
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  python toolkits/lerobot/calculate_norm_stats.py \
    --config-name "$OPENPI_CONFIG_NAME" \
    --repo-id "$DATA_DIR" \
    --output-dir "$PREPARED_BASE"
else
  echo "[pi05base] reusing cached norm_stats at $NS_FILE (set FORCE_NORM_STATS=1 to recompute)"
fi

# --- (c) GPU placement (same logic as sft_finetune.sh) ---
IFS="," read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
if ((${#GPU_ID_ARRAY[@]} == 0)); then
  echo "GPU_IDS must contain at least one GPU id." >&2; exit 1
fi
for gpu_id in "${GPU_ID_ARRAY[@]}"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "GPU_IDS must be a comma-separated list of physical GPU ids, got: ${GPU_IDS}" >&2; exit 1
  fi
done

SFT_COMPONENT_PLACEMENT="${GPU_IDS}"
if ((${#GPU_ID_ARRAY[@]} == 1)); then
  SFT_COMPONENT_PLACEMENT="${GPU_ID_ARRAY[0]}"
else
  consecutive=1
  prev_gpu_id="${GPU_ID_ARRAY[0]}"
  for gpu_id in "${GPU_ID_ARRAY[@]:1}"; do
    if ((gpu_id != prev_gpu_id + 1)); then consecutive=0; break; fi
    prev_gpu_id="${gpu_id}"
  done
  if ((consecutive)); then
    last_gpu_index=$((${#GPU_ID_ARRAY[@]} - 1))
    SFT_COMPONENT_PLACEMENT="${GPU_ID_ARRAY[0]}-${GPU_ID_ARRAY[${last_gpu_index}]}"
  fi
fi

export SFT_COMPONENT_PLACEMENT
unset RAY_ADDRESS
unset CUDA_VISIBLE_DEVICES

echo "[pi05base] Physical GPU_IDS=${GPU_IDS}; cluster.component_placement=${SFT_COMPONENT_PLACEMENT}"

if [[ "${SFT_STOP_RAY_BEFORE_START:-1}" == "1" ]]; then
  echo "[pi05base] Stopping existing Ray before SFT so Ray redetects physical GPU resources."
  ray stop --force >/dev/null 2>&1 || true
fi

# --- (d) launch the existing Hydra SFT entrypoint, overriding model_path + experiment_name ---
bash examples/sft/run_vla_sft.sh \
  "$CONFIG_NAME" \
  data.train_data_paths="${DATA_DIR}" \
  actor.model.model_path="${PREPARED_BASE}" \
  runner.logger.experiment_name="peg_insertion_sft_pi05base" \
  cluster.component_placement="{actor\\,env\\,rollout:${SFT_COMPONENT_PLACEMENT}}" \
  "${RESUME_DIR:+runner.resume_dir=${RESUME_DIR}}"
