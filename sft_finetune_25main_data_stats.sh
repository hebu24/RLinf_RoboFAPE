#!/usr/bin/env bash
# Independent SFT finetune script that starts from the generic pi05_base checkpoint
# (/opt/kairan/models/RLinf-Pi05-ManiSkill-25Main-SFT) and computes fresh norm_stats from the
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
export SFT_RAY_PORT="${SFT_RAY_PORT:-6379}"
export RAY_TMPDIR="${SFT_RAY_TMPDIR:-/tmp/ray_sft_${SFT_RAY_PORT}}"
# Dashboard-agent listen port MUST be unique per Ray cluster on the same host: Ray's
# default 52365 is a FIXED (non-random) port. If a concurrent cluster (e.g. an eval
# sweep) already binds 52365, this head's raylet crashes in its HTTP loop on bind.
# Default 52366 (eval sweep uses 52365). Override per concurrent SFT run.
export SFT_DASHBOARD_AGENT_PORT="${SFT_DASHBOARD_AGENT_PORT:-52366}"
export CUDA_LAUNCH_BLOCKING=1

# Raise fd limit so the raylet + torch.distributed.checkpoint shard saves do not hit
# the default 1024 ("Too many open files" -> raylet grpc errors / save failure). Two
# concurrent SFT clusters on this box need this badly.
ulimit -n 1048576 2>/dev/null || true

# --- inputs ---
DATA_DIR="${DATA_DIR:-/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_insert_only_3200}"
PI05_BASE="${PI05_BASE:-/opt/kairan/models/RLinf-Pi05-ManiSkill-25Main-SFT}"
PREPARED_BASE="${PREPARED_BASE:-/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/base/pi05_25main_insert_only_wrist_stats}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
# logger experiment name (default keeps the original behavior; override for the
# actual-EE-delta variant so its logs/checkpoints are distinguishable).
EXPERIMENT_NAME="${EXPERIMENT_NAME:-peg_insertion_sft_insert_only_wrist_25main_data_stats}"

# Hydra config (examples/sft/config/<NAME>.yaml) and the matching openpi config_name
# used to compute norm_stats. Override both together for the wrist variant.
CONFIG_NAME="${CONFIG_NAME:-peg_insertion_sft_openpi_pi05_wrist_25main}"
OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_maniskill_peg_insertion_wrist}"
FORCE_NORM_STATS="${FORCE_NORM_STATS:-0}"
NORM_STATS_ASSET="physical-intelligence/maniskill"

# --- (a) prepare base dir: symlink pi05_base weights (do not copy 16.5GB / do not touch shared dir) ---
mkdir -p "$PREPARED_BASE"
ln -sfn "$PI05_BASE/model.safetensors" "$PREPARED_BASE/model.safetensors"
[[ -f "$PI05_BASE/config.json" ]] && ln -sfn "$PI05_BASE/config.json" "$PREPARED_BASE/config.json"
echo "[pi05base] prepared base dir: $PREPARED_BASE (symlinks -> $PI05_BASE)"

# --- (b) restore or compute OpenPI norm_stats ---
# Keep a dataset-local cache, namespaced by OpenPI config because different camera/
# state transforms can produce different statistics from the same LeRobot dataset.
NS_FILE="$PREPARED_BASE/$NORM_STATS_ASSET/norm_stats.json"
DATA_NS_FILE="$DATA_DIR/meta/openpi/$OPENPI_CONFIG_NAME/norm_stats.json"
if [[ "$FORCE_NORM_STATS" != "1" && -f "$DATA_NS_FILE" ]]; then
  mkdir -p "$(dirname "$NS_FILE")"
  cp -f "$DATA_NS_FILE" "$NS_FILE"
  echo "[pi05base] restored dataset-cached norm_stats: $DATA_NS_FILE -> $NS_FILE"
elif [[ "$FORCE_NORM_STATS" != "1" && -f "$NS_FILE" ]]; then
  echo "[pi05base] reusing cached norm_stats at $NS_FILE (set FORCE_NORM_STATS=1 to recompute)"
else
  echo "[pi05base] computing norm_stats from $DATA_DIR -> $NS_FILE"
  # Norm_stats is a pure-statistics (CPU) pass: hide all GPUs so the data loader does
  # not allocate VRAM and compete with training on other cards. GPU visibility is
  # restored (unset CUDA_VISIBLE_DEVICES) below before launching the SFT trainer.
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  python toolkits/lerobot/calculate_norm_stats.py \
    --config-name "$OPENPI_CONFIG_NAME" \
    --repo-id "$DATA_DIR" \
    --output-dir "$PREPARED_BASE"
fi

# Persist the exact OpenPI stats next to the dataset for future prepared-base dirs.
mkdir -p "$(dirname "$DATA_NS_FILE")"
cp -f "$NS_FILE" "$DATA_NS_FILE"
echo "[pi05base] dataset norm_stats cache: $DATA_NS_FILE"

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
# SFT runs its OWN detached Ray head on SFT_RAY_PORT (default 6379), isolated from
# any eval sweep cluster (port 6380). Never a bare `ray stop` (that kills ALL ray on
# the host, incl. eval); teardown is scoped to SFT_RAY_PORT only.
export RAY_ADDRESS="127.0.0.1:${SFT_RAY_PORT}"
unset CUDA_VISIBLE_DEVICES

_sft_scoped_ray_kill() {
  pkill -9 -f "gcs_server.*--gcs_server_port=${SFT_RAY_PORT}"  >/dev/null 2>&1 || true
  pkill -9 -f "raylet.*--gcs-address=[^ ]*:${SFT_RAY_PORT}"    >/dev/null 2>&1 || true
  pkill -9 -f "dashboard.*--gcs-address=[^ ]*:${SFT_RAY_PORT}" >/dev/null 2>&1 || true
  sleep 2
}

echo "[pi05base] Physical GPU_IDS=${GPU_IDS}; cluster.component_placement=${SFT_COMPONENT_PLACEMENT}"
echo "[pi05base] SFT Ray head: port=${SFT_RAY_PORT}, temp_dir=${RAY_TMPDIR}, RAY_ADDRESS=${RAY_ADDRESS}"

if [[ "${SFT_STOP_RAY_BEFORE_START:-1}" == "1" ]]; then
  echo "[pi05base] Clearing stale SFT Ray on port ${SFT_RAY_PORT} (scoped; does not touch other clusters)."
  _sft_scoped_ray_kill
fi

# On SFT exit (incl. ray-start failure or interrupt), tear down ONLY the SFT head
# (port SFT_RAY_PORT), never other clusters. Armed before `ray start` so a failed
# start is cleaned up too.
trap '_sft_scoped_ray_kill' EXIT

# Start the SFT detached head. The driver + workers attach to it via RAY_ADDRESS.
# --dashboard-agent-listen-port must differ from any concurrent cluster (see above).
ray start --head --port="${SFT_RAY_PORT}" --temp-dir="${RAY_TMPDIR}" --dashboard-agent-listen-port="${SFT_DASHBOARD_AGENT_PORT}"

# --- (d) launch the existing Hydra SFT entrypoint, overriding model_path + experiment_name ---
bash examples/sft/run_vla_sft.sh \
  "$CONFIG_NAME" \
  data.train_data_paths="${DATA_DIR}" \
  actor.model.model_path="${PREPARED_BASE}" \
  runner.logger.experiment_name="${EXPERIMENT_NAME}" \
  cluster.component_placement="{actor\\,env\\,rollout:${SFT_COMPONENT_PLACEMENT}}" \
  "${RESUME_DIR:+runner.resume_dir=${RESUME_DIR}}" \
  ${EXTRA_HYDRA:-}
