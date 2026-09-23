#!/usr/bin/env bash
# Independent SFT finetune script that starts from the generic pi05_base checkpoint
# (/data/yingxi/weights/pi05_base) and computes fresh norm_stats from the
# training data when the dataset does not already ship them.
#
# It does NOT modify the shared pi05_base dir. Instead it builds a "prepared base"
# directory that symlinks pi05_base/model.safetensors and holds a freshly-computed
# <asset_id>/norm_stats.json, then points actor.model.model_path at it. The rest of
# the SFT chain (get_model weight+norm_stats loading, FSDPVlaSftWorker.save_checkpoint
# norm_stats copy) is reused unchanged, so the saved checkpoint layout is the standard
# pi0.5 SFT layout.
#
# Usage:
#   GPU_IDS=0,1,2,3 bash sft_finetune_pi05base.sh
#   DATA_DIR=... GPU_IDS=0 bash sft_finetune_pi05base.sh
#   CONFIG_NAME=peg_insertion_sft_openpi_pi05_wrist \
#     OPENPI_CONFIG_NAME=pi05_maniskill_peg_insertion_wrist bash sft_finetune_pi05base.sh
#   TASK_ID=PegInsertionVertical-v1 PEG_INSERTION_MODE=full \
#     GPU_IDS=0,1,2,3 bash sft_finetune_pi05base.sh
set -euo pipefail

cd /data/yingxi/RLinf_RoboFAPE

# Resolve the repo-local rlinf package (an older installed copy at
# /opt/kairan/RLinf shadows it otherwise and lacks the peg_insertion configs).
export PYTHONPATH=/data/yingxi/RLinf_RoboFAPE:${PYTHONPATH:-}
VENV_DIR="${VENV_DIR:-/data/yingxi/RLinf_RoboFAPE/.venv}"
export PATH="${VENV_DIR}/bin:/data/yingxi/kairan/envs/rlinf/bin:${PATH}"
export SFT_RAY_PORT="${SFT_RAY_PORT:-6379}"
export RAY_TMPDIR="${SFT_RAY_TMPDIR:-/tmp/ray_sft_${SFT_RAY_PORT}}"
export SFT_DASHBOARD_PORT="${SFT_DASHBOARD_PORT:-8265}"
export SFT_RAY_CLIENT_SERVER_PORT="${SFT_RAY_CLIENT_SERVER_PORT:-10001}"
export SFT_RAY_MIN_WORKER_PORT="${SFT_RAY_MIN_WORKER_PORT:-10002}"
export SFT_RAY_MAX_WORKER_PORT="${SFT_RAY_MAX_WORKER_PORT:-19999}"
# Dashboard-agent listen port MUST be unique per Ray cluster on the same host: Ray's
# default 52365 is a FIXED (non-random) port. If a concurrent cluster (e.g. an eval
# sweep) already binds 52365, this head's raylet crashes in its HTTP loop on bind.
# Default 52366 (eval sweep uses 52365). Override per concurrent SFT run.
export SFT_DASHBOARD_AGENT_PORT="${SFT_DASHBOARD_AGENT_PORT:-52366}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

# Raise fd limit so the raylet + torch.distributed.checkpoint shard saves do not hit
# the default 1024 ("Too many open files" -> raylet grpc errors / save failure). Two
# concurrent SFT clusters on this box need this badly.
ulimit -n 1048576 2>/dev/null || true

# --- inputs ---
TASK_ID="${TASK_ID:-PegInsertionVertical-v1}"
PEG_INSERTION_MODE="${PEG_INSERTION_MODE:-insert_only}"
if [[ "${TASK_ID}" == "PegInsertionVertical-v1" && \
      "${PEG_INSERTION_MODE}" != "insert_only" && \
      "${PEG_INSERTION_MODE}" != "full" ]]; then
  echo "PEG_INSERTION_MODE must be insert_only or full, got: ${PEG_INSERTION_MODE}" >&2
  exit 1
fi
if [[ -z "${DATA_DIR:-}" ]]; then
  case "${TASK_ID}" in
    PegInsertionSide-v1)
      DATA_DIR="/data/yingxi/datasets/robofpe_sft/PegInsertionSide-v1_render_wrist_filtered_q99_tcp/PegInsertionSide-v1_render_wrist_filtered_q99/lerobot"
      ;;
    PegInsertionVertical-v1)
      DATA_DIR="/data/yingxi/datasets/robofpe_sft/PegInsertionVertical-v1_render_wrist_filtered_q99/lerobot"
      ;;
    StackCube-v1)
      DATA_DIR="/data/yingxi/datasets/robofpe_sft/StackCube-v1_render_wrist_filtered_q99/lerobot"
      ;;
    UprightStack-v1)
      DATA_DIR="/data/yingxi/datasets/robofpe_sft/uprightstack_sft_gate_20260921/UprightStack-v1_render_wrist_filtered_q99/lerobot"
      ;;
    PullCubeTool-golf)
      DATA_DIR="/data/yingxi/datasets/robofpe_sft/releases/PullCubeTool-golf_20260921/lerobot_filtered_q99"
      ;;
    LiftPegUpright-box|PickCube-ball|PullCube-block)
      DATA_DIR="/data/yingxi/datasets/robofpe_sft/${TASK_ID}_render_wrist_filtered_q99/lerobot"
      ;;
    *)
      echo "Unsupported TASK_ID=${TASK_ID}; set DATA_DIR explicitly." >&2
      exit 1
      ;;
  esac
fi
PI05_BASE="${PI05_BASE:-/data/yingxi/weights/pi05_base}"
MODE_SUFFIX=""
if [[ "${TASK_ID}" == "PegInsertionVertical-v1" ]]; then
  MODE_SUFFIX="_${PEG_INSERTION_MODE}"
fi
PREPARED_BASE="${PREPARED_BASE:-/data/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/base/pi05_base_${TASK_ID}${MODE_SUFFIX}_wrist}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
OPENPI_NUM_WORKERS="${OPENPI_NUM_WORKERS:-0}"
# logger experiment name (override to distinguish logs/checkpoints across runs).
EXPERIMENT_NAME="${EXPERIMENT_NAME:-peg_insertion_${TASK_ID}${MODE_SUFFIX}_sft_wrist}"

# Hydra config (examples/sft/config/<NAME>.yaml) and the matching openpi config_name
# used to compute norm_stats. Both default to the wrist insert-only variant.
NORM_STATS_ASSET="physical-intelligence/maniskill"
EXPECTED_ACTION_DIM=7
if [[ "${TASK_ID}" == "PegInsertionVertical-v1" && "${PEG_INSERTION_MODE}" == "full" ]]; then
  CONFIG_NAME="${CONFIG_NAME:-robofpe_maniskill_sft_openpi_pi05_wrist}"
  OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_maniskill_wrist}"
  EXPECTED_ACTION_DIM=8
elif [[ "${TASK_ID}" == "StackCube-v1" || "${TASK_ID}" == "UprightStack-v1" || "${TASK_ID}" == "LiftPegUpright-box" || "${TASK_ID}" == "PickCube-ball" || "${TASK_ID}" == "PullCube-block" || "${TASK_ID}" == "PullCubeTool-golf" ]]; then
  CONFIG_NAME="${CONFIG_NAME:-robofpe_maniskill_sft_openpi_pi05_wrist}"
  OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_maniskill_wrist}"
  EXPECTED_ACTION_DIM=8
else
  CONFIG_NAME="${CONFIG_NAME:-peg_insertion_sft_openpi_pi05_wrist}"
  OPENPI_CONFIG_NAME="${OPENPI_CONFIG_NAME:-pi05_maniskill_peg_insertion_wrist}"
fi
TASK_NORM_STATS_SOURCE=""
if [[ "${TASK_ID}" == "PegInsertionSide-v1" ]]; then
  TASK_NORM_STATS_SOURCE="/data/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/base/pi05_base_peg_insertion_side_wrist_filtered_q99_tcp/${NORM_STATS_ASSET}/norm_stats.json"
fi

# Fail early instead of silently starting from the base model when a resume path is
# misspelled or points at the actor subdirectory rather than global_step_<N>.
if [[ -n "${RESUME_DIR:-}" ]]; then
  if [[ ! "$RESUME_DIR" =~ /global_step_[0-9]+/?$ ]]; then
    echo "RESUME_DIR must point to a global_step_<N> directory, got: $RESUME_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$RESUME_DIR/actor/dcp_checkpoint" ]] || ! compgen -G "$RESUME_DIR/actor/dcp_checkpoint/*.distcp" >/dev/null; then
    echo "RESUME_DIR has no complete actor DCP checkpoint: $RESUME_DIR" >&2
    exit 1
  fi
  echo "[pi05base] strict resume checkpoint: $RESUME_DIR"
else
  echo "[pi05base] RESUME_DIR is empty; starting a fresh run from PREPARED_BASE."
fi

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
if [[ -n "$TASK_NORM_STATS_SOURCE" && -f "$TASK_NORM_STATS_SOURCE" ]]; then
  mkdir -p "$(dirname "$NS_FILE")" "$(dirname "$DATA_NS_FILE")"
  [[ "$TASK_NORM_STATS_SOURCE" == "$NS_FILE" ]] || cp -f "$TASK_NORM_STATS_SOURCE" "$NS_FILE"
  [[ "$TASK_NORM_STATS_SOURCE" == "$DATA_NS_FILE" ]] || cp -f "$TASK_NORM_STATS_SOURCE" "$DATA_NS_FILE"
  echo "[pi05base] using audited ${TASK_ID} norm_stats: $TASK_NORM_STATS_SOURCE"
elif [[ -f "$DATA_NS_FILE" ]] && \
  [[ "$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d["norm_stats"]["actions"]["mean"]))' "$DATA_NS_FILE")" == "${EXPECTED_ACTION_DIM}" ]]; then
  mkdir -p "$(dirname "$NS_FILE")"
  cp -f "$DATA_NS_FILE" "$NS_FILE"
  echo "[pi05base] using dataset norm_stats: $DATA_NS_FILE -> $NS_FILE"
else
  echo "[pi05base] computing norm_stats from $DATA_DIR -> $NS_FILE"
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
  python precompute_openpi_norm_stats.py \
    --config-name "$OPENPI_CONFIG_NAME" \
    --data-dir "$DATA_DIR"
  [[ -f "$DATA_NS_FILE" ]] || { echo "[pi05base] norm_stats missing: $DATA_NS_FILE" >&2; exit 1; }
  mkdir -p "$(dirname "$NS_FILE")"
  cp -f "$DATA_NS_FILE" "$NS_FILE"
fi

# Persist the exact OpenPI stats next to the dataset for future prepared-base dirs.
mkdir -p "$(dirname "$DATA_NS_FILE")"
cp -f "$NS_FILE" "$DATA_NS_FILE"
echo "[pi05base] dataset norm_stats cache: $DATA_NS_FILE"

# --- (c) GPU placement ---
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
  pkill -9 -f "ray.util.client.server.*--address=[^ ]*:${SFT_RAY_PORT}" >/dev/null 2>&1 || true
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
ray start --head \
  --port="${SFT_RAY_PORT}" \
  --temp-dir="${RAY_TMPDIR}" \
  --dashboard-port="${SFT_DASHBOARD_PORT}" \
  --dashboard-agent-listen-port="${SFT_DASHBOARD_AGENT_PORT}" \
  --ray-client-server-port="${SFT_RAY_CLIENT_SERVER_PORT}" \
  --min-worker-port="${SFT_RAY_MIN_WORKER_PORT}" \
  --max-worker-port="${SFT_RAY_MAX_WORKER_PORT}"

# --- (d) launch the existing Hydra SFT entrypoint, overriding model_path + experiment_name ---
bash examples/sft/run_vla_sft.sh \
  "$CONFIG_NAME" \
  data.train_data_paths="${DATA_DIR}" \
  actor.model.model_path="${PREPARED_BASE}" \
  +actor.openpi_num_workers="${OPENPI_NUM_WORKERS}" \
  runner.logger.experiment_name="${EXPERIMENT_NAME}" \
  cluster.component_placement="{actor\\,env\\,rollout:${SFT_COMPONENT_PLACEMENT}}" \
  "${RESUME_DIR:+runner.resume_dir=${RESUME_DIR}}" \
  ${EXTRA_HYDRA:-}
