# RoboFPE render-camera + wrist-camera SFT collection

The collection runtime is:

```text
/data/yingxi/robometer/failure_detection_env/bin/python
```

Do not replace it with `/root/failure_detection_env`.

The policy still receives exactly two images:

- `observation.images.top`: pixels from the human `render_camera`
- `observation.images.wrist`: pixels from `hand_camera`

The historical `top` field name is retained for OpenPI compatibility. Raw
episodes may also contain `base_camera`, but base-camera pixels are not used
for new SFT training data.

Default task: `PushCube-v1` with `robot_uids=panda_wristcam`.

Collect successful trajectories and convert to LeRobot:

```bash
cd /data/yingxi/RLinf_RoboFAPE
/data/yingxi/robometer/failure_detection_env/bin/python run_train/robofpe_sft_data/collect_sft_data.py collect \
  --task-id PushCube-v1 \
  --num-traj 3200 \
  --output-dir /data/yingxi/datasets/robofpe_sft/PushCube-v1_render_wrist \
  --robot-uids panda_wristcam \
  --success-only \
  --randomize-initial-poses \
  --randomize-wrist-camera \
  --randomize-render-camera \
  --randomize-lighting \
  --save-video --sim-backend gpu \
  --num-workers 1 --gpu-ids 0 \
  --max-attempts-per-traj 100 \
  --seed 20260902
```

The command above is a single-GPU/single-worker example. For the requested
4-GPU collection, use the commands below. Run tasks sequentially.

```bash
cd /data/yingxi/RLinf_RoboFAPE
export PYTHON_BIN=/data/yingxi/robometer/failure_detection_env/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_PYTHON_LIB_ROOT=/data/yingxi/robofac/lib/python3.10/site-packages/nvidia
export CUDA_LIB_DIRS="$(find "$CUDA_PYTHON_LIB_ROOT" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:')"
export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MS_ASSET_DIR=/data/yingxi/robofac

"$PYTHON_BIN" run_train/robofpe_sft_data/collect_sft_data.py collect \
  --task-id PegInsertionVertical-v1 \
  --num-traj 3200 \
  --output-dir /data/yingxi/datasets/robofpe_sft/PegInsertionVertical-v1_render_wrist \
  --robot-uids panda_wristcam --success-only --randomize-initial-poses \
  --randomize-wrist-camera --randomize-render-camera --randomize-lighting \
  --save-video --sim-backend gpu --num-workers 16 --gpu-ids 0,1,2,3 \
  --max-attempts-per-traj 100 --seed 20260902

"$PYTHON_BIN" run_train/robofpe_sft_data/collect_sft_data.py collect \
  --task-id PegInsertionSide-v1 \
  --num-traj 3200 \
  --output-dir /data/yingxi/datasets/robofpe_sft/PegInsertionSide-v1_render_wrist \
  --robot-uids panda_wristcam --success-only --randomize-initial-poses \
  --randomize-wrist-camera --randomize-render-camera --randomize-lighting \
  --save-video --sim-backend gpu --num-workers 16 --gpu-ids 0,1,2,3 \
  --max-attempts-per-traj 100 --seed 30260902

"$PYTHON_BIN" run_train/robofpe_sft_data/collect_sft_data.py collect \
  --task-id PlugCharger-v1 \
  --num-traj 3200 \
  --output-dir /data/yingxi/datasets/robofpe_sft/PlugCharger-v1_render_wrist \
  --robot-uids panda_wristcam --success-only --randomize-initial-poses \
  --randomize-wrist-camera --randomize-render-camera --randomize-lighting \
  --save-video --sim-backend gpu --num-workers 16 --gpu-ids 0,1,2,3 \
  --max-attempts-per-traj 100 --seed 40260902

"$PYTHON_BIN" run_train/robofpe_sft_data/collect_sft_data.py collect \
  --task-id StackCube-v1 \
  --num-traj 3200 \
  --output-dir /data/yingxi/datasets/robofpe_sft/StackCube-v1_render_wrist \
  --robot-uids panda_wristcam --success-only --randomize-initial-poses \
  --randomize-wrist-camera --randomize-render-camera --randomize-lighting \
  --save-video --sim-backend gpu --num-workers 16 --gpu-ids 0,1,2,3 \
  --max-attempts-per-traj 100 --seed 50260902
```

Each task uses 16 independent single-environment subprocesses, four workers
per GPU. Worker seeds are separated by `1_000_000`; the merged manifest
contains all worker manifests and the final `lerobot` directory is generated
after all successful shards are present.

Render randomization extremes:

```bash
/data/yingxi/robometer/failure_detection_env/bin/python run_train/robofpe_sft_data/collect_sft_data.py preview-randomization \
  --task-id PushCube-v1 \
  --output-dir /data/yingxi/datasets/robofpe_sft/PushCube-v1_render_wrist/qc/extreme_randomization \
  --num-samples 8 \
  --extreme \
  --randomize-wrist-camera \
  --randomize-render-camera \
  --randomize-lighting
```

SFT config: `examples/sft/config/pushcube_sft_openpi_pi05_wrist.yaml`.

## Camera and SFT contract

`rlinf/models/embodiment/openpi/dataconfig/maniskill_dataconfig.py` keeps the
two OpenPI input slots unchanged. The dataset field named `top` is now
render-camera data, while the wrist field remains hand-camera data. Therefore
existing two-image SFT configs continue to read two images without adding a
third input.

## Future RL note

Future RL changes are intentionally not implemented in this update. The
planned runtime change is to capture `render_camera` as `main_images`, keep
`hand_camera` as `wrist_images`, and continue using exactly two policy image
inputs. The RL environment and RL launch configurations remain unchanged here.
