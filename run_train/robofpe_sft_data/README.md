# RoboFPE wrist-camera SFT collection

Default task: `PushCube-v1` with `robot_uids=panda_wristcam`.

Collect successful trajectories and convert to LeRobot:

```bash
/data/yingxi/robofac/bin/python run_train/robofpe_sft_data/collect_sft_data.py collect \
  --task-id PushCube-v1 \
  --num-traj 3200 \
  --output-dir /data/yingxi/datasets/robofpe_sft/PushCube-v1_wrist \
  --robot-uids panda_wristcam \
  --success-only \
  --randomize-initial-poses \
  --randomize-top-camera \
  --randomize-wrist-camera \
  --randomize-render-camera \
  --randomize-lighting \
  --seed 20260822
```

Render randomization extremes:

```bash
/data/yingxi/robofac/bin/python run_train/robofpe_sft_data/collect_sft_data.py preview-randomization \
  --task-id PushCube-v1 \
  --output-dir /data/yingxi/datasets/robofpe_sft/PushCube-v1_wrist/qc/extreme_randomization \
  --num-samples 8 \
  --extreme \
  --randomize-top-camera \
  --randomize-wrist-camera \
  --randomize-render-camera \
  --randomize-lighting
```

SFT config: `examples/sft/config/pushcube_sft_openpi_pi05_wrist.yaml`.
