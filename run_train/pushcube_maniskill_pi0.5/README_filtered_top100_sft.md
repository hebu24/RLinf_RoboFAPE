# PushCube pi0.5 SFT from Robometer-filtered Top100

This experiment finetunes `pi0.5` on PushCube trajectories selected by the
Robometer data-filtering protocol.

## Fixed Setting

- Query task: `PushCube-v1`
- Filtering domain: `IND`
- Filtering model: `ours_224`
- Filtering score: `final`
- K: `100`
- SFT source task: `PushCube-v1`
- SFT cameras: `observation.images.top` from `base_camera`, and
  `observation.images.wrist` from `hand_camera`
- Evaluation seeds: `0-7`

The Robometer filtering Top100 can include non-PushCube candidates.  Those rows
cannot be replayed in `PushCube-v1`, so the SFT builder skips non-PushCube rows
and continues down the same ranked list until it writes 100 PushCube episodes.
The skipped rows and final selected episodes are recorded in
`filtering_selection_manifest.{json,csv}` next to the generated dataset.

## Build SFT Dataset Only

```bash
cd /data/yingxi/RLinf_RoboFAPE

/data/yingxi/RLinf_RoboFAPE/.venv/bin/python \
  run_train/robofpe_sft_data/build_pushcube_sft_from_filtering.py \
  --out-dir /data/yingxi/datasets/robofpe_sft/PushCube-v1_wrist_filtered_top100_robometer/lerobot \
  --domain IND \
  --model-name ours_224 \
  --query-task PushCube-v1 \
  --source-task-id PushCube-v1 \
  --score-method final \
  --top-k 100 \
  --overwrite
```

For a fast smoke test:

```bash
cd /data/yingxi/RLinf_RoboFAPE

/data/yingxi/RLinf_RoboFAPE/.venv/bin/python \
  run_train/robofpe_sft_data/build_pushcube_sft_from_filtering.py \
  --out-dir /tmp/pushcube_filtering_sft_smoke/lerobot \
  --limit 1 \
  --max-episode-frames 4 \
  --overwrite
```

## Train SFT

```bash
cd /data/yingxi/RLinf_RoboFAPE

GPU_IDS=0,1 \
OVERWRITE_DATASET=1 \
bash run_train/pushcube_maniskill_pi0.5/run_sft_filtered_top100.sh
```

Defaults use the existing config
`examples/sft/config/pushcube_sft_openpi_pi05_wrist.yaml`, i.e.
`max_steps=40000`, `save_interval=5000`, and `lr=3e-6`.  To override explicitly:

```bash
EXTRA_HYDRA='runner.max_steps=10000 actor.optim.total_training_steps=10000 runner.save_interval=1000' \
GPU_IDS=0,1 \
BUILD_DATASET=0 \
bash run_train/pushcube_maniskill_pi0.5/run_sft_filtered_top100.sh
```

## Evaluate SR vs Training Step

After the SFT run starts writing `global_step_*/actor` checkpoints:

```bash
cd /data/yingxi/RLinf_RoboFAPE

CHECKPOINT_DIR=/path/to/sft_run/checkpoints \
SWEEP_GPU_IDS=0,1 \
SWEEP_SEEDS=0-7 \
bash run_train/pushcube_maniskill_pi0.5/watch_eval_sft_filtered_top100.sh
```

The sweep writes:

- `pushcube_sweep_metrics.csv`
- `pushcube_sweep_metrics.json`
- `sr_vs_step_multiseed.png`
- `index.html`
