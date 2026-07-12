# PegInsertionVertical pi0.5 SFT Workflow

This workflow uses controller-domain data.  The SFT label is the same action
domain used at evaluation time:

```text
policy output: [dx, dy, dz, droll, dpitch, dyaw, gripper]
rotation: Euler XYZ
eval controller: ManiSkill pd_ee_target_delta_pose
env action mapping: env_action[:3] = action[:3] / 0.1; env_action[3:6] = -action[3:6] / 0.1
required action scale: 1.0
```

Do not train new checkpoints from the old FK-converted dataset
`peg_insertion_vertical_3200`.  That dataset stores actions inferred from
motion-planning qpos/TCP deltas and may contain old gripper/rotation semantics,
not the exact target-delta controller domain used for eval.

## 1. Collect Controller-Domain Data
First run a 10-episode GPU smoke collection:
```bash
cd /opt/yingxi/RLinf_RoboFAPE

MPLCONFIGDIR=/tmp/matplotlib \
/opt/kairan/envs/rlinf/bin/python run_train/peginsertion_maniskill_pi0.5/collect_peg_insertion_controller_data.py \
  --num-traj 10 \
  --output-dir /opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_smoke \
  --seed 0 \
  --num-workers 1 \
  --gpu-ids 0
```

Every accepted episode is flushed immediately: videos, per-episode parquet, and
lightweight meta are written before the next seed starts.  `stats.json` is
finalized at the end of collection.  If a long smoke run has already written one
or more parquet files, you can replay the available subset with
`--num-episodes 1` or another number not larger than the current parquet count.

The collector also appends one diagnostic row per attempted seed:

```text
<data-dir>/meta/collection_attempts.jsonl
```

Use it to locate low pass rates.  `ref_fail` means the motion-planning
reference did not succeed; `ctrl_dry_fail` means the `pd_ee_target_delta_pose`
controller could not reproduce a successful reference without RGB capture;
`ctrl_capture_fail` means dry-run succeeded but the RGB replay did not.
Successful rows also include target-delta plan length, split count, controller
bound summary, and base-model normalized state/action p99/max diagnostics.
Final `meta/stats.json` also includes `base_norm.actions.abs` and
`base_norm.observation.state_tcp.abs` so you can check OOD ranges before SFT.
The generated LeRobot metadata uses `codebase_version: v2.0` because the
collector writes global `meta/stats.json`, which matches the v2.0 layout used
by the installed LeRobot reader.

Then run replay smoke on that dataset:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/opt/kairan/envs/rlinf/bin/python run_train/peginsertion_maniskill_pi0.5/replay_controller_dataset.py \
  --data-dir /opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_smoke \
  --num-episodes 10 \
  --render-backend gpu:0 \
  --control-mode pd_ee_target_delta_pose \
  --action-scale 1.0
```

Do not proceed to full collection, SFT, or eval unless this replay passes with
`env_success_rate` close to `1.0` and `env_action_max_abs_error` near `0`.

After the 10-episode smoke gate passes, collect the full dataset:

```bash
cd /opt/yingxi/RLinf_RoboFAPE

/opt/kairan/envs/rlinf/bin/ray stop 

MPLCONFIGDIR=/tmp/matplotlib \
/opt/kairan/envs/rlinf/bin/python run_train/peginsertion_maniskill_pi0.5/collect_peg_insertion_controller_data.py \
  --num-traj 12800 \
  --output-dir /opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_12800 \
  --seed 0 \
  --num-workers 32 \
  --gpu-ids 0,1,2,3 \
  --worker-stagger 5.0
```

Expected dataset fields include:

| Field | Meaning |
|---|---|
| `actions` | Raw policy labels: physical target-delta meter/radian deltas plus binary gripper (`+1` open, `-1` close). |
| `debug.env_action` | Exact normalized action sent to `pd_ee_target_delta_pose`; note the rotation sign flip. |
| `observation.state_tcp` | 8D pi0.5 TCP proprio. |
| `episode_reset_state` | ManiSkill flat state for strict replay. |
| `debug.tcp_before` / `debug.tcp_after` | TCP matrices for replay diagnostics. |

## 2. Replay Smoke

Run strict replay before training.  This restores each saved episode state and
executes the stored controller-domain actions with `action_scale=1.0`.

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/opt/kairan/envs/rlinf/bin/python run_train/peginsertion_maniskill_pi0.5/replay_controller_dataset.py \
  --data-dir /opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200 \
  --num-episodes 20 \
  --seed 0 \
  --render-backend gpu:0 \
  --control-mode pd_ee_target_delta_pose \
  --action-scale 1.0
```

The output is written to:

```text
run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200/meta/controller_replay_smoke.json
```

Blocking checks:

| Metric | Expected |
|---|---|
| `env_success_rate` | Close to `1.0` for sampled successful episodes. |
| `env_action_max_abs_error` | Near `0`; validates the documented position/rotation env-action mapping. |
| `qpos_error_max` / `tcp_error_max` | Small; validates deterministic controller replay. |

If this replay needs `--action-scale 2.5`, the data is still wrong for the eval
controller.  Do not train from that dataset.

## 3. SFT Training

Train from the base pi0.5 ManiSkill checkpoint.  Do not resume a checkpoint that
was trained on the old FK-converted data.

```bash
cd /opt/yingxi/RLinf_RoboFAPE

DATA_DIR=/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200 \
bash sft_finetune.sh
```

Wrist SFT:

```bash
cd /opt/yingxi/RLinf_RoboFAPE

DATA_DIR=/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200 \
bash sft_finetune_wrist.sh
```

The SFT wrappers set `RAY_TMPDIR` under `/tmp` to avoid Ray socket path length
limits on long repo paths.

The base SFT config uses:

```text
config_name: pi05_maniskill_peg_insertion
num_images_in_input: 1
num_action_chunks: 10
action_horizon: 10
add_value_head: False
```

The wrist SFT config uses:

```text
config_name: pi05_maniskill_peg_insertion_wrist
num_images_in_input: 2
num_action_chunks: 10
action_horizon: 10
add_value_head: False
```

It also sets `env.train.use_wrist_image: true` and `env.eval.use_wrist_image: true`
so ManiSkill's `hand_camera` is routed into `wrist_images`.

## 4. Evaluate SFT Checkpoint

Use the actor checkpoint produced by SFT:

```text
logs/<run>/peg_insertion_sft/checkpoints/global_step_<N>/actor
```

Run evaluation with `EVAL_ACTION_SCALE=1.0`:

```bash
cd /opt/yingxi/RLinf_RoboFAPE

VENV_DIR=/opt/kairan/envs/rlinf \
CHECKPOINT_PATH=/opt/yingxi/RLinf_RoboFAPE/logs/20260712-17:15:54-peg_insertion_sft_openpi_pi05-3200/peg_insertion_sft/checkpoints/global_step_20000/actor \
GPU_IDS=0 \
NUM_EVAL_EPISODES=8 \
NUM_ENVS=1 \
EVAL_ACTION_SCALE=1.0 \
SAVE_VIDEO=true \
bash run_train/eval_checkpoint/run_peginsertion.sh
```

`run_peginsertion.sh` defaults to
`maniskill_peg_insertion_vertical_sft_eval_openpi_pi05`.  The eval guard rejects
generic `pi05_maniskill` configs, wrong image counts, and wrong action horizons
for `PegInsertionVertical-v1`.

### Wrist checkpoint evaluation

Use the wrist eval wrapper for checkpoints trained with `sft_finetune_wrist.sh`:

```bash
cd /opt/yingxi/RLinf_RoboFAPE

VENV_DIR=/opt/kairan/envs/rlinf \
CHECKPOINT_PATH=/opt/yingxi/RLinf_RoboFAPE/logs/20260711-00:18:32-peg_insertion_sft_openpi_pi05_wrist-3200/peg_insertion_sft_wrist/checkpoints/global_step_20000/actor \
GPU_IDS=0-3 \
NUM_EVAL_EPISODES=8 \
NUM_ENVS=4 \
EVAL_ACTION_SCALE=1.0 \
SAVE_VIDEO=true \
bash run_train/eval_checkpoint/run_peginsertion_wrist.sh
```

`run_peginsertion_wrist.sh` defaults to
`maniskill_peg_insertion_vertical_wrist_sft_eval_openpi_pi05`.  The eval guard
rejects generic `pi05_maniskill` configs, wrong image counts, and wrong action
horizons for `PegInsertionVertical-v1`.  The saved eval videos show the base
camera and wrist camera side by side.

To evaluate every `global_step_*/actor` checkpoint under a wrist SFT run, run:

```bash
cd /opt/yingxi/RLinf_RoboFAPE

MPLCONFIGDIR=/tmp/matplotlib \
/opt/kairan/envs/rlinf/bin/python run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
  --checkpoint-dir /opt/yingxi/RLinf_RoboFAPE/logs/20260711-00:18:32-peg_insertion_sft_openpi_pi05_wrist-3200/peg_insertion_sft_wrist/checkpoints \
  --output-dir /opt/yingxi/RLinf_RoboFAPE/logs/20260711-00:18:32-peg_insertion_sft_openpi_pi05_wrist-3200/peg_insertion_sft_wrist/wrist_eval_sweep \
  --num-eval-episodes 10 \
  --num-envs 2 \
  --gpu-ids 2,3 \
  --action-scale 1.0
```

The sweep runs one checkpoint per GPU.  For example, `--gpu-ids 2,3` evaluates
two checkpoints concurrently, assigns each checkpoint to one GPU, and continues
round-robin as GPUs finish.  Videos are saved by default; pass `--no-save-video`
for a metrics-only sweep.

The sweep writes:

| Path | Content |
|---|---|
| `wrist_sweep_metrics.csv` | Per-checkpoint step, mean success rate, and mean trajectory max reward. |
| `wrist_sweep_metrics.json` | Same metrics plus checkpoint/log paths. |
| `wrist_sweep_curves.png` | Mean success-rate and mean max-reward curves. |
| `success_rate_vs_step.png` | Mean SR vs. training step. |
| `max_reward_vs_step.png` | Mean trajectory max reward vs. training step. |

Key output files:

| Path | Content |
|---|---|
| `logs/<timestamp>-eval-PegInsertionVertical-v1/eval.log` | Resolved config and metrics. |
| `logs/<timestamp>-eval-PegInsertionVertical-v1/evaluation_summary.json` | Evaluation summary. |
| `logs/<timestamp>-eval-PegInsertionVertical-v1/video/eval/` | Evaluation videos. |

## 5. PPO / RL From SFT

Only start RL after the SFT checkpoint passes the SFT eval sanity check above.
The peg PPO config is aligned to the same controller-domain interface:

```text
config_name: pi05_maniskill_peg_insertion
num_images_in_input: 1
num_action_chunks: 10
action_horizon: 10
policy_setup: panda-ee-target-dpose
env.train.action_scale: 1.0
env.eval.action_scale: 1.0
```

Run RL with an SFT checkpoint as `MODEL_PATH`:

```bash
MODEL_PATH=/opt/yingxi/RLinf_RoboFAPE/logs/<run>/peg_insertion_sft/checkpoints/global_step_<N>/actor \
bash run_train/peginsertion_maniskill_pi0.5/run.sh
```

## Troubleshooting

- `failed to find device "cuda:0"`: run collection/eval on a node with Vulkan
  render devices visible, or choose valid `--gpu-ids`.
- Repeated create/close reproductions that fail after a fixed number of env
  creations are svulkan2 lifecycle failures, not headless-mode failures. Use the
  controller collector's persistent worker envs and avoid scripts that recreate
  ManiSkill envs for every seed.
- `env_success_rate=0` in strict replay: do not train.  The controller-domain
  dataset is not replayable with `action_scale=1.0`.
- Eval guard rejects config: use the peg SFT eval config or fix
  `config_name`, `num_images_in_input`, `num_action_chunks`, and
  `action_horizon`.
- Old checkpoints trained before controller-domain collection should be
  discarded for SFT validation.  Re-train from the base model.
