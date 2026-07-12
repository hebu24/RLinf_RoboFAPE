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

The collector first generates a successful motion-planning reference trajectory,
then dry-runs a no-image `pd_ee_target_delta_pose` tracker.  The reference TCP
trajectory is converted to physical target-delta actions; any step outside the
Panda 0.1m / 0.1rad controller bounds is split into smaller controller steps
and recorded as actual training frames. Only controller dry-runs that reach task
success are replayed once more with RGB sensors enabled and written as training
episodes.

The default collection command uses CPU physics simulation and GPU/Vulkan
rendering (`sim_backend="cpu"`, `render_backend="gpu:<id>"`) and avoids RGB
observations for failed reference/controller attempts.  Each worker keeps
persistent envs alive and loops with `reset(seed=...)` only:

- one no-image `pd_joint_pos` reference env
- one no-image `pd_ee_target_delta_pose` dry-run env
- one RGB `pd_ee_target_delta_pose` capture env

This avoids repeated svulkan2 renderer create/close cycles, which can poison
Vulkan state and later appear as misleading `IncompatibleDriver` or "driver does
not support Vulkan" errors.  It also avoids rendering images for failed
controller attempts.

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

MPLCONFIGDIR=/tmp/matplotlib \
/opt/kairan/envs/rlinf/bin/python run_train/peginsertion_maniskill_pi0.5/collect_peg_insertion_controller_data.py \
  --num-traj 3200 \
  --output-dir /opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200 \
  --seed 0 \
  --num-workers 32 \
  --gpu-ids 0,1,2,3 \
  --worker-stagger 5.0
```

CPU renderer fallback for small smoke tests:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/opt/kairan/envs/rlinf/bin/python run_train/peginsertion_maniskill_pi0.5/collect_peg_insertion_controller_data.py \
  --num-traj 1 \
  --output-dir /tmp/peg_controller_cpu_render_smoke \
  --seed 0 \
  --num-workers 1 \
  --gpu-ids 0 \
  --render-backend-prefix cpu
```

CPU rendering is much slower.  Use it to verify the data path or as a fallback
when Vulkan is not exposed in the current container/session.

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

## 3. Optional State-TCP Refresh

Controller-domain collection already writes `observation.state_tcp`, so this
step is normally unnecessary.  If you need to refresh proprio only, run:

```bash
/opt/kairan/envs/rlinf/bin/python run_train/peginsertion_maniskill_pi0.5/convert_qpos_to_tcp_proprio.py \
  --data-dir /opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200 \
  --render-backend gpu:0
```

This preserves `actions` by default and writes FK deltas only to
`debug.fk_delta_action`.  The legacy flag `--overwrite-actions-from-fk` should
not be used for controller-domain SFT data.

## 4. SFT Training

Train from the base pi0.5 ManiSkill checkpoint.  Do not resume a checkpoint that
was trained on the old FK-converted data.

```bash
cd /opt/yingxi/RLinf_RoboFAPE

DATA_DIR=/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200 \
bash sft_finetune.sh
```

The base SFT config uses:

```text
config_name: pi05_maniskill_peg_insertion
num_images_in_input: 1
num_action_chunks: 10
action_horizon: 10
add_value_head: False
```

## 5. Evaluate SFT Checkpoint

Use the actor checkpoint produced by SFT:

```text
logs/<run>/peg_insertion_sft/checkpoints/global_step_<N>/actor
```

Run evaluation with `EVAL_ACTION_SCALE=1.0`:

```bash
cd /opt/yingxi/RLinf_RoboFAPE

VENV_DIR=/opt/kairan/envs/rlinf \
CHECKPOINT_PATH=/opt/yingxi/RLinf_RoboFAPE/logs/<run>/peg_insertion_sft/checkpoints/global_step_<N>/actor \
GPU_IDS=0-3 \
NUM_EVAL_EPISODES=25 \
NUM_ENVS=5 \
EVAL_ACTION_SCALE=1.0 \
SAVE_VIDEO=true \
bash run_train/eval_checkpoint/run_peginsertion.sh
```

`run_peginsertion.sh` defaults to
`maniskill_peg_insertion_vertical_sft_eval_openpi_pi05`.  The eval guard rejects
generic `pi05_maniskill` configs, wrong image counts, and wrong action horizons
for `PegInsertionVertical-v1`.

Key output files:

| Path | Content |
|---|---|
| `logs/<timestamp>-eval-PegInsertionVertical-v1/eval.log` | Resolved config and metrics. |
| `logs/<timestamp>-eval-PegInsertionVertical-v1/evaluation_summary.json` | Evaluation summary. |
| `logs/<timestamp>-eval-PegInsertionVertical-v1/video/eval/` | Evaluation videos. |

## 6. PPO / RL From SFT

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
