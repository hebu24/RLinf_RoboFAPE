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
  --gpu-ids 4,5,6,7 \
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

### 3.1 Finetune from the generic `pi05_base` (dataset-computed norm_stats)

`sft_finetune.sh` starts from `/opt/kairan/models/RLinf-Pi05-ManiSkill-25Main-SFT`,
a checkpoint that **already bundles** an openpi `norm_stats.json` at
`<model_path>/physical-intelligence/maniskill/norm_stats.json`. The SFT chain loads
those stats in `get_model` (`load_norm_stats(checkpoint_dir, asset_id)`) and copies
them verbatim into each saved checkpoint — they are never recomputed from the SFT
dataset.

To finetune from the **generic** base `/opt/zhangchenyu/weights/pi05_base` instead,
and to compute **fresh norm_stats from the training data**, use the independent
wrapper `sft_finetune_pi05base.sh`:

```bash
cd /opt/yingxi/RLinf_RoboFAPE

DATA_DIR=/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200 \
GPU_IDS=0,1,2,3 \
bash sft_finetune_pi05base.sh
```

What it does (no core-library changes; the shared `pi05_base` dir is left untouched):

1. Builds a **prepared base** directory (default
   `run_train/peginsertion_maniskill_pi0.5/base/pi05_base_peg`) that *symlinks*
   `pi05_base/model.safetensors` (the 16.5 GB weights are not copied) and
   `config.json`.
2. Computes `norm_stats` from `DATA_DIR` by streaming the dataset through the same
   repack + data transforms the model sees (`observation.state_tcp -> state`,
   `actions -> actions`), accumulating openpi `RunningStats` (mean/std/q01/q99),
   and writing
   `base/pi05_base_peg/physical-intelligence/maniskill/norm_stats.json` in the
   standard openpi format (`toolkits/lerobot/calculate_norm_stats.py --output-dir`).
   The result is cached and reused on subsequent runs; set `FORCE_NORM_STATS=1` to
   recompute.
3. Launches the same Hydra SFT entrypoint as `sft_finetune.sh`, overriding
   `actor.model.model_path` to the prepared base and
   `runner.logger.experiment_name=peg_insertion_sft_pi05base`.

Because `actor.model.model_path` points at the prepared base, `get_model` loads the
pi05_base weights from the symlinked `model.safetensors` and loads the
freshly-computed `norm_stats.json`; on save, `FSDPVlaSftWorker.save_checkpoint`
copies that `norm_stats.json` into the new checkpoint. The saved checkpoint layout
is therefore identical to `sft_finetune.sh`
(`actor/{dcp_checkpoint/, model_state_dict/full_weights.pt, physical-intelligence/maniskill/norm_stats.json, data.pt, rng.pt}`)
and is drop-in compatible with section 4 (eval) and section 5 (PPO).

Environment variables:

```text
DATA_DIR            training data (default: .../peg_insertion_vertical_controller_3200)
PI05_BASE           generic base weights dir (default: /opt/zhangchenyu/weights/pi05_base)
PREPARED_BASE       where the symlinks + computed norm_stats live (default: .../base/pi05_base_peg)
GPU_IDS             physical GPU ids (default: 0,1,2,3)
FORCE_NORM_STATS=1  recompute norm_stats even if a cached file exists
CONFIG_NAME         hydra config (default: peg_insertion_sft_openpi_pi05)
OPENPI_CONFIG_NAME  openpi config used for norm_stats (default: pi05_maniskill_peg_insertion)
RESUME_DIR          optional checkpoint dir to resume from (same as sft_finetune.sh)
```

For the wrist variant, run in a detached tmux session so it survives SSH disconnect.
It starts its own Ray head on port `6379` (`SFT_RAY_PORT`), isolated from any eval
sweep on `6380`; GPUs 4-7 are disjoint from the eval's 0-3 so they don't OOM each
other. Safe to run concurrently with the §4 insert-only sweep — neither calls a bare
`ray stop`; teardown is scoped to each script's own port.

```bash
tmux new-session -d -s sft_pi05_wrist "cd /opt/yingxi/RLinf_RoboFAPE && \
  DATA_DIR=/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_12800 \
  GPU_IDS=4,5,6,7 \
  CONFIG_NAME=peg_insertion_sft_openpi_pi05_wrist \
  OPENPI_CONFIG_NAME=pi05_maniskill_peg_insertion_wrist \
  PREPARED_BASE=/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/base/pi05_base_peg_wrist \
  PYTHONUNBUFFERED=1 bash sft_finetune_pi05base.sh 2>&1 \
    | tee /opt/yingxi/RLinf_RoboFAPE/logs/sft_pi05base_wrist_tmux.log"
tmux attach -t sft_pi05_wrist   # detach: Ctrl-b d
```

### 3.2 Actual-EE-delta training (use_target=False eval)

Variant whose SFT labels are **actual-EE frame-to-frame deltas** so they match
the `use_target=False` eval regime (`pd_ee_delta_pose`, closed-loop: each delta
integrates from the actual EE). This is the closed-loop counterpart to the
default target-delta workflow and is more robust to drift (small-range retry
instead of large-range OOD departure).

The fastest path is to **convert the existing 3200 dataset** (which already
stores `debug.tcp_after` per step from a `use_target=True` collection run):
labels are rewritten from target-delta to actual-EE-delta in place, videos are
symlinked, and `stats.json` is recomputed.

```bash
cd /opt/yingxi/RLinf_RoboFAPE
/opt/kairan/envs/rlinf/bin/python run_train/peginsertion_maniskill_pi0.5/convert_controller_to_actual_ee.py \
  --src  /opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_controller_3200 \
  --dst  /opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_actual_ee_3200
```

To collect fresh actual-EE-delta data instead (execution still uses
`use_target=True` for high tracking success; labels are derived from the
recorded post-step TCP), run the variant collector:

```bash
cd /opt/yingxi/RLinf_RoboFAPE
MPLCONFIGDIR=/tmp/matplotlib \
/opt/kairan/envs/rlinf/bin/python run_train/peginsertion_maniskill_pi0.5/collect_peg_insertion_actual_ee_delta.py \
  --num-traj 3200 \
  --output-dir /opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_actual_ee_3200 \
  --seed 0 \
  --gpu-ids 0
```

Train on the actual-EE-delta dataset **from the generic `pi05_base` with
freshly-computed norm_stats** (the actual-EE-delta action distribution differs
from the target-delta/base stats, so norm_stats must be recomputed). Reuse
`sft_finetune_pi05base.sh` (§3.1) with the actual-EE config/data/asset dirs;
it symlinks `pi05_base` weights, computes `norm_stats.json` from the
actual-EE-delta dataset, and writes both into a dedicated prepared-base dir:

```bash
cd /opt/yingxi/RLinf_RoboFAPE
DATA_DIR=/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_actual_ee_3200 \
CONFIG_NAME=peg_insertion_sft_openpi_pi05_actual_ee \
OPENPI_CONFIG_NAME=pi05_maniskill_peg_insertion_actual_ee \
PREPARED_BASE=/opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/base/pi05_base_peg_actual_ee \
EXPERIMENT_NAME=peg_insertion_sft_actual_ee \
GPU_IDS=0,1,2,3 \
bash sft_finetune_pi05base.sh
```

`PREPARED_BASE` is kept separate (`pi05_base_peg_actual_ee` vs the target-delta
`pi05_base_peg`) so the two norm_stats files do not overwrite each other.
`FORCE_NORM_STATS=1` recomputes; otherwise the cached `norm_stats.json` is
reused. The SFT config (`examples/sft/config/peg_insertion_sft_openpi_pi05_actual_ee.yaml`) uses:

```text
config_name: pi05_maniskill_peg_insertion_actual_ee
num_images_in_input: 2          # wrist variant
num_action_chunks: 10
action_horizon: 10
add_value_head: False
```

Action label semantics (also written to the dataset `meta/info.json`):

```text
policy output: [dx, dy, dz, droll, dpitch, dyaw, gripper]   # actual-EE-delta
rotation: Euler XYZ
execution controller: ManiSkill pd_ee_target_delta_pose (use_target=True, for tracking success)
label source: actions[t] = tcp_after[t+1] - tcp_after[t]  (from recorded post-step TCP)
required eval controller: pd_ee_delta_pose (use_target=False, closed-loop)
required eval action scale: 1.0
```

## 4. Evaluate SFT Checkpoint

Use the actor checkpoint produced by SFT:

```text
logs/<run>/peg_insertion_sft/checkpoints/global_step_<N>/actor
```

### Ray isolation: run eval concurrently with SFT training

SFT (`sft_finetune*.sh`) and eval each own a dedicated Ray head on a distinct GCS
port, so neither can kill the other (the node-wide `ray stop` is gone; teardown is
scoped to each script's own port). Defaults:

- SFT head → port `6379` (`SFT_RAY_PORT`, temp-dir `/tmp/ray_sft*`).
- Eval head → port `6380` (`EVAL_RAY_PORT` for single-eval, `--ray-port` for the
  sweep; temp-dir `/tmp/ray_eval_wrist*`).

Two rules for concurrent SFT + eval:

1. **Disjoint GPUs** — Ray isolation does not isolate GPU memory. Keep SFT and eval
   on different GPUs (e.g. SFT `GPU_IDS=4,5,6,7`, eval `--gpu-ids 0,1,2,3`), otherwise
   they OOM each other.
2. **Single-checkpoint eval needs `MANAGE_RAY=true`** — the default
   `MANAGE_RAY=false` only attaches and errors if no 6380 head is up. The sweep
   manages its own 6380 head automatically; override either port if needed
   (`EVAL_RAY_PORT=6381` / `--ray-port 6381`).

Run evaluation with `EVAL_ACTION_SCALE=1.0`:

```bash
cd /opt/yingxi/RLinf_RoboFAPE

# Concurrent with SFT: own Ray head on port 6380, disjoint GPUs.
VENV_DIR=/opt/kairan/envs/rlinf \
CHECKPOINT_PATH=/opt/yingxi/RLinf_RoboFAPE/logs/20260712-17:15:54-peg_insertion_sft_openpi_pi05-3200/peg_insertion_sft/checkpoints/global_step_20000/actor \
GPU_IDS=0 \
NUM_EVAL_EPISODES=8 \
NUM_ENVS=1 \
EVAL_ACTION_SCALE=1.0 \
SAVE_VIDEO=true \
MANAGE_RAY=true \
EVAL_RAY_PORT=6380 \
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

# Concurrent with SFT: own Ray head on port 6380, disjoint GPUs.
VENV_DIR=/opt/kairan/envs/rlinf \
CHECKPOINT_PATH=/opt/yingxi/RLinf_RoboFAPE/logs/20260711-00:18:32-peg_insertion_sft_openpi_pi05_wrist-3200/peg_insertion_sft_wrist/checkpoints/global_step_20000/actor \
GPU_IDS=0-3 \
NUM_EVAL_EPISODES=8 \
NUM_ENVS=4 \
EVAL_ACTION_SCALE=1.0 \
SAVE_VIDEO=true \
MANAGE_RAY=true \
EVAL_RAY_PORT=6380 \
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

# Own Ray head on port 6380, disjoint GPUs from SFT (SFT on 0-3, sweep on 4-7 here).
MPLCONFIGDIR=/tmp/matplotlib \
/opt/kairan/envs/rlinf/bin/python run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
  --ray-port 6380 \
  --checkpoint-dir /opt/yingxi/RLinf_RoboFAPE/logs/20260713-01:53:11-peg_insertion_sft_openpi_pi05_wrist-3200/peg_insertion_sft_wrist/checkpoints \
  --output-dir /opt/yingxi/RLinf_RoboFAPE/logs/20260713-01:53:11-peg_insertion_sft_openpi_pi05_wrist-3200/peg_insertion_sft_wrist/wrist_eval_sweep_rtc \
  --num-eval-episodes 10 \
  --num-envs 1 \
  --gpu-ids 4,5,6,7 \
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

### Insert-only wrist checkpoint evaluation

The wrist SFT policy is trained on full pick-up-and-insert demonstrations, so a
plain eval conflates the difficult pick-up phase (which suffers from BC compound
errors) with the insert phase. The insert-only setting isolates the move-and-
insert skill: every episode is initialized by motion-planning the **grasp +
lift** with the existing RoboFPE solver (`PegInsertionLiftPlanner`, the same one
used for SFT data collection), then handing the grasped, lifted peg to the
policy. The policy is evaluated only on transport + align + descend + insert.

Each episode re-runs the solver fresh on a CPU single-environment (per-episode
planning, max variety); the lifted state `(robot_qpos, peg_pose, hole_pose)` is
replayed kinematically into the GPU eval env via reset options, so no IK or
hand-constructed grasp pose is needed. The obs/action config is identical to the
wrist eval above, so the eval guard passes unchanged.

Use the insert-only launcher for checkpoints trained with `sft_finetune_wrist.sh`:

```bash
cd /opt/yingxi/RLinf_RoboFAPE

# Concurrent with SFT: own Ray head on port 6380, disjoint GPUs.
VENV_DIR=/opt/kairan/envs/rlinf \
CHECKPOINT_PATH=/opt/yingxi/RLinf_RoboFAPE/logs/20260711-00:18:32-peg_insertion_sft_openpi_pi05_wrist-3200/peg_insertion_sft_wrist/checkpoints/global_step_20000/actor \
GPU_IDS=0-3 \
NUM_EVAL_EPISODES=8 \
NUM_ENVS=4 \
EVAL_ACTION_SCALE=1.0 \
SAVE_VIDEO=true \
MANAGE_RAY=true \
EVAL_RAY_PORT=6380 \
bash run_train/eval_checkpoint/run_peginsertion_wrist_insert_only.sh
```

`run_peginsertion_wrist_insert_only.sh` defaults to
`maniskill_peg_insertion_vertical_wrist_sft_eval_openpi_pi05_insert_only` and a
200-step horizon (transport+align+insert; divisible by both `num_action_chunks`
and `execute_action_chunks`). The saved eval videos show the peg starting
grasped in the air with the hole offset, and the policy transporting/aligning/
inserting (base and wrist camera side by side).

To sweep every `global_step_*/actor` checkpoint under a wrist SFT run in
insert-only mode, pass `--run-script`:

```bash
tmux new-session -d -s eval_sweep "cd /opt/yingxi/RLinf_RoboFAPE && \
  MPLCONFIGDIR=/tmp/matplotlib \
  /opt/kairan/envs/rlinf/bin/python run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
    --ray-port 6380 \
    --run-script run_train/eval_checkpoint/run_peginsertion_wrist_insert_only.sh \
    --checkpoint-dir /opt/yingxi/RLinf_RoboFAPE/logs/20260713-01:53:11-peg_insertion_sft_openpi_pi05_wrist-3200/peg_insertion_sft_wrist/checkpoints \
    --output-dir /opt/yingxi/RLinf_RoboFAPE/logs/20260713-01:53:11-peg_insertion_sft_openpi_pi05_wrist-3200/peg_insertion_sft_wrist/wrist_insert_only_eval_sweep \
    --num-eval-episodes 10 --num-envs 1 --gpu-ids 0,1,2,3 --action-scale 1.0 \
    2>&1 | tee /opt/yingxi/RLinf_RoboFAPE/logs/eval_sweep_insert_only_tmux.log"
tmux attach -t eval_sweep   # detach: Ctrl-b d
```

Per-episode planning adds a CPU solve (~seconds) per episode, so insert-only
sweeps are slower than full-task sweeps at the same episode count.

### Actual-EE-delta checkpoint evaluation

Evaluate checkpoints trained with `sft_finetune_pi05base.sh` (actual-EE-delta
config, §3.2) under the `use_target=False` regime (`pd_ee_delta_pose`,
closed-loop). Use the `run_peginsertion_actual_ee.sh` wrapper, which points at
`maniskill_peg_insertion_vertical_sft_eval_openpi_pi05_actual_ee` (sets
`policy_setup: panda-ee-dpose`). `eval_checkpoint.py` accepts both
`panda-ee-target-dpose` and `panda-ee-dpose`.

```bash
cd /opt/yingxi/RLinf_RoboFAPE

VENV_DIR=/opt/kairan/envs/rlinf \
CHECKPOINT_PATH=/path/to/peg-insertion-actual-ee-sft/checkpoints/global_step_N/actor \
GPU_IDS=0-3 \
NUM_EVAL_EPISODES=8 \
NUM_ENVS=4 \
EVAL_ACTION_SCALE=1.0 \
SAVE_VIDEO=true \
MANAGE_RAY=true \
EVAL_RAY_PORT=6380 \
bash run_train/eval_checkpoint/run_peginsertion_actual_ee.sh
```

To sweep every `global_step_*/actor` checkpoint under an actual-EE SFT run:

```bash
cd /opt/yingxi/RLinf_RoboFAPE

MPLCONFIGDIR=/tmp/matplotlib \
/opt/kairan/envs/rlinf/bin/python run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
  --run-script run_train/eval_checkpoint/run_peginsertion_actual_ee.sh \
  --ray-port 6390 \
  --checkpoint-dir /path/to/peg-insertion-actual-ee-sft/checkpoints \
  --output-dir  /path/to/actual_ee_eval_sweep \
  --num-eval-episodes 10 \
  --num-envs 1 \
  --gpu-ids 4,5,7 \
  --action-scale 1.0 \
  --resume --continue-on-error
```

Avoid `--gpu-ids 6` on this host: GPU6's Vulkan renderer intermittently hangs
(`ErrorDeviceLost` / camera-group creation failure). `--ray-port 6390` keeps the
sweep's Ray head disjoint from SFT (6379) and the target-delta eval sweeps
(6380). `--resume` skips checkpoints that already wrote `trajectory_metrics.json`;
`--continue-on-error` records a failed checkpoint and continues.

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
