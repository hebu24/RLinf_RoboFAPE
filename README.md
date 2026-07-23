# PegInsertionVertical pi0.5 SFT + Eval (single-wrist, insert-only)

Single-wrist peg-insertion VLA fine-tuning and evaluation on OpenPI pi0.5,
starting from the generic `pi05_base` weights, on the insert-only task.
Controller-domain target-delta actions throughout (policy output
`[dx, dy, dz, droll, dpitch, dyaw, gripper]`, Euler XYZ, executed by the
ManiSkill `pd_ee_target_delta_pose` controller at `action_scale=1.0`).

## Setup

- Repo: `/data/yingxi/RLinf_RoboFAPE`
- Venv: `/data/yingxi/kairan/envs/rlinf` (`bin/python`, `bin/ray`)
- Base weights: `/data/yingxi/weights/pi05_base`
- xulab `/` fills up fast — keep Ray tmp, `HF_HOME`, and `TMPDIR` on `/data`
  (`export TMPDIR=/data/yingxi/tmp HF_HOME=/data/yingxi/.cache/huggingface`).
- Insert-only wrist data: `run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_insert_only_3200`
  (regenerate with `collect_peg_insertion_controller_data.py --collect-mode insert_only`).
- RoboFAPE solver: the insert-only eval's `PegInsertionLiftPlanner` (and the data collector) import `solutions` from here — the code default points at the H100 path, so set it on xulab:
  `export RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs`
- Ray isolation: every Ray cluster needs a distinct GCS port + dashboard port +
  temp-dir + disjoint GPUs; never a bare `ray stop` (it kills all clusters on the
  host). See `RAY_ISOLATION.md` for the port allocation table and scoped teardown.

## Current setting (spec for future PPO)

Any new RL/PPO must match this so the SFT checkpoint and the policy interface line up:

| Item | Value |
|---|---|
| Base model | `pi05_base` (`/data/yingxi/weights/pi05_base`) |
| OpenPI config | `pi05_maniskill_peg_insertion_wrist` |
| Camera | base + single wrist (`use_wrist_image=true`, `num_images_in_input=2`) |
| Task | insert-only (`reset_options.pre_grasped=true`, 600 steps) |
| Action chunks | `num_action_chunks=10`, `action_horizon=10`, `execute_action_chunks=10` |
| Action | `[dx,dy,dz,droll,dpitch,dyaw,gripper]`, Euler XYZ, target-delta |
| Controller | `panda-ee-target-dpose` / `pd_ee_target_delta_pose`, `use_target=True`, `action_scale=1.0` |
| norm_stats | use the dataset's `meta/openpi/<config>/norm_stats.json` if present, else recompute (asset_id `physical-intelligence/maniskill`) |

## 1. SFT training (single-wrist insert-only)

`run_sft_insert_wrist_v2.sh` is the canonical launcher: it points at the
single-wrist insert-only dataset, the wrist OpenPI config, the pi05_base
prepared-base, and the insert-only hydra tweaks (mask gripper loss, lower LR +
warmup for the OOD insert-only start distribution). Run it in a persistent tmux:

```bash
cd /data/yingxi/RLinf_RoboFAPE
tmux new -s sft_wrist
bash run_sft_insert_wrist_v2.sh        # tees to logs/sft_insert_wrist_v2_tmux.log
```

It wraps `sft_finetune_pi05base.sh` with:

```text
DATA_DIR=.../data/peg_insertion_vertical_insert_only_3200
GPU_IDS=4,5,6,7
SFT_RAY_PORT=6379  SFT_DASHBOARD_AGENT_PORT=52366
CONFIG_NAME=peg_insertion_sft_openpi_pi05_wrist
OPENPI_CONFIG_NAME=pi05_maniskill_peg_insertion_wrist
PREPARED_BASE=.../base/pi05_base_peg_wrist_insert
EXPERIMENT_NAME=peg_insertion_sft_insert_only_wrist_v2
```

`pi05_base` weights are symlinked into `$PREPARED_BASE` (never copied). For
norm_stats, if `$DATA_DIR/meta/openpi/$OPENPI_CONFIG_NAME/norm_stats.json`
exists it is used directly; otherwise it is recomputed from the data via
`toolkits/lerobot/calculate_norm_stats.py` and cached back into the dataset.
Checkpoints land in
`logs/<ts>-<exp>/peg_insertion_sft*/checkpoints/global_step_<N>/actor`.
If you change `runner.max_steps`, also set `actor.optim.total_training_steps`
to match (cosine LR). Each pi0.5 checkpoint is ~16-32GB; check `df -h /data`.

## 2. Evaluate an SFT checkpoint

Actor checkpoint to evaluate:

```text
logs/<ts>-peg_insertion_sft_*/peg_insertion_sft*/checkpoints/global_step_<N>/actor
```

### Wrist insert-only eval

For insert-only-trained checkpoints (peg pre-grasped + lifted by
`PegInsertionLiftPlanner`, policy does transport + align + insert, default
600 steps — 200 cuts off ~half the successes):

```bash
cd /data/yingxi/RLinf_RoboFAPE
export TMPDIR=/data/yingxi/tmp HF_HOME=/data/yingxi/.cache/huggingface \
       RAY_TMP_DIR=/data/yingxi/ray_tmp_eval_wrist \
       RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs
VENV_DIR=/data/yingxi/kairan/envs/rlinf \
CHECKPOINT_PATH=<...>/global_step_<N>/actor \
GPU_IDS=0,1,2,3 \
NUM_EVAL_EPISODES=8 NUM_ENVS=4 \
EVAL_ACTION_SCALE=1.0 SAVE_VIDEO=true \
MANAGE_RAY=true EVAL_RAY_PORT=6380 \
bash run_train/eval_checkpoint/run_peginsertion_wrist_insert_only.sh
```

### Sweep all checkpoints under a run

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/data/yingxi/kairan/envs/rlinf/bin/python run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
  --ray-port 6380 \
  --run-script run_train/eval_checkpoint/run_peginsertion_wrist_insert_only.sh \
  --checkpoint-dir <...>/peg_insertion_sft*/checkpoints \
  --output-dir <...>/wrist_insert_only_eval_sweep \
  --num-eval-episodes 10 --num-envs 1 \
  --gpu-ids 0,1,2,3 --action-scale 1.0
```

Writes `wrist_sweep_metrics.{csv,json}` + curve PNGs. `--resume` skips
checkpoints that already wrote `trajectory_metrics.json`;
`--continue-on-error` records a failed checkpoint and continues.

**Concurrency:** SFT (port 6379, GPUs 4-7) and eval (port 6380, GPUs 0-3) can
run at the same time — disjoint ports + disjoint GPUs. Two eval sweeps must not
share 6380; give the second `--ray-port 6390` and a distinct dashboard port.

## 3. Future PPO

The legacy base/1-image PPO, actual-EE-delta, and dual-wrist tracks were removed.
Any new RL must match the Current Setting above (single-wrist, `pi05_base`,
insert-only, action horizon 10, `execute_action_chunks` 10, target-delta
controller) so the SFT checkpoint's policy interface and norm_stats line up.

## Troubleshooting

- `failed to find device "cuda:0"`: run on a node with Vulkan render devices, or
  pick valid `--gpu-ids`.
- Eval guard rejects config: use the peg wrist SFT eval config; fix
  `config_name` / `num_images_in_input` / `num_action_chunks` / `action_horizon`.
- `env_success_rate=0` in strict replay: do not train — the dataset is not
  replayable at `action_scale=1.0`.
- `Errno 28` / disk full: move Ray tmp, `HF_HOME`, `TMPDIR` onto `/data`.
- insert-only eval crash `No module named 'solutions'` / `PegInsertionLiftPlanner worker exited unexpectedly`: set `RLINF_ROBOFPE_PATH` (Setup). The eval launcher runs Ray with `--include-dashboard=false`, so a lift-planner-worker death cascades into a dashboard-API error — fixing the solver path resolves it.
- Do not train new checkpoints from the old FK-converted `peg_insertion_vertical_3200`
  dataset (stale gripper/rotation semantics).
