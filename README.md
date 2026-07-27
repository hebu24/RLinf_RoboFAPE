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
CHECKPOINT_PATH=/data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor \
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

## 3. Robometer RL (async PPO)

`run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh` is the
canonical RL launcher for this project. It trains from the wrist insert-only SFT
checkpoint with chunk-level PPO, while Robometer reward is reconstructed on the
low-level env-step axis and then aggregated back to chunk rewards.

### Reward semantics

- The env worker stores `render_images` at every low-level env step, not just at
  chunk boundaries.
- For each finished rollout, the reward worker builds one `pick up + insertion`
  video, uniformly downsamples it to at most 60 frames, and sends one request to
  the Robometer server.
- Robometer returns per-frame `progress` in `[0, 1]`.
- Pick-up frames are context only. They help Robometer judge insertion progress
  but never enter RL training data.
- For insertion frames, progress is mapped back onto low-level env steps with
  linear interpolation. Before the first labeled insertion frame and after the
  last labeled insertion frame, the nearest labeled value is held constant.
- Low-level reward is applied per insertion step:
  - if that low-level step is successful: `reward = progress`
  - otherwise: `reward = progress - 1`
- PPO still trains on chunk rewards, not on low-level steps directly. Each
  10-step action chunk is reduced with discounted sum using `algorithm.gamma`.
- Success must come from env/eval-aligned signals only, with priority:
  `final_info.episode.success_once` -> `episode.success_once` -> root `success`.
  If all are missing, RL raises and stops. It must not fall back to Robometer
  `success_probs[-1]`.

### 3.1 Start the Robometer server

Use a dedicated GPU that is not part of the Ray RL cluster:

```bash
cd ~/RoboFAC/robometer
CUDA_VISIBLE_DEVICES=0 uv run python robometer/evals/eval_server.py \
  model_path=/data/yingxi/robometer/logs/checkpoint-400 \
  server_url=0.0.0.0 \
  server_port=8000 \
  num_gpus=1 \
  batch_size=4
```

Keep it in a persistent tmux session, for example `tmux new -s robometer_server`.
The RL launcher probes `http://127.0.0.1:8000/health` before training.

### 3.2 Launch RL training

This run must stay aligned with the wrist insert-only SFT/eval setting:
single wrist, `pi05_maniskill_peg_insertion_wrist`, insert-only reset,
`num_action_chunks=10`, `action_horizon=10`, and
`panda-ee-target-dpose` / `pd_ee_target_delta_pose`.

```bash
cd /data/yingxi/RLinf_RoboFAPE
export TMPDIR=/data/yingxi/tmp
export HF_HOME=/data/yingxi/.cache/huggingface
export RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RL_RAY_PORT=6381
export RAY_DASHBOARD_AGENT_PORT=52367
bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh
```

Useful Hydra overrides:

```bash
bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
  actor.model.model_path=<...>/global_step_<N>/actor \
  rollout.model.model_path=<...>/global_step_<N>/actor \
  runner.resume_dir=<...>/checkpoints/global_step_<M>
```

What the launcher sets for you:

- `EMBODIED_PATH=/data/yingxi/RLinf_RoboFAPE/examples/embodiment`
- `PYTHONPATH=/data/yingxi/RLinf_RoboFAPE:${PYTHONPATH}`
- `MUJOCO_GL=egl`
- `PYOPENGL_PLATFORM=egl`
- `RAY_TMPDIR=/data/yingxi/ray_tmp_rl_${RL_RAY_PORT}`

Logs and checkpoints:

- Training log: `logs/<timestamp>-peg_insertion_rl_async/run.log`
- TensorBoard + metrics: under the same `logs/<timestamp>-peg_insertion_rl_async/`
- Checkpoints: `logs/<timestamp>-peg_insertion_rl_async/checkpoints/global_step_<N>/`

### 3.3 Ray isolation and shared-host rules

- Never run a bare `ray stop`; it will kill other clusters on the same host.
- RL, SFT, and eval must use different Ray GCS ports, dashboard agent ports,
  temp dirs, and disjoint GPU sets.
- The current convention is:
  - SFT: port `6379`
  - wrist eval: port `6380`
  - RL: port `6381`
- Put `TMPDIR`, `HF_HOME`, and every `RAY_TMPDIR` on `/data`, not `/`.
- The Robometer server must run outside Ray on its own physical GPU.

See `RAY_ISOLATION.md` for the full host-level isolation rules.

### 3.4 Validate success semantics end to end

The complete validation has two halves:

1. the wrist insert-only eval script defines the ground-truth success label
   (`success_once`)
2. the RL reward path must resolve the same label and assign rewards with the
   new low-level reconstruction rules

Do not treat the eval command alone as a full validation. It only proves the
eval-side success label exists. The reward-side check must be run separately.

#### Step A. Save per-trajectory eval labels

Run the exact wrist insert-only eval path and persist per-trajectory metrics:

```bash
cd /data/yingxi/RLinf_RoboFAPE
export TMPDIR=/data/yingxi/tmp HF_HOME=/data/yingxi/.cache/huggingface \
       RAY_TMP_DIR=/data/yingxi/ray_tmp_eval_wrist \
       RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs
VENV_DIR=/data/yingxi/kairan/envs/rlinf \
CHECKPOINT_PATH=/data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor \
GPU_IDS=0,1,2,3 \
NUM_EVAL_EPISODES=8 NUM_ENVS=4 \
EVAL_ACTION_SCALE=1.0 SAVE_VIDEO=true \
MANAGE_RAY=true EVAL_RAY_PORT=6380 \
bash run_train/eval_checkpoint/run_peginsertion_wrist_insert_only.sh --save-episode-metrics
```

Check the output directory printed by the script. It should now contain:

- `evaluation_summary.json`
- `trajectory_metrics.json`
- `eval.log`

Inspect `trajectory_metrics.json` first. This is the eval-side reference label:

```bash
cat logs/<eval-run>/trajectory_metrics.json
```

The key field is `success_once`. Example interpretation:

- `success_once[i] == 1`: trajectory `i` is a success under the eval script
- `success_once[i] == 0`: trajectory `i` is a failure under the eval script

This step answers only one question: which trajectories are successful under
the wrist eval definition.

#### Step B. Save per-rollout videos and reward curves

If you want artifacts that are easy to inspect by eye, run the smoke rollout
visualizer first. It uses the same peg-insertion env, the same SFT checkpoint,
the same Robometer reward model, and the same low-level interpolation helper as
RL training, but writes one directory per rollout:

```bash
cd /data/yingxi/RLinf_RoboFAPE
export TMPDIR=/data/yingxi/tmp
export HF_HOME=/data/yingxi/.cache/huggingface
export RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs
PYTHONPATH=/data/yingxi/RLinf_RoboFAPE \
/data/yingxi/kairan/envs/rlinf/bin/python \
  run_train/peginsertion_maniskill_pi0.5/robometer_smoke_rollout.py \
  --ckpt /data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor \
  --gpu 1 \
  --num-traj 8 \
  --max-chunks 60 \
  --max-robometer-frames 60 \
  --out-dir /data/yingxi/RLinf_RoboFAPE/logs/robometer_smoke_gs40000
```

Each `traj_XXX/` directory now contains:

- `robometer_input.mp4`
  - the full pick-up + insertion video actually fed into Robometer
- `rollout_render_envstep.mp4`
  - the rollout video at every env step
- `downsampled_progress.png`
  - Robometer's <=60-frame progress curve
- `reward_curve.png`
  - interpolated insertion-step progress and low-level reward curves
- `downsampled_reward.npy`
- `downsampled_progress.npy`
- `low_level_reward.npy`
- `low_level_progress.npy`
- `meta.json`

This is the easiest way to inspect whether a single rollout's reward rises or
falls at the right moments.

#### Step C. Run one RL reward-side debug pass

Next, run the actual RL reward path with debug logging enabled. This uses the
same Robometer reward reconstruction code as training:

```bash
cd /data/yingxi/RLinf_RoboFAPE
rm -f /tmp/robometer_rdebug.log
export TMPDIR=/data/yingxi/tmp
export HF_HOME=/data/yingxi/.cache/huggingface
export RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RL_RAY_PORT=6381
export RAY_DASHBOARD_AGENT_PORT=52367
export RLINF_REWARD_DEBUG=1
bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
  actor.model.model_path=/data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor \
  rollout.model.model_path=/data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor \
  runner.max_epochs=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1
```

This one-epoch run is the training-path validation. It should generate
`/tmp/robometer_rdebug.log`.

Key debug fields in that file:

- reward-worker success resolution:
  - whether the trajectory was treated as success or failure under
    `final_info.episode.success_once -> episode.success_once -> success`
- `ds=...`
  - the downsample indices used for the Robometer request
- `per_step_reward[:10]=...`
  - the reconstructed low-level insertion reward after interpolation
- `loss_mask_tail=...`
  - which low-level insertion steps were treated as valid training reward
- `rewards_tail=...`
  - the low-level rewards actually scattered back into the rollout tensors

Useful quick inspection commands:

```bash
grep -n "\\[assign\\]" /tmp/robometer_rdebug.log
grep -n "per_step_reward" /tmp/robometer_rdebug.log
grep -n "loss_mask_tail" /tmp/robometer_rdebug.log
```

#### Step D. Compare eval labels against RL reward labels

The validation passes only if all of the following are true:

1. Every trajectory marked successful by eval `success_once` is also treated as
   success by the RL reward path.
2. For those successful trajectories, the reconstructed low-level insertion
   reward contains values `> 0`.
3. For low-level insertion steps that are not successful, including steps inside
   otherwise successful trajectories before insertion actually succeeds, the RL
   reward path applies `reward = progress - 1`.
4. The reward is not accidentally multiplied by substep broadcast. The RL debug
   log should show one reconstructed low-level reward per insertion env step,
   not one identical chunk reward copied across all 10 substeps twice.

Concretely, review:

- eval side:
  - `trajectory_metrics.json`
  - `evaluation_summary.json`
- rollout visualization side:
  - `logs/robometer_smoke_gs40000/traj_XXX/reward_curve.png`
  - `logs/robometer_smoke_gs40000/traj_XXX/low_level_reward.npy`
  - `logs/robometer_smoke_gs40000/traj_XXX/meta.json`
- RL reward side:
  - `/tmp/robometer_rdebug.log`
  - `logs/<rl-run>/run.log`

Expected outcomes:

- If eval reports `success_once = 0.25` over 8 trajectories, then 2 of those
  trajectories should also be treated as successful by the RL reward path.
- Successful RL trajectories should show positive values in
  `per_step_reward[:10]` or later `rewards_tail`.
- Non-success insertion steps should show values shifted by `-1` relative to
  raw progress, even inside an eventually successful trajectory.
- Missing env-side success fields should raise an explicit error and stop the
  RL run. This is the intended behavior.

If Step A succeeds but Step B does not produce matching labels, the success
semantics are not yet aligned. If Step B produces aligned labels but no
positive reward on successful trajectories, the interpolation or reward shift
logic is wrong.

## Troubleshooting

- `failed to find device "cuda:0"`: run on a node with Vulkan render devices, or
  pick valid `--gpu-ids`.
- Eval guard rejects config: use the peg wrist SFT eval config; fix
  `config_name` / `num_images_in_input` / `num_action_chunks` / `action_horizon`.
- `env_success_rate=0` in strict replay: do not train — the dataset is not
  replayable at `action_scale=1.0`.
- `Errno 28` / disk full: move Ray tmp, `HF_HOME`, `TMPDIR` onto `/data`.
- insert-only eval crash `No module named 'solutions'` / `PegInsertionLiftPlanner worker exited unexpectedly`: set `RLINF_ROBOFPE_PATH` (Setup). The eval launcher runs Ray with `--include-dashboard=false`, so a lift-planner-worker death cascades into a dashboard-API error — fixing the solver path resolves it.
- RL crashes with a reward-side `ValueError` about missing success fields: this is
  expected under the new strict contract. The peg-insertion RL path now requires
  one of `final_info.episode.success_once`, `episode.success_once`, or root
  `success`; it will not use Robometer `success_probs[-1]`.
- RL reward looks too large by about `x10`: this should no longer happen. The
  new path reconstructs low-level rewards, then aggregates each chunk once with
  discounted sum. If you still see scale inflation, inspect
  `/tmp/robometer_rdebug.log` and confirm the same insertion steps are not being
  broadcast across every substep twice.
- Successful eval video but non-positive RL reward: compare eval `success_once`
  with the RL debug log and confirm the same trajectory received a positive
  insertion progress curve after interpolation. A mismatch usually means the
  env-side success fields were missing or the wrong checkpoint/config pair was
  used.
- Do not train new checkpoints from the old FK-converted `peg_insertion_vertical_3200`
  dataset (stale gripper/rotation semantics).
