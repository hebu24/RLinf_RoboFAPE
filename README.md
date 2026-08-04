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
mkdir -p /data/yingxi/tmp
TMPDIR=/data/yingxi/tmp CUDA_VISIBLE_DEVICES=4 uv run python robometer/evals/eval_server.py \
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
export CUDA_VISIBLE_DEVICES=0,1
export RL_RAY_PORT=6384
export RAY_DASHBOARD_PORT=8264
export RAY_DASHBOARD_AGENT_PORT=52370
export RAY_MIN_WORKER_PORT=10002
export RAY_MAX_WORKER_PORT=10399
export CONFIG_NAME=maniskill_async_ppo_peg_insertion_pi05   # absolute shaping (default)
# env.train.rollout_window_mode=independent: each rollout window is self-contained —
# unfinished episodes are force-closed as a synthetic truncation at the window
# boundary, Robometer settles them, then all train envs reset + history clears so
# the next window starts fresh (no cross-window last_obs resume). Default is
# `continuous` (legacy cross-window continuation). See
# ASYNC_INDEPENDENT_ROLLOUT_WINDOW_IMPLEMENTATION.md.
bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
  env.train.rollout_window_mode=independent \
  reward.model.timeout_s=600 \
  algorithm.rollout_store_wait_timeout_s=1800
```

For `reward.shaping=delta` instead, export
`CONFIG_NAME=maniskill_async_ppo_peg_insertion_pi05_delta` and add the Hydra
overrides `reward.shaping=delta reward.model.server_url=http://127.0.0.1:8001`
(delta uses a second Robometer server on `:8001`; see §3.5). The launcher
auto-selects the `_delta` config when `reward.shaping=delta` is passed, but
setting `CONFIG_NAME` explicitly is clearer for concurrent runs.

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
- `RAY_ADDRESS=127.0.0.1:${RL_RAY_PORT}` (pins driver+workers to this cluster)
- `RAY_TMPDIR=/data/yingxi/ray_tmp_rl_${RL_RAY_PORT}`
- Its own `ray start --head` on `RL_RAY_PORT` with a scoped EXIT trap
  (`_rl_scoped_ray_kill` by port) — it no longer relies on `cluster.py`'s
  `ray.init(address="auto")` (which ps-scans to any GCS) and never does a bare
  `ray stop` (which would kill other clusters on the host).

Logs and checkpoints (the launcher appends the reward shaping tag — `_<absolute|delta>` — to the log dir, parsed from the `reward.shaping=` Hydra override):

- Training log: `logs/<timestamp>-peg_insertion_rl_async_<shaping>/run.log`
- TensorBoard + metrics: under the same `logs/<timestamp>-peg_insertion_rl_async_<shaping>/`
- Checkpoints: `logs/<timestamp>-peg_insertion_rl_async_<shaping>/<experiment_name>/checkpoints/global_step_<N>/`

For the peg-insertion Robometer async PPO configs, reward reconstruction still
keeps the full pick-up+insert history across rollout windows, but actor
training consumes the current rollout window directly (`reward.history_train_mode:
rollout_window`). Complete episodes are no longer required as the training unit.

### 3.3 Ray isolation and shared-host rules

- Never run a bare `ray stop`; it will kill other clusters on the same host.
- RL, SFT, and eval must use different Ray GCS ports, dashboard agent ports,
  temp dirs, and disjoint GPU sets.
- The current convention is:
  - SFT: port `6379`, dashboard `52366`
  - wrist eval: port `6380`, dashboard `52365`
  - RL (absolute): port `6381`, dashboard `52367`
  - RL (delta): port `6382`, dashboard `52368`
- The RL launcher now starts its own `ray start --head` with `RAY_ADDRESS` set
  and a scoped EXIT trap (`_rl_scoped_ray_kill` by port). It no longer relies on
  a pre-existing head or on `cluster.py`'s `ray.init(address="auto")`.
- Put `TMPDIR`, `HF_HOME`, and every `RAY_TMPDIR` on `/data`, not `/`.
- The Robometer server runs outside Ray (an HTTP server on `:8000`). It may
  overlap one RL cluster's GPU set (HBM contention accepted — monitor with
  `nvidia-smi`); both RL runs point `server_url` to `http://127.0.0.1:8000`
  (the config default). See §3.5 for the shared-server dual-run setup.

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
  --gpu 7 \
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
export CUDA_VISIBLE_DEVICES=0,1
export RL_RAY_PORT=6381
export RAY_DASHBOARD_AGENT_PORT=52367
export RLINF_REWARD_DEBUG=1
# Optional: write the reward debug log to a per-run path (defaults to the shared
# /tmp/robometer_rdebug.log when unset).
export RLINF_REWARD_DEBUG_LOG=/tmp/robometer_rdebug.log
bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
  actor.model.model_path=/data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor \
  rollout.model.model_path=/data/yingxi/RLinf_RoboFAPE/logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor \
  runner.max_epochs=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1
```

This one-epoch run is the training-path validation (the log dir is
`logs/<timestamp>-peg_insertion_rl_async_absolute/` since no `reward.shaping=`
override is passed). It should generate `/tmp/robometer_rdebug.log`.

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
  - `/tmp/robometer_rdebug.log` (or the `RLINF_REWARD_DEBUG_LOG` path)
  - `logs/<timestamp>-peg_insertion_rl_async_<shaping>/run.log`

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

### 3.5 Dual concurrent RL runs (absolute vs delta)

Two async-PPO runs with different reward shaping train concurrently on the same
host, both in **independent rollout-window mode** (`env.train.rollout_window_mode=independent`):
Run A with `reward.shaping=absolute` on GPUs 0-1 (GCS port `6384`), Run B with
`reward.shaping=delta` on GPUs 2-3 (GCS port `6386`). Each run HTTP-calls its own
Robometer server (`:8000` for absolute, `:8001` for delta) and uses disjoint GPU
sets + fully distinct Ray GCS/dashboard/agent/worker ports + temp-dirs, so each
run's scoped teardown touches only its own head.

| run | shaping | GPUs | GCS port | dashboard | agent | worker ports | robometer | CONFIG_NAME |
|---|---|---|---|---|---|---|---|---|
| A | `absolute` | 0,1 | `6384` | `8264` | `52370` | `10002-10399` | `:8000` (GPU 0) | `maniskill_async_ppo_peg_insertion_pi05` |
| B | `delta` | 2,3 | `6386` | `8266` | `52372` | `13000-13399` | `:8001` (GPU 2) | `maniskill_async_ppo_peg_insertion_pi05_delta` |

Launch order: Robometer `:8000` → Robometer `:8001` → wait for both `/health` →
Run A → Run B (A/B order does not matter — ports are distinct). Each
`tmux new-session -d` survives SSH disconnect; attach with
`tmux attach -t rl_abs_independent_window_gpu01` / `rl_delta_independent_window_gpu23`
(detach: `Ctrl-b d`). The launcher auto-selects the `_delta` config when
`reward.shaping=delta` is passed, but `CONFIG_NAME` is set explicitly here so the
pane-root cmdline is unambiguous. Both runs pass
`env.train.rollout_window_mode=independent` so every window force-closes
unfinished episodes, settles them via Robometer, then resets all train envs +
clears history before the next window (see
`ASYNC_INDEPENDENT_ROLLOUT_WINDOW_IMPLEMENTATION.md`).

```bash
# Step 0 — two Robometer servers (one per run; each co-located on the run's
# first GPU, with the eval_server empty_cache mitigation for co-location).
tmux new-session -d -s robometer_server \
  "cd /home/yingxi/RoboFAC/robometer && \
   mkdir -p /data/yingxi/tmp && \
   TMPDIR=/data/yingxi/tmp CUDA_VISIBLE_DEVICES=0 uv run python robometer/evals/eval_server.py \
     model_path=/data/yingxi/robometer/logs/checkpoint-400 \
     server_url=0.0.0.0 server_port=8000 num_gpus=1 batch_size=4 \
   2>&1 | tee /data/yingxi/RLinf_RoboFAPE/logs/robometer_server_8000.log"
tmux new-session -d -s robometer_server_8001_gpu2 \
  "cd /home/yingxi/RoboFAC/robometer && \
   mkdir -p /data/yingxi/tmp && \
   TMPDIR=/data/yingxi/tmp CUDA_VISIBLE_DEVICES=2 uv run python robometer/evals/eval_server.py \
     model_path=/data/yingxi/robometer/logs/checkpoint-400 \
     server_url=0.0.0.0 server_port=8001 num_gpus=1 batch_size=4 \
   2>&1 | tee /data/yingxi/RLinf_RoboFAPE/logs/robometer_server_8001.log"
until curl -sS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1; do sleep 2; done && echo "Robometer :8000 healthy"
until curl -sS --max-time 5 http://127.0.0.1:8001/health >/dev/null 2>&1; do sleep 2; done && echo "Robometer :8001 healthy"

# Step 1 — Run A: absolute reward + independent windows, GPUs 0-1, port 6384
tmux new-session -d -s rl_abs_independent_window_gpu01 -c /data/yingxi/RLinf_RoboFAPE \
  "cd /data/yingxi/RLinf_RoboFAPE && \
   CUDA_VISIBLE_DEVICES=0,1 \
   RL_RAY_PORT=6384 RAY_DASHBOARD_PORT=8264 RAY_DASHBOARD_AGENT_PORT=52370 \
   RAY_MIN_WORKER_PORT=10002 RAY_MAX_WORKER_PORT=10399 \
   CONFIG_NAME=maniskill_async_ppo_peg_insertion_pi05 \
   LOG_DIR=logs/\$(date +%Y%m%d-%H:%M:%S)-peg_insertion_rl_async_absolute_independent_window_gpu01 \
   bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
     env.train.rollout_window_mode=independent \
     reward.model.timeout_s=600 algorithm.rollout_store_wait_timeout_s=1800 \
   ; echo ===EXIT=\$?=== ; exec bash"

# Step 2 — Run B: delta reward + independent windows, GPUs 2-3, port 6386
tmux new-session -d -s rl_delta_independent_window_gpu23 -c /data/yingxi/RLinf_RoboFAPE \
  "cd /data/yingxi/RLinf_RoboFAPE && \
   CUDA_VISIBLE_DEVICES=2,3 \
   RL_RAY_PORT=6386 RAY_DASHBOARD_PORT=8266 RAY_DASHBOARD_AGENT_PORT=52372 \
   RAY_MIN_WORKER_PORT=13000 RAY_MAX_WORKER_PORT=13399 \
   CONFIG_NAME=maniskill_async_ppo_peg_insertion_pi05_delta \
   LOG_DIR=logs/\$(date +%Y%m%d-%H:%M:%S)-peg_insertion_rl_async_delta_independent_window_gpu23 \
   bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
     reward.shaping=delta reward.model.server_url=http://127.0.0.1:8001 \
     env.train.rollout_window_mode=independent \
     reward.model.timeout_s=600 algorithm.rollout_store_wait_timeout_s=1800 \
   ; echo ===EXIT=\$?=== ; exec bash"
```

The `LOG_DIR` names carry `independent_window` so these runs are
distinguishable from legacy continuous-window runs at a glance. Each run writes
its Robometer reward debug log to `/tmp/robometer_rdebug.log` (or
`$RLINF_REWARD_DEBUG_LOG` if set with `RLINF_REWARD_DEBUG=1`). Confirm both
clusters coexist:

```bash
pgrep -fa gcs_server | grep -oE 'gcs_server_port=[0-9]+' | sort -u   # both 6384 and 6386
ss -tlnp | grep -E '6384|6386|52370|52372|8000|8001'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader         # A uses 0-1, B uses 2-3
```

Scoped teardown of one run leaves the other alive — target that run's GCS port
(`6384` for Run A / absolute, `6386` for Run B / delta; see `RAY_ISOLATION.md`).
Use a **character class** (`638[4]` / `638[6]`) in the pattern: a bare `6384`
makes `pkill -f` match the shell running the command itself (its cmdline contains
the pattern string), killing your shell before the `|| true` runs:

```bash
# Run A (absolute, GCS 6384) — swap to 638[6] for Run B (delta)
pkill -9 -f 'ray_tmp_rl_638[4]'                 || true
pkill -9 -f 'gcs_server_port=638[4]'            || true
pkill -9 -f 'raylet.*gcs-address=[^ ]*:638[4]'   || true
pkill -9 -f 'dashboard.*gcs-address=[^ ]*:638[4]' || true
sleep 2
```

**Known risks / caveats:**

- **HBM contention on the Robometer GPU** (GPU 0 here): the Robometer VLM server
  (~20-40 GB) shares the GPU with Run A's FSDP workers. Monitor with
  `watch nvidia-smi`; if it OOMs, move the Robometer to a GPU outside both
  clusters (requires re-splitting) or lower Robometer `batch_size`.
- **`staleness_filter_mode` desync (affects both runs):** the config defaults to
  `trajectory` (stale trajectories are silently dropped → data-pipeline desync →
  NCCL collective timeout). The `chunk_mask` mode is implemented but not yet
  enabled. To enable for one run, add `algorithm.staleness_filter_mode=chunk_mask`;
  validate it separately before flipping the default.
- **`RLINF_REWARD_DEBUG` logs:** set `RLINF_REWARD_DEBUG=1` plus a per-run
  `RLINF_REWARD_DEBUG_LOG` (the launch commands above use `_absolute.log` /
  `_delta.log`) so the two runs' reward-debug output lands in separate files
  instead of interleaving. Without `RLINF_REWARD_DEBUG_LOG`, both fall back to the
  shared `/tmp/robometer_rdebug.log`.
- **Disk:** two runs each save ~16-32 GB checkpoints every `save_interval` steps;
  verify `df -h /data` has ≥ ~500 GB free before launching.


### 3.6 Evaluate an RL checkpoint

The wrist insert-only eval launcher evaluates an RL actor checkpoint exactly
the way it evaluates an SFT one: same config
(`maniskill_async_ppo_peg_insertion_pi05`), same observations (base + single
wrist, `wrap_obs_mode=simple`, `num_images_in_input=2`), same 10-step action
chunks (`execute_action_chunks=10`, `action_horizon=10`), same 600-step
insert-only rollout, same controller (`pd_ee_target_delta_pose`,
`action_scale=1.0`), and the same base+wrist concatenated video. The only
differences from the SFT eval (Section 2) are the checkpoint path, the GPU, and
the Ray port. The eval runs `EmbodiedEvalRunner` (no reward worker), so it does
not touch the Robometer server.

RL checkpoints land in
`logs/<timestamp>-peg_insertion_rl_async_<shaping>/<experiment_name>/checkpoints/global_step_<N>_trainenvstep_<M>/actor`
and contain `model_state_dict/full_weights.pt` + `dcp_checkpoint/` (model
weights load fine for inference; only the RL optimizer state is incomplete, which
does not matter for eval). The RL `save_checkpoint` does NOT write
`norm_stats.json`, so before the first eval of a fresh RL checkpoint, copy the
norm-stats dir from the SFT checkpoint it was initialized from (they are
identical fixed input constants, not learned during RL):

```bash
SFT=logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor
RL=logs/20260730-23:30:28-peg_insertion_rl_async_delta_resume50_postppo_shape/peg_insertion_async_ppo_pi05_robometer/checkpoints/global_step_140_trainenvstep_132205/actor
cp -r "$SFT/physical-intelligence" "$RL/"
```

Then run the eval on one relatively-idle GPU from 0-3 (so the co-located
training run never OOMs) and an isolated Ray port (the training/eval port
ranges are in `RAY_ISOLATION.md`):

```bash
cd /data/yingxi/RLinf_RoboFAPE
export TMPDIR=/data/yingxi/tmp HF_HOME=/data/yingxi/.cache/huggingface \
       RAY_TMP_DIR=/data/yingxi/ray_tmp_eval_6500 \
       RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs
VENV_DIR=/data/yingxi/kairan/envs/rlinf \
CHECKPOINT_PATH=logs/20260730-23:30:28-peg_insertion_rl_async_delta_resume50_postppo_shape/peg_insertion_async_ppo_pi05_robometer/checkpoints/global_step_140_trainenvstep_132205/actor \
GPU_IDS=2 \
NUM_EVAL_EPISODES=8 NUM_ENVS=2 \
EVAL_ACTION_SCALE=1.0 SAVE_VIDEO=true \
MANAGE_RAY=true EVAL_RAY_PORT=6500 \
LOG_DIR=logs/20260730-23:30:28-peg_insertion_rl_async_delta_resume50_postppo_shape/peg_insertion_async_ppo_pi05_robometer/checkpoints/global_step_140_trainenvstep_132205/eval \
bash run_train/eval_checkpoint/run_peginsertion_wrist_insert_only.sh --save-episode-metrics
```

Run it in a persistent tmux so it survives SSH disconnect:

```bash
tmux new-session -d -s rl_eval "cd /data/yingxi/RLinf_RoboFAPE && \
  TMPDIR=/data/yingxi/tmp HF_HOME=/data/yingxi/.cache/huggingface \
  RAY_TMP_DIR=/data/yingxi/ray_tmp_eval_6500 RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs \
  VENV_DIR=/data/yingxi/kairan/envs/rlinf CHECKPOINT_PATH=<...>/actor \
  GPU_IDS=0 NUM_EVAL_EPISODES=8 NUM_ENVS=2 EVAL_ACTION_SCALE=1.0 SAVE_VIDEO=true \
  MANAGE_RAY=true EVAL_RAY_PORT=6500 LOG_DIR=logs/<eval-run> \
  bash run_train/eval_checkpoint/run_peginsertion_wrist_insert_only.sh --save-episode-metrics \
  ; echo ===EXIT=\$?=== ; exec bash"
```

Outputs (under `LOG_DIR`):

- `evaluation_summary.json` -- aggregate metrics (`success_once`, `reward`,
  `max_reward`, `return`, `episode_len`, `num_trajectories`).
- `trajectory_metrics.json` -- per-episode `success_once` (needs
  `--save-episode-metrics`).
- `eval.log`.
- `video/eval/seed_0/<epoch>.mp4` -- base+wrist concatenated video, same format
  as the SFT eval (Section 2). With `NUM_ENVS=2` each frame tiles 2 envs x
  (base+wrist) = 896x224; use `NUM_ENVS=1` for single-env 448x224 videos.

Verified on the delta-smoke `global_step_50` checkpoint (delta reward,
`staleness_filter_mode=chunk_mask`): `success_once=0.125` (1/8), `max_reward`
`0.885`, `reward=0.699`, `return=419`, `episode_len=600`, `num_trajectories=8`,
exit 0, both eval processes pinned to the chosen GPU (no spill to the training
GPUs).


### 3.7 Sweep all RL checkpoints under a run

The same `sweep_peginsertion_wrist.py` driver (§2) sweeps **every** RL
checkpoint under a run's `checkpoints/` dir. It matches both SFT
(`global_step_<N>`) and RL (`global_step_<N>_trainenvstep_<M>`) directory
names, so the only RL-specific steps are:

1. point `--norm-stats-source` at the SFT checkpoint the policy was initialized
   from — RL checkpoints do not save `norm_stats.json` (the assets are fixed
   input constants, identical across the run), and the sweep copies the
   `physical-intelligence/` dir in idempotently before each eval (no-op if
   already present);
2. pick GPUs and a Ray port disjoint from the still-training RL cluster and the
   Robometer server.

The output is identical to the SFT sweep: `wrist_sweep_metrics.{csv,json}` +
`success_rate_vs_step.png` / `max_reward_vs_step.png` / `wrist_sweep_curves.png`,
plus a per-checkpoint subdir (`global_step_<N>_trainenvstep_<M>/`) holding
`evaluation_summary.json`, `trajectory_metrics.json`, `eval.log`, and
`video/eval/...`. The CSV/JSON rows carry an extra `trainenvstep` and
`checkpoint_name` column for RL checkpoints; the plot x-axis is the PPO step
(group 1 of the checkpoint name).

```bash
cd /data/yingxi/RLinf_RoboFAPE
export TMPDIR=/data/yingxi/tmp HF_HOME=/data/yingxi/.cache/huggingface \
       RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs MPLCONFIGDIR=/tmp/matplotlib

# SFT checkpoint the RL policy was initialized from — its physical-intelligence/
# assets hold norm_stats.json (fixed input constants; not learned during RL, so
# the same source serves every RL checkpoint in the run).
SFT_BASE=logs/20260719-16:44:47-peg_insertion_sft_openpi_pi05_wrist-3200/checkpoints/global_step_40000/actor

/data/yingxi/kairan/envs/rlinf/bin/python run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
  --ray-port 6501 --ray-dashboard-port 8267 \
  --checkpoint-dir logs/20260731-11:18:41-peg_insertion_rl_async_absolute_16ep_single_step/peg_insertion_async_ppo_pi05_robometer/checkpoints \
  --norm-stats-source "$SFT_BASE" \
  --output-dir logs/20260731-11:18:41-peg_insertion_rl_async_absolute_16ep_single_step/peg_insertion_async_ppo_pi05_robometer/rl_eval_sweep \
  --num-eval-episodes 24 --num-envs 8 \
  --gpu-ids 1,4,5 --action-scale 1.0 \
  --save-video --continue-on-error
```

Run it in a persistent tmux so it survives SSH disconnect (the sweep starts its
own scoped Ray head on `--ray-port` and tears it down on exit via a scoped
port-keyed kill — never a bare `ray stop`, so the training cluster and the
Robometer server are never touched):

```bash
tmux new-session -d -s rl_eval_sweep "cd /data/yingxi/RLinf_RoboFAPE && \
  TMPDIR=/data/yingxi/tmp HF_HOME=/data/yingxi/.cache/huggingface \
  RLINF_ROBOFPE_PATH=/home/yingxi/RoboFAC/mani_envs MPLCONFIGDIR=/tmp/matplotlib \
  /data/yingxi/kairan/envs/rlinf/bin/python run_train/eval_checkpoint/sweep_peginsertion_wrist.py \
    --ray-port 6501 --ray-dashboard-port 8267 \
    --checkpoint-dir <...>/<run>/peg_insertion_async_ppo_pi05_robometer/checkpoints \
    --norm-stats-source <...>/<sft_run>/checkpoints/global_step_<N>/actor \
    --output-dir <...>/<run>/peg_insertion_async_ppo_pi05_robometer/rl_eval_sweep \
    --num-eval-episodes 24 --num-envs 8 \
    --gpu-ids 1,4,5 --action-scale 1.0 \
    --save-video --continue-on-error \
  ; echo ===EXIT=\$?=== ; exec bash"
```

Notes:

- **GPUs:** pick GPUs disjoint from the still-training RL cluster (the absolute
  run trains on 2-3, so the example uses 1,4,5) and from the Robometer server's
  GPU (0). The eval pins each checkpoint's env+rollout actors to the listed
  GPUs via `cluster.component_placement` — no spill to the training or
  Robometer GPUs.
- **Ray ports:** `--ray-port 6501` is isolated from the RL cluster (6381/6383),
  the SFT cluster (6379), the wrist-eval port (6380), and the Robometer HTTP
  server (:8000). Two concurrent RL sweeps need distinct ports + dashboard ports
  (cf. §2 / `RAY_ISOLATION.md`).
- **Ray tmp on /data:** the sweep head's `--ray-tmp-dir` defaults to
  `/data/yingxi/ray_tmp_eval_sweep` (NOT `/tmp`) — xulab's root-backed `/tmp`
  fills to 100% and Ray's object store + logs would crash it with `Errno 28`.
- **`--resume`** skips checkpoints that already wrote `trajectory_metrics.json`
  (re-run the same command after more checkpoints appear to fill in the curve
  without re-evaluating finished steps); **`--continue-on-error`** records a
  failed checkpoint (e.g. one superseded/removed by the still-training run's
  checkpoint rotation mid-sweep) and continues.
- **Moving checkpoint set:** the still-training RL run saves a new checkpoint
  every `checkpoint_interval` steps and removes the previous non-permanent one
  (checkpoints at multiples of `checkpoint_permanent_interval` are kept). The
  sweep snapshots the set at discovery time; a checkpoint removed mid-eval is
  caught by `--continue-on-error`. Permanent checkpoints are stable.
- **Seed / episode semantics:** every checkpoint is evaluated with the same base
  seed (`--seed 0`, the default) so the comparison across checkpoints is
  controlled (identical eval conditions). The `--num-envs` parallel envs each get
  a distinct seed derived from the base seed (`env.seed = cfg.seed +
  seed_offset`, `seed_offset = rank * stage_num + stage_id`); ManiSkill
  re-seeds each env to its fixed seed on every reset, so `--num-eval-episodes N`
  spans `num_envs` seed-derived scenarios × `N/num_envs` stochastic-policy
  repeats (pi0.5 samples actions), not `N` distinct scenarios. The reported
  success rate is therefore a single-base-seed point estimate; re-run with
  different `--seed` values to get variance bars.
- **Covering the training seeds:** the example uses `--num-envs 8` (24 = 8 × 3,
  a multiple of 8) to mirror the training run's `total_num_envs=8`. Note the
  topology difference: with the default (one GPU per parallel checkpoint slot)
  the eval runs its 8 envs under **1 env worker** (rank 0, env seed 0), so its
  8 sub-env seeds are all derived from the rank-0 env seed — this reproduces
  training's rank-0 half (4 seeds) plus 4 additional seed-0-derived seeds, but
  **not** training's rank-1 half (training used 2 env workers, ranks 0+1, 4
  sub-envs each). For **exact** coverage of all 8 training seeds, pass
  `--gpus-per-ckpt` so every checkpoint uses the full `--gpu-ids` set (e.g.
  `--gpu-ids 4,5` → 2 env workers, ranks 0+1, env seeds 0+1, 4 sub-envs each =
  training's exact topology); checkpoints then run **sequentially** (one at a
  time across both cards). Use `--num-eval-episodes 48` (6 × 8) for 6 repeats of
  each of the 8 training seeds:
  ```bash
  ... --num-eval-episodes 48 --num-envs 8 --gpu-ids 4,5 --gpus-per-ckpt ...
  ```

Verified on the absolute run
`logs/20260731-11:18:41-peg_insertion_rl_async_absolute_16ep_single_step` (4
checkpoints: steps 50/100/150/180, 24 episodes each = 8 parallel eval envs × 3
epochs, on GPUs 1,4,5 — disjoint from the still-training absolute cluster on
2-3 and the Robometer server on 0): exit 0, `success_rate`
0.167/0.167/0.125/0.167, `max_reward` 1.0 on every step, `mean_max_reward`
0.882-0.888, eval actors pinned to the chosen GPUs with no spill, norm_stats
copied in idempotently from the SFT base.


## Troubleshooting

- `failed to find device "cuda:0"`: run on a node with Vulkan render devices, or
  pick valid `--gpu-ids`.
- Eval guard rejects config: use the peg wrist SFT eval config; fix
  `config_name` / `num_images_in_input` / `num_action_chunks` / `action_horizon`.
- `env_success_rate=0` in strict replay: do not train — the dataset is not
  replayable at `action_scale=1.0`.
- `Errno 28` / disk full: move Ray tmp, `HF_HOME`, `TMPDIR` onto `/data`.
  For Robometer server 500s during `/evaluate_batch_npy`, restart the server with
  `TMPDIR=/data/yingxi/tmp`; FastAPI may otherwise spool uploaded `.npy` payloads
  into root-backed `/tmp`.
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
