# UprightStack SFT checkpoint rollout/eval

This note records the rollout path used for the UprightStack SFT checkpoint sweep, using the
`global_step_40000` checkpoint as the concrete example.

## Concrete checkpoint

```bash
REPO=/data/yingxi/RLinf_RoboFAPE
RUN_DIR=${REPO}/logs/20260922_uprightstack_sft_3168/uprightstack_sft_wrist_3168
CKPT=${RUN_DIR}/checkpoints/global_step_40000/actor
OUT=${RUN_DIR}/uprightstack_sweep_seed0_49_eval1/global_step_40000/seed_0
```

The monitor pins checkpoints before eval, so the actual completed sweep used:

```bash
${RUN_DIR}/uprightstack_sweep_seed0_49_eval1/checkpoint_pins/global_step_40000/actor
```

The pinned actor directory is hardlinked/copied from the training checkpoint after the checkpoint
has all required actor entries:

```text
dcp_checkpoint/
model_state_dict/
trainer_state.json
```

## Single-seed rollout command

To reproduce one rollout for `global_step_40000`, run from the repo root:

```bash
cd /data/yingxi/RLinf_RoboFAPE

CHECKPOINT_PATH=/data/yingxi/RLinf_RoboFAPE/logs/20260922_uprightstack_sft_3168/uprightstack_sft_wrist_3168/checkpoints/global_step_40000/actor \
LOG_DIR=/data/yingxi/RLinf_RoboFAPE/logs/20260922_uprightstack_sft_3168/uprightstack_sft_wrist_3168/manual_eval_global_step_40000_seed0 \
GPU_IDS=0 \
SEED=0 \
SEEDS= \
NUM_EVAL_EPISODES=1 \
NUM_ENVS=1 \
MAX_EPISODE_STEPS=1000 \
SAVE_VIDEO=true \
MANAGE_RAY=true \
EVAL_RAY_PORT=6395 \
RAY_TMP_DIR=/data/yingxi/ray_tmp_eval_uprightstack_manual \
bash run_train/eval_checkpoint/run_uprightstack_wrist.sh --save-episode-metrics
```

Key points:

- `CHECKPOINT_PATH` must point to one `global_step_<N>/actor` directory, not the `checkpoints/` root.
- `NUM_EVAL_EPISODES=1` and `NUM_ENVS=1` mean exactly one rollout for this seed.
- `SEED=0` sets the single rollout seed.
- `SEEDS=` must be empty for the single worker path. If `SEEDS=0-49` leaks into this command, that worker can create nested or duplicated seed outputs.
- `MAX_EPISODE_STEPS=1000` matches the UprightStack config and the completed sweep.
- `SAVE_VIDEO=true` writes render-camera video under `${LOG_DIR}/video/eval/seed_0/0.mp4`.

## Eval stack

The command above dispatches through:

```text
run_train/eval_checkpoint/run_uprightstack_wrist.sh
  -> run_train/eval_checkpoint/run_pushcube_wrist.sh
    -> run_train/eval_checkpoint/eval_checkpoint.py
      -> run_train/pushcube_maniskill_pi0.5/config/maniskill_uprightstack_wrist_sft_eval_openpi_pi05.yaml
```

`run_uprightstack_wrist.sh` sets the UprightStack-specific defaults:

```text
CONFIG_NAME=maniskill_uprightstack_wrist_sft_eval_openpi_pi05
TASK_ID=UprightStack-v1
TASK_DESCRIPTION="Stand the brick upright and stack it on the red cube."
MAX_EPISODE_STEPS=1000
CONTROL_MODE=pd_joint_pos
PYTHONPATH=/data/yingxi/RoboFPE:...
MS_ASSET_DIR=/data/yingxi/robofac
```

The Hydra eval config uses:

```text
task id:             UprightStack-v1
robot_uids:          panda_wristcam
control_mode:        pd_joint_pos
obs_mode:            rgb
sim_backend:         gpu
reward_mode:         normalized_dense
render_mode:         all
main render camera:  render_camera
wrist image:         enabled
base_camera:         224 x 224
hand_camera:         224 x 224
render_camera video: 512 x 512
action chunks:       10
temporal ensemble:   0.01
```

## 50 seeds x 1 rollout sweep

The completed checkpoint sweep was launched through the watcher:

```bash
cd /data/yingxi/RLinf_RoboFAPE

CHECKPOINT_DIR=/data/yingxi/RLinf_RoboFAPE/logs/20260922_uprightstack_sft_3168/uprightstack_sft_wrist_3168/checkpoints \
OUT_DIR=/data/yingxi/RLinf_RoboFAPE/logs/20260922_uprightstack_sft_3168/uprightstack_sft_wrist_3168/uprightstack_sweep_seed0_49_eval1 \
GPU_IDS=0,1 \
SEEDS=0-49 \
NUM_EVAL_EPISODES=1 \
NUM_ENVS=1 \
MAX_EPISODE_STEPS=1000 \
SAVE_VIDEO=true \
VIDEO_SEEDS=0,10,20,30,40 \
RAY_PORT=6395 \
RAY_INCLUDE_DASHBOARD=false \
RAY_TEMP_DIR=/data/yingxi/ray_tmp_uprightstack_sweep_50x1 \
bash run_train/eval_checkpoint/watch_uprightstack_sft_eval.sh
```

The watcher invokes `sweep_pushcube_wrist.py` with:

```text
--run-script run_train/eval_checkpoint/run_uprightstack_wrist.sh
--watch
--resume
--continue-on-error
--pin-checkpoints
--seeds 0-49
--num-eval-episodes 1
--num-envs 1
--max-episode-steps 1000
--save-video
--video-seeds 0,10,20,30,40
```

For each checkpoint and seed, the sweep runs one child eval with environment variables equivalent to:

```text
CHECKPOINT_PATH=<pinned global_step_N/actor>
LOG_DIR=<OUT_DIR>/global_step_N/seed_S
GPU_IDS=<assigned gpu>
SEED=S
SEEDS=
NUM_EVAL_EPISODES=1
NUM_ENVS=1
MAX_EPISODE_STEPS=1000
SAVE_VIDEO=true only for seeds 0,10,20,30,40; otherwise false
MANAGE_RAY=false
RAY_ADDRESS=127.0.0.1:6395
```

The output layout is:

```text
<OUT_DIR>/
  checkpoint_pins/global_step_40000/actor/
  global_step_40000/
    seed_0/
      trajectory_metrics.json
      sweep_call.log
      eval.log
      video/eval/seed_0/0.mp4
    seed_1/
      trajectory_metrics.json
      sweep_call.log
      eval.log
    ...
  pushcube_sweep_metrics.csv
  pushcube_sweep_metrics.json
  sr_vs_step_multiseed.png
  index.html
  monitor.log
```

## Completed `global_step_40000` result

For the finished sweep:

```text
checkpoint: global_step_40000
seeds:      0-49
rollouts:   1 per seed
videos:     seeds 0,10,20,30,40
```

Each standard metrics file has the form:

```json
{
  "success_once": [false],
  "success_at_end": [false],
  "return": [0.0],
  "reward": [0.0],
  "max_reward": [0.0],
  "episode_len": [1000],
  "num_trajectories": 1
}
```

The sweep summary showed `success_rate=0.0` for all 50 seeds at `global_step_40000`.

## Reward caveat

`return`, `reward`, and `max_reward` are expected to be zero for this task implementation. The
UprightStack environment in RoboFPE currently defines dense reward as all zeros:

```text
/data/yingxi/RoboFPE/mani_envs/tasks/task_UprightStack.py
compute_dense_reward(...) -> torch.zeros(self.num_envs, device=self.device)
```

Therefore reward fields are not evidence that rollout failed. For this task, use `success_once`,
`success_at_end`, and saved render-camera videos to judge behavior unless the task reward function
is implemented.
