# PegInsertionVertical 训练与 Checkpoint 测试

本文说明如何在 RLinf 中使用 `PegInsertionVertical-v1` 进行 pi0.5 PPO 训练，并用保存的 actor checkpoint 做独立评测。

## 相关文件

| 用途 | 路径 |
|---|---|
| ManiSkill task | `rlinf/envs/maniskill/tasks/peg_insertion_vertical.py` |
| 训练入口 | `run_train/peginsertion_maniskill_pi0.5/run.sh` |
| 训练配置 | `run_train/peginsertion_maniskill_pi0.5/config/maniskill_peg_insertion_vertical_ppo_openpi_pi05.yaml` |
| 环境配置 | `run_train/peginsertion_maniskill_pi0.5/config/env/maniskill_peg_insertion_vertical.yaml` |
| checkpoint 评测入口 | `run_train/eval_checkpoint/run_peginsertion.sh` |
| checkpoint 评测实现 | `run_train/eval_checkpoint/eval_checkpoint.py` |

## 前置条件

1. 使用带 `openpi`、ManiSkill 和 Ray 的环境，默认脚本读取：

```bash
VENV_DIR=/opt/kairan/envs/rlinf
```

2. 准备初始模型或 actor checkpoint。训练配置默认使用：

```bash
/opt/yingxi/rlinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise/checkpoints/global_step_150/actor
```

3. 确认 GPU 可用。脚本会 `unset CUDA_VISIBLE_DEVICES`，Ray 会发现所有 GPU，再通过 `GPU_ID` 或 `GPU_IDS` 做 RLinf placement。

## 任务配置要点

`PegInsertionVertical-v1` 当前使用以下关键设置：

| 项 | 值 |
|---|---|
| `env_type` | `maniskill` |
| `obs_mode` | `rgb` |
| `robot_uids` | `panda_wristcam` |
| `sim_backend` | `gpu` |
| `reward_mode` | `normalized_dense` |
| `max_episode_steps` | `600` |
| prompt | `insert the blue peg vertically into the orange hole` |
| pi0.5 controller | `actor.model.policy_setup=panda-ee-dpose` |
| OpenPI config | `actor.model.openpi.config_name=pi05_maniskill` |

训练默认开启随机 reset：

```yaml
env:
  train:
    reset_options:
      randomize_initial_poses: True
  eval:
    reset_options:
      randomize_initial_poses: False
```

## 启动训练

最小命令：

```bash
cd /home/hebu/code/robofape/RLinf_RoboFAPE

VENV_DIR=/opt/kairan/envs/rlinf \
MODEL_PATH=/path/to/initial_or_actor_checkpoint \
GPU_ID=0 \
bash run_train/peginsertion_maniskill_pi0.5/run.sh
```

含义：

| 变量 | 说明 |
|---|---|
| `VENV_DIR` | Python/Ray/OpenPI 所在虚拟环境。 |
| `MODEL_PATH` | 同时覆盖 `actor.model.model_path` 和 `rollout.model.model_path`。 |
| `GPU_ID` | 训练使用的物理 GPU id。 |

常用小规模 smoke：

```bash
VENV_DIR=/opt/kairan/envs/rlinf \
MODEL_PATH=/path/to/initial_or_actor_checkpoint \
GPU_ID=0 \
bash run_train/peginsertion_maniskill_pi0.5/run.sh \
  runner.max_epochs=1 \
  runner.save_interval=1 \
  runner.val_check_interval=1 \
  env.train.total_num_envs=2 \
  env.eval.total_num_envs=2 \
  actor.micro_batch_size=2 \
  actor.global_batch_size=8
```

常用正式训练覆盖项：

```bash
bash run_train/peginsertion_maniskill_pi0.5/run.sh \
  runner.max_epochs=1000 \
  runner.save_interval=50 \
  runner.val_check_interval=10 \
  env.train.total_num_envs=16 \
  actor.micro_batch_size=32 \
  actor.global_batch_size=256
```

训练日志会写到：

```text
logs/<timestamp>-maniskill_peg_insertion_vertical_ppo_openpi_pi05/
```

常看文件：

| 文件/目录 | 内容 |
|---|---|
| `run_embodiment.log` | 训练 stdout/stderr。 |
| `checkpoints/global_step_<N>/actor` | 可用于评测的 actor checkpoint。 |
| `video/eval/` | 验证阶段视频，取决于配置。 |
| TensorBoard log | `runner.logger.logger_backends=["tensorboard"]`。 |

查看曲线：

```bash
tensorboard --logdir logs
```

重点观察 `eval/success_once`、`env/success_once`、reward 和 loss 是否稳定。

## Checkpoint 测试

使用训练保存的 actor 目录，例如：

```text
logs/<run>/checkpoints/global_step_50/actor
```

运行评测：

```bash
cd /home/hebu/code/robofape/RLinf_RoboFAPE

VENV_DIR=/opt/kairan/envs/rlinf \
CHECKPOINT_PATH=/path/to/checkpoints/global_step_50/actor \
GPU_IDS=0 \
NUM_EVAL_EPISODES=25 \
NUM_ENVS=5 \
SAVE_VIDEO=true \
bash run_train/eval_checkpoint/run_peginsertion.sh
```

要求：`NUM_EVAL_EPISODES` 必须能被 `NUM_ENVS` 整除，因为评测按固定并行 batch 跑完。

常用评测变量：

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `CHECKPOINT_PATH` | 见脚本默认值 | 被测试的 actor checkpoint。 |
| `GPU_IDS` | `0` | 评测用 GPU placement，可写单卡 id 或范围字符串。 |
| `NUM_EVAL_EPISODES` | `25` | 总评测轨迹数。 |
| `NUM_ENVS` | `5` | 并行环境数。 |
| `MAX_EPISODE_STEPS` | `600` | 单条轨迹最大步数。 |
| `SEED` | `0` | 评测 seed。 |
| `SAVE_VIDEO` | `true` | 是否保存视频。 |
| `IGNORE_TERMINATIONS` | `true` | 成功后是否继续跑到 rollout 长度。 |
| `FIXED_RESET_STATE_IDS` | `false` | 是否固定 reset state id。 |

评测输出目录：

```text
logs/<timestamp>-eval-PegInsertionVertical-v1/
```

重点文件：

| 文件/目录 | 内容 |
|---|---|
| `eval.log` | 完整评测日志和 resolved config。 |
| `evaluation_summary.json` | checkpoint 路径、轨迹数和 metrics 汇总。 |
| `video/eval/` | 评测视频。 |


