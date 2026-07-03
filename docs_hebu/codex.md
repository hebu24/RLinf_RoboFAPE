# ManiSkill pi0.5 + PPO 后训练运行链路

ManiSkill `PutOnPlateInScene25Main-v3` 任务上对pi0.5 VLA + PPO 后训练

## Config

- 主配置：`examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`
- task_env：`examples/embodiment/config/env/maniskill_put_on_plate_in_scene_25_main.yaml`
- base policy：`examples/embodiment/config/model/pi0_5.yaml`
- entrance：`examples/embodiment/run_embodiment.sh`
- SFT checkpoint：`RLinf/RLinf-Pi05-ManiSkill-25Main-SFT` (Supervised Fine-Tuning)


## 安装环境

安装 OpenPI 模型栈和 ManiSkill/LIBERO 环境依赖：

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero
source .venv/bin/activate
```

下载较慢时使用镜像：

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero --use-mirror
source .venv/bin/activate
```

这里的几个标识分别表示：

| 用途 | 值 |
|---|---|
| 安装模型目标 | `--model openpi` |
| 安装环境目标 | `--env maniskill_libero` |
| 运行时环境类型 | `env_type: maniskill` |
| ManiSkill 任务 ID | `PutOnPlateInScene25Main-v3` |
| RLinf 模型类型 | `model_type: openpi` |
| OpenPI 数据配置 | `pi05_maniskill` |

安装后主要资源位于：

- Python 环境：`.venv/`
- ManiSkill assets：`$HOME/.maniskill`
- SAPIEN PhysX assets：`$HOME/.sapien/physx/...`
- OpenPI tokenizer：`$HOME/.cache/openpi`

## 下载并配置模型

下载与该任务匹配的 pi0.5 SFT checkpoint：

```bash
hf download RLinf/RLinf-Pi05-ManiSkill-25Main-SFT \
  --local-dir /path/to/model/RLinf-Pi05-ManiSkill-25Main-SFT
```

然后在 `examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml` 中同时修改：

```yaml
rollout:
  model:
    model_path: /path/to/model/RLinf-Pi05-ManiSkill-25Main-SFT

actor:
  model:
    model_path: /path/to/model/RLinf-Pi05-ManiSkill-25Main-SFT
```

Actor 和 Rollout 必须使用同一套初始 checkpoint。不要直接使用 PickCube MLP
配置中的空 `model_path`。

## 启动训练

启动单机 Ray head，再运行同步 PPO：

```bash
ray start --head
bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05
```

该命令会创建带时间戳的 `logs/...-maniskill_ppo_openpi_pi05/` 日志目录，并通过
`examples/embodiment/train_embodied_agent.py` 启动训练。

首次测试建议先在配置中降低资源规模，例如：

```yaml
env:
  train:
    total_num_envs: 8
  eval:
    total_num_envs: 8

actor:
  micro_batch_size: 1
  global_batch_size: 128
```

这里每个环境产生 `80 / 5 = 16` 个动作 chunk，因此 8 个环境对应 128 个训练
样本。`global_batch_size` 还必须能被 `micro_batch_size * actor_world_size` 整除。
小规模参数仅用于验证初始化和数据链路，不代表正式训练参数。

## Hydra 配置组成

`maniskill_ppo_openpi_pi05.yaml` 通过 Hydra `defaults` 合并环境、模型、训练后端和
权重同步配置：

```yaml
defaults:
  - env/maniskill_put_on_plate_in_scene_25_main@env.train
  - env/maniskill_put_on_plate_in_scene_25_main@env.eval
  - model/pi0_5@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
```

主配置的关键字段如下：

```yaml
runner:
  task_type: embodied

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all

algorithm:
  adv_type: gae
  loss_type: actor_critic
  reward_type: chunk_level
  logprob_type: chunk_level
  update_epoch: 5
  gamma: 0.99
  gae_lambda: 0.95

actor:
  training_backend: fsdp
  model:
    model_type: openpi
    add_value_head: true
    num_action_chunks: 5
    num_steps: 4
    policy_setup: widowx_bridge
    openpi:
      config_name: pi05_maniskill
      num_images_in_input: 1
      noise_method: flow_noise
      action_horizon: 8
      joint_logprob: true

reward:
  use_reward_model: false
```

环境配置来自 `maniskill_put_on_plate_in_scene_25_main.yaml`：

```yaml
env_type: maniskill
max_steps_per_rollout_epoch: 80
max_episode_steps: 80

init_params:
  id: PutOnPlateInScene25Main-v3
  obs_mode: rgb+segmentation
  sim_backend: gpu
  max_episode_steps: 80
  render_mode: all
  obj_set: train
  use_multiple_plates: false
```

## 端到端数据流

```text
Shell 脚本
  -> Hydra 合并配置
  -> validate_cfg 校验
  -> Cluster 连接 Ray
  -> HybridComponentPlacement 分配资源
  -> EnvWorker 创建 PutOnPlateInScene25Main-v3
  -> RolloutWorker 用 pi0.5 生成动作 chunk
  -> EnvWorker 执行动作并产生轨迹与环境奖励
  -> Actor 计算 GAE 和 PPO actor-critic loss
  -> FSDP 更新 pi0.5 与 value head
  -> WeightSyncer 将新权重同步给 RolloutWorker
  -> 进入下一轮采样
```

## 1. Shell 入口

运行：

```bash
bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05
```

脚本设置以下关键环境变量：

- `EMBODIED_PATH=examples/embodiment`
- `REPO_PATH=<repo root>`
- `SRC_FILE=examples/embodiment/train_embodied_agent.py`
- `MUJOCO_GL=egl`
- `PYOPENGL_PLATFORM=egl`
- `PYTHONPATH=${REPO_PATH}:...`

最终执行等价于：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config/ \
  --config-name maniskill_ppo_openpi_pi05 \
  runner.logger.log_path=<repo>/logs/<time>-maniskill_ppo_openpi_pi05
```

## 2. Hydra 和配置校验

入口 `examples/embodiment/train_embodied_agent.py` 接收 Hydra 合并后的
`DictConfig`，再调用 `rlinf/config.py::validate_cfg`。

本任务的关键最终值应为：

```text
cfg.runner.task_type = embodied
cfg.actor.model.model_type = openpi
cfg.actor.model.openpi.config_name = pi05_maniskill
cfg.env.train.env_type = maniskill
cfg.env.train.init_params.id = PutOnPlateInScene25Main-v3
cfg.algorithm.loss_type = actor_critic
```

`validate_cfg` 会校验任务类型、模型与算法字段，并补齐 worker 日志和 profiling
等通用配置。

## 3. Cluster 和 Placement

同步入口创建：

```python
cluster = Cluster(cluster_cfg=cfg.cluster, ...)
component_placement = HybridComponentPlacement(cfg, cluster)
```

默认配置把 Actor、Env 和 Rollout 放到所有可见 GPU：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all
```

pi0.5 的显存占用明显高于 MLP。正式修改 placement 前，应同时检查可见 GPU 数、
`total_num_envs`、`micro_batch_size` 和 offload 设置。

## 4. WorkerGroup 初始化

同步入口启动三个核心 worker group：

```python
actor_group = EmbodiedFSDPActor.create_group(cfg).launch(...)
rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(...)
env_group = EnvWorker.create_group(cfg).launch(...)
```

当前配置设置：

```yaml
reward:
  use_reward_model: false
```

因此奖励由 ManiSkill 环境直接产生，不启动独立 RewardWorker。

## 5. EnvWorker

`rlinf/workers/env/env_worker.py::EnvWorker.init_worker` 执行以下步骤：

1. 根据 worker rank 更新环境配置。
2. 调用 `get_env_cls("maniskill", cfg.env.train)`。
3. 创建 `ManiskillEnv`。
4. 通过 `cfg.env.train.init_params` 创建 `PutOnPlateInScene25Main-v3`。

环境向策略提供 RGB、分割信息、机器人状态和语言指令。包装器负责 reset、step、
环境奖励、终止状态、指标及轨迹传输。

## 6. RolloutWorker

`rlinf/workers/rollout/hf/huggingface_worker.py::MultiStepRolloutWorker` 使用
`model_type: openpi` 构建 pi0.5 rollout policy。

关键接口参数：

```yaml
actor:
  model:
    policy_setup: widowx_bridge
    num_action_chunks: 5
    openpi:
      config_name: pi05_maniskill
      num_images_in_input: 1
      action_horizon: 8
      noise_method: flow_noise
```

RolloutWorker 接收环境观测和语言指令，通过 flow-noise 采样生成动作 chunk，再把
动作发送给 EnvWorker。`widowx_bridge` 决定 ManiSkill 动作的格式化和映射方式。

## 7. ActorWorker

同步配置使用：

```yaml
algorithm:
  loss_type: actor_critic
```

因此 Actor 类型为 `EmbodiedFSDPActor`。初始化阶段会：

1. 从 SFT checkpoint 构建 pi0.5。
2. 添加 value head。
3. 初始化 FSDP 和 optimizer。
4. 准备 PPO 所需的 log-prob、value 和训练状态。

OpenPI 当前不支持此配置中的 gradient checkpointing，因此保持：

```yaml
actor:
  fsdp_config:
    gradient_checkpointing: false
```

## 8. 同步 PPO 循环

核心循环位于 `rlinf/runners/embodied_runner.py::EmbodiedRunner.run`。

### 8.1 同步权重

Runner 将 Actor 的最新权重同步给 RolloutWorker，保证后续采样使用最新策略。

### 8.2 环境交互

EnvWorker 和 RolloutWorker 通过 channel 并行交互：

```text
EnvWorker --observation/instruction--> RolloutWorker
EnvWorker <--action chunk------------- RolloutWorker
EnvWorker --trajectory---------------> ActorWorker
```

每个动作 chunk 最多向环境提供 5 个动作，OpenPI 内部 action horizon 为 8。

### 8.3 计算优势

Actor 根据环境奖励、value、termination 和 truncation 计算 GAE：

```yaml
algorithm:
  adv_type: gae
  gamma: 0.99
  gae_lambda: 0.95
  bootstrap_type: always
```

### 8.4 PPO 更新

Actor 对采样数据执行多轮更新：

```yaml
algorithm:
  update_epoch: 5
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  value_clip: 0.2
  entropy_bonus: 0.005
```

训练同时更新 pi0.5 policy 和 value head。动作与 log-prob 按 chunk 粒度处理。

### 8.5 验证、日志与 checkpoint

默认间隔：

```yaml
runner:
  val_check_interval: 10
  save_interval: 50
```

重点观察：

- `env/success_once`
- eval success 指标
- Actor policy/total loss
- critic value loss
- rollout 和 step 时间

checkpoint 保存到日志目录下的：

```text
checkpoints/global_step_<N>/
```

## 常见问题

### 模型路径错误

确认 `actor.model.model_path` 与 `rollout.model.model_path` 都指向同一个本地
`RLinf-Pi05-ManiSkill-25Main-SFT` 目录。

### EGL 或渲染错误

确认：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

同时检查当前进程可见 GPU 和 EGL/驱动安装状态。

### 显存不足

依次降低：

1. `env.train.total_num_envs`
2. `env.eval.total_num_envs`
3. `actor.micro_batch_size`
4. `actor.global_batch_size`

必要时重新规划 Actor、Rollout 和 Env 的 GPU placement，或启用配置中支持的
offload。

### 动作不匹配

不要沿用 PickCube MLP 的 `panda-qpos`、`action_dim: 8` 或
`num_action_chunks: 1`。本任务应保持 `policy_setup: widowx_bridge`、
`config_name: pi05_maniskill` 和现有 pi0.5 action-chunk 配置。

## 推荐源码阅读顺序

1. `examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`
2. `examples/embodiment/config/env/maniskill_put_on_plate_in_scene_25_main.yaml`
3. `examples/embodiment/config/model/pi0_5.yaml`
4. `examples/embodiment/run_embodiment.sh`
5. `examples/embodiment/train_embodied_agent.py`
6. `rlinf/config.py::validate_cfg`
7. `rlinf/utils/placement.py`
8. `rlinf/runners/embodied_runner.py`
9. `rlinf/workers/env/env_worker.py`
10. `rlinf/envs/maniskill/maniskill_env.py`
11. `rlinf/workers/rollout/hf/huggingface_worker.py`
12. `rlinf/models/embodiment/openpi/`
13. `rlinf/workers/actor/fsdp_actor_worker.py`
14. `rlinf/envs/action_utils.py::prepare_actions_for_maniskill`

## 最小心智模型

```text
ManiskillEnv(PutOnPlateInScene25Main-v3)
  -> EnvWorker
  -> rollout_channel(RGB + state + language instruction)
  -> MultiStepRolloutWorker(pi0.5/OpenPI)
  -> env_channel(action chunk)
  -> EnvWorker.step()
  -> actor_channel(trajectory)
  -> EmbodiedFSDPActor(pi0.5 + value head)
  -> GAE + PPO actor_critic loss
  -> FSDP optimizer step
  -> WeightSyncer
  -> RolloutWorker
```

当前测试目标是先验证该现成链路可以完成模型加载、环境创建、采样、PPO 更新和
checkpoint 保存，再扩大并行环境数和 batch size。
