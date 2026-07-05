# ManiSkill pi0.5 + PPO 后训练运行链路

ManiSkill `PutOnPlateInScene25Main-v3` 任务上对pi0.5 VLA + PPO 后训练

## Config

- 主配置：`examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`
- task_env：`examples/embodiment/config/env/maniskill_put_on_plate_in_scene_25_main.yaml`
- base policy：`examples/embodiment/config/model/pi0_5.yaml`
- entrance：`examples/embodiment/run_embodiment.sh`
- SFT checkpoint：`/opt/yingxi/rlinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise/checkpoints/global_step_150/actor`
  - `actor.model.model_path` 和 `rollout.model.model_path` 都直接指向这个目录


## .env

```bash
bash requirements/install.sh embodied --model openpi --env maniskill_libero
source .venv/bin/activate
```

## 模型路径引用

```yaml
rollout:
  model:
    model_path: /opt/yingxi/rlinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise/checkpoints/global_step_150/actor

actor:
  model:
    model_path: /opt/yingxi/rlinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise/checkpoints/global_step_150/actor
```

## 启动训练

使用 `/opt/kairan/envs/rlinf` 环境和 GPU 6，以较小并行参数启动单卡同步 PPO：

watch -n 1 nvidia-smi

```bash
source /opt/kairan/envs/rlinf/bin/activate
bash run_train/test_maniskill_pi0.5/run.sh
```

测试配置位于 `run_train/test_maniskill_pi0.5/config/`，固定使用物理 GPU 6；训练环境数为 16，评估环境数为 4，global batch size 为 256。评估视频每 10 个 step 保存到本次日志目录的 `video/eval/`。

创建目录： `logs/<时间>-maniskill_ppo_openpi_pi05-gpu6/` 日志目录
启动训练： `examples/embodiment/train_embodied_agent.py` 

## Hydra 配置组成

Hydra合并环境、模型、训练后端和权重同步配置：

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


```bash
bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05
```
等价于：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config/ \
  --config-name maniskill_ppo_openpi_pi05 \
  runner.logger.log_path=<repo>/logs/<time>-maniskill_ppo_openpi_pi05
```

## 2. Hydra 和配置校验

`examples/embodiment/train_embodied_agent.py` 
接收 Hydra 合并后的`DictConfig`，再调用 `rlinf/config.py::validate_cfg`。

```text
cfg.runner.task_type = embodied
cfg.actor.model.model_type = openpi
cfg.actor.model.openpi.config_name = pi05_maniskill
cfg.env.train.env_type = maniskill
cfg.env.train.init_params.id = PutOnPlateInScene25Main-v3
cfg.algorithm.loss_type = actor_critic
```

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

## 4. WorkerGroup 初始化

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
**外接reward model**
如果你要接外部 reward model，把它改到 `reward:` 下面，不要放进 `actor` 或 `rollout`。
这会额外启动 `EmbodiedRewardWorker`，由它根据轨迹/观测计算奖励，再回传给 `EnvWorker`。

```yaml
reward:
  use_reward_model: true
  group_name: "RewardGroup"
  reward_mode: "terminal"   # 或 per_step
  reward_weight: 1.0
  env_reward_weight: 0.0
  model:
    model_type: "resnet"    # 也可以是 history_vlm
    model_path: /path/to/reward_model_checkpoint
    precision: "fp32"
```

这几个字段的含义是：

- `use_reward_model: true`：启用外接 reward model
- `group_name: "RewardGroup"`：给 reward worker 起名
- `reward_mode: "terminal"`：只在终止步算奖励；`per_step` 则每步都算
- `reward_weight` / `env_reward_weight`：控制 reward model 分数和环境奖励的加权和
- `reward.model.model_path`：reward 模型权重路径，和 actor/rollout 的 SFT checkpoint 不是一回事

对 ManiSkill 来说，你可以先用环境奖励跑通，再把 reward model 打开做加权融合；如果只是做 reward model 打分，通常把 `env_reward_weight` 设成 `0.0`。


## 5. EnvWorker

`rlinf/workers/env/env_worker.py::EnvWorker.init_worker` 执行以下步骤：

1. 根据 worker rank 更新环境配置。
2. 调用 `get_env_cls("maniskill", cfg.env.train)`。
3. 创建 `ManiskillEnv`。
4. 通过 `cfg.env.train.init_params` 创建 `PutOnPlateInScene25Main-v3`。


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
动作发送给 EnvWorker。

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

```text
EnvWorker --observation/instruction--> RolloutWorker
EnvWorker <--action chunk------------- RolloutWorker
EnvWorker --trajectory---------------> ActorWorker
```

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
