# 自定义 ManiSkill PegInsertion 迁移到 RLinf

## 结论

 `env_type`：`maniskill` + VLA：pi0.5
**重点是让你的自定义 ManiSkill task 满足 RLinf 的 obs/action/reward 接口**

把实验室相似的 sim task 固化成一个可稳定 `gym.make()` 的 ManiSkill env，并补齐 pi0.5 需要的图像、状态、语言指令和动作格式。

## 需要准备的任务文件

| Task 主文件 | `lab_peg_insertion.py` | 定义场景、物体、reset、reward、success、camera。 |
| 资产文件 | `assets/...` | peg、孔、治具、桌面、相机外参、纹理、URDF/MJCF/mesh。 |
| 机器人文件 | 自定义 robot/agent 文件，如有 | 如果不是 ManiSkill 内置机器人，需要一起迁。 |
| 注册入口 | `@register_env("LabPegInsertion-v1", ...)` | 让 `gym.make("LabPegInsertion-v1")` 可用。 |
| 配置记录 | env id、robot、control mode、camera、reward、max steps | 后续写 RLinf YAML 的依据。 |
| 演示数据 | motion planning / teleop 轨迹，可选但建议 | 用于 pi0.5 SFT 或检查动作尺度。 |
| 安装依赖 | fork 版本、额外 pip 包、资产下载脚本 | 保证别人能复现环境。 |

## 需要放到 RLinf 的文件

| RLinf 文件 | 必需性 | 要做的事 |
|---|---:|---|
| `rlinf/envs/maniskill/tasks/lab_peg_insertion.py` | 必需 | 放自定义 task 或适配 subclass。 |
| `rlinf/envs/maniskill/tasks/__init__.py` | 视情况 | 如果 task 在子目录，手动 import；顶层 `.py` 通常会被自动遍历导入。 |
| `examples/embodiment/config/env/maniskill_lab_peg_insertion.yaml` | 必需 | 写 `env_type: maniskill` 和 `init_params.id`。 |
| `examples/embodiment/config/maniskill_lab_peg_insertion_ppo_openpi_pi05.yaml` | 必需 | 复制 pi0.5 PPO 配置并替换 env、模型路径、batch。 |
| `rlinf/envs/maniskill/maniskill_env.py` | 大概率需要 | 补 `task_descriptions`、`wrist_images`、`extra_view_images`、reward 兼容。 |
| `rlinf/config.py` | 视动作而定 | 如果新增 `policy_setup`，补到 `get_robot_control_mode()`。 |
| `requirements/install.sh` | 视依赖而定 | 如果依赖 ManiSkill fork 或额外包，固定安装方式。 |
| `requirements/embodied/download_assets.sh` | 视资产而定 | 如果资产不在 pip 包里，补下载/复制逻辑。 |

## Step 1: Freeze Task Contract

先把自定义任务接口定死：

| 项 | 必须确认 |
|---|---|
| `env_id` | 例如 `LabPegInsertion-v1`。 |
| `robot_uids` | 用 Panda、WidowX，还是实验室自定义 robot。 |
| `control_mode` | 必须和 pi0.5 输出动作维度一致。 |
| `action_dim` | pi0.5 常用 7 维：xyz delta + rot delta + gripper。 |
| `obs_mode` | 建议 `rgb` 或 `rgb+segmentation`，同时能取 state。 |
| camera keys | 至少有主视角；有腕部相机更好。 |
| reward | 推荐先用 ManiSkill dense / normalized dense，经 RLinf `reward_mode: raw` 透传。 |
| success info | `info["success"]` 必须存在，RLinf 会记录 success。 |
| prompt | 固定任务语言，比如 `insert the peg into the socket`。 |

## Step 2: Port Task Code

把自定义 ManiSkill task 迁到 `rlinf/envs/maniskill/tasks/`。注意：

- 不要保留绝对路径，如 `/home/.../assets`。
- 资产路径用环境变量或相对路径，例如 `MANISKILL_ASSET_DIR`。
- `evaluate()` 至少返回 `{"success": tensor}`。
- `compute_dense_reward()` 或 `compute_normalized_dense_reward()` 保留原任务逻辑。
- 如果 reset 依赖实验室标定参数，把这些参数放进 YAML 或资产文件，不要写死在代码里。

## Step 3: Adapt Observation for pi0.5

pi0.5 需要 RLinf 统一 obs：

```python
{
    "main_images": ...,
    "wrist_images": None or ...,
    "extra_view_images": None or ...,
    "states": ...,
    "task_descriptions": ["insert the peg into the socket", ...],
}
```

需要做两件事：

1. 在自定义 task 中提供 `get_language_instruction()`，或在 `ManiskillEnv` 里从 `cfg.task_description` 兜底。
2. 在 `ManiskillEnv._wrap_obs()` 中保证 `wrist_images` 和 `extra_view_images` 键始终存在，可为 `None`。

## Step 4: Align Action for pi0.5

继续使用现有 pi0.5 时，优先让任务使用 7 维 EE pose delta action：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

如果你的自定义任务用 Panda EE pose 控制，建议新增：

```python
elif robot == "panda-ee-dpose":
    return "pd_ee_delta_pose"
```

并在 pi0.5 配置里写：

```yaml
actor:
  model:
    model_type: openpi
    action_dim: 7
    num_action_chunks: 5
    policy_setup: panda-ee-dpose
    openpi:
      config_name: pi05_maniskill
```

如果实验室任务使用自定义 robot/control mode，就需要新增对应 `policy_setup -> control_mode` 映射，并确认 `prepare_actions_for_maniskill()` 不会错误缩放或重排动作。

## Step 5: Add Env Config

新增 `examples/embodiment/config/env/maniskill_lab_peg_insertion.yaml`：

```yaml
env_type: maniskill

total_num_envs: null
wrap_obs_mode: simple
auto_reset: false
ignore_terminations: false
use_rel_reward: false
reward_mode: raw
seed: 0
group_size: 1
use_fixed_reset_state_ids: false
max_steps_per_rollout_epoch: 100
max_episode_steps: 100
task_description: "insert the peg into the socket"

video_cfg:
  save_video: false
  info_on_video: true
  video_base_dir: ${runner.logger.log_path}/video/train

enable_offload: false

init_params:
  id: LabPegInsertion-v1
  num_envs: null
  obs_mode: rgb
  robot_uids: panda_wristcam
  control_mode: null
  sim_backend: gpu
  reward_mode: normalized_dense
  max_episode_steps: ${..max_episode_steps}
  render_mode: all
```

## Step 6: Add pi0.5 Training Config

复制 `examples/embodiment/config/maniskill_ppo_openpi_pi05.yaml`，改成：

```yaml
defaults:
  - env/maniskill_lab_peg_insertion@env.train
  - env/maniskill_lab_peg_insertion@env.eval
  - model/pi0_5@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - weight_syncer/patch_syncer@weight_syncer
  - override hydra/job_logging: stdout

runner:
  logger:
    experiment_name: maniskill_lab_peg_insertion_ppo_openpi_pi05

actor:
  model:
    model_path: /path/to/pi05/checkpoint
    model_type: openpi
    add_value_head: true
    action_dim: 7
    num_action_chunks: 5
    num_steps: 4
    policy_setup: panda-ee-dpose
    openpi:
      config_name: pi05_maniskill
      num_images_in_input: 1
      action_horizon: 8
      noise_method: flow_noise
      joint_logprob: true

rollout:
  model:
    model_path: ${actor.model.model_path}
```

如果现有 pi0.5 checkpoint 没见过你的实验室视角和 peg/socket，建议先用自定义任务的 demo 做一次 SFT，再跑 RL。

## Step 7: Validate

按顺序检查：

| 检查 | 通过标准 |
|---|---|
| ManiSkill smoke | `gym.make("LabPegInsertion-v1")` 能 reset/step/render。 |
| RLinf env smoke | `ManiskillEnv` 输出含 `main_images/states/task_descriptions`。 |
| pi0.5 obs | 不报 `KeyError: task_descriptions/wrist_images/extra_view_images`。 |
| action shape | pi0.5 输出 `[B, chunks, 7]`，env 接收无 shape mismatch。 |
| reward | `reward_mode: raw` 下 return 正常，`info["success"]` 正常。 |
| 小规模训练 | `total_num_envs=1~4` 能完成一次 rollout 和 actor update。 |

## 最短实施顺序

1. **Task Contract**：固定 env id、robot、control mode、obs、reward、success。
2. **Task Files**：迁 `lab_peg_insertion.py` 和资产。
3. **RLinf Env YAML**：新增 `maniskill_lab_peg_insertion.yaml`。
4. **Observation Patch**：补齐 pi0.5 需要的 obs keys。
5. **Action Mapping**：补 `policy_setup -> control_mode`。
6. **pi0.5 Config**：复制并改 `maniskill_ppo_openpi_pi05.yaml`。
7. **Smoke Then Scale**：先 1 个 env 跑通，再扩大并行数。
