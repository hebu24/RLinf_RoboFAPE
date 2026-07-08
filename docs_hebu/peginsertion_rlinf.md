# PegInsertionVertical 部署 RLinf 调研

## 结论

RoboFPE 当前 `PegInsertionVertical` 已经适合迁入 RLinf 做最小闭环：任务主体、成功判定、dense reward、随机 reset、calibrated 外部相机和 hand camera 都在任务文件内，运行时不依赖导出的外部 mesh。

调研版本：RoboFPE `b7c276f`，RLinf `09326e2a`。

## 本次部署结果

已将 `PegInsertionVertical-v1` 接入 RLinf，并完成可用环境内的静态检查和 wrapper smoke。

主要改动：

| 文件 | 状态 | 说明 |
|---|---|---|
| `rlinf/envs/maniskill/tasks/peg_insertion_vertical.py` | 已新增 | 从 RoboFPE 迁入任务，补 Apache 头、`get_language_instruction()` 和 `get_pi05_proprio()`。 |
| `rlinf/config.py` | 已修改 | 新增 `panda-ee-dpose -> pd_ee_delta_pose` 映射。 |
| `rlinf/envs/maniskill/maniskill_env.py` | 已修改 | `wrap_obs_mode: simple` 返回 `task_descriptions`、`wrist_images: None`；支持 `cfg.reset_options` 透传；任务提供 `get_pi05_proprio()` 时使用 7D state。 |
| `rlinf/models/embodiment/openpi/policies/maniskill_policy.py` | 已修改 | 将 `observation/extra_view_image` 第一视角映射到 `left_wrist_0_rgb`，并打开对应 image mask。 |
| `run_train/peginsertion_maniskill_pi0.5/` | 已新增 | 独立 run script、env YAML 和 PPO pi0.5 配置。 |

关键配置：

```yaml
env:
  train:
    reset_options:
      randomize_initial_poses: True
  eval:
    reset_options:
      randomize_initial_poses: False

actor:
  model:
    action_dim: 7
    policy_setup: panda-ee-dpose
    openpi:
      config_name: pi05_maniskill
      num_images_in_input: 2
```

已验证：

| 检查 | 结果 |
|---|---|
| RoboFPE controller smoke | 通过：`pd_ee_delta_pose`、action space `(2, 7)`、`base_camera`/`hand_camera`、`step ok`。 |
| RLinf task 注册 | 通过：`rlinf.envs.maniskill` 自动注册 `PegInsertionVertical-v1`。 |
| RLinf wrapper GPU smoke | 通过：`states=(2,7)`、`main_images=(2,224,224,3)`、`extra_view_images=(2,1,224,224,3)`、prompt 正确、7D chunk step 正常。 |
| Hydra compose | 通过：`maniskill_peg_insertion_vertical_ppo_openpi_pi05` 可合成，`_self_` 已补，无 defaults 警告。 |
| Python 语法 / diff whitespace | 通过：`compileall` 和 `git diff --check`。 |

当前环境限制：

| 项 | 结果 |
|---|---|
| `validate_cfg` / `validate_embodied_cfg` | 未完整通过：会初始化 Ray，本机 `.venv` 启动 Ray GCS 超时。 |
| OpenPI model smoke | 未跑：当前可用环境没有 `openpi`，`/opt/kairan/envs/rlinf` 也不存在。 |
| ManiSkill editable 路径 | 当前 `maniskill_py311` 的 editable 安装指向不存在的 `/home/hebu/code/ManiSkill`；测试时临时用 `PYTHONPATH=/home/hebu/code/mam/ManiSkill`。 |
| libstdc++ | 导入全量 RLinf ManiSkill tasks 时需要 `LD_PRELOAD=/home/hebu/miniconda3/envs/maniskill_py311/lib/libstdc++.so.6`，否则会加载 `/usr/lib/nvidia/libstdc++.so.6` 并缺 `CXXABI_1.3.15`。 |

关键变化：

| 项 | 当前 RoboFPE 契约 | RLinf 部署影响 |
|---|---|---|
| env id | `PegInsertionVertical-v1` | 直接注册到 RLinf ManiSkill task。 |
| episode 长度 | `max_episode_steps=600` | RLinf env YAML 要同步覆盖训练/评估长度。 |
| 默认机器人 | `panda_wristcam`，也支持 `panda` | 优先用 `panda_wristcam`，保留 hand camera。 |
| 主相机 | `base_camera`，224×224，calibrated pose `p=[0.705400,-0.086655,0.686691]`、`q=[0.025112,-0.237384,-0.033640,0.970508]` | RLinf `wrap_obs_mode: simple` 会把它作为 `main_images`，不再缺主相机适配。 |
| hand camera | `robot_uids=panda_wristcam` 时自动补 `sensor_configs.hand_camera.width/height=224` | RLinf 会把非 `base_camera` sensor 堆到 `extra_view_images`；是否真正喂给 pi0.5 还要改 OpenPI transform。 |
| render camera | `render_camera`，640×480，同 calibrated pose，可由 `render_randomization_spec.camera` 覆盖 | 只用于视频/可视化，不作为 RLinf 主观测。 |
| 颜色 | peg 蓝色 `[0,134,214,255]/255`，孔座橙色 `[249,140,54,255]/255` | 迁入后按该版本重新生成截图。 |

## 源文件判定

| RoboFPE 文件 | 是否迁移 | 说明 |
|---|---:|---|
| `mani_envs/tasks/task_PegInsertionVertical.py` | 是 | 场景、`panda_wristcam`、相机、reset、success、reward 均在此。 |
| `mani_envs/solutions/solve_PegInsertionVertical.py` | 可选 | 用于 smoke、专家轨迹和后续 SFT，不是在线 RL 必需。 |
| `mani_envs/data_collection/task_descriptions/peg_insertion_vertical.json` | 可选 | 有 500 条 prompt；RLinf 在线训练可先用固定 prompt。 |
| `assets/peg_insertion_vertical_meshes/*` | 否 | 由导出脚本从 primitive 几何生成，任务运行时不读取。 |
| data collection / error solution / robometer | 否 | 属于数据筛选、奖励建模和失败标注流程，不是 RLinf 环境依赖。 |

源任务保留 `evaluate()["success"]`、dense reward 上限 10、normalized dense 上限 1。随机化仍通过 `reset(options={"randomize_initial_poses": True})` 触发。

## RLinf 所需改动

| 文件 | 改动 |
|---|---|
| `rlinf/envs/maniskill/tasks/peg_insertion_vertical.py` | 迁入当前 RoboFPE task；补 Apache 头、类型整理、`get_language_instruction()`，可选增加构造参数接入随机 reset。 |
| `rlinf/envs/maniskill/tasks/__init__.py` | 通常不需要改；RLinf 会自动 import `tasks/` 顶层 `.py`。只有放进子目录时才手动接入。 |
| `run_train/peginsertion_maniskill_pi0.5/config/env/maniskill_peg_insertion_vertical.yaml` | 定义 `env_type: maniskill`、`obs_mode: rgb`、`robot_uids: panda_wristcam`、`reward_mode: normalized_dense`、600 step、视频路径。 |
| `run_train/peginsertion_maniskill_pi0.5/config/maniskill_peg_insertion_vertical_ppo_openpi_pi05.yaml` | 以现有 pi0.5 PPO 配置为模板，替换 env、checkpoint、并行数和 batch。 |
| `rlinf/config.py` | 新增 `panda-ee-dpose -> pd_ee_delta_pose`，避免把 `panda_wristcam` 映射到只属于 RLinf 自定义 Panda agent 的 controller。 |
| `rlinf/envs/maniskill/maniskill_env.py` | 必须补固定 `task_descriptions`；若要配置化随机 reset，也在这里透传 `cfg.reset_options`。 |
| `rlinf/models/embodiment/openpi/...` | 若要使用 hand camera，新增 PegInsertion/OpenPI transform，把 `observation/extra_view_image` 映射到 wrist image 并打开 image mask。 |
| `tests/...` | 增加 task reset/step/render、obs shape、action shape、reward/success、向量环境 smoke；再补最小 e2e 配置。 |

prompt 直接补这句：

```python
def get_language_instruction(self):
    return ["insert the blue peg vertically into the orange hole"] * self.num_envs
```

控制名称不要混淆：`panda-*` 是 RLinf 的 `policy_setup` 标签，`pd_*` 是 ManiSkill 的 `control_mode`。建议新增：

```python
elif robot == "panda-ee-dpose":
    return "pd_ee_delta_pose"
```

然后在 pi0.5 配置里使用：

```yaml
actor:
  model:
    action_dim: 7
    policy_setup: panda-ee-dpose
```

这样 OpenPI 的 `[dx, dy, dz, droll, dpitch, dyaw, gripper]` 7D action 可直接进入标准 Panda/PandaWristCam EE delta pose controller。

## 建议初始 Env YAML

```yaml
env_type: maniskill
total_num_envs: null

wrap_obs_mode: simple
auto_reset: true
ignore_terminations: false
use_rel_reward: false
reward_mode: raw
seed: 0
group_size: 1
use_fixed_reset_state_ids: false
max_steps_per_rollout_epoch: 600
max_episode_steps: 600
task_description: "insert the blue peg vertically into the orange hole"
reset_options:
  randomize_initial_poses: true

video_cfg:
  save_video: false
  info_on_video: true
  video_base_dir: ${runner.logger.log_path}/video/train

init_params:
  id: PegInsertionVertical-v1
  num_envs: null
  obs_mode: rgb
  robot_uids: panda_wristcam
  control_mode: null
  sim_backend: gpu
  reward_mode: normalized_dense
  max_episode_steps: ${..max_episode_steps}
  render_mode: all
  sensor_configs:
    hand_camera:
      width: 224
      height: 224
```

`reset_options` 现在还不是 RLinf `ManiskillEnv.reset()` 的通用透传字段；部署时需要补实现，或把 `randomize_initial_poses` 做成任务构造参数。

## 7D State 具体定义

这里的 7D state 不是 peg/孔位姿，也不是 Panda 的完整 9D `qpos`。它是给 pi0.5 的 proprioception，必须是 `[num_envs, 7]`：

```text
[tcp_x, tcp_y, tcp_z, tcp_roll, tcp_pitch, tcp_yaw, gripper]
```

按 RLinf 现有 Panda ManiSkill 任务保持同一契约：TCP pose 用机器人 root/base frame，旋转用 `sxyz` Euler，gripper 用最后一个 finger qpos 乘 2。

```python
from transforms3d.euler import mat2euler

def get_pi05_proprio(self):
    tcp_pose_in_root = self.agent.robot.pose.inv() * self.agent.tcp.pose
    tcp_T = tcp_pose_in_root.to_transformation_matrix().cpu().numpy()

    pos = torch.as_tensor(tcp_T[:, :3, 3], device=self.device, dtype=torch.float32)
    euler = torch.as_tensor(
        np.stack([mat2euler(tcp_T[i, :3, :3], "sxyz") for i in range(self.num_envs)]),
        device=self.device,
        dtype=torch.float32,
    )
    gripper = self.agent.robot.get_qpos().to(torch.float32)[:, -1:] * 2
    return torch.cat([pos, euler, gripper], dim=1)
```

部署时让 `ManiskillEnv._wrap_obs(wrap_obs_mode="simple")` 对 `PegInsertionVertical-v1` 使用这个张量作为 `states`，不要使用 `common.flatten_state_dict(raw_obs)` 的完整任务状态。

## 动作控制器验证命令

在云服务器 RoboFPE 环境里先验证 `pd_ee_delta_pose` 是否可用，以及 action 是否就是 7D：

```bash
cd /home/hebu/code/robofape/RoboFPE
python - <<'PY'
import gymnasium as gym
import torch

import mani_envs.tasks  # noqa: F401, register PegInsertionVertical-v1

env = gym.make(
    "PegInsertionVertical-v1",
    num_envs=2,
    obs_mode="rgb",
    robot_uids="panda_wristcam",
    control_mode="pd_ee_delta_pose",
    sim_backend="gpu",
    reward_mode="normalized_dense",
    render_mode="rgb_array",
    max_episode_steps=20,
    sensor_configs={"hand_camera": {"width": 224, "height": 224}},
)

print("control_mode:", env.unwrapped.control_mode)
print("action_space:", env.action_space)

obs, info = env.reset(seed=0, options={"randomize_initial_poses": True})
print("sensor keys:", sorted(obs["sensor_data"].keys()))
print("base_camera:", tuple(obs["sensor_data"]["base_camera"]["rgb"].shape))
print("hand_camera:", tuple(obs["sensor_data"]["hand_camera"]["rgb"].shape))

action = torch.zeros((env.unwrapped.num_envs, 7), device=env.unwrapped.device)
action[:, -1] = 1.0
obs, reward, terminated, truncated, info = env.step(action)

print("step ok")
print("reward:", reward.detach().cpu().tolist())
print("success:", info["success"].detach().cpu().tolist())
print("terminated:", terminated.detach().cpu().tolist())
print("truncated:", truncated.detach().cpu().tolist())

env.close()
PY
```

通过标准：

| 检查 | 期望 |
|---|---|
| `control_mode` | 打印 `pd_ee_delta_pose`。 |
| `action_space` | 最后一维是 7。 |
| `sensor keys` | 同时包含 `base_camera` 和 `hand_camera`。 |
| `step ok` | 不报 shape mismatch / controller not found。 |

如果这里失败，先不要接 RLinf；说明 `panda_wristcam + pd_ee_delta_pose` 在当前 ManiSkill/RoboFPE 环境里不成立，需要换 controller 或补 agent controller。

## 随机 Reset 具体含义

RoboFPE 当前任务有两种 reset：

| reset 方式 | 结果 |
|---|---|
| `env.reset(options={})` | 固定初始位姿：peg、hole、robot qpos 每次基本相同。 |
| `env.reset(options={"randomize_initial_poses": True})` | 每个 episode 随机 hole xy、peg xy 和 robot qpos。 |

结论：`options={}` 是固定初始位置；`options={"randomize_initial_poses": True}` 会随机 peg、hole 和 robot 初始位姿。RL 训练要后者，否则只会学一个固定摆放。建议 train 开 `true`，eval 先用 `false` 固定评估。

RLinf 现在的问题是：`ManiskillEnv.reset()` 自动 reset 时只传 `{}`、`episode_id` 或 `env_idx`，不会自动带上 `randomize_initial_poses=True`。所以要补一个配置透传：

```yaml
env:
  train:
    reset_options:
      randomize_initial_poses: true
  eval:
    reset_options:
      randomize_initial_poses: false
```

期望实现效果：

```python
base_options = OmegaConf.to_container(
    getattr(self.cfg, "reset_options", {}),
    resolve=True,
) or {}
options = dict(base_options)
```

然后原来需要追加 `env_idx` / `episode_id` 的地方继续追加，不要覆盖 `randomize_initial_poses`。

## 当前仍需确认

| 项 | 状态 | 处理 |
|---|---|---|
| 主相机 | 已解决 | `base_camera` 已是 224×224 calibrated 外部视角，可直接作为 `main_images`。 |
| hand camera | 已接入 wrapper 和 OpenPI transform | `extra_view_images[:, 0]` 会进入 `left_wrist_0_rgb`；仍需在完整 OpenPI 环境确认 checkpoint/norm stats 对 2 图输入的表现。 |
| prompt | 已实现 | 固定使用 `insert the blue peg vertically into the orange hole`。 |
| 7D state | 已实现并 smoke | 使用 `TCP xyz + TCP sxyz Euler + gripper`，shape 为 `[num_envs, 7]`。 |
| action controller | 已 smoke | `panda-ee-dpose -> pd_ee_delta_pose`；RoboFPE 和 RLinf wrapper 均可 7D step。 |
| 随机 reset | 已实现 | `ManiskillEnv.reset()` 会合并 `cfg.reset_options`，train 随机、eval 固定。 |
| checkpoint | 仍需完整训练环境确认 | `/opt/yingxi/rlinf/RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise/checkpoints/global_step_150/actor`；当前环境缺 `openpi`，未做模型加载和 norm stats smoke。 |

## 实施顺序

1. 迁入 `task_PegInsertionVertical.py`，保留 `base_camera` 与 `hand_camera` 当前定义。
2. 增加 `get_language_instruction()` 和随机 reset 配置接入。
3. 增加 `panda-ee-dpose` 控制映射，做单环境 `reset/step/render` smoke。
4. 在 `ManiskillEnv._wrap_obs()` 或任务适配层固定 7D state。
5. 先按 1 图像输入跑 pi0.5 最小 PPO；确认 action、reward、success 和 checkpoint 加载。
6. 再接 hand camera 到 OpenPI transform，更新 `num_images_in_input`、norm stats 和 smoke。
7. 生成少量 solver 轨迹检查动作尺度；稳定后补 unit/e2e 和正式 EN/ZH 文档。
