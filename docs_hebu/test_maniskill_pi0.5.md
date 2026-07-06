# ManiSkill pi0.5 PPO 训练阶段分析

## 1. 分析范围

- 指标文件：`logs/metrics.log`，共 386 个连续记录，覆盖本次运行的 step 1–386/1000（38.6%）。
- 中断发生在 step 386 完成后、下一轮 rollout 期间；按每 50 step 保存一次的配置，最近的完整 checkpoint 应为 `global_step_350`，恢复后会丢弃尚未保存的 36 step 更新。
- 本文只依据 `metrics.log` 和 `run_train/test_maniskill_pi0.5/` 中的静态配置分析，没有读取 TensorBoard event、评估视频或 checkpoint 权重。
- 这不是从零开始训练：actor 和 rollout 都由已有的 `RLinf-Pi05-ManiSkill-25Main-RL-FlowNoise/checkpoints/global_step_150/actor` 初始化。本日志中的 step 是新运行的局部计数。

## 2. 关键训练参数

| 项目 | 当前设置 |
| --- | --- |
| 任务 | `PutOnPlateInScene25Main-v3`，ManiSkill，训练对象集 `train` |
| 模型 | OpenPI pi0.5，`flow_noise`，1 张输入图像，value head 接在 VLM 后 |
| 动作 | `num_action_chunks=5`，`action_horizon=8`，flow 推理步数 4 |
| 算法 | PPO actor-critic + GAE，chunk-level reward/logprob/entropy |
| PPO | `update_epoch=5`，clip=0.2，`gamma=0.99`，`gae_lambda=0.95` |
| 正则 | advantage normalization，entropy bonus=0.005，`kl_beta=0` |
| 优化器 | actor LR `7.91e-6`，value LR `1.55e-4`，grad clip=1.0 |
| batch | micro batch=32，global batch=256 |
| 环境 | train 16 个 env；eval 4 个固定 reset state；每个 episode 最长 80 step |
| reward | ManiSkill relative reward（相邻 dense reward 的差值） |
| 调度 | 每 10 step eval，每 50 step 保存，最多 1000 step |
| 设备 | 单卡 GPU 6，FSDP `no_shard` |

## 3. 总体结论

训练过程没有出现 NaN、Inf、OOM 或明显数值爆炸，checkpoint/恢复机制也足以继续运行。但是，截至 step 386，**没有证据表明策略性能在稳定提升**：

1. 训练环境的成功率和回报相对开头下降，episode 变长，说明当前 policy 在训练分布上的完成效率变差。
2. 固定 eval 的 `success_once` 有局部改善，但整体高度波动且不单调；eval 只有 4 条轨迹，每次结果只能按 0.25 跳变，统计量不足以判断优劣。
3. critic 的 explained variance 从较低水平升到约 0.6，说明 value head 确实学到了可用的回报排序。
4. actor 的 approximate KL 和 clip fraction 持续升高；最近约 34% 的样本触发 PPO clipping，且记录到的梯度范数始终远高于 1.0，实际每次更新都依赖梯度裁剪。训练尚未发散，但更新偏激进。
5. 初始模型本身已经是 RL checkpoint，训练早期 train success 很高；继续 PPO 后的下降更像是对强初始化策略的漂移，而不是正常的“从低到高”学习过程。

因此，当前状态可以评价为：**流程正常、critic 学习合理，但 actor 的收益不明确并存在退化信号，不建议在完全不调整评估和超参数的情况下直接跑满 1000 step。**

## 4. 训练指标趋势

以下均为对应区间内的均值；最后一行只有 36 个 step。

| step 区间 | train success_once | train return | episode_len | rollout rewards | approx KL | clip fraction | critic EV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–50 | 0.831 | 0.952 | 38.9 | 0.140 | 0.049 | 0.213 | 0.244 |
| 51–100 | 0.811 | 0.914 | 40.4 | 0.195 | 0.058 | 0.231 | 0.500 |
| 101–150 | 0.763 | 0.825 | 42.4 | 0.206 | 0.061 | 0.255 | 0.511 |
| 151–200 | 0.718 | 0.781 | 45.5 | 0.155 | 0.081 | 0.301 | 0.561 |
| 201–250 | 0.709 | 0.744 | 45.7 | 0.117 | 0.082 | 0.332 | 0.514 |
| 251–300 | 0.698 | 0.781 | 46.7 | 0.119 | 0.079 | 0.325 | 0.517 |
| 301–350 | 0.743 | 0.834 | 43.3 | 0.134 | 0.084 | 0.328 | 0.581 |
| 351–386 | 0.712 | 0.814 | 46.7 | 0.141 | 0.086 | 0.345 | 0.614 |

将前 50 step 与最后 50 step（337–386）比较：

- train `success_once`：0.831 → 0.736，下降 9.4 个百分点（相对下降 11.4%）。
- train `return`：0.952 → 0.843，相对下降 11.4%。
- `episode_len`：38.9 → 45.1，增长 16.0%，完成任务更慢。
- approximate KL：0.049 → 0.085，增长 72%。
- clip fraction：0.213 → 0.340，增长 60%。
- critic explained variance：0.244 → 0.616，明显改善。

`rollout/rewards` 和 `returns_mean` 不应直接当作任务成功率：runner 会进行 GAE/bootstrap，量纲也受 value 预测影响。判断任务表现应优先看环境 `success_once`、`return` 以及独立 eval。

## 5. Eval 结果与局限

eval 每 10 step 运行一次，每次只有 4 个固定环境。分段均值如下：

| eval step | 次数 | success_once | success_at_end | return |
| --- | ---: | ---: | ---: | ---: |
| 10–100 | 10 | 0.425 | 0.100 | 0.255 |
| 110–200 | 10 | 0.575 | 0.200 | 0.378 |
| 210–300 | 10 | 0.500 | 0.300 | 0.403 |
| 310–380 | 8 | 0.563 | 0.219 | 0.394 |
| 全部 | 38 | 0.513 | 0.204 | 0.355 |

可以看到 eval 比最初阶段有所改善，但没有单调趋势。例如：

- step 200：`success_once=0.75`、`success_at_end=0.75`；
- step 210：两者均为 1.0；
- step 220：立即降至 0.50/0.25；
- step 330：两者均为 0；
- step 380：0.50/0.50。

这些剧烈跳变主要来自样本数只有 4，而不是足够可信的策略突变。单次 eval 的标准误非常大，不能据此认定 step 200 或 step 210 就是最佳模型。

此外，eval 设置了 `ignore_terminations=true`，所有轨迹都会运行到 80 step：

- `success_once` 表示 80 step 内曾经成功；
- `success_at_end` 表示第 80 step 结束时仍处于成功状态。

两者长期存在明显差距，说明策略经常短暂完成任务、随后又破坏成功状态。但训练环境成功后会正常终止并 reset，因此它没有被训练去维持成功状态。如果实际目标只要求“曾经放置成功”，应以 `success_once` 为主；如果要求最终稳定放置，那么当前训练目标与评估目标不完全一致，需要调整终止或奖励设计。

## 6. PPO 与 critic 是否合理

### 正常部分

- advantage mean 基本保持在 0 附近，符合开启 advantage normalization 的预期。
- critic explained variance 从初期接近 0、偶尔为负，上升并稳定在约 0.5–0.65，value head 学习有效。
- value loss 大致保持在 0.1–0.18，没有持续爆炸。
- actor ratio 均值约 0.96–0.98，没有出现明显异常值或数值失稳。
- 普通训练 step 平均约 131 秒；每 10 step 的 eval 额外约 66.6 秒，总体运行速度稳定。

### 风险部分

- 最近 approximate KL 均值约 0.085，明显高于早期的 0.049。
- 最近 clip fraction 约 0.34，表示约三分之一有效样本的 policy loss 被 PPO clipping。
- 日志中的 grad norm 常见 15–25，而配置阈值是 1.0；该指标记录的是裁剪前范数，因此当前几乎每次 optimizer step 都在强裁剪。
- `update_epoch=5`、actor LR `7.91e-6`，同时没有基于 target KL 的 early stop；对已经较强的初始化策略，这组设置可能导致过度更新。
- train success 的下降与 KL/clip fraction 的上升方向一致，支持“policy drift”判断，但仅凭现有日志不能证明唯一因果关系。

## 7. 建议

### 恢复与 checkpoint 选择

1. 若优先保留训练进度，可从 `global_step_350` 恢复；step 351–386 没有 checkpoint，无法精确续上。
2. 不要默认最新 checkpoint 就是最佳模型。建议对 `global_step_100/150/200/250/300/350` 使用同一组更大的固定 eval 集重新评估。
3. 现有单次记录中 step 200 checkpoint 的 eval 最好，但样本只有 4，必须复测后才能用于模型选择。

### 继续训练前

1. 将 eval `total_num_envs` 从 4 提高到至少 25；如果显存允许，建议 50 或更多。25 个样本时成功率粒度为 4%，比当前 25% 更可用。
2. 重点监控 20–50 step 滑动均值，而不是单 step：train/eval success、return、approx KL、clip fraction、critic EV。
3. 若复测确认 step 200–350 已出现退化，优先尝试以下两项中的一项，避免同时修改后无法归因：
   - 将 actor LR 降到约 `3e-6`–`4e-6`；或
   - 将 `update_epoch` 从 5 降到 2–3。
4. 为 PPO 增加 target-KL 监控或 early stop；当前 `kl_beta=0`，没有显式约束策略相对参考模型的漂移。
5. 如果最终稳定放置是目标，应让训练奖励覆盖成功后的稳定性，或让训练同样忽略成功终止并对最终状态给奖励；否则 `success_at_end` 很难稳定提高。

## 8. 最终判断

- **训练流程是否正常：**正常，未见数值故障，critic 学习有效，吞吐稳定。
- **训练结果是否合理：**指标变化在 PPO 训练中可以解释，但相对强初始化模型出现了可见退化，不能认为当前训练已经取得正收益。
- **是否应该继续：**可以从 step 350 恢复，但应先扩大 eval、横向评估已有 checkpoint；若退化得到确认，再降低 actor 更新强度后继续。
