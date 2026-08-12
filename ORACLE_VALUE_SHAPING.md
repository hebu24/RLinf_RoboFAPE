# Oracle-Value Robometer Shaping —— 用 Progress 直接做 RL Value

> 用 robometer server 输出的 per-frame progress 曲线直接当 GAE 的 value V(s_chunk)，跳过「学习型 value head」。适用于 peg-insertion async-PPO pipeline。

本文档说明：为什么这么做、数学原理、实现方式、配置、运行命令、验证、回滚、以及一个零代码 fallback。

---

## 1. 为什么需要这个

xulab 上 `/data/yingxi/RLinf_RoboFAPE` 的 async-PPO（peg-insertion）原本用「学习型 value head」做 GAE critic。从最近的 Robometer-4B run 的 metrics 看，这个 value head **基本没学到东西**：

| 指标 | 实测（rbm4b run） | 健康值 |
|---|---|---|
| `critic/explained_variance` | 0.004 – 0.23（多数 ~0.1） | > 0.5 |
| `critic/value_loss` | 0.2 – 0.55，不降 | 持续下降 |
| `reward/success_bonus_fraction` | **0.0 全程** | > 0 |

而 robometer server 输出的 per-frame progress∈[0,1]（VLM judge 对任务进度的稠密估计）是**唯一的真实信号**。进一步分析发现：

- `robometer_initial_progress` 全程 0.54–0.66（pickup 阶段视觉上已「完成」→ 基线偏高），final ≈ 0.77
- absolute-progress reward 的有效动态范围只有 **~0.15**，还带噪声、且零成功 → 绝对 reward 信号弱、糊
- **真正有意义的信号是 chunk 间的 progress 增量**（delta）

此前 `compare_rl_value.py` 的结论也印证：用 rbm4b 当 reward 去训 value，value 反而退化（与 GT progress 负相关）。

**结论**：既然学的 value 没用、progress 才是真信号，直接拿 robometer 的 progress 当 V(s)，跳过 critic。

---

## 2. 数学原理

设 Φ(s) = robometer progress（per-chunk 边界值，terminal 用 success-shift）。

**Oracle-value 模式（本方案）：**
- `V(s_chunk) = Φ(s_chunk)`（不学，直接用 robometer 输出）
- reward 只在 terminal chunk 给 `success_bonus`（成功）或 `failure_terminal_penalty`（失败），其余 chunk reward = 0
- GAE：

```
δ_t = r_t + γ·V[t+1]·(~done) - V[t]
```

- 非 terminal chunk：`δ_t ≈ γ·Φ_{t+1} − Φ_t`（≈ chunk 间 progress 增量，稠密低方差）
- terminal chunk（done）：`δ_T = r_terminal − V[T−1]`
- `A_t = Σ (γλ)^l δ_{t+l}`

这正是 **Ng et al. 1999 的 potential-based reward shaping**：shaped reward `r' = r + γΦ(s') − Φ(s)`，**不改变最优策略**（policy-invariant），但给了一个 state-dependent baseline → 比 REINFORCE 方差低。progress 本身是对任务进度的稠密估计，天然适合做 Φ。

**对比「delta reward + critic_free」模式（零代码 fallback，见 §9）：**

| | Oracle-value（本方案） | delta + critic_free（fallback） |
|---|---|---|
| advantage 公式 | `γ·prog[t+1] − prog[t]`（单步 delta） | `Σ未来 delta = 剩余 progress`（累积） |
| GAE 路径 | `else` 分支（values 非 None），标准 GAE 带 λ | `critic_free` 分支（values=None），γ=λ=1，REINFORCE |
| 方差 | O(1) per step | O(T) —— 随 episode 长度增长 |
| 偏置 | 无（oracle 是真值） | 无（REINFORCE 无偏） |

Oracle-value 形式**方差严格更低**，收敛更快。

---

## 3. 这种情况下 RL 怎么学（一句话）

PPO actor 只用 advantage 更新，advantage = robometer progress 的 TD delta（chunk 间增量）；**不训 critic、不跑 value head forward、不算 value loss**。policy 被推向「让 progress 增量变正」的动作 —— 等价于 potential-based shaping，最优策略不变，但信号稠密、低方差、且不会再退化。

---

## 4. 实现方式

复用三处已有机制：
1. delta 路径的 `_robometer_boundary_frame_indices`（chunk 边界 progress）
2. rollout batch 的 `prev_values` 字段（env_worker 侧注入，actor worker `values = prev_values`）
3. 已注册的 `loss_type: "actor"`（`losses.py:564`，只算 actor loss、不调 critic loss）

`adv_type` 保持 `gae`，GAE 的 `else` 分支（`advantages.py:61-79`）天然处理非 None 的注入 value。

### 4.1 `rlinf/models/embodiment/reward/robometer_reward_model.py`

**(a) `RobometerEpisodeReward` dataclass 加 `chunk_values` 字段**（~L270）：

```python
chunk_values: np.ndarray = None  # per-chunk oracle V(s), layout 同 chunk_reward
```

**(b) 新增 `reconstruct_robometer_oracle_value_reward()`**（在 `reconstruct_robometer_delta_reward` 之后）：

镜像 delta 函数的 boundary-frame 校验与 layout，但把 progress 从 reward 挪到 value：

- `chunk_values[i, 0] = prog[i]`（成功）或 `prog[i] − fail_shift`（失败，per-episode 常数 shift）
- `chunk_reward[i] = 0`（非 terminal）；terminal 给 `success_bonus`（成功）/ `failure_terminal_penalty`（失败）
- `loss_mask` 对每个 insertion chunk 的全部 sub-step 为 True（与 absolute/delta 一致，保证 `masked_mean_ratio` 的 scaling）
- **terminal 处理**：用固定 bonus/penalty，而非 `gamma*V` bootstrap（因为 `assign_history_reward` 会覆写 env reward，env 侧的 value-head bootstrap 不再生效；与 delta 模式一致）

### 4.2 `rlinf/workers/env/env_worker.py`

**(a) import**：加 `reconstruct_robometer_oracle_value_reward`。

**(b) `__init__` 读配置**（`reward.oracle_value.*`）：

```python
self.oracle_success_bonus = float(
    self.cfg.reward.get("oracle_value", {}).get("success_bonus", 1.0)
)
self.oracle_failure_terminal_penalty = float(
    self.cfg.reward.get("oracle_value", {}).get("failure_terminal_penalty", -0.4)
)
```

**(c) `assign_history_reward`（~L1596）**：加 `oracle_value_mode` 分支，与 delta 共用 boundary-frame 路径（`boundary_mode = delta_mode or oracle_value_mode`）：

```python
if oracle_value_mode:
    assignment = reconstruct_robometer_oracle_value_reward(
        env_progress[:expected_progress],
        history_len=history_len, pickup_count=pickup_count,
        success_trace=success_trace, chunk_size=chunk_size,
        total_chunks=total_chunks,
        success_bonus=self.oracle_success_bonus,
        failure_terminal_penalty=self.oracle_failure_terminal_penalty,
        fail_shift=fail_shift,
    )
```

**(d) prev_values 散播**（scatter chunk_reward 处之后，~L1791）——**核心改动**，把 value head 输出覆写成 oracle progress：

```python
if (oracle_value_mode
    and assignment.chunk_values is not None
    and len(self.rollout_results[stage_id].prev_values) > local_chunk_idx):
    pv_target = self.rollout_results[stage_id].prev_values[local_chunk_idx]
    pv_target[env_id] = torch.as_tensor(
        [float(assignment.chunk_values[episode_chunk_idx, 0])],
        dtype=pv_target.dtype, device=pv_target.device,
    )
```

> `prev_values` 是 per-chunk 的 `[B,1]` 张量列表（`EmbodiedRolloutResult.prev_values`），GAE 消费时 `values = prev_values`。需要 `collect_prev_infos=True`（默认 True）才会被填充。

### 4.3 `rlinf/workers/actor/async_ppo_fsdp_worker.py`

**(a) `use_oracle_value` 跳过 value-head forward**（两处）：

```python
# 训练 forward（~L1471）+ compute_post_update_outputs（~L1269）
compute_values = self.cfg.algorithm.adv_type == "gae" and not (
    self.cfg.algorithm.get("use_oracle_value", False)
)
```

- 训练时 `compute_values=False` → `out["values"]=None`；`loss_type="actor"` 不用 values
- post-update 时返回 None；`if post_update_values is not None`（L1671）已 guard，EV metric 自动跳过

**(b) `critic_free` 开关**（`compute_advantages_and_returns`，~L1067，给 fallback 用）：

```python
critic_free = self.cfg.algorithm.get("critic_free", False)
gae_values = None if critic_free else (proximal_values if proximal_values is not None else prev_values)
```

### 4.4 数据流

```
rollout model forward（默认仍跑 value head，产物被覆写）
 → prev_values = value head 输出 [B,1] per chunk
episode done → robometer 查询 boundary frames → boundary_progress
assign_history_reward(shaping=oracle_value)
 → chunk_reward = [0..0, terminal_bonus]            （per chunk）
 → chunk_values = [prog0..progN] (success-shifted)  （per chunk）
 → 散进 rollout_rewards / prev_values（覆写 value head）
actor worker compute_advantages_and_returns: values = prev_values (oracle)
 → GAE else 分支: δ_t = γ·V[t+1] − V[t]（稠密 delta）；terminal: δ = r_term − V[T−1]
训练: compute_values=False（不跑 head）；loss_type=actor 只用 advantage
post_update: 不跑 value forward，EV metric 跳过
```

---

## 5. 配置

新配置文件：`examples/embodiment/config/maniskill_async_ppo_peg_insertion_pi05_oracle_value.yaml`（基于 `_delta.yaml`，关键改动）：

```yaml
algorithm:
  adv_type: gae                  # GAE else 分支消费注入的 oracle value
  loss_type: actor               # actor-only，不算 critic loss
  use_oracle_value: True         # 跳过 value-head forward
  normalize_advantages: True     # 必须：oracle advantage ~0.01-0.05，不归一化 PPO clip 过大
  normalize_returns: False       # 无 critic，returns 只用于 log
  gamma: 0.99
  gae_lambda: 0.95

reward:
  shaping: oracle_value
  oracle_value:
    success_bonus: 1.0
    failure_terminal_penalty: -0.4
  model:
    fail_shift: 1.0              # 失败轨迹 V -= fail_shift

actor:
  model:
    add_value_head: True         # 保留：checkpoint 权重兼容；训练时不 forward
    openpi:
      detach_critic_input: True  # 保留，无害（无 critic grad）
rollout:
  collect_prev_infos: True       # 填充 prev_values 供覆写
critic:
  use_critic_model: False
```

---

## 6. 运行命令

### 前置条件检查

```bash
ssh xulab
# 1) robometer server 在跑（应返回 healthy）
curl -s http://127.0.0.1:8000/health
# 2) 根盘别满（/ 应有 >20G 空闲；历史曾 Errno 28）
df -h / /data
# 3) GPU 空闲（至少 2 张）
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

### 启动 robometer server（若未跑）

用 xulab `.venv`（Py3.10，bf16，~9GB/查询）；**别用** Py3.11 预建 env（fp32，~17GB，OOM）：

```bash
cd /home/yingxi/RoboFAC
source /data/yingxi/RLinf_RoboFAPE/run_robometer_4b_server.sh  # 或直接：
CUDA_VISIBLE_DEVICES=<空闲GPU> TMPDIR=/data/yingxi/tmp \
  /home/yingxi/RoboFAC/robometer/.venv/bin/python \
  robometer/evals/eval_server.py \
  model_path=/data/yingxi/robometer/Robometer-4B \
  server_url=0.0.0.0 server_port=8000 num_gpus=1 batch_size=4
```

### 启动 RL 训练（oracle_value 模式）

**方式 A：直接用新配置**（改 `_cmd` 脚本或现写）：

```bash
cd /data/yingxi/RLinf_RoboFAPE
export CUDA_VISIBLE_DEVICES=0,1            # 两张空闲 GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RL_RAY_PORT=6386                    # 避开已占用的 6384
export RAY_DASHBOARD_AGENT_PORT=52372
export LOG_DIR="logs/$(date +'%Y%m%d-%H:%M:%S')-peg_insertion_rl_oracle_value_gpu01"
export TMPDIR=/data/yingxi/tmp             # 根盘满，tmp 放 /data
export HF_HOME=/data/yingxi/hf_cache

bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
  reward.model.server_url=http://127.0.0.1:8000
```

> `run_peg_insertion_rl_async.sh` 会从 `reward.shaping=` 推断 config 名：设 `reward.shaping=oracle_value` 时会选 `maniskill_async_ppo_peg_insertion_pi05_oracle_value`（若脚本未自动映射，用方式 B）。

**方式 B：显式指定 config + Hydra 覆盖**：

```bash
cd /data/yingxi/RLinf_RoboFAPE
source /data/yingxi/kairan/envs/rlinf/bin/activate rlinf
export EMBODIED_PATH=/data/yingxi/RLinf_RoboFAPE/examples/embodiment
export REPO_PATH=/data/yingxi/RLinf_RoboFAPE
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1
export LOG_DIR="logs/$(date +'%Y%m%d-%H:%M:%S')-peg_insertion_rl_oracle_value_gpu01"

CUDA_VISIBLE_DEVICES=0,1 python examples/embodiment/train_async.py \
  --config-path ${EMBODIED_PATH}/config/ \
  --config-name maniskill_async_ppo_peg_insertion_pi05_oracle_value \
  reward.model.server_url=http://127.0.0.1:8000 \
  runner.logger.log_path=${LOG_DIR}
```

### 短 smoke 验证（先验能跑通）

加 `RLINF_REWARD_DEBUG=1` 跑 1-2 个 step，确认 reward/value 接线：

```bash
RLINF_REWARD_DEBUG=1 CUDA_VISIBLE_DEVICES=0,1 bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
  reward.shaping=oracle_value reward.model.server_url=http://127.0.0.1:8000 \
  runner.max_epochs=2 env.train.total_num_envs=8
```

---

## 7. 验证

### 7.1 单元测试（已通过 6/6）

```bash
cd /data/yingxi/RLinf_RoboFAPE
CUDA_VISIBLE_DEVICES= PYTHONPATH=/data/yingxi/RLinf_RoboFAPE \
  /data/yingxi/kairan/envs/rlinf/bin/python test_oracle_value.py
```

覆盖：
- success/failure case：`chunk_values` = prog / prog−fail_shift，非 terminal `chunk_reward`=0，terminal bonus 正确
- boundary parity：oracle 与 delta 用相同的 `_robometer_boundary_frame_indices`（total_chunks+1）
- validation：短 progress / 非有限 / 坏 chunk_size 都 raise
- **GAE 数值**：oracle values 走 `else` 分支，非 terminal `delta_0 = γ·prog[1]−prog[0] = 0.060`，terminal `delta_T = success_bonus − prog[T−1] = 0.067`，advantage 非 0 有限；critic_free **未**误触发

### 7.2 e2e smoke 检查清单

训练起来后看 `metrics.log`：

```bash
LOG=logs/<你的run>/metrics.log
# 应【不再出现】critic 相关（loss_type=actor）：
grep -c "critic/value_loss\|critic/explained_variance" $LOG   # 期望 0
# 应仍记录 reward/progress：
grep -oE "robometer_initial_progress=[0-9.]+" $LOG | head
grep -oE "success_bonus_fraction=[0-9.]+" $LOG | head
# advantage 非 0（量级 ~0.01-0.05，归一前）
```

`RLINF_REWARD_DEBUG=1` 的 debug 日志（`/tmp/robometer_rdebug_absolute.log`）应见：

```
[assign] env_id=0 shaping=oracle_value ... oracle_v0=0.6230 ... episode_success=...
```

确认 `prev_values` 被覆写成 robometer progress（而非 value head 输出）。

### 7.3 对照实验

同 seed 跑 oracle_value vs 现有 absolute(+gae)，比 success 曲线 / progress 增量方差 / 收敛速度。预期 oracle 方差更低、且不再有 value 退化。

---

## 8. 风险与 caveat

1. **4B VLM judge 噪声**：progress 单帧噪声大，但 delta（chunk 间差分）会抵消共模噪声，GAE λ=0.95 进一步平滑；总体仍比 EV≈0.1 的学习 head 稳。
2. **不随训练改进**：oracle 是冻结的，robometer 的系统偏置会全程存在。但学习 head 也没在改进（EV 不升），所以不亏。
3. **success-shift 用了 hindsight**（episode 结果只有结束后才知道）：此处可接受 —— reward 本来就是 post-hoc 算的（robometer 在 episode 结束后才查）；既有 absolute 模式 reward 也用同样 shift。失败轨迹的常数 shift 在非 terminal chunk 只留 `(γ−1)·shift` ≈ +0.01/chunk 的极小残差（γ=0.99 时）。
4. **per-chunk 粒度**：progress 只在 chunk 边界有（每 10 low-level step），GAE 本就在 chunk 级，粒度匹配，无需插值。
5. **oracle value 无梯度**：来自 HTTP 响应的 numpy/tensor，无计算图，actor loss 只用 advantage（也 detached）。自动隔离，无需额外 detach。
6. **+1 bootstrap boundary 未运行时验证**：env_worker 的 prev_values scatter（`[B,1]` per-chunk）和 GAE 需要的 `values[T+1]` bootstrap boundary，unit test 没覆盖运行时张量形状。若 GAE 报 shape mismatch 或 advantage 异常，问题在这里 —— bootstrap boundary 当前留给 value-head 输出，仅影响 truncated 失败轨迹的最后 chunk，`failure_terminal_penalty` 兜底。

---

## 9. 零代码 Fallback（先试可行性的话）

不想跑 oracle 接线，可用现有 `_delta.yaml` 加两个覆盖，验「progress 当信号、不训 critic」：

```bash
bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
  reward.shaping=delta \
  algorithm.critic_free=True \
  algorithm.loss_type=actor \
  reward.model.server_url=http://127.0.0.1:8000
```

- delta shaping：`chunk_reward[i] = prog[i+1] − prog[i] + success_bonus·1[success_i]`
- `critic_free=True` → GAE `critic_free` 分支（values=None，γ=λ=1）→ `A_t = 剩余 progress`（REINFORCE，方差 O(T)）
- 方差比 oracle_value 高，但**不需任何新代码**，policy 梯度方向一致

要低方差还是得上 §4 的 oracle_value。

---

## 10. 回滚

改动的三文件已备份：

```bash
cd /data/yingxi/RLinf_RoboFAPE
for f in rlinf/models/embodiment/reward/robometer_reward_model.py \
         rlinf/workers/env/env_worker.py \
         rlinf/workers/actor/async_ppo_fsdp_worker.py; do
  cp "$f.bak.oracle_value" "$f"
done
# 新增的 config + test 可直接删：
rm examples/embodiment/config/maniskill_async_ppo_peg_insertion_pi05_oracle_value.yaml
rm test_oracle_value.py
```

---

## 附：关键文件位置

| 文件 | 作用 |
|---|---|
| `rlinf/models/embodiment/reward/robometer_reward_model.py` | `reconstruct_robometer_oracle_value_reward()` + `chunk_values` 字段 |
| `rlinf/workers/env/env_worker.py` | `assign_history_reward` 的 `oracle_value` 分支 + prev_values 散播 |
| `rlinf/workers/actor/async_ppo_fsdp_worker.py` | `use_oracle_value` / `critic_free` 开关 |
| `rlinf/algorithms/advantages.py` | GAE（无需改，`else` 分支处理注入 value） |
| `rlinf/algorithms/losses.py` | `loss_type: "actor"`（无需改，已注册） |
| `examples/embodiment/config/maniskill_async_ppo_peg_insertion_pi05_oracle_value.yaml` | 新配置 |
| `test_oracle_value.py` | 单元测试 |

**环境**：训练用 `/data/yingxi/kairan/envs/rlinf/bin/python`；robometer server 用 `/home/yingxi/RoboFAC/robometer/.venv`（Py3.10）。
