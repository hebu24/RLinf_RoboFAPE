> **Status: COMPLETED 2026-07-28.** Code (all 5 plan sections) + 6 unit-test files implemented; peg-insertion config flipped to staleness_filter_mode=chunk_mask + actor_channel_keyed_routing=true; 4-GPU smoke validated 3 clean global steps — staleness metrics present at every step, masking kicked in at step 3 (staleness_masked_chunks 15/25, staleness_masked_fraction 0.0625/0.1116), no NCCL timeout, no "Unsupported object type" cascade, all 4 ranks synchronized by the two-phase all_reduce(MIN) collective. Root-cause fixes added during completion: a metric_utils->algorithms circular import (lazy-imported) and an ND-shape bug in staleness_mask (real versions is [n_chunk,B,...] 4D, not [n_chunk,B,1] — reduced to per-chunk-step version via min over trailing dims). See the "Async-PPO completed-episode staleness fix" section in AGENTS.md.

# 修复 Async PPO Completed-Episode Staleness 死锁

## Summary

- 修复 completed episode 跨 rollout 保留多个 policy version 后，被 actor 按整条 trajectory 的 `versions.min()` 静默丢弃，导致部分 rank 无数据、其他 rank 进入 FSDP collective 的死锁。
- 保留完整 episode 用于 Robometer reward 重建和 GAE；staleness 改为 chunk 级 mask，过期 chunk 不参与 policy/value loss。
- 使用确定性 actor channel 路由、全 rank readiness 协议和有界 completed-episode 队列，消除 rank 饥饿和旧 episode backlog。

## Implementation Changes

### 1. Chunk 级 Staleness Mask

- 新增 `algorithm.staleness_filter_mode`：
  - 默认 `trajectory`，保持其他 async PPO 配置兼容。
  - Peg-insertion Robometer 配置设为 `chunk_mask`。
- `chunk_mask` 模式下，actor 接收线程禁止按 `versions.min()` 丢弃 trajectory，只负责接收并入队。
- 根据原始 low-level `loss_mask` 计算 freshness：
  - cutoff 为 `actor_version - staleness_threshold`。
  - policy chunk version 必须满足 `version >= cutoff`。
  - pickup、padding 及原始 `loss_mask=False` 的位置不参与 version min/max。
  - 同一 policy chunk 内所有 action version 应一致，否则直接报错。
- 保存两个 mask：
  - `effective_low_level_loss_mask = original_low_level_mask & freshness_mask`
  - `effective_chunk_loss_mask = effective_low_level_loss_mask.any(-1, keepdim=True)`
- GAE 使用完整 episode 的 rewards、dones 和 values，但 reward 聚合使用 `effective_low_level_loss_mask`。
- Policy/value/entropy loss 使用 `effective_chunk_loss_mask`。
- 重新根据有效 low-level step 数计算 `loss_mask_sum`，禁止沿用 staleness 屏蔽前的计数。
- PPO reward 日志改用最终 effective mask，确保 `rollout/reward_*` 表示真正进入训练的数据。

### 2. Actor 全局 Readiness 协议

- 接收线程捕获所有异常，保存异常状态并通知主协程；禁止 daemon thread 静默退出。
- 接收日志必须包含 rank、actor version、trajectory 有效 version min/mean/max、有效 chunk 数和 channel queue 大小。
- PriorityStore 增加：
  - `peek_topn(n)`：查看候选但不标记使用。
  - `take_topn(n)`：取出并移除已接受候选。
  - `discard_topn(n)`：全局重收时移除当前候选。
- `chunk_mask` 模式不再调用 trajectory-level `remove_below()`。
- `_wait_for_rollout_store_ready()` 使用两阶段 collective：
  1. `all_reduce(MIN)` 确认所有 actor rank 都有候选 trajectory。
  2. `all_reduce(MIN)` 确认所有 rank 的候选都至少包含一个有效 fresh chunk。
- 若任一 rank 没有候选，其余 rank 保留候选并共同等待。
- 若所有 rank 都有候选，但任一 rank 有效 chunk 数为 0，则所有 rank 同步 `discard_topn()`，共同等待下一轮。
- 全局 readiness 成功后使用 `take_topn()`，保证 completed episode 每个 global step 最多训练一次。
- readiness collective 已形成同步点，删除随后独立的 `torch.distributed.barrier()`。

### 3. 超时和故障诊断

- 新增配置：
  - `algorithm.rollout_store_wait_timeout_s: 600`
  - `algorithm.rollout_store_status_interval_s: 30`
- 等待期间每 30 秒记录：
  - actor rank/version
  - receive thread 是否存活
  - received/queued/store trajectory 数
  - candidate version 范围
  - original/fresh/stale chunk 数
  - global retry 次数
- 超时时通过 `all_reduce(MAX)` 同步 abort 状态，所有 rank 在同一 readiness 循环中退出并抛出明确异常。
- 新增 rollout 指标：
  - `rollout/staleness_received_trajectories`
  - `rollout/staleness_masked_chunks`
  - `rollout/staleness_masked_fraction`
  - `rollout/staleness_effective_chunks`
  - `rollout/staleness_global_retry_rounds`
  - `rollout/staleness_wait_seconds`
  - `rollout/staleness_version_min/mean/max`

### 4. 确定性 Trajectory 路由

- Peg-insertion 配置启用 `algorithm.actor_channel_keyed_routing: true`。
- Env worker 使用 `CommMapper.get_dst_ranks()` 计算每个 stage 到 actor rank 的分片。
- 每个分片通过 `CommMapper.build_channel_key(actor_rank, actor_rank, "async_actor")` 写入专属队列。
- Actor rank 仅从自己的 channel key 接收，禁止多个 rank 竞争默认队列。
- 当前 4 env worker / 4 actor rank 配置下，env rank 与 actor rank 一一对应。
- 同时覆盖 env/actor world size 不相等的 M:N 分片，不硬编码 4 卡拓扑。

### 5. Completed Episode 队列有界化

- `CompletedEpisodeBuffer` 将单个全局 ready FIFO 改为 per-env ready slot。
- 同一 env 在一个收集窗口内完成多条 episode 时，只保留最新一条。
- 被更新 episode 替换的旧 episode 记入 `superseded_completed_episodes`，不再进入 actor channel。
- 每个训练 batch 按 env id 顺序各取一条，共 `train_num_envs_per_stage` 条，保证 env 贡献均衡。
- 若完整 `max_episode_steps` 收集窗口结束后某个 env 仍没有 completed episode，直接报错并输出 pending 长度、done 记录和 reward assignment 数。
- 新增环境指标：
  - `env/reward/superseded_completed_episodes`
  - `env/reward/selected_episode_version_min`
  - `env/reward/selected_episode_version_max`
  - `env/reward/selected_episode_wait_rollouts`
- 保留现有 pending、completed、cross-rollout、valid/padding chunk 指标。

## Test Plan

- 构造 versions `[8, 8, 9, 10]`、actor version 10、threshold 1，验证仅 version 9/10 chunk 保留。
- 构造 pickup version 0 但原始 mask 为 False，验证它不影响 trajectory freshness。
- 验证 effective low-level mask 用于 reward 聚合和日志，effective chunk mask 用于 policy/value loss。
- 验证 GAE 仍遍历完整 terminal episode，stale 前缀不参与 loss，但 fresh 后缀获得正确终局 return。
- 模拟四个 actor rank：
  - 一个 rank 尚无候选时，任何 rank 都不进入训练。
  - 一个 rank 候选全 stale 时，所有 rank 同步丢弃并重收。
  - 所有 rank 有 fresh chunk 时，同步取出 batch。
  - 接收线程异常或等待超时时，所有 rank 同步失败并打印状态。
- 验证 keyed routing 的 4→4、2→4 和 4→2 映射，确保每个 actor key 收到预期 batch。
- 同一 env 一轮完成多条 episode 时，验证只保留最新一条并正确记录 superseded 数。
- 运行现有 advantage、reward metrics、CompletedEpisodeBuffer、Robometer 和 PriorityStore 单测。
- 运行 4 GPU 小规模 async smoke，至少训练 3 个 global step；注入 rank 3 延迟和旧 version trajectory，要求无 NCCL timeout、所有 rank global step 一致且 staleness 指标符合预期。

## Assumptions

- 保持 `staleness_threshold: 1`，不通过扩大阈值掩盖问题。
- 只为 peg-insertion Robometer 配置启用 `chunk_mask` 和 keyed routing；其他 async PPO 配置默认保持原有 trajectory-level 行为。
- Completed episode 必须完整保留用于 reward reconstruction 和 GAE，不允许在 episode buffer 内截断旧前缀。
- 同一 policy chunk 内的 version 按 rollout 实现应完全一致，不一致视为数据损坏。
- 本次不修改 Robometer 服务、Ray 部署、FSDP 参数或 GPU 拓扑。
