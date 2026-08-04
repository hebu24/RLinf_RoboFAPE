Async PPO 独立 Rollout Window 改造实施方案
1. 目标与精确定义
为 PegInsertion + Async PPO + Robometer 增加显式 opt-in 模式：
yml


Apply
env:
  train:
    rollout_window_mode: independent
开启后，一个训练 rollout window 的语义：
window 开始前，所有训练环境均显式 reset()；不得使用上一 window 的 last_obs、仿真状态、history 或 auto-reset 后的尾部 episode。
window 内保留 auto_reset: true：自然结束的 episode 仍按当前逻辑记录、奖励回填、自动 reset。
window 最后一个 action chunk 执行后，仍未结束的 episode 被人工结束为失败轨迹：
在训练轨迹 bootstrap 边界设置 done=True、truncation=True；
不改写已自然结束的 termination/truncation；
Robometer 使用截至边界的完整 history 回填失败 episode 的 reward；
GAE 在该边界停止，不能 bootstrap 到下一 window。
reward 回填、trajectory 发送 actor 后，统一 reset 所有环境，并清空/初始化 history。
下一 window 从新的 reset observation 开始，不得 resume 上一 window。
该方案只把窗口末尾尚未完成的 episode 标记为 synthetic timeout/failure，绝不把自然成功改成失败。
2. 为什么不能只设 auto_reset: false
不建议用 auto_reset: false 实现。
原因：
早结束环境仍会在后续采样 loop 中继续被 step，可能违反环境生命周期或产生不合法轨迹；
会浪费同一 window 剩余时间；
不支持一个环境在 window 内自然完成后安全开始新 episode；
会让现有 async EnvWorker 的 rollout/observation 通信逻辑变复杂。
应保持：
yml


Apply
env:
  train:
    auto_reset: true
并在 window 边界由 EnvWorker 统一人工截断与 reset。
3. 涉及文件
核心修改：
rlinf/workers/env/env_worker.py
rlinf/workers/env/history_manager.py（仅当缺少清空 history 的公开接口时）
rlinf/envs/maniskill/maniskill_env.py（可选，增加具名 reset 包装）
examples/embodiment/config/env/maniskill_peg_insertion_vertical_rl.yaml
当前实际训练的 peg-insertion async PPO config
tests/unit_tests/test_env_worker_rollout_window_reward_assignment.py
tests/unit_tests/test_delta_async_batch_config.py
不修改：
PPO loss；
critic warmup；
action formatting；
eval 环境逻辑；
CompletedEpisodeBuffer 的跨窗口逻辑。
独立 window 模式不应该使用：
yml


Apply
reward:
  history_train_mode: complete_episode
因为 CompletedEpisodeBuffer 天然会保留跨 rollout 的 completed episode，这与“当前训练 batch 严格只属于当前 window”的目标冲突。
4. 当前关键数据流
rlinf/workers/env/env_worker.py::_run_interact_once 是主入口。
关键位置：
初始化本次 rollout buffer：
py


Apply
self.rollout_results = [...]
self._reset_window_chunk_refs()
action chunk 主循环，大约在 1918-2031：
从 rollout worker 接收动作；
调用 env_interact_step()；
将环境输出存进 env_outputs[stage_id]；
下一轮使用该输出构造训练 chunk。
post-loop bootstrap，大约在 2032-2098：
使用最后的 env_outputs[stage_id] 查询 Robometer；
接收 bootstrap values；
把 dones / truncations / terminations 追加到 trajectory 的 T+1 边界；
用 assign_history_reward() 回填当前 window 内已有 action chunk 的 reward。
现有跨 window resume 来源：
bootstrap_step() 在 auto_reset=true 时直接使用 self.last_obs_list；
当前 window 结束调用 store_last_obs_and_intervened_info(env_outputs)；
下一次 rollout 因而从旧环境状态继续。
因此，必须：
在 action loop 结束、post-loop reward query 之前，对最后 env_output 注入 synthetic terminal；
在 reward 回填完成、保存 last_obs_list 之前，reset 全部训练 env；
让 last_obs_list 保存 reset 后的新 observation，而不是旧 env_outputs。
5. 配置改动
5.1 环境默认配置
在 examples/embodiment/config/env/maniskill_peg_insertion_vertical_rl.yaml 增加：
yml


Apply
# continuous: legacy cross-window continuation behavior.
# independent: force-close unfinished episodes and reset all train envs
# after every rollout window.
rollout_window_mode: continuous
保持默认 continuous，避免影响其它实验。
5.2 实际独立训练配置
在要重跑的实验 YAML 中覆盖：
yml


Apply
env:
  train:
    auto_reset: true
    rollout_window_mode: independent
保留：
yml


Apply
reward:
  reward_mode: history_buffer
  history_train_mode: rollout_window
  history_reward_assign: true
如果当前训练是 delta 配置，则建议单独创建明确命名的 config，例如：
plain text


Apply
maniskill_async_ppo_peg_insertion_pi05_delta_independent_window.yaml
日志目录也必须包含 independent_window，避免与 legacy 连续窗口实验混淆。
6. EnvWorker.__init__：解析与校验模式
在 train_env_cfg 已获得、self.rollout_epoch 附近加入：
py


Apply
rollout_window_mode = (
    train_env_cfg.get("rollout_window_mode", "continuous")
    if train_env_cfg is not None
    else "continuous"
)
if rollout_window_mode not in {"continuous", "independent"}:
    raise ValueError(
        "env.train.rollout_window_mode must be 'continuous' or "
        f"'independent', got {rollout_window_mode!r}."
    )

self.independent_rollout_windows = rollout_window_mode == "independent"
展开
开启 independent 时增加校验：
py


Apply
if self.independent_rollout_windows:
    if not train_env_cfg.auto_reset:
        raise ValueError(
            "env.train.rollout_window_mode='independent' requires "
            "env.train.auto_reset=true."
        )
    if self.history_train_mode == "complete_episode":
        raise ValueError(
            "Independent rollout windows require "
            "reward.history_train_mode='rollout_window'; "
            "'complete_episode' retains trajectories across windows."
        )
展开
当前 peg insertion 配置 rollout_epoch: 1，建议也显式限制：
py


Apply
if self.independent_rollout_windows and self.rollout_epoch != 1:
    raise ValueError(
        "Independent rollout windows currently require rollout_epoch=1."
    )
或者将每一个 internal epoch 都当成独立 window；前者更安全。
7. EnvWorker：强制关闭 window 边界
在 record_env_metrics() 后新增 helper：
py


Apply
def _finalize_independent_window_boundary(
    self,
    env_output: EnvOutput,
    stage_id: int,
    env_metrics: dict[str, list],
) -> EnvOutput:
    """Force unfinished env slots to end at an independent window boundary."""
建议实现：
py


Apply
def _finalize_independent_window_boundary(
    self,
    env_output: EnvOutput,
    stage_id: int,
    env_metrics: dict[str, list],
) -> EnvOutput:
    if env_output.dones is None:
        raise RuntimeError("Cannot finalize rollout window without done flags.")

    dones = env_output.dones.clone().to(dtype=torch.bool)
    terminations = (
        env_output.terminations.clone().to(dtype=torch.bool)
        if env_output.terminations is not None
        else torch.zeros_like(dones)
    )
    truncations = (
        env_output.truncations.clone().to(dtype=torch.bool)
        if env_output.truncations is not None
        else torch.zeros_like(dones)
    )

    naturally_done = dones[:, -1].clone()
    forced_timeout = ~naturally_done

    # Artificial boundary is modeled as a truncation, not a task termination.
    dones[:, -1] = True
    truncations[:, -1] = torch.logical_or(truncations[:, -1], forced_timeout)

    env_metrics["window/episodes"].append(
        torch.tensor([dones.shape[0]], dtype=torch.float32)
    )
    env_metrics["window/natural_terminal_episodes"].append(
        torch.tensor([naturally_done.sum().item()], dtype=torch.float32)
    )
    env_metrics["window/forced_timeout_episodes"].append(
        torch.tensor([forced_timeout.sum().item()], dtype=torch.float32)
    )
    env_metrics["window/forced_timeout_fraction"].append(
        forced_timeout.float().mean().reshape(1).cpu()
    )

    return EnvOutput(
        obs=env_output.obs,
        final_obs=env_output.final_obs,
        rewards=env_output.rewards,
        env_infos=env_output.env_infos,
        dones=dones,
        terminations=terminations,
        truncations=truncations,
        intervene_actions=env_output.intervene_actions,
        intervene_flags=env_output.intervene_flags,
    )
展开
注意事项：
一定要 clone，不能原地修改环境拥有的 dones tensor；
只修改最后一个时间步 [:, -1]；
synthetic completion 用 truncation=True 而不是 termination=True；
保持自然结束 env 的原始 flags；
不额外创建 action chunk；
不要将 synthetic done 填进最后一个 action chunk；应只位于 trajectory 的 T+1 bootstrap 边界。
8. 在正确的时机调用 synthetic boundary
在 _run_interact_once 中，action loop 完成后、post-loop 开始前插入：
py


Apply
if self.independent_rollout_windows:
    for stage_id in range(self.stage_num):
        env_outputs[stage_id] = self._finalize_independent_window_boundary(
            env_outputs[stage_id],
            stage_id,
            env_metrics,
        )
位置必须在当前结构的这段之前：
py


Apply
for stage_id in range(self.stage_num):
    env_output = env_outputs[stage_id]
    ...
    reward_model_output = self.get_reward_model_output(...)
原因：
get_reward_model_output() 会依赖 env_output.dones 决定哪些 history 应提交给 Robometer；
强制未完成 env 的 done=True 后，Robometer 会结算它们；
post-loop 的 ChunkStepResult 会保存 synthetic terminal flag；
GAE 会在 synthetic boundary 停止，避免使用下一 window 的 value bootstrap。
在 independent mode 下，post-loop reward query 应视为当前 window 最终结算：
py


Apply
last_run = (
    self.independent_rollout_windows
    or epoch == self.rollout_epoch - 1
)
然后：
py


Apply
reward_model_output = self.get_reward_model_output(
    env_output,
    send_channel=reward_channel,
    recv_channel=input_channel,
    stage_id=stage_id,
    last_run=last_run,
)
9. Robometer reward 与 history 结算
9.1 需要保证的语义
对每个 forced timeout env：
history 截止于最后实际执行 action；
success_trace 的最后结果必须为 False；
该 env 必须被 reward query 发给 Robometer；
assign_history_reward() 必须把 reward scatter 到当前 window 的已存在 chunk refs；
delta mode 必须走现有 failure_terminal_penalty；
absolute mode 必须走现有 failure handling / fail_shift。
不要为了 synthetic timeout 另写一个 reward path。
9.2 get_reward_model_output() 适配建议
已有函数在 last_run=True 时支持查询 unfinished prefixes；已有单测：
plain text


Apply
test_get_reward_model_output_queries_unfinished_prefixes_at_window_end
建议给函数增加一个显式参数，避免只依赖隐式 last_run：
py


Apply
def get_reward_model_output(
    self,
    env_output: EnvOutput,
    send_channel: Channel | None,
    recv_channel: Channel | None,
    stage_id: int,
    last_run: bool = False,
    force_finalize_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
emit mask 应表达为：
py


Apply
emit_mask = natural_done_mask | force_finalize_mask | (
    last_run & has_window_chunk
)
实际实现要保证类型和 batch 对齐。调用时：
py


Apply
force_finalize_mask = (
    ~original_dones[:, -1]
    if self.independent_rollout_windows
    else None
)
如果 helper 内未保留 original mask，可以让 _finalize_independent_window_boundary() 返回 (env_output, forced_timeout)，或者把 forced_timeout 按 stage 放在：
py


Apply
self._independent_window_forced_timeout_masks[stage_id]
推荐显式保存 mask，以便：
reward query；
metrics；
post-condition assertions；
debug logs。
9.3 必要断言
在 independent mode 下，reward assignment 完成后检查每个 forced env：
py


Apply
forced_env_ids = torch.nonzero(forced_timeout, as_tuple=False).flatten().tolist()
要求：
env id 出现在 self._last_history_query_info[stage_id]；
assign_history_reward() 返回的 assignments 中存在该 env id；
assignments[env_id].episode_success is False；
当前 window 的所有 chunk refs 均被该 assignment 覆盖。
不满足时抛出 RuntimeError，错误信息包含：
plain text


Apply
stage_id
env_id
episode_id
history_len
pickup_count
window_chunk_refs
forced_timeout
禁止静默丢弃未完成轨迹。
10. 强制 reset 与 history 清空
10.1 Worker helper
新增：
py


Apply
def _reset_train_stage_for_next_independent_window(
    self,
    stage_id: int,
) -> None:
推荐流程：
py


Apply
def _reset_train_stage_for_next_independent_window(self, stage_id: int) -> None:
    env = self.env_list[stage_id]
    env.is_start = True
    extracted_obs, _ = env.reset()

    if self.reward_mode == "history_buffer":
        history_manager = self.train_history_managers[stage_id]
        history_manager.reset_all()

        consume_pickup_frames = get_env_attr(env, "consume_pickup_frames")
        pickup_frames = (
            consume_pickup_frames()
            if callable(consume_pickup_frames)
            else (consume_pickup_frames or {})
        )
        for env_id, frames in pickup_frames.items():
            if frames:
                history_manager.prepend_history_entries(int(env_id), frames)

    self.last_obs_list[stage_id] = extracted_obs
    self.last_intervened_info_list[stage_id] = (None, None)
展开
关键要求：
reset 必须在 Robometer query 与 reward assignment 之后；
reset 后才清空 history；
last_obs_list 必须写入 reset 后 observation；
last_intervened_info_list 清空；
不能再调用 store_last_obs_and_intervened_info(env_outputs) 覆盖 reset 后的新状态。
10.2 替代原来的保存状态逻辑
当前代码接近 window 末尾有：
py


Apply
self.store_last_obs_and_intervened_info(env_outputs)
self.finish_rollout()
改为：
py


Apply
if self.independent_rollout_windows:
    for stage_id in range(self.stage_num):
        self._reset_train_stage_for_next_independent_window(stage_id)
else:
    self.store_last_obs_and_intervened_info(env_outputs)

self.finish_rollout()
下一 window 的 bootstrap_step() 仍从 self.last_obs_list 获取数据，但此时内容已经是 fresh reset observation，因此不再 resume。
11. HistoryManager：新增 reset_all()
检查 rlinf/workers/env/history_manager.py 是否已有公开 reset 接口。
如果没有，新增：
py


Apply
def reset_all(self) -> None:
    """Clear all per-environment histories before a new independent rollout window."""
该函数应清空或恢复为构造态：
py


Apply
self.history_entries
self.success_history_entries
self.pickup_counts
以及所有与 episode/history 对齐的 bookkeeping。
要求：
不能在 EnvWorker 中散落地直接修改 manager 私有成员；
不应重建 HistoryManager 对象；
reset 后由 consume_pickup_frames() + prepend_history_entries() 重新建立当前 reset 的 pickup prefix；
下一 window 的 Robometer 视频不得包含上一个 window 的任何 frame。
12. ManiskillEnv 的最小改动
不应在 ManiskillEnv.chunk_step() 内伪造 done 或截断。
原因：独立 window 是 runner/worker 采样语义，不是环境动力学语义。
如果希望让 worker 调用意图更清晰，可在 rlinf/envs/maniskill/maniskill_env.py 增加一个轻量包装：
py


Apply
def reset_all_for_rollout_window(self):
    """Reset all vectorized environments for an independent rollout window."""
    self.is_start = True
    return self.reset()
然后 worker 调用：
py


Apply
extracted_obs, _ = env.reset_all_for_rollout_window()
如果不增加该接口，直接 env.is_start = True; env.reset() 也可。
不要在这里调用 update_reset_state_ids()，除非实验设计明确要求每个 window 采样新的固定 episode state。现有：
yml


Apply
shared_reset_seed: true
use_fixed_reset_state_ids: true
有其既定复现语义，不能在本改造中擅自改变。
13. 预取 bootstrap 的风险
检查：
py


Apply
prefetch_train_bootstrap()
_bootstrap_and_send_train()
_prefetched_train_bootstrap
independent mode 下，不允许预取缓存上一 window state。
最安全的方式：
py


Apply
def prefetch_train_bootstrap(self, rollout_channel: Channel) -> None:
    if self.independent_rollout_windows:
        return
    ...
前提是 runner 不依赖该方法一定产生缓存。
如果 runner 必须依赖 prefetch，则必须先 reset，再 prefetch，且用一个 bool 防止双 reset：
py


Apply
self._independent_window_reset_ready
避免以下错误序列：
window 结束；
prefetch 旧 state；
reset；
下一个 window 消费 prefetch 的旧 state。
14. 测试改动
主要扩展：
plain text


Apply
tests/unit_tests/test_env_worker_rollout_window_reward_assignment.py
14.1 Synthetic boundary flags
构造两个 env：
py


Apply
dones = [[False, True], [False, False]]
terminations = [[False, True], [False, False]]
truncations = [[False, False], [False, False]]
断言：
env 0 保持自然终止；
env 1 最后变成：
dones[-1] == True
truncations[-1] == True
terminations[-1] == False
earlier chunk flags 未改变；
metric：
window/episodes == 2
window/natural_terminal_episodes == 1
window/forced_timeout_episodes == 1
window/forced_timeout_fraction == 0.5
14.2 GAE 不跨窗口 bootstrap
构造末尾 synthetic done 的 trajectory，传入不同的下一 window bootstrap value。
验证：
forced timeout env 的最后一个 action return/advantage 不随下一 window bootstrap value 改变；
legacy continuous mode 不改变现有 GAE 行为。
可通过 calculate_adv_and_returns() 或已有 GAE helper 做最小单测。
14.3 Robometer query 覆盖 forced timeout
mock HistoryManager 与 reward channel。
断言：
forced timeout env 出现在 _last_history_query_info[stage_id]；
自然 done env 仍按原逻辑查询；
自动 reset 后没有当前 window chunk ref 的 tail 不会被错误查询；
query 的 success trace 对 forced env 为失败结尾。
14.4 Delta failure terminal penalty
mock：
py


Apply
reconstruct_robometer_delta_reward
验证它收到：
py


Apply
failure_terminal_penalty == cfg.reward.delta.failure_terminal_penalty
并且 force timeout assignment：
py


Apply
assignment.episode_success is False
absolute mode 同样确认走 failure path，而不是成功 bonus。
14.5 Reset 时序
使用 fake env / fake history manager 记录调用顺序：
plain text


Apply
get_reward_model_output
assign_history_reward
history_manager.reset_all
env.reset
consume_pickup_frames
prepend_history_entries
断言：
reward query 与 assignment 都发生在 reset 前；
reset 后 last_obs_list 是 fresh observation；
history 只保留新 reset 的 pickup frames；
旧 history 不存在。
14.6 Prefetch guard
independent mode 下调用：
py


Apply
prefetch_train_bootstrap()
断言不产生：
py


Apply
self._prefetched_train_bootstrap
或者断言它是 fresh reset 后的 observation，不能是旧 state。
14.7 Config 校验
新增单测：
independent + auto_reset=false 抛异常；
independent + history_train_mode=complete_episode 抛异常；
continuous 保持 legacy 行为。
15. 更新 delta config 单测
更新：
plain text


Apply
tests/unit_tests/test_delta_async_batch_config.py
对于独立配置，增加：
py


Apply
assert cfg.env.train.rollout_window_mode == "independent"
assert bool(cfg.env.train.auto_reset)
assert cfg.reward.history_train_mode == "rollout_window"
原本的函数名：
py


Apply
test_delta_async_update_uses_16_completed_episodes_and_one_optimizer_step
建议改名为：
py


Apply
test_delta_async_update_uses_16_window_local_episodes_and_one_optimizer_step
原因：在 independent mode，16 个 env slot 在每个 window 中最终都形成一条本 window 内结算的轨迹；不应暗示每条都是自然 completed episode。
保留：
py


Apply
episodes_per_update == 16
但将其解释为：
plain text


Apply
每个训练更新使用 16 条 window-local finalized episodes；
其中一部分可以是 natural completion，另一部分可以是 synthetic timeout failure。
16. 运行测试
使用项目虚拟环境：
sh


Apply
/data/yingxi/kairan/envs/rlinf/bin/python -m pytest \
  tests/unit_tests/test_env_worker_rollout_window_reward_assignment.py \
  tests/unit_tests/test_delta_async_batch_config.py \
  tests/unit_tests/test_staleness_mask.py \
  tests/unit_tests/test_completed_episode_buffer.py \
  -q
静态检查：
sh


Apply
/data/yingxi/kairan/envs/rlinf/bin/python -m ruff check \
  rlinf/workers/env/env_worker.py \
  rlinf/envs/maniskill/maniskill_env.py \
  rlinf/workers/env/history_manager.py \
  tests/unit_tests/test_env_worker_rollout_window_reward_assignment.py \
  tests/unit_tests/test_delta_async_batch_config.py
如项目有 format check：
sh


Apply
/data/yingxi/kairan/envs/rlinf/bin/python -m ruff format --check \
  rlinf/workers/env/env_worker.py \
  rlinf/envs/maniskill/maniskill_env.py \
  rlinf/workers/env/history_manager.py \
  tests/unit_tests/test_env_worker_rollout_window_reward_assignment.py \
  tests/unit_tests/test_delta_async_batch_config.py
17. Smoke 训练的验收标准
先运行短 smoke：2 或 4 个 env、1 至 3 个 rollout windows。
应新增并观察：
plain text


Apply
window/episodes
window/natural_terminal_episodes
window/forced_timeout_episodes
window/forced_timeout_fraction
每个 worker/stage 必须满足：
plain text


Apply
window/natural_terminal_episodes
+ window/forced_timeout_episodes
== window/episodes
并验证：
plain text


Apply
reward/cross_rollout_episodes == 0
独立模式不应使用 CompletedEpisodeBuffer，因此其相关 pending/cross-rollout 指标应该为 0 或不产生。
此外，调试日志建议对每个 env 写入：
plain text


Apply
window_id
stage_id
env_id
episode_id
forced_timeout
natural_done
history_len
pickup_count
assignment_success
必须确认相邻两个 window 之间：
没有 history frame 延续；
没有上一 window episode id 被继续 assignment；
没有通过 last_obs_list resume 旧仿真状态。
18. 正式重跑时的 Ray 隔离要求
启动前：
检查当前训练、eval、Ray cluster 和 GPU 占用；
禁止执行裸 ray stop；
新训练设置独立：
GCS port；
dashboard agent listen port；
Ray temp directory；
日志目录；
与并行任务不重叠的 GPU 集合。
遵守仓库 RAY_ISOLATION.md。
日志目录应包含：
plain text

Apply
independent_window
例如：
plain text

Apply
peg_insertion_rl_async_delta_independent_window
正式训练先观察：
plain text

Apply
window/forced_timeout_fraction
env/success_once
reward_model_output
rollout/reward funnel 指标
每个 actor rank 的 receive/send 数量
FSDP collective 是否正常
只有确认 reward assignment 覆盖所有 forced timeout、两 actor rank 正常消费等量数据后，才扩大运行时长。
19. 非目标与安全约束
本改动不应：
改变 eval 的 episode success 统计；
改变 critic warmup；
改变 actor/PPO loss；
改变模型输出 action 的格式；
把 synthetic timeout 伪装成任务自然失败 termination；
修改 CompletedEpisodeBuffer 以尝试“兼容”独立模式；
静默丢弃未完成 episode；
静默覆盖用户 YAML 中的 rollout_window_mode。