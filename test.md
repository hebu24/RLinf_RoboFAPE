# Peg Insertion Async PPO: 16 Environments / Batch 960

跨机器运行 absolute Robometer reward 实验的启动记录。

## 目标配置

- `reward.shaping=absolute`（由 `maniskill_async_ppo_peg_insertion_pi05` 默认配置提供）
- 两张训练卡：由 `CUDA_VISIBLE_DEVICES` 指定
- `env.train.total_num_envs=16`
- `actor.global_batch_size=960`
- `actor.micro_batch_size=8`
- `actor.optim.lr=1e-6`
- 两个 FSDP actor rank；每个 rank 负责 8 个环境。
- 每轮每环境最多 60 个 action chunk，因此总采样容量为 `16 × 60 = 960` chunks，正好匹配 global batch。

这不是恢复已停止的 GPU 0/1 短跑；默认从实验 YAML 所引用的 SFT checkpoint 开始训练。

## 新机器准备

1. 安装 RLinf、ManiSkill 和 OpenPI 依赖。
2. 确保 `examples/embodiment/config/maniskill_async_ppo_peg_insertion_pi05.yaml`
   中引用的 SFT checkpoint 在本机可访问；若路径不同，覆盖 actor 和 rollout
   的 `model_path`。
3. 在启动训练前启动 Robometer HTTP 服务（端口 `8000`）。奖励 worker 会在
   rollout 时访问它。给该服务设置有足够空间的 `TMPDIR`。
4. 若 RoboFAPE/ManiSkill planner 位于其他目录，修改
   `run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh`
   中的 `RLINF_ROBOFPE_PATH`。

Robometer 服务示例（替换路径和 GPU）：

```bash
cd /path/to/RoboFAC/robometer
mkdir -p /data/$USER/tmp

TMPDIR=/data/$USER/tmp \
CUDA_VISIBLE_DEVICES=<robometer_gpu> \
uv run python robometer/evals/eval_server.py \
  model_path=/path/to/robometer/checkpoint-400 \
  server_url=0.0.0.0 \
  server_port=8000 \
  num_gpus=1 \
  batch_size=4
```

## 启动训练

选择两张训练 GPU。每个并行 Ray 集群必须使用不同的：

- `RL_RAY_PORT`
- `RAY_DASHBOARD_AGENT_PORT`
- Ray 临时目录（launcher 根据 `RL_RAY_PORT` 自动导出）

不要使用裸 `ray stop`；launcher 会仅按自身 `RL_RAY_PORT` 清理对应的 Ray
进程。

```bash
cd /path/to/RLinf_RoboFAPE
mkdir -p /data/$USER/tmp

tmux new-session -d -s peg_abs_lr1e6_env16_batch960 \
  "cd /path/to/RLinf_RoboFAPE && \
  export CUDA_VISIBLE_DEVICES=<gpu0>,<gpu1> \
    RL_RAY_PORT=6384 \
    RAY_DASHBOARD_AGENT_PORT=52370 \
    CONFIG_NAME=maniskill_async_ppo_peg_insertion_pi05 \
    LOG_DIR=logs/\$(date +'%Y%m%d-%H:%M:%S')-peg_insertion_rl_async_absolute_lr1e6_env16_batch960; \
  bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
    actor.optim.lr=1e-6 \
    actor.global_batch_size=960 \
    env.train.total_num_envs=16 \
    reward.model.timeout_s=600 \
    algorithm.rollout_store_wait_timeout_s=1800; \
  rc=\$?; echo ===SESSION_EXIT=\$rc===; exec bash"
```

若新机器的仓库根目录不同，需要修改 launcher 顶部硬编码的：

- `cd`
- `REPO_PATH`
- `EMBODIED_PATH`
- `RLINF_ROBOFPE_PATH`

或者复制该 launcher 并替换成新机器路径。

## 监控

```bash
tmux attach -t peg_abs_lr1e6_env16_batch960

RAY_ADDRESS=127.0.0.1:6384 ray status
```

初始化后每个 actor rank 应接收到近似 `[60, 8, 10, 7]` 的轨迹；两 rank 合计为
960 chunks。第一次更新要等待两个 rank 都收到 completed episode trajectory，
这是 async completed-episode buffer 的正常行为。