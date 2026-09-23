## RoboFAPE
tmux new -s train_400_5tasks
  cd /data/yingxi/RLinf_RoboFAPE

  LOG_DIR=/data/yingxi/RLinf_RoboFAPE/logs/pushcube_rl_env32_abs30_400_5tasks_$(date +%Y%m%d_%H%M%S) \
  RL_GPU_IDS=0,1 \
  START_ROBOMETER_SERVER=false \
  ROBOMETER_REQUIRE_HEALTH=true \
  ROBOMETER_PORT=8001 \
  RL_RAY_PORT=6386 \
  RAY_DASHBOARD_PORT=8266 \
  RAY_DASHBOARD_AGENT_PORT=52372 \
  RAY_TMPDIR=/data/yingxi/ray_tmp_pushcube_rl_6386 \
  bash run_train/maniskill_robometer_rl/launch_pushcube_rl_seed0_31.sh

tmux new -s eval_400_5tasks
  cd /data/yingxi/RLinf_RoboFAPE

  RL_LOG_DIR=/data/yingxi/RLinf_RoboFAPE/logs/pushcube_rl_env32_abs30_debugfix2_seed0_31_20260826_112825 \
  OUT_DIR=$RL_LOG_DIR/sweep_multiseed \
  SWEEP_GPU_IDS=3 \
  SWEEP_RAY_PORT=6387 \
  SWEEP_RAY_DASHBOARD_PORT=8267 \
  SWEEP_RAY_DASHBOARD_AGENT_PORT=52373 \
  SWEEP_RAY_TEMP_DIR=/data/yingxi/ray_tmp_pushcube_eval_6387 \
  bash run_train/eval_checkpoint/watch_pushcube_eval_sweep_seed0_31.sh

## Robometer
复现命令
  先在 ssh -p 30088 root@10.210.22.182 上准备 ckpt：

  cd /data/yingxi/robometer
  if [[ ! -d robometer-4b_basefixed ]]; then
    cp -a robometer-4b robometer-4b_basefixed
  fi
  sed -i 's#/opt/caoyuhang/Pretrained_models/Qwen3-VL-4B-Instruct#/data/yingxi/robometer/Qwen3-VL-4B-Instruct/#' \
    robometer-4b_basefixed/config.yaml

  启动 train + server：

  cd /data/yingxi/RLinf_RoboFAPE
  RUN_DIR=/data/yingxi/RLinf_RoboFAPE/logs/pushcube_rl_env32_abs30_robometer4b_parallel_threadcap_$(date +%Y%m%d_%H%M%S)
  mkdir -p "$RUN_DIR"

  tmux new-session -d -s robometer4b_train \
  "cd /data/yingxi/RLinf_RoboFAPE && env \
  LOG_DIR=$RUN_DIR \
  RL_GPU_IDS=0,1 \
  ROBOMETER_GPU_ID=2 \
  START_ROBOMETER_SERVER=true \
  ROBOMETER_CKPT=/data/yingxi/robometer/robometer-4b_basefixed \
  ROBOMETER_PORT=8001 \
  ROBOMETER_STARTUP_WAIT_S=120 \
  ROBOMETER_HEALTH_RETRIES=36 \
  RL_RAY_PORT=6400 \
  RAY_DASHBOARD_PORT=8280 \
  RAY_DASHBOARD_AGENT_PORT=52380 \
  RAY_CLIENT_SERVER_PORT=10040 \
  RAY_NUM_CPUS=8 \
  RAY_TMPDIR=/data/yingxi/r6400r4b \
  RAY_MIN_WORKER_PORT=14800 \
  RAY_MAX_WORKER_PORT=15199 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  TOKENIZERS_PARALLELISM=false \
  bash run_train/maniskill_robometer_rl/launch_pushcube_rl_seed0_31.sh \
  > $RUN_DIR/launcher.log 2>&1"

  提前启动 sweep eval monitor：

  cd /data/yingxi/RLinf_RoboFAPE
  tmux new-session -d -s robometer4b_eval \
  "cd /data/yingxi/RLinf_RoboFAPE && env \
  RL_LOG_DIR=$RUN_DIR \
  OUT_DIR=$RUN_DIR/sweep_gpu3 \
  SWEEP_GPU_IDS=3 \
  SWEEP_RAY_PORT=6401 \
  SWEEP_RAY_DASHBOARD_PORT=8281 \
  SWEEP_RAY_DASHBOARD_AGENT_PORT=52381 \
  SWEEP_RAY_CLIENT_SERVER_PORT=10041 \
  SWEEP_RAY_MIN_WORKER_PORT=15200 \
  SWEEP_RAY_MAX_WORKER_PORT=15599 \
  SWEEP_RAY_TEMP_DIR=/data/yingxi/r6401e4b \
  bash run_train/eval_checkpoint/watch_pushcube_eval_sweep_seed0_31.sh \
  > $RUN_DIR/sweep_gpu3/eval_watch.log 2>&1"

  并行注意点
  必须唯一隔离：RL_RAY_PORT、RAY_DASHBOARD_PORT、RAY_DASHBOARD_AGENT_PORT、RAY_CLIENT_SERVER_PORT、worker port range、
  RAY_TMPDIR、LOG_DIR、eval 侧对应端口和 tmp dir。RAY_TMPDIR 要短，否则 Ray 会报 AF_UNIX socket path 超长。这里
  RAY_NUM_CPUS=8 和 thread caps 是为避免之前的 can't start new thread。
## Tensorboard
更新后的 TB 重启工作流

  以后重启 compare 面板，在 30088 这边执行：

  ssh 10.210.22.182
  cd /data/yingxi/RLinf_RoboFAPE
  bash /tmp/start_tb_compare_30088.sh

  确认 logdir：

  cat logs/tensorboard/tb_compare.logdir_spec

  确认 TB 进程：

  ps -eo pid=,args= | grep -i tensorboard | grep -v grep

  确认 HTTP/run 已加载：

  curl -sS http://127.0.0.1:6006/data/runs

  确认 checkpoint400_current_31065 的 step：

  .venv/bin/python - <<'PY'
  import json, urllib.parse, urllib.request
  base = "http://127.0.0.1:6006"
  run = "checkpoint400_current_31065/."
  for tag in ["env/success_once", "train/actor/current_version"]:
      qs = urllib.parse.urlencode({"run": run, "tag": tag})
      vals = json.load(urllib.request.urlopen(base + "/data/plugin/scalars/scalars?" + qs))
      print(tag, "count", len(vals), "last_step", vals[-1][1], "last_value", vals[-1][2])
  PY
