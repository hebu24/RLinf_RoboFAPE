## Motion planning data collection
```bash
cd /opt/yingxi/RLinf_RoboFAPE

/opt/kairan/envs/rlinf/bin/python run_train/peginsertion_maniskill_pi0.5/collect_peg_insertion_data.py \
  --num-traj 3200 \
  --output-dir /opt/yingxi/RLinf_RoboFAPE/run_train/peginsertion_maniskill_pi0.5/data/peg_insertion_vertical_3200 \
  --seed 0 \
  --num-workers 32 \
  --gpu-ids 4,5,6,7 \
  --worker-stagger 5.0
```

## Finetune SFT
```bash
bash sft_finetune.sh

# Resume training
RESUME_DIR=/opt/yingxi/RLinf_RoboFAPE/logs/20260709-15:07:21-peg_insertion_sft_openpi_pi05/peg_insertion_sft/checkpoints/global_step_8000 bash sft_finetune.sh
```

## Checkpoint 测试

使用训练保存的 actor 目录，例如：

```text
logs/<run>/checkpoints/global_step_50/actor
```

运行评测：

```bash
cd /home/hebu/code/robofape/RLinf_RoboFAPE

# Optional: stop all running ray jobs
/opt/kairan/envs/rlinf/bin/ray stop

VENV_DIR=/opt/kairan/envs/rlinf \
CHECKPOINT_PATH=/opt/yingxi/RLinf_RoboFAPE/logs/20260709-15:07:21-peg_insertion_sft_openpi_pi05/peg_insertion_sft/checkpoints/global_step_8000/actor \
GPU_IDS=0-3 \
NUM_EVAL_EPISODES=4 \
NUM_ENVS=4 \
EVAL_ACTION_SCALE=10.0 \
SAVE_VIDEO=true \
bash run_train/eval_checkpoint/run_peginsertion.sh
```

要求：`NUM_EVAL_EPISODES` 必须能被 `NUM_ENVS` 整除，因为评测按固定并行 batch 跑完。

常用评测变量：

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `CHECKPOINT_PATH` | 见脚本默认值 | 被测试的 actor checkpoint。 |
| `GPU_IDS` | `0` | 评测用 GPU placement，可写单卡 id 或范围字符串。 |
| `NUM_EVAL_EPISODES` | `25` | 总评测轨迹数。 |
| `NUM_ENVS` | `5` | 并行环境数。 |
| `MAX_EPISODE_STEPS` | `600` | 单条轨迹最大步数。 |
| `SEED` | `0` | 评测 seed。 |
| `SAVE_VIDEO` | `true` | 是否保存视频。 |
| `IGNORE_TERMINATIONS` | `true` | 成功后是否继续跑到 rollout 长度。 |
| `FIXED_RESET_STATE_IDS` | `false` | 是否固定 reset state id。 |
| `EVAL_ACTION_SCALE` | `1.0` | 动作缩放。 |

评测输出目录：

```text
logs/<timestamp>-eval-PegInsertionVertical-v1/
```

重点文件：

| 文件/目录 | 内容 |
|---|---|
| `eval.log` | 完整评测日志和 resolved config。 |
| `evaluation_summary.json` | checkpoint 路径、轨迹数和 metrics 汇总。 |
| `video/eval/` | 评测视频。 |


