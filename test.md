# Fixed 16-Episode Delta Training

This command starts a fresh run from the configured SFT checkpoint. It does not
resume an RL checkpoint. The async rollout capacity now accounts for two stored
trajectories per actor rank; the planner timeout remains defensive subprocess
recovery and is not the fix for the previous `candidates=1/2` deadlock.

```bash
cd /data/yingxi/RLinf_RoboFAPE
tmux new-session -d -s rl_delta_16ep_lr3e7_fixed \
  "export CUDA_VISIBLE_DEVICES=0,1 \
RL_RAY_PORT=6381 \
RAY_DASHBOARD_AGENT_PORT=52367 \
RLINF_PLANNER_REQUEST_TIMEOUT_S=60 \
LOG_DIR=logs/\$(date +'%Y%m%d-%H:%M:%S')-peg_insertion_rl_async_delta_16ep_lr3e7_fixed_full \
CONFIG_NAME=maniskill_async_ppo_peg_insertion_pi05_delta; \
bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
reward.shaping=delta \
actor.optim.lr=3e-7 \
reward.model.timeout_s=600 \
algorithm.rollout_store_wait_timeout_s=1800; \
rc=\$?; echo ===SESSION_EXIT=\$rc===; exec bash"
```

Monitor the run with:

```bash
tmux attach -t rl_delta_16ep_lr3e7_fixed
```

For the first three actor versions, verify both ranks repeatedly reach
`candidates=2`, each version performs two optimizer steps, and the pre-update
proximal KL is approximately zero. During critic warmup `actor/lr` is zero; after
warmup it must be `3e-7`. Checkpoints are complete every 10 steps, with multiples
of 50 retained permanently.
