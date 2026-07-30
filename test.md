# 16-Episode Single-Step Delta Training

This command starts a fresh run from the configured SFT checkpoint. It does not
resume an RL checkpoint. The async rollout capacity now accounts for two stored
trajectories per actor rank; the planner timeout remains defensive subprocess
recovery and is not the fix for the previous `candidates=1/2` deadlock.
The global batch covers all 16 episodes in one optimizer step, uses
`value_loss_coef=0.1` after a 50-step critic warmup, and emits sampled gradient
conflict diagnostics every 10 actor versions.

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
`candidates=2`, each version performs one optimizer step, and the pre-update
proximal KL is approximately zero. During critic warmup `actor/lr` is zero; after
warmup it must be `3e-7`. After actor updates begin, monitor
`actor/adv_weighted_policy_logprob_delta` and the sampled policy/critic gradient
norm and cosine metrics. Checkpoints are complete every 10 steps, with multiples
of 50 retained permanently.
