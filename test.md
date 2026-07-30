# 16-Episode Single-Step Delta Resume

This command resumes the complete step-50 checkpoint after the FSDP sampled
gradient diagnostic crash. That diagnostic is disabled because it performed
multiple backward calls over one retained FSDP graph. The async rollout capacity
accounts for two stored trajectories per actor rank; the planner timeout remains
defensive subprocess recovery and is not the fix for the previous `candidates=1/2`
deadlock.
The global batch covers all 16 episodes in one optimizer step. The critic input
is detached from the policy backbone, while the value head uses the unscaled
value loss after a 50-step critic warmup. Post-update PPO surrogate improvement,
proximal KL, ratio, clip fraction, and logprob-delta metrics measure the actual
optimizer step without extra backward calls.

```bash
cd /data/yingxi/RLinf_RoboFAPE
tmux new-session -d -s rl_delta_resume50_postppo_shape \
  "export CUDA_VISIBLE_DEVICES=0,1 \
RL_RAY_PORT=6381 \
RAY_DASHBOARD_AGENT_PORT=52367 \
RLINF_PLANNER_REQUEST_TIMEOUT_S=60 \
LOG_DIR=logs/\$(date +'%Y%m%d-%H:%M:%S')-peg_insertion_rl_async_delta_resume50_postppo_shape \
CONFIG_NAME=maniskill_async_ppo_peg_insertion_pi05_delta; \
bash run_train/peginsertion_maniskill_pi0.5/run_peg_insertion_rl_async.sh \
reward.shaping=delta \
actor.optim.lr=3e-7 \
reward.model.timeout_s=600 \
algorithm.rollout_store_wait_timeout_s=1800 \
runner.resume_dir=/data/yingxi/RLinf_RoboFAPE/logs/20260730-14:09:03-peg_insertion_rl_async_delta_16ep_single_step_vloss01_fixed/peg_insertion_async_ppo_pi05_robometer/checkpoints/global_step_50_trainenvstep_47234; \
rc=\$?; echo ===SESSION_EXIT=\$rc===; exec bash"
```

Monitor the run with:

```bash
tmux attach -t rl_delta_resume50_postppo_shape
```

For the first three actor versions, verify both ranks repeatedly reach
`candidates=2`, each version performs one optimizer step, and the pre-update
proximal KL is approximately zero. During critic warmup `actor/lr` is zero; after
warmup it must be `3e-7`. After actor updates begin, monitor
`actor/post_update_ppo_surrogate_improvement` as the primary update-direction
signal, with `actor/post_update_proximal_approx_kl` and
`actor/post_update_proximal_clip_fraction` as update-size signals. Checkpoints
are complete every 10 steps, with multiples of 50 retained permanently.
