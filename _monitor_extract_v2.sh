#!/usr/bin/env bash
# One-shot extract: echo latest step metrics for both runs + crash flags.
# Called once per poll by the local monitor loop (fresh ssh each time).
cd /data/yingxi/RLinf_RoboFAPE 2>/dev/null || exit 0
# tag:tmux_session:metrics.log path
for spec in \
  "gpu01-abs:rl_abs_bonus0_fresh_gpu01:logs/20260805-09:42:31-peg_insertion_rl_async_absolute_independent_window_bonus0_fresh_gpu01/metrics.log" \
  "gpu23-delta:rl_delta_skip0_w15_p1_fresh_v3_gpu23:logs/20260805-02:46:41-peg_insertion_rl_async_delta_independent_window_skip0_w15_p1_fresh_v3_gpu23/metrics.log"
do
  IFS=: read -r tag sess f <<< "$spec"
  ln=$(grep -n "Global Step:" "$f" 2>/dev/null | tail -1 | cut -d: -f1)
  if [ -z "$ln" ]; then echo "$tag NONE"; continue; fi
  box=$(tail -n +"$ln" "$f" | head -60)
  s=$(echo "$box" | grep -oE "Global Step: +[0-9]+" | head -1 | grep -oE "[0-9]+$")
  if [ -z "$s" ]; then echo "$tag NONE"; continue; fi
  esr=$(echo "$box" | grep -oE "episode_success_rate=[0-9.]+" | head -1 | cut -d= -f2)
  so=$(echo "$box" | grep -oE "success_once=[0-9.]+" | head -1 | cut -d= -f2)
  ff=$(echo "$box" | grep -oE "window/forced_timeout_fraction=[0-9.]+" | head -1 | cut -d= -f2)
  corr=$(echo "$box" | grep -oE "policy_adv_logprob_corr=[0-9.naN-]+" | head -1 | cut -d= -f2)
  st=$(echo "$box" | grep -oE "Step Time: [0-9.]+s" | head -1 | sed 's/Step Time: //')
  echo "$tag $s $esr $so $ff $corr $st"
done
# crash check (last 15 pane lines only — avoids stale scrollback false-positives)
for sess in rl_abs_bonus0_fresh_gpu01 rl_delta_skip0_w15_p1_fresh_v3_gpu23; do
  c=$(tmux capture-pane -t "$sess" -p -S -15 2>/dev/null | grep -cE "Traceback|SESSION_EXIT=[^0]|actor died|RuntimeError")
  [ "$c" -gt 0 ] && echo "CRASH $sess"
done
