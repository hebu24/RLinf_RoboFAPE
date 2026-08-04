#!/usr/bin/env bash
# Per-step success-rate monitor for the two independent_window RL runs.
# Echoes one line per NEW step per run; also flags crashes. Run via:
#   ssh -o ServerAliveInterval=20 xulab 'bash /data/yingxi/RLinf_RoboFAPE/_remote_monitor_independent.sh'
cd /data/yingxi/RLinf_RoboFAPE
last1=7   # GPU01 currently at step 7
last2=6   # GPU23 currently at step 6
LOGBASE="logs/20260804-16:14:23-peg_insertion_rl_async"
while true; do
  for pair in "absolute_independent_window_gpu01:gpu01-abs" "delta_independent_window_gpu23:gpu23-delta"; do
    run=${pair%%:*}; tag=${pair##*:}
    f="$LOGBASE_$run/metrics.log"
    ln=$(grep -n "Global Step:" "$f" 2>/dev/null | tail -1 | cut -d: -f1)
    [ -z "$ln" ] && continue
    box=$(tail -n +"$ln" "$f" | head -40)
    s=$(echo "$box" | grep -oE "Global Step: +[0-9]+" | head -1 | grep -oE "[0-9]+$")
    [ -z "$s" ] && continue
    should_emit=0
    if [ "$tag" = "gpu01-abs" ]; then
      if [ "$s" -gt "$last1" ]; then last1=$s; should_emit=1; fi
    else
      if [ "$s" -gt "$last2" ]; then last2=$s; should_emit=1; fi
    fi
    [ "$should_emit" -eq 1 ] || continue
    esr=$(echo "$box" | grep -oE "episode_success_rate=[0-9.]+" | head -1 | cut -d= -f2)
    so=$(echo "$box" | grep -oE "success_once=[0-9.]+" | head -1 | cut -d= -f2)
    nt=$(echo "$box" | grep -oE "window/natural_terminal_episodes=[0-9.]+" | head -1 | cut -d= -f2)
    ft=$(echo "$box" | grep -oE "window/forced_timeout_episodes=[0-9.]+" | head -1 | cut -d= -f2)
    we=$(echo "$box" | grep -oE "window/episodes=[0-9.]+" | head -1 | cut -d= -f2)
    ff=$(echo "$box" | grep -oE "window/forced_timeout_fraction=[0-9.]+" | head -1 | cut -d= -f2)
    st=$(echo "$box" | grep -oE "Step Time: [0-9.]+s" | head -1 | sed "s/Step Time: //")
    echo "[$tag] step=$s episode_succ=$esr succ_once=$so window: ${we}eps(${nt}nat/${ft}forced,${ff}timeout) ${st}"
  done
  # crash guard
  for s in rl_abs_independent_window_gpu01 rl_delta_independent_window_gpu23; do
    c=$(tmux capture-pane -t "$s" -p -S -80 2>/dev/null | grep -cE "Traceback|SESSION_EXIT=[^0]|RuntimeError|actor died")
    if [ "$c" -gt 0 ]; then echo "[$s] CRASH/EXIT DETECTED — check pane"; fi
  done
  sleep 60
done
