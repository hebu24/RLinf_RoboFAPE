#!/bin/bash
RUN=/data/yingxi/RLinf_RoboFAPE/logs/20260731-11:18:41-peg_insertion_rl_async_absolute_16ep_single_step/peg_insertion_async_ppo_pi05_robometer
OUT=$RUN/rl_eval_sweep_48ep
LOG=$OUT/resume_run3_350.log
S01=rl_abs_resume220_gpu01_6384
ver () { tmux capture-pane -t "$1" -p -S -2000 2>/dev/null | grep -oE "version=[0-9]+" | tail -1 | cut -d= -f2; }
m350 () { [ -f "$OUT/global_step_350_trainenvstep_329787/trajectory_metrics.json" ] && echo 1 || echo 0; }
p01=$(ver $S01); pmet=$(m350); perr=""
echo "baseline350 resume220=$p01 350metrics=$pmet"
while true; do
  sleep 90
  v01=$(ver $S01); [ "$v01" != "$p01" ] && { echo "resume220 training: v$p01 -> v$v01 (alive)"; p01=$v01; }
  m=$(m350); [ "$m" != "$pmet" ] && { echo "eval 350 DONE: metrics written ($pmet -> $m)"; pmet=$m; }
  err=$(grep -E "Traceback|OutOfMemory|CUDA out of memory|350 eval exit=|Could not read .dashboard. from GCS|raylet.*crash|ActorDiedError|ModuleNotFoundError" "$LOG" 2>/dev/null | tail -1)
  [ -n "$err" ] && [ "$err" != "$perr" ] && { echo "EVALLOG: $err"; perr=$err; }
  grep -qE "350 eval exit=0" "$LOG" 2>/dev/null && { echo "EVAL DONE (exit 0)"; break; }
  grep -qE "350 eval exit=[^0]" "$LOG" 2>/dev/null && { echo "EVAL FAILED (nonzero)"; break; }
done
echo "monitor350 exit"
