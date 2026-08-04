#!/bin/bash
# Emit only on meaningful change: training version advance (healthy), eval
# per-ckpt start/done, or any error/crash signal. Polls every 90s.
RUN=/data/yingxi/RLinf_RoboFAPE/logs/20260731-11:18:41-peg_insertion_rl_async_absolute_16ep_single_step/peg_insertion_async_ppo_pi05_robometer
LOG=$RUN/rl_eval_sweep_48ep/resume_run.log
S01=rl_abs_resume220_gpu01_6384
S23=rl_abs_resume10_env16_gpu23_6385
ver () { tmux capture-pane -t "$1" -p -S -2000 2>/dev/null | grep -oE "version=[0-9]+" | tail -1 | cut -d= -f2; }
p01=$(ver $S01); p23=$(ver $S23); pev=""; perr=""; stall01=0; stall23=0
echo "baseline GPU01=$p01 GPU23=$p23"
while true; do
  sleep 90
  v01=$(ver $S01); v23=$(ver $S23)
  [ "$v01" != "$p01" ] && { echo "GPU01 training: v$p01 -> v$v01 (alive)"; p01=$v01; stall01=0; } || { stall01=$((stall01+1)); [ $stall01 -eq 16 ] && echo "WARN: GPU01 no version advance in ~24min (was v$p01) - check for stall"; }
  [ "$v23" != "$p23" ] && { echo "GPU23 training: v$p23 -> v$v23 (alive)"; p23=$v23; stall23=0; } || { stall23=$((stall23+1)); [ $stall23 -eq 16 ] && echo "WARN: GPU23 no version advance in ~24min (was v$p23) - check for stall"; }
  ev=$(grep -oE "\[gpu [0-9] step [0-9]+\] evaluating" "$LOG" 2>/dev/null | sort -u | tail -1)
  [ "$ev" != "$pev" ] && [ -n "$ev" ] && { echo "eval ckpt start: $ev"; pev=$ev; }
  err=$(grep -E "Traceback|OutOfMemory|CUDA out of memory|resume sweep exit=|Could not read .dashboard. from GCS|raylet.*crash|ConnectionResetError|EXIT 255" "$LOG" 2>/dev/null | tail -1)
  [ -n "$err" ] && [ "$err" != "$perr" ] && { echo "EVALLOG: $err"; perr=$err; }
  grep -qE "resume sweep exit=0" "$LOG" 2>/dev/null && { echo "EVAL DONE (exit 0)"; break; }
  grep -qE "resume sweep exit=[^0]|EXIT 255" "$LOG" 2>/dev/null && { echo "EVAL FAILED (nonzero exit)"; break; }
done
echo "monitor exit"
