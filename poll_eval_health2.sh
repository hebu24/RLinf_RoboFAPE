#!/bin/bash
RUN=/data/yingxi/RLinf_RoboFAPE/logs/20260731-11:18:41-peg_insertion_rl_async_absolute_16ep_single_step/peg_insertion_async_ppo_pi05_robometer
OUT=$RUN/rl_eval_sweep_48ep
LOG=$OUT/resume_run2.log
S01=rl_abs_resume220_gpu01_6384; S23=rl_abs_resume10_env16_gpu23_6385
ver () { tmux capture-pane -t "$1" -p -S -2000 2>/dev/null | grep -oE "version=[0-9]+" | tail -1 | cut -d= -f2; }
newmetrics () { find "$OUT" -path "*global_step_25[0-9]_trainenvstep_*trajectory_metrics.json" -o -path "*global_step_30[0-9]_trainenvstep_*trajectory_metrics.json" -o -path "*global_step_31[0-9]_trainenvstep_*trajectory_metrics.json" 2>/dev/null | sort -u | wc -l; }
p01=$(ver $S01); p23=$(ver $S23); pev=""; perr=""; pmet=$(newmetrics); stall01=0; stall23=0
echo "baseline2 GPU01=$p01 GPU23=$p23 new_metrics=$pmet"
while true; do
  sleep 90
  v01=$(ver $S01); v23=$(ver $S23)
  [ "$v01" != "$p01" ] && { echo "GPU01 training: v$p01 -> v$v01 (alive)"; p01=$v01; stall01=0; } || { stall01=$((stall01+1)); [ $stall01 -eq 16 ] && echo "WARN: GPU01 no advance ~24min (v$p01)"; }
  [ "$v23" != "$p23" ] && { echo "GPU23 training: v$p23 -> v$v23 (alive)"; p23=$v23; stall23=0; } || { stall23=$((stall23+1)); [ $stall23 -eq 16 ] && echo "WARN: GPU23 no advance ~24min (v$p23)"; }
  m=$(newmetrics); [ "$m" != "$pmet" ] && { echo "eval NEW ckpt done: $pmet -> $m of 3 new metrics written"; pmet=$m; }
  err=$(grep -E "Traceback|OutOfMemory|CUDA out of memory|sweep2 exit=|Could not read .dashboard. from GCS|raylet.*crash|ActorDiedError" "$LOG" 2>/dev/null | tail -1)
  [ -n "$err" ] && [ "$err" != "$perr" ] && { echo "EVALLOG: $err"; perr=$err; }
  grep -qE "sweep2 exit=0" "$LOG" 2>/dev/null && { echo "EVAL DONE (exit 0)"; break; }
  grep -qE "sweep2 exit=[^0]" "$LOG" 2>/dev/null && { echo "EVAL FAILED (nonzero)"; break; }
done
echo "monitor2 exit"
