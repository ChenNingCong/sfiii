#!/bin/bash
# Watchdog for the 20M async run. Emits ONE stdout line per actionable event:
# heartbeat (every 2M steps), STALL (no progress 5min), DIED, CRASH, COMPLETE.
LOG=/home/zzhang18/nchen3/sfiii/train_ft2e5.log
PAT='[p]po_sfiii.py --cfgFile config/config_finetune2e5.yaml'
last_hb=0; laststep=0; lastchange=$(date +%s)
cur_step() { grep -oE 'step [0-9]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+'; }
while true; do
  s=$(cur_step); s=${s:-0}; now=$(date +%s)
  # crash markers (check before liveness so we catch the traceback)
  if grep -qE 'Traceback|Fatal Python|Segmentation|invalid values|CUDA error|out of memory|oom-kill' "$LOG" 2>/dev/null; then
    echo "CRASH at step $s :: $(grep -E 'Traceback|Fatal|Segmentation|invalid|CUDA error|out of memory' "$LOG" | tail -1)"
    break
  fi
  if ! pgrep -f "$PAT" >/dev/null; then
    if [ "$s" -ge 19900000 ]; then echo "COMPLETE: 20M run finished at step $s"; else echo "DIED: process gone at step $s (see train_ft2e5.log)"; fi
    break
  fi
  if [ "$s" -gt "$laststep" ]; then laststep=$s; lastchange=$now; fi
  if [ $((now-lastchange)) -ge 300 ]; then echo "STALL: no step progress ${now}-${lastchange}s (stuck at step $s)"; lastchange=$now; fi
  if [ "$s" -ge $((last_hb+2000000)) ]; then echo "HEARTBEAT $(grep -E 'it [0-9]+/' "$LOG" | tail -1)"; last_hb=$s; fi
  sleep 60
done
