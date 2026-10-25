#!/bin/bash
# Protects the sandbox: if MemAvailable drops below a floor, SIGTERM experiment processes before
# the container hits its box limit (cycle-1 lesson: a box-level kill wipes all unpushed work).
FLOOR_MB=${FLOOR_MB:-1000}
LOG=$(dirname "$0")/../logs/memwatch.log
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt "$FLOOR_MB" ]; then
    echo "$(date -u +%FT%TZ) MemAvailable=${avail}MB < ${FLOOR_MB}MB: killing experiment processes" >> "$LOG"
    pkill -TERM -f '^python .*(run|run_many)\.py' ; sleep 5
  fi
  sleep 2
done
