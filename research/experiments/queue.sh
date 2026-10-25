#!/bin/bash
# Sequential experiment queue: one model process at a time (8 GiB box). Usage: queue.sh STEP[:VARIANT] ...
cd "$(dirname "$0")"; source ../../.venv/bin/activate
for spec in "$@"; do
  e=${spec%%:*}; v=${spec#*:}; [ "$v" = "$spec" ] && v=it
  echo "[$(date -u +%FT%TZ)] start $e variant=$v"
  VARIANT=$v python ../tools/procmetrics.py --cycle ${CYCLE:-2} --step "$e:$v" --kind exp --artifact "experiments/$e" -- \
    python $e/run.py > $e/stdout_$v.log 2> $e/stderr_$v.log
  echo "[$(date -u +%FT%TZ)] end $e variant=$v exit=$?"
done
