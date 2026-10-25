#!/bin/bash
# Sequential experiment queue: one model process at a time (8 GiB box). Usage: queue.sh STEP[:VARIANT] ...
cd "$(dirname "$0")"; source ../../.venv/bin/activate
for spec in "$@"; do
  e=${spec%%:*}; v=${spec#*:}; [ "$v" = "$spec" ] && v=it
  echo "[$(date -u +%FT%TZ)] start $e variant=$v"
  # PL-16: one model process at a time is enforced here (blocking flock), not by external waiters
  VARIANT=$v flock /tmp/lab_model.lock python ../tools/procmetrics.py --cycle ${CYCLE:-2} --step "$e:$v" --kind exp --artifact "experiments/$e" -- \
    python $e/${SCRIPT:-run.py} > $e/stdout_$v.log 2> $e/stderr_$v.log
  rc=$?  # capture before $(date) in the echo resets $? (cycle-2 bug: rc=3 was logged as exit=0)
  echo "[$(date -u +%FT%TZ)] end $e variant=$v exit=$rc"
done
