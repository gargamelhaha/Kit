#!/bin/bash
# List PIDs whose argv[1] (after the interpreter) exactly matches $1 — never matches the calling shell (PL-14).
ps -eo pid,args | awk -v t="$1" '{for(i=2;i<=NF;i++) if($i==t){print $1; break}}' | grep -vw $$
