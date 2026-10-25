#!/usr/bin/env python3
"""Run a command and append a process-metrics line (meta-research skill §1).
Usage: procmetrics.py --cycle 1 --step E002 --kind exp [--artifact path] -- cmd args..."""
import argparse, json, os, resource, subprocess, sys, time
ap = argparse.ArgumentParser()
for a in ("--cycle", "--step", "--kind", "--artifact", "--notes"):
    ap.add_argument(a, default="")
args, rest = ap.parse_known_args()
cmd = rest[1:] if rest and rest[0] == "--" else rest
t0 = time.time(); rc = subprocess.call(cmd)
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cycle": args.cycle, "step": args.step, "kind": args.kind,
        "wall_s": round(time.time() - t0, 1), "cpu_s": round(ru.ru_utime + ru.ru_stime, 1), "peak_rss_gb": round(ru.ru_maxrss / 1e6, 2),
        "outcome": "ok" if rc == 0 else f"fail(rc={rc})", "artifact": args.artifact, "notes": args.notes}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "process_metrics.jsonl"), "a") as f:
    f.write(json.dumps(line) + "\n")
sys.exit(rc)
