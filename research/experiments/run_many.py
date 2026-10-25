"""Lab server: load each checkpoint once and run several experiment scripts in one process.

Why (PROCESS_LEDGER #5): both sandbox losses happened minutes after starting the 3rd/4th consecutive model
process; one long-lived process removes repeated 4.6 GB loads (and saves 30-60 s each). After every
experiment its directory and the logs are committed and pushed, so a sandbox loss costs at most one experiment.
Usage: python run_many.py E002_logit_lens:it E004_factual_patching:it E002_logit_lens:base
"""
import json, os, resource, runpy, subprocess, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "lib"))
import gemma_loader, runner
_cache = {}
_orig_load, _OrigRunner = gemma_loader.load, runner.Runner


def cached_load(variant="it"):
    if variant not in _cache:
        for v in list(_cache):  # one checkpoint in RAM at a time
            del _cache[v]
        import gc; gc.collect()
        _cache[variant] = list(_orig_load(variant)) + [None]
    m, t, info, _ = _cache[variant]
    return m, t, dict(info, cached=True)


def cached_runner(model):
    for v, entry in _cache.items():
        if entry[0] is model:
            if entry[3] is None:
                entry[3] = _OrigRunner(model)
            return entry[3]
    return _OrigRunner(model)


gemma_loader.load, runner.Runner = cached_load, cached_runner


def git_push(paths, msg):
    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    cmd = f"cd {root} && flock /tmp/lab_git.lock sh -c 'git add {' '.join(paths)} && git commit -qm \"{msg}\" && git push -q origin HEAD'"
    return subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


for spec in sys.argv[1:]:
    exp, _, variant = spec.partition(":")
    variant = variant or "it"
    os.environ["VARIANT"] = variant
    t0, rc = time.time(), "ok"
    out = open(os.path.join(HERE, exp, f"stdout_{variant}.log"), "w")
    old = sys.stdout; sys.stdout = out
    try:
        runpy.run_path(os.path.join(HERE, exp, "run.py"), run_name="__main__")
    except BaseException:
        rc = "fail"; traceback.print_exc(file=out)
    finally:
        sys.stdout = old; out.close()
    line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cycle": "1", "step": f"{exp}:{variant}", "kind": "exp",
            "wall_s": round(time.time() - t0, 1), "peak_rss_gb_process": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
            "outcome": rc, "artifact": f"experiments/{exp}", "runner": "run_many"}
    with open(os.path.join(HERE, "..", "logs", "process_metrics.jsonl"), "a") as f:
        f.write(json.dumps(line) + "\n")
    print(json.dumps(line), flush=True)
    pushed = git_push([f"research/experiments/{exp}", "research/logs"], f"Auto: {exp}:{variant} {rc}")
    print("pushed" if pushed == 0 else f"push rc={pushed}", flush=True)
