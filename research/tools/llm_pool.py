"""Parallel client for the lab's external OpenAI-compatible LLM API (used as logical subagents / auditors).

Credentials are read from env or ~/.config/lab/api.env (LAB_API_BASE, LAB_API_KEY) and never written to the repo.
Every call is appended to research/logs/llm_calls.jsonl (model, prompt hash, latency, tokens, outcome; no key).
Identical (model, messages, temperature, max_tokens) requests are served from research/logs/llm_cache/ unless
cache=False, so re-running an analysis is free and deterministic.

    from llm_pool import Pool
    outs = Pool(concurrency=16).map([{"model": "free-deepseek-4.1-flash", "messages": [...]}, ...])
CLI: python llm_pool.py ramp   -> concurrency/throughput measurement (logs/llm_ramp.json)
"""
import fcntl, hashlib, json, os, sys, tempfile, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(ROOT, "logs", "llm_calls.jsonl")
CACHE = os.path.join(ROOT, "logs", "llm_cache")
MODELS = {"deepseek": "free-deepseek-4.1-flash", "minimax": "minimax-m3.1-flash"}  # display names: Deepseek 4.1 Flash, MiniMax M3.1 Flash
_lock = threading.Lock()
MAX_PARALLEL = 6  # server-enforced per key (HTTP 429 "up to 6 at a time", measured 2026-10-25, logs/llm_ramp.json)
SLOT_DIR = os.path.join(tempfile.gettempdir(), "lab_llm_slots")


class _Slot:
    """Cross-process slot (flock on one of MAX_PARALLEL files): the cap is per key, so it must hold across processes.
    PL-7b: a per-process semaphore let 3 concurrent tools put up to 18 requests in flight -> 78 HTTP 429s."""
    def __enter__(self):
        os.makedirs(SLOT_DIR, exist_ok=True)
        while True:
            for i in range(MAX_PARALLEL):
                f = open(os.path.join(SLOT_DIR, f"{i}.lock"), "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); self.f = f; return self
                except BlockingIOError:
                    f.close()
            time.sleep(0.2)

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()


def _creds():
    base, key = os.environ.get("LAB_API_BASE"), os.environ.get("LAB_API_KEY")
    p = os.path.expanduser("~/.config/lab/api.env")
    if (not base or not key) and os.path.exists(p):
        kv = dict(l.strip().split("=", 1) for l in open(p) if "=" in l)
        base, key = base or kv.get("LAB_API_BASE"), key or kv.get("LAB_API_KEY")
    if not base or not key:
        raise RuntimeError("LAB_API_BASE / LAB_API_KEY not configured")
    return base.rstrip("/"), key


def _read_sse(resp):
    content, reasoning, usage, finish = [], [], {}, None
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        d = json.loads(data)
        if d.get("error"):
            raise RuntimeError(f"stream error: {str(d['error'])[:200]}")
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            content.append(delta.get("content") or ""); reasoning.append(delta.get("reasoning_content") or "")
            finish = ch.get("finish_reason") or finish
    if finish is None:
        raise RuntimeError("stream ended without finish_reason")
    return "".join(content), "".join(reasoning), usage, finish


class Pool:
    def __init__(self, concurrency=MAX_PARALLEL, timeout=1800, retries=4, tag=""):
        self.base, self.key = _creds()
        self.concurrency, self.timeout, self.retries, self.tag = concurrency, timeout, retries, tag
        os.makedirs(CACHE, exist_ok=True)

    def _key(self, req):
        blob = json.dumps({k: req.get(k) for k in ("model", "messages", "temperature", "max_tokens", "reasoning_effort")}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def call(self, req, cache=True):
        req = {"temperature": 0.0, "max_tokens": 4096, **req}
        req["model"] = MODELS.get(req["model"], req["model"])
        h = self._key(req); cp = os.path.join(CACHE, h + ".json")
        if cache and os.path.exists(cp):
            return {**json.load(open(cp)), "cached": True}
        # PL-9: the gateway (nginx) cuts non-streaming responses at 300 s (HTTP 504); SSE streaming keeps the connection alive
        body = json.dumps({**{k: v for k, v in req.items() if k not in ("meta",)}, "stream": True,
                           "stream_options": {"include_usage": True}}).encode()
        err, t0 = None, time.time()
        attempt, busy = 0, 0
        while attempt <= self.retries:
            t0 = time.time()
            try:
                r = urllib.request.Request(self.base + "/chat/completions", data=body, headers={
                    "Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
                with _Slot():
                    content, reasoning, usage, finish = _read_sse(urllib.request.urlopen(r, timeout=self.timeout))
                out = {"content": content, "reasoning": reasoning,
                       "usage": usage, "finish": finish,
                       "latency_s": round(time.time() - t0, 2), "model": req["model"], "hash": h}
                if cache:
                    json.dump(out, open(cp, "w"))
                self._log(req, h, out["latency_s"], "ok", out["usage"], attempt)
                return {**out, "cached": False}
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: {e.read()[:200]!r}"
                if e.code == 429 and busy < 150:  # parallelism cap hit (other process sharing the key): wait, don't burn a retry
                    busy += 1; time.sleep(2); continue
            except Exception as e:  # network / timeout / malformed
                err = f"{type(e).__name__}: {e}"[:300]
            self._log(req, h, round(time.time() - t0, 2), "error", {}, attempt, err)
            time.sleep(min(30, 2 ** attempt)); attempt += 1
        return {"content": "", "error": err, "model": req["model"], "hash": h, "cached": False}

    def _log(self, req, h, lat, outcome, usage, attempt, err=None):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tag": self.tag, "model": req["model"], "hash": h,
               "latency_s": lat, "outcome": outcome, "attempt": attempt, "prompt_tokens": usage.get("prompt_tokens"),
               "completion_tokens": usage.get("completion_tokens"), "meta": req.get("meta")}
        if err:
            rec["error"] = err
        with _lock, open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def map(self, reqs, cache=True):
        with ThreadPoolExecutor(self.concurrency) as ex:
            return list(ex.map(lambda r: self.call(r, cache), reqs))


def ramp(levels=(1, 4, 8, 16, 32, 64), model="free-deepseek-4.1-flash"):
    """Throughput vs concurrency; stops when errors >10% or throughput gain <10% at a level."""
    res, prev = [], 0.0
    for c in levels:
        n = max(8, 2 * c)
        reqs = [{"model": model, "max_tokens": 64, "meta": {"ramp": c},
                 "messages": [{"role": "user", "content": f"Ramp probe {c}-{i}: reply with the single word OK."}]}
                for i in range(n)]
        t0 = time.time(); outs = Pool(concurrency=c, retries=0, timeout=120, tag="ramp").map(reqs, cache=False)
        wall = time.time() - t0
        ok = [o for o in outs if not o.get("error")]
        lat = sorted(o["latency_s"] for o in ok)
        row = {"model": model, "concurrency": c, "n": n, "ok": len(ok), "wall_s": round(wall, 1),
               "req_per_s": round(len(ok) / wall, 3), "p50_s": lat[len(lat) // 2] if lat else None,
               "p90_s": lat[int(len(lat) * 0.9)] if lat else None,
               "errors": sorted({o["error"][:80] for o in outs if o.get("error")})}
        print(json.dumps(row), flush=True); res.append(row)
        if len(ok) < 0.9 * n or row["req_per_s"] < 1.1 * prev:
            break
        prev = row["req_per_s"]
    return res


if __name__ == "__main__":
    if sys.argv[1:2] == ["ramp"]:
        out = {m: ramp(model=m) for m in sys.argv[2:] or MODELS.values()}
        json.dump(out, open(os.path.join(ROOT, "logs", "llm_ramp.json"), "w"), indent=1)
