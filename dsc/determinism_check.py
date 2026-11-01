"""Probe a local/remote OpenAI-compatible endpoint for run-to-run and cross-batch determinism.

Sends the same prompt at increasing concurrency (a proxy for the server's batch size) and
compares outputs. Batch invariance is the property that matters; temperature=0 is not enough.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def first_divergence(a: str, b: str) -> int | None:
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()[:16]


def run_probe(call_fn, prompt: str, batch_sizes=(1, 8, 32), repeat: int = 2) -> dict:
    all_out = []
    per = []
    errors_total = 0
    for b in batch_sizes:
        outs = []
        for _ in range(max(1, repeat)):
            with ThreadPoolExecutor(max_workers=max(1, int(b))) as ex:
                outs.extend(ex.map(lambda _i: call_fn(prompt), range(max(1, int(b)))))
        errs = [o for o in outs if o.startswith("ERROR:")]
        ok = [o for o in outs if not o.startswith("ERROR:")]
        errors_total += len(errs)
        all_out.extend(ok)
        if not ok:
            per.append({"concurrency": int(b), "n": len(outs), "unique": 0,
                        "errors": len(errs), "first_divergence": [], "hashes": [],
                        "example": errs[0][:120] if errs else ""})
            continue
        ref = ok[0]
        div = [first_divergence(ref, o) for o in ok[1:]]
        per.append({
            "concurrency": int(b), "n": len(outs), "unique": len(set(ok)),
            "errors": len(errs),
            "first_divergence": [d for d in div if d is not None][:5],
            "hashes": sorted({_sha(o) for o in ok}),
            "example": ref[:120],
        })
    stable = bool(all_out) and len(set(all_out)) == 1 and errors_total == 0
    return {
        "prompt_sha": _sha(prompt),
        "cross_batch_equal": stable,
        "unique_total": len(set(all_out)),
        "errors_total": errors_total,
        "runs": per,
        "interpretation": ("batch-invariant for this probe" if stable
                           else ("ERRORS: %d requests failed; result inconclusive" % errors_total
                                 if errors_total else
                                 "NOT batch-invariant: outputs change with concurrency/batch")),
    }


def http_call(url: str, model: str, api_key: str = "", max_tokens: int = 64, timeout: float = 60.0):
    base = url.rstrip("/")
    if not base.endswith("/chat/completions"):
        base += "/chat/completions"

    def fn(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "seed": 0, "max_tokens": max_tokens,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        req = urllib.request.Request(base, data=body, headers=headers)
        last = ""
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    d = json.loads(r.read().decode())
                return d["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as e:
                last = "HTTP %s: %s" % (e.code, e.reason)
                if 400 <= int(e.code) < 500:
                    break
            except Exception as e:
                last = str(e)
            time.sleep(0.4 * (attempt + 1))
        return "ERROR:" + last[:160]
    return fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsc.determinism_check")
    ap.add_argument("--url", required=True, help="npr. http://127.0.0.1:8100/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--prompt", default="Tell me about Richard Feynman")
    ap.add_argument("--batch-sizes", default="1,8,32")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args(argv)
    sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    fn = http_call(args.url, args.model, args.api_key, args.max_tokens)
    res = run_probe(fn, args.prompt, sizes, args.repeat)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["cross_batch_equal"] else 2


if __name__ == "__main__":
    sys.exit(main())
