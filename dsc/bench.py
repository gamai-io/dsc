"""Offline benchmark: latency, hit/miss behaviour, verification accuracy, determinism."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

from .api import Core
from .fingerprints import sha256_hex, simhash64
from .hdc import vector
from .nondeterminism import local_vllm_env, receipt, verify_receipt
from .normalize import canonical, features
from .verify import verify_answer


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round((p / 100.0) * len(xs) + 0.5)) - 1))
    return xs[k]


def run(fn, n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000.0)
    return out


def main() -> int:
    db = os.path.join(tempfile.gettempdir(), "dsc_bench.db")
    for suf in ("", "-wal", "-shm"):
        try:
            os.unlink(db + suf)
        except OSError:
            pass
    core = Core(cache_path=db, cache_enabled=True)

    base = "kupac ACME Hadzici Sarajevo faktura PDV 17% kompresor 10 bar Graco Ultra 795"
    lat_norm = run(lambda: features(base), 5000)
    lat_sim = run(lambda: simhash64(base), 3000)
    lat_vec = run(lambda: vector(base), 3000)
    lat_canon = run(lambda: canonical(base), 5000)

    for i in range(500):
        core.remember("koliko ima kupaca u bazi %d" % (2000 + (i % 20)),
                      "kupaca ima %d" % (1000 + i), sid="bench", domain="doc")
    lat_put = []
    for i in range(200):
        t = time.perf_counter()
        core.remember("pitanje %d o zalihama artikla %d" % (i, i),
                      "zalihe su %d komada" % (10 + i), sid="bench", domain="doc")
        lat_put.append((time.perf_counter() - t) * 1000.0)

    lat_exact = run(lambda: core.recall("koliko ima kupaca u bazi 2005", sid="bench", domain="doc"), 500)
    lat_sem = run(lambda: core.recall("koliko kupaca ima u bazi 2005", sid="bench", domain="doc"), 500)
    lat_miss = run(lambda: core.recall("kompresor 10 bar", sid="bench", domain="doc"), 500)

    src = {"id": 1, "text": "Graco Ultra Max II 795 ima 3300 PSI, protok 0.95 GPM, cijena 1450 KM."}
    good = "Graco Ultra 795 ima 3300 PSI [1]."
    bad = "Graco Ultra 795 ima 3000 PSI [1]."
    neg = "Pumpa ne radi [1]."
    lat_ver = run(lambda: verify_answer(good, [src]), 300)
    env = local_vllm_env("qwen7b", batch_size=8, batch_signature="a,b,c")
    rec = receipt("pitanje", "odgovor", env)
    lat_rec = run(lambda: receipt("pitanje", "odgovor", env), 3000)
    lat_recv = run(lambda: verify_receipt(rec, "pitanje", "odgovor"), 3000)
    ok_good = sum(1 for _ in range(200) if verify_answer(good, [src])["verdict"] == "SUPPORTED") / 200.0
    ok_bad = sum(1 for _ in range(200) if verify_answer(bad, [src])["verdict"] == "UNSUPPORTED") / 200.0
    ok_neg = sum(1 for _ in range(200) if verify_answer(neg, [src])["verdict"] == "UNSUPPORTED") / 200.0

    negatives = [
        ("promet 2025", "promet 2026"),
        ("pumpa radi 10 bar", "pumpa ne radi 10 bar"),
        ("cijena 1450 KM", "cijena 1500 KM"),
        ("tlak 3300 PSI", "tlak 200 bar"),
    ]
    fp = 0
    for store_q, ask_q in negatives:
        core.remember(store_q, "odgovor %s" % store_q, sid="bench", domain="doc")
        if core.recall(ask_q, sid="bench", domain="doc"):
            fp += 1

    code = ("import sys;sys.path.insert(0,'.');from dsc.hdc import vector;"
            "from dsc.fingerprints import simhash64;"
            "print(simhash64('Graco Ultra 795'), __import__('hashlib').sha256(vector('Graco')).hexdigest()[:16])")
    outs = [subprocess.check_output([sys.executable, "-c", code], cwd=".").decode().strip()
            for _ in range(2)]

    res = {
        "deterministic_cross_process": outs[0] == outs[1],
        "latency_ms": {
            "normalize_p50": round(pct(lat_norm, 50), 3), "normalize_p95": round(pct(lat_norm, 95), 3),
            "canonical_p50": round(pct(lat_canon, 50), 3),
            "simhash_p50": round(pct(lat_sim, 50), 3),
            "vector_p50": round(pct(lat_vec, 50), 3),
            "cache_put_p50": round(pct(lat_put, 50), 3), "cache_put_p95": round(pct(lat_put, 95), 3),
            "cache_exact_p50": round(pct(lat_exact, 50), 3),
            "cache_semantic_p50": round(pct(lat_sem, 50), 3),
            "cache_miss_p50": round(pct(lat_miss, 50), 3),
            "verify_p50": round(pct(lat_ver, 50), 3), "verify_p95": round(pct(lat_ver, 95), 3),
            "receipt_p50": round(pct(lat_rec, 50), 3), "receipt_verify_p50": round(pct(lat_recv, 50), 3),
        },
        "quality": {
            "exact_hit": core.recall("koliko ima kupaca u bazi 2005", sid="bench", domain="doc") is not None,
            "semantic_hit": core.recall("koliko kupaca ima u bazi 2005", sid="bench", domain="doc") is not None,
            "negative_false_hits": fp,
            "verify_supported": ok_good,
            "verify_wrong_number": ok_bad,
            "verify_negation": ok_neg,
            "inference_level": env.level(),
            "replay_bound": env.replay_bound(),
        },
        "cache_stats": core.stats(),
    }
    core.close()
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
