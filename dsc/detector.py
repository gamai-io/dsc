"""DSC detector: PoA (proof of action/answer), FDS (fleet determinism), SDM (semantic drift)."""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import random
import time
from statistics import mean, pstdev

from .fingerprints import sha256_hex
from .nondeterminism import required_ok


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hmac_hex(key: bytes, msg: str) -> str:
    return hmac.new(key, msg.encode("utf-8", "replace"), hashlib.sha256).hexdigest()


def load_key(path: str) -> bytes:
    st = os.stat(path)
    if st.st_mode & 0o077:
        raise PermissionError("evidence key must be 0600: %s" % path)
    with open(path, "rb") as f:
        return f.read().strip()


def poa_record(kind: str, inputs=None, sources=None, tools=None, result=None,
               prev: str | None = None, ts: float | None = None,
               inference: dict | None = None, key: bytes | None = None,
               prev_chain: str | None = None, seq: int | None = None,
               extra: dict | None = None) -> dict:
    payload = {
        "kind": kind,
        "inputs": inputs or {},
        "sources": sources or [],
        "tools": tools or [],
        "result_hash": sha256_hex(_canon(result)) if result is not None else None,
        "inference": inference,
        "extra": extra or {},
        "prev": prev,
    }
    h = sha256_hex(_canon(payload))
    chain = _hmac_hex(key, (prev_chain or "") + h) if key else None
    return {"id": h[:16], "hash": h, "chain": chain, "prev": prev,
            "prev_chain": prev_chain, "seq": seq,
            "ts": time.time() if ts is None else ts, "payload": payload}


def poa_verify(rec: dict, key: bytes | None = None, prev_chain: str | None = None) -> bool:
    try:
        h = sha256_hex(_canon(rec["payload"]))
        if h != rec["hash"]:
            return False
        if key is not None:
            exp = _hmac_hex(key, (prev_chain or "") + h)
            if not hmac.compare_digest(str(rec.get("chain") or ""), exp):
                return False
        return True
    except (KeyError, TypeError):
        return False


def poa_chain(records: list[dict], key: bytes | None = None,
              anchor: str | None = None) -> dict:
    prev = None
    prev_chain = ""
    bad = []
    for i, r in enumerate(records):
        if not poa_verify(r, key=key, prev_chain=prev_chain):
            bad.append((i, "hash" if not poa_verify(r) else "chain"))
        if (r.get("prev") or None) != prev:
            bad.append((i, "prev"))
        if r.get("seq") is not None and int(r["seq"]) != i + 1:
            bad.append((i, "seq"))
        prev = r.get("hash")
        if key is not None:
            prev_chain = str(r.get("chain") or "")
    out = {"ok": not bad, "n": len(records), "bad": bad, "head": prev}
    if anchor:
        out["anchor"] = verify_anchor(anchor, records)
        if not out["anchor"]["ok"]:
            out["ok"] = False
    return out


def write_anchor(path: str, records: list[dict], mode: int = 0o600) -> int:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, mode)
    try:
        for i, r in enumerate(records):
            seq = int(r.get("seq") or (i + 1))
            val = r.get("chain") or r.get("hash")
            os.write(fd, ("%d %s\n" % (seq, val)).encode("ascii"))
    finally:
        os.close(fd)
    try:
        os.chmod(path, mode)
    except OSError:
        pass
    return len(records)


def verify_anchor(path: str, records: list[dict]) -> dict:
    try:
        with open(path, encoding="ascii") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return {"ok": False, "reason": "missing", "n": 0}
    if len(lines) != len(records):
        return {"ok": False, "reason": "length", "n": len(lines), "expected": len(records)}
    for i, (ln, r) in enumerate(zip(lines, records)):
        parts = ln.split()
        if len(parts) != 2:
            return {"ok": False, "reason": "format", "first_bad": i + 1}
        seq = int(r.get("seq") or (i + 1))
        val = r.get("chain") or r.get("hash")
        if parts[0] != str(seq) or parts[1] != str(val):
            return {"ok": False, "reason": "mismatch", "first_bad": i + 1}
    return {"ok": True, "n": len(lines)}


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def fleet_determinism(traces: dict) -> dict:
    nodes = sorted(traces)
    if len(nodes) < 2:
        return {"n": len(nodes), "mean": None, "min": None, "std": None,
                "verdict": "INSUFFICIENT", "pairs": 0}
    vals = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            vals.append(_jaccard(traces[nodes[i]], traces[nodes[j]]))
    m = mean(vals)
    mn = min(vals)
    sd = pstdev(vals) if len(vals) > 1 else 0.0
    if m >= 0.95 and mn >= 0.80:
        v = "STABLE"
    elif m >= 0.80:
        v = "DEGRADED"
    else:
        v = "DIVERGENT"
    return {"n": len(nodes), "pairs": len(vals), "mean": round(m, 6),
            "min": round(mn, 6), "std": round(sd, 6), "verdict": v,
            "per_node": {k: len(v or []) for k, v in traces.items()}}


def fds_permutation(traces: dict, n_perm: int = 1000, seed: int = 7) -> dict:
    base = fleet_determinism(traces)
    if base["n"] < 2:
        return {**base, "p": None}
    nodes = sorted(traces)
    sizes = [len(traces[k]) for k in nodes]
    pool = [x for k in nodes for x in traces[k]]
    obs = base["mean"]
    rnd = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        rnd.shuffle(pool)
        pos = 0
        sh = {}
        for k, sz in zip(nodes, sizes):
            sh[k] = pool[pos:pos + sz]
            pos += sz
        if mean([_jaccard(sh[nodes[i]], sh[nodes[j]])
                 for i in range(len(nodes)) for j in range(i + 1, len(nodes))]) >= obs:
            ge += 1
    p = (ge + 1.0) / (n_perm + 1.0)
    return {**base, "p": round(p, 6)}


def semantic_drift(history: list[dict], now: float | None = None,
                   half_life: float = 7 * 86400.0) -> dict:
    now = time.time() if now is None else now
    latest = {}
    for entry in history or []:
        ts = float(entry.get("ts") or 0)
        for c in entry.get("claims") or []:
            key = str(c.get("key") or c.get("claim") or "")
            if not key:
                continue
            if key not in latest or ts >= float(latest[key][0].get("ts") or 0):
                latest[key] = (entry, c)
    if not latest:
        return {"drift": None, "contradiction_rate": None, "stale_ratio": None,
                "n": 0, "verdict": "NO-DATA"}
    q = []
    contra = 0
    stale = 0
    for entry, c in latest.values():
        ts = float(entry.get("ts") or 0)
        age = max(0.0, now - ts)
        fresh = math.exp(-math.log(2.0) * age / half_life)
        if fresh < 0.5:
            stale += 1
        if c.get("contradicted"):
            contra += 1
        conf = float(c.get("confidence", 1.0 if c.get("supported") else 0.0))
        good = 0.0 if c.get("contradicted") else conf
        q.append(good * fresh)
    drift = 1.0 - mean(q)
    cr = contra / float(len(latest))
    sr = stale / float(len(latest))
    if drift <= 0.20:
        v = "CONSISTENT"
    elif drift <= 0.50:
        v = "WATCH"
    else:
        v = "DRIFT"
    return {"drift": round(drift, 6), "contradiction_rate": round(cr, 6),
            "stale_ratio": round(sr, 6), "n": len(latest), "verdict": v}


def verdict(poa_ok: bool, fds: dict | None = None, sdm: dict | None = None,
            inference: dict | None = None, requirement: str = "VERSIONED") -> dict:
    if not poa_ok:
        return {"verdict": "UNPROVEN", "reason": "no valid PoA"}
    f = (fds or {}).get("mean")
    d = (sdm or {}).get("drift")
    inf_level = None
    if isinstance(inference, dict):
        inf_level = inference.get("level")
    if inf_level is not None and not required_ok(str(inf_level), requirement):
        return {"verdict": "DOUBTFUL", "fds": f, "drift": d, "inference": inf_level,
                "reason": "inference level %s < required %s" % (inf_level, requirement)}
    if (f is not None and f < 0.80) or (d is not None and d > 0.50):
        return {"verdict": "DOUBTFUL", "fds": f, "drift": d, "inference": inf_level}
    if (f is not None and f < 0.95) or (d is not None and d > 0.20):
        return {"verdict": "DEGRADED", "fds": f, "drift": d, "inference": inf_level}
    return {"verdict": "TRUSTED", "fds": f, "drift": d, "inference": inf_level}
