"""Deterministic batch protocol + end-to-end measurement (R7).

Why this exists: the Rust sidecar reaches ~38k req/s, but that number is
meaningless if the Python side reorders requests, loses the seed, or measures
the wrong thing. This module owns the *boundary*:

  pack(requests, seed)   -> canonical NDJSON bytes (same input => same bytes)
  verify(payload, drops) -> ids complete, no duplicates, hashes match, order by id
  bench(...)             -> end-to-end p50/p99 measured on the client side

Determinism rules:
  * header carries "seed" and item count; without it a replay is not comparable;
  * every item is JCS-encoded (sorted keys) so the digest is stable;
  * results are ordered by id, never by arrival time.
"""
from __future__ import annotations

import hashlib
import json
import socket
import statistics
import time
from typing import Any, Iterable

from . import jcs

PROTO_VERSION = 1


def _digest(obj: Any) -> str:
    return hashlib.sha256(jcs.to_bytes(obj)).hexdigest()


def pack(requests: Iterable[dict], seed: int = 0) -> bytes:
    """Canonical batch payload: header line + one JSON line per request."""
    items = []
    for idx, req in enumerate(requests, start=1):
        item = {"id": idx, "op": str(req.get("op", "echo")), "args": req.get("args") or {}}
        item["hash"] = _digest({"id": item["id"], "op": item["op"], "args": item["args"]})
        items.append(item)
    header = {"v": PROTO_VERSION, "kind": "batch", "n": len(items), "seed": int(seed)}
    lines = [jcs.dumps(header)] + [jcs.dumps(i) for i in items]
    return ("\n".join(lines) + "\n").encode("utf-8")


def unpack(payload: bytes) -> dict:
    lines = [ln for ln in payload.decode("utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("prazan batch")
    header = json.loads(lines[0])
    items = [json.loads(ln) for ln in lines[1:]]
    if header.get("kind") != "batch" or header.get("v") != PROTO_VERSION:
        raise ValueError("nepodrzan batch header")
    if len(items) != header.get("n"):
        raise ValueError("n se ne poklapa: %s/%s" % (len(items), header.get("n")))
    return {"header": header, "items": items, "sha256": hashlib.sha256(payload).hexdigest()}


def verify(payload: bytes, results: list[dict]) -> dict:
    """Check completeness, duplicates, per-item hash and ordering."""
    parsed = unpack(payload)
    n = parsed["header"]["n"]
    by_id: dict[int, dict] = {}
    dupes: list[int] = []
    for res in results:
        rid = res.get("id")
        if rid in by_id:
            dupes.append(rid)
        by_id[rid] = res
    missing = [i for i in range(1, n + 1) if i not in by_id]
    bad_hash = []
    for item in parsed["items"]:
        res = by_id.get(item["id"])
        if res is None:
            continue
        expect = _digest({"id": item["id"], "op": item["op"], "args": item["args"]})
        if item.get("hash") != expect or res.get("input_hash") != expect:
            bad_hash.append(item["id"])
    ordered = [by_id[i] for i in sorted(by_id)]
    return {
        "ok": not (missing or dupes or bad_hash),
        "n": n,
        "missing": missing,
        "dupes": dupes,
        "bad_hash": bad_hash,
        "seed": parsed["header"]["seed"],
        "payload_sha256": parsed["sha256"],
        "ordered": ordered,
    }


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    vals = sorted(values)
    def pick(q: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
        return round(vals[idx], 3)
    return {"n": len(vals), "p50": pick(0.50), "p90": pick(0.90), "p99": pick(0.99),
            "mean": round(statistics.fmean(vals), 3), "max": round(vals[-1], 3)}


def echo_sidecar(handler) -> None:
    """Minimal NDJSON server used by the bench (mirrors the Rust sidecar shape)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        print(json.dumps({"ready": port}), flush=True)
        conn, _ = srv.accept()
        with conn:
            f = conn.makefile("rwb")
            header_line = f.readline()
            if not header_line:
                return
            header = json.loads(header_line)
            for _ in range(int(header.get("n", 0))):
                line = f.readline()
                if not line:
                    break
                req = json.loads(line)
                resp = handler(req)
                f.write((jcs.dumps(resp) + "\n").encode("utf-8"))
                f.flush()


def bench(requests: list[dict], port: int, seed: int = 0, timeout: float = 30.0) -> dict:
    """Send one batch, measure per-request latency end-to-end, verify results."""
    payload = pack(requests, seed=seed)
    started = time.perf_counter()
    lat: list[float] = []
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.sendall(payload)
        f = sock.makefile("rb")
        results = []
        while len(results) < len(requests):
            line = f.readline()
            if not line:
                break
            lat.append((time.perf_counter() - started) * 1000)  # client-observed e2e
            results.append(json.loads(line))
    total_ms = (time.perf_counter() - started) * 1000
    check = verify(payload, results)
    return {
        "ok": check["ok"],
        "sent": len(requests),
        "received": len(results),
        "seed": seed,
        "payload_bytes": len(payload),
        "payload_sha256": check["payload_sha256"],
        "total_ms": round(total_ms, 2),
        "req_per_s": round(len(results) / max(total_ms / 1000.0, 1e-9), 1),
        "latency": percentiles(lat),
        "missing": check["missing"][:5],
    }
