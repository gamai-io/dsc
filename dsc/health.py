"""Health negotiation for DSC L 2.0 (R6).

Principle: `dsc_core` being offline is a NORMAL state, not a failure.
  * all infra up + core ok      -> status "ok"        (exit 0)
  * all infra up + core failing -> status "degraded"  (exit 75)
  * any infra down              -> status "down"      (exit 1)

No third-party deps; every probe is injectable so tests never touch the network.
"""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.request
from typing import Callable, Iterable, Sequence

EXIT_OK = 0
EXIT_DEGRADED = 75
EXIT_DOWN = 1

Probe = Callable[[str, str, float], "tuple[bool, str]"]

def default_probes() -> tuple[tuple[str, str, str], ...]:
    """Probes derived from env so the same code works on every node.

    DSC_WORKER_URL   default http://127.0.0.1:8790
    DSC_HARNESS_URL  default http://127.0.0.1:7125
    DSC_HC_TARGET    default 127.0.0.1:7100
    DSC_ENGINE_TARGET default 127.0.0.1:8040 (engine host may live on another node)
    """
    worker = os.environ.get("DSC_WORKER_URL", "http://127.0.0.1:8790").rstrip("/")
    harness = os.environ.get("DSC_HARNESS_URL", "http://127.0.0.1:7125").rstrip("/")
    hc = os.environ.get("DSC_HC_TARGET", "127.0.0.1:7100")
    engine = os.environ.get("DSC_ENGINE_TARGET", "127.0.0.1:8040")
    return (
        ("worker", worker + "/health", "http"),
        ("harness", harness + "/health", "http"),
        ("hc", hc, "tcp"),
        ("engine", engine, "tcp"),
    )

def default_core_probe() -> tuple[str, str, str]:
    worker = os.environ.get("DSC_WORKER_URL", "http://127.0.0.1:8790").rstrip("/")
    return ("core", worker + "/core?q=ping", "http")


def _probe_tcp(target: str, timeout: float) -> tuple[bool, str]:
    host, _, port = target.rpartition(":")
    if not host or not port.isdigit():
        return False, "neispravan target"
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True, "tcp ok"
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        return False, repr(exc)[:120]


def _probe_http(url: str, timeout: float) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "dsc-health/1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(512).decode("utf-8", "replace")
            return 200 <= int(r.status) < 300, body[:120]
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)[:120]


def probe(target: str, kind: str, timeout: float) -> tuple[bool, str]:
    """Run a single probe; returns (ok, detail). Never raises."""
    if kind == "tcp":
        return _probe_tcp(target, timeout)
    return _probe_http(target, timeout)


def _load_extra_probes() -> Sequence[tuple[str, str, str]]:
    """Optional JSON list in DSC_DSC_PROBES: [[name, target, kind], ...]."""
    raw = os.environ.get("DSC_DSC_PROBES", "").strip()
    if not raw:
        return ()
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return ()
    out: list[tuple[str, str, str]] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            out.append((str(item[0]), str(item[1]), str(item[2])))
    return tuple(out)


def health(
    probes: Iterable[tuple[str, str, str]] | None = None,
    core_probe: tuple[str, str, str] | None = None,
    timeout: float = 2.0,
    probe_fn: Probe | None = None,
    core_timeout: float | None = None,
) -> dict:
    """Collect probe results and derive a single status.

    Returns dict: {status, exit_code, degraded, checks[], core{}, ms}.
    """
    fn = probe_fn or probe
    started = time.time()
    items = tuple(probes if probes is not None else default_probes()) + _load_extra_probes()
    core = core_probe if core_probe is not None else default_core_probe()

    checks: list[dict] = []
    for name, target, kind in items:
        ok, detail = fn(target, kind, timeout)
        checks.append({"name": name, "kind": kind, "target": target, "ok": ok, "detail": detail})

    core_ok, core_detail = fn(core[1], core[2], core_timeout or max(timeout * 6, 12.0))
    core_state = {
        "name": core[0],
        "kind": core[2],
        "target": core[1],
        "ok": core_ok,
        "detail": core_detail,
    }

    infra_down = [c["name"] for c in checks if not c["ok"]]
    if infra_down:
        status, code = "down", EXIT_DOWN
    elif not core_ok:
        status, code = "degraded", EXIT_DEGRADED
    else:
        status, code = "ok", EXIT_OK

    return {
        "status": status,
        "exit_code": code,
        "degraded": status == "degraded",
        "down": infra_down,
        "checks": checks,
        "core": core_state,
        "ms": int((time.time() - started) * 1000),
    }


def describe() -> dict:
    """Static description (no probing) - used by CLI --help and docs."""
    return {
        "probes": [{"name": n, "target": t, "kind": k} for n, t, k in default_probes()],
        "core_probe_name": default_core_probe()[0],
        "exit_codes": {"ok": EXIT_OK, "degraded": EXIT_DEGRADED, "down": EXIT_DOWN},
        "rule": "dsc_core offline => degraded (75), not failure",
    }
