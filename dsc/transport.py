"""Transport (R2): one call surface over DSC services.

The whole system is reachable through the worker (:8790) which already proxies
/core, /dag, /orch and /sem; /ask adds model answers. This module is the single
`call()` used by the language layer and by the CLI - one place for timeouts,
error normalisation and trace ids.

Design rules (from the gap analysis):
  * one transport layer, N backends - never one client per endpoint;
  * protocol error (bad request/unknown cap) vs execution error (tool failed) are
    different outcomes and must not be merged;
  * `degraded` (core offline) is reported as data, not raised as an exception.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.error
import urllib.request
import uuid
from typing import Any

WORKER_URL = os.environ.get("DSC_WORKER_URL", "http://127.0.0.1:8790").rstrip("/")
DEFAULT_TIMEOUT = float(os.environ.get("DSC_CALL_TIMEOUT", "120"))

# cap name -> (path, payload key)
CAPS: dict[str, str] = {
    "core": "/core",
    "dag": "/dag",
    "orch": "/orch",
    "sem": "/sem",
    "ask": "/ask",
}

PROTOCOL_ERRORS = {"unknown_cap", "bad_payload"}
EXECUTION_ERRORS = {"upstream_error", "timeout", "unavailable"}


class CapError(Exception):
    """Raised only for programmer errors (unknown cap / bad payload shape)."""


def _encode(cap: str, payload: dict[str, Any]) -> str:
    if cap == "core":
        return "/core?" + urllib.parse.urlencode({"q": payload.get("q", "")})
    if cap == "ask":
        return "/ask?" + urllib.parse.urlencode({"q": payload.get("q", "")})
    if cap == "dag":
        q = {"goal": payload.get("goal", ""), "levels": payload.get("levels", "2,2")}
        return "/dag?" + urllib.parse.urlencode(q)
    if cap == "orch":
        return "/orch?" + urllib.parse.urlencode({"path": payload.get("path", "/jobs")})
    if cap == "sem":
        q = {"q": payload.get("q", ""), "k": payload.get("k", 5)}
        return "/sem?" + urllib.parse.urlencode(q)
    raise CapError("unknown_cap: %s" % cap)


def _post(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "dsc-transport/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def call(
    cap: str,
    payload: dict[str, Any] | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
    fetch: Any = None,
) -> dict:
    """Call a DSC capability and return a normalised envelope.

    Result: {ok, cap, data, error, kind, via, ms, trace_id}
      kind in {"", "protocol", "execution"}.
    """
    if cap not in CAPS:
        raise CapError("unknown_cap: %s (dozvoljeno: %s)" % (cap, ",".join(sorted(CAPS))))
    body = payload or {}
    path = _encode(cap, body)
    url = (base_url or WORKER_URL) + path
    trace_id = uuid.uuid4().hex[:12]
    started = time.time()
    fn = fetch or _post

    try:
        raw = fn(url, timeout or DEFAULT_TIMEOUT)
        ok = bool(raw.get("ok", True))
        data = {k: v for k, v in raw.items() if k != "ok"}
        kind = "" if ok else "execution"
        return {
            "ok": ok,
            "cap": cap,
            "data": data,
            "error": None if ok else (data.get("err") or "upstream_error"),
            "kind": kind,
            "via": url,
            "ms": int((time.time() - started) * 1000),
            "trace_id": trace_id,
        }
    except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
        return {
            "ok": False,
            "cap": cap,
            "data": {},
            "error": "http_%s" % exc.code,
            "kind": "protocol" if 400 <= exc.code < 500 else "execution",
            "via": url,
            "ms": int((time.time() - started) * 1000),
            "trace_id": trace_id,
        }
    except Exception as exc:  # noqa: BLE001 - transport never raises on I/O
        return {
            "ok": False,
            "cap": cap,
            "data": {},
            "error": repr(exc)[:160],
            "kind": "execution",
            "via": url,
            "ms": int((time.time() - started) * 1000),
            "trace_id": trace_id,
        }


def describe() -> dict:
    return {
        "worker": WORKER_URL,
        "caps": sorted(CAPS),
        "default_timeout_s": DEFAULT_TIMEOUT,
        "error_kinds": {"protocol": sorted(PROTOCOL_ERRORS), "execution": sorted(EXECUTION_ERRORS)},
    }
