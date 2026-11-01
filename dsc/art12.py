"""EU AI Act Article 12 - record-keeping fields for DSC receipts (R5).

Art. 12(3) requires, at minimum, logging of:
  (a) start and end date/time of each use,
  (b) the reference database against which input data was checked,
  (c) the input data that produced an output,
  (d) identification of the persons involved in verification.

We do NOT claim "Art. 12 compliant" - that is a legal statement. We provide the
technical fields so an auditor can check them, and we hash them into the receipt.

Design: `record()` returns a plain dict with only canonicalisable values
(str / int / None / list / dict), so it can be embedded in a signed receipt.
"""
from __future__ import annotations

import time
from typing import Any

from . import jcs

REQUIRED = (
    "session_start",
    "session_end",
    "reference_db",
    "input_data_hash",
    "verifier_person",
)


def _iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts if ts is not None else time.time()))


def record(
    input_data: Any,
    reference_db: str,
    verifier_person: str,
    *,
    started: float | None = None,
    ended: float | None = None,
) -> dict:
    """Build the Art. 12 minimum block.

    input_data is hashed with JCS - never stored raw (privacy + size).
    """
    if not reference_db or not verifier_person:
        raise ValueError("reference_db i verifier_person su obavezni")
    start = started if started is not None else time.time()
    end = ended if ended is not None else time.time()
    if end < start:
        raise ValueError("session_end ne moze biti prije session_start")
    return {
        "session_start": _iso(start),
        "session_end": _iso(end),
        "reference_db": str(reference_db),
        "input_data_hash": jcs.digest(input_data or {}),
        "verifier_person": str(verifier_person),
    }


def validate(block: dict | None) -> dict:
    """Check an Art. 12 block. Returns {ok, missing[], invalid[]}."""
    if not isinstance(block, dict):
        return {"ok": False, "missing": list(REQUIRED), "invalid": ["not_a_dict"]}
    missing = [k for k in REQUIRED if not block.get(k)]
    invalid: list[str] = []
    for key in ("session_start", "session_end"):
        val = block.get(key)
        if val and not (isinstance(val, str) and len(val) >= 19 and val.endswith("Z")):
            invalid.append(key)
    h = block.get("input_data_hash")
    if h and not (isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)):
        invalid.append("input_data_hash")
    return {"ok": not missing and not invalid, "missing": missing, "invalid": invalid}


def attach(receipt: dict, block: dict) -> dict:
    """Validate and attach the block under receipt['payload']['extra']['art12']."""
    check = validate(block)
    if not check["ok"]:
        raise ValueError("neispravan Art.12 blok: %s" % check)
    payload = receipt.setdefault("payload", {})
    extra = payload.setdefault("extra", {})
    extra["art12"] = block
    return receipt
