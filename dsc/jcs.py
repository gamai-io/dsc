"""RFC 8785 JCS (JSON Canonicalization Scheme) - deterministic subset.

Why: every receipt is hashed and (optionally) signed. Without one canonical
form, two nodes produce different bytes for the same data and signatures never
verify. RFC 8785 defines that form; we implement the subset we actually use.

Supported:  None, bool, int, str, list/tuple, dict[str, ...]
Rejected:   float - RFC 8785 Appendix D requires money/large numbers to be
            strings, and binary floats are not usable as signed values.

Known, documented deviation: RFC 8785 sorts property names by UTF-16 code units;
we sort by Unicode code points (identical for ASCII keys, which is all we use).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


class JCSError(ValueError):
    """Raised when a value cannot be canonicalised without ambiguity."""


def _encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        raise JCSError("float nije dozvoljen u potpisanom sadrzaju; koristi int ili decimalni string")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode(v) for v in value) + "]"
    if isinstance(value, dict):
        for k in value:
            if not isinstance(k, str):
                raise JCSError("kljuc mora biti string: %r" % (k,))
        parts = []
        for k in sorted(value):
            parts.append(json.dumps(k, ensure_ascii=False) + ":" + _encode(value[k]))
        return "{" + ",".join(parts) + "}"
    raise JCSError("nepodrzan tip: %s" % type(value).__name__)


def dumps(value: Any) -> str:
    """Canonical JSON text (no whitespace, sorted keys, UTF-8-friendly)."""
    return _encode(value)


def to_bytes(value: Any) -> bytes:
    return dumps(value).encode("utf-8")


def digest(value: Any, algo: str = "sha256") -> str:
    """Hex digest over canonical bytes - the only correct way to hash JSON."""
    h = hashlib.new(algo)
    h.update(to_bytes(value))
    return h.hexdigest()


def loads(text: str | bytes) -> Any:
    """Parse canonical JSON back to Python (inverse of dumps for our subset)."""
    return json.loads(text)


def check(value: Any) -> None:
    """Raise JCSError if the value is not canonicalisable."""
    _encode(value)
