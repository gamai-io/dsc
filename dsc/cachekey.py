"""One cache key for the whole deterministic pre-LLM stack.

Every layer used to hash its own tuple of values:

* ``dsc.cache.SemanticCache._exact_key`` — ``sha256("sid|%s|%s%s")``, i.e. raw
  ``|`` concatenation of session, canonical text and a ``|env:`` suffix;
* ``gama-fastpath.cache_key`` — ``sha256`` of a canonical JSON payload of
  ``{"v","t","s"}``;
* ``gama-offline.cache._key`` — ``md5("sid|norm(q)")``, no tenant, no schema.

They disagreed on hash, scope and framing.  ``request_key`` is the single
implementation: a SHA-256 over one canonical JSON object, with an explicit
scheme prefix so an old key space can be recognised (and safely missed)
instead of silently reused.

Contract
--------
``request_key(text, normalized=None, tenant="", session="", context=None)``

* returns ``"v2:" + 64 lowercase hex chars`` (SHA-256);
* the hashed payload is the JSON object
  ``{"v": "v2", "t": <text>, "tenant": ..., "session": ..., "ctx": ...}``
  serialised with ``sort_keys=True``, ``separators=(",", ":")`` and UTF-8;
* every caller-controlled field is a *value*, never concatenated into a
  delimiter string, so ``tenant="a|b", session="c"`` and
  ``tenant="a", session="b|c"`` are different keys;
* ``tenant`` and ``session`` default to ``""`` (isolated namespaces, not a
  wildcard);
* ``context`` must be JSON-serialisable; use it for envelope ids, domains or
  service namespaces, not for free-form prose;
* the same arguments always produce the same key, in any process and under
  any ``PYTHONHASHSEED``.

Normalisation
-------------
``normalized`` lets a caller keep its existing text normalisation (dsc's
``canonical``, offline's lowercase).  When omitted, ``normalize_for_key`` is
used: diacritics folded, lowercased, whitespace collapsed.  It deliberately
does **not** canonicalise numbers — ``"neto 2400"`` and ``"neto 2.400"`` stay
distinct, because in a money field those are different claims (see
``dsc.amount.parse_amount``) and a cache hit must not merge them.

Legacy keys
-----------
There is no legacy reader on purpose.  A key that does not start with the
current ``SCHEME`` can never be produced by ``request_key``, so old rows simply
miss and are ignored until TTL/eviction removes them.  The first run after the
switch therefore has a 0% hit rate: that is the expected, safe outcome, not a
regression.
"""
from __future__ import annotations

import json
import re

from .fingerprints import sha256_hex
from .normalize import fold

SCHEME = "v2"
_HEX_LEN = 64
_WS = re.compile(r"\s+")


def normalize_for_key(text: str) -> str:
    """Fold diacritics, lowercase, collapse whitespace.  Numbers untouched."""
    return _WS.sub(" ", fold(text or "").lower().strip())


def request_key(text: str, normalized: str | None = None, tenant: str = "",
                session: str = "", context=None) -> str:
    """Return the scheme-prefixed cache key for one full request.

    ``text`` is the raw request; ``normalized`` is an optional pre-normalised
    form supplied by the caller.  ``tenant``/``session`` scope the entry;
    ``context`` is any JSON-serialisable extra (envelope, domain, service).
    """
    payload = {
        "v": SCHEME,
        "t": normalize_for_key(text) if normalized is None else str(normalized),
        "tenant": str(tenant or ""),
        "session": str(session or ""),
        "ctx": context,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
    return "%s:%s" % (SCHEME, sha256_hex(blob))


def is_current(key: str) -> bool:
    """True when ``key`` was produced by this scheme (and can be trusted)."""
    k = str(key or "")
    if not k.startswith(SCHEME + ":") or len(k) != len(SCHEME) + 1 + _HEX_LEN:
        return False
    return all(c in "0123456789abcdef" for c in k[len(SCHEME) + 1:])
