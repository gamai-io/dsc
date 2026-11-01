"""Optional backend discovery (polymorphic contract): offline HDC is always available;
bge-m3 / NLI are attached only if their health probe passes."""
from __future__ import annotations

import json
import urllib.request

from .verify import embed_http, nli_http


def _probe(url: str, timeout: float = 1.2) -> bool:
    if not url:
        return False
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= int(r.status) < 300
    except Exception:
        return False


def auto(embed_url: str | None = None, nli_url: str | None = None,
         embed_model: str = "bge-m3") -> dict:
    out = {"embed_fn": None, "nli_fn": None, "sources": [], "offline": True}
    if embed_url:
        base = embed_url.rstrip("/")
        health = base[:-len("/v1/embeddings")] + "/health" if base.endswith("/v1/embeddings") else base + "/health"
        if _probe(health):
            out["embed_fn"] = embed_http(embed_url, model=embed_model)
            out["sources"].append("embed:" + embed_url)
    if nli_url:
        base = nli_url.rstrip("/")
        if _probe(base + "/health"):
            out["nli_fn"] = nli_http(base)
            out["sources"].append("nli:" + nli_url)
    out["offline"] = out["embed_fn"] is None and out["nli_fn"] is None
    return out


def describe() -> str:
    return json.dumps({"offline_core": ["normalize", "simhash", "minhash", "hdc", "bm25", "hardcheck", "cache"],
                       "optional": ["bge-m3 embedding", "mDeBERTa NLI", "bge-reranker", "Qdrant"],
                       "contract": "Core(embed_fn=..., nli_fn=...) + SemanticCache(embed_fn=...)"},
                      ensure_ascii=False)
