"""Deterministic retrieval/classification model trained offline (no GPU, no LLM).

DSC-DSC model = BM25 stats + int8 HDC document vectors + per-category centroids.
It is a real trainable model: fit() on a corpus, save()/load() JSON, predict offline.
"""
from __future__ import annotations

import base64
import json
import os

from . import hdc
from .fusion import wrrf
from .lexical import BM25Index


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


class DeterministicModel:
    def __init__(self, k1: float = 1.5, b: float = 0.75, dim: int = hdc.DIM):
        self.k1 = k1
        self.b = b
        self.dim = dim
        self.docs: list[dict] = []
        self.vectors: dict[str, bytes] = {}
        self.centroids: dict[str, bytes] = {}
        self.bm: BM25Index | None = None
        self.meta: dict = {}

    def fit(self, docs: list[dict]) -> "DeterministicModel":
        self.docs = [{"id": str(d["id"]), "text": str(d.get("text", "")),
                      "cat": str(d.get("cat", "general"))} for d in docs]
        self.bm = BM25Index(self.k1, self.b)
        sums: dict[str, list[float]] = {}
        for d in self.docs:
            self.bm.add(d["id"], d["text"])
            v = hdc.vector(d["text"], self.dim)
            self.vectors[d["id"]] = v
            raw = [x / 127.0 for x in hdc.unquantize(v)]
            acc = sums.setdefault(d["cat"], [0.0] * len(raw))
            for i, x in enumerate(raw):
                acc[i] += x
        self.bm.finalize()
        for cat, acc in sums.items():
            self.centroids[cat] = hdc.quantize(acc)
        self.meta = {"docs": len(self.docs), "cats": sorted(self.centroids),
                     "dim": self.dim, "model": "DSC-DSC-0.1"}
        return self

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.bm:
            return []
        lex = [r["id"] for r in self.bm.search(query, k=len(self.docs))]
        qv = hdc.vector(query, self.dim)
        hdcs = sorted(((hdc.cosine_q(qv, self.vectors[d["id"]]), d["id"]) for d in self.docs),
                      key=lambda x: (-x[0], x[1]))
        fused = wrrf([lex, [i for _, i in hdcs]], [1.0, 0.8], k=10)
        by_id = {d["id"]: d for d in self.docs}
        out = []
        for did, sc in sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]:
            out.append({"id": did, "text": by_id[did]["text"], "cat": by_id[did]["cat"],
                        "fused": round(sc, 6), "cos": round(hdc.cosine_q(qv, self.vectors[did]), 6)})
        return out

    def classify(self, query: str) -> dict:
        if not self.centroids:
            return {"cat": None, "score": 0.0}
        qv = hdc.vector(query, self.dim)
        best = max(((hdc.cosine_q(qv, v), c) for c, v in self.centroids.items()),
                   key=lambda x: (x[0], x[1]))
        return {"cat": best[1], "score": round(best[0], 6)}

    def save(self, path: str) -> int:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {"meta": self.meta, "docs": self.docs,
                   "vectors": {k: _b64(v) for k, v in self.vectors.items()},
                   "centroids": {k: _b64(v) for k, v in self.centroids.items()},
                   "k1": self.k1, "b": self.b, "dim": self.dim}
        blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(blob)
        return len(blob)

    @classmethod
    def load(cls, path: str) -> "DeterministicModel":
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        m = cls(p.get("k1", 1.5), p.get("b", 0.75), int(p.get("dim", hdc.DIM)))
        m.meta = p.get("meta", {})
        m.docs = p.get("docs", [])
        m.vectors = {k: _unb64(v) for k, v in (p.get("vectors") or {}).items()}
        m.centroids = {k: _unb64(v) for k, v in (p.get("centroids") or {}).items()}
        m.bm = BM25Index(m.k1, m.b)
        for d in m.docs:
            m.bm.add(d["id"], d["text"])
        m.bm.finalize()
        return m

    def stats(self) -> dict:
        return {"meta": self.meta, "docs": len(self.docs), "cats": len(self.centroids),
                "dim": self.dim, "trained": bool(self.bm)}
