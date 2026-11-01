"""One facade: analyze / search / verify / remember / recall / detect."""
from __future__ import annotations

import hashlib
import os

from . import hdc
from .cache import SemanticCache
from .detector import fleet_determinism, poa_chain, poa_record, semantic_drift, verdict
from .fingerprints import jaccard_est, minhash128, sha256_hex, simhash64
from .fusion import wrrf
from .lexical import BM25Index
from .nondeterminism import Envelope, describe as determinism_describe, guard, receipt
from .normalize import features
from .verify import verify_answer, verify_claim


class Core:
    def __init__(self, cache_path: str | None = None, embed_fn=None, nli_fn=None,
                 dim: int = hdc.DIM, cache_threshold: float | None = None,
                 cache_enabled: bool = True):
        self.embed_fn = embed_fn
        self.nli_fn = nli_fn
        self.dim = dim
        self.cache = SemanticCache(cache_path, embed_fn=embed_fn, dim=dim,
                                   threshold=cache_threshold, enabled=cache_enabled)
        self._vec = {}

    def _doc_vec(self, doc_id, text: str) -> bytes:
        key = (doc_id, sha256_hex(text))
        v = self._vec.get(key)
        if v is None:
            v = hdc.vector(text, self.dim)
            if len(self._vec) > 20000:
                self._vec.clear()
            self._vec[key] = v
        return v

    def analyze(self, text: str) -> dict:
        f = features(text)
        mh = minhash128(text)
        v = hdc.vector(text, self.dim)
        return {
            "canonical": f["canonical"],
            "tokens": f["tokens"],
            "content": f["content"],
            "numbers": f["numbers"],
            "dates": f["dates"],
            "years": f["years"],
            "negations": f["negations"],
            "flags": {"action": f["action"], "fresh": f["fresh"], "multi": f["multi"]},
            "sha256": sha256_hex(f["canonical"]),
            "simhash64": simhash64(text),
            "minhash128": list(mh),
            "vector_dim": len(v),
            "vector_int8_sha": hashlib.sha256(v).hexdigest()[:16],
        }

    def search(self, query: str, docs: list, k: int = 5, embed_weight: float = 1.0) -> list[dict]:
        norm = []
        for i, d in enumerate(docs):
            if isinstance(d, str):
                norm.append({"id": str(i), "text": d})
            else:
                norm.append({"id": str(d.get("id", i)), "text": str(d.get("text") or d.get("content") or "")})
        if not norm:
            return []
        bm = BM25Index()
        for d in norm:
            bm.add(d["id"], d["text"], d)
        bm.finalize()
        lex = bm.search(query, k=len(norm))
        lex_ids = [r["id"] for r in lex]
        qv = hdc.vector(query, self.dim)
        hs = []
        for d in norm:
            hs.append((hdc.cosine_q(qv, self._doc_vec(d["id"], d["text"])), d["id"]))
        hs.sort(key=lambda x: (-x[0], x[1]))
        hdc_ids = [i for c, i in hs if c > 0.05]
        lists = [lex_ids, hdc_ids]
        weights = [1.0, 0.8]
        emb_scores = {}
        if self.embed_fn is not None:
            try:
                vs = self.embed_fn([query] + [d["text"] for d in norm])
                qe = vs[0]
                for d, ve in zip(norm, vs[1:]):
                    dot = sum(x * y for x, y in zip(qe, ve))
                    na = sum(x * x for x in qe) ** 0.5
                    nb = sum(x * x for x in ve) ** 0.5
                    emb_scores[d["id"]] = dot / (na * nb) if na and nb else 0.0
                emb_ids = [i for i, _ in sorted(emb_scores.items(), key=lambda x: (-x[1], x[0]))]
                lists.append(emb_ids)
                weights.append(embed_weight)
            except Exception:
                pass
        fused = wrrf(lists, weights, k=10)
        by_id = {d["id"]: d for d in norm}
        rank = {i: r for r, i in enumerate(hdc_ids)}
        emb_rank = {i: r for r, i in enumerate(emb_scores)}
        out = []
        for did, score in sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]:
            d = by_id[did]
            out.append({"id": did, "text": d["text"], "fused": round(score, 6),
                        "cos": round(hdc.cosine_q(qv, self._doc_vec(did, d["text"])), 6),
                        "embed_cos": round(emb_scores[did], 6) if did in emb_scores else None,
                        "rank": rank.get(did), "embed_rank": emb_rank.get(did),
                        "meta": d.get("meta")})
        return out

    def verify(self, answer: str, sources: list, inference: dict | None = None,
               accounting: bool = False) -> dict:
        src = []
        for i, s in enumerate(sources):
            if isinstance(s, str):
                src.append({"id": i + 1, "text": s})
            else:
                src.append({"id": s.get("id", i + 1), "text": s.get("text", ""),
                            "trust": s.get("trust")})
        out = verify_answer(answer, src, embed_fn=self.embed_fn, nli_fn=self.nli_fn,
                            accounting=accounting)
        if inference is not None:
            out["inference"] = inference
        return out

    def verify_one(self, claim: str, sources: list, accounting: bool = False) -> dict:
        src = [{"id": i + 1, "text": s if isinstance(s, str) else s.get("text", "")}
               for i, s in enumerate(sources)]
        return verify_claim(claim, src, embed_fn=self.embed_fn, nli_fn=self.nli_fn,
                            accounting=accounting)

    def remember(self, query: str, answer: str, inference: dict | None = None, **kw) -> bool:
        if inference is not None and "envelope_id" not in kw:
            kw["envelope_id"] = inference.get("envelope_id")
        if inference is not None and "inference_level" not in kw:
            kw["inference_level"] = inference.get("level")
        return self.cache.put(query, answer, **kw)

    def recall(self, query: str, inference: dict | None = None, **kw) -> dict | None:
        if inference is not None and "envelope_id" not in kw:
            kw["envelope_id"] = inference.get("envelope_id")
        return self.cache.get(query, **kw)

    def invalidate(self, domain: str | None = None) -> int:
        return self.cache.invalidate(domain)

    def stats(self) -> dict:
        return self.cache.stats()

    def evidence(self, kind: str, **kw) -> dict:
        return poa_record(kind, **kw)

    def attest(self, prompt: str, output: str, env: Envelope) -> dict:
        return receipt(prompt, output, env)

    def determinism(self) -> dict:
        return determinism_describe()

    def inference_guard(self, env: Envelope, requirement: str = "VERSIONED") -> dict:
        return guard(env, requirement)

    def evidence_chain(self, records: list) -> dict:
        return poa_chain(records)

    def fleet(self, traces: dict) -> dict:
        return fleet_determinism(traces)

    def drift(self, history: list, now: float | None = None) -> dict:
        return semantic_drift(history, now=now)

    def verdict(self, poa_ok: bool, fds: dict | None = None, sdm: dict | None = None,
                inference: dict | None = None, requirement: str = "VERSIONED") -> dict:
        return verdict(poa_ok, fds, sdm, inference=inference, requirement=requirement)

    def close(self) -> None:
        self.cache.close()
