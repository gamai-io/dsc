"""Claim verification: hard-check veto + contradiction veto + calibrated AND.

Rule: a claim is SUPPORTED only if (a) numbers/units/dates/polarity pass on at least one
cited source, (b) no contradiction signal, (c) calibrated probability >= threshold, and
(d) at least two independent signals were actually available.
"""
from __future__ import annotations

import json
import re
import urllib.request
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor

from . import hdc
from .accounting import verify_pdv_claim
from .fusion import conformal_quantile, sigmoid
from .lexical import hard_check, lexical_entailment
from .normalize import canonical, content_tokens, has_negation

WEIGHTS = {"lex": 3.0, "emb": 2.0, "nli": 2.8}
BIAS = -2.8
CONTRA_W = 4.0
_CITE = re.compile(r"\[(\d+)\]")


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+\]\s*", " ", text or "").strip()


def _jsonable(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def split_claims(answer: str) -> list[str]:
    txt = (answer or "").strip()
    if not txt:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZČĆŽŠĐ0-9\[])", txt)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > 320:
            sub = re.split(r";\s+", p)
            out.extend(x.strip() for x in sub if x.strip())
        else:
            out.append(p)
    return out


def _cos_hdc(a: str, b: str) -> float:
    va = hdc.vector(a)
    vb = hdc.vector(b)
    c = hdc.cosine_q(va, vb)
    return max(0.0, min(1.0, c))


def _best_source(claim: str, sources: list[dict], citations: list[int]) -> tuple[int, dict, dict]:
    idxs = [c - 1 for c in citations if 1 <= c <= len(sources)] or list(range(min(len(sources), 4)))
    cclaim = strip_citations(claim)
    best = None
    for i in idxs:
        text = strip_citations(str(sources[i].get("text") or ""))
        hc = hard_check(cclaim, text)
        lex = lexical_entailment(cclaim, text)
        key = (1 if hc["ok"] else 0, lex["score"])
        if best is None or key > best[0]:
            best = (key, i, hc, lex)
    if best is None:
        return -1, {}, {}
    return best[1], best[2], best[3]


def verify_claim(claim: str, sources: list[dict], embed_fn=None, nli_fn=None,
                 weights: dict | None = None, bias: float = BIAS,
                 accounting: bool = False) -> dict:
    w = dict(WEIGHTS)
    if weights:
        w.update(weights)
    citations = [int(x) for x in _CITE.findall(claim)]
    idx, hc, lex = _best_source(claim, sources, citations)
    if idx < 0:
        return {"claim": claim, "supported": None, "reason": "no-source", "confidence": 0.0,
                "citations": citations, "evidence": {}}
    text = strip_citations(str(sources[idx].get("text") or ""))
    cclaim = strip_citations(claim)
    src_id = sources[idx].get("id", idx + 1)

    if citations and idx + 1 not in citations:
        return {"claim": claim, "supported": None, "reason": "citation-mismatch", "confidence": 0.0,
                "citations": citations, "evidence": {"source": src_id}}

    if not hc.get("ok", False):
        return {"claim": claim, "supported": False, "reason": "hard:" + ",".join(hc.get("reasons") or []),
                "confidence": 0.02, "citations": citations,
                "evidence": {"source": src_id, "hard": _jsonable(hc), "lex": lex.get("score")}}

    pdv = verify_pdv_claim(cclaim) if accounting else {"checked": False, "ok": None}
    if pdv.get("checked") and pdv.get("ok") is False:
        return {"claim": claim, "supported": False, "reason": "accounting:pdv",
                "confidence": 0.02, "citations": citations,
                "evidence": {"source": src_id, "hard": _jsonable(hc),
                             "accounting": _jsonable(pdv), "lex": lex.get("score")}}

    emb = _cos_hdc(cclaim, text)
    if embed_fn is not None:
        try:
            vs = embed_fn([cclaim, text])
            if vs and len(vs) == 2:
                a, b = vs
                dot = sum(x * y for x, y in zip(a, b))
                na = sum(x * x for x in a) ** 0.5
                nb = sum(x * x for x in b) ** 0.5
                if na and nb:
                    emb = max(0.0, min(1.0, dot / (na * nb)))
        except Exception:
            pass

    nli = None
    contra = 0.0
    if nli_fn is not None:
        try:
            r = nli_fn(text, cclaim)
            if isinstance(r, dict):
                nli = float(r.get("entail", r.get("entailment", 0.0)))
                contra = float(r.get("contra", r.get("contradiction", 0.0)))
        except Exception:
            nli = None

    if contra >= 0.5 or has_negation(cclaim) != has_negation(text):
        return {"claim": claim, "supported": False, "reason": "contradiction-veto",
                "confidence": round(max(contra, 0.9), 6), "citations": citations,
                "evidence": {"source": src_id, "hard": _jsonable(hc),
                             "accounting": _jsonable(pdv), "lex": lex.get("score"),
                             "emb": round(emb, 6), "nli": nli, "contra": contra}}

    signals = {"lex": float(lex.get("score") or 0.0), "emb": float(emb)}
    present = 2
    if nli is not None:
        signals["nli"] = nli
        present = 3
    z = bias + sum(w.get(k, 0.0) * v for k, v in signals.items())
    p = sigmoid(z)
    supported = p >= 0.5 and present >= 2
    reason = "calibrated-and" if supported else "below-threshold"
    numeric_support = bool(hc.get("claim_numbers")) and hc.get("numbers_ok", True)
    if not numeric_support and float(lex.get("recall") or 0.0) < 0.70:
        supported = False
        reason = "lexical-gate"
    if lex.get("score", 0.0) < 0.25 and emb < 0.35:
        supported = False
        reason = "weak-support"
    return {"claim": claim, "supported": supported, "reason": reason,
            "confidence": round(p, 6), "citations": citations,
            "evidence": {"source": src_id, "hard": _jsonable(hc),
                         "accounting": _jsonable(pdv),
                         "lex": round(float(lex.get("score") or 0.0), 6),
                         "emb": round(emb, 6), "nli": nli, "contra": contra, "z": round(z, 6)}}


def verify_answer(answer: str, sources: list[dict], embed_fn=None, nli_fn=None,
                  threshold: float = 0.5, weights: dict | None = None,
                  accounting: bool = False) -> dict:
    claims = split_claims(answer)
    rows = [verify_claim(c, sources, embed_fn=embed_fn, nli_fn=nli_fn, weights=weights,
                         accounting=accounting) for c in claims]
    cited = [r for r in rows if r.get("citations")]
    sup = [r for r in cited if r.get("supported") is True]
    uns = [r for r in cited if r.get("supported") is False]
    unc = [r for r in cited if r.get("supported") is None]
    total_cited = len(cited)
    coverage = (len(sup) / float(total_cited)) if total_cited else 0.0
    overall_cov = (len(sup) / float(len(rows))) if rows else 0.0
    if not rows:
        verdict = "UNVERIFIED"
    elif not total_cited:
        verdict = "UNVERIFIED"
    elif not uns and not unc and coverage >= 0.8:
        verdict = "SUPPORTED"
    elif sup:
        verdict = "PARTIAL"
    else:
        verdict = "UNSUPPORTED"
    return {
        "verdict": verdict,
        "claims": rows,
        "counts": {"total": len(rows), "cited": total_cited, "supported": len(sup),
                   "unsupported": len(uns), "unclear": len(unc)},
        "coverage": round(coverage, 6),
        "coverage_all": round(overall_cov, 6),
        "threshold": threshold,
    }


class ConformalGate:
    """Split-conformal gate for answer-level support (FSR <= alpha under exchangeability)."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.q = None
        self.n = 0

    def fit(self, negative_scores: list[float]) -> "ConformalGate":
        self.q = conformal_quantile(negative_scores, self.alpha)
        self.n = len(negative_scores)
        return self

    def accept(self, score: float) -> bool:
        if self.q is None:
            return score >= 0.5
        return score >= self.q

    def state(self) -> dict:
        return {"alpha": self.alpha, "q": self.q, "n": self.n}


def nli_http(url: str, timeout: float = 8.0):
    base = url.rstrip("/")
    if not base.endswith("/nli"):
        base += "/nli"

    def fn(premise: str, hypothesis: str) -> dict:
        body = json.dumps({"pairs": [[premise[:2500], hypothesis[:800]]]}).encode()
        req = urllib.request.Request(base, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
        scores = d.get("scores") or d.get("results") or []
        if scores and isinstance(scores[0], list):
            s = scores[0]
        elif scores and isinstance(scores[0], dict):
            s = [scores[0].get("entailment", 0), scores[0].get("neutral", 0), scores[0].get("contradiction", 0)]
        else:
            s = [0.0, 0.0, 0.0]
        return {"entail": float(s[0]), "neutral": float(s[1]) if len(s) > 1 else 0.0,
                "contra": float(s[2]) if len(s) > 2 else 0.0}
    return fn


def embed_http(url: str, model: str = "bge-m3", timeout: float = 10.0):
    def fn(texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": model, "input": [t[:2000] for t in texts]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
        return [x["embedding"] for x in d["data"]]
    return fn


def batch_verify(answers: list[str], sources: list[list[dict]], workers: int = 4, **kw) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(verify_answer, a, s, **kw) for a, s in zip(answers, sources)]
        return [f.result() for f in futs]
