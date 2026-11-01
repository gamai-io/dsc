"""Lexical scoring (BM25-lite) and deterministic hard checks (numbers, units, dates, polarity)."""
from __future__ import annotations

import math
from collections import Counter

from .accounting import verify_arithmetic
from .normalize import (
    antonym_conflict, content_tokens, extract_dates, extract_numbers, extract_years,
    has_negation, tokens,
)


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer=None):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer or content_tokens
        self.ids: list = []
        self.docs: list[list[str]] = []
        self.meta: list = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def add(self, doc_id, text: str, meta=None) -> None:
        toks = self.tokenizer(text)
        self.ids.append(doc_id)
        self.docs.append(toks)
        self.meta.append(meta)
        c = Counter(toks)
        self.tf.append(c)
        for t in c:
            self.df[t] += 1

    def finalize(self) -> "BM25Index":
        self.avgdl = (sum(len(d) for d in self.docs) / float(len(self.docs))) if self.docs else 0.0
        return self

    def _idf(self, t: str) -> float:
        n = len(self.docs)
        df = self.df.get(t, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[dict]:
        if not self.docs:
            return []
        qt = self.tokenizer(query)
        if not qt:
            return []
        scored = []
        for i, toks in enumerate(self.docs):
            dl = len(toks)
            if dl == 0:
                continue
            score = 0.0
            c = self.tf[i]
            for t in qt:
                f = c.get(t, 0)
                if not f:
                    continue
                score += self._idf(t) * (f * (self.k1 + 1.0)) / (
                    f + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0)))
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], str(self.ids[x[1]])))
        return [{"id": self.ids[i], "score": round(s, 6), "meta": self.meta[i]} for s, i in scored[:k]]


_PHYS = {"pressure", "power", "volume", "mass", "length", "time"}


def _num_match(claim: dict, src: dict) -> bool:
    cd, sd = claim.get("dim"), src.get("dim")
    if cd and sd and cd != sd:
        return False
    rel = 0.01 if (cd in _PHYS or sd in _PHYS) else 1e-9
    cbase = claim.get("base")
    sbase = src.get("base")
    if cbase is not None and sbase is not None:
        try:
            cv, sv = float(cbase), float(sbase)
        except (TypeError, ValueError):
            return False
        if cv == sv:
            return True
        return abs(cv - sv) <= rel * max(1.0, abs(cv), abs(sv))
    try:
        return float(claim["value"]) == float(src["value"])
    except (TypeError, ValueError):
        return False


def hard_check(claim: str, source: str) -> dict:
    cn = extract_numbers(claim)
    sn = extract_numbers(source)
    used = [False] * len(sn)
    unmatched = []
    for c in cn:
        hit = False
        for i, s in enumerate(sn):
            if used[i]:
                continue
            if _num_match(c, s):
                used[i] = True
                hit = True
                break
        if not hit:
            unmatched.append(c)
    numbers_ok = not unmatched

    cd = set(extract_dates(claim))
    sd = set(extract_dates(source))
    cy = set(extract_years(claim))
    sy = set(extract_years(source))
    dates_ok = True
    if cd and not cd.issubset(sd):
        dates_ok = False
    if cy and not cy.issubset(sy):
        dates_ok = False

    polarity_ok = has_negation(claim) == has_negation(source)
    ant = antonym_conflict(claim, source)
    arith = verify_arithmetic(claim)
    arithmetic_ok = (not arith.get("checked")) or bool(arith.get("ok"))
    reasons = []
    if not numbers_ok:
        reasons.append("number: " + ", ".join("%s %s" % (x["value"], x["unit"] or "") for x in unmatched))
    if not dates_ok:
        reasons.append("date")
    if not polarity_ok:
        reasons.append("polarity")
    if ant:
        reasons.append("antonym: " + ",".join(ant))
    if not arithmetic_ok:
        bad = [c for c in arith.get("checks", []) if not c.get("ok")]
        reasons.append("arithmetic: " + "; ".join(
            "%s expected %s got %s" % (c["kind"], c["expected"], c["got"]) for c in bad[:3]))
    return {
        "ok": numbers_ok and dates_ok and polarity_ok and not ant and arithmetic_ok,
        "numbers_ok": numbers_ok,
        "dates_ok": dates_ok,
        "polarity_ok": polarity_ok,
        "antonym_ok": not ant,
        "arithmetic_ok": arithmetic_ok,
        "arithmetic": arith,
        "reasons": reasons,
        "claim_numbers": cn,
        "source_numbers": sn,
    }


def _bigrams(ts: list[str]) -> set[tuple[str, str]]:
    return {(ts[i], ts[i + 1]) for i in range(len(ts) - 1)}


def lexical_entailment(claim: str, source: str) -> dict:
    ct = content_tokens(claim)
    st = content_tokens(source)
    st_set = set(st)
    if not ct:
        return {"score": 0.0, "recall": 0.0, "bigram": 0.0, "numeric": 1.0, "missing": []}
    matched = [t for t in ct if t in st_set]
    recall = len(matched) / float(len(ct))
    bc, bs = _bigrams(ct), _bigrams(st)
    bigram = (len(bc & bs) / float(len(bc))) if bc else 0.0
    hc = hard_check(claim, source)
    numeric = 1.0 if hc["numbers_ok"] and hc["dates_ok"] else 0.0
    score = 0.65 * recall + 0.20 * bigram + 0.15 * numeric
    if not hc["polarity_ok"] or not hc["antonym_ok"]:
        score *= 0.25
    return {
        "score": round(min(1.0, max(0.0, score)), 6),
        "recall": round(recall, 6),
        "bigram": round(bigram, 6),
        "numeric": numeric,
        "missing": [t for t in ct if t not in st_set][:20],
        "hard": hc,
    }


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))
