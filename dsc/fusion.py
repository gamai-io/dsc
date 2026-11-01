"""Rank fusion, log-odds calibration, cost-optimal thresholds, conformal quantiles."""
from __future__ import annotations

import math
import random
from statistics import mean


def sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def logit(p: float) -> float:
    p = min(max(float(p), 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def wrrf(rank_lists: list[list], weights: list[float] | None = None, k: int = 10) -> dict:
    if weights is None:
        weights = [1.0] * len(rank_lists)
    if len(weights) != len(rank_lists):
        raise ValueError("weights must match rank_lists")
    out: dict = {}
    for lst, w in zip(rank_lists, weights):
        for rank, doc in enumerate(lst):
            out[doc] = out.get(doc, 0.0) + w / (k + rank + 1.0)
    return out


def normalize_scores(scores: dict) -> dict:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi <= lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def log_odds(signals: dict, weights: dict, bias: float = 0.0) -> float:
    z = bias
    for k, v in signals.items():
        z += float(weights.get(k, 0.0)) * float(v)
    return z


def decide(signals: dict, weights: dict, bias: float = -3.2) -> dict:
    z = log_odds(signals, weights, bias)
    p = sigmoid(z)
    return {"z": round(z, 6), "p": round(p, 6), "supported": p >= 0.5}


def calibrate_threshold(scores: list[float], labels: list[int],
                        cost_miss: float = 1.0, cost_err: float = 10.0) -> dict:
    if len(scores) != len(labels):
        raise ValueError("scores/labels length mismatch")
    if not scores:
        return {"threshold": 0.5, "cost": 0.0, "fpr": 0.0, "fnr": 0.0, "n": 0}
    cand = sorted(set([0.0, 1.0] + [round(s, 6) for s in scores]))
    best = None
    for t in cand:
        fp = fn = tp = tn = 0
        for s, y in zip(scores, labels):
            pred = 1 if s >= t else 0
            if pred and y:
                tp += 1
            elif pred and not y:
                fp += 1
            elif not pred and y:
                fn += 1
            else:
                tn += 1
        cost = fn * cost_miss + fp * cost_err
        fpr = fp / float(fp + tn) if (fp + tn) else 0.0
        fnr = fn / float(fn + tp) if (fn + tp) else 0.0
        rec = {"threshold": t, "cost": round(cost, 6), "fpr": round(fpr, 6),
               "fnr": round(fnr, 6), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": len(scores)}
        if best is None or (rec["cost"], rec["fpr"]) < (best["cost"], best["fpr"]):
            best = rec
    return best


def conformal_quantile(scores: list[float], alpha: float = 0.05) -> float:
    if not scores:
        return float("inf")
    s = sorted(scores)
    n = len(s)
    idx = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
    idx = min(max(idx, 0), n - 1)
    return s[idx]


def ece(scores: list[float], labels: list[int], bins: int = 10) -> float:
    if not scores:
        return 0.0
    total = len(scores)
    err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if (lo <= s < hi) or (b == bins - 1 and s == 1.0)]
        if not idx:
            continue
        conf = mean(scores[i] for i in idx)
        acc = mean(labels[i] for i in idx)
        err += (len(idx) / total) * abs(conf - acc)
    return round(err, 6)


def bootstrap_ci(values: list[float], n: int = 1000, alpha: float = 0.05, seed: int = 7) -> dict:
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rnd = random.Random(seed)
    means = []
    k = len(values)
    for _ in range(n):
        means.append(sum(values[rnd.randrange(k)] for _ in range(k)) / float(k))
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return {"mean": round(mean(values), 6), "lo": round(lo, 6), "hi": round(hi, 6), "n": k}


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict:
    if len(a_correct) != len(b_correct):
        raise ValueError("length mismatch")
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if (not a) and b)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and (not b))
    if b01 + b10 == 0:
        return {"b01": 0, "b10": 0, "chi2": 0.0, "p": 1.0}
    chi2 = max(0.0, abs(b10 - b01) - 1.0) ** 2 / float(b01 + b10)
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return {"b01": b01, "b10": b10, "chi2": round(chi2, 6), "p": round(p, 6)}
