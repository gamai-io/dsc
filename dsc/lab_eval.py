"""Evaluate DSC retrieval on the lab golden set (85 bs queries / 40 docs).

Compares: lab-tokenizer BM25, DSC BM25, HDC (int8), optional bge-m3, and wRRF fusions.
Metrics: R@1, R@3, R@5, MRR@10, nDCG@10 (binary relevance, multi-gold).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys

from .fusion import wrrf
from .hdc import cosine_q, vector
from .lexical import BM25Index
from .verify import embed_http


def lab_tokens(text: str) -> list[str]:
    return re.findall(r"[0-9a-zčćžšđ]+", (text or "").lower())


def _rank_by_scores(scores: dict) -> list[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def evaluate(rankings: dict, queries: list[dict], ks=(1, 3, 5, 10)) -> dict:
    out = {("R@%d" % k): 0.0 for k in ks}
    out["MRR@10"] = 0.0
    out["nDCG@10"] = 0.0
    n = 0
    for q in queries:
        ranked = rankings.get(q["q"])
        if ranked is None:
            continue
        gold = set(q.get("gold") or [])
        if not gold:
            continue
        n += 1
        for k in ks:
            if any(x in gold for x in ranked[:k]):
                out["R@%d" % k] += 1.0
        rr = 0.0
        for i, x in enumerate(ranked[:10]):
            if x in gold:
                rr = 1.0 / (i + 1)
                break
        out["MRR@10"] += rr
        graded = q.get("graded") or {g: 1 for g in gold}
        rel = [graded.get(x, 0) for x in ranked[:10]]
        ideal = sorted(graded.values(), reverse=True)[:10]
        ideal = ideal + [0] * max(0, 10 - len(ideal))
        idcg = _dcg(ideal)
        out["nDCG@10"] += (_dcg(rel) / idcg) if idcg else 0.0
    if n:
        for k in list(out):
            out[k] = round(out[k] / n, 4)
    out["n"] = n
    return out


def run(data: dict, embed_url: str | None = None, embed_model: str = "bge-m3",
        return_rankings: bool = False) -> dict:
    docs = data["docs"]
    queries = data["queries"]
    ids = [d["id"] for d in docs]
    texts = {d["id"]: d["text"] for d in docs}

    bm_lab = BM25Index(tokenizer=lab_tokens)
    bm_dsc = BM25Index()
    for d in docs:
        bm_lab.add(d["id"], d["text"])
        bm_dsc.add(d["id"], d["text"])
    bm_lab.finalize()
    bm_dsc.finalize()

    vecs = {d["id"]: vector(d["text"]) for d in docs}
    methods = {"bm25_lab": {}, "bm25_dsc": {}, "hdc": {}}
    for q in queries:
        qt = q["q"]
        methods["bm25_lab"][qt] = [r["id"] for r in bm_lab.search(qt, k=len(docs))]
        methods["bm25_dsc"][qt] = [r["id"] for r in bm_dsc.search(qt, k=len(docs))]
        qv = vector(qt)
        methods["hdc"][qt] = _rank_by_scores({i: cosine_q(qv, vecs[i]) for i in ids})

    if embed_url:
        emb = embed_http(embed_url, model=embed_model, timeout=180)
        dv = emb([texts[i] for i in ids])
        qv = emb([q["q"] for q in queries])
        dvec = {i: v for i, v in zip(ids, dv)}
        methods["bge_m3"] = {}
        for q, qq in zip(queries, qv):
            sc = {}
            for i in ids:
                a, b = qq, dvec[i]
                dot = sum(x * y for x, y in zip(a, b))
                na = sum(x * x for x in a) ** 0.5
                nb = sum(x * x for x in b) ** 0.5
                sc[i] = dot / (na * nb) if na and nb else 0.0
            methods["bge_m3"][q["q"]] = _rank_by_scores(sc)

    def fuse(a: str, b: str, wa: float = 1.0, wb: float = 1.0, name: str | None = None):
        out = {}
        for q in queries:
            qt = q["q"]
            sc = wrrf([methods[a][qt], methods[b][qt]], [wa, wb], k=10)
            out[qt] = [k for k, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))]
        methods[name or (a + "+" + b)] = out

    fuse("bm25_dsc", "hdc", 1.0, 0.6, "bm25_dsc+hdc")
    fuse("bm25_lab", "hdc", 1.0, 0.6, "bm25_lab+hdc")
    if "bge_m3" in methods:
        fuse("bm25_dsc", "bge_m3", 1.0, 1.0, "bm25_dsc+bge")
        fuse("hdc", "bge_m3", 0.6, 1.0, "hdc+bge")
        fuse("bm25_dsc", "hdc", 1.0, 0.6, "bm25_dsc+hdc(pre)")
        sc = {}
        for q in queries:
            qt = q["q"]
            sc[qt] = wrrf([methods["bm25_dsc"][qt], methods["hdc"][qt], methods["bge_m3"][qt]],
                          [1.0, 0.6, 1.0], k=10)
        methods["bm25+hdc+bge"] = {
            qt: [k for k, _ in sorted(v.items(), key=lambda x: (-x[1], x[0]))] for qt, v in sc.items()
        }

    out = {"metrics": {m: evaluate(r, queries) for m, r in methods.items()},
           "n_docs": len(docs), "n_queries": len(queries)}
    if return_rankings:
        out["rankings"] = methods
    return out


def per_query(rankings: dict, queries: list[dict], metric: str = "R@1") -> list[float]:
    vals = []
    for q in queries:
        ranked = rankings.get(q["q"]) or []
        gold = set(q.get("gold") or [])
        if metric == "R@1":
            vals.append(1.0 if ranked and ranked[0] in gold else 0.0)
        elif metric == "MRR@10":
            rr = 0.0
            for i, x in enumerate(ranked[:10]):
                if x in gold:
                    rr = 1.0 / (i + 1)
                    break
            vals.append(rr)
        elif metric == "nDCG@10":
            graded = q.get("graded") or {g: 1 for g in gold}
            rel = [graded.get(x, 0) for x in ranked[:10]]
            ideal = sorted(graded.values(), reverse=True)[:10]
            ideal = ideal + [0] * max(0, 10 - len(ideal))
            idcg = _dcg(ideal)
            vals.append((_dcg(rel) / idcg) if idcg else 0.0)
        else:
            raise ValueError(metric)
    return vals


def paired_bootstrap(rankings_a: dict, rankings_b: dict, queries: list[dict],
                     metric: str = "R@1", n: int = 4000, seed: int = 7) -> dict:
    a = per_query(rankings_a, queries, metric)
    b = per_query(rankings_b, queries, metric)
    if len(a) != len(b) or not a:
        return {"diff": None, "lo": None, "hi": None, "p": None, "n": 0}
    diffs = [x - y for x, y in zip(a, b)]
    obs = sum(diffs) / len(diffs)
    rnd = random.Random(seed)
    boots = []
    k = len(diffs)
    for _ in range(n):
        boots.append(sum(diffs[rnd.randrange(k)] for _ in range(k)) / k)
    boots.sort()
    lo = boots[int(0.025 * n)]
    hi = boots[min(n - 1, int(0.975 * n))]
    le = sum(1 for x in boots if x <= 0) / float(n)
    ge = sum(1 for x in boots if x >= 0) / float(n)
    p = min(1.0, 2.0 * min(le, ge))
    return {"metric": metric, "diff": round(obs, 4), "lo": round(lo, 4),
            "hi": round(hi, 4), "p": round(p, 4), "n": k}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsc.lab_eval")
    ap.add_argument("--data", default="/tmp/lab_embed/data85.json")
    ap.add_argument("--embed-url", default="")
    ap.add_argument("--embed-model", default="bge-m3")
    ap.add_argument("--out", default="")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--bootstrap-n", type=int, default=4000)
    args = ap.parse_args(argv)
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    res = run(data, args.embed_url or None, args.embed_model,
              return_rankings=args.bootstrap)
    if args.bootstrap:
        r = res.pop("rankings")
        pairs = [("bm25_dsc+hdc", "bm25_dsc"), ("hdc+bge", "bge_m3"),
                 ("bm25+hdc+bge", "bge_m3"), ("bm25+hdc+bge", "hdc+bge")]
        cmp = {}
        for a, b in pairs:
            if a in r and b in r:
                cmp[a + " vs " + b] = {
                    "R@1": paired_bootstrap(r[a], r[b], data["queries"], "R@1", args.bootstrap_n),
                    "nDCG@10": paired_bootstrap(r[a], r[b], data["queries"], "nDCG@10", args.bootstrap_n),
                }
        res["bootstrap"] = cmp
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
