"""Calibrate the semantic-cache threshold on lab-derived paraphrases.

Store the 85 original queries (answer = gold doc id), probe with 170 variants
(diacritic fold, word order), and 40 unrelated doc spans as negatives.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from .cache import SemanticCache
from .normalize import canonical, features, has_money_unit_percent


def _answer(q: dict) -> str | None:
    gold = q.get("gold") or []
    return gold[0] if len(gold) == 1 else None


def _semantic_ok(text: str) -> bool:
    f = features(text)
    if f["action"] or f["fresh"] or f["multi"]:
        return False
    if has_money_unit_percent(text):
        return False
    if any(("." in n["raw"] or "," in n["raw"]) for n in f["numbers"]):
        return False
    return True


def calibrate(data: dict, thresholds: list[float], sid: str = "calib",
              domain: str = "doc") -> dict:
    bases = [q for q in data["queries"] if not q.get("base") and _answer(q)]
    probes = [q for q in data["queries"] if q.get("base")]
    stored_q = {canonical(q["q"]) for q in bases}
    negatives = []
    for d in data["docs"]:
        words = (d.get("text") or "").split()
        for a, b in ((8, 16), (16, 24), (24, 32), (32, 40)):
            w = " ".join(words[a:b])
            if w and canonical(w) not in stored_q:
                negatives.append({"q": w, "gold": [d["id"]]})
                break
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    c = SemanticCache(path, max_entries=5000, min_answer_len=1)
    try:
        stored = 0
        for q in bases:
            ans = _answer(q)
            if ans and c.put(q["q"], ans, sid=sid, domain=domain, trust="verified"):
                stored += 1
        eligible_probes = [p for p in probes if _answer(p) and _semantic_ok(p["q"])]
        rows = []
        for t in thresholds:
            c.threshold = float(t)
            hit = wrong = miss = 0
            for p in probes:
                ans = _answer(p)
                if ans is None:
                    continue
                r = c.get(p["q"], sid=sid, domain=domain)
                if r is None:
                    miss += 1
                elif r["answer"] == ans:
                    hit += 1
                else:
                    wrong += 1
            ehit = emis = ewrong = 0
            for p in eligible_probes:
                ans = _answer(p)
                r = c.get(p["q"], sid=sid, domain=domain)
                if r is None:
                    emis += 1
                elif r["answer"] == ans:
                    ehit += 1
                else:
                    ewrong += 1
            fp = 0
            for n in negatives:
                if c.get(n["q"], sid=sid, domain=domain) is not None:
                    fp += 1
            n_probe = hit + wrong + miss
            precision = hit / float(hit + wrong + fp) if (hit + wrong + fp) else 0.0
            recall = hit / float(n_probe) if n_probe else 0.0
            fpr = fp / float(len(negatives)) if negatives else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            er = ehit / float(ehit + ewrong + emis) if (ehit + ewrong + emis) else 0.0
            rows.append({"threshold": round(float(t), 3), "hit": hit, "wrong": wrong,
                         "miss": miss, "negatives": len(negatives), "false_pos": fp,
                         "precision": round(precision, 4), "recall": round(recall, 4),
                         "fpr": round(fpr, 4), "f1": round(f1, 4),
                         "eligible_probes": len(eligible_probes), "eligible_hit": ehit,
                         "eligible_recall": round(er, 4)})
        eligible_rows = [r for r in rows if r["fpr"] <= 0.02]
        best = max(eligible_rows, key=lambda r: (r["eligible_recall"], r["precision"])) \
            if eligible_rows else max(rows, key=lambda r: r["f1"])
        return {"stored": stored, "base_queries": len(bases), "probes": len(probes),
                "eligible_probes": len(eligible_probes), "negatives": len(negatives),
                "rows": rows, "best": best}
    finally:
        c.close()
        for suf in ("", "-wal", "-shm"):
            try:
                os.unlink(path + suf)
            except OSError:
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsc.cache_calib")
    ap.add_argument("--data", default="docs/dsc/golden_synth300.json")
    ap.add_argument("--thresholds", default="0.55,0.60,0.65,0.70,0.72,0.75,0.80,0.85,0.90,0.95")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    th = [float(x) for x in args.thresholds.split(",") if x.strip()]
    res = calibrate(data, th)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
