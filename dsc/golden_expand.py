"""Build a deterministic 300-query stress set from the lab docs/queries.

Queries are synthetic: exact spans, keyword bags, typos, diacritic folds, word-order swaps,
and category queries. Labels are derived from the source doc/category. Use for regression and
threshold calibration, not as a scientific benchmark.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

STOP = set("""
i u na za od do sa se je su da li kao po iz ali ili te ni biti koji koja koje the a an of to in on is are
""".split())

_DIA = str.maketrans({"č": "c", "ć": "c", "ž": "z", "š": "s", "đ": "d",
                      "Č": "C", "Ć": "C", "Ž": "Z", "Š": "S", "Đ": "D"})


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zČĆŽŠĐčćžšđ0-9]+", text or "")


def dediac(text: str) -> str:
    return (text or "").translate(_DIA)


def top_words(text: str, n: int = 5) -> list[str]:
    freq = {}
    for w in words(text):
        lw = w.lower()
        if len(lw) < 2 or lw in STOP:
            continue
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))[:n]]


def typo(word: str) -> str:
    if len(word) < 4:
        return word + "x"
    i = len(word) // 2
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def generate(data: dict) -> dict:
    docs = data["docs"]
    queries = data["queries"]
    out = []
    seen = set()

    def add(q: str, gold: list[str], cat: str, base: str | None = None,
            origin: bool = False):
        key = " ".join((q or "").split()).lower()
        if not key or key in seen:
            return
        seen.add(key)
        out.append({"q": q, "gold": list(gold), "cat": cat,
                    "graded": {g: 2 for g in gold}, "base": base, "origin": origin})

    for q in queries:
        add(q["q"], q["gold"], q.get("cat", "exact"), origin=True)

    for d in docs:
        w = words(d["text"])
        add(" ".join(w[:8]), [d["id"]], "exact")
        tw = top_words(d["text"], 5)
        if tw:
            add(" ".join(tw), [d["id"]], "spec")
            t = typo(tw[0])
            add(" ".join([t] + tw[1:4]), [d["id"]], "typo")

    for q in queries:
        add(dediac(q["q"]), q["gold"], "diacritic", base=q["q"])
        parts = q["q"].split()
        if len(parts) >= 3:
            add(" ".join([parts[1], parts[0]] + parts[2:]), q["gold"], "paraphrase", base=q["q"])
        else:
            add(q["q"] + " molim", q["gold"], "paraphrase", base=q["q"])

    for c in sorted({d.get("cat") for d in docs if d.get("cat")}):
        gold = [d["id"] for d in docs if d.get("cat") == c]
        if gold:
            add("kategorija " + str(c), gold, "category")

    return {"synthetic": True,
            "note": "Synthetic stress set from lab docs; not a scientific benchmark.",
            "docs": docs, "queries": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsc.golden_expand")
    ap.add_argument("--data", default="/tmp/lab_embed/data85.json")
    ap.add_argument("--out", default="docs/dsc/golden_synth300.json")
    args = ap.parse_args(argv)
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    out = generate(data)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({"docs": len(out["docs"]), "queries": len(out["queries"]),
                      "out": args.out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
