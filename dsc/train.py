"""Train the deterministic DSC-DSC model on a docs/queries JSON (offline, no GPU)."""
from __future__ import annotations

import argparse
import json
import sys

from .model import DeterministicModel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsc.train")
    ap.add_argument("--data", default="/tmp/lab_embed/data85.json")
    ap.add_argument("--out", default="docs/dsc/dsc_model.json")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args(argv)
    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    m = DeterministicModel().fit(data["docs"])
    size = m.save(args.out)
    out = {"out": args.out, "bytes": size, **m.stats()}
    if args.eval:
        hits = 0
        qs = data.get("queries") or []
        for q in qs:
            gold = set(q.get("gold") or [])
            if not gold:
                continue
            top = m.search(q["q"], k=1)
            if top and top[0]["id"] in gold:
                hits += 1
        out["R@1"] = round(hits / float(len(qs)), 4) if qs else None
        out["n"] = len(qs)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
