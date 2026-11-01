"""Matrix determinism probe: prompts x generation lengths x concurrency levels."""
from __future__ import annotations

import argparse
import json
import sys

from .determinism_check import http_call, run_probe

PROMPTS = [
    "Tell me about Richard Feynman",
    "Objasni u tri recenice kako se racuna PDV u Bosni i Hercegovini.",
    "Nabroji pet koraka za sigurno rukovanje kompresorom.",
]


def summarize(results: list[dict]) -> dict:
    stable = []
    unstable = []
    inconclusive = []
    for r in results:
        p = r.get("probe", {})
        if p.get("cross_batch_equal"):
            stable.append(r)
        elif p.get("errors_total"):
            inconclusive.append(r)
        else:
            unstable.append(r)

    def brief(r):
        return {"prompt": r.get("prompt"), "max_tokens": r.get("max_tokens"),
                "unique_total": r.get("probe", {}).get("unique_total"),
                "errors_total": r.get("probe", {}).get("errors_total"),
                "interpretation": r.get("probe", {}).get("interpretation")}

    if unstable:
        v = "NOT-BATCH-INVARIANT"
    elif inconclusive:
        v = "INCONCLUSIVE"
    else:
        v = "STABLE"
    return {
        "cases": len(results), "stable": len(stable),
        "unstable": len(unstable), "inconclusive": len(inconclusive),
        "unstable_cases": [brief(r) for r in unstable],
        "inconclusive_cases": [brief(r) for r in inconclusive],
        "verdict": v,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsc.determinism_matrix")
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--token-lengths", default="16,64")
    ap.add_argument("--concurrency", default="1,8,16")
    ap.add_argument("--prompts-file", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    prompts = PROMPTS
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [x.strip() for x in f if x.strip()]
    token_lengths = [int(x) for x in args.token_lengths.split(",") if x.strip()]
    conc = tuple(int(x) for x in args.concurrency.split(",") if x.strip())
    results = []
    for p in prompts:
        for mt in token_lengths:
            fn = http_call(args.url, args.model, args.api_key, max_tokens=mt)
            probe = run_probe(fn, p, batch_sizes=conc, repeat=1)
            results.append({"prompt": p, "max_tokens": mt, "concurrency": list(conc), "probe": probe})
            print(json.dumps({"prompt": p[:40], "max_tokens": mt,
                              "unique": probe["unique_total"],
                              "errors": probe["errors_total"],
                              "stable": probe["cross_batch_equal"]}, ensure_ascii=False), flush=True)
    out = {"matrix": results, "summary": summarize(results)}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
    return 0 if out["summary"]["verdict"] == "STABLE" else 2


if __name__ == "__main__":
    sys.exit(main())
