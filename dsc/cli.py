"""CLI: python -m dsc.cli <command> ..."""
from __future__ import annotations

import argparse
import json
import sys
import unittest

from .api import Core
from .normalize import features


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsc")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze")
    a.add_argument("text")

    s = sub.add_parser("sim")
    s.add_argument("a")
    s.add_argument("b")

    se = sub.add_parser("search")
    se.add_argument("query")
    se.add_argument("--file", required=True, help="JSON list of strings or {id,text}")
    se.add_argument("-k", type=int, default=5)

    v = sub.add_parser("verify")
    v.add_argument("--answer", required=True)
    v.add_argument("--source", action="append", default=[])

    c = sub.add_parser("cache-stats")
    c.add_argument("--db", default=None)

    h = sub.add_parser("health")
    h.add_argument("--json", action="store_true")
    h.add_argument("--timeout", type=float, default=2.0)

    cl = sub.add_parser("call")
    cl.add_argument("cap", choices=["ask", "core", "dag", "orch", "sem"])
    cl.add_argument("--q", default="")
    cl.add_argument("--goal", default="")
    cl.add_argument("--path", default="/jobs")
    cl.add_argument("--levels", default="2,2")
    cl.add_argument("--timeout", type=float, default=None)

    ir = sub.add_parser("ir")
    ir_sub = ir.add_subparsers(dest="ir_cmd", required=True)
    ip = ir_sub.add_parser("pack")
    ip.add_argument("file")
    ip.add_argument("--out", default=None)
    ip.add_argument("--compress", action="store_true")
    ii = ir_sub.add_parser("info")
    ii.add_argument("file")

    sub.add_parser("determinism")
    sub.add_parser("selftest")

    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        print(_json(features(args.text)))
    elif args.cmd == "sim":
        from .fingerprints import similarity
        print(_json(similarity(args.a, args.b)))
    elif args.cmd == "search":
        with open(args.file, encoding="utf-8") as f:
            docs = json.load(f)
        core = Core(cache_path="/tmp/dsc_cli.db")
        print(_json(core.search(args.query, docs, k=args.k)))
    elif args.cmd == "verify":
        core = Core(cache_path="/tmp/dsc_cli.db")
        print(_json(core.verify(args.answer, args.sources if hasattr(args, "sources") else args.source)))
    elif args.cmd == "cache-stats":
        from .cache import SemanticCache
        print(_json(SemanticCache(args.db or "/tmp/dsc_cache.db").stats()))
    elif args.cmd == "health":
        from .health import health as _health
        res = _health(timeout=args.timeout)
        print(_json(res) if args.json else res["status"])
        return int(res["exit_code"])
    elif args.cmd == "call":
        from .transport import call as _call
        payload = {"q": args.q, "goal": args.goal, "path": args.path, "levels": args.levels}
        res = _call(args.cap, payload, timeout=args.timeout)
        print(_json(res))
        return 0 if res["ok"] else 1
    elif args.cmd == "ir":
        from . import ir as _ir
        if args.ir_cmd == "pack":
            src = open(args.file, encoding="utf-8").read()
            blob = _ir.pack_program(src, compress=args.compress)
            out = args.out or (args.file.rsplit(".", 1)[0] + ".gbin")
            with open(out, "wb") as f:
                f.write(blob)
            print(_json(_ir.info(blob) | {"out": out}))
        else:
            with open(args.file, "rb") as f:
                print(_json(_ir.info(f.read())))
    elif args.cmd == "determinism":
        from .nondeterminism import describe
        print(_json(describe()))
    elif args.cmd == "selftest":
        suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
        res = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if res.wasSuccessful() else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
