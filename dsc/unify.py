"""One registry over all tools, wrapped by the DSC Trust Layer.

Built-ins are deliberately small and offline: calc, normalize, fingerprint, search, verify,
remember/recall, echo, clock. External tools (SQL, HC, mail, web) can be registered by the
caller; the trust contract is the same.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import tempfile
import time
from decimal import Decimal, InvalidOperation

from .api import Core
from .fingerprints import minhash128, simhash64
from .normalize import canonical, dec_str, features
from .caps import CapRegistry, cap_id as _cap_id
from .gtl import TrustLayer
from .model import DeterministicModel

_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
            ast.UAdd, ast.USub)


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            if b == 0:
                raise ValueError("division by zero")
            return a / b
        if isinstance(node.op, ast.Mod):
            if b == 0:
                raise ValueError("modulo by zero")
            return a % b
        if isinstance(node.op, ast.Pow):
            if b != b.to_integral_value() or abs(int(b)) > 100:
                raise ValueError("exponent must be an integer <= 100")
            return a ** int(b)
    raise ValueError("unsupported expression")


def safe_calc(args: dict) -> dict:
    expr = str(args.get("expr", ""))[:200]
    tree = ast.parse(expr, mode="eval")
    for n in ast.walk(tree):
        if not isinstance(n, _ALLOWED):
            raise ValueError("unsupported node: %s" % type(n).__name__)
    v = _eval(tree)
    return {"expr": expr, "value": dec_str(v.normalize())}


class Unified:
    def __init__(self, cache_path: str | None = None, gtl_path: str | None = None,
                 embed_fn=None, nli_fn=None, chain_key: bytes | None = None,
                 sign_key=None, key_id: str | None = None, model_path: str | None = None):
        self.core = Core(cache_path=cache_path, embed_fn=embed_fn, nli_fn=nli_fn)
        self.model = DeterministicModel.load(model_path) if model_path else None
        self.gtl = TrustLayer(path=gtl_path, chain_key=chain_key,
                              sign_key=sign_key, key_id=key_id)
        self.caps = CapRegistry()
        self._register()
        self._register_caps()

    def _register(self):
        g = self.gtl
        g.register("calc", safe_calc, cache_ttl=3600, description="safe Decimal arithmetic")
        g.register("normalize", lambda a: features(str(a.get("text", ""))), cache_ttl=60,
                   description="NFC/fold/numbers/units/negation")
        g.register("fingerprint", lambda a: {
            "canonical": canonical(str(a.get("text", ""))),
            "simhash64": simhash64(str(a.get("text", ""))),
            "minhash128": list(minhash128(str(a.get("text", "")))),
        }, cache_ttl=60, description="identity + SimHash + MinHash")
        g.register("search", lambda a: self.core.search(
            str(a.get("query", "")), a.get("docs") or [], int(a.get("k", 5))),
                   cache_ttl=30, description="BM25+HDC(/bge) fusion over supplied docs")
        g.register("verify", lambda a: self.core.verify(
            str(a.get("answer", "")), a.get("sources") or [],
            inference=a.get("inference"), accounting=bool(a.get("accounting", False))),
                   cache_ttl=60, description="hard checks + calibrated verification")
        g.register("remember", lambda a: {"stored": self.core.remember(
            str(a.get("query", "")), str(a.get("answer", "")), sid=a.get("sid"),
            domain=a.get("domain", "general"), trust=a.get("trust", "verified"),
            envelope_id=a.get("envelope_id"), inference_level=a.get("inference_level"))},
                   side_effect=True, cache_ttl=0, description="semantic cache write")
        g.register("recall", lambda a: self.core.recall(
            str(a.get("query", "")), sid=a.get("sid"), domain=a.get("domain", "general"),
            envelope_id=a.get("envelope_id")),
                   cache_ttl=0, description="semantic cache read")
        if self.model is not None:
            g.register("model_search", lambda a: self.model.search(
                str(a.get("query", "")), int(a.get("k", 5))),
                       cache_ttl=30, description="trained deterministic model: search")
            g.register("model_classify", lambda a: self.model.classify(str(a.get("query", ""))),
                       cache_ttl=30, description="trained deterministic model: category")
        g.register("echo", lambda a: a, cache_ttl=0, description="identity tool")
        g.register("clock", lambda a: {"ts": time.time()}, deterministic=False,
                   cache_ttl=0, description="wall clock (stochastic)")

    def _register_caps(self) -> None:
        """Every local tool is a 'skill' provider; worker routes are 'http'."""
        for spec in self.gtl.capabilities():
            name = spec.get("name")
            if not name:
                continue
            self.caps.register(name, "skill", _cap_id("dsc", name),
                               version=str(spec.get("version") or "1"), ref=("gtl", name))
        for route in ("core", "dag", "orch", "sem", "ask"):
            self.caps.register(route, "http", _cap_id("worker", route), version="1",
                               ref=("transport", route))

    def cap_list(self, op: str | None = None) -> list[dict]:
        return [{k: v for k, v in item.items() if k != "ref"} for item in self.caps.list(op)]

    def resolve(self, op: str, target: str | None = None) -> dict:
        item = self.caps.resolve(op, target)
        return {k: v for k, v in item.items() if k != "ref"}

    def call_cap(self, op: str, args: dict | None = None, target: str | None = None, **kw):
        item = self.caps.resolve(op, target)
        kind, ref = item["ref"]
        if kind == "gtl":
            return self.gtl.call(ref, args or {}, **kw)
        from .transport import call as _tcall
        return _tcall(ref, args or {})

    def call(self, tool: str, args: dict | None = None, **kw):
        return self.gtl.call(tool, args, **kw)

    def verify(self, receipt: dict, result=None, public_key=None):
        return self.gtl.verify(receipt, result=result, public_key=public_key)

    def capabilities(self):
        return self.gtl.capabilities()

    def stats(self):
        return {"gtl": self.gtl.stats(), "cache": self.core.stats()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsc.unify")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    sub.add_parser("capabilities")
    args = ap.parse_args(argv)
    d = tempfile.mkdtemp(prefix="dsc_unify_")
    u = Unified(cache_path=os.path.join(d, "cache.db"), gtl_path=os.path.join(d, "gtl.db"),
                model_path="docs/dsc/dsc_model.json" if os.path.exists("docs/dsc/dsc_model.json") else None)
    if args.cmd == "capabilities":
        for c in u.capabilities():
            print(c)
        return 0
    print("calc:", u.call("calc", {"expr": "(2+3)*4"})["result"]["value"])
    print("calc cached:", u.call("calc", {"expr": "(2+3)*4"})["cached"])
    print("search:", u.call("search", {"query": "kompresor 10 bar",
                                       "docs": ["kompresor 10 bar 500 L", "graco airless 795"],
                                       "k": 1})["result"])
    print("echo:", u.call("echo", {"x": 1})["result"])
    r1 = u.call("remember", {"query": "koliko ima kupaca", "answer": "1114 kupaca", "sid": "demo"},
                idem_key="demo-1")
    r2 = u.call("remember", {"query": "koliko ima kupaca", "answer": "1114 kupaca", "sid": "demo"},
                idem_key="demo-1")
    print("side effect:", r1["result"], "replayed:", r2["replayed"])
    if u.model is not None:
        print("model_search:", u.call("model_search", {"query": "kompresor", "k": 1})["result"])
        print("model_classify:", u.call("model_classify", {"query": "airless farbanje"})["result"])
    print("stats:", u.stats()["gtl"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
