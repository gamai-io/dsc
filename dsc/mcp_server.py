"""Minimal MCP (stdio JSON-RPC) adapter for GTL/1.

initialize / tools/list / tools/call / ping. Every tools/call returns the GTL envelope
(result + receipt + trust) in structuredContent, so an MCP client gets the proof, not just text.
"""
from __future__ import annotations

import argparse
import json
import sys

from .unify import Unified

PROTO = "2025-11-25"
SERVER = {"name": "dsc", "version": "0.3"}

SCHEMAS = {
    "calc": {"type": "object", "properties": {"expr": {"type": "string"}},
             "required": ["expr"], "additionalProperties": False},
    "normalize": {"type": "object", "properties": {"text": {"type": "string"}},
                  "required": ["text"], "additionalProperties": False},
    "fingerprint": {"type": "object", "properties": {"text": {"type": "string"}},
                    "required": ["text"], "additionalProperties": False},
    "search": {"type": "object", "properties": {"query": {"type": "string"},
                                                "docs": {"type": "array"},
                                                "k": {"type": "integer"}},
               "required": ["query", "docs"], "additionalProperties": False},
    "verify": {"type": "object", "properties": {"answer": {"type": "string"},
                                                "sources": {"type": "array"},
                                                "accounting": {"type": "boolean"}},
               "required": ["answer", "sources"], "additionalProperties": False},
    "remember": {"type": "object", "properties": {"query": {"type": "string"},
                                                 "answer": {"type": "string"},
                                                 "sid": {"type": "string"},
                                                 "domain": {"type": "string"},
                                                 "_idem_key": {"type": "string"}},
                 "required": ["query", "answer"], "additionalProperties": False},
    "recall": {"type": "object", "properties": {"query": {"type": "string"},
                                                "sid": {"type": "string"},
                                                "domain": {"type": "string"}},
               "required": ["query"], "additionalProperties": False},
    "model_search": {"type": "object", "properties": {"query": {"type": "string"},
                                                      "k": {"type": "integer"}},
                     "required": ["query"], "additionalProperties": False},
    "model_classify": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"], "additionalProperties": False},
    "echo": {"type": "object", "additionalProperties": True},
    "clock": {"type": "object", "additionalProperties": True},
}


def _tools(u: Unified) -> list[dict]:
    out = []
    for c in u.capabilities():
        out.append({"name": c["name"], "description": c["description"],
                    "inputSchema": SCHEMAS.get(c["name"], {"type": "object",
                                                           "additionalProperties": True})})
    return out


def handle(req: dict, u: Unified) -> dict | None:
    rid = req.get("id")
    method = req.get("method")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": PROTO, "capabilities": {"tools": {}},
            "serverInfo": SERVER}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": _tools(u)}}
    if method == "tools/call":
        p = req.get("params") or {}
        name = p.get("name")
        args = dict(p.get("arguments") or {})
        idem = args.pop("_idem_key", None)
        inference = args.pop("_inference", None)
        use_cache = args.pop("_use_cache", True)
        res = u.call(str(name), args, idem_key=idem, inference=inference, use_cache=bool(use_cache))
        text = json.dumps(res, ensure_ascii=False)
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "content": [{"type": "text", "text": text}],
            "structuredContent": res,
            "isError": not bool(res.get("ok", False))}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601,
                                                   "message": "method not found: %s" % method}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dsc.mcp_server")
    ap.add_argument("--db", default="/tmp/dsc_mcp_gtl.db")
    ap.add_argument("--cache", default="/tmp/dsc_mcp_cache.db")
    ap.add_argument("--model", default="")
    args = ap.parse_args(argv)
    u = Unified(cache_path=args.cache, gtl_path=args.db, model_path=args.model or None)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            out = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            out = handle(req, u)
        if out is not None:
            sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
