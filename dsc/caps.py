"""Capability ids and deterministic POLY resolution (R3).

Two ideas, one module:

  cap_id      stable address of a capability:  "<server>.<tool>@<schema8>"
              schema8 = first 8 hex of the JCS digest of the tool's input schema,
              so a tool that changes its schema gets a NEW id (no silent drift).

  svc:// URI  address of a service surface:    "svc://harness/dag#run"
              parsed into {server, surface, op}.

POLY resolution is deterministic by construction: explicit target wins, otherwise
a FIXED priority list (skill -> hc -> sql -> http -> model), then lowest version,
then lexicographic cap id. Insertion order of the registry never matters.
"""
from __future__ import annotations

from typing import Any, Iterable

from . import jcs

PRIORITY: tuple[str, ...] = ("skill", "hc", "sql", "http", "model")
_PRIORITY_INDEX = {t: i for i, t in enumerate(PRIORITY)}
NO_SCHEMA = "noschema"


class CapError(LookupError):
    """Raised for malformed ids or unresolvable operations."""


def schema_hash(schema: dict | None) -> str:
    if not schema:
        return NO_SCHEMA
    return jcs.digest(schema)[:8]


def cap_id(server: str, tool: str, schema: dict | None = None) -> str:
    if not server or not tool:
        raise CapError("server i tool su obavezni")
    if any(ch in (server + tool) for ch in ".@#/ "):
        raise CapError("nedozvoljen znak u imenu: %s.%s" % (server, tool))
    return "%s.%s@%s" % (server, tool, schema_hash(schema))


def parse_cap_id(value: str) -> dict:
    if not isinstance(value, str) or value.count("@") != 1:
        raise CapError("neispravan cap_id: %r" % (value,))
    head, ver = value.split("@", 1)
    if "." not in head:
        raise CapError("neispravan cap_id (nema server.tool): %r" % (value,))
    server, tool = head.split(".", 1)
    return {"server": server, "tool": tool, "version": ver, "id": value}


def parse_svc(uri: str) -> dict:
    """svc://server/surface#op  ->  {server, surface, op}"""
    if not isinstance(uri, str) or not uri.startswith("svc://"):
        raise CapError("neispravan svc uri: %r" % (uri,))
    rest = uri[len("svc://"):]
    server, _, tail = rest.partition("/")
    if not server or not tail:
        raise CapError("svc uri mora imati server i povrsinu: %r" % (uri,))
    surface, _, op = tail.partition("#")
    if not surface or not op:
        raise CapError("svc uri mora imati #operaciju: %r" % (uri,))
    return {"server": server, "surface": surface, "op": op, "uri": uri}


class CapRegistry:
    """Deterministic registry: op -> list of providers, resolve() is total."""

    def __init__(self, priority: Iterable[str] = PRIORITY):
        self.priority = tuple(priority)
        self._order = {t: i for i, t in enumerate(self.priority)}
        self._items: list[dict] = []

    def register(self, op: str, target: str, cap: str | None = None,
                 version: str = "1", ref: Any = None) -> dict:
        if target not in self._order:
            raise CapError("nepoznat target %r (dozvoljeno: %s)" % (target, ",".join(self.priority)))
        item = {"op": op, "target": target, "version": str(version),
                "cap": cap or op, "ref": ref}
        self._items.append(item)
        return item

    def _rank(self, item: dict) -> tuple:
        return (self._order[item["target"]], item["version"], item["cap"])

    def candidates(self, op: str, target: str | None = None) -> list[dict]:
        items = [i for i in self._items if i["op"] == op]
        if target is not None:
            items = [i for i in items if i["target"] == target]
        return sorted(items, key=self._rank)

    def resolve(self, op: str, target: str | None = None) -> dict:
        cands = self.candidates(op, target)
        if not cands:
            raise CapError("E_NO_PROVIDER: %s%s" % (op, " target=%s" % target if target else ""))
        return cands[0]

    def list(self, op: str | None = None) -> list[dict]:
        items = [i for i in self._items if op is None or i["op"] == op]
        return sorted(items, key=lambda i: (i["op"],) + self._rank(i))

    def targets(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for item in self.list():
            out.setdefault(item["op"], [])
            if item["target"] not in out[item["op"]]:
                out[item["op"]].append(item["target"])
        return out


def describe() -> dict:
    return {
        "priority": list(PRIORITY),
        "cap_id": "<server>.<tool>@<schema_sha256[:8]>",
        "svc": "svc://<server>/<surface>#<op>",
        "tie_break": ["priority", "version", "cap_id"],
        "no_schema_token": NO_SCHEMA,
    }
