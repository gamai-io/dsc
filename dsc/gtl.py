"""DSC Trust Layer (GTL/1): one envelope for tool calls, receipts, replay, cache.

Properties:
  * every call produces a PoA receipt (hash + optional HMAC chain + optional Ed25519),
  * side-effect tools require an idempotency key (fail-closed),
  * replay of the same key returns the recorded result, never re-executes,
  * cache is allowed only for deterministic, non-side-effect tools and carries the receipt,
  * trust level is derived from the tool and the inference envelope (EXACT/VERSIONED/STOCHASTIC).
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .detector import poa_record, poa_verify
from .nondeterminism import required_ok, verify_inference
from .sigil import sigil_receipt, sign_receipt, verify_sigil
from .normalize import canonical

LEVELS = ("EXACT", "BATCH_INVARIANT", "VERSIONED", "STOCHASTIC")


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha(s) -> str:
    if not isinstance(s, (str, bytes)):
        s = _canon(s)
    if isinstance(s, str):
        s = s.encode("utf-8", "replace")
    return hashlib.sha256(s).hexdigest()


@dataclass
class ToolSpec:
    name: str
    fn: Callable[[dict], Any]
    side_effect: bool = False
    deterministic: bool = True
    cache_ttl: int = 0
    description: str = ""
    version: str = "1"


class TrustLayer:
    def __init__(self, path: str | None = None, chain_key: bytes | None = None,
                 sign_key=None, key_id: str | None = None, idem_ttl: int = 86400,
                 require_chain: bool = False, require_signature: bool = False,
                 inference_public_key=None):
        self.path = path or os.environ.get("GTL_DB", "/tmp/dsc_gtl.db")
        self.chain_key = chain_key
        self.sign_key = sign_key
        self.key_id = key_id
        self.idem_ttl = idem_ttl
        self.require_chain = bool(require_chain)
        self.require_signature = bool(require_signature)
        self.inference_public_key = inference_public_key
        self._owner = "%s:%d:%s" % (os.uname().nodename, os.getpid(), uuid.uuid4().hex[:8])
        self.tools: dict[str, ToolSpec] = {}
        self._lock = threading.RLock()
        self._prev = None
        self._prev_chain = ""
        self._seq = 0
        self._stats = {"calls": 0, "executed": 0, "replayed": 0, "cached": 0, "errors": 0}
        self._db()

    def _db(self):
        d = os.path.dirname(os.path.abspath(self.path))
        if d:
            os.makedirs(d, exist_ok=True)
        c = sqlite3.connect(self.path, check_same_thread=False, timeout=30,
                            isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=3000")
        c.executescript("""
        CREATE TABLE IF NOT EXISTS receipts(
          seq INTEGER PRIMARY KEY AUTOINCREMENT,
          receipt_id TEXT UNIQUE, receipt_hash TEXT, prev TEXT, chain TEXT,
          tool TEXT, intent_hash TEXT, result_digest TEXT, trust TEXT,
          inference_level TEXT, ts REAL, receipt_json TEXT
        );
        CREATE TABLE IF NOT EXISTS idem(
          key TEXT PRIMARY KEY, intent_hash TEXT, receipt_json TEXT,
          result_json TEXT, ts REAL, state TEXT, owner TEXT
        );
        CREATE TABLE IF NOT EXISTS result_cache(
          key TEXT PRIMARY KEY, result_json TEXT, trust TEXT, receipt_json TEXT,
          ts REAL, expires REAL
        );
        CREATE TABLE IF NOT EXISTS chain_state(
          id INTEGER PRIMARY KEY, prev TEXT, prev_chain TEXT, seq INTEGER
        );
        """)
        for col, typ in (("state", "TEXT"), ("owner", "TEXT")):
            try:
                c.execute("ALTER TABLE idem ADD COLUMN %s %s" % (col, typ))
            except Exception:
                pass
        c.commit()
        row = c.execute("SELECT prev,prev_chain,seq FROM chain_state WHERE id=1").fetchone()
        if row:
            self._prev, self._prev_chain, self._seq = row[0], row[1] or "", int(row[2] or 0)
        else:
            c.execute("INSERT INTO chain_state(id,prev,prev_chain,seq) VALUES(1,NULL,'',0)")
            c.commit()
        self._conn = c

    def register(self, name: str, fn: Callable[[dict], Any], *, side_effect: bool = False,
                 deterministic: bool = True, cache_ttl: int = 0, description: str = "",
                 version: str = "1") -> None:
        self.tools[name] = ToolSpec(name, fn, side_effect, deterministic, int(cache_ttl),
                                    description, str(version))

    def capabilities(self) -> list[dict]:
        return [{"name": t.name, "side_effect": t.side_effect, "deterministic": t.deterministic,
                 "cache_ttl": t.cache_ttl, "description": t.description, "version": t.version}
                for t in sorted(self.tools.values(), key=lambda x: x.name)]

    def _level(self, tool: ToolSpec, inference: dict | None) -> tuple[str, bool]:
        """Return (level, inference_bound). Never let a caller upgrade beyond the tool.

        An unbound EXACT/BATCH_INVARIANT claim is clamped to VERSIONED; a signed/receipt-bound
        inference (receipt_id/verified/signature) may keep its claimed level.
        """
        if not tool.deterministic:
            return "STOCHASTIC", False
        inf = inference or {}
        lvl = inf.get("level")
        bound = False
        if self.inference_public_key is not None and inf.get("signature"):
            try:
                bound = bool(verify_inference(inf, self.inference_public_key))
            except Exception:
                bound = False
        if lvl in ("EXACT", "BATCH_INVARIANT", "VERSIONED"):
            if lvl in ("EXACT", "BATCH_INVARIANT") and not bound:
                return "VERSIONED", False
            return str(lvl), bound
        return "EXACT", bound

    def call(self, tool: str, args: dict | None = None, *, inference: dict | None = None,
             idem_key: str | None = None, use_cache: bool = True,
             art12: dict | None = None, pin: dict | None = None) -> dict:
        args = args or {}
        spec = self.tools.get(tool)
        if spec is None:
            return {"ok": False, "error": "E_UNKNOWN_TOOL", "tool": tool}
        try:
            _canon(args)
        except (TypeError, ValueError):
            return {"ok": False, "error": "E_ARGS_NOT_JSON", "tool": tool}
        intent = {"tool": tool, "args": args}
        ih = _sha(_canon(intent))
        with self._lock:
            self._stats["calls"] += 1
        if spec.side_effect and not idem_key:
            return {"ok": False, "error": "E_IDEM_REQUIRED", "tool": tool,
                    "intent_hash": ih, "hint": "side-effect tool needs idem_key"}

        with self._lock:
            c = self._conn
            if idem_key:
                c.execute("BEGIN IMMEDIATE")
                row = c.execute("SELECT intent_hash,receipt_json,result_json,ts,state,owner "
                                "FROM idem WHERE key=?", (idem_key,)).fetchone()
                if row:
                    if row[0] != ih:
                        c.rollback()
                        return {"ok": False, "error": "E_IDEMPOTENCY_CONFLICT", "tool": tool,
                                "idem_key": idem_key, "intent_hash": ih}
                    state = row[4] or "done"
                    if state == "done":
                        c.commit()
                        self._stats["replayed"] += 1
                        lvl, bound = self._level(spec, inference)
                        return {"ok": True, "result": json.loads(row[2]),
                                "receipt": json.loads(row[1]), "trust": lvl,
                                "inference_bound": bound, "replayed": True, "cached": False}
                    if state == "claimed" and (time.time() - float(row[3] or 0)) < self.idem_ttl:
                        c.commit()
                        return {"ok": False, "error": "E_IN_PROGRESS", "tool": tool,
                                "idem_key": idem_key, "owner": row[5], "intent_hash": ih}
                    c.execute("UPDATE idem SET state='claimed',owner=?,ts=? WHERE key=?",
                              (self._owner, time.time(), idem_key))
                else:
                    c.execute("INSERT INTO idem(key,intent_hash,receipt_json,result_json,ts,state,owner) "
                              "VALUES(?,?,?,?,?,?,?)",
                              (idem_key, ih, None, None, time.time(), "claimed", self._owner))
                c.commit()

            ckey = _sha(_canon({"tool": tool, "version": spec.version, "intent": ih,
                                "env": (inference or {}).get("identity", "none")}))
            if use_cache and spec.cache_ttl > 0 and not spec.side_effect and spec.deterministic:
                row = c.execute("SELECT result_json,trust,receipt_json,expires FROM result_cache WHERE key=?",
                                (ckey,)).fetchone()
                if row and float(row[3]) > time.time():
                    self._stats["cached"] += 1
                    return {"ok": True, "result": json.loads(row[0]), "trust": row[1],
                            "receipt": json.loads(row[2]), "replayed": False, "cached": True}

        t0 = time.time()
        try:
            result = spec.fn(args)
            ok = True
        except Exception as e:
            result = {"error": type(e).__name__, "message": str(e)[:300]}
            ok = False
            with self._lock:
                self._stats["errors"] += 1

        level, inference_bound = self._level(spec, inference)
        with self._lock:
            # CHAIN_LOCK_V1: uzmi write-lock i PONOVO procitaj stanje lanca.
            # Bez ovoga dva procesa/niti procitaju isti seq i lanac se razdvoji.
            c = self._conn
            _locked = False
            for _try in range(50):
                try:
                    c.execute("BEGIN IMMEDIATE")
                    _locked = True
                    break
                except Exception:
                    time.sleep(0.02 * min(_try + 1, 10))
            if not _locked:  # CHAIN_LOCK_V1: nikad ne pisati bez zakljucavanja
                raise RuntimeError("chain lock nije dobijen (db=%s)" % self.path)
            row = c.execute("SELECT prev,prev_chain,seq FROM chain_state WHERE id=1").fetchone()
            if row:
                self._prev, self._prev_chain, self._seq = row[0], row[1] or "", int(row[2] or 0)
            seq = self._seq + 1
            poa = poa_record("tool_call", inputs=intent, tools=[tool], result=result,
                             inference=inference, key=self.chain_key, prev=self._prev,
                             prev_chain=self._prev_chain, seq=seq,
                             extra={"level": level, "inference_bound": inference_bound,
                                    "tool_version": spec.version, "idem_key": idem_key,
                                    "ok": ok, "elapsed_ms": int(round((time.time() - t0) * 1000)),
                                    **({"art12": art12} if art12 else {}),
                                    **({"pin": pin} if pin else {})})
            receipt = poa
            if self.sign_key is not None:
                sig = sign_receipt(sigil_receipt(poa), self.sign_key, self.key_id)
                receipt = {**poa, "sigil": sig}
            self._prev = poa["hash"]
            self._prev_chain = poa["chain"] or ""
            self._seq = seq
            rj = json.dumps(receipt, ensure_ascii=False)
            c = self._conn
            c.execute("INSERT INTO receipts(receipt_id,receipt_hash,prev,chain,tool,intent_hash,"
                      "result_digest,trust,inference_level,ts,receipt_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      (poa["id"], poa["hash"], poa["prev"], poa["chain"], tool, ih,
                       _sha(_canon(result)), level, (inference or {}).get("level"),
                       poa["ts"], rj))
            c.execute("UPDATE chain_state SET prev=?,prev_chain=?,seq=? WHERE id=1",
                      (poa["hash"], poa["chain"] or "", seq))
            if idem_key:
                c.execute("UPDATE idem SET state='done',receipt_json=?,result_json=?,ts=? "
                          "WHERE key=? AND owner=?",
                          (rj, json.dumps(result, ensure_ascii=False), poa["ts"],
                           idem_key, self._owner))
            if ok and use_cache and spec.cache_ttl > 0 and not spec.side_effect and spec.deterministic:
                c.execute("INSERT OR REPLACE INTO result_cache(key,result_json,trust,receipt_json,ts,expires) "
                          "VALUES(?,?,?,?,?,?)",
                          (ckey, json.dumps(result, ensure_ascii=False), level, rj,
                           poa["ts"], poa["ts"] + spec.cache_ttl))
            for _c_try in range(6):  # CHAIN_COMMIT_RETRY_V1
                try:
                    c.commit()
                    break
                except Exception:
                    time.sleep(0.05 * (_c_try + 1))
            else:
                try:
                    c.rollback()
                except Exception:
                    pass
                raise RuntimeError("chain commit nije uspio (db=%s)" % self.path)
            self._stats["executed"] += 1
        return {"ok": ok, "result": result, "receipt": receipt, "trust": level,
                "inference_bound": inference_bound,
                "replayed": False, "cached": False, "intent_hash": ih}

    def verify(self, receipt: dict, result=None, public_key=None) -> dict:
        ok = poa_verify(receipt, key=self.chain_key, prev_chain=receipt.get("prev_chain"))
        if self.require_chain and not receipt.get("chain"):
            ok = False
        sigil = receipt.get("sigil")
        sig_ok = None
        if sigil is not None:
            sig_ok = verify_sigil(sigil, result=result, public_key=public_key,
                                  require_signature=True)["ok"] if public_key else None
        if self.require_signature and (sigil is None or public_key is None or sig_ok is False):
            sig_ok = False
        res_ok = None
        if result is not None:
            res_ok = _sha(_canon(result)) == receipt.get("payload", {}).get("result_hash")
        return {"ok": bool(ok and (sig_ok is not False) and (res_ok is not False)),
                "hash_ok": bool(ok), "signature_ok": sig_ok, "result_ok": res_ok,
                "trust": receipt.get("payload", {}).get("extra", {}).get("level"),
                "inference_bound": receipt.get("payload", {}).get("extra", {}).get("inference_bound")}

    def gc(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            c = self._conn
            n1 = c.execute("DELETE FROM idem WHERE ts<?", (now - self.idem_ttl,)).rowcount
            n2 = c.execute("DELETE FROM result_cache WHERE expires<=?", (now,)).rowcount
            c.commit()
            return {"idem": int(n1), "cache": int(n2)}

    def stats(self) -> dict:
        with self._lock:
            c = self._conn
            receipts = c.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
            d = dict(self._stats)
        d.update({"receipts": int(receipts), "tools": len(self.tools), "db": self.path,
                  "chained": self.chain_key is not None, "signed": self.sign_key is not None})
        return d
