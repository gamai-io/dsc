"""Two-stage semantic cache with epochs, domain TTL, and evidence metadata.

Stage 1: exact canonical hash.
Stage 2: SimHash candidates -> int8 cosine -> lexical gate -> polarity/number gate.

Hard rule: queries that carry money/percent/physical units or decimal values are exact-only.
A cache hit never claims truth: it carries cached_at, epoch, sources, evidence_id, trust.
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time

from . import hdc
from .fingerprints import sha256_hex, simhash64
from .lexical import token_overlap
from .normalize import (
    canonical, features, has_money_unit_percent, numeric_signature,
)

MASK64 = (1 << 64) - 1
DEFAULT_TTL = {"price": 30, "stock": 180, "doc": 86400, "general": 900}
_TAU = 18


def _signed64(x: int) -> int:
    x &= MASK64
    return x - (1 << 64) if x >= (1 << 63) else x


class SemanticCache:
    PROFILES = {"balanced": 0.72, "strict": 0.80, "paranoid": 0.86}

    def __init__(self, path: str | None = None, embed_fn=None, dim: int = hdc.DIM,
                 threshold: float | None = None, max_entries: int = 5000,
                 ttl: dict | None = None, enabled: bool = True,
                 min_answer_len: int = 8, profile: str | None = None):
        self.path = path or os.environ.get("DSC_SEMANTIC_CACHE", "/tmp/dsc_cache.db")
        self.embed_fn = embed_fn
        self.dim = dim
        base = 0.86 if embed_fn else 0.72
        self.threshold = threshold if threshold is not None else self.PROFILES.get(profile or "", base)
        self.max_entries = max_entries
        self.ttl = dict(DEFAULT_TTL)
        if ttl:
            self.ttl.update(ttl)
        self.enabled = enabled
        self.min_answer_len = int(min_answer_len)
        self._lock = threading.RLock()
        self._conn = None
        self._stats = {"get": 0, "hit_exact": 0, "hit_semantic": 0, "miss": 0, "put": 0, "evict": 0}
        self._db()

    def _db(self):
        if self._conn is not None:
            return self._conn
        d = os.path.dirname(os.path.abspath(self.path))
        if d:
            os.makedirs(d, exist_ok=True)
        c = sqlite3.connect(self.path, check_same_thread=False, timeout=5)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=3000")
        c.execute("PRAGMA synchronous=NORMAL")
        c.executescript("""
        CREATE TABLE IF NOT EXISTS entries(
          h TEXT PRIMARY KEY, q TEXT, qc TEXT, sh INTEGER, vec BLOB, dim INTEGER,
          answer TEXT, meta TEXT, created REAL, expires REAL, epoch TEXT, dom TEXT,
          sid TEXT, scope TEXT, trust TEXT, exact_only INTEGER DEFAULT 0,
          nsig TEXT, neg INTEGER DEFAULT 0, hits INTEGER DEFAULT 0, last REAL,
          env_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_entries_exp ON entries(expires);
        CREATE INDEX IF NOT EXISTS idx_entries_dom ON entries(dom, expires);
        CREATE INDEX IF NOT EXISTS idx_entries_sid ON entries(sid);
        CREATE INDEX IF NOT EXISTS idx_entries_env ON entries(env_id);
        CREATE TABLE IF NOT EXISTS epochs(domain TEXT PRIMARY KEY, ver INTEGER, ts REAL);
        CREATE TABLE IF NOT EXISTS emb(k TEXT PRIMARY KEY, vec BLOB, dim INTEGER, ts REAL);
        CREATE TABLE IF NOT EXISTS stats(k TEXT PRIMARY KEY, v INTEGER);
        """)
        try:
            c.execute("ALTER TABLE entries ADD COLUMN env_id TEXT")
        except Exception:
            pass
        c.commit()
        self._conn = c
        return c

    def _epoch(self, domain: str) -> int:
        row = self._db().execute("SELECT ver FROM epochs WHERE domain=?", (domain,)).fetchone()
        return int(row[0]) if row else 0

    def set_epoch(self, domain: str, ver: int | None = None) -> int:
        with self._lock:
            c = self._db()
            cur = self._epoch(domain)
            nv = (cur + 1) if ver is None else int(ver)
            c.execute("INSERT INTO epochs(domain,ver,ts) VALUES(?,?,?) "
                      "ON CONFLICT(domain) DO UPDATE SET ver=excluded.ver, ts=excluded.ts",
                      (domain, nv, time.time()))
            c.commit()
            return nv

    def _emb_key(self, text: str) -> str:
        return sha256_hex("emb|%d|%s" % (self.dim, canonical(text)))

    def _embed(self, text: str) -> bytes:
        key = self._emb_key(text)
        with self._lock:
            row = self._db().execute("SELECT vec FROM emb WHERE k=?", (key,)).fetchone()
        if row:
            return row[0]
        vec = None
        if self.embed_fn is not None:
            try:
                vs = self.embed_fn([text])
                if vs:
                    raw = list(vs[0])
                    n = math.sqrt(sum(x * x for x in raw))
                    if n:
                        raw = [x / n for x in raw]
                    vec = hdc.quantize(raw)
            except Exception:
                vec = None
        if vec is None:
            vec = hdc.vector(text, self.dim)
        with self._lock:
            self._db().execute("INSERT OR REPLACE INTO emb(k,vec,dim,ts) VALUES(?,?,?,?)",
                               (key, vec, len(vec), time.time()))
            self._db().commit()
        return vec

    def _ttl_for(self, domain: str, ttl: int | None) -> int:
        if ttl is not None:
            return int(ttl)
        return int(self.ttl.get(domain, self.ttl.get("general", 900)))

    def _exact_key(self, qc: str, sid, global_scope: bool = False,
                   envelope_id: str | None = None) -> str:
        env = "|env:%s" % (envelope_id or "")
        if global_scope:
            return sha256_hex("global||" + qc + env)
        return sha256_hex("sid|%s|%s%s" % (sid or "", qc, env))

    def _row_valid(self, row, now: float, domain: str) -> bool:
        if row is None:
            return False
        expires, epoch = row[6], row[7]
        return float(expires) > now and int(epoch) == self._epoch(domain)

    def _payload(self, row, how: str) -> dict:
        (h, q, qc, answer, meta, created, expires, epoch, dom, sid, scope, trust, hits) = row
        m = {}
        try:
            m = json.loads(meta or "{}")
        except Exception:
            m = {}
        m.update({"cache": how, "cached_at": created, "expires_at": expires, "epoch": epoch,
                  "domain": dom, "scope": scope, "trust": trust, "cache_hits": int(hits or 0),
                  "cache_key": h})
        m.setdefault("evidence_id", None)
        m.setdefault("sources", [])
        return {"answer": answer, "meta": m, "cached": True, "cache": how}

    def get(self, query: str, sid: str | None = None, domain: str = "general",
            now: float | None = None, envelope_id: str | None = None) -> dict | None:
        if not self.enabled or not query or not str(query).strip():
            return None
        now = time.time() if now is None else now
        with self._lock:
            c = self._db()
            qc = canonical(query)
            self._stats["get"] += 1
            for glob in ((False, True) if sid else (True,)):
                h = self._exact_key(qc, sid, global_scope=glob, envelope_id=envelope_id)
                row = c.execute(
                    "SELECT h,q,qc,answer,meta,created,expires,epoch,dom,sid,scope,trust,hits "
                    "FROM entries WHERE h=?", (h,)).fetchone()
                if self._row_valid(row, now, domain):
                    c.execute("UPDATE entries SET hits=hits+1,last=? WHERE h=?", (now, h))
                    c.commit()
                    self._stats["hit_exact"] += 1
                    row = row[:12] + (int(row[12]) + 1,)
                    return self._payload(row, "exact")

        qf = features(query)
        if qf["action"] or qf["fresh"] or qf["multi"]:
            self._stats["miss"] += 1
            return None
        exact_only = has_money_unit_percent(query) or any(
            ("." in n["raw"] or "," in n["raw"]) for n in qf["numbers"])
        if exact_only:
            self._stats["miss"] += 1
            return None

        qvec = self._embed(query)
        qsig = tuple(sorted(qf["numeric_signature"]))
        with self._lock:
            c = self._db()
            ep = self._epoch(domain)
            rows = c.execute(
                "SELECT h,q,qc,sh,vec,dim,answer,meta,created,expires,epoch,dom,sid,scope,trust,hits,nsig,neg "
                "FROM entries WHERE dom=? AND epoch=? AND expires>? AND exact_only=0 "
                "AND (scope='global' OR sid=?) "
                "AND ((env_id IS NULL AND ? IS NULL) OR env_id=?)",
                (domain, ep, now, sid, envelope_id, envelope_id)).fetchall()
        cands = []
        qsh = simhash64(query)
        for r in rows:
            sh = int(r[3]) & MASK64
            ham = (sh ^ qsh).bit_count()
            if ham <= _TAU:
                cands.append((ham, r))
        cands.sort(key=lambda x: x[0])
        best = None
        for ham, r in cands[:48]:
            try:
                nsig = tuple(tuple(x) for x in json.loads(r[16] or "[]"))
            except Exception:
                nsig = ()
            if tuple(sorted(nsig)) != qsig:
                continue
            if int(r[17] or 0) != int(qf["negations"] > 0):
                continue
            cos = hdc.cosine_q(qvec, r[4])
            if cos < self.threshold:
                continue
            ov = token_overlap(query, r[1])
            if ov < 0.34 and cos < 0.94:
                continue
            score = cos + 0.10 * ov
            if best is None or score > best[0]:
                best = (score, cos, ov, r)
        if best is None:
            self._stats["miss"] += 1
            return None
        score, cos, ov, r = best
        if str(r[14]) != "verified":
            self._stats["miss"] += 1
            return None
        with self._lock:
            c = self._db()
            c.execute("UPDATE entries SET hits=hits+1,last=? WHERE h=?", (now, r[0]))
            c.commit()
        self._stats["hit_semantic"] += 1
        row = (r[0], r[1], r[2], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15] + 1)
        out = self._payload(row, "semantic")
        out["meta"].update({"sim": round(cos, 6), "overlap": round(ov, 6), "score": round(score, 6)})
        return out

    def put(self, query: str, answer: str, sid: str | None = None, domain: str = "general",
            sources: list | None = None, evidence_id: str | None = None,
            trust: str = "verified", ttl: int | None = None, now: float | None = None,
            envelope_id: str | None = None, inference_level: str | None = None) -> bool:
        if not self.enabled or not query or not answer or len(str(answer)) < self.min_answer_len:
            return False
        now = time.time() if now is None else now
        qf = features(query)
        if qf["action"] or qf["multi"]:
            return False
        exact_only = has_money_unit_percent(query) or any(
            ("." in n["raw"] or "," in n["raw"]) for n in qf["numbers"])
        if qf["fresh"] and not exact_only:
            return False
        if inference_level == "STOCHASTIC":
            exact_only = True
        scope = "sid" if sid else "global"
        h = self._exact_key(canonical(query), sid, global_scope=(scope == "global"),
                            envelope_id=envelope_id)
        vec = self._embed(query)
        sh = simhash64(query)
        meta = {"domain": domain, "sources": sources or [], "evidence_id": evidence_id,
                "trust": trust, "as_of": now, "exact_only": bool(exact_only),
                "envelope_id": envelope_id, "inference_level": inference_level,
                "numeric_signature": list(qf["numeric_signature"]),
                "polarity": "neg" if qf["negations"] else "pos"}
        ttl_s = self._ttl_for(domain, ttl)
        with self._lock:
            c = self._db()
            c.execute(
                "INSERT INTO entries(h,q,qc,sh,vec,dim,answer,meta,created,expires,epoch,dom,sid,scope,trust,"
                "exact_only,nsig,neg,hits,last,env_id) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?) "
                "ON CONFLICT(h) DO UPDATE SET q=excluded.q,qc=excluded.qc,sh=excluded.sh,vec=excluded.vec,"
                "dim=excluded.dim,answer=excluded.answer,meta=excluded.meta,expires=excluded.expires,"
                "epoch=excluded.epoch,dom=excluded.dom,sid=excluded.sid,scope=excluded.scope,"
                "trust=excluded.trust,exact_only=excluded.exact_only,nsig=excluded.nsig,neg=excluded.neg,"
                "last=excluded.last,env_id=excluded.env_id",
                (h, query, canonical(query), _signed64(sh), vec, len(vec), answer, json.dumps(meta),
                 now, now + ttl_s, self._epoch(domain), domain, sid, scope, trust,
                 1 if exact_only else 0, json.dumps(meta["numeric_signature"]),
                 1 if qf["negations"] else 0, now, envelope_id))
            n = c.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            if n > self.max_entries:
                k = n - self.max_entries
                c.execute("DELETE FROM entries WHERE h IN (SELECT h FROM entries "
                          "ORDER BY ((hits+1.0)/(((? - last)/3600.0)+1.0)) ASC LIMIT ?)", (now, k))
                self._stats["evict"] += k
            c.commit()
        self._stats["put"] += 1
        return True

    def invalidate(self, domain: str | None = None) -> int:
        with self._lock:
            c = self._db()
            if domain is None:
                n = c.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
                c.execute("DELETE FROM entries")
            else:
                n = c.execute("SELECT COUNT(*) FROM entries WHERE dom=?", (domain,)).fetchone()[0]
                c.execute("DELETE FROM entries WHERE dom=?", (domain,))
                c.execute("INSERT INTO epochs(domain,ver,ts) VALUES(?,?,?) "
                          "ON CONFLICT(domain) DO UPDATE SET ver=ver+1, ts=excluded.ts",
                          (domain, 1, time.time()))
            c.commit()
            return int(n)

    def gc(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            c = self._db()
            n = c.execute("SELECT COUNT(*) FROM entries WHERE expires<=?", (now,)).fetchone()[0]
            c.execute("DELETE FROM entries WHERE expires<=?", (now,))
            c.execute("DELETE FROM emb WHERE ts<?", (now - 7 * 86400,))
            c.commit()
            return int(n)

    def stats(self) -> dict:
        with self._lock:
            c = self._db()
            total = c.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            exp = c.execute("SELECT COUNT(*) FROM entries WHERE expires<=?", (time.time(),)).fetchone()[0]
            size = 0
            try:
                size = os.path.getsize(self.path)
            except OSError:
                pass
            d = dict(self._stats)
        d.update({"entries": int(total), "expired": int(exp), "db": self.path,
                  "size_bytes": size, "threshold": self.threshold, "ttl": self.ttl,
                  "enabled": self.enabled})
        d["hit_rate"] = round(100.0 * (d["hit_exact"] + d["hit_semantic"]) / d["get"], 2) if d["get"] else 0.0
        return d

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
