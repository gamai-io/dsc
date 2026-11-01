import json
import os
import tempfile
import unittest

from dsc.gtl import TrustLayer, _canon, _sha
from dsc.sigil import generate_keypair, load_private, load_public
from dsc.nondeterminism import local_vllm_env, receipt, sign_inference


class TestGTL(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.path)
        self.t = TrustLayer(self.path)

    def tearDown(self):
        for suf in ("", "-wal", "-shm"):
            try:
                os.unlink(self.path + suf)
            except OSError:
                pass

    def test_calc_cache_and_verify(self):
        calls = {"n": 0}

        def fn(a):
            calls["n"] += 1
            return a["a"] + a["b"]

        self.t.register("calc", fn, cache_ttl=60)
        r1 = self.t.call("calc", {"a": 2, "b": 3})
        r2 = self.t.call("calc", {"a": 2, "b": 3})
        self.assertEqual(r1["result"], 5)
        self.assertFalse(r1["cached"])
        self.assertTrue(r2["cached"])
        self.assertEqual(calls["n"], 1)
        self.assertTrue(self.t.verify(r1["receipt"], result=5)["ok"])

    def test_side_effect_idempotency(self):
        self.t.register("mail", lambda a: {"sent": a["to"]}, side_effect=True)
        self.assertFalse(self.t.call("mail", {"to": "x"})["ok"])
        r1 = self.t.call("mail", {"to": "x"}, idem_key="k1")
        r2 = self.t.call("mail", {"to": "x"}, idem_key="k1")
        self.assertTrue(r1["ok"])
        self.assertTrue(r2["replayed"])
        self.assertEqual(r1["receipt"]["hash"], r2["receipt"]["hash"])
        bad = self.t.call("mail", {"to": "y"}, idem_key="k1")
        self.assertEqual(bad["error"], "E_IDEMPOTENCY_CONFLICT")

    def test_tamper(self):
        self.t.register("calc", lambda a: a["a"] + 1)
        r = self.t.call("calc", {"a": 1})
        self.assertTrue(self.t.verify(r["receipt"], result=2)["ok"])
        self.assertFalse(self.t.verify(r["receipt"], result=3)["ok"])
        r["receipt"]["payload"]["result_hash"] = "x"
        self.assertFalse(self.t.verify(r["receipt"])["ok"])

    def test_levels(self):
        self.t.register("det", lambda a: 1)
        self.t.register("nd", lambda a: 2, deterministic=False)
        self.assertEqual(self.t.call("det", {})["trust"], "EXACT")
        self.assertEqual(self.t.call("det", {}, inference={"level": "VERSIONED"})["trust"], "VERSIONED")
        self.assertEqual(self.t.call("nd", {})["trust"], "STOCHASTIC")

    def test_idem_in_progress(self):
        self.t.register("mail", lambda a: {"sent": a["to"]}, side_effect=True)
        ih = _sha(_canon({"tool": "mail", "args": {"to": "x"}}))
        self.t._conn.execute("INSERT INTO idem(key,intent_hash,receipt_json,result_json,ts,state,owner) "
                             "VALUES(?,?,?,?,?,?,?)",
                             ("k9", ih, None, None, __import__("time").time(), "claimed", "other"))
        self.t._conn.commit()
        r = self.t.call("mail", {"to": "x"}, idem_key="k9")
        self.assertEqual(r["error"], "E_IN_PROGRESS")

    def test_strict_chain_and_signature(self):
        d = tempfile.mkdtemp()
        priv, pub = os.path.join(d, "k.pem"), os.path.join(d, "k.pub")
        kp = generate_keypair(priv, pub)
        t = TrustLayer(self.path + ".strict", chain_key=b"c" * 32,
                       sign_key=load_private(priv), key_id=kp["key_id"],
                       require_chain=True, require_signature=True)
        try:
            t.register("calc", lambda a: a["a"] + 1)
            r = t.call("calc", {"a": 1})
            self.assertTrue(t.verify(r["receipt"], result=2, public_key=load_public(pub))["ok"])
            self.assertFalse(t.verify(r["receipt"], result=2)["ok"])
            bad = json.loads(json.dumps(r["receipt"]))
            bad.pop("sigil", None)
            bad["chain"] = None
            self.assertFalse(t.verify(bad, result=2, public_key=load_public(pub))["ok"])
        finally:
            for suf in ("", "-wal", "-shm"):
                try:
                    os.unlink(self.path + ".strict" + suf)
                except OSError:
                    pass

    def test_level_clamp(self):
        self.t.register("det", lambda a: 1)
        self.assertEqual(self.t.call("det", {}, inference={"level": "EXACT"})["trust"], "VERSIONED")
        self.assertEqual(self.t.call("det", {}, inference={"level": "EXACT", "receipt_id": "r1"})["trust"], "VERSIONED")
        self.assertFalse(self.t.call("det", {}, inference={"level": "EXACT"})["inference_bound"])
        d = tempfile.mkdtemp()
        priv, pub = os.path.join(d, "k.pem"), os.path.join(d, "k.pub")
        kp = generate_keypair(priv, pub)
        t2 = TrustLayer(self.path + ".inf", inference_public_key=load_public(pub))
        try:
            t2.register("det", lambda a: 1)
            signed = sign_inference(receipt("q", "a", local_vllm_env("m")), load_private(priv), kp["key_id"])
            r = t2.call("det", {}, inference=signed)
            self.assertEqual(r["trust"], "EXACT")
            self.assertTrue(r["inference_bound"])
            bad = dict(signed)
            bad["level"] = "BATCH_INVARIANT"
            self.assertEqual(t2.call("det", {}, inference=bad)["trust"], "VERSIONED")
        finally:
            for suf in ("", "-wal", "-shm"):
                try:
                    os.unlink(self.path + ".inf" + suf)
                except OSError:
                    pass

    def test_error_not_cached(self):
        def boom(a):
            raise RuntimeError("x")
        self.t.register("boom", boom, cache_ttl=60)
        r1 = self.t.call("boom", {})
        r2 = self.t.call("boom", {})
        self.assertFalse(r1["ok"])
        self.assertFalse(r2["ok"])
        self.assertFalse(r2["cached"])
        self.assertEqual(self.t._conn.execute("SELECT COUNT(*) FROM result_cache").fetchone()[0], 0)

    def test_restart_chain(self):
        self.t.register("calc", lambda a: a["a"] + 1)
        r1 = self.t.call("calc", {"a": 1})
        t2 = TrustLayer(self.path)
        t2.register("calc", lambda a: a["a"] + 1)
        r2 = t2.call("calc", {"a": 2})
        self.assertEqual(r2["receipt"]["prev"], r1["receipt"]["hash"])
        self.assertTrue(t2.verify(r2["receipt"], result=3)["ok"])

    def test_tool_version_cache(self):
        self.t.register("t", lambda a: 1, cache_ttl=60, version="1")
        r1 = self.t.call("t", {})
        self.t.register("t", lambda a: 2, cache_ttl=60, version="2")
        r2 = self.t.call("t", {})
        self.assertEqual(r1["result"], 1)
        self.assertEqual(r2["result"], 2)
        self.assertFalse(r2["cached"])

    def test_signature(self):
        d = tempfile.mkdtemp()
        priv, pub = os.path.join(d, "k.pem"), os.path.join(d, "k.pub")
        kp = generate_keypair(priv, pub)
        t2 = TrustLayer(self.path + ".sig", sign_key=load_private(priv), key_id=kp["key_id"])
        try:
            t2.register("calc", lambda a: a["a"] * 2)
            r = t2.call("calc", {"a": 4})
            self.assertEqual(r["trust"], "EXACT")
            self.assertTrue(t2.verify(r["receipt"], result=8, public_key=load_public(pub))["ok"])
            self.assertFalse(t2.verify(r["receipt"], result=9, public_key=load_public(pub))["ok"])
        finally:
            try:
                os.unlink(self.path + ".sig")
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
