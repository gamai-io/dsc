import unittest

import os
import tempfile

from dsc.detector import (
    fleet_determinism, load_key, poa_chain, poa_record, poa_verify, semantic_drift,
    verdict, verify_anchor, write_anchor,
)


class TestDetector(unittest.TestCase):
    def test_poa(self):
        r = poa_record("answer", {"q": "x"}, [{"id": 1}], ["sql"], {"a": 1})
        self.assertTrue(poa_verify(r))
        r["payload"]["inputs"]["q"] = "y"
        self.assertFalse(poa_verify(r))

    def test_chain(self):
        a = poa_record("a", {"x": 1})
        b = poa_record("b", {"x": 2}, prev=a["hash"])
        self.assertTrue(poa_chain([a, b])["ok"])
        b["payload"]["x"] = 3
        self.assertFalse(poa_chain([a, b])["ok"])

    def test_fds(self):
        stable = fleet_determinism({"a": ["x", "y"], "b": ["x", "y"]})
        self.assertEqual(stable["verdict"], "STABLE")
        div = fleet_determinism({"a": ["x"], "b": ["y"]})
        self.assertEqual(div["verdict"], "DIVERGENT")

    def test_sdm(self):
        old = semantic_drift([{"ts": 0, "claims": [{"key": "k", "supported": True}]}], now=10**9)
        self.assertGreater(old["drift"], 0.9)
        fresh = semantic_drift([{"ts": 1000, "claims": [{"key": "k", "supported": True, "confidence": 1}]}], now=1000)
        self.assertLess(fresh["drift"], 0.1)

    def test_verdict(self):
        self.assertEqual(verdict(False)["verdict"], "UNPROVEN")
        self.assertEqual(verdict(True, {"mean": 1.0}, {"drift": 0.0})["verdict"], "TRUSTED")
        self.assertEqual(verdict(True, {"mean": 0.5}, {"drift": 0.0})["verdict"], "DOUBTFUL")

    def _chain(self, key=None, n=3):
        rs, prev, pc = [], None, ""
        for i in range(n):
            r = poa_record("a", {"i": i}, key=key, prev=prev, prev_chain=pc, seq=i + 1)
            rs.append(r)
            prev, pc = r["hash"], r["chain"] or ""
        return rs

    def test_hmac_chain(self):
        key = b"k" * 32
        rs = self._chain(key)
        self.assertTrue(poa_chain(rs, key=key)["ok"])
        rs[1]["payload"]["inputs"]["i"] = 99
        self.assertFalse(poa_chain(rs, key=key)["ok"])

    def test_anchor(self):
        key = b"k" * 32
        rs = self._chain(key)
        fd, path = tempfile.mkstemp()
        os.close(fd)
        os.unlink(path)
        try:
            write_anchor(path, rs)
            self.assertTrue(verify_anchor(path, rs)["ok"])
            self.assertFalse(verify_anchor(path, rs[:-1])["ok"])
            r0 = rs[0]["payload"]["inputs"]["i"]
            rs[0]["payload"]["inputs"]["i"] = 42
            self.assertTrue(verify_anchor(path, rs)["ok"])
            self.assertFalse(poa_chain(rs, key=key, anchor=path)["ok"])
            rs[0]["payload"]["inputs"]["i"] = r0
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_key_permissions(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        try:
            os.chmod(path, 0o600)
            self.assertTrue(load_key(path) is not None)
            os.chmod(path, 0o644)
            with self.assertRaises(PermissionError):
                load_key(path)
        finally:
            os.unlink(path)

    def test_inference_downgrade(self):
        r = verdict(True, {"mean": 1.0}, {"drift": 0.0},
                    inference={"level": "STOCHASTIC"}, requirement="EXACT")
        self.assertEqual(r["verdict"], "DOUBTFUL")
        self.assertIn("inference", r["reason"])
        ok = verdict(True, {"mean": 1.0}, {"drift": 0.0},
                     inference={"level": "EXACT"}, requirement="EXACT")
        self.assertEqual(ok["verdict"], "TRUSTED")


if __name__ == "__main__":
    unittest.main()
