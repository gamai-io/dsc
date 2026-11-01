"""Tests for RFC 8785 JCS subset + Art. 12 record block + receipt integration."""
from __future__ import annotations

import unittest

from dsc import art12, jcs
from dsc.gtl import TrustLayer


class TestJCS(unittest.TestCase):
    def test_no_whitespace_and_sorted(self):
        self.assertEqual(jcs.dumps({"b": 1, "a": 2}), '{"a":2,"b":1}')

    def test_nested(self):
        val = {"z": [1, 2, {"y": None, "x": True}], "a": "ć"}
        self.assertEqual(jcs.dumps(val), '{"a":"ć","z":[1,2,{"x":true,"y":null}]}')

    def test_float_rejected(self):
        with self.assertRaises(jcs.JCSError):
            jcs.dumps({"price": 10.5})

    def test_money_as_string_allowed(self):
        self.assertEqual(jcs.dumps({"price": "10.50"}), '{"price":"10.50"}')

    def test_digest_stable(self):
        a = jcs.digest({"b": 1, "a": [1, 2]})
        b = jcs.digest({"a": [1, 2], "b": 1})
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)


class TestArt12(unittest.TestCase):
    def test_record_has_all_fields(self):
        blk = art12.record({"kupac": "ACME"}, "demo_db", "operator")
        self.assertEqual(set(art12.REQUIRED), set(blk))
        self.assertTrue(art12.validate(blk)["ok"])

    def test_input_is_hashed_not_stored(self):
        blk = art12.record({"kupac": "ACME"}, "demo_db", "operator")
        self.assertNotIn("kupac", blk)
        self.assertEqual(len(blk["input_data_hash"]), 64)

    def test_missing_field_detected(self):
        blk = art12.record({"a": 1}, "demo_db", "operator")
        blk.pop("verifier_person")
        check = art12.validate(blk)
        self.assertFalse(check["ok"])
        self.assertIn("verifier_person", check["missing"])

    def test_bad_hash_detected(self):
        blk = art12.record({"a": 1}, "demo_db", "operator")
        blk["input_data_hash"] = "xyz"
        self.assertIn("input_data_hash", art12.validate(blk)["invalid"])

    def test_end_before_start_rejected(self):
        with self.assertRaises(ValueError):
            art12.record({"a": 1}, "demo_db", "operator", started=100.0, ended=50.0)


class TestReceiptWithArt12(unittest.TestCase):
    def test_receipt_carries_and_hashes_art12(self):
        gtl = TrustLayer(path=None, chain_key=b"k" * 32)
        gtl.register("echo", lambda a: {"echo": a.get("x")})
        blk = art12.record({"x": 1}, "demo_db", "operator")
        out = gtl.call("echo", {"x": 1}, art12=blk)
        self.assertTrue(out["ok"])
        rec = out["receipt"]
        self.assertEqual(rec["payload"]["extra"]["art12"], blk)
        self.assertIsInstance(rec["payload"]["extra"]["elapsed_ms"], int)
        self.assertTrue(gtl.verify(rec)["ok"])

    def test_tampered_art12_breaks_receipt(self):
        gtl = TrustLayer(path=None, chain_key=b"k" * 32)
        gtl.register("echo", lambda a: {"echo": a.get("x")})
        blk = art12.record({"x": 1}, "demo_db", "operator")
        rec = gtl.call("echo", {"x": 1}, art12=blk)["receipt"]
        rec["payload"]["extra"]["art12"]["verifier_person"] = "napadac"
        self.assertFalse(gtl.verify(rec)["ok"])

    def test_no_art12_by_default(self):
        gtl = TrustLayer(path=None, chain_key=b"k" * 32)
        gtl.register("echo", lambda a: {"echo": a.get("x")})
        rec = gtl.call("echo", {"x": 1})["receipt"]
        self.assertNotIn("art12", rec["payload"]["extra"])


if __name__ == "__main__":
    unittest.main()
