"""Tests for PIN (R4): validity, float rejection, equality, receipt binding."""
from __future__ import annotations

import unittest

from dsc import art12, pin
from dsc.gtl import TrustLayer


class TestPin(unittest.TestCase):
    def test_make_and_validate(self):
        p = pin.make("qwen2.5-7b-awq", "1.0", seed=42, temperature="0.0",
                     quant="awq", batch=8)
        self.assertTrue(pin.validate(p)["ok"])
        self.assertEqual(p["batch"], 8)
        self.assertEqual(p["temperature"], "0.0")

    def test_float_temperature_rejected(self):
        with self.assertRaises(pin.PinError):
            pin.make("m", "1", seed=1, temperature=0.0)

    def test_float_extra_rejected(self):
        with self.assertRaises(pin.PinError):
            pin.make("m", "1", seed=1, temperature="0.0", top_p=0.9)

    def test_missing_fields(self):
        self.assertEqual(pin.validate({"model": "m"})["missing"],
                         ["version", "seed", "temperature"])
        self.assertFalse(pin.validate({"model": "m"})["ok"])

    def test_same_ignores_extra_metadata(self):
        a = pin.make("m", "1", seed=7, temperature="0.0", batch=1)
        b = pin.make("m", "1", seed=7, temperature="0.0", batch=1, note="x")
        self.assertTrue(pin.same(a, b))

    def test_same_detects_batch_drift(self):
        a = pin.make("m", "1", seed=7, temperature="0.0", batch=1)
        b = pin.make("m", "1", seed=7, temperature="0.0", batch=32)
        self.assertFalse(pin.same(a, b))

    def test_same_detects_version_drift(self):
        a = pin.make("m", "1.0", seed=7, temperature="0.0")
        b = pin.make("m", "1.1", seed=7, temperature="0.0")
        self.assertFalse(pin.same(a, b))


class TestPinInReceipt(unittest.TestCase):
    def setUp(self):
        self.gtl = TrustLayer(path=None, chain_key=b"k" * 32)
        self.gtl.register("model_call", lambda a: {"text": "odgovor"})

    def test_pin_is_hashed_into_receipt(self):
        p = pin.make("qwen2.5-7b-awq", "1.0", seed=1, temperature="0.0")
        rec = self.gtl.call("model_call", {}, pin=p)["receipt"]
        self.assertEqual(rec["payload"]["extra"]["pin"], p)
        self.assertTrue(self.gtl.verify(rec)["ok"])

    def test_pin_tamper_breaks_receipt(self):
        p = pin.make("qwen2.5-7b-awq", "1.0", seed=1, temperature="0.0")
        rec = self.gtl.call("model_call", {}, pin=p)["receipt"]
        rec["payload"]["extra"]["pin"]["seed"] = 999
        self.assertFalse(self.gtl.verify(rec)["ok"])

    def test_no_pin_by_default(self):
        rec = self.gtl.call("model_call", {})["receipt"]
        self.assertNotIn("pin", rec["payload"]["extra"])

    def test_pin_and_art12_together(self):
        p = pin.make("m", "1", seed=1, temperature="0.0")
        blk = art12.record({"q": "x"}, "demo_db", "operator")
        rec = self.gtl.call("model_call", {}, pin=p, art12=blk)["receipt"]
        extra = rec["payload"]["extra"]
        self.assertIn("pin", extra)
        self.assertIn("art12", extra)
        self.assertTrue(self.gtl.verify(rec)["ok"])


if __name__ == "__main__":
    unittest.main()
