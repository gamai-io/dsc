"""Tests for the transport layer (R2). No network: fetch is injected."""
from __future__ import annotations

import unittest
import urllib.error

from dsc import transport as T


class TestTransport(unittest.TestCase):
    def test_caps_cover_worker_routes(self):
        for cap in ("core", "dag", "orch", "sem", "ask"):
            self.assertIn(cap, T.CAPS)

    def test_unknown_cap_raises(self):
        with self.assertRaises(T.CapError):
            T.call("nope")

    def test_core_call_encodes_query(self):
        seen = {}

        def fetch(url, timeout):
            seen["url"] = url
            return {"ok": True, "answer": "246.5"}

        res = T.call("core", {"q": "17 posto od 1450"}, fetch=fetch)
        self.assertIn("/core?", seen["url"])
        self.assertIn("17+posto", seen["url"].replace("%20", "+"))
        self.assertTrue(res["ok"])
        self.assertEqual(res["data"]["answer"], "246.5")
        self.assertEqual(len(res["trace_id"]), 12)
        self.assertGreaterEqual(res["ms"], 0)

    def test_dag_call_passes_levels(self):
        seen = {}

        def fetch(url, timeout):
            seen["url"] = url
            return {"ok": True, "dag": {"nodes": 2}}

        res = T.call("dag", {"goal": "test", "levels": "1"}, fetch=fetch)
        self.assertIn("goal=test", seen["url"])
        self.assertIn("levels=1", seen["url"])
        self.assertTrue(res["ok"])

    def test_execution_error_is_data_not_exception(self):
        def fetch(url, timeout):
            return {"ok": False, "err": "HC timeout"}

        res = T.call("orch", {"path": "/jobs"}, fetch=fetch)
        self.assertFalse(res["ok"])
        self.assertEqual(res["kind"], "execution")
        self.assertIn("HC timeout", res["error"])

    def test_http_4xx_is_protocol_error(self):
        def fetch(url, timeout):
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)

        res = T.call("core", {"q": "x"}, fetch=fetch)
        self.assertFalse(res["ok"])
        self.assertEqual(res["kind"], "protocol")
        self.assertEqual(res["error"], "http_404")

    def test_http_5xx_is_execution_error(self):
        def fetch(url, timeout):
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)

        res = T.call("core", {"q": "x"}, fetch=fetch)
        self.assertEqual(res["kind"], "execution")

    def test_transport_never_raises_on_io(self):
        def fetch(url, timeout):
            raise TimeoutError("boom")

        res = T.call("sem", {"q": "pjeskara"}, fetch=fetch)
        self.assertFalse(res["ok"])
        self.assertIn("TimeoutError", res["error"])

    def test_describe(self):
        d = T.describe()
        self.assertIn("caps", d)
        self.assertEqual(sorted(d["caps"]), sorted(T.CAPS))


if __name__ == "__main__":
    unittest.main()
