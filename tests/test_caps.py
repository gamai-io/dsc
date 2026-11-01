"""Tests for capability ids, svc:// URIs and deterministic POLY resolution (R3)."""
from __future__ import annotations

import random
import unittest

from dsc import caps


class TestCapId(unittest.TestCase):
    def test_stable_id(self):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        a = caps.cap_id("worker", "core", schema)
        b = caps.cap_id("worker", "core", dict(reversed(list(schema.items()))))
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("worker.core@"))
        self.assertEqual(len(a.split("@")[1]), 8)

    def test_schema_change_changes_id(self):
        a = caps.cap_id("worker", "core", {"type": "object", "properties": {"q": {}}})
        b = caps.cap_id("worker", "core", {"type": "object", "properties": {"q": {}, "k": {}}})
        self.assertNotEqual(a, b)

    def test_no_schema(self):
        self.assertTrue(caps.cap_id("dsc", "echo").endswith("@noschema"))

    def test_rejects_bad_names(self):
        with self.assertRaises(caps.CapError):
            caps.cap_id("a.b", "tool")
        with self.assertRaises(caps.CapError):
            caps.cap_id("", "tool")

    def test_parse_roundtrip(self):
        cid = caps.cap_id("harness", "dag", {"x": 1})
        parsed = caps.parse_cap_id(cid)
        self.assertEqual(parsed["server"], "harness")
        self.assertEqual(parsed["tool"], "dag")


class TestSvcUri(unittest.TestCase):
    def test_parse(self):
        got = caps.parse_svc("svc://harness/dag#run")
        self.assertEqual(got, {"server": "harness", "surface": "dag", "op": "run",
                               "uri": "svc://harness/dag#run"})

    def test_invalid(self):
        for bad in ("https://x/y", "svc://harness", "svc:///dag#run", "svc://harness/dag"):
            with self.assertRaises(caps.CapError):
                caps.parse_svc(bad)


class TestResolution(unittest.TestCase):
    def setUp(self):
        self.reg = caps.CapRegistry()
        self.reg.register("answer", "model", "m.answer@1", version="2")
        self.reg.register("answer", "http", "w.answer@1", version="1")
        self.reg.register("answer", "skill", "d.answer@1", version="1")
        self.reg.register("answer", "sql", "s.answer@1", version="1")

    def test_fixed_priority(self):
        self.assertEqual(self.reg.resolve("answer")["target"], "skill")

    def test_explicit_target(self):
        self.assertEqual(self.reg.resolve("answer", "sql")["target"], "sql")

    def test_unknown_trget(self):
        with self.assertRaises(caps.CapError):
            self.reg.resolve("answer", "quantum")

    def test_missing_op(self):
        with self.assertRaises(caps.CapError) as ctx:
            self.reg.resolve("nema")
        self.assertIn("E_NO_PROVIDER", str(ctx.exception))

    def test_insertion_order_does_not_matter(self):
        items = [("answer", "model", "m", "2"), ("answer", "http", "w", "1"),
                 ("answer", "skill", "d", "1"), ("answer", "sql", "s", "1")]
        picks = set()
        for _ in range(10):
            random.shuffle(items)
            reg = caps.CapRegistry()
            for op, target, cap, ver in items:
                reg.register(op, target, cap, version=ver)
            picks.add(reg.resolve("answer")["target"])
        self.assertEqual(picks, {"skill"})

    def test_version_tiebreak(self):
        reg = caps.CapRegistry()
        reg.register("x", "skill", "a", version="2")
        reg.register("x", "skill", "b", version="1")
        self.assertEqual(reg.resolve("x")["version"], "1")

    def test_list_and_targets(self):
        listed = self.reg.list("answer")
        self.assertEqual(len(listed), 4)
        self.assertEqual([i["target"] for i in listed], ["skill", "hc"][:1] + ["sql", "http", "model"])
        self.assertEqual(self.reg.targets()["answer"], ["skill", "sql", "http", "model"])

    def test_bad_target_rejected(self):
        with self.assertRaises(caps.CapError):
            self.reg.register("op", "quantum", "q")


class TestUnifiedIntegration(unittest.TestCase):
    def test_unified_exposes_caps(self):
        from dsc.unify import Unified
        u = Unified(cache_path="/tmp/_caps_test.db", gtl_path=None)
        names = {c["op"] for c in u.cap_list()}
        self.assertIn("calc", names)
        self.assertIn("core", names)
        self.assertEqual(u.resolve("calc")["target"], "skill")
        self.assertEqual(u.resolve("core")["target"], "http")


if __name__ == "__main__":
    unittest.main()
