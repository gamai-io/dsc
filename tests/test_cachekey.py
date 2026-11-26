"""Contract tests for the single cache key (``dsc.cachekey.request_key``).

The properties, not the digest, are what the stack depends on:

* stable — same request, same key, in any process / hash seed;
* framed — tenant, session and context are JSON values, so a ``|`` in one
  field can never be confused with a separator in another;
* scoped — no tenant, session, envelope or domain shares a key;
* no collision between different requests, including two different amounts;
* scheme-tagged — old rows are recognisable and simply miss.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

from dsc.cachekey import SCHEME, is_current, normalize_for_key, request_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestShape(unittest.TestCase):
    def test_scheme_prefix_and_hex(self):
        k = request_key("zdravo")
        self.assertTrue(k.startswith("v2:"))
        self.assertEqual(len(k), 3 + 64)
        self.assertRegex(k[3:], r"^[0-9a-f]{64}$")
        self.assertEqual(SCHEME, "v2")
        self.assertTrue(is_current(k))

    def test_no_plaintext_in_key(self):
        k = request_key("PDV 17% na neto 2400", session="tenant-a")
        self.assertNotIn("2400", k)
        self.assertNotIn("tenant", k)
        self.assertNotIn("pdv", k.lower())

    def test_legacy_shapes_are_not_current(self):
        # old fastpath: "q:" + sha256; old offline: bare md5; old dsc: bare sha256
        for old in ("q:" + "a" * 64, "b" * 32, "c" * 64,
                    "v1:" + "d" * 64, "", None):
            self.assertFalse(is_current(old), old)

    def test_stable_across_processes(self):
        """PYTHONHASHSEED must not matter: no dict/set iteration in the payload."""
        code = ("import json;from dsc.cachekey import request_key;"
                "print(json.dumps(request_key('kupi 2 mixera', session='s1',"
                " context={'env': 'E', 'domain': 'doc'})))")
        outs = []
        for seed in ("0", "1", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=ROOT)
            p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, env=env, cwd=ROOT)
            self.assertEqual(p.returncode, 0, p.stderr)
            outs.append(json.loads(p.stdout))
        self.assertEqual(outs[0], request_key("kupi 2 mixera", session="s1",
                                              context={"env": "E", "domain": "doc"}))
        self.assertEqual(len(set(outs)), 1)


class TestCollisions(unittest.TestCase):
    def test_two_amounts_do_not_collide(self):
        self.assertNotEqual(request_key("PDV 17% na neto 2400"),
                            request_key("PDV 17% na neto 4800"))
        # no numeric canonicalisation in the key: a comma is not a thousands dot
        self.assertNotEqual(request_key("neto 1,234"), request_key("neto 1234"))

    def test_one_character_never_collides(self):
        base = "koliko ima kupaca"
        seen = {request_key(base[:i] + c + base[i:]) for i in range(len(base))
                for c in "abč "}
        self.assertGreater(len(seen), 1)
        self.assertEqual(request_key(base), request_key(base))

    def test_delimiter_injection_does_not_collide(self):
        """Raw `sid|text` concatenation had this collision; JSON framing does not."""
        self.assertNotEqual(request_key("c", tenant="a|b", session="c"),
                            request_key("c", tenant="a", session="b|c"))
        self.assertNotEqual(request_key("b|c", session="a"),
                            request_key("c", session="a|b"))
        self.assertNotEqual(request_key("x", context={"env": "a|b"}),
                            request_key("x", context={"env": "a", "extra": "b"}))

    def test_context_is_part_of_the_key(self):
        self.assertNotEqual(request_key("x", context={"env": "A"}),
                            request_key("x", context={"env": "B"}))
        self.assertNotEqual(request_key("x", context={"domain": "doc"}),
                            request_key("x", context={"domain": "price"}))
        self.assertNotEqual(request_key("x", context=None),
                            request_key("x", context={}))


class TestScoping(unittest.TestCase):
    def test_tenants_are_isolated(self):
        self.assertNotEqual(request_key("zdravo", tenant="A"),
                            request_key("zdravo", tenant="B"))
        self.assertNotEqual(request_key("zdravo", tenant="A"),
                            request_key("zdravo"))

    def test_sessions_are_isolated(self):
        self.assertNotEqual(request_key("zdravo", session="s1"),
                            request_key("zdravo", session="s2"))
        self.assertNotEqual(request_key("zdravo", session="s1"),
                            request_key("zdravo", session=""))

    def test_tenant_session_swap_is_not_the_same_request(self):
        self.assertNotEqual(request_key("zdravo", tenant="A", session="B"),
                            request_key("zdravo", tenant="B", session="A"))

    def test_same_request_twice_same_key(self):
        kw = dict(tenant="t", session="s", context={"env": "E", "domain": "doc"})
        self.assertEqual(request_key("kupi 2 mixera", **kw),
                         request_key("kupi 2 mixera", **kw))


class TestNormalisation(unittest.TestCase):
    def test_case_and_whitespace_fold(self):
        self.assertEqual(request_key("  PDV   17%  "), request_key("pdv 17%"))

    def test_diacritics_fold(self):
        self.assertEqual(normalize_for_key("Obriši sve"), "obrisi sve")
        self.assertEqual(request_key("Obriši sve"), request_key("obrisi sve"))

    def test_explicit_normalized_wins(self):
        a = request_key("PDV 17% na neto 2.400", normalized="pdv #2400")
        b = request_key("pdv 17% na neto 2400", normalized="pdv #2400")
        self.assertEqual(a, b)
        self.assertNotEqual(a, request_key("pdv 17% na neto 2400"))

    def test_unicode_is_stable(self):
        k = request_key("kupi 2 m³ betona — hitno")
        self.assertTrue(is_current(k))
        self.assertEqual(k, request_key("kupi 2 m³ betona — hitno"))


if __name__ == "__main__":
    unittest.main()
