"""Tests for health negotiation (R6). No network: probe_fn is injected."""
from __future__ import annotations

import os
import unittest

from dsc import health as H


def _fake(results: dict[str, tuple[bool, str]]):
    def fn(target: str, kind: str, timeout: float):
        return results.get(target, (False, "missing"))
    return fn


class TestHealth(unittest.TestCase):
    def setUp(self):
        self.probes = (("a", "t-a", "http"), ("b", "t-b", "tcp"))
        self.core = ("core", "t-core", "http")

    def test_all_ok(self):
        fn = _fake({"t-a": (True, "ok"), "t-b": (True, "ok"), "t-core": (True, '{"ok":true}')})
        res = H.health(self.probes, self.core, probe_fn=fn)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["exit_code"], H.EXIT_OK)
        self.assertFalse(res["degraded"])
        self.assertEqual(res["down"], [])

    def test_infra_down(self):
        fn = _fake({"t-a": (True, "ok"), "t-b": (False, "refused"), "t-core": (True, "ok")})
        res = H.health(self.probes, self.core, probe_fn=fn)
        self.assertEqual(res["status"], "down")
        self.assertEqual(res["exit_code"], H.EXIT_DOWN)
        self.assertEqual(res["down"], ["b"])

    def test_core_offline_is_degraded_not_failure(self):
        fn = _fake({"t-a": (True, "ok"), "t-b": (True, "ok"), "t-core": (False, "timeout")})
        res = H.health(self.probes, self.core, probe_fn=fn)
        self.assertEqual(res["status"], "degraded")
        self.assertEqual(res["exit_code"], H.EXIT_DEGRADED)
        self.assertTrue(res["degraded"])
        self.assertFalse(res["core"]["ok"])

    def test_timeout_arg_reaches_probe(self):
        seen = []

        def fn(target, kind, timeout):
            seen.append((target, timeout))
            return True, "ok"

        H.health((("a", "t-a", "tcp"),), ("core", "t-core", "http"), timeout=1.5, probe_fn=fn)
        self.assertIn(("t-a", 1.5), seen)
        core_timeout = [t for target, t in seen if target == "t-core"][0]
        self.assertGreaterEqual(core_timeout, 12.0)

    def test_env_driven_defaults(self):
        old = {k: os.environ.get(k) for k in ("DSC_WORKER_URL", "DSC_HC_TARGET")}
        try:
            os.environ["DSC_WORKER_URL"] = "http://127.0.0.1:8790/"
            os.environ["DSC_HC_TARGET"] = "127.0.0.1:7100"
            probes = dict((n, t) for n, t, _ in H.default_probes())
            self.assertEqual(probes["worker"], "http://127.0.0.1:8790/health")
            self.assertEqual(probes["hc"], "127.0.0.1:7100")
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_describe_shape(self):
        d = H.describe()
        self.assertIn("probes", d)
        self.assertIn("exit_codes", d)
        self.assertEqual(d["exit_codes"]["degraded"], 75)


if __name__ == "__main__":
    unittest.main()
