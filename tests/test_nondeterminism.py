import unittest

from dsc.nondeterminism import (
    Envelope, compare, guard, local_vllm_env, receipt, remote_api_env, required_ok,
    verify_receipt,
)


class TestNondeterminism(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(local_vllm_env("q").level(), "EXACT")
        e = local_vllm_env("q", batch_invariant=True)
        e = Envelope(**{**e.to_dict(), "deterministic_kernels": False})
        self.assertEqual(e.level(), "BATCH_INVARIANT")
        self.assertEqual(remote_api_env("cloud", "m").level(), "VERSIONED")
        s = Envelope(engine="vllm", model="m", temperature=0.7)
        self.assertEqual(s.level(), "STOCHASTIC")

    def test_identity_ignores_batch(self):
        a = local_vllm_env("q", batch_size=1, batch_signature="a")
        b = local_vllm_env("q", batch_size=64, batch_signature="b")
        self.assertEqual(a.identity(), b.identity())
        self.assertNotEqual(a.full_id(), b.full_id())

    def test_replay_key(self):
        a = remote_api_env("cloud", "m", batch_size=4, batch_signature="a,b")
        b = remote_api_env("cloud", "m", batch_size=8, batch_signature="c,d")
        self.assertNotEqual(a.replay_key("p"), b.replay_key("p"))
        self.assertTrue(a.replay_bound())
        c = remote_api_env("cloud", "m")
        self.assertFalse(c.replay_bound())
        bi = local_vllm_env("q", batch_size=1)
        self.assertEqual(bi.replay_key("p", strict_batch=False),
                         bi.replay_key("p", strict_batch=False))

    def test_receipt(self):
        e = local_vllm_env("q")
        r = receipt("pitanje", "odgovor", e)
        self.assertTrue(verify_receipt(r, "pitanje", "odgovor")["ok"])
        self.assertIn("output", verify_receipt(r, "pitanje", "drugi")["bad"])

    def test_guard(self):
        self.assertTrue(guard(local_vllm_env("q"), "EXACT")["ok"])
        g = guard(remote_api_env("cloud", "m"), "EXACT")
        self.assertFalse(g["ok"])
        self.assertTrue(guard(remote_api_env("cloud", "m"), "VERSIONED")["ok"])
        self.assertTrue(required_ok("BATCH_INVARIANT", "VERSIONED"))
        self.assertFalse(required_ok("VERSIONED", "EXACT"))

    def test_compare(self):
        e = local_vllm_env("q")
        a = receipt("p", "o", e)
        b = receipt("p", "o", e)
        c = receipt("p", "x", e)
        self.assertTrue(compare(a, b)["same_output"])
        self.assertFalse(compare(a, c)["same_output"])


if __name__ == "__main__":
    unittest.main()
