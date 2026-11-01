import os
import tempfile
import unittest

from dsc.api import Core


class TestAPI(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.path)
        self.core = Core(cache_path=self.path)

    def tearDown(self):
        self.core.close()
        for suf in ("", "-wal", "-shm"):
            try:
                os.unlink(self.path + suf)
            except OSError:
                pass

    def test_analyze(self):
        a = self.core.analyze("PDV je 17% u BiH")
        self.assertEqual(a["canonical"], "pdv je 17% u bih")
        self.assertIn("simhash64", a)
        self.assertEqual(len(a["minhash128"]), 128)

    def test_search(self):
        docs = ["Graco Ultra 795 airless farbanje", "Kompresor 10 bar 500 L", "ACME Hadzici Sarajevo"]
        r = self.core.search("graco farbanje", docs, k=2)
        self.assertTrue(r)
        self.assertEqual(r[0]["id"], "0")
        if len(r) > 1:
            self.assertGreater(r[0]["fused"], r[1]["fused"])

    def test_verify(self):
        r = self.core.verify("Graco Ultra 795 ima 3300 PSI [1].",
                             [{"id": 1, "text": "Graco Ultra 795: 3300 PSI"}])
        self.assertEqual(r["verdict"], "SUPPORTED")

    def test_cache(self):
        self.core.remember("koliko ima kupaca", "1114 kupaca", sid="s", domain="doc")
        self.assertIsNotNone(self.core.recall("koliko kupaca ima", sid="s", domain="doc"))

    def test_attest(self):
        from dsc.nondeterminism import local_vllm_env
        env = local_vllm_env("qwen")
        rec = self.core.attest("p", "o", env)
        self.assertEqual(rec["level"], "EXACT")
        self.assertTrue(self.core.inference_guard(env, "EXACT")["ok"])
        self.assertEqual(self.core.determinism()["levels"][0], "EXACT")


if __name__ == "__main__":
    unittest.main()
