import os
import tempfile
import unittest

from dsc.cache import SemanticCache


class TestCache(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(self.path)
        self.c = SemanticCache(self.path, max_entries=50)

    def tearDown(self):
        self.c.close()
        for suf in ("", "-wal", "-shm"):
            try:
                os.unlink(self.path + suf)
            except OSError:
                pass

    def test_exact_and_paraphrase(self):
        self.assertTrue(self.c.put("koliko ima kupaca u bazi", "1114 kupaca", sid="s1", domain="doc"))
        self.assertEqual(self.c.get("koliko ima kupaca u bazi", sid="s1", domain="doc")["cache"], "exact")
        self.assertEqual(self.c.get("koliko kupaca ima u bazi", sid="s1", domain="doc")["cache"], "semantic")

    def test_sid_isolation(self):
        self.c.put("koliko ima kupaca", "1114 kupaca", sid="s1", domain="doc")
        self.assertIsNone(self.c.get("koliko ima kupaca", sid="s2", domain="doc"))

    def test_number_mismatch(self):
        self.c.put("promet 2025", "promet je 1000 KM", sid="s1", domain="doc")
        self.assertIsNone(self.c.get("promet 2026", sid="s1", domain="doc"))

    def test_price_exact_only(self):
        self.c.put("cijena 1450 KM", "cijena je 1450 KM", sid="s1", domain="price", ttl=30)
        self.assertEqual(self.c.get("cijena 1450 km", sid="s1", domain="price")["cache"], "exact")
        self.assertIsNone(self.c.get("cijena 1450 KM danas", sid="s1", domain="price"))

    def test_epoch(self):
        self.c.put("koliko ima kupaca", "1114 kupaca", sid="s1", domain="doc")
        self.assertIsNotNone(self.c.get("koliko ima kupaca", sid="s1", domain="doc"))
        self.c.set_epoch("doc")
        self.assertIsNone(self.c.get("koliko ima kupaca", sid="s1", domain="doc"))

    def test_ttl(self):
        self.c.put("koliko ima kupaca", "1114 kupaca", sid="s1", domain="doc", ttl=10, now=1000)
        self.assertIsNotNone(self.c.get("koliko ima kupaca", sid="s1", domain="doc", now=1005))
        self.assertIsNone(self.c.get("koliko ima kupaca", sid="s1", domain="doc", now=1011))

    def test_risky_not_stored(self):
        self.assertFalse(self.c.put("otvori notepad i ukucaj tekst", "ok", sid="s1"))
        self.assertFalse(self.c.put("koliko ima kupaca danas", "1114", sid="s1"))

    def test_envelope_isolation(self):
        self.c.put("koliko ima kupaca u bazi", "1114 kupaca", sid="s1", domain="doc", envelope_id="A")
        self.assertEqual(self.c.get("koliko ima kupaca u bazi", sid="s1", domain="doc",
                                    envelope_id="A")["cache"], "exact")
        self.assertIsNone(self.c.get("koliko ima kupaca u bazi", sid="s1", domain="doc",
                                     envelope_id="B"))
        self.assertIsNone(self.c.get("koliko ima kupaca u bazi", sid="s1", domain="doc"))

    def test_stochastic_exact_only(self):
        self.c.put("koliko ima kupaca u bazi", "1114 kupaca", sid="s1", domain="doc",
                   envelope_id="A", inference_level="STOCHASTIC")
        self.assertEqual(self.c.get("koliko ima kupaca u bazi", sid="s1", domain="doc",
                                    envelope_id="A")["cache"], "exact")
        self.assertIsNone(self.c.get("koliko kupaca ima u bazi", sid="s1", domain="doc",
                                     envelope_id="A"))

    def test_unverified_not_semantic(self):
        self.c.put("koliko ima kupaca u bazi", "1114 kupaca", sid="s1", domain="doc", trust="unverified")
        self.assertIsNone(self.c.get("koliko kupaca ima u bazi", sid="s1", domain="doc"))


if __name__ == "__main__":
    unittest.main()
