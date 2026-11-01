import unittest

from dsc.lexical import BM25Index, hard_check, lexical_entailment


class TestLexical(unittest.TestCase):
    def test_bm25(self):
        b = BM25Index()
        b.add(1, "kupac ACME Hadzici Sarajevo")
        b.add(2, "kompresor 10 bar")
        b.finalize()
        r = b.search("kupac Sarajevo")
        self.assertEqual(r[0]["id"], 1)

    def test_numbers(self):
        self.assertTrue(hard_check("cijena 1450 KM", "cijena 1.450,00 KM")["ok"])
        self.assertFalse(hard_check("cijena 1500 KM", "cijena 1.450,00 KM")["ok"])

    def test_units(self):
        self.assertTrue(hard_check("3300 PSI", "227 bar")["ok"])
        self.assertFalse(hard_check("3000 PSI", "200 bar")["ok"])

    def test_percent(self):
        self.assertTrue(hard_check("PDV 17%", "stopa PDV je 17 posto")["ok"])

    def test_polarity(self):
        r = hard_check("pumpa radi", "pumpa ne radi")
        self.assertFalse(r["ok"])
        self.assertIn("polarity", r["reasons"])

    def test_date(self):
        self.assertTrue(hard_check("rok 31.12.2025", "rok je 2025-12-31")["ok"])
        self.assertFalse(hard_check("rok 31.12.2025", "rok je 2026-01-01")["ok"])

    def test_entailment(self):
        hi = lexical_entailment("kupac ACME je u Sarajevu", "ACME Hadzici, Sarajevo")
        lo = lexical_entailment("kupac ACME je u Sarajevu", "kompresor 10 bar")
        self.assertGreater(hi["score"], lo["score"])


if __name__ == "__main__":
    unittest.main()
