import unittest

from dsc.hdc import cosine_q, dense, quantize, vector


class TestHDC(unittest.TestCase):
    def test_same_bytes(self):
        self.assertEqual(vector("kupac ACME"), vector("kupac ACME"))

    def test_paraphrase(self):
        a = vector("koliko ima kupaca u bazi")
        b = vector("koliko kupaca ima u bazi")
        c = vector("kompresor 10 bar")
        self.assertGreater(cosine_q(a, b), 0.70)
        self.assertLess(cosine_q(a, c), 0.20)

    def test_quantize_bounds(self):
        v = dense("test")
        q = quantize(v)
        self.assertEqual(len(q), len(v))
        self.assertTrue(all(0 <= x <= 255 for x in q))


if __name__ == "__main__":
    unittest.main()
