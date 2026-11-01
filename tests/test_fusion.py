import unittest

from dsc.fusion import (
    calibrate_threshold, conformal_quantile, decide, ece, mcnemar, sigmoid, wrrf,
)


class TestFusion(unittest.TestCase):
    def test_wrrf(self):
        r = wrrf([["a", "b"], ["b", "a"]], [1, 1], k=10)
        self.assertAlmostEqual(r["a"], r["b"])

    def test_conformal(self):
        self.assertEqual(conformal_quantile([0.1, 0.2, 0.3, 0.4], 0.25), 0.4)

    def test_threshold(self):
        r = calibrate_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], 1, 10)
        self.assertEqual(r["threshold"], 0.8)
        self.assertEqual(r["fpr"], 0.0)

    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertGreater(sigmoid(2), 0.8)

    def test_ece(self):
        self.assertAlmostEqual(ece([0.0, 1.0], [0, 1]), 0.0)

    def test_mcnemar(self):
        r = mcnemar([True, False, True], [True, True, False])
        self.assertEqual((r["b01"], r["b10"]), (1, 1))
        self.assertEqual(r["p"], 1.0)


if __name__ == "__main__":
    unittest.main()
