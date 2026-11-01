import unittest

from dsc.lab_eval import evaluate, paired_bootstrap


class TestLabEval(unittest.TestCase):
    def test_evaluate(self):
        qs = [{"q": "a", "gold": ["d1"], "graded": {"d1": 2}},
              {"q": "b", "gold": ["d2"], "graded": {"d2": 2}}]
        good = {"a": ["d1", "d2"], "b": ["d2", "d1"]}
        bad = {"a": ["d2", "d1"], "b": ["d1", "d2"]}
        m = evaluate(good, qs)
        self.assertEqual(m["R@1"], 1.0)
        self.assertEqual(m["nDCG@10"], 1.0)
        self.assertLess(evaluate(bad, qs)["R@1"], 1.0)

    def test_bootstrap(self):
        qs = [{"q": str(i), "gold": ["g"], "graded": {"g": 2}} for i in range(10)]
        a = {str(i): ["g"] for i in range(10)}
        b = {str(i): ["x"] for i in range(10)}
        r = paired_bootstrap(a, b, qs, "R@1", n=200)
        self.assertGreater(r["diff"], 0)
        self.assertLessEqual(r["p"], 0.05)


if __name__ == "__main__":
    unittest.main()
