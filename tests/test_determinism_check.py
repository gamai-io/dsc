import itertools
import threading
import unittest

from dsc.determinism_check import first_divergence, run_probe


class TestDeterminismCheck(unittest.TestCase):
    def test_divergence(self):
        self.assertIsNone(first_divergence("abc", "abc"))
        self.assertEqual(first_divergence("abc", "abd"), 2)
        self.assertEqual(first_divergence("ab", "abcd"), 2)

    def test_stable(self):
        r = run_probe(lambda p: "same output", "p", batch_sizes=(1, 4), repeat=2)
        self.assertTrue(r["cross_batch_equal"])
        self.assertEqual(r["unique_total"], 1)

    def test_unstable(self):
        c = itertools.count()
        lock = threading.Lock()

        def fn(_p):
            with lock:
                return "out-%d" % next(c)
        r = run_probe(fn, "p", batch_sizes=(1, 4), repeat=1)
        self.assertFalse(r["cross_batch_equal"])
        self.assertGreater(r["unique_total"], 1)


if __name__ == "__main__":
    unittest.main()
