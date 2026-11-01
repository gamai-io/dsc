import unittest

from dsc.cache_calib import _semantic_ok, calibrate


class TestCacheCalib(unittest.TestCase):
    def test_semantic_ok(self):
        self.assertTrue(_semantic_ok("kako izabrati airless pumpu"))
        self.assertFalse(_semantic_ok("otvori notepad"))
        self.assertFalse(_semantic_ok("cijena 1450 KM"))

    def test_calibrate(self):
        data = {
            "docs": [{"id": "d1", "text": "Graco Ultra 795 airless aparat za farbanje profesionalaca"}],
            "queries": [
                {"q": "kako izabrati graco ultra 795", "gold": ["d1"], "cat": "exact", "origin": True},
                {"q": "kako izabrati graco ultra 795", "gold": ["d1"], "cat": "diacritic", "base": "kako izabrati graco ultra 795"},
            ],
        }
        r = calibrate(data, [0.5], )
        self.assertGreaterEqual(r["stored"], 1)
        self.assertGreaterEqual(r["probes"], 1)
        self.assertTrue(r["rows"])


if __name__ == "__main__":
    unittest.main()
