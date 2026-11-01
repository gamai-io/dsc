import unittest

from dsc.golden_expand import dediac, generate, typo


class TestGoldenExpand(unittest.TestCase):
    def test_dediac_typo(self):
        self.assertEqual(dediac("čćžšđ"), "cczsd")
        self.assertNotEqual(typo("kompresor"), "kompresor")

    def test_generate(self):
        data = {
            "docs": [{"id": "d1", "cat": "airless", "text": "Graco Ultra 795 je airless aparat za farbanje."}],
            "queries": [{"q": "čišćenje filtera", "gold": ["d1"], "cat": "exact"}],
        }
        out = generate(data)
        self.assertTrue(out["synthetic"])
        self.assertEqual(len(out["docs"]), 1)
        qs = out["queries"]
        self.assertTrue(any(q.get("origin") for q in qs))
        self.assertTrue(any(q["cat"] == "diacritic" for q in qs))
        self.assertTrue(any(q["cat"] == "category" for q in qs))
        self.assertTrue(all(q.get("graded") for q in qs))


if __name__ == "__main__":
    unittest.main()
