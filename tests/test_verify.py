import unittest

from dsc.verify import ConformalGate, split_claims, verify_answer, verify_claim

SRC = [
    {"id": 1, "text": "Graco Ultra Max II 795: maksimalni pritisak 3300 PSI, protok 0.95 GPM."},
    {"id": 2, "text": "Cijena masine je 1450 KM."},
]


class TestVerify(unittest.TestCase):
    def test_supported(self):
        r = verify_answer("Graco Ultra 795 ima pritisak 3300 PSI [1].", SRC)
        self.assertEqual(r["verdict"], "SUPPORTED")
        self.assertEqual(r["claims"][0]["reason"], "calibrated-and")

    def test_wrong_number(self):
        r = verify_answer("Graco Ultra 795 ima pritisak 3000 PSI [1].", SRC)
        self.assertEqual(r["verdict"], "UNSUPPORTED")
        self.assertTrue(r["claims"][0]["reason"].startswith("hard:number"))

    def test_negation_veto(self):
        r = verify_answer("Pumpa ne radi [1].", SRC)
        self.assertFalse(r["claims"][0]["supported"])

    def test_citation_missing(self):
        r = verify_answer("Cijena je 1450 KM.", SRC)
        self.assertEqual(r["verdict"], "UNVERIFIED")

    def test_wrong_citation(self):
        r = verify_answer("Cijena je 1450 KM [1].", SRC)
        self.assertFalse(r["claims"][0]["supported"])
        self.assertTrue(r["claims"][0]["reason"].startswith("hard:number"))

    def test_no_claims(self):
        self.assertEqual(split_claims(""), [])
        self.assertEqual(verify_answer("", SRC)["verdict"], "UNVERIFIED")

    def test_conformal(self):
        g = ConformalGate(0.1).fit([0.1, 0.2, 0.3])
        self.assertFalse(g.accept(0.05))
        self.assertTrue(g.accept(0.35))


if __name__ == "__main__":
    unittest.main()
