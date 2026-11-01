import unittest
from dsc.normalize import (
    canonical, extract_numbers, features, fold, parse_number, to_nfc,
)


class TestNormalize(unittest.TestCase):
    def test_fold(self):
        self.assertEqual(fold("ČĆŽŠĐčćžšđ"), "cczsdcczsd")

    def test_nfc_nfd(self):
        a = "čćžšđ"
        b = "c\u030cc\u0301z\u030cs\u030cđ"
        self.assertEqual(to_nfc(b), a)
        self.assertEqual(canonical(a), canonical(b))

    def test_numbers_bs(self):
        from decimal import Decimal
        self.assertEqual(parse_number("2.400,00"), Decimal("2400.00"))
        self.assertEqual(parse_number("1450,50"), Decimal("1450.5"))
        self.assertEqual(parse_number("1.234"), Decimal("1234"))
        self.assertEqual(parse_number("0.95"), Decimal("0.95"))

    def test_units(self):
        n = extract_numbers("pritisak 3300 PSI, PDV 17%, cijena 1450 KM, 10 bar")
        self.assertEqual(n[0]["dim"], "pressure")
        self.assertEqual(n[1]["dim"], "percent")
        self.assertEqual(n[2]["dim"], "currency")
        self.assertEqual(n[3]["dim"], "pressure")

    def test_canonical_equivalent(self):
        self.assertEqual(canonical("cijena 2.400,00 KM"), canonical("cijena 2400 KM"))

    def test_negation_flags(self):
        f = features("nemoj poslati mail danas")
        self.assertTrue(f["action"])
        self.assertTrue(f["fresh"])
        self.assertGreater(f["negations"], 0)

    def test_dates(self):
        f = features("rok je 31.12.2025")
        self.assertIn("2025-12-31", f["dates"])


if __name__ == "__main__":
    unittest.main()
