"""Regression tests for the single money-amount parser.

Carried over from ``gama-fastpath/tests/test_fastpath.py::test_parse_amount``
plus the cases the extraction had to pin: currency stripping, the documented
disagreement with ``normalize.parse_number``, and Decimal-only output.
"""
import unittest
from decimal import Decimal

from dsc import normalize
from dsc.accounting import pdv_exclusive
from dsc.amount import parse_amount


class AmountTest(unittest.TestCase):
    def test_bhs_eu_us_formats(self):
        cases = {
            "2.400,00": "2400.00",
            "1.234.567,89": "1234567.89",
            "2,400.00": "2400.00",
            "2400": "2400",
            "1.450,00": "1450.00",
            "2.400": "2400",
            "-17,5": "-17.5",
        }
        for raw, want in cases.items():
            self.assertEqual(parse_amount(raw), Decimal(want), raw)

    def test_currency_and_letters_are_stripped(self):
        for raw in ("KM 2.400,00", "2.400,00 KM", "EUR 2.400,00", "\u20ac 2.400,00",
                    " 2.400,00 ", "2\u00a0400,00"):
            self.assertEqual(parse_amount(raw), Decimal("2400.00"), raw)

    def test_no_amount_raises_value_error(self):
        for raw in ("", "   ", None, "abc", "KM", ",,", "."):
            with self.assertRaises(ValueError, msg=repr(raw)):
                parse_amount(raw)

    def test_returns_decimal_never_float(self):
        self.assertIsInstance(parse_amount("2400"), Decimal)
        self.assertEqual(parse_amount("0,1") + parse_amount("0,2"), Decimal("0.3"))

    def test_feeds_pdv_without_rounding_drift(self):
        total = pdv_exclusive(parse_amount("2.400,00"))["total"]
        self.assertEqual(total, Decimal("2808.00"))

    def test_documented_divergence_from_parse_number(self):
        # A lone comma is a decimal separator in a money field ...
        self.assertEqual(parse_amount("1,234"), Decimal("1.234"))
        self.assertEqual(parse_amount("12,345"), Decimal("12.345"))
        # ... but a thousands group in running prose, which is parse_number's job.
        self.assertEqual(normalize.parse_number("1,234"), Decimal("1234"))
        self.assertEqual(normalize.parse_number("12,345"), Decimal("12345"))
        # Both agree once the format is unambiguous.
        for raw in ("2.400,00", "2,400.00", "1.234.567,89"):
            self.assertEqual(parse_amount(raw), normalize.parse_number(raw), raw)


if __name__ == "__main__":
    unittest.main()
