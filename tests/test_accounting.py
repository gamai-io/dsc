import unittest
from decimal import Decimal

from dsc.accounting import (
    fifo, fx, pdv_exclusive, pdv_inclusive, verify_arithmetic, verify_fx_claim, verify_pdv_claim,
)
from dsc.lexical import hard_check
from dsc.verify import verify_answer


class TestAccounting(unittest.TestCase):
    def test_pdv(self):
        e = pdv_exclusive(2400)
        self.assertEqual((e["base"], e["vat"], e["total"]), (Decimal("2400.00"), Decimal("408.00"), Decimal("2808.00")))
        i = pdv_inclusive(2808)
        self.assertEqual((i["base"], i["vat"]), (Decimal("2400.00"), Decimal("408.00")))

    def test_fifo(self):
        r = fifo([(10, "10"), (10, "20")], 15)
        self.assertEqual(r["cogs"], Decimal("200.00"))
        self.assertEqual(r["remaining_value"], Decimal("100.00"))
        with self.assertRaises(ValueError):
            fifo([(1, "10")], 2)

    def test_fx(self):
        self.assertEqual(fx(400, "1.95583"), Decimal("782.33"))
        self.assertTrue(verify_fx_claim(400, "1.95583", "782.33")["ok"])
        self.assertFalse(verify_fx_claim(400, "1.95000", "782.33")["ok"])

    def test_arithmetic(self):
        self.assertTrue(verify_arithmetic("17% od 2400 = 408")["ok"])
        self.assertFalse(verify_arithmetic("17% od 2400 = 480")["ok"])
        self.assertTrue(verify_arithmetic("400 * 1.95583 = 782.33")["ok"])
        self.assertTrue(verify_arithmetic("2400 + 408 = 2808")["ok"])
        self.assertTrue(verify_arithmetic("2400 / 1.17 = 2051.28")["ok"])

    def test_hard_check_arithmetic(self):
        r = hard_check("PDV 17% od 2400 = 480, ukupno 2880", "PDV 17%, osnovica 2400, ukupno 2808")
        self.assertFalse(r["ok"])
        self.assertFalse(r["arithmetic_ok"])
        self.assertTrue(any("arithmetic" in x for x in r["reasons"]))

    def test_pdv_claim(self):
        r = verify_pdv_claim("Osnovica 2400, PDV 17% = 408, ukupno 2808")
        self.assertTrue(r["checked"])
        self.assertTrue(r["ok"])

    def test_accounting_veto(self):
        bad = "Osnovica 2400, PDV 480, ukupno 2880 [1]."
        src = [{"id": 1, "text": "Osnovica 2400, PDV 480, ukupno 2880"}]
        r = verify_answer(bad, src, accounting=True)
        self.assertEqual(r["claims"][0]["reason"], "accounting:pdv")
        generic = verify_answer(bad, src)
        self.assertNotEqual(generic["claims"][0]["reason"], "accounting:pdv")
        good = "Osnovica 2400, PDV 408, ukupno 2808 [1]."
        src2 = [{"id": 1, "text": "Osnovica 2400, PDV 408, ukupno 2808"}]
        self.assertTrue(verify_answer(good, src2, accounting=True)["claims"][0]["supported"])


if __name__ == "__main__":
    unittest.main()
