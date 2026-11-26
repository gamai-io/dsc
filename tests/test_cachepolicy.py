"""Tests for the shared cacheability policy (``dsc.cachepolicy``).

The point of the module is that callers keep their storage and counters but not
their own rule lists, so these tests pin *behaviour* (which queries may be
replayed) and the *shape* of the rule table (data, not prose).
"""
from __future__ import annotations

import unittest

from dsc.cachepolicy import (
    EFFECTS, ERROR_MARKERS, FRESH_RULES, MIN_ANSWER_LEN, MULTI_STEP_RULES,
    MUTATING_TOOLS, RULES, block_reason, cacheable, fresh, fresh_rule,
    multi_step, mutating_tools, text_block_reason,
)

ANSWER = "PDV 17% na neto 2400 je 2808,00 KM (2400 + 408)."


class TestRuleTable(unittest.TestCase):
    def test_effects_are_a_closed_vocabulary(self):
        for rule in RULES:
            self.assertIn(rule.effect, EFFECTS, rule.name)

    def test_every_rule_is_named_and_justified(self):
        names = [r.name for r in RULES]
        self.assertEqual(len(names), len(set(names)), "duplicate rule name")
        for rule in RULES:
            self.assertTrue(rule.name)
            self.assertGreater(len(rule.why), 20, rule.name)
            self.assertTrue(rule.pattern is not None or rule.tools, rule.name)

    def test_mutating_tools_are_data(self):
        self.assertIsInstance(MUTATING_TOOLS, frozenset)
        for tool in ("mail", "bash", "write", "postis", "kasa"):
            self.assertIn(tool, MUTATING_TOOLS)
        for tool in ("sql", "read", "search", "skill"):
            self.assertNotIn(tool, MUTATING_TOOLS)
        self.assertEqual(RULES[-1].tools, MUTATING_TOOLS)

    def test_rules_have_a_firing_example(self):
        """Every rule must be reachable from at least one phrase."""
        examples = {
            "NADZOR_FRESH_V1": "koliko kosta XHD211",
            "ERP_FRESH_V2": "koliko imamo na stanju",
            "CLOCK_FRESH_V1": "koliko je sati",
            "CACHE_MULTI_GUARD": "posalji mail kupcu i otvori ponudu",
        }
        for rule in FRESH_RULES + MULTI_STEP_RULES:
            self.assertTrue(rule.matches(examples[rule.name]), rule.name)
        res = {"content": ANSWER, "tool_log": [{"tool": "mail"}]}
        self.assertEqual(block_reason(res, "posalji ponudu"), "mutating")


class TestFreshness(unittest.TestCase):
    def test_pdv_amount_query_is_cacheable(self):
        self.assertFalse(fresh("PDV 17% na neto 2400"))
        self.assertIsNone(text_block_reason("PDV 17% na neto 2400"))
        self.assertTrue(cacheable({"content": ANSWER}, "PDV 17% na neto 2400"))

    def test_clock_is_never_cacheable(self):
        for q in ("koliko je sati", "Koliko je sati?", "koje je vrijeme",
                  "koji je datum"):
            self.assertTrue(fresh(q), q)
            self.assertEqual(text_block_reason(q), "fresh", q)
            self.assertFalse(cacheable({"content": "Sada je 14:05, 26.11.2026."}, q), q)

    def test_stock_price_and_balance_are_never_cacheable(self):
        for q in ("stanje zaliha za 100011", "lager 100011",
                  "koliko imamo na stanju", "saldo kupca 211008",
                  "dugovanja kupaca", "zadnja cijena XHD211",
                  "trenutna cijena mixera", "kolicina na skladistu",
                  "koliko kosta XHD211", "kurs EUR danas"):
            self.assertTrue(fresh(q), q)
            self.assertFalse(cacheable({"content": ANSWER}, q), q)

    def test_explanations_stay_cacheable(self):
        for q in ("kako se kontira ulaz robe", "objasni sta je false success",
                  "koliko ima kupaca", "zdravo"):
            self.assertFalse(fresh(q), q)
            self.assertTrue(cacheable({"content": ANSWER}, q), q)

    def test_rule_name_is_reported(self):
        self.assertEqual(fresh_rule("lager 100011").name, "ERP_FRESH_V2")
        self.assertEqual(fresh_rule("koliko je sati").name, "CLOCK_FRESH_V1")
        self.assertIsNone(fresh_rule("zdravo"))


class TestResponseGuards(unittest.TestCase):
    def test_short_answer_is_not_cached(self):
        self.assertEqual(block_reason({"content": "x" * (MIN_ANSWER_LEN - 1)}, "zdravo"),
                         "empty")
        self.assertEqual(block_reason({"content": ""}, "zdravo"), "empty")
        self.assertEqual(block_reason({"say": "ok"}, "zdravo"), "empty")
        self.assertEqual(block_reason(None, "zdravo"), "empty")

    def test_stopped_and_error_are_not_cached(self):
        self.assertEqual(block_reason({"content": ANSWER, "stopped": True}, "zdravo"),
                         "stopped")
        for c in ("(greska: timeout pri pozivu alata", "ERROR: no backend",
                  "prekinuto nakon 3 koraka"):
            self.assertEqual(block_reason({"content": c}, "zdravo"), "error", c)

    def test_mutating_tool_is_not_cached(self):
        res = {"content": ANSWER, "tool_log": [{"tool": "Mail"}, {"tool": "sql"}]}
        self.assertEqual(mutating_tools(res), ("mail",))
        self.assertEqual(block_reason(res, "posalji ponudu kupcu"), "mutating")

    def test_read_only_tool_is_cacheable(self):
        res = {"content": ANSWER, "tool_log": [{"tool": "sql"}, {"tool": "read"}]}
        self.assertEqual(mutating_tools(res), ())
        self.assertTrue(cacheable(res, "koliko ima kupaca"))

    def test_freshness_reported_before_side_effects(self):
        res = {"content": ANSWER, "tool_log": [{"tool": "mail"}]}
        self.assertEqual(block_reason(res, "saldo kupca 211008"), "fresh")

    def test_multi_step_is_not_cached(self):
        q = "posalji mail kupcu i otvori ponudu"
        self.assertTrue(multi_step(q))
        self.assertEqual(block_reason({"content": ANSWER}, q), "multi_step")
        self.assertEqual(block_reason({"content": ANSWER}, "posalji mail kupcu"),
                         None)

    def test_error_markers_are_data(self):
        self.assertIn("(greska", ERROR_MARKERS)


if __name__ == "__main__":
    unittest.main()
