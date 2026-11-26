"""Cacheability policy — the single copy of "may this answer be replayed?".

Before this module the rule lived in ``gama_offline/cache.py`` as three
module-level objects (``_FRESH``, ``_MULTI``, ``_MUTATING``) plus a handful of
``if`` statements inside ``cacheable()``.  ``gama-fastpath`` had no copy at all,
which is how a stored ``"koje je vrijeme"`` answer could outrank its own clock
route.  Both layers now call this module.

The rules are **data**: an ordered table of named patterns, each carrying the
effect it has and why it exists.  Adding a phrase means adding a row, not
editing an ``if``.  Tests iterate the table, so a rule that is added without a
matching test is visible in review, and a rule that is deleted breaks a test.

Effects (the closed vocabulary in :data:`EFFECTS`):

``fresh``
    The answer depends on the moment it was produced (clock, search, price,
    stock, balance).  Never serve it from a cache; a miss costs one call, a
    stale hit costs a wrong number to a customer.
``mutating``
    The answer came from a tool that changed state (mail, shell, write, a
    machine).  Replaying it would claim an action happened when it did not.
``multi_step``
    The request is a chain of actions (``... i otvori ...``).  Two chains can
    normalise to similar text but need different execution.
``error`` / ``stopped``
    The answer is a failure or a cancellation; caching it would make the
    failure permanent for the TTL.
``empty``
    Too short to be a real answer (``MIN_ANSWER_LEN``).

The policy is text-in / reason-out: :func:`block_reason` returns a short code
(``"fresh"``, ``"mutating"`` …) or ``None``.  Callers keep their own counters
and their own storage; they do not keep their own patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "MIN_ANSWER_LEN", "EFFECTS", "Rule", "FRESH_RULES", "MULTI_STEP_RULES",
    "MUTATING_TOOLS", "ERROR_MARKERS", "RULES", "fresh", "fresh_rule",
    "multi_step", "multi_step_rule", "mutating_tools", "block_reason",
    "text_block_reason", "cacheable",
]

# An answer shorter than this is a placeholder, not content worth replaying.
MIN_ANSWER_LEN = 15

# Closed vocabulary.  A rule whose ``effect`` is outside this set is a bug and
# ``test_cachepolicy`` fails on it.
EFFECTS = frozenset({"fresh", "mutating", "multi_step", "error", "stopped", "empty"})


@dataclass(frozen=True)
class Rule:
    """One named policy rule.

    ``pattern`` is a compiled regex for text rules; ``tools`` is a frozenset
    for tool rules.  ``why`` is the cost of getting it wrong, not a restatement
    of the regex.
    """

    name: str
    effect: str
    why: str
    pattern: re.Pattern | None = None
    tools: frozenset = field(default_factory=frozenset)

    def matches(self, text: str) -> bool:
        if self.pattern is None:
            return False
        return bool(self.pattern.search(text or ""))


def _rx(name: str, effect: str, why: str, pattern: str) -> Rule:
    return Rule(name, effect, why, re.compile(pattern, re.IGNORECASE | re.UNICODE))


# ---------------------------------------------------------------------------
# freshness — never serve a value that changes without the cache knowing
# ---------------------------------------------------------------------------
# NADZOR_FRESH_V1: the original live-data words, kept byte-for-byte as a regex
# so the widening below cannot silently drop one of them.
# ERP_FRESH_V2: stock/balance/price are rewritten by every receipt; a cached
# "na stanju" is a wrong number in front of a customer.
# CLOCK_FRESH_V1: a clock/datetime answer is fresh by definition.  Offline did
# not catch this and fastpath checked its cache before its own time route, so a
# stored "koje je vrijeme" could be served days later.
FRESH_RULES: tuple[Rule, ...] = (
    _rx("NADZOR_FRESH_V1", "fresh",
        "live search/price/state query: an answer is only true at request time",
        r"(pretrazi|search|cijena|cijenu|cijene|najnovij|najnove|online|"
        r"na internetu|na netu|koliko ko[s\u0161]ta|trenutno|danas|ju[c\u010d]e|"
        r"sada|sada[s\u0161]nj|kurs|prognoza|vremenska|stanje na tr[z\u017ei]|"
        r"berza|akcije|kripto|nadzor|prozor|screenshot|prebaci|ekran|flota)"),
    _rx("ERP_FRESH_V2", "fresh",
        "stock/balance/price is mutated by every receipt: a stale hit is a wrong number",
        r"(zalih|lager|na stanju|stanje robe|stanje artikla|saldo|dugovanj|"
        r"potra[z\u017e]ivanj|koli[c\u010d]in|kolicin|zadnji ulaz|zadnja cijena|"
        r"trenutna cijena)"),
    _rx("CLOCK_FRESH_V1", "fresh",
        "clock/datetime is fresh by definition; caching it serves yesterday's time",
        r"(koliko je sati|koliko je vremena|koje je vrijeme|koji je datum|"
        r"what time is it|current time|time now)"),
)

# CACHE_MULTI_GUARD: a chain of actions ("... i otvori ...", "... pa posalji
# ...") can only be replayed as a whole, and the chain is the request.
MULTI_STEP_RULES: tuple[Rule, ...] = (
    _rx("CACHE_MULTI_GUARD", "multi_step",
        "multi-step chain: replaying the answer does not replay the steps",
        r"(?:\s+i\s+|\s+pa\s+|\s+zatim\s+|\s+potom\s+|\s+onda\s+|;|\s+&\s+)"
        r".{0,80}?\b(otvori|pokreni|napi[s\u0161]i|po[s\u0161]alji|posalji|"
        r"ukucaj|klikni|napravi|kreiraj|obri[s\u0161]i|pusti|prebaci|dodaj|"
        r"unesi|instaliraj)\b"),
)

# STATE_CHANGING_TOOL: the response used a tool with side effects, so it is a
# report of an action, not a reusable answer.  Read-only tools (sql/read/search
# /skill/...) stay cacheable.
MUTATING_TOOLS: frozenset = frozenset({
    "mail", "email", "sms", "whatsapp", "write", "bash", "python", "masina",
    "racunar", "postis", "otvori", "pokreni", "obrisi", "obri\u0161i",
    "kopiraj", "premjesti", "instaliraj", "deploy", "github", "zakazi",
    "obavijesti", "nauci_skill", "self_mod", "tring", "kasa", "rdp",
})

MUTATING_RULE = Rule(
    "STATE_CHANGING_TOOL", "mutating",
    "the response reports a side effect; replay would claim it happened again",
    tools=MUTATING_TOOLS,
)

# Markers of a failed run that still produced text.  Lower-cased before search.
ERROR_MARKERS: tuple[str, ...] = ("(greska", "error:", "prekinuto nakon")

# Every rule, in the order the policy applies them.  ``RULES`` is the table the
# tests walk; it must not diverge from the sequences above.
RULES: tuple[Rule, ...] = FRESH_RULES + MULTI_STEP_RULES + (MUTATING_RULE,)


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------
def fresh_rule(text: str) -> Rule | None:
    """The first freshness rule that fires, or ``None``."""
    for rule in FRESH_RULES:
        if rule.matches(text):
            return rule
    return None


def fresh(text: str) -> bool:
    """True when the query asks for a value that must not be replayed."""
    return fresh_rule(text) is not None


def multi_step_rule(text: str) -> Rule | None:
    for rule in MULTI_STEP_RULES:
        if rule.matches(text):
            return rule
    return None


def multi_step(text: str) -> bool:
    return multi_step_rule(text) is not None


def _tools(res) -> list[str]:
    out = []
    for x in ((res.get("tool_log") or []) if isinstance(res, dict) else []):
        if isinstance(x, dict) and x.get("tool"):
            out.append(str(x["tool"]).lower())
    return out


def mutating_tools(res) -> tuple[str, ...]:
    """The state-changing tool names in ``res["tool_log"]`` (order kept)."""
    return tuple(t for t in _tools(res) if t in MUTATING_TOOLS)


def text_block_reason(text: str) -> str | None:
    """Text-only check, for a cache read/store that has no response yet."""
    if fresh(text):
        return "fresh"
    if multi_step(text):
        return "multi_step"
    return None


def block_reason(res, text: str) -> str | None:
    """Why ``res`` must not be cached for ``text``; ``None`` means "cache it".

    Order matters and is frozen: shape, size, stopped, error, freshness,
    side effects, multi-step.  Freshness is checked before side effects so a
    price question that happened to use a tool reports ``"fresh"``.
    """
    if not isinstance(res, dict):
        return "empty"
    content = res.get("content")
    if not content or not isinstance(content, str) or len(content) < MIN_ANSWER_LEN:
        return "empty"
    if res.get("stopped"):
        return "stopped"
    low = content.lower()
    if any(marker in low for marker in ERROR_MARKERS):
        return "error"
    if fresh(text):
        return "fresh"
    if mutating_tools(res):
        return "mutating"
    if multi_step(text):
        return "multi_step"
    return None


def cacheable(res, text: str) -> bool:
    """Convenience wrapper; callers that keep counters use ``block_reason``."""
    return block_reason(res, text) is None
