"""Deterministic money-amount parsing — the single copy in the GAMA stack.

Moved 2026-11-26 out of ``gama-fastpath/fastpath.py`` (``parse_amount``) so the
gate, the HTTP API and the offline layer share one implementation.  Pure
stdlib: ``re`` + ``Decimal``, no locale, no floats.

Accepted formats (BHS / EU / US, all pinned by tests)::

    "2.400,00"      -> Decimal("2400.00")
    "1.234.567,89"  -> Decimal("1234567.89")
    "2,400.00"      -> Decimal("2400.00")
    "2400"          -> Decimal("2400")
    "-17,5"         -> Decimal("-17.5")
    "KM 2.400,00"   -> Decimal("2400.00")   (currency tokens stripped)

Why this is not :func:`dsc.normalize.parse_number`: ``parse_number`` extracts a
number that is embedded in *prose*, where ``1,234`` is a thousands group.  A
money *field* is parsed with a lone comma as the decimal separator
(``12,345`` -> 12.345) unless commas split three-digit groups twice or more.
Both rules are pinned by tests; they intentionally disagree on ``1,234`` and
:func:`parse_amount` documents that with ``test_amount.py``.  The accounting
module keeps using ``Decimal`` inputs, so callers convert once, here.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

__all__ = ["parse_amount"]

_NOT_MONEY = re.compile(r"[^0-9.,]")


def parse_amount(raw) -> Decimal:
    """Parse one money field.  Raise ``ValueError`` when there is no amount.

    ``raw`` may contain currency symbols/letters and thousands separators; they
    are stripped before the separator disambiguation, never guessed from locale.
    """
    t = str(raw or "").strip().replace("\u00a0", "").replace(" ", "")
    neg = t.startswith("-")
    t = t.lstrip("+-")
    t = _NOT_MONEY.sub("", t).strip(".,")
    if not t:
        raise ValueError("no amount")
    if "." in t and "," in t:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        parts = t.split(",")
        if len(parts) > 2:
            t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
    elif "." in t:
        parts = t.split(".")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3):
            t = t.replace(".", "")
    try:
        v = Decimal(t)
    except InvalidOperation:
        raise ValueError("no amount")
    if not v.is_finite():
        raise ValueError("no amount")
    return -v if neg else v
