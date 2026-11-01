"""Deterministic accounting checks: PDV 17%, FIFO, FX, and arithmetic claims.

All math uses Decimal + ROUND_HALF_UP. No floats in the decision path.
"""
from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP

from .normalize import parse_number

CENT = Decimal("0.01")
DEFAULT_VAT = Decimal("17")


def q2(x) -> Decimal:
    return Decimal(x).quantize(CENT, rounding=ROUND_HALF_UP)


def pdv_exclusive(base, rate=DEFAULT_VAT) -> dict:
    b = q2(base)
    vat = q2(b * Decimal(rate) / Decimal(100))
    return {"base": b, "vat": vat, "total": q2(b + vat), "rate": Decimal(rate)}


def pdv_inclusive(total, rate=DEFAULT_VAT) -> dict:
    t = q2(total)
    b = q2(t / (Decimal(1) + Decimal(rate) / Decimal(100)))
    return {"base": b, "vat": q2(t - b), "total": t, "rate": Decimal(rate)}


def fx(amount, rate) -> Decimal:
    return q2(Decimal(amount) * Decimal(rate))


def fifo(lots: list[tuple], qty) -> dict:
    need = Decimal(qty)
    if need < 0:
        raise ValueError("qty must be >= 0")
    cogs = Decimal(0)
    left = need
    remaining = []
    used = []
    for q, cost in lots:
        q = Decimal(q)
        cost = Decimal(cost)
        if q < 0 or cost < 0:
            raise ValueError("negative lot")
        take = min(q, left)
        if take > 0:
            cogs += take * cost
            used.append({"qty": take, "cost": cost})
            left -= take
        rem = q - take
        if rem > 0:
            remaining.append((rem, cost))
        if left == 0:
            pass
    if left > 0:
        raise ValueError("insufficient stock: missing %s" % left)
    return {"cogs": q2(cogs), "used": used, "remaining": remaining,
            "remaining_value": q2(sum(q * c for q, c in remaining))}


_PCT = re.compile(r"([\d.,]+)\s*%\s*(?:od|na)\s*([\d.,]+)\s*(?:=|je|iznosi)\s*([\d.,]+)", re.I)
_MUL = re.compile(r"([\d.,]+)\s*(?:\*|x|×)\s*([\d.,]+)\s*(?:=|je|iznosi)\s*([\d.,]+)", re.I)
_ADD = re.compile(r"([\d.,]+)\s*\+\s*([\d.,]+)\s*(?:=|je|iznosi)\s*([\d.,]+)", re.I)
_SUB = re.compile(r"([\d.,]+)\s*-\s*([\d.,]+)\s*(?:=|je|iznosi)\s*([\d.,]+)", re.I)
_DIV = re.compile(r"([\d.,]+)\s*/\s*([\d.,]+)\s*(?:=|je|iznosi)\s*([\d.,]+)", re.I)


def _dec(s: str) -> Decimal | None:
    d = parse_number(s)
    return d.normalize() if d is not None else None


def verify_arithmetic(text: str, tol: Decimal = CENT) -> dict:
    t = text or ""
    checks = []
    for m in _PCT.finditer(t):
        p, b, r = (_dec(m.group(i)) for i in (1, 2, 3))
        if None in (p, b, r):
            continue
        exp = q2(b * p / Decimal(100))
        checks.append({"kind": "percent", "expected": exp, "got": r, "ok": abs(exp - r) <= tol})
    for m in _MUL.finditer(t):
        a, b, r = (_dec(m.group(i)) for i in (1, 2, 3))
        if None in (a, b, r):
            continue
        exp = q2(a * b)
        checks.append({"kind": "mul", "expected": exp, "got": r, "ok": abs(exp - r) <= tol})
    for m in _ADD.finditer(t):
        a, b, r = (_dec(m.group(i)) for i in (1, 2, 3))
        if None in (a, b, r):
            continue
        exp = q2(a + b)
        checks.append({"kind": "add", "expected": exp, "got": r, "ok": abs(exp - r) <= tol})
    for m in _SUB.finditer(t):
        a, b, r = (_dec(m.group(i)) for i in (1, 2, 3))
        if None in (a, b, r):
            continue
        exp = q2(a - b)
        checks.append({"kind": "sub", "expected": exp, "got": r, "ok": abs(exp - r) <= tol})
    for m in _DIV.finditer(t):
        a, b, r = (_dec(m.group(i)) for i in (1, 2, 3))
        if None in (a, b, r) or b == 0:
            continue
        exp = q2(a / b)
        checks.append({"kind": "div", "expected": exp, "got": r, "ok": abs(exp - r) <= tol})
    return {"checked": bool(checks), "ok": all(c["ok"] for c in checks), "checks": checks}


def verify_pdv_claim(text: str, rate=DEFAULT_VAT) -> dict:
    """Check PDV relations that are fully stated in the claim.

    Recognizes: base/pdv/total keywords, percent-of equations, and exclusive/inclusive pairs.
    """
    t = text or ""
    ar = verify_arithmetic(t)
    nums = []
    for m in re.finditer(r"(?<![\w])([\d.,]+)(?![\w])", t):
        d = parse_number(m.group(1))
        if d is not None:
            nums.append((m.start(), q2(d)))
    low = t.lower()
    out = {"checked": False, "ok": None, "arith": ar, "pdv": None}
    if ar.get("checked"):
        out.update({"checked": True, "ok": ar["ok"]})
    if "pdv" not in low and "porez" not in low:
        return out
    # base + pdv = total pattern
    if len(nums) >= 3 and any(w in low for w in ("osnovic", "base")):
        vals = [n for _, n in nums]
        for i in range(len(vals)):
            for j in range(len(vals)):
                for k in range(len(vals)):
                    if i == j or j == k or i == k:
                        continue
                    if abs((vals[i] + vals[j]) - vals[k]) <= CENT:
                        exp_vat = q2(vals[i] * Decimal(rate) / Decimal(100))
                        ok = abs(exp_vat - vals[j]) <= CENT
                        out.update({"checked": True, "ok": ok,
                                    "pdv": {"base": vals[i], "vat": vals[j], "total": vals[k],
                                            "expected_vat": exp_vat}})
                        return out
    # inclusive: total and vat -> base
    if len(nums) >= 2 and ("ukupno" in low or "bruto" in low):
        vals = [n for _, n in nums]
        total, vat = vals[0], vals[1]
        exp_base = q2(total / (Decimal(1) + Decimal(rate) / Decimal(100)))
        exp_vat = q2(total - exp_base)
        out.update({"checked": True, "ok": abs(vat - exp_vat) <= CENT,
                    "pdv": {"total": total, "vat": vat,
                            "expected_base": exp_base, "expected_vat": exp_vat}})
        return out
    return out


def verify_fx_claim(amount, rate, expected) -> dict:
    exp = fx(amount, rate)
    got = q2(expected)
    return {"ok": abs(exp - got) <= CENT, "expected": exp, "got": got,
            "amount": q2(amount), "rate": Decimal(rate)}
