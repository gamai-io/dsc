"""DSC deterministic normalization (NFC, diacritics, numbers, units, negation).

Pure stdlib. Every function is deterministic across processes: no built-in hash(),
no locale, no randomness, no network.
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

MOJIBAKE = ("Ã", "Â", "â€", "Å¾", "Ä‡", "Ä", "Å¡", "Å", "Å¾")

_FOLD_MAP = str.maketrans({
    "đ": "d", "Đ": "d", "ß": "ss", "ø": "o", "Ø": "o", "ł": "l", "Ł": "l",
    "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "þ": "th", "ð": "d",
})

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")
_QUOTES = {ord("\u2018"): "'", ord("\u2019"): "'", ord("\u201c"): '"', ord("\u201d"): '"',
           ord("\u00ab"): '"', ord("\u00bb"): '"'}
_LIGATURES = {ord("ﬁ"): "fi", ord("ﬂ"): "fl", ord("ﬀ"): "ff", ord("ﬃ"): "ffi", ord("ﬄ"): "ffl"}

_NUM_RE = re.compile(
    r"(?<![\w])"
    r"([-+]?\d{1,3}(?:[.\u00a0]\d{3})+(?:,\d+)?"
    r"|[-+]?\d+(?:[.,]\d+)?)"
    r"(?![\w])"
)

_UNITS = {
    "PSI": ("pressure", 6894.757293168), "psi": ("pressure", 6894.757293168),
    "bar": ("pressure", 100000.0), "mbar": ("pressure", 100.0),
    "MPa": ("pressure", 1000000.0), "kPa": ("pressure", 1000.0), "Pa": ("pressure", 1.0),
    "kW": ("power", 1000.0), "KW": ("power", 1000.0), "KS": ("power", 735.49875),
    "m3": ("volume", 1.0), "m³": ("volume", 1.0), "L": ("volume", 0.001), "l": ("volume", 0.001),
    "ml": ("volume", 1e-6),
    "kg": ("mass", 1.0), "g": ("mass", 0.001), "t": ("mass", 1000.0),
    "mm": ("length", 0.001), "cm": ("length", 0.01), "km": ("length", 1000.0), "m": ("length", 1.0),
    "min": ("time", 60.0), "h": ("time", 3600.0), "s": ("time", 1.0),
    "KM": ("currency", 1.0), "EUR": ("currency", 1.0), "€": ("currency", 1.0),
    "USD": ("currency", 1.0), "$": ("currency", 1.0),
    "%": ("percent", 1.0),
}
_UNIT_KEYS = sorted(_UNITS, key=len, reverse=True)

_DATE_DMY = re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})(?!\d)")
_DATE_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_YEAR = re.compile(r"(?<!\d)(19\d{2}|20\d{2}|21\d{2})(?!\d)")

NEGATORS = frozenset({
    "ne", "nije", "nisu", "nema", "nemoj", "nemojte", "bez", "nikad", "nikada",
    "ni", "niti", "nista", "iskljuci", "ukini", "zabrani", "prekini", "ponisti",
    "stop", "necu", "nece", "nezelim",
})
ANTONYMS = (
    ("ukljuci", "iskljuci"), ("povecaj", "smanji"), ("dodaj", "obrisi"),
    ("prije", "poslije"), ("vece", "manje"), ("iznad", "ispod"),
    ("pocni", "zaustavi"), ("otvori", "zatvori"), ("da", "ne"),
    ("true", "false"), ("omoguci", "onemoguci"), ("aktivan", "neaktivan"),
)
ACTION_WORDS = frozenset({
    "posalji", "otvori", "pokreni", "obrisi", "ukucaj", "klikni", "napravi",
    "kreiraj", "instaliraj", "deploy", "kupi", "plati", "prebaci", "pusti",
    "dodaj", "unesi", "sacuvaj", "zamijeni", "promijeni", "restartuj", "ugasi",
    "napis", "napisi", "izbrisi", "kopiraj", "premjesti", "zakazi", "posaljite",
    "poslati", "posaljem", "posaljemo", "posaljete", "posalji", "posaljite",
    "prebaciti", "otvoriti", "pokrenuti", "obrisati", "ukucati", "kliknuti",
})
FRESH_WORDS = frozenset({
    "pretrazi", "search", "cijena", "cijenu", "cijene", "najnovij", "najnove",
    "online", "internet", "netu", "trenutno", "danas", "juce", "sada", "kurs",
    "prognoza", "vremenska", "berza", "akcije", "kripto", "nadzor", "prozor",
    "screenshot", "prebaci", "ekran", "flota", "zaliha", "zalihe", "stock",
    "lager", "stanje", "raspolozivo", "slobodno",
})

_STOP = frozenset({
    "i", "u", "na", "za", "od", "do", "sa", "se", "je", "su", "da", "li",
    "kao", "po", "iz", "ali", "ili", "te", "ni", "biti", "koji", "koja",
    "koje", "koji", "the", "a", "an", "of", "to", "in", "on", "is", "are",
})

_SUFFIXES = ("ovima", "evima", "anjima", "enjima", "nost", "osti", "ima", "ama",
             "oga", "ega", "omu", "emu", "ije", "je", "ih", "om", "em", "og",
             "eg", "u", "a", "e", "i", "o")


def fix_mojibake(s: str) -> str:
    if not s or not any(m in s for m in MOJIBAKE):
        return s or ""
    for enc in ("cp1252", "latin-1"):
        try:
            t = s.encode(enc).decode("utf-8")
            if t:
                return t
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


def to_nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.translate(_FOLD_MAP)


def clean(s: str) -> str:
    s = fix_mojibake(s)
    s = to_nfc(s)
    s = s.translate(_DASHES).translate(_QUOTES).translate(_LIGATURES)
    s = s.replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", s).strip()


def parse_number(raw: str) -> Decimal | None:
    t = (raw or "").replace("\u00a0", "").replace(" ", "").strip()
    if not t:
        return None
    sign = ""
    if t[0] in "+-":
        sign, t = t[0], t[1:]
    if not t:
        return None
    if "." in t and "," in t:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", t):
            t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
    elif "." in t:
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
            t = t.replace(".", "")
    try:
        d = Decimal(sign + t)
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None
    return d.normalize()


def dec_str(d: Decimal) -> str:
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def _unit_at(text: str, pos: int) -> tuple[str | None, str | None, float | None]:
    tail = text[pos:pos + 12].lstrip()
    for u in _UNIT_KEYS:
        if tail.startswith(u):
            nxt = tail[len(u):len(u) + 1]
            if nxt.isalpha():
                continue
            dim, mult = _UNITS[u]
            return u, dim, mult
    return None, None, None


def extract_numbers(text: str) -> list[dict]:
    masked = _DATE_ISO.sub(lambda m: " " * len(m.group(0)), text or "")
    masked = _DATE_DMY.sub(lambda m: " " * len(m.group(0)), masked)
    out = []
    for m in _NUM_RE.finditer(masked):
        d = parse_number(m.group(1))
        if d is None:
            continue
        unit, dim, mult = _unit_at(masked, m.end())
        base = None
        if dim is not None and mult is not None:
            base = d * Decimal(str(mult))
        out.append({
            "raw": m.group(1), "value": dec_str(d), "unit": unit, "dim": dim,
            "base": dec_str(base) if base is not None else None,
            "pos": m.start(),
        })
    return out


def extract_dates(text: str) -> list[str]:
    out = []
    for m in _DATE_ISO.finditer(text or ""):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            out.append("%04d-%02d-%02d" % (y, mo, d))
    for m in _DATE_DMY.finditer(text or ""):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        if 1 <= mo <= 12 and 1 <= d <= 31:
            out.append("%04d-%02d-%02d" % (y, mo, d))
    return out


def extract_years(text: str) -> list[str]:
    return [m.group(1) for m in _YEAR.finditer(text or "")]


def tokens(text: str) -> list[str]:
    f = fold(clean(text))
    f = re.sub(r"[^a-z0-9%.,/+\- ]+", " ", f)
    return [t for t in f.split() if t]


def stem(tok: str) -> str:
    if len(tok) <= 5 or not tok.isalpha():
        return tok
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[:-len(suf)]
    return tok


def content_tokens(text: str) -> list[str]:
    return [stem(t) for t in tokens(text) if t not in _STOP and len(t) > 1]


def canonical_number_tokens(text: str) -> list[str]:
    out = []
    for t in tokens(text):
        d = parse_number(t) if re.fullmatch(r"[-+]?\d[\d.,]*", t) else None
        if d is not None:
            out.append("#" + dec_str(d))
        else:
            out.append(t)
    return out


def canonical(text: str) -> str:
    return " ".join(canonical_number_tokens(text))


def numeric_signature(text: str) -> tuple:
    vals = []
    for n in extract_numbers(text):
        vals.append((n["dim"] or "", n["base"] if n["base"] is not None else n["value"]))
    return tuple(sorted(vals))


def has_money_unit_percent(text: str) -> bool:
    return any(n["dim"] in ("currency", "percent", "pressure", "power", "volume", "mass", "length", "time")
               for n in extract_numbers(text))


def negation_count(text: str) -> int:
    return sum(1 for t in tokens(text) if t in NEGATORS)


def has_negation(text: str) -> bool:
    return negation_count(text) > 0


def antonym_conflict(a: str, b: str) -> list[str]:
    ta, tb = set(tokens(a)), set(tokens(b))
    hits = []
    for x, y in ANTONYMS:
        if (x in ta and y in tb) or (y in ta and x in tb):
            hits.append("%s/%s" % (x, y))
    return hits


def is_action(text: str) -> bool:
    return any(t in ACTION_WORDS for t in tokens(text))


def is_fresh(text: str) -> bool:
    return any(t in FRESH_WORDS for t in tokens(text))


_MULTI = re.compile(
    r"(?i)(?:\s+i\s+|\s+pa\s+|\s+zatim\s+|\s+potom\s+|\s+onda\s+|;|\s+&\s+)"
    r".{0,80}?\b(otvori|pokreni|napisi|posalji|ukucaj|klikni|napravi|kreiraj|"
    r"obrisi|pusti|prebaci|dodaj|unesi|instaliraj)\b"
)


def is_multi_step(text: str) -> bool:
    return bool(_MULTI.search(text or ""))


def is_risky(text: str) -> bool:
    return is_action(text) or is_fresh(text) or is_multi_step(text)


def features(text: str) -> dict:
    nfc = to_nfc(clean(text))
    nums = extract_numbers(nfc)
    return {
        "raw": text or "",
        "nfc": nfc,
        "folded": fold(nfc),
        "tokens": tokens(nfc),
        "content": content_tokens(nfc),
        "numbers": nums,
        "numeric_signature": numeric_signature(nfc),
        "dates": extract_dates(nfc),
        "years": extract_years(nfc),
        "negations": negation_count(nfc),
        "action": is_action(nfc),
        "fresh": is_fresh(nfc),
        "multi": is_multi_step(nfc),
        "canonical": canonical(nfc),
    }
