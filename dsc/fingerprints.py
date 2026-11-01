"""Deterministic fingerprints: SHA-256 identity, SimHash (cosine), MinHash (Jaccard)."""
from __future__ import annotations

import hashlib
import struct
from typing import Iterable

from .normalize import canonical, content_tokens, fold, tokens


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8", "replace")
    return hashlib.sha256(data).hexdigest()


def _h64(data: bytes, key: bytes = b"") -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8, key=key).digest(), "big")


def shingles(text: str, k: int = 5) -> list[str]:
    w = content_tokens(text)
    if len(w) >= k:
        return [" ".join(w[i:i + k]) for i in range(len(w) - k + 1)]
    s = fold(canonical(text)).replace(" ", "_")
    if len(s) <= 4:
        return [s] if s else []
    return [s[i:i + 4] for i in range(len(s) - 3)]


def _weights(text: str) -> list[tuple[str, float]]:
    out: dict[str, float] = {}
    for t in content_tokens(text):
        out[t] = out.get(t, 0.0) + 1.0
    f = fold(canonical(text)).replace(" ", "_")
    for n in (3, 4):
        for i in range(max(0, len(f) - n + 1)):
            g = f[i:i + n]
            out["~" + g] = out.get("~" + g, 0.0) + 0.3
    return [(k, v) for k, v in out.items() if v > 0]


def simhash64(text: str) -> int:
    acc = [0.0] * 64
    for tok, w in _weights(text):
        h = _h64(tok.encode("utf-8", "replace"))
        for b in range(64):
            acc[b] += w if (h >> b) & 1 else -w
    out = 0
    for b in range(64):
        if acc[b] > 0:
            out |= 1 << b
    return out


def hamming(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def minhash128(text: str, num_perm: int = 128) -> tuple[int, ...]:
    sh = shingles(text)
    if not sh:
        return tuple([0] * num_perm)
    enc = [s.encode("utf-8", "replace") for s in sh]
    sig = []
    for i in range(num_perm):
        key = ("dsc-minhash-v1-%03d" % i).encode("ascii")
        sig.append(min(_h64(x, key) for x in enc) & 0xFFFFFFFF)
    return tuple(sig)


def jaccard_est(a: Iterable[int], b: Iterable[int]) -> float:
    a, b = tuple(a), tuple(b)
    if not a or len(a) != len(b):
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / float(len(a))


def jaccard_true(a: str, b: str, k: int = 5) -> float:
    sa, sb = set(shingles(a, k)), set(shingles(b, k))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def pack_sig(sig: Iterable[int]) -> bytes:
    return struct.pack("<%dI" % len(tuple(sig)), *tuple(sig))


def unpack_sig(blob: bytes) -> tuple[int, ...]:
    n = len(blob) // 4
    return struct.unpack("<%dI" % n, blob)


def lsh_bands(sig: Iterable[int], bands: int = 16, rows: int = 8) -> list[str]:
    s = tuple(sig)
    out = []
    for b in range(bands):
        chunk = s[b * rows:(b + 1) * rows]
        out.append(hashlib.blake2b(struct.pack("<%dI" % len(chunk), *chunk),
                                   digest_size=8, key=b"dsc-lsh-v1").hexdigest())
    return out


def similarity(a: str, b: str) -> dict:
    sa, sb = simhash64(a), simhash64(b)
    return {
        "sha_equal": sha256_hex(canonical(a)) == sha256_hex(canonical(b)),
        "simhash": {"a": sa, "b": sb, "hamming": hamming(sa, sb)},
        "minhash_jaccard": round(jaccard_est(minhash128(a), minhash128(b)), 4),
        "true_jaccard": round(jaccard_true(a, b), 4),
    }
