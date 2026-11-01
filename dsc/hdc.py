"""Deterministic hyperdimensional vector: hashed n-grams -> signed random projection -> int8.

Same text in, same bytes out, on every process and every machine (BLAKE2b, no built-in hash).
"""
from __future__ import annotations

import hashlib
import math
import struct

from .normalize import canonical, fold, tokens

DIM = 1024
_K = b"dsc-hdc-v1"


def _h(data: bytes, key: bytes = _K) -> bytes:
    return hashlib.blake2b(data, digest_size=8, key=key).digest()


def dense(text: str, dim: int = DIM) -> list[float]:
    acc = [0.0] * dim
    f = fold(canonical(text)).replace(" ", "_")
    grams: dict[str, float] = {}
    for n in (3, 4, 5):
        for i in range(max(0, len(f) - n + 1)):
            g = f[i:i + n]
            grams[g] = grams.get(g, 0.0) + 1.0
    for g, tf in grams.items():
        h = _h(g.encode("utf-8", "replace"))
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        acc[idx] += sign * (1.0 / math.sqrt(len(g))) * (1.0 + math.log(tf))
    for t in tokens(text):
        h = _h(b"w:" + t.encode("utf-8", "replace"))
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        acc[idx] += 2.0 * sign
    norm = math.sqrt(sum(x * x for x in acc))
    if norm > 0:
        acc = [x / norm for x in acc]
    return acc


def quantize(vec: list[float]) -> bytes:
    mx = max((abs(x) for x in vec), default=0.0)
    if mx == 0:
        return bytes(len(vec))
    scale = 127.0 / mx
    out = bytearray(len(vec))
    for i, x in enumerate(vec):
        q = int(round(x * scale))
        if q > 127:
            q = 127
        elif q < -127:
            q = -127
        out[i] = q & 0xFF
    return bytes(out)


def unquantize(blob: bytes) -> list[int]:
    return list(struct.unpack("%db" % len(blob), blob))


def vector(text: str, dim: int = DIM) -> bytes:
    return quantize(dense(text, dim))


def cosine_q(a: bytes, b: bytes) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = struct.unpack("%db" % len(a), a)
    vb = struct.unpack("%db" % len(b), b)
    dot = 0
    na = 0
    nb = 0
    for x, y in zip(va, vb):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / math.sqrt(na * nb)


def cosine_f(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def binary(vec: list[float]) -> bytes:
    out = bytearray((len(vec) + 7) // 8)
    for i, x in enumerate(vec):
        if x > 0:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def hamming_bin(a: bytes, b: bytes) -> int:
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def bind(a: list[float], b: list[float]) -> list[float]:
    return [x * y for x, y in zip(a, b)]


def bundle(vs: list[list[float]]) -> list[float]:
    if not vs:
        return []
    dim = len(vs[0])
    out = [0.0] * dim
    for v in vs:
        for i in range(dim):
            out[i] += v[i]
    norm = math.sqrt(sum(x * x for x in out))
    return [x / norm for x in out] if norm else out


def permute(v: list[float], k: int = 1) -> list[float]:
    if not v:
        return v
    k %= len(v)
    return v[-k:] + v[:-k] if k else list(v)
