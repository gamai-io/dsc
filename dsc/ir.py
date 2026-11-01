"""DSC IR container v1 (GBIN) - R1.

What it is: a small, versioned, content-addressed container for
  * kind=1  a DSC L program (source + metadata),
  * kind=2  an arbitrary canonical value (JSON subset, RFC 8785).

What it is NOT: a bytecode VM. The language stays interpreted (v7); this is the
serialization boundary that lets a program/value cross a process or node,
be hashed reliably, and be verified on the other side.

Layout (all integers big-endian):
   0..3   magic  b"GBIN"
   4      version   u8
   5      kind      u8   (1=program, 2=value)
   6      flags     u8   (bit0 = payload is zstd-compressed)
   7      reserved  u8   (must be 0)
   8..11  length    u32  (payload length in bytes)
   12..   payload
   end-32 digest  sha256(payload)  - integrity, checked before parsing

The payload for both kinds is RFC 8785 JCS bytes (see jcs.py), so the same
logical content always yields the same container bytes -> same cid.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Any

from . import jcs

MAGIC = b"GBIN"
VERSION = 1
KIND_PROGRAM = 1
KIND_VALUE = 2
FLAG_ZSTD = 0b0000_0001
_HEADER = struct.Struct(">4sBBBB I")  # magic, ver, kind, flags, reserved, length


class IRError(ValueError):
    """Structured IR error; `code` is a stable machine-readable token."""

    def __init__(self, code: str, message: str = ""):
        super().__init__("%s: %s" % (code, message) if message else code)
        self.code = code


def _zstd_compress(data: bytes) -> bytes:
    try:
        import zstandard as zstd  # optional dependency
    except Exception as exc:  # noqa: BLE001
        raise IRError("E_NO_ZSTD", "zstandard nije instaliran") from exc
    return zstd.ZstdCompressor(level=9).compress(data)


def _zstd_decompress(data: bytes) -> bytes:
    try:
        import zstandard as zstd  # optional dependency
    except Exception as exc:  # noqa: BLE001
        raise IRError("E_NO_ZSTD", "zstandard nije instaliran") from exc
    return zstd.ZstdDecompressor().decompress(data)


def _pack(kind: int, payload_bytes: bytes, *, compress: bool = False) -> bytes:
    body = payload_bytes
    flags = 0
    if compress:
        body = _zstd_compress(body)
        flags |= FLAG_ZSTD
    header = _HEADER.pack(MAGIC, VERSION, kind, flags, 0, len(body))
    return header + body + hashlib.sha256(body).digest()


def _unpack(blob: bytes) -> tuple[int, int, bytes]:
    if len(blob) < _HEADER.size + 32:
        raise IRError("E_TRUNCATED", "prekratak container")
    magic, version, kind, flags, reserved, length = _HEADER.unpack(blob[: _HEADER.size])
    if magic != MAGIC:
        raise IRError("E_MAGIC", repr(magic))
    if version != VERSION:
        raise IRError("E_VERSION", str(version))
    if reserved != 0:
        raise IRError("E_RESERVED", str(reserved))
    body = blob[_HEADER.size : _HEADER.size + length]
    digest = blob[_HEADER.size + length : _HEADER.size + length + 32]
    if len(body) != length or len(digest) != 32:
        raise IRError("E_LENGTH", "%d/%d" % (len(body), length))
    if hashlib.sha256(body).digest() != digest:
        raise IRError("E_DIGEST", "sadrzaj ne odgovara digestu")
    if flags & FLAG_ZSTD:
        body = _zstd_decompress(body)
    if flags & ~FLAG_ZSTD:
        raise IRError("E_FLAGS", str(flags))
    return version, kind, body


def pack_program(source: str, meta: dict | None = None, *, compress: bool = False) -> bytes:
    """Serialize a DSC L program (source + optional metadata)."""
    if not isinstance(source, str) or not source.strip():
        raise IRError("E_EMPTY_PROGRAM", "prazan izvor")
    doc = {"kind": "program", "lang": "dsc-l/1", "source": source, "meta": meta or {}}
    return _pack(KIND_PROGRAM, jcs.to_bytes(doc), compress=compress)


def unpack_program(blob: bytes) -> dict:
    """Return {source, meta, cid} after full integrity verification."""
    version, kind, body = _unpack(blob)
    if kind != KIND_PROGRAM:
        raise IRError("E_KIND", "ocekivan program, dobio %d" % kind)
    try:
        doc = jcs.loads(body.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise IRError("E_PAYLOAD", repr(exc)[:120]) from exc
    if doc.get("lang") != "dsc-l/1":
        raise IRError("E_LANG", str(doc.get("lang")))
    return {"source": doc["source"], "meta": doc.get("meta") or {}, "cid": cid(blob), "version": version}


def pack_value(value: Any, *, compress: bool = False) -> bytes:
    return _pack(KIND_VALUE, jcs.to_bytes(value), compress=compress)


def unpack_value(blob: bytes) -> Any:
    _version, kind, body = _unpack(blob)
    if kind != KIND_VALUE:
        raise IRError("E_KIND", "ocekivan value, dobio %d" % kind)
    return jcs.loads(body.decode("utf-8"))


def cid(blob: bytes) -> str:
    """Content id of the whole container - stable across nodes."""
    return hashlib.sha256(blob).hexdigest()


def info(blob: bytes) -> dict:
    """Header + cid without decoding the payload (still verifies integrity)."""
    version, kind, body = _unpack(blob)
    return {
        "magic": MAGIC.decode(),
        "version": version,
        "kind": "program" if kind == KIND_PROGRAM else "value",
        "payload_bytes": len(body),
        "container_bytes": len(blob),
        "cid": cid(blob),
        "payload_sha256": hashlib.sha256(body).hexdigest(),
    }
