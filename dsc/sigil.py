"""Map PoA records to a SIGIL/1-style receipt (no signatures yet).

SIGIL/1 = Signed Intent, Grant, and Invariant Ledger. We map the deterministic parts:
intent_hash, policy_hash, result_digest, chain, anchor, receipt_hash.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()


def _key_id(pub_bytes: bytes) -> str:
    return _sha(pub_bytes.hex())[:16]


def generate_keypair(private_path: str, public_path: str) -> dict:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.generate()
    priv = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    with open(private_path, "wb") as f:
        f.write(priv)
    os.chmod(private_path, 0o600)
    with open(public_path, "wb") as f:
        f.write(pub)
    return {"private": private_path, "public": public_path,
            "key_id": _key_id(key.public_key().public_bytes_raw())}


def load_private(path: str):
    from cryptography.hazmat.primitives import serialization
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def load_public(path: str):
    from cryptography.hazmat.primitives import serialization
    with open(path, "rb") as f:
        return serialization.load_pem_public_key(f.read())


def sign_receipt(rec: dict, private_key, key_id: str | None = None) -> dict:
    core = {k: v for k, v in rec.items() if k not in ("signature", "receipt_hash")}
    # receipt_hash is unsigned metadata; sign the canonical core including it
    payload = dict(core)
    payload["receipt_hash"] = rec.get("receipt_hash")
    sig = private_key.sign(_canon(payload).encode("utf-8", "replace"))
    out = dict(rec)
    out["signature"] = {"algo": "ed25519",
                        "key_id": key_id or _key_id(private_key.public_key().public_bytes_raw()),
                        "sig": base64.b64encode(sig).decode("ascii")}
    return out


def verify_signature(rec: dict, public_key) -> bool:
    from cryptography.exceptions import InvalidSignature
    sig = rec.get("signature") or {}
    if sig.get("algo") != "ed25519" or not sig.get("sig"):
        return False
    core = {k: v for k, v in rec.items() if k not in ("signature", "receipt_hash")}
    payload = dict(core)
    payload["receipt_hash"] = rec.get("receipt_hash")
    try:
        public_key.verify(base64.b64decode(sig["sig"]), _canon(payload).encode("utf-8", "replace"))
        return True
    except (InvalidSignature, ValueError, KeyError):
        return False


def sigil_receipt(poa: dict, policy: dict | None = None, intent=None,
                  anchor: str | None = None) -> dict:
    payload = poa.get("payload") or {}
    intent_obj = intent if intent is not None else payload.get("inputs", {})
    pol = policy or {}
    policy_obj = {"decision": pol.get("decision", "allow"),
                  "rule_ids": list(pol.get("rule_ids") or [])}
    rec = {
        "sigil": "1.0",
        "receipt_id": poa.get("id"),
        "ts": poa.get("ts"),
        "intent_hash": _sha(_canon(intent_obj)),
        "policy": {**policy_obj, "policy_hash": _sha(_canon(policy_obj))},
        "result_digest": payload.get("result_hash"),
        "chain": poa.get("chain"),
        "anchor": anchor,
        "signature": None,
    }
    core = {k: v for k, v in rec.items() if k not in ("signature", "receipt_hash")}
    rec["receipt_hash"] = _sha(_canon(core))
    return rec


def verify_sigil(rec: dict, result=None, intent=None, require_signature: bool = False,
                 public_key=None) -> dict:
    errs = []
    core = {k: v for k, v in rec.items() if k not in ("signature", "receipt_hash")}
    if rec.get("receipt_hash") != _sha(_canon(core)):
        errs.append("E_RECEIPT_HASH")
    if result is not None and rec.get("result_digest") != _sha(_canon(result)):
        errs.append("E_RESULT_DIGEST")
    if intent is not None and rec.get("intent_hash") != _sha(_canon(intent)):
        errs.append("E_INTENT_HASH")
    pol = rec.get("policy") or {}
    if pol.get("policy_hash") != _sha(_canon({"decision": pol.get("decision"),
                                             "rule_ids": pol.get("rule_ids")})):
        errs.append("E_POLICY_HASH")
    if require_signature and not rec.get("signature"):
        errs.append("E_UNSIGNED")
    if public_key is not None and rec.get("signature") and not verify_signature(rec, public_key):
        errs.append("E_SIGNATURE")
    return {"ok": not errs, "errors": errs, "signed": bool(rec.get("signature")),
            "receipt_hash": rec.get("receipt_hash"),
            "key_id": (rec.get("signature") or {}).get("key_id")}


def to_sigil(*args, **kwargs) -> dict:
    return sigil_receipt(*args, **kwargs)
