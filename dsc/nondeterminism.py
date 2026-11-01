"""LLM nondeterminism: envelope, receipt, replay key, determinism levels.

Based on Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" (Sep 2025):
temperature=0 is not enough. The hidden input is batch size/composition; without
batch-invariant kernels the same prompt gives different outputs under different load.

We cannot make a remote LLM batch-invariant. We can:
  1. record the envelope (engine/model/version/quant/batch/seed/temperature),
  2. refuse to claim EXACT reproducibility without batch-invariant kernels,
  3. bind cache entries to the envelope so a hit is never replayed across batches,
  4. downgrade PoA verdict when a critical claim needs EXACT but gets STOCHASTIC.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields

from .normalize import canonical

LEVELS = ("EXACT", "BATCH_INVARIANT", "VERSIONED", "STOCHASTIC")
_RANK = {k: i for i, k in enumerate(LEVELS)}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class Envelope:
    engine: str = "unknown"
    engine_version: str = "0"
    model: str = "unknown"
    model_hash: str = ""
    quantization: str = ""
    precision: str = "bf16"
    device: str = "cpu"
    driver: str = ""
    batch_size: int = 1
    batch_signature: str = ""
    seed: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 0
    batch_invariant: bool = False
    deterministic_kernels: bool = False
    extra: tuple = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["extra"] = list(self.extra)
        return d

    def identity_fields(self) -> dict:
        return {
            "engine": self.engine, "engine_version": self.engine_version,
            "model": self.model, "model_hash": self.model_hash,
            "quantization": self.quantization, "precision": self.precision,
            "device": self.device, "driver": self.driver,
            "batch_invariant": self.batch_invariant,
            "deterministic_kernels": self.deterministic_kernels,
            "extra": list(self.extra),
        }

    def identity(self) -> str:
        return _sha(_canon(self.identity_fields()))

    def full_id(self) -> str:
        return _sha(_canon(self.to_dict()))

    def level(self) -> str:
        if self.temperature > 0.0:
            return "STOCHASTIC"
        if self.batch_invariant and self.deterministic_kernels:
            return "EXACT"
        if self.batch_invariant:
            return "BATCH_INVARIANT"
        if self.temperature <= 0.0:
            return "VERSIONED"
        return "STOCHASTIC"

    def replay_bound(self) -> bool:
        if self.batch_invariant:
            return True
        return bool(self.batch_signature)

    def replay_key(self, prompt: str, strict_batch: bool | None = None) -> str:
        strict = (not self.batch_invariant) if strict_batch is None else bool(strict_batch)
        batch = "%d|%s" % (self.batch_size, self.batch_signature) if strict else "batch-invariant"
        parts = [
            canonical(prompt), self.identity(), str(self.seed), repr(float(self.temperature)),
            repr(float(self.top_p)), str(self.top_k), str(self.max_tokens), batch,
        ]
        return _sha("|".join(parts))


def required_ok(level: str, required: str = "VERSIONED") -> bool:
    return _RANK.get(level, 99) <= _RANK.get(required, 2)


def receipt(prompt: str, output: str, env: Envelope, logprobs: list | None = None,
            ts: float | None = None) -> dict:
    level = env.level()
    rec = {
        "schema": "dsc/receipt/1",
        "ts": time.time() if ts is None else ts,
        "envelope": env.to_dict(),
        "envelope_id": env.full_id(),
        "identity": env.identity(),
        "level": level,
        "replay_bound": env.replay_bound(),
        "replay_key": env.replay_key(prompt),
        "prompt_sha256": _sha(canonical(prompt)),
        "output_sha256": _sha(output or ""),
        "logprobs_sha256": _sha(_canon(logprobs)) if logprobs is not None else None,
    }
    rec["receipt_id"] = _sha(_canon({k: rec[k] for k in (
        "envelope_id", "replay_key", "prompt_sha256", "output_sha256", "level")}))
    return rec


def verify_receipt(rec: dict, prompt: str | None = None, output: str | None = None) -> dict:
    bad = []
    if rec.get("schema") != "dsc/receipt/1":
        bad.append("schema")
    try:
        env = Envelope(**{k: v for k, v in rec["envelope"].items() if k in
                          {f.name for f in fields(Envelope)}})
    except Exception:
        return {"ok": False, "bad": ["envelope"]}
    if env.full_id() != rec.get("envelope_id"):
        bad.append("envelope_id")
    if env.level() != rec.get("level"):
        bad.append("level")
    if prompt is not None and _sha(canonical(prompt)) != rec.get("prompt_sha256"):
        bad.append("prompt")
    if output is not None and _sha(output or "") != rec.get("output_sha256"):
        bad.append("output")
    if prompt is not None and env.replay_key(prompt) != rec.get("replay_key"):
        bad.append("replay_key")
    expect = _sha(_canon({k: rec.get(k) for k in (
        "envelope_id", "replay_key", "prompt_sha256", "output_sha256", "level")}))
    if expect != rec.get("receipt_id"):
        bad.append("receipt_id")
    return {"ok": not bad, "bad": bad, "level": env.level(),
            "replay_bound": env.replay_bound(), "identity": env.identity()}


def sign_inference(rec: dict, private_key, key_id: str | None = None) -> dict:
    """Sign a receipt() with Ed25519. Signature covers the full receipt except itself."""
    core = {k: v for k, v in rec.items() if k != "signature"}
    sig = private_key.sign(_canon(core).encode("utf-8", "replace"))
    out = dict(rec)
    pub = private_key.public_key().public_bytes_raw()
    out["signature"] = {"algo": "ed25519",
                        "key_id": key_id or _sha(pub.hex())[:16],
                        "sig": base64.b64encode(sig).decode("ascii")}
    return out


def verify_inference(rec: dict, public_key) -> bool:
    from cryptography.exceptions import InvalidSignature
    sig = rec.get("signature") or {}
    if sig.get("algo") != "ed25519" or not sig.get("sig"):
        return False
    core = {k: v for k, v in rec.items() if k != "signature"}
    try:
        public_key.verify(base64.b64decode(sig["sig"]), _canon(core).encode("utf-8", "replace"))
        return True
    except (InvalidSignature, ValueError, KeyError):
        return False


def compare(a: dict, b: dict) -> dict:
    return {
        "same_identity": a.get("identity") == b.get("identity"),
        "same_level": a.get("level") == b.get("level"),
        "same_replay_key": a.get("replay_key") == b.get("replay_key"),
        "same_output": a.get("output_sha256") == b.get("output_sha256"),
        "levels": (a.get("level"), b.get("level")),
    }


def guard(env: Envelope, requirement: str = "VERSIONED") -> dict:
    lv = env.level()
    ok = required_ok(lv, requirement)
    reasons = []
    if not ok:
        reasons.append("level %s < required %s" % (lv, requirement))
    if not env.batch_invariant:
        reasons.append("no batch-invariant kernels: output may depend on batch size/composition")
    if not env.replay_bound() and lv != "EXACT":
        reasons.append("no batch_signature: replay not bound")
    return {"ok": ok, "level": lv, "required": requirement,
            "replay_bound": env.replay_bound(), "reasons": reasons}


def local_vllm_env(model: str, model_hash: str = "", batch_size: int = 1,
                   batch_signature: str = "", seed: int = 0,
                   batch_invariant: bool = True, max_tokens: int = 0) -> Envelope:
    return Envelope(engine="vllm", engine_version="batch-invariant-ops",
                    model=model, model_hash=model_hash, quantization="awq",
                    precision="bf16", device="cuda", batch_size=batch_size,
                    batch_signature=batch_signature, seed=seed, temperature=0.0,
                    max_tokens=max_tokens, batch_invariant=batch_invariant,
                    deterministic_kernels=batch_invariant)


def remote_api_env(provider: str, model: str, batch_size: int = 0,
                   batch_signature: str = "", seed: int = 0) -> Envelope:
    return Envelope(engine=provider, engine_version="remote", model=model,
                    device="remote", batch_size=batch_size, batch_signature=batch_signature,
                    seed=seed, temperature=0.0, batch_invariant=False,
                    deterministic_kernels=False)


def describe() -> dict:
    return {
        "note": "temperature=0 is not determinism; batch invariance is the missing input",
        "levels": list(LEVELS),
        "exact_requires": ["batch_invariant=True", "deterministic_kernels=True"],
        "cache_rule": "LLM answers are keyed by envelope identity + replay key; semantic hits across envelopes are disabled",
        "source": "Thinking Machines Lab, Defeating Nondeterminism in LLM Inference, Sep 2025",
    }
