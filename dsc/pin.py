"""PIN - the determinism contract for model calls (R4).

A remote LLM is not reproducible on its own: the same prompt gives different
bytes under a different batch composition. We cannot fix that from here, so we
do the only honest thing - record everything that influences the output and
refuse to call a run "reproducible" without it.

PIN = {model, version, quant, seed, temperature, batch, extra}
All values are canonicalisable (strings/ints) so the PIN can live inside a
signed receipt (floats must be passed as strings, e.g. temperature="0.0").
"""
from __future__ import annotations

from typing import Any

REQUIRED = ("model", "version", "seed", "temperature")


class PinError(ValueError):
    pass


def _check_scalar(name: str, value: Any) -> str:
    if isinstance(value, float):
        raise PinError("%s mora biti string (float nije dozvoljen u potpisu): %r" % (name, value))
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise PinError("%s mora biti string ili int: %r" % (name, value))
    return str(value)


def make(model: str, version: str, seed: int, temperature: Any = "0.0",
         *, quant: str | None = None, batch: int | None = None,
         **extra: Any) -> dict:
    """Build a PIN. `temperature` must be a string ("0.0") or int, never a float."""
    if not model or not version:
        raise PinError("model i version su obavezni")
    pin: dict[str, Any] = {
        "model": _check_scalar("model", model),
        "version": _check_scalar("version", version),
        "seed": int(seed),
        "temperature": _check_scalar("temperature", temperature),
    }
    if quant:
        pin["quant"] = _check_scalar("quant", quant)
    if batch is not None:
        pin["batch"] = int(batch)
    for key, value in extra.items():
        if isinstance(value, float):
            raise PinError("extra.%s ne smije biti float" % key)
        if isinstance(value, (str, int, bool)) or value is None:
            pin[key] = value
        else:
            raise PinError("extra.%s: nepodrzan tip" % key)
    return pin


def validate(pin: dict | None) -> dict:
    if not isinstance(pin, dict):
        return {"ok": False, "missing": list(REQUIRED), "invalid": ["not_a_dict"]}
    missing = [k for k in REQUIRED if pin.get(k) in (None, "")]
    invalid: list[str] = []
    if "seed" in pin and not isinstance(pin["seed"], int):
        invalid.append("seed")
    if "temperature" in pin and isinstance(pin["temperature"], float):
        invalid.append("temperature")
    return {"ok": not missing and not invalid, "missing": missing, "invalid": invalid}


def same(a: dict | None, b: dict | None) -> bool:
    """True when two PINs describe the same inferential conditions."""
    if not validate(a)["ok"] or not validate(b)["ok"]:
        return False
    keys = ("model", "version", "quant", "seed", "temperature", "batch")
    return all(a.get(k) == b.get(k) for k in keys)


def describe() -> dict:
    return {
        "required": list(REQUIRED),
        "optional": ["quant", "batch", "extra"],
        "rule": "model bez PIN-a nije reproducibilan; temperature kao string",
    }
