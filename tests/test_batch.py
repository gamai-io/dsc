"""Tests for the deterministic batch protocol (R7) - no network, no Rust needed."""
from __future__ import annotations

import hashlib
import unittest

from dsc import batch, jcs


def _hash(req_id: int, op: str, args: dict) -> str:
    return hashlib.sha256(jcs.to_bytes({"id": req_id, "op": op, "args": args})).hexdigest()


def _fake_results(payload: bytes, *, drop: int | None = None, dupe: bool = False,
                  bad_hash: int | None = None, reverse: bool = False) -> list[dict]:
    parsed = batch.unpack(payload)
    out = []
    for item in parsed["items"]:
        if drop is not None and item["id"] == drop:
            continue
        h = _hash(item["id"], item["op"], item["args"])
        if bad_hash is not None and item["id"] == bad_hash:
            h = "0" * 64
        out.append({"id": item["id"], "input_hash": h, "ok": True})
    if dupe:
        out.append(dict(out[0]))
    if reverse:
        out = list(reversed(out))
    return out


class TestPack(unittest.TestCase):
    def setUp(self):
        self.reqs = [{"op": "echo", "args": {"x": i}} for i in range(1, 6)]

    def test_pack_is_deterministic(self):
        a = batch.pack(self.reqs, seed=7)
        b = batch.pack(self.reqs, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest())

    def test_seed_changes_payload(self):
        self.assertNotEqual(batch.pack(self.reqs, seed=1), batch.pack(self.reqs, seed=2))

    def test_header_fields(self):
        parsed = batch.unpack(batch.pack(self.reqs, seed=42))
        self.assertEqual(parsed["header"], {"v": 1, "kind": "batch", "n": 5, "seed": 42})

    def test_args_order_does_not_matter(self):
        a = batch.pack([{"op": "e", "args": {"b": 1, "a": 2}}])
        b = batch.pack([{"op": "e", "args": {"a": 2, "b": 1}}])
        self.assertEqual(a, b)

    def test_count_mismatch_detected(self):
        payload = batch.pack(self.reqs)
        broken = payload.splitlines()
        broken[0] = broken[0].replace(b'"n":5', b'"n":9')
        with self.assertRaises(ValueError):
            batch.unpack(b"\n".join(broken) + b"\n")


class TestVerify(unittest.TestCase):
    def setUp(self):
        self.reqs = [{"op": "echo", "args": {"x": i}} for i in range(1, 6)]
        self.payload = batch.pack(self.reqs, seed=3)

    def test_complete_batch_ok(self):
        res = batch.verify(self.payload, _fake_results(self.payload))
        self.assertTrue(res["ok"])
        self.assertEqual(res["seed"], 3)
        self.assertEqual([r["id"] for r in res["ordered"]], [1, 2, 3, 4, 5])

    def test_missing_detected(self):
        res = batch.verify(self.payload, _fake_results(self.payload, drop=3))
        self.assertFalse(res["ok"])
        self.assertEqual(res["missing"], [3])

    def test_duplicate_detected(self):
        res = batch.verify(self.payload, _fake_results(self.payload, dupe=True))
        self.assertFalse(res["ok"])
        self.assertTrue(res["dupes"])

    def test_hash_mismatch_detected(self):
        res = batch.verify(self.payload, _fake_results(self.payload, bad_hash=2))
        self.assertFalse(res["ok"])
        self.assertIn(2, res["bad_hash"])

    def test_arrival_order_irrelevant(self):
        res = batch.verify(self.payload, _fake_results(self.payload, reverse=True))
        self.assertTrue(res["ok"])
        self.assertEqual([r["id"] for r in res["ordered"]], [1, 2, 3, 4, 5])


class TestPercentiles(unittest.TestCase):
    def test_basic(self):
        p = batch.percentiles([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertEqual(p["n"], 5)
        self.assertLessEqual(p["p50"], p["p99"])
        self.assertEqual(p["max"], 100.0)

    def test_empty(self):
        self.assertEqual(batch.percentiles([])["n"], 0)


if __name__ == "__main__":
    unittest.main()
