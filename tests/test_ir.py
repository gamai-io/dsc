"""Tests for the GBIN IR container (R1): layout, round-trip, tamper, versions."""
from __future__ import annotations

import struct
import unittest

from dsc import ir


PROG = 'neka x = 7\nneka y = [x * 6]\nispisi y\n'


class TestLayout(unittest.TestCase):
    def test_header_fields(self):
        blob = ir.pack_program(PROG)
        magic, version, kind, flags, reserved, length = struct.unpack(">4sBBBB I", blob[:12])
        self.assertEqual(magic, b"GBIN")
        self.assertEqual(version, 1)
        self.assertEqual(kind, ir.KIND_PROGRAM)
        self.assertEqual(flags, 0)
        self.assertEqual(reserved, 0)
        self.assertEqual(len(blob), 12 + length + 32)

    def test_magic_is_first(self):
        self.assertTrue(ir.pack_program(PROG).startswith(b"GBIN"))


class TestProgramRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        blob = ir.pack_program(PROG, meta={"ime": "test"})
        out = ir.unpack_program(blob)
        self.assertEqual(out["source"], PROG)
        self.assertEqual(out["meta"], {"ime": "test"})
        self.assertEqual(len(out["cid"]), 64)

    def test_cid_stable_and_metadata_sensitive(self):
        a = ir.cid(ir.pack_program(PROG))
        b = ir.cid(ir.pack_program(PROG))
        c = ir.cid(ir.pack_program(PROG, meta={"v": 2}))
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_unicode_program(self):
        src = 'ispisi "čćžšđ — 42"\n'
        out = ir.unpack_program(ir.pack_program(src))
        self.assertEqual(out["source"], src)

    def test_empty_source_rejected(self):
        with self.assertRaises(ir.IRError) as ctx:
            ir.pack_program("   ")
        self.assertEqual(ctx.exception.code, "E_EMPTY_PROGRAM")


class TestValueRoundTrip(unittest.TestCase):
    def test_value_round_trip(self):
        val = {"b": [1, 2, {"c": None}], "a": "đ"}
        self.assertEqual(ir.unpack_value(ir.pack_value(val)), val)

    def test_float_value_rejected(self):
        with self.assertRaises(Exception) as ctx:
            ir.pack_value({"x": 1.5})
        self.assertIn("float", str(ctx.exception))

    def test_kind_mismatch(self):
        blob = ir.pack_value({"a": 1})
        with self.assertRaises(ir.IRError) as ctx:
            ir.unpack_program(blob)
        self.assertEqual(ctx.exception.code, "E_KIND")


class TestIntegrity(unittest.TestCase):
    def test_tampered_payload_detected(self):
        blob = bytearray(ir.pack_program(PROG))
        blob[20] ^= 0xFF
        with self.assertRaises(ir.IRError) as ctx:
            ir.unpack_program(bytes(blob))
        self.assertIn(ctx.exception.code, ("E_DIGEST", "E_PAYLOAD"))

    def test_wrong_magic(self):
        blob = b"XXXX" + ir.pack_program(PROG)[4:]
        with self.assertRaises(ir.IRError) as ctx:
            ir.unpack_program(blob)
        self.assertEqual(ctx.exception.code, "E_MAGIC")

    def test_wrong_version(self):
        blob = bytearray(ir.pack_program(PROG))
        blob[4] = 9
        with self.assertRaises(ir.IRError) as ctx:
            ir.unpack_program(bytes(blob))
        self.assertEqual(ctx.exception.code, "E_VERSION")

    def test_truncated(self):
        with self.assertRaises(ir.IRError) as ctx:
            ir.unpack_program(b"GBIN\x01\x01")
        self.assertEqual(ctx.exception.code, "E_TRUNCATED")

    def test_length_mismatch(self):
        blob = bytearray(ir.pack_program(PROG))
        struct.pack_into(">I", blob, 8, 9999)
        with self.assertRaises(ir.IRError) as ctx:
            ir.unpack_program(bytes(blob))
        self.assertEqual(ctx.exception.code, "E_LENGTH")


class TestInfo(unittest.TestCase):
    def test_info_does_not_leak_source(self):
        inf = ir.info(ir.pack_program(PROG))
        self.assertEqual(inf["magic"], "GBIN")
        self.assertEqual(inf["kind"], "program")
        self.assertEqual(len(inf["cid"]), 64)
        self.assertNotIn("source", inf)
        self.assertNotIn("neka", str(inf))


if __name__ == "__main__":
    unittest.main()
