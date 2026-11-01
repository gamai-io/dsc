import os
import tempfile
import unittest

from dsc.detector import poa_record
from dsc.sigil import (
    generate_keypair, load_private, load_public, sigil_receipt, sign_receipt, verify_sigil,
)


class TestSigil(unittest.TestCase):
    def setUp(self):
        self.poa = poa_record("answer", {"q": "x"}, [{"id": 1}], ["sql"], {"a": 1})

    def test_verify(self):
        rec = sigil_receipt(self.poa, policy={"decision": "allow", "rule_ids": ["R1"]})
        r = verify_sigil(rec, result={"a": 1}, intent={"q": "x"})
        self.assertTrue(r["ok"])
        self.assertFalse(r["signed"])

    def test_tamper(self):
        rec = sigil_receipt(self.poa)
        self.assertIn("E_RESULT_DIGEST", verify_sigil(rec, result={"a": 2})["errors"])
        rec["policy"]["decision"] = "deny"
        self.assertIn("E_POLICY_HASH", verify_sigil(rec)["errors"])

    def test_require_signature(self):
        rec = sigil_receipt(self.poa)
        self.assertIn("E_UNSIGNED", verify_sigil(rec, require_signature=True)["errors"])

    def test_ed25519(self):
        d = tempfile.mkdtemp()
        priv, pub = os.path.join(d, "k.pem"), os.path.join(d, "k.pub")
        kp = generate_keypair(priv, pub)
        rec = sigil_receipt(self.poa)
        rec = sign_receipt(rec, load_private(priv), kp["key_id"])
        r = verify_sigil(rec, result={"a": 1}, require_signature=True,
                         public_key=load_public(pub))
        self.assertTrue(r["ok"])
        self.assertTrue(r["signed"])
        rec["policy"]["decision"] = "deny"
        errs = verify_sigil(rec, public_key=load_public(pub))["errors"]
        self.assertIn("E_SIGNATURE", errs)
        self.assertIn("E_POLICY_HASH", errs)


if __name__ == "__main__":
    unittest.main()
