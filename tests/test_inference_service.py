import json
import os
import tempfile
import threading
import unittest
import urllib.request
from unittest.mock import patch

from dsc.inference_service import build_server, signed_result
from dsc.nondeterminism import verify_inference
from dsc.sigil import generate_keypair, load_private, load_public


class TestInferenceService(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.priv = os.path.join(self.dir, "infer.pem")
        self.pub = os.path.join(self.dir, "infer.pub")
        self.kp = generate_keypair(self.priv, self.pub)

    def test_signed_exact(self):
        r = signed_result("q", "a", "m", load_private(self.priv), self.kp["key_id"],
                          batch_invariant=True)
        self.assertEqual(r["trust"], "EXACT")
        self.assertTrue(verify_inference(r["receipt"], load_public(self.pub)))

    def test_unsigned_is_versioned(self):
        r = signed_result("q", "a", "m", batch_invariant=False)
        self.assertEqual(r["trust"], "VERSIONED")
        self.assertNotIn("signature", r["receipt"])

    def test_temperature_is_stochastic(self):
        r = signed_result("q", "a", "m", load_private(self.priv), self.kp["key_id"],
                          batch_invariant=True, temperature=0.7)
        self.assertEqual(r["trust"], "STOCHASTIC")
        self.assertTrue(verify_inference(r["receipt"], load_public(self.pub)))

    def test_http(self):
        srv = build_server(port=0, upstream="http://fake", model="m",
                           sign_key=self.priv, key_id=self.kp["key_id"],
                           batch_invariant=True)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            base = "http://127.0.0.1:%d" % srv.server_address[1]
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                h = json.loads(r.read().decode())
            self.assertTrue(h["ok"])
            self.assertTrue(h["signed"])
            with patch("dsc.inference_service.call_upstream",
                       return_value=("4", {"usage": {}})):
                req = urllib.request.Request(base + "/infer",
                                             data=json.dumps({"prompt": "2+2"}).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    out = json.loads(r.read().decode())
            self.assertEqual(out["output"], "4")
            self.assertEqual(out["trust"], "EXACT")
            self.assertTrue(verify_inference(out["receipt"], load_public(self.pub)))
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
