import os
import tempfile
import unittest

from dsc.mcp_server import handle
from dsc.unify import Unified


class TestMCP(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.u = Unified(cache_path=os.path.join(self.dir, "c.db"),
                         gtl_path=os.path.join(self.dir, "g.db"))

    def test_initialize_and_list(self):
        r = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, self.u)
        self.assertEqual(r["result"]["protocolVersion"], "2025-11-25")
        r = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, self.u)
        names = {t["name"] for t in r["result"]["tools"]}
        self.assertIn("calc", names)
        self.assertIn("verify", names)

    def test_calc(self):
        r = handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "calc", "arguments": {"expr": "(2+3)*4"}}}, self.u)
        self.assertFalse(r["result"]["isError"])
        self.assertEqual(r["result"]["structuredContent"]["result"]["value"], "20")
        self.assertEqual(r["result"]["structuredContent"]["trust"], "EXACT")

    def test_side_effect_requires_idem(self):
        r = handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "remember", "arguments": {"query": "q", "answer": "a"}}}, self.u)
        self.assertTrue(r["result"]["isError"])
        self.assertEqual(r["result"]["structuredContent"]["error"], "E_IDEM_REQUIRED")
        r2 = handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                     "params": {"name": "remember",
                                "arguments": {"query": "q", "answer": "a", "_idem_key": "k"}}}, self.u)
        self.assertFalse(r2["result"]["isError"])

    def test_unknown(self):
        r = handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                    "params": {"name": "nope", "arguments": {}}}, self.u)
        self.assertTrue(r["result"]["isError"])
        r = handle({"jsonrpc": "2.0", "id": 7, "method": "nope"}, self.u)
        self.assertEqual(r["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
