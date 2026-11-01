import os
import tempfile
import unittest

from dsc.unify import Unified, safe_calc


class TestUnify(unittest.TestCase):
    def test_calc(self):
        self.assertEqual(safe_calc({"expr": "(2+3)*4"})["value"], "20")
        with self.assertRaises(ValueError):
            safe_calc({"expr": "__import__('os').system('id')"})
        with self.assertRaises(ValueError):
            safe_calc({"expr": "1/0"})

    def test_registry(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(path)
        u = Unified(cache_path=path + ".cache", gtl_path=path)
        try:
            names = {c["name"] for c in u.capabilities()}
            self.assertIn("calc", names)
            self.assertIn("verify", names)
            r = u.call("calc", {"expr": "2**10"})
            self.assertEqual(r["result"]["value"], "1024")
            self.assertEqual(r["trust"], "EXACT")
            self.assertFalse(u.call("remember", {"query": "q", "answer": "a"})["ok"])
        finally:
            for p in (path, path + "-wal", path + "-shm", path + ".cache",
                      path + ".cache-wal", path + ".cache-shm"):
                try:
                    os.unlink(p)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
