import json
import os
import tempfile
import unittest

from dsc.label_tool import build_queue, load_labels, save_label, to_golden


class TestLabelTool(unittest.TestCase):
    def data(self):
        return {
            "docs": [
                {"id": "d1", "cat": "airless", "text": "Graco Ultra 795 airless aparat za farbanje"},
                {"id": "d2", "cat": "airless", "text": "Graco Ultra 1095 veca airless masina"},
                {"id": "d3", "cat": "kompresor", "text": "vijcani kompresor 10 bar"},
            ],
            "queries": [{"q": "Graco Ultra 795", "gold": ["d1"], "cat": "exact", "origin": True}],
        }

    def test_build(self):
        q = build_queue(self.data(), top_k=2, max_queries=1)
        self.assertEqual(len(q["items"]), 1)
        ids = [d["id"] for d in q["items"][0]["docs"]]
        self.assertIn("d1", ids)
        self.assertEqual(q["items"][0]["docs"][ids.index("d1")]["rel"], 2)

    def test_labels_export(self):
        q = build_queue(self.data(), top_k=2, max_queries=1)
        labels = {q["items"][0]["q"]: {"d1": 2, "d2": 1}}
        g = to_golden(q, labels)
        self.assertEqual(g["queries"][0]["graded"], {"d1": 2, "d2": 1})
        self.assertEqual(g["queries"][0]["gold"], ["d1"])
        self.assertTrue(g["queries"][0]["labeled"])

    def test_save_load(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        os.unlink(path)
        try:
            save_label(path, "q", {"d1": 2})
            self.assertEqual(load_labels(path)["q"]["d1"], 2)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
