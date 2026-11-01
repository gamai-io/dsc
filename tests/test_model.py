import os
import tempfile
import unittest

from dsc.model import DeterministicModel


class TestModel(unittest.TestCase):
    def docs(self):
        return [
            {"id": "d1", "cat": "airless", "text": "Graco Ultra 795 airless aparat za farbanje"},
            {"id": "d2", "cat": "kompresor", "text": "vijcani kompresor 10 bar 500 litara"},
            {"id": "d3", "cat": "airless", "text": "Graco Ultra 1095 airless masina za farbanje"},
        ]

    def test_fit_search_classify(self):
        m = DeterministicModel().fit(self.docs())
        r = m.search("kompresor 10 bar", 1)
        self.assertEqual(r[0]["id"], "d2")
        self.assertEqual(m.classify("airless farbanje")["cat"], "airless")
        self.assertEqual(m.stats()["docs"], 3)

    def test_save_load(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            m = DeterministicModel().fit(self.docs())
            m.save(path)
            m2 = DeterministicModel.load(path)
            self.assertEqual(m2.search("Graco 795", 1)[0]["id"], "d1")
            self.assertEqual(m2.classify("kompresor")["cat"], "kompresor")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
