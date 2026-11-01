import subprocess
import sys
import unittest

from dsc.normalize import canonical
from dsc.fingerprints import (
    hamming, jaccard_est, jaccard_true, minhash128, sha256_hex, simhash64,
)


class TestFingerprints(unittest.TestCase):
    def test_simhash_stable(self):
        a = simhash64("Graco Ultra 795 farbanje")
        self.assertEqual(a, simhash64("Graco Ultra 795 farbanje"))
        self.assertGreater(hamming(a, simhash64("kompresor 10 bar")), 10)

    def test_hamming_symmetry(self):
        a, b = simhash64("kupci u Sarajevu"), simhash64("kupci Sarajevo")
        self.assertEqual(hamming(a, b), hamming(b, a))

    def test_minhash_estimate(self):
        a = "kupac ACME Hadzici Sarajevo dobavljač"
        b = "kupac ACME Hadzici Sarajevo"
        est = jaccard_est(minhash128(a), minhash128(b))
        true = jaccard_true(a, b)
        self.assertLess(abs(est - true), 0.15)

    def test_sha_canonical(self):
        self.assertEqual(sha256_hex(canonical("cijena 2.400,00 KM")),
                         sha256_hex(canonical("cijena 2400 KM")))

    def test_cross_process(self):
        code = ("import sys;sys.path.insert(0,'.');"
                "from dsc.fingerprints import simhash64;"
                "print(simhash64('Graco Ultra 795 farbanje'))")
        out = [subprocess.check_output([sys.executable, "-c", code], cwd=".").decode().strip()
               for _ in range(2)]
        self.assertEqual(out[0], out[1])


if __name__ == "__main__":
    unittest.main()
