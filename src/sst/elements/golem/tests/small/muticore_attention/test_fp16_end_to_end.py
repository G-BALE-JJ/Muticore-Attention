import math
import struct
import tempfile
import unittest
from pathlib import Path

import attention_case
from verify_fused_attention_scale_output import compute_attention_blocked, verify


class FP16OutputVerificationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.q, self.k, self.v = (self.root / f"{name}.bin" for name in "qkv")
        attention_case.generate_case(256, 256, 64, self.q, self.k,
                                     v_path=self.v, heads=2, kv_heads=1, dtype="fp16")

    def write_output(self, invalid=None):
        q = attention_case._read_tensor(self.q, 2 * 256 * 64, "fp16")
        k = attention_case._read_tensor(self.k, 256 * 64, "fp16")
        v = attention_case._read_tensor(self.v, 256 * 64, "fp16")
        expected = [compute_attention_blocked(q[h * 256 * 64:(h + 1) * 256 * 64],
                    k, v, 256, 256, 64) for h in range(2)]
        for node in range(1, 5):
            values = [expected[h][row * 64 + dim]
                      for row in range((node - 1) * 64, node * 64)
                      for h in range(2) for dim in range(64)]
            if invalid is not None:
                values = [invalid] * len(values)
            attention_case._write_tensor(self.root / f"hbm_out_node{node}.bin", values, "fp16")

    def result(self):
        return verify(self.q, self.k, self.v, self.root, 0, 256, 256, 2, 1, 64, 64, "fp16")

    def test_fp16_query_major_output(self):
        self.write_output()
        result = self.result()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["checked"], 32768)
        self.assertLess(result["max_abs_error"], 2e-5)

    def test_zero_output_is_rejected(self):
        self.write_output(0.0)
        self.assertEqual(self.result()["status"], "FAIL")

    def test_nonfinite_output_is_rejected(self):
        self.write_output(math.nan)
        self.assertEqual(self.result()["status"], "FAIL")

    def test_missing_and_truncated_output_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self.result()
        self.write_output()
        (self.root / "hbm_out_node1.bin").write_bytes(b"\0\0")
        with self.assertRaises(ValueError):
            self.result()

    def test_fp16_rounds_nonrepresentable_values(self):
        path = self.root / "rounding.bin"
        attention_case._write_tensor(path, [0.1, 1e-5, -3.14], "fp16")
        self.assertEqual(path.stat().st_size, 6)
        self.assertEqual(attention_case._read_tensor(path, 3, "fp16"),
                         list(struct.unpack("<3e", path.read_bytes())))


if __name__ == "__main__":
    unittest.main()
