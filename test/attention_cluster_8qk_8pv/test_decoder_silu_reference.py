import unittest

import numpy as np

from run_decoder_stack import scalar_silu_reference


class ScalarSiluReferenceTest(unittest.TestCase):
    def test_independent_oracle_preserves_two_rounding_boundaries(self):
        x = np.linspace(-8, 8, 8192, dtype='<f2').reshape(8, 1024)
        act = scalar_silu_reference(x)
        naive = (x.astype('f4') / (np.float32(1) + np.exp(-x.astype('f4')))).astype('<f2')
        self.assertEqual(act.shape, x.shape)
        self.assertEqual(act.dtype, np.dtype('<f2'))
        self.assertLessEqual(float(np.max(np.abs(act.astype('f4') - naive.astype('f4')))), 0.004)
        up = np.full(x.shape, 0.375, dtype='<f2')
        gated = (act.astype('f4') * up.astype('f4')).astype('<f2')
        # Replay produces exact values, unlike deleting the intermediate FP16
        # rounding and directly evaluating the complete real-valued formula.
        self.assertTrue(np.array_equal(gated, (scalar_silu_reference(x).astype('f4') * up).astype('<f2')))
        self.assertTrue(np.any(gated != ((x.astype('f4') / (1 + np.exp(-x.astype('f4')))) * up).astype('<f2')))


if __name__ == '__main__':
    unittest.main()
