"""Reject invalid performance pairs before accepting any cycle improvement."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from compare_final_norm_lm_stream import (BINARY_KEYS, FLAG, check_pair_configuration,
                                         cycle_comparison, normalized_sources)


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.cases = [Path('/control'), Path('/candidate')]
        self.manifests = [dict(final_norm_lm_stream=False, sequence=1024),
                          dict(final_norm_lm_stream=True, sequence=1024)]
        self.provenance = []
        for i, case in enumerate(self.cases):
            self.provenance.append(dict(
                **{k: 'same-binary' for k in BINARY_KEYS},
                source_sha256={f'.all_gemm_reuse_runner_{i}/run_decoder_stack.py': 'same-source'},
                environment={FLAG: str(i), 'GOLEM_ARTIFACT_ROOT': str(case),
                             'GOLEM_GM_BANKS': '8',
                             'GOLEM_MPI_ARGS': '--cpu-set '+('8-15' if i == 0 else '0-7')}))

    def check(self):
        with patch('compare_final_norm_lm_stream.comparable_inputs',
                   side_effect=copy.deepcopy(self.manifests)):
            return check_pair_configuration(self.cases, self.provenance)

    def test_paths_and_host_placement_normalize(self):
        self.assertEqual(self.check(), {FLAG: dict(control='0', candidate='1')})

    def test_changed_hardware_rejected(self):
        self.provenance[1]['environment']['GOLEM_GM_BANKS'] = '16'
        with self.assertRaisesRegex(ValueError, 'environment'):
            self.check()

    def test_changed_source_in_temporary_runner_rejected(self):
        self.provenance[1]['source_sha256']['.all_gemm_reuse_runner_1/run_decoder_stack.py'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'source'):
            self.check()

    def test_changed_library_rejected(self):
        self.provenance[1]['sst_library_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'binaries'):
            self.check()

    def test_changed_shape_rejected(self):
        self.manifests[1]['sequence'] = 2048
        with self.assertRaisesRegex(ValueError, 'input'):
            self.check()

    def test_wrong_switch_direction_rejected(self):
        self.manifests.reverse()
        with self.assertRaisesRegex(ValueError, 'stream-off'):
            self.check()

    def test_source_normalization_does_not_hide_collision(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            normalized_sources({'.all_gemm_reuse_runner_a/x.py': 'a',
                                '.all_gemm_reuse_runner_b/x.py': 'a'})

    def test_regression_has_negative_savings(self):
        result = cycle_comparison({'full': 100}, {'full': 125})['full']
        self.assertEqual(result['saved_cycles'], -25)
        self.assertEqual(result['reduction_percent'], -25)


if __name__ == '__main__':
    unittest.main()
