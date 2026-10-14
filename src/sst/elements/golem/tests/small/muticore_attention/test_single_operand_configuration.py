"""Reject removed array storage before launching a simulation."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SingleOperandConfigurationTest(unittest.TestCase):
    def run_rejected(self, environment, args, message):
        runner = Path(__file__).with_name('run_fused_attention_scale.sh')
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, 'GOLEM_ATTENTION_CROSS_TILE_OPERAND_PIPELINE': '0',
                   'GOLEM_ARRAY_OPERAND_CONTEXT_BANKS': '1', **environment}
            result = subprocess.run([str(runner), '--query-length', '1024', '--kv-length', '1024',
                                     '--head-dim', '64', '--artifact-root', tmp, *args],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn(message, result.stderr)

    def test_second_operand_bank_is_rejected(self):
        self.run_rejected({'GOLEM_ARRAY_OPERAND_CONTEXT_BANKS': '2'}, [], 'must be 1')

    def test_cross_tile_programming_is_rejected(self):
        self.run_rejected({}, ['--cross-tile-operand-pipeline'], 'extra array operand banks were removed')

    def test_retired_weight_stationary_is_rejected(self):
        self.run_rejected({'GOLEM_PROJECTION_WEIGHT_STATIONARY': '1'}, [],
                          'GOLEM_PROJECTION_WEIGHT_STATIONARY is retired')

    def test_overlap_requires_retained_output(self):
        self.run_rejected({'GOLEM_PROJECTION_ARRAYS': '64', 'GOLEM_PROJECTION_RETAIN_OUTPUT': '0',
                           'GOLEM_PROJECTION_ARRAY_GROUPS': '2'}, [],
                          'Projection overlap requires 64 arrays, retained output')

    def test_invalid_group_count_is_rejected(self):
        self.run_rejected({'GOLEM_PROJECTION_ARRAY_GROUPS': '3'}, [],
                          'GOLEM_PROJECTION_ARRAY_GROUPS must be 1 or 2')


if __name__ == '__main__':
    unittest.main()
