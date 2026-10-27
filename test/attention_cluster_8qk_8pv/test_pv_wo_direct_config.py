"""Reject unsupported direct consumers before expensive simulation setup."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from verification_retained_input import _inputs

HERE = Path(__file__).resolve().parent


class DirectConfigTests(unittest.TestCase):
    def invoke(self, runner, flags):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'case'
            result = subprocess.run([sys.executable, str(HERE/runner), '--artifact-root', str(root), *flags],
                                    capture_output=True, text=True)
            schedule = root/'all_gemm_reuse_window_schedule.json'
            return result, json.loads(schedule.read_text()) if schedule.exists() else None

    def test_direct_requires_handoff_in_both_runners(self):
        for runner in ('run_decoder_stack.py', 'run_all_gemm_reuse_window_baseline.py'):
            result, _ = self.invoke(runner, ['--pv-wo-direct-input'])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires --pv-wo-gm-handoff', result.stderr)

    def test_direct_rejects_unsupported_shape(self):
        result, _ = self.invoke('run_all_gemm_reuse_window_baseline.py',
                                ['--pv-wo-gm-handoff', '--pv-wo-direct-input', '--shape', 'small'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires --shape formal', result.stderr)

    def test_direct_schedule_is_explicit_and_default_off(self):
        for flags, enabled in (([], False), (['--pv-wo-gm-handoff', '--pv-wo-direct-input'], True)):
            result, schedule = self.invoke('run_all_gemm_reuse_window_baseline.py', flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(schedule['decoder_window']['pv_wo_direct_input'], enabled)

    def test_host_reference_ignores_only_equivalent_software_flags(self):
        a = dict(hidden=2048, oracle_sha256={'weights': 'fixed'}, preloads=[dict(path='/a', sha256='fixed')])
        b = dict(a, pv_wo_direct_input=True, pv_wo_gm_handoff=True)
        self.assertEqual(_inputs(a), _inputs(b))
        b['hidden'] = 1024
        self.assertNotEqual(_inputs(a), _inputs(b))


if __name__ == '__main__':
    unittest.main()
