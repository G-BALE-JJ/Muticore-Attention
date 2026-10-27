"""Catch accidental scope expansion and reject unsupported configurations."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import json

from summarize_gm_prefetch import flow_stats

HERE = Path(__file__).resolve().parent


class FfnScopedConfigurationTest(unittest.TestCase):
    def invoke(self, runner, flags):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'case'
            result = subprocess.run([sys.executable, str(HERE/runner), '--artifact-root', str(root), *flags],
                                    capture_output=True, text=True)
            schedule = root/'all_gemm_reuse_window_schedule.json'
            return result, json.loads(schedule.read_text()) if schedule.exists() else None

    def test_schedule_records_scope_and_defaults_to_shared_setting(self):
        for flags, expected in (([], 0), (['--ffn-gm-lookahead', '2'], 2)):
            result, schedule = self.invoke('run_all_gemm_reuse_window_baseline.py', flags)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(schedule['decoder_window']['ffn_gm_lookahead'], expected)

    def test_standalone_requires_fused_ffn(self):
        result, _ = self.invoke('run_decoder_stack.py', ['--ffn-gm-lookahead', '2'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires fused FFN', result.stderr)

    def test_invalid_depth_is_rejected_by_both_runners(self):
        for name in ('run_decoder_stack.py', 'run_all_gemm_reuse_window_baseline.py'):
            result, _ = self.invoke(name, ['--ffn-gm-lookahead', '3'])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('invalid choice', result.stderr)

    def test_trace_audit_rejects_scope_leak_or_missing_override(self):
        env = {'GOLEM_REUSE_WINDOW_PREFETCH_SLOTS': '4', 'GOLEM_REUSE_WINDOW_GM_LOOKAHEAD': '4',
               'GOLEM_DECODER_FFN_GM_LOOKAHEAD': '2'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);(root/'logs').mkdir();log = root/'logs/attention-sst.log'
            def line(operator, depth):
                return (f'[GEMM_WINDOW_FLOW] operator={operator} gm_lookahead={depth} a_first=0 '
                        'phase_load=1 active_cycles=1 wait_a=0 wait_b_gm=0 wait_b_dma=0 wait_array=0\n')
            log.write_text(line('wo', 4)+line('gate', 2))
            self.assertEqual(flow_stats(root, env)['window_records'], 2)
            for text in (line('wo', 2)+line('gate', 2), line('wo', 4)+line('gate', 4)):
                log.write_text(text)
                with self.assertRaisesRegex(ValueError, 'lookahead'):
                    flow_stats(root, env)


if __name__ == '__main__':
    unittest.main()
