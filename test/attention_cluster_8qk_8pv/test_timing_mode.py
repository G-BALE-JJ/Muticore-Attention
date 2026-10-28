import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from timing_mode import verify_timing_backend

ROOT = Path(__file__).resolve().parents[2]


class TimingModeTest(unittest.TestCase):
    def verify_modes(self, modes):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'run.log'
            log.write_text(''.join(
                f'[ARRAY_NUMERICAL_DATA] core={core} functional={functional} timing_fast_path={fast} materialize={materialize}\n'
                for core, functional, fast, materialize in modes))
            return verify_timing_backend(log)

    def test_timing_with_and_without_materialization(self):
        for fast in (0, 1):
            result = self.verify_modes([(core, 0, fast, 1-fast) for core in range(4, 20)])
            self.assertFalse(result['array_numerical_macs'])
            self.assertEqual(result['numerics']['status'], 'SKIPPED')

    def test_reject_missing_functional_or_conflicting_workers(self):
        timing = [(core, 0, 1, 0) for core in range(4, 20)]
        for modes in ([], timing[:-1], timing + [(4, 1, 1, 1)],
                      [(core, 1, 1, 1) for core in range(4, 20)],
                      [(core, 0, 1, 1) for core in range(4, 20)]):
            with self.assertRaises(ValueError):
                self.verify_modes(modes)

    def test_attention_cli_mode_precedence_and_invalid_mode(self):
        env = dict(os.environ, GOLEM_SIM_MODE='full-timing')
        command = ['bash', str(ROOT/'scripts/test_flash_attention.sh'), '--show-config']
        self.assertIn('GOLEM_SIM_MODE=full-timing', subprocess.check_output(command, env=env, text=True))
        self.assertIn('GOLEM_SIM_MODE=full-functional', subprocess.check_output(
            command + ['--sim-mode', 'full-functional'], env=env, text=True))
        result = subprocess.run(command + ['--sim-mode', 'invalid'], env=env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
