import json
from pathlib import Path
import tempfile
import unittest

from audit_e2e_measurement import audit


class TimingContractRejectionTest(unittest.TestCase):
    def rejected(self, update, message):
        provenance = {
            'environment': {'VANADIS_CPU_CLOCK': '1.0GHz',
                            'GOLEM_ARRAY_CLOCK': '1.0GHz',
                            'GOLEM_DIAGNOSTIC_ORACLE': 'none'},
            'compile_time_startup_diagnostics': 0,
        }
        update(provenance)
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / 'provenance.json').write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, message):
                audit(case)

    def test_mixed_array_clock_rejected_before_reading_cycles(self):
        self.rejected(lambda p: p['environment'].update(GOLEM_ARRAY_CLOCK='2GHz'), '1 GHz')

    def test_normalized_cycles_cannot_mix_with_native_cycles(self):
        self.rejected(lambda p: p['environment'].update(GOLEM_ATTENTION_NORMALIZATION_CLOCK='2GHz'),
                      'normalization clock')

    def test_oracle_cannot_be_a_performance_reference(self):
        self.rejected(lambda p: p['environment'].update(GOLEM_DIAGNOSTIC_ORACLE='worker_start'),
                      'oracle active')

    def test_logged_guest_cannot_be_a_clean_reference(self):
        self.rejected(lambda p: p.update(compile_time_startup_diagnostics=1), 'clean-startup')

    def test_diagnostic_dump_rejected(self):
        self.rejected(lambda p: p['environment'].update(GOLEM_ATTENTION_DIAGNOSTIC_MVM_DUMP='1'),
                      'diagnostic MVM dump')


if __name__ == '__main__':
    unittest.main()
