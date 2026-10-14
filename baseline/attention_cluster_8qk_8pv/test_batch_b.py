import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from run_batch_b import MATRIX, ROOT, cases_for, environment, factorial_effects


class BatchBTests(unittest.TestCase):
    def test_complete_factorial_has_unique_combinations(self):
        self.assertEqual(len(MATRIX), 8)
        self.assertEqual(len(set(MATRIX.values())), 8)
        self.assertEqual(len(cases_for('all')), 15)

    def test_factorial_isolates_interaction(self):
        values = {name: 100 - 10*i - 8*w + 3*i*w + 4*(g-1)
                  for name, (i,w,g) in MATRIX.items()}
        effects = factorial_effects(values)
        self.assertEqual(effects['interaction_IW_cycles'], {'G1':3,'G2':3})
        self.assertEqual(effects['interaction_IWG_cycles'], 0)
        self.assertEqual(effects['main_effect_cycles']['G'], 4)
        self.assertEqual(effects['main_effect_cycles']['I'], -8.5)

    def test_partial_matrix_cannot_claim_effects(self):
        with self.assertRaisesRegex(ValueError,'eight'):
            factorial_effects({'B':100,'IW':80})

    def test_environment_does_not_inherit_extra_hardware_or_oracles(self):
        with patch.dict(os.environ, {'GOLEM_DIAGNOSTIC_ORACLE':'weight_arrival',
                                    'GOLEM_LOCAL_GM_READ_PORTS':'12', 'GOLEM_UNRELATED_TEST':'1'}):
            env = environment(Path('/tmp/b'), (0,0,1), 'none',1024,'formal')
        self.assertEqual(env['GOLEM_DIAGNOSTIC_ORACLE'],'none')
        self.assertEqual(env['GOLEM_LOCAL_GM_READ_PORTS'],'1')
        self.assertNotIn('GOLEM_UNRELATED_TEST',env)
        self.assertEqual(env['GOLEM_ARRAY_OPERAND_CONTEXT_BANKS'],'1')

    def test_worker_start_requires_frozen_origin(self):
        with self.assertRaisesRegex(ValueError,'baseline origin'):
            environment(Path('/tmp/b'),(0,0,1),'worker_start',1024,'formal')

    def test_native_oracle_zero_preserves_noneligible_service(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            code=d/'test.cpp'
            code.write_text('#include "src/sst/elements/golem/attention/diagnosticOracle.h"\n'
                'int main() { using namespace SST::Golem; '
                'if(oracleServiceCycles("gm_read",true,17)!=0) return 1; '
                'if(oracleServiceCycles("gm_read",false,17)!=17) return 2; '
                'if(oracleServiceCycles("matrix_program",true,39)!=39) return 3; return 0; }\n')
            subprocess.run(['g++','-std=c++17','-I',str(ROOT),'-I',str(ROOT/'src'),str(code),'-o',str(d/'test')],check=True)
            subprocess.run([str(d/'test')],env={**os.environ,'GOLEM_DIAGNOSTIC_ORACLE':'gm_read'},check=True)


if __name__=='__main__':
    unittest.main()
