"""Reject GM lifetime conflicts before creating a simulation workload."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNNER=Path(__file__).with_name('run_all_gemm_reuse_window_baseline.py')


class InputResidencyConfigTest(unittest.TestCase):
    def run_case(self, flags):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'case'
            result=subprocess.run([sys.executable,str(RUNNER),'--artifact-root',str(root),*flags],
                                  capture_output=True,text=True)
            schedule=root/'all_gemm_reuse_window_schedule.json'
            return result,json.loads(schedule.read_text()) if schedule.exists() else None

    def test_requires_layer_residency(self):
        result,_=self.run_case(['--gm-stage-chain','--input-residency'])
        self.assertNotEqual(result.returncode,0)
        self.assertIn('requires --layer-residency',result.stderr)

    def test_output_elision_requires_resident_residual_input(self):
        result,_=self.run_case(['--gm-stage-chain','--layer-residency','--elide-intermediate-output'])
        self.assertNotEqual(result.returncode,0)
        self.assertIn('requires --input-residency',result.stderr)

    def test_rejects_projection_c_spills_into_retained_input(self):
        result,_=self.run_case(['--gm-stage-chain','--layer-residency','--input-residency',
                                '--shape','formal','--window-k-tiles','8'])
        self.assertNotEqual(result.returncode,0)
        self.assertIn('full projection K window',result.stderr)

    def test_rejects_concurrent_scratch_owners(self):
        for flags in (['--ffn-block-pipeline','1'],['--weight-cohort','2']):
            result,_=self.run_case(['--gm-stage-chain','--layer-residency','--input-residency',*flags])
            self.assertNotEqual(result.returncode,0)
            self.assertIn('requires private weights',result.stderr)

    def test_small_shape_full_k_window_is_valid(self):
        result,schedule=self.run_case(['--gm-stage-chain','--layer-residency','--input-residency',
                                       '--shape','small','--window-k-tiles','4'])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(schedule['decoder_window']['input_residency'])

    def test_default_remains_opt_in(self):
        result,schedule=self.run_case([])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(schedule['decoder_window']['input_residency'])
        self.assertFalse(schedule['decoder_window']['elide_intermediate_output'])

    def test_output_elision_requires_layer_residency(self):
        result,_=self.run_case(['--elide-intermediate-output'])
        self.assertNotEqual(result.returncode,0)
        self.assertIn('requires --layer-residency',result.stderr)

    def test_output_elision_with_gm_chain(self):
        result,schedule=self.run_case(['--gm-stage-chain','--layer-residency','--input-residency',
                                      '--elide-intermediate-output'])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(schedule['decoder_window']['elide_intermediate_output'])


if __name__=='__main__':unittest.main()
