import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
class PublicSimModeTests(unittest.TestCase):
    def command(self,stage,*flags,env=None):
        p=subprocess.run([sys.executable,str(ROOT/'test/run.py'),stage,'--show-config',*flags],
                         env={**os.environ,'GOLEM_SIM_MODE':'full-functional',**(env or {})},capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        return json.loads(p.stdout)
    def test_all_stages_accept_both_argument_positions(self):
        for stage in ('GEMM','Attention','GQA','prefill'):
            for flags in [('--sim-mode','full-timing'),('--','--sim-mode=full-timing')]:
                d=self.command(stage,*flags)
                self.assertEqual(d['sim_mode'],'full-timing')
                self.assertEqual(d['command'][-2:],['--sim-mode','full-timing'])
    def test_gemm_removes_only_default_numerical_check(self):
        self.assertIn('--verify-c',self.command('GEMM')['command'])
        self.assertNotIn('--verify-c',self.command('GEMM','--sim-mode','full-timing')['command'])
        self.assertIn('--verify-c',self.command('GEMM','--sim-mode','full-timing','--','--verify-c')['command'])
    def test_native_llm_operator_is_explicit(self):
        d=self.command('prefill','--native-operator','qkv','--sim-mode','full-timing')
        self.assertEqual(d['native_operator'],'qkv')
        self.assertTrue(d['command'][1].endswith('run_native_llm_operator.py'))
        self.assertEqual(d['sim_mode'],'full-timing')
    def test_environment_and_cli_precedence(self):
        for stage in ('GEMM','Attention','GQA','prefill'):
            self.assertEqual(self.command(stage,env={'GOLEM_SIM_MODE':'full-timing'})['sim_mode'],'full-timing')
            self.assertEqual(self.command(stage,'--sim-mode','full-functional',env={'GOLEM_SIM_MODE':'full-timing'})['sim_mode'],'full-functional')
if __name__=='__main__':unittest.main()
