"""Configuration and attribution boundaries for the opt-in software tail."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from summarize_tail_notification import configuration_pair, fixed_hardware_across_builds
from compare_final_norm_lm_stream import BINARY_KEYS

HERE = Path(__file__).resolve().parent


class TailConfigTest(unittest.TestCase):
    def test_cross_build_comparison_preserves_hardware_and_inputs(self):
        cases=[Path('/default'),Path('/experimental')]
        manifests=[dict(sequence=1024,tail_schedule_experiment=False),
                   dict(sequence=1024,tail_schedule_experiment=True,
                        lm_handoff_protocol='producer_remote_store_v1')]
        provenance=[dict(**{k:'same' for k in BINARY_KEYS},
                         source_sha256={'src/sst/elements/golem/globalmemory/globalmemory.h':'same'},
                         environment={'GOLEM_GM_BANKS':'8',
                                      'GOLEM_ATTENTION_GUEST_EXECUTABLE':str(i)}) for i in range(2)]
        provenance[1]['guest_sha256']='experimental'
        def check():
            with patch('summarize_tail_notification.comparable_inputs',side_effect=copy.deepcopy(manifests)):
                return fixed_hardware_across_builds(cases,provenance)
        self.assertEqual(check()['status'],'PASS')
        provenance[1]['sst_library_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'library'):check()
        provenance[1]['sst_library_sha256']='same'
        provenance[1]['environment']['GOLEM_GM_BANKS']='16'
        with self.assertRaisesRegex(ValueError,'environment'):check()
        provenance[1]['environment']['GOLEM_GM_BANKS']='8'
        hardware='src/sst/elements/golem/globalmemory/globalmemory.h'
        provenance[1]['source_sha256'][hardware]='changed'
        with self.assertRaisesRegex(ValueError,'hardware sources'):check()
        provenance[1]['source_sha256'][hardware]='same'
        manifests[1]['sequence']=512
        with self.assertRaisesRegex(ValueError,'inputs'):check()

    def invoke(self, flags):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'case'
            p = subprocess.run([sys.executable,str(HERE/'run_all_gemm_reuse_window_baseline.py'),
                                '--artifact-root',str(root),*flags],capture_output=True,text=True)
            schedule = root/'all_gemm_reuse_window_schedule.json'
            return p,json.loads(schedule.read_text()) if schedule.exists() else None

    def test_default_excludes_experimental_guest(self):
        p,s = self.invoke([])
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertFalse(s['decoder_window']['tail_schedule_experiment'])

    def test_priority_requires_producer_notification(self):
        p,_ = self.invoke(['--final-norm-producer-first'])
        self.assertNotEqual(p.returncode,0)
        self.assertIn('requires --final-norm-lm-stream',p.stderr)

    def test_stream_selects_isolated_guest(self):
        p,s = self.invoke(['--gm-stage-chain','--layer-residency','--input-residency',
                           '--eager-final-norm','--final-norm-lm-stream','--final-norm-producer-first'])
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertTrue(s['decoder_window']['tail_schedule_experiment'])
        self.assertTrue(s['decoder_window']['final_norm_producer_first'])

    def test_hardware_and_content_differences_cannot_be_normalized(self):
        cases=[Path('/control'),Path('/candidate')]
        manifests=[dict(sequence=1024,final_norm_lm_stream=False),
                   dict(sequence=1024,final_norm_lm_stream=True)]
        provenance=[dict(**{k:'binary' for k in BINARY_KEYS},source_sha256={'runtime.cpp':'source'},
                         environment={'GOLEM_GM_BANKS':'8','GOLEM_DECODER_FINAL_NORM_LM_STREAM':str(i)})
                    for i in range(2)]
        def check():
            with patch('summarize_tail_notification.comparable_inputs',side_effect=copy.deepcopy(manifests)):
                return configuration_pair(cases,provenance)
        self.assertEqual(set(check()),{'GOLEM_DECODER_FINAL_NORM_LM_STREAM'})
        provenance[1]['environment']['GOLEM_GM_BANKS']='16'
        with self.assertRaisesRegex(ValueError,'environment'):check()
        provenance[1]['environment']['GOLEM_GM_BANKS']='8'
        provenance[1]['source_sha256']['runtime.cpp']='changed'
        with self.assertRaisesRegex(ValueError,'sources'):check()


if __name__ == '__main__':
    unittest.main()
