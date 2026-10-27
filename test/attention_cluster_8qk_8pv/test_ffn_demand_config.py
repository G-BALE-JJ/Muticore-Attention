"""FFN demand priority must be opt-in, scoped, and present in actual traces."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from summarize_gm_prefetch import flow_stats

HERE=Path(__file__).resolve().parent


class DemandConfigTest(unittest.TestCase):
    def test_configuration_and_default(self):
        for flags,expected in (([],False),(['--ffn-demand-order'],True)):
            with tempfile.TemporaryDirectory() as directory:
                p=subprocess.run([sys.executable,str(HERE/'run_all_gemm_reuse_window_baseline.py'),
                                  '--artifact-root',directory,*flags],capture_output=True,text=True)
                self.assertEqual(p.returncode,0,p.stderr)
                s=json.loads((Path(directory)/'all_gemm_reuse_window_schedule.json').read_text())
                self.assertEqual(s['decoder_window']['ffn_demand_order'],expected)

    def test_standalone_requires_fused_ffn(self):
        with tempfile.TemporaryDirectory() as directory:
            p=subprocess.run([sys.executable,str(HERE/'run_decoder_stack.py'),
                              '--artifact-root',directory,'--ffn-demand-order'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('requires fused FFN',p.stderr)

    def test_trace_scope_and_actual_bypass(self):
        env=dict(GOLEM_REUSE_WINDOW_PREFETCH_SLOTS='4',GOLEM_REUSE_WINDOW_GM_LOOKAHEAD='4',
                 GOLEM_DECODER_FFN_DEMAND_ORDER='1')
        def line(operator,enabled,bypass=0):
            return (f'[GEMM_WINDOW_FLOW] operator={operator} demand_order={enabled} '
                    f'b_head_bypass_admissions={bypass} gm_lookahead=4 a_first=0 '
                    'phase_load=1 active_cycles=1 wait_a=0 wait_b_gm=0 wait_b_dma=0 wait_array=0\n')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'logs').mkdir();p=root/'logs/attention-sst.log'
            p.write_text(line('linear',0)+line('gate',1))
            self.assertEqual(flow_stats(root,env)['window_records'],2)
            for text in (line('linear',1)+line('gate',1),line('gate',0),line('gate',1,1)):
                p.write_text(text)
                with self.assertRaisesRegex(ValueError,'demand|bypass'):flow_stats(root,env)


if __name__=='__main__':unittest.main()
