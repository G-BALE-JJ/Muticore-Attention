"""Audit actual Q/K/V panels even when raw projection output DMA is elided."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from run_all_gemm_reuse_window_baseline import projection_panel_audit
from run_decoder_layer import linear_reference
from run_decoder_stack import rms


class PanelAuditTest(unittest.TestCase):
    def case(self, root):
        (root/'hbm').mkdir()
        shared=dict(q_panel=0x10000,k_panel=0x12000,v_panel=0x14000)
        manifest=dict(layers=1,num_query_heads=1,num_kv_heads=1,sequence=256,hidden=64,
                      layout=dict(layers=[dict(input=0x1000)],shared=shared))
        (root/'decoder_stack_inputs.json').write_text(json.dumps(manifest))
        (root/'decoder_stack_provenance.json').write_text(json.dumps(dict(
            environment=dict(GOLEM_FUSION_PROJECTION_ROPE='0'))))
        gamma=np.ones(64,dtype='<f2');gamma.tofile(root/'layer1_gamma1.bin')
        weight=np.concatenate([np.eye(64,dtype='<f2')*s for s in (1,.5,.25)])
        np.save(root/'layer1_qkv.npy',weight)
        x=np.random.default_rng(7).normal(size=(256,64)).astype('<f2')
        expected=linear_reference(rms(x,gamma),weight).reshape(4,64,3,64)
        for node in range(1,5):
            data=bytearray(0x16000)
            data[0x1000:0x3000]=x[(node-1)*64:node*64].tobytes()
            for index,name in enumerate(('q','k','v')):
                panel=expected[node-1,:,index]
                if name=='v':panel=panel.T.copy()
                offset=shared[name+'_panel'];data[offset:offset+8192]=panel.tobytes()
            (root/'hbm'/f'hbm_out_node{node}.bin').write_bytes(data)

    def test_materialized_panels_and_transposed_v(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.case(root)
            self.assertEqual(projection_panel_audit(root)['status'],'PASS')

    def test_q_corruption_fails_when_raw_outputs_are_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.case(root)
            with (root/'hbm/hbm_out_node4.bin').open('r+b') as stream:
                stream.seek(0x10000);stream.write(np.array([100],dtype='<f2').tobytes())
            result=projection_panel_audit(root)
            self.assertEqual(result['status'],'FAIL')
            self.assertEqual(result['panels']['q']['mismatches'],1)
            self.assertEqual(result['panels']['k']['status'],'PASS')
            self.assertEqual(result['panels']['v']['status'],'PASS')

if __name__=='__main__':unittest.main()
