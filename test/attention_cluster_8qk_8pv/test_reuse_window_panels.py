"""Audit actual Q/K/V panels even when raw projection output DMA is elided."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from run_all_gemm_reuse_window_baseline import projection_panel_audit
from run_decoder_layer import linear_reference
from run_decoder_stack import rms
from verification_retained_input import last_layer_input


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

    def retained_case(self, directory):
        parent=Path(directory);reference=parent/'reference';case=parent/'candidate'/'probe'
        reference.mkdir();case.mkdir(parents=True)
        for root in (reference,case):
            self.case(root)
            manifest=json.loads((root/'decoder_stack_inputs.json').read_text())
            manifest['layers']=2;manifest['layout']['layers']*=2
            manifest['elide_intermediate_output']=root==case
            (root/'decoder_stack_inputs.json').write_text(json.dumps(manifest))
            for suffix in ('gamma1.bin','qkv.npy'):
                (root/f'layer2_{suffix}').write_bytes((root/f'layer1_{suffix}').read_bytes())
        (reference/'decoder_stack_result.json').write_text(json.dumps(dict(status='PASS')))
        (case.parent/'verification_reference.json').write_text(json.dumps(dict(
            result=str(reference/'decoder_stack_result.json'))))
        for node in range(1,5):
            with (case/'hbm'/f'hbm_out_node{node}.bin').open('r+b') as stream:
                stream.seek(0x1000);stream.write(bytes(8192))
        return case,reference,json.loads((case/'decoder_stack_inputs.json').read_text())

    def test_retained_input_ignores_stale_hbm_without_weakening_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            case,_,_=self.retained_case(directory)
            result=projection_panel_audit(case)
            self.assertEqual(result['status'],'PASS')
            self.assertTrue(result['input_reference']['candidate_hbm_input_intentionally_absent'])
            with (case/'hbm/hbm_out_node4.bin').open('r+b') as stream:
                stream.seek(0x10000);stream.write(np.array([100],dtype='<f2').tobytes())
            self.assertEqual(projection_panel_audit(case)['panels']['q']['mismatches'],1)

    def test_retained_input_rejects_unverified_or_different_counterpart(self):
        for mutation in ('failed','elided','different'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                case,reference,m=self.retained_case(directory)
                p=reference/'decoder_stack_inputs.json';other=json.loads(p.read_text())
                if mutation=='failed':
                    (reference/'decoder_stack_result.json').write_text(json.dumps(dict(status='FAIL')))
                elif mutation=='elided':other['elide_intermediate_output']=True
                else:other['seed']=123
                p.write_text(json.dumps(other))
                with self.assertRaises(ValueError):last_layer_input(case,m)

    def test_retained_input_requires_explicit_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            case,_,m=self.retained_case(directory)
            (case.parent/'verification_reference.json').unlink()
            with self.assertRaises(ValueError):last_layer_input(case,m)

if __name__=='__main__':unittest.main()
