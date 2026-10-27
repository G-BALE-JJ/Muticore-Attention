"""Reject incomplete fusion payloads and unsafe eager worker dependencies."""
import json
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from verify_add_norm_fusion import audit
from verify_software_stage_chain import audit as chain_audit
from verification_retained_input import last_layer_input
import test_decoder_stack as norm_fixture
import test_reuse_window_panels as panels_fixture
import test_software_stage_chain as chain_fixture
import test_input_residency_config as config_fixture


class FusionPayloadTest(unittest.TestCase):
    def test_lm_tail_coverage_counts_only_workers_with_real_tiles(self):
        from run_all_gemm_reuse_window_baseline import coverage
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir();(root/'logs/attention-sst.log').write_text('')
            (root/'decoder_stack_inputs.json').write_text('{}')
            (root/'decoder_stack_provenance.json').write_text(json.dumps(dict(environment={
                'GOLEM_REUSE_WINDOW_K_TILES':'32','GOLEM_REUSE_WINDOW_ARRAY_GROUPS':'1'})))
            shape=dict(hidden=256,intermediate=1024,layers=2,padded_S=1024)
            for vocab,expected in ((9,1),(8201,17),(16384,32),(128256,256)):
                result=coverage(root,dict(shape=shape,lm_head=dict(vocab=vocab)))
                self.assertEqual(result['expected_window_job_counts']['lm_head'],expected)
                self.assertEqual(result['status'],'FAIL') # Missing actual work is still rejected.

    def test_both_software_switches_reach_the_simulated_guest(self):
        from run_decoder_stack import ROOT
        source=ROOT/'src/sst/elements/golem/tests/architecture/ncores_selfcom_dma_ctrl.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign) and
                  any(isinstance(t,ast.Name) and t.id=='process_env_keys' for t in n.targets))
        keys=ast.literal_eval(node.value)
        for flag in ('GOLEM_DECODER_ADD_NORM_FUSION','GOLEM_DECODER_EAGER_FINAL_NORM'):
            self.assertEqual(keys.count(flag),1)

    def test_fusion_switches_require_real_gm_owners(self):
        fixture=config_fixture.InputResidencyConfigTest()
        for flags in (['--add-norm-fusion'],['--eager-final-norm']):
            result,_=fixture.run_case(flags);self.assertNotEqual(result.returncode,0)
        result,schedule=fixture.run_case(['--gm-stage-chain','--layer-residency','--input-residency',
                                        '--add-norm-fusion','--eager-final-norm'])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(schedule['decoder_window']['add_norm_fusion'])
        self.assertTrue(schedule['decoder_window']['eager_final_norm'])

    def fixture(self, root):
        root=Path(root);(root/'logs').mkdir();size=32768;core=4
        log=root/'logs/attention-sst.log';lines=[];events=[]
        for chunk in range(2):
            begin=100+chunk*100
            for op,a,b in ((3,begin,begin+10),(2,begin+20,begin+40)):
                lines.append(f'[DECODER_ADD_NORM_SFU] core=4 epoch=1 op={op} elements=8192 submit_tick={a} done_tick={b}')
            for k in range(4):
                events.append(dict(resource='local_write',core=core,address=core*0x200000+0x10000+(k*64+chunk*32)*128,
                    bytes=4096,submit_tick=begin+41,start_tick=begin+42,done_tick=begin+50,ok=1))
            events.append(dict(resource='local_read',core=core,address=core*0x200000+0xb0000+chunk*16384,
                bytes=16384,submit_tick=begin,start_tick=begin,done_tick=begin+1,ok=1))
        log.write_text('\n'.join(lines)+'\n')
        resource=log.with_suffix('.resources.jsonl')
        resource.write_text('\n'.join('[RESOURCE_INTERVAL] '+' '.join(f'{k}={v}' for k,v in e.items()) for e in events)+'\n')
        return root,{4:[dict(core=4,epoch=1,stage=2,start_tick=90,tick=300)]},dict(hidden=256,add_norm_fusion=True),log,resource

    def test_actual_payload_and_bounded_sfu_work(self):
        with tempfile.TemporaryDirectory() as d:
            root,jobs,m,_,_=self.fixture(d);result=audit(root,jobs,m)
            self.assertEqual(result['saved_gm_read_bytes'],32768)
            self.assertEqual(result['norm_write_requests'],8)

    def test_missing_duplicate_early_or_failed_work_rejected(self):
        for mutation in ('read_duplicate','write_missing','write_failed','early_write','sfu_missing','sfu_unbounded'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                root,jobs,m,log,resource=self.fixture(d)
                lines=resource.read_text().splitlines(True)
                if mutation=='read_duplicate':lines.append(next(l for l in lines if 'local_read ' in l))
                elif mutation=='write_missing':lines.pop(0)
                elif mutation=='write_failed':lines[0]=lines[0].replace('ok=1','ok=0')
                elif mutation=='early_write':lines[0]=lines[0].replace('submit_tick=141','submit_tick=139')
                elif mutation=='sfu_missing':log.write_text('\n'.join(log.read_text().splitlines()[1:])+'\n')
                else:log.write_text(log.read_text().replace('elements=8192','elements=8193',1))
                resource.write_text(''.join(lines))
                with self.assertRaises(ValueError):audit(root,jobs,m)

    def test_software_fusion_input_oracle_keeps_weight_and_input_identity(self):
        with tempfile.TemporaryDirectory() as d:
            case,reference,m=panels_fixture.PanelAuditTest().retained_case(d)
            m.update(add_norm_fusion=True,eager_final_norm=True)
            _,proof=last_layer_input(case,m)
            self.assertEqual(len(proof['equivalent_software_schedules']),2)
            m['preloads']=[dict(sha256='changed')]
            with self.assertRaises(ValueError):last_layer_input(case,m)

    def test_fused_stage_chain_has_one_combined_job_and_same_hbm_savings(self):
        with tempfile.TemporaryDirectory() as d:
            case,m,jobs,log=chain_fixture.SoftwareStageChainAuditTest().fixture(d)
            m.update(input_residency=True,add_norm_fusion=True)
            p=case/'decoder_stack_provenance.json';env=json.loads(p.read_text())
            env['environment']['GOLEM_DECODER_ADD_NORM_FUSION']='1';p.write_text(json.dumps(env))
            for group in jobs.values():
                group[:]=[j for j in group if j['stage']!=3]
                for j in group:
                    if j['stage']==2:j['hbm_read']=512
            lines=[]
            for line in log.read_text().splitlines(True):
                if 'stage=3 ' in line:continue
                if 'stage=2 ' in line:
                    line=line.replace('step=2 state=2','step=2 state=3').replace('32768','65536')
                    line=line.replace('row_bytes=65536','row_bytes=32768')
                    fields=dict(item.split('=') for item in line.split()[1:])
                    lines.append(f"[DECODER_ADD_NORM_FUSION] core={fields['core']} epoch={fields['epoch']} stage=2 chunks=2 saved_gm_read=32768 norm_bytes=32768 fp16_boundary=1 capacity_added=0 status=0\n")
                lines.append(line)
            log.write_text(''.join(lines));result,_,_=chain_audit(case,jobs,m)
            self.assertEqual(result['incremental_saved_gm_read_bytes'],32*32768)
            self.assertEqual(result['saved_hbm_write_bytes'],2*3*1024*256*2)


class EagerNormTest(unittest.TestCase):
    def check(self, directory, invalid=False):
        original=norm_fixture.verify_final_norm

        def verify(root,m,log,end,timing):
            m['eager_final_norm']=True;end.update({1:200,2:350})
            lines=log.read_text().splitlines(True)
            for core in range(4,20):
                lines.append(f'[DECODER_OP] event=done core={core} stage=1 op=1 epoch=2 status=0 start_tick=260 tick=270\n')
                lines.append(f'[DECODER_OP] event=done core={core} stage=4 op=9 epoch=2 status=0 start_tick=275 tick={350 if core==19 else 290}\n')
            lines=[line.replace('core=19 stage=10 op=3 epoch=3 status=0 start_tick=300 tick=400',
                                'core=19 stage=10 op=3 epoch=3 status=0 start_tick=360 tick=460') for line in lines]
            if invalid:lines=[line.replace('stage=4 op=9 epoch=2 status=0 start_tick=275 tick=290',
                                         'stage=4 op=9 epoch=2 status=0 start_tick=275 tick=320') for line in lines]
            log.write_text(''.join(line if line.endswith('\n') else line+'\n' for line in lines))
            resource=log.with_suffix('.resources.jsonl');events=resource.read_text().splitlines(True)
            def shift(line):
                if 'core=19 ' not in line:return line
                import re
                return re.sub(r'((?:start|done)_tick)=(\d+)',lambda f:f'{f[1]}={int(f[2])+60}',line)
            resource.write_text(''.join(map(shift,events)))
            (root/'decoder_stack_provenance.json').write_text(json.dumps(dict(environment={'GOLEM_DECODER_EAGER_FINAL_NORM':'1'})))
            return original(root,m,log,end,timing)

        with patch.object(norm_fixture,'verify_final_norm',verify):
            return norm_fixture.DecoderStackTest().final_fixture(Path(directory))

    def test_worker_local_dependency_allows_overlap_with_slow_worker(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.check(d);self.assertTrue(result['eager_worker_handoff'])
            self.assertEqual(result['overlap_with_decoder_tail_cycles'],.05)

    def test_worker_cannot_normalize_before_own_ffn_ack(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):self.check(d,True)


if __name__=='__main__':unittest.main()
