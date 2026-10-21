import json
from pathlib import Path
import tempfile
import unittest
from verify_layer_residency import audit, audit_input_residency, verify_input_payload, verify_projection_payload


class LayerResidencyAuditTest(unittest.TestCase):
    def evidence(self, root):
        (root/'logs').mkdir()
        (root/'decoder_stack_provenance.json').write_text(json.dumps({'environment':{'GOLEM_DECODER_LAYER_RESIDENCY':'1'}}))
        m=dict(sequence=1024,hidden=256,layers=2,layer_residency=True)
        jobs={};lines=[]
        for core in range(4,20):
            jobs[core]=[dict(core=core,epoch=e,stage=4,op=9) for e in (1,2)]+[dict(core=core,epoch=3,stage=10,op=3,hbm_read=512)]
            for e in (1,2):lines.append(f'[SOFTWARE_LAYER_RESIDENCY] core={core} epoch={e} stage=4 producer=1 bytes=32768 saved_hbm_read=0 metadata_bytes=64 capacity_added=0 status=0')
            lines.append(f'[SOFTWARE_LAYER_RESIDENCY] core={core} epoch=3 stage=10 producer=0 bytes=0 saved_hbm_read=32768 metadata_bytes=64 capacity_added=0 status=0')
            lines.append(f'[PROJECTION_LAYER_RESIDENCY] core={core} epoch=2 saved_hbm_read=32768 metadata_bytes=64 capacity_added=0 status=0')
        path=root/'logs/attention-sst.log';path.write_text('\n'.join(lines)+'\n')
        return m,jobs,path

    def check(self, transform=None, manifest=False):
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);m,j,p=self.evidence(r)
            if transform:p.write_text(transform(p.read_text()))
            if manifest:m['layer_residency']=False
            return audit(r,j,m)

    def test_complete_evidence(self):
        self.assertEqual(self.check()['saved_hbm_read_bytes'],1048576)

    def test_missing_publisher(self):
        with self.assertRaisesRegex(ValueError,'lifecycle'):
            self.check(lambda s:'\n'.join(s.splitlines()[1:]))

    def test_wrong_payload_bytes(self):
        with self.assertRaisesRegex(ValueError,'bytes'):
            self.check(lambda s:s.replace('bytes=32768 saved','bytes=16384 saved',1))

    def test_extra_capacity(self):
        with self.assertRaisesRegex(ValueError,'bytes/ownership'):
            self.check(lambda s:s.replace('capacity_added=0','capacity_added=8192',1))

    def test_manifest_mismatch(self):
        with self.assertRaisesRegex(ValueError,'manifest'):
            self.check(manifest=True)

    def test_duplicate_projection_consumer(self):
        with self.assertRaisesRegex(ValueError,'projection residency lifecycle'):
            self.check(lambda s:s+next(l for l in s.splitlines() if '[PROJECTION_LAYER_RESIDENCY]' in l)+'\n')


class ProjectionPayloadAuditTest(unittest.TestCase):
    def check(self, transform=None):
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);(r/'logs').mkdir();log=r/'logs/attention-sst.log'
            m=dict(layer_residency=True,sequence=1024,hidden=64,layers=2,
                   layout=dict(node_stride=0x80000000,layers=[dict(input=0x10000),dict(input=0x20000)]))
            lines=[];events=[]
            for core in range(4,20):
                lines += [f'[CONTROL_EVENT] stage=projection_accept core={core} epoch=1 tick=1000',
                          f'[PROJECTION_JOB] core={core} start=1 end=50',
                          f'[CONTROL_EVENT] stage=projection_accept core={core} epoch=2 tick=100000',
                          f'[PROJECTION_JOB] core={core} start=100 end=200',
                          f'[SFU_RMSNORM] core={core} tag={(core-4)//4*64} rows=64 cols=64 issue_tick=110000 complete_tick=180000']
                events.append(f'resource=local_read core={core} address={core*0x200000+0xb0000} bytes=8192 start_tick=120000 done_tick=125000 ok=1')
            log.write_text('\n'.join(lines)+'\n')
            text='\n'.join(events)+'\n'
            log.with_suffix('.resources.jsonl').write_text(transform(text) if transform else text)
            return verify_projection_payload(r,m)

    def test_full_rows_real_ack(self):
        self.assertEqual(self.check()['saved_hbm_read_bytes'],16*8192)

    def test_missing_payload(self):
        with self.assertRaisesRegex(ValueError,'missing/duplicate'):
            self.check(lambda s:'\n'.join(s.splitlines()[1:]))

    def test_duplicate_payload(self):
        with self.assertRaisesRegex(ValueError,'missing/duplicate'):
            self.check(lambda s:s+s.splitlines()[0]+'\n')

    def test_hidden_hbm_reread(self):
        with self.assertRaisesRegex(ValueError,'still reads HBM'):
            self.check(lambda s:s+f'resource=dma_read_transport core=4 address={0x80000000+0x20000} bytes=128 start_tick=120000 done_tick=125000 ok=1\n')

    def test_wrong_row_address(self):
        with self.assertRaisesRegex(ValueError,'wrong row'):
            self.check(lambda s:s.replace(f'address={4*0x200000+0xb0000}',f'address={4*0x200000+0xb0000+128}',1))


class InputResidencyAuditTest(unittest.TestCase):
    def check(self, corrupt=None):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'logs').mkdir()
            (root/'decoder_stack_provenance.json').write_text(json.dumps({'environment':{
                'GOLEM_DECODER_INPUT_RESIDENCY':'1','GOLEM_DECODER_LAYER_RESIDENCY':'1',
                'GOLEM_DECODER_GM_STAGE_CHAIN':'1'}}))
            manifest=dict(sequence=1024,hidden=256,layers=2,input_residency=True)
            jobs={};lines=[]
            for core in range(4,20):
                jobs[core]=[dict(core=core,epoch=e,stage=2,op=2,hbm_read=0) for e in (1,2)]
                for epoch in (1,2):
                    lines.append(f'[DECODER_INPUT_RESIDENCY] core={core} epoch={epoch} stage=2 saved_hbm_read=32768 metadata_bytes=64 status=0')
                lines.append(f'[PROJECTION_INPUT_RESIDENCY] core={core} epoch=1 retained_bytes=32768 metadata_bytes=64 status=0')
            text='\n'.join(lines)+'\n'
            if corrupt:text=corrupt(text)
            (root/'logs/attention-sst.log').write_text(text)
            return audit_input_residency(root,jobs,manifest)

    def test_complete_evidence(self):
        self.assertEqual(self.check()['saved_hbm_read_bytes'],16*2*32768)

    def test_missing_projection_seed(self):
        with self.assertRaisesRegex(ValueError,'projection input retention'):
            self.check(lambda s:'\n'.join(line for line in s.splitlines() if 'PROJECTION_INPUT' not in line))

    def test_residual_hbm_read_rejects_candidate(self):
        with self.assertRaisesRegex(ValueError,'no HBM reads'):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'logs').mkdir()
                (root/'decoder_stack_provenance.json').write_text(json.dumps({'environment':{
                    'GOLEM_DECODER_INPUT_RESIDENCY':'1','GOLEM_DECODER_LAYER_RESIDENCY':'1','GOLEM_DECODER_GM_STAGE_CHAIN':'1'}}))
                manifest=dict(sequence=1024,hidden=256,layers=2,input_residency=True)
                jobs={4:[dict(core=4,epoch=1,stage=2,op=2,hbm_read=32768)]}
                (root/'logs/attention-sst.log').write_text('[DECODER_INPUT_RESIDENCY] core=4 epoch=1 stage=2 saved_hbm_read=32768 metadata_bytes=64 status=0\n')
                audit_input_residency(root,jobs,manifest)


class InputPayloadAuditTest(unittest.TestCase):
    def check(self, mutation=None):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'logs').mkdir();log=root/'logs/attention-sst.log'
            manifest=dict(sequence=1024,hidden=64,layers=1,input_residency=True)
            jobs={4:[dict(core=4,epoch=1,stage=2,start_tick=40000,tick=100000)]}
            log.write_text('[CONTROL_EVENT] stage=projection_accept core=4 epoch=1 tick=1000\n'
                           '[SFU_RMSNORM] core=4 rows=64 issue_tick=10000 complete_tick=20000\n'
                           '[PROJECTION_JOB] core=4 start=1 end=30\n')
            base=4*0x200000+0xb0000
            events=[]
            for offset in (0,4096):
                events.append(f'resource=local_read core=4 address={base+offset} bytes=4096 submit_tick=45000 start_tick=46000 done_tick=50000 ok=1')
                events.append(f'resource=local_write core=4 address={base+offset} bytes=4096 submit_tick=60000 start_tick=61000 done_tick=65000 ok=1')
            text='\n'.join(events)+'\n'
            if mutation:text=mutation(text)
            log.with_suffix('.resources.jsonl').write_text(text)
            return verify_input_payload(root,jobs,manifest)

    def test_complete_timed_reads_before_inplace_write(self):
        self.assertEqual(self.check()['saved_hbm_read_bytes'],8192)

    def test_overwrite_during_attention_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'between Norm1 and Add'):
            self.check(lambda s:s.replace('submit_tick=60000','submit_tick=30000',1))

    def test_write_before_input_read_ack_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'unread residual'):
            self.check(lambda s:s.replace('start_tick=61000','start_tick=47000',1))

    def test_missing_payload_read_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'coverage missing/duplicate'):
            self.check(lambda s:'\n'.join(s.splitlines()[1:]))


if __name__=='__main__':unittest.main()
