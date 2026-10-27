"""Reject invalid residency evidence, not just incorrect numerical outputs."""
from pathlib import Path
import json
import tempfile
import unittest
from verify_software_stage_chain import audit


class SoftwareStageChainAuditTest(unittest.TestCase):
    def fixture(self,root,enabled=True):
        case=Path(root);(case/'logs').mkdir();manifest=dict(sequence=1024,hidden=256,gm_stage_chain=enabled)
        (case/'decoder_stack_provenance.json').write_text(json.dumps(dict(environment={'GOLEM_DECODER_GM_STAGE_CHAIN':str(int(enabled))})))
        size=64*256*2;jobs={};lines=[]
        for core in range(4,20):
            jobs[core]=[]
            for epoch in (1,2):
                for step in range(1,5):
                    reads={1:0,2:size,3:size,4:2*size}[step];writes=size if step<4 else 0
                    jobs[core].append(dict(core=core,epoch=epoch,stage=step,hbm_read=size if step==2 else 512 if step==3 else 1000000,hbm_write=0 if step<4 else size+8))
                    lines.append(f'[DECODER_STAGE_CHAIN] core={core} epoch={epoch} stage={step} step={step} state={step if step<4 else 0} row_bytes={size} local_produced={size if step<4 else 0} saved_hbm_read={reads} saved_hbm_write={writes} capacity_added=0 status=0\n')
        log=case/'logs/attention-sst.log';log.write_text(''.join(lines) if enabled else '')
        return case,manifest,jobs,log

    def test_exact_bytes_and_drained_ownership(self):
        with tempfile.TemporaryDirectory() as d:
            case,m,jobs,_=self.fixture(d);result,ffn,residual=audit(case,jobs,m)
            self.assertEqual(result['saved_hbm_read_bytes'],2*4*1024*256*2)
            self.assertEqual(result['saved_hbm_write_bytes'],2*3*1024*256*2)
            self.assertEqual(len(ffn),32);self.assertEqual(len(residual),32)

    def test_disabled_has_no_flows(self):
        with tempfile.TemporaryDirectory() as d:
            case,m,jobs,_=self.fixture(d,False);self.assertFalse(audit(case,jobs,m)[0]['enabled'])

    def test_missing_duplicate_or_undrained_evidence_is_rejected(self):
        for mutation in ('missing','duplicate','state','capacity','bytes'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                case,m,jobs,log=self.fixture(d);text=log.read_text();lines=text.splitlines(True)
                if mutation=='missing':text=''.join(lines[1:])
                elif mutation=='duplicate':text+=lines[0]
                elif mutation=='state':text=text.replace('step=4 state=0','step=4 state=1',1)
                elif mutation=='capacity':text=text.replace('capacity_added=0','capacity_added=1',1)
                else:text=text.replace('saved_hbm_read=32768','saved_hbm_read=32767',1)
                log.write_text(text)
                with self.assertRaises(ValueError):audit(case,jobs,m)

    def test_materialized_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            case,m,jobs,_=self.fixture(d);jobs[4][0]['hbm_write']=1
            with self.assertRaises(ValueError):audit(case,jobs,m)

    def test_manifest_switch_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            case,m,jobs,_=self.fixture(d);m['gm_stage_chain']=False
            with self.assertRaises(ValueError):audit(case,jobs,m)

if __name__=='__main__':unittest.main()
