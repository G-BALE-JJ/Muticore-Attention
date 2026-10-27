import json
from pathlib import Path
import tempfile
import unittest
from verify_output_residency import audit


class OutputResidencyTest(unittest.TestCase):
    def fixture(self,root):
        case=Path(root);(case/'logs').mkdir();log=case/'logs/attention-sst.log'
        (case/'decoder_stack_provenance.json').write_text(json.dumps(dict(environment={
            'GOLEM_DECODER_ELIDE_INTERMEDIATE_OUTPUT':'1'})))
        manifest=dict(sequence=1024,hidden=256,layers=2,gm_stage_chain=True,
                      layer_residency=True,input_residency=True,elide_intermediate_output=True)
        size=64*256*2;base=4*0x200000
        jobs={4:[dict(core=4,epoch=1,stage=4,op=9,start_tick=100,tick=200,hbm_write=8),
                 dict(core=4,epoch=2,stage=4,op=9,start_tick=300,tick=400,hbm_write=size+8)]}
        log.write_text(f'[DECODER_OUTPUT_RESIDENCY] core=4 epoch=1 stage=4 local_produced={size} saved_hbm_write={size} published=1 capacity_added=0 status=0\n')
        log.with_suffix('.resources.jsonl').write_text(
            f'[RESOURCE_INTERVAL] resource=local_write core=4 start_tick=110 done_tick=140 address={base+0xb0000} bytes={size} ok=1\n'
            f'[RESOURCE_INTERVAL] resource=local_write core=4 start_tick=150 done_tick=170 address={base+0x3400} bytes=64 ok=1\n')
        return case,manifest,jobs,log

    def test_exact_timed_payload_and_final_layer_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            case,m,jobs,_=self.fixture(d);result=audit(case,jobs,m)
            self.assertEqual(result['verified_jobs'],1)
            self.assertEqual(result['saved_hbm_write_bytes'],32768)
            self.assertEqual(result['actual_gm_payload_bytes'],32768)

    def test_rejects_missing_duplicate_failed_or_early_publication(self):
        for mutation in ('missing','duplicate','failed','early','short','final'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                case,m,jobs,log=self.fixture(d);p=log.with_suffix('.resources.jsonl');s=p.read_text()
                if mutation=='missing':s=s.splitlines(True)[1]
                elif mutation=='duplicate':s+=s.splitlines(True)[0]
                elif mutation=='failed':s=s.replace('ok=1','ok=0',1)
                elif mutation=='early':s=s.replace('start_tick=150','start_tick=130')
                elif mutation=='short':s=s.replace('bytes=32768','bytes=32767')
                else:log.write_text(log.read_text().replace('epoch=1','epoch=2'))
                p.write_text(s)
                with self.assertRaises(ValueError):audit(case,jobs,m)

    def test_rejects_materialized_or_unpublished_y(self):
        with tempfile.TemporaryDirectory() as d:
            case,m,jobs,log=self.fixture(d);jobs[4][0]['hbm_write']+=32768
            with self.assertRaises(ValueError):audit(case,jobs,m)
            jobs[4][0]['hbm_write']=8
            log.write_text(log.read_text().replace('published=1','published=0'))
            with self.assertRaises(ValueError):audit(case,jobs,m)


if __name__=='__main__':unittest.main()
