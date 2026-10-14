import json
from pathlib import Path
import tempfile
import unittest

from analyze_projection_causal import analyze


class ProjectionCausalTest(unittest.TestCase):
    def case(self, root):
        case=Path(root);(case/'logs').mkdir()
        (case/'resource_analysis.json').write_text(json.dumps({'ticks_per_cycle':1000}))
        main=[];events=[];resources=[]
        for core in range(4,20):
            main.append(f'[PROJECTION_JOB] core={core} start=0 end=200 cycles=200')
            key=f'core={core} block=0 kind=0 head=0 dim=0 row=0 tile=0'
            for name,role,tick,tag,address,size in (
                ('read_submit','foreground_weight',5000,1,900,4096),('read_done','foreground_weight',22000,1,900,4096),
                ('decode_start','weight',30000,0,0,0),('decode_done','weight',30000,0,0,0),
                ('matrix_submit','weight',30000,2,0,0),('matrix_done','weight',70000,2,0,0),
                ('scatter_submit','input',71000,3,0,0),('scatter_done','input',105000,3,0,0),
                ('mac_submit','compute',106000,0,0,0),('mac_done','compute',174000,0,0,0),
                ('dma_submit','weight',1000,0,9000,8192),('dma_done','weight',20000,0,9000,8192)):
                events.append(f'[PROJECTION_CAUSAL] {key} event={name} role={role} tick={tick} tag={tag} address={address} bytes={size}')
            for domain,resource,tag,address,size,submit,start,done in (
                ('gm','local_read',1,900,4096,5000,5000,22000),
                ('array','matrix_broadcast',2,0,8192,31000,31000,70000),
                ('array','input_scatter',3,0,8192,72000,72000,105000),
                ('array','compute',0,0,0,107000,107000,173000),
                ('gm','dma_read_transport',0,9000,8192,1000,1000,19000)):
                resources.append(f'[RESOURCE_INTERVAL] domain={domain} resource={resource} core={core} tag={tag} address={address} bytes={size} submit_tick={submit} start_tick={start} done_tick={done} visible_tick={done}')
        (case/'logs/attention-sst-test.log').write_text('\n'.join(main))
        (case/'logs/attention-sst-test.causal.jsonl').write_text('\n'.join(events))
        (case/'logs/attention-sst-test.resources.jsonl').write_text('\n'.join(resources))
        return case

    def test_joins_preserve_timing_units_and_zero_decode(self):
        with tempfile.TemporaryDirectory() as root:
            result=analyze(self.case(root))
            worker=result['by_worker']['4']
            self.assertEqual(worker['joined_local_read_requests'],1)
            self.assertEqual(worker['joined_array_commands'],2)
            self.assertEqual(worker['array_command']['matrix_service']['mean_cycles'],39)
            self.assertEqual(worker['array_command']['decode']['sum_cycles'],0)
            self.assertEqual(worker['local_read_by_role']['foreground_weight']['queue']['sum_cycles'],0)
            self.assertEqual(worker['weight_dma']['transport']['mean_cycles'],18)

    def test_wrong_tag_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            case=self.case(root);p=case/'logs/attention-sst-test.resources.jsonl'
            p.write_text(p.read_text().replace('tag=1 address=900','tag=999 address=900'))
            with self.assertRaisesRegex(ValueError,'missing read join'): analyze(case)

    def test_wrong_callback_tick_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            case=self.case(root);p=case/'logs/attention-sst-test.causal.jsonl'
            p.write_text(p.read_text().replace('tick=22000 tag=1','tick=23000 tag=1'))
            with self.assertRaisesRegex(ValueError,'timestamp/payload mismatch'): analyze(case)

if __name__=='__main__':unittest.main()
