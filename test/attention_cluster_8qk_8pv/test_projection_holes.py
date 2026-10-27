import json
from pathlib import Path
import tempfile
import unittest

from analyze_projection_holes import analyze


class ProjectionHoleTest(unittest.TestCase):
    def case(self, root, arrays=64):
        case=Path(root);(case/'logs').mkdir()
        (case/'resource_analysis.json').write_text(json.dumps({'ticks_per_cycle':1000}))
        (case/'projection_e2e_result.json').write_text(json.dumps({'stages':{'projection':{'elapsed_cycles':100}}}))
        log=[]; resources=[]
        for core in range(4,20):
            log.append(f'[PROJECTION_JOB] core={core} start=0 end=100 cycles=100')
            for i,(kind,phase,cycle) in enumerate((('phase_start','norm_wait',0),('phase_done','norm_wait',60),
                        ('phase_start','program_input',60),('phase_done','program_input',100)),1):
                log.append(f'[CRITICAL_EVENT] core={core} event_id={i} kind={kind} phase={phase} cycle={cycle}')
            for domain,resource,lo,hi,client in (('array','compute',40,50,arrays),('array','compute',70,80,arrays),
                        ('array','matrix_broadcast',55,65,64),('gm','dma_read_transport',10,90,0),
                        ('gm','local_read',56,66,1)):
                resources.append(f'[RESOURCE_INTERVAL] domain={domain} resource={resource} core={core} client={client} submit_tick={lo*1000} start_tick={lo*1000} done_tick={hi*1000}')
        (case/'logs/attention-sst-test.log').write_text('\n'.join(log))
        (case/'logs/attention-sst-test.resources.jsonl').write_text('\n'.join(resources))
        return case

    def test_overlapping_resources_do_not_inflate_exclusive_gaps(self):
        with tempfile.TemporaryDirectory() as root:
            result=analyze(self.case(root))['by_worker']['4']
            self.assertEqual(result['between_mac_gap_cycles'],20)
            self.assertEqual(result['gap_cycles_by_controller_phase'],{'norm_wait':10,'program_input':10})
            self.assertEqual(result['gap_cycles_with_resource_active']['gm_active'],20)
            self.assertEqual(result['gap_cycles_with_resource_active']['matrix_service'],10)
            self.assertEqual(result['mac_utilization'],.2)

    def test_array_weighted_occupancy(self):
        with tempfile.TemporaryDirectory() as root:
            result=analyze(self.case(root,arrays=32))['by_worker']['4']
            self.assertEqual(result['mac_utilization'],.1)

    def test_missing_worker_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            case=self.case(root)
            p=case/'logs/attention-sst-test.log'
            p.write_text('\n'.join(line for line in p.read_text().splitlines() if 'core=19 ' not in line))
            with self.assertRaisesRegex(ValueError,'job windows'): analyze(case)

if __name__=='__main__':unittest.main()
