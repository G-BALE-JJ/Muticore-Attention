import json
from pathlib import Path
import tempfile
import unittest
from analyze_batch_b_waits import analyze, overlap


class ProjectionWindowTests(unittest.TestCase):
    def test_clips_wait_to_projection_window(self):
        self.assertEqual(overlap(50,120,100,200),20)
        self.assertEqual(overlap(200,220,100,200),0)

    def test_weights_mac_by_active_arrays_and_excludes_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp);(case/'logs').mkdir()
            (case/'projection_e2e_result.json').write_text('{}')
            (case/'resource_analysis.json').write_text(json.dumps({'ticks_per_cycle':10}))
            log=case/'logs'/'attention-sst.log'
            log.write_text(''.join(f'[PROJECTION_JOB] core={c} start=10 end=20\n' for c in range(4,20)))
            log.with_suffix('.resources.jsonl').write_text(
                '[RESOURCE_INTERVAL] domain=array resource=compute core=4 client=32 submit_tick=50 start_tick=50 done_tick=150 visible_tick=150\n'
                '[RESOURCE_INTERVAL] domain=gm resource=local_read core=4 client=1 submit_tick=50 start_tick=120 done_tick=140 visible_tick=140\n'
                '[RESOURCE_INTERVAL] domain=array resource=compute core=4 client=64 submit_tick=220 start_tick=220 done_tick=240 visible_tick=240\n')
            result=analyze(case)['by_worker']['4']
            self.assertEqual(result['array_mac_cycles'],160)
            self.assertEqual(result['mac_array_utilization'],0.25)
            self.assertEqual(result['resources']['gm.local_read']['queue_cycles'],2)
            self.assertEqual(result['resources']['gm.local_read']['service_cycles'],2)
            self.assertEqual(result['resources']['array.compute']['count'],1)


if __name__=='__main__':unittest.main()
