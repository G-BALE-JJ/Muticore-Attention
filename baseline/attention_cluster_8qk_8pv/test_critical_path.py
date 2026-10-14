import json
from pathlib import Path
import tempfile
import unittest

from analyze_critical_path import analyze


def event(event_id, predecessor, kind, phase, cycle, core):
    return (f"[CRITICAL_EVENT] event_id={event_id} predecessor={predecessor} kind={kind} "
            f"phase={phase} resource=control cycle={cycle} core={core} job=100 block=0 "
            f"head=0 tile=0 generation=1")


class CriticalPathTraceTests(unittest.TestCase):
    def make_case(self, log):
        root = Path(tempfile.mkdtemp())
        (root / 'logs').mkdir()
        (root / 'logs' / 'attention-sst-test.log').write_text(log)
        jobs = '\n'.join(f'[PROJECTION_JOB] core={core} start=10 end=100 cycles=90 weight_loads=1 weight_programs=1 weight_reuses=0 input_loads=1 status=0'
                          for core in range(4, 20))
        waits = '\n'.join(f'[PROJECTION_SYNC] core={core} stage=local_wait cycle=100 status=0'
                          for core in range(4, 20))
        milestones = ('[ATTENTION_MILESTONE] stage=root_descriptor_accept status=done sst_tick=110 rocc_cycle=110 core=0 role=manager job=1 tag=2 query_tile=-1 kv_tile=-1\n'
                      '[ATTENTION_MILESTONE] stage=worker_complete status=done sst_tick=200 rocc_cycle=200 core=4 role=worker job=1 tag=2 query_tile=-1 kv_tile=-1\n'
                      '[ATTENTION_MILESTONE] stage=root_tensor_complete status=done sst_tick=200 rocc_cycle=200 core=0 role=manager job=1 tag=2 query_tile=-1 kv_tile=-1\n')
        (root / 'projection_e2e_result.json').write_text(json.dumps({
            'end_to_end_cycles': 190,
            'chain_start_cycle': 10,
            'stage_handoff_cycles': {'projection_to_attention': 5},
            'stages': {'projection': {'elapsed_cycles': 90, 'start_cycle': 10, 'end_cycle': 100}},
        }))
        (root / 'logs' / 'attention-sst-test.log').write_text(log + '\n' + jobs + '\n' + waits + '\n' + milestones)
        return root

    def test_accepts_complete_phase_chains(self):
        lines = []
        for core in range(4, 20):
            lines.extend([event(1, 0, 'phase_start', 'load_input', 10, core),
                          event(2, 1, 'phase_done', 'load_input', 20, core),
                          event(3, 2, 'phase_start', 'complete', 20, core)])
        result = analyze(self.make_case('\n'.join(lines)))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['critical_event_count'], 48)

    def test_rejects_broken_predecessor(self):
        lines = []
        for core in range(4, 20):
            lines.extend([event(1, 0, 'phase_start', 'load_input', 10, core),
                          event(3, 1, 'phase_done', 'load_input', 20, core),
                          event(4, 2, 'phase_start', 'complete', 20, core)])
        with self.assertRaisesRegex(ValueError, 'broken event predecessor'):
            analyze(self.make_case('\n'.join(lines)))


if __name__ == '__main__':
    unittest.main()
