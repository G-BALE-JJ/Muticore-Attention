import copy
import json
from pathlib import Path
import tempfile
import unittest

from analyze_projection_startup import startup_metrics, compare, analyze


class StartupAccountingTests(unittest.TestCase):
    def fixture(self):
        controls = {str(c): {'release_issue_cycle': 100 + 10 * ((c - 4) // 4),
            'job_accept_cycle': 150 + 10 * ((c - 4) // 4),
            'release_to_observed_cycles': 2, 'observed_to_accept_cycles': 48}
            for c in range(4, 20)}
        resources = {'worker_control': controls, 'last_attention_output_ack_cycle': 299,
                     'last_root_tensor_complete_cycle': 301}
        projection = {'chain_start_cycle': 150, 'end_to_end_cycles': 150,
                      'stages': {'attention': {'end_cycle': 300}}}
        return {'input_sha256': {'x': 'same'},
                'startup': startup_metrics(projection, resources, 320)}

    def test_includes_preparation_in_absolute_completion(self):
        row = self.fixture()['startup']
        self.assertEqual(row['reported_e2e_cycles'], 150)
        self.assertEqual(row['release_to_attention_complete_cycles'], 200)
        self.assertEqual(row['attention_complete_cycle_from_simulation_zero'], 300)
        self.assertEqual(row['manager_release_span_cycles']['0'], 30)

    def test_moving_start_later_is_not_real_improvement(self):
        baseline = self.fixture()
        candidate = copy.deepcopy(baseline)
        candidate['startup']['reported_e2e_cycles'] -= 30
        result = compare(baseline, candidate)
        self.assertEqual(result['reported_e2e_cycles']['saved_cycles'], 30)
        self.assertFalse(result['real_completion_improves'])

    def test_output_and_process_exit_must_both_improve(self):
        baseline = self.fixture()
        candidate = copy.deepcopy(baseline)
        for key in ('last_output_ack_cycle_from_simulation_zero',
                    'root_complete_cycle_from_simulation_zero'):
            candidate['startup'][key] -= 20
        self.assertFalse(compare(baseline, candidate)['real_completion_improves'])
        candidate['startup']['simulation_end_cycles_from_zero'] -= 20
        self.assertTrue(compare(baseline, candidate)['real_completion_improves'])

    def test_rejects_input_change(self):
        baseline = self.fixture()
        candidate = copy.deepcopy(baseline)
        candidate['input_sha256']['x'] = 'different'
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            compare(baseline, candidate)

    def test_rejects_missing_worker(self):
        controls = {'4': {'release_issue_cycle': 100}}
        with self.assertRaisesRegex(ValueError, 'missing worker'):
            startup_metrics({}, {'worker_control': controls}, 320)

    def test_guest_dispatches_are_read_from_guest_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'logs').mkdir()
            (root / 'stdout/overlap0/run').mkdir(parents=True)
            (root / 'logs/attention-sst-test.log').write_text('Simulation is complete, simulated time: 0.320 us\n')
            (root / 'stdout/overlap0/run/stderr-100').write_text(''.join(
                f'[PROJECTION_DISPATCH] manager={m} worker={4+4*lane+m} row_offset={lane*64} row_count=64\n'
                for m in range(4) for lane in range(4)))
            projection = {'chain_start_cycle': 150, 'end_to_end_cycles': 150,
                          'stages': {'attention': {'end_cycle': 300}}}
            (root / 'projection_e2e_result.json').write_text(json.dumps(projection))
            resources = {'worker_control': {str(c): {'release_issue_cycle': 100,
                'job_accept_cycle': 150, 'release_to_observed_cycles': 2,
                'observed_to_accept_cycles': 48} for c in range(4,20)},
                'last_attention_output_ack_cycle': 299, 'last_root_tensor_complete_cycle': 301}
            self.assertEqual(analyze(root,resources)['simulation_end_cycles_from_zero'],320)


if __name__ == '__main__':
    unittest.main()
