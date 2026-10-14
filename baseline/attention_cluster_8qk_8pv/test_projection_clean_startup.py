import json
from pathlib import Path
import tempfile
import unittest

from run_projection_clean_startup import analyze_clean
from run_projection_overlap import digest
from summarize_projection_clean_startup import validate_snapshot


class CleanStartupTests(unittest.TestCase):
    def fixture(self, root, diagnostic=''):
        (root/'stdout/overlap0/run').mkdir(parents=True)
        (root/'logs').mkdir()
        (root/'stdout/overlap0/run/stderr-100').write_text(diagnostic)
        (root/'logs/attention-sst-test.log').write_text('Simulation is complete, simulated time: 0.320 us\n')
        (root/'projection_e2e_result.json').write_text(json.dumps({
            'chain_start_cycle':150,'end_to_end_cycles':150,
            'stages':{'attention':{'end_cycle':300}}}))
        return {'worker_control':{str(c):{'release_issue_cycle':100,'job_accept_cycle':150,
                    'release_to_observed_cycles':2,'observed_to_accept_cycles':48} for c in range(4,20)},
                'last_attention_output_ack_cycle':299,'last_root_tensor_complete_cycle':301}

    def test_accepts_guest_without_startup_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);resources=self.fixture(root)
            self.assertEqual(analyze_clean(root,resources)['simulation_end_cycles_from_zero'],320)

    def test_rejects_each_startup_diagnostic(self):
        for marker in ('ATTENTION_GUEST_START','PROJECTION_START','PROJECTION_DISPATCH'):
            with self.subTest(marker=marker),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);resources=self.fixture(root,f'[{marker}] core=4\n')
                with self.assertRaisesRegex(ValueError,'diagnostics executed'):
                    analyze_clean(root,resources)

    def test_requires_process_exit_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);resources=self.fixture(root)
            (root/'logs/attention-sst-test.log').write_text('partial simulator output\n')
            with self.assertRaisesRegex(ValueError,'completion time missing'):
                analyze_clean(root,resources)

    def test_archived_source_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'guest.cpp';p.write_text('original')
            hashes={'guest.cpp':digest(p)}
            validate_snapshot(root,hashes)
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                validate_snapshot(root,hashes)


if __name__=='__main__':unittest.main()
