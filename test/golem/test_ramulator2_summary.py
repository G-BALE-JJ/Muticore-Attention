import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BACKEND = ('RAMULATOR2_BACKEND_SUMMARY node=1 read_count=6 write_count=1 '
           'read_bytes=192 write_bytes=32 first_read_arrival_cycle=0 '
           'last_read_complete_cycle=20 read_window_cycles=21 '
           'avg_read_latency_cycles=5 p95_read_latency_cycles=10 '
           'p99_read_latency_cycles=10 max_read_latency_cycles=10 pending=0')
COMMAND = ('RAMULATOR2_COMMAND_SUMMARY node=1 channel=0 rd=6 wr=1 '
           'first_rd=0 last_rd=13 read_window=14 tccd_stats_version=2 '
           'pc_read_sequences=2 global_read_pairs=4 '
           'tccd_l_pairs=2 tccd_l_gap_sum=8 tccd_l_min_gap=4 tccd_l_violations=0 '
           'tccd_s_pairs=2 tccd_s_gap_sum=13 tccd_s_min_gap=4 tccd_s_violations=0 '
           'cross_pc_pairs=3 cross_pc_gap_sum=5 cross_pc_min_gap=1 '
           'rw_switches=2 rw_switch_gap_sum=4')
CONTROLLER = ('RAMULATOR2_CONTROLLER_SUMMARY node=1 channel=0 '
              'read_row_hits=6 read_row_misses=0 read_row_conflicts=0')


class ReadSummaryTests(unittest.TestCase):
    def extract(self, text):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'commands.log'
            log.write_text(text)
            process = subprocess.run([
                sys.executable, str(Path(__file__).parent / 'stats/extract_ramulator2_summary.py'),
                '--log', str(log), '--channels-per-stack', '1', '--memory-clock', '1250MHz',
                '--memory-summary', str(root / 'memory.csv'),
                '--hbm-summary', str(root / 'hbm.csv'), '--hbm-nodes', str(root / 'nodes.csv'),
            ], capture_output=True, text=True)
            metrics = {}
            if process.returncode == 0:
                with (root / 'hbm.csv').open() as handle:
                    metrics = dict(list(csv.reader(handle))[1:])
            return process, metrics

    def test_independent_denominators(self):
        process, metrics = self.extract('\n'.join([BACKEND, COMMAND, CONTROLLER]))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(metrics['hbm_read_command_tccd_l_pair_pct'], '50.000000')
        self.assertEqual(metrics['hbm_read_command_tccd_s_pair_pct'], '50.000000')
        self.assertEqual(metrics['hbm_read_command_cross_pseudo_channel_pair_pct'], '75.000000')
        self.assertEqual(metrics['hbm_read_command_tccd_pair_scope'], 'per_pc_reads')

    def test_legacy_scope_is_explicit(self):
        command = COMMAND.replace('tccd_stats_version=2 ', '').replace('tccd_l_pairs=2', 'tccd_l_pairs=1').replace('tccd_s_pairs=2', 'tccd_s_pairs=0')
        process, metrics = self.extract('\n'.join([BACKEND, command, CONTROLLER]))
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(metrics['hbm_read_command_tccd_l_pair_pct'], '100.000000')
        self.assertEqual(metrics['hbm_read_command_cross_pseudo_channel_pair_pct'], '75.000000')
        self.assertEqual(metrics['hbm_read_command_tccd_pair_scope'], 'adjacent_channel_read_columns')

    def test_pair_conservation(self):
        process, _ = self.extract('\n'.join([BACKEND, COMMAND.replace('tccd_l_pairs=2', 'tccd_l_pairs=3'), CONTROLLER]))
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('per-PC read pair conservation failed', process.stderr)

    def test_mixed_versions_rejected(self):
        text = '\n'.join([BACKEND, COMMAND, CONTROLLER])
        process, _ = self.extract(text + '\n' + text.replace('node=1', 'node=2').replace('tccd_stats_version=2', 'tccd_stats_version=1'))
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('incompatible tCCD statistics versions', process.stderr)


if __name__ == '__main__':
    unittest.main()
