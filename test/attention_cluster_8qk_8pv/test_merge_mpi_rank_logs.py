import tempfile
from pathlib import Path
import unittest

from merge_mpi_rank_logs import merge


class RankLogTest(unittest.TestCase):
    def test_resource_sidecar_preserves_rank_and_control_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rank = root / '1' / 'rank.0'
            rank.mkdir(parents=True)
            (rank / 'stdout').write_text(
                '[RESOURCE_TIMEBASE] ticks_per_cycle=1000\n'
                '[RESOURCE_INTERVAL] request=1\n'
                '[CONTROL_EVENT] stage=projection_accept\n')
            output = root / 'runtime.log'
            output.write_text('raw console\n')
            merge(root, output, 1)
            self.assertNotIn('[RESOURCE_INTERVAL]', output.read_text())
            self.assertIn('[RESOURCE_TIMEBASE]', output.read_text())
            self.assertIn('[CONTROL_EVENT]', output.read_text())
            self.assertEqual(output.with_suffix('.resources.jsonl').read_text(),
                             '[RESOURCE_INTERVAL] request=1\n')
            self.assertIn('[RESOURCE_INTERVAL]', (rank / 'stdout').read_text())

    def test_independent_streams_preserve_split_record_and_raw_console(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rank, text in enumerate(('core=4 cycles=100\n', 'core=5 cycles=200\n')):
                p = root / '1' / f'rank.{rank}'
                p.mkdir(parents=True)
                (p / 'stdout').write_text(text)
            output = root / 'runtime.log'
            output.write_text('core=4 cycles=core=5 cycles=200\n100\n')
            merge(root, output, 2)
            self.assertEqual(output.read_text(), 'core=4 cycles=100\ncore=5 cycles=200\n')
            self.assertIn('cycles=core=5', output.with_suffix('.mpi-console-raw').read_text())

    def test_missing_rank_does_not_replace_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'runtime.log'
            output.write_text('original')
            with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
                merge(root, output, 4)
            self.assertEqual(output.read_text(), 'original')

    def test_duplicate_rank_stream_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for job in ('1', '2'):
                p = root / job / 'rank.0'
                p.mkdir(parents=True)
                (p / 'stdout').write_text('record\n')
            with self.assertRaisesRegex(ValueError, 'missing or duplicate'):
                merge(root, root / 'runtime.log', 2)


if __name__ == '__main__':
    unittest.main()
