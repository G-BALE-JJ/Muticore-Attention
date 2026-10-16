import json
from pathlib import Path
import tempfile
import unittest

from analyze_decoder_weights import analyze


class WeightLifetimeAuditTests(unittest.TestCase):
    def fixture(self, root):
        (root/'logs').mkdir()
        lines=[]
        for core in (4,8,12,16):
            lines.append(f'[DECODER_PREFETCH_FLOW] core={core} stage=4 epoch=1 enabled=1 slots=4 requests=2 hits=2 wait_total=5 wait_dma=3 wait_gm=2 wait_program=0 wait_array=0 dma_sum=20 gm_sum=4 program_sum=6 compute_sum=8\n')
            for n in range(2):
                start=100+n*50+core
                lines.append(f'[DECODER_WEIGHT_TILE] core={core} stage=4 epoch=1 row=0 path=0 n={n} k=0 address={8192*(n+1)} submit={start} dma_done={start+10} gm_done={start+12} consume={start+13}\n')
        path=root/'logs/attention-sst.log';path.write_text(''.join(lines));return path

    def test_valid_trace_partitions_stalls_and_replays_bounded_byte_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);r=analyze(root)
            self.assertEqual(r['tiles'],8)
            self.assertEqual(r['latency_cycles']['HBM_DMA']['p95'],10)
            self.assertEqual(r['bounded_reuse_replay']['2']['optimistic_read_saved_percent'],75)
            self.assertIn('no simulated cycle speedup',r['replay_scope'])

    def test_rejects_unfinished_dma_missing_trace_or_bad_latency_sums(self):
        for old,new in (('consume=117','consume=110'),('requests=2','requests=3'),
                        ('dma_sum=20','dma_sum=19'),('wait_total=5','wait_total=6')):
            with self.subTest(old=old),tempfile.TemporaryDirectory() as d:
                root=Path(d);path=self.fixture(root);path.write_text(path.read_text().replace(old,new))
                with self.assertRaises(ValueError):analyze(root)

    def test_down_panel_local_k_does_not_alias_global_weight_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=self.fixture(root)
            path.write_text(path.read_text().replace('n=1 k=0','n=0 k=0'))
            self.assertEqual(analyze(root)['tiles'],8)
            path.write_text(path.read_text().replace('address=16384','address=8192'))
            with self.assertRaisesRegex(ValueError,'duplicate weight'):analyze(root)


if __name__=='__main__':unittest.main()
