import tempfile
import unittest
from pathlib import Path

from sweep_attention import load_rope_metrics, static_causal_exp_floor


class CausalRopeSweepTest(unittest.TestCase):
    def test_static_exp_floor_respects_fixed_worker_ownership(self):
        self.assertEqual(static_causal_exp_floor(1, 1, 2048), 31232)
        self.assertEqual(static_causal_exp_floor(2, 1, 1024), 14848)
        self.assertEqual(static_causal_exp_floor(32, 8, 2048), 933888)

    def test_rope_metrics_parse_worker_and_hbm_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats_dir = root / "stats" / "run"
            stats_dir.mkdir(parents=True)
            (stats_dir / "stats_selfcom_0.txt").write_text(
                "core4:rocc:sfu,sfu_vector_ops,,Accumulator,0,0,3,3,3,1,1\n"
                "core4:rocc:sfu,sfu_vector_elems,,Accumulator,0,0,12288,0,3,4096,4096\n"
            )
            logs_dir = root / "logs"
            logs_dir.mkdir()
            (logs_dir / "attention-sst.log").write_text(
                "[ATTENTION_MILESTONE] stage=worker_dispatch_accept status=done "
                "sst_tick=1000 rocc_cycle=1 core=4 role=worker\n"
                "[ATTENTION_ROPE_TABLE] core=4 bytes=2048 cycles=10\n"
                "[ATTENTION_ROPE_TABLE_CACHE] core=4 bytes=2048\n"
                "[ATTENTION_MILESTONE] stage=worker_complete status=done "
                "sst_tick=16000 rocc_cycle=16 core=4 role=worker\n"
            )
            metrics = load_rope_metrics(root)
            self.assertEqual(metrics["vector_ops"], 3)
            self.assertEqual(metrics["vector_elements"], 12288)
            self.assertEqual(metrics["rope_table_loads"], 1)
            self.assertEqual(metrics["rope_table_cache_hits"], 1)
            self.assertEqual(metrics["rope_table_bytes"], 2048)
            self.assertEqual(metrics["rope_table_load_max_cycles"], 10)
            self.assertEqual(metrics["worker_dispatch_to_complete_cycles"], 15)


if __name__ == "__main__":
    unittest.main()
