import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PATH = Path(__file__).with_name("attention_cluster.py")
SPEC = importlib.util.spec_from_file_location("attention_cluster", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AttentionClusterTest(unittest.TestCase):
    def args(self):
        return argparse.Namespace(
            query_length=1024, kv_length=1024,
            num_query_heads=4, num_kv_heads=2, head_dim=128,
            hbm_bytes_per_cycle=1280, local_bytes_per_cycle=256,
            broadcast_bytes_per_cycle=256, cluster_link_bytes_per_cycle=256,
            sfu_window_cycles=3064,
        )

    def test_role_partition_and_reuse_windows(self):
        events, result = MODULE.build_schedule(self.args())
        self.assertEqual(result["clusters"]["qk_softmax_workers"], [4, 5, 6, 7])
        self.assertEqual(result["clusters"]["pv_workers"], list(range(8, 20)))
        qk = [event for event in events if event["stage"] == "qk_gemm"]
        pv = [event for event in events if event["stage"] == "pv_gemm"]
        self.assertTrue(all(event["reuse_window"] == [2, 4] for event in qk))
        self.assertTrue(all(event["reuse_window"][1] == 2 for event in pv))
        self.assertEqual({event["worker"] for event in qk}, set(range(4)))
        self.assertEqual({event["worker"] for event in pv}, set(range(12)))
        self.assertEqual({event["worker_core"] for event in qk}, set(range(4, 8)))
        self.assertEqual({event["worker_core"] for event in pv}, set(range(8, 20)))

    def test_no_score_or_probability_hbm_traffic(self):
        _, result = MODULE.build_schedule(self.args())
        self.assertEqual(result["hbm"]["score_probability_bytes"], 0)
        self.assertEqual(result["hbm"]["total_bytes"], 6 * 1024 * 1024)
        self.assertEqual(result["hbm_floor_cycles"], 4916)

    def test_online_softmax_matches_full_attention(self):
        rng = np.random.default_rng(9)
        q = (rng.normal(size=(4, 128, 128)) * 0.2).astype(np.float32)
        k = (rng.normal(size=(2, 256, 128)) * 0.2).astype(np.float32)
        v = (rng.normal(size=(2, 256, 128)) * 0.2).astype(np.float32)
        actual = MODULE.cluster_attention(q, k, v)
        expected = MODULE.full_attention(q, k, v)
        self.assertLessEqual(float(np.max(np.abs(actual - expected))), 3e-5)

    def test_qk_and_pv_overlap(self):
        events, result = MODULE.build_schedule(self.args())
        first_pv = min(event["start"] for event in events if event["stage"] == "pv_gemm")
        last_qk = result["stage_last_cycle"]["qk"]
        self.assertLess(first_pv, last_qk)

    def test_two_score_slots_apply_backpressure(self):
        events, _ = MODULE.build_schedule(self.args())
        for worker in range(4):
            qk = sorted(
                (event for event in events
                 if event["stage"] == "qk_gemm" and event["worker"] == worker),
                key=lambda event: event["end"],
            )
            sfu_by_slot = {
                (event["kv_head"], event["key_window"], event["row_window"]): event
                for event in events
                if event["stage"] == "online_softmax" and event["worker"] == worker
            }
            prior_release = [0, 0]
            for event in qk:
                slot = event["score_fifo_slot"]
                self.assertGreaterEqual(event["end"], prior_release[slot])
                key = (event["kv_head"], event["key_window"], event["row_window"])
                prior_release[slot] = sfu_by_slot[key]["end"]


if __name__ == "__main__":
    unittest.main()
