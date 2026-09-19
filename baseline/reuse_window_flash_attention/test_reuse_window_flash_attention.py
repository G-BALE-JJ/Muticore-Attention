import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("reuse_window_flash_attention.py")
SPEC = importlib.util.spec_from_file_location("reuse_window_flash_attention", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ReuseWindowFlashAttentionTest(unittest.TestCase):
    def args(self):
        return argparse.Namespace(
            query_length=1024, kv_length=1024,
            num_query_heads=4, num_kv_heads=2, head_dim=128,
            window_m_tiles=2, window_n_tiles=4,
            hbm_nodes=2, hbm_node_bytes_per_cycle=320,
            prefetch_windows=2, local_slots=24, c_buffer_bytes=1024 * 1024,
        )

    def test_gqa_mapping_and_window_contract(self):
        args = self.args()
        schedule = MODULE.build_schedule(args)
        qk = [entry for entry in schedule if entry["op"] == "qk"]
        self.assertEqual(len(qk), 4 * 8 * 4)
        self.assertEqual([qk[h * 32]["kv_head"] for h in range(4)], [0, 0, 1, 1])
        self.assertEqual(len({entry["gemm_submission"] for entry in qk}), 4)
        self.assertEqual(qk[0]["logical_gemm"], {"m": 1024, "n": 1024, "k": 128})
        self.assertEqual(sorted({entry["worker_core"] for entry in qk[:32]}), list(range(16)))
        self.assertEqual(qk[0]["gemm"], {"m": 128, "n": 256, "k": 128})
        self.assertEqual(qk[0]["wcp"]["a_reuse_n_tiles"], 4)
        self.assertEqual(qk[0]["wcp"]["b_reuse_m_tiles"], 2)

    def test_two_node_supply_exceeds_compute_floor_demand(self):
        args = self.args()
        traffic = MODULE.traffic_and_bounds(args, MODULE.build_schedule(args))
        proof = traffic["supply_proof_per_gqa_group"]
        self.assertAlmostEqual(proof["required_bytes_per_cycle"], 384.0)
        self.assertEqual(proof["available_bytes_per_cycle"], 640)
        self.assertTrue(proof["supply_keeps_up"])
        self.assertEqual(traffic["physical_hbm"]["score_probability_bytes"], 0)

    def test_windowed_online_softmax_matches_full_attention(self):
        rng = np.random.default_rng(11)
        q = rng.normal(size=(128, 128)).astype(np.float32) * 0.1
        k = rng.normal(size=(256, 128)).astype(np.float32) * 0.1
        v = rng.normal(size=(256, 128)).astype(np.float32) * 0.1
        actual = MODULE.online_window_attention(q, k, v, 128, 128)
        expected = MODULE.full_attention(q, k, v)
        self.assertLessEqual(float(np.max(np.abs(actual - expected))), 2e-5)

    def test_group_of_four_runs_one_full_gemm_at_a_time(self):
        args = self.args()
        args.num_query_heads = 8
        schedule = MODULE.build_schedule(args)
        group_zero_qk = [
            entry for entry in schedule
            if entry["op"] == "qk" and entry["kv_head"] == 0
        ]
        submissions = [
            {entry["gemm_submission"] for entry in group_zero_qk if entry["query_head"] == head}
            for head in range(4)
        ]
        self.assertTrue(all(len(item) == 1 for item in submissions))
        self.assertEqual(len(set.union(*submissions)), 4)
        self.assertLess(max(entry["worker_core"] for entry in group_zero_qk), 16)
        proof = MODULE.traffic_and_bounds(args, schedule)["supply_proof_per_gqa_group"]
        self.assertAlmostEqual(proof["required_bytes_per_cycle"], 320.0)

    def test_larger_head_dimension_reports_required_resources(self):
        args = self.args()
        args.head_dim = 256
        args.local_slots = 48
        MODULE.validate(args)
        traffic = MODULE.traffic_and_bounds(args, MODULE.build_schedule(args))
        self.assertEqual(traffic["wcp_capacity"]["qk_required_local_slots"], 48)
        self.assertEqual(traffic["wcp_capacity"]["pv_required_local_slots"], 48)
        self.assertTrue(traffic["wcp_capacity"]["fits"])


if __name__ == "__main__":
    unittest.main()
