import argparse
import importlib.util
import json
import sys
import unittest
from pathlib import Path


MODEL = Path(__file__).parents[1] / "attention_cluster" / "attention_cluster.py"
WCP_SOURCE = Path(__file__).parents[2] / "src/sst/elements/golem/workercmdproc/workercmdproc.h"
RUNNER_SOURCE = Path(__file__).with_name("run_sst.sh")
GOLDEN_RESULT = Path(__file__).parent / "artifacts/golden/sst_result.json"
GOLDEN_MANIFEST = Path(__file__).parent / "artifacts/golden/manifest.json"
COMMON_RUNNER_SOURCE = (
    Path(__file__).parents[2]
    / "src/sst/elements/golem/tests/small/muticore_attention/run_fused_attention_scale.sh"
)
SPEC = importlib.util.spec_from_file_location("attention_cluster_8qk_model", MODEL)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AttentionCluster8Qk8PvTest(unittest.TestCase):
    def test_stable_defaults_match_archived_golden_result(self):
        runner = RUNNER_SOURCE.read_text(encoding="utf-8")
        result = json.loads(GOLDEN_RESULT.read_text(encoding="utf-8"))
        manifest = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))

        self.assertIn(
            'GOLEM_ATTENTION_WORKER_CLUSTER_DYNAMIC_PV:-0', runner)
        self.assertIn(
            'GOLEM_ATTENTION_WORKER_CLUSTER_ROW_PRIORITY:-1', runner)
        self.assertIn(
            'GOLEM_ATTENTION_WORKER_CLUSTER_V_BROADCAST:-1', runner)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["end_to_end_cycles"], 47193)
        self.assertEqual(result["qk_worker_count"], 8)
        self.assertEqual(result["pv_worker_count"], 8)
        self.assertEqual(result["pv_windows"], 256)
        self.assertEqual(
            manifest["measurement"]["theoretical_resource_floor_cycles"],
            32768,
        )
        self.assertFalse(manifest["stable_policy"]["dynamic_pv"])

    def test_pv_panel_pipeline_uses_full_four_tile_window(self):
        runner = RUNNER_SOURCE.read_text(encoding="utf-8")
        common_runner = COMMON_RUNNER_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            'GOLEM_DMA_WINDOW_K_TILES="${GOLEM_DMA_WINDOW_K_TILES:-4}"',
            runner,
        )
        self.assertIn(
            'GOLEM_DMA_WINDOW_K_TILES=${GOLEM_DMA_WINDOW_K_TILES:-',
            common_runner,
        )

    def test_pv_panel_prefetch_crosses_reuse_n_boundary(self):
        source = WCP_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "nextReuseN + 1u >= currentReuseNCount_",
            source,
        )
        self.assertIn("nextReuseN += 1u;", source)
        self.assertIn(
            "nextTile = 0;",
            source,
        )

    def test_pv_resident_p_bypasses_matrix_dma(self):
        source = WCP_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "txn.skipMatRead = usesAttentionPvPanels() ||",
            source,
        )

    def test_role_split_and_resource_floor(self):
        args = argparse.Namespace(
            query_length=1024, kv_length=1024,
            num_query_heads=4, num_kv_heads=2, head_dim=128,
            qk_workers_per_manager=2,
            hbm_bytes_per_cycle=1280, local_bytes_per_cycle=256,
            broadcast_bytes_per_cycle=256, cluster_link_bytes_per_cycle=256,
            sfu_window_cycles=3064,
        )
        events, result = MODULE.build_schedule(args)
        self.assertEqual(result["clusters"]["qk_softmax_workers"], list(range(4, 12)))
        self.assertEqual(result["clusters"]["pv_workers"], list(range(12, 20)))
        self.assertEqual(
            result["compute_floor_cycles"]["perfect_pipeline"], 32768)
        self.assertEqual(
            {event["worker_core"] for event in events if event["stage"] == "qk_gemm"},
            set(range(4, 12)),
        )
        self.assertEqual(
            {event["worker_core"] for event in events if event["stage"] == "pv_gemm"},
            set(range(12, 20)),
        )


if __name__ == "__main__":
    unittest.main()
