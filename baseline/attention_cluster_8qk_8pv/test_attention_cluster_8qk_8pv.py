import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODEL = Path(__file__).with_name("attention_cluster_model.py")
WCP_SOURCE = Path(__file__).parents[2] / "src/sst/elements/golem/workercmdproc/workercmdproc.h"
RUNNER_SOURCE = Path(__file__).with_name("run_sst.sh")
REPORT_SOURCE = Path(__file__).with_name("report_sst.py")
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
    def test_sst_report_accepts_one_pv_lane_per_manager(self):
        lines = [
            "[Core 12] [wcp] PV_V_RESIDENCY hit=1 source=100 local_addr=200",
            "[Core 13] [wcp] PV_V_RESIDENCY hit=0 source=300 local_addr=400",
        ]
        for manager in range(4):
            for qk_column in range(2):
                qk_core = 4 + qk_column * 4 + manager
                lines.extend((
                    f"GOLEM_SFU_HW_PIPELINE core={qk_core} unit=exp lanes=16 "
                    "latency=8 ii=1 depth=16 accepted_tokens=2048",
                    f"[ATTENTION_REUSE_WINDOW_QK] core={qk_core} cycles=11 "
                    "start=10 end=20 fusion_tiles=4 expected_tiles=4",
                    f"[ATTENTION_WORKER_CLUSTER_SFU] core={qk_core} "
                    "start=18 end=30 cycles=13",
                    f"[ATTENTION_WORKER_CLUSTER_E2E] core={qk_core} "
                    "start=10 end=50 cycles=41",
                ))
                for window in range(2):
                    lines.append(
                        f"[ATTENTION_WORKER_CLUSTER_PV] core={12 + manager} "
                        f"qk_core={qk_core} row=0 window={window} "
                        "cycles=21 start=25 end=45"
                    )
        with tempfile.TemporaryDirectory() as temp_dir:
            log = Path(temp_dir) / "sst.log"
            result = Path(temp_dir) / "report.json"
            log.write_text("\n".join(lines), encoding="utf-8")
            subprocess.run(
                [sys.executable, str(REPORT_SOURCE), "--log", str(log),
                 "--num-query-heads", "1", "--num-kv-heads", "1",
                 "--query-length", "512", "--kv-length", "512",
                 "--head-dim", "64",
                 "--qk-workers-per-manager", "2", "--output", str(result)],
                check=True, capture_output=True, text=True,
            )
            report = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["pv_cores"], [12, 13, 14, 15])
            self.assertEqual(report["shape"]["head_dim"], 64)
            self.assertFalse(report["shape"]["causal"])
            self.assertEqual(report["v_cache"], {"hits": 1, "misses": 1})
            self.assertEqual(report["pv_service"]["average_window_cycles"], 21)
            self.assertEqual(report["pv_service"]["core_busy_cycles"]["12"], 84)
            self.assertEqual(report["pv_tail_after_sfu_cycles"], 20)
            self.assertTrue(report["sfu_exp_work"]["complete"])
            log.write_text("\n".join(lines).replace(
                "accepted_tokens=2048", "accepted_tokens=2047", 1), encoding="utf-8")
            failed = subprocess.run(
                [sys.executable, str(REPORT_SOURCE), "--log", str(log),
                 "--num-query-heads", "1", "--num-kv-heads", "1",
                 "--query-length", "512", "--kv-length", "512",
                 "--head-dim", "64", "--qk-workers-per-manager", "2",
                 "--output", str(result)], capture_output=True, text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse(json.loads(result.read_text())["sfu_exp_work"]["complete"])

    def test_sst_report_includes_overlapping_stage_spans(self):
        lines = []
        for manager in range(4):
            qk_core = 4 + manager
            lines.extend((
                f"[ATTENTION_REUSE_WINDOW_QK] core={qk_core} cycles=11 "
                f"start=10 end=20 fusion_tiles=4 expected_tiles=4",
                f"[ATTENTION_WORKER_CLUSTER_SFU] core={qk_core} "
                "start=18 end=30 cycles=13",
                f"[ATTENTION_WORKER_CLUSTER_E2E] core={qk_core} "
                "start=10 end=50 cycles=41",
            ))
            for row in range(8):
                for window in range(4):
                    pv_core = 8 + manager + (row % 3) * 4
                    lines.append(
                        f"[ATTENTION_WORKER_CLUSTER_PV] core={pv_core} "
                        f"qk_core={qk_core} row={row} window={window} "
                        "cycles=21 start=25 end=45 p_ready=20 pv_send=21 "
                        "pv_receive=22 pv_start=30"
                    )
        with tempfile.TemporaryDirectory() as temp_dir:
            log = Path(temp_dir) / "sst.log"
            result = Path(temp_dir) / "report.json"
            log.write_text("\n".join(lines), encoding="utf-8")
            command = [sys.executable, str(REPORT_SOURCE), "--log", str(log),
                       "--num-query-heads", "4", "--num-kv-heads", "1",
                       "--query-length", "512", "--kv-length", "1024",
                       "--output", str(result)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            report = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["stage_spans"]["qk"]["elapsed_cycles"], 10)
            self.assertEqual(report["stage_spans"]["sfu"]["elapsed_cycles"], 12)
            self.assertEqual(report["stage_spans"]["pv"]["count"], 128)
            self.assertEqual(report["end_to_end_cycles"], 40)
            self.assertEqual(
                report["pv_transport_waits"]["receive_to_start"]["average_cycles"],
                8,
            )

            log.write_text("\n".join(line for line in lines
                                     if "[ATTENTION_WORKER_CLUSTER_SFU]" not in line),
                           encoding="utf-8")
            failed = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)

    def test_sst_report_counts_short_kv_windows(self):
        script = REPORT_SOURCE.read_text(encoding="utf-8")
        self.assertIn("args.kv_length // 256", script)
        self.assertIn("group_size * args.query_length", script)

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
            "nextReuseN + 1u < currentReuseNCount_",
            source,
        )
        self.assertIn(
            "nextTile = 0;",
            source,
        )
        self.assertIn("nextKBegin = nextPrefetchK_;", source)
        self.assertIn("current transaction is still computing", source)

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
