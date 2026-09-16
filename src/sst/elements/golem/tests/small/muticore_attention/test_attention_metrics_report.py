import csv
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest


HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from report_attention_metrics import build_report
from verify_fused_attention_scale_stats import read_model_clocks_from_run_config


RUNNER = HERE / "run_fused_attention_scale.sh"
REPORTER = HERE / "report_attention_metrics.py"
PROGRESS_AWK = HERE / "attention_progress.awk"
TERMINAL_PROGRESS_AWK = HERE / "attention_terminal_progress.awk"
TERMINAL_UI = HERE / "attention_terminal_ui.sh"
ROCC_SOURCE = HERE.parents[2] / "rocc" / "roccAnalog.h"


def synthetic_lifecycle():
    return {
        "status": "PASS",
        "mismatches": {},
        "lifecycle": {
            "clock_contract": {
                "sst_timebase_ticks_per_second": 1_000_000_000_000,
                "normalization_clock_hz": 1_000_000_000,
                "normalized_cycle_definition": "synthetic",
                "normalized_cycles_are_model_native_cycles": False,
            },
            "accelerator_completion_ticks": 10_000,
            "accelerator_completion_cycles": 10,
            "accelerator_completion_milliseconds": 0.00001,
            "wait_return_ticks": 11_000,
            "wait_return_cycles": 11,
            "wait_return_milliseconds": 0.000011,
            "manager_descriptor_accept_skew_cycles": 1,
            "manager_local_complete_skew_cycles": 2,
            "system_frontier": {
                "milestone_ticks": {
                    "root_descriptor_accept": 100,
                    "manager_dispatch_complete": 1100,
                    "root_tensor_complete": 10100,
                    "software_wait_observed": 11100,
                },
                "stage_ticks": {
                    "root_descriptor_accept_to_manager_dispatch_complete": 1000,
                    "manager_dispatch_complete_to_root_tensor_complete": 9000,
                    "root_tensor_complete_to_software_wait_observed": 1000,
                },
                "stage_cycles": {
                    "root_descriptor_accept_to_manager_dispatch_complete": 1,
                    "manager_dispatch_complete_to_root_tensor_complete": 9,
                    "root_tensor_complete_to_software_wait_observed": 1,
                },
                "accelerator_attribution": {
                    "total_ticks": 10_000,
                    "attributed_ticks": 10_000,
                    "unattributed_ticks": 0,
                    "coverage_ratio": 1.0,
                    "conservation_valid": True,
                },
                "order_valid": True,
            },
            "worker_critical_path": {
                "slowest_worker_core": 19,
                "stage_ticks": {"dispatch_to_final_qk": 7000},
                "stage_cycles": {"dispatch_to_final_qk": 7},
                "aggregate_online_pipeline_cycles": {
                    "dispatch_to_first_qk": 3,
                },
                "tile_pipeline_breakdown": {
                    "total_cycles": 100,
                    "phase_cycles": {
                        "kv_load": 5,
                        "q_local_read": 2,
                        "qk_matrix_program": 11,
                        "qk_input_program": 3,
                        "qk_compute_readout": 7,
                        "softmax": 4,
                        "pv_matrix_program": 31,
                        "pv_input_program": 9,
                        "pv_restore_output": 6,
                        "pv_compute": 8,
                        "pv_output_readwrite": 14,
                    },
                },
                "kv_prefetch_timing": {
                    "ticks": {"dma": 9000, "ready_lead": 3000, "wait": 2000},
                    "cycles": {"dma": 9, "ready_lead": 3, "wait": 2},
                    "counts": {"dma": 8, "ready_lead": 5, "wait": 3},
                },
                "order_valid": True,
            },
            "kv_second_lookahead_window": {
                "max_candidates": 16,
                "candidates": 12,
                "prefetches": 12,
                "candidate_rate": 0.75,
                "ready_at_release": 10,
                "ready_after_release_before_boundary": 2,
                "mean_cycles": {"available_lead": 4.5},
                "max_available_lead_cycles": 7,
            },
            "wcp_gemm_proxy": {
                "queue_depth": 32,
                "issue_width": 1,
                "command_latency_cycles": 1,
                "completion_latency_cycles": 1,
                "max_worker_queue_wait_cycles": 16,
                "worker_totals": {
                    "gemm_proxy_commands_issued": 128,
                    "gemm_proxy_queue_full_stalls": 0,
                    "gemm_proxy_queue_wait_cycles": 2048,
                    "gemm_proxy_launch_commands": 2048,
                    "gemm_proxy_completion_callbacks": 128,
                    "gemm_proxy_completion_delay_cycles": 128,
                },
            },
            "qk_score_row_burst": {
                "enabled": True,
                "mode": "ATTENTION_TILE_STORAGE",
                "capacity_bytes": 1024,
                "banks": 16,
                "bank_bytes_per_cycle": 64,
                "row_burst_bytes": 64,
                "worker_totals": {
                    "attention_tile_storage_acquires": 256,
                    "attention_tile_storage_releases": 256,
                    "attention_tile_storage_mode_conflicts": 0,
                    "attention_tile_storage_capacity_rejections": 0,
                    "attention_tile_storage_column_writes": 4096,
                    "attention_tile_storage_row_reads": 4096,
                    "attention_tile_storage_write_bytes": 262144,
                    "attention_tile_storage_read_bytes": 262144,
                    "attention_tile_storage_write_wait_cycles": 0,
                    "attention_tile_storage_read_wait_cycles": 0,
                },
            },
            "matrix_broadcast_fabric": {
                "enabled": True,
                "pv_enabled": True,
                "qk_enabled": False,
                "topology": "binary_tree",
                "shares_array_buffer_ports": True,
                "max_fanout": 16,
                "bytes_per_cycle": 64,
                "base_latency_cycles": 1,
                "stage_latency_cycles": 1,
                "tree_stages": 4,
                "fanout": 16,
                "payload_bytes": 8192,
                "cycles_per_request": 133,
                "max_observed_fanout": 16,
                "worker_totals": {
                    "matrix_broadcast_requests": 16384,
                    "matrix_broadcast_rejected": 0,
                    "matrix_broadcast_ingress_bytes": 134217728,
                    "matrix_broadcast_sink_bytes": 2147483648,
                    "matrix_broadcast_transfer_cycles": 2179072,
                    "matrix_broadcast_fanout": 262144,
                },
            },
            "pv_active_k": {
                "enabled": True,
                "active_columns": 32,
                "mac_per_cu_per_cycle": 1.0,
                "pipeline_depth": 2,
                "worker_totals": {
                    "active_k_launches": 2048,
                    "active_k_columns": 65536,
                    "active_k_compute_cycles": 69632,
                    "active_k_full_width_cycles_avoided": 196608,
                },
            },
            "pv_v_tile_buffer": {
                "enabled": True,
                "capacity_bytes": 16384,
                "bytes_per_cycle": 64,
                "base_latency_cycles": 1,
                "panel_bytes": 2048,
                "worker_totals": {
                    "hits": 1536,
                    "misses": 512,
                    "bytes_read": 8388608,
                    "bytes_reused": 3145728,
                    "wait_ticks": 50688,
                    "capacity_rejections": 0,
                },
            },
        },
    }


def synthetic_numerical_result():
    return {
        "status": "PASS",
        "checked": 1024,
        "mismatches": 0,
        "max_abs_error": 1.0e-9,
    }


def synthetic_mpi_result(status="PASS"):
    return {
        "status": status,
        "mpi_ranks": 2,
        "missing_rank_files": [],
        "file_rank_mismatches": [],
    }


class AttentionMetricsReportTest(unittest.TestCase):
    def test_corrupt_lifecycle_still_writes_structured_fail_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lifecycle = root / "lifecycle.json"
            numerical = root / "numerical.json"
            output = root / "metrics.json"
            lifecycle.write_text("{broken", encoding="ascii")
            numerical.write_text(
                json.dumps(synthetic_numerical_result()), encoding="ascii"
            )
            result = subprocess.run(
                [
                    "python3", str(REPORTER), "--lifecycle-json", str(lifecycle),
                    "--numerical-json", str(numerical), "--profile", "corrupt",
                    "--mpi-ranks", "1", "--sst-wall-seconds", "1",
                    "--pipeline-wall-seconds", "2", "--generic-gemm",
                    "--output-json", str(output), "--output-csv", str(root / "m.csv"),
                ],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Status          FAIL", result.stdout)
            self.assertNotIn("Traceback", result.stderr)
            report = json.loads(output.read_text(encoding="ascii"))
            self.assertEqual(report["source_status"]["lifecycle"], "FAIL")
            self.assertIn("lifecycle", report["source_errors"])

    def test_baseline_failure_is_part_of_overall_status(self):
        report = build_report(
            synthetic_lifecycle(), synthetic_numerical_result(), profile="custom",
            mpi_ranks=1, sst_wall_seconds=1.0, pipeline_wall_seconds=2.0,
            generic_gemm=True, baseline_result={"status": "FAIL"},
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["source_status"]["baseline"], "FAIL")

    def test_build_report_preserves_time_domains_and_stage_cycles(self):
        report = build_report(
            synthetic_lifecycle(),
            synthetic_numerical_result(),
            profile="e3",
            mpi_ranks=2,
            sst_wall_seconds=12.5,
            pipeline_wall_seconds=14.0,
            generic_gemm=True,
            mpi_partition_result=synthetic_mpi_result(),
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["configuration"]["profile"], "e3")
        self.assertEqual(report["configuration"]["mpi_ranks"], 2)
        self.assertTrue(report["configuration"]["wcp_enabled"])
        self.assertEqual(report["host_timing"]["sst_wall_seconds"], 12.5)
        self.assertEqual(report["host_timing"]["pipeline_wall_seconds"], 14.0)
        self.assertEqual(report["simulated_timing"]["total_cycles"], 10)
        self.assertEqual(report["simulated_timing"]["wait_return_cycles"], 11)
        self.assertFalse(
            report["time_domains"]["normalized_cycles_are_model_native_cycles"]
        )
        self.assertEqual(
            report["system_stages"][0]["duration_cycles"], 1
        )
        self.assertEqual(report["critical_worker"]["core"], 19)
        self.assertEqual(
            report["critical_worker"]["kv_prefetch_timing"]["counts"]["wait"],
            3,
        )
        self.assertEqual(report["kv_second_lookahead_window"]["prefetches"], 12)
        self.assertEqual(
            report["wcp"]["worker_totals"]["gemm_proxy_commands_issued"], 128
        )
        self.assertTrue(report["configuration"]["pv_matrix_broadcast"])
        self.assertEqual(
            report["matrix_broadcast_fabric"]["clock_domain"],
            "mvm_array_component_cycles",
        )
        self.assertEqual(report["pv_active_k"]["active_columns"], 32)
        self.assertEqual(
            report["pv_active_k"]["worker_totals"][
                "active_k_full_width_cycles_avoided"
            ],
            196608,
        )
        self.assertEqual(report["pv_v_tile_buffer"]["capacity_bytes"], 16384)
        self.assertEqual(report["numerical_verification"]["mismatches"], 0)

    def test_cli_prints_summary_and_writes_json_and_long_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lifecycle_path = root / "lifecycle.json"
            numerical_path = root / "numerical.json"
            mpi_path = root / "mpi.json"
            report_path = root / "attention_metrics.json"
            csv_path = root / "attention_metrics.csv"
            lifecycle_path.write_text(
                json.dumps(synthetic_lifecycle()), encoding="ascii"
            )
            numerical_path.write_text(
                json.dumps(synthetic_numerical_result()), encoding="ascii"
            )
            mpi_path.write_text(
                json.dumps(synthetic_mpi_result()), encoding="ascii"
            )

            result = subprocess.run(
                [
                    "python3", str(REPORTER),
                    "--lifecycle-json", str(lifecycle_path),
                    "--numerical-json", str(numerical_path),
                    "--mpi-partition-json", str(mpi_path),
                    "--profile", "e3",
                    "--mpi-ranks", "2",
                    "--sst-wall-seconds", "12.5",
                    "--pipeline-wall-seconds", "14.0",
                    "--generic-gemm",
                    "--output-json", str(report_path),
                    "--output-csv", str(csv_path),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("== RESULT ==", result.stdout)
            self.assertIn("Configuration", result.stdout)
            self.assertNotIn("Simulated", result.stdout)
            self.assertIn("Total cycles", result.stdout)
            self.assertIn("Wait return", result.stdout)
            self.assertIn("Numerical", result.stdout)
            self.assertIn("Critical-worker operator cycles", result.stdout)
            self.assertIn("K/V prefetch", result.stdout)
            self.assertIn("N+2 lookahead", result.stdout)
            self.assertIn("Input movement", result.stdout)
            self.assertRegex(result.stdout, r"Input movement\s+7")
            self.assertRegex(result.stdout, r"QK\s+21")
            self.assertRegex(result.stdout, r"Softmax\s+4")
            self.assertRegex(result.stdout, r"PV\s+68")
            self.assertNotIn("Key stage cycles", result.stdout)
            self.assertNotRegex(result.stdout, r"(?m)^\s+dispatch\s+")
            self.assertNotRegex(result.stdout, r"(?m)^\s+finalize\s+")
            self.assertIn("WCP GEMM proxy", result.stdout)
            self.assertIn("QK row burst", result.stdout)
            self.assertIn("row bursts=4,096", result.stdout)
            self.assertIn("PV broadcast", result.stdout)
            self.assertIn("binary tree", result.stdout)
            self.assertIn("Broadcast work", result.stdout)
            self.assertIn("PV active-K", result.stdout)
            self.assertIn("active=32 columns", result.stdout)
            self.assertIn("PV V-tile", result.stdout)
            self.assertIn("hits=1,536", result.stdout)
            self.assertIn("RoCC component-cycle sum", result.stdout)
            self.assertNotIn("\x1b[", result.stdout)
            self.assertEqual(json.loads(report_path.read_text())["status"], "PASS")
            with csv_path.open(newline="", encoding="ascii") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                rows[0].keys(), {"scope", "metric", "value", "unit"}
            )
            metrics = {(row["scope"], row["metric"]) for row in rows}
            self.assertIn(("summary", "total_cycles"), metrics)
            self.assertIn(
                (
                    "system_stage",
                    "root_descriptor_accept_to_manager_dispatch_complete",
                ),
                metrics,
            )
            self.assertIn(("wcp", "gemm_proxy_commands_issued"), metrics)
            self.assertIn(
                ("qk_score_row_burst", "attention_tile_storage_row_reads"),
                metrics,
            )
            self.assertIn(("kv_prefetch", "wait_cycles"), metrics)
            self.assertIn(("kv_second_lookahead", "prefetches"), metrics)
            self.assertIn(
                ("matrix_broadcast", "matrix_broadcast_sink_bytes"), metrics
            )
            self.assertIn(
                ("pv_active_k", "active_k_full_width_cycles_avoided"), metrics
            )
            self.assertIn(("pv_v_tile_buffer", "wait_ticks"), metrics)
            units = {
                (row["scope"], row["metric"]): row["unit"] for row in rows
            }
            self.assertEqual(
                units[("wcp", "gemm_proxy_commands_issued")], "count"
            )
            self.assertEqual(
                units[("wcp", "gemm_proxy_queue_wait_cycles")],
                "wcp_component_cycles",
            )
            self.assertEqual(
                units[("matrix_broadcast", "matrix_broadcast_transfer_cycles")],
                "array_component_cycles",
            )
            self.assertEqual(
                units[("pv_active_k", "active_k_compute_cycles")],
                "array_component_cycles",
            )
            self.assertEqual(
                units[("pv_active_k", "active_k_columns")], "columns"
            )
            self.assertEqual(
                units[("pv_v_tile_buffer", "wait_ticks")],
                "rocc_component_cycles",
            )

            colored = subprocess.run(
                [
                    "python3", str(REPORTER),
                    "--lifecycle-json", str(lifecycle_path),
                    "--numerical-json", str(numerical_path),
                    "--mpi-partition-json", str(mpi_path),
                    "--profile", "e3", "--mpi-ranks", "2",
                    "--sst-wall-seconds", "12.5",
                    "--pipeline-wall-seconds", "14.0", "--generic-gemm",
                    "--output-json", str(report_path),
                    "--output-csv", str(csv_path),
                ],
                env={**os.environ, "GOLEM_ATTENTION_COLOR": "1"},
                capture_output=True, text=True,
            )
            self.assertEqual(colored.returncode, 0, colored.stderr)
            self.assertIn("\x1b[", colored.stdout)
            self.assertIn("== RESULT ==", colored.stdout)

            failed_numerical = synthetic_numerical_result()
            failed_numerical["status"] = "FAIL"
            numerical_path.write_text(json.dumps(failed_numerical), encoding="ascii")
            failed_colored = subprocess.run(
                [
                    "python3", str(REPORTER),
                    "--lifecycle-json", str(lifecycle_path),
                    "--numerical-json", str(numerical_path),
                    "--mpi-partition-json", str(mpi_path),
                    "--profile", "e3", "--mpi-ranks", "2",
                    "--sst-wall-seconds", "12.5",
                    "--pipeline-wall-seconds", "14.0", "--generic-gemm",
                    "--output-json", str(report_path),
                    "--output-csv", str(csv_path),
                ],
                env={**os.environ, "GOLEM_ATTENTION_COLOR": "1"},
                capture_output=True, text=True,
            )
            numerical_line = next(
                line for line in failed_colored.stdout.splitlines()
                if "Numerical" in line
            )
            self.assertNotEqual(failed_colored.returncode, 0)
            self.assertIn("\x1b[1;31m", numerical_line)

    def test_cli_writes_failed_report_for_mpi_failure_before_returning_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            lifecycle_path = root / "lifecycle.json"
            numerical_path = root / "numerical.json"
            mpi_path = root / "mpi.json"
            report_path = root / "attention_metrics.json"
            csv_path = root / "attention_metrics.csv"
            lifecycle_path.write_text(
                json.dumps(synthetic_lifecycle()), encoding="ascii"
            )
            numerical_path.write_text(
                json.dumps(synthetic_numerical_result()), encoding="ascii"
            )
            mpi_path.write_text(
                json.dumps(synthetic_mpi_result("FAIL")), encoding="ascii"
            )

            result = subprocess.run(
                [
                    "python3", str(REPORTER),
                    "--lifecycle-json", str(lifecycle_path),
                    "--numerical-json", str(numerical_path),
                    "--mpi-partition-json", str(mpi_path),
                    "--profile", "e3", "--mpi-ranks", "2",
                    "--sst-wall-seconds", "1", "--pipeline-wall-seconds", "2",
                    "--generic-gemm", "--output-json", str(report_path),
                    "--output-csv", str(csv_path),
                ],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["source_status"]["mpi_partition"], "FAIL")
            self.assertTrue(csv_path.is_file())

    def test_resolved_run_config_supplies_actual_model_clocks(self):
        with tempfile.TemporaryDirectory() as directory:
            config = pathlib.Path(directory) / "run_config.env"
            config.write_text(
                "  VANADIS_CPU_CLOCK=1.0GHz\n"
                "  GOLEM_ARRAY_CLOCK=1.0GHz\n"
                "  GOLEM_MEMCTRL_CLOCK=1.25GHz\n",
                encoding="ascii",
            )
            self.assertEqual(
                read_model_clocks_from_run_config(config),
                {"cpu": 1_000_000_000, "array": 1_000_000_000,
                 "memctrl": 1_250_000_000},
            )

    def test_live_progress_waits_for_all_worker_frontiers(self):
        def marker(stage, core):
            return (
                f"[ATTENTION_MILESTONE] stage={stage} status=done "
                f"sst_tick={core} rocc_cycle={core} core={core}\n"
            )

        first_fifteen = "".join(
            marker("final_qk_tile_complete", core) for core in range(4, 19)
        )
        partial = subprocess.run(
            ["awk", "-v", "ui=1", "-f", str(PROGRESS_AWK)], input=first_fifteen,
            capture_output=True, text=True, check=True,
        )
        self.assertIn("QK 15/16", partial.stdout)
        self.assertIn("Softmax 0/16", partial.stdout)
        self.assertTrue(partial.stdout.startswith("66|QK|"), partial.stdout)

        first_worker = subprocess.run(
            ["awk", "-v", "ui=1", "-f", str(PROGRESS_AWK)],
            input=marker("final_qk_tile_complete", 4),
            capture_output=True, text=True, check=True,
        )
        self.assertIn("QK 1/16", first_worker.stdout)
        self.assertTrue(first_worker.stdout.startswith("19|QK|"), first_worker.stdout)

        complete = subprocess.run(
            ["awk", "-v", "ui=1", "-f", str(PROGRESS_AWK)],
            input=first_fifteen + marker("final_qk_tile_complete", 19),
            capture_output=True, text=True, check=True,
        )
        self.assertIn("QK 16/16", complete.stdout)
        self.assertIn("Softmax 0/16", complete.stdout)

        ui_progress = subprocess.run(
            ["awk", "-v", "ui=1", "-f", str(PROGRESS_AWK)],
            input=first_fifteen,
            capture_output=True, text=True, check=True,
        )
        self.assertTrue(ui_progress.stdout.startswith("66|QK|"), ui_progress.stdout)

        plain_progress = subprocess.run(
            ["awk", "-f", str(PROGRESS_AWK)], input=first_fifteen,
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(plain_progress.stdout, "66|QK\n")

    def test_terminal_progress_prints_each_attention_phase_once(self):
        progress = """\
  SIM   5%  Attention descriptor accepted               elapsed 00:00:01
  SIM   8%  Attention worker dispatch                   elapsed 00:00:02
  SIM   8%  Attention worker dispatch                   elapsed 00:00:03
generic configuration line that must remain archived
  SIM  89%  Attention QK running                        elapsed 00:00:20
  SIM  89%  Attention QK running                        elapsed 00:00:21
  SIM 100%  complete                                    elapsed 00:00:22
"""
        result = subprocess.run(
            ["awk", "-f", str(TERMINAL_PROGRESS_AWK)], input=progress,
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout.count("Attention worker dispatch"), 1)
        self.assertEqual(result.stdout.count("Attention QK running"), 1)
        self.assertNotIn("generic configuration", result.stdout)
        self.assertLessEqual(len(result.stdout.splitlines()), 4)

    def test_terminal_ui_renders_colored_dynamic_progress(self):
        command = (
            f'source "{TERMINAL_UI}"; '
            "GOLEM_ATTENTION_FORCE_TTY=1 GOLEM_ATTENTION_COLOR=1 "
            "TERM=xterm-256color "
            "attention_ui_render_progress 42 QK "
            "'QK 8/16 | Softmax 7/16 | PV 1/16 | DMA 0/16' 7 2"
        )
        result = subprocess.run(
            ["bash", "-c", command], capture_output=True, text=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"\x1b[", result.stdout)
        self.assertIn(b"\r", result.stdout)
        self.assertIn(b"42%", result.stdout)
        self.assertIn(b"QK 8/16", result.stdout)
        self.assertIn(b"DMA 0/16", result.stdout)
        self.assertNotIn(b"SIM", result.stdout)

    def test_terminal_ui_colors_stage_details_and_artifact_paths(self):
        command = (
            f'source "{TERMINAL_UI}"; '
            "GOLEM_ATTENTION_FORCE_TTY=1 GOLEM_ATTENTION_COLOR=1 "
            "TERM=xterm-256color; "
            "attention_ui_stage_result 'numerical verification' PASS 1.250s; "
            "attention_ui_key_value Artifacts /tmp/attention-results"
        )
        result = subprocess.run(
            ["bash", "-c", command], capture_output=True, text=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(result.stdout.count(b"\x1b["), 6)
        self.assertIn(b"numerical verification", result.stdout)
        self.assertIn(b"1.250s", result.stdout)
        self.assertIn(b"Artifacts", result.stdout)
        self.assertIn(b"/tmp/attention-results", result.stdout)

    def test_terminal_ui_progress_never_exceeds_terminal_width(self):
        for columns in (110, 109, 100, 80, 60, 59, 40, 30, 29):
            command = (
                f'source "{TERMINAL_UI}"; '
                "GOLEM_ATTENTION_FORCE_TTY=1 GOLEM_ATTENTION_COLOR=1 "
                f"GOLEM_ATTENTION_COLUMNS={columns} TERM=xterm-256color "
                "attention_ui_render_progress 74 QK "
                "'QK 15/16 | Softmax 14/16 | PV 5/16 | DMA 5/16' 7 2"
            )
            result = subprocess.run(
                ["bash", "-c", command], capture_output=True, text=False,
            )
            plain = re.sub(rb"\x1b\[[0-9;]*[A-Za-z]", b"", result.stdout)
            plain = plain.replace(b"\r", b"").rstrip(b"\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertLessEqual(len(plain), columns, plain)
            self.assertIn(b"QK", plain)
            self.assertIn(b"74%", plain)

    def test_terminal_ui_reads_detailed_internal_milestones(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            stage_log = root / "sst.log"
            progress_log = root / "runtime.log"
            markers = "".join(
                f"[ATTENTION_MILESTONE] stage=worker_dispatch_accept "
                f"status=done core={core}\n"
                for core in range(4, 20)
            )
            markers += (
                "[ATTENTION_MILESTONE] stage=final_qk_tile_complete "
                "status=done core=4\n"
            )
            progress_log.write_text(markers)
            command = (
                f'source "{TERMINAL_UI}"; '
                "GOLEM_ATTENTION_FORCE_TTY=1 GOLEM_ATTENTION_COLOR=0 "
                "GOLEM_ATTENTION_TERMINAL_REFRESH_SECONDS=0.05 "
                f'ATTENTION_UI_PROGRESS_LOG="{progress_log}" '
                f'ATTENTION_UI_MILESTONE_FILTER="{PROGRESS_AWK}" '
                f'attention_ui_run_sst "{stage_log}" "{TERMINAL_PROGRESS_AWK}" '
                "bash -c 'sleep 0.15'"
            )
            result = subprocess.run(
                ["bash", "-c", command], capture_output=True, text=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"QK", result.stdout)
        self.assertIn(b"QK 1/16", result.stdout)
        self.assertIn(b"Softmax 0/16", result.stdout)

    def test_terminal_ui_preserves_background_command_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            stage_log = pathlib.Path(directory) / "sst.log"
            command = (
                f'source "{TERMINAL_UI}"; '
                "GOLEM_ATTENTION_FORCE_TTY=1 GOLEM_ATTENTION_COLOR=0 "
                f'attention_ui_run_sst "{stage_log}" "{TERMINAL_PROGRESS_AWK}" '
                "bash -c 'echo sentinel-output; exit 37'; "
                "rc=$?; printf 'rc=%s\\n' \"$rc\"; exit 0"
            )
            result = subprocess.run(
                ["bash", "-c", command], capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("rc=37", result.stdout)
            self.assertIn("sentinel-output", stage_log.read_text())

    def test_terminal_ui_disables_fancy_mode_for_dumb_terminal(self):
        command = (
            f'source "{TERMINAL_UI}"; '
            "TERM=dumb GOLEM_ATTENTION_FORCE_TTY=0; "
            "if attention_ui_supports_fancy; then exit 1; fi"
        )
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_terminal_ui_forwards_term_and_reaps_background_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            stage_log = root / "sst.log"
            child_pid_file = root / "child.pid"
            inner = (
                f"echo $$ > '{child_pid_file}'; echo child-started; "
                "trap 'exit 0' TERM INT HUP; while :; do sleep 0.1; done"
            )
            command = (
                f'source "{TERMINAL_UI}"; '
                "GOLEM_ATTENTION_FORCE_TTY=1 GOLEM_ATTENTION_COLOR=0 "
                f'attention_ui_run_sst "{stage_log}" "{TERMINAL_PROGRESS_AWK}" '
                f'bash -c "{inner}"; exit $?'
            )
            process = subprocess.Popen(
                ["bash", "-c", command], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if child_pid_file.is_file():
                        child_pid = int(child_pid_file.read_text().strip())
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(child_pid, "background child did not start")
                process.send_signal(signal.SIGTERM)
                process.communicate(timeout=5)
                returncode = process.returncode
                deadline = time.monotonic() + 2
                while pathlib.Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertEqual(returncode, 143)
                self.assertFalse(pathlib.Path(f"/proc/{child_pid}").exists())
                self.assertIn("child-started", stage_log.read_text())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if child_pid and pathlib.Path(f"/proc/{child_pid}").exists():
                    os.kill(child_pid, signal.SIGTERM)

    def test_runner_enables_job_milestones_and_reports_artifacts(self):
        result = subprocess.run(
            [
                str(RUNNER), "--queries", "256", "--keys", "256",
                "--head-dim", "64", "--dry-run",
            ],
            cwd=HERE,
            env={**os.environ, "GOLEM_MPI_RANKS": "1"},
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GOLEM_ATTENTION_MILESTONE_TRACE=1", result.stdout)
        self.assertIn("GOLEM_ATTENTION_TILE_TRACE=0", result.stdout)
        self.assertIn("GOLEM_PROGRESS_HEARTBEAT=0", result.stdout)
        self.assertIn("--log attention-sst.log", result.stdout)
        self.assertIn("report_attention_metrics.py", result.stdout)
        self.assertIn("attention_metrics.json", result.stdout)
        self.assertIn("attention_metrics.csv", result.stdout)
        self.assertIn("--run-config", result.stdout)
        self.assertIn("--mpi-partition-json", result.stdout)
        runner_text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("GOLEM_ATTENTION_TERMINAL_VERBOSE", runner_text)
        self.assertIn("attention_terminal_progress.awk", runner_text)
        self.assertIn("ATTENTION_UI_PROGRESS_LOG", runner_text)
        self.assertIn("ATTENTION_UI_MILESTONE_FILTER", runner_text)
        self.assertIn("GOLEM_ATTENTION_TERMINAL_REFRESH_SECONDS", TERMINAL_UI.read_text())
        self.assertIn("attention_terminal_ui.sh", runner_text)
        self.assertIn("driver_logs", runner_text)
        self.assertNotIn(
            'echo "Fused Attention ${SCALE_POINT^^} PASS', runner_text
        )
        self.assertIn(
            'if run_attention_stage numerical_verify "${VERIFY_CMD[@]}"',
            runner_text,
        )

    def test_rocc_exposes_parseable_attention_milestones(self):
        text = ROCC_SOURCE.read_text(encoding="utf-8")
        self.assertIn("[ATTENTION_MILESTONE]", text)
        self.assertIn("stage=%s", text)
        self.assertIn("status=%s", text)
        self.assertIn("sst_tick=", text)
        self.assertIn("rocc_cycle=", text)
        self.assertIn("completedQueryTile", text)
        self.assertIn("completedKvTile", text)


if __name__ == "__main__":
    unittest.main()
