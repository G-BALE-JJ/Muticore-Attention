import copy
import json
import pathlib
import os
import subprocess
import sys
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from verify_attention_mpi_partition import expected_component_ranks
from verify_fused_attention_scale_stats import (
    PROFILES,
    attention_cluster_ii_sample_counts,
    expected_attention_cluster_broadcast_activity,
    expected_matrix_broadcast_activity,
    expected_v_tile_buffer_activity,
    make_attention_activity,
    make_clock_contract,
    parse_dma_runtime_invariants,
    summarize_attention_cluster_resource_profile,
)
from report_attention_gpu_comparison import build_report
from gpu_attention_stage_benchmark import make_correctness, make_result


WRAPPER = HERE / "run_flash_attention.sh"
SCALE_RUNNER = HERE / "run_fused_attention_scale.sh"
UNIFIED_RUNNER = HERE.parents[6] / "scripts" / "test_flash_attention.sh"
BUILD_SCRIPT = HERE.parents[6] / "scripts" / "build_and_install_local.sh"
ARCHIVE_ARCH = HERE.parents[1] / "architecture" / "archive" / "ncores_selfcom_dma.py"
BASELINE_ROOT = HERE.parents[6] / "baseline"
GPU_BASELINE = BASELINE_ROOT / "gpu_attention_rtx5060.json"
GPU_SCHEMA = HERE / "gpu_attention_stage_schema.json"
ROCC_SOURCE = HERE.parents[2] / "rocc" / "roccAnalog.h"
WCP_SOURCE = HERE.parents[2] / "workercmdproc" / "workercmdproc.h"
CPU_BUILDER = HERE.parents[1] / "architecture" / "cpu_builder.py"
ARRAY_SOURCE = HERE.parents[2] / "array" / "computeArray.h"
MVM_ARRAY_SOURCE = HERE.parents[2] / "array" / "mvmComputeArray.h"
GLOBAL_MEMORY_SOURCE = HERE.parents[2] / "globalmemory" / "globalmemory.h"
MEMNIC_SOURCE = HERE.parents[3] / "memHierarchy" / "memNICBase.h"
BASELINE_VERIFIER = HERE / "verify_flash_attention_baseline.py"


class FlashAttentionBaselineContractTest(unittest.TestCase):
    def test_attention_resource_profile_separates_union_from_worker_sum(self):
        observed = {}
        maxima = {}
        resources = (
            ("rocc", "attention_cluster_qk_array"),
            ("rocc", "attention_cluster_pv_array"),
            ("rocc:array", "attention_cluster_buffer"),
            ("rocc:global_memory", "attention_cluster_local_read"),
            ("rocc:global_memory", "attention_cluster_local_write"),
            ("rocc:sfu", "attention_cluster_sfu"),
        )
        for core in range(4, 20):
            for suffix, prefix in resources:
                component = f"core{core}:{suffix}"
                observed[(component, f"{prefix}_busy_union_ticks")] = core * 1000
                observed[(component, f"{prefix}_busy_span_ticks")] = core * 2000
                observed[(component, f"{prefix}_idle_gap_ticks")] = core * 1000
                maxima[(component, f"{prefix}_max_concurrency")] = core
        profile = summarize_attention_cluster_resource_profile(
            observed, maxima, 19, 1_000_000_000, 1_000_000_000_000,
        )
        qk = profile["resources"]["qk_array_active"]
        self.assertEqual(qk["critical_worker"]["busy_ticks"], 19000)
        self.assertEqual(qk["critical_worker"]["busy_cycles"], 19)
        self.assertEqual(qk["critical_worker"]["busy_fraction_of_span"], 0.5)
        self.assertEqual(qk["worker_totals"]["busy_ticks"], 184000)
        self.assertEqual(qk["max_worker_concurrency"], 19)
        self.assertIn("not end-to-end latency", profile["interval_semantics"])

    def test_attention_cluster_ii_counts_cover_partial_and_small_groups(self):
        self.assertEqual(
            attention_cluster_ii_sample_counts(1, 1, 4),
            {"steady": 0, "boundary": 0},
        )
        self.assertEqual(
            attention_cluster_ii_sample_counts(2, 1, 4),
            {"steady": 0, "boundary": 0},
        )
        self.assertEqual(
            attention_cluster_ii_sample_counts(4, 1, 4),
            {"steady": 2, "boundary": 0},
        )
        self.assertEqual(
            attention_cluster_ii_sample_counts(5, 3, 4),
            {"steady": 6, "boundary": 5},
        )

    def test_dma_runtime_invariants_check_real_queue_and_credit_conservation(self):
        valid_lines = "".join(
            f"[memNICBase bridge] CREDIT_OWNER_SUMMARY "
            f"name=dirctrl_{node}:highlink cap=128 available=128 "
            f"admitted={40 if node else 0} released={40 if node else 0} pending=0\n"
            for node in range(5)
        ) + "".join(
            f"GOLEM_MEMNIC_DMA_RESPONSE_STATS component=dirctrl_{node}:highlink "
            "attempted=40 immediate=10 enqueued=30 drained=30 pending=0 "
            "responses_d0=10 responses_d1=10 responses_d2=10 responses_far=10\n"
            for node in range(1, 5)
        )
        invalid_lines = valid_lines.replace("released=40", "released=39").replace(
            "drained=30", "drained=29"
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime_log = pathlib.Path(directory) / "runtime.log"
            runtime_log.write_text(valid_lines, encoding="ascii")
            valid = parse_dma_runtime_invariants(runtime_log)
            self.assertEqual(valid["mismatches"], {})
            runtime_log.write_text(invalid_lines, encoding="ascii")
            invalid = parse_dma_runtime_invariants(runtime_log)
            self.assertIn(
                "dirctrl_1:highlink/dma_credit_conservation",
                invalid["mismatches"],
            )
            self.assertIn(
                "dirctrl_1:highlink/dma_response_drain", invalid["mismatches"]
            )
            cap_mismatch = parse_dma_runtime_invariants(
                runtime_log, expected_credit_cap=64)
            self.assertIn(
                "dirctrl_1:highlink/dma_credit_cap", cap_mismatch["mismatches"]
            )
            response_mismatch_lines = valid_lines.replace(
                "component=dirctrl_1:highlink attempted=40 immediate=10",
                "component=dirctrl_1:highlink attempted=39 immediate=9",
            ).replace(
                "responses_d0=10 responses_d1=10 responses_d2=10 responses_far=10",
                "responses_d0=9 responses_d1=10 responses_d2=10 responses_far=10",
                1,
            )
            runtime_log.write_text(response_mismatch_lines, encoding="ascii")
            response_mismatch = parse_dma_runtime_invariants(runtime_log)
            self.assertIn(
                "dirctrl_1:highlink/dma_credit_response_conservation",
                response_mismatch["mismatches"],
            )
            runtime_log.write_text(
                valid_lines.replace(
                    "component=dirctrl_4:highlink", "component=dirctrl_3:highlink"
                ),
                encoding="ascii",
            )
            truncated = parse_dma_runtime_invariants(runtime_log)
            self.assertIn("dma_response/runtime_summary", truncated["mismatches"])
            self.assertIn("dma_response/runtime_duplicates", truncated["mismatches"])

    def test_dma_runtime_invariants_allow_disabled_credit_admission(self):
        responses = "".join(
            f"GOLEM_MEMNIC_DMA_RESPONSE_STATS component=dirctrl_{node}:highlink "
            "attempted=1 immediate=1 enqueued=0 drained=0 pending=0 "
            "responses_d0=1 responses_d1=0 responses_d2=0 responses_far=0\n"
            for node in range(1, 5)
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime_log = pathlib.Path(directory) / "runtime.log"
            runtime_log.write_text(responses, encoding="ascii")
            result = parse_dma_runtime_invariants(
                runtime_log, expected_credit_cap=0)
            self.assertEqual(result["mismatches"], {})

    def write_placement_manifest(self, directory, mpi_ranks, overrides=None):
        component_ranks = expected_component_ranks(mpi_ranks)
        component_ranks.update(overrides or {})
        placement_file = pathlib.Path(directory) / "attention_mpi_placement.json"
        placement_file.write_text(
            json.dumps(
                {"mpi_ranks": mpi_ranks, "component_ranks": component_ranks}
            ),
            encoding="ascii",
        )
        return placement_file

    def write_ranked_stats(self, directory, mpi_ranks):
        stats_file = pathlib.Path(directory) / "stats_selfcom.txt"
        for rank in range(mpi_ranks):
            rows = ["ComponentName,Rank"]
            rows.extend(
                f"core{core_id}:rocc,{rank}"
                for core_id in range(20)
                if core_id % mpi_ranks == rank
            )
            rows.extend(
                f"rtr_{router_id},{rank}"
                for router_id in range(28)
                if (router_id % mpi_ranks if router_id < 24 else 0) == rank
            )
            stats_file.with_name(f"stats_selfcom_{rank}.txt").write_text(
                "\n".join(rows) + "\n", encoding="ascii"
            )
        return stats_file

    def run_partition_verifier(self, stats_file, placement_file, mpi_ranks):
        result = subprocess.run(
            [
                "python3",
                str(HERE / "verify_attention_mpi_partition.py"),
                "--stats-file",
                str(stats_file),
                "--mpi-ranks",
                str(mpi_ranks),
                "--placement-file",
                str(placement_file),
            ],
            capture_output=True,
            text=True,
        )
        return result, json.loads(result.stdout)

    def test_wrapper_uses_parameterized_default_case(self):
        result = subprocess.run(
            [str(WRAPPER), "--dry-run"],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("fused_attention_q1024_k1024_d128", result.stdout)
        self.assertIn("GOLEM_ATTENTION_GUEST_MANAGER_QUERIES=256", result.stdout)
        self.assertNotIn("PROFILE=", WRAPPER.read_text())

    def test_wrapper_targets_local_scale_runner(self):
        text = WRAPPER.read_text()
        self.assertIn(
            'exec "$SCRIPT_DIR/run_fused_attention_scale.sh" "$@"', text
        )
        self.assertNotIn("RISC-V-CIM-Manycore-SST", text)

    def test_scale_runner_forwards_multirank_execution(self):
        result = subprocess.run(
            [str(WRAPPER), "--dry-run"],
            cwd=HERE,
            env={**os.environ, "GOLEM_MPI_RANKS": "2"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mpi-ranks 2", result.stdout)
        self.assertIn("--mpi-partitioner sst.self", result.stdout)
        self.assertIn("--lib-path=", result.stdout)
        self.assertIn("/install/lib/sst-elements-library", result.stdout)
        self.assertIn("architecture/archive/ncores_selfcom_dma.py", result.stdout)
        self.assertIn("GOLEM_ATTENTION_QUERY_BLOCK_MPI=1", result.stdout)
        self.assertIn("verify_attention_mpi_partition.py", result.stdout)
        self.assertNotIn("/data/shun/", result.stdout)
        self.assertNotIn("make -C", result.stdout)

    def test_attention_uses_shared_gemm_compute_engine_by_default(self):
        result = subprocess.run(
            [str(WRAPPER), "--dry-run"],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("GOLEM_ATTENTION_GENERIC_GEMM_ENABLE=1", result.stdout)
        self.assertIn("GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=1", result.stdout)

    def test_pv_matrix_broadcast_is_a_bounded_modeled_array_fabric(self):
        enabled = subprocess.run(
            [
                str(SCALE_RUNNER), "--queries", "256", "--keys", "256",
                "--head-dim", "64", "--dry-run",
            ],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )
        disabled = subprocess.run(
            [
                str(SCALE_RUNNER), "--queries", "256", "--keys", "256",
                "--head-dim", "64",
                "--no-pv-matrix-broadcast", "--dry-run",
            ],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("GOLEM_ATTENTION_PV_MATRIX_BROADCAST=1", enabled.stdout)
        self.assertIn("GOLEM_ATTENTION_PV_MATRIX_BROADCAST=0", disabled.stdout)
        for name, value in (
            ("GOLEM_MATRIX_BROADCAST_MAX_FANOUT", "16"),
            ("GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE", "64"),
            ("GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES", "1"),
            ("GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES", "1"),
        ):
            self.assertIn(f"{name}={value}", enabled.stdout)

        array_source = ARRAY_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()
        verifier = (HERE / "verify_fused_attention_scale_stats.py").read_text()
        for parameter in (
            "matrixBroadcastMaxFanout",
            "matrixBroadcastBytesPerCycle",
            "matrixBroadcastBaseLatencyCycles",
            "matrixBroadcastStageLatencyCycles",
        ):
            self.assertIn(parameter, array_source)
            self.assertIn(parameter, builder)
        self.assertIn("enqueueMatrixBroadcastTransfer", array_source)
        self.assertIn("validateMatrixBroadcastRequest", array_source)
        self.assertIn("uniqueIDs.size() == arrayIDs.size()", array_source)
        self.assertIn(
            "!array_->validateMatrixBroadcastRequest",
            WCP_SOURCE.read_text(),
        )
        self.assertIn("matrix_broadcast_ingress_bytes", array_source)
        self.assertIn("matrix_broadcast_sink_bytes", array_source)
        self.assertIn("matrix_broadcast_transfer_cycles", array_source)
        self.assertIn("matrix_broadcast_fanout", array_source)
        self.assertIn("matrix_broadcast_fabric", verifier)

        expected = expected_matrix_broadcast_activity(
            PROFILES["e3"], True, False, False, 16, 64, 1, 1
        )
        self.assertEqual(expected["requests"], 1024)
        self.assertEqual(expected["payload_bytes"], 8192)
        self.assertEqual(expected["ingress_bytes"], 8 * 1024 * 1024)
        self.assertEqual(expected["sink_bytes"], 128 * 1024 * 1024)
        self.assertEqual(expected["tree_stages"], 4)
        self.assertEqual(expected["cycles_per_request"], 133)
        self.assertEqual(expected["transfer_cycles"], 136192)

        mixed_active = expected_matrix_broadcast_activity(
            PROFILES["e3"], True, True, False, 16, 64, 1, 1, True
        )
        self.assertEqual(mixed_active["qk_payload_bytes"], 8192)
        self.assertEqual(mixed_active["pv_payload_bytes"], 2048)
        self.assertEqual(mixed_active["requests"], 1152)
        self.assertEqual(mixed_active["ingress_bytes"], 3 * 1024 * 1024)
        self.assertEqual(mixed_active["sink_bytes"], 48 * 1024 * 1024)
        self.assertEqual(mixed_active["transfer_cycles"], 54912)

        cluster = expected_attention_cluster_broadcast_activity(
            make_attention_activity(256, 256, 128), 64, 1, 1
        )
        self.assertEqual(cluster["requests"], 416)
        self.assertEqual(cluster["ingress_bytes"], 475136)
        self.assertEqual(cluster["sink_bytes"], 4292608)
        self.assertEqual(cluster["transfer_cycles"], 8080)
        self.assertEqual(cluster["fanout_sum"], 896)
        self.assertEqual(
            [cluster["min_fanout"], cluster["max_observed_fanout_expected"]],
            [1, 16],
        )

        cluster_24_40 = expected_attention_cluster_broadcast_activity(
            make_attention_activity(256, 128, 128, key_block_rows=64),
            64, 1, 1, qk_arrays=24,
        )
        self.assertEqual(cluster_24_40["fanout_sum"], 240)
        self.assertEqual(cluster_24_40["sink_bytes"], 1867776)

        cluster_32_32 = expected_attention_cluster_broadcast_activity(
            make_attention_activity(256, 256, 128), 64, 1, 1, qk_arrays=32
        )
        self.assertEqual(cluster_32_32["fanout_sum"], 1024)
        self.assertEqual(cluster_32_32["sink_bytes"], 6389760)

    def test_cluster_uses_chapter4_64_array_contract(self):
        cluster = (HERE.parents[2] / "attention" / "attentionCluster.h").read_text()
        runner = SCALE_RUNNER.read_text()
        rocc = ROCC_SOURCE.read_text()

        for contract in (
            "uint32_t arrays = 64",
            "uint32_t qkArrays = 16",
            "uint32_t pvArrays = 48",
            "uint32_t arrayInputs = 64",
            "uint32_t arrayOutputs = 64",
        ):
            self.assertIn(contract, cluster)
        self.assertIn("ARRAY_INPUT=64", runner)
        self.assertIn("ARRAY_OUTPUT=64", runner)
        self.assertIn("NUM_ARRAYS=64", runner)
        self.assertIn("attentionClusterQkArray", rocc)
        self.assertIn("attentionClusterPvArray", rocc)
        self.assertIn("readAttentionClusterScorePairAsync", rocc)

        array = (HERE.parents[2] / "array" / "mvmComputeArray.h").read_text()
        wcp = WCP_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()
        for source in (array, wcp):
            self.assertIn("attention_cluster_qk_arrays", source)
            self.assertIn("attentionClusterQkArrays", source)
        self.assertGreaterEqual(
            builder.count('"attention_cluster_qk_arrays": attention_cluster_qk_arrays'),
            3,
        )

        cluster_bc64 = expected_attention_cluster_broadcast_activity(
            make_attention_activity(256, 256, 128, key_block_rows=64),
            64, 1, 1,
        )
        self.assertEqual(cluster_bc64["requests"], 208)
        self.assertEqual(cluster_bc64["ingress_bytes"], 311296)
        self.assertEqual(cluster_bc64["sink_bytes"], 3211264)
        self.assertEqual(cluster_bc64["transfer_cycles"], 5192)
        self.assertEqual(cluster_bc64["fanout_sum"], 448)

    def test_qk_matrix_broadcast_has_an_explicit_disable_option(self):
        result = subprocess.run(
            [
                str(SCALE_RUNNER),
                "--queries", "256", "--keys", "256", "--head-dim", "64",
                "--qk-matrix-broadcast",
                "--no-qk-matrix-broadcast",
                "--dry-run",
            ],
            cwd=HERE,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GOLEM_ATTENTION_QK_MATRIX_BROADCAST=0", result.stdout)

    def test_attention_direct_gemm_fallback_is_explicit(self):
        result = subprocess.run(
            [str(WRAPPER), "--direct-gemm", "--dry-run"],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("GOLEM_ATTENTION_GENERIC_GEMM_ENABLE=0", result.stdout)
        self.assertIn("GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=0", result.stdout)
        self.assertIn("GOLEM_ATTENTION_QK_PANEL_ROW_BURST=0", result.stdout)
        self.assertIn("architecture/archive/ncores_selfcom_dma.py", result.stdout)

    def test_scale_runner_records_clock_contract_without_changing_model(self):
        result = subprocess.run(
            [str(WRAPPER), "--dry-run"],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )
        for option, value in (
            ("--normalization-clock", "1.0GHz"),
            ("--run-config", "run_config.env"),
            ("--model-platform-clock", "2.0GHz"),
        ):
            self.assertIn(option, result.stdout)
            self.assertIn(value, result.stdout)

    def test_clock_contract_separates_model_and_normalized_cycles(self):
        contract = make_clock_contract(
            1_000_000_000, 1_000_000_000_000,
            2_300_000_000, 2_300_000_000,
            2_300_000_000, 2_000_000_000,
        )
        self.assertFalse(contract["normalized_cycles_are_model_native_cycles"])
        self.assertEqual(contract["normalization_clock_hz"], 1_000_000_000)
        self.assertEqual(
            contract["model_clocks_hz"]["vanadis_cpu_rocc_sfu_local_gm"],
            2_300_000_000,
        )

    def test_scale_runner_ignores_polluting_generic_sst_args(self):
        result = subprocess.run(
            [str(WRAPPER), "--dry-run"],
            cwd=HERE,
            env={
                **os.environ,
                "GOLEM_MPI_RANKS": "2",
                "GOLEM_SST_ARGS": "--partitioner=sst.simple --lib-path=/tmp/foreign",
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mpi-partitioner sst.self", result.stdout)
        self.assertNotIn("sst.simple", result.stdout)
        self.assertNotIn("/tmp/foreign", result.stdout)

    def test_scale_runner_rejects_untracked_timebase_override(self):
        result = subprocess.run(
            [str(WRAPPER), "--dry-run"],
            cwd=HERE,
            env={
                **os.environ,
                "GOLEM_ATTENTION_SST_ARGS": "--timebase=10ps",
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot override", result.stderr)
        self.assertIn("timebase", result.stderr)

    def test_archive_architecture_supports_sst_mpi_partitioning(self):
        text = ARCHIVE_ARCH.read_text()
        self.assertIn('MPI_PARTITIONING = _env_flag("GOLEM_MPI_PARTITIONING", False)', text)
        self.assertIn('DRAMSIM3_OUT_DIR = os.getenv(', text)
        self.assertIn('node_output_dir = os.path.join(DRAMSIM3_OUT_DIR, f"node{idx}")', text)
        self.assertGreaterEqual(text.count("if not MPI_PARTITIONING:"), 3)
        self.assertIn("def attention_rank_for_core(core_id: int) -> int:", text)
        self.assertIn("def set_attention_cpu_rank(prefix: str, rank: int) -> None:", text)
        self.assertIn("component = sst.findComponentByName(component_name)", text)
        self.assertIn("for router_id, router in enumerate(noc.routers):", text)
        self.assertIn("set_attention_cpu_rank(", text)
        self.assertIn("def set_attention_component_rank(", text)
        self.assertIn("GOLEM_ATTENTION_PLACEMENT_FILE", text)

    def test_unified_runner_uses_worktree_install_and_explicit_parameters(self):
        text = UNIFIED_RUNNER.read_text()
        verifier = BASELINE_VERIFIER.read_text()
        self.assertIn('source "$SCRIPT_DIR/env_local_install.sh"', text)
        self.assertIn('"$ATTENTION_DIR/run_flash_attention.sh"', text)
        self.assertIn('--queries "$QUERIES"', text)
        self.assertIn('--keys "$KEYS"', text)
        self.assertIn('--head-dim "$HEAD_DIM"', text)
        self.assertIn('--timeout "$TIMEOUT" --artifact-root "$ARTIFACT_ROOT"', text)
        self.assertIn('export SST_LIB_PATH="$WORKTREE_ROOT/install/lib/sst-elements-library"', text)
        self.assertIn('--mpi-ranks) MPI_RANKS="$2"', text)
        self.assertIn('--baseline) BASELINE_JSON="$2"', text)
        self.assertNotIn('BASELINE_DIR="$WORKTREE_ROOT/baseline/$PROFILE"', text)
        self.assertNotIn('--profile', text)
        self.assertIn('"verification.score_probability_hbm_bytes"', verifier)
        self.assertIn(
            '--generic-gemm --pv-matrix-broadcast --qk-matrix-broadcast '
            '--kv-double-buffer',
            text,
        )
        self.assertIn('GOLEM_MATRIX_BROADCAST_MAX_FANOUT=16', text)
        self.assertIn('GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE=64', text)
        self.assertIn('GOLEM_MATRIX_BROADCAST_BASE_LATENCY_CYCLES=1', text)
        self.assertIn('GOLEM_MATRIX_BROADCAST_STAGE_LATENCY_CYCLES=1', text)
        self.assertIn('"architecture.pv_matrix_broadcast"', verifier)
        self.assertIn('"architecture.kv_double_buffer"', verifier)
        self.assertIn('"architecture.wcp_gemm_proxy"', verifier)
        self.assertIn('"architecture.matrix_broadcast_fabric"', verifier)
        self.assertIn(
            '"lifecycle.worker_critical_path.kv_prefetch_timing"', verifier
        )
        self.assertIn("verify_flash_attention_baseline.py", text)
        self.assertIn('if [[ -n "$BASELINE_JSON" ]]', text)
        self.assertIn("Build it first with scripts/build_and_install_local.sh", text)
        self.assertNotIn("export SST_SOFTMAX_LD_LIBRARY_PATH=", text)
        self.assertNotIn("BUILD_ARGS", text)
        self.assertNotIn("SKIP_BUILD", text)

    def test_frozen_baseline_verifier_rejects_architecture_mutations(self):
        baseline = json.loads(
            (BASELINE_ROOT / "e3" / "result.json").read_text(encoding="ascii")
        )
        result = {
            "status": "PASS",
            "checked": baseline["verification"]["checked"],
            "mismatches": baseline["verification"]["mismatches"],
            "shape": {
                "queries": baseline["shape"]["queries"],
                "keys": baseline["shape"]["keys"],
                "head_dim": baseline["shape"]["head_dim"],
            },
            "score_probability_hbm_bytes": baseline["verification"][
                "score_probability_hbm_bytes"
            ],
            "max_abs_error": baseline["verification"]["max_abs_error"],
        }
        fabric = copy.deepcopy(
            baseline["architecture"]["matrix_broadcast_fabric"]
        )
        fabric.update({"pv_enabled": True, "qk_enabled": True})
        kv_prefetch_timing = copy.deepcopy(
            baseline["lifecycle"]["worker_critical_path"]["kv_prefetch_timing"]
        )
        lifecycle = {
            "status": "PASS",
            "lifecycle": {
                "worker_critical_path": {
                    "order_valid": True,
                    "inter_tile_breakdown": {"conservation_valid": True},
                    "kv_prefetch_timing": kv_prefetch_timing,
                },
                "accelerator_completion_cycles": baseline["lifecycle"][
                    "accelerator_completion_cycles"
                ],
                "wait_return_cycles": baseline["lifecycle"][
                    "wait_return_cycles"
                ],
                "wcp_gemm_proxy": copy.deepcopy(
                    baseline["architecture"]["wcp_gemm_proxy"]
                ),
                "matrix_broadcast_fabric": fabric,
            },
        }
        mutations = {
            "control": None,
            "architecture.generic_gemm_wcp": lambda doc: doc["lifecycle"].pop(
                "wcp_gemm_proxy"
            ),
            "architecture.wcp_gemm_proxy": lambda doc: doc["lifecycle"].update(
                wcp_gemm_proxy={"garbage": 1}
            ),
            "architecture.pv_matrix_broadcast": lambda doc: doc["lifecycle"][
                "matrix_broadcast_fabric"
            ].update(pv_enabled=False),
            "architecture.qk_matrix_broadcast": lambda doc: doc["lifecycle"][
                "matrix_broadcast_fabric"
            ].update(qk_enabled=False),
            "architecture.kv_double_buffer": lambda doc: doc["lifecycle"][
                "worker_critical_path"
            ].pop("kv_prefetch_timing"),
            "lifecycle.worker_critical_path.kv_prefetch_timing": lambda doc: doc[
                "lifecycle"
            ]["worker_critical_path"]["kv_prefetch_timing"]["counts"].update(
                dma=0
            ),
            "lifecycle.worker_critical_path.kv_prefetch_timing.counts.dma": lambda doc: doc[
                "lifecycle"
            ]["worker_critical_path"]["kv_prefetch_timing"]["counts"].update(
                dma=1
            ),
            "architecture.matrix_broadcast_fabric": lambda doc: doc[
                "lifecycle"
            ]["matrix_broadcast_fabric"].update(bytes_per_cycle=63),
            "architecture.matrix_broadcast_fabric.worker_totals.matrix_broadcast_requests": lambda doc: doc[
                "lifecycle"
            ]["matrix_broadcast_fabric"]["worker_totals"].update(
                matrix_broadcast_requests=16383
            ),
            "lifecycle.accelerator_completion_cycles": lambda doc: doc[
                "lifecycle"
            ].update(accelerator_completion_cycles=2232813),
        }

        for expected_failure, mutate in mutations.items():
            with self.subTest(expected_failure=expected_failure), \
                    tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                actual_lifecycle = copy.deepcopy(lifecycle)
                if mutate is not None:
                    mutate(actual_lifecycle)
                paths = {
                    "baseline": root / "baseline.json",
                    "result": root / "result.json",
                    "lifecycle": root / "lifecycle.json",
                }
                paths["baseline"].write_text(json.dumps(baseline), encoding="ascii")
                paths["result"].write_text(json.dumps(result), encoding="ascii")
                paths["lifecycle"].write_text(
                    json.dumps(actual_lifecycle), encoding="ascii"
                )
                completed = subprocess.run(
                    [
                        "python3", str(BASELINE_VERIFIER),
                        "--baseline", str(paths["baseline"]),
                        "--result", str(paths["result"]),
                        "--lifecycle", str(paths["lifecycle"]),
                        "--mpi-ranks", "1",
                    ],
                    capture_output=True,
                    text=True,
                )
                if mutate is None:
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    self.assertIn("baseline MATCH", completed.stdout)
                else:
                    self.assertEqual(completed.returncode, 1, completed.stdout)
                    self.assertIn(expected_failure, completed.stderr)

    def test_unified_runner_routes_custom_shape_without_starting_sst(self):
        result = subprocess.run(
            [
                str(UNIFIED_RUNNER),
                "--queries", "768",
                "--keys", "512",
                "--head-dim", "64",
                "--mpi-ranks", "4",
                "--timeout", "99",
                "--artifact-root", "/tmp/attention-custom",
                "--show-config",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        config = dict(line.split("=", 1) for line in result.stdout.splitlines())
        self.assertEqual(config["CASE_ID"], "fused_attention_q768_k512_d64")
        self.assertEqual(config["QUERIES"], "768")
        self.assertEqual(config["KEYS"], "512")
        self.assertEqual(config["HEAD_DIM"], "64")
        self.assertEqual(config["MANAGER_QUERIES"], "192")
        self.assertEqual(config["TIMEOUT"], "99")
        self.assertEqual(config["MPI_RANKS"], "4")
        self.assertEqual(config["BASELINE_JSON"], "")
        self.assertEqual(config["ARTIFACT_ROOT"], "/tmp/attention-custom")
        self.assertEqual(config["QK_MATRIX_BROADCAST"], "1")
        self.assertEqual(config["KV_DOUBLE_BUFFER"], "1")
        self.assertEqual(config["KV_PAIR_REUSE"], "1")
        self.assertEqual(config["KV_QUERY_GROUP_SIZE"], "4")

        group_one = subprocess.run(
            [str(UNIFIED_RUNNER), "--show-config"],
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE": "1",
            },
        )
        group_one_config = dict(
            line.split("=", 1) for line in group_one.stdout.splitlines()
        )
        self.assertEqual(group_one_config["KV_PAIR_REUSE"], "0")
        self.assertEqual(group_one_config["KV_QUERY_GROUP_SIZE"], "1")

    def test_unified_runner_keeps_experimental_mechanisms_out_of_public_cli(self):
        help_result = subprocess.run(
            [str(UNIFIED_RUNNER), "--help"], capture_output=True, text=True,
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertNotIn("--direct-gemm", help_result.stdout)
        rejected = subprocess.run(
            [str(UNIFIED_RUNNER), "--direct-gemm"], capture_output=True, text=True,
        )
        self.assertEqual(rejected.returncode, 2)

    def test_baseline_preflight_rejects_shape_before_simulation(self):
        result = subprocess.run(
            [
                "python3", str(BASELINE_VERIFIER),
                "--baseline", str(BASELINE_ROOT / "e3" / "result.json"),
                "--mpi-ranks", "1", "--queries", "768", "--keys", "512",
                "--head-dim", "64", "--preflight-only",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("baseline.shape", result.stderr)

    def test_unified_runner_rejects_missing_option_values(self):
        for option in (
            "--queries", "--keys", "--head-dim", "--timeout", "--mpi-ranks",
            "--artifact-root", "--baseline",
        ):
            with self.subTest(option=option):
                result = subprocess.run(
                    [str(UNIFIED_RUNNER), option],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(f"Missing value for {option}", result.stderr)
                self.assertNotIn("unbound variable", result.stderr)

    def test_unified_runner_rejects_shapes_that_do_not_fit_fixed_topology(self):
        invalid_cases = (
            ("--queries", "257", "--queries must be divisible by 256"),
            ("--keys", "129", "--keys must be divisible by 128"),
            ("--head-dim", "65", "--head-dim must be 64 or 128"),
            ("--head-dim", "16", "--head-dim must be 64 or 128"),
            ("--head-dim", "32", "--head-dim must be 64 or 128"),
            ("--head-dim", "80", "--head-dim must be 64 or 128"),
        )
        for option, value, message in invalid_cases:
            with self.subTest(option=option, value=value):
                result = subprocess.run(
                    [str(UNIFIED_RUNNER), option, value, "--show-config"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)

        lower_d16 = subprocess.run(
            [
                str(SCALE_RUNNER), "--queries", "256", "--keys", "256",
                "--head-dim", "16", "--dry-run",
            ],
            cwd=HERE,
            capture_output=True,
            text=True,
        )
        self.assertEqual(lower_d16.returncode, 2)
        self.assertIn("64 or 128", lower_d16.stderr)

        d32 = subprocess.run(
            [str(UNIFIED_RUNNER), "--head-dim", "32", "--show-config"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(d32.returncode, 2)
        self.assertIn("64 or 128", d32.stderr)

    def test_scale_runner_forwards_runtime_shape_to_single_guest(self):
        result = subprocess.run(
            [
                str(SCALE_RUNNER),
                "--queries", "768",
                "--keys", "512",
                "--head-dim", "64",
                "--dry-run",
            ],
            cwd=HERE,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VANADIS_EXE=", result.stdout)
        self.assertIn("/riscv64/fused_attention", result.stdout)
        self.assertIn("GOLEM_ATTENTION_GUEST_MANAGER_QUERIES=192", result.stdout)
        self.assertIn("GOLEM_ATTENTION_GUEST_KEYS=512", result.stdout)
        self.assertIn("GOLEM_ATTENTION_GUEST_HEAD_DIM=64", result.stdout)
        self.assertIn("--queries 768", result.stdout)
        self.assertIn("--keys 512", result.stdout)
        self.assertIn("--head-dim 64", result.stdout)
        self.assertNotIn("--scale-point", result.stdout)

    def test_lower_runner_enforces_tensor_windows_and_uint32_dimensions(self):
        cases = (
            (["--queries", "8448", "--keys", "512", "--head-dim", "128"],
             "query band exceeds"),
            (["--queries", "256", "--keys", "8320", "--head-dim", "128"],
             "K/V shard exceeds"),
            (["--queries", "4294967552", "--keys", "512", "--head-dim", "64"],
             "fit uint32"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [str(SCALE_RUNNER), *arguments, "--dry-run"],
                    cwd=HERE, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)

    def test_baseline_input_cannot_collide_with_generated_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            baseline = root / "attention_baseline_verification.json"
            baseline.write_text(
                (BASELINE_ROOT / "e3" / "result.json").read_text(encoding="ascii"),
                encoding="ascii",
            )
            result = subprocess.run(
                [
                    str(SCALE_RUNNER), "--queries", "1024", "--keys", "1024",
                    "--head-dim", "128", "--artifact-root", str(root),
                    "--baseline", str(baseline), "--dry-run",
                ],
                cwd=HERE, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("conflicts", result.stderr)
            self.assertTrue(baseline.exists())

    def test_rocc_accepts_dimension_driven_streaming_attention(self):
        source = ROCC_SOURCE.read_text()
        self.assertIn("isStreamingAttentionWorkerShape", source)
        self.assertIn("streamingAttentionWindowBytes", source)
        self.assertIn("desc.queries % 64 == 0", source)
        self.assertIn("desc.keys % 128 == 0", source)
        self.assertNotIn("const bool e3Shape", source)

    def test_lifecycle_skew_requires_all_manager_timestamps(self):
        verifier = (HERE / "verify_fused_attention_scale_stats.py").read_text()
        self.assertIn("all_manager_ticks_present", verifier)
        self.assertIn("attention_manager_lifecycle_complete", verifier)

    def test_default_dataflow_releases_operands_and_launches_second_lookahead(self):
        source = ROCC_SOURCE.read_text()
        self.assertIn("recordAttentionKvOperandRelease(true);", source)
        self.assertGreaterEqual(
            source.count("recordAttentionKvOperandRelease(true);"), 2
        )
        self.assertIn("recordAttentionKvOperandRelease(false);", source)
        self.assertGreaterEqual(
            source.count("recordAttentionKvOperandRelease(false);"), 2
        )
        self.assertIn("launchAttentionKvSecondLookahead", source)
        self.assertIn("completeAttentionKvPrefetch(\n            uint64_t generation", source)
        self.assertIn("attentionCallbackGenerationMatches(generation)", source)
        self.assertIn("attention_kv_second_lookahead_prefetches", source)

    def test_attention_dma_response_admission_uses_explicit_consumer_metadata(self):
        global_memory = GLOBAL_MEMORY_SOURCE.read_text()
        memnic = MEMNIC_SOURCE.read_text()
        rocc = ROCC_SOURCE.read_text()

        for token in (
            "DmaConsumerMetadata",
            "jobId",
            "consumerQueryBlock",
            "consumerTile",
            "targetQueryBlock",
            "targetTile",
            "DmaOperand::AttentionK",
            "DmaOperand::AttentionV",
        ):
            self.assertIn(token, global_memory + rocc)
        self.assertIn("DMA_CONSUMER_PROGRESS", global_memory)
        self.assertIn("dma_update_consumer_progress", global_memory + rocc)
        self.assertIn("golemDmaConsumerDistance", memnic)
        self.assertIn("golem_dma_response_distance_wait_cycles_", memnic)
        self.assertIn("golem_dma_response_distance_queue_wait_cycles_", memnic)
        self.assertIn("golem_dma_response_late_ready_", memnic)
        self.assertIn("golem_dma_tile_starvation_", memnic)

    def test_attention_dma_credit_release_and_fallback_are_bounded(self):
        memnic = MEMNIC_SOURCE.read_text()
        dma_config = (HERE.parents[1] / "configs" / "20_dma.env").read_text()
        runner = (HERE.parents[1] / "run_noc_dma_pipeline.sh").read_text()
        self.assertIn("releaseGolemDmaCredits", memnic)
        self.assertIn("golem_dma_response_max_starvation_cycles", memnic)
        self.assertIn("golem_dma_admission_max_starvation_cycles", memnic)
        self.assertIn("golem_dma_consumer_distance_wait_cycles_", memnic)
        self.assertIn("GOLEM_DMA_RESPONSE_MAX_STARVATION_CYCLES:=65536", dma_config)
        self.assertIn("--dma-response-max-starvation-cycles", runner)
        self.assertIn("--dma-admission-max-starvation-cycles", runner)
        self.assertNotIn(
            "golem_dma_credit_available_ += info.creditUnits;\n"
            "                golem_dma_credit_released_requests_++;",
            memnic,
        )

    def test_qk_readout_overlap_is_bounded_and_defaults_off(self):
        rocc = ROCC_SOURCE.read_text()
        compute_array = ARRAY_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()
        runner = SCALE_RUNNER.read_text()

        self.assertIn("attention_qk_readout_overlap", rocc + builder)
        self.assertIn("attention_qk_readout_window", rocc + builder)
        self.assertIn("qkReadCommitIndex", rocc)
        self.assertIn(
            "qkReadIssueIndex - state.qkReadCommitIndex <",
            rocc,
        )
        self.assertIn("outputReadInFlight_ >= arrayOutputReadCredits_", compute_array)
        self.assertIn("outputReadBankInFlight_", compute_array)
        self.assertIn('QK_READOUT_OVERLAP="${GOLEM_ATTENTION_QK_READOUT_OVERLAP:-0}"', runner)
        self.assertIn("--qk-readout-overlap", runner)

    def test_qk_panel_row_burst_reuses_bounded_wcp_c_buffer(self):
        rocc = ROCC_SOURCE.read_text()
        wcp = WCP_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()
        runner = SCALE_RUNNER.read_text()
        unified = UNIFIED_RUNNER.read_text()

        self.assertIn("attention_qk_panel_row_burst", rocc + builder)
        default_on = 'QK_PANEL_ROW_BURST="${GOLEM_ATTENTION_QK_PANEL_ROW_BURST:-1}"'
        self.assertIn(default_on, runner)
        self.assertIn(default_on, unified)
        self.assertIn("--qk-panel-row-burst", runner + unified)
        self.assertIn("--no-qk-panel-row-burst", runner + unified)
        self.assertIn("CBufferMode::ATTENTION_TILE_STORAGE", wcp)
        self.assertIn("CBufferMode::GEMM_PARTIAL_C", wcp)
        self.assertIn("beginAttentionTileStorage", rocc + wcp)
        self.assertIn("endAttentionTileStorage", rocc + wcp)
        self.assertIn("attentionTileStorageColumnValid_", wcp)
        self.assertIn("attentionTileStorageBankNextReadCycle_", wcp)
        self.assertIn("attentionTileStorageBankNextWriteCycle_", wcp)
        self.assertIn("attentionTileStorageRowWriteReadyCycle_", wcp)
        self.assertIn("attention_tile_storage_capacity_rejections", wcp)
        self.assertIn("attention_tile_storage_mode_conflicts", wcp)
        self.assertIn("bytes.size() != static_cast<size_t>(panelKeys) * sizeof(float)", rocc)
        self.assertNotIn("std::vector<std::vector<float>> qkRowBurst", rocc)

    def test_qk_input_pipeline_is_bounded_and_tagged(self):
        rocc = ROCC_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()
        runner = SCALE_RUNNER.read_text()

        self.assertIn("attention_qk_input_pipeline", rocc + builder)
        self.assertIn("std::array<AttentionQkInputSlot, 2>", rocc)
        for field in (
            "generation", "queryBlock", "keyTileOrdinal", "panel",
            "arrayId", "transferTag",
        ):
            self.assertIn(field, rocc)
        self.assertIn("AttentionQkInputSlotState::Reading", rocc)
        self.assertIn("AttentionQkInputSlotState::Ready", rocc)
        self.assertIn("AttentionQkInputSlotState::Programming", rocc)
        self.assertIn("attentionQkInputPipelineDepth", rocc)
        self.assertIn("attention_qk_input_pipeline_tag_mismatches", rocc)
        self.assertIn(
            'QK_INPUT_PIPELINE="${GOLEM_ATTENTION_QK_INPUT_PIPELINE:-1}"',
            runner,
        )

    def test_qk_input_pipeline_default_and_ablation_modes(self):
        clean_env = os.environ.copy()
        clean_env.pop("GOLEM_ATTENTION_QK_INPUT_PIPELINE", None)
        base_args = [
            str(SCALE_RUNNER), "--queries", "256", "--keys", "256",
            "--head-dim", "128", "--dry-run",
        ]
        default_run = subprocess.run(
            base_args, cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(default_run.returncode, 0, default_run.stderr)
        self.assertIn(
            "GOLEM_ATTENTION_QK_INPUT_PIPELINE=1", default_run.stdout,
        )

        disabled_run = subprocess.run(
            base_args[:-1] + ["--no-qk-input-pipeline", "--dry-run"],
            cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(disabled_run.returncode, 0, disabled_run.stderr)
        self.assertIn(
            "GOLEM_ATTENTION_QK_INPUT_PIPELINE=0", disabled_run.stdout,
        )

        invalid_run = subprocess.run(
            base_args[:-1] + [
                "--qk-dataflow-transpose", "--qk-input-pipeline", "--dry-run",
            ],
            cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertNotEqual(invalid_run.returncode, 0)

    def test_qk_panel_row_burst_starts_sram_at_wcp_dispatch(self):
        wcp = WCP_SOURCE.read_text()
        self.assertIn("uint64_t dispatchCycle", wcp)
        self.assertIn("command.storageGeneration, dispatchCycle", wcp)
        self.assertIn(
            "command.index, command.storageGeneration,\n"
            "                    dispatchCycle, data, readyCycle",
            wcp,
        )

    def test_qk_panel_row_burst_cancel_purges_its_generation(self):
        wcp = WCP_SOURCE.read_text()
        self.assertIn("completion.storageGeneration = command.storageGeneration", wcp)
        self.assertIn("purgeAttentionTileStorageCommands", wcp)

    def test_qk_panel_row_burst_bank_resources_are_bounded(self):
        wcp = WCP_SOURCE.read_text()
        self.assertIn("maxRowsPerBank", wcp)
        self.assertIn("attention_tile_storage_banks must be in range [1, 16]", wcp)

    def test_qk_panel_row_burst_default_and_ablation_modes(self):
        base_args = [
            str(SCALE_RUNNER), "--queries", "256", "--keys", "256",
            "--head-dim", "128", "--dry-run",
        ]
        default_run = subprocess.run(
            base_args, cwd=HERE, capture_output=True, text=True,
        )
        self.assertEqual(default_run.returncode, 0, default_run.stderr)
        self.assertIn("GOLEM_ATTENTION_QK_PANEL_ROW_BURST=1", default_run.stdout)

        disabled_run = subprocess.run(
            base_args[:-1] + ["--no-qk-panel-row-burst", "--dry-run"],
            cwd=HERE,
            capture_output=True,
            text=True,
        )
        self.assertEqual(disabled_run.returncode, 0, disabled_run.stderr)
        self.assertIn("GOLEM_ATTENTION_QK_PANEL_ROW_BURST=0", disabled_run.stdout)

        invalid_run = subprocess.run(
            base_args[:-1] + ["--direct-gemm", "--qk-panel-row-burst", "--dry-run"],
            cwd=HERE,
            capture_output=True,
            text=True,
        )
        self.assertEqual(invalid_run.returncode, 2)
        self.assertIn("requires generic GEMM/WCP", invalid_run.stderr)


    def test_qk_panel_row_burst_runner_rejects_oversized_bank_count(self):
        bank_env = os.environ.copy()
        bank_env["GOLEM_ATTENTION_TILE_STORAGE_BANKS"] = "17"
        result = subprocess.run(
            [str(SCALE_RUNNER), "--queries", "256", "--keys", "256",
             "--head-dim", "128", "--dry-run"],
            cwd=HERE,
            env=bank_env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be an integer from 1 through 16", result.stderr)

    def test_pv_residency_and_o_accumulator_use_bounded_existing_storage(self):
        rocc = ROCC_SOURCE.read_text()
        wcp = WCP_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()

        self.assertIn("attention_pv_input_residency", rocc + builder)
        self.assertIn("attentionPvInputResidentQueryBlock", rocc)
        self.assertIn("attentionPvInputResidentKeyTile", rocc)
        self.assertIn("invalidateAttentionPvInputResidency", rocc)
        self.assertIn("beginAttentionStorageSession", rocc + wcp)
        self.assertIn("endAttentionStorageSession", rocc + wcp)
        self.assertIn("attentionStorageQkScratchBytes_", wcp)
        self.assertIn("attentionAccumulatorOffset_", wcp)
        self.assertIn("attentionAccumulatorWriteReadyCycle_", wcp)
        self.assertIn("attentionTileStorageBankNextReadCycle_", wcp)
        self.assertIn("attentionTileStorageBankNextWriteCycle_", wcp)
        self.assertIn("requiredBytes > cBufferBytes_", wcp)
        self.assertIn("attention_o_accumulator_cbuffer", rocc + builder)
        self.assertIn("attentionPvRestoreReadRetry", rocc)
        self.assertIn(
            "if (attentionWorker_ && "
            "attentionWorker_->attentionPvRestoreReadRetry)",
            rocc,
        )

    def test_cluster_pv_pipelines_o_commits_until_the_tile_boundary(self):
        rocc = ROCC_SOURCE.read_text()
        start = rocc.index("    void readAttentionClusterPvOutput()")
        end = rocc.index("    void continueAttentionAfterSoftmaxAndPvMatrix()", start)
        output_path = rocc[start:end]

        self.assertIn(
            "state.index + 1 == attentionQueryRows(state) &&\n"
            "                state.attentionClusterOCommitsPending != 0",
            output_path,
        )
        self.assertNotIn(
            "state.clusterPvOutputInFlight == 0 &&\n"
            "            state.attentionClusterOCommitsPending == 0",
            output_path,
        )
        self.assertIn(
            "attentionWorker_->attentionClusterOCommitsPending == 0",
            rocc,
        )

    def test_cluster_group_commands_reject_permanent_errors_before_enqueue(self):
        array = ARRAY_SOURCE.read_text()
        mvm = MVM_ARRAY_SOURCE.read_text()
        wcp = WCP_SOURCE.read_text()

        self.assertIn("validateOperandContextRequest", array + mvm + wcp)
        self.assertIn("validateOutputGroupRequest", array + mvm + wcp)
        self.assertIn("elemBytes == sizeof(T)", mvm)
        self.assertIn(
            "!array_->validateOperandContextRequest(arrayId, operandBank)",
            wcp,
        )
        self.assertIn(
            "!array_->validateOutputGroupRequest(\n"
            "                arrayIds, elemBytes, trafficClass)",
            wcp,
        )

    def test_near_array_verifier_uses_resolved_hardware_timing(self):
        verifier = (HERE / "verify_fused_attention_scale_stats.py").read_text()
        runner = SCALE_RUNNER.read_text()

        self.assertIn("near_array_output_bytes_per_cycle", verifier)
        self.assertIn("array_buffer_base_latency_cycles", verifier)
        self.assertIn("--near-array-output-bytes-per-cycle", runner)
        self.assertIn("--array-buffer-base-latency-cycles", runner)
        self.assertIn("the two durable PV overlaps", verifier)

    def test_pv_residency_and_o_accumulator_have_independent_ablations(self):
        clean_env = os.environ.copy()
        clean_env.pop("GOLEM_ATTENTION_PV_INPUT_RESIDENCY", None)
        clean_env.pop("GOLEM_ATTENTION_O_ACCUMULATOR_CBUFFER", None)
        base_args = [
            str(SCALE_RUNNER), "--queries", "256", "--keys", "256",
            "--head-dim", "128", "--dry-run",
        ]
        default_run = subprocess.run(
            base_args, cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(default_run.returncode, 0, default_run.stderr)
        self.assertIn(
            "GOLEM_ATTENTION_PV_INPUT_RESIDENCY=1", default_run.stdout,
        )
        self.assertIn(
            "GOLEM_ATTENTION_O_ACCUMULATOR_CBUFFER=0", default_run.stdout,
        )

        enabled_run = subprocess.run(
            base_args[:-1] + [
                "--pv-input-residency", "--o-accumulator-cbuffer", "--dry-run",
            ],
            cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(enabled_run.returncode, 0, enabled_run.stderr)
        self.assertIn(
            "GOLEM_ATTENTION_PV_INPUT_RESIDENCY=1", enabled_run.stdout,
        )
        self.assertIn(
            "GOLEM_ATTENTION_O_ACCUMULATOR_CBUFFER=1", enabled_run.stdout,
        )
        self.assertIn(
            "GOLEM_ATTENTION_KV_PAIR_REUSE=0", enabled_run.stdout,
        )
        self.assertIn("--pv-input-residency", enabled_run.stdout)
        self.assertIn("--o-accumulator-cbuffer", enabled_run.stdout)

        disabled_run = subprocess.run(
            base_args[:-1] + ["--no-pv-input-residency", "--dry-run"],
            cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(disabled_run.returncode, 0, disabled_run.stderr)
        self.assertIn(
            "GOLEM_ATTENTION_PV_INPUT_RESIDENCY=0", disabled_run.stdout,
        )
        self.assertIn("--no-pv-input-residency", disabled_run.stdout)

        invalid_run = subprocess.run(
            base_args[:-1] + [
                "--direct-gemm", "--pv-input-residency", "--dry-run",
            ],
            cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(invalid_run.returncode, 2)
        self.assertIn("require generic GEMM/WCP", invalid_run.stderr)

    def test_kv_query_group_reuse_has_bounded_state_and_v_retention_ablation(self):
        rocc = ROCC_SOURCE.read_text()
        sfu = (HERE.parents[2] / "sfu" / "sfu.cc").read_text()
        builder = CPU_BUILDER.read_text()

        self.assertIn("attention_kv_pair_reuse", rocc + builder)
        self.assertIn("qLocalBuffers", rocc)
        self.assertIn("oLocalBuffers", rocc)
        self.assertIn("attentionHasNextGroupedQuery", rocc)
        self.assertIn("attentionKvGroupOwnerQueryBlock", rocc)
        self.assertIn(
            "targetQueryBlock != attentionKvGroupOwnerQueryBlock(state)", rocc,
        )
        self.assertIn("16u * attentionKvQueryGroupSize", sfu)
        self.assertIn("globalRow % attentionOnlineContexts_.size()", sfu)
        self.assertIn("context.row % attentionOnlineContexts_.size()", sfu)
        self.assertNotIn("attentionOnlineContexts_[contextIndex]", sfu)
        self.assertIn(
            '"16" if attention_kv_pair_reuse else "4"', builder,
        )
        self.assertIn(
            "attention_kv_pair_reuse and int(sfu_row_contexts) < 16", builder,
        )
        self.assertIn("GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE", builder)
        self.assertIn("GOLEM_ATTENTION_PV_V_TILE_GROUP_RETENTION", builder)

        clean_env = os.environ.copy()
        clean_env.pop("GOLEM_ATTENTION_KV_PAIR_REUSE", None)
        clean_env.pop("GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE", None)
        clean_env.pop("GOLEM_ATTENTION_PV_V_TILE_GROUP_RETENTION", None)
        base_args = [
            str(SCALE_RUNNER), "--queries", "1024", "--keys", "1024",
            "--head-dim", "128", "--dry-run",
        ]
        default_run = subprocess.run(
            base_args, cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(default_run.returncode, 0, default_run.stderr)
        self.assertIn("GOLEM_ATTENTION_KV_PAIR_REUSE=1", default_run.stdout)
        self.assertIn("GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE=4", default_run.stdout)
        self.assertIn("GOLEM_ATTENTION_WINDOW_BYTES=149632", default_run.stdout)
        self.assertIn("--kv-query-group-size 4", default_run.stdout)
        self.assertIn("GOLEM_ATTENTION_PV_V_TILE_GROUP_RETENTION=1", default_run.stdout)

        optimized_run = subprocess.run(
            base_args[:-1] + ["--pv-v-tile-group-retention", "--dry-run"],
            cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(optimized_run.returncode, 0, optimized_run.stderr)
        self.assertIn(
            "GOLEM_ATTENTION_PV_V_TILE_GROUP_RETENTION=1",
            optimized_run.stdout,
        )
        self.assertIn("--pv-v-tile-group-retention", optimized_run.stdout)

        retention_disabled_run = subprocess.run(
            base_args[:-1] + ["--no-pv-v-tile-group-retention", "--dry-run"],
            cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(
            retention_disabled_run.returncode, 0,
            retention_disabled_run.stderr,
        )
        self.assertIn(
            "GOLEM_ATTENTION_PV_V_TILE_GROUP_RETENTION=0",
            retention_disabled_run.stdout,
        )

        disabled_run = subprocess.run(
            base_args[:-1] + ["--no-kv-pair-reuse", "--dry-run"],
            cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(disabled_run.returncode, 0, disabled_run.stderr)
        self.assertIn("GOLEM_ATTENTION_KV_PAIR_REUSE=0", disabled_run.stdout)

        two_way_run = subprocess.run(
            base_args[:-1] + ["--kv-query-group-size", "2", "--dry-run"],
            cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(two_way_run.returncode, 0, two_way_run.stderr)
        self.assertIn("GOLEM_ATTENTION_KV_PAIR_REUSE=1", two_way_run.stdout)
        self.assertIn("GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE=2", two_way_run.stdout)
        self.assertIn("GOLEM_ATTENTION_WINDOW_BYTES=116864", two_way_run.stdout)
        self.assertIn("--kv-query-group-size 2", two_way_run.stdout)

        one_way_run = subprocess.run(
            base_args[:-1] + ["--kv-query-group-size", "1", "--dry-run"],
            cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(one_way_run.returncode, 0, one_way_run.stderr)
        self.assertIn("GOLEM_ATTENTION_KV_PAIR_REUSE=0", one_way_run.stdout)
        self.assertIn("GOLEM_ATTENTION_KV_QUERY_GROUP_SIZE=1", one_way_run.stdout)
        self.assertIn("GOLEM_ATTENTION_WINDOW_BYTES=100480", one_way_run.stdout)

        invalid_group_run = subprocess.run(
            base_args[:-1] + ["--kv-query-group-size", "3", "--dry-run"],
            cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(invalid_group_run.returncode, 2)
        self.assertIn("must be 1, 2, or 4", invalid_group_run.stderr)

        incompatible_run = subprocess.run(
            base_args[:-1] + [
                "--kv-pair-reuse", "--o-accumulator-cbuffer", "--dry-run",
            ],
            cwd=HERE, env=clean_env, capture_output=True, text=True,
        )
        self.assertEqual(incompatible_run.returncode, 2)
        self.assertIn("incompatible", incompatible_run.stderr)

    def test_frozen_e3_and_e4_baselines_cover_all_rank_modes(self):
        for profile, queries in (("e3", 1024), ("e4", 2048)):
            for mpi_ranks, relative in (
                (1, "result.json"),
                (2, "mpi2/result.json"),
                (4, "mpi4/result.json"),
            ):
                with self.subTest(profile=profile, mpi_ranks=mpi_ranks):
                    baseline = json.loads(
                        (BASELINE_ROOT / profile / relative).read_text(encoding="ascii")
                    )
                    self.assertEqual(baseline["profile"], profile)
                    self.assertEqual(baseline["shape"]["queries"], queries)
                    self.assertEqual(baseline["shape"]["keys"], queries)
                    self.assertEqual(baseline["topology"]["mpi_ranks"], mpi_ranks)
                    self.assertEqual(baseline["verification"]["mismatches"], 0)
                    architecture = baseline["architecture"]
                    self.assertTrue(architecture["generic_gemm_wcp"])
                    self.assertTrue(architecture["pv_matrix_broadcast"])
                    self.assertTrue(architecture["qk_matrix_broadcast"])
                    self.assertTrue(architecture["kv_double_buffer"])
                    wcp = architecture["wcp_gemm_proxy"]
                    self.assertEqual(wcp["queue_depth"], 32)
                    self.assertEqual(wcp["issue_width"], 1)
                    self.assertEqual(wcp["command_latency_cycles"], 1)
                    self.assertEqual(wcp["completion_latency_cycles"], 1)
                    self.assertEqual(
                        wcp["worker_totals"]["gemm_proxy_queue_full_stalls"], 0
                    )
                    fabric = architecture["matrix_broadcast_fabric"]
                    self.assertEqual(fabric["topology"], "binary_tree")
                    self.assertTrue(fabric["shares_array_buffer_ports"])
                    self.assertEqual(fabric["max_fanout"], 16)
                    self.assertEqual(fabric["bytes_per_cycle"], 64)
                    self.assertEqual(fabric["base_latency_cycles"], 1)
                    self.assertEqual(fabric["stage_latency_cycles"], 1)
                    self.assertEqual(fabric["fanout"], 16)
                    self.assertEqual(fabric["payload_bytes"], 8192)
                    self.assertEqual(fabric["cycles_per_request"], 133)
                    requests = 18432 if profile == "e3" else 73728
                    totals = fabric["worker_totals"]
                    self.assertEqual(
                        totals["matrix_broadcast_requests"], requests
                    )
                    self.assertEqual(totals["matrix_broadcast_rejected"], 0)
                    self.assertEqual(
                        totals["matrix_broadcast_ingress_bytes"], requests * 8192
                    )
                    self.assertEqual(
                        totals["matrix_broadcast_sink_bytes"], requests * 8192 * 16
                    )
                    self.assertEqual(
                        totals["matrix_broadcast_transfer_cycles"], requests * 133
                    )
                    prefetch = baseline["lifecycle"]["worker_critical_path"][
                        "kv_prefetch_timing"
                    ]
                    tiles, query_blocks = (
                        (128, 4) if profile == "e3" else (512, 8)
                    )
                    self.assertEqual(
                        prefetch["counts"]["dma"], tiles - query_blocks
                    )
                    self.assertEqual(
                        prefetch["counts"]["ready_lead"]
                        + prefetch["counts"]["wait"],
                        prefetch["counts"]["dma"],
                    )

    def test_gpu_raw_baseline_and_stage_schema_cover_e3_e4(self):
        schema = json.loads(GPU_SCHEMA.read_text(encoding="ascii"))
        gpu = json.loads(GPU_BASELINE.read_text(encoding="ascii"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertEqual(set(gpu["profiles"]), {"e3", "e4"})
        self.assertEqual(
            gpu["profiles"]["e3"]["fp32_math"]["scope_a"]["median_normalized_cycles"],
            97568,
        )
        self.assertEqual(
            gpu["profiles"]["e4"]["fp32_math"]["stage_chain"]["status"],
            "measured",
        )
        self.assertEqual(
            gpu["profiles"]["e3"]["fp32_math"]["evidence_level"],
            "raw_samples",
        )
        self.assertEqual(
            len(gpu["profiles"]["e3"]["fp32_math"]["scope_a"]["samples_ms"]),
            200,
        )
        self.assertTrue(gpu["audit"]["passed"])
        self.assertEqual(len(gpu["measurement"]["empty_event_samples_ms"]), 1000)

    def test_gpu_collector_emits_finalized_import_contract(self):
        correctness = make_correctness((1, 1, 1024, 128), True, 0.0)
        self.assertEqual(
            set(correctness),
            {
                "passed", "output_shape", "finite", "scope_a_max_abs_error",
                "stage_chain_max_abs_error", "max_abs_error", "tolerance",
            },
        )
        result = make_result(
            "NVIDIA GeForce RTX 5060", (12, 0), "test", "test",
            50, 200, [0.002, 0.004], {"e3": {}},
        )
        self.assertEqual(
            result["benchmark"],
            "single-head FP32 non-causal scaled dot-product attention",
        )
        measurement = result["measurement"]
        self.assertEqual(measurement["empty_event_iterations"], 2)
        self.assertEqual(measurement["empty_event_median_ms"], 0.003)
        self.assertEqual(measurement["empty_event_samples_ms"], [0.002, 0.004])
        self.assertNotIn("empty_event_interval", measurement)

    def test_gpu_comparison_report_preserves_latency_and_work_semantics(self):
        clock = make_clock_contract(
            1_000_000_000, 1_000_000_000_000,
            2_300_000_000, 2_300_000_000,
            2_300_000_000, 2_000_000_000,
        )

        def lifecycle(total_ticks):
            phase_ticks = {
                "kv_load": 100,
                "q_local_read": 100,
                "qk_matrix_program": 200,
                "qk_input_program": 100,
                "qk_compute_readout": 100,
                "softmax": 100,
                "pv_matrix_program": 200,
                "pv_input_program": 100,
                "pv_restore_output": 100,
                "pv_compute": 100,
                "pv_output_readwrite": 100,
            }
            worker_ticks = sum(phase_ticks.values())
            return {
                "status": "PASS",
                "lifecycle": {
                    "clock_contract": clock,
                    "accelerator_completion_ticks": total_ticks,
                    "worker_critical_path": {
                        "slowest_worker_core": 19,
                        "milestone_ticks": {
                            "dispatch_accept": 0,
                            "final_output_dma_ack": total_ticks,
                        },
                        "tile_pipeline_breakdown": {
                            "tile_count": 1,
                            "phase_ticks": phase_ticks,
                            "total_ticks": worker_ticks,
                            "conservation_valid": True,
                        },
                    },
                    "system_frontier": {
                        "stage_ticks": {"descriptor_to_complete": total_ticks},
                        "accelerator_attribution": {
                            "total_ticks": total_ticks,
                            "attributed_ticks": total_ticks,
                            "unattributed_ticks": 0,
                            "coverage_ratio": 1.0,
                            "conservation_valid": True,
                        },
                        "interpretation": "synthetic test",
                    },
                },
            }

        gpu = json.loads(GPU_BASELINE.read_text(encoding="ascii"))
        summary_gpu = json.loads(json.dumps(gpu))
        for profile in ("e3", "e4"):
            fp32 = summary_gpu["profiles"][profile]["fp32_math"]
            fp32["evidence_level"] = "summary_only"
            fp32["scope_a"]["samples_ms"] = None
            fp32["stage_chain"] = {"status": "pending_external_measurement"}
        report = build_report(
            {"e3": lifecycle(1350), "e4": lifecycle(1350)}, summary_gpu
        )
        self.assertEqual(report["sst_report_status"], "PASS")
        self.assertEqual(
            report["gpu_stage_status"], "pending_external_measurement"
        )
        e3 = report["profiles"]["e3"]
        self.assertEqual(
            e3["sst"]["accelerator_completion"]["normalized_cycles"], 2
        )
        self.assertAlmostEqual(
            e3["sst"]["slowest_worker_work"][
                "coverage_of_accelerator_completion_ratio"
            ],
            26 / 27,
        )
        self.assertIn(
            "Do not sum", e3["sst"]["slowest_worker_work"]["interpretation"]
        )

        measured_gpu = json.loads(json.dumps(gpu))
        measured_gpu["measurement"].update({
            "iterations": 2,
            "tf32_allowed": False,
            "dtype_conversion_included": False,
            "stream": "current default stream",
        })
        for profile in ("e3", "e4"):
            fp32 = measured_gpu["profiles"][profile]["fp32_math"]
            fp32["evidence_level"] = "raw_samples"
            fp32["scope_a"].update({
                "median_ms": 0.04,
                "samples_ms": [0.04, 0.04],
                "median_normalized_cycles": 40000,
            })
            fp32["stage_chain"] = {
                "status": "measured",
                "method": "consecutive CUDA Events on one stream; one final synchronize",
                "stage_median_ms": {
                    "qk": 0.01,
                    "scale": 0.002,
                    "softmax": 0.008,
                    "pv": 0.02,
                },
                "stage_samples_ms": {
                    "qk": [0.01, 0.01],
                    "scale": [0.002, 0.002],
                    "softmax": [0.008, 0.008],
                    "pv": [0.02, 0.02],
                },
                "end_to_end_median_ms": 0.04,
                "end_to_end_samples_ms": [0.04, 0.04],
                "output_writeback": "included in the PV matmul event interval",
            }
        measured = build_report(
            {"e3": lifecycle(1350), "e4": lifecycle(1350)}, measured_gpu
        )
        self.assertEqual(measured["gpu_stage_status"], "measured")
        self.assertAlmostEqual(
            measured["profiles"]["e3"]["stage_comparison"]["stages"]
            ["scale_softmax"]["gpu_stage_ms"],
            0.01,
        )

        for case in (
            "h2d", "dtype", "tf32", "missing_samples", "bad_cycles",
            "missing_stream", "missing_method", "bad_correctness",
            "missing_floor_samples", "wrong_gpu", "wrong_benchmark",
        ):
            with self.subTest(case=case):
                invalid = json.loads(json.dumps(measured_gpu))
                if case == "h2d":
                    invalid["measurement"]["h2d_included"] = True
                elif case == "dtype":
                    invalid["measurement"]["dtype_conversion_included"] = True
                elif case == "tf32":
                    invalid["measurement"]["tf32_allowed"] = True
                elif case == "missing_samples":
                    del invalid["profiles"]["e3"]["fp32_math"][
                        "stage_chain"
                    ]["stage_samples_ms"]
                elif case == "missing_stream":
                    del invalid["measurement"]["stream"]
                elif case == "missing_method":
                    del invalid["profiles"]["e3"]["fp32_math"][
                        "stage_chain"
                    ]["method"]
                elif case == "bad_correctness":
                    invalid["profiles"]["e3"]["fp32_math"][
                        "correctness"
                    ]["passed"] = False
                elif case == "missing_floor_samples":
                    del invalid["measurement"]["empty_event_samples_ms"]
                elif case == "wrong_gpu":
                    invalid["hardware"]["gpu"] = "different GPU"
                elif case == "wrong_benchmark":
                    invalid["benchmark"] = "different workload"
                else:
                    invalid["profiles"]["e3"]["fp32_math"]["scope_a"][
                        "median_normalized_cycles"
                    ] = 1
                with self.assertRaises(ValueError):
                    build_report(
                        {"e3": lifecycle(1350), "e4": lifecycle(1350)}, invalid
                    )

        overcovered = build_report(
            {"e3": lifecycle(1200), "e4": lifecycle(1200)}, gpu
        )
        self.assertEqual(overcovered["sst_report_status"], "FAIL")
        self.assertFalse(overcovered["profiles"]["e3"]["sst_coverage_gate"]["pass"])

    def test_build_script_builds_attention_guests(self):
        text = BUILD_SCRIPT.read_text()
        self.assertIn('make -C "$ATTENTION_DIR"', text)
        self.assertIn('make -C "$ATTENTION_DIR" -j"$jobs" scale', text)
        self.assertNotIn("scale-e2", text)
        self.assertNotIn('make -C "$SCRIPT_DIR"', SCALE_RUNNER.read_text())

    def test_v_tile_reuse_has_bounded_buffer_contract(self):
        rocc = ROCC_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()
        self.assertIn('attention_pv_v_tile_buffer_bytes', rocc)
        self.assertIn('attention_pv_v_tile_buffer_hit_ticks', rocc)
        self.assertIn('attention_pv_v_tile_buffer_bytes_per_cycle', rocc)
        self.assertIn('vTileTag', rocc)
        self.assertIn('attention_pv_v_tile_buffer_hits', rocc)
        self.assertIn('attention_pv_v_tile_buffer_bytes_reused', rocc)
        self.assertIn('attention_pv_v_tile_buffer_wait_ticks', rocc)
        self.assertIn('GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES', builder)
        self.assertIn('GOLEM_ATTENTION_PV_V_TILE_BUFFER_HIT_TICKS', builder)
        self.assertIn('GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES_PER_CYCLE', builder)

    def test_v_tile_buffer_overrides_close_runner_and_verifier_contract(self):
        result = subprocess.run(
            [
                str(SCALE_RUNNER), "--queries", "256", "--keys", "256",
                "--head-dim", "64", "--dry-run",
            ],
            cwd=HERE,
            env={
                **os.environ,
                "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES": "32768",
                "GOLEM_ATTENTION_PV_V_TILE_BUFFER_HIT_TICKS": "7",
                "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES_PER_CYCLE": "32",
            },
            check=True,
            capture_output=True,
            text=True,
        )
        for setting in (
            "GOLEM_ATTENTION_WINDOW_BYTES=141440",
            "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES=32768",
            "GOLEM_ATTENTION_PV_V_TILE_BUFFER_HIT_TICKS=7",
            "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES_PER_CYCLE=32",
            "--pv-v-tile-buffer-bytes 32768",
            "--pv-v-tile-buffer-hit-ticks 7",
            "--pv-v-tile-buffer-bytes-per-cycle 32",
        ):
            self.assertIn(setting, result.stdout)

        too_large = subprocess.run(
            [str(SCALE_RUNNER), "--dry-run"],
            cwd=HERE,
            env={
                **os.environ,
                "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES": "262144",
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(too_large.returncode, 2)
        self.assertIn("exceeds the 256 KiB local-GM window", too_large.stderr)

    def test_pv_pipeline_and_active_k_are_default_benchmark_mechanisms(self):
        runner = SCALE_RUNNER.read_text()
        resolved = subprocess.run(
            [str(UNIFIED_RUNNER), "--show-config"],
            check=True,
            capture_output=True,
            text=True,
        )
        for setting in (
            "PV_V_TILE_REUSE=1",
            "PV_INPUT_PIPELINE=1",
            "PV_COMPACT_INPUT=1",
            "PV_RESTORE_PIPELINE=1",
            "PV_OUTPUT_PIPELINE=1",
            "PV_EARLY_COMPUTE=1",
            "PV_MATRIX_SOFTMAX_OVERLAP=1",
            "PV_ACTIVE_K=1",
        ):
            self.assertIn(setting, runner)
        for setting in (
            "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES=16384",
            "GOLEM_ATTENTION_PV_V_TILE_BUFFER_HIT_TICKS=1",
            "GOLEM_ATTENTION_PV_V_TILE_BUFFER_BYTES_PER_CYCLE=64",
            "GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE=1.0",
            "GOLEM_ARRAY_PIPELINE_DEPTH=2",
        ):
            self.assertIn(setting, resolved.stdout)
        for option in (
            "--no-pv-v-tile-reuse",
            "--no-pv-input-pipeline",
            "--no-pv-compact-input",
            "--no-pv-restore-pipeline",
            "--no-pv-output-pipeline",
            "--no-pv-early-compute",
            "--no-pv-matrix-softmax-overlap",
            "--no-pv-active-k",
        ):
            self.assertIn(option, runner)

    def test_pv_active_k_crosses_wcp_and_array_model_boundaries(self):
        compute_array = ARRAY_SOURCE.read_text()
        mvm = MVM_ARRAY_SOURCE.read_text()
        wcp = WCP_SOURCE.read_text()
        rocc = ROCC_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()

        for api in (
            "programMatrixActiveAsync",
            "programMatrixGroupActiveAsync",
            "programInputActiveAsync",
            "beginComputationActive",
            "getArrayLatencyActive",
        ):
            self.assertIn(api, compute_array)
            self.assertIn(api, mvm)
        for api in (
            "programGemmMatrixActiveAsync",
            "programGemmMatrixGroupActiveAsync",
            "programGemmInputActiveAsync",
            "launchGemmArrayActive",
        ):
            self.assertIn(api, wcp)
        for api in (
            "programGemmMatrixActiveBankAsync",
            "programGemmMatrixGroupActiveBankAsync",
            "programGemmInputActiveBankAsync",
            "launchGemmArrayActiveBank",
        ):
            self.assertIn(api, wcp)
            self.assertIn(api, rocc)

        self.assertIn("activeInputColumns", mvm)
        self.assertIn("arrayMacPerCuPerCycle", compute_array)
        self.assertIn("arrayPipelineDepth", compute_array)
        self.assertIn("GOLEM_ARRAY_MAC_PER_CU_PER_CYCLE", builder)
        self.assertIn("GOLEM_ARRAY_PIPELINE_DEPTH", builder)
        self.assertIn("GOLEM_ATTENTION_PV_ACTIVE_K", builder)
        self.assertIn("attention_pv_active_k_launches", rocc)
        self.assertIn("attention_pv_active_k_columns", rocc)
        self.assertIn("attention_pv_active_k_matrix_elements", rocc)
        self.assertIn("LastTickCycle >= attentionWorker_->vTileBufferWaitUntilTick", rocc)
        self.assertIn("message.headDim == 64 || message.headDim == 128", rocc)
        self.assertIn("desc.head_dim == 64 || desc.head_dim == 128", rocc)
        for validator in (
            "validateActiveMatrixRequest",
            "validateActiveInputRequest",
            "validateActiveLaunchRequest",
        ):
            self.assertIn(validator, compute_array)
            self.assertIn(validator, wcp)

    def test_attention_qk_pv_route_through_shared_gemm_engine(self):
        rocc = ROCC_SOURCE.read_text()
        wcp = WCP_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()

        for api in (
            "programGemmMatrixAsync",
            "programGemmMatrixGroupAsync",
            "programGemmInputAsync",
            "writeGemmOutputAsync",
            "readGemmOutputAsync",
            "launchGemmArray",
        ):
            self.assertIn(api, wcp)
        for api in (
            "programGemmMatrixBankAsync",
            "programGemmMatrixGroupBankAsync",
            "programGemmInputBankAsync",
            "writeGemmOutputAsync",
            "readGemmOutputAsync",
            "launchGemmArrayBank",
        ):
            self.assertIn(api, wcp)
            self.assertIn(api, rocc)
        self.assertIn("GOLEM_ATTENTION_GENERIC_GEMM_ENABLE", builder)
        self.assertIn('"1" if attention_fused else "0"', builder)
        self.assertIn("workerCommandProcessor->isBusy()", rocc)
        self.assertIn("attention_generic_gemm_enable", rocc)
        self.assertIn("attention_generic_gemm_qk_ops", rocc)
        self.assertIn("attention_generic_gemm_pv_ops", rocc)

    def test_bounded_cross_tile_operand_pipeline_contract(self):
        rocc = ROCC_SOURCE.read_text()
        compute_array = ARRAY_SOURCE.read_text()
        mvm = MVM_ARRAY_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()
        runner = SCALE_RUNNER.read_text()

        for token in (
            "AttentionAheadOperandContext",
            "attention_cross_tile_operand_pipeline",
            "attention_cross_tile_operand_promotions",
            "attention_cross_tile_operand_tag_mismatches",
        ):
            self.assertIn(token, rocc)
        self.assertIn("operandContextBanks", compute_array)
        self.assertIn("hasPendingBufferTransfers", compute_array)
        self.assertIn("operandIndex", mvm)
        self.assertIn("arrayID >= numArrays", mvm)
        self.assertIn("attentionOperandContextBanks_ != 2", rocc)
        self.assertIn("GOLEM_ARRAY_OPERAND_CONTEXT_BANKS", builder)
        self.assertIn("--no-cross-tile-operand-pipeline", runner)

        default = subprocess.run(
            [str(WRAPPER), "--dry-run"], cwd=HERE, check=True,
            capture_output=True, text=True,
        )
        self.assertIn(
            "GOLEM_ATTENTION_CROSS_TILE_OPERAND_PIPELINE=1", default.stdout
        )
        self.assertIn("GOLEM_ARRAY_OPERAND_CONTEXT_BANKS=2", default.stdout)
        direct = subprocess.run(
            [str(WRAPPER), "--direct-gemm", "--dry-run"], cwd=HERE,
            check=True, capture_output=True, text=True,
        )
        self.assertIn(
            "GOLEM_ATTENTION_CROSS_TILE_OPERAND_PIPELINE=0", direct.stdout
        )
        self.assertIn("GOLEM_ARRAY_OPERAND_CONTEXT_BANKS=1", direct.stdout)
        group1 = subprocess.run(
            [str(WRAPPER), "--kv-query-group-size", "1", "--dry-run"],
            cwd=HERE, check=True, capture_output=True, text=True,
        )
        self.assertIn(
            "GOLEM_ATTENTION_CROSS_TILE_OPERAND_PIPELINE=0", group1.stdout
        )
        self.assertIn("GOLEM_ARRAY_OPERAND_CONTEXT_BANKS=1", group1.stdout)
        invalid_group1 = subprocess.run(
            [
                str(WRAPPER),
                "--kv-query-group-size", "1",
                "--cross-tile-operand-pipeline",
                "--dry-run",
            ],
            cwd=HERE, check=False, capture_output=True, text=True,
        )
        self.assertEqual(invalid_group1.returncode, 2)
        self.assertIn("grouped K/V reuse", invalid_group1.stderr)

    def test_attention_generic_gemm_models_wcp_control_timing(self):
        result = subprocess.run(
            [str(WRAPPER), "--dry-run"],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )
        wcp = WCP_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()
        wcp_builder = builder.split(
            '"worker_command_processor", "golem.WorkerCommandProcessorLocal"', 1
        )[1].split('cpu_rocc.setSubComponent(', 1)[0]

        for env_name, default in (
            ("GOLEM_WCP_GEMM_PROXY_QUEUE_DEPTH", "32"),
            ("GOLEM_WCP_GEMM_PROXY_ISSUE_WIDTH", "1"),
            ("GOLEM_WCP_GEMM_PROXY_COMMAND_LATENCY_CYCLES", "1"),
            ("GOLEM_WCP_GEMM_PROXY_COMPLETION_LATENCY_CYCLES", "1"),
        ):
            self.assertIn(env_name, builder)
            self.assertIn(f"{env_name}={default}", result.stdout)

        for param in (
            "gemm_proxy_queue_depth",
            "gemm_proxy_issue_width",
            "gemm_proxy_command_latency_cycles",
            "gemm_proxy_completion_latency_cycles",
        ):
            self.assertIn(param, wcp)
            self.assertIn(param, wcp_builder)
        self.assertIn("workerCommandProcessor.enableAllStatistics()", wcp_builder)

        self.assertIn("gemmProxyCommands_", wcp)
        self.assertIn("pendingGemmCompletions_", wcp)
        for statistic in (
            "gemm_proxy_commands_issued",
            "gemm_proxy_queue_full_stalls",
            "gemm_proxy_queue_wait_cycles",
            "gemm_proxy_launch_commands",
            "gemm_proxy_completion_callbacks",
            "gemm_proxy_completion_delay_cycles",
        ):
            self.assertIn(statistic, wcp)
        self.assertIn(
            "--wcp-gemm-proxy-completion-latency-cycles 1", result.stdout
        )
        verifier = (HERE / "verify_fused_attention_scale_stats.py").read_text()
        self.assertIn('lifecycle["wcp_gemm_proxy"]', verifier)
        self.assertIn("gemm_completions * wcp_gemm_proxy_completion_latency_cycles", verifier)

    def test_attention_wcp_control_timing_rejects_invalid_configuration(self):
        invalid_environments = (
            {"GOLEM_WCP_GEMM_PROXY_QUEUE_DEPTH": "15"},
            {"GOLEM_WCP_GEMM_PROXY_ISSUE_WIDTH": "0"},
            {"GOLEM_WCP_GEMM_PROXY_COMMAND_LATENCY_CYCLES": "-1"},
            {"GOLEM_WCP_GEMM_PROXY_COMMAND_LATENCY_CYCLES": "0"},
            {"GOLEM_WCP_GEMM_PROXY_COMPLETION_LATENCY_CYCLES": "0"},
            {"GOLEM_WCP_GEMM_PROXY_COMPLETION_LATENCY_CYCLES": "bad"},
        )
        for override in invalid_environments:
            with self.subTest(override=override):
                result = subprocess.run(
                    [str(WRAPPER), "--dry-run"],
                    cwd=HERE,
                    env={**os.environ, **override},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("WCP", result.stderr)

    def test_v_tile_buffer_expected_activity(self):
        self.assertEqual(
            expected_v_tile_buffer_activity(PROFILES["e3"], True, 16384, 1),
            {
                "hits": 896,
                "misses": 128,
                "bytes_read": 2 * 1024 * 1024,
                "bytes_reused": 1792 * 1024,
                "wait_ticks": 29568,
                "capacity_rejections": 0,
                "group_hits": 0,
            },
        )
        self.assertEqual(
            expected_v_tile_buffer_activity(PROFILES["e3"], True, 8192, 3),
            {
                "hits": 0,
                "misses": 1024,
                "bytes_read": 16 * 1024 * 1024,
                "bytes_reused": 0,
                "wait_ticks": 0,
                "capacity_rejections": 1024,
                "group_hits": 0,
            },
        )
        self.assertEqual(
            expected_v_tile_buffer_activity(PROFILES["e3"], False, 16384, 1),
            {
                "hits": 0,
                "misses": 0,
                "bytes_read": 0,
                "bytes_reused": 0,
                "wait_ticks": 0,
                "capacity_rejections": 0,
                "group_hits": 0,
            },
        )
        self.assertEqual(
            expected_v_tile_buffer_activity(
                PROFILES["e3"], True, 16384, 1, 64, True, 32,
            ),
            {
                "hits": 992,
                "misses": 32,
                "bytes_read": 512 * 1024,
                "bytes_reused": 1984 * 1024,
                "wait_ticks": 32736,
                "capacity_rejections": 0,
                "group_hits": 96,
            },
        )

    def test_partition_verifier_accepts_query_block_placement(self):
        with tempfile.TemporaryDirectory() as directory:
            stats_file = pathlib.Path(directory) / "stats_selfcom.txt"
            placement_file = self.write_placement_manifest(directory, 2)
            for rank in range(2):
                rows = ["ComponentName,Rank"]
                rows.extend(
                    f"core{core_id}:rocc,{rank}"
                    for core_id in range(20)
                    if core_id % 2 == rank
                )
                rows.extend(
                    f"rtr_{router_id},{rank}"
                    for router_id in range(28)
                    if (router_id % 2 if router_id < 24 else 0) == rank
                )
                stats_file.with_name(f"stats_selfcom_{rank}.txt").write_text(
                    "\n".join(rows) + "\n", encoding="ascii"
                )
            result = subprocess.run(
                [
                    "python3",
                    str(HERE / "verify_attention_mpi_partition.py"),
                    "--stats-file",
                    str(stats_file),
                    "--mpi-ranks",
                    "2",
                    "--placement-file",
                    str(placement_file),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"status": "PASS"', result.stdout)

    def test_partition_verifier_rejects_misplaced_core(self):
        with tempfile.TemporaryDirectory() as directory:
            stats_file = pathlib.Path(directory) / "stats_selfcom.txt"
            placement_file = self.write_placement_manifest(directory, 2)
            for rank in range(2):
                rows = ["ComponentName,Rank"]
                rows.extend(
                    f"core{core_id}:rocc,{0 if core_id == 1 else rank}"
                    for core_id in range(20)
                    if core_id % 2 == rank
                )
                rows.extend(
                    f"rtr_{router_id},{rank}"
                    for router_id in range(28)
                    if (router_id % 2 if router_id < 24 else 0) == rank
                )
                stats_file.with_name(f"stats_selfcom_{rank}.txt").write_text(
                    "\n".join(rows) + "\n", encoding="ascii"
                )
            result = subprocess.run(
                [
                    "python3",
                    str(HERE / "verify_attention_mpi_partition.py"),
                    "--stats-file",
                    str(stats_file),
                    "--mpi-ranks",
                    "2",
                    "--placement-file",
                    str(placement_file),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('"status": "FAIL"', result.stdout)

    def test_partition_verifier_accepts_four_rank_placement(self):
        with tempfile.TemporaryDirectory() as directory:
            stats_file = pathlib.Path(directory) / "stats_selfcom.txt"
            placement_file = self.write_placement_manifest(directory, 4)
            for rank in range(4):
                rows = ["ComponentName,Rank"]
                rows.extend(
                    f"core{core_id}:rocc,{rank}"
                    for core_id in range(20)
                    if core_id % 4 == rank
                )
                rows.extend(
                    f"rtr_{router_id},{rank}"
                    for router_id in range(28)
                    if (router_id % 4 if router_id < 24 else 0) == rank
                )
                stats_file.with_name(f"stats_selfcom_{rank}.txt").write_text(
                    "\n".join(rows) + "\n", encoding="ascii"
                )
            result = subprocess.run(
                [
                    "python3",
                    str(HERE / "verify_attention_mpi_partition.py"),
                    "--stats-file",
                    str(stats_file),
                    "--mpi-ranks",
                    "4",
                    "--placement-file",
                    str(placement_file),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"status": "PASS"', result.stdout)

    def test_partition_verifier_rejects_misplaced_memory_component(self):
        with tempfile.TemporaryDirectory() as directory:
            stats_file = pathlib.Path(directory) / "stats_selfcom.txt"
            placement_file = self.write_placement_manifest(
                directory, 2, {"memory_2": 0}
            )
            for rank in range(2):
                rows = ["ComponentName,Rank"]
                rows.extend(
                    f"core{core_id}:rocc,{rank}"
                    for core_id in range(20)
                    if core_id % 2 == rank
                )
                rows.extend(
                    f"rtr_{router_id},{rank}"
                    for router_id in range(28)
                    if (router_id % 2 if router_id < 24 else 0) == rank
                )
                stats_file.with_name(f"stats_selfcom_{rank}.txt").write_text(
                    "\n".join(rows) + "\n", encoding="ascii"
                )
            result = subprocess.run(
                [
                    "python3",
                    str(HERE / "verify_attention_mpi_partition.py"),
                    "--stats-file",
                    str(stats_file),
                    "--mpi-ranks",
                    "2",
                    "--placement-file",
                    str(placement_file),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('"memory_2"', result.stdout)
            self.assertIn('"status": "FAIL"', result.stdout)

    def test_partition_verifier_rejects_manifest_integrity_errors(self):
        for case in ("missing", "unexpected", "rank_count"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                stats_file = self.write_ranked_stats(directory, 2)
                placement_file = self.write_placement_manifest(directory, 2)
                placement = json.loads(placement_file.read_text(encoding="ascii"))
                if case == "missing":
                    del placement["component_ranks"]["os"]
                elif case == "unexpected":
                    placement["component_ranks"]["foreign_component"] = 0
                else:
                    placement["mpi_ranks"] = 4
                placement_file.write_text(json.dumps(placement), encoding="ascii")
                result, report = self.run_partition_verifier(
                    stats_file, placement_file, 2
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(report["status"], "FAIL")

    def test_partition_verifier_rejects_rank_file_integrity_errors(self):
        for case in ("missing", "unexpected", "reported_rank"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                stats_file = self.write_ranked_stats(directory, 2)
                placement_file = self.write_placement_manifest(directory, 2)
                if case == "missing":
                    stats_file.with_name("stats_selfcom_1.txt").unlink()
                elif case == "unexpected":
                    stats_file.with_name("stats_selfcom_2.txt").write_text(
                        "ComponentName,Rank\n", encoding="ascii"
                    )
                else:
                    rank_one = stats_file.with_name("stats_selfcom_1.txt")
                    rank_one.write_text(
                        rank_one.read_text(encoding="ascii").replace(",1\n", ",0\n"),
                        encoding="ascii",
                    )
                result, report = self.run_partition_verifier(
                    stats_file, placement_file, 2
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
