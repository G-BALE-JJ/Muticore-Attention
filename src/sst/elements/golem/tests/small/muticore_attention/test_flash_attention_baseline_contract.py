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
from verify_flash_attention_baseline import build_checks
from verify_fused_attention_scale_stats import (
    attention_cluster_ii_sample_counts,
    expected_attention_cluster_broadcast_activity,
    expected_matrix_broadcast_activity,
    expected_sequential_qk_score_readout,
    expected_sequential_64_broadcast_activity,
    expected_sequential_64_scatter_activity,
    expected_v_tile_buffer_activity,
    make_attention_activity,
    make_sequential_64_activity,
    make_clock_contract,
    manager_slot_stalls_valid,
    parse_dma_runtime_invariants,
    summarize_attention_cluster_resource_profile,
)
from attention_case import generate_case


WRAPPER = HERE / "run_flash_attention.sh"
SCALE_RUNNER = HERE / "run_fused_attention_scale.sh"
UNIFIED_RUNNER = HERE.parents[6] / "scripts" / "test_flash_attention.sh"
BUILD_SCRIPT = HERE.parents[6] / "scripts" / "build_and_install_local.sh"
ARCHIVE_ARCH = HERE.parents[1] / "architecture" / "archive" / "ncores_selfcom_dma.py"
ROCC_SOURCE = HERE.parents[2] / "rocc" / "roccAnalog.h"
ROCC_FLOAT = HERE.parents[2] / "rocc" / "roccAnalogFloat.h"
ROCC_INT = HERE.parents[2] / "rocc" / "roccAnalogInt.h"
WCP_SOURCE = HERE.parents[2] / "workercmdproc" / "workercmdproc.h"
CPU_BUILDER = HERE.parents[1] / "architecture" / "cpu_builder.py"
ARRAY_SOURCE = HERE.parents[2] / "array" / "computeArray.h"
MVM_ARRAY_SOURCE = HERE.parents[2] / "array" / "mvmComputeArray.h"
GLOBAL_MEMORY_SOURCE = HERE.parents[2] / "globalmemory" / "globalmemory.h"
GLOBAL_MEMORY_IMPL = HERE.parents[2] / "globalmemory" / "globalmemory.cc"
SFU_SOURCE = HERE.parents[2] / "sfu" / "sfu.cc"
SFU_HEADER = HERE.parents[2] / "sfu" / "sfu.h"
ATTENTION_CLUSTER_HEADER = HERE.parents[2] / "attention" / "attentionCluster.h"
GROUP_CTRL_HEADER = HERE.parents[2] / "groupctrl" / "groupctrl.h"
GROUP_CTRL_SOURCE = HERE.parents[2] / "groupctrl" / "groupctrl.cc"
MEMNIC_SOURCE = HERE.parents[3] / "memHierarchy" / "memNICBase.h"
BASELINE_VERIFIER = HERE / "verify_flash_attention_baseline.py"
ATTENTION_GUEST = HERE / "golem_attention_runtime.cpp"
HBM_GENERATOR = HERE.parents[1] / "tools" / "gen_hbm_init.py"


class FlashAttentionBaselineContractTest(unittest.TestCase):
    def test_attention_uses_canonical_shape_and_transport_terminology(self):
        configured = subprocess.run(
            [
                str(UNIFIED_RUNNER),
                "--query-length", "1024", "--kv-length", "1024",
                "--num-query-heads", "4", "--num-kv-heads", "1",
                "--show-config",
            ],
            check=True, capture_output=True, text=True,
        )
        values = dict(line.split("=", 1) for line in configured.stdout.splitlines())
        self.assertEqual(values["QUERY_LENGTH"], "1024")
        self.assertEqual(values["KV_LENGTH"], "1024")
        self.assertEqual(values["NUM_QUERY_HEADS"], "4")
        self.assertEqual(values["NUM_KV_HEADS"], "1")

        guest_header = (HERE / "golem_attention_runtime.h").read_text(encoding="utf-8")
        rocc = ROCC_SOURCE.read_text(encoding="utf-8")
        transport = GLOBAL_MEMORY_SOURCE.read_text(encoding="utf-8")
        self.assertIn("struct GolemAttentionDescV2", guest_header)
        for field in (
            "group_query_rows", "kv_length", "query_tile_rows", "kv_tile_rows",
            "num_query_heads", "num_kv_heads", "group_query_row_begin",
        ):
            self.assertIn(field, guest_header)
        self.assertNotIn("struct GolemAttentionDescV1", guest_header)
        self.assertIn("ControlTransportMessage", transport)
        self.assertIn("controlNetworkAvailable", transport)
        self.assertIn("qkReductionSlice", rocc)
        self.assertIn("pvOutputSlice", rocc)
        rocc_variants = ROCC_FLOAT.read_text() + ROCC_INT.read_text()
        self.assertIn("attention_cluster_qk_k_tile_broadcasts", rocc_variants)
        self.assertIn("attention_cluster_qk_k_tile_bytes", rocc_variants)

    def test_gqa_cli_and_tensor_cardinality_contract(self):
        configured = subprocess.run(
            [
                str(UNIFIED_RUNNER), "--query-heads", "4", "--kv-heads", "1",
                "--show-config",
            ],
            check=True, capture_output=True, text=True,
        )
        values = dict(line.split("=", 1) for line in configured.stdout.splitlines())
        self.assertEqual(values["QUERY_HEADS"], "4")
        self.assertEqual(values["KV_HEADS"], "1")
        self.assertEqual(values["GQA_GROUP_SIZE"], "4")

        invalid = subprocess.run(
            [
                str(UNIFIED_RUNNER), "--query-heads", "6", "--kv-heads", "4",
                "--show-config",
            ],
            capture_output=True, text=True,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("divisible", invalid.stderr)

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            q_file, k_file, v_file = root / "q.bin", root / "k.bin", root / "v.bin"
            generate_case(
                256, 128, 128, q_file, k_file, v_path=v_file,
                heads=4, kv_heads=1,
            )
            self.assertEqual(q_file.stat().st_size, 4 * 256 * 128 * 4)
            self.assertEqual(k_file.stat().st_size, 1 * 128 * 128 * 4)
            self.assertEqual(v_file.stat().st_size, 1 * 128 * 128 * 4)

    def test_gqa_uses_composite_kv_group_jobs_and_worker_dispatch_fifo(self):
        guest_header = (HERE / "golem_attention_runtime.h").read_text(encoding="utf-8")
        guest = ATTENTION_GUEST.read_text(encoding="utf-8")
        rocc = ROCC_SOURCE.read_text(encoding="utf-8")
        transport = GLOBAL_MEMORY_SOURCE.read_text(encoding="utf-8")
        memnic = MEMNIC_SOURCE.read_text(encoding="utf-8")

        for field in ("num_query_heads", "num_kv_heads", "kv_head_index"):
            self.assertIn(field, guest_header)
            self.assertIn(field, rocc)
        self.assertIn("gqa_group_size", guest)
        self.assertIn("for (uint32_t kv_head = 0; kv_head < num_kv_heads; ++kv_head)", guest)
        self.assertIn("attentionPendingDispatches_", rocc)
        self.assertIn("numQueryHeads", transport)
        self.assertIn("servedConsumers", memnic)

    def test_gqa_writes_direct_query_major_concatenated_output(self):
        rocc = ROCC_SOURCE.read_text(encoding="utf-8")
        transport = GLOBAL_MEMORY_SOURCE.read_text(encoding="utf-8")
        verifier = (HERE / "verify_fused_attention_scale_output.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("queryLength", transport)
        self.assertIn("groupQueryRowBegin", transport)
        self.assertIn("issueAttentionOutputDmaRows", rocc)
        self.assertIn("kScatterDmaWindow = 16", rocc)
        self.assertIn("state.dispatch.numQueryHeads + queryHead", rocc)
        self.assertIn('"output_layout": "query_major_[query_length,num_query_heads,head_dim]"', verifier)

    def test_multi_head_compatibility_alias_uses_gqa_interfaces(self):
        unified = UNIFIED_RUNNER.read_text(encoding="utf-8")
        runner = SCALE_RUNNER.read_text(encoding="utf-8")
        guest = ATTENTION_GUEST.read_text(encoding="utf-8")
        architecture = ARCHIVE_ARCH.read_text(encoding="utf-8")
        hbm_generator = HBM_GENERATOR.read_text(encoding="utf-8")

        self.assertIn("--heads N", unified)
        self.assertIn("NUM_QUERY_HEADS=1", unified)
        self.assertIn('GOLEM_ATTENTION_NUM_QUERY_HEADS=$NUM_QUERY_HEADS', runner)
        self.assertIn('GOLEM_ATTENTION_GUEST_NUM_QUERY_HEADS=$NUM_QUERY_HEADS', runner)
        self.assertIn('"GOLEM_ATTENTION_GUEST_NUM_QUERY_HEADS"', architecture)
        self.assertIn("for (uint32_t kv_head = 0; kv_head < num_kv_heads; ++kv_head)", guest)
        self.assertIn("query_head_stride", guest)
        self.assertIn("kv_head_stride", guest)
        self.assertIn("ATTENTION_NUM_QUERY_HEADS", hbm_generator)

    def test_unified_runner_exposes_multi_head_configuration(self):
        default = subprocess.run(
            [str(UNIFIED_RUNNER), "--show-config"],
            check=True, capture_output=True, text=True,
        )
        default_config = dict(
            line.split("=", 1) for line in default.stdout.splitlines()
        )
        self.assertEqual(default_config["HEADS"], "1")

        configured = subprocess.run(
            [str(UNIFIED_RUNNER), "--heads", "4", "--show-config"],
            check=True, capture_output=True, text=True,
        )
        configured_values = dict(
            line.split("=", 1) for line in configured.stdout.splitlines()
        )
        self.assertEqual(configured_values["HEADS"], "4")

    def test_sfu_uses_mac_equivalent_physical_operator_pipelines(self):
        source = SFU_SOURCE.read_text(encoding="utf-8")
        header = SFU_HEADER.read_text(encoding="utf-8")
        builder = CPU_BUILDER.read_text(encoding="utf-8")

        for unit in (
            "scalePipeline_",
            "maxReductionPipeline_",
            "expPipeline_",
            "onlineExpPipeline_",
            "reciprocalPipeline_",
            "normalizePipeline_",
        ):
            self.assertIn(unit, header + source)
        self.assertIn("struct TensorHardwarePipeline", header)
        self.assertIn("initiationInterval", header)
        self.assertIn("completionCycles", header)
        self.assertIn("TensorRowEngineEventKind::HardwareRetry", source)
        self.assertIn("TensorRowEngineEventKind::OnlineAddDone", source)
        self.assertIn("TensorRowEngineEventKind::ReciprocalDone", source)
        self.assertIn("GOLEM_SFU_EXP_II", builder)
        self.assertIn("GOLEM_SFU_PIPELINE_QUEUE_DEPTH", builder)
        self.assertNotIn("rowEngineVectorFreeCycle_", header + source)
        self.assertNotIn("rowEngineExpFreeCycle_", header + source)

    def test_sfu_exposes_split_math_unit_latency_and_ii_parameters(self):
        header = SFU_HEADER.read_text(encoding="utf-8")
        source = SFU_SOURCE.read_text(encoding="utf-8")
        builder = CPU_BUILDER.read_text(encoding="utf-8")
        combined = header + source
        for unit in (
            "scalePipeline_", "maxComparePipeline_", "maxReductionPipeline_",
            "expPipeline_", "sumReductionPipeline_", "onlineMaxPipeline_",
            "onlineExpPipeline_", "onlineMulPipeline_", "onlineAddPipeline_",
            "reciprocalPipeline_", "normalizePipeline_",
        ):
            self.assertIn(unit, combined)
        for parameter in (
            "GOLEM_SFU_SCALE_LATENCY", "GOLEM_SFU_SCALE_II",
            "GOLEM_SFU_MAX_COMPARE_LATENCY", "GOLEM_SFU_MAX_COMPARE_II",
            "GOLEM_SFU_MAX_REDUCTION_LATENCY", "GOLEM_SFU_MAX_REDUCTION_II",
            "GOLEM_SFU_SUM_REDUCTION_LATENCY", "GOLEM_SFU_SUM_REDUCTION_II",
            "GOLEM_SFU_ONLINE_MAX_LATENCY", "GOLEM_SFU_ONLINE_MAX_II",
            "GOLEM_SFU_ONLINE_EXP_LATENCY", "GOLEM_SFU_ONLINE_EXP_II",
            "GOLEM_SFU_ONLINE_MUL_LATENCY", "GOLEM_SFU_ONLINE_MUL_II",
            "GOLEM_SFU_ONLINE_ADD_LATENCY", "GOLEM_SFU_ONLINE_ADD_II",
        ):
            self.assertIn(parameter, builder)
        self.assertNotIn("onlineUpdatePipeline_", combined)
        self.assertNotIn("GOLEM_SFU_ONLINE_UPDATE_LATENCY", builder)

    def test_sequential_sfu_coalesces_each_score_and_p_tile(self):
        header = SFU_HEADER.read_text(encoding="utf-8")
        source = SFU_SOURCE.read_text(encoding="utf-8")
        builder = CPU_BUILDER.read_text(encoding="utf-8")

        self.assertIn("TensorRowEngineEventKind::TileStreamReadDone", source)
        self.assertIn("TensorRowEngineEventKind::TileStreamWriteDone", source)
        self.assertIn("issueAttentionTileStreamRead", header + source)
        self.assertIn("issueAttentionTileStreamWrite", header + source)
        self.assertIn("tileScoreValues", header)
        self.assertIn("tilePValues", header)
        self.assertIn("sfu_tensor_tile_stream_read_requests", source)
        self.assertIn("sfu_tensor_tile_stream_write_requests", source)
        self.assertIn("sfu_tensor_tile_stream_bytes_per_cycle", source)
        self.assertIn("GOLEM_SFU_TILE_STREAM_ENABLE", builder)
        self.assertIn(
            'GOLEM_SFU_TILE_STREAM_BYTES_PER_CYCLE", "256"', builder
        )
        self.assertIn("GOLEM_SFU_TILE_STREAM_BASE_LATENCY_CYCLES", builder)
        self.assertIn("sfu_tile_stream_bytes_per_cycle", builder)
        self.assertIn('"tile_stream_bytes_per_cycle"', builder)

    def test_sequential_sfu_paces_row_dispatch_at_four_cycles(self):
        header = SFU_HEADER.read_text(encoding="utf-8")
        source = SFU_SOURCE.read_text(encoding="utf-8")
        builder = CPU_BUILDER.read_text(encoding="utf-8")

        self.assertIn("TensorRowEngineEventKind::RowDispatch", source)
        self.assertIn("scheduleAttentionRowDispatch", header + source)
        self.assertIn("rowDispatchIntervalCycles_", header + source)
        self.assertIn("tensorRowDispatches_", header + source)
        self.assertIn(
            'GOLEM_SFU_ROW_DISPATCH_INTERVAL_CYCLES", "4"', builder
        )
        self.assertIn('"row_dispatch_interval_cycles"', builder)

    def test_k_first_cross_query_wait_is_classified(self):
        source = ROCC_SOURCE.read_text(encoding="utf-8")
        start = source.index("void continueAttentionAfterVReady()")
        end = source.index("void beginAttentionSoftmax()", start)
        body = source[start:end]

        self.assertIn("if (descriptor.crossQuery)", body)
        self.assertIn("statAttentionKvCrossQueryHits_->addData(1)", body)
        self.assertIn("statAttentionKvCrossQueryWaits_->addData(1)", body)

    def test_sequential_qk_group_readout_supports_tail_query_rows(self):
        source = ROCC_SOURCE.read_text(encoding="utf-8")
        start = source.index("void readAttentionSequentialQkTile()")
        end = source.index("void readAttentionQkOutput()", start)
        body = source[start:end]

        self.assertNotIn("rows != 64", body)
        self.assertIn("rows > arrayOutputSize", body)
        self.assertIn("arrayOutputSize + query", body)

    def test_sequential_vector_scatter_accepts_partial_lane_groups(self):
        source = WCP_SOURCE.read_text(encoding="utf-8")
        start = source.index("bool programGemmInputScatterBankAsync(",
                             source.index("class WorkerCommandProcessorLocal"))
        end = source.index("bool programGemmMatrixActiveBankAsync(", start)
        body = source[start:end]

        self.assertNotIn("arrayIds.size() != 64", body)
        self.assertIn("arrayIds.size() > 64", body)

    def test_sequential_pv_output_group_accepts_tail_rows(self):
        source = MVM_ARRAY_SOURCE.read_text(encoding="utf-8")
        start = source.index(
            "trafficClass == AttentionClusterTrafficClass::SequentialPvORestore"
        )
        end = source.index(
            "if (arrayIDs.size() != 2", start
        )
        body = source[start:end]

        self.assertNotIn("arrayIDs.size() != 64", body)
        self.assertIn("arrayIDs.size() > 64", body)

    def test_public_attention_runner_defaults_to_sequential_64(self):
        runner = WRAPPER.read_text(encoding="utf-8")
        unified = UNIFIED_RUNNER.read_text(encoding="utf-8")
        self.assertIn("GOLEM_ATTENTION_SEQUENTIAL_64_ENABLE", runner)
        self.assertNotIn("--attention-cluster", runner)
        self.assertNotIn("--partitioned-qk-pv", runner)
        self.assertNotIn("--no-sequential-64", runner)
        self.assertIn("KV_PAIR_REUSE=0", unified)
        self.assertIn("KV_QUERY_GROUP_SIZE=1", unified)

        rejected = subprocess.run(
            [str(WRAPPER), "--attention-cluster", "--show-config"],
            cwd=HERE, capture_output=True, text=True,
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("Unknown", rejected.stderr)

    def test_attention_sfu_keeps_intermediate_rows_resident(self):
        source = SFU_SOURCE.read_text(encoding="utf-8")
        self.assertIn("const bool rowResident = worker.localTileMode;", source)
        self.assertIn("const bool loadWholeRow", source)
        self.assertIn("context.residentValues.begin()", source)
        max_stage = source[source.index(
            "if (stage == TensorRowEngineStage::Max)"
        ):source.index("if (stage == TensorRowEngineStage::ExpSum)", source.index(
            "if (stage == TensorRowEngineStage::Max)"
        ))]
        self.assertIn("residentValues.assign", max_stage)

    def test_attention_cluster_uses_fine_grained_sfu_slots_and_pv_draining(self):
        rocc = ROCC_SOURCE.read_text(encoding="utf-8")
        cluster = ATTENTION_CLUSTER_HEADER.read_text(encoding="utf-8")
        wcp = WCP_SOURCE.read_text(encoding="utf-8")
        self.assertIn("static constexpr uint32_t kSlots = 16;", cluster)
        self.assertIn("constexpr uint32_t sliceRows = 4;", rocc)
        self.assertIn("std::array<int32_t, 16> reuseWindowScoreOwners", rocc)
        self.assertIn("if (ready) {", rocc)
        self.assertNotIn("rowContextSwitchThreshold", rocc)
        self.assertIn("attentionWorkerClusterPvDraining_", rocc)
        self.assertIn(
            "dispatchAttentionReuseWindowPvAggregates(*draining);", rocc
        )
        self.assertIn("constexpr size_t cacheEntries = 4;", wcp)
        self.assertIn("commitAttentionPvVectorCache();", wcp)
        self.assertIn(
            "usesAttentionPvPanels() && attentionPvVectorResidentHit_", wcp
        )
        self.assertIn("attention_pv_cache_gm_addr", wcp)
        self.assertIn("attention_pv_cache_bytes", wcp)
        self.assertIn(
            "reuseNIndex_ * totalKTileCount_ + activeTxnKBegin_ + local_tile_idx",
            wcp,
        )
        self.assertIn("LocalMemoryClient::WCP", wcp)
        self.assertIn("attentionPvVectorReadInFlight_", wcp)
        cache_entry = wcp[wcp.index("struct AttentionPvVectorCacheEntry"):
                          wcp.index("void commitAttentionPvVectorCache")]
        self.assertNotIn("std::vector<uint8_t> payload", cache_entry)

    def test_attention_cluster_uses_local_gm_for_v_residency_at_256_bpc(self):
        runner = (HERE.parents[6] / "baseline" / "attention_cluster" /
                  "run_sst.sh").read_text(encoding="utf-8")
        rocc = ROCC_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            'GOLEM_LOCAL_GM_BYTES_PER_CYCLE="${GOLEM_LOCAL_GM_BYTES_PER_CYCLE:-256}"',
            runner,
        )
        self.assertIn("header.attention_pv_cache_gm_addr", rocc)
        self.assertIn("header.attention_pv_cache_bytes", rocc)

    def test_8qk_8pv_prioritizes_rows_and_tree_broadcasts_v_into_local_sram(self):
        runner = (HERE.parents[6] / "baseline" / "attention_cluster_8qk_8pv" /
                  "run_sst.sh").read_text(encoding="utf-8")
        rocc = ROCC_SOURCE.read_text(encoding="utf-8")
        wcp = WCP_SOURCE.read_text(encoding="utf-8")

        self.assertIn("GOLEM_ATTENTION_WORKER_CLUSTER_ROW_PRIORITY", runner)
        self.assertIn("GOLEM_ATTENTION_WORKER_CLUSTER_V_BROADCAST", runner)
        self.assertIn("AttentionClusterPvVPrefetchHint", rocc)
        self.assertIn("deliveryChunkBytes = 16u * 1024u", rocc)
        self.assertIn("distributeAttentionWorkerClusterV", rocc)
        self.assertIn("installAttentionPvVector", rocc)
        self.assertIn("localWriteAsync", wcp)

    def test_8qk_8pv_dynamic_assignment_binds_rows_and_isolates_accumulators(self):
        runner = (HERE.parents[6] / "baseline" / "attention_cluster_8qk_8pv" /
                  "run_sst.sh").read_text(encoding="utf-8")
        rocc = ROCC_SOURCE.read_text(encoding="utf-8")
        transport = GLOBAL_MEMORY_SOURCE.read_text(encoding="utf-8")

        self.assertIn("GOLEM_ATTENTION_WORKER_CLUSTER_DYNAMIC_PV", runner)
        self.assertIn("AttentionClusterPvAllocationRequest", transport)
        self.assertIn("AttentionClusterPvAllocationResponse", transport)
        self.assertIn("AttentionClusterPvAllocationRelease", transport)
        self.assertIn("reuseWindowPvRowCore", rocc)
        self.assertIn("attentionWorkerClusterPvOutputs_", rocc)
        self.assertIn("const uint32_t manager = (message.ownerCore - 4u) % 4u", rocc)
        self.assertIn("const uint32_t alternate = selected == manager ? manager + 4u : manager", rocc)

    def test_dma_landing_retries_transient_local_memory_backpressure(self):
        source = GLOBAL_MEMORY_IMPL.read_text(encoding="utf-8")
        begin = source.index(
            "void GlobalMemoryImplement::issueNextDmaLandingChunk"
        )
        end = source.index(
            "void GlobalMemoryImplement::finishDmaReadLanding", begin
        )
        landing = source[begin:end]
        self.assertIn("landing.chunkInFlight = true;", landing)
        self.assertIn("scheduleDmaLandingRetry();", landing)
        self.assertIn("retryDmaLandingChunks()", landing)
        rejected = landing[landing.index("if (accepted)") :]
        self.assertNotIn("finishDmaReadLanding(failedOp, false)", rejected)

    def test_sequential_qk_reduction_restarts_for_every_key_tile(self):
        source = ROCC_SOURCE.read_text(encoding="utf-8")
        begin = source.index("    void beginAttentionKvTile()")
        end = source.index("    bool attentionClusterCallbackMatches", begin)
        key_tile_entry = source[begin:end]
        self.assertIn(
            "attentionWorker_->qkReductionSlice = 0;", key_tile_entry
        )

    def test_sequential_qk_reads_and_writes_only_the_final_score_tile(self):
        rocc = ROCC_SOURCE.read_text(encoding="utf-8")
        array = MVM_ARRAY_SOURCE.read_text(encoding="utf-8")
        traffic = (HERE.parents[2] / "attention" / "attentionCluster.h").read_text(
            encoding="utf-8"
        )

        self.assertIn("SequentialQkScoreOut", traffic)
        self.assertIn("AttentionClusterTrafficClass::SequentialQkScoreOut", array)
        self.assertIn("arrayIDs.size() == 64", array)

        begin = rocc.index("    void completeAttentionSequentialQkWave()")
        end = rocc.index("    void readAttentionQkOutput()", begin)
        final_only = rocc[begin:end]
        self.assertIn("state.qkReductionSlice == 0", final_only)
        self.assertIn("beginAttentionQkKvSubtile();", final_only)
        self.assertIn("readAttentionSequentialQkTile();", final_only)
        self.assertEqual(final_only.count("readAttentionSequentialQkTile();"), 1)

        begin = rocc.index("    void readAttentionSequentialQkTile()")
        end = rocc.index("    void readAttentionQkOutput()", begin)
        readout = rocc[begin:end]
        self.assertIn("readAttentionSequentialQkGroupAsync", readout)
        self.assertEqual(readout.count("attentionLocalWrite("), 1)

    def test_sequential_pv_reuses_one_full_v_tile_across_d64_panels(self):
        source = ROCC_SOURCE.read_text(encoding="utf-8")
        begin = source.index("    void beginAttentionSequentialPvWave()")
        end = source.index(
            "    void programAttentionSequentialPvMatrices()", begin
        )
        pv_entry = source[begin:end]
        self.assertIn("state.phaseSliceIndex != 0", pv_entry)
        self.assertIn("state.vPayload.size() == tileValues", pv_entry)
        self.assertIn("programAttentionSequentialPvMatrices();", pv_entry)

    def test_sequential_64_uses_independent_256_byte_vector_scatter(self):
        result = subprocess.run(
            [
                str(SCALE_RUNNER), "--queries", "1024", "--keys", "1024",
                "--head-dim", "128", "--sequential-64", "--dry-run",
            ],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE=256", result.stdout)
        self.assertIn("GOLEM_INPUT_SCATTER_BYTES_PER_CYCLE=256", result.stdout)

        array = ARRAY_SOURCE.read_text(encoding="utf-8")
        mvm = MVM_ARRAY_SOURCE.read_text(encoding="utf-8")
        wcp = WCP_SOURCE.read_text(encoding="utf-8")
        rocc = ROCC_SOURCE.read_text(encoding="utf-8")
        builder = CPU_BUILDER.read_text(encoding="utf-8")
        for source in (array, builder):
            self.assertIn("inputScatterBytesPerCycle", source)
        self.assertIn("enqueueInputScatterTransfer", array)
        self.assertIn("programInputScatterBankAsync", mvm)
        self.assertIn("programGemmInputScatterBankAsync", wcp)

        qk_begin = rocc.index("    void programAttentionQkInput()")
        qk_end = rocc.index("    bool attentionQkInputSlotMatches", qk_begin)
        qk = rocc[qk_begin:qk_end]
        self.assertIn("programAttentionSequentialInputScatterAsync", qk)
        self.assertIn("state.index != activeArrays", qk)

        pv_begin = rocc.index("    void programAttentionSequentialPvInputs()")
        pv_end = rocc.index("    void prepareAttentionSequentialPvOutputs()", pv_begin)
        pv = rocc[pv_begin:pv_end]
        self.assertIn("programAttentionSequentialInputScatterAsync", pv)
        self.assertNotIn("sequentialPvInputRow", pv)

    def test_sequential_64_uses_grouped_256_byte_o_scatter_gather(self):
        result = subprocess.run(
            [
                str(SCALE_RUNNER), "--queries", "1024", "--keys", "1024",
                "--head-dim", "128", "--sequential-64", "--dry-run",
            ],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn(
            "GOLEM_OUTPUT_SCATTER_GATHER_BYTES_PER_CYCLE=256",
            result.stdout,
        )

        array = ARRAY_SOURCE.read_text(encoding="utf-8")
        mvm = MVM_ARRAY_SOURCE.read_text(encoding="utf-8")
        wcp = WCP_SOURCE.read_text(encoding="utf-8")
        rocc = ROCC_SOURCE.read_text(encoding="utf-8")
        builder = CPU_BUILDER.read_text(encoding="utf-8")
        for source in (array, builder):
            self.assertIn("outputScatterGatherBytesPerCycle", source)
        self.assertIn("enqueueOutputScatterGatherTransfer", array)
        self.assertIn("writeOutputGroupClassAsync", mvm)
        self.assertIn("writeGemmOutputGroupClassAsync", wcp)

        restore_begin = rocc.index(
            "    void prepareAttentionSequentialPvOutputs()"
        )
        restore_end = rocc.index(
            "    void launchAttentionSequentialPvComputation()", restore_begin
        )
        restore = rocc[restore_begin:restore_end]
        self.assertIn("writeAttentionSequentialPvOutputGroupAsync", restore)
        self.assertEqual(restore.count("attentionLocalRead("), 2)
        self.assertNotIn("++attentionWorker_->index", restore)

        output_begin = rocc.index(
            "    void readAttentionSequentialPvOutputs()"
        )
        output_end = rocc.index(
            "    bool attentionClusterPvMatrixMatches", output_begin
        )
        output = rocc[output_begin:output_end]
        self.assertIn("readAttentionSequentialPvOutputGroupAsync", output)
        self.assertEqual(output.count("attentionLocalWrite("), 2)
        self.assertNotIn("++attentionWorker_->index", output)

    def test_sequential_pv_pipeline_separates_matrix_input_and_restore(self):
        source = ROCC_SOURCE.read_text(encoding="utf-8")
        matrix_begin = source.index(
            "    void programAttentionSequentialPvMatrices()"
        )
        input_begin = source.index(
            "    void programAttentionSequentialPvInputs()", matrix_begin
        )
        restore_begin = source.index(
            "    void prepareAttentionSequentialPvOutputs()", input_begin
        )
        launch_begin = source.index(
            "    void launchAttentionSequentialPvComputation()", restore_begin
        )

        matrix_program = source[matrix_begin:input_begin]
        self.assertIn(
            "AttentionTilePipelinePhase::PvInputProgram", matrix_program
        )

        input_program = source[input_begin:restore_begin]
        self.assertIn("attentionWorker_->kvTileIndex != 0", input_program)
        self.assertIn(
            "AttentionTilePipelinePhase::PvRestoreOutput", input_program
        )

        restore_output = source[restore_begin:launch_begin]
        self.assertNotIn(
            "AttentionTilePipelinePhase::PvMatrixProgram", restore_output
        )

    def test_sequential_64_mode_reaches_the_guest_environment(self):
        architecture = ARCHIVE_ARCH.read_text(encoding="utf-8")
        self.assertIn(
            '"GOLEM_ATTENTION_SEQUENTIAL_64_ENABLE",', architecture
        )

    def test_sequential_64_broadcast_contract(self):
        activity = make_sequential_64_activity(256, 128, 128, 64)
        expected = expected_sequential_64_broadcast_activity(
            activity, 256, 1, 1
        )
        self.assertEqual(expected["requests"], 8)
        self.assertEqual(expected["ingress_bytes"], 131072)
        self.assertEqual(expected["sink_bytes"], 5242880)
        self.assertEqual(expected["transfer_cycles"], 560)
        self.assertEqual(expected["fanout_sum"], 320)
        self.assertEqual(expected["max_observed_fanout_expected"], 64)
        scatter = expected_sequential_64_scatter_activity(activity, 256, 1)
        self.assertEqual(scatter["requests"], 6)
        self.assertEqual(scatter["bytes"], 73728)
        self.assertEqual(scatter["transfer_cycles"], 294)

    def test_sequential_64_q1024_activity_uses_full_worker_blocks(self):
        activity = make_sequential_64_activity(1024, 1024, 128, 64)
        self.assertEqual(activity["qblocks"], 1)
        self.assertEqual(activity["jobs"], 16)
        self.assertEqual(activity["rows"], 1024)
        self.assertEqual(activity["qk"], 2048)
        self.assertEqual(activity["pv"], 2048)
        expected = expected_sequential_64_broadcast_activity(
            activity, 256, 1, 1
        )
        self.assertEqual(expected["requests"], 64)
        self.assertEqual(expected["transfer_cycles"], 4544)
        scatter = expected_sequential_64_scatter_activity(activity, 256, 1)
        self.assertEqual(scatter["requests"], 48)
        self.assertEqual(scatter["transfer_cycles"], 3120)

        score_readout = expected_sequential_qk_score_readout(activity)
        self.assertEqual(score_readout["requests"], 16)
        self.assertEqual(score_readout["bytes"], 262144)

    def test_manager_slot_stalls_count_queued_worker_requests(self):
        self.assertTrue(manager_slot_stalls_valid(124, 122, 4, 4))
        self.assertTrue(manager_slot_stalls_valid(124, 124, 0, 4))
        self.assertFalse(manager_slot_stalls_valid(124, 122, 9, 4))

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

    def test_dma_runtime_invariants_prefer_mpi_safe_compact_summary(self):
        compact_credit_lines = "".join(
            f"GOLEM_MEMNIC_DMA_CREDIT_CONSERVATION "
            f"name=dirctrl_{node}:highlink cap=128 available=128 "
            f"admitted={40 if node else 0} released={40 if node else 0} "
            "pending=0\n"
            for node in range(5)
        )
        diagnostic_credit_lines = "".join(
            f"CREDIT_OWNER_SUMMARY name=dirctrl_{node}:highlink cap=128 "
            f"available=128 admitted={40 if node else 0} "
            f"released={40 if node else 0} pending=0\n"
            for node in range(5)
        )
        compact_lines = "".join(
            f"GOLEM_MEMNIC_DMA_RESPONSE_CONSERVATION "
            f"component=dirctrl_{node}:highlink attempted=40 immediate=10 "
            "enqueued=30 drained=30 pending=0 responses_d0=10 "
            "responses_d1=10 responses_d2=10 responses_far=10\n"
            for node in range(1, 5)
        )
        interleaved_long_lines = (
            "CREDIT_OWNER_SUMMARY name=dirctrl_1:highlink cap=128 "
            "available=128 admitted=40 released=40 [rank output interleaved]\n"
            "GOLEM_MEMNIC_DMA_RESPONSE_STATS component=dirctrl_1:highlink "
            "attempted=40 immediate=10 enqueued=30 [rank output interleaved]\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime_log = pathlib.Path(directory) / "runtime.log"
            runtime_log.write_text(
                compact_credit_lines + diagnostic_credit_lines
                + compact_lines + interleaved_long_lines,
                encoding="ascii",
            )
            result = parse_dma_runtime_invariants(runtime_log)
            self.assertEqual(result["mismatches"], {})
            self.assertEqual(len(result["credit_owners"]), 5)
            self.assertEqual(len(result["response_nodes"]), 4)

        memnic = MEMNIC_SOURCE.read_text(encoding="utf-8")
        self.assertIn("GOLEM_MEMNIC_DMA_CREDIT_CONSERVATION", memnic)

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
        self.assertNotIn("/data/", text)

    def test_attention_runner_selects_reuse_windows_for_bridge_only(self):
        runner = SCALE_RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            'GOLEM_A_REUSE_N_TILES=$((REUSE_WINDOW_QK_BRIDGE ? 4 : 1))',
            runner,
        )
        self.assertIn(
            'GOLEM_B_REUSE_M_TILES=$((REUSE_WINDOW_QK_BRIDGE ? 2 : 1))',
            runner,
        )
        self.assertIn(
            "GOLEM_ATTENTION_KV_BUFFER_COUNT=$KV_BUFFER_COUNT_EFFECTIVE",
            runner,
        )
        self.assertIn("GOLEM_DMA_READ_MAX_RETRIES=32", runner)

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
            ("GOLEM_MATRIX_BROADCAST_MAX_FANOUT", "64"),
            ("GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE", "256"),
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

        activity = make_attention_activity(1024, 1024, 128)
        expected = expected_matrix_broadcast_activity(
            activity, True, False, False, 16, 64, 1, 1
        )
        self.assertEqual(expected["requests"], 1024)
        self.assertEqual(expected["payload_bytes"], 8192)
        self.assertEqual(expected["ingress_bytes"], 8 * 1024 * 1024)
        self.assertEqual(expected["sink_bytes"], 128 * 1024 * 1024)
        self.assertEqual(expected["tree_stages"], 4)
        self.assertEqual(expected["cycles_per_request"], 133)
        self.assertEqual(expected["transfer_cycles"], 136192)

        mixed_active = expected_matrix_broadcast_activity(
            activity, True, True, False, 16, 64, 1, 1, True
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

    def test_scale_runner_uses_model_native_cycle_contract(self):
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

    def test_attention_archive_hbm_node_link_matches_320_gbps_stack(self):
        architecture = ARCHIVE_ARCH.read_text(encoding="utf-8")
        runner = SCALE_RUNNER.read_text(encoding="utf-8")
        dirctrl = architecture[architecture.index(
            'dir_hi = dirctrl.setSubComponent("highlink", "memHierarchy.MemNIC")'
        ):architecture.index(
            'dir_lo = dirctrl.setSubComponent("lowlink", "memHierarchy.MemLink")'
        )]
        self.assertIn(
            'os.getenv("GOLEM_DIRCTRL_HIGHLINK_BW", "320GB/s")', dirctrl
        )
        self.assertNotIn('"network_bw": "25GB/s"', dirctrl)
        self.assertIn(
            'DIRCTRL_HIGHLINK_BW="${GOLEM_DIRCTRL_HIGHLINK_BW:-320GB/s}"',
            runner,
        )
        self.assertIn(
            '"GOLEM_DIRCTRL_HIGHLINK_BW=$DIRCTRL_HIGHLINK_BW"', runner
        )

    def test_attention_runner_instantiates_ramulator2_hbm2e(self):
        runner = SCALE_RUNNER.read_text()
        architecture = ARCHIVE_ARCH.read_text()

        self.assertIn("GOLEM_MEMORY_BACKEND=ramulator2", runner)
        self.assertIn("hbm2e_2500.yaml", runner)
        self.assertIn(
            'MEMORY_BACKEND = os.getenv("GOLEM_MEMORY_BACKEND", "ramulator2")',
            architecture,
        )
        self.assertIn('f"memHierarchy.{MEMORY_BACKEND}"', architecture)
        self.assertIn('"configFile": RAMULATOR2_CONFIG', architecture)
        self.assertIn('"request_width": 32', architecture)

        result = subprocess.run(
            [str(WRAPPER), "--dry-run"],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("GOLEM_MEMORY_BACKEND=ramulator2", result.stdout)
        self.assertIn("hbm2e_2500.yaml", result.stdout)
        self.assertIn(
            'grep -Fq "RAMULATOR2_BACKEND_SUMMARY"', runner,
        )
        self.assertIn(
            'grep -Fq "DRAMSIM3_BACKEND_"', runner,
        )

    def test_unified_runner_uses_worktree_install_and_explicit_parameters(self):
        text = UNIFIED_RUNNER.read_text()
        verifier = BASELINE_VERIFIER.read_text()
        self.assertIn('source "$SCRIPT_DIR/env_local_install.sh"', text)
        self.assertIn('"$ATTENTION_DIR/run_flash_attention.sh"', text)
        self.assertIn('--query-length "$QUERY_LENGTH"', text)
        self.assertIn('--kv-length "$KV_LENGTH"', text)
        self.assertIn('--head-dim "$HEAD_DIM"', text)
        self.assertIn('--timeout "$TIMEOUT" --artifact-root "$ARTIFACT_ROOT"', text)
        self.assertIn('export SST_LIB_PATH="$WORKTREE_ROOT/install/lib/sst-elements-library"', text)
        self.assertIn('--mpi-ranks) MPI_RANKS="$2"', text)
        self.assertIn('MPI_RANKS=4', text)
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
        self.assertIn('GOLEM_MATRIX_BROADCAST_BYTES_PER_CYCLE=256', text)
        self.assertIn('GOLEM_INPUT_SCATTER_BYTES_PER_CYCLE=256', text)
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
        self.assertEqual(config["KV_PAIR_REUSE"], "0")
        self.assertEqual(config["KV_QUERY_GROUP_SIZE"], "1")

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
        self.assertEqual(group_one_config["MPI_RANKS"], "4")
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
        with tempfile.TemporaryDirectory() as directory:
            baseline = pathlib.Path(directory) / "baseline.json"
            baseline.write_text(json.dumps({
                "shape": {"queries": 1024, "keys": 1024, "head_dim": 128},
                "topology": {"mpi_ranks": 1},
            }), encoding="ascii")
            result = subprocess.run(
                [
                    "python3", str(BASELINE_VERIFIER),
                    "--baseline", str(baseline),
                    "--mpi-ranks", "1", "--queries", "768", "--keys", "512",
                    "--head-dim", "64", "--preflight-only",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("baseline.shape", result.stderr)

    def test_generic_baseline_comparison_is_shape_driven(self):
        baseline = {
            "shape": {"queries": 1024, "keys": 1024, "head_dim": 128},
            "topology": {"mpi_ranks": 1},
            "verification": {
                "status": "PASS", "checked": 131072, "mismatches": 0,
                "score_probability_hbm_bytes": 0, "max_abs_error": 0.0,
            },
            "lifecycle": {
                "status": "PASS", "accelerator_completion_cycles": 10,
                "wait_return_cycles": 11,
                "worker_critical_path": {"kv_prefetch_timing": {}},
            },
            "architecture": {
                "generic_gemm_wcp": False,
                "pv_matrix_broadcast": False,
                "qk_matrix_broadcast": False,
                "kv_double_buffer": False,
                "wcp_gemm_proxy": {},
                "matrix_broadcast_fabric": {},
            },
        }
        result = {
            "status": "PASS", "checked": 131072, "mismatches": 0,
            "shape": {"queries": 1024, "keys": 1024, "head_dim": 128},
            "score_probability_hbm_bytes": 0, "max_abs_error": 0.0,
        }
        lifecycle = {
            "status": "PASS",
            "lifecycle": {
                "accelerator_completion_cycles": 10,
                "wait_return_cycles": 11,
                "worker_critical_path": {
                    "order_valid": True,
                    "inter_tile_breakdown": {"conservation_valid": True},
                    "kv_prefetch_timing": {},
                },
                "wcp_gemm_proxy": {},
                "matrix_broadcast_fabric": {
                    "pv_enabled": False, "qk_enabled": False,
                },
            },
        }
        checks = build_checks(baseline, result, lifecycle, 1)
        self.assertTrue(all(checks.values()))

        result["shape"]["queries"] = 768
        checks = build_checks(baseline, result, lifecycle, 1)
        self.assertFalse(checks["verification.shape"])

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
            ("--queries", "257", "--query-length must be divisible by 256"),
            ("--keys", "129", "--kv-length must be divisible by 128"),
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
        self.assertIn("--query-length 768", result.stdout)
        self.assertIn("--kv-length 512", result.stdout)
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
            baseline.write_text("{}\n", encoding="ascii")
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
        self.assertIn("desc.group_query_rows % 64 == 0", source)
        self.assertIn("desc.kv_length % 128 == 0", source)
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
            "consumerQueryTile",
            "consumerKvTileIndex",
            "targetQueryTile",
            "targetKvTileIndex",
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

    def test_sequential_attention_uses_shared_streaming_kv_supply(self):
        memnic = MEMNIC_SOURCE.read_text()
        architecture = ARCHIVE_ARCH.read_text()
        runner = SCALE_RUNNER.read_text()
        rocc = ROCC_SOURCE.read_text()

        for token in (
            "golem_dma_kv_coalesce_enable",
            "golem_dma_kv_multicast_bytes_per_cycle",
            "GolemDmaKvSubscriber",
            "GolemDmaKvCacheEntry",
            "tryCoalesceGolemDmaRead",
            "tryServeGolemDmaKvCache",
            "golem_dma_kv_physical_reads_",
            "golem_dma_kv_coalesced_requests_",
            "golem_dma_kv_cache_hits_",
            "golem_dma_kv_multicast_receivers_",
            "golem_dma_kv_multicast_bytes_",
            "GOLEM_MEMNIC_DMA_KV_MULTICAST_STATS",
        ):
            self.assertIn(token, memnic)
        self.assertIn("GOLEM_ATTENTION_KV_SHARED_STREAM_ENABLE", architecture)
        self.assertIn("GOLEM_ATTENTION_KV_STREAM_BYTES_PER_CYCLE", architecture)
        self.assertIn("GOLEM_ATTENTION_KV_CONSUMERS_PER_NODE", architecture)
        self.assertIn("attentionWaitingForV", rocc)
        self.assertIn("continueAttentionAfterVReady", rocc)
        self.assertIn("--no-kv-shared-stream", runner)

        base_args = [
            str(SCALE_RUNNER), "--sequential-64", "--queries", "1024",
            "--keys", "1024", "--head-dim", "128", "--dry-run",
        ]
        enabled = subprocess.run(
            base_args, cwd=HERE, check=True, capture_output=True, text=True,
        )
        disabled = subprocess.run(
            base_args[:-1] + ["--no-kv-shared-stream", "--dry-run"],
            cwd=HERE, check=True, capture_output=True, text=True,
        )
        self.assertIn("GOLEM_ATTENTION_KV_SHARED_STREAM_ENABLE=1", enabled.stdout)
        self.assertIn("GOLEM_ATTENTION_KV_STREAM_BYTES_PER_CYCLE=256", enabled.stdout)
        self.assertIn("GOLEM_ATTENTION_KV_BUFFER_COUNT=2", enabled.stdout)
        self.assertIn("GOLEM_ATTENTION_KV_SHARED_STREAM_ENABLE=0", disabled.stdout)

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

    def test_qk_score_row_burst_reuses_bounded_wcp_c_buffer(self):
        rocc = ROCC_SOURCE.read_text()
        wcp = WCP_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()
        runner = SCALE_RUNNER.read_text()
        unified = UNIFIED_RUNNER.read_text()

        self.assertIn("attention_qk_score_row_burst", rocc + builder)
        self.assertIn("GOLEM_ATTENTION_QK_SCORE_ROW_BURST", runner + unified)
        self.assertIn("--qk-score-row-burst", runner)
        self.assertIn("--no-qk-score-row-burst", runner)
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
        self.assertIn("bytes.size() != static_cast<size_t>(kvSubtileRows) * sizeof(float)", rocc)
        self.assertNotIn("std::vector<std::vector<float>> qkRowBurst", rocc)

    def test_qk_input_pipeline_is_bounded_and_tagged(self):
        rocc = ROCC_SOURCE.read_text()
        builder = CPU_BUILDER.read_text()
        runner = SCALE_RUNNER.read_text()

        self.assertIn("attention_qk_input_pipeline", rocc + builder)
        self.assertIn("std::array<AttentionQkInputSlot, 2>", rocc)
        for field in (
            "generation", "queryTileIndex", "kvTileIndex", "phaseSliceIndex",
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

    def test_qk_score_row_burst_starts_sram_at_wcp_dispatch(self):
        wcp = WCP_SOURCE.read_text()
        self.assertIn("uint64_t dispatchCycle", wcp)
        self.assertIn("command.storageGeneration, dispatchCycle", wcp)
        self.assertIn(
            "command.index, command.storageGeneration,\n"
            "                    dispatchCycle, data, readyCycle",
            wcp,
        )

    def test_qk_score_row_burst_cancel_purges_its_generation(self):
        wcp = WCP_SOURCE.read_text()
        self.assertIn("completion.storageGeneration = command.storageGeneration", wcp)
        self.assertIn("purgeAttentionTileStorageCommands", wcp)

    def test_qk_score_row_burst_bank_resources_are_bounded(self):
        wcp = WCP_SOURCE.read_text()
        self.assertIn("maxRowsPerBank", wcp)
        self.assertIn("attention_tile_storage_banks must be in range [1, 16]", wcp)

    def test_qk_score_row_burst_runner_rejects_oversized_bank_count(self):
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
        self.assertIn("attentionPvInputResidentQueryTile", rocc)
        self.assertIn("attentionPvInputResidentKvTile", rocc)
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

    def test_near_array_verifier_uses_resolved_hardware_timing(self):
        verifier = (HERE / "verify_fused_attention_scale_stats.py").read_text()
        runner = SCALE_RUNNER.read_text()

        self.assertIn("near_array_output_bytes_per_cycle", verifier)
        self.assertIn("array_buffer_base_latency_cycles", verifier)
        self.assertIn("--near-array-output-bytes-per-cycle", runner)
        self.assertIn("--array-buffer-base-latency-cycles", runner)
        self.assertIn("the two durable PV overlaps", verifier)


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
        activity = make_attention_activity(1024, 1024, 128)
        self.assertEqual(
            expected_v_tile_buffer_activity(activity, True, 16384, 1),
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
            expected_v_tile_buffer_activity(activity, True, 8192, 3),
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
            expected_v_tile_buffer_activity(activity, False, 16384, 1),
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
                activity, True, 16384, 1, 64, True, 32,
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
