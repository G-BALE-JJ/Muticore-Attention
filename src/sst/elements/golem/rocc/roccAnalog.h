// Copyright 2009-2025 NTESS. Under the terms
// of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2025, NTESS
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// of the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#ifndef _H_ANALOG_ROCC
#define _H_ANALOG_ROCC

#include <sst/core/output.h>
#include <sst/core/component.h>
#include <sst/core/subcomponent.h>
#include <sst/core/statapi/statbase.h>
#include <sst/core/interfaces/stdMem.h>
#include <sst/elements/golem/array/computeArray.h>
#include <sst/elements/golem/groupctrl/groupctrl.h>
#include <sst/elements/golem/requestscheduler/requestscheduler.h>
#include <sst/elements/golem/sfu/sfu.h>
#include <sst/elements/golem/workercmdproc/workercmdproc.h>
#include <sst/elements/golem/attention/attentionCluster.h>
#include <sst/elements/vanadis/rocc/vroccinterface.h>
#include <sst/elements/golem/globalmemory/globalmemory.h>
#include <sst/elements/golem/fp16.h>


#include <array>
#include <cinttypes>
#include <algorithm>
#include <cmath>
#include <functional>
#include <deque>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <map>
#include <numeric>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <iostream>

using namespace SST::Interfaces;
using namespace SST::Golem;

namespace SST {
namespace Golem {

constexpr uint32_t GOLEM_ROCC_FLAG_SYNC_MATRIX = 0x0;
constexpr uint32_t GOLEM_ROCC_FLAG_SYNC_VECTOR = 0x1;
constexpr uint32_t GOLEM_ROCC_FLAG_ASYNC_BASE = 0x80000000u;
constexpr uint32_t GOLEM_ROCC_FLAG_ASYNC_MATRIX = 0x40000000u;
constexpr uint32_t GOLEM_ROCC_FLAG_ASYNC_ARRAY_SHIFT = 16u;
constexpr uint32_t GOLEM_ROCC_FLAG_ASYNC_ARRAY_MASK = 0x00FF0000u;
constexpr uint8_t GOLEM_ROCC_FUNC7_TILE_MVM_BATCH = 0x11;
constexpr uint8_t GOLEM_ROCC_FUNC7_TILE_WAIT_BATCH = 0x12;
constexpr uint8_t GOLEM_ROCC_FUNC7_TILE_GM2IMAT_BCAST = 0x13;
constexpr uint8_t GOLEM_ROCC_FUNC7_TILE_GM2IVEC_BATCH = 0x14;
constexpr uint8_t GOLEM_ROCC_FUNC7_WCP_START = 0x15;
constexpr uint8_t GOLEM_ROCC_FUNC7_WCP_WAIT = 0x16;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE = 0x17;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_WAIT = 0x18;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_PRIMITIVE = 0x19;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_WAIT = 0x1a;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH = 0x1b;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH_WAIT = 0x1c;
constexpr uint8_t GOLEM_ROCC_FUNC7_SFU_JOB = 0x1d;
constexpr uint8_t GOLEM_ROCC_FUNC7_REMOTE_STORE_WAIT = 0x1e;
constexpr uint8_t GOLEM_ROCC_FUNC7_TENSOR_MANAGER_JOB = 0x1f;
constexpr uint8_t GOLEM_ROCC_FUNC7_TENSOR_MANAGER_WAIT = 0x20;
constexpr uint8_t GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_JOB = 0x21;
constexpr uint8_t GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_WAIT = 0x22;
constexpr uint32_t GOLEM_ATTENTION_DESC_MAGIC = 0x41545431u;
constexpr uint16_t GOLEM_ATTENTION_DESC_VERSION = 2u;
constexpr uint32_t GOLEM_ATTENTION_FLAG_CAUSAL = 0x1u;
constexpr uint64_t ATTENTION_C1_WINDOW_BYTES = 26752;
constexpr uint64_t ATTENTION_D1_WINDOW_BYTES = 43136;
constexpr uint64_t ATTENTION_D3_WINDOW_BYTES = 46208;
struct GolemAttentionDescV2 {
    uint32_t magic;
    uint16_t version;
    uint16_t size_bytes;
    uint64_t job_id;
    uint64_t q_addr;
    uint64_t k_addr;
    uint64_t v_addr;
    uint64_t output_addr;
    uint64_t topology_gm_addr;
    uint32_t group_query_rows;
    uint32_t kv_length;
    uint32_t head_dim;
    uint32_t query_tile_rows;
    uint32_t kv_tile_rows;
    uint32_t worker_count;
    uint32_t flags;
    uint32_t group_query_row_begin;
    uint32_t kv_rows_per_memory_node;
    uint64_t kv_node_stride_bytes;
    uint32_t tensor_root_core;
    uint32_t tensor_manager_slot;
    uint32_t tensor_manager_count;
    uint32_t num_query_heads;
    uint32_t num_kv_heads;
    uint32_t kv_head_index;
};

static_assert(sizeof(GolemAttentionDescV2) == 128,
              "GolemAttentionDescV2 ABI must remain fixed");

template <typename T>
class RoCCAnalog : public SST::Vanadis::VanadisRoCCInterface {

public:
    SST_ELI_REGISTER_SUBCOMPONENT_DERIVED_API(RoCCAnalog<T>, SST::Vanadis::VanadisRoCCInterface)
  
    RoCCAnalog(ComponentId_t id, Params &params)
        : VanadisRoCCInterface(id, params),
          max_instructions(params.find<size_t>("max_instructions", 8)) {

        stat_cycles_mvm_set = registerStatistic<uint64_t>("cycles_mvm_set");
        stat_cycles_mvm_l   = registerStatistic<uint64_t>("cycles_mvm_l");
        stat_cycles_mvm     = registerStatistic<uint64_t>("cycles_mvm");
        stat_cycles_mvm_s   = registerStatistic<uint64_t>("cycles_mvm_s");
        stat_cycles_mvm_mv  = registerStatistic<uint64_t>("cycles_mvm_mv");
        stat_cycles_mvm_ovec2gm = registerStatistic<uint64_t>("cycles_mvm_ovec2gm");
        stat_cycles_mvm_gm2ivec = registerStatistic<uint64_t>("cycles_mvm_gm2ivec");
        stat_cycles_mvm_gm2imat = registerStatistic<uint64_t>("cycles_mvm_gm2imat");
        stat_cycles_remote_st = registerStatistic<uint64_t>("cycles_remote_st");
        stat_cycles_remote_ld = registerStatistic<uint64_t>("cycles_remote_ld");
        statTensorManagerJobsIssued_ = registerStatistic<uint64_t>("tensor_manager_jobs_issued");
        statTensorManagerWorkersMapped_ = registerStatistic<uint64_t>("tensor_manager_workers_mapped");
        statTensorManagerRowsDispatched_ = registerStatistic<uint64_t>("tensor_manager_rows_dispatched");
        statTensorManagerRowsCompleted_ = registerStatistic<uint64_t>("tensor_manager_rows_completed");
        statTensorManagerJobsCompleted_ = registerStatistic<uint64_t>("tensor_manager_jobs_completed");
        statTensorManagerDescriptorAcceptTick_ = registerStatistic<uint64_t>("tensor_manager_descriptor_accept_tick");
        statTensorManagerBandDispatchTick_ = registerStatistic<uint64_t>("tensor_manager_band_dispatch_tick");
        statTensorManagerCompletionReceivedTick_ = registerStatistic<uint64_t>("tensor_manager_completion_received_tick");
        statTensorManagerCompleteTick_ = registerStatistic<uint64_t>("tensor_manager_complete_tick");
        statTensorManagerWaitObservedTick_ = registerStatistic<uint64_t>("tensor_manager_wait_observed_tick");
        statAttentionManagerJobsIssued_ = registerStatistic<uint64_t>("attention_manager_jobs_issued");
        statAttentionManagerJobsCompleted_ = registerStatistic<uint64_t>("attention_manager_jobs_completed");
        statAttentionManagerBandsCompleted_ = registerStatistic<uint64_t>("attention_manager_bands_completed");
        statAttentionManagerBandCompletionsReceived_ = registerStatistic<uint64_t>("attention_manager_band_completions_received");
        statAttentionTensorJobsCompleted_ = registerStatistic<uint64_t>("attention_tensor_jobs_completed");
        statAttentionManagerDescriptorAcceptTick_ = registerStatistic<uint64_t>("attention_manager_descriptor_accept_tick");
        statAttentionManagerDispatchTick_ = registerStatistic<uint64_t>("attention_manager_dispatch_tick");
        statAttentionManagerLocalCompleteTick_ = registerStatistic<uint64_t>("attention_manager_local_complete_tick");
        statAttentionManagerBandCompletionReceivedTick_ = registerStatistic<uint64_t>("attention_manager_band_completion_received_tick");
        statAttentionTensorCompleteTick_ = registerStatistic<uint64_t>("attention_tensor_complete_tick");
        statAttentionManagerWaitObservedTick_ = registerStatistic<uint64_t>("attention_manager_wait_observed_tick");
        statAttentionWorkerDispatchAcceptTick_ = registerStatistic<uint64_t>("attention_worker_dispatch_accept_tick");
        statAttentionWorkerQkTileCompleteTick_ = registerStatistic<uint64_t>("attention_worker_qk_tile_complete_tick");
        statAttentionWorkerSoftmaxTileCompleteTick_ = registerStatistic<uint64_t>("attention_worker_softmax_tile_complete_tick");
        statAttentionWorkerPvTileCompleteTick_ = registerStatistic<uint64_t>("attention_worker_pv_tile_complete_tick");
        statAttentionWorkerOutputDmaAckTick_ = registerStatistic<uint64_t>("attention_worker_output_dma_ack_tick");
        statAttentionWorkerInterTileTotalTicks_ = registerStatistic<uint64_t>("attention_worker_intertile_total_ticks");
        statAttentionWorkerInterTileOutputDmaTicks_ = registerStatistic<uint64_t>("attention_worker_intertile_output_dma_ticks");
        statAttentionWorkerInterTileQueryLoadTicks_ = registerStatistic<uint64_t>("attention_worker_intertile_query_load_ticks");
        statAttentionWorkerInterTileKvLoadTicks_ = registerStatistic<uint64_t>("attention_worker_intertile_kv_load_ticks");
        statAttentionWorkerInterTileQLocalReadTicks_ = registerStatistic<uint64_t>("attention_worker_intertile_q_local_read_ticks");
        statAttentionWorkerInterTileQkMatrixProgramTicks_ = registerStatistic<uint64_t>("attention_worker_intertile_qk_matrix_program_ticks");
        statAttentionWorkerInterTileQkInputProgramTicks_ = registerStatistic<uint64_t>("attention_worker_intertile_qk_input_program_ticks");
        statAttentionWorkerInterTileQkComputeReadoutTicks_ = registerStatistic<uint64_t>("attention_worker_intertile_qk_compute_readout_ticks");
        statAttentionWorkerTileTotalTicks_ = registerStatistic<uint64_t>("attention_worker_tile_total_ticks");
        statAttentionWorkerTileKvLoadTicks_ = registerStatistic<uint64_t>("attention_worker_tile_kv_load_ticks");
        statAttentionWorkerTileQLocalReadTicks_ = registerStatistic<uint64_t>("attention_worker_tile_q_local_read_ticks");
        statAttentionWorkerTileQkMatrixProgramTicks_ = registerStatistic<uint64_t>("attention_worker_tile_qk_matrix_program_ticks");
        statAttentionWorkerTileQkInputProgramTicks_ = registerStatistic<uint64_t>("attention_worker_tile_qk_input_program_ticks");
        statAttentionWorkerTileQkComputeReadoutTicks_ = registerStatistic<uint64_t>("attention_worker_tile_qk_compute_readout_ticks");
        statAttentionWorkerTileSoftmaxTicks_ = registerStatistic<uint64_t>("attention_worker_tile_softmax_ticks");
        statAttentionWorkerTilePvMatrixProgramTicks_ = registerStatistic<uint64_t>("attention_worker_tile_pv_matrix_program_ticks");
        statAttentionWorkerTilePvInputProgramTicks_ = registerStatistic<uint64_t>("attention_worker_tile_pv_input_program_ticks");
        statAttentionWorkerTilePvRestoreOutputTicks_ = registerStatistic<uint64_t>("attention_worker_tile_pv_restore_output_ticks");
        statAttentionWorkerTilePvComputeTicks_ = registerStatistic<uint64_t>("attention_worker_tile_pv_compute_ticks");
        statAttentionWorkerTilePvOutputReadwriteTicks_ = registerStatistic<uint64_t>("attention_worker_tile_pv_output_readwrite_ticks");
        statAttentionKvPrefetchTiles_ = registerStatistic<uint64_t>("attention_kv_prefetch_tiles");
        statAttentionKvPrefetchHits_ = registerStatistic<uint64_t>("attention_kv_prefetch_hits");
        statAttentionKvPrefetchWaits_ = registerStatistic<uint64_t>("attention_kv_prefetch_waits");
        statAttentionKvPrefetchDmaTicks_ = registerStatistic<uint64_t>("attention_kv_prefetch_dma_ticks");
        statAttentionKvPrefetchReadyLeadTicks_ = registerStatistic<uint64_t>("attention_kv_prefetch_ready_lead_ticks");
        statAttentionKvPrefetchWaitTicks_ = registerStatistic<uint64_t>("attention_kv_prefetch_wait_ticks");
        statAttentionKvKReleaseTicks_ = registerStatistic<uint64_t>("attention_kv_k_release_ticks");
        statAttentionKvVReleaseTicks_ = registerStatistic<uint64_t>("attention_kv_v_release_ticks");
        statAttentionKvNextReadyAtReleaseTiles_ = registerStatistic<uint64_t>("attention_kv_next_ready_at_release_tiles");
        statAttentionKvSecondLookaheadCandidates_ = registerStatistic<uint64_t>("attention_kv_second_lookahead_candidates");
        statAttentionKvSecondLookaheadPrefetches_ = registerStatistic<uint64_t>("attention_kv_second_lookahead_prefetches");
        statAttentionKvSecondLookaheadLeadTicks_ = registerStatistic<uint64_t>("attention_kv_second_lookahead_lead_ticks");
        statAttentionKvCrossQueryPrefetches_ = registerStatistic<uint64_t>("attention_kv_cross_query_prefetches");
        statAttentionKvCrossQueryHits_ = registerStatistic<uint64_t>("attention_kv_cross_query_hits");
        statAttentionKvCrossQueryWaits_ = registerStatistic<uint64_t>("attention_kv_cross_query_waits");
        statAttentionKvCrossQueryWaitTicks_ = registerStatistic<uint64_t>("attention_kv_cross_query_wait_ticks");
        statAttentionKvPairReuseTiles_ = registerStatistic<uint64_t>("attention_kv_pair_reuse_tiles");
        statAttentionKvPairReuseBytes_ = registerStatistic<uint64_t>("attention_kv_pair_reuse_bytes");
        statAttentionPvInputPipelineRows_ = registerStatistic<uint64_t>("attention_pv_input_pipeline_rows");
        statAttentionClusterPvWavefrontRows_ =
            registerStatistic<uint64_t>("attention_cluster_pv_wavefront_rows");
        statAttentionClusterQkMatrixLookaheadLaunches_ = registerStatistic<uint64_t>(
            "attention_cluster_qk_matrix_lookahead_launches");
        statAttentionClusterQkMatrixLookaheadHits_ = registerStatistic<uint64_t>(
            "attention_cluster_qk_matrix_lookahead_hits");
        statAttentionClusterPvMatrixLookaheadLaunches_ = registerStatistic<uint64_t>(
            "attention_cluster_pv_matrix_lookahead_launches");
        statAttentionClusterPvMatrixLookaheadHits_ = registerStatistic<uint64_t>(
            "attention_cluster_pv_matrix_lookahead_hits");
        statAttentionPvRestorePipelineRows_ = registerStatistic<uint64_t>("attention_pv_restore_pipeline_rows");
        statAttentionPvOutputPipelineRows_ = registerStatistic<uint64_t>("attention_pv_output_pipeline_rows");
        statAttentionPvEarlyComputeArrays_ = registerStatistic<uint64_t>("attention_pv_early_compute_arrays");
        statAttentionPvMatrixOverlapTiles_ = registerStatistic<uint64_t>("attention_pv_matrix_overlap_tiles");
        statAttentionPvMatrixOverlapHits_ = registerStatistic<uint64_t>("attention_pv_matrix_overlap_hits");
        statAttentionPvMatrixOverlapWaits_ = registerStatistic<uint64_t>("attention_pv_matrix_overlap_waits");
        statAttentionQkMatrixBroadcasts_ = registerStatistic<uint64_t>("attention_qk_matrix_broadcasts");
        statAttentionPvMatrixBroadcasts_ = registerStatistic<uint64_t>("attention_pv_matrix_broadcasts");
        statAttentionQkArrayOps_ = registerStatistic<uint64_t>("attention_qk_array_ops");
        statAttentionQkEarlyComputeArrays_ = registerStatistic<uint64_t>("attention_qk_early_compute_arrays");
        statAttentionQkInputPipelineRowsFetched_ = registerStatistic<uint64_t>("attention_qk_input_pipeline_rows_fetched");
        statAttentionQkInputPipelineRowsProgrammed_ = registerStatistic<uint64_t>("attention_qk_input_pipeline_rows_programmed");
        statAttentionQkInputPipelineOverlapTicks_ = registerStatistic<uint64_t>("attention_qk_input_pipeline_overlap_ticks");
        statAttentionQkInputPipelineSlotFullStalls_ = registerStatistic<uint64_t>("attention_qk_input_pipeline_slot_full_stalls");
        statAttentionQkInputPipelineMaxDepth_ = registerStatistic<uint64_t>("attention_qk_input_pipeline_max_depth");
        statAttentionQkInputPipelineTagMismatches_ = registerStatistic<uint64_t>("attention_qk_input_pipeline_tag_mismatches");
        statAttentionQkReadoutAheadDepth_ = registerStatistic<uint64_t>("attention_qk_readout_ahead_depth");
        statAttentionCrossTileOperandCandidates_ = registerStatistic<uint64_t>("attention_cross_tile_operand_candidates");
        statAttentionCrossTileOperandLaunches_ = registerStatistic<uint64_t>("attention_cross_tile_operand_launches");
        statAttentionCrossTileOperandMatrixPrograms_ = registerStatistic<uint64_t>("attention_cross_tile_operand_matrix_programs");
        statAttentionCrossTileOperandInputRows_ = registerStatistic<uint64_t>("attention_cross_tile_operand_input_rows");
        statAttentionCrossTileOperandReadyHits_ = registerStatistic<uint64_t>("attention_cross_tile_operand_ready_hits");
        statAttentionCrossTileOperandWaits_ = registerStatistic<uint64_t>("attention_cross_tile_operand_waits");
        statAttentionCrossTileOperandWaitTicks_ = registerStatistic<uint64_t>("attention_cross_tile_operand_wait_ticks");
        statAttentionCrossTileOperandPromotions_ = registerStatistic<uint64_t>("attention_cross_tile_operand_promotions");
        statAttentionCrossTileOperandTagMismatches_ = registerStatistic<uint64_t>("attention_cross_tile_operand_tag_mismatches");
        statAttentionPvArrayOps_ = registerStatistic<uint64_t>("attention_pv_array_ops");
        statAttentionGenericGemmQkOps_ = registerStatistic<uint64_t>("attention_generic_gemm_qk_ops");
        statAttentionGenericGemmPvOps_ = registerStatistic<uint64_t>("attention_generic_gemm_pv_ops");
        statAttentionPvActiveKLaunches_ = registerStatistic<uint64_t>("attention_pv_active_k_launches");
        statAttentionPvActiveKColumns_ = registerStatistic<uint64_t>("attention_pv_active_k_columns");
        statAttentionPvActiveKMatrixElements_ = registerStatistic<uint64_t>("attention_pv_active_k_matrix_elements");
        statAttentionPvVTileBufferHits_ = registerStatistic<uint64_t>("attention_pv_v_tile_buffer_hits");
        statAttentionPvVTileBufferMisses_ = registerStatistic<uint64_t>("attention_pv_v_tile_buffer_misses");
        statAttentionPvVTileBufferBytesRead_ = registerStatistic<uint64_t>("attention_pv_v_tile_buffer_bytes_read");
        statAttentionPvVTileBufferBytesReused_ = registerStatistic<uint64_t>("attention_pv_v_tile_buffer_bytes_reused");
        statAttentionPvVTileBufferWaitTicks_ = registerStatistic<uint64_t>("attention_pv_v_tile_buffer_wait_ticks");
        statAttentionPvVTileBufferCapacityRejections_ = registerStatistic<uint64_t>("attention_pv_v_tile_buffer_capacity_rejections");
        statAttentionPvVTileBufferGroupHits_ = registerStatistic<uint64_t>("attention_pv_v_tile_buffer_group_hits");
        statAttentionPvInputResidencyHits_ = registerStatistic<uint64_t>("attention_pv_input_residency_hits");
        statAttentionPvInputResidencyRowsReused_ = registerStatistic<uint64_t>("attention_pv_input_residency_rows_reused");
        statAttentionPvInputResidencyInvalidations_ = registerStatistic<uint64_t>("attention_pv_input_residency_invalidations");
        statAttentionOAccumulatorStores_ = registerStatistic<uint64_t>("attention_o_accumulator_stores");
        statAttentionOAccumulatorRestores_ = registerStatistic<uint64_t>("attention_o_accumulator_restores");
        statAttentionOAccumulatorBytes_ = registerStatistic<uint64_t>("attention_o_accumulator_bytes");
        statAttentionSpHbmBytes_ = registerStatistic<uint64_t>("attention_sp_hbm_bytes");
        statAttentionClusterConfigFingerprint_ =
            registerStatistic<uint64_t>("attention_cluster_config_fingerprint");
        statAttentionClusterWorkerJobs_ =
            registerStatistic<uint64_t>("attention_cluster_worker_jobs");
        statAttentionClusterContextsIssued_ =
            registerStatistic<uint64_t>("attention_cluster_contexts_issued");
        statAttentionClusterContextsCompleted_ =
            registerStatistic<uint64_t>("attention_cluster_contexts_completed");
        statAttentionClusterContextsCancelled_ =
            registerStatistic<uint64_t>("attention_cluster_contexts_cancelled");
        statAttentionClusterBankRefsCancelled_ =
            registerStatistic<uint64_t>("attention_cluster_bank_refs_cancelled");
        statAttentionClusterMemoryRequestsCancelled_ =
            registerStatistic<uint64_t>("attention_cluster_memory_requests_cancelled");
        statAttentionClusterContextHighWater_ =
            registerStatistic<uint64_t>("attention_cluster_context_high_water");
        statAttentionClusterStaleCallbacks_ =
            registerStatistic<uint64_t>("attention_cluster_stale_callbacks");
        statAttentionClusterIllegalTransitions_ =
            registerStatistic<uint64_t>("attention_cluster_illegal_transitions");
        statAttentionClusterScoreSlotReservations_ =
            registerStatistic<uint64_t>("attention_cluster_score_slot_reservations");
        statAttentionClusterScoreSlotReleases_ =
            registerStatistic<uint64_t>("attention_cluster_score_slot_releases");
        statAttentionClusterScoreSlotFullStalls_ =
            registerStatistic<uint64_t>("attention_cluster_score_slot_full_stalls");
        statAttentionClusterScoreSlotHighWater_ =
            registerStatistic<uint64_t>("attention_cluster_score_slot_high_water");
        statAttentionClusterOContextReservations_ =
            registerStatistic<uint64_t>("attention_cluster_o_context_reservations");
        statAttentionClusterOContextReleases_ =
            registerStatistic<uint64_t>("attention_cluster_o_context_releases");
        statAttentionClusterOContextCancelled_ =
            registerStatistic<uint64_t>("attention_cluster_o_context_cancelled");
        statAttentionClusterOContextHighWater_ =
            registerStatistic<uint64_t>("attention_cluster_o_context_high_water");
        statAttentionClusterOScaleSegments_ =
            registerStatistic<uint64_t>("attention_cluster_o_scale_segments");
        statAttentionClusterOAccumulateSegments_ =
            registerStatistic<uint64_t>("attention_cluster_o_accumulate_segments");
        statAttentionClusterOFusedRows_ =
            registerStatistic<uint64_t>("attention_cluster_o_fused_rows");
        statAttentionClusterOFusedBytes_ =
            registerStatistic<uint64_t>("attention_cluster_o_fused_bytes");
        statAttentionClusterODrainRequests_ =
            registerStatistic<uint64_t>("attention_cluster_o_drain_requests");
        statAttentionClusterODrainBytes_ =
            registerStatistic<uint64_t>("attention_cluster_o_drain_bytes");
        statAttentionClusterOReadWaitCycles_ =
            registerStatistic<uint64_t>("attention_cluster_o_read_wait_cycles");
        statAttentionClusterOWriteWaitCycles_ =
            registerStatistic<uint64_t>("attention_cluster_o_write_wait_cycles");
        statAttentionClusterOAluWaitCycles_ =
            registerStatistic<uint64_t>("attention_cluster_o_alu_wait_cycles");
        statAttentionClusterOBankConflictCycles_ =
            registerStatistic<uint64_t>("attention_cluster_o_bank_conflict_cycles");
        statAttentionClusterODrainWaitCycles_ =
            registerStatistic<uint64_t>("attention_cluster_o_drain_wait_cycles");
        statAttentionClusterOReadBusyUnionCycles_ = registerStatistic<uint64_t>(
            "attention_cluster_o_read_busy_union_cycles");
        statAttentionClusterOReadBusySpanCycles_ = registerStatistic<uint64_t>(
            "attention_cluster_o_read_busy_span_cycles");
        statAttentionClusterOReadIdleGapCycles_ = registerStatistic<uint64_t>(
            "attention_cluster_o_read_idle_gap_cycles");
        statAttentionClusterOReadMaxConcurrency_ = registerStatistic<uint64_t>(
            "attention_cluster_o_read_max_concurrency");
        statAttentionClusterOWriteBusyUnionCycles_ = registerStatistic<uint64_t>(
            "attention_cluster_o_write_busy_union_cycles");
        statAttentionClusterOWriteBusySpanCycles_ = registerStatistic<uint64_t>(
            "attention_cluster_o_write_busy_span_cycles");
        statAttentionClusterOWriteIdleGapCycles_ = registerStatistic<uint64_t>(
            "attention_cluster_o_write_idle_gap_cycles");
        statAttentionClusterOWriteMaxConcurrency_ = registerStatistic<uint64_t>(
            "attention_cluster_o_write_max_concurrency");
        statAttentionClusterOAluBusyUnionCycles_ = registerStatistic<uint64_t>(
            "attention_cluster_o_alu_busy_union_cycles");
        statAttentionClusterOAluBusySpanCycles_ = registerStatistic<uint64_t>(
            "attention_cluster_o_alu_busy_span_cycles");
        statAttentionClusterOAluIdleGapCycles_ = registerStatistic<uint64_t>(
            "attention_cluster_o_alu_idle_gap_cycles");
        statAttentionClusterOAluMaxConcurrency_ = registerStatistic<uint64_t>(
            "attention_cluster_o_alu_max_concurrency");
        statAttentionClusterQkArrayBusyUnionTicks_ =
            registerStatistic<uint64_t>("attention_cluster_qk_array_busy_union_ticks");
        statAttentionClusterQkArrayBusySpanTicks_ =
            registerStatistic<uint64_t>("attention_cluster_qk_array_busy_span_ticks");
        statAttentionClusterQkArrayIdleGapTicks_ =
            registerStatistic<uint64_t>("attention_cluster_qk_array_idle_gap_ticks");
        statAttentionClusterQkArrayMaxConcurrency_ =
            registerStatistic<uint64_t>("attention_cluster_qk_array_max_concurrency");
        statAttentionClusterPvArrayBusyUnionTicks_ =
            registerStatistic<uint64_t>("attention_cluster_pv_array_busy_union_ticks");
        statAttentionClusterPvArrayBusySpanTicks_ =
            registerStatistic<uint64_t>("attention_cluster_pv_array_busy_span_ticks");
        statAttentionClusterPvArrayIdleGapTicks_ =
            registerStatistic<uint64_t>("attention_cluster_pv_array_idle_gap_ticks");
        statAttentionClusterPvArrayMaxConcurrency_ =
            registerStatistic<uint64_t>("attention_cluster_pv_array_max_concurrency");
        statAttentionClusterKTileBroadcasts_ =
            registerStatistic<uint64_t>("attention_cluster_qk_k_tile_broadcasts");
        statAttentionClusterKTileBytes_ =
            registerStatistic<uint64_t>("attention_cluster_qk_k_tile_bytes");
        statAttentionClusterQPairMulticasts_ =
            registerStatistic<uint64_t>("attention_cluster_qk_q_pair_multicasts");
        statAttentionClusterQPairBytes_ =
            registerStatistic<uint64_t>("attention_cluster_qk_q_pair_bytes");
        statAttentionClusterScoreBeats_ =
            registerStatistic<uint64_t>("attention_cluster_qk_score_beats");
        statAttentionClusterScoreBytes_ =
            registerStatistic<uint64_t>("attention_cluster_qk_score_bytes");
        statAttentionClusterAheadContextsLaunched_ =
            registerStatistic<uint64_t>("attention_cluster_ahead_contexts_launched");
        statAttentionClusterAheadContextsCompleted_ =
            registerStatistic<uint64_t>("attention_cluster_ahead_contexts_completed");
        statAttentionClusterAheadContextsPromoted_ =
            registerStatistic<uint64_t>("attention_cluster_ahead_contexts_promoted");
        statAttentionClusterPromotionWaits_ =
            registerStatistic<uint64_t>("attention_cluster_promotion_waits");
        statAttentionClusterInitialEnqueueRetries_ =
            registerStatistic<uint64_t>("attention_cluster_initial_enqueue_retries");
        statAttentionClusterQkTileStarts_ =
            registerStatistic<uint64_t>("attention_cluster_qk_tile_starts");
        statAttentionClusterQkTileIiCycles_ =
            registerStatistic<uint64_t>("attention_cluster_qk_tile_ii_cycles");
        statAttentionClusterQkBoundaryIiCycles_ =
            registerStatistic<uint64_t>("attention_cluster_qk_boundary_ii_cycles");
        statAttentionClusterQkSteadyIiOverTarget_ =
            registerStatistic<uint64_t>("attention_cluster_qk_steady_ii_over_target");
        statAttentionClusterQkSfuOverlapCycles_ =
            registerStatistic<uint64_t>("attention_cluster_qk_sfu_overlap_cycles");
        statAttentionClusterSfuPvOverlapCycles_ =
            registerStatistic<uint64_t>("attention_cluster_sfu_pv_overlap_cycles");
        statAttentionClusterQkPvOverlapCycles_ =
            registerStatistic<uint64_t>("attention_cluster_qk_pv_overlap_cycles");
        statAttentionClusterThreeStageOverlapCycles_ =
            registerStatistic<uint64_t>("attention_cluster_three_stage_overlap_cycles");
        statAttentionSequentialQkWaves_ =
            registerStatistic<uint64_t>("attention_sequential_qk_waves");
        statAttentionSequentialPvWaves_ =
            registerStatistic<uint64_t>("attention_sequential_pv_waves");
        statAttentionSequentialQkActiveArrays_ = registerStatistic<uint64_t>(
            "attention_sequential_qk_active_arrays");
        statAttentionSequentialPvActiveArrays_ = registerStatistic<uint64_t>(
            "attention_sequential_pv_active_arrays");

        latency_mvm_ovec2gm = params.find<uint64_t>("latency_mvm_ovec2gm", 10);
        latency_mvm_gm2ivec = params.find<uint64_t>("latency_mvm_gm2ivec", 15);
        latency_mvm_gm2imat = params.find<uint64_t>("latency_mvm_gm2imat", 20);
        latency_remote_st = params.find<uint64_t>("latency_remote_st", 20);
        latency_remote_ld = params.find<uint64_t>("latency_remote_ld", 25);
        enable_async_array_load = params.find<int>("enable_async_array_load", 1) != 0;
        progress_heartbeat = params.find<int>("progress_heartbeat", 0) != 0;
        progress_interval_cycles = params.find<uint64_t>("progress_interval_cycles", 50000);
        progress_total_mvm_ops = params.find<uint64_t>("progress_total_mvm_ops", 0);
        if (progress_interval_cycles == 0) {
            progress_interval_cycles = 50000;
        }
        if (progress_total_mvm_ops == 0) {
            progress_heartbeat = false;
        }
        progress_next_cycle = progress_interval_cycles;

        coreID = params.find<uint64_t>("core_id", 0);
        StartTickCycle = 0;
        LastTickCycle = 0;
  
        try {
            UnitAlgebra clock = params.find<UnitAlgebra>("clock", "1GHz");
  
            if (!(clock.hasUnits("Hz") || clock.hasUnits("s")) || 
                clock.getRoundedValue() <= 0) {
                output->fatal(CALL_INFO, -1,
                    "%s, Error - Invalid param: clock.\n"
                    "Must have units of Hz or s and be > 0.\n"
                    "SI prefixes ok. You specified '%s'\n",
                    getName().c_str(), clock.toString().c_str());
            }
        } catch (const UnitAlgebra::UnitAlgebraException& exc) {
            output->fatal(CALL_INFO, -1,
                "%s, Invalid param: Exception while parsing 'clock'.\n"
                "'%s'\n",
                getName().c_str(), exc.what());
        }
  
        mmioStartAddr = params.find<uint64_t>("mmioAddr", 0);
        arrayInputSize = params.find<uint64_t>("arrayInputSize", 2);
        arrayOutputSize = params.find<uint64_t>("arrayOutputSize", 2);
  
        numArrays = params.find<uint64_t>("numArrays", 1);
        inputOperandSize = params.find<uint64_t>("inputOperandSize", 4);
        outputOperandSize = params.find<uint64_t>("outputOperandSize", 4);
        attentionWindowOffset_ = params.find<uint64_t>("attention_window_offset", 0xC0000);
        attentionWindowBytes_ = params.find<uint64_t>("attention_window_bytes", 0x10000);
        attentionKvTileRotation_ = params.find<bool>("attention_kv_tile_rotation", false);
        attentionKvDoubleBuffer_ = params.find<bool>("attention_kv_double_buffer", false);
        attentionKvBufferCount_ = params.find<uint32_t>(
            "attention_kv_buffer_count", attentionKvDoubleBuffer_ ? 2u : 1u);
        attentionKvDistributionEnable_ = params.find<bool>(
            "attention_kv_distribution_enable", false);
        if (attentionKvBufferCount_ == 0 || attentionKvBufferCount_ > 3 ||
            (!attentionKvDoubleBuffer_ && attentionKvBufferCount_ != 1)) {
            output->fatal(CALL_INFO, -1,
                "attention_kv_buffer_count must be 1 without prefetch or 2..3 with prefetch\n");
        }
        attentionKvSecondLookahead_ = params.find<bool>("attention_kv_second_lookahead", true);
        attentionKvCrossQueryPrefetch_ = params.find<bool>("attention_kv_cross_query_prefetch", false);
        attentionKvPairReuse_ = params.find<bool>("attention_kv_pair_reuse", false);
        attentionKvQueryGroupSize_ =
            params.find<uint32_t>("attention_kv_query_group_size", 2);
        attentionPvVTileReuse_ = params.find<bool>("attention_pv_v_tile_reuse", false);
        attentionPvVTileGroupRetention_ =
            params.find<bool>("attention_pv_v_tile_group_retention", false);
        attentionPvVTileBufferBytes_ = params.find<uint64_t>("attention_pv_v_tile_buffer_bytes", 16384);
        attentionPvVTileBufferHitTicks_ = params.find<uint64_t>("attention_pv_v_tile_buffer_hit_ticks", 1);
        attentionPvVTileBufferBytesPerCycle_ = std::max<uint64_t>(
            params.find<uint64_t>("attention_pv_v_tile_buffer_bytes_per_cycle", 64), 1);
        attentionPvVTileBufferOffset_ = params.find<uint64_t>("attention_pv_v_tile_buffer_offset", 0);
        attentionPvInputPipeline_ = params.find<bool>("attention_pv_input_pipeline", false);
        attentionClusterPvRowWavefront_ = params.find<bool>(
            "attention_cluster_pv_row_wavefront", false);
        attentionClusterQkMatrixLookahead_ = params.find<bool>(
            "attention_cluster_qk_matrix_lookahead", false);
        attentionClusterPvMatrixLookahead_ = params.find<bool>(
            "attention_cluster_pv_matrix_lookahead", false);
        attentionPvCompactInput_ = params.find<bool>("attention_pv_compact_input", false);
        attentionPvInputResidency_ =
            params.find<bool>("attention_pv_input_residency", false);
        attentionOAccumulatorCBuffer_ =
            params.find<bool>("attention_o_accumulator_cbuffer", false);
        attentionPvRestorePipeline_ = params.find<bool>("attention_pv_restore_pipeline", false);
        attentionPvOutputPipeline_ = params.find<bool>("attention_pv_output_pipeline", false);
        attentionPvORowFusion_ =
            params.find<bool>("attention_pv_o_row_fusion", false);
        attentionPvEarlyCompute_ = params.find<bool>("attention_pv_early_compute", false);
        attentionPvMatrixSoftmaxOverlap_ =
            params.find<bool>("attention_pv_matrix_softmax_overlap", false);
        attentionPvActiveK_ = params.find<bool>("attention_pv_active_k", false);
        attentionQkDataflowTranspose_ = params.find<bool>("attention_qk_dataflow_transpose", false);
        attentionQkEarlyCompute_ = params.find<bool>("attention_qk_early_compute", false);
        attentionQkInputPipeline_ =
            params.find<bool>("attention_qk_input_pipeline", false);
        attentionQkReadoutOverlap_ = params.find<bool>("attention_qk_readout_overlap", false);
        attentionQkReadoutWindow_ = std::max<uint32_t>(
            1u, std::min<uint32_t>(16u,
                params.find<uint32_t>("attention_qk_readout_window", 2)));
        const bool legacyQkPanelRowBurst =
            params.find<bool>("attention_qk_panel_row_burst", false);
        attentionQkScoreRowBurst_ = params.find<bool>(
            "attention_qk_score_row_burst", legacyQkPanelRowBurst);
        attentionQkMatrixBroadcast_ = attentionQkDataflowTranspose_ ||
            params.find<bool>("attention_qk_matrix_broadcast", false);
        attentionCrossTileOperandPipeline_ =
            params.find<bool>("attention_cross_tile_operand_pipeline", false);
        attentionOperandContextBanks_ =
            params.find<uint32_t>("attention_operand_context_banks", 1);
        attentionPvMatrixBroadcast_ = params.find<bool>("attention_pv_matrix_broadcast", false);
        attentionGenericGemmEnable_ =
            params.find<bool>("attention_generic_gemm_enable", false);
        attentionReuseWindowQkBridge_ =
            params.find<bool>("attention_reuse_window_qk_bridge", false);
        attentionWorkerClusterBridge_ =
            params.find<bool>("attention_worker_cluster_bridge", false);
        attentionWorkerClusterQkWorkersPerManager_ = std::max<uint32_t>(
            1u, std::min<uint32_t>(2u, params.find<uint32_t>(
                "attention_worker_cluster_qk_workers_per_manager", 1)));
        attentionWorkerClusterRowPriority_ = params.find<bool>(
            "attention_worker_cluster_row_priority", false);
        attentionWorkerClusterVBroadcast_ = params.find<bool>(
            "attention_worker_cluster_v_broadcast", false);
        attentionWorkerClusterDynamicPv_ = params.find<bool>(
            "attention_worker_cluster_dynamic_pv", false);
        attentionMilestoneTrace_ =
            params.find<bool>("attention_milestone_trace", false);
        attentionTileTrace_ = params.find<bool>("attention_tile_trace", false);
        attentionClusterEnable_ =
            params.find<bool>("attention_cluster_enable", false);
        attentionSequential64Enable_ = params.find<bool>(
            "attention_sequential_64_enable", false);
        attentionClusterConfig_.qkArrays = params.find<uint32_t>(
            "attention_cluster_qk_arrays", 16);
        attentionClusterConfig_.kvTileRows = params.find<uint32_t>(
            "attention_key_block_rows", 32);

        if (attentionSequential64Enable_) {
            if (attentionClusterEnable_ || numArrays != 64 ||
                arrayInputSize != 64 || arrayOutputSize != 64) {
                output->fatal(CALL_INFO, -1,
                    "attention_sequential_64_enable requires non-cluster 64x64 arrays\n");
            }
            if (attentionQkDataflowTranspose_ || attentionQkEarlyCompute_ ||
                attentionQkInputPipeline_ || attentionQkReadoutOverlap_ ||
                attentionQkScoreRowBurst_ || attentionPvInputPipeline_ ||
                attentionPvEarlyCompute_ || attentionPvMatrixSoftmaxOverlap_ ||
                attentionOAccumulatorCBuffer_) {
                output->fatal(CALL_INFO, -1,
                    "attention_sequential_64_enable requires phase-serial QK/PV options\n");
            }
        }

        vectorStrideBytes = params.find<uint64_t>(
            "vectorStrideBytes", static_cast<uint64_t>(arrayInputSize) * inputOperandSize);

        remoteTransferLength = defaultRemoteLength();
  
        output->verbose(
            CALL_INFO, 1, 0,
            "%s: numArrays: %d, arrayInputSize: %d, arrayOutputSize: %d \n",
            getName().c_str(), numArrays, arrayInputSize, arrayOutputSize);
        //std_mem_handlers 是内存请求处理器，其作用是处理从内存系统
        //返回的响应。它解析内存的响应数据，并执行相应的后续操作
        std_mem_handlers = new StandardMemHandlers(this, output);
  
        busy = false;
        //memInterface 是内存接口，它代表了与内存系统的实际通信渠道,
        //负责发起内存读写请求，并处理这些请求的发送和响应
        memInterface = loadUserSubComponent<Interfaces::StandardMem>(
            "memory_interface",
            ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS,
            getTimeConverter("1ps"),
            new StandardMem::Handler2<RoCCAnalog<T>, &RoCCAnalog<T>::processIncomingDataCacheEvent>(this));

        if ( nullptr == memInterface ) {
            output->fatal(
                CALL_INFO, -1,
                "Error: unable to load memory interface subcomponent for RoCCAnalog.\n");
        }
        //加载 Golem 的计算阵列子组件，用于进行阵列计算
        array = loadUserSubComponent<Golem::ComputeArray>(
            "array", ComponentInfo::SHARE_NONE, getTimeConverter("1ps"),
            new SST::Event::Handler2<RoCCAnalog<T>, &RoCCAnalog<T>::handleArrayEvent>(this));

        if ( nullptr == array ) {
            output->fatal(
                CALL_INFO, -1,
                "Error: Unable to load array model subcomponent for RoCCAnalog.\n");
        }

        // 新增：加载 GlobalMemory 子组件（可选）。
        globalMem = loadUserSubComponent<SST::Golem::GlobalMemoryAPI>(
            "global_memory", ComponentInfo::SHARE_NONE);
        uint64_t globalMemStride = params.find<uint64_t>("globalMemStride", 0x4000);
        uint64_t globalMemBase = params.find<uint64_t>("globalMemBase", 0x0);
        if (nullptr == globalMem) {
            // 如果测试未提供，回退到本地的无网络实现，避免fatal。
            output->verbose(
                CALL_INFO, 1,0,
                "Warning: Unable to load Network globalmemory subcomponent for RoCCAnalog, turn to network free implementation.\n");
            Params gmFallbackParams;
            gmFallbackParams.insert("src_id", std::to_string(coreID));
            gmFallbackParams.insert("size", std::to_string(globalMemStride));
            globalMem = loadAnonymousSubComponent<SST::Golem::GlobalMemoryAPI>(
                "golem.GlobalMemoryLocal", "global_memory", 0,
                ComponentInfo::SHARE_NONE, gmFallbackParams);
        }
        // **配置 GlobalMemory 基地址**：根据核心 ID 计算基地址 
        // 获取当前核ID（默认为0）
        // 每个核的地址空间跨度，默认0x4000
        uint64_t baseAddr = globalMemBase + coreID * globalMemStride;        // 计算该核 GlobalMemory 的基地址
        if (globalMem) {
            globalMem->setBaseAddr(baseAddr);                                // 设置 GlobalMemory 子模块的基地址
        }
        output->verbose(CALL_INFO, 1, 0, 
                        "RoCCAnalog: 核心%" PRIu64 " 的 GlobalMemory 基地址配置为 0x%" PRIx64 "\n", 
                        coreID, baseAddr);

        sfu = nullptr;
        sfuEnable = params.find<int>("sfuEnable", 0) != 0;
        if (sfuEnable) {
            sfu = loadUserSubComponent<SST::Golem::SFUAPI>(
                "sfu", ComponentInfo::SHARE_NONE);
            if (nullptr == sfu) {
                output->fatal(CALL_INFO, -1,
                    "Error: sfuEnable=1 but required user subcomponent 'sfu' is missing for RoCCAnalog.\n");
            }
            sfu->bindGlobalMemory(globalMem);
            sfu->setCoreInfo(
                static_cast<uint32_t>(coreID),
                params.find<uint32_t>("active_worker_cores", 1));
        }
        globalMem->setControlMessageHandler(
            [this](const ControlTransportMessage& message) {
                handleControlTransportMessage(message);
            });

        groupCtrl = nullptr;
        if (params.find<int>("groupCtrlEnable", 0) != 0) {
            groupCtrl = loadUserSubComponent<SST::Golem::GroupCtrlAPI>(
                "group_ctrl", ComponentInfo::SHARE_NONE);

            if (nullptr == groupCtrl) {
                output->fatal(CALL_INFO, -1,
                    "Error: groupCtrlEnable=1 but required user subcomponent 'group_ctrl' is missing for RoCCAnalog.\n"
                    "Please wire RoCC slot 'group_ctrl' in the architecture script (setSubComponent).\n");
            }
        }
        if (attentionKvDistributionEnable_ &&
            (groupCtrl == nullptr || !groupCtrl->attentionKvDistributionEnabled())) {
            output->fatal(CALL_INFO, -1,
                "Attention K/V distribution requires an enabled GroupCtrl endpoint\n");
        }

        requestScheduler = nullptr;
        if (params.find<int>("requestSchedulerEnable", 0) != 0) {
            requestScheduler = loadUserSubComponent<SST::Golem::RequestSchedulerAPI>(
                "request_scheduler", ComponentInfo::SHARE_NONE);
            if (nullptr == requestScheduler) {
                output->fatal(CALL_INFO, -1,
                    "Error: requestSchedulerEnable=1 but required user subcomponent 'request_scheduler' is missing for RoCCAnalog.\n");
            }
        }

        if (groupCtrl) {
            groupCtrl->bindGlobalMemory(globalMem);
        }
        if (requestScheduler) {
            requestScheduler->bindGlobalMemory(globalMem);
        }

        workerCommandProcessor = nullptr;
        if (params.find<int>("workerCommandProcessorEnable", 0) != 0) {
            workerCommandProcessor = loadUserSubComponent<SST::Golem::WorkerCommandProcessorAPI>(
                "worker_command_processor", ComponentInfo::SHARE_NONE);
            if (nullptr == workerCommandProcessor) {
                output->fatal(CALL_INFO, -1,
                    "Error: workerCommandProcessorEnable=1 but required user subcomponent 'worker_command_processor' is missing for RoCCAnalog.\n");
            }
            workerCommandProcessor->bindResources(static_cast<uint32_t>(coreID), output, globalMem, array, requestScheduler);
        }
        if (attentionGenericGemmEnable_ && workerCommandProcessor == nullptr) {
            output->fatal(
                CALL_INFO, -1,
                "Error: attention_generic_gemm_enable=1 requires workerCommandProcessorEnable=1.\n");
        }
        if (attentionQkScoreRowBurst_ &&
            (!attentionGenericGemmEnable_ || attentionQkDataflowTranspose_ ||
             attentionQkReadoutOverlap_)) {
            output->fatal(
                CALL_INFO, -1,
                "Error: attention_qk_score_row_burst requires generic GEMM/WCP, "
                "non-transposed QK, and disabled QK readout overlap.\n");
        }
        if (attentionQkInputPipeline_ && attentionQkDataflowTranspose_) {
            output->fatal(
                CALL_INFO, -1,
                "Error: attention_qk_input_pipeline requires non-transposed QK.\n");
        }
        if (attentionCrossTileOperandPipeline_ &&
            (!attentionGenericGemmEnable_ || attentionQkDataflowTranspose_ ||
             !attentionQkMatrixBroadcast_ || !attentionKvPairReuse_ ||
             !attentionKvDoubleBuffer_ || attentionOperandContextBanks_ != 2)) {
            output->fatal(
                CALL_INFO, -1,
                "Error: attention_cross_tile_operand_pipeline requires generic "
                "GEMM/WCP, grouped K/V reuse, non-transposed broadcast QK, "
                "and exactly two operand banks.\n");
        }
        if ((attentionPvInputResidency_ || attentionOAccumulatorCBuffer_) &&
            !attentionGenericGemmEnable_) {
            output->fatal(
                CALL_INFO, -1,
                "Error: PV input residency and O accumulator C-buffer require "
                "generic GEMM/WCP.\n");
        }
        if (attentionPvVTileGroupRetention_ && !attentionPvVTileReuse_) {
            output->fatal(
                CALL_INFO, -1,
                "Error: PV V-tile group retention requires PV V-tile reuse.\n");
        }
        if (attentionClusterPvRowWavefront_ && !attentionClusterEnable_) {
            output->fatal(CALL_INFO, -1,
                "Error: attention_cluster_pv_row_wavefront requires the Attention cluster\n");
        }
        if (attentionClusterQkMatrixLookahead_ && !attentionClusterEnable_) {
            output->fatal(CALL_INFO, -1,
                "Error: attention_cluster_qk_matrix_lookahead requires the Attention cluster\n");
        }
        if (attentionClusterPvMatrixLookahead_ && !attentionClusterEnable_) {
            output->fatal(CALL_INFO, -1,
                "Error: attention_cluster_pv_matrix_lookahead requires the Attention cluster\n");
        }
        if (attentionClusterEnable_) {
            attentionClusterConfig_.arrays = numArrays;
            attentionClusterConfig_.pvArrays = attentionClusterConfig_.arrays >=
                    attentionClusterConfig_.qkArrays
                ? attentionClusterConfig_.arrays - attentionClusterConfig_.qkArrays
                : 0;
            attentionClusterConfig_.arrayInputs = arrayInputSize;
            attentionClusterConfig_.arrayOutputs = arrayOutputSize;
            attentionClusterConfig_.operandBanks = attentionOperandContextBanks_;
            attentionClusterConfig_.groupSize = attentionKvQueryGroupSize_;
            attentionClusterConfig_.elemBytes = inputOperandSize;
            const std::string error = attentionClusterConfig_.validate();
            if (!error.empty() || outputOperandSize != 4 ||
                !attentionGenericGemmEnable_ || workerCommandProcessor == nullptr ||
                sfu == nullptr || !attentionKvPairReuse_ || !attentionPvActiveK_) {
                output->fatal(
                    CALL_INFO, -1,
                    "Error: attention_cluster_enable configuration rejected: %s; "
                    "requires FP32 generic WCP/SFU, group-4 K/V reuse, and "
                    "active-K PV.\n",
                    error.empty() ? "component contract" : error.c_str());
            }
            statAttentionClusterConfigFingerprint_->addData(
                attentionClusterConfig_.fingerprint());
        }

    }
  
    virtual ~RoCCAnalog() {
        for (auto roccCmd_q_itr = roccCmd_q.begin(); roccCmd_q_itr != roccCmd_q.end();) {
            delete (*roccCmd_q_itr);
            roccCmd_q_itr = roccCmd_q.erase(roccCmd_q_itr);
        }

        for (auto& inflight : inflight_compute_cmds) {
            if (inflight.cmd != nullptr) {
                delete inflight.cmd;
                inflight.cmd = nullptr;
            }
        }

        while (!resp_q.empty()) {
            delete resp_q.front();
            resp_q.pop_front();
        }

        delete std_mem_handlers;
    }
    //RoCC指令队列满/是否忙/队列当前大小
    bool RoCCFull() override { return roccCmd_q.size() >= max_instructions; }
  
    bool isBusy() override { return busy; }

    bool isCPUWaitBlocked() override {
        if (workerCommandProcessor == nullptr || !workerCommandProcessor->isBusy() || roccCmd_q.empty()) {
            return false;
        }
        const auto* cmd = roccCmd_q.front();
        return cmd != nullptr && cmd->inst != nullptr &&
               cmd->inst->func7 == GOLEM_ROCC_FUNC7_WCP_WAIT;
    }
  
    size_t roccQueueSize() override { return roccCmd_q.size(); }

    //入队一条新的RoCC指令，同时统计数据加1
    void push(SST::Vanadis::RoCCCommand *rocc_me) override {
        stat_rocc_issued->addData(1);
        roccCmd_q.push_back(rocc_me);
    }
  
    //返回一条已完成响应；若无响应则返回 nullptr。
    SST::Vanadis::RoCCResponse *respond() override {
        if (resp_q.empty()) {
            return nullptr;
        }
        SST::Vanadis::RoCCResponse *temp = resp_q.front();
        resp_q.pop_front();
        return temp;
    }
  
    // Initialize subcomponents and parameterizable data structures
    void init(unsigned int phase) override {
  
        // Initialize arrayStates 调整其大小为 numArrays，即有多少阵列就有多少状态记录
        arrayStates.resize(numArrays);
        async_matrix_loads.resize(numArrays);
        async_vector_loads.resize(numArrays);
        inflight_compute_cmds.resize(numArrays);
        async_compute_states.resize(numArrays);
  
        // Set the address delimiters
        //inputOperandSize：每个输入操作数占用多少字节（比如float就是4字节）。
        //arrayInputSize：输入向量的长度
        inputDataSize = inputOperandSize * arrayInputSize;
        inputTotalSize = inputDataSize * numArrays;
        outputDataSize = outputOperandSize * arrayOutputSize;
        outputTotalSize = outputDataSize * numArrays;
        inputStartAddr = mmioStartAddr + numArrays;
        outputStartAddr = inputStartAddr + inputTotalSize;
  
        for (int i = 0; i < numArrays; i++) {
            arrayStates[i] = 0;
        }
        //配置RoCC使用的MMIO区间，把从mmioStartAddr开始、长度为inputTotalSize的区域
        //映射给内存接口，便于后续数据传输
    memInterface->setMemoryMappedAddressRegion(mmioStartAddr, inputTotalSize);
    memInterface->init(phase);
    array->init(phase);
    globalMem->init(phase);
    if (groupCtrl) {
        groupCtrl->init(phase);
    }
    if (requestScheduler) {
        requestScheduler->init(phase);
    }
    if (sfu) {
        sfu->init(phase);
    }
    }

    void setup() override {
        if (memInterface) {
            memInterface->setup();
        }
        if (array) {
            array->setup();
        }
        if (globalMem) {
            globalMem->setup();
        }
        if (groupCtrl) {
            groupCtrl->setup();
        }
        if (requestScheduler) {
            requestScheduler->setup();
        }
        if (sfu) {
            sfu->setup();
        }
    }

    void complete(unsigned int phase) override {
        if (memInterface) {
            memInterface->complete(phase);
        }
        if (array) {
            array->complete(phase);
        }
        if (globalMem) {
            globalMem->complete(phase);
        }
        if (groupCtrl) {
            groupCtrl->complete(phase);
        }
        if (requestScheduler) {
            requestScheduler->complete(phase);
        }
        if (sfu) {
            sfu->complete(phase);
        }
    }

    void finish() override {
        maybeReportMvmProgress(true);
        if (attentionClusterEnable_) {
            const uint64_t now = getCurrentSimCycle();
            statAttentionClusterQkArrayBusyUnionTicks_->addData(
                attentionClusterQkArrayActivity_.unionCycles(now));
            statAttentionClusterQkArrayBusySpanTicks_->addData(
                attentionClusterQkArrayActivity_.spanCycles(now));
            statAttentionClusterQkArrayIdleGapTicks_->addData(
                attentionClusterQkArrayActivity_.idleCycles(now));
            statAttentionClusterQkArrayMaxConcurrency_->addData(
                attentionClusterQkArrayActivity_.maxConcurrency());
            statAttentionClusterPvArrayBusyUnionTicks_->addData(
                attentionClusterPvArrayActivity_.unionCycles(now));
            statAttentionClusterPvArrayBusySpanTicks_->addData(
                attentionClusterPvArrayActivity_.spanCycles(now));
            statAttentionClusterPvArrayIdleGapTicks_->addData(
                attentionClusterPvArrayActivity_.idleCycles(now));
            statAttentionClusterPvArrayMaxConcurrency_->addData(
                attentionClusterPvArrayActivity_.maxConcurrency());
            statAttentionClusterOReadWaitCycles_->addData(
                attentionOAccumulator_.readWaitCycles());
            statAttentionClusterOWriteWaitCycles_->addData(
                attentionOAccumulator_.writeWaitCycles());
            statAttentionClusterOAluWaitCycles_->addData(
                attentionOAccumulator_.aluWaitCycles());
            statAttentionClusterOBankConflictCycles_->addData(
                attentionOAccumulator_.bankConflictCycles());
            statAttentionClusterODrainWaitCycles_->addData(
                attentionOAccumulator_.drainWaitCycles());
            statAttentionClusterOReadBusyUnionCycles_->addData(
                attentionOAccumulator_.readBusyCycles());
            statAttentionClusterOReadBusySpanCycles_->addData(
                attentionOAccumulator_.readBusySpanCycles());
            statAttentionClusterOReadIdleGapCycles_->addData(
                attentionOAccumulator_.readIdleCycles());
            statAttentionClusterOReadMaxConcurrency_->addData(
                attentionOAccumulator_.readMaxConcurrency());
            statAttentionClusterOWriteBusyUnionCycles_->addData(
                attentionOAccumulator_.writeBusyCycles());
            statAttentionClusterOWriteBusySpanCycles_->addData(
                attentionOAccumulator_.writeBusySpanCycles());
            statAttentionClusterOWriteIdleGapCycles_->addData(
                attentionOAccumulator_.writeIdleCycles());
            statAttentionClusterOWriteMaxConcurrency_->addData(
                attentionOAccumulator_.writeMaxConcurrency());
            statAttentionClusterOAluBusyUnionCycles_->addData(
                attentionOAccumulator_.aluBusyCycles());
            statAttentionClusterOAluBusySpanCycles_->addData(
                attentionOAccumulator_.aluBusySpanCycles());
            statAttentionClusterOAluIdleGapCycles_->addData(
                attentionOAccumulator_.aluIdleCycles());
            statAttentionClusterOAluMaxConcurrency_->addData(
                attentionOAccumulator_.aluMaxConcurrency());
        }
        if (memInterface) {
            memInterface->finish();
        }
        if (array) {
            array->finish();
        }
        if (globalMem) {
            globalMem->finish();
        }
        if (groupCtrl) {
            groupCtrl->finish();
        }
        if (requestScheduler) {
            requestScheduler->finish();
        }
        if (sfu) {
            sfu->finish();
        }
    }
  
    // Main clock cycle tick function
    //每一个时钟周期调用一次tick
    void tick(uint64_t cycle) override {
        output->verbose(CALL_INFO, 16, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
        LastTickCycle = cycle;
        // Keep draining async array-load commands even while a synchronous command is in flight.
        tryIssueAsyncArrayLoadCommand(cycle);
        tryCompleteAsyncArrayLoads(cycle);
        progressManagerTensorJobs();
        progressManagerAttentionJobs();
        progressAttentionWorker();
        progressAttentionWorkerClusterVBroadcast();
        if (workerCommandProcessor != nullptr && workerCommandProcessor->isBusy()) {
            workerCommandProcessor->tick(cycle);
        }

        if (roccCmd_q.empty() && !busy) {
            output->verbose(CALL_INFO, 16, 0, "--> nothing to do in RoCC\n");
            return;
        }
        output->verbose(CALL_INFO, 16, 0, "busy? %d\n", busy);
  
        if (!busy) {
            if (roccCmd_q.empty()) {
                return;
            }

            auto* next_cmd = roccCmd_q.front();
            if (next_cmd == nullptr || next_cmd->inst == nullptr) {
                roccCmd_q.pop_front();
                delete next_cmd;
                return;
            }
            if (next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_WAIT &&
                sfuWaitBlocked_ && next_cmd->cmd_id == sfuWaitBlockedCmdId_ &&
                getCurrentSimCycle() < sfuWaitBlockedUntilTick_) {
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == 0x3) {
                const uint64_t array_id = next_cmd->rs1;
                const bool is_async_compute = (next_cmd->inst->rd == 0);
                if (array_id >= static_cast<uint64_t>(numArrays)) {
                    enqueueResponse(new SST::Vanadis::RoCCResponse(next_cmd->inst->rd, 1, next_cmd->cmd_id, next_cmd->hw_thread));
                    roccCmd_q.pop_front();
                    delete next_cmd;
                    return;
                }
                if (hasArrayLoadFailure(static_cast<uint32_t>(array_id))) {
                    output->verbose(CALL_INFO, 0, 0,
                        "[RoCC ERROR] async load failed earlier for array=%" PRIu64 ", reject compute cmd_id=%" PRIu64 "\n",
                        array_id,
                        next_cmd->cmd_id);
                    enqueueResponse(new SST::Vanadis::RoCCResponse(next_cmd->inst->rd, 1, next_cmd->cmd_id, next_cmd->hw_thread));
                    roccCmd_q.pop_front();
                    delete next_cmd;
                    return;
                }
                if (array_id < static_cast<uint64_t>(numArrays) &&
                    isArrayLoadInflight(static_cast<uint32_t>(array_id))) {
                    // Keep queue order stable and retry in next cycle.
                    return;
                }

                const uint32_t array_id_u32 = static_cast<uint32_t>(array_id);
                auto& async_state = async_compute_states[array_id_u32];
                if (async_state.submitted) {
                    if (isArrayComputeInflight(array_id_u32) || !async_state.completed) {
                        return;
                    }

                    if (!is_async_compute) {
                        enqueueResponse(new SST::Vanadis::RoCCResponse(
                            next_cmd->inst->rd,
                            async_state.rd_val,
                            next_cmd->cmd_id,
                            next_cmd->hw_thread));
                        async_state = AsyncComputeState{};
                        roccCmd_q.pop_front();
                        delete next_cmd;
                        return;
                    }

                    // A new async submit on the same array must wait until software
                    // retires the prior async completion via a synchronous wait.
                    enqueueResponse(new SST::Vanadis::RoCCResponse(
                        next_cmd->inst->rd,
                        1,
                        next_cmd->cmd_id,
                        next_cmd->hw_thread));
                    roccCmd_q.pop_front();
                    delete next_cmd;
                    return;
                }

                if (isArrayComputeInflight(array_id_u32)) {
                    return;
                }

                roccCmd_q.pop_front();
                issueArrayCompute(next_cmd, array_id_u32, cycle);
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_TILE_MVM_BATCH) {
                roccCmd_q.pop_front();
                if (!tryIssueBatchComputeCommand(next_cmd, cycle)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_TILE_WAIT_BATCH) {
                roccCmd_q.pop_front();
                if (!tryWaitBatchComputeCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_TILE_GM2IMAT_BCAST) {
                roccCmd_q.pop_front();
                if (!tryIssueBatchArrayLoadCommand(next_cmd, cycle, true)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_TILE_GM2IVEC_BATCH) {
                roccCmd_q.pop_front();
                if (!tryIssueBatchArrayLoadCommand(next_cmd, cycle, false)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_WCP_START) {
                roccCmd_q.pop_front();
                if (!tryStartWorkerWindow(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_WCP_WAIT) {
                roccCmd_q.pop_front();
                if (!tryWaitWorkerWindow(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_SOFTMAX_TILE) {
                roccCmd_q.pop_front();
                if (!tryIssueSfuSoftmaxTileCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_WAIT) {
                roccCmd_q.pop_front();
                if (!tryWaitSfuCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_PRIMITIVE) {
                roccCmd_q.pop_front();
                if (!tryIssueSfuPrimitiveCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_WAIT) {
                roccCmd_q.pop_front();
                if (!tryWaitSfuPrimitiveCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH) {
                roccCmd_q.pop_front();
                if (!tryIssueSfuPrimitiveBatchCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_PRIMITIVE_BATCH_WAIT) {
                roccCmd_q.pop_front();
                if (!tryWaitSfuPrimitiveBatchCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd != nullptr && next_cmd->inst != nullptr && next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_SFU_JOB) {
                roccCmd_q.pop_front();
                if (!tryIssueSfuJobCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_TENSOR_MANAGER_JOB) {
                roccCmd_q.pop_front();
                if (!tryIssueManagerTensorJobCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_TENSOR_MANAGER_WAIT) {
                roccCmd_q.pop_front();
                if (!tryWaitManagerTensorJobCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }
            if (next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_JOB) {
                roccCmd_q.pop_front();
                tryIssueManagerAttentionJobCommand(next_cmd);
                return;
            }
            if (next_cmd->inst->func7 == GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_WAIT) {
                roccCmd_q.pop_front();
                if (!tryWaitManagerAttentionJobCommand(next_cmd)) {
                    roccCmd_q.push_front(next_cmd);
                }
                return;
            }

            busy = true;
            curr_cmd = next_cmd;
            roccCmd_q.pop_front();
            StartTickCycle = cycle;
            //根据当前命令的操作码（func7）选择要执行的功能
            switch (curr_cmd->inst->func7) {
                case 0x1: // Set Matrix
                {   
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                              "the Instruction read: mvm.set (MVM set matrix)\n");
                    setMatrix();
                } break;
                case 0x2: // Load Vector
                {   
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                              "the Instruction read: mvm.l (MVM load vector)\n");
                    loadVector();
                } break;
                case 0x3: // Compute MVM
                {   
                    const uint64_t array_id = curr_cmd->rs1;
                    if (array_id < static_cast<uint64_t>(numArrays) && hasArrayLoadFailure(static_cast<uint32_t>(array_id))) {
                        output->verbose(CALL_INFO, 0, 0,
                            "[RoCC ERROR] async load failed earlier for array=%" PRIu64 ", reject compute cmd_id=%" PRIu64 "\n",
                            array_id,
                            curr_cmd->cmd_id);
                        completeRoCC(1);
                        break;
                    }
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                              "the Instruction read: mvm (MVM compute)\n");
                    computeMVM();
                } break;
                case 0x4: // Store Vector
                {   
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                              "the Instruction read: mvm.s (MVM store vector)\n");
                    storeVector();
                } break;
                case 0x5: // Move Vector
                {   
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                              "the Instruction read: mvm.mv (MVM move vector)\n");
                    moveVector();
                } break;
                case 0x6: //mvm.ovec2gm  
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: mvm.ovec2gm (MVM outputvector store)\n");
                    OutputvectorStore(cycle);
                } break;
                case 0x7: //mvm.gm2ivec 
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: mvm.gm2vec (MVM inputvector load)\n");
                    IntputvectorLoad(cycle);
                } break;
                case 0x8: //mvm.gm2imat
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: mvm.gm2imat (MVM inputmatrix load)\n");
                    InputMatrixLoad(cycle);
                } break;
                case 0x9: //remote_st Remote Store 
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: remote_st (MVM remote store)\n");
                    RemoteStore(cycle);
                } break;
                case 0xA: //remote_ld Remote Load 
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: remote_ld (MVM remote load)\n");
                    RemoteLoad(cycle);
                } break;
                case GOLEM_ROCC_FUNC7_REMOTE_STORE_WAIT:
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: remote_st.wait (waitable HBM store)\n");
                    RemoteStoreWait(cycle);
                } break;
                case 0xB: //mvm.slen Remote transfer length setup
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: mvm.slen (MVM remote length setup)\n");
                    SetRemoteLength();
                } break;
                case 0xC: //mvm.ocfg Output buffer configuration
                {
                    output->verbose(CALL_INFO, 1, 0,
                              "Instruction read: mvm.ocfg (MVM output config)\n");
                    ConfigureOutputMode();
                } break;
                case 0xD: // mm2gm (Main Memory -> Global Memory)
                {
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                            "Instruction read: mm2gm (Main Memory -> Global Memory)\n");
                    MainMem2GlobalMem();
                } break;
                case 0xE: // gm2mm (Global Memory -> Main Memory)
                {
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                            "Instruction read: gm2mm (Global Memory -> Main Memory)\n");
                    GlobalMem2MainMem();
                } break;
                case 0xF: // reg2gm (Register -> Global Memory)
                {
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                            "Instruction read: reg2gm (Register -> Global Memory)\n");
                    Reg2GlobalMem();
                } break;
                case 0x10: // gm2reg (Global Memory -> Register)
                {
                    output->verbose(CALL_INFO, 1, 0, "[Core %" PRIu64 "] -> tick RoCC at cycle %" PRIu64 "\n", coreID, cycle);
                    output->verbose(CALL_INFO, 1, 0,
                            "Instruction read: gm2reg (Global Memory -> Register)\n");
                    GlobalMem2Reg();
                } break;
                case GOLEM_ROCC_FUNC7_TILE_MVM_BATCH:
                {
                    if (!tryIssueBatchComputeCommand(curr_cmd, cycle)) {
                        busy = false;
                        curr_cmd = nullptr;
                        return;
                    }
                    busy = false;
                    curr_cmd = nullptr;
                } break;
                case GOLEM_ROCC_FUNC7_TILE_WAIT_BATCH:
                {
                    if (!tryWaitBatchComputeCommand(curr_cmd)) {
                        busy = false;
                        curr_cmd = nullptr;
                        return;
                    }
                    busy = false;
                    curr_cmd = nullptr;
                } break;
                case GOLEM_ROCC_FUNC7_TILE_GM2IMAT_BCAST:
                {
                    if (!tryIssueBatchArrayLoadCommand(curr_cmd, cycle, true)) {
                        busy = false;
                        curr_cmd = nullptr;
                        return;
                    }
                    busy = false;
                    curr_cmd = nullptr;
                } break;
                case GOLEM_ROCC_FUNC7_TILE_GM2IVEC_BATCH:
                {
                    if (!tryIssueBatchArrayLoadCommand(curr_cmd, cycle, false)) {
                        busy = false;
                        curr_cmd = nullptr;
                        return;
                    }
                    busy = false;
                    curr_cmd = nullptr;
                } break;
                case GOLEM_ROCC_FUNC7_WCP_START:
                {
                    if (!tryStartWorkerWindow(curr_cmd)) {
                        busy = false;
                        curr_cmd = nullptr;
                        return;
                    }
                    busy = false;
                    curr_cmd = nullptr;
                } break;
                case GOLEM_ROCC_FUNC7_WCP_WAIT:
                {
                    if (!tryWaitWorkerWindow(curr_cmd)) {
                        busy = false;
                        curr_cmd = nullptr;
                        return;
                    }
                    busy = false;
                    curr_cmd = nullptr;
                } break;
                default: {
                    output->verbose(CALL_INFO, 0, 0, "ERROR: unrecognized RoCC func7\n");
                    completeRoCC(1);
                } break;
            }
        } else {
            if (curr_cmd != nullptr) {
                switch (curr_cmd->inst->func7) {
                    case 0x6:
                        OutputvectorStore(cycle);
                        break;
                    case 0x7:
                        IntputvectorLoad(cycle);
                        break;
                    case 0x8:
                        InputMatrixLoad(cycle);
                        break;
                    case 0x9:
                        RemoteStore(cycle);
                        break;
                    case 0xA:
                        RemoteLoad(cycle);
                        break;
                    case GOLEM_ROCC_FUNC7_REMOTE_STORE_WAIT:
                        RemoteStoreWait(cycle);
                        break;
                    default:
                        break;
                }
            }
        }
    }
  
    // Issues the read request for the matrix that will be set in the analog array
    void setMatrix() {
        //取出当前RoCC命令的rs1寄存器值，通常代表了矩阵数据的物理地址
        uint64_t rs1 = curr_cmd->rs1;
        output->verbose(CALL_INFO, 1, 0, "RoCC setMatrix rs1: 0x%" PRIx64 "\n", rs1);
        uint32_t load_matrix_flag = 0x0;
        //计算本次要加载的矩阵总字节数，等于输入维数 × 输出维数 × 每个元素的字节数
        matrix_total_size = arrayInputSize * arrayOutputSize * inputOperandSize;
        //查询内存子系统缓存行大小
        uint64_t cache_line_size = memInterface->getLineSize();

        matrix_read_offset = 0;

        //直接用rs1作为物理地址 
        uint64_t physAddr = rs1; // Assuming rs1 is physical address
        //计算这个物理地址在当前cache line中的偏移量 比如 cache_line_size=64，physAddr=0x108，那么offset=8
        uint64_t addr_offset = physAddr % cache_line_size;

        // Calculate initial request size
        //本次首个内存读取请求的字节数：如果起始地址没对齐，需要先补齐一个cache line不能超过本次矩阵的总大小
        uint32_t request_size = std::min(static_cast<uint64_t>(cache_line_size - addr_offset), matrix_total_size);

        // Send first cache request
        //构造一个内存读取请求，请求地址为physAddr，请求长度为request_size，标记“这是个矩阵数据”
        //然后调用memInterface->send()发给内存子系统
        auto *load_req = new StandardMem::Read(physAddr, request_size, load_matrix_flag);
        memInterface->send(load_req);
    }
    
    void loadVector() {
        uint64_t rs1 = curr_cmd->rs1;
        output->verbose(CALL_INFO, 1, 0, "RoCC loadVector rs1: 0x%" PRIx64 "\n", rs1);
        uint32_t load_vector_flag = 0x1;
        vector_total_size = arrayInputSize * inputOperandSize;
        uint64_t cache_line_size = memInterface->getLineSize();

        vector_read_offset = 0;

        uint64_t physAddr = rs1; // Assuming rs1 is physical address
        uint64_t addr_offset = physAddr % cache_line_size;

        // Calculate initial request size
        uint32_t request_size = std::min(static_cast<uint64_t>(cache_line_size - addr_offset), vector_total_size);

        // Send first cache request
        auto *load_req = new StandardMem::Read(physAddr, request_size, load_vector_flag);
        memInterface->send(load_req);
    }
    
    void computeMVM() {
        uint64_t rs1 = curr_cmd->rs1;//rs1表示阵列ID，在多阵列场景下，每个阵列都有唯一编号
        arrayStates[rs1] = 1;//标记“这个阵列正在计算中”
        array->beginComputation(static_cast<uint32_t>(rs1));   //调用Golem的ComputeArray子组件的beginComputation方法
    }

    bool tryIssueBatchComputeCommand(SST::Vanadis::RoCCCommand* cmd, uint64_t cycle) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        const uint64_t start_array = cmd->rs1;
        const uint64_t count = cmd->rs2;
        if (count == 0 || start_array >= static_cast<uint64_t>(numArrays) || (start_array + count) > static_cast<uint64_t>(numArrays)) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }

        for (uint64_t idx = 0; idx < count; ++idx) {
            const uint32_t array_id = static_cast<uint32_t>(start_array + idx);
            if (hasArrayLoadFailure(array_id)) {
                enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
                delete cmd;
                return true;
            }
            if (isArrayLoadInflight(array_id) || isArrayComputeInflight(array_id)) {
                return false;
            }
            if (async_compute_states[array_id].submitted) {
                enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
                delete cmd;
                return true;
            }
        }

        for (uint64_t idx = 0; idx < count; ++idx) {
            const uint32_t array_id = static_cast<uint32_t>(start_array + idx);
            auto* array_cmd = new SST::Vanadis::RoCCCommand(cmd->inst, array_id, 0, cmd->cmd_id, cmd->hw_thread);
            auto& inflight = inflight_compute_cmds[array_id];
            inflight.cmd = array_cmd;
            inflight.start_cycle = cycle;
            inflight.async_mode = true;
            auto& async_state = async_compute_states[array_id];
            async_state.submitted = true;
            async_state.completed = false;
            async_state.rd_val = 0;
            arrayStates[array_id] = 1;
            array->beginComputation(array_id);
        }

        enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryWaitBatchComputeCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        const uint64_t start_array = cmd->rs1;
        const uint64_t count = cmd->rs2;
        if (count == 0 || start_array >= static_cast<uint64_t>(numArrays) || (start_array + count) > static_cast<uint64_t>(numArrays)) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }

        uint64_t aggregate_rd_val = 0;
        for (uint64_t idx = 0; idx < count; ++idx) {
            const uint32_t array_id = static_cast<uint32_t>(start_array + idx);
            auto& async_state = async_compute_states[array_id];
            if (!async_state.submitted) {
                enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
                delete cmd;
                return true;
            }
            if (isArrayComputeInflight(array_id) || !async_state.completed) {
                return false;
            }
            aggregate_rd_val |= async_state.rd_val;
        }

        for (uint64_t idx = 0; idx < count; ++idx) {
            async_compute_states[static_cast<uint32_t>(start_array + idx)] = AsyncComputeState{};
        }

        enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, aggregate_rd_val, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryIssueBatchArrayLoadCommand(SST::Vanadis::RoCCCommand* cmd, uint64_t cycle, bool is_matrix) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        const uint64_t base_addr = cmd->rs1;
        const uint64_t count = cmd->rs2;
        if (count == 0 || count > static_cast<uint64_t>(numArrays)) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }

        const uint64_t vector_data_bytes = static_cast<uint64_t>(arrayInputSize) * static_cast<uint64_t>(inputOperandSize);
        const uint64_t vector_stride = std::max<uint64_t>(vectorStrideBytes, vector_data_bytes);
        for (uint64_t idx = 0; idx < count; ++idx) {
            const uint32_t array_id = static_cast<uint32_t>(idx);
            auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
            if (state.inflight || isArrayComputeInflight(array_id)) {
                return false;
            }
        }

        for (uint64_t idx = 0; idx < count; ++idx) {
            const uint32_t array_id = static_cast<uint32_t>(idx);
            auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
            const uint64_t address =
                is_matrix ? base_addr : (base_addr + idx * vector_stride);
            const uint64_t total_size = is_matrix
                ? static_cast<uint64_t>(arrayInputSize) * static_cast<uint64_t>(arrayOutputSize) * static_cast<uint64_t>(inputOperandSize)
                : vector_data_bytes;
            initializeAsyncArrayLoad(
                state, array_id, address, total_size, false, 0);
            progressAsyncArrayLoad(array_id, is_matrix);
        }

        enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryStartWorkerWindow(SST::Vanadis::RoCCCommand* cmd) {
        if (workerCommandProcessor == nullptr || cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (workerCommandProcessor->isBusy()) {
            return false;
        }
        std::vector<uint8_t> raw;
        globalMem->rd_from_globalmem(cmd->rs1, sizeof(WorkerTaskListHeader), raw);
        if (raw.size() < sizeof(WorkerTaskListHeader)) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        WorkerTaskListHeader header{};
        std::memcpy(&header, raw.data(), sizeof(header));
        const bool blockKSupported =
            arrayInputSize > 0 &&
            (header.block_k <= static_cast<uint32_t>(arrayInputSize) ||
             (header.block_k % static_cast<uint32_t>(arrayInputSize)) == 0);
        if (header.block_n == 0 || header.block_n > numArrays ||
            header.block_k == 0 || !blockKSupported ||
            header.block_m != arrayOutputSize ||
            header.elem_bytes != inputOperandSize ||
            header.elem_bytes != outputOperandSize) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        const bool ok = workerCommandProcessor->startWindow(header);
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, ok ? 0 : 1, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return ok;
    }

    bool tryWaitWorkerWindow(SST::Vanadis::RoCCCommand* cmd) {
        if (workerCommandProcessor == nullptr || cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (workerCommandProcessor->isBusy()) {
            return false;
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryIssueSfuSoftmaxTileCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (sfu == nullptr) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        if (!sfu->issueSoftmaxTile(cmd->rs1, cmd->rs2)) {
            return false;
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryWaitSfuCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (sfu == nullptr) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        uint64_t completionTick = 0;
        if (sfuWaitBlocked_ && cmd->cmd_id == sfuWaitBlockedCmdId_) {
            if (getCurrentSimCycle() < sfuWaitBlockedUntilTick_) {
                return false;
            }
            sfuWaitBlocked_ = false;
        }
        if (sfu->completionTick(cmd->rs1, &completionTick) &&
            getCurrentSimCycle() < completionTick) {
            sfuWaitBlocked_ = true;
            sfuWaitBlockedCmdId_ = cmd->cmd_id;
            sfuWaitBlockedUntilTick_ = completionTick;
            return false;
        }
        uint64_t status = 1;
        if (!sfu->wait(cmd->rs1, &status)) {
            return false;
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, status, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryIssueSfuPrimitiveCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (sfu == nullptr) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        if (!sfu->issuePrimitive(cmd->rs1, cmd->rs2)) {
            return false;
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryWaitSfuPrimitiveCommand(SST::Vanadis::RoCCCommand* cmd) {
        return tryWaitSfuCommand(cmd);
    }

    bool tryIssueSfuPrimitiveBatchCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (sfu == nullptr) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        if (!sfu->issuePrimitiveBatch(cmd->rs1, cmd->rs2)) {
            return false;
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryWaitSfuPrimitiveBatchCommand(SST::Vanadis::RoCCCommand* cmd) {
        return tryWaitSfuCommand(cmd);
    }

    bool tryIssueSfuJobCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        if (sfu == nullptr) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        if (!sfu->issueJob(cmd->rs1, cmd->rs2)) {
            return false;
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    enum class AttentionWorkerPhase : uint8_t {
        Idle,
        LoadingKv,
        LoadingQ,
        QkProgramMatrix,
        QkProgramInputs,
        QkCompute,
        QkReadOutputs,
        Softmax,
        PvProgramMatrix,
        PvProgramInputs,
        PvRestoreOutput,
        PvCompute,
        PvReadOutputs,
        OutputDma,
        Complete,
    };

    enum class AttentionInterTilePhase : uint8_t {
        OutputDma,
        QueryLoad,
        KvLoad,
        QLocalRead,
        QkMatrixProgram,
        QkInputProgram,
        QkComputeReadout,
        Count,
    };

    enum class AttentionTilePipelinePhase : uint8_t {
        KvLoad,
        QLocalRead,
        QkMatrixProgram,
        QkInputProgram,
        QkComputeReadout,
        Softmax,
        PvMatrixProgram,
        PvInputProgram,
        PvRestoreOutput,
        PvCompute,
        PvOutputReadwrite,
        Count,
    };

    struct AttentionKvBufferState {
        uint32_t queryTileIndex = UINT32_MAX;
        uint32_t kvTileIndex = UINT32_MAX;
        uint32_t physicalKvTileIndex = UINT32_MAX;
        uint32_t loadsPending = 0;
        uint64_t issueTick = 0;
        uint64_t readyTick = 0;
        uint64_t consumeTick = 0;
        bool kReady = false;
        bool ready = false;
        bool consuming = false;
        bool pairUseRecorded = false;
        bool crossQuery = false;
    };

    enum class AttentionQkInputSlotState : uint8_t {
        Free,
        Reading,
        Ready,
        Programming,
    };

    struct AttentionQkInputSlot {
        AttentionQkInputSlotState state = AttentionQkInputSlotState::Free;
        uint64_t generation = 0;
        uint32_t queryTileIndex = UINT32_MAX;
        uint32_t kvTileIndex = UINT32_MAX;
        uint32_t phaseSliceIndex = UINT32_MAX;
        uint32_t arrayId = UINT32_MAX;
        uint64_t transferTag = 0;
        std::vector<uint8_t> payload;
    };

    enum class AttentionAheadOperandPhase : uint8_t {
        Idle,
        QueryRead,
        MatrixProgram,
        InputRead,
        InputProgram,
        Ready,
    };

    struct AttentionAheadOperandContext {
        AttentionAheadOperandPhase phase = AttentionAheadOperandPhase::Idle;
        uint64_t generation = 0;
        uint32_t queryTileIndex = UINT32_MAX;
        uint32_t kvTileIndex = UINT32_MAX;
        uint32_t physicalKvTileIndex = UINT32_MAX;
        uint32_t operandBank = 0;
        uint64_t qLocal = 0;
        uint64_t kLocal = 0;
        size_t queryOffset = 0;
        uint32_t inputIndex = 0;
        uint64_t transferTag = 0;
        uint64_t startTick = 0;
        uint64_t waitStartTick = 0;
        bool requestInFlight = false;
        bool commitWaiting = false;
        std::vector<uint8_t> queryBytes;
        std::vector<uint8_t> inputBytes;
    };

    enum class AttentionClusterAheadPhase : uint8_t {
        Idle,
        QueryDma,
        QueryRead,
        InputProgram,
        Launch,
        Compute,
        OutputRead,
        ScoreWrite,
        SoftmaxIssue,
        SoftmaxRunning,
        Ready,
    };

    struct AttentionClusterAheadContext {
        AttentionClusterAheadPhase phase = AttentionClusterAheadPhase::Idle;
        uint64_t generation = 0;
        uint32_t queryTileIndex = UINT32_MAX;
        uint32_t kvTileIndex = UINT32_MAX;
        uint32_t physicalKvTileIndex = UINT32_MAX;
        uint32_t operandBank = 0;
        uint64_t qLocal = 0;
        AttentionClusterTag tag = {};
        int32_t qkContext = -1;
        int32_t scoreContext = -1;
        int32_t pContext = -1;
        size_t queryOffset = 0;
        uint32_t wave = 0;
        uint32_t pair = 0;
        uint32_t launchIndex = 0;
        uint32_t arraysPending = 0;
        uint32_t outputIndex = 0;
        uint64_t transferTag = 0;
        bool requestInFlight = false;
        uint64_t qkStartCycle = 0;
        uint64_t qkEndCycle = 0;
        uint64_t sfuStartCycle = 0;
        uint64_t sfuEndCycle = 0;
        std::vector<uint8_t> queryBytes;
        std::vector<double> qPayload;
        std::vector<float> pendingScoreBeat;
        std::vector<float> outputScales;
    };

    enum class AttentionClusterQkMatrixAheadPhase : uint8_t {
        Idle,
        Reading,
        Programming,
        Ready,
    };

    struct AttentionClusterQkMatrixAheadContext {
        AttentionClusterQkMatrixAheadPhase phase =
            AttentionClusterQkMatrixAheadPhase::Idle;
        uint64_t generation = 0;
        uint32_t kvTileIndex = UINT32_MAX;
        uint32_t physicalKvTileIndex = UINT32_MAX;
        uint32_t operandBank = 0;
        uint32_t buffer = UINT32_MAX;
        uint32_t half = 0;
        size_t readOffset = 0;
        bool requestInFlight = false;
        AttentionClusterTag tag = {};
        std::vector<uint8_t> bytes;
        std::vector<double> payload;
    };

    struct AttentionClusterPvMatrixAheadContext {
        AttentionClusterQkMatrixAheadPhase phase =
            AttentionClusterQkMatrixAheadPhase::Idle;
        uint64_t generation = 0;
        uint32_t kvTileIndex = UINT32_MAX;
        uint32_t physicalKvTileIndex = UINT32_MAX;
        uint32_t operandBank = 0;
        uint32_t buffer = UINT32_MAX;
        uint32_t half = 0;
        size_t readOffset = 0;
        bool requestInFlight = false;
        AttentionClusterTag tag = {};
        std::vector<uint8_t> bytes;
        std::vector<double> payload;
    };

    struct AttentionReuseWindowRowContext {
        uint32_t rowBlock = 0;
        uint32_t rowOffset = 0;
        uint32_t nextKvTile = 0;
        int32_t scoreSlot = -1;
        int32_t pSlot = -1;
        bool requestInFlight = false;
        uint8_t fifoPhase = 0;
        AttentionClusterTag fifoTag = {};
        std::vector<float> p;
        std::vector<float> groupScale;
        std::vector<float> outputScale;
    };

    struct AttentionReuseWindowPvAggregate {
        std::vector<float> p;
        std::vector<float> scales;
        uint16_t rowSliceMask = 0;
        bool vPrefetchReadyToHint = false;
        bool vPrefetchHinted = false;
        bool dispatched = false;
    };

    struct AttentionWorkerState {
        ControlTransportMessage dispatch = {};
        uint64_t generation = 0;
        AttentionWorkerPhase phase = AttentionWorkerPhase::Idle;
        uint64_t qLocal = 0;
        uint64_t kLocal = 0;
        uint64_t vLocal = 0;
        std::vector<uint64_t> qLocalBuffers;
        std::vector<uint64_t> kLocalBuffers;
        std::vector<uint64_t> vLocalBuffers;
        uint64_t spLocal = 0;
        uint64_t oLocal = 0;
        std::vector<uint64_t> oLocalBuffers;
        uint32_t activeOperandBank = 0;
        AttentionAheadOperandContext aheadOperands;
        std::array<AttentionClusterAheadContext, 4> clusterAhead = {};
        AttentionClusterQkMatrixAheadContext clusterQkMatrixAhead;
        AttentionClusterPvMatrixAheadContext clusterPvMatrixAhead;
        std::array<bool, 4> clusterQueryLoaded = {{false, false, false, false}};
        bool clusterAheadPromotionWaiting = false;
        bool clusterQkMatrixAheadPromotionWaiting = false;
        bool clusterPvMatrixAheadPromotionWaiting = false;
        bool clusterAheadEnabledForTile = false;
        std::function<void()> clusterOwnerRetry;
        std::deque<std::function<void()>> clusterOwnerRetryQueue;
        uint32_t clusterOwnerLaunchNext = 0;
        uint32_t clusterOwnerLaunchEnd = 0;
        bool clusterOwnerLaunchQk = false;
        uint32_t pvOutputSliceToProgram = 0;
        uint32_t sequentialPvOutputArray = 0;
        std::vector<double> sequentialKTilePayload;
        std::vector<double> sequentialPPayload;
        std::vector<double> sequentialOPayload;
        uint64_t reuseWindowFusionTiles = 0;
        uint64_t reuseWindowExpectedFusionTiles = 0;
        uint64_t reuseWindowQkStartCycle = 0;
        uint64_t reuseWindowQkEndCycle = 0;
        uint64_t reuseWindowPvCycles = 0;
        uint32_t reuseWindowAttentionTile = 0;
        uint32_t reuseWindowPvFusionTiles = 0;
        std::vector<float> reuseWindowScores;
        std::vector<float> reuseWindowOutput;
        std::vector<float> reuseWindowP;
        std::vector<float> reuseWindowGroupScale;
        uint32_t reuseWindowClusterPvCompleted = 0;
        std::vector<uint8_t> reuseWindowScoreReady;
        bool reuseWindowSoftmaxBusy = false;
        bool reuseWindowQkComplete = false;
        std::vector<AttentionReuseWindowRowContext> reuseWindowRowContexts;
        std::vector<AttentionReuseWindowPvAggregate> reuseWindowPvAggregates;
        std::array<int32_t, 16> reuseWindowScoreOwners = {};
        std::array<int32_t, 16> reuseWindowPOwners = {};
        std::array<int32_t, 16> reuseWindowSliceOwners = {};
        std::array<uint32_t, 16> reuseWindowSliceNextRowBlock = {};
        std::array<uint64_t, 16> reuseWindowSliceWaitStart = {};
        uint32_t reuseWindowNextScoreContext = 0;
        uint32_t reuseWindowNextPContext = 0;
        uint64_t reuseWindowScoreSlotStallCycles = 0;
        uint64_t reuseWindowPSlotStallCycles = 0;
        uint64_t reuseWindowRowContextSwitches = 0;
        uint64_t reuseWindowLastScoreSlotStallCycle = UINT64_MAX;
        uint64_t reuseWindowLastPSlotStallCycle = UINT64_MAX;
        std::array<uint32_t, 3> reuseWindowPvNextRowBlock = {{0, 1, 2}};
        std::array<uint32_t, 3> reuseWindowPvNextWindow = {};
        uint32_t reuseWindowPvDispatchLane = 0;
        uint64_t reuseWindowPvNextDispatchCycle = 0;
        std::vector<int32_t> reuseWindowPvRowCore;
        std::vector<uint8_t> reuseWindowPvAllocationPending;
        std::vector<uint32_t> reuseWindowPvRowNextWindow;
        std::deque<uint32_t> reuseWindowReadQueue;
        bool reuseWindowIoBusy = false;
        uint32_t reuseWindowRowContextsCompleted = 0;
        bool clusterQkTileStartValid = false;
        uint64_t clusterLastQkTileStartCycle = 0;
        uint32_t clusterLastQkTileStartGroup = UINT32_MAX;
        uint32_t clusterLastQkTileStartKvTile = UINT32_MAX;
        uint64_t vTileBufferLocal = 0;
        uint32_t queryTileIndex = 0;
        uint32_t physicalKvTileIndex = 0;
        uint32_t kvTileIndex = 0;
        uint32_t phaseSliceIndex = 0;
        uint32_t qkReductionSlice = 0;
        uint32_t index = 0;
        uint32_t lane = 0;
        uint32_t arraysPending = 0;
        int32_t clusterQkContext = -1;
        int32_t clusterScoreContext = -1;
        int32_t clusterPContext = -1;
        int32_t clusterPvContext = -1;
        std::array<int32_t, 4> clusterOContextSlots = {{-1, -1, -1, -1}};
        std::array<AttentionClusterTag, 4> clusterOTags = {};
        AttentionClusterTag clusterTileTag = {};
        uint32_t clusterQkReductionSlice = 0;
        uint32_t clusterQkWave = 0;
        uint32_t clusterQkPair = 0;
        uint32_t clusterPvWave = 0;
        uint32_t clusterPvOperandBank = 0;
        bool clusterPvBankLeaseHeld = false;
        AttentionClusterTag clusterPvBankLeaseTag = {};
        bool clusterPvMatrixResident = false;
        uint32_t clusterPvMatrixKvTile = UINT32_MAX;
        uint32_t clusterPvMatrixGroupOwner = UINT32_MAX;
        uint32_t clusterPvOutputIssued = 0;
        uint32_t clusterPvOutputCompleted = 0;
        uint32_t clusterPvOutputInFlight = 0;
        std::vector<double> clusterPvNextInput;
        uint32_t clusterPvNextInputRow = UINT32_MAX;
        bool clusterPvNextInputReady = false;
        bool clusterPvNextInputInFlight = false;
        bool clusterPvWavefrontProgrammingComplete = false;
        uint32_t attentionKvLoadsPending = 0;
        uint32_t attentionPvInputsPending = 0;
        uint32_t attentionPvRestoresPending = 0;
        uint32_t attentionPvOutputWritesPending = 0;
        uint32_t attentionOutputDmaRowsPending = 0;
        uint32_t attentionOutputDmaNextRow = 0;
        uint32_t attentionOutputDmaRowsInFlight = 0;
        uint32_t attentionPvMatrixProgramsPending = 0;
        uint32_t activeKvBuffer = 0;
        std::vector<AttentionKvBufferState> kvBuffers;
        uint32_t attentionWaitingKvBuffer = UINT32_MAX;
        uint64_t attentionKvKReleaseTick = 0;
        uint64_t attentionKvVReleaseTick = 0;
        uint64_t attentionKvSecondLookaheadEligibleTick = 0;
        bool attentionKvNextReadyAtRelease = false;
        bool attentionWaitingForPrefetch = false;
        bool attentionWaitingForV = false;
        uint64_t attentionVWaitTick = 0;
        bool attentionSoftmaxComplete = false;
        bool attentionPvMatrixComplete = false;
        bool attentionPvRestoreReadRetry = false;
        bool attentionPvOutputWriteRetry = false;
        bool attentionPvOutputWriteToCBuffer = false;
        uint32_t attentionClusterOCommitsPending = 0;
        bool attentionClusterODrainPending = false;
        uint64_t attentionClusterODrainId = 0;
        bool attentionPvPreparationComplete = false;
        bool attentionPvInputResidentValid = false;
        uint32_t attentionPvInputResidentQueryTile = UINT32_MAX;
        uint32_t attentionPvInputResidentKvTile = UINT32_MAX;
        bool attentionOAccumulatorStorageActive = false;
        bool attentionQkInputProgrammingComplete = false;
        std::array<AttentionQkInputSlot, 2> qkInputSlots;
        uint32_t qkInputNextFetch = 0;
        uint32_t qkInputNextProgram = 0;
        bool qkInputReadInFlight = false;
        bool qkInputProgramInFlight = false;
        bool qkInputOverlapActive = false;
        uint64_t qkInputOverlapStartTick = 0;
        uint32_t qkInputMaxDepth = 0;
        uint32_t qkReadKvSubtile = UINT32_MAX;
        uint32_t qkReadIssueIndex = 0;
        uint32_t qkReadCommitIndex = 0;
        uint32_t qkReadInFlight = 0;
        uint32_t qkReadCommitLane = 0;
        bool qkReadCommitInFlight = false;
        std::vector<std::vector<uint8_t>> qkReadBuffers;
        std::vector<uint8_t> qkReadReady;
        uint32_t qkRowBurstKvSubtile = UINT32_MAX;
        bool qkRowBurstStorageActive = false;
        uint64_t attentionPvOutputWriteAddr = 0;
        uint32_t attentionPvOutputWriteAccumulatorRow = UINT32_MAX;
        std::vector<uint8_t> attentionPvOutputWriteBytes;
        std::vector<uint8_t> transferBytes;
        std::vector<double> arrayPayload;
        std::vector<double> qPayload;
        std::vector<double> vPayload;
        uint32_t vTileTag = UINT32_MAX;
        uint64_t vTileGeneration = 0;
        uint32_t vTileGroupOwner = UINT32_MAX;
        bool vTileValid = false;
        bool vTileBufferWaiting = false;
        bool vTileBufferBypassWait = false;
        uint64_t vTileBufferWaitUntilTick = 0;
        uint64_t vTileBufferCurrentWaitTicks = 0;
        std::vector<uint8_t> readOutputBytes;
        std::vector<float> outputScales;
        uint64_t localAddr = 0;
        size_t localLength = 0;
        size_t localOffset = 0;
        bool localInflight = false;
        bool localWrite = false;
        bool interTileActive = false;
        uint64_t interTileStartTick = 0;
        uint64_t interTilePhaseStartTick = 0;
        std::array<uint64_t, static_cast<size_t>(AttentionInterTilePhase::Count)>
            interTilePhaseTicks = {};
        bool tilePipelineActive = false;
        uint64_t tilePipelineStartTick = 0;
        uint64_t tilePipelinePhaseStartTick = 0;
        AttentionTilePipelinePhase tilePipelinePhase =
            AttentionTilePipelinePhase::KvLoad;
        std::array<uint64_t, static_cast<size_t>(AttentionTilePipelinePhase::Count)>
            tilePipelinePhaseTicks = {};
        std::function<void(bool, const std::vector<uint8_t>&)> localCallback;
    };

    struct AttentionWorkerClusterPvState {
        bool busy = false;
        ControlTransportMessage active = {};
        uint32_t fusionTiles = 0;
        uint64_t startCycle = 0;
        bool nextPrefetchAttempted = false;
        bool nextPrefetchPending = false;
        bool nextPrefetchReady = false;
        uint64_t nextPrefetchGeneration = 0;
        uint64_t nextPrefetchSource = 0;
        size_t nextPrefetchBytesReceived = 0;
        uint32_t nextPrefetchRelayMask = 0;
        WorkerTaskListHeader nextPrefetchHeader = {};
        std::vector<uint8_t> nextPrefetchPayload;
        std::vector<uint8_t> nextPrefetchChunkSeen;
    };

    struct AttentionWorkerClusterVEntry {
        bool loading = false;
        bool ready = false;
        size_t bytes = 0;
        std::vector<uint8_t> data;
        std::unordered_set<uint32_t> subscribers;
        std::unordered_set<uint32_t> delivered;
    };

    struct AttentionWorkerClusterVLoad {
        bool active = false;
        bool dmaIssued = false;
        bool dmaDone = false;
        bool dmaOk = false;
        bool readInFlight = false;
        uint64_t source = 0;
        uint64_t scratch = 0;
        size_t bytes = 0;
        size_t readOffset = 0;
    };

    uint64_t attentionTransferTag() { return allocateLocalTransferTag(); }

    bool programAttentionGemmMatrixAsync(
        uint32_t arrayId, const std::vector<double>& matrix, size_t elemBytes,
        uint64_t tag, Golem::ComputeArray::BufferCallback callback) {
        const uint32_t bank = attentionWorker_
            ? (attentionClusterEnable_ &&
                    arrayId >= attentionClusterConfig_.qkArrays
                ? attentionWorker_->clusterPvOperandBank
                : attentionWorker_->activeOperandBank)
            : 0;
        if (attentionGenericGemmEnable_) {
            return workerCommandProcessor->programGemmMatrixBankAsync(
                arrayId, bank, matrix, elemBytes, tag, LastTickCycle,
                std::move(callback));
        }
        return array->programMatrixBankAsync(
            arrayId, bank, matrix, elemBytes, tag, std::move(callback));
    }

    bool programAttentionGemmMatrixGroupAsync(
        const std::vector<uint32_t>& arrayIds,
        const std::vector<double>& matrix, size_t elemBytes, uint64_t tag,
        Golem::ComputeArray::BufferCallback callback) {
        const uint32_t bank = attentionWorker_
            ? attentionWorker_->activeOperandBank : 0;
        if (attentionGenericGemmEnable_) {
            return workerCommandProcessor->programGemmMatrixGroupBankAsync(
                arrayIds, bank, matrix, elemBytes, tag, LastTickCycle,
                std::move(callback));
        }
        return array->programMatrixGroupBankAsync(
            arrayIds, bank, matrix, elemBytes, tag, std::move(callback));
    }

    bool programAttentionGemmInputAsync(
        uint32_t arrayId, const std::vector<double>& input, size_t elemBytes,
        uint64_t tag, Golem::ComputeArray::BufferCallback callback) {
        const uint32_t bank = attentionWorker_
            ? (attentionClusterEnable_ &&
                    arrayId >= attentionClusterConfig_.qkArrays
                ? attentionWorker_->clusterPvOperandBank
                : attentionWorker_->activeOperandBank)
            : 0;
        if (attentionGenericGemmEnable_) {
            return workerCommandProcessor->programGemmInputBankAsync(
                arrayId, bank, input, elemBytes, tag, LastTickCycle,
                std::move(callback));
        }
        return array->programInputBankAsync(
            arrayId, bank, input, elemBytes, tag, std::move(callback));
    }

    bool programAttentionGemmMatrixActiveAsync(
        uint32_t arrayId, const std::vector<double>& matrix,
        uint32_t activeColumns, size_t elemBytes, uint64_t tag,
        Golem::ComputeArray::BufferCallback callback) {
        const uint32_t bank = attentionWorker_
            ? (attentionClusterEnable_ &&
                    arrayId >= attentionClusterConfig_.qkArrays
                ? attentionWorker_->clusterPvOperandBank
                : attentionWorker_->activeOperandBank)
            : 0;
        if (attentionGenericGemmEnable_) {
            return workerCommandProcessor->programGemmMatrixActiveBankAsync(
                arrayId, bank, matrix, activeColumns, elemBytes, tag, LastTickCycle,
                std::move(callback));
        }
        return array->programMatrixActiveBankAsync(
            arrayId, bank, matrix, activeColumns, elemBytes, tag,
            std::move(callback));
    }

    bool programAttentionGemmMatrixGroupActiveAsync(
        const std::vector<uint32_t>& arrayIds,
        const std::vector<double>& matrix, uint32_t activeColumns,
        size_t elemBytes, uint64_t tag,
        Golem::ComputeArray::BufferCallback callback) {
        const uint32_t bank = attentionWorker_
            ? attentionWorker_->activeOperandBank : 0;
        if (attentionGenericGemmEnable_) {
            return workerCommandProcessor->programGemmMatrixGroupActiveBankAsync(
                arrayIds, bank, matrix, activeColumns, elemBytes, tag, LastTickCycle,
                std::move(callback));
        }
        return array->programMatrixGroupActiveBankAsync(
            arrayIds, bank, matrix, activeColumns, elemBytes, tag,
            std::move(callback));
    }

    bool programAttentionGemmInputActiveAsync(
        uint32_t arrayId, const std::vector<double>& input,
        uint32_t activeColumns, size_t elemBytes, uint64_t tag,
        Golem::ComputeArray::BufferCallback callback) {
        const uint32_t bank = attentionWorker_
            ? attentionWorker_->activeOperandBank : 0;
        if (attentionGenericGemmEnable_) {
            return workerCommandProcessor->programGemmInputActiveBankAsync(
                arrayId, bank, input, activeColumns, elemBytes, tag, LastTickCycle,
                std::move(callback));
        }
        return array->programInputActiveBankAsync(
            arrayId, bank, input, activeColumns, elemBytes, tag,
            std::move(callback));
    }

    bool writeAttentionGemmOutputAsync(
        uint32_t arrayId, const std::vector<double>& values, size_t elemBytes,
        uint64_t tag, Golem::ComputeArray::BufferCallback callback) {
        if (attentionGenericGemmEnable_) {
            return workerCommandProcessor->writeGemmOutputAsync(
                arrayId, values, elemBytes, tag, LastTickCycle, std::move(callback));
        }
        return array->writeOutputAsync(
            arrayId, values, elemBytes, tag, std::move(callback));
    }

    bool readAttentionGemmOutputAsync(
        uint32_t arrayId, size_t elemBytes, uint64_t tag,
        Golem::ComputeArray::BufferReadCallback callback) {
        if (attentionGenericGemmEnable_) {
            return workerCommandProcessor->readGemmOutputAsync(
                arrayId, elemBytes, tag, LastTickCycle, std::move(callback));
        }
        return array->readOutputAsync(
            arrayId, elemBytes, tag, std::move(callback));
    }

    bool programAttentionClusterKTileAsync(
        const std::vector<uint32_t>& arrayIds,
        const std::vector<double>& matrix, uint64_t tag,
        Golem::ComputeArray::BufferCallback callback) {
        if (!attentionWorker_ || !workerCommandProcessor) return false;
        return workerCommandProcessor->programGemmMatrixGroupClassBankAsync(
            arrayIds, attentionWorker_->activeOperandBank, matrix,
            sizeof(float), AttentionClusterTrafficClass::QkKMatrix, tag,
            LastTickCycle, std::move(callback));
    }

    bool programAttentionClusterQPairAsync(
        const std::vector<uint32_t>& arrayIds,
        const std::vector<double>& input, uint64_t tag,
        Golem::ComputeArray::BufferCallback callback) {
        if (!attentionWorker_ || !workerCommandProcessor) return false;
        return workerCommandProcessor->programGemmInputGroupBankAsync(
            arrayIds, attentionWorker_->activeOperandBank, input,
            sizeof(float), AttentionClusterTrafficClass::QkQPair, tag,
            LastTickCycle, std::move(callback));
    }

    bool programAttentionClusterPRowAsync(
        const std::vector<uint32_t>& arrayIds,
        const std::vector<double>& input, uint64_t tag,
        Golem::ComputeArray::BufferCallback callback) {
        if (!attentionWorker_ || !workerCommandProcessor) return false;
        return workerCommandProcessor->programGemmInputGroupBankAsync(
            arrayIds, attentionWorker_->clusterPvOperandBank, input,
            sizeof(float), AttentionClusterTrafficClass::PvPInput, tag,
            LastTickCycle, std::move(callback));
    }

    bool programAttentionSequentialInputGroupAsync(
        const std::vector<uint32_t>& arrayIds,
        const std::vector<double>& input, uint64_t tag,
        Golem::ComputeArray::BufferCallback callback) {
        if (!attentionWorker_ || !workerCommandProcessor) return false;
        return workerCommandProcessor->programGemmInputGroupBankAsync(
            arrayIds, attentionWorker_->activeOperandBank, input,
            sizeof(float), AttentionClusterTrafficClass::SequentialPvInput, tag,
            LastTickCycle, std::move(callback));
    }

    bool programAttentionSequentialInputScatterAsync(
        const std::vector<uint32_t>& arrayIds,
        const std::vector<double>& inputs,
        AttentionClusterTrafficClass trafficClass, uint64_t tag,
        Golem::ComputeArray::BufferCallback callback) {
        if (!attentionWorker_ || !workerCommandProcessor) return false;
        return workerCommandProcessor->programGemmInputScatterBankAsync(
            arrayIds, attentionWorker_->activeOperandBank, inputs,
            sizeof(float), trafficClass, tag, LastTickCycle,
            std::move(callback));
    }

    bool readAttentionClusterScoreAsync(
        uint32_t arrayId, uint64_t tag,
        Golem::ComputeArray::BufferReadCallback callback) {
        if (!workerCommandProcessor) return false;
        return workerCommandProcessor->readGemmOutputClassAsync(
            arrayId, sizeof(float), AttentionClusterTrafficClass::QkScoreOut,
            tag, LastTickCycle, std::move(callback));
    }

    bool readAttentionClusterScorePairAsync(
        const std::vector<uint32_t>& arrayIds, uint64_t tag,
        Golem::ComputeArray::BufferReadCallback callback) {
        return workerCommandProcessor &&
            workerCommandProcessor->readGemmOutputGroupClassAsync(
                arrayIds, sizeof(float),
                AttentionClusterTrafficClass::QkScoreOut, tag,
                LastTickCycle, std::move(callback));
    }

    bool readAttentionClusterPvGroupAsync(
        const std::vector<uint32_t>& arrayIds, uint64_t tag,
        Golem::ComputeArray::BufferReadCallback callback) {
        return workerCommandProcessor &&
            workerCommandProcessor->readGemmOutputGroupClassAsync(
                arrayIds, sizeof(float),
                AttentionClusterTrafficClass::PvOFinalDrain, tag,
                LastTickCycle, std::move(callback));
    }

    bool readAttentionSequentialQkGroupAsync(
        const std::vector<uint32_t>& arrayIds, uint64_t tag,
        Golem::ComputeArray::BufferReadCallback callback) {
        return workerCommandProcessor &&
            workerCommandProcessor->readGemmOutputGroupClassAsync(
                arrayIds, sizeof(float),
                AttentionClusterTrafficClass::SequentialQkScoreOut, tag,
                LastTickCycle, std::move(callback));
    }

    bool writeAttentionSequentialPvOutputGroupAsync(
        const std::vector<uint32_t>& arrayIds,
        const std::vector<double>& outputs, uint64_t tag,
        Golem::ComputeArray::BufferCallback callback) {
        return workerCommandProcessor &&
            workerCommandProcessor->writeGemmOutputGroupClassAsync(
                arrayIds, outputs, sizeof(float),
                AttentionClusterTrafficClass::SequentialPvORestore, tag,
                LastTickCycle, std::move(callback));
    }

    bool readAttentionSequentialPvOutputGroupAsync(
        const std::vector<uint32_t>& arrayIds, uint64_t tag,
        Golem::ComputeArray::BufferReadCallback callback) {
        return workerCommandProcessor &&
            workerCommandProcessor->readGemmOutputGroupClassAsync(
                arrayIds, sizeof(float),
                AttentionClusterTrafficClass::SequentialPvOOutput, tag,
                LastTickCycle, std::move(callback));
    }

    void recordAttentionClusterQkTileStart(
        uint64_t cycle, uint32_t kvTileIndex, uint32_t queryTileIndex) {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        statAttentionClusterQkTileStarts_->addData(1);
        if (state.clusterQkTileStartValid) {
            const uint64_t interval = cycle - state.clusterLastQkTileStartCycle;
            const uint32_t group = queryTileIndex / attentionClusterConfig_.groupSize;
            const bool samePhysicalTile =
                state.clusterLastQkTileStartGroup == group &&
                state.clusterLastQkTileStartKvTile == kvTileIndex;
            const uint32_t queryInGroup =
                queryTileIndex % attentionClusterConfig_.groupSize;
            if (samePhysicalTile && queryInGroup >= 2) {
                statAttentionClusterQkTileIiCycles_->addData(interval);
                if (interval > 625) {
                    statAttentionClusterQkSteadyIiOverTarget_->addData(1);
                }
            } else if (!samePhysicalTile) {
                statAttentionClusterQkBoundaryIiCycles_->addData(interval);
            }
        }
        state.clusterLastQkTileStartCycle = cycle;
        state.clusterLastQkTileStartGroup =
            queryTileIndex / attentionClusterConfig_.groupSize;
        state.clusterLastQkTileStartKvTile = kvTileIndex;
        state.clusterQkTileStartValid = true;
    }

    void recordAttentionClusterPipelineOverlap() {
        if (!attentionWorker_ || !attentionClusterEnable_) return;
        const AttentionWorkerState& state = *attentionWorker_;
        const bool qkActive = attentionClusterQkArrayActivity_.active() != 0;
        const bool pvActive = attentionClusterPvArrayActivity_.active() != 0;
        bool sfuActive = state.phase == AttentionWorkerPhase::Softmax &&
            !state.attentionSoftmaxComplete;
        for (const auto& ahead : state.clusterAhead) {
            sfuActive = sfuActive ||
                ahead.phase == AttentionClusterAheadPhase::SoftmaxRunning;
        }
        if (qkActive && sfuActive) {
            statAttentionClusterQkSfuOverlapCycles_->addData(1);
        }
        if (sfuActive && pvActive) {
            statAttentionClusterSfuPvOverlapCycles_->addData(1);
        }
        if (qkActive && pvActive) {
            statAttentionClusterQkPvOverlapCycles_->addData(1);
        }
        if (qkActive && sfuActive && pvActive) {
            statAttentionClusterThreeStageOverlapCycles_->addData(1);
        }
    }

    bool launchAttentionGemmArray(
        uint32_t arrayId, uint64_t outputMode, bool qkOperation,
        uint32_t activeColumns = 0) {
        const uint32_t bank = attentionWorker_
            ? (attentionClusterEnable_ && !qkOperation
                ? attentionWorker_->clusterPvOperandBank
                : attentionWorker_->activeOperandBank)
            : 0;
        if (attentionClusterEnable_ &&
            (!attentionCluster_ ||
             !attentionCluster_->owns(
                 arrayId, qkOperation ? AttentionArrayOwner::Qk
                                      : AttentionArrayOwner::Pv))) {
            statAttentionClusterIllegalTransitions_->addData(1);
            return false;
        }
        if (!attentionGenericGemmEnable_) {
            array->configureOutputMode(arrayId, outputMode);
            if (activeColumns != 0) {
                array->beginComputationActiveBank(arrayId, bank, activeColumns);
            } else {
                array->beginComputationBank(arrayId, bank);
            }
            if (activeColumns != 0) {
                statAttentionPvActiveKLaunches_->addData(1);
                statAttentionPvActiveKColumns_->addData(activeColumns);
            }
            return true;
        }
        const uint64_t generation = attentionWorker_
            ? attentionWorker_->generation : 0;
        auto completion = [this, generation, qkOperation](
                              uint32_t completedArrayId, uint64_t) {
            if (!attentionCallbackGenerationMatches(generation)) {
                if (attentionClusterEnable_) {
                    BusyActivityTracker& activity = qkOperation
                        ? attentionClusterQkArrayActivity_
                        : attentionClusterPvArrayActivity_;
                    if (!activity.leave(getCurrentSimCycle())) {
                        statAttentionClusterIllegalTransitions_->addData(1);
                    }
                }
                return;
            }
            handleAttentionArrayDone(completedArrayId);
        };
        const bool accepted = activeColumns != 0 ?
            workerCommandProcessor->launchGemmArrayActiveBank(
                arrayId, bank, outputMode, activeColumns, LastTickCycle,
                std::move(completion)) :
            workerCommandProcessor->launchGemmArrayBank(
                arrayId, bank, outputMode, LastTickCycle,
                std::move(completion));
        if (accepted) {
            if (attentionClusterEnable_ && qkOperation && arrayId == 0 &&
                attentionWorker_ && attentionWorker_->clusterQkWave == 0) {
                recordAttentionClusterQkTileStart(
                    LastTickCycle, attentionWorker_->kvTileIndex,
                    attentionWorker_->queryTileIndex);
            }
            if (attentionClusterEnable_) {
                BusyActivityTracker& activity = qkOperation
                    ? attentionClusterQkArrayActivity_
                    : attentionClusterPvArrayActivity_;
                if (!activity.enter(getCurrentSimCycle())) {
                    statAttentionClusterIllegalTransitions_->addData(1);
                    return false;
                }
            }
            if (qkOperation) statAttentionGenericGemmQkOps_->addData(1);
            else statAttentionGenericGemmPvOps_->addData(1);
            if (activeColumns != 0) {
                statAttentionPvActiveKLaunches_->addData(1);
                statAttentionPvActiveKColumns_->addData(activeColumns);
            }
        }
        return accepted;
    }

    void beginAttentionInterTile() {
        AttentionWorkerState& state = *attentionWorker_;
        state.interTileActive = true;
        state.interTileStartTick = getCurrentSimCycle();
        state.interTilePhaseStartTick = state.interTileStartTick;
        state.interTilePhaseTicks.fill(0);
    }

    void recordAttentionInterTilePhase(AttentionInterTilePhase phase) {
        AttentionWorkerState& state = *attentionWorker_;
        if (!state.interTileActive) return;
        const uint64_t now = getCurrentSimCycle();
        state.interTilePhaseTicks[static_cast<size_t>(phase)] +=
            now - state.interTilePhaseStartTick;
        state.interTilePhaseStartTick = now;
    }

    void finishAttentionInterTile() {
        AttentionWorkerState& state = *attentionWorker_;
        if (!state.interTileActive) return;
        const auto& ticks = state.interTilePhaseTicks;
        statAttentionWorkerInterTileTotalTicks_->addData(
            getCurrentSimCycle() - state.interTileStartTick);
        statAttentionWorkerInterTileOutputDmaTicks_->addData(
            ticks[static_cast<size_t>(AttentionInterTilePhase::OutputDma)]);
        statAttentionWorkerInterTileQueryLoadTicks_->addData(
            ticks[static_cast<size_t>(AttentionInterTilePhase::QueryLoad)]);
        statAttentionWorkerInterTileKvLoadTicks_->addData(
            ticks[static_cast<size_t>(AttentionInterTilePhase::KvLoad)]);
        statAttentionWorkerInterTileQLocalReadTicks_->addData(
            ticks[static_cast<size_t>(AttentionInterTilePhase::QLocalRead)]);
        statAttentionWorkerInterTileQkMatrixProgramTicks_->addData(
            ticks[static_cast<size_t>(AttentionInterTilePhase::QkMatrixProgram)]);
        statAttentionWorkerInterTileQkInputProgramTicks_->addData(
            ticks[static_cast<size_t>(AttentionInterTilePhase::QkInputProgram)]);
        statAttentionWorkerInterTileQkComputeReadoutTicks_->addData(
            ticks[static_cast<size_t>(AttentionInterTilePhase::QkComputeReadout)]);
        state.interTileActive = false;
    }

    void beginAttentionTilePipeline() {
        AttentionWorkerState& state = *attentionWorker_;
        const uint64_t now = getCurrentSimCycle();
        state.tilePipelineActive = true;
        state.tilePipelineStartTick = now;
        state.tilePipelinePhaseStartTick = now;
        state.tilePipelinePhase = AttentionTilePipelinePhase::KvLoad;
        state.tilePipelinePhaseTicks.fill(0);
        state.attentionKvKReleaseTick = 0;
        state.attentionKvVReleaseTick = 0;
        state.attentionKvSecondLookaheadEligibleTick = 0;
        state.attentionKvNextReadyAtRelease = false;
    }

    int32_t findAttentionKvBuffer(uint32_t ordinal, uint32_t physicalTile,
                                  int64_t queryTileIndex = -1) const {
        if (!attentionWorker_) return -1;
        const uint32_t targetQueryTile = queryTileIndex < 0
            ? attentionWorker_->queryTileIndex : static_cast<uint32_t>(queryTileIndex);
        for (uint32_t buffer = 0; buffer < attentionWorker_->kvBuffers.size(); ++buffer) {
            const AttentionKvBufferState& descriptor =
                attentionWorker_->kvBuffers[buffer];
            if (descriptor.queryTileIndex == targetQueryTile &&
                descriptor.kvTileIndex == ordinal &&
                descriptor.physicalKvTileIndex == physicalTile) {
                return static_cast<int32_t>(buffer);
            }
        }
        return -1;
    }

    int32_t findFreeAttentionKvBuffer(uint32_t avoid = UINT32_MAX) const {
        if (!attentionWorker_) return -1;
        for (uint32_t buffer = 0; buffer < attentionWorker_->kvBuffers.size(); ++buffer) {
            if (buffer == avoid) continue;
            const AttentionKvBufferState& descriptor =
                attentionWorker_->kvBuffers[buffer];
            if (descriptor.kvTileIndex == UINT32_MAX &&
                descriptor.loadsPending == 0 && !descriptor.ready) {
                return static_cast<int32_t>(buffer);
            }
        }
        return -1;
    }

    void launchAttentionKvSecondLookahead() {
        if (!attentionWorker_ || !attentionKvDoubleBuffer_ ||
            !attentionKvSecondLookahead_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (!state.tilePipelineActive ||
            state.attentionKvSecondLookaheadEligibleTick != 0 ||
            state.attentionKvKReleaseTick == 0 ||
            state.attentionKvVReleaseTick == 0 ||
            state.kvTileIndex + 2 >= attentionKvTileCountForQueryTile(state)) {
            return;
        }
        const uint32_t nextOrdinal = state.kvTileIndex + 1;
        const uint32_t nextTile = attentionPhysicalKvTileForIndex(state, nextOrdinal);
        const uint32_t targetQueryTile = attentionKvGroupOwnerQueryTile(state);
        const int32_t nextBuffer = findAttentionKvBuffer(
            nextOrdinal, nextTile, targetQueryTile);
        if (nextBuffer < 0 || !state.kvBuffers[nextBuffer].ready) return;
        const int32_t freeBuffer = findFreeAttentionKvBuffer(state.activeKvBuffer);
        const uint32_t targetBuffer = freeBuffer >= 0
            ? static_cast<uint32_t>(freeBuffer) : state.activeKvBuffer;
        if (targetBuffer == state.activeKvBuffer &&
            attentionHasNextGroupedQuery(state)) {
            return;
        }
        if (targetBuffer == state.activeKvBuffer) {
            state.kvBuffers[targetBuffer] = {};
        }
        const uint64_t releaseTick = std::max(
            state.attentionKvKReleaseTick, state.attentionKvVReleaseTick);
        state.attentionKvSecondLookaheadEligibleTick = getCurrentSimCycle();
        state.attentionKvNextReadyAtRelease =
            state.kvBuffers[nextBuffer].readyTick <= releaseTick;
        launchAttentionKvPrefetch(
            state.kvTileIndex + 2, targetBuffer, true, targetQueryTile);
    }

    void recordAttentionKvOperandRelease(bool keyOperand) {
        if (!attentionWorker_ || !attentionKvDoubleBuffer_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (!state.tilePipelineActive) return;
        uint64_t& releaseTick = keyOperand
            ? state.attentionKvKReleaseTick : state.attentionKvVReleaseTick;
        if (releaseTick != 0) return;
        releaseTick = getCurrentSimCycle();
        (keyOperand ? statAttentionKvKReleaseTicks_ : statAttentionKvVReleaseTicks_)
            ->addData(releaseTick - state.tilePipelineStartTick);
        launchAttentionKvSecondLookahead();
    }

    void transitionAttentionTilePipeline(AttentionTilePipelinePhase phase) {
        AttentionWorkerState& state = *attentionWorker_;
        if (!state.tilePipelineActive || state.tilePipelinePhase == phase) return;
        const uint64_t now = getCurrentSimCycle();
        state.tilePipelinePhaseTicks[static_cast<size_t>(state.tilePipelinePhase)] +=
            now - state.tilePipelinePhaseStartTick;
        state.tilePipelinePhaseStartTick = now;
        state.tilePipelinePhase = phase;
    }

    void finishAttentionTilePipeline() {
        AttentionWorkerState& state = *attentionWorker_;
        if (!state.tilePipelineActive) return;
        const uint64_t now = getCurrentSimCycle();
        if (state.attentionKvSecondLookaheadEligibleTick != 0) {
            statAttentionKvSecondLookaheadCandidates_->addData(1);
            statAttentionKvSecondLookaheadLeadTicks_->addData(
                now - state.attentionKvSecondLookaheadEligibleTick);
            if (state.attentionKvNextReadyAtRelease) {
                statAttentionKvNextReadyAtReleaseTiles_->addData(1);
            }
        }
        state.tilePipelinePhaseTicks[static_cast<size_t>(state.tilePipelinePhase)] +=
            now - state.tilePipelinePhaseStartTick;
        statAttentionWorkerTileTotalTicks_->addData(now - state.tilePipelineStartTick);
        const auto& ticks = state.tilePipelinePhaseTicks;
        statAttentionWorkerTileKvLoadTicks_->addData(
            ticks[static_cast<size_t>(AttentionTilePipelinePhase::KvLoad)]);
        statAttentionWorkerTileQLocalReadTicks_->addData(
            ticks[static_cast<size_t>(AttentionTilePipelinePhase::QLocalRead)]);
        statAttentionWorkerTileQkMatrixProgramTicks_->addData(
            ticks[static_cast<size_t>(AttentionTilePipelinePhase::QkMatrixProgram)]);
        statAttentionWorkerTileQkInputProgramTicks_->addData(
            ticks[static_cast<size_t>(AttentionTilePipelinePhase::QkInputProgram)]);
        statAttentionWorkerTileQkComputeReadoutTicks_->addData(
            ticks[static_cast<size_t>(AttentionTilePipelinePhase::QkComputeReadout)]);
        statAttentionWorkerTileSoftmaxTicks_->addData(
            ticks[static_cast<size_t>(AttentionTilePipelinePhase::Softmax)]);
        statAttentionWorkerTilePvMatrixProgramTicks_->addData(
            ticks[static_cast<size_t>(AttentionTilePipelinePhase::PvMatrixProgram)]);
        statAttentionWorkerTilePvInputProgramTicks_->addData(
            ticks[static_cast<size_t>(AttentionTilePipelinePhase::PvInputProgram)]);
        statAttentionWorkerTilePvRestoreOutputTicks_->addData(
            ticks[static_cast<size_t>(AttentionTilePipelinePhase::PvRestoreOutput)]);
        statAttentionWorkerTilePvComputeTicks_->addData(
            ticks[static_cast<size_t>(AttentionTilePipelinePhase::PvCompute)]);
        statAttentionWorkerTilePvOutputReadwriteTicks_->addData(
            ticks[static_cast<size_t>(AttentionTilePipelinePhase::PvOutputReadwrite)]);
        state.tilePipelineActive = false;
    }

    bool attentionCausal(const AttentionWorkerState& state) const {
        return (state.dispatch.flags & GOLEM_ATTENTION_FLAG_CAUSAL) != 0;
    }

    uint32_t attentionQueryRowsForTile(
        const AttentionWorkerState& state, uint32_t queryTileIndex) const {
        const uint32_t begin = queryTileIndex * state.dispatch.queryTileRows;
        return std::min(state.dispatch.queryTileRows,
                        state.dispatch.expectedRows - begin);
    }

    uint32_t attentionQueryRows(const AttentionWorkerState& state) const {
        return attentionQueryRowsForTile(state, state.queryTileIndex);
    }

    uint32_t attentionKvRowsForTile(const AttentionWorkerState& state,
                                     uint32_t physicalKvTileIndex) const {
        const uint32_t begin = physicalKvTileIndex * state.dispatch.kvTileRows;
        return std::min(state.dispatch.kvTileRows,
                        state.dispatch.expectedCols - begin);
    }

    uint32_t attentionKeyCols(const AttentionWorkerState& state) const {
        return attentionKvRowsForTile(state, state.physicalKvTileIndex);
    }

    uint32_t attentionPvActiveColumns(const AttentionWorkerState& state) const {
        const uint32_t keyCols = attentionKeyCols(state);
        return attentionPvActiveK_ && keyCols < arrayInputSize ? keyCols : 0;
    }

    uint32_t attentionQueryTileCount(const AttentionWorkerState& state) const {
        return (state.dispatch.expectedRows + state.dispatch.queryTileRows - 1) /
            state.dispatch.queryTileRows;
    }

    bool attentionKvPairReuseActive(const AttentionWorkerState& state) const {
        return attentionKvPairReuse_ && attentionKvDoubleBuffer_ &&
            attentionStreamKv(state) && !attentionCausal(state);
    }

    uint32_t attentionKvQueryGroupSize(
        const AttentionWorkerState& state) const {
        return attentionKvPairReuseActive(state)
            ? attentionKvQueryGroupSize_ : 1u;
    }

    AttentionClusterTag attentionClusterTag(
        const AttentionWorkerState& state, uint32_t sequence = 0) const {
        AttentionClusterTag tag;
        tag.generation = state.generation;
        tag.jobId = state.dispatch.jobId;
        tag.group = state.queryTileIndex / attentionClusterConfig_.groupSize;
        tag.queryContext = state.queryTileIndex % attentionClusterConfig_.groupSize;
        tag.queryTileIndex = state.queryTileIndex;
        tag.kvTileIndex = state.kvTileIndex;
        tag.sequence = sequence;
        return tag;
    }

    bool attentionCallbackGenerationMatches(uint64_t generation) {
        if (attentionWorker_ && attentionWorker_->generation == generation) {
            return true;
        }
        if (attentionClusterEnable_) {
            statAttentionClusterStaleCallbacks_->addData(1);
        }
        return false;
    }

    void recordAttentionClusterContextHighWater() {
        if (!attentionCluster_) return;
        statAttentionClusterContextHighWater_->addData(
            attentionCluster_->liveContexts());
        statAttentionClusterScoreSlotHighWater_->addData(
            attentionCluster_->liveContexts(AttentionClusterContextKind::Score));
    }

    bool reserveAttentionClusterQkAndScore() {
        if (!attentionWorker_ || !attentionCluster_) return false;
        AttentionWorkerState& state = *attentionWorker_;
        state.clusterTileTag = attentionClusterTag(state);
        const int qk = attentionCluster_->reserve(
            AttentionClusterContextKind::Qk, state.clusterTileTag,
            state.activeOperandBank);
        if (qk < 0) return false;
        const int score = attentionCluster_->reserve(
            AttentionClusterContextKind::Score, state.clusterTileTag,
            state.activeOperandBank);
        if (score < 0) {
            attentionCluster_->release(
                AttentionClusterContextKind::Qk, qk, state.clusterTileTag);
            statAttentionClusterScoreSlotFullStalls_->addData(1);
            return false;
        }
        if (sfu == nullptr || !sfu->reserveAttentionScoreSlot(
                static_cast<uint32_t>(score), state.clusterTileTag,
                static_cast<size_t>(attentionQueryRows(state)) *
                    attentionKeyCols(state))) {
            attentionCluster_->release(
                AttentionClusterContextKind::Score, score, state.clusterTileTag);
            attentionCluster_->release(
                AttentionClusterContextKind::Qk, qk, state.clusterTileTag);
            statAttentionClusterScoreSlotFullStalls_->addData(1);
            return false;
        }
        const bool useLookahead = attentionClusterQkMatrixLookahead_ &&
            state.clusterQkMatrixAhead.phase ==
                AttentionClusterQkMatrixAheadPhase::Ready &&
            state.clusterQkMatrixAhead.generation == state.generation &&
            state.clusterQkMatrixAhead.kvTileIndex == state.kvTileIndex &&
            state.clusterQkMatrixAhead.physicalKvTileIndex == state.physicalKvTileIndex &&
            state.clusterQkMatrixAhead.operandBank == state.activeOperandBank &&
            state.clusterQkMatrixAhead.tag == state.clusterTileTag;
        uint32_t acquired = 0;
        for (uint32_t arrayId = 0;
             arrayId < attentionClusterConfig_.qkArrays; ++arrayId) {
            const bool bankOk = useLookahead
                ? attentionCluster_->bankMatches(
                    arrayId, state.activeOperandBank, AttentionArrayOwner::Qk,
                    state.clusterTileTag)
                : attentionCluster_->acquireBank(
                    arrayId, state.activeOperandBank, AttentionArrayOwner::Qk,
                    state.clusterTileTag);
            if (!bankOk) {
                for (uint32_t releaseId = 0; releaseId < acquired; ++releaseId) {
                    if (!useLookahead) {
                        attentionCluster_->releaseBank(
                            releaseId, state.activeOperandBank,
                            AttentionArrayOwner::Qk, state.clusterTileTag);
                    }
                }
                attentionCluster_->release(
                    AttentionClusterContextKind::Score, score, state.clusterTileTag);
                sfu->releaseAttentionScoreSlot(
                    static_cast<uint32_t>(score), state.clusterTileTag);
                attentionCluster_->release(
                    AttentionClusterContextKind::Qk, qk, state.clusterTileTag);
                return false;
            }
            ++acquired;
        }
        if (useLookahead) {
            state.clusterQkReductionSlice = attentionClusterHeadHalves(state);
            state.clusterQkMatrixAhead = {};
            statAttentionClusterQkMatrixLookaheadHits_->addData(1);
        }
        state.clusterQkContext = qk;
        state.clusterScoreContext = score;
        statAttentionClusterContextsIssued_->addData(2);
        statAttentionClusterScoreSlotReservations_->addData(1);
        recordAttentionClusterContextHighWater();
        return true;
    }

    bool releaseAttentionClusterQk() {
        if (!attentionWorker_ || !attentionCluster_ ||
            attentionWorker_->clusterQkContext < 0) return false;
        AttentionWorkerState& state = *attentionWorker_;
        bool ok = true;
        for (uint32_t arrayId = 0;
             arrayId < attentionClusterConfig_.qkArrays; ++arrayId) {
            ok = attentionCluster_->releaseBank(
                arrayId, state.activeOperandBank, AttentionArrayOwner::Qk,
                state.clusterTileTag) && ok;
        }
        ok = attentionCluster_->release(
            AttentionClusterContextKind::Qk,
            static_cast<uint32_t>(state.clusterQkContext),
            state.clusterTileTag) && ok;
        state.clusterQkContext = -1;
        if (ok) statAttentionClusterContextsCompleted_->addData(1);
        return ok;
    }

    bool releaseAttentionClusterScore() {
        if (!attentionWorker_ || !attentionCluster_ ||
            attentionWorker_->clusterScoreContext < 0) return false;
        AttentionWorkerState& state = *attentionWorker_;
        const bool fifoOk = sfu != nullptr && sfu->releaseAttentionScoreSlot(
            static_cast<uint32_t>(state.clusterScoreContext),
            state.clusterTileTag);
        const bool ok = fifoOk && attentionCluster_->release(
            AttentionClusterContextKind::Score,
            static_cast<uint32_t>(state.clusterScoreContext),
            state.clusterTileTag);
        state.clusterScoreContext = -1;
        if (ok) {
            statAttentionClusterContextsCompleted_->addData(1);
            statAttentionClusterScoreSlotReleases_->addData(1);
        }
        return ok;
    }

    bool releaseAttentionClusterP() {
        if (!attentionWorker_ || sfu == nullptr ||
            attentionWorker_->clusterPContext < 0) return false;
        AttentionWorkerState& state = *attentionWorker_;
        const bool ok = sfu->releaseAttentionPSlot(
            static_cast<uint32_t>(state.clusterPContext),
            state.clusterTileTag);
        state.clusterPContext = -1;
        return ok;
    }

    bool reserveAttentionClusterPv() {
        if (!attentionWorker_ || !attentionCluster_) return false;
        AttentionWorkerState& state = *attentionWorker_;
        const int slot = attentionCluster_->reserve(
            AttentionClusterContextKind::Pv, state.clusterTileTag,
            state.clusterPvOperandBank);
        if (slot < 0) return false;
        AttentionClusterTag bankTag = state.clusterTileTag;
        bankTag.queryContext = 0;
        bankTag.queryTileIndex = attentionKvGroupOwnerQueryTile(state);
        const bool useLookahead = attentionClusterPvMatrixLookahead_ &&
            state.clusterPvMatrixAhead.phase ==
                AttentionClusterQkMatrixAheadPhase::Ready &&
            attentionClusterPvMatrixAheadMatchesCurrent();
        if (useLookahead) {
            for (uint32_t arrayId = attentionClusterConfig_.qkArrays;
                 arrayId < attentionClusterConfig_.arrays; ++arrayId) {
                if (!attentionCluster_->bankMatches(
                        arrayId, state.clusterPvOperandBank,
                        AttentionArrayOwner::Pv, bankTag)) {
                    attentionCluster_->release(
                        AttentionClusterContextKind::Pv, slot,
                        state.clusterTileTag);
                    return false;
                }
            }
            state.clusterPvBankLeaseHeld = true;
            state.clusterPvBankLeaseTag = bankTag;
            state.clusterPvMatrixResident = true;
            state.clusterPvMatrixKvTile = state.kvTileIndex;
            state.clusterPvMatrixGroupOwner =
                attentionKvGroupOwnerQueryTile(state);
            state.clusterPvMatrixAhead = {};
            statAttentionClusterPvMatrixLookaheadHits_->addData(1);
        } else if (attentionClusterPvMatrixLookahead_ &&
                   attentionClusterPvMatrixAheadMatchesCurrent()) {
            state.clusterPvMatrixAheadPromotionWaiting = true;
            attentionCluster_->release(
                AttentionClusterContextKind::Pv, slot, state.clusterTileTag);
            return false;
        } else if (state.clusterPvBankLeaseHeld) {
            if (!(state.clusterPvBankLeaseTag == bankTag)) {
                attentionCluster_->release(
                    AttentionClusterContextKind::Pv, slot, state.clusterTileTag);
                return false;
            }
        } else {
            uint32_t acquired = 0;
            for (uint32_t arrayId = attentionClusterConfig_.qkArrays;
                 arrayId < attentionClusterConfig_.arrays; ++arrayId) {
                if (!attentionCluster_->acquireBank(
                        arrayId, state.clusterPvOperandBank,
                        AttentionArrayOwner::Pv, bankTag)) {
                    for (uint32_t offset = 0; offset < acquired; ++offset) {
                        attentionCluster_->releaseBank(
                            attentionClusterConfig_.qkArrays + offset,
                            state.clusterPvOperandBank,
                            AttentionArrayOwner::Pv, bankTag);
                    }
                    attentionCluster_->release(
                        AttentionClusterContextKind::Pv, slot,
                        state.clusterTileTag);
                    return false;
                }
                ++acquired;
            }
            state.clusterPvBankLeaseHeld = true;
            state.clusterPvBankLeaseTag = bankTag;
        }
        state.clusterPvContext = slot;
        statAttentionClusterContextsIssued_->addData(1);
        recordAttentionClusterContextHighWater();
        return true;
    }

    bool releaseAttentionClusterPv() {
        if (!attentionWorker_ || !attentionCluster_ ||
            attentionWorker_->clusterPvContext < 0) return false;
        AttentionWorkerState& state = *attentionWorker_;
        bool ok = true;
        if (!attentionHasNextGroupedQuery(state)) {
            if (!state.clusterPvBankLeaseHeld) return false;
            for (uint32_t arrayId = attentionClusterConfig_.qkArrays;
                 arrayId < attentionClusterConfig_.arrays; ++arrayId) {
                ok = attentionCluster_->releaseBank(
                    arrayId, state.clusterPvOperandBank, AttentionArrayOwner::Pv,
                    state.clusterPvBankLeaseTag) && ok;
            }
            state.clusterPvBankLeaseHeld = false;
            state.clusterPvBankLeaseTag = {};
        }
        ok = attentionCluster_->release(
            AttentionClusterContextKind::Pv,
            static_cast<uint32_t>(state.clusterPvContext),
            state.clusterTileTag) && ok;
        state.clusterPvContext = -1;
        ok = releaseAttentionClusterP() && ok;
        if (ok) statAttentionClusterContextsCompleted_->addData(1);
        return ok;
    }

    bool ensureAttentionClusterOContext() {
        if (!attentionWorker_ || !attentionCluster_) return false;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t queryContext =
            state.queryTileIndex % attentionClusterConfig_.groupSize;
        if (state.clusterOContextSlots[queryContext] >= 0) return true;
        AttentionClusterTag tag = attentionClusterTag(state);
        tag.kvTileIndex = 0;
        const int slot = attentionCluster_->reserve(
            AttentionClusterContextKind::O, tag, state.activeOperandBank);
        if (slot < 0) {
            output->output(
                "Attention O metadata reserve rejected core=%" PRIu64
                " generation=%" PRIu64 " query=%u kv_tiles=%u\n",
                coreID, state.generation, state.queryTileIndex,
                attentionKvTileCountForQueryTile(state));
            return false;
        }
        if (!attentionOAccumulator_.reserve(
                static_cast<uint32_t>(slot), tag,
                attentionKvTileCountForQueryTile(state))) {
            output->output(
                "Attention O accumulator reserve rejected core=%" PRIu64
                " slot=%d generation=%" PRIu64
                " query=%u kv_tiles=%u occupied=%u\n",
                coreID, slot, state.generation, state.queryTileIndex,
                attentionKvTileCountForQueryTile(state),
                attentionOAccumulator_.occupied());
            attentionCluster_->release(
                AttentionClusterContextKind::O,
                static_cast<uint32_t>(slot), tag);
            return false;
        }
        state.clusterOContextSlots[queryContext] = slot;
        state.clusterOTags[queryContext] = tag;
        statAttentionClusterContextsIssued_->addData(1);
        statAttentionClusterOContextReservations_->addData(1);
        statAttentionClusterOContextHighWater_->addData(
            attentionOAccumulator_.occupied());
        recordAttentionClusterContextHighWater();
        return true;
    }

    bool releaseAttentionClusterOContext() {
        if (!attentionWorker_ || !attentionCluster_) return false;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t queryContext =
            state.queryTileIndex % attentionClusterConfig_.groupSize;
        const int32_t slot = state.clusterOContextSlots[queryContext];
        if (slot < 0) return false;
        const bool accumulatorOk = attentionOAccumulator_.release(
            static_cast<uint32_t>(slot), state.clusterOTags[queryContext]);
        const bool ok = accumulatorOk && attentionCluster_->release(
            AttentionClusterContextKind::O, static_cast<uint32_t>(slot),
            state.clusterOTags[queryContext]);
        state.clusterOContextSlots[queryContext] = -1;
        if (ok) {
            statAttentionClusterContextsCompleted_->addData(1);
            statAttentionClusterOContextReleases_->addData(1);
        }
        return ok;
    }

    uint32_t attentionKvGroupOwnerQueryTile(
        const AttentionWorkerState& state) const {
        const uint32_t groupSize = attentionKvQueryGroupSize(state);
        return state.queryTileIndex - state.queryTileIndex % groupSize;
    }

    uint32_t attentionKvGroupEndQueryTile(
        const AttentionWorkerState& state) const {
        return std::min(
            attentionKvGroupOwnerQueryTile(state) +
                attentionKvQueryGroupSize(state),
            attentionQueryTileCount(state));
    }

    bool attentionHasNextGroupedQuery(const AttentionWorkerState& state) const {
        return attentionKvPairReuseActive(state) &&
            state.queryTileIndex + 1 < attentionKvGroupEndQueryTile(state);
    }

    bool attentionAheadOperandsMatch(
        const AttentionWorkerState& state,
        const AttentionAheadOperandContext& ahead) const {
        return ahead.phase != AttentionAheadOperandPhase::Idle &&
            ahead.generation == state.generation &&
            ahead.queryTileIndex == state.queryTileIndex + (ahead.commitWaiting ? 0u : 1u) &&
            ahead.kvTileIndex == state.kvTileIndex &&
            ahead.physicalKvTileIndex == state.physicalKvTileIndex;
    }

    void promoteAttentionAheadOperands() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        AttentionAheadOperandContext& ahead = state.aheadOperands;
        if (ahead.phase != AttentionAheadOperandPhase::Ready ||
            ahead.generation != state.generation ||
            ahead.queryTileIndex != state.queryTileIndex ||
            ahead.kvTileIndex != state.kvTileIndex ||
            ahead.physicalKvTileIndex != state.physicalKvTileIndex || ahead.operandBank == state.activeOperandBank) {
            finishAttentionWorker(false);
            return;
        }
        if (ahead.commitWaiting) {
            statAttentionCrossTileOperandWaitTicks_->addData(
                getCurrentSimCycle() - ahead.waitStartTick);
        } else {
            statAttentionCrossTileOperandReadyHits_->addData(1);
        }
        state.activeOperandBank = ahead.operandBank;
        state.aheadOperands = {};
        statAttentionCrossTileOperandPromotions_->addData(1);

        beginAttentionTilePipeline();
        const DmaConsumerMetadata consumerProgress = attentionDmaConsumerMetadata(
            state, state.queryTileIndex, state.kvTileIndex, DmaOperand::Unknown);
        globalMem->dma_update_consumer_progress(
            attentionKvHostAddr(state, state.dispatch.kAddr),
            attentionKvHostAddr(state, state.dispatch.vAddr), consumerProgress);
        recordAttentionInterTilePhase(AttentionInterTilePhase::KvLoad);
        invalidateAttentionPvInputResidency();
        if (attentionSequential64Enable_) state.qkReductionSlice = 0;
        if (!attentionPvVTileGroupRetention_ ||
            !attentionVTileMatchesCurrentGroup(state)) {
            invalidateAttentionVTileStaging();
        }
        state.vTileBufferWaiting = false;
        state.vTileBufferBypassWait = false;
        state.vTileBufferCurrentWaitTicks = 0;
        state.phaseSliceIndex = 0;
        state.index = 0;
        state.attentionQkInputProgrammingComplete = true;
        transitionAttentionTilePipeline(
            AttentionTilePipelinePhase::QkComputeReadout);
        startAttentionQkCompute(attentionKvSubtileRows(state));
    }

    void pumpAttentionAheadOperands() {
        if (!attentionWorker_ || !attentionCrossTileOperandPipeline_) return;
        AttentionWorkerState& state = *attentionWorker_;
        AttentionAheadOperandContext& ahead = state.aheadOperands;
        if (ahead.phase == AttentionAheadOperandPhase::Idle ||
            ahead.phase == AttentionAheadOperandPhase::Ready ||
            ahead.requestInFlight) return;
        if (!attentionAheadOperandsMatch(state, ahead)) {
            statAttentionCrossTileOperandTagMismatches_->addData(1);
            finishAttentionWorker(false);
            return;
        }

        const uint64_t generation = ahead.generation;
        const uint32_t queryTileIndex = ahead.queryTileIndex;
        const uint32_t kvTileIndex = ahead.kvTileIndex;
        const uint32_t physicalKvTileIndex = ahead.physicalKvTileIndex;
        const uint32_t bank = ahead.operandBank;
        if (ahead.phase == AttentionAheadOperandPhase::QueryRead) {
            if (ahead.queryOffset == ahead.queryBytes.size()) {
                ahead.phase = AttentionAheadOperandPhase::MatrixProgram;
                pumpAttentionAheadOperands();
                return;
            }
            const size_t chunk = std::min(
                ahead.queryBytes.size() - ahead.queryOffset,
                globalMem->localMaxRequestBytes());
            const size_t offset = ahead.queryOffset;
            const uint64_t tag = attentionTransferTag();
            const bool accepted = globalMem->localReadAsync(
                ahead.qLocal + offset, chunk, LocalMemoryClient::RoCC, tag,
                [this, generation, queryTileIndex, kvTileIndex, physicalKvTileIndex, bank,
                 tag, offset, chunk](bool ok, uint64_t callbackTag,
                                    const std::vector<uint8_t>& bytes) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    AttentionAheadOperandContext& callbackAhead =
                        attentionWorker_->aheadOperands;
                    if (!ok || callbackTag != tag || bytes.size() != chunk ||
                        callbackAhead.phase != AttentionAheadOperandPhase::QueryRead ||
                        callbackAhead.queryTileIndex != queryTileIndex ||
                        callbackAhead.kvTileIndex != kvTileIndex ||
                        callbackAhead.physicalKvTileIndex != physicalKvTileIndex ||
                        callbackAhead.operandBank != bank ||
                        callbackAhead.queryOffset != offset) {
                        statAttentionCrossTileOperandTagMismatches_->addData(1);
                        finishAttentionWorker(false);
                        return;
                    }
                    std::copy(bytes.begin(), bytes.end(),
                        callbackAhead.queryBytes.begin() + offset);
                    callbackAhead.queryOffset += chunk;
                    callbackAhead.requestInFlight = false;
                    pumpAttentionAheadOperands();
                });
            if (accepted) {
                ahead.transferTag = tag;
                ahead.requestInFlight = true;
                if (offset == 0) statAttentionCrossTileOperandLaunches_->addData(1);
            }
            return;
        }
        if (ahead.phase == AttentionAheadOperandPhase::MatrixProgram) {
            std::vector<double> matrix(
                static_cast<size_t>(arrayOutputSize) * arrayInputSize, 0.0);
            const std::vector<double> query =
                attentionBytesToDoubles(ahead.queryBytes);
            std::copy(query.begin(), query.end(), matrix.begin());
            std::vector<uint32_t> arrayIds(attentionKvSubtileRows(state));
            std::iota(arrayIds.begin(), arrayIds.end(), 0);
            const uint64_t tag = attentionTransferTag();
            const bool accepted = workerCommandProcessor->programGemmMatrixGroupBankAsync(
                arrayIds, bank, matrix, sizeof(float), tag, LastTickCycle,
                [this, generation, queryTileIndex, kvTileIndex, physicalKvTileIndex, bank, tag]
                (bool ok, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    AttentionAheadOperandContext& callbackAhead =
                        attentionWorker_->aheadOperands;
                    if (!ok || callbackTag != tag ||
                        callbackAhead.phase != AttentionAheadOperandPhase::MatrixProgram ||
                        callbackAhead.queryTileIndex != queryTileIndex ||
                        callbackAhead.kvTileIndex != kvTileIndex ||
                        callbackAhead.physicalKvTileIndex != physicalKvTileIndex ||
                        callbackAhead.operandBank != bank) {
                        statAttentionCrossTileOperandTagMismatches_->addData(1);
                        finishAttentionWorker(false);
                        return;
                    }
                    callbackAhead.requestInFlight = false;
                    callbackAhead.phase = AttentionAheadOperandPhase::InputRead;
                    statAttentionCrossTileOperandMatrixPrograms_->addData(1);
                    pumpAttentionAheadOperands();
                });
            if (accepted) {
                ahead.transferTag = tag;
                ahead.requestInFlight = true;
            }
            return;
        }
        if (ahead.phase == AttentionAheadOperandPhase::InputRead) {
            if (ahead.inputIndex == attentionKvSubtileRows(state)) {
                ahead.phase = AttentionAheadOperandPhase::Ready;
                if (ahead.commitWaiting) promoteAttentionAheadOperands();
                return;
            }
            const uint32_t arrayId = ahead.inputIndex;
            const uint32_t key = attentionKvLocalKey(state, arrayId);
            const size_t bytesExpected =
                static_cast<size_t>(state.dispatch.headDim) * sizeof(float);
            const uint64_t tag = attentionTransferTag();
            const bool accepted = globalMem->localReadAsync(
                ahead.kLocal + static_cast<uint64_t>(key) *
                    state.dispatch.headDim * sizeof(float),
                bytesExpected, LocalMemoryClient::RoCC, tag,
                [this, generation, queryTileIndex, kvTileIndex, physicalKvTileIndex, bank,
                 arrayId, tag, bytesExpected]
                (bool ok, uint64_t callbackTag, const std::vector<uint8_t>& bytes) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    AttentionAheadOperandContext& callbackAhead =
                        attentionWorker_->aheadOperands;
                    if (!ok || callbackTag != tag || bytes.size() != bytesExpected ||
                        callbackAhead.phase != AttentionAheadOperandPhase::InputRead ||
                        callbackAhead.queryTileIndex != queryTileIndex ||
                        callbackAhead.kvTileIndex != kvTileIndex ||
                        callbackAhead.physicalKvTileIndex != physicalKvTileIndex ||
                        callbackAhead.operandBank != bank ||
                        callbackAhead.inputIndex != arrayId) {
                        statAttentionCrossTileOperandTagMismatches_->addData(1);
                        finishAttentionWorker(false);
                        return;
                    }
                    callbackAhead.inputBytes = bytes;
                    callbackAhead.requestInFlight = false;
                    callbackAhead.phase = AttentionAheadOperandPhase::InputProgram;
                    pumpAttentionAheadOperands();
                });
            if (accepted) {
                ahead.transferTag = tag;
                ahead.requestInFlight = true;
            }
            return;
        }
        if (ahead.phase == AttentionAheadOperandPhase::InputProgram) {
            const uint32_t arrayId = ahead.inputIndex;
            const std::vector<double> input =
                attentionBytesToDoubles(ahead.inputBytes);
            const uint64_t tag = attentionTransferTag();
            const bool accepted = workerCommandProcessor->programGemmInputBankAsync(
                arrayId, bank, input, sizeof(float), tag, LastTickCycle,
                [this, generation, queryTileIndex, kvTileIndex, physicalKvTileIndex, bank,
                 arrayId, tag](bool ok, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    AttentionAheadOperandContext& callbackAhead =
                        attentionWorker_->aheadOperands;
                    if (!ok || callbackTag != tag ||
                        callbackAhead.phase != AttentionAheadOperandPhase::InputProgram ||
                        callbackAhead.queryTileIndex != queryTileIndex ||
                        callbackAhead.kvTileIndex != kvTileIndex ||
                        callbackAhead.physicalKvTileIndex != physicalKvTileIndex ||
                        callbackAhead.operandBank != bank ||
                        callbackAhead.inputIndex != arrayId) {
                        statAttentionCrossTileOperandTagMismatches_->addData(1);
                        finishAttentionWorker(false);
                        return;
                    }
                    callbackAhead.inputBytes.clear();
                    callbackAhead.inputIndex += 1;
                    callbackAhead.requestInFlight = false;
                    callbackAhead.phase = AttentionAheadOperandPhase::InputRead;
                    statAttentionCrossTileOperandInputRows_->addData(1);
                    pumpAttentionAheadOperands();
                });
            if (accepted) {
                ahead.transferTag = tag;
                ahead.requestInFlight = true;
            }
        }
    }

    void maybeStartAttentionAheadOperands() {
        if (!attentionWorker_ || !attentionCrossTileOperandPipeline_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.aheadOperands.phase != AttentionAheadOperandPhase::Idle ||
            state.kvTileIndex == 0 || !attentionHasNextGroupedQuery(state)) return;
        statAttentionCrossTileOperandCandidates_->addData(1);
        AttentionAheadOperandContext ahead;
        ahead.phase = AttentionAheadOperandPhase::QueryRead;
        ahead.generation = state.generation;
        ahead.queryTileIndex = state.queryTileIndex + 1;
        ahead.kvTileIndex = state.kvTileIndex;
        ahead.physicalKvTileIndex = state.physicalKvTileIndex;
        ahead.operandBank = (state.activeOperandBank + 1) % attentionOperandContextBanks_;
        const uint32_t querySlot =
            ahead.queryTileIndex % attentionKvQueryGroupSize(state);
        if (querySlot >= state.qLocalBuffers.size() ||
            ahead.operandBank == state.activeOperandBank) {
            finishAttentionWorker(false);
            return;
        }
        ahead.qLocal = state.qLocalBuffers[querySlot];
        ahead.kLocal = state.kLocal;
        const uint32_t queryRows = std::min<uint32_t>(
            state.dispatch.queryTileRows,
            state.dispatch.expectedRows -
                ahead.queryTileIndex * state.dispatch.queryTileRows);
        ahead.queryBytes.resize(
            static_cast<size_t>(queryRows) * state.dispatch.headDim * sizeof(float));
        ahead.startTick = getCurrentSimCycle();
        state.aheadOperands = std::move(ahead);
        pumpAttentionAheadOperands();
    }

    AttentionClusterAheadContext* attentionClusterAhead(
        uint32_t queryTileIndex, uint32_t kvTileIndex) {
        if (!attentionWorker_) return nullptr;
        AttentionClusterAheadContext& ahead =
            attentionWorker_->clusterAhead[
                queryTileIndex % attentionClusterConfig_.groupSize];
        return ahead.phase != AttentionClusterAheadPhase::Idle &&
            ahead.generation == attentionWorker_->generation &&
            ahead.queryTileIndex == queryTileIndex &&
            ahead.kvTileIndex == kvTileIndex ? &ahead : nullptr;
    }

    bool reserveAttentionClusterAheadQk(AttentionClusterAheadContext& ahead) {
        if (!attentionWorker_ || !attentionCluster_ || sfu == nullptr) return false;
        const int qk = attentionCluster_->reserve(
            AttentionClusterContextKind::Qk, ahead.tag, ahead.operandBank);
        if (qk < 0) return false;
        const int score = attentionCluster_->reserve(
            AttentionClusterContextKind::Score, ahead.tag, ahead.operandBank);
        if (score < 0 || !sfu->reserveAttentionScoreSlot(
                static_cast<uint32_t>(score), ahead.tag,
                static_cast<size_t>(attentionQueryRows(*attentionWorker_)) *
                    attentionKeyCols(*attentionWorker_))) {
            if (score >= 0) attentionCluster_->release(
                AttentionClusterContextKind::Score,
                static_cast<uint32_t>(score), ahead.tag);
            attentionCluster_->release(
                AttentionClusterContextKind::Qk,
                static_cast<uint32_t>(qk), ahead.tag);
            statAttentionClusterScoreSlotFullStalls_->addData(1);
            return false;
        }
        uint32_t acquired = 0;
        for (uint32_t arrayId = 0;
             arrayId < attentionClusterConfig_.qkArrays; ++arrayId) {
            if (!attentionCluster_->acquireBank(
                    arrayId, ahead.operandBank, AttentionArrayOwner::Qk,
                    ahead.tag)) {
                for (uint32_t releaseId = 0; releaseId < acquired; ++releaseId) {
                    attentionCluster_->releaseBank(
                        releaseId, ahead.operandBank, AttentionArrayOwner::Qk,
                        ahead.tag);
                }
                sfu->releaseAttentionScoreSlot(
                    static_cast<uint32_t>(score), ahead.tag);
                attentionCluster_->release(
                    AttentionClusterContextKind::Score,
                    static_cast<uint32_t>(score), ahead.tag);
                attentionCluster_->release(
                    AttentionClusterContextKind::Qk,
                    static_cast<uint32_t>(qk), ahead.tag);
                return false;
            }
            ++acquired;
        }
        ahead.qkContext = qk;
        ahead.scoreContext = score;
        statAttentionClusterContextsIssued_->addData(2);
        statAttentionClusterScoreSlotReservations_->addData(1);
        recordAttentionClusterContextHighWater();
        return true;
    }

    bool releaseAttentionClusterAheadQk(AttentionClusterAheadContext& ahead) {
        if (!attentionCluster_ || ahead.qkContext < 0) return false;
        bool ok = true;
        for (uint32_t arrayId = 0;
             arrayId < attentionClusterConfig_.qkArrays; ++arrayId) {
            ok = attentionCluster_->releaseBank(
                arrayId, ahead.operandBank, AttentionArrayOwner::Qk,
                ahead.tag) && ok;
        }
        ok = attentionCluster_->release(
            AttentionClusterContextKind::Qk,
            static_cast<uint32_t>(ahead.qkContext), ahead.tag) && ok;
        ahead.qkContext = -1;
        if (ok) statAttentionClusterContextsCompleted_->addData(1);
        return ok;
    }

    bool releaseAttentionClusterAheadScore(AttentionClusterAheadContext& ahead) {
        if (!attentionCluster_ || sfu == nullptr || ahead.scoreContext < 0)
            return false;
        const bool fifoOk = sfu->releaseAttentionScoreSlot(
            static_cast<uint32_t>(ahead.scoreContext), ahead.tag);
        const bool ok = fifoOk && attentionCluster_->release(
            AttentionClusterContextKind::Score,
            static_cast<uint32_t>(ahead.scoreContext), ahead.tag);
        ahead.scoreContext = -1;
        if (ok) {
            statAttentionClusterContextsCompleted_->addData(1);
            statAttentionClusterScoreSlotReleases_->addData(1);
        }
        return ok;
    }

    bool attentionClusterQkMatrixAheadMatchesCurrent() const {
        if (!attentionWorker_) return false;
        const AttentionWorkerState& state = *attentionWorker_;
        const AttentionClusterQkMatrixAheadContext& ahead =
            state.clusterQkMatrixAhead;
        return ahead.phase != AttentionClusterQkMatrixAheadPhase::Idle &&
            ahead.generation == state.generation &&
            ahead.kvTileIndex == state.kvTileIndex &&
            ahead.physicalKvTileIndex == state.physicalKvTileIndex &&
            ahead.tag == attentionClusterTag(state);
    }

    void pumpAttentionClusterQkMatrixLookahead() {
        if (!attentionWorker_ || !attentionClusterQkMatrixLookahead_ ||
            !attentionCluster_ || !workerCommandProcessor) return;
        AttentionWorkerState& state = *attentionWorker_;
        AttentionClusterQkMatrixAheadContext& ahead =
            state.clusterQkMatrixAhead;
        if (ahead.phase == AttentionClusterQkMatrixAheadPhase::Idle ||
            ahead.phase == AttentionClusterQkMatrixAheadPhase::Ready ||
            ahead.requestInFlight) return;
        const uint64_t generation = ahead.generation;
        const uint32_t ordinal = ahead.kvTileIndex;
        const uint32_t physicalTile = ahead.physicalKvTileIndex;
        const uint32_t bank = ahead.operandBank;
        if (ahead.phase == AttentionClusterQkMatrixAheadPhase::Reading) {
            if (ahead.readOffset == ahead.bytes.size()) {
                ahead.payload = attentionBytesToDoubles(ahead.bytes);
                ahead.bytes.clear();
                ahead.phase = AttentionClusterQkMatrixAheadPhase::Programming;
                pumpAttentionClusterQkMatrixLookahead();
                return;
            }
            const size_t offset = ahead.readOffset;
            const size_t chunk = std::min(
                ahead.bytes.size() - offset, globalMem->localMaxRequestBytes());
            const uint64_t tag = attentionTransferTag();
            const uint64_t addr = state.kLocalBuffers[ahead.buffer] + offset;
            if (globalMem->localReadAsync(
                    addr, chunk, LocalMemoryClient::RoCC, tag,
                    [this, generation, ordinal, physicalTile, bank, offset,
                     chunk, tag](bool ok, uint64_t callbackTag,
                                 const std::vector<uint8_t>& bytes) {
                        if (!attentionCallbackGenerationMatches(generation)) return;
                        AttentionClusterQkMatrixAheadContext& callbackAhead =
                            attentionWorker_->clusterQkMatrixAhead;
                        if (!ok || callbackTag != tag || bytes.size() != chunk ||
                            callbackAhead.phase !=
                                AttentionClusterQkMatrixAheadPhase::Reading ||
                            callbackAhead.kvTileIndex != ordinal ||
                            callbackAhead.physicalKvTileIndex != physicalTile ||
                            callbackAhead.operandBank != bank ||
                            callbackAhead.readOffset != offset) {
                            finishAttentionWorker(false);
                            return;
                        }
                        std::copy(bytes.begin(), bytes.end(),
                            callbackAhead.bytes.begin() + offset);
                        callbackAhead.readOffset += chunk;
                        callbackAhead.requestInFlight = false;
                        pumpAttentionClusterQkMatrixLookahead();
                    })) {
                ahead.requestInFlight = true;
            }
            return;
        }
        const uint32_t halves = attentionClusterHeadHalves(state);
        if (ahead.half == halves) {
            ahead.payload.clear();
            ahead.phase = AttentionClusterQkMatrixAheadPhase::Ready;
            if (state.clusterQkMatrixAheadPromotionWaiting &&
                attentionClusterQkMatrixAheadMatchesCurrent()) {
                state.clusterQkMatrixAheadPromotionWaiting = false;
                beginAttentionKvTile();
            }
            return;
        }
        const uint32_t half = ahead.half;
        const uint32_t keyCols = attentionKvRowsForTile(state, physicalTile);
        std::vector<double> matrix(
            static_cast<size_t>(arrayOutputSize) * arrayInputSize, 0.0);
        for (uint32_t key = 0; key < keyCols; ++key) {
            for (uint32_t dim = 0; dim < static_cast<uint32_t>(arrayInputSize);
                 ++dim) {
                matrix[static_cast<size_t>(key) * arrayInputSize + dim] =
                    ahead.payload[static_cast<size_t>(key) *
                        state.dispatch.headDim + half * arrayInputSize + dim];
            }
        }
        std::vector<uint32_t> arrayIds;
        arrayIds.reserve(attentionClusterQkLanes(state));
        for (uint32_t row = 0; row < attentionClusterQkLanes(state); ++row)
            arrayIds.push_back(attentionClusterQkArray(row, half));
        const uint64_t transferTag = attentionTransferTag();
        if (workerCommandProcessor->programGemmMatrixGroupClassBankAsync(
                arrayIds, bank, matrix, sizeof(float),
                AttentionClusterTrafficClass::QkKMatrix, transferTag,
                LastTickCycle,
                [this, generation, ordinal, physicalTile, bank, half,
                 transferTag, bytesCount = matrix.size() * sizeof(float)](
                    bool ok, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    AttentionClusterQkMatrixAheadContext& callbackAhead =
                        attentionWorker_->clusterQkMatrixAhead;
                    if (!ok || callbackTag != transferTag ||
                        callbackAhead.phase !=
                            AttentionClusterQkMatrixAheadPhase::Programming ||
                        callbackAhead.kvTileIndex != ordinal ||
                        callbackAhead.physicalKvTileIndex != physicalTile ||
                        callbackAhead.operandBank != bank ||
                        callbackAhead.half != half) {
                        finishAttentionWorker(false);
                        return;
                    }
                    callbackAhead.requestInFlight = false;
                    ++callbackAhead.half;
                    statAttentionClusterKTileBroadcasts_->addData(1);
                    statAttentionClusterKTileBytes_->addData(bytesCount);
                    statAttentionQkMatrixBroadcasts_->addData(1);
                    pumpAttentionClusterQkMatrixLookahead();
                })) {
            ahead.requestInFlight = true;
        }
    }

    void maybeStartAttentionClusterQkMatrixLookahead(
            int32_t preferredBuffer = -1) {
        if (!attentionWorker_ || !attentionClusterQkMatrixLookahead_ ||
            !attentionCluster_ ||
            attentionWorker_->clusterQkMatrixAhead.phase !=
                AttentionClusterQkMatrixAheadPhase::Idle) return;
        AttentionWorkerState& state = *attentionWorker_;
        uint32_t ordinal = state.kvTileIndex + 1;
        uint32_t physicalTile = ordinal < attentionKvTileCountForQueryTile(state)
            ? attentionPhysicalKvTileForIndex(state, ordinal) : UINT32_MAX;
        uint32_t owner = attentionKvGroupOwnerQueryTile(state);
        int32_t buffer = preferredBuffer;
        if (buffer >= 0) {
            if (static_cast<size_t>(buffer) >= state.kvBuffers.size()) return;
            const AttentionKvBufferState& descriptor = state.kvBuffers[buffer];
            ordinal = descriptor.kvTileIndex;
            physicalTile = descriptor.physicalKvTileIndex;
            owner = descriptor.queryTileIndex;
            if (!descriptor.kReady || ordinal == UINT32_MAX ||
                physicalTile == UINT32_MAX || owner == UINT32_MAX ||
                (ordinal != state.kvTileIndex &&
                 ordinal != state.kvTileIndex + 1)) return;
        } else {
            if (ordinal >= attentionKvTileCountForQueryTile(state)) return;
            buffer = findAttentionKvBuffer(ordinal, physicalTile, owner);
            if (buffer < 0 || !state.kvBuffers[buffer].kReady) return;
        }
        AttentionClusterQkMatrixAheadContext ahead;
        ahead.phase = AttentionClusterQkMatrixAheadPhase::Reading;
        ahead.generation = state.generation;
        ahead.kvTileIndex = ordinal;
        ahead.physicalKvTileIndex = physicalTile;
        ahead.operandBank = (state.activeOperandBank + 1) %
            attentionOperandContextBanks_;
        ahead.buffer = static_cast<uint32_t>(buffer);
        ahead.tag = attentionClusterTag(state);
        ahead.tag.group = owner / attentionClusterConfig_.groupSize;
        ahead.tag.queryTileIndex = owner;
        ahead.tag.queryContext = owner % attentionClusterConfig_.groupSize;
        ahead.tag.kvTileIndex = ordinal;
        const size_t bytes = static_cast<size_t>(
            attentionKvRowsForTile(state, physicalTile)) *
            state.dispatch.headDim * sizeof(float);
        ahead.bytes.resize(bytes);
        uint32_t acquired = 0;
        for (uint32_t arrayId = 0;
             arrayId < attentionClusterConfig_.qkArrays; ++arrayId) {
            if (!attentionCluster_->acquireBank(
                    arrayId, ahead.operandBank, AttentionArrayOwner::Qk,
                    ahead.tag)) {
                for (uint32_t releaseId = 0; releaseId < acquired; ++releaseId) {
                    attentionCluster_->releaseBank(
                        releaseId, ahead.operandBank,
                        AttentionArrayOwner::Qk, ahead.tag);
                }
                return;
            }
            ++acquired;
        }
        state.clusterQkMatrixAhead = std::move(ahead);
        statAttentionClusterQkMatrixLookaheadLaunches_->addData(1);
        pumpAttentionClusterQkMatrixLookahead();
    }

    bool attentionClusterPvMatrixAheadMatchesCurrent() const {
        if (!attentionWorker_) return false;
        const AttentionWorkerState& state = *attentionWorker_;
        const AttentionClusterPvMatrixAheadContext& ahead =
            state.clusterPvMatrixAhead;
        AttentionClusterTag tag = state.clusterTileTag;
        tag.queryContext = 0;
        tag.queryTileIndex = attentionKvGroupOwnerQueryTile(state);
        return ahead.phase != AttentionClusterQkMatrixAheadPhase::Idle &&
            ahead.generation == state.generation &&
            ahead.kvTileIndex == state.kvTileIndex &&
            ahead.physicalKvTileIndex == state.physicalKvTileIndex &&
            ahead.operandBank == state.clusterPvOperandBank &&
            ahead.tag == tag;
    }

    void resumeAttentionClusterPvAfterLookahead() {
        if (!attentionWorker_ || !reserveAttentionClusterPv()) {
            if (attentionWorker_) finishAttentionWorker(false);
            return;
        }
        transitionAttentionTilePipeline(
            AttentionTilePipelinePhase::PvMatrixProgram);
        beginAttentionPvOutputSlice();
        pumpAttentionClusterAhead();
    }

    void pumpAttentionClusterPvMatrixLookahead() {
        if (!attentionWorker_ || !attentionClusterPvMatrixLookahead_ ||
            !attentionCluster_ || !workerCommandProcessor) return;
        AttentionWorkerState& state = *attentionWorker_;
        AttentionClusterPvMatrixAheadContext& ahead = state.clusterPvMatrixAhead;
        if (ahead.phase == AttentionClusterQkMatrixAheadPhase::Idle ||
            ahead.phase == AttentionClusterQkMatrixAheadPhase::Ready ||
            ahead.requestInFlight) return;
        const uint64_t generation = ahead.generation;
        const uint32_t ordinal = ahead.kvTileIndex;
        const uint32_t physicalTile = ahead.physicalKvTileIndex;
        const uint32_t bank = ahead.operandBank;
        if (ahead.phase == AttentionClusterQkMatrixAheadPhase::Reading) {
            if (ahead.readOffset == ahead.bytes.size()) {
                ahead.payload = attentionBytesToDoubles(ahead.bytes);
                ahead.bytes.clear();
                ahead.phase = AttentionClusterQkMatrixAheadPhase::Programming;
                pumpAttentionClusterPvMatrixLookahead();
                return;
            }
            const size_t offset = ahead.readOffset;
            const size_t chunk = std::min(
                ahead.bytes.size() - offset, globalMem->localMaxRequestBytes());
            const uint64_t transferTag = attentionTransferTag();
            const uint64_t addr = state.vLocalBuffers[ahead.buffer] + offset;
            if (globalMem->localReadAsync(
                    addr, chunk, LocalMemoryClient::RoCC, transferTag,
                    [this, generation, ordinal, physicalTile, bank, offset,
                     chunk, transferTag](bool ok, uint64_t callbackTag,
                                         const std::vector<uint8_t>& bytes) {
                        if (!attentionCallbackGenerationMatches(generation)) return;
                        auto& callbackAhead = attentionWorker_->clusterPvMatrixAhead;
                        if (!ok || callbackTag != transferTag ||
                            bytes.size() != chunk ||
                            callbackAhead.phase !=
                                AttentionClusterQkMatrixAheadPhase::Reading ||
                            callbackAhead.kvTileIndex != ordinal ||
                            callbackAhead.physicalKvTileIndex != physicalTile ||
                            callbackAhead.operandBank != bank ||
                            callbackAhead.readOffset != offset) {
                            finishAttentionWorker(false);
                            return;
                        }
                        std::copy(bytes.begin(), bytes.end(),
                            callbackAhead.bytes.begin() + offset);
                        callbackAhead.readOffset += chunk;
                        callbackAhead.requestInFlight = false;
                        pumpAttentionClusterPvMatrixLookahead();
                    })) ahead.requestInFlight = true;
            return;
        }
        const uint32_t halves = attentionClusterHeadHalves(state);
        if (ahead.half == halves) {
            ahead.payload.clear();
            ahead.phase = AttentionClusterQkMatrixAheadPhase::Ready;
            if (state.clusterPvMatrixAheadPromotionWaiting &&
                attentionClusterPvMatrixAheadMatchesCurrent()) {
                state.clusterPvMatrixAheadPromotionWaiting = false;
                resumeAttentionClusterPvAfterLookahead();
            }
            return;
        }
        const uint32_t half = ahead.half;
        const uint32_t keyCols = attentionKvRowsForTile(state, physicalTile);
        std::vector<double> matrix(
            static_cast<size_t>(arrayOutputSize) * keyCols, 0.0);
        for (uint32_t dim = 0; dim < static_cast<uint32_t>(arrayOutputSize); ++dim) {
            for (uint32_t key = 0; key < keyCols; ++key) {
                matrix[static_cast<size_t>(dim) * keyCols + key] =
                    ahead.payload[static_cast<size_t>(key) *
                        state.dispatch.headDim + half * arrayOutputSize + dim];
            }
        }
        std::vector<uint32_t> arrayIds;
        arrayIds.reserve(attentionClusterPvLanes(state));
        for (uint32_t row = 0; row < attentionClusterPvLanes(state); ++row)
            arrayIds.push_back(attentionClusterPvArray(row, half));
        const uint64_t transferTag = attentionTransferTag();
        if (workerCommandProcessor->programGemmMatrixGroupActiveBankAsync(
                arrayIds, bank, matrix, keyCols, sizeof(float), transferTag,
                LastTickCycle,
                [this, generation, ordinal, physicalTile, bank, half,
                 transferTag](bool ok, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    auto& callbackAhead = attentionWorker_->clusterPvMatrixAhead;
                    if (!ok || callbackTag != transferTag ||
                        callbackAhead.phase !=
                            AttentionClusterQkMatrixAheadPhase::Programming ||
                        callbackAhead.kvTileIndex != ordinal ||
                        callbackAhead.physicalKvTileIndex != physicalTile ||
                        callbackAhead.operandBank != bank ||
                        callbackAhead.half != half) {
                        finishAttentionWorker(false);
                        return;
                    }
                    callbackAhead.requestInFlight = false;
                    ++callbackAhead.half;
                    pumpAttentionClusterPvMatrixLookahead();
                })) ahead.requestInFlight = true;
    }

    void maybeStartAttentionClusterPvMatrixLookahead(int32_t buffer = -1) {
        if (!attentionWorker_ || !attentionClusterPvMatrixLookahead_ ||
            !attentionCluster_ ||
            attentionWorker_->clusterPvMatrixAhead.phase !=
                AttentionClusterQkMatrixAheadPhase::Idle) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t nextOrdinal = state.kvTileIndex + 1;
        if (buffer < 0) {
            if (nextOrdinal >= attentionKvTileCountForQueryTile(state)) return;
            buffer = findAttentionKvBuffer(
                nextOrdinal,
                attentionPhysicalKvTileForIndex(state, nextOrdinal),
                attentionKvGroupOwnerQueryTile(state));
        }
        if (buffer < 0) return;
        if (static_cast<size_t>(buffer) >= state.kvBuffers.size()) return;
        const AttentionKvBufferState& descriptor = state.kvBuffers[buffer];
        if (!descriptor.ready || descriptor.kvTileIndex == UINT32_MAX ||
            descriptor.physicalKvTileIndex == UINT32_MAX ||
            descriptor.queryTileIndex == UINT32_MAX ||
            (descriptor.kvTileIndex != state.kvTileIndex &&
             descriptor.kvTileIndex != nextOrdinal)) return;
        AttentionClusterPvMatrixAheadContext ahead;
        ahead.phase = AttentionClusterQkMatrixAheadPhase::Reading;
        ahead.generation = state.generation;
        ahead.kvTileIndex = descriptor.kvTileIndex;
        ahead.physicalKvTileIndex = descriptor.physicalKvTileIndex;
        ahead.operandBank = descriptor.kvTileIndex %
            attentionOperandContextBanks_;
        ahead.buffer = static_cast<uint32_t>(buffer);
        ahead.tag = attentionClusterTag(state);
        ahead.tag.group = descriptor.queryTileIndex /
            attentionClusterConfig_.groupSize;
        ahead.tag.queryContext = 0;
        ahead.tag.queryTileIndex = descriptor.queryTileIndex;
        ahead.tag.kvTileIndex = descriptor.kvTileIndex;
        ahead.bytes.resize(static_cast<size_t>(
            attentionKvRowsForTile(state, descriptor.physicalKvTileIndex)) *
            state.dispatch.headDim * sizeof(float));
        statAttentionPvVTileBufferMisses_->addData(1);
        statAttentionPvVTileBufferBytesRead_->addData(ahead.bytes.size());
        uint32_t acquired = 0;
        for (uint32_t arrayId = attentionClusterConfig_.qkArrays;
             arrayId < attentionClusterConfig_.arrays; ++arrayId) {
            if (!attentionCluster_->acquireBank(
                    arrayId, ahead.operandBank, AttentionArrayOwner::Pv,
                    ahead.tag)) {
                for (uint32_t offset = 0; offset < acquired; ++offset) {
                    attentionCluster_->releaseBank(
                        attentionClusterConfig_.qkArrays + offset,
                        ahead.operandBank, AttentionArrayOwner::Pv, ahead.tag);
                }
                return;
            }
            ++acquired;
        }
        state.clusterPvMatrixAhead = std::move(ahead);
        statAttentionClusterPvMatrixLookaheadLaunches_->addData(1);
        pumpAttentionClusterPvMatrixLookahead();
    }

    bool startAttentionClusterAheadContext(uint32_t queryTileIndex) {
        if (!attentionWorker_ || !attentionClusterEnable_) return false;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t slot = queryTileIndex % attentionClusterConfig_.groupSize;
        AttentionClusterAheadContext& ahead = state.clusterAhead[slot];
        if (ahead.phase != AttentionClusterAheadPhase::Idle) return false;
        ahead.generation = state.generation;
        ahead.queryTileIndex = queryTileIndex;
        ahead.kvTileIndex = state.kvTileIndex;
        ahead.physicalKvTileIndex = state.physicalKvTileIndex;
        ahead.operandBank = state.activeOperandBank;
        ahead.qLocal = state.qLocalBuffers[slot];
        ahead.tag.generation = state.generation;
        ahead.tag.jobId = state.dispatch.jobId;
        ahead.tag.group = queryTileIndex / attentionClusterConfig_.groupSize;
        ahead.tag.queryContext = slot;
        ahead.tag.queryTileIndex = queryTileIndex;
        ahead.tag.kvTileIndex = state.kvTileIndex;
        ahead.queryBytes.resize(
            static_cast<size_t>(attentionQueryRows(state)) *
            state.dispatch.headDim * sizeof(float));
        if (!reserveAttentionClusterAheadQk(ahead)) {
            ahead = {};
            return false;
        }
        statAttentionClusterAheadContextsLaunched_->addData(1);
        ahead.phase = state.clusterQueryLoaded[slot]
            ? AttentionClusterAheadPhase::QueryRead
            : AttentionClusterAheadPhase::QueryDma;
        return true;
    }

    bool attentionClusterHasQkProducer() const {
        if (!attentionWorker_) return false;
        for (const auto& ahead : attentionWorker_->clusterAhead) {
            if (ahead.phase >= AttentionClusterAheadPhase::QueryDma &&
                ahead.phase <= AttentionClusterAheadPhase::ScoreWrite) return true;
        }
        return false;
    }

    void scheduleAttentionClusterOwnerRetry(std::function<void()> retry) {
        if (!attentionWorker_ || !retry) return;
        auto& state = *attentionWorker_;
        constexpr size_t kRetryQueueLimit = 16;
        statAttentionClusterInitialEnqueueRetries_->addData(1);
        if (state.clusterOwnerRetry || state.clusterOwnerRetryQueue.size() >= kRetryQueueLimit) {
            if (state.clusterOwnerRetryQueue.size() >= kRetryQueueLimit) {
                statAttentionClusterIllegalTransitions_->addData(1);
                finishAttentionWorker(false);
            } else {
                state.clusterOwnerRetryQueue.push_back(std::move(retry));
            }
            return;
        }
        state.clusterOwnerRetry = std::move(retry);
    }

    void maybeStartAttentionClusterAhead() {
        if (!attentionWorker_ || !attentionClusterEnable_ ||
            !attentionWorker_->clusterAheadEnabledForTile ||
            attentionClusterHasQkProducer()) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t end = attentionKvGroupEndQueryTile(state);
        for (uint32_t query = state.queryTileIndex + 1; query < end; ++query) {
            if (attentionClusterAhead(query, state.kvTileIndex)) continue;
            if (startAttentionClusterAheadContext(query)) break;
            return;
        }
    }

    void pumpAttentionClusterAhead() {
        if (!attentionWorker_ || !attentionClusterEnable_) return;
        maybeStartAttentionClusterAhead();
        AttentionWorkerState& state = *attentionWorker_;
        AttentionClusterAheadContext* selected = nullptr;
        for (auto& candidate : state.clusterAhead) {
            if (candidate.phase >= AttentionClusterAheadPhase::QueryDma &&
                candidate.phase <= AttentionClusterAheadPhase::SoftmaxIssue) {
                selected = &candidate;
                break;
            }
        }
        if (selected == nullptr) return;
        AttentionClusterAheadContext& ahead = *selected;
        const uint64_t generation = ahead.generation;
        const uint32_t queryTileIndex = ahead.queryTileIndex;
        const uint32_t kvTileIndex = ahead.kvTileIndex;
        const uint32_t slot = queryTileIndex % attentionClusterConfig_.groupSize;

        if (ahead.phase == AttentionClusterAheadPhase::QueryDma) {
            if (ahead.requestInFlight) return;
            ahead.requestInFlight = true;
            globalMem->dma_read_from_host_to_globalmem(
                state.dispatch.qAddr + static_cast<uint64_t>(queryTileIndex) *
                    state.dispatch.queryTileRows * state.dispatch.headDim *
                    sizeof(float),
                ahead.queryBytes.size(), ahead.qLocal,
                [this, generation, queryTileIndex, kvTileIndex, slot](bool ok) {
                    AttentionClusterAheadContext* callbackAhead =
                        attentionClusterAhead(queryTileIndex, kvTileIndex);
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok || callbackAhead == nullptr ||
                        callbackAhead->phase !=
                            AttentionClusterAheadPhase::QueryDma ||
                        !callbackAhead->requestInFlight) {
                        finishAttentionWorker(false);
                        return;
                    }
                    callbackAhead->requestInFlight = false;
                    callbackAhead->phase = AttentionClusterAheadPhase::QueryRead;
                    attentionWorker_->clusterQueryLoaded[slot] = true;
                    pumpAttentionClusterAhead();
                }, DmaRequestKind::AttentionQuery);
            return;
        }
        if (ahead.phase == AttentionClusterAheadPhase::QueryRead) {
            if (ahead.requestInFlight) return;
            if (ahead.queryOffset == ahead.queryBytes.size()) {
                ahead.qPayload = attentionBytesToDoubles(ahead.queryBytes);
                ahead.queryBytes.clear();
                ahead.phase = AttentionClusterAheadPhase::InputProgram;
                pumpAttentionClusterAhead();
                return;
            }
            const size_t chunk = std::min(
                ahead.queryBytes.size() - ahead.queryOffset,
                globalMem->localMaxRequestBytes());
            const size_t offset = ahead.queryOffset;
            const uint64_t tag = attentionTransferTag();
            if (globalMem->localReadAsync(
                    ahead.qLocal + offset, chunk, LocalMemoryClient::RoCC, tag,
                    [this, generation, queryTileIndex, kvTileIndex, offset, chunk,
                     tag](bool ok, uint64_t callbackTag,
                          const std::vector<uint8_t>& bytes) {
                        AttentionClusterAheadContext* callbackAhead =
                            attentionClusterAhead(queryTileIndex, kvTileIndex);
                        if (!attentionCallbackGenerationMatches(generation)) return;
                        if (!ok || callbackTag != tag || callbackAhead == nullptr ||
                            callbackAhead->phase !=
                                AttentionClusterAheadPhase::QueryRead ||
                            callbackAhead->queryOffset != offset ||
                            bytes.size() != chunk) {
                            finishAttentionWorker(false);
                            return;
                        }
                        callbackAhead->requestInFlight = false;
                        std::copy(bytes.begin(), bytes.end(),
                            callbackAhead->queryBytes.begin() + offset);
                        callbackAhead->queryOffset += chunk;
                        pumpAttentionClusterAhead();
                    })) {
                ahead.requestInFlight = true;
            } else {
                statAttentionClusterInitialEnqueueRetries_->addData(1);
            }
            return;
        }
        if (ahead.phase == AttentionClusterAheadPhase::InputProgram) {
            if (ahead.requestInFlight) return;
            const uint32_t halves = attentionClusterHeadHalves(state);
            const uint32_t lanes = attentionClusterQkLanes(state);
            const uint32_t activeArrays = attentionClusterWaveRows(
                state, ahead.wave, lanes) * halves;
            if (ahead.pair == activeArrays) {
                ahead.launchIndex = 0;
                ahead.arraysPending = 0;
                ahead.phase = AttentionClusterAheadPhase::Launch;
                pumpAttentionClusterAhead();
                return;
            }
            const uint32_t physical = ahead.pair;
            const uint32_t wave = ahead.wave;
            const uint32_t query = attentionClusterWaveRowBegin(
                ahead.wave, lanes) + physical / halves;
            const uint32_t half = physical % halves;
            if (query >= attentionQueryRowsForTile(state, queryTileIndex)) {
                finishAttentionWorker(false);
                return;
            }
            std::vector<double> input(
                ahead.qPayload.begin() +
                    static_cast<size_t>(query) * state.dispatch.headDim +
                        half * arrayInputSize,
                ahead.qPayload.begin() +
                    static_cast<size_t>(query) * state.dispatch.headDim +
                        (half + 1) * arrayInputSize);
            std::vector<uint32_t> arrays = {
                attentionClusterQkArray(query, half)};
            const uint64_t tag = attentionTransferTag();
            if (workerCommandProcessor->programGemmInputGroupBankAsync(
                    arrays, ahead.operandBank, input, sizeof(float),
                    AttentionClusterTrafficClass::QkQPair, tag, LastTickCycle,
                    [this, generation, queryTileIndex, kvTileIndex, physical, wave,
                     tag, bytes = input.size() * sizeof(float)](
                        bool ok, uint64_t callbackTag) {
                        AttentionClusterAheadContext* callbackAhead =
                            attentionClusterAhead(queryTileIndex, kvTileIndex);
                        if (!attentionCallbackGenerationMatches(generation)) return;
                        if (!ok || callbackTag != tag || callbackAhead == nullptr ||
                            callbackAhead->phase !=
                                AttentionClusterAheadPhase::InputProgram ||
                            callbackAhead->wave != wave ||
                            callbackAhead->pair != physical) {
                            finishAttentionWorker(false);
                            return;
                        }
                        callbackAhead->requestInFlight = false;
                        statAttentionClusterQPairMulticasts_->addData(1);
                        statAttentionClusterQPairBytes_->addData(bytes);
                        ++callbackAhead->pair;
                        pumpAttentionClusterAhead();
                    })) {
                ahead.requestInFlight = true;
            } else {
                statAttentionClusterInitialEnqueueRetries_->addData(1);
            }
            return;
        }
        if (ahead.phase == AttentionClusterAheadPhase::Launch) {
            if (ahead.requestInFlight || ahead.launchIndex != 0) return;
            const uint32_t activeArrays = attentionClusterWaveRows(
                state, ahead.wave, attentionClusterQkLanes(state)) *
                attentionClusterHeadHalves(state);
            std::vector<uint32_t> arrayIds(activeArrays);
            std::iota(arrayIds.begin(), arrayIds.end(), 0);
            const uint64_t acceptedCycle = LastTickCycle;
            const bool accepted = workerCommandProcessor->launchGemmArrayGroupActiveBank(
                arrayIds, ahead.operandBank, 0, 0, LastTickCycle,
                [this, generation, queryTileIndex, kvTileIndex, activeArrays](
                    uint32_t completedArrayId, uint64_t) {
                    AttentionClusterAheadContext* callbackAhead =
                        attentionClusterAhead(queryTileIndex, kvTileIndex);
                    if (!attentionCallbackGenerationMatches(generation)) {
                        if (!attentionClusterQkArrayActivity_.leave(
                                getCurrentSimCycle())) {
                            statAttentionClusterIllegalTransitions_->addData(1);
                        }
                        return;
                    }
                    if (callbackAhead == nullptr ||
                        callbackAhead->phase !=
                            AttentionClusterAheadPhase::Compute ||
                        callbackAhead->arraysPending == 0) {
                        finishAttentionWorker(false);
                        return;
                    }
                    arrayStates[completedArrayId] = 0;
                    if (!attentionClusterQkArrayActivity_.leave(
                            getCurrentSimCycle())) {
                        finishAttentionWorker(false);
                        return;
                    }
                    --callbackAhead->arraysPending;
                    if (callbackAhead->arraysPending == 0 &&
                        callbackAhead->launchIndex == activeArrays) {
                        callbackAhead->qkEndCycle = LastTickCycle;
                        callbackAhead->outputIndex = attentionClusterWaveRowBegin(
                            callbackAhead->wave,
                            attentionClusterQkLanes(*attentionWorker_));
                        callbackAhead->phase =
                            AttentionClusterAheadPhase::OutputRead;
                        pumpAttentionClusterAhead();
                    }
                });
            if (!accepted) {
                statAttentionClusterInitialEnqueueRetries_->addData(1);
                return;
            }
            ahead.qkStartCycle = acceptedCycle;
            if (ahead.wave == 0) {
                recordAttentionClusterQkTileStart(
                    acceptedCycle, ahead.kvTileIndex, ahead.queryTileIndex);
            }
            for (uint32_t arrayId : arrayIds) {
                arrayStates[arrayId] = 1;
                if (!attentionClusterQkArrayActivity_.enter(getCurrentSimCycle())) {
                    finishAttentionWorker(false);
                    return;
                }
                statAttentionQkArrayOps_->addData(1);
                statAttentionGenericGemmQkOps_->addData(1);
            }
            ahead.launchIndex = activeArrays;
            ahead.arraysPending = activeArrays;
            ahead.phase = AttentionClusterAheadPhase::Compute;
            return;
        }
        if (ahead.phase == AttentionClusterAheadPhase::OutputRead) {
            if (ahead.requestInFlight) return;
            const uint32_t lanes = attentionClusterQkLanes(state);
            const uint32_t waveEnd = attentionClusterWaveRowBegin(
                ahead.wave, lanes) + attentionClusterWaveRows(
                    state, ahead.wave, lanes);
            if (ahead.outputIndex == waveEnd) {
                ++ahead.wave;
                if (ahead.wave < attentionClusterQkWaves(state)) {
                    ahead.pair = 0;
                    ahead.phase = AttentionClusterAheadPhase::InputProgram;
                    pumpAttentionClusterAhead();
                    return;
                }
                statAttentionWorkerQkTileCompleteTick_->addData(
                    getCurrentSimCycle());
                if (attentionTileTrace_) {
                    traceAttentionMilestone(
                        "worker", "qk_tile_complete", "done",
                        state.dispatch.jobId, state.dispatch.tag,
                        queryTileIndex, kvTileIndex);
                }
                if (queryTileIndex + 1 == attentionQueryTileCount(state) &&
                    kvTileIndex + 1 ==
                        attentionKvTileCountForQueryTile(state)) {
                    traceAttentionMilestone(
                        "worker", "final_qk_tile_complete", "done",
                        state.dispatch.jobId, state.dispatch.tag,
                        queryTileIndex, kvTileIndex);
                }
                if (!releaseAttentionClusterAheadQk(ahead)) {
                    finishAttentionWorker(false);
                    return;
                }
                ahead.phase = AttentionClusterAheadPhase::SoftmaxIssue;
                pumpAttentionClusterAhead();
                return;
            }
            const uint32_t logicalQuery = ahead.outputIndex;
            const uint32_t wave = ahead.wave;
            const uint32_t halves = attentionClusterHeadHalves(state);
            std::vector<uint32_t> arrays(halves);
            for (uint32_t half = 0; half < halves; ++half)
                arrays[half] = attentionClusterQkArray(logicalQuery, half);
            const uint64_t tag = attentionTransferTag();
            if (workerCommandProcessor->readGemmOutputGroupClassAsync(
                    arrays, sizeof(float),
                    AttentionClusterTrafficClass::QkScoreOut, tag,
                    LastTickCycle,
                    [this, generation, queryTileIndex, kvTileIndex,
                     logicalQuery, halves, tag](
                        bool ok, uint64_t callbackTag,
                        const std::vector<double>& values) {
                        AttentionClusterAheadContext* callbackAhead =
                            attentionClusterAhead(queryTileIndex, kvTileIndex);
                        if (!attentionCallbackGenerationMatches(generation)) return;
                        if (!ok || callbackTag != tag || callbackAhead == nullptr ||
                            callbackAhead->phase !=
                                AttentionClusterAheadPhase::OutputRead ||
                            callbackAhead->outputIndex != logicalQuery ||
                            values.size() != static_cast<size_t>(halves) *
                                attentionClusterConfig_.arrayOutputs) {
                            finishAttentionWorker(false);
                            return;
                        }
                        callbackAhead->requestInFlight = false;
                        const uint32_t keyCols =
                            attentionKeyCols(*attentionWorker_);
                        callbackAhead->pendingScoreBeat.assign(keyCols, 0.0f);
                        for (uint32_t key = 0; key < keyCols; ++key) {
                            double sum = 0.0;
                            for (uint32_t half = 0; half < halves; ++half) {
                                sum += values[static_cast<size_t>(half) *
                                    attentionClusterConfig_.arrayOutputs + key];
                            }
                            callbackAhead->pendingScoreBeat[key] =
                                static_cast<float>(sum);
                        }
                        callbackAhead->transferTag =
                            static_cast<uint64_t>(logicalQuery) *
                                attentionKeyCols(*attentionWorker_);
                        callbackAhead->phase =
                            AttentionClusterAheadPhase::ScoreWrite;
                        pumpAttentionClusterAhead();
                    })) {
                ahead.requestInFlight = true;
            } else {
                statAttentionClusterInitialEnqueueRetries_->addData(1);
            }
            return;
        }
        if (ahead.phase == AttentionClusterAheadPhase::ScoreWrite) {
            if (ahead.requestInFlight) return;
            const uint64_t tag = attentionTransferTag();
            const size_t offset = static_cast<size_t>(ahead.transferTag);
            const size_t bytes = ahead.pendingScoreBeat.size() * sizeof(float);
            if (sfu->writeAttentionScoreBeatAsync(
                    static_cast<uint32_t>(ahead.scoreContext), ahead.tag,
                    offset, ahead.pendingScoreBeat, tag,
                    [this, generation, queryTileIndex, kvTileIndex, tag, bytes](
                        bool ok, uint64_t callbackTag) {
                        AttentionClusterAheadContext* callbackAhead =
                            attentionClusterAhead(queryTileIndex, kvTileIndex);
                        if (!attentionCallbackGenerationMatches(generation)) return;
                        if (!ok || callbackTag != tag || callbackAhead == nullptr ||
                            callbackAhead->phase !=
                                AttentionClusterAheadPhase::ScoreWrite) {
                            finishAttentionWorker(false);
                            return;
                        }
                        callbackAhead->requestInFlight = false;
                        statAttentionClusterScoreBeats_->addData(1);
                        statAttentionClusterScoreBytes_->addData(bytes);
                        callbackAhead->pendingScoreBeat.clear();
                        ++callbackAhead->outputIndex;
                        callbackAhead->phase =
                            AttentionClusterAheadPhase::OutputRead;
                        pumpAttentionClusterAhead();
                    })) {
                ahead.requestInFlight = true;
            } else {
                statAttentionClusterInitialEnqueueRetries_->addData(1);
            }
            return;
        }
        if (ahead.phase == AttentionClusterAheadPhase::SoftmaxIssue) {
            if (ahead.pContext < 0) {
                const size_t pElements =
                    static_cast<size_t>(attentionQueryRows(state)) *
                    attentionKeyCols(state);
                const AttentionClusterAdmission admission =
                    sfu->attentionPSlotAdmission(
                        static_cast<uint32_t>(ahead.scoreContext), ahead.tag,
                        pElements);
                if (admission == AttentionClusterAdmission::Invalid) {
                    finishAttentionWorker(false);
                    return;
                }
                if (admission == AttentionClusterAdmission::Retry) return;
                if (!sfu->reserveAttentionPSlot(
                        static_cast<uint32_t>(ahead.scoreContext), ahead.tag,
                        pElements)) {
                    finishAttentionWorker(false);
                    return;
                }
                ahead.pContext = ahead.scoreContext;
            }
            AttentionTileRequest request;
            request.tag = state.dispatch.tag +
                queryTileIndex * ((state.dispatch.expectedCols +
                    state.dispatch.kvTileRows - 1) /
                    state.dispatch.kvTileRows) + kvTileIndex + 1;
            request.jobId = state.dispatch.jobId;
            request.globalRowBegin = state.dispatch.row +
                queryTileIndex * state.dispatch.queryTileRows;
            request.keyBegin = ahead.physicalKvTileIndex * state.dispatch.kvTileRows;
            request.rows = attentionQueryRows(state);
            request.cols = attentionKeyCols(state);
            request.headDim = state.dispatch.headDim;
            request.kvTileIndex = kvTileIndex;
            request.numKvTiles = attentionKvTileCountForQueryTile(state);
            request.causal = false;
            request.firstTileForJob = false;
            request.directScoreMode = true;
            request.generation = generation;
            request.scoreSlot = static_cast<uint32_t>(ahead.scoreContext);
            request.scoreTag = ahead.tag;
            request.directPMode = true;
            request.pSlot = static_cast<uint32_t>(ahead.pContext);
            request.pTag = ahead.tag;
            const AttentionClusterAdmission admission =
                sfu->attentionTileAdmission(request);
            if (admission == AttentionClusterAdmission::Invalid) {
                finishAttentionWorker(false);
                return;
            }
            if (admission == AttentionClusterAdmission::Retry) {
                statAttentionClusterInitialEnqueueRetries_->addData(1);
                return;
            }
            if (!sfu->issueAttentionTile(
                    request,
                    [this, generation, queryTileIndex, kvTileIndex](
                        bool ok, const AttentionTileResult& result) {
                        AttentionClusterAheadContext* callbackAhead =
                            attentionClusterAhead(queryTileIndex, kvTileIndex);
                        if (!attentionCallbackGenerationMatches(generation)) return;
                        if (!ok || callbackAhead == nullptr ||
                            callbackAhead->phase !=
                                AttentionClusterAheadPhase::SoftmaxRunning ||
                            !releaseAttentionClusterAheadScore(*callbackAhead)) {
                            finishAttentionWorker(false);
                            return;
                        }
                        callbackAhead->outputScales.assign(
                            result.oldOutputScale.begin(),
                            result.oldOutputScale.begin() + result.rows);
                        callbackAhead->sfuEndCycle = LastTickCycle;
                        statAttentionWorkerSoftmaxTileCompleteTick_->addData(
                            getCurrentSimCycle());
                        if (attentionTileTrace_) {
                            traceAttentionMilestone(
                                "worker", "softmax_tile_complete", "done",
                                attentionWorker_->dispatch.jobId,
                                attentionWorker_->dispatch.tag,
                                queryTileIndex, kvTileIndex);
                        }
                        if (queryTileIndex + 1 ==
                                attentionQueryTileCount(*attentionWorker_) &&
                            kvTileIndex + 1 ==
                                attentionKvTileCountForQueryTile(
                                    *attentionWorker_)) {
                            traceAttentionMilestone(
                                "worker", "final_softmax_tile_complete", "done",
                                attentionWorker_->dispatch.jobId,
                                attentionWorker_->dispatch.tag,
                                queryTileIndex, kvTileIndex);
                        }
                        callbackAhead->phase =
                            AttentionClusterAheadPhase::Ready;
                        statAttentionClusterAheadContextsCompleted_->addData(1);
                        if (attentionWorker_->clusterAheadPromotionWaiting &&
                            attentionWorker_->queryTileIndex == queryTileIndex) {
                            attentionWorker_->clusterAheadPromotionWaiting = false;
                            promoteAttentionClusterAhead();
                            return;
                        }
                        pumpAttentionClusterAhead();
                    })) {
                finishAttentionWorker(false);
                return;
            }
            ahead.sfuStartCycle = LastTickCycle;
            ahead.phase = AttentionClusterAheadPhase::SoftmaxRunning;
            maybeStartAttentionClusterAhead();
            pumpAttentionClusterAhead();
        }
    }

    bool promoteAttentionClusterAhead() {
        if (!attentionWorker_ || !attentionClusterEnable_) return false;
        AttentionWorkerState& state = *attentionWorker_;
        AttentionClusterAheadContext* ahead =
            attentionClusterAhead(state.queryTileIndex, state.kvTileIndex);
        if (ahead == nullptr || ahead->phase != AttentionClusterAheadPhase::Ready)
            return false;
        state.activeOperandBank = ahead->operandBank;
        state.clusterTileTag = ahead->tag;
        state.clusterQkContext = -1;
        state.clusterScoreContext = -1;
        state.clusterPContext = ahead->pContext;
        state.outputScales = std::move(ahead->outputScales);
        ahead->pContext = -1;
        *ahead = {};
        statAttentionClusterAheadContextsPromoted_->addData(1);
        state.phaseSliceIndex = 0;
        state.clusterPvWave = 0;
        state.attentionSoftmaxComplete = true;
        beginAttentionTilePipeline();
        const DmaConsumerMetadata consumerProgress = attentionDmaConsumerMetadata(
            state, state.queryTileIndex, state.kvTileIndex, DmaOperand::Unknown);
        globalMem->dma_update_consumer_progress(
            attentionKvHostAddr(state, state.dispatch.kAddr),
            attentionKvHostAddr(state, state.dispatch.vAddr), consumerProgress);
        if (!ensureAttentionClusterOContext()) {
            finishAttentionWorker(false);
            return true;
        }
        if (!reserveAttentionClusterPv()) {
            if (state.clusterPvMatrixAheadPromotionWaiting) return true;
            finishAttentionWorker(false);
            return true;
        }
        transitionAttentionTilePipeline(
            AttentionTilePipelinePhase::PvMatrixProgram);
        beginAttentionPvOutputSlice();
        pumpAttentionClusterAhead();
        return true;
    }

    bool selectAttentionQueryStorage(AttentionWorkerState& state) {
        const uint32_t slot = state.queryTileIndex % attentionKvQueryGroupSize(state);
        if (slot >= state.qLocalBuffers.size() ||
            state.oLocalBuffers.empty()) {
            return false;
        }
        state.qLocal = state.qLocalBuffers[slot];
        state.oLocal = state.oLocalBuffers[
            attentionClusterEnable_ ? 0 : slot];
        return true;
    }

    uint32_t attentionKvSubtileCount(const AttentionWorkerState& state) const {
        if (attentionSequential64Enable_) return 1;
        return (attentionKeyCols(state) + 15) / 16;
    }

    bool buildAttentionSequentialQkMatrix(AttentionWorkerState& state) {
        if (!attentionSequential64Enable_ || state.qkReductionSlice >= 2 ||
            state.dispatch.headDim != 2u * static_cast<uint32_t>(arrayInputSize)) {
            return false;
        }
        state.arrayPayload.assign(
            static_cast<size_t>(arrayOutputSize) * arrayInputSize, 0.0);
        const size_t qOffset =
            static_cast<size_t>(state.qkReductionSlice) * arrayInputSize;
        const uint32_t rows = attentionQueryRows(state);
        if (state.qPayload.size() <
            static_cast<size_t>(rows) * state.dispatch.headDim) {
            return false;
        }
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < static_cast<uint32_t>(arrayInputSize);
                 ++col) {
                state.arrayPayload[static_cast<size_t>(row) * arrayInputSize + col] =
                    state.qPayload[static_cast<size_t>(row) * state.dispatch.headDim +
                                   qOffset + col];
            }
        }
        return true;
    }

    uint32_t attentionClusterQkLanes(const AttentionWorkerState& state) const {
        const uint32_t halves = attentionClusterHeadHalves(state);
        return halves == 0 ? 0 : std::min(
            attentionQueryRows(state), attentionClusterConfig_.qkArrays / halves);
    }

    uint32_t attentionClusterQkWaves(const AttentionWorkerState& state) const {
        const uint32_t lanes = attentionClusterQkLanes(state);
        return lanes == 0 ? 0 :
            (attentionQueryRows(state) + lanes - 1) / lanes;
    }

    uint32_t attentionClusterPvLanes(const AttentionWorkerState& state) const {
        const uint32_t halves = attentionClusterHeadHalves(state);
        return halves == 0 ? 0 : std::min(
            attentionQueryRows(state), attentionClusterConfig_.pvArrays / halves);
    }

    uint32_t attentionClusterWaveRowBegin(
            uint32_t wave, uint32_t lanes) const {
        return wave * lanes;
    }

    uint32_t attentionClusterWaveRows(
            const AttentionWorkerState& state, uint32_t wave,
            uint32_t lanes) const {
        const uint32_t begin = attentionClusterWaveRowBegin(wave, lanes);
        return begin >= attentionQueryRows(state) ? 0 :
            std::min(lanes, attentionQueryRows(state) - begin);
    }

    uint32_t attentionClusterPvWaves(const AttentionWorkerState& state) const {
        const uint32_t lanes = attentionClusterPvLanes(state);
        return lanes == 0 ? 0 :
            (attentionQueryRows(state) + lanes - 1) / lanes;
    }

    uint32_t attentionOutputSliceCount(const AttentionWorkerState& state) const {
        if (attentionSequential64Enable_)
            return (state.dispatch.headDim + arrayOutputSize - 1) / arrayOutputSize;
        return (state.dispatch.headDim + 15) / 16;
    }

    uint32_t attentionSequentialPvOutputSliceCount(
            const AttentionWorkerState& state) const {
        if (!attentionSequential64Enable_ || state.phaseSliceIndex >=
                attentionOutputSliceCount(state)) return 0;
        return 1;
    }

    uint32_t attentionSequentialPvArray(
            const AttentionWorkerState& state, uint32_t localOutputSlice,
            uint32_t row) const {
        (void)localOutputSlice;
        return row;
    }

    uint32_t attentionClusterHeadHalves(
            const AttentionWorkerState& state) const {
        return (state.dispatch.headDim + attentionClusterConfig_.arrayInputs - 1) /
            attentionClusterConfig_.arrayInputs;
    }

    uint32_t attentionClusterQkArray(uint32_t row, uint32_t half) const {
        const uint32_t lanes = attentionClusterQkLanes(*attentionWorker_);
        return (row % lanes) * attentionClusterHeadHalves(*attentionWorker_) + half;
    }

    uint32_t attentionClusterPvArray(uint32_t row, uint32_t half) const {
        const uint32_t lanes = attentionClusterPvLanes(*attentionWorker_);
        return attentionClusterConfig_.qkArrays +
            (row % lanes) * attentionClusterHeadHalves(*attentionWorker_) + half;
    }

    uint32_t attentionOAccumulatorRow(
        const AttentionWorkerState& state, uint32_t arrayId) const {
        return arrayId * attentionOutputSliceCount(state) + state.phaseSliceIndex;
    }

    bool attentionPvInputResidentForCurrentTile(
        const AttentionWorkerState& state) const {
        return attentionPvInputResidency_ &&
            state.attentionPvInputResidentValid &&
            state.attentionPvInputResidentQueryTile == state.queryTileIndex &&
            state.attentionPvInputResidentKvTile == state.kvTileIndex;
    }

    void invalidateAttentionPvInputResidency() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.attentionPvInputResidentValid) {
            statAttentionPvInputResidencyInvalidations_->addData(1);
        }
        state.attentionPvInputResidentValid = false;
        state.attentionPvInputResidentQueryTile = UINT32_MAX;
        state.attentionPvInputResidentKvTile = UINT32_MAX;
    }

    bool attentionVTileMatchesCurrentGroup(
        const AttentionWorkerState& state) const {
        return state.vTileValid && state.vTileGeneration == state.generation &&
            state.vTileGroupOwner == attentionKvGroupOwnerQueryTile(state) &&
            state.vTileTag == state.kvTileIndex;
    }

    void invalidateAttentionVTileStaging() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.vTileValid = false;
        state.vTileTag = UINT32_MAX;
        state.vTileGeneration = 0;
        state.vTileGroupOwner = UINT32_MAX;
        state.vPayload.clear();
    }

    uint32_t attentionKvSubtileRows(const AttentionWorkerState& state) const {
        if (attentionSequential64Enable_) return attentionKeyCols(state);
        return std::min<uint32_t>(16, attentionKeyCols(state) - state.phaseSliceIndex * 16);
    }

    uint32_t attentionKvTileCountForQueryTile(const AttentionWorkerState& state) const {
        const uint32_t totalNumKvTiles =
            (state.dispatch.expectedCols + state.dispatch.kvTileRows - 1) /
            state.dispatch.kvTileRows;
        if (!attentionCausal(state)) return totalNumKvTiles;
        const uint32_t queryEnd = state.dispatch.row + std::min(
            state.dispatch.expectedRows,
            (state.queryTileIndex + 1) * state.dispatch.queryTileRows) - 1;
        return std::min(totalNumKvTiles, queryEnd / state.dispatch.kvTileRows + 1);
    }

    uint32_t attentionPhysicalKvTileForIndex(
            const AttentionWorkerState& state, uint32_t ordinal) const {
        const uint32_t numKvTiles = attentionKvTileCountForQueryTile(state);
        if (!attentionKvTileRotation_ || !attentionStreamKv(state) || numKvTiles <= 1) {
            return ordinal;
        }
        const uint32_t dataBands =
            (state.dispatch.expectedCols + state.dispatch.rowsPerBand - 1) /
            state.dispatch.rowsPerBand;
        const uint32_t tilesPerBand = std::max<uint32_t>(
            1, state.dispatch.rowsPerBand / state.dispatch.kvTileRows);
        const uint32_t startBand =
            (state.dispatch.ownerCore + state.dispatch.workerSlot) % dataBands;
        return (ordinal + startBand * tilesPerBand) % numKvTiles;
    }

    uint32_t attentionPhysicalKvTile(const AttentionWorkerState& state) const {
        return attentionPhysicalKvTileForIndex(state, state.kvTileIndex);
    }

    bool attentionStreamKv(const AttentionWorkerState& state) const {
        return state.dispatch.nodeStrideBytes != 0 && state.dispatch.rowsPerBand != 0;
    }

    uint64_t attentionKvHostAddrForTile(const AttentionWorkerState& state,
                                        uint64_t tensorBase,
                                        uint32_t physicalKvTileIndex) const {
        const uint32_t keyBegin = physicalKvTileIndex * state.dispatch.kvTileRows;
        const uint32_t nodeBand = keyBegin / state.dispatch.rowsPerBand;
        const uint32_t rowInBand = keyBegin % state.dispatch.rowsPerBand;
        return tensorBase + static_cast<uint64_t>(nodeBand) *
            state.dispatch.nodeStrideBytes + static_cast<uint64_t>(rowInBand) *
            state.dispatch.headDim * sizeof(float);
    }

    uint64_t attentionKvHostAddr(const AttentionWorkerState& state,
                                 uint64_t tensorBase) const {
        return attentionKvHostAddrForTile(state, tensorBase, state.physicalKvTileIndex);
    }

    void dmaAttentionKvTileToLocal(
            const AttentionWorkerState& state, uint64_t tensorBase,
            uint32_t physicalKvTileIndex, uint64_t localBase, DmaRequestKind kind,
            const DmaConsumerMetadata& consumer,
            std::function<void(bool)> callback) {
        const uint32_t tileRows = attentionKvRowsForTile(state, physicalKvTileIndex);
        const uint32_t firstRow = physicalKvTileIndex * state.dispatch.kvTileRows;
        if (!attentionStreamKv(state) || tileRows <= state.dispatch.rowsPerBand -
                firstRow % state.dispatch.rowsPerBand) {
            globalMem->dma_read_from_host_to_globalmem(
                attentionKvHostAddrForTile(state, tensorBase, physicalKvTileIndex),
                static_cast<uint64_t>(tileRows) * state.dispatch.headDim *
                    sizeof(float),
                localBase, std::move(callback), kind, consumer);
            return;
        }

        const uint32_t segmentCount =
            (firstRow % state.dispatch.rowsPerBand + tileRows +
             state.dispatch.rowsPerBand - 1) / state.dispatch.rowsPerBand;
        auto pending = std::make_shared<uint32_t>(segmentCount);
        auto allOk = std::make_shared<bool>(true);
        uint32_t rowsIssued = 0;
        while (rowsIssued < tileRows) {
            const uint32_t globalRow = firstRow + rowsIssued;
            const uint32_t nodeBand = globalRow / state.dispatch.rowsPerBand;
            const uint32_t rowInBand = globalRow % state.dispatch.rowsPerBand;
            const uint32_t rows = std::min(
                tileRows - rowsIssued,
                state.dispatch.rowsPerBand - rowInBand);
            const uint64_t src = tensorBase +
                static_cast<uint64_t>(nodeBand) * state.dispatch.nodeStrideBytes +
                static_cast<uint64_t>(rowInBand) * state.dispatch.headDim *
                    sizeof(float);
            const uint64_t dst = localBase +
                static_cast<uint64_t>(rowsIssued) * state.dispatch.headDim *
                    sizeof(float);
            const uint64_t bytes = static_cast<uint64_t>(rows) *
                state.dispatch.headDim * sizeof(float);
            globalMem->dma_read_from_host_to_globalmem(
                src, bytes, dst,
                [pending, allOk, callback](bool ok) {
                    *allOk = *allOk && ok;
                    if (--*pending == 0) callback(*allOk);
                }, kind, consumer);
            rowsIssued += rows;
        }
    }

    void loadAttentionKvPairToLocal(
            const AttentionWorkerState& state, uint32_t physicalKvTileIndex,
            uint32_t targetQueryTile, uint32_t targetKvTileIndex,
            uint64_t kLocal, uint64_t vLocal, DmaRequestKind kind,
            std::function<void(bool)> callback,
            std::function<void(bool)> kReadyCallback = {}) {
        if (attentionKvDistributionEnable_) {
            AttentionKvRequest request;
            request.generation = state.generation;
            request.jobTag = state.dispatch.tag;
            request.queryGroup = targetQueryTile /
                attentionKvQueryGroupSize(state);
            request.kvTileIndex = targetKvTileIndex;
            request.physicalKvTileIndex = physicalKvTileIndex;
            request.numKvTiles = attentionKvTileCountForQueryTile(state);
            request.kvLength = state.dispatch.expectedCols;
            request.kvTileRows = state.dispatch.kvTileRows;
            request.tileRows = attentionKvRowsForTile(state, physicalKvTileIndex);
            request.rowsPerBand = state.dispatch.rowsPerBand;
            request.headDim = state.dispatch.headDim;
            request.nodeStrideBytes = state.dispatch.nodeStrideBytes;
            request.kAddr = state.dispatch.kAddr;
            request.vAddr = state.dispatch.vAddr;
            request.kDstAddr = kLocal;
            request.vDstAddr = vLocal;
            if (!groupCtrl->requestAttentionKvPair(
                    request, callback, std::move(kReadyCallback))) {
                callback(false);
            }
            return;
        }

        auto pending = std::make_shared<uint32_t>(2);
        auto allOk = std::make_shared<bool>(true);
        auto complete = [pending, allOk, callback](bool ok) {
            *allOk = *allOk && ok;
            if (--*pending == 0) callback(*allOk);
        };
        dmaAttentionKvTileToLocal(
            state, state.dispatch.kAddr, physicalKvTileIndex, kLocal, kind,
            attentionDmaConsumerMetadata(
                state, targetQueryTile, targetKvTileIndex,
                DmaOperand::AttentionK),
            [complete, kReadyCallback](bool ok) {
                if (kReadyCallback) kReadyCallback(ok);
                complete(ok);
            });
        dmaAttentionKvTileToLocal(
            state, state.dispatch.vAddr, physicalKvTileIndex, vLocal, kind,
            attentionDmaConsumerMetadata(
                state, targetQueryTile, targetKvTileIndex,
                DmaOperand::AttentionV), complete);
    }

    DmaConsumerMetadata attentionDmaConsumerMetadata(
            const AttentionWorkerState& state, uint32_t targetQueryTile,
            uint32_t targetKvTileIndex, DmaOperand operand) const {
        DmaConsumerMetadata metadata;
        metadata.valid = 1;
        // A composite GQA job is the cross-worker identity of one K/V head.
        // Local generations may differ when managers dispatch queued groups in
        // different orders, so they cannot safely key shared-node coalescing.
        metadata.jobId = state.dispatch.jobId;
        metadata.worker = static_cast<uint32_t>(coreID);
        metadata.consumerQueryTile = state.queryTileIndex;
        metadata.consumerKvTileIndex = state.kvTileIndex;
        metadata.targetQueryTile = targetQueryTile;
        metadata.targetKvTileIndex = targetKvTileIndex;
        metadata.operand = operand;
        return metadata;
    }

    void traceAttentionMilestone(const char* role, const char* stage,
                                 const char* status, uint64_t jobId,
                                 uint64_t tag, int64_t queryTileIndex = -1,
                                 int64_t physicalKvTileIndex = -1) {
        if (!attentionMilestoneTrace_) return;
        output->output(
            "[ATTENTION_MILESTONE] stage=%s status=%s sst_tick=%" PRIu64
            " rocc_cycle=%" PRIu64 " core=%" PRIu64 " role=%s job=%" PRIu64
            " tag=%" PRIu64 " query_tile=%" PRId64 " kv_tile=%" PRId64 "\n",
            stage, status, getCurrentSimCycle(), LastTickCycle, coreID, role,
            jobId, tag, queryTileIndex, physicalKvTileIndex);
    }

    bool isFinalAttentionTile(const AttentionWorkerState& state) const {
        return state.queryTileIndex + 1 == attentionQueryTileCount(state) &&
            state.kvTileIndex + 1 == attentionKvTileCountForQueryTile(state);
    }

    void traceAttentionTileCompletion(const char* stage,
                                      const char* finalStage,
                                      const AttentionWorkerState& state) {
        if (attentionTileTrace_) {
            traceAttentionMilestone(
                "worker", stage, "done", state.dispatch.jobId,
                state.dispatch.tag, state.queryTileIndex, state.kvTileIndex);
        }
        if (isFinalAttentionTile(state)) {
            traceAttentionMilestone(
                "worker", finalStage, "done", state.dispatch.jobId,
                state.dispatch.tag, state.queryTileIndex, state.kvTileIndex);
        }
    }

    uint32_t attentionKvLocalKey(const AttentionWorkerState& state,
                                 uint32_t keyInTile) const {
        return attentionStreamKv(state) ? keyInTile :
            state.physicalKvTileIndex * state.dispatch.kvTileRows + keyInTile;
    }

    void finishAttentionWorker(bool ok) {
        if (!attentionWorker_) return;
        const uint64_t generation = attentionWorker_->generation;
        if (!ok && attentionKvDistributionEnable_ && groupCtrl != nullptr) {
            groupCtrl->cancelAttentionKvGeneration(generation);
        }
        if ((attentionWorker_->qkRowBurstStorageActive ||
             attentionWorker_->attentionOAccumulatorStorageActive) &&
            workerCommandProcessor != nullptr &&
            !workerCommandProcessor->cancelAttentionTileStorage(
                attentionWorker_->generation)) {
            output->output(
                "Attention QK tile storage failed to drain on worker failure "
                "core=%" PRIu64 " generation=%" PRIu64 "\n",
                coreID, attentionWorker_->generation);
        }
        if (attentionClusterEnable_ && attentionCluster_) {
            if (ok && (!attentionCluster_->drained() ||
                       !attentionOAccumulator_.drained() ||
                       !globalMem->attentionGenerationDrained(generation))) {
                statAttentionClusterIllegalTransitions_->addData(1);
                ok = false;
            }
            if (ok && !globalMem->retireAttentionGeneration(generation)) {
                statAttentionClusterIllegalTransitions_->addData(1);
                ok = false;
            }
            if (!ok) {
                if (sfu != nullptr) {
                    sfu->cancelAttentionClusterGeneration(generation);
                }
                const uint32_t cancelledO =
                    attentionOAccumulator_.cancelGeneration(generation);
                if (cancelledO != 0) {
                    statAttentionClusterOContextCancelled_->addData(cancelledO);
                }
                const uint32_t cancelledMemory =
                    globalMem->cancelAttentionGeneration(generation);
                if (cancelledMemory != 0) {
                    statAttentionClusterMemoryRequestsCancelled_->addData(
                        cancelledMemory);
                }
            }
            const uint32_t cancelledBankRefs = attentionCluster_->liveBankRefs();
            const uint32_t cancelled = attentionCluster_->cancel();
            if (cancelled != 0) {
                statAttentionClusterContextsCancelled_->addData(cancelled);
            }
            if (cancelledBankRefs != 0) {
                statAttentionClusterBankRefsCancelled_->addData(cancelledBankRefs);
            }
            if (!attentionCluster_->drained()) {
                statAttentionClusterIllegalTransitions_->addData(1);
                ok = false;
            }
        }
        const int64_t completedQueryTile = ok && attentionWorker_->queryTileIndex > 0 ?
            static_cast<int64_t>(attentionWorker_->queryTileIndex - 1) :
            static_cast<int64_t>(attentionWorker_->queryTileIndex);
        const int64_t completedKvTile = ok && attentionWorker_->kvTileIndex > 0 ?
            static_cast<int64_t>(attentionWorker_->kvTileIndex - 1) :
            static_cast<int64_t>(attentionWorker_->kvTileIndex);
        traceAttentionMilestone(
            "worker", "worker_complete", ok ? "done" : "fail",
            attentionWorker_->dispatch.jobId, attentionWorker_->dispatch.tag,
            completedQueryTile, completedKvTile);
        if (!ok) {
            output->output(
                "Attention worker failure core=%" PRIu64 " phase=%u query_tile=%u "
                "kv_tile=%u phaseSliceIndex=%u index=%u\n",
                coreID, static_cast<unsigned>(attentionWorker_->phase),
                attentionWorker_->queryTileIndex, attentionWorker_->physicalKvTileIndex,
                attentionWorker_->phaseSliceIndex, attentionWorker_->index);
        }
        ControlTransportMessage completion = attentionWorker_->dispatch;
        completion.kind = ControlTransportMessageKind::AttentionComplete;
        completion.sendCycle = getCurrentSimCycle();
        completion.value = ok ? 1.0 : 0.0;
        attentionWorker_.reset();
        std::fill(attentionArrayPending_.begin(), attentionArrayPending_.end(), 0);
        if (!globalMem->sendControlMessage(completion.ownerCore, completion)) {
            output->verbose(CALL_INFO, 1, 0, "Attention completion send failed\n");
        }
        if (ok && attentionWorkerClusterQkAhead_) {
            attentionWorker_ = std::move(attentionWorkerClusterQkAhead_);
            startAttentionReuseWindowSoftmax();
        } else {
            if (!ok && attentionWorkerClusterQkAhead_) {
                ControlTransportMessage rejected =
                    attentionWorkerClusterQkAhead_->dispatch;
                rejected.kind = ControlTransportMessageKind::AttentionComplete;
                rejected.value = 0.0;
                rejected.sendCycle = getCurrentSimCycle();
                globalMem->sendControlMessage(rejected.ownerCore, rejected);
                attentionWorkerClusterQkAhead_.reset();
            }
            if (attentionPendingDispatches_.empty()) return;
            ControlTransportMessage next = attentionPendingDispatches_.front();
            attentionPendingDispatches_.pop_front();
            startAttentionWorker(next);
        }
    }

    void issueAttentionLocalTransferChunk() {
        if (!attentionWorker_ || attentionWorker_->localInflight ||
            !attentionWorker_->localCallback) return;
        AttentionWorkerState& state = *attentionWorker_;
        const size_t remaining = state.localLength - state.localOffset;
        if (remaining == 0) {
            auto callback = std::move(state.localCallback);
            const std::vector<uint8_t> bytes = state.transferBytes;
            callback(true, bytes);
            return;
        }
        const size_t chunk = std::min(remaining, globalMem->localMaxRequestBytes());
        const uint64_t tag = attentionTransferTag();
        const uint64_t generation = state.generation;
        bool accepted = false;
        if (!state.localWrite) {
            accepted = globalMem->localReadAsync(
                state.localAddr + state.localOffset, chunk, LocalMemoryClient::RoCC, tag,
                [this, tag, chunk, generation](bool ok, uint64_t callbackTag,
                                   const std::vector<uint8_t>& bytes) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    AttentionWorkerState& callbackState = *attentionWorker_;
                    callbackState.localInflight = false;
                    if (!ok || callbackTag != tag || bytes.size() != chunk) {
                        auto callback = std::move(callbackState.localCallback);
                        callback(false, {});
                        return;
                    }
                    callbackState.transferBytes.insert(
                        callbackState.transferBytes.end(), bytes.begin(), bytes.end());
                    callbackState.localOffset += chunk;
                    issueAttentionLocalTransferChunk();
                });
        } else {
            std::vector<uint8_t> bytes(
                state.transferBytes.begin() + state.localOffset,
                state.transferBytes.begin() + state.localOffset + chunk);
            accepted = globalMem->localWriteAsync(
                state.localAddr + state.localOffset, bytes, LocalMemoryClient::RoCC, tag,
                [this, tag, chunk, generation](bool ok, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    AttentionWorkerState& callbackState = *attentionWorker_;
                    callbackState.localInflight = false;
                    if (!ok || callbackTag != tag) {
                        auto callback = std::move(callbackState.localCallback);
                        callback(false, {});
                        return;
                    }
                    callbackState.localOffset += chunk;
                    issueAttentionLocalTransferChunk();
                });
        }
        if (accepted) state.localInflight = true;
    }

    void attentionLocalRead(uint64_t addr, size_t length,
                            std::function<void(bool, const std::vector<uint8_t>&)> callback) {
        AttentionWorkerState& state = *attentionWorker_;
        state.localAddr = addr;
        state.localLength = length;
        state.localOffset = 0;
        state.localInflight = false;
        state.localWrite = false;
        state.transferBytes.clear();
        state.localCallback = std::move(callback);
        issueAttentionLocalTransferChunk();
    }

    void attentionLocalWrite(uint64_t addr, const std::vector<uint8_t>& bytes,
                             std::function<void(bool)> callback) {
        AttentionWorkerState& state = *attentionWorker_;
        state.localAddr = addr;
        state.localLength = bytes.size();
        state.localOffset = 0;
        state.localInflight = false;
        state.localWrite = true;
        state.transferBytes = bytes;
        state.localCallback = [callback = std::move(callback)](
                                  bool ok, const std::vector<uint8_t>&) { callback(ok); };
        issueAttentionLocalTransferChunk();
    }

    std::vector<double> attentionBytesToDoubles(const std::vector<uint8_t>& bytes) const {
        const size_t count = bytes.size() / sizeof(float);
        std::vector<double> values(count, 0.0);
        for (size_t i = 0; i < count; ++i) {
            float value = 0.0f;
            std::memcpy(&value, bytes.data() + i * sizeof(float), sizeof(float));
            values[i] = value;
        }
        return values;
    }

    std::vector<uint8_t> attentionDoublesToBytes(const std::vector<double>& values) const {
        std::vector<uint8_t> bytes(values.size() * sizeof(float));
        for (size_t i = 0; i < values.size(); ++i) {
            const float value = static_cast<float>(values[i]);
            std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(float));
        }
        return bytes;
    }

    std::vector<uint8_t> attentionFloatsToBytes(
        const std::vector<float>& values) const {
        std::vector<uint8_t> bytes(values.size() * sizeof(float));
        if (!bytes.empty()) {
            std::memcpy(bytes.data(), values.data(), bytes.size());
        }
        return bytes;
    }

    void beginAttentionQueryTile() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (attentionClusterEnable_) {
            state.clusterAheadEnabledForTile = false;
            state.clusterAheadPromotionWaiting = false;
        }
        if (attentionClusterEnable_ &&
            state.queryTileIndex % attentionClusterConfig_.groupSize == 0) {
            state.clusterQueryLoaded.fill(false);
        }
        if (!selectAttentionQueryStorage(state)) {
            finishAttentionWorker(false);
            return;
        }
        if (attentionOAccumulatorCBuffer_) {
            const uint64_t qkScratchBytes =
                static_cast<uint64_t>(attentionQueryRows(state)) * 16 * sizeof(float);
            const uint32_t accumulatorRows = attentionQueryRows(state) *
                attentionOutputSliceCount(state);
            if (!workerCommandProcessor->beginAttentionStorageSession(
                    qkScratchBytes, accumulatorRows, 16 * sizeof(float),
                    state.generation)) {
                finishAttentionWorker(false);
                return;
            }
            state.attentionOAccumulatorStorageActive = true;
        }
        const int32_t crossQueryBuffer = attentionKvDoubleBuffer_ &&
            attentionKvCrossQueryPrefetch_ && state.queryTileIndex != 0
            ? findAttentionKvBuffer(0,
                attentionPhysicalKvTileForIndex(state, 0), state.queryTileIndex)
            : -1;
        state.phase = AttentionWorkerPhase::LoadingQ;
        state.phaseSliceIndex = 0;
        state.kvTileIndex = 0;
        state.physicalKvTileIndex = attentionPhysicalKvTile(state);
        state.activeKvBuffer = crossQueryBuffer >= 0
            ? static_cast<uint32_t>(crossQueryBuffer) : 0;
        state.kLocal = state.kLocalBuffers[state.activeKvBuffer];
        state.vLocal = state.vLocalBuffers[state.activeKvBuffer];
        if (crossQueryBuffer < 0) {
            for (auto& buffer : state.kvBuffers) buffer = {};
        } else {
            for (uint32_t buffer = 0; buffer < state.kvBuffers.size(); ++buffer) {
                if (static_cast<int32_t>(buffer) != crossQueryBuffer) {
                    state.kvBuffers[buffer] = {};
                }
            }
        }
        state.attentionWaitingKvBuffer = UINT32_MAX;
        state.attentionWaitingForPrefetch = false;
        const uint64_t generation = state.generation;
        globalMem->dma_read_from_host_to_globalmem(
            state.dispatch.qAddr + static_cast<uint64_t>(state.queryTileIndex) *
                state.dispatch.queryTileRows * state.dispatch.headDim * sizeof(float),
            static_cast<uint64_t>(attentionQueryRows(state)) *
                state.dispatch.headDim * sizeof(float),
            state.qLocal, [this, generation](bool ok) {
                if (!attentionCallbackGenerationMatches(generation)) return;
                if (!ok) {
                    finishAttentionWorker(false);
                    return;
                }
                if (attentionClusterEnable_) {
                    attentionWorker_->clusterQueryLoaded[
                        attentionWorker_->queryTileIndex %
                            attentionClusterConfig_.groupSize] = true;
                }
                recordAttentionInterTilePhase(AttentionInterTilePhase::QueryLoad);
                loadAttentionKvTile();
            }, DmaRequestKind::AttentionQuery);
    }

    void beginAttentionKvTile() {
        if (!attentionWorker_) return;
        // Every sequential D128 QK tile starts with the low K64 reduction.
        // Some tile transitions bypass the ahead-operand promotion path, so
        // resetting only during promotion can silently skip this first half.
        if (attentionSequential64Enable_) {
            attentionWorker_->qkReductionSlice = 0;
            attentionWorker_->sequentialKTilePayload.clear();
            attentionWorker_->sequentialPPayload.clear();
            attentionWorker_->sequentialOPayload.clear();
        }
        if (attentionClusterEnable_) {
            if (attentionClusterQkMatrixAheadMatchesCurrent()) {
                if (attentionWorker_->clusterQkMatrixAhead.phase !=
                        AttentionClusterQkMatrixAheadPhase::Ready) {
                    attentionWorker_->clusterQkMatrixAheadPromotionWaiting = true;
                    return;
                }
                attentionWorker_->activeOperandBank =
                    attentionWorker_->clusterQkMatrixAhead.operandBank;
                recordAttentionKvOperandRelease(true);
            }
            attentionWorker_->clusterAheadEnabledForTile = false;
            attentionWorker_->clusterPvOperandBank =
                attentionWorker_->kvTileIndex % attentionOperandContextBanks_;
        }
        invalidateAttentionPvInputResidency();
        if (!attentionPvVTileGroupRetention_ ||
            !attentionVTileMatchesCurrentGroup(*attentionWorker_)) {
            invalidateAttentionVTileStaging();
        }
        attentionWorker_->vTileBufferWaiting = false;
        attentionWorker_->vTileBufferBypassWait = false;
        attentionWorker_->vTileBufferCurrentWaitTicks = 0;
        transitionAttentionTilePipeline(AttentionTilePipelinePhase::QLocalRead);
        attentionLocalRead(attentionWorker_->qLocal,
            static_cast<uint64_t>(attentionQueryRows(*attentionWorker_)) *
                attentionWorker_->dispatch.headDim * sizeof(float),
            [this](bool readOk, const std::vector<uint8_t>& bytes) {
                if (!attentionWorker_ || !readOk) {
                    finishAttentionWorker(false);
                    return;
                }
                recordAttentionInterTilePhase(AttentionInterTilePhase::QLocalRead);
                transitionAttentionTilePipeline(
                    AttentionTilePipelinePhase::QkMatrixProgram);
                const std::vector<double> q = attentionBytesToDoubles(bytes);
                if (attentionClusterEnable_) {
                    AttentionWorkerState& state = *attentionWorker_;
                    state.qPayload = q;
                    state.clusterQkReductionSlice = 0;
                    state.clusterQkWave = 0;
                    state.clusterQkPair = 0;
                    if (!reserveAttentionClusterQkAndScore()) {
                        state.index = 101;
                        finishAttentionWorker(false);
                        return;
                    }
                    if (!ensureAttentionClusterOContext()) {
                        state.index = 102;
                        finishAttentionWorker(false);
                        return;
                    }
                    beginAttentionClusterKTile();
                    return;
                }
                if (attentionQkDataflowTranspose_) {
                    attentionWorker_->qPayload = q;
                    attentionWorker_->phaseSliceIndex = 0;
                    beginAttentionQkTransposedKvSubtile();
                    return;
                }
                attentionWorker_->qPayload = q;
                if (attentionSequential64Enable_ &&
                    !buildAttentionSequentialQkMatrix(*attentionWorker_)) {
                    finishAttentionWorker(false);
                    return;
                }
                attentionWorker_->phase = AttentionWorkerPhase::QkProgramMatrix;
                attentionWorker_->phaseSliceIndex = 0;
                attentionWorker_->index = 0;
                beginAttentionQkKvSubtile();
            });
    }

    bool attentionClusterCallbackMatches(
        AttentionClusterContextKind kind, int32_t slot,
        const AttentionClusterTag& tag) {
        if (!attentionCluster_ || slot < 0 ||
            !attentionCluster_->callbackMatches(
                kind, static_cast<uint32_t>(slot), tag)) {
            statAttentionClusterStaleCallbacks_->addData(1);
            return false;
        }
        return true;
    }

    void beginAttentionClusterKTile() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.phase = AttentionWorkerPhase::QkProgramMatrix;
        const uint32_t halves = attentionClusterHeadHalves(state);
        if (state.clusterQkReductionSlice == halves) {
            recordAttentionInterTilePhase(
                AttentionInterTilePhase::QkMatrixProgram);
            transitionAttentionTilePipeline(
                AttentionTilePipelinePhase::QkInputProgram);
            state.phase = AttentionWorkerPhase::QkProgramInputs;
            state.clusterQkWave = 0;
            state.clusterQkPair = 0;
            programAttentionClusterQPair();
            return;
        }
        if (state.clusterQkReductionSlice != 0 || !state.arrayPayload.empty()) {
            programAttentionClusterKHalf();
            return;
        }
        const uint64_t generation = state.generation;
        const AttentionClusterTag clusterTag = state.clusterTileTag;
        const int32_t contextSlot = state.clusterQkContext;
        attentionLocalRead(
            state.kLocal + static_cast<uint64_t>(attentionKvLocalKey(state, 0)) *
                state.dispatch.headDim * sizeof(float),
            static_cast<uint64_t>(attentionKeyCols(state)) *
                state.dispatch.headDim * sizeof(float),
            [this, generation, clusterTag, contextSlot](
                bool ok, const std::vector<uint8_t>& bytes) {
                if (!attentionCallbackGenerationMatches(generation)) return;
                if (!ok || attentionWorker_->clusterQkReductionSlice != 0 ||
                    !attentionClusterCallbackMatches(
                        AttentionClusterContextKind::Qk, contextSlot,
                        clusterTag)) {
                    if (attentionWorker_) finishAttentionWorker(false);
                    return;
                }
                attentionWorker_->arrayPayload = attentionBytesToDoubles(bytes);
                programAttentionClusterKHalf();
            });
    }

    void programAttentionClusterKHalf() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t half = state.clusterQkReductionSlice;
        const uint32_t halves = attentionClusterHeadHalves(state);
        if (half >= halves) {
            beginAttentionClusterKTile();
            return;
        }
        const uint32_t keyCols = attentionKeyCols(state);
        std::vector<double> matrix(
            static_cast<size_t>(arrayOutputSize) * arrayInputSize, 0.0);
        for (uint32_t key = 0; key < keyCols; ++key) {
            for (uint32_t dim = 0; dim < static_cast<uint32_t>(arrayInputSize);
                 ++dim) {
                matrix[static_cast<size_t>(key) * arrayInputSize + dim] =
                    state.arrayPayload[static_cast<size_t>(key) *
                        state.dispatch.headDim + half * arrayInputSize + dim];
            }
        }
        std::vector<uint32_t> arrayIds;
        arrayIds.reserve(attentionClusterQkLanes(state));
        for (uint32_t row = 0; row < attentionClusterQkLanes(state); ++row)
            arrayIds.push_back(attentionClusterQkArray(row, half));
        const uint64_t generation = state.generation;
        const AttentionClusterTag clusterTag = state.clusterTileTag;
        const int32_t contextSlot = state.clusterQkContext;
        const uint64_t transferTag = attentionTransferTag();
        if (!programAttentionClusterKTileAsync(
                arrayIds, matrix, transferTag,
                [this, generation, clusterTag, contextSlot, half, transferTag,
                 bytesCount = matrix.size() * sizeof(float)](
                    bool ok, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok || callbackTag != transferTag ||
                        !attentionWorker_ ||
                        attentionWorker_->clusterQkReductionSlice != half ||
                        !attentionClusterCallbackMatches(
                            AttentionClusterContextKind::Qk, contextSlot,
                            clusterTag)) {
                        if (attentionWorker_) finishAttentionWorker(false);
                        return;
                    }
                    statAttentionClusterKTileBroadcasts_->addData(1);
                    statAttentionClusterKTileBytes_->addData(bytesCount);
                    statAttentionQkMatrixBroadcasts_->addData(1);
                    ++attentionWorker_->clusterQkReductionSlice;
                    if (attentionWorker_->clusterQkReductionSlice ==
                            attentionClusterHeadHalves(*attentionWorker_)) {
                        attentionWorker_->arrayPayload.clear();
                        recordAttentionKvOperandRelease(true);
                    }
                    beginAttentionClusterKTile();
                })) {
            scheduleAttentionClusterOwnerRetry(
                [this, generation, clusterTag, contextSlot, half]() {
                    if (!attentionCallbackGenerationMatches(generation) ||
                        !attentionWorker_ ||
                        attentionWorker_->clusterQkReductionSlice != half ||
                        !attentionClusterCallbackMatches(
                            AttentionClusterContextKind::Qk, contextSlot,
                            clusterTag)) return;
                    programAttentionClusterKHalf();
                });
        }
    }

    void programAttentionClusterQPair() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t halves = attentionClusterHeadHalves(state);
        const uint32_t lanes = attentionClusterQkLanes(state);
        const uint32_t rows = attentionClusterWaveRows(
            state, state.clusterQkWave, lanes);
        const uint32_t physicalInputs = rows * halves;
        if (state.clusterQkPair == physicalInputs) {
            recordAttentionInterTilePhase(
                AttentionInterTilePhase::QkInputProgram);
            transitionAttentionTilePipeline(
                AttentionTilePipelinePhase::QkComputeReadout);
            state.attentionQkInputProgrammingComplete = true;
            startAttentionQkCompute(physicalInputs);
            return;
        }
        const uint32_t physical = state.clusterQkPair;
        const uint32_t query = attentionClusterWaveRowBegin(
            state.clusterQkWave, lanes) + physical / halves;
        const uint32_t half = physical % halves;
        // Admission requires full Br=16 blocks; fail closed if a future
        // dispatch path reaches this slicer with a partial block.
        if (query >= attentionQueryRowsForTile(state, state.queryTileIndex)) {
            finishAttentionWorker(false);
            return;
        }
        const uint32_t wave = state.clusterQkWave;
        const uint64_t generation = state.generation;
        const AttentionClusterTag clusterTag = state.clusterTileTag;
        const int32_t contextSlot = state.clusterQkContext;
        std::vector<double> input(
            state.qPayload.begin() +
                static_cast<size_t>(query) * state.dispatch.headDim +
                    half * arrayInputSize,
            state.qPayload.begin() +
                static_cast<size_t>(query) * state.dispatch.headDim +
                    (half + 1) * arrayInputSize);
        std::vector<uint32_t> arrayIds = {
            attentionClusterQkArray(query, half)};
        const uint64_t transferTag = attentionTransferTag();
        if (!programAttentionClusterQPairAsync(
                arrayIds, input, transferTag,
                [this, generation, clusterTag, contextSlot, physical, wave,
                 transferTag, bytes = input.size() * sizeof(float)](
                    bool ok, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok ||
                        callbackTag != transferTag ||
                        attentionWorker_->clusterQkWave != wave ||
                        attentionWorker_->clusterQkPair != physical ||
                        !attentionClusterCallbackMatches(
                            AttentionClusterContextKind::Qk, contextSlot,
                            clusterTag)) {
                        if (attentionWorker_) finishAttentionWorker(false);
                        return;
                    }
                    statAttentionClusterQPairMulticasts_->addData(1);
                    statAttentionClusterQPairBytes_->addData(bytes);
                    attentionWorker_->clusterQkPair += 1;
                    programAttentionClusterQPair();
                })) {
            scheduleAttentionClusterOwnerRetry(
                [this, generation, clusterTag, contextSlot, physical, wave]() {
                    if (!attentionCallbackGenerationMatches(generation) ||
                        !attentionWorker_ ||
                        attentionWorker_->clusterQkWave != wave ||
                        attentionWorker_->clusterQkPair != physical ||
                        !attentionClusterCallbackMatches(
                            AttentionClusterContextKind::Qk,
                            contextSlot, clusterTag)) return;
                    programAttentionClusterQPair();
                });
        }
    }

    void readAttentionClusterQkOutput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t lanes = attentionClusterQkLanes(state);
        const uint32_t waveEnd = attentionClusterWaveRowBegin(
            state.clusterQkWave, lanes) + attentionClusterWaveRows(
                state, state.clusterQkWave, lanes);
        if (state.index == waveEnd) {
            state.clusterQkWave += 1;
            if (state.clusterQkWave < attentionClusterQkWaves(state)) {
                state.clusterQkPair = 0;
                state.phase = AttentionWorkerPhase::QkProgramInputs;
                transitionAttentionTilePipeline(
                    AttentionTilePipelinePhase::QkInputProgram);
                programAttentionClusterQPair();
                return;
            }
            if (!releaseAttentionClusterQk()) {
                statAttentionClusterIllegalTransitions_->addData(1);
                finishAttentionWorker(false);
                return;
            }
            maybeStartAttentionClusterQkMatrixLookahead();
            maybeStartAttentionClusterPvMatrixLookahead();
            state.clusterAheadEnabledForTile = true;
            recordAttentionInterTilePhase(
                AttentionInterTilePhase::QkComputeReadout);
            transitionAttentionTilePipeline(
                AttentionTilePipelinePhase::Softmax);
            statAttentionWorkerQkTileCompleteTick_->addData(getCurrentSimCycle());
            traceAttentionTileCompletion(
                "qk_tile_complete", "final_qk_tile_complete", state);
            finishAttentionInterTile();
            beginAttentionSoftmax();
            return;
        }
        const uint32_t query = state.index;
        const uint32_t wave = state.clusterQkWave;
        const uint32_t halves = attentionClusterHeadHalves(state);
        std::vector<uint32_t> arrays(halves);
        for (uint32_t half = 0; half < halves; ++half)
            arrays[half] = attentionClusterQkArray(query, half);
        const uint64_t generation = state.generation;
        const AttentionClusterTag clusterTag = state.clusterTileTag;
        const int32_t scoreSlot = state.clusterScoreContext;
        const uint64_t transferTag = attentionTransferTag();
        if (!readAttentionClusterScorePairAsync(
                arrays, transferTag,
                [this, generation, clusterTag, scoreSlot, query, wave,
                 halves, transferTag](
                    bool ok, uint64_t callbackTag,
                    const std::vector<double>& values) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok ||
                        callbackTag != transferTag ||
                        values.size() != static_cast<size_t>(halves) *
                            attentionClusterConfig_.arrayOutputs ||
                        attentionWorker_->clusterQkWave != wave ||
                        attentionWorker_->index != query ||
                        !attentionClusterCallbackMatches(
                            AttentionClusterContextKind::Score, scoreSlot,
                            clusterTag)) {
                        finishAttentionWorker(false);
                        return;
                    }
                    const uint32_t keyCols =
                        attentionKeyCols(*attentionWorker_);
                    const size_t base = static_cast<size_t>(query) * keyCols;
                    std::vector<float> beat(keyCols, 0.0f);
                    for (uint32_t key = 0; key < keyCols; ++key) {
                        double sum = 0.0;
                        for (uint32_t half = 0; half < halves; ++half) {
                            sum += values[static_cast<size_t>(half) *
                                attentionClusterConfig_.arrayOutputs + key];
                        }
                        beat[key] = static_cast<float>(sum);
                    }
                    if (!sfu->writeAttentionScoreBeatAsync(
                            static_cast<uint32_t>(scoreSlot), clusterTag, base,
                            beat, transferTag,
                            [this, generation, clusterTag, scoreSlot, query,
                             wave, transferTag, bytes = beat.size() * sizeof(float)](
                                bool writeOk, uint64_t callbackWriteTag) {
                                if (!attentionCallbackGenerationMatches(generation)) return;
                                if (!writeOk || callbackWriteTag != transferTag ||
                                    attentionWorker_->clusterQkWave != wave ||
                                    attentionWorker_->index != query ||
                                    !attentionClusterCallbackMatches(
                                        AttentionClusterContextKind::Score,
                                        scoreSlot, clusterTag)) {
                                    finishAttentionWorker(false);
                                    return;
                                }
                                statAttentionClusterScoreBeats_->addData(1);
                                statAttentionClusterScoreBytes_->addData(bytes);
                                ++attentionWorker_->index;
                                readAttentionClusterQkOutput();
                            })) {
                        scheduleAttentionClusterOwnerRetry(
                            [this, generation, clusterTag, scoreSlot, query, wave]() {
                                if (!attentionCallbackGenerationMatches(generation) ||
                                    !attentionWorker_ ||
                                    attentionWorker_->clusterQkWave != wave ||
                                    attentionWorker_->index != query ||
                                    !attentionClusterCallbackMatches(
                                        AttentionClusterContextKind::Score,
                                        scoreSlot, clusterTag)) return;
                                readAttentionClusterQkOutput();
                            });
                    }
                })) {
            scheduleAttentionClusterOwnerRetry(
                [this, generation, clusterTag, scoreSlot, query, wave]() {
                    if (!attentionCallbackGenerationMatches(generation) ||
                        !attentionWorker_ ||
                        attentionWorker_->clusterQkWave != wave ||
                        attentionWorker_->index != query ||
                        !attentionClusterCallbackMatches(
                            AttentionClusterContextKind::Score,
                            scoreSlot, clusterTag)) return;
                    readAttentionClusterQkOutput();
                });
        }
    }

    void loadAttentionKvTile() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (attentionClusterEnable_) {
            state.clusterAheadEnabledForTile = false;
        }
        beginAttentionTilePipeline();
        const DmaConsumerMetadata consumerProgress = attentionDmaConsumerMetadata(
            state, state.queryTileIndex, state.kvTileIndex, DmaOperand::Unknown);
        globalMem->dma_update_consumer_progress(
            attentionKvHostAddr(state, state.dispatch.kAddr),
            attentionKvHostAddr(state, state.dispatch.vAddr), consumerProgress);
        if (!attentionStreamKv(state)) {
            recordAttentionInterTilePhase(AttentionInterTilePhase::KvLoad);
            beginAttentionKvTile();
            return;
        }
        if (attentionKvDoubleBuffer_ &&
            (state.kvTileIndex != 0 ||
             (attentionKvCrossQueryPrefetch_ && state.queryTileIndex != 0))) {
            const int32_t buffer = findAttentionKvBuffer(
                state.kvTileIndex, state.physicalKvTileIndex);
            if (buffer < 0) {
                finishAttentionWorker(false);
                return;
            }
            AttentionKvBufferState& descriptor = state.kvBuffers[buffer];
            if (descriptor.kReady) {
                descriptor.consumeTick = getCurrentSimCycle();
                if (descriptor.ready) {
                    statAttentionKvPrefetchHits_->addData(1);
                    descriptor.pairUseRecorded = true;
                    if (state.queryTileIndex != 0 && state.kvTileIndex == 0) {
                        statAttentionKvCrossQueryHits_->addData(1);
                    }
                    statAttentionKvPrefetchReadyLeadTicks_->addData(
                        descriptor.consumeTick - descriptor.readyTick);
                }
                activateAttentionKvPrefetch(static_cast<uint32_t>(buffer));
                return;
            }
            if (descriptor.loadsPending != 0) {
                descriptor.consumeTick = getCurrentSimCycle();
                statAttentionKvPrefetchWaits_->addData(1);
                descriptor.pairUseRecorded = true;
                if (state.queryTileIndex != 0 && state.kvTileIndex == 0) {
                    statAttentionKvCrossQueryWaits_->addData(1);
                }
                state.attentionWaitingForPrefetch = true;
                state.attentionWaitingKvBuffer = static_cast<uint32_t>(buffer);
                return;
            }
            finishAttentionWorker(false);
            return;
        }
        state.attentionKvLoadsPending = 1;
        loadAttentionKvPairToLocal(
            state, state.physicalKvTileIndex, attentionKvGroupOwnerQueryTile(state),
            state.kvTileIndex, state.kLocal, state.vLocal,
            DmaRequestKind::AttentionKv,
            [this, generation = state.generation](bool ok) {
                completeAttentionKvLoad(generation, ok);
            });
    }

    void completeAttentionKvLoad(uint64_t generation, bool ok) {
        if (!attentionCallbackGenerationMatches(generation)) return;
        if (!ok) {
            finishAttentionWorker(false);
            return;
        }
        if (attentionWorker_->attentionKvLoadsPending == 0) {
            finishAttentionWorker(false);
            return;
        }
        attentionWorker_->attentionKvLoadsPending -= 1;
        if (attentionWorker_->attentionKvLoadsPending == 0) {
            const int32_t nextBuffer = findFreeAttentionKvBuffer(
                attentionWorker_->activeKvBuffer);
            if (nextBuffer < 0) {
                finishAttentionWorker(false);
                return;
            }
            launchAttentionKvPrefetch(
                attentionWorker_->kvTileIndex + 1,
                static_cast<uint32_t>(nextBuffer), false);
            recordAttentionInterTilePhase(AttentionInterTilePhase::KvLoad);
            beginAttentionKvTile();
        }
    }

    void launchAttentionKvPrefetch(
        uint32_t ordinal, uint32_t buffer, bool secondLookahead,
        uint32_t queryTileIndex = UINT32_MAX) {
        if (!attentionWorker_ || !attentionKvDoubleBuffer_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t targetQueryTile = queryTileIndex == UINT32_MAX
            ? attentionKvGroupOwnerQueryTile(state) : queryTileIndex;
        if (ordinal >= attentionKvTileCountForQueryTile(state)) return;
        if (buffer >= state.kvBuffers.size()) {
            finishAttentionWorker(false);
            return;
        }
        AttentionKvBufferState& descriptor = state.kvBuffers[buffer];
        if (descriptor.kvTileIndex != UINT32_MAX ||
            descriptor.loadsPending != 0 || descriptor.ready) {
            output->output(
                "Attention KV failure core=%" PRIu64
                " reason=launch_busy current=%" PRIu32 " target=%" PRIu32
                " buffer=%" PRIu32 " desc_tile=%" PRIu32
                " pending=%" PRIu32 " ready=%u\n",
                coreID, state.kvTileIndex, ordinal, buffer,
                descriptor.kvTileIndex, descriptor.loadsPending,
                descriptor.ready ? 1u : 0u);
            finishAttentionWorker(false);
            return;
        }
        descriptor.queryTileIndex = targetQueryTile;
        descriptor.kvTileIndex = ordinal;
        descriptor.physicalKvTileIndex = attentionPhysicalKvTileForIndex(state, ordinal);
        descriptor.loadsPending = 1;
        descriptor.issueTick = getCurrentSimCycle();
        descriptor.readyTick = 0;
        descriptor.consumeTick = 0;
        descriptor.kReady = false;
        descriptor.ready = false;
        descriptor.crossQuery =
            targetQueryTile != attentionKvGroupOwnerQueryTile(state);
        statAttentionKvPrefetchTiles_->addData(1);
        if (secondLookahead) {
            statAttentionKvSecondLookaheadPrefetches_->addData(1);
        }
        if (descriptor.crossQuery) {
            statAttentionKvCrossQueryPrefetches_->addData(1);
        }
        loadAttentionKvPairToLocal(
            state, descriptor.physicalKvTileIndex, targetQueryTile, ordinal,
            state.kLocalBuffers[buffer], state.vLocalBuffers[buffer],
            DmaRequestKind::AttentionKvPrefetch,
            [this, generation = state.generation, buffer](bool ok) {
                completeAttentionKvPrefetch(generation, buffer, ok);
            },
            [this, generation = state.generation, buffer](bool ok) {
                if (!attentionCallbackGenerationMatches(generation)) return;
                if (!ok || buffer >= attentionWorker_->kvBuffers.size()) {
                    output->output(
                        "Attention KV failure core=%" PRIu64
                        " reason=k_ready_callback current=%" PRIu32
                        " buffer=%" PRIu32 " ok=%u\n",
                        coreID, attentionWorker_->kvTileIndex, buffer,
                        ok ? 1u : 0u);
                    finishAttentionWorker(false);
                    return;
                }
                AttentionWorkerState& callbackState = *attentionWorker_;
                callbackState.kvBuffers[buffer].kReady = true;
                maybeStartAttentionClusterQkMatrixLookahead(
                    static_cast<int32_t>(buffer));
                if (callbackState.attentionWaitingForPrefetch &&
                    callbackState.attentionWaitingKvBuffer == buffer &&
                    callbackState.kvBuffers[buffer].kvTileIndex ==
                        callbackState.kvTileIndex &&
                    callbackState.kvBuffers[buffer].physicalKvTileIndex ==
                        callbackState.physicalKvTileIndex) {
                    const uint64_t waitTicks = getCurrentSimCycle() -
                        callbackState.kvBuffers[buffer].consumeTick;
                    statAttentionKvPrefetchWaitTicks_->addData(waitTicks);
                    if (callbackState.kvBuffers[buffer].crossQuery) {
                        statAttentionKvCrossQueryWaitTicks_->addData(waitTicks);
                    }
                    activateAttentionKvPrefetch(buffer);
                }
            });
    }

    void completeAttentionKvPrefetch(
            uint64_t generation, uint32_t buffer, bool ok) {
        if (!attentionCallbackGenerationMatches(generation)) return;
        if (!ok) {
            output->output(
                "Attention KV failure core=%" PRIu64
                " reason=pair_callback current=%" PRIu32
                " buffer=%" PRIu32 "\n",
                coreID, attentionWorker_->kvTileIndex, buffer);
            finishAttentionWorker(false);
            return;
        }
        AttentionWorkerState& state = *attentionWorker_;
        if (buffer >= state.kvBuffers.size()) {
            finishAttentionWorker(false);
            return;
        }
        AttentionKvBufferState& descriptor = state.kvBuffers[buffer];
        if (descriptor.loadsPending == 0) {
            output->output(
                "Attention KV failure core=%" PRIu64
                " reason=completion_without_pending current=%" PRIu32
                " buffer=%" PRIu32 " desc_tile=%" PRIu32 " ready=%u\n",
                coreID, state.kvTileIndex, buffer,
                descriptor.kvTileIndex, descriptor.ready ? 1u : 0u);
            finishAttentionWorker(false);
            return;
        }
        descriptor.loadsPending -= 1;
        if (descriptor.loadsPending != 0) return;
        descriptor.readyTick = getCurrentSimCycle();
        statAttentionKvPrefetchDmaTicks_->addData(
            descriptor.readyTick - descriptor.issueTick);
        descriptor.ready = true;
        maybeStartAttentionClusterQkMatrixLookahead();
        maybeStartAttentionClusterPvMatrixLookahead(
            static_cast<int32_t>(buffer));
        if (descriptor.consuming && state.attentionWaitingForV &&
            state.activeKvBuffer == buffer) {
            statAttentionKvPrefetchWaitTicks_->addData(
                descriptor.readyTick - state.attentionVWaitTick);
            state.attentionWaitingForV = false;
            state.attentionVWaitTick = 0;
            continueAttentionAfterVReady();
            return;
        }
        if (state.attentionWaitingForPrefetch &&
            state.attentionWaitingKvBuffer == buffer) {
            const uint64_t waitTicks = descriptor.readyTick - descriptor.consumeTick;
            statAttentionKvPrefetchWaitTicks_->addData(waitTicks);
            if (descriptor.crossQuery) {
                statAttentionKvCrossQueryWaitTicks_->addData(waitTicks);
            }
            activateAttentionKvPrefetch(buffer);
            return;
        }
        launchAttentionKvSecondLookahead();
    }

    void activateAttentionKvPrefetch(uint32_t buffer) {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (buffer >= state.kvBuffers.size()) {
            finishAttentionWorker(false);
            return;
        }
        AttentionKvBufferState& descriptor = state.kvBuffers[buffer];
        if (!descriptor.kReady || descriptor.kvTileIndex != state.kvTileIndex ||
            descriptor.physicalKvTileIndex != state.physicalKvTileIndex) {
            output->output(
                "Attention KV failure core=%" PRIu64
                " reason=activate_mismatch current=%" PRIu32
                " physical=%" PRIu32 " buffer=%" PRIu32
                " desc_tile=%" PRIu32 " desc_physical=%" PRIu32
                " pending=%" PRIu32 " ready=%u\n",
                coreID, state.kvTileIndex, state.physicalKvTileIndex, buffer,
                descriptor.kvTileIndex, descriptor.physicalKvTileIndex,
                descriptor.loadsPending, descriptor.ready ? 1u : 0u);
            finishAttentionWorker(false);
            return;
        }
        state.activeKvBuffer = buffer;
        state.kLocal = state.kLocalBuffers[state.activeKvBuffer];
        state.vLocal = state.vLocalBuffers[state.activeKvBuffer];
        descriptor.consuming = true;
        state.attentionWaitingForPrefetch = false;
        state.attentionWaitingKvBuffer = UINT32_MAX;
        const uint32_t nextOrdinal = state.kvTileIndex + 1;
        const uint32_t nextTile = nextOrdinal < attentionKvTileCountForQueryTile(state)
            ? attentionPhysicalKvTileForIndex(state, nextOrdinal) : UINT32_MAX;
        if (nextOrdinal < attentionKvTileCountForQueryTile(state) &&
            findAttentionKvBuffer(nextOrdinal, nextTile) < 0) {
            const int32_t nextBuffer = findFreeAttentionKvBuffer(state.activeKvBuffer);
            if (nextBuffer < 0) {
                finishAttentionWorker(false);
                return;
            }
            launchAttentionKvPrefetch(nextOrdinal, static_cast<uint32_t>(nextBuffer), false);
        }
        recordAttentionInterTilePhase(AttentionInterTilePhase::KvLoad);
        beginAttentionKvTile();
    }

    void beginAttentionQkKvSubtile() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.phase == AttentionWorkerPhase::QkProgramMatrix) {
            if (attentionQkMatrixBroadcast_ && state.index == 0) {
                std::vector<uint32_t> arrayIDs(attentionKvSubtileRows(state));
                std::iota(arrayIDs.begin(), arrayIDs.end(), 0);
                const uint64_t tag = attentionTransferTag();
                if (!programAttentionGemmMatrixGroupAsync(
                        arrayIDs, state.arrayPayload, sizeof(float), tag,
                        [this, tag](bool ok, uint64_t callbackTag) {
                            if (!attentionWorker_ || !ok || callbackTag != tag) {
                                finishAttentionWorker(false); return;
                            }
                            statAttentionQkMatrixBroadcasts_->addData(1);
                            attentionWorker_->index = attentionKvSubtileRows(*attentionWorker_);
                            beginAttentionQkKvSubtile();
                        })) finishAttentionWorker(false);
                return;
            }
            if (state.index == attentionKvSubtileRows(state)) {
                recordAttentionInterTilePhase(
                    AttentionInterTilePhase::QkMatrixProgram);
                transitionAttentionTilePipeline(
                    AttentionTilePipelinePhase::QkInputProgram);
                state.phase = AttentionWorkerPhase::QkProgramInputs;
                state.index = 0;
                state.arraysPending = 0;
                state.attentionQkInputProgrammingComplete = false;
                if (attentionQkInputPipeline_) {
                    resetAttentionQkInputPipeline();
                    pumpAttentionQkInputPipeline();
                } else {
                    programAttentionQkInput();
                }
                return;
            }
            const uint32_t arrayId = state.index;
            const uint64_t tag = attentionTransferTag();
            if (!programAttentionGemmMatrixAsync(arrayId, state.arrayPayload, sizeof(float), tag,
                    [this, tag](bool ok, uint64_t callbackTag) {
                        if (!attentionWorker_ || !ok || callbackTag != tag) {
                            finishAttentionWorker(false); return;
                        }
                        attentionWorker_->index += 1;
                        beginAttentionQkKvSubtile();
                    })) finishAttentionWorker(false);
            return;
        }
    }

    void launchAttentionCrossQueryPrefetch() {
        if (!attentionWorker_ || !attentionKvDoubleBuffer_ ||
            !attentionKvCrossQueryPrefetch_ || !attentionStreamKv(*attentionWorker_)) {
            return;
        }
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t nextQueryTile = state.queryTileIndex + 1;
        if (nextQueryTile >= attentionQueryTileCount(state)) return;
        const uint32_t firstTile = attentionPhysicalKvTileForIndex(state, 0);
        if (findAttentionKvBuffer(0, firstTile, nextQueryTile) >= 0) return;
        const int32_t freeBuffer = findFreeAttentionKvBuffer(state.activeKvBuffer);
        if (freeBuffer < 0) return;
        const uint32_t buffer = static_cast<uint32_t>(freeBuffer);
        AttentionKvBufferState& candidate = state.kvBuffers[buffer];
        if (candidate.queryTileIndex != UINT32_MAX ||
            candidate.loadsPending != 0 || candidate.ready) {
            return;
        }
        launchAttentionKvPrefetch(0, buffer, false, nextQueryTile);
    }

    void beginAttentionQkTransposedKvSubtile() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.phase = AttentionWorkerPhase::QkProgramMatrix;
        const uint32_t kvSubtileRows = attentionKvSubtileRows(state);
        const uint32_t firstKey = state.phaseSliceIndex * 16;
        attentionLocalRead(
            state.kLocal + static_cast<uint64_t>(attentionKvLocalKey(state, firstKey)) *
                state.dispatch.headDim * sizeof(float),
            static_cast<uint64_t>(kvSubtileRows) * state.dispatch.headDim * sizeof(float),
            [this, kvSubtileRows](bool ok, const std::vector<uint8_t>& bytes) {
                if (!attentionWorker_ || !ok) { finishAttentionWorker(false); return; }
                const std::vector<double> keys = attentionBytesToDoubles(bytes);
                AttentionWorkerState& callbackState = *attentionWorker_;
                if (callbackState.phaseSliceIndex + 1 == attentionKvSubtileCount(callbackState)) {
                    recordAttentionKvOperandRelease(true);
                }
                callbackState.arrayPayload.assign(
                    static_cast<size_t>(arrayOutputSize) * arrayInputSize, 0.0);
                std::copy(keys.begin(), keys.end(), callbackState.arrayPayload.begin());
                std::vector<uint32_t> arrayIDs(attentionQueryRows(callbackState));
                std::iota(arrayIDs.begin(), arrayIDs.end(), 0);
                const uint64_t tag = attentionTransferTag();
                if (!programAttentionGemmMatrixGroupAsync(
                        arrayIDs, callbackState.arrayPayload, sizeof(float), tag,
                        [this, tag](bool programOk, uint64_t callbackTag) {
                            if (!attentionWorker_ || !programOk || callbackTag != tag) {
                                finishAttentionWorker(false); return;
                            }
                            statAttentionQkMatrixBroadcasts_->addData(1);
                            if (attentionWorker_->phaseSliceIndex == 0) {
                                attentionWorker_->arraysPending = 0;
                                attentionWorker_->attentionQkInputProgrammingComplete = false;
                                recordAttentionInterTilePhase(
                                    AttentionInterTilePhase::QkMatrixProgram);
                                transitionAttentionTilePipeline(
                                    AttentionTilePipelinePhase::QkInputProgram);
                                attentionWorker_->phase =
                                    AttentionWorkerPhase::QkProgramInputs;
                                attentionWorker_->index = 0;
                                programAttentionQkTransposedInput();
                            } else {
                                transitionAttentionTilePipeline(
                                    AttentionTilePipelinePhase::QkComputeReadout);
                                startAttentionQkCompute(attentionQueryRows(*attentionWorker_));
                            }
                        })) finishAttentionWorker(false);
            });
    }

    void programAttentionQkTransposedInput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t activeArrays = attentionQueryRows(state);
        if (state.index == activeArrays) {
            recordAttentionInterTilePhase(AttentionInterTilePhase::QkInputProgram);
            transitionAttentionTilePipeline(
                AttentionTilePipelinePhase::QkComputeReadout);
            state.attentionQkInputProgrammingComplete = true;
            if (!attentionQkEarlyCompute_) {
                startAttentionQkCompute(activeArrays);
            } else if (state.arraysPending == 0) {
                state.index = 0;
                state.phase = AttentionWorkerPhase::QkReadOutputs;
                readAttentionQkOutput();
            }
            return;
        }
        const uint32_t arrayId = state.index;
        std::vector<double> input(
            state.qPayload.begin() + static_cast<size_t>(arrayId) * state.dispatch.headDim,
            state.qPayload.begin() + static_cast<size_t>(arrayId + 1) * state.dispatch.headDim);
        const uint64_t tag = attentionTransferTag();
        if (!programAttentionGemmInputAsync(arrayId, input, sizeof(float), tag,
                [this, tag, arrayId](bool ok, uint64_t callbackTag) {
                    if (!attentionWorker_ || !ok || callbackTag != tag) {
                        finishAttentionWorker(false); return;
                    }
                    if (attentionQkEarlyCompute_ &&
                        !startAttentionQkArrayComputation(arrayId)) {
                        return;
                    }
                    attentionWorker_->index += 1;
                    programAttentionQkTransposedInput();
                })) finishAttentionWorker(false);
    }

    void startAttentionQkCompute(uint32_t activeArrays) {
        if (!attentionWorker_ || activeArrays == 0) {
            finishAttentionWorker(false); return;
        }
        AttentionWorkerState& state = *attentionWorker_;
        state.phase = AttentionWorkerPhase::QkCompute;
        if (!attentionClusterEnable_) {
            state.arraysPending = activeArrays;
            if (attentionArrayPending_.size() < static_cast<size_t>(numArrays)) {
                attentionArrayPending_.resize(numArrays, 0);
            }
            if (attentionSequential64Enable_) {
                std::vector<uint32_t> arrayIds(activeArrays);
                std::iota(arrayIds.begin(), arrayIds.end(), 0);
                const uint64_t generation = state.generation;
                if (!workerCommandProcessor ||
                    !workerCommandProcessor->launchGemmArrayGroupActiveBank(
                    arrayIds, state.activeOperandBank,
                    (state.qkReductionSlice == 0 ? 0 : 1), 0, LastTickCycle,
                        [this, generation](uint32_t arrayId, uint64_t) {
                            if (!attentionCallbackGenerationMatches(generation)) return;
                            handleAttentionArrayDone(arrayId);
                        })) {
                    finishAttentionWorker(false);
                    return;
                }
                for (uint32_t arrayId : arrayIds) {
                    attentionArrayPending_[arrayId] = 1;
                    arrayStates[arrayId] = 1;
                    statAttentionQkArrayOps_->addData(1);
                    statAttentionGenericGemmQkOps_->addData(1);
                }
                statAttentionSequentialQkWaves_->addData(1);
                statAttentionSequentialQkActiveArrays_->addData(activeArrays);
                return;
            }
            for (uint32_t arrayId = 0; arrayId < activeArrays; ++arrayId) {
                attentionArrayPending_[arrayId] = 1;
                arrayStates[arrayId] = 1;
                if (!launchAttentionGemmArray(arrayId, 0, true)) {
                    finishAttentionWorker(false);
                    return;
                }
                statAttentionQkArrayOps_->addData(1);
            }
            return;
        }
        state.arraysPending = 0;
        state.clusterOwnerLaunchNext = 0;
        state.clusterOwnerLaunchEnd = activeArrays;
        state.clusterOwnerLaunchQk = true;
        if (attentionArrayPending_.size() < static_cast<size_t>(numArrays))
            attentionArrayPending_.resize(numArrays, 0);
        pumpAttentionClusterOwnerLaunch();
    }

    void pumpAttentionClusterOwnerLaunch() {
        if (!attentionWorker_ ||
            attentionWorker_->clusterOwnerLaunchNext >=
                attentionWorker_->clusterOwnerLaunchEnd) return;
        AttentionWorkerState& state = *attentionWorker_;
        std::vector<uint32_t> arrayIds;
        for (uint32_t logical = state.clusterOwnerLaunchNext;
             logical < state.clusterOwnerLaunchEnd; ++logical) {
            arrayIds.push_back(state.clusterOwnerLaunchQk
                ? logical
                : attentionClusterConfig_.qkArrays + logical);
        }
        for (uint32_t arrayId : arrayIds) {
            attentionArrayPending_[arrayId] = 1;
            arrayStates[arrayId] = 1;
        }
        const uint64_t generation = state.generation;
        const bool qkOperation = state.clusterOwnerLaunchQk;
        const uint32_t bank = qkOperation
            ? state.activeOperandBank : state.clusterPvOperandBank;
        const uint32_t activeColumns = qkOperation
            ? 0 : attentionPvActiveColumns(state);
        const bool accepted = workerCommandProcessor &&
            workerCommandProcessor->launchGemmArrayGroupActiveBank(
                arrayIds, bank, 0, activeColumns, LastTickCycle,
                [this, generation, qkOperation](uint32_t arrayId, uint64_t) {
                    if (!attentionCallbackGenerationMatches(generation)) {
                        BusyActivityTracker& activity = qkOperation
                            ? attentionClusterQkArrayActivity_
                            : attentionClusterPvArrayActivity_;
                        if (!activity.leave(getCurrentSimCycle()))
                            statAttentionClusterIllegalTransitions_->addData(1);
                        return;
                    }
                    handleAttentionArrayDone(arrayId);
                });
        if (!accepted) {
            for (uint32_t arrayId : arrayIds) {
                attentionArrayPending_[arrayId] = 0;
                arrayStates[arrayId] = 0;
            }
            statAttentionClusterInitialEnqueueRetries_->addData(1);
            return;
        }
        if (qkOperation && state.clusterQkWave == 0) {
            recordAttentionClusterQkTileStart(
                LastTickCycle, state.kvTileIndex, state.queryTileIndex);
        }
        BusyActivityTracker& activity = qkOperation
            ? attentionClusterQkArrayActivity_
            : attentionClusterPvArrayActivity_;
        for (size_t index = 0; index < arrayIds.size(); ++index) {
            if (!activity.enter(getCurrentSimCycle())) {
                statAttentionClusterIllegalTransitions_->addData(1);
                finishAttentionWorker(false);
                return;
            }
            (qkOperation ? statAttentionQkArrayOps_ : statAttentionPvArrayOps_)
                ->addData(1);
            (qkOperation ? statAttentionGenericGemmQkOps_
                         : statAttentionGenericGemmPvOps_)->addData(1);
            if (activeColumns != 0) {
                statAttentionPvActiveKLaunches_->addData(1);
                statAttentionPvActiveKColumns_->addData(activeColumns);
            }
        }
        state.arraysPending += arrayIds.size();
        state.clusterOwnerLaunchNext = 0;
        state.clusterOwnerLaunchEnd = 0;
    }

    bool startAttentionQkArrayComputation(uint32_t arrayId) {
        if (!attentionWorker_ || arrayId >= attentionArrayPending_.size() ||
            attentionArrayPending_[arrayId] != 0) {
            finishAttentionWorker(false);
            return false;
        }
        AttentionWorkerState& state = *attentionWorker_;
        attentionArrayPending_[arrayId] = 1;
        arrayStates[arrayId] = 1;
        state.arraysPending += 1;
        state.phase = AttentionWorkerPhase::QkCompute;
        if (!launchAttentionGemmArray(arrayId, 0, true)) {
            finishAttentionWorker(false);
            return false;
        }
        statAttentionQkArrayOps_->addData(1);
        statAttentionQkEarlyComputeArrays_->addData(1);
        return true;
    }

    void programAttentionQkInput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t activeArrays = attentionKvSubtileRows(state);
        if (attentionSequential64Enable_ && state.index != activeArrays) {
            const uint32_t rows = attentionKeyCols(state);
            const size_t tileElements =
                static_cast<size_t>(rows) * state.dispatch.headDim;
            auto issueScatter = [this, rows]() {
                if (!attentionWorker_) return;
                AttentionWorkerState& current = *attentionWorker_;
                std::vector<uint32_t> arrayIds(rows);
                std::iota(arrayIds.begin(), arrayIds.end(), 0);
                std::vector<double> inputs(
                    static_cast<size_t>(rows) * arrayInputSize);
                const size_t halfOffset =
                    static_cast<size_t>(current.qkReductionSlice) * arrayInputSize;
                for (uint32_t key = 0; key < rows; ++key) {
                    std::copy_n(
                        current.sequentialKTilePayload.begin() +
                            static_cast<size_t>(key) * current.dispatch.headDim +
                            halfOffset,
                        arrayInputSize,
                        inputs.begin() + static_cast<size_t>(key) * arrayInputSize);
                }
                const uint64_t generation = current.generation;
                const uint64_t tag = attentionTransferTag();
                if (!programAttentionSequentialInputScatterAsync(
                        arrayIds, inputs,
                        AttentionClusterTrafficClass::SequentialQkInputScatter,
                        tag,
                        [this, generation, tag](bool ok, uint64_t callbackTag) {
                            if (!attentionCallbackGenerationMatches(generation)) return;
                            if (!ok || callbackTag != tag) {
                                finishAttentionWorker(false);
                                return;
                            }
                            AttentionWorkerState& completed = *attentionWorker_;
                            completed.index = attentionKvSubtileRows(completed);
                            programAttentionQkInput();
                        })) finishAttentionWorker(false);
            };
            if (state.sequentialKTilePayload.size() == tileElements) {
                issueScatter();
                return;
            }
            const uint64_t generation = state.generation;
            attentionLocalRead(
                state.kLocal, tileElements * sizeof(float),
                [this, generation, tileElements, issueScatter](
                    bool ok, const std::vector<uint8_t>& bytes) mutable {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok) { finishAttentionWorker(false); return; }
                    attentionWorker_->sequentialKTilePayload =
                        attentionBytesToDoubles(bytes);
                    if (attentionWorker_->sequentialKTilePayload.size() !=
                            tileElements) {
                        finishAttentionWorker(false);
                        return;
                    }
                    issueScatter();
                });
            return;
        }
        if (state.index == activeArrays) {
            if (state.phaseSliceIndex + 1 == attentionKvSubtileCount(state) &&
                (!attentionSequential64Enable_ || state.qkReductionSlice == 1)) {
                recordAttentionKvOperandRelease(true);
            }
            recordAttentionInterTilePhase(AttentionInterTilePhase::QkInputProgram);
            transitionAttentionTilePipeline(
                AttentionTilePipelinePhase::QkComputeReadout);
            state.attentionQkInputProgrammingComplete = true;
            if (!attentionQkEarlyCompute_) {
                startAttentionQkCompute(activeArrays);
                return;
            }
            if (state.arraysPending == 0) {
                state.index = 0;
                state.phase = AttentionWorkerPhase::QkReadOutputs;
                readAttentionQkOutput();
            }
            return;
        }
        const uint32_t arrayId = state.index;
        const uint32_t key = attentionKvLocalKey(state, arrayId);
        attentionLocalRead(state.kLocal + static_cast<uint64_t>(key) *
                state.dispatch.headDim * sizeof(float) +
                static_cast<uint64_t>(state.qkReductionSlice) * arrayInputSize * sizeof(float),
            arrayInputSize * sizeof(float),
            [this, arrayId](bool ok, const std::vector<uint8_t>& bytes) {
                if (!attentionWorker_ || !ok) { finishAttentionWorker(false); return; }
                const std::vector<double> input = attentionBytesToDoubles(bytes);
                const uint64_t tag = attentionTransferTag();
                if (!programAttentionGemmInputAsync(arrayId, input, sizeof(float), tag,
                        [this, tag, arrayId](bool programOk, uint64_t callbackTag) {
                            if (!attentionWorker_ || !programOk || callbackTag != tag) {
                                finishAttentionWorker(false); return;
                            }
                            if (attentionQkEarlyCompute_ &&
                                !startAttentionQkArrayComputation(arrayId)) {
                                return;
                            }
                            attentionWorker_->index += 1;
                            programAttentionQkInput();
                        })) finishAttentionWorker(false);
            });
    }

    bool attentionQkInputSlotMatches(
        const AttentionWorkerState& state, const AttentionQkInputSlot& slot,
        uint32_t arrayId, uint64_t tag) const {
        return slot.generation == state.generation &&
            slot.queryTileIndex == state.queryTileIndex &&
            slot.kvTileIndex == state.kvTileIndex &&
            slot.phaseSliceIndex == state.phaseSliceIndex && slot.arrayId == arrayId &&
            slot.transferTag == tag;
    }

    void updateAttentionQkInputOverlap() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const bool active =
            state.qkInputReadInFlight && state.qkInputProgramInFlight;
        if (active && !state.qkInputOverlapActive) {
            state.qkInputOverlapActive = true;
            state.qkInputOverlapStartTick = LastTickCycle;
        } else if (!active && state.qkInputOverlapActive) {
            statAttentionQkInputPipelineOverlapTicks_->addData(
                LastTickCycle - state.qkInputOverlapStartTick);
            state.qkInputOverlapActive = false;
            state.qkInputOverlapStartTick = 0;
        }
    }

    void resetAttentionQkInputPipeline() {
        if (!attentionWorker_) return;
        updateAttentionQkInputOverlap();
        AttentionWorkerState& state = *attentionWorker_;
        for (auto& slot : state.qkInputSlots) slot = {};
        state.qkInputNextFetch = 0;
        state.qkInputNextProgram = 0;
        state.qkInputReadInFlight = false;
        state.qkInputProgramInFlight = false;
        state.qkInputOverlapActive = false;
        state.qkInputOverlapStartTick = 0;
        state.qkInputMaxDepth = 0;
    }

    uint32_t attentionQkInputPipelineDepth(const AttentionWorkerState& state) const {
        return static_cast<uint32_t>(std::count_if(
            state.qkInputSlots.begin(), state.qkInputSlots.end(),
            [](const AttentionQkInputSlot& slot) {
                return slot.state != AttentionQkInputSlotState::Free;
            }));
    }

    void pumpAttentionQkInputPipeline() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.phase != AttentionWorkerPhase::QkProgramInputs &&
            state.phase != AttentionWorkerPhase::QkCompute) {
            finishAttentionWorker(false);
            return;
        }
        const uint32_t activeArrays = attentionKvSubtileRows(state);
        if (state.qkInputNextProgram == activeArrays &&
            !state.qkInputReadInFlight && !state.qkInputProgramInFlight) {
            if (attentionQkInputPipelineDepth(state) != 0) {
                finishAttentionWorker(false);
                return;
            }
            statAttentionQkInputPipelineMaxDepth_->addData(state.qkInputMaxDepth);
            resetAttentionQkInputPipeline();
            state.index = activeArrays;
            programAttentionQkInput();
            return;
        }

        if (!state.qkInputProgramInFlight &&
            state.qkInputNextProgram < activeArrays) {
            const uint32_t arrayId = state.qkInputNextProgram;
            auto slotIt = std::find_if(
                state.qkInputSlots.begin(), state.qkInputSlots.end(),
                [arrayId](const AttentionQkInputSlot& slot) {
                    return slot.state == AttentionQkInputSlotState::Ready &&
                        slot.arrayId == arrayId;
                });
            if (slotIt != state.qkInputSlots.end()) {
                const uint32_t slotIndex = static_cast<uint32_t>(
                    std::distance(state.qkInputSlots.begin(), slotIt));
                AttentionQkInputSlot& slot = *slotIt;
                const uint64_t generation = state.generation;
                const uint64_t tag = slot.transferTag;
                const std::vector<double> input =
                    attentionBytesToDoubles(slot.payload);
                slot.state = AttentionQkInputSlotState::Programming;
                state.qkInputProgramInFlight = true;
                updateAttentionQkInputOverlap();
                if (!programAttentionGemmInputAsync(
                        arrayId, input, sizeof(float), tag,
                        [this, slotIndex, arrayId, generation, tag](
                            bool ok, uint64_t callbackTag) {
                            if (!attentionCallbackGenerationMatches(generation)) return;
                            AttentionWorkerState& callbackState = *attentionWorker_;
                            AttentionQkInputSlot& callbackSlot =
                                callbackState.qkInputSlots[slotIndex];
                            if (!ok || callbackTag != tag ||
                                !attentionQkInputSlotMatches(
                                    callbackState, callbackSlot, arrayId, tag) ||
                                callbackSlot.state !=
                                    AttentionQkInputSlotState::Programming) {
                                statAttentionQkInputPipelineTagMismatches_->addData(1);
                                finishAttentionWorker(false);
                                return;
                            }
                            callbackSlot = {};
                            callbackState.qkInputProgramInFlight = false;
                            callbackState.qkInputNextProgram += 1;
                            statAttentionQkInputPipelineRowsProgrammed_->addData(1);
                            updateAttentionQkInputOverlap();
                            if (attentionQkEarlyCompute_ &&
                                !startAttentionQkArrayComputation(arrayId)) {
                                return;
                            }
                            pumpAttentionQkInputPipeline();
                        })) {
                    finishAttentionWorker(false);
                    return;
                }
            }
        }

        if (!attentionWorker_) return;
        AttentionWorkerState& current = *attentionWorker_;
        if (!current.qkInputReadInFlight &&
            current.qkInputNextFetch < activeArrays) {
            auto slotIt = std::find_if(
                current.qkInputSlots.begin(), current.qkInputSlots.end(),
                [](const AttentionQkInputSlot& slot) {
                    return slot.state == AttentionQkInputSlotState::Free;
                });
            if (slotIt == current.qkInputSlots.end()) {
                statAttentionQkInputPipelineSlotFullStalls_->addData(1);
                return;
            }
            const uint32_t slotIndex = static_cast<uint32_t>(
                std::distance(current.qkInputSlots.begin(), slotIt));
            const uint32_t arrayId = current.qkInputNextFetch++;
            const uint32_t key = attentionKvLocalKey(
                current, current.phaseSliceIndex * 16 + arrayId);
            const uint64_t generation = current.generation;
            const uint64_t tag = attentionTransferTag();
            AttentionQkInputSlot& slot = *slotIt;
            slot.state = AttentionQkInputSlotState::Reading;
            slot.generation = current.generation;
            slot.queryTileIndex = current.queryTileIndex;
            slot.kvTileIndex = current.kvTileIndex;
            slot.phaseSliceIndex = current.phaseSliceIndex;
            slot.arrayId = arrayId;
            slot.transferTag = tag;
            slot.payload.clear();
            current.qkInputReadInFlight = true;
            current.qkInputMaxDepth = std::max(
                current.qkInputMaxDepth,
                attentionQkInputPipelineDepth(current));
            updateAttentionQkInputOverlap();
            const bool accepted = globalMem->localReadAsync(
                current.kLocal + static_cast<uint64_t>(key) *
                    current.dispatch.headDim * sizeof(float),
                current.dispatch.headDim * sizeof(float),
                LocalMemoryClient::RoCC, tag,
                [this, slotIndex, arrayId, generation, tag](
                    bool ok, uint64_t callbackTag,
                    const std::vector<uint8_t>& bytes) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    AttentionWorkerState& callbackState = *attentionWorker_;
                    AttentionQkInputSlot& callbackSlot =
                        callbackState.qkInputSlots[slotIndex];
                    if (!ok || callbackTag != tag ||
                        bytes.size() != callbackState.dispatch.headDim * sizeof(float) ||
                        !attentionQkInputSlotMatches(
                            callbackState, callbackSlot, arrayId, tag) ||
                        callbackSlot.state != AttentionQkInputSlotState::Reading) {
                        statAttentionQkInputPipelineTagMismatches_->addData(1);
                        finishAttentionWorker(false);
                        return;
                    }
                    callbackSlot.payload = bytes;
                    callbackSlot.state = AttentionQkInputSlotState::Ready;
                    callbackState.qkInputReadInFlight = false;
                    statAttentionQkInputPipelineRowsFetched_->addData(1);
                    updateAttentionQkInputOverlap();
                    pumpAttentionQkInputPipeline();
                });
            if (!accepted) {
                finishAttentionWorker(false);
                return;
            }
        }
    }

    void writeAttentionQkRowBurst() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.lane == attentionQueryRows(state)) {
            if (!workerCommandProcessor->endAttentionTileStorage(state.generation)) {
                finishAttentionWorker(false);
                return;
            }
            state.qkRowBurstStorageActive = false;
            state.qkRowBurstKvSubtile = UINT32_MAX;
            finishAttentionQkReadoutKvSubtile();
            return;
        }
        const uint32_t query = state.lane;
        const uint32_t phaseSliceIndex = state.phaseSliceIndex;
        const uint32_t kvSubtileRows = attentionKvSubtileRows(state);
        const uint64_t generation = state.generation;
        const uint64_t tag = attentionTransferTag();
        if (!workerCommandProcessor->readAttentionTileRowAsync(
                query, generation, tag, LastTickCycle,
                [this, query, phaseSliceIndex, kvSubtileRows, generation, tag](
                    bool ok, uint64_t callbackTag,
                    const std::vector<uint8_t>& bytes) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok || callbackTag != tag ||
                        attentionWorker_->phaseSliceIndex != phaseSliceIndex ||
                        bytes.size() != static_cast<size_t>(kvSubtileRows) * sizeof(float)) {
                        finishAttentionWorker(false);
                        return;
                    }
                    const uint64_t addr = attentionWorker_->spLocal +
                        (static_cast<uint64_t>(query) *
                             attentionKeyCols(*attentionWorker_) +
                         static_cast<uint64_t>(phaseSliceIndex) * 16) * sizeof(float);
                    attentionLocalWrite(addr, bytes, [this, generation](bool writeOk) {
                        if (!attentionCallbackGenerationMatches(generation)) return;
                        if (!writeOk) {
                            finishAttentionWorker(false);
                            return;
                        }
                        attentionWorker_->lane += 1;
                        writeAttentionQkRowBurst();
                    });
                })) {
            finishAttentionWorker(false);
        }
    }

    void readAttentionQkOutputRowBurst() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t kvSubtileRows = attentionKvSubtileRows(state);
        if (state.qkRowBurstKvSubtile != state.phaseSliceIndex) {
            if (!workerCommandProcessor->beginAttentionTileStorage(
                    attentionQueryRows(state), kvSubtileRows, sizeof(float),
                    state.generation)) {
                finishAttentionWorker(false);
                return;
            }
            state.qkRowBurstKvSubtile = state.phaseSliceIndex;
            state.qkRowBurstStorageActive = true;
            state.index = attentionClusterWaveRowBegin(
                state.clusterPvWave, attentionClusterPvLanes(state));
            state.lane = 0;
        }
        if (state.index == kvSubtileRows) {
            writeAttentionQkRowBurst();
            return;
        }
        const uint32_t arrayId = state.index;
        const uint32_t phaseSliceIndex = state.phaseSliceIndex;
        const uint64_t generation = state.generation;
        const uint64_t readTag = attentionTransferTag();
        if (!readAttentionGemmOutputAsync(
                arrayId, sizeof(float), readTag,
                [this, arrayId, phaseSliceIndex, generation, readTag](
                    bool ok, uint64_t callbackTag,
                    const std::vector<double>& values) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok || callbackTag != readTag ||
                        attentionWorker_->phaseSliceIndex != phaseSliceIndex ||
                        values.size() != attentionQueryRows(*attentionWorker_)) {
                        finishAttentionWorker(false);
                        return;
                    }
                    const std::vector<uint8_t> bytes =
                        attentionDoublesToBytes(values);
                    const uint64_t writeTag = attentionTransferTag();
                    if (!workerCommandProcessor->writeAttentionTileColumnAsync(
                            arrayId, bytes, generation, writeTag, LastTickCycle,
                            [this, arrayId, phaseSliceIndex, generation, writeTag](
                                bool writeOk, uint64_t callbackWriteTag) {
                                if (!attentionCallbackGenerationMatches(generation)) return;
                                if (!writeOk || callbackWriteTag != writeTag ||
                                    attentionWorker_->phaseSliceIndex != phaseSliceIndex ||
                                    attentionWorker_->index != arrayId) {
                                    finishAttentionWorker(false);
                                    return;
                                }
                                attentionWorker_->index += 1;
                                readAttentionQkOutputRowBurst();
                            })) {
                        finishAttentionWorker(false);
                    }
                })) {
            finishAttentionWorker(false);
        }
    }

    void completeAttentionSequentialQkWave() {
        if (!attentionWorker_ || !attentionSequential64Enable_) {
            finishAttentionWorker(false);
            return;
        }
        AttentionWorkerState& state = *attentionWorker_;
        if (state.qkReductionSlice == 0) {
            state.qkReductionSlice = 1;
            state.index = 0;
            state.phaseSliceIndex = 0;
            if (!buildAttentionSequentialQkMatrix(state)) {
                finishAttentionWorker(false);
                return;
            }
            state.phase = AttentionWorkerPhase::QkProgramMatrix;
            beginAttentionQkKvSubtile();
            return;
        }
        state.phase = AttentionWorkerPhase::QkReadOutputs;
        readAttentionSequentialQkTile();
    }

    void readAttentionSequentialQkTile() {
        if (!attentionWorker_ || !attentionSequential64Enable_ ||
            attentionWorker_->qkReductionSlice != 1) {
            finishAttentionWorker(false);
            return;
        }
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t rows = attentionQueryRows(state);
        const uint32_t keys = attentionKvSubtileRows(state);
        if (rows == 0 || rows > arrayOutputSize || keys != 64) {
            finishAttentionWorker(false);
            return;
        }
        std::vector<uint32_t> arrayIds(keys);
        std::iota(arrayIds.begin(), arrayIds.end(), 0);
        const uint64_t generation = state.generation;
        const uint64_t tag = attentionTransferTag();
        if (!readAttentionSequentialQkGroupAsync(
                arrayIds, tag,
                [this, generation, tag, rows, keys](
                    bool ok, uint64_t callbackTag,
                    const std::vector<double>& values) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok || callbackTag != tag ||
                        values.size() !=
                            static_cast<size_t>(arrayOutputSize) * keys) {
                        finishAttentionWorker(false);
                        return;
                    }
                    std::vector<float> scores(
                        static_cast<size_t>(rows) * keys, 0.0f);
                    for (uint32_t key = 0; key < keys; ++key) {
                        for (uint32_t query = 0; query < rows; ++query) {
                            scores[static_cast<size_t>(query) * keys + key] =
                                static_cast<float>(
                                    values[static_cast<size_t>(key) *
                                        arrayOutputSize + query]);
                        }
                    }
                    const std::vector<uint8_t> bytes =
                        attentionFloatsToBytes(scores);
                    attentionLocalWrite(
                        attentionWorker_->spLocal, bytes,
                        [this, generation](bool writeOk) {
                            if (!attentionCallbackGenerationMatches(generation)) return;
                            if (!writeOk) {
                                finishAttentionWorker(false);
                                return;
                            }
                            recordAttentionInterTilePhase(
                                AttentionInterTilePhase::QkComputeReadout);
                            transitionAttentionTilePipeline(
                                AttentionTilePipelinePhase::Softmax);
                            statAttentionWorkerQkTileCompleteTick_->addData(
                                getCurrentSimCycle());
                            traceAttentionTileCompletion(
                                "qk_tile_complete", "final_qk_tile_complete",
                                *attentionWorker_);
                            finishAttentionInterTile();
                            beginAttentionSoftmax();
                        });
                })) {
            finishAttentionWorker(false);
        }
    }

    void readAttentionQkOutput() {
        if (!attentionWorker_) return;
        if (attentionClusterEnable_) {
            readAttentionClusterQkOutput();
            return;
        }
        if (attentionQkDataflowTranspose_) {
            readAttentionQkTransposedOutput();
            return;
        }
        if (attentionQkScoreRowBurst_) {
            readAttentionQkOutputRowBurst();
            return;
        }
        if (attentionQkReadoutOverlap_) {
            readAttentionQkOutputPipelined();
            return;
        }
        AttentionWorkerState& state = *attentionWorker_;
        if (state.index == attentionKvSubtileRows(state)) {
            recordAttentionInterTilePhase(
                AttentionInterTilePhase::QkComputeReadout);
            if (attentionSequential64Enable_ && state.qkReductionSlice == 0) {
                state.qkReductionSlice = 1;
                state.index = 0;
                state.phaseSliceIndex = 0;
                if (!buildAttentionSequentialQkMatrix(state)) {
                    finishAttentionWorker(false);
                    return;
                }
                state.phase = AttentionWorkerPhase::QkProgramMatrix;
                beginAttentionQkKvSubtile();
            } else if (++state.phaseSliceIndex < attentionKvSubtileCount(state)) {
                transitionAttentionTilePipeline(
                    AttentionTilePipelinePhase::QkInputProgram);
                state.phase = AttentionWorkerPhase::QkProgramInputs;
                state.index = 0;
                if (attentionQkInputPipeline_) {
                    resetAttentionQkInputPipeline();
                    pumpAttentionQkInputPipeline();
                } else {
                    programAttentionQkInput();
                }
            } else {
                transitionAttentionTilePipeline(AttentionTilePipelinePhase::Softmax);
                statAttentionWorkerQkTileCompleteTick_->addData(getCurrentSimCycle());
                traceAttentionTileCompletion(
                    "qk_tile_complete", "final_qk_tile_complete", state);
                finishAttentionInterTile();
                beginAttentionSoftmax();
            }
            return;
        }
        const uint32_t arrayId = state.index;
        const uint64_t tag = attentionTransferTag();
        if (!readAttentionGemmOutputAsync(arrayId, sizeof(float), tag,
                [this, tag, arrayId](bool ok, uint64_t callbackTag,
                                     const std::vector<double>& values) {
                    if (!attentionWorker_ || !ok || callbackTag != tag || values.size() != arrayOutputSize) {
                        finishAttentionWorker(false); return;
                    }
                    attentionWorker_->readOutputBytes = attentionDoublesToBytes(values);
                    attentionWorker_->lane = 0;
                    const auto issueLane = [this, arrayId](auto&& self) -> void {
                        if (!attentionWorker_) return;
                        AttentionWorkerState& laneState = *attentionWorker_;
                        if (laneState.lane == attentionQueryRows(laneState)) {
                            laneState.index += 1;
                            readAttentionQkOutput();
                            return;
                        }
                        const uint32_t query = laneState.lane;
                        const uint32_t key = arrayId;
                        std::vector<uint8_t> scalar(
                            laneState.readOutputBytes.begin() + query * sizeof(float),
                            laneState.readOutputBytes.begin() + (query + 1) * sizeof(float));
                        const uint64_t addr = laneState.spLocal +
                            (static_cast<uint64_t>(query) * attentionKeyCols(laneState) + key) *
                                sizeof(float);
                        attentionLocalWrite(addr, scalar, [this, self](bool writeOk) mutable {
                            if (!attentionWorker_ || !writeOk) {
                                finishAttentionWorker(false); return;
                            }
                            attentionWorker_->lane += 1;
                            self(self);
                        });
                    };
                    issueLane(issueLane);
                })) finishAttentionWorker(false);
    }

    void finishAttentionQkReadoutKvSubtile() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.qkReadKvSubtile = UINT32_MAX;
        state.qkReadBuffers.clear();
        state.qkReadReady.clear();
        recordAttentionInterTilePhase(AttentionInterTilePhase::QkComputeReadout);
        if (++state.phaseSliceIndex < attentionKvSubtileCount(state)) {
            transitionAttentionTilePipeline(AttentionTilePipelinePhase::QkInputProgram);
            state.phase = AttentionWorkerPhase::QkProgramInputs;
            state.index = 0;
            if (attentionQkInputPipeline_) {
                resetAttentionQkInputPipeline();
                pumpAttentionQkInputPipeline();
            } else {
                programAttentionQkInput();
            }
            return;
        }
        transitionAttentionTilePipeline(AttentionTilePipelinePhase::Softmax);
        statAttentionWorkerQkTileCompleteTick_->addData(getCurrentSimCycle());
        traceAttentionTileCompletion("qk_tile_complete", "final_qk_tile_complete", state);
        finishAttentionInterTile();
        beginAttentionSoftmax();
    }

    void commitAttentionQkReadoutLane() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t arrayId = state.qkReadCommitIndex;
        if (arrayId >= state.qkReadBuffers.size() ||
            state.qkReadCommitLane >= attentionQueryRows(state)) {
            if (arrayId < state.qkReadReady.size()) state.qkReadReady[arrayId] = 0;
            if (arrayId < state.qkReadBuffers.size()) state.qkReadBuffers[arrayId].clear();
            state.qkReadCommitIndex += 1;
            state.qkReadCommitLane = 0;
            state.qkReadCommitInFlight = false;
            issueAttentionQkReadAhead();
            commitAttentionQkReadout();
            return;
        }
        const uint32_t query = state.qkReadCommitLane;
        const uint32_t key = state.phaseSliceIndex * 16 + arrayId;
        const auto& bytes = state.qkReadBuffers[arrayId];
        if (bytes.size() != 16 * sizeof(float)) {
            finishAttentionWorker(false);
            return;
        }
        std::vector<uint8_t> scalar(
            bytes.begin() + query * sizeof(float),
            bytes.begin() + (query + 1) * sizeof(float));
        const uint64_t addr = state.spLocal +
            (static_cast<uint64_t>(query) * attentionKeyCols(state) + key) * sizeof(float);
        const uint64_t generation = state.generation;
        attentionLocalWrite(addr, scalar, [this, generation](bool writeOk) {
            if (!attentionCallbackGenerationMatches(generation)) return;
            if (!writeOk) {
                finishAttentionWorker(false);
                return;
            }
            attentionWorker_->qkReadCommitLane += 1;
            commitAttentionQkReadoutLane();
        });
    }

    void commitAttentionQkReadout() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t kvSubtileRows = attentionKvSubtileRows(state);
        if (state.qkReadCommitInFlight) return;
        if (state.qkReadCommitIndex == kvSubtileRows) {
            if (state.qkReadInFlight == 0 && state.qkReadIssueIndex == kvSubtileRows) {
                finishAttentionQkReadoutKvSubtile();
            }
            return;
        }
        if (state.qkReadCommitIndex >= state.qkReadReady.size() ||
            state.qkReadReady[state.qkReadCommitIndex] == 0) {
            return;
        }
        state.qkReadCommitInFlight = true;
        state.qkReadCommitLane = 0;
        commitAttentionQkReadoutLane();
    }

    void issueAttentionQkReadAhead() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t kvSubtileRows = attentionKvSubtileRows(state);
        while (state.qkReadIssueIndex < kvSubtileRows &&
               state.qkReadIssueIndex - state.qkReadCommitIndex <
                   attentionQkReadoutWindow_) {
            const uint32_t arrayId = state.qkReadIssueIndex++;
            statAttentionQkReadoutAheadDepth_->addData(
                state.qkReadIssueIndex - state.qkReadCommitIndex);
            const uint32_t phaseSliceIndex = state.phaseSliceIndex;
            const uint64_t generation = state.generation;
            const uint64_t tag = attentionTransferTag();
            state.qkReadInFlight += 1;
            if (!readAttentionGemmOutputAsync(arrayId, sizeof(float), tag,
                    [this, tag, arrayId, phaseSliceIndex, generation](
                        bool ok, uint64_t callbackTag,
                        const std::vector<double>& values) {
                        if (!attentionCallbackGenerationMatches(generation)) {
                            return;
                        }
                        if (!ok || callbackTag != tag ||
                            attentionWorker_->phaseSliceIndex != phaseSliceIndex || values.size() != 16 ||
                            arrayId >= attentionWorker_->qkReadBuffers.size()) {
                            finishAttentionWorker(false);
                            return;
                        }
                        AttentionWorkerState& callbackState = *attentionWorker_;
                        callbackState.qkReadBuffers[arrayId] =
                            attentionDoublesToBytes(values);
                        callbackState.qkReadReady[arrayId] = 1;
                        if (callbackState.qkReadInFlight > 0) {
                            callbackState.qkReadInFlight -= 1;
                        }
                        issueAttentionQkReadAhead();
                        commitAttentionQkReadout();
                    })) {
                finishAttentionWorker(false);
                return;
            }
        }
    }

    void readAttentionQkOutputPipelined() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.qkReadKvSubtile != state.phaseSliceIndex) {
            const uint32_t kvSubtileRows = attentionKvSubtileRows(state);
            state.qkReadKvSubtile = state.phaseSliceIndex;
            state.qkReadIssueIndex = 0;
            state.qkReadCommitIndex = 0;
            state.qkReadInFlight = 0;
            state.qkReadCommitLane = 0;
            state.qkReadCommitInFlight = false;
            state.qkReadBuffers.assign(kvSubtileRows, {});
            state.qkReadReady.assign(kvSubtileRows, 0);
        }
        issueAttentionQkReadAhead();
        commitAttentionQkReadout();
    }

    void readAttentionQkTransposedOutput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.index == attentionQueryRows(state)) {
            if (++state.phaseSliceIndex < attentionKvSubtileCount(state)) {
                transitionAttentionTilePipeline(
                    AttentionTilePipelinePhase::QkMatrixProgram);
                state.index = 0;
                beginAttentionQkTransposedKvSubtile();
            } else {
                recordAttentionInterTilePhase(
                    AttentionInterTilePhase::QkComputeReadout);
                transitionAttentionTilePipeline(AttentionTilePipelinePhase::Softmax);
                statAttentionWorkerQkTileCompleteTick_->addData(getCurrentSimCycle());
                traceAttentionTileCompletion(
                    "qk_tile_complete", "final_qk_tile_complete", state);
                finishAttentionInterTile();
                beginAttentionSoftmax();
            }
            return;
        }
        const uint32_t arrayId = state.index;
        const uint64_t tag = attentionTransferTag();
        if (!readAttentionGemmOutputAsync(arrayId, sizeof(float), tag,
                [this, tag, arrayId](bool ok, uint64_t callbackTag,
                                     const std::vector<double>& values) {
                    if (!attentionWorker_ || !ok || callbackTag != tag ||
                        values.size() != 16) {
                        finishAttentionWorker(false); return;
                    }
                    std::vector<uint8_t> bytes = attentionDoublesToBytes(values);
                    bytes.resize(attentionKvSubtileRows(*attentionWorker_) * sizeof(float));
                    const uint64_t addr = attentionWorker_->spLocal +
                        (static_cast<uint64_t>(arrayId) *
                             attentionKeyCols(*attentionWorker_) +
                         attentionWorker_->phaseSliceIndex * 16) * sizeof(float);
                    attentionLocalWrite(addr, bytes, [this](bool writeOk) {
                        if (!attentionWorker_ || !writeOk) {
                            finishAttentionWorker(false); return;
                        }
                        attentionWorker_->index += 1;
                        readAttentionQkTransposedOutput();
                    });
                })) finishAttentionWorker(false);
    }

    void continueAttentionAfterVReady() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (attentionSequential64Enable_ && attentionKvDoubleBuffer_ &&
            state.activeKvBuffer < state.kvBuffers.size()) {
            AttentionKvBufferState& descriptor =
                state.kvBuffers[state.activeKvBuffer];
            if (descriptor.consuming &&
                descriptor.kvTileIndex == state.kvTileIndex &&
                descriptor.physicalKvTileIndex == state.physicalKvTileIndex) {
                if (!descriptor.pairUseRecorded) {
                    descriptor.pairUseRecorded = true;
                    if (descriptor.ready) {
                        statAttentionKvPrefetchHits_->addData(1);
                        if (descriptor.crossQuery) {
                            statAttentionKvCrossQueryHits_->addData(1);
                        }
                        statAttentionKvPrefetchReadyLeadTicks_->addData(
                            getCurrentSimCycle() - descriptor.readyTick);
                    } else {
                        statAttentionKvPrefetchWaits_->addData(1);
                        if (descriptor.crossQuery) {
                            statAttentionKvCrossQueryWaits_->addData(1);
                        }
                    }
                }
                if (!descriptor.ready) {
                    state.attentionWaitingForV = true;
                    state.attentionVWaitTick = getCurrentSimCycle();
                    return;
                }
            }
        }
        if (attentionClusterEnable_ && !reserveAttentionClusterPv()) {
            if (state.clusterPvMatrixAheadPromotionWaiting) return;
            finishAttentionWorker(false);
            return;
        }
        if (attentionPvMatrixSoftmaxOverlap_) {
            continueAttentionAfterSoftmaxAndPvMatrix();
            return;
        }
        transitionAttentionTilePipeline(
            AttentionTilePipelinePhase::PvMatrixProgram);
        beginAttentionPvOutputSlice();
    }

    void beginAttentionSoftmax() {
        if (!attentionWorker_ || sfu == nullptr) { finishAttentionWorker(false); return; }
        AttentionWorkerState& state = *attentionWorker_;
        state.phase = AttentionWorkerPhase::Softmax;
        state.attentionSoftmaxComplete = false;
        state.attentionPvMatrixComplete = false;
        maybeStartAttentionAheadOperands();
        AttentionTileRequest request;
        request.tag = state.dispatch.tag +
            state.queryTileIndex * ((state.dispatch.expectedCols +
                state.dispatch.kvTileRows - 1) / state.dispatch.kvTileRows) +
            state.kvTileIndex + 1;
        request.jobId = state.dispatch.jobId;
        request.localScoreAddr = state.spLocal;
        request.globalRowBegin = state.dispatch.row +
            state.queryTileIndex * state.dispatch.queryTileRows;
        request.keyBegin = state.physicalKvTileIndex * state.dispatch.kvTileRows;
        request.rows = attentionQueryRows(state);
        request.cols = attentionKeyCols(state);
        request.headDim = state.dispatch.headDim;
        request.kvTileIndex = state.kvTileIndex;
        request.numKvTiles = attentionKvTileCountForQueryTile(state);
        request.causal = attentionCausal(state);
        request.firstTileForJob = state.queryTileIndex == 0 && state.kvTileIndex == 0;
        if (attentionClusterEnable_) {
            if (state.clusterScoreContext < 0) {
                finishAttentionWorker(false);
                return;
            }
            if (state.clusterPContext < 0) {
                const size_t pElements =
                    static_cast<size_t>(request.rows) * request.cols;
                const AttentionClusterAdmission admission =
                    sfu->attentionPSlotAdmission(
                        static_cast<uint32_t>(state.clusterScoreContext),
                        state.clusterTileTag, pElements);
                if (admission == AttentionClusterAdmission::Invalid) {
                    finishAttentionWorker(false);
                    return;
                }
                if (admission == AttentionClusterAdmission::Retry) {
                    const uint64_t retryGeneration = state.generation;
                    const AttentionClusterTag retryTag = state.clusterTileTag;
                    scheduleAttentionClusterOwnerRetry(
                        [this, retryGeneration, retryTag]() {
                            if (!attentionCallbackGenerationMatches(retryGeneration) ||
                                !attentionWorker_ ||
                                !(attentionWorker_->clusterTileTag == retryTag)) return;
                            beginAttentionSoftmax();
                        });
                    return;
                }
                if (!sfu->reserveAttentionPSlot(
                        static_cast<uint32_t>(state.clusterScoreContext),
                        state.clusterTileTag, pElements)) {
                    finishAttentionWorker(false);
                    return;
                }
                state.clusterPContext = state.clusterScoreContext;
            } else if (state.clusterPContext != state.clusterScoreContext) {
                finishAttentionWorker(false);
                return;
            }
            request.directScoreMode = true;
            request.generation = state.generation;
            request.scoreSlot = static_cast<uint32_t>(state.clusterScoreContext);
            request.scoreTag = state.clusterTileTag;
            request.directPMode = true;
            request.pSlot = static_cast<uint32_t>(state.clusterPContext);
            request.pTag = state.clusterTileTag;
        }
        const uint64_t generation = state.generation;
        if (attentionClusterEnable_) {
            const AttentionClusterAdmission admission =
                sfu->attentionTileAdmission(request);
            if (admission == AttentionClusterAdmission::Invalid) {
                finishAttentionWorker(false);
                return;
            }
            if (admission == AttentionClusterAdmission::Retry) {
                const uint64_t retryGeneration = state.generation;
                const AttentionClusterTag retryTag = state.clusterTileTag;
                scheduleAttentionClusterOwnerRetry(
                    [this, retryGeneration, retryTag]() {
                        if (!attentionCallbackGenerationMatches(retryGeneration) ||
                            !attentionWorker_ ||
                            !(attentionWorker_->clusterTileTag == retryTag)) return;
                        beginAttentionSoftmax();
                    });
                return;
            }
        }
        if (!sfu->issueAttentionTile(request, [this, generation](
                bool ok, const AttentionTileResult& result) {
                if (!attentionCallbackGenerationMatches(generation)) return;
                if (!ok) { finishAttentionWorker(false); return; }
                statAttentionWorkerSoftmaxTileCompleteTick_->addData(getCurrentSimCycle());
                traceAttentionTileCompletion(
                    "softmax_tile_complete", "final_softmax_tile_complete",
                    *attentionWorker_);
                const auto continueToPv = [this, result]() {
                    if (!attentionWorker_) return;
                    attentionWorker_->outputScales.assign(
                        result.oldOutputScale.begin(),
                        result.oldOutputScale.begin() + result.rows);
                    attentionWorker_->phaseSliceIndex = 0;
                    attentionWorker_->clusterPvWave = 0;
                    attentionWorker_->attentionSoftmaxComplete = true;
                    continueAttentionAfterVReady();
                };
                if (attentionClusterEnable_) {
                    if (!releaseAttentionClusterScore()) {
                        finishAttentionWorker(false);
                        return;
                    }
                    continueToPv();
                    return;
                }
                continueToPv();
            })) {
            output->output(
                "Attention SFU issue rejected core=%" PRIu64
                " kv_tile=%" PRIu32 " rows=%" PRIu32 " cols=%" PRIu32 "\n",
                coreID, request.kvTileIndex, request.rows, request.cols);
            if (attentionClusterEnable_) {
                finishAttentionWorker(false);
            } else {
                finishAttentionWorker(false);
            }
            return;
        }
        if (attentionPvMatrixSoftmaxOverlap_ && !attentionClusterEnable_) {
            statAttentionPvMatrixOverlapTiles_->addData(1);
            state.phaseSliceIndex = 0;
            beginAttentionPvOutputSlice();
        }
        if (attentionClusterEnable_) pumpAttentionClusterAhead();
    }

    void beginAttentionPvOutputSlice() {
        if (!attentionWorker_) return;
        if (attentionClusterEnable_) {
            beginAttentionClusterPvOutputSlice();
            return;
        }
        if (attentionSequential64Enable_) {
            beginAttentionSequentialPvWave();
            return;
        }
        AttentionWorkerState& state = *attentionWorker_;
        state.attentionPvMatrixComplete = false;
        if (!attentionPvMatrixSoftmaxOverlap_ || state.phaseSliceIndex != 0 ||
            state.attentionSoftmaxComplete) {
            state.phase = AttentionWorkerPhase::PvProgramMatrix;
        }
        const auto programMatrix = [this](const std::vector<double>& v) {
                if (!attentionWorker_) return;
                AttentionWorkerState& callbackState = *attentionWorker_;
                const uint32_t keyCols = attentionKeyCols(callbackState);
                const uint32_t pvActiveColumns =
                    attentionPvActiveColumns(callbackState);
                const uint32_t matrixColumns = pvActiveColumns != 0 ?
                    pvActiveColumns : arrayInputSize;
                callbackState.arrayPayload.assign(
                    static_cast<size_t>(arrayOutputSize) * matrixColumns, 0.0);
                for (uint32_t dim = 0; dim < arrayOutputSize; ++dim) {
                    for (uint32_t key = 0; key < keyCols; ++key) {
                        callbackState.arrayPayload[dim * matrixColumns + key] =
                            v[key * callbackState.dispatch.headDim +
                              callbackState.phaseSliceIndex * arrayOutputSize + dim];
                    }
                }
                if (attentionPvMatrixBroadcast_) {
                    std::vector<uint32_t> arrayIDs(attentionQueryRows(callbackState));
                    std::iota(arrayIDs.begin(), arrayIDs.end(), 0);
                    const uint64_t tag = attentionTransferTag();
                    const bool accepted = pvActiveColumns != 0 ?
                        programAttentionGemmMatrixGroupActiveAsync(
                            arrayIDs, callbackState.arrayPayload, pvActiveColumns,
                            sizeof(float), tag,
                            [this, tag](bool programOk, uint64_t callbackTag) {
                                if (!attentionWorker_ || !programOk || callbackTag != tag) {
                                    finishAttentionWorker(false); return;
                                }
                                completeAttentionPvMatrixProgram();
                            }) :
                        programAttentionGemmMatrixGroupAsync(
                            arrayIDs, callbackState.arrayPayload, sizeof(float), tag,
                            [this, tag](bool programOk, uint64_t callbackTag) {
                                if (!attentionWorker_ || !programOk || callbackTag != tag) {
                                    finishAttentionWorker(false); return;
                                }
                                completeAttentionPvMatrixProgram();
                            });
                    if (!accepted) {
                        finishAttentionWorker(false);
                    } else {
                        statAttentionPvMatrixBroadcasts_->addData(1);
                        if (pvActiveColumns != 0) {
                            statAttentionPvActiveKMatrixElements_->addData(
                                callbackState.arrayPayload.size() * arrayIDs.size());
                        }
                    }
                    return;
                }
                callbackState.attentionPvMatrixProgramsPending =
                    attentionQueryRows(callbackState);
                for (uint32_t arrayId = 0;
                     arrayId < attentionQueryRows(callbackState); ++arrayId) {
                    const uint64_t tag = attentionTransferTag();
                    const uint32_t activeColumns =
                        attentionPvActiveColumns(callbackState);
                    const bool accepted = activeColumns != 0 ?
                        programAttentionGemmMatrixActiveAsync(
                            arrayId, callbackState.arrayPayload, activeColumns,
                            sizeof(float), tag, [this, tag](bool programOk,
                                                            uint64_t callbackTag) {
                                if (!attentionWorker_ || !programOk || callbackTag != tag) {
                                    finishAttentionWorker(false); return;
                                }
                                if (attentionWorker_->attentionPvMatrixProgramsPending == 0) {
                                    finishAttentionWorker(false); return;
                                }
                                attentionWorker_->attentionPvMatrixProgramsPending -= 1;
                                if (attentionWorker_->attentionPvMatrixProgramsPending == 0) {
                                    completeAttentionPvMatrixProgram();
                                }
                            }) :
                        programAttentionGemmMatrixAsync(
                            arrayId, callbackState.arrayPayload, sizeof(float), tag,
                            [this, tag](bool programOk, uint64_t callbackTag) {
                                if (!attentionWorker_ || !programOk || callbackTag != tag) {
                                    finishAttentionWorker(false); return;
                                }
                                if (attentionWorker_->attentionPvMatrixProgramsPending == 0) {
                                    finishAttentionWorker(false); return;
                                }
                                attentionWorker_->attentionPvMatrixProgramsPending -= 1;
                                if (attentionWorker_->attentionPvMatrixProgramsPending == 0) {
                                    completeAttentionPvMatrixProgram();
                                }
                            });
                    if (!accepted) {
                        finishAttentionWorker(false);
                        return;
                    }
                    if (activeColumns != 0) {
                        statAttentionPvActiveKMatrixElements_->addData(
                            callbackState.arrayPayload.size());
                    }
                }
        };
        const uint32_t keyCols = attentionKeyCols(state);
        const uint64_t tileBytes = static_cast<uint64_t>(keyCols) *
            state.dispatch.headDim * sizeof(float);
        const uint64_t vTileBytes = static_cast<uint64_t>(keyCols) *
            arrayOutputSize * sizeof(float);
        const bool bufferFits = tileBytes <= attentionPvVTileBufferBytes_;
        const bool groupFollowerHit = attentionPvVTileGroupRetention_ &&
            state.phaseSliceIndex == 0 &&
            state.queryTileIndex != attentionKvGroupOwnerQueryTile(state);
        if (attentionPvVTileReuse_ && (state.phaseSliceIndex != 0 || groupFollowerHit) &&
            bufferFits && attentionVTileMatchesCurrentGroup(state) &&
            state.vPayload.size() == static_cast<size_t>(keyCols) * state.dispatch.headDim) {
            if (state.vTileBufferWaiting) return;
            const uint64_t accessTicks = attentionPvVTileBufferHitTicks_ +
                (vTileBytes + attentionPvVTileBufferBytesPerCycle_ - 1) /
                    attentionPvVTileBufferBytesPerCycle_;
            if (!state.vTileBufferBypassWait) {
                state.vTileBufferWaiting = true;
                state.vTileBufferCurrentWaitTicks = accessTicks;
                state.vTileBufferWaitUntilTick = LastTickCycle +
                    accessTicks;
                return;
            }
            state.vTileBufferBypassWait = false;
            statAttentionPvVTileBufferHits_->addData(1);
            statAttentionPvVTileBufferBytesReused_->addData(vTileBytes);
            if (groupFollowerHit) {
                statAttentionPvVTileBufferGroupHits_->addData(1);
                recordAttentionKvOperandRelease(false);
            }
            programMatrix(state.vPayload);
            return;
        }
        if (attentionPvVTileReuse_) {
            statAttentionPvVTileBufferMisses_->addData(1);
            if (!bufferFits) {
                statAttentionPvVTileBufferCapacityRejections_->addData(1);
            }
        }
        attentionLocalRead(
            state.vLocal + (attentionStreamKv(state) ? 0 :
                static_cast<uint64_t>(state.physicalKvTileIndex) *
                    state.dispatch.kvTileRows * state.dispatch.headDim * sizeof(float)),
            static_cast<uint64_t>(keyCols) * state.dispatch.headDim * sizeof(float),
            [this, programMatrix](bool ok, const std::vector<uint8_t>& bytes) {
                if (!attentionWorker_ || !ok) { finishAttentionWorker(false); return; }
                if (attentionPvVTileReuse_) {
                    statAttentionPvVTileBufferBytesRead_->addData(bytes.size());
                }
                attentionWorker_->vPayload = attentionBytesToDoubles(bytes);
                // The Local-GM V buffer is no longer needed after the tile has
                // been copied into the bounded staging payload. Release it
                // early so the alternate buffer can prefetch the next tile
                // while PV matrix programming/compute is in flight.
                recordAttentionKvOperandRelease(false);
                attentionWorker_->vTileValid = attentionPvVTileReuse_ &&
                    static_cast<uint64_t>(bytes.size()) <= attentionPvVTileBufferBytes_;
                attentionWorker_->vTileTag = attentionWorker_->vTileValid ?
                    attentionWorker_->kvTileIndex : UINT32_MAX;
                attentionWorker_->vTileGeneration = attentionWorker_->vTileValid ?
                    attentionWorker_->generation : 0;
                attentionWorker_->vTileGroupOwner = attentionWorker_->vTileValid ?
                    attentionKvGroupOwnerQueryTile(*attentionWorker_) : UINT32_MAX;
                if (attentionPvVTileReuse_ && attentionWorker_->phaseSliceIndex == 0 &&
                    attentionWorker_->vTileValid) {
                    // Keep the staging path's explicit release marker; the
                    // release operation is idempotent after the early V read.
                    recordAttentionKvOperandRelease(false);
                    const std::vector<uint8_t> bufferBytes = bytes;
                    attentionLocalWrite(attentionWorker_->vTileBufferLocal, bufferBytes,
                        [this, programMatrix](bool writeOk) {
                            if (!attentionWorker_ || !writeOk) {
                                finishAttentionWorker(false);
                                return;
                            }
                            programMatrix(attentionWorker_->vPayload);
                        });
                    return;
                }
                programMatrix(attentionWorker_->vPayload);
            });
    }

    void beginAttentionSequentialPvWave() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.phase = AttentionWorkerPhase::PvProgramMatrix;
        state.pvOutputSliceToProgram = 0;
        const uint32_t keyCols = attentionKeyCols(state);
        const size_t tileValues = static_cast<size_t>(keyCols) *
            state.dispatch.headDim;
        if (state.phaseSliceIndex != 0 && state.vPayload.size() == tileValues) {
            programAttentionSequentialPvMatrices();
            return;
        }
        const uint64_t generation = state.generation;
        attentionLocalRead(
            state.vLocal + (attentionStreamKv(state) ? 0 :
                static_cast<uint64_t>(state.physicalKvTileIndex) *
                    state.dispatch.kvTileRows * state.dispatch.headDim * sizeof(float)),
            static_cast<uint64_t>(keyCols) * state.dispatch.headDim * sizeof(float),
            [this, generation](bool ok, const std::vector<uint8_t>& bytes) {
                if (!attentionCallbackGenerationMatches(generation)) return;
                if (!ok) { finishAttentionWorker(false); return; }
                attentionWorker_->vPayload = attentionBytesToDoubles(bytes);
                if (attentionWorker_->vPayload.size() !=
                    static_cast<size_t>(attentionKeyCols(*attentionWorker_)) *
                        attentionWorker_->dispatch.headDim) {
                    finishAttentionWorker(false);
                    return;
                }
                recordAttentionKvOperandRelease(false);
                programAttentionSequentialPvMatrices();
            });
    }

    void programAttentionSequentialPvMatrices() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t numWaveOutputSlices = attentionSequentialPvOutputSliceCount(state);
        if (state.pvOutputSliceToProgram == numWaveOutputSlices) {
            transitionAttentionTilePipeline(
                AttentionTilePipelinePhase::PvInputProgram);
            state.phase = AttentionWorkerPhase::PvProgramInputs;
            programAttentionSequentialPvInputs();
            return;
        }
        const uint32_t localOutputSlice = state.pvOutputSliceToProgram;
        const uint32_t outputSlice = state.phaseSliceIndex + localOutputSlice;
        const uint32_t keyCols = attentionKeyCols(state);
        std::vector<double> matrix(
            static_cast<size_t>(arrayOutputSize) * keyCols, 0.0);
        for (uint32_t dim = 0; dim < arrayOutputSize; ++dim) {
            for (uint32_t key = 0; key < keyCols; ++key) {
                matrix[static_cast<size_t>(dim) * keyCols + key] =
                    state.vPayload[static_cast<size_t>(key) *
                        state.dispatch.headDim + outputSlice * arrayOutputSize + dim];
            }
        }
        std::vector<uint32_t> arrays(attentionQueryRows(state));
        for (uint32_t row = 0; row < arrays.size(); ++row) arrays[row] = row;
        const uint64_t generation = state.generation;
        const uint64_t tag = attentionTransferTag();
        const uint32_t activeColumns = attentionPvActiveColumns(state);
        const bool accepted = activeColumns != 0 ?
            programAttentionGemmMatrixGroupActiveAsync(
                arrays, matrix, activeColumns, sizeof(float), tag,
                [this, generation, tag](bool ok, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok || callbackTag != tag) {
                        finishAttentionWorker(false); return;
                    }
                    ++attentionWorker_->pvOutputSliceToProgram;
                    programAttentionSequentialPvMatrices();
                }) :
            programAttentionGemmMatrixGroupAsync(
                arrays, matrix, sizeof(float), tag,
                [this, generation, tag](bool ok, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok || callbackTag != tag) {
                        finishAttentionWorker(false); return;
                    }
                    ++attentionWorker_->pvOutputSliceToProgram;
                    programAttentionSequentialPvMatrices();
                });
        if (!accepted) finishAttentionWorker(false);
        else {
            statAttentionPvMatrixBroadcasts_->addData(1);
            if (activeColumns != 0) {
                statAttentionPvActiveKMatrixElements_->addData(
                    matrix.size() * arrays.size());
            }
        }
    }

    void programAttentionSequentialPvInputs() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t rows = attentionQueryRows(state);
        const uint32_t keyCols = attentionKeyCols(state);
        const size_t tileElements = static_cast<size_t>(rows) * keyCols;
        auto completeInputProgramming = [this]() {
            if (!attentionWorker_) return;
            attentionWorker_->index = 0;
            if (attentionWorker_->kvTileIndex != 0) {
                transitionAttentionTilePipeline(
                    AttentionTilePipelinePhase::PvRestoreOutput);
            }
            prepareAttentionSequentialPvOutputs();
        };
        if (state.phaseSliceIndex != 0 &&
            state.sequentialPPayload.size() == tileElements) {
            completeInputProgramming();
            return;
        }
        auto issueScatter = [this, rows, completeInputProgramming]() {
            if (!attentionWorker_) return;
            AttentionWorkerState& current = *attentionWorker_;
            std::vector<uint32_t> arrayIds(rows);
            std::iota(arrayIds.begin(), arrayIds.end(), 0);
            const uint64_t generation = current.generation;
            const uint64_t tag = attentionTransferTag();
            if (!programAttentionSequentialInputScatterAsync(
                    arrayIds, current.sequentialPPayload,
                    AttentionClusterTrafficClass::SequentialPvInputScatter,
                    tag,
                    [this, generation, tag, completeInputProgramming](
                        bool ok, uint64_t callbackTag) {
                        if (!attentionCallbackGenerationMatches(generation)) return;
                        if (!ok || callbackTag != tag) {
                            finishAttentionWorker(false);
                            return;
                        }
                        completeInputProgramming();
                    })) finishAttentionWorker(false);
        };
        if (state.sequentialPPayload.size() == tileElements) {
            issueScatter();
            return;
        }
        const uint64_t generation = state.generation;
        attentionLocalRead(
            state.spLocal, tileElements * sizeof(float),
            [this, generation, tileElements, issueScatter](
                bool ok, const std::vector<uint8_t>& bytes) mutable {
                if (!attentionCallbackGenerationMatches(generation)) return;
                if (!ok) { finishAttentionWorker(false); return; }
                attentionWorker_->sequentialPPayload =
                    attentionBytesToDoubles(bytes);
                if (attentionWorker_->sequentialPPayload.size() != tileElements) {
                    finishAttentionWorker(false);
                    return;
                }
                issueScatter();
            });
    }

    void prepareAttentionSequentialPvOutputs() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t activeArrays = attentionQueryRows(state);
        if (state.kvTileIndex == 0) {
            launchAttentionSequentialPvComputation();
            return;
        }
        const uint32_t rows = attentionQueryRows(state);
        const size_t oElements =
            static_cast<size_t>(rows) * state.dispatch.headDim;
        const uint64_t generation = state.generation;
        if (state.sequentialOPayload.size() != oElements) {
            const size_t halfBytes = oElements * sizeof(float) / 2;
            attentionLocalRead(state.oLocal, halfBytes,
                [this, generation, halfBytes, oElements](
                    bool ok, const std::vector<uint8_t>& lowBytes) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok) { finishAttentionWorker(false); return; }
                    attentionLocalRead(attentionWorker_->oLocal + halfBytes,
                        halfBytes,
                        [this, generation, lowBytes, oElements](
                            bool highOk,
                            const std::vector<uint8_t>& highBytes) {
                            if (!attentionCallbackGenerationMatches(generation)) return;
                            if (!highOk) { finishAttentionWorker(false); return; }
                            std::vector<uint8_t> bytes = lowBytes;
                            bytes.insert(bytes.end(), highBytes.begin(), highBytes.end());
                            attentionWorker_->sequentialOPayload =
                                attentionBytesToDoubles(bytes);
                            if (attentionWorker_->sequentialOPayload.size() !=
                                oElements) {
                                finishAttentionWorker(false);
                                return;
                            }
                            prepareAttentionSequentialPvOutputs();
                        });
                });
            return;
        }

        std::vector<uint32_t> arrayIds(activeArrays);
        std::iota(arrayIds.begin(), arrayIds.end(), 0);
        std::vector<double> outputs(
            static_cast<size_t>(activeArrays) * arrayOutputSize, 0.0);
        for (uint32_t row = 0; row < rows; ++row) {
            if (row >= state.outputScales.size()) {
                finishAttentionWorker(false);
                return;
            }
            const double scale = state.outputScales[row];
            for (uint32_t dim = 0; dim < arrayOutputSize; ++dim) {
                outputs[static_cast<size_t>(row) * arrayOutputSize + dim] =
                    state.sequentialOPayload[
                        static_cast<size_t>(row) * state.dispatch.headDim +
                        state.phaseSliceIndex * arrayOutputSize + dim] * scale;
            }
        }
        const uint64_t tag = attentionTransferTag();
        if (!writeAttentionSequentialPvOutputGroupAsync(
                arrayIds, outputs, tag,
                [this, generation, tag](bool ok, uint64_t callbackTag) {
                if (!attentionCallbackGenerationMatches(generation)) return;
                if (!ok || callbackTag != tag) {
                    finishAttentionWorker(false);
                    return;
                }
                launchAttentionSequentialPvComputation();
            })) finishAttentionWorker(false);
    }

    void launchAttentionSequentialPvComputation() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        transitionAttentionTilePipeline(AttentionTilePipelinePhase::PvCompute);
        state.phase = AttentionWorkerPhase::PvCompute;
        const uint32_t activeArrays = attentionQueryRows(state);
        state.arraysPending = activeArrays;
        state.index = 0;
        std::vector<uint32_t> arrayIds(activeArrays);
        std::iota(arrayIds.begin(), arrayIds.end(), 0);
        const uint64_t generation = state.generation;
        const uint32_t activeColumns = attentionPvActiveColumns(state);
        const bool accepted = workerCommandProcessor &&
            workerCommandProcessor->launchGemmArrayGroupActiveBank(
                arrayIds, state.activeOperandBank,
                state.kvTileIndex == 0 ? 0 : 1, activeColumns,
                LastTickCycle,
                [this, generation](uint32_t arrayId, uint64_t) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    handleAttentionArrayDone(arrayId);
                });
        if (!accepted) {
            finishAttentionWorker(false);
            return;
        }
        for (uint32_t arrayId : arrayIds) {
            attentionArrayPending_[arrayId] = 1;
            arrayStates[arrayId] = 1;
            statAttentionPvArrayOps_->addData(1);
            statAttentionGenericGemmPvOps_->addData(1);
            if (activeColumns != 0) {
                statAttentionPvActiveKLaunches_->addData(1);
                statAttentionPvActiveKColumns_->addData(activeColumns);
            }
        }
        statAttentionSequentialPvWaves_->addData(1);
        statAttentionSequentialPvActiveArrays_->addData(activeArrays);
    }

    void readAttentionSequentialPvOutputs() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t rows = attentionQueryRows(state);
        std::vector<uint32_t> arrayIds(rows);
        std::iota(arrayIds.begin(), arrayIds.end(), 0);
        const uint64_t generation = state.generation;
        const uint64_t tag = attentionTransferTag();
        if (!readAttentionSequentialPvOutputGroupAsync(
                arrayIds, tag,
                [this, generation, tag, rows](
                    bool ok, uint64_t callbackTag,
                    const std::vector<double>& values) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    const size_t outputSliceElements =
                        static_cast<size_t>(rows) * arrayOutputSize;
                    if (!ok || callbackTag != tag ||
                        values.size() != outputSliceElements) {
                        finishAttentionWorker(false); return;
                    }
                    const size_t oElements = static_cast<size_t>(rows) *
                        attentionWorker_->dispatch.headDim;
                    if (attentionWorker_->sequentialOPayload.size() != oElements) {
                        attentionWorker_->sequentialOPayload.assign(oElements, 0.0);
                    }
                    for (uint32_t row = 0; row < rows; ++row) {
                        for (uint32_t dim = 0; dim < arrayOutputSize; ++dim) {
                            attentionWorker_->sequentialOPayload[
                                static_cast<size_t>(row) *
                                    attentionWorker_->dispatch.headDim +
                                attentionWorker_->phaseSliceIndex * arrayOutputSize + dim] =
                                values[static_cast<size_t>(row) *
                                    arrayOutputSize + dim];
                        }
                    }
                    if (attentionWorker_->phaseSliceIndex + 1 <
                        attentionOutputSliceCount(*attentionWorker_)) {
                        completeAttentionPvOutputSlice();
                        return;
                    }
                    const std::vector<uint8_t> bytes = attentionDoublesToBytes(
                        attentionWorker_->sequentialOPayload);
                    const size_t halfBytes = bytes.size() / 2;
                    std::vector<uint8_t> low(bytes.begin(), bytes.begin() + halfBytes);
                    std::vector<uint8_t> high(bytes.begin() + halfBytes, bytes.end());
                    attentionLocalWrite(attentionWorker_->oLocal, low,
                        [this, generation, halfBytes, high](bool lowOk) {
                            if (!attentionCallbackGenerationMatches(generation)) return;
                            if (!lowOk) { finishAttentionWorker(false); return; }
                            attentionLocalWrite(
                                attentionWorker_->oLocal + halfBytes, high,
                                [this, generation](bool highOk) {
                                    if (!attentionCallbackGenerationMatches(generation)) return;
                                    if (!highOk) {
                                        finishAttentionWorker(false);
                                        return;
                                    }
                                    completeAttentionPvOutputSlice();
                                });
                        });
                })) finishAttentionWorker(false);
    }

    bool attentionClusterPvMatrixMatches(
            const AttentionWorkerState& state) const {
        return state.clusterPvMatrixResident &&
            state.clusterPvMatrixKvTile == state.kvTileIndex &&
            state.clusterPvMatrixGroupOwner ==
                attentionKvGroupOwnerQueryTile(state);
    }

    void programAttentionClusterPvMatrices() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t halves = attentionClusterHeadHalves(state);
        if (state.phaseSliceIndex == halves) {
            state.clusterPvMatrixResident = true;
            state.clusterPvMatrixKvTile = state.kvTileIndex;
            state.clusterPvMatrixGroupOwner =
                attentionKvGroupOwnerQueryTile(state);
            state.phaseSliceIndex = 0;
            state.index = 0;
            beginAttentionClusterPvInputProgram();
            return;
        }
        const uint32_t half = state.phaseSliceIndex;
        const uint32_t keyCols = attentionKeyCols(state);
        const uint32_t activeColumns = attentionPvActiveColumns(state);
        const uint32_t matrixColumns = activeColumns != 0
            ? activeColumns : arrayInputSize;
        if (state.vPayload.size() !=
                static_cast<size_t>(keyCols) * state.dispatch.headDim) {
            finishAttentionWorker(false);
            return;
        }
        std::vector<double> matrix(
            static_cast<size_t>(arrayOutputSize) * matrixColumns, 0.0);
        for (uint32_t dim = 0; dim < static_cast<uint32_t>(arrayOutputSize);
             ++dim) {
            for (uint32_t key = 0; key < keyCols; ++key) {
                matrix[dim * matrixColumns + key] =
                    state.vPayload[key * state.dispatch.headDim +
                                   half * arrayOutputSize + dim];
            }
        }
        std::vector<uint32_t> arrayIds;
        arrayIds.reserve(attentionClusterPvLanes(state));
        for (uint32_t row = 0; row < attentionClusterPvLanes(state); ++row)
            arrayIds.push_back(attentionClusterPvArray(row, half));
        const uint64_t generation = state.generation;
        const AttentionClusterTag clusterTag = state.clusterTileTag;
        const int32_t contextSlot = state.clusterPvContext;
        const uint64_t transferTag = attentionTransferTag();
        const bool accepted = activeColumns != 0
            ? workerCommandProcessor->programGemmMatrixGroupActiveBankAsync(
                arrayIds, state.clusterPvOperandBank, matrix, activeColumns,
                sizeof(float), transferTag, LastTickCycle,
                [this, generation, half, transferTag](
                    bool ok, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok || callbackTag != transferTag ||
                        attentionWorker_->phaseSliceIndex != half) {
                        if (attentionWorker_) finishAttentionWorker(false);
                        return;
                    }
                    ++attentionWorker_->phaseSliceIndex;
                    programAttentionClusterPvMatrices();
                })
            : workerCommandProcessor->programGemmMatrixGroupBankAsync(
                arrayIds, state.clusterPvOperandBank, matrix, sizeof(float),
                transferTag, LastTickCycle,
                [this, generation, half, transferTag](
                    bool ok, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok || callbackTag != transferTag ||
                        attentionWorker_->phaseSliceIndex != half) {
                        if (attentionWorker_) finishAttentionWorker(false);
                        return;
                    }
                    ++attentionWorker_->phaseSliceIndex;
                    programAttentionClusterPvMatrices();
                });
        if (!accepted) {
            scheduleAttentionClusterOwnerRetry(
                [this, generation, half, clusterTag, contextSlot]() {
                    if (!attentionCallbackGenerationMatches(generation) ||
                        !attentionWorker_ || attentionWorker_->phaseSliceIndex != half ||
                        !attentionClusterCallbackMatches(
                            AttentionClusterContextKind::Pv,
                            contextSlot, clusterTag)) return;
                    programAttentionClusterPvMatrices();
                });
        } else if (activeColumns != 0) {
            statAttentionPvActiveKMatrixElements_->addData(matrix.size());
        }
    }

    void beginAttentionClusterPvOutputSlice() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.phase = AttentionWorkerPhase::PvProgramMatrix;
        const uint32_t keyCols = attentionKeyCols(state);
        const uint64_t generation = state.generation;
        const AttentionClusterTag clusterTag = state.clusterTileTag;
        const int32_t contextSlot = state.clusterPvContext;
        const uint64_t tileBytes = static_cast<uint64_t>(keyCols) *
            state.dispatch.headDim * sizeof(float);
        if (attentionClusterPvMatrixMatches(state)) {
            recordAttentionKvOperandRelease(false);
            state.phaseSliceIndex = 0;
            state.index = attentionClusterWaveRowBegin(
                state.clusterPvWave, attentionClusterPvLanes(state));
            beginAttentionClusterPvInputProgram();
            return;
        }
        if (attentionPvVTileReuse_) {
            statAttentionPvVTileBufferMisses_->addData(1);
            if (tileBytes > attentionPvVTileBufferBytes_) {
                statAttentionPvVTileBufferCapacityRejections_->addData(1);
            }
        }
        attentionLocalRead(
            state.vLocal + (attentionStreamKv(state) ? 0 :
                static_cast<uint64_t>(state.physicalKvTileIndex) *
                    state.dispatch.kvTileRows * state.dispatch.headDim *
                    sizeof(float)),
            tileBytes,
            [this, generation, clusterTag, contextSlot](
                bool ok, const std::vector<uint8_t>& bytes) {
                if (!attentionCallbackGenerationMatches(generation)) return;
                if (!ok ||
                    !attentionClusterCallbackMatches(
                        AttentionClusterContextKind::Pv, contextSlot,
                        clusterTag)) {
                    if (attentionWorker_) finishAttentionWorker(false);
                    return;
                }
                const std::vector<double> v = attentionBytesToDoubles(bytes);
                if (attentionPvVTileReuse_ && bytes.size() <= attentionPvVTileBufferBytes_) {
                    statAttentionPvVTileBufferBytesRead_->addData(bytes.size());
                    attentionWorker_->vPayload = v;
                    attentionWorker_->vTileValid = true;
                    attentionWorker_->vTileTag = attentionWorker_->kvTileIndex;
                    attentionWorker_->vTileGeneration = attentionWorker_->generation;
                    attentionWorker_->vTileGroupOwner =
                        attentionKvGroupOwnerQueryTile(*attentionWorker_);
                    recordAttentionKvOperandRelease(false);
                    attentionWorker_->phaseSliceIndex = 0;
                    programAttentionClusterPvMatrices();
                    return;
                }
                attentionWorker_->vPayload = v;
                recordAttentionKvOperandRelease(false);
                attentionWorker_->phaseSliceIndex = 0;
                programAttentionClusterPvMatrices();
            });
    }

    void beginAttentionClusterPvInputProgram() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        transitionAttentionTilePipeline(
            AttentionTilePipelinePhase::PvInputProgram);
        state.phase = AttentionWorkerPhase::PvProgramInputs;
        state.clusterPvWavefrontProgrammingComplete = false;
        programAttentionClusterPvInput();
    }

    void completeAttentionClusterPvWavefrontCompute() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (!attentionClusterPvRowWavefront_ ||
            !state.clusterPvWavefrontProgrammingComplete ||
            state.arraysPending != 0) return;
        transitionAttentionTilePipeline(
            AttentionTilePipelinePhase::PvOutputReadwrite);
        state.phase = AttentionWorkerPhase::PvReadOutputs;
        state.index = attentionClusterWaveRowBegin(
            state.clusterPvWave, attentionClusterPvLanes(state));
        state.clusterPvOutputIssued = 0;
        state.clusterPvOutputCompleted = 0;
        state.clusterPvOutputInFlight = 0;
        readAttentionClusterPvOutput();
    }

    void launchAttentionClusterPvWavefrontRow(uint32_t logicalRow) {
        if (!attentionWorker_ || !attentionClusterPvRowWavefront_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.phase != AttentionWorkerPhase::PvProgramInputs ||
            state.index != logicalRow) return;
        const uint32_t halves = attentionClusterHeadHalves(state);
        std::vector<uint32_t> arrayIds(halves);
        for (uint32_t half = 0; half < halves; ++half) {
            arrayIds[half] = attentionClusterPvArray(logicalRow, half);
            if (arrayIds[half] >= attentionArrayPending_.size() ||
                attentionArrayPending_[arrayIds[half]] != 0) {
                finishAttentionWorker(false);
                return;
            }
        }
        const uint64_t generation = state.generation;
        const AttentionClusterTag clusterTag = state.clusterTileTag;
        const int32_t contextSlot = state.clusterPvContext;
        const uint32_t activeColumns = attentionPvActiveColumns(state);
        const bool accepted = workerCommandProcessor &&
            workerCommandProcessor->launchGemmArrayGroupActiveBank(
                arrayIds, state.clusterPvOperandBank, 0, activeColumns,
                LastTickCycle,
                [this, generation](uint32_t arrayId, uint64_t) {
                    if (!attentionCallbackGenerationMatches(generation)) {
                        if (!attentionClusterPvArrayActivity_.leave(
                                getCurrentSimCycle())) {
                            statAttentionClusterIllegalTransitions_->addData(1);
                        }
                        return;
                    }
                    handleAttentionArrayDone(arrayId);
                });
        if (!accepted) {
            scheduleAttentionClusterOwnerRetry(
                [this, generation, clusterTag, contextSlot, logicalRow]() {
                    if (!attentionCallbackGenerationMatches(generation) ||
                        !attentionWorker_ ||
                        attentionWorker_->phase !=
                            AttentionWorkerPhase::PvProgramInputs ||
                        attentionWorker_->index != logicalRow ||
                        !attentionClusterCallbackMatches(
                            AttentionClusterContextKind::Pv,
                            contextSlot, clusterTag)) return;
                    launchAttentionClusterPvWavefrontRow(logicalRow);
                });
            return;
        }
        for (uint32_t arrayId : arrayIds) {
            attentionArrayPending_[arrayId] = 1;
            arrayStates[arrayId] = 1;
            if (!attentionClusterPvArrayActivity_.enter(getCurrentSimCycle())) {
                statAttentionClusterIllegalTransitions_->addData(1);
                finishAttentionWorker(false);
                return;
            }
            statAttentionPvArrayOps_->addData(1);
            statAttentionGenericGemmPvOps_->addData(1);
            if (activeColumns != 0) {
                statAttentionPvActiveKLaunches_->addData(1);
                statAttentionPvActiveKColumns_->addData(activeColumns);
            }
        }
        state.arraysPending += arrayIds.size();
        statAttentionClusterPvWavefrontRows_->addData(1);
        ++state.index;
        programAttentionClusterPvInput();
    }

    void prefetchAttentionClusterPvInput(uint32_t logicalRow) {
        if (!attentionPvInputPipeline_ || !attentionWorker_ ||
            logicalRow >= attentionQueryRows(*attentionWorker_) ||
            attentionWorker_->clusterPvNextInputReady ||
            attentionWorker_->clusterPvNextInputInFlight ||
            attentionWorker_->clusterPContext < 0 || sfu == nullptr) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint64_t generation = state.generation;
        const AttentionClusterTag clusterTag = state.clusterTileTag;
        const int32_t contextSlot = state.clusterPContext;
        const uint32_t keyCols = attentionKeyCols(state);
        const uint64_t transferTag = attentionTransferTag();
        state.clusterPvNextInputRow = logicalRow;
        state.clusterPvNextInputInFlight = true;
        if (!sfu->readAttentionPRowAsync(
                static_cast<uint32_t>(state.clusterPContext), state.clusterTileTag,
                static_cast<size_t>(logicalRow) * keyCols, keyCols, transferTag,
                [this, generation, logicalRow, transferTag, clusterTag, contextSlot](
                    bool ok, uint64_t callbackTag, const std::vector<float>& values) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!attentionWorker_ || !ok || callbackTag != transferTag ||
                        attentionWorker_->clusterPvNextInputInFlight == false ||
                        attentionWorker_->clusterPContext != contextSlot ||
                        !(attentionWorker_->clusterTileTag == clusterTag)) {
                        if (attentionWorker_) finishAttentionWorker(false);
                        return;
                    }
                    AttentionWorkerState& callbackState = *attentionWorker_;
                    callbackState.clusterPvNextInput.assign(values.begin(), values.end());
                    callbackState.clusterPvNextInputRow = logicalRow;
                    callbackState.clusterPvNextInputReady = true;
                    callbackState.clusterPvNextInputInFlight = false;
                    if (callbackState.phase == AttentionWorkerPhase::PvProgramInputs &&
                        callbackState.index == logicalRow) {
                        programAttentionClusterPvInput();
                    }
                })) {
            state.clusterPvNextInputInFlight = false;
            scheduleAttentionClusterOwnerRetry(
                [this, generation, logicalRow, clusterTag, contextSlot]() {
                    if (!attentionCallbackGenerationMatches(generation) ||
                        !attentionWorker_ ||
                        attentionWorker_->clusterPContext != contextSlot ||
                        !(attentionWorker_->clusterTileTag == clusterTag) ||
                        logicalRow != attentionWorker_->index + 1 ||
                        attentionWorker_->phase !=
                            AttentionWorkerPhase::PvProgramInputs) return;
                    prefetchAttentionClusterPvInput(logicalRow);
                });
        }
    }

    void programAttentionClusterPvInputPayload(
            uint32_t logicalRow, const std::vector<double>& input) {
        if (!attentionWorker_ ||
            attentionWorker_->phase != AttentionWorkerPhase::PvProgramInputs ||
            attentionWorker_->index != logicalRow) return;
        const uint64_t generation = attentionWorker_->generation;
        const uint64_t tag = attentionTransferTag();
        const uint32_t halves =
            attentionClusterHeadHalves(*attentionWorker_);
        std::vector<uint32_t> arrays(halves);
        for (uint32_t half = 0; half < halves; ++half)
            arrays[half] = attentionClusterPvArray(logicalRow, half);
        if (!programAttentionClusterPRowAsync(
                arrays, input, tag,
                [this, generation, logicalRow, tag](
                    bool programOk, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!programOk || callbackTag != tag ||
                        attentionWorker_->index != logicalRow) {
                        if (attentionWorker_) finishAttentionWorker(false);
                        return;
                    }
                    if (attentionClusterPvRowWavefront_) {
                        launchAttentionClusterPvWavefrontRow(logicalRow);
                        return;
                    }
                    ++attentionWorker_->index;
                    programAttentionClusterPvInput();
                })) {
            scheduleAttentionClusterOwnerRetry(
                [this, logicalRow, input]() {
                    programAttentionClusterPvInputPayload(logicalRow, input);
                });
            return;
        }
        if (attentionPvInputPipeline_) {
            statAttentionPvInputPipelineRows_->addData(1);
            const uint32_t lanes = attentionClusterPvLanes(*attentionWorker_);
            const uint32_t waveEnd = attentionClusterWaveRowBegin(
                attentionWorker_->clusterPvWave, lanes) +
                attentionClusterWaveRows(
                    *attentionWorker_, attentionWorker_->clusterPvWave, lanes);
            if (logicalRow + 1 < waveEnd) {
                prefetchAttentionClusterPvInput(logicalRow + 1);
            }
        }
    }

    void programAttentionClusterPvInput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint32_t lanes = attentionClusterPvLanes(state);
        const uint32_t waveEnd = attentionClusterWaveRowBegin(
            state.clusterPvWave, lanes) + attentionClusterWaveRows(
                state, state.clusterPvWave, lanes);
        if (state.index == waveEnd) {
            if (attentionClusterPvRowWavefront_) {
                state.clusterPvWavefrontProgrammingComplete = true;
                transitionAttentionTilePipeline(
                    AttentionTilePipelinePhase::PvCompute);
                state.phase = AttentionWorkerPhase::PvCompute;
                completeAttentionClusterPvWavefrontCompute();
                return;
            }
            startAttentionClusterPvComputation();
            return;
        }
        const uint32_t logicalRow = state.index;
        if (state.clusterPvNextInputReady &&
            state.clusterPvNextInputRow == logicalRow) {
            std::vector<double> input = std::move(state.clusterPvNextInput);
            state.clusterPvNextInput.clear();
            state.clusterPvNextInputReady = false;
            state.clusterPvNextInputRow = UINT32_MAX;
            programAttentionClusterPvInputPayload(logicalRow, input);
            return;
        }
        if (state.clusterPvNextInputInFlight &&
            state.clusterPvNextInputRow == logicalRow) return;
        const uint32_t keyCols = attentionKeyCols(state);
        const uint64_t generation = state.generation;
        const AttentionClusterTag clusterTag = state.clusterTileTag;
        const int32_t contextSlot = state.clusterPContext;
        const uint64_t transferTag = attentionTransferTag();
        if (state.clusterPContext < 0 || sfu == nullptr ||
            !sfu->readAttentionPRowAsync(
                static_cast<uint32_t>(state.clusterPContext),
                state.clusterTileTag,
                static_cast<size_t>(logicalRow) * keyCols,
                keyCols, transferTag,
            [this, generation, logicalRow, transferTag, clusterTag, contextSlot](
                bool ok, uint64_t callbackTag, const std::vector<float>& values) {
                if (!attentionCallbackGenerationMatches(generation)) return;
                if (!ok || callbackTag != transferTag ||
                    attentionWorker_->index != logicalRow ||
                    attentionWorker_->clusterPContext != contextSlot ||
                    !(attentionWorker_->clusterTileTag == clusterTag)) {
                    if (attentionWorker_) finishAttentionWorker(false);
                    return;
                }
                const std::vector<double> input(values.begin(), values.end());
                programAttentionClusterPvInputPayload(logicalRow, input);
            })) {
            scheduleAttentionClusterOwnerRetry(
                [this, generation, logicalRow]() {
                    if (!attentionCallbackGenerationMatches(generation) ||
                        !attentionWorker_ ||
                        attentionWorker_->phase !=
                            AttentionWorkerPhase::PvProgramInputs ||
                        attentionWorker_->index != logicalRow) return;
                    programAttentionClusterPvInput();
                });
        }
    }

    void prepareAttentionClusterPvOutput() {
        if (!attentionWorker_) return;
        startAttentionClusterPvComputation();
    }

    void startAttentionClusterPvComputation() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        transitionAttentionTilePipeline(AttentionTilePipelinePhase::PvCompute);
        state.phase = AttentionWorkerPhase::PvCompute;
        state.index = attentionClusterWaveRowBegin(
            state.clusterPvWave, attentionClusterPvLanes(state));
        state.arraysPending = 0;
        state.clusterOwnerLaunchNext = 0;
        state.clusterOwnerLaunchEnd = attentionClusterWaveRows(
            state, state.clusterPvWave, attentionClusterPvLanes(state)) *
            attentionClusterHeadHalves(state);
        state.clusterOwnerLaunchQk = false;
        state.clusterPvOutputIssued = 0;
        state.clusterPvOutputCompleted = 0;
        state.clusterPvOutputInFlight = 0;
        pumpAttentionClusterOwnerLaunch();
    }

    void readAttentionClusterPvOutput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.phase != AttentionWorkerPhase::PvReadOutputs) return;
        const uint32_t phaseSliceCount = attentionOutputSliceCount(state);
        if (state.clusterPvOutputCompleted == phaseSliceCount &&
            state.clusterPvOutputInFlight == 0) {
            if (state.index + 1 == attentionQueryRows(state) &&
                state.attentionClusterOCommitsPending != 0) return;
            state.phaseSliceIndex = 0;
            ++state.index;
            const uint32_t lanes = attentionClusterPvLanes(state);
            const uint32_t waveEnd = attentionClusterWaveRowBegin(
                state.clusterPvWave, lanes) + attentionClusterWaveRows(
                    state, state.clusterPvWave, lanes);
            if (state.index == waveEnd &&
                state.clusterPvWave + 1 < attentionClusterPvWaves(state)) {
                ++state.clusterPvWave;
                state.index = attentionClusterWaveRowBegin(
                    state.clusterPvWave, lanes);
                state.clusterPvOutputIssued = 0;
                state.clusterPvOutputCompleted = 0;
                beginAttentionClusterPvInputProgram();
                return;
            }
            if (state.index == attentionQueryRows(state)) {
                completeAttentionPvOutputSlice();
                return;
            }
            state.clusterPvOutputIssued = 0;
            state.clusterPvOutputCompleted = 0;
            readAttentionClusterPvOutput();
            return;
        }
        if (state.clusterPvOutputIssued != 0 ||
            state.clusterPvOutputInFlight != 0) return;
        const uint32_t logicalRow = state.index;
        const uint64_t tag = attentionTransferTag();
        const uint64_t generation = state.generation;
        const uint32_t halves = attentionClusterHeadHalves(state);
        std::vector<uint32_t> arrays(halves);
        for (uint32_t half = 0; half < halves; ++half)
            arrays[half] = attentionClusterPvArray(logicalRow, half);
        state.clusterPvOutputIssued = halves;
        state.clusterPvOutputInFlight = 1;
        if (!readAttentionClusterPvGroupAsync(
                arrays, tag,
                [this, generation, logicalRow, phaseSliceCount, tag](
                    bool ok, uint64_t callbackTag,
                    const std::vector<double>& values) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok || callbackTag != tag || !attentionWorker_ ||
                        attentionWorker_->index != logicalRow ||
                        values.size() != static_cast<size_t>(phaseSliceCount) * 16) {
                        if (attentionWorker_) finishAttentionWorker(false);
                        return;
                    }
                    AttentionWorkerState& callbackState = *attentionWorker_;
                    const uint32_t queryContext = callbackState.queryTileIndex %
                        attentionClusterConfig_.groupSize;
                    const int32_t oSlot =
                        callbackState.clusterOContextSlots[queryContext];
                    if (oSlot < 0 ||
                        callbackState.outputScales.size() <= logicalRow ||
                        callbackState.clusterPvOutputInFlight != 1) {
                        finishAttentionWorker(false);
                        return;
                    }
                    if (attentionPvORowFusion_) {
                        std::vector<float> row(values.size());
                        std::transform(values.begin(), values.end(), row.begin(),
                            [](double value) { return static_cast<float>(value); });
                        uint64_t operation = 0;
                        uint64_t readyCycle = 0;
                        if (!attentionOAccumulator_.submitRow(
                                static_cast<uint32_t>(oSlot),
                                callbackState.clusterOTags[queryContext],
                                callbackState.kvTileIndex, logicalRow,
                                callbackState.outputScales[logicalRow], row,
                                LastTickCycle, &operation, &readyCycle)) {
                            finishAttentionWorker(false);
                            return;
                        }
                        ++callbackState.attentionClusterOCommitsPending;
                        statAttentionClusterOAccumulateSegments_->addData(phaseSliceCount);
                        statAttentionClusterOFusedRows_->addData(1);
                        statAttentionClusterOFusedBytes_->addData(
                            row.size() * sizeof(float));
                        if (callbackState.kvTileIndex != 0)
                            statAttentionClusterOScaleSegments_->addData(phaseSliceCount);
                    } else {
                        for (uint32_t phaseSliceIndex = 0; phaseSliceIndex < phaseSliceCount; ++phaseSliceIndex) {
                            std::vector<float> segment(16);
                            std::transform(
                                values.begin() + static_cast<size_t>(phaseSliceIndex) * 16,
                                values.begin() + static_cast<size_t>(phaseSliceIndex + 1) * 16,
                                segment.begin(),
                                [](double value) { return static_cast<float>(value); });
                            uint64_t operation = 0;
                            uint64_t readyCycle = 0;
                            if (!attentionOAccumulator_.submitSegment(
                                    static_cast<uint32_t>(oSlot),
                                    callbackState.clusterOTags[queryContext],
                                    callbackState.kvTileIndex, logicalRow, phaseSliceIndex,
                                    callbackState.outputScales[logicalRow], segment,
                                    LastTickCycle, &operation, &readyCycle)) {
                                finishAttentionWorker(false);
                                return;
                            }
                            ++callbackState.attentionClusterOCommitsPending;
                            statAttentionClusterOAccumulateSegments_->addData(1);
                            if (callbackState.kvTileIndex != 0)
                                statAttentionClusterOScaleSegments_->addData(1);
                        }
                    }
                    callbackState.clusterPvOutputInFlight = 0;
                    callbackState.clusterPvOutputCompleted = phaseSliceCount;
                    readAttentionClusterPvOutput();
                })) {
            state.clusterPvOutputIssued = 0;
            state.clusterPvOutputInFlight = 0;
            scheduleAttentionClusterOwnerRetry(
                [this, generation, logicalRow]() {
                    if (!attentionCallbackGenerationMatches(generation) ||
                        !attentionWorker_ ||
                        attentionWorker_->phase !=
                            AttentionWorkerPhase::PvReadOutputs ||
                        attentionWorker_->index != logicalRow) return;
                    readAttentionClusterPvOutput();
                });
        }
    }

    void continueAttentionAfterSoftmaxAndPvMatrix() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (!state.attentionSoftmaxComplete) return;
        if (!state.attentionPvMatrixComplete) {
            if (state.phase == AttentionWorkerPhase::Softmax) {
                statAttentionPvMatrixOverlapWaits_->addData(1);
                transitionAttentionTilePipeline(
                    AttentionTilePipelinePhase::PvMatrixProgram);
                state.phase = AttentionWorkerPhase::PvProgramMatrix;
            }
            return;
        }
        if (state.phase == AttentionWorkerPhase::Softmax) {
            statAttentionPvMatrixOverlapHits_->addData(1);
        }
        beginAttentionPvInputProgram();
    }

    void beginAttentionPvInputProgram() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        transitionAttentionTilePipeline(AttentionTilePipelinePhase::PvInputProgram);
        state.phase = AttentionWorkerPhase::PvProgramInputs;
        state.index = 0;
        if (state.phaseSliceIndex != 0 && attentionPvInputResidentForCurrentTile(state)) {
            statAttentionPvInputResidencyHits_->addData(1);
            statAttentionPvInputResidencyRowsReused_->addData(
                attentionQueryRows(state));
            state.index = attentionQueryRows(state);
            if (attentionPvEarlyCompute_ && state.kvTileIndex == 0) {
                state.attentionPvPreparationComplete = false;
                state.arraysPending = 0;
                std::fill(
                    attentionArrayPending_.begin(),
                    attentionArrayPending_.end(), 0);
                for (uint32_t arrayId = 0;
                     arrayId < attentionQueryRows(state) && attentionWorker_;
                     ++arrayId) {
                    startAttentionPvArrayComputation(arrayId);
                }
                if (attentionWorker_) completeAttentionPvPreparation();
            } else {
                completeAttentionPvInputProgram();
            }
            return;
        }
        if (attentionPvEarlyCompute_) {
            state.arraysPending = 0;
            state.attentionPvPreparationComplete = false;
            std::fill(attentionArrayPending_.begin(), attentionArrayPending_.end(), 0);
        }
        programAttentionPvInput();
    }

    void completeAttentionPvMatrixProgram() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.attentionPvMatrixComplete = true;
        if (attentionPvMatrixSoftmaxOverlap_ && state.phaseSliceIndex == 0) {
            continueAttentionAfterSoftmaxAndPvMatrix();
            return;
        }
        beginAttentionPvInputProgram();
    }

    void startAttentionPvArrayComputation(uint32_t arrayId) {
        if (!attentionWorker_ || arrayId >= attentionArrayPending_.size() ||
            attentionArrayPending_[arrayId] != 0) {
            finishAttentionWorker(false);
            return;
        }
        attentionArrayPending_[arrayId] = 1;
        arrayStates[arrayId] = 1;
        attentionWorker_->arraysPending += 1;
        const uint64_t outputMode = attentionWorker_->kvTileIndex == 0 ? 0 : 1;
        const uint32_t activeColumns =
            attentionPvActiveColumns(*attentionWorker_);
        if (!launchAttentionGemmArray(
                arrayId, outputMode, false, activeColumns)) {
            finishAttentionWorker(false);
            return;
        }
        statAttentionPvArrayOps_->addData(1);
        statAttentionPvEarlyComputeArrays_->addData(1);
    }

    void programAttentionPvInput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.index == attentionQueryRows(state)) {
            if (attentionPvInputPipeline_ && state.attentionPvInputsPending != 0) return;
            completeAttentionPvInputProgram();
            return;
        }
        if (attentionPvInputPipeline_ && state.index == 0) {
            state.attentionPvInputsPending = 0;
        }
        const uint32_t arrayId = state.index;
        if (attentionPvInputPipeline_) {
            state.index += 1;
            state.attentionPvInputsPending += 1;
        }
        const uint32_t keyCols = attentionKeyCols(state);
        attentionLocalRead(state.spLocal + static_cast<uint64_t>(arrayId) * keyCols * sizeof(float),
            keyCols * sizeof(float), [this, arrayId](bool ok, const std::vector<uint8_t>& bytes) {
                if (!attentionWorker_ || !ok) { finishAttentionWorker(false); return; }
                std::vector<double> input(arrayInputSize, 0.0);
                const std::vector<double> p = attentionBytesToDoubles(bytes);
                std::copy(p.begin(), p.end(), input.begin());
                const std::vector<double>& programmedInput =
                    attentionPvCompactInput_ ? p : input;
                const uint64_t tag = attentionTransferTag();
                const uint32_t activeColumns =
                    attentionPvActiveColumns(*attentionWorker_);
                const bool accepted = activeColumns != 0 ?
                    programAttentionGemmInputActiveAsync(
                        arrayId, p, activeColumns, sizeof(float), tag,
                        [this, tag, arrayId](bool programOk, uint64_t callbackTag) {
                            if (!attentionWorker_ || !programOk || callbackTag != tag) {
                                finishAttentionWorker(false); return;
                            }
                            if (attentionPvEarlyCompute_ &&
                                attentionWorker_->kvTileIndex == 0) {
                                startAttentionPvArrayComputation(arrayId);
                                if (!attentionWorker_) return;
                            }
                            if (attentionPvInputPipeline_) {
                                if (attentionWorker_->attentionPvInputsPending == 0) {
                                    finishAttentionWorker(false); return;
                                }
                                attentionWorker_->attentionPvInputsPending -= 1;
                                if (attentionWorker_->index ==
                                        attentionQueryRows(*attentionWorker_) &&
                                    attentionWorker_->attentionPvInputsPending == 0) {
                                    completeAttentionPvInputProgram();
                                }
                                return;
                            }
                            attentionWorker_->index += 1;
                            programAttentionPvInput();
                        }) :
                    programAttentionGemmInputAsync(arrayId, programmedInput, sizeof(float), tag,
                        [this, tag, arrayId](bool programOk, uint64_t callbackTag) {
                            if (!attentionWorker_ || !programOk || callbackTag != tag) {
                                finishAttentionWorker(false); return;
                            }
                            if (attentionPvEarlyCompute_ &&
                                attentionWorker_->kvTileIndex == 0) {
                                startAttentionPvArrayComputation(arrayId);
                                if (!attentionWorker_) return;
                            }
                            if (attentionPvInputPipeline_) {
                                if (attentionWorker_->attentionPvInputsPending == 0) {
                                    finishAttentionWorker(false); return;
                                }
                                attentionWorker_->attentionPvInputsPending -= 1;
                                if (attentionWorker_->index ==
                                        attentionQueryRows(*attentionWorker_) &&
                                    attentionWorker_->attentionPvInputsPending == 0) {
                                    completeAttentionPvInputProgram();
                                }
                                return;
                            }
                            attentionWorker_->index += 1;
                            programAttentionPvInput();
                        });
                if (!accepted) {
                    finishAttentionWorker(false);
                    return;
                }
                if (attentionPvInputPipeline_) {
                    statAttentionPvInputPipelineRows_->addData(1);
                    programAttentionPvInput();
                }
            });
    }

    void completeAttentionPvInputProgram() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (attentionPvInputPipeline_ && state.attentionPvInputsPending != 0) {
            finishAttentionWorker(false);
            return;
        }
        if (attentionPvInputResidency_ && state.phaseSliceIndex == 0) {
            state.attentionPvInputResidentValid = true;
            state.attentionPvInputResidentQueryTile = state.queryTileIndex;
            state.attentionPvInputResidentKvTile = state.kvTileIndex;
        }
        if (attentionPvEarlyCompute_ && state.kvTileIndex == 0) {
            completeAttentionPvPreparation();
            return;
        }
        transitionAttentionTilePipeline(
            AttentionTilePipelinePhase::PvRestoreOutput);
        state.phase = AttentionWorkerPhase::PvRestoreOutput;
        state.index = 0;
        prepareAttentionPvOutput();
    }

    void completeAttentionPvPreparation() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.attentionPvPreparationComplete = true;
        transitionAttentionTilePipeline(AttentionTilePipelinePhase::PvCompute);
        state.phase = AttentionWorkerPhase::PvCompute;
        state.index = 0;
        if (attentionArrayPending_.empty() || attentionArrayPending_[0] == 0) {
            transitionAttentionTilePipeline(
                AttentionTilePipelinePhase::PvOutputReadwrite);
            state.phase = AttentionWorkerPhase::PvReadOutputs;
            state.attentionPvOutputWritesPending = 0;
            state.attentionPvOutputWriteRetry = false;
            state.attentionPvOutputWriteBytes.clear();
            readAttentionPvOutput();
        }
    }

    void startAttentionPvComputation() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        transitionAttentionTilePipeline(AttentionTilePipelinePhase::PvCompute);
        state.phase = AttentionWorkerPhase::PvCompute;
        const uint32_t activeArrays = attentionQueryRows(state);
        state.arraysPending = activeArrays;
        for (uint32_t arrayId = 0; arrayId < activeArrays; ++arrayId) {
            attentionArrayPending_[arrayId] = 1;
            arrayStates[arrayId] = 1;
            const uint64_t outputMode = state.kvTileIndex == 0 ? 0 : 1;
            const uint32_t activeColumns = attentionPvActiveColumns(state);
            if (!launchAttentionGemmArray(
                    arrayId, outputMode, false, activeColumns)) {
                finishAttentionWorker(false);
                return;
            }
            statAttentionPvArrayOps_->addData(1);
        }
    }

    void programAttentionPvRestoredOutput(
        uint32_t arrayId, const std::vector<uint8_t>& bytes) {
        if (!attentionWorker_ || bytes.size() != 16 * sizeof(float) ||
            attentionWorker_->outputScales.size() !=
                attentionQueryRows(*attentionWorker_)) {
            finishAttentionWorker(false);
            return;
        }
        std::vector<double> output = attentionBytesToDoubles(bytes);
        const double scale = attentionWorker_->outputScales[arrayId];
        for (double& value : output) value *= scale;
        const uint64_t tag = attentionTransferTag();
        if (!writeAttentionGemmOutputAsync(arrayId, output, sizeof(float), tag,
                [this, tag, arrayId](bool writeOk, uint64_t callbackTag) {
                    if (!attentionWorker_ || !writeOk || callbackTag != tag) {
                        finishAttentionWorker(false); return;
                    }
                    if (attentionPvEarlyCompute_) {
                        startAttentionPvArrayComputation(arrayId);
                        if (!attentionWorker_) return;
                    }
                    if (attentionPvRestorePipeline_) {
                        if (attentionWorker_->attentionPvRestoresPending == 0) {
                            finishAttentionWorker(false); return;
                        }
                        attentionWorker_->attentionPvRestoresPending -= 1;
                        if (attentionWorker_->index ==
                                attentionQueryRows(*attentionWorker_) &&
                            attentionWorker_->attentionPvRestoresPending == 0) {
                            completeAttentionPvRestore();
                        }
                        return;
                    }
                    attentionWorker_->index += 1;
                    prepareAttentionPvOutput();
                })) {
            finishAttentionWorker(false);
            return;
        }
        if (attentionPvRestorePipeline_) {
            statAttentionPvRestorePipelineRows_->addData(1);
            prepareAttentionPvOutput();
        }
    }

    void prepareAttentionPvOutput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (state.kvTileIndex == 0) {
            if (attentionPvEarlyCompute_) {
                completeAttentionPvPreparation();
                return;
            }
            startAttentionPvComputation();
            return;
        }
        if (state.index == attentionQueryRows(state)) {
            if (attentionPvRestorePipeline_ && state.attentionPvRestoresPending != 0) return;
            completeAttentionPvRestore();
            return;
        }
        if (attentionPvRestorePipeline_ && state.index == 0) {
            state.attentionPvRestoresPending = 0;
        }
        const uint32_t arrayId = state.index;
        if (attentionOAccumulatorCBuffer_) {
            const uint32_t accumulatorRow =
                attentionOAccumulatorRow(state, arrayId);
            const uint64_t tag = attentionTransferTag();
            state.attentionPvRestoreReadRetry = false;
            if (!workerCommandProcessor->readAttentionAccumulatorRowAsync(
                    accumulatorRow, state.generation, tag, LastTickCycle,
                    [this, arrayId, tag](
                        bool ok, uint64_t callbackTag,
                        const std::vector<uint8_t>& bytes) {
                        if (!attentionWorker_ || !ok || callbackTag != tag) {
                            finishAttentionWorker(false);
                            return;
                        }
                        statAttentionOAccumulatorRestores_->addData(1);
                        statAttentionOAccumulatorBytes_->addData(bytes.size());
                        programAttentionPvRestoredOutput(arrayId, bytes);
                    })) {
                state.attentionPvRestoreReadRetry = true;
                return;
            }
            if (attentionPvRestorePipeline_) {
                state.index += 1;
                state.attentionPvRestoresPending += 1;
            }
            return;
        }
        if (attentionPvRestorePipeline_) {
            state.index += 1;
            state.attentionPvRestoresPending += 1;
        }
        const uint64_t addr = state.oLocal +
            (static_cast<uint64_t>(arrayId) * state.dispatch.headDim +
             state.phaseSliceIndex * 16) * sizeof(float);
        attentionLocalRead(
            addr, 16 * sizeof(float),
            [this, arrayId](bool ok, const std::vector<uint8_t>& bytes) {
                if (!attentionWorker_ || !ok) {
                    finishAttentionWorker(false);
                    return;
                }
                programAttentionPvRestoredOutput(arrayId, bytes);
            });
    }

    void completeAttentionPvRestore() {
        if (!attentionWorker_) return;
        if (attentionPvRestorePipeline_ &&
            attentionWorker_->attentionPvRestoresPending != 0) {
            finishAttentionWorker(false);
            return;
        }
        if (attentionPvEarlyCompute_) completeAttentionPvPreparation();
        else startAttentionPvComputation();
    }

    void beginAttentionGroupedQueryTileAfterQueryLoad() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        beginAttentionTilePipeline();
        const DmaConsumerMetadata consumerProgress = attentionDmaConsumerMetadata(
            state, state.queryTileIndex, state.kvTileIndex, DmaOperand::Unknown);
        globalMem->dma_update_consumer_progress(
            attentionKvHostAddr(state, state.dispatch.kAddr),
            attentionKvHostAddr(state, state.dispatch.vAddr), consumerProgress);
        recordAttentionInterTilePhase(AttentionInterTilePhase::KvLoad);
        beginAttentionKvTile();
    }

    void beginAttentionNextGroupedQueryTile() {
        if (!attentionWorker_ ||
            !attentionHasNextGroupedQuery(*attentionWorker_)) {
            finishAttentionWorker(false);
            return;
        }
        AttentionWorkerState& state = *attentionWorker_;
        const bool hasAheadOperands = attentionCrossTileOperandPipeline_ &&
            state.aheadOperands.phase != AttentionAheadOperandPhase::Idle;
        state.queryTileIndex += 1;
        if (!selectAttentionQueryStorage(state)) {
            finishAttentionWorker(false);
            return;
        }
        state.phaseSliceIndex = 0;
        state.physicalKvTileIndex = attentionPhysicalKvTile(state);
        state.attentionWaitingKvBuffer = UINT32_MAX;
        state.attentionWaitingForPrefetch = false;
        statAttentionKvPairReuseTiles_->addData(1);
        statAttentionKvPairReuseBytes_->addData(
            static_cast<uint64_t>(2) * attentionKeyCols(state) *
                state.dispatch.headDim * sizeof(float));
        if (attentionClusterEnable_) {
            if (promoteAttentionClusterAhead()) return;
            state.clusterAheadPromotionWaiting = true;
            statAttentionClusterPromotionWaits_->addData(1);
            pumpAttentionClusterAhead();
            return;
        }
        if (state.kvTileIndex != 0 && hasAheadOperands) {
            AttentionAheadOperandContext& ahead = state.aheadOperands;
            if (ahead.generation != state.generation ||
                ahead.queryTileIndex != state.queryTileIndex ||
                ahead.kvTileIndex != state.kvTileIndex ||
                ahead.physicalKvTileIndex != state.physicalKvTileIndex) {
                statAttentionCrossTileOperandTagMismatches_->addData(1);
                finishAttentionWorker(false);
                return;
            }
            if (ahead.phase == AttentionAheadOperandPhase::Ready) {
                promoteAttentionAheadOperands();
                return;
            }
            ahead.commitWaiting = true;
            ahead.waitStartTick = getCurrentSimCycle();
            statAttentionCrossTileOperandWaits_->addData(1);
            pumpAttentionAheadOperands();
            return;
        }
        if (state.kvTileIndex != 0) {
            beginAttentionGroupedQueryTileAfterQueryLoad();
            return;
        }
        state.phase = AttentionWorkerPhase::LoadingQ;
        const uint64_t generation = state.generation;
        globalMem->dma_read_from_host_to_globalmem(
            state.dispatch.qAddr + static_cast<uint64_t>(state.queryTileIndex) *
                state.dispatch.queryTileRows * state.dispatch.headDim * sizeof(float),
            static_cast<uint64_t>(attentionQueryRows(state)) *
                state.dispatch.headDim * sizeof(float),
            state.qLocal, [this, generation](bool ok) {
                if (!attentionCallbackGenerationMatches(generation)) return;
                if (!ok) {
                    finishAttentionWorker(false);
                    return;
                }
                recordAttentionInterTilePhase(AttentionInterTilePhase::QueryLoad);
                beginAttentionGroupedQueryTileAfterQueryLoad();
            }, DmaRequestKind::AttentionQuery);
    }

    void writeAttentionQueryTileOutput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.phase = AttentionWorkerPhase::OutputDma;
        if (!attentionClusterEnable_) {
            dmaAttentionQueryTileOutput();
            return;
        }
        const uint32_t queryContext =
            state.queryTileIndex % attentionClusterConfig_.groupSize;
        const int32_t slot = state.clusterOContextSlots[queryContext];
        uint64_t drainId = 0;
        uint64_t readyCycle = 0;
        if (slot < 0 || state.attentionClusterODrainPending ||
            !attentionOAccumulator_.requestDrain(
                static_cast<uint32_t>(slot), state.clusterOTags[queryContext],
                LastTickCycle, &drainId, &readyCycle)) {
            finishAttentionWorker(false);
            return;
        }
        (void)readyCycle;
        state.attentionClusterODrainPending = true;
        state.attentionClusterODrainId = drainId;
        statAttentionClusterODrainRequests_->addData(1);
    }

    void completeAttentionQueryTileOutputDma(uint64_t generation) {
        if (!attentionCallbackGenerationMatches(generation)) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (attentionReuseWindowQkBridge_) {
            output->output(
                "[ATTENTION_REUSE_WINDOW_E2E] core=%" PRIu64
                " cycles=%" PRIu64 " start=%" PRIu64 " end=%" PRIu64
                " qk_end=%" PRIu64 " pv_cycles=%" PRIu64 "\n",
                coreID, LastTickCycle - state.reuseWindowQkStartCycle,
                state.reuseWindowQkStartCycle, LastTickCycle,
                state.reuseWindowQkEndCycle, state.reuseWindowPvCycles);
        }
        if (attentionClusterEnable_ && !releaseAttentionClusterOContext()) {
            statAttentionClusterIllegalTransitions_->addData(1);
            finishAttentionWorker(false);
            return;
        }
        recordAttentionInterTilePhase(AttentionInterTilePhase::OutputDma);
        statAttentionWorkerOutputDmaAckTick_->addData(getCurrentSimCycle());
        const bool finalOutput = state.queryTileIndex + 1 == attentionQueryTileCount(state);
        if (attentionTileTrace_ || finalOutput) {
            const uint32_t numKvTiles = attentionKvTileCountForQueryTile(state);
            const uint32_t completedKvTile = state.kvTileIndex < numKvTiles
                ? state.kvTileIndex : numKvTiles - 1;
            traceAttentionMilestone(
                "worker", finalOutput ? "final_output_dma_ack" : "output_dma_ack",
                "done", state.dispatch.jobId, state.dispatch.tag,
                state.queryTileIndex, completedKvTile);
        }
        if (attentionHasNextGroupedQuery(state)) {
            beginAttentionNextGroupedQueryTile();
            return;
        }
        const uint32_t numQueryTiles = attentionQueryTileCount(state);
        if (++state.queryTileIndex < numQueryTiles) beginAttentionQueryTile();
        else finishAttentionWorker(true);
    }

    void issueAttentionOutputDmaRows(uint64_t generation) {
        if (!attentionCallbackGenerationMatches(generation)) return;
        AttentionWorkerState& state = *attentionWorker_;
        constexpr uint32_t kScatterDmaWindow = 16;
        const uint32_t rows = attentionQueryRows(state);
        const uint64_t rowBytes =
            static_cast<uint64_t>(state.dispatch.headDim) * sizeof(float);
        const uint32_t groupSize =
            state.dispatch.numQueryHeads / state.dispatch.numKvHeads;
        const uint32_t groupRowBegin = state.dispatch.groupQueryRowBegin +
            state.queryTileIndex * state.dispatch.queryTileRows;
        const uint32_t queryHead = state.dispatch.kvHeadIndex * groupSize +
            groupRowBegin / state.dispatch.queryLength;
        const uint32_t queryRow = groupRowBegin % state.dispatch.queryLength;

        while (state.attentionOutputDmaRowsInFlight < kScatterDmaWindow &&
               state.attentionOutputDmaNextRow < rows) {
            const uint32_t row = state.attentionOutputDmaNextRow++;
            ++state.attentionOutputDmaRowsInFlight;
            const uint64_t source =
                state.oLocal + static_cast<uint64_t>(row) * rowBytes;
            const uint64_t destination = state.dispatch.oAddr +
                (static_cast<uint64_t>(queryRow + row) *
                     state.dispatch.numQueryHeads + queryHead) * rowBytes;
            globalMem->dma_write_from_globalmem_to_host(
                source, destination, rowBytes,
                [this, generation](bool ok) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    AttentionWorkerState& callbackState = *attentionWorker_;
                    if (!ok || callbackState.attentionOutputDmaRowsPending == 0 ||
                        callbackState.attentionOutputDmaRowsInFlight == 0) {
                        finishAttentionWorker(false);
                        return;
                    }
                    --callbackState.attentionOutputDmaRowsInFlight;
                    if (--callbackState.attentionOutputDmaRowsPending == 0) {
                        completeAttentionQueryTileOutputDma(generation);
                        return;
                    }
                    issueAttentionOutputDmaRows(generation);
                }, DmaRequestKind::AttentionOutput);
        }
    }

    void dmaAttentionQueryTileOutput() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        state.phase = AttentionWorkerPhase::OutputDma;
        const uint32_t rows = attentionQueryRows(state);
        const uint64_t rowBytes =
            static_cast<uint64_t>(state.dispatch.headDim) * sizeof(float);
        const uint64_t generation = state.generation;
        if (state.dispatch.numQueryHeads <= 1 ||
            state.dispatch.queryLength == 0) {
            globalMem->dma_write_from_globalmem_to_host(
                state.oLocal, state.dispatch.oAddr +
                    (static_cast<uint64_t>(state.dispatch.groupQueryRowBegin) +
                     static_cast<uint64_t>(state.queryTileIndex) *
                        state.dispatch.queryTileRows) * rowBytes,
                static_cast<uint64_t>(rows) * rowBytes,
                [this, generation](bool ok) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok) { finishAttentionWorker(false); return; }
                    completeAttentionQueryTileOutputDma(generation);
                }, DmaRequestKind::AttentionOutput);
            return;
        }

        const uint32_t groupSize =
            state.dispatch.numQueryHeads / state.dispatch.numKvHeads;
        const uint32_t groupRowBegin = state.dispatch.groupQueryRowBegin +
            state.queryTileIndex * state.dispatch.queryTileRows;
        const uint32_t queryHead = state.dispatch.kvHeadIndex * groupSize +
            groupRowBegin / state.dispatch.queryLength;
        const uint32_t queryRow = groupRowBegin % state.dispatch.queryLength;
        if (queryHead >= state.dispatch.numQueryHeads ||
            queryRow + rows > state.dispatch.queryLength) {
            finishAttentionWorker(false);
            return;
        }
        state.attentionOutputDmaRowsPending = rows;
        state.attentionOutputDmaNextRow = 0;
        state.attentionOutputDmaRowsInFlight = 0;
        issueAttentionOutputDmaRows(generation);
    }

    void completeAttentionPvOutputSlice() {
        if (!attentionWorker_) return;
        AttentionWorkerState& state = *attentionWorker_;
        if (attentionClusterEnable_) {
            state.clusterPvWave = 0;
            state.phaseSliceIndex = attentionOutputSliceCount(state);
        } else if (attentionSequential64Enable_) {
            state.phaseSliceIndex += attentionSequentialPvOutputSliceCount(state);
        } else {
            ++state.phaseSliceIndex;
        }
        if (state.phaseSliceIndex < attentionOutputSliceCount(state)) {
                transitionAttentionTilePipeline(
                    AttentionTilePipelinePhase::PvMatrixProgram);
                beginAttentionPvOutputSlice();
        } else {
                if (attentionClusterEnable_ && !releaseAttentionClusterPv()) {
                    statAttentionClusterIllegalTransitions_->addData(1);
                    finishAttentionWorker(false);
                    return;
                }
                finishAttentionTilePipeline();
                if (attentionKvDoubleBuffer_ &&
                    state.activeKvBuffer < state.kvBuffers.size()) {
                    AttentionKvBufferState& completed =
                        state.kvBuffers[state.activeKvBuffer];
                    if (completed.consuming &&
                        completed.kvTileIndex == state.kvTileIndex &&
                        completed.physicalKvTileIndex == state.physicalKvTileIndex) {
                        completed = {};
                    }
                }
                statAttentionWorkerPvTileCompleteTick_->addData(getCurrentSimCycle());
                traceAttentionTileCompletion(
                    "pv_tile_complete", "final_pv_tile_complete", state);
                const uint32_t numKvTiles = attentionKvTileCountForQueryTile(state);
                const bool hasNextKvTile = state.kvTileIndex + 1 < numKvTiles;
                if (attentionOAccumulatorCBuffer_ && !hasNextKvTile) {
                    if (!workerCommandProcessor->endAttentionStorageSession(
                            state.generation)) {
                        finishAttentionWorker(false);
                        return;
                    }
                    state.attentionOAccumulatorStorageActive = false;
                }
                const bool hasNextQueryTile =
                    !hasNextKvTile && state.queryTileIndex + 1 < attentionQueryTileCount(state);
                if (attentionHasNextGroupedQuery(state)) {
                    beginAttentionInterTile();
                    if (hasNextKvTile) beginAttentionNextGroupedQueryTile();
                    else writeAttentionQueryTileOutput();
                    return;
                }
                if (attentionKvPairReuseActive(state) &&
                    state.queryTileIndex != attentionKvGroupOwnerQueryTile(state) &&
                    hasNextKvTile) {
                    beginAttentionInterTile();
                    state.queryTileIndex = attentionKvGroupOwnerQueryTile(state);
                    if (!selectAttentionQueryStorage(state)) {
                        finishAttentionWorker(false);
                        return;
                    }
                    state.kvTileIndex += 1;
                    state.physicalKvTileIndex = attentionPhysicalKvTile(state);
                    loadAttentionKvTile();
                    return;
                }
                if (hasNextKvTile || hasNextQueryTile) beginAttentionInterTile();
                if (hasNextQueryTile &&
                    (!attentionKvPairReuseActive(state) ||
                     !attentionHasNextGroupedQuery(state))) {
                    launchAttentionCrossQueryPrefetch();
                }
                if (++state.kvTileIndex < numKvTiles) {
                    state.physicalKvTileIndex = attentionPhysicalKvTile(state);
                    loadAttentionKvTile();
                    return;
                }
                writeAttentionQueryTileOutput();
        }
    }

    void issueAttentionPvOutputWrite() {
        if (!attentionWorker_ || !attentionWorker_->attentionPvOutputWriteRetry) return;
        AttentionWorkerState& state = *attentionWorker_;
        const uint64_t jobId = state.dispatch.jobId;
        const uint32_t phaseSliceIndex = state.phaseSliceIndex;
        const uint32_t issuedIndex = state.index;
        const uint64_t addr = state.attentionPvOutputWriteAddr;
        const uint32_t accumulatorRow =
            state.attentionPvOutputWriteAccumulatorRow;
        const bool toCBuffer = state.attentionPvOutputWriteToCBuffer;
        const std::vector<uint8_t> bytes = state.attentionPvOutputWriteBytes;
        const size_t writeBytes = bytes.size();
        const uint64_t tag = attentionTransferTag();
        state.attentionPvOutputWriteRetry = false;
        state.attentionPvOutputWritesPending += 1;
        state.index += 1;
        const auto completion =
            [this, tag, jobId, toCBuffer, writeBytes](
                bool ok, uint64_t callbackTag) {
                if (!attentionWorker_ || attentionWorker_->dispatch.jobId != jobId) return;
                if (!ok || callbackTag != tag ||
                    attentionWorker_->attentionPvOutputWritesPending == 0) {
                    finishAttentionWorker(false); return;
                }
                if (toCBuffer) {
                    statAttentionOAccumulatorStores_->addData(1);
                    statAttentionOAccumulatorBytes_->addData(writeBytes);
                }
                AttentionWorkerState& callbackState = *attentionWorker_;
                callbackState.attentionPvOutputWritesPending -= 1;
                if (callbackState.index == attentionQueryRows(callbackState) &&
                    callbackState.attentionPvOutputWritesPending == 0 &&
                    !callbackState.attentionPvOutputWriteRetry) {
                    completeAttentionPvOutputSlice();
                }
            };
        const bool accepted = toCBuffer ?
            workerCommandProcessor->writeAttentionAccumulatorRowAsync(
                accumulatorRow, bytes, state.generation, tag, LastTickCycle,
                completion) :
            globalMem->localWriteAsync(
                addr, bytes, LocalMemoryClient::RoCC, tag, completion);
        if (!accepted) {
            if (!attentionWorker_ || attentionWorker_->dispatch.jobId != jobId ||
                attentionWorker_->phase != AttentionWorkerPhase::PvReadOutputs ||
                attentionWorker_->phaseSliceIndex != phaseSliceIndex || attentionWorker_->index != issuedIndex + 1 ||
                attentionWorker_->attentionPvOutputWritesPending == 0) {
                if (attentionWorker_) finishAttentionWorker(false);
                return;
            }
            attentionWorker_->attentionPvOutputWritesPending -= 1;
            attentionWorker_->index = issuedIndex;
            attentionWorker_->attentionPvOutputWriteRetry = true;
            return;
        }
        statAttentionPvOutputPipelineRows_->addData(1);
        if (attentionWorker_ && attentionWorker_->dispatch.jobId == jobId &&
            attentionWorker_->phase == AttentionWorkerPhase::PvReadOutputs &&
            attentionWorker_->phaseSliceIndex == phaseSliceIndex) {
            attentionWorker_->attentionPvOutputWriteBytes.clear();
            attentionWorker_->attentionPvOutputWriteToCBuffer = false;
            attentionWorker_->attentionPvOutputWriteAccumulatorRow = UINT32_MAX;
            readAttentionPvOutput();
        }
    }

    void readAttentionPvOutput() {
        if (!attentionWorker_) return;
        if (attentionClusterEnable_) {
            readAttentionClusterPvOutput();
            return;
        }
        if (attentionSequential64Enable_) {
            readAttentionSequentialPvOutputs();
            return;
        }
        AttentionWorkerState& state = *attentionWorker_;
        if (state.index == attentionQueryRows(state)) {
            if (attentionPvOutputPipeline_ &&
                (state.attentionPvOutputWritesPending != 0 ||
                 state.attentionPvOutputWriteRetry)) return;
            completeAttentionPvOutputSlice();
            return;
        }
        if (attentionPvEarlyCompute_ &&
            attentionArrayPending_[state.index] != 0) return;
        const uint32_t arrayId = state.index;
        const uint64_t tag = attentionTransferTag();
        if (!readAttentionGemmOutputAsync(arrayId, sizeof(float), tag,
                [this, tag, arrayId](bool ok, uint64_t callbackTag,
                                     const std::vector<double>& values) {
                    if (!attentionWorker_ || !ok || callbackTag != tag || values.size() != 16) {
                        finishAttentionWorker(false); return;
                    }
                    const std::vector<uint8_t> bytes = attentionDoublesToBytes(values);
                    const bool storeInCBuffer = attentionOAccumulatorCBuffer_ &&
                        attentionWorker_->kvTileIndex + 1 <
                            attentionKvTileCountForQueryTile(*attentionWorker_);
                    const uint64_t addr = attentionWorker_->oLocal +
                        (static_cast<uint64_t>(arrayId) *
                             attentionWorker_->dispatch.headDim +
                         attentionWorker_->phaseSliceIndex * 16) * sizeof(float);
                    if (attentionPvOutputPipeline_) {
                        AttentionWorkerState& callbackState = *attentionWorker_;
                        callbackState.attentionPvOutputWriteAddr = addr;
                        callbackState.attentionPvOutputWriteAccumulatorRow =
                            attentionOAccumulatorRow(callbackState, arrayId);
                        callbackState.attentionPvOutputWriteToCBuffer =
                            storeInCBuffer;
                        callbackState.attentionPvOutputWriteBytes = bytes;
                        callbackState.attentionPvOutputWriteRetry = true;
                        issueAttentionPvOutputWrite();
                        return;
                    }
                    if (storeInCBuffer) {
                        const uint32_t accumulatorRow =
                            attentionOAccumulatorRow(*attentionWorker_, arrayId);
                        const uint64_t writeTag = attentionTransferTag();
                        if (!workerCommandProcessor->writeAttentionAccumulatorRowAsync(
                                accumulatorRow, bytes,
                                attentionWorker_->generation, writeTag,
                                LastTickCycle,
                                [this, writeTag](
                                    bool writeOk, uint64_t callbackWriteTag) {
                                    if (!attentionWorker_ || !writeOk ||
                                        callbackWriteTag != writeTag) {
                                        finishAttentionWorker(false);
                                        return;
                                    }
                                    statAttentionOAccumulatorStores_->addData(1);
                                    statAttentionOAccumulatorBytes_->addData(
                                        16 * sizeof(float));
                                    attentionWorker_->index += 1;
                                    readAttentionPvOutput();
                                })) {
                            finishAttentionWorker(false);
                        }
                        return;
                    }
                    attentionLocalWrite(addr, bytes, [this](bool writeOk) {
                        if (!attentionWorker_ || !writeOk) {
                            finishAttentionWorker(false); return;
                        }
                        attentionWorker_->index += 1;
                        readAttentionPvOutput();
                    });
                })) finishAttentionWorker(false);
    }

    bool handleAttentionArrayDone(uint32_t arrayId) {
        if (!attentionWorker_ || arrayId >= attentionArrayPending_.size() ||
            attentionArrayPending_[arrayId] == 0) return false;
        attentionArrayPending_[arrayId] = 0;
        arrayStates[arrayId] = 0;
        AttentionWorkerState& state = *attentionWorker_;
        if (attentionClusterEnable_) {
            BusyActivityTracker& activity =
                arrayId < attentionClusterConfig_.qkArrays
                ? attentionClusterQkArrayActivity_
                : attentionClusterPvArrayActivity_;
            if (!activity.leave(getCurrentSimCycle())) {
                statAttentionClusterIllegalTransitions_->addData(1);
                finishAttentionWorker(false);
                return true;
            }
        }
        if (state.arraysPending == 0) { finishAttentionWorker(false); return true; }
        state.arraysPending -= 1;
        if (attentionClusterEnable_ && state.arraysPending == 0 &&
            state.clusterOwnerLaunchNext < state.clusterOwnerLaunchEnd) {
            return true;
        }
        if (attentionPvEarlyCompute_ &&
            (state.phase == AttentionWorkerPhase::PvProgramInputs ||
             state.phase == AttentionWorkerPhase::PvRestoreOutput ||
             state.phase == AttentionWorkerPhase::PvCompute ||
             state.phase == AttentionWorkerPhase::PvReadOutputs)) {
            if (state.attentionPvPreparationComplete && state.index == arrayId) {
                if (state.phase == AttentionWorkerPhase::PvCompute) {
                    transitionAttentionTilePipeline(
                        AttentionTilePipelinePhase::PvOutputReadwrite);
                    state.phase = AttentionWorkerPhase::PvReadOutputs;
                    state.attentionPvOutputWritesPending = 0;
                    state.attentionPvOutputWriteRetry = false;
                    state.attentionPvOutputWriteBytes.clear();
                }
                readAttentionPvOutput();
            }
            return true;
        }
        if (state.arraysPending == 0) {
            if (attentionClusterEnable_ && attentionClusterPvRowWavefront_ &&
                state.phase == AttentionWorkerPhase::PvCompute) {
                completeAttentionClusterPvWavefrontCompute();
                return true;
            }
            if (state.phase == AttentionWorkerPhase::QkCompute &&
                (!attentionQkEarlyCompute_ ||
                 state.attentionQkInputProgrammingComplete)) {
                if (attentionSequential64Enable_) {
                    completeAttentionSequentialQkWave();
                    return true;
                }
                state.index = attentionClusterEnable_
                    ? attentionClusterWaveRowBegin(
                        state.clusterQkWave, attentionClusterQkLanes(state))
                    : 0;
                state.phase = AttentionWorkerPhase::QkReadOutputs;
                            readAttentionQkOutput();
            } else if (state.phase == AttentionWorkerPhase::PvCompute) {
                transitionAttentionTilePipeline(
                    AttentionTilePipelinePhase::PvOutputReadwrite);
                state.phase = AttentionWorkerPhase::PvReadOutputs;
                if (attentionClusterEnable_) {
                    state.phaseSliceIndex = 0;
                }
                state.attentionPvOutputWritesPending = 0;
                state.attentionPvOutputWriteRetry = false;
                state.attentionPvOutputWriteBytes.clear();
                readAttentionPvOutput();
            } else {
                finishAttentionWorker(false);
            }
        }
        return true;
    }

    bool isStreamingAttentionWorkerShape(
        const ControlTransportMessage& message) const {
        return (message.expectedWorkers == 4 ||
                (attentionWorkerClusterBridge_ &&
                 message.expectedWorkers ==
                    attentionWorkerClusterQkWorkersPerManager_)) &&
            message.expectedRows != 0 &&
            message.expectedRows % 16 == 0 && message.expectedCols != 0 &&
            message.expectedCols % 128 == 0 && message.rowsPerBand != 0 &&
            message.rowsPerBand == message.expectedCols / 4 &&
            (message.headDim == 64 || message.headDim == 128) &&
            message.nodeStrideBytes != 0;
    }

    uint64_t streamingAttentionWindowBytes(
        uint32_t queryTileRows, uint32_t kvTileRows,
        uint32_t headDim) const {
        const uint64_t qTileBytes = static_cast<uint64_t>(queryTileRows) *
            headDim * sizeof(float);
        const uint64_t kvTileBytes = static_cast<uint64_t>(kvTileRows) *
            headDim * sizeof(float);
        const uint64_t scoreBytes = static_cast<uint64_t>(queryTileRows) *
            kvTileRows * sizeof(float);
        const uint32_t kvBuffers = attentionKvDoubleBuffer_
            ? attentionKvBufferCount_ : 1u;
        const uint32_t queryGroupSize = attentionKvPairReuse_
            ? attentionKvQueryGroupSize_ : 1u;
        const uint32_t queryStorageCopies = attentionClusterEnable_
            ? queryGroupSize + 1u : 2u * queryGroupSize;
        return sizeof(GolemAttentionDescV2) + queryStorageCopies * qTileBytes + scoreBytes +
            2 * kvBuffers * kvTileBytes;
    }

    void requestAttentionWorkerClusterPvAllocation(
        AttentionWorkerState& state, uint32_t rowBlock) {
        if (!attentionWorkerClusterDynamicPv_ ||
            rowBlock >= state.reuseWindowPvAllocationPending.size() ||
            state.reuseWindowPvAllocationPending[rowBlock] != 0 ||
            state.reuseWindowPvRowCore[rowBlock] >= 0) {
            return;
        }
        ControlTransportMessage request = state.dispatch;
        request.kind =
            ControlTransportMessageKind::AttentionClusterPvAllocationRequest;
        request.ownerCore = static_cast<uint32_t>(coreID);
        request.workerCore = 0;
        request.row = rowBlock;
        request.groupQueryRowBegin = rowBlock * 64;
        request.sendCycle = getCurrentSimCycle();
        if (globalMem->sendControlMessage(0, request)) {
            state.reuseWindowPvAllocationPending[rowBlock] = 1;
        }
    }

    void sendAttentionWorkerClusterVHint(
        AttentionWorkerState& state, uint32_t rowBlock, uint32_t window,
        uint32_t workerCore) {
        constexpr uint32_t windowsPerRowBlock = 4;
        const size_t aggregateIndex =
            static_cast<size_t>(rowBlock) * windowsPerRowBlock + window;
        if (!attentionWorkerClusterVBroadcast_ ||
            aggregateIndex >= state.reuseWindowPvAggregates.size()) {
            return;
        }
        AttentionReuseWindowPvAggregate& aggregate =
            state.reuseWindowPvAggregates[aggregateIndex];
        if (!aggregate.vPrefetchReadyToHint || aggregate.vPrefetchHinted) return;
        ControlTransportMessage hint = state.dispatch;
        hint.kind =
            ControlTransportMessageKind::AttentionClusterPvVPrefetchHint;
        hint.ownerCore = static_cast<uint32_t>(coreID);
        hint.workerCore = workerCore;
        hint.groupQueryRowBegin = rowBlock * 64;
        hint.expectedRows = 64;
        hint.expectedCols = 128;
        hint.kvTileRows = window;
        hint.sendCycle = getCurrentSimCycle();
        if (globalMem->sendControlMessage(workerCore, hint)) {
            aggregate.vPrefetchHinted = true;
        }
    }

    void dispatchAttentionReuseWindowPvAggregates(AttentionWorkerState& state) {
        if (!attentionWorkerClusterBridge_) return;
        constexpr uint32_t windowsPerRowBlock = 4;
        constexpr uint64_t pWindowBytes =
            (64u * 256u + 64u) * sizeof(float);
        constexpr uint64_t injectionBytesPerCycle = 256;
        const uint32_t pvLaneCount =
            4u - attentionWorkerClusterQkWorkersPerManager_;
        const uint32_t rowBlocks = state.dispatch.expectedRows / 64;
        if (LastTickCycle < state.reuseWindowPvNextDispatchCycle) return;
        if (attentionWorkerClusterDynamicPv_) {
            for (uint32_t attempt = 0; attempt < rowBlocks; ++attempt) {
                const uint32_t rowBlock =
                    (state.reuseWindowPvDispatchLane + attempt) % rowBlocks;
                if (rowBlock >= state.reuseWindowPvRowNextWindow.size()) return;
                const uint32_t window =
                    state.reuseWindowPvRowNextWindow[rowBlock];
                if (window >= windowsPerRowBlock) continue;
                const size_t aggregateIndex =
                    static_cast<size_t>(rowBlock) * windowsPerRowBlock + window;
                if (aggregateIndex >= state.reuseWindowPvAggregates.size()) return;
                AttentionReuseWindowPvAggregate& aggregate =
                    state.reuseWindowPvAggregates[aggregateIndex];
                if (aggregate.rowSliceMask != 0xffff || aggregate.dispatched) {
                    continue;
                }
                if (state.reuseWindowPvRowCore[rowBlock] < 0) {
                    requestAttentionWorkerClusterPvAllocation(state, rowBlock);
                    continue;
                }

                ControlTransportMessage pv = state.dispatch;
                pv.kind = ControlTransportMessageKind::AttentionClusterPvDispatch;
                pv.ownerCore = static_cast<uint32_t>(coreID);
                pv.workerCore = static_cast<uint32_t>(
                    state.reuseWindowPvRowCore[rowBlock]);
                pv.groupQueryRowBegin = rowBlock * 64;
                pv.expectedRows = 64;
                pv.expectedCols = 128;
                pv.kvTileRows = window;
                pv.payload = aggregate.p;
                pv.scales = aggregate.scales;
                pv.sendCycle = getCurrentSimCycle();
                if (!globalMem->sendControlMessage(pv.workerCore, pv)) return;
                aggregate.dispatched = true;
                ++state.reuseWindowPvRowNextWindow[rowBlock];
                state.reuseWindowPvDispatchLane =
                    (rowBlock + 1) % rowBlocks;
                state.reuseWindowPvNextDispatchCycle = LastTickCycle +
                    (pWindowBytes + injectionBytesPerCycle - 1) /
                        injectionBytesPerCycle;
                return;
            }
            return;
        }
        for (uint32_t attempt = 0; attempt < pvLaneCount; ++attempt) {
            const uint32_t lane =
                (state.reuseWindowPvDispatchLane + attempt) % pvLaneCount;
            while (state.reuseWindowPvNextRowBlock[lane] < rowBlocks) {
                const uint32_t rowBlock = state.reuseWindowPvNextRowBlock[lane];
                const uint32_t window = state.reuseWindowPvNextWindow[lane];
                const size_t aggregateIndex =
                    static_cast<size_t>(rowBlock) * windowsPerRowBlock + window;
                if (aggregateIndex >= state.reuseWindowPvAggregates.size()) return;
                AttentionReuseWindowPvAggregate& aggregate =
                    state.reuseWindowPvAggregates[aggregateIndex];
                if (aggregate.rowSliceMask != 0xffff || aggregate.dispatched) break;

                ControlTransportMessage pv = state.dispatch;
                pv.kind = ControlTransportMessageKind::AttentionClusterPvDispatch;
                pv.ownerCore = static_cast<uint32_t>(coreID);
                const uint32_t manager = state.dispatch.ownerCore;
                pv.workerCore = 4 +
                    (attentionWorkerClusterQkWorkersPerManager_ + lane) * 4 +
                    manager;
                pv.groupQueryRowBegin = rowBlock * 64;
                pv.expectedRows = 64;
                pv.expectedCols = 128;
                pv.kvTileRows = window;
                pv.payload = aggregate.p;
                pv.scales = aggregate.scales;
                pv.sendCycle = getCurrentSimCycle();
                if (!globalMem->sendControlMessage(pv.workerCore, pv)) break;
                aggregate.dispatched = true;
                state.reuseWindowPvDispatchLane =
                    (lane + 1) % pvLaneCount;
                state.reuseWindowPvNextDispatchCycle = LastTickCycle +
                    (pWindowBytes + injectionBytesPerCycle - 1) /
                        injectionBytesPerCycle;
                if (++state.reuseWindowPvNextWindow[lane] == windowsPerRowBlock) {
                    state.reuseWindowPvNextWindow[lane] = 0;
                    state.reuseWindowPvNextRowBlock[lane] += pvLaneCount;
                }
                return;
            }
        }
    }

    void dispatchAttentionReuseWindowPvAggregates() {
        if (attentionWorker_) {
            dispatchAttentionReuseWindowPvAggregates(*attentionWorker_);
        }
    }

    void completeAttentionReuseWindowRowTile(
        uint32_t contextIndex, const std::vector<float>& values) {
        if (!attentionWorker_ ||
            contextIndex >= attentionWorker_->reuseWindowRowContexts.size()) {
            finishAttentionWorker(false);
            return;
        }
        AttentionWorkerState& state = *attentionWorker_;
        AttentionReuseWindowRowContext& context =
            state.reuseWindowRowContexts[contextIndex];
        constexpr uint32_t sliceRows = 4;
        constexpr uint32_t cols = 64;
        constexpr uint32_t tilesPerWindow = 4;
        constexpr uint32_t tilesPerRowBlock = 16;
        if (values.size() != sliceRows * cols ||
            context.nextKvTile >= tilesPerRowBlock ||
            context.outputScale.size() != sliceRows) {
            finishAttentionWorker(false);
            return;
        }

        const uint32_t kvTile = context.nextKvTile;
        const uint32_t tileInWindow = kvTile % tilesPerWindow;
        const uint32_t window = kvTile / tilesPerWindow;
        const size_t aggregateIndex =
            static_cast<size_t>(context.rowBlock) * 4 + window;
        if (aggregateIndex >= state.reuseWindowPvAggregates.size()) {
            finishAttentionWorker(false);
            return;
        }
        AttentionReuseWindowPvAggregate& aggregate =
            state.reuseWindowPvAggregates[aggregateIndex];
        const uint32_t pvLaneCount =
            4u - attentionWorkerClusterQkWorkersPerManager_;
        if (tileInWindow == 0) {
            aggregate.vPrefetchReadyToHint = true;
            if (attentionWorkerClusterDynamicPv_) {
                if (context.rowBlock < state.reuseWindowPvRowCore.size() &&
                    state.reuseWindowPvRowCore[context.rowBlock] >= 0) {
                    sendAttentionWorkerClusterVHint(
                        state, context.rowBlock, window,
                        static_cast<uint32_t>(
                            state.reuseWindowPvRowCore[context.rowBlock]));
                }
            } else {
                const uint32_t lane = context.rowBlock % pvLaneCount;
                const uint32_t workerCore = 4 +
                    (attentionWorkerClusterQkWorkersPerManager_ + lane) * 4 +
                    state.dispatch.ownerCore;
                sendAttentionWorkerClusterVHint(
                    state, context.rowBlock, window, workerCore);
            }
        }
        if (tileInWindow == 0) {
            std::fill(context.p.begin(), context.p.end(), 0.0f);
            std::fill(context.groupScale.begin(), context.groupScale.end(), 1.0f);
        } else {
            for (uint32_t prior = 0; prior < tileInWindow; ++prior) {
                for (uint32_t row = 0; row < sliceRows; ++row) {
                    const float scale = context.outputScale[row];
                    const size_t base =
                        (static_cast<size_t>(prior) * sliceRows + row) * cols;
                    for (uint32_t col = 0; col < cols; ++col) {
                        context.p[base + col] *= scale;
                    }
                }
            }
        }
        for (uint32_t row = 0; row < sliceRows; ++row) {
            context.groupScale[row] *= context.outputScale[row];
        }
        std::copy(values.begin(), values.end(),
                  context.p.begin() +
                      static_cast<size_t>(tileInWindow) * sliceRows * cols);

        if (tileInWindow == tilesPerWindow - 1) {
            const uint32_t slice = context.rowOffset / sliceRows;
            for (uint32_t panel = 0; panel < tilesPerWindow; ++panel) {
                for (uint32_t row = 0; row < sliceRows; ++row) {
                    const size_t src =
                        (static_cast<size_t>(panel) * sliceRows + row) * cols;
                    const size_t dst =
                        (static_cast<size_t>(panel) * 64 + context.rowOffset + row) * cols;
                    std::copy_n(context.p.begin() + src, cols,
                                aggregate.p.begin() + dst);
                }
            }
            std::copy(context.groupScale.begin(), context.groupScale.end(),
                      aggregate.scales.begin() + context.rowOffset);
            aggregate.rowSliceMask |= static_cast<uint16_t>(1u << slice);
            if (aggregate.rowSliceMask == 0xffff) {
                state.reuseWindowAttentionTile += tilesPerWindow;
            }
        }

        ++context.nextKvTile;
        context.requestInFlight = false;
        const uint32_t slice = context.rowOffset / sliceRows;
        const bool rowComplete = context.nextKvTile == tilesPerRowBlock;
        const uint32_t nextReadyIndex =
            context.rowBlock * tilesPerRowBlock + context.nextKvTile;
        const bool nextScoreReady = !rowComplete &&
            nextReadyIndex < state.reuseWindowScoreReady.size() &&
            state.reuseWindowScoreReady[nextReadyIndex] != 0;
        if (rowComplete) {
            if (slice >= state.reuseWindowSliceOwners.size() ||
                state.reuseWindowSliceOwners[slice] !=
                    static_cast<int32_t>(contextIndex)) {
                finishAttentionWorker(false);
                return;
            }
            state.reuseWindowSliceOwners[slice] = -1;
            state.reuseWindowSliceNextRowBlock[slice] =
                (context.rowBlock + 1) %
                (state.dispatch.expectedRows / 64);
            state.reuseWindowSliceWaitStart[slice] = UINT64_MAX;
        } else if (nextScoreReady) {
            state.reuseWindowSliceWaitStart[slice] = UINT64_MAX;
        } else if (state.reuseWindowSliceWaitStart[slice] == UINT64_MAX) {
            state.reuseWindowSliceWaitStart[slice] = LastTickCycle;
        }
        if (rowComplete) {
            if (context.scoreSlot >= 0 || context.pSlot >= 0) {
                finishAttentionWorker(false);
                return;
            }
            ++state.reuseWindowRowContextsCompleted;
        }
        dispatchAttentionReuseWindowPvAggregates();
    }

    void completeAttentionWorkerClusterDraining(size_t index) {
        if (index >= attentionWorkerClusterPvDraining_.size()) return;
        std::unique_ptr<AttentionWorkerState> completed =
            std::move(attentionWorkerClusterPvDraining_[index]);
        attentionWorkerClusterPvDraining_.erase(
            attentionWorkerClusterPvDraining_.begin() + index);
        const uint64_t endCycle = LastTickCycle;
        output->output("[ATTENTION_WORKER_CLUSTER_E2E] core=%" PRIu64
            " start=%" PRIu64 " end=%" PRIu64 " cycles=%" PRIu64 "\n",
            coreID, completed->reuseWindowQkStartCycle, endCycle,
            endCycle - completed->reuseWindowQkStartCycle + 1);
        output->output("[ATTENTION_FIFO_DECOUPLED] core=%" PRIu64
            " score_slot_stall_cycles=%" PRIu64
            " p_slot_stall_cycles=%" PRIu64
            " row_context_switches=%" PRIu64 "\n",
            coreID, completed->reuseWindowScoreSlotStallCycles,
            completed->reuseWindowPSlotStallCycles,
            completed->reuseWindowRowContextSwitches);
        traceAttentionMilestone(
            "worker", "worker_complete", "done",
            completed->dispatch.jobId, completed->dispatch.tag);
        ControlTransportMessage completion = completed->dispatch;
        completion.kind = ControlTransportMessageKind::AttentionComplete;
        completion.sendCycle = getCurrentSimCycle();
        completion.value = 1.0;
        if (!globalMem->sendControlMessage(completion.ownerCore, completion)) {
            output->verbose(CALL_INFO, 1, 0,
                            "Attention draining completion send failed\n");
        }
    }

    bool promoteAttentionWorkerClusterQkAheadForSfu() {
        if (!attentionWorker_ || !attentionWorkerClusterQkAhead_ ||
            attentionWorker_->reuseWindowRowContextsCompleted !=
                attentionWorker_->reuseWindowRowContexts.size()) {
            return false;
        }
        attentionWorkerClusterPvDraining_.push_back(std::move(attentionWorker_));
        attentionWorker_ = std::move(attentionWorkerClusterQkAhead_);
        output->output("[ATTENTION_WORKER_CLUSTER_JOB_OVERLAP] core=%" PRIu64
            " draining_job=%" PRIu64 " active_job=%" PRIu64 " cycle=%" PRIu64
            "\n", coreID,
            attentionWorkerClusterPvDraining_.back()->dispatch.jobId,
            attentionWorker_->dispatch.jobId, getCurrentSimCycle());
        startAttentionWorkerClusterQkAhead();
        startAttentionReuseWindowSoftmax();
        return true;
    }

    void pumpAttentionReuseWindowSoftmax() {
        if (!attentionWorker_ || !attentionWorkerClusterBridge_ || sfu == nullptr)
            return;
        AttentionWorkerState& state = *attentionWorker_;
        constexpr uint32_t sliceRows = 4;
        constexpr uint32_t cols = 64;
        constexpr uint32_t fifoSlots = 16;
        constexpr uint32_t tilesPerRowBlock = 16;
        const uint32_t rowBlocks = state.dispatch.expectedRows / 64;
        const uint32_t contextCount =
            static_cast<uint32_t>(state.reuseWindowRowContexts.size());
        const uint64_t generation = state.generation;

        // A physical slice follows a logical row only while its next Score tile
        // is ready. Online state lives in the logical context, so another ready
        // row can fill QK producer gaps without losing softmax state.
        for (uint32_t slice = 0; slice < fifoSlots; ++slice) {
            const int32_t owner = state.reuseWindowSliceOwners[slice];
            if (owner < 0 || static_cast<uint32_t>(owner) >= contextCount) {
                continue;
            }
            AttentionReuseWindowRowContext& context =
                state.reuseWindowRowContexts[static_cast<uint32_t>(owner)];
            if (context.requestInFlight || context.fifoPhase != 0 ||
                context.nextKvTile >= tilesPerRowBlock) {
                continue;
            }
            const uint32_t readyIndex =
                context.rowBlock * tilesPerRowBlock + context.nextKvTile;
            const bool ready = readyIndex < state.reuseWindowScoreReady.size() &&
                state.reuseWindowScoreReady[readyIndex] != 0;
            if (ready) {
                state.reuseWindowSliceWaitStart[slice] = UINT64_MAX;
                continue;
            }
            state.reuseWindowSliceOwners[slice] = -1;
            state.reuseWindowSliceNextRowBlock[slice] =
                (context.rowBlock + 1) % rowBlocks;
            state.reuseWindowSliceWaitStart[slice] = UINT64_MAX;
            ++state.reuseWindowRowContextSwitches;
        }
        for (uint32_t slice = 0; slice < fifoSlots; ++slice) {
            if (state.reuseWindowSliceOwners[slice] >= 0) continue;
            int32_t selectedContext = -1;
            for (uint32_t attempt = 0; attempt < rowBlocks; ++attempt) {
                const uint32_t rowBlock =
                    (state.reuseWindowSliceNextRowBlock[slice] + attempt) %
                        rowBlocks;
                const uint32_t contextIndex = rowBlock * fifoSlots + slice;
                AttentionReuseWindowRowContext& candidate =
                    state.reuseWindowRowContexts[contextIndex];
                const uint32_t readyIndex =
                    candidate.rowBlock * tilesPerRowBlock +
                    candidate.nextKvTile;
                const bool ready = candidate.nextKvTile < tilesPerRowBlock &&
                    readyIndex < state.reuseWindowScoreReady.size() &&
                    state.reuseWindowScoreReady[readyIndex] != 0;
                if (!ready) continue;
                if (selectedContext < 0 ||
                    (attentionWorkerClusterRowPriority_ &&
                     candidate.nextKvTile > state.reuseWindowRowContexts[
                         static_cast<uint32_t>(selectedContext)].nextKvTile)) {
                    selectedContext = static_cast<int32_t>(contextIndex);
                    if (!attentionWorkerClusterRowPriority_) break;
                }
            }
            if (selectedContext >= 0) {
                state.reuseWindowSliceOwners[slice] = selectedContext;
                const uint32_t rowBlock = state.reuseWindowRowContexts[
                    static_cast<uint32_t>(selectedContext)].rowBlock;
                state.reuseWindowSliceNextRowBlock[slice] =
                    (rowBlock + 1) % rowBlocks;
            }
        }
        const auto contextIsActive = [&state](uint32_t contextIndex) {
            if (contextIndex >= state.reuseWindowRowContexts.size()) {
                return false;
            }
            const AttentionReuseWindowRowContext& context =
                state.reuseWindowRowContexts[contextIndex];
            const uint32_t slice = context.rowOffset / sliceRows;
            return slice < state.reuseWindowSliceOwners.size() &&
                state.reuseWindowSliceOwners[slice] ==
                    static_cast<int32_t>(contextIndex);
        };

        dispatchAttentionReuseWindowPvAggregates();
        if (state.reuseWindowRowContextsCompleted ==
                state.reuseWindowRowContexts.size()) {
            if (state.reuseWindowClusterPvCompleted == rowBlocks) {
                const uint64_t endCycle = LastTickCycle;
                output->output("[ATTENTION_WORKER_CLUSTER_E2E] core=%" PRIu64
                    " start=%" PRIu64 " end=%" PRIu64 " cycles=%" PRIu64 "\n",
                    coreID, state.reuseWindowQkStartCycle, endCycle,
                    endCycle - state.reuseWindowQkStartCycle + 1);
                output->output("[ATTENTION_FIFO_DECOUPLED] core=%" PRIu64
                    " score_slot_stall_cycles=%" PRIu64
                    " p_slot_stall_cycles=%" PRIu64
                    " row_context_switches=%" PRIu64 "\n",
                    coreID, state.reuseWindowScoreSlotStallCycles,
                    state.reuseWindowPSlotStallCycles,
                    state.reuseWindowRowContextSwitches);
                finishAttentionWorker(true);
            } else {
                promoteAttentionWorkerClusterQkAheadForSfu();
            }
            return;
        }

        // P drains are independent of Score production. Releasing a P slot may
        // allow another Score-ready context to enter the SFU immediately.
        for (uint32_t contextIndex = 0; contextIndex < contextCount;
                ++contextIndex) {
            AttentionReuseWindowRowContext& context =
                state.reuseWindowRowContexts[contextIndex];
            if (!contextIsActive(contextIndex) || context.fifoPhase != 4 ||
                context.requestInFlight ||
                context.pSlot < 0) continue;
            const uint32_t pSlot = static_cast<uint32_t>(context.pSlot);
            context.requestInFlight = true;
            const uint64_t transferTag = attentionTransferTag();
            if (!sfu->readAttentionPRowAsync(
                    pSlot, context.fifoTag, 0, sliceRows * cols, transferTag,
                    [this, generation, contextIndex, pSlot, transferTag](
                        bool ok, uint64_t callbackTag,
                        const std::vector<float>& values) {
                        if (!attentionCallbackGenerationMatches(generation)) return;
                        if (!ok || callbackTag != transferTag ||
                            contextIndex >= attentionWorker_->reuseWindowRowContexts.size()) {
                            finishAttentionWorker(false);
                            return;
                        }
                        AttentionWorkerState& completedState = *attentionWorker_;
                        AttentionReuseWindowRowContext& completed =
                            completedState.reuseWindowRowContexts[contextIndex];
                        completed.requestInFlight = false;
                        if (completed.pSlot != static_cast<int32_t>(pSlot) ||
                            completedState.reuseWindowPOwners[pSlot] !=
                                static_cast<int32_t>(contextIndex) ||
                            !sfu->releaseAttentionPSlot(pSlot, completed.fifoTag)) {
                            finishAttentionWorker(false);
                            return;
                        }
                        completedState.reuseWindowPOwners[pSlot] = -1;
                        completed.pSlot = -1;
                        completed.fifoPhase = 0;
                        completeAttentionReuseWindowRowTile(contextIndex, values);
                        pumpAttentionReuseWindowSoftmax();
                    })) {
                context.requestInFlight = false;
            }
        }

        // A Score-ready context independently claims any free P slot. The two
        // FIFO indices deliberately need not match.
        bool pSlotBlocked = false;
        const uint32_t pScanBegin = state.reuseWindowNextPContext;
        std::vector<uint32_t> pOrder(contextCount);
        for (uint32_t attempt = 0; attempt < contextCount; ++attempt) {
            pOrder[attempt] = (pScanBegin + attempt) % contextCount;
        }
        if (attentionWorkerClusterRowPriority_) {
            std::stable_sort(pOrder.begin(), pOrder.end(),
                [&state](uint32_t lhs, uint32_t rhs) {
                    return state.reuseWindowRowContexts[lhs].nextKvTile >
                        state.reuseWindowRowContexts[rhs].nextKvTile;
                });
        }
        for (uint32_t contextIndex : pOrder) {
            AttentionReuseWindowRowContext& context =
                state.reuseWindowRowContexts[contextIndex];
            if (!contextIsActive(contextIndex) || context.fifoPhase != 2 ||
                context.requestInFlight) continue;
            const uint32_t kvTile = context.nextKvTile;
            const uint32_t rowBegin = context.rowBlock * 64 + context.rowOffset;
            if (context.pSlot < 0) {
                uint32_t pSlot = fifoSlots;
                for (uint32_t slot = 0; slot < fifoSlots; ++slot) {
                    if (state.reuseWindowPOwners[slot] < 0) {
                        pSlot = slot;
                        break;
                    }
                }
                if (pSlot == fifoSlots) {
                    pSlotBlocked = true;
                    continue;
                }
                const size_t elements = static_cast<size_t>(sliceRows) * cols;
                if (!sfu->reserveAttentionPSlot(
                        pSlot, context.fifoTag, elements)) {
                    pSlotBlocked = true;
                    continue;
                }
                context.pSlot = static_cast<int32_t>(pSlot);
                state.reuseWindowPOwners[pSlot] =
                    static_cast<int32_t>(contextIndex);
                state.reuseWindowNextPContext =
                    (contextIndex + 1) % contextCount;
            }
            if (context.scoreSlot < 0) {
                finishAttentionWorker(false);
                return;
            }
            const uint32_t scoreSlot = static_cast<uint32_t>(context.scoreSlot);
            const uint32_t pSlot = static_cast<uint32_t>(context.pSlot);
            AttentionTileRequest request{};
            request.tag = state.dispatch.tag + 1 +
                static_cast<uint64_t>(contextIndex) * tilesPerRowBlock + kvTile;
            request.jobId = state.dispatch.jobId;
            request.globalRowBegin = rowBegin;
            request.keyBegin = kvTile * cols;
            request.rows = sliceRows;
            request.cols = cols;
            request.headDim = state.dispatch.headDim;
            request.kvTileIndex = kvTile;
            request.numKvTiles = tilesPerRowBlock;
            request.causal = false;
            request.firstTileForJob = kvTile == 0;
            request.directScoreMode = true;
            request.generation = generation;
            request.scoreSlot = scoreSlot;
            request.scoreTag = context.fifoTag;
            request.directPMode = true;
            request.pSlot = pSlot;
            request.pTag = context.fifoTag;
            const AttentionClusterAdmission admission =
                sfu->attentionTileAdmission(request);
            if (admission == AttentionClusterAdmission::Invalid) {
                finishAttentionWorker(false);
                return;
            }
            if (admission == AttentionClusterAdmission::Retry) continue;
            context.requestInFlight = true;
            if (!sfu->issueAttentionTile(request,
                    [this, generation, contextIndex, scoreSlot](
                        bool softmaxOk, const AttentionTileResult& result) {
                        if (!attentionCallbackGenerationMatches(generation) ||
                            !softmaxOk || result.rows != sliceRows ||
                            contextIndex >=
                                attentionWorker_->reuseWindowRowContexts.size()) {
                            finishAttentionWorker(false);
                            return;
                        }
                        AttentionReuseWindowRowContext& completed =
                            attentionWorker_->reuseWindowRowContexts[contextIndex];
                        completed.requestInFlight = false;
                        if (completed.scoreSlot !=
                                static_cast<int32_t>(scoreSlot) ||
                            attentionWorker_->reuseWindowScoreOwners[scoreSlot] !=
                                static_cast<int32_t>(contextIndex) ||
                            !sfu->releaseAttentionScoreSlot(
                                scoreSlot, completed.fifoTag)) {
                            finishAttentionWorker(false);
                            return;
                        }
                        attentionWorker_->reuseWindowScoreOwners[scoreSlot] = -1;
                        completed.scoreSlot = -1;
                        completed.outputScale.assign(
                            result.oldOutputScale.begin(),
                            result.oldOutputScale.begin() + result.rows);
                        completed.fifoPhase = 4;
                        pumpAttentionReuseWindowSoftmax();
                    })) {
                context.requestInFlight = false;
            } else {
                context.fifoPhase = 3;
            }
        }
        if (pSlotBlocked &&
            state.reuseWindowLastPSlotStallCycle != LastTickCycle) {
            ++state.reuseWindowPSlotStallCycles;
            state.reuseWindowLastPSlotStallCycle = LastTickCycle;
        }

        // Score production has its own free list and can run while older P
        // slots are being drained.
        bool scoreSlotBlocked = false;
        const uint32_t scoreScanBegin = state.reuseWindowNextScoreContext;
        std::vector<uint32_t> scoreOrder(contextCount);
        for (uint32_t attempt = 0; attempt < contextCount; ++attempt) {
            scoreOrder[attempt] = (scoreScanBegin + attempt) % contextCount;
        }
        if (attentionWorkerClusterRowPriority_) {
            std::stable_sort(scoreOrder.begin(), scoreOrder.end(),
                [&state](uint32_t lhs, uint32_t rhs) {
                    return state.reuseWindowRowContexts[lhs].nextKvTile >
                        state.reuseWindowRowContexts[rhs].nextKvTile;
                });
        }
        for (uint32_t contextIndex : scoreOrder) {
            AttentionReuseWindowRowContext& context =
                state.reuseWindowRowContexts[contextIndex];
            if (!contextIsActive(contextIndex) || context.fifoPhase != 0 ||
                context.requestInFlight ||
                context.nextKvTile >= tilesPerRowBlock) continue;
            const uint32_t kvTile = context.nextKvTile;
            const uint32_t readyIndex =
                context.rowBlock * tilesPerRowBlock + kvTile;
            if (readyIndex >= state.reuseWindowScoreReady.size() ||
                state.reuseWindowScoreReady[readyIndex] == 0) continue;
            uint32_t scoreSlot = fifoSlots;
            for (uint32_t slot = 0; slot < fifoSlots; ++slot) {
                if (state.reuseWindowScoreOwners[slot] < 0) {
                    scoreSlot = slot;
                    break;
                }
            }
            if (scoreSlot == fifoSlots) {
                scoreSlotBlocked = true;
                continue;
            }

            const uint32_t rowBegin = context.rowBlock * 64 + context.rowOffset;
            std::vector<float> score(sliceRows * cols, 0.0f);
            for (uint32_t row = 0; row < sliceRows; ++row) {
                const size_t src = static_cast<size_t>(rowBegin + row) *
                    state.dispatch.expectedCols + kvTile * cols;
                std::copy_n(state.reuseWindowScores.begin() + src, cols,
                            score.begin() + static_cast<size_t>(row) * cols);
            }
            AttentionClusterTag fifoTag;
            fifoTag.generation = generation;
            fifoTag.jobId = state.dispatch.jobId;
            fifoTag.group = context.rowBlock;
            fifoTag.queryContext = context.rowOffset / sliceRows;
            fifoTag.queryTileIndex = context.rowBlock;
            fifoTag.kvTileIndex = kvTile;
            fifoTag.sequence = contextIndex;
            const size_t elements = static_cast<size_t>(sliceRows) * cols;
            if (!sfu->reserveAttentionScoreSlot(
                    scoreSlot, fifoTag, elements)) {
                scoreSlotBlocked = true;
                continue;
            }
            context.fifoTag = fifoTag;
            context.scoreSlot = static_cast<int32_t>(scoreSlot);
            state.reuseWindowScoreOwners[scoreSlot] =
                static_cast<int32_t>(contextIndex);
            state.reuseWindowNextScoreContext =
                (contextIndex + 1) % contextCount;
            context.fifoPhase = 1;
            context.requestInFlight = true;
            const uint64_t transferTag = attentionTransferTag();
            if (!sfu->writeAttentionScoreBeatAsync(
                    scoreSlot, fifoTag, 0, score, transferTag,
                    [this, generation, contextIndex, scoreSlot, transferTag](
                        bool ok, uint64_t callbackTag) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (!ok || callbackTag != transferTag ||
                        contextIndex >=
                            attentionWorker_->reuseWindowRowContexts.size()) {
                        finishAttentionWorker(false);
                        return;
                    }
                    AttentionReuseWindowRowContext& completed =
                        attentionWorker_->reuseWindowRowContexts[contextIndex];
                    if (completed.scoreSlot != static_cast<int32_t>(scoreSlot) ||
                        attentionWorker_->reuseWindowScoreOwners[scoreSlot] !=
                            static_cast<int32_t>(contextIndex) ||
                        completed.fifoPhase != 1) {
                        finishAttentionWorker(false);
                        return;
                    }
                    completed.requestInFlight = false;
                    completed.fifoPhase = 2;
                    pumpAttentionReuseWindowSoftmax();
                })) {
                context.requestInFlight = false;
                context.fifoPhase = 0;
                context.scoreSlot = -1;
                state.reuseWindowScoreOwners[scoreSlot] = -1;
                if (!sfu->releaseAttentionScoreSlot(scoreSlot, fifoTag)) {
                    finishAttentionWorker(false);
                    return;
                }
            }
        }
        if (scoreSlotBlocked &&
            state.reuseWindowLastScoreSlotStallCycle != LastTickCycle) {
            ++state.reuseWindowScoreSlotStallCycles;
            state.reuseWindowLastScoreSlotStallCycle = LastTickCycle;
        }
    }

    void startAttentionReuseWindowSoftmax() {
        if (!attentionWorker_ || sfu == nullptr) {
            finishAttentionWorker(false);
            return;
        }
        AttentionWorkerState& state = *attentionWorker_;
        if (attentionWorkerClusterBridge_) {
            pumpAttentionReuseWindowSoftmax();
            return;
        }
        constexpr uint32_t rows = 64;
        constexpr uint32_t cols = 64;
        constexpr uint32_t tilesPerHalf = 16;
        const uint32_t tileIndex = state.reuseWindowAttentionTile;
        const uint32_t rowBlocks = state.dispatch.expectedRows / rows;
        if (tileIndex >= rowBlocks * tilesPerHalf) {
            if (attentionWorkerClusterBridge_) {
                if (state.reuseWindowClusterPvCompleted == rowBlocks) {
                    const uint64_t endCycle = LastTickCycle;
                    output->output("[ATTENTION_WORKER_CLUSTER_E2E] core=%" PRIu64
                        " start=%" PRIu64 " end=%" PRIu64 " cycles=%" PRIu64 "\n",
                        coreID, state.reuseWindowQkStartCycle, endCycle,
                        endCycle - state.reuseWindowQkStartCycle + 1);
                    finishAttentionWorker(true);
                }
                return;
            }
            const uint64_t localBase = globalMem->getBaseAddr();
            state.oLocal = localBase + 0x10000;
            state.dispatch.queryTileRows = state.dispatch.expectedRows;
            state.queryTileIndex = 0;
            attentionLocalWrite(
                state.oLocal,
                attentionFloatsToBytes(state.reuseWindowOutput),
                [this](bool ok) {
                    if (!attentionWorker_ || !ok) {
                        finishAttentionWorker(false);
                        return;
                    }
                    dmaAttentionQueryTileOutput();
                });
            return;
        }
        if (attentionWorkerClusterBridge_) {
            if (state.reuseWindowSoftmaxBusy ||
                tileIndex >= state.reuseWindowScoreReady.size() ||
                state.reuseWindowScoreReady[tileIndex] == 0) {
                return;
            }
            state.reuseWindowSoftmaxBusy = true;
        }
        const uint32_t rowBegin = (tileIndex / tilesPerHalf) * rows;
        const uint32_t kvTile = tileIndex % tilesPerHalf;
        std::vector<float> score(rows * cols, 0.0f);
        for (uint32_t row = 0; row < rows; ++row) {
            const size_t src = static_cast<size_t>(rowBegin + row) *
                state.dispatch.expectedCols + kvTile * cols;
            std::copy_n(state.reuseWindowScores.begin() + src, cols,
                        score.begin() + static_cast<size_t>(row) * cols);
        }
        const uint64_t scoreAddr = globalMem->getBaseAddr() + 0x10000;
        const uint64_t generation = state.generation;
        attentionLocalWrite(
            scoreAddr, attentionFloatsToBytes(score),
            [this, generation, rowBegin, kvTile, scoreAddr](bool ok) {
                if (!attentionCallbackGenerationMatches(generation) || !ok) {
                    finishAttentionWorker(false);
                    return;
                }
                AttentionTileRequest request{};
                request.tag = attentionWorker_->dispatch.tag +
                    attentionWorker_->reuseWindowAttentionTile + 1;
                request.jobId = attentionWorker_->dispatch.jobId;
                request.localScoreAddr = scoreAddr;
                request.globalRowBegin = rowBegin;
                request.keyBegin = kvTile * 64;
                request.rows = 64;
                request.cols = 64;
                request.headDim = attentionWorker_->dispatch.headDim;
                request.kvTileIndex = kvTile;
                request.numKvTiles = 16;
                request.causal = false;
                request.firstTileForJob = kvTile == 0;
                request.generation = generation;
                if (!sfu->issueAttentionTile(
                        request,
                        [this, generation, scoreAddr, rowBegin](
                            bool softmaxOk, const AttentionTileResult& result) {
                            if (!attentionCallbackGenerationMatches(generation) ||
                                !softmaxOk || result.rows != 64) {
                                finishAttentionWorker(false);
                                return;
                            }
                            for (uint32_t row = 0; row < 64; ++row) {
                                const float scale = result.oldOutputScale[row];
                                const size_t base =
                                    static_cast<size_t>(rowBegin + row) * 128;
                                for (uint32_t dim = 0; dim < 128; ++dim) {
                                    attentionWorker_->reuseWindowOutput[base + dim] *= scale;
                                }
                            }
                            attentionWorker_->outputScales.assign(
                                result.oldOutputScale.begin(),
                                result.oldOutputScale.begin() + result.rows);
                            attentionLocalRead(
                                scoreAddr, 64u * 64u * sizeof(float),
                                [this, generation](
                                    bool readOk, const std::vector<uint8_t>& raw) {
                                    if (!attentionCallbackGenerationMatches(generation) ||
                                        !readOk || raw.size() != 64u * 64u * sizeof(float)) {
                                        finishAttentionWorker(false);
                                        return;
                                    }
                                    const uint32_t kvTile =
                                        attentionWorker_->reuseWindowAttentionTile % 16;
                                    const uint32_t tileInWindow = kvTile % 4;
                                    if (tileInWindow == 0) {
                                        std::fill(
                                            attentionWorker_->reuseWindowP.begin(),
                                            attentionWorker_->reuseWindowP.end(), 0.0f);
                                        attentionWorker_->reuseWindowGroupScale.assign(64, 1.0f);
                                    } else {
                                        for (uint32_t prior = 0;
                                             prior < tileInWindow; ++prior) {
                                            for (uint32_t row = 0; row < 64; ++row) {
                                                const float scale =
                                                    attentionWorker_->outputScales[row];
                                                const size_t base =
                                                    (static_cast<size_t>(prior) * 64 + row) * 64;
                                                for (uint32_t col = 0; col < 64; ++col) {
                                                    attentionWorker_->reuseWindowP[base + col] *= scale;
                                                }
                                            }
                                        }
                                    }
                                    for (uint32_t row = 0; row < 64; ++row) {
                                        attentionWorker_->reuseWindowGroupScale[row] *=
                                            attentionWorker_->outputScales[row];
                                    }
                                    std::memcpy(
                                        attentionWorker_->reuseWindowP.data() +
                                            static_cast<size_t>(tileInWindow) * 64 * 64,
                                        raw.data(), raw.size());
                                    if (tileInWindow != 3) {
                                        ++attentionWorker_->reuseWindowAttentionTile;
                                        attentionWorker_->reuseWindowSoftmaxBusy = false;
                                        startAttentionReuseWindowSoftmax();
                                        return;
                                    }
                                    if (attentionWorkerClusterBridge_) {
                                        ControlTransportMessage pv = attentionWorker_->dispatch;
                                        pv.kind = ControlTransportMessageKind::AttentionClusterPvDispatch;
                                        pv.ownerCore = static_cast<uint32_t>(coreID);
                                        const uint32_t manager = attentionWorker_->dispatch.ownerCore;
                                        const uint32_t rowBlock =
                                            attentionWorker_->reuseWindowAttentionTile / 16;
                                        const uint32_t pvLaneCount = 4u -
                                            attentionWorkerClusterQkWorkersPerManager_;
                                        const uint32_t pvLane = rowBlock % pvLaneCount;
                                        pv.workerCore = 4 +
                                            (attentionWorkerClusterQkWorkersPerManager_ +
                                             pvLane) * 4 + manager;
                                        pv.groupQueryRowBegin = rowBlock * 64;
                                        pv.expectedRows = 64;
                                        pv.expectedCols = 128;
                                        pv.kvTileRows = kvTile / 4;
                                        pv.payload = attentionWorker_->reuseWindowP;
                                        pv.scales = attentionWorker_->reuseWindowGroupScale;
                                        pv.sendCycle = getCurrentSimCycle();
                                        if (!globalMem->sendControlMessage(pv.workerCore, pv)) {
                                            finishAttentionWorker(false);
                                            return;
                                        }
                                        ++attentionWorker_->reuseWindowAttentionTile;
                                        attentionWorker_->reuseWindowSoftmaxBusy = false;
                                        startAttentionReuseWindowSoftmax();
                                        return;
                                    }
                                    constexpr size_t panelBytes = 64u * 64u * sizeof(float);
                                    std::vector<uint8_t> packed(6u * panelBytes, 0);
                                    std::memcpy(packed.data(),
                                        attentionWorker_->reuseWindowP.data(), 2u * panelBytes);
                                    std::memcpy(packed.data() + 4u * panelBytes,
                                        attentionWorker_->reuseWindowP.data() + 2u * 64u * 64u,
                                        2u * panelBytes);
                                    attentionLocalWrite(
                                        globalMem->getBaseAddr() + 0x10000,
                                        packed,
                                        [this, generation](bool writeOk) {
                                            if (!attentionCallbackGenerationMatches(generation) ||
                                                !writeOk) {
                                                finishAttentionWorker(false);
                                                return;
                                            }
                                            startAttentionReuseWindowPv();
                                        });
                                });
                        })) {
                    finishAttentionWorker(false);
                }
            });
    }

    void startAttentionReuseWindowPv() {
        if (!attentionWorker_ || workerCommandProcessor == nullptr) {
            finishAttentionWorker(false);
            return;
        }
        AttentionWorkerState& state = *attentionWorker_;
        constexpr uint32_t block = 64;
        constexpr uint64_t panelVOffset = 0x06000000;
        const uint32_t keyWindow =
            (state.reuseWindowAttentionTile % 16) / 4;
        const uint64_t panelBytes = block * block * sizeof(float);
        const uint64_t vHeadOffset =
            static_cast<uint64_t>(state.dispatch.kvHeadIndex) * 2 * 4 * panelBytes;
        WorkerTaskListHeader header{};
        header.worker_slot = 0;
        header.task_count = 1;
        header.active_worker_cores = 1;
        header.total_groups = 4;
        header.data_memory_node_count = 4;
        header.mem_node_size = state.dispatch.nodeStrideBytes;
        header.m = 64;
        header.n = 128;
        header.k = 256;
        header.hw_input_size = block;
        header.hw_output_size = block;
        header.block_m = block;
        header.block_n = block;
        header.block_k = block;
        header.elem_bytes = sizeof(float);
        header.mat_stride_bytes = panelBytes;
        header.vec_stride_bytes = block * sizeof(float);
        header.off_gemm_vec_base =
            static_cast<uint64_t>(1 + keyWindow) * state.dispatch.nodeStrideBytes +
            panelVOffset + vHeadOffset;
        const uint64_t localBase = globalMem->getBaseAddr();
        header.local_mat_ping_gm_addr = localBase + 0x10000;
        header.local_vec_ping_gm_addr = localBase + 0x70000;
        header.local_mat_slot_stride_bytes = panelBytes;
        header.local_vec_slot_stride_bytes = panelBytes;
        header.local_slot_count = 24;
        header.local_accum_gm_addr = localBase + 0xD0000;
        header.local_out_gm_addr = localBase + 0xD4000;
        header.a_reuse_n_tiles = 2;
        header.n_group_count = 1;
        header.b_reuse_m_tiles = 1;
        header.m_group_count = 1;
        header.descriptor_start_cycle = getCurrentSimCycle();
        header.operand_layout = 3;
        header.scheduler_worker_slot = state.dispatch.workerSlot;
        header.attention_pv_cache_gm_addr = localBase + 0x100000;
        header.attention_pv_cache_bytes = 0x80000;

        state.reuseWindowPvFusionTiles = 0;
        const uint64_t generation = state.generation;
        if (!workerCommandProcessor->setWindowCallbacks(
                [this, generation](uint64_t taskId, uint64_t,
                                   const std::vector<uint8_t>& tile) {
                    if (!attentionCallbackGenerationMatches(generation) ||
                        taskId >= 2 || tile.size() != 64u * 64u * sizeof(float)) {
                        return false;
                    }
                    const uint32_t rowBegin =
                        (attentionWorker_->reuseWindowAttentionTile / 16) * 64;
                    for (uint32_t col = 0; col < 64; ++col) {
                        for (uint32_t row = 0; row < 64; ++row) {
                            float value = 0.0f;
                            std::memcpy(&value,
                                tile.data() +
                                    (static_cast<size_t>(col) * 64 + row) * sizeof(float),
                                sizeof(float));
                            const size_t dst =
                                static_cast<size_t>(rowBegin + row) * 128 +
                                static_cast<uint32_t>(taskId) * 64 + col;
                            attentionWorker_->reuseWindowOutput[dst] += value;
                        }
                    }
                    ++attentionWorker_->reuseWindowPvFusionTiles;
                    return true;
                },
                [this, generation](uint64_t cycles, uint64_t, uint64_t) {
                    if (!attentionCallbackGenerationMatches(generation)) return;
                    if (attentionWorker_->reuseWindowPvFusionTiles != 2) {
                        finishAttentionWorker(false);
                        return;
                    }
                    attentionWorker_->reuseWindowPvCycles += cycles;
                    ++attentionWorker_->reuseWindowAttentionTile;
                    startAttentionReuseWindowSoftmax();
                }) || !workerCommandProcessor->startWindow(header)) {
            finishAttentionWorker(false);
        }
    }

    bool startAttentionReuseWindowQkBridge(AttentionWorkerState& state) {
        if (!attentionReuseWindowQkBridge_ || workerCommandProcessor == nullptr ||
            state.dispatch.nodeStrideBytes == 0 || coreID < 4 ||
            state.dispatch.queryLength == 0 || state.dispatch.headDim == 0) {
            return false;
        }
        const uint32_t groupSize =
            state.dispatch.numQueryHeads / state.dispatch.numKvHeads;
        const uint32_t globalWorkerSlot = attentionWorkerClusterBridge_
            ? 0u : state.dispatch.workerSlot;
        const uint32_t activeWorkers = attentionWorkerClusterBridge_ ? 1u : 4u;
        constexpr uint32_t blockM = 64;
        constexpr uint32_t blockN = 64;
        constexpr uint32_t blockK = 64;
        constexpr uint32_t reuseM = 2;
        constexpr uint32_t reuseN = 4;
        const uint32_t m = attentionWorkerClusterBridge_
            ? state.dispatch.expectedRows
            : state.dispatch.queryLength * groupSize;
        const uint32_t n = state.dispatch.expectedCols;
        const uint32_t mGroups = (m / blockM + reuseM - 1) / reuseM;
        const uint32_t nGroups = (n / blockN + reuseN - 1) / reuseN;
        const uint32_t macroTasks = mGroups * nGroups;
        if (m % (blockM * reuseM) != 0 || n % (blockN * reuseN) != 0 ||
            state.dispatch.headDim % blockK != 0 || globalWorkerSlot >= activeWorkers ||
            state.dispatch.rowsPerBand == 0 ||
            state.dispatch.rowsPerBand % (blockM * reuseM) != 0) {
            return false;
        }

        WorkerTaskListHeader header{};
        header.worker_slot = globalWorkerSlot;
        header.task_count = globalWorkerSlot < macroTasks
            ? (macroTasks - globalWorkerSlot + activeWorkers - 1) / activeWorkers : 0;
        header.active_worker_cores = activeWorkers;
        header.total_groups = 4;
        header.data_memory_node_count = 4;
        header.mem_node_size = state.dispatch.nodeStrideBytes;
        header.m = m;
        header.n = n;
        header.k = state.dispatch.headDim;
        header.hw_input_size = blockK;
        header.hw_output_size = blockM;
        header.block_m = blockM;
        header.block_n = blockN;
        header.block_k = blockK;
        header.elem_bytes = sizeof(float);
        header.mat_stride_bytes =
            static_cast<uint64_t>(blockM) * blockK * sizeof(float);
        header.vec_stride_bytes = static_cast<uint64_t>(blockK) * sizeof(float);
        constexpr uint64_t panelQOffset = 0x04000000;
        constexpr uint64_t panelKOffset = 0x05000000;
        header.off_gemm_mat_base = panelQOffset +
            static_cast<uint64_t>(state.dispatch.kvHeadIndex) * groupSize *
                (state.dispatch.rowsPerBand / blockM) *
                (state.dispatch.headDim / blockK) * blockM * blockK * sizeof(float) +
            (attentionWorkerClusterBridge_
                ? static_cast<uint64_t>(state.dispatch.groupQueryRowBegin) *
                    state.dispatch.headDim * sizeof(float)
                : 0u);
        header.off_gemm_vec_base = panelKOffset +
            static_cast<uint64_t>(state.dispatch.kvHeadIndex) *
                (state.dispatch.rowsPerBand / blockN) *
                (state.dispatch.headDim / blockK) * blockN * blockK * sizeof(float);
        const uint64_t localBase = globalMem->getBaseAddr();
        header.local_mat_ping_gm_addr = localBase + 0x10000;
        header.local_vec_ping_gm_addr = localBase + 0x40000;
        header.local_mat_slot_stride_bytes = header.mat_stride_bytes;
        header.local_vec_slot_stride_bytes =
            static_cast<uint64_t>(blockN) * blockK * sizeof(float);
        header.local_slot_count = 24;
        header.local_accum_gm_addr = localBase + 0xA0000;
        header.local_out_gm_addr = localBase + 0xA4000;
        header.a_reuse_n_tiles = reuseN;
        header.n_group_count = nGroups;
        header.b_reuse_m_tiles = reuseM;
        header.m_group_count = mGroups;
        header.descriptor_start_cycle = getCurrentSimCycle();
        header.operand_layout = 2;
        header.attention_query_length = state.dispatch.queryLength;
        header.attention_rows_per_band = state.dispatch.rowsPerBand;
        header.scheduler_worker_slot = state.dispatch.workerSlot;
        header.attention_node_stride_bytes = state.dispatch.nodeStrideBytes;

        state.reuseWindowExpectedFusionTiles =
            static_cast<uint64_t>(header.task_count) * reuseM * reuseN;
        state.reuseWindowScores.assign(
            static_cast<size_t>(state.dispatch.expectedRows) *
                state.dispatch.expectedCols, 0.0f);
        state.reuseWindowOutput.assign(
            static_cast<size_t>(state.dispatch.expectedRows) *
                state.dispatch.headDim, 0.0f);
        state.reuseWindowP.assign(64u * 256u, 0.0f);
        state.reuseWindowAttentionTile = 0;
        state.reuseWindowScoreReady.assign(
            static_cast<size_t>(state.dispatch.expectedRows / blockM) *
                (state.dispatch.expectedCols / blockN), 0);
        state.reuseWindowSoftmaxBusy = false;
        state.reuseWindowQkComplete = false;
        state.reuseWindowClusterPvCompleted = 0;
        state.reuseWindowRowContexts.clear();
        state.reuseWindowPvAggregates.clear();
        state.reuseWindowReadQueue.clear();
        state.reuseWindowIoBusy = false;
        state.reuseWindowRowContextsCompleted = 0;
        state.reuseWindowScoreOwners.fill(-1);
        state.reuseWindowPOwners.fill(-1);
        state.reuseWindowSliceOwners.fill(-1);
        state.reuseWindowSliceNextRowBlock.fill(0);
        state.reuseWindowSliceWaitStart.fill(UINT64_MAX);
        state.reuseWindowNextScoreContext = 0;
        state.reuseWindowNextPContext = 0;
        state.reuseWindowScoreSlotStallCycles = 0;
        state.reuseWindowPSlotStallCycles = 0;
        state.reuseWindowRowContextSwitches = 0;
        state.reuseWindowLastScoreSlotStallCycle = UINT64_MAX;
        state.reuseWindowLastPSlotStallCycle = UINT64_MAX;
        state.reuseWindowPvNextRowBlock = {{0, 1, 2}};
        state.reuseWindowPvNextWindow.fill(0);
        state.reuseWindowPvDispatchLane = 0;
        state.reuseWindowPvNextDispatchCycle = 0;
        state.reuseWindowPvRowCore.clear();
        state.reuseWindowPvAllocationPending.clear();
        state.reuseWindowPvRowNextWindow.clear();
        if (attentionWorkerClusterBridge_) {
            constexpr uint32_t sliceRows = 4;
            constexpr uint32_t slicesPerRowBlock = 16;
            const uint32_t rowBlocks = state.dispatch.expectedRows / blockM;
            state.reuseWindowPvRowCore.assign(rowBlocks, -1);
            state.reuseWindowPvAllocationPending.assign(rowBlocks, 0);
            state.reuseWindowPvRowNextWindow.assign(rowBlocks, 0);
            state.reuseWindowRowContexts.resize(
                static_cast<size_t>(rowBlocks) * slicesPerRowBlock);
            for (uint32_t rowBlock = 0; rowBlock < rowBlocks; ++rowBlock) {
                for (uint32_t slice = 0; slice < slicesPerRowBlock; ++slice) {
                    AttentionReuseWindowRowContext& context =
                        state.reuseWindowRowContexts[
                            rowBlock * slicesPerRowBlock + slice];
                    context.rowBlock = rowBlock;
                    context.rowOffset = slice * sliceRows;
                    context.p.assign(4u * sliceRows * blockN, 0.0f);
                    context.groupScale.assign(sliceRows, 1.0f);
                }
            }
            state.reuseWindowPvAggregates.resize(
                static_cast<size_t>(rowBlocks) * 4);
            for (AttentionReuseWindowPvAggregate& aggregate :
                    state.reuseWindowPvAggregates) {
                aggregate.p.assign(4u * blockM * blockN, 0.0f);
                aggregate.scales.assign(blockM, 1.0f);
            }
        }
        state.reuseWindowQkStartCycle = LastTickCycle;
        AttentionWorkerState* const target = &state;
        const uint64_t generation = state.generation;
        if (!workerCommandProcessor->setWindowCallbacks(
                [this, target, generation](uint64_t taskId, uint64_t,
                                   const std::vector<uint8_t>& tile) {
                    if (!attentionReuseWindowStateValid(target, generation) ||
                        tile.size() != 64u * 64u * sizeof(float)) {
                        return false;
                    }
                    const uint32_t nTiles =
                        target->dispatch.expectedCols / 64;
                    const uint32_t mTile = static_cast<uint32_t>(taskId) / nTiles;
                    const uint32_t nTile = static_cast<uint32_t>(taskId) % nTiles;
                    const uint32_t ownedMStart = attentionWorkerClusterBridge_
                        ? 0u : target->dispatch.workerSlot * 2;
                    const uint32_t ownedMTiles = attentionWorkerClusterBridge_
                        ? target->dispatch.expectedRows / 64 : 2u;
                    if (mTile < ownedMStart || mTile >= ownedMStart + ownedMTiles) {
                        return false;
                    }
                    const uint32_t localRowBegin = (mTile - ownedMStart) * 64;
                    for (uint32_t col = 0; col < 64; ++col) {
                        for (uint32_t row = 0; row < 64; ++row) {
                            float value = 0.0f;
                            std::memcpy(&value,
                                tile.data() +
                                    (static_cast<size_t>(col) * 64 + row) * sizeof(float),
                                sizeof(float));
                            target->reuseWindowScores[
                                static_cast<size_t>(localRowBegin + row) *
                                    target->dispatch.expectedCols +
                                nTile * 64 + col] = value;
                        }
                    }
                    target->reuseWindowFusionTiles++;
                    if (attentionWorkerClusterBridge_) {
                        const uint32_t readyIndex =
                            (localRowBegin / blockM) * nTiles + nTile;
                        target->reuseWindowScoreReady[readyIndex] = 1;
                        if (attentionWorker_.get() == target) {
                            startAttentionReuseWindowSoftmax();
                        }
                    }
                    return true;
                },
                [this, target, generation](uint64_t cycles, uint64_t start, uint64_t end) {
                    if (!attentionReuseWindowStateValid(target, generation)) return;
                    const bool ok = target->reuseWindowFusionTiles ==
                        target->reuseWindowExpectedFusionTiles;
                    output->output(
                        "[ATTENTION_REUSE_WINDOW_QK] core=%" PRIu64
                        " cycles=%" PRIu64 " start=%" PRIu64 " end=%" PRIu64
                        " fusion_tiles=%" PRIu64 " expected_tiles=%" PRIu64 "\n",
                        coreID, cycles, start, end,
                        target->reuseWindowFusionTiles,
                        target->reuseWindowExpectedFusionTiles);
                    if (!ok) {
                        if (attentionWorker_.get() == target) finishAttentionWorker(false);
                        return;
                    }
                    target->reuseWindowQkEndCycle = end;
                    target->reuseWindowPvCycles = 0;
                    target->reuseWindowQkComplete = true;
                    if (attentionWorker_.get() == target) {
                        startAttentionWorkerClusterQkAhead();
                        startAttentionReuseWindowSoftmax();
                    }
                }) || !workerCommandProcessor->startWindow(header)) {
            return false;
        }
        state.phase = AttentionWorkerPhase::QkCompute;
        return true;
    }

    bool attentionReuseWindowStateValid(const AttentionWorkerState* state,
                                        uint64_t generation) const {
        return state != nullptr && state->generation == generation &&
            (attentionWorker_.get() == state ||
             attentionWorkerClusterQkAhead_.get() == state);
    }

    void startAttentionWorkerClusterQkAhead() {
        if (!attentionWorkerClusterBridge_ || !attentionWorker_ ||
            !attentionWorker_->reuseWindowQkComplete ||
            attentionWorkerClusterQkAhead_ || attentionPendingDispatches_.empty() ||
            workerCommandProcessor == nullptr || workerCommandProcessor->isBusy()) return;
        ControlTransportMessage message = attentionPendingDispatches_.front();
        auto ahead = std::make_unique<AttentionWorkerState>();
        ahead->generation = nextAttentionWorkerGeneration_++;
        ahead->dispatch = message;
        ahead->phase = AttentionWorkerPhase::LoadingKv;
        AttentionWorkerState* const target = ahead.get();
        attentionWorkerClusterQkAhead_ = std::move(ahead);
        if (!startAttentionReuseWindowQkBridge(*target)) {
            attentionWorkerClusterQkAhead_.reset();
            return;
        }
        attentionPendingDispatches_.pop_front();
        statAttentionWorkerDispatchAcceptTick_->addData(getCurrentSimCycle());
        traceAttentionMilestone("worker", "worker_qk_ahead", "done",
            message.jobId, message.tag);
    }

    void startAttentionWorker(const ControlTransportMessage& message) {
        if (attentionWorker_) {
            const bool validGqa = message.numQueryHeads != 0 &&
                message.numKvHeads != 0 &&
                message.numQueryHeads % message.numKvHeads == 0 &&
                message.kvHeadIndex < message.numKvHeads;
            if (validGqa && attentionPendingDispatches_.size() < 1024) {
                attentionPendingDispatches_.push_back(message);
                startAttentionWorkerClusterQkAhead();
                return;
            }
        }
        const bool c1Shape = message.expectedRows == 32 && message.expectedCols == 32;
        const bool d1Shape = message.expectedRows == 64 && message.expectedCols == 64;
        const bool d3Shape = message.expectedRows == 20 && message.expectedCols == 70;
        const bool streamingShape = isStreamingAttentionWorkerShape(message);
        const uint64_t requiredWindow = streamingShape ?
            streamingAttentionWindowBytes(
                message.queryTileRows, message.kvTileRows, message.headDim) :
            (d3Shape ? ATTENTION_D3_WINDOW_BYTES :
             (d1Shape ? ATTENTION_D1_WINDOW_BYTES :
              ATTENTION_C1_WINDOW_BYTES));
        const uint64_t requiredWindowWithVTileBuffer = requiredWindow +
            (!attentionClusterEnable_ && attentionPvVTileReuse_ &&
             attentionPvVTileBufferOffset_ == 0 ?
                attentionPvVTileBufferBytes_ : 0);
        if (attentionWorker_ || globalMem == nullptr || array == nullptr || sfu == nullptr ||
            (attentionGenericGemmEnable_ && workerCommandProcessor->isBusy()) ||
            message.workerCore != coreID ||
            (!c1Shape && !d1Shape && !d3Shape && !streamingShape) ||
            ((!streamingShape && message.headDim != 64)) ||
            (attentionSequential64Enable_ ? message.queryTileRows != 64 :
             message.queryTileRows != 16) ||
            (attentionClusterEnable_ ?
                message.kvTileRows != attentionClusterConfig_.kvTileRows :
                (message.kvTileRows != 32u && message.kvTileRows != 64u)) ||
            (message.flags & ~GOLEM_ATTENTION_FLAG_CAUSAL) != 0 ||
            (attentionKvPairReuse_ &&
             ((message.flags & GOLEM_ATTENTION_FLAG_CAUSAL) != 0 ||
              !streamingShape || !attentionKvDoubleBuffer_ ||
               attentionOAccumulatorCBuffer_ ||
              (attentionKvQueryGroupSize_ != 2 &&
               attentionKvQueryGroupSize_ != 4))) ||
            (attentionClusterEnable_ &&
             (!streamingShape ||
              (message.flags & GOLEM_ATTENTION_FLAG_CAUSAL) != 0 ||
              message.headDim != 128 ||
              message.expectedRows % 16 != 0 ||
              message.expectedCols % 64 != 0)) ||
            (attentionClusterEnable_
                ? (numArrays < static_cast<int>(attentionClusterConfig_.arrays) ||
                   arrayInputSize * 2 != static_cast<int>(message.headDim) ||
                   arrayOutputSize !=
                       static_cast<int>(attentionClusterConfig_.arrayOutputs))
                : (attentionSequential64Enable_ ?
                   (numArrays < 64 || arrayInputSize != 64 ||
                    arrayOutputSize != 64) :
                  (numArrays < 16 ||
                   arrayInputSize != static_cast<int>(message.headDim) ||
                   arrayOutputSize != 16))) ||
            attentionWindowBytes_ < requiredWindowWithVTileBuffer ||
                attentionWindowOffset_ + requiredWindowWithVTileBuffer > globalMem->getSize()) {
            ControlTransportMessage rejected = message;
            rejected.kind = ControlTransportMessageKind::AttentionComplete;
            rejected.value = 0.0;
            traceAttentionMilestone(
                "worker", "worker_dispatch_accept", "fail", message.jobId,
                message.tag);
            globalMem->sendControlMessage(message.ownerCore, rejected);
            return;
        }
        attentionWorker_ = std::make_unique<AttentionWorkerState>();
        AttentionWorkerState& state = *attentionWorker_;
        state.generation = nextAttentionWorkerGeneration_++;
        state.dispatch = message;
        if (attentionClusterEnable_) {
            attentionCluster_ = std::make_unique<AttentionClusterState>();
            if (!attentionCluster_->configure(
                    state.generation, attentionClusterConfig_.qkArrays) ||
                !globalMem->beginAttentionGeneration(state.generation)) {
                finishAttentionWorker(false);
                return;
            }
            statAttentionClusterWorkerJobs_->addData(1);
        }
        statAttentionWorkerDispatchAcceptTick_->addData(getCurrentSimCycle());
        traceAttentionMilestone(
            "worker", "worker_dispatch_accept", "done", message.jobId,
            message.tag);
        state.phase = AttentionWorkerPhase::LoadingKv;
        const uint64_t base = globalMem->getBaseAddr() + attentionWindowOffset_;
        const uint64_t qTileBytes = static_cast<uint64_t>(message.queryTileRows) *
            message.headDim * sizeof(float);
        const uint64_t kvBytes = streamingShape ?
            static_cast<uint64_t>(message.kvTileRows) * message.headDim * sizeof(float) :
            static_cast<uint64_t>(message.expectedCols) * message.headDim * sizeof(float);
        const uint32_t kvBufferCount = attentionKvDoubleBuffer_ && streamingShape
            ? attentionKvBufferCount_ : 1u;
        const uint32_t queryBufferCount = attentionKvPairReuse_
            ? attentionKvQueryGroupSize_ : 1u;
        state.qLocalBuffers.resize(queryBufferCount);
        state.oLocalBuffers.resize(attentionClusterEnable_ ? 1u : queryBufferCount);
        for (uint32_t buffer = 0; buffer < state.qLocalBuffers.size(); ++buffer) {
            state.qLocalBuffers[buffer] = base +
                static_cast<uint64_t>(buffer) * qTileBytes;
        }
        const uint64_t kvBase = base +
            static_cast<uint64_t>(queryBufferCount) * qTileBytes;
        state.kLocalBuffers.resize(kvBufferCount);
        state.vLocalBuffers.resize(kvBufferCount);
        state.kvBuffers.resize(kvBufferCount);
        for (uint32_t buffer = 0; buffer < kvBufferCount; ++buffer) {
            state.kLocalBuffers[buffer] = kvBase +
                static_cast<uint64_t>(buffer) * 2 * kvBytes;
            state.vLocalBuffers[buffer] = state.kLocalBuffers[buffer] + kvBytes;
        }
        state.kLocal = state.kLocalBuffers[0];
        state.vLocal = state.vLocalBuffers[0];
        state.spLocal = state.vLocalBuffers[kvBufferCount - 1] + kvBytes;
        const uint64_t outputBase = state.spLocal +
            static_cast<uint64_t>(message.queryTileRows) *
                message.kvTileRows * sizeof(float);
        for (uint32_t buffer = 0; buffer < state.oLocalBuffers.size(); ++buffer) {
            state.oLocalBuffers[buffer] = outputBase +
                static_cast<uint64_t>(buffer) * qTileBytes;
        }
        if (!selectAttentionQueryStorage(state)) {
            finishAttentionWorker(false);
            return;
        }
        state.vTileBufferLocal = base + (attentionPvVTileBufferOffset_ != 0 ?
            attentionPvVTileBufferOffset_ : requiredWindow);
        if (attentionPvVTileReuse_ &&
            ((attentionPvVTileBufferOffset_ != 0 &&
              attentionPvVTileBufferOffset_ < requiredWindow) ||
             attentionPvVTileBufferOffset_ > attentionWindowBytes_ ||
             attentionPvVTileBufferBytes_ >
                 attentionWindowBytes_ - attentionPvVTileBufferOffset_)) {
            finishAttentionWorker(false);
            return;
        }
        attentionArrayPending_.assign(numArrays, 0);
        if (attentionReuseWindowQkBridge_) {
            if (!startAttentionReuseWindowQkBridge(state)) {
                finishAttentionWorker(false);
            }
            return;
        }
        if (streamingShape) {
            beginAttentionQueryTile();
            return;
        }
        const uint64_t generation = state.generation;
        globalMem->dma_read_from_host_to_globalmem(
            message.kAddr, kvBytes, state.kLocal,
            [this, generation](bool ok) {
                if (!attentionCallbackGenerationMatches(generation)) return;
                if (!ok) { finishAttentionWorker(false); return; }
                globalMem->dma_read_from_host_to_globalmem(
                    attentionWorker_->dispatch.vAddr,
                    static_cast<uint64_t>(attentionWorker_->dispatch.expectedCols) *
                        attentionWorker_->dispatch.headDim * sizeof(float),
                    attentionWorker_->vLocal, [this, generation](bool vOk) {
                        if (!attentionCallbackGenerationMatches(generation)) return;
                        if (!vOk) { finishAttentionWorker(false); return; }
                        beginAttentionQueryTile();
                    }, DmaRequestKind::AttentionKv);
            }, DmaRequestKind::AttentionKv);
    }

    void progressAttentionWorker() {
        recordAttentionClusterPipelineOverlap();
        if (!attentionWorkerClusterPv_.busy &&
            !attentionWorkerClusterPvQueue_.empty()) {
            startAttentionWorkerClusterPv();
        }
        for (const auto& draining : attentionWorkerClusterPvDraining_) {
            dispatchAttentionReuseWindowPvAggregates(*draining);
        }
        if (attentionWorker_ && attentionWorkerClusterBridge_ &&
            !attentionWorker_->reuseWindowRowContexts.empty()) {
            pumpAttentionReuseWindowSoftmax();
        }
        const auto oCompletions = attentionOAccumulator_.progress(LastTickCycle);
        bool resumePvOutput = false;
        for (const auto& completion : oCompletions) {
            if (!attentionWorker_) break;
            if (!attentionCallbackGenerationMatches(
                    completion.tag.generation)) continue;
            AttentionWorkerState& state = *attentionWorker_;
            const uint32_t queryContext =
                state.queryTileIndex % attentionClusterConfig_.groupSize;
            const int32_t slot = state.clusterOContextSlots[queryContext];
            if (!completion.ok || slot < 0 ||
                static_cast<uint32_t>(slot) != completion.slot ||
                !(state.clusterOTags[queryContext] == completion.tag)) {
                finishAttentionWorker(false);
                break;
            }
            if (completion.drain) {
                if (!state.attentionClusterODrainPending ||
                    state.attentionClusterODrainId != completion.id ||
                    completion.values.size() !=
                        AttentionOAccumulator::kValuesPerContext) {
                    finishAttentionWorker(false);
                    break;
                }
                state.attentionClusterODrainPending = false;
                state.attentionClusterODrainId = 0;
                const std::vector<uint8_t> bytes =
                    attentionFloatsToBytes(completion.values);
                statAttentionClusterODrainBytes_->addData(bytes.size());
                const uint64_t generation = state.generation;
                attentionLocalWrite(state.oLocal, bytes,
                    [this, generation](bool ok) {
                        if (!attentionCallbackGenerationMatches(generation)) return;
                        if (!ok) {
                            finishAttentionWorker(false);
                            return;
                        }
                        dmaAttentionQueryTileOutput();
                    });
                continue;
            }
            if (state.attentionClusterOCommitsPending == 0) {
                finishAttentionWorker(false);
                break;
            }
            --state.attentionClusterOCommitsPending;
            resumePvOutput = true;
        }
        if (attentionWorker_ && resumePvOutput &&
            attentionWorker_->phase == AttentionWorkerPhase::PvReadOutputs &&
            attentionWorker_->attentionClusterOCommitsPending == 0 &&
            attentionWorker_->clusterPvOutputCompleted ==
                attentionOutputSliceCount(*attentionWorker_) &&
            attentionWorker_->clusterPvOutputInFlight == 0) {
            readAttentionClusterPvOutput();
        }
        if (attentionWorker_ && attentionCrossTileOperandPipeline_) {
            pumpAttentionAheadOperands();
        }
        if (attentionWorker_ && attentionClusterEnable_) {
            pumpAttentionClusterAhead();
            pumpAttentionClusterQkMatrixLookahead();
            pumpAttentionClusterPvMatrixLookahead();
        }
        if (attentionWorker_ && attentionWorker_->clusterOwnerRetry) {
            std::function<void()> retry =
                std::move(attentionWorker_->clusterOwnerRetry);
            attentionWorker_->clusterOwnerRetry = {};
            retry();
        }
        if (attentionWorker_ && !attentionWorker_->clusterOwnerRetry &&
            !attentionWorker_->clusterOwnerRetryQueue.empty()) {
            attentionWorker_->clusterOwnerRetry =
                std::move(attentionWorker_->clusterOwnerRetryQueue.front());
            attentionWorker_->clusterOwnerRetryQueue.pop_front();
        }
        if (attentionWorker_ &&
            attentionWorker_->clusterOwnerLaunchNext <
                attentionWorker_->clusterOwnerLaunchEnd) {
            pumpAttentionClusterOwnerLaunch();
        }
        if (attentionWorker_ && attentionWorker_->vTileBufferWaiting &&
            LastTickCycle >= attentionWorker_->vTileBufferWaitUntilTick) {
            attentionWorker_->vTileBufferWaiting = false;
            attentionWorker_->vTileBufferBypassWait = true;
            statAttentionPvVTileBufferWaitTicks_->addData(
                attentionWorker_->vTileBufferCurrentWaitTicks);
            attentionWorker_->vTileBufferCurrentWaitTicks = 0;
            beginAttentionPvOutputSlice();
        }
        if (attentionWorker_ && attentionWorker_->attentionPvRestoreReadRetry) {
            prepareAttentionPvOutput();
        }
        if (attentionWorker_ && attentionWorker_->attentionPvOutputWriteRetry) {
            issueAttentionPvOutputWrite();
        }
        if (attentionWorker_ && attentionWorker_->localCallback &&
            !attentionWorker_->localInflight &&
            attentionWorker_->localOffset < attentionWorker_->localLength) {
            issueAttentionLocalTransferChunk();
        }
    }

    enum class ManagerAttentionJobPhase : uint8_t {
        ReadDescriptor,
        ReadTopology,
        Dispatch,
        Running,
        Complete,
    };

    struct ManagerAttentionJobState {
        uint64_t tag = 0;
        uint64_t descAddr = 0;
        GolemAttentionDescV2 desc = {};
        std::array<uint32_t, SFU_WORKER_TOPOLOGY_MAX_WORKERS> workerCoreIds = {};
        uint32_t workersDispatched = 0;
        uint32_t workersCompleted = 0;
        uint32_t completionBitmap = 0;
        uint32_t managersCompleted = 0;
        uint32_t managerCompletionBitmap = 0;
        SFUStatus status = SFUStatus::Pending;
        ManagerAttentionJobPhase phase = ManagerAttentionJobPhase::ReadDescriptor;
        bool readInflight = false;
        uint64_t readTag = 0;
    };

    uint32_t managerAttentionWorkerCount(const ManagerAttentionJobState& state) const {
        return attentionWorkerClusterBridge_
            ? std::min(attentionWorkerClusterQkWorkersPerManager_,
                       state.desc.worker_count)
            : state.desc.worker_count;
    }

    void failManagerAttentionJob(ManagerAttentionJobState& state, SFUStatus status) {
        if (state.phase != ManagerAttentionJobPhase::Complete) {
            traceAttentionMilestone(
                "manager", "manager_job", "fail", state.desc.job_id,
                state.tag);
        }
        state.status = status;
        state.readInflight = false;
        state.phase = ManagerAttentionJobPhase::Complete;
    }

    bool tryIssueManagerAttentionJobCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) return true;
        const uint64_t tag = cmd->rs2;
        uint64_t response = 0;
        if (globalMem == nullptr || managerAttentionJobs_.count(tag) != 0) {
            response = static_cast<uint64_t>(SFUStatus::InvalidDescriptor);
        } else {
            ManagerAttentionJobState state;
            state.tag = tag;
            state.descAddr = cmd->rs1;
            managerAttentionJobs_.emplace(tag, std::move(state));
            statAttentionManagerJobsIssued_->addData(1);
            statAttentionManagerDescriptorAcceptTick_->addData(getCurrentSimCycle());
            traceAttentionMilestone(
                "manager",
                coreID == 0 ? "root_descriptor_accept" :
                    "manager_descriptor_accept",
                "done", 0, tag);
        }
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, response, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryWaitManagerAttentionJobCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) return true;
        auto it = managerAttentionJobs_.find(cmd->rs1);
        if (it == managerAttentionJobs_.end()) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, static_cast<uint64_t>(SFUStatus::InvalidDescriptor),
                cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        if (it->second.phase != ManagerAttentionJobPhase::Complete) return false;
        const uint64_t status = static_cast<uint64_t>(it->second.status);
        statAttentionManagerWaitObservedTick_->addData(getCurrentSimCycle());
        traceAttentionMilestone(
            "manager",
            coreID == 0 ? "software_wait_observed" :
                "manager_wait_observed",
            it->second.status == SFUStatus::Success ? "done" : "fail",
            it->second.desc.job_id, it->second.tag);
        managerAttentionJobs_.erase(it);
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, status, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool validateManagerAttentionDescriptor(const GolemAttentionDescV2& desc) const {
        const bool c1Shape = desc.group_query_rows == 32 && desc.kv_length == 32;
        const bool d1Shape = desc.group_query_rows == 64 && desc.kv_length == 64;
        const bool d3Shape = desc.group_query_rows == 20 && desc.kv_length == 70;
        const bool streamingShape = desc.group_query_rows != 0 && desc.group_query_rows % 64 == 0 &&
            desc.kv_length != 0 && desc.kv_length % 128 == 0 &&
            (desc.head_dim == 64 || desc.head_dim == 128) &&
            desc.worker_count == 4 &&
            desc.group_query_row_begin == static_cast<uint32_t>(coreID) * desc.group_query_rows &&
            desc.kv_rows_per_memory_node == desc.kv_length / 4 &&
            desc.kv_node_stride_bytes != 0 && desc.tensor_root_core == 0 &&
            desc.tensor_manager_count == 4 &&
            desc.tensor_manager_slot < desc.tensor_manager_count &&
            desc.tensor_manager_slot == coreID;
        const bool validGqa = desc.num_query_heads != 0 && desc.num_kv_heads != 0 &&
            desc.num_query_heads % desc.num_kv_heads == 0 &&
            (desc.num_query_heads / desc.num_kv_heads == 1 ||
             desc.num_query_heads / desc.num_kv_heads == 2 ||
             desc.num_query_heads / desc.num_kv_heads == 4) &&
            desc.kv_head_index < desc.num_kv_heads &&
            desc.group_query_rows % (desc.num_query_heads / desc.num_kv_heads) == 0;
        const uint64_t requiredWindow = streamingShape ?
            streamingAttentionWindowBytes(
                desc.query_tile_rows, desc.kv_tile_rows, desc.head_dim) :
            (d3Shape ? ATTENTION_D3_WINDOW_BYTES :
            (d1Shape ? ATTENTION_D1_WINDOW_BYTES :
             ATTENTION_C1_WINDOW_BYTES));
        return desc.magic == GOLEM_ATTENTION_DESC_MAGIC &&
            desc.version == GOLEM_ATTENTION_DESC_VERSION &&
            desc.size_bytes == sizeof(GolemAttentionDescV2) &&
            validGqa &&
            (c1Shape || d1Shape || d3Shape || streamingShape) &&
            ((!streamingShape && desc.head_dim == 64) || streamingShape) &&
            (attentionSequential64Enable_ ? desc.query_tile_rows == 64 :
             desc.query_tile_rows == 16) &&
            (attentionClusterEnable_ ?
                desc.kv_tile_rows == attentionClusterConfig_.kvTileRows :
                (desc.kv_tile_rows == 32u || desc.kv_tile_rows == 64u)) &&
            ((streamingShape && desc.worker_count == 4) ||
             (!streamingShape && desc.worker_count == 1)) &&
            (desc.flags & ~GOLEM_ATTENTION_FLAG_CAUSAL) == 0 &&
            desc.q_addr != 0 && desc.k_addr != 0 && desc.v_addr != 0 &&
            desc.output_addr != 0 && desc.topology_gm_addr != 0 &&
            attentionWindowBytes_ >= requiredWindow;
    }

    void progressManagerAttentionJobs() {
        if (globalMem == nullptr) return;
        for (auto& entry : managerAttentionJobs_) {
            ManagerAttentionJobState& state = entry.second;
            if (state.readInflight || state.phase == ManagerAttentionJobPhase::Running ||
                state.phase == ManagerAttentionJobPhase::Complete) continue;
            if (state.phase == ManagerAttentionJobPhase::ReadDescriptor) {
                const uint64_t jobTag = state.tag;
                const uint64_t readTag = allocateLocalTransferTag();
                const bool accepted = globalMem->localReadAsync(
                    state.descAddr, sizeof(GolemAttentionDescV2), LocalMemoryClient::Control,
                    readTag, [this, jobTag, readTag](bool ok, uint64_t tag,
                                                     const std::vector<uint8_t>& bytes) {
                        auto it = managerAttentionJobs_.find(jobTag);
                        if (it == managerAttentionJobs_.end()) return;
                        ManagerAttentionJobState& callbackState = it->second;
                        callbackState.readInflight = false;
                        if (!ok || tag != readTag || bytes.size() != sizeof(GolemAttentionDescV2)) {
                            failManagerAttentionJob(callbackState, SFUStatus::InvalidDescriptor);
                            return;
                        }
                        std::memcpy(&callbackState.desc, bytes.data(), sizeof(callbackState.desc));
                        if (!validateManagerAttentionDescriptor(callbackState.desc)) {
                            failManagerAttentionJob(callbackState, SFUStatus::InvalidDescriptor);
                            return;
                        }
                        callbackState.phase = ManagerAttentionJobPhase::ReadTopology;
                    });
                if (accepted) {
                    state.readInflight = true;
                    state.readTag = readTag;
                }
            } else if (state.phase == ManagerAttentionJobPhase::ReadTopology) {
                const uint64_t jobTag = state.tag;
                const uint64_t readTag = allocateLocalTransferTag();
                const bool accepted = globalMem->localReadAsync(
                    state.desc.topology_gm_addr, sizeof(SFUWorkerTopologyMapV1),
                    LocalMemoryClient::Control, readTag,
                    [this, jobTag, readTag](bool ok, uint64_t tag,
                                            const std::vector<uint8_t>& bytes) {
                        auto it = managerAttentionJobs_.find(jobTag);
                        if (it == managerAttentionJobs_.end()) return;
                        ManagerAttentionJobState& callbackState = it->second;
                        callbackState.readInflight = false;
                        SFUWorkerTopologyMapV1 topology = {};
                        if (!ok || tag != readTag || bytes.size() != sizeof(topology)) {
                            failManagerAttentionJob(callbackState, SFUStatus::InvalidDescriptor);
                            return;
                        }
                        std::memcpy(&topology, bytes.data(), sizeof(topology));
                        if (topology.magic != SFU_WORKER_TOPOLOGY_MAP_MAGIC ||
                            topology.version != SFU_WORKER_TOPOLOGY_MAP_VERSION ||
                            topology.size_bytes != sizeof(topology) ||
                            topology.worker_count != callbackState.desc.worker_count ||
                            topology.worker_count == 0 ||
                            topology.worker_count > SFU_WORKER_TOPOLOGY_MAX_WORKERS) {
                            failManagerAttentionJob(callbackState, SFUStatus::InvalidDescriptor);
                            return;
                        }
                        for (uint32_t slot = 0; slot < topology.worker_count; ++slot) {
                            const uint32_t workerCore = topology.worker_core_ids[slot];
                            if (workerCore == coreID) {
                                failManagerAttentionJob(
                                    callbackState, SFUStatus::InvalidDescriptor);
                                return;
                            }
                            for (uint32_t prior = 0; prior < slot; ++prior) {
                                if (topology.worker_core_ids[prior] == workerCore) {
                                    failManagerAttentionJob(
                                        callbackState, SFUStatus::InvalidDescriptor);
                                    return;
                                }
                            }
                            callbackState.workerCoreIds[slot] = workerCore;
                        }
                        callbackState.phase = ManagerAttentionJobPhase::Dispatch;
                    });
                if (accepted) {
                    state.readInflight = true;
                    state.readTag = readTag;
                }
            } else if (state.phase == ManagerAttentionJobPhase::Dispatch) {
                const uint32_t workerSlot = state.workersDispatched;
                const uint32_t dispatchedWorkers = managerAttentionWorkerCount(state);
                if (workerSlot >= dispatchedWorkers) {
                    state.phase = ManagerAttentionJobPhase::Running;
                    continue;
                }
                const uint32_t workerCore = state.workerCoreIds[workerSlot];
                const uint32_t rowsPerWorker = dispatchedWorkers == 1 ?
                    state.desc.group_query_rows :
                    (state.desc.group_query_rows + dispatchedWorkers - 1) /
                        dispatchedWorkers;
                const uint32_t localQueryBegin = workerSlot * rowsPerWorker;
                const uint32_t workerRows = dispatchedWorkers == 1 ? state.desc.group_query_rows :
                    std::min(rowsPerWorker,
                             state.desc.group_query_rows - localQueryBegin);
                ControlTransportMessage message = {};
                message.kind = ControlTransportMessageKind::AttentionDispatch;
                message.jobId = state.desc.job_id;
                message.tag = state.tag;
                message.ownerCore = static_cast<uint32_t>(coreID);
                message.workerSlot = workerSlot;
                message.workerCore = workerCore;
                message.row = state.desc.group_query_row_begin + localQueryBegin;
                message.expectedWorkers = dispatchedWorkers;
                message.expectedRows = workerRows;
                message.expectedCols = state.desc.kv_length;
                message.headDim = state.desc.head_dim;
                message.queryTileRows = state.desc.query_tile_rows;
                message.kvTileRows = state.desc.kv_tile_rows;
                message.flags = state.desc.flags;
                message.nodeStrideBytes = state.desc.kv_node_stride_bytes;
                message.rowsPerBand = state.desc.kv_rows_per_memory_node;
                message.rowsPerBand = state.desc.kv_rows_per_memory_node;
                message.qAddr = state.desc.q_addr +
                    static_cast<uint64_t>(localQueryBegin) * state.desc.head_dim * sizeof(float);
                message.kAddr = state.desc.k_addr;
                message.vAddr = state.desc.v_addr;
                message.oAddr = state.desc.output_addr;
                message.numQueryHeads = state.desc.num_query_heads;
                message.numKvHeads = state.desc.num_kv_heads;
                message.kvHeadIndex = state.desc.kv_head_index;
                message.queryLength =
                    state.desc.group_query_rows /
                    (state.desc.num_query_heads / state.desc.num_kv_heads);
                message.groupQueryRowBegin = localQueryBegin;
                message.sendCycle = getCurrentSimCycle();
                if (!globalMem->sendControlMessage(workerCore, message)) {
                    failManagerAttentionJob(state, SFUStatus::GlobalMemoryUnavailable);
                } else {
                    state.workersDispatched += 1;
                    if (state.workersDispatched == dispatchedWorkers) {
                        state.phase = ManagerAttentionJobPhase::Running;
                        statAttentionManagerDispatchTick_->addData(getCurrentSimCycle());
                        traceAttentionMilestone(
                            "manager", "manager_dispatch_complete", "done",
                            state.desc.job_id, state.tag);
                    }
                }
            }
        }
    }

    bool handleManagerAttentionCompletion(const ControlTransportMessage& message) {
        if (message.kind != ControlTransportMessageKind::AttentionComplete ||
            message.ownerCore != coreID) return false;
        auto it = managerAttentionJobs_.find(message.tag);
        if (it == managerAttentionJobs_.end()) return false;
        ManagerAttentionJobState& state = it->second;
        const uint32_t dispatchedWorkers = managerAttentionWorkerCount(state);
        const uint32_t workerSlot = message.workerSlot;
        const bool validPhase = state.phase == ManagerAttentionJobPhase::Dispatch ||
            state.phase == ManagerAttentionJobPhase::Running;
        if (!validPhase || message.jobId != state.desc.job_id ||
            message.expectedWorkers != dispatchedWorkers ||
            workerSlot >= dispatchedWorkers ||
            message.workerCore != state.workerCoreIds[workerSlot] ||
            (state.completionBitmap & (1u << workerSlot)) != 0 || message.value != 1.0) {
            failManagerAttentionJob(state, SFUStatus::InvalidDescriptor);
            return true;
        }
        state.completionBitmap |= 1u << workerSlot;
        state.workersCompleted += 1;
        if (state.workersCompleted == dispatchedWorkers) {
            statAttentionManagerBandsCompleted_->addData(1);
            statAttentionManagerLocalCompleteTick_->addData(getCurrentSimCycle());
            traceAttentionMilestone(
                "manager", "manager_local_complete", "done",
                state.desc.job_id, state.tag);
            if (state.desc.tensor_manager_count <= 1) {
                state.status = SFUStatus::Success;
                state.phase = ManagerAttentionJobPhase::Complete;
                statAttentionManagerJobsCompleted_->addData(1);
                statAttentionTensorJobsCompleted_->addData(1);
                statAttentionTensorCompleteTick_->addData(getCurrentSimCycle());
                traceAttentionMilestone(
                    "manager", "root_tensor_complete", "done",
                    state.desc.job_id, state.tag);
            } else {
                ControlTransportMessage managerCompletion = {};
                managerCompletion.kind =
                    ControlTransportMessageKind::AttentionManagerComplete;
                managerCompletion.jobId = state.desc.job_id;
                managerCompletion.tag = state.tag;
                managerCompletion.ownerCore = state.desc.tensor_root_core;
                managerCompletion.workerSlot = state.desc.tensor_manager_slot;
                managerCompletion.workerCore = static_cast<uint32_t>(coreID);
                managerCompletion.expectedWorkers = state.desc.tensor_manager_count;
                managerCompletion.sendCycle = getCurrentSimCycle();
                managerCompletion.value = 1.0;
                if (coreID == state.desc.tensor_root_core) {
                    handleAttentionManagerBandCompletion(managerCompletion);
                } else if (!globalMem->sendControlMessage(
                               state.desc.tensor_root_core, managerCompletion)) {
                    failManagerAttentionJob(
                        state, SFUStatus::GlobalMemoryUnavailable);
                } else {
                    state.status = SFUStatus::Success;
                    state.phase = ManagerAttentionJobPhase::Complete;
                    statAttentionManagerJobsCompleted_->addData(1);
                }
            }
        }
        return true;
    }

    bool handleAttentionManagerBandCompletion(
        const ControlTransportMessage& message) {
        if (message.kind != ControlTransportMessageKind::AttentionManagerComplete) {
            return false;
        }
        auto it = managerAttentionJobs_.find(message.tag);
        if (it == managerAttentionJobs_.end()) return true;
        ManagerAttentionJobState& state = it->second;
        const uint32_t managerSlot = message.workerSlot;
        if (coreID != state.desc.tensor_root_core || message.ownerCore != coreID ||
            message.jobId != state.desc.job_id ||
            message.expectedWorkers != state.desc.tensor_manager_count ||
            managerSlot >= state.desc.tensor_manager_count ||
            message.workerCore != managerSlot ||
            (state.managerCompletionBitmap & (1u << managerSlot)) != 0 ||
            message.value != 1.0) {
            failManagerAttentionJob(state, SFUStatus::InvalidDescriptor);
            return true;
        }
        state.managerCompletionBitmap |= 1u << managerSlot;
        state.managersCompleted += 1;
        statAttentionManagerBandCompletionsReceived_->addData(1);
        statAttentionManagerBandCompletionReceivedTick_->addData(getCurrentSimCycle());
        if (state.managersCompleted == state.desc.tensor_manager_count) {
            state.status = SFUStatus::Success;
            state.phase = ManagerAttentionJobPhase::Complete;
            statAttentionManagerJobsCompleted_->addData(1);
            statAttentionTensorJobsCompleted_->addData(1);
            statAttentionTensorCompleteTick_->addData(getCurrentSimCycle());
            traceAttentionMilestone(
                "manager", "root_tensor_complete", "done",
                state.desc.job_id, state.tag);
        }
        return true;
    }

    enum class ManagerTensorJobPhase : uint8_t {
        ReadDescriptor,
        ReadParams,
        ReadTopology,
        Dispatch,
        Running,
        Complete,
    };

    struct ManagerTensorJobState {
        uint64_t tag = 0;
        uint64_t descAddr = 0;
        SFUJobDesc desc = {};
        SFUSoftmaxJobParamsV1 params = {};
        std::vector<uint32_t> workerCoreIds;
        std::vector<uint8_t> completionSeen;
        uint32_t rowsCompleted = 0;
        SFUStatus status = SFUStatus::Pending;
        ManagerTensorJobPhase phase = ManagerTensorJobPhase::ReadDescriptor;
        bool readInflight = false;
        uint64_t readTag = 0;
    };

    bool tryIssueManagerTensorJobCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        const uint64_t tag = cmd->rs2;
        if (globalMem == nullptr || managerTensorJobs_.find(tag) != managerTensorJobs_.end()) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        ManagerTensorJobState state;
        state.tag = tag;
        state.descAddr = cmd->rs1;
        managerTensorJobs_.emplace(tag, std::move(state));
        statTensorManagerJobsIssued_->addData(1);
        statTensorManagerDescriptorAcceptTick_->addData(getCurrentSimCycle());
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    bool tryWaitManagerTensorJobCommand(SST::Vanadis::RoCCCommand* cmd) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return true;
        }
        auto it = managerTensorJobs_.find(cmd->rs1);
        if (it == managerTensorJobs_.end()) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                cmd->inst->rd, static_cast<uint64_t>(SFUStatus::InvalidDescriptor),
                cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return true;
        }
        if (it->second.phase != ManagerTensorJobPhase::Complete) {
            return false;
        }
        const uint64_t status = static_cast<uint64_t>(it->second.status);
        statTensorManagerWaitObservedTick_->addData(getCurrentSimCycle());
        managerTensorJobs_.erase(it);
        enqueueResponse(new SST::Vanadis::RoCCResponse(
            cmd->inst->rd, status, cmd->cmd_id, cmd->hw_thread));
        delete cmd;
        return true;
    }

    void failManagerTensorJob(ManagerTensorJobState& state, SFUStatus status) {
        state.status = status;
        state.readInflight = false;
        state.phase = ManagerTensorJobPhase::Complete;
    }

    template <typename Metadata>
    void issueManagerMetadataRead(ManagerTensorJobState& state,
                                  uint64_t address,
                                  ManagerTensorJobPhase nextPhase,
                                  Metadata ManagerTensorJobState::* destination) {
        if (state.readInflight || sizeof(Metadata) > globalMem->localMaxRequestBytes()) {
            if (sizeof(Metadata) > globalMem->localMaxRequestBytes()) {
                failManagerTensorJob(state, SFUStatus::InvalidDescriptor);
            }
            return;
        }
        const uint64_t tag = allocateLocalTransferTag();
        const uint64_t jobTag = state.tag;
        const bool accepted = globalMem->localReadAsync(
            address, sizeof(Metadata), LocalMemoryClient::Control, tag,
            [this, jobTag, tag, nextPhase, destination](
                bool success, uint64_t callbackTag, const std::vector<uint8_t>& bytes) {
                auto it = managerTensorJobs_.find(jobTag);
                if (it == managerTensorJobs_.end()) {
                    return;
                }
                ManagerTensorJobState& callbackState = it->second;
                if (!callbackState.readInflight || callbackState.readTag != callbackTag ||
                    callbackTag != tag || !success || bytes.size() != sizeof(Metadata)) {
                    failManagerTensorJob(callbackState, SFUStatus::InvalidDescriptor);
                    return;
                }
                std::memcpy(&(callbackState.*destination), bytes.data(), sizeof(Metadata));
                callbackState.readInflight = false;
                callbackState.phase = nextPhase;
            });
        if (accepted) {
            state.readInflight = true;
            state.readTag = tag;
        }
    }

    bool validateManagerDescriptor(const ManagerTensorJobState& state) const {
        const SFUJobDesc& desc = state.desc;
        return desc.owner_core == coreID && desc.input0_addr != 0 && desc.output_addr != 0 &&
            desc.params_addr != 0 && desc.rows != 0 && desc.cols != 0 &&
            desc.chunk_elems != 0 && desc.worker_cores != 0 &&
            desc.worker_cores <= SFU_WORKER_TOPOLOGY_MAX_WORKERS &&
            desc.dtype == SFU_JOB_DTYPE_FP32 &&
            desc.op_type == static_cast<uint32_t>(SFUJobOp::SOFTMAX_ROW) &&
            (desc.flags & (SFU_JOB_FLAG_ROW_ENGINE_MODEL |
                           SFU_JOB_FLAG_TENSOR_ROW_ENGINE)) ==
                (SFU_JOB_FLAG_ROW_ENGINE_MODEL | SFU_JOB_FLAG_TENSOR_ROW_ENGINE);
    }

    bool validateManagerParams(const ManagerTensorJobState& state) const {
        const SFUSoftmaxJobParamsV1& params = state.params;
        return params.magic == SFU_SOFTMAX_JOB_PARAMS_MAGIC &&
            params.version == SFU_SOFTMAX_JOB_PARAMS_VERSION_MANAGER &&
            params.size_bytes == sizeof(SFUSoftmaxJobParamsV1) &&
            params.mapping_policy == SFU_SOFTMAX_MAPPING_EXPLICIT_TOPOLOGY &&
            params.hbm_layout == SFU_SOFTMAX_HBM_LAYOUT_BAND_STRIPED &&
            params.data_node_mask != 0 && params.node_stride_bytes != 0 &&
            params.rows_per_band != 0 && params.coordinator_core == coreID &&
            params.completion_addr != 0;
    }

    bool validateAndInstallManagerTopology(ManagerTensorJobState& state,
                                           const SFUWorkerTopologyMapV1& topology) {
        if (topology.magic != SFU_WORKER_TOPOLOGY_MAP_MAGIC ||
            topology.version != SFU_WORKER_TOPOLOGY_MAP_VERSION ||
            topology.size_bytes != sizeof(SFUWorkerTopologyMapV1) ||
            topology.worker_count != state.desc.worker_cores) {
            return false;
        }
        state.workerCoreIds.clear();
        for (uint32_t slot = 0; slot < topology.worker_count; ++slot) {
            const uint32_t workerCore = topology.worker_core_ids[slot];
            if (workerCore == coreID ||
                std::find(state.workerCoreIds.begin(), state.workerCoreIds.end(), workerCore) !=
                    state.workerCoreIds.end()) {
                return false;
            }
            state.workerCoreIds.push_back(workerCore);
        }
        statTensorManagerWorkersMapped_->addData(state.workerCoreIds.size());
        return true;
    }

    void dispatchManagerTensorJob(ManagerTensorJobState& state) {
        const uint32_t rowsPerBand = state.params.rows_per_band;
        const uint32_t bands = (state.desc.rows + rowsPerBand - 1) / rowsPerBand;
        if (bands == 0 || bands > state.workerCoreIds.size()) {
            failManagerTensorJob(state, SFUStatus::InvalidShape);
            return;
        }
        state.completionSeen.assign(bands, 0);
        for (uint32_t band = 0; band < bands; ++band) {
            const uint32_t workerSlot = band % state.desc.worker_cores;
            const uint32_t workerCore = state.workerCoreIds[workerSlot];
            ControlTransportMessage dispatch = {};
            dispatch.kind = ControlTransportMessageKind::TensorRowDispatch;
            dispatch.jobId = state.desc.job_id;
            dispatch.tag = state.tag;
            dispatch.ownerCore = static_cast<uint32_t>(coreID);
            dispatch.workerSlot = workerSlot;
            dispatch.workerCore = workerCore;
            dispatch.row = band * rowsPerBand;
            dispatch.expectedWorkers = state.desc.worker_cores;
            dispatch.expectedRows = std::min(rowsPerBand, state.desc.rows - dispatch.row);
            dispatch.expectedCols = state.desc.cols;
            dispatch.inputAddr = state.desc.input0_addr;
            dispatch.outputAddr = state.desc.output_addr;
            dispatch.nodeStrideBytes = state.params.node_stride_bytes;
            dispatch.dataNodeMask = state.params.data_node_mask;
            dispatch.rowsPerBand = rowsPerBand;
            if ((state.params.flags & SFU_SOFTMAX_PARAMS_FLAG_ATTENTION) != 0) {
                const uint64_t headDim = state.params.reserved0;
                if (headDim == 0) {
                    failManagerTensorJob(state, SFUStatus::InvalidShape);
                    return;
                }
                const double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
                dispatch.value =
                    (state.params.flags & SFU_SOFTMAX_PARAMS_FLAG_CAUSAL) != 0
                        ? -scale : scale;
            }
            if (!globalMem->sendControlMessage(workerCore, dispatch)) {
                failManagerTensorJob(state, SFUStatus::GlobalMemoryUnavailable);
                return;
            }
            statTensorManagerBandDispatchTick_->addData(getCurrentSimCycle());
        }
        statTensorManagerRowsDispatched_->addData(state.desc.rows);
        state.phase = ManagerTensorJobPhase::Running;
    }

    void progressManagerTensorJobs() {
        for (auto& entry : managerTensorJobs_) {
            ManagerTensorJobState& state = entry.second;
            if (state.phase == ManagerTensorJobPhase::ReadDescriptor) {
                issueManagerMetadataRead(state, state.descAddr,
                    ManagerTensorJobPhase::ReadParams, &ManagerTensorJobState::desc);
            } else if (state.phase == ManagerTensorJobPhase::ReadParams && !state.readInflight) {
                if (!validateManagerDescriptor(state)) {
                    failManagerTensorJob(state, SFUStatus::InvalidDescriptor);
                } else {
                    issueManagerMetadataRead(state, state.desc.params_addr,
                        ManagerTensorJobPhase::ReadTopology, &ManagerTensorJobState::params);
                }
            } else if (state.phase == ManagerTensorJobPhase::ReadTopology &&
                       !state.readInflight) {
                if (!validateManagerParams(state)) {
                    failManagerTensorJob(state, SFUStatus::InvalidDescriptor);
                    continue;
                }
                if (sizeof(SFUWorkerTopologyMapV1) > globalMem->localMaxRequestBytes()) {
                    failManagerTensorJob(state, SFUStatus::InvalidDescriptor);
                    continue;
                }
                const uint64_t jobTag = state.tag;
                const uint64_t readTag = allocateLocalTransferTag();
                const bool accepted = globalMem->localReadAsync(
                    state.params.completion_addr, sizeof(SFUWorkerTopologyMapV1),
                    LocalMemoryClient::Control, readTag,
                    [this, jobTag, readTag](bool success, uint64_t callbackTag,
                                            const std::vector<uint8_t>& bytes) {
                        auto it = managerTensorJobs_.find(jobTag);
                        if (it == managerTensorJobs_.end()) return;
                        ManagerTensorJobState& callbackState = it->second;
                        if (!callbackState.readInflight || callbackTag != readTag ||
                            callbackState.readTag != callbackTag || !success ||
                            bytes.size() != sizeof(SFUWorkerTopologyMapV1)) {
                            failManagerTensorJob(callbackState, SFUStatus::InvalidDescriptor);
                            return;
                        }
                        SFUWorkerTopologyMapV1 topology = {};
                        std::memcpy(&topology, bytes.data(), sizeof(topology));
                        callbackState.readInflight = false;
                        if (!validateAndInstallManagerTopology(callbackState, topology)) {
                            failManagerTensorJob(callbackState, SFUStatus::InvalidDescriptor);
                            return;
                        }
                        callbackState.phase = ManagerTensorJobPhase::Dispatch;
                    });
                if (accepted) {
                    state.readInflight = true;
                    state.readTag = readTag;
                }
            } else if (state.phase == ManagerTensorJobPhase::Dispatch) {
                dispatchManagerTensorJob(state);
            }
        }
    }

    bool handleManagerTensorCompletion(const ControlTransportMessage& message) {
        if (message.kind != ControlTransportMessageKind::TensorRowComplete ||
            message.ownerCore != coreID) {
            return false;
        }
        auto it = managerTensorJobs_.find(message.tag);
        if (it == managerTensorJobs_.end()) {
            return false;
        }
        ManagerTensorJobState& state = it->second;
        const uint32_t rowsPerBand = state.params.rows_per_band;
        if (state.phase != ManagerTensorJobPhase::Running || rowsPerBand == 0 ||
            message.jobId != state.desc.job_id || message.row % rowsPerBand != 0) {
            failManagerTensorJob(state, SFUStatus::InvalidDescriptor);
            return true;
        }
        const uint32_t band = message.row / rowsPerBand;
        const uint32_t expectedSlot = band % state.desc.worker_cores;
        const uint32_t expectedRows = std::min(rowsPerBand, state.desc.rows - message.row);
        if (band >= state.completionSeen.size() || message.workerSlot != expectedSlot ||
            message.workerCore != state.workerCoreIds[expectedSlot] ||
            message.expectedWorkers != state.desc.worker_cores ||
            message.expectedRows != expectedRows || message.expectedCols != state.desc.cols ||
            message.rowsPerBand != rowsPerBand || state.completionSeen[band] != 0 ||
            message.value != 1.0) {
            failManagerTensorJob(state, SFUStatus::InvalidDescriptor);
            return true;
        }
        state.completionSeen[band] = 1;
        statTensorManagerCompletionReceivedTick_->addData(getCurrentSimCycle());
        statTensorManagerRowsCompleted_->addData(expectedRows);
        state.rowsCompleted += expectedRows;
        if (state.rowsCompleted == state.desc.rows &&
            std::all_of(state.completionSeen.begin(), state.completionSeen.end(),
                        [](uint8_t seen) { return seen != 0; })) {
            state.status = SFUStatus::Success;
            state.phase = ManagerTensorJobPhase::Complete;
            statTensorManagerJobsCompleted_->addData(1);
            statTensorManagerCompleteTick_->addData(getCurrentSimCycle());
        }
        return true;
    }

    WorkerTaskListHeader makeAttentionWorkerClusterPvHeader(
        const ControlTransportMessage& msg) const {
        constexpr uint32_t block = 64;
        constexpr size_t panelBytes = block * block * sizeof(float);
        const uint64_t localBase = globalMem->getBaseAddr();
        WorkerTaskListHeader header{};
        header.worker_slot = 0;
        header.task_count = 1;
        header.active_worker_cores = 1;
        header.total_groups = 4;
        header.data_memory_node_count = 4;
        header.mem_node_size = msg.nodeStrideBytes;
        header.m = block; header.n = 128; header.k = 256;
        header.hw_input_size = block; header.hw_output_size = block;
        header.block_m = block; header.block_n = block; header.block_k = block;
        header.elem_bytes = sizeof(float);
        header.mat_stride_bytes = panelBytes;
        header.vec_stride_bytes = block * sizeof(float);
        constexpr uint64_t panelVOffset = 0x06000000;
        const uint64_t vHeadOffset =
            static_cast<uint64_t>(msg.kvHeadIndex) * 8 * panelBytes;
        header.off_gemm_vec_base =
            static_cast<uint64_t>(1 + msg.kvTileRows) * msg.nodeStrideBytes +
            panelVOffset + vHeadOffset;
        header.local_mat_ping_gm_addr = localBase + 0x10000;
        header.local_vec_ping_gm_addr = localBase + 0x70000;
        header.local_mat_slot_stride_bytes = panelBytes;
        header.local_vec_slot_stride_bytes = panelBytes;
        header.local_slot_count = 24;
        header.local_accum_gm_addr = localBase + 0xD0000;
        header.local_out_gm_addr = localBase + 0xD4000;
        header.a_reuse_n_tiles = 2; header.n_group_count = 1;
        header.b_reuse_m_tiles = 1; header.m_group_count = 1;
        header.descriptor_start_cycle = getCurrentSimCycle();
        header.operand_layout = 3;
        header.scheduler_worker_slot = 0;
        header.attention_pv_cache_gm_addr = localBase + 0x100000;
        header.attention_pv_cache_bytes = 0x80000;
        return header;
    }

    size_t attentionWorkerClusterVEntryBytes(
        const WorkerTaskListHeader& header) const {
        const uint32_t reuseN = std::max<uint32_t>(header.a_reuse_n_tiles, 1u);
        const uint32_t kTiles = header.block_k == 0 ? 0 : header.k / header.block_k;
        return static_cast<size_t>(reuseN) * kTiles *
            header.local_vec_slot_stride_bytes;
    }

    uint32_t attentionWorkerClusterVLeader(uint32_t kvHead) const {
        const uint32_t pvBase = 4u +
            attentionWorkerClusterQkWorkersPerManager_ * 4u;
        const uint32_t pvCount =
            (4u - attentionWorkerClusterQkWorkersPerManager_) * 4u;
        return pvBase + kvHead % std::max<uint32_t>(pvCount, 1u);
    }

    void releaseAttentionWorkerClusterVStagingIfComplete(
        AttentionWorkerClusterVEntry& entry) {
        const uint32_t pvCount =
            (4u - attentionWorkerClusterQkWorkersPerManager_) * 4u;
        if (entry.delivered.size() >= pvCount) {
            entry.data.clear();
            entry.data.shrink_to_fit();
        }
    }

    void deliverAttentionWorkerClusterV(
        uint32_t destination, uint64_t source,
        const std::vector<uint8_t>& bytes, uint32_t relayMask = 0) {
        constexpr size_t deliveryChunkBytes = 16u * 1024u;
        for (size_t offset = 0; offset < bytes.size();
             offset += deliveryChunkBytes) {
            const size_t chunkBytes =
                std::min(deliveryChunkBytes, bytes.size() - offset);
            ControlTransportMessage delivery{};
            delivery.kind =
                ControlTransportMessageKind::AttentionClusterPvVDelivery;
            delivery.ownerCore = static_cast<uint32_t>(coreID);
            delivery.workerCore = destination;
            delivery.inputAddr = source;
            delivery.outputAddr = offset;
            delivery.dataNodeMask = relayMask;
            delivery.expectedRows = static_cast<uint32_t>(bytes.size());
            delivery.expectedCols = static_cast<uint32_t>(chunkBytes);
            delivery.payload.resize(chunkBytes / sizeof(float));
            std::memcpy(
                delivery.payload.data(), bytes.data() + offset, chunkBytes);
            delivery.sendCycle = getCurrentSimCycle();
            if (destination == coreID) {
                handleControlTransportMessage(delivery);
            } else {
                globalMem->sendControlMessage(destination, delivery);
            }
        }
    }

    void distributeAttentionWorkerClusterV(
        uint32_t destinationMask, uint64_t source,
        const std::vector<uint8_t>& bytes) {
        if (coreID < 32) {
            destinationMask &=
                ~(uint32_t{1} << static_cast<uint32_t>(coreID));
        }
        std::array<uint32_t, 2> groupMasks = {};
        uint32_t destinationCount = 0;
        for (uint32_t core = 0; core < 32; ++core) {
            if ((destinationMask & (uint32_t{1} << core)) == 0) continue;
            groupMasks[destinationCount++ & 1u] |= uint32_t{1} << core;
        }
        for (uint32_t groupMask : groupMasks) {
            if (groupMask == 0) continue;
            const uint32_t destination =
                static_cast<uint32_t>(__builtin_ctz(groupMask));
            deliverAttentionWorkerClusterV(
                destination, source, bytes,
                groupMask & ~(uint32_t{1} << destination));
        }
    }

    void handleAttentionWorkerClusterVRequest(
        const ControlTransportMessage& message) {
        if (!attentionWorkerClusterVBroadcast_ ||
            message.ownerCore < 4 || message.expectedRows == 0 ||
            message.inputAddr == 0 ||
            coreID != attentionWorkerClusterVLeader(message.kvHeadIndex)) {
            return;
        }
        AttentionWorkerClusterVEntry& entry =
            attentionWorkerClusterVEntries_[message.inputAddr];
        if (entry.bytes != 0 && entry.bytes != message.expectedRows) return;
        entry.bytes = message.expectedRows;
        if (entry.ready) {
            if (entry.delivered.count(message.ownerCore) != 0 ||
                entry.data.empty()) {
                return;
            }
            deliverAttentionWorkerClusterV(
                message.ownerCore, message.inputAddr, entry.data);
            entry.delivered.insert(message.ownerCore);
            releaseAttentionWorkerClusterVStagingIfComplete(entry);
            return;
        }
        entry.subscribers.insert(message.ownerCore);
        if (!entry.loading) {
            entry.loading = true;
            attentionWorkerClusterVLoadQueue_.push_back(message.inputAddr);
        }
    }

    void completeAttentionWorkerClusterVLoad(bool ok) {
        const uint64_t source = attentionWorkerClusterVLoad_.source;
        auto entryIt = attentionWorkerClusterVEntries_.find(source);
        if (entryIt != attentionWorkerClusterVEntries_.end()) {
            AttentionWorkerClusterVEntry& entry = entryIt->second;
            entry.loading = false;
            entry.ready = ok && entry.data.size() == entry.bytes;
            if (entry.ready) {
                output->output(
                    "[ATTENTION_PV_V_BROADCAST] leader=%" PRIu64
                    " source=%" PRIu64 " hbm_bytes=%zu subscribers=%zu\n",
                    coreID, source, entry.bytes, entry.subscribers.size());
                const std::vector<uint32_t> subscribers(
                    entry.subscribers.begin(), entry.subscribers.end());
                entry.subscribers.clear();
                uint32_t remoteMask = 0;
                for (uint32_t subscriber : subscribers) {
                    entry.delivered.insert(subscriber);
                    if (subscriber == coreID) {
                        deliverAttentionWorkerClusterV(
                            subscriber, source, entry.data);
                    } else if (subscriber < 32) {
                        remoteMask |= uint32_t{1} << subscriber;
                    }
                }
                distributeAttentionWorkerClusterV(
                    remoteMask, source, entry.data);
                releaseAttentionWorkerClusterVStagingIfComplete(entry);
            } else {
                entry.subscribers.clear();
                entry.data.clear();
            }
        }
        attentionWorkerClusterVLoad_ = {};
    }

    void progressAttentionWorkerClusterVBroadcast() {
        if (!attentionWorkerClusterVBroadcast_ || globalMem == nullptr) return;
        if (!attentionWorkerClusterVLoad_.active) {
            while (!attentionWorkerClusterVLoadQueue_.empty()) {
                const uint64_t source = attentionWorkerClusterVLoadQueue_.front();
                attentionWorkerClusterVLoadQueue_.pop_front();
                auto it = attentionWorkerClusterVEntries_.find(source);
                if (it == attentionWorkerClusterVEntries_.end() ||
                    !it->second.loading || it->second.bytes == 0) {
                    continue;
                }
                attentionWorkerClusterVLoad_.active = true;
                attentionWorkerClusterVLoad_.source = source;
                attentionWorkerClusterVLoad_.bytes = it->second.bytes;
                attentionWorkerClusterVLoad_.scratch =
                    globalMem->getBaseAddr() + 0x180000;
                it->second.data.assign(it->second.bytes, 0);
                break;
            }
        }
        AttentionWorkerClusterVLoad& load = attentionWorkerClusterVLoad_;
        if (!load.active) return;
        if (!load.dmaIssued) {
            load.dmaIssued = true;
            const uint64_t source = load.source;
            globalMem->dma_read_from_host_to_globalmem(
                load.source, load.bytes, load.scratch,
                [this, source](bool ok) {
                    if (!attentionWorkerClusterVLoad_.active ||
                        attentionWorkerClusterVLoad_.source != source) return;
                    attentionWorkerClusterVLoad_.dmaDone = true;
                    attentionWorkerClusterVLoad_.dmaOk = ok;
                }, DmaRequestKind::AttentionKvPrefetch);
            return;
        }
        if (!load.dmaDone || load.readInFlight) return;
        if (!load.dmaOk) {
            completeAttentionWorkerClusterVLoad(false);
            return;
        }
        if (load.readOffset == load.bytes) {
            completeAttentionWorkerClusterVLoad(true);
            return;
        }
        const size_t chunk = std::min(
            globalMem->localMaxRequestBytes(), load.bytes - load.readOffset);
        if (chunk == 0) {
            completeAttentionWorkerClusterVLoad(false);
            return;
        }
        const uint64_t source = load.source;
        const size_t offset = load.readOffset;
        const uint64_t tag = attentionTransferTag();
        const bool accepted = globalMem->localReadAsync(
            load.scratch + offset, chunk, LocalMemoryClient::Control, tag,
            [this, source, offset, tag](
                bool ok, uint64_t callbackTag,
                const std::vector<uint8_t>& bytes) {
                if (!attentionWorkerClusterVLoad_.active ||
                    attentionWorkerClusterVLoad_.source != source) return;
                attentionWorkerClusterVLoad_.readInFlight = false;
                auto it = attentionWorkerClusterVEntries_.find(source);
                if (!ok || callbackTag != tag ||
                    it == attentionWorkerClusterVEntries_.end() ||
                    offset + bytes.size() > it->second.data.size()) {
                    attentionWorkerClusterVLoad_.dmaOk = false;
                    return;
                }
                std::copy(bytes.begin(), bytes.end(),
                          it->second.data.begin() + offset);
                attentionWorkerClusterVLoad_.readOffset += bytes.size();
            });
        if (accepted) load.readInFlight = true;
    }

    void requestAttentionWorkerClusterV(
        const ControlTransportMessage& next,
        const WorkerTaskListHeader& header) {
        auto& state = attentionWorkerClusterPv_;
        const uint64_t source = header.off_gemm_vec_base;
        const size_t bytes = attentionWorkerClusterVEntryBytes(header);
        const uint64_t generation = ++state.nextPrefetchGeneration;
        state.nextPrefetchAttempted = true;
        state.nextPrefetchSource = source;
        state.nextPrefetchBytesReceived = 0;
        state.nextPrefetchRelayMask = 0;
        state.nextPrefetchHeader = header;
        state.nextPrefetchPayload.clear();
        state.nextPrefetchChunkSeen.clear();
        if (attentionWorkerClusterVResident_.count(source) != 0) {
            state.nextPrefetchPending = false;
            state.nextPrefetchReady = true;
            return;
        }
        state.nextPrefetchPending = true;
        state.nextPrefetchReady = false;
        ControlTransportMessage request{};
        request.kind = ControlTransportMessageKind::AttentionClusterPvVRequest;
        request.ownerCore = static_cast<uint32_t>(coreID);
        request.workerCore = attentionWorkerClusterVLeader(next.kvHeadIndex);
        request.inputAddr = source;
        request.expectedRows = static_cast<uint32_t>(bytes);
        request.kvHeadIndex = next.kvHeadIndex;
        request.sendCycle = getCurrentSimCycle();
        if (request.workerCore == coreID) {
            handleAttentionWorkerClusterVRequest(request);
        } else if (!globalMem->sendControlMessage(request.workerCore, request) &&
                   generation == state.nextPrefetchGeneration) {
            state.nextPrefetchPending = false;
            state.nextPrefetchAttempted = false;
        }
    }

    void handleAttentionWorkerClusterVDelivery(
        const ControlTransportMessage& message) {
        auto& state = attentionWorkerClusterPv_;
        if (!attentionWorkerClusterVBroadcast_ || !state.nextPrefetchPending ||
            state.nextPrefetchSource != message.inputAddr ||
            workerCommandProcessor == nullptr ||
            message.expectedRows == 0 || message.expectedCols == 0 ||
            message.payload.size() * sizeof(float) != message.expectedCols ||
            message.outputAddr + message.expectedCols > message.expectedRows) {
            return;
        }
        constexpr size_t deliveryChunkBytes = 16u * 1024u;
        if (state.nextPrefetchPayload.empty()) {
            state.nextPrefetchPayload.assign(message.expectedRows, 0);
            state.nextPrefetchRelayMask = message.dataNodeMask;
            state.nextPrefetchChunkSeen.assign(
                (message.expectedRows + deliveryChunkBytes - 1) /
                    deliveryChunkBytes,
                0);
        }
        if (state.nextPrefetchPayload.size() != message.expectedRows ||
            state.nextPrefetchRelayMask != message.dataNodeMask ||
            message.outputAddr % deliveryChunkBytes != 0) {
            return;
        }
        const size_t chunkIndex = message.outputAddr / deliveryChunkBytes;
        if (chunkIndex >= state.nextPrefetchChunkSeen.size()) return;
        if (state.nextPrefetchChunkSeen[chunkIndex] == 0) {
            std::memcpy(
                state.nextPrefetchPayload.data() + message.outputAddr,
                message.payload.data(), message.expectedCols);
            state.nextPrefetchChunkSeen[chunkIndex] = 1;
            state.nextPrefetchBytesReceived += message.expectedCols;
        }
        if (state.nextPrefetchBytesReceived != state.nextPrefetchPayload.size()) {
            return;
        }
        distributeAttentionWorkerClusterV(
            state.nextPrefetchRelayMask, message.inputAddr,
            state.nextPrefetchPayload);
        const WorkerTaskListHeader header = state.nextPrefetchHeader;
        const uint64_t generation = state.nextPrefetchGeneration;
        const uint64_t source = message.inputAddr;
        const bool accepted = workerCommandProcessor->installAttentionPvVector(
            header, state.nextPrefetchPayload,
            [this, generation, source](bool ok) {
                auto& callbackState = attentionWorkerClusterPv_;
                if (generation != callbackState.nextPrefetchGeneration ||
                    source != callbackState.nextPrefetchSource) return;
                callbackState.nextPrefetchPending = false;
                callbackState.nextPrefetchReady = ok;
                callbackState.nextPrefetchPayload.clear();
                callbackState.nextPrefetchChunkSeen.clear();
                callbackState.nextPrefetchBytesReceived = 0;
                callbackState.nextPrefetchRelayMask = 0;
                if (ok) attentionWorkerClusterVResident_.insert(source);
                if (!callbackState.busy) startAttentionWorkerClusterPv();
            });
        if (!accepted && generation == state.nextPrefetchGeneration) {
            state.nextPrefetchPending = false;
            state.nextPrefetchAttempted = false;
            state.nextPrefetchPayload.clear();
            state.nextPrefetchChunkSeen.clear();
            state.nextPrefetchBytesReceived = 0;
            state.nextPrefetchRelayMask = 0;
        }
    }

    void prepareAttentionWorkerClusterPvNext() {
        auto& state = attentionWorkerClusterPv_;
        if (attentionWorkerClusterQkWorkersPerManager_ != 2 ||
            !state.busy || attentionWorkerClusterPvQueue_.empty() ||
            workerCommandProcessor == nullptr || state.nextPrefetchAttempted) {
            return;
        }
        const auto& next = attentionWorkerClusterPvQueue_.front();
        const WorkerTaskListHeader header =
            makeAttentionWorkerClusterPvHeader(next);
        if (attentionWorkerClusterVBroadcast_) {
            requestAttentionWorkerClusterV(next, header);
            return;
        }
        const uint64_t generation = ++state.nextPrefetchGeneration;
        state.nextPrefetchAttempted = true;
        state.nextPrefetchPending = true;
        const bool accepted = workerCommandProcessor->prefetchAttentionPvVector(
            header, [this, generation](bool ok) {
                auto& callbackState = attentionWorkerClusterPv_;
                if (generation != callbackState.nextPrefetchGeneration) return;
                callbackState.nextPrefetchPending = false;
                callbackState.nextPrefetchReady = ok;
                if (!callbackState.busy) {
                    startAttentionWorkerClusterPv();
                }
            });
        if (!accepted && generation == state.nextPrefetchGeneration) {
            state.nextPrefetchPending = false;
        }
    }

    void handleAttentionWorkerClusterVPrefetchHint(
        const ControlTransportMessage& message) {
        auto& state = attentionWorkerClusterPv_;
        if (!attentionWorkerClusterVBroadcast_ ||
            attentionWorkerClusterQkWorkersPerManager_ != 2 ||
            workerCommandProcessor == nullptr ||
            state.nextPrefetchAttempted) {
            return;
        }
        const WorkerTaskListHeader header =
            makeAttentionWorkerClusterPvHeader(message);
        requestAttentionWorkerClusterV(message, header);
    }

    void startAttentionWorkerClusterPv() {
        auto& pvState = attentionWorkerClusterPv_;
        if (pvState.busy) {
            prepareAttentionWorkerClusterPvNext();
            return;
        }
        if (attentionWorkerClusterPvQueue_.empty() ||
            workerCommandProcessor == nullptr) return;
        if (pvState.nextPrefetchPending) return;
        if (attentionWorkerClusterVBroadcast_) {
            const auto& next = attentionWorkerClusterPvQueue_.front();
            const WorkerTaskListHeader header =
                makeAttentionWorkerClusterPvHeader(next);
            if (!pvState.nextPrefetchReady ||
                pvState.nextPrefetchSource != header.off_gemm_vec_base) {
                if (pvState.nextPrefetchReady &&
                    pvState.nextPrefetchSource != header.off_gemm_vec_base) {
                    pvState.nextPrefetchAttempted = false;
                    pvState.nextPrefetchReady = false;
                    pvState.nextPrefetchSource = 0;
                }
                if (!pvState.nextPrefetchAttempted) {
                    requestAttentionWorkerClusterV(next, header);
                }
                return;
            }
        }
        pvState.active = std::move(attentionWorkerClusterPvQueue_.front());
        attentionWorkerClusterPvQueue_.pop_front();
        pvState.nextPrefetchAttempted = false;
        pvState.nextPrefetchPending = false;
        pvState.nextPrefetchReady = false;
        pvState.nextPrefetchSource = 0;
        pvState.nextPrefetchBytesReceived = 0;
        pvState.nextPrefetchRelayMask = 0;
        pvState.nextPrefetchHeader = {};
        pvState.nextPrefetchPayload.clear();
        pvState.nextPrefetchChunkSeen.clear();
        const auto& msg = pvState.active;
        constexpr uint32_t block = 64;
        constexpr size_t panelFloats = block * block;
        constexpr size_t panelBytes = panelFloats * sizeof(float);
        if (msg.payload.size() != 4 * panelFloats || msg.scales.size() != block ||
            msg.expectedRows != block || msg.expectedCols != 128 ||
            msg.kvTileRows >= 4) {
            output->output("Attention worker-cluster PV invalid message core=%" PRIu64 "\n", coreID);
            return;
        }
        pvState.busy = true;
        pvState.startCycle = getCurrentSimCycle();
        pvState.fusionTiles = 0;
        std::vector<uint8_t> packed(6 * panelBytes, 0);
        std::memcpy(packed.data(), msg.payload.data(), 2 * panelBytes);
        std::memcpy(packed.data() + 4 * panelBytes,
                    msg.payload.data() + 2 * panelFloats, 2 * panelBytes);
        WorkerTaskListHeader header = makeAttentionWorkerClusterPvHeader(msg);
        const bool started = workerCommandProcessor->setWindowCallbacks(
                [this](uint64_t taskId, uint64_t, const std::vector<uint8_t>& tile) {
                    auto& state = attentionWorkerClusterPv_;
                    if (!state.busy || taskId >= 2 || tile.size() != 64u * 64u * sizeof(float))
                        return false;
                    const AttentionWorkerClusterPvOutputKey outputKey =
                        std::make_tuple(
                            state.active.jobId, state.active.tag,
                            state.active.ownerCore,
                            state.active.groupQueryRowBegin);
                    auto outputIt = attentionWorkerClusterPvOutputs_.find(outputKey);
                    if (outputIt == attentionWorkerClusterPvOutputs_.end() ||
                        outputIt->second.size() != 64u * 128u) return false;
                    std::vector<float>& rowOutput = outputIt->second;
                    for (uint32_t col = 0; col < 64; ++col) {
                        for (uint32_t row = 0; row < 64; ++row) {
                            float value = 0.0f;
                            std::memcpy(&value, tile.data() +
                                (static_cast<size_t>(col) * 64 + row) * sizeof(float), sizeof(float));
                            rowOutput[static_cast<size_t>(row) * 128 +
                                taskId * 64 + col] += value;
                        }
                    }
                    ++state.fusionTiles;
                    return true;
                },
                [this](uint64_t cycles, uint64_t start, uint64_t end) {
                    auto& state = attentionWorkerClusterPv_;
                    if (!state.busy || state.fusionTiles != 2) return;
                    output->output("[ATTENTION_WORKER_CLUSTER_PV] core=%" PRIu64
                        " qk_core=%u row=%u window=%u cycles=%" PRIu64
                        " start=%" PRIu64 " end=%" PRIu64 "\n",
                        coreID, state.active.ownerCore, state.active.groupQueryRowBegin,
                        state.active.kvTileRows, cycles, start, end);
                    if (state.active.kvTileRows == 3) {
                        const AttentionWorkerClusterPvOutputKey outputKey =
                            std::make_tuple(
                                state.active.jobId, state.active.tag,
                                state.active.ownerCore,
                                state.active.groupQueryRowBegin);
                        ControlTransportMessage done = state.active;
                        done.kind = ControlTransportMessageKind::AttentionClusterPvComplete;
                        done.workerCore = static_cast<uint32_t>(coreID);
                        done.value = 1.0;
                        done.payload.clear(); done.scales.clear();
                        done.sendCycle = getCurrentSimCycle();
                        globalMem->sendControlMessage(done.ownerCore, done);
                        attentionWorkerClusterPvOutputs_.erase(outputKey);
                    }
                    state.busy = false;
                    startAttentionWorkerClusterPv();
                }) &&
            workerCommandProcessor->setWindowResidentMatrixPayload(
                std::move(packed)) &&
            workerCommandProcessor->startWindow(header);
        if (!started) {
            pvState.busy = false;
            attentionWorkerClusterPvQueue_.push_front(std::move(pvState.active));
            return;
        }
        const AttentionWorkerClusterPvOutputKey outputKey = std::make_tuple(
            msg.jobId, msg.tag, msg.ownerCore, msg.groupQueryRowBegin);
        std::vector<float>& rowOutput =
            attentionWorkerClusterPvOutputs_[outputKey];
        if (msg.kvTileRows == 0 || rowOutput.size() != block * 128) {
            rowOutput.assign(block * 128, 0.0f);
        } else {
            for (uint32_t row = 0; row < block; ++row) {
                for (uint32_t dim = 0; dim < 128; ++dim) {
                    rowOutput[static_cast<size_t>(row) * 128 + dim] *=
                        msg.scales[row];
                }
            }
        }
        prepareAttentionWorkerClusterPvNext();
    }

    AttentionWorkerState* findAttentionWorkerClusterState(
        uint64_t jobId, uint64_t tag) {
        if (attentionWorker_ && attentionWorker_->dispatch.jobId == jobId &&
            attentionWorker_->dispatch.tag == tag) {
            return attentionWorker_.get();
        }
        if (attentionWorkerClusterQkAhead_ &&
            attentionWorkerClusterQkAhead_->dispatch.jobId == jobId &&
            attentionWorkerClusterQkAhead_->dispatch.tag == tag) {
            return attentionWorkerClusterQkAhead_.get();
        }
        for (auto& draining : attentionWorkerClusterPvDraining_) {
            if (draining->dispatch.jobId == jobId &&
                draining->dispatch.tag == tag) {
                return draining.get();
            }
        }
        return nullptr;
    }

    void handleAttentionWorkerClusterPvAllocationRequest(
        const ControlTransportMessage& message) {
        if (!attentionWorkerClusterDynamicPv_ || coreID != 0) return;
        if (message.ownerCore < 4u || message.ownerCore >= 12u) return;
        const uint32_t pvBase = 4u +
            attentionWorkerClusterQkWorkersPerManager_ * 4u;
        const uint32_t pvCount =
            (4u - attentionWorkerClusterQkWorkersPerManager_) * 4u;
        if (pvCount == 0 || pvCount > attentionWorkerClusterPvAssignedRows_.size())
            return;
        const uint32_t manager = (message.ownerCore - 4u) % 4u;
        uint32_t selected = manager;
        if ((attentionWorkerClusterPvAssignCursor_++ & 1u) != 0) {
            selected += 4u;
        }
        const uint32_t alternate = selected == manager ? manager + 4u : manager;
        if (alternate < pvCount &&
            attentionWorkerClusterPvAssignedRows_[alternate] <
                attentionWorkerClusterPvAssignedRows_[selected]) {
            selected = alternate;
        }
        ++attentionWorkerClusterPvAssignedRows_[selected];

        ControlTransportMessage response = message;
        response.kind =
            ControlTransportMessageKind::AttentionClusterPvAllocationResponse;
        response.workerCore = pvBase + selected;
        response.sendCycle = getCurrentSimCycle();
        globalMem->sendControlMessage(message.ownerCore, response);
    }

    void handleAttentionWorkerClusterPvAllocationResponse(
        const ControlTransportMessage& message) {
        if (!attentionWorkerClusterDynamicPv_ || message.ownerCore != coreID)
            return;
        AttentionWorkerState* state =
            findAttentionWorkerClusterState(message.jobId, message.tag);
        if (state == nullptr || message.row >= state->reuseWindowPvRowCore.size() ||
            state->reuseWindowPvAllocationPending[message.row] == 0) {
            return;
        }
        state->reuseWindowPvAllocationPending[message.row] = 0;
        state->reuseWindowPvRowCore[message.row] =
            static_cast<int32_t>(message.workerCore);
        for (uint32_t window = 0; window < 4; ++window) {
            sendAttentionWorkerClusterVHint(
                *state, message.row, window, message.workerCore);
        }
        output->output(
            "[ATTENTION_DYNAMIC_PV_ASSIGN] qk_core=%" PRIu64
            " job=%" PRIu64 " row=%u pv_core=%u\n",
            coreID, message.jobId, message.row * 64, message.workerCore);
        dispatchAttentionReuseWindowPvAggregates(*state);
    }

    void handleAttentionWorkerClusterPvAllocationRelease(
        const ControlTransportMessage& message) {
        if (!attentionWorkerClusterDynamicPv_ || coreID != 0) return;
        const uint32_t pvBase = 4u +
            attentionWorkerClusterQkWorkersPerManager_ * 4u;
        const uint32_t pvCount =
            (4u - attentionWorkerClusterQkWorkersPerManager_) * 4u;
        if (message.workerCore < pvBase ||
            message.workerCore >= pvBase + pvCount) return;
        const uint32_t index = message.workerCore - pvBase;
        if (attentionWorkerClusterPvAssignedRows_[index] != 0) {
            --attentionWorkerClusterPvAssignedRows_[index];
        }
    }

    void handleControlTransportMessage(const ControlTransportMessage& message) {
        if (message.kind ==
                ControlTransportMessageKind::AttentionClusterPvAllocationRequest) {
            handleAttentionWorkerClusterPvAllocationRequest(message);
            return;
        }
        if (message.kind ==
                ControlTransportMessageKind::AttentionClusterPvAllocationResponse) {
            handleAttentionWorkerClusterPvAllocationResponse(message);
            return;
        }
        if (message.kind ==
                ControlTransportMessageKind::AttentionClusterPvAllocationRelease) {
            handleAttentionWorkerClusterPvAllocationRelease(message);
            return;
        }
        if (message.kind ==
                ControlTransportMessageKind::AttentionClusterPvVPrefetchHint) {
            handleAttentionWorkerClusterVPrefetchHint(message);
            return;
        }
        if (message.kind == ControlTransportMessageKind::AttentionClusterPvVRequest) {
            handleAttentionWorkerClusterVRequest(message);
            return;
        }
        if (message.kind == ControlTransportMessageKind::AttentionClusterPvVDelivery) {
            handleAttentionWorkerClusterVDelivery(message);
            return;
        }
        if (message.kind == ControlTransportMessageKind::AttentionClusterPvDispatch) {
            attentionWorkerClusterPvQueue_.push_back(message);
            startAttentionWorkerClusterPv();
            return;
        }
        if (message.kind == ControlTransportMessageKind::AttentionClusterPvComplete) {
            if (message.ownerCore == coreID && message.value == 1.0) {
                if (attentionWorkerClusterDynamicPv_) {
                    ControlTransportMessage release = message;
                    release.kind = ControlTransportMessageKind::
                        AttentionClusterPvAllocationRelease;
                    release.ownerCore = static_cast<uint32_t>(coreID);
                    release.sendCycle = getCurrentSimCycle();
                    globalMem->sendControlMessage(0, release);
                }
                AttentionWorkerState* target = nullptr;
                if (attentionWorker_ &&
                    attentionWorker_->dispatch.jobId == message.jobId &&
                    attentionWorker_->dispatch.tag == message.tag) {
                    target = attentionWorker_.get();
                }
                if (target != nullptr) {
                    ++target->reuseWindowClusterPvCompleted;
                    const uint32_t rowBlocks = target->dispatch.expectedRows / 64;
                    if (target->reuseWindowClusterPvCompleted == rowBlocks &&
                        target->reuseWindowRowContextsCompleted ==
                            target->reuseWindowRowContexts.size()) {
                        startAttentionReuseWindowSoftmax();
                    }
                    return;
                }
                for (size_t index = 0;
                     index < attentionWorkerClusterPvDraining_.size(); ++index) {
                    AttentionWorkerState& draining =
                        *attentionWorkerClusterPvDraining_[index];
                    if (draining.dispatch.jobId != message.jobId ||
                        draining.dispatch.tag != message.tag) continue;
                    ++draining.reuseWindowClusterPvCompleted;
                    const uint32_t rowBlocks = draining.dispatch.expectedRows / 64;
                    if (draining.reuseWindowClusterPvCompleted == rowBlocks) {
                        completeAttentionWorkerClusterDraining(index);
                    }
                    return;
                }
            }
            return;
        }
        if (message.kind == ControlTransportMessageKind::AttentionDispatch) {
            startAttentionWorker(message);
            return;
        }
        if (handleManagerAttentionCompletion(message)) {
            return;
        }
        if (handleAttentionManagerBandCompletion(message)) {
            return;
        }
        if (handleManagerTensorCompletion(message)) {
            return;
        }
        if (sfu != nullptr) {
            sfu->receiveControlMessage(message);
        }
    }

    bool isArrayComputeInflight(uint32_t array_id) const {
        if (array_id >= inflight_compute_cmds.size()) {
            return false;
        }
        return inflight_compute_cmds[array_id].cmd != nullptr;
    }

    void issueArrayCompute(SST::Vanadis::RoCCCommand* cmd, uint32_t array_id, uint64_t cycle) {
        if (cmd == nullptr || cmd->inst == nullptr) {
            return;
        }
        if (array_id >= inflight_compute_cmds.size()) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return;
        }

        auto& inflight = inflight_compute_cmds[array_id];
        if (inflight.cmd != nullptr) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 1, cmd->cmd_id, cmd->hw_thread));
            delete cmd;
            return;
        }

        inflight.cmd = cmd;
        inflight.start_cycle = cycle;
        inflight.async_mode = (cmd->inst->rd == 0);
        if (inflight.async_mode) {
            auto& async_state = async_compute_states[array_id];
            async_state.submitted = true;
            async_state.completed = false;
            async_state.rd_val = 0;
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
        }
        arrayStates[array_id] = 1;
        array->beginComputation(array_id);
    }

    void completeArrayCompute(uint32_t array_id, uint64_t rd_val) {
        if (array_id >= inflight_compute_cmds.size()) {
            return;
        }

        auto& inflight = inflight_compute_cmds[array_id];
        if (inflight.cmd == nullptr || inflight.cmd->inst == nullptr) {
            output->verbose(CALL_INFO, 0, 0,
                "[RoCC ERROR] array completion without inflight compute array=%" PRIu32 "\n",
                array_id);
            return;
        }

        const uint64_t cycles_spent = (LastTickCycle >= inflight.start_cycle)
            ? (LastTickCycle - inflight.start_cycle + 1)
            : 0;
        stat_cycles_mvm->addData(cycles_spent);
        mvm_ops_completed++;
        maybeReportMvmProgress(false);

        output->verbose(CALL_INFO, 1, 0,
            "Finalize RoCC compute command array=%" PRIu32 " rd=%" PRIu16 " cmd_id=%" PRIu64 " rd_val=%" PRIu64 "\n",
            array_id,
            inflight.cmd->inst->rd,
            inflight.cmd->cmd_id,
            rd_val);

        if (inflight.async_mode) {
            auto& async_state = async_compute_states[array_id];
            async_state.submitted = true;
            async_state.completed = true;
            async_state.rd_val = rd_val;
        } else {
            enqueueResponse(new SST::Vanadis::RoCCResponse(
                inflight.cmd->inst->rd,
                rd_val,
                inflight.cmd->cmd_id,
                inflight.cmd->hw_thread));
        }

        delete inflight.cmd;
        inflight.cmd = nullptr;
        inflight.start_cycle = 0;
        inflight.async_mode = false;
    }
  
    T decodeOperand(const uint8_t* raw) const {
        T value{};
        if constexpr (std::is_same<T, float>::value) {
            if (inputOperandSize == 2) {
                uint16_t bits = 0;
                std::memcpy(&bits, raw, sizeof(bits));
                return static_cast<T>(SST::Golem::golem_fp16_to_float(bits));
            }
        }
        const size_t copyBytes = std::min<size_t>(inputOperandSize, sizeof(T));
        std::memcpy(&value, raw, copyBytes);
        return value;
    }

    void encodeOperand(T value, uint8_t* dst, size_t bytes) const {
        std::memset(dst, 0, bytes);
        if constexpr (std::is_same<T, float>::value) {
            if (bytes == 2) {
                const uint16_t bits = SST::Golem::golem_float_to_fp16(value);
                std::memcpy(dst, &bits, sizeof(bits));
                return;
            }
        }
        std::memcpy(dst, &value, std::min<size_t>(bytes, sizeof(T)));
    }

    void storeVector() {
        uint64_t rs1 = curr_cmd->rs1; // Destination address (physical)
        uint64_t rs2 = curr_cmd->rs2; // Array ID or source vector index
        vector_total_size = arrayOutputSize * outputOperandSize;
        uint64_t cache_line_size = memInterface->getLineSize();

        write_offset = 0;
        uint64_t physAddr = rs1; // Assuming rs1 is physical address
        uint64_t addr_offset = physAddr % cache_line_size;

        // Resize the output payload to hold the entire vector
        outputPayload.resize(vector_total_size);

        // Reference to the output vector we need to store
        auto& outputVector = *static_cast<std::vector<T>*>(array->getOutputVector(rs2));

        // Fill the output payload with the vector data
        for (size_t i = 0; i < static_cast<size_t>(arrayOutputSize); i++) {
            T value = outputVector[i];
            encodeOperand(value, &outputPayload[i * outputOperandSize], outputOperandSize);
        }

        // Optional: Output the stored array for debugging purposes
        output->verbose(CALL_INFO, 9, 0, "Stored array %" PRIu64 ":\n", rs2);
        for (size_t i = 0; i < static_cast<size_t>(arrayOutputSize); i++) {
            if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
                output->verbose(CALL_INFO, 9, 0, "%f ", static_cast<double>(outputVector[i]));
            } else {
                output->verbose(CALL_INFO, 9, 0, "%lld ", static_cast<long long>(outputVector[i]));
            }
        }
        output->verbose(CALL_INFO, 9, 0, "\n\n");

        // Calculate the size of the first memory request
        uint32_t request_size = static_cast<uint32_t>(std::min(
            cache_line_size - addr_offset, 
            vector_total_size - write_offset
        ));

        // Prepare the first chunk of data to write
        std::vector<uint8_t> data_chunk(
            outputPayload.begin() + write_offset,
            outputPayload.begin() + write_offset + request_size
        );

        // Create a new write request to send to the memory interface
        auto* store_req = new StandardMem::Write(physAddr + write_offset, request_size, data_chunk, false, 0, rs1, 0, 0);

        // Send the write request
        memInterface->send(store_req);

        // Update the write offset for subsequent writes
        write_offset += request_size;
    }
    
    void moveVector() {
        uint64_t rs1 = curr_cmd->rs1;//RoCC指令传递的源阵列ID
        uint64_t rs2 = curr_cmd->rs2;//RoCC指令传递的目标阵列ID
        //调用array的moveOutputToInput方法，把ID为rs1阵列的输出向量作为ID为rs2阵列的输入向量
        array->moveOutputToInput(rs1, rs2);

        //拿到“目标阵列rs2的输入向量”对象（即move之后的)
        auto& inputVector = *static_cast<std::vector<T>*>(array->getInputVector(rs2));

        output->verbose(CALL_INFO, 9, 0,
                      "Moved array %" PRIu64 " to array %" PRIu64 ". Array %" PRIu64 ":\n", rs1, rs2, rs2);

        for (int i = 0; i < arrayInputSize; i++) {
            if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
                output->verbose(CALL_INFO, 9, 0, "%f ", static_cast<double>(inputVector[i]));
            } else {
                output->verbose(CALL_INFO, 9, 0, "%ld ", static_cast<long>(inputVector[i]));
            }
        }
        output->verbose(CALL_INFO, 9, 0, "\n");

        completeRoCC(0);
    }

    void ConfigureOutputMode() {
        uint64_t command = curr_cmd->rs1; // 输出模式命令
        uint64_t array_id = curr_cmd->rs2; // 阵列编号

        if (array_id >= static_cast<uint64_t>(numArrays)) {
            output->verbose(CALL_INFO, 0, 0,
                            "mvm.ocfg: invalid array id %" PRIu64 " (numArrays=%d)\n",
                            array_id, numArrays);
            completeRoCC(1);
            return;
        }

        array->configureOutputMode(static_cast<uint32_t>(array_id), command);
        completeRoCC(0);
    }

    void OutputvectorStore(uint64_t) {
        if (curr_cmd == nullptr || curr_cmd->rs2 >= static_cast<uint64_t>(numArrays)) {
            completeRoCC(1);
            return;
        }
        if (!legacy_output_store_.active) {
            legacy_output_store_ = LegacyOutputStoreState{};
            legacy_output_store_.active = true;
            legacy_output_store_.command_id = curr_cmd->cmd_id;
            legacy_output_store_.dest_addr = curr_cmd->rs1;
            legacy_output_store_.array_id = static_cast<uint32_t>(curr_cmd->rs2);
        }
        progressLegacyOutputStore();
    }

    void IntputvectorLoad(uint64_t) {
        beginBlockingArrayLoad(false);
    }

    void InputMatrixLoad(uint64_t) {
        beginBlockingArrayLoad(true);
    }
    void SetRemoteLength() {
        size_t fallback = defaultRemoteLength();
        size_t requested = static_cast<size_t>(curr_cmd->rs1);
        if (requested == 0) {
            requested = fallback;
        }
        if (requested == 0) {
            requested = 1; // 确保非零长度
        }

        remoteTransferLength = requested;
        output->verbose(CALL_INFO, 9, 0,
                        "Remote transfer length updated to %zu bytes (rs1=0x%" PRIx64 ")\n",
                        remoteTransferLength, curr_cmd->rs1);
        completeRoCC(0);
    }

    void RemoteStore(uint64_t cycle) {
        uint64_t cycles_elapsed = (cycle >= StartTickCycle) ? (cycle - StartTickCycle + 1) : 0;

        if (cycles_elapsed < latency_remote_st) {
            return;
        }

        output->verbose(CALL_INFO, 9, 0,
                        "RemoteStore: Executing after %" PRIu64 " cycles (at cycle %" PRIu64 ")\n",
                        cycles_elapsed, cycle);
        uint64_t local_addr  = curr_cmd->rs1;  // 本地 GlobalMemory 源地址
        uint64_t remote_addr = curr_cmd->rs2;  // 远端 GlobalMemory 目标地址

    // 传输字节数：以 mvm.slen 设置的 remoteTransferLength 为准；
    // 早期实现曾用 rd 寄存器号作为长度（会被编译器分配为如 x15），导致固定为 15 等错误值。
    // 这里不再使用 rd 覆盖，统一按照配置长度来传输。
    uint16_t rd_reg_index = curr_cmd->inst->rd; // 仅用于调试观测（指令目的寄存器号）
    size_t length = resolveRemoteLength();

        std::vector<uint8_t> data(length);
        globalMem->rd_from_globalmem(local_addr, length, data);   // 先读本地
        globalMem->wr_to_network(remote_addr, length, data);      // 再经 NoC 写对端
        completeRoCC(0);  // 发送后即可返回（若需 ACK，可扩展为等待网络回包）
    }

    void RemoteStoreWait(uint64_t cycle) {
        const uint64_t cycles_elapsed =
            (cycle >= StartTickCycle) ? (cycle - StartTickCycle + 1) : 0;
        if (remoteStoreCompletionToken == 0) {
            if (cycles_elapsed < latency_remote_st) {
                return;
            }
            const uint64_t local_addr = curr_cmd->rs1;
            const uint64_t host_addr = curr_cmd->rs2;
            const size_t length = resolveRemoteLength();
            std::vector<uint8_t> data(length);
            globalMem->rd_from_globalmem(local_addr, length, data);
            remoteStoreCompletionToken =
                globalMem->dma_write_to_host_async(host_addr, length, data);
        }
        if (!globalMem->dma_completion_done(remoteStoreCompletionToken)) {
            return;
        }
        globalMem->dma_completion_retire(remoteStoreCompletionToken);
        remoteStoreCompletionToken = 0;
        completeRoCC(0);
    }

    void RemoteLoad(uint64_t cycle) {
        uint64_t cycles_elapsed = (cycle >= StartTickCycle) ? (cycle - StartTickCycle + 1) : 0;

        if (cycles_elapsed < latency_remote_ld) {
            return;
        }
        uint64_t remote_addr = curr_cmd->rs1; // 远端 GlobalMemory 源地址
        uint64_t local_addr  = curr_cmd->rs2; // 本地 GlobalMemory 目标地址
        (void)local_addr; // 当前实现中不直接在此函数写入，读回后由 GlobalMemory 统一存放

    // 传输字节数：统一以 mvm.slen 设置的 remoteTransferLength 为准
    uint16_t rd_reg_index = curr_cmd->inst->rd; // 仅用于调试观测（指令目的寄存器号）
    size_t length = resolveRemoteLength();

        output->verbose(CALL_INFO, 1, 2,
            "RemoteLoad issued (GM base 0x%" PRIx64 "): remote_addr=0x%" PRIx64 " local_addr=0x%" PRIx64 " length=%zu rd_reg=%u\n",
            globalMem ? globalMem->getBaseAddr() : 0, remote_addr, local_addr, length, rd_reg_index);

        // 发起网络读请求；数据返回后由 GlobalMemory 的回调统一写入本地存储
        globalMem->rd_to_network(remote_addr, length, local_addr);
        completeRoCC(0);
    }


    void MainMem2GlobalMem() {
        uint64_t rs1 = curr_cmd->rs1; // 主存源物理地址 (Main Memory Source)
        uint64_t rs2 = curr_cmd->rs2; // GlobalMemory 目标地址 (Destination)
        
        // 初始化状态
        gm_write_dst_addr = rs2;
        gm_write_offset = 0;
        
        // 确定传输长度：优先使用 mvm.slen 设置的长度，或者使用默认向量长度
        gm_write_total_size = resolveRemoteLength();

        // 定义一个新的 Flag，用于在回调中识别这是 write_gm 的数据
        // 0x0 是 Matrix, 0x1 是 Vector, 我们用 0x2
        uint32_t load_gm_flag = 0x10; 
        
        uint64_t cache_line_size = memInterface->getLineSize();
        
        // 计算地址对齐和第一次请求的大小
        uint64_t physAddr = rs1; 
        uint64_t addr_offset = physAddr % cache_line_size;
        
        uint32_t request_size = std::min(static_cast<uint64_t>(cache_line_size - addr_offset), gm_write_total_size);

        output->verbose(CALL_INFO, 1, 0, 
            "write_gm Start: MainMem Addr: 0x%" PRIx64 ", GlobalMem Addr: 0x%" PRIx64 ", Size: %" PRIu64 "\n",
            physAddr, gm_write_dst_addr, gm_write_total_size);

        // 发起读取请求 (Read Request)
        auto *load_req = new StandardMem::Read(physAddr, request_size, load_gm_flag);
        memInterface->send(load_req);
    }

    // 实现 GlobalMem2MainMem (GM -> MainMem)
    void GlobalMem2MainMem() {
        uint64_t mm_dst_addr = curr_cmd->rs1; // Main Memory 目标地址
        uint64_t gm_src_addr = curr_cmd->rs2; // Global Memory 源地址
        
        // 1. 确定传输长度 (使用 mvm.slen 设置的长度)
        // 复用 vector_total_size 变量，因为它在 WriteResp 中被用于检查结束条件
        vector_total_size = resolveRemoteLength();
        write_offset = 0;

        uint64_t cache_line_size = memInterface->getLineSize();
        uint64_t physAddr = mm_dst_addr; 
        uint64_t addr_offset = physAddr % cache_line_size;

        output->verbose(CALL_INFO, 1, 0, 
            "write_mm Start: GlobalMem Addr: 0x%" PRIx64 ", MainMem Addr: 0x%" PRIx64 ", Size: %" PRIu64 "\n",
            gm_src_addr, mm_dst_addr, vector_total_size);

        // 2. 从 GlobalMemory 读取数据到本地 buffer (outputPayload)
        outputPayload.resize(vector_total_size);
        globalMem->rd_from_globalmem(gm_src_addr, vector_total_size, outputPayload);

        // [优化打印] 打印从 GM 读出的全部数据
        if (!outputPayload.empty()) {
            std::string hex_str;
            char buf[8];
            for (size_t i = 0; i < outputPayload.size(); ++i) {
                snprintf(buf, sizeof(buf), "%02X ", outputPayload[i]);
                hex_str += buf;
            }
            output->verbose(CALL_INFO, 1, 0, "Data Write to MainMemory\n");
            output->verbose(CALL_INFO, 10, 0, "Data: %s\n", hex_str.c_str());
        } else {
            output->verbose(CALL_INFO, 1, 0, "[EMPTY]\n");
        }

        // 3. 发起第一个主存写入请求 (DMA Write)
        uint32_t request_size = static_cast<uint32_t>(std::min(
            cache_line_size - addr_offset, 
            vector_total_size - write_offset
        ));

        std::vector<uint8_t> data_chunk(
            outputPayload.begin() + write_offset,
            outputPayload.begin() + write_offset + request_size
        );

        // 构造写请求
        // 注意：storeVector 使用的是 rs1 作为目标，而 write_mm 使用的是 rs2
        // 这里我们在创建请求时传入物理地址，回调中会根据 func7 区分计算下一个地址
        auto* store_req = new StandardMem::Write(physAddr + write_offset, request_size, data_chunk, false, 0, physAddr, 0, 0);

        memInterface->send(store_req);
        write_offset += request_size;
    }

    // reg2gm: 将 rs1 寄存器的值直接写入到 GlobalMemory 地址 rs2
    void Reg2GlobalMem() {
        uint64_t val = curr_cmd->rs1;       // 数据 (来自 Vanadis 寄存器)
        uint64_t dst_addr = curr_cmd->rs2;  // 目标 GlobalMemory 地址

        // 准备数据包：Vanadis 寄存器是 64 位的，所以写入 8 字节
        size_t data_size = sizeof(uint64_t);
        std::vector<uint8_t> payload(data_size);
        memcpy(payload.data(), &val, data_size);

        output->verbose(CALL_INFO, 1, 0, 
            "reg2gm: Writing register value 0x%" PRIx64 " to GlobalMemory Address 0x%" PRIx64 "\n", 
            val, dst_addr);

        globalMem->wr_to_globalmem(dst_addr, data_size, payload);
        completeRoCC(0);
    }


    // gm2reg: 从 GlobalMemory 地址 rs1 读取数据，写入到目标寄存器 rd
    void GlobalMem2Reg() {
        uint64_t src_addr = curr_cmd->rs1;  // 源 GlobalMemory 地址

        // 准备读取缓冲区
        size_t data_size = sizeof(uint64_t);
        std::vector<uint8_t> read_buffer;
        read_buffer.resize(data_size); 

        // 调用 GlobalMemory 读接口
        globalMem->rd_from_globalmem(src_addr, data_size, read_buffer);
        uint64_t val = 0;
        memcpy(&val, read_buffer.data(), data_size);

        output->verbose(CALL_INFO, 1, 0, 
            "gm2reg: Read value 0x%" PRIx64 " from GlobalMemory Address 0x%" PRIx64 "\n", 
            val, src_addr);

        // 完成指令，并将读取到的值作为结果返回
        completeRoCC(val);
    }


    
    //本质上，它是每条RoCC指令生命周期的结束收尾工作
    void completeRoCC(uint64_t rd_val) {
        uint64_t cycles_spent = (LastTickCycle >= StartTickCycle) ? (LastTickCycle - StartTickCycle + 1) : 0;
        if (curr_cmd == nullptr || curr_cmd->inst == nullptr) {
            output->verbose(CALL_INFO, 0, 0, "[RoCC ERROR] completeRoCC with null command\n");
            busy = false;
            return;
        }
        if (curr_cmd != nullptr) {
            switch (curr_cmd->inst->func7) {
                case 0x1: stat_cycles_mvm_set->addData(cycles_spent); break;
                case 0x2: stat_cycles_mvm_l->addData(cycles_spent); break;
                case 0x3: stat_cycles_mvm->addData(cycles_spent); break;
                case 0x4: stat_cycles_mvm_s->addData(cycles_spent); break;
                case 0x5: stat_cycles_mvm_mv->addData(cycles_spent); break;
                case 0x6: stat_cycles_mvm_ovec2gm->addData(cycles_spent); break;
                case 0x7: stat_cycles_mvm_gm2ivec->addData(cycles_spent); break;
                case 0x8: stat_cycles_mvm_gm2imat->addData(cycles_spent); break;
                case 0x9:
                case GOLEM_ROCC_FUNC7_REMOTE_STORE_WAIT:
                    stat_cycles_remote_st->addData(cycles_spent);
                    break;
                case 0xA: stat_cycles_remote_ld->addData(cycles_spent); break;
                default: break;
            }
            if (curr_cmd->inst->func7 == 0x3) {
                mvm_ops_completed++;
                maybeReportMvmProgress(false);
            }
        }
        output->verbose(CALL_INFO, 1, 0,
            "Finalize RoCC command w/ func7=0x%02" PRIx8 " rd=%" PRIu16 " xd=%u xs1=%u xs2=%u rs1=0x%" PRIx64 " rs2=0x%" PRIx64 " rd_val=%" PRIu64 "\n",
            curr_cmd->inst->func7,
            curr_cmd->inst->rd,
            static_cast<unsigned>(curr_cmd->inst->xd),
            static_cast<unsigned>(curr_cmd->inst->xs1),
            static_cast<unsigned>(curr_cmd->inst->xs2),
            curr_cmd->rs1,
            curr_cmd->rs2,
            rd_val
        );

        busy = false;
        enqueueResponse(new SST::Vanadis::RoCCResponse(curr_cmd->inst->rd, rd_val, curr_cmd->cmd_id, curr_cmd->hw_thread));
        delete curr_cmd;
        curr_cmd = nullptr;
    }

    void maybeReportMvmProgress(bool force) {
        if (!progress_heartbeat || progress_total_mvm_ops == 0) {
            return;
        }
        if (!force && LastTickCycle < progress_next_cycle && mvm_ops_completed < progress_total_mvm_ops) {
            return;
        }

        uint64_t completed = mvm_ops_completed;
        if (completed > progress_total_mvm_ops) {
            completed = progress_total_mvm_ops;
        }
        const uint64_t pct = (completed * 100) / progress_total_mvm_ops;
        if (force || pct != progress_last_percent) {
            output->output("RoCC core=%" PRIu64 " MVM_PROGRESS: completed=%" PRIu64 "/%" PRIu64
                           " (%" PRIu64 "%%) cycle=%" PRIu64 "\n",
                           coreID,
                           completed,
                           progress_total_mvm_ops,
                           pct,
                           LastTickCycle);
            progress_last_percent = pct;
        }

        while (progress_next_cycle <= LastTickCycle) {
            progress_next_cycle += progress_interval_cycles;
        }
    }

    void recordWcpArrayCompletion(uint32_t array_id) {
        if (array_id < arrayStates.size()) {
            arrayStates[array_id] = 0;
        }
        mvm_ops_completed++;
        maybeReportMvmProgress(false);
    }

    //在阵列（Array）计算完成后，被SST模拟框架自动调用的,标记这个阵列空闲
    void handleArrayEvent(Event *ev) {
        Golem::ArrayEvent *aev = static_cast<Golem::ArrayEvent *>(ev);
        uint32_t arrayID = aev->getArrayID();
        if (workerCommandProcessor != nullptr && workerCommandProcessor->handleArrayDone(arrayID, LastTickCycle)) {
            recordWcpArrayCompletion(arrayID);
            delete ev;
            return;
        }
        if (handleAttentionArrayDone(arrayID)) {
            delete ev;
            return;
        }
        if (arrayID >= arrayStates.size()) {
            delete ev;
            return;
        }
        arrayStates[arrayID] = 0;
        completeArrayCompute(arrayID, 0);
        delete ev;
    }
  
    class StandardMemHandlers : public Interfaces::StandardMem::RequestHandler {
    public:
        StandardMemHandlers(RoCCAnalog *rocc, SST::Output *output)
            : Interfaces::StandardMem::RequestHandler(output), rocc(rocc) {}
  
        virtual ~StandardMemHandlers() {}
  
        virtual void handle(StandardMem::ReadResp *ev) {
            out->verbose(CALL_INFO, 9, 0,
                     "-> handle read-response (virt-addr: 0x%" PRI_ADDR ")\n", ev->vAddr);
            const uint32_t flags = ev->getAllFlags();

            SST::Vanadis::RoCCCommand *rocc_cmd = rocc->curr_cmd;
  
            if (ev->getFail()) {
                out->verbose(CALL_INFO, 9, 0, "RoCC load failed\n");
                rocc->completeRoCC(1);
                delete ev;
                return;
            }
            //阵列ID（或向量/矩阵的编号），直接从指令的rs2字段获取，用于确定数据写到哪个计算阵列
            int32_t array_id = rocc_cmd->rs2;  // Array ID is in rs2
            switch (flags) {
                //处理了一个矩阵的读取操作。它从内存中读取矩阵数据，并将这些数据存储到计算阵列中
                case GOLEM_ROCC_FLAG_SYNC_MATRIX: // Read response data is matrix to be set
                {
                    rocc->output->verbose(CALL_INFO, 9, 0,
                                "Set matrix read response detected\n");
  
                    size_t payload_size = ev->size;  //payload_size 获取响应数据的大小
                    unsigned char *payload_data = ev->data.data(); //ev->data.data()返回一个字节数组，包含了实际的矩阵数据

                    // Assign the received data to the matrix
                    for (size_t i = 0; i < payload_size; i += rocc->inputOperandSize) {
                        T value = rocc->decodeOperand(&payload_data[i]);
                        int index = (rocc->matrix_read_offset + i) / rocc->inputOperandSize;
                        rocc->array->setMatrixItem(array_id, index, value);
                    }
  
                    rocc->matrix_read_offset += payload_size;
  
                    if (rocc->matrix_read_offset < rocc->matrix_total_size) {

                        // Send the next read request
                        uint64_t cache_line_size = rocc->memInterface->getLineSize();
                        uint32_t request_size = static_cast<uint32_t>(std::min(
                        cache_line_size, rocc->matrix_total_size - rocc->matrix_read_offset));
                        uint64_t next_addr = rocc_cmd->rs1 + rocc->matrix_read_offset;
                        auto *load_req = new StandardMem::Read(next_addr, request_size, GOLEM_ROCC_FLAG_SYNC_MATRIX);
                        rocc->memInterface->send(load_req);
                    } else {
                        // Matrix read complete
                        if (array_id >= 0 && static_cast<uint32_t>(array_id) < rocc->async_matrix_loads.size()) {
                            rocc->markArrayLoadReady(static_cast<uint32_t>(array_id), true);
                        }
                        rocc->completeRoCC(0);
                    }
                } break;
                //处理了一个向量的读取操作。它从内存中读取向量数据，并将这些数据存储到计算阵列中
                case GOLEM_ROCC_FLAG_SYNC_VECTOR: // Read response data is input vector
                {
                    rocc->output->verbose(CALL_INFO, 9, 0,
                                "Input vector read response detected\n");
  
                    size_t payload_size = ev->size;
                    unsigned char *payload_data = ev->data.data();
  
                    // Assign the received data to the input vector
                    for (size_t i = 0; i < payload_size; i += rocc->inputOperandSize) {
                        T value = rocc->decodeOperand(&payload_data[i]);
                        int index = (rocc->vector_read_offset + i) / rocc->inputOperandSize;
                        rocc->array->setVectorItem(array_id, index, value);
                    }
  
                    rocc->vector_read_offset += payload_size;
  
                    if (rocc->vector_read_offset < rocc->vector_total_size) {

                        // Send the next read request
                        uint64_t cache_line_size = rocc->memInterface->getLineSize();
                        uint32_t request_size = static_cast<uint32_t>(std::min(
                            cache_line_size, 
                            rocc->vector_total_size - rocc->vector_read_offset
                        ));

                        uint64_t next_addr = rocc_cmd->rs1 + rocc->vector_read_offset;
                        auto *load_req = new StandardMem::Read(next_addr, request_size, GOLEM_ROCC_FLAG_SYNC_VECTOR);
                        rocc->memInterface->send(load_req);
                    } else {
                        if (array_id >= 0 && static_cast<uint32_t>(array_id) < rocc->async_vector_loads.size()) {
                            rocc->markArrayLoadReady(static_cast<uint32_t>(array_id), false);
                        }
                        rocc->completeRoCC(0);
                    }
                } break;
                // 处理 write_gm 的主存读取响应
                case 0x10: // Read response data is for GlobalMemory Write
                {
                    rocc->output->verbose(CALL_INFO, 1, 0, "GlobalMemory write-back data received\n");

                    size_t payload_size = ev->size;
                    // print debug data in hex format
                    rocc->output->verbose(CALL_INFO, 1, 0, "Data received from MainMemory \n");
                    
                    // [优化打印] 将数据拼接成字符串，一次性打印
                    if (!ev->data.empty()) {
                        std::string hex_str;
                        char buf[8]; // 临时缓存
                        for (size_t i = 0; i < payload_size; ++i) {
                            // 将每个字节格式化为 "XX " 并追加到字符串
                            snprintf(buf, sizeof(buf), "%02X ", ev->data[i]);
                            hex_str += buf;
                        }
                        // 只调用一次 verbose，这样前缀只会出现一次
                        rocc->output->verbose(CALL_INFO, 10, 0, "Data: %s\n", hex_str.c_str());
                    } else {
                        rocc->output->verbose(CALL_INFO, 1, 0, "[EMPTY]\n");
                    }
                    
                    // 1. 将读取到的数据写入 GlobalMemory
                    // 注意：ev->data 是 std::vector<uint8_t>，直接传给 GlobalMemory 接口
                    // 目标地址 = 基地址 + 当前偏移
                    uint64_t current_dst_addr = rocc->gm_write_dst_addr + rocc->gm_write_offset;
                    
                    // 调用 GlobalMemory 的写接口 (写入本地 GM)
                    rocc->globalMem->wr_to_globalmem(current_dst_addr, payload_size, ev->data);

                    // 2. 更新偏移量
                    rocc->gm_write_offset += payload_size;

                    // 3. 检查是否还有剩余数据需要读取
                    if (rocc->gm_write_offset < rocc->gm_write_total_size) {
                        // 发起下一次读取请求
                        uint64_t cache_line_size = rocc->memInterface->getLineSize();
                        uint32_t request_size = static_cast<uint32_t>(std::min(
                            cache_line_size, 
                            rocc->gm_write_total_size - rocc->gm_write_offset
                        ));

                        // 下一次读取的主存地址 = 初始源地址 (rs1) + 新偏移
                        uint64_t next_src_addr = rocc->curr_cmd->rs1 + rocc->gm_write_offset;
                        
                        auto *load_req = new StandardMem::Read(next_src_addr, request_size, 0x10); // 保持 flag 为 0x2
                        rocc->memInterface->send(load_req);
                    } else {
                        // 全部传输完成
                        rocc->output->verbose(CALL_INFO, 9, 0, "write_gm completed.\n");
                        rocc->completeRoCC(0);
                    }
                } break;

                default:
                {
                    rocc->output->verbose(CALL_INFO, 9, 0,
                                "ERROR: unrecognized read response flag\n");
                    rocc->completeRoCC(1);
                } break;
            }
  
            delete ev;
        }
  
        virtual void handle(StandardMem::WriteResp *ev) {
            out->verbose(CALL_INFO, 9, 0,
                     "-> handle write-response (virt-addr: 0x%" PRI_ADDR ")\n", ev->vAddr);

            if (ev->getFail()) {
                out->verbose(CALL_INFO, 9, 0,
                       "RoCC store failed, responding with error code 1\n");
                rocc->completeRoCC(1);

            } else {
                
                // Continue sending write requests if there is remaining data
                if (rocc->write_offset < rocc->vector_total_size) {

                    // Calculate the size of the next write request
                    uint64_t cache_line_size = rocc->memInterface->getLineSize();
                    uint32_t request_size = static_cast<uint32_t>(std::min(
                        cache_line_size, 
                        rocc->vector_total_size - rocc->write_offset
                    ));
  
                    // Prepare the next chunk of data to write
                    std::vector<uint8_t> data_chunk(
                        rocc->outputPayload.begin() + rocc->write_offset,
                        rocc->outputPayload.begin() + rocc->write_offset + request_size
                    );
      
                    // Compute the next physical address to write to
                    uint64_t next_addr = rocc->curr_cmd->rs1 + rocc->write_offset;
      
                    // Create a new write request
                    auto* store_req = new StandardMem::Write(
                        next_addr, request_size, data_chunk,
                        false, 0, rocc->curr_cmd->rs1, 0, 0
                    );
      
                    // Send the write request
                    rocc->memInterface->send(store_req);
      
                    // Update the write offset
                    rocc->write_offset += request_size;
                } else {
                    // All data has been written; complete the RoCC command
                    rocc->completeRoCC(0);
                }
            }
            delete ev;
        }
  
    private:
        RoCCAnalog *rocc;
    };
  
    void processIncomingDataCacheEvent(StandardMem::Request *ev) {
        output->verbose(CALL_INFO, 9, 0,
                      "received incoming data cache request -> "
                      "processIncomingDataCacheEvent()\n");
  
        assert(ev != nullptr);
        assert(std_mem_handlers != nullptr);
  
        ev->handle(std_mem_handlers);
        output->verbose(CALL_INFO, 9, 0,
                      "completed pass off to incoming handlers\n");
    }

    size_t defaultRemoteLength() const {
        size_t bytes = static_cast<size_t>(arrayInputSize) * static_cast<size_t>(inputOperandSize);
        return (bytes == 0) ? 1 : bytes;
    }

    size_t resolveRemoteLength() const {
        // 统一不再使用 rd 寄存器覆盖长度，避免目标寄存器号（如 x15）无意间成为传输长度。
        // 优先使用通过 mvm.slen 设置的 remoteTransferLength；未设置则退回到一个“输入向量”大小。
        size_t chosen = (remoteTransferLength != 0) ? remoteTransferLength : defaultRemoteLength();
        return (chosen == 0) ? 1 : chosen;
    }
  
private:
    struct InflightComputeState {
        SST::Vanadis::RoCCCommand* cmd = nullptr;
        uint64_t start_cycle = 0;
        bool async_mode = false;
    };

    struct AsyncComputeState {
        bool submitted = false;
        bool completed = false;
        uint64_t rd_val = 0;
    };

    struct AsyncArrayLoadState {
        bool inflight = false;
        uint64_t base_addr = 0;
        uint64_t total_size = 0;
        uint64_t offset = 0;
        uint64_t request_tag = 0;
        uint64_t command_id = 0;
        uint32_t array_id = 0;
        bool ready = false;
        bool failed = false;
        bool local_request_inflight = false;
        bool array_request_inflight = false;
        bool completes_command = false;
        std::vector<uint8_t> payload;
    };

    struct LegacyOutputStoreState {
        bool active = false;
        bool array_request_inflight = false;
        bool local_request_inflight = false;
        uint64_t command_id = 0;
        uint64_t dest_addr = 0;
        uint64_t offset = 0;
        uint64_t request_tag = 0;
        uint32_t array_id = 0;
        std::vector<uint8_t> payload;
    };

    bool isAsyncArrayLoadCommand(const SST::Vanadis::RoCCCommand* cmd) const {
        if (!enable_async_array_load) {
            return false;
        }
        if (cmd == nullptr || cmd->inst == nullptr) {
            return false;
        }
        const uint8_t op = cmd->inst->func7;
        return (cmd->inst->rd == 0) && (op == 0x7 || op == 0x8);
    }

    bool isArrayLoadInflight(uint32_t array_id) const {
        if (array_id >= async_matrix_loads.size() || array_id >= async_vector_loads.size()) {
            return false;
        }
        return async_matrix_loads[array_id].inflight || async_vector_loads[array_id].inflight;
    }

    bool hasArrayLoadFailure(uint32_t array_id) const {
        if (array_id >= async_matrix_loads.size() || array_id >= async_vector_loads.size()) {
            return false;
        }
        return async_matrix_loads[array_id].failed || async_vector_loads[array_id].failed;
    }

    void markArrayLoadReady(uint32_t array_id, bool is_matrix) {
        if (array_id >= async_matrix_loads.size() || array_id >= async_vector_loads.size()) {
            return;
        }
        auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
        state = AsyncArrayLoadState{};
        state.ready = true;
    }

    void markAsyncLoadFailed(uint32_t array_id, bool is_matrix) {
        if (array_id >= async_matrix_loads.size() || array_id >= async_vector_loads.size()) {
            return;
        }
        auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
        const bool completes_command = state.completes_command;
        const uint64_t command_id = state.command_id;
        state = AsyncArrayLoadState{};
        state.failed = true;
        output->verbose(CALL_INFO, 0, 0,
            "[RoCC ERROR] async %s load failed on array=%" PRIu32 "\n",
            is_matrix ? "matrix" : "vector",
            array_id);
        if (completes_command && curr_cmd != nullptr &&
            curr_cmd->cmd_id == command_id) {
            completeRoCC(1);
        }
    }

    uint64_t allocateLocalTransferTag() {
        const uint64_t tag = next_local_transfer_tag_++;
        if (next_local_transfer_tag_ == 0) {
            next_local_transfer_tag_ = 1;
        }
        return tag;
    }

    void initializeAsyncArrayLoad(AsyncArrayLoadState& state,
                                  uint32_t array_id,
                                  uint64_t base_addr,
                                  uint64_t total_size,
                                  bool completes_command,
                                  uint64_t command_id) {
        state = AsyncArrayLoadState{};
        state.inflight = true;
        state.array_id = array_id;
        state.base_addr = base_addr;
        state.total_size = total_size;
        state.completes_command = completes_command;
        state.command_id = command_id;
        state.payload.resize(total_size);
    }

    void finishAsyncArrayLoad(uint32_t array_id, bool is_matrix) {
        if (array_id >= async_matrix_loads.size() ||
            array_id >= async_vector_loads.size()) {
            return;
        }
        auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
        const bool completes_command = state.completes_command;
        const uint64_t command_id = state.command_id;
        markArrayLoadReady(array_id, is_matrix);
        if (completes_command && curr_cmd != nullptr &&
            curr_cmd->cmd_id == command_id) {
            completeRoCC(0);
        }
    }

    void progressAsyncArrayLoad(uint32_t array_id, bool is_matrix) {
        if (array_id >= async_matrix_loads.size() ||
            array_id >= async_vector_loads.size()) {
            return;
        }
        auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
        if (!state.inflight || state.local_request_inflight ||
            state.array_request_inflight) {
            return;
        }
        if (inputOperandSize == 0 || state.offset > state.total_size) {
            markAsyncLoadFailed(array_id, is_matrix);
            return;
        }

        if (state.offset < state.total_size) {
            const size_t max_request =
                std::max<size_t>(globalMem->localMaxRequestBytes(), 1);
            const size_t chunk_size = std::min<size_t>(
                max_request, state.total_size - state.offset);
            const uint64_t chunk_offset = state.offset;
            const uint64_t tag = allocateLocalTransferTag();
            const bool accepted = globalMem->localReadAsync(
                state.base_addr + chunk_offset, chunk_size,
                LocalMemoryClient::RoCC, tag,
                [this, array_id, is_matrix, chunk_offset, chunk_size](
                    bool success, uint64_t callback_tag,
                    const std::vector<uint8_t>& data) {
                    auto& callback_state = is_matrix
                        ? async_matrix_loads[array_id]
                        : async_vector_loads[array_id];
                    if (!callback_state.inflight ||
                        callback_state.request_tag != callback_tag) {
                        return;
                    }
                    callback_state.local_request_inflight = false;
                    if (!success || data.size() != chunk_size ||
                        callback_state.offset != chunk_offset) {
                        markAsyncLoadFailed(array_id, is_matrix);
                        return;
                    }
                    std::copy(data.begin(), data.end(),
                              callback_state.payload.begin() + chunk_offset);
                    callback_state.offset += chunk_size;
                    progressAsyncArrayLoad(array_id, is_matrix);
                });
            if (accepted) {
                state.local_request_inflight = true;
                state.request_tag = tag;
            }
            return;
        }

        std::vector<double> values(state.total_size / inputOperandSize, 0.0);
        for (size_t i = 0; i < state.total_size; i += inputOperandSize) {
            T value = 0;
            memcpy(&value, &state.payload[i],
                   std::min<size_t>(sizeof(T), inputOperandSize));
            values[i / inputOperandSize] = static_cast<double>(value);
        }
        const uint64_t tag = allocateLocalTransferTag();
        auto callback = [this, array_id, is_matrix](bool success,
                                                    uint64_t callback_tag) {
            auto& callback_state = is_matrix
                ? async_matrix_loads[array_id]
                : async_vector_loads[array_id];
            if (!callback_state.inflight ||
                callback_state.request_tag != callback_tag) {
                return;
            }
            callback_state.array_request_inflight = false;
            if (!success) {
                markAsyncLoadFailed(array_id, is_matrix);
                return;
            }
            finishAsyncArrayLoad(array_id, is_matrix);
        };
        const bool accepted = is_matrix
            ? array->programMatrixAsync(
                  array_id, values, inputOperandSize, tag, callback)
            : array->programInputAsync(
                  array_id, values, inputOperandSize, tag, callback);
        if (accepted) {
            state.array_request_inflight = true;
            state.request_tag = tag;
        }
    }

    bool tryCompleteAsyncArrayLoads(uint64_t) {
        bool progressed = false;
        for (uint32_t array_id = 0; array_id < async_matrix_loads.size(); ++array_id) {
            auto& mstate = async_matrix_loads[array_id];
            if (mstate.inflight) {
                progressAsyncArrayLoad(array_id, true);
                progressed = true;
            }
            auto& vstate = async_vector_loads[array_id];
            if (vstate.inflight) {
                progressAsyncArrayLoad(array_id, false);
                progressed = true;
            }
        }
        return progressed;
    }

    bool tryIssueAsyncArrayLoadCommand(uint64_t cycle) {
        if (roccCmd_q.empty()) {
            return false;
        }
        auto* cmd = roccCmd_q.front();
        if (!isAsyncArrayLoadCommand(cmd)) {
            return false;
        }

        const bool is_matrix = (cmd->inst->func7 == 0x8);
        const uint32_t array_id = static_cast<uint32_t>(cmd->rs2);
        if (array_id >= static_cast<uint32_t>(numArrays)) {
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
            roccCmd_q.pop_front();
            delete cmd;
            return true;
        }

        auto& state = is_matrix ? async_matrix_loads[array_id] : async_vector_loads[array_id];
        if (state.inflight) {
            return false;
        }
        if (isArrayComputeInflight(array_id)) {
            return false;
        }

        const uint64_t total_size = is_matrix
            ? static_cast<uint64_t>(arrayInputSize) * static_cast<uint64_t>(arrayOutputSize) * static_cast<uint64_t>(inputOperandSize)
            : static_cast<uint64_t>(arrayInputSize) * static_cast<uint64_t>(inputOperandSize);
        initializeAsyncArrayLoad(
            state, array_id, cmd->rs1, total_size, false, 0);

        if (total_size == 0) {
            markArrayLoadReady(array_id, is_matrix);
            enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));
            roccCmd_q.pop_front();
            delete cmd;
            return true;
        }

        progressAsyncArrayLoad(array_id, is_matrix);
        enqueueResponse(new SST::Vanadis::RoCCResponse(cmd->inst->rd, 0, cmd->cmd_id, cmd->hw_thread));

        roccCmd_q.pop_front();
        delete cmd;
        return true;
    }

    void beginBlockingArrayLoad(bool is_matrix) {
        if (curr_cmd == nullptr ||
            curr_cmd->rs2 >= static_cast<uint64_t>(numArrays)) {
            completeRoCC(1);
            return;
        }
        const uint32_t array_id = static_cast<uint32_t>(curr_cmd->rs2);
        auto& state = is_matrix
            ? async_matrix_loads[array_id]
            : async_vector_loads[array_id];
        if (state.inflight) {
            progressAsyncArrayLoad(array_id, is_matrix);
            return;
        }
        const uint64_t total_size = is_matrix
            ? static_cast<uint64_t>(arrayInputSize) *
                  static_cast<uint64_t>(arrayOutputSize) *
                  static_cast<uint64_t>(inputOperandSize)
            : static_cast<uint64_t>(arrayInputSize) *
                  static_cast<uint64_t>(inputOperandSize);
        initializeAsyncArrayLoad(
            state, array_id, curr_cmd->rs1, total_size, true, curr_cmd->cmd_id);
        progressAsyncArrayLoad(array_id, is_matrix);
    }

    void finishLegacyOutputStore(bool success) {
        const uint64_t command_id = legacy_output_store_.command_id;
        legacy_output_store_ = LegacyOutputStoreState{};
        if (curr_cmd != nullptr && curr_cmd->cmd_id == command_id) {
            completeRoCC(success ? 0 : 1);
        }
    }

    void progressLegacyOutputStore() {
        auto& state = legacy_output_store_;
        if (!state.active || state.array_request_inflight ||
            state.local_request_inflight) {
            return;
        }
        if (state.payload.empty()) {
            const uint64_t tag = allocateLocalTransferTag();
            const bool accepted = array->readOutputBytesAsync(
                state.array_id, outputOperandSize, tag,
                [this](bool success, uint64_t callback_tag,
                       const std::vector<uint8_t>& bytes) {
                    auto& callback_state = legacy_output_store_;
                    if (!callback_state.active ||
                        callback_state.request_tag != callback_tag) {
                        return;
                    }
                    callback_state.array_request_inflight = false;
                    if (!success ||
                        bytes.size() != static_cast<size_t>(arrayOutputSize) *
                            outputOperandSize ||
                        outputOperandSize == 0) {
                        finishLegacyOutputStore(false);
                        return;
                    }
                    callback_state.payload = bytes;
                    progressLegacyOutputStore();
                });
            if (accepted) {
                state.array_request_inflight = true;
                state.request_tag = tag;
            }
            return;
        }

        if (state.offset >= state.payload.size()) {
            finishLegacyOutputStore(true);
            return;
        }
        const size_t max_request =
            std::max<size_t>(globalMem->localMaxRequestBytes(), 1);
        const size_t chunk_size = std::min<size_t>(
            max_request, state.payload.size() - state.offset);
        const uint64_t chunk_offset = state.offset;
        std::vector<uint8_t> chunk(
            state.payload.begin() + chunk_offset,
            state.payload.begin() + chunk_offset + chunk_size);
        const uint64_t tag = allocateLocalTransferTag();
        const bool accepted = globalMem->localWriteAsync(
            state.dest_addr + chunk_offset, chunk, LocalMemoryClient::RoCC, tag,
            [this, chunk_offset, chunk_size](bool success,
                                             uint64_t callback_tag) {
                auto& callback_state = legacy_output_store_;
                if (!callback_state.active ||
                    callback_state.request_tag != callback_tag) {
                    return;
                }
                callback_state.local_request_inflight = false;
                if (!success || callback_state.offset != chunk_offset) {
                    finishLegacyOutputStore(false);
                    return;
                }
                callback_state.offset += chunk_size;
                progressLegacyOutputStore();
            });
        if (accepted) {
            state.local_request_inflight = true;
            state.request_tag = tag;
        }
    }

    void enqueueResponse(SST::Vanadis::RoCCResponse *resp) {
        if (resp != nullptr) {
            resp_q.push_back(resp);
        }
    }

    std::deque<SST::Vanadis::RoCCCommand *> roccCmd_q;
    std::deque<SST::Vanadis::RoCCResponse *> resp_q;
    std::vector<AsyncComputeState> async_compute_states;
    bool busy;
    SST::Vanadis::RoCCCommand *curr_cmd;
  
    StandardMemHandlers *std_mem_handlers;
    StandardMem *memInterface;
  
    int max_instructions;
  
    Golem::ComputeArray *array;
    std::vector<char> arrayStates;
    SST::Golem::GlobalMemoryAPI *globalMem; //GlobalMemory子组件指针
    SST::Golem::GroupCtrlAPI *groupCtrl;
    SST::Golem::RequestSchedulerAPI *requestScheduler;
    SST::Golem::WorkerCommandProcessorAPI *workerCommandProcessor;
    SST::Golem::SFUAPI *sfu;
    std::unordered_map<uint64_t, ManagerTensorJobState> managerTensorJobs_;
    std::unordered_map<uint64_t, ManagerAttentionJobState> managerAttentionJobs_;
    std::unique_ptr<AttentionWorkerState> attentionWorker_;
    std::deque<ControlTransportMessage> attentionPendingDispatches_;
    uint64_t nextAttentionWorkerGeneration_ = 1;
    std::vector<uint8_t> attentionArrayPending_;
    bool sfuWaitBlocked_ = false;
    uint64_t sfuWaitBlockedCmdId_ = 0;
    uint64_t sfuWaitBlockedUntilTick_ = 0;
    uint64_t coreID;
    uint64_t attentionWindowOffset_;
    uint64_t attentionWindowBytes_;
    bool attentionPvMatrixBroadcast_;
    bool attentionQkMatrixBroadcast_;
    bool attentionQkDataflowTranspose_;
    bool attentionQkEarlyCompute_;
    bool attentionQkInputPipeline_;
    bool attentionQkReadoutOverlap_;
    uint32_t attentionQkReadoutWindow_;
    bool attentionQkScoreRowBurst_;
    bool attentionCrossTileOperandPipeline_;
    uint32_t attentionOperandContextBanks_;
    bool attentionKvTileRotation_;
    bool attentionKvDoubleBuffer_;
    uint32_t attentionKvBufferCount_;
    bool attentionKvDistributionEnable_;
    bool attentionKvSecondLookahead_;
    bool attentionKvCrossQueryPrefetch_;
    bool attentionKvPairReuse_;
    uint32_t attentionKvQueryGroupSize_;
    bool attentionPvVTileReuse_;
    bool attentionPvVTileGroupRetention_;
    uint64_t attentionPvVTileBufferBytes_;
    uint64_t attentionPvVTileBufferHitTicks_;
    uint64_t attentionPvVTileBufferBytesPerCycle_;
    uint64_t attentionPvVTileBufferOffset_;
    bool attentionPvInputPipeline_;
    bool attentionClusterPvRowWavefront_;
    bool attentionClusterQkMatrixLookahead_;
    bool attentionClusterPvMatrixLookahead_;
    bool attentionPvCompactInput_;
    bool attentionPvInputResidency_;
    bool attentionOAccumulatorCBuffer_;
    bool attentionPvRestorePipeline_;
    bool attentionPvOutputPipeline_;
    bool attentionPvORowFusion_;
    bool attentionPvEarlyCompute_;
    bool attentionPvMatrixSoftmaxOverlap_;
    bool attentionPvActiveK_;
    bool attentionGenericGemmEnable_;
    bool attentionReuseWindowQkBridge_ = false;
    bool attentionWorkerClusterBridge_ = false;
    uint32_t attentionWorkerClusterQkWorkersPerManager_ = 1;
    bool attentionWorkerClusterRowPriority_ = false;
    bool attentionWorkerClusterVBroadcast_ = false;
    bool attentionWorkerClusterDynamicPv_ = false;
    bool attentionMilestoneTrace_;
    bool attentionTileTrace_;
    bool attentionClusterEnable_ = false;
    bool attentionSequential64Enable_ = false;
    AttentionClusterConfig attentionClusterConfig_ = {};
    std::unique_ptr<AttentionClusterState> attentionCluster_;
    BusyActivityTracker attentionClusterQkArrayActivity_;
    BusyActivityTracker attentionClusterPvArrayActivity_;

  
    // Tile Parameters
    int numArrays;
    int arrayInputSize;
    int arrayOutputSize;
    int inputOperandSize;
    int outputOperandSize;
    uint64_t vectorStrideBytes;
  
    // MMIO range delimiters
    uint64_t mmioStartAddr;
    uint64_t inputDataSize;
    uint64_t outputDataSize;
    uint64_t inputTotalSize;
    uint64_t outputTotalSize;
    uint64_t inputStartAddr;
    uint64_t outputStartAddr;    
    // virtual void compute(uint32_t arrayID) override {
    //     auto& inputVector = inputVectors[arrayID];
    //     auto& outputVector = outputVectors[arrayID];
    //     auto& matrix = matrixData[arrayID];

    //     // Ensure output vector is correctly sized
    //     outputVector.resize(outputArraySize);

    //     // Initialize output vector to zero
    //     std::fill(outputVector.begin(), outputVector.end(), T());

    //     // Print input vector
    //     out.verbose(CALL_INFO, 2, 0, "MVM for array %u:\n\n", arrayID);
    //     for (uint32_t col = 0; col < inputArraySize; col++) {
    //         printValue(inputVector[col]);
    //     }
    //     out.verbose(CALL_INFO, 2, 0, "\n\n");

    //     // Perform matrix-vector multiplication
    //     for (uint32_t row = 0; row < outputArraySize; row++) {
    //         for (uint32_t col = 0; col < inputArraySize; col++) {
    //             outputVector[row] += matrix[row * inputArraySize + col] * inputVector[col];
    //             printValue(matrix[row * inputArraySize + col]);
    //         }
    //         out.verbose(CALL_INFO, 2, 0, "  ");
    //         printValue(outputVector[row]);
    //         out.verbose(CALL_INFO, 2, 0, "\n");
    //     }
    //     out.verbose(CALL_INFO, 2, 0, "\n\n");
    // }

    // virtual SimTime_t getArrayLatency(uint32_t arrayID) override {
    //     return 1;
    // }
  
    // Variables to keep track of read/write request progress
    uint64_t matrix_read_offset;
    uint64_t matrix_total_size;
    uint64_t vector_read_offset;
    uint64_t vector_total_size;
    uint64_t write_offset;
    std::vector<uint8_t> outputPayload;
    size_t remoteTransferLength;
    uint64_t remoteStoreCompletionToken = 0;
    bool enable_async_array_load = true;
    bool sfuEnable = false;

    // 用于 write_gm 的状态变量
    uint64_t gm_write_dst_addr;   // GlobalMemory 的目标地址 (rs2)
    uint64_t gm_write_offset;     // 当前已处理的字节偏移量
    uint64_t gm_write_total_size; // 总共需要传输的字节数

    Statistics::Statistic<uint64_t>* stat_cycles_mvm_set;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm_l;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm_s;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm_mv;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm_ovec2gm;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm_gm2ivec;
    Statistics::Statistic<uint64_t>* stat_cycles_mvm_gm2imat;
    Statistics::Statistic<uint64_t>* stat_cycles_remote_st;
    Statistics::Statistic<uint64_t>* stat_cycles_remote_ld;
    Statistics::Statistic<uint64_t>* statTensorManagerJobsIssued_;
    Statistics::Statistic<uint64_t>* statTensorManagerWorkersMapped_;
    Statistics::Statistic<uint64_t>* statTensorManagerRowsDispatched_;
    Statistics::Statistic<uint64_t>* statTensorManagerRowsCompleted_;
    Statistics::Statistic<uint64_t>* statTensorManagerJobsCompleted_;
    Statistics::Statistic<uint64_t>* statTensorManagerDescriptorAcceptTick_;
    Statistics::Statistic<uint64_t>* statTensorManagerBandDispatchTick_;
    Statistics::Statistic<uint64_t>* statTensorManagerCompletionReceivedTick_;
    Statistics::Statistic<uint64_t>* statTensorManagerCompleteTick_;
    Statistics::Statistic<uint64_t>* statTensorManagerWaitObservedTick_;
    Statistics::Statistic<uint64_t>* statAttentionManagerJobsIssued_;
    Statistics::Statistic<uint64_t>* statAttentionManagerJobsCompleted_;
    Statistics::Statistic<uint64_t>* statAttentionManagerBandsCompleted_;
    Statistics::Statistic<uint64_t>* statAttentionManagerBandCompletionsReceived_;
    Statistics::Statistic<uint64_t>* statAttentionTensorJobsCompleted_;
    Statistics::Statistic<uint64_t>* statAttentionManagerDescriptorAcceptTick_;
    Statistics::Statistic<uint64_t>* statAttentionManagerDispatchTick_;
    Statistics::Statistic<uint64_t>* statAttentionManagerLocalCompleteTick_;
    Statistics::Statistic<uint64_t>* statAttentionManagerBandCompletionReceivedTick_;
    Statistics::Statistic<uint64_t>* statAttentionTensorCompleteTick_;
    Statistics::Statistic<uint64_t>* statAttentionManagerWaitObservedTick_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerDispatchAcceptTick_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerQkTileCompleteTick_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerSoftmaxTileCompleteTick_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerPvTileCompleteTick_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerOutputDmaAckTick_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerInterTileTotalTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerInterTileOutputDmaTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerInterTileQueryLoadTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerInterTileKvLoadTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerInterTileQLocalReadTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerInterTileQkMatrixProgramTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerInterTileQkInputProgramTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerInterTileQkComputeReadoutTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerTileTotalTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerTileKvLoadTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerTileQLocalReadTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerTileQkMatrixProgramTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerTileQkInputProgramTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerTileQkComputeReadoutTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerTileSoftmaxTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerTilePvMatrixProgramTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerTilePvInputProgramTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerTilePvRestoreOutputTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerTilePvComputeTicks_;
    Statistics::Statistic<uint64_t>* statAttentionWorkerTilePvOutputReadwriteTicks_;
    Statistics::Statistic<uint64_t>* statAttentionKvPrefetchTiles_;
    Statistics::Statistic<uint64_t>* statAttentionKvPrefetchHits_;
    Statistics::Statistic<uint64_t>* statAttentionKvPrefetchWaits_;
    Statistics::Statistic<uint64_t>* statAttentionKvPrefetchDmaTicks_;
    Statistics::Statistic<uint64_t>* statAttentionKvPrefetchReadyLeadTicks_;
    Statistics::Statistic<uint64_t>* statAttentionKvPrefetchWaitTicks_;
    Statistics::Statistic<uint64_t>* statAttentionKvKReleaseTicks_;
    Statistics::Statistic<uint64_t>* statAttentionKvVReleaseTicks_;
    Statistics::Statistic<uint64_t>* statAttentionKvNextReadyAtReleaseTiles_;
    Statistics::Statistic<uint64_t>* statAttentionKvSecondLookaheadCandidates_;
    Statistics::Statistic<uint64_t>* statAttentionKvSecondLookaheadPrefetches_;
    Statistics::Statistic<uint64_t>* statAttentionKvSecondLookaheadLeadTicks_;
    Statistics::Statistic<uint64_t>* statAttentionKvCrossQueryPrefetches_;
    Statistics::Statistic<uint64_t>* statAttentionKvCrossQueryHits_;
    Statistics::Statistic<uint64_t>* statAttentionKvCrossQueryWaits_;
    Statistics::Statistic<uint64_t>* statAttentionKvCrossQueryWaitTicks_;
    Statistics::Statistic<uint64_t>* statAttentionKvPairReuseTiles_;
    Statistics::Statistic<uint64_t>* statAttentionKvPairReuseBytes_;
    Statistics::Statistic<uint64_t>* statAttentionPvInputPipelineRows_;
    Statistics::Statistic<uint64_t>* statAttentionClusterPvWavefrontRows_;
    Statistics::Statistic<uint64_t>*
        statAttentionClusterQkMatrixLookaheadLaunches_;
    Statistics::Statistic<uint64_t>*
        statAttentionClusterQkMatrixLookaheadHits_;
    Statistics::Statistic<uint64_t>*
        statAttentionClusterPvMatrixLookaheadLaunches_;
    Statistics::Statistic<uint64_t>*
        statAttentionClusterPvMatrixLookaheadHits_;
    Statistics::Statistic<uint64_t>* statAttentionPvRestorePipelineRows_;
    Statistics::Statistic<uint64_t>* statAttentionPvOutputPipelineRows_;
    Statistics::Statistic<uint64_t>* statAttentionPvEarlyComputeArrays_;
    Statistics::Statistic<uint64_t>* statAttentionPvMatrixOverlapTiles_;
    Statistics::Statistic<uint64_t>* statAttentionPvMatrixOverlapHits_;
    Statistics::Statistic<uint64_t>* statAttentionPvMatrixOverlapWaits_;
    Statistics::Statistic<uint64_t>* statAttentionQkMatrixBroadcasts_;
    Statistics::Statistic<uint64_t>* statAttentionPvMatrixBroadcasts_;
    Statistics::Statistic<uint64_t>* statAttentionQkArrayOps_;
    Statistics::Statistic<uint64_t>* statAttentionQkEarlyComputeArrays_;
    Statistics::Statistic<uint64_t>* statAttentionQkInputPipelineRowsFetched_;
    Statistics::Statistic<uint64_t>* statAttentionQkInputPipelineRowsProgrammed_;
    Statistics::Statistic<uint64_t>* statAttentionQkInputPipelineOverlapTicks_;
    Statistics::Statistic<uint64_t>* statAttentionQkInputPipelineSlotFullStalls_;
    Statistics::Statistic<uint64_t>* statAttentionQkInputPipelineMaxDepth_;
    Statistics::Statistic<uint64_t>* statAttentionQkInputPipelineTagMismatches_;
    Statistics::Statistic<uint64_t>* statAttentionQkReadoutAheadDepth_;
    Statistics::Statistic<uint64_t>* statAttentionCrossTileOperandCandidates_;
    Statistics::Statistic<uint64_t>* statAttentionCrossTileOperandLaunches_;
    Statistics::Statistic<uint64_t>* statAttentionCrossTileOperandMatrixPrograms_;
    Statistics::Statistic<uint64_t>* statAttentionCrossTileOperandInputRows_;
    Statistics::Statistic<uint64_t>* statAttentionCrossTileOperandReadyHits_;
    Statistics::Statistic<uint64_t>* statAttentionCrossTileOperandWaits_;
    Statistics::Statistic<uint64_t>* statAttentionCrossTileOperandWaitTicks_;
    Statistics::Statistic<uint64_t>* statAttentionCrossTileOperandPromotions_;
    Statistics::Statistic<uint64_t>* statAttentionCrossTileOperandTagMismatches_;
    Statistics::Statistic<uint64_t>* statAttentionPvArrayOps_;
    Statistics::Statistic<uint64_t>* statAttentionGenericGemmQkOps_;
    Statistics::Statistic<uint64_t>* statAttentionGenericGemmPvOps_;
    Statistics::Statistic<uint64_t>* statAttentionPvActiveKLaunches_;
    Statistics::Statistic<uint64_t>* statAttentionPvActiveKColumns_;
    Statistics::Statistic<uint64_t>* statAttentionPvActiveKMatrixElements_;
    Statistics::Statistic<uint64_t>* statAttentionPvVTileBufferHits_;
    Statistics::Statistic<uint64_t>* statAttentionPvVTileBufferMisses_;
    Statistics::Statistic<uint64_t>* statAttentionPvVTileBufferBytesRead_;
    Statistics::Statistic<uint64_t>* statAttentionPvVTileBufferBytesReused_;
    Statistics::Statistic<uint64_t>* statAttentionPvVTileBufferWaitTicks_;
    Statistics::Statistic<uint64_t>* statAttentionPvVTileBufferCapacityRejections_;
    Statistics::Statistic<uint64_t>* statAttentionPvVTileBufferGroupHits_;
    Statistics::Statistic<uint64_t>* statAttentionPvInputResidencyHits_;
    Statistics::Statistic<uint64_t>* statAttentionPvInputResidencyRowsReused_;
    Statistics::Statistic<uint64_t>* statAttentionPvInputResidencyInvalidations_;
    Statistics::Statistic<uint64_t>* statAttentionOAccumulatorStores_;
    Statistics::Statistic<uint64_t>* statAttentionOAccumulatorRestores_;
    Statistics::Statistic<uint64_t>* statAttentionOAccumulatorBytes_;
    Statistics::Statistic<uint64_t>* statAttentionSpHbmBytes_;
    Statistics::Statistic<uint64_t>* statAttentionClusterConfigFingerprint_;
    Statistics::Statistic<uint64_t>* statAttentionClusterWorkerJobs_;
    Statistics::Statistic<uint64_t>* statAttentionClusterContextsIssued_;
    Statistics::Statistic<uint64_t>* statAttentionClusterContextsCompleted_;
    Statistics::Statistic<uint64_t>* statAttentionClusterContextsCancelled_;
    Statistics::Statistic<uint64_t>* statAttentionClusterBankRefsCancelled_;
    Statistics::Statistic<uint64_t>* statAttentionClusterMemoryRequestsCancelled_;
    Statistics::Statistic<uint64_t>* statAttentionClusterContextHighWater_;
    Statistics::Statistic<uint64_t>* statAttentionClusterStaleCallbacks_;
    Statistics::Statistic<uint64_t>* statAttentionClusterIllegalTransitions_;
    Statistics::Statistic<uint64_t>* statAttentionClusterScoreSlotReservations_;
    Statistics::Statistic<uint64_t>* statAttentionClusterScoreSlotReleases_;
    Statistics::Statistic<uint64_t>* statAttentionClusterScoreSlotFullStalls_;
    Statistics::Statistic<uint64_t>* statAttentionClusterScoreSlotHighWater_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOContextReservations_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOContextReleases_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOContextCancelled_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOContextHighWater_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOScaleSegments_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOAccumulateSegments_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOFusedRows_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOFusedBytes_;
    Statistics::Statistic<uint64_t>* statAttentionClusterODrainRequests_;
    Statistics::Statistic<uint64_t>* statAttentionClusterODrainBytes_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOReadWaitCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOWriteWaitCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOAluWaitCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOBankConflictCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterODrainWaitCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOReadBusyUnionCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOReadBusySpanCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOReadIdleGapCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOReadMaxConcurrency_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOWriteBusyUnionCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOWriteBusySpanCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOWriteIdleGapCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOWriteMaxConcurrency_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOAluBusyUnionCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOAluBusySpanCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOAluIdleGapCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterOAluMaxConcurrency_;
    Statistics::Statistic<uint64_t>* statAttentionClusterQkArrayBusyUnionTicks_;
    Statistics::Statistic<uint64_t>* statAttentionClusterQkArrayBusySpanTicks_;
    Statistics::Statistic<uint64_t>* statAttentionClusterQkArrayIdleGapTicks_;
    Statistics::Statistic<uint64_t>* statAttentionClusterQkArrayMaxConcurrency_;
    Statistics::Statistic<uint64_t>* statAttentionClusterPvArrayBusyUnionTicks_;
    Statistics::Statistic<uint64_t>* statAttentionClusterPvArrayBusySpanTicks_;
    Statistics::Statistic<uint64_t>* statAttentionClusterPvArrayIdleGapTicks_;
    Statistics::Statistic<uint64_t>* statAttentionClusterPvArrayMaxConcurrency_;
    Statistics::Statistic<uint64_t>* statAttentionClusterKTileBroadcasts_;
    Statistics::Statistic<uint64_t>* statAttentionClusterKTileBytes_;
    Statistics::Statistic<uint64_t>* statAttentionClusterQPairMulticasts_;
    Statistics::Statistic<uint64_t>* statAttentionClusterQPairBytes_;
    Statistics::Statistic<uint64_t>* statAttentionClusterScoreBeats_;
    Statistics::Statistic<uint64_t>* statAttentionClusterScoreBytes_;
    Statistics::Statistic<uint64_t>* statAttentionClusterAheadContextsLaunched_;
    Statistics::Statistic<uint64_t>* statAttentionClusterAheadContextsCompleted_;
    Statistics::Statistic<uint64_t>* statAttentionClusterAheadContextsPromoted_;
    Statistics::Statistic<uint64_t>* statAttentionClusterPromotionWaits_;
    Statistics::Statistic<uint64_t>* statAttentionClusterInitialEnqueueRetries_;
    Statistics::Statistic<uint64_t>* statAttentionClusterQkTileStarts_;
    Statistics::Statistic<uint64_t>* statAttentionClusterQkTileIiCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterQkBoundaryIiCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterQkSteadyIiOverTarget_;
    Statistics::Statistic<uint64_t>* statAttentionClusterQkSfuOverlapCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterSfuPvOverlapCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterQkPvOverlapCycles_;
    Statistics::Statistic<uint64_t>* statAttentionClusterThreeStageOverlapCycles_;
    Statistics::Statistic<uint64_t>* statAttentionSequentialQkWaves_;
    Statistics::Statistic<uint64_t>* statAttentionSequentialPvWaves_;
    Statistics::Statistic<uint64_t>* statAttentionSequentialQkActiveArrays_;
    Statistics::Statistic<uint64_t>* statAttentionSequentialPvActiveArrays_;

    AttentionOAccumulator attentionOAccumulator_;
    std::unique_ptr<AttentionWorkerState> attentionWorkerClusterQkAhead_;
    std::deque<std::unique_ptr<AttentionWorkerState>>
        attentionWorkerClusterPvDraining_;
    AttentionWorkerClusterPvState attentionWorkerClusterPv_;
    std::deque<ControlTransportMessage> attentionWorkerClusterPvQueue_;
    using AttentionWorkerClusterPvOutputKey =
        std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>;
    std::map<AttentionWorkerClusterPvOutputKey, std::vector<float>>
        attentionWorkerClusterPvOutputs_;
    std::array<uint32_t, 8> attentionWorkerClusterPvAssignedRows_ = {};
    uint32_t attentionWorkerClusterPvAssignCursor_ = 0;
    std::unordered_map<uint64_t, AttentionWorkerClusterVEntry>
        attentionWorkerClusterVEntries_;
    std::deque<uint64_t> attentionWorkerClusterVLoadQueue_;
    AttentionWorkerClusterVLoad attentionWorkerClusterVLoad_;
    std::unordered_set<uint64_t> attentionWorkerClusterVResident_;

    uint64_t StartTickCycle;
    uint64_t LastTickCycle;

    uint64_t latency_mvm_ovec2gm;
    uint64_t latency_mvm_gm2ivec;
    uint64_t latency_mvm_gm2imat;
    uint64_t latency_remote_st;
    uint64_t latency_remote_ld;
    std::vector<InflightComputeState> inflight_compute_cmds;
    std::vector<AsyncArrayLoadState> async_matrix_loads;
    std::vector<AsyncArrayLoadState> async_vector_loads;
    LegacyOutputStoreState legacy_output_store_;
    uint64_t next_local_transfer_tag_ = 1;
    bool progress_heartbeat = false;
    uint64_t progress_interval_cycles = 50000;
    uint64_t progress_total_mvm_ops = 0;
    uint64_t mvm_ops_completed = 0;
    uint64_t progress_next_cycle = 0;
    uint64_t progress_last_percent = 101;
};
  
} // namespace Golem
} // namespace SST
  
#endif
