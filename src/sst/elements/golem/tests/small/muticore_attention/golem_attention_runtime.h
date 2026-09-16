#pragma once

#include <cstdint>

#include "../mvm_noc_int_array/ex_instr.h"

constexpr uint32_t GOLEM_ATTENTION_DESC_MAGIC = 0x41545431u;
constexpr uint16_t GOLEM_ATTENTION_DESC_VERSION = 2u;
constexpr uint32_t GOLEM_ATTENTION_FLAG_CAUSAL = 0x1u;
constexpr uint32_t GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_JOB = 0x21u;
constexpr uint32_t GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_WAIT = 0x22u;
constexpr uint32_t SFU_WORKER_TOPOLOGY_MAP_MAGIC = 0x574d4150u;
constexpr uint16_t SFU_WORKER_TOPOLOGY_MAP_VERSION = 1u;
constexpr uint32_t SFU_WORKER_TOPOLOGY_MAX_WORKERS = 16u;

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

struct SFUWorkerTopologyMapV1 {
    uint32_t magic;
    uint16_t version;
    uint16_t size_bytes;
    uint32_t worker_count;
    uint32_t reserved0;
    uint32_t worker_core_ids[SFU_WORKER_TOPOLOGY_MAX_WORKERS];
};

static_assert(sizeof(GolemAttentionDescV2) == 128, "Attention ABI mismatch");
static_assert(sizeof(SFUWorkerTopologyMapV1) == 80, "Topology ABI mismatch");

inline void attention_manager_job(uint64_t desc_gm_addr, uint64_t tag) {
    asm volatile(
        ".insn r 0x0b, 7, %2, x0, %0, %1"
        :
        : "r"(desc_gm_addr), "r"(tag),
          "i"(GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_JOB)
        : "memory");
}

inline uint64_t attention_manager_wait(uint64_t tag) {
    uint64_t status;
    asm volatile(
        ".insn r 0x0b, 7, %2, %0, %1, x0"
        : "=r"(status)
        : "r"(tag), "i"(GOLEM_ROCC_FUNC7_ATTENTION_MANAGER_WAIT)
        : "memory");
    return status;
}

template <typename Metadata>
inline void attention_write_metadata(uint64_t gm_addr, const Metadata& metadata) {
    const auto* words = reinterpret_cast<const uint64_t*>(&metadata);
    for (uint64_t index = 0; index < sizeof(Metadata) / sizeof(uint64_t); ++index) {
        reg2gm(words[index], gm_addr + index * sizeof(uint64_t));
    }
}
