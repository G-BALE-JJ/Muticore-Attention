#include "golem_attention_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <sched.h>

#ifndef GOLEM_ATTENTION_Q_ADDR
#define GOLEM_ATTENTION_Q_ADDR 0x0A000000ull
#endif
#ifndef GOLEM_ATTENTION_K_ADDR
#define GOLEM_ATTENTION_K_ADDR 0x0A010000ull
#endif
#ifndef GOLEM_ATTENTION_V_ADDR
#define GOLEM_ATTENTION_V_ADDR 0x0A020000ull
#endif
#ifndef GOLEM_ATTENTION_O_ADDR
#define GOLEM_ATTENTION_O_ADDR 0x0A030000ull
#endif
#ifndef GOLEM_ATTENTION_QUERY_LENGTH
#ifdef GOLEM_ATTENTION_QUERIES
#define GOLEM_ATTENTION_QUERY_LENGTH GOLEM_ATTENTION_QUERIES
#else
#define GOLEM_ATTENTION_QUERY_LENGTH 32u
#endif
#endif
#ifndef GOLEM_ATTENTION_KV_LENGTH
#ifdef GOLEM_ATTENTION_KEYS
#define GOLEM_ATTENTION_KV_LENGTH GOLEM_ATTENTION_KEYS
#else
#define GOLEM_ATTENTION_KV_LENGTH 32u
#endif
#endif
#ifndef GOLEM_ATTENTION_CAUSAL
#define GOLEM_ATTENTION_CAUSAL 0
#endif
#ifndef GOLEM_ATTENTION_SCALE
#define GOLEM_ATTENTION_SCALE 0
#endif
#ifndef GOLEM_ATTENTION_HEAD_DIM
#define GOLEM_ATTENTION_HEAD_DIM 64u
#endif
#ifndef GOLEM_ATTENTION_MEM_NODE_BYTES
#define GOLEM_ATTENTION_MEM_NODE_BYTES 0x08000000ull
#endif
#ifndef GOLEM_ATTENTION_Q_OFFSET
#define GOLEM_ATTENTION_Q_OFFSET 0x02000000ull
#endif
#ifndef GOLEM_ATTENTION_K_OFFSET
#define GOLEM_ATTENTION_K_OFFSET 0x02100000ull
#endif
#ifndef GOLEM_ATTENTION_V_OFFSET
#define GOLEM_ATTENTION_V_OFFSET 0x02200000ull
#endif
#ifndef GOLEM_ATTENTION_O_OFFSET
#define GOLEM_ATTENTION_O_OFFSET 0x02300000ull
#endif
#ifndef GOLEM_GLOBAL_STRIDE_BYTES
#define GOLEM_GLOBAL_STRIDE_BYTES 0x00200000ull
#endif

static bool parse_positive_u32(const char* text, uint32_t* value) {
    if (text == nullptr || value == nullptr || *text == '\0') return false;
    uint32_t parsed = 0;
    for (const char* cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') return false;
        const uint32_t digit = static_cast<uint32_t>(*cursor - '0');
        if (parsed > (UINT32_MAX - digit) / 10) return false;
        parsed = parsed * 10 + digit;
    }
    if (parsed == 0) return false;
    *value = parsed;
    return true;
}

static bool parse_u64(const char* text, uint64_t* value) {
    if (text == nullptr || value == nullptr || *text == '\0') return false;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 0);
    if (end == text || *end != '\0') return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
}

int main(int argc, char** argv) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= (GOLEM_ATTENTION_SCALE ? 4 : 1)) return 0;
    const uint32_t manager_id = static_cast<uint32_t>(core_id);
    uint32_t manager_query_rows = GOLEM_ATTENTION_QUERY_LENGTH;
    uint32_t kv_length = GOLEM_ATTENTION_KV_LENGTH;
    uint32_t num_query_heads = 1;
    uint32_t num_kv_heads = 1;
    uint32_t head_dim = GOLEM_ATTENTION_HEAD_DIM;
    uint32_t kv_tile_rows = 32;
    uint64_t q_offset = GOLEM_ATTENTION_Q_OFFSET;
    uint64_t k_offset = GOLEM_ATTENTION_K_OFFSET;
    uint64_t v_offset = GOLEM_ATTENTION_V_OFFSET;
    uint64_t o_offset = GOLEM_ATTENTION_O_OFFSET;
    if (GOLEM_ATTENTION_SCALE && argc != 12) {
        std::fprintf(stderr, "Attention guest expects core-id, manager-query-rows, kv-length, num-query-heads, num-kv-heads, head-dim, kv-tile-rows, q/k/v/o-offsets\n");
        return 1;
    }
    if (GOLEM_ATTENTION_SCALE &&
        (!parse_positive_u32(argv[2], &manager_query_rows) ||
         !parse_positive_u32(argv[3], &kv_length) ||
         !parse_positive_u32(argv[4], &num_query_heads) ||
         !parse_positive_u32(argv[5], &num_kv_heads) ||
         !parse_positive_u32(argv[6], &head_dim) ||
         !parse_positive_u32(argv[7], &kv_tile_rows) ||
         !parse_u64(argv[8], &q_offset) ||
         !parse_u64(argv[9], &k_offset) ||
         !parse_u64(argv[10], &v_offset) ||
         !parse_u64(argv[11], &o_offset) ||
         num_query_heads % num_kv_heads != 0 ||
         (num_query_heads / num_kv_heads != 1 && num_query_heads / num_kv_heads != 2 &&
          num_query_heads / num_kv_heads != 4) ||
         (kv_tile_rows != 32 && kv_tile_rows != 64))) {
        std::fprintf(stderr, "Invalid Attention guest dimensions\n");
        return 1;
    }

    const uint64_t managerGmBase = manager_id * GOLEM_GLOBAL_STRIDE_BYTES;
    const uint64_t topology_gm = managerGmBase + 0x1100;
    const uint64_t desc_gm_base = managerGmBase + 0x2000;

    SFUWorkerTopologyMapV1 topology = {};
    topology.magic = SFU_WORKER_TOPOLOGY_MAP_MAGIC;
    topology.version = SFU_WORKER_TOPOLOGY_MAP_VERSION;
    topology.size_bytes = sizeof(topology);
    topology.worker_count = GOLEM_ATTENTION_SCALE ? 4 : 1;
    if (GOLEM_ATTENTION_SCALE) {
        topology.worker_core_ids[0] = 4 + manager_id;
        topology.worker_core_ids[1] = 8 + manager_id;
        topology.worker_core_ids[2] = 12 + manager_id;
        topology.worker_core_ids[3] = 16 + manager_id;
    } else {
        topology.worker_core_ids[0] = 1;
    }

    const uint64_t managerNodeBase =
        static_cast<uint64_t>(manager_id + 1) * GOLEM_ATTENTION_MEM_NODE_BYTES;
    const uint64_t firstDataNodeBase = GOLEM_ATTENTION_MEM_NODE_BYTES;
    attention_write_metadata(topology_gm, topology);
    const uint64_t query_head_stride =
        static_cast<uint64_t>(manager_query_rows) * head_dim * sizeof(float);
    const uint64_t kv_head_stride =
        static_cast<uint64_t>(kv_length / (GOLEM_ATTENTION_SCALE ? 4u : 1u)) *
        head_dim * sizeof(float);
    const char* sequential64 = std::getenv("GOLEM_ATTENTION_SEQUENTIAL_64_ENABLE");
    const uint32_t gqa_group_size = num_query_heads / num_kv_heads;
    for (uint32_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
        const uint32_t first_query_head = kv_head * gqa_group_size;
        const uint64_t job_id = 0xD1000001ull + kv_length +
            (static_cast<uint64_t>(kv_head) << 32);
        const uint64_t tag = 0xD1000101ull + kv_length +
            (static_cast<uint64_t>(kv_head) << 32);
        GolemAttentionDescV2 desc = {};
        desc.magic = GOLEM_ATTENTION_DESC_MAGIC;
        desc.version = GOLEM_ATTENTION_DESC_VERSION;
        desc.size_bytes = sizeof(desc);
        desc.job_id = job_id;
        desc.q_addr = GOLEM_ATTENTION_SCALE ?
            managerNodeBase + q_offset + first_query_head * query_head_stride :
            GOLEM_ATTENTION_Q_ADDR + first_query_head * query_head_stride;
        desc.k_addr = GOLEM_ATTENTION_SCALE ?
            firstDataNodeBase + k_offset + kv_head * kv_head_stride :
            GOLEM_ATTENTION_K_ADDR + kv_head * kv_head_stride;
        desc.v_addr = GOLEM_ATTENTION_SCALE ?
            firstDataNodeBase + v_offset + kv_head * kv_head_stride :
            GOLEM_ATTENTION_V_ADDR + kv_head * kv_head_stride;
        desc.output_addr = GOLEM_ATTENTION_SCALE ?
            managerNodeBase + o_offset : GOLEM_ATTENTION_O_ADDR;
        desc.topology_gm_addr = topology_gm;
        desc.group_query_rows = manager_query_rows * gqa_group_size;
        desc.kv_length = kv_length;
        desc.head_dim = head_dim;
        desc.query_tile_rows =
            sequential64 && std::atoi(sequential64) != 0 ? 64 : 16;
        desc.kv_tile_rows = kv_tile_rows;
        desc.worker_count = topology.worker_count;
        desc.flags = GOLEM_ATTENTION_CAUSAL ? GOLEM_ATTENTION_FLAG_CAUSAL : 0;
        desc.tensor_root_core = 0;
        desc.tensor_manager_slot = manager_id;
        desc.tensor_manager_count = GOLEM_ATTENTION_SCALE ? 4 : 1;
        if (GOLEM_ATTENTION_SCALE) {
            desc.group_query_row_begin = manager_id * desc.group_query_rows;
            desc.kv_rows_per_memory_node = kv_length / 4;
            desc.kv_node_stride_bytes = GOLEM_ATTENTION_MEM_NODE_BYTES;
        }
        desc.num_query_heads = num_query_heads;
        desc.num_kv_heads = num_kv_heads;
        desc.kv_head_index = kv_head;

        const uint64_t desc_gm = desc_gm_base +
            static_cast<uint64_t>(kv_head) * sizeof(GolemAttentionDescV2);
        attention_write_metadata(desc_gm, desc);
    }

    for (uint32_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
        const uint64_t tag = 0xD1000101ull + kv_length +
            (static_cast<uint64_t>(kv_head) << 32);
        const uint64_t desc_gm = desc_gm_base +
            static_cast<uint64_t>(kv_head) * sizeof(GolemAttentionDescV2);
        attention_manager_job(desc_gm, tag);
    }
    uint64_t final_status = 0;
    for (uint32_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
        const uint64_t tag = 0xD1000101ull + kv_length +
            (static_cast<uint64_t>(kv_head) << 32);
        const uint64_t status = attention_manager_wait(tag);
        if (status != 0) {
            final_status = status;
            break;
        }
    }
    std::printf("FUSED_ATTENTION status=%llu job=%llu manager=%u num_query_heads=%u num_kv_heads=%u manager_query_rows=%u kv_length=%u causal=%u\n",
                static_cast<unsigned long long>(final_status),
                static_cast<unsigned long long>(0xD1000001ull + kv_length),
                manager_id, num_query_heads, num_kv_heads, manager_query_rows, kv_length,
                static_cast<unsigned>(GOLEM_ATTENTION_CAUSAL));
    return final_status == 0 ? 0 : 1;
}
