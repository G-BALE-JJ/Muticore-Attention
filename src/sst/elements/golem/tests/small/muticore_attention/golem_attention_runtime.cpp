#include "golem_attention_runtime.h"

#include <cstdio>
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
#ifndef GOLEM_ATTENTION_QUERIES
#define GOLEM_ATTENTION_QUERIES 32u
#endif
#ifndef GOLEM_ATTENTION_KEYS
#define GOLEM_ATTENTION_KEYS 32u
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
#ifndef GOLEM_ATTENTION_GM_STRIDE
#define GOLEM_ATTENTION_GM_STRIDE 0x00100000ull
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

int main(int argc, char** argv) {
    const int core_id = sched_getcpu();
    if (core_id < 0 || core_id >= (GOLEM_ATTENTION_SCALE ? 4 : 1)) return 0;
    const uint32_t manager_id = static_cast<uint32_t>(core_id);
    uint32_t manager_queries = GOLEM_ATTENTION_QUERIES;
    uint32_t keys = GOLEM_ATTENTION_KEYS;
    uint32_t head_dim = GOLEM_ATTENTION_HEAD_DIM;
    uint32_t key_block_rows = 32;
    if (GOLEM_ATTENTION_SCALE && argc != 6) {
        std::fprintf(stderr, "Attention guest expects core-id, manager-queries, keys, head-dim, key-block-rows\n");
        return 1;
    }
    if (GOLEM_ATTENTION_SCALE &&
        (!parse_positive_u32(argv[2], &manager_queries) ||
         !parse_positive_u32(argv[3], &keys) ||
         !parse_positive_u32(argv[4], &head_dim) ||
         !parse_positive_u32(argv[5], &key_block_rows) ||
         (key_block_rows != 32 && key_block_rows != 64))) {
        std::fprintf(stderr, "Invalid Attention guest dimensions\n");
        return 1;
    }

    const uint64_t managerGmBase = manager_id * GOLEM_ATTENTION_GM_STRIDE;
    const uint64_t desc_gm = managerGmBase + 0x1000;
    const uint64_t topology_gm = managerGmBase + 0x1100;
    const uint64_t job_id = 0xD1000001ull + keys;
    const uint64_t tag = 0xD1000101ull + keys;

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

    GolemAttentionDescV1 desc = {};
    desc.magic = GOLEM_ATTENTION_DESC_MAGIC;
    desc.version = GOLEM_ATTENTION_DESC_VERSION;
    desc.size_bytes = sizeof(desc);
    desc.job_id = job_id;
    const uint64_t managerNodeBase =
        static_cast<uint64_t>(manager_id + 1) * GOLEM_ATTENTION_MEM_NODE_BYTES;
    const uint64_t firstDataNodeBase = GOLEM_ATTENTION_MEM_NODE_BYTES;
    desc.q_addr = GOLEM_ATTENTION_SCALE ? managerNodeBase + GOLEM_ATTENTION_Q_OFFSET :
        GOLEM_ATTENTION_Q_ADDR;
    desc.k_addr = GOLEM_ATTENTION_SCALE ? firstDataNodeBase + GOLEM_ATTENTION_K_OFFSET :
        GOLEM_ATTENTION_K_ADDR;
    desc.v_addr = GOLEM_ATTENTION_SCALE ? firstDataNodeBase + GOLEM_ATTENTION_V_OFFSET :
        GOLEM_ATTENTION_V_ADDR;
    desc.output_addr = GOLEM_ATTENTION_SCALE ?
        managerNodeBase + GOLEM_ATTENTION_O_OFFSET : GOLEM_ATTENTION_O_ADDR;
    desc.topology_gm_addr = topology_gm;
    desc.queries = manager_queries;
    desc.keys = keys;
    desc.head_dim = head_dim;
    desc.query_block_rows = 16;
    desc.key_block_rows = key_block_rows;
    desc.worker_count = topology.worker_count;
    desc.flags = GOLEM_ATTENTION_CAUSAL ? GOLEM_ATTENTION_FLAG_CAUSAL : 0;
    desc.tensor_root_core = 0;
    desc.tensor_manager_slot = manager_id;
    desc.tensor_manager_count = GOLEM_ATTENTION_SCALE ? 4 : 1;
    if (GOLEM_ATTENTION_SCALE) {
        desc.query_row_begin = manager_id * manager_queries;
        desc.kv_rows_per_node = keys / 4;
        desc.kv_node_stride_bytes = GOLEM_ATTENTION_MEM_NODE_BYTES;
    }

    attention_write_metadata(topology_gm, topology);
    attention_write_metadata(desc_gm, desc);
    attention_manager_job(desc_gm, tag);
    const uint64_t status = attention_manager_wait(tag);
    std::printf("FUSED_ATTENTION status=%llu job=%llu manager=%u queries=%u keys=%u causal=%u\n",
                static_cast<unsigned long long>(status),
                static_cast<unsigned long long>(job_id),
                manager_id,
                manager_queries, keys,
                static_cast<unsigned>(GOLEM_ATTENTION_CAUSAL));
    return status == 0 ? 0 : 1;
}
