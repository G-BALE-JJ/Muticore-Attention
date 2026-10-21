#include "golem_attention_runtime.h"
#include "golem_projection_runtime.h"
#include "golem_decoder_runtime.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#define GOLEM_ATTENTION_MEM_NODE_BYTES SST::Golem::DecoderMemoryLayout::node_stride
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
#define GOLEM_GLOBAL_STRIDE_BYTES SST::Golem::DecoderMemoryLayout::gm_stride
#endif
#ifndef GOLEM_PROJECTION_BATCH_RELEASE
#define GOLEM_PROJECTION_BATCH_RELEASE 0
#endif
#if GOLEM_PROJECTION_BATCH_RELEASE != 0 && GOLEM_PROJECTION_BATCH_RELEASE != 1
#error "GOLEM_PROJECTION_BATCH_RELEASE must be 0 or 1"
#endif

#ifndef GOLEM_DECODER_LAYERS
#define GOLEM_DECODER_LAYERS 0
#endif
#ifndef GOLEM_DECODER_FINAL_NORM
#define GOLEM_DECODER_FINAL_NORM 0
#endif
#ifndef GOLEM_DECODER_LM_HEAD
#define GOLEM_DECODER_LM_HEAD 0
#endif
#ifndef GOLEM_DECODER_EMBEDDING
#define GOLEM_DECODER_EMBEDDING 0
#endif
#if GOLEM_DECODER_EMBEDDING && !GOLEM_DECODER_LM_HEAD
#error "embedding checkpoint requires the full prefill path"
#endif
#if GOLEM_DECODER_FINAL_NORM && (!GOLEM_DECODER_LAYERS || GOLEM_DECODER_STAGE != 4)
#error "final RMSNorm requires a complete decoder stack"
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
#if GOLEM_GUEST_STARTUP_DIAGNOSTICS
    std::fprintf(stderr, "[ATTENTION_GUEST_START] core=%d argc=%d\n", core_id, argc);
#endif
    if (core_id < 0 || core_id >= (GOLEM_ATTENTION_SCALE ? 20 : 1)) return 0;
    const bool projection_worker = GOLEM_ATTENTION_SCALE && core_id >= 4;
    const uint32_t manager_id = projection_worker ? (core_id - 4) % 4 : core_id;
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
    if (GOLEM_ATTENTION_SCALE && argc != 12 && argc != 13 && argc != 14 && argc != 15 && argc != 16) {
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

    const char* dtype = argc >= 13 ? argv[12] : std::getenv("GOLEM_ATTENTION_DTYPE");
    if (dtype && std::strcmp(dtype, "fp16") != 0 && std::strcmp(dtype, "fp32") != 0) return 1;
    const bool causal = argc >= 14 ? std::strcmp(argv[13], "1") == 0 :
        GOLEM_ATTENTION_CAUSAL != 0;
    if (argc >= 14 && std::strcmp(argv[13], "0") != 0 && !causal) return 1;
    const bool rope = argc >= 15 && std::strcmp(argv[14], "1") == 0;
    if (argc >= 15 && std::strcmp(argv[14], "0") != 0 && !rope) return 1;
    const bool projection = argc >= 16 && std::strcmp(argv[15], "1") == 0;
    if (argc >= 16 && std::strcmp(argv[15], "0") != 0 && !projection) return 1;
    uint32_t layer_count=GOLEM_DECODER_LAYERS ? GOLEM_DECODER_LAYERS : 1;
    SST::Golem::DecoderMemoryLayout stack_layout;
    const SST::Golem::DecoderMemoryLayout* layout=nullptr;
    if (GOLEM_DECODER_LAYERS) {
        const char* configured=std::getenv("GOLEM_DECODER_LAYERS");
        if (!projection || !parse_positive_u32(configured,&layer_count) || layer_count>32) return 1;
        const char* vocab_env=std::getenv("GOLEM_DECODER_VOCAB");
        const char* stride_env=std::getenv("GOLEM_DECODER_NODE_BYTES");
        const uint32_t vocab=GOLEM_DECODER_LM_HEAD && vocab_env ? std::strtoul(vocab_env,nullptr,10):0;
        const uint64_t node_bytes=stride_env ? std::strtoull(stride_env,nullptr,10):SST::Golem::DecoderMemoryLayout::node_stride;
        if (GOLEM_DECODER_LM_HEAD && !vocab) return 1;
        const char* retain_env=std::getenv("GOLEM_DECODER_RETAIN_OUTPUTS");
        const bool retain=retain_env && std::strcmp(retain_env,"1")==0;
        stack_layout=SST::Golem::makeDecoderMemoryLayout(kv_length,num_query_heads,num_kv_heads,head_dim,layer_count,GOLEM_DECODER_FINAL_NORM,vocab,node_bytes,GOLEM_DECODER_EMBEDDING,retain);
        if (!stack_layout.valid || manager_query_rows!=kv_length/4) return 1;
        layout=&stack_layout;
        q_offset=layout->q; k_offset=layout->k; v_offset=layout->v; o_offset=layout->o;
    }
    if (projection_worker) {
        if (!projection) return 0;
#if GOLEM_DECODER_EMBEDDING
        while (gm2reg(uint64_t(core_id)*layout->gm_stride+layout->embedding_release)!=1) asm volatile("nop");
        if (!run_embedding(core_id,manager_query_rows,num_query_heads*head_dim,*layout)) return 1;
#endif
        for (uint32_t layer=0;layer<layer_count;++layer) {
        const uint64_t epoch=layer+1;
        const uint64_t release = static_cast<uint64_t>(core_id) * GOLEM_GLOBAL_STRIDE_BYTES + SST::Golem::DecoderMemoryLayout::projection_release;
        while (gm2reg(release) != epoch) asm volatile("nop");
        if (!run_projection(manager_id, manager_query_rows, kv_length,
                num_query_heads, num_kv_heads, head_dim, q_offset, k_offset, v_offset,
                causal, rope, core_id, layout, layer) || !finish_projection_worker()) {
            std::fprintf(stderr, "[PROJECTION_GUEST_FAIL] worker=%d manager=%u\n", core_id, manager_id);
            return 1;
        }
#if GOLEM_DECODER_STAGE
        const uint64_t decoder_release = static_cast<uint64_t>(core_id) * GOLEM_GLOBAL_STRIDE_BYTES + SST::Golem::DecoderMemoryLayout::decoder_release;
        while (gm2reg(decoder_release) != epoch) asm volatile("nop");
        if (!run_decoder(core_id, manager_query_rows, num_query_heads * head_dim, head_dim, o_offset, layout, layer)) {
            std::fprintf(stderr, "[DECODER_GUEST_FAIL] worker=%u\n", core_id);
            return 1;
        }
#endif
        }
#if GOLEM_DECODER_FINAL_NORM
        const uint64_t release=uint64_t(core_id)*layout->gm_stride+layout->final_norm_release;
        while (gm2reg(release)!=layer_count+1) asm volatile("nop");
        if (!run_final_norm(core_id,manager_query_rows,num_query_heads*head_dim,*layout)) return 1;
#endif
#if GOLEM_DECODER_LM_HEAD
        while (gm2reg(uint64_t(core_id)*layout->gm_stride+layout->lm_head_release)!=layer_count+2) asm volatile("nop");
        if (!run_lm_head(core_id,num_query_heads*head_dim,*layout)) return 1;
#endif
        return 0;
    }
#if GOLEM_DECODER_EMBEDDING
    if (manager_id==0) {
        for (uint32_t node=1;node<=4;++node) for (uint32_t lane=0;lane<4;++lane) {
            const uint64_t flag=node*layout->hbm_node_stride+layout->embedding_done+lane*8;
            uint64_t status;
            asm volatile(".insn r 0x0b, 7, 0x28, %0, %1, %2" : "=r"(status) : "r"(flag), "r"(0ull) : "memory");
            if (status) return 1;
        }
        reg2gm(1,layout->embedding_release);
        for (uint32_t worker=4;worker<20;++worker)
            remote_store(layout->embedding_release,uint64_t(worker)*layout->gm_stride+layout->embedding_release);
        for (uint32_t node=1;node<=4;++node) for (uint32_t lane=0;lane<4;++lane) {
            const uint64_t flag=node*layout->hbm_node_stride+layout->embedding_done+lane*8;
            uint64_t status;
            asm volatile(".insn r 0x0b, 7, 0x25, %0, %1, %2" : "=r"(status) : "r"(flag), "r"(1ull) : "memory");
            if (status) return 1;
        }
        std::printf("DECODER_EMBEDDING status=0 tokens=%u hidden=%u vocab=%u\n",kv_length,num_query_heads*head_dim,layout->vocab);
    }
#endif
    for (uint32_t layer=0;layer<layer_count;++layer) {
    const uint64_t epoch=layer+1;
    if (layout) {
        if (manager_id==0) {
            // Previous layer wait covers every worker output and flag ACK.
            // Clear both flag sets before any manager releases new work.
            const char* epoch_flags=std::getenv("GOLEM_DECODER_EPOCH_FLAGS");
            // The first generation initializes poisoned flags. Subsequent
            // generations use distinct exact-match epochs, and the root has
            // already waited for every preceding output/completion ACK.
            if(layer==0 || !epoch_flags || std::strcmp(epoch_flags,"1")!=0)
            for (uint32_t node=1;node<=4;++node)
            for (uint64_t offset : {layout->projection_done,layout->decoder_done})
            for (uint32_t lane=0;lane<4;++lane) {
                const uint64_t address=node*layout->hbm_node_stride+offset+lane*8;
                uint64_t status;
                asm volatile(".insn r 0x0b, 7, 0x28, %0, %1, %2" :
                    "=r"(status) : "r"(address), "r"(0ull) : "memory");
                if (status) return 1;
            }
            reg2gm(epoch,layout->layer_release);
            for (uint32_t manager=1;manager<4;++manager)
                remote_store(layout->layer_release,manager*layout->gm_stride+layout->layer_release);
        } else {
            while (gm2reg(manager_id*layout->gm_stride+layout->layer_release)!=epoch) asm volatile("nop");
        }
        std::printf("DECODER_STACK event=layer_begin layer=%u manager=%u input_offset=%llu output_offset=%llu\n",
            layer+1,manager_id,(unsigned long long)layout->layers[layer].input,(unsigned long long)layout->layers[layer].output);
    }
    if (projection) {
        reg2gm(epoch, static_cast<uint64_t>(manager_id) * GOLEM_GLOBAL_STRIDE_BYTES + SST::Golem::DecoderMemoryLayout::projection_release);
        for (uint32_t lane = 0; lane < 4; ++lane) {
            const uint32_t worker = 4 + lane * 4 + manager_id;
            remote_store(static_cast<uint64_t>(manager_id) * GOLEM_GLOBAL_STRIDE_BYTES + SST::Golem::DecoderMemoryLayout::projection_release,
                         static_cast<uint64_t>(worker) * GOLEM_GLOBAL_STRIDE_BYTES + SST::Golem::DecoderMemoryLayout::projection_release);
#if !GOLEM_PROJECTION_BATCH_RELEASE && GOLEM_GUEST_STARTUP_DIAGNOSTICS
            std::fprintf(stderr, "[PROJECTION_DISPATCH] manager=%u worker=%u row_offset=%u row_count=64\n",
                         manager_id, worker, lane * 64);
#endif
        }
#if GOLEM_PROJECTION_BATCH_RELEASE && GOLEM_GUEST_STARTUP_DIAGNOSTICS
        // Keep guest logging, but release the complete group before formatting
        // its diagnostics. The four writes still use the existing control path.
        for (uint32_t lane = 0; lane < 4; ++lane) {
            const uint32_t worker = 4 + lane * 4 + manager_id;
            std::fprintf(stderr, "[PROJECTION_DISPATCH] manager=%u worker=%u row_offset=%u row_count=64\n",
                         manager_id, worker, lane * 64);
        }
#endif
    }

    const uint64_t managerGmBase = manager_id * GOLEM_GLOBAL_STRIDE_BYTES;
    const uint64_t topology_gm = managerGmBase + SST::Golem::DecoderMemoryLayout::topology;
    const uint64_t desc_gm_base = managerGmBase + SST::Golem::DecoderMemoryLayout::attention_descriptor;

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
        static_cast<uint64_t>(manager_id + 1) * (layout?layout->hbm_node_stride:GOLEM_ATTENTION_MEM_NODE_BYTES);
    const uint64_t firstDataNodeBase = layout?layout->hbm_node_stride:GOLEM_ATTENTION_MEM_NODE_BYTES;
    attention_write_metadata(topology_gm, topology);
    const char* balance_env = std::getenv("GOLEM_ATTENTION_CAUSAL_BALANCE");
    if (balance_env && std::strcmp(balance_env, "0") != 0 &&
        std::strcmp(balance_env, "1") != 0) return 1;
    const bool causal_balance = GOLEM_ATTENTION_SCALE && causal &&
        balance_env && std::strcmp(balance_env, "1") == 0;
    // One existing topology descriptor per head, within the reserved metadata
    // gap. No tensor partition, scratch allocation, or ABI change is needed.
    if (causal_balance && topology_gm +
        static_cast<uint64_t>(num_kv_heads) * sizeof(topology) > desc_gm_base) return 1;
    const uint32_t elem_bytes = dtype && std::strcmp(dtype, "fp16") == 0 ? 2u : 4u;
    if (rope && elem_bytes != 2) return 1;
    const uint64_t query_head_stride =
        static_cast<uint64_t>(manager_query_rows) * head_dim * elem_bytes;
    const uint64_t kv_head_stride =
        static_cast<uint64_t>(kv_length / (GOLEM_ATTENTION_SCALE ? 4u : 1u)) *
        head_dim * elem_bytes;
    const char* sequential64 = std::getenv("GOLEM_ATTENTION_SEQUENTIAL_64_ENABLE");
    const uint32_t gqa_group_size = num_query_heads / num_kv_heads;
    for (uint32_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
        uint64_t head_topology_gm = topology_gm;
        if (causal_balance) {
            // Adjacent heads pair complementary causal query bands. Each
            // head's mapping is a permutation, so existing job tags stay
            // unique at each worker. An odd final head keeps the old mapping.
            const uint32_t worker_group = (kv_head & 1u) ?
                3u - manager_id : manager_id;
            for (uint32_t lane = 0; lane < 4; ++lane)
                topology.worker_core_ids[lane] = 4u + lane * 4u + worker_group;
            head_topology_gm += static_cast<uint64_t>(kv_head) * sizeof(topology);
            attention_write_metadata(head_topology_gm, topology);
        }
        const uint32_t first_query_head = kv_head * gqa_group_size;
        const uint64_t job_id = 0xD1000001ull + kv_length +
            (static_cast<uint64_t>(kv_head) << 32) + (uint64_t(layer) << 16);
        const uint64_t tag = 0xD1000101ull + kv_length +
            (static_cast<uint64_t>(kv_head) << 32) + (uint64_t(layer) << 16);
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
        desc.topology_gm_addr = head_topology_gm;
        desc.group_query_rows = manager_query_rows * gqa_group_size;
        desc.kv_length = kv_length;
        desc.head_dim = head_dim;
        desc.query_tile_rows =
            sequential64 && std::atoi(sequential64) != 0 ? 64 : 16;
        desc.kv_tile_rows = kv_tile_rows;
        desc.worker_count = topology.worker_count;
        desc.flags = (causal ? GOLEM_ATTENTION_FLAG_CAUSAL : 0) |
            (rope && !(projection && std::getenv("GOLEM_FUSION_PROJECTION_ROPE") &&
                       std::strcmp(std::getenv("GOLEM_FUSION_PROJECTION_ROPE"), "1") == 0)
                ? GOLEM_ATTENTION_FLAG_ROPE : 0);
        desc.tensor_root_core = 0;
        desc.tensor_manager_slot = manager_id;
        desc.tensor_manager_count = GOLEM_ATTENTION_SCALE ? 4 : 1;
        if (GOLEM_ATTENTION_SCALE) {
            desc.group_query_row_begin = manager_id * desc.group_query_rows;
            desc.kv_rows_per_memory_node = kv_length / 4;
            desc.kv_node_stride_bytes = layout?layout->hbm_node_stride:GOLEM_ATTENTION_MEM_NODE_BYTES;
        }
        desc.num_query_heads = num_query_heads;
        desc.num_kv_heads = num_kv_heads;
        desc.kv_head_index = kv_head;

        const uint64_t desc_gm = desc_gm_base +
            static_cast<uint64_t>(kv_head) * sizeof(GolemAttentionDescV2);
        attention_write_metadata(desc_gm, desc);
    }

    if (projection && !finish_projection(manager_id,layout,epoch)) {
        std::fprintf(stderr, "[PROJECTION_GUEST_FAIL] manager=%u\n", manager_id);
        return 1;
    }
    for (uint32_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
        const uint64_t tag = 0xD1000101ull + kv_length +
            (static_cast<uint64_t>(kv_head) << 32) + (uint64_t(layer) << 16);
        const uint64_t desc_gm = desc_gm_base +
            static_cast<uint64_t>(kv_head) * sizeof(GolemAttentionDescV2);
        attention_manager_job(desc_gm, tag);
    }
    uint64_t final_status = 0;
    for (uint32_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
        const uint64_t tag = 0xD1000101ull + kv_length +
            (static_cast<uint64_t>(kv_head) << 32) + (uint64_t(layer) << 16);
        const uint64_t status = attention_manager_wait(tag);
        if (status != 0) {
            final_status = status;
            break;
        }
    }
    std::printf("FUSED_ATTENTION status=%llu job=%llu manager=%u num_query_heads=%u num_kv_heads=%u manager_query_rows=%u kv_length=%u causal=%u rope=%u\n",
                static_cast<unsigned long long>(final_status),
                static_cast<unsigned long long>(0xD1000001ull + kv_length),
                manager_id, num_query_heads, num_kv_heads, manager_query_rows, kv_length,
                static_cast<unsigned>(causal), static_cast<unsigned>(rope));
#if GOLEM_DECODER_STAGE
    if (manager_id == 0 && final_status == 0) {
        // Root attention wait includes all head writes and global completion.
        reg2gm(epoch, SST::Golem::DecoderMemoryLayout::decoder_release);
        for (uint32_t worker=4; worker<20; ++worker)
            remote_store(SST::Golem::DecoderMemoryLayout::decoder_release, uint64_t(worker)*GOLEM_GLOBAL_STRIDE_BYTES+SST::Golem::DecoderMemoryLayout::decoder_release);
        for (uint32_t manager=0; manager<4; ++manager) for (uint32_t lane=0; lane<4; ++lane) {
            const uint64_t flag=uint64_t(manager+1)*(layout?layout->hbm_node_stride:SST::Golem::DecoderMemoryLayout::node_stride)+(layout?layout->decoder_done:SST::Golem::DEC_DONE)+lane*8;
            uint64_t status;
            asm volatile(".insn r 0x0b, 7, 0x25, %0, %1, %2" : "=r"(status) : "r"(flag), "r"(epoch) : "memory");
            if (status) final_status=status;
        }
        std::printf("DECODER_LAYER status=%llu stage=%u hidden=%u intermediate=%u layer=%u\n",
            (unsigned long long)final_status,GOLEM_DECODER_STAGE,num_query_heads*head_dim,4*num_query_heads*head_dim,layer+1);
    }
#endif
    if (final_status) return 1;
    }
#if GOLEM_DECODER_FINAL_NORM
    if (manager_id==0) {
        // Layer completion above includes all output and completion DMA ACKs.
        for (uint32_t node=1;node<=4;++node) for (uint32_t lane=0;lane<4;++lane) {
            const uint64_t flag=node*layout->hbm_node_stride+layout->final_done+lane*8;
            uint64_t status;
            asm volatile(".insn r 0x0b, 7, 0x28, %0, %1, %2" : "=r"(status) : "r"(flag), "r"(0ull) : "memory");
            if (status) return 1;
        }
        reg2gm(layer_count+1,layout->final_norm_release);
        for (uint32_t worker=4;worker<20;++worker)
            remote_store(layout->final_norm_release,uint64_t(worker)*layout->gm_stride+layout->final_norm_release);
        for (uint32_t node=1;node<=4;++node) for (uint32_t lane=0;lane<4;++lane) {
            const uint64_t flag=node*layout->hbm_node_stride+layout->final_done+lane*8;
            uint64_t status;
            asm volatile(".insn r 0x0b, 7, 0x25, %0, %1, %2" : "=r"(status) : "r"(flag), "r"(uint64_t(layer_count+1)) : "memory");
            if (status) return 1;
        }
        std::printf("DECODER_FINAL_NORM status=0 hidden=%u epoch=%u\n",num_query_heads*head_dim,layer_count+1);
#if GOLEM_DECODER_LM_HEAD
        for (uint32_t node=1;node<=4;++node) for (uint32_t lane=0;lane<4;++lane) {
            const uint64_t flag=node*layout->hbm_node_stride+layout->lm_done+lane*8;
            uint64_t status;
            asm volatile(".insn r 0x0b, 7, 0x28, %0, %1, %2" : "=r"(status) : "r"(flag), "r"(0ull) : "memory");
            if (status) return 1;
        }
        reg2gm(layer_count+2,layout->lm_head_release);
        for (uint32_t worker=4;worker<20;++worker)
            remote_store(layout->lm_head_release,uint64_t(worker)*layout->gm_stride+layout->lm_head_release);
        const uint64_t completed_epoch=layer_count+2+(layout->vocab+8191)/8192-1;
        for (uint32_t node=1;node<=4;++node) for (uint32_t lane=0;lane<4;++lane) {
            const uint64_t flag=node*layout->hbm_node_stride+layout->lm_done+lane*8;
            uint64_t status;
            asm volatile(".insn r 0x0b, 7, 0x25, %0, %1, %2" : "=r"(status) : "r"(flag), "r"(completed_epoch) : "memory");
            if (status) return 1;
        }
        const char* logical_env=std::getenv("GOLEM_DECODER_LOGICAL_SEQUENCE");
        uint32_t logical=kv_length;
        if (logical_env && (!parse_positive_u32(logical_env,&logical) || logical>kv_length)) return 1;
        std::printf("DECODER_LM_HEAD status=0 vocab=%u epoch=%llu token=%u\n",layout->vocab,(unsigned long long)completed_epoch,logical-1);
#endif
    }
#endif
    return 0;
}
