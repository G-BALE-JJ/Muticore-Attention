#ifndef GOLEM_PROJECTION_RUNTIME_H
#define GOLEM_PROJECTION_RUNTIME_H

#include "../mvm_noc_int_array/ex_instr.h"
#include "../../../attention/projectionJobAbi.h"
#include "../../../attention/decoderMemoryLayout.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

// Legacy traces retain guest diagnostics. Performance runs can compile them
// out without changing errors, completion reporting, or accelerator tracing.
#ifndef GOLEM_GUEST_STARTUP_DIAGNOSTICS
#define GOLEM_GUEST_STARTUP_DIAGNOSTICS 1
#endif
#if GOLEM_GUEST_STARTUP_DIAGNOSTICS != 0 && GOLEM_GUEST_STARTUP_DIAGNOSTICS != 1
#error "GOLEM_GUEST_STARTUP_DIAGNOSTICS must be 0 or 1"
#endif

struct ProjectionSfuJobDesc {
    uint64_t job_id, input0_addr, input1_addr, output_addr;
    uint64_t params_addr, scratch_addr;
    uint32_t op_type, sub_op, dtype, layout, rows, cols;
    uint32_t elem_count, chunk_elems, worker_cores, owner_core;
    uint32_t flags, reserved0;
    uint64_t reserved1, reserved2, reserved3, reserved4;
};
static_assert(sizeof(ProjectionSfuJobDesc) == 128, "SFU ABI mismatch");

static bool run_projection(uint32_t manager, uint32_t rows, uint32_t sequence,
                           uint32_t hq, uint32_t hkv, uint32_t dim,
                           uint64_t q_offset, uint64_t k_offset, uint64_t v_offset,
                           bool causal, bool rope, uint32_t worker_core,
                           const SST::Golem::DecoderMemoryLayout* layout=nullptr, uint32_t layer=0) {
#if GOLEM_GUEST_STARTUP_DIAGNOSTICS
    std::fprintf(stderr, "[PROJECTION_START] core=%u owner_manager=%u rows=%u sequence=%u heads=%u/%u dim=%u causal=%u rope=%u\n",
                 worker_core, manager, rows, sequence, hq, hkv, dim, causal, rope);
#endif
    if (!causal || !rope || rows != sequence / 4 || sequence % 1024 ||
        dim == 0 || dim % 64 || hq == 0 || hkv == 0) return false;
    const uint32_t hidden = hq * dim;
    if (hidden == 0 || hidden > 2048 || hidden % 64) return false;
    const uint64_t nodeStride = layout ? layout->hbm_node_stride : SST::Golem::DecoderMemoryLayout::node_stride;
    constexpr uint64_t gmStride = SST::Golem::DecoderMemoryLayout::gm_stride;
    const uint64_t normOffset = layout ? layout->norm1 : 0x01100000ull;
    const uint64_t inputOffset = layout ? layout->layers.at(layer).input : 0;
    const uint64_t gammaOffset = layout ? layout->layers.at(layer).gamma1 : 0x01000000ull;
    const uint64_t node = (manager + 1) * nodeStride;
    const uint64_t gm = worker_core * gmStride;
    const uint32_t lane = (worker_core - 4) / 4;
    const uint32_t batch = 8192 / hidden;
    const uint64_t normBytes = static_cast<uint64_t>(rows) * hidden * 2;
    const uint64_t weightsOffset = layout ? layout->layers.at(layer).qkv_weight :
        (normOffset + normBytes + 0xfffffull) & ~0xfffffull;
    const auto enabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value && std::strcmp(value, "1") == 0;
    };
    const bool fusedNorm = enabled("GOLEM_FUSION_NORM_PROJECTION");
    const bool fusedRope = enabled("GOLEM_FUSION_PROJECTION_ROPE");
    const bool audit = !enabled("GOLEM_FUSION_SKIP_AUDIT_OUTPUT");
    for (uint32_t block = 0; !fusedNorm && block < rows; block += 256)
    for (uint32_t first = block + lane * 64; first < block + (lane + 1) * 64; first += batch) {
        ProjectionSfuJobDesc norm = {};
        norm.job_id = 0x524d0000ull + manager * 4096 + first;
        norm.input0_addr = node + inputOffset + first * hidden * 2;
        norm.input1_addr = node + gammaOffset;
        norm.output_addr = node + normOffset + first * hidden * 2;
        norm.scratch_addr = gm + 0x40000;
        norm.op_type = 0x13;
        norm.dtype = 2;
        norm.rows = std::min(batch, block + (lane + 1) * 64 - first);
        norm.cols = hidden;
        norm.elem_count = norm.rows * hidden;
        norm.chunk_elems = hidden;
        norm.worker_cores = 1;
        norm.owner_core = worker_core;
        const float epsilon = 1.0e-6f;
        uint32_t epsilon_bits;
        std::memcpy(&epsilon_bits, &epsilon, 4);
        norm.reserved1 = epsilon_bits;
        const auto* words = reinterpret_cast<const uint64_t*>(&norm);
        for (uint32_t word = 0; word < sizeof(norm) / 8; ++word)
            reg2gm(words[word], gm + SST::Golem::DecoderMemoryLayout::descriptor + word * 8);
        asm volatile(".insn r 0x0b, 7, 0x1d, x0, %0, %1" : :
            "r"(gm + SST::Golem::DecoderMemoryLayout::descriptor), "r"(norm.job_id) : "memory");
        uint64_t status;
        asm volatile(".insn r 0x0b, 7, 0x18, %0, %1, x0" :
            "=r"(status) : "r"(norm.job_id) : "memory");
        if (status != 0) {
            std::fprintf(stderr, "[PROJECTION_RMSNORM_FAIL] manager=%u row=%u status=%llu\n",
                manager, first, static_cast<unsigned long long>(status));
            return false;
        }
    }

    SST::Golem::ProjectionJobDesc job = {};
    job.magic = SST::Golem::GOLEM_PROJECTION_JOB_MAGIC;
    job.size_bytes = sizeof(job);
    job.hidden_dim = hidden;
    job.head_dim = dim;
    job.query_heads = hq;
    job.kv_heads = hkv;
    job.rows_per_node = rows;
    job.manager_slot = manager;
    job.reserved = lane + 1;
    job.input_addr = node + (fusedNorm ? inputOffset : normOffset);
    job.gamma_addr = node + gammaOffset;
    job.norm_output_addr = node + normOffset;
    job.rope_table_addr = nodeStride + (layout ? layout->rope : 0x07000000ull);
    job.flags = (fusedNorm ? SST::Golem::GOLEM_PROJECTION_FUSE_NORM : 0) |
                (fusedRope ? SST::Golem::GOLEM_PROJECTION_FUSE_ROPE : 0) |
                (audit ? SST::Golem::GOLEM_PROJECTION_AUDIT_OUTPUT : 0);
    job.weights_addr = node + weightsOffset;
    job.q_addr = node + q_offset;
    job.k_addr = node + k_offset;
    job.v_addr = node + v_offset;
    job.q_panel_addr = node + (layout ? layout->q_panel : 0x04000000ull);
    job.k_panel_addr = node + (layout ? layout->k_panel : 0x05000000ull);
    job.v_panel_addr = node + (layout ? layout->v_panel : 0x06000000ull);
    job.completion_addr = node + (layout ? layout->projection_done : 0x01ff0000ull) + lane * 8;
    job.completion_value = layout ? layer+1 : 0;
    job.scratch_addr = gm + SST::Golem::DecoderMemoryLayout::projection_scratch;
    const auto* words = reinterpret_cast<const uint64_t*>(&job);
    for (uint32_t word = 0; word < sizeof(job) / 8; ++word)
        reg2gm(words[word], gm + SST::Golem::DecoderMemoryLayout::descriptor + word * 8);
    uint64_t status;
    asm volatile(".insn r 0x0b, 7, 0x23, %0, %1, x0" :
        "=r"(status) : "r"(gm + SST::Golem::DecoderMemoryLayout::descriptor) : "memory");
    if (status != 0) {
        std::fprintf(stderr, "[PROJECTION_JOB_FAIL] manager=%u status=%llu\n",
            manager, static_cast<unsigned long long>(status));
        return false;
    }
    return true;
}

static bool finish_projection(uint32_t manager,
        const SST::Golem::DecoderMemoryLayout* layout=nullptr,uint32_t epoch=1) {
    const uint64_t nodeStride = layout ? layout->hbm_node_stride : SST::Golem::DecoderMemoryLayout::node_stride;
    uint64_t status;
    for (uint32_t slot = 0; slot < 4; ++slot) {
      for (uint32_t lane = 0; lane < 4; ++lane) {
        const uint64_t flag = (slot + 1) * nodeStride + (layout ? layout->projection_done : 0x01ff0000ull) + lane * 8;
        asm volatile(".insn r 0x0b, 7, 0x25, %0, %1, %2" :
            "=r"(status) : "r"(flag), "r"(uint64_t(epoch)) : "memory");
        if (status != 0) return false;
      }
    }
    return true;
}

static bool finish_projection_worker() {
    uint64_t status;
    asm volatile(".insn r 0x0b, 7, 0x24, %0, x0, x0" :
        "=r"(status) : : "memory");
    return status == 0;
}

#endif
