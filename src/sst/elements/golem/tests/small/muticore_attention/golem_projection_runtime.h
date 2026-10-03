#ifndef GOLEM_PROJECTION_RUNTIME_H
#define GOLEM_PROJECTION_RUNTIME_H

#include "../mvm_noc_int_array/ex_instr.h"
#include "../../../attention/projectionJobAbi.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

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
                           bool causal, bool rope) {
    std::fprintf(stderr, "[PROJECTION_START] manager=%u rows=%u sequence=%u heads=%u/%u dim=%u causal=%u rope=%u\n",
                 manager, rows, sequence, hq, hkv, dim, causal, rope);
    if (!causal || !rope || rows != sequence / 4 || sequence % 1024 ||
        dim == 0 || dim % 64 || hq == 0 || hkv == 0) return false;
    const uint32_t hidden = hq * dim;
    if (hidden == 0 || hidden > 2048 || hidden % 64) return false;
    constexpr uint64_t nodeStride = 0x08000000ull;
    constexpr uint64_t gmStride = 0x00200000ull;
    constexpr uint64_t normOffset = 0x01100000ull;
    const uint64_t node = (manager + 1) * nodeStride;
    const uint64_t gm = manager * gmStride;
    const uint32_t batch = 8192 / hidden;
    const uint64_t normBytes = static_cast<uint64_t>(rows) * hidden * 2;
    const uint64_t weightsOffset = (normOffset + normBytes + 0xfffffull) & ~0xfffffull;
    for (uint32_t first = 0; first < rows; first += batch) {
        ProjectionSfuJobDesc norm = {};
        norm.job_id = 0x524d0000ull + manager * 4096 + first;
        norm.input0_addr = node + first * hidden * 2;
        norm.input1_addr = node + 0x01000000ull;
        norm.output_addr = node + normOffset + first * hidden * 2;
        norm.scratch_addr = gm + 0x40000;
        norm.op_type = 0x13;
        norm.dtype = 2;
        norm.rows = std::min(batch, rows - first);
        norm.cols = hidden;
        norm.elem_count = norm.rows * hidden;
        norm.chunk_elems = hidden;
        norm.worker_cores = 1;
        norm.owner_core = manager;
        const float epsilon = 1.0e-6f;
        uint32_t epsilon_bits;
        std::memcpy(&epsilon_bits, &epsilon, 4);
        norm.reserved1 = epsilon_bits;
        const auto* words = reinterpret_cast<const uint64_t*>(&norm);
        for (uint32_t word = 0; word < sizeof(norm) / 8; ++word)
            reg2gm(words[word], gm + 0x3000 + word * 8);
        asm volatile(".insn r 0x0b, 7, 0x1d, x0, %0, %1" : :
            "r"(gm + 0x3000), "r"(norm.job_id) : "memory");
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
    job.input_addr = node + normOffset;
    job.weights_addr = node + weightsOffset;
    job.q_addr = node + q_offset;
    job.k_addr = node + k_offset;
    job.v_addr = node + v_offset;
    job.q_panel_addr = node + 0x04000000ull;
    job.k_panel_addr = node + 0x05000000ull;
    job.v_panel_addr = node + 0x06000000ull;
    job.completion_addr = node + 0x01ff0000ull;
    job.scratch_addr = gm + 0x8000;
    const auto* words = reinterpret_cast<const uint64_t*>(&job);
    for (uint32_t word = 0; word < sizeof(job) / 8; ++word)
        reg2gm(words[word], gm + 0x3000 + word * 8);
    uint64_t status;
    asm volatile(".insn r 0x0b, 7, 0x23, %0, %1, x0" :
        "=r"(status) : "r"(gm + 0x3000) : "memory");
    if (status != 0) {
        std::fprintf(stderr, "[PROJECTION_JOB_FAIL] manager=%u status=%llu\n",
            manager, static_cast<unsigned long long>(status));
        return false;
    }
    return true;
}

static bool finish_projection(uint32_t manager) {
    constexpr uint64_t nodeStride = 0x08000000ull;
    uint64_t status;
    asm volatile(".insn r 0x0b, 7, 0x24, %0, x0, x0" :
        "=r"(status) : : "memory");
    if (status != 0) {
        std::fprintf(stderr, "[PROJECTION_WAIT_FAIL] manager=%u status=%llu\n",
            manager, static_cast<unsigned long long>(status));
        return false;
    }
    for (uint32_t slot = 0; slot < 4; ++slot) {
        const uint64_t flag = (slot + 1) * nodeStride + 0x01ff0000ull;
        asm volatile(".insn r 0x0b, 7, 0x25, %0, %1, %2" :
            "=r"(status) : "r"(flag), "r"(1ull) : "memory");
        if (status != 0) return false;
    }
    return true;
}

#endif
