#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sched.h>

#include "../mvm_noc_int_array/ex_instr.h"

#ifndef GOLEM_SFU_RMSNORM_ROWS
#define GOLEM_SFU_RMSNORM_ROWS 16u
#endif
#ifndef GOLEM_SFU_RMSNORM_COLS
#define GOLEM_SFU_RMSNORM_COLS 128u
#endif
#ifndef GOLEM_SFU_RMSNORM_EPSILON
#define GOLEM_SFU_RMSNORM_EPSILON 1.0e-6f
#endif

struct SFUJobDesc {
    uint64_t job_id;
    uint64_t input0_addr;
    uint64_t input1_addr;
    uint64_t output_addr;
    uint64_t params_addr;
    uint64_t scratch_addr;
    uint32_t op_type;
    uint32_t sub_op;
    uint32_t dtype;
    uint32_t layout;
    uint32_t rows;
    uint32_t cols;
    uint32_t elem_count;
    uint32_t chunk_elems;
    uint32_t worker_cores;
    uint32_t owner_core;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t reserved1;
    uint64_t reserved2;
    uint64_t reserved3;
    uint64_t reserved4;
};

static_assert(sizeof(SFUJobDesc) == 128, "SFU job ABI mismatch");

int main() {
    if (sched_getcpu() != 0) return 0;
    constexpr uint64_t nodeBase = 0x08000000ull;
    constexpr uint64_t tag = 0x524d5301ull;
    SFUJobDesc desc = {};
    desc.job_id = tag;
    desc.input0_addr = nodeBase + 0x02000000ull;
    desc.input1_addr = nodeBase + 0x02100000ull;
    desc.output_addr = nodeBase + 0x02200000ull;
    desc.scratch_addr = 0x8000ull;
    desc.op_type = 0x13u;
    desc.dtype = 2u;
    desc.rows = GOLEM_SFU_RMSNORM_ROWS;
    desc.cols = GOLEM_SFU_RMSNORM_COLS;
    desc.elem_count = desc.rows * desc.cols;
    desc.chunk_elems = desc.cols;
    desc.worker_cores = 1;
    desc.owner_core = 0;
    const float epsilon = GOLEM_SFU_RMSNORM_EPSILON;
    uint32_t epsilonBits = 0;
    std::memcpy(&epsilonBits, &epsilon, sizeof(epsilon));
    desc.reserved1 = epsilonBits;

    const auto* words = reinterpret_cast<const uint64_t*>(&desc);
    for (uint32_t index = 0; index < sizeof(desc) / sizeof(uint64_t); ++index) {
        reg2gm(words[index], 0x2000ull + index * sizeof(uint64_t));
    }
    asm volatile(
        ".insn r 0x0b, 7, 0x1d, x0, %0, %1"
        : : "r"(0x2000ull), "r"(tag) : "memory");
    uint64_t status = 0;
    asm volatile(
        ".insn r 0x0b, 7, 0x18, %0, %1, x0"
        : "=r"(status) : "r"(tag) : "memory");
    std::printf("[RMSNORM_GUEST] status=%llu rows=%u cols=%u\n",
                static_cast<unsigned long long>(status), desc.rows, desc.cols);
    return status == 0 ? 0 : 1;
}
