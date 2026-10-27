#pragma once

#include <cstdint>
#include <cstdio>

// Topology parameters
#define NUM_ROWS 4
#define CORES_PER_ROW 4

#ifndef GOLEM_TOTAL_CORES
#define GOLEM_TOTAL_CORES 16
#endif

#define TOTAL_CORES GOLEM_TOTAL_CORES

// GlobalMemory layout
#define GLOBAL_BASE 0x00000
#ifndef GOLEM_GLOBAL_STRIDE_BYTES
#define GOLEM_GLOBAL_STRIDE_BYTES 65536
#endif
#define GLOBAL_STRIDE GOLEM_GLOBAL_STRIDE_BYTES  // per-core GM window size
#define DATA_OFFSET 0x00000
#define MAILBOX_OFFSET 0xFF00

// DMA completion flag layout (last 32 bytes of each core's GM window)
#define GM_FLAG_REGION_SIZE 0x20
#define GM_READ_SEQ_OFFSET  0x20  // size - 0x20
#define GM_READ_VAL_OFFSET  0x18  // size - 0x18
#define GM_WRITE_SEQ_OFFSET 0x10  // size - 0x10
#define GM_WRITE_VAL_OFFSET 0x08  // size - 0x08

// CPU frequency (GHz) for cycle-to-time conversion
#define CPU_FREQ_GHZ 2.0

// Memory barrier (RISC-V fence)
#define MEMORY_BARRIER() __asm__ __volatile__("fence rw,rw" ::: "memory")

// Lightweight debug print (flush to keep ordering during multi-process runs)
#define DEBUG_PRINT(...) do { \
    printf(__VA_ARGS__); \
    fflush(stdout); \
} while (0)

// // rdcycle helper
static inline uint64_t read_cycles() {
    uint64_t v;
    asm volatile("rdcycle %0" : "=r"(v));
    return v;
}

// Cycle-to-time conversion helpers
static inline double cycles_to_ns(uint64_t cycles) { return (double)cycles / CPU_FREQ_GHZ; }
static inline double cycles_to_us(uint64_t cycles) { return cycles_to_ns(cycles) / 1000.0; }
static inline double cycles_to_ms(uint64_t cycles) { return cycles_to_ns(cycles) / 1e6; }

// Address helpers
static inline uint64_t get_core_base_addr(int core_id) {
    return GLOBAL_BASE + (uint64_t)core_id * GLOBAL_STRIDE;
}
static inline uint64_t get_core_data_addr(int core_id) {
    return get_core_base_addr(core_id) + DATA_OFFSET;
}
static inline uint64_t get_core_mailbox_addr(int core_id) {
    return get_core_base_addr(core_id) + MAILBOX_OFFSET;
}

static inline uint64_t get_core_read_seq_addr(int core_id) {
    return get_core_base_addr(core_id) + GLOBAL_STRIDE - GM_READ_SEQ_OFFSET;
}

static inline uint64_t get_core_read_flag_addr(int core_id) {
    return get_core_base_addr(core_id) + GLOBAL_STRIDE - GM_READ_VAL_OFFSET;
}

static inline uint64_t get_core_write_seq_addr(int core_id) {
    return get_core_base_addr(core_id) + GLOBAL_STRIDE - GM_WRITE_SEQ_OFFSET;
}

static inline uint64_t get_core_write_flag_addr(int core_id) {
    return get_core_base_addr(core_id) + GLOBAL_STRIDE - GM_WRITE_VAL_OFFSET;
}

// Sync flag helper
static inline int32_t get_completion_flag(int col_id) {
    return (col_id + 1) * 2;
}
