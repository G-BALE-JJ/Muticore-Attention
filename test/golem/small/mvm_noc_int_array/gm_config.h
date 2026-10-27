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
#define GROUP_CTRL_OFFSET 0x1000
#define GROUP_CTRL_SLOT_SIZE 0x200
#define GROUP_GRANT_OFFSET 0x80
#define GROUP_REQ_RING_DEPTH 4
#define GROUP_REQ_ENTRY_SIZE 0x40

#define GROUP_REQ_HEAD_OFF    0x00
#define GROUP_REQ_TAIL_OFF    0x08
#define GROUP_GRANTED_SEQ_OFF 0x10
#define GROUP_DONE_SEQ_OFF    0x18
#define GROUP_STATE_OFF       0x20
#define GROUP_WORKER_FINISHED_OFF 0x28
#define GROUP_RING_BASE_OFF   0x40

#define GROUP_META_BASE_OFF   (GROUP_CTRL_SLOT_SIZE * 4)
#define GROUP_DRAIN_OFF       0x00
#define GROUP_DONE_OFF        0x08
#define GROUP_READY_OFF       0x10

#define GROUP_REQ_SEQ_OFF     0x00
#define GROUP_OP_OFF          0x08
#define GROUP_SRC_OFF         0x10
#define GROUP_DST_OFF         0x18
#define GROUP_BYTES_OFF       0x20
#define GROUP_NODE_OFF        0x28
#define GROUP_SLOT_STATE_OFF  0x30
#define GROUP_SLOT_DONE_OFF   0x38

// DMA completion flag layout (tail of each core's GM window)
#define GM_FLAG_REGION_SIZE  0x40
#define GM_READ0_SEQ_OFFSET  0x40  // size - 0x40
#define GM_READ0_VAL_OFFSET  0x38  // size - 0x38
#define GM_READ1_SEQ_OFFSET  0x30  // size - 0x30
#define GM_READ1_VAL_OFFSET  0x28  // size - 0x28
#define GM_WRITE_SEQ_OFFSET  0x20  // size - 0x20
#define GM_WRITE_VAL_OFFSET  0x18  // size - 0x18
#define GM_READ_SLOT_OFFSET  0x10  // size - 0x10

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

static inline uint64_t get_group_ctrl_base_addr(int manager_core_id) {
    return get_core_base_addr(manager_core_id) + GROUP_CTRL_OFFSET;
}

static inline uint64_t get_group_ctrl_slot_addr(int manager_core_id, int local_index) {
    return get_group_ctrl_base_addr(manager_core_id) + (uint64_t)local_index * GROUP_CTRL_SLOT_SIZE;
}

static inline uint64_t get_group_req_entry_addr(int manager_core_id, int local_index, int ring_index) {
    return get_group_ctrl_slot_addr(manager_core_id, local_index) + GROUP_RING_BASE_OFF + (uint64_t)ring_index * GROUP_REQ_ENTRY_SIZE;
}

static inline uint64_t get_core_group_grant_addr(int core_id) {
    return get_core_mailbox_addr(core_id) + GROUP_GRANT_OFFSET;
}

static inline uint64_t get_core_read_seq_addr(int core_id) {
    return get_core_base_addr(core_id) + GLOBAL_STRIDE - GM_READ0_SEQ_OFFSET;
}

static inline uint64_t get_core_read_flag_addr(int core_id) {
    return get_core_base_addr(core_id) + GLOBAL_STRIDE - GM_READ0_VAL_OFFSET;
}

static inline uint64_t get_core_read_seq_addr_slot(int core_id, int slot) {
    return get_core_base_addr(core_id) + GLOBAL_STRIDE - (slot == 0 ? GM_READ0_SEQ_OFFSET : GM_READ1_SEQ_OFFSET);
}

static inline uint64_t get_core_read_flag_addr_slot(int core_id, int slot) {
    return get_core_base_addr(core_id) + GLOBAL_STRIDE - (slot == 0 ? GM_READ0_VAL_OFFSET : GM_READ1_VAL_OFFSET);
}

static inline uint64_t get_core_read_slot_selector_addr(int core_id) {
    return get_core_base_addr(core_id) + GLOBAL_STRIDE - GM_READ_SLOT_OFFSET;
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
