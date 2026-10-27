#pragma once

#include <cstdint>

#include "ex_instr.h"
#include "gm_config.h"

constexpr uint64_t SCHED_LOCAL_MAILBOX_BASE = 0x1A00;
constexpr uint64_t SCHED_LOCAL_SUBMIT_HEAD_OFF = 0x00;
constexpr uint64_t SCHED_LOCAL_SUBMIT_TAIL_OFF = 0x08;
constexpr uint64_t SCHED_LOCAL_DONE_HEAD_OFF = 0x10;
constexpr uint64_t SCHED_LOCAL_DONE_TAIL_OFF = 0x18;

constexpr uint64_t SCHED_LOCAL_SUBMIT_RING_BASE = 0x40;
constexpr uint64_t SCHED_LOCAL_SUBMIT_ENTRY_SIZE = 0x30;
constexpr uint64_t SCHED_LOCAL_SUBMIT_ID_OFF = 0x00;
constexpr uint64_t SCHED_LOCAL_SUBMIT_SRC_OFF = 0x08;
constexpr uint64_t SCHED_LOCAL_SUBMIT_DST_OFF = 0x10;
constexpr uint64_t SCHED_LOCAL_SUBMIT_BYTES_OFF = 0x18;
constexpr uint64_t SCHED_LOCAL_SUBMIT_AUX0_OFF = 0x20;
constexpr uint64_t SCHED_LOCAL_SUBMIT_AUX1_OFF = 0x28;

constexpr uint64_t SCHED_LOCAL_DONE_RING_BASE = 0x280;
constexpr uint64_t SCHED_LOCAL_DONE_ENTRY_SIZE = 0x08;
constexpr uint64_t SCHED_LOCAL_DONE_ID_OFF = 0x00;

constexpr uint64_t SCHED_LOCAL_SUBMIT_RING_DEPTH = 8;
constexpr uint64_t SCHED_LOCAL_DONE_RING_DEPTH = 8;
constexpr uint8_t SCHED_PAIR_SLOT_MARKER = 0xFE;

struct SchedRingShadowState {
    uint64_t submit_head_shadow;
    uint64_t submit_tail_shadow;
    bool submit_shadow_initialized;
};

inline SchedRingShadowState& sched_ring_shadow_state(int core_id) {
    static SchedRingShadowState states[GOLEM_TOTAL_CORES];
    static SchedRingShadowState fallback = {0, 0, false};
    if (core_id < 0 || core_id >= GOLEM_TOTAL_CORES) {
        return fallback;
    }
    return states[core_id];
}

inline uint64_t sched_local_mailbox_addr(int core_id, uint64_t off) {
    return GLOBAL_BASE + static_cast<uint64_t>(core_id) * GLOBAL_STRIDE + SCHED_LOCAL_MAILBOX_BASE + off;
}

inline uint64_t sched_local_submit_entry_addr(int core_id, uint64_t ring_idx, uint64_t field_off) {
    const uint64_t slot = ring_idx % SCHED_LOCAL_SUBMIT_RING_DEPTH;
    return sched_local_mailbox_addr(core_id, SCHED_LOCAL_SUBMIT_RING_BASE + slot * SCHED_LOCAL_SUBMIT_ENTRY_SIZE + field_off);
}

inline uint64_t sched_local_done_entry_addr(int core_id, uint64_t ring_idx, uint64_t field_off = SCHED_LOCAL_DONE_ID_OFF) {
    const uint64_t slot = ring_idx % SCHED_LOCAL_DONE_RING_DEPTH;
    return sched_local_mailbox_addr(core_id, SCHED_LOCAL_DONE_RING_BASE + slot * SCHED_LOCAL_DONE_ENTRY_SIZE + field_off);
}

inline void sched_delay_cycles(volatile uint32_t cycles) {
    while (cycles--) {
        asm volatile ("nop");
    }
}

inline void sched_local_mailbox_init(int core_id) {
    reg2gm(0, sched_local_mailbox_addr(core_id, SCHED_LOCAL_SUBMIT_HEAD_OFF));
    reg2gm(0, sched_local_mailbox_addr(core_id, SCHED_LOCAL_SUBMIT_TAIL_OFF));
    reg2gm(0, sched_local_mailbox_addr(core_id, SCHED_LOCAL_DONE_HEAD_OFF));
    reg2gm(0, sched_local_mailbox_addr(core_id, SCHED_LOCAL_DONE_TAIL_OFF));

    for (uint64_t i = 0; i < SCHED_LOCAL_SUBMIT_RING_DEPTH; ++i) {
        reg2gm(0, sched_local_submit_entry_addr(core_id, i, SCHED_LOCAL_SUBMIT_ID_OFF));
        reg2gm(0, sched_local_submit_entry_addr(core_id, i, SCHED_LOCAL_SUBMIT_SRC_OFF));
        reg2gm(0, sched_local_submit_entry_addr(core_id, i, SCHED_LOCAL_SUBMIT_DST_OFF));
        reg2gm(0, sched_local_submit_entry_addr(core_id, i, SCHED_LOCAL_SUBMIT_BYTES_OFF));
        reg2gm(0, sched_local_submit_entry_addr(core_id, i, SCHED_LOCAL_SUBMIT_AUX0_OFF));
        reg2gm(0, sched_local_submit_entry_addr(core_id, i, SCHED_LOCAL_SUBMIT_AUX1_OFF));
    }
    for (uint64_t i = 0; i < SCHED_LOCAL_DONE_RING_DEPTH; ++i) {
        reg2gm(0, sched_local_done_entry_addr(core_id, i));
    }

    SchedRingShadowState& shadow = sched_ring_shadow_state(core_id);
    shadow.submit_head_shadow = 0;
    shadow.submit_tail_shadow = 0;
    shadow.submit_shadow_initialized = true;
}

inline void sched_wait_local_eq(uint64_t addr, uint64_t expected) {
    uint32_t backoff = 8;
    while (gm2reg(addr) != expected) {
        sched_delay_cycles(backoff);
        if (backoff < 2048) {
            backoff <<= 1;
        }
    }
}

inline void sched_wait_submit_slots(int core_id, uint64_t need_slots) {
    const uint64_t head_addr = sched_local_mailbox_addr(core_id, SCHED_LOCAL_SUBMIT_HEAD_OFF);
    SchedRingShadowState& shadow = sched_ring_shadow_state(core_id);
    if (!shadow.submit_shadow_initialized) {
        shadow.submit_head_shadow = gm2reg(head_addr);
        shadow.submit_tail_shadow = gm2reg(sched_local_mailbox_addr(core_id, SCHED_LOCAL_SUBMIT_TAIL_OFF));
        shadow.submit_shadow_initialized = true;
    }
    while (true) {
        if (shadow.submit_tail_shadow - shadow.submit_head_shadow + need_slots <= SCHED_LOCAL_SUBMIT_RING_DEPTH) {
            return;
        }
        shadow.submit_head_shadow = gm2reg(head_addr);
        if (shadow.submit_tail_shadow - shadow.submit_head_shadow + need_slots <= SCHED_LOCAL_SUBMIT_RING_DEPTH) {
            return;
        }
        sched_delay_cycles(8);
    }
}

inline void sched_wait_done_slots(int core_id, uint64_t need_slots) {
    const uint64_t head_addr = sched_local_mailbox_addr(core_id, SCHED_LOCAL_DONE_HEAD_OFF);
    const uint64_t tail_addr = sched_local_mailbox_addr(core_id, SCHED_LOCAL_DONE_TAIL_OFF);
    while (true) {
        const uint64_t head = gm2reg(head_addr);
        const uint64_t tail = gm2reg(tail_addr);
        if (tail - head + need_slots <= SCHED_LOCAL_DONE_RING_DEPTH) {
            return;
        }
        sched_delay_cycles(8);
    }
}

struct SchedSubmitEntry {
    uint64_t request_id;
    uint64_t src_addr;
    uint64_t dst_addr;
};

struct SchedSubmitProfile {
    uint64_t queue_wait_cycles;
    uint64_t write_cycles;
    uint64_t queue_wait_polls;
};

inline uint64_t sched_read_cycle_counter() {
    uint64_t cycles;
    asm volatile (
        "rdcycle %0"
        : "=r"(cycles)
        :
        :
    );
    return cycles;
}

inline SchedSubmitProfile sched_wait_submit_slots_profiled(int core_id, uint64_t need_slots) {
    SchedSubmitProfile prof = {0, 0, 0};
    const uint64_t begin = sched_read_cycle_counter();
    const uint64_t head_addr = sched_local_mailbox_addr(core_id, SCHED_LOCAL_SUBMIT_HEAD_OFF);
    SchedRingShadowState& shadow = sched_ring_shadow_state(core_id);
    if (!shadow.submit_shadow_initialized) {
        shadow.submit_head_shadow = gm2reg(head_addr);
        shadow.submit_tail_shadow = gm2reg(sched_local_mailbox_addr(core_id, SCHED_LOCAL_SUBMIT_TAIL_OFF));
        shadow.submit_shadow_initialized = true;
    }
    while (true) {
        if (shadow.submit_tail_shadow - shadow.submit_head_shadow + need_slots <= SCHED_LOCAL_SUBMIT_RING_DEPTH) {
            break;
        }
        shadow.submit_head_shadow = gm2reg(head_addr);
        if (shadow.submit_tail_shadow - shadow.submit_head_shadow + need_slots <= SCHED_LOCAL_SUBMIT_RING_DEPTH) {
            break;
        }
        prof.queue_wait_polls++;
        sched_delay_cycles(8);
    }
    prof.queue_wait_cycles = sched_read_cycle_counter() - begin;
    return prof;
}

inline void sched_write_submit_entry_compact(int core_id, uint64_t ring_idx, const SchedSubmitEntry& entry) {
    reg2gm(entry.request_id, sched_local_submit_entry_addr(core_id, ring_idx, SCHED_LOCAL_SUBMIT_ID_OFF));
    reg2gm(entry.src_addr, sched_local_submit_entry_addr(core_id, ring_idx, SCHED_LOCAL_SUBMIT_SRC_OFF));
    reg2gm(entry.dst_addr, sched_local_submit_entry_addr(core_id, ring_idx, SCHED_LOCAL_SUBMIT_DST_OFF));
}

inline void sched_write_submit_pair_entry_compact(
    int core_id,
    uint64_t ring_idx,
    uint64_t pair_request_id,
    uint64_t mat_src_addr,
    uint64_t mat_dst_addr,
    uint64_t vec_src_addr,
    uint64_t vec_dst_addr
) {
    reg2gm(pair_request_id, sched_local_submit_entry_addr(core_id, ring_idx, SCHED_LOCAL_SUBMIT_ID_OFF));
    reg2gm(mat_src_addr, sched_local_submit_entry_addr(core_id, ring_idx, SCHED_LOCAL_SUBMIT_SRC_OFF));
    reg2gm(mat_dst_addr, sched_local_submit_entry_addr(core_id, ring_idx, SCHED_LOCAL_SUBMIT_DST_OFF));
    reg2gm(vec_src_addr, sched_local_submit_entry_addr(core_id, ring_idx, SCHED_LOCAL_SUBMIT_AUX0_OFF));
    reg2gm(vec_dst_addr, sched_local_submit_entry_addr(core_id, ring_idx, SCHED_LOCAL_SUBMIT_AUX1_OFF));
}

inline uint64_t sched_compose_request_id(int core_id, int slot, uint64_t seq, uint64_t node) {
    return (static_cast<uint64_t>(core_id & 0xff) << 56) |
           (static_cast<uint64_t>(slot & 0xff) << 48) |
           (static_cast<uint64_t>(node & 0xffff) << 32) |
           (seq & 0xffffffffULL);
}

inline uint64_t sched_compose_pair_request_id(int core_id, uint64_t seq, uint64_t node) {
    return sched_compose_request_id(core_id, SCHED_PAIR_SLOT_MARKER, seq, node);
}

inline int sched_request_core_id(uint64_t request_id) {
    return static_cast<int>((request_id >> 56) & 0xffULL);
}

inline int sched_request_slot(uint64_t request_id) {
    return static_cast<int>((request_id >> 48) & 0xffULL);
}

inline uint64_t sched_request_node(uint64_t request_id) {
    return (request_id >> 32) & 0xffffULL;
}

inline uint64_t sched_request_seq(uint64_t request_id) {
    return request_id & 0xffffffffULL;
}

inline bool sched_request_is_pair(uint64_t request_id) {
    return static_cast<uint8_t>(sched_request_slot(request_id) & 0xff) == SCHED_PAIR_SLOT_MARKER;
}

inline void sched_publish_submit_local(
    int core_id,
    uint64_t request_id,
    uint64_t src_addr,
    uint64_t dst_addr,
    uint64_t bytes
) {
    (void)bytes;
    sched_wait_submit_slots(core_id, 1);
    SchedRingShadowState& shadow = sched_ring_shadow_state(core_id);
    const SchedSubmitEntry entry = {
        .request_id = request_id,
        .src_addr = src_addr,
        .dst_addr = dst_addr,
    };
    sched_write_submit_entry_compact(core_id, shadow.submit_tail_shadow, entry);
    MEMORY_BARRIER();
    shadow.submit_tail_shadow += 1;
    reg2gm(shadow.submit_tail_shadow, sched_local_mailbox_addr(core_id, SCHED_LOCAL_SUBMIT_TAIL_OFF));
}

inline void sched_publish_submit_pair_local(
    int core_id,
    const SchedSubmitEntry& first,
    const SchedSubmitEntry& second
) {
    sched_wait_submit_slots(core_id, 2);
    SchedRingShadowState& shadow = sched_ring_shadow_state(core_id);
    sched_write_submit_entry_compact(core_id, shadow.submit_tail_shadow, first);
    sched_write_submit_entry_compact(core_id, shadow.submit_tail_shadow + 1, second);
    MEMORY_BARRIER();
    shadow.submit_tail_shadow += 2;
    reg2gm(shadow.submit_tail_shadow, sched_local_mailbox_addr(core_id, SCHED_LOCAL_SUBMIT_TAIL_OFF));
}

inline SchedSubmitProfile sched_publish_submit_pair_local_profiled(
    int core_id,
    const SchedSubmitEntry& first,
    const SchedSubmitEntry& second
) {
    SchedSubmitProfile prof = sched_wait_submit_slots_profiled(core_id, 2);
    const uint64_t write_begin = sched_read_cycle_counter();
    SchedRingShadowState& shadow = sched_ring_shadow_state(core_id);
    sched_write_submit_entry_compact(core_id, shadow.submit_tail_shadow, first);
    sched_write_submit_entry_compact(core_id, shadow.submit_tail_shadow + 1, second);
    MEMORY_BARRIER();
    shadow.submit_tail_shadow += 2;
    reg2gm(shadow.submit_tail_shadow, sched_local_mailbox_addr(core_id, SCHED_LOCAL_SUBMIT_TAIL_OFF));
    prof.write_cycles = sched_read_cycle_counter() - write_begin;
    return prof;
}

inline void sched_publish_submit_pair_compact_local(
    int core_id,
    uint64_t pair_request_id,
    uint64_t mat_src_addr,
    uint64_t mat_dst_addr,
    uint64_t vec_src_addr,
    uint64_t vec_dst_addr
) {
    sched_wait_submit_slots(core_id, 1);
    SchedRingShadowState& shadow = sched_ring_shadow_state(core_id);
    sched_write_submit_pair_entry_compact(
        core_id,
        shadow.submit_tail_shadow,
        pair_request_id,
        mat_src_addr,
        mat_dst_addr,
        vec_src_addr,
        vec_dst_addr);
    MEMORY_BARRIER();
    shadow.submit_tail_shadow += 1;
    reg2gm(shadow.submit_tail_shadow, sched_local_mailbox_addr(core_id, SCHED_LOCAL_SUBMIT_TAIL_OFF));
}

inline SchedSubmitProfile sched_publish_submit_pair_compact_local_profiled(
    int core_id,
    uint64_t pair_request_id,
    uint64_t mat_src_addr,
    uint64_t mat_dst_addr,
    uint64_t vec_src_addr,
    uint64_t vec_dst_addr
) {
    SchedSubmitProfile prof = sched_wait_submit_slots_profiled(core_id, 1);
    const uint64_t write_begin = sched_read_cycle_counter();
    SchedRingShadowState& shadow = sched_ring_shadow_state(core_id);
    sched_write_submit_pair_entry_compact(
        core_id,
        shadow.submit_tail_shadow,
        pair_request_id,
        mat_src_addr,
        mat_dst_addr,
        vec_src_addr,
        vec_dst_addr);
    MEMORY_BARRIER();
    shadow.submit_tail_shadow += 1;
    reg2gm(shadow.submit_tail_shadow, sched_local_mailbox_addr(core_id, SCHED_LOCAL_SUBMIT_TAIL_OFF));
    prof.write_cycles = sched_read_cycle_counter() - write_begin;
    return prof;
}

inline void sched_publish_done_local(int core_id, uint64_t request_id) {
    sched_wait_done_slots(core_id, 1);
    const uint64_t tail_addr = sched_local_mailbox_addr(core_id, SCHED_LOCAL_DONE_TAIL_OFF);
    const uint64_t tail = gm2reg(tail_addr);
    reg2gm(request_id, sched_local_done_entry_addr(core_id, tail));
    MEMORY_BARRIER();
    reg2gm(tail + 1, tail_addr);
}
