#pragma once

#include <cstdint>

#include "ex_instr.h"
#include "gm_config.h"
#include "pipeline_config.h"

// Local mailbox owned by the current core. Software only polls local addresses.
constexpr uint64_t CTRL_LOCAL_MAILBOX_BASE = 0x1900;
constexpr uint64_t CTRL_LOCAL_REQ_SEQ_OFF = 0x00;
constexpr uint64_t CTRL_LOCAL_REQ_VALID_OFF = 0x08;
constexpr uint64_t CTRL_LOCAL_GRANT_SEQ_OFF = 0x10;
constexpr uint64_t CTRL_LOCAL_GRANT_WINDOW_OFF = 0x18;
constexpr uint64_t CTRL_LOCAL_DONE_SEQ_OFF = 0x20;
constexpr uint64_t CTRL_LOCAL_DONE_VALID_OFF = 0x28;
constexpr uint64_t CTRL_LOCAL_FINISHED_OFF = 0x30;
constexpr uint64_t CTRL_LOCAL_GROUP_DONE_OFF = 0x38;
constexpr uint64_t CTRL_LOCAL_REQ_SRC_OFF = 0x40;
constexpr uint64_t CTRL_LOCAL_REQ_DST_OFF = 0x48;
constexpr uint64_t CTRL_LOCAL_REQ_BYTES_OFF = 0x50;
constexpr uint64_t CTRL_LOCAL_REQ_NODE_OFF = 0x58;
constexpr uint64_t CTRL_LOCAL_REQ_WINDOW_OFF = 0x60;

inline uint64_t ctrl_local_mailbox_addr(int core_id, uint64_t off) {
    return GLOBAL_BASE + static_cast<uint64_t>(core_id) * GLOBAL_STRIDE + CTRL_LOCAL_MAILBOX_BASE + off;
}

inline void ctrl_local_mailbox_init(int core_id) {
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_SEQ_OFF));
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_VALID_OFF));
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_GRANT_SEQ_OFF));
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_GRANT_WINDOW_OFF));
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_DONE_SEQ_OFF));
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_DONE_VALID_OFF));
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_FINISHED_OFF));
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_GROUP_DONE_OFF));
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_SRC_OFF));
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_DST_OFF));
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_BYTES_OFF));
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_NODE_OFF));
    reg2gm(0, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_WINDOW_OFF));
}

inline uint64_t ctrl_local_grant_seq(int core_id) {
    return gm2reg(ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_GRANT_SEQ_OFF));
}

inline uint64_t ctrl_local_grant_window(int core_id) {
    return gm2reg(ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_GRANT_WINDOW_OFF));
}

inline uint64_t ctrl_local_group_done(int core_id) {
    return gm2reg(ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_GROUP_DONE_OFF));
}

inline void ctrl_delay_cycles(volatile uint32_t cycles) {
    while (cycles--) {
        asm volatile ("nop");
    }
}

inline void ctrl_wait_local_eq(uint64_t addr, uint64_t expected) {
    uint32_t backoff = 8;
    while (gm2reg(addr) != expected) {
        ctrl_delay_cycles(backoff);
        if (backoff < 2048) {
            backoff <<= 1;
        }
    }
}

inline void ctrl_publish_request_local(
    int core_id,
    uint64_t req_seq,
    uint64_t src_addr,
    uint64_t dst_addr,
    uint64_t bytes,
    uint64_t node,
    uint64_t window
) {
    ctrl_wait_local_eq(ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_VALID_OFF), 0);
    reg2gm(src_addr, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_SRC_OFF));
    reg2gm(dst_addr, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_DST_OFF));
    reg2gm(bytes, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_BYTES_OFF));
    reg2gm(node, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_NODE_OFF));
    reg2gm(window, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_WINDOW_OFF));
    reg2gm(req_seq, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_SEQ_OFF));
    reg2gm(1, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_REQ_VALID_OFF));
}

inline void ctrl_publish_done_local(int core_id, uint64_t req_seq) {
    ctrl_wait_local_eq(ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_DONE_VALID_OFF), 0);
    reg2gm(req_seq, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_DONE_SEQ_OFF));
    reg2gm(1, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_DONE_VALID_OFF));
}

inline void ctrl_publish_finished_local(int core_id) {
    reg2gm(1, ctrl_local_mailbox_addr(core_id, CTRL_LOCAL_FINISHED_OFF));
}
