#ifndef SST_GOLEM_PROJECTION_JOB_ABI_H
#define SST_GOLEM_PROJECTION_JOB_ABI_H

#include <cstdint>

namespace SST { namespace Golem {

constexpr uint32_t GOLEM_PROJECTION_JOB_MAGIC = 0x50524a31u;

struct ProjectionJobDesc {
    uint32_t magic;
    uint32_t size_bytes;
    uint32_t hidden_dim;
    uint32_t head_dim;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t rows_per_node;
    uint32_t manager_slot;
    uint64_t input_addr;
    uint64_t weights_addr;
    uint64_t q_addr;
    uint64_t k_addr;
    uint64_t v_addr;
    uint64_t q_panel_addr;
    uint64_t k_panel_addr;
    uint64_t v_panel_addr;
    uint64_t completion_addr;
    uint64_t scratch_addr;
};

static_assert(sizeof(ProjectionJobDesc) == 112, "projection descriptor ABI mismatch");

}} // namespace SST::Golem
#endif
