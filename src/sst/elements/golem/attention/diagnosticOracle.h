#ifndef SST_GOLEM_DIAGNOSTIC_ORACLE_H
#define SST_GOLEM_DIAGNOSTIC_ORACLE_H

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <sst/elements/golem/attention/projectionJobAbi.h>

namespace SST { namespace Golem {
// Diagnostic interventions only. A single oracle is selected per simulation;
// these are never enabled by the production dataflow defaults.
inline const std::string& diagnosticOracle() {
    static const std::string name = []() {
        const char* raw = std::getenv("GOLEM_DIAGNOSTIC_ORACLE");
        std::string value = raw ? raw : "none";
        for (const char* allowed : {"none", "gm_read", "matrix_program", "input_scatter",
                                   "weight_arrival", "worker_start", "projection_barrier", "attention_barrier"})
            if (value == allowed) return value;
        throw std::runtime_error("unknown GOLEM_DIAGNOSTIC_ORACLE: " + value);
    }();
    return name;
}
inline bool oracleEnabled(const char* name) { return diagnosticOracle() == name; }
inline uint64_t oracleServiceCycles(const char* name, bool eligible, uint64_t normal) {
    return eligible && oracleEnabled(name) ? 0 : normal;
}
inline uint64_t oracleStartCycle() {
    const char* raw = std::getenv("GOLEM_ORACLE_WORKER_START_CYCLE");
    if (!raw || !*raw) throw std::runtime_error("worker_start oracle requires an explicit baseline origin");
    char* end = nullptr;
    const uint64_t value = std::strtoull(raw, &end, 10);
    if (!value || !end || *end) throw std::runtime_error("invalid oracle worker start cycle");
    return value;
}
inline uint64_t oracleEnvInteger(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) throw std::runtime_error(std::string("missing oracle shape: ") + name);
    char* end = nullptr;
    uint64_t value = std::strtoull(raw, &end, 0);
    if (!end || *end) throw std::runtime_error(std::string("invalid oracle shape: ") + name);
    return value;
}
inline ProjectionJobDesc oracleProjectionDescriptor(uint32_t core) {
    ProjectionJobDesc d{};
    const uint32_t sequence = oracleEnvInteger("GOLEM_ATTENTION_QUERY_LENGTH");
    d.magic = GOLEM_PROJECTION_JOB_MAGIC;
    d.size_bytes = sizeof(d);
    d.query_heads = oracleEnvInteger("GOLEM_ATTENTION_NUM_QUERY_HEADS");
    d.kv_heads = oracleEnvInteger("GOLEM_ATTENTION_NUM_KV_HEADS");
    d.head_dim = oracleEnvInteger("GOLEM_ATTENTION_HEAD_DIM");
    d.hidden_dim = d.query_heads * d.head_dim;
    d.rows_per_node = sequence / 4;
    d.manager_slot = (core - 4) % 4;
    d.reserved = (core - 4) / 4 + 1;
    const uint64_t node = (d.manager_slot + 1) * 0x08000000ull;
    const uint64_t gm = core * 0x00200000ull;
    d.flags = GOLEM_PROJECTION_FUSE_NORM;
    d.input_addr = node;
    d.gamma_addr = node + 0x01000000ull;
    d.norm_output_addr = node + 0x01100000ull;
    d.weights_addr = node + ((0x01100000ull + uint64_t(d.rows_per_node) * d.hidden_dim * 2 + 0xfffffull) & ~0xfffffull);
    d.q_addr = node + oracleEnvInteger("GOLEM_ATTENTION_Q_OFFSET");
    d.k_addr = node + oracleEnvInteger("GOLEM_ATTENTION_K_OFFSET");
    d.v_addr = node + oracleEnvInteger("GOLEM_ATTENTION_V_OFFSET");
    d.q_panel_addr = node + 0x04000000ull;
    d.k_panel_addr = node + 0x05000000ull;
    d.v_panel_addr = node + 0x06000000ull;
    d.rope_table_addr = 0x08000000ull + 0x07000000ull;
    d.completion_addr = node + 0x01ff0000ull + (d.reserved - 1) * 8;
    d.scratch_addr = gm + 0x8000;
    return d;
}
} }
#endif
