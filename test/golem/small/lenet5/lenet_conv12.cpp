#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core_bind.h"
#include "conv1_ops.h"
#include "conv2_ops.h"
#include "fc1_ops.h"
#include "fc23_ops.h"
#include "gm_config.h"
#include "pipeline_config.h"
#include "golem_matmul_runtime.h"

static int64_t read_i64_env_or_default(const char* name, int64_t default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(raw, &end, 10);
    if (end == raw || (end != nullptr && *end != '\0')) {
        return default_value;
    }
    return static_cast<int64_t>(parsed);
}

int main(int argc, char* argv[]) {
    const int core_id = bind_and_resolve_core_from_argv_or_exit(argc, argv, TOTAL_CORES);

    auto log_milestone = [core_id](const char* stage, const char* status) {
        if (core_id != 0) {
            return;
        }
        const uint64_t cyc = read_cycles();
        std::fprintf(stdout, "[MILESTONE] stage=%s status=%s cycle=%llu\n",
                     stage, status, static_cast<unsigned long long>(cyc));
        std::fflush(stdout);
    };

    const golem_dtype_t dtype = GOLEM_DTYPE_FP32;
    golem_matmul_op_desc_t conv1_desc = {
        .m = 768,
        .n = 6,
        .k = 64,
        .block_m = 64,
        .block_n = 6,
        .block_k = 64,
        .dtype = dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
        .transpose_a = 0,
        .transpose_b = 0,
    };

    if (conv1_ops::run_conv1(core_id, conv1_desc, dtype) != 0) {
        return 1;
    }

    golem_matmul_op_desc_t conv2_desc = {
        .m = read_i64_env_or_default("GOLEM_CONV2_M", 256),
        .n = read_i64_env_or_default("GOLEM_CONV2_N", 16),
        .k = read_i64_env_or_default("GOLEM_CONV2_K", 192),
        .block_m = read_i64_env_or_default("GOLEM_CONV2_BLOCK_M", 64),
        .block_n = read_i64_env_or_default("GOLEM_CONV2_BLOCK_N", 16),
        .block_k = read_i64_env_or_default("GOLEM_CONV2_BLOCK_K", 64),
        .dtype = dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
        .transpose_a = 0,
        .transpose_b = 0,
    };

    if (conv2_ops::run_conv2(core_id, conv2_desc, dtype) != 0) {
        return 1;
    }

    log_milestone("fc1", "start");
    if (fc1_ops::run_fc1_distributed(core_id) != 0) {
        log_milestone("fc1", "fail");
        return 1;
    }
    log_milestone("fc1", "done");

    log_milestone("fc2", "start");
    const int fc2_ret = fc23_ops::run_fc2_core0(core_id);
    if (fc2_ret != 0) {
        log_milestone("fc2", "fail");
        return fc2_ret;
    }
    log_milestone("fc2", "done");

    log_milestone("fc3", "start");
    const int fc3_ret = fc23_ops::run_fc3_core0(core_id);
    if (fc3_ret != 0) {
        log_milestone("fc3", "fail");
        return fc3_ret;
    }
    log_milestone("fc3", "done");

    return 0;
}
