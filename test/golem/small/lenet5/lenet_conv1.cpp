#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core_bind.h"
#include "conv1_ops.h"
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

static golem_dtype_t read_dtype_env_or_default(const char* name, golem_dtype_t default_value) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    if (std::strcmp(raw, "int32") == 0 || std::strcmp(raw, "i32") == 0) {
        return GOLEM_DTYPE_INT32;
    }
    if (std::strcmp(raw, "fp32") == 0 || std::strcmp(raw, "float32") == 0 || std::strcmp(raw, "float") == 0) {
        return GOLEM_DTYPE_FP32;
    }
    return default_value;
}

int main(int argc, char* argv[]) {
    const int core_id = bind_and_resolve_core_from_argv_or_exit(argc, argv, TOTAL_CORES);
    const golem_dtype_t dtype = read_dtype_env_or_default("GOLEM_MATMUL_DTYPE", GOLEM_DTYPE_INT32);

    golem_matmul_op_desc_t op_desc = {
        .m = read_i64_env_or_default("GOLEM_MATMUL_M", GEMM_M),
        .n = read_i64_env_or_default("GOLEM_MATMUL_N", GEMM_N),
        .k = read_i64_env_or_default("GOLEM_MATMUL_K", GEMM_K),
        .block_m = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_M", TILE_DIM),
        .block_n = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_N", TILE_DIM),
        .block_k = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_K", TILE_DIM),
        .dtype = dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
        .transpose_a = 0,
        .transpose_b = 0,
    };

    return conv1_ops::run_conv1(core_id, op_desc, dtype);
}
