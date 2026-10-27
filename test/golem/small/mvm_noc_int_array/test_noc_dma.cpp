#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core_bind.h"
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
    if (std::strcmp(raw, "fp16") == 0 || std::strcmp(raw, "float16") == 0 || std::strcmp(raw, "half") == 0) {
        return GOLEM_DTYPE_FP16;
    }
    return default_value;
}

static int32_t read_flag_env_or_default(const char* name, int32_t default_value) {
    return static_cast<int32_t>(read_i64_env_or_default(name, default_value));
}

int main(int argc, char* argv[]) {
    bind_and_resolve_core_from_argv_or_exit(argc, argv, TOTAL_CORES);
    const golem_dtype_t dtype = read_dtype_env_or_default("GOLEM_MATMUL_DTYPE", GOLEM_DTYPE_INT32);

    golem_matmul_op_desc_t op_desc = {
        .m = read_i64_env_or_default("GOLEM_MATMUL_M", GEMM_M),
        .n = read_i64_env_or_default("GOLEM_MATMUL_N", GEMM_N),
        .k = read_i64_env_or_default("GOLEM_MATMUL_K", GEMM_K),
        .block_m = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_M", TILE_M),
        .block_n = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_N", TILE_N_MAX),
        .block_k = read_i64_env_or_default("GOLEM_MATMUL_BLOCK_K", TILE_K),
        .dtype = dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
        .transpose_a = 0,
        .transpose_b = read_flag_env_or_default("GOLEM_MATMUL_TRANSPOSE_B", 0),
    };

    golem_tensor_desc_t a_desc = {
        .data = nullptr,
        .ndim = 2,
        .shape = {op_desc.m, op_desc.k},
        .stride = {op_desc.k, 1},
        .dtype = dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };
    golem_tensor_desc_t b_desc = {
        .data = nullptr,
        .ndim = 2,
        .shape = {
            op_desc.transpose_b ? op_desc.n : op_desc.k,
            op_desc.transpose_b ? op_desc.k : op_desc.n,
        },
        .stride = {
            op_desc.transpose_b ? op_desc.k : op_desc.n,
            1,
        },
        .dtype = dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };
    golem_tensor_desc_t c_desc = {
        .data = nullptr,
        .ndim = 2,
        .shape = {op_desc.m, op_desc.n},
        .stride = {op_desc.n, 1},
        .dtype = dtype,
        .layout = GOLEM_LAYOUT_ROW_MAJOR,
    };

    golem_kernel_handle_t kernel = nullptr;
    golem_status_t st = golemCreateMatmulKernel(&op_desc, &kernel);
    if (st != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[ERROR] golemCreateMatmulKernel failed: %s\n", golemGetLastErrorString());
        return 1;
    }

    st = golemRunMatmul(kernel, &a_desc, &b_desc, &c_desc);
    if (st != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[ERROR] golemRunMatmul failed: %s\n", golemGetLastErrorString());
        golemDestroyKernel(kernel);
        return 1;
    }

    st = golemDestroyKernel(kernel);
    if (st != GOLEM_STATUS_OK) {
        std::fprintf(stderr, "[ERROR] golemDestroyKernel failed: %s\n", golemGetLastErrorString());
        return 1;
    }
    return 0;
}
