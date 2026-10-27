#include "golem_matmul_runtime.h"

#include <cstdarg>
#include <cstdio>
#include <new>
#include <string>

#include "gemm_postop_hook.h"
#include "gemm_matmul_op.h"
#include "pipeline_config.h"

namespace {

struct GolemMatmulKernel {
    golem_matmul_op_desc_t op_desc;
};

thread_local std::string g_last_error;
thread_local GolemFp32TilePostOpHook g_fp32_tile_hook = {nullptr, nullptr};

void set_last_error(const char* fmt, ...) {
    char buffer[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    g_last_error = buffer;
}

bool is_row_major_contiguous(const golem_tensor_desc_t* t) {
    return t->stride[1] == 1 && t->stride[0] == t->shape[1];
}

bool has_data_ptr(const golem_tensor_desc_t* t) {
    return t != nullptr && t->data != nullptr;
}

const char* dtype_name(golem_dtype_t dtype) {
    switch (dtype) {
        case GOLEM_DTYPE_INT32:
            return "GOLEM_DTYPE_INT32";
        case GOLEM_DTYPE_FP32:
            return "GOLEM_DTYPE_FP32";
        default:
            return "GOLEM_DTYPE_UNKNOWN";
    }
}

golem_status_t validate_op_desc_v1(const golem_matmul_op_desc_t* op) {
    if (op == nullptr) {
        set_last_error("op_desc is null");
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (op->m <= 0 || op->n <= 0 || op->k <= 0) {
        set_last_error("M/N/K must be positive, got (%lld,%lld,%lld)",
                       static_cast<long long>(op->m),
                       static_cast<long long>(op->n),
                       static_cast<long long>(op->k));
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (op->block_m <= 0 || op->block_n <= 0 || op->block_k <= 0) {
        set_last_error("block_M/N/K must be positive, got (%lld,%lld,%lld)",
                       static_cast<long long>(op->block_m),
                       static_cast<long long>(op->block_n),
                       static_cast<long long>(op->block_k));
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if ((op->m % op->block_m) != 0 || (op->n % op->block_n) != 0 || (op->k % op->block_k) != 0) {
        set_last_error("M/N/K must be divisible by block_M/N/K, got M/N/K=(%lld,%lld,%lld), block=(%lld,%lld,%lld)",
                       static_cast<long long>(op->m),
                       static_cast<long long>(op->n),
                       static_cast<long long>(op->k),
                       static_cast<long long>(op->block_m),
                       static_cast<long long>(op->block_n),
                       static_cast<long long>(op->block_k));
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (op->dtype != GOLEM_DTYPE_INT32 && op->dtype != GOLEM_DTYPE_FP32) {
        set_last_error("unsupported matmul dtype: %d", static_cast<int>(op->dtype));
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (op->layout != GOLEM_LAYOUT_ROW_MAJOR) {
        set_last_error("v1 only supports GOLEM_LAYOUT_ROW_MAJOR");
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (op->transpose_a != 0 || op->transpose_b != 0) {
        set_last_error("v1 only supports transpose_a=0 and transpose_b=0");
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (op->block_m != TILE_DIM || op->block_k != TILE_DIM) {
        set_last_error("v1 requires block_M/block_K == GOLEM_DIM(%d), got (%lld,%lld)",
                       TILE_DIM,
                       static_cast<long long>(op->block_m),
                       static_cast<long long>(op->block_k));
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (op->block_n > TILE_DIM) {
        set_last_error("v1 requires block_N <= GOLEM_DIM(%d), got %lld",
                       TILE_DIM,
                       static_cast<long long>(op->block_n));
        return GOLEM_STATUS_UNSUPPORTED;
    }
    return GOLEM_STATUS_OK;
}

golem_status_t validate_tensor_desc_v1(const golem_tensor_desc_t* t, int64_t expected_rows, int64_t expected_cols, const char* name) {
    if (t == nullptr) {
        set_last_error("%s tensor descriptor is null", name);
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (t->ndim != 2) {
        set_last_error("%s tensor ndim must be 2, got %lld", name, static_cast<long long>(t->ndim));
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (t->shape[0] != expected_rows || t->shape[1] != expected_cols) {
        set_last_error("%s tensor shape mismatch, expected [%lld,%lld], got [%lld,%lld]",
                       name,
                       static_cast<long long>(expected_rows),
                       static_cast<long long>(expected_cols),
                       static_cast<long long>(t->shape[0]),
                       static_cast<long long>(t->shape[1]));
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    if (t->dtype != GOLEM_DTYPE_INT32 && t->dtype != GOLEM_DTYPE_FP32) {
        set_last_error("%s tensor dtype unsupported: %d", name, static_cast<int>(t->dtype));
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (t->layout != GOLEM_LAYOUT_ROW_MAJOR) {
        set_last_error("%s tensor layout must be GOLEM_LAYOUT_ROW_MAJOR", name);
        return GOLEM_STATUS_UNSUPPORTED;
    }
    if (!is_row_major_contiguous(t)) {
        set_last_error("%s tensor must be row-major contiguous, got stride=(%lld,%lld)",
                       name,
                       static_cast<long long>(t->stride[0]),
                       static_cast<long long>(t->stride[1]));
        return GOLEM_STATUS_UNSUPPORTED;
    }
    return GOLEM_STATUS_OK;
}

golem_status_t validate_tensor_desc_for_op(const golem_tensor_desc_t* t, int64_t expected_rows, int64_t expected_cols, const char* name, golem_dtype_t op_dtype) {
    golem_status_t st = validate_tensor_desc_v1(t, expected_rows, expected_cols, name);
    if (st != GOLEM_STATUS_OK) {
        return st;
    }
    if (t->dtype != op_dtype) {
        set_last_error("%s tensor dtype (%s) does not match op dtype (%s)",
                       name,
                       dtype_name(t->dtype),
                       dtype_name(op_dtype));
        return GOLEM_STATUS_UNSUPPORTED;
    }
    return GOLEM_STATUS_OK;
}

golem_status_t run_matmul_int32(const golem_matmul_op_desc_t& op,
                                const golem_tensor_desc_t* a,
                                const golem_tensor_desc_t* b,
                                const golem_tensor_desc_t* c,
                                bool has_tensor_bindings) {
    if (has_tensor_bindings) {
        MatmulTensorBindings tensors = {
            .a = reinterpret_cast<const int32_t*>(a->data),
            .b = reinterpret_cast<const int32_t*>(b->data),
            .c = reinterpret_cast<int32_t*>(c->data),
            .a_stride0 = a->stride[0],
            .a_stride1 = a->stride[1],
            .b_stride0 = b->stride[0],
            .b_stride1 = b->stride[1],
            .c_stride0 = c->stride[0],
            .c_stride1 = c->stride[1],
        };

        matmul_with_tensors(
            static_cast<int>(op.m),
            static_cast<int>(op.n),
            static_cast<int>(op.k),
            static_cast<int>(op.block_m),
            static_cast<int>(op.block_n),
            static_cast<int>(op.block_k),
            tensors);
        return GOLEM_STATUS_OK;
    }

    matmul(
        static_cast<int>(op.m),
        static_cast<int>(op.n),
        static_cast<int>(op.k),
        static_cast<int>(op.block_m),
        static_cast<int>(op.block_n),
        static_cast<int>(op.block_k));

    return GOLEM_STATUS_OK;
}

golem_status_t run_matmul_fp32(const golem_matmul_op_desc_t& op,
                               const golem_tensor_desc_t* a,
                               const golem_tensor_desc_t* b,
                               const golem_tensor_desc_t* c,
                               bool has_tensor_bindings) {
    if (has_tensor_bindings) {
        MatmulTensorBindingsT<float> tensors = {
            .a = reinterpret_cast<const float*>(a->data),
            .b = reinterpret_cast<const float*>(b->data),
            .c = reinterpret_cast<float*>(c->data),
            .a_stride0 = a->stride[0],
            .a_stride1 = a->stride[1],
            .b_stride0 = b->stride[0],
            .b_stride1 = b->stride[1],
            .c_stride0 = c->stride[0],
            .c_stride1 = c->stride[1],
        };

        matmul_with_tensors_fp32(
            static_cast<int>(op.m),
            static_cast<int>(op.n),
            static_cast<int>(op.k),
            static_cast<int>(op.block_m),
            static_cast<int>(op.block_n),
            static_cast<int>(op.block_k),
            tensors);
        return GOLEM_STATUS_OK;
    }

    matmul_fp32(
        static_cast<int>(op.m),
        static_cast<int>(op.n),
        static_cast<int>(op.k),
        static_cast<int>(op.block_m),
        static_cast<int>(op.block_n),
        static_cast<int>(op.block_k));

    return GOLEM_STATUS_OK;
}

}  // namespace

GolemFp32TilePostOpHook golem_get_fp32_tile_postop_hook() {
    return g_fp32_tile_hook;
}

void golem_set_fp32_tile_postop_hook(GolemFp32TilePostOpHook hook) {
    g_fp32_tile_hook = hook;
}

extern "C" golem_status_t golemCreateMatmulKernel(
    const golem_matmul_op_desc_t* op_desc,
    golem_kernel_handle_t* out_handle) {
    if (out_handle == nullptr) {
        set_last_error("out_handle is null");
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = nullptr;

    golem_status_t st = validate_op_desc_v1(op_desc);
    if (st != GOLEM_STATUS_OK) {
        return st;
    }

    GolemMatmulKernel* kernel = new (std::nothrow) GolemMatmulKernel{*op_desc};
    if (kernel == nullptr) {
        set_last_error("failed to allocate kernel handle");
        return GOLEM_STATUS_INTERNAL_ERROR;
    }

    *out_handle = reinterpret_cast<golem_kernel_handle_t>(kernel);
    return GOLEM_STATUS_OK;
}

extern "C" golem_status_t golemRunMatmul(
    golem_kernel_handle_t handle,
    const golem_tensor_desc_t* a,
    const golem_tensor_desc_t* b,
    const golem_tensor_desc_t* c) {
    if (handle == nullptr) {
        set_last_error("kernel handle is null");
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }

    const GolemMatmulKernel* kernel = reinterpret_cast<const GolemMatmulKernel*>(handle);
    const golem_matmul_op_desc_t& op = kernel->op_desc;

    golem_status_t st = validate_tensor_desc_v1(a, op.m, op.k, "A");
    if (st != GOLEM_STATUS_OK) {
        return st;
    }
    st = validate_tensor_desc_for_op(a, op.m, op.k, "A", op.dtype);
    if (st != GOLEM_STATUS_OK) {
        return st;
    }
    st = validate_tensor_desc_for_op(b, op.k, op.n, "B", op.dtype);
    if (st != GOLEM_STATUS_OK) {
        return st;
    }
    st = validate_tensor_desc_for_op(c, op.m, op.n, "C", op.dtype);
    if (st != GOLEM_STATUS_OK) {
        return st;
    }

    const bool has_tensor_bindings = has_data_ptr(a) && has_data_ptr(b) && has_data_ptr(c);

    switch (op.dtype) {
        case GOLEM_DTYPE_INT32:
            return run_matmul_int32(op, a, b, c, has_tensor_bindings);
        case GOLEM_DTYPE_FP32:
            return run_matmul_fp32(op, a, b, c, has_tensor_bindings);
        default:
            set_last_error("unsupported op dtype: %d", static_cast<int>(op.dtype));
            return GOLEM_STATUS_UNSUPPORTED;
    }
}

extern "C" golem_status_t golemDestroyKernel(golem_kernel_handle_t handle) {
    if (handle == nullptr) {
        set_last_error("kernel handle is null");
        return GOLEM_STATUS_INVALID_ARGUMENT;
    }
    GolemMatmulKernel* kernel = reinterpret_cast<GolemMatmulKernel*>(handle);
    delete kernel;
    return GOLEM_STATUS_OK;
}

extern "C" const char* golemGetLastErrorString(void) {
    return g_last_error.c_str();
}
