#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GOLEM_DTYPE_INT32 = 0,
    GOLEM_DTYPE_FP32 = 1,
} golem_dtype_t;

typedef enum {
    GOLEM_LAYOUT_ROW_MAJOR = 0,
} golem_layout_t;

typedef enum {
    GOLEM_STATUS_OK = 0,
    GOLEM_STATUS_INVALID_ARGUMENT = 1,
    GOLEM_STATUS_UNSUPPORTED = 2,
    GOLEM_STATUS_RUNTIME_ERROR = 3,
    GOLEM_STATUS_INTERNAL_ERROR = 4,
} golem_status_t;

typedef struct {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t block_m;
    int64_t block_n;
    int64_t block_k;
    golem_dtype_t dtype;
    golem_layout_t layout;
    int32_t transpose_a;
    int32_t transpose_b;
} golem_matmul_op_desc_t;

typedef struct {
    void* data;
    int64_t ndim;
    int64_t shape[2];
    int64_t stride[2];
    golem_dtype_t dtype;
    golem_layout_t layout;
} golem_tensor_desc_t;

typedef void* golem_kernel_handle_t;

golem_status_t golemCreateMatmulKernel(
    const golem_matmul_op_desc_t* op_desc,
    golem_kernel_handle_t* out_handle);

golem_status_t golemRunMatmul(
    golem_kernel_handle_t handle,
    const golem_tensor_desc_t* a,
    const golem_tensor_desc_t* b,
    const golem_tensor_desc_t* c);

golem_status_t golemDestroyKernel(golem_kernel_handle_t handle);

const char* golemGetLastErrorString(void);

#ifdef __cplusplus
}
#endif
