#ifndef GOLEM_OPERATOR_API_H
#define GOLEM_OPERATOR_API_H

#include <cstdint>

enum class GolemOpStatus : uint8_t {
    OK = 0,
    INVALID_ARGUMENT = 1,
    UNSUPPORTED = 2,
};

enum class GolemOperatorKind : uint8_t {
    CONV2D_IM2COL = 0,
    MAXPOOL2D = 1,
    DENSE = 2,
    SOFTMAX = 3,
};

enum class TensorDataType : uint8_t {
    FP32 = 0,
    INT32 = 1,
};

enum class TensorLayout : uint8_t {
    ROW_MAJOR = 0,
};

enum class TensorLocation : uint8_t {
    HOST = 0,
    HBM = 1,
    LOCAL_GM = 2,
};

struct TensorDesc {
    void* data;
    int64_t ndim;
    int64_t shape[4];
    // Stride is measured in elements (not bytes).
    int64_t stride[4];
    TensorDataType dtype;
    TensorLayout layout;
    TensorLocation location;
};

struct Conv2dIm2colOpDesc {
    uint32_t version;
    int in_channels;
    int out_channels;
    int in_size;
    int kernel_size;
    bool allow_golem;
    TensorDataType dtype;
    TensorLayout layout;
};

struct MaxPool2dOpDesc {
    uint32_t version;
    int channels;
    int in_size;
    int pool_size;
    TensorDataType dtype;
    TensorLayout layout;
};

struct DenseOpDesc {
    uint32_t version;
    int in_features;
    int out_features;
    bool apply_relu;
    bool allow_golem;
    TensorDataType dtype;
    TensorLayout layout;
};

struct SoftmaxOpDesc {
    uint32_t version;
    int64_t outer;
    int64_t dim;
    int64_t axis;
    bool allow_golem;
    TensorDataType dtype;
    TensorLayout layout;
};

const char* golem_op_last_error();
const char* golem_op_status_string(GolemOpStatus status);

GolemOpStatus validate_tensor_desc(const TensorDesc& t,
                                   int64_t expected_ndim,
                                   TensorDataType expected_dtype,
                                   TensorLayout expected_layout);

GolemOpStatus validate_op_desc(const Conv2dIm2colOpDesc& op);
GolemOpStatus validate_op_desc(const MaxPool2dOpDesc& op);
GolemOpStatus validate_op_desc(const DenseOpDesc& op);
GolemOpStatus validate_op_desc(const SoftmaxOpDesc& op);

GolemOpStatus validate_compatibility(const TensorDesc& input,
                                     const TensorDesc& output,
                                     const TensorDesc& weights,
                                     const TensorDesc& bias,
                                     const Conv2dIm2colOpDesc& op);

GolemOpStatus validate_compatibility(const TensorDesc& input,
                                     const TensorDesc& output,
                                     const MaxPool2dOpDesc& op);

GolemOpStatus validate_compatibility(const TensorDesc& input,
                                     const TensorDesc& output,
                                     const TensorDesc& weights,
                                     const TensorDesc& bias,
                                     const DenseOpDesc& op);

GolemOpStatus validate_compatibility(const TensorDesc& input,
                                     const TensorDesc& output,
                                     const SoftmaxOpDesc& op);

#endif
