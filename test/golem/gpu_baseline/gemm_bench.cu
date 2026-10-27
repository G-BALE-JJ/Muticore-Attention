#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(err__));                \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                       \
        cublasStatus_t st__ = (call);                                          \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,     \
                         __LINE__, static_cast<int>(st__));                   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

struct Args {
    int dim = 1024;
    int warmup = 20;
    int iters = 100;
    int batch = 10;
    int device = 0;
    double peak_tflops = 15.7;
    bool csv = false;
};

static int parse_int(const char* s, const char* name) {
    char* end = nullptr;
    long value = std::strtol(s, &end, 10);
    if (!end || *end != '\0' || value <= 0) {
        std::fprintf(stderr, "invalid %s: %s\n", name, s);
        std::exit(2);
    }
    return static_cast<int>(value);
}

static int parse_nonnegative_int(const char* s, const char* name) {
    char* end = nullptr;
    long value = std::strtol(s, &end, 10);
    if (!end || *end != '\0' || value < 0) {
        std::fprintf(stderr, "invalid %s: %s\n", name, s);
        std::exit(2);
    }
    return static_cast<int>(value);
}

static double parse_double(const char* s, const char* name) {
    char* end = nullptr;
    double value = std::strtod(s, &end);
    if (!end || *end != '\0' || value <= 0.0) {
        std::fprintf(stderr, "invalid %s: %s\n", name, s);
        std::exit(2);
    }
    return value;
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--dim") args.dim = parse_int(need_value("--dim"), "dim");
        else if (a == "--warmup") args.warmup = parse_int(need_value("--warmup"), "warmup");
        else if (a == "--iters") args.iters = parse_int(need_value("--iters"), "iters");
        else if (a == "--batch") args.batch = parse_int(need_value("--batch"), "batch");
        else if (a == "--device") args.device = parse_nonnegative_int(need_value("--device"), "device");
        else if (a == "--peak-tflops") args.peak_tflops = parse_double(need_value("--peak-tflops"), "peak-tflops");
        else if (a == "--csv") args.csv = true;
        else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            std::exit(2);
        }
    }
    return args;
}

__global__ void fill_kernel(float* data, size_t n, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = scale * static_cast<float>((idx % 251) + 1) / 251.0f;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    CHECK_CUDA(cudaSetDevice(args.device));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, args.device));

    const int n = args.dim;
    const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = elems * sizeof(float);

    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    CHECK_CUDA(cudaMalloc(&a, bytes));
    CHECK_CUDA(cudaMalloc(&b, bytes));
    CHECK_CUDA(cudaMalloc(&c, bytes));

    const int block = 256;
    const int grid = static_cast<int>((elems + block - 1) / block);
    fill_kernel<<<grid, block>>>(a, elems, 1.0f);
    fill_kernel<<<grid, block>>>(b, elems, 0.5f);
    CHECK_CUDA(cudaMemset(c, 0, bytes));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    for (int i = 0; i < args.warmup; ++i) {
        for (int j = 0; j < args.batch; ++j) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                     &alpha, a, n, b, n, &beta, c, n));
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(args.iters);
    for (int i = 0; i < args.iters; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        for (int j = 0; j < args.batch; ++j) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                     &alpha, a, n, b, n, &beta, c, n));
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        times_ms.push_back(elapsed / static_cast<float>(args.batch));
    }

    const double avg_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    const double min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    const double max_ms = *std::max_element(times_ms.begin(), times_ms.end());
    const double flops = 2.0 * static_cast<double>(n) * n * n;
    const double avg_tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;
    const double peak_tflops = flops / (min_ms * 1.0e-3) / 1.0e12;
    const double avg_util = 100.0 * avg_tflops / args.peak_tflops;
    const double peak_util = 100.0 * peak_tflops / args.peak_tflops;

    if (args.csv) {
        std::printf("dim,device,gpu,latency_avg_ms,latency_min_ms,latency_max_ms,avg_tflops,peak_tflops,avg_util_pct,peak_util_pct,iters,warmup,batch,ref_peak_tflops\n");
        std::printf("%d,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%d,%d,%d,%.3f\n",
                    n, args.device, prop.name, avg_ms, min_ms, max_ms,
                    avg_tflops, peak_tflops, avg_util, peak_util,
                    args.iters, args.warmup, args.batch, args.peak_tflops);
    } else {
        std::printf("GPU: %s\n", prop.name);
        std::printf("dim=%d latency_avg_ms=%.6f latency_min_ms=%.6f latency_max_ms=%.6f\n",
                    n, avg_ms, min_ms, max_ms);
        std::printf("avg_tflops=%.6f peak_tflops=%.6f avg_util_pct=%.3f peak_util_pct=%.3f batch=%d ref_peak_tflops=%.3f\n",
                    avg_tflops, peak_tflops, avg_util, peak_util, args.batch, args.peak_tflops);
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(a));
    CHECK_CUDA(cudaFree(b));
    CHECK_CUDA(cudaFree(c));
    return 0;
}
