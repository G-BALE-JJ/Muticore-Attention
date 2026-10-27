#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// ==========================================
// 1. 全局权重与偏置内存分配
// ==========================================
float b_conv1[6];
float w_conv1[6 * 1 * 5 * 5];

float b_conv2[16];
float w_conv2[16 * 6 * 5 * 5];

float b_fc1[120];
float w_fc1[120 * 256];

float b_fc2[84];
float w_fc2[84 * 120];

float b_fc3[10];
float w_fc3[10 * 84];

// 为 im2col 预分配全局静态内存池 
static float im2col_buffer[15000];

// ==========================================
// 2. 核心数学与计算层定义
// ==========================================
inline float relu(float x) { return x > 0 ? x : 0; }

// ★2D 卷积层 (彻底拆分三段计时) ★
void conv2d(const float* input, float* output, const float* weights, const float* bias, 
            int in_channels, int out_channels, int in_size, int kernel_size,
            double* t_im2col_out, double* t_gemm_out, double* t_relu_out) {
    
    int out_size = in_size - kernel_size + 1;
    int M = out_channels;
    int K_dim = in_channels * kernel_size * kernel_size; 
    int N = out_size * out_size;                         

    // [阶段 1]: CPU 特征重排
    double t0 = omp_get_wtime();
    #pragma omp parallel for
    for (int c = 0; c < K_dim; ++c) {
        int w_offset = c % kernel_size;
        int h_offset = (c / kernel_size) % kernel_size;
        int c_im = c / (kernel_size * kernel_size);
        for (int h = 0; h < out_size; ++h) {
            for (int w = 0; w < out_size; ++w) {
                int im_row = h + h_offset;
                int im_col = w + w_offset;
                im2col_buffer[c * N + (h * out_size + w)] = 
                    input[(c_im * in_size + im_row) * in_size + im_col];
            }
        }
    }
    double t1 = omp_get_wtime();

    // [阶段 2]: 纯 GEMM 计算 (未来由 Golem 阵列负责)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = bias[m];
            for (int k = 0; k < K_dim; ++k) {
                sum += weights[m * K_dim + k] * im2col_buffer[k * N + n];
            }
            output[m * N + n] = sum; 
        }
    }
    double t2 = omp_get_wtime();

    // [阶段 3]: 独立 ReLU 后处理 (CPU 负责)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M * N; ++i) {
        output[i] = relu(output[i]);
    }
    double t3 = omp_get_wtime();

    // 严谨传出三段耗时
    if (t_im2col_out) *t_im2col_out = (t1 - t0) * 1000.0;
    if (t_gemm_out)   *t_gemm_out   = (t2 - t1) * 1000.0;
    if (t_relu_out)   *t_relu_out   = (t3 - t2) * 1000.0;
}

// 最大池化层 (CPU)
void max_pool(const float* input, float* output, int channels, int in_size, int pool_size) {
    int out_size = in_size / pool_size;
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < out_size; ++i) {
            for (int j = 0; j < out_size; ++j) {
                float max_val = -1e5;
                for (int pi = 0; pi < pool_size; ++pi) {
                    for (int pj = 0; pj < pool_size; ++pj) {
                        int in_idx = c * (in_size * in_size) + (i * pool_size + pi) * in_size + (j * pool_size + pj);
                        if (input[in_idx] > max_val) max_val = input[in_idx];
                    }
                }
                int out_idx = c * (out_size * out_size) + i * out_size + j;
                output[out_idx] = max_val;
            }
        }
    }
}

// 全连接层 (同样拆分计时)
void dense(const float* input, float* output, const float* weights, const float* bias, 
           int in_features, int out_features, int apply_relu,
           double* t_gemm_out, double* t_relu_out) {
    
    // [阶段 1]: 纯 GEMM
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_features; ++o) {
        float sum = bias[o];
        for (int i = 0; i < in_features; ++i) {
            sum += input[i] * weights[o * in_features + i];
        }
        output[o] = sum;
    }
    double t1 = omp_get_wtime();

    // [阶段 2]: 独立 ReLU
    if (apply_relu) {
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < out_features; ++o) {
            output[o] = relu(output[o]);
        }
    }
    double t2 = omp_get_wtime();

    if (t_gemm_out) *t_gemm_out = (t1 - t0) * 1000.0;
    if (t_relu_out) *t_relu_out = (t2 - t1) * 1000.0;
}

// ==========================================
// 3. 模型权重加载
// ==========================================
void load_darknet_weights(const char* filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { printf("【错误】找不到权重文件: %s\n", filename); exit(1); }
    int header[5];
    if (fread(header, sizeof(int), 5, fp) != 5) exit(1);

    fread(b_conv1, sizeof(float), 6, fp);     fread(w_conv1, sizeof(float), 6 * 1 * 5 * 5, fp);
    fread(b_conv2, sizeof(float), 16, fp);    fread(w_conv2, sizeof(float), 16 * 6 * 5 * 5, fp);
    fread(b_fc1, sizeof(float), 120, fp);     fread(w_fc1, sizeof(float), 120 * 256, fp);
    fread(b_fc2, sizeof(float), 84, fp);      fread(w_fc2, sizeof(float), 84 * 120, fp);
    fread(b_fc3, sizeof(float), 10, fp);      fread(w_fc3, sizeof(float), 10 * 84, fp);
    fclose(fp);
}

// ==========================================
// 4. LeNet-5 前向推理流水线
// ==========================================
void lenet5_inference(const float* image) {
    static float conv1_out[6 * 24 * 24]; static float pool1_out[6 * 12 * 12];
    static float conv2_out[16 * 8 * 8];  static float pool2_out[16 * 4 * 4];
    static float fc1_out[120]; static float fc2_out[84]; static float fc3_out[10];

    double t_start, t_end;
    double t_layer[7] = {0}; 
    
    // Conv 细分时间
    double t_im2col1 = 0, t_gemm1 = 0, t_relu1 = 0;
    double t_im2col2 = 0, t_gemm2 = 0, t_relu2 = 0;
    // FC 细分时间
    double t_fc1_gemm = 0, t_fc1_relu = 0;
    double t_fc2_gemm = 0, t_fc2_relu = 0;
    double t_fc3_gemm = 0, t_fc3_relu = 0;

    // 1. Conv1
    t_start = omp_get_wtime();
    conv2d(image, conv1_out, w_conv1, b_conv1, 1, 6, 28, 5, &t_im2col1, &t_gemm1, &t_relu1);
    t_end = omp_get_wtime();
    t_layer[0] = (t_end - t_start) * 1000.0;

    // 2. Pool1
    t_start = omp_get_wtime();
    max_pool(conv1_out, pool1_out, 6, 24, 2);
    t_end = omp_get_wtime();
    t_layer[1] = (t_end - t_start) * 1000.0;

    // 3. Conv2
    t_start = omp_get_wtime();
    conv2d(pool1_out, conv2_out, w_conv2, b_conv2, 6, 16, 12, 5, &t_im2col2, &t_gemm2, &t_relu2);
    t_end = omp_get_wtime();
    t_layer[2] = (t_end - t_start) * 1000.0;

    // 4. Pool2
    t_start = omp_get_wtime();
    max_pool(conv2_out, pool2_out, 16, 8, 2);
    t_end = omp_get_wtime();
    t_layer[3] = (t_end - t_start) * 1000.0;

    // 5. FC1
    t_start = omp_get_wtime();
    dense(pool2_out, fc1_out, w_fc1, b_fc1, 256, 120, 1, &t_fc1_gemm, &t_fc1_relu);
    t_end = omp_get_wtime();
    t_layer[4] = (t_end - t_start) * 1000.0;

    // 6. FC2
    t_start = omp_get_wtime();
    dense(fc1_out, fc2_out, w_fc2, b_fc2, 120, 84, 1, &t_fc2_gemm, &t_fc2_relu);
    t_end = omp_get_wtime();
    t_layer[5] = (t_end - t_start) * 1000.0;

    // 7. FC3
    t_start = omp_get_wtime();
    dense(fc2_out, fc3_out, w_fc3, b_fc3, 84, 10, 0, &t_fc3_gemm, &t_fc3_relu);
    t_end = omp_get_wtime();
    t_layer[6] = (t_end - t_start) * 1000.0;

    double pure_sum = 0;
    for (int i = 0; i < 7; i++) pure_sum += t_layer[i];
    
    printf("\n--- 总计算耗时分布 ---\n");
    printf("1. Conv1 总耗时: %8.3f ms  [im2col: %6.3f, GEMM: %6.3f, ReLU: %6.3f]\n", t_layer[0], t_im2col1, t_gemm1, t_relu1);
    printf("2. Pool1 总耗时: %8.3f ms\n", t_layer[1]);
    printf("3. Conv2 总耗时: %8.3f ms  [im2col: %6.3f, GEMM: %6.3f, ReLU: %6.3f]\n", t_layer[2], t_im2col2, t_gemm2, t_relu2);
    printf("4. Pool2 总耗时: %8.3f ms\n", t_layer[3]);
    printf("5. FC1   总耗时: %8.3f ms  [GEMM: %6.3f, ReLU: %6.3f]\n", t_layer[4], t_fc1_gemm, t_fc1_relu);
    printf("6. FC2   总耗时: %8.3f ms  [GEMM: %6.3f, ReLU: %6.3f]\n", t_layer[5], t_fc2_gemm, t_fc2_relu);
    printf("7. FC3   总耗时: %8.3f ms  [GEMM: %6.3f, ReLU: %6.3f]\n", t_layer[6], t_fc3_gemm, t_fc3_relu);
    printf("------------------------------------------------------------------\n");
    printf("总计算耗时:             %8.3f ms\n", pure_sum);

    printf("\n--- LeNet-5最终预测得分分布 ---\n");
    float max_score = -1e5;
    int predict_class = -1;
    for (int i = 0; i < 10; i++) {
        printf("数字 [%d] 得分: %8.3f\n", i, fc3_out[i]);
        if (fc3_out[i] > max_score) {
            max_score = fc3_out[i];
            predict_class = i;
        }
    }
    printf("------------------------------------------------------------------\n");
    printf(">>> 最终预测数字: %d (最高得分: %.3f) <<<\n\n", predict_class, max_score);
}

int main(int argc, char* argv[]) {
    int num_threads = 1;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        if (num_threads <= 0) num_threads = 1;
    }
    omp_set_num_threads(num_threads);
    
    printf("===============================\n");
    printf("启动LeNet-5推理测试\n");
    printf(">>> 线程数量: %d\n", num_threads);
    printf("===============================\n\n");

    load_darknet_weights("/data3/lzq/SST/sst_workspace/testbench/lenet5/lenet5_py/lenet5_fp32.weights");

    static float test_image[28 * 28];
    FILE *img_fp = fopen("/data3/lzq/SST/sst_workspace/testbench/lenet5/data/image7.bin", "rb");
    if (!img_fp) {
        printf("【错误】找不到图片文件\n");
        return 1;
    }
    fread(test_image, sizeof(float), 28 * 28, img_fp);
    fclose(img_fp);

    lenet5_inference(test_image); 

    printf("===============================\n");
    printf("测试完成退出。\n");
    printf("===============================\n");


    return 0;
}