#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <inttypes.h>

// 将线程绑定到指定的 CPU 核心（当前使用：pid=0 表示“当前线程”）
void bind_thread_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);        // 清空 CPU 集
    CPU_SET(cpu_id, &cpuset); // 将指定 CPU 加入集合

    // pid = 0 表示“当前线程”
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        perror("sched_setaffinity failed");
        exit(EXIT_FAILURE);
    }
}

// 用常数填充一维数组
void fill_array(int32_t* arr, int32_t value, uint32_t size) {
    for (uint32_t i = 0; i < size; i++) {
        arr[i] = value;
    }
}

// 以行优先的方式打印矩阵
void print_matrix(int32_t* mat, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            printf("%d ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// 打印向量
void print_vector(int32_t* vec, uint32_t cols) {
    for (uint32_t i = 0; i < cols; i++) {
        printf("%d ", vec[i]);
    }
    printf("\n\n");
}

// 返回值：长度为 cols 的指针数组，每个指针指向一个长度为 rows 的列向量
int32_t** mat2col(const int32_t* matB, int rows, int cols) {
    int32_t** vecB = new int32_t*[cols]; //申请一块能装下 cols 个指针的连续内存
    for (int j = 0; j < cols; ++j) vecB[j] = new int32_t[rows]; //为每一列再各自分配一段连续的 rows 个 int32_t 的空间

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            vecB[j][i] = matB[i * cols + j];
        }
    }
    return vecB;
}

// 对应释放函数
void free_columns(int32_t** vecB, int cols) {
    for (int j = 0; j < cols; ++j) delete[] vecB[j];
    delete[] vecB;
}



// 配置阵列：设置矩阵与向量到指定 tile（自定义 mvm 指令）
void setup_array(int32_t *mat, int32_t* vec, uint32_t tile_id) {
    int status_flag = 0;
    asm volatile (
        "mvm.set %0, %1, %2"
      : "=r" (status_flag)
      : "r"(mat), "r"(tile_id)
      : "memory"
    );
    asm volatile (
        "mvm.l %0, %1, %2"
      : "=r" (status_flag)
      : "r"(vec), "r"(tile_id)
      : "memory"
    );
}

// 执行一次矩阵-向量乘（自定义 mvm 指令）
void execute_mvm(uint32_t tile_id) {
    int status_flag = 0;
    asm volatile (
        "mvm %0, %1, x0"
      : "=r" (status_flag)
      : "r"(tile_id)
    );
}

// 将计算结果向量存回内存（自定义 mvm 指令）
void store_vector(int32_t* out, uint32_t tile_id) {
    int status_flag = 0;
    asm volatile (
        "mvm.s %0, %1, %2"
      : "=r" (status_flag)
      : "r"(out), "r"(tile_id)
      : "memory"
    );
}

// 在不同 tile 之间移动向量（自定义 mvm 指令）
void move_vector(uint32_t tile_id, uint32_t tile_id_new) {
    int status_flag = 0;
    asm volatile (
        "mvm.mv %0, %1, %2"
      : "=r" (status_flag)
      : "r"(tile_id), "r"(tile_id_new)
      : "memory"
    );
}

int main() {
    // 定义矩阵维度8*8
    int rows = 8;
    int cols = 8;

    // 分配矩阵 A 及其四个子块
    int32_t* matA   = new int32_t[rows * cols];
    int32_t* matA00 = new int32_t[(rows / 2) * (cols / 2)];
    int32_t* matA01 = new int32_t[(rows / 2) * (cols / 2)];
    int32_t* matA10 = new int32_t[(rows / 2) * (cols / 2)];
    int32_t* matA11 = new int32_t[(rows / 2) * (cols / 2)];

    // 用常数填充矩阵 A
    fill_array(matA, 3, rows * cols);
    // 将 A 按象限拆分为 4 个子块
    for (int i = 0; i < rows / 2; i++) {
        for (int j = 0; j < cols / 2; j++) {
            // 填充 A00
            matA00[i * (cols / 2) + j] = matA[i * cols + j];
            // 填充 A01
            matA01[i * (cols / 2) + j] = matA[i * cols + j + (cols / 2)];
            // 填充 A10
            matA10[i * (cols / 2) + j] = matA[(i + rows / 2) * cols + j];
            // 填充 A11
            matA11[i * (cols / 2) + j] = matA[(i + rows / 2) * cols + j + (cols / 2)];
        }
    }
    // 打印矩阵内容（用于检查）
    printf("A矩阵:\n");
    print_matrix(matA, rows, cols);

    // 分配矩阵 B 及其四个子块
    int32_t* matB   = new int32_t[rows * cols];
    int32_t* matB00 = new int32_t[(rows / 2) * (cols / 2)];
    int32_t* matB01 = new int32_t[(rows / 2) * (cols / 2)];
    int32_t* matB10 = new int32_t[(rows / 2) * (cols / 2)];
    int32_t* matB11 = new int32_t[(rows / 2) * (cols / 2)];

    // 用常数填充矩阵 B
    fill_array(matB, 2, rows * cols);
    // 将 B 按象限拆分为 4 个子块
    for (int i = 0; i < rows / 2; i++) {
        for (int j = 0; j < cols / 2; j++) {
            // 填充 B00
            matB00[i * (cols / 2) + j] = matB[i * cols + j];
            // 填充 B01
            matB01[i * (cols / 2) + j] = matB[i * cols + j + (cols / 2)];
            // 填充 B10
            matB10[i * (cols / 2) + j] = matB[(i + rows / 2) * cols + j];
            // 填充 B11
            matB11[i * (cols / 2) + j] = matB[(i + rows / 2) * cols + j + (cols / 2)];
        }
    }
    printf("B矩阵:\n");
    print_matrix(matB, rows, cols);

    //把B的子矩阵拆分成向量并打印出来
    int32_t** B00_cols = mat2col(matB00, rows/2, cols/2);
    int32_t** B01_cols = mat2col(matB01, rows/2, cols/2);
    int32_t** B10_cols = mat2col(matB10, rows/2, cols/2);
    int32_t** B11_cols = mat2col(matB11, rows/2, cols/2);

    int32_t* matC   = new int32_t[rows * cols];
    
    // 设置 OpenMP 线程数为 16
    omp_set_num_threads(16);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num(); // 当前线程 ID
        int cpu_id = thread_id;               // 简单映射：线程 ID -> CPU ID

        bind_thread_to_cpu(cpu_id);           // 将当前线程绑定到指定 CPU

        //for (int i = 0; i < 15; i++) {
            if (thread_id == 0) {
                printf("Thread 0 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_0_0   = new int32_t[cols/2];
                int32_t* outc_0_1   = new int32_t[cols/2];
                //在tile0上计算outc_0_0
                setup_array(matA00, B00_cols[0], 0);
                execute_mvm(0);
                store_vector(outc_0_0, 0);
                //在tile0上计算outc_0_1
                setup_array(matA01, B10_cols[0], 0);
                execute_mvm(0);
                store_vector(outc_0_1, 0);
                //将outc_0_0和outc_0_1相加的结果存储到输出矩阵matC的对应列
                for (int i = 0; i < cols/2; i++) {
                    matC[i * rows] = outc_0_0[i] + outc_0_1[i];
                }
            }

            if (thread_id == 1) {
                printf("Thread 1 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_1_0   = new int32_t[cols/2];
                int32_t* outc_1_1   = new int32_t[cols/2];
                setup_array(matA00, B00_cols[1], 0);
                execute_mvm(0);
                store_vector(outc_1_0, 0);
                setup_array(matA01, B10_cols[1], 0);
                execute_mvm(0);
                store_vector(outc_1_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[i * rows + 1] = outc_1_0[i] + outc_1_1[i];
                }
            }

            if (thread_id == 2) {
                printf("Thread 2 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_2_0   = new int32_t[cols/2];
                int32_t* outc_2_1   = new int32_t[cols/2];
                setup_array(matA00, B00_cols[2], 0);
                execute_mvm(0);
                store_vector(outc_2_0, 0);
                setup_array(matA01, B10_cols[2], 0);
                execute_mvm(0);
                store_vector(outc_2_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[i * rows + 2] = outc_2_0[i] + outc_2_1[i];
                }
            }

            if (thread_id == 3) {
                printf("Thread 3 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_3_0   = new int32_t[cols/2];
                int32_t* outc_3_1   = new int32_t[cols/2];
                setup_array(matA00, B00_cols[3], 0);
                execute_mvm(0);
                store_vector(outc_3_0, 0);
                setup_array(matA01, B10_cols[3], 0);
                execute_mvm(0);
                store_vector(outc_3_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[i * rows + 3] = outc_3_0[i] + outc_3_1[i];
                }
            }

            if (thread_id == 4) {
                printf("Thread 4 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_4_0   = new int32_t[cols/2];
                int32_t* outc_4_1   = new int32_t[cols/2];
                setup_array(matA00, B01_cols[0], 0);
                execute_mvm(0);
                store_vector(outc_4_0, 0);
                setup_array(matA01, B11_cols[0], 0);
                execute_mvm(0);
                store_vector(outc_4_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[i * rows + 4] = outc_4_0[i] + outc_4_1[i];
                }
            }

            if (thread_id == 5) {
                printf("Thread 5 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_5_0   = new int32_t[cols/2];
                int32_t* outc_5_1   = new int32_t[cols/2];
                setup_array(matA00, B01_cols[1], 0);
                execute_mvm(0);
                store_vector(outc_5_0, 0);
                setup_array(matA01, B11_cols[1], 0);
                execute_mvm(0);
                store_vector(outc_5_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[i * rows + 5] = outc_5_0[i] + outc_5_1[i];
                }
            }

            if (thread_id == 6) {
                printf("Thread 6 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_6_0   = new int32_t[cols/2];
                int32_t* outc_6_1   = new int32_t[cols/2];
                setup_array(matA00, B01_cols[2], 0);
                execute_mvm(0);
                store_vector(outc_6_0, 0);
                setup_array(matA01, B11_cols[2], 0);
                execute_mvm(0);
                store_vector(outc_6_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[i * rows + 6] = outc_6_0[i] + outc_6_1[i];
                }
            }

            if (thread_id == 7) {
                printf("Thread 7 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_7_0   = new int32_t[cols/2];
                int32_t* outc_7_1   = new int32_t[cols/2];
                setup_array(matA00, B01_cols[3], 0);
                execute_mvm(0);
                store_vector(outc_7_0, 0);
                setup_array(matA01, B11_cols[3], 0);
                execute_mvm(0);
                store_vector(outc_7_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[i * rows + 7] = outc_7_0[i] + outc_7_1[i];
                }
            }

            if (thread_id == 8) {
                printf("Thread 8 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_8_0   = new int32_t[cols/2];
                int32_t* outc_8_1   = new int32_t[cols/2];
                setup_array(matA10, B00_cols[0], 0);
                execute_mvm(0);
                store_vector(outc_8_0, 0);
                setup_array(matA11, B10_cols[0], 0);
                execute_mvm(0);
                store_vector(outc_8_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[(i+cols/2) * rows + 0] = outc_8_0[i] + outc_8_1[i];
                }
            }

            if (thread_id == 9) {
                printf("Thread 9 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_9_0   = new int32_t[cols/2];
                int32_t* outc_9_1   = new int32_t[cols/2];
                setup_array(matA10, B00_cols[1], 0);
                execute_mvm(0);
                store_vector(outc_9_0, 0);
                setup_array(matA11, B10_cols[1], 0);
                execute_mvm(0);
                store_vector(outc_9_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[(i+cols/2) * rows + 1] = outc_9_0[i] + outc_9_1[i];
                }
            }

            if (thread_id == 10) {
                printf("Thread 10 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_10_0   = new int32_t[cols/2];
                int32_t* outc_10_1   = new int32_t[cols/2];
                setup_array(matA10, B00_cols[2], 0);
                execute_mvm(0);
                store_vector(outc_10_0, 0);
                setup_array(matA11, B10_cols[2], 0);
                execute_mvm(0);
                store_vector(outc_10_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[(i+cols/2) * rows + 2] = outc_10_0[i] + outc_10_1[i];
                }
            }

            if (thread_id == 11) {
                printf("Thread 11 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_11_0   = new int32_t[cols/2];
                int32_t* outc_11_1   = new int32_t[cols/2];
                setup_array(matA10, B00_cols[3], 0);
                execute_mvm(0);
                store_vector(outc_11_0, 0);
                setup_array(matA11, B10_cols[3], 0);
                execute_mvm(0);
                store_vector(outc_11_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[(i+cols/2) * rows + 3] = outc_11_0[i] + outc_11_1[i];
                }
            }

            if (thread_id == 12) {
                printf("Thread 12 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_12_0   = new int32_t[cols/2];
                int32_t* outc_12_1   = new int32_t[cols/2];
                setup_array(matA10, B01_cols[0], 0);
                execute_mvm(0);
                store_vector(outc_12_0, 0);
                setup_array(matA11, B11_cols[0], 0);
                execute_mvm(0);
                store_vector(outc_12_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[(i+cols/2) * rows + 4] = outc_12_0[i] + outc_12_1[i];
                }
            }

            if (thread_id == 13) {
                printf("Thread 13 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_13_0   = new int32_t[cols/2];
                int32_t* outc_13_1   = new int32_t[cols/2];
                setup_array(matA10, B01_cols[1], 0);
                execute_mvm(0);
                store_vector(outc_13_0, 0);
                setup_array(matA11, B11_cols[1], 0);
                execute_mvm(0);
                store_vector(outc_13_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[(i+cols/2) * rows + 5] = outc_13_0[i] + outc_13_1[i];
                }
            }

            if (thread_id == 14) {
                printf("Thread 14 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_14_0   = new int32_t[cols/2];
                int32_t* outc_14_1   = new int32_t[cols/2];
                setup_array(matA10, B01_cols[2], 0);
                execute_mvm(0);
                store_vector(outc_14_0, 0);
                setup_array(matA11, B11_cols[2], 0);
                execute_mvm(0);
                store_vector(outc_14_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[(i+cols/2) * rows + 6] = outc_14_0[i] + outc_14_1[i];
                }
            }
            if (thread_id == 15) {
                printf("Thread 15 bound to CPU %d\n", sched_getcpu());
                int32_t* outc_15_0   = new int32_t[cols/2];
                int32_t* outc_15_1   = new int32_t[cols/2];
                setup_array(matA10, B01_cols[3], 0);
                execute_mvm(0);
                store_vector(outc_15_0, 0);
                setup_array(matA11, B11_cols[3], 0);
                execute_mvm(0);
                store_vector(outc_15_1, 0);
                for (int i = 0; i < cols/2; i++) {
                    matC[(i+cols/2) * rows + 7] = outc_15_0[i] + outc_15_1[i];
                }
            }

            #pragma omp barrier // 同步屏障：所有线程在此处汇合
        //}
    }
    printf("C矩阵:\n");
    print_matrix(matC, rows, cols);

    return 0;
}
