#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <inttypes.h>

int32_t* matrixMultiply(int32_t* matA, int32_t* matB, int rows, int cols) {
    // 输出矩阵D
    int32_t* matD = new int32_t[rows * cols];

    // 为每一列分配一个向量数组，用来存储矩阵B的每一列
    int32_t** vecB = new int32_t*[cols];  // 存储矩阵B每一列的向量
    for (int i = 0; i < cols; i++) {
        vecB[i] = new int32_t[rows];  // 每一列作为一个大小为'rows'的向量
    }

    // 为存储计算结果分配临时向量
    int32_t* tempVec = new int32_t[rows];  // 临时向量存储每列的结果

    // 将矩阵B的每一列提取为向量存储
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            vecB[j][i] = matB[i * cols + j];  // 将矩阵B的第j列存储到vecB[j]中
        }
    }

    int status_flag;
    int tile_id = 0;

    // 对每一列执行矩阵-向量乘法
    for (int col_idx = 0; col_idx < cols; col_idx++) {
        // Step 1: 使用mvm.set指令将矩阵A设置到硬件
        asm volatile (
            "mvm.set %0, %1, %2"   // 将矩阵A设置到硬件
          : "=r" (status_flag)
          : "r"(matA), "r"(tile_id)
          : "memory"
        );

        // Step 2: 使用mvm.l指令加载矩阵B的第col_idx列（作为向量）到硬件
        asm volatile (
            "mvm.l %0, %1, %2"      // 将矩阵B的第col_idx列（vecB[col_idx]）加载到硬件
          : "=r" (status_flag)
          : "r"(vecB[col_idx]), "r"(tile_id)
          : "memory"
        );

        // Step 3: 执行矩阵A与矩阵B的第col_idx列（存储在vecB[col_idx]中的向量）进行矩阵-向量乘法
        asm volatile (
            "mvm %0, %1, x0"        // 执行矩阵A与vecB[col_idx]的矩阵-向量乘法
          : "=r" (status_flag)
          : "r"(tile_id)
        );

        // Step 4: 将计算结果存储到临时向量tempVec
        asm volatile (
            "mvm.s %0, %1, %2"      // 将计算结果存储到临时向量tempVec
          : "=r" (status_flag)
          : "r"(tempVec), "r"(tile_id)
          : "memory"
        );

        // 将临时向量的结果存储到输出矩阵matD的对应列
        for (int i = 0; i < rows; i++) {
            matD[i * cols + col_idx] = tempVec[i];  // 将结果存储到输出矩阵matD的第col_idx列
        }
    }

    // 释放内存
    for (int i = 0; i < cols; i++) {
        delete[] vecB[i];  // 释放每列的向量
    }
    delete[] vecB;         // 释放列向量数组
    delete[] tempVec;      // 释放临时向量

    return matD;  // 返回计算得到的矩阵D
}

int main() {
    int rows = 6, cols = 6;
    int32_t* matA = new int32_t[rows * cols];  // 第一个矩阵A
    int32_t* matB = new int32_t[rows * cols];  // 第二个矩阵B

    // 填充矩阵A并打印
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // 修改矩阵A为转置后的格式
            matA[j * cols + i] = j + 1;  // 每列的元素都相同，且等于列号（从1到6）
        }
    }
    printf("Matrix A:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matA[i * cols + j]);  // 打印输出矩阵D的元素
        }
        printf("\n");
    }

    // 填充矩阵B并打印
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matB[i * cols + j] = j + 1;  // 每列的元素都相同，且等于列号（从1到6）
        }
    }

    printf("Matrix B:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matB[i * cols + j]);  // 打印输出矩阵D的元素
        }
        printf("\n");
    }
    // 调用函数执行矩阵乘法
    int32_t* matD = matrixMultiply(matA, matB, rows, cols);
    //int32_t* matE = matrixMultiply(matA, matB, rows, cols);
    //int32_t* matF = matrixMultiply(matA, matB, rows, cols);

    // 打印输出矩阵D
    printf("\nOutput Matrix D (Matrix A * Matrix B):\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matD[i * cols + j]);  // 打印输出矩阵D的元素
        }
        printf("\n");
    }

    // 释放内存
    delete[] matA;         // 释放矩阵A
    delete[] matB;         // 释放矩阵B
    delete[] matD;         // 释放输出矩阵D

    return 0;
}






