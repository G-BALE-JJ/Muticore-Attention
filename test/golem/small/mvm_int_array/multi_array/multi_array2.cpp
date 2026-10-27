#include <cstdio>
#include <cstdint>
#include <omp.h>
#include <sched.h>

// ===================== RoCC 指令封装（保持你原来的写法） =====================
inline void set_matrix(const int32_t* mat, uint32_t tile_id) {
    int status;
    asm volatile(
        "mvm.set %0, %1, %2"
        : "=r"(status)
        : "r"(mat), "r"(tile_id)
        : "memory");
}

inline void load_vector(const int32_t* vec, uint32_t tile_id) {
    int status;
    asm volatile(
        "mvm.l %0, %1, %2"
        : "=r"(status)
        : "r"(vec), "r"(tile_id)
        : "memory");
}

inline void compute_mvm(uint32_t tile_id) {
    int status;
    asm volatile(
        "mvm %0, %1, x0"
        : "=r"(status)
        : "r"(tile_id));
}

inline void store_vector(int32_t* dest, uint32_t tile_id) {
    int status;
    asm volatile(
        "mvm.s %0, %1, %2"
        : "=r"(status)
        : "r"(dest), "r"(tile_id)
        : "memory");
}

// （以下几个保留，但本测试不再使用）
inline void output_store(int32_t* dest_addr, uint32_t tile_id) {
    int status;
    asm volatile(
        "mvm.ost %0, %1, %2"
        : "=r"(status)
        : "r"(dest_addr), "r"(tile_id)
        : "memory");
}
inline void input_load(int32_t* src_addr, uint32_t tile_id) {
    int status;
    asm volatile(
        "mvm.ild %0, %1, %2"
        : "=r"(status)
        : "r"(src_addr), "r"(tile_id)
        : "memory");
}
inline void remote_store(int32_t* local_addr, int32_t* remote_addr) {
    int status;
    asm volatile(
        "mvm.rst %0, %1, %2"
        : "=r"(status)
        : "r"(local_addr), "r"(remote_addr)
        : "memory");
}
inline void remote_load(uint64_t remote_addr, int32_t* local_addr) {
    int status;
    asm volatile(
        "mvm.rld %0, %1, %2"
        : "=r"(status)
        : "r"(remote_addr), "r"(local_addr)
        : "memory");
}

// 绑定线程到指定 CPU
static inline void bind_thread_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

int main() {
    constexpr int N = 6;         // 矩阵与向量维度
    constexpr int ITER = 50;     // 执行 10 次

    // 分配一批矩阵/输入/输出
    int32_t* matrices     = new int32_t[ITER * N * N];
    int32_t* input_vecs   = new int32_t[ITER * N];
    int32_t* output_vecs  = new int32_t[ITER * N];

    // 生成“不同”的测试数据：
    // 第 k 次：矩阵 = diag(k+1)，向量[i] = (i+1) + 10*k
    for (int k = 0; k < ITER; ++k) {
        int32_t* M = matrices + k * N * N;
        int32_t* x = input_vecs + k * N;
        // 清零
        for (int i = 0; i < N * N; ++i) M[i] = 0;
        // 对角线
        for (int i = 0; i < N; ++i) M[i * N + i] = (k + 1);
        // 向量
        for (int i = 0; i < N; ++i) x[i] = (i + 1) + 10 * k;
    }

    omp_set_num_threads(2);
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        bind_thread_to_cpu(tid);

        if (tid == 0) {
            printf("Core0 (tid=%d) on CPU %d: 执行 %d 次 MVM（tile0）\n", tid, sched_getcpu(), ITER);
            for (int k = 0; k < ITER; ++k) {
                int32_t* M = matrices + k * N * N;
                int32_t* x = input_vecs + k * N;
                int32_t* y = output_vecs + k * N;

                // 1) 加载矩阵到 tile0
                set_matrix(M, 0);
                // 2) 加载向量到 tile0 输入
                load_vector(x, 0);
                // 3) 执行 MVM
                compute_mvm(0);
                // 4) 将结果写回主存 y
                store_vector(y, 0);

                // 简单打印一行核对（可按需关闭）
                printf("  iter %2d: M=diag(%2d), x=[", k, (k + 1));
                for (int i = 0; i < N; ++i) printf("%d%s", x[i], (i < N - 1 ? ", " : "] "));
                printf("=> y=[");
                for (int i = 0; i < N; ++i) printf("%d%s", y[i], (i < N - 1 ? ", " : "]\n"));
            }
        } else if (tid == 1) {
            // 核心1空转：不进行任何 RoCC 访问
            printf("Core1 (tid=%d) on CPU %d: 空闲（不执行 RoCC 指令）\n", tid, sched_getcpu());
        }
        // 并行区末尾有隐式 barrier，Core1 会在这里等待 Core0 完成
    }

    delete[] matrices;
    delete[] input_vecs;
    delete[] output_vecs;
    return 0;
}
