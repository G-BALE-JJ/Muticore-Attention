#define _GNU_SOURCE

// =======================================
// 16-core 行间流水线 - 多进程版本（零 LLSC）
// 架构: 4行 × 4核/行，使用 remote_load/store
// 通信: 零 LLSC，零共享内存，纯 NoC 消息传递
// 同步: 轮询本地 Mailbox 的完成标志
// 模式: 每个进程绑定一个核心，独立执行对应的流水线阶段
// 关键: 无 mutex，无 atomic，无共享内存同步
// =======================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <unistd.h>
#include <cerrno>

// ==================== 配置参数 ====================
#define N 4                    // 向量维度
#define NUM_ROWS 4             // 流水线行数
#define CORES_PER_ROW 4        // 每行核心数
#define TOTAL_CORES 16         // 总核心数

// GlobalMemory 地址布局
#define GLOBAL_BASE 0x00000    // 起始地址
#define GLOBAL_STRIDE 0x4000   // 16KB per core
#define DATA_OFFSET 0x0000     // 数据区域起始偏移
#define MAILBOX_OFFSET 0x3F00  // Mailbox 在每个核心 GM 的末端 256B
// CPU 频率 (GHz) 用于周期转时间
#define CPU_FREQ_GHZ 2.0


// ==================== 内存屏障 ====================
#define MEMORY_BARRIER() __asm__ __volatile__("fence rw,rw" ::: "memory")

// ==================== 简单打印（无锁）====================
// 注意：多线程同时打印可能交错，但不会触发 LLSC
#define DEBUG_PRINT(...) do { \
    printf(__VA_ARGS__); \
    fflush(stdout); \
} while(0)

// ==================== MVM 指令封装 ====================

inline void set_matrix(const int32_t* mat, uint32_t tile_id) {
    int status;
    asm volatile("mvm.set %0, %1, %2" : "=r"(status) : "r"(mat), "r"(tile_id) : "memory");
}

inline void load_vector(const int32_t* vec, uint32_t tile_id) {
    int status;
    asm volatile("mvm.l %0, %1, %2" : "=r"(status) : "r"(vec), "r"(tile_id) : "memory");
}

inline void compute_mvm(uint32_t tile_id) {
    int status;
    asm volatile("mvm %0, %1, x0" : "=r"(status) : "r"(tile_id));
}

inline void store_vector(int32_t* dest, uint32_t tile_id) {
    int status;
    asm volatile("mvm.s %0, %1, %2" : "=r"(status) : "r"(dest), "r"(tile_id) : "memory");
}

inline void output_store(uint64_t dest_addr, uint32_t tile_id) {
    int status;
    asm volatile("mvm.ost %0, %1, %2" : "=r"(status) : "r"(dest_addr), "r"(tile_id) : "memory");
}

inline void input_load(uint64_t src_addr, uint32_t tile_id) {
    int status;
    asm volatile("mvm.ild %0, %1, %2" : "=r"(status) : "r"(src_addr), "r"(tile_id) : "memory");
}

// ==================== Main Memory <-> Global Memory 传输指令 ====================

inline void mvm_set_len(uint64_t byte_len) {
    int status;
    asm volatile (
        "mvm.slen %0, %1, x0"
        : "=r"(status)
        : "r"(byte_len)
        : "memory"
    );
}

inline void write_gm(const int32_t* mm_addr, uint64_t gm_addr) {
    int status;
    asm volatile (
        "write_gm %0, %1, %2"
        : "=r"(status)
        : "r"(mm_addr), "r"(gm_addr)
        : "memory"
    );
}

inline void write_mm(int32_t* mm_addr, uint64_t gm_addr) {
    int status;
    asm volatile (
        "write_mm %0, %1, %2"
        : "=r"(status)
        : "r"(mm_addr), "r"(gm_addr)
        : "memory"
    );
}

// ==================== 远程 GM 操作 ====================

inline void set_remote_transfer_len(uint64_t byte_len) {
    int status;
    asm volatile(
        "mvm.slen %0, %1, x0"
        : "=r"(status)
        : "r"(byte_len)
        : "memory");
}

inline void remote_load(uint64_t remote_global_addr, uint64_t local_global_addr) {
    int status;
    asm volatile(
        "mvm.rld %0, %1, %2"
        : "=r"(status)
        : "r"(remote_global_addr), "r"(local_global_addr)
        : "memory"
    );
}

inline void remote_store(uint64_t local_global_addr, uint64_t remote_global_addr) {
    int status;
    asm volatile(
        "mvm.rst %0, %1, %2"
        : "=r"(status)
        : "r"(local_global_addr), "r"(remote_global_addr)
        : "memory"
    );
}

// ==================== 地址计算辅助函数 ====================

inline uint64_t get_core_base_addr(int core_id) {
    return GLOBAL_BASE + (uint64_t)core_id * GLOBAL_STRIDE;
}

inline uint64_t get_core_data_addr(int core_id) {
    return get_core_base_addr(core_id) + DATA_OFFSET;
}

inline uint64_t get_core_mailbox_addr(int core_id) {
    return get_core_base_addr(core_id) + MAILBOX_OFFSET;
}

inline int32_t get_completion_flag(int col_id) {
    return (col_id + 1) * 2;
}

// ==================== 绑核工具 ====================

static inline uint64_t read_cycles() {
    uint64_t v;
    asm volatile("rdcycle %0" : "=r"(v));
    return v;
}

static void bind_process_to_core(int core_id) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);

    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        perror("sched_setaffinity");
    } else {
        DEBUG_PRINT("[Core%d] sched_setaffinity success\n", core_id);
    }
}

static int resolve_core_id(int requested_core) {
    int actual_core = sched_getcpu();
    if (actual_core < 0) {
        perror("sched_getcpu");
        if (requested_core >= 0) {
            DEBUG_PRINT("[Core%d] fallback to requested core id\n", requested_core);
            return requested_core;
        }
        fprintf(stderr, "ERROR: 无法获取实际核心 ID，sched_getcpu 失败 (errno=%d)\n", errno);
        exit(1);
    }

    if (requested_core >= 0 && requested_core != actual_core) {
        printf("[WARN] 请求绑定核心 %d，但实际运行在核心 %d。将使用实际核心 ID。\n",
               requested_core, actual_core);
    }

    return actual_core;
}


// ==================== 线程工作函数 ====================
bool run_pipeline_stage(int thread_id) {
    bool success = true;
    
    int row_id = thread_id / CORES_PER_ROW;
    int col_id = thread_id % CORES_PER_ROW;
    
    DEBUG_PRINT("\n[Thread%d] 启动: Row%d-Col%d\n", thread_id, row_id, col_id);
    
    // 地址配置
    uint64_t my_base_addr = get_core_base_addr(thread_id);
    uint64_t my_data_addr = get_core_data_addr(thread_id);
    uint64_t my_mailbox_addr = get_core_mailbox_addr(thread_id);
    
    DEBUG_PRINT("[Thread%d] 地址配置:\n", thread_id);
    DEBUG_PRINT("  BaseAddr    = 0x%lx\n", my_base_addr);
    DEBUG_PRINT("  DataAddr    = 0x%lx\n", my_data_addr);
    DEBUG_PRINT("  MailboxAddr = 0x%lx\n", my_mailbox_addr);
    
    // 分配对角矩阵（对角线为 2）
    int32_t matrix[N * N];
    for (int i = 0; i < N * N; ++i) matrix[i] = 0;
    for (int i = 0; i < N; ++i) matrix[i * N + i] = 2;
    
    // 分配输入向量（每列不同）
    int32_t input_vec[N];
    for (int i = 0; i < N; ++i) {
        input_vec[i] = (col_id + 1) * (i + 1);
    }
    
    // 分配工作缓冲区
    int32_t work_buffer[N];
    
    // 计算期望的同步信号值
    int32_t expected_flag = get_completion_flag(col_id);
    DEBUG_PRINT("[Thread%d] 期望同步信号值: %d\n", thread_id, expected_flag);
    
    // ==================== Row 0: 初始计算 ====================
    if (row_id == 0) {
        DEBUG_PRINT("[Thread%d] Row0: 开始初始计算\n", thread_id);
        DEBUG_PRINT("[Thread%d] 输入向量: [%d,%d,%d,%d]\n",
                   thread_id, input_vec[0], input_vec[1], input_vec[2], input_vec[3]);
        
        // 设置矩阵和向量 - tile_id 始终为 0
        set_matrix(matrix, 0);
        load_vector(input_vec, 0);
        
        // 计算
        compute_mvm(0);
        
        // 存储到 GlobalMemory
        output_store(my_data_addr, 0);
        
        // 验证结果
        store_vector(work_buffer, 0);
        DEBUG_PRINT("[Thread%d] 计算结果: [%d,%d,%d,%d]\n",
                   thread_id, work_buffer[0], work_buffer[1], work_buffer[2], work_buffer[3]);
        
        // 发送同步信号到下一行的同列核心
        if (row_id < NUM_ROWS - 1) {
            int next_thread = (row_id + 1) * CORES_PER_ROW + col_id;
            uint64_t next_mailbox = get_core_mailbox_addr(next_thread);
            uint64_t next_data_addr = get_core_data_addr(next_thread);
            DEBUG_PRINT("[Thread%d] 发送同步信号到 Thread%d (Mailbox=0x%lx, 信号值=%d)\n",
                       thread_id, next_thread, next_mailbox, work_buffer[0]);
            
            // 转发数据：本地 GM → 远程 GM
            set_remote_transfer_len(N * sizeof(int32_t));
            remote_store(my_data_addr, next_data_addr);
            MEMORY_BARRIER();
            
            // 将本地信号写入自己的 mailbox，再转发到下一行
            int32_t signal_value = work_buffer[0];
            mvm_set_len(sizeof(int32_t));
            write_gm(&signal_value, my_mailbox_addr);
            MEMORY_BARRIER();

            set_remote_transfer_len(sizeof(int32_t));
            remote_store(my_mailbox_addr, next_mailbox);  // 再转发到远程
            MEMORY_BARRIER();
        }
    }
    
    // ==================== Row 1-3: 流水线计算 ====================
    else {
        int prev_thread = (row_id - 1) * CORES_PER_ROW + col_id;
        DEBUG_PRINT("[Thread%d] Row%d: 等待 Thread%d 的同步信号 (期望值=%d)...\n",
                   thread_id, row_id, prev_thread, expected_flag);
        
        // 1. 轮询本地 mailbox，等待同步信号
        int32_t mailbox_value = 0;
        int retry_count = 0;
        const int MAX_RETRIES = 1000000;
        
        while (retry_count < MAX_RETRIES) {
            // 从本地 GM mailbox 读取到 Main Memory
            mvm_set_len(sizeof(int32_t));
            write_mm(&mailbox_value, my_mailbox_addr);
            MEMORY_BARRIER();
            
            // 检查是否匹配期望的同步信号
            if (mailbox_value == expected_flag) {
                DEBUG_PRINT("[Thread%d] 收到同步信号！(值=%d)\n", thread_id, mailbox_value);
                break;
            }
            
            retry_count++;
            if (retry_count % 100 == 0) {
                DEBUG_PRINT("[Thread%d] 轮询中... 当前值=%d (期望=%d)\n",
                           thread_id, mailbox_value, expected_flag);
            }
            
            // 短暂延迟避免过度轮询
            for (volatile int i = 0; i < 100; i++);
        }
        
        if (retry_count >= MAX_RETRIES) {
            DEBUG_PRINT("[Thread%d] ERROR: 等待超时！\n", thread_id);
            return false;
        }
        
        // 2. 数据已在数据区，直接加载并计算
        DEBUG_PRINT("[Thread%d] 开始计算...\n", thread_id);
        
        // 2a. 从上一行的数据区加载数据到本地 Tile
        uint64_t prev_data_addr = get_core_data_addr(prev_thread);
        
        // 步骤1: 远程 GM → 本地 GM
        set_remote_transfer_len(N * sizeof(int32_t));
        remote_load(prev_data_addr, my_data_addr);
        MEMORY_BARRIER();
        
        // 步骤2: 本地 GM → Tile
        set_matrix(matrix, 0);
        input_load(my_data_addr, 0);
        
        // 验证读取的数据
        store_vector(work_buffer, 0);
        // DEBUG_PRINT("[Thread%d] 读取到数据: [%d,%d,%d,%d]\n",
        //            thread_id, work_buffer[0], work_buffer[1], work_buffer[2], work_buffer[3]);
        
        // 2b. 计算
        compute_mvm(0);
        
        // 2c. 结果存储到 GlobalMemory
        output_store(my_data_addr, 0);
        
        // 验证结果
        store_vector(work_buffer, 0);
        DEBUG_PRINT("[Thread%d] 计算结果: [%d,%d,%d,%d]\n",
                   thread_id, work_buffer[0], work_buffer[1], work_buffer[2], work_buffer[3]);
        
        // 3. 如果不是最后一行，转发数据和同步信号到下一行
        if (row_id < NUM_ROWS - 1) {
            int next_thread = (row_id + 1) * CORES_PER_ROW + col_id;
            uint64_t next_data_addr = get_core_data_addr(next_thread);
            uint64_t next_mailbox_addr = get_core_mailbox_addr(next_thread);
            
            DEBUG_PRINT("[Thread%d] 转发数据到 Thread%d...\n", thread_id, next_thread);
            
            // 转发数据：本地 GM → 远程 GM
            set_remote_transfer_len(N * sizeof(int32_t));
            remote_store(my_data_addr, next_data_addr);
            MEMORY_BARRIER();
            

            set_remote_transfer_len(sizeof(int32_t));
            remote_store(my_mailbox_addr, next_mailbox_addr);  // 
            MEMORY_BARRIER();
            
            DEBUG_PRINT("[Thread%d] 转发完成！(信号值=%d)\n", thread_id, work_buffer[0]);
        }
    }
    
    // ==================== Row 3: 验证最终结果 ====================
    if (row_id == NUM_ROWS - 1) {
        store_vector(work_buffer, 0);
        
        // 计算预期结果（输入 × 2^4）
        int32_t expected[N];
        for (int i = 0; i < N; ++i) {
            expected[i] = input_vec[i] * 16;  // 2^4 = 16
        }
        
        DEBUG_PRINT("\n[Thread%d] *** 最终结果 ***\n", thread_id);
        DEBUG_PRINT("[Thread%d] 实际输出: [%d,%d,%d,%d]\n",
                   thread_id, work_buffer[0], work_buffer[1], work_buffer[2], work_buffer[3]);
        DEBUG_PRINT("[Thread%d] 预期输出: [%d,%d,%d,%d]\n",
                   thread_id, expected[0], expected[1], expected[2], expected[3]);
        
        // 验证
        bool pass = true;
        for (int i = 0; i < N; ++i) {
            if (work_buffer[i] != expected[i]) {
                pass = false;
                break;
            }
        }
        
        success = pass;
        DEBUG_PRINT("[Thread%d] %s 验证！\n", thread_id, pass ? "✓" : "✗");
    }
    
    DEBUG_PRINT("[Thread%d] 完成！\n\n", thread_id);
    return success;
}

// ==================== 主函数 ====================
int main(int argc, char* argv[]) {
    printf("\n========================================\n");
    printf("16-Core 流水线测试 - 多进程版本（零 LLSC）\n");
    printf("架构: 4行 × 4核/行\n");
    printf("通信: Mailbox (NoC)\n");
    printf("模式: 16 个进程，每个进程映射 1 个核心\n");
    printf("特性: 无 mutex，无 atomic，无共享内存同步\n");
    printf("========================================\n\n");
    
    int requested_core = -1;
    if (argc >= 2) {
        requested_core = atoi(argv[1]);
        if (requested_core < 0 || requested_core >= TOTAL_CORES) {
            printf("ERROR: core_id=%d 无效，应在 [0, %d) 范围内。\n", requested_core, TOTAL_CORES);
            return 1;
        }
        bind_process_to_core(requested_core);
    } else {
        printf("[INFO] 未提供 core_id 参数，将完全依赖 sched_getcpu()。\n");
    }

    int actual_core = resolve_core_id(requested_core);
    if (actual_core < 0 || actual_core >= TOTAL_CORES) {
        printf("ERROR: 实际核心 ID=%d 越界 (TOTAL_CORES=%d)。\n", actual_core, TOTAL_CORES);
        return 1;
    }

        uint64_t c_start = read_cycles();

        bool ok = run_pipeline_stage(actual_core);

        uint64_t c_end = read_cycles();
        uint64_t c_delta = c_end - c_start;
        double ns = (double)c_delta / CPU_FREQ_GHZ;           // 2.0 GHz => 0.5 ns per cycle
        double us = ns / 1000.0;
        double ms = ns / 1e6;
        printf("[TIMING] run_pipeline_stage 周期数: %llu cycles, %.0f ns (%.3f us, %.3f ms)\n",
            (unsigned long long)c_delta, ns, us, ms);

    printf("\n========================================\n");
    printf("核心 %d 流水线阶段执行%s\n", actual_core, ok ? "成功" : "失败");
    printf("========================================\n\n");

    return ok ? 0 : 1;
}