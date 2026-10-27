#define _GNU_SOURCE
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include "core_bind.h"
#include "gm_config.h"
#include "ex_instr.h"

// ----------------------------------------------------------------------------
// Mailbox同步测试 - 双向握手协议（优化版：使用reg2gm发送信号）
//
// 优化：在进程开始时将信号值放入寄存器，使用reg2gm直接写入mailbox
// 避免每次发送信号都调用mm2gm，减少开销
//
// 流程：
// Core 0:                          Core 1:
// 1. mm2gm                         1. 轮询"mm2gm完成"信号
// 2. reg2gm发送"mm2gm完成"信号 ──→ 2. 检测到信号 → 记录起始时间
// 3. 等待"就绪"信号 ←──────────── 3. reg2gm发送"就绪"信号
// 4. remote_store data             4. 轮询"数据完成"信号
// 5. reg2gm发送"数据完成"信号 ──→ 5. 检测到信号 → 记录结束时间
//                                  6. Pure sync = 结束时间 - 起始时间
//                                  7. gm2mm
// ----------------------------------------------------------------------------

#ifndef BUF_SIZE_BYTES
#define BUF_SIZE_BYTES 256
#endif

#define BUF_SIZE_INTS (BUF_SIZE_BYTES / sizeof(int32_t))

// Mailbox flag offsets
#define MAILBOX_FLAG_MM2GM_DONE  0x00  // Core 0 mm2gm完成信号
#define MAILBOX_FLAG_READY       0x08  // Core 1 就绪信号
#define MAILBOX_FLAG_DATA_DONE   0x10  // Core 0 数据传输完成信号

// Signal values (stored in registers) - 使用U后缀确保是无符号类型
#define SIGNAL_MM2GM_DONE  0x11111111U
#define SIGNAL_READY       0x22222222U
#define SIGNAL_DATA_DONE   0x33333333U

// 测试：打印宏的实际值
static const uint32_t TEST_SIGNAL_MM2GM_DONE = SIGNAL_MM2GM_DONE;
static const uint32_t TEST_SIGNAL_READY = SIGNAL_READY;
static const uint32_t TEST_SIGNAL_DATA_DONE = SIGNAL_DATA_DONE;

// Global buffers
int32_t g_local_buf[BUF_SIZE_INTS] __attribute__((aligned(64)));
volatile int32_t g_shared_buf[BUF_SIZE_INTS] __attribute__((aligned(64)));

int main(int argc, char* argv[]) {
    int requested_core = -1;
    if (argc >= 2) {
        requested_core = atoi(argv[1]);
        if (requested_core < 0 || requested_core >= TOTAL_CORES) {
            printf("ERROR: core_id=%d 无效，应在 [0, %d) 范围内。\n", requested_core, TOTAL_CORES);
            return 1;
        }
        bind_process_to_core(requested_core);
    }

    int core_id = resolve_core_id(requested_core);
    if (core_id < 0 || core_id >= TOTAL_CORES) {
        printf("ERROR: 实际核心 ID=%d 越界 (TOTAL_CORES=%d)。\n", core_id, TOTAL_CORES);
        return 1;
    }

    // Initialize buffers
    for (int i = 0; i < BUF_SIZE_INTS; i++) {
        g_shared_buf[i] = 0;
        g_local_buf[i] = i;
    }

    // Calculate addresses
    uint64_t core_0_data_addr = get_core_data_addr(0);
    uint64_t core_1_data_addr = get_core_data_addr(1);
    uint64_t core_0_mailbox_addr = get_core_mailbox_addr(0);
    uint64_t core_1_mailbox_addr = get_core_mailbox_addr(1);
    uint64_t transfer_len = BUF_SIZE_INTS * sizeof(int32_t);

    // printf("[Core %d] Addresses: data=0x%lx, mailbox=0x%lx\n",
    //        core_id,
    //        core_id == 0 ? core_0_data_addr : core_1_data_addr,
    //        core_id == 0 ? core_0_mailbox_addr : core_1_mailbox_addr);
    
    // 打印信号值以确认
    printf("[Core %d] Signal values: MM2GM_DONE=0x%x, READY=0x%x, DATA_DONE=0x%x\n",
           core_id, TEST_SIGNAL_MM2GM_DONE, TEST_SIGNAL_READY, TEST_SIGNAL_DATA_DONE);

    if (core_id == 0) { // ========== Sender (Core 0) ==========
        printf("[Sender] Starting mailbox sync test (%dB).\n", BUF_SIZE_BYTES);
        
        // Step 1: mm2gm
        uint64_t mm2gm_start = read_cycles();
        set_len(transfer_len);
        mm2gm(g_local_buf, core_0_data_addr);
        uint64_t mm2gm_end = read_cycles();
        // printf("[Sender] Step 1: mm2gm completed (%lu cycles)\n", mm2gm_end - mm2gm_start);
        
        // Step 2: 使用mm2gm发送"mm2gm完成"信号
        uint64_t signal_start = read_cycles();
        uint64_t local_mm2gmdonesignal_addr = core_0_mailbox_addr + MAILBOX_FLAG_MM2GM_DONE;
        uint64_t remote_mm2gmdonesignal_addr = core_1_mailbox_addr + MAILBOX_FLAG_MM2GM_DONE;
        
        // 准备信号数据
        uint64_t signal_value = SIGNAL_MM2GM_DONE;
        printf("[Sender] Data load done signal value: 0x%016lx\n", signal_value);
        set_len(sizeof(uint64_t));
        mm2gm((void*)&signal_value, local_mm2gmdonesignal_addr);

        
        // 通过remote_store发送到Core 1
        set_len(sizeof(uint64_t));
        remote_store(local_mm2gmdonesignal_addr, remote_mm2gmdonesignal_addr);
        uint64_t signal_end = read_cycles();
        printf("[Sender] Step 2: Sent mm2gm_done signal via mm2gm (%lu cycles)\n",
               signal_end - signal_start);
        
        // Step 3: 等待Core 1的"就绪"信号
        uint64_t poll_addr_ready = core_0_mailbox_addr + MAILBOX_FLAG_READY;
        printf("[Sender] Step 3: Waiting for ready signal from Core 1 at addr 0x%lx...\n", poll_addr_ready);
        int wait_count = 0;
        const int wait_iterations = 1000000;
        while (wait_count < wait_iterations) {
            uint64_t raw_value = 0;
            set_len(sizeof(uint64_t));
            gm2mm((void*)&raw_value, poll_addr_ready);
            uint32_t ready_flag = (uint32_t)(raw_value & 0xFFFFFFFF);
            printf("poll_addr_ready_64=0x%016lx\n", raw_value);
            printf("poll_addr_ready_32=0x%08x\n", ready_flag);
            printf("exp_value=0x%08x\n", SIGNAL_READY);
            if (ready_flag == SIGNAL_READY) {
                printf("[DEBUG-Sender] Core 1 is ready!\n");
                break;
            }

            wait_count++;
            
            if (wait_count % 100 == 0) {
            }
            
            for (volatile int i = 0; i < 100; i++);
        }
        printf("[Sender] Step 3: Received ready signal (%d iterations)\n", wait_count);
        
        // Step 4: remote_store data
        uint64_t remote_store_start = read_cycles();
        set_len(transfer_len);
        remote_store(core_0_data_addr, core_1_data_addr);
        uint64_t remote_store_end = read_cycles();
        printf("[Sender] Step 4: remote_store completed (%lu cycles)\n", 
               remote_store_end - remote_store_start);
        
        // Step 5: 使用mm2gm发送"数据完成"信号
        signal_start = read_cycles();
        uint64_t local_datasignal_addr = core_0_mailbox_addr + MAILBOX_FLAG_DATA_DONE;
        uint64_t remote_datasignal_addr = core_1_mailbox_addr + MAILBOX_FLAG_DATA_DONE;
        
        // 准备信号数据
        signal_value = SIGNAL_DATA_DONE;
        set_len(sizeof(uint64_t));
        mm2gm((void*)&signal_value, local_datasignal_addr);
        
        // 通过remote_store发送到Core 1
        set_len(sizeof(uint64_t));
        remote_store(local_datasignal_addr, remote_datasignal_addr);
        signal_end = read_cycles();
        printf("[Sender] Step 5: Sent data_done signal via mm2gm (%lu cycles)\n",
               signal_end - signal_start);
        
        printf("[Sender] All steps completed\n");
        
    } else if (core_id == 1) { // ========== Receiver (Core 1) ==========
        // printf("[Receiver] Starting mailbox sync test...\n");
        
        // Step 1: 轮询"mm2gm完成"信号
        uint64_t poll_addr_begin = core_1_mailbox_addr + MAILBOX_FLAG_MM2GM_DONE;
        
        // printf("[Receiver] Step 1: Polling for mm2gm_done signal at addr 0x%lx...\n", poll_addr_begin);
        uint64_t poll_start_time = read_cycles();
        int poll1_iterations = 0;
        const int poll1_max_iterations = 1000000;
        while (poll1_iterations < poll1_max_iterations) {
            uint64_t raw_value = 0;
            set_len(sizeof(uint64_t));
            gm2mm((void*)&raw_value, poll_addr_begin);
            uint32_t mm2gm_done_flag = (uint32_t)(raw_value & 0xFFFFFFFF);
            // printf("poll_addr_begin_64=0x%016lx\n", raw_value);
            // printf("poll_addr_begin_32=0x%08x\n", mm2gm_done_flag);
            // printf("exp_value=0x%08x\n", SIGNAL_MM2GM_DONE);

            
            if (mm2gm_done_flag == SIGNAL_MM2GM_DONE) {
                // printf("[DEBUG-Receiver] mm2gm_done signal detected!\n");
                break;
            }

            poll1_iterations++;
            
            if (poll1_iterations % 100 == 0) {
            }
            
            for (volatile int i = 0; i < 100; i++);
        }
        
        // Step 2: 记录起始时间

        // printf("[Receiver] Step 2: Detected mm2gm_done signal, recorded start time (cycle %lu)\n", 
        //        sync_start_cyc);
        
        // Step 3: 使用mm2gm发送"就绪"信号
        uint64_t signal_start = read_cycles();

        uint64_t local_readysignal_addr = core_1_mailbox_addr + MAILBOX_FLAG_READY;
        uint64_t remote_readysignal_addr = core_0_mailbox_addr + MAILBOX_FLAG_READY;
        
        // 准备信号数据
        uint64_t signal_value = SIGNAL_READY;
        uint64_t sync_start_cyc = read_cycles();
        set_len(sizeof(uint64_t));
        mm2gm((void*)&signal_value, local_readysignal_addr);
        
        // 通过remote_store发送到Core 0
        set_len(sizeof(uint64_t));
        remote_store(local_readysignal_addr, remote_readysignal_addr);

        uint64_t signal_end = read_cycles();
        // printf("[Receiver] Step 3: Sent ready signal via mm2gm (%lu cycles)\n",
        //        signal_end - signal_start);
        
        // Step 4: 轮询"数据完成"信号
        uint64_t poll_addr_data = core_1_mailbox_addr + MAILBOX_FLAG_DATA_DONE;
        // printf("[Receiver] Step 4: Polling for data_done signal at addr 0x%lx...\n", poll_addr_data);
        int poll2_iterations = 0;
        const int poll2_max_iterations = 1000000;

        while (poll2_iterations < poll2_max_iterations) {
            uint64_t raw_value = 0;
            set_len(sizeof(uint64_t));
            gm2mm((void*)&raw_value, poll_addr_data);
            uint32_t data_done_flag = (uint32_t)(raw_value & 0xFFFFFFFF);
            // printf("poll_addr_data_64=0x%016lx\n", raw_value);
            // printf("poll_addr_data_32=0x%08x\n", data_done_flag);
            // printf("exp_value=0x%08x\n", SIGNAL_DATA_DONE);

            
            if (data_done_flag == SIGNAL_DATA_DONE) {
                // printf("[DEBUG-Receiver] data_done signal detected!\n");
                break;
            }

            poll2_iterations++;

            if (poll2_iterations % 100 == 0) {
            }
            
            for (volatile int i = 0; i < 100; i++);

        }
        // Step 5: 记录结束时间
        uint64_t sync_end_cyc = read_cycles();
        // printf("[Receiver] Step 5: Detected data_done signal, recorded end time (cycle %lu)\n", 
        //        sync_end_cyc);
        
        // Step 6: 计算Pure Mailbox Sync
        uint64_t pure_mailbox_sync = sync_end_cyc - sync_start_cyc;
        // printf("[Receiver] *** PURE MAILBOX SYNC OVERHEAD: %lu cycles ***\n", pure_mailbox_sync);
        // printf("[Receiver] (Start: %lu, End: %lu, Duration: %lu)\n",
        //        sync_start_cyc, sync_end_cyc, pure_mailbox_sync);
        
        // Step 7: gm2mm
        uint64_t gm2mm_start = read_cycles();
        set_len(transfer_len);
        gm2mm((int32_t*)g_shared_buf, core_1_data_addr);
        uint64_t gm2mm_end = read_cycles();
        // printf("[Receiver] Step 7: gm2mm completed (%lu cycles)\n", gm2mm_end - gm2mm_start);
        
        printf("[Receiver] Data verification: %d %d %d %d\n",
               g_shared_buf[0], g_shared_buf[1], g_shared_buf[2], g_shared_buf[3]);
        
        // Print statistics (compatible format)
        printf("\n========== Receiver Statistics ==========\n");
        printf("[STATS] Data_Size: %d\n", BUF_SIZE_BYTES);
        printf("[STATS] Barrier_Cycles: %lu\n", pure_mailbox_sync);
        printf("[STATS] Sender_Total_Cycles: 0\n");
        printf("[STATS] Sender_HW_Cycles: 0\n");
        printf("[STATS] Sender_SW_Cycles: 0\n");
        printf("[STATS] Receiver_Total_Cycles: %lu\n", (sync_start_cyc - poll_start_time) + pure_mailbox_sync + (gm2mm_end - gm2mm_start));
        printf("[STATS] Receiver_HW_Cycles: %lu\n", (gm2mm_end - gm2mm_start));
        printf("[STATS] Receiver_SW_Cycles: 0\n");
        printf("[STATS] RemoteLoad_Cycles: %lu\n", (gm2mm_end - gm2mm_start));
        
        printf("[DEBUG] Pure_Mailbox_Sync_Cycles: %lu\n", pure_mailbox_sync);
        printf("[DEBUG] Poll1_Iterations: %d\n", poll1_iterations);
        printf("[DEBUG] Poll2_Iterations: %d\n", poll2_iterations);
        printf("[DEBUG] GM2MM_Cycles: %lu\n", gm2mm_end - gm2mm_start);
        printf("=========================================\n");
        
        // printf("\n========== Performance Comparison ==========\n");
        // printf("[INFO] Software Barrier Overhead (baseline): ~20,000-86,000 cycles\n");
        // printf("[INFO] Pure Mailbox Sync Overhead (this test): %lu cycles\n", pure_mailbox_sync);
        
        // if (pure_mailbox_sync < 1000) {
        //     printf("[INFO] Performance: ✅ Excellent (>95%% reduction)\n");
        // } else if (pure_mailbox_sync < 5000) {
        //     printf("[INFO] Performance: ✅ Good (>75%% reduction)\n");
        // } else if (pure_mailbox_sync < 20000) {
        //     printf("[INFO] Performance: ⚠️  Moderate improvement\n");
        // } else {
        //     printf("[INFO] Performance: ❌ Needs investigation\n");
        // }
        // printf("============================================\n");
    }
    return 0;
}
