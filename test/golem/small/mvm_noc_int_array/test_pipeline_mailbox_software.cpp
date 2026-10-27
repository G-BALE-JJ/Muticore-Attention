#define _GNU_SOURCE
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include "gm_config.h"
#include "ex_instr.h"

// ----------------------------------------------------------------------------
// DEBUG VERSION: Add detailed timing logs to diagnose barrier cycle anomaly
// ----------------------------------------------------------------------------

#ifndef BUF_SIZE_BYTES
#define BUF_SIZE_BYTES 256
#endif

#define BUF_SIZE_INTS (BUF_SIZE_BYTES / sizeof(int32_t))

volatile int32_t g_shared_buf[BUF_SIZE_INTS] __attribute__((aligned(64)));
int32_t g_local_buf[BUF_SIZE_INTS] __attribute__((aligned(64)));

void pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

int main() {
    printf("[OMP] Starting DEBUG version (%dB).\n", BUF_SIZE_BYTES);

    // Timing variables - separate for each thread to avoid race conditions
    uint64_t thread0_barrier_entry_cyc = 0;
    uint64_t thread0_barrier_exit_cyc = 0;
    uint64_t thread1_barrier_entry_cyc = 0;
    uint64_t thread1_barrier_exit_cyc = 0;
    
    // Work timing
    uint64_t thread0_work_start_cyc = 0;
    uint64_t thread0_work_end_cyc = 0;
    uint64_t thread1_work_start_cyc = 0;
    uint64_t thread1_work_end_cyc = 0;
    
    // Sender statistics
    uint64_t sender_total_start_cyc = 0, sender_total_end_cyc = 0;
    uint64_t sender_mm2gm_start_cyc = 0, sender_mm2gm_end_cyc = 0;
    
    // Receiver statistics
    uint64_t receiver_total_start_cyc = 0, receiver_total_end_cyc = 0;
    uint64_t receiver_remote_ld_start_cyc = 0, receiver_remote_ld_end_cyc = 0;
    uint64_t receiver_gm2mm_start_cyc = 0, receiver_gm2mm_end_cyc = 0;

    omp_set_num_threads(2);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        pin_to_core(tid);

        // Initialize buffers
        for (int i = 0; i < BUF_SIZE_INTS; i++) {
            g_shared_buf[i] = 0;
            g_local_buf[i] = i;
        }
        printf("[DEBUG] Thread %d pinned to core %d\n", tid, tid);

        if (tid == 0) { // Sender
            thread0_work_start_cyc = read_cycles();
            printf("[DEBUG-T0] Work started at cycle %lu\n", thread0_work_start_cyc);
            
            sender_total_start_cyc = read_cycles();
            
            uint64_t core_0_gm_base = get_core_data_addr(0);
            uint64_t core_1_gm_base = get_core_data_addr(1);
            uint64_t transfer_len = BUF_SIZE_INTS * sizeof(int32_t);
            
            sender_mm2gm_start_cyc = read_cycles();
            set_len(transfer_len);
            mm2gm((int32_t*)g_local_buf, core_0_gm_base);
            sender_mm2gm_end_cyc = read_cycles();
            
            sender_total_end_cyc = read_cycles();
            thread0_work_end_cyc = read_cycles();
            
            printf("[DEBUG-T0] Work ended at cycle %lu (duration: %lu)\n", 
                   thread0_work_end_cyc, thread0_work_end_cyc - thread0_work_start_cyc);
            printf("[Sender] Data transfer completed\n");
            
            // Record barrier entry time
            thread0_barrier_entry_cyc = read_cycles();
            printf("[DEBUG-T0] Entering barrier at cycle %lu\n", thread0_barrier_entry_cyc);
        } else { // Receiver (tid == 1)
            // Record when receiver reaches barrier
            thread1_barrier_entry_cyc = read_cycles();
            printf("[DEBUG-T1] Entering barrier at cycle %lu\n", thread1_barrier_entry_cyc);
        }

        #pragma omp barrier

        if (tid == 0) {
            thread0_barrier_exit_cyc = read_cycles();
            printf("[DEBUG-T0] Exited barrier at cycle %lu (barrier wait: %lu)\n", 
                   thread0_barrier_exit_cyc, thread0_barrier_exit_cyc - thread0_barrier_entry_cyc);
        } else { // Receiver
            thread1_barrier_exit_cyc = read_cycles();
            printf("[DEBUG-T1] Exited barrier at cycle %lu (barrier wait: %lu)\n", 
                   thread1_barrier_exit_cyc, thread1_barrier_exit_cyc - thread1_barrier_entry_cyc);
            
            thread1_work_start_cyc = read_cycles();
            printf("[DEBUG-T1] Work started at cycle %lu\n", thread1_work_start_cyc);
            
            receiver_total_start_cyc = read_cycles();
            
            uint64_t core_0_gm_base = get_core_data_addr(0);
            uint64_t core_1_gm_base = get_core_data_addr(1);
            uint64_t transfer_len = BUF_SIZE_INTS * sizeof(int32_t);
            
            receiver_remote_ld_start_cyc = read_cycles();
            set_len(transfer_len);
            remote_load(core_0_gm_base, core_1_gm_base);
            receiver_remote_ld_end_cyc = read_cycles();
            
            receiver_gm2mm_start_cyc = read_cycles();
            set_len(transfer_len);
            gm2mm((int32_t*)g_shared_buf, core_1_gm_base);
            receiver_gm2mm_end_cyc = read_cycles();
            
            receiver_total_end_cyc = read_cycles();
            thread1_work_end_cyc = read_cycles();
            
            printf("[DEBUG-T1] Work ended at cycle %lu (duration: %lu)\n", 
                   thread1_work_end_cyc, thread1_work_end_cyc - thread1_work_start_cyc);
        }
    }
    
    // Calculate statistics
    uint64_t sender_sw_cyc = (sender_total_end_cyc - sender_total_start_cyc) - 
                             (sender_mm2gm_end_cyc - sender_mm2gm_start_cyc);
    
    uint64_t receiver_hw_cyc = (receiver_remote_ld_end_cyc - receiver_remote_ld_start_cyc) +
                               (receiver_gm2mm_end_cyc - receiver_gm2mm_start_cyc);
    
    uint64_t receiver_sw_cyc = (receiver_total_end_cyc - receiver_total_start_cyc) - receiver_hw_cyc;
    
    // Calculate barrier metrics
    uint64_t thread0_barrier_wait = thread0_barrier_exit_cyc - thread0_barrier_entry_cyc;
    uint64_t thread1_barrier_wait = thread1_barrier_exit_cyc - thread1_barrier_entry_cyc;
    uint64_t thread0_work_duration = thread0_work_end_cyc - thread0_work_start_cyc;
    
    // Original calculation (for comparison)
    uint64_t original_barrier_calc = thread1_barrier_exit_cyc - thread0_barrier_entry_cyc;
    
    printf("\n========== DIAGNOSTIC ANALYSIS ==========\n");
    printf("[DIAG] Thread 0 work duration: %lu cycles\n", thread0_work_duration);
    printf("[DIAG] Thread 0 barrier wait: %lu cycles\n", thread0_barrier_wait);
    printf("[DIAG] Thread 1 barrier wait: %lu cycles\n", thread1_barrier_wait);
    printf("[DIAG] Thread 1 arrived at barrier at: %lu\n", thread1_barrier_entry_cyc);
    printf("[DIAG] Thread 0 arrived at barrier at: %lu\n", thread0_barrier_entry_cyc);
    printf("[DIAG] Time difference (T1 entry - T0 entry): %ld cycles\n", 
           (int64_t)(thread1_barrier_entry_cyc - thread0_barrier_entry_cyc));
    printf("[DIAG] Original barrier calculation: %lu cycles\n", original_barrier_calc);
    printf("[DIAG] Max barrier wait (actual barrier overhead): %lu cycles\n", 
           thread0_barrier_wait > thread1_barrier_wait ? thread0_barrier_wait : thread1_barrier_wait);
    
    printf("\n========== Performance Breakdown ==========\n");
    printf("[STATS] Data_Size: %d\n", BUF_SIZE_BYTES);
    printf("[STATS] Barrier_Cycles: %lu\n", original_barrier_calc);

    
    
    printf("\n--- Sender (Core 0) ---\n");
    printf("[STATS] Sender_Total_Cycles: %lu\n", sender_total_end_cyc - sender_total_start_cyc);
    printf("[STATS] Sender_HW_Cycles: %lu\n", sender_mm2gm_end_cyc - sender_mm2gm_start_cyc);
    printf("[STATS] Sender_SW_Cycles: %lu\n", sender_sw_cyc);
    
    printf("\n--- Receiver (Core 1) ---\n");
    printf("[STATS] Receiver_Total_Cycles: %lu\n", receiver_total_end_cyc - receiver_total_start_cyc);
    printf("[STATS] Receiver_HW_Cycles: %lu\n", receiver_hw_cyc);
    printf("[STATS] Receiver_SW_Cycles: %lu\n", receiver_sw_cyc);
    printf("[STATS] RemoteLoad_Cycles: %lu\n", receiver_remote_ld_end_cyc - receiver_remote_ld_start_cyc);
    printf("==========================================\n");
    
    return 0;
}