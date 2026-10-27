#pragma once

#include <cstdint>

static constexpr uint32_t GOLEM_ROCC_FUNC7_TILE_MVM_BATCH = 0x11;
static constexpr uint32_t GOLEM_ROCC_FUNC7_TILE_WAIT_BATCH = 0x12;
static constexpr uint32_t GOLEM_ROCC_FUNC7_TILE_GM2IMAT_BCAST = 0x13;
static constexpr uint32_t GOLEM_ROCC_FUNC7_TILE_GM2IVEC_BATCH = 0x14;
static constexpr uint32_t GOLEM_ROCC_FUNC7_WCP_START = 0x15;
static constexpr uint32_t GOLEM_ROCC_FUNC7_WCP_WAIT = 0x16;

// Stock GNU binutils do not know the project-specific instruction mnemonics.
#define GOLEM_R_INSN(F7, RD, RS1, RS2) ".insn r 0x0b, 7, " F7 ", " RD ", " RS1 ", " RS2

static inline void mvm_set_matrix(const void* mm_addr) {
    long dummy;
    asm volatile (
        GOLEM_R_INSN("0x1", "%0", "%1", "x0")
        : "=r"(dummy)        // output: rd 
        : "r"(mm_addr)       // input:  rs1
        : "memory"
    );
}

// =============================================================
// 2. 加载输入向量 (mvm.l)
// -------------------------------------------------------------
// 用法: mvm_load_vector(主存地址)
// 汇编: mvm.l rd, rs1, x0
// =============================================================
static inline void mvm_load_vector(const void* mm_addr) {
    long dummy;
    asm volatile (
        GOLEM_R_INSN("0x2", "%0", "%1", "x0")
        : "=r"(dummy)
        : "r"(mm_addr)
        : "memory"
    );
}

// =============================================================
// 3. 启动计算 (mvm)
// -------------------------------------------------------------
// 用法: mvm_compute(阵列ID)
// 汇编: mvm rd, rs1, x0
// =============================================================
static inline void mvm_compute(uint64_t array_id) {
    long dummy;
    asm volatile (
        GOLEM_R_INSN("0x3", "%0", "%1", "x0")
        : "=r"(dummy)
        : "r"(array_id)
        : "memory"
    );
}

// Queue-friendly variant: submit compute without destination writeback.
static inline void mvm_compute_async(uint64_t array_id) {
    asm volatile (
        GOLEM_R_INSN("0x3", "x0", "%0", "x0")
        :
        : "r"(array_id)
        : "memory"
    );
}

static inline void tile_mvm_batch_async(uint64_t start_array, uint64_t count) {
    asm volatile(
        ".insn r 0x0b, 7, %2, x0, %0, %1"
        :
        : "r"(start_array), "r"(count), "i"(GOLEM_ROCC_FUNC7_TILE_MVM_BATCH)
        : "memory");
}

static inline void tile_mvm_batch_wait(uint64_t start_array, uint64_t count) {
    long dummy;
    asm volatile(
        ".insn r 0x0b, 7, %3, %0, %1, %2"
        : "=r"(dummy)
        : "r"(start_array), "r"(count), "i"(GOLEM_ROCC_FUNC7_TILE_WAIT_BATCH)
        : "memory");
}

static inline void tile_gm2imat_broadcast(uint64_t gm_addr, uint64_t count) {
    asm volatile(
        ".insn r 0x0b, 7, %2, x0, %0, %1"
        :
        : "r"(gm_addr), "r"(count), "i"(GOLEM_ROCC_FUNC7_TILE_GM2IMAT_BCAST)
        : "memory");
}

static inline void tile_gm2ivec_batch(uint64_t gm_base_addr, uint64_t count) {
    asm volatile(
        ".insn r 0x0b, 7, %2, x0, %0, %1"
        :
        : "r"(gm_base_addr), "r"(count), "i"(GOLEM_ROCC_FUNC7_TILE_GM2IVEC_BATCH)
        : "memory");
}

static inline void wcp_start(uint64_t desc_gm_addr) {
    asm volatile(
        ".insn r 0x0b, 7, %1, x0, %0, x0"
        :
        : "r"(desc_gm_addr), "i"(GOLEM_ROCC_FUNC7_WCP_START)
        : "memory");
}

static inline void wcp_wait() {
    long dummy;
    asm volatile(
        ".insn r 0x0b, 7, %1, %0, x0, x0"
        : "=r"(dummy)
        : "i"(GOLEM_ROCC_FUNC7_WCP_WAIT)
        : "memory");
}

// =============================================================
// 4. 存储输出向量 (mvm.s)
// -------------------------------------------------------------
// 用法: mvm_store_vector(主存地址, 源阵列ID)
// 汇编: mvm.s rd, rs1, rs2
// =============================================================
static inline void mvm_store_vector(void* mm_addr, uint64_t array_id) {
    long dummy;
    asm volatile (
        GOLEM_R_INSN("0x4", "%0", "%1", "%2")
        : "=r"(dummy)
        : "r"(mm_addr), "r"(array_id)
        : "memory"
    );
}

// =============================================================
// 5. 向量搬运 (mvm.mv)
// -------------------------------------------------------------
// 用法: mvm_move_vector(源阵列ID, 目标阵列ID)
// 汇编: mvm.mv rd, rs1, rs2
// =============================================================
static inline void mvm_move_vector(uint64_t src_array_id, uint64_t dst_array_id) {
    long dummy;
    asm volatile (
        GOLEM_R_INSN("0x5", "%0", "%1", "%2")
        : "=r"(dummy)
        : "r"(src_array_id), "r"(dst_array_id)
        : "memory"
    );
}

// =============================================================
// 6. 本地 Global Memory 存储 (mvm.ovec2gm)
// -------------------------------------------------------------
// 用法: outputvectorstore(GM偏移地址, 源阵列ID)
// 汇编: mvm.ovec2gm rd, rs1, rs2
// =============================================================
static inline void outputvectorstore(uint64_t local_gm_addr, uint64_t array_id) {
    long dummy;
    asm volatile (
        GOLEM_R_INSN("0x6", "%0", "%1", "%2")
        : "=r"(dummy)
        : "r"(local_gm_addr), "r"(array_id)
        : "memory"
    );
}

// =============================================================
// 7. 本地 Global Memory 加载 (mvm.gm2ivec)
// -------------------------------------------------------------
// 用法: inputvectorload(GM偏移地址, 目标阵列ID)
// 汇编: mvm.gm2ivec rd, rs1, rs2
// =============================================================
static inline void inputvectorload(uint64_t local_gm_addr, uint64_t array_id) {
    long dummy;
    asm volatile (
        GOLEM_R_INSN("0x7", "%0", "%1", "%2")
        : "=r"(dummy)
        : "r"(local_gm_addr), "r"(array_id)
        : "memory"
    );
}

// Queue-friendly variant: submit without destination writeback.
static inline void inputvectorload_async(uint64_t local_gm_addr, uint64_t array_id) {
    asm volatile (
        GOLEM_R_INSN("0x7", "x0", "%0", "%1")
        :
        : "r"(local_gm_addr), "r"(array_id)
        : "memory"
    );
}

// =============================================================
// 8. 本地 Global Memory 加载 (mvm.gm2imat)
// -------------------------------------------------------------
// 用法: inputvectorload(GM偏移地址, 目标阵列ID)
// 汇编: mvm.gm2imat rd, rs1, rs2
// =============================================================
static inline void inputmatrixload(uint64_t local_gm_addr, uint64_t array_id) {
    long dummy;
    asm volatile (
        GOLEM_R_INSN("0x8", "%0", "%1", "%2")
        : "=r"(dummy)
        : "r"(local_gm_addr), "r"(array_id)
        : "memory"
    );
}

// Queue-friendly variant: submit without destination writeback.
static inline void inputmatrixload_async(uint64_t local_gm_addr, uint64_t array_id) {
    asm volatile (
        GOLEM_R_INSN("0x8", "x0", "%0", "%1")
        :
        : "r"(local_gm_addr), "r"(array_id)
        : "memory"
    );
}

// =============================================================
// 9. 远端存储 (remote.st)
// -------------------------------------------------------------
// 用法: remote_store(本地GM地址, 远端GM地址)
// 汇编: remote.st rd, rs1, rs2
// =============================================================
static inline void remote_store(uint64_t local_gm_addr, uint64_t remote_gm_addr) {
    asm volatile (
        GOLEM_R_INSN("0x9", "x0", "%0", "%1")
        :
        : "r"(local_gm_addr), "r"(remote_gm_addr)
        : "memory"
    );
}

// =============================================================
// 10. 远端加载 (remote.ld)
// -------------------------------------------------------------
// 用法: remote_load(远端GM地址, 本地GM地址)
// 汇编: remote.ld, rs1, rs2
// =============================================================
static inline void remote_load(uint64_t remote_gm_addr, uint64_t local_gm_addr) {
    asm volatile (
        GOLEM_R_INSN("0xA", "x0", "%0", "%1")
        :
        : "r"(remote_gm_addr), "r"(local_gm_addr)
        : "memory"
    );
}

// =============================================================
// 11. 设置传输长度 (mvm.slen)
// -------------------------------------------------------------
// 用法: mvm_set_len(字节长度)
// 汇编: mvm.slen rd, rs1, x0
// =============================================================
static inline void set_len(uint64_t byte_len) {
    asm volatile (
        GOLEM_R_INSN("0xB", "x0", "%0", "x0")
        :
        : "r"(byte_len)
        : "memory"
    );
}

// =============================================================
// 12. 配置输出模式 (mvm.ocfg)
// -------------------------------------------------------------
// 用法: mvm_config_output(命令字, 目标阵列ID)
// 汇编: mvm.ocfg rd, rs1, rs2
// =============================================================
// 配置输出模式：mode 0=覆盖，1=累加；clear_now=1 会额外触发一次清空命令
inline void configure_output_mode(uint32_t tile_id, uint32_t mode, bool clear_now) {
    int status;
    const uint32_t mode_cmd = mode ? 1u : 0u;
    asm volatile(
        GOLEM_R_INSN("0xC", "%0", "%1", "%2")
        : "=r"(status)
        : "r"(mode_cmd), "r"(tile_id)
        : "memory");
    if (clear_now) {
        const uint32_t clear_cmd = 2u;
        asm volatile(
            GOLEM_R_INSN("0xC", "%0", "%1", "%2")
            : "=r"(status)
            : "r"(clear_cmd), "r"(tile_id)
            : "memory");
    }
}

// =============================================================
// 13. 主存 -> Global Memory 
// -------------------------------------------------------------
// 用法: mm2gm(主存地址, GM地址)
// 汇编: mm2gm rd, rs1, rs2
// =============================================================
static inline void mm2gm(const void* mm_addr, uint64_t gm_addr) {
    asm volatile (
        GOLEM_R_INSN("0xD", "x0", "%0", "%1")
        :
        : "r"(mm_addr), "r"(gm_addr)
        : "memory"
    );
}

// =============================================================
// 14. Global Memory -> 主存 
// -------------------------------------------------------------
// 用法: gm2mm(主存地址, GM地址)
// 汇编: gm2mm rd, rs1, rs2
// =============================================================
static inline void gm2mm(void* mm_addr, uint64_t gm_addr) {
    asm volatile (
        GOLEM_R_INSN("0xE", "x0", "%0", "%1")
        :
        : "r"(mm_addr), "r"(gm_addr)
        : "memory"
    );
}

// =============================================================
// 15. 寄存器 -> Global Memory 
// -------------------------------------------------------------
// 用法: reg2gm(数值, GM地址)
// 汇编: reg2gm rd, rs1, rs2
// =============================================================
static inline void reg2gm(uint64_t value, uint64_t gm_addr) {
    asm volatile (
        GOLEM_R_INSN("0xF", "x0", "%0", "%1")
        :
        : "r"(value), "r"(gm_addr)
        : "memory"
    );
}
// =============================================================
// 16. Global Memory -> 寄存器 
// -------------------------------------------------------------
// 用法: value = gm2reg(GM地址)
// 汇编: gm2reg rd, rs1, x0
// =============================================================
static inline uint64_t gm2reg(uint64_t gm_addr) {
    uint64_t value;
    asm volatile (
        GOLEM_R_INSN("0x10", "%0", "%1", "x0")
        : "=r"(value)     // output: rd (target register)
        : "r"(gm_addr)    // input: rs1 (GM address)
        : "memory"
    );
    return value;
}

// =============================================================
// 17. 读取cycle counter (RISC-V CSR)
// -------------------------------------------------------------
// 用法: cycles = read_cycles()
// 读取 cycle CSR寄存器
// =============================================================
// static inline uint64_t read_cycles() {
//     uint64_t cycles;
//     asm volatile (
//         "rdcycle %0"
//         : "=r"(cycles)
//         :
//         :
//     );
//     return cycles;
// }

// =============================================================
// 18. 读取instruction counter (RISC-V CSR)
// -------------------------------------------------------------
// 用法: instrs = read_instret()
// 读取 instret CSR寄存器（已退休指令数）
// =============================================================
static inline uint64_t read_instret() {
    uint64_t instrs;
    asm volatile (
        "rdinstret %0"
        : "=r"(instrs)
        :
        :
    );
    return instrs;
}

#undef GOLEM_R_INSN

// // 获取核心的Global Memory基地址
// // 每个核心的GM基地址 = 核心ID * 64KB (0x10000)
// static inline uint64_t get_core_data_addr(int core_id) {
//     return (uint64_t)core_id * 0x10000;
// }
