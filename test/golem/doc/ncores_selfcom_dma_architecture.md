# ncores_selfcom_dma 架构说明

范围：本文档基于自通信 DMA 测试配置及其依赖的构建器与组件，形成可复用的架构说明，避免后续重复理解。

## 入口文件（配置 + 构建器）
- 配置文件： [golem/tests/ncores_selfcom_dma.py](golem/tests/ncores_selfcom_dma.py)
- CPU 构建器： [golem/tests/cpu_builder.py](golem/tests/cpu_builder.py)
- NoC 构建器： [golem/tests/noc_builder.py](golem/tests/noc_builder.py)
- GlobalMemory 实现： [golem/globalmemory/globalmemory.h](golem/globalmemory/globalmemory.h) 与 [golem/globalmemory/globalmemory.cc](golem/globalmemory/globalmemory.cc)
- GM 地址辅助： [golem/tests/small/mvm_noc_int_array/gm_config.h](golem/tests/small/mvm_noc_int_array/gm_config.h)
- RoCC 指令集： [golem/tests/small/mvm_noc_int_array/ex_instr.h](golem/tests/small/mvm_noc_int_array/ex_instr.h)

## 系统概览
- 核心：16 个 Vanadis RISC-V 核心，每核 1 个硬件线程。
- 进程：每核 1 进程，单线程（多进程共享地址空间），以 arg1=core_id 启动。
- NoC：4x5 mesh（20 个路由器），每路由器本地端口数为 3。
- 内存：4 个 NUMA 节点，对应路由器 16-19。默认总内存 256MiB（每节点 64MiB）。
- 一致性协议：MESI。

## 拓扑
Mesh 结构（路由器 ID）：
- CPU 路由器：0-15（16 核）
- 内存路由器：16-19（NUMA 节点）
- OS 路由器：16（NodeOS L1 缓存挂载）

Mesh 由 [golem/tests/noc_builder.py](golem/tests/noc_builder.py) 创建，使用 merlin.hr_router 与 merlin.mesh。链路带宽 25GB/s，flit 72B，链路延迟 1ns，输入/输出缓冲 8KB，虚拟网络数 3。

## CPU + Cache 层次（每核）
定义见 [golem/tests/cpu_builder.py](golem/tests/cpu_builder.py)。
- Vanadis CPU 核心（decoder、LSQ、branch unit、TLB）。
- L1 D-cache：32KB，8 路，2 cycles，64B line，MSHR 32。
- L1 I-cache：32KB，8 路，2 cycles，64B line，MSHR 16，next-block 预取。
- L2 cache：1MB，16 路，14 cycles，64B line。
- L2 <-> NoC：MemNIC，group=1，destinations=100-103，link_bw=50GB/s，num_vns=3。

CPU_Builder 返回端口供 [golem/tests/ncores_selfcom_dma.py](golem/tests/ncores_selfcom_dma.py) 连接 L2 与 GlobalMemory 到 mesh。

## RoCC + Golem 加速器
同样定义于 [golem/tests/cpu_builder.py](golem/tests/cpu_builder.py)。
- RoCC 子组件（默认 golem.RoCCAnalogInt），搭配 MVM 阵列后端。
- 指令集见 [golem/tests/small/mvm_noc_int_array/ex_instr.h](golem/tests/small/mvm_noc_int_array/ex_instr.h)：
  - mvm.*（load/compute/store）、remote.ld/remote.st、mm2gm/gm2mm 等。
- 每核提供一个 GlobalMemory 子组件。

## GlobalMemory（每核）
实现见 [golem/globalmemory/globalmemory.h](golem/globalmemory/globalmemory.h) 与 [golem/globalmemory/globalmemory.cc](golem/globalmemory/globalmemory.cc)。
- 容量：每核 64KB（GLOBAL_STRIDE=0x10000）。
- 基址：GLOBAL_BASE=0x00000，因此 core N 的基址为 N*0x10000。
- GlobalMemory 通过 SimpleNetwork 接入 NoC（merlin.linkcontrol）。
- DMA 与 RDMA 复用同一 link_control（无独立 DMA NIC）。

地址辅助见 [golem/tests/small/mvm_noc_int_array/gm_config.h](golem/tests/small/mvm_noc_int_array/gm_config.h)。
- Data 区： [base, base + 0xFEFF]
- Mailbox 区： [base + 0xFF00, base + 0xFFFF]
- DMA 完成标志区：每个 64KB 窗口尾部 32 字节。

## DMA + Identity Window
关键行为见 [golem/globalmemory/globalmemory.cc](golem/globalmemory/globalmemory.cc)。
- identityWindowBase 默认 0x04000000（可通过 GOLEM_IDENTITY_BASE 配置）。
- 任意 GM 请求 addr >= identityWindowBase 时，走 DMA 访问主存。
- DMA 通过 StandardMem 接口访问 memHierarchy，并将完成标志写回 GM 尾部区域。
- DMA 路由所需的每节点大小由 GOLEM_MEM_NODE_SIZE 配置（在 ncores_selfcom_dma.py 中设置）。

## NodeOS + MMU
见 [golem/tests/ncores_selfcom_dma.py](golem/tests/ncores_selfcom_dma.py)。
- NodeOS：vanadis.VanadisNodeOS。
- MMU：simpleMMU。
- OS L1 cache：32KB，inclusive，64B line，MESI，通过 MemNIC 连接 NoC（group=1，destinations=100-103）。
- OS L1 挂载到 router 16（与 NUMA node 0 共享）。

## NUMA 内存节点
见 [golem/tests/ncores_selfcom_dma.py](golem/tests/ncores_selfcom_dma.py)。
- 4 个节点映射到 router 16-19。
- 每节点：DirectoryController + MemController + dramsim3 后端。
- 地址空间按 memBytesPerNode 连续切分。
- Node 0 使用 malloc backing（initBacking=0）。
- Node 1-3 使用 mmap backing（hbm_init_node1.bin、hbm_init_node2.bin、hbm_init_zero.bin）。

## 统计
- 统计输出：CSV，路径 stats/stats_selfcom.txt。
- 所有组件启用统计（sst.AccumulatorStatistic）。

---

# 设计优化建议

以下建议面向当前配置，可在不改变编程模型的前提下评估。

## 1) 带宽对齐，避免人为瓶颈
- L2 MemNIC 为 50GB/s，但 mesh 链路为 25GB/s，存在过订阅。
- 建议：要么提升 mesh link/xbar 到 50GB/s；要么降低 L2 与 GlobalMemory link_bw 到 25GB/s，以保持一致性。

## 2) 使用虚拟网络区分流量类型
- RDMA、DMA 与缓存流量共用 num_vns=3，但未显式分流。
- 建议：预留 VN（如 0=coherence，1=DMA，2=RDMA），降低 HOL 阻塞并提升可解释性。

## 3) 校验地址映射与 NUMA 切分
- DMA 路由依赖 GOLEM_MEM_NODE_SIZE 与 identityWindowBase。
- 建议：确保 GOLEM_MEM_NODE_SIZE 与 physMemSize/NUM_MEMORY_NODES 一致，并检查 DMA 目标与 DirectoryController 地址范围一致。

## 4) 本地端口资源留量
- local_ports=3 足够当前连接，但扩展空间有限。
- 建议：若后续加入更多设备（NIC/额外 GM 端口），提高 local_ports 或引入聚合层。

## 5) L2 大小/延迟与工作负载匹配
- L2 为 1MB/14 cycles，L1 较小但延迟低。
- 建议：若 mailbox 流量占主导，可考虑降低 L2 容量或让 GM/remote 路径绕过 L2，减少排队。

## 6) DMA 完成标志位置
- DMA 完成标志占用每个 GM 窗口尾部 32 字节。
- 建议：软件侧避免使用该尾部区域作为数据有效载荷，防止覆盖。

## 7) OS L1 与内存节点共用路由器 16
- OS L1 与 DirCtrl0 共挂 router 16。
- 建议：若 OS 流量较重，可考虑将 OS L1 移至独立路由器，或提高 router 16 的端口/带宽。
