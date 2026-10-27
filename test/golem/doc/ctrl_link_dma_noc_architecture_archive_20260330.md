# Golem Ctrl Link / DMA / NoC 架构归档

范围：基于 `tests/run_noc_dma_pipeline.sh` 的默认 control-link 运行链路，对多核组织、控制网络、DMA 机制与 NoC 流量做归档，作为后续问题讨论与优化分析的共同基线。

## 1. 入口与默认运行链路

- 入口脚本：`tests/run_noc_dma_pipeline.sh`
- 默认会自动加载：`tests/configs/default.env`
- `default.env` 继续串联加载：
  - `10_core_gemm.env`
  - `20_dma.env`
  - `30_network.env`
  - `40_debug_io.env`
  - `50_tensor_verify.env`
  - `60_run.env`
- 默认架构脚本：`tests/architecture/ncores_selfcom_dma_ctrl.py`
- 默认二进制：`tests/small/mvm_noc_int_array/riscv64/test_noc_dma`

默认 control 配置来自 `60_run.env`：

- `GOLEM_CTRL_LINK_ENABLE=1`
- `GOLEM_REQUEST_SCHEDULER_ENABLE=1`
- `GOLEM_GROUP_MANAGER_ENABLE=1`
- `GOLEM_CTRL_OVERLAP_AB=1`
- `GOLEM_ARCH_SCRIPT=architecture/ncores_selfcom_dma_ctrl.py`

默认网络与 DMA 配置来自 `20_dma.env` 和 `30_network.env`：

- `GOLEM_DMA_BURST_BYTES=512`
- `GOLEM_DMA_MAX_INFLIGHT=8`
- `GOLEM_DMA_READ_RETRY_TICKS=32`
- `GOLEM_DMA_READ_MAX_RETRIES=8`
- `GOLEM_NUM_MEMORY_NODES=5`
- `GOLEM_MEM_NODE_SIZE_BYTES=134217728`
- `GOLEM_IDENTITY_BASE=${GOLEM_MEM_NODE_SIZE_BYTES}`
- `GOLEM_NOC_LINK_BW=100GB/s`
- `GOLEM_NOC_XBAR_BW=100GB/s`
- `GOLEM_NOC_FLIT_SIZE=128B`

默认核心配置来自 `10_core_gemm.env`：

- `GOLEM_TOTAL_GROUPS=4`
- `GOLEM_TOTAL_CORES=20`
- `GOLEM_TOTAL_GEMM_CORES=20`

脚本里还有一个关键自动修正：如果用户沿用旧的 `16/16` 配置，在 `group manager + ctrl link` 打开时，会自动补一整行 manager 核，将核心数扩展为 `20`，见 `run_noc_dma_pipeline.sh` 中 `758-762` 行附近。

## 2. 系统级拓扑

架构脚本 `tests/architecture/ncores_selfcom_dma_ctrl.py` 使用 `MeshNoCBuilder` 构建 2D mesh。

### 2.1 Mesh 结构

- `MESH_DIM_X = 4`
- `cpu_rows = ceil(numCpus / 4)`
- `MESH_DIM_Y = cpu_rows + 2`
- 顶部 1 行放 data memory routers
- 中间若干行放 CPU routers
- 底部 1 行放 OS router

在默认 `20` 核下：

- CPU 行数 `cpu_rows = 5`
- 总 mesh 为 `4 x 7`

### 2.2 路由器资源

NoC 构建器位于 `tests/architecture/noc_builder.py`：

- 每个 router 本地端口数：`local_ports = 3`
- 4 个方向端口固定保留给 mesh cardinal links
- 本地设备通过 `attach_local()` 连接到 router 的额外本地端口

### 2.3 行布局

默认 `top_hbm` 模式下：

- 第 0 行：data memory routers
- 第 1~5 行：CPU routers
- 第 6 行：OS router

`GOLEM_NUM_MEMORY_NODES=5` 表示：

- Node 0：OS memory node
- Node 1~4：data / HBM nodes

脚本会检查 `data memory nodes <= mesh_dim_x`，确保所有数据内存节点可以在单独一行按列铺开。

## 3. 多核组织方式

### 3.1 核心角色

每个 core 由 `CPU_Builder` 构建，包含：

- Vanadis CPU
- L1 I/D cache
- L2 cache
- RoCC golem accelerator
- 每核一个 `GlobalMemory`

在 control-link 模式下，还会额外挂接两个子组件：

- `GroupCtrlEndpoint`
- `RequestSchedulerEndpoint`

### 3.2 分组规则

分组规则定义在 `ncores_selfcom_dma_ctrl.py`：

- `group_id = core_id % MESH_DIM_X`

因此分组不是按行，而是按列。

默认 `20` 核、`4` 列时：

- 一共 `4` 个 group
- 每组沿一列向下分布
- 每组最顶上的 core 是 manager
- 每组其余 `4` 个 core 是 worker

### 3.3 固定 4 worker slot

`GroupCtrlEndpoint` 和 `RequestSchedulerEndpoint` 都只为 manager 暴露：

- `req_in_0`
- `req_in_1`
- `req_in_2`
- `req_in_3`

因此当前设计隐含约束是：每个 group 固定管理 `4` 个 worker。脚本也会对此做一致性检查。

## 4. 本地存储与地址空间

### 4.1 每核 GlobalMemory

每个 core 挂一个 `GlobalMemoryImplement`：

- `baseAddr = GLOBAL_BASE + core_id * GLOBAL_STRIDE`
- `size = GLOBAL_STRIDE`

默认 `GOLEM_GLOBAL_STRIDE_KB=256`，因此每核 GM 窗口大小是 `256KB`。

### 4.2 GM 尾部 flag 区

`gm_config.h` 和 `globalmemory.cc` 在每个 core 的 GM 尾部预留 DMA 完成标记区，用于 software polling：

- read slot 0：seq + flag
- read slot 1：seq + flag
- write slot：seq + flag
- read slot selector

这个区域不是普通 payload 区，软件不能与有效数据混用。

### 4.3 Identity Window

DMA 访问主存依赖 identity window：

- `identityWindowBase = GOLEM_IDENTITY_BASE`
- 默认 `GOLEM_IDENTITY_BASE = GOLEM_MEM_NODE_SIZE_BYTES`

这意味着：

- `0 ~ Node0-1`：OS 物理空间
- `>= Node1 base`：被 Golem 视为“可 DMA 的主存数据空间”

对当前默认值：

- Node 0: `0x00000000 - 0x07ffffff`
- Node 1: `0x08000000 - 0x0fffffff`
- Node 2: `0x10000000 - 0x17ffffff`
- Node 3: `0x18000000 - 0x1fffffff`
- Node 4: `0x20000000 - 0x27ffffff`

软件里所有 `remote.ld(src>=IDENTITY_BASE, dst_gm)` 都会被导向 DMA 路径。

## 5. 控制面主线

当前架构最核心的特点是：控制面和数据面分离。

### 5.1 第一层：GroupCtrlEndpoint

位置：`groupctrl/groupctrl.cc`

职责：做组内准入控制，决定哪个 worker 可以对哪个 memory node 发起下一批 DMA。

worker 软件通过本地 GM mailbox 发布请求：

- `CTRL_LOCAL_REQ_SEQ_OFF`
- `CTRL_LOCAL_REQ_VALID_OFF`
- `CTRL_LOCAL_REQ_SRC_OFF`
- `CTRL_LOCAL_REQ_DST_OFF`
- `CTRL_LOCAL_REQ_BYTES_OFF`
- `CTRL_LOCAL_REQ_NODE_OFF`
- `CTRL_LOCAL_REQ_WINDOW_OFF`

worker 侧 endpoint 会轮询这些 mailbox：

- 当看到 `REQ_VALID=1` 时，向本组 manager 发送 `REQUEST`
- 当软件写入 `DONE_VALID=1` 时，向 manager 发送 `DONE`
- 当软件写入 `FINISHED=1` 时，向 manager 发送 `FINISHED`

manager 侧维护：

- `pendingQ_`
- `workers_[]` 状态
- `inflightPerNode_[]`

调度条件：

- 该 worker 当前没有 inflight 请求
- `targetNode` 未超过 `maxInflightPerNode`
- 每轮最多发送 `maxGrantsPerSchedule` 个 grant

如果满足，manager 发送 `GRANT`，worker 将其写回本地 mailbox：

- `CTRL_LOCAL_GRANT_SEQ_OFF`
- `CTRL_LOCAL_GRANT_WINDOW_OFF`

所有 worker `FINISHED` 且组内请求全部 drain 后，manager 广播 `GROUP_DONE`。

### 5.2 第二层：RequestSchedulerEndpoint

位置：`requestscheduler/requestscheduler.cc`

职责：manager 代 worker 将 DMA 请求真正发到 NoC。

worker 软件通过本地 scheduler mailbox 发布 `SUBMIT`：

- `SCHED_LOCAL_SUBMIT_ID_OFF`
- `SCHED_LOCAL_SUBMIT_SRC_OFF`
- `SCHED_LOCAL_SUBMIT_DST_OFF`
- `SCHED_LOCAL_SUBMIT_BYTES_OFF`
- `SCHED_LOCAL_SUBMIT_NODE_OFF`
- `SCHED_LOCAL_SUBMIT_FLAG_ADDR_OFF`
- `SCHED_LOCAL_SUBMIT_FLAG_VALUE_OFF`

worker endpoint 轮询这些字段后发送 `SUBMIT` 给 manager。

manager endpoint 维护：

- `pendingQ_`
- `nodeCredits_[]`

调度条件：

- 请求目标 node 有 credit
- manager 的 network link 可发送

发送成功后：

- 对应 `targetNode` 的 credit 减一
- 等 worker 软件后续通过 `DONE` 归还 credit

### 5.3 两层控制面的关系

当前不是单层调度，而是两层：

- `GroupCtrl`: 决定“是否允许这个 worker 对该 node 发起下一批 DMA”
- `RequestScheduler`: 决定“manager 何时真正把这个 DMA 请求打进网络”

所以这是一个“准入控制 + 代发调度”的组合设计。

## 6. DMA 数据面机制

DMA 数据面主要由 `GlobalMemoryImplement` 实现。

### 6.1 进入 DMA 路径的条件

在 `GlobalMemoryImplement::rd_to_network()` 和 `wr_to_network()` 中：

- 如果地址 `< identityWindowBase`，按普通 remote GM 流量处理
- 如果地址 `>= identityWindowBase`，则转为 DMA 主存流量

也就是说，软件层的 `remote.ld` / `remote.st` 指令在运行时由地址空间决定是：

- remote GM 访问
- 还是 DMA 主存访问

### 6.2 DMA read 分块

`dma_read_from_host_to_globalmem()` 会：

- 按 `dma_burst_bytes` 切块，默认 `512B`
- 为每个 chunk 创建 `PendingDmaOp`
- 将所有 chunk 放入 `dma_pending`

### 6.3 在途窗口

DMA read 在发包前还会经过 `issue_pending_dma_read_window()`：

- 统计当前已发出的 read chunk 数
- 若达到 `dma_read_max_inflight` 则停止继续发包
- 否则继续发出新的 chunk

因此每核 DMA 并发度是由 `GlobalMemory` 本地窗口进一步约束的。

### 6.4 重试机制

每个 DMA read chunk 还带有：

- `retry_ticks_left`
- `retry_attempts`

通过 `schedule_dma_retry_event()` 和 `process_dma_read_retries()` 驱动重试，避免 chunk 永久丢失或卡死。

### 6.5 完成通知

DMA read 完成后，worker 端 `GlobalMemory` 收到 `DMA_READ_COMPLETE`：

- 将数据写入本地 GM 指定目的地址
- 将完成 flag 写为对应 seq
- 软件通过轮询 flag 判定 DMA 完成

这也是为什么软件在发起 DMA 前要先：

- 自增 seq
- 清零 flag
- 再发 `remote.ld`

## 7. 一次完整事务的时序

以 worker 请求一批 A/B 数据为例：

1. worker 软件调用 `ctrl_publish_request_local()` 写本地 ctrl mailbox
2. 本地 `GroupCtrlEndpoint(worker)` 轮询到 `REQ_VALID=1`
3. worker endpoint 向 manager 发送 `REQUEST`
4. manager 根据 `inflightPerNode` 和 worker 状态决定是否 `GRANT`
5. worker 收到 `GRANT`，本地 mailbox 中 `GRANT_SEQ` 更新
6. worker 软件继续执行 `scheduler_submit_read_ticket_slot()`
7. 软件将 `SUBMIT` 写入本地 scheduler mailbox
8. 本地 `RequestSchedulerEndpoint(worker)` 轮询到 `SUBMIT_VALID=1`
9. worker scheduler endpoint 向 manager scheduler endpoint 发送 `SUBMIT`
10. manager scheduler endpoint 检查 `nodeCredits[targetNode]`
11. manager 代 worker 通过 NoC 发出真正的 READ 请求
12. memory node 返回 `DMA_READ_COMPLETE`
13. 数据不经过 manager，直接回到 worker 的 `GlobalMemory`
14. worker GM 写入本地数据并设置 completion flag
15. worker 软件轮询 flag，确认 DMA 完成
16. worker 软件再发布 `DONE` 给 scheduler / ctrl 层
17. manager 回收 node inflight / credit

这里最重要的一点是：

- 请求是 manager 代发
- 返回数据是 memory node 直接回 worker

因此 manager 串行化的是“发起权限”，不是“完整数据回传路径”。

## 8. NoC 流量分类

当前网络里的流量至少可以分成四类。

### 8.1 Cache / coherence 流量

- 来源：L2 MemNIC、OS L1 MemNIC
- 目的：DirectoryController highlink
- 属于 memHierarchy 一致性流量

### 8.2 普通 remote GM 流量

- 来源：各 core 的 `GlobalMemory.link_control`
- 用途：core 之间 `remote.ld/remote.st`
- 地址在 per-core GM window 内

### 8.3 DMA 主存流量

- 来源：各 core 的 `GlobalMemory.link_control`，或 manager scheduler 的 `linkControl_`
- 请求类型：`READ` / `DMA_WRITE`
- 响应类型：`DMA_READ_COMPLETE` / `DMA_WRITE_COMPLETE`
- 目标：memory node 对应的 dirctrl MemNIC endpoint

### 8.4 控制链路流量

- `GroupCtrlEndpoint` 的 `REQUEST/GRANT/DONE/FINISHED/GROUP_DONE`
- `RequestSchedulerEndpoint` 的 `SUBMIT/DONE`

这部分不是走 mesh，而是 SST direct link。

## 9. 当前设计的主要优点

- 控制面与数据面解耦，ctrl traffic 不占 mesh 带宽
- manager 对热点 memory node 有显式限流能力
- 数据回包直达 worker，不需要 manager 二次转发
- DMA、completion flag、software polling 协议比较清晰，便于调试
- 每核 GM + 每组 manager 的职责边界明确

## 10. 当前设计的主要约束与潜在瓶颈

### 10.1 固定 group 大小

manager 端口固定只有 4 个 worker slot，导致 group size 基本硬编码为 `1 manager + 4 worker`。

### 10.2 双重节流

当前至少有三层限流：

- `GroupCtrl.maxInflightPerNode`
- `RequestScheduler.nodeCredits`
- `GlobalMemory.dma_read_max_inflight`

如果参数不协调，容易形成过度串行化。

### 10.3 credit 回收路径偏长

DMA 完成后并不是硬件直接回收 scheduler credit，而是：

- worker GM 写 flag
- worker 软件轮询到完成
- worker 软件再发 `DONE`
- manager 再回收 credit

因此 credit 回收晚于真实的网络完成时刻。

### 10.4 DMA 与普通 GM 请求共用同一类 networkIF

`GlobalMemory` 里 request/reply 只做了基础 VN 区分，没有把 DMA request、DMA response、普通 remote GM 明确隔离为独立 traffic class，存在 HOL blocking 风险。

### 10.5 HBM 热点由地址布局直接决定

memory node 映射规则很直接：

- `node = phys_addr / memNodeSize`

如果 GEMM tile 地址分布不均，就会天然形成 node hotspot。

## 11. 后续讨论建议

建议后续讨论按下面顺序推进：

1. 先区分瓶颈在 control 面还是 data 面
2. 再区分是 GroupCtrl 过紧，还是 scheduler / HBM / NoC 过载
3. 最后再决定是否需要改成单层调度

优先关注的讨论主题：

1. 是否需要把 `GroupCtrl` 和 `RequestScheduler` 合并为单层 manager 调度
2. credit 回收是否应该从“软件 DONE”改成“DMA 完成即回收”
3. DMA 是否应该使用独立 VN，进一步隔离普通 GM 流量
4. A/B tile 在 HBM 节点上的布局是否导致热点集中
5. 当前 `maxInflightPerNode / nodeCredits / dma_read_max_inflight` 三层窗口是否过度保守

## 12. 一句话总结

当前架构的本质是：

`按列分组的多核阵列 + manager 主导的控制面 + worker 直达回包的数据面 + 基于 identity window 的 DMA 主存访问模型`。

后续所有性能问题，基本都可以落回到下面四个关键词：

- grant
- issue
- hotspot
- overlap
