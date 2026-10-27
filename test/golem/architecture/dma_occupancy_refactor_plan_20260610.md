# DMA 机制宏观重构计划：从"带宽配给"到"占用率/延迟隐藏"

> 日期：2026-06-10
> 范围：方向 C —— 与"充裕带宽 + 固定阵列(64×64) + ~66cyc 计算"匹配的供给机制重构
> 目标读者：golem 数据通路（WCP / RequestScheduler / GlobalMemory / GroupCtrl）维护者
> 配套分析：见 `architecture/多核架构与DMA机制分析.md` 与会话审查记录

---

## 0. 一页纸结论

当前阵列利用率 ~58–74%，损失**不是带宽**问题。用 roofline 验证：

| reuse 倍率 r | 全局 HBM 流量 | 需求带宽 | vs 可用 4×1200=4.8 TB/s |
|---|---|---|---|
| r=1（无 reuse） | 128 MB | 7.5 TB/s | **超 → 带宽墙，利用率天花板 ~63%** |
| **r=4（现状）** | 32 MB | **1.9 TB/s** | **40% 占用，2.5× 余量** |

> 单 MVM micro-step = `ceil(block_k/mac_per_cu)+pipeline_depth = 64+2 = 66 cycle`（`cpu_builder.py:292`）。
> 一个 16KB tile 纯带宽只需 ~13 cycle 就能搬完，而阵列要算 66 cycle。带宽不缺，**缺的是把往返延迟 R 提前藏住的预取深度**。

**核心病灶**：现有架构叠了三套为"带宽稀缺"设计的控制平面（GroupCtrl 的 REQUEST/GRANT/DONE、RequestScheduler 的全局节点信用 + 每 tick 发射预算 + worker 信用、GlobalMemory 的 inflight/retry/flag 轮询）。这套机器**既拉高 R、又压缩可用预取深度**，亲手饿死阵列。

**重构方向**：保留 2D reuse 的**数据局部性**（它正是"带宽充裕"前提的物理来源），删掉它身上挂的**信用/授权/全局仲裁**，换成一个**深度有界、事件驱动、每核自发射的预取环**。

---

## 1. 目标与不变量

### 1.1 必须保留（KEEP）
- **2D reuse 数据局部性**：reuse 沿 (M,N) 摊销读取，使算术强度从 16→64 FLOP/byte。这是 roofline 不撞墙的唯一原因。保留其**地址映射与循环顺序**：
  - `deriveTask()`（`workercmdproc.h:1842-1924`）：macro task → (m_tile,n_tile) → HBM 源地址、C 目标地址。
  - `groupMatSlotFor/groupVecSlotFor()`（`:633-651`）：reuse 索引 → 本地 GM slot。
  - `submit2DWindowTransactions()` 里的地址计算（`:1039-1068`，`matGroupBase/vecGroupBase/localMatBufferBase`）。
- **数值路径**：micro-K-step 累加、partial-C 跨 K 窗口归约的结果必须 bit-identical（`verify-c` 必须仍通过）。
- **NoC 拓扑 / HBM 配置 / 阵列模型**：不动（`ncores_selfcom_dma_ctrl.py`、`dram/HBM_4Gb_x128.ini`、`array/*.h`）。

### 1.2 必须删除/改造（DROP / REPLACE）
- 全局跨组节点信用池（`requestscheduler.cc:55-66` `GlobalNodeCreditState`）。
- worker 信用与每 tick 发射预算（`computeWorkerCreditCap`、`manager_issue_budget_per_tick`，`requestscheduler.h:174-176`）。
- GroupCtrl 在 GEMM 读路径上的 REQUEST/GRANT/DONE 仲裁。
- 完成靠"写 GM flag + 每 cycle 全量轮询"（`workercmdproc.h:753-800` + `requestscheduler.cc` 的 `ctrlIsReadRequestPending`）。
- GM 异步 DMA 路径的反重叠点：`dma_burst_bytes=64` 碎片（`globalmemory.h:359`）、direct-WCP `inflight>=1` 钳制（`globalmemory.cc:687-714`）、超时假完成（`globalmemory.cc:759-785`）。

### 1.3 设计不变式（INVARIANT）
> **预取环深度 D ≥ ⌈R / C_window⌉ + 1**
> 其中 R = 单 tile 端到端往返延迟（cycle），C_window = 一个常驻窗口的计算时间（cycle）。
> 只要任意时刻有 ≥1 个"已就绪未计算"的 tile，阵列就不空转。重构的全部目的就是廉价地维持这个不变式。

---

## 2. 目标数据通路

```
                ┌─────────────────────────── 每个 worker 核内部 ───────────────────────────┐
   HBM 节点  ◄──┤  GlobalMemory(异步DMA, 大块, inflight=D, 完成回调)                        │
      ▲         │        ▲ issue(src,dst,len)            │ on_complete(token)               │
      │ NoC     │        │                               ▼                                  │
      └─────────┤   WCP 预取环  ── ready_queue ──►  阵列(66cyc/micro-step, block_n 并行)     │
                │   · 地址来自 deriveTask/groupSlotFor（保留 2D reuse）                       │
                │   · handleArrayDone 链式发下一个 ready micro-op                             │
                └──────────────────────────────────────────────────────────────────────────┘
```

与现状的本质差别：
- **去掉 group-manager 漏斗**：每个 worker 直接用自己的 GM 发读（写回路径 `dma_write_to_host_async` 早已这么做），不再经 `req_in_0..3` 汇聚到 1 个 manager 的发射预算。
- **完成事件化**：GM 收齐一个 tile 的字节后回调 WCP，取消每 cycle O(tiles×k) 轮询。
- **缓冲与 reuse 解耦**：预取深度由 RTT 决定，不再被 `slot/(buffers×reuse)` 公式压塌。

---

## 3. 组件级改造清单

### 3.1 GlobalMemory（`globalmemory.cc/.h`）—— 让异步 DMA 成为真正的预取引擎

| # | 改动 | 位置 | 说明 |
|---|---|---|---|
| G1 | **完成回调接口** | `.h:199-215` 附近 | 新增 `dma_read_from_host_to_globalmem_async(src,len,dst, DmaCallback cb)` 重载，或 `registerCompletionCallback(token, cb)`。在 `handle_receives` 收齐 `received_len>=total_len`（`globalmemory.cc:~1433`）时触发 `cb(token)`，不再只写 flag。WCP 用它替代轮询。 |
| G2 | **大块传输** | `.h:359` `dma_burst_bytes` | 默认从 64 提到 ≥ 一个 panel（16384）。异步读路径按 panel/16KB 发，不再切 64B 碎包。 |
| G3 | **inflight = 环深度** | `globalmemory.cc:687-714` | 删除 `direct_wcp_in_flight >= 1` 硬钳；`dma_read_max_inflight` 改为由 WCP 环深度 D 驱动（≥ D×2，覆盖 A+B）。 |
| G4 | **超时不假完成** | `globalmemory.cc:759-785` | retry 耗尽不再写 flag+cb(false) 掩盖丢包；改为显式错误/统计计数，让上层看到真实 stall（带宽充裕下本就不该超时）。 |
| G5 | **取消常驻自轮询** | `globalmemory.cc:801-803` | 完成事件化后，retry self-event 只在确有在途且接近超时才排，去掉每 cycle 重排。 |

保留：`dma_read_from_host_to_globalmem_async / dma_completion_done / dma_completion_retire`（`.h:199-202`）作为环的底层原语（写回路径已在用，成熟）。

### 3.2 RequestScheduler（`requestscheduler.cc/.h`）—— 退化或删除

两条路线（见 §5 分阶段，先 R2 后 R1）：

**R2（过渡，保留 API 形状）**：保留 `RequestSchedulerAPI`（`requestscheduler.h:136-148`）的 `submitWindowTransaction/isTileReady/...` 签名，但**掏空内部**：
- 删 `GlobalNodeCreditState` 全局池与 `acquireGlobalNodeCredits`（`requestscheduler.cc:55-122`、`:682-688`）。
- 删 `workerCredits_`/`computeWorkerCreditCap`（`.cc:377-385`）、`manager_issue_budget_per_tick` 上限（`.cc:1585-1591`）—— 收到 submit 立即 `issueTransfer`，不再按 2/tick 限速。
- `isTileReady` 由"轮询 GM pending"改为读环维护的就绪位（O(1)）。

**R1（终态，删除漏斗）**：彻底移除 RequestScheduler 子组件与 `req_in_0..3` 控制链（`ncores_selfcom_dma_ctrl.py:438-452`、`cpu_builder.py:720-741`）。地址簿记搬进 WCP（地址逻辑本就在 WCP，见 §1.1）。GM 直接发读。

### 3.3 GroupCtrl / Manager —— 从"每-tile 数据漏斗"降级为"冷路径协调器"（不是删除）

> 关键澄清：要删的是 manager 作为 **per-tile 数据漏斗 + 信用仲裁**的角色，**不是 manager 本身**。manager 作为粗粒度协调器应当保留。原则是**按时间尺度分离控制平面**（见 §3.6）。

- GEMM 数据读不再经 manager / REQUEST/GRANT/DONE（移出热路径）。
- `ncores_selfcom_dma_ctrl.py:419-436` 的 **per-tile** ctrl_req/rsp 链删除；但 manager↔worker 之间保留一条**低频**协调链用于 §3.6 的功能。
- `maxInflightPerNode_/maxGrantsPerSchedule_` 等 tile 粒度节流参数旁路。

### 3.6 Manager 的保留角色（按时间尺度分离）

| 时间尺度 | 归属 | 说明 |
|---|---|---|
| **per-tile（热, ~66cyc）** | 每 worker 自理 | 喂阵列关键路径；任何跨核协调都变成 `tile_ready_wait`。**红线：发射/信用/GRANT 绝不回 manager。** |
| **per-task / per-phase（冷, 10³~10⁴ cyc）** | manager 协调 | 不在 tile 供给路径上，开销可忽略。 |

manager 可保留/承接的功能（均为冷路径）：
1. **任务分发 / 工作窃取**：替代 `deriveTask:1843` 的静态 `worker_slot` 切分；空闲 worker 在 `Phase::DONE`(`workercmdproc.h:360`) 向 manager 领下一个 macro-task。削尾延迟，对多层异形负载尤其有用。
2. **阶段屏障 / 组间同步**：多阶段负载（LeNet `conv→fc` milestone）需"全组跑完一层才进下一层"。现状靠 `finished_mailbox_addr`(`:386`)+脚本扫 milestone，散且脆；改由 manager 收 DONE、达成 barrier 后广播放行。**每 phase 一次。**
3. **粗粒度 HBM 节点再均衡（可选）**：地址映射(`deriveTask:1881`)已静态铺开 worker→node；manager 仅在运行时热点出现时做粗调，均匀 GEMM 用不上。

**红线（绝不放回 manager）**：per-tile 读发射、per-tile 信用/配额、tile 粒度 GRANT/DONE。只要这三样不回热路径，manager 保留多少冷路径协调都不伤利用率。

> 对当前**单次 GEMM**：manager 运行时职责其实只剩漏斗（task 静态分、phase 同步走 mailbox），删之不丢功能；保留薄协调器是为多阶段/异形负载的**可扩展性**，非 GEMM 必需。

### 3.4 WorkerCommandProcessor（`workercmdproc.h`）—— 新预取环（核心）

保留的地址/数值函数（**不改语义**）：`deriveTask`、`groupMatSlotFor/groupVecSlotFor`、`submit2DWindowTransactions` 的地址段、`loadActiveMicroTileToArrays`（`:1458`）、`savePartialCFromArray/loadPartialCToArray`（`:1552/1564`）、`captureArrayOutput`。

改造点：

| # | 改动 | 现状位置 | 目标 |
|---|---|---|---|
| W1 | **缓冲与 reuse 解耦** | `residentKTileCount()` `:538-547` | 删掉 `slot/(buffers×reuse)` 压制。常驻 K 由独立参数 `window_k_tiles` 与物理 slot 上限决定；环深度 D 由 RTT 推导，不再耦合 reuse。`startWindow` 的 `residentK==0` 拒绝（`:178`）随之消失。 |
| W2 | **就绪改事件驱动** | `updateActiveTileInputReadiness()` `:753-800`（每 tick 调，`:278`） | 由 G1 的完成回调置位 scoreboard，取消每 cycle 全量遍历 `getTileKStepReadiness/isTileReady`。 |
| W3 | **放开 in-order K 门** | `isMicroOpReady()` `:732-734`（`op.kStep==lastCompletedKStep+1`） | 允许已就绪的 K-step 乱序进入阵列（同一 tile 内 K-step 之间无数据依赖，只是累加顺序——浮点可接受或按需保序）。消除"前一个 K-step 迟到→后面全堵"的气泡。 |
| W4 | **compute 链式发射** | `tick` RUN `:257`、`handleArrayDone` `:402-419` | `handleArrayDone` 在 `pendingArrays_==0` 后**直接从 ready_queue 取下一个 micro-op 发射**，不等下一 tick；消除每 tile ≥1 cycle 轮询气泡。`computeInFlight_` 单 tile 串行放宽为"ready_queue 非空即连续发"。 |
| W5 | **窗口转换渐进化** | `advance2DWindowEngine()` `:1676` 用 all-or-nothing `are2DTransactionsReady` | 改用已存在的渐进就绪逻辑（`:1300-1308`）：w+1 的就绪 tile 先算，不等整窗到齐。消除单 tile 迟到 stall 整个窗口（`windowAdvanceWaitPrefetchCount_`）。 |
| W6 | **环式预取** | `fill2DPrefetchQueue()` `:1325-1345`（深度 `prefetchWindowDepth_`） | 维持 D 个在途窗口/tile；每完成一个就补一个（滑动环）。D 由 §1.3 不变式设定，默认按 `⌈R/C_window⌉+1`，给独立配置项 `GOLEM_PREFETCH_RING_DEPTH`。 |
| W7 | **(可选) partial-C 移出阵列** | `loadPartialCToArray/savePartialCFromArray` `:1508-1517` | 终态把跨 K 窗口累加从阵列输出寄存器挪到 GM accum 缓冲，解除"同一 (M,N) 的下个 K 窗口必须等当前算完"的串行（§5 P3，收益视 K 占比，最后做）。 |

### 3.5 cpu_builder.py / 配置层

| 文件 | 改动 |
|---|---|
| `cpu_builder.py:720-741` | R2：保留 RS 子组件但去掉信用/预算参数传递；R1：删除 `request_scheduler` setSubComponent 与链路（`:918-952`）。 |
| `cpu_builder.py:756-765` | WCP 增加 `prefetch_ring_depth` 参数；`window_k_tiles` 与 `local_slot_count` 解耦。 |
| `configs/20_dma.env` | 删 `GOLEM_DMA_NODE_*CREDIT*`、`GOLEM_SCHED_ISSUE_BUDGET_PER_TICK`；加 `GOLEM_PREFETCH_RING_DEPTH`；`GOLEM_DMA_SLOT_COUNT` 提到能容纳 D×(reuseM+reuseN)。 |
| `run_noc_dma_pipeline.sh:1085-1129` | 删除 `mat_resident_k=slot/(buffers×reuse)` 那段约束推导（`:1102-1114`）与对应 ERROR 校验。 |

---

## 4. 新预取环状态机（伪码）

```text
# 每个 worker，2D reuse 地址逻辑不变，只换"供给+发射"骨架
state:
  ring        : 环形队列，容量 D，元素 = {tile_id, mat_token, vec_token, mat_ready, vec_ready}
  ready_queue : 已就绪、未计算的 micro-op
  compute_busy: 当前阵列是否在算

on tick(cycle):
  # 1) 补满预取环（地址来自 deriveTask/groupSlotFor，保留 reuse）
  while ring.size < D and has_next_tile():
      (src_m, dst_m, src_v, dst_v, len_m, len_v) = next_tile_addresses()   # 现有地址数学
      t = ring.alloc(tile_id)
      t.mat_token = gm.dma_read_async(src_m, len_m, dst_m, cb=on_complete) # G1/G3 大块多发
      t.vec_token = gm.dma_read_async(src_v, len_v, dst_v, cb=on_complete)
  # 2) 阵列空闲则发射就绪 micro-op
  if not compute_busy and not ready_queue.empty():
      issue_micro_tile(ready_queue.pop())   # loadActiveMicroTileToArrays + beginComputation
      compute_busy = true

on_complete(token):                          # G1 事件，替代轮询(W2)
  t = ring.find(token); mark t.mat_ready/vec_ready
  if t.mat_ready and t.vec_ready:
      ready_queue.push(build_micro_ops(t))   # W3：K-step 可乱序
      # skipMatRead/skipVecRead 命中 reuse 时连 DMA 都不发

on_array_done(arrayId):                       # W4 链式
  if --pending_arrays == 0:
      retire_micro_op(); free_slot_if_window_done()   # 保留 in_use/slot 归还
      if not ready_queue.empty():
          issue_micro_tile(ready_queue.pop())  # 不等下一 tick
      else:
          compute_busy = false
```

环深度估算（落地时用实测 R 校准）：当前 `array_input_size=64 → C_window≈66×k_window`。若实测单 tile RTT R≈300cyc、k_window=4（C_window≈264），则 D ≈ ⌈300/264⌉+1 = 3 个窗口；A/B 各 D 个在途 read，需 `slot_count ≥ D×(reuseM+reuseN)=3×8=24`。先按此设 `DMA_SLOT_COUNT=32` 留余量。

---

## 5. 分阶段落地与验证门

每阶段独立可验证、可回滚。验证统一看 `artifacts/stats/.../execution_summary.csv` 的 `array_utilization_pct` 与 WCP `LATENCY` 行的 `tile_ready_wait / wait_2d_active_not_ready / wait_2d_activate`，以及 RS `pressure*` 计数。

### P0 —— 纯参数（0 代码，先证伪"带宽瓶颈"）
- `GOLEM_DMA_SLOT_COUNT` ≥48（让 `residentK=48/(3×4)=4` 生效）。
- `GOLEM_SCHED_ISSUE_BUDGET_PER_TICK` 2→8~16。
- `GOLEM_WCP_PREFETCH_WINDOWS` 2→4。
- **不要**靠降 reuse 救 residentK（会撞带宽墙，见 §0）。
- **门**：`tile_ready_wait` 显著下降 → 证明瓶颈在控制平面，放心做 P1+；若不降 → 先查 R 构成（`noc_latency_summary` / `memory_queue_summary`）。

### P1 —— 事件化 + 掏空 RS 信用（R2，中churn）
- 落地 G1（完成回调）、G2（大块）、W2（去轮询）、W4（链式发射）、RS 信用/预算删除。
- 保留 RS API 形状与 group-manager 漏斗（拓扑不动），降低风险。
- **门**：`array_utilization_pct` 提升；`pressureIssuePaceBlocked_/pressureNodeCreditBlocked_` 归零；`verify-c` 通过。

### P2 —— 每核自发射 + 删漏斗（R1，高churn）
- 落地 G3/G4/G5、W1/W5/W6，删 RequestScheduler 子组件与 ctrl 链（§3.2/§3.5）。
- 地址簿记并入 WCP（地址逻辑本就在此）。
- **门**：利用率进一步提升且趋稳；NoC/HBM 真实 backpressure 接管；`verify-c` 通过。

### P3 —— K 方向并行（可选，W7）
- partial-C 移到 GM accum，解除跨 K 窗口串行。
- **门**：仅当 P2 后 `wait_2d_active_not_ready` 仍由 K 串行主导时才做；收益/复杂度权衡后决定。

---

## 6. 风险与回滚

| 风险 | 缓解 |
|---|---|
| 删信用后无人对 HBM 限流 | 带宽 2.5× 余量；DRAMSim3 + memNIC 自带队列 backpressure。G4 把真实 stall 暴露为统计，便于发现异常。 |
| 乱序 K-step（W3）改变浮点累加顺序 | fp32 下保留可配置"保序"开关；int32 无影响。`verify-c` 容差按需放宽并记录。 |
| 删 RS 漏斗影响其它实验脚本 | 用 `GOLEM_REQUEST_SCHEDULER_ENABLE` 旁路而非物理删除；R2 阶段保留 API，R1 才删。每阶段独立 git commit，可回滚。 |
| 事件回调与 SST 串行语义 | 回调在 `handle_receives` 的事件上下文内置位，不引入跨组件直接调用；遵循现有 self-link 模式。 |

---

## 7. 落地检查清单（PR 粒度）

- [ ] P0：sweep 配置 + run_summary 对比，确认 `tile_ready_wait` 下降（无代码）。
- [ ] P1-a：`globalmemory.{h,cc}` 完成回调 G1 + 大块 G2 + 去自轮询 G5。
- [ ] P1-b：`requestscheduler.cc` 删全局信用/worker信用/发射预算；`isTileReady` 改读就绪位。
- [ ] P1-c：`workercmdproc.h` W2 去轮询 + W4 链式发射。
- [ ] P1 验证：`verify-c` 通过 + 利用率提升 + `pressure*` 归零。
- [ ] P2-a：`workercmdproc.h` W1/W5/W6 环式预取 + 缓冲解耦 + 渐进窗口。
- [ ] P2-b：`globalmemory.cc` G3/G4 去 inflight=1/去假完成。
- [ ] P2-c：`cpu_builder.py` + `ncores_selfcom_dma_ctrl.py` 删 RS 子组件与 ctrl 链；配置层清理。
- [ ] P2 验证：`verify-c` 通过 + 利用率趋稳。
- [ ] P3（可选）：partial-C 移 GM accum + K 并行验证。

---

## 附：关键代码锚点速查

- 计算模型 66cyc：`tests/architecture/cpu_builder.py:292-294`、`array/computeArray.h:77`、`array/mvmComputeArray.h:156`
- WCP 环/状态机：`workercmdproc.h` tick`:244` / handleArrayDone`:402` / residentK`:538` / inorderK`:732` / 预取`:1325` / 窗口推进`:1621-1718` / partial-C`:1552/1564`
- WCP 地址数学（保留）：`deriveTask:1842` / groupSlotFor`:633-651` / submit2DWindow`:1029-1071`
- RS API 与信用：`requestscheduler.h:136-148`；`requestscheduler.cc` 全局信用`:55-122` / 发射预算`:1585` / issueTransfer
- GM 异步 DMA 与完成：`globalmemory.h:199-215`；`globalmemory.cc` 完成路径`:~1419-1444` / inflight 钳`:687-714` / 假完成`:759-785` / 自轮询`:801`
- 拓扑与 wiring：`cpu_builder.py:720-765`、`architecture/ncores_selfcom_dma_ctrl.py:419-452`
</content>
</invoke>
