# Golem Micro-Tiling / Scoreboard Refactor Plan

## 背景

当前系统虽然已经支持部分 `block_k` 整数倍扩展，但主瓶颈仍然不是 compute，而是数据供给与执行粒度不匹配导致的等待时间：

- `prefetch_wait` 长期占主导
- 其本质不是纯 DMA latency，而是 `ready -> compute_start` 的等待
- 当前 `WCP / scheduler / array` 三层使用的执行粒度并不一致：
  - scheduler 以逻辑 tile 组织 DMA
  - WCP 以逻辑 tile 驱动执行
  - array 真正高效消费的是 hardware micro-tile

结果是：

- tile 很早 ready，但不能及时转化为 compute
- buffer 占用和 compute 单位不对齐
- density-up 时收益存在，但远低于理想线性扩展
- 当前次级瓶颈已经明显转向 NoC / memory pressure

## 重构目标

目标不是继续修补状态机，而是重构执行与数据流机制，使系统从：

- `data-movement / ready-queue bound`

向：

- `dependency-aware, micro-op driven, compute-dominant`

演进。

明确目标：

1. 统一全链路执行粒度为 hardware micro-op
2. 用 scoreboard 定义 readiness，而不是整 tile ready
3. 让 partial sum 常驻本地 buffer，而不是每步 spill/reload
4. 让 request scheduler 以 compute-ready queue 非空为目标，而不是被动处理请求
5. 保证 correctness：同一 partial sum 的 `k_step` 仍严格有序

## 现状问题定义

### 当前真正瓶颈

1. `prefetch_wait` 是最大项
2. `tile_ready_wait` 说明大量时间花在 ready tile 排队
3. density 扩展后，NoC stall 和 memory backend latency 上升

### 当前结构性问题

1. 逻辑 block 和 hardware micro-tile 混用
2. ready 的定义过粗：整 tile ready 才能执行
3. partial sum 生命周期不显式
4. 本地 GM 更像地址空间，不像真正 buffer hierarchy
5. request scheduler 仍然偏 request-driven，不是 demand-driven

## 目标架构

### 1. 统一执行单位：MicroOp

建议新增统一执行单元：

- `task_id`
- `m_step`
- `k_step`
- `n_group`

语义：

- `m_step = block_m / hw_output_size`
- `k_step = block_k / hw_input_size`
- `n_group` 为 `block_n` 的子分组，允许将 N 向执行粒度与 `num_arrays` 解耦

逻辑 block 不再是执行单位，只是 micro-op 的容器。

### 2. 统一 readiness：Scoreboard

每个 task 维护 scoreboard：

- `mat_ready[m_step][k_step]`
- `vec_ready[n_group][k_step]`
- `partial_state[m_step][n_group]`

其中 `partial_state` 至少包含：

- `resident`
- `dirty`
- `last_completed_k_step`
- `buffer_slot`

某个 micro-op 可执行的条件是：

1. 该 micro-op 的 mat/vec 输入 ready
2. 若 `k_step > 0`，其依赖的 partial sum 已 resident 或可 reload
3. 满足同一 `(m_step, n_group)` 的 `k_step` 有序约束

### 3. 显式本地 buffer hierarchy

建议拆出三类 buffer：

1. `A-buffer`
   - 存储 matrix micro-tile
2. `B-buffer`
   - 存储 vector / packed-B micro-tile
3. `Partial-C buffer`
   - 存储 `(m_step, n_group)` 对应的部分和

原则：

- partial sum 优先常驻
- 仅在 eviction 或 task 完成时写回
- 不再默认每个 micro-step 都访问 GM

### 4. 调度策略：依赖约束下的 earliest-ready

禁止继续使用“全局严格顺序消费逻辑 tile”。

替换为：

- 对同一 `(m_step, n_group)`，`k_step` 严格递增
- 不同 `(m_step, n_group)` 之间可乱序
- 从 compute-ready queue 里选择最早 ready / 最适合本地性 / 最低网络压力的 micro-op

这不是之前失败的 `ready-first consume`：

- 之前失败是 tile 级乱序，破坏 accumulate 语义
- 新方案是 dependency-aware micro-op 乱序，不破坏 correctness

### 5. Request scheduler：从 request-driven 到 demand-driven

当前 scheduler 更像：

- worker 缺什么就提交什么
- manager 有 credit 就发

目标改成：

- 维护未来若干步 demand window
- 优先填满即将进入 compute-ready queue 的输入
- 以“compute-ready queue 不空”为目标进行预取

即：

- scheduler 不只是调请求
- scheduler 调的是数据流

## 分阶段落地计划

### Phase 1: 数据结构重构（低风险）

目标：先统一表示，不改大策略。

需要做：

1. 在 WCP 中引入 `MicroOp` 结构
2. 引入 scoreboard 数据结构
3. 将逻辑 task 展开为 micro-op 空间
4. 明确 partial sum 状态表达

建议修改文件：

- `src/sst/elements/golem/workercmdproc/workercmdproc.h`
- 如有必要：`src/sst/elements/golem/requestscheduler/requestscheduler.h`

验证目标：

- 不要求性能变化
- 要求状态表达完整、可打印、可调试

### Phase 2: WCP 执行重构（中风险，高收益）

目标：把 WCP 从 tile FSM 改为 micro-op scheduler。

需要做：

1. 不再按逻辑 tile 直接 `selectReadyTile()` -> execute
2. 使用 scoreboard 生成 compute-ready micro-op
3. 对同一 partial sum 保持 `k_step` 有序
4. 建立 `compute-ready queue`

验证目标：

- baseline PASS
- `1x2 / 2x1 / 2x2` PASS
- `tile_ready_wait` 开始下降

### Phase 3: Partial-C residency（中风险）

目标：去掉高频 spill/reload。

需要做：

1. partial sum buffer 显式化
2. 仅在 eviction / task 完成时写回
3. array output 作为最内层 partial buffer 使用时，需要由外层管理 residency 和 dirty

验证目标：

- `writeback_wait` 显著下降
- `prefetch_wait` 进一步下降

### Phase 4: Scheduler 重构（高收益）

目标：从 request-driven 变成 readiness-driven。

需要做：

1. scheduler 维护 demand window
2. 调度优先级考虑：
   - earliest-ready
   - buffer locality
   - NoC pressure
   - memory node pressure
3. 以 compute-ready queue occupancy 为目标控制预取

验证目标：

- density-up 更接近线性扩展
- NoC stall 增长速度变缓

## 最小可执行版本建议

如果下个 session 只做一件最值得做的事，建议做：

### "Scoreboard + k-step micro-op queue"

最小版本范围：

1. 先不做 `m_step` 乱序
2. 先保留 `block_m == hw_output_size`
3. 只把逻辑 tile 内的 `k_step` 执行单位显式化
4. partial sum 暂时仍可留在 array output
5. 用 scoreboard 保证：
   - `mat_ready[k_step]`
   - `vec_ready[k_step]`
   - `last_completed_k_step`

这一步就能回答一个关键问题：

- 当前主要损失到底是不是来自 coarse-grain tile scheduling

## 正确性约束（必须遵守）

1. 同一 `(m_step, n_group)` 的 `k_step` 必须严格递增
2. 不允许重新引入 tile 级 ready-first 乱序
3. baseline 和整数倍 block case 都必须 `VERIFY-C = PASS`
4. 不要把 debug 状态机改成影响真实语义的路径

## 推荐评估指标

每阶段都至少看：

1. `Simulation is complete`
2. `VERIFY-C = PASS`
3. `total_cycles`
4. `avg_throughput_ops_per_cycle`
5. `compute_active_time`
6. `prefetch_wait_time`
7. `writeback_wait_time`
8. `tile_ready_wait`
9. `noc_total_xbar_stalls`
10. `memory_backend_read_latency_avg_cycles`

## 与当前代码的边界关系

### 当前已有可复用基础

1. `block_k` 整数倍支持已经打通
2. `k=128` 已验证能跑通
3. density-up sweep / profiling / 可视化链路已经可复用

### 当前不要回退的部分

1. `block_k=128` 默认脚本支持
2. `WRITEBACK` 从 array output 直接取结果的修复
3. `RoCC` 对 `block_k` 的整数倍校验放宽

## 下个 Session 建议起手动作

1. 先读：
   - `tests/cmd.md`
   - 本文档
2. 先确认当前最新 baseline 成功 run
3. 只做一小步：
   - 在 WCP 内引入 micro-op/scoreboard 表达
4. 不要一开始就碰 globalmemory
5. 不要并行跑 `run_noc_dma_pipeline.sh`
