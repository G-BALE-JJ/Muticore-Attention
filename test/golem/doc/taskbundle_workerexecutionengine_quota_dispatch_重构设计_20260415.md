# TaskBundle + WorkerExecutionEngine + Quota Dispatch 重构设计（2026-04-15）

## 1. 设计目标

本设计面向以下核心目标：

1. 让执行真正集中在计算，而不是集中在控制协议。
2. 明确从 mailbox/request 驱动模型切换到 descriptor/engine 驱动模型。
3. 让 manager 退出细粒度 request/pair 审批，转为 coarse quota dispatch。
4. 让 worker 节点具备本地自治执行能力，使数据搬运与计算形成流水。
5. 从根本上降低：
   - `issue_write`
   - `sched_protocol`
   - `task_desc`
   - `group_wait`

当前系统的根本问题不是单个组件慢，而是：

- task 太细
- 执行推进职责分散在 runtime / mailbox / scheduler / rocc 之间
- worker 节点不是执行引擎，而是协议脚本执行者
- manager 过细参与执行推进

因此，要让计算成为主导，必须从：

- request / response / mailbox 驱动

切换到：

- task bundle / local execution engine / quota dispatch

## 2. 当前架构问题归纳

### 2.1 Task 粒度过细

当前一个 task 对应：

- 一个 `(m_tile, n_tile)`
- 沿 `k_tile=0..7` 做完整累加

虽然已经比 request 更粗，但从执行系统视角看，仍然太细。其后果是：

- task 切换太频繁
- A tile / B tile 复用不足
- 固定控制成本被重复支付

### 2.2 Worker 本地协议负担过重

即使已经引入 WCP 原型，worker 侧仍有大量控制成本：

- header/descriptor 写入
- coarse start/wait
- 旧控制面残留

同时，WCP 内部执行仍偏串行，导致：

- `data_movement_time` 仍高
- `control_overhead_time` 仍高

### 2.3 Manager 过度参与执行节奏

当前 manager 侧虽然开始 coarse 化，但仍然在概念上接近：

- 发 grant
- 等完成
- 维护 worker inflight / node inflight

而不是纯粹的：

- 资源配额发放者
- 粗粒度任务分配者

### 2.4 数据流和控制流没有真正解耦

当前系统依然存在：

- “收到控制动作才做一步执行”

这与“让计算成为主导”目标相冲突。

## 3. 目标架构概览

目标架构分为三层：

### 3.1 Runtime（任务生成层）

职责：

- 生成 TaskBundle 列表
- 写入全局 task queue / worker task queue 元数据
- 启动 worker 执行
- 等待最终结果
- 结果校验 / 归档 / 统计

不再负责：

- 每个 task 的推进
- 每个 DMA 的提交与回收
- 每个 compute 的显式发射

### 3.2 Manager（全局资源控制层）

职责：

- coarse quota 发放
- TaskBundle dispatch
- memory node 资源 / 公平性控制
- coarse completion 回收

不再负责：

- request/pair 生命周期
- worker 内部 `k_tile` 推进
- batch load / compute 执行细节

### 3.3 WorkerExecutionEngine（本地执行层）

职责：

- 从本地或分配队列取 TaskBundle
- 自主推进 prefetch / load / compute / writeback
- 在本地做 completion 聚合
- 在 bundle 粒度回报完成

这将成为 worker 节点的真正执行中心。

## 4. 新核心概念

## 4.1 TaskBundle

TaskBundle 是新的调度和执行粒度。

它不是 request，不是 pair request，也不只是单 task，而是：

- 一组可在 worker 本地连续执行、并且具有数据复用关系的任务集合

推荐第一版定义：

- 同一 worker 下的连续 2~4 个 `(m_tile, n_tile)` task

推荐更优方向：

- 固定一个 `m_tile`
- 打包多个相邻 `n_tile`

这样可以最大化复用：

- 同一个 A tile

## 4.2 WorkerExecutionEngine（WEE）

WEE 是对当前 WCP 原型的正式升级版名称，强调其最终职责不是“命令处理器”，而是“本地执行引擎”。

WEE 负责：

1. 取 TaskBundle
2. 管理本地多级流水
3. 驱动 DMA engine
4. 驱动 batch load / batch compute
5. 聚合 bundle 完成
6. 发布 coarse completion

## 4.3 Quota Dispatch

manager 不再审批 request，而是发配额（quota）。

quota 的含义是：

- 某 worker 在一段时间内可以占用多少数据搬运与执行资源
- 某 worker 可以领走多少 TaskBundle

典型形式：

```text
QuotaGrant {
  worker_id
  bundle_budget
  dma_budget_bytes
  node_mask
  valid_until_epoch
}
```

其本质是：

- manager 只定义边界
- WEE 在边界内自治推进

## 5. TaskBundle 设计建议

### 5.1 第一版 TaskBundle 字段

```text
TaskBundleHeader {
  bundle_id
  worker_slot
  task_count
  task_begin_id
  task_stride
  M
  N
  K
  block_m
  block_n
  block_k
  elem_bytes
  array_input_size
  array_output_size
  data_memory_node_count
  mem_node_size
  off_gemm_mat_base
  off_gemm_vec_base
  off_gemm_out_base
  local_buffer_layout
}
```

说明：

- 第一版仍允许用 `task_begin_id + task_stride + task_count` 推导 task
- 避免逐 task 写完整 descriptor

### 5.2 长期方向

随着 bundle 复用关系更复杂，可以进一步支持：

- bundle 内 task map
- A tile 常驻 hint
- B tile 序列布局 hint
- C tile merge / reduce 策略

## 6. WorkerExecutionEngine 设计

## 6.1 WEE 负责的完整生命周期

对一个 TaskBundle，WEE 必须完整负责：

1. 取 bundle
2. 解析本地映射关系
3. prefetch A/B
4. local GM -> array load
5. batch compute
6. 延迟 writeback / 最终结果写回
7. bundle done publish

Runtime 和 Manager 不再参与中间推进。

## 6.2 WEE 状态机

推荐状态机：

```text
IDLE
  -> FETCH_BUNDLE
  -> PREPARE_TILESET
  -> PREFETCH_STAGE
  -> LOAD_STAGE
  -> COMPUTE_STAGE
  -> WRITEBACK_STAGE
  -> NEXT_TILESET
  -> BUNDLE_COMPLETE
  -> FETCH_BUNDLE / IDLE
```

其中：

- `PREPARE_TILESET`：计算 task / tile 映射、buffer 分配
- `PREFETCH_STAGE`：HBM -> local GM
- `LOAD_STAGE`：local GM -> array
- `COMPUTE_STAGE`：batch compute
- `WRITEBACK_STAGE`：结果 materialize / store

## 6.3 WEE 内部必须支持的关键机制

### 6.3.1 双缓冲 / 多缓冲

必须引入：

- `buffer0`
- `buffer1`

推荐长期支持：

- triple buffering

作用：

- 当前 tile compute 时，下一 tile prefetch
- 上一 tile writeback 尽量后台化

### 6.3.2 A tile 常驻

在 bundle 内，如果多个 task / 多个 `n_tile` 共用相同 A tile，则：

- A tile 在 local GM 常驻
- 只做一次 prefetch
- 多次复用

这是让计算成为主导的关键。

### 6.3.3 延迟 writeback

不要每个 `k_tile` 都 materialize。

建议：

- 中间累加结果留在 local GM / array output
- bundle 尾部再统一 writeback

## 7. Manager 设计

## 7.1 Manager 的新职责

Manager 仅保留：

1. 按 worker 发放 quota
2. 按 quota 分发 TaskBundle
3. 跟踪 bundle 完成
4. 做全局 drain / group done

### 不再保留

- request/pair scheduling
- DMA request 生命周期管理
- per-task 完成审批

## 7.2 Quota 模型

推荐用双 quota：

1. `bundle_quota`
- 控 worker 同时可拥有多少 bundle

2. `dma_quota`
- 控 worker 可占用多少数据搬运资源

这样 manager 的作用从：

- 逐请求仲裁

变成：

- 定资源边界

## 8. 彻底去 mailbox 化的目标

### 8.1 要保留的 mailbox / queue

只保留：

- coarse task queue
- coarse completion queue

### 8.2 要删除的 worker 本地协议

最终应删除：

- request submit ring
- done ring
- per-request mailbox publish
- per-k-tile completion protocol

否则 `issue_write` 无法真正消失。

## 9. 新旧职责边界

### Runtime

保留：

- task/bundle 生成
- 启动执行
- 最终收尾

删除：

- task 级推进
- request 级推进
- completion 级推进

### Manager

保留：

- coarse dispatch
- quota control
- bundle completion 回收

删除：

- request/pair 级语义

### WEE

新增承担：

- bundle 内所有执行推进
- 本地 completion 聚合
- 本地 DMA/load/compute/writeback 编排

## 10. 为什么当前 WCP/WEE 原型没有赢

当前原型虽然证明了方向可行，但收益没释放，是因为：

1. 仍然没有 TaskBundle 复用收益
2. WEE 仍偏串行执行
3. manager 还没有真正 quota 化
4. worker 本地写仍未根除

所以当前原型只是：

- “职责切换正确”

还不是：

- “执行组织高效”

## 11. 实施路线图

### Phase 1：TaskBundle Header

1. Runtime 只写 `TaskBundleHeader`
2. WEE 内部推导 bundle 内 task 地址
3. 删除逐 task descriptor

### Phase 2：真正 WEE 流水化

1. 引入双缓冲
2. prefetch / load / compute / writeback 重叠
3. A tile 常驻复用

### Phase 3：Quota Dispatch

1. manager 从 request/pair 语义切到 quota 语义
2. 发放 bundle quota + dma quota
3. bundle completion 回收

### Phase 4：删旧 mailbox

1. 删除 worker 本地 request/done ring
2. 删旧 request/pair 统计路径

## 12. 预期收益

如果这套方案真正完成，预期：

### 12.1 直接下降

- `issue_write`
- `sched_protocol`
- `task_desc`
- `group_wait`
- `nloop`

### 12.2 直接上升

- `avg_throughput_ops_per_cycle`
- `array_utilization_pct`

### 12.3 关键性变化

- `control_overhead_time` 不再主导
- `data_movement_time` 被部分隐藏
- `array_compute_active_time` 占比显著上升

## 13. 设计结论

要真正让计算成为主导，不该继续围绕旧 mailbox 协议做 patch，而应该明确切换到：

1. `TaskBundle`：新的调度与执行粒度
2. `WorkerExecutionEngine`：worker 节点内唯一执行推进者
3. `QuotaDispatch`：manager 的 coarse-grain 资源发放模型

这是从架构上摆脱 mailbox 协议低效、冲击高吞吐和高硬件利用率的正确路线。
