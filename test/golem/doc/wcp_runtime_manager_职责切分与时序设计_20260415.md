# WCP / Runtime / Manager 职责切分与时序设计（2026-04-15）

## 1. 目标

本设计文档面向以下目标：

1. 让执行尽可能集中在计算，而不是集中在控制协议。
2. 让平均吞吐率和阵列利用率成为主优化目标。
3. 从架构上摆脱 worker 本地 mailbox `submit/done` 驱动模式。
4. 将 worker 节点从“运行时协议脚本执行者”重构为“本地自主执行体”。

当前数据表明：

- `control_overhead_time` 长期大于 `data_movement_time`
- `array_compute_active_time` 占总执行时间比例极低
- `issue_write`、`sched_protocol`、`task_desc` 等控制项长期高位

这说明当前系统不是 compute-bound，而是 control-bound。

因此，本设计的中心任务不是继续优化单条 request，而是重构职责边界，使：

- Runtime 只负责任务生成与收尾
- Manager 只负责 coarse-grain 的全局调度
- WCP 负责 worker 本地的完整执行推进

## 2. 当前问题总结

### 2.1 当前 worker 本地控制过重

当前 worker 在执行一个 GEMM task 时，仍然显式参与：

- DMA submit
- DMA completion 回收
- group grant / done
- batch load / batch compute 的软件层推进
- 输出结果回写组织

即使已经做了 batch load / batch compute，worker runtime 仍然承担了太多推进职责。

### 2.2 当前 manager 职责过细

当前 manager / `RequestScheduler` 在概念上仍然接近：

- 管 request
- 管 pair request
- 管每轮 grant/done

这会导致：

- 过细的 credit 粒度
- 过高的协议消息频率
- worker 本地 mailbox 写放大

### 2.3 当前 WCP 原型仍处于“新旧路径叠加”阶段

目前 `WorkerCommandProcessor` 已经能：

- 接收 descriptor
- 发 DMA
- 做 batch load
- 做 batch compute
- 做 writeback

但目前新旧路径职责仍有叠加：

- Runtime 仍在构造和写 descriptor
- Runtime 仍在触发 `WCP_START/WAIT`
- task 生命周期收尾仍部分依赖旧控制面

因此没有形成显著性能收益。

## 3. 目标架构总图

目标架构关系如下：

```text
Application Runtime
  -> TaskQueue (coarse descriptor enqueue only)

Manager
  -> GroupCtrl (coarse admission / drain only)
  -> RequestScheduler (coarse task/window dispatch only)

Worker Node
  -> WorkerCommandProcessor (local execution state machine)
       -> GlobalMemory / DMA engine
       -> RoCC / batch load / batch compute
       -> ComputeArray
       -> coarse completion publish
```

核心变化是：

- worker 不再 per-request submit/done
- manager 不再 per-request/pair 直接参与执行推进
- WCP 成为 worker 节点内唯一的执行推进者

## 4. 组件职责切分

### 4.1 Runtime 职责

Runtime 只保留以下职责：

1. 根据 `M/N/K` 和 block 配置生成 coarse task 列表。
2. 初始化 worker 上的 task queue / task list 元数据。
3. 启动 worker 执行（一次启动或极少数粗粒度启动）。
4. 等待 final task completion / final group done。
5. 负责最终结果校验、日志、统计归档。

Runtime 不再负责：

- 每个 `k_tile` 的 DMA submit
- 每个 pair request 的 completion publish
- 每个 batch load / compute 的显式推进

### 4.2 Manager 职责

Manager 只保留以下职责：

1. coarse-grain 任务分发
2. 组级准入控制
3. memory node 资源与公平性控制
4. 组级 / task 级完成回收

Manager 不再负责：

- 单个 worker 内部的 `k_tile` 推进
- 单个 DMA 的生命周期管理
- 单个 batch compute 的发射时机

Manager 的调度对象应从：

- request / pair request

提升为：

- task descriptor
或
- execution window descriptor

### 4.3 WCP 职责

WCP 是本设计的核心。

WCP 负责：

1. 读取 worker 的 coarse task / window descriptor
2. 管理本地执行状态机
3. 发 DMA 预取
4. 触发 batch load
5. 触发 batch compute / wait
6. 处理 writeback
7. 聚合 array / DMA completion
8. 发布 coarse task completion

WCP 不负责：

- 全局公平性
- memory node 全局仲裁
- runtime 配置生成

### 4.4 `GlobalMemory` / DMA engine 职责

继续保留现有 DMA engine，不推翻。

职责：

- 真实执行 HBM <-> GM 搬运
- 维护 inflight、重试、完成

变化仅在于：

- 上游由 mailbox 驱动改成 WCP 驱动

### 4.5 `RoCCAnalog` / `ComputeArray` 职责

继续保留。

职责：

- batch load / batch compute / array 执行
- 返回 array done 事件

变化：

- 它们的上游执行控制从 runtime 显式调用，改成由 WCP 调度。

## 5. 新的任务与执行模型

### 5.1 Task 粒度

推荐将 coarse task 保持为当前的 `(m_tile, n_tile)`。

也就是说，一个 task 对应：

- 一个 `64 x 16` 输出 tile
- 沿 `k_tile=0..7` 的完整累加

这是当前最合理的 coarse 粒度。

### 5.2 Window 粒度

在目标架构里，WCP 内部可继续维护 window 概念，但它应成为内部实现细节，而不是外部高频协议对象。

对外：

- manager 发 task

对内：

- WCP 可以将 task 分成：
  - `window-2`
  - `window-4`
  - whole-task

但外部协议不再围绕这些 window 高频交互。

## 6. 新的时序设计

### 6.1 旧时序（问题根源）

旧时序是：

1. runtime 构造 task
2. worker 每个 `k_tile` 发 DMA submit
3. worker 等 completion
4. worker 发 batch load
5. worker 发 batch compute
6. worker 发布 done
7. manager 再准入下一轮

这导致：

- 每轮都有协议
- worker 本地 mailbox 写放大
- manager 过细参与执行推进

### 6.2 新时序（目标）

新时序应为：

1. runtime 一次性提交该 worker 的 task list 元数据
2. manager 一次 grant 一个 coarse task bundle 或 task queue ownership
3. WCP 本地循环：
   - 取 task
   - 对 task 的全部 `k_tile` 做 DMA / load / compute / writeback
   - 任务完成
4. WCP 只在 task 粒度上发布完成
5. manager 只在 task 粒度上回收资源

其中 WCP 内部状态机如下：

```text
IDLE
  -> FETCH_TASK
  -> ISSUE_DMA
  -> WAIT_DMA
  -> LOAD_ARRAYS
  -> ISSUE_COMPUTE
  -> WAIT_COMPUTE
  -> WRITEBACK
  -> NEXT_K_OR_DONE
  -> TASK_COMPLETE
  -> FETCH_NEXT_TASK / IDLE
```

### 6.3 关键约束

1. 一个 task 内部不再依赖 worker 本地 submit/done mailbox。
2. task 的每个 `k_tile` 在 WCP 内部推进，不再由 runtime 显式发起。
3. task 的最终完成必须是 coarse-grain completion，而不是 per-request completion。

## 7. Descriptor 设计建议

### 7.1 Runtime 侧不再逐 task 写完整 descriptor

Runtime 不应再对每个 task 写一大块 descriptor 到 GM。

原因：

- 会放大 `task_desc`
- 仍然会放大 `issue_write`

### 7.2 推荐：轻量 task list header

Runtime 只写一个轻量 header，例如：

```text
WorkerTaskListHeader {
  worker_slot
  first_task_id
  task_stride
  task_count
  M/N/K
  block_m/block_n/block_k
  array_input_size
  array_output_size
  elem_bytes
  local_mat_addr
  local_vec_addr
  local_accum_addr
  c_store_enable
}
```

然后 WCP 内部根据：

- `task_id`
- `cfg`
- `worker_slot`

直接按现有 `pipeline_config.h` 的公式推导：

- `a_base_mm`
- `b_pack_base_mm`
- `c_base_mm`
- `task owner`
- `data node`

这一步是下一阶段最关键的优化点。

## 8. 为什么当前 multi-task WCP 没有收益

当前 multi-task WCP 已经能工作，但没有形成收益，根因是：

1. Runtime 仍然逐 task 构造 descriptor
2. Runtime 仍然把 descriptor 写入 GM
3. task lifecycle 还没有完全切到 WCP / manager 新语义
4. 因此 `task_desc` 和部分本地写成本仍然高

这说明：

- WCP 方向正确
- 但必须继续把“描述符生成和完成语义”从 runtime 里抽走，才能真正释放收益

## 9. 迁移计划

### Phase A：已完成

1. 新增 `WorkerCommandProcessor`
2. 非 overlap 路径接入 WCP
3. WCP 接管 DMA / load / compute / writeback 原型
4. 默认回归通过

### Phase B：下一步必须做

1. Runtime 改为只写 `WorkerTaskListHeader`
2. WCP 内部根据 task index 推导地址
3. 去掉逐 task 写完整 descriptor

目标：

- 直接打掉 `task_desc`
- 进一步打掉 `issue_write`

### Phase C：进一步 coarse completion

1. task 完成由 WCP 发布 coarse completion
2. manager 按 task 粒度回收 credit / 统计 done
3. runtime 不再直接参与 task 粒度完成协议

目标：

- 降低 `sched_protocol`

### Phase D：manager 侧收缩

1. `RequestScheduler` 只保留 coarse task dispatch
2. `GroupCtrl` 只保留组级准入和 drain

目标：

- manager 退出细粒度执行推进

## 10. 预期收益

在职责彻底切分后的目标状态下，预期变化应为：

1. `task_desc` 显著下降
2. `issue_write` 显著下降
3. `sched_protocol` 显著下降
4. `control_overhead_time` 大幅下降
5. `avg_throughput_ops_per_cycle` 上升
6. `array_utilization_pct` 上升

其中最关键的是：

- 控制协议不再主导执行
- 计算与数据搬运成为主要活动

## 11. 设计结论

为了让计算成为主导，必须执行以下架构原则：

1. Runtime 只保留任务生成和最终收尾
2. Manager 只保留 coarse-grain 调度和回收
3. WCP 成为 worker 节点内唯一的执行推进器
4. DMA engine 保留，但只由 WCP 驱动
5. RoCC / Array 保留，但执行控制由 WCP 调度

这不是继续 patch mailbox，而是明确切换到：

- descriptor 驱动
- local autonomous execution
- coarse completion

这是当前架构想要冲击高平均吞吐和高阵列利用率的唯一正确方向。
