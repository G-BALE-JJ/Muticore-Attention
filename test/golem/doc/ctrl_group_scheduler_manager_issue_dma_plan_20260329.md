# CTRL 路径现状归档与改造方案（2026-03-29）

## 1. 目标

你提出的目标是：

1. worker 只提交请求，不直接发 DMA。
2. manager 统一做流量调度（按 node 预算、公平性、队列）。
3. manager 统一执行 DMA issue。
4. DMA 返回仍回到对应 worker 的 completion endpoint/flag，worker 只等待完成。

## 2. 当前实现现状（代码真实行为）

### 2.1 GroupCtrl 侧

- worker 在 CTRL mailbox 发布请求并等待 grant：
  - `ctrl_publish_request_local(...)`
  - `adaptive_wait_eq_profiled(... CTRL_LOCAL_GRANT_SEQ_OFF, req_seq)`
- 位置：`small/mvm_noc_int_array/gemm_matmul_op_ctrl.h`

### 2.2 Group manager 侧

- `group_manager_service(...)` 只负责：
  - 扫描 worker pending 请求
  - 按 `inflight_per_node` 与 `GROUP_MAX_INFLIGHT_PER_NODE` 发放 grant
  - 回收 done 并清理 inflight
- 不负责 DMA issue。
- 位置：`small/mvm_noc_int_array/operators.h`

### 2.3 Scheduler 侧

- 当前 `request_scheduler_runtime.h` 只定义本地 mailbox 协议写入：
  - submit（0x1A00+）
  - done（0x1A40+）
- `scheduler_submit_read_ticket_slot(...)` 当前行为是：
  - `prepare_dma_read_ticket_slot(...)`
  - `sched_publish_submit_local(...)` 写 mailbox
  - 返回 ticket
- 没有可见的“scheduler consumer loop”去读取 submit_valid 并执行 `dma_remote_load_issue_slot(...)`。

### 2.4 关键偏差

当前不是“manager 统一提交 DMA”，而是“worker 拿到 grant 后写 scheduler mailbox（且 scheduler 执行环未闭合）”。

## 3. 推荐职责划分（目标架构）

### 3.1 GroupCtrl（调度面）

- 只做 admission control：
  - 请求入队
  - node 预算控制
  - 公平性与优先级
  - 生命周期（finished/group_done）
- 输出可执行任务给 Request Scheduler。

### 3.2 Request Scheduler（执行面）

- 只做执行：
  - 从 manager 投递队列取任务
  - 统一 issue DMA
  - 维护 request_id -> worker completion 映射
  - 完成/超时/重试处理
  - 回写 completion 到 worker

### 3.3 Worker（业务面）

- 只发请求、等完成。
- 不直接 issue DMA。

## 4. 修改方案（分阶段，低风险）

## Phase A：先打通 manager->scheduler->DMA 执行闭环

1. 在 group manager（leader core）驻留循环中增加“可执行任务投递”。
2. 在 request scheduler 增加 consumer 执行循环：
   - 读取 submit_valid
   - 调用 `dma_remote_load_issue_slot(...)`
   - 记录 request_id 与 completion 元数据
   - 清 submit_valid
3. 在 DMA completion 回调处根据 request_id 回写 worker completion 标记。

验收标准：

- worker 不再直接 issue DMA。
- `read_issue_count` 从 manager/scheduler 侧增长。
- worker completion 正常推进，MVM_PROGRESS 能递增。

## Phase B：去除 worker 侧 DMA issue 依赖

1. 将 `scheduler_submit_read_ticket_slot(...)` 改为纯提交请求，不再调用任何 worker 侧 issue 相关流程。
2. `wait_dma_read_ticket(...)` 改为等待 scheduler 回写的 worker completion（保持 seq 语义）。
3. 保留 request_id 全链路（worker -> manager -> scheduler -> completion）。

验收标准：

- worker 侧不出现 DMA issue 调用。
- 完成路径仅由 scheduler 回写触发。

## Phase C：优化流量与尾延迟

1. 节点级调度策略：
   - per-node deficit round robin / aging
   - 长短请求分层队列
2. 重试与背压策略：
   - scheduler 内部重试与退避
   - 限制同 node 最大并发
3. 观测增强：
   - 每 node inflight、排队长度、P50/P95/P99 tail latency

验收标准：

- 热点 node 场景下 tail latency 收敛。
- 无异常抖动或 starvation。

## 5. 建议保留的协议字段

1. `request_id`：全链路追踪主键。
2. `node`：预算与调度依据。
3. `completion_flag_addr/value`：保持 worker 兼容等待模型。
4. `submit_valid/done_valid`：作为 scheduler 执行层 mailbox 信号。

## 6. 风险与注意事项

1. manager 单点执行 DMA 可能成为瓶颈：可先“一组一 manager”，后续可扩展到“组内多 scheduler worker”。
2. completion 回写必须原子/有序，避免重复完成与丢完成。
3. 需防止 request_id 回卷带来的匹配冲突（建议高位包含 core/slot/epoch）。

## 7. 结论

- 你的目标架构是正确方向：统一调度 + 统一执行更有利于流量控制和尾延迟治理。
- 当前代码处于“调度和执行边界未闭合”状态，应优先完成 manager->scheduler->DMA 闭环，再移除 worker 侧执行路径。
