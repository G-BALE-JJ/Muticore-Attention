# Compute 占比提升方案归档（2026-04-10）

## 1. 背景与目标

当前默认 GEMM 路径已经具备：

- manager 统一做 DMA issue
- worker 只提交请求并等待 completion
- array 间 `mvm` compute 已支持 async 并发
- 默认脚本 `run_noc_dma_pipeline.sh` 在 `512x64x512` 下可稳定通过 `verify_c`

但从执行统计看，系统仍然不是 compute-bound，而是明显被控制面和细粒度运行时开销支配。

当前稳定默认回归（`512x64x512`, `block=64x16x64`）的代表性统计为：

- `total ≈ 612151 cycles`
- `compute ≈ 96860 cycles`
- `compute_share_pct ≈ 15.82%`
- `dma_total ≈ 97828 cycles`
- `sched_protocol ≈ 41434 cycles`
- `nloop ≈ 265358 cycles`

这说明：

1. 纯 array 计算并不是主导项。
2. 控制路径、循环体 bookkeeping、DMA issue/completion 协议占了大量时间。
3. 只靠小参数调优，很难把 compute 占比继续明显拉高。

本归档的目标是把下一阶段的优化方向系统化，形成可执行的设计路线，优先面向以下三条主线：

1. tile 内批量 RoCC / 微码化
2. scheduler / group 的更粗粒度协议
3. 矩阵广播 / 共享装载

## 2. 当前瓶颈的真实位置

### 2.1 当前热点不是 NoC 和 HBM 带宽

已有统计显示：

- NoC 平均端口利用率不高
- memory queue delay 很低
- HBM 没有出现持续性重压满载

因此当前主瓶颈并非“网络不够快”或“主存不够宽”，而是：

- 每个 `k_tile` 需要重复执行大量控制步骤
- 每个 `n_col` / array 要发多条 RoCC 指令
- 每对 A/B 数据都要走一次 submit/wait/complete 协议
- loop 内存在大量非计算性 bookkeeping

### 2.2 当前运行时的基本单位太细

当前默认 kernel 的核心工作单元是：

- `block_m = 64`
- `block_n = 16`
- `block_k = 64`

对每个 `k_tile`，每个 worker 大致要做：

1. task descriptor 解析与地址计算
2. A tile DMA submit
3. B tile DMA submit
4. A/B completion wait
5. 对 16 个 array 做：
   - `gm2imat`
   - `gm2ivec`
   - `mvm`
   - `wait/retire`
6. `ovec2gm`
7. scheduler done / group 协议推进

也就是说，真正一次有效计算外面包着很多层控制壳。

### 2.3 `nloop` 本质上是“控制/循环体剩余时间桶”

当前 `nloop` 不是阵列时间，而是 `n_col` 循环总时间减去显式记入 `compute_cycles` 的部分。

所以 `nloop` 大，代表：

- load wrapper
- array 指令发射
- loop 控制
- 函数调用/寄存器搬运
- 未归类等待

这些开销都在主循环里，而且当前量级非常大。

### 2.4 当前改造平台期的原因

已经验证过的一些优化方向：

- async compute 修复：必要，但主要解决 correctness 和重复计算问题
- `CTRL_OVERLAP_AB=1`：在当前默认规模上反而拉高 total
- 更大 DMA burst：在当前配置上也不增益
- 更高 node credits / prefetch depth：增益有限

这说明当前系统已经不适合继续做“小修小补式调参”，而要改“工作粒度”和“控制抽象”。

## 3. 设计原则

后续改造应遵守以下原则：

1. 提高每次控制动作对应的有效计算量。
2. 尽量把 per-array、per-k-tile、per-completion 的协议动作批量化。
3. 保持默认脚本可回归，不接受破坏 forward progress 的协议修改。
4. 先做最能打掉 `nloop` 的改造，再考虑进一步压缩 DMA / 调度协议时间。
5. 保持 `verify_c` 为强约束，任何性能优化都必须通过默认回归。

## 4. 方案一：Tile 内批量 RoCC / 微码化

### 4.1 核心问题

当前一个 `k_tile` 内，对 16 个 array 的操作仍然是“显式逐条发射”：

- 16 次 `gm2imat`
- 16 次 `gm2ivec`
- 16 次 `mvm async`
- 16 次 `mvm wait`

这会导致：

- 指令条数多
- loop 控制多
- RoCC 命令队列频繁往返
- 统计中的 `nloop` 和 `compute` 外围时间偏高

### 4.2 改造目标

引入“tile 级批量命令”，让 worker 以更粗粒度向 RoCC 描述一个完整 tile 的 array 操作，而不是逐 array 显式下发。

目标是把：

- `16 * (load_mat + load_vec + mvm + wait)`

收缩为类似：

- `tile_load_mat_batch`
- `tile_load_vec_batch`
- `tile_mvm_batch`
- `tile_wait_batch`

甚至进一步压缩成：

- `tile_execute_batch`

### 4.3 可选实现方式

#### 方案 A：批量 RoCC ISA

新增若干 RoCC func7：

- `tile.gm2imat_batch`
- `tile.gm2ivec_batch`
- `tile.mvm_batch`
- `tile.wait_batch`

每条命令通过 `rs1/rs2` 或 GM descriptor 描述：

- 起始 array_id
- array 个数
- base GM 地址
- stride
- 模式位（async / sync / clear / accumulate）

优点：

- 与当前 RoCC 编程模型兼容性高
- 便于逐步替换现有 helper

缺点：

- ISA 数量增加
- runtime 和 RoCC 解码都要改

#### 方案 B：RoCC 微码描述符

在 GM 中放置 `TileBatchDescriptor`，worker 只发一条 `tile.exec(desc_addr)`。

descriptor 包含：

- tile 类型
- array 范围
- 每个 array 的 mat / vec / out 地址
- 是否 clear
- 是否 accumulate
- 批量 wait 策略

优点：

- 指令最少
- 便于未来扩展 conv / attention / reduction

缺点：

- RoCC 内部控制器更复杂
- descriptor 协议需要一次性设计稳妥

### 4.4 推荐落地顺序

建议先做 **方案 A 的简化版**，不要一开始就做完整微码解释器。

推荐第一步：

1. 新增 `tile.mvm_batch(start_array, count)`
2. 新增 `tile.wait_batch(start_array, count)`
3. 保留现有 `gm2imat/gm2ivec`

这样能先证明：

- 只压缩 compute 发射和 wait，就能显著减少 `nloop`

如果有效，再做第二步：

1. `tile.gm2imat_batch`
2. `tile.gm2ivec_batch`

### 4.5 组件改动点

主要涉及：

- `rocc/roccAnalog.h`
- `array/mvmComputeArray.h`
- `tests/small/mvm_noc_int_array/ex_instr.h`
- `tests/small/mvm_noc_int_array/gemm_matmul_op*.h`

需要新增的内部状态：

- batch inflight bitmap
- batch wait completion counter
- batch command completion response

### 4.6 风险点

1. batch 中部分 array load 未 ready 时的等待策略
2. async compute 已有状态机与 batch wait 的兼容性
3. response 语义要保持与 Vanadis/RoCC 接口一致
4. 不能引入新的重复计算或双 retire

### 4.7 验收指标

1. 默认脚本 `verify_c` 通过
2. `nloop` 显著下降
3. `compute_share_pct` 提升
4. `roccs_issued` 数量下降
5. `cycles_mvm` 总量不变或更合理，不出现重复计算

## 5. 方案二：Scheduler / Group 更粗粒度协议

### 5.1 核心问题

当前 manager 协议粒度基本仍是：

- per-k-tile submit
- per DMA completion done

即便已经有 pair submit，worker 还是频繁做 mailbox 与 completion 协议动作。

这导致：

- `sched_protocol` 高
- `submit_pack` 高
- `issue_write` 高
- `group_wait` 持续存在

### 5.2 改造目标

把当前“每个 tile / 每条传输都要回收一次协议状态”的模式，改成“一个较大 batch 内只做一次或少数几次协议动作”。

### 5.3 设计方向

#### 方向 A：per-task credit window

manager 一次 grant 一个更大的 window，例如覆盖：

- 一个 task 的多个 `k_tile`
- 或一个 worker 的多个连续 tile

worker 在 window 内可连续提交，不必每次重新 grant。

#### 方向 B：per-batch completion

不是对每个 DMA ticket 都回 done，而是：

- 一个 batch 内部维护本地完成计数
- 只有 batch 完成时，worker 才向 scheduler / group 回一次完成

#### 方向 C：scheduler 内部 credit 回收与聚合

manager 侧 scheduler 内部跟踪：

- 某 worker 的 batch inflight 数
- 某 memory node 的 batch inflight 数

完成时按 batch 回收信用，而非按单条 request 回收。

### 5.4 推荐改造顺序

建议先不改 `GroupCtrl` 基本消息类型，而是：

1. 在 `RequestScheduler` 内部引入 `BatchSubmit` / `BatchDone`
2. worker 的 submit ring 一次提交多个 k_tile 的描述
3. manager 侧只在 batch 尾部写 done

这样可以把主要复杂度先收敛到 scheduler，不急着同时修改 group 协议。

### 5.5 组件改动点

- `requestscheduler/requestscheduler.h`
- `requestscheduler/requestscheduler.cc`
- `tests/small/mvm_noc_int_array/request_scheduler_runtime.h`
- `tests/small/mvm_noc_int_array/gemm_matmul_op_ctrl.h`

可能新增的数据结构：

- `BatchTransfer`
- `BatchCompletionState`
- `WorkerBatchCredit`

### 5.6 风险点

1. credit 计数不一致容易活锁
2. batch 内部分完成与最终 done 的一致性
3. worker abort / early exit 时的回收逻辑
4. manager 公平性可能被更粗粒度 batch 破坏

### 5.7 验收指标

1. 默认脚本稳定完成，无超时/活锁
2. `sched_protocol` 显著下降
3. `issue_write` 和 `submit_pack` 下降
4. `group_wait` 至少不恶化

## 6. 方案三：矩阵广播 / 共享装载

### 6.1 核心问题

当前 tile 内 16 个 array 在逻辑上共享同一 A tile，但执行上往往仍表现为：

- 每个 array 单独 `gm2imat`
- 每个 array 单独写入本地 array storage

如果 A tile 本质相同，这就是重复装载。

### 6.2 改造目标

让同一个 A tile 在 tile 内只装一次，然后广播或共享给多个 array 使用，减少：

- `gm2imat` 次数
- array 内 matrix 写入次数
- loop 控制和 load wrapper 时间

### 6.3 可选实现方式

#### 方案 A：RoCC 内部 matrix shadow buffer

RoCC / array 控制器先把 A tile 装进共享 shadow buffer，多个 array 引用同一份 matrix 数据。

优点：

- 对软件透明
- 不必改 GM 地址协议太多

缺点：

- array 模型要支持 shared matrix view

#### 方案 B：array 级广播加载

增加批量命令：

- `tile.gm2imat_broadcast(gm_addr, start_array, count)`

语义是：

- 从 GM 读一次矩阵
- fan-out 到多个 array 的 matrix bank

优点：

- 更贴近现有 array 独立存储模型

缺点：

- 数据仍然会复制到每个 array，只是外部 load 次数减少

#### 方案 C：共享 matrix / 独立 vector

这是更推荐的长期方案：

- A tile 共享
- B 向量仍按 array 独立

因为当前 `block_n=16`，不同 array 的主要差异在于 vector 列，而不是 matrix。

### 6.4 推荐落地顺序

建议先做 **广播加载**，不要一步走到“共享 matrix 生命周期管理”。

推荐第一步：

1. 新增 `tile.gm2imat_broadcast`
2. 保持每个 array 的 vector load 独立
3. 只减少 matrix load 的指令与拷贝次数

### 6.5 组件改动点

- `rocc/roccAnalog.h`
- `array/mvmComputeArray.h`
- `tests/small/mvm_noc_int_array/gemm_matmul_op*.h`

### 6.6 风险点

1. array 内 matrix state 生命周期
2. clear / overwrite 时机
3. 与 accumulate mode 的交互

### 6.7 验收指标

1. `cycles_mvm_gm2imat` 相对下降
2. `roccs_issued` 下降
3. `nloop` 下降
4. `verify_c` 不受影响

## 7. 三条方案的优先级建议

### 第一优先级：Tile 内批量 RoCC / 微码化

原因：

- 直接作用于 `nloop`
- 不需要先重写 scheduler 协议
- 最容易在默认回归中看到正收益

### 第二优先级：Scheduler / Group 更粗粒度协议

原因：

- 能继续压 `sched_protocol`、`issue_write`、`submit_pack`
- 但 forward progress 风险更高

### 第三优先级：矩阵广播 / 共享装载

原因：

- 长期收益很大
- 但与 batch RoCC 的设计耦合较深
- 最好在 batch command 形成后再做

## 8. 推荐实施路线图

### Phase 1：Batch Compute/WaIt

目标：

- 引入 `tile.mvm_batch`
- 引入 `tile.wait_batch`

不改：

- scheduler 协议
- group 协议
- matrix load 路径

验收：

- `verify_c` 通过
- `nloop` 有可观下降

### Phase 2：Batch Load

目标：

- 引入 `tile.gm2imat_batch`
- 引入 `tile.gm2ivec_batch`

验收：

- `roccs_issued` 继续下降
- `nloop` 再下降

### Phase 3：Batch Scheduler Protocol

目标：

- worker 一次提交多个 k_tile
- manager 按 batch credit 调度
- batch done 统一回收信用

验收：

- `sched_protocol` 下降
- `issue_write` / `submit_pack` 下降

### Phase 4：Matrix Broadcast

目标：

- 同一 A tile 只装一次
- 多 array 复用 matrix 数据

验收：

- `gm2imat` 相关开销下降
- `compute_share_pct` 进一步提升

## 9. 修改方案讨论结论

基于当前代码状态，推荐的具体修改顺序如下：

1. 先在 `RoCCAnalog` 增加 batch compute / batch wait 的最小闭环。
2. 在 `gemm_matmul_op*.h` 增加 batch helper，但保留旧 helper，便于 A/B 对比回归。
3. 用默认脚本跑性能回归，只接受“可完成 + verify 正确 + 指标改善”的修改。
4. 等 batch compute 路径稳定后，再触碰 scheduler 的 batch 协议。
5. matrix broadcast 放到第三阶段以后，避免同时改动 RoCC、scheduler、array 三层导致问题难以定位。

## 10. 当前建议的立即行动项

建议下一轮从以下任务开始：

1. 设计 `tile.mvm_batch(start_array,count)` 的 ISA 与 helper 封装。
2. 在 `RoCCAnalog` 内实现 batch submit / batch wait 状态机。
3. 在默认 kernel 中只替换 compute/wait 路径，不动 load 路径。
4. 用默认脚本回归 `512x64x512`，对比：
   - `nloop`
   - `compute_share_pct`
   - `roccs_issued`
   - `total`

这一步是当前最值得做、且最容易形成正反馈的主线。
