# 当前架构问题分析报告（Compute Dominate 视角，2026-04-15）

## 1. 报告目的

本报告的目标有两个：

1. 对当前工程的整体结构与执行链路做一次统一说明。
2. 以“让计算成为主导（compute dominate）”为优化目标，分析当前架构的主要问题与瓶颈所在。

这里所说的“compute dominate”不是指某个统计桶名字变大，而是指：

- 平均吞吐率明显提升
- 阵列利用率明显提升
- 数据搬运尽量被计算覆盖
- 控制和运行时开销退出关键路径
- 最终总时间主要由有效计算决定，而不是由调度、协议、等待决定

本报告不讨论具体 patch 细节，而是站在当前工程整体视角，说明：

- 这个工程现在是什么架构
- 当前执行是如何发生的
- 为什么当前仍然做不到 compute dominate
- 后续架构修改为什么必须朝数据流驱动方向走

## 2. 工程整体介绍

本工程位于：

- `src/sst/elements/golem`

本质上是建立在 SST 上的一个异构加速系统模拟环境，目标是模拟：

- 多核 CPU + RoCC 加速器
- 多个 array 并行计算
- local GlobalMemory / HBM / NoC / manager 控制面

其测试入口通常为：

- `tests/run_noc_dma_pipeline.sh`

当前默认实验路径以 GEMM 为主，核心工作负载是：

- `M=512`
- `N=64`
- `K=512`
- `block_m=64`
- `block_n=16`
- `block_k=64`

即：

- 输出空间沿 `m_tile` 和 `n_tile` 切分
- 沿 `k_tile` 累加完成矩阵乘

## 3. 当前架构组件

从系统视角看，当前工程中与执行主路径强相关的组件主要有：

### 3.1 Runtime / 测试程序

位置：

- `tests/small/mvm_noc_int_array/*.h/*.cpp`

职责：

- 生成任务
- 组织 matmul runtime 配置
- 发起 worker 执行
- 最终收尾与校验

### 3.2 RoCCAnalog

位置：

- `rocc/roccAnalog.h`

职责：

- 接收 RoCC 指令
- 驱动 batch load / batch compute
- 维护 array 侧完成状态与统计

### 3.3 ComputeArray / MVMComputeArray

位置：

- `array/computeArray.h`
- `array/mvmComputeArray.h`

职责：

- 模拟 array 计算执行
- 提供 compute done 事件
- 提供硬件 MVM 周期统计

### 3.4 GlobalMemory

位置：

- `globalmemory/globalmemory.h`
- `globalmemory/globalmemory.cc`

职责：

- 本地 GM 存储
- 远端 GM 访问
- HBM/host DMA
- DMA 完成与 flag 机制

### 3.5 GroupCtrl

位置：

- `groupctrl/groupctrl.h/.cc`

职责：

- 组级 admission
- manager 侧 drain / group done 判定

### 3.6 RequestScheduler

位置：

- `requestscheduler/requestscheduler.h/.cc`

职责：

- worker / node credit
- 控制哪些 worker 什么时候可以发 request
- 负责 coarse 或细粒度的数据搬运调度（当前仍有旧 request 语义残留）

### 3.7 WorkerCommandProcessor / WorkerExecutionEngine 原型

位置：

- `workercmdproc/workercmdproc.h`

职责（当前原型）：

- 接收 WCP header
- 本地推进 DMA / load / compute / writeback
- 尝试承担 worker 本地 coarse completion

注意：当前它已经是一个“方向正确”的原型，但还不是成熟的高效执行引擎。

## 4. 当前执行路径概述

从默认 GEMM 路径看，当前执行可以概括为：

1. Runtime 生成任务
2. Worker 拿到任务（或 header）
3. 触发 DMA / batch load / batch compute
4. 计算完成后 writeback
5. coarse completion / group finished
6. 最终输出与校验

历史上，这条路径更多是：

- CPU 指令 / mailbox 协议驱动

而不是：

- 本地数据流自动推进

即使当前已经引入 WCP/WEE 原型，系统仍未真正转向高效数据流驱动。

## 5. 当前统计口径（用于理解问题）

当前 summary/comparison 已经重构为：

### 5.1 主 summary

- `total_cycles`
- `avg_throughput_ops_per_cycle`
- `peak_throughput_ops_per_cycle`
- `array_utilization_pct`
- `array_compute_active_time`
- `array_load_active_time`
- `data_movement_time`
- `control_overhead_time`

### 5.2 debug summary

主要用于定位 runtime/control 路径：

- `issue_write`
- `sched_protocol`
- `task_desc`
- `group_wait`
- `nloop`
- `submit_pack`
- `dma_total`

这些统计已经足够清楚地说明当前问题，不再需要依赖旧的 `compute_share_pct_runtime` 作为主结论。

## 6. 当前系统的真实状态

从最新一轮真实接入 WCP 热路径的实验看，系统呈现出非常稳定的结论：

### 6.1 计算本体占比极低

`array_compute_active_time` 持续远小于：

- `data_movement_time`
- `control_overhead_time`

这说明阵列算得不满，不是计算能力本身不足，而是算阵列没有被持续喂饱。

### 6.2 控制开销长期主导

`control_overhead_time` 长期是最大项，且主要来自：

- `issue_write`
- `sched_protocol`
- `task_desc`
- `group_wait`

这说明当前系统本质上仍然是 control-bound，而不是 compute-bound。

### 6.3 数据搬运未被有效隐藏

`data_movement_time` 长期很高，说明：

- DMA / GM -> array load / writeback 还暴露在关键路径上
- 并没有形成“计算在前台、搬运在后台”的稳态流水

## 7. 围绕 compute dominate 目标，当前架构的核心问题

## 7.1 问题一：执行仍然是 CPU / 控制驱动，而不是数据流驱动

这是当前架构的第一性问题。

当前系统里，执行推进仍然带有强烈的以下特征：

- CPU/runtime 仍深度参与任务推进
- manager / scheduler 仍参与过多中间步骤
- worker 本地控制还需要频繁表达“下一步做什么”

compute dominate 的要求是：

- 数据 ready 触发下一阶段
- 本地 buffer ready 触发下一阶段
- array idle 且输入 ready 时自动进入计算

而不是：

- runtime 再发命令
- manager 再发许可
- mailbox 再通知下一步

当前这一点没有实现，所以系统天然不可能让计算主导。

## 7.2 问题二：worker 节点不是高效执行引擎，而是本地控制脚本执行者

虽然已经引入 WCP/WEE 原型，但从数据上看：

- WEE 已经能工作
- 但还不是高效执行器

它当前更像：

- 本地顺序控制器

而不是：

- 流水化、数据流驱动的本地执行引擎

这会导致：

- DMA / load / compute / writeback 仍偏串行
- `data_movement_time` 无法被计算覆盖
- array 依旧经常等待数据

## 7.3 问题三：当前 task 粒度太小，无法摊薄固定控制成本

当前基础 task 粒度仍然是：

- 单个 `(m_tile, n_tile)`
- 沿 `k_tile` 完整累加

这种粒度对于“正确性”足够，但对于“高利用率”不够。

原因：

- task 切换频繁
- 同一个 A tile 复用不足
- 控制与数据搬运固定成本被重复支付

即使 WCP 介入，如果 task 本身太小，仍然很难把系统推到 compute-dominant。

## 7.4 问题四：A tile 没有被真正作为长驻复用对象来设计

从矩阵乘数据流看，真正值得复用的是：

- 固定 `m_tile` 下的 A tile

它应尽量：

- 一次搬入
- 多次服务多个 `n_tile`

当前架构虽然已经在朝这个方向思考，但执行组织还没真正围绕它展开。

因此：

- A tile 的 DMA 次数没有得到根本性下降
- A tile 对控制的摊销不足

## 7.5 问题五：partial sum 驻留层次设计不对，阻碍真正的数据流组织

当前计算模式天然适合：

- 沿 `k` 方向在同一个 output buffer 上累加

但一旦想在 bundle 粒度上做 A tile 复用、跨多个 `n_tile` 推进，就会遇到一个关键问题：

- array output buffer 不能同时长期驻留多个不同 `n_tile` 的 partial sum

这意味着：

- 必须引入更大的本地 accum store
- partial sum 应从 array output buffer spill 到本地累加层
- 最终再统一 writeback

当前架构还没有成熟的这一层设计，所以它阻碍了 TaskBundle 化和真正的 A tile 复用。

## 7.6 问题六：manager 仍然没有真正退化为 coarse quota dispatcher

虽然 manager 侧已经开始 coarse 化，但还没有真正完全收敛到：

- 发 quota
- 发 coarse task/bundle dispatch
- 回收 coarse completion

当前它在逻辑上仍残留：

- request/pair 时代的调度思维
- group 语义与 worker 本地执行之间的耦合

这会导致：

- `group_wait` 仍然高
- coarse dispatch 粗了但不高效

## 7.7 问题七：控制状态和数据没有真正分层

compute dominate 的关键要求之一是：

- 数据层和控制层分离

也就是：

- GM / HBM 主要承载数据
- 本地执行引擎主要承载状态

当前系统仍然存在较多“用存储层表达控制语义”的倾向，这会天然放大：

- `issue_write`
- `sched_protocol`

从根本上说，这也是 control-bound 的重要来源。

## 8. 为什么当前 patch 型优化已经接近极限

历史上已经做过：

- batch compute / batch load
- window-2
- WCP 原型
- coarse completion 原型
- header-only WCP
- 双缓冲原型

这些步骤都证明：

- 局部 patch 可以带来正确性和一定收益
- 但不能从根上改变系统性质

当前已经可以明确判断：

**继续围绕旧 task 粒度、旧控制模型做 patch，已经很难让系统真正转向 compute dominate。**

## 9. 围绕 compute dominate，真正需要的架构方向

基于以上问题，当前系统如果想真正实现 compute dominate，必须沿以下方向重构：

### 9.1 从 request/pair 级控制切换到 TaskBundle

新的基本执行粒度不应是 request/pair，也不应只是单 task，而应是：

- `TaskBundle`

建议第一版 bundle 粒度：

- 固定一个 `m_tile`
- 覆盖多个连续 `n_tile`

这有利于：

- A tile 复用
- 控制摊销
- 数据流连续性

### 9.2 从 WCP 原型升级为真正的 WorkerExecutionEngine

WEE 不应只是“会按步骤做事”，而应是：

- 双缓冲 / 多缓冲
- prefetch/load/compute/writeback 显式重叠
- 事件驱动推进
- 本地状态驻留

### 9.3 引入本地 accum store

为了让 bundle 下多个 `n_tile` 共用 A tile，同时不引入核间规约，必须有：

- worker 私有的 local accum store

这样：

- partial sum 可本地 spill / reload
- array output buffer 只服务当前活跃 tile

### 9.4 manager 彻底收敛为 Quota Dispatch

manager 应最终只做：

- quota 发放
- TaskBundle dispatch
- coarse completion 回收

而不再参与细粒度推进。

### 9.5 mailbox 退出执行路径

最终 mailbox 只保留：

- coarse task queue
- coarse completion queue

必须删除：

- worker request ring
- worker done ring
- per-step mailbox 状态表达

## 10. 当前报告的结论

围绕 compute dominate 目标，当前架构的问题可以概括为：

1. 执行仍然是 CPU / 控制驱动，不是数据流驱动
2. worker 侧还没有成为高效本地执行引擎
3. task 粒度太细，不利于数据复用和控制摊销
4. A tile 复用尚未成为执行组织的中心原则
5. partial sum 驻留层次不合理，阻碍 bundle 化
6. manager 仍未彻底退化为 coarse quota dispatcher
7. 控制状态与数据没有完全分层，导致 control-bound 问题长期存在

因此，要实现 compute dominate，必须接受一个结论：

**这不是再优化几个 mailbox 参数、再 patch 几个状态机就能解决的问题，而是必须围绕 TaskBundle + WorkerExecutionEngine + Quota Dispatch 做架构级重构。**

## 11. 后续建议

建议后续所有实现工作，都以如下优先级推进：

1. `TaskBundle` 设计与实现
2. `local accum store` 设计与实现
3. WEE 真正流水化（prefetch/load/compute/writeback overlap）
4. manager quota dispatch 收口
5. 删除旧 request/done ring

只有在这条主线下推进，系统才有现实可能从当前的 control-bound 状态，转向 compute-dominant 状态。
