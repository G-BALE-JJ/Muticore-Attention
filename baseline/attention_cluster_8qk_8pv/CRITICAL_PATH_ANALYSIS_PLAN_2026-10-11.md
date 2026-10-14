# 端到端关键路径重建计划

日期：2026-10-11

## 目标

把优化目标从“减少某个 phase 的统计值”改成“缩短最终 HBM 输出完成事件的依赖链”。所有候选优化必须先经过关键路径分析，再决定是否实现。

当前正式范围固定为：Hq=32、Hkv=8、D=64、hidden=2048、causal+RoPE，S1024/S2048；流程为 RMSNorm→QKV→RoPE→causal GQA→O，不包含 Wo、residual、MLP。当前推荐数据流为 64 arrays、单 operand context、output retention、输入与本地权重预取、array_groups=1。

## 阶段一：冻结可复现基线

保留以下三个正式 baseline 作为参考：

| S | E2E | Projection | Attention |
|---:|---:|---:|---:|
| 1024 | 573,772 | 416,538 | 155,215 |
| 2048 | 1,354,890 | 801,164 | 549,932 |

另外保留推荐读取流水版本：544,931 / 1,304,066 cycles。

每个实验必须归档：

- 输入文件 SHA256；
- guest、共享库和关键源码 SHA256；
- 所有 `GOLEM_*` 配置；
- MPI rank 日志；
- HBM 区域逐字节比较；
- NumPy 输出误差；
- 16 个 worker 的开始、完成和失败状态。

任何 baseline 数字变化时，先停止优化实验，重新解释变化原因。

## 阶段二：增加端到端事件时间线

现有 phase aggregate 不够用。需要给每个 worker、manager 和 Attention job 增加带唯一 ID 的事件记录：

```text
job_create
descriptor_submit
descriptor_accept
norm_batch_ready
gm_read_submit
gm_read_service_start
gm_read_complete
weight_program_submit
weight_program_complete
input_scatter_submit
input_scatter_complete
array_launch
array_compute_complete
output_read_submit
output_read_complete
worker_complete
manager_observed_worker_complete
attention_descriptor_submit
attention_descriptor_accept
qk_tile_start / qk_tile_complete
softmax_tile_start / softmax_tile_complete
pv_tile_start / pv_tile_complete
final_output_dma_submit
final_output_dma_complete
root_tensor_complete
```

每条记录至少包含：

```text
stage, job_id, core, worker, head, block, tile,
resource, generation, submit_cycle, start_cycle, done_cycle,
predecessor_id, queue_depth
```

事件记录必须区分三个时间：请求被软件接受、资源实际开始服务、结果对消费者可见。不能用 callback 时间代替资源服务开始时间。

## 阶段三：建立关键路径提取器

新增离线分析脚本，读取所有 rank 日志并构建 DAG：

```text
event_start  → event_done
producer_done → consumer_start
同一资源前一请求 done → 后一请求 start
同一 operand/accumulator 的 generation 依赖
worker_complete → manager_observed
manager_observed → Attention dispatch
```

分析器输出：

1. 最终 `root_tensor_complete` 的关键路径事件列表；
2. 每个事件的 slack；
3. 每种资源的服务时间、排队时间、空闲时间；
4. 关键路径上每个等待的直接前驱；
5. 16 个 worker 的 start skew 和 finish skew；
6. Projection→Attention 阶段屏障的等待时间。

关键路径上的事件才允许进入下一轮优化候选。只在 phase aggregate 中变短、但不在关键路径上的阶段标记为“非关键改进”。

## 阶段四：做零成本 oracle 实验

这些实验只用于定位，不作为性能结果发布，也不改变硬件结论。通过 trace/replay 或受控延迟替换，分别设置一种理想条件：

| Oracle | 目的 |
|---|---|
| GM operand read latency = 0 | 判断 GM→operand 是否仍在最终关键路径 |
| weight arrival = instant | 判断 HBM→GM 权重供给是否主导等待 |
| matrix program latency = 0 | 判断阵列编程是否限制最终完成时间 |
| input scatter latency = 0 | 判断输入搬运是否限制最终完成时间 |
| worker start skew = 0 | 判断 release/启动串行是否值得优先处理 |
| Projection→Attention barrier = 0 | 判断跨阶段流水是否有潜在收益 |
| Attention internal barrier = 0 | 判断 Attention 内部调度是否主导剩余差距 |

每个 oracle 记录：理想 E2E、相对 baseline 的最大可得收益、关键路径是否改变。若理想收益低于总延迟的 1%，停止该方向的实现工作。

## 阶段五：完成正交实验矩阵

正式 S1024/S2048 采用同一输入和同一硬件配置，完成以下 2³ 矩阵：

| 方案 | Input prefetch | Local weight prefetch | Array groups |
|---|---:|---:|---:|
| B | 0 | 0 | 1 |
| I | 1 | 0 | 1 |
| W | 0 | 1 | 1 |
| IW | 1 | 1 | 1 |
| G | 0 | 0 | 2 |
| IG | 1 | 0 | 2 |
| WG | 0 | 1 | 2 |
| IWG | 1 | 1 | 2 |

每个 case 必须同时检查：

- E2E 和 critical-path length；
- Projection、Attention、handoff；
- GM / array-buffer 实际服务量；
- queue wait 和 producer-not-ready wait；
- worker start/finish skew；
- 输出数值和 HBM byte-exact。

使用二阶差分判断交互项：

```text
interaction(I,W) = E(IW) - E(I) - E(W) + E(B)
```

这样可以确认两个预取是否互相争用，而不是只看组合开关的最终数字。

## 阶段六：按收益门槛实现优化

候选方向按以下顺序处理：

### 1. Worker 启动 skew

先记录 release issue、worker observed、job accept 三个时间点。只有确认约 30k cycles 位于最终关键路径，才修改 guest release 顺序。目标是集中提交 release，去掉不必要的打印和逐 worker 等待，同时保持 descriptor 顺序和资源归属不变。

验收：关键 worker start skew 至少下降 50%，Projection 和输出数值不变，E2E 必须下降。

### 2. HBM→GM 权重供给

当前本地读取流水已经把 GM→operand 的一部分等待隐藏，但下一个权重未 ready 的等待暴露出来。先记录 DMA submit、HBM complete、GM cache visible、local weight consume 四个时间点，再测试：

- 同一 head 内更早提交；
- 已有 outstanding 额度内的有限多请求；
- head 边界首 weight 预取；
- 缓存槽 generation 和消费顺序。

不增加 GM 容量、DMA 通道或 operand context。只有关键路径中的权重等待下降且 E2E 下降，才保留实现。

### 3. Projection→Attention 跨阶段流水

确认 Attention 是否必须等待所有 Projection worker。按 KV head 或可验证的 ready unit 建立 producer/consumer 依赖，让已经完成且资源空闲的 worker 提前处理 Attention。

验收：第一个 Attention tile 提前，Projection 不被反向阻塞，最终 HBM 输出 byte-exact，E2E 净下降。若只提前首个 tile、但最终 worker 完成时间不变，则不算收益。

### 4. Norm→Projection 跨 block 流水

只在 S2048/S3072 多 block 中评估。使用已有 GM 空闲行区作为有界 generation slot，记录 Norm ready、Projection consume 和 GM 争用。S1024 不把该方案当作收益来源。

### 5. Array groups

暂不默认启用。当前实验已经证明它有真实编程/MAC 重叠，但额外 matrix broadcast 抵消收益。除非新的关键路径模型显示阵列端口成本可以被隐藏，否则保持 `array_groups=1`。

## 阶段七：每个候选的停止规则

出现以下任一情况就停止该候选：

- oracle 最大收益低于 E2E 的 1%；
- 关键路径没有改变；
- 局部统计改善但 E2E 连续两次没有改善；
- 需要新增 operand context、GM 容量、端口或 DMA 通道；
- 只能通过增加硬件资源才能取得收益；
- 数值、HBM、generation 或资源归属验证不稳定。

## 下一步执行顺序

1. 先实现事件时间线和关键路径提取器；
2. 用不改变数据流的 baseline 运行 S1024/S2048，验证提取器重建 E2E；
3. 做七类 oracle，得到每类瓶颈的最大理论收益；
4. 完成正式 2³ 正交矩阵；
5. 只选择 oracle 收益最大且位于关键路径的一个方向实现；
6. 重新跑完整正确性和 E2E 对照；
7. 更新本报告、交接文档和机器汇总，再决定下一方向。

本计划的核心判据是：**任何优化都必须同时缩短最终关键路径，并在完整 E2E 中得到可重复收益。**

## 可以合并执行的任务

### 批次 A：一次完成测量基础设施

以下任务属于同一个事件数据契约，可以在一次代码变更中完成：

1. 定义统一事件格式和 generation/producer/consumer ID；
2. 在 Projection、worker 控制路径和 Attention 完成路径写入事件；
3. 编写 rank 日志合并器；
4. 编写关键路径 DAG 提取器；
5. 编写 baseline 重建检查，确认提取出的最终完成时间等于现有报告的 E2E；
6. 为事件解析器增加缺失、重复、乱序和跨 generation 的测试。

这批任务可以一起开发，因为它们都不改变调度行为，只增加观测能力。

### 批次 B：一次完成实验编排

在批次 A 的解析器通过 baseline 重建后，可以一起完成：

1. 七类 oracle 延迟开关；
2. 2³ 正交实验 runner；
3. 输入、源码、库、配置、MPI 和 HBM 输出的自动一致性检查；
4. slack、关键路径、资源服务量和 E2E 的汇总报告；
5. S1024/S2048 的正式实验队列。

oracle 和正交矩阵可以共用同一个 runner，但每个 case 仍必须独立归档，不能把不同 case 的统计混合。

### 批次 C：一次补全资源时间线

以下观测可以同时加入，因为它们只记录时间，不改变算法：

- worker release issue / observed / accept；
- GM DMA submit / service start / complete / cache visible；
- array-buffer program/scatter/launch 的排队和服务；
- Projection completion 到 Attention dispatch 的等待；
- 最终 HBM output 的提交和完成。

这批可以与批次 B 的 runner 一起准备，但正式运行必须先完成批次 A 的 baseline 重建。

### 不能合并的任务

以下修改必须分别验证：

- worker release 集中提交；
- 权重 DMA 多请求或跨 head 预取；
- Projection→Attention 跨阶段流水；
- Norm→Projection 跨 block 流水；
- array groups 或其它阵列调度改变。

它们都可能改变关键路径和资源竞争。不能在同一个仿真 binary 中同时启用，否则无法判断收益来源，也无法解释新的等待。

推荐的实际执行方式是：先一次完成批次 A+C，再用未修改调度的 baseline 验证；随后一次完成批次 B 的 runner 和 oracle/matrix case，最后才逐个实现真正的调度优化。

## 批次 A+C 执行状态（2026-10-12）

批次 A+C 已完成第一版并通过正式基线验证：

- `ProjectionJob` 增加 opt-in phase start/done 事件，带 `event_id/predecessor/core/job/generation/resource/cycle`；
- 保留并接入已有 Attention milestone，覆盖 descriptor、worker complete、manager wait、root tensor；
- `analyze_critical_path.py` 检查 16 个 worker 的事件链、phase 配对、Projection 同步边界，并用 `chain_start_cycle + worker_complete` 精确重建 E2E；
- 资源累计服务量、关键 worker 的最长 phase、worker 启动/完成跨度和 Projection→Attention handoff 已汇总；
- 正式 S1024/S2048 的重建结果为 573,772 / 1,354,890 cycles，与既有 baseline 完全一致；
- 详细结果见 `CRITICAL_PATH_TRACE_RESULTS_2026-10-12.md` 和 `results/critical_path_trace_20261012/critical_path_analysis.json`。

第一版只有 phase 级时间线。2026-10-12 后续已补齐 GM 端口、DMA transport/landing、HBM admission/首命令/完成、array-buffer、物理 MAC、WCP、SFU、snapshot ready/consume 和 worker release/observed/accept 资源时间，详见 [细粒度资源时间报告](FINE_RESOURCE_TIMELINE_2026-10-12.md)。

需要区分完成范围：当前提取器能够校验资源区间并重建 phase/control 链与原报告 E2E；全部跨层 producer/consumer 边、资源争用依赖和 slack 的完整 DAG 尚未实现。oracle 不能通过简单减去 phase 或资源累计服务量来替代。

## 批次 B 执行状态（2026-10-12）

已实现单选诊断开关、完整 2³ 软件矩阵、固定起点与输入/源码/库一致性校验，并完成 S1024/S2048 共 30 次正式运行；所有 case 的 NumPy 和基线逐字节输出检查通过。默认 all-off E2E 仍为 573,772 / 1,354,890 cycles。

具体实现范围和正式报告见 [BATCH_B_2026-10-12.md](BATCH_B_2026-10-12.md)，完整汇总为 `results/batch_b_20261012/summary.json`。两个 barrier oracle 仅覆盖全局 producer 就绪后的提交间隙、以及已就绪状态的调度推进；它们没有实现跨 head/tile 的提前消费 oracle。完整跨资源 DAG/slack 仍待补齐，不能将本批结果解释为其已完成。

真实组合中，IW 在 S1024/S2048 得到 544,931 / 1,304,066 cycles；IG 在 S1024 仅再省 23 cycles，但在 S2048 多耗 100 cycles。三个因素全开均慢于 IW。下一步应先选定稳定组合，再分别测量启动错峰压缩和单 GM 端口下的预取顺序，避免继续叠加未经归因的开关。
