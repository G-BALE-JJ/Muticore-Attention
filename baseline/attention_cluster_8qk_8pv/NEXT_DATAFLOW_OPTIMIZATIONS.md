# 下一轮固定资源数据流优化候选

> **架构纠正（2026-10-10）：** 本文的历史投影/E2E 测量或重叠估计使用了 manager 本地 arrays，不符合「manager 只控制、16 个 worker 拥有阵列」的目标拓扑，不能作为该架构的性能证据。新的 worker 执行基准见 [WORKER_PROJECTION_ARCHITECTURE_FIX.md](WORKER_PROJECTION_ARCHITECTURE_FIX.md)。

本分析使用 `results/projection_partial_restore/restore_ahead` 的最新正式结果，
不沿用旧 FP32 非因果测试的瓶颈结论。当前 E2E 为 S1024 1,265,454、S2048
2,974,207 cycles；producer span 为 1,013,877、2,028,116；Attention 为
250,095、943,486。该文记录实施前的候选分析。候选 1、2 现已实现并验证，结果见
[`PROJECTION_INPUT_FLOW.md`](PROJECTION_INPUT_FLOW.md)：新最佳 E2E 为
1,161,859 / 2,766,728，输出读取的暴露排队明显减少。下面的旧计数仅用于
解释候选方案来源，后续优先级应按新结果调整。

## 1. RMSNorm 写出时直接生成投影消费布局

当前规范化 GM 输入块按 row-major 保存。投影每次需要 64 行各自的两块相邻
64-column hidden tiles，每行 256 B、行间步长 4096 B，必须 gather 64 次。
这 16 KiB 不是物理连续区域，不能简单把 64 个请求合成四个请求。

候选：在现有规范化 GM 区域内按 64-row group / hidden-tile pair / row 排列，
由 RMSNorm 的写出直接使用这个地址布局。每个 16 KiB 输入 pair 就连续了，
可按原有 4096 B API 上限读取；不做第二次全块转置，不扩大 GM 或 input
snapshot，内部 scratch 布局与消费地址一起调整，外部张量布局不变。

| 单个 64-row 输入 pair | 当前 | 候选布局 |
| --- | ---: | ---: |
| 读取 payload | 16 KiB | 16 KiB |
| GM 请求数 | 64 × 256 B | 4 × 4096 B |
| 非流水 GM 的纯端口服务 | 128 cycles | 68 cycles |

S1024 每 manager 有 3,072 个输入 pair，因此这部分请求可从 196,608 降到
12,288，纯读端口服务可少 184,320 cycles。这个数字是资源服务减少量，**不是
E2E 收益预测**：大部分输入读取已有后台预取，真正收益取决于与权重读取的争用。

代价必须一起测量：当前 hidden2048 RMSNorm 每批四行，输出共 16 KiB；写到
上述布局后会分为 16 段各 1 KiB，相比现有四段各 4 KiB，GM 写请求及控制
开销会增加。应按批直接写已有 SFU 结果，保持现有 payload 上限，避免额外
整块 snapshot。还需支持尾行、审计输出模式和非配对路径的布局回退。

## 2. 输入预取的提交批量和权重请求次序

当前 `pumpRead` 循环尽量提交整个 future-input gather，并且 tick 内先执行
`pumpBackground`，再执行 `pumpLocalAccess`。同一个 GM 读端口随后才处理
前台权重请求。S1024 每 manager 前台权重读占 99,840 ticks，即 1,536 次
8 KiB 权重读取平均 65 ticks；两次 4 KiB 请求的纯服务只有 34 cycles。
多出的部分包括排队和控制，不能全部归因于单一输入预取。

候选：在请求提交端分批推进已有输入 snapshot；存在急需前台权重读取时，
先提交其请求，再继续输入预取。保持 GM 原 FIFO、端口、带宽和总工作量不变。
先试少数固定批量，观察权重 wait、input-not-ready、compute idle 和 E2E，
不能仅以权重 wait 下降判断成功。输入不能被饿死；也不应新增 snapshot。

## 3. 共享 array-buffer 端口的截止时间调度

S1024 每 manager 输出 read 前台占 163,920 ticks，其中 partial 154,800、
final 9,120。3,072 次 gather 的纯服务为 101,376 cycles；两者相差 62,544，
包含共享端口排队、WCP 派发和状态切换。提前 restore 后，restore 含 dispatch
仍占 72,576 ticks。它们是尚存的较大搬运阶段。

候选：进一步记录 output request 的提交、端口启动和完成时间，并标记排队在
input scatter 或 weight broadcast 之后的等待。只在确认冲突来源后，调整
现有请求的提前提交时机，使无紧迫依赖的预取避免阻塞 output drain/restore。
输出必须完整 drain 后才能覆盖，restore 完成后才能 compute。

三个流共享一个端口，重新排序不减少总服务量，也可能让 input/weight wait
增加。因此不采用盲目“输出总是最高优先级”，而是比较整体 E2E 和空闲窗口。

## 4. 因果 Attention 的工作量分配与 K/V 复用

本候选现已在现有 descriptor 粒度实现并验证，见
[`CAUSAL_BALANCE.md`](CAUSAL_BALANCE.md)。早晚查询区间按 KV head 配对，
同库 E2E 为 S1024 1,162,378 → 1,066,757、S2048 2,767,997 → 2,373,527；
Q/K/V 搬运量不变，独立 NumPy 与所有保留输出 byte-exact 通过。
下面保留实施前的分析背景；最新瓶颈及下界比较以新报告为准。

当前运行时按连续 query 区间分派。causal 下较晚的 query 需要更多 key windows。
S2048 当前 QK compute 忙时约为 38,016 到 240,768 cycles，PV 忙时约为
57,984 到 289,949 cycles。两者均有明显不均衡。

Attention stage span 为 943,486；QK span 为 777,038，SFU span 为 936,488，
PV span 为 935,995。PV 在最后 SFU 完成后的尾部仅 1,259 cycles。各 stage
重叠，不能相加；仅平衡 PV 可能不改变 E2E。

候选：按每个 query row group 的有效 causal windows 估算代价，将早晚 row
groups 配对分配到现有 QK/SFU workers，并让 PV 分配匹配该代价。按每行维持
原有 key-window/在线 softmax/PV 累加顺序，优先复用现有 KV 本地副本与路由。
首先核对固定 HBM/GM 所有权是否限制可调度范围；跨区分配若增加 K/V DMA 或
破坏 window-major 复用，需要纳入对照，而不能假设负载均衡免费。

同一 KV head 的四个 Q heads 是 GQA 复用机会，但现有 V cache 已有零 start
miss、window-major 和广播；应先统计真实重复 HBM loads/GM payload reads，
再决定是否有可消除的重复搬运，不能把“零 miss”解释为零预加载流量。

## 建议执行次序与验证

最有价值的布局实验是候选 1：它在保留 GM 时序模型的同时减少小请求，直接
针对已确认的固定请求开销。最小的调度实验是候选 2，可先用它验证共享 GM
争用是否正在限制权重供给。两者应分别对照测量，再测试组合，不混合归因。
然后做候选 3 的端口等待细分；长序列优化并行推进候选 4 的代价/归属分析。

继续使用同一输入、同一库、同一固定硬件的 S1024/S2048 正式端到端验证，
独立 NumPy O、全部保留输出 byte-exact、请求/字节/服务统计和关键路径 cycle。
布局方案预期请求数变化，应按新的地址布局逐项核对；纯调度方案则要求工作量
不变。GM 流水接口校准单独研究，不计入以上固定模型的数据流收益。

此阶段不优先增加 partial 输出上下文、片上缓存或端口；也不优先把遍历改成
单 row group 完成全部 hidden tiles，因为后者虽然减少 partial 往返，却可能
使四个 row groups 重复权重编程。完整 RoPE 融合的既有正式结果更慢，不能
仅因融合程度增加就作为默认方向。
