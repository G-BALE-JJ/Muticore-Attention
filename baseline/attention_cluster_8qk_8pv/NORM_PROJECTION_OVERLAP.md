# RMSNorm 与投影分块重叠分析

> **架构纠正（2026-10-10）：** 本文的历史投影/E2E 测量或重叠估计使用了 manager 本地 arrays，不符合「manager 只控制、16 个 worker 拥有阵列」的目标拓扑，不能作为该架构的性能证据。新的 worker 执行基准见 [WORKER_PROJECTION_ARCHITECTURE_FIX.md](WORKER_PROJECTION_ARCHITECTURE_FIX.md)。

结论：SFU 与投影 arrays 可以独立推进，但当前 weight-major 遍历与单份输入
区限制了收益。最值得试验的是 **保持 256 行权重复用，在当前块投影期间
预规范化下一块的前 128 行**。它复用当前最佳路径的 GM 空闲区，不增加资源。
S2048 乐观估计约省 68,000 cycles（E2E 2.87%）；S1024 没有下一块，
不能获得这项跨块收益。本报告是源码/日志分析，尚未实现或实测新调度。

## 真实依赖和时间

采用最新 `results/attention_causal_balance/final/balanced` 正式结果，
FP16、Hq32/Hkv8/D64、hidden2048、64 arrays、packed Norm、shared input、
audit=0、projection RoPE fusion=0，既有 GM/NoC/SFU 配置不变。

每个 manager 处理 S/4 行：S1024 有一个 256 行块，S2048 有两个。
`LoadBlock → NormBatch/NormWait → 全部写完 → LoadInput` 是当前控制流。
每个 Norm batch 4 行，共 64 batches/块；每个投影 row group 64 行。

| manager 0 / 每个 256 行块 | 第一块 | S2048 第二块 |
| --- | ---: | ---: |
| 全部 Norm 就绪 | 136,100 | 136,134 |
| SFU vector 工作 | 111,808 | 111,808 |
| 前 64 行写完 | 33,956 | 33,990 |
| 前 128 行写完 | 68,004 | 68,025 |
| 前 192 行写完 | 102,052 | 102,076 |

时间由 `[SFU_RMSNORM]` 的 issue/complete ticks 推导，1GHz 下除以 1000。
“就绪”包含最终 GM 写完成，不是仅 SFU vector-ready。第一块的 Norm
非 SFU span 为 24,292 cycles；另有块初始化/gamma 首次 DMA 开销。

RMSNorm 对整行 hidden2048 求平方和再缩放；不能在 reduction 未结束时
把一个 hidden pair 当成最终规范化结果交付。可以按行分批，不应改变每行
reduction、epsilon 或 FP16 编码。`SFU::issueVectorOp` 有既有独立的
`vectorOpFreeTick_`；projection arrays 经 WCP 操作，因此可以并行，
但 HBM/DMA、GM 读写端口仍共享。

## 为什么简单的 64 行生产/消费不是主要收益

当前顺序是：同一个 output head/hidden pair 下处理行组 0、1、2、3，
再进入下一个 hidden pair，然后下一个 Q/K/V head。第一个行组首 pair
只有 136 compute/launch cycles，加入读写与派发后仍远小于生产下一组
Norm 所需约 34,000 cycles。提前启动后，consumer 会立即追上 producer。

保留此遍历，能隐藏的是前三个行组在首 hidden pair 的实际投影服务，
不是后三组的约 102,000 Norm cycles。日志的平均 pair 前台时间约数百
cycles，只支持“小幅启动重叠”的判断；精确首 pair 收益需追加逐 pair 时间戳。

实现时即使是这个小试验，也必须同时阻止 demand read、future-input
prefetch 和 scatter 读取尚未写完的行组。仅提前设置 `blockLoaded_`
会让 `Launch` 中的预取读到旧数据。SFU-ready 后还有 16 个 packed 写段/
batch，不能提前发布 ready。

不直接把 reuse block 改成 128/64 行：这样会增加整套 Q/K/V 权重 DMA、
GM 读取和 array programming。按当前前台调度，S1024 从 256 改成 128
会多一套约 55,296 weight-read + 64,512 program cycles，合计 119,808，
超过第二个 128 行块约 68,000 Norm cycles 的可重叠窗口，且未计额外 HBM。
这不是所有新调度的绝对下界，但足以否定“拆小即可免费重叠”的假设。

## 推荐计划：跨块预规范化 128 行

1. 第一块保持完整 256 行 Norm，随后执行原来的全部 Q/K/V 投影。
2. 同时利用 manager SFU，按原 4 行 batch 计算下一块前 128 行，写到
   两个已有 GM 空闲槽；只保留一个 Norm 运算/输出 batch，不增加在途 SFU 任务。
3. 当前块的所有 input prefetch、array compute、Q/K/V head 消费完成后，
   切换输入映射：下一块前两个 64 行组指向预生产槽。
4. 下一块剩余 128 行写到刚释放的原输入区，全部就绪后开始原投影遍历。
   初版保留这个全块 barrier，可避免收益很小而复杂的首 pair 并发。
5. 后续块交换两个 prefix 槽与原输入区前两个行组的归属；不做复制/转置，
   不增加权重重载或改变 hidden-pair/partial 的运算顺序。

256 行 Norm 输入本身是 1 MiB。当前 2 MiB GM 内再放一整块 1 MiB，并加上
权重、Norm scratch 和 staging，会超容量，因此不采用完整双块 ping-pong。
前 128 行为 512 KiB，下面的两个区域在选定最佳路径中可复用。

| 相对 manager GM 的区域 | 当前最佳路径的使用 | 候选用途 |
| --- | --- | --- |
| `0x08000..0x48000`，256 KiB | legacy input/weight temp、融合 RoPE scratch 未使用 | 下一块一个 64 行组 |
| `0x48000..0x4C000`，16 KiB | Norm 原始输入 batch | 保留 |
| `0x58000..0x59000`，4 KiB | gamma | 保留，每 batch 仍计 GM 读取 |
| `0x68000..0xA8000`，256 KiB | D64 实际使用的 32 个权重 slots | 保留 |
| `0xA8000..0xE8000`，256 KiB | 64-entry 软件权重区的后 32 slots；D64/hidden2048 不访问 | 下一块另一个 64 行组 |
| `0xE8000..0xF0000`，32 KiB | partial GM 预留 | 保留，不借用 C-buffer |
| `0xF0000..0x1F0000`，1 MiB | 当前块规范化输入 | 保留，按行组切换部分地址映射 |
| `0x1F0000..0x1F8000`，32 KiB | 已有 projection snapshot 容量预算 | 保留，不扩容 |

源码已检查 selected path 的 scratch 访问，分析器验证静态范围不重叠且不越过
2 MiB GM 及末尾 DMA 状态字。动态 callback/prefetch 生命周期尚需实现验证。
软件权重区是 GM 地址区，不是把物理 SRAM 扩大或给 array 增加 operand bank。
只在条件满足时采用该布局；D128、融合 RoPE、非 packed、非共享/非配对等
路径应保留原映射，不使 accelerator 的其他形状丧失可用性。

## 预期收益和剩余开销

| 方案 | S1024 | S2048 |
| --- | ---: | ---: |
| 当前实测 E2E | 1,066,757 | 2,373,527 |
| 跨块预规范化 64 行，乐观隐藏量 | 0 | 33,990（约 1.43%） |
| 跨块预规范化 128 行，乐观隐藏量 | 0 | 68,025（约 2.87%） |
| 128 行方案，无新增争用时的 E2E 估计 | 1,066,757 | 约 2,305,502 |

这些是保持其他阶段不变的估算，不是测试结果或严格完整系统下界。四个
manager 的 prefix 时间略有差异，最终由整个 producer/handoff/Attention
关键路径决定。当前块投影约 774,000 cycles，为未来 prefix 的约 68,000
cycles 提供足够长的时间窗；真正限制是竞争和控制实现。

Norm 与投影共用 GM FIFO。Norm 输入的 DMA landing、raw/gamma read、
packed write 会延长 weight/input read 等待，因此默认利用 compute 阶段和
无紧急前台 GM demand 的空闲窗口提交，每 lane 有明确上限并重试。不能只把
Norm wait 降幅当收益，更不能改变 GM 请求接收间隔、带宽或增加 queue credits。

S1024 的更直接候选是 **Norm batch 内预取**：在当前 batch 的 1,747
SFU cycles 中预取下一批 4 行的 X，并在现有 Norm scratch 内用两个 16 KiB
原始输入槽交替。当前单块 Norm DMA/dispatch 为约 12,389、raw/gamma GM
read 为 5,760 cycles，合计约 18,149 是这部分统计的乐观可重叠量，约 E2E
1.70%；最终输出写回与 batch 生命周期仍要支付，不能保证全部隐藏。
若采用它，必须按新 Norm 时长重算跨块收益，两项不能直接相加。

## 实施与验收边界

Norm producer 需要独立的目标 block/row、DMA/GM 回调进度及完成标记。
当前 `phase_`、`pending_`、`localData_`、`localAddr_` 是全局串行上下文，
不能简单在 projection tick 中调用 `beginNormBatch()`：它会覆盖前台状态。
应复用已有 Norm batch 操作数/结果及 GM scratch，明确它们与 projection
snapshot 的归属；不新增未计入空间预算的 payload 队列。

验收使用同库开关对照、现有 Attention balance 和最佳投影计划，独立 NumPy
及全部保留输出 byte-exact；要求 Norm jobs/元素、weight DMA/programs、
partial 与 input payload 总量不变。记录 Norm producer 起止、prefix ready、
block switch、consumer stall、GM queue/retry 和完整 E2E。补测单块回退、
多块槽交换、D128/RoPE 回退及小请求/低带宽压力，不扩大硬件资源来取得通过。

分析数据见 [`norm_projection_overlap_analysis.json`](norm_projection_overlap_analysis.json)。

```bash
python3 baseline/attention_cluster_8qk_8pv/analyze_norm_projection_overlap.py \
  --artifact-root results/attention_causal_balance/final/balanced \
  --output baseline/attention_cluster_8qk_8pv/norm_projection_overlap_analysis.json
```
