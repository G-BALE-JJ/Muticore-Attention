# GPU-Competitive FlashAttention Roadmap

## 1. Objective

当前项目只保留一个性能目标：在相同的单头、FP32、non-causal Attention
负载下，使 SST 架构的 accelerator completion latency 优于实测 RTX 5060。
在达到该目标或证明当前资源约束下无法达到之前，暂不扩展多头 Attention 和
MoE。

MPI 仍用于缩短 SST 仿真的主机 wall time。MPI rank 数不改变被模拟架构的
completion latency，因此不能作为超过 GPU 的架构优化手段。

## 2. Comparison Contract

主比较采用以下共同负载：

| Profile | Shape | Dtype | Mode |
|---|---|---|---|
| E3 | `B=1,H=1,S=1024,D=128` | FP32 | non-causal |
| E4 | `B=1,H=1,S=2048,D=128` | FP32 | non-causal |

主口径是：

- SST：从 root descriptor accept 到 accelerator tensor completion；包含架构内的
  HBM/Local GM 搬运、manager/worker dispatch、QK、scale、online Softmax、PV
  和输出 DMA。
- GPU Scope A：CUDA Event 包围 GPU 上的 Attention 调用；包含 QK、scale、
  Softmax、PV 和 device-side 输出写回；不包含 H2D、D2H、输入生成和 dtype 转换。
- GPU FP32 math 是同精度主对照。FP16/BF16 FlashAttention 作为低精度性能上界，
  不与 SST FP32 宣称同精度公平比较。
- 所有时间均同时报告毫秒。`1 GHz normalized cycles = latency_seconds * 1e9`，
  只是统一的延迟表示，不是 RTX 原生 SM cycle 计数。
- SST 文档中的现有 completion cycle 字段也是由 verifier 按 1 GHz 换算得到的
  延迟值。下一阶段必须同时核对 SST timebase、CPU clock、array clock 和 verifier
  conversion，避免把归一化周期误写成某个组件的原生周期。

GPU Scope B（Python 调用、launch、CPU 等待）和 Scope C（额外 D2D staging）
作为敏感性分析；Scope D 的 H2D 只作数据准备诊断，不进入主比较。

## 3. Current Baseline And Gap

| Profile | SST FP32 baseline | RTX 5060 FP32 Scope A | Current gap | Pass target |
|---|---:|---:|---:|---:|
| E3 | 0.491071 ms / 491,072 normalized cycles | 0.097568 ms / 97,568 normalized cycles | GPU 5.033x faster | SST `< 97,568` |
| E4 | 1.899764 ms / 1,899,765 normalized cycles | 0.345984 ms / 345,984 normalized cycles | GPU 5.491x faster | SST `< 345,984` |

当前默认路径使用通用 GEMM/WCP、QK+PV 二叉树矩阵广播、滚动 N+2 K/V 双缓冲、
group4 K/V reuse、group-owned 16 KiB V staging、depth-2 QK input pipeline、PV
input residency 和 depth-2 operand-only cross-tile pipeline。
QK/PV 分块、数值次序和 Attention 映射保持不变。统计同时校验 fanout/广播流量、
active-K 启动与动态阵列周期、V-tile 容量/端口/带宽等待，以及关键 worker 的预取
DMA、命中和等待。当前 Q256 group1/2/4 均为 71,916 cycles；Q1024/Q2048 的
重复运行分别精确复现 491,072/1,899,765 cycles 和对应 SST ticks。

为避免仅在测量噪声内“险胜”，工程目标保留约 20% 裕量：E3 `< 78,000`，
E4 `< 276,000` normalized cycles。正式成功门禁仍以同精度 Scope A 的严格小于为准。

已审计的 GPU Scope A 低精度结果如下：

| Profile | FP16 Flash | BF16 Flash |
|---|---:|---:|
| E3 | 0.045824 ms / 45,824 normalized cycles | 0.046976 ms / 46,976 normalized cycles |
| E4 | 0.101664 ms / 101,664 normalized cycles | 0.102208 ms / 102,208 normalized cycles |

历史 Phase A/B 基线下，Scope B 的 E3 FP32/FP16/BF16 加速比分别约为
`32.4x/58.7x/57.1x`，E4 为 `38.2x/117.5x/114.0x`。Scope C 下分别为
E3 `29.5x/56.1x/55.9x`、E4 `38.7x/121.8x/117.4x`。这些敏感性口径均未改变
GPU 显著领先的结论，但不作为当前冻结基准的通过门禁。

CUDA Event 空测中位开销约为 0.0023 ms，低于实际 Attention 延迟。当前 CUPTI
返回 `CUPTI_ERROR_INVALID_DEVICE`，所以 fused FlashAttention 暂无可信的 kernel
级拆分；不得用估算值代替。

## 4. Dual-Sided Stage Decomposition

下一阶段先建立 SST/GPU 两侧可映射的阶段数据，而不是立即修改数据流。

| Common stage | SST observations | GPU FP32 math measurement |
|---|---|---|
| Input movement | HBM DMA、Local GM load、queue wait | kernel 内 device memory/cache read；Scope C 另测 D2D staging |
| QK | matrix program、input program、compute/readout | explicit `Q @ K^T` CUDA Event interval |
| Scale | SFU/worker scale interval | explicit score scaling interval |
| Softmax | online max/exp/sum/norm and SFU wait | explicit `softmax` interval |
| PV | V program/reuse、input program、restore、compute | explicit `P @ V` interval |
| Output | Local GM/HBM output DMA and ack | device output writeback interval |
| Control | manager dispatch、worker wait、barrier | launch/dispatch only in Scope B |

SST 侧沿 slowest-worker critical path 输出非重叠区间，并分别保留每 worker 的累计
工作量；不能把 16 个并行 worker 的累计 ticks 直接相加后与端到端 latency 比较。
阶段覆盖率目标为 critical-path interval 的 95% 以上，同时维持既有 lifecycle
conservation 检查。

GPU FP32 分解应在同一 CUDA stream 中按 `QK -> scale -> Softmax -> PV -> output`
连续记录 Event，只在末尾同步一次。必须同时记录同一次迭代的端到端 Event，报告
阶段和与端到端之差。单独运行的 isolated microbenchmark 只能用于吞吐上界，不能
把各自中位数相加冒充端到端 latency。

GPU FlashAttention 仍只报告 fused Scope A/B/C。在 CUPTI 或 Nsight 能提供可靠
kernel 证据前，不对其虚构 QK/Softmax/PV 阶段比例。

## 5. Counterfactual Upper-Bound Experiments

完成归因后，增加仅用于分析的理想化开关。每次先做单变量，再做组合 lower-bound；
这些结果不得替换正式 baseline：

1. ideal control：manager/worker dispatch 和 barrier 延迟归零；
2. ideal HBM DMA：保留字节数和依赖，只把传输/排队延迟归零；
3. ideal Local GM：保留访问次数，只把 Local GM 等待归零；
4. ideal matrix program：QK/PV 阵列 programming 延迟归零；
5. ideal SFU：scale/Softmax 延迟归零；
6. ideal array compute：阵列计算延迟归零；
7. ideal array parallelism：移除当前实现中可证明为人工串行的阵列 program/execute；
8. ideal residency：假设容量足够时，K/V tile 跨 query block 驻留；
9. ideal combined：组合以上开关，形成当前拓扑的乐观 lower-bound。

每个开关必须保持数值结果、操作数量和依赖顺序可验证，不能通过跳过有效计算获得
假收益。若 ideal combined 仍无法达到 GPU FP32 Scope A，则停止局部软件优化，进入
阵列数量、互连、存储带宽、SFU 吞吐和频率的资源级重设计。

## 6. Dataflow Redesign Basis

数据流重构不是无依据地增加机制。已有研究给出的方向包括：

- [FlashAttention](https://arxiv.org/abs/2205.14135)：以 IO-aware tiling 减少
  HBM 访问，并避免物化完整 attention matrix；
- [Online normalizer calculation](https://arxiv.org/abs/1805.02867)：在线维护
  Softmax 归一化状态；
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)：改进单 head 内的 work
  partitioning、并行度和非矩阵操作开销；
- [FLAT](https://arxiv.org/abs/2107.06419)：在空间加速器上联合考虑 Attention 的
  fusion、loop tiling 和数据移动。

当前项目已经实现 query/key blocking、online Softmax、不向 HBM 物化完整 score/
probability matrix，以及 query-group 并行。这些是已验证事实。

下列内容是需要用阶段数据和理想化实验验证的项目假设，不应提前写成既定收益：

- K/V tile 跨多个 query block 驻留能否显著减少 HBM/Local GM 和 matrix program；
- QK 与 PV 使用独立阵列组或双 bank，能否消除反复重编程的关键路径；
- matrix programming、SFU 和另一个 tile 的 compute 是否能够形成真实流水；
- descriptor-driven autonomous worker 是否能降低 CPU/RoCC 控制空洞；
- Local GM 容量、端口、NoC 和 HBM 带宽能否支撑上述并发而不把瓶颈转移。

早期 V tile 实验只证明无界 `vPayload` staging 有潜力，不能作为硬件收益。当前实现
已用有界 16 KiB/tagged buffer 替代该假设：fill 经过 Local-GM 路径，2 KiB panel hit
按 64 B/cycle 加一级基础延迟占用时间，并统计容量拒绝与等待。Q1024 同二进制消融从
1,290,536 降到 1,163,611 cycles（9.84%），因此该机制现已进入默认回归。

## 7. Ordered Development Plan

### Phase A: Measurement Contract

- 冻结 E3/E4 FP32 Scope A 为主比较；FP16/BF16 单列；
- 审计 SST 时钟和 timebase，统一输出 ps、ms、normalized cycles；
- 为外部 GPU 结果定义可导入的 JSON schema，保留 GPU、软件栈、warmup、iterations、
  raw samples 和统计量。

完成门禁：同一份结果可从原始时间重新计算，且不会把 normalized cycles 误解为
GPU 或 SST 子组件的原生周期。

### Phase B: Dual-Sided Breakdown

- SST 输出 end-to-end critical path、阶段区间、累计工作量和未归因区间；
- GPU 环境补测 FP32 连续阶段 Event，并继续保留 fused FP16/BF16 end-to-end；
- 生成同一张 E3/E4 阶段对照表，定位绝对 gap 最大的 1 至 2 个阶段。

完成门禁：SST critical path 归因覆盖率至少 95%，阶段顺序和 conservation PASS；
GPU 分段与同轮端到端差异有明确解释。

### Phase C: Counterfactual Bound

- 实现并运行第 5 节的单变量和组合理想化开关；
- 只在 E3 先筛选，结论明确后再运行昂贵的 E4；
- 给出当前拓扑的 lower-bound 和每类开销的最大可恢复 latency。

决策门禁：若 lower-bound 不能超过 GPU，则直接进入资源/拓扑重设计；若能超过，
只实现贡献最大的真实机制。

### Phase D: Remove Artificial Serialization

- 优先移除统计证明存在的 array programming、SFU、DMA 和 worker 控制串行；
- 每次只引入一个机制，保持 E3/E4 数值、lifecycle 和 MPI rank 一致性；
- MPI 仅用于更快完成仿真回归，不计入 simulated latency 收益。

### Phase E: Evidence-Driven Dataflow

- 根据 Phase B/C 结果选择 K/V residency、QK/PV array separation、tile overlap 或
  autonomous scheduling；
- 对新增 buffer 明确容量、地址映射、端口、带宽、排队、tag 和 replacement；
- 禁止用无界 host vector 或零延迟 callback 代表硬件存储。

### Phase F: Resource Balance

- 扫描 array count/shape、Local GM 容量与端口、NoC/HBM 带宽、SFU 吞吐和频率；
- 同时报告 latency、资源增量和带宽利用率，避免用无限资源获得不可实现结果；
- E3 用于迭代，E4 用于扩展性和最终确认。

### Phase G: GPU Gate

- 首先达到 E3 `< 97,568` 和 E4 `< 345,984` normalized cycles；
- 随后争取 E3 `< 78,000`、E4 `< 276,000` 的工程裕量；
- 使用至少三次 SST 可重复运行与 GPU raw samples 重新计算结论。

只有通过同精度 GPU Gate 后，才进入多头 Attention。多头阶段先复用单头数据流并
验证 head scheduling 和共享资源竞争；MoE 继续作为更后的独立架构阶段。

## 8. Explicitly Deferred Work

以下工作当前不进入直线开发路径：

- tile-level MPI；现有 rank 工作量已基本均衡，它只可能改善主机仿真 wall time；
- 更复杂的 Local GM port model；1 到 2 read ports 的 E3/E4 收益均约 0.1%；
- 更大容量或多端口 V-tile 扫描；当前默认固定为有界单 tile 模型；
- E5 长时间压力测试；
- 多头 Attention 和 MoE；
- 在缺少 profiler 证据时拆分 GPU fused kernel。

## 9. Immediate Next Deliverable

当前 Q2048 critical worker 中，PV matrix/input/restore/compute/output 合计约
692,659 cycles，占端到端约 36.5%；QK compute/readout 及被提前 operand 工作重新
暴露的 K/V/Local-GM 共享路径随后。operand-only cross-tile pipeline 已把 Q-local、
QK matrix 和 panel-0 input setup 大幅前移，但约 169k setup-cycle 减少只有约 51k
转化为端到端收益，因此继续增加第三 operand bank 或更深 lookahead 不合理。

下一交付应首先分解并减少 PV matrix programming、O restore 和 output read/write
跨 LocalGM/WCP/array boundary 的真实数据移动。若评估新的 near-array O bank，必须
明确 8 KiB/worker 级容量、FP32 量化点、online Softmax 顺序、端口竞争和消融收益。
第三 K/V buffer、NoC/LocalGM 全局扩宽以及更深 cross-tile pipeline 继续延期。
