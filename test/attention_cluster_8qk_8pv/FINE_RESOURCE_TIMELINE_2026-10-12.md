# 细粒度资源时间与批次 B 实验说明

日期：2026-10-12。

## 本次范围

补齐资源内部时间，保持算法、调度顺序、时延配置和硬件数量不变。不执行批次 B 的 oracle 或正式 2³ 矩阵。

正式复现使用 Hq=32、Hkv=8、D=64、hidden=2048、causal+RoPE、64 arrays、单 operand context、output retention、array_groups=1。为了与冻结基线严格比较，正式 `baseline` 的 input/local-weight prefetch 均为 0；另用 Hq=4、Hkv=2、hidden=256、S1024 的 reads 配置验证两个预取打开时的观测。小配置只验证追踪能力，不能与正式配置比较性能。

产物根目录：`results/fine_resource_timeline_20261012/`。正式数据使用 `formal_v3/baseline`，预取观测使用 `prefetch_validation_v3/reads`。此前调试目录不用于最终结论。

## 时间契约

每个完成请求输出一条 `[RESOURCE_INTERVAL]`，字段包括：

```text
domain resource core request tag generation
submit_tick start_tick done_tick visible_tick
bytes queue_depth address client traffic ok
```

请求 ID 的唯一性范围为 `(domain, resource, core, request)`。`tag/generation` 保留本层协议身份；不同层未共享同一个 tag 时，不能仅凭 request 数字相同拼接依赖。

```text
提交 submit ──排队── 实际开始 start ──服务── 完成 done ──可见/消费等待── visible
```

SST core ticks 通过 `[RESOURCE_TIMEBASE] ticks_per_cycle` 转为 1 GHz cycles，本次为 1000 ticks/cycle。资源间可以并行、嵌套；累计服务量不能直接相加成 E2E。忙碌和重叠采用区间并集/交集。Projection 重叠只统计各 worker 的 Projection start/end 窗口，排除后续 Attention 的同资源使用。

`visible` 的含义随资源边界而明确区分：DMA landing 是 GM 数据可用；WCP compute 是最后一个消费者回调；snapshot residence 是数据已 ready 后实际消费。相同的四个字段不意味着所有资源都存在独立的第四段等待。

同样，汇总中的 queue 是 `start-submit`，须按资源解释：snapshot 的这一段是生产者数据尚未 ready 的等待，WCP 包含固定 command latency，HBM admission 包含 backend 重试；它们不能全部称为端口争用。`*_snapshot_residence` 的服务区间长度为零，真实读服务在对应 GM/DMA 记录中，ready→consume 才是预取数据驻留时间。

## 已覆盖的资源边界

| 记录 | 开始/完成的实际含义 |
|---|---|
| `gm.local_read/local_write` | SRAM 请求入队、端口实际发起、服务完成/回调 |
| `gm.dma_read_transport` | burst 提交、实际首次 NIC send、读响应到达 GM；包含 NoC 与 HBM |
| `gm.dma_read_landing` | 响应到达、SRAM 落地、cache 可见；与 transport 用 request/tag/generation 配对 |
| `gm.dma_write_transport` | payload 提交、实际 NIC send、写 ACK |
| `hbm.backend_admission` | backend 第一次尝试到实际接纳，包含 admission backpressure |
| `hbm.dram_read/dram_write` | 接纳、Ramulator 第一个 DRAM 命令、backend completion callback |
| `hbm.no_dram_command` | 没有实际 DRAM 命令的转发/合并请求，不伪装成普通 DRAM 服务 |
| `array.matrix_broadcast/input_scatter/output_scatter_gather/...` | buffer 请求入队、实际调度、真实完成回调 |
| `array.compute` | 同一 tick 启动的一组阵列到最后一个物理 MAC 完成；同时启动的 64 个阵列不重复记 64 次墙钟时间 |
| `wcp.command_queue` | 接受命令到下游实际接受，含配置的 command latency |
| `wcp.array_compute` | 命令提交、实际 launch、物理计算完成、消费者回调；与 array.compute 嵌套 |
| `projection.*_snapshot_residence` | 预取提交、数据 ready、实际 consume，区分“已读完”与“仍在等待消费” |
| `sfu.*` | scale、max、exp、sum、online 更新、reciprocal、normalize 的硬件提交、实际开始、完成回调 |
| `sfu_model.*` | Norm/RoPE vector 和 score/P FIFO 的现有预约模型时间 |
| `cbuffer_model.*` | C-buffer 现有预约模型的排队、传输、ready 时间；当前正式配置不使用该可选路径 |
| `CONTROL_EVENT` | worker release issue、首次 observed、Projection accept 三个独立时刻 |

HBM 的第一个命令可以是 ACT/PRE 等，并非只记录 RD。写请求的 backend ACK 可能发生在 WR 命令发出时，出现 `start==done` 是模型的完成语义，不表示 DRAM 写入物理耗时为零。GM transport 包含 HBM，二者不能相加。Norm/RoPE 和 FIFO 的 `sfu_model` 是模拟器原有的预约机制，不冒充独立硬件完成回调。

最终 O 写回按通过验证的 HBM layout 地址范围识别，因为当前 Attention 输出 DMA 的 traffic kind 为 Unknown。归属在离线分析中完成，未改变调度器使用的请求类型。报告同时记录最后 O ACK、worker complete 和 root tensor complete，保持不同结束边界。

## 分析器与校验

[`analyze_resource_timeline.py`](analyze_resource_timeline.py) 检查：时间戳顺序、请求身份重复、DMA 到达/落地缺失或跨 generation、16 个 worker 的控制路径、timebase、主要资源覆盖、最后 O ACK 与 root 完成顺序。输出分资源和分 core 的排队、服务、可见等待，以及 Projection 的读取/编程/scatter 与 MAC 实际重叠量。

[`merge_mpi_rank_logs.py`](merge_mpi_rank_logs.py) 改为流式合并，保留 rank 原始日志，避免把多 GiB 资源记录一次性读入内存。资源 sidecar 的历史后缀为 `.resources.jsonl`，实际内容为上述 key=value trace 行；使用本分析器读取，不能当作标准 JSONL。

复现：

```bash
python3 baseline/attention_cluster_8qk_8pv/run_projection_overlap.py \
  --stage formal --variants baseline --critical-trace \
  --artifact-root results/fine_resource_timeline_20261012/formal_v3 \
  --reference-root results/retired_ws_final_verified

python3 baseline/attention_cluster_8qk_8pv/analyze_resource_timeline.py \
  --artifact-root results/fine_resource_timeline_20261012/formal_v3/baseline \
  --output results/fine_resource_timeline_20261012/resource_timeline_analysis.json
```

运行器冻结输入、guest、源码、配置、MPI 日志以及 `libgolem.so/libmemHierarchy.so` 的身份，验证原始 HBM/数值结果，并与参考输出比较。构建时必须先把修改的源码同步到 `build/sst-elements` 的复制树，再 make/install；仅在旧复制树中 make 不会编译仓库中的新埋点。

当前分析器仍是“资源区间校验 + phase/control 链重建”，不是包含全部跨层 producer/consumer、资源争用边和 slack 的完整反事实 DAG。不能把资源忙碌时间直接称为关键路径贡献；批次 B 的 oracle 必须补齐所需依赖或在模拟器中受控运行。

HBM 统计覆盖整个仿真，包括 guest 初始化/CPU 访问；不是所有 HBM 请求都属于被报告的 E2E 窗口。HBM 的 core 字段是 backend/node ID，其他资源的 core 字段通常是 worker ID。跨节点/跨 worker 的累计请求服务量不能解释为单个 worker 的忙碌时间。

## 已验证的具体观测

正式两组仿真均通过数值、HBM layout 和参考输出逐字节对照，且 E2E 没有变化：

| S | E2E / 重建 E2E | Projection | 资源区间数 | DMA 到达/落地配对数 |
|---:|---:|---:|---:|---:|
| 1024 | 573,772 / 573,772 | 416,538 | 24,704,286 | 38,802 |
| 2048 | 1,354,890 / 1,354,890 | 801,164 | 71,106,341 | 84,009 |

正式 HBM byte-exact 分别验证 16 个区域、10,485,760 / 20,971,520 bytes。基线目录 Python 测试 39 项、Attention 目录 140 项均通过；新增测试覆盖时间边界、重复/缺失/乱序、跨 generation、输出地址归属、区间并集及 Projection 窗口裁剪。Golem 与 memHierarchy 编译安装通过，Python 编译与 `git diff --check` 通过。

最终汇总：

- [完整资源汇总](../../results/fine_resource_timeline_20261012/resource_timeline_analysis.json)
- [E2E 重建汇总](../../results/fine_resource_timeline_20261012/critical_path_analysis.json)
- [可审阅的精简结果](fine_resource_timeline_results.json)
- [构建与验证日志](../../results/fine_resource_timeline_20261012/validation_logs/)

S1024 正式 baseline 的最后 Projection worker 为 core 16，Projection job 窗口为 385,776 cycles。该窗口内的实际忙碌区间并集为：

| 资源 | 忙碌 cycles | 与 MAC 的重叠 cycles |
|---|---:|---:|
| MAC | 101,376 | — |
| GM local read | 105,808 | 0 |
| matrix program | 59,904 | 0 |
| input scatter | 50,688 | 0 |

S2048 最后 Projection worker 为 core 18，job 窗口 770,901 cycles，MAC/GM-read/matrix/scatter 的实际忙碌时间分别为 202,752 / 211,616 / 119,808 / 101,376 cycles；三种资源与 MAC 的重叠也均为 0。两组中所有 16 个 Projection worker 均观察到零重叠。

这确认当前冻结 baseline 的读取、编程、输入搬运、MAC 仍是串行；但不能据此把所有读/编程耗时都当作可获得的软件收益。实现重叠可能引入端口争用、增加广播或减少同时参与计算的阵列。

单次物理 matrix broadcast 为 39 cycles，input scatter 为 33 cycles，MAC 为 66 cycles；此前软件 phase 中对应 41/35/68 cycles。新记录能区分资源服务与软件发起/完成观察的附加时间。

S1024 release issue 跨度为 31,208 cycles，accept 跨度为 30,701 cycles；信号 issue→observed 仅 9–25 cycles，observed→accept 为 40,132–40,933 cycles。因此启动不齐主要与 release 发出节奏和后续 guest 准备有关，不能归因为 release 信号传输慢。原 E2E 从最早 Projection job start 计时，这约 40k 的共同前置准备并非全部落在 E2E 内；不能直接宣称优化它能减少同样多的原报告 E2E。

较小 heads 的 reads 观测通过 1,862,852 条区间、1,121 对 DMA 配对和 16 个 worker 控制路径校验；记录到 363 个 weight snapshot、128 个 input snapshot 的 ready/consume residence。每个 worker 的 Projection MAC 为 2,112 cycles，GM-read 与 MAC 重叠 528–629 cycles；matrix/scatter 与 MAC 重叠仍为 0。这表明本地读取预取确实产生了重叠，也说明它未让全部阵列操作流水化。不能外推这一小配置的收益到正式配置。

正式 S1024 最后 O ACK 为绝对时间 684,033.25 cycles，root tensor complete 为 684,057.00，相差 23.75 cycles；报告最后 worker complete 为 684,045、chain start 为 110,273，原 E2E 为 573,772。三个完成时刻保留各自定义，不通过修改 E2E 终点来“制造收益”。

S2048 最后 O ACK 为 1,465,147.75 cycles，root tensor complete 为 1,465,171.50，相差也为 23.75 cycles。两组识别的最终 O 写入字节分别为 4,194,304 / 8,388,608，与各自四个节点 O 区域总大小相符。

## B 批次的 oracle 是什么

它是定位瓶颈的理想化实验：每次只让一个环节变得理想，其余流程继续执行，观察最终 E2E 是否下降以及瓶颈是否转移。

例如“阵列编程延迟为 0”：仍然按原来顺序提供正确矩阵，保留读输入、计算、输出及资源身份，只在诊断模型中免去编程等待。若最终几乎没有变快，优化编程很难获得大收益；若明显变快，再寻找现有硬件上的编程/计算重叠方案。oracle 本身不是可部署优化，也不是增加硬件的建议。

| Oracle | 要回答的问题 |
|---|---|
| GM operand read = 0 | 本地读取若完全被隐藏，最终还能快多少？ |
| weight arrival = instant | 权重不再晚到，是否解除关键等待？ |
| matrix program = 0 | 阵列编程若不暴露等待，是否缩短最终完成？ |
| input scatter = 0 | 输入搬运是否限制完成时间？ |
| worker start skew = 0 | worker 若统一启动，慢 worker 是否明显提前结束？ |
| Projection→Attention barrier = 0 | 各 ready unit 提前被消费，是否能缩短最后输出时间？ |
| Attention internal barrier = 0 | 去掉纯调度等待后，还剩多少收益空间？ |

屏障 oracle 只能去掉可消除的调度等待，必须保留真实数据依赖、generation 和容量限制：Q/K/V 未 ready 时不能凭空计算。trace/replay 需要重新计算下游开始时间及资源争用，不能从旧 E2E 简单减去某个 phase 总量。各 oracle 的收益不能相加，因为去掉一个瓶颈后关键路径可能改变。

这里得到的是“在指定理想条件下的潜在收益”，不是已经证明的软件优化收益，更不是整个系统的全局最优下界。比较 E2E 时沿用冻结报告的结束边界，同时单列最终 O ACK 和 root complete。

## B 批次的正交实验是什么

这里具体指 3 个二值因素的完整 2³ 因子实验，共 8 组。因素为 input prefetch、local weight prefetch、array_groups=1/2。每组都在 S1024/S2048、相同输入、相同硬件上独立运行、归档和验证。

| 组 | Input prefetch | Local weight prefetch | 阵列分组 |
|---|---:|---:|---|
| B | 0 | 0 | 64 |
| I | 1 | 0 | 64 |
| W | 0 | 1 | 64 |
| IW | 1 | 1 | 64 |
| G | 0 | 0 | 32+32 |
| IG | 1 | 0 | 32+32 |
| WG | 0 | 1 | 32+32 |
| IWG | 1 | 1 | 32+32 |

32+32 是对已有 64 个阵列进行调度分组，硬件数量不增加。

完整组合用于区分“谁有收益”和“谁与谁争用”。例如输入预取单独节省 10、权重预取单独节省 8，组合只节省 5，则它们可能互相争用 GM 端口或队列。数字仅用于解释，实际判断必须结合资源记录。固定 G 时的二阶差分：

```text
interaction(I,W) = E(IW) - E(I) - E(W) + E(B)
```

E 越小越好，因此正值表示组合收益弱于两者独立收益之和，负值表示协同。分别在 G=1 和 G=2 中计算，并检查三因素交互。单变量差值用于主效应，不能拿几个不完整、不同配置的历史实验替代这个矩阵。

oracle 用来决定优先研究哪个瓶颈；正交矩阵用来判断真实软件优化的独立收益和组合关系。本报告对应的观测批次只完成资源时间与基线验证。后续 B 批次已完成 30 次正式实验，具体 oracle 范围、完整矩阵及结果见 [BATCH_B_2026-10-12.md](BATCH_B_2026-10-12.md)；完整跨资源 DAG/slack 仍未完成。
