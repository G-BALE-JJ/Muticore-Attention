# 舍弃 weight-stationary 后的数据流重分析

> 后续实测更新：下一input pair与本地weight预取已实现，正式E2E下降5.03%/3.75%；现有64阵列两组错开也已实现，但没有额外净收益。以 [完整流水实验报告](PROJECTION_READ_PROGRAM_COMPUTE_OVERLAP_2026-10-11.md) 为最新状态，下文保留实施前归因和候选排序。

日期：2026-10-11。按用户决定，撤销 32-array weight-stationary 原型，不再修补或扩展该路线。
它的状态机和专属统计公式已移除；旧开关设为 1 会被拒绝。既有实验目录保留用于追溯。
本文替代旧 `LATENCY_GAP_ANALYSIS_2026-10-11.md` 的优化排序。

## 1. 先纠正归因和实验口径

- 正式 64-array 基线从未开启该 WS 原型。因此舍弃 WS 是停止一条无收益的实验路线，不能直接消除正式基线的原始差距。
- 当前 64-array 数据流已经让每个阵列在完整 hidden dot product 中保留输出，可视为输出驻留。真正未完成的是跨 GM、阵列搬运、MAC、SFU 和 Attention 的流水调度。
- WS 的严格隔离实验使用 Hq/Hkv=4/2、hidden=256，并非正式 32/8、hidden=2048。固定 pipeline=0 等条件时，S1024 的 E2E 为 91,109→91,828，S2048 为 201,067→201,755。
- 之前 77,966→79,911 的比较还改变了 pipeline 等条件，不能作为 WS 单变量性能证据。
- 旧“六步执行记录”中步骤 2–6 是代码检查，未完成实现和受控 A/B；不能据此说这些优化已经证明无效。
- Norm 输入布局确实影响 WS 原型正确性，但不能把数值错误直接归因于 WS 思想。静态检查还发现：原型奇数 tile 重读时，非 packed 地址没有向偶数 pair 对齐；packed 地址的 `tile / 2` 会掩盖这一问题。该线索未单独修复验证，随原型一起退役。

## 2. 当前差距可以精确对账

正式形状：S1024/S2048，Hq32/Hkv8/D64、hidden2048、causal+RoPE。资源：4 个 manager 控制，16 个 worker，每 worker 原有 64 arrays、SFU、2 MiB GM，单 operand context、单 output accumulator，端口及带宽不变。

| 指标（cycles） | S1024 | S2048 |
|---|---:|---:|
| Projection（含 Norm） | 416,538 | 801,164 |
| Projection→Attention 交接 | 2,019 | 3,794 |
| Attention | 155,215 | 549,932 |
| E2E | 573,772 | 1,354,890 |
| 原计算下界参考 | 268,592 | 799,328 |
| 差距 | 305,180 | 555,562 |

下表按照最后完成的 worker（S1024 core16、S2048 core18）的互斥状态时间拆分，再加阶段交接和 Attention 超过 exp 的时间。最后完成以 `local_wait` 为准，不能只挑最后一个 `PROJECTION_JOB end`。
融合配置的Projection span已经包含Norm，不能再把report中的RMSNorm span加一次。其`rmsnorm_to_projection`为负，是最早ProjectionJob开始与所有worker/blocks最后Norm结束的聚合时间差，不证明同一worker的Norm producer与MAC已经有效重叠。

| 原计算下界没有计入的时间 | S1024 | S2048 |
|---|---:|---:|
| 关键 worker 相对最早 worker 的启动差 | 30,701 | 30,190 |
| 前台 Local GM read | 112,032 | 224,064 |
| matrix program + submit | 64,512 | 129,024 |
| input scatter + prepare | 55,296 | 110,592 |
| launch 超过 MAC 工作参考的开销 | 3,072 | 6,144 |
| 最终 output read | 1,680 | 3,360 |
| 其他 Projection 状态、completion/guest 等待 | 19,917 | 39,134 |
| 阶段交接 | 2,019 | 3,794 |
| Attention 超过 exp 工作参考 | 15,951 | 9,260 |
| 合计 | **305,180** | **555,562** |

复现：`python3 baseline/attention_cluster_8qk_8pv/analyze_projection_latency.py --artifact-root results/retired_ws_final_verified --output baseline/attention_cluster_8qk_8pv/latency_reanalysis_without_ws.json`。
脚本检查 Projection 分解和 E2E 差距均精确相加；结果 JSON 保留原始 input/pipeline 统计。
这些是当前时间分解，不能当成彼此独立、可相加的优化收益。
Projection超出Norm+MAC参考的时间分别为287,210和542,508 cycles，占原E2E差距的94.1%和97.7%。因此当前应优先处理Projection数据流，不能仅凭Attention接近exp参考就断言其QK/PV/通信完全没有优化空间。

原“理论下界”仅为当前分阶段执行的计算工作参考，不含搬运，也不是未来跨阶段重叠调度的严格下界。当前全组同步路径中，每个 tile 的 8 KiB matrix broadcast 约需 39 cycles、8 KiB input scatter 约需 33 cycles，MAC 需 66 cycles。只计算 MAC 会漏掉大量必需服务。若保持全组串行顺序，即使 GM 读完全隐藏，每 tile 也约需 39+33+66=138 cycles，另有输出和控制时间。这是固定调度的服务参考，不能用于证明所有合法软件调度都无法更快。

## 3. 已确认的软件串行点

### 3.0 主状态机将独立资源串成同一条等待链

`ProjectionJob::tick()`虽然先推进有限后台流程，但前台由单一`phase_`、`pending_`控制。`beginLocalRead()`把主phase切为LocalRead；Norm也通过同一phase进入NormBatch/NormWait，回调完成才交还主流程。weight HBM预取和future input有独立状态，Norm和前台GM读取却尚未解耦。
因此“pipeline enabled”不等于已经覆盖完整数据流。重分析的共同方向是：在原有有界槽、队列和端口上，把数据准备、SFU producer、阵列consumer分别维护ready/generation和资源占用，再允许互不冲突的工作推进。不能用一个更大的全局pending锁替代这些依赖，也不能让不同flow同时改写同一foreground callback的数据。
当前S1024关键worker真正MAC工作101,376 cycles，只占其job时间约26.3%；这是整组同步与前台搬运暴露的直接表现。

### 3.1 有 input 预取槽，但 64-array 路径不使用它

`ProjectionJob::Launch` 的 future input 逻辑只处理下一行组，或 `rowGroup()<64` 时的下一对 tile。64 arrays 恰好覆盖 worker 的 64 行，两个条件都不成立。
正式日志虽然 `pipeline enabled=1`，但 `input_prefetches=0`、`background_input_read_requests=0`。
因此下一对 hidden tile 的输入始终到消费时才读取。

现有 `prefetchInput_` 足以容纳 64×256=16 KiB 的下一 input pair，且现有 `stagingBytes()` 已计入容量。这一步无需第二个阵列 context，也无需新增硬件。
消费顺序是：当前 input pair 留在 `inputRaw_`，两次 MAC 分别使用偶/奇 tile；future slot 保存下一 pair，直到当前 pair 完成后才交付。不能提早覆盖当前 inputRaw 或 array input。
输出 retention 路径没有 partial，扩展预取时必须跳过 partial-prefetch，不能照搬 32-array 的 restore 行为。

### 3.2 HBM→GM 预取不等于 GM→operand 数据准备已经重叠

`ProgramWeight` 仍先同步进行 GM→weightRaw 读取，完成后才广播；packed input 同样走前台 GM read。
`pumpLocalAccess()` 每个 4096-byte chunk 等 callback 后才发下一块：每份 8 KiB weight 两块，每份 16 KiB input pair 四块。
S1024 每 worker 的 weight/input 前台读时间各为 55,296 cycles；总 operand bytes 为 24 MiB，S2048 为 48 MiB。

已有后台 `AsyncRead` 可以在队列容量内提交多个块；这是同一端口上的排队，不是增加端口或服务带宽。优化目标是减少发请求的空档，并让请求与 MAC/array-buffer 服务重叠，实际 bytes 和端口服务仍必须计费。
64-array retention 不使用未来partial槽，可评估将原预留的8 KiB partial快照改为下一weight快照：future input16 KiB+future weight8 KiB+output8 KiB仍在原32 KiB staging预算内。实现时必须核对当前weightRaw、callback payload与槽的生命周期，不能把模拟器host vector视为免费硬件存储。
此外 `pumpWeightPrefetch()` 的范围限定同一 head，最后一个 head tile 后不会预取下一 head 的第一个 weight。关键 worker 的 `load_weight` 仍约占 13.6k/26.4k cycles。这是另一个可用已有 cache slot 测试的边界空档，不能用“预取命中率高”断言 HBM 等待全被隐藏。

### 3.3 阵列全组同步造成搬运与 MAC 交替闲置

当前 64 arrays 一起广播、一起 scatter、一起 MAC；MAC 时不允许覆盖这些阵列的唯一 operand context。这是同一阵列的正确性约束，不能扩大解释成“所有其他阵列也不能准备数据”。
现有 WCP matrix/input/launch API 已接受 array-ID 子集，completion 也按阵列跟踪。可以研究原有 64 arrays 分成固定两组32：各自始终负责32行并保留输出，A组 MAC 时只编程 B组，反之亦然。两组使用不同物理阵列的既有 context，不增加第二份 context，也不让同一 accumulator 在不同 row group 间切换，因此不需要 WS 原型的逐 tile partial spill。

代价不能忽略：一份 matrix 要广播两次，搬运端口占用增加。简化服务模型中，每半组准备约为 matrix38+scatter17=55 cycles，MAC66 cycles；两个固定半组可错开，存在比整组串行138 cycles更短的稳态周期，但这不是实测收益。
分析脚本给出的单head、32 hidden tile示例为：整组4416 cycles，两固定半组3927 cycles，同时端口服务量从2304增加到3520 cycles。模型忽略GM、output、WCP命令延迟和跨head边界，只用于判断是否值得实现实验，不能作为E2E收益报告。
不建议直接分成四组16：每组matrix37+scatter9=46 cycles，四组每hidden tile端口服务就有184 cycles，已经超过整组的138 cycles编程+MAC服务参考。增加分组数量不能无限提高并行收益。
实现前要核对 FIFO 顺序、每组 generation、MAC 完成依赖、GM 快照容量，以及当前 `ProjectionOutput` 校验只接受从 array0 开始的前缀集合这一软件限制。允许合法非连续子集不改变硬件；必须保持互斥写入和输出顺序验证。

### 3.4 Norm 和 Projection 没有并发 producer/consumer

融合当前主要省掉 HBM 往返：同一 block 先完成该 worker 的64行 Norm，再投影所有 heads。Projection 期间 SFU 空闲；S2048 下一 block 的 Norm 没有提前准备。
当前 GM 输入布局预留256行，而一个 worker只拥有其中64行。下一 block 可评估使用未消费的其他行区，明确区分逻辑行号、物理槽和 generation，全部 raw read、gamma read、packed write仍走当前 GM 服务。
因此不能把“没有第二个软件槽”当成需要增加 GM 的证据。S1024只有一个本地 block，跨 block overlap没有收益；块内微流水需单独建模，每个Norm batch仍跨完整hidden维度，不能把Norm错误拆成独立hidden tile归一化。

### 3.5 启动和阶段屏障仍值得单独验证

同一 manager 的 lane0..3 进入投影相差约10k cycles。guest release循环在每次 `remote_store` 后执行一次 `fprintf`，下一worker的release必须等前一次guest打印路径完成。
这是具体的软件串行点；它能否解释30k启动差仍需release issue、worker observed、job accept三个时间戳和受控实验。不能归因于未被该guest调用的 `WORKER_START_GUARD`，也不能把“release在descriptor之前”当成release本身已并行的证据。

Projection目前按 Q全体→K全体→V全体遍历；manager等待全部16worker完成才提交Attention。
删除约2k/3.8k交接指令只能省有限时间。更大机会是按KV head将相关K、V、四个Q一起产出，并在依赖数据ready且worker资源释放时执行Attention。
S2048跨sequence block时，一个head的完整K/V还依赖各block、各HBM节点；必须把遍历顺序和ready单位一起设计，不能只换head顺序。
Projection和Attention共用worker/array/SFU/C-buffer，二者不可能在同一资源上免费并行。应比较分时或暂时分配已有worker的策略，保留最终fence；若Projection因此变慢，必须看E2E净收益。

## 4. 新的实验次序和验收

| 顺序 | 单变量实验 | 原有资源上的操作 | 验收重点 |
|---|---|---|---|
| 1 | 64-array同一行组下一input pair预取 | 启用已容量计费的future slot；保留output | 理论有768×15/16=720次每worker预取（S1024；S2048翻倍）；input总bytes/请求数不增加，前台wait下降，E2E下降 |
| 2 | GM读取流水，再分别测试weight本地预取、跨head首weight预取 | 固定4096请求上限/1读端口/queue；使用已有快照和cache | 分别比较前台暴露时间、实际GM服务、queue/retry和E2E；不可一次合并三个改动 |
| 3 | worker release集中提交 | 先release四worker，再执行guest日志；补三段时间戳 | 同一manager各lane启动差缩小，完成flag及数值不变；不修改硬件指令延迟 |
| 4 | 现有64阵列两组错开 | 固定每组32行、每阵列输出驻留；共享既有搬运端口 | 先成本模型，后A/B；新增broadcast代价计费，partial仍零，单context不变 |
| 5 | S2048跨block Norm→Projection | 原GM空闲行区作为有界软件槽，SFU后台producer | 记录Norm ready/consumer stall/GM竞争；S1024单独测试块内方案 |
| 6 | 按KV head Projection→Attention overlap | 重排producer顺序，ready+ownership控制已有worker | 首个Attention实际提前，HBM各节点完整，资源占用不冲突，E2E净下降 |

步骤1的720次来自48输出heads×(16个input pair−1)，不是新增输入读；首pair仍前台读取。
命令callback合并优先级降低：当前matrix/input服务39/33与阶段42/36已较接近，最多压缩几cycles/命令，无法消除必需服务。
GM→Attention直接消费者路径暂留为后续问题，只有HBM中间往返被证明影响关键路径才投入，避免为了局部减少bytes占满原有GM。

所有实验固定输入seed/hash、源码/二进制hash、正式形状、Norm/audit/RoPE配置、operand banks、端口/带宽与cache容量；保留实验前后JSON和完整日志。
先小形状生命周期/慢GM回归，再跑正式S1024/S2048。必须通过NumPy O、HBM byte-exact、资源拓扑和请求/bytes对账。
某个wait下降而E2E无收益时应记录“该实验未改善E2E”，不能升级成“整个软件流水方向无效”。

## 5. 本次实现与验证记录

本节记录上一轮退役WS时的验证：当时只恢复原paired遍历和报告口径、加入旧开关拒绝测试，并生成重分析脚本，上表新性能方案尚未实施。后续实施状态见本文顶部的流水实验报告。
重新构建安装通过；136项Attention单元测试、19项baseline测试通过，7项资源归属/多block/32-array/慢GM等回归通过（`results/retired_ws_regressions/summary.json`）。
S1024/S2048正式retention复测均通过，E2E分别573,772/1,354,890 cycles，Projection分别416,538/801,164 cycles，与原参考一致。每个case的16个HBM区域分别共10/20 MiB逐字节一致，O最大绝对误差均0.0001220703125，单context和资源归属检查通过。
完整phase分解与旧基线一致，归档的source/binary/input SHA256在复测后重新核验通过。正式结果在`results/retired_ws_final_verified/summary.json`，构建/单元测试日志在其`validation_logs/`。
第一次retention仿真因未启用逐MPI rank日志，控制台交错造成report漏读2条QK完成记录，不能记为验证通过；其阶段时间与旧基线相同，失败artifacts保留。上述正式复测使用逐rank日志及原资源/输入配置。
