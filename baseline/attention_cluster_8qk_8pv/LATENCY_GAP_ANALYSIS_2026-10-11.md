# 端到端延迟差距分析与软件数据流路线

> 本文已被 [舍弃WS后的重分析](LATENCY_REANALYSIS_WITHOUT_WEIGHT_STATIONARY_2026-10-11.md) 替代。下文保留历史分析，其中启动guard归因、64-array future slot没有消费对象、WS非单变量性能比较均已纠正；不再执行WS优先路线。

本文只使用当前合法资源拓扑的结果：4 个 manager 只控制，16 个 worker 各有 64 个 array，物理 array 只有一个 operand context 和一个 output accumulator。历史 manager-array 或双 operand context 结果不作为性能证据。

## 基线和下界

| S | 投影（含 RMSNorm） | 交接 | Attention | E2E | 乐观下界 | 差距 | E2E/下界 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 416,538 | 2,019 | 155,215 | 573,772 | 268,592 | 305,180 | 2.136x |
| 2048 | 801,164 | 3,794 | 549,932 | 1,354,890 | 799,328 | 555,562 | 1.695x |

下界只包含 RMSNorm 向量工作、当前 64-array tiling 的阵列 MAC 和 Attention exp issue work：

| S | RMSNorm vector | Projection array | Attention exp |
|---:|---:|---:|---:|
| 1024 | 27,952 | 101,376 | 139,264 |
| 2048 | 55,904 | 202,752 | 540,672 |

下界没有包含 GM/SRAM 读写、HBM DMA、权重编程、输入 scatter、output read/restore、RoPE、WCP/NoC、控制派发、worker 启动和阶段同步。因此它是“理想计算资源下界”，不是在当前单 context 数据流上可以直接达到的时间。

当前三个阶段没有跨阶段重叠：投影结束后 manager 通过 16 个 worker completion flags 做全局等待，随后才提交 Attention descriptor。两个长度上阶段时间都精确相加。这解释了为什么 Attention 内部已经接近 exp 下界，仍然不能隐藏投影尾部。

## 差距的根因

### 1. 单 operand context 把每个 tile 变成串行关键路径

`ProjectionJob` 的合法路径是 `ProgramWeight -> ProgramInput -> Launch -> ReadOutput`。唯一 operand context 不允许在当前 MAC 期间写入下一份 weight/input；因此 GM 预取只能把数据放入既有 GM/cache，不能隐藏 array-side programming。

当前 64-array worker 负责一个 64-row group，跨 row group 没有 weight reuse。正式结果中每个 worker 的 `weight_reuses=0`；S=1024/S=2048 全系统分别执行 24,576/49,152 次 weight load 和 program。深度 1 到 8 的 lookahead 只改变 HBM→GM 时序，不能减少矩阵 broadcast 数量，所以没有端到端收益。

关键 worker 的已验证分解（保留 output 前的合法拓扑结果）显示，S=1024/S=2048 的主要暴露项分别是：

| 项目 | S=1024 | S=2048 | 含义 |
|---|---:|---:|---|
| GM weight/input read | 110,592 | 221,184 | 当前两个 operand 流的本地 GM 服务 |
| weight program/submit | 64,512 | 129,024 | 1 个 context 下每份矩阵必须前台编程 |
| input scatter/submit | 55,296 | 110,592 | 64 个 array 的输入散布，共享 array-buffer port |
| array compute/launch | 104,448 | 208,896 | 其中真正的 MAC 下界为 101,376/202,752 |
| worker start skew | 30,701 | 30,701 | 启动 release/guard 的固定开销 |

矩阵 broadcast、input scatter 和 projection output 共用每个 worker 的单一 array-buffer port；队列调度是 FIFO。矩阵和输入请求的纯 service 已接近公式下界，剩余部分主要是命令派发和状态机等待，不能靠继续增加 lookahead 消失。

### 2. packed input 已经消除了主要小请求问题，但当前 64-array 路径仍暴露前台读

当前 formal run 已启用 packed Norm input。每个 worker S=1024 的 local GM read 为 24 MiB、读 residence 112,032 cycles；weight/input 两类角色各约 55,296 cycles，S=2048 近似翻倍。输入预取统计为 0，因为 64-array worker 没有“下一个 row group”；现有 future-input slot 对此形状没有可消费对象。继续增大 input/weight lookahead 不会改变这部分服务量，正确方向是改变消费顺序或把 Norm/GM 操作放进 array compute 的空档。

### 3. partial 已经优化，当前 64-array 的剩余 output 开销很小

output retention 已把 S=1024/S=2048 的 E2E 分别从 645,470/1,499,733 降到 573,772/1,354,890，投影节省 71,642/144,836 cycles；partial reads/restore 由 11,520/23,040 降为 0。当前 critical worker 的 output read 只有 1,680/3,360 cycles。因此 partial restore 不是当前 64-array 主瓶颈，但它是 32-array weight-stationary 能否正确实现的主要生命周期约束。

### 4. RMSNorm 的计算已融合，但没有形成跨 batch 的 producer/consumer 流水

融合 Norm 去掉了完整的中间 HBM round trip，但同一 `ProjectionJob` 仍然先完成当前块的 Norm，再进入投影。每 worker 的 Norm vector floor 为 27,952/55,904 cycles，另外还有 raw/gamma GM 读取、packed 写回和控制。S=2048 有两个 256-row block，具备跨 block 预规范化的窗口；S=1024 没有这个窗口。

### 5. Attention 本身已接近计算下界，真正的问题是阶段边界

当前 causal balance 已使 QK/SFU/PV worker 负载接近均衡，V cache 为全命中。Attention span 相对 exp floor 只多 15,951 cycles（S=1024）和 9,260 cycles（S=2048）；剩余主要是 QK DMA/WCP readiness、PV transport、RoPE 和最终 ACK。单独优化 SFU 或继续调因果负载不会消除投影阶段的等待。

## 推荐的软件优化顺序

### P0：先完成 32-array weight-stationary 正确性

这是当前最有希望减少“必须支付的 weight program”的路径，也是交接文档指定的下一步。每个 row group 必须显式维护 `current_row_group / hidden_tile / partial_write_issued / partial_write_ready / partial_restore_consumed`，并严格实现：

```text
row group 0, tile t: launch -> read -> partial write
row group 1, tile t: launch -> read -> partial write
tile t+1, row group 0: partial restore -> accumulate
tile t+1, row group 1: partial restore -> accumulate
```

验证顺序必须是：关闭 pipeline、restore-ahead、融合 Norm、packed input；先做最小 C-buffer 版本，再逐一打开优化。必须逐字节比较 Q/K/V/O，并记录 weight program、partial read/write、input scatter 和 projection span。原型的 64→32 program 计数只能作为假设，不能当成收益；32 arrays 会改变 scatter 请求粒度并引入 partial 往返，只有 E2E 变快才保留。

第 1 步隔离回归（S=1024，Hq/Hkv=4/2，D=64）已完成：开启 `GOLEM_PROJECTION_WEIGHT_STATIONARY=1`、32 arrays 和 C-buffer 时，16 个 worker 的 numerical/HBM/资源拓扑检查通过，报告显示每 worker 的 weight program 从 64 降到 32；partial store/read 从 16 组增到 48 组。由于首版没有第二个 input snapshot，输入读取从 64 个请求增到 128 个，投影 span 从 50,842 增到 52,655 cycles，E2E 从 77,966 增到 79,911 cycles，因此该路径暂不作为默认性能路径，只保留为正确性基线。尝试重新打开后台 pair prefetch 会产生最大绝对误差 0.1257，已关闭并记录为后续流水化任务的回归样例。

### P1：去掉不必要的 worker 启动空转

当前关键路径包含 30,701 cycles 的 worker start skew。可在 manager 已完成输入/HBM 初始化后更早发布现有 release flag，或让 projection worker 直接消费已准备好的描述信息；不新增同步资源。先做单变量 A/B：只改变 release 时机，保持所有 payload 和输出不变。验收条件是 completion flag、资源归属和数值完全不变，并且这部分 skew 实际下降。

### P1：融合每个 tile 的命令派发

对现有 WCP 接口增加软件级的“matrix broadcast + input scatter”批处理/预约，保持单 array-buffer port 和单 operand context 不变。目标是减少每个 transfer 的 callback、queue transition 和状态切换；不能把两个 transfer 当成并行 port。先记录 enqueue/start/end，再决定是否可以把派发开销压缩到当前纯 service 之上。这个方向比改带宽更符合当前证据：矩阵和 scatter 的 residence 比纯 service 多出的部分就是可疑的控制开销。

### P1：做有限的 Norm→Projection 微流水

利用数组计算期间 SFU 空闲的事实，在既有 GM staging slot 中预计算下一个 4-row Norm batch，并以 generation/ready 标记交付给 projection。只允许一个后台 Norm batch，不能覆盖当前 input/partial；所有 raw read、gamma read 和 packed write 仍走现有 GM timing。先做 S=2048 的跨 256-row block prefix，再考虑 S=1024 的 batch 内双槽。必须记录 producer ready、consumer stall、GM queue/retry；不能只看 Norm vector 时间下降。

### P2：把 GM/partial 调度改成有截止时间的后台调度

对 input、weight、partial restore/store 增加 enqueue/start/end 时间戳。后台请求只占用当前 array compute 的 slack；当下一个 `ProgramWeight/ProgramInput` 或 output drain 有明确截止时间时，暂停不紧急的 partial/Norm 写入。单一 GM read/write port 和单一 array-buffer port 不变，不能用“某个 wait 变小”替代 E2E 收益证明。

### P2：按 KV head 发布 Projection→Attention

当前 `finish_projection(manager)` 等待全部 16 个 projection jobs，之后才提交所有 Attention heads。可以把投影遍历重排为 K/V 与对应 Q heads 成组，利用已有 HBM completion 写入后发布 per-head ready flag；当某个 head 的 Q/K/V 都 ready 且其 Attention worker 已释放时，立即提交该 head。保留最终全局 fence，避免把不完整 panel 暴露给 Attention。由于 projection 和 Attention 复用同一批 worker，必须先定义 worker ownership/时间片，不能只删除全局 barrier。S=1024 预计收益受尾部长度限制，S=2048 更值得测试。

### P3：最后再研究投影输出的直接消费者路径

若 per-head overlap 证明 HBM Q/K/V 往返是关键，可评估用现有 GM/NoC staging 让 Attention 直接消费已 ready 的 panel，同时异步保留 HBM byte-exact 输出。这个方向需要重新核对 2 MiB GM scratch、panel ownership、descriptor ABI 和 worker 复用，优先级低于前面的计数减少和阶段重叠。

## 不建议优先做的方向

- 继续把 weight lookahead 从 8 扩大；它已经命中，但不触及 array-side matrix programming。
- 恢复 `GOLEM_ARRAY_OPERAND_CONTEXT_BANKS=2` 或跨 tile operand pipeline；这违反当前资源约束。
- 只调整模型带宽、array-buffer port 或 queue depth；这会改变硬件资源，且当前证据首先指向请求数、生命周期和阶段同步。
- 继续单独压缩 Attention exp/SFU；当前 Attention 已接近 exp floor，应优先和投影做跨阶段重叠。
- 默认启用 RoPE fusion；既有匹配实验显示它把 RoPE 工作移到 projection critical path 后反而变慢，除非新的 per-head streaming 能隐藏这部分代价。

## 每个实验的统一验收指标

固定输入和二进制构建方式，分别跑 S=1024、S=2048，并记录：E2E、projection span、Attention span、阶段 handoff、worker start skew、weight load/program/reuse、GM 请求和 bytes、array-buffer enqueue/start/end、partial 生命周期、SFU ready/consumer stall。每个实验必须通过独立 NumPy O、四个 HBM node 的 Q/K/V/O byte-exact、资源拓扑检查以及现有 attention/baseline tests；任何只减少某个局部 counter、但 E2E 不下降的方案都不进入默认路径。
