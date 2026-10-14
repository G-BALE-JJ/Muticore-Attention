# 去除额外阵列 operand banks

每个物理阵列现在只分配一份 matrix 和一份 input 操作数上下文。
16 个 worker 各有 64 个阵列；4 个 manager 没有阵列或 SFU。
`MVMComputeArray` 分配长度直接为 `numArrays`，不再乘 bank 数；输出
累加器仍为每阵列一份。原额外 bank 是阵列侧独立建模存储，不是 2 MiB
GM 中的软件分区。删除它不会占用或扩大 GM，也不添加端口、带宽或阵列。

配置层和模型层拒绝 `GOLEM_ARRAY_OPERAND_CONTEXT_BANKS=2`。
Attention 的跨 tile operand 编程和 QK matrix lookahead 同时禁用，不能
通过其他入口重新启用额外存储。旧 bank API 仅为接口兼容保留，合法 bank
只有 0；不构成第二份物理存储。

## 数据流影响

| 路径 | 单 context 行为 |
| --- | --- |
| HBM/GM 下一份权重或输入预取 | 保留，在既有缓存/GM 中进行 |
| 阵列权重、输入编程 | 等当前计算完成，使用唯一 context |
| 提前把下一行组输入写进阵列 | 取消；由当前行组的 ProgramInput 阶段编程 |
| hidden tile 之间输出累加 | 64-array 单行组保留输出累加器优化 |
| 多行组 partial | 保留原 GM 或 C-buffer 保存/恢复路径 |
| Attention 跨 tile operand 编程重叠 | 取消；既有 GM/KV 预取仍保留 |

单 context 的输入覆盖是关键正确性约束：偶数 tile 计算完成，并不意味着
当前行组已完成；下一奇数 tile 还会重新编程并使用相同输入 context。
因此后台仅预取 GM 数据，不直接编程未来行组的阵列输入。

32-array 路径每个 worker 有两个行组，两组交替计算偶/奇 hidden tiles。
单 context 不能同时保留两份矩阵，下一行组需要重新编程，不能沿用旧的
双 bank matrix reuse 计数。端到端报告按行组计算实际编程次数，同时保持
HBM 权重缓存加载与阵列编程次数分开核算。

## 验证范围

端到端为合成 FP16 X/gamma/weights 的 RMSNorm → Q/K/V → RoPE → causal
GQA → O 写回 HBM；不含 Wo、残差或 MLP。Hq32/Hkv8/D64、hidden2048，
S1024 和 S2048 检查全部输出和历史参考逐字节一致，并检查 NumPy 数值误差。
每次运行记录源码/二进制 hash，严格检查 16 个 worker 各有 64 份 matrix/input
context、bank 数为 1，以及 manager 无计算资源。

额外回归覆盖 3 个 sequence blocks、投影融合 RoPE、非融合 Norm、D128
Attention、32-array C-buffer partial、32-array 慢 GM partial、hidden128 通用
投影路径。135 项 Attention 单元测试和 19 项 baseline 测试检查报告及配置约束。

最终 cycle 和结果由 `single_operand_context_results.json` 记录。
旧双 bank 结果仅作为资源撤销前后的影响比较，不标为同资源调度收益。

## 多行组的实际影响

相同输入的 S1024/Hq4/Hkv2/D64、32-array 回归检查了 16 个 HBM 输出
区域逐字节一致。撤销额外 operand 存储前后的测量为：

| partial 路径 | 原双 context cycle | 单 context cycle | 全 worker 权重编程次数 |
| --- | ---: | ---: | --- |
| 既有 C-buffer | 74,571 | 77,966 | 512 → 1,024 |
| 慢 GM（64 bytes/cycle、base latency 3） | 104,484 | 104,971 | 512 → 1,024 |

该差异是撤销额外资源后的影响，不是同硬件的调度收益。相关输出比对证据为
`results/single_operand_context/regressions_verified/*/single_operand_impact_comparison.json`。

## 当前完整端到端 cycle

| S | 投影（包含融合 RMSNorm） | 阶段交接 | Attention | 总 cycle |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 416,538 | 2,019 | 155,215 | 573,772 |
| 2048 | 801,164 | 3,794 | 549,932 | 1,354,890 |

64-array、每个 worker 单行组配置不需要跨行组 operand bank，因此完整
S1024/S2048 的实测 cycle 与撤销前一致。输出分别比对 10 MiB/20 MiB、
16 个 HBM 区域逐字节一致；O 对 NumPy 的最大绝对误差均为
0.0001220703125。源码/二进制 13 项 hash 在测试后保持一致。

正式结果位于 `results/single_operand_context/verified/summary.json`；7 项回归
位于 `results/single_operand_context/regressions_verified/summary.json`。

## 后续数据流优化实现

已实现并验证以下调度优化：GM 权重有限 lookahead（1–8 tile，默认 4）、输入
pair 的后台读取、C-buffer/GM partial 后台读取和已有 ahead restore、单 context
下的行组顺序调度，以及 64-array 行组的输出累加。所有路径复用既有 GM、cache、
C-buffer 和输出累加器，不增加硬件资源。

深度 8 的回归中每个 worker 的权重预取请求 72 次、命中 72 次；三 block 总 cycle
为 243,842，投影为 76,979，与深度 4 相同，说明该配置的瓶颈已经转移到唯一
operand context 的矩阵编程和阵列计算。不能通过继续扩大 GM lookahead 消除该瓶颈。
完整优化记录见 `OPTIMIZATION_RESULTS.json`。


## 固定调度策略

当前不引入运行时成本选择器。软件固定采用：64 arrays 使用 output retention；
32 arrays 使用已有 C-buffer partial，并把 GM 权重 lookahead 设为 8 tile。
这条规则只依据 array 数量选择已验证的数据流，不改变硬件资源。


> 32-array weight-stationary 原型已按用户决定退役，状态机及专属统计已移除。
> `GOLEM_PROJECTION_WEIGHT_STATIONARY=1` 被拒绝，32-array仍使用原paired调度。
> 后续优化以现有64-array输出驻留为基础，见
> [舍弃WS后的重分析](LATENCY_REANALYSIS_WITHOUT_WEIGHT_STATIONARY_2026-10-11.md)。
