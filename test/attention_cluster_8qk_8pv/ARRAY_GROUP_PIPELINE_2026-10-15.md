# FFN 的两组阵列交错调度

## 资源与执行范围

本轮比较原来每个 tile 使用 64 arrays 的单组调度，与同一批 arrays 分为
`0..31`、`32..63` 两组交错执行的调度。每个 array 仍然只有一个 operand context。
没有修改阵列/SFU/GM/WCP 实现、端口数、队列容量、带宽、NoC 或 HBM 时序与组织。
GM 仍为 2 MiB，最大布局边界 2,039,808 B；C-buffer 仍为 1 MiB，FFN 使用 256 KiB partial。

这是 `DecoderOpJob` 调用现有异步接口的调度变更，不增加 opcode 或 descriptor 字段。
`GOLEM_DECODER_ARRAY_GROUPS=1` 为默认路径，`=2` 显式选择实验路径。
只作用于 op8/op9 的 FFN；每个行组超过 32 行时拆成两组，其余尾行沿用原路径。
独立 Linear、Attention、Wo 和 M=1 LM head 的映射保持原有设置。

## 调度、存储和精度

原路径对同一组全部 arrays 执行 `matrix broadcast → input scatter → MAC`，
等 MAC 回调后才开始下一 K tile。新路径先提交 A 组，再提交 B 组；
只等待即将复用的那一组完成，另一组可继续计算。
输出 gather、FFN panel 切换和 down partial 保存之前必须收齐两组计算回调。
down 的跨 panel partial 通过既有计时 C-buffer 恢复，之后才允许开始该输出 tile 的计算。

每一行仍按原先的 K 顺序累加，没有跨组做新的 reduction，也没有改变 FP16 舍入边界或 FP32 内部计算。
同一权重 tile 只做一次原有 HBM/GM 读取，但在既有 array buffer 端口上广播两次。
控制器保留当前 tile 的功能快照直至两组提交完毕；原有 GM 预取槽在这之前不会被下一 tile 复用。
输入仍只通过原有 GM 端口读取一次，scatter 时按数组行号切片；不新增 GM scratch、权重缓存或旁路读取。

## 验证方法

行为测试使用生产 `DecoderOpJob`，每个 array 的矩阵/输入/输出独立存储，
计算完成时才读取 operand，并延迟、打乱各 array 的完成回调。
测试覆盖 33/64 行、尾行、空 lane、多行组、2/4/8 预取槽、预取开关、
不同 FFN panel 宽度、端口拒绝重试、延迟与乱序 DMA ACK。
同时断言：不覆盖忙碌 array、输出读取等待全部 MAC、实际出现一组计算期间另一组编程。
这些 mock tick 仅验证行为，不用于报告性能。

真实 SST 对照沿用独立 FP16 oracle、完整 dense 权重读取、真实 GM 端口和共享 SFU，
计时从第一个输入 DMA 发出到最终输出 DMA ACK。两条路径采用相同输入、guest、
SST library、硬件环境和 CPU placement，只有分组调度开关不同。

正式 H=2048、I=8192，每个 worker 有 12,288 个 FFN 权重 tile。
候选必须执行 24,576 个 32-array launch，array 加权回调周期为
`12,288 × 64 × 67`，其中 67 包括 66-cycle 物理 MAC 与 WCP 回调。
额外审计真实阵列统计，要求矩阵广播次数翻倍、scatter 次数翻倍，但矩阵 sink bytes、
scatter bytes、output gather/scatter 工作量不变。

`compute_sum` 在分组后累加两组可能重叠的 launch 回调时长，不能当作 MAC wall time。
`both_groups_compute_cycles` 也是两组回调尚未全部返回的计数，包含 WCP 完成递交时间。
物理有效工作量和端口服务使用各自的计数，不能把这些重叠量相加为端到端耗时。

## 局部结果

| 形状 | 单组 64 arrays | 两组 32 arrays | cycle 降幅 |
| --- | ---: | ---: | ---: |
| H=256，64 rows，1 worker | 110,738.125 | 109,777.125 | 0.87% |
| H=2048，64 rows，1 worker | 1,942,869.125 | 1,776,512.625 | 8.56% |
| H=2048，1024 rows，16 worker / MPI8 | 2,003,388.5 | 1,911,141.375 | 4.60% |

六个 case 均通过独立 FP16 oracle、完整 HBM 读写/ACK、实际 EXP/倒数工作量与 partial SRAM 审计。
正式单 worker 的有效 MAC 峰值效率从 40.48% 提升到 44.27%。
物理 MAC 的 array 加权忙碌率从 41.74% 提升到 45.65%；不是“至少一个 array 忙碌”的时间并集。

正式每 worker 的 array buffer 物理服务从 909,024 增到 1,375,968 cycles，
其中矩阵广播从 479,232 增到 933,888，input scatter 从 405,504 增到 417,792。
两个子组虽然各少一个 fanout tree stage，但广播次数翻倍，额外增加 96 MiB ingress；
矩阵 sink bytes、scatter 输入字节、输出 scatter/gather 工作量完全相同。
这是用更多已有端口服务换取编程与其他 arrays 的 MAC 重叠，不是消除了编程成本。

竞争环境的收益小于单 worker：core16 前台权重就绪等待从 37,013 增至 81,982 cycles，
其中 DMA 等待从 23,084 增至 45,905，GM 等待从 13,929 增至 36,077。
这个分解只能解释部分收益收窄，不能把后台各 DMA 的 RTT 累加值当作可消除的关键路径。
两组模式的 `wait_array` 还包含等待当前子组完成的调度周期。

## 正式两层端到端验收

S=1024、H=2048、I=8192、两层，含架构内 Embedding、final RMSNorm 和最后 token 的
128256 词表 LM head，MPI rank=8。沿用合成权重，完整执行全部 dense tile；没有推进真实 checkpoint 导入。

**6,519,275.75 → 6,262,408.75 cycles**，节省 **256,867 cycles（3.9401%）**；
1 GHz 下从 6.5193 ms 降为 6.2624 ms。当前版本的单组基线精确复现上一轮已验收的端到端周期。

| 阶段 | 单组基线 cycle | 两组候选 cycle |
| --- | ---: | ---: |
| 第一层 Attention O → 层输出 Y | 2,323,972 | 2,190,696 |
| 第二层 Attention O → 层输出 Y | 2,351,692 | 2,216,076 |
| LM head | 621,965.75 | 633,901.75 |
| 完整 token DMA → logits ACK | 6,519,275.75 | 6,262,408.75 |

前两行是可加阶段边界；其中包含 Wo、残差1、Norm2、FFN 和残差2。
完整的 11 段边界及其严格求和见归档。最后完成的 worker 仍为 core16；其两层 FFN 作业分别从
1,961,475 / 1,965,789 降至 1,828,716 / 1,830,574 cycles。
两层这些 FFN 作业合计减少 267,974 cycles；全链路减少 256,867，不能把作业节省直接当作最终节省。
LM head 的映射没有改变，但上游排程改变了其进入时的 HBM/NoC/刷新状态，阶段耗时增加了 11,936 cycles。
这是下游实测差异，不是对某一种单独等待来源的因果归因。

两组各 416 个 Decoder job 的 DMA/GM 请求和字节完全相同；每组 393,216 个 FFN 权重 tile
均通过生命周期与唯一性审计。候选每 worker、每层执行 24,576 个子组 launch；
矩阵广播 ingress 增加，阵列 sink bytes、scatter bytes、输出 gather/scatter 工作量相同。
全部实际 SFU unit 的 accepted-token 计数相同，包含 EXP、倒数及 vector-op。
两层层输出、Embedding、保留的 Wo/H1/Norm2、final norm 和 logits 全部 FP16 逐位一致，
greedy token 都是 99037。153 项生产、140 项 host、6 项架构配置回归共 **299 项通过**；
新增的异步所有权测试也在最终断言版本下单独通过。

全链路有用 MAC 工作量不变，按现有 65,536 MAC/cycle 峰值计算的有效效率从
**30.22% 提升至 31.46%**。这是数学有效工作 / 峰值，不是逐阵列忙碌时间的实测并集。
旧的保留全 64 阵列串行顺序下界不能直接用于该分组调度。

SST wall 基线 1220.47 秒（约 20 分 20 秒），候选 1172.36 秒（约 19 分 32 秒）。
两组在不同 CPU 集合上并行运行，wall 只记录本轮运行成本，不作为可复现的独立性能结论。
cycle 来自真实 DMA 边界，host 输入准备、验证和 greedy argmax 均不纳入。

## 证据与后续

轻量归档：[array_groups_pipeline](progress_results/2026-10-15/array_groups_pipeline/)。
主要文件包括 [完整交叉对照](progress_results/2026-10-15/array_groups_pipeline/full_comparison.json)、
[可加阶段](progress_results/2026-10-15/array_groups_pipeline/phase_comparison.json)、
[硬件与编译树哈希约束](progress_results/2026-10-15/array_groups_pipeline/hardware_contract.json)，
以及各级局部结果、真实端口计数、完整运行配置、回归日志和权重审计。
原始资源日志和 HBM backing 位于 `results/array_groups_20261015`。

为容纳配对完整仿真，对此前完成的 HBM dump 中全零文件页使用 `fallocate --dig-holes`。
历史文件逐一验证完整 SHA256 不变；新文件在所有 SST 进程退出后才处理。
逻辑文件长度、非零数据、配置的 HBM 节点大小及模型时序均不变，没有删除结果。
相关空间整理记录另行归档，不把宿主文件占用变化算作硬件容量优化。

保持 opt-in：正式形状可在既有四槽预取、16-tile FFN、down/residual 融合上叠加
`--ffn-array-groups 2`。小 K 的收益很小，不把该选项直接设为通用默认值。
本轮完成 FFN 分组实验及多级验收；下一项优先对 LM head 做 N 维并行的局部实验，
利用现有闲置 arrays，再推进 Wo → 残差1 → Norm2 的完整行流水。

## 局部复现

```bash
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_decoder_micro.py \
  --artifact-root results/groups_trial --shape formal --workers 16 \
  --mpi-ranks 8 --cpu-set 0-7 --with-down --with-residual --prefetch-slots 4 \
  --variants residual16q4 groups2residual16q4 --reference-variant residual16q4 --weight-trace

python3 baseline/attention_cluster_8qk_8pv/analyze_decoder_groups.py \
  --root results/groups_trial --output results/groups_trial/array_audit.json
```

完整两层在原参数上增加 `--ffn-array-groups 2`；只有局部与 16 worker 竞争测试胜出的方案才验收。

```bash
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/groups_stack_trial --shape formal --embedding \
  --mpi-ranks 8 --cpu-set 0-7 --gemm-prefetch --ffn-down-fusion \
  --ffn-panel-tiles 16 --prefetch-slots 4 --weight-trace --down-residual-fusion \
  --ffn-array-groups 2
```
