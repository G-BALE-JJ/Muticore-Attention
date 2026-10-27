# 软件算子融合与末层流水整组优化

本轮以 `7d3970e` 的中间层 Y 驻留方案为起点，统一实现残差 Add/RMSNorm2
融合、归一化结果打包写入，以及末层 FFN/最终 RMSNorm 的 worker 级流水。
同时移除 decoder/LM 消费链上的重复环境配置解析。
全部工作位于独立的 `all-gemm-reuse-window` 分支和工作区，原始
`Muticore-Attention` 工作区及其安装库保持原状。

## 正式结果与采用结论

两组正式仿真、完整数值参考、实际 payload/ACK 及 reuse-window 审计全部
通过。按 1 GHz 换算，下表单位为 ms；保留原始 cycles 的机器可读结果见
[汇总证据](progress_results/2026-10-22/software_fusion_batch_summary.json)。

| 测量范围 | 上一轮采用版本 `7d3970e` | 本次同构建对照 | 本次整组候选 | 相对上一轮下降 | 相对同构建下降 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 两层 decoder | 5.458952 | 5.414856 | **5.400835** | **1.0646%** | **0.2589%** |
| token IDs → 最后 token 的全部 logits | 6.288746 | 6.165061 | **6.134778** | **2.4483%** | **0.4912%** |
| LM head | 0.659918 | 0.580844 | 0.582536 | 11.7259% | −0.2913% |

本次候选两个总延迟口径都优于同构建对照及上一轮采用版本，因此采用本次
候选，保留两个新开关为显式启用。完整端到端的原始结果为
6,134,778.375 cycles，两层 decoder 为 5,400,835.25 cycles。

同构建对照和候选都包含不可变配置快照及 LM 循环外解析。因而相对上一轮
的 2.4483% 是整组软件修改收益，不能全部归因于 Add/Norm 融合；两个新
开关的配对增益是 0.4912%。LM 比配对对照慢 0.2913%，两层 FFN 的局部
span 也不都下降，说明全局依赖、资源竞争和 worker 偏斜仍然存在。

| 实际计数 | 同构建对照 | 整组候选 | 变化 |
| --- | ---: | ---: | ---: |
| decoder-op 作业数（含最终 Norm/LM） | 416 | 384 | 减少 32 |
| decoder-op GM 读取字节 | 7,637,720,064 | 7,629,331,456 | 减少 **8 MiB** |
| decoder-op GM 写入字节 | 155,513,984 | 155,513,984 | 相同 |
| Norm2 打包 GM 写请求 | 65,536 | 16,384 | 减少 **75%** |
| decoder-op HBM 读取字节 | 4,027,846,656 | 4,027,846,656 | 相同 |
| decoder-op HBM 写入字节 | 12,840,064 | 12,840,064 | 相同 |

上述 HBM/GM 字节为 decoder-op 日志的计数范围，不是整个 Attention 集群
的总流量；另行核对的 GEMM window 流量和执行工作量也完全相同。本轮相对
已驻留基线的新增 HBM 节省为 **0 B**。

第 1/2 层每 worker 的平均 Wo 完成→FFN 启动，从
49,031.875/48,937.4375 cycles 降为 46,365.5625/46,375.625 cycles。
消除重复配置解析后，两组 Norm 完成→FFN 启动均为 476 cycles；融合前
试跑的约 27,300 cycles 串行等待已移出消费链。

最终 Norm 与 decoder 尾部实际交叠 40,189 cycles。其跨 worker span 为
81,471.75 cycles，虽大于对照的 44,797.375 cycles，但 decoder 结束后
剩余尾部由 59,251.625 降至 41,357.5 cycles。不能把重叠区间再加到
decoder 延迟上。

最终 Y 的 2,097,152 个 FP16 元素、final Norm 的 2,097,152 个元素及
128,256 个 logits，与同构建对照和上一轮采用版本均逐位相同，差异数为
0，预测 token 为 99037。32 个 QKV、64 个 Wo、各 256 个 Gate/Up/Down
和 256 个 LM window 记录全部通过；共 539,264 次 tile-group launch，
16 个 dense worker 均覆盖。另有 128 个 QK 作业、2,560 个 PV window，
其 reuse-window、C 流量及实际流水执行审计均通过。

实测 guest SHA256 为
`7ee4f229860f8c83dea431a72f10cbcf58e02d6de4bb76c454807e95f067f57c`，
SST Golem 库为
`ebe5bd84dbfffa3baf98780cf214b73c2d01db4b958c268ae9b910e8f0fb48f0`；
同构建对照与候选完全一致，详细源文件及验证器哈希保存在汇总证据中。

## 已实现的数据流

### Add 与 RMSNorm2 的实际融合

`--add-norm-fusion` 使 stage 2 在同一个 descriptor 内调用已有 SFU Add 和
RMSNorm，不再提交独立的 stage 3 作业。两层、16 worker 共减少 32 个
descriptor 的写入、启动及等待，而不是隐藏原来的作业记录。

每次处理至多 8192 个 FP16 元素：

1. 通过计时 GM 端口读取 Wo 和保留的原始 X。
2. 调用原有 SFU Add，保留原有 FP16 舍入边界。
3. 将 H1 写入原有 GM 驻留区域并等待 ACK。H1 后续还要作为 FFN 残差，
   因此这次保存仍然必要。
4. 同一个有界结果直接交给原有 RMSNorm SFU，不再从 GM 重读 H1。
5. gamma 仍经 HBM/GM 计时路径读取；启用原有 gamma reuse 后，每个作业
   只从 HBM 载入一次，每个向量请求的 GM gamma 读取仍保留。
6. Norm2 结果写入既有 GM 的 tile-major 输入布局，所有写 ACK 完成之后，
   才允许 FFN 消费。FFN 的 H1 残差输入保持原地址和所有权检查。

没有增加 opcode 或 descriptor 长度。Add 原来不用的 `weights_offset` 存放
gamma2 的节点内偏移，`head_dim` 存放 Norm2 的逻辑地址偏移；软件开关、
stage-chain、形状及必要字段不匹配时拒绝该融合请求。没有修改 SFU、阵列、
DMA、NoC、端口、队列、时钟或存储容量。

### Norm2 结果的请求合并

在已有的有界 SFU 结果内改变打包顺序，将同一 K tile 的连续行合并到一个
计时 GM 写请求。H=2048 时，每个请求对应 4 行、512 字节；原先每行的
128 字节请求合并为一个，仍不超过现有 4096 字节请求限制。

这不是省去有效数据的写入，也没有引入整张量的无计时缓存。H1 和 Norm2 的
完整 GM 写字节保持不变，所有读取/写入、有限队列、重试及 ACK 都继续计时。

### 末层流水

`--eager-final-norm` 让每个 worker 在自己的最后一层 FFN 完成之后立即
启动最终 RMSNorm，允许较早完成的 worker 与其他 worker 的 FFN 尾部交叠。

root 在释放最后一层 decoder 之前，先初始化全部 16 个 final Norm 完成标志
并等待初始化 ACK，避免快 worker 的完成标志被后续清零覆盖。worker 仍须
等待自己的 FFN、最终 Y HBM 输出、GM 所有权发布及完成写回 ACK，才进入
final Norm。root 继续等待全部 final Norm 完成之后才释放 LM head。

### 消除消费链上的重复配置解析

首组完整试验揭示一个会抵消融合收益的 guest 控制开销：融合完成到 FFN
descriptor 启动之间，单 worker 仍串行执行三次 `getenv`，约耗费 27,300
cycles。原来分开的 Add/Norm 路径也有这段开销，只是落在 Add→Norm 之间。
所以跨 worker 的 Add/Norm 总 span 缩短，不能直接当作整个 FFN 开始时间的
等量提前；必须比较每个 worker 的完整 Wo 完成→FFN 启动路径。

正式版本把五个不可变的 decoder 融合开关在 worker 启动时读取一次，以五个
布尔值传入各层调度，生产/消费交接不再调用 `getenv`。LM head 的逻辑序列
长度也在词表分块循环外读取并验证一次，而非每个 descriptor 重读；正式
词表每 worker 从 16 次降为 1 次。描述符、逻辑 token、词表分片及 ACK
依赖保持不变，配置快照不存储任何张量。

这属于实际 guest 软件优化。两组正式对照都使用该配置快照版本，仅融合和
eager final Norm 两个开关不同；与上一轮结果比较时，明确包含配置解析优化
和新 guest 控制路径的差异。原有测量契约仍从首次 token IDs 读取开始，
不包括进程初始化；不将一次性启动配置解析计作张量搬运或 HBM 节省。

## 融合边界

| 区段 | 本轮之后的执行方式 |
| --- | --- |
| Norm1 → QKV | 继续使用已有融合及 packed GM 输入 |
| QK → online softmax → PV | 继续使用已有流水和 reuse-window |
| PV → Wo | 保留原有 HBM 重分布和全局 Attention 完成依赖 |
| Wo → Add/RMSNorm2 | Wo 保存于现有 GM；Add/RMSNorm2 为一个融合 descriptor |
| Norm2 → Gate/Up | 保持已打包的 GM 输入交接 |
| Gate/Up → SiLU → Multiply → Down → 残差 | 继续使用已有有界面板融合及 Down 部分和机制 |
| 中间层 Y → 下一层 Norm1/残差 | 保留 GM 驻留，继续省略非末层 Y 的 HBM 镜像 |
| 最后一层 FFN → final Norm | 按 worker 完成状态流水启动 |
| final Norm → LM head | 保留最终归一化输出及全局消费者同步 |

因此，整组优化不等于把整个网络变成一个算子。Q/K/V 面板、PV→Wo 重分布、
最后一层 Y、final Norm 输出、logits 和完成标志的 HBM 写回仍然存在。
Wo 与归一化/FFN 之间仍有 GM 交接。跨 worker 所有权、完整 hidden 行归约、
单操作数上下文和既有存储限制，仍是后续调度需要遵守的边界。

## 测量与审计方法

正式形状为 S=1024、2 层、H=2048、FFN=8192、vocab=128256、16 worker、
8 MPI ranks。两组使用同一重新构建的 SST 库和 guest，仅改变两个软件开关。
对照沿用上一轮采用的 GM、输入、跨层、LM 输入驻留及中间 Y 写回省略配置。
输入为原有确定性测试输入与稀疏测试权重，全部 dense MAC 和权重读取仍执行。

完整端到端以 token IDs 首次读取至最后有效 token 的全部词表 logits DMA ACK
为界；LM 输出为 128256 个元素，沿用原有测试口径。两层 decoder
以首层 X 首次读取至末层 Y 最后 DMA ACK 为界。cycles 来自模拟器，按 1 GHz
换算为 ms，排除 host 准备、仿真墙钟和校验时间。

启用末层流水后，final Norm 的全 worker span 包含更早启动及 worker 完成
时间差，可能比对照更长；它与 FFN 尾部交叠，不能和 decoder 时间直接
相加。是否优化以最后输出 ACK、decoder 结束到 final Norm 完成的尾部，
以及完整端到端指标共同判断。

`summarize_fusion_batch.py` 核对两组输入/权重哈希、软件配置、实际库和 guest
哈希、全部 GEMM 工作量、HBM/GM 流量以及最终 FP16 输出。除配对结果外，也
与上一轮采用方案比较，避免把新增 guest 分支的成本漏出比较口径。

47 个固定硬件/支撑源文件哈希完全不变。另两个文件分别进行限定审计：
GM 布局仍只有此前的 Wo 软件别名；系统配置文件只新增两个 guest 环境变量
白名单项，删除这两行后必须与原固定快照哈希完全一致。环境转发不改变硬件
拓扑或参数。两组测量使用相同的这份软件配置。

数值检查继续从初始输入独立计算完整两层，保持原有误差阈值，并检查全部
最终 Y、final Norm 和 logits 的 FP16 位一致性。因中间 Y 不在 HBM 中，局部
算子和 Q/K/V 面板检查沿用已验证的 materialized counterpart 作为 host 输入
参考；只允许明确列出的等价软件调度差异，输入、权重哈希和布局必须相同。

实际 payload 审计检查：原始残差完整读取一次、Norm2 完整打包写入、写 ACK
不越过作业完成、每个 Add/RMSNorm 请求至多 8192 元素、归一化写入不早于
其 SFU 结果，以及 final Norm 对自己 worker 的 FFN 完成依赖。空的 LM 尾部
分片不被当作 GEMM 窗口，也不虚报输入读取节省。

## 回归覆盖

- host 全量测试 218 项通过，包含缺失/重复/提前/失败的实际数据访问、融合
  数值参考身份、LM 空分片、软件环境转发及 worker 局部依赖的反例。
- 既有生产 C++ 路径与新增融合路径合并运行 20 项通过。融合用 H=1088、
  2048 的非均匀 FP16 数据及非整向量尾块，比较全部 H1、打包 Norm2 和
  最终 Y，覆盖请求重试、延迟 ACK、gamma 回退、错误 generation 和消费者。
- 新增 guest 调度测试 2 项通过：只替换 RoCC 提交边界，执行生产调度函数，
  检查融合/非融合描述符顺序、残差/完成依赖、跨层配置快照、完整 16 个 LM
  分块的地址与尾部，以及无效逻辑序列拒绝。LM 环境读取一次的检查在旧实现
  中实际失败，修改后通过。
- 正式 SST guest 与 LM micro guest 构建通过；固定硬件及原始工作区审计
  通过。测试数量分别统计，未将同一测试的多次运行重复相加。

## 复现

在本工作区已验证的 materialized counterpart 存在时运行：

```bash
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_all_gemm_reuse_window_baseline.py \
  --artifact-root results/fusion_batch_reproduction/final_control \
  --baseline-result results/software_output_residency_20261022/final_candidate/reuse_window_probe/decoder_stack_result.json \
  --verification-reference results/software_output_residency_20261022/final_control/reuse_window_probe/decoder_stack_result.json \
  --shape formal --run --window-a-prefetch 1 --lm-n-parallel 1 \
  --gm-stage-chain --layer-residency --lm-input-residency --input-residency \
  --elide-intermediate-output --cpu-set 8-15
```

候选使用同一命令，将 artifact root 改为 `final_candidate`，再增加
`--add-norm-fusion --eager-final-norm`；独立 CPU 绑定可使用 `0-7`。
两组完成后运行：

```bash
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/summarize_fusion_batch.py \
  --experiment-root results/fusion_batch_reproduction
```

## 试跑记录

`attempt1_*` 在回归发现旧 Add 描述符的未使用字段兼容性问题后停止。
`attempt2_*` 在发现新开关未传入模拟 guest 后停止。两组均不用于延迟比较。
旧 Add 路径已恢复兼容；guest 转发只新增两个环境变量项，并有专门回归检查。

小形状集成运行使用 S=1024、H=256、vocab=8201，完整数值、资源和
reuse-window 检查通过。它暴露了既有 host 审计对空 LM 尾部分片的误计数；
修正仅影响 host 校验，对原始模拟器日志和运行结果只重新校验，没有重写。
另一个请求额外 CPU 集的试跑在 MPI 绑定阶段失败，不作为性能证据。

`pre_control_cache_control` 和 `pre_control_cache_candidate` 是移除重复配置
解析之前已完成且通过审计的一组正式仿真，分别为 6,317,502 和
6,286,617.625 cycles；它们不作为最终版本的延迟。其原始汇总保留在
`pre_control_cache_summary.json`，其中采用标记为当时汇总脚本的旧规则，
不能用于选择最终版本。最终汇总同时要求相对同构建对照及上一轮采用版本，
完整端到端和两层 decoder 两个口径都改善，才选用新候选。

正式有效证据位于 `results/software_fusion_batch_20261022/final_control` 和
`final_candidate`；多 GB 日志、HBM 镜像及生成二进制不提交 Git。
