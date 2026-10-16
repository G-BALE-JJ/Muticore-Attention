# 当前两层 prefill 的延迟、下界与利用率

本报告保存分组调度优化前 **6,519,275.75 cycles** 的分析口径。后续已完成
[FFN 两组交错验收](ARRAY_GROUP_PIPELINE_2026-10-15.md)，最新为 **6,262,408.75 cycles**；
下文保留全 64 阵列串行顺序的下界不能直接套用于新调度。

分析对象为 `results/residual_pipeline_20261015/stack_fused`，不是旧的未融合测试。
S=1024、H=2048、I=8192，两个 decoder layer，含 Embedding、final RMSNorm 和最后一个
token 的 128256 词表 LM head。使用合成权重，但完整执行所有 dense tile；不代表完整模型的真实 checkpoint 推理。
本次只增加离线分析工具和报告，没有修改硬件或调度，也没有重新进行 SST 全局仿真。

计时从首个 token-ID DMA 发出到最后 logits DMA ACK；1000 tick/cycle，1 GHz。
总计 **6,519,275.75 cycles = 6.5193 ms**。上一组配对基线为 6,575,251.75，
down → 第二残差融合减少 55,976 cycles，下降 0.8513%。

## 延迟分布

下表来自不重叠的 DMA 边界，严格可加；不可把不同 worker 的 job 活动窗口相加。

| 阶段 | cycle | 占比 |
| --- | ---: | ---: |
| Embedding | 26,856.75 | 0.41% |
| Embedding → Decoder 交接 | 55,704.25 | 0.85% |
| 第一层 Norm1/QKV/RoPE/Attention | 507,826.75 | 7.79% |
| 第一层 Wo/残差1/Norm2/FFN/残差2 | 2,323,972 | 35.65% |
| 层间交接 | 52,683.25 | 0.81% |
| 第二层 Norm1/QKV/RoPE/Attention | 504,192.75 | 7.73% |
| 第二层 Wo/残差1/Norm2/FFN/残差2 | 2,351,692 | 36.07% |
| final norm 交接 | 15,457.25 | 0.24% |
| final norm | 48,965.75 | 0.75% |
| LM head 交接 | 9,959.25 | 0.15% |
| LM head | 621,965.75 | 9.54% |

两层最后完成的 worker 都是 core16。其 FFN（含已融合的残差2）分别为
1,961,475 和 1,965,789 cycles，合计占端到端 60.24%。这是关键 worker 的
作业时长占比，不是所有 FFN job 时长的总和。

## 三种下界的适用范围

| 下界 | cycle | 实际 / 下界 | 保留的约束 |
| --- | ---: | ---: | --- |
| 纯有用 MAC 工作量 / 阵列峰值 | 1,970,152 | 3.31× | 只保留数学 MAC 数与硬件峰值 |
| 当前 DMA payload / HBM 峰值 | 3,560,702.8 | 1.83× | 当前重复读取与流量分布，每节点 320 GB/s |
| 保留当前后半层串行顺序的保守下界 | 4,185,886.8 | 1.56× | Wo → Add1 → Norm2 → FFN；每 tile 编程 → scatter → MAC |

第一项的峰值是 16 worker × 64 arrays × 64 MAC/cycle = 65,536 MAC/cycle，
即 65.536 TMAC/s。工作量含 QKV、Wo、FFN、因果 Attention 的有效 QK/PV 和 LM head；
不含 masked pair、padding 的额外计算及 SFU 工作。

第二项逐条扫描 GM trace 中的数据节点 DMA transport，每个请求只计一次 payload，
以每节点负载 / 320 B/cycle 的最大值作下界。四节点总平均约 698.76 GB/s。
只纳入计时边界内完整结束且大小至少 128 B 的请求；小控制请求另行统计，128 B 的控制请求可能仍纳入。
这是当前流量的乐观下界；软件减少重复流量后需重算。没有把 DMA RTT 相加当作 HBM 服务时间。

第三项故意不计 Embedding、两层 Norm1/QKV/RoPE/Attention 及交接，用于保守地展示
当前串行调度本身的成本，并非完整最优调度预测。每层计入：

```
Wo:          1024 × (39 + 33 + 66) =   141,312
Add1:          16 × 515           =     8,240
Norm2:         16 × 1747          =    27,952
fused FFN:  12288 × (39 + 33 + 66) = 1,695,744
每层后半段下界                       = 1,873,248

两层后半段 + final norm 27,952 + LM-head payload / 1280
= 4,185,886.8 cycles
```

当前每个 FFN tile 使用全部 64 arrays，单 operand context，因此同一 tile 的
matrix program、input scatter、MAC 顺序不可重叠。该下界没有相加可能重叠的
SFU/DMA/GM 工作，也没有计入输出 gather、partial restore、回调和前台等待。
若改变软件分组、循环顺序或融合，上述串行下界也需重算。
三个下界不可相加；差额不能解释为一定可消除的等待，也不能据此承诺加速比。

## 当前利用率

| 指标 | 数值 | 解释 |
| --- | ---: | --- |
| 全链路有用 MAC / 峰值 | 30.22% | 有效计算效率，不是逐阵列忙碌时间的实测并集 |
| HBM 全链路 payload / 峰值 | 54.59% | 四节点约 54.56%–54.62%，负载均衡 |
| 阵列 buffer 端口服务占比 | 35.77%–35.79% | 单共享端口的实际 transfer cycles / 全链路周期 |
| GM 读端口服务占比 | 35.09%–35.56% | 实测 local_read service，裁剪至全链路边界 |
| GM 写端口服务占比 | 19.03%–19.13% | 实测 local_write service，同上 |
| FFN 关键 worker 的 MAC 忙碌率 | 41.26%–41.35% | 12288 次 launch × 66 / FFN 作业时长 |
| FFN 关键 worker 的有效峰值效率 | 40.01%–40.09% | 扣除每次 launch 的 2-cycle fill/drain |
| FFN 倒数流水线 issue 容量占比 | 26.67%–26.73% | 单 lane、II=1；524288 tokens / FFN 时长 |
| LM head 映射的阵列空间使用率 | 1/64 = 1.5625% | M=1 路径每 worker 同时使用一个 array |
| LM head 有用 MAC / 全阵列峰值 | 0.6444% | 包含其读权重、装载和等待的整个阶段 |

以上百分比的分母和资源不同，不能相加。MAC 总工作量没有逐周期 trace，不能伪装成实测
全系统阵列 busy 并集。HBM 54.59% 是 payload 带宽占比，不是队列或存储容量占用。
NoC、C-buffer 的完整利用率没有在本报告中重建，不能由 HBM/GM 的平均值推断。

`run_summary.csv` 中旧的 `exec_array_utilization_pct=24.19%` 只对应旧 GEMM
统计窗口；其 `hbm_utilization_pct=68.27%` 使用另一套 tCCD roofline 和分母。
Ramulator backend 的约 54.61% 是 backend read service window 口径。
这些数值不能直接作为本次完整 prefill 的利用率。

## 下一步的优先级

优先在 FFN 局部实验中检查现有 64 arrays 的软件分组交错：比较当前全 64 阵列一组与
两个 32 阵列组，让一组计算时准备另一组的已有 operand context。仍然只有原来的
64 arrays、单 context、原有 GM/C-buffer/端口；不得为实现重叠新增 bank、端口或带宽。
必须重新计入较小广播组产生的重复编程、GM 流量、buffer 端口占用和同步，不能预设一定更快。
先做排程和资源预算，若现有接口不能表达就说明限制；局部无收益则停止该方向。

原因是关键 worker 每层 FFN 的 12288 次 matrix 编程回调累计 491,520 cycles、
input scatter 物理服务 405,504 cycles、MAC 物理服务 811,008 cycles。
相同阵列的顺序约束已经解释约 86.3%–86.5% FFN 作业时长。
两层权重预取前台等待仅 38,343 / 41,840 cycles，合计约端到端 1.23%；
不能把已经隐藏在后台的数百万 DMA RTT 累加值当作可优化的关键路径。
维持当前串行顺序时，进一步消除这两个作业高于上述 tile 服务下界的全部差额，
也仅约端到端 8.2%，且其中仍含必要输出与 partial 操作。

第二优先做 LM head 的 N 维分组交错局部实验。M=1 的空间利用率低，但该阶段
526,641,664 B 的输入/权重/输出流量本身已有 411,438.8 cycles 的 HBM 理想下界。
在当前流量下，即便忽略其他全部成本，也至多直接节省该阶段约 33.85%，折合
端到端约 3.23%；不能从 1/64 映射直接推导 64 倍加速。

Wo → 第一残差 → Norm2 的行级流水仍可作为收益较可控的后续优化。
Norm2 必须收齐一整条 hidden 行完成归约后，才可开始该行的 FFN；
不能把部分列的归约当作完整 RMSNorm。继续沿用局部正确性/资源审计 → 单 worker
正式形状 → 16 worker 竞争 → 仅对胜出方案做两层端到端验收的流程。

## 复现

```bash
python3 baseline/attention_cluster_8qk_8pv/analyze_decoder_utilization.py \
  --case results/residual_pipeline_20261015/stack_fused \
  --archive baseline/attention_cluster_8qk_8pv/progress_results/2026-10-15/residual_pipeline \
  --output baseline/attention_cluster_8qk_8pv/progress_results/2026-10-15/decoder_utilization_analysis.json
```

完整结果见 [decoder_utilization_analysis.json](progress_results/2026-10-15/decoder_utilization_analysis.json)。
工具检查完整 416 个 decoder job 的 DMA 读写字节、单 worker 作业顺序、GM 单端口
服务区间不重叠、32 个 FFN 的 dense launch 数和回调计数；本次全部通过。
