# FFN 关键路径预算与需求优先实验

本轮固定硬件资源、FP16 运算边界和全部稠密工作量，先量化机会，再用同构建 SST 配对决定是否进入全链路。全部加速器延迟使用 cycles，1000 simulator ticks/cycle。

基准为 p7_restored_a（与历史 selected 周期一致）：完整端到端 6,084,167.75 cycles，两层 decoder 5,343,797.75 cycles。完整边界为首次 token-ID DMA send 到最后有效 token 的完整 logits DMA ACK，host argmax 不计入。

`analyze_ffn_critical_budget.py` 逐层定位最后完成 worker，核对窗口独占 phase 和 job 时长，分别计算理想化消除全部 B-GM 等待、全部 acquire phase 后的阶段完成边界。它对所有 worker 重新取最大完成时间，不把 worker wait 或 stage span 相加为端到端收益。

| 层 | 最后 worker | FFN job cycles | acquire phase cycles | B-GM wait cycles | 理想消除全部 B-GM wait 的阶段提前 cycles |
|---|---:|---:|---:|---:|---:|
| 1 | 18 | 1,922,396 | 292,265 | 202,109 | 202,109 |
| 2 | 17 | 1,942,894 | 311,659 | 216,074 | 216,074 |

这些是固定 starts、忽略共享资源反馈的理想预算，不是实现收益预测，也不能跨层相加。末层 matrix/scatter phase 分别为 1,032,470 / 475,877 cycles；它们包含必须的编程/搬运工作和等待，不能全部当作可删空闲。

本轮只检验一个软件假设：原 FFN 的 GM 权重读按物理槽号 0→3 提交，轮转后可能使未来 tile 先于当前消费需求进入队列。`--ffn-demand-order` 改为按当前消费 sequence 开始的顺序提交；当前 demand 被拒时，暂缓当次未来 B 提交。槽数仍为 4，所有 GM/HBM 读、C spill/restore、阵列容量、operand context、数值累加顺序均不变。只作用于 Gate/Up/Down，不影响 QKV/Wo/LM。

新增实际 `demand_order`、`b_head_bypass_admissions`、`b_head_bypass_cycles` 计数。开启模式必须没有未来请求抢先于同一轮待提交 demand 的已接受请求；计数为零不表示没有 GM 等待，也不保证降低 cycles。带延迟、重排 HBM 回复、GM 回压和槽位回绕的生产代码测试验证顺序及数值；独立 trace 检查拒绝开关未生效、作用范围外溢或 demand 被越过。

实验门槛在结果产生前固定：正式 16-worker FFN micro 至少改善 0.3%，且输入、同构建、数值、稠密工作量和资源检查通过，才执行完整候选。关闭组仍需完整端到端确认新库默认路径复现旧周期。候选采用还需同时改善同构建完整端到端和 decoder，并优于原 selected。筛选不通过的候选不升级到全链路。

原始证据与执行状态：`results/ffn_critical_path_20261025/`。`baseline_budget.json` 为诊断；`micro_comparison.json` / `formal_comparison.json` 为配对结果（仅实际完成的范围生成）。硬件保护、初始源码及原库分别保存在 `hardware_preservation.json`、`initial_state/`。

复核预算：

```bash
python3 baseline/attention_cluster_8qk_8pv/analyze_ffn_critical_budget.py --case results/optimization_20261024/p7_restored_a/reuse_window_probe --output results/ffn_critical_path_20261025/baseline_budget.json
```

## 已完成的筛选结果

| 正式 16-worker FFN micro | cycles | 相对关闭组节省 cycles |
|---|---:|---:|
| 同构建关闭组 | 1,967,583.5 | 0 |
| FFN GM demand order | 1,967,583.5 | 0 |

两组使用同一 guest、SST 和 memhierarchy 二进制，输入/数值/资源审计 PASS，唯一环境差异为 `GOLEM_DECODER_FFN_DEMAND_ORDER=0/1`。384 条 window flow 记录除模式字段外全部相同，完整 GM resource trace 逐字节相同；两组均未出现计数定义内的 demand 被未来请求抢先提交。该候选在这个正式负载上没有改变实际请求时间线，收益为零，未达到 0.3% 门槛，不进入完整候选仿真。这个结果否定的是特定顺序假设在当前负载上的收益，不是证明整个 FFN 已无优化空间。

补充矩阵编程核对：当前每 worker FFN 的实际物理 group launch 数为 24,576，编程 callback 累计 958,464 cycles（平均 39 cycles/次），两层关键作业占比分别约 49.86% / 49.33%。它是当前调度的已测成本，不是所有软件映射的通用硬件下界。`compute_array_cycles=52,690,944` 是逐阵列时间之和，不能当作 FFN 延迟。原始核对在 `matrix_service_budget.json`。

2026-10-26 已完成新库关闭路径完整端到端回归。候选未运行完整端到端，不能将其 micro 结果填成候选端到端实测值，也不能将理想等待预算填成实测收益。


## GM 流量与固定硬件约束

`critical_gm_intervals.json` 对基准每层最后完成 worker 的作业内请求逐一核对，使用区间并集避免同类请求的重复计时。第二层本地 GM 服务区间并集为 1,777,241 cycles，占该 worker 作业 1,942,894 cycles 的约 91.47%；这不是可消除等待，也不等于所有 bank 的利用率。排队区间与其他请求服务、阵列计算相互重叠，不能相加。

固定硬件 `LocalGmBankScheduler` 有 compute/control 和 DMA 两个入口 FIFO，每个入口每周期最多发出一个 128-byte beat；8 banks 并不表示 compute 入口每周期可发 8 beats。`GlobalMemoryImplement::enqueueBankedLocalAccess` 将 DMA 分配到入口 1，其余到入口 0。当前每个关键 FFN 作业的 compute 本地读为 203,948,032 bytes，对应实测 1,593,344 cycles，恰为 bytes / 128；其中 B 槽读为 100,663,296 bytes / 786,432 cycles。compute 本地写另有 28,674 cycles。服务已发生的搬运不因调换预取顺序而消失。

这些数据将后续软件研究收敛到“降低实际进入 compute GM 入口的 bytes / beats，或降低现有映射的重复编程次数”。继续尝试之前，必须先给出可在现有 operand context、容量和 FP16 累加边界内实现的复用方案，并证明它实际删掉了请求；不能通过增加端口、槽位、阵列 context 或绕过定时读接口获得软件收益。本轮没有实现新的复用方案，也没有宣称该方向已有周期收益。

## 最终验收与保留配置

| 测量范围 | 历史默认 cycles | 新库关闭路径 cycles | 节省 cycles |
|---|---:|---:|---:|
| 完整端到端 | 6,084,167.75 | 6,084,167.75 | 0 |
| 两层 decoder | 5,343,797.75 | 5,343,797.75 | 0 |
| 第 1 层 FFN span | 1,938,881 | 1,938,881 | 0 |
| 第 2 层 FFN span | 1,960,384 | 1,960,384 | 0 |

Attention、最终 Norm 和 LM head 范围也逐项精确复现。末层 Y、最终 Norm、全部 128256 logits 与历史默认的 FP16 位差均为 0；数值、稠密 GEMM 覆盖、projection panels、DMA 生命周期、Ramulator2 后端和资源检查 PASS，untimed tensor bytes 为 0。原始外层 runner 中断后，使用这次完整仿真的原始 MPI streams、work 下的 guest stdout 和 HBM dumps 补齐归档并调用原有验证器，未重造输出。恢复记录为 `resumed_validation.json`。

脚本侧 279 项、生产代码侧 180 项测试通过。相对本轮初始快照，只修改现有 `decoderOpJob.h` 和 `reuseWindowGemm.h` 两个软件调度文件；未新增硬件组件，未修改阵列、GM、WCP、SFU、NoC、HBM 实现或资源配置。候选保留为默认关闭的 `--ffn-demand-order` 实验开关，不采用为性能优化。

当前 [SELECTED_PREFILL_2026-10-26.json](SELECTED_PREFILL_2026-10-26.json) 保留原软件调度和全部原 flags，指向本轮关闭路径实测结果及新库哈希。`run_selected_prefill.py` 启动前检查 guest 和 library 哈希。原 2026-10-24 配置作为历史证据保留，其原库哈希未修改。新库默认回归通过只说明实验代码关闭时等价，不代表候选带来端到端收益。

汇总为 `results/ffn_critical_path_20261025/summary.json`。本轮结论：诊断与单变量实验均完成；实测没有周期收益，默认延迟没有增加。仍有潜在软件研究方向，但尚无经过实验验证的下一项收益方案。
