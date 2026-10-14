# 以首个有效数据 DMA 为起点的端到端口径

本结论依据用户定义的数据传输/计算范围，以及 `/data/jjgong/RISC-V-CIM-Manycore-SST` 的现有计时实现。它替代前一份计时审计报告将 release→root 建议为主指标的建议；前一份报告的原始事件与数值仍成立。此次只分析并重新计算归档数据，未改模型或执行新的优化实验。

## 参考项目实际统计什么

1. `src/sst/elements/golem/tests/stats/extract_latency_csv.py:352`：`gemm_system_start_cycle = gemm_descriptor_start_cycle or gemm_wcp_start_min_cycle`；终点为所有 WCP 的最大 end，延迟为 end−start+1。
2. `tests/small/mvm_noc_int_array/gemm_matmul_op_ctrl.h:672`：descriptor_start 来自 worker 描述符同步屏障，发生在构造/写入 task-list header、启动 WCP 之前。因此它不是首个向 HBM 发出的数据 DMA 时间，确实覆盖一部分准备成本。
3. `workercmdproc/workercmdproc.h:313`：WCP start 是 busy 状态下的首个 tick；`WRITEBACK_WAIT` 先排空 pending writeback，再进入 `DONE`，记录 workerEndCycle。终点包含写回完成检测和 DONE 状态推进，发生在写回 ACK 之后。
4. 同一解析器的 `total_cycles` 是各 worker 的 total 平均值，当前归档也明确输出 worker_avg_total_cycles。它不是整个算子的首尾跨度，不能直接拿来与 Attention E2E 比较。

例如 `artifacts/gemm_256_2048_3072/stats/overlap0/run_20261010_191713_356032/execution_summary.csv`：worker 平均 total 为 73,090.666667；descriptor_start=245,465、最早 WCP start=247,443、最后 WCP end=323,346，system latency=77,882。两个指标有不同分母与边界。

参考项目值得沿用的是：全系统取最早起点/最晚终点，不能取 worker 平均；完成必须包括写回；准备、执行、统计生命周期应分开。其描述符屏障起点与用户此次定义不同，不机械照搬。

## 最终主口径

**主指标：本次完整 workload 的首个有效 HBM 数据读 DMA 实际发出，至完整最终输出 O 的最后一个写回 ACK 返回。**

```text
guest 启动 / 参数解析 / release / 描述符准备
                  ↓（单列，不计入主指标）
首个有效数据 DMA 成功送入 NoC                         最后 O 写回 ACK
                  |——数据传输、计算、同步、等待、流水排空——|
                                                          ↓
                                                 root 完成通知 / guest 输出 / exit
```

公式使用同一 SST 时间基：

`T_data = (max(final_O_write_ack_tick) − min(workload_data_read_first_send_tick)) × reference_clock_hz / ticks_per_second`

这里不加 1。两个事件时间戳相减表示经过的时间；参考工程对离散 worker cycle 标签使用的 inclusive +1 是另一种计数约定。归档保存原始 ticks，报告时间及按明确 reference clock 换算的 cycle，保留分数精度；需要整数时只对完整时长统一向上取整，不分别舍入两个端点。HBM 时钟、array 时钟、CPU cycle 不直接混加。

### 起点如何落实

`GlobalMemoryImplement::mark_dma_read_sent()` 在 `link_control->send()` 成功后记录 `first_send_cycle = getCurrentSimCycle()`；若队列拥塞，直到首次成功发送才记录。当前 Attention 的 `dma_read_transport.start_tick` 就使用这个时间，而 `submit_tick` 是 DMA 创建/提交时间。虽然字段叫 cycle，`getCurrentSimCycle()` 在这里是 SST 核心时间基 tick，必须按时间基换算。

按用户“实际发出请求开始传输”的定义，选择 **first_send**。它包括后续 NoC 路由、HBM 排队和服务、响应、落入 local GM 等开销。不能以 HBM 控制器实际发出 DRAM READ 命令为起点，否则会漏掉请求传输与 HBM 队列等待；也不能等首批数据到 GM 才开始，否则漏掉第一波访存。

创建请求至首次成功发送的等待单列为 `initial_dma_admission_delay`；它不进入此主口径。若以后研究 DMA 提交/注入调度，可补充 submit→O ACK 作为辅助指标，避免把初始注入等待迁移到主窗口外后宣称全面加速。

“首个”必须覆盖所有执行核/内存节点以及本次 workload 的所有必要输入数据，包括 RMSNorm gamma、X、投影权重和必要 RoPE 表；不是仅 X，也不是仅最慢 worker。允许实际预取时，预取的首次发出同样计入，不能把它叫准备而挪到窗口外。地址/请求类别与 workload 归属要明确，排除程序/运行库装载、描述符和 mailbox/flag 读取、测试输出读取。不能对整个仿真所有 DMA 无条件取最小值。

当前范围是 RMSNorm→QKV Projection→RoPE→Attention→O，不是预先给定 Q/K/V 的纯 Attention kernel。若另测后者，必须另立 workload 名称、首个 Q/K/V 数据 DMA 与输入驻留假设。

### 终点如何落实

对本次 workload 的最终 O 地址范围，覆盖所有 query/head、全部输出分块和四个输出 HBM 节点，取成功写回 ACK 的最大时间。不是最后 MAC 结束、最后一次 write 提交或 guest printf。现有模型在收到 `DMA_WRITE_COMPLETE` 时记录 `dma_write_transport.done_tick`，已用于 `last_attention_output_ack_cycle`。

这里定义的是**模型承诺的写完成 ACK**。它不额外推断 DRAM 最后一条写命令的物理落盘时间；若要考察后端写 ACK 是否准确反映服务完成，属于内存模型语义审计，应另作验证。

辅助指标保留 `T_completion = root_tensor_complete − first_data_DMA_send`，衡量调用方硬件控制接口收到全局完成所需的延迟；`root−last_O_ACK` 是完成通知尾部。根据输出可用性评价主数据路径，用 T_data；根据调用接口返回时间评价，用 T_completion。两者标签不能都笼统写成 E2E。guest 打印与进程退出属于测试生命周期。

## 计入和排除的边界

首个数据 DMA **之前**的 CPU 启动、参数解析、任务 release、尚未伴随数据传输的描述符准备不计入 T_data；单独报告启动准备成本即可。

首个数据 DMA **之后**直至 O ACK 的所有自然执行等待继续计入，包括其他 worker 的迟启动、未完成的描述符准备、全局 flag barrier、HBM/GM 争用、array programming、scatter、MAC、SFU 和流水填充/排空。不能从墙钟跨度再扣去这些阶段。首个 worker 已开始数据传输后，其他 worker 的准备就是当前调度形成的执行空洞。

诊断打印若落在此窗口里并影响请求/争用，仍会改变 T_data。缩小统计范围不能排除这种因果干扰，应通过关闭诊断后的对照解决。上一轮发现的非 root manager 末尾打印问题因此仍值得验证。

初始状态须一致：本轮 cold data transfer 的输入在 HBM，必要 gamma/权重读取包括在内；不混用预热到 GM/array 的结果。理论下界也必须覆盖同一输入驻留假设、同一完整 workload 和 O 写回边界。

## 当前归档重新计算

两个 clean-startup IW 用例的最早有效读都是 worker 16 的 4,096B RMSNorm gamma 请求，地址 `0x09000000 = node1 + 0x01000000`。其 submit 与 first_send 都为 77,848,000 ticks，在 1GHz/1ps 时间基下为 77,848 cycles；它恰好等于旧报告 chain_start。不是由 release 时间推算，而是扫描原始 `dma_read_transport` 时间戳并确认数据地址得到。

| 指标（1GHz cycles） | S1024 | S2048 |
|---|---:|---:|
| 首个有效数据 DMA send（绝对时间） | 77,848 | 77,848 |
| 最后 O 写回 ACK（绝对时间） | 551,750.75 | 1,316,662.75 |
| **主指标 DMA→最后 O ACK** | **473,902.75** | **1,238,814.75** |
| 辅助 DMA→root 全局完成 | 473,926.5 | 1,238,838.5 |
| O ACK→root 通知尾部 | 23.75 | 23.75 |
| 历史 job→worker 完成 | 473,914 | 1,238,826 |

因此，原先 18,985 cycles 的 release→job 准备跨度不应加到用户定义的主指标。旧指标在当前两例的起点恰好正确，终点位于 O ACK 与 root 之间，需要按上面事件重新命名与固定。未来不能仅依赖这种巧合，必须真正记录首个有效数据 DMA。

机器结果：`results/dma_timing_contract_20261012/summary.json`。目前仍使用原 clean-startup guest，未关闭末尾成功打印；本次数值是重新解释边界，不是新的优化收益。

后续实现应添加独立 workload-aware 的 DMA 起止统计，不依赖是否开启 host trace；旧字段保留历史兼容、新字段明确命名。再执行末尾打印和 trace 开关隔离，所有实验及资源下界使用同一固定窗口。
