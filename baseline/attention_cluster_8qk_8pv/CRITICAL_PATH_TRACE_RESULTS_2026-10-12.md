# 批次 A+C：端到端关键路径追踪结果

日期：2026-10-12

## 运行范围

使用正式配置 `Hq=32, Hkv=8, D=64, hidden=2048, causal+RoPE`，冻结基线：64 arrays、单 operand context、output retention、输入/本地权重预取均关闭、`array_groups=1`。推荐读取流水版本另行比较，不能与本表 baseline 混同。追踪由 `GOLEM_CRITICAL_PATH_TRACE=1` opt-in 开启，不改变调度和数据流。

运行产物：

- [关键路径汇总 JSON](../../results/critical_path_trace_20261012/critical_path_analysis.json)
- [S1024 产物](../../results/critical_path_trace_20261012/formal_v4/baseline/s1024/)
- [S2048 产物](../../results/critical_path_trace_20261012/formal_v4/baseline/s2048/)
- [验证日志](../../results/critical_path_trace_20261012/validation_logs/)

## 已加入的观测

`ProjectionJob` 在每个 worker 上记录带 `event_id/predecessor` 的 phase start/done 事件，并把 phase 映射到 `local_gm_read`、`local_gm_write`、`array_matrix`、`array_input`、`array_compute`、`array_output`、`sfu` 或 `control` 资源。已有 Attention milestone 记录继续提供 descriptor accept、worker complete、manager wait 和 root tensor 边界。离线分析器会检查 16 个 worker 的事件链、阶段配对、Projection worker timing、Attention milestone 完整性，并用 `chain_start_cycle + 最后 worker_complete` 重建 E2E。

## 基线重建结果

| 序列长度 | 报告 E2E | 追踪重建 E2E | Projection | Projection→Attention handoff | 关键 worker | worker 启动跨度 | worker 完成跨度 | 事件数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 573,772 | 573,772 | 416,538 | 2,019 | 16 | 30,701 | 32,489 | 375,440 |
| 2048 | 1,354,890 | 1,354,890 | 801,164 | 3,794 | 18 | 30,701 | 31,977 | 750,768 |

两组结果均为 `PASS`，HBM、数值和既有 projection report 校验保持通过。S1024/S2048 的事件链分别包含 187,712/375,376 个完整 phase pair；日志中的最后 `idle` 起始事件是 `retire()` 的终止标记，分析器将其作为合法终点处理。

## 当前可以确认的事实

1. worker 的启动偏斜约 30.7k cycles，且两个序列长度基本相同。这是跨 worker release/accept 数据流的明确观测对象；它是否位于最终关键路径，还需要 Batch B 的零延迟 oracle 或 release 实验确认。
2. Projection 完成到 Attention 第一批 descriptor 的 handoff 为 2,019/3,794 cycles，随序列长度增加而增加。这个等待边界已经可以作为后续跨阶段流水实验的基准。
3. 资源 phase 总量必须按并行 worker 解读，不能直接相加当作 E2E。比如 S1024 的累计 `array_compute` phase 为 1,671,168 cycles、`local_gm_read` 为 1,792,512 cycles，但这些工作分布在 16 个 worker 上；它们只说明服务量，不证明某个资源独占最终关键路径。
4. 当前关键 worker 上最长的单个观测 phase 是 SFU `norm_wait`，单次约 1,747 cycles。它在多个 tile/block 上重复出现，需要结合 producer-not-ready 与资源服务时间进一步拆分，不能仅凭 phase aggregate 判定 SFU 是瓶颈。

## 观测边界

本批次记录的是不改变调度的 phase 级资源时间线和已有 Attention milestone。phase 时间戳在下一次 worker tick 观察到状态转换，精度为 tick；它可重建 E2E 和阶段边界，但还没有把 DMA 请求的 queue wait、物理 service start、cache-visible 和 consumer consume 分成四个独立事件。下一轮应使用同一事件格式加入这些资源内部边界，再运行 Batch B 的七类 oracle和 2³ 矩阵。

上述为第一版的观测边界。后续资源内部时间已经补齐，更新内容、物理边界与模型预约的区别、验证产物及批次 B 含义见 [FINE_RESOURCE_TIMELINE_2026-10-12.md](FINE_RESOURCE_TIMELINE_2026-10-12.md)。资源区间的补齐不等于完整跨层 DAG/slack 已实现。
