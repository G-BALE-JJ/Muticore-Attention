# Muticore-Attention 项目交接状态

> 最新状态（2026-10-14）：已验证 token IDs → 定时 Embedding → 两层完整 Decoder → final RMSNorm → 最后 token 的 LM head → host greedy argmax。正式 4/8 rank 输出完全相同，20,167,497.75 cycles；8 rank wall 从 1566.43 s 降到 1377.20 s。当前交接入口为 [PROGRESS_2026-10-14.md](PROGRESS_2026-10-14.md)，下文保留历史 Projection 实验记录。

> 2026-10-12 Projection 因果诊断：新增逐 tile 请求/阵列关联并重跑六个正式用例。前台 GM 读排队均为 0；慢 worker 的权重供给与阵列消费节拍失配。S2048 core 12：weight DMA 完整回调平均 198.62 cycles，而数据齐备稳态为 147 cycles/tile；未来 weight DMA 受软件单在途限制。input budget=1 在 S1024 回退，lookahead=8 两规模仅降 0.29%/1.06%，保持默认不变。详见 [Projection 因果分析](PROJECTION_CAUSAL_ANALYSIS_2026-10-12.md)。


> 2026-10-12 固化统计与下界：新增 `dma_e2e_result.json`，版本 `hbmdma_to_O_ack_v1`；主指标 S1024/S2048=473,902.75 / 1,238,814.75 cycles。当前 groups=1 与全局阶段屏障下的必要资源下界=352,816 / 967,776，尚非已证明可达到的最优调度；当前超出下界的差距约 86% / 95% 在 Projection。新入口默认采集时间戳并生成主报告，旧 job-to-worker 字段仅作历史兼容。详见 [固定口径与资源下界](DMA_LOWER_BOUND_2026-10-12.md)。

> 2026-10-12 DMA 计时口径澄清：按用户要求，主指标采用首个有效 HBM 数据 DMA 实际发送→最终 O 写回 ACK，排除此前启动准备；当前 clean IW 的 S1024/S2048 为 473,902.75 / 1,238,814.75 cycles。DMA→root 全局完成作为辅助指标。此结论替代下方审计更新推荐 release→root 为主指标的建议；末尾打印干扰仍待隔离。参考工程实际起点是描述符屏障，并非首个数据 DMA，详见 [最终 DMA 统计口径](DMA_TIMING_CONTRACT_2026-10-12.md)。

> 2026-10-12 计时审计更新：暂停数据流实验，已复查两个 clean-startup 正式归档。旧 E2E 漏计 release→job 的 18,985 cycles；后续主要执行指标改用 release→root 全局完成（492,911.5 / 1,257,823.5 cycles）。发现非 root manager 末尾成功打印可能与剩余计算重叠，尚待对照隔离；旧资源 floor 不是已证明的理论下界。详见 [端到端计时审计](E2E_MEASUREMENT_AUDIT_2026-10-12.md)。

> 2026-10-12 最新性能测量：定位到 guest 启动诊断 `fprintf` 使 worker release 串行错开。6 组正式实验全部通过，关闭三类启动诊断后 IW 的 S1024/S2048 E2E 为 473,914/1,238,826 cycles；仿真起点→最终 O ACK 为 551,750.75/1,316,662.75 cycles。以后性能工作采用 `make clean-startup` 构建的 guest；旧日志版本保留作历史复现。详情及 S2048 时序敏感性见 [启动实验与干净基线](STARTUP_RELEASE_2026-10-12.md)。这属于清理测量开销，不能包装成算法加速；下方历史数据均需区分 guest 版本。

> 最新实验（2026-10-11）：已实现64-array下一input pair预取、本地weight预取和现有阵列两组错开。六个正式A/B全部通过；推荐读取流水、array_groups=1，S1024/S2048 E2E为544,931/1,304,066 cycles，相对旧基线下降5.03%/3.75%。分组虽有编程/MAC重叠，却未优于读取流水，保持实验选项。140项Attention测试、23项baseline测试及原路径/慢GM/非融合多block回归通过。完整结果、复现与下一瓶颈见 [读取/编程/计算重叠报告](PROJECTION_READ_PROGRAM_COMPUTE_OVERLAP_2026-10-11.md) 和 [机器汇总](projection_overlap_results.json)。下文第3–4节数值为优化前对照。

> 2026-10-11 后续更新：用户已决定舍弃 32-array weight-stationary；其可执行分支已移除，旧开关为1时拒绝运行。本文第6–7节保留为历史交接内容，不再作为下一任务。最新分析与实验次序见 [舍弃WS后的重分析](LATENCY_REANALYSIS_WITHOUT_WEIGHT_STATIONARY_2026-10-11.md)，严格隔离结果见 [WS隔离实验](WEIGHT_STATIONARY_ISOLATION_2026-10-11.md)。正式64-array基线原本就未启用WS。

日期：2026-10-11

本文用于新会话继续开发。当前工作树包含大量前序数据流优化修改，应以本文和验证结果为准。

## 1. 当前资源约束

目标是只优化软件数据流，不增加硬件架构：

- core 0..3 是 manager，只负责控制和数据分发，没有计算阵列和 SFU。
- core 4..19 是 worker，每个 worker 有 64 个计算阵列和 SFU。
- 每个物理阵列只有一个 operand context：一份 matrix context 和一份 input context。
- GM、SRAM、阵列数量、端口、带宽和输出累加器容量均不增加。
- `GOLEM_ARRAY_OPERAND_CONTEXT_BANKS=2` 和跨 tile operand pipeline 必须被拒绝。

## 2. 已验证的正式数据流

当前正式路径是单 operand context：

1. RMSNorm 可以与投影输入准备融合。
2. GM 中可以预取下一份输入和权重。
3. 阵列侧只能由前台按顺序编程唯一 operand context。
4. 64-array 路径使用 output retention：hidden tile 0 overwrite，后续 tile accumulate，最后统一读取输出。
5. 32-array 路径使用已有 C-buffer partial，保持原始 row-group-first 调度。
6. C-buffer/GM partial 可以后台读取，已有 partial ahead restore 逻辑可以启用。
7. Attention 使用 worker arrays，manager 不参与计算。

主要实现文件是 `src/sst/elements/golem/attention/projectionJob.h` 和
`src/sst/elements/golem/tests/small/muticore_attention/run_fused_attention_scale.sh`。

## 3. 端到端基线

形状为 Hq=32、Hkv=8、D=64、hidden=2048、causal+RoPE，输出与参考逐字节一致，
NumPy O 的最大绝对误差为 `0.0001220703125`。

| S | 投影（含融合 RMSNorm） | 交接 | Attention | 总 cycle | 理论下界 | 比值 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 416,538 | 2,019 | 155,215 | 573,772 | 268,592 | 2.14x |
| 2048 | 801,164 | 3,794 | 549,932 | 1,354,890 | 799,328 | 1.70x |

结果目录：`results/single_operand_context/verified/`；汇总文件为
`baseline/attention_cluster_8qk_8pv/single_operand_context_results.json`。

理论下界是乐观计算下界，包含 RMSNorm 向量计算、投影阵列 MAC 和 Attention SFU/exp，
不包含完整 GM/SRAM 搬运、权重编程、输入 scatter、partial 保存恢复、RoPE 和同步调度。

## 4. 当前主要瓶颈

| 投影阶段 | S=1024 典型值 | S=2048 典型值 |
|---|---:|---:|
| 阵列计算 | 104,448 | 208,896 |
| 矩阵编程 | 62,976 | 125,952 |
| 输入 scatter | 53,760 | 107,520 |
| 前台 Local GM 读取（weight+input+fusion） | 112,032 | 224,064 |
| 其中 weight GM 读取 | 55,296 | 110,592 |
| 权重 HBM/GM-cache 准备（load_weight，非 Local GM read） | 约 13,600 | 约 26,400 |

权重 GM lookahead 已有 1..8 tile 配置；深度 8 的回归中请求和命中均为 72 次，但总 cycle 没有下降，
说明继续扩大同一 head 内的 HBM→GM lookahead 没有收益；前台 Local GM 读取、矩阵编程和 input scatter 仍需分别分析。

## 5. 已完成的优化和验证

- 修正 manager/worker 资源归属。
- RMSNorm→投影融合、packed Norm input、RoPE 融合。
- GM 权重和输入预取。
- partial C-buffer/GM 后台读取及 partial restore ahead。
- 64-array output retention。
- 去除额外 operand banks。
- 权重预取 lookahead 参数 `GOLEM_PROJECTION_WEIGHT_PREFETCH_DEPTH`，范围 1..8。
- 固定策略：64 arrays 使用 output retention；32 arrays 默认使用 C-buffer partial。
- Attention 单元测试 135 项通过。
- baseline 测试 19 项通过。
- 单 operand 多分块、D128、融合 RoPE、非融合 Norm、慢 GM 等回归通过。

## 6. 未完成且已失败的优化

目标是 32-array weight-stationary：固定一个 hidden tile，依次处理两个 row group，
使用已有 C-buffer 保存 partial，从而把矩阵编程次数从 64 降到 32。

原型观察到 `weight_programs: 64 → 32`、`weight_reuses: 0 → 32`，但端到端输出错误，最大误差约 `0.318`。
关闭 RMSNorm、packed input 和 GM 预取后仍然错误，因此根因在 row-group 重排和 partial 生命周期。

该模式当前默认关闭：

```bash
GOLEM_PROJECTION_WEIGHT_STATIONARY=0
```

不能把原型的 cycle 或矩阵编程次数当作正式收益。

## 7. 下一会话的明确任务

只修复 32-array weight-stationary 的正确性，不再扩大 GM lookahead：

1. 为每个 row group 显式跟踪当前 row group、hidden tile、partial write issued、partial write ready、partial restore consumed。
2. 严格保证：

   ```text
   row group 0, tile t: launch -> read -> partial write
   row group 1, tile t: launch -> read -> partial write
   tile t+1, row group 0: partial restore -> accumulate
   tile t+1, row group 1: partial restore -> accumulate
   ```

3. 先关闭 pipeline、restore-ahead、融合 Norm 和 packed input，使用最小 32-array C-buffer 配置定位错误。
4. 逐步重新启用这些优化，每一步检查 Q/K/V/O HBM 输出逐字节一致。
5. 数值正确后比较权重编程次数、partial 次数、投影 cycle 和端到端 cycle。
6. 如果正确但没有端到端收益，保留当前 row-group-first 路径，不做运行时成本选择器。

## 8. 推荐验证命令

```bash
source scripts/env_local_install.sh
python3 -m unittest discover -s src/sst/elements/golem/tests/small/muticore_attention -p 'test_*.py'
python3 -m unittest discover -s baseline/attention_cluster_8qk_8pv -p 'test_*.py'
```

32-array 调试使用新的 artifact 目录，并显式设置：

```bash
GOLEM_PROJECTION_ARRAYS=32
GOLEM_PROJECTION_CBUFFER_PARTIAL=1
GOLEM_PROJECTION_WEIGHT_STATIONARY=1
```

正式结果不能覆盖 `results/single_operand_context/verified`。

## 9. 2026-10-12 后续观测工作

最新任务已补齐细粒度资源时间，见 [FINE_RESOURCE_TIMELINE_2026-10-12.md](FINE_RESOURCE_TIMELINE_2026-10-12.md)。这次只增加 opt-in 追踪，没有新增硬件或修改数据流；不继续扩大 weight-stationary 调试范围。正式冻结 baseline 使用 64 arrays、array_groups=1、两个读取预取关闭，推荐 reads 配置另行验证，避免混淆实验对照。

资源区间分析和原 E2E 重建已经支持，不代表完整跨层依赖 DAG 与 slack 已完成；后续必须保留真实数据依赖和资源竞争，再判断哪些等待值得优化。

## 10. 2026-10-12 批次 B

已完成 7 类受控 oracle 和完整 2³ 因子矩阵，S1024/S2048 共 30 次正式运行，输出均与冻结基线逐字节一致；详细范围、交互效应与重跑入口见 [BATCH_B_2026-10-12.md](BATCH_B_2026-10-12.md)。最新工作基于已移除 weight-stationary 的路径，不应使用第 8 节的旧调试开关恢复该机制。

默认 all-off E2E 保持 573,772 / 1,354,890 cycles；IW 为 544,931 / 1,304,066，分别改善 5.03% / 3.75%。三个开关全开反而慢于 IW。同步启动 oracle 节省约 30k cycles，但这不是已实现的 guest 优化，也不能直接与 IW 收益相加。

下一步分别验证启动错峰压缩、单 GM 读端口下的预取顺序；跨 head/tile 的阶段流水仍需完整依赖分析。两个 barrier oracle 仅测试全局就绪后的提交间隙和 ready 调度推进，完整跨资源 DAG/slack 尚未完成。
