# GPU-Competitive FlashAttention Plan

## Goal

以 E3/E4、`B=1,H=1,D=128`、FP32、non-causal Attention 为固定负载，先让
SST accelerator completion latency 优于 RTX 5060 FP32 Scope A，再考虑多头
Attention 和 MoE。MPI 只服务于缩短仿真 wall time，不作为 simulated cycles 优化。

完整方案与测量口径见
[GPU_COMPETITIVE_ROADMAP.md](src/sst/elements/golem/tests/small/muticore_attention/GPU_COMPETITIVE_ROADMAP.md)。

## Current Evidence

- SST E3 baseline：`3,561,012` 1 GHz normalized cycles；GPU FP32：`97,568`；
  当前差距 `36.50x`。
- SST E4 baseline：`14,169,968` 1 GHz normalized cycles；GPU FP32：`345,984`；
  当前差距 `40.96x`。
- 单头 E3/E4 数值、lifecycle、1/2/4-rank MPI 和本地构建已经完成。
- V-tile staging/direct-hit、Local GM 端口及 MPI 粒度实验已经完成并保留为历史证据；
  不进入当前正式 baseline。

## Completed Foundation

- [completed] 本地 SST/Golem 和 RISC-V guest 构建闭环
- [completed] fused/online/scale Attention 收敛为唯一 FlashAttention 路径
- [completed] E3/E4 单头 FP32 正确性、非物化和 lifecycle baseline
- [completed] query-group MPI 1/2/4-rank 数值、分区和 wall-time 验证
- [completed] PV V-tile reuse、Local GM direct-hit 和端口敏感性实验
- [completed] RTX 5060 Scope A/B/C/D 计时边界审计与 E3/E4 对照
- [completed] 明确暂缓 tile-level MPI、多头 Attention 和 MoE

## Active Phases

- [completed] A. 审计 SST clock/timebase，并冻结 GPU/SST 共同测量契约
- [completed] B1. 建立 E3/E4 SST critical-path 和 slowest-worker work breakdown
- [completed] B2. 实现 GPU FP32 Scope A/stage-chain benchmark、JSON schema 和导入报告
- [completed] B3. 在外部 RTX 5060 主机执行 GPU stage benchmark 并导入原始样本
- [completed] B4. 参考 `tilelang_three_limitations` 审计 GPU 端到端时间打点口径
- [paused] C. 运行 control/DMA/Local GM/program/SFU/compute/residency 理想化上界实验
- [pending] D. 根据归因移除 simulator 中可证明的人工串行
- [pending] E. 只实现有数据支持的 K/V residency、QK/PV array separation 或 tile overlap
- [pending] F. 扫描 array、Local GM、NoC/HBM、SFU 和 clock 的资源平衡
- [pending] G. 通过 E3 `<97,568`、E4 `<345,984` 的同精度 GPU gate
- [pending] H. GPU gate 通过后扩展多头 Attention
- [pending] I. 多头稳定后再单独评估 Top-1 MoE 和 MPI token dispatch

## Next Step

B4 结论：当前 Scope A 的 CUDA Event 边界能准确表示单次、warm-cache GPU device
timeline latency，足以确认 SST 约 40x 的量级差距；它不是包含 H2D/Python/D2H 的
application end-to-end，也不是强制 HBM cold-cache 的结果。E3 单批次波动较大，不能
作为最终 GPU gate 的唯一统计样本。

当前唯一下一步是在 RTX 5060 上按参考项目方法补做三次独立批次，并同时保留两种
模式：isolated per-sample synchronize（单请求 latency）与 continuous enqueue/final
synchronize（饱和 device timeline）。显式记录 SM clock 和 cache state，再用分层
bootstrap 汇总。Phase C 在用户决定前继续暂停。

## Overlap Continuation

- [completed] 为 QK 非转置和转置首 panel 增加逐阵列 input-program -> compute overlap
- [completed] 为 QK early-compute 增加 ELI 参数、统计、统一 runner 开关和 verifier/report 输出
- [completed] 保持 QK 后续 panel、Softmax、PV 和输出顺序不变，并通过同一 binary 对照
- [completed] Q256、Q1024 单 MPI 及 Q1024 MPI2 数值/lifecycle/placement 回归
- [completed] 提前释放 V Local-GM buffer，并将非广播 PV matrix programming 改为 16 路有界提交
- [blocked-by-model] QK readout 暂不并发化；2-credit ROCC 实验仍受单端口 buffer/Local-GM 写回语义阻塞
- [completed] 在 ComputeArray buffer transfer 层增加 output-read credit/bank ownership 和冲突统计
- [completed] 基于底层模型完成 1/1 与 2/2 QK readout 资源对照；无端到端 cycle 收益，不启用额外并发状态机

历史迁移过程和实验错误保留在 `progress.md` 与 `findings.md`，不再占用 active plan。

## Current Errors

| Error | Attempt | Resolution |
|---|---:|---|
| 旧 `/tmp/fused_attention_*` CSV 缺少后来新增的 VTileBuffer 零值统计，当前 verifier 拒绝重解析 | 1 | 已改用含完整统计的后续 baseline artifacts，E3/E4 均重新验证通过；未放宽正式 verifier 契约 |
| 最终可复现性检查命令因包含临时文件 `rm -f` 被安全策略拒绝 | 1 | 改用固定 `/tmp/phase_ab_repro_report.json` 并保留文件，不再执行删除 |
| measured GPU 合约测试夹具的 raw samples 与沿用的旧 Scope A median 不一致 | 1 | 将 synthetic median/normalized cycles 同步为 `0.04 ms/40,000`，保持严格 median 校验 |
| GPU 摘要引用的 `/home/jiajun/attention_gpu/gpu_attention_rtx5060_stages.json` 属于另一台主机，当前 SST 主机和首次上传附件中均无该文件 | 1 | 用户随后上传完整 121 KB JSON；全部 raw samples 已通过 evidence gate 并导入，B3 completed |
| 用 `apply_patch` 导入附件时，首次新增补丁因源文件末尾无换行而缺少独立终止标记 | 1 | 显式补入补丁换行后重新添加；导入 JSON 语义与附件一致并通过报告器 |
| B3 最终报告复现时 `/tmp/phase_ab_final_e3.json` 生命周期临时文件已被系统清理 | 1 | 后续确认原始 CSV 同样位于临时目录，最终改为在 E3/E4 baseline 中固化最小 verified lifecycle summary；不再依赖 `/tmp` |
| 尝试重生成 lifecycle 时发现对应 `/tmp/vtile_repeat_*` 原始 CSV 也已清理，仓库仅保留综合报告 | 1 | 从已经过 verifier 的 checked-in 综合报告提取最小、可复现 SST lifecycle summary，并固化到 E3/E4 baseline 目录；报告生成不再依赖 `/tmp` |
