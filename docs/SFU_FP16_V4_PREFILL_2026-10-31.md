# SFU FP16 v4 主工程接入（2026-10-31）

主工程 `src/sst/elements/golem/sfu` 已同步实验区通过验收的 v4 实现。SFU 默认内部精度为 FP16，所有加法、乘法、归约、online softmax 状态和 RMSNorm 状态按 binary16 RNE 舍入；FP32 仍只能通过 `GOLEM_SFU_INTERNAL_PRECISION=fp32` 显式选择。

本次接入包括 v4 行级 RMSNorm 流水和双 FP16 补偿字、直接 SiLU FP16 ROM（65536 项、131072 bytes）、EXP/倒数/平方根 ROM 后端，以及 Row/Vector pipeline 的 buffer、beat、调度和共享仲裁参数。主工程架构生成器默认使用 `GOLEM_SFU_NONLINEAR_BACKEND=lut`。

实验区完整 v4 Prefill 证据位于 `Muticore-Attention-all-gemm/baseline/attention_cluster_8qk_8pv/progress_results/2026-10-31/sfu_silu_v4/validation.json`，9/9 PASS，覆盖 1/3/16 层、MPI、保留/乒乓生命周期路径。Prefill 模拟范围为 token 输入到 logits 的完整运行；K/V 仅为本次运行的临时张量，不声明持久 KV cache、跨请求复用或 Decode ABI。

`docs/SFU_FP16_INTEGRATION_2026-10-29.md` 保留为历史记录，其中的旧 prefill FAIL 不覆盖本次 v4 结果。
