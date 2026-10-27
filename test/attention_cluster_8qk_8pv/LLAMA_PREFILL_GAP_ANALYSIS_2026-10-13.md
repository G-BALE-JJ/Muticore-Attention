# Llama 完整 prefill 功能缺口与实施顺序

日期：2026-10-13。基于当前工作树与已完成正式测试；本次未实现新算子、未运行新的 SST 性能实验。

## 建议与目标

先完成一个固定模型、固定输入形状、真实权重、可逐阶段验证的最小完整 prefill，再优化跨算子融合。保留当前已经验证的 Attention 流水和 RMSNorm/QKV 等融合；不必为了做基线把它们拆回去。

“先功能闭环”不等于先实现全部推理产品功能。首个目标建议明确为：Llama-3.2-1B、B=1、S=1024、无历史 KV、16 层前向、返回最后一个有效 token 的 logits，必要时在 host 用 argmax 得到首个生成 token。暂不扩展 decode、连续 batching、复杂采样、所有模型与任意形状。

功能闭环需要有界缓冲、真实数据依赖、DMA 完成可见性和一致精度。为适配 FFN/LM head 做分块属于必要实现；在功能版本里就应保留算子与传输的基础计时，不必等融合阶段才补观测。

## 当前实现完成到哪里

正式验证链路是：HBM 中合成 X/gamma/投影权重 → 融合 RMSNorm → QKV → RoPE → causal GQA/FlashAttention → HBM Attention O。

证据：

- [LLAMA_PROJECTION_BASELINE.md](LLAMA_PROJECTION_BASELINE.md) 明确排除 Wo、残差和 MLP。
- [projection_case.py](../../src/sst/elements/golem/tests/small/muticore_attention/projection_case.py) 使用 seed=1742 生成合成输入、gamma 和权重，没有接入 checkpoint。
- [golem_attention_runtime.cpp](../../src/sst/elements/golem/tests/small/muticore_attention/golem_attention_runtime.cpp) 当前 guest 运行 Projection→Attention 后结束，没有完整 decoder layer 循环。
- [vector_math.h](../../src/sst/elements/golem/sfu/vector_math.h) 的专用向量请求支持 RoPE 和 RMSNorm。
- [sfu.cc](../../src/sst/elements/golem/sfu/sfu.cc) 的 primitive 验证和求值确实支持 Sigmoid；未发现完整 Llama FFN、残差 ADD 或 SwiGLU 二元 MUL 的已验证作业路径。内部 Softmax 的加乘不等于已经暴露了这些通用作业。
- [golem_matmul_runtime.h](../../src/sst/elements/golem/tests/small/mvm_noc_int_array/golem_matmul_runtime.h) 与相邻 `.cpp` 有通用 M/N/K、FP16/FP32、B 转置的 GEMM 基础。它属于另一个 workload/runtime 路径，尚未证明能直接复用当前 Attention guest 的 core 分配、描述符和内存布局。

当前 Attention 输出按 query-major `[S,Hq,D]` 写回，在 Hq×D=hidden 时可以视为 `[S,hidden]`。接 Wo 前需验证全局分片、stride、head 顺序；不应预设还要物化一个新的 concat 张量。

## 单层 decoder 的缺口

设 `h=2048`、`i=8192`。以下矩阵形状按 `X @ W` 表示；checkpoint 通常按 `[out,in]` 存权重，导入时要处理转置和 tile packing。

```text
x ── RMSNorm₁ ── QKV ── RoPE ── causal Attention ── Wo ── +x ── h₁
                                                                   │
                h₂ ← +h₁ ← Wdown ← [SiLU(gate) × up] ← RMSNorm₂(h₁)
```

| 功能 | 当前状态 | 完成内容 |
|---|---|---|
| RMSNorm₁、QKV、RoPE、Attention | 已有合成用例验证 | 对齐 checkpoint 参数、数值精度和权重布局 |
| Wo 输出投影 | 未接入 | `[S,2048] @ [2048,2048]`，输出同 token 分片 |
| 第一残差 ADD | 未发现完整作业实现 | 保存原始 x，计算 `h₁=x+Wo(attention)`；不能用 norm(x) 替代残差输入 |
| RMSNorm₂ | 算子已有，未接 FFN | 对 h₁ 运行，用该层独立的 post-attention norm weight |
| Wgate | 未接入 | `[S,2048] @ [2048,8192]` |
| Wup | 未接入 | `[S,2048] @ [2048,8192]` |
| SiLU 与门控乘 | Sigmoid primitive 已有；完整作业缺失 | `SiLU(gate)=gate*sigmoid(gate)`，再逐元素乘 up；需要二元 MUL、分块、读取和写回 |
| Wdown | 未接入 | `[S,8192] @ [8192,2048]`，跨 K tiles 保留正确 partial accumulation |
| 第二残差 ADD | 未发现完整作业实现 | `h₂=h₁+Wdown(SiLU(gate)*up)` |

Llama FFN 的激活是 SiLU/SwiGLU，不是 GELU；FFN 不是两次矩阵乘法，而是 gate/up/down 三次。RMSNorm₂ 也不是新增一种 norm 算子，而是复用已存在功能并接入正确权重和数据源。

完整层的结构与 [Hugging Face Llama 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) 一致：Attention 带 o_proj，层内两次 norm 和两次 residual，MLP 为 `down(act(gate(x))*up(x))`。

## 从单层到一次完整模型 prefill

| 功能 | 当前缺口 | 第一个闭环的范围 |
|---|---|---|
| checkpoint/config 导入 | 当前是合成张量 | 导入各层 norm、QKV、Wo、FFN、final norm、embedding，生成可复现 manifest 与 tile 文件 |
| tokenizer/输入准备 | 当前 guest 输入是 hidden state | 可先在 host 用模型 tokenizer 生成 token IDs；记录是否包含 BOS/chat template 与有效长度 |
| Embedding lookup | 未接入模型链 | gather 得到 `[S,2048]`；初期可 host 提供 embeddings，但报告必须明确设备窗口从 embeddings 开始 |
| 16 层调度 | 当前为单段单层子链 | 层权重切换、上一层输出→下一层输入、ping-pong hidden buffers、scratch 生命周期和全局完成 |
| Final RMSNorm | 算子可复用，未接输出层 | 使用独立 final norm weight |
| LM head | 未接入 | `[1,2048] @ [2048,128256]`，仅取最后有效 token；tied embedding 复用同一逻辑权重 |
| 输出 token | 无 logits | host argmax 即可作 greedy 首 token；不需要先实现 vocab Softmax/top-p |
| 逐层数值基准 | 当前参照合成 NumPy 算术 | checkpoint 前向参考和硬件算术参考分开；验证真实 hidden state、logits、首 token |
| 新端到端计时 | 旧边界止于 Attention O | 完整设备路径应止于 final logits ACK；分开 tokenizer/准备、计算、完成通知 |

仅计算最后一个 token 的 LM head 已受参考实现的 `logits_to_keep` 支持；无需为了首 token 输出物化 `[S,128256]` 的全序列 logits。若目标改为每个位置的概率或打分，需要另加全位置输出。

对单次无历史 prefill→logits，持久 KV cache 不是正确性的前置条件，可以暂缓跨调用接口。若 prefill 要交给后续 decode，就必须按层保存 RoPE 后 K 与 V，并维护长度、地址和布局。当前 raw K/V panel、局部 RoPE cache 与测试中的临时 HBM 输入不能自动视为可供 decode 使用的模型 KV cache；无需把 GQA K/V 复制成 32 个 head。

## 需要在真实权重接入初期处理的约束

### 1. 当前 QKV 作业不是通用 Linear

[projectionJob.h](../../src/sst/elements/golem/attention/projectionJob.h) 校验 `hidden_dim<=2048`，描述符直接绑定 Q/K/V heads、panel 地址和 norm/rope。FFN down 的输入维度为 8192，不能直接套此接口；FFN up/gate 输出 8192 和 LM head 输出 128256 也需要通用输出描述与分块。

建议先调查现有通用 GEMM/WCP 路径，形成可重复调用的 Linear 作业与稳定 ABI，而非把 FFN 伪装成带 dummy heads 的 QKV。Wo、gate、up、down、LM head 共享这个基础。最先验证的真实尺寸至少包括 `(M,N,K)=(S,2048,2048)`、`(S,8192,2048)`、`(S,2048,8192)` 和最后 token 的 `(1,128256,2048)`。

最后 token LM head 的 M=1 还可能需要边缘 tile/padding：通用 runtime 当前要求维度可被 block 整除，不能认为已有 GEMM 入口就已经覆盖这个尺寸。

### 2. 模型容量与现有地址 ABI

当前正式 `attention_hbm_layout.json` 为 node1–4 各 128 MiB，数据节点合计 512 MiB；GM 为每 worker 2 MiB。这是当前模拟容量，不是宿主 RAM 容量。

按项目 Llama-3.2-1B 尺寸且 embedding/lm_head 权重共享：

- 参数约 1,235,814,400 个，FP16 约 **2.30 GiB**。
- 单层 QKV+Wo+FFN+两份 norm 权重约 **116.01 MiB**。
- embedding / tied LM head 权重约 **501 MiB**。
- S1024 的一个 `[S,8192]` FP16 FFN 中间张量为 **16 MiB**；S2048 为 32 MiB。

因此完整 FP16 权重不能直接常驻当前数据 HBM；必须明确采用更大模拟 HBM 容量还是有真实传输语义的分层/分块供给。更改 HBM 容量时，当前写死的 node stride、数据偏移和计时 reporter 地址筛选也需一起迁移。选择 streaming 时，不能由 host 无成本覆盖活跃 HBM 权重来声称完整部署延迟。

完整容量选择影响性能分母，应先确定；此阶段不必追求最优权重预取，但需要不覆盖 residual、KV 和未消费中间结果的内存规划。

### 3. 模型语义并未因形状一致而自动匹配

- 当前 RMSNorm 合成参考、guest 和 fused request 使用 epsilon=1e-6；项目模型笔记记录目标 Llama-3.2-1B epsilon=1e-5。实际接入以 checkpoint config 为准，不能只改 reference。
- 当前 RoPE 表默认 base=10000，采用相邻偶奇 interleaved 配对；模型笔记记录目标 theta=500000。还需按实际 config 处理 RoPE scaling/position IDs，并匹配 checkpoint 的 split-half/interleaved 权重约定。可以选择转换 Q/K 权重布局或适配 kernel，但必须保持等价。
- 当前 Projection 在 [mvmComputeArray.h](../../src/sst/elements/golem/array/mvmComputeArray.h) 用 `GolemFp16` 进行 dot 与 tile accumulation；合成 reference 同样模拟 FP16 舍入。对照真实模型时须固定精度合同，优先评估 FP32 accumulation/FP16 output 的可行性与误差；不能用当前合成 PASS 保证 16 层 checkpoint 输出质量。
- 当前 Projection host generator 和 guest 限制 S 为 1024 的倍数。第一阶段可固定 S1024；面向任意文本部署时还要支持有效长度、padding/mask 和最后有效 token 索引。

以上属于正确性适配，不能延后到算子融合阶段。

## 推荐实施顺序与验收

1. **真实模型接口与算术合同。** 确定一个 checkpoint、dtype、RoPE/epsilon 与内存容量方案；导出真实第一层参数和 PyTorch/HF golden。确认现有 QKV/Attention 在真实输入下逐阶段误差可接受。
2. **单层完整 decoder。** 建立 Linear 作业，补 Wo、ADD、RMSNorm₂、gate/up、SiLU/MUL、down、第二 ADD。先小规模检查资源与数值，再真实 h2048/i8192/S1024。目标输出是正确 h₂；保留独立阶段边界。
3. **多层真实 prefill。** 在同一运行里串接 2 层验证权重切换、job generation 与缓冲重用，再扩到 16 层；补 embedding、final norm、last-token LM head。最小终点为 logits；greedy argmax 可在 host。
4. **冻结基线后做融合。** 记录完整模型及分阶段 cycles、实际数据搬运、数值误差和容量合同，再选择高收益边界，如 Wo+residual、residual+norm、SiLU+gate、FFN 的 tile 消费流水。每项与冻结基线作同输入/同资源对照。

先完成单层已足以观察 FFN 的真实份额；未跑通 16 层前可以修正明显实现缺陷和内存越界，但不再根据现有 Attention 子链比例推动新一轮大范围融合优化。

已有 host tokenizer/embedding 可用于初期混合闭环；只要关键层结果仍在 host 计算，就应标明哪些阶段实际在 SST/设备模型执行，不把 host 参考计算计成设备算子零 cycle。

## 调查边界

本报告核对当前 Attention guest、Projection/SFU 数学与 ABI、通用 matmul runtime，以及正式 HBM 布局。对“未接入”的判断是当前 Llama workload 没有实现和验证，不等于所有底层组件都没有可复用能力。没有下载模型权重，也没有据此声称新增功能已经完成。
