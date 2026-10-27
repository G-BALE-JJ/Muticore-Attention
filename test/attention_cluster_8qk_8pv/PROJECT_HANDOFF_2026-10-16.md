# Muticore-Attention 项目交接文档

更新日期：2026-10-16。本文供新会话接手当前 SST/Golem LLM prefill 项目使用。

## 1. 目标与边界

项目目标是在 SST/Golem 中实现并验证 Llama 风格的 prefill 数据流，统计每阶段
cycle、GM/HBM 流量、资源冲突和数值误差。当前使用合成权重，采用
hidden=2048、FFN=8192、词表=128256 的两层正式形状。真实 checkpoint、完整层数、
tokenizer、文本解码和自回归 decode/KV-cache 尚未接入；GPU RTX 5060 基线已有交接
资料但尚未实测。

用户约束：

- 每 worker GM 固定 2 MiB，不扩大容量；
- 不随意增加 array、bank、端口、带宽、C-buffer 或 SFU；
- 只做已有硬件上的软件数据流调度；
- 先局部筛选，再做完整端到端验收；
- 保留有效证据，避免重复 HBM backing、build 和 MPI 日志撑爆磁盘。

## 2. 代码状态

当前分支为 `softmax-update`，HEAD 为 `9d7bd12`。工作树包含大量尚未提交的
后续改动（banked GM、完整 decoder、层数/序列泛化和验证脚本），不能只 clone
GitHub 后假定代码相同。接手先执行：

```bash
cd /data/jjgong/LLM/Muticore-Attention
git status --short
git log -5 --oneline
```

未经确认不要 `git reset`、`git clean` 或覆盖未提交改动。

## 3. 当前硬件合同

| 资源 | 当前模型 |
|---|---|
| 核心 | 4 managers + 16 workers（通常 core 4–19） |
| ComputeArray | 每 worker 64 个 array，64×64 tile GEMM；每 array 一个 operand context |
| Worker GM | 2 MiB，固定 |
| C-buffer | 1 MiB |
| HBM | 正式配置 4 个数据节点，每节点逻辑 2 GiB；容量不等于带宽 |
| SFU | 共享 vector、EXP、倒数流水，执行 RMSNorm、Add、SiLU、Multiply、RoPE |
| GM banked | 8 banks×256 KiB、128 B line；计算/控制和 DMA 两个有界入口，返回队列有界 |

GM banked 模型依据参考工程 HDL 的 bank、握手和仲裁关系校准，但参考 HDL 不是
完整 SST cycle-exact 规范，不能机械复制所有参数。详细时序见
[`GM_TIMING_CONTRACT.md`](GM_TIMING_CONTRACT.md) 和
[`GM_ALIGNMENT_VALIDATION_2026-10-16.md`](GM_ALIGNMENT_VALIDATION_2026-10-16.md)。

## 4. 计算图和硬件映射

```text
token IDs → Embedding
→ 每层 RMSNorm1 → QKV GEMM → RoPE → causal GQA Attention
→ Wo GEMM → 残差1 → RMSNorm2
→ Gate/Up GEMM → SiLU × Up → Down GEMM → 残差2
→ final RMSNorm → 最后一个 token 的 LM head GEMM
→ host greedy argmax
```

- QKV、Wo、Gate、Up、Down、LM head：WCP 调度 ComputeArray 做 tiled GEMM；
  M=1 LM head 仍复用 GEMM 路径。
- Embedding：HBM 查表/gather，不走通用 GEMM。
- Add、Multiply、RoPE、RMSNorm、SiLU：SFU/vector pipeline。
- QKᵀ、PV：Attention cluster 内部矩阵乘法，不属于 decoder `Linear` 层。
- HBM 保存权重；GM 保存 panel、输入、部分结果和控制状态；GM 不保存完整权重。

关键代码：

- [`decoderOpAbi.h`](../../src/sst/elements/golem/attention/decoderOpAbi.h)
- [`golem_decoder_runtime.h`](../../src/sst/elements/golem/tests/small/muticore_attention/golem_decoder_runtime.h)
- [`decoderOpJob.h`](../../src/sst/elements/golem/attention/decoderOpJob.h)
- [`projectionJob.h`](../../src/sst/elements/golem/attention/projectionJob.h)
- [`computeArray.h`](../../src/sst/elements/golem/array/computeArray.h)
- [`decoderMemoryLayout.h`](../../src/sst/elements/golem/attention/decoderMemoryLayout.h)
- [`vector_math.h`](../../src/sst/elements/golem/sfu/vector_math.h)

## 5. 已完成的功能与优化

已完成完整两层链：

- worker 侧 Embedding；
- QKV、RoPE、causal GQA Attention；
- Wo、残差、两套 RMSNorm；
- Gate/Up、SiLU、Multiply、Down、第二残差；
- final RMSNorm；
- M=1 LM head、词表分块、host greedy argmax；
- 两层统一 layout、activation ping-pong、epoch/reset、DMA ACK、scratch retire；
- 逻辑 S=1–8192 入口，非整块 S 向上补齐到 1024 倍数。

已验证的软件数据流优化：

1. GEMM 输出 panel 写回；
2. 4 槽权重预取（2/4/8 中正式选 4）；
3. Gate/Up panel 与 SiLU×Up 融合；
4. Down partial 与第二残差融合；
5. 两个 32-array 软件组交错，仍复用原 64 arrays；
6. LM head N 维并行，正式选 `lm-n-parallel=4`；
7. GM banked 时序、DMA landing、bank 冲突、返回背压和 ready 发布。

这些是软件调度和布局变化，没有新增硬件资源；每 array 仍只有一个 operand
context，所有 GM/array/SFU 流量和 ACK 必须计入。

## 6. 性能口径

必须区分以下结果：

| 口径 | cycle | 用途 |
|---|---:|---|
| legacy GM 历史优化结果 | 6,231,011.75 | 旧时序合同，不能作为当前硬件基线 |
| 通用性改造后的结果 | 7,424,562.00 | 旧的 prefill/GPU 交接锚点 |
| **当前权威 banked 对齐结果** | **7,483,267.75** | 当前正式两层验收 |

当前权威配置：S=1024、两层、Hq/Hkv=32/8、head dim=64、MPI8、四槽预取、
FFN 两组、down+residual fusion、LM N 并行度 4。计时范围是首个 token-ID DMA
到最后 logits DMA ACK；1 GHz 下约 7.483 ms。SST wall 约 1,161.668 s，只表示
宿主仿真成本，不与 GPU 毫秒直接比较。

权威 banked 阶段 cycle：

| 阶段 | cycle |
|---|---:|
| Embedding | 27,730.000 |
| Embedding→L1 交接 | 55,903.000 |
| L1 Norm1/QKV/RoPE/Attention | 546,598.875 |
| L1 Wo/残差/Norm2/FFN/残差 | 2,776,402.500 |
| 层间交接 | 52,523.625 |
| L2 Norm1/QKV/RoPE/Attention | 536,731.875 |
| L2 Wo/残差/Norm2/FFN/残差 | 2,765,787.500 |
| final norm 交接 | 15,525.625 |
| final RMSNorm | 54,307.750 |
| LM head 交接 | 9,855.250 |
| 最后 token LM head | 641,901.750 |

阶段按 DMA ACK 边界可加；内部 worker job span 有重叠，不能把所有 job 时长相加。

当前 banked 参照：

- MAC 峰值下界 1,970,152 cycles，实际/下界约 3.798×；
- 最忙 GM 入口下界 4,470,405 cycles，实际/下界约 1.674×；
- HBM payload 下界 3,562,969.275 cycles，实际/下界约 2.100×；
- 有用 MAC/峰值约 26.33%；计算入口 grant 忙碌 59.10%–59.74%，DMA 入口 29.59%–30.42%；
- bank 冲突 110,512，返回队列阻塞为 0。

下界可以重叠，不能相加。banked 相对 legacy 增加约 20.097% 是时序模型校准影响，
不能称为软件优化回退。

## 7. 验收状态和数值限制

当前合成两层 banked 验收通过：

- greedy token=**99037**；
- 层输出、final norm、logits 与 legacy 结果 FP16 逐位一致；
- flags、epoch、reset、scratch retire、层间同步、词表块 ACK 通过；
- 416 个 decoder jobs 的 DMA/GM 生命周期审计通过；
- 20 个 GM 均为 2 MiB，结束时 pending/queued 清空；
- 张量区未计时直访为 0 B；
- banked GM 微测、参考 RTL 对照、局部筛选、MPI 一致性和正式两层验收均通过。

数值是 FP16 存储/传输及既有 FP32 内部路径。组合 Attention 使用独立高精度参考，
不能宣称整链与 FP32 oracle bit-exact。合成权重结果不能代表真实 checkpoint 的任务质量。

## 8. 复现入口

```bash
cd /data/jjgong/LLM/Muticore-Attention
source scripts/env_local_install.sh

python3 baseline/attention_cluster_8qk_8pv/run_prefill_validation.py \
  --artifact-root results/my_prefill_validation --stage all --cpu-set 0-7

python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --shape formal --embedding --mpi-ranks 8 --gm-model banked \
  --gemm-prefetch --prefetch-slots 4 --ffn-down-fusion \
  --down-residual-fusion --ffn-panel-tiles 16 --ffn-array-groups 2 \
  --ffn-layout 2 --lm-n-parallel 4 --artifact-root results/new_formal_case
```

GM 验证入口：

```bash
python3 baseline/attention_cluster_8qk_8pv/probe_gm_timing.py \
  --output results/gm_alignment_20261016/production_probe.json
python3 baseline/attention_cluster_8qk_8pv/run_gm_reference_rtl.py \
  --artifact-root results/gm_alignment_20261016/reference_rtl
python3 baseline/attention_cluster_8qk_8pv/verify_gm_banked.py \
  results/gm_alignment_20261016/accepted_small
```

优化流程必须是：小尺寸筛选 → 单 worker 正式形状 → 16 worker/MPI8 竞争 →
只对优胜者做两层完整验收。

## 9. 产物、GPU 和下一步

大规模仿真应使用 `artifact_storage.py` 的 compact policy；调试才使用
`GOLEM_ARTIFACT_POLICY=full`。验证完成且不再需要张量复核时，才对单个 case 执行
`artifact_storage.py finalize <case> --discard-tensors`。不要删除源码、manifest、
provenance 或正在运行的 backing，也不要复制整个 build/install/results。

GPU 交接见 [`GPU_PREFILL_HANDOFF_2026-10-16.md`](GPU_PREFILL_HANDOFF_2026-10-16.md)，
它冻结了输入、权重 shape、QKV/GQA/RoPE、计时边界、误差指标和 transfer manifest。
RTX 5060 首轮应复现同一合成两层链，权重常驻显存，用 CUDA events、多次样本记录
阶段和完整链延迟；不要改 SST 硬件或输入。

推荐下一步：

1. 以当前 banked GM 重新做 FFN GM 重复读、Down 输入复用和 Wo→残差→Norm2 行级流水的局部实验；
2. 检查 LM N 并行在 banked GM 下是否仍有收益；
3. 完成 RTX 5060 baseline；
4. 之后再考虑真实 checkpoint、完整层数、原生尾行和 decode/KV-cache。

任何方案都要重新检查 GM 2 MiB 高水位、单 operand context、SFU 共享、bank/入口/
返回背压、DMA ACK 和 epoch/reset；若需要新增硬件资源，先报告并等待评判。

## 10. 证据入口

- [`GM_ALIGNMENT_VALIDATION_2026-10-16.md`](GM_ALIGNMENT_VALIDATION_2026-10-16.md)
- [`PREFILL_GENERALIZATION_2026-10-16.md`](PREFILL_GENERALIZATION_2026-10-16.md)
- [`FFN_GM_FLOW_2026-10-16.md`](FFN_GM_FLOW_2026-10-16.md)
- [`PROGRESS_2026-10-14.md`](PROGRESS_2026-10-14.md)
- [`progress_results/2026-10-16/acceptance_manifest.json`](progress_results/2026-10-16/acceptance_manifest.json)
- [`progress_results/2026-10-16/gpu_prefill_handoff/transfer_manifest.json`](progress_results/2026-10-16/gpu_prefill_handoff/transfer_manifest.json)
- 参考工程：`/data/jjgong/RISC-V-CIM-Manycore-SST`，重点为 HDL README 和 GM RTL。

新会话先阅读本文、GM_ALIGNMENT_VALIDATION、PREFILL_GENERALIZATION 和 GPU_PREFILL_HANDOFF，
再检查工作树与 provenance。不要把 6,231,011.75 legacy cycle 当作当前 banked 基线。
