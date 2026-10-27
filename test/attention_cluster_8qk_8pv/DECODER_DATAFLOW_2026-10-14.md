# 通用 GEMM 输出 panel、预取与首个分块融合

按 Wo 验证 → FFN 覆盖 → 预取 → 分块融合的顺序实现。保留 16 workers、每 worker 64 阵列、单 operand bank、2 MiB GM、原 GM/NoC/HBM 带宽和 FP16 张量舍入规则。

## 实现

### 输出 panel

`DecoderOpJob` 默认使用最多 16 个相邻 64-column tile，形成最多 `64 rows × 1024 columns` 的 FP16 panel。两块 panel 各 128 KiB，交替计算和写回。

阵列输出原本为 tile-major。先通过定时 GM read 读取 panel，按行重排，再通过定时 GM write 写入同一 panel 槽位，才进行 DMA。这样每行输出从 128 B 请求变成最多 2 KiB 的连续请求，最多 8 个 DMA write 在途。重排引入的额外 GM 读写没有跳过计时。

新 tile 可以在另一 panel 中计算。正在 drain 的槽位只有全部 HBM ACK 收齐才允许覆盖；尾 panel 按实际 tile/row 数发送。job 的完成标志和后续层 release 必须等所有 panel ACK。M=1 的词表列条带不是连续块，保持原列条带写回，避免错误合并。

### 预取

可选预取路径在两份独立 8 KiB GM 权重槽之间交替：先将当前权重读出，然后把下一 K tile（含跨 N tile）通过真实 HBM→GM DMA 和真实 local GM read 准备好，同时 program/scatter/MAC 当前 tile。下一权重只有当前计算完成后才写入阵列 operand。

当前输入的 local GM read 与 matrix programming 重叠。后台读取使用独立状态，不覆盖前台 read/write callback；请求被端口拒绝时重试。没有额外阵列 operand bank、额外 GM 端口或无计时 tensor 读取。

### 分块 SiLU × up

新增 `DecoderOp::SiluMultiply`，每块最多 8192 元素。gate/up 都已由各自线性 job 完成写回；融合 job 读取这两个输入，先提交真实 SFU SiLU 请求，再将其 FP16-rounded 输出提交给真实 Multiply 请求，仅把 gated 结果写回 HBM。

保留 EXP、add-one、scalar reciprocal、multiply 的共享 SFU 竞争以及 SiLU 输出的 FP16 舍入点。消除 `act` 的写回和回读，正式规模每层少 32 MiB，两层少 64 MiB。gate/up producer 和 down consumer 本轮仍是独立线性 job；没有宣称完成整个 FFN 的片上融合。

融合校验使用独立 scalar C++ SiLU oracle，执行 FP32 `EXP → add → reciprocal → multiply → FP16`。原 NumPy 商式与这一顺序可能差一个 FP16 舍入单位；未放宽 gated 的零误差标准，也没有改变设备计算。

## SRAM 布局

| 区域 | GM offset | 大小 |
|---|---:|---:|
| 最大 K=8192 的输入 | 0x10000 | 1 MiB |
| 前台权重/向量系数 scratch | 0x110000 | 依当前算子复用 |
| 输出 panel 0 | 0x112000 | 128 KiB |
| 输出 panel 1 | 0x132000 | 128 KiB |
| 预取权重 0 | 0x152000 | 8 KiB |
| 预取权重 1 | 0x154000 | 8 KiB |

最高使用地址为 `0x156000`，从 GM 起点计约 **1.336 MiB**，小于 2 MiB。所有常量在 `decoderMemoryLayout.h` 统一管理。向量 job 与线性 job 串行，向量临时输出和已读取的系数延续原有 scratch 复用；不存在仍在 drain 的 linear panel 被向量输出覆盖的情况。

## Wo 正式规模对照

两组均 S=1024、hidden=2048、MPI rank=8，完全相同输入和 Attention cycle。

| 指标 | 原 128 B 串行写回 | 输出 panel |
|---|---:|---:|
| Attention cycles | 488,962.25 | 488,962.25 |
| Attention 最后 O ACK → Wo 最后 ACK | 660,544.5 | 459,999.5 |
| 首次输入 DMA → Wo 最后 ACK | 1,149,506.75 | 948,961.75 |
| 每 worker 输出 DMA 请求数 | 2,048 | 128 |
| Wo reference 最大误差 | 0 | 0 |

post-attention 减少 **30.36%**；这段包含 Attention 的端到端减少 **17.45%**。Wo 张量逐位一致。

同一 guest/library 的详细结果与输入 hash 对照见 [decoder_dataflow_wo.json](progress_results/2026-10-14/decoder_dataflow_wo.json)。

Wo 基线的 SST 和数值验证已完成，但旧 MPI verifier 不接受 rank=8，导致其后续报告流程中断。复用既有仿真输出补做正确的 rank=8 placement、HBM layout 和 DMA/数值检查，没有再次跑基线。MPI verifier 现在直接复用 architecture 的 rank=8 query-block placement 规则，回归测试与真实 rank CSV 均通过。

## 小规模两层完整链路对照

S=1024、hidden=256、FFN=1024、词表=8201，包含 timed Embedding、两层 decoder、final RMSNorm、最后 token LM head。四组使用同一 guest binary 和 SST library，MPI rank=8；输入 preloads 的实际文件 hash 相同。

| 方案 | 总 cycles | 相对原路径减少 | 两层线性输出 DMA 请求数 | SST wall seconds |
|---|---:|---:|---:|---:|
| 原路径 | 1,109,385.875 | — | 81,920 | 44.151 |
| 输出 panel | 750,280.625 | 32.37% | 8,192 | 39.169 |
| panel + 预取 | 681,630.625 | 38.56% | 8,192 | 38.112 |
| panel + 预取 + 分块融合 | 655,743.625 | 40.89% | 8,192 | 39.212 |

预取相对 panel 再减少 **9.15%**；融合相对预取再减少 **3.80%**。两层输出、Wo/h1/Norm2/gate/up/gated/down、final RMSNorm、logits 逐位一致，greedy token 均为 5313。panel 与预取保持 HBM 字节量不变；融合两层共少 8 MiB，符合小规模 act 张量的写/读大小。wall 为各一次观测，融合未继续降低 wall，不能把设备 cycle 改善当作相同幅度的仿真 wall 改善。

详细结果：[decoder_dataflow_small.json](progress_results/2026-10-14/decoder_dataflow_small.json)。

## 正式规模两层完整链路

S=1024、hidden=2048、FFN=8192、词表=128256，包含 Embedding、两层 decoder、final RMSNorm 和最后 token 的 LM head；MPI rank=8。仍使用合成权重，所有 dense tile 都执行。

| 指标 | 优化前归档 | 本轮 panel | 本轮 panel + 预取 + 融合 |
|---|---:|---:|---:|
| 总 cycles | 20,167,497.75 | 15,453,844.75 | **10,210,612.75** |
| 1 GHz 延迟 ms | 20.167498 | 15.453845 | **10.210613** |
| 相对归档总 cycle 减少 | — | 23.37% | **49.37%** |
| 第一层 post-attention cycles | 8,795,572.5 | 6,428,749.25 | 4,003,090.25 |
| 第二层 post-attention cycles | 8,793,658 | 6,458,898 | 4,084,662 |
| LM head cycles | 1,341,425.75 | 1,341,262.75 | 893,458.75 |
| 两层线性输出 DMA 请求数 | 655,360 | 40,960 | 40,960 |
| SST wall seconds | 1,377.199 | 1,299.043 | 1,180.949 |

本轮 panel → 全部优化是同一 guest binary、同一 SST library、实际输入 hash 相同的对照，进一步减少 **33.93%** 的端到端 cycle。正式规模没有单独跑 pref-only，故这 33.93% 是预取与融合的合计收益，不能分别归因；小规模四组对照提供了各步独立收益。优化前归档使用旧 binary/library，只作为历史对照，不能称为同二进制 A/B。

两层输出、最后层 Wo/h1/Norm2/gate/up/gated/down、final norm、logits 的 FP16 bit patterns 与归档逐位一致；greedy token 均为 **99037**。panel 本身没有减少 HBM 数据字节，输出请求减少 **93.75%（16 倍）**。融合两层减少 32 MiB read 和 32 MiB write，合计 **64 MiB**，与消除 act 的往返完全一致。优化版 decoder read/write 合计分别为 3,659,530,240 / 142,606,592 B；完成 490,112 次权重预取，输出在途高水位为 8。

两层都由 core16 最后完成，其第一层局部 job 时长如下。这些是单 worker 的 job duration，包含资源等待，不是全局 stage window：

| 第一层 core16 job | panel cycles | panel + 预取 + 融合 cycles |
|---|---:|---:|
| Wo | 438,980 | 296,129 |
| gate | 1,686,055 | 937,290 |
| up | 1,680,073 | 964,121 |
| SiLU 与 Multiply（相加 / 融合） | 780,424 | 714,687 |
| down | 1,686,662 | 918,728 |

正式 panel 与全部优化仿真分别绑定 CPU 8–15 和 0–7，并发运行了一段时间；上述 wall 是各一次观测，受到共享宿主资源影响。不能用它们与旧归档直接推断独立运行的 wall 加速比，也不能把 cycle 的 49.37% 当作 wall 改善比例。

结果、来源 hash 和逐位对照见 [正式历史对照](progress_results/2026-10-14/decoder_dataflow_formal.json) 与 [同 binary/library 的增量对照](progress_results/2026-10-14/decoder_dataflow_formal_incremental.json)。两组分别通过 [576-job panel 资源核对](progress_results/2026-10-14/decoder_dataflow_formal_panel_resources.json) 和 [544-job fused 资源核对](progress_results/2026-10-14/decoder_dataflow_formal_fused_resources.json)：DMA 字节与 job counter 对齐，无任何在途 GM interval 越过 job 完成。

## 开关与复现

```bash
source scripts/env_local_install.sh
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/decoder_dataflow_new/formal \
  --shape formal --embedding --mpi-ranks 8 \
  --gemm-prefetch --silu-multiply-fusion
```

- `--output-panel-tiles 16`：默认；范围 0..16，0 恢复旧写回。
- `--output-write-window 8`：默认；范围 1..16。
- `--gemm-prefetch`：默认不启用，可独立 A/B。
- `--silu-multiply-fusion`：默认不启用，可独立 A/B。
- 单层 `run_decoder_layer.py` 也支持 panel、window、prefetch 和 rank=8，可先以 `--checkpoint wo` 验证。

原有 HBM region、层间标志和 descriptor 96-byte ABI 保持兼容，新增 op=7 表示 fused SiLU/Multiply。用于比较的 `summarize_decoder_dataflow.py` 验证输入 hash、各层输出和 logits 的逐位一致性，分别记录 binary/library 是否相同，避免把旧归档与新同 binary 对照混为一谈。

## 验证

- 当前 baseline 分析/host 检查 123 项通过；attention/SFU/guest 检查 150 项通过；architecture 检查 6 项通过，共 279 项。日志保存在 `progress_results/2026-10-14/decoder_dataflow_*tests.log` 和 `decoder_dataflow_architecture_checks.log`。
- 生产 DecoderOpJob 的测试覆盖尾行、尾 panel、M=1、延迟/乱序 DMA ACK、延迟权重到达、GM 端口拒绝重试、panel 复用、SFU 请求重试、FP16 中间舍入及融合没有 act DMA。
- 资源 sidecar 检查器逐 job 对齐 read/write bytes，确认预取和 drain 无 GM 请求跨越 job 完成。小规模 fused 320 个、正式 panel 576 个、正式 fused 544 个 decoder/embedding/head/norm job 全部通过。
- SST elements 及 guest 已重编译；所有真实仿真继续使用 Ramulator2，diagnostic oracle=none。

后续更大范围 FFN 融合应围绕 gate/up panel producer 与 down 的 partial accumulation 调度设计，并重新核算有限 SRAM 与共享阵列占用。本轮的优化结果可作为其独立基线。
