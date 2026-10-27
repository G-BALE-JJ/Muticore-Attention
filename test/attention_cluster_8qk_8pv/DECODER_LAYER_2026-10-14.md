# 单层 decoder prefill 接入（2026-10-14）

## 容量结论

每 worker 2 MiB GM 是当前 baseline 的配置，不是 GlobalMemory 实现规定的最大值。
GM 是本地片上 SRAM，HBM 是外部内存，二者分别配置。增加 HBM 容量不会扩大 GM，
增加 GM 容量也不会自动增加 GM 端口数或带宽。

- `architecture/cpu_builder.py` 通过 `GOLEM_GLOBAL_STRIDE_BYTES` 或
  `GOLEM_GLOBAL_STRIDE_KB` 计算 GM 窗口并传给 GlobalMemory 的 `baseAddr`/`size`。
- 正式 runner 固定 `--global-stride-kb 2048`，guest 中也使用 2097152 字节步长。
  改容量必须同步 guest、RoCC 中涉及 GM 地址的常量以及 CPU identity 映射。
  例如 20 核 × 4 MiB = 80 MiB 已超过默认 64 MiB 的 identity 起点，不能只改一项。
- 正式 runner 固定 5 个 SST 内存节点，每节点 128 MiB；node 0 给 OS，node 1–4
  给数据，所以当前数据空间共 **512 MiB**，总物理地址空间 **640 MiB**。
- SST 架构支持 `--mem-node-size` / `GOLEM_MEM_NODE_SIZE_BYTES` 和
  `GOLEM_PHYS_MEM_SIZE`；但当前 Attention/Projection guest 的节点步长、预加载和
  校验也按 0x08000000 布局，不能只给 backend 改 `mem_size`。

当前 `architecture/ramulator/hbm2e_2500.yaml` 有 **8 个 controller**。
`org.count = [1,2,1,4,4,16384,128]`；地址映射跳过 Channel，Column 扣掉
prefetch=4，transaction=32 字节。因此每 controller 在规定 Row 数内的地址范围为
`2 × 1 × 4 × 4 × 16384 × (128/4) × 32 = 512 MiB`，8 个 controller 合计
**4 GiB / Ramulator backend**。这是该配置的映射范围，并不是目前 SST 数据区容量。
SST `memoryController.cc` 在交给 backend 前将全局地址转换为节点本地地址。
因此扩大 SST 节点到 1 GiB 仍在这个范围内，通常无需修改 YAML 的 Row 或 controller 数。
若超过几何范围，应调整 Row 等组织参数，并重新检查地址映射和时序合理性；
增加 controller 数还会改变并行带宽，不能把它当成纯容量变化。

完整模型 FP16 权重约 **2.302 GiB**。建议未来按 4 个数据节点各 1 GiB 分布，
共 4 GiB 数据空间，留出激活、KV 和对齐空间。这里假定权重均衡分片存储，
不能把整个模型按当前 QKV 的方式复制到每个节点。若 5 节点继续等大，系统总空间是
5 GiB（含 OS 节点）。这是后续配置目标，本次测试保持 **512 MiB HBM / 2 MiB GM**。
单层权重可容纳于现有 HBM，无需先扩容才能实现 Wo/FFN。

## 新增计算链

```text
原始 X → 原有 RMSNorm/QKV/RoPE/causal GQA Attention → O[token,hidden]
                                                       ↓
                                              Wo Linear(hidden,hidden)
                                                       ↓
                                              H1 = X + Wo(O)
                                                       ↓
                                              RMSNorm(H1, gamma2)
                                                ↙          ↘
                                       gate Linear       up Linear
                                            ↓                │
                                          SiLU ──────────────×
                                                       ↓
                                           down Linear(intermediate,hidden)
                                                       ↓
                                               Y = H1 + down
```

新增 `attention/decoderOpAbi.h` 和 `attention/decoderOpJob.h`。
RoCC 0x26 提交通用算子，0x27 等待其最后写回 ACK；manager 0 在所有 Attention
head 完成后释放 16 个 worker，worker 按既有 4×4 行分工执行后续算子。
最后一个算子写回自己的 HBM 完成标志，root 等待全部 16 个标志。

Linear 支持 K/N 为 64 的倍数且不超过 8192。每次保留 64 行输入，按 K tile
整理到有限 GM，64 个既有阵列分别处理一行，使用单一 operand context。
权重按输出 64-column tile 在四个 HBM 节点轮转分片；每次只搬一个 8 KiB
权重 tile。输出按每个 tile 的 64 行逐行 DMA 写回，每次等待 ACK；这是便于验证的
串行基线，后续可单独优化写回批量与权重复用，不需要先做跨算子融合。
K=8192 时本地输入为 **1 MiB**，包括权重/输出/元数据仍能放入
2 MiB GM。所有 tensor DMA、GM local read/write、阵列 programming/scatter/compute
均调用既有计时接口，不在 host 用矩阵乘法代替设备计算。

Attention O 本身是行主序 `[token,Hq,Dh]`，无需额外 head 拼接。
原 QKV job 的 `hidden_dim <= 2048` 检查和 head/panel 输出语义不能用于
FFN `8192→2048`；通用 Linear 的 K 和 N 独立，绕开的是专用 ABI 的限制，
不是扩大硬件阵列尺寸。

Add/Multiply 复用 SFU vector pipeline；SiLU 是已有 exp、reciprocal、vector
pipeline 的保守串行复合时序。SiLU 一个算子调用完成自己的定义，未融合前后
Linear 或 gate/up 乘法。各中间张量均在 HBM 保留，可分别读取和验证。
SiLU 的时序是当前复合模型，不是已校准的最优微架构。

## 数值和范围

- 继承现有 Projection 精度：FP16 权重、输入、逐乘积、dot 和 K-tile 累加，
  最后 FP16 写回；Attention 仍沿用既有内部精度。
- FP16 helper 会将绝对值小于 2^-24 的 FP32 输入截为带符号零，和严格 IEEE
  转换在最小 subnormal 附近有区别。独立 golden 显式复现这一现有行为。
  本次没有借机修改全局数值契约。
- 第二次 RMSNorm 暂用当前合成 workload 的 epsilon=1e-6；真实 checkpoint
  的 epsilon、RoPE 排列/常数以及权重加载仍需后续统一。
- 当前 runner 限定 **B=1, S=1024, Dh=64, hidden=256 或 2048,
  intermediate=4×hidden**。不是任意序列长度的完整模型部署。
- 后续 decoder 权重是固定种子的非平凡稀疏合成权重，每行涉及多个 K tile，
  包括最后一个 input column；硬件模拟仍执行所有密集 tile，无零跳过或周期折扣。
  验证的是接线、分片、维度、数值与写回，不是实际 Llama checkpoint 的语义质量。
- 现有 QKV/Attention 审计 buffer、HBM 区域和旧 baseline guest 均保留。
- 仍未完成 16 层调度、真实权重输入、最终 RMSNorm/LM head/token 生成。

## 使用

```bash
source scripts/env_local_install.sh
make -C src/sst/elements/golem/tests/small/muticore_attention \
  decoder-wo decoder-residual decoder-ffn decoder-layer
# 先构建并安装包含新 RoCC job / SFU vector 支持的 SST 插件。
python3 baseline/attention_cluster_8qk_8pv/run_decoder_layer.py \
  --artifact-root results/my_decoder_layer --checkpoint layer --shape formal \
  --reference results/startup_release_20261012/clean_formal_s1024
```

检查点是 `wo`、`residual`、`ffn`、`layer`。`--shape small` 使用 hidden=256，
`formal` 使用 hidden=2048。`--verify-only` 从既有 HBM dump 重做验证，不重新模拟。

每阶段报告所有 worker 最早开始到最晚完成的 span；worker 之间允许阶段重叠，
所以 **不能把这些 span 相加**。开启 GM resource trace 时另报告从首个输入 data
DMA send 到最终 checkpoint 输出 ACK 的完整区间，包含阶段 handoff，排除 ACK
后的完成标志等待/打印/退出。最终输出必须有完整且无重复的 ACK 地址覆盖。

原模拟中的 Attention 数值/HBM 布局已通过。首次 Wo 小规模旧报告因新增 decoder
标志被误计入 Projection barrier 而拒绝；修正后对同一日志重建报告通过。FFN/layer
小规模首次 host golden 未复现极小 FP16 截零行为，修正后对相同 HBM dump 重验通过。
这些修正没有重算或替换设备输出。

测试结果和 cycle 表见 `results/decoder_layer_20261014/summary.md`，每个 case 的
`decoder_result.json` 包含逐阶段误差、16-worker 完成情况、DMA 字节和周期。
