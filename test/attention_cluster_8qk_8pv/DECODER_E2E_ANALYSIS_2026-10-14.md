# 两层 prefill 端到端 cycle、下界与优化优先级

本报告分析本轮数据流优化前的归档。输出 panel、预取与分块 SiLU/Multiply 融合后的实际结果见 [优化报告](DECODER_DATAFLOW_2026-10-14.md)，最新正式端到端为 10,210,612.75 cycles。下面的串行路径下界专指旧调度，不应用于已启用预取的路径。

结论：当前性能主要受通用 Wo/FFN 的串行数据流限制。先优化线性算子的权重预取和输出写回，再做有限 SRAM 分块的 FFN 融合。SiLU 的单 lane 倒数是另一项真实的计算约束，融合本身不会消除它。

## 测量范围与证据

- 正式 workload：S=1024，hidden=2048，FFN=8192，Hq/Hkv=32/8，D=64，2 层，词表 128256。
- token ID DMA → timed Embedding → 两层 decoder → final RMSNorm → 最后 token 的 LM head → logits 最后 ACK。
- FP16 张量输入/输出；RMSNorm 等内部数学精度沿用当前方案。
- 16 workers，每个 64 个既有阵列、单 operand bank，2 MiB GM、1 MiB WCP C-buffer；4 个数据 HBM node，每个 2 GiB。
- 是合成权重测试，阵列执行 dense tile 工作；不是已加载真实 checkpoint 的完整 16 层模型延迟。
- MPI rank=4/8 的 cycle、logits 一致。1 GHz 下 1 cycle = 1 ns。
- 使用已归档的 formal_rank4 结果和 GM resource sidecar；分析时未重新运行 SST。原有 provenance source hashes 与本轮数据流优化前的源码快照一致。
- 数据载入、CPU 启动、host argmax、CPU 数值验证和仿真退出不包含在下面的设备 cycle 中。

输入证据是 `results/embedding_token_20261014/formal_rank4/decoder_stack_result.json`、同目录 `logs/attention-sst*.log/.resources.jsonl`。可长期审查的汇总是 [embedding_token_summary.json](progress_results/2026-10-14/embedding_token_summary.json) 和新增 [decoder_e2e_analysis.json](progress_results/2026-10-14/decoder_e2e_analysis.json)。

## 可加和的端到端拆分

总计 **20,167,497.75 cycles = 20.16749775 ms**。

| 阶段边界 | cycles | 占总延迟 |
|---|---:|---:|
| Embedding | 31,508.625 | 0.156% |
| Embedding → 第一层交接 | 51,908.375 | 0.257% |
| 第一层 RMSNorm1/QKV/RoPE/Attention | 519,926.25 | 2.578% |
| 第一层 Wo/残差/Norm2/FFN/残差 | 8,795,572.5 | 43.613% |
| 两层交接/标志同步 | 49,240.25 | 0.244% |
| 第二层 RMSNorm1/QKV/RoPE/Attention | 509,969.75 | 2.529% |
| 第二层 Wo/残差/Norm2/FFN/残差 | 8,793,658 | 43.603% |
| final RMSNorm 前交接 | 15,584.25 | 0.077% |
| final RMSNorm | 48,740.75 | 0.242% |
| LM head 前交接 | 9,963.25 | 0.049% |
| 最后 token 的 LM head | 1,341,425.75 | 6.651% |

上表连续、互不重叠。RMSNorm1/QKV 已有融合、预取和流水执行，因此保留这部分的联合边界，不把重叠的算子窗口当作独立延迟相加。

两层 post-attention 合计 **87.216%**；其中 Wo/gate/up/down 在最后完成 worker 的链上累计约占端到端 **77.768%**。层间、final RMSNorm 前、LM head 前三次显式交接合计约 0.371%，不是当前第一优化目标。

## 把 Wo、FFN、激活进一步拆开

两层最终都是 core=16 最后完成。以下记录该 worker 每个 job 从启动到结束的时间：包括其计算、DMA 和等待。在同一个 worker 上这些 job 串行，因而可加和；不是全体 worker 的算子窗口。

| 算子 | 第一层 core16 cycles | 第二层 core16 cycles | 保留当前串行顺序的局部下界 |
|---|---:|---:|---:|
| Wo | 659,185 | 658,863 | 213,088 |
| 第一残差 | 36,869 | 35,514 | 8,240 |
| RMSNorm2 | 55,402 | 55,871 | 27,952 |
| gate projection | 2,691,446 | 2,682,955 | 852,352 |
| up projection | 2,593,587 | 2,590,424 | 852,352 |
| SiLU | 622,745 | 624,169 | 525,888 |
| SiLU(gate) × up | 180,287 | 181,778 | 32,960 |
| down projection | 1,899,529 | 1,907,812 | 845,920 |
| 第二残差 | 30,006 | 30,119 | 8,240 |
| worker job 合计 | **8,769,056** | **8,767,505** | **3,366,992** |

各层 worker job 之间还有 3,116 cycles 的 dispatch 间隔。第一层相对 post-attention 的边界调整为 23,400.5 cycles，第二层为 23,037 cycles，包含 Attention ACK 到 Wo 开始的交接，并扣除最终完成标志 ACK 相对最后张量 ACK 的差值。

JSON 另记录每个算子的全局 activity window、worker 最小/中位/最大 duration。比如第一层 SiLU 全局窗口是 1,306,644 cycles，但 core16 实际 job 只有 622,745 cycles；其余部分主要反映不同 worker 进入 SiLU 的时间不同。不能把全局窗口相加推断总延迟。

## 下界如何得到

这里计算的是必要下界，没有宣称存在可达到该下界的完整调度。固定现有 dense 工作、worker 分配、各大阶段 barrier 和 SFU 请求粒度；不依赖权重值恰好稀疏。

当前阵列一组 64 个，每个处理一行。每个 K=64 tile 的 MAC 为 `64+2=66 cycles`。64 个阵列共享同一份 8 KiB 权重，阵列内广播为 `1+8192/256+ceil(log2(64))=39 cycles`，输入 scatter 为 33 cycles。8 KiB GM 显式读分成两次 4 KiB，每次 17 cycles，共 34 cycles。

通用线性 job 的当前 callback 顺序是：

```text
HBM 权重到达并落入 GM
→ GM 权重读取(34) → matrix broadcast(39)
→ GM 输入读取(34) → input scatter(33) → MAC(66)
→ 下一个 K tile
→ K 结束后 output gather(33) → GM 写入(34) → HBM 输出 ACK
```

仅保留这一顺序、暂不收费 HBM/NoC 传输时间，每 K tile 也至少有 **206 cycles**。于是：

- Wo：`1024×206 + 32×67 = 213088`。
- gate/up：各 `4096×206 + 128×67 = 852352`。
- down：`4096×206 + 32×67 = 845920`。
- SiLU：64 次 8192 元素请求，每次 8217 cycles，共 525888。
- Add：16 次请求，每次 515 cycles，共 8240。
- Multiply：64 次请求，每次 515 cycles，共 32960。
- RMSNorm：16 次四行请求，每次 1747 cycles，共 27952。

这个顺序下界故意略去权重 DMA、输入准备、输出 DMA 和部分控制开销。预取、改变缓冲及融合可能解除其中某些顺序约束，因而需要重算下界。

第二种更宽松的 resource-capacity 下界允许 GM、buffer 和 MAC 重叠。不同独立资源取 **max**，同一个 worker 上有依赖的 job 则相加；不能简单把所有资源 service 加起来。实际扫描 GM local_read/local_write service，避免遗漏 DMA 自己消耗的端口。

QKV 每 worker 1536 次 launch、48 次 drain；其 buffer-port 下界为 112176，保留单 bank 串行 program/scatter/MAC/drain 后为 213552。每层 Attention 的 EXP issue work 为 139264 cycles，由真实 pipeline token 数扣除两层 SiLU 的 EXP 工作核对。

HBM 配置为 [hbm2e_2500.py](../../src/sst/elements/golem/tests/architecture/ramulator/hbm2e_2500.py)：每节点 8 个 128-bit channel，2500 MT/s，峰值 **320 GB/s/node**；四数据节点共 1280 GB/s。2 GiB/node 是地址/存储容量。峰值不是实际可持续带宽，读写切换、refresh、bank 冲突和 NoC 路径还会提高最低耗时。

| 比较口径 | cycles | 当前 / 下界 |
|---|---:|---:|
| 实测端到端 | **20,167,497.75** | 1.00 |
| 全链路 HBM payload/每节点峰值：最繁忙节点 | 3,799,521.325 | 5.31 |
| 保留 barrier/job 分配，允许独立资源重叠，加入阶段 HBM payload 下界 | **4,086,942.8** | **4.93** |
| 进一步保留当前线性算子内部串行顺序 | **8,020,456** | **2.51** |

resource-capacity 的合成计算为：

```text
2 × (QKV buffer 112176 + Attention EXP 139264 + post-attention 1572336)
+ final RMSNorm 27952
+ LM head max(worker compute capacity 266112, HBM tensor payload 411438.8)
= 4086942.8 cycles
```

serial-order 计算为：

```text
2 × (QKV serial array 213552 + Attention EXP 139264 + post-attention 3366992)
+ final RMSNorm 27952 + LM head serial 552888
= 8020456 cycles
```

两种合成下界都略去 embedding、控制交接、部分准备/收尾和完整 DRAM/NoC 依赖。倍数表示当前延迟与必要下界之间的差距，**不是已经验证可获得 4.93× 或 2.51× 加速**。独立的全链路 HBM 下界还保留了节点流量不均衡。

## 优先优化的数据流

**第一优先：通用 GEMM 输出写回。**

`decoderOpJob.h::storeTileRow()` 每得到一个 `64行×64列` 输出 tile，就逐行发送 128 B，等待该行 ACK 后才发送下一行；64 行全完成后才继续下一输出 tile。

每 worker、每层：Wo 2048 次，gate 8192 次，up 8192 次，down 2048 次，总计 20480 次。16 workers 共 **327680 次 128 B 写回/层**，仅搬运 40 MiB 输出。

第一层 core16 四个线性 job 的 DMA write RTT 累计 **2,385,885.625 cycles**，约占该层 post-attention 的 27.1%。这是同一个 worker 串行 job 上的诊断值，不是全局总线 service；也不能解释为能全部消掉的延迟。

建议先用有限 GM 的输出 panel 保留、打包后按连续行写回，或支持有界 scatter/gather / 多请求在途，和下一 tile 计算重叠。当前 tile 中相邻行对应 HBM 目标地址相隔 N 列，**不能直接把现有 8 KiB buffer 当成 HBM 连续块写出去**。应先选布局和 panel 大小，确保 GM 显式搬运有计时、ACK 后才允许下游读取、容量不超限。QKV 已有 retain-output/prefetch 路径可参考，但 K/N 可变的 FFN 需要通用化。

**第二优先：权重/输入 GM 预取。**

现有 Wo/gate/up/down 完全等待本 tile 权重 DMA、GM read、program、scatter、compute 完成才加载下一权重，未用 QKV 已有的 weight prefetch 和 local weight/input prefetch。

每层这些线性算子的权重读量为 **1664 MiB**：Wo 128 MiB，gate/up/down 各 512 MiB。唯一权重共 104 MiB，因每个 worker 负责不同的 64 行而读取 16 遍；已经利用阵列内同权重服务 64 行的 reuse。不要把“跨 worker 读 16 遍”直接宣称为能零代价消除：需要改变分配或者添加实际存在且有计时的跨 worker 广播。

先在 **原有单 array operand bank** 下重叠 HBM→GM、GM 本地读与当前 tile MAC，保持原容量、端口和正确的 buffer 生命周期。不要让下一权重提前覆盖仍在计算的阵列 operand。仅增加 HBM 容量不会改善这一串行等待。

**第三优先：LM head 的 M=1 分配。**

当前 `activeRows=1`，每 worker 只启用一个阵列，16 workers 只使用 1024 阵列中的 16 个。现有 row-parallel 映射对 M=1 不合适，应研究输出列并行、多组独立阵列共享最后 token 的输入。

但输出列不同就需要不同权重；当前广播将相同矩阵发给一组阵列，不能直接把阵列个数乘上去而宣称 64× 加速。head 本身读取约 501 MiB 唯一权重，HBM payload 下界约 411439 cycles，MAC 增加并行度后仍受它约束。先复用最后 token 输入、预取权重，再评估现有 matrix/input programming 接口能否实现列并行。

## 有价值的算子融合

建议分块推进，不一次让整个 FFN 的中间张量驻留 GM：每 worker gate/up 各 1 MiB，二者合起来已是 2 MiB，尚未包含其他 scratch、输入、输出与权重。

| 融合 | 可以减少的张量 HBM 往返 | 注意事项 |
|---|---:|---|
| SiLU + Multiply | `act` 写/读，共 **32 MiB/层** | 保留 SiLU 后 FP16 舍入点；标量 reciprocal 不会自动变快 |
| 分块 gate/up + SiLU + Multiply | gate/up 写/读 + act 写/读，共 **96 MiB/层** | 保持有限 SRAM panel；gated 输出先继续写回供 down 使用 |
| 再让 down 直接消费 gated panel | 额外 **32 MiB/层**，累计最多 128 MiB/层 | 需要下游 partial accumulation、K 顺序和阵列资源排程；不是删掉 DMA 即可 |
| Wo + Add1、down + Add2 | 分别减少 wo/down 的写/读，各 **8 MiB/层** | 使用 producer tile 与 residual 对齐的行列布局 |
| Add1 + RMSNorm2 | 减少立即回读 h1 的开销 | h1 还要供 Add2 使用，不能直接删除其所有存储；全 hidden 归约需要对应完整行可用 |

表中是这些张量边的逻辑 DMA payload，诊断 dump 若保留还会继续产生写回。对于 FFN 进一步融合的 128 MiB/层节约，上限按四节点 HBM 峰值仅相当于约 104858 cycles 的 payload service；实际收益还包含请求减少与调度重叠，但不能据此保证某个 cycle 加速比例。

SiLU 每 worker 64×8192=524288 元素，reciprocal 为 **1 lane、II=1**，光 issue 就至少 524288 cycles。现有 64 次有界请求计算耗时 525888，实际 core16 为 622745/624169。由此可见，融合能消除搬运和重复 launch，但不能将这 52 万计算 cycle 消掉。扩宽 reciprocal 或采用不同近似属于后续硬件/数值方案变更，本轮分析不修改它，也不改变当前 FP32 内部方案。

建议下一项实际任务：**只优化通用 GEMM 的有界输出 panel 写回，先覆盖 Wo，再验证 gate/up/down 和 M=1；然后加入权重/GM prefetch，最后做 SiLU+Multiply 及 FFN 分块融合。** 每一步分别报告相同 workload、精度和资源下的 cycle、DMA 请求数、有效 bytes、端口 service 与 SST wall time，避免一次改动过多而无法归因。

## 复现与验证

```bash
python3 baseline/attention_cluster_8qk_8pv/analyze_decoder_e2e.py \
  --case results/embedding_token_20261014/formal_rank4 \
  --scan-resources \
  --output baseline/attention_cluster_8qk_8pv/progress_results/2026-10-14/decoder_e2e_analysis.json
```

扫描逐 job 验证 DMA read/write bytes 与 DECODER_OP 汇总完全一致，验证同一 worker job 不重叠、两个 rank 的总 cycle 相同、Attention EXP token 工作一致。新增分析器 6 项单元检查通过，检查不同端口取 max、串行顺序与容量下界的关系、M=1 不凭空减少 K 计算工作、尾词表 tile、SiLU scalar reciprocal 下界和输出 tile 工作量缩放。没有更改仿真器实现，没有重跑仿真。

相关实现：[decoderOpJob.h](../../src/sst/elements/golem/attention/decoderOpJob.h)、[projectionJob.h](../../src/sst/elements/golem/attention/projectionJob.h)、[computeArray.h](../../src/sst/elements/golem/array/computeArray.h)、[sfu.cc](../../src/sst/elements/golem/sfu/sfu.cc)、[golem_projection_runtime.h](../../src/sst/elements/golem/tests/small/muticore_attention/golem_projection_runtime.h)、[golem_attention_runtime.cpp](../../src/sst/elements/golem/tests/small/muticore_attention/golem_attention_runtime.cpp)。
