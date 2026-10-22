# 中间层输出保留与 HBM 写回消除

本轮在独立的 `all-gemm-reuse-window` 基线上实现中间层 Y 只在片上保留。
原始 `Muticore-Attention` 工作区保持原状。16 worker、每 worker 的 64 个阵列、
单操作数上下文、2 MiB GM、端口、队列、时钟及 HBM/NoC 时序均不调整。

## 数据流与生效条件

此前已经实现 Wo → 残差 Add → RMSNorm → FFN 的 GM 交接，以及下一层
Norm1 和残差 Add 对 Y 的片上复用，但每层末尾仍把 Y 镜像写回 HBM。
本轮增加可选参数 `--elide-intermediate-output`，消除非末层的这次镜像写回。

1. FFN Down 的残差 epilogue 从现有 GM 读取 H1，经过 SFU Add，向已有的
   GM `0xb0000..0xf0000` 写入 Y。全部写入仍通过计时端口并等待 ACK。
2. 所有输出 panel 写入完成后，向 GM `0x3400` 写入原有的 64 字节所有权
   记录；记录包含 epoch、地址、形状及 worker 归属，等待写 ACK 后才能完成。
3. 中间层不再提交 Y 的 HBM DMA；原有 8 字节作业完成写回保留。
4. 下一层 Norm1/QKV 从保留的 GM 读取输入；残差 Add 也从该区域读取原输入。
   原有读取 ACK 与原地覆盖顺序约束继续生效。
5. 最后一层 Y 仍写回 HBM，最终 RMSNorm 和 logits 的输出契约保持有效。

开关要求同时启用 `--gm-stage-chain --layer-residency --input-residency`。
正式支持的片上驻留配置是 S<=1024、H<=2048，每 worker 一个 64 行组；
禁止同时要求保留所有层 HBM 输出。QKV 的 K 窗口必须覆盖整个 hidden 维度，
避免中间 C spill 覆盖驻留输入。原有私有权重、FFN layout=2 及 stage-chain
兼容性检查保留。运行时仅在有效的非末层 epoch、实际 GM producer 和
input-resident 消费路径均成立时省略写回；其他路径保留 HBM 输出。

新开关默认关闭，因此旧实验配置可以继续复现。本轮没有修改 reuse-window
调度器、权重预取、C 部分和或阵列映射，也没有新增无计时的张量缓存。

## S=1024、两层正式实测

H=2048、FFN=8192、vocab=128256、16 worker、8 MPI ranks。
输入与稀疏确定性权重用作数值测试，但所有 dense MAC、权重读取及窗口工作量
均照常执行并校验。下面是同一构建下的配对仿真；对照精确复现上轮采用方案。
1 GHz 下 cycles / 1e6 = ms，排除 host 准备和校验时间。

| 范围 | 对照 cycles | 本轮 cycles | 延迟降低 |
| --- | ---: | ---: | ---: |
| 完整端到端：token IDs 首次读取至完整 logits DMA ACK | 6,303,926 | 6,288,746 | 0.2408% |
| 两层 decoder prefill | 5,463,352.75 | 5,458,951.75 | 0.0806% |
| 最终 RMSNorm | 43,832.375 | 44,345.375 | -1.1704% |
| LM head | 671,248 | 659,918 | 1.6879% |
| 第 1 层 FFN span | 1,963,060 | 1,958,252 | 0.2449% |
| 第 2 层 FFN span | 1,961,415 | 1,965,098 | -0.1878% |

完整端到端 **6.303926 → 6.288746 ms**，减少 **15.180 us**；
两层 decoder **5.463353 → 5.458952 ms**，减少 **4.401 us**。
采用候选配置，因为这两个范围均有改善。

这不是每个阶段都加速：末层 FFN 与最终 RMSNorm 有轻微退化，LM 则变快。
LM 的实现和工作量没有改变，因此其收益不能归因于新增 LM 算法；前序完成
时刻变化会影响后续共享端口、队列和存储请求的竞争。这是本次确定性配对配置
的实测结果，不代表其他形状或权重下也能获得相同比例。阶段 span 可能交叠，
不能直接相加。本轮收益有限，权重流量与阵列执行仍占据主要成本。

## HBM 流量与 reuse-window

16 个首层 FFN/残差作业各省去 `64*2048*2 = 262,144` 字节的 Y 写回，
合计 **4,194,304 字节，即 4 MiB**。下一层输入读取在上轮已改为 GM，
因此本轮新增 HBM 读取节省是 0。末层输出继续写回。

| Decoder-op 作业计数口径 | 对照 bytes | 本轮 bytes |
| --- | ---: | ---: |
| HBM 读 | 4,027,846,656 | 4,027,846,656 |
| HBM 写 | 17,034,368 | 12,840,064 |

该口径包含 LM descriptor，不包含 projection/attention 作业，不能称为全芯片
HBM 总量。其中 HBM 写减少 24.62%。省略镜像 DMA 同时省去其 GM 源读取，
生产 Y 的计时 GM 写入与所有权发布并未省略。

所有 GEMM 的 reuse-window 检查通过。QKV、Wo、Gate、Up、Down、LM head
均记录到全部 16 worker 参与；这表示工作覆盖，不表示所有 worker 每周期都在计算。
QK 的 128 个作业与 PV 的 2,560 个窗口完成。通用 GEMM 窗口记录共 1,120 条，
两组均有 539,264 次 tile-group launch；HBM 权重字节、GEMM GM 字节、C drain
和 restore 次数完全一致。A 预取状态高水位仍为 2，结束时 live=0。

PV → Wo 的 HBM 重分布、Q/K/V 面板、末层 Y、最终 RMSNorm、logits 和控制
完成写回仍存在。本轮没有消除所有中间 HBM 操作，也没有实现完整跨层流水。

## 校验与证据

- 两组完整数值、生命周期、DMA、资源、GM bank 及 all-GEMM 覆盖检查均 PASS。
- 新增实际 payload 审计：16 个首层作业的 Y GM 写入完整且不重复，全部 ACK
  先于所有权记录发布；仅保留 8 字节作业完成写回。最后一层没有误省略。
- 最后一层 Y 的 2,097,152 个元素、最终 RMSNorm 的 2,097,152 个元素、
  全部 128,256 个 logits，与配对对照及上一轮采用方案的 FP16 位差异均为 0。
  贪心 token 保持 99037，Q/K/V 面板也逐位一致。
- 原有完整数学参考仍从初始输入独立计算两层，最后 Y 的误差上限仍为 0.004，
  本次最大误差 0.0009765625。没有放宽原有误差阈值。
- 中间层 Y 已不在 HBM 中，因此最后一层局部算子与 QKV 面板校验使用一个
  已通过验证、输入/权重哈希/布局/配置相同、保留 HBM 镜像的运行作为 host
  输入参考。记录参考结果 SHA256 和输入 SHA256；这只用于 host 校验，不向
  模拟器注入数据，也不把旧 HBM 数据当作片上 payload。独立完整链校验仍保留。
- 仿真启动后更新了 host 校验脚本，以正确处理不存在的中间 HBM 数据；初次
  外层面板报告因旧脚本读取失效 HBM 而失败，随后只重做 host 校验并通过。
  原始模拟器、guest、运行环境与计时日志均不变，校验脚本哈希另行保存。
- 209 个 baseline host 测试和 14 个生产 C++ 回归测试通过。覆盖错误参考、
  被篡改 Q 面板、缺失/重复/失败 GM 写入、提前发布、末层排除与回退路径。
- 48 个底层硬件及支撑源文件哈希不变；另单独审计布局文件仅有此前的软件
  Wo 别名，共检查 49 个文件。原始工作区 diff 与安装库保持原状态。

最早一组未完成的试跑因收紧残差输入驻留条件而主动终止，不纳入延迟。
有效证据位于 `results/software_output_residency_20261022/final_control` 与
`final_candidate`。汇总、两组工作量和脚本/二进制哈希发布在
[`software_output_residency_summary.json`](progress_results/2026-10-22/software_output_residency_summary.json)。
实测 SST 库 SHA256：
`710f45338107a850b1ce230f048b195f81266b9ba0b9ac89412192c644cb4366`。
多 GB 的日志、HBM 镜像及生成二进制不提交 Git。

## 复现

先用同一构建运行对照，保留上轮已验证、同配置且有 HBM 镜像的结果供局部
数值检查使用。以下路径对应本次工作区的既有结果：

```bash
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_all_gemm_reuse_window_baseline.py \
  --artifact-root results/output_residency_reproduction/final_control \
  --baseline-result results/software_input_residency_next_20261021/fixed_candidate/reuse_window_probe/decoder_stack_result.json \
  --shape formal --run --window-a-prefetch 1 --lm-n-parallel 1 \
  --gm-stage-chain --layer-residency --lm-input-residency --input-residency --cpu-set 0-7
```

再运行候选，把 artifact root 改为 `final_candidate`，其余参数相同，新增
`--elide-intermediate-output`；可使用 `--cpu-set 8-15`。
`--baseline-result` 也可指向刚完成且 PASS 的配对对照。
最后执行：

```bash
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/summarize_output_residency.py \
  --experiment-root results/output_residency_reproduction
```

汇总要求对照精确复现上轮采用的完整端到端和 decoder 延迟，并验证原工作区
及固定硬件快照。本地历史证据是这部分审计的前提，Git 中的紧凑汇总用于查看
本次测量结果，不能替代完整日志。
