# 局部筛选，再做端到端验收

每次改动都重跑两层、Embedding 和 LM head，会重复执行大量与候选优化无关的工作。
正式端到端最近一次 SST wall 为 1056.14 秒。后续采用以下顺序：

1. **行为/协议测试**：检查 FP16 舍入、尾块、反压、乱序 ACK、buffer 生命周期。
   `test_decoder_ffn_panels.py` 使用生产 job 和 mock 接口，其 tick 只用于行为测试，不能当作硬件性能。
2. **局部 SST**：只跑受影响子图，先小形状，再正式 K/N 宽度。
3. **并发 SST**：16 worker 保留 HBM/NoC/GM 竞争；固定输入/build/资源。
4. **端到端验收**：筛选通过后，对一个候选批次跑完整两层推理，确认最终输出 DMA ACK 的总 cycle 和全部输出。

局部收益不能直接当作端到端收益。它不能覆盖前后阶段的资源状态、同步、热数据和跨阶段干扰。
局部数值失败或明显退化的候选可提前淘汰；跨阶段调度候选应结合数据流判断，不设统一百分比门槛。

2026-10-15 的 FFN 两组交错已按该流程完成：小形状减少 0.87%、正式单 worker 减少 8.56%、
正式 16 worker 减少 4.60%，最后完整两层减少 3.94%。局部配对使用
`--with-down --with-residual --prefetch-slots 4 --variants residual16q4 groups2residual16q4 --reference-variant residual16q4`；
完整候选增加 `--ffn-array-groups 2`，并保留四槽预取、16-tile FFN、down/residual 融合。
额外真实端口审计入口为 `analyze_decoder_groups.py`，完整交叉审计为 `compare_decoder_groups_stack.py`。
参见 [完整记录](ARRAY_GROUP_PIPELINE_2026-10-15.md)。

## 已实现的 FFN 局部入口

`run_decoder_micro.py` 启动原 20-core/4 数据 HBM 节点的生产 SST 架构。
专用 guest `golem_decoder_micro.cpp` 直接提交真实 `DecoderOpJob`：

```
预置 Norm2 FP16 输入 → gate/up → SiLU×up → [可选 down] → 最终输出 DMA ACK
```

跳过 Embedding、Norm1/QKV/RoPE/Attention、Wo/残差/Norm2 计算、final norm 和 LM head。
Norm2 结果预置于 HBM，仍通过真实 DMA 进入 GM。所有 dense 权重 tile 都执行；合成权重的稀疏值仅用于 CPU oracle 加速。
地址来自统一 `DecoderMemoryLayout`；每 worker 2 MiB GM、64 arrays、一个 operand context，使用实际 SFU EXP/倒数/FIFO、GM 端口和 Ramulator2。
每节点 backing 128 MiB，HBM 的组织/时序不变。支持 1/4/16 个活跃 worker，分别处理 64/256/1024 rows，全部硬件仍实例化。
H=256/2048 对应 FFN=1024/8192。每个 case 在独立工作目录运行，避免 guest stdout 相互覆盖。

默认 `baseline` 为两次 Linear + 已有 SiLUMultiply；`fused16`/`fused8` 使用生产者融合 op8。
默认开启 GEMM 预取；`--no-prefetch` 用于配对检查。`--with-down` 追加 down。

```bash
# 快速筛选，含 down
python3 baseline/attention_cluster_8qk_8pv/run_decoder_micro.py \
  --artifact-root results/local_ffn_trial --with-down --cpu-set 0

# 16 worker 并发，MPI rank=8
python3 baseline/attention_cluster_8qk_8pv/run_decoder_micro.py \
  --artifact-root results/local_ffn_contention --workers 16 --mpi-ranks 8 \
  --with-down --cpu-set 0-7

# 正式 K/N 宽度，先做单 worker
python3 baseline/attention_cluster_8qk_8pv/run_decoder_micro.py \
  --artifact-root results/local_ffn_formal --shape formal --with-down \
  --variants baseline fused8 --cpu-set 0

# 重查已有输出和 ACK，无需重跑 SST
python3 baseline/attention_cluster_8qk_8pv/run_decoder_micro.py \
  --artifact-root results/local_ffn_trial --verify-only
```

`result.json` 保存阶段 cycle、HBM 字节、数值、资源和 wall；`summary.json` 验证输入/build/硬件以及 EXP/倒数工作量一致后计算收益。
输入、guest、library 和相关 source 都记录 SHA256。输出与独立 CPU oracle 的 **FP16 位模式完全一致**。
检查每次完整 dense 权重读取、最终输出地址连续覆盖、无重复写入、guest 成功，以及所有 DMA/GM ACK 均在 job 完成前。
cycle 从首个 Norm2 输入 DMA 发出到最后 gated/down 输出 ACK，包含子图 dispatch 间隙，排除 CPU 启动/退出。
wall 从 SST 子进程启动到结束，含初始化/HBM dump，排除输入生成和 CPU oracle。

## 本轮实测

H=256，含 down，各方案通过数值、guest 完成、SFU 实际工作量与 DMA ACK 审计：

| 活跃 worker / 有效 rows | 方案 | 局部 cycle | 相对 baseline 降幅 | SST wall |
| --- | --- | ---: | ---: | ---: |
| 1 / 64，rank=1 | baseline | 125,910.125 | — | 12.67 s |
| 1 / 64，rank=1 | fused16 | 117,162.125 | 6.95% | 12.33 s |
| 1 / 64，rank=1 | fused8 | 102,563.625 | 18.54% | 11.78 s |
| 16 / 1024，rank=8 | baseline | 144,482.0 | — | 12.38 s |
| 16 / 1024，rank=8 | fused16 | 133,413.0 | 7.66% | 11.83 s |
| 16 / 1024，rank=8 | fused8 | 116,916.5 | 19.08% | 11.83 s |

原始证据在 `results/decoder_micro_20261014/single_final` 和 `concurrent_verified`。
对应旧 H=256 完整两层/Embedding/LM-head SST wall 约 38.46 秒，局部入口约 12 秒/候选。
执行内容不同，因此这是测试反馈加快，不是硬件端到端加速倍数。
这些局部 wall 测试与另一个 CPU 上的正式宽度局部测试存在并发，不能据此推断纯净环境的 wall 加速比。
当前环境重复单 worker 的 cycle 完全一致。
DMA retry 参数与完整 runner 对齐为 4096 ticks / 32 次，避免过短超时产生重复请求或 down 输入读失败。

正式宽度 H=2048、FFN=8192、64 rows、单 worker、rank=1，含 down，证据在 `results/decoder_micro_20261014/formal_verified`：

| 方案 | 局部 cycle | 相对 baseline 降幅 | SST wall |
| --- | ---: | ---: | ---: |
| baseline | 3,157,549.625 | — | 122.92 s |
| fused8 | 2,546,632.625 | 19.35% | 104.84 s |

两组均通过独立 FP16 逐位参考、完整 dense 权重流量、最终输出 ACK、SFU 实际工作量和 guest 完成检查。
正式局部约 2 分钟/候选，远少于完整链路最近的约 17.6 分钟，但只处理一个 worker 的 64 rows，并非相同执行内容的 wall 加速比。
轻量证据归档在 `progress_results/2026-10-14/decoder_micro_*.json`。
本轮 host 回归 132 项、architecture 回归 6 项通过；新增 8 项检查计时边界、缺失输出、跳过权重、失败完成、配置一致性与实际 SFU 统计。

下一步分块融合先用本入口筛选。涉及 down 消费 gated panel、残差或 RMSNorm 时扩展子图边界，再进入全模型验证。

## 2026-10-15：down 分块消费入口

新增 `down8` / `down16`：op9 在 GM 内消费 gated panel，并使用既有定时 C-buffer 保存与恢复部分和，最终输出为 down。比较对象应显式选当前 op8 基线，避免把较早的 baseline 当作新增融合的收益：

```bash
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_decoder_micro.py \
  --artifact-root results/local_down_contention --shape formal --with-down \
  --workers 16 --mpi-ranks 8 --cpu-set 0-7 \
  --variants fused8 down8 down16 --reference-variant fused8
```

`down_partial` 另外核对每个 worker 的 panel 数、C-buffer 字节和恢复次数。正式单 worker down8 减少 4.34%，16 worker 局部却退化 3.35%；调为 down16 后，局部减少 2.37%，正式两层减少 1.96%。down8 完整链路也有 0.55% 小幅收益，说明跨阶段调度候选不能只凭局部百分比硬性淘汰或推广。当前正式形状建议显式使用 `--ffn-down-fusion --ffn-panel-tiles 16`。完整实验记录见 [down 融合实验](FFN_DOWN_FUSION_2026-10-15.md)。

## 权重队列实验（2026-10-15）

`down16q2` / `down16q4` / `down16q8` 固定 op9 和 16-tile panel，仅改变活跃预取槽数。
各配置使用同一 guest/library 与统一的最大 8 槽布局，GM 容量仍为 2 MiB，
最大使用 2,039,808 B。可选 `--weight-trace` 记录真实 DMA→GM→消费时间。

```bash
# 正式宽度的并发筛选
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_decoder_micro.py \
  --artifact-root results/weight_trial --shape formal --workers 16 \
  --mpi-ranks 8 --cpu-set 0-7 --with-down --weight-trace \
  --variants down16q2 down16q4 down16q8 --reference-variant down16q2

# 对完成的 case 审计权重等待，并回放有限容量共享机会
python3 baseline/attention_cluster_8qk_8pv/analyze_decoder_weights.py \
  --case results/weight_trial/down16q4 \
  --output results/weight_trial/down16q4/weight_analysis.json

# 完整链路配对验收；另运行一次 --prefetch-slots 2 作为同版本基线
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/weight_stack_trial --shape formal --embedding \
  --mpi-ranks 8 --cpu-set 0-7 --gemm-prefetch --ffn-down-fusion \
  --ffn-panel-tiles 16 --prefetch-slots 4 --weight-trace
```

`wait_dma + wait_gm = wait_total` 是前台权重就绪等待；`dma_sum` / `gm_sum`
是可以重叠的请求延迟之和，不是端到端阶段时间。共享回放只估计可减少的 HBM 字节，
没有实现真实 NoC 共享传输，也不预测 cycle 加速。详见
[本轮实验报告](WEIGHT_PREFETCH_2026-10-15.md)。

## 第二残差局部对照

比较 `down16q4` 与 `residual16q4` 时同时添加 `--with-down --with-residual --prefetch-slots 4`，
使基线包含独立 Add，候选包含同样的最终计算。`--verify-only` 从已保存的 shape 读取残差开关。
正式 16 worker 局部减少 1.55%，完整两层减少 0.85%；完整链路通过后再推广。
详见 [残差流水](RESIDUAL_PIPELINE_2026-10-15.md)。
