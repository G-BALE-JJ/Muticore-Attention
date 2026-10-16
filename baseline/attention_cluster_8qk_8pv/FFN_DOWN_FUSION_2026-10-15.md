# gated panel → down 分块融合实验

本轮完成可选的 `GateUpSiluDown`（op9）、单 worker 局部 SST、16 worker 竞争及正式两层验收。正式配置推荐显式使用 `--ffn-down-fusion --ffn-panel-tiles 16`：相对同版本最佳 op8/8-tile 基线，完整链路 cycle 减少 **1.9555%**，所有保留输出逐位一致。开关默认仍关闭，未把该形状的配置推广为任意形状的最优值。

## 实现与资源边界

`--ffn-down-fusion --ffn-panel-tiles 8` 在既有 gate/up producer 与共享 SFU 后直接消费 GM 中的 gated panel。gate/up/down 共用原 64 arrays 和一个 operand context；两套 panel 由生产者、SFU 和 down 的所有权约束保护，down 消费结束才能复用。

down 按原有 K tile 顺序执行，每个 tile 后仍使用原 FP16 结果舍入。每个 panel 后将全部输出列的部分和写入既有 WCP C-buffer；下一 panel 恢复部分和后继续 mode=1。C-buffer 的读、写、阵列输出读取和恢复都走原定时接口，等待 ready cycle，不做免费数据直通。最终结果通过独立两组输出 panel 排空，等全部真实 HBM ACK 后完成。

| 每 worker 资源 | 正式 H=2048 的使用量 | 既有容量 |
| --- | ---: | ---: |
| GM 最高使用边界 | 1,941,504 B（1.8515625 MiB） | 2 MiB |
| C-buffer 部分和 | 262,144 B（256 KiB） | 1 MiB |
| 阵列 | 64 | 64 |
| operand context bank | 1 | 1 |

GM 在原 op8 布局之后增加两个 8 KiB down 权重预取槽和两个 128 KiB down 输出 panel。gated 经真实 GM 端口重新排布到已消费完的 up 槽。没有增加硬件容量、带宽、阵列或 SFU，FP32 内部计算方案保持不变。

当前 op9 限制 hidden≤2048、intermediate≤8192；down 权重在节点内的字节偏移使用 32 位字段。它覆盖本项目当前 2048→8192→2048，不代表任意大模型的 FFN 都能直接复用同样的 SRAM 布局。

正式每 worker 每层，8-tile panel 共 16 个，部分和读/写各 3.75 MiB，恢复 480 次；16-tile panel 共 8 个，读/写各 1.75 MiB，恢复 224 次。两层 16 worker 共省去 gated 的 HBM 写入 32 MiB 和读取 32 MiB，dense 权重读取量不变。

## 局部实验

正式配置 H=2048、FFN=8192；单 worker 64 rows，16 worker 1024 rows。基线为已经推荐的 op8、8-tile producer + 独立 down，预取开启。小规模 H=256 的两组初筛使用更早的两次 Linear + SiLUMultiply 基线，单独列出，不能与正式基线混算。

| 范围 | 候选 | 基线 cycle | 候选 cycle | cycle 降幅 |
| --- | --- | ---: | ---: | ---: |
| H=256，1 worker，旧基线 | down8 | 125,603.125 | 96,102.625 | 23.49% |
| H=256，16 worker，旧基线 | down8 | 147,756.5 | 108,535.625 | 26.54% |
| H=2048，1 worker，op8 基线 | down8 | 2,546,336.625 | 2,435,939.625 | 4.34% |
| H=2048，16 worker，op8 基线 | down8 | 2,999,659.75 | 3,100,182.75 | **−3.35%（退化）** |
| H=2048，16 worker，op8 基线 | down16 | 2,999,659.75 | 2,928,586.75 | **2.37%** |

计时为首个 Norm2 输入 DMA 发出到最终 down 输出 ACK，包含子图 dispatch 间隙；不包含 CPU 启动/退出。每组核对 guest/library/source/input SHA256、硬件环境、实际 EXP/倒数接受 token 数、全部 dense 权重读取、输出覆盖和全部 ACK。

原第一组正式 16 worker 在运行期间修改了 host 审计脚本，严格 source hash 比较拒绝了汇总。这组原始 provenance 保留；使用当前脚本和相同输入重跑基线，再比较已有候选，不修改原 SHA256，也不放宽比较条件。

重跑基线 cycle 与第一组完全相同。以 core 4 为例，op8+down 的预取等待计数合计 995,217 cycles，down8 为 1,060,201；完整 dense 权重预取均为 12,288 次。down8 另外执行 480 次部分和恢复。它改变了三组权重交错与跨 worker 竞争的访问顺序，因此仅消除 32 MiB/层的 gated HBM 流量不足以保证 cycle 收益。这些计数支持继续检查部分和与预取调度，但不能当作可相加的关键路径分解。

down16 的 core 4 预取等待降至 950,876 cycles，部分和恢复降至 224 次，C-buffer 读、写分别从 3,932,160 B 降至 1,835,008 B。保持相同 dense 权重读取、EXP/倒数工作量和 GM 容量后，局部测试重新取得收益。该结果支持在正式规模使用较大的消费 panel，但是否推荐仍需完整链路确认。

流量占比也解释了为什么收益不宜预期过高：正式 16 worker 每层 FFN 的 dense 权重读取为 1,536 MiB，原局部总 HBM 读写为 1,576 MiB，消除 gated 往返只减少其中 32 MiB，约 2.03%。这是字节占比，不是 cycle 收益上限；仍受调度、重叠与部分和流量影响。后续更大收益应先研究权重跨行块/worker 复用与预取调度，不能只继续扩大逐元素融合范围。

## 完整链路

完整测试覆盖架构内 Embedding → 两层 Decoder → final RMSNorm → 最后 token 的 LM head，使用合成权重与独立 CPU oracle，MPI rank=8。

H=256、FFN=1024、S=1024、vocab=8201，同二进制 op8 基线 631,059.625 cycles，down8 599,310.625 cycles，减少 **5.03%**。两层输出、保留的中间张量、final norm 与 logits 全部 FP16 位模式相同，greedy token 均为 5313。

正式 H=2048、FFN=8192、S=1024、vocab=128256：同版本 op8 基线 **9,003,939 cycles**，down8 **8,954,669.75 cycles**，减少 **0.5472%**。两层输出、final norm、保留的中间张量和 logits 逐位一致，greedy token 均为 99037；基线/op9 的 480/448 个 decoder job 全部通过 DMA 字节及退役检查，guest/library/source 相同，EXP/倒数工作量相同。

down8 的局部并发退化没有简单外推到完整链路：完整推理改变进入 FFN 时的 HBM/NoC 状态与 worker 相位。这正是局部测试用于筛选、完整链路负责最终验收的原因。收益只属于当前合成权重用例，未验证真实 checkpoint。

| 正式两层完整链路 | 总 cycle | 相对本轮基线降幅 | SST wall（并发环境） |
| --- | ---: | ---: | ---: |
| op8，8-tile producer + 独立 down | 9,003,939 | — | 1,341.80 s |
| op9，8-tile panel | 8,954,669.75 | 0.5472% | 1,730.08 s |
| op9，16-tile panel | **8,827,863** | **1.9555%** | 1,218.47 s |

down16 通过与 down8 相同的独立参考、逐位比较、448 个 job 的 DMA/GM 退役及 32 个 worker-generation 的部分和生命周期审计。第一/第二层 post-attention 分别从 3,473,186 / 3,407,870 降至 3,340,994 / 3,372,639.25 cycles。两种 op9 都少 64 MiB HBM 往返，Decoder 线性输出请求从 40,960 降至 8,192；权重预取总次数保持 490,112。详见 [正式配对汇总](progress_results/2026-10-15/down_stack_formal_all_comparison.json) 及 [build/环境审计](progress_results/2026-10-15/down_build_environment_audit.json)。

因此正式形状选择 16-tile；小规模已验证的 down8 有收益，但不自动套用正式形状的最优配置。下一步优先在局部 SST 中检查权重跨行块/worker 复用及预取调度，再决定是否增加融合范围。

原始记录在 `results/decoder_down_20261014/`；目录日期是实验启动日期。down8 首次启动误用了无效的 MPI physical CPU set 16–23，SST 未启动即失败；保存失败 provenance 后改为有效的 0–7 并重跑，最终记录如上。各次 wall 含不同的主机竞争，不能用表中 wall 比值宣称加速。

## 验证与复现

生产 `DecoderOpJob` 的异步接口测试覆盖 65/257 rows、M=1、无有效行 worker、17-tile 尾 panel、3/8/16-tile panel、预取开关、GM 拒绝重试、延迟/乱序 ACK 与部分和 ready 等待。所有比较保留原 K 顺序、逐 tile FP16 舍入。mock tick 只用于协议测试，性能结论均来自真实 SST。

host 额外拒绝跳过 down dense 权重、错误写回 gated、缺失/重复 panel、少计 C-buffer 读写和漏恢复。完整运行检查 epoch/reset、共享 scratch 复用与各资源退役。

本轮回归 **292 项通过**：生产计算/异步协议 152 项、host 审计 134 项、架构配置 6 项。轻量结果、版本哈希、资源汇总和回归日志归档于 `progress_results/2026-10-15/`。

正式端到端以本轮 9,003,939-cycle 基线为比较对象。历史 8,807,050.75-cycle 记录使用不同 guest/library，保留为历史数据，不用于计算本轮收益。

```bash
# 正式单 worker，比较当前 op8 与 op9
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_decoder_micro.py \
  --artifact-root results/down_trial --shape formal --with-down --cpu-set 0 \
  --variants fused8 down8 down16 --reference-variant fused8

# 并发筛选：增加这三个参数
# --workers 16 --mpi-ranks 8 --cpu-set 0-7

# 完整链路候选；本文基线使用 --ffn-panel-fusion --ffn-panel-tiles 8
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/down_stack_trial --shape formal --embedding \
  --mpi-ranks 8 --cpu-set 0-7 --gemm-prefetch \
  --ffn-down-fusion --ffn-panel-tiles 16
```

仿真并发时 wall 包含主机资源竞争，不作为纯净环境 wall 加速比。构建准备脚本另增加顶层 `/results/` 排除，避免把大体积仿真 backing 复制到 build tree；已完成文件只做内容不变的稀疏洞回收，未删除实验数据。
