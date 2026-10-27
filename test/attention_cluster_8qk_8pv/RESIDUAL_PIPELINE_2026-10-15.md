# down → 第二残差的分块流水

本轮实现第一条软件流水：FFN down 形成最终输出 panel 后，直接交给现有 SFU Add，
再写最终层输出。局部结果不替代端到端验收。

## 硬件约束

保持 16 worker、每 worker 64 arrays、单 operand context、2 MiB GM、1 MiB C-buffer。
没有修改阵列/SFU 实现、GM 端口或带宽、NoC 参数、HBM 组织和时序。
两组正式测试的每个数据节点均为 2 GiB backing，容量不作为本次实验变量。

现有软件布局的 GM 最大边界仍为 2,039,808 B；op9 部分和仍占用既有 C-buffer 的 256 KiB。
op9 输入最多为 64×2048 FP16，在原有 1 MiB 输入区域的空闲尾部复用 8 KiB：
`decoder_epilogue_gm=0x50000`，没有扩容。原有两个 down 输出 panel 保留所有权，
直到 Add 写回完成且最终 DMA ACK 全部返回才可复用。

## 调度和数值

原路径：`down → HBM down → 独立 Add 读取 down/H1 → HBM Y`。

新路径：`down FP16 panel → 共享 SFU Add(H1) → HBM Y`。
第一个最终 down panel 的残差读取和 Add 可与下一个 panel 的阵列任务重叠。
H1 经真实 HBM DMA 和 GM 端口读取；down panel 经真实 GM 端口读取、改写。
SFU 忙时重试，所有端口拒绝和 DMA ACK 都参与仿真，未直接从 HBM backing 旁路读取。

down 的 FP16 存储舍入和 Add 前后的精度边界保留，内部 FP32 计算方案保持一致。
没有新增 opcode：op9 复用原本不用的 `output_tile_begin` 字段表示 H1 在节点内的偏移；
descriptor 仍为 96 B。该字段在 Linear 中仍表示词表分块的列偏移。
`--down-residual-fusion` 显式开启本优化，并启用既有 FFN down/panel 融合；默认路径保留。

## 验证与复现

行为测试覆盖 2/4/8 槽、不同 FFN panel 宽度、prefetch 开关、尾行、空 lane、
多行组、端口反压、延迟和乱序 ACK，比较原路径与融合路径的 FP16 位模式。
152 项生产回归、139 项 host 回归、6 项 architecture 回归通过。

```bash
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_decoder_micro.py \
  --artifact-root results/residual_trial --shape formal --workers 16 \
  --mpi-ranks 8 --cpu-set 0-7 --with-down --with-residual --prefetch-slots 4 \
  --variants down16q4 residual16q4 --reference-variant down16q4 --weight-trace

OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/residual_stack_trial --shape formal --embedding \
  --mpi-ranks 8 --cpu-set 0-7 --gemm-prefetch --ffn-down-fusion \
  --ffn-panel-tiles 16 --prefetch-slots 4 --weight-trace --down-residual-fusion
```

局部基线 `down16q4` 必须同时使用 `--with-residual`，以比较相同的最终输出。
完整基线使用同一 build 和相同参数，省略 `--down-residual-fusion`。
cycle 仍从首个数据 DMA 发送计到最终输出 DMA ACK；wall 仅作运行成本记录。
沿用现有合成权重；所有 dense tile 的读取、编程和计算仍完整执行，未推进真实 checkpoint 导入。

新增 `[DECODER_RESIDUAL_FLOW]` 审计每 worker、每 epoch 的残差读取字节、chunk 数和 8 KiB scratch。
`gemm_overlap_cycles` 是 epilogue 活跃且阵列命令处理器忙的周期数，包含编程活动，
不能解释为纯 MAC 重叠，也不能与其他计数相加得到端到端时间。
资源审计同时拒绝缺失 H1、遗漏 dense 权重、多写 down 和未结束的 DMA/GM 操作。

## 测量结果

| 形状 | 原 down + 独立 Add | 分块残差 | cycle 降幅 |
| --- | ---: | ---: | ---: |
| H=256，64 rows，1 worker | 111,881.625 | 110,738.125 | 1.02% |
| H=2048，64 rows，1 worker | 1,959,928.625 | 1,942,869.125 | 0.87% |
| H=2048，1024 rows，16 worker / MPI8 | 2,034,887.375 | 2,003,388.5 | 1.55% |
| 正式两层 + Embedding + 最后 token LM head / MPI8 | 6,575,251.75 | 6,519,275.75 | 0.85% |

两组均通过独立 FP16 逐位 oracle、完整 dense 权重流量、最终输出覆盖、所有 ACK
和 SFU EXP/倒数实际工作量一致性检查。正式单 worker 记录到 6,871 个周期的
epilogue/阵列命令活动重叠。16 worker 局部审计覆盖全部 196,608 个 FFN 权重 tile。
完整两层减少 **55,976 cycles（0.8513%）**，计时从首个 token-ID DMA 发出到最终 logits ACK。
基线/候选的 448/416 个 decoder job 全部通过 DMA/GM 审计，每组 393,216 个 FFN
权重 tile 均完整执行。候选的 32 个残差 epilogue 生命周期、H1 字节和 scratch 审计通过。
两层输出、final norm、logits，以及仍保留的 Wo/H1/Norm2 中间结果全部 FP16 逐位一致；
greedy token 均为 99037。实际 EXP、倒数与 vector-op 流水线 token 计数均相同。
本轮改动包含 guest 对融合开关的读取，必须使用当前 build 配对基线，不能直接减去较早归档的 6,559,174.75。

可加阶段的变化如下；完整的 11 段划分见归档，求和严格等于各自的总 cycle。

| DMA ACK 阶段 | 基线 cycle | 候选 cycle |
| --- | ---: | ---: |
| 第一层 Attention O → 层输出 Y | 2,350,628 | 2,323,972 |
| 第二层 Attention O → 层输出 Y | 2,375,640 | 2,351,692 |
| LM head | 624,965.75 | 621,965.75 |

总收益也包含调度改变后下游 HBM/NoC 状态和阶段交接的变化；不能把各 worker 的
重叠计数或 job span 相加解释为总降幅。正式完整基线/候选的 SST wall 为
1277.88 / 1621.88 秒，运行期间与其他实验存在不同程度的 CPU/I/O 竞争，不用于 wall 加速比较。
保持显式开关，不改变默认路径；这是有效但幅度较小的软件优化。

对 H=2048、S=1024，两层理论上省去 down 张量的各一次 HBM 写和重读，合计 16 MiB；
H1 仍需读取，Add 元素数不减少。上一轮完整 4 槽结果中，最终关键 worker 的独立
第二残差分别约 29,963 / 30,409 cycles，远小于约 196 万周期的 FFN。
因此单靠本条 epilogue 流水无法带来大幅端到端降幅；后续应继续检查主 FFN 的串行编程和数据搬运。

原始数据在 `results/residual_pipeline_20261015/{small,formal_single,formal_concurrent,stack_baseline,stack_fused}`。
轻量证据、配置差异、测试日志和 SHA256 清单在
[residual_pipeline](progress_results/2026-10-15/residual_pipeline/)，包括
[完整对照](progress_results/2026-10-15/residual_pipeline/full_comparison.json)、
[阶段 cycle](progress_results/2026-10-15/residual_pipeline/phase_comparison.json) 与
[硬件约束](progress_results/2026-10-15/residual_pipeline/hardware_contract.json)。

本轮不把 Wo → 第一残差 → Norm2 → FFN 的整条流水、跨 worker 权重共享或 LM-head
阵列重映射计为已实现。后续逐项做局部实验；RMSNorm 需要完整 hidden 行归约，
不能在一行只完成部分列时直接启动该行的 FFN。
