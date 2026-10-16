# gate/up producer 到 SiLU × up 的分块融合

本轮实现 `Norm2 → gate/up panels → SiLU × up → gated HBM → 原 down`。gate/up、act 不再作为 HBM tensor 写出；down 和第二残差继续使用既有接口。所有 dense GEMM tile、单 operand bank、SFU 共享 EXP/倒数竞争及 FP16 中间舍入保留。

## 有限 SRAM 调度

新增 `DecoderOp::GateUpSiluMultiply=8`，descriptor 保持 96 bytes。`weights_offset` 是 gate 权重的 node 内偏移；op8 的 `input1_addr` 是 up 权重的 node 内偏移，而不是绝对 activation 地址。guest 仅在显式开启 `GOLEM_DECODER_FFN_PANEL_FUSION=1` 时提交该 op。

每个 worker 对自己的最多 64 行，依次执行一个输出 panel 的 gate GEMM、up GEMM，复用同一份已读入且已打包的 Norm2 输入。每个线性输出经真实 GM write、GM read、按行重排和 GM write 才交给 consumer，未跳过本地端口计时。

两组配对 panel 在 producer/consumer 间轮换。consumer 以最多 8192 元素为一块，通过真实 local GM read 取得 gate/up，提交真实 SiLU，保留其 FP16-rounded 输出，再提交真实 Multiply，将 gated 经真实 GM write 覆写已消费的 gate 块。完整 gated panel 才进入 HBM drain，最多 8 个 write 在途。consumer 状态独立于 GEMM 的前台 callback，可与另一组 panel 的矩阵编程、scatter、MAC 重叠。

producer 只能复用既未 ready、未被 consumer 持有、也未 drain 的槽位；最后一个 gated ACK 前不完成 job。完成标志、两层 epoch/reset/scratch 同步沿用原协议。gate 和 up 仍在同一组阵列上顺序执行，没有增加 operand bank 或矩阵并行资源。

| GM 区域 | offset | 大小 |
|---|---:|---:|
| 原最大 K=8192 input | 0x10000 | 1 MiB |
| 原 foreground weight scratch | 0x110000 | 8 KiB |
| gate/gated panel 0、1 | 0x112000、0x132000 | 各 128 KiB |
| 原 prefetch weight 0、1 | 0x152000、0x154000 | 各 8 KiB |
| up panel 0、1 | 0x156000、0x176000 | 各 128 KiB |

最高使用地址 `0x196000`，约 **1.586 MiB**，仍小于 worker GM 的 2 MiB。编译期 static_assert 和 runtime capacity 检查都约束该边界。consumer 的输入、系数和结果各自最多 8192 元素，是 SFU 请求的有限快照；不是额外完整 FFN 张量缓存。

## 对照与验证

小规模 S=1024、hidden=256、FFN=1024、词表=8201，包含 Embedding、两层、final norm、最后 token LM head，MPI rank=8。三组使用同一 guest/library、实际 preload hash 相同。

| 方案 | 总 cycles | 相对本轮基线减少 |
|---|---:|---:|
| 已有 panel + 预取 + SiLU/Multiply 融合 | 678,453.625 | — |
| producer 融合，16-tile panel | 647,322.625 | 4.59% |
| producer 融合，8-tile panel | 617,759.625 | 8.95% |

两层输出、Wo/h1/Norm2/gated/down、final norm 和 logits 的 FP16 bit patterns 全部一致。被省略的 gate/up 不作为 HBM 输出比较；独立线性 reference 从设备 Norm2 重建它们，再用独立 scalar SiLU reference 严格检查 gated，未放宽零误差容限。EXP 和 reciprocal accepted token 数逐 worker 与基线一致。

小规模两层减少 **17 MiB** HBM 流量：gate/up 往返 16 MiB，Norm2 两个 producer 复用一次输入再省 1 MiB。正式规模对应预期为 **136 MiB**：128 MiB gate/up 往返加 8 MiB 输入重复读取。这里均相对于已消除 act 往返的前一轮融合路径。

小规模 16-tile 只有一个 panel，SFU 没有下一 panel 的 GEMM 可以重叠。8-tile 有两个 panel；实际记录到 297,381 个 worker overlap cycles（全部 worker/job 求和，不能当作端到端节省）。两种 fused 配置均通过 256 个 job 的资源核对，逐 job read/write bytes 与 trace 一致；op8 精确读取一份 Norm2 和全部 dense gate/up 权重，仅写 gated，无请求跨越 job 完成。

正式规模 S=1024、hidden=2048、FFN=8192、词表=128256，包含相同的两层完整链路。

| 方案 | 总 cycles | 相对本轮基线减少 |
|---|---:|---:|
| 已有 panel + 预取 + SiLU/Multiply 融合 | 10,275,978.75 | — |
| producer 融合，16-tile panel | 8,928,309.75 | 13.11% |
| producer 融合，8-tile panel | **8,807,050.75** | **14.29%** |

以上两组同 guest binary/library，实际输入 hash 相同，所有保留输出的 FP16 bits 相同。旧归档的 10,210,612.75 cycles 使用旧 guest/library；相对它的 12.56% 是历史改善，不能代替上述同二进制 13.11% 对照。本轮增加了 guest 的融合分支，CPU 指令和 worker release 时刻也会影响端到端的总 cycle，因此必须重跑当前二进制的非融合基线。

正式 16-tile 两层实测 read/write 分别为 3,584,032,768 / 75,497,728 B，相比本轮基线 read 少 72 MiB、write 少 64 MiB，合计 **136 MiB**。prefetch request 数仍为 490,112；每个 worker 的 EXP/reciprocal accepted token 数与基线相同，未少做 dense 权重或 SFU 计算。两层由 core16 最后完成，其原 gate + up + SiLU/Multiply 的 job 合计分别为 2,630,563 / 2,620,715 cycles，融合 job 分别为 1,987,292 / 1,992,985 cycles。这是单 worker duration，包含等待，不是全局窗口。

正式基线和融合分别通过 544 / 480 个 job 的资源核对。每层每 worker 恰好生产、消费 8 个 panel、64 个 SFU chunk；GM 要求为 1,662,976 B，全部 GM interval 都在 job 完成之前结束。

正式 8-tile 相比 16-tile 进一步减少 **1.36%**，本轮推荐显式选择 `--ffn-panel-tiles 8`。每 worker 每层生产、消费 16 个 panel、64 个 SFU chunk；仍只写相同 gated 字节、GM 上限相同，480 个 job 的资源核对通过。其两层输出、保留中间输出和 logits 的 FP16 bits 与本轮基线一致，greedy token 仍为 **99037**；EXP/reciprocal 和 prefetch 工作量相同。相对上一轮旧归档减少 13.75%，与同二进制的 14.29% 分开记录。

三组 SST wall 分别为 1166.599 / 1098.155 / 1056.140 seconds。基线和 16-tile 曾在 CPU 8–15 / 0–7 并发，8-tile 在 16-tile 退出后使用 CPU 0–7，开始时与基线短暂并发。这些仅是单次观测，不用于推断独立运行的 wall 加速比。8-tile 首次启动误用了不存在的物理核范围 16–23，未产生 SST 事件；保留失败与重试 provenance，重试使用原输入且所有 hash 相同。

详见 [小规模对照](progress_results/2026-10-14/ffn_panel_small.json)、[16-tile 资源核对](progress_results/2026-10-14/ffn_panel_small_fused_resources.json) 和 [8-tile 资源核对](progress_results/2026-10-14/ffn_panel_small_fused8_resources.json)。

正式结果见 [同二进制对照](progress_results/2026-10-14/ffn_panel_formal.json)、[历史对照](progress_results/2026-10-14/ffn_panel_formal_historical.json)、[16-tile 资源核对](progress_results/2026-10-14/ffn_panel_formal_fused_resources.json) 和 [8-tile 资源核对](progress_results/2026-10-14/ffn_panel_formal_fused8_resources.json)。

## 复现

```bash
source scripts/env_local_install.sh
make -C src/sst/elements/golem/tests/small/muticore_attention decoder-embedding
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/ffn_panel_NEW/formal --shape formal --embedding \
  --mpi-ranks 8 --gemm-prefetch --ffn-panel-fusion --ffn-panel-tiles 8
```

`--ffn-panel-fusion` 默认关闭，`--ffn-panel-tiles` 支持 1..16，默认 16；与普通 GEMM `--output-panel-tiles` 独立。本轮小规模和正式规模都推荐显式选择 8；未做完整粒度 sweep，不声称 8 是全局最优。

## 检查范围

- host/分析 124 项、attention/SFU/guest 151 项、architecture 6 项通过，共 **281 项**；日志保存在 `progress_results/2026-10-14/ffn_panel_*tests.log`。
- 生产 `DecoderOpJob` 测试覆盖 17 个输出 tile 的尾 panel、非整行组、空 worker、M=1、panel=3/16、prefetch 开/关、gate→up→下一 panel 的权重切换、SFU busy 拒绝重试、GM read/write 拒绝重试及延迟/乱序 output ACK。输出 bytes 与独立的旧 linear + linear + SiLU/Multiply 路径比较。
- runtime 和 host verifier 将原 stages 4/5/6/7 合并为 stage4 op8，检查所有 32 个两层 worker generation、生产/消费 panel 数和 GM 上限。
- 默认 down 路径仍有 gated 的 HBM 往返；本轮没有实现 down partial accumulation 融合。
