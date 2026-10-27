# 投影输入布局与预取调度优化

> **架构纠正（2026-10-10）：** 本文的历史投影/E2E 测量或重叠估计使用了 manager 本地 arrays，不符合「manager 只控制、16 个 worker 拥有阵列」的目标拓扑，不能作为该架构的性能证据。新的 worker 执行基准见 [WORKER_PROJECTION_ARCHITECTURE_FIX.md](WORKER_PROJECTION_ARCHITECTURE_FIX.md)。

本轮实现两项独立、默认关闭的执行计划：

- `GOLEM_PROJECTION_INPUT_PREFETCH_BUDGET=4`：限制现有 future-input snapshot 的
  在途请求数量，并在每 tick 先提交前台 GM 请求。零表示原调度，允许范围 0–64。
- `GOLEM_PROJECTION_PACKED_NORM_INPUT=1`：RMSNorm 直接在原 GM 输入区域写出
  按 64-row group / hidden-tile pair / row 排列的消费布局。

GM 的非流水服务模型保持原样：256 B/cycle，base latency 1，读写各一个端口，
单请求上限 4096 B、共享 pending 容量 256。数组、operand banks、output
contexts、DMA/NoC 带宽、C-buffer 容量及所有 payload snapshot 上限均不变。
没有新增一遍全块转置、HBM 中间张量或持久缓存。

## 实现与兼容性

预取预算约束软件提交时机，不增加硬件 DMA credit。只有原 paired pipeline
启用时预算才生效。提交被拒绝时继续使用原 snapshot 重试，数据就绪与数组
bank 所有权约束保持不变。

消费布局使一个 row group 的两个相邻 64-column tiles 在 GM 内连续。foreground
和 background 两条输入读取路径都使用同一地址转换。RMSNorm 写入复用现有
批次输出 snapshot，按 pair 分段写到目标地址；投影在整块 Norm 完成后才读取。
布局是内部 scratch 格式，外部 Q/K/V panels、O 和描述符 ABI 不变。

布局只在已融合 Norm、paired reuse、无审计输出的路径生效，并要求 SFU 批次
行数整除 array row group。其他情况保留 row-major。具体包括 audit 模式、
非融合 Norm、非配对形状、row-first 及不兼容批次/数组宽度。该选择不改变
RMSNorm 或矩阵运算顺序、FP16 编码、partial rounding 和累加顺序。

## 请求计数与净开销

正式 S1024 每 manager 的输入流：

| 指标 | 原最佳 | 预算 4 | 消费布局 |
| --- | ---: | ---: | ---: |
| 输入 pair 数 | 3,072 | 3,072 | 3,072 |
| 每 pair 读取 | 64 × 256 B | 64 × 256 B | 4 × 4096 B |
| 输入读请求 | 196,608 | 196,608 | **12,288** |
| 输入读字节 | 48 MiB | 48 MiB | 48 MiB |
| Norm 写请求 | 256 | 256 | **1,024** |
| Norm 写字节 | 1 MiB | 1 MiB | 1 MiB |

读取请求减少 93.75%，原 GM 模型的输入纯读端口服务从 393,216 降到 208,896
cycles。但写请求变多：hidden2048 每个 Norm batch 为四行，原布局的四个
4 KiB writes 改成 16 个 1 KiB writes。该额外开销全部计入 E2E。
S2048 的工作量和上述请求数/字节数翻倍。

新 `[PROJECTION_INPUT_FLOW]` 记录实际布局、有效预算、请求上限、foreground /
background 输入请求、权重请求、Norm 写请求与预算暂停次数。报告根据真实
shape 和请求上限推导计数，拒绝丢失请求或不完整 manager 记录。预算暂停
表示达到软件在途上限，不等同于前台停顿，不能加入 E2E。

## End-to-end 对照

正式 shape：FP16 causal Hq32/Hkv8/D64/hidden2048，seed1742。
所有计划均启用 Norm/projection 融合、跳过审计 writeback、next-pair input
prefetch 和 partial ahead restore；RoPE fusion 关闭。范围是 RMSNorm、QKV、
RoPE、Attention 到 O HBM，不包含 Wo、residual 或 MLP。

| 计划 | S1024 E2E cycles | S2048 E2E cycles | 相对原最佳降幅 |
| --- | ---: | ---: | --- |
| 原最佳 / control | 1,265,454 | 2,974,207 | — |
| 仅预取预算 4 | 1,229,964 | 2,902,471 | 2.80% / 2.41% |
| 仅消费布局 | **1,161,859** | **2,766,728** | **8.19% / 6.98%** |
| 消费布局 + 预算 4 | **1,161,859** | **2,766,728** | **8.19% / 6.98%** |

组合与单独布局在两种长度上 cycle 相同。因此推荐 packed=1、budget=0；
预算 4 可作为 row-major 的独立优化选项。两项收益不能相加。

四个计划在两种长度上使用同一 simulator library，SHA256：
`2d6b26635ee3fbfdad74a2bdc133c66bb9a211fa34a6560b7d377aa830e5aeb4`。
输入 SHA256 一致，硬件与执行配置只允许两项新计划参数及 host affinity
不同。匹配分析要求 GM/C-buffer payload、snapshot 容量、数组计算、融合
运算工作量与 output fabric 请求/服务成本相同。原配置重现旧最佳 cycle。
优化方案逐个对照全部保留 Q/K/V panels 和 O，共 16 个区域，S1024 10 MiB、
S2048 20 MiB；另运行独立 NumPy 数值、MPI、HBM 布局和生命周期验证。

## Cycle 收益来源

S1024 每 manager 的独占 controller tick 分解：

| 项目 | 原最佳 | 预算 4 | 消费布局 |
| --- | ---: | ---: | ---: |
| 权重 GM 读取 | 99,840 | 65,280 | **55,296** |
| 前台输入 GM 读取 | 6,192 | 6,192 | **3,456** |
| GM 写入 | 4,608 | 4,608 | **6,144** |
| 输出读取 | 163,920 | 163,920 | **107,520** |
| 输出恢复，含 dispatch | 72,576 | 72,576 | 72,576 |
| Weight broadcast/program | 64,512 | 64,512 | 64,512 |
| 输入 scatter | 33,888 | 33,888 | 33,888 |
| 数组 compute/launch | 417,792 | 417,792 | 417,792 |

限流减少输入请求对权重的 GM 排队。布局既减少 GM 小请求开销，也使输入
及其 scatter 更早完成，输出 drain 的暴露等待随之下降。输出 gather / restore
仍是原来的 3,072 / 2,880 次，服务带宽和 service cycles 没有变化。

3,072 次 gather 的纯 service 为 101,376 cycles。消费布局后的 ReadOutput
仅比它多 6,144 controller ticks，原来差 62,544。因此继续调整输出端口
优先级的空间已经明显缩小；不应再把旧 163,920 ticks 全当成可消除排队。

消费布局的 producer span 从 1,013,877 / 2,028,116 降到
910,830 / 1,820,471；Attention span 从 250,095 / 943,486 变为
249,486 / 943,701，只有小幅请求时序变化。完整 exclusive partitions
保存在机器报告，异步 resource spans 不能再次相加。

## 压力测试、回退与既有边界

Hq8/Hkv2/D64/S1024、16 projection arrays、GM 64 B/cycle、base latency 3、
单请求上限 1 KiB、queue 256：消费布局+预算 4 通过独立 E2E 验证，
316,522 cycles。每 manager 输入请求为 3,072、Norm writes 为 256，验证
不同数组宽度及更小请求分块。改变的压力参数不纳入正式性能比较。

Hq4/Hkv2、16 arrays、audit 打开时自动回退 row-major，完整 E2E 验证通过，
157,188 cycles。另有单独 audit 回退（Hq8/16 arrays，334,000 cycles）和不兼容批次回退
（Hq4/16 arrays、audit 关闭，149,456 cycles）测试，均通过并实际记录 packed=0。

把同一 1 KiB 压力配置的 queue 降为 64 时，投影正常结束，但原有 Attention
RoPE table bulk read 在 LoadingKv 失败。关闭新布局和预算的匹配对照同样
失败。现有 `readAttentionRopeTableLocal` 一次提交全部 128 个 table chunks，
拒绝提交被当作读取失败而不是重试。正式对照始终使用原有 queue 256 和
4 KiB cap；没有修改这个 Attention 路径或任何硬件参数来取得正式收益。
失败记录保存在 `failed_small_queue_packed` / `stress_small_queue_control`。

最初使用 standalone runner 的非集群尝试和无效 host affinity 的组合启动
也保留在 `failed_*` 目录；不作为性能或正确性通过证据。

Python regression 共 137 tests（125 Attention/report、12 baseline），包含
新增请求计数、invalid limit、缺失 manager 的拒绝测试。C++ 本地构建、两份
runner shell syntax、Python 编译和 diff checks 均通过。

## 复现

保持此前最佳融合和 partial 调度设置，只增加一个参数即可使用消费布局：

```bash
source scripts/env_local_install.sh
GOLEM_PROJECTION_CBUFFER_PARTIAL=1 GOLEM_PROJECTION_WRAP_PREFETCH=1 \
GOLEM_PROJECTION_PARTIAL_RESTORE_AHEAD=1 GOLEM_FUSION_NORM_PROJECTION=1 \
GOLEM_FUSION_PROJECTION_ROPE=0 GOLEM_FUSION_SKIP_AUDIT_OUTPUT=1 \
GOLEM_PROJECTION_PACKED_NORM_INPUT=1 GOLEM_PROJECTION_INPUT_PREFETCH_BUDGET=0 \
python3 baseline/attention_cluster_8qk_8pv/run_llama_projection.py \
  --sequences 1024 2048 --artifact-root results/fresh_input_flow/packed \
  --reference-root results/projection_partial_restore/restore_ahead
```

正式四方案归档在 `results/projection_input_flow`。分析器可复查同库、同输入、
同资源、物理 output fabric 和 byte-exact 对照：

```bash
python3 baseline/attention_cluster_8qk_8pv/analyze_projection_input_flow.py \
  --artifact-root results/projection_input_flow \
  --variants budget4 packed packed_budget4
```

紧凑的完整结果保存在
[`projection_input_flow_results.json`](projection_input_flow_results.json)。
