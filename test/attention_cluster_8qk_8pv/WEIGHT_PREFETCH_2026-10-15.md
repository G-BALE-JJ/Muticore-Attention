# 权重等待、预取深度与共享可行性实验

本轮以已经通过的 op9 / 16-tile panel 为基础，固定 FP16 I/O、逐 K tile 舍入、阵列数、operand bank、GM 容量及所有端口/NoC/HBM 参数，比较 2/4/8 槽权重预取。四组实验均已完成；原始目录为 `results/weight_prefetch_20261015/`。

## 测量口径

新增可选 `GOLEM_DECODER_WEIGHT_TRACE=1` / `--weight-trace`，每次权重消费记录 core、epoch、row、gate/up/down 身份、权重 HBM 地址以及提交、DMA 完成、GM 读完和消费 cycle。每个 FFN job 的 `DECODER_PREFETCH_FLOW` 保存：

- `wait_dma`：前台需要该权重、其 DMA 仍未完成时的停顿。
- `wait_gm`：DMA 已完成、其定时 GM 读取尚未完成时的停顿。
- `wait_total = wait_dma + wait_gm`。两者是权重就绪等待的互斥分类。
- `dma_sum` / `gm_sum`：各权重请求的延迟之和，可在不同请求间重叠，不能相加作为端到端 cycle 分解。
- `program_sum`：成功接受的阵列权重编程持续时间；`compute_sum`：阵列执行持续时间。
- `wait_program` / `wait_array`：对应 API 拒绝提交后的重试次数，不包含已接受操作的执行时间。

DMA 延迟从提交到完成回调，包含网络、排队和 GM 落地，不等于纯 DRAM
服务时间。本轮不能仅凭 `wait_dma` 在 NoC/HBM/落地端口间确定主因。

该测量没有把所有未计算时间都归因于权重：输入编程、partial 恢复、SFU、panel 重排和输出排空仍有独立成本。`analyze_decoder_weights.py` 对每个 tile 核对时间顺序、请求/命中数量、延迟汇总和唯一地址。down 的 K 编号是 panel 内相对值，唯一性以完整权重地址判断。

## 真实预取队列

原双缓冲改为 2/4/8 槽环形队列，前瞻生成与 gate→up→下一 panel 完全一致的权重序列。down 按当前 gated panel 的 K 与输出 N 顺序生成前瞻，不越过尚未产生的 gated panel。槽只在 DMA 和 GM 读取完成、数据被消费后复用，乱序完成不改变计算顺序。

在统一布局中两条路径各预留最多 8 个 8 KiB 权重槽，所有配置使用同一地址布局和二进制，运行时仅启用对应数量的槽。最大 GM 使用边界为 **2,039,808 B（1.9453125 MiB）**，剩余 **56 KiB**，容量仍为 2 MiB；C-buffer 使用仍为 256 KiB / 1 MiB。软件请求队列没有增加 operand context bank。

这里的“槽”是 GM 中一个 64×64 FP16 权重 tile 的 8 KiB 缓冲区，
不是新增阵列或 operand bank。4 槽配置的 gate/up 路径共用 32 KiB，
down 路径另用 32 KiB；为配对比较统一按 8 槽预留地址。相对旧的
每路径 2 槽布局，最大预留边界增加 96 KiB，仍处于原有 2 MiB 内。
改动实现于加速器侧 `DecoderOpJob` 控制状态机；RTL 实现需要相应的
队列元数据和请求管理逻辑。本实验固定被建模的数据通路和资源容量，
没有评估该控制逻辑的面积。

## 跨 worker 共享可行性

使用真实 SST 记录的权重请求轨迹，将同一输入节点的四个行 owner（例如 core 4/8/12/16）分组，统计相同权重 tile 请求的时间差，并回放每组 2/4/8/16/32 个 tile 的有限 LRU 容量。分别统计在途请求合并与已驻留权重命中。

这是**流量机会实验**：回放在请求时乐观地接纳条目，没有实现 multicast，也不计分发延迟、额外 GM 读写、同步和 ACK。它不能给出 cycle 加速比。8 个以上的额外 8 KiB cache 槽也放不进当前剩余 56 KiB GM，需重排现有 scratch 或改变调度；不能将回放容量直接当作免费硬件。

理想四 worker 共用一次读取能将 1,536 MiB/层的 FFN 权重 HBM 读减少至 384 MiB/层；三个 follower 的 NoC 分发和各自 GM 落地仍需发生。是否值得实现协同协议，要结合实测请求偏移、有限容量命中及端口开销判断。

## 验收与复现

局部形状为 H=256/2048、FFN=4H；1 worker 64 rows，16 worker 1024 rows、MPI=8。周期从首个 Norm2 DMA 发出到最后 down 输出 ACK。完整测试包含 Embedding、两层、final norm 和最后 token LM head。

```bash
OPENBLAS_NUM_THREADS=1 python3 baseline/attention_cluster_8qk_8pv/run_decoder_micro.py \
  --artifact-root results/weight_trial --shape formal --workers 16 \
  --mpi-ranks 8 --cpu-set 0-7 --with-down --weight-trace \
  --variants down16q2 down16q4 down16q8 --reference-variant down16q2

python3 baseline/attention_cluster_8qk_8pv/analyze_decoder_weights.py \
  --case results/weight_trial/down16q4 --output results/weight_trial/down16q4/weight_analysis.json
```

源码、guest、library 和输入哈希在运行前冻结，所有 dense 权重读取与 SFU 工作量保持相同。重跑同版本端到端基线，不用历史二进制结果计算收益。仿真存在主机并发，wall 只记录，不据此计算仿真加速比。

## 局部预筛选结果

| 形状 / worker | 2 槽 cycles | 4 槽 cycles | 8 槽 cycles | 4 / 8 槽降幅 |
| --- | ---: | ---: | ---: | ---: |
| H=256 / 1 | 114,965.125 | 108,293.125 | 108,649.125 | 5.80% / 5.49% |
| H=2048 / 1 | 2,499,164.125 | 1,936,404.625 | 1,936,393.125 | 22.52% / 22.52% |

正式单 worker 的权重前台等待为：

| 槽数 | wait_dma | wait_gm | wait_total | program_sum | compute_sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 375,744 | 252,647 | 628,391 | 491,520 | 823,296 |
| 4 | 1,241 | 425 | 1,666 | 491,520 | 823,296 |
| 8 | 1,156 | 445 | 1,601 | 491,520 | 823,296 |

三者均消费 12,288 个权重 tile，HBM 读 100,925,440 B、写 262,144 B，
输出与独立 CPU oracle 的 FP16 位模式一致。请求延迟总和与计算总时长
不能直接拼成总 cycle：预取、GM 读取与当前阵列任务存在重叠。

## 2 槽并发轨迹的共享机会

正式 16 worker 记录 196,608 个 FFN 权重 tile。相同 tile 的四个请求
最大时间跨度 p50/p95/max = 29,184 / 84,074 / 112,427 cycles；
单次 DMA 延迟 p50/p95/max = 177 / 297 / 1,357 cycles。

| 每组保留 tile | 容量 | 乐观可省 FFN 权重 HBM 读 |
| --- | ---: | ---: |
| 2 | 16 KiB | 6.94% |
| 4 | 32 KiB | 7.91% |
| 8 | 64 KiB | 11.72% |
| 16 | 128 KiB | 19.45% |
| 32 | 256 KiB | 27.30% |

4 槽回放的 15,550 次命中包括 14,093 次在途请求合并和 1,457 次驻留命中。
这与“worker 权重相同，所以自然节省 75%”不同：请求时间已经明显错开，
小容量下较早的 tile 会在后续 worker 请求前被淘汰。真实协议还需支付
跟随者的数据分发、GM 落地和确认成本，且请求轨迹会随共享发生改变。
本表只是原始 2 槽调度的流量机会，不是加速测量。

## 正式 16 worker 并发筛选

| 队列 | 局部 FFN cycle | 相对 2 槽 | SST wall |
| --- | ---: | ---: | ---: |
| 2 槽 | 2,928,586.75 | — | 447.06 s |
| 4 槽 | 1,995,820.625 | −31.85% | 440.22 s |
| 8 槽 | 2,390,014.25 | −18.39% | 455.19 s |

4 槽的平均 worker 权重前台等待从 988,775.25 降至 48,664.4375 cycles；
其中 DMA 585,355.875→31,734.625，GM 403,419.375→16,929.8125。
每 worker 编程 491,520 cycles、阵列执行 823,296 cycles，保持相同。
平均 DMA 延迟没有改善（p50 177→203，p95 297→425），但预取隐藏了绝大部分等待。
这说明不能用各 DMA 延迟之和解释总 cycle。所有候选 HBM 读/写均为
1,614,807,040 / 4,194,304 B；循环仍消费全部 dense 权重。

4 槽轨迹的同权重请求跨度 p50/p95/max = 15,024 / 19,805 / 25,137 cycles。
2/4/8/16/32 tile 共享回放分别仅省 1.75% / 2.54% / 4.13% / 7.63% / 14.83%
的 FFN 权重读取；原始独立 DMA 轨迹中的偏移远大于单次 DMA 延迟。

主机 wall 受事件数量、数值运算、日志和本轮并发影响。这里的 cycle 改善是
模拟硬件等待减少，不能解释为同等比例的 SST wall 加速。

### 为什么 8 槽不如 4 槽

8 槽平均权重前台等待回升到 450,650.4375 cycles，其中 DMA 为
391,039.5625、GM 为 59,610.875；DMA 延迟 p50/p95/max 为
313 / 3,028 / 6,498 cycles，明显高于 4 槽的 203 / 425 / 1,357。
更长前瞻扩大了并发传输压力；本轮选择 4 槽进行全链路验收，保持
2 槽默认值，通过显式 `--prefetch-slots 4` 启用候选配置。

8 槽轨迹的 4-tile 共享回放虽能省 18.33% 权重读取，却不能据此选择
它：实测 cycle 已比 4 槽差。共享回放不是实际共享调度，不能抵消该
实测结果，也不能把 8 槽的命中率套到 4 槽方案。

## 完整两层验收

重跑的 2 槽基线为 **8,827,863 cycles**，与上一轮 down16 实测完全一致。
SST wall 1,272.560515 s；448 个 Decoder job 的 DMA 字节与 GM 退役审计
通过，32 个 FFN job 共消费 393,216 个权重 tile，greedy token 为 99037。

完整 2 槽轨迹中，同权重请求跨度 p50/p95/max 为
92,103 / 159,989 / 189,489 cycles；4-tile（32 KiB）回放仅有
236 / 393,216 次命中，乐观权重读节省 **0.0600%**。
这比局部入口更差，说明真实阶段间调度会进一步错开 worker。
共享方案需要在完整链路上评估同步、容量和分发成本，不能从局部
轨迹的命中比例直接推断可用收益。

### 4 槽正式端到端验收结果

| 指标 | 2 槽 | 4 槽 |
| --- | ---: | ---: |
| token IDs → logits ACK cycles | 8,827,863 | **6,559,174.75** |
| 相对 2 槽 cycle 降幅 | — | **25.70%** |
| SST wall | 1272.56 s | 1149.71 s |
| FFN 平均 worker 权重等待 | 936,299.156 | 30,771.594 |
| 其中 DMA 等待 | 524,003.406 | 17,301.438 |
| 其中 GM 等待 | 412,295.75 | 13,470.156 |

两组均为 formal、S=1024、hidden=2048、FFN=8192、vocab=128256，
16 worker / MPI=8，包含架构内 Embedding、两层、final norm 和 M=1 LM head。
合成权重下，层输出、保留中间结果、final norm 和 logits **FP16 逐位相同**，
greedy token 均为 **99037**；每组 448 个 Decoder job、
393,216 个 FFN 权重 tile 通过时间顺序、计数、DMA 字节和 GM 退役审计。
源码/guest/library 哈希、输入 preloads、硬件参数、HBM 字节及 EXP/倒数工作量
一致。主机 CPU 绑定和测试并发不同，wall 仅记录，不计算 wall 加速比。

下面按互不重叠的 DMA 边界分阶段，包含交接；每列之和等于端到端总 cycle。
通用 GEMM 队列也影响 Wo 和 LM head，因此完整收益不是纯 FFN 单算子收益。

| 阶段 | 2 槽 cycles | 4 槽 cycles |
| --- | ---: | ---: |
| Embedding | 25,751.375 | 25,751.375 |
| Embedding → Decoder | 55,144.625 | 55,144.625 |
| Layer 1 Norm1/QKV/RoPE/Attention | 506,538.75 | 506,538.75 |
| Layer 1 Wo/残差/Norm2/FFN | 3,340,994 | 2,350,584 |
| 层间交接 | 51,684.25 | 51,719.25 |
| Layer 2 Norm1/QKV/RoPE/Attention | 510,546.5 | 511,990.75 |
| Layer 2 Wo/残差/Norm2/FFN | 3,372,639.25 | 2,363,075 |
| final norm 交接 | 15,443.25 | 15,593.25 |
| final norm | 48,520.75 | 48,513.75 |
| LM head 交接 | 9,956.25 | 9,911.25 |
| LM head | 890,644 | 620,352.75 |

4 槽完整轨迹的同权重四 worker 请求跨度 p50/p95/max 为
12,120 / 60,890 / 66,791 cycles。
4-tile（32 KiB）与 32-tile（256 KiB）回放分别只能乐观节省
**1.7408% / 10.23%**
的 FFN 权重读取。这个可行性实验没有实现真实 NoC multicast，
也没有模拟共享产生的 cycle 收益；下一步若推进共享，应先验证同步与
有限窗口调度能否覆盖分发/落地成本。

### 验收结论与证据

- 本轮四项：等待拆分、真实 2/4/8 槽队列、跨 worker 有限容量共享回放、
  局部→16 worker→完整两层验收，均已完成。推荐当前正式形状显式使用
  `--gemm-prefetch --ffn-down-fusion --ffn-panel-tiles 16 --prefetch-slots 4`。
- 295 项回归通过：production 152、host 137、architecture 6。
- GM 2 MiB、C-buffer 1 MiB、64 arrays、单 operand bank 及端口/带宽配置不变；
  GM 最大预留边界 1.9453125 MiB。新增控制队列元数据的 RTL 面积没有建模。
- 精简结果见 [weight_prefetch](progress_results/2026-10-15/weight_prefetch/)，
  其中 `full_comparison.json` 保存逐位比较、SFU 与构建/硬件审计，
  `phase_comparison.json` 保存可加的分阶段 cycle，`local_experiments.json` 保存九组局部结果。
  原始大 trace 和 HBM 保留于本地 `results/`。
