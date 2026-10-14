# 固定资源的因果 Attention 调度

> **架构纠正（2026-10-10）：** 本文的历史投影/E2E 测量或重叠估计使用了 manager 本地 arrays，不符合「manager 只控制、16 个 worker 拥有阵列」的目标拓扑，不能作为该架构的性能证据。新的 worker 执行基准见 [WORKER_PROJECTION_ARCHITECTURE_FIX.md](WORKER_PROJECTION_ARCHITECTURE_FIX.md)。

本轮完成：因果窗口代价核算、描述符/地址/完成归属检查、早晚查询任务配对、
同库端到端与搬运量验证。实现粒度是已有的 **KV head × manager 查询区间**，
不引入任意行范围的新描述符。

## 调度和归属

启用 `GOLEM_ATTENTION_CAUSAL_BALANCE=1` 后，偶数 KV head 保留原 worker
group，奇数 KV head 使用 `3 - manager` 的 worker group。因此相邻两个
head 在每组执行早晚互补的查询区间：0/3 配对，1/2 配对。奇数 head 总数
的最后一个 head 保持原映射；非因果任务不迁移。默认开关为 0。

64-row QK tile 的有效 key tile 数为 `floor(last_query / 64) + 1`，
256-key PV window 数为 `floor(last_query / 256) + 1`。按现有 descriptor
覆盖的行组累加代价。32:8/D64 的每个 QK worker 仍有 8 个任务；配对后
S1024 每核 544 QK tiles、160 PV windows，S2048 每核 2,112 tiles、576 windows。

每个 head 的映射是 worker group 的置换，同一个物理 worker 不会收到两个
具有相同 head/job/tag 的查询区间。原 manager 仍检查 worker ID、slot 和
completion bitmap，再参与原来的全局完成汇聚。

Q/O 的 HBM 归属、查询因果位置、RoPE 位置均来自原查询 manager。实现修正了
packed Q 地址中依赖执行核编号的旧假设，并从 PV 消息已有的 `row` 恢复原
查询归属。PV 任务和预取提示跟随实际 QK worker group，返回完成消息仍发送
给 QK 执行核。各行的 key-window、online softmax、PV 累加顺序不变。

## 硬件边界

没有修改硬件拓扑、array 数量、GM 容量/端口/时序、C-buffer、KV buffers、
SFU lanes/II/depth、NoC 带宽、DMA credits、请求队列容量或消息/描述符 ABI。
现有 topology 指针本来就支持显式 worker IDs；本轮由 guest 给每个 head
写一个已有格式的 80 B topology，在既有 `0x1100..0x2000` 元数据间隙内保存。
8 个 head 占 640 B。该开关对超出这个元数据间隙的 head 数明确报错，不扩大
区域；开关关闭时沿用原路径。这是当前软件计划的适用范围，不是新增硬件限制。

固定正式配置：4 managers、8 QK/SFU、8 PV；GM 256 B/cycle、read/write 各
1 port、base latency 1、max request 4096 B、queue 256；projection arrays 64。
保留已有最佳投影融合、packed norm input、wrap prefetch、partial restore。

## 验证证据

正式结果位于 `results/attention_causal_balance/final`。control/balanced 使用
相同源文件、仿真库、guest 和输入 hashes；仅调度开关及宿主机 MPI affinity
不同。对旧最佳 packed 结果也再次做输出字节比较。本轮收益以同库 control
为基准，避免把旧版本少量启动 cycle 差异混入调度收益。

| 序列 | 同库 control E2E | balanced E2E | 节省 | E2E 降幅 | Attention 前 → 后 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1024 | 1,162,378 | 1,066,757 | 95,621 | 8.23% | 249,766 → 154,391 |
| S2048 | 2,767,997 | 2,373,527 | 394,470 | 14.25% | 943,687 → 550,110 |

Attention 降幅为 38.19% / 41.71%。Projection span 为
911,236 → 910,789 / 1,821,309 → 1,820,535，少量变化来自共同传输/启动时序；
投影计算和搬运计划未改变。此前最佳 E2E 为 1,161,859 / 2,766,728，本轮
优化后也分别低于它们 95,102 / 393,201 cycles。

| HBM operand 搬运 | S1024 前/后相同 | S2048 前/后相同 |
| --- | ---: | ---: |
| Q panel 请求 / payload | 1,280 / 10 MiB | 4,608 / 36 MiB |
| K panel 请求 / payload | 2,560 / 20 MiB | 9,216 / 72 MiB |
| V 广播 DMA 载入 / payload | 32 / 1 MiB | 64 / 2 MiB |

Q/K 统计来自实际接受的 WCP window transactions：每个独立 panel 在
RequestScheduler 中提交一个 HBM read。V 统计来自实际广播-cache DMA
载入，包括 reload，而不是用零 start miss 推断载入量。这里是 operand payload，
不含 NoC 控制包、头部或 multicast copies；既有 GM 读取服务与网络资源照常计时。

| 各核工作/等待范围 | S1024 前 → 后 | S2048 前 → 后 |
| --- | ---: | ---: |
| QK compute cycles | 10,560～61,248 → 35,904 | 38,016～240,768 → 139,392 |
| QK DMA wait cycles | 27,553～92,228 → 58,220～60,603 | 67,708～309,238 → 186,672～191,275 |
| SFU exp issue tokens | 40,960～237,568 → 139,264 | 147,456～933,888 → 540,672 |
| SFU exp issue wait cycles | 23,051～133,655 → 78,336～78,359 | 82,955～525,335 → 304,128～304,151 |
| PV busy cycles | 19,328～77,312 → 48,320～48,349 | 57,984～289,923 → 173,952～173,982 |

这是工作和等待的重新分布，不代表所有等待总和下降。SFU tokens 是工作量，
issue waits 是该单元的统计，两者不能直接当作端到端可加和分解。完整
QK/SFU/PV stage spans 与 PV 传输/队列等待也保存在结果 JSON。

正式输出检查包括 5 个 PASS 报告、独立 NumPy O、4 个 HBM 节点的全部保留
Q/K/V panels 和 O 逐字节比较，每种长度均有 16 个区域，分别覆盖 10/20 MiB。
两种长度 O 最大绝对误差均为 0.0001220703125。完整数据见
[`causal_balance_results.json`](causal_balance_results.json)。4 个正式端到端
运行、8 个扩展形状运行均通过；125 个 Attention/投影报告测试和 16 个
baseline/调度审计测试通过。构建、Python 编译、shell syntax、diff check 通过。

实际库 SHA256：`de1c14a862eeae8a87ea5a0c8e33d657c2b3c4092fca652201a5a342dcfe8d8e`。

S512 的额外配对回归全部通过独立 NumPy 和 O byte-exact：奇数 KV heads
6:3/D64、GQA=1 的 4:4/D64、8:2/D128、非因果 8:2/D64。非因果只验证开关
回退正确性，不将少量启动 cycle 差异解释为负载均衡收益。

首次实验保留在 `results/attention_causal_balance/balanced/s1024`：迁移 head
因 packed Q 读取错误 HBM node 而失败，未纳入收益。未迁移 head 全部字节一致，
迁移 head 出现偏差；修正查询归属后，正式实验和各通用形状回归通过。

## 重现

```bash
source scripts/env_local_install.sh
GOLEM_MPI_ARGS='--bind-to core --cpu-set 4-7' \
python3 baseline/attention_cluster_8qk_8pv/run_causal_balance.py \
  --artifact-root results/attention_causal_balance/reproduce \
  --reference-root results/projection_input_flow/packed

GOLEM_MPI_ARGS='--bind-to core --cpu-set 4-7' \
python3 baseline/attention_cluster_8qk_8pv/verify_causal_balance_shapes.py \
  --artifact-root results/attention_causal_balance/reproduce_shapes
```

端到端范围仍是 RMSNorm → QKV → RoPE → causal Attention → O HBM，
不包含 Wo、residual 或 MLP。阶段 span 相互重叠，不能相加。

## 现在的优化优先级

以当前相同 exp 工作量、8 个 SFU、每 SFU 16 lanes/II=1 计算，均衡后的
exp 发射下界为 S1024 139,264、S2048 540,672 cycles。Attention 154,391 /
550,110 比这个 **仅 exp 的局部下界** 高 10.86% / 1.75%；该下界不包含
QK/PV、RoPE、DMA、控制和启动开销，也不是整个网络的理论下界。

投影 producer span 已占新 E2E 的约 85.4% / 76.7%。下一步优先分析现有
RMSNorm/SFU 与 projection arrays 能否分块流水重叠，或进一步安排投影共享
array-buffer 端口；先核对已有输入区域的写入/消费生命周期，再做同资源对照。
不优先继续平衡 PV，也不增加缓存、输出上下文或带宽。
