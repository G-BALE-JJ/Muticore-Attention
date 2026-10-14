# 在既有输出累加器中保留投影 partial

> 资源审计补充（2026-10-10）：本文结果使用两个阵列侧 operand contexts。
> manager/worker 拓扑已纠正，但该额外 operand 存储不符合当前目标资源约束。
> 历史数值结果可用于正确性比对；当前单 context 实现和验证见
> [SINGLE_OPERAND_CONTEXT.md](SINGLE_OPERAND_CONTEXT.md)。

本轮在已纠正的 4 个控制 manager、16 个计算 worker 拓扑上，只修改投影
调度。每个 worker 负责一个 64 行组，原 paired traversal 仍在每两个
hidden tiles 后读出 partial，并在下一对 tiles 前恢复。这期间没有其他
行组覆盖输出，因此可直接在同一输出累加器继续累加。

`GOLEM_PROJECTION_RETAIN_OUTPUT=1` 在 paired/D64、非 row-first、64 arrays
路径启用；多个行组（例如只使用 32 arrays）自动保留原保存/恢复路径。
每个 head 在 hidden tile 0 overwrite，后续 tiles 按原顺序 accumulate，
仅在完整 hidden 维度计算完后读出结果。FP16 每次乘法、dot 和累加舍入
不变；不增加 bank、输出向量、GM/cache、计算阵列、端口或带宽。

融合 Norm、packed Norm input、RoPE、audit 和最终 Attention 的复用流程
均保留。该模式不需要 projection partial C-buffer，也不提交 ahead restore。

## 完整端到端结果

shape 为 Hq32/Hkv8/D64、hidden2048、causal+RoPE。基线为修正架构后的
`results/worker_projection_architecture_fix/verified`，属于同一合法资源拓扑，
不使用历史 manager-array 性能作为基线。

| S | 投影之前 | 投影之后 | 投影减少 | 端到端之前 | 端到端之后 | 端到端减少 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 488,180 | 416,538 | 14.68% | 645,470 | 573,772 | 11.11% |
| 2048 | 946,000 | 801,164 | 15.31% | 1,499,733 | 1,354,890 | 9.66% |

S1024 正式结果在 `results/projection_retained_output/verified/s1024`，
S2048 在 `results/projection_retained_output/final/s2048`。两者数值、HBM
布局、MPI、实际资源归属、工作/流量计数、同步检查均 PASS，四个 HBM
node 的 Q/K/V panels 和 O 共 10 MiB/20 MiB 与基线逐字节相同，独立
NumPy O 最大绝对误差均为 0.0001220703125。

整个系统的 partial 读取及恢复次数由 11,520/23,040 降至零，保留最终
输出读取 768/1,536 次。每核 partial read、restore、C-buffer read/write
均为零。权重 DMA、权重编程、MAC、输入消费 payload 和 Norm 工作量不变。

| 最后完成的 worker（core16）上的状态计时 | S1024 之前→之后 | S2048 之前→之后 |
| --- | ---: | ---: |
| 输出读取（partial+final） | 26,880 → 1,680 | 53,760 → 3,360 |
| partial restore ready 等待 | 48,240 → 0 | 96,480 → 0 |
| partial 保存及 restore 提交 | 1,440 → 0 | 2,880 → 0 |
| 权重 LoadWeight 状态 | 10,704 → 13,628 | 21,942 → 26,431 |
| 阵列计算和 launch | 104,448 → 104,448 | 208,896 → 208,896 |
| GM 权重及输入读取 | 110,592 → 110,592 | 221,184 → 221,184 |
| 权重编程及提交 | 64,512 → 64,512 | 129,024 → 129,024 |

删除搬运状态直接省掉 74,880/149,760 cycles，但权重预取原来可以利用
这些间隔。间隔缩短后，LoadWeight 等待增加约 2,924/4,489 cycles，
因此投影净收益为 71,642/144,836 cycles。不是把删除状态的理论收益
全部当成端到端收益。

当前计算下界不变，仍为 268,592/799,328，实际/下界降为 2.136×/1.695×。
该下界未包含传输、编程、控制、RoPE 和 softmax 其他工作。

## 验证与日志

同一份二进制的小例关闭/开启对照为 74,318→73,588 cycles，16 个输出
region 逐字节一致。额外覆盖多 block、融合 RoPE、独立 Norm+慢 GM、D128
Attention，以及 32 arrays 下的 C-buffer 和慢 GM partial 回退。

新增 `[PROJECTION_ACCUMULATION]` 日志，并严格核验所有 worker 的模式、
retained pairs、partial 次数和配置一致性。131 项 attention 测试和 19 项
baseline 测试通过，总计 150 项；本地 library/guest 已重建。

首次 S1024 运行的 MPI 汇总 stdout 在一条 QK 记录中间插入其他 rank
的记录，因此 64 个 marker 只能完整解析 63 条，报告拒绝该运行。未放宽
完成数量检查。正式重跑使用 Open MPI `--output-filename` 独立 rank
stdout，`GOLEM_MPI_RANK_LOG_ROOT` 指向其目录；仿真完成后合并完整 rank
流用于报告，同时保留原 console 文件为 `.mpi-console-raw`。该改变仅影响
主机日志采集，不改变 SST 时间或硬件模型。

## 启用方式与下一步

在原已验证 worker-only 配置上增加 `GOLEM_PROJECTION_RETAIN_OUTPUT=1`。
默认值为 0，便于同一二进制对照；实际是否启用由报告和 ownership 检查
确认。配置、输入、源码和二进制 SHA256 保存在各 case 的 provenance 中。

本轮先移除了 partial 往返。下一项仍是 GM 权重/输入读取及权重编程，
优先评估以既有两份 operand banks 重叠下一 tile 的读取和编程，并严格
等待旧 bank 的计算完成后复用；之后再评估按输出 head/tile 分工恢复
跨行组权重复用，计入 Norm 数据分发和输入复制成本。
