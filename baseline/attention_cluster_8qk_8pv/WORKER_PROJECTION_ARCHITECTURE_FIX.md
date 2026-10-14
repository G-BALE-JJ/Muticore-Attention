# 投影执行归属纠正

> 资源审计补充（2026-10-10）：本文结果使用两个阵列侧 operand contexts。
> manager/worker 拓扑已纠正，但该额外 operand 存储不符合当前目标资源约束。
> 历史数值结果可用于正确性比对；当前单 context 实现和验证见
> [SINGLE_OPERAND_CONTEXT.md](SINGLE_OPERAND_CONTEXT.md)。

目标拓扑为四个专用 manager（core 0..3）和十六个 worker（core 4..19）。
manager 仅有控制、GM、调度等资源，不实例化 array 或 SFU。此前投影使用
manager 本地 arrays 的实现违反此约束，其历史投影/E2E 性能不适用于目标
架构；历史数值结果保留作为输出兼容性参考，不作为有效性能基准。

## 修正后的执行

每个 manager 用既有 remote-store 控制接口发布四个 worker 的启动标记。
worker IDs 为 `4 + manager + lane*4`，lane=0..3。每个 256 行块按 lane
分成四个互不重叠的 64 行范围，每个 worker 只执行自己范围的 RMSNorm
和全部 Q/K/V heads。所有十六个 worker 都参与投影。

manager 不执行 GEMM/Norm，不创建 ProjectionJob；RoCC 拒绝 manager
projection 指令。manager 最后通过既有 HBM flag-wait 等待所有十六份完成
标记后提交 Attention。worker 等待自身 ProjectionJob 完整退役，之后原有
Attention worker 引擎复用相同 arrays/SFU。两阶段没有独立的额外阵列。

描述符保持 144 字节；原 `reserved` 字段表示软件 worker lane+1。
`manager_slot` 继续表示输入/输出 HBM 的数据归属。GM scratch 使用执行
worker 自己的既有 2 MiB GM。panel 外部布局、行号、RoPE 位置、hidden tile
累加次序不变；各行 Norm 只计算一次。

原 FP16 manager-array 的数值语义由既有 worker float-array 的 projection
命令 dtype 模式模拟：权重、乘法、累加按既有 GolemFp16 舍入，命令仍使用
原两份 operand bank 和单份输出。不增加数组、bank、MAC、端口或队列；
非 projection 权重编程清除该模式，Attention 继续使用原 FP32 数值语义。
MAC 时序、带宽、GM/array-buffer 真实异步搬运计时保持原配置。

## 明确的调度代价

每个 worker 仅消费一个 64 行组，原一对权重供四个行组复用的收益不再
发生在单个核内。整个系统的 weight loads/programming 因而增为原错误
实现的四倍；这是迁回合法资源后的调度代价，不能声称搬运总量不变。
总 MAC、Norm 行数、输入消费 payload、partial 保存/恢复 payload 不变。

只有一个行组时禁用跨 pair 的 wrap prefetch：当前输出尚未保存到 C-buffer，
提前读取同一行的下一 partial 会得到旧结果。多行组路径仍保留已有窗口。

日志增加 `[CORE_RESOURCE_OWNERSHIP]` 和 `[PROJECTION_WORKER_OWNER]`。
端到端报告要求四个 manager 的 array/SFU 均不存在、十六个 worker 均存在，
并验证每个 manager 的四个行分区完整、互斥。报告中的 per-core projection
数据使用 `*_by_worker`。`[PROJECTION_*] core=` 记录实际执行 worker ID；
ownership 行中的 `manager=` 单独记录 HBM 数据 owner。解析器兼容旧日志，
但不会把旧 manager-array 运行作为合法架构的性能基准。

## 验证

正式 Llama 32:8/D64、hidden=2048、causal+RoPE 运行位于
`results/worker_projection_architecture_fix/verified`。投影 producer span
包含融合的 RMSNorm；独立 RMSNorm span 与它重叠，不能重复相加。
端到端定义为第一份 Norm/Projection 开始至最后一份 Attention 输出完成，
不包含 guest 输入生成、启动及结束后的 dump。

| 序列长度 | 投影（含 RMSNorm） | 投影→Attention 交接 | Attention | 端到端总 cycle |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 488,180 | 2,081 | 155,209 | 645,470 |
| 2048 | 946,000 | 3,193 | 550,540 | 1,499,733 |

两项均为 PASS，每项十六个 projection jobs。Q/K/V panels 与 O 总计分别
10 MiB、20 MiB 逐字节一致，独立 NumPy 最终 O 最大绝对误差均为
0.0001220703125。紧凑结果、资源配置与源码/二进制 SHA256 位于
[`worker_projection_architecture_results.json`](worker_projection_architecture_results.json)。

资源检查使用 SST 实际组件绑定日志，数值检查使用独立 NumPy 参考，同时
将四个 HBM node 上全部保留的 Q/K/V panels 与 O 和旧输出逐字节比较。
旧结果仅用于数值兼容性；比较报告明确标记 `performance_comparable=false`，
不再输出旧错误拓扑到新拓扑的 saved cycles 或 reduction percent。

额外回归位于 `results/worker_projection_architecture_fix/regressions_verified`：

| 场景 | 结果 | cycles |
| --- | --- | ---: |
| S3072，多 256 行块，Hq4/Hkv2/D64 | PASS | 245,628 |
| S1024，融合 RoPE，Hq4/Hkv2/D64 | PASS | 70,186 |
| S1024，独立 Norm、GM partial、64 B/cycle GM | PASS | 104,993 |
| S512，D128 Attention，Hq8/Hkv2 | PASS | 22,663 |

这些不同 shape/配置的 cycle 不互相作为性能基准。第三项仅用于验证既有
GM 配置和回压路径，正式测量仍采用原 256 B/cycle、1-cycle base latency。
融合 RoPE 的 QK operand 流量检查从 worker 全生命周期 GM 读取中扣除
独立核验过的 projection operand 和 fusion payload，避免两阶段重复计数。

现有 attention 129 项和 baseline 16 项测试均通过，总计 145 项；新增资源
归属拒绝测试覆盖 manager array、manager SFU、重复行分区、缺失 worker。
本地 library/guest 已重新构建，`git diff --check` 和 Python 编译检查通过。

回归可由以下命令重跑（artifact-root 必须是新目录）：

```bash
source scripts/env_local_install.sh
python3 baseline/attention_cluster_8qk_8pv/verify_worker_projection_architecture.py \
  --artifact-root results/worker_projection_architecture_fix/new_regressions
```

下一轮数据流优化必须基于十六个共享计算 worker。可评估按输出 head/tile
分工以恢复权重复用，但需要同时计入 Norm 输入分发、HBM panel 所有权和
同步成本。不能再假设 manager 拥有独立计算阵列来隐藏 Norm/投影开销。
