# 固定 DMA 口径与重新计算资源下界

## 已固定的口径

主报告为 `dma_e2e_result.json`，计时合同版本 `hbmdma_to_O_ack_v1`：完整 RMSNorm→QKV Projection→RoPE→causal Attention workload，从首个必要 HBM 数据读 DMA 成功发送到 NoC，到覆盖所有最终 O 分块的最后写回 ACK。此前 guest 准备与此后 root 通知/打印/exit 单列；窗口内所有传输、同步、等待和流水排空保留。时长取统一 SST tick 差，不加 inclusive +1。

`report_dma_e2e.py` 按 X/gamma/norm/weights、RoPE 表及 QKV 的数据地址区间筛选输入，排除描述符/mailbox/flag；按经过数值与布局验证的四个 O 区间检查 ACK 覆盖完整性。验证解析出的 CPU/array 1GHz 配置与 ticks_per_cycle=1000。该版本明确支持当前 square、causal、D64 Projection 路径；其他频率/地址 ABI 不静默套用。

`run_sst.sh --projection` 默认启用资源时间戳；worker-cluster shell runner 在旧阶段报告后生成新的 DMA 主报告，并停止把旧 floor/ratio 打印成理论目标。历史 `projection_e2e_result.json` 与依赖其阶段重构的字段保留兼容，旧 console 文案改为 legacy job-to-worker。实验 helper 增加 `primary_end_to_end_cycles`、`primary_timing_contract` 与 DMA→root，旧 `end_to_end_cycles` 同时明确标为 legacy；新性能比较应读取主报告/primary 字段，不能与旧字段混用。

没有资源事件的运行不输出伪造的 DMA 主指标，不回退到 job/release 起点。当前版本仍需要 host trace 来获得时间戳，trace=0 的观测隔离实验必须用独立 lifecycle stats 验证，不能声称由此 reporter 得到了同口径指标。未修改计算模型或新增硬件。

## 当前延迟

依据已有 clean-startup、IW、64-array、单 operand context、array_groups=1、4 manager + 8 QK/SFU + 8 PV 的正式归档重新生成报告；不是新的模拟或优化收益。成功路径末尾打印仍存在，其影响尚未做隔离对照。

| 1GHz cycles | S1024 | S2048 |
|---|---:|---:|
| **DMA→最终 O ACK，主指标** | **473,902.75** | **1,238,814.75** |
| DMA→root 完成，辅助 | 473,926.5 | 1,238,838.5 |
| DMA 起点（绝对时间） | 77,848 | 77,848 |
| 最后 O ACK（绝对时间） | 551,750.75 | 1,316,662.75 |
| O ACK→root | 23.75 | 23.75 |

## 下界必须明确约束

**本轮得到可核验的必要资源下界，不能将其称为已证明可达到的最短延迟。** 真正的最优执行时间还须联合求解完整数据依赖、HBM bank 与 NoC 竞争、单 operand context 生命周期、有限缓冲、控制提交与填充/排空。原先用几个工作量直接相加的 floor 不满足这个要求；不能仅为给出“精准数值”再制造一个最优点。

以下三层全部使用当前实际工作量、现有资源和 QK/PV ownership，不增加硬件。不对队列等待或当前测得的长请求 RTT 宣称不可避免；容量用实际资源服务工作，queue/RTT 不混入。

| 约束层级 | S1024 下界 | S2048 下界 | 含义 |
|---|---:|---:|---|
| **保留整组阵列顺序与全局 Projection→Attention 屏障** | **352,816** | **967,776** | 当前 groups=1 数据流的必要下界 |
| 放宽整组阵列顺序、仍保留阶段屏障 | 251,440 | 765,024 | 各阶段取资源瓶颈，再串接；放宽了单 context 排程约束，不能保证软件可达到 |
| 进一步放宽阶段屏障，允许跨阶段流水 | 143,344 | 540,672 | 全窗口资源容量下界，很乐观；不是跨阶段流水的预测延迟 |

如果改变重用、请求分块、mask 计算或 QK/PV ownership，则实际工作量或资源容量变了，必须重新计算，以上固定工作下界不直接继承。

### Projection 工作与约束

每个 worker 的完整 64-array launch 数：

`N = (S / 16 / 64) × (Hq + 2Hkv) × (hidden / 64)`

Hq=32、Hkv=8、D=64、hidden=2048，S1024/2048 得到 N=1536/3072。

从模型公式与物理事件交叉验证：

- full-width MAC：`64 / 1 + pipeline_depth 2 = 66 cycles`。
- 8KB matrix broadcast：`base 1 + 8192/256 + log2(64) × stage 1 = 39 cycles`。
- 8KB input scatter：`base 1 + 8192/256 = 33 cycles`。
- 每 head/block 一次 8KB 输出 gather：S1024 每 worker 48 次、S2048 96 次，每次 33 cycles。
- local GM read 服务为 `base 1 + ceil(bytes / 256)`，一读端口；取服务工作，不取排队等待。

groups=1 路径先等 matrix program，再等 input scatter，再 launch MAC，计算完成后推进下一 tile；输出 gather 位于 head/block 尾部。源码状态机与归档 interval 交集均表明这四类阵列工作没有重叠，因此当前顺序约束下：

`P_serial = N × (39 + 33 + 66) + gather_count × 33`

S1024 为 **213,552**；S2048 为 **427,104**。GM read 与 RMSNorm 可在旁路重叠，所以不能再把它们直接加上去；应比较这些资源是否更慢。

| Projection 资源工作下界 | S1024 | S2048 |
|---|---:|---:|
| MAC | 101,376 | 202,752 |
| matrix broadcast | 59,904 | 119,808 |
| input scatter | 50,688 | 101,376 |
| output gather | 1,584 | 3,168 |
| **共享 array buffer 单端口（三类 transfer 相加）** | **112,176** | **224,352** |
| GM read 单端口 | 105,808 | 211,616 |
| GM write 单端口 | 54,609 | 109,201 |
| RMSNorm vector 工作参考量 | 27,952 | 55,904 |

matrix/scatter/gather 都进入 `enqueueModeledBufferTransfer(..., false, ...)`，共享一个 buffer 端口，所以这三项容量占用可以相加。MAC 与该端口是不同资源，若放宽整组顺序，取最大值而不是相加；因此放宽后的 Projection 容量下界为 **112,176 / 224,352**。这只是 relaxation：在单 operand context 上如何实现重叠还须满足 bank 依赖，不能用该数字保证速度。

RMSNorm vector_cycles 是已有工作量参考，本轮不把它当独占 SFU 服务后与其他 engine 相加；在两例中它也不决定 Projection 最大项。

### Attention 的 EXP 下界

在每个 QK/SFU core 上读取 `GOLEM_SFU_HW_PIPELINE unit=exp` 的实际 accepted_tokens、lanes、II、latency；八个 core 均 lanes=16、II=1、latency=8，工作已均衡：

| EXP 工作 | S1024 | S2048 |
|---|---:|---:|
| 实际执行元素 | 17,825,792 | 69,206,016 |
| 逻辑有效元素 | 16,793,600 | 67,141,632 |
| 被 mask 但仍执行的元素 | 1,032,192 | 2,064,384 |
| **每个 SFU 的 token issue 工作** | **139,264** | **540,672** |

`A_EXP = max_core(accepted_tokens × II)`，与 observed_elements / (8×16) 一致。masked score 先变成 −∞，当前代码仍送入 EXP pipeline，不能只算逻辑三角形。假设完美消除 masked 元素，省下的 EXP 工作至多为 8,064 / 16,128 cycles；这还不是可直接兑现的 E2E 收益。

当前 Attention 其他已核验容量项：array group compute 最大 42,240 / 152,064；GM read 最大 37,536 / 144,704；GM write 最大 3,264 / 29,512，均不超过 EXP。故可证明的 Attention 容量下界为 **139,264 / 540,672**，未额外加入测得的 SFU issue_wait、窗口空洞、NoC P transport 等等待。

### 怎样组合

当前 manager `finish_projection()` 等所有 producer flag 后才提交 Attention，所以两个阶段有全局依赖屏障，可以串接。

`L_current = max(P_serial, P_GM_read, P_GM_write, ...) + A_EXP`

得到 352,816 / 967,776。旧报告把 fused RMSNorm 与 Projection MAC 直接相加的办法被弃用。

取消阶段屏障时，各阶段工作可交错，只能在同一物理资源上累计服务，再对资源取最大。本轮全窗口 GM read 单端口工作为 143,344 / 356,320，EXP 为 139,264 / 540,672，得到乐观的 143,344 / 540,672。

特意不将 `array.compute` 的 QK/PV group interval 全部加到 Projection MAC 上：它是 group 区间，没有每个 lane 的占用信息，直接相加会错误地禁止现有不同阵列之间的并行。该累计量仅保留为诊断字段，不作为跨阶段硬件容量下界。

## 当前差距在哪里

| 与当前顺序约束下界比较 | S1024 | S2048 |
|---|---:|---:|
| 实测/下界 | 1.3432× | 1.2801× |
| 主延迟超过必要下界 | 121,086.75 | 271,038.75 |
| 其中 Projection 超出 P_serial | 104,082 | 257,536 |
| Attention 窗口超过 EXP 容量项 | 15,431.75 | 10,276.75 |
| Projection→Attention handoff | 1,573 | 3,226 |
| **Projection 占上述差距** | **85.96%** | **95.02%** |

这三项分解是同一窗口的精确算术分解，不是已经证明全部可消除的等待。Projection 超出串行阵列服务的部分仍包含必要数据依赖、GM/HBM 等待、RMSNorm 与调度推进；要判断哪些可删，继续用关键路径证据。Attention 在 S2048 已接近 EXP 吞吐容量；此时反复微调 Attention 调度的潜力比 Projection 小。

因此后续优先级是：先隔离末尾打印，随后对 Projection 的访存到可编程时刻、GM read 队列和阵列推进空洞做关键路径分析。整组阵列顺序约束是另一层潜力，但解除它必须满足单 context 生命周期，并评估分组增加的 broadcast 工作，不能只看到容量 relaxation 就承诺收益。跨阶段流式执行则是再下一层，涉及真实 producer/consumer 可见性，不能用原来的有限 barrier oracle 代替证明。

## 产物与验证

`results/dma_bound_20261012/s1024`、`s2048` 各有 `dma_e2e_result.json` 和 `lower_bound.json`。结果保留 per-core EXP、分阶段服务工作、假设与尚未覆盖的约束。报告与分析脚本已加入未来实验的 source snapshot 清单。

重放命令只分析已有归档：

```bash
python3 baseline/attention_cluster_8qk_8pv/report_dma_e2e.py --case CASE --output OUTPUT
python3 baseline/attention_cluster_8qk_8pv/analyze_dma_lower_bound.py --case CASE --output OUTPUT
```

5 项主边界回归涵盖：提前 metadata DMA 排除、以 send 而非 submit 起算、分块覆盖、分数 cycle、非法时钟及事件顺序；3 项下界回归涵盖 fused overlap、端口瓶颈和正式工作量。baseline 共 79 项、原 Projection 报告 4 项通过，两个 shell 脚本语法检查通过。本轮没有新增 SST 性能实验，因此自动运行入口的 shell 集成已检查语法，尚未通过一次新模拟完整执行；两个正式归档的主报告与下界分析均已实际运行通过。
