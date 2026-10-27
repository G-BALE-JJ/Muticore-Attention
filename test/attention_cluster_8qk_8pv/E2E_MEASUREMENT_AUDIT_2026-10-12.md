# 端到端 cycle 统计审计（2026-10-12）

> 后续口径澄清：用户指定从首个有效数据 DMA 实际发出开始计时。本报告建议的 release→root 主口径已由 [DMA 统计口径](DMA_TIMING_CONTRACT_2026-10-12.md) 替代；本报告原始证据与干扰项分析仍有效。

本轮暂停数据流优化，检查 guest、模型计时、资源统计、报告解析和完成边界。复查对象为 `results/startup_release_20261012/clean_formal_s1024` 与 `clean_formal_s2048`；只读取已完成实验，不改变归档结果、不运行新的性能实验。

结论：现有 clean-startup 的 cycle 算术与原始事件一致，但它只去除了启动诊断，仍不能认为所有测量干扰都已排除。发现末尾 guest 打印可能与全局剩余计算重叠；原来的 E2E 和“理论下界”也需要重新定义。

## 1. 发现与证据

| 项目 | 审计结论 | 处理方式 |
|---|---|---|
| 启动 guest `fprintf` | 已由上一轮对照证实会串行错开 release；本轮核验两个 clean 用例确实没有三类启动日志 | 使用关闭诊断的 guest；旧结果仅作历史对照，不事后减去打印时间 |
| 最后的 guest `printf` | **仍待隔离**。非 root manager 只等本地 worker 完成和 completion message 发出，随后返回 guest 打印；不等待全局 root 完成 | 下一轮只关闭成功路径末尾打印做对照；错误路径保留。不能宣称已证明它造成多少延迟 |
| guest 完成标志轮询 | `finish_projection()` 每个 manager 等全部 16 个 flag，4 个 manager 共 64 次成功等待；未就绪时 `waitHbmFlag()` 会重复发 8B DMA read | 这是同步协议和真实资源竞争，不能当诊断开销删掉；以后作为同步优化单独研究 |
| guest 描述符准备/写入与控制命令 | 位于 release 之后、job 接受之前；会消耗 guest CPU 与 GM/控制链路资源 | 计入新的请求执行指标；约 1.9 万 cycle 不能全部归因于描述符，也不能全部扣除 |
| fused norm/raw 审计输出 | 两个用例全部 16 个 worker 的 `audit=0`、`norm_audit_bytes=raw_audit_bytes=0` | 当前没有这两类用于校验的额外模型输出流量；真实 QKV panel、RoPE、O 写回继续保留 |
| host 资源日志 | `resourceTrace.h` 在模拟事件内执行 host `std::printf`，函数本身不发模型请求、不调度新事件 | 直接开销是 wall time。整条调用路径是否完全无模型时序副作用，仍需 trace on/off 对照，不能仅凭 helper 注释宣布排除 |
| host NumPy/HBM 布局验证 | shell runner 在 SST 返回后执行 | 不直接进入模拟 cycle；耗时只属于 pipeline wall time |
| 时钟/单位 | Projection 报告存在写死 `/1000` 的 tick→cycle 换算；本次 CPU/array 均 1GHz，原始 `RESOURCE_TIMEBASE=1000`，因此当前换算成立 | 新审计入口拒绝混合或非 1GHz 配置；未来支持其他频率必须先重构换算，不能直接复用当前报告 |
| oracle / 归档一致性 | 两个用例均 oracle=none；每例 34 项归档源码/二进制 SHA256 一致，输出 PASS | 可作为当前审计依据；不代表末尾打印已无干扰 |
| 资源累计周期 | 不同 worker、资源及 transport/landing 区间可能重叠、嵌套；phase 包含等待与轮询 | 不将累计 DMA、program、MAC、queue 周期相加作为端到端“损失”；继续使用事件依赖与区间交集 |

### 末尾打印为什么仍值得检查

源码 `golem_attention_runtime.cpp` 在全部本地 `attention_manager_wait(tag)` 返回后打印 `FUSED_ATTENTION status=...`。`roccAnalog.h::handleManagerAttentionCompletion()` 对非 root manager 在成功发送本地 completion 后将 job 标记 Complete；root 则等待所有 manager completion。

原始 milestone 证明非 root manager 可以提前离开等待：

| 序列 | manager 1 距最后 worker 完成 | manager 2 | manager 3 |
|---|---:|---:|---:|
| S1024 | 6,487.625 | 7,704.625 | 4,835.625 |
| S2048 | 2,160.625 | 912.625 | 18,247.625 |

单位为 cycle。这些数是“允许打印与剩余计算重叠的窗口”，不是打印耗时，也不是可获得的加速。现有日志没有 guest 格式化和 syscall 的精确起止时间，因此必须对照，不能做事后扣减。进程退出也可能在其他 manager 仍执行时发生，要与打印分开诊断。

## 2. 明确端到端边界

旧报告的 `end_to_end_cycles` 从首个 Projection job 开始，结束于最后 Attention worker 的 component-cycle 完成记录。它不包含 release 后的准备过程，也没有覆盖 root 完成确认。原字段保留用于历史复现，之后称为 **job→worker 完成**。

| 指标（cycles） | S1024 | S2048 |
|---|---:|---:|
| 旧 job→worker 完成 | 473,914 | 1,238,826 |
| 首次 release→首个 job（旧指标漏计） | 18,985 | 18,985 |
| 首次 release→最终 O 写回 ACK | 492,887.75 | 1,257,799.75 |
| **首次 release→root 全局完成** | **492,911.5** | **1,257,823.5** |
| 仿真零点→最终 O ACK | 551,750.75 | 1,316,662.75 |
| 仿真零点→root 完成 | 551,774.5 | 1,316,686.5 |
| root 完成→仿真退出 | 31,661.5 | 31,613.5 |

后续固定 workload 的主要执行延迟使用 release→root 完成，同时报告 release→O ACK 与从仿真零点到 O ACK/root 完成，避免起点后移造成假收益。首次 release 是当前可观测的控制边界；并非外部请求到达点，也不覆盖此前 CPU 启动、参数读取和 manager 准备。优化这些前置过程时必须检查绝对完成时间，未来若要报告完整请求响应延迟，需要加入真实请求入口事件。

进程退出时间保留为测试生命周期指标，包含成功信息格式化、syscall 和退出，不作为 attention kernel 延迟。component-cycle 完成记录会截去不足一个 cycle 的部分；本轮已与原始 SST tick 交叉检查，差异严格小于 1 cycle，不把舍入误差当加速。

## 3. 原来的“下界”不能继续当最优目标

`report_projection_e2e.py` 计算：

`max(worker RMSNorm vector work) + ideal projection launches × full-width MAC latency + estimated attention EXP work / lanes`

但当前 RMSNorm 已融合进 Projection，二者可以重叠。只有证明全局依赖强制这两段全部串行，才能把两项相加作为下界；当前报告没有这个证明。报告也没有完整覆盖 Attention QK/PV MAC、实际 mask/EXP 工作、共享端口、DMA/NoC 带宽、关键依赖与控制开销。S1024 的 268,592 和对应 1.764 倍只保留为历史**限定范围工作量估算**，不解释为“还可加速 1.764 倍”。

下一步用同一计时窗口和实际工作量计算各资源容量下界，并用依赖路径下界约束，允许重叠的部分取最大值、仅在已证明的阶段屏障处串接。先逐项确认 SFU 实际 EXP 工作及其因果 mask 口径；不凭 tile 数量直接假设全部元素有效。完整依赖 DAG 尚未建立，本轮不声称已给出严格最优下界。

## 4. 审计脚本与验证

新增 `audit_e2e_measurement.py`：核验归档 hash、1GHz 时钟合同、原始 tick/component-cycle 关系、oracle 关闭、审计写流量关闭、启动日志缺失、四个成功 guest 状态、各 manager 的 Hkv 次等待、E2E 算术和完成事件顺序。输出明确保留 pending isolation，不将 PASS 解释为“已排除全部干扰”。

```bash
python3 baseline/attention_cluster_8qk_8pv/audit_e2e_measurement.py \
  results/startup_release_20261012/clean_formal_s1024 \
  results/startup_release_20261012/clean_formal_s2048 \
  --output results/e2e_measurement_audit_20261012/summary.json
```

两个正式归档通过；5 个拒绝错误计时配置/诊断配置的回归测试及现有 10 个启动分析测试，共 15 个通过。未修改模型、旧报告、历史结果或 guest 二进制。

## 5. 后续执行顺序

1. 测量隔离：以当前 clean-startup IW 为对照，只关闭成功路径末尾打印，保留计算、同步、错误处理和原进程退出行为。成功判断移到 host 的模型完成计数、状态和数值检查，避免为了证明成功重新引入 guest 打印。S1024/S2048 都测 release→root、O ACK 和绝对完成时间，检查工作量/输入/输出不变。
2. 观测隔离：固定上一步 guest，只切换 host trace，使用独立 lifecycle stats 和数值结果比较模拟时间；trace-off 时不能依赖被关掉的 resource log 当计时来源。如果退出仍提前影响其他 manager，再作为独立变量研究，不能与关打印混成一组。
3. 固定计时协议与干净基线，完善各资源容量/依赖路径下界，再回到软件数据流实验。将启动准备、全局 flag barrier、HBM/GM 等待与 program/scatter/MAC 串行区分开，按真正处在关键路径上的可消除等待排序。

前两步是测量系统验证，不计作算法优化收益。确认这些干扰的影响后，才继续性能优化。
