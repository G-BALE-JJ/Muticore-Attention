# Worker 启动实验与干净性能基线

## 问题来源

B 批次 IW 路径的 16 个 worker 接受时间跨度为 30,701 cycles。每个 manager 的四次 release 跨度为约 30,200–30,700 cycles，而控制信号传播仅为 9–25 cycles。

guest 的原循环为 `remote_store → fprintf → 下一个 remote_store`。`fprintf` 在被模拟的 CPU 上进行格式化和系统调用；其成本进入模拟时间，并改变下游任务提交及资源竞争。宿主侧资源 trace 仅影响仿真墙钟时间，两者不能混为一谈。

另有 `ATTENTION_GUEST_START` 和 `PROJECTION_START` 纯启动诊断。worker 的 observed→accept 约 40k cycles 包含 guest 准备和打印，不能直接全归因给描述符计算；后续干净版本实测它们去除后的时间。

## 分两步实验

第一步仅改变释放顺序：使用原控制通道连续释放四个 worker，然后打印相同的四条 `PROJECTION_DISPATCH`。保留其它 guest 打印、相同描述符/计算/输入与输出，验证错峰的原因。新编译的默认 guest 哈希与 B 批次逐字节相同，两组 IW 对照精确复现 544,931 / 1,304,066 cycles。

第二步关闭三个纯启动诊断：`ATTENTION_GUEST_START`、`PROJECTION_START`、`PROJECTION_DISPATCH`，使用编译设置 `GOLEM_GUEST_STARTUP_DIAGNOSTICS=0`。错误路径报告和最终 `FUSED_ATTENTION` 完成报告保留，NumPy/HBM 检查和宿主控制/phase/resource trace 保留。

关闭这些打印后，原循环与 batch-release 编译出的 guest 二进制完全一致。因此没有必要为干净性能路径维护两套释放调度。移除诊断的收益应解释为清理性能测量干扰，不是算法或硬件加速。

## 资源与测量契约

两步均使用 IW：input-pair prefetch=1、local-weight prefetch=1、array_groups=1，64 arrays、单 operand context、单 GM 读端口和原 DMA/缓存配额，所有 oracle 为 none。测试为 Hq32/Hkv8/D64/hidden2048、causal+RoPE、S1024/S2048。

全部正式输出与冻结 IW 基线逐字节一致，并检查实际 `array_launches`、`compute_busy_cycles` 和 weight programs 不变。输入、源码、两份仿真库、guest 和完整环境归档并保存 SHA256。

除原 E2E（首个 Projection job→最后 Attention worker 完成）之外，同时比较首个 release→完成、仿真起点→最后 O ACK、仿真起点→root 完成，以及仿真起点→所有进程退出。后三项包含 guest 准备/日志/收尾，不通过推迟计时起点宣称收益；没有从旧结果中直接减去打印开销。

## 保留日志的单变量结果

| 指标（cycles） | S1024 对照 | S1024 连续释放 | S2048 对照 | S2048 连续释放 |
|---|---:|---:|---:|---:|
| 原 E2E | 544,931 | 473,619 | 1,304,066 | 1,205,515 |
| 全部 worker 接受跨度 | 30,701 | 1,182 | 30,701 | 1,182 |
| 每 manager 四次 release 跨度 | 约 30k | 65 | 约 30k | 65 |
| 仿真起点→最后 O ACK | 655,192.25 | 583,204.875 | 1,414,327.75 | 1,315,100.25 |
| 仿真起点→全部进程退出 | 684,488 | 612,109 | 1,443,430 | 1,344,170 |

E2E 分别减少 71,312（13.09%）/98,551（7.56%）cycles。固定计算量，所有输出一致，实际 O ACK 和全部进程退出均提前。

收益不止来自启动跨度：S1024 的 worker 16，串行 phase 中 load_weight 从 55,024 降至 32,849，program_weight 从 14,333 降至 6,261，local_read 从 30,024 降至 17,604 cycles；program_weight_ready/input/launch 的固定工作不变。GM 读请求累计排队反而从 99,474 增至 111,471，实际 GM 读服务均为 105,808 cycles。说明等待更好地重叠、权重供给时序改善，而非 GM 服务加速。累计请求排队不能当作 E2E 的可加损失。

B 的同步启动 oracle 在 all-off 调度上测试；本实验在 IW 调度上改变真实提交顺序和资源竞争。两者不是相同反事实，不能将约 30k 的 oracle 差值作为本实验的严格收益上限。

## 构建与复现

```bash
make -C src/sst/elements/golem/tests/small/muticore_attention scale batch-release clean-startup

python3 baseline/attention_cluster_8qk_8pv/run_projection_clean_startup.py \
  --stage formal --sequence 1024 --cpu-set 0-3 \
  --reference results/startup_release_20261012/reference_formal/s1024 \
  --artifact-root results/clean_startup_reproduction_s1024
```

性能 guest 为 `riscv64/clean_startup/fused_attention`，通用 runner 通过 `GOLEM_ATTENTION_GUEST_EXECUTABLE` 选择。旧默认 guest 保留历史打印，以便复现已有结果；正式性能工作应显式选择 clean guest。

原始 case、源码快照、输入/输出、日志和测量结果均保存在 `results/startup_release_20261012`。保留日志实验的完成汇总为 `logged_release_summary.json`，两个编译阶段独立归档，阶段间代码变更不混为同一版本。

第一次 small 仿真完成后，汇总器误将 guest stderr 当作宿主 SST log，导致 dispatch 数量校验失败；已修正并补回归测试。失败产物 `small` 保留，最终 small 对照使用 `small_v2`，另有 `clean_small`。

## 干净性能基线结果

| 指标（cycles） | S1024 原日志对照 | S1024 干净版本 | S2048 原日志对照 | S2048 干净版本 |
|---|---:|---:|---:|---:|
| 原 E2E | 544,931 | 473,914 | 1,304,066 | 1,238,826 |
| 全部 worker 接受跨度 | 30,701 | 991 | 30,701 | 991 |
| 首个 release→Attention 完成 | 585,590 | 492,899 | 1,344,725 | 1,257,811 |
| 仿真起点→最后 O ACK | 655,192.25 | 551,750.75 | 1,414,327.75 | 1,316,662.75 |
| 仿真起点→root 完成 | 655,216 | 551,774.5 | 1,414,351.5 | 1,316,686.5 |
| 仿真起点→全部进程退出 | 684,488 | 583,436 | 1,443,430 | 1,348,300 |

相对原日志对照，原 E2E 减少 13.03% / 5.00%；包含准备时间的最终 O ACK 提前 15.79% / 6.91%。worker observed→accept 从约 40k 缩到 18,827–19,125 cycles，仍有真实 guest 准备工作，不能把剩余时间直接当作纯计算下界。

与“保留日志、连续释放”相比，S1024 干净版本的原 E2E 多 295 cycles，但实际 O ACK 提前 31,454.125 cycles，说明只看首个计算任务之后的 E2E 会漏掉启动前的节省。S2048 干净版本的原 E2E 多 33,311 cycles，实际 O ACK 晚 1,562.5 cycles（0.12%）；不能宣称所有指标单调改善。

S2048 这项差异主要落在 Projection：总阶段增加 33,059 cycles，Attention 仅增加 307，handoff 减少 55。两个 case 的关键 Projection worker 都是 core 12，其 load_weight phase 增加 22,840、local_read 增加 10,728 cycles；MAC、输入 scatter、权重编程固定服务量不变。GM 请求累计排队反而减少，不能据此否认暴露在串行路径中的读等待增加。证据支持启动时刻变化后供给/消费时序发生重排；具体 HBM/NoC/GM 跨层争用边仍需完整 DAG，尚未完成根因细分。

选择不含启动诊断的版本作为以后性能工作的对照，而不是用打印产生的隐含延时调节调度。输入＋权重预取此前是在有日志版本中选出的组合；本批没有重新运行干净版本的整个 2³ 矩阵，因此不把 IW 宣称为干净版本的全局最佳配置。

验证完成：6/6 正式 case、3/3 最终 small case，NumPy/HBM 输出及逐字节对照通过，实际 Projection 计算量一致；66 项工具测试＋140 项 Attention 测试通过。三个启动诊断的字面量在 clean 二进制中不存在，错误/完成字面量保留；两个开关的非法编译值会被拒绝。模型仿真库无需修改或重建。

完整表格：[summary.md](../../results/startup_release_20261012/summary.md)，机器数据：[summary.json](../../results/startup_release_20261012/summary.json)。工具入口：[run_projection_startup.py](run_projection_startup.py)、[run_projection_clean_startup.py](run_projection_clean_startup.py)、[summarize_projection_clean_startup.py](summarize_projection_clean_startup.py)、[analyze_projection_startup_resources.py](analyze_projection_startup_resources.py)。

下一步应在干净版本上重新核算理论下界差距，并补齐权重供给、GM 读取、消费之间的依赖和争用边；本批出现的 S2048 时序敏感性是直接证据。不能继续基于旧日志开销扩大硬件瓶颈判断，也不能把移除打印的差值与其它 oracle 节省相加。
