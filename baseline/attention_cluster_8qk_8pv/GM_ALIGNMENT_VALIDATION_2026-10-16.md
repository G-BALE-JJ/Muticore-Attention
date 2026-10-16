# GM banked 模型与验收（2026-10-16）

本页记录 [对齐审查](GM_ALIGNMENT_REVIEW_2026-10-16.md) 的实现与验证。本轮约定的模型对齐与验证已完成：生产微测试、参考 RTL 对照、局部筛选、MPI 一致性及正式两层完整验收均通过。
每 worker GM 保持 **2 MiB**。本轮属于时序模型校准，不是软件优化收益；历史 6,231,011.75-cycle 结果属于 legacy 模型。

## 最终资源合同

- 8 banks × 256 KiB，line=128 B；`bank=((addr-base)/128)%8`，没有因扩容而增加 bank 数。
- 计算/控制来源共用入口 0；DMA 读出、普通 DMA 落地及控制器发起的 DMA 落地共用入口 1。每入口每 cycle 最多一条 line，读写合计最多 256 B/cycle。
- 每 bank 每 cycle 最多接受一个读或写。相撞时计算入口优先，采用参考 RTL 的 client-index 顺序；不同 bank 可以并行，不要求等待前一读返回。
- API 请求仍可达 4 KiB，有界 pending=256；逐 line 发射，非对齐访问按实际跨越 line 数计费，部分写保留其他字节。
- 读在 grant 边沿采样，写在 grant 边沿可见。整个 API 请求的所有 beat 返回后才调用 callback。不是原子 4 KiB 读写；有依赖的数据流必须等待完成。
- 返回延迟默认 1 cycle，beat 返回容量 32。返回队列满则停止发射，不能通过增加软件 job 获取额外入口或 SRAM 端口。写 callback 也按同样的有界完成路径返回。
- 接收同 tick 请求后在下一个全局 GM 时钟边沿仲裁；新请求不能靠 C++ 调用先后抢占当前边沿。旧请求完成 callback 与新提交的执行顺序也不能改变该规则。
- 不叠加 legacy 大请求传输延迟。`local_access_bytes_per_cycle/read_ports/write_ports/base_latency_cycles` 仅控制 legacy；banked 使用独立的 line/bank/return 参数。

`run_decoder_stack.py`、`run_decoder_micro.py` 默认选择 banked；可用 `--gm-model legacy` 复现旧合同。通用 C++ 子组件保留 legacy 兼容默认值，正式 baseline 的 `GM_PROFILE` 显式选择 banked。每次运行必须核对 provenance 与日志，不能只相信目录名或命令行参数。

## 实际补齐的遗漏与保留的控制路径

普通 HBM→GM DMA 原本已有 timed landing。但 WCP 经控制器发起、由 `request_pending` 跟踪的回复，原先在接收分支直接写入 GM 并立即发布 ready。实际 Attention 用例确实经过此路径，存在 8 KiB 矩阵/向量直写。

banked 模式把这些回复统一放入现有 DMA landing 队列，拆分为不超过 4 KiB 的 `localWriteAsync(Dma)`；只在所有网络块完成 SRAM 落地后累加 received bytes、发布 ready 并通知控制器。完成前不得复用缓冲区。记录请求身份与 generation，防止旧 callback 完成复用的 request ID；取消时同时清理控制请求注册。legacy 保持原行为供对照。

描述符/状态控制页 `[0,0x8000)`、DMA 状态尾区的 64-bit 寄存器访问仍按原控制协议处理。它们承担命令解码、状态轮询或 completion 发布，不宣称已经过 SRAM bank 数据端口。其 RoCC/CPU/网络协议时间保留；本轮没有把所有控制操作机械拆成 SRAM beat。实际张量区的 `rd_from_globalmem/wr_to_globalmem` 直访会记录为 `untimed_read/write`，验收器要求 **0 B**。`GlobalMemoryLocal` 是功能 mock，不是正式拓扑的 GM。

## 证据与测试

- `probe_gm_timing.py` 原样提取生产 GM 的入队、调度、采样、写入、DMA 分块及 ready 发布方法，替换 SST 周边事件环境。验证 legacy 时序保留、同 bank 冲突、同 tick 顺序、RAW/WAR、跨 line byte mask、4 KiB 拆分、pending/return 满、取消复用、时钟相位、控制 DMA 的多包乱序/重试/ready 顺序。它不是物理 RTL 验证。
- `run_gm_reference_rtl.py` 直接编译参考工程未经修改的 bank cluster 与 `GOLEM_SRAM_SIM` SRAM，用 2 MiB / 2 clients 参数覆盖；包含低地址与最后 16 条 SRAM line。随机 9,999 周期、13,209 grants、6,641 reads、634 同 bank 冲突，逐周期 grant、返回时刻、完整 line 数据和部分写对照全部通过。没有把参考 HDL 中简化的 DMA/NoC 当作全架构周期规范。
- `verify_gm_banked.py` 同时读取合并主日志与 `.resources.jsonl`，检查 20 个 GM 均为 2 MiB、物理 beat/逻辑 payload 字节守恒、逐请求跨 line 数与两入口 beat 数逐项对应、bank grant 总数、返回水位、有界请求排空、实际服务/返回时钟边沿以及张量直访为零。缺少资源记录直接失败；负例覆盖时钟错误、残留 pending、容量缩水、计数不守恒与数据绕行。
- 142 个 baseline host 测试和 154 个 attention/decoder 测试通过；另有 5 个新审计负例测试。

小型两层 MPI2 与 MPI8 的总 cycle 均为 **675,639.625**；所有 DMA 阶段时间相同，生命周期记录按任务排序后相同，四个 HBM 输出文件逐字节一致。该检查验证 rank 数不改变所模拟的硬件时序与数值；不同运行的 host wall 仅作记录。

### 构建一致性

`memHierarchy/memNICBase.h` 包含 GM ELI 类型注册，所以修改 `GlobalMemoryImplement` 对象布局后必须同时重编 **golem 与 memHierarchy**。只重编 golem 会使工厂按旧 sizeof 分配对象，造成初始化堆损坏。本轮完成两库 clean rebuild，恢复正常实例成员实现，没有保留调试期间的类外状态表。完整运行 provenance 同时记录两个动态库哈希。

### 局部验收与延迟假设

当前已通过的小型两层 MPI8：包含 Wo、两次残差、FFN、层间 reset/sync、Embedding、final RMSNorm、M=1 LM head、两段词表输出；最终 token=5313。张量区未计时访问为 0 B，94,773 条逻辑 GM 区间满足时钟边沿与生命周期要求。完整 token IDs DMA→logits ACK 为 **675,639.625 cycles**，SST wall **40.136 s**。

正式单 worker FFN（H=2048、I=8192，两组阵列、四槽预取、down+residual 融合）通过：**2,375,059.625 cycles**，SST wall **119.009 s**；77,376 条逻辑 GM 区间通过审计；张量直访为 0 B。

正式 16-worker / MPI8 FFN 对照（H=2048、I=8192、S=1024）：

| 同一 banked 模型下的调度 | cycles | SST wall | 数值与 GM 审计 |
| --- | ---: | ---: | --- |
| 一组阵列、四槽预取 | 2,634,007.5 | 461.149 s | PASS |
| 两组阵列、四槽预取 | 2,388,853.625 | 429.256 s | PASS |

两组调度减少 **9.307% cycles**，HBM 读写量相同。这是同模型下可比较的软件调度收益。两种方案都没有增加阵列、GM bank、端口或容量。

每个工作 worker 的计算入口需要 **1,679,360 个 GM beat cycles**，超过纯 MAC 下界 **786,432 cycles**，也超过既有阵列 buffer 端口下界 1,375,968 cycles。一组方案为 GM 计算入口必要下界的 1.568 倍；两组为 1.422 倍。该下界按实际流量计数，不意味着所有 beat 都能无依赖地连续发射。不能将可重叠的 GM、MAC、HBM 下界相加，也不能直接把逻辑请求区间长度当作 SRAM 物理忙碌时间。

小型同软件 FFN 的 SRAM 返回延迟敏感性：

| grant→返回 | cycle | 相对 1-cycle |
| --- | ---: | ---: |
| 1 cycle | 124,654.625 | 基准 |
| 2 cycles | 125,342.125 | +0.552% |
| 4 cycles | 126,960.125 | +1.850% |

这只说明该局部形状的敏感性，不能外推所有 workload。2 MiB SRAM 在 1 GHz 是否满足一周期读返回仍是显式物理假设，需要 SRAM 宏/实现时序支持；RTL 行为对照不能替代物理时序签核。

## 正式两层最终验收

权重沿用可独立重建的合成稀疏值，所有 dense 硬件 tile 均执行；实际 checkpoint 导入仍按此前决定暂缓。

配置：S=1024、H=2048、I=8192、两层、词表 128256，16 workers / MPI8；四槽预取、两组阵列 FFN 调度、down+residual 融合、LM N 并行度 4。每 worker GM **2 MiB**，4 个 HBM 数据节点各 **2 GiB**；阵列/SFU/NoC/HBM 带宽/C-buffer 配置与本轮局部筛选一致。

**总延迟 7,483,267.75 cycles**，范围为首笔 token IDs DMA 到最后 logits 输出 ACK。SST wall **1,161.668 s（19 分 21.668 秒）**，包含准备后 driver/验证约 **1,208.416 s**。本次 HBM mmap 输出使用 tmpfs，不将该 wall time 与磁盘 backing 的旧运行直接作为仿真加速对照。Tokenizer、host 权重装载和 host argmax 不计入加速器 cycle。

| 可加和阶段 | cycles |
| --- | ---: |
| Embedding | 27,730.000 |
| Embedding → L1 交接 | 55,903.000 |
| L1 前半段（Norm1/QKV/RoPE/Attention） | 546,598.875 |
| L1 后半段（Wo/残差/Norm2/FFN/残差） | 2,776,402.500 |
| 层间交接与同步 | 52,523.625 |
| L2 前半段 | 536,731.875 |
| L2 后半段 | 2,765,787.500 |
| 最终 Norm 交接 | 15,525.625 |
| 最终 RMSNorm | 54,307.750 |
| LM head 交接 | 9,855.250 |
| 最后 token 的 LM head | 641,901.750 |

这些阶段按 DMA ACK 边界划分，可加和；阶段内部 job spans 有重叠，不能继续直接相加。

验收结果：

- 两层、最终 Norm、LM head 的独立 CPU 参考均通过，greedy token=**99037**。组合误差遵守原容差（两层最大绝对误差 0.0009765625、最终 Norm 0.00390625）；不把这些结果宣称为与数学 FP32 oracle 逐位一致。
- 与旧 legacy 全量运行比较，两层输出、最终 Norm、logits **FP16 逐位一致**；Wo、第一残差、第二残差/融合 FFN 的局部参考最大误差为 0。
- 两层 flags/reset、各 epoch、层间与词表块复用/ACK 均通过；**416 个 decoder jobs** 的 DMA 字节与完成生命周期审计通过。
- **3,475,464 条逻辑 GM 请求**的时钟边沿、跨 line 数与计算/DMA 两入口 beat 计数对应。20 个 GM 的物理 bank grants、payload 字节守恒，结束时 pending/queued=0；返回水位最多 2/32，返回满阻塞为 0。
- 张量区绕行未计时访问 **0 B**。控制页例外见前文，不将功能状态轮询宣称为数据 SRAM 端口工作。

### 校准后的下界与利用率

| 必要下界 | cycles | 总延迟 / 下界 |
| --- | ---: | ---: |
| 纯有用 MAC / 峰值 | 1,970,152 | 3.798 |
| 实际流量的最忙 GM 入口（1 beat/cycle） | 4,470,405 | 1.674 |
| 实际 HBM 节点 payload / 320 B/cycle | 3,562,969.275 | 2.100 |

GM 下界按端到端区间内逐请求 line beat 计数。HBM 取四个数据节点各自读写总量的最大值；两笔跨测量边界的 8 B root 状态读被单独记录并排除。三个下界可重叠，不能相加。

全链路有用 MAC / 峰值为 **26.33%**；worker 计算入口实际 grant 忙碌占比 **59.10%–59.74%**，DMA 入口 **29.59%–30.42%**。这是各自资源的比例，不可相加；MAC 工作量效率也不等于实测阵列占用率。全量 bank 冲突累计 110,512（跨 core 汇总），返回阻塞为 0，说明当前实验没有触发返回容量瓶颈。

旧 legacy 总延迟 6,231,011.75 → banked **7,483,267.75（+20.097%）**，是模型校准影响，不能称为软件优化回退或收益。校准后优先考虑已有片上容量内的数据复用、减少重复 GM 访问和流水重叠；不能仅凭 MAC 利用率低就扩大硬件端口。本轮完成验证，不追加硬件资源。

## 宿主机 backing 存储

正式形状每个 HBM 数据节点的逻辑容量仍为 2 GiB。初次全量运行在 memHierarchy mmap backing 构造中触发 SIGBUS，尚未进入计算；根分区可用空间同时下降至约 2 GiB。保持相同输入、硬件参数和动态库，将本次 HBM 输出映射移至 `/dev/shm` 后可正常越过初始化。此变更只涉及宿主机工件存储，不改变模拟 HBM 时序；wall time 需注明 tmpfs 条件。初次失败日志保留在完整用例的 `initial_backing_failure/`，不作为有效性能结果。最终输出以保留字节内容的 sparse 文件归档回结果目录。

## 复查入口

所有本轮工件位于 `results/gm_alignment_20261016/`；正式接受结果只使用 `accepted_*`、`latency1/2/4`、`reference_rtl/result.json`、`production_probe.json`、`accepted_build.json`。其他目录为定位失败或候选中间版本，不能混入最终 cycle 对比。轻量 JSON/测试日志归档在 `progress_results/2026-10-16/`，`acceptance_manifest.json` 记录各工件 SHA-256、验收范围与最终状态。

```bash
python3 baseline/attention_cluster_8qk_8pv/probe_gm_timing.py --output results/gm_alignment_20261016/production_probe.json
python3 baseline/attention_cluster_8qk_8pv/run_gm_reference_rtl.py --artifact-root results/gm_alignment_20261016/reference_rtl
python3 baseline/attention_cluster_8qk_8pv/verify_gm_banked.py results/gm_alignment_20261016/accepted_small
```

参考 RTL 工具使用 Verilator（本轮 4.038，`--unroll-count 1024`），脚本支持 `--verilator` 指定可执行文件。用户目录解包工具还需要设置 `VERILATOR_ROOT`；不需要修改参考源码或系统级安装。
