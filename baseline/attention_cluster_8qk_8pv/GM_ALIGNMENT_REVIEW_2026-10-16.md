# GM 合理性与对齐方案（2026-10-16）

后续实现与验证见 [GM banked 模型与验收](GM_ALIGNMENT_VALIDATION_2026-10-16.md)。下文保留初始审查时点的记录。

本轮是源码审查与建模方案，不修改生产时序、硬件参数或既有软件调度，不启动全局 SST 仿真。
用户明确要求每 worker 保持 **2 MiB = 2,097,152 B**，不改为参考 HDL 的 512 KiB。
目标是让共享资源、依赖与冲突有明确且有界的解释，不机械复制参考工程的所有参数。

## 对齐依据与结论

参考工程 `/data/jjgong/RISC-V-CIM-Manycore-SST` 同时包含两种抽象：

- SST C++：`GlobalMemoryImplement::rd_from_globalmem/wr_to_globalmem` 直接读写 backing storage；WCP 的 `loadTilePayload()` 直接调用它们。本地 SRAM bank 仲裁没有在这些调用中独立计时，调用方还可能计其他操作的延迟。因此不能把参考 SST 当作完整 bank 时序规范。
- HDL：`tests/architecture/HDL/README.md` 明确是面积/功耗估计用的资源实现，不是 SST 的 cycle-exact 替代。GM cluster 可以作为银行组织、握手和仲裁的依据，但 DMA、NoC、控制器等不能自动全部当作当前性能模型的规范。

当前项目正式 workload 使用有时序的 `localReadAsync/localWriteAsync`，读写各一条 FIFO、各一个端口；延迟与端口占用均为 `1 + ceil(bytes/256)` cycles。它不是无限带宽模型，但没有地址相关的 bank 冲突。
相对参考 HDL，有些访问偏乐观，有些偏保守；不能假设误差抵消，也不能预判对齐后的端到端 cycle 增减。

## 本轮纠正的比较口径

1. `golem_gm_bank_cluster.sv` 模块默认 CLIENTS=4，不代表 worker 有四条活动数据通路。worker top 中 client0=WCP、client1=DMA、client2=debug、client3 无效。生产 WCP/DMA 不应被当成四个满速 client。
2. 参考每个 client 每周期最多给一个 128 B line；8 banks 并不自动产生 8×128 B/cycle 的可达客户端吞吐。
3. 仲裁器每 bank 每周期选一个读或写，即使 SRAM wrapper 名称为 1R1W，也不能宣称同 bank 读写可同时通过 cluster。
4. 在 `GOLEM_SRAM_SIM` 中，读于接受沿 E0 采样，cluster 注册响应在 E1 后可见。这是相邻边沿关系；客户端时序逻辑何时消费响应还取决于它的实现，不能把全部 DMA 时长简化为“一周期”。
5. 容量、DMA burst、DMA outstanding、本地请求 pending、SRAM line 和端口发射速率是不同参数。这里不把参考 DMA 的 128 outstanding 等容量参数一起改入生产模型。

## 保留与补齐

| 项目 | 处理 | 理由 |
| --- | --- | --- |
| 每 worker 2 MiB GM | 固定保留 | 用户确定的容量合同；由当前统一布局检查容量 |
| 当前 array/SFU/NoC/HBM/C-buffer 参数 | 保留 | 本轮不扩大成全架构重建 |
| 4 KiB 本地大请求接口 | 可以保留 | API 请求可在控制器内部拆成 line；不要求 guest 改成逐 line 提交 |
| 有界异步队列与完成回调 | 保留 | 入队、实际服务、数据可见与完成必须区分 |
| 现有 FP16 舍入及 FP32 SFU 路径 | 保留 | 时序校准不应混入数值变更 |
| 地址到 bank 的映射 | 补齐 | 当前同地址模式与不同地址模式的争用没有区别 |
| 同 bank 读/写及跨 client 仲裁 | 补齐 | 独立全局读写端口不能代替实际共享 bank |
| client 到物理入口的映射 | 补齐 | RoCC/WCP/SFU/Control 是软件来源标签，不是独立物理端口 |
| 入口、bank、返回队列背压 | 一起约束 | 仅增加 bank 并行会无意提高总带宽与在途能力 |
| 读采样、写可见及部分写语义 | 明确定义 | 不能依赖同一 tick 中 callback 的偶然执行次序 |
| 未计时 backing-storage 访问 | 按路径分类审计 | host 初始化允许直接访问；生产数据搬运应计时或说明已在哪个资源计费 |

未计时访问仅是源码审计发现的待分类项，不表示全部进入当前正式用例。
例如 SFU 保留旧 primitive 直接访问路径，WCP 保留非 timedQkPayload 路径，GlobalMemory 也保留远端访问与控制分支；不能凭函数名数量推断遗漏流量。
正常 DMA READ_TO_GM 接收已有 `completeDmaReadToLocalMemory()` → `localWriteAsync()`，不能说全部 DMA 落地免费。

## 建议首版候选合同（尚未实现或验收）

采用 **8 banks × 256 KiB、128 B line** 作为参考结构扩容到 2 MiB 的候选组织：

```
offset    = address - worker_gm_base
line      = floor(offset / 128)
bank      = line % 8
bank_row  = floor(line / 8)
```

这保持 2 MiB，总容量变化不自动增加 bank 数或带宽。8 banks 是本次建议的显式建模参数，不是声称现有 SST 已有八个物理 bank。

为避免把参考结构变成“免费八路并行”，首版候选采用有界的两条 128 B/cycle 请求入口：

- 计算/控制入口：合并 RoCC、WCP、SFU、Control 及诊断 operand 标签；每周期最多一条 line，不能通过创建多个软件 job 增加发射宽度。
- DMA 入口：DMA 落地写与输出读取共用；每周期最多一条读或写 line。
- 两入口总计最多 **256 B/cycle，读写合计**；同 bank 每周期最多一条 line，冲突请求保留等待，不额外增加端口。
- 总吞吐受 `min(入口能力、目标 bank 能力、下游接受能力)` 约束。参考 WCP 优先级可作为对照；公平轮转可作为另一控制策略，均必须报告仲裁规则及等待，不静默替换。
- 请求先在有界队列接受，再按入口、bank 与返回容量发射；每个 4 KiB API 请求包含 32 条 line。跨 line 的小请求要占两条 line，部分写用 byte-enable，不覆盖未写字节。
- 读在实际 bank 接受时采样，写在接受沿提交；读完成增加明确的返回延迟。保留每入口返回次序/标签与有界返回容量，完成回调仅在整个请求的所有 line 完成后调用。

此候选是偏约束的参考兼容方案，不是已经确定的物理实现：当前生产配置是独立读 256 B/cycle、写 256 B/cycle，而候选是两条 128 B 入口共享的 256 B/cycle 合计；**两者带宽合同不同，结果差异必须归为模型校准，不能归为软件优化收益。**
如果后续有依据证明计算入口能每周期发两条 line，应单独给出该配置与资源解释；不能以“大请求接口已有 256 B/cycle 参数”证明双 line 入口存在。
同样，2 MiB 的容量不证明 1 GHz 下的宏延迟；首版沿用一周期读返回作为显式假设，可局部比较 1/2/4 周期的敏感性，不能称为物理时序验证。

bank 服务、入口服务与返回延迟必须按真实依赖安排，不能先收取整个请求的旧端口延迟，再把相同字节的 bank 服务额外累加一遍。
API pending=256 是当前软件请求队列界限，不是 256 个 SRAM 端口。line 拆分器及返回缓存也必须有明确容量，不能一次生成任意多在途 beat。

## 验证与停止条件

先做独立控制器/参考 RTL 子模块的行为对照，再进入局部真实 SST：

| 用例 | 必须验证的性质 |
| --- | --- |
| 连续同 bank 读取 | bank 接受间隔与返回延迟分别记录；无免费多端口 |
| 两入口同 bank 读/读、读/写、写/写 | 每周期最多一条通过，另一条确实等待 |
| 两入口不同 bank | 允许并行，但受每入口及总 256 B/cycle 限制 |
| 单计算入口 4 KiB 请求 | 拆成 32 条 line，不能借用空闲 DMA 入口加倍吞吐 |
| 地址间隔 1024 B | 连续请求映射同 bank；不能被当成均匀分散 |
| 非对齐及带 byte-enable 的写 | 正确跨 line、边界检查、保留未覆盖字节 |
| 有依赖的写后读 | 写完成后读必见新值；读采样后写不倒改已采样数据 |
| 请求队列或返回队列满 | 请求停发/重试，不丢失、超量发射或提前完成 |
| generation 取消/复用 | 取消事务不污染新作业，不错误释放旧缓冲区 |
| 同一批到达请求变更提交顺序 | 同 tick 仲裁按合同决定，不能由 C++ 调用先后抢占所有 bank |

报告拆开 request queue、client arbitration、bank conflict、return/backpressure 等等待。
统计每入口 accepted beats、每 bank grant、payload/line bytes、队列峰值与真实临界路径。
完整读/写数据与现有数值参考保持一致。

通过微测试后，先测 Wo 与正式 FFN 的单 worker、16-worker/MPI8；在同一新模型下重新比较已有四槽预取/两阵列组方案，再挑选完整两层验收配置。
不得用旧模型 baseline 对新模型 candidate 计算软件加速比。
若模型假设变化会改变方案排序或产生明显 cycle 差异，先报告该敏感性，再增加融合复杂度。

本轮只完成上述审查与候选合同；生产实现、RTL 仿真对照、局部 SST 和新端到端 cycle 均尚未执行。

## 证据入口

- 当前行为：[GM_TIMING_CONTRACT.md](GM_TIMING_CONTRACT.md)。
- 当前本地队列、发射、读写提交：`src/sst/elements/golem/globalmemory/globalmemory.cc` 的 `localReadAsync`、`localWriteAsync`、`tryIssueLocalAccesses`、`handleLocalAccessEvent`。
- 当前 timed DMA 落地：同文件 `completeDmaReadToLocalMemory`、`issueNextDmaLandingChunk`；其他 backing-storage 分支需要按实际调用路径审计。
- 来源标签定义：`src/sst/elements/golem/globalmemory/globalmemory.h` 的 `LocalMemoryClient`。
- 正式历史参数：`results/lm_parallel_20261015/stack_n4/decoder_stack_provenance.json`，历史总延迟 6,231,011.75 cycles 继续限定为旧模型结果。
- 参考范围：`/data/jjgong/RISC-V-CIM-Manycore-SST/src/sst/elements/golem/tests/architecture/HDL/README.md`。
- 参考 bank/握手：该目录 `rtl/golem_gm_bank_cluster.sv`、`rtl/golem_sram_macros.sv`。
- 参考活动 client：该目录 `rtl/golem_worker_core_top.sv` 的 `gm_req_valid[0..3]` 连接。
- 参考 SST：`src/sst/elements/golem/globalmemory/globalmemory.cc` 的直接 storage 访问与 `workercmdproc/workercmdproc.h::loadTilePayload`。
