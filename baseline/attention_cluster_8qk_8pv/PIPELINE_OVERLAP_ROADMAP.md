# 固定资源并行与流水重叠路线

> **架构纠正（2026-10-10）：** 本文的历史投影/E2E 测量或重叠估计使用了 manager 本地 arrays，不符合「manager 只控制、16 个 worker 拥有阵列」的目标拓扑，不能作为该架构的性能证据。新的 worker 执行基准见 [WORKER_PROJECTION_ARCHITECTURE_FIX.md](WORKER_PROJECTION_ARCHITECTURE_FIX.md)。

本文件是源码与既有正式日志分析，尚未实现或实测以下新增调度。
基准为 `results/attention_causal_balance/final/balanced`：S1024 / S2048
E2E 为 1,066,757 / 2,373,527 cycles，projection producer 为
910,789 / 1,820,535，Attention 为 154,391 / 550,110。
不同 stage spans 有重叠，不可直接相加。范围为 RMSNorm → QKV → RoPE →
causal Attention → O HBM，不包括 Wo/residual/MLP。

## 1. 投影双 operand bank 的权重预编程

当前已预取权重到 GM，并预取 inactive-bank 输入，但权重的 GM read 和
array programming 仍在前台串行阶段。manager 0 / 每个 256 行块，两项
分别约 55,296 和 64,512 cycles。这是暴露阶段预算，不是可保证隐藏量。

保留同一 hidden pair 对四个 64 行组的权重复用。仅在**最后一个行组**
使用旧 pair 时，提前准备下一 pair：

1. 最后行组的 odd tile 在 bank 1 计算时，bank 0 的旧 even weight 已无消费者；
   利用已有权重 scratch 和命令 payload 生命周期，读取并编程下一 even weight。
2. 旧 odd 计算完成后 bank 1 才可释放；下一 even 在 bank 0 计算时，
   读取并编程下一 odd weight。
3. 下一输入 scatter 必须排在对应 bank 的权重 programming 完成之后；
   consumer 以明确 tile/bank/generation 匹配完成状态。

不能在前几个行组提前覆盖 bank：后续行组仍要复用这对旧权重。不能并发启动
两组计算：两个 operand banks 仍共享一份 output accumulator。output drain
与下一 partial restore 的原有依赖保持不变。首 pair 仍需启动成本。

同一个 array-buffer port 串行处理 weight broadcast、input scatter、output
drain/restore。预编程尽量落在 compute 内，不能把长 broadcast 排到紧迫 output
之前。初版只用一个后台权重任务，不增加 payload snapshot、bank 或 credits；
须审计其 payload 与已有 `weightRaw_`/命令副本的峰值空间。

另一个低风险子实验是 head 边界预取：最后 pair 时旧 head 的 cache slots 0/1
已无消费者，可预取下一个 head 的首 pair 到这些已有 slots。当前
`pumpWeightPrefetch()` 只前瞻同一 head。此项与预编程分别测量，避免混合归因。

## 2. RMSNorm batch 内 DMA / SFU / GM 写流水

当前 4 行 batch 的 SFU 工作为 1,747 cycles；每块 SFU 总工作 111,808，
实际 Norm span 约 136,100。约 24,292 是非 SFU 的 span 差值，包含必须支付的
传输与控制，不能都认定为空闲或可消除。

在现有 64 KiB Norm raw scratch 内交替使用两个 16 KiB 输入槽：当前 batch
计算时预取下一 batch 原始 X。先只实现 raw DMA 前瞻；进一步尝试在 SFU wait
内安排 raw/gamma GM read，必须保持既有操作数/结果空间预算和单个 SFU vector
任务。当前 batch packed 写回期间可推进下一 raw DMA，但对应 slot 未释放时
不覆盖。最终 ready 以所有 packed write 回调完成为准。

现有 DMA/dispatch 与 raw/gamma read 前台统计合计约 18,149 cycles/块，是
候选窗口预算，非收益承诺。gamma 不通过新增隐式寄存器缓存绕过原 GM 计时。
单块 S1024 也能受益，故优先于仅适用多块的跨块方案。

## 3. RMSNorm 下一块 128 行与本块投影重叠

沿用 [NORM_PROJECTION_OVERLAP.md](NORM_PROJECTION_OVERLAP.md) 的两个既有 GM
空闲槽和 per-row-group 地址映射，保留 256 行 weight-major reuse。
S2048 无新增争用时乐观可隐藏约 68,025 cycles，约 E2E 2.87%；S1024
没有下一块。这一估计须在 batch 内流水优化后重新计算，不能把两者收益相加。

Norm producer 使用独立控制状态，复用已有操作数与结果空间，不能调用会覆盖
前台 `phase_`/`pending_`/local read 上下文的现有串行函数。GM/DMA 相互争用，
只在不阻塞即将使用的 projection operand 时提交后台请求。

## 4. 共享 array-buffer 端口的提交时机

每块 output read 约 107,520、restore 含派发约 72,576 cycles。当前已经
实现 ahead restore；继续优化必须区分服务、排队、WCP 派发和依赖等待。

为请求记录 enqueue/start/end、traffic class、bank、row/tile，并记录每次
compute launch 的前置条件。仅调整 producer 提交时机，保留原 FIFO/端口：
避开即将 output drain 的窗口，提前提交确实落在 compute slack 内的 scatter
或 weight；output 全部 drain 后才能 restore，restore 完成后才能 launch。
不新增硬件 priority queue，也不以减少某个 wait 代替 net E2E 收益。

## 5. 投影与 Attention 按 KV head 交接

现在 guest 在 `finish_projection()` 等待全部四个 manager 完成全部 QKV，
随后提交所有 Attention head。`cpu_builder.py` 给各 core 实例化自己的 array，
manager 的 projection arrays 与 worker 的 Attention arrays 可独立计算；
共享 HBM/NoC/GM 传输仍会竞争，manager 调度上下文也须审计。

保持块优先、256 行 Norm 复用，但把块内顺序从全部 Q、全部 K、全部 V 改为
每个 KV head 的 K、V 及对应 GQA Q heads。每个 head 内的 hidden-pair 次序与
FP16 partial 边界不变，不应增加权重 programming 或 Norm 次数。

在**最后一个输入块**，首个 KV group 在所有 manager 的对应 Q/K/V panels
完成 HBM 写回后，可由已有 descriptor 启动该 head 的 Attention，同时 manager
计算其余 KV groups 的投影。需要逐 head 完成发布，而非只看 array compute
完成；原全局 completion 保留作最终 fence。同步标记限于已有元数据空间，
不改变 tensor descriptor/message ABI 或新增资源队列。

S2048 第一块结束还不能直接运行完整 head：后续 query bands 的 K/V 未生产，
且现有 descriptor 要求完整 kv_length。细到 causal key-window 的增量消费需要
额外可见性协议，列为后续研究，不作为初版前提。最后块的 head 交接则可保持
现有完整 head descriptor，并继续使用原有早晚 head 配对。

此候选潜力大于消除 Attention 尾部的小等待，但尚无逐 head readiness 时间线，
不提供预测收益。须确认早晚配对下先发出的 head 没有改变全程负载均衡，
并测试 projection DMA 被 Attention 抢占后是否反而延迟关键路径。

## 6. Attention 内已有 KV buffers 的预取与 GQA 复用

现有双 KV buffers、QK input pipeline、PV prefetch、V broadcast 已开启。
下一步只检查仍暴露的 buffer-not-ready/NoC backpressure/SFU starvation，以及
同一 K panel 在 GQA Q heads 间是否重复加载且可在既有 buffer 生命周期内复用。
真实 Q/K payload 为 S1024 10/20 MiB、S2048 36/72 MiB；V 为 1/2 MiB。
这些统计说明需追踪重复来源，不足以证明已有可用容量或可消除流量。

保持 exp 工作、16 lanes、II=1 的同一算法下，S2048 exp 局部工作量下界
540,672 已接近 Attention span 550,110，单独压缩此 stage 的理想剩余差值
仅 9,438 cycles。优先跨阶段重叠；不增加 SFU lanes、不改 softmax 数值算法。

## 实施顺序与验收

建议先做单一后台权重预编程（包含 bank 生命周期与端口时间线），再做 Norm
raw DMA batch 前瞻，第三步做跨块 prefix；随后用新时间线决定端口提交调整。
按 head 的 Projection→Attention 交接作为独立实验，不能直接删除全局 barrier。

每个实验默认关闭，单独同库对照后再测试组合。资源配置、输入 hashes、输出
形状相同；NumPy 检查及全部保留 Q/K/V/O byte-exact；纯调度的 Norm/weight/
program/input/partial payload 总量保持不变，去重实验则解释每项减少来源。
记录峰值 scratch/snapshot/在途数，测试 S1024/S2048、多于两块、D128/RoPE
回退、低带宽/小 queue，比较全程 E2E、stage span、compute idle 和传输等待。
所有不满足布局或生命周期前提的形状沿用原执行计划。
