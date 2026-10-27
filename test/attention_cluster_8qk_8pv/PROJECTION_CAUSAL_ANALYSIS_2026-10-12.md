# Projection 访存等待与阵列推进空洞：逐 tile 因果分析

2026-10-12。正式 S1024/S2048、Hq=32/Hkv=8、D64、FP16、causal，64 arrays/worker、array_groups=1、单 operand bank，IW 读取预取。使用 clean_startup guest。端到端严格使用 `hbmdma_to_O_ack_v1`：第一笔工作负载 HBM DMA 成功发送到 NoC → 最后一笔 O 写回 ACK；准备与最终 ACK 后的工作不计入。

本轮完成 opt-in 因果追踪、两个规模的对照重跑、逐请求/逐 tile 关联、空洞闭合统计，以及两个软件单变量实验。没有增加 GM/阵列端口、带宽、阵列、operand bank、DMA slots 或 payload snapshot；默认数据流参数没有修改。机器报告位于 `results/projection_causal_20261012/summary.json`。

## 1. 测量可信度与产物

`GOLEM_PROJECTION_CAUSAL_TRACE=1` 增加 `[PROJECTION_CAUSAL]`，保留完整 `(core, job, block, Q/K/V kind, head, dim, row, hidden tile)`。读取 submit/done、snapshot consume、weight decode、matrix/scatter submit/done、MAC submit/done，以及 weight DMA submit/done 均使用 SST core tick。事件发出不改变原请求 tag、发射顺序、模拟延迟或资源配置。

本地 GM 读按 `(core, tag)` 精确关联，并交叉验证 submit tick、地址、字节数和 callback/visible tick；矩阵和 scatter 按资源/tag 与 callback/visible tick 校验；MAC tile 按提交/完成窗口与物理计算区间一一配对。权重 DMA 按 worker/地址及提交顺序关联，校验数量与时间顺序；该路径每个 cache slot 不允许重复在途请求。DMA transport 包含 NoC、内存与响应往返，不能称为纯 HBM 服务。

追踪前后主口径精确一致：S1024=473,902.75，S2048=1,238,814.75 cycles。S1024 对照关联 24,576 tiles、99,584 笔本地读、49,152 个矩阵/scatter 命令；S2048 数量加倍。所有正式实验数值校验、HBM 输出逐字节校验通过。85 项 baseline Python 测试通过，覆盖资源重叠不重复计入、阵列数量加权、丢失 tag/错误 callback 的拒绝。

逐 tile 记录：各 case 下 `projection_tile_timeline.jsonl`；关联汇总 `projection_causal_analysis.json`；空洞分类 `projection_holes.json`。MPI 合并器将事件保存到独立 `.causal.jsonl`，物理资源在 `.resources.jsonl`。

S2048 对照汇总的来源检查发现仿真期间离线空洞分析器版本变化；模拟器、guest、运行、日志合并和主口径报告代码/二进制均未变。原版本保留于 `source_snapshot`，新版本与差异记录在 `analysis_revision` / `baseline_provenance.json.postprocessing_revision`。六个正式用例的模型与 guest 哈希一致。两个 S1024 实验最初选择了超出 Open MPI 物理 core 编号范围的宿主绑核，未启动仿真；失败目录保留，最终结果采用 `_v2`（有效宿主 core 0–7）。

## 2. 纠正“GM 排队很大，所以关键路径读端口堵塞”的判断

对照两个规模，所有 worker 的前台 input/weight/fusion 读请求 GM 排队均为 **0**。累计排队来自后台预取；S2048 慢 worker core 12：后台 input 排队累计 154,132，后台 weight 49,921，共 204,053 cycles。这些是请求等待累计值，包含自身批量请求在单读端口上的排队和已被其它操作覆盖的等待，不能加到 E2E，也不能直接当作可节省时间。

前台本地读仍有暴露的服务与软件串行开销：8 KiB 权重拆成两笔 4 KiB 请求，每笔服务 17 cycles，前台一次只有一笔在途，阶段中每笔还要下一 tick 推进。问题不是前台在读端口 FIFO 中等很久，而是某些权重没能提前进入本地 snapshot，只好等 HBM 到达后再进行前台读。

core 12 的前台 weight 读为 2,438 笔（1,219 个 tile 未命中本地 weight snapshot）；最快 core 7 为 1,420 笔（710 个 tile）。同样的总工作量、端口和服务时延，慢组更多次退回前台。这与已有 `local_weight_prefetches/hits` 一致。

## 3. MAC 空洞的精确闭合

S2048 的 Projection 全局阶段为 684,640 cycles，最后完成的 worker 为 core 12。其 job 窗口为 77,907 → 762,430，共 684,523 cycles：

|互斥时间区间|cycles|
|---|---:|
|job 开始 → 第一段物理 MAC 开始|35,576|
|物理 MAC 服务区间并集|202,752|
|相邻 MAC 服务之间的间隙|445,868|
|最后物理 MAC 完成 → job 结束|327|
|合计|684,523|

MAC 阵列利用率 29.62%。该指标对 32/64 阵列区间按真实 array count 加权；全组路径的 compute union 与等效 MAC 服务相同。

445,868 cycles 的间隙按控制器阶段互斥拆分：

|间隙中的控制器阶段|cycles|含义|
|---|---:|---|
|program_weight_ready|125,911|矩阵命令、39-cycle 物理 broadcast 和 tick 推进|
|program_input|107,485|输入命令、33-cycle 物理 scatter 和 tick 推进|
|load_weight|97,733|消费所需权重尚未在 GM cache 可用；包含等未来 DMA 到达|
|local_read|52,128|前台本地读服务/逐 chunk 推进，包含输入与第二块 norm 读取|
|norm_wait|27,952|中途第二个输入块的 SFU RMSNorm|
|program_weight|16,232|等待本地 weight snapshot / 状态推进|
|其余阶段|18,427|MAC completion/issue、输出读取、norm 准备与写入等|
|合计|445,868|只计算相邻物理 MAC 间隙，首尾另列|

间隙中实际矩阵服务=119,769，scatter 服务=101,343，二者互不重叠；剩余=224,756 cycles。不能把阶段周期与嵌套的物理服务再相加。MAC 间隙中位数 106、P95 273 cycles。最大间隙约 35,474 cycles，发生在输入块切换/RMSNorm；该间隙不能笼统归为权重访存。

S1024 最慢 core 18：MAC 服务 101,376，MAC 间隙 180,017，中位间隙 81 cycles。单 context 下，数据准备好的稳态推进为 **66-cycle MAC + 81-cycle 间隙 = 147-cycle/tile**；间隙中的 39-cycle matrix 与 33-cycle scatter 是真实服务，其余约 9 cycles 是命令、回调与状态推进。阵列没有请求 FIFO 堵塞：matrix/scatter queue=0，命令到物理资源提交各 1 cycle。单 context 上不能在 MAC 尚使用 operands 时重编程整组数组。

权重解码是宿主 C++ 的循环：decode start/done 在同一个 SST tick，模型中为 0 cycles。因此本轮不做“解码 cycle 优化”实验；它只能改善仿真墙钟，不能降低这套模型的架构 E2E。

[实际逐 tile 时序图：core 12 与 core 7，第一个 head 的 tiles 0–15](../../results/projection_causal_20261012/tile_timeline.svg)。图是原始物理服务区间，空白表示没有该类服务；weight ready 点来自对应 operand read 的完成事件。

## 4. S2048 长尾的来源

core 12 job 比最快 core 7 多 63,154 cycles；全局 job **完成时间**跨度为 62,222 cycles（两者开始时间不同，不能混用）。二者 MAC/矩阵/scatter 工作量和物理服务完全相同；core 12 的 MAC 间隙比 core 7 多 63,646：load_weight 多 37,486、local_read 多 18,324、program_weight 多 7,611，其余约 225 cycles。首尾时间差抵消了一小部分间隙差。

权重 DMA 逐请求对照：

|S2048 worker|weight transport 平均|前台 weight 本地读请求数|前台 GM queue|
|---|---:|---:|---:|
|core 12（最后完成）|164.61 cycles|2,438|0|
|core 7（最早完成）|141.06 cycles|1,420|0|

core 12 的 DMA 到达 → GM landing 完成/回调平均另需 34.01 cycles，提交 → 完整回调平均 **198.62 cycles**。代码中的 `weightPrefetchBusy_` 允许一次只有一个**未来权重** DMA 在途；另有需要时的前台 demand DMA，不能说所有权重请求全局最多只有一个。在理想稳态 tile 147-cycle 节拍面前，单未来请求的 RTT/landing 很难持续提供充分的前瞻库存。把 lookahead 由 4 改成 8 并没有解除这一并发限制。

这一证据定位到供给与消费的节拍失配，以及依赖未来 DMA/本地 snapshot 的软件推进。它尚未隔离 HBM 控制器、DRAM service 和 NoC 各自对 transport RTT 的贡献；当前选择的 trace domains 不包含 HBM 的内部资源。在此层面不能直接判定为 HBM 带宽饱和。

## 5. 两个正交软件实验

各用例都与原 clean 对照使用同一输入；总本地读服务与工作量不减少。只改变以下一个参数，硬件保持相同。

|配置|S1024 E2E|相对对照节省|S2048 E2E|相对对照节省|
|---|---:|---:|---:|---:|
|对照：input budget=0，weight lookahead=4|473,902.75|—|1,238,814.75|—|
|input budget=1|478,884.25|−4,981.5（−1.05%）|1,233,475.75|5,339（0.43%）|
|weight lookahead=8|472,533.875|1,368.875（0.29%）|1,225,694.75|13,120（1.06%）|

`input budget=1` 限制现有未来 input snapshot 的本地读在途请求数，改发送顺序以减少预取 burst；对 S1024 造成回退。`lookahead=8` 使用原有 weight GM cache 槽位增加前瞻，不增加容量；两规模小幅受益。两个实验均保持单未来 weight DMA，与矩阵/scatter/MAC 的单 context 串行边界，因此收益有限。S2048 的最后完成 worker 均切换到 core 18，不能只观察原来 core 12 的改善来宣称全局提升。

本轮不改默认参数：input budget=1 不稳定；lookahead=8 可作为已验证的可选小幅优化，但不能当作主要问题已解决。对矩阵/scatter 与 MAC 的重叠，既有两组 32-array 实验已有数据且没优于全组读取流水；本轮不重复该试验，也不添加第二 operand bank。

## 6. 后续应针对明确的软件约束

优先实验应是：**在既有 GM cache 槽位和硬件 DMA credits 内，允许两笔有界的未来 weight DMA 同时在途**。为每个 cache slot 维护未完成目标 tile，禁止重复请求、覆盖当前/未消费数据；完整 landing 后才能标记 cache ready；结束与 head/block 切换必须等该代请求安全退出。一次只改并发度 1→2，lookahead 先固定 4，其它配置保持对照。评价 E2E、weight arrival 暴露阶段、前台 fallback 数、GM write landing 排队以及快慢 worker 长尾；不要只看 cache hit。

随后再评估：在同一输入 block 已有 GM 空间内，为下一 norm 子块提前发出必要读/准备，验证能否消除约 35k 的块切换 MAC 空洞；不能让当前仍被消费的 normalized input 覆盖。最后对 81-cycle 稳态间隙里的约 9-cycle 控制开销做保持时序模型含义的提交/回调推进实验。39+33 cycles 的真实物理传输，要靠现有阵列分组调度评估可重叠性；不能直接删延迟。

这些是下一轮实验方案，尚未在本轮实现。现阶段已经确定应先处理权重供给的软件在途限制和分块生产者边界，不能继续把后台累计 GM 排队当作端到端瓶颈。

## 7. 复现

```bash
# 编译修改后的 projectionJob.h 并安装 worktree 本地 libgolem.so 后：
python3 baseline/attention_cluster_8qk_8pv/run_projection_causal.py \
  --artifact-root results/my_causal_baseline_s2048 \
  --reference results/startup_release_20261012/clean_formal_s2048 \
  --stage formal --sequence 2048 --variant baseline --cpu-set 0-3
# variant: baseline / budget1 / depth8；另一个规模用 --sequence 1024 及其对应 reference。
python3 baseline/attention_cluster_8qk_8pv/analyze_projection_causal.py results/my_causal_baseline_s2048
python3 baseline/attention_cluster_8qk_8pv/analyze_projection_holes.py \
  results/my_causal_baseline_s2048 --output results/my_holes.json
python3 baseline/attention_cluster_8qk_8pv/summarize_projection_causal.py results/projection_causal_20261012
# 可选图：需要 matplotlib，不影响核心分析器和仿真。
python3 baseline/attention_cluster_8qk_8pv/plot_projection_causal.py \
  results/projection_causal_20261012/baseline_s2048 --output results/my_timeline.svg
```
