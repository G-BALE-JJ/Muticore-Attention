# FFN GM 数据流实验（2026-10-16）

本轮以校准后的 banked GM 为基线，只改变软件布局与循环顺序。每 worker
仍为 2 MiB GM、64 arrays、每 array 一个 operand context、1 MiB C-buffer；
GM 入口、bank 仲裁、响应队列、SFU、NoC/HBM 参数均未改变。
`results/ffn_gm_flow_20261016/experiment_build.json` 保存源码与安装库哈希，
并与 GM 验收版本逐项核对硬件模型文件。

## 1. 流量归因与关键路径

正式局部问题：每 worker 64 行，H=2048、I=8192，完整 gate/up、SiLU/Multiply、
down、第二残差；权重数值稀疏，但所有 dense tiles 均实际执行。
保留原来的 16-tile panel、4-slot prefetch、两组 32 arrays 调度。

每 worker 每次 FFN 的主要 compute GM 流量为：

| 数据流 | 原路径流量 | 可以消除的部分 |
|---|---:|---|
| gate/up/down 权重读取 | 96 MiB | 本轮保留 |
| 重复 array 输入读取 | 96 MiB | 单独实验 down 输入复用 |
| 输入打包读写 | 0.5 MiB | 本轮保留 |
| gate/up 的 tile→row 转换读写 | 4 MiB | layout1 消除 |
| gated 的 row→K 转换读写 | 2 MiB | layout1 消除 |
| down 输出的 tile→row 转换读写 | 0.5 MiB | layout2 进一步消除 |

这些是流量，不能将异步请求的整段时延相加当成关键路径。原路径实测
compute GM 入口为 1,679,360 beats，128 B/beat，即至少 1,679,360 cycles
的入口工作；array buffer 实际传输工作为 1,375,968 cycles；array compute
服务折合 823,296 cycles（所有 array 理想满载，含每 tile 的 3-cycle 延迟）。
实测局部总延迟 2,375,059.625 cycles。
这几个下界互相重叠，应该取 max，不能相加，也不能把间隙全归为 GM 冲突。

对照中 12,288 次权重预取全部命中，前台等待合计 6,706 cycles，panel
slot 等待为 0；GEMM/SFU 已重叠 444,236 cycles。这里的“命中”包含此前台
等待完成的情况，不能理解为所有权重从不等待。`preparePanel`、gated→down
和 down 最终 repack 则都是消费前必须完成的真实 GM 往返，因此优先消除
它们。layout2 后预取等待为 6,959 cycles，GEMM/SFU 重叠为 450,684 cycles；
并没有靠降低权重或 SFU 工作量取得收益。计数证据为 local_path_counters.json。

`analyze_ffn_gm_flow.py` 从真实 local read/write 区间按 worker、epoch、
client、地址区域统计请求、字节与跨行 beats，并与 DECODER_OP 计数核对。
`verify_gm_banked.py` 进一步核对物理 bank grants、队列排空、容量、
响应边界及是否存在未计时张量访问。

## 2. 去掉内部布局转换

`GOLEM_DECODER_FFN_LAYOUT=0|1|2` 默认 0，保留旧路径作为同 binary 对照。

- **1**：gate/up/gated 保持 tile-major，逐元素 SFU 直接消费同一物理顺序；
  down 直接从 gated panel 读 K tiles。省去两次完整布局往返。
- **2**：在 1 的基础上，down gather 的已有 tile 快照直接分行写入最终
  row-major GM panel，省去最终输出布局往返。

模式 2 的 strided 写回沿用现有缓冲与最多 8 个写请求窗口，每 job tick
最多提交一次，所有真实 GM ACK 排空后才发布 panel ready。没有新的张量
缓存、端口、bank、容量或未计时数据通路。支持拒绝重试和延迟/乱序 ACK。
两层/后续算子看到的外部 HBM 布局不变。

每个正式 FFN job，模式 1 减少 49,152 个 compute GM beats；模式 2 减少
53,248 beats（6.5 MiB）。HBM、矩阵编程、输入 scatter、MAC、SFU token
和部分和工作均保持不变。

## 3. 输入复用实验与淘汰理由

`GOLEM_DECODER_DOWN_INPUT_REUSE=1` 将 down 改成 K 外层、N 内层，输入
保留在现有 array input register，后续 N 只改权重。不增加 operand context。
gate/up 仍用两组 arrays；down 使用全部 64 arrays。

代价是在每个 K tile 后将所有 N 的部分和存回现有 C-buffer，下次 K 再恢复。
正式尺寸仍只占原来的 256 KiB 部分和区，但读写量从各 1,835,008 B 增加
到各 33,292,288 B，恢复次数从 224 增至 4,064。每个 K 后原本已有 FP16
舍入，因此 spill 不引入新的数值舍入点；输出按 FP16 比特比较验证。

每 job 虽然减少 31 MiB GM 输入读取，但多出 **60 MiB C-buffer 读写**。
array output scatter/gather 服务从 24,288 增至 277,728 cycles。
局部实测比旧路径慢 2.46%，比 layout1 慢约 4.73%，因此不进入完整两层候选。
这说明仅看 GM bytes 或仅追求输入驻留不足以判断收益。

## 4. 局部筛选和完整验收

所有正式竞争使用相同源码、guest、安装库、输入哈希和硬件环境，仅允许
软件实验开关及文件路径不同。首次 small 运行期间 runner 有编辑，最终
provenance 比较拒绝了该轮；它只保留为诊断，采用冻结后的 small_verified。

| 局部方案 | 小尺寸 H256/I1024 cycles | 正式单 worker cycles | 相对正式对照 |
|---|---:|---:|---:|
| 原路径 | 124,654.625 | 2,375,059.625 | — |
| layout1 | 118,549.625 | 2,323,501.625 | −2.17% |
| layout2 | 117,783.625 | 2,318,719.625 | −2.37% |
| layout1 + 输入复用 | 121,129.125 | 2,433,427.625 | +2.46% |

小尺寸完整两层 layout2 为 661,217.625 cycles，数值、epoch/reset、同步、
资源审计和 banked GM 核对均通过。该数值用于集成验证，正式收益由下方
同构建的正式两层对照确定。

正式 16-worker / MPI8 对照 **2,388,853.625 cycles**，layout2
**2,331,514.625 cycles**，减少 **57,339 cycles（2.40%）**。
16 个 FFN job 共减少 **851,968 compute GM beats**；实际 array 传输、
C-buffer partial 工作、SFU token、DMA beats 均不变，资源审计通过。
正式两层 S1024/H2048/I8192、Embedding、final RMSNorm、最后 token 的
128256 词表 LM head 均通过，同构建对照如下。计时从首个 token-ID DMA
发出，到最后 logits DMA ACK；不含 host argmax。

| 可加和阶段 | 原路径 cycles | layout2 cycles | 节省 cycles |
|---|---:|---:|---:|
| Embedding | 27,730 | 27,730 | 0 |
| Embedding handoff | 55,903 | 55,903 | 0 |
| 第一层 RMSNorm/QKV/Attention | 546,598.875 | 546,598.875 | 0 |
| 第一层 Wo/残差/Norm2/FFN | 2,776,402.5 | 2,711,558.5 | 64,844 |
| 两层 handoff | 52,523.625 | 52,736.625 | −213 |
| 第二层 RMSNorm/QKV/Attention | 536,731.875 | 536,900.875 | −169 |
| 第二层 Wo/残差/Norm2/FFN | 2,765,787.5 | 2,711,316.5 | 54,471 |
| final norm handoff | 15,525.625 | 15,530.625 | −5 |
| final RMSNorm | 54,307.75 | 54,550.75 | −243 |
| LM handoff | 9,855.25 | 9,863.25 | −8 |
| LM head | 641,901.75 | 640,290.75 | 1,611 |
| **总计** | **7,483,267.75** | **7,362,979.75** | **120,288（1.6074%）** |

后续阶段开始时刻改变，NoC/HBM 状态和轮询对齐可导致少量阶段波动；
这些不是额外的 LM/Attention 优化。实测 SST wall 分别为 1230.612 s、
1240.496 s；宿主同时进行了清理和两组仿真，不以此宣称 wall-time 加速。

最终核验：

- 两层输出、共享 Wo/H1/Norm2、Embedding、final norm、全部 logits 的
  FP16 比特相同；两组下一 token 均为 99037。
- 每组 416 个 decoder jobs、epoch/reset、scratch 生命周期与资源审计通过。
- compute GM 从 71,133,412 降至 69,429,476 beats，恰好少
  32×53,248=1,703,936 beats，即 **208 MiB**；每 worker 少 106,496 beats。
- 每个 worker 的 DMA beats、array 物理工作、C-buffer partial 工作和
  SFU accepted tokens 相同，所有阶段的张量 DMA 请求数/字节相同。
- 全局 DMA beats 从 36,016,152 降至 36,013,161。差值 **2,991** 逐项追到
  root/managers 的控制区 `0x3800`：少了同样数量的 8 B 轮询落地写入，
  payload 共少 23,928 B。不能把它误报为张量 DMA 优化，也不能要求总
  控制轮询数在运行时间变短后保持相同。
- GM bank grants 与逻辑跨行 beats 一致，队列排空，张量未计时访问为 **0 B**。
  bank conflict cycles 从 110,512 增至 123,709，未放宽冲突模型。

验收工具 `compare_ffn_gm_stack.py` 对来源、库、环境、输入、输出、各资源
和控制差值执行断言。原始证据在 `results/ffn_gm_flow_20261016/`，轻量
归档在 [progress_results/2026-10-16/ffn_gm_flow](progress_results/2026-10-16/ffn_gm_flow)。

第一次正式两层对照遇到宿主磁盘写满，最终 MPI 统计日志被截断；
第一次候选则在输入准备阶段退出。这两次均不作为验收结果。
失败日志已保留压缩归档，随后以相同输入、源码/库哈希和环境重跑。
HBM backing 与大型日志均放在 tmpfs，模拟硬件参数不变。
磁盘审计确认旧构建目录复制了约 770 GiB 实验副本；经用户授权已清理。

已审计的部分局部资源轨迹以 `.resources.jsonl.gz` 归档；需要再次调用
读取明文轨迹的验证器时先 `gzip -dk <该轨迹路径>`，避免误认为轨迹缺失。

新增生产控制器测试覆盖三种布局、输入复用开关、单/双 array 组、预取
开关、尾 panel、1/33/65/257 行与空 worker，以及拒绝重试、乱序回调。
host 147 项和生产侧 155 项测试全部通过（共 302 项）。

## 复现

```bash
python3 baseline/attention_cluster_8qk_8pv/run_decoder_micro.py \
  --artifact-root results/ffn_gm_flow_20261016/formal16 \
  --shape formal --workers 16 --mpi-ranks 8 --cpu-set 8-15 \
  --with-down --with-residual --prefetch-slots 4 --weight-trace \
  --variants groups2residual16q4 layout2 \
  --reference-variant groups2residual16q4

python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/ffn_gm_flow_20261016/stack_layout2 \
  --shape formal --embedding --mpi-ranks 8 --cpu-set 8-15 \
  --gm-model banked --gemm-prefetch --prefetch-slots 4 --weight-trace \
  --ffn-down-fusion --down-residual-fusion --ffn-panel-tiles 16 \
  --ffn-array-groups 2 --ffn-layout 2 --lm-n-parallel 4
```

正式对照将 `--ffn-layout 2` 改为 0，其余相同。可用 `--host-hbm-root`
将 HBM host backing files 放到充足的 tmpfs，减少宿主磁盘压力；它不改变
模拟 HBM 容量、带宽或时序。wall time 是单次宿主测量，不作为 cycle 收益。
