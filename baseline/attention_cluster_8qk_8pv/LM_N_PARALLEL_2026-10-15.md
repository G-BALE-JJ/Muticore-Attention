# LM head N 维并行：局部筛选与完整验收（2026-10-15）

## 软件映射与硬件边界

针对最后 token 的 `M=1` Linear，新增显式开关 `--lm-n-parallel 1|2|4|8`，默认仍为 1。
每个 worker 保留原有的 `tile % 16` 列块归属；每个 8192-column chunk 最多有 8 个归属于它的 N tile。
候选把其中 P 个 tile 映射到现有 array 0..P-1，按 K 轮次推进，每个 array 计算独立的 64 列。
各输出仍按 K=0..31 的原顺序累加，FP16 存储与舍入边界、内部计算精度不变。

每个 K 轮次依次通过原端口编程 P 份不同矩阵，读取一次 128-byte 输入并通过原 scatter 接口发往 P 个 array，
随后 group launch；全部 array 回调完成才进入下一 K 轮次。最终 gather 后按原 strided 地址分别写回，
等待每个输出 DMA ACK，最后一个词表 chunk 才写 completion flag。
这版保留组内同步；没有把编程和本组 MAC 进一步交错。

每份不同的 8 KiB 权重都执行真实 HBM DMA、GM read 和单阵列 matrix programming，
没有把不同权重当作广播省略。复用原有 4-slot 权重预取和 GM input/output arena；
选择的 P=4 使用至多 512-byte 输入 scatter/输出 gather，没有增加 SRAM 或 bank。
每 worker 仍为 64 arrays、单 operand context、GM 2 MiB、C-buffer 1 MiB；端口、队列、NoC、HBM 带宽与时序均不变。
原 FFN 两个 32-array 子组调度保持开启且不受影响。仅 `DecoderOpJob` 和 full runner 与上一验收版本有源文件差异；
另有新的局部 guest、runner、审计与测试文件。13 个专门的硬件文件及 build 副本也通过逐文件 SHA256 检查。

## 测试流程与局部筛选

直接调用生产 `run_lm_head` 和 `DecoderOpJob`。所有候选使用相同输入、guest、动态库、源文件、硬件环境和实际执行 placement。
对每个 job 核对完整权重/输出地址覆盖、DMA ACK 生命周期、唯一 weight-tile 消费、真实 array buffer 端口计数、
GM 请求差值及独立 CPU FP16 oracle。行为测试另外覆盖独立矩阵、K 舍入顺序、空 owner、尾块、
2/4/8 个预取槽、预取开关、端口重试、错序 MAC 回调与延迟 DMA ACK。

| 局部形状 | N=1 cycle | N=2 cycle | N=4 cycle | N=8 cycle |
| --- | ---: | ---: | ---: | ---: |
| H=256，vocab=13689，1 worker | 8,614.625 | 6,986.625 | 6,686.625 | 6,325.625 |
| H=2048，vocab=128256，1 worker | 463,316.625 | 378,561.625 | 344,977.625 | 331,561.625 |
| H=2048，vocab=128256，16 worker / MPI8 | 627,605.75 | 633,581.75 | **598,204.75** | 600,088.75 |

正式单 worker：2/4/8 路分别减少 18.29% / 25.54% / 28.44%。
正式 16 worker：2 路退化 0.95%，4/8 路分别减少 4.68% / 4.38%。
因此只选择 **4 路**进入端到端验收，不能由单 worker 的 8 路最优外推全局结论。
12 个真实 SST case 均通过；每个正式 16-worker case 执行 256 个 job 和 64,128 个 dense 权重 tile。
局部 backing 每节点 256 MiB，完整模型每节点 2 GiB；这只隔离工作负载，不修改 Ramulator 组织或带宽。
不同容量不用于跨实验声称 cycle 收益，所有百分比来自各自严格配对。

局部 SST wall：小形状约 8.3–8.7 秒，正式单 worker 约 23.1–25.3 秒，正式 16 worker / MPI8 约 28.9–29.8 秒。
不包含输入准备、host oracle 和审计；单 worker 与多 worker 实验并行运行，wall 用于记录试验成本。

## 为什么并行收益有限

正式 case 的权重入口流量均为 **525,336,576 bytes**，输入/权重/输出 payload 总计 **526,641,664 bytes**，
各候选完全相同。4 路消除了重复的 GM input read 和部分 scatter/gather 请求；矩阵请求、矩阵入口字节、
矩阵 sink 字节、scatter 字节和输出字节不变。所有减少的请求数及端口服务周期都有精确差值审计。

16 worker 下，core4 的 4,032 次矩阵编程及其 133,056 个物理端口服务 cycle 保持不变；
input scatter 从 4,032 次降为 1,024 次，output gather 从 126 次降为 32 次。
core16 的前台预取等待从 47,328 增至 247,143 cycles，平均权重 DMA RTT 从 237.89 增至 290.05 cycles；
8 路 RTT 进一步增至 373.81 cycles。计算间隔缩短后等待会显露更多，DMA 调度顺序也发生改变；
这些数值不能相加为 wall time，也不能单独证明某一种 DRAM bank 冲突。

按每节点 320 bytes/cycle 的 payload 必要下界，最重的 node4 为 **413,896.4 cycles**；
比聚合带宽下界 411,438.8 cycles 略强，因为 node4 还负责所有最后 token 输入读取。
这不包含 DRAM command/refresh、NoC 和软件控制开销，也不是可达到延迟的预测。
N=1 / N=4 局部时间分别是该下界的 1.516 / 1.445 倍。

LM head 有 262,668,288 个有效 MAC，16-worker 峰值为 65,536 MAC/cycle，纯计算下界只有 4,008 cycles。
有效 MAC 峰值效率从 **0.639% → 0.670%**；按 66-cycle 物理 MAC 的阵列加权忙碌率从 0.659% → 0.691%。
这些是工作量/总阵列时间，不是“任意一个阵列忙”的时间并集。
同一轮最多活跃的阵列由每 worker 1 个变为 4 个，但权重带宽约束决定了它不可能得到 4 倍阶段加速。

## 正式两层端到端验收

S=1024、H=2048、I=8192，两层，包含架构内 Embedding、final RMSNorm 和最后 token 的 128256 词表输出。
使用合成权重、完整执行 dense tile；真实 checkpoint 导入仍未推进。两组均为 MPI rank=8，
保留已验收的四槽预取、FFN down/残差融合与两个 32-array 子组，唯一功能差异为 LM N 并行开关。

| 阶段 | N=1 cycle | N=4 cycle | 节省 |
| --- | ---: | ---: | ---: |
| LM head | 633,901.75 | **602,504.75** | 31,397（4.9530%） |
| token IDs DMA → 最终 logits ACK | 6,262,408.75 | **6,231,011.75** | 31,397（0.5014%） |

基线精确复现上一轮已验收结果。其余 10 个可加阶段（含所有 handoff）周期完全相同；
前序计算、LM 首个输入 DMA 时间相同，阶段节省精确等于端到端节省。1 GHz 下完整延迟为 **6.2310 ms**。
全链路数学有效 MAC / 峰值效率从 31.4600% 提升到 31.6185%；不是逐阵列实际忙碌率。

两组各 416 个 Decoder job 的资源生命周期审计通过；256 个 LM job、64,128 个 LM 权重 tile，
以及 393,216 个 FFN 权重 tile 的工作量保持不变。候选提交 16,384 次多阵列 LM launch，
array 加权 K-tile 数仍为 64,128。两层层输出、Embedding、Wo/H1/Norm2、final norm 与 logits 均 FP16 逐位一致；
greedy token 均为 **99037**。所有实际 SFU unit 的 accepted-token 数以及 FFN 分组工作量完全相同。

所有阶段的 DMA 读写字节和请求数相同；只有 LM 的软件局部调度改变：
GM read 减少 **6,111,232 bytes / 47,744 requests**，GM write 字节不变、请求减少 **1,492 次**；
实际 array buffer 服务合计减少 **82,104 个端口 cycle**。这是跨 worker 的服务量之和，不能与端到端节省相加。
矩阵编程成本、物理输入/输出字节不变。完整 LM 时间仍是 413,896.4-cycle payload 下界的约 **1.456 倍**。

生产 154 项、host 142 项、架构 6 项，共 **302 项回归通过**，另外完成 12 个真实局部 case 与 2 个完整 case。
13 个硬件文件及编译副本与修改前相同；上一轮验收快照中另外 25 个共同源文件保持相同 SHA256，
包括 SFU、WCP、CPU builder、拓扑与 guest runtime；只有调度头文件和 full runner 发生预期变化。

完整 SST wall 为基线 **1137.64 秒（18分58秒）**、候选 **1158.12 秒（19分18秒）**；
包含 host 验证的 driver wall 分别为 1185.05 / 1203.90 秒，均不包含之前的输入生成。
两组在不同 CPU 集合并行运行，这组 wall 不支持“仿真更快”的结论；cycle 优化与 simulator wall 分开报告。

本轮只对局部优胜者 4 路做完整验收，未声称它是所有调度的全局最优。
保留默认 N=1，正式配置可显式增加 `--lm-n-parallel 4`。后续若继续做 LM 调度，
可以先在局部试验中交错不同 N tile 的编程与 MAC，逐步取消当前的批内同步，再决定是否值得完整仿真。

轻量证据归档见 [lm_n_parallel](progress_results/2026-10-15/lm_n_parallel/)，
其中 `full_comparison.json` 含全部 11 段可加周期、张量哈希、资源和端口差值。

## 复现命令

```bash
# 先局部筛选：小形状使用 --shape small --workers 1 --mpi-ranks 1 --vocab 13689。
python3 baseline/attention_cluster_8qk_8pv/run_lm_head_micro.py \
  --artifact-root results/lm_local_repro --shape formal --workers 16 \
  --mpi-ranks 8 --cpu-set 0-7 --parallel 1 2 4 8
python3 baseline/attention_cluster_8qk_8pv/analyze_lm_head_local.py \
  --root results/lm_local_repro --output results/lm_local_repro/analysis.json

# 完整模型仅保留局部优胜者；基线把最后参数改成 1。
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/lm_stack_repro --shape formal --embedding \
  --mpi-ranks 8 --cpu-set 0-7 --gemm-prefetch --prefetch-slots 4 \
  --ffn-down-fusion --down-residual-fusion --ffn-panel-tiles 16 \
  --ffn-array-groups 2 --weight-trace --lm-n-parallel 4
```

原始证据在 `results/lm_parallel_20261015/`。局部入口最初暴露的环境变量转发和审计配置问题已修复并单独保留失败记录；
上表只使用最终完整通过的严格配对结果。已结束仿真的 backing 文件通过 `fallocate --dig-holes` 回收零页，
每个文件前后 SHA256 相同、逻辑大小与地址不变；没有删除结果或修改正在运行的文件。

完整比较命令：

```bash
python3 baseline/attention_cluster_8qk_8pv/compare_lm_parallel_stack.py \
  --baseline results/lm_parallel_20261015/stack_baseline \
  --candidate results/lm_parallel_20261015/stack_n4 \
  --output results/lm_parallel_20261015/full_comparison.json
```
