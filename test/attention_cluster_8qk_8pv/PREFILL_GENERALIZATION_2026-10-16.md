# Prefill 通用性与数值验证（2026-10-16）

本轮优先补齐部署前的功能与数值证据，未修改加速器硬件配置或时序模型。
每 worker 的 GM 仍为 2 MiB，沿用现有 GM bank/端口、阵列、SFU 共享管线和
HBM 模型。guest 的循环、地址布局及 host 输入/验证工具发生变化。
使用合成权重；真实 checkpoint 导入仍按此前决定暂缓。

## 四项实现

1. **层数参数化**：guest 通过 `GOLEM_DECODER_LAYERS` 执行 1–32 层。
   所有层共用布局生成器与 scratch，逐层复用 completion 标志并检查 epoch、
   清零 ACK、scratch retire 和输出 DMA ACK。默认双缓冲，不逐层新增硬编码地址。
2. **变长输入**：CLI 接受逻辑 S=1–8192，物理执行长度向上补齐到 1024 的倍数。
   causal mask 与 RoPE 位置照常执行；验证只对真实前缀作数值判定，LM head
   从真实 `S-1` 的 node/row 取输入，支持跨 node 及跨 1024 行组。
   **没有实现原生任意尾行 kernel**，补齐行的计算、访存全部计入 cycle。
3. **稠密权重与精度测量**：新增固定种子的稠密随机权重，并保留既有 sparse-valued
   回归输入。二者都执行完整 dense 硬件 tiles。独立 host 标量 GEMM 模拟既有
   FP16 舍入边界，组合整层及整条链进行功能验证；另用纯 FP32 网络计算每层
   relative-L2/RMSE/max-abs、最终 norm/logits 误差及 greedy top-1 是否一致。
   FP32 误差是测量值，不能据此推断真实模型任务质量。
4. **分级验收**：局部小尺寸筛选后，运行正式宽度两层回归，以及小尺寸 16 层
   稠密测试。诊断模式 `--retain-layer-outputs` 为各层输出分配现有 HBM 空间，
   不增加 GM、不额外复制张量；逐层验证后与常规双缓冲的最终输出逐位比较。

## 复现

```bash
source scripts/env_local_install.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 baseline/attention_cluster_8qk_8pv/run_prefill_validation.py \
  --artifact-root results/my_prefill_validation --stage all --cpu-set 0-7
```

也可按 `--stage local`、`formal`、`target` 分开执行。工具冻结源码及库哈希，
已完成的 PASS case 可复用。只修改 host 验证逻辑时，可先使用
`run_decoder_stack.py --artifact-root <case> --verify-only` 重放原始结果。
复用旧仿真要求验证器哈希更新、原始源码快照可信，且剔除 host 检查分支后
输入生成、guest 环境和仿真启动代码的 AST 完全相同；其余源码与库仍须同哈希。

本轮证据在 `results/prefill_generalization_20261016/`。轻量归档位于
`progress_results/2026-10-16/prefill_generalization/`，包括验收 JSON、周期汇总、
测试日志及来源记录，不复制 HBM 大文件。

## 实测验收结果

9 个 SST 用例全部通过数值、标志/epoch、scratch、DMA、资源流量及 GM bank 审计。
所有 case 的 GM 非定时张量访问字节数均为 0。测试结果为 166 项 host 测试和
155 项 production 测试通过。完整 case 表及逐层连续阶段统计见归档
[`summary.md`](progress_results/2026-10-16/prefill_generalization/summary.md)，
总验收见 [`all_validation.json`](progress_results/2026-10-16/prefill_generalization/all_validation.json)。

| 验收形状 | 权重 / 模式 | 完整入口 cycle | SST wall 秒 |
| --- | --- | ---: | ---: |
| H256 / 1 层 / S1→1024 / vocab65 | dense，8 rank | 375,056.625 | 30.269 |
| 同上 | dense，2 rank | 375,056.625 | 53.894 |
| H256 / 3 层 / S65→1024 / vocab8201 | dense，留存输出 | 992,489.625 | 82.790 |
| H256 / 3 层 / S257→1024 / vocab65 | dense，留存输出 | 977,558.625 | 85.207 |
| H256 / 3 层 / S1025→2048 / vocab8201 | sparse-valued，双缓冲 | 1,595,437.625 | 106.553 |
| H2048 / 2 层 / S1024 / vocab128256 | sparse-valued，双缓冲 | 7,424,562.000 | 1,091.151 |
| H256 / 16 层 / S65→1024 / vocab8201 | dense，留存输出 | 4,936,749.625 | 437.787 |
| 同上 | dense，双缓冲 | 4,934,678.625 | 440.342 |

此外 dense 的 1 层 S33→1024 测试通过，cycle 为 374,803.625。
MPI 2/8 的全部执行行最终输出、final norm 和 logits 逐位一致，cycle 相同；
wall 是单次实测，不能视为严格统计性能结论。
16 层留存/双缓冲也逐位一致，两种布局的地址与同步会造成小幅 cycle 差异。
留存模式检查了每一层；常规模式前 14 层输出被覆盖，验证器独立组合完整链，
再验证仍驻留的末两层，不能声称从双缓冲 HBM 读回了所有中间输出。
16 层核对了 256 个 projection jobs、1024 个 decoder ops、512 次 reset ACK
和 512 次 scratch retire。

16 层 relative-L2：末层输出相对纯 FP32 为 **0.1255%**，final norm 为
**0.1263%**，logits 为 **0.1752%**；greedy token 均为 6888。
这些指标仅适用于当前合成权重和输入。现有内部计算精度保持不变。

正式尺寸与此前 `ffn_gm_flow_20261016/stack_layout2` 的 105 项 preload 地址及
载荷哈希相同，两层输出、final norm、logits 逐位相同，token 仍为 99037。
总 cycle 从 7,362,979.75 增至 7,424,562，**增加 0.8364%**；LM head 算子跨度从
640,290.75 增至 694,273 cycle。不能把本轮功能改造宣传为性能优化。
当前 guest 在每个词表块内重复读取/解析逻辑长度，是后续可用局部实验排查的
软件开销；尚未通过配对实验将全部 cycle 增量归因于该操作。

通过后的重复 MPI 日志已按哈希核对并删除，零块已回收；本批保留约 7.3 GiB，
build 刷新后仍约 9.4 GiB。失败诊断及修复前/后的来源记录保留，归档不复制大张量。

## 验收范围与限制

小尺寸使用 H=256、FFN=1024、Hq=4、Hkv=2；正式尺寸使用 H=2048、
FFN=8192、Hq=32、Hkv=8。两者 head dimension 均为 64。
本轮完整链包含 Embedding、N 层 decoder、final RMSNorm、最后一个真实 token
的 LM head 和 host greedy 选择。MPI rank 数仅影响宿主仿真分工。

16 层正式宽度的布局已检查能放入现有每数据节点 2 GiB HBM，但本轮不把
布局容量检查称为正式宽度 16 层 SST 验收；后者需另行运行。
同样，合成权重验证不等价于真实 Llama checkpoint 推理质量验证，且尚未覆盖 decode。

短序列 cycle 包含大量补齐计算，因此不能直接当作高效短序列部署性能。
诊断留存布局与普通双缓冲布局也不能直接作优化收益对比。
真实 checkpoint 导入继续暂缓；后续可评估原生尾行和 decode。
获得真实权重后，仍需补充对应的数值与质量验证。

## 数值契约与诊断记录

组合参考的 GEMM、SiLU 和存储边界模拟既有舍入，但 Attention 沿用独立的
FP64 数学参考，并非设备分块计算的逐位复制。因此组合链允许小幅误差，
不能宣称所有算子都是 bit-exact。16 层每层输出仍使用固定 0.004 绝对阈值；
末层保留下来的算子另外逐项检查，Wo、残差及 LM head 的局部 GEMM 要求零误差。

16 层诊断发现：逐层输出最大误差为 0.001953125，final RMSNorm 局部误差为 0，
但组合 final RMSNorm 最大误差 0.0078125 超过此前两层使用的固定 0.006 阈值。
原始失败保留在 `sixteen_layer_numerical_diagnosis.json`。
这不是把该固定阈值的结果改称通过；多层验收改用明确的输入误差传播界，
同时保留局部算子检查及原始最大误差。

令输入绝对误差预算为 δ=0.004，实际输入为 x，
`r=sqrt(mean(x²)+epsilon)`，`r_min=max(r-δ,sqrt(epsilon))`。
RMS 是 Lipschitz 连续的，因此归一化前的误差不超过 δ 时，每个输出的连续值误差满足：

```
|Δy_i| <= |gamma_i| * δ / r_min * (1 + |x_i| / r)
```

验收先检查输入确实满足 δ，再检查设备 norm 输出与对实际输入计算的 host norm
在 0.002 内一致，最后检查组合输出处于上述传播界加 FP16 舍入余量和局部预算内。
误差界是保守的最坏情况界，不等于精度质量目标；FP32 relative-L2/RMSE/max-abs
作为独立指标报告，不用该传播界替代真实 checkpoint 质量验收。

另一个 host 验证器问题是 S=2048 时误将 QK job 数乘以行组数。
实际 worker/head 分区 job 数固定，增长的是每个 job 的行块数。
修复后复用已经完成的 SST 输出进行验证，未为计数公式重新跑全局仿真。
