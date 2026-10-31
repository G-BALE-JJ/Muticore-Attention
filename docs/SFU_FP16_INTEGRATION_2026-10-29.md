# SFU 全 FP16 同步与四阶段回归

已将 `Muticore-Attention-all-gemm` 的 SFU 精度改动同步到公共架构。
默认不设置环境变量时，SFU 的算术、归约和 online softmax 状态逐步舍入到 binary16。
显式兼容模式为 `GOLEM_SFU_INTERNAL_PRECISION=fp32`。
矩阵阵列的 dtype、累加规则、硬件资源和流水线时序参数沿用各测试原配置。

本轮不能宣布四项全部通过：GEMM、Attention、GQA 的正式配置通过；prefill 的旧验收契约尚不兼容。
所有实际仿真使用同一个公共安装库及 `full-functional` 模式，未跳过数值 MAC。
安装库 SHA256：`07f86174f4682c9f89915f05613697df3f9f8d280546179e323e79cee27f75d5`。

## 已完成的正式配置

| 测试 | 配置 | 结果 | 加速器延迟 |
| --- | --- | --- | ---: |
| GEMM | M/N/K=128，原 FP32 阵列 dtype | PASS | 3,500 cycles |
| Attention | Sq/Skv=1024，Hq/Hkv=1/1，D=128，MPI4 | PASS | 12,852 cycles |
| GQA | Sq/Skv=1024，Hq/Hkv=4/2，D=128，MPI4 | PASS | 37,928 cycles |
| prefill | 历史 selected 两层正式入口 / 显式完整候选 | FAIL：库哈希保护 / 原组合数值阈值 | 不作已验收延迟声明 |

GEMM 不是 SFU 算术覆盖证据；另行执行 FP16 阵列 dtype 的补充测试也通过，
原校验器抽查 128 个输出、零不匹配。
Attention 检查 131,072 个输出，GQA 检查 524,288 个输出，均为零不匹配；
两者相对原独立数学参考的最大绝对误差均为 `5.35836333483055e-6`，没有放宽原阈值。
两者的内存后端、MPI 放置/分区、HBM 布局检查通过，日志确认实际 `internal=fp16`。

GEMM 延迟边界为首个 descriptor 开始至最后 WCP 完成（含 C 写回，inclusive +1）；
Attention/GQA 为 QK WCP、online SFU、NoC P 分发、PV WCP 至最终 PV ACK。
Attention/GQA 各算子 span 互相重叠，不能相加。SST timebase 为 1 ps，1 GHz 参考下 1000 ticks = 1 cycle。
这些结果与旧历史值相同；本轮没有以不同数值契约宣称调度优化或硬件加速。

## prefill 的具体结果

`prefill --run` 保留历史 `SELECTED_PREFILL_2026-10-26.json` 的库/guest 哈希校验。
旧库哈希与新 SFU 库不同，因此正式入口在仿真前拒绝运行；本轮未重写历史 PASS、哈希或性能结论。

修正首次尝试的无效 MPI 绑核参数后，小规模配置 S=1（实际执行 padding 到 S=1024）、
H=256、一层 decoder、embedding、final RMSNorm、LM head 的 SST 功能仿真执行完成，
内存后端检查通过，但旧数值校验失败：

```text
final_norm_local_contract: max_abs_error=0.005859375, tolerance=0.002
```

对本次设备输入、gamma 和最终 RMSNorm 输出另作独立 NumPy 逐步 FP16 验算，
256 个输出全部一致、最大误差为 0。该失败来自旧 FP32 内部参考与新规则的差异，
不能通过放宽阈值或回退 FP32 将其标为通过，也不代表已完成整条 prefill 的 FP16 参考验收。

另以历史 selected 的完整参数运行显式候选，独立于固定哈希入口：
S1024 / H2048 / FFN8192 / 2 层 / vocab128256，MPI8，16 workers。
完整 SST 功能仿真与内存后端检查通过，原组合数值验收失败：

```text
layer2_composed: max_abs_error=0.017578125, tolerance=0.004
```

最终 RMSNorm 的全部 2,097,152 个输出与独立逐步 FP16 参考完全一致。
同一算子的设备输出相对旧 FP32 局部参考最大差异为 0.1171875；
这进一步确认新模式与旧精度规则的差异，不能据此宣称完整 FP16 decoder 参考已通过。
本轮保留原阈值和历史选择，未把数值失败改记为 PASS。
完整候选的模拟器宿主运行时间为 1059.847421 秒，仅为宿主墙钟时间，不是加速器延迟。
另行运行的生命周期、DMA 时序、final RMSNorm 数据流、物理资源和 GM bank 审计全部通过；
这些只证明执行和资源约束，不能覆盖数值验收失败。完整结果记录在下方汇总。
特别是历史物化中间结果属于旧数值模式，不能直接当作新 FP16 的局部输入参考。

## 同步范围与检查

- 新增 `sfu/arithmetic.h`，同步 SFU primitive、tensor/Attention、vector math 和 vector pipeline。
- `cpu_builder.py` 为所有 SFU worker 传入默认 FP16，保留显式 FP32，拒绝无效精度。
- 四阶段公共入口在执行元数据中记录实际 SFU 内部精度。
- 移植独立 binary16 转换和真实 SST SFU 精度测试，适配公共 `test/` 路径；同步逐步 FP16 SiLU 测试参考。
- Attention 的 `--check` 纳入新 SFU 精度/流水线测试，prefill 纳入 SiLU 融合测试；后续 decoder 源码快照包含 `arithmetic.h`。
- 已重新编译安装，四阶段主机检查通过，新增 SFU 10 项测试及 host SFU audit 6 项通过。
- 公共目录完整 185 项生产路径回归通过；四阶段 `--check` 分别执行 37、123、28、41 项检查（含重复覆盖）。

移植前试验区证据另见 [SFU 数值规则与试验区验证](../test/attention_cluster_8qk_8pv/SFU_FULL_FP16_2026-10-29.md)，
不可将其库哈希或测试结果误当作公共库本轮四阶段验收。

本轮原始仿真在 `test/<阶段>/results/sfu_fp16_20261029*`，
紧凑机器记录与验证日志见 [汇总 JSON](../test/reports/sfu_fp16_20261029/summary.json)。

复现：

```bash
cd /data/jjgong/LLM
bash scripts/build_and_install_local.sh --no-autogen --jobs 8
bash test/GEMM/run.sh --check
bash test/Attention/run.sh --check
bash test/GQA/run.sh --check
bash test/prefill/run.sh --check
# 使用新的、空的 --artifact-root，避免覆盖本轮结果
bash test/GEMM/run.sh --run
bash test/Attention/run.sh --run
bash test/GQA/run.sh --run
bash test/GEMM/run.sh --run -- --dtype fp16
bash test/prefill/run.sh --smoke -- --cpu-set 8-9
# 历史选择仍按原哈希保护拒绝新库；不会自动更新旧选择
bash test/prefill/run.sh --run
```
