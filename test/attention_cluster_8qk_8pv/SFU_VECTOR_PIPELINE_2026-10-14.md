# SFU Add / Multiply / SiLU 资源调度

本次将 Add、Multiply、SiLU 从“功能计算 + 整块计费”改成 SFU 的异步流水线调度。EXP 和倒数直接引用 Attention RowEngine 使用的同一个 `TensorHardwarePipeline` 对象；Add、Multiply 及 SiLU 的向量步骤使用既有 `vectorOpPipeline_`。没有增加 EXP、倒数或矩阵阵列单元，也没有做跨算子融合。

## 数值契约

输入和 HBM/GM 输出仍为 FP16。向量/SFU 内部运算按 FP32 定义，输出使用项目既有 FP16 helper 舍入。此次明确并保留这一精度契约，没有把改变精度作为加速手段。

| 操作 | 功能语义 | 舍入位置 |
| --- | --- | --- |
| Add | FP16 输入转 FP32，单次加法 | 最终 FP16 输出 |
| Multiply | FP16 输入转 FP32，单次乘法 | 最终 FP16 输出 |
| SiLU | `-x → exp → +1 → reciprocal → ×x`，各步 FP32 | 每步 FP32，最终 FP16 输出 |

SiLU 现在按实际阶段用 `x * reciprocal(1 + exp(-x))`，而不是一次 `x / denominator`。两种表达可能有 FP32 最末位差异；独立 CPU 数学参考仍按除法计算，以已有 FP16 误差限检验。不是 LUT/多项式或 RTL 位精确 EXP 模型，EXP/倒数的 latency/II 是可配置微架构假设。

C++ 的 `float` 表示功能数值，并不自动生成硬件周期。当前假设是内部 FP32 数据通路；FP16 原生 Add/Multiply 或其他近似 SFU 需要单独定义相应数值规则和 latency/II/lanes，不能只修改 C++ 类型后沿用周期参数。

## 调度及资源

生产代码在 `src/sst/elements/golem/sfu/vector_pipeline.h`；`hardware_pipeline.h` 提取既有管线结构，保留原有 `reserve()` 语义。`sfu.cc` 通过现有 SFU self-link 推进调度器。

1. 已经经过定时 GM 读取的输入复制到请求快照，最多 8192 元素。调用者可以释放输入 vector，但 result 必须存活到任务完成。
2. SiLU 分成 Negate、EXP、AddOne、Reciprocal、Multiply 五步，每一批都经过实际管线预占。功能计算发生在相应 token 的完成时刻。
3. 依赖通过携带元素索引/值的四级 FIFO 传递。发射前预留目的 FIFO 容量；已在途的数据也计入容量，队列满时停止发射。
4. 向量和 EXP 按实际 lane 数打包，仅最终尾块允许部分 lane；倒数保持既有标量单元，每 token 一个元素。各阶段可以同时执行。
5. `(request, stage)` 轮转分配向量资源，多个请求也轮转申请 EXP/倒数。与 Attention 的竞争进入原有管线队列，受同一 `nextIssueCycle`、有限深度和完成时间约束。
6. 完整输出只在最后一步全部完成后发布。`readyTick` 在此之前为无穷大，decoder 同时检查 `completed`；拒绝请求时保留输入并重试。

`vector_max_requests` 默认 8；每请求每级 FIFO 默认 64 个 FP32 元素，四级合计 1 KiB。请求快照最多 8192 元素，并发请求数也有上限。此次沿用原向量接口“先整块读取 GM，再提交计算”的边界，尚未实现 GM 读与 SFU 计算的逐 beat 重叠，也没有将 C++ 容器大小作为 RTL SRAM 面积模型。FIFO 是本次明确建模的内部缓冲，不能无限堆积元素来隐藏反压。当前 decoder 每 worker 同时只有一个向量请求。

RoPE/RMSNorm 暂时保留聚合时序模型，本次让它们保守地独占向量资源区间，避免与新请求无约束重叠。它们的归约、rsqrt 仍未按本次 SiLU 的粒度细化。Attention/decoder 的 GM、DMA、HBM 路径以及各算子 HBM 中间结果继续保留。

## 参数

参数由 `tests/architecture/cpu_builder.py` 接入；下面都是加速器 cycle。

| 配置 | 环境变量 | 组件默认 | 当前端到端脚本 |
| --- | --- | ---: | ---: |
| vector_op lanes | `GOLEM_SFU_VECTOR_OP_LANES` | 16 | 16 |
| vector_op latency / II | `GOLEM_SFU_VECTOR_OP_LATENCY` / `GOLEM_SFU_VECTOR_OP_II` | 3 / 1 | 3 / 1 |
| EXP lanes | `GOLEM_SFU_EXP_LANES` | 4 | **16** |
| EXP latency / II | `GOLEM_SFU_EXP_LATENCY` / `GOLEM_SFU_EXP_II` | 8 / 1 | 8 / 1 |
| reciprocal latency / II | `GOLEM_SFU_RECIPROCAL_LATENCY` / `GOLEM_SFU_RECIPROCAL_II` | 8 / 1 | 8 / 1 |
| 所有管线 queue depth | `GOLEM_SFU_PIPELINE_QUEUE_DEPTH` | 16 | 16 |
| 每级 FIFO 元素数 | `GOLEM_SFU_VECTOR_FIFO_ELEMENTS` | 64 | 64 |
| 请求 context 数 | `GOLEM_SFU_VECTOR_MAX_REQUESTS` | 8 | 8 |

`vector_op_lanes` 环境变量缺省跟随 `GOLEM_SFU_VECTOR_LANES`。FIFO 至少容纳一个完整 vector/EXP beat；非法容量在初始化时拒绝。当前 launcher 明确将 EXP 设为 16 lanes，所以此前按组件默认 4 lanes 算出的 11798 cycles 不能直接用来比较正式实验。

## 验证

`test_sfu_vector_pipeline.py` 执行生产调度器，覆盖：

- Add/Multiply 数值及完整输出发布，SiLU 负数尾部、非整 lane 尾块、异步输入快照。
- 每个元素的五级依赖、阶段重叠、向量/EXP/scalar reciprocal 的实际 token 数。
- 多个请求及 Add 与 SiLU 的竞争；外部请求按 Attention 的真实 `reserve()` API，在 SiLU 已在途时申请同一个 EXP/倒数对象，并交换提交顺序。
- queue depth=1、II=3 和小 FIFO 下的反压、容量界限、context 用尽后的拒绝与回收。

组件默认 EXP=4 时，8192 元素 SiLU 从 cycle 0 开始发射，在 cycle 8219 完成；端到端 EXP=16 配置下为 8216。SST 首次 pump 多一个 cycle，正式请求为 **8217 cycles**。相同实际配置下，原串行计费公式为 **10262 cycles**，计算部分下降约 **19.93%**。Add/Multiply 的 SST 请求仍为 **515 cycles**。

共享竞争测试中，257 元素 SiLU 与 Add 同时提交：无外部 EXP/倒数请求为 284 cycles，加入请求为 377 cycles；交换提交顺序也检测到竞争延迟。这是生产模型级的竞争测试，并非当前两层 decoder 真实发生了 Attention/FFN 交叠——两者目前仍由 barrier 分开。

`test_sfu_pipeline_audit.py` 验证报告拒绝丢失 completion、缺失实际 EXP 预占、FIFO 溢出、早于倒数吞吐下界的完成，以及不同输入/权重的前后比较。

53 项相关测试通过，统一日志为 `results/sfu_vector_pipeline_20261014/unit_tests.log`。旧单层小尺寸 guest 回归也通过，产物为 `legacy_layer_small_v2/decoder_result.json`。

兼容性回归发现并修复旧 Attention 报告的共享资源计数问题：此前 `report_sst.py` 将运行内全部 EXP token 当作 Attention，SiLU 进入真实 EXP 管线后造成假失败。现在按已完成 SiLU 请求及实际 EXP lane 数扣除对应的 padded work，并继续严格匹配 Attention 工作量；缺失/重复 SiLU completion 仍失败。旧单层 Attention 为 2228224 元素，另有 524288 元素属于 QK worker 上的 SiLU。修复只涉及报告脚本，没有改动仿真数据或关闭资源检查。

两层 SST 产物在 `results/sfu_vector_pipeline_20261014/`。每次运行保存 SFU、配置和 decoder 源码快照/哈希以及实际 guest/libgolem 哈希；`sfu_pipeline_audit.json` 检查每个 worker 新增加的 EXP/倒数 accepted_tokens 与 SiLU 工作量精确相等，向量 token 数也与全部 Add/Multiply/SiLU 指令数相等。

周期结果以 [实际测量对比](../../results/sfu_vector_pipeline_20261014/sfu_pipeline_comparison.md) 为准。计算周期下降只来自调度模型的改进，端到端仍包含相同的 HBM 中间张量、GM/DMA 访问及控制开销。逐算子跨 worker 的跨度不可相加。

| 已通过两层 case | 原总 cycle | 新总 cycle | 下降 | 新延迟 @ 1 GHz |
| --- | ---: | ---: | ---: | ---: |
| hidden=256 / FFN=1024 | 1013850.75 | 983693.75 | 2.975% | 0.983694 ms |
| hidden=2048 / FFN=8192 | 18690216.75 | 18453909.75 | 1.264% | 18.453910 ms |

正式配置两层独立组合参考的最大绝对误差均为 0.0009765625，末层 SiLU 为 0.0001220703125，其他末层 post-attention 算子与实际输入的逐阶段参考精确匹配。两层共 5120 个向量请求：1024 个 Add、2048 个 Multiply、2048 个 SiLU。每个 worker 的向量硬件 token 为 294912，FIFO 高水位不超过 64，实际请求高水位为 1。共享管线计数与旧基线之差精确覆盖 SiLU 工作量。

正式配置连续边界为第一层 9108137.75 cycles、层间交接 47440.25 cycles、第二层 9298331.75 cycles，三者精确相加等于总周期。仍沿用从第一层 X DMA 发出到第二层 Y 最后 DMA ACK 的计时口径。SFU 的 source/library provenance 与运行时快照一致。

## 复现

先将变更构建并安装到本工作区的 SST elements，然后运行：

```bash
source scripts/env_local_install.sh
python3 -m unittest discover \
  -s src/sst/elements/golem/tests/small/muticore_attention \
  -p 'test_sfu_vector_pipeline.py' -v
python3 -m unittest discover \
  -s baseline/attention_cluster_8qk_8pv -p 'test_sfu_pipeline_audit.py' -v
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/sfu_vector_pipeline_NEW/two_layer_small --shape small
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/sfu_vector_pipeline_NEW/two_layer_formal --shape formal
python3 baseline/attention_cluster_8qk_8pv/summarize_sfu_vector_pipeline.py \
  --artifact-root results/sfu_vector_pipeline_NEW \
  --baseline-root results/decoder_stack_20261014
python3 baseline/attention_cluster_8qk_8pv/summarize_decoder_stack.py \
  --artifact-root results/sfu_vector_pipeline_NEW
```
