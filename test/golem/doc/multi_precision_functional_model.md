# LLM 原生多精度 PE 功能模型

本次只移植原生 CU/TU/TE 数值契约和 PE 时序表，不包含 TileLang、TileBin、
编译器描述符或新 guest ISA。来源为 RISC-V-CIM-Manycore-SST 的版本化模型；
本项目的实际验证以机制移植 gate 和本地专项测试为准，不引用来源仓库的
历史实验作为 LLM 通过证据。

## 精度与数据布局

| compute_mode | 输入位宽 | 每个 32-bit 包的 K lanes | 累加/输出 |
| --- | ---: | ---: | --- |
| int4 | 4 | 8 | INT32 |
| int8 | 8 | 4 | INT32 |
| fp16 | 16 | 2 | FP32 |
| bf16 | 16 | 2 | FP32 |
| tf32 | 32 | 1 | FP32 |
| fp32 | 32 | 1 | FP32 |
| fp8e4m3 | 8 | 4 | FP32 |
| fp8e5m2 | 8 | 4 | FP32 |

INT4 有符号值范围为 [-8,7]，沿 K 维较早元素占低 nibble。
BF16 和 FP8 使用原始位；TF32 使用 FP32 存储，计算前截断尾数。
A 按行、B 按列打包，每条 K 向量为 `ceil(K * input_bits / 8)` 字节。
部分 C 和最终 C 始终为 32 位。

原生 TU 通过 `configurePrecision(spec, activeK)` 显式启用，数值客户端使用
`setMatrixBits` / `setVectorBits` 和 `outputBits` / `setOutputBits`。
TE 使用 `startPrecisionWindow(header, spec)`，要求 `operand_layout=0`、
`elem_bytes=4`、`block_k=hw_input_size` 以及正确的 packed 行跨度。
`PrecisionSpecV02` 的保留字段必须为零；只接受支持的版本和精度组合。
`useLegacyPrecision()` 恢复原有算术与时序，默认 LLM guest 不启用新精度。

## CU 数值契约

`arithmetic_version=1` 定义功能契约，不宣称与 RTL 逐位相同。
浮点乘法、加法分别采用 host FP32 最近偶数舍入，不融合为 FMA。

- INT4/INT8：有符号乘法，累加按 modulo 2^32 定义。
- FP16/BF16：两个 FP32 累加链，偶数 K lane 的链接受外部部分和，
  奇数 lane 链从零开始，最后合并。每个 K 分块保留该拓扑。
- FP32/TF32：单链累加，TF32 输入保留高 10 位显式尾数。
- FP8：每包四个乘积使用共同最大指数，经 16-bit 对阶窗口和整数归约，
  再转为 FP32 累加。指数为零的输入清零，最大指数编码按普通指数解释，
  未引入完整 IEEE/FN 特殊值处理。

来源 RTL 的相消和 BF16 指数回绕缺陷不复制到功能模型。双链归约顺序、
TF32 截断和 FP8 有限对阶可能使一般输入的结果与普通 matmul 不同。
既有 LLM FP16 的逐乘加舍入契约不变；它不使用此新精度路径。

## 层级与时序

`MultiPrecisionComputeUnit` 为普通 C++ CU；TU 每个输出行对应一个 CU。
TE 负责输出列、K 分块及部分 C 保存/恢复。没有逐 CU 或逐包的 SST
component、link、clock、event；一次 MV 仍产生一个 TU 完成事件。

C++ 与 Python 共用 `computeunit/peTiming.def`，单位均为 PE cycles：

```text
P = ceil(active_K / lanes_per_beat)
completion_cycles = (P - 1) * II + L
start_interval = P * II + gap
```

所有模式 II=1；INT8 L=4、FP32 L=6，其余 L=5。
FP16/BF16 gap=1，其余 gap=0。当前 TE 串行等待完成，没有跨点积重叠。
精度选择后使用这些值；默认路径保留既有 latency 配置。
DMA、NoC、缓冲和部分 C 访问费用仍由相应组件计算。

## 本地验证

```bash
g++ -std=c++17 -O2 -Isrc test/golem/unit/multi_precision_test.cpp -o /tmp/llm-precision-test
/tmp/llm-precision-test
python3 -m unittest discover -s test/golem -p test_native_packed_array.py
python3 -m unittest discover -s test/golem -p test_native_engine.py
```

CU 回归覆盖八模式、K 尾 lane、INT32 溢出、负数、种子、双链舍入顺序、
TF32 截断、FP8 对阶损失和无效精度。原生 SST TU 测试覆盖实际库的算术、
完成事件及默认时序恢复。TE 测试在确定性 memory/scheduler fixture 上
检查两个 K 分块和功能/timing 完成周期一致；完整 NoC/HBM 的默认 LLM
正确性由每阶段 GEMM、Attention、GQA、prefill 四组正式实验验证。

Timing 模式的输出占位数据不能用于数值校验。接口细节和显式 placement
限制见 `test/maintenance/mechanism_alignment/NATIVE_INTERFACES.md`。
