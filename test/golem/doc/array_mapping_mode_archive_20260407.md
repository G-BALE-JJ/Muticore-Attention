# Golem Array 映射模式归档（2026-04-07）

## 1. 目标与映射语义

本归档定义一套面向 GEMM 的核心映射模式，满足以下前提：

- task 以 C tile 切分。
- 单个 task 绑定到单个 core。
- 单个 core 内有 ncol 个 array。
- 每个 array 内有 64 个 CU，64 为可配置参数。
- 每个 array 执行 MVM，并在 k 方向跨 ktile 累加，最终得到该列输出。

该模式属于 output-stationary（按输出列常驻）数据流：

- 对固定 n_col（映射到固定 array_id），在所有 ktile 上持续累加。
- k 循环结束后，array 内的累加结果即该列最终输出。

## 2. 计算微语义（CU / Array）

### 2.1 CU 级

每个 CU 执行若干步 MAC，不是一次完成整行 dot-product：

acc <- acc + A(i, k) * x(k)

可将 CU 抽象为 mul + accumulator 的迭代链。

### 2.2 Array 级

- 一个 array 对应一列输出的并行行累加。
- 64 个 CU 并行维护 64 路输出行（或行分块）累加状态。
- array 的一次 mvm 调用对应一轮向量片段参与累加。

## 3. Core 内并行语义

单个 core 内存在 ncol 个 array 并行域：

- 每个 n_col 映射到一个独立 array_id。
- 理想执行是同一 ktile 内多个 array 并行推进。
- 控制面应尽量采用批量下发 + 统一等待，而非每列串行阻塞。

## 4. 与当前实现的一致性与偏差

当前实现中，n_col 与 array_id 的映射是存在的，且输出模式可配置为累加。

已对齐点：

- n_col -> array_id 映射：
  - tests/small/mvm_noc_int_array/gemm_matmul_op_ctrl.h
- array 累加模式（Accumulate）存在：
  - golem/array/mvmComputeArray.h
  - golem/tests/small/mvm_noc_int_array/ex_instr.h

当前偏差（主要性能口径）：

- 每次 run_mvm_compute_only 仍执行 inputmatrixload + inputvectorload + mvm。
- 在 n_col 循环内调用 run_mvm_compute_only，导致 gm2imat 按列重复。
- RoCC 当前为 busy 串行执行模型，CPU 往返开销较高。

## 5. Array 相关代码改造建议

以下按“先低风险、再结构升级”排序。

### 5.1 第一优先级（低风险，立即可做）

目标：在保持多 array 映射不变的前提下，减少重复矩阵装载。

改造思路：

- 将矩阵装载从 per-n_col 调用中拆出，改为 per-(k_tile, array_id) 装载一次。
- 在同一 k_tile 内，只更新各列向量并触发 mvm。

涉及文件：

- tests/small/mvm_noc_int_array/gemm_matmul_op.h
  - 拆分 run_mvm_compute_only 为：
    - run_mvm_load_matrix_only
    - run_mvm_load_vector_only
    - run_mvm_compute_only_no_load
- tests/small/mvm_noc_int_array/gemm_matmul_op_ctrl.h
  - 在 n_col 循环外增加矩阵装载阶段。
  - n_col 循环内只做向量装载 + mvm。

注意：

- 若每个 array 的矩阵内容相同，该改造可显著减少 CPU-RoCC 指令往返。
- 若阵列矩阵地址本就不同，需保留 array_id 维度的地址映射。

### 5.2 第二优先级（中风险，中等收益）

目标：提升 array 并行效率，降低控制串行化。

改造思路：

- 在 RoCC 侧增加按 array 的待执行上下文（轻量队列或状态表）。
- 将“提交命令”和“等待完成”分离：支持先批量提交多个 array，再统一等待。

涉及文件：

- golem/rocc/roccAnalog.h
  - 从单 busy 全局串行，演进为有限队列 + per-array in-flight 状态。
  - 为 mvm/gm2ivec/gm2imat 增加 completion token（或使用序号）。
- golem/tests/small/mvm_noc_int_array/ex_instr.h
  - 增加异步接口（submit + wait），保留现有同步接口兼容。

### 5.3 第三优先级（结构升级，需验证）

目标：引入受控广播/多播矩阵分发（若与硬件假设一致）。

改造思路：

- 增加组内广播语义，仅在同 core 内 ncol 个 array 生效。
- 广播必须计入成本：扇出、链路复制、拥塞排队。
- 提供开关回退到 unicast，便于公平对比。

涉及文件：

- golem/rocc/roccAnalog.h
  - 新增 broadcast load matrix 命令语义。
- golem/array/mvmComputeArray.h
  - 增加批量设置矩阵入口（同一 payload 分发到多个 array）。
- tests/architecture/cpu_builder.py
  - 增加广播配置参数并注入组件。

## 6. 参数与约束建议

建议显式化以下参数，避免模型漂移：

- GOLEM_NUM_ARRAYS：core 内并行 array 数，需 >= block_n。
- GOLEM_ARRAY_INPUT_SIZE：默认 64，可配置。
- GOLEM_ARRAY_OUTPUT_SIZE：默认 64，可配置。
- GOLEM_ARRAY_CLOCK：array 时钟域。
- GOLEM_LATENCY_MVM_COMPUTE_CYCLES：单次阵列计算延迟。

约束：

- block_n > GOLEM_NUM_ARRAYS 时必须报错或分批调度。
- 输入维度非 64 整数倍时，需尾块掩码或边界填充。

## 7. 验收口径

每次改造后至少记录以下指标：

- execution_summary.csv
  - compute
  - dma_issue / dma_wait
  - task_desc / nloop
- stats_selfcom.txt
  - cycles_mvm
  - cycles_mvm_gm2ivec
  - cycles_mvm_gm2imat
- run_summary.csv
  - total 与 compute share

建议新增派生指标：

- compute_per_mvm = compute_mean / mvm_count_per_core
- rocc_single_mvm = mvm + gm2ivec + gm2imat
- cpu_overhead_per_mvm = compute_per_mvm - rocc_single_mvm

该三项可直接反映控制面改造收益。
