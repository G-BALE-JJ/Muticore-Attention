# GEMM Demo 执行流与 SST 多核拓扑归档

本文归档以 `tests/fronted/gemm_demo.py` 为入口，对当前 GEMM demo 的完整执行流、SST 多核拓扑、NoC/CPU/GlobalMemory/RoCC/MVM array 的连接关系，以及指令流和数据流做集中说明，便于后续继续开发、定位问题和做架构分析。

## 1. 总览

这条链路的本质不是“Python 直接做 GEMM”，而是：

1. Python 前端负责准备输入矩阵、padding、落盘。
2. shell 脚本负责导出环境变量、生成 HBM backing file、编译 RISC-V 测试程序、启动 SST。
3. `test_noc_dma` 在每个 core 上运行，调用 matmul runtime。
4. runtime 进入 tiled GEMM 内核，通过 RoCC 指令、GlobalMemory、NoC、DMA 和 memory node 完成真实模拟执行。
5. 结果最终写入 `hbm_out_node*.bin`，再由 Python 工具反解成 C 矩阵输出。

## 2. 执行入口

### 2.1 Python 前端

入口文件：`tests/fronted/gemm_demo.py`

- `MatmulKernel.__call__()` 会先校验矩阵形状，再对 M/N/K 按 block 维度补齐，见 `tests/fronted/gemm_demo.py:150`、`tests/fronted/gemm_demo.py:156`。
- 补齐后的 A/B 被写为二进制：`tests/data/py_a.bin`、`tests/data/py_b.bin`，见 `tests/fronted/gemm_demo.py:162`、`tests/fronted/gemm_demo.py:167`。
- 随后拼接命令行调用 `./run_noc_dma_pipeline.sh`，传入：
  - `--dim`
  - `--global-stride-kb`
  - `--gemm-m/n/k`
  - `--gemm-block-m/n/k`
  - `--dtype`
  - `--dma-overlap`
  - `--tensor-source file`
  - `--tensor-a/--tensor-b`
  - `--dump-c`
  - `--verify-c`
  - `--orig-m/n/k`
  见 `tests/fronted/gemm_demo.py:172`。
- SST 运行完后，读取 `py_c_out.csv`，裁剪回原始未补齐形状，见 `tests/fronted/gemm_demo.py:214`。

### 2.2 当前 demo 实例参数

在 `__main__` 中，demo 默认请求：

- `M=576`
- `N=6`
- `K=64`
- `BM=64`
- `BN=1`
- `BK=64`
- `dtype="fp32"`

见 `tests/fronted/gemm_demo.py:286`、`tests/fronted/gemm_demo.py:292`。

因此本例对应：

- `m_tiles = 576 / 64 = 9`
- `n_tiles = 6 / 1 = 6`
- `k_tiles = 64 / 64 = 1`
- `total_tasks = 9 * 6 = 54`

## 3. Shell 总控流程

总控脚本：`tests/run_noc_dma_pipeline.sh`

脚本头部已经明确说明它负责四件事，见 `tests/run_noc_dma_pipeline.sh:4`：

1. 设置 GOLEM 维度、RoCC、NoC、DMA 等参数。
2. 生成 HBM 初始化文件。
3. 编译 `test_noc_dma`。
4. 运行 `sst architecture/ncores_selfcom_dma.py`。

实际执行阶段在：

- 生成 HBM：`tests/run_noc_dma_pipeline.sh:1068`
- 编译 binary：`tests/run_noc_dma_pipeline.sh:1075`
- 启动 SST：`tests/run_noc_dma_pipeline.sh:1082`
- 反解 C / 校验 / 导出统计：`tests/run_noc_dma_pipeline.sh:1127`

### 3.1 关键约束

这个脚本对 phase-1 GEMM 有几条很重要的约束：

- `M/N/K` 必须能整除 `block_M/N/K`，见 `tests/run_noc_dma_pipeline.sh:662`。
- `block_M == GOLEM_DIM`，见 `tests/run_noc_dma_pipeline.sh:677`。
- `block_K == GOLEM_DIM`，见 `tests/run_noc_dma_pipeline.sh:677`。
- `block_N <= GOLEM_DIM`，见 `tests/run_noc_dma_pipeline.sh:682`。
- `dtype` 仅支持 `int32|fp32`，见 `tests/run_noc_dma_pipeline.sh:692`。

### 3.2 它导出的关键环境变量

脚本在 `tests/run_noc_dma_pipeline.sh:840` 之后统一 export 环境变量，供三个层面消费：

- `tools/gen_hbm_init.py`
- `architecture/ncores_selfcom_dma.py`
- C++ runtime / test binary

关键变量包括：

- GEMM 形状：`GOLEM_GEMM_M/N/K`
- block：`GOLEM_GEMM_BLOCK_M/N/K`
- 类型：`GOLEM_MATMUL_DTYPE`
- 并行度：`GOLEM_TOTAL_CORES`、`GOLEM_TOTAL_GEMM_CORES`
- NUMA：`GOLEM_NUM_MEMORY_NODES`、`GOLEM_MEM_NODE_SIZE_BYTES`
- GM：`GOLEM_GLOBAL_STRIDE_KB/BYTES`
- DMA：`GOLEM_DMA_OVERLAP`、`GOLEM_DMA_STAGGER_CYCLES`、`GOLEM_DMA_MAX_INFLIGHT`、`GOLEM_DMA_READ_RETRY_TICKS`、`GOLEM_DMA_READ_MAX_RETRIES`、`GOLEM_DMA_BURST_BYTES`
- NoC：`GOLEM_NOC_LINK_BW`、`GOLEM_NOC_XBAR_BW`、`GOLEM_NOC_FLIT_SIZE`、`GOLEM_MESH_DIM_X`
- 输入输出文件：`GOLEM_TENSOR_A_FILE`、`GOLEM_TENSOR_B_FILE`、`GOLEM_DUMP_C_FILE`

## 4. HBM 初始化与任务数据布局

数据生成器：`tests/tools/gen_hbm_init.py`

### 4.1 总体布局原则

- memory node 0 作为 OS 节点，不放 GEMM 数据，见 `tests/tools/gen_hbm_init.py:48`、`tests/tools/gen_hbm_init.py:49`。
- 其余节点为 data node，见 `tests/tools/gen_hbm_init.py:50`。
- `IDENTITY_BASE = MEM_NODE_SIZE`，即 identity window 从第一个 data memory node 起步，见 `tests/tools/gen_hbm_init.py:40`、`tests/tools/gen_hbm_init.py:41`。

### 4.2 GEMM 数据区偏移

与 `pipeline_config.h` 一致的布局在：`tests/tools/gen_hbm_init.py:119` 之后。

- A tile 基区：`OFF_GEMM_MAT = 0x0`
- B packed vector 基区：`OFF_GEMM_VEC_BASE`
- C 输出基区：`OFF_GEMM_OUT_BASE`
- bias 基区：`OFF_GEMM_BIAS_BASE = MEM_NODE_SIZE - bias_stride`

### 4.3 task 到 node/slot 的映射

规则：

- `task_id = m_tile * GEMM_N_TILES + n_tile`
- task 均匀分摊到 data nodes：`_data_node_for_task()`，见 `tests/tools/gen_hbm_init.py:188`
- 每个 task 在目标 node 内再编号 slot：`_task_slot_in_node()`，见 `tests/tools/gen_hbm_init.py:192`

### 4.4 写入内容

对于每个 task：

- 按 `k_tile` 写 A tile，见 `tests/tools/gen_hbm_init.py:715`。
- 按 `(k_tile, n_col)` 写 B packed vector，见 `tests/tools/gen_hbm_init.py:735`。
- bias 若存在，则复制到每个 data node，见 `tests/tools/gen_hbm_init.py:752`。
- 最后输出：
  - `hbm_init_nodeX.bin`
  - `hbm_out_nodeX.bin`
  见 `tests/tools/gen_hbm_init.py:762`。

## 5. test_noc_dma 与 runtime 入口

入口程序：`tests/small/mvm_noc_int_array/test_noc_dma.cpp`

- 程序启动后先根据 `argv[1]` 绑定并确认当前 core，见 `tests/small/mvm_noc_int_array/test_noc_dma.cpp:37`。
- 读取环境变量中的 GEMM 形状和 dtype，见 `tests/small/mvm_noc_int_array/test_noc_dma.cpp:38`。
- 构造 A/B/C tensor descriptor，但注意 `data = nullptr`，见 `tests/small/mvm_noc_int_array/test_noc_dma.cpp:53`、`tests/small/mvm_noc_int_array/test_noc_dma.cpp:61`、`tests/small/mvm_noc_int_array/test_noc_dma.cpp:69`。
- 调用 `golemCreateMatmulKernel()` 和 `golemRunMatmul()`，见 `tests/small/mvm_noc_int_array/test_noc_dma.cpp:79`、`tests/small/mvm_noc_int_array/test_noc_dma.cpp:85`。

运行时桥接在 `tests/small/mvm_noc_int_array/golem_matmul_runtime.cpp`：

- 校验 op desc 与 tensor desc，见 `tests/small/mvm_noc_int_array/golem_matmul_runtime.cpp:47`、`tests/small/mvm_noc_int_array/golem_matmul_runtime.cpp:104`。
- `has_tensor_bindings = has_data_ptr(a) && has_data_ptr(b) && has_data_ptr(c)`，见 `tests/small/mvm_noc_int_array/golem_matmul_runtime.cpp:291`。
- 因为 `test_noc_dma.cpp` 传入的 `data` 均为空，所以 `has_tensor_bindings == false`，最终走：
  - `matmul()`
  - 或 `matmul_fp32()`
  见 `tests/small/mvm_noc_int_array/golem_matmul_runtime.cpp:293`。

这意味着：当前 demo 的真实数据输入来自 HBM backing file + identity-window DMA，而不是用户直接传给 runtime 的 host tensor 指针。

## 6. SST 顶层拓扑

顶层配置：`tests/architecture/ncores_selfcom_dma.py`

### 6.1 全局形状

- `numCpus = VANADIS_NUM_CORES`，默认 16，见 `tests/architecture/ncores_selfcom_dma.py:93`。
- `NUM_MEMORY_NODES` 默认 4，见 `tests/architecture/ncores_selfcom_dma.py:104`。
- `MESH_DIM_X` 默认 4，见 `tests/architecture/ncores_selfcom_dma.py:262`。
- `cpu_rows = ceil(numCpus / MESH_DIM_X)`，见 `tests/architecture/ncores_selfcom_dma.py:270`。
- `MESH_DIM_Y = cpu_rows + 1`，最后一行留给 memory routers，见 `tests/architecture/ncores_selfcom_dma.py:271`。

因此默认结构是：

- Mesh = `4 x 5`
- 总 router 数 = 20
- router `0..15` 服务 CPU
- router `16..19` 服务 memory nodes

### 6.2 CPU 连接

每个 core 由 `CPU_Builder.build()` 生成，见 `tests/architecture/ncores_selfcom_dma.py:254`。

每个 CPU router 挂两个本地端口：

- 一个给 core 的 L2 MemNIC
- 一个给 core 的 GlobalMemory NIC

连接发生在 `tests/architecture/ncores_selfcom_dma.py:337`。

### 6.3 NodeOS / MMU

- NodeOS：`vanadis.VanadisNodeOS`，见 `tests/architecture/ncores_selfcom_dma.py:351`
- MMU：`simpleMMU`，见 `tests/architecture/ncores_selfcom_dma.py:362`
- OS L1 cache：见 `tests/architecture/ncores_selfcom_dma.py:371`
- OS L1 通过 MemNIC 挂到 OS router，见 `tests/architecture/ncores_selfcom_dma.py:390`

### 6.4 NUMA memory nodes

每个 memory node 包含：

- `DirectoryController`
- `MemController`
- `dramsim3` backend

见 `tests/architecture/ncores_selfcom_dma.py:407`、`tests/architecture/ncores_selfcom_dma.py:429`、`tests/architecture/ncores_selfcom_dma.py:460`。

地址区间按 node 连续切分：

- node `idx` 对应 `[idx * memBytesPerNode, (idx+1)*memBytesPerNode - 1]`

见 `tests/architecture/ncores_selfcom_dma.py:402`。

backing 策略：

- node0：malloc，不用文件，见 `tests/architecture/ncores_selfcom_dma.py:441`
- node1..N-1：mmap，用 `hbm_init_nodeX.bin` 初始化，并把结果写到 `hbm_out_nodeX.bin`，见 `tests/architecture/ncores_selfcom_dma.py:448`

## 7. CPU、缓存、RoCC、GlobalMemory 结构

定义文件：`tests/architecture/cpu_builder.py`

### 7.1 每核组件

每个核心都包含：

- Vanadis CPU，见 `tests/architecture/cpu_builder.py:423`
- decoder / OS handler / branch unit，见 `tests/architecture/cpu_builder.py:431`
- LSQ，见 `tests/architecture/cpu_builder.py:443`
- DTLB / ITLB，见 `tests/architecture/cpu_builder.py:545`、`tests/architecture/cpu_builder.py:550`
- L1D / L1I，见 `tests/architecture/cpu_builder.py:530`、`tests/architecture/cpu_builder.py:537`
- L2 cache，见 `tests/architecture/cpu_builder.py:575`
- RoCC，见 `tests/architecture/cpu_builder.py:466`
- per-core `golem.GlobalMemory`，见 `tests/architecture/cpu_builder.py:489`

### 7.2 Cache 参数

- L1D：32KB，8-way，2 cycles，64B line，MSHR 32，见 `tests/architecture/cpu_builder.py:347`
- L1I：32KB，8-way，2 cycles，64B line，next-block prefetch，见 `tests/architecture/cpu_builder.py:363`
- L2：1MB，16-way，14 cycles，64B line，见 `tests/architecture/cpu_builder.py:381`

### 7.3 RoCC / Array 类型

根据 dtype 自动选择：

- `int32 -> golem.RoCCAnalogInt + golem.MVMIntArray`
- `fp32 -> golem.RoCCAnalogFloat + golem.MVMFloatArray`

见 `tests/architecture/cpu_builder.py:130`、`tests/architecture/cpu_builder.py:136`、`tests/architecture/cpu_builder.py:142`。

### 7.4 per-core GlobalMemory

GlobalMemory 的关键参数：

- `baseAddr = GLOBAL_BASE + cpuId * GLOBAL_STRIDE`
- `size = GLOBAL_STRIDE`
- `identityWindowBase = SPLIT_BASE`

见 `tests/architecture/cpu_builder.py:493`、`tests/architecture/cpu_builder.py:494`、`tests/architecture/cpu_builder.py:499`。

这里 `GLOBAL_BASE = 0x0`，`GLOBAL_STRIDE` 来自 `GOLEM_GLOBAL_STRIDE_BYTES`，见 `tests/architecture/cpu_builder.py:303`。

也就是说，每个 core 都有一个独立的 64KB/256KB 等大小的 GM 窗口，按 stride 切开。

## 8. NoC 结构

NoC builder：`tests/architecture/noc_builder.py`

- router 组件：`merlin.hr_router`，见 `tests/architecture/noc_builder.py:149`
- 拓扑：`merlin.mesh`，见 `tests/architecture/noc_builder.py:153`
- cardinal port：0-3 固定给 `+X/-X/+Y/-Y`，见 `tests/architecture/noc_builder.py:42`、`tests/architecture/noc_builder.py:159`
- local port 从 4 开始分配，见 `tests/architecture/noc_builder.py:42`、`tests/architecture/noc_builder.py:110`

默认支持的 NoC 参数：

- `link_bw`
- `xbar_bw`
- `flit_size`
- `input_buf_size`
- `output_buf_size`
- `num_vns`

定义见 `tests/architecture/noc_builder.py:48`。

## 9. 编译期地址布局与 task 描述

关键头文件：`tests/small/mvm_noc_int_array/pipeline_config.h`

### 9.1 GEMM 编译期配置

- `TILE_DIM = GOLEM_DIM`，见 `tests/small/mvm_noc_int_array/pipeline_config.h:71`
- `GEMM_M/N/K` 来自宏，见 `tests/small/mvm_noc_int_array/pipeline_config.h:72`
- `TOTAL_GEMM_CORES = GOLEM_TOTAL_GEMM_CORES`，见 `tests/small/mvm_noc_int_array/pipeline_config.h:67`
- `DMA_OVERLAP_ENABLED` 由 `GOLEM_DMA_OVERLAP` 控制，见 `tests/small/mvm_noc_int_array/pipeline_config.h:69`

### 9.2 identity window 与 memory node

- `MEM_NODE_SIZE = GOLEM_MEM_NODE_SIZE_BYTES`
- `IDENTITY_BASE = MEM_NODE_SIZE`

见 `tests/small/mvm_noc_int_array/pipeline_config.h:102`、`tests/small/mvm_noc_int_array/pipeline_config.h:103`。

这意味着：地址 `>= IDENTITY_BASE` 被视为主存/identity-window 区域，会触发 DMA 访问 memory nodes。

### 9.3 task 描述符

`GemmTaskDescriptor` 包含：

- `task_id`
- `m_tile`
- `n_tile`
- `data_node_idx`
- `task_slot_in_node`
- `a_base_mm`
- `b_pack_base_mm`
- `c_base_mm`
- `bias_base_mm`

定义见 `tests/small/mvm_noc_int_array/pipeline_config.h:147`。

构造函数在 `tests/small/mvm_noc_int_array/pipeline_config.h:231`，真正把 task 映射成物理主存地址。

### 9.4 本地 GM 工作区布局

每个 core 本地 GM 的 GEMM working set：

- `tmp = 0x0800`
- `mat_ping = 0x2000`
- `mat_pong = mat_ping + aligned(mat_bytes)`
- `vec_in = ...`
- `out = ...`

见 `tests/small/mvm_noc_int_array/pipeline_config.h:291`、`tests/small/mvm_noc_int_array/pipeline_config.h:298`。

### 9.5 DMA flag 尾部区

另外，每个 GM window 尾部 32 字节保留给 DMA 完成标志，软件禁止覆盖，见：

- `tests/small/mvm_noc_int_array/gm_config.h:25`
- `tests/small/mvm_noc_int_array/pipeline_config.h:328`

## 10. 指令封装层

指令 API 在：`tests/small/mvm_noc_int_array/ex_instr.h`

主要接口：

- `inputmatrixload()` -> `mvm.gm2imat`，见 `tests/small/mvm_noc_int_array/ex_instr.h:117`
- `inputvectorload()` -> `mvm.gm2ivec`，见 `tests/small/mvm_noc_int_array/ex_instr.h:101`
- `mvm_compute()` -> `mvm`，见 `tests/small/mvm_noc_int_array/ex_instr.h:37`
- `outputvectorstore()` -> `mvm.ovec2gm`，见 `tests/small/mvm_noc_int_array/ex_instr.h:85`
- `remote_store()` -> `remote.st`，见 `tests/small/mvm_noc_int_array/ex_instr.h:133`
- `remote_load()` -> `remote.ld`，见 `tests/small/mvm_noc_int_array/ex_instr.h:148`
- `set_len()` -> `mvm.slen`，见 `tests/small/mvm_noc_int_array/ex_instr.h:163`
- `mm2gm()`，见 `tests/small/mvm_noc_int_array/ex_instr.h:203`
- `gm2mm()`，见 `tests/small/mvm_noc_int_array/ex_instr.h:218`
- `reg2gm()`，见 `tests/small/mvm_noc_int_array/ex_instr.h:233`
- `gm2reg()`，见 `tests/small/mvm_noc_int_array/ex_instr.h:247`

## 11. 软件辅助原语

封装头：`tests/small/mvm_noc_int_array/operators.h`

### 11.1 DMA issue/wait

- `dma_remote_load_issue()`：
  - 读当前 read seq
  - seq 加一写回
  - 清零 read flag
  - `set_len(bytes)`
  - 发 `remote_load(mm_src, gm_dst)`
  见 `tests/small/mvm_noc_int_array/operators.h:77`
- `dma_remote_load_wait()` 轮询 read flag，见 `tests/small/mvm_noc_int_array/operators.h:89`
- `dma_remote_load_to_gm()` 是 issue + wait 的同步封装，见 `tests/small/mvm_noc_int_array/operators.h:94`

### 11.2 MVM stage

`run_mvm_stage()` 顺序非常明确：

1. `inputmatrixload(mat_gm)`
2. `inputvectorload(vec_gm)`
3. `mvm_compute(array_id)`
4. `outputvectorstore(out_gm)`

见 `tests/small/mvm_noc_int_array/operators.h:99`。

## 12. GEMM 内核数据流

核心逻辑在：`tests/small/mvm_noc_int_array/gemm_matmul_op.h`

### 12.1 task 调度

每个 core 处理：

- `task_id = core_id, core_id + ACTIVE_GEMM_CORES, ...`

见 `tests/small/mvm_noc_int_array/gemm_matmul_op.h:393`。

因此 16 个 active GEMM core 会轮询处理 54 个 task。

### 12.2 baseline 路径

baseline kernel 见 `tests/small/mvm_noc_int_array/gemm_matmul_op.h:149`。

对每个 `k_tile`：

1. `dma_remote_load_to_gm(core_id, A_tile_mm, local_mat, MAT_BYTES)`，见 `tests/small/mvm_noc_int_array/gemm_matmul_op.h:161`
2. 对 block 内每个 `n_col`：
   - `dma_remote_load_to_gm(core_id, B_vec_mm, local_vec_in, VEC_BYTES)`，见 `tests/small/mvm_noc_int_array/gemm_matmul_op.h:167`
   - `run_mvm_stage()` 触发 RoCC/MVM，见 `tests/small/mvm_noc_int_array/gemm_matmul_op.h:172`
   - `gm2mm(out_vec.data(), rt.local_out)` 把局部结果拉回软件缓冲累加，见 `tests/small/mvm_noc_int_array/gemm_matmul_op.h:114`
3. 所有 `k_tile` 完成后，可选做 bias，见 `tests/small/mvm_noc_int_array/gemm_matmul_op.h:178`
4. `mm2gm(c_tile -> rt.local_out)`，再 `remote_store(rt.local_out, desc.c_base_mm)`，见 `tests/small/mvm_noc_int_array/gemm_matmul_op.h:124`、`tests/small/mvm_noc_int_array/gemm_matmul_op.h:125`

### 12.3 overlap 路径

overlap kernel 见 `tests/small/mvm_noc_int_array/gemm_matmul_op.h:182`。

策略是：

- A tile 使用 `mat_ping` / `mat_pong` 双缓冲。
- 在当前 `k_tile` 的最后一个 `n_col` 计算时，提前 issue 下一块 A tile 的 DMA。
- 当前列计算完成后再 wait 下一块 DMA 完成，然后 swap active/next buffer。

真正重叠的是：

- “下一块 A tile DMA issue + 在途传输”
- 和“当前块最后一列 MVM 计算”

对应代码见 `tests/small/mvm_noc_int_array/gemm_matmul_op.h:212`、`tests/small/mvm_noc_int_array/gemm_matmul_op.h:225`。

### 12.4 bias 路径

可选 bias 逻辑在 `tests/small/mvm_noc_int_array/gemm_matmul_op.h:128`：

- 每个输出列通过 DMA 从 memory node 拉一份 bias scalar 到本地 GM。
- 用 `gm2reg()` 读出该标量。
- 对整列累加。

### 12.5 结果落点

最终结果先进入 memory node 对应的输出 backing file：

- `remote_store(rt.local_out, desc.c_base_mm)`

也就是写到 `hbm_out_node*.bin` 的 `OFF_GEMM_OUT_BASE + slot * out_tile_stride` 区域。

## 13. RoCC 内部执行流

实现文件：`golem/rocc/roccAnalog.h`

### 13.1 指令 decode

`func7` 到动作的映射：

- `0x6 -> mvm.ovec2gm`
- `0x7 -> mvm.gm2ivec`
- `0x8 -> mvm.gm2imat`
- `0x9 -> remote_st`
- `0xA -> remote_ld`
- `0xB -> mvm.slen`
- `0xC -> mvm.ocfg`
- `0xD -> mm2gm`
- `0xE -> gm2mm`
- `0xF -> reg2gm`
- `0x10 -> gm2reg`

见 `golem/rocc/roccAnalog.h:323`。

### 13.2 与 array / GM 的实际交互

- `OutputvectorStore()`：从 array 输出向量组包，写入本地 GlobalMemory，见 `golem/rocc/roccAnalog.h:576`
- `IntputvectorLoad()`：从本地 GlobalMemory 读向量，装入 array 输入，见 `golem/rocc/roccAnalog.h:615`
- `InputMatrixLoad()`：从本地 GlobalMemory 读矩阵，装入 array 权重/矩阵输入，见 `golem/rocc/roccAnalog.h:654`
- `RemoteStore()`：从本地 GM 读数据，再调用 `globalMem->wr_to_network()`，见 `golem/rocc/roccAnalog.h:714`
- `RemoteLoad()`：调用 `globalMem->rd_to_network(remote_addr, length, local_addr)`，见 `golem/rocc/roccAnalog.h:739`

## 14. GlobalMemory、NoC 与 DMA 路径

实现文件：`golem/globalmemory/globalmemory.cc`

### 14.1 两种地址语义

`GlobalMemory` 会先判断访问地址：

- 若 `addr < identityWindowBase`：按 per-core GM 网络地址解释，走“远端 GM 访问”。
- 若 `addr >= identityWindowBase`：按主存物理地址解释，走“DMA 到 host/memory node”。

对应入口：

- `wr_to_network()`，见 `golem/globalmemory/globalmemory.cc:300`
- `rd_to_network()`，见 `golem/globalmemory/globalmemory.cc:342`

判断点分别在：

- `golem/globalmemory/globalmemory.cc:307`
- `golem/globalmemory/globalmemory.cc:355`

### 14.2 identity-window DMA read

当 `remote.ld` 的源地址在 identity window 时：

1. `rd_to_network()` 进入 `dma_read_from_host_to_globalmem()`，见 `golem/globalmemory/globalmemory.cc:361`
2. DMA 按 `dma_burst_bytes` 切块，见 `golem/globalmemory/globalmemory.cc:853`
3. chunk 被记录进 `dma_pending`，见 `golem/globalmemory/globalmemory.cc:882`
4. 按 `dma_read_max_inflight` 窗口发出，见 `golem/globalmemory/globalmemory.cc:483`
5. 若超时则按 retry ticks / max retries 重发，见 `golem/globalmemory/globalmemory.cc:508`
6. 收到 `DMA_READ_COMPLETE` 后，把数据写回本地 GM，并更新 completion flag，见 `golem/globalmemory/globalmemory.cc:999`、`golem/globalmemory/globalmemory.cc:1035`、`golem/globalmemory/globalmemory.cc:1041`

### 14.3 identity-window DMA write

当 `remote.st` 的目标地址在 identity window 时：

1. `wr_to_network()` 进入 `dma_write_to_host()`，见 `golem/globalmemory/globalmemory.cc:312`
2. 同样按 burst 分块，见 `golem/globalmemory/globalmemory.cc:745`
3. memory node 端收到 `DMA_WRITE` 后写入本地 backing，并回送 `DMA_WRITE_COMPLETE`，见 `golem/globalmemory/globalmemory.cc:1194`
4. 源端收到完成后更新 write completion flag，见 `golem/globalmemory/globalmemory.cc:1064`

### 14.4 非 identity-window 远端 GM 访问

若地址落在别的 core 的 GM 窗口内：

- `WRITE`：直接对目标 GM 存储写入，见 `golem/globalmemory/globalmemory.cc:957`
- `READ`：目标端读出数据后，回送一个 `WRITE` reply，见 `golem/globalmemory/globalmemory.cc:968`

### 14.5 endpoint 与 router 解析

- 按 per-core GM 地址解析 endpoint：`resolveEndpointForAddress()`，见 `golem/globalmemory/globalmemory.cc:1330`
- 按物理地址解析目标 memory node：`getDmaTargetNode()`，见 `golem/globalmemory/globalmemory.cc:1120`
- 转 router id：`getDmaTargetRouter()`，见 `golem/globalmemory/globalmemory.cc:1125`
- 尝试找 DirectoryController 的 MemNIC endpoint：`getMemNicEndpointId()`，见 `golem/globalmemory/globalmemory.cc:1137`

## 15. mailbox 与同步机制

### 15.1 mailbox 布局

`pipeline_config.h` 预留了 mailbox：

- `seq = 0x10`
- `ack = 0x40`

见 `tests/small/mvm_noc_int_array/pipeline_config.h:315`。

`operators.h` 也实现了跨 group 的 mailbox helper，见：

- `seq_addr_for_group()`：`tests/small/mvm_noc_int_array/operators.h:52`
- `ack_addr_for_group()`：`tests/small/mvm_noc_int_array/operators.h:56`
- `notify_next_group_ready()`：`tests/small/mvm_noc_int_array/operators.h:124`
- `notify_prev_group_done()`：`tests/small/mvm_noc_int_array/operators.h:130`

### 15.2 但当前 GEMM 主路径主要用 DMA flag

虽然 mailbox 机制存在，但在这个 GEMM demo 主路径里，核心同步并不是 mailbox，而是每核 GM window 尾部的 DMA 完成标志：

- `read seq`
- `read flag`
- `write seq`
- `write flag`

见 `tests/small/mvm_noc_int_array/gm_config.h:25`。

软件侧 issue DMA 前写 seq、清 flag，完成后由 `GlobalMemory` 写回 flag；等待方通过轮询 `adaptive_wait_eq()` 完成同步，见 `tests/small/mvm_noc_int_array/operators.h:20`、`tests/small/mvm_noc_int_array/operators.h:77`。

## 16. 输出 C 的反解路径

工具：`tests/tools/unpack_c_from_hbm.py`

- 它会重新根据 `m/n/k`、`block_m/n/k`、`num_memory_nodes` 以及 task/node/slot 映射规则，遍历所有 task，见 `tests/tools/unpack_c_from_hbm.py:64`、`tests/tools/unpack_c_from_hbm.py:85`。
- 对每个 task，定位其在 `hbm_out_nodeX.bin` 中的 tile 偏移：`off_gemm_out_base + slot * out_tile_stride`，见 `tests/tools/unpack_c_from_hbm.py:103`。
- 然后把 tile 内容拼回最终二维矩阵 C，见 `tests/tools/unpack_c_from_hbm.py:109`。

这就是 `gemm_demo.py` 最后读到的 `py_c_out.csv` 的来源。

## 17. 关键文件职责索引

### 前端 / orchestration

- `tests/fronted/gemm_demo.py`
  - Python API、padding、二进制输入输出、shell 调度。
- `tests/run_noc_dma_pipeline.sh`
  - 全流程总控：HBM、编译、SST、校验、统计。

### HBM 与文件工具

- `tests/tools/gen_hbm_init.py`
  - 生成每个 data memory node 的初始化 backing file。
- `tests/tools/unpack_c_from_hbm.py`
  - 从 `hbm_out_node*.bin` 反解最终 C。
- `tests/golem_dtype.py`
  - dtype 归一化、pack/unpack。

### SST 拓扑

- `tests/architecture/ncores_selfcom_dma.py`
  - 顶层系统：CPU、NodeOS、NoC、NUMA memory nodes。
- `tests/architecture/cpu_builder.py`
  - 每核 CPU/cache/RoCC/GM 结构。
- `tests/architecture/noc_builder.py`
  - merlin mesh 构建与 local attach。

### test binary / runtime

- `tests/small/mvm_noc_int_array/test_noc_dma.cpp`
  - 主程序入口。
- `tests/small/mvm_noc_int_array/golem_matmul_runtime.cpp`
  - runtime API、参数校验、分派到底层 kernel。
- `tests/small/mvm_noc_int_array/golem_matmul_runtime.h`
  - ABI / runtime type 定义。

### GEMM kernel / software ISA wrapper

- `tests/small/mvm_noc_int_array/pipeline_config.h`
  - task 划分、主存地址、local GM 布局。
- `tests/small/mvm_noc_int_array/gemm_matmul_op.h`
  - tiled GEMM、DMA、overlap、bias、结果落盘。
- `tests/small/mvm_noc_int_array/operators.h`
  - DMA 和 MVM 封装、mailbox 工具。
- `tests/small/mvm_noc_int_array/ex_instr.h`
  - 内联汇编指令包装。
- `tests/small/mvm_noc_int_array/gm_config.h`
  - 每核 GM 窗口与 DMA flag 地址助手。

### SST element internals

- `golem/rocc/roccAnalog.h`
  - RoCC 指令 decode 与执行。
- `golem/rocc/roccAnalogInt.h`
- `golem/rocc/roccAnalogFloat.h`
  - int / fp32 变体。
- `golem/globalmemory/globalmemory.h`
- `golem/globalmemory/globalmemory.cc`
  - GM 本地存储、远端 GM 访问、identity-window DMA、retry、completion。
- `golem/array/mvmIntArray.h`
- `golem/array/mvmFloatArray.h`
- `golem/array/mvmComputeArray.h`
- `golem/array/computeArray.h`
  - MVM array 抽象与实现。

## 18. 最终结论

当前这条 GEMM demo 链路可以概括为：

1. Python 前端把补齐后的 A/B 写入文件，并调用 shell 总控。
2. shell 根据环境变量和命令行，把 A/B/bias 按 task/node/slot 布局写入 `hbm_init_node*.bin`。
3. shell 编译 `test_noc_dma`，再启动 SST 顶层配置。
4. SST 实例化 16 个 Vanadis core、每核 cache/RoCC/GlobalMemory、4x5 mesh NoC、NodeOS/MMU、4 个 NUMA memory node。
5. 每个 core 运行 `test_noc_dma`，进入 runtime，再进入 `gemm_matmul_op.h` 的 HBM-backed tiled GEMM 路径。
6. kernel 通过 `remote.ld` 从 identity-window 触发 DMA，把 A/B tile 拉到本地 GM。
7. kernel 再通过 `mvm.gm2imat`、`mvm.gm2ivec`、`mvm`、`mvm.ovec2gm` 驱动计算阵列。
8. 累加后的 C tile 通过 `remote.st` 远端写回 data memory node 的 backing file。
9. 运行结束后，`unpack_c_from_hbm.py` 从 `hbm_out_node*.bin` 拼回最终 C 矩阵，并供 Python 校验。

## 19. 后续建议

如果后续继续做分析或开发，最建议沿下面三条线展开：

1. 按某个具体 task 画地址流：`a_base_mm -> local_mat -> array -> local_out -> c_base_mm`。
2. 结合 `stats_selfcom.txt`、latency csv、DMA stats csv 分析瓶颈到底在 NoC、DMA 还是 MVM compute。
3. 若要改功能，优先先判断应该改哪一层：
   - 数据布局改 `gen_hbm_init.py` / `pipeline_config.h`
   - kernel 调度改 `gemm_matmul_op.h`
   - 指令行为改 `roccAnalog.h`
   - DMA/identity-window 改 `globalmemory.cc`
   - 拓扑改 `ncores_selfcom_dma.py` / `cpu_builder.py` / `noc_builder.py`
