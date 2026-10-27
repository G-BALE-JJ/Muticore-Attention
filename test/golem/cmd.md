# Golem Current Handoff

你在 `/data4/lishun/pkg/sst-elements/src/sst/elements/golem/tests` 工作，目标是继续把 golem 当前架构从 data-movement bound 推向 compute dominate。

先读这些上下文并严格遵守。

## 当前主路径和默认配置

- 当前主路径已从 `GOLEM_DIM / GOLEM_ARRAY_DIM` 基本解耦到：
  - `GOLEM_ARRAY_INPUT_SIZE`
  - `GOLEM_ARRAY_OUTPUT_SIZE`
  - `GOLEM_NUM_ARRAYS`
- 当前默认配置来自：
  - `tests/configs/default.env`
  - `tests/configs/10_core_gemm.env`
  - `tests/configs/20_dma.env`
  - `tests/configs/25_latency.env`
  - `tests/configs/30_network.env`
  - `tests/configs/40_debug_io.env`
  - `tests/configs/50_tensor_verify.env`
  - `tests/configs/60_run.env`
- 当前默认主路径稳定可跑通，参考成功 run：
  - `run_20260422_113448_329667`
  - 或更新成功 run，但务必先确认 log 中有 `Simulation is complete` 且 `VERIFY-C = PASS`
- 当前默认典型配置：
  - `GOLEM_ARRAY_INPUT_SIZE=64`
  - `GOLEM_ARRAY_OUTPUT_SIZE=64`
  - `GOLEM_NUM_ARRAYS=16`
  - `GOLEM_MATMUL_BLOCK_M=64`
  - `GOLEM_MATMUL_BLOCK_N=16`
  - `GOLEM_MATMUL_BLOCK_K=64`
  - `GOLEM_DMA_NODE_CREDITS=32`
  - `GOLEM_WCP_PREFETCH_WINDOWS=2`
  - `GOLEM_CTRL_OVERLAP_AB=0`
  - `GOLEM_WORKER_COMMAND_PROCESSOR_ENABLE=1`
  - `GOLEM_REQUEST_SCHEDULER_ENABLE=1`
  - `GOLEM_GROUP_MANAGER_ENABLE=1`

## 当前互斥 breakdown 口径

- `extract_latency_csv.py` 已改成互斥 breakdown：
  - `compute_active_time`
  - `prefetch_wait_time`
  - `writeback_wait_time`
  - `control_other_time`
- 稳定基线（如 `run_20260422_113448_329667`）大致是：
  - `total_cycles ≈ 24815.81`
  - `throughput ≈ 1352.14 ops/cycle`
  - `compute_active_time ≈ 2112`
  - `prefetch_wait_time ≈ 19545`
  - `writeback_wait_time ≈ 3153`
  - `control_other_time ≈ 5`
  - `compute ≈ 8.5%`
  - `prefetch_wait ≈ 78.8%`
  - `writeback_wait ≈ 12.7%`
- `execution_debug_summary.csv` 关键项：
  - `tile_ready_wait ≈ 19543`
  - `txn_wait ≈ 2`
  - `compute ≈ 2112`
  - `writeback_wait ≈ 3153`
  - `dma_total ≈ 22698`
- 当前真实瓶颈不是 control，不是 txn 边界，而是：
  - `ready -> compute_start` 的巨大排队时间
  - 即 tile 很早 ready，但严格顺序消费导致其长期排队

## 当前代码结构关键点

### 已完成工程修复

1. 主路径参数解耦：
   - `tests/configs/10_core_gemm.env`
   - `tests/architecture/cpu_builder.py`
   - `tests/run_noc_dma_pipeline.sh`
   - `tests/small/mvm_noc_int_array/pipeline_config.h`
   - `tests/small/mvm_noc_int_array/test_noc_dma.cpp`
   - `tests/small/mvm_noc_int_array/gemm_matmul_op.h`
   - `tests/small/mvm_noc_int_array/golem_matmul_runtime.cpp`
   - `tests/tools/gen_hbm_init.py`
   - `tests/tools/unpack_c_from_hbm.py`

2. builder 语义已统一成：
   - `num_cu = array_output_size`
   - `runtime input size = block_k`
   - `modeledComputeCycles = ceil(block_k / mac_per_cu_per_cycle) + pipeline_depth`

3. `WCP` 的明显 2-slot 残留 bug 已修：
   - `buffers_[2] -> buffers_[4]`
   - `activeComputeTileIndex_ / activeComputeSlotIndex_` 分离

4. 脚本问题已修：
   - `run_noc_dma_pipeline.sh` 有 trap，可在脚本退出/中断时杀掉独立 `setsid` 启动的 `sst`
   - `Simulation is complete` 后会进入第 4 阶段
   - NoC heatmap 默认关闭，由 `GOLEM_EXPORT_NOC_HEATMAPS` 控制
   - 注意：`run_noc_dma_pipeline.sh` 绝对不要并行跑

### 当前最关键的结构性限制

- `WCP` 运行时仍然把逻辑 block 当成 hardware tile 直接执行
- `loadTileToArrays()` 仍按整个 slot 数据整体装阵列
- 还没有真正实现 `m_step / k_step` micro-tiling
- 因此 `block_m / block_k` 的整数倍自由化还没真正落地

### 已加的调试

- `WCP TILE TRACE` 已经加到日志里，字段包括：
  - `submit`
  - `mat_done`
  - `vec_done`
  - `ready`
  - `compute_start`
  - `compute_done`
  - `retire`

## 编译边界（必须遵守）

参考：`tests/doc/compile_boundaries.md`

规则：
1. 如果改了 `globalmemory/globalmemory.h` 或 `.cc`，必须全量 clean rebuild：
   - `cd /data4/lishun/pkg/sst-elements`
   - `make clean`
   - `./configure --prefix=/data4/lishun/pkg/sst_install --with-dramsim3=/data4/lishun/pkg/DRAMsim3`
   - `make -j4`
   - `make install`
2. 如果只改 `rocc / requestscheduler / groupctrl / workercmdproc`，只重编 `golem`：
   - `cd /data4/lishun/pkg/sst-elements/src/sst/elements/golem`
   - `make -j4`
   - `make install`
3. 只改 `tests/configs / tests/stats / tests/tools / run_noc_dma_pipeline.sh`，不用重编库
4. 绝对不要并行跑 `run_noc_dma_pipeline.sh`

## 已试过但目前不要重复踩坑的方向

1. 直接 `ready-first consume`：
   - 曾试过“乱序计算 + 有序退休”
   - 结果：`Simulation is complete` 但 `VERIFY-C = FAIL`
   - 根因：overwrite/accumulate 语义没有正确绑定到逻辑 `k` 顺序
   - 当前不要继续在这版代码上叠改

2. 扫 `node credit / worker credit`：
   - 不是当前主问题
   - 当前主要瓶颈仍然是 ready tile 的排队等待

3. 直接试新硬件尺寸组合：
   - `32x32` / `32x128` 已试
   - 参数入口已经支持，但运行时还没真正支持
   - 当前根因不是 builder，而是 WCP 执行层仍把逻辑 block 当硬件 tile

## 当前最重要的技术结论

- 当前系统最大瓶颈是：
  - tile 很早 ready，但由于严格顺序消费，长期排队等着被算
- 所以 `prefetch_wait` 的本质已经不是纯 DMA latency，而是 `ready -> compute_start` 的队列等待
- 当前还没真正支持整数倍 block 自由化，因为 WCP 还没有实现 micro-tiling

## 下一步推荐工作

### 当前最应该做的事
正式做 `WCP` 内部的 micro-tiling：

- slot 只装一个 hardware micro-tile
- 一个逻辑 task 通过：
  - `for m_step in block_m / hw_output_size`
  - `for k_step in block_k / hw_input_size`
  逐步 accumulate 完成

### 建议推进顺序
1. 在 `WorkerTaskListHeader / WorkerWindowDescriptor` 中显式保留：
   - `hw_input_size`
   - `hw_output_size`
   - `block_m`
   - `block_k`
2. 放宽检查到“整数倍支持”：
   - `block_m % hw_output_size == 0`
   - `block_k % hw_input_size == 0`
3. `workercmdproc.h` 改成 micro-tiling 执行循环
4. 每一步只做小改动并回归：
   - `Simulation is complete`
   - `VERIFY-C = PASS`
   - `throughput`
   - `prefetch_wait`
   - `writeback_wait`
   - `compute_active`

### 当前不要做的事
- 不要再并行回归
- 不要再基于旧失败 run 做判断
- 不要在 debug 逻辑里引入会改变语义的状态机改动
- 不要轻易再碰 `globalmemory`

## 新 session 开始时必须先做的事

1. 先确认最新成功 baseline：
   - `Simulation is complete`
   - `VERIFY-C = PASS`
2. 读取：
   - `execution_summary.csv`
   - `execution_debug_summary.csv`
   - `WCP TILE TRACE`
3. 说明本次准备改哪些文件，以及是否触发全量重编边界
4. 每次只做一小步，然后回归
