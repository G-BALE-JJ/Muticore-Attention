# LeNet5 技术附录（一页版）

## 1. 分层架构与职责

| 层级 | 主要职责 | 关键文件 |
|---|---|---|
| 应用/模型层 | 输入、权重、语义参考 | `task/task lenet5/*`, `lenet_gemm_demo.py` |
| 算子编译后端 | 任务切分、布局映射、bin生成 | `fronted/lenet_pipeline/prepare_real_lenet5_bins.py`, `fronted/lenet_pipeline/plan.py`, `lenet_plan_v2.json` |
| 编排层 | HBM初始化、编译、SST启动、统计导出 | `run_noc_dma_pipeline.sh`, `gen_hbm_init.py` |
| 执行运行时层 | Vanadis上执行算子与同步协议 | `lenet_conv12.cpp`, `conv1_ops.h`, `conv2_ops.h`, `fc1_ops.h`, `fc23_ops.h` |
| 平台建模层 | CPU/RoCC/GM/NoC参数与连线 | `ncores_selfcom_dma.py`, `cpu_builder.py` |
| 观测分析层 | 阶段延迟/统计/瓶颈报告 | `extract_latency_by_stage_csv.py`, `stats/*` |

## 2. 端到端执行链（单次仿真）

`conv1_gemm -> conv1_relu -> pool1 -> conv2_im2col -> conv2_gemm -> conv2_relu -> pool2 -> fc1 -> fc2 -> fc3`

说明：
- `conv1_im2col` 当前采用固定值注入（默认 `1.423ms`）用于统一报表。
- 其余阶段由里程碑日志 `start/done` 周期差计算得到。

## 3. 核心 ABI 契约

- **指令 ABI**：`mm2gm/gm2mm/remote_load/remote_store/run_mvm_stage`
- **布局 ABI**：`lenet5_layout.h` 中各 `*_OFF` 地址与 tensor 排布
- **映射 ABI**：`lenet_plan_v2.json` 的 `task_groups` 与 `stages[*].tasks`（`task_id -> node/slot/m_tile/n_tile`）
- **同步 ABI**：ready flag 与规约顺序（尤其 fc1 split-K）

### Plan v2 关键字段（对接必读）

- `version`: `lenet_plan_v2`
- `memory`: 节点拓扑与 data nodes
- `layouts`: 各阶段输出张量 offset/shape
- `stage_flow`: 全阶段执行顺序与阶段类型（含 `conv1_im2col` 固定注入）
- `stages`: gemm/repack 等参数与任务明细
- `task_groups`: 执行主体分组（`conv1_gemm_tasks`/`conv2_gemm_tasks`/`core0`/`host`）

说明：
- 当前统计脚本 `extract_latency_by_stage_csv.py` 已支持 `--plan-file` 与 `--strict-stage-check`，会按 `stage_flow` 校验阶段完整性。

## 4. 当前任务边界（技术视角）

- A（编译后端）：模型解析、bin打包、plan版本化
- B（执行层）：算子实现、同步协议、里程碑打点
- C（平台层）：SST参数、稳定性、日志噪声控制
- D（分析层）：数值门禁、阶段延迟、瓶颈归因

## 5. 当前瓶颈与优化入口

- 瓶颈1：`conv2_im2col`（分布式重排与远程访存开销）
- 瓶颈2：`conv2_gemm`

优先优化项：
1. `conv2_im2col` 访存批量化与缓存化（减少碎片 remote load/store）
2. 并行阶段“全核完成后打点”，提升阶段延迟口径严谨性
3. ONNX 真值锚定，完成“数值一致 + 标签一致”双闭环

## 6. 结果口径说明（对外统一）

- `PASS`：硬件路径输出与本地参考计算一致（max_abs_diff 通过阈值）。
- `标签正确`：需额外对齐 ONNX 真值（当前为独立验收项）。
- 延迟换算：按 `VANADIS_CPU_CLOCK` 自动换算 cycles -> ms。
