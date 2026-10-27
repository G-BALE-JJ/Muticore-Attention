# LeNet5 分层分工边界（A/B/C）

## 1. 分层架构与职责

| 层级 | 主要职责 | 关键文件 |
|---|---|---|
| 应用/模型层 | 输入、权重、语义参考 | `task/task lenet5/*`, `fronted/lenet_gemm_demo.py` |
| 算子编译后端 | 任务切分、布局映射、bin 生成 | `fronted/lenet_pipeline/prepare_real_lenet5_bins.py`, `fronted/lenet_pipeline/plan.py`, `artifacts_lenet/contracts/lenet_plan_v2.json` |
| 编排层 | HBM 初始化、编译、SST 启动、统计导出 | `run_noc_dma_pipeline.sh`, `gen_hbm_init.py` |
| 执行运行时层 | Vanadis 上执行算子与同步协议 | `small/lenet5/lenet_conv12.cpp`, `small/lenet5/conv1_ops.h`, `small/lenet5/conv2_ops.h`, `small/lenet5/fc1_ops.h`, `small/lenet5/fc23_ops.h` |
| 平台建模层 | CPU/RoCC/GM/NoC 参数与连线 | `ncores_selfcom_dma.py`, `cpu_builder.py` |
| 观测分析层 | 阶段延迟/统计/瓶颈报告 | `stats/extract_latency_by_stage_csv.py`, `artifacts_lenet/stats/*` |

---

## 2. A/B/C 边界定义

### A：模型真值 + 算子编译后端 Owner

- 负责层：应用/模型层 + 算子编译后端。
- 主要职责：
  - 以 ONNX 作为主线真值（权重/偏置/输入语义）。
  - 产出同名 bin（`conv1/conv2/fc1/fc2/fc3` 权重与 bias，及输入 A）。
  - 维护 `lenet_plan_v2.json` 的任务映射正确性（`task_id -> node/slot/m_tile/n_tile`）。
  - 提供 Python 语义参考与最小校验（shape/长度/文件完整性）。
- 交付物：
  - `data/real_lenet5/*.bin`
  - `artifacts_lenet/contracts/lenet_plan_v2.json`
  - ONNX 对齐说明（真值来源、版本、可复现实验参数）
- 不负责：
  - SST 调度性能问题。
  - C++ runtime 内核实现细节。
  - 平台连线与硬件参数调优。

### B：编排 + 执行运行时 Owner

- 负责层：编排层 + 执行运行时层。
- 主要职责：
  - 将 A 的 bin/plan 正确装载到 HBM（地址、长度、节点映射）。
  - 负责端到端流程：构建、启动、结束、失败回滚与日志落盘。
  - 保证各阶段执行协议正确：`conv1 -> conv2 -> fc1 -> fc2 -> fc3`。
  - 维护跨核/跨阶段同步语义（ready flag、规约顺序、阶段里程碑）。
- 交付物：
  - 可执行流程（`run_noc_dma_pipeline.sh`）
  - 可运行 runtime（二进制与关键日志）
  - 里程碑日志（阶段开始/完成/失败）
- 不负责：
  - ONNX 模型语义真实性定义。
  - NoC/CPU Builder 结构设计与平台级实验矩阵。

### C：平台建模 + 观测分析 Owner

- 负责层：平台建模层 + 观测分析层。
- 主要职责：
  - 维护 SST 平台参数、拓扑连线与建模一致性。
  - 输出标准化统计（总延迟、阶段延迟、DMA/NoC 指标）。
  - 对性能瓶颈进行归因，给出可复现实验对比。
  - 保证统计口径稳定（同版本脚本、同口径 CSV、严格阶段完整性检查）。
- 交付物：
  - 平台配置快照（核心参数、网络参数、memory 参数）
  - `latency_by_stage_breakdown.csv` 等统计文件
  - 性能分析结论与复现实验命令
- 不负责：
  - 上游模型权重正确性。
  - 运行时功能代码的业务逻辑实现。

---

## 3. 核心 ABI 契约与归属

| ABI | 契约内容 | Owner | 主要实现/消费方 |
|---|---|---|---|
| 指令 ABI | `mm2gm/gm2mm/remote_load/remote_store/run_mvm_stage` 行为与调用约束 | B | B 实现，C 观测验证 |
| 布局 ABI | `lenet5_layout.h` 的 `*_OFF` 地址、tensor 排布与对齐 | A（定义）+ B（落地） | A 产 bin，B/C 验证装载与访问 |
| 映射 ABI | `lenet_plan_v2.json` 的 task groups 与 stage tasks 映射 | A | B 执行，C 统计按此口径聚合 |
| 同步 ABI | ready flag、规约顺序（尤其 fc1 split-K） | B | B 实现，C 通过日志/统计核验 |

说明：

- 任何 ABI 变更都必须带版本化说明，并同步更新消费端脚本。
- 破坏性变更至少要包含：变更点、迁移步骤、回归命令、预期结果。

---

## 4. 交付验收（DoD）

### A 验收

- 默认 ONNX 路径可生成全套同名 bin。
- bin 尺寸、dtype、布局与 plan 一致。
- Python 参考链路可复现（至少给出 logits 对齐基线）。

### B 验收

- 单命令可从 bin 启动到仿真结束，关键阶段里程碑完整。
- 无 ABI 断裂（参数名、地址、长度、同步信号一致）。
- 失败时可从日志定位到阶段级别。

### C 验收

- 统计脚本对 plan 的阶段完整性检查通过。
- 关键 CSV 可稳定导出，口径一致。
- 可给出最少一条性能瓶颈归因结论并可复现。

---

## 5. 变更流程（建议）

1. A 先冻结 ABI 输入（bin + plan + 版本说明）。
2. B 对接并验证可运行性（功能正确优先）。
3. C 在固定版本上做性能采集与归因。
4. 若出现问题，按“先 ABI 再实现再平台”顺序回溯定位。
