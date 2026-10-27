# 团队分工说明（面向多模型：LeNet / GCN）

这份文档回答三个问题：

1. 三个人分别做什么；
2. 怎么协作最快、最少互相踩；
3. 用 LeNet 举例说明“每层具体产出”。

---

## 1. 一句话总览

- 你（PhD）负责**系统层**：编排、平台、验证口径与最终结论。
- Master A 负责**模型前端层**：模型解析、lowering、bin 与 plan。
- Master B 负责**执行层**：C++ runtime 与算子执行正确性。

目标是：同一套系统既能跑 LeNet，也能继续扩展到 GCN。

---

## 2. 按目录划分 Owner（最清晰）

| 负责人 | 目录/层级 | 主要职责 | 不负责 |
|---|---|---|---|
| PhD（你） | `run_noc_dma_pipeline.sh`、`architecture/`、`verify/`、`stats/` | 流程编排、SST 平台建模、统一验证口径、性能/正确性最终报告 | 模型权重解析细节、C++算子内部实现 |
| Master A | `fronted/` | 模型插件、ONNX 读取、lowering、plan/bin 生成、Python reference | SST 参数/拓扑、C++ runtime 逻辑 |
| Master B | `small/`（尤其 `small/lenet5`） | C++ runtime 执行、同步协议、里程碑日志、算子正确性 | 模型前端转换、平台连线/DRAM 参数 |

---

## 3. 每个人“必须交付”的东西（DoD）

### 3.1 PhD（系统 Owner）

- 一键可跑：`run_noc_dma_pipeline.sh` 在主线配置下可执行。
- 平台可复现：`architecture/` 参数可追踪（版本化）。
- 验证可复现：`verify/` 与 `stats/` 输出稳定、口径一致。
- 给出最终结论：PASS/FAIL + 关键误差 + 性能摘要。

### 3.2 Master A（前端 Owner）

- 给出模型输入产物：`*.bin`（输入、权重、bias）。
- 给出执行计划：`plan`（后续统一到 `model_plan_v1.json`）。
- 给出参考结果：Python reference（可用于逐阶段对拍）。
- 保证来源可追溯：权重来源（例如 ONNX）明确。

### 3.3 Master B（Runtime Owner）

- C++ runtime 能按 plan 正确执行各 stage。
- 关键同步协议正确（ready flag / reduce 顺序）。
- 里程碑日志完整（便于 PhD 侧分析）。
- 通过最小回归（至少 1 个端到端 case + 1 个算子级 case）。

---

## 4. 协作流程（固定顺序，减少返工）

1. Master A 先交付：`bin + plan + reference`。
2. Master B 对接执行：保证 runtime 跑通并产生日志。
3. PhD 集成：统一编排运行、验证、统计导出。
4. 出结论：通过/失败与根因归属。

问题回溯顺序固定：

- 先查前端契约（A） -> 再查执行实现（B） -> 最后查平台与口径（PhD）。

---

## 5. 用 LeNet 举例（最直观）

以 `fronted/lenet_gemm_demo.py` 这条链路为例：

### A（Master A）做的事

- 从 ONNX 读取权重；
- 生成 LeNet 所需 bin（如 `b_conv1_kn_64x6.bin` 等）；
- 生成/维护 plan；
- 生成 Python reference（conv1/conv2/fc1/fc2/fc3）。

### B（Master B）做的事

- 在 `small/lenet5` 中执行 `conv1 -> conv2 -> fc1 -> fc2 -> fc3`；
- 保证写回与同步协议正确；
- 输出可解析的阶段日志。

### PhD 做的事

- 通过 `run_noc_dma_pipeline.sh` 串起装载、编译、SST、统计；
- 用 `verify/` + `stats/` 输出误差、阶段延迟、瓶颈；
- 定义“是否通过”的统一标准。

---

## 6. 为什么这样分对 GCN 更有利

- GCN 到来时，A 只需新增 GCN 插件，不必改平台与 runtime 主框架。
- B 只需补 GCN 相关算子（如 SpMM）执行，不必改 ONNX/前端。
- PhD 复用现有编排与验证口径，保证跨模型可比性。

这就是“最少重叠技术栈”的核心：

- A 主要 Python 前端；
- B 主要 C++ runtime；
- PhD 主要 Bash SST 统计分析。

---

## 7. 日常协作约定（建议）

- 互相讨论协同设计。
- 跨层改动先提接口变更说明，再改代码。
- 定期集成，避免“各自都对，合起来不对”。
