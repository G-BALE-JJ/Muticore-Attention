# PhD 角色说明（系统 Owner）

适用范围：LeNet 当前主线 + 后续 GCN/更多模型。

---

## 1. 你负责什么

- 负责目录：`run_noc_dma_pipeline.sh`、`architecture/`、`verify/`、`stats/`。
- 负责目标：把“前端产物 + runtime 执行”变成稳定、可复现、可解释的系统结果。
- 你是最终集成 owner：谁都可以提改动，但跨层接口由你拍板。

---

## 2. 你不负责什么

- 不负责模型权重解析细节（这是前端 Owner 的工作）。
- 不负责 C++ 算子内部实现（这是 runtime Owner 的工作）。

---

## 3. 你的输入与输出

- 输入（来自 Master A）
  - `bin` 文件（输入/权重/bias）
  - `plan` 文件（任务映射）
  - reference 基线（用于对拍）
- 输入（来自 Master B）
  - runtime 可执行与阶段日志
- 输出（给团队）
  - 端到端 PASS/FAIL
  - 阶段误差、阶段延迟、DMA/NoC 统计
  - 问题归因结论与下一步动作

---

## 4. 你每天做什么（建议）

1. 先拉最新前端产物和 runtime 分支。
2. 跑统一入口，确认是否可复现。
3. 若失败，按固定顺序定位：A（契约/数据）-> B（执行/同步）-> 平台口径。
4. 出当日结论（问题在哪层、谁修、怎么验）。

---

## 5. 你维护的“硬接口”

- `GOLEM_*` 环境变量契约。
- HBM layout 与 stage log 解析口径。
- `verify/` 与 `stats/` 口径一致性。

原则：没有接口说明的跨层改动不合并。

---

## 6. LeNet 例子（你如何工作）

- A 产出 ONNX 主线 bin/plan。
- B 提供 `conv1->conv2->fc1->fc2->fc3` runtime 日志。
- 你统一跑 `run_noc_dma_pipeline.sh`，检查：
  - 数值是否对齐 reference；
  - 阶段统计是否完整；
  - dramsim3、NoC 热图是否可导出。
- 最终给项目状态：可发布 / 阻塞及原因。

---

## 7. 面向 GCN 的额外关注

- 优先保障“统一接口”，不要为单模型硬编码。
- 新增 op（如 SpMM）时，先定验证口径再上性能。
- 保证 LeNet 与 GCN 可共存在同一编排入口。
