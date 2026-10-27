# 团队分工一页版（汇报用）

## 目标

- 面向多模型（LeNet、GCN）建立稳定交付链路：前端可扩展、执行可复用、验证可复现。

---

## 三人分工总表

| 角色 | 负责范围 | 关键交付 | 成功标准 |
|---|---|---|---|
| PhD（系统Owner） | `run_noc_dma_pipeline.sh`、`architecture/`、`verify/`、`stats/` | 一键流程、平台配置、统一验证口径、最终报告 | 任意模型可复现跑通；结论可追溯 |
| Master A（前端Owner） | `fronted/` | 模型解析、lowering、plan/bin、reference | 产物完整且与模型真值一致 |
| Master B（RuntimeOwner） | `small/` | C++执行链路、同步协议、阶段日志 | runtime 正确执行并可定位问题 |

---

## 协作流程（固定）

```text
A交付(bin + plan + reference)
      -> B对接(runtime执行 + 阶段日志)
      -> PhD集成(编排 + 平台 + verify/stats)
      -> 发布结论(PASS/FAIL + 性能归因)
```

问题回溯顺序：A（契约/数据）-> B（执行/同步）-> PhD（平台/口径）。

---

## LeNet 例子（当前实践）

- A：生成 ONNX 主线 bin 与 plan。
- B：执行 `conv1 -> conv2 -> fc1 -> fc2 -> fc3`。
- PhD：统一跑仿真并输出阶段误差、延迟、瓶颈。

结论：当前链路已可稳定用于 LeNet，并可平滑扩展到 GCN。
