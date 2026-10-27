# Master A 角色说明（模型前端 Owner）

适用范围：LeNet 当前主线 + 后续 GCN 插件。

---

## 1. 你负责什么

- 负责目录：`fronted/`。
- 负责目标：把模型语义（ONNX/数据）变成系统可执行产物（plan + bin + reference）。
- 你是“模型到执行计划”的 owner（compiler-like lowering）。

---

## 2. 你不负责什么

- 不负责 SST 平台参数与连线。
- 不负责 C++ runtime 算子实现。

---

## 3. 你的输入与输出

- 输入
  - 模型文件（例如 `lenet5.onnx`）
  - 输入样本（例如 `image*.bin`）
- 输出
  - 执行产物：`*.bin`
  - 计划产物：`plan`（后续统一到 `model_plan_v1.json`）
  - 对拍基线：Python reference 输出

---

## 4. 你每天做什么（建议）

1. 先保证 ONNX 主线可复现（权重来源明确）。
2. 生成 bin + plan，并输出尺寸/校验信息。
3. 产出 reference 结果给系统 owner 对拍。
4. 改动后做最小回归（至少 1 个样本端到端）。

---

## 5. 你的质量标准（DoD）

- 同一输入重复生成的 bin 一致。
- plan/task 映射完整，无缺失 stage。
- reference 结果可被 verify 直接消费。
- 产物命名与路径稳定，不随临时实验漂移。

---

## 6. LeNet 例子（你如何工作）

- 从 ONNX 读取 `conv/fc` 权重。
- 生成 `b_conv1_kn_64x6.bin`、`conv2_bpack`、`fc*_wpack` 等产物。
- 写出 `lenet plan`（任务到 node/slot 映射）。
- 提供 `conv1/conv2/fc1/fc2/fc3` reference，用于 HBM 结果对拍。

---

