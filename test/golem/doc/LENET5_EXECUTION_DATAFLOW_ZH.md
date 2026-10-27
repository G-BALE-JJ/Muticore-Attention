# LeNet5 单次仿真执行图（中文说明）

本文档描述当前已跑通版本的完整数据流，覆盖：阶段、输入/输出、地址、任务切分、核心参与、同步点与关键指令。

## 1. 总体执行链

单次仿真内按如下顺序串行执行：

1. `conv1 GEMM + ReLU + Pool1`
2. `conv2_repack (pool1 -> conv2 A)`
3. `conv2 GEMM + ReLU + Pool2`
4. `fc1 split-K (4 task) + 树规约 + bias/relu`
5. `fc2 (core0 单核阵列, 直接 bias/relu)`
6. `fc3 (core0 单核阵列, 直接 bias)`

入口可执行：`tests/small/lenet5/lenet_conv12.cpp`

---

## 2. 任务切分总表

| 阶段 | 逻辑维度 | 执行维度 | block | task 数 | 核心参与 |
|---|---:|---:|---:|---:|---|
| conv1 GEMM | M=576, N=6, K=25 | M=768, N=6, K=64 | 64/6/64 | 12 | core0..11（每 task 一核） |
| conv2 GEMM | M=64, N=16, K=150 | M=256, N=16, K=192 | 64/16/64 | 4 | core0..3（每 task 一核） |
| fc1 split-K | K=256 -> 4x64 | - | - | 4 | core0..3 |
| fc2 | 120->84 | 128->128(分块) | 64x64 阵列块 | 1（单核） | core0 |
| fc3 | 84->10 | 128->128(分块) | 64x64 阵列块 | 1（单核） | core0 |

说明：
- conv1/conv2 的 task 到 node 映射由 `lenet_plan_v1.json` 的 `stages.*.tasks` 给出。
- fc1 复用 conv2 的 4 个 task 映射（task0..3）。

---

## 3. 关键地址布局（HBM Offsets）

定义来源：`tests/small/lenet5/lenet5_layout.h`

- `POOL1_OFF = 0x01006000`
- `POOL1_READY_OFF = 0x01007000`
- `POOL2_OFF = 0x0100A000`

- `CONV2_BPACK_OFF = 0x01240000`
- `CONV2_BIAS_OFF  = 0x01248000`

- `FC1_WSLICE_OFF  = 0x01250000`
- `FC1_BIAS_OFF    = 0x01270000`
- `FC1_PARTIAL_OFF = 0x01271000`
- `FC1_READY_OFF   = 0x01272000`
- `FC1_OUT_OFF     = 0x01273000`

- `FC2_WPACK_OFF   = 0x01274000`
- `FC2_BIAS_OFF    = 0x01285000`
- `FC2_OUT_OFF     = 0x01286000`

- `FC3_WPACK_OFF   = 0x01287000`
- `FC3_BIAS_OFF    = 0x01298000`
- `FC3_OUT_OFF     = 0x01299000`

---

## 4. 分阶段执行图（阶段/输入/输出/地址/task/核心/同步）

### 阶段 A：conv1 GEMM + ReLU + Pool1

- 输入
  - `A`: conv1 banded im2col（执行 `768x64`）
  - `B`: `64x6`（权重）
  - `bias`: 6
- task/核心
  - 12 task（task0..11），core0..11 参与
- 输出
  - `pool1`：`6x12x12`，按 band 分布到 node1/2/3 的 `POOL1_OFF`
  - ready 标志：`POOL1_READY_OFF + band*8` 写 1
- 同步点
  - conv2_repack 读取前按 band 等待 `POOL1_READY_OFF`

### 阶段 B：conv2_repack（Vanadis 内）

- 输入
  - 分布式 `pool1`（`POOL1_OFF`）
- task/核心
  - 4 task（对应 conv2 的 4 个 m_tile band），core0..3
- 处理
  - 每 task 缓存本 band 所需 `pool1` 行
  - 生成 `A_conv2`（执行 `256x192`，K=150 pad 到 192）
- 输出
  - 写回 conv2 GEMM 的 A 区（按 task/k_tile 布局）
- 同步点
  - 依赖 `POOL1_READY_OFF`

### 阶段 C：conv2 GEMM + ReLU + Pool2

- 输入
  - `A_conv2`: `256x192`
  - `W_conv2`: 预置 `CONV2_BPACK_OFF`
  - `bias2`: `CONV2_BIAS_OFF`
- task/核心
  - 4 task，core0..3
- 输出
  - `pool2`：`16x4x4`，分布写入 `POOL2_OFF`

### 阶段 D：fc1 split-K（4 task）+ 树规约

- 输入
  - `pool2` 分布数据（每 task 对应 64 维 slice）
  - `FC1_WSLICE_OFF`（每 task 两个 64x64 chunk）
  - `FC1_BIAS_OFF`
- task/核心
  - task0..3，core0..3
- 处理
  1. 每 task 计算 `y_partial[120]`
  2. 树规约：`t0+=t1`、`t2+=t3`、`t0+=t2`
  3. 仅 t0 做 `bias + relu`
- 输出
  - `FC1_OUT_OFF`（120，位于 task0 对应 node）
  - 中间：`FC1_PARTIAL_OFF`、`FC1_READY_OFF`
- 同步点
  - `FC1_READY_OFF` 用于 partial/l1 规约同步

### 阶段 E：fc2（core0 单核阵列）

- 输入
  - `FC1_OUT_OFF`（120，pad 到 128）
  - `FC2_WPACK_OFF`（按 `(out_chunk, in_chunk)` 的 4 个 64x64 矩阵）
  - `FC2_BIAS_OFF`（84）
- task/核心
  - 无 task 切分，仅 core0
- 处理
  - 对每个 `out_chunk`，累加两个 `in_chunk` 的 MVM 输出
  - 层末一次 `bias + relu`
- 输出
  - `FC2_OUT_OFF`（84）

### 阶段 F：fc3（core0 单核阵列）

- 输入
  - `FC2_OUT_OFF`（84，pad 到 128）
  - `FC3_WPACK_OFF`（4 个 64x64）
  - `FC3_BIAS_OFF`（10）
- task/核心
  - 仅 core0
- 处理
  - 与 fc2 同构：按 `(out_chunk, in_chunk)` 累加
  - 层末一次 bias（一般不 relu，输出 logits）
- 输出
  - `FC3_OUT_OFF`（10）

---

## 5. 指令级数据推动（关键原语）

常用原语定义在：`tests/small/lenet5/operators.h`, `tests/small/lenet5/ex_instr.h`

- 远端读（并等待完成）
  - `dma_remote_load_to_gm(core, remote_addr, local_gm_addr, bytes)`
  - 内部流程：`set_len -> remote_load -> 等待读完成 flag`
- 远端写
  - `remote_store(local_gm_addr, remote_addr)`
- MM/GM 拷贝
  - `mm2gm(mm_ptr, gm_addr)`
  - `gm2mm(mm_ptr, gm_addr)`
- 标志同步
  - `remote_write_u64(core, value, remote_addr)`
- 阵列计算
  - `run_mvm_stage(mat_gm, vec_gm, out_gm)`
  - 等价序列：`inputmatrixload -> inputvectorload -> mvm_compute -> outputvectorstore`

---

## 6. 验证路径

Python 入口：`tests/lenet_gemm_demo.py`

- 读取并验证：
  - `pool1`（按 conv1 task map 分节点拼回）
  - `pool2`（按 conv2 task map 分节点拼回）
  - `fc1/fc2/fc3`（从 task0 对应 node 的 `FC*_OUT_OFF` 读取）
- 参考实现：
  - `conv1_ref_direct`
  - `conv2_ref_direct`
  - `fc1_ref_direct`
  - `fc2_ref_direct`
  - `fc3_ref_direct`

---

## 7. 当前稳定版本结论

- 单次仿真链路：`conv1 -> conv2 -> fc1 -> fc2 -> fc3`
- 已通过数值校验（max abs diff 全为 0）
- 可作为后续优化（性能/带宽/并行度）基线版本。
