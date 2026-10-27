# LeNet5 执行链路汇报简版（1页结构图 + 1页表格）

## 第1页：结构图（单次仿真端到端）

```text
单次仿真（lenet_conv12）

输入图像(28x28)
   |
   v
[Conv1 GEMM] M=768,N=6,K=64, block=64/6/64, task=12 (core0..11)
   |
   +--> task侧 ReLU + Pool1
   |        输出: pool1(6x12x12) -> POOL1_OFF (分布到node1/2/3)
   |        同步: POOL1_READY_OFF
   v
[Conv2 Repack] task=4 (core0..3)
   读取 pool1(分布) -> 生成 A_conv2(256x192)
   |
   v
[Conv2 GEMM] M=256,N=16,K=192, block=64/16/64, task=4 (core0..3)
   |
   +--> task侧 ReLU + Pool2
   |        输出: pool2(16x4x4) -> POOL2_OFF (分布)
   v
[FC1 split-K] task=4 (core0..3)
   每task计算 partial(120) -> 树规约(t0+=t1, t2+=t3, t0+=t2)
   仅t0做 bias+ReLU
   输出: FC1_OUT_OFF (120)
   |
   v
[FC2 单核阵列] core0
   输入120(pad128), 4个64x64块累加
   层末 bias+ReLU
   输出: FC2_OUT_OFF (84)
   |
   v
[FC3 单核阵列] core0
   输入84(pad128), 4个64x64块累加
   层末 bias（logits）
   输出: FC3_OUT_OFF (10)
   |
   v
Python 读回验证（conv1/conv2/fc1/fc2/fc3）
```

---

## 第2页：执行总表（阶段/输入/输出/地址/task/核心/同步）

| 阶段 | 输入 | 输出 | 关键地址 | task切分 | 核心参与 | 同步点 |
|---|---|---|---|---|---|---|
| Conv1 GEMM + ReLU/Pool1 | A(768x64), B(64x6), bias(6) | pool1(6x12x12) | `POOL1_OFF`, `POOL1_READY_OFF` | 12 task（m_tile=12,n_tile=1） | core0..11 | 每band写 ready=1 |
| Conv2 Repack | pool1(分布) | A_conv2(256x192) | 读 `POOL1_OFF` | 4 task | core0..3 | 先等 `POOL1_READY_OFF` |
| Conv2 GEMM + ReLU/Pool2 | A(256x192), W(192x16), bias(16) | pool2(16x4x4) | `CONV2_BPACK_OFF`, `CONV2_BIAS_OFF`, `POOL2_OFF` | 4 task | core0..3 | task内后处理后写回 |
| FC1 split-K + 树规约 | pool2分片 + fc1分片权重 | fc1_out(120) | `FC1_WSLICE_OFF`, `FC1_PARTIAL_OFF`, `FC1_READY_OFF`, `FC1_BIAS_OFF`, `FC1_OUT_OFF` | 4 task（每task K=64） | core0..3 | `t0+=t1`,`t2+=t3`,`t0+=t2` |
| FC2 单核阵列 | fc1_out(120,pad128) + fc2权重 + bias | fc2_out(84) | `FC2_WPACK_OFF`, `FC2_BIAS_OFF`, `FC2_OUT_OFF` | 不切分（单核） | core0 | 无跨核同步 |
| FC3 单核阵列 | fc2_out(84,pad128) + fc3权重 + bias | fc3_out(10) | `FC3_WPACK_OFF`, `FC3_BIAS_OFF`, `FC3_OUT_OFF` | 不切分（单核） | core0 | 无跨核同步 |

### 关键指令（汇报关键词）

- 数据远端读取：`dma_remote_load_to_gm`（`remote_load + 完成等待`）
- 数据远端写回：`remote_store`
- MM/GM搬运：`mm2gm`, `gm2mm`
- 标志同步：`remote_write_u64`
- 阵列计算：`run_mvm_stage`（`inputmatrixload -> inputvectorload -> mvm_compute -> outputvectorstore`）

### 当前状态（用于汇报结论）

- 单次仿真端到端跑通
- 验证通过：conv1 / conv2 / fc1 / fc2 / fc3 全部 `max_abs_diff=0`
