# 73k GEMM 与当前投影的可比性核查

> **架构纠正（2026-10-10）：** 本文的历史投影/E2E 测量或重叠估计使用了 manager 本地 arrays，不符合「manager 只控制、16 个 worker 拥有阵列」的目标拓扑，不能作为该架构的性能证据。新的 worker 执行基准见 [WORKER_PROJECTION_ARCHITECTURE_FIX.md](WORKER_PROJECTION_ARCHITECTURE_FIX.md)。

对照用户提供的 `/data/jjgong/RISC-V-CIM-Manycore-SST/artifacts/gemm_256_2048_3072_functional/stats/overlap0/run_20261010_192123_357378`。
这是既有 artifacts 与源码核查，没有重新运行 SST 或修改执行路径。

## 统计和工作量

该 run 的 `execution_summary.csv`：worker 平均 73,090.666667 cycles，
最大 75,895，descriptor-to-system-end 延迟 77,882。仅 core 4..9 六个
worker 有 GEMM 执行记录，每个 compute 为 67,584 cycles。一个全局
256×2048×3072 GEMM 分成六个 macro tasks，每 worker 处理一个 4×8
输出 tile macro（32 个 64×64 输出 tiles）。每核 32×32=1024 次 launch。

Attention S1024 的投影则由四个 manager 各自处理 256 行，即全局
1024×2048×3072 GEMM 等价工作量。每个 manager 用自己的 64 arrays，
执行 4×48×32=6144 次 launch；不是将一个 256 行块分给六个核。

两边都是 array/CPU 1GHz、64 arrays/core、64 CU/array、MAC/CU/cycle=1、
pipeline depth=2，因此单次计算模型为 66 cycles：

```
GEMM 每个 worker：1024 × 66 = 67,584
投影每个 manager：6144 × 66 = 405,504
投影 compute/launch 前台统计：417,792
```

6×67,584=405,504。投影另有 12,288 cycles 的 launch/controller residence
差值。不能将它与未扣除 Norm 的 producer 910,789 cycles直接做算法加速比。
投影 manager 0 独占统计为 910,155；其中 Norm 相关 136,186，剩余
773,969 包含 GEMM 计算、搬运、派发和结果写回。系统 span 边界不同。

## 搬运计时路径不同

GEMM run_command 指向另一仓库的 install library 和 build 配置。对应
build/sst-elements/src/sst/elements/golem/workercmdproc/workercmdproc.h：

- `loadTilePayload` 使用同步 `rd_from_globalmem` 获取 GM 内容（约 1846 行）。
- `loadActiveMicroTileToArrays` 直接 `setMatrixItem`/`setVectorItem`（约 1871/1881 行）。
- `captureArrayOutput` 直接 `getOutputVector`（约 2011 行）。

这些操作没有走本项目投影使用的显式 timed local GM、matrix programming、
input scatter、output gather/restore 异步端口路径。HBM/DMA 和 C-buffer
并非都没有计时；具体差异是上述内部搬运接口。当前另一仓库 build 源码没有
历史 binary/source hash 证据，所以源码解释以当前对应 build 为依据，不声称
已验证历史共享库的逐行对应关系。

## 正确性核查

原 `full_output_verification.txt` 报告 786,432 元素中 16,384 不匹配，
max abs error=954。读取 artifacts 原始 FP32 A/B/C，用独立 NumPy A@B
复查得到同样结果；错误集中在 row 0..63、64-column tiles 8/16/24/40，
每 tile 4096 个错误。该结果不能作为已通过数值验证的性能基准，但不因此
否定其六核任务分配和 compute 工作量证据。根因尚未定位，本轮不修改其他仓库。

## 下一步比较边界

先对齐总工作量、实际参与核数、FP16/FP32、GM/array-buffer计时和测量起止；
随后分别评估 generic GEMM 的 macro reuse/输出列映射与投影的权重广播/输入行
映射。复用调度必须保留真实搬运计时，并通过数值验证。使用现有闲置 worker
协助投影可以作为固定拓扑调度实验，但不能只让一个块占用六核，又忽略另外
三个块的资源占用、任务交接及 Attention 阶段。
