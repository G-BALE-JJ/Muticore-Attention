# GOLEM Onboarding Context (精简归档)

## 1) 30 秒上手
- 统一入口：`tests/run_noc_dma_pipeline.sh`
- 一条链路：`生成HBM -> 编译RISC-V程序 -> 跑SST拓扑 -> 导出统计/图`
- 主 workload：`tests/small/mvm_noc_int_array/test_noc_dma.cpp`
- 主架构脚本：
  - 无控制链路：`tests/architecture/ncores_selfcom_dma.py`
  - 控制链路：`tests/architecture/ncores_selfcom_dma_ctrl.py`

## 2) 核心调用链（新对话最重要）
1. `run_noc_dma_pipeline.sh`
2. `tools/gen_hbm_init.py`
3. `small/mvm_noc_int_array/Makefile` 编译 `riscv64/test_noc_dma`
4. `sst architecture/ncores_selfcom_dma(_ctrl).py`
5. 程序侧：`test_noc_dma.cpp -> golem_matmul_runtime.cpp -> gemm_matmul_op*.h`
6. 组件侧：`rocc/roccAnalog.h -> globalmemory/globalmemory.cc`
7. 后处理：`stats/extract_*.py` + `verify/*.py`

## 3) 组件职责一览
- `rocc/roccAnalog.h`：RoCC 指令解释与分发（mvm/remote/mm2gm/gm2mm/reg2gm/gm2reg）
- `array/mvmComputeArray.h`：阵列MVM计算、输出模式（overwrite/accumulate）、可选dump
- `globalmemory/globalmemory.cc`：本地GM存储、NoC收发、DMA分块/重试/完成标志
- `groupctrl/groupctrl.cc`：REQUEST/GRANT/DONE/FINISHED/GROUP_DONE 控制平面调度
- `architecture/cpu_builder.py`：Vanadis+Cache+RoCC+GlobalMemory 组装
- `architecture/noc_builder.py`：Merlin mesh 拓扑构建

## 4) 关键协议与约束
- DMA flag 区占每核 GM 窗口末尾 32B（读/写各自 seq/flag）
- `identityWindowBase` 以上地址走 DMA 主存路径
- phase-1 约束：
  - `block_M == GOLEM_DIM`
  - `block_K == GOLEM_DIM`
  - `block_N <= GOLEM_DIM`
  - `M/N/K` 必须可整除 `block_M/N/K`
- 控制链路模式下 worker/manager 槽位数必须匹配（当前 manager 固定 4 个 worker slot）

## 5) 常用目录（按功能归档）
- `tests/architecture`：SST系统拓扑
- `tests/configs`：分层预设（10/20/30/40/50/60）
- `tests/small/mvm_noc_int_array`：主GEMM workload
- `tests/small/lenet5`：LeNet5流水线 workload
- `tests/tools`：HBM生成、样本生成、C反解包
- `tests/stats`：统计抽取/可视化
- `tests/verify`：数值正确性校验
- `tests/data`：输入与真实LeNet5权重/中间数据
- `tests/artifacts*`：运行输出（log/stdout/hbm/stats）

## 6) 推荐运行命令
```bash
cd /data4/lishun/pkg/sst-elements/src/sst/elements/golem/tests
./run_noc_dma_pipeline.sh --dim 16 --gemm-m 256 --gemm-n 16 --gemm-k 256 --dtype fp32
```

控制链路模式（通常由默认配置自动开启）：
```bash
./run_noc_dma_pipeline.sh --dim 64 --dma-overlap 0
```

## 7) 快速排障（高频）
- 卡住看点：
  - 是否反复等待 GM 尾部 flag
  - 是否只有 `RemoteLoad issued` 没有 `DMA_READ_COMPLETE`
  - `GOLEM_MEM_NODE_SIZE` 与 `physMemSize/numMemNodes` 是否一致
  - `GOLEM_NUM_ARRAYS` 是否满足控制模式下 `>= block_n`
- 关键输出：
  - 主日志：`artifacts/logs/*.log`
  - 分片stdout：`artifacts/stdout/**/stdout-*`
  - 汇总：`artifacts/stats/**/{execution_summary.csv,dma_summary.csv,noc_summary.csv,memory_summary.csv}`

## 8) 新对话可直接粘贴的最小上下文
```text
项目入口是 tests/run_noc_dma_pipeline.sh。
请按这条链路理解：gen_hbm_init.py -> 编译 small/mvm_noc_int_array/test_noc_dma ->
sst architecture/ncores_selfcom_dma_ctrl.py(或ncores_selfcom_dma.py) ->
rocc/roccAnalog.h + globalmemory/globalmemory.cc -> stats/extract_*.py。
重点关注 DMA flag 尾部协议、identity window DMA 路由、groupctrl 的 REQUEST/GRANT 调度。
```

## 9) 备注
- `golem/globalmemory.h` 与 `golem/globalmemory.cc` 为空占位；有效实现在 `golem/globalmemory/` 子目录。
- `tests/tilelang_gemm_ex.py` 当前语法不完整，属于草稿/实验脚本，不是主链路。
