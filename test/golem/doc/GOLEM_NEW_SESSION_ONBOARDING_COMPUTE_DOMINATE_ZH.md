# GOLEM 新 Session 上手文档（架构/瓶颈/Compute-Dominate）

适用目录：`/data4/lishun/pkg/sst-elements/src/sst/elements/golem/tests`

文档目的：让新 session 能在 10 分钟内掌握项目结构、关键代码路径、编译边界、当前瓶颈结论与优化方向，并直接接手下一步实验。

---

## 0) 一句话结论（必须先记住）

当前系统主瓶颈是 **返回路径（return path）**，不是 manager 调度，也不是 DRAM 平均服务时延。

网络热点主要由 **OS 侧路径汇聚** 引发（当前观测集中在 `rtr_4`/`port0`），最终表现为 `issue -> pending_clear` 过长，进而导致 `prefetch_wait` 主导总周期。

---

## 1) 当前权威基线（2026-04-24）

推荐直接以这组结果作为新 session 的起点：

- run id：`run_20260424_event_causal_k128`
- log：`tests/artifacts/logs/test_default_run_20260424_event_causal_k128.log`
- stats 目录：`tests/artifacts/stats/overlap0/run_20260424_event_causal_k128/`

关键有效性信号：

- `Simulation is complete`：PASS
- `causal_model_source=event`
- `event_full_match_count=496`
- `event_invalid_order_count=0`

以上字段可在：

- `tests/artifacts/stats/run_summary.csv`
- `tests/artifacts/stats/overlap0/run_20260424_event_causal_k128/submit_ready_causal_summary.csv`

---

## 2) 最小阅读顺序（按这个顺序不会迷路）

1. 本文档：`tests/doc/GOLEM_NEW_SESSION_ONBOARDING_COMPUTE_DOMINATE_ZH.md`
2. 编译边界：`tests/doc/compile_boundaries.md`
3. 运行入口：`tests/run_noc_dma_pipeline.sh`
4. 架构装配入口：`tests/architecture/ncores_selfcom_dma_ctrl.py`
5. 因果拆解脚本：`tests/stats/extract_submit_ready_causal_csv.py`
6. NoC 热点脚本：`tests/stats/extract_noc_hotspot_csv.py`
7. 核心运行时代码：
   - `src/sst/elements/golem/workercmdproc/workercmdproc.h`
   - `src/sst/elements/golem/requestscheduler/requestscheduler.h`
   - `src/sst/elements/golem/requestscheduler/requestscheduler.cc`
   - `src/sst/elements/golem/globalmemory/globalmemory.h`
   - `src/sst/elements/golem/globalmemory/globalmemory.cc`
   - `src/sst/elements/memHierarchy/memNICBase.h`

---

## 3) 项目架构与代码路径（新同学必看）

### 3.1 逻辑数据流

1. RoCC 触发任务
2. WCP 组织 window/tile 请求
3. worker `RequestScheduler` 发起读请求（mat/vec）
4. manager `RequestScheduler` 在 credit 约束下下发 NoC
5. NoC -> memNIC -> DRAM
6. READ_RESP 回程到 worker `GlobalMemory`
7. pending 清除，tile ready，进入 compute

### 3.2 关键路径对应代码

- 请求发起/完成生命周期：`src/sst/elements/golem/requestscheduler/requestscheduler.cc`
- tile ready 与 pending 管理：`src/sst/elements/golem/globalmemory/globalmemory.cc`
- memNIC 桥接收发：`src/sst/elements/memHierarchy/memNICBase.h`
- 系统拓扑与参数注入：`tests/architecture/ncores_selfcom_dma_ctrl.py`

### 3.3 事件级因果链路（event model）

因果拆解依赖以下日志事件（按 requestId 关联）：

- `TRACE_REQ_ISSUE ... req=... cycle=...`
- `[memNICBase bridge] recv READ cycle=... req=...`
- `[memNICBase bridge] send READ_RESP cycle=... req=...`
- `TRACE_REQ_DONE ... req=... issue_cycle=... pending_clear_cycle=...`

最终三段定义：

- `forward_to_memnic = issue -> memNIC recv`
- `memory_service = memNIC recv -> memNIC send`
- `return_path = memNIC send -> pending_clear`

---

## 4) 当前瓶颈证据（基于最新 k=128 event 有效 run）

来源：`tests/artifacts/stats/overlap0/run_20260424_event_causal_k128/`

### 4.1 执行时间构成

- `total_cycles = 78307.94`
- `compute_active_time_share = 5.39%`
- `prefetch_wait_time_share = 82.33%`
- `writeback_wait_time_share = 12.27%`

结论：运行时间被 prefetch wait 主导。

### 4.2 event 因果三段

- `issue_to_pending_clear`：mat `7144.33`，vec `7467.39` cycles
- `forward_to_memnic`：`16.76` cycles（很小）
- `memory_service`：`1515.17` cycles
- `return_path`：mat `5717.52`，vec `5833.66` cycles
- `return_path_share`：mat `80.03%`，vec `78.12%`

结论：**返回路径是主导瓶颈**。

### 4.3 内存侧并非主瓶颈

- `mem_avg_read_latency_cycles = 23.46`
- backend `memory_backend_read_latency_p95_cycles = 207`

结论：DRAM 服务时间存在但不是主导项。

### 4.4 NoC 热点（OS 侧汇聚特征明显）

- `total_xbar_stalls = 34361`
- `top1 router = rtr_4`，占 `67.66%`
- `top1 port = rtr_4:port0`，占全局 `49.28%`
- `output_port_stalls = 0`

结论：瓶颈表现为 xbar 仲裁热点（路径汇聚/角色流叠加），与当前 OS 侧热点判断一致。

---

## 5) 编译边界与原则（必须遵守）

权威说明见：`tests/doc/compile_boundaries.md`

### 5.1 必须全量 clean rebuild 的改动

- 改动 `src/sst/elements/golem/globalmemory/*`
- 改动 `src/sst/elements/memHierarchy/*`

命令：

```bash
cd /data4/lishun/pkg/sst-elements
make clean
./configure --prefix=/data4/lishun/pkg/sst_install --with-dramsim3=/data4/lishun/pkg/DRAMsim3
make -j4
make install
```

### 5.2 只需重编 libgolem 的改动

- 改动 `src/sst/elements/golem/rocc/*`
- 改动 `src/sst/elements/golem/requestscheduler/*`
- 改动 `src/sst/elements/golem/groupctrl/*`
- 改动 `src/sst/elements/golem/workercmdproc/*`

命令：

```bash
cd /data4/lishun/pkg/sst-elements/src/sst/elements/golem
make -j4
make install
```

### 5.3 不需要重编库的改动

- `tests/configs/*`
- `tests/stats/*`
- `tests/tools/*`
- `tests/run_noc_dma_pipeline.sh`

### 5.4 运行原则

- `run_noc_dma_pipeline.sh` 禁止并行执行
- 一次只跑一个配置，避免 artifacts 互相污染
- 每次改动后先确认 baseline 可复现，再做下一步

---

## 6) 新 session 的工作原则（执行纪律）

1. 先证据、后结论：所有判断必须能落到 csv/log 行为证据。
2. 先小改、再回归：单次只引入一个变量，防止多因素耦合。
3. 不在 debug 打点中改变语义：trace 只能观测，不改行为。
4. 涉及跨时钟域比较时，统一到 scheduler 域再下结论（关注 `memnic_cycle_scale`）。
5. 不轻易触碰 `globalmemory` 与 `memHierarchy`，除非明确接受全量重编成本。

---

## 7) Compute-Dominate 优化需求（目标与验收）

### 7.1 目标定义

从当前的 data-movement bound，逐步转向 compute-dominate：

- `compute_share` 明显上升
- `prefetch_wait_share` 明显下降
- `issue->pending_clear` 显著缩短

### 7.2 当前优先级（按收益排序）

1. 回程路径疏导：针对 OS 侧热点 router/port 的汇聚缓解。
2. 请求窗口与 credit 协同：减少长尾排队和持续满队。
3. 保持 event-level 三段观测闭环：每次改动都看 `forward/service/return` 是否改善。
4. 在正确性不退化前提下推进微流水（micro-tiling / dataflow 化）。

### 7.3 验收口径（建议）

- 功能：`Simulation is complete`，并保持当前验证流程通过。
- 因果：`causal_model_source=event`，`event_invalid_order_count=0`。
- 性能：`return_path_share` 下降，`compute_active_time_share` 上升。

---

## 8) 新 session 首小时 checklist

1. 运行基线：

```bash
GOLEM_RUN_ID=run_YYYYMMDD_k128_check GOLEM_GEMM_BLOCK_K=128 ./run_noc_dma_pipeline.sh
```

2. 检查产物目录：`tests/artifacts/stats/overlap0/<run_id>/`
3. 必看文件：
   - `execution_summary.csv`
   - `submit_ready_causal_summary.csv`
   - `noc_hotspot_summary.csv`
   - `noc_hotspot_port_table.csv`
4. 必看日志信号：
   - `Simulation is complete`
   - `TRACE_REQ_ISSUE`
   - `TRACE_REQ_DONE`
   - `memNICBase bridge] recv READ cycle=`
   - `memNICBase bridge] send READ_RESP cycle=`
5. 对比 `tests/artifacts/stats/run_summary.csv` 的最新一行，确认口径一致。

---

## 9) 常见误判（避免重复踩坑）

- 误判 1：把低 `compute_share` 解释成 compute 单元算力问题。
  - 实情：当前主要是等数据返回，不是算得慢。
- 误判 2：把瓶颈归因给 DRAM 平均时延。
  - 实情：event 分解显示 return path 占比更高。
- 误判 3：看到 HBM 通道均衡就认为 NoC 无问题。
  - 实情：xbar 热点可在中间/OS 侧 router 形成，与通道均衡不矛盾。

---

## 10) 维护说明

- 每次确认新“权威基线”后，更新第 1 节 run id 与第 4 节关键数值。
- 若统计字段变更，同时更新：
  - `tests/stats/extract_submit_ready_causal_csv.py`
  - `tests/run_noc_dma_pipeline.sh`
  - 本文档对应字段定义。
