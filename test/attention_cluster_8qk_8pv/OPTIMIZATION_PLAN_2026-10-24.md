# 下一阶段优化执行计划（2026-10-24）

本文件给出执行顺序、实验边界与验收条件。P0–P4 已获授权执行，实际状态与结果记录在 `PROGRESS_2026-10-24.md`。
所有新增加速器延迟统一使用 cycles；原始 tick 仅作为时间戳证据保留。
主线是确认当前最快配置、优化 FFN 暴露开销，再处理 PV→Wo 的 GM 二次搬移。

## 实际执行结果（已封存）

P0–P4 已全部完成。所有正式候选使用同一最终库和固定 workload，延迟统一为 1 GHz reference cycles（1000 SST ticks/cycle）；最终采用 control，目标 `<6,000,000 cycles` 未达到。五组的完整端到端 / decoder 结果如下：

| 配置 | Full cycles | Decoder cycles | 结论 |
| --- | ---: | ---: | --- |
| control：全局 lookahead 4、FFN 0 | 6,084,167.75 | 5,343,797.75 | 采用 |
| FFN 专用 lookahead 2 | 6,142,633.75 | 5,412,252.75 | 淘汰 |
| P3 relocation handoff | 6,156,245.75 | 5,414,887.25 | 功能通过、变慢 |
| P4 direct PV→Wo | 6,130,817.375 | 5,408,678.25 | 功能通过、变慢，opt-in |
| P4 combined FFN2+direct | 6,150,203.75 | 5,409,250.75 | 淘汰 |

五组 bitwise 输出、dense work、HBM/GM/NoC 资源审计通过；direct/combined 的 32 jobs 均实际直读 268,435,456 bytes GEMM A，relocation read/write 为 0，租约在最后 ACK 后释放。host 250 项和 production 176 项回归全部通过。完整证据与最终配置见 `results/optimization_20261024/summary.json`、`summary.md` 和 `SELECTED_PREFILL_2026-10-24.json`。

## 0. 对照与固定条件

| 编号 | 配置 | 已记录完整端到端 cycles | 定位 |
| --- | --- | ---: | --- |
| B0 | 10-22 融合/驻留采用版本，window groups=1 | 6,134,778.375 | 已采用配置，保留回归对照 |
| B1 | 10-23 window groups=2，PV→Wo 仍走 HBM | 6,084,167.75 | 当前最快通过配置，待同构建复现后作为新实验基准 |
| H0 | B1 + PV→Wo NoC/GM handoff，保留 relocation | 6,156,245.75 | 流量优化对照，性能尚未通过 |

固定 S=1024、H=2048、FFN=8192、两层、vocab=128256、16 workers、MPI8。
固定 banked GM、每 worker 2 MiB GM、64 arrays、单 operand context、现有端口/队列/时钟和 HBM/NoC 模型。
固定输入、权重、FP16 舍入边界、dense MAC 工作量和输出契约。
完整端到端边界为首次 token IDs 读取至最后有效 token 的完整 logits DMA ACK；host argmax 不计入。

注意：`GOLEM_REUSE_WINDOW_ARRAY_GROUPS` 由共享 `ReuseWindowGemm` 读取，
Projection/QKV 与 Decoder/Wo/FFN 均有调用入口。不能把 B0→B1 的整链收益全部归因于 FFN。
`--ffn-array-groups` 与 `--window-array-groups` 也不能混用；以实际 environment 和逐算子 flow 为准。

## P0：冻结来源并确认 B1

1. 保存当前 tracked diff、相关 untracked 源码、guest/library/验证器哈希及完整运行参数；不自动提交或清理已有改动。
2. 在同一个当前构建下配对运行 B0/B1，PV→Wo handoff 均关闭，只改变 window groups。
3. 核对 QKV、Wo、Gate、Up、Down 的实际 groups、launch、MAC 和 C drain/restore；LM 配置保持一致。
4. 输出 B0/B1 的完整端到端、decoder、逐层关键路径和输出逐位比较。历史数字用于对照，采用决策使用本轮配对数据。

交付：配置/来源快照、配对 comparison.json、明确的采用结论。
通过条件：完整数值/资源/生命周期/覆盖审计通过，B1 在完整端到端和 decoder 两个口径均可复现地更快。
未通过则继续以 B0 为基准，B1 保持实验配置；不因历史单次快 0.825% 自动改默认值。

## P1：用现有证据划分暴露开销

先分析已有日志，仅在缺少必要事件时增加有界的统计/时间戳，且确认采集开关不改变模拟周期。

- FFN：按 layer/operator/worker 汇总 B DMA、B GM、matrix programming、A scatter、MAC、C drain/restore、output 与 panel 等待。
- 同时报告最后完成 worker 和最长单作业 worker；两者不必相同。
- PV→Wo：拆分 PV 最后输出就绪、NoC 排队/交付、landing 最后 ACK、relocation 读/写、首个权重预取、首个 MAC 和 Wo 最后 ACK。
- 区分请求提交到启动的排队时间与真实服务时间；等待计数和相互重叠的 stage span 不相加。

交付：FFN 暴露开销表、PV→Wo 时间线、下一实验对应的具体等待来源。
通过条件：每个实验都能指出要减少的暴露区间；HBM 权重字节占比不能单独证明带宽瓶颈。

## P2：FFN 调度主线

优先处理约占已采用版本关键路径 72.49% 的 post-Attention 路径中的 FFN 部分。

1. 先查已有同源码/同配置结果，避免重复已否定实验。
2. 在 P0 选定的 groups 上，固定 K=32、4 个权重槽、16-tile panel、现有 A 预取和输出重叠。
3. 单变量筛选现有 `--window-gm-lookahead 1/2/4`；先保留当前值作为配对对照。
4. 若 P1 显示 A 就绪被 B GM 请求压住，再独立比较 `--window-a-first 0/1`；确认两个方向的等待以及总时延。
5. 将较优的两个单变量结果组合一次，验证交互；如果共享开关改善 FFN 却拖慢 QKV/Wo，再加入按算子选择的调度配置，不全局套用。
6. 若暴露成本主要是 matrix programming/array 端口，而非 GM 供给，则停止预取深度扫描，按时间线实现一个有界的提交顺序候选；不得在 MAC 完成前重编程同一阵列。

执行梯度：生产调度器行为测试 → 正式宽度单 worker → 最多两个候选进入 16-worker MPI8 → 一个候选进入完整两层配对验收。
单 worker 的资源竞争与全链不同，跨 worker 调度候选可以基于明确原因进入并发测试，不设统一局部百分比淘汰线。
不重复采用已变慢的 two-wave broadcast；不扩大权重槽、GM、阵列上下文或端口。

交付：每个候选的配置、局部/并发周期、权重编程/等待变化及一组完整配对结果。
采用条件：相对同构建基准，完整端到端和 decoder 均改善，数值和资源审计全部通过；局部更快或 wait 降低不足以采用。

## P3：PV→Wo 直接消费 landing，消除二次搬移

当前每 worker 先将 256 KiB landing 通过计时 GM 读写搬入普通输入区。
两层可消除的 relocation 为 8 MiB GM 读 + 8 MiB GM 写；NoC payload 与 landing 写入仍必须保留。

### P3a：先证明 GM 生命周期无冲突

- 保持 landing 在 `[0x1a0000,0x1e0000)`，不移回 Attention 正在使用的低地址区。
- 不能仅修改 `inputGM()`：当前 Wo 的窗口 B scratch 使用 `decoder_down_weight_gm=0x1a2000`，与 landing 重叠。
- 候选方案是仅在该 Wo 路径，将原有四个 B 槽放到已不再承载整块输入的普通输入 arena（候选起点 `0x10000`，共 32 KiB）。实施前核对该阶段全部 live 区间和实际预取写入；C scratch、输出、残差输入都必须保持不相交。
- Wo A 读取通过专用地址选择直接引用 head-major、Dh=64 的 K-panel landing；不要全局重定向所有 GM 访问。
- 建立 ready→消费中→释放的所有权状态。释放必须晚于最后 A 读取 ACK；进入 FFN 前允许原有 Up panel 安全复用尾区。
- 先维持原有全局 Attention 完成依赖，只改变消费地址和 B scratch 的软件分配，避免同时改变启动时序。

### P3b：验证和比较

1. 扩展真实 DecoderOpJob 测试：HBM O 投毒；检查直接 landing 读取、B 槽不覆盖、延迟/失败 ACK、两代复用及 FFN 后续复用。
2. 明确旧 Wo loop 的行为：若暂不支持直接消费，则保留 relocation fallback；开启直读优化必须实际走已验证路径。
3. 扩展流量审计，分别统计 landing、relocation 和正常 GEMM A 读取；直读时 relocation=0，禁止把实际 A 读取也误计为消除。
4. 建立足够覆盖 Wo→Add/Norm2→FFN scratch 复用的局部运行入口；再进入完整两层测试。
5. 同构建保留 HBM 路径、旧 handoff+relocation、新 handoff+直读三组，其他调度设置完全相同。

交付：生命周期图/区间检查、生产路径行为测试、三组流量与时延对比。
功能通过条件：O HBM 读/写为 0，landing 覆盖完整，relocation 读/写为 0，GM/NoC 全部计时且所有权合法。
性能采用条件：完整端到端和 decoder 均优于同构建 HBM 路径；只比 H0 更快仍不足以采用。
若仍慢，按 P1 证据决定是否做一次有界 landing 请求调度实验；未有证据前不扩充队列或缩小 NoC 计费。

## P4：组合验收、回归与阶段收敛

对 P2/P3 各自胜出的修改做四格比较：原基准、仅 FFN、仅直读 handoff、两者组合。
复用同批次/同构建且配置完全匹配的已有控制结果；代码构建不同则重新配对。
如果 P3 只通过功能，性能配置只采用 P2，P3 保持 opt-in。

- 全链独立数学参考沿用既有误差阈值；保持计算顺序/舍入边界的候选还要求 Y、final Norm、全部 logits 与配对对照逐位相同。
- 继续检查 QKV panels、全部 QK/PV 与通用 GEMM 窗口、dense MAC、权重流量、C 部分和、DMA/GM/NoC ACK 和 banked GM；未计时张量流量必须为 0。
- 覆盖 groups 1/2、handoff 开/关和不支持形状的显式拒绝/原有回退；不把 S=1024 的驻留结果推广到更长序列。
- materialized counterpart 仅供 host 验证，不能注入模拟；保留最初失败记录与最终通过报告的对应关系，入口明确指向最终审计。
- 使用现有 compact 产物策略及空间检查；每组完成后再启动下一完整实验，不无条件并发多组大仿真。
- 更新专项进度与 README，分别标明已采用、功能通过但变慢、未验证；需要提交时按审计/调度/直读路径分组，不混入无关工作区改动。

阶段性能目标可设为完整端到端低于 6,000,000 cycles（相对当前 6,084,167.75 cycles 还需约 1.38%），
这是实验目标而非预计收益。未达到目标也必须给出采用/淘汰结论，不能扩大硬件或改变测量边界凑结果。

## 主要入口与现有证据

- 当前进度：`PROGRESS_2026-10-23.md`。
- B1 最终通过报告：`results/ffn_boundary_audit_20261023/full_groups2_reuse_audit/all_gemm_reuse_window_report.json`。
- H0：`results/pv_wo_handoff_20261023/summary.json`。
- 关键路径与 HBM：`analyze_reuse_window_critical_path.py`、`audit_prefill_hbm.py`。
- 局部筛选：`run_reuse_window_micro.py`；完整验收：`run_all_gemm_reuse_window_baseline.py`。
- 主要生产代码：`reuseWindowGemm.h`、`decoderOpJob.h`、`pvWoHandoff.h`、`pvWoInputInterface.h`、`decoderMemoryLayout.h`。
