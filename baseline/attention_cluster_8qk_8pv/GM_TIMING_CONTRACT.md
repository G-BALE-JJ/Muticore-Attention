# GM 请求接收间隔与返回延迟核查

## 结论

当前 GM 实现是**非流水请求服务模型**：请求可以连续入队，但每个端口只能有一个正在服务的请求；端口在请求返回时释放。因此启动间隔与服务延迟绑定。配置的 256 B/cycle 是传输项的速率，并不保证 256 B 小请求流能持续达到这个速率。

源码和时序探针已确认模型行为。目前没有足以确认当前配置物理 SRAM 接口的 RTL/宏时序契约，因而**本次保留生产模型**。把启动间隔改成仅由传输长度决定，会改变既有性能假设，不能当成已经证实的 bug 修复或调度收益。

## 三个不同时间点

正式配置：GM 时钟随 CPU 为 1 GHz；base latency 1；每端口 256 B/cycle；读写各一个端口；共享 pending 容量 256；单请求上限 4096 B。

1. **API 接收**：`localReadAsync` / `localWriteAsync` 在参数合法且 pending 未满时立即返回 true，并记录请求。没有每 cycle 只能接收一次的限制；同一仿真时间可以接收多个请求。这表示进入队列，不表示 SRAM 已开始服务。
2. **端口启动**：`tryIssueLocalAccesses` 只在 `localReadsInFlight_ < readPorts` 或对应 write 条件满足时启动服务。单读端口最多一个 in-flight read；单写端口最多一个 in-flight write。读写分别排队，可以同时启动。
3. **返回/提交可见**：`issueLocalAccess` 发送延迟事件，延迟为 `L + ceil(bytes/B)`。`handleLocalAccessEvent` 到达后才释放端口、读取 backing storage 或提交写入，然后启动下一个请求并调用旧请求的 callback。callback 第二参数是 tag，不是返回时间。

其中 `L=1`、`B=256`。无排队时返回延迟是 `L+ceil(bytes/B)`；从 API 接收到返回还要加排队时间。连续同类型请求的端口启动间隔也为 `L+ceil(bytes/B)`。

| 请求长度 | API 接收间隔下限 | 当前端口启动间隔 | 无排队返回延迟 | 连续请求有效吞吐 |
| --- | --- | ---: | ---: | ---: |
| 256 B | 无固定限制，受队列容量约束 | 2 cycles | 2 cycles | 128 B/cycle |
| 4096 B | 同上 | 17 cycles | 17 cycles | 240.94 B/cycle |

不要混用统计单位：GM queue_cycles 字段实际累计 `getCurrentSimCycle()` 差值，即 SST 核心时间 ticks；延迟事件的 serviceCycles 使用 `local_access_clock`。换算应使用相应 time converter，不能直接把 queue_cycles 当 CPU/GM cycle 相加。

## 可复查的时序探针

[`probe_gm_timing.py`](probe_gm_timing.py) 从当前生产 C++ 文件原样提取 `localAccessValid`、读写提交、延迟计算、issue、仲裁和完成方法，使用独立事件投递环境编译执行。它没有重写这些方法的调度逻辑；替代的是 SST 周边类型与事件投递。这是源码调度探针，**不是完整 SST 集成测试或物理 RTL 验证**。探针采用一 GM cycle 为一个时间单位，不用于验证 SST 的统计时间单位换算。

连续三个读请求在 t=0 全部成功入队：

| 长度 | 端口启动时间 | 返回时间 |
| --- | --- | --- |
| 256 B | 0、2、4 | 2、4、6 |
| 4096 B | 0、17、34 | 17、34、51 |

独立读写端口测试中，一个 256 B read 和一个不重叠地址的 256 B write 均在 t=0 启动、t=2 完成，并验证返回数据与写入可见性。探针包含断言并已通过。结果及源码 SHA256 保存在 [`gm_timing_probe_results.json`](gm_timing_probe_results.json)。

```bash
python3 baseline/attention_cluster_8qk_8pv/probe_gm_timing.py
```

## 其他实现提供的证据边界

同一工程的 C-buffer 已分开建模传输占用和完成延迟：

```text
start = max(issue, next_port_free, matching_write_ready_for_read)
next_port_free = start + ceil(bytes/B)
ready = next_port_free + L
```

因此 C-buffer 的独立无依赖请求启动间隔不包含 L，GM 则包含 L。这确认了两者建模策略不同；不能仅据此认定 GM 错误，二者也可能对应不同接口。

当前项目没有检出 HDL、SRAM 宏 `.lib` 或给出 GM 启动间隔的硬件时序规范。相邻 `/data/jjgong/RISC-V-CIM-Manycore-SST` 工程有 HDL 面积/功耗包：

- `golem_gm_bank_cluster.sv` 的组合仲裁不等待上一响应，可每 cycle 授权一个请求到每个 bank；同 bank 同 cycle 的客户端请求冲突由固定优先级解决。
- 仿真 SRAM wrapper 在时钟沿执行同步读；GM 将 grant 和读数据再经响应寄存器送出。若在 E0 沿接受请求，wrapper 数据在 E0 后更新，响应在 E1 后有效。这个边沿描述仅对应 `GOLEM_SRAM_SIM` 行为模型，不是已绑定 foundry 宏的物理时序。
- 尽管 wrapper 为 1R1W，该 bank 仲裁每 cycle 仅选择一个请求，不能直接等同于当前 SST 的独立全局读写端口。
- README 明确说明该 RTL 不是 SST 的 cycle-exact 替代。其 GM 为 512 KiB、8 banks、128 B line，当前投影配置 GM 为 2 MiB；因此不能直接移植吞吐数字。默认 SRAM wrapper 为 black box，还需宏绑定。

该相邻工程仅只读检查，未修改。它支持“启动间隔和返回延迟可以分离”的实现方向，但不足以证明当前 256 B/cycle 物理接口的契约。

## 修正模型的判据

如果目标 GM 确认每 cycle 可接收一个 256 B beat，而额外 L 仅表示返回流水延迟，则应将**端口占用**改为 `ceil(bytes/B)`，保留**完成延迟** `L+ceil(bytes/B)`。例如三个 256 B 请求可在 0、1、2 启动，仍在 2、3、4 返回。这是接口模型校准，不增加物理端口，也不改变数据流的请求数量。

若目标控制器只能在上一请求完成后接受下一请求，则当前绑定方式正确，无需修正。

校准必须同时明确读写冲突/同地址访问顺序、bank 仲裁、最大 outstanding 及返回背压；不能仅提前减 in-flight 计数，因为那会连带改变队列容量、写入可见性和读写依赖。现有 queue depth 包含等待与 in-flight 请求，端口数则不应同时充当流水 outstanding 深度。

没有物理接口依据时，建议把流水服务作为显式的独立模型选项进行敏感性实验，保留当前模型为对照；其 cycle 差异应报告为建模假设变化。此前固定资源调度结果仍保持原 GM 模型，数字不重算。

## 源码依据

- [`globalmemory.cc`](../../src/sst/elements/golem/globalmemory/globalmemory.cc)：请求入队、延迟计算、端口仲裁、完成释放和存储可见性。
- [`globalmemory.h`](../../src/sst/elements/golem/globalmemory/globalmemory.h)：参数与 pending/in-flight 状态。
- [`cpu_builder.py`](../../src/sst/elements/golem/tests/architecture/cpu_builder.py)：GM 参数映射及 CPU 时钟选择。
- [`workercmdproc.h`](../../src/sst/elements/golem/workercmdproc/workercmdproc.h)：C-buffer 服务占用与完成延迟分离。
- 正式配置记录：`results/projection_partial_restore/control/s1024/baseline_provenance.json`。
