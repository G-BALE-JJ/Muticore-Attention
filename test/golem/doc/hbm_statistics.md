# Ramulator2 HBM 带宽统计

`GOLEM_MEMCTRL_CLOCK` 是内存控制器驱动 Ramulator2 的时钟，启动脚本通过 `--memory-clock` 将实际配置传给 `stats/extract_ramulator2_summary.py`。独立使用统计脚本时也可使用原有的 `--memory-clock-ghz` 参数。

带宽单位采用十进制 GB/s：`read_bytes / read_window_ck × memory_clock_ghz`。`run_summary.csv` 的 `hbm_peak_bandwidth_gbps` 同样将聚合峰值 B/CK 乘以该频率；`hbm_tccdl_roofline_bytes_per_cycle` 保留 B/CK 单位。当前 4 个数据节点、每节点 8 个物理通道、每通道 32 B/CK、1.25 GHz，对应 1024 B/CK 和 1280 GB/s。

`hbm_utilization_pct` 为实测读带宽 / 峰值带宽，不包含 C 写回占用。`hbm_read_command_utilization_pct` 使用首末读命令发射窗口；读服务窗口指标使用请求到达至完成窗口。不同窗口的百分比不能直接当作同一个指标。

`tccd_stats_version=2`：每个物理 channel 的每个 pseudo-channel 独立记录上一次 RD/RDA。与该 PC 上一次读相比，同 bank group 计 L，不同 bank group 计 S；另一 PC 的访问或中间的写命令不打断这条读序列。L/S 百分比的分母为这两类数量之和；每通道满足 `L + S = RD - 有读访问的 PC 数`。

跨 PC 交替率保持物理 channel 上相邻列命令均为读时的统计口径，分母是单独记录的 `global_read_pairs`，不与 per-PC 的 L/S 数量相加。日志及 hbm_read_command_summary.csv 标记版本和口径；旧日志仍支持提取并标为 version 1，混合版本会拒绝汇总。历史占比不会自动修正，必须重新运行仿真才能获得 per-PC 计数。

L/S 比例描述已发射读命令的访问分布，不是 tCCD_L/S 阻塞时间占比，也不是完整的器件时序合规检查。Ramulator 继续负责全部时序约束。`hbm_tccd_l_cycles=1` 在 Ramulator2 路径表示跨 pseudo-channel 交错的聚合峰值间隔；实际同 bank group 的 tCCD_L 以 HBM YAML 中的时序约束为准，当前为 4 CK。

隔离 C 写回影响可使用现有 `--output-mode fusion`（TileLang JSON 中为 `"output-mode": "fusion"`）。它关闭最终 C 的 HBM 写回，保留计算及本地 C 缓冲路径；需要输出数值时可使用 fusion sink 的 dump 功能。修改统计脚本不改变模拟时序，历史 CSV 不会自动重算。
