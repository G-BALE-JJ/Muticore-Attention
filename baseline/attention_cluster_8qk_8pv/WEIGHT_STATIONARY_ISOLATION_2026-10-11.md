# 32-array weight-stationary 隔离实验

状态：用户决定舍弃该原型，代码与专属统计已移除，不再增加snapshot或修补WS调度。正式64-array基线始终未启用它；新路线见 [舍弃WS后的重分析](LATENCY_REANALYSIS_WITHOUT_WEIGHT_STATIONARY_2026-10-11.md)。

这里的实验形状是 Hq/Hkv=4/2、D64、hidden256，并非正式 Hq/Hkv=32/8、hidden2048。

实验固定以下条件，只切换 `GOLEM_PROJECTION_WEIGHT_STATIONARY`：32 arrays、C-buffer partial、pipeline=0、wrap-prefetch=0、partial-restore-ahead=0、fused Norm=1、packed Norm input=1、RoPE fusion=0、audit output=0。

| 序列长度 | WS | numerical/HBM | projection span | E2E | 每 worker matrix program | 每 worker partial read | 每 worker input read request |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1024 | 0 | PASS | 52,118 | 91,109 | 64 | 16 | 64 |
| 1024 | 1 | PASS | 52,655 | 91,828 | 32 | 48 | 128 |
| 2048 | 0 | PASS | 72,773 | 201,067 | 128 | 32 | 128 |
| 2048 | 1 | PASS | 73,862 | 201,755 | 64 | 96 | 256 |

weight-stationary 确实把 matrix program 减半，但它同时使每个 row group 的输入 pair 重新读取，并且每个 hidden tile 之间都需要 C-buffer restore。以 S=1024、core=4 为例，`matrix_program` 从 2560 降到 1280，但 `output_restore` 从 560 增到 1680，`output_read` 从 608 增到 1216；因此减少的 weight programming 被单一 output accumulator 上的 partial 流量抵消，projection 和 E2E 都变慢。

另外，关闭 fused Norm 的实验中，WS=1 没有通过数值校验：249,340 个 mismatch，最大绝对误差 0.220458984375。此前后台 pair prefetch 实验也出现最大绝对误差 0.12567138671875。静态检查发现奇数tile重读时非packed地址未按偶数pair对齐，而packed地址使用tile/2；这可解释布局敏感性，但未做单变量修复验证。不能据这些失败就断言WS思想必然算错，或把另一个配置的错误直接归因为snapshot竞争。

## 结论

1. 对该小形状和当前实现，WS引入的额外input/partial流量抵消了matrix program节省，受控实验的E2E变慢。
2. 这不能解释正式64-array基线的延迟差距，也不证明所有硬件上的WS都不可行。
3. 按用户决定退役该路线，后续只研究既有64-array输出驻留路径的软件流水重叠。

实验 artifacts：

- `results/ws_isolation_s1024_fused_off`
- `results/ws_isolation_s1024_fused_on`
- `results/ws_isolation_s2048_fused_off`
- `results/ws_isolation_s2048_fused_on`
