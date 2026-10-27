# 两层 Decoder 后的 final RMSNorm

本步将 `Y2 → final RMSNorm → HBM normalized hidden state` 接入同一次 SST 运行。最终归一化使用独立的 gamma，保留 FP16 输入/输出、FP32 平方和、均值、倒数平方根及缩放计算，epsilon=1e-6。所有运算沿用 worker SFU，guest 只提交描述符和同步。

## 内存布局

`makeDecoderMemoryLayout(..., final_norm=true)` 在已有层权重之后追加 `final_gamma`、`final_output`、`final_done`。旧 Decoder 激活、scratch 和权重地址完全相同；host 继续编译共享 C++ allocator 获取布局。final 输出与第二层 Y 独立，避免覆盖检查所需的两层激活。

每节点小尺寸占用 5,337,120 B（5.089874 MiB）；正式尺寸占用 109,211,680 B（104.152374 MiB）。均低于现有 128 MiB/node，不扩大 SST backing、Ramulator geometry、worker 的 2 MiB GM 或 1 MiB WCP C-buffer。

## 同步契约

1. Root 收齐第二层 16 个 decoder completion；每个 completion 先于 guest 返回的条件是 Y 输出和 completion 写入均已 ACK。
2. Root 对 16 个独立 `final_done` 标志定时 DMA 写零，等待所有 reset ACK。
3. Root 用统一 GM 布局中的 `final_norm_release` 向 16 个 worker 发布 epoch=3。
4. Worker 使用现有 DecoderOp ABI，提交 stage=10、op=RmsNorm、epoch=3，读取自己的 64 行 Y 和独立 gamma。
5. 每块输出经 GM 写入及 HBM DMA ACK 后才推进。所有块完成，再写自身的 final completion，等待 ACK 后完成 job。
6. Root 收齐 16 个 epoch=3 标志，打印 `DECODER_FINAL_NORM status=0`。

## 验证

两层输入、权重和 layer gamma 的随机生成序列保持不变。final gamma 在生成两层后独立生成；各节点输出初始化为 NaN，final flags 初始化为非零 `0xfeedface`。

保留原两层完整 Attention/Decoder 数值和生命周期检查，新增两条数值参考：从 X0 独立组合两层后执行 final norm；以及对设备实际 Y 执行局部 final norm。独立组合误差包含两层已有的 FP16 舍入差异，归一化可能放大这些差异，阈值为 0.006；局部参考阈值为 0.002，报告实际误差。

新增检查逐 worker 的输入、gamma 和输出 DMA 地址及字节数，验证输出覆盖完整且无重复；所有 final 输入都晚于第二层全局 Y ACK；每个完成标志只写一次，且其发送和 ACK 都晚于该 worker 最后输出 ACK。最终旧 flags=2、final flags=3；root 成功完成必须唯一。

布局/host 检查测试覆盖容量与无重叠、旧地址保持、独立 gamma 与 NaN 初始化、完整输出 ACK，以及错误 epoch、提前完成、缺失输出和错误 reset 时序的拒绝。

周期边界为 `第一层 X 首次 input DMA send → final normalized hidden state 最后 output DMA ACK`。报告同时保留两层 Y 边界，以便区分 final 计算/搬运与 final 交接。RMSNorm 周期包含输入/输出 DMA、GM 和 SFU，不等于纯 SFU 运算耗时。

## 复现与产物

```bash
source scripts/env_local_install.sh
make -C src/sst/elements/golem/tests/small/muticore_attention decoder-final-norm
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py --final-norm \
  --shape small --artifact-root results/final_norm_NEW/two_layer_small
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py --final-norm \
  --shape formal --artifact-root results/final_norm_NEW/two_layer_formal
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --verify-only --artifact-root results/final_norm_NEW/two_layer_formal
python3 baseline/attention_cluster_8qk_8pv/summarize_final_norm.py \
  --artifact-root results/final_norm_NEW
```

当前测试产物位于 `results/final_rmsnorm_20261014`。每例保存 source snapshot、guest/library/source hashes、HBM 输入 manifest、HBM 输出、rank logs、DMA traces 和 `decoder_stack_result.json`。汇总见该目录的 `summary.md` 和 `summary.json`。

## 已完成结果

小尺寸与正式尺寸均为 PASS。两例的 final norm 局部参考误差均为 0，独立组合整条链路的最大绝对误差均为 0.00390625。正式尺寸检查全部 2,097,152 个 final 输出元素，4 MiB 输出 ACK 完整无重复；16 个 worker 的 final completion ACK 全部晚于各自输出 ACK，最终标志全部为 3。

| 尺寸 | 两层 cycle | final 交接 cycle | final RMSNorm cycle | 含 final 总 cycle | 1 GHz 延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| hidden=256 / FFN=1024 | 979,026.75 | 14,782.25 | 8,165.75 | 1,001,974.75 | 1.001975 ms |
| hidden=2048 / FFN=8192 | 18,568,208.75 | 15,606.25 | 49,058.75 | 18,632,873.75 | 18.632874 ms |

正式尺寸 final 尾部合计 64,665 cycles，占本次端到端周期约 0.347%。这个开销在同一次运行的第二层 Y ACK 与 final 输出 ACK 之间测量；不要用历史两层总时长的差值代替。guest 编译及控制调度改变可能影响前两层运行周期。输入对比审计确认原两层 preload hashes 和地址完全相同，新增 12 项 final preloads。

11 项 Decoder 布局/数值/新同步检查测试及 25 项相关资源/配置/向量回归测试全部通过。关闭 final norm 的原两层入口重新编译并完成小尺寸 SST 回归，结果为 PASS，产物 `results/final_rmsnorm_20261014_legacy_small`。正式尺寸 SST wall 为 1,516.297035 s，约 25.3 分钟；此值与上述模拟推理延迟不同。

审计产物：`provenance_audit.json`、`input_comparison_audit.json`、`legacy_regression_result.json`，以及 `unit_tests.log`、`runtime_regression_tests.log`。

这是两层合成权重 prefill 的最终隐藏状态归一化。下一步接最后一个 token 的 LM head，并处理线性算子的 M=1 尾行及 vocab 输出分块；真实 checkpoint、模型配置对齐和完整层数容量扩展仍待后续。
