# 移除旧路径后的四阶段正式回归

全部使用同一安装库，数值与相应资源检查通过。延迟单位为 1 GHz 参考 cycles；SST tick 为 1 ps。

| 测试 | 状态 | 端到端 cycles | 边界 |
| --- | --- | ---: | --- |
| GEMM | PASS | 3,500.000 | first descriptor start -> last WCP completion (includes C writeback), inclusive +1 |
| Attention | PASS | 12,852.000 | QK WCP + online SFU + NoC P dispatch + PV WCP + final PV ACK |
| GQA | PASS | 37,928.000 | QK WCP + online SFU + NoC P dispatch + PV WCP + final PV ACK |
| prefill | PASS | 6,084,167.750 | first token-ID DMA send to last-token logits final output DMA ACK |

## prefill 完整分解

S1024 / H2048 / FFN8192 / 2 层 / vocab128256，16 workers、MPI8；合成输入与权重，所有稠密 tile 实际执行。

| 互不重叠的区间 | cycles |
| --- | ---: |
| embedding | 28,828.375 |
| embedding -> decoder handoff | 65,077.625 |
| layer 1 norm/QKV/RoPE/Attention to O ACK | 443,604.250 |
| layer 1 Wo/residual/norm/FFN to Y ready | 2,193,363.750 |
| layer 1 -> 2 handoff | 61,064.000 |
| layer 2 norm/QKV/RoPE/Attention to O ACK | 429,054.875 |
| layer 2 Wo/residual/norm/FFN to Y ready | 2,216,710.875 |
| final RMSNorm tail after decoder Y | 41,563.000 |
| final norm -> LM head handoff | 16,966.250 |
| last-token LM head, all vocabulary logits | 587,934.750 |
| **合计** | **6,084,167.750** |

两层 decoder：5,343,797.750 cycles。
相对已选配置差值：0.000 cycles。
最终 greedy token ID：99037，local/composed reference 检查通过。

Final RMSNorm 提前启动并与 decoder 尾部重叠，上表只累计 decoder 完成后的剩余尾部。
完整延迟从首次 token-ID DMA 发送到最后有效 token 的全词表 logits DMA ACK；主机 argmax 不计入。
四个独立算子的配置和测量边界不同，不能相加得到 prefill 延迟。原始形状、命令、检查、证据哈希和主机时间见 summary.json。

## decoder 算子活动跨度

下表为跨 worker 的算子活动跨度，彼此可能重叠，仅作诊断，不能累加为端到端延迟。

| 层 | 算子 | span cycles |
| --- | --- | ---: |
| 1 | Wo | 201,977.000 |
| 1 | residual Add / Norm2 fusion | 62,606.000 |
| 1 | Gate/Up/SiLU/Multiply/Down/residual fusion | 1,938,881.000 |
| 2 | Wo | 203,401.000 |
| 2 | residual Add / Norm2 fusion | 62,959.000 |
| 2 | Gate/Up/SiLU/Multiply/Down/residual fusion | 1,960,384.000 |

## 迁移验收

运行时顶层 `baseline` 与 `Muticore-Attention` 均已移除，四个入口直接访问 `test/`。
全部延迟与迁移前正式回归一致，架构 44 个源文件未改动。
库与 guest SHA256 均与已选配置一致；完整 prefill 的数值、DMA、资源与投影面板审计通过。

- 库 SHA256：`cb211e1a1b2eb88715d236284ca3346675908b32271eb9682bd7deba63340e60`。
- guest SHA256：`7ee4f229860f8c83dea431a72f10cbcf58e02d6de4bb76c454807e95f067f57c`。
- GEMM 全部 16,384 个 C 元素与 A@B 比较，零不匹配。
- 本轮各阶段结果：`test/<阶段>/results/path_migration_20261027/`。
- [完整汇总 JSON](../test/reports/path_migration_20261027/summary.json)、[CSV](../test/reports/path_migration_20261027/summary.csv)。

重新汇总：

```bash
cd /data/jjgong/LLM
python3 test/summarize_stages.py --run-name path_migration_20261027 \
  --output-dir test/reports/path_migration_20261027
```
