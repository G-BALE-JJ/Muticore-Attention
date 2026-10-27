# 四阶段正式重跑与端到端延迟

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
