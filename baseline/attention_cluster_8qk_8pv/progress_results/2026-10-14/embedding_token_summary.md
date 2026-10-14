# Embedding → 两层 prefill → logits → greedy token

SST wall 单独计时，排除 host 初始化和验证；4/8 rank 顺序运行。每组输入、内存布局、guest 和 SST library hashes 相同。

| 规模 | rank 4 wall s | rank 8 wall s | 加速比 | wall 减少 | rank 4 cycle | rank 8 cycle | token ID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| formal | 1566.425133 | 1377.198749 | 1.137× | 12.08% | 20,167,497.750 | 20,167,497.750 | 99037 |
| small | 48.166472 | 46.140689 | 1.044× | 4.21% | 1,101,944.750 | 1,101,944.750 | 5313 |

| Case | embedding cycle | 首层交接 cycle | 全链路 cycle | 1 GHz ms | node 使用 MiB | argmax host µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| formal_rank4 | 31,508.625 | 51,908.375 | 20,167,497.750 | 20.167498 | 355.910187 | 761.203 |
| formal_rank8 | 31,508.625 | 51,908.375 | 20,167,497.750 | 20.167498 | 355.910187 | 763.892 |
| small_rank4 | 19,728.625 | 51,552.375 | 1,101,944.750 | 1.101945 | 7.281281 | 62.729 |
| small_rank8 | 19,728.625 | 51,552.375 | 1,101,944.750 | 1.101945 | 7.281281 | 63.529 |

| 阶段（互不重叠，含数据搬运） | rank 4 cycle | rank 8 cycle | rank 4 占比 |
| --- | ---: | ---: | ---: |
| Embedding | 31,508.625 | 31,508.625 | 0.156% |
| Embedding → 首层交接 | 51,908.375 | 51,908.375 | 0.257% |
| 第一层 Norm1/QKV/Attention | 519,926.250 | 519,926.250 | 2.578% |
| 第一层 Wo/残差/Norm2/FFN/残差 | 8,795,572.500 | 8,795,572.500 | 43.613% |
| 层间交接 | 49,240.250 | 49,240.250 | 0.244% |
| 第二层 Norm1/QKV/Attention | 509,969.750 | 509,969.750 | 2.529% |
| 第二层 Wo/残差/Norm2/FFN/残差 | 8,793,658.000 | 8,793,658.000 | 43.603% |
| final norm 交接 | 15,584.250 | 15,584.250 | 0.077% |
| final RMSNorm | 48,740.750 | 48,740.750 | 0.242% |
| LM head 交接 | 9,963.250 | 9,963.250 | 0.049% |
| LM head | 1,341,425.750 | 1,341,425.750 | 6.651% |

Embedding 数值要求逐元素精确匹配；检查 token/表/输出 DMA 地址与字节覆盖、独立 flag reset/completion ACK、第一层在全局完成后读取。
模拟 cycle 的起点为首个 token-ID DMA send，终点为最后 logits output DMA ACK。host argmax 不计入加速器 cycle。
MPI rank 数只改变仿真任务分区，硬件仍为 20 cores / 16 workers / 4 data HBM nodes，各 node 2 GiB，GM 2 MiB，带宽未修改。
8 rank 将同一列的 QK 与 PV worker 分到成对 rank；每 rank 两个 worker。此对比每种配置各运行一次，wall time 可能受宿主机波动影响。
使用合成权重和 1024 token ID，未接 tokenizer、真实 checkpoint、完整模型层数或自回归 decode。
