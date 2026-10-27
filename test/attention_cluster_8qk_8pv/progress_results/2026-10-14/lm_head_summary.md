# 两层 prefill + final RMSNorm + 最后一个 token 的 LM head

全部 case 完成数值、epoch、输入/权重/输出字节覆盖及 DMA ACK 检查。1 GHz 下每 cycle 为 1 ns。

| Case | hidden / vocab | node 容量 MiB | node 使用 MiB | norm ACK 总 cycle | head 交接 cycle | head cycle | logits ACK 总 cycle | 延迟 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small_vocab9_2gib | 256 / 9 | 2,048 | 5.128937 | 1,011,693.62 | 10,039.38 | 1,342.62 | 1,023,075.62 | 1.023076 |
| two_layer_formal_2gib | 2048 / 128256 | 2,048 | 229.652374 | 18,587,742.75 | 9,849.25 | 1,339,615.75 | 19,937,207.75 | 19.937208 |
| two_layer_small | 256 / 8201 | 128 | 6.144562 | 1,007,124.75 | 9,788.25 | 12,161.88 | 1,029,074.88 | 1.029075 |

| Case | chunks / 尾块列数 | head 局部参考误差 | 完整组合参考误差 | logits ACK bytes |
| --- | --- | ---: | ---: | ---: |
| small_vocab9_2gib | 1 / 64 | 0.000000000 | 0.000000000 | 128 |
| two_layer_formal_2gib | 16 / 5376 | 0.000000000 | 0.000244141 | 256,512 |
| two_layer_small | 2 / 64 | 0.000000000 | 0.000244141 | 16,512 |

每个 worker 只读取 token 1023 的 hidden 向量，M=1，不读取其余 63 个填充 token。每块 N<=8192，尾块补齐到 64 列；host 仅返回有效 vocab logits。
每个 worker 按 core 分配词表 tile，只激活一条 array 行；所有 weight tile 经现有 DMA、GM、WCP 和真实 array 执行。
head 完成标志只在最后词表块输出 ACK 后发布；不同 worker 可以交叠不同块，逐块时间不能直接相加。
计时包含层间、final norm 和 LM head 交接及数据搬运，排除 host preload、启动及最终 logits ACK 后的通知/退出。
2 GiB/node 是容量；SST functional backing 与地址 stride 同步扩大，链路带宽和 Ramulator geometry 不随容量扩大。
原两层使用合成权重，尚未加载真实 checkpoint；输出保存为 last_token_logits.npy，尚未接 token 采样。
