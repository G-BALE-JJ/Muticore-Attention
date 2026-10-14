# 定时 Embedding、greedy token 和 MPI 8 rank

当前功能链路为 `HBM token IDs → worker Embedding gather → 两层 Decoder → final RMSNorm → 最后 token 的 LM head → host argmax token ID`。本次保持原有 FP16 I/O、FP32 norm 内部计算、硬件规模、带宽和未融合算子路径。

## Embedding 数据路径

新增 `DecoderOp::Embedding=6`，stage=12。描述符的 `input_dim` 在该操作中表示有效 vocab，`output_dim` 表示 hidden；token ID 为 uint32，表与输出为 FP16。操作通过既有 DecoderOp issue/wait 接口提交，不需要 array 或 SFU 数值计算。

Embedding 表按完整 token 行跨四个 HBM 节点 stripe：`node=1+token%4`，节点内行号为 `token/4`。16 个 worker 各处理自己所属节点的连续 64 个输入 token。每个 worker 定时 DMA 读取 token IDs 到 GM，经 GM 本地读解析，检查范围后，逐行读取表到 GM，再 DMA 写到第一层输入。重复 token 仍逐行读取，没有隐式缓存或 host 查表代替仿真。

统一 C++ allocator 追加 tokens、table、output 和独立 done flags；原两层 scratch、权重、final norm、LM head 偏移保留。首层 input 指向专用 embedding_output，后续两层结果仍复用原两个 activation slots。专用初始输出保留到结束，可从 HBM dump 独立验证。

Host 仅预载 token IDs、表和 NaN 输出，不预载查表后的 X0。`projection_x.bin` 是 CPU 参考输入，不进入 HBM preload；测试从保存的完整表和 token IDs 独立得到参考结果。真实模型 tokenizer 仍未接入。

Root 清零并等待 16 个 Embedding done flags 的 DMA ACK 后释放 worker。每个 worker 等全部输出 ACK 后写自身 epoch=1 标志，Root 收齐全部标志后才能释放第一层 Decoder。重复 token、词表最后一个 token 和跨节点读取都纳入测试。

## 最后 token 的选择

保留有效 vocab logits，host 使用 `select_next_token.py` 的 greedy argmax。它拒绝有效词表内的 NaN/Inf，忽略填充词表列，最高值并列时取最小 token ID。无需 softmax 即可确定最大值索引。

`next_token.json` 保存 token ID、logit、局部/完整组合 CPU 参考的 token ID 和 host 选择耗时。局部参考 token ID 必须一致；完整组合参考的 agreement 单独记录，避免把允许的累计 FP16 误差误当作完全相同的 logits。该功能为确定性 greedy，不包含温度或随机采样、tokenizer 解码或后续 decode。

## MPI 对比

硬件仍为 20 cores、16 workers、4 data HBM nodes，每 node 2 GiB、每 worker GM 2 MiB、WCP C-buffer 1 MiB。正式尺寸为 S=1024、hidden=2048、FFN=8192、vocab=128256，仍只运行两个合成 Decoder 层；每 node allocator 使用约 355.91 MiB。Embedding 与 LM head 使用独立合成表。

4 rank 每列 memory、manager、QK/PV worker 放在同一个 rank。8 rank 每列使用一对 rank：memory、manager 和两个 QK worker 在偶数 rank，两个 PV worker 在配对奇数 rank。所有八个 rank 各有两个 worker。原 shell 的 1/2/4 限制已扩展，2/4 的原映射保持一致。

MPI 数只影响宿主机上的 SST 分区。对比顺序运行，并分别限制在 4/8 个物理 core 对应的 CPU 集合；不增加模拟硬件。当前 OpenMPI 的 `--cpu-set` 将全部 rank 限制在相同集合，未使每个 rank 独占某一 core，实际 affinity 保存于 `host_binding.json`。`attention_mpi_placement.json` 记录实际分区，`simulation_wall_time.json` 的 `sst_wall_seconds` 排除 host 初始化和校验。driver wall 另外报告，包含 run_sst 和校验，不包含 runner 在启动前的 prepare。

## 验证与计时

Embedding 要求逐元素精确匹配；检查 16 个 worker 的 token/table/output DMA 地址、重复读取次数、字节覆盖、输出 ACK 后的 completion、flag reset 和首层全局屏障。保留两层、final norm、LM head 的所有数值与同步检查，并验证有效 logits 的 greedy 结果。

全链路计时边界现在为 `首个 token-ID DMA send → 最后 logits output DMA ACK`，包括 Embedding 和到首层的交接。host argmax 单独计时，数值只代表内存内 logits 的选择，不代表 HBM 到 CPU 的完整读回延迟。单次 wall 比较反映当前宿主机与 trace 配置，不是统计稳定的多轮基准。

## 复现

```bash
source scripts/env_local_install.sh
make -C src/sst/elements/golem/tests/small/muticore_attention decoder-embedding
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py --embedding \
  --shape small --mpi-ranks 4 --artifact-root results/embedding_NEW/small_rank4
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py --embedding \
  --shape small --mpi-ranks 8 --artifact-root results/embedding_NEW/small_rank8
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py --embedding \
  --shape formal --mpi-ranks 8 --artifact-root results/embedding_NEW/formal_rank8
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py --embedding \
  --shape formal --mpi-ranks 4 --artifact-root results/embedding_NEW/formal_rank4
python3 baseline/attention_cluster_8qk_8pv/summarize_embedding_token.py \
  --artifact-root results/embedding_NEW
```

结果保存在 `results/embedding_token_20261014`，汇总脚本核对输入 hashes、source snapshots、相同 guest/library、分区、logits 和 token ID。小规模首次 8 rank 被旧 shell guard 拒绝，未启动 SST；该尝试保留为 `small_rank8_rejected_legacy_guard`，不计入成功结果。小规模 4/8 源码快照之间的 shell 差异仅为允许 8 rank；正式对比使用相同源版本。

## 完成结果

小规模与正式规模的 4/8 rank 共四次 SST 仿真均通过。34 项相关自动化测试通过，guest、共享库编译安装通过。正式两次运行的全部相关源码、guest/library、preload hashes 和统一内存布局一致。

| 规模 | 4 rank SST wall | 8 rank SST wall | wall 减少 | 加速比 | 两者相同的全链路 cycle | 相同 token ID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hidden=256 / vocab=8201 | 48.166472 s | 46.140689 s | 4.21% | 1.044× | 1,101,944.75 | 5313 |
| hidden=2048 / vocab=128256 | 1566.425133 s | 1377.198749 s | 12.08% | 1.137× | 20,167,497.75 | 99037 |

正式 8 rank 为 22 分 57 秒，4 rank 为 26 分 6 秒，节省约 189.23 秒。host 完整校验分别 60.350076 s / 59.924719 s，未计入 SST wall。四次中的各组 4/8 logits 逐元素完全相同，全部数值结果与 DMA 周期边界相同；MPI 分区没有改变模拟推理延迟。

正式全链路在 1 GHz 下为 20.167498 ms。Embedding 本身 31,508.625 cycles，到第一层交接 51,908.375 cycles；二者合计占全链路约 0.414%。每 worker 读取 256 B token IDs 和 262144 B 表行，写 262144 B 输出和 8 B completion，16 worker 的地址/次数/ACK 检查通过。Embedding 2,097,152 个元素误差为零，LM head 局部参考误差为零，完整组合参考最大误差为 0.000244140625；CPU 两种参考均选出 99037。

`results/embedding_token_20261014/summary.md` 和 `summary.json` 保存完整分阶段 cycle、wall 对比与数值审计；`test_checks.json` 保存测试命令。每个 case 保存 `next_token.json`、`last_token_logits.npy`、manifest、source snapshots、provenance、HBM dump 和 rank 日志。建议后续正式仿真可显式用 `--mpi-ranks 8`；默认仍保留 4，需要更多重复测量后再决定是否修改全局默认。
