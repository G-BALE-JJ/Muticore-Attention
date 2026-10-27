# 两层 Decoder prefill：统一布局与重复执行

后续已追加独立 final RMSNorm；本页保留两层 Y 输出的测量边界，最新尾部流程与验证见 [FINAL_RMSNORM_2026-10-14.md](FINAL_RMSNORM_2026-10-14.md)。

目标：在一次 SST 仿真内执行两层完整 Decoder，覆盖 RMSNorm → QKV → RoPE/Attention → Wo → 第一残差 → RMSNorm → gate/up → SiLU/乘法 → down → 第二残差。第二层直接读取第一层的 HBM 输出，host 只在仿真前初始化 X0 和权重。

## 地址的唯一来源

`src/sst/elements/golem/attention/decoderMemoryLayout.h` 定义 `makeDecoderMemoryLayout()`。guest 和 Attention bridge 调用同一函数；host 编译 `dump_decoder_layout.cpp`，读取该函数输出的 JSON，再生成 HBM preload manifest。Python 不再为第二层维护另一套十六进制地址。

布局按 4 KiB 对齐、顺序分配，启动前检查是否超出每节点 128 MiB。Tensor 中间结果、Q/K/V/O、Attention panels、RoPE 表和两套完成标志只有一份。每层独立分配 QKV/Wo/gate/up/down 权重以及两次 RMSNorm 的 gamma。

| 激活 | 第一层 | 第二层 |
| --- | --- | --- |
| input | activation0 = X0 | activation1 = 第一层 Y |
| output | activation1 | activation0 = 第二层 Y |

每个数据节点的正式尺寸占用 103.1484375 MiB，小尺寸占用 4.9609375 MiB。数据节点仍是 4 × 128 MiB，worker GM 仍是 2 MiB，独立 WCP C-buffer 仍是 1 MiB。扩展层数只增加权重分配并循环复用两份激活；容量不足时拒绝运行，不能把“地址可扩展”理解为目前可驻留 16 层正式尺寸权重。

旧单层 checkpoint 使用原布局，以便继续比较历史测量；新增 stack 路径使用统一布局。GM 控制邮箱、metadata 地址和 decoder scratch 地址也集中在该头文件。

## SST 节点容量与 Ramulator 几何容量

两者不是同一个配置。`ncores_selfcom_dma_ctrl.py` 根据 SST physical memory 配置分配每个节点的地址范围、`backend.mem_size` 和 mmap backing。本次数据节点各 128 MiB，四个节点的 functional 权重存储空间合计 512 MiB。

每个节点另接一个独立的 Ramulator2 backend 实例，加载同一份 `hbm2e_2500.yaml`。该配置含 8 个 controller，每个的组织为 `[1,2,1,4,4,16384,128]`；HBM2 internal prefetch=4，transaction=32 B。地址映射的每 controller 空间为 `2 × 1 × 4 × 4 × 16384 × (128/4) × 32 B = 512 MiB`，8 个 controller 合计约 4 GiB/backend。这是 timing/address geometry；权重 payload 仍保存在 SST backing 中。

因此“现有 512 MiB 放不下完整权重”指 SST 对外暴露的数据容量，不能解释为 Ramulator 几何只有 512 MiB。扩大 SST 节点时需同步改地址范围、preload/backing、guest/GM 路由 node stride 和统一布局，再校验 Ramulator 地址映射覆盖范围；只改 YAML 几何不会扩大 SST functional 存储。本步两层不修改这两项容量。

## 完成标志与同步

1. Root 等上一层所有 16 个 worker 的完成标志。每个标志只有在输出 DMA 写入 ACK 后才写入，job 在标志写入 ACK 后完成。
2. Root 对 4 个节点、每节点 4 个 lane 的 projection/decoder 标志，共 32 个地址写零。新 `0x28` 控制指令通过现有定时 DMA 写入，收到 ACK 后才回应 guest。
3. Root 发布当前 layer epoch，其他 manager 等到此 epoch 后再释放自己的 projection worker。
4. QKV completion 值为 epoch，所有 manager 等四节点的 16 个标志后才派发 Attention。每层每个 head 使用独立 Attention job/tag。
5. Root Attention wait 覆盖所有 manager/head 的输出 ACK，再用 epoch 释放 16 个 decoder worker。
6. 最后一条残差完成输出和标志 ACK；root 收齐 16 个 epoch 标志，再开始下一层。

GM release 邮箱也是递增 epoch，避免把上一层的 `1` 当作新层的 release。既显式清零，也检查 generation，分别验证标志生命周期和迟到完成的隔离。

## Operand cache 与 GM 所有权

旧 PV V-cache 以 HBM source 地址识别 tensor，跨 head 可以复用，但跨层同一地址装入了新的 V；FFN/projection 还会覆盖缓存所在的 worker GM。两层测试已实际暴露这一问题：清零 completion 和换 job tag 仍不够。

投影开始前、Attention 转交 Wo/FFN 前，先等待 WCP 和 V DMA/prefetch/install 完全空闲，再撤销 WCP V-cache tags、RoPE panel cache、RoCC V broadcast delivery history，清理 PV prefetch 调度状态；物理 GM 容量和端口没有增加。这一处理保留层内 head/window 的缓存复用，阻止跨层读到旧数据。每个 worker 两层共四次 scratch ownership transition，trace 检查合计 64 次。

## 验证方法

- 两层使用不同的合成稀疏权重与不同 gamma；硬件仍执行完整 dense tile，不跳过零权重计算。
- 标志初始设置为 `0xfeedface`，验证第一层开始前的清零。
- 每层必须有 16 个 projection job、9 × 16 个 decoder op；两层合计 32 个 projection job、288 个 decoder op、64 个 reset ACK。
- 检查两个 reset barrier、层序号、实际 input/weight 地址、Attention QK/SFU/PV job 数量、tile/window 数量、root tag 唯一性、8 次 manager Attention 完成和两次 root Decoder 完成。
- CPU 独立组合两层，从 X0 开始计算，不用硬件中间结果生成完整层预期值；另检查第二层保留下来的全部后处理阶段。
- 检查每层 Attention O 与 Y 的 DMA ACK 字节覆盖完整且无重复，并检查所有 O ACK 先于 Wo、第一层全部 Y ACK 先于第二层 input DMA。
- 最终两组 HBM 标志必须全部为 2。

统计周期采用 `第一层 X0 首次 input DMA send → 第二层 Y 最后输出 DMA ACK`，包括中间层交接、标志 reset 和计算/数据搬运，排除首个 input DMA 以前的启动和最后输出之后的打印/退出。各算子 stage span 在不同 worker 之间重叠，不能相加。

## 复现

```bash
source scripts/env_local_install.sh
# 修改 C++ 后按项目流程重新编译、安装 libgolem。
make -C src/sst/elements/golem/tests/small/muticore_attention decoder-two-layer
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/decoder_stack_NEW/small --shape small
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/decoder_stack_NEW/formal --shape formal
# 重做 host 检查，复用仿真产物而不复用仿真状态：
python3 baseline/attention_cluster_8qk_8pv/run_decoder_stack.py \
  --artifact-root results/decoder_stack_NEW/formal --verify-only
```

每个 case 保存 preload manifest、布局区域表、guest/library/source hashes、源码快照、仿真配置、HBM 输出、rank logs、DMA trace、numerics/lifecycle/timing JSON。

本步仍是 S=1024、H=256/2048、FFN=4H 的合成权重 prefill，沿用项目 FP16 累加、RMSNorm epsilon=1e-6、RoPE base=10000 的行为。真实模型权重/配置对齐、容量扩展、final norm/LM head、decode 和融合优化仍属后续工作。

## 已完成的小尺寸验证

`results/decoder_stack_20261014/two_layer_small_v4/decoder_stack_result.json` 为 PASS。两层独立组合参考的最大绝对误差均为 0.00048828125，第二层 Attention 最大绝对误差 0.00006103515625。Wo/残差/RMSNorm/gate/up/乘法/down 的逐阶段检查为精确匹配，SiLU 最大绝对误差 0.0001220703125。

小尺寸两层 canonical DMA E2E 为 1,013,850.75 cycles（1 GHz 下 1.01385075 ms）；第一层 485,310.75 cycles，层间交接 46,504.25 cycles，第二层 482,035.75 cycles。交接包括完成标志轮询、guest 控制流、清零及 release；不能把它全算成纯硬件 barrier。

旧单层小尺寸 `results/decoder_stack_20261014/legacy_layer_small/decoder_result.json` 同样 PASS。Projection descriptor 新增 8 字节 completion_value，因此 guest 与 libgolem 必须一起重编译；本次已重建普通、clean-startup、batch-release 和各 decoder checkpoint guest。36 项布局/算子/相关回归单元测试通过。

## 正式尺寸两层验证

`results/decoder_stack_20261014/two_layer_formal/decoder_stack_result.json` 为 PASS。S=1024、Hq=32、Hkv=8、D=64、hidden=2048、FFN=8192。两层共 128 个 QK/SFU/Attention completion job、2,560 个 PV window，所有 head 的 root tag 在层间独立。32 个 projection job、288 个 decoder op、64 个 reset ACK、64 个 scratch retire 均完整，最终两组 HBM 标志全部为 2。

正式尺寸 canonical 两层延迟 18,690,216.75 cycles，1 GHz 下 **18.69021675 ms**。第一层 9,277,585.75 cycles，交接 47,451.25 cycles，第二层 9,365,179.75 cycles。两层输出各 4 MiB，Attention O 和 Y 的 ACK 覆盖完整且无重复。SST wall 1,363.012152 s，完整 host 验证约 23.76 s。

两层独立组合参考的最大绝对误差分别为 0.0009765625 和 0.0009765625；第二层 Attention 最大绝对误差 0.0001220703125。第二层 Wo/两次残差/RMSNorm/gate/up/乘法/down 精确匹配逐阶段参考，SiLU 最大绝对误差 0.0001220703125。

逐阶段周期及小尺寸对照见 `results/decoder_stack_20261014/summary.md`、`summary.json`。输出 correctness 为本项目当前的 FP16/合成权重契约，不代表已完成真实 Llama 模型数值对齐。
