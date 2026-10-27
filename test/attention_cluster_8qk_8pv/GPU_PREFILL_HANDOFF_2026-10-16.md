# RTX 5060 prefill 对照实验交接

更新日期：2026-10-16。源主机项目根目录：`/data/jjgong/LLM/Muticore-Attention`。
本文供通过 SSH 查阅源代码、在另一台 RTX 5060 主机继续实验的人或编码代理使用。

## 1. 接手目标与当前边界

下一项任务是实现并运行与当前 SST **相同计算图、相同输入权重**的 GPU prefill，
测量总延迟、各阶段延迟及精度差异，再决定下一轮软件融合/流水化方向。

**GPU prefill runner 尚未实现，也没有 RTX 5060 的实测结果。**
本次交接已准备文件清单、哈希、从已验收 HBM 导出的普通数组参考和 SST 阶段统计。
这些可直接用于下一步实现，无需再跑 SST。

用户约束：

- 不随意修改硬件架构；GM 保持每 worker 2 MiB。阵列、operand context、
  SFU、GM 端口/bank/仲裁、NoC/HBM 时序均沿用已校准版本。
- 当前只做 prefill，真实 checkpoint 导入暂缓；使用现有合成权重即可建立对照。
- 先局部筛选，再做完整链验收；不要每次修改都重跑全局 SST。
- 保留有效证据，避免复制 HBM 镜像、build 和重复日志导致磁盘膨胀。

源工作树有大量未提交/未跟踪修改，**仅 clone GitHub 不能保证拿到当前代码和数据**。
以本文配套清单的文件内容及 SHA256 为准，不要求清理、提交或覆盖当前工作树。
参考工程 `/data/jjgong/RISC-V-CIM-Manycore-SST` 用于此前模型校准，GPU 首轮实验不需要复制它。

## 2. 首轮冻结任务

| 参数 | 当前正式 SST case |
| --- | --- |
| batch / 逻辑 S / 实际执行 S | 1 / 1024 / 1024，无序列 padding |
| decoder 层数 | 2；各层独立参数 |
| hidden / FFN intermediate | 2048 / 8192 |
| Q heads / KV heads / head dimension | 32 / 8 / 64，GQA，每 4 个 Q heads 共用一组 K/V |
| Attention | causal，包含对角线；scale=1/sqrt(64)=1/8；dropout=0 |
| RMSNorm | epsilon=1e-6；每层两套 gamma，末尾另有 final gamma |
| RoPE | 相邻偶/奇维配对；位置 0..1023；base=10000，直接读已导出的 FP16 表 |
| 词表 / LM 输入 | 128256；仅最后一个真实 token，索引 1023 |
| 线性层 | 无 bias；文件权重为 `[out_features, in_features]`，计算 `X @ W.T` |
| Embedding / LM 权重 | **两份独立矩阵，不绑定权重** |
| 存储 | FP16 参数和张量；token IDs 为 little-endian uint32 |
| 权重类型 | decoder/LM 数值稀疏，但 SST 执行完整 dense tiles；Embedding 为稠密随机表 |

这是 Llama 风格的合成两层计算图，**不是已经导入真实 Llama checkpoint 的整模型结果**。
不要调用框架默认 Llama 配置替换上述 epsilon、RoPE、head 数、层数或权重绑定方式。
GPU 同样使用 dense GEMM，不能将 sparse-valued 权重自动改成稀疏 kernel。

源数据：`results/prefill_generalization_20261016/formal_l2_s1024/`。
机器可读交接清单：
`baseline/attention_cluster_8qk_8pv/progress_results/2026-10-16/gpu_prefill_handoff/transfer_manifest.json`。
清单内 `config` 固定上述参数，`files` 给出相对项目根目录的路径、角色、大小和 SHA256。
清单本身作为数据包索引，未将自己的哈希放入自己的 `files`，传输包另有整体 SHA256。

## 3. 正确计算图及加载方式

每层按以下顺序执行，先验证清晰的基础实现，再接优化后端：

```text
token IDs → Embedding lookup → X
for each layer:
    N1 = RMSNorm(X, gamma1)
    QKV = N1 @ Wqkv.T
    Q, K, V = split(QKV)
    Q, K = adjacent-pair RoPE(Q, K)
    O = causal_attention(Q, K, V, GQA)
    H1 = X + O @ Wo.T
    N2 = RMSNorm(H1, gamma2)
    G = N2 @ Wgate.T
    U = N2 @ Wup.T
    X = H1 + (SiLU(G) * U) @ Wdown.T
Y = RMSNorm(X, final_gamma)
logits = Y[:, logical_S-1, :] @ Wlm.T
```

QKV 的 **列排列是所有 Q heads、所有 K heads、所有 V heads**，不是每个 head 的 Q/K/V 交错：
`QKV.reshape(B,S,Hq+2*Hkv,D)`，Q 为前 32 个 heads，K 为后续 8 个，V 为最后 8 个。
转给 Attention 后端前通常需转为 `[B, heads, S, D]`；结果再转回 `[B,S,Hq,D]` 后展平 hidden。
Q head `h` 对应 KV head `h // 4`，不是 `h % 8`。

RoPE 表 `rope_64.bin` reshape 为 `[S,64]`，0::2 为 cosine，1::2 为 sine：
`out_even = even*cos - odd*sin`，`out_odd = even*sin + odd*cos`。
某些框架默认按 hidden 的前后两半旋转，不能直接套用。

文件约定：

| 文件名，均位于正式 case 目录 | dtype / shape / 含义 |
| --- | --- |
| `input_token_ids.bin` | `<u4`，`[1024]`；加载到 PyTorch Embedding 前转 int64 |
| `embedding_weight.npy` | FP16，`[128256,2048]`，标准 row-major |
| `projection_x.bin` | `<f2`，`[1024,2048]`，必须等于 `embedding_weight[token_ids]` |
| `layer{1,2}_qkv.npy` | FP16，`[3072,2048]` |
| `layer{1,2}_wo.npy` | FP16，`[2048,2048]` |
| `layer{1,2}_{gate,up}.npy` | FP16，`[8192,2048]` |
| `layer{1,2}_down.npy` | FP16，`[2048,8192]` |
| `layer{1,2}_gamma{1,2}.bin`、`final_gamma.bin` | `<f2`，各 `[2048]` |
| `rope_64.bin` | `<f2`，`[1024,64]` |
| `lm_weight.npy` | FP16，`[128256,2048]`，与 Embedding 独立 |
| `last_token_logits.npy` | SST FP16 输出，`[128256]` |

**GPU 不需要 `_node*.bin` 权重文件**，它们是 SST 的分块/条带化布局。
`decoder_stack_inputs.json` 的 `preloads.path` 含原主机绝对路径，只用于溯源；
GPU loader 按交接清单中的相对路径加载标准数组，不照抄这些绝对路径。

从已验收输出提取的额外参考位于 `results/gpu_prefill_handoff_20261016/reference/`：
`sst_layer1_output.npy`、`sst_layer2_output.npy`、`sst_final_norm.npy`，
均为标准 `[1024,2048]` FP16 数组，可直接比较，无需 GPU 主机解析 HBM。
`reference_provenance.json` 记录提取来源、偏移及内容哈希。

## 4. SSH 查阅与精简传输

在 GPU 主机的 Linux/WSL2 shell 设置源主机登录地址，替换下方占位符：

```bash
GPU_BASELINE_SST_HOST='用户名@源Linux主机地址'
ssh "$GPU_BASELINE_SST_HOST"
```

登录源主机后：

```bash
cd /data/jjgong/LLM/Muticore-Attention
less baseline/attention_cluster_8qk_8pv/GPU_PREFILL_HANDOFF_2026-10-16.md
git status --short
```

本次交接同时提供源主机上的预打包文件：
`results/gpu_prefill_handoff_20261016/rtx5060_prefill_bundle.tar.gz` 及 `.tar.gz.sha256`。
可直接使用后面的 scp 命令下载；若未来清单或源码更新，需要重新核对并另建传输包。

以下命令用于**缺少传输包时重建**，在源主机项目根目录运行。
它按清单打包，先核对文件哈希，
不扫描或复制整个 `results/`。正式输入的逻辑体积约 1.2 GiB，压缩包大小以实测为准；
Embedding 表本身是随机稠密数据，不能假设所有文件都能像零块一样大幅压缩。
代码目录与结果目录的层级在包内保持不变。

```bash
python3 - <<'PY'
import hashlib, json, pathlib, tarfile
root = pathlib.Path.cwd()
manifest = pathlib.Path('baseline/attention_cluster_8qk_8pv/progress_results/2026-10-16/gpu_prefill_handoff/transfer_manifest.json')
report = json.loads(manifest.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()
for entry in report['files']:
    path = root / entry['path']
    assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], entry['path']
destination = root / 'results/gpu_prefill_handoff_20261016/rtx5060_prefill_bundle.tar.gz'
assert not destination.exists(), '包已存在；确认用途后另起文件名，避免覆盖'
with tarfile.open(destination, 'w:gz', compresslevel=1) as archive:
    archive.add(root/manifest, arcname=str(manifest))
    for entry in report['files']:
        archive.add(root/entry['path'], arcname=entry['path'])
destination.with_suffix(destination.suffix+'.sha256').write_text(sha(destination)+'  '+destination.name+'\n')
print(destination)
print('compressed bytes:', destination.stat().st_size)
PY
```

回到 **GPU 主机 shell**，下载并解包到独立工作目录：

```bash
mkdir -p ~/rtx5060-prefill
cd ~/rtx5060-prefill
scp "${GPU_BASELINE_SST_HOST}:/data/jjgong/LLM/Muticore-Attention/results/gpu_prefill_handoff_20261016/rtx5060_prefill_bundle.tar.gz" .
scp "${GPU_BASELINE_SST_HOST}:/data/jjgong/LLM/Muticore-Attention/results/gpu_prefill_handoff_20261016/rtx5060_prefill_bundle.tar.gz.sha256" .
sha256sum -c rtx5060_prefill_bundle.tar.gz.sha256
tar -xzf rtx5060_prefill_bundle.tar.gz
```

该包不含 SST 安装库/guest 二进制、完整 build 或 HBM 镜像。
包里的 `.py` 源码供实现/核对 GPU 计算图；不要在 GPU 主机执行 `run_prefill_validation.py`
或 `run_decoder_stack.py` 试图启动 SST。后续新增 GPU runner 可独立使用 PyTorch/NumPy。

## 5. GPU 环境检查

Windows 主机推荐 WSL2 + Ubuntu。若 GPU 主机本身已是原生 Linux，也可直接使用，
但结果必须注明是哪种环境，不能混合不同环境的延迟样本。
WSL2 使用 Windows NVIDIA 驱动，不要在 WSL 内安装 Linux 显示驱动。
参考 [NVIDIA CUDA on WSL 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装支持 RTX 5060
Blackwell 的 CUDA 构建，并记录实际版本。不要照搬旧 V100 文档的环境或硬编码旧 wheel。
先做一次真实 FP16 矩阵乘法，`is_available()` 单独为真还不够：

```bash
nvidia-smi
python3 - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available()
print('device:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
print('compiled architectures:', torch.cuda.get_arch_list())
a = torch.ones((256,256), device='cuda', dtype=torch.float16)
b = a @ a
torch.cuda.synchronize()
assert torch.isfinite(b).all() and torch.all(b == 256)
print('FP16 GEMM smoke: PASS')
PY
```

首轮不需要自己编译 CUDA 扩展，也不需要安装/运行 SST。
显存不足时先检查误留的 FP32 全模型副本、autograd 和大 Attention 中间量，
不要悄悄启用 CPU offload、量化或删层后仍称为相同任务。

## 6. GPU runner 的实现任务

建议新建独立目录 `gpu_prefill_baseline/`，至少交付：
`run_gpu_prefill.py`、依赖/环境记录、`README.md` 和结果目录。
这些是**待实现的文件名**，本文没有声称仓库已存在完整 GPU runner。

实施顺序：

1. **loader/配置验证**：按清单核对 SHA、shape、dtype、层数；确认 Embedding gather
   与 `projection_x.bin` 逐位一致。校验 Q/K/V 排列、相邻维 RoPE、GQA 和最后 token。
2. **基础计算图**：`inference_mode`，FP16 参数与存储，Norm/Softmax 等按明确的 FP32
   策略计算。先小输入做 CPU/FP32 数值检查，再运行冻结的正式两层输入。
3. **优化实现**：dense GEMM 允许 Tensor Cores；Attention 可使用 SDPA 等可用优化后端，
   causal、dropout=0、GQA 配置须一致。记录实际后端，不能默认一定运行 FlashAttention。
   如手动展开 KV heads，记录扩展的分配/复制发生在计时内还是计时外。
4. **分开计时**：完整链无逐阶段同步地测量；另开诊断轮次采集阶段/kernel 时间。
   若再使用 compile/CUDA Graph，单列结果，编译与 capture 不计入稳态执行。
5. **对照与回传**：完成下一节指标，保存小型 JSON/CSV/输出数组与脚本快照。

GPU 通常具有与 SST 不同的 GEMM 累加/舍入方式。不要用慢速 GPU 逐元素模拟
SST 的每个 FP16 舍入点作为唯一性能基线，也不要据此修改 SST 精度或时序。
原始标准 GPU 实现和优化 GPU 实现均应有可核对的数值结果。

当前 `prefill_precision_reference.py` 仅依赖 NumPy，是独立 FP32 数学参考，可用于
小尺寸验证或分层计算；正式输入用 CPU FP32 参考会较慢，需记录独立的验证 wall。
GPU FP32 参考若使用 TF32，应明确关闭或单独标注，不能默认为严格 FP32 参照。

## 7. 对比口径与验收

**主指标**：输入 token IDs 和全部权重已经驻留设备存储后，从 Embedding 开始，
到最后 token 的完整 logits 写回设备存储结束。覆盖完整两层和 final norm。
模型加载、文件 I/O、H2D/D2H、Tokenizer、编译、host argmax 均不计入此主指标，
需要时另报应用端到端延迟。SST 的 RISC-V 描述符派发、同步及轮询在计时区间内
仍然计入；它与 GPU 主机发射机制不同，应作为执行环境差异注明。

GPU 使用 CUDA events 和同步读取结果；先预热，例如 20 次，再测至少 100 次。
报告中位数、p95、最小值和样本数，保留所有小型时间样本。
避免计时循环中 `.item()`、转 CPU、打印或重新加载权重。
完整链的 host 发射间隙可能进入设备事件区间，不要宣称已绝对排除所有 CPU 影响。
参考 [PyTorch CUDA 异步计时说明](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)。

分阶段至少记录 Embedding、Norm1/QKV/RoPE/Attention、Wo/残差/Norm2/完整 FFN、
final RMSNorm、最后 token 的 LM head；有 profiler 时再拆分 GEMM、SFU 类操作。
**独立算子耗时不能简单相加替代完整链延迟**；要记录重叠、发射间隙与缓存状态。

数值至少保存每层输出、final norm、logits 的 max-abs、RMSE、relative-L2，
以及 greedy token 和 top-1/top-2 margin。首轮预先采用 `atol=0.01, rtol=0.01`
作为逐元素诊断门槛，并报告失败元素数；这是 GPU 对接筛查门槛，**不是此前已验收的
SST 精度标准，也不等于任务质量达标**。超出门槛先检查数据排列和数值路径，
不要在看到结果后静默改阈值。与 SST 和共同 FP32 参考的误差分开记录。
GPU 不要求逐位复现 SST，也不能将 token 相同作为唯一的数值通过条件。

SST 当前数值限制：组合 Attention 参考采用独立高精度数学计算，不能称全链 bit-exact。
16 层 H256 验证曾超过旧 final norm 固定 0.006 阈值，改用有记录的输入误差传播界
加局部算子检查；这不能当作真实 checkpoint 质量证据。首轮 GPU 比较使用无 padding
的 H2048 两层，仍应公开全部误差。详情见
[prefill 验证报告](PREFILL_GENERALIZATION_2026-10-16.md)。

## 8. 已冻结的 SST 性能锚点

完整入口 **7,424,562 cycle**，按模型 1 GHz 为 **7.424562 ms**。
起止时间戳：285,672,000 → 7,710,234,000，1000 tick/cycle。
SST wall=1091.150795 秒只表示宿主仿真成本，**不能与 GPU 推理毫秒数比较**。
SST greedy token=99037 是现有合成模型的输出索引，不是输入长度，也不代表真实文本语义。

| 连续、不重叠的区段 | cycle |
| --- | ---: |
| Embedding | 29,749.625 |
| Embedding → decoder 交接 | 56,627.375 |
| 第 1 层 Norm1/QKV/Attention | 546,031.750 |
| 第 1 层 Wo/残差/Norm2/FFN | 2,716,415.000 |
| 两层交接 | 53,895.250 |
| 第 2 层 Norm1/QKV/Attention | 537,044.875 |
| 第 2 层 Wo/残差/Norm2/FFN | 2,704,514.000 |
| final norm 交接 | 15,513.125 |
| final RMSNorm | 53,832.375 |
| LM head 交接 | 16,665.625 |
| LM head | 694,273.000 |
| **合计** | **7,424,562.000** |

GPU 相对 SST 的同任务加速比统一用 `7.424562 / GPU_median_ms`，
必须标明 GPU 实现、精度和后端。
若改了层数、S、权重或边界，需要新的匹配 SST 对照，不能继续拿此分母比较。
不得用 GPU boost clock 将时间转换成“等价 cycle”后直接比较两个架构的原始 cycle。
不能从延迟单独推导能效、面积效率或同等资源优势。

此前 7,362,979.75 cycle 属于通用性改造前的 guest。当前增加 0.8364%，输出逐位相同。
LM head 中每个词表块重复解析逻辑 S 是候选软件开销，尚未配对证明全部增量来源。
首轮 GPU 对比固定当前版本，不同时修复这个问题而混用基线。

## 9. 源码阅读导航与后续顺序

路径均相对项目根目录：

| 路径 | 用途 |
| --- | --- |
| `baseline/attention_cluster_8qk_8pv/run_decoder_stack.py` | 合成输入、权重生成、RMS/RoPE/组合链及验收契约 |
| `baseline/attention_cluster_8qk_8pv/prefill_precision_reference.py` | 独立 FP32 层数学参考、误差指标 |
| `baseline/attention_cluster_8qk_8pv/run_decoder_layer.py` | FP16 存储和 GEMM 舍入参考、HBM 输出读取 |
| `baseline/attention_cluster_8qk_8pv/dense_linear_reference.cpp` | 独立 host 标量 GEMM 参考，GPU 性能无需逐步照搬 |
| `baseline/attention_cluster_8qk_8pv/verify_lm_head.py` | 真实最后 token、完整词表、SST 计时边界 |
| `src/sst/elements/golem/tests/small/muticore_attention/attention_case.py` | 当前 RoPE 表生成 |
| `src/sst/elements/golem/tests/small/muticore_attention/golem_decoder_runtime.h` | guest Wo/FFN/final norm/LM head 顺序及描述符 |
| `src/sst/elements/golem/attention/decoderMemoryLayout.h` | 统一 HBM 布局和 2 MiB GM stride |
| `baseline/attention_cluster_8qk_8pv/progress_results/2026-10-16/prefill_generalization/` | 9 个验收 case 的轻量证据 |
| `src/sst/elements/golem/tests/gpu_baseline/` | 历史 V100 FP32 方阵 GEMM，仅作计时方法参考 |

后续先完成当前正式两层对比，再考虑正式 16 层和 S 扫描。
**已有 16 层 SST 实测为 H256，正式 H2048 仅两层**；容量检查不能代替正式 16 层实测。
非整块 S 当前向上补齐到 1024 倍数，额外工作计时。将来比较变长性能时，
要同时标注 SST 逻辑/物理 S 和 GPU 是否按真实 S 执行，不能隐藏 padding 成本。

GPU 数据回来后，再筛选 RMSNorm→投影按行块衔接或 O→Wo 分块衔接。
这些仅是候选，必须重查 worker 数据归属、2 MiB 占用和真实 GM/NoC 争用。
不要机械沿用已注明架构失效的旧 Norm overlap 报告或未校准 GM 的历史收益。

## 10. 结果回传与产物管理

建议返回单独目录 `gpu_prefill_results/<run_id>/`，其中包含：

- `environment.json`：GPU 名称/显存、原生 Linux 或 WSL2、OS、驱动、Python、
  PyTorch/CUDA/cuDNN 版本、TF32/累加设置、实际 Attention 后端、代码哈希。
- `config.json`：冻结配置、输入哈希、精度/舍入说明、计时边界、预热/重复次数。
- `latency_samples.csv`、`stage_times.csv`、`summary.json`：原始时间样本、阶段时间、
  总延迟 median/p95/min、峰值 allocated/reserved 显存、与 SST 的匹配性及加速比。
- `numerics.json` 和必要 `.npy`：每层/最终 norm/logits 的误差、阈值、失败元素数、
  greedy token 与 margin；计时结束后再回传到 CPU。
- GPU runner 源码及版本/依赖记录。失败结果明确写 FAIL，不能只回传最好一次。

GPU 主机可用以下形式回传，目录名按实际 run_id 替换；不要覆盖 SST 验收目录：

```bash
ssh "$GPU_BASELINE_SST_HOST" 'mkdir -p /data/jjgong/LLM/Muticore-Attention/results/gpu_prefill_import'
scp -r gpu_prefill_results/你的run_id "${GPU_BASELINE_SST_HOST}:/data/jjgong/LLM/Muticore-Attention/results/gpu_prefill_import/"
```

不要复制 `build/`、`install/`、全部 `results/` 或重复 MPI rank 日志。
同一批输入只保留一份，timing 循环不逐次 dump 大张量；正式输出只在数值验证轮保存。
核对 SHA、解包文件及有效结果后，可删除**自己本次创建**的临时 tar 包以避免双份占用，
不要删除源 case 或交接清单引用的数据。

## 11. 可直接给下一位编码代理的任务

> 阅读本文及 transfer_manifest.json，通过 SSH 使用源主机当前工作树和冻结输入。
> 在 RTX 5060 主机实现独立 PyTorch prefill runner，复现 H2048/S1024/两层合成计算图。
> 先核对 Embedding、QKV 排列、GQA、相邻维 RoPE、epsilon、权重转置和最后 token，
> 再分别验证基础与优化后端的数值及稳态延迟。保留共同 FP32 参考和 SST 对照的误差。
> 输入权重常驻显存，主计时为 Embedding→完整 logits，使用预热、CUDA events 和重复样本。
> 不修改 SST 硬件、精度、输入或现有验收结果，不要求重新跑 SST，不下载真实 checkpoint。
> 返回代码、环境、输入哈希、阶段/端到端延迟、数值误差及峰值显存。
