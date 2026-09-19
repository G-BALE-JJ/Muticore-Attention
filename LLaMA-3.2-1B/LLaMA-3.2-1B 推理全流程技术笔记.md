# LLaMA-3.2-1B 推理全流程技术笔记

> 本文档基于 Meta 官方公开的 LLaMA-3.2-1B 模型配置（config.json）及 Transformer Decoder 架构通用原理整理，用于学习记录单次推理（Prefill + Decode）的算子级拆解过程。

---

## 目录

1. [模型规格](#1-模型规格)
2. [核心术语表](#2-核心术语表)
3. [推理全流程（附具体数字 SQ=1024）](#3-推理全流程)
4. [关键概念答疑](#4-关键概念答疑)
5. [数据量总览表](#5-数据量总览表)

---

## 1. 模型规格

LLaMA-3.2-1B 官方真实配置：

| 参数 | 数值 | 说明 |
|---|---|---|
| hidden_size (d_model) | 2048 | 主干道宽度，贯穿全网络不变 |
| n_layers | 16 | Decoder Block 堆叠层数 |
| n_attention_heads (Q) | 32 | Query 头数 |
| n_kv_heads (GQA) | 8 | Key/Value 头数，采用分组查询注意力 |
| head_dim | 64 | 每个注意力头的维度，= d_model / n_heads |
| intermediate_size (FFN) | 8192 | FFN 中间层维度，= 4 × d_model |
| vocab_size | 128256 | 词表大小 |
| rope_theta | 500000 | RoPE 基频，支持长上下文（128K） |
| rms_norm_eps | 1e-5 | RMSNorm 防除零系数 |
| tie_word_embeddings | True | Embedding 与 LM Head 权重共享 |

**不同规模模型配置对比**（同属 LLaMA3 家族，配置并不统一）：

| 模型 | d_model | n_heads | n_kv_heads | head_dim | layers | Q:KV比例 |
|---|---|---|---|---|---|---|
| 1B | 2048 | 32 | 8 | 64 | 16 | 4:1 |
| 3B | 3072 | 24 | 8 | 128 | 28 | 3:1 |
| 8B | 4096 | 32 | 8 | 128 | 32 | 4:1 |
| 70B | 8192 | 64 | 8 | 128 | 80 | 8:1 |
| 405B | 16384 | 128 | 8 | 128 | 126 | 16:1 |

> ⚠️ 结论：`n_heads`、`n_kv_heads` **并非所有 LLaMA3 模型固定不变**，必须查看具体模型的 `config.json`。但 `n_kv_heads=8` 在多个规模上保持不变，是 Meta 的设计选择。

---

## 2. 核心术语表

| 术语 | 全称/含义 |
|---|---|
| Token | 词元，文字被切分后的最小处理单元 |
| Embedding | 将 token ID 映射为高维语义向量 |
| d_model | 模型主干道宽度（每个token向量的总维度） |
| head_dim | Attention 内部单个头的维度，`d_model = n_heads × head_dim` |
| Q / K / V | Query（查询）/ Key（标签）/ Value（内容），Attention 三要素 |
| RoPE | 旋转位置编码，将位置信息编码进 Q、K 的旋转角度 |
| GQA | 分组查询注意力，多个 Q head 共享少量 K/V head，节省显存 |
| KV Cache | 缓存历史 K、V 结果，避免生成时重复计算 |
| RMSNorm | 均方根归一化，比 LayerNorm 少一步减均值操作 |
| FFN | 前馈神经网络，Attention 之后的逐 token 独立加工模块 |
| SwiGLU/SiLU | LLaMA 使用的门控激活函数，SiLU(x) = x × Sigmoid(x) |
| Softmax | 把分数转换为总和为1的概率分布 |
| Logits | 模型输出的原始未归一化分数 |
| Residual | 残差连接，将处理前后的结果相加，防止信息丢失 |
| Temperature | 采样温度，控制生成随机性 |

---

## 3. 推理全流程

以下以 **B=1, SQ(T)=1024** 为例，展示 Prefill 阶段（首次处理整段输入）的完整张量形状变化。

### ① Tokenize：文字 → 数字ID

```
input_ids: [1, 1024]   # 1024个token的ID
```

### ② Embedding：数字 → 向量

$$
h_0 = W_{emb}[\text{input\_ids}]
$$

```
W_emb 形状: [128256, 2048]
h_0 形状:   [1, 1024, 2048]
```

算子：`Gather`

### ③ 单层 Decoder Block（重复16次）

#### Step 1 — Pre-Attention RMSNorm

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2+\epsilon}} \odot \gamma
$$

算子链：`Pow(2) → ReduceMean → Add(eps) → Rsqrt → Mul → Mul(weight)`

形状：`[1,1024,2048] → [1,1024,2048]`（不变）

#### Step 2 — QKV 投影（体现 GQA）

```
Q = MatMul(x, Wq)   Wq:[2048,2048] → Q:[1,1024,2048] → reshape [1,32,1024,64]
K = MatMul(x, Wk)   Wk:[2048,512]  → K:[1,1024,512]  → reshape [1,8,1024,64]
V = MatMul(x, Wv)   Wv:[2048,512]  → V:[1,1024,512]  → reshape [1,8,1024,64]
```

> K、V 的输出维度仅为 Q 的 1/4，是 GQA 节省显存的核心设计。

#### Step 3 — RoPE 位置编码

$$
q' = q\cos\theta + \text{rotate\_half}(q)\sin\theta
$$

算子链：`Slice → Neg → Concat → Mul(cos) → Mul(sin) → Add`

形状不变，仅数值旋转。

#### Step 4 — KV Cache 写入

```
K_cache: [1,8,1024,64]
V_cache: [1,8,1024,64]
```

单层KV Cache显存占用（fp16）：
$$
2\times8\times1024\times64\times2\text{bytes}\approx2\text{MB} \Rightarrow \text{16层}\approx32\text{MB}
$$

#### Step 5 — GQA 广播

```
K: [1,8,1024,64] → repeat×4 → [1,32,1024,64]
V: 同理 → [1,32,1024,64]
```

#### Step 6 — Attention 核心计算

```
① scores = MatMul(Q, K^T)          → [1,32,1024,1024]  (约3300万数字)
② scores = scores / √64
③ scores += causal_mask            (上三角为-∞)
④ attn权重 = Softmax(scores)       (Exp→ReduceSum→Div)
⑤ attn_out = MatMul(attn权重, V)   → [1,32,1024,64]
```

> Attention计算量 ∝ SQ²，是长文本推理的主要瓶颈；生产环境常用 **FlashAttention** 融合上述5步，避免完整落盘 `[T,T]` 矩阵。

#### Step 7 — 多头拼接 + 输出投影

```
Transpose+Reshape: [1,32,1024,64] → [1,1024,2048]
attn_output = MatMul(拼接结果, Wo)   Wo:[2048,2048] → [1,1024,2048]
```

> 拼接仅是物理排列，Wo矩阵乘法才是让32个头信息真正融合的关键步骤。

#### Step 8 — Residual

$$
h_1 = h_0 + \text{attn\_output}
$$

#### Step 9 — Pre-FFN RMSNorm

（同Step 1公式）

#### Step 10 — SwiGLU FFN

```
gate = MatMul(x, W_gate)   [2048,8192] → [1,1024,8192]
up   = MatMul(x, W_up)     [2048,8192] → [1,1024,8192]
act  = gate × Sigmoid(gate)             (SiLU激活)
h    = act × up
ffn_out = MatMul(h, W_down) [8192,2048] → [1,1024,2048]
```

#### Step 11 — Residual

$$
h_2 = h_1 + \text{ffn\_out}
$$

**`h_2` 即第1层最终输出，重复Step1~11共16次，得到 `h_16`。**

### ④ 输出层

```
Final RMSNorm(h_16) → [1,1024,2048]
logits = MatMul(h_final, W_emb^T)   # tied embedding，复用W_emb
       → [1,1024,128256]
取最后一位: logits[:,-1,:] → [1,128256]
```

### ⑤ 采样解码

```
概率 = Softmax(logits/temperature)
[可选] Top-k: Sort→Slice   Top-p: CumSum→Mask
next_token = Argmax(概率) 或 Multinomial(概率)
```

### ⑥ Decode 阶段（生成第2个及后续token）

利用 KV Cache，仅处理新增1个token：

```
Q_new: [1,32,1,64]
K_cache更新: [1,8,1024,64] concat [1,8,1,64] → [1,8,1025,64]
scores = MatMul(Q_new, K_cache^T) → [1,32,1,1025]  # 注意Q长度≠KV长度
```

> **Prefill阶段** Q、K、V序列长度相等（同为T）；**Decode阶段** Q长度=1，K/V长度=历史累积长度，两者不相等。这正是KV Cache发挥加速作用的地方。

---

## 4. 关键概念答疑

### Q: d_model 和 head_dim 谁才是"维度"？

两者都是维度，层级不同：
- **d_model**：贯穿全网络的"主干道宽度"，Embedding输出、每层输入输出、最终输出始终是这个宽度（如2048）
- **head_dim**：仅在Attention内部临时出现，将d_model"切片"给多个头分别计算（如2048÷32=64），算完后立刻拼回d_model宽度

关系：`d_model = n_heads × head_dim`（针对Q这一侧恒成立）

### Q: n_heads=32、n_kv_heads=8 是所有LLaMA3固定的吗？

**不是**，仅为1B模型的具体配置。不同规模（3B/8B/70B/405B）的头数、head_dim均不同，需查看对应`config.json`。

### Q: Q和KV的序列长度一定一样吗？

- **Prefill阶段**：一样（自注意力，Q/K/V均来自同批输入）
- **Decode阶段**：不一样（Q=1，K/V=历史累积长度，因KV Cache机制）

---

## 5. 数据量总览表（SQ=1024示例）

| 阶段 | 关键形状 | 数据量级 |
|---|---|---|
| Tokenize | [1,1024] | 1024个整数 |
| Embedding后 | [1,1024,2048] | 约200万浮点数 |
| 单层Q | [1,32,1024,64] | 约210万数 |
| 单层K/V(各自) | [1,8,1024,64] | 约52万数 |
| Attention矩阵 | [1,32,1024,1024] | 约3300万数 |
| 单层FFN中间层 | [1,1024,8192] | 约840万数 |
| LM Head输出 | [1,1024,128256] | 约1.31亿数 |
| KV Cache(16层总计) | — | 约32MB (fp16) |
| 参数量验证 | — | Embedding约2.63亿 + 16层约9.73亿 ≈ 1.24B |

---

*文档整理自 LLaMA-3.2 官方公开配置与 Transformer Decoder 架构通用推理原理，用于个人学习笔记。*
