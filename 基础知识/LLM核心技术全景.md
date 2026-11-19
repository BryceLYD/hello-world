# LLM 核心技术全景图

---

## 一、这些技术在模型的什么位置

一个大语言模型本质上是一个"积木堆叠"结构，一层 Transformer Block 重复几十次：

```
输入 Token
   │
   ▼
┌─────────────────────────────┐
│  Token Embedding            │ ← 埋层：把 token 变成向量
├─────────────────────────────┤
│  ┌─── Transformer Block ×N ─┐│
│  │                          ││
│  │  ★ 注意力层 (Attention)   ││ ← MLA / GQA / 线性注意力 /
│  │    + RoPE 位置编码        ││    Sliding Window 都在这里
│  │                          ││
│  │  ★ 前馈网络 (FFN / MoE)  ││ ← SwiGLU / 稀疏化 / 专家混合
│  │                          ││
│  └──────────────────────────┘│
├─────────────────────────────┤
│  RMSNorm / LayerNorm         │ ← 归一化（散布在块内）
├─────────────────────────────┤
│  输出层 (lm_head)            │ ← 预测下一个 token
└─────────────────────────────┘

推理时的"外挂"系统（不在模型权重里，但决定体验）：
  ★ KV Cache 管理 ← PagedAttention / KV量化 / MLA压缩
  ★ 采样策略      ← temperature / top-p
  ★ 加速技术      ← 投机解码 / Continuous Batching
```

所以听到的技术可以分成两大类：

- **改模型结构的**（MLA、MoE、线性注意力）→ 影响训练和推理
- **不改结构、只优化推理的**（PagedAttention、投机解码）→ 部署侧技术

---

## 二、发展时间线：每个技术解决什么危机

Transformer 的历史就是一部"哪里疼医哪里"的历史：

| 时间 | 技术 | 解决的危机 |
|---|---|---|
| 2017 | **Transformer / MHA** | RNN 无法并行训练（源头） |
| 2019 | **MQA** | KV Cache 显存爆炸 → 多头共享一份 KV |
| 2020 | **GPT-3 规模定律** | 发现"堆参数"这条路有效 |
| 2021 | RoPE 成为主流 | 绝对位置编码外推差 |
| 2022 | **FlashAttention** | 注意力计算显存瓶颈（不改变数学，只优化访存） |
| 2023 | **GQA**（Llama 2-70B 采用） | MQA 质量损失太大 → 折中方案 |
| 2023 | **Mamba / RWKV / RetNet** | 注意力 O(n²) 复杂度 → 线性递归路线 |
| 2024.05 | **MLA**（DeepSeek-V2） | KV Cache 仍太大 → 把 KV 压缩进低秩潜空间 |
| 2024.12 | **DeepSeek-V3**：MLA + MoE + 无辅助损失负载均衡 | 训练成本大降 |
| 2025 | **混合架构**（Qwen3-Next 等） | 纯线性注意力丢信息 → 3:1 交错"线性层 + 1层全注意力" |
| 进行中 | 深度思考（o系列 / R1）、测试时扩展 | 从"堆参数"转向"堆推理时间" |

**关键洞察**：MLA 和线性注意力是两条并行的路线——
- MLA 是"把全注意力做便宜"（保留精确注意力，压缩缓存）
- 线性注意力是"换一种便宜的近似"（牺牲精确性换 O(n) 复杂度）
- 当前共识是**混合使用**两者。

---

## 三、技术家族树

### 家族 1：注意力机制（核心）

```
注意力机制
├── 精确注意力（每对 token 都算相似度，O(n²)）
│   ├── MHA   —— 多头注意力，始祖，每个头独立 KV
│   ├── MQA   —— 所有头共享 1 份 KV（省显存，质量降）
│   ├── GQA   —— 分组共享（Llama 2/3、Qwen 用它）✓ 生产主流
│   ├── MLA   —— 把 KV 投影到低秩潜向量再缓存（DeepSeek 独创）
│   │           KV Cache 缩小到 MHA 的 1/10 以下，质量不降反升
│   └── 变体：Multi-head Latent + 窗口注意力（混合）
│
├── 线性注意力（O(n) 复杂度，可处理超长序列）
│   ├── RWKV     —— 最早出圈的可并行 RNN 式路线
│   ├── RetNet   —— 微软提出，推理时退化为递归
│   ├── Mamba    —— 选择性状态空间模型（SSM），2023 最热
│   │   └── Mamba-2（SSD 理论，和注意力统一了数学框架）
│   ├── GLA      —— 门控线性注意力
│   └── DeltaNet —— 固定容量的"神经记忆"思想
│
└── 混合架构（2025 的共识答案）
    └── 例如 Qwen3-Next：每 4 层中 3 层用线性注意力 + 1 层用 GQA
        —— 线性层负责"日常信息传递"，全注意力层负责"远距离精确检索"
```

**发展方向：MQA → GQA → MLA 是"KV Cache 减肥三部曲"；Mamba/RWKV 是"造新引擎"；混合架构是"两台引擎都要"。**

### 家族 2：其他结构组件（同属一个 Block 内）

| 位置 | 技术 | 作用 |
|---|---|---|
| 位置编码 | RoPE、YaRN（外推扩展） | 让模型知道 token 顺序；YaRN 让 8K 训练的模型跑 128K |
| FFN 层 | SwiGLU 激活 | 现代模型标配（Llama、Qwen、DeepSeek 全用） |
| FFN 层 | **MoE（专家混合）** | 把大 FFN 拆成多个专家，每次只激活少数 → 参数大、计算省（Mixtral、DeepSeek-V3、Qwen3 有 MoE 版） |
| 归一化 | RMSNorm、Pre-Norm | 训练更稳，已是标配 |
| 整体结构 | 深窄 vs 宽浅、Embedding 共享 | 影响性价比 |

### 家族 3：推理侧技术（不改权重，改服务）

| 技术 | 解决什么 |
|---|---|
| **FlashAttention 1/2/3** | 让注意力算得快、省显存（数学不变，改 GPU 访存模式） |
| **PagedAttention（vLLM）** | KV Cache 显存管理，像操作系统管内存页一样管缓存 |
| KV Cache 量化 | 缓存从 FP16 压到 FP8/INT4 |
| 投机解码 | 小模型起草、大模型验证 → 生成加速 2-3 倍 |
| Continuous Batching | 多用户请求打包调度，吞吐量翻倍 |

### 家族 4：训练与对齐（另一条主线）

```
预训练（Next Token Prediction）
    → SFT（监督微调）
    → RLHF / DPO（对齐人类偏好）
    → 推理时扩展（o1 / R1 的 CoT 强化学习，2025 热点）
```

---

## 四、用 DeepSeek-V3 串一遍（学以致用）

当听说 "DeepSeek-V3 = MLA + DeepSeekMoE + MTP + FP8"，翻译过来就是：

| 技术 | 所在位置 | 效果 |
|---|---|---|
| **MLA** | 注意力层 | KV Cache 缩小 ~10 倍 → 推理便宜 |
| **DeepSeekMoE** | FFN 层 | 6710 亿参数但每 token 只激活 370 亿 → 训练和推理便宜 |
| **MTP（多 Token 预测）** | 输出层 | 训练信号更密集 + 可用于投机解码 |
| **FP8 训练** | 数值精度 | 训练省钱 |

**现代大模型设计的统一哲学：参数可以大，每个 token 的计算和显存必须小。**

MLA 管显存，MoE 管计算，混合注意力管长序列——三者殊途同归。

---

## 附：常见缩写速查

| 缩写 | 全称 | 中文含义 |
|---|---|---|
| MHA | Multi-Head Attention | 多头注意力 |
| MQA | Multi-Query Attention | 多查询注意力 |
| GQA | Grouped-Query Attention | 分组查询注意力 |
| MLA | Multi-head Latent Attention | 多头潜注意力 |
| MoE | Mixture of Experts | 专家混合 |
| SSM | State Space Model | 状态空间模型 |
| RoPE | Rotary Position Embedding | 旋转位置编码 |
| KV Cache | Key-Value Cache | 键值缓存 |
| MTP | Multi-Token Prediction | 多 Token 预测 |
| CoT | Chain-of-Thought | 思维链 |
