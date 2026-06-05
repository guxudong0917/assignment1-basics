---
tags:
  - CS336
  - Transformer
  - Architecture
  - Attention
  - LanguageModel
---

# Transformer LM 架构笔记（补充版）

## 相关笔记

- [[CS336 笔记索引|CS336 笔记索引]]
- [[BPE分词笔记|BPE 分词笔记]]
- [[Transformer训练笔记|Transformer 训练笔记]]
- [[Transformer配图素材|Transformer 配图素材]]

这份笔记按 Transformer Language Model 的数据流整理，同时尽量保留理解过程中的直觉类比。

整体流程是：

```text
token IDs
-> embedding
-> 多层 Transformer blocks
   -> Q/K projection
   -> RoPE on Q/K
-> final RMSNorm
-> linear / LM head
-> logits
```

注意：如果使用绝对位置编码，位置向量通常会加到 embedding 上；但 RoPE 不是加在 embedding 后面的一步，而是在每一层 attention 内部作用到 Q 和 K 上。

其中每个 Transformer block 内部主要由两部分组成：

```text
Multi-Head Self-Attention
Feed-Forward Network
```

可以先用一句话抓住整体分工：

> Attention 负责让每个 token 从上下文中提取信息，FFN 负责对每个位置已经聚合到的信息做非线性加工。

---

## 0. 任务定义：Transformer LM 到底在预测什么

语言模型的输入是一段 token ID 序列：

```text
input_ids: batch_size x sequence_length
```

模型输出：

```text
logits: batch_size x sequence_length x vocab_size
```

也就是说，模型会对每一个位置都输出一个 vocabulary 上的预测分布。

训练时通常是 next-token prediction：

```text
输入:   t0, t1, t2, ..., t_{T-1}
目标:   t1, t2, t3, ..., t_T
```

所以第 0 个位置预测第 1 个 token，第 1 个位置预测第 2 个 token，依此类推。

推理生成时则通常只取最后一个位置的 logits：

```text
logits[:, -1, :]
```

用它来采样或选择下一个 token，然后把新 token 接到输入后面，继续下一步生成。

---

## 0.1 sequence_length 和 context_length

这两个概念容易混。

```text
sequence_length: 当前这次输入序列的实际长度
context_length: 模型最多支持的上下文长度
```

一般有：

```text
sequence_length <= context_length
```

训练时常常直接把文本切成固定长度块，所以：

```text
sequence_length = context_length
```

例如 resource accounting 里如果给：

```text
context_length = 1024
```

并且题目说假设输入有 context_length 个 token，那么就令：

```text
T = sequence_length = 1024
```

推理时则可能输入更短，比如模型最大支持 1024，但当前 prompt 只有 120 个 token，那么：

```text
sequence_length = 120
context_length = 1024
```

---

## 0.2 PyTorch 中的行向量约定和 einsum

数学里常写列向量：

$$
y = Wx
$$

其中：

$$
W \in \mathbb{R}^{d_{out}\times d_{in}}, \quad x \in \mathbb{R}^{d_{in}}
$$

但 PyTorch 中输入通常把 batch 维放在前面，最后一维才是 feature：

```text
x: (..., d_in)
```

如果 Linear 的权重存成：

```text
W: (d_out, d_in)
```

那么 forward 实际上要写：

$$
y = xW^T
$$

代码上是：

```python
out = x @ weight.T
```

或者用 einsum 表达得更清楚：

```python
out = einsum(x, weight, "... d_in, d_out d_in -> ... d_out")
```

这也是为什么 Transformer 代码里 einsum 很有用。

它不是为了装复杂，而是为了避免每一步都在脑子里想：

- 这个矩阵到底要不要转置？
- batch 维会不会被乘掉？
- sequence 维应该保留在哪？
- head 维是不是被当作 batch-like 维？

读 einsum 时只需要看：

> 输出里出现的维度保留；输入里出现但输出里没有的维度会被求和消掉。

例如 Linear：

```text
... d_in, d_out d_in -> ... d_out
```

其中 `d_in` 消失了，说明沿输入特征维求和；`d_out` 出现了，说明输出特征维变成 `d_out`。

---

## 0.3 nn.Module、Parameter 和 buffer

CS336 里要求很多模块自己实现，但依然要继承 `nn.Module`。

原因是：继承 `nn.Module` 后，PyTorch 才能自动管理：

- 参数；
- 子模块；
- 保存和加载 `state_dict`；
- `.to(device)` 设备迁移；
- 训练/推理模式；
- optimizer 能看到哪些参数。

只要写：

```python
class Linear(nn.Module):
    def __init__(self, ...):
        super().__init__()
```

就一定要在 `__init__` 开头调用：

```python
super().__init__()
```

否则父类 `nn.Module` 里面用来注册参数、子模块、buffer 的机制还没初始化。

### Parameter

如果某个 tensor 是可训练参数，就用：

```python
self.weight = nn.Parameter(...)
```

例如：

- `Linear.weight`
- `Embedding.weight`
- `RMSNorm.gain`

这些会出现在：

```python
model.parameters()
```

optimizer 会更新它们。

### Buffer

如果某个 tensor 是模型 forward 需要用到，但不应该被训练，就用：

```python
self.register_buffer("name", tensor)
```

典型例子是 RoPE 里的 sin/cos cache。

它们是公式算出来的固定值，不应该被 optimizer 更新；但又需要跟着模型一起移动到 GPU。所以应该用 buffer，而不是普通属性，也不是 `nn.Parameter`。

如果写：

```python
self.register_buffer("cos_cached", cos, persistent=False)
```

那么 forward 里直接取：

```python
cos = self.cos_cached[token_positions]
```

`persistent=False` 表示这个 buffer 不保存进 `state_dict`，因为它可以根据 `theta, d_k, max_seq_len` 重新计算。

---

## 1. Embedding

语言模型的输入不是字符串，而是 token ID 序列。

例如：

```text
[10, 429, 438, 259, ...]
```

Embedding 层的作用是根据每个 token ID 查表，取出对应的向量：

```text
token_id -> embedding vector
```

如果 embedding matrix 是：

```text
E: vocab_size x d_model
```

那么输入 token ID `i` 对应的 embedding 就是：

```text
E[i]
```

于是输入：

```text
batch_size x sequence_length
```

经过 embedding 后变成：

```text
batch_size x sequence_length x d_model
```

也就是：

$$
(B,T) \rightarrow (B,T,d_{model})
$$

### 1.1 Embedding 向量的意义

Embedding 向量不是人工定义的词义表，而是训练出来的连续表示。

可以把它理解为：

> 模型给每个 token 分配的初始语义坐标。

这个坐标的每个维度本身不一定有明确的人类语言含义，但整体向量会在训练中逐渐学到某些统计规律。

例如语义相近、用法相近、上下文相近的 token，它们的 embedding 可能会更接近。

不过在 Transformer 中，embedding 只是第一步。真正的上下文含义会在后续 attention 和 FFN 中不断更新。

### 1.2 Embedding 和 Linear 的区别

Embedding 不是矩阵乘法，而是查表。

```text
token_ids: (B, T)
weight:    (vocab_size, d_model)
output:    (B, T, d_model)
```

实现上类似：

```python
output = weight[token_ids]
```

而 Linear 是对最后一维做线性变换：

```text
x:      (..., d_in)
weight: (d_out, d_in)
output: (..., d_out)
```

这两者都产生向量表示，但操作方式不同。

### 1.3 初始化

CS336 中 Embedding 使用截断正态分布初始化：

$$
\mathcal{N}(0,1)
$$

并截断在：

$$
[-3,3]
$$

---

## 2. Linear 模块

Transformer 里大量操作都是 Linear：

- Q projection
- K projection
- V projection
- output projection, `W_O`
- FFN 里的 `W_1, W_2, W_3`
- LM head

CS336 里要求自己实现无 bias 的 Linear。

数学上写：

$$
y = Wx
$$

其中：

```text
W: d_out x d_in
x: d_in
```

但 PyTorch 中输入通常是：

```text
x: (..., d_in)
```

所以实现时：

```python
out = x @ W.T
```

或者：

```python
out = einsum(x, W, "... d_in, d_out d_in -> ... d_out")
```

### 2.1 为什么不带 bias

现代 LLM 中很多 Linear 层不使用 bias。这样参数更少，实现更简洁，而且通常效果足够好。

### 2.2 Linear 初始化

Linear 权重用截断正态分布：

$$
W \sim \mathcal{N}(0,\sigma^2)
$$

其中：

$$
\sigma^2=\frac{2}{d_{in}+d_{out}}
$$

即：

$$
\sigma=\sqrt{\frac{2}{d_{in}+d_{out}}}
$$

截断范围是：

$$
[-3\sigma,3\sigma]
$$

实现时用：

```python
torch.nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3*std, b=3*std)
```

---

## 3. Transformer Block

一个 Transformer LM 通常堆叠多个 Transformer block。

每一层 block 接收：

```text
x: batch_size x sequence_length x d_model
```

输出同样形状的 tensor：

```text
batch_size x sequence_length x d_model
```

这样多层 block 可以串联起来。

### 3.1 Post-Norm 和 Pre-Norm

早期 Transformer 常用 post-norm：

```text
x = LayerNorm(x + Sublayer(x))
```

也就是先做子层，再残差连接，最后归一化。

现代大模型更常用 pre-norm：

```text
x = x + Sublayer(Norm(x))
```

也就是先归一化，再送入子层，最后做残差连接。

一个典型 pre-norm Transformer block 是：

```text
x = x + MHA(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

也可以写成：

$$
y=x+\mathrm{MHA}(\mathrm{RMSNorm}(x))
$$

$$
z=y+\mathrm{FFN}(\mathrm{RMSNorm}(y))
$$

### 3.2 为什么现在常用 Pre-Norm

Pre-norm 的好处是训练更稳定。

残差路径几乎是直接贯穿整个网络的：

```text
x -> x + something
```

梯度可以沿着残差路径更容易地向前传播。对于深层 Transformer，这通常比 post-norm 更稳定。

Post-norm 在较浅模型中可以正常工作，但当层数很深时更容易出现训练不稳定。

可以把 pre-norm 的残差流理解为：

> 主干通道一直保留原始信息，每个子层只是往主干上添加一部分修正量。

### 3.3 每个模块大致在做什么

MHA 模块：

> 让每个 token 去看上下文里的其他 token，从全文中提取它需要的信息。

FFN 模块：

> 对每个位置已经聚合到的信息做非线性变换，通常先升维，再激活/门控，再降维。

可以粗略理解为：

- Attention 负责“从哪里拿信息”。
- FFN 负责“拿到信息后怎么加工”。

---

## 4. RMSNorm 和 LayerNorm

Norm 层的作用是控制激活值的尺度，让训练更稳定。

它们一般都沿着最后一维做，也就是对每个 token 的 hidden vector 单独做：

```text
x[b, t, :]
```

如果输入是：

```text
(B, T, d_model)
```

输出还是：

```text
(B, T, d_model)
```

### 4.1 LayerNorm

LayerNorm 对每个 token 的 hidden vector 做归一化。

对于一个向量：

```text
x = [x1, x2, ..., xd]
```

LayerNorm 会计算均值和方差：

```text
mean = average(x)
var = average((x - mean)^2)
```

然后归一化：

```text
y = (x - mean) / sqrt(var + eps)
```

最后再乘上可学习参数并加上 bias：

```text
output = gamma * y + beta
```

公式是：

$$
\mathrm{LayerNorm}(x_i)=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\gamma_i+\beta_i
$$

LayerNorm 做了两件事：

- 减均值，让向量中心接近 0；
- 除标准差，让整体尺度稳定。

### 4.2 RMSNorm

RMSNorm 不减去均值，只用均方根控制尺度。

计算：

```text
rms = sqrt(mean(x^2) + eps)
```

然后：

```text
y = x / rms
output = gamma * y
```

公式是：

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
$$

$$
\mathrm{RMSNorm}(x_i)=\frac{x_i}{\mathrm{RMS}(x)}g_i
$$

RMSNorm 通常没有 bias，也不做 mean centering。

### 4.3 为什么 RMSNorm 现在用得更多

RMSNorm 相比 LayerNorm 更简单：

- 少算一个 mean；
- 少一个 bias 参数；
- 计算更便宜；
- 在很多大语言模型中效果足够好。

直观上，RMSNorm 主要关心：

> 当前 hidden vector 的整体尺度是不是太大或太小。

而不强制把均值归零。

对于现代 pre-norm Transformer，这种尺度控制通常已经足够。

### 4.4 实现时为什么要 upcast 到 float32

RMSNorm 中会计算平方：

```text
x^2
```

如果输入是 float16/bfloat16，平方时可能出现数值问题。

所以通常实现时：

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# compute rmsnorm
return result.to(in_dtype)
```

也就是内部计算用 float32，输出再转回原 dtype。

### 4.5 RMSNorm 初始化

RMSNorm 的 gain 参数初始化为全 1：

$$
g=\mathbf{1}
$$

这样一开始不会额外改变尺度，只负责正常归一化。

---

## 5. Attention 注意力机制

Attention 的核心问题是：

> 对于当前位置的 token，它应该从上下文中的哪些 token 提取信息？

Self-attention 中，每个 token 会产生三种向量：

```text
Query:  Q
Key:    K
Value:  V
```

它们都由当前 hidden states 线性变换得到。

假设输入序列有三个 token：

```text
x0, x1, x2
```

每个 `xi` 都是一个行向量：

```text
xi: 1 x d_model
```

把它们堆起来：

```text
X: sequence_length x d_model
```

得到：

```text
Q = X W_q^T
K = X W_k^T
V = X W_v^T
```

如果有 batch：

```text
X: B x T x d_model
Q/K/V: B x T x d_k
```

### 5.1 Query：我想找什么信息

Query 由：

```text
q_i = x_i W_q^T
```

得到。

如果把 `W_q` 的每一行看成一个扫描器，那么 `q_i` 的每个维度可以理解为：

> 当前 token 在多大程度上需要某一类信息。

例如：

```text
q0[0] = 0.3
```

可以类比成：

> x0 这个 token 对“名词相关信息”的需求程度是 0.3。

```text
q0[1] = 0.8
```

可以类比成：

> x0 对“与高度/程度相关的信息”的需求程度是 0.8。

这里的“名词”“高度”只是帮助理解的例子。真实模型中的每个维度通常没有这么清晰的人类语义，它们是训练中自动形成的抽象特征。

更准确地说：

> q_i 的每个维度像是在提出一个抽象问题，而这个问题具体是什么，是训练学出来的。

### 5.2 Key：我能提供什么索引信息

Key 由：

```text
k_i = x_i W_k^T
```

得到。

Key 可以理解为：

> 当前 token 对外展示的索引标签，告诉其他 token：我这里有什么类型的信息。

例如：

```text
k1[0] = 0.2
```

可以类比成：

> x1 和“名词相关信息”的匹配程度是 0.2。

```text
k1[1] = 0.9
```

可以类比成：

> x1 和“高度/程度相关信息”的匹配程度是 0.9。

所以 Query 和 Key 可以粗略看成一问一答：

```text
Query: 我需要什么？
Key:   我这里有什么？
```

### 5.3 为什么 Q 和 K 的同一维能对齐

一个自然的问题是：

> 为什么 q[0] 和 k[0] 可以认为是在讨论同一类事情？

答案是：这是训练学出来的。

模型并不是一开始就知道 q[0] 和 k[0] 该对齐什么语义，而是在训练过程中，如果某种 Q/K 对齐方式能降低 loss，它就会被强化；如果 Q/K 的维度完全对不上，attention 就提取不到有用信息，loss 会变大。

所以 Q 和 K 的语义对齐不是人工规定的，而是由训练目标逼出来的。

类比可以继续这样理解：

> q 像问题表，k 像答案索引表。训练会逼着它们把同一列逐渐学成“能对话”的抽象方向。

### 5.4 QK^T：计算谁应该关注谁

把所有 query 放成矩阵：

```text
Q =
q0
q1
q2
```

把所有 key 放成矩阵：

```text
K =
k0
k1
k2
```

那么：

```text
Q K^T
```

会得到一个 attention score matrix：

```text
score[i, j] = q_i · k_j
```

含义是：

> 第 i 个 token 想要的信息，在第 j 个 token 那里有多少。

如果 `q_i · k_j` 很大，说明第 i 个 token 应该更关注第 j 个 token。

shape 是：

```text
Q:     T x d_k
K^T:   d_k x T
score: T x T
```

有 batch 和 head 时：

```text
Q:     B x H x T x d_head
K:     B x H x T x d_head
score: B x H x T x T
```

所以 attention score 的两个 `T` 分别表示：

```text
query position x key position
```

每一行表示一个 query token 对所有 key token 的关注分数。

### 5.5 为什么要除以 sqrt(d_k)

Attention score 通常计算为：

```text
score = Q K^T / sqrt(d_k)
```

原因是当 `d_k` 很大时，点积的数值尺度容易变大。

如果 score 太大，经过 softmax 后会变得非常尖锐：

```text
[0.01, 0.02, 20.0] -> softmax -> [接近 0, 接近 0, 接近 1]
```

这样梯度会变差，模型训练不稳定。

除以 `sqrt(d_k)` 是为了把点积尺度拉回一个更合理的范围。

### 5.6 Softmax 和数值稳定

Softmax 把 score 转成概率分布：

$$
\mathrm{softmax}(v)_i=\frac{\exp(v_i)}{\sum_j\exp(v_j)}
$$

但如果 `v_i` 很大，`exp(v_i)` 可能溢出成 `inf`。

所以实现 softmax 时通常先减最大值：

```python
x = x - x.max(dim=dim, keepdim=True).values
```

因为：

$$
\mathrm{softmax}(x)=\mathrm{softmax}(x-c)
$$

对所有元素减同一个常数，不改变 softmax 结果。

通常取：

```text
c = max(x)
```

这样最大值变成 0，可以避免 `exp` 爆掉。

### 5.7 Causal Mask：不能偷看未来

Decoder-only LM 训练时，每个位置只能看自己和过去 token，不能看未来 token。

如果序列长度为 4，causal mask 可以写成：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

也就是：

```text
mask[i, j] = True   if j <= i
mask[i, j] = False  if j > i
```

在 softmax 前，对 mask 为 False 的位置加上：

```text
-inf
```

这样 softmax 后这些位置概率就是 0。

直觉是：

> 第 i 个 token 做 next-token prediction 时，不能提前看到 i 后面的真实 token，否则训练目标就被泄露了。

### 5.8 Value：真正被搬运的信息

Value 由：

```text
v_i = x_i W_v^T
```

得到。

如果说 Key 是“索引”，那么 Value 就是：

> 当前 token 真正提供给别人的内容。

Attention 权重经过 softmax 后得到：

```text
A = softmax(QK^T / sqrt(d_k))
```

其中第 i 行：

```text
A[i]
```

表示第 i 个 token 对所有 token 的关注程度。

最后：

```text
output_i = A[i,0] v0 + A[i,1] v1 + A[i,2] v2 + ...
```

也就是：

> 第 i 个 token 按照注意力权重，从全文的 value 中加权提取信息。

矩阵形式：

```text
A:      T x T
V:      T x d_v
output: T x d_v
```

有 batch 和 head 时：

```text
A:      B x H x T x T
V:      B x H x T x d_head
output: B x H x T x d_head
```

因此 attention 的完整流程是：

```text
Q: 我想找什么
K: 我这里有什么索引
QK^T: 谁和谁匹配
/sqrt(d_k): 控制分数尺度
mask: 防止偷看未来
softmax: 变成关注比例
V: 真正被取走的内容
A V: 聚合上下文信息
```

一句话：

> Q/K 决定关注谁，V 决定被搬运什么内容。

---

## 6. 位置编码

Attention 本身不天然知道顺序。

如果没有位置编码，输入：

```text
x0, x1, x2
```

和输入：

```text
x0, x2, x1
```

在 self-attention 看来只是元素顺序换了。模型可以得到对应换位后的输出，但它不知道“第 1 个 token”和“第 2 个 token”的位置意义。

这会导致模型难以理解：

- 谁在前，谁在后；
- 当前 token 到另一个 token 距离多远；
- 语言中的顺序关系。

所以 Transformer 需要位置编码。

### 6.1 绝对位置编码

经典 Transformer 使用 sinusoidal absolute positional encoding。

每个位置 `k` 有一个位置向量：

```text
p_k: d_model
```

其中：

```text
p_k[2i]     = sin(k / theta_i)
p_k[2i + 1] = cos(k / theta_i)
```

或者更常见地写成：

$$
p_k[2i]=\sin\left(\frac{k}{10000^{2i/d_{model}}}\right)
$$

$$
p_k[2i+1]=\cos\left(\frac{k}{10000^{2i/d_{model}}}\right)
$$

可以理解为每两个维度组成一个二维向量：

```text
(sin angle, cos angle)
```

当位置 `k` 改变时，这个二维向量就在单位圆上旋转。

不同的维度对使用不同频率，因此旋转周期不同。

如果 `d_model=128`，就可以理解为有 64 个不同频率的“时钟”：

- 有的转得很快，像秒针；
- 有的转得慢一点，像分针；
- 有的周期很长，像时针。

多个频率组合起来，就能表示不同位置。

这个“秒针、分针、时针”的类比非常好：

> 单独一个时钟可能会循环，不能唯一表示很长的位置；但很多个转速不同的时钟组合在一起，就能给位置形成更丰富的指纹。

### 6.2 为什么是 sin 和 cos

sin/cos 的一个重要性质是：

> 平移后的位置编码可以由原位置编码通过一个只和相对距离有关的旋转矩阵表示。

对于某一对维度，设：

$$
\theta=\frac{k}{10000^{2i/d}}
$$

$$
\phi=\frac{r}{10000^{2i/d}}
$$

原位置是：

$$
\begin{bmatrix}
\sin\theta\\
\cos\theta
\end{bmatrix}
$$

平移 `r` 后是：

$$
\begin{bmatrix}
\sin(\theta+\phi)\\
\cos(\theta+\phi)
\end{bmatrix}
$$

展开：

$$
\sin(\theta+\phi)=\sin\theta\cos\phi+\cos\theta\sin\phi
$$

$$
\cos(\theta+\phi)=\cos\theta\cos\phi-\sin\theta\sin\phi
$$

所以：

$$
\begin{bmatrix}
\sin(\theta+\phi)\\
\cos(\theta+\phi)
\end{bmatrix}
=
\begin{bmatrix}
\cos\phi & \sin\phi\\
-\sin\phi & \cos\phi
\end{bmatrix}
\begin{bmatrix}
\sin\theta\\
\cos\theta
\end{bmatrix}
$$

这说明：

```text
p_{k+r}
```

可以由：

```text
p_k
```

经过一个只和相对距离 `r` 有关的旋转矩阵得到。

这就是 sinusoidal encoding 的关键优点：

> 它不仅提供绝对位置，也天然带有相对位置结构。

注意：这里矩阵右上角是 `+sin`，左下角是 `-sin`，是因为这一对维度按 `[sin, cos]` 排列。标准线代中常见的 `[cos, sin]` 顺序会得到另一种符号形式。

### 6.3 RoPE

RoPE 是 Rotary Position Embedding。

它不直接把位置向量加到 token embedding 上，而是把位置信息注入到 Q 和 K 中。

具体来说，得到 Q 和 K 后，对每个 token 位置 `n`，把它的 Q/K 向量按维度两两分组：

```text
(x0, x1), (x2, x3), ...
```

然后每一组按照位置 `n` 旋转一个角度：

```text
n * theta_i
```

不同维度对的 `theta_i` 不同，所以它们的旋转速度不同。

这里可以继续沿用“时钟”的类比：

> 每一对维度像一块小表盘，不同表盘转速不同。token 在第几个位置，就让它的 Q/K 在这些表盘上转过对应角度。

### 6.3.1 RoPE 为什么能表达相对位置

RoPE 的关键性质是：

> 两个向量同时旋转同一个角度，它们的内积不变。

也就是说：

```text
R(a)q · R(a)k = q · k
```

当 query 在位置 `m`，key 在位置 `n` 时：

```text
q_m = R(m) q
k_n = R(n) k
```

做内积：

```text
R(m)q · R(n)k
```

如果同时把两个位置都平移 `r`：

```text
R(m+r)q · R(n+r)k
```

两者共同多出来的那一段旋转会抵消，因此 attention score 主要和相对距离有关。

更准确地说，RoPE 让 Q/K 点积可以写成依赖相对位置 `m-n` 的形式。

所以 RoPE 很适合 attention：attention 本来就通过 Q/K 点积计算匹配程度，而 RoPE 正好把相对位置信息写进这个点积里。

### 6.3.2 为什么 RoPE 只作用在 Q 和 K

Attention score 来自：

```text
Q K^T
```

位置关系主要影响的是：

> 第 i 个 token 应该关注第 j 个 token 的程度。

这个关注程度由 Q/K 决定。

V 是被加权搬运的内容，不负责决定关注权重。因此 RoPE 通常只旋转 Q 和 K，不旋转 V。

一句话：

> RoPE 改的是“谁关注谁”的匹配过程，不是“被搬运的内容”。

### 6.4 RoPE 的具体实现

实现 RoPE 时，不需要真的构造巨大的旋转矩阵。

数学上可以写：

$$
x' = R_n x
$$

其中 `R_n` 是一个 `d_k x d_k` 的旋转矩阵。

但这个大矩阵其实是很多个 `2 x 2` 小旋转矩阵组成的 block diagonal matrix。

所以工程上只需要对每两个维度一组做旋转。

### 6.4.1 预计算 angles

通常提前构造：

```text
positions:   context_length
inv_freq:    d_k / 2
```

其中：

$$
\mathrm{inv\_freq}_i=\Theta^{-2i/d_k}
$$

角度是：

$$
angle[n,i]=n\cdot \mathrm{inv\_freq}_i
$$

这其实就是把角度拆成两部分：

```text
位置 n:        只和 token position 有关
频率 inv_freq: 只和维度对 i 有关
```

代码直觉：

```python
positions = torch.arange(max_seq_len)  # (max_seq_len,)
inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2) / d_k))  # (d_k/2,)
angles = positions[:, None] * inv_freq[None, :]  # (max_seq_len, d_k/2)
```

然后预计算：

```python
cos_cached = torch.cos(angles)
sin_cached = torch.sin(angles)
```

这两个 tensor 只由位置和维度决定，和具体输入内容无关，所以可以复用在不同 batch、不同 layer 中。

### 6.4.2 为什么用 register_buffer

`cos_cached` 和 `sin_cached` 不是可学习参数，不应该被 optimizer 更新。

所以不应该写成：

```python
nn.Parameter(cos_cached)
```

而应该写成：

```python
self.register_buffer("cos_cached", cos_cached, persistent=False)
self.register_buffer("sin_cached", sin_cached, persistent=False)
```

这样它们会跟着模型 `.to(device)` 移动，但不会出现在 `model.parameters()` 里。

### 6.4.3 forward 中如何取出对应位置

RoPE 的 forward 输入通常是：

```text
x: (..., seq_len, d_k)
token_positions: (..., seq_len)
```

其中 `...` 表示任意 batch-like 维度，比如：

```text
batch
batch x num_heads
```

根据 token_positions 取出需要的 cos/sin：

```python
cos = self.cos_cached[token_positions]
sin = self.sin_cached[token_positions]
```

如果：

```text
token_positions: (seq_len,)
```

那么：

```text
cos: (seq_len, d_k/2)
```

如果：

```text
token_positions: (batch, seq_len)
```

那么：

```text
cos: (batch, seq_len, d_k/2)
```

### 6.4.4 奇偶维旋转

把输入拆成偶数维和奇数维：

```python
x_even = x[..., 0::2]
x_odd  = x[..., 1::2]
```

一种常见 `[cos, sin]` 约定下的旋转可以写成：

```python
rot_even = x_even * cos - x_odd * sin
rot_odd  = x_even * sin + x_odd * cos
```

如果使用 `[sin, cos]` 排列，符号形式可能变成：

```python
rot_even = x_even * cos + x_odd * sin
rot_odd  = -x_even * sin + x_odd * cos
```

最后再把 even/odd 交错放回原来的维度顺序。

注意：不同代码库可能采用不同的维度排列约定，例如 `[cos, sin]` 顺序，或者使用 `rotate_half`。这时符号可能看起来不同。

关键是：

> Q 和 K 必须使用同一套旋转约定，并且要和测试/框架定义一致。

### 6.4.5 token_positions 和 KV cache

训练时，token positions 通常就是：

```text
[0, 1, 2, ..., seq_len-1]
```

但推理时如果使用 KV cache，当前 forward 可能只输入一个新 token：

```text
seq_len = 1
```

但这个 token 在完整上下文里的真实位置可能是 128。

所以这时：

```text
token_positions = [128]
```

这也是为什么 RoPE forward 不应该永远假设位置从 0 开始，而应该显式接收 `token_positions`。

---

## 7. Multi-Head Attention

单头 attention 会让每个 token 用一套 Q/K/V 去看全文。

Multi-head attention 的想法是：

> 不同 head 可以从不同角度看上下文。

如果：

```text
d_model = 512
num_heads = 16
```

那么每个 head 的维度通常是：

```text
d_head = 512 / 16 = 32
```

输入先被投影成：

```text
Q, K, V: sequence_length x d_model
```

再 reshape 成：

```text
num_heads x sequence_length x d_head
```

有 batch 时：

```text
B x T x d_model
-> B x H x T x d_head
```

每个 head 独立做 attention。

### 7.1 多头和单头的区别

多头不是把文本分成不同片段看。

每个 head 仍然可以看全局上下文，只是：

- 每个 head 使用不同的参数；
- 每个 head 看到的是不同的子空间；
- 每个 head 计算出的 attention 分数不同。

可以理解为：

> 同一个 token，在不同 head 中会问不同的问题。

例如某个 head 可能更关注语法关系，另一个 head 可能更关注实体指代，还有一个 head 可能更关注局部搭配。

这些语义不是人工指定的，而是训练中自动形成的。

### 7.2 为什么多头不是简单拆分后再加回来

如果只是把一个完整的 `q` 和 `k` 切成几段，然后把每段点积再加起来：

$$
q\cdot k=\sum_h q^{(h)}\cdot k^{(h)}
$$

那确实不会带来本质区别。

但多头注意力不是这样。

多头里每个 head 会单独算 attention score，并且单独 softmax：

$$
\alpha^{(h)}=\mathrm{softmax}(Q^{(h)}K^{(h)T})
$$

所以每个 head 有自己的一套 attention pattern。

关键区别是：

```text
单头: 一套 attention 分布
多头: 多套 attention 分布
```

因此同一个 token 可以在不同 head 中同时关注不同位置、不同关系。

### 7.3 Attention score 仍然是 T x T

每个 head 中：

```text
Q_h: T x d_head
K_h: T x d_head
```

所以：

```text
Q_h K_h^T: T x T
```

也就是说，每个 head 仍然对所有位置计算注意力分数。

切分后变小的是 Value 的内容维度：

```text
V_h: T x d_head
```

所以每个 head 输出：

```text
T x d_head
```

所有 head 拼接后：

```text
T x d_model
```

### 7.4 多头输出和 W_o

每个 head 得到：

```text
sequence_length x d_head
```

把所有 heads 拼接：

```text
sequence_length x d_model
```

再经过 output projection：

```text
W_o
```

把多个 head 的信息重新混合。

注意：

> W_o 不是简单为了降维，因为 concat 后本来已经回到了 d_model。它更重要的作用是混合不同 head 的信息。

可以类比为：

```text
每个 head 提交一份观察报告；
concat 是把报告摞在一起；
W_o 是总编辑，把不同报告重新整理成统一表示。
```

### 7.5 为什么 RoPE 要在切 head 后做

RoPE 的旋转维度应该是每个 head 内部的：

```text
d_head
```

而不是整个：

```text
d_model
```

例如：

```text
d_model = 768
num_heads = 12
d_head = 64
```

RoPE 应该对每个 head 的 64 维向量两两旋转：

```text
(0,1), (2,3), ..., (62,63)
```

每个 head 都从自己的第 0 对维度开始使用同一套频率。

如果直接把 768 维整体当成一个大向量做 RoPE，那么第 2 个 head 会接着第 1 个 head 的维度继续算频率，这通常不是标准 multi-head RoPE 的定义。

所以实际流程一般是：

```text
X -> Q/K/V projection
Q/K/V reshape to (B, H, T, d_head)
apply RoPE to Q and K
attention
concat heads
W_o
```

### 7.6 num_heads 和 FLOPs 的关系

标准 MHA 中：

$$
d_{head}=\frac{d_{model}}{H}
$$

因此主矩阵乘法 FLOPs 和 `num_heads` 在公式上会抵消。

每层 MHA 的 FLOPs 是：

$$
8Td_{model}^2+4T^2d_{model}
$$

其中：

- `8Td_model^2`：Q/K/V/O 四个 projection；
- `4T^2d_model`：`QK^T` 和 `AV`。

这里没有出现 `H`。

这说明：

> num_heads 改变的是注意力分布的数量和每个 head 的子空间维度，而不是标准 FLOPs 公式中的总主计算量。

当然，实际工程效率仍然会受 head 数、kernel 实现、memory layout 影响。

---

## 8. FFN

Attention 之后，每个位置已经从上下文中提取了一些信息。

FFN 的作用是对每个位置单独做非线性加工。

注意：

> FFN 不在 sequence 维度上混合信息。

它对每个 token position 独立使用同一套参数。

所以它叫 position-wise FFN。

如果输入是：

```text
B x T x d_model
```

FFN 是对每个：

```text
x[b, t, :]
```

单独应用同一个网络。

Attention 负责 token 之间交流；FFN 负责每个 token 内部加工。

### 8.1 经典 FFN

早期 Transformer 使用：

```text
FFN(x) = W_2 ReLU(W_1 x)
```

通常先升维再降维：

```text
d_model -> d_ff -> d_model
```

原始 Transformer 中 `d_ff` 常常是：

```text
4 * d_model
```

升维可以理解为给模型更大的中间空间，让它对当前 token 的表示做更丰富的非线性变换。

### 8.2 SiLU

SiLU 激活函数是：

```text
SiLU(x) = x * sigmoid(x)
```

公式：

$$
\mathrm{SiLU}(x)=x\sigma(x)=\frac{x}{1+e^{-x}}
$$

相比 ReLU，SiLU 更平滑，并且不会像 ReLU 那样直接把负数硬截断为 0。

可以理解为：

```text
ReLU 像硬开关。
SiLU 像软旋钮。
```

### 8.3 GLU 门控

GLU 的思想是引入一个 gate：

```text
output = value * gate
```

其中 gate 控制哪些信息应该通过，哪些信息应该被压制。

类比：

> 一条分支负责产生内容，另一条分支负责控制阀门开多大。

### 8.4 SwiGLU

SwiGLU 把 SiLU 和 GLU 结合起来。

一种常见形式是：

```text
SwiGLU(x) = W_2( SiLU(W_1 x) * W_3 x )
```

其中：

```text
W_1: d_ff x d_model
W_3: d_ff x d_model
W_2: d_model x d_ff
```

`W_1 x` 经过 SiLU 产生一个门控信号，`W_3 x` 产生被门控的内容，两者逐元素相乘，再通过 `W_2` 投影回 `d_model`。

所以 SwiGLU FFN 有三组线性参数：

```text
W_1, W_2, W_3
```

而经典 ReLU FFN 只有两组：

```text
W_1, W_2
```

### 8.5 d_ff 为什么不是 4d_model

原始 ReLU FFN 常用：

```text
d_ff = 4 * d_model
```

但 SwiGLU 有三个线性层，如果仍然用 `4d_model`，参数和计算量会更大。

所以很多现代 LLM 会用较小的中间维度。

CS336 中使用：

$$
d_{ff}\approx \frac{8}{3}d_{model}
$$

并且 round 到接近的 64 的倍数。

例如 GPT-2 XL-shaped 配置里：

```text
d_model = 1600
d_ff = 4288
```

而：

$$
\frac{8}{3}\times 1600\approx 4266.67
$$

4288 是附近的 64 倍数。

---

## 9. Final RMSNorm

经过多层 Transformer blocks 后，通常还会接一个 final norm：

```text
x = RMSNorm(x)
```

它的作用是把最终 hidden states 的尺度整理到稳定范围，再送入最后的线性层。

对于 pre-norm Transformer，block 内部每个子层前都有 norm，但最后输出处仍然通常保留一个 final RMSNorm。

---

## 10. Linear / LM Head

最后一层 linear 把每个位置的 hidden vector 转成对 vocabulary 的预测。

如果：

```text
x: batch_size x sequence_length x d_model
```

LM head 权重是：

```text
W: vocab_size x d_model
```

那么输出：

```text
logits: batch_size x sequence_length x vocab_size
```

每个位置对应一个长度为 `vocab_size` 的向量。

这个向量不是概率，而是 logits。

训练时把 logits 送入 cross entropy。

生成时通常对最后一个位置的 logits 做：

```text
softmax(logits / temperature)
```

得到下一个 token 的概率分布，再采样。

---

## 11. Resource Accounting：参数量和 FLOPs

CS336 这一节还要求做 resource accounting，也就是统计参数量和 FLOPs。

这里只统计主要的矩阵乘法 FLOPs，忽略 softmax、norm、RoPE、激活、逐元素乘法、residual add 等较小项。

注意：这里默认统计的是 forward FLOPs。训练时还需要 backward，粗略估计一次 forward + backward 通常约为 forward FLOPs 的 3 倍。

### 11.1 参数量

假设：

```text
L: num_layers
d: d_model
d_ff: FFN hidden dimension
V: vocab_size
```

Embedding 参数：

$$
Vd
$$

每个 Transformer block 中：

Attention 参数，Q/K/V/O 四个矩阵：

$$
4d^2
$$

FFN 参数，SwiGLU 有三个矩阵：

$$
3dd_{ff}
$$

RMSNorm 参数，pre-norm block 里有两个 RMSNorm：

$$
2d
$$

所以每层 block 参数：

$$
4d^2+3dd_{ff}+2d
$$

所有 block：

$$
L(4d^2+3dd_{ff}+2d)
$$

final RMSNorm：

$$
d
$$

LM head：

$$
Vd
$$

如果 embedding 和 LM head 不共享权重，总参数量：

$$
Vd+L(4d^2+3dd_{ff}+2d)+d+Vd
$$

即：

$$
2Vd+L(4d^2+3dd_{ff}+2d)+d
$$

### 11.2 GPT-2 XL-shaped 参数量例子

配置：

```text
vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 4288
```

总参数量：

$$
50257\times1600
+48(2\times1600+4\times1600^2+3\times1600\times4288)
+1600
+50257\times1600
$$

也就是：

$$
1,640,452,800
$$

约 1.64B 参数。

如果使用 fp32，每个参数 4 bytes，仅加载权重需要：

$$
1,640,452,800\times4=6,561,811,200\text{ bytes}
$$

约 6.56 GB，或者约 6.11 GiB。

### 11.3 FLOPs 规则

矩阵乘法：

$$
A\in\mathbb{R}^{m\times n},\quad B\in\mathbb{R}^{n\times p}
$$

那么：

$$
AB
$$

需要：

$$
2mnp
$$

FLOPs。

原因是结果矩阵有 `mp` 个元素，每个元素是长度 `n` 的点积，大约需要 `n` 次乘法和 `n` 次加法。

### 11.4 每层 MHA FLOPs

MHA 分两部分。

Q/K/V/O projection

每个 projection：

$$
2Td^2
$$

四个 projection：

$$
8Td^2
$$

Attention 内部

每个 head：

$$
Q_hK_h^T: 2T^2d_{head}
$$

所有 head：

$$
2T^2d
$$

因为：

$$
H d_{head}=d
$$

再算：

$$
AV: 2T^2d
$$

所以 attention 内部合计：

$$
4T^2d
$$

因此每层 MHA FLOPs：

$$
\boxed{8Td^2+4T^2d}
$$

### 11.5 每层 FFN FLOPs

SwiGLU 有三个 Linear：

```text
W1: d -> d_ff
W3: d -> d_ff
W2: d_ff -> d
```

所以：

$$
2Tdd_{ff}+2Tdd_{ff}+2Td_{ff}d
$$

即：

$$
\boxed{6Tdd_{ff}}
$$

如果近似：

$$
d_{ff}\approx \frac{8}{3}d
$$

那么：

$$
6Td\cdot \frac{8}{3}d=16Td^2
$$

所以可以粗略记：

$$
\mathrm{FFN}\approx 16Td^2
$$

但具体题目里要用实际给出的 `d_ff`，例如 4288。

### 11.6 LM head FLOPs

LM head：

```text
T x d  @  d x vocab_size
```

FLOPs：

$$
2TdV
$$

### 11.7 总 FLOPs

总 forward FLOPs 近似为：

$$
L(8Td^2+4T^2d+6Tdd_{ff})+2TdV
$$

其中：

```text
L: num_layers
T: sequence_length
d: d_model
d_ff: FFN hidden dimension
V: vocab_size
```

### 11.8 GPT-2 XL-shaped FLOPs 例子

代入：

```text
L = 48
T = 1024
d = 1600
d_ff = 4288
V = 50257
```

得到：

$$
48(8\times1024\times1600^2+4\times1024^2\times1600+6\times1024\times1600\times4288)
+2\times1024\times1600\times50257
$$

结果约为：

$$
3.517\times10^{12}
$$

也就是约 3.52 TFLOPs。

### 11.9 为什么大模型里 FFN 常常是大头

每层近似：

$$
\mathrm{MHA}=8Td^2+4T^2d
$$

$$
\mathrm{FFN}\approx16Td^2
$$

如果 `T` 不特别长，而 `d` 很大，那么和 `d^2` 相关的项会占大头。

FFN 的 `16Td^2` 通常比 MHA projection 的 `8Td^2` 还大，所以常规上下文长度下，大模型里 FFN 往往是最大计算部分。

### 11.10 为什么长上下文会让 MHA 占比上升

MHA 中有一项：

$$
4T^2d
$$

它随着 sequence length 二次增长。

FFN 是：

$$
6Tdd_{ff}
$$

只随着 `T` 一次增长。

所以：

> 固定模型大小，只增加 sequence_length 时，MHA 中 attention score 和 AV 的占比会明显上升。

粗略比较 MHA 和 FFN：

$$
\frac{\mathrm{MHA}}{\mathrm{FFN}}
\approx
\frac{8Td^2+4T^2d}{16Td^2}
=
\frac{1}{2}+\frac{T}{4d}
$$

当 `T` 越大，MHA 相对 FFN 的比例越高。

所以这两句话并不矛盾：

```text
常规长度 + 大模型：FFN 往往是大头。
超长上下文：MHA 的 T^2 项会越来越显著。
```

---

## 12. 总结：完整数据流

Transformer LM 的核心数据流可以总结为：

```text
token IDs
-> embedding
-> hidden states
-> repeated Transformer blocks
   -> RMSNorm
   -> Multi-Head Attention
      -> Q/K/V projection
      -> RoPE on Q/K
      -> QK^T / sqrt(d_k)
      -> causal mask
      -> softmax
      -> weighted sum over V
      -> concat heads
      -> W_o
   -> residual
   -> RMSNorm
   -> FFN / SwiGLU
      -> W1 branch + SiLU
      -> W3 branch as content
      -> elementwise gate
      -> W2 projection
   -> residual
-> final RMSNorm
-> LM head
-> logits
```

其中：

- Embedding 把离散 token ID 变成连续向量；
- Position / RoPE 告诉模型顺序和相对位置；
- Attention 让每个 token 从上下文中提取信息；
- Q/K 像“提问和索引”，V 像“真正被搬运的内容”；
- Multi-head 让同一个 token 从多个子空间问不同问题；
- FFN 对每个位置的信息做非线性加工；
- RMSNorm 和 residual 保证深层训练稳定；
- LM head 把 hidden state 转成对下一个 token 的预测 logits。

最终可以用一句话理解：

> Transformer LM 先把 token ID 变成向量，再用带位置感知的多头注意力从上下文中取信息，用 FFN 加工每个位置的表示，最后把每个位置的 hidden state 映射成对下一个 token 的预测。
