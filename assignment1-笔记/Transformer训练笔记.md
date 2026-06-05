---
tags:
  - CS336
  - Transformer
  - Training
  - Optimizer
  - AdamW
  - Memory
  - Decoding
---

# Transformer 训练笔记

## 相关笔记

- [[CS336 笔记索引|CS336 笔记索引]]
- [[BPE分词笔记|BPE 分词笔记]]
- [[Transformer架构笔记|Transformer 架构笔记]]
- [[Transformer配图素材|Transformer 配图素材]]

Transformer 训练可以理解成一条完整链路：

```text
tokenized data
-> get batch
-> model forward
-> logits
-> cross-entropy loss
-> backward
-> 参数梯度
-> gradient clipping
-> AdamW 更新参数和 m/v
-> logging
-> checkpoint
-> decoding 生成文本
```

训练系统里最重要的几件事是：

- loss 如何计算；
- optimizer 如何根据梯度更新参数；
- 显存主要被哪些东西占用；
- 一个 training step 大约需要多少 FLOPs；
- 如何记录实验和保存 checkpoint；
- 训练好的模型如何 decode 生成文本。

---

## 0. Loss：语言模型到底在优化什么

Transformer LM 的训练目标是 next-token prediction。

给定一段 token 序列：

$$
x=(x_1,x_2,\dots,x_{m+1})
$$

语言模型在每个位置 $i$ 都要根据前面的 token 预测下一个 token：

$$
p_\theta(x_{i+1}\mid x_{1:i})
$$

也就是说：

- 给 $x_1$，预测 $x_2$；
- 给 $x_1,x_2$，预测 $x_3$；
- 给 $x_1,x_2,\dots,x_i$，预测 $x_{i+1}$。

所以一段长度为 $m+1$ 的序列，可以产生 $m$ 个 next-token prediction 训练目标。

### 0.1 从最大似然到 Cross-Entropy

语言模型希望真实文本序列出现的概率尽可能大。

对一条序列来说，根据链式法则，模型希望最大化：

$$
\prod_{i=1}^{m}p_\theta(x_{i+1}\mid x_{1:i})
$$

也就是让每个真实 next token 的概率都尽量大。

取 log 后，乘积变成求和：

$$
\sum_{i=1}^{m}\log p_\theta(x_{i+1}\mid x_{1:i})
$$

训练时通常写成最小化 loss，所以取负号：

$$
-\sum_{i=1}^{m}\log p_\theta(x_{i+1}\mid x_{1:i})
$$

如果训练集 $D$ 中有很多条序列，并且每条序列长度都是 $m+1$，那么标准 cross-entropy / negative log-likelihood loss 是：

$$
\ell(\theta;D)=\frac{1}{|D|m}\sum_{x\in D}\sum_{i=1}^{m}-\log p_\theta(x_{i+1}\mid x_{1:i})
$$

这里的 $\frac{1}{|D|m}$ 表示对所有 sequence、所有 token prediction 取平均。

CS336 也是这样定义 Transformer LM 的 cross-entropy loss：对训练集中每条长度为 $m+1$ 的序列，在 $i=1,\dots,m$ 的每个位置计算 $-\log p_\theta(x_{i+1}\mid x_{1:i})$，然后平均。

### 0.2 logits、softmax 和 cross-entropy

模型在每个位置 $i$ 输出一个 logits 向量：

$$
o_i\in\mathbb{R}^{vocab\_size}
$$

这个 logits 还不是概率，而是每个 token 的未归一化分数。

通过 softmax 得到概率：

$$
p_\theta(x_{i+1}\mid x_{1:i})=\mathrm{softmax}(o_i)[x_{i+1}]
$$

也就是：

$$
\mathrm{softmax}(o_i)[k]=\frac{\exp(o_i[k])}{\sum_a\exp(o_i[a])}
$$

因此单个位置的 cross-entropy loss 是：

$$
\ell_i=-\log \mathrm{softmax}(o_i)[x_{i+1}]
$$

展开后：

$$
\ell_i=-o_i[x_{i+1}]+\log\sum_a\exp(o_i[a])
$$

### 0.3 Cross-entropy 实现时的数值稳定

直接计算 softmax 时会出现：

$$
\exp(o_i[k])
$$

如果 logits 很大，比如某些值接近 1000，那么 $\exp(1000)$ 会直接数值溢出。

所以实际实现 cross-entropy 时，不应该先完整计算 softmax，再取 log，而是用更稳定的形式：

$$
\ell_i=-o_i[y_i]+\log\sum_a\exp(o_i[a])
$$

其中 $y_i=x_{i+1}$ 是真实下一个 token。

为了数值稳定，令：

$$
M=\max_a o_i[a]
$$

则：

$$
\log\sum_a\exp(o_i[a])=M+\log\sum_a\exp(o_i[a]-M)
$$

所以稳定版本可以写成：

$$
\ell_i=-o_i[y_i]+M+\log\sum_a\exp(o_i[a]-M)
$$

因为所有 logits 都减去了最大值 $M$，所以指数里面的最大值变成 0，不容易溢出。

CS336 的 cross-entropy 实现也要求：减去最大元素保证数值稳定，并且尽可能消掉不必要的 log 和 exp。

### 0.4 Perplexity：困惑度

Cross-entropy 是训练时直接优化的 loss，但在评估语言模型时，经常还会报告 perplexity，也就是困惑度。

如果一条长度为 $m$ 的预测序列，每个位置的 cross-entropy loss 分别是：

$$
\ell_1,\ell_2,\dots,\ell_m
$$

那么平均 cross-entropy 是：

$$
\bar{\ell}=\frac{1}{m}\sum_{i=1}^{m}\ell_i
$$

perplexity 定义为：

$$
\mathrm{PPL}=\exp\left(\frac{1}{m}\sum_{i=1}^{m}\ell_i\right)
$$

也就是：

$$
\mathrm{PPL}=\exp(\text{average cross-entropy})
$$

CS336 也是这样定义 perplexity：对一个长度为 $m$ 的序列，先计算平均 cross-entropy，再取指数。

### 0.5 困惑度的直观理解

因为单个位置的 loss 是：

$$
\ell_i=-\log p_\theta(x_{i+1}\mid x_{1:i})
$$

如果模型给真实 token 的概率越大，loss 就越小。

假设模型平均给正确 token 的概率是 0.5：

$$
\bar{\ell}=-\log0.5\approx0.693
$$

那么：

$$
\mathrm{PPL}=\exp(0.693)\approx2
$$

可以粗略理解成：

> 模型平均每一步像是在 2 个差不多可能的 token 之间犹豫。

如果模型平均给正确 token 的概率是 0.1：

$$
\bar{\ell}=-\log0.1\approx2.303
$$

那么：

$$
\mathrm{PPL}=\exp(2.303)\approx10
$$

可以理解成：

> 模型平均每一步像是在 10 个候选 token 里猜。

如果平均 loss 是 5：

$$
\mathrm{PPL}=e^5\approx148
$$

说明模型非常困惑，平均每一步像是在 148 个候选 token 中选择。

所以：

- cross-entropy 越低，perplexity 越低；
- perplexity 越低，模型越确定；
- perplexity 越高，模型越迷茫。

### 0.6 Loss 和 Perplexity 的关系

二者本质上是同一个指标的两种表达方式。

Cross-entropy 是 log 空间里的损失：

$$
\bar{\ell}=\frac{1}{m}\sum_i-\log p_i
$$

Perplexity 是把它从 log 空间变回普通尺度：

$$
\mathrm{PPL}=\exp(\bar{\ell})
$$

所以训练时通常优化 cross-entropy，因为它数学性质好、梯度好算；评估时报告 perplexity，因为它更直观。

可以记成：

$$
\boxed{\mathrm{PPL}=\exp(\text{cross-entropy})}
$$

如果 loss = 1.0：

$$
\mathrm{PPL}=e^1\approx2.72
$$

如果 loss = 2.0：

$$
\mathrm{PPL}=e^2\approx7.39
$$

如果 loss = 4.0：

$$
\mathrm{PPL}=e^4\approx54.6
$$

因此，loss 每增加 1，perplexity 会乘以大约 $e\approx2.718$。

---

## 1. 优化器：SGD、Adam、AdamW

训练语言模型时，模型先通过 forward 得到 logits，再用 cross-entropy 得到 loss。

之后调用：

```python
loss.backward()
```

PyTorch 会把每个可学习参数的梯度写入对应参数的 `.grad` 中。

优化器的作用就是读取这些参数梯度，并根据某种更新规则修改参数。

CS336 这一章也是按 loss、optimizer、training loop 的顺序来组织训练系统的。

### 1.1 SGD

SGD 的基本更新公式是：

$$
\theta_{t+1}=\theta_t-\alpha_t\nabla L(\theta_t;B_t)
$$

其中：

- $\theta_t$ 是当前参数；
- $\alpha_t$ 是学习率；
- $B_t$ 是当前 batch；
- $\nabla L(\theta_t;B_t)$ 是当前 batch 上 loss 对参数的梯度。

直观理解：

> 梯度指向 loss 增大的方向，所以参数沿着负梯度方向走一步，让 loss 尽量下降。

SGD 的优点是简单。

但它只看当前 batch 的梯度，容易受梯度噪声影响；而且所有参数通常共用一个全局学习率，不同参数之间不能自适应调整步长。

### 1.2 Adam

Adam 可以理解成：

```text
momentum + 自适应学习率
```

它不只是看当前梯度，而是为每个参数维护两个状态：

- `m`：一阶矩估计，近似梯度的滑动平均；
- `v`：二阶矩估计，近似梯度平方的滑动平均。

公式是：

$$
m \leftarrow \beta_1m+(1-\beta_1)g
$$

$$
v \leftarrow \beta_2v+(1-\beta_2)g^2
$$

其中 `g` 是当前参数梯度。

`m` 的作用：

> `m` 记录最近一段时间梯度的大致方向。如果某个方向的梯度连续多步都差不多，`m` 会积累这个方向，从而让更新更稳定。

`v` 的作用：

> `v` 记录最近一段时间梯度大小的平方平均。如果某个参数的梯度经常很大，那么 `v` 会比较大，更新时会被除小；如果某个参数的梯度经常很小，那么 `v` 比较小，对应方向的步长相对不会太小。

Adam 的核心更新大致是：

$$
\theta \leftarrow \theta-\alpha\frac{m}{\sqrt{v}+\epsilon}
$$

其中 $\epsilon$ 是一个很小的数，用于数值稳定。

所以可以这样理解：

- `m` 决定更新方向；
- `v` 调整每个参数的更新尺度；
- `eps` 防止除以 0 或极小值。

### 1.3 Bias Correction

Adam 中 `m` 和 `v` 一开始初始化为 0，因此训练初期会偏小。

比如 $\beta_1=0.9$，$m_0=0$，第一步梯度是 `g`：

$$
m_1=0.9\cdot0+0.1g=0.1g
$$

但真实梯度是 `g`，$m_1$ 只有 `0.1g`，所以明显被 0 初始化拉低了。

因此 Adam 使用 bias correction：

$$
\hat{m}=\frac{m}{1-\beta_1^t}
$$

$$
\hat{v}=\frac{v}{1-\beta_2^t}
$$

也可以把修正合并到学习率中：

$$
\alpha_t=\alpha\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}
$$

CS336 的 AdamW 伪代码就是采用这种 adjusted learning rate 的写法。

### 1.4 AdamW

AdamW 是 Adam 的改进版，主要区别是 weight decay 被解耦了。

AdamW 的更新可以分成两部分。

第一部分，weight decay：

$$
\theta \leftarrow \theta-\alpha\lambda\theta
$$

这一步直接把参数往 0 拉一点，起正则化作用。

第二部分，Adam 的 moment-adjusted update：

$$
m \leftarrow \beta_1m+(1-\beta_1)g
$$

$$
v \leftarrow \beta_2v+(1-\beta_2)g^2
$$

$$
\theta \leftarrow \theta-\alpha_t\frac{m}{\sqrt{v}+\epsilon}
$$

AdamW 的关键是：

> weight decay 不再混进梯度 `g` 里，而是单独作用在参数 $\theta$ 上。

如果把 L2 正则直接加进 loss，那么梯度会变成：

$$
g_{total}=g+\lambda\theta
$$

Adam 会对这个 $g_{total}$ 做 `m/v` 自适应缩放，导致 weight decay 的效果被 Adam 的自适应机制改变。

AdamW 则直接做：

$$
\theta \leftarrow \theta-\alpha\lambda\theta
$$

这样 weight decay 更像真正的“每一步把权重缩小一点”。

---

## 2. Optimizer 实现中的 param_groups 和 state

### 2.1 param_groups 是什么

在 PyTorch 的 optimizer 里，`param_groups` 用来管理：

- 哪些参数要被优化；
- 这些参数使用什么超参数。

一个参数组大概可以理解成：

- `params`：这一组参数；
- `lr`：学习率；
- `betas`：AdamW 的 $\beta_1,\beta_2$；
- `eps`：数值稳定项；
- `weight_decay`：权重衰减系数。

如果直接写：

```python
optimizer = AdamW(model.parameters(), lr=1e-3)
```

那么所有参数会被放进同一个 param group。

如果想给不同参数设置不同超参数，可以分组。例如：

- 一组参数有 weight decay；
- 另一组参数没有 weight decay。

LLM 训练里常见做法是：

- Linear/Embedding 权重使用 weight decay；
- bias、LayerNorm/RMSNorm 参数不使用 weight decay。

所以：

> `param_groups` 是优化器的“参数分组配置表”。

### 2.2 optimizer.state 是什么

`optimizer.state` 是优化器给每个参数保存历史信息的地方。

它本质上是一个字典：

```text
state[p] = 参数 p 对应的状态
```

对于 SGD 的某些变体，可能只需要存：

- `t`：当前参数更新了多少步。

对于 AdamW，需要为每个参数存：

- `t`：当前步数；
- `m`：一阶矩；
- `v`：二阶矩。

所以如果某个参数 `p` 的形状是 `D x D`，那么：

- `m` 的形状也是 `D x D`；
- `v` 的形状也是 `D x D`。

也就是说，AdamW 会给每个参数额外保存两份同形状张量。

这就是为什么 AdamW 很吃显存。

### 2.3 优化器需要保存哪些东西

以 AdamW 为例，训练过程中和优化器相关的长期状态有：

- 参数本身 $\theta$；
- 参数梯度 `grad`，也就是 `p.grad`；
- 一阶矩 `m`；
- 二阶矩 `v`；
- 每组参数的超参数，例如 `lr`、`betas`、`eps`、`weight_decay`；
- 当前 step 数 `t`。

其中：

- 参数属于 model state；
- grad 存在参数的 `.grad` 中；
- `m`、`v`、`t` 存在 optimizer state 中；
- 超参数存在 `param_groups` 中。

保存 checkpoint 时，不能只保存 model，还要保存 optimizer，因为 AdamW 的 `m`、`v`、`t` 都会影响恢复训练后的连续性。

CS336 的 checkpointing 也明确要求保存：

```text
model state
optimizer state
iteration
```

---

## 3. 训练内存由哪些部分组成

Transformer 训练显存主要由四部分组成：

- `parameters`：模型参数；
- `activations`：前向传播保存的中间结果；
- `gradients`：参数梯度；
- `optimizer state`：优化器状态，例如 AdamW 的 `m` 和 `v`。

CS336 的 AdamW accounting 题目也是要求把内存拆成：

```text
parameters
activations
gradients
optimizer state
```

注意：这一节的公式是 CS336 手算 accounting 口径，不等价于 PyTorch 实际 `torch.cuda.max_memory_allocated()`。真实显存会受 autograd 保存策略、fused kernel、FlashAttention、gradient checkpointing、dtype、memory fragmentation 等影响。

### 3.1 参数 parameters

参数是训练过程中要被更新的张量。

例如：

- Embedding table；
- Q/K/V/O projection 权重；
- FFN 中的 `W1, W2, W3`；
- RMSNorm 的 scale 参数；
- output embedding / LM head 权重。

参数会长期存储在模型里。

### 3.2 激活值 activations

激活值是 forward 过程中产生、并且 backward 需要用到的中间张量。

例如：

- RMSNorm 输出；
- Q、K、V；
- $QK^T$ attention scores；
- softmax attention weights；
- $AV$ 的输出；
- FFN 中的 `W1x`、`W3x`、`SiLU(W1x)`、逐元素乘积结果；
- 最终 logits。

训练时要保存很多 activation，因为 backward 需要用它们来计算梯度。

例如：

$$
Q=XW_q^T
$$

为了计算 $W_q$ 的梯度，backward 需要用到前向时的 `X`。

### 3.3 梯度 gradients：参数梯度和临时梯度

这里一定要区分两种梯度。

第一种是参数梯度。

参数梯度是 loss 对参数的导数，例如：

```text
∂L/∂Wq
∂L/∂Wk
∂L/∂Wo
∂L/∂W1
∂L/∂gamma
```

这些梯度会保存到参数的 `.grad` 中，等待 `optimizer.step()` 使用。

例如：

```text
Wq.grad = ∂L/∂Wq
```

这些是显存 accounting 里通常说的 gradients。

如果参数量是 `P`，那么参数梯度数量通常也是 `P`。

第二种是临时中间梯度。

临时梯度是 backward 过程中为了链式法则临时产生的梯度。

例如 attention 里：

$$
S=QK^T
$$

$$
A=\mathrm{softmax}(S)
$$

$$
O=AV
$$

反向传播时会产生：

```text
∂L/∂O
∂L/∂A
∂L/∂S
∂L/∂Q
∂L/∂K
∂L/∂V
```

这些梯度是为了继续往前传播，最终算出参数梯度。

它们通常用完就可以释放，不会像参数梯度一样保存到 `optimizer.step()`。

所以显存 accounting 里说 gradients，主要是指参数梯度，不是所有中间激活的反传梯度。

### 3.4 用 Wq 举例理解参数梯度

设：

$$
Q=XW_q^T
$$

其中：

$$
X\in\mathbb{R}^{BT\times D}
$$

$$
W_q\in\mathbb{R}^{D\times D}
$$

$$
Q\in\mathbb{R}^{BT\times D}
$$

假设从后面传回来的临时梯度是：

$$
G_Q=\frac{\partial L}{\partial Q}
$$

那么：

$$
\frac{\partial L}{\partial W_q}=G_Q^TX
$$

这就是 $W_q$ 的参数梯度，需要保存到 `Wq.grad`。

同时：

$$
\frac{\partial L}{\partial X}=G_QW_q
$$

这只是继续往前传的中间梯度。

如果同一个 `X` 同时生成 Q、K、V：

$$
Q=XW_q^T
$$

$$
K=XW_k^T
$$

$$
V=XW_v^T
$$

那么对 `X` 的梯度是三条路径加起来：

$$
\frac{\partial L}{\partial X}=G_QW_q+G_KW_k+G_VW_v
$$

其中 $G_Q,G_K,G_V$ 都是临时梯度；而 $\partial L/\partial W_q$、$\partial L/\partial W_k$、$\partial L/\partial W_v$ 是参数梯度，需要保存在 `.grad` 中。

### 3.5 优化器状态 optimizer state

AdamW 对每个参数保存两份状态：

- `m`：一阶矩；
- `v`：二阶矩。

如果参数量是 `P`，则 AdamW optimizer state 数量是：

$$
2P
$$

如果使用 float32，每个元素 4 bytes，那么 optimizer state 内存是：

$$
2P\times4\text{ bytes}
$$

注意：

> $QK^T$、softmax、$AV$ 这些中间激活不是参数，所以没有 optimizer state。

---

## 4. 训练总内存计算

设：

```text
B = batch_size
T = context_length
L = num_layers
D = d_model
H = num_heads
V = vocab_size
d_ff = 8/3 D
```

这里要先明确一个口径：

- 如果 input embedding 和 LM head 共享权重，只统计一份 `VD`。
- 如果 input embedding 和 LM head 不共享权重，要统计两份 `VD`。

CS336 的不同题目和不同实现可能口径不一样。你的实现如果 `Embedding` 和 `linear_final` 是两个独立参数，那么主公式应该使用 `2VD`。

### 4.1 每层 Transformer block 的参数量

MHA 部分包含：

- RMSNorm：`D`
- QKV projection：`3D²`
- output projection：`D²`

所以：

$$
P_{MHA}=4D^2+D
$$

SwiGLU FFN：

$$
\mathrm{FFN}(x)=W_2(\mathrm{SiLU}(W_1x)\odot W_3x)
$$

其中：

```text
W1: D -> d_ff
W3: D -> d_ff
W2: d_ff -> D
```

参数量：

$$
Dd_{ff}+Dd_{ff}+d_{ff}D=3Dd_{ff}
$$

代入：

$$
d_{ff}=\frac{8}{3}D
$$

得到：

$$
3D\cdot\frac{8}{3}D=8D^2
$$

再加 FFN 前的 RMSNorm：

$$
D
$$

所以：

$$
P_{FFN}=8D^2+D
$$

单层 block 总参数：

$$
P_{block}=P_{MHA}+P_{FFN}
$$

$$
=(4D^2+D)+(8D^2+D)
$$

$$
=12D^2+2D
$$

`L` 层：

$$
P_{layers}=L(12D^2+2D)
$$

再加 final RMSNorm：

$$
D
$$

如果 input embedding 和 output embedding 共享或只统计一份：

$$
P=L(12D^2+2D)+D+VD
$$

如果 input embedding 和 output embedding 不共享：

$$
P=L(12D^2+2D)+D+2VD
$$

### 4.2 参数、梯度、优化器状态内存

以参数量 `P` 为基础：

```text
parameters = P
gradients = P
optimizer state = 2P
```

所以三者总元素数量：

$$
P+P+2P=4P
$$

如果使用 float32，这三者内存：

$$
4P\times4\text{ bytes}
$$

### 4.3 每层 activation 计算

MHA activation 包含：

- RMSNorm output：`BTD`
- QKV：`3BTD`
- $QK^T$ scores：`BHT²`
- softmax attention weights：`BHT²`
- $AV$ output：`BTD`
- output projection output：`BTD`

所以：

$$
A_{MHA}=BTD+3BTD+BHT^2+BHT^2+BTD+BTD
$$

$$
=6BTD+2BHT^2
$$

SwiGLU FFN activation 包含：

- FFN 前 RMSNorm：`BTD`
- `W1x`：`BTd_ff`
- `W3x`：`BTd_ff`
- `SiLU(W1x)`：`BTd_ff`
- element-wise product：`BTd_ff`
- `W2` output：`BTD`

所以：

$$
A_{FFN}=BTD+4BTd_{ff}+BTD
$$

$$
=2BTD+4BTd_{ff}
$$

代入：

$$
d_{ff}=\frac{8}{3}D
$$

得到：

$$
A_{FFN}=2BTD+4BT\cdot\frac{8}{3}D
$$

$$
=2BTD+\frac{32}{3}BTD
$$

$$
=\frac{38}{3}BTD
$$

单层 block activation：

$$
A_{block}=A_{MHA}+A_{FFN}
$$

$$
=6BTD+2BHT^2+\frac{38}{3}BTD
$$

$$
=\frac{56}{3}BTD+2BHT^2
$$

写成提取 `BT` 的形式：

$$
A_{block}=BT\left(\frac{56}{3}D+2HT\right)
$$

`L` 层：

$$
A_{layers}=BTL\left(\frac{56}{3}D+2HT\right)
$$

### 4.4 block 之后的 activation

final RMSNorm：

$$
BTD
$$

output embedding logits：

$$
BTV
$$

cross-entropy on logits：

$$
BTV
$$

所以额外 activation：

$$
A_{extra}=BTD+2BTV
$$

$$
=BT(D+2V)
$$

注意：这里是按“显式 logits + 显式 cross-entropy/logsumexp 中间量”的保守手算估计。如果使用 fused cross entropy，实际中间激活不一定保存两份 `BTV`。

### 4.5 总 activation

$$
A=BT\left[L\left(\frac{56}{3}D+2HT\right)+D+2V\right]
$$

### 4.6 总训练内存

元素数量：

$$
N_{elements}=4P+A
$$

如果共享或只统计一份 embedding / LM head：

$$
P=L(12D^2+2D)+D+VD
$$

如果 input embedding 和 output embedding 不共享：

$$
P=L(12D^2+2D)+D+2VD
$$

activation：

$$
A=BT\left[L\left(\frac{56}{3}D+2HT\right)+D+2V\right]
$$

如果 float32，每个元素 4 bytes：

$$
\mathrm{Memory}_{bytes}=4\cdot N_{elements}
$$

---

## 5. 训练总 FLOPs

训练一步主要包括：

- forward pass；
- backward pass；
- optimizer step。

题目中通常假设 backward pass 的 FLOPs 是 forward pass 的 2 倍。

### 5.1 Forward FLOPs

单条长度 `T` 的序列，Transformer LM 的 forward FLOPs 近似：

$$
F_{forward,single}=L(24TD^2+4T^2D)+2TDV
$$

这里 `24TD²` 的来源是：

MHA：

$$
8TD^2+4T^2D
$$

SwiGLU FFN：

$$
6TDd_{ff}
$$

如果：

$$
d_{ff}=\frac{8}{3}D
$$

则：

$$
6TDd_{ff}=6TD\cdot\frac{8}{3}D=16TD^2
$$

所以每层中和 `D²` 相关的部分是：

$$
8TD^2+16TD^2=24TD^2
$$

attention score 和 `AV` 贡献：

$$
4T^2D
$$

LM head：

$$
2TDV
$$

如果 batch size 是 `B`：

$$
F_{forward,batch}=B\left[L(24TD^2+4T^2D)+2TDV\right]
$$

### 5.2 Backward FLOPs

题目给定：

$$
F_{backward}\approx2F_{forward}
$$

所以 forward + backward：

$$
F_{train,model}=3F_{forward,batch}
$$

### 5.3 AdamW optimizer step FLOPs

AdamW 对每个参数大约做：

- weight decay；
- `m` 更新；
- `v` 更新；
- moment-adjusted parameter update。

每个参数大约 14 FLOPs。

如果参数量是 `P`：

$$
F_{AdamW}\approx14P
$$

如果严格计入 bias-corrected learning rate $\alpha_t$ 的计算，还要加 $O(1)$，因为 $\alpha_t$ 是每个 step 或每个 param group 计算一次的标量。

### 5.4 总训练 step FLOPs

$$
F_{step}\approx3B\left[L(24TD^2+4T^2D)+2TDV\right]+14P
$$

其中：

$$
P=L(12D^2+2D)+D+VD
$$

如果 input embedding 和 output embedding 不共享，把 `VD` 改成 `2VD`。

实际大模型训练中，`14P` 通常远小于 forward/backward，可以近似忽略。

---

## 6. 训练细节

前面已经介绍了 loss、optimizer、内存和 FLOPs。真正写训练脚本时，需要把这些模块串起来：

- 从 tokenized dataset 中采样 batch；
- 输入模型，得到 logits；
- 计算 cross-entropy loss；
- 反向传播，得到参数梯度；
- 梯度裁剪；
- 根据 scheduler 更新 learning rate；
- `optimizer.step()` 更新参数；
- 定期记录训练指标；
- 定期在 validation set 上评估；
- 定期保存 checkpoint。

### 6.1 Training Loop 伪代码

一个标准的 Transformer LM 训练循环可以写成：

```python
for step in range(max_steps):
    model.train()

    # 1. 采样 batch
    x, y = get_batch(
        train_data,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )

    # 2. forward
    logits = model(x)  # shape: (B, T, vocab_size)

    # 3. cross-entropy loss
    loss = cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
    )

    # 4. 清空旧梯度
    optimizer.zero_grad()

    # 5. backward
    loss.backward()

    # 6. gradient clipping
    grad_norm = clip_grad_norm_(
        model.parameters(),
        max_l2_norm,
    )

    # 7. 更新学习率
    lr = get_lr_cosine_schedule(
        step,
        max_learning_rate,
        min_learning_rate,
        warmup_iters,
        cosine_cycle_iters,
    )

    for group in optimizer.param_groups:
        group["lr"] = lr

    # 8. optimizer step
    optimizer.step()

    # 9. logging
    if step % log_interval == 0:
        log_train_metrics(...)

    # 10. validation
    if step % eval_interval == 0:
        val_loss = evaluate(...)
        log_val_metrics(...)

    # 11. checkpointing
    if step % checkpoint_interval == 0:
        save_checkpoint(...)
```

这条训练链路可以概括成：

```text
tokens -> model -> logits -> loss -> backward -> gradients -> AdamW update
```

### 6.2 Data Loading

语言模型训练数据通常是一条很长的 token 序列：

$$
x=(x_1,x_2,\dots,x_n)
$$

即使原始文本来自很多文档，也常见地把所有文档拼接成一个长序列，并在文档之间加入 `<|endoftext|>` 这样的特殊 token。

每次训练时，从长 token 序列中随机采样起点 $s$，构造：

```text
input  = x[s:s+T]
target = x[s+1:s+T+1]
```

其中 `T` 是 context length。

例如：

```text
x = [10, 25, 31, 9, 7, 4]
context_length = 4
```

如果随机起点是 1：

```text
input  = [25, 31, 9, 7]
target = [31, 9, 7, 4]
```

也就是每个位置都预测下一个 token。

注意，起点不能选得太靠后。必须满足：

$$
s+T<n
$$

否则 target 会越界。

### 6.3 Learning Rate Schedule

常见训练流程会使用：

```text
linear warmup + cosine decay
```

warmup 阶段：

$$
\alpha_t=\frac{t}{T_w}\alpha_{max}
$$

cosine decay 阶段：

$$
\alpha_t=\alpha_{min}+\frac{1}{2}\left(1+\cos\left(\frac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_{max}-\alpha_{min})
$$

post-annealing 阶段：

$$
\alpha_t=\alpha_{min}
$$

直觉：

- 前期学习率大，快速学习；
- 后期学习率小，稳定收敛。

### 6.4 Gradient Clipping

有时某些 batch 会产生特别大的梯度，导致训练不稳定。

gradient clipping 的做法是：

先计算所有参数梯度拼起来的 L2 norm：

$$
\|g\|_2
$$

如果：

$$
\|g\|_2 \le M
$$

则不变。

如果：

$$
\|g\|_2 > M
$$

则整体缩放：

$$
g \leftarrow g\cdot \frac{M}{\|g\|_2+\epsilon}
$$

注意：

> 梯度裁剪应该放在 `loss.backward()` 之后，`optimizer.step()` 之前。

因为只有 backward 后才有梯度，而 optimizer step 后参数已经更新了。

### 6.5 用 wandb 记录实验

训练语言模型时，不只要看最后结果，还要记录整个训练过程。

常见记录内容包括：

- `train_loss`
- `validation_loss`
- `perplexity`
- `learning_rate`
- `gradient_norm`
- `tokens_per_sec`
- `wall_clock_time`
- 当前实验的超参数配置
- 生成样例

wandb 的典型用法是：

```python
import wandb

wandb.init(
    project="cs336-transformer-lm",
    name="tinystories_lr_6e-4",
    config={
        "vocab_size": vocab_size,
        "context_length": context_length,
        "num_layers": num_layers,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "batch_size": batch_size,
        "max_learning_rate": max_learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "cosine_cycle_iters": cosine_cycle_iters,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
        "max_l2_norm": max_l2_norm,
    },
)
```

训练时记录：

```python
wandb.log(
    {
        "train/loss": train_loss,
        "train/perplexity": math.exp(train_loss),
        "train/lr": lr,
        "train/grad_norm": grad_norm,
        "perf/tokens_per_sec": tokens_per_sec,
        "time/wall_clock_sec": wall_clock,
    },
    step=step,
)
```

验证时记录：

```python
wandb.log(
    {
        "val/loss": val_loss,
        "val/perplexity": math.exp(val_loss),
    },
    step=step,
)
```

注意：最好显式传入 `step=step`。

因为 train log、validation log、checkpoint log 的频率可能不同。如果不传 step，wandb 会使用自己的内部 step 计数，可能导致曲线横轴和真实训练 step 对不上。

### 6.6 用 TensorBoard / SummaryWriter 记录实验

实验记录工具可以用 wandb，也可以用 TensorBoard。

TensorBoard 更适合本地或 AutoDL 上直接看曲线，例如 AutoDL 映射 6006 端口后，可以启动：

```bash
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

wandb 更适合云端实验追踪，方便保存不同 run 的 config、曲线和对比结果。

如果不使用 wandb，也可以用 TensorBoard。

初始化：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/tinystories_lr_6e-4")
```

记录 train loss：

```python
writer.add_scalar("loss/train", train_loss, step)
```

记录 validation loss：

```python
writer.add_scalar("loss/val", val_loss, step)
```

记录 perplexity：

```python
writer.add_scalar("ppl/train", math.exp(train_loss), step)
writer.add_scalar("ppl/val", math.exp(val_loss), step)
```

记录 learning rate：

```python
writer.add_scalar("lr", lr, step)
```

记录 gradient norm：

```python
writer.add_scalar("grad_norm", grad_norm, step)
```

记录吞吐速度：

```python
writer.add_scalar("perf/tokens_per_sec", tokens_per_sec, step)
```

记录配置：

```python
writer.add_text("config", json.dumps(config, indent=2), 0)
```

记录生成文本：

```python
writer.add_text("samples/generated_text", generated_text, step)
```

训练结束后关闭：

```python
writer.close()
```

查看 TensorBoard：

```bash
tensorboard --logdir runs
```

原因是：

> 训练中有 train log、eval log、checkpoint 等不同频率，如果不显式传 step，横轴可能不对应真实训练步数。

所以建议统一用训练迭代步数作为 step。

### 6.7 记录 GPU 时间时要同步

CUDA 操作默认是异步的。也就是说，Python 提交 GPU 任务后，可能不会等 GPU 真正执行完就继续往下走。

因此，如果要记录准确的 step time 或 tokens/sec，需要同步：

```python
if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

# training steps

if device.type == "cuda":
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start
```

不过不要每一步都同步，否则会降低训练效率。更常见的是每隔若干步同步一次，统计平均 tokens/sec。

### 6.8 定期 checkpoint

训练大模型时必须定期保存 checkpoint。

一个 checkpoint 至少应该包含：

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iteration": step,
}
```

保存：

```python
torch.save(checkpoint, path)
```

恢复：

```python
checkpoint = torch.load(path, map_location=device)

model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])

start_step = checkpoint["iteration"]
```

为什么 optimizer 也要保存？

> 因为 AdamW 的 `m`、`v`、`t` 都存在 optimizer state 里。如果只保存模型参数，不保存 optimizer，那么恢复训练时 AdamW 会忘记历史动量和二阶矩，训练状态不连续。

如果只保存 model，不保存 optimizer，那么恢复训练时：

- 模型参数能恢复；
- 但 AdamW 的历史动量和二阶矩没了；
- 学习率 schedule 也可能接不上。

所以标准 checkpoint 结构是：

```text
model state
optimizer state
current iteration
best validation loss，可选
config，可选
```

建议保存两类 checkpoint：

- latest checkpoint：定期保存，用于中断恢复；
- best checkpoint：验证集 loss 最低时保存，用于最终评估和生成。

---

## 7. Decoder 部分

训练时，模型学习的是：

$$
p_\theta(x_{i+1}\mid x_{1:i})
$$

生成时，就是反复调用这个 next-token distribution。

给定 prompt：

```text
x1, x2, ..., xt
```

模型输出 logits：

```text
v = TransformerLM(x1:t)_t
```

也就是最后一个位置的 vocab logits。

注意：

> `TransformerLM(...)[-1]` 输出的是 logits，不是概率。要经过 temperature scaling 和 softmax 后才是概率分布。

然后把 logits 变成概率分布，并采样下一个 token：

```text
x_{t+1}
```

再把这个 token 接回输入，继续生成。

CS336 的 decoding 部分也是这个逻辑：反复从 one-step conditional 中采样，并把生成的 token append 到下一步输入。

### 7.1 Temperature Sampling

模型输出 logits `v` 后，不直接 softmax，而是先除以 temperature $\tau$：

$$
\mathrm{softmax}(v,\tau)_i=\frac{\exp(v_i/\tau)}{\sum_j\exp(v_j/\tau)}
$$

temperature 控制随机性。

当 $\tau < 1$：

- 分布更尖锐；
- 高概率 token 更容易被选中；
- 生成更保守、更稳定。

当 $\tau > 1$：

- 分布更平坦；
- 低概率 token 更容易被选中；
- 生成更多样，但更容易乱。

当 $\tau \to 0$：

> 接近 greedy decoding，每一步都选最大概率 token。

### 7.2 Top-p / Nucleus Sampling

top-p 的思想是：

> 不从整个词表采样，而是只保留概率最高的一小批 token，使它们的累计概率至少达到 `p`。

步骤：

1. 对 logits 做 temperature scaling。
2. softmax 得到概率分布 `q`。
3. 把 `q` 按概率从大到小排序。
4. 从最大概率 token 开始累加，直到累计概率 `>= p`。
5. 只保留这些 token，其他 token 概率设为 0。
6. 对保留下来的概率重新归一化。
7. 从这个截断后的分布中采样。

例如排序后概率是：

```text
0.90, 0.02, 0.01, 0.005, ...
```

如果：

```text
top_p = 0.93
```

那么保留：

```text
0.90, 0.02, 0.01
```

因为前三个累计刚好达到 0.93。

然后重新归一化，在这三个 token 中采样。

注意：

> top-p 和 top-k 不同。

- top-k 固定保留 `k` 个 token；
- top-p 保留的 token 数量不固定，取决于当前概率分布有多尖锐。

### 7.3 逐个添加 token

decoder 不是逐个添加字符，而是逐个添加 token。

流程是：

1. 先把 prompt encode 成 token IDs；
2. 输入模型，得到 logits；
3. 取最后一个位置的 logits；
4. 应用 temperature 和 top-p；
5. 采样得到 `next_token_id`；
6. 把 `next_token_id` append 到当前 token 序列后面；
7. 如果 `next_token_id` 是 `<|endoftext|>`，停止；
8. 如果达到 `maximum_token`，停止；
9. 否则继续下一轮。

还要注意：

> 模型有最大 `context_length`。

如果生成过程中 token 序列长度超过 `context_length`，就只把最后 `context_length` 个 token 输入模型。

否则 RoPE / positional embedding / attention mask 可能超过模型支持范围。

伪代码：

```python
tokens = tokenizer.encode(prompt)

for _ in range(max_new_tokens):
    input_tokens = tokens[-context_length:]

    logits = model(input_tokens)

    next_logits = logits[-1]

    next_logits = next_logits / temperature

    probs = softmax(next_logits)

    probs = top_p_filter_and_renormalize(probs, top_p)

    next_token = sample(probs)

    tokens.append(next_token)

    if next_token == eos_token_id:
        break

text = tokenizer.decode(tokens)
```

---

## 8. 总结

Transformer LM 的完整训练与生成流程可以总结为：

```text
文本数据
-> tokenizer
-> token IDs
-> get_batch(input, target)
-> Transformer forward
-> logits
-> cross-entropy loss
-> perplexity
-> backward
-> 参数梯度
-> gradient clipping
-> AdamW 更新参数、m、v
-> logging
-> validation
-> checkpoint
-> decoder 生成文本
```

优化器部分要抓住：

- SGD 只看当前梯度；
- Adam 保存 `m` 和 `v`；
- AdamW 把 weight decay 从梯度更新中解耦；
- `param_groups` 管参数组和超参数；
- `optimizer.state` 管每个参数的历史状态。

内存部分要抓住：

- parameters、gradients、optimizer state 与参数量 `P` 相关；
- activations 与 `B, T, L, D, H, V` 相关；
- 参数梯度需要保存到 `optimizer.step()`；
- 临时中间梯度只是 backward 链式法则中的过程量，用完可以释放。

decoder 部分要抓住：

- 语言模型生成就是反复预测下一个 token；
- logits 不是概率，要经过 softmax；
- temperature 控制分布尖锐程度；
- top-p 截掉低概率长尾；
- 生成时逐个 append token，不是逐个 append 字符。

核心概念对应关系：

| 模块 | 作用 |
|---|---|
| cross-entropy | 训练目标，最大化真实 next token 的概率 |
| perplexity | 评估指标，表示模型平均每步有多困惑 |
| AdamW | 用参数梯度更新参数，同时维护 `m/v` |
| param_groups | 管理不同参数组及其超参数 |
| optimizer state | 保存 AdamW 的 `m`、`v`、`t` 等历史状态 |
| activations | forward 中保存的中间结果，用于 backward |
| parameter gradients | loss 对参数的梯度，需要保存到 `optimizer.step()` |
| temporary gradients | backward 中间梯度，用于链式法则，通常临时使用 |
| checkpoint | 保存 model、optimizer、iteration，用于恢复训练 |
| decoder | 反复 next-token sampling 生成文本 |
| temperature | 控制采样分布尖锐程度 |
| top-p | 截断低概率 token，只在高概率 nucleus 内采样 |
