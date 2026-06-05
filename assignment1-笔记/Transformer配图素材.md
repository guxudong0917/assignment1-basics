---
tags:
  - CS336
  - Transformer
  - 配图素材
---

# Transformer 架构笔记配图素材

## 相关笔记

- [[CS336 笔记索引|CS336 笔记索引]]
- [[Transformer架构笔记|Transformer 架构笔记]]
- [[Transformer训练笔记|Transformer 训练笔记]]
- [[BPE分词笔记|BPE 分词笔记]]

这些图都是从现有网页/论文来源收集的，不是自己画的。建议先作为个人 Obsidian 笔记配图使用；如果以后要公开发布，记得按原网页/图片许可做 attribution。

## 1. 总体架构：原始 Transformer 图

适合放在：

- `Transformer Block`
- `Post-Norm 和 Pre-Norm`
- `原始 Transformer vs Decoder-only LM`

![Attention Is All You Need architecture](https://upload.wikimedia.org/wikipedia/commons/4/49/Attention_Is_All_You_Need_-_Encoder-decoder_Architecture.png)

来源：[Wikimedia Commons: Attention Is All You Need - Encoder-decoder Architecture](https://commons.wikimedia.org/wiki/File:Attention_Is_All_You_Need_-_Encoder-decoder_Architecture.png)

说明：这是原论文架构图，适合说明最早期 Transformer 是 encoder-decoder 且使用 post-norm。Wikimedia 页面标注为 CC BY-SA 4.0。

---

## 2. Decoder-only LM：GPT-2 堆叠 decoder block

适合放在：

- `任务定义：Transformer LM 到底在预测什么`
- `Decoder-only LM`
- `sequence_length/context_length`

![GPT-2 decoder stack](https://jalammar.github.io/images/gpt2/gpt-2-layers-2.png)

来源：[Jay Alammar, The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

说明：这张图比原始 encoder-decoder 图更适合你的笔记，因为你实现的是 decoder-only language model。

---

## 3. Decoder block 内部：Masked Self-Attention + FFN

适合放在：

- `Transformer Block`
- `MHA 模块和 FFN 模块分别在干什么`
- `Causal Mask`

![Transformer decoder block](https://jalammar.github.io/images/xlnet/transformer-decoder-intro.png)

来源：[Jay Alammar, The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

说明：这张图简洁地显示了 decoder block 里主要是 masked self-attention 和 feed-forward network。

---

## 4. Embedding 查表：Token Embedding Matrix

适合放在：

- `Embedding`
- `Embedding 和 Linear 的区别`
- `vocab_size x d_model`

![Token embeddings matrix](https://jalammar.github.io/images/gpt2/gpt2-token-embeddings-wte-2.png)

来源：[Jay Alammar, The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

说明：非常适合解释 embedding 是查表，不是普通矩阵乘法。

---

## 5. Token embedding + positional encoding

适合放在：

- `位置编码`
- `绝对位置编码`
- `embedding 后加入位置信息`

![Input embedding plus positional encoding](https://jalammar.github.io/images/gpt2/gpt2-input-embedding-positional-encoding-3.png)

来源：[Jay Alammar, The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

说明：这张适合讲绝对位置编码或 GPT-2 learned position embedding。注意：RoPE 不是这样加在 embedding 上，而是在 Q/K 上旋转。

---

## 6. Positional encoding 表：context length x embedding size

适合放在：

- `sequence_length 和 context_length`
- `位置编码表`
- `GPT-2 learned positional embedding`

![GPT-2 positional encoding table](https://jalammar.github.io/images/gpt2/gpt2-positional-encoding.png)

来源：[Jay Alammar, The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

说明：这张图把 position embedding 表的形状讲得很清楚：context size x embedding size。

---

## 7. Sinusoidal positional encoding heatmap

适合放在：

- `绝对位置编码`
- `sin/cos 不同频率`
- `多个时钟：秒针/分针/时针`

![Sinusoidal positional encoding heatmap](https://jalammar.github.io/images/t/transformer_positional_encoding_large_example.png)

来源：[Jay Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

说明：很适合配合你“多个不同频率的时钟”这个类比。横轴是 embedding dimension，纵轴是 position。

---

## 8. Q/K/V 投影：X 乘 Wq/Wk/Wv

适合放在：

- `Query：我想找什么信息`
- `Key：我能提供什么索引信息`
- `Value：真正被搬运的信息`
- `PyTorch 行向量和 Linear`

![QKV matrix projection](https://jalammar.github.io/images/t/self-attention-matrix-calculation.png)

来源：[Jay Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

说明：这张非常贴合你的解释：输入 X 分别乘 Wq/Wk/Wv，得到 Q/K/V。

---

## 9. Scaled Dot-Product Attention 公式图

适合放在：

- `QK^T：计算谁应该关注谁`
- `为什么要除以 sqrt(d_k)`
- `Softmax 和数值稳定`
- `Value 加权求和`

![Scaled dot-product attention](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png)

来源：[Jay Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

说明：这张图直接对应公式 `softmax(QK^T / sqrt(d_k))V`。

---

## 10. Q/K/V 直觉类比：文件夹检索

适合放在：

- `为什么 Q/K/V 可以理解成一问一答`
- `Q 像问题，K 像索引，V 像内容`

![QKV filing cabinet analogy](https://jalammar.github.io/images/gpt2/self-attention-example-folders-3.png)

来源：[Jay Alammar, The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

说明：这张图和你笔记里的 Q/K/V 类比特别合。Query 是要查的问题，Key 是文件夹标签，Value 是文件夹内容。

---

## 11. Attention 权重乘以 Value

适合放在：

- `Value：真正被搬运的信息`
- `A V: 聚合上下文信息`
- `第 i 个 token 对全文的信息提取`

![Weighted values in self-attention](https://jalammar.github.io/images/gpt2/gpt2-value-vector-sum.png)

来源：[Jay Alammar, The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

说明：这张图直观展示每个 value vector 被 attention score 缩放，然后求和。

---

## 12. Masked Self-Attention：不能看未来

适合放在：

- `Causal Mask`
- `Decoder-only LM`
- `训练时不能偷看未来 token`

![Self-attention vs masked self-attention](https://jalammar.github.io/images/gpt2/self-attention-and-masked-self-attention.png)

来源：[Jay Alammar, The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

说明：非常适合解释 causal attention。左边普通 self-attention 能看全局，右边 masked self-attention 只能看当前和过去。

---

## 13. Multi-Head Attention 总流程

适合放在：

- `Multi-Head Attention`
- `为什么多头不是简单拆分后再加回来`
- `concat heads + W_o`

![Multi-head attention recap](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

来源：[Jay Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

说明：这张图很完整：输入、Q/K/V、多个 head、concat、乘 W_o 都有。

---

## 14. Concat heads + W_o

适合放在：

- `多头输出和 W_o`
- `W_o 不是简单降维，而是混合不同 head`

![Concatenate attention heads and output projection](https://jalammar.github.io/images/t/transformer_attention_heads_weight_matrix_o.png)

来源：[Jay Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

说明：适合配合“每个 head 提交观察报告，W_o 是总编辑”这个类比。

---

## 15. Attention 可视化：不同 head 关注不同位置

适合放在：

- `多头和单头的区别`
- `不同 head 关注不同关系`

![Attention head visualization](https://jalammar.github.io/images/t/transformer_self-attention_visualization_3.png)

来源：[Jay Alammar, The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

说明：这张图适合展示 attention head 真的会把某个 token 和上下文中的其他 token 关联起来。

---

## 16. RoPE：旋转位置编码

适合放在：

- `RoPE`
- `为什么 RoPE 只作用 Q/K`
- `相对位置来自旋转角度差`
- `奇偶维旋转`

推荐页面：

- [Abhik Sarkar, Rotary Position Embeddings](https://www.abhik.ai/concepts/transformers/rotary-position-embeddings)
- [ZeroEntropy, RoPE concept page](https://zeroentropy.dev/concepts/rope-rotary-positional-embedding/)
- [Michael Brenndoerfer, RoPE interactive explanation](https://mbrenndoerfer.com/writing/rotary-position-embedding-rope-transformers)

说明：RoPE 最好用交互图或网页截图。ZeroEntropy 页面有“position becomes rotation”和“dimension pair rotation”的可视化；Michael Brenndoerfer 页面有 attention score Toeplitz heatmap，适合解释相对位置。

---

## 17. RMSNorm / LayerNorm

适合放在：

- `RMSNorm 和 LayerNorm`
- `为什么 RMSNorm 更简单`
- `RMSNorm 少了 mean-centering`

推荐页面：

- [PyTorch RMSNorm 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.modules.normalization.RMSNorm.html)
- [Sebastian Raschka: RMSNorm vs LayerNorm](https://sebastianraschka.com/faq/docs/rmsnorm-vs-layernorm.html)
- [RMSNorm 原论文](https://arxiv.org/abs/1910.07467)

说明：RMSNorm 这块网上高质量“图”不多，很多是公式和表格。更适合在笔记里用公式截图或表格，不建议随便找一张泛泛的 normalization 图。
