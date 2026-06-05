---
tags:
  - CS336
  - Tokenizer
  - BPE
---

# BPE 分词笔记

## 相关笔记

- [[CS336 笔记索引|CS336 笔记索引]]
- [[Transformer架构笔记|Transformer 架构笔记]]
- [[Transformer训练笔记|Transformer 训练笔记]]
- [[Transformer配图素材|Transformer 配图素材]]

这份笔记按实现顺序整理 BPE tokenizer：先理解 BPE 的思想，再看训练阶段如何得到 `vocab` 和 `merges`，最后看 `Tokenizer` 如何 encode/decode。

## 1. BPE 的核心思想

BPE 是 **Byte Pair Encoding**。它的基本思想是：

> 从最小单位开始，不断把语料中最常见的相邻 pair 合并成一个新 token。

在 byte-level BPE 中，最小单位是 byte，而不是字符或单词。

例如某个 pre-token 初始表示为：

```text
l o w
```

如果 `(o, w)` 是当前最高频 pair，就把它合并成一个新 token：

```text
l ow
```

如果下一轮 `(l, ow)` 又是最高频 pair，就继续合并：

```text
low
```

所以 BPE 的训练过程本质上是在学习一系列 merge rules：

```text
(o, w) -> ow
(l, ow) -> low
...
```

这些 merge rules 按创建顺序记录下来，就是 `merges`。

## 2. 不同分词方法的区别

| 方法 | 基本单位 | 优点 | 缺点 |
|---|---|---|---|
| Word-level | 单词 | token 有语义，序列短 | 词表巨大，容易 OOV |
| Character-level | 字符 | 词表小，几乎无 OOV | 序列长，模型要自己学字符组合 |
| Byte-level | 字节 | 只需要 256 个初始 token，任意文本都能表示 | 初始序列更长 |
| BPE | 学习得到的子词/byte 片段 | 折中词表大小和序列长度 | 训练和编码逻辑更复杂 |

### Byte 和 Character 的区别

Character 是 Unicode 层面的字符，例如 `你` 是一个字符。

Byte 是 UTF-8 编码后的字节。一个 Unicode 字符可能对应多个 bytes。

Byte-level BPE 的初始 vocabulary 是所有单字节值：

```text
0, 1, 2, ..., 255
```

因此初始化时应该是：

```text
0 -> b"\x00"
1 -> b"\x01"
...
97 -> b"a"
...
255 -> b"\xff"
```

也就是说，初始 byte vocab 应该使用“单个 byte”的概念，而不是把 `chr(i)` 再做 UTF-8 编码。

## 3. BPE 训练阶段概览

BPE 训练阶段的目标是从语料中得到：

```text
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

其中：

- `vocab` 是 token ID 到 token bytes 的映射。
- `merges` 是训练过程中按顺序学到的 merge rules。

训练阶段大致分成：

1. Pre-tokenization，统计每个 pre-token 出现次数。
2. 初始化 vocab。
3. 统计所有 pair 的频率。
4. 反复选择最高频 pair。
5. 合并该 pair，更新 vocab 和 merges。

如果训练目标是 `vocab_size`，初始 byte vocab 有 256 个 token，special tokens 有 `num_special_tokens` 个，那么最多需要学习：

```text
vocab_size - 256 - num_special_tokens
```

条 merge rules。比如 `vocab_size=10000` 且有 1 个 special token 时，最多学习 `10000 - 256 - 1 = 9743` 条 merges。

如果语料太小，中途已经没有任何可合并 pair，则应该提前停止，而不是继续调用 `max`。

## 4. Pre-tokenization：先统计每个词出现了多少次

直接在整个原始文本上统计 byte pair 很慢，也容易跨越不该跨越的边界。

所以训练前先做 pre-tokenization，把文本粗分成 pre-tokens。

CS336 使用 GPT-2 风格 regex，例如：

```text
"some text that i'll pre-tokenize"
-> ["some", " text", " that", " i", "'ll", " pre", "-", "tokenize"]
```

注意：空格通常会被并入后面的 pre-token，例如 `" text"`。

Pre-tokenization 后，先统计每个 pre-token 总共出现多少次：

```text
low: 5
lower: 2
widest: 3
newest: 6
```

然后把每个 pre-token 转成 UTF-8 bytes 序列，并只在这个 pre-token 内部统计相邻 pair。

例如：

```text
low -> l o w
```

如果 `low` 出现了 5 次，那么：

```text
(l, o) += 5
(o, w) += 5
```

更一般地说：

> 一个 pair 的总次数 = 它在每个 pre-token 内出现的次数 × 当前 pre-token 的出现次数，再对所有 pre-token 求和。

这一步非常关键：**pair 不能跨 pre-token 边界统计**。

## 5. 为什么需要 chunk

真实训练语料可能非常大，不能一次性全部读入内存。

所以需要把原始文件切成多个 chunk，每个 chunk 独立做 pre-tokenization 和 Counter 统计，最后把多个 Counter 合并。

CS336 给的 `find_chunk_boundaries` 思路是：

1. 先按文件大小粗略切分。
2. 对每个内部边界，向后寻找 special token，例如 `<|endoftext|>`。
3. 把 chunk 边界放到 special token 开始的位置。

这样做的原因是：

```text
[Doc 1]<|endoftext|>[Doc 2]
```

不应该让 Doc 1 末尾和 Doc 2 开头产生 merge。

Special token 是硬边界。训练时应该按 special token split：

```text
[Doc 1]
[Doc 2]
```

然后分别 pre-tokenize。

Special token 本身要加入 vocab，但不参与 merge count。

## 6. Vocab 初始化

BPE 训练返回的 vocab 类型是：

```text
dict[int, bytes]
```

初始 byte vocab 是 256 个单 byte：

```text
0 -> b"\x00"
1 -> b"\x01"
...
255 -> b"\xff"
```

如果有 special tokens，也加入 vocab：

```text
256 -> b"<|endoftext|>"
```

注意：

- 普通 byte vocab 是 `bytes([i])` 这种单 byte。
- Special token 是字符串，需要用 UTF-8 编码成 bytes。
- 后续 merge 得到的新 token 也都是 bytes 拼接结果。

## 7. 训练方法一：朴素 pair_count + 全量 merge

最直接的训练方法是维护：

```text
byte_counter:
  tuple[token_id, ...] -> count
```

例如：

```text
(l, o, w): 5
(l, o, w, e, r): 2
```

每一轮：

1. 遍历所有 pre-token tuple。
2. 统计所有相邻 pair 的频率，得到 `pair_count`。
3. 用 `max(pair_count.items(), key=...)` 找最高频 pair。
4. 如果频率相同，选择 bytes 字典序更大的 pair。
5. 遍历所有 `byte_counter`，把出现该 pair 的地方替换成新 token ID。
6. 更新 vocab 和 merges。

也就是说，best pair 的排序 key 可以理解为：

```text
(pair_count[pair], vocab[pair[0]], vocab[pair[1]])
```

频率最大优先；频率相同时，先比较第一个 token 对应的 bytes，再比较第二个 token 对应的 bytes，选择字典序更大的 pair。

### 7.1 朴素 merge 的问题

朴素 merge 每轮都会遍历所有 pre-token。

但是一次 merge 只会影响包含目标 pair 的 pre-token。

例如把 `(b, c)` 合并成 `x`：

```text
a b c d
```

合并前的相邻 pair：

```text
(a, b), (b, c), (c, d)
```

合并后：

```text
a x d
```

新的相邻 pair：

```text
(a, x), (x, d)
```

变化只发生在目标 pair 和它前后的局部区域：

- `(b, c)` 减少。
- `(a, b)` 减少。
- `(c, d)` 减少。
- `(a, x)` 增加。
- `(x, d)` 增加。

所以没有必要每轮重新扫描所有 pre-token。

## 8. 训练方法二：维护 pair 到 word 的倒排索引

为了避免每轮遍历所有 pre-token，可以维护：

```text
pair_to_word_idx[pair] = set[word_idx]
```

也就是记录每个 pair 当前出现在哪些 pre-token 里。

同时把 pre-token 存成列表：

```text
byte_counter_list[word_idx] = (tuple_word, count)
```

这样选中 `best_pair` 后，只需要处理：

```text
pair_to_word_idx[best_pair]
```

中的那些 word。

### 8.1 更新过程

对于每个受影响的 word：

1. 取出旧的 token tuple 和 count。
2. 遍历旧 tuple 中所有相邻 old pairs。
3. 从 `pair_count` 中减去这些 old pairs 的贡献。
4. 从 `pair_to_word_idx[old_pair]` 中删去当前 word_idx。
5. 把目标 pair 合并成新 token ID。
6. 遍历新 tuple 中所有相邻 new pairs。
7. 给 `pair_count` 加回这些 new pairs 的贡献。
8. 给 `pair_to_word_idx[new_pair]` 加入当前 word_idx。
9. 更新 `byte_counter_list[word_idx]`。

这个方法的核心是：

> 只更新本轮 merge 真正影响到的 pre-tokens。

### 8.2 需要维护的 invariants

实现时需要保证：

```text
pair_to_word_idx[pair]
```

里的每个 word_idx 当前真的包含这个 pair。

否则如果索引里有过期 word_idx，就会错误扣减 pair_count。

可以用这个思路检查：

```text
当前 word 的 adjacent pairs 中必须包含当前 pair。
```

## 9. 最高频 pair 选择的优化：max 的瓶颈

当 vocab size 较小时，每轮使用：

```text
max(pair_count.items(), key=...)
```

还能接受。

但是当 `vocab_size=10000` 或 `32000` 时，merge 轮数很多：

```text
10000 vocab -> 约 9743 次 merge
32000 vocab -> 约 31743 次 merge
```

如果每轮都扫描整个 `pair_count`，`max` 会成为主要瓶颈。

### 9.1 最大堆思路

一种方法是用 heap 维护候选 pair。

由于 pair_count 会不断变化，heap 中会有过期项，所以需要 lazy deletion：

1. pair count 改变时，把新状态 push 进 heap。
2. 取堆顶时检查它是否仍然等于当前真实 pair_count。
3. 如果过期，就丢掉继续 pop。

这个方法可行，但 tie-breaking 和过期项处理比较麻烦。

### 9.2 Bucket 方法

另一种方法是维护精确 count bucket：

```text
count_to_pairs[count] = set[pair]
max_count = 当前最高频率
```

初始化时，遍历 `pair_count`：

```text
pair_count[pair] = count
=> count_to_pairs[count].add(pair)
```

每次 pair count 变化时，都要把 pair 从旧 count bucket 移到新 count bucket：

```text
old_count -> remove pair
new_count -> add pair
```

选择最高频 pair 时：

1. 找到当前非空的 `count_to_pairs[max_count]`。
2. 在这个 bucket 里按 bytes 字典序选择最大的 pair。

这样避免每轮扫描完整 `pair_count`。

注意 bucket 只负责快速定位“最高频率”这一层；同频 pair 的 tie-breaking 仍然要在该 bucket 内用 bytes 字典序完成，不能随便取 set 里的某个元素。

### 9.3 Bucket 的终止条件

如果最高 bucket 为空，需要不断降低 `max_count`，直到找到非空 bucket：

```text
while max_count > 0 and bucket[max_count] empty:
    max_count -= 1
```

如果 `max_count == 0`，说明当前已经没有可 merge 的 pair，此时应该终止训练。

正常大语料通常能训练到目标 vocab size；小语料可能提前没有可合并 pair。

### 9.4 Bucket 的关键 invariant

必须始终满足：

```text
pair_count[pair] == c
```

当且仅当：

```text
pair in count_to_pairs[c]
```

如果一个 pair 同时存在多个 bucket，或者存在于错误 bucket，后续最高频 pair 会选错。

因此更新 pair count 时，不应该只改 `pair_count`，还必须同步移动 bucket。

## 10. Tokenizer 类的组成

训练完成后，就可以实现 `Tokenizer`。

Tokenizer 需要保存：

```text
vocab: id -> bytes
bytes_to_id: bytes -> id
merge_rank: (bytes, bytes) -> rank
special_tokens
```

其中：

- `vocab` 用于 decode。
- `bytes_to_id` 用于 encode 最后查 token ID。
- `merge_rank` 用于 encode 时判断哪个 pair 应该先合并。
- `special_tokens` 用于保证特殊字符不被拆开。

如果构造 tokenizer 时提供了 special tokens，而它们不在 vocab 中，需要追加到 vocab。

## 11. Encode 过程

Encode 的目标：

```text
text: str -> list[int]
```

### 11.1 先按 special tokens 划分

Special token 不能被普通 regex 或 BPE 拆开。

例如：

```text
hello<|endoftext|>world
```

应该处理成：

```text
encode("hello") + [special_id] + encode("world")
```

如果使用 regex split，需要用捕获组保留 special token 本身：

```text
(special_token_pattern)
```

这样 split 后 special token 不会丢失。

如果 special tokens 有重叠，应该优先匹配最长的 special token。

例如同时有：

```text
<|endoftext|>
<|endoftext|><|endoftext|>
```

编码：

```text
<|endoftext|><|endoftext|>
```

时应该优先把它识别为一个长 special token，而不是两个短 special token。因此构造 special token regex 前，最好按长度从大到小排序。

### 11.2 对普通文本使用 PATTERN pre-tokenize

对非 special token 的部分，使用 GPT-2 regex：

```text
PATTERN.finditer(text_part)
```

得到一系列 pre-tokens，例如：

```text
"the cat ate"
-> "the", " cat", " ate"
```

每个 pre-token 独立 encode。

### 11.3 把 pre-token 转成 byte tokens

例如：

```text
"the"
```

先变成 UTF-8 bytes：

```text
b"the"
```

遍历 `b"the"` 得到的是整数 byte values：

```text
116, 104, 101
```

所以需要转回单字节 bytes tokens：

```text
b"t", b"h", b"e"
```

这样才能和 `merges` 中的 `tuple[bytes, bytes]` 对齐。

### 11.4 对单个 pre-token 应用 merges

对当前 pre-token 的 token 序列：

```text
[b"t", b"h", b"e"]
```

遍历它的相邻 pairs：

```text
(b"t", b"h")
(b"h", b"e")
```

查看哪些 pair 在 `merge_rank` 中。

如果多个 pair 都能合并，选择 rank 最小的 pair，也就是训练时最早学到的 merge。

合并该 pair 后，更新当前 pre-token，再继续查找可合并 pair。

直到找不到任何可应用的 merge。

最后得到的 bytes tokens 用 `bytes_to_id` 转成 token IDs。

### 11.5 Encode 缓存优化

同一个 pre-token 会反复出现，例如：

```text
" the"
" and"
"."
```

所以可以缓存：

```text
pretoken -> encoded token ids
```

这样再次遇到同一个 pre-token 时，就不需要重新执行 merge。

## 12. Decode 过程

Decode 的目标：

```text
list[int] -> str
```

流程：

1. 遍历 token IDs。
2. 用 `vocab[id]` 查出每个 token 对应的 bytes。
3. 把所有 bytes 拼接。
4. 用 UTF-8 decode。

如果 token IDs 对应的 bytes 不是合法 UTF-8，应该用：

```text
errors="replace"
```

这样非法 byte 序列会被替换成 Unicode replacement character：

```text
�
```

这是必要的，因为单独 decode 某个 token 时，它可能只是一个多字节 UTF-8 字符的一部分。

## 13. encode_iterable

`encode(text)` 假设整个文本已经在内存中。

如果要编码大文件，需要用：

```text
encode_iterable(iterable: Iterable[str]) -> Iterator[int]
```

它应该逐块读取文本，并逐个 yield token ID。

注意返回类型是：

```text
Iterator[int]
```

因此每次 yield 的应该是一个整数 token ID，而不是一个 `list[int]`。

基础逻辑：

```text
for chunk in iterable:
    ids = encode(chunk)
    yield each id
```

严格来说，chunk 边界可能影响 tokenization，所以更完整的实现需要 buffer。基础版本可以先逐行 encode。

原因是 BPE 的 pre-token 可能跨过普通 chunk 边界。例如直接把文本按固定字符数切开，某个单词可能被切成前后两半，tokenization 就会和一次性编码全文不同。更严格的流式实现应该保留一小段未确认边界的 buffer，或者只在明确不会跨越 token 的边界处切分。

## 14. 序列化与实验指标

### 14.1 保存 vocab 和 merges

训练结束后要把 `vocab` 和 `merges` 保存到磁盘，后续 tokenizer 可以直接加载，不需要重新训练。

可以用 pickle 保存 Python 对象，也可以把 bytes 转成 hex/base64 后保存成 JSON。

### 14.2 保存 token IDs

训练语言模型时，需要把语料 encode 成 token ID 序列。

如果 vocab size 小于 65536，可以使用：

```text
uint16
```

因为：

```text
uint16 范围是 0 到 65535
```

对于 `vocab_size=10000` 和 `vocab_size=32000` 都足够。

### 14.3 压缩率

一种常用指标：

```text
原始 UTF-8 byte 数 / token ID 数量
```

例如：

```text
4.15 bytes/token
```

表示平均每个 token 覆盖约 4.15 个原始 bytes。

## 15. Debug 检查清单

- [ ] 初始 vocab 是 256 个单 byte。
- [ ] Special tokens 加入 vocab，但不参与训练 merge counts。
- [ ] Pre-tokenization 后只在每个 pre-token 内部统计 pair。
- [ ] `merges` 返回的是 `tuple[bytes, bytes]`。
- [ ] Tie-breaking 使用 bytes 字典序。
- [ ] `pair_to_word_idx` 中没有过期 word index。
- [ ] `count_to_pairs` 和 `pair_count` 始终同步。
- [ ] Encode 时 special tokens 不被拆开。
- [ ] Overlapping special tokens 优先匹配最长项。
- [ ] Decode 使用 `errors="replace"`。
- [ ] `encode_iterable` yield 单个 token ID。
