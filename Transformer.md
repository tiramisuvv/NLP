# Transformers

[toc]

# 1. Motivation 

- From recurrence(RNN) to attention-based NLP models

## 1.1 Recall 

- encoder-decoder models
- encoder: encode sentences with a bidirectional LSTM
- decoder: an LSTM to generate the output
- **Use attention to allow flexible access to memory**

=> the best **building blocks** to plug into our models and enable broad progress.

<img src="/Users/Wei/Documents/NLP/NLP/transformer/rnn2transformer.png" alt="rnn2transformer" style="zoom:50%;" />

## 1.2 Issues with recurrent models

### 1.2.1 Linear interaction distance

- RNNs are unrolled “left-to-right”.
- **Problem**: RNNs take **O(sequence length)** steps for distant word pairs to interact.
  - 从 **chef** 到 **was**
  - <img src="/Users/Wei/Documents/NLP/NLP/transformer/rnn_issue.png" alt="rnn_issue" style="zoom:50%;" />
- **O(sequence length)** steps for distant word pairs to interact means:
  - Hard to learn long-distance dependencies (because gradient problems!) 
  - Linear order of words is “baked in”; we already know linear order isn’t the right way to think about sentences…
  - 上面例子里，**chef** 的信息在 **was** 之前就已经消失殆尽了。

### 1.2.2 Lack of parallelizability

- RNN Forward and backward passes have **O(sequence length)** unparallelizable operations
- <img src="/Users/Wei/Documents/NLP/NLP/transformer/rnn_issue2.png" alt="rnn_issue2" style="zoom:50%;" />

## 1.3 Word windows

### 1.3.1 Word window models aggregate local contexts

- Also known as 1D convolution
- Number of unparallelizable operations does not increase sequence length!

<img src="/Users/Wei/Documents/NLP/NLP/transformer/word_window1.png" alt="word_window1" style="zoom:50%;" />

### 1.3.2 Long-distance dependency

- 通过 **Stacking word window layes** 增加 interaction between farther words
- 但有最大限制
  - Maximum Interaction distance = **sequence length / window size**
- if your sequences are too long, you’ll just ignore long-distance context
- <img src="/Users/Wei/Documents/NLP/NLP/transformer/word_window2.png" alt="word_window2" style="zoom:50%;" />
  - $h_k$ 可以‘看到’标红的状态	
  - 超出范围的（$h_1$​）就不行了

## 1.4 Attention

### 1.4.1 Recall

- **Attention** treats each word’s representation as a **query** to access and incorporate information from **a set of values**.

- 之前用到attention from the decoder to the encoder; 

  - on each step of the decoder, **use direct connection to the encoder** to **focus on a particular part** of the source sequence
  - 

- <img src="/Users/Wei/Documents/NLP/NLP/transformer/seq2seq_attention.png" alt="seq2seq_attention" style="zoom:50%;" />

  - **attention output** 

    - Use the **attention distribution** to take a weighted sum of the encoder hidden states.

      - $$
        a_t = \sum_{i=1}^{N}\alpha_i^th_i \in \mathbb{R}^h
        $$

        其中 $\alpha_i^t$​  是 attention distribution

    - The **attention output** mostly contains information from the hidden states that received high attention.

  - 用 $[a_t; s_t] \in \mathbb{R}^{2h}$​​ 来生成  $\hat{y}_1$​, 其中 $s_t$​​ 是decoder hidden state.

- today we’ll think about attention **within a single sentence**. (self-attention)

### 1.4.2 More properties 

- Number of unparallelizable operations does not increase sequence length.

- Maximum interaction distance: O(1), since all words interact at every layer!

  <img src="/Users/Wei/Documents/NLP/NLP/transformer/attention_prop.png" alt="attention_prop" style="zoom:50%;" /> 

**Idea** 💡 用 self-attention 来填充 encoder 和 decoder 的block.

# 2. Architecture

## 2.1 High Level 

- encoder  和 decoder  都是由 N = 6 个identical layer stack 在一起得到的

<img src="/Users/Wei/Documents/NLP/NLP/transformer/architecture1.png" alt="architecture1" style="zoom:50%;" />

- encoder 每个layer 包含 两个 sub-layer :

  - multi-head self-attention 
  - A simple, position- wise fully connected feed-forward network

- decoder 每个layer 包含 三个 sub-layer :

  - multi-head self-attention 
  - Encoder- Decoder Attention
    - similar what attention does in seq2seq model 
  - A simple, position- wise fully connected feed-forward network

  <img src="/Users/Wei/Documents/NLP/NLP/transformer/encoder&decoder.png" alt="encoder&decoder" style="zoom:50%;" />

- 除了上面两层，还增加了 **residual connections** around each of the sub-layers, followed by layer **normalization**.

<img src="/Users/Wei/Documents/NLP/NLP/transformer/architecture2.jpeg" alt="architecture2" style="zoom:100%;" />

## 2.1 Self-Attention (1, 2, 3)

### 2.1.1 Self-Attention

#### 2.1.1.1 原理

- Attention operates on **queries**, **keys**, and **values**.

- Self-Attention 把一个单词的嵌入向量映射成三个相同维度的新向量：**查询 Query**、**键 Key**、**值 Value**

- **Focus on word $x_i$: ** 

  - Step 1. Compute **key-query** affinities (dot-product)

    - 
      $$
      e_{ij} = q_i^Tk_j
      $$

    -  当处理某一个单词时，这个单词就会用自己的 **Query** ($q_i$) 进行查询，跟当前序列中，所有单词的键 **Key** 进行相似度比对，得到相似度向量。这个向量的维度和当前句子里的单词数量相等

  - Step 2. Compute attention weights from affinities (softmat)

    - $$
      \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{j'}\exp(e_{ij'})}
      $$

    - 把上一步的value 通过 softmax 转化成 probability。每一个元素就代表对应位置的单词和发出 Query 的单词之间的相似度，值越大，相似度越高

  - Step 3 Compute outputs as weighted sum of **values** 

    - $$
      \text{output}_i = \sum_{j}\alpha_{ij}v_j
      $$

    - 用相似度向量对每个单词的值 **Value** 进行加权求和了，最终得到一个新的向量，这个向量就融入了相关单词的信息

- Matrix version

  - Let $x_1,...,x_T$​ be input vectors to the Transformer encoder;  $x_i \in \mathbb{R}^d$​

  - $X = [x_1; ...; x_T] \in \mathbb{R}^{T\times d}$​​ 

  - The keys, queries and values are 

    - $q_i = W^Q x_i$ , where $W^Q \in \mathbb{R}^{d \times d}$ is the query weight matrix
      - $Q = XW^Q \in \mathbb{R}^{T\times d}$ 代表 Query 矩阵
    - $k_i = W^K x_i$​​​​ , where $W^K \in \mathbb{R}^{d \times d}$​​​​​ is the key weight matrix
      - $K = XW^K \in \mathbb{R}^{T\times d}$ 代表 Key 矩阵
    - $v_i = W^V x_i$​ , where $W^V \in \mathbb{R}^{d \times d}$​ is the value weight matrix
      - $ V = XW^V \in \mathbb{R}^{T\times d}$ 代表 Value 矩阵
    - These matrices allow different aspects of the 𝑥 vectors to be used/emphasized in each of the three roles. 

  - **Step1** query-key dot products

  - **Step 2** Softmax & weighted average
    $$
    \text{Attention}(Q, K, V) = \text{softmax}(QK^T)V.
    $$
    下图截自PPT，XQ 对应 Q，XK 对应 K，XV 对应 V

    <img src="/Users/Wei/Documents/NLP/NLP/transformer/matrix_version.png" alt="matrix_version" style="zoom:50%;" />

### 2.1.2 Scaled Dot-Product Attention

#### 2.1.2.1 Architecture

<img src="/Users/Wei/Documents/NLP/NLP/transformer/scaled_dot_product.png" alt="scaled_dot_product" style="zoom:30%;" />

- 整体结构和上面一致，增加scale 层

#### 2.1.2.2 Why scaled?

- **Problem 1** in self-attention:  $QK^T$​ 方差大，影响训练稳定性.
  - 设 $q\in \mathbb{R}^d$ 是 Q 中一个单词的向量， $k\in \mathbb{R}^d$是 K 中一个单词的向量。
  - 假设 $q, k$ 中的每一维都是均值为 0，方差为 1 的独立同分布。
  - 由统计学知识可得 $q\cdot k = \sum{i=1}^dq_ik_i$ 的均值为 0，方差为 $d_k$ 
  - 如果模型中某一层的方差过大时，训练就会不稳定。

- **Problem 2** When dimensionality 𝑑 becomes large, dot products between vectors tend to become large.

  - Because of this, inputs to the softmax function can be large, making the gradients small.

- **Solution: Scaled by $\sqrt{d}$​** 使得方差归 1

  - $$
    \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V.
    $$

#### 2.1.2.3 Illustration 

##### a. Elementwise

<img src="/Users/Wei/Documents/NLP/NLP/transformer/self-attention-output.png" alt="self-attention-output" style="zoom:80%;" /> 

##### b. Matrix

- Step 1<img src="/Users/Wei/Documents/NLP/NLP/transformer/self-attention-matrix-calculation.png" alt="self-attention-matrix-calculation" style="zoom:45%;" />-Step2<img src="/Users/Wei/Documents/NLP/NLP/transformer/self-attention-matrix-calculation-2.png" alt="self-attention-matrix-calculation-2" style="zoom:40%;" />

### 2.1.3 Multi-head attention (1)

#### 2.1.3.1 Why need multi-head?

-  **Value 加权求和会降低词语分辨率**
  - 对于一个单词，如果他依赖的其他单词只有一个，那加权求和效果很好，但实际情况是多个。即使多个依赖单词的权重都很高，在加权求和后，神经网络也无法清晰地使用多个单词，只能笼统地使用一个“语境”。
- What if we want to look in multiple places in the sentence at once? 
  - For word 𝑖, self-attention “looks” where $q_i^Tk_j$​ is high, but maybe we want to focus on different 𝑗 for different reasons (query)?

#### 2.1.3.2 Solution

<img src="/Users/Wei/Documents/NLP/NLP/transformer/multi-head_attention.jpeg" alt="multi-head_attention" style="zoom:29%;" />

- Equation:
  $$
  \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
  $$
  其中  $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$, 

### 2.1.4 Masked Multi-Head Attention (2)

#### 2.1.4.1 Why 'masked'?

- Need to ensure we don’t “look at the future” when predicting a sequence
- 

#### 2.1.4.2 Solution : masked

- <img src="/Users/Wei/Documents/NLP/NLP/transformer/masked.jpeg" alt="masked" style="zoom:50%;" />
- 让当前单词及之前的单词向量通过，当前单词以后的设为 $-\infty$​
- 上图中，在 Mask 里面，灰色填充表示通过，白色表示屏蔽，Attention matrix 经过 Mask 后，通过的元素不变，不通过的元素用负无穷替换。

### 2.1.5 Encoder-Decoder Attention

<img src="/Users/Wei/Documents/NLP/NLP/transformer/encoder-decoder.png" alt="encoder-decoder" style="zoom:50%;" />

#### 2.1.5.1 Why need this layer?

**Ans** 把 encoder 的编码信息（K, V）融入 decoder 中：

- K，V 来自Encoder
- Q 来自 Decoder

#### 2.1.5.2 Details

- Let $h_1, ..., h_T \in \mathbb{R}^d$​​ be output vectors from the Transformer **encoder**;
- Let $z_1, ..., z_T \in \mathbb{R}^d$​ be input vectors from the Transformer **decoder**;
- keys and values are drawn from the **encoder** (like a memory): 
  - $𝑘_𝑖 = W^𝐾ℎ_𝑖 , 𝑣_𝑖 = W^𝑉h_𝑖$
- queries are drawn from the decoder, $q_i = W^Qz_i$ .

## 2.2 Feed Forward Network 

- 普普通通的全连接层，用来映射**每个单词**的向量。Transformer 用了两个全连接层，中间使用了一个 ReLU 激活函数，公式如下：
  $$
  \text{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2
  $$

- **Question** 为什么需要FFN

- **Ans** Note that there are **no elementwise nonlinearities** in self-attention

- stacking more self-attention layers just re-averages value vectors

<img src="/Users/Wei/Documents/NLP/NLP/transformer/ffn.png" alt="ffn" style="zoom:30%;" />

这里需要注意下：

- Attention 处理单词之间的交互;
- FFN 则处理一个单词向量的表示，不涉及单词交互，从公式上就可以看出来，公式里面只有一个单词向量 ![[公式]](https://www.zhihu.com/equation?tex=x) ，没有其他单词向量。

## 2.3 Add & Norm

### 2.3.1 Residual

- **目的**：Residual connections are a trick to help models train better.

- <img src="/Users/Wei/Documents/NLP/NLP/transformer/residual.png" alt="residual" style="zoom:50%;" />

- $$
  X^{(i)} = X^{(i-1)} + \text{Layer}(X^{(i-1)})
  $$

- Residual connections are thought to make the loss landscape considerably smoother (thus easier training!)

- <img src="/Users/Wei/Documents/NLP/NLP/transformer/residual2.png" alt="residual2" style="zoom:50%;" />

### 2.3.2 Layer Normalization

- **目的**：Layer normalization is a trick to help models **train faster**

- **Idea**: cut down on uninformative variation in hidden vector values by normalizing to unit mean and standard deviation **within each layer**

- **Equation**

  - $x\in \mathbb{R}^d$ : an individual (word) vector

  - $\mu = \frac{1}{d} \sum_{j=1}^d x_j \in \mathbb{R}$ is the mean

  - $\sigma = \sqrt{\frac{1}{d}\sum_{j=1}^d(x_j-\mu)^2} \in \mathbb{R}$​​​ is the standard deviation

  - $\gamma \in \mathbb{R}^d$​, $\beta \in \mathbb{R}^d$​​ are learned "gain" and "bias" parameters (can omit!)

  - Layer normalization:

    <img src="/Users/Wei/Documents/NLP/NLP/transformer/layer_normal.png" alt="layer_normal" style="zoom:50%;" />



### 2.3.3 Illutration

<img src="/Users/Wei/Documents/NLP/NLP/transformer/transformer_resideual_layer_norm_2.png" alt="transformer_resideual_layer_norm_2" style="zoom:50%;" />

## 2.4 Positional Encoding

### 2.4.1 Why need this?

- **Problem** self-attention is an **unordered function of its inputs**
- Idea:  Add position representations to the input to specify the sequence order
  - **sequence index** -- vector

### 2.4.2 Positional Encoding

Let $p_i \in \mathbb{R}^d$, for $i \in \{1, 2,...,T\}$​​ are the position vectors

=> word i (position = i) input 更新为 $x_i + p_i$​

<img src="/Users/Wei/Documents/NLP/NLP/transformer/transformer_positional_encoding_vectors.png" alt="transformer_positional_encoding_vectors" style="zoom:50%;" />

### 2.4.3 Sinusoidal position representations

<img src="/Users/Wei/Documents/NLP/NLP/transformer/position1.png" alt="position1" style="zoom:50%;" />

- Pros:

  - Periodicity indicates that maybe “absolute position” isn’t as important 

    - 两个间隔为k的位置向量，他们可以由一个只与 k 相关的矩阵进行转换

    - $$
      p_{i+k} = T(k)p_i
      $$

      其中 $\lambda_{2i} = \frac{1}{10000^{2i/d}}$ , $T(k)$ 如下 只与k相关

      <img src="/Users/Wei/Documents/NLP/NLP/transformer/tk.png" alt="tk" style="zoom:50%;" />

      

  - Maybe can extrapolate to longer sequences as periods restart! 

- Cons:

  - Not learnable; also the extrapolation doesn’t really work!



### 2.4.4 Learned absolute position representations

- Let all $p_i$ be learnable parameters! 
- Learn a matrix $p \in \mathbb{R}^{d\times T}$ , and let each $p_i$ be a column of that matrix!
- Pros:
  - Flexibility: each position gets to be learned to fit the data
- Cons: 
  - Definitely can’t extrapolate to indices outside 1, … , 𝑇. 
- Most systems use this!

## 2.5 Output

<img src="/Users/Wei/Documents/NLP/NLP/transformer/transformer_decoder_output_softmax.png" alt="transformer_decoder_output_softmax" style="zoom:50%;" />



## 2.6 Summary

<img src="/Users/Wei/Documents/NLP/NLP/transformer/transformer_resideual_layer_norm_3.png" alt="transformer_resideual_layer_norm_3" style="zoom:80%;" />

# 3. Performance

- Machine Translation from the original Transformers paper!

  <img src="/Users/Wei/Documents/NLP/NLP/transformer/result_MT.png" alt="result_MT" style="zoom:50%;" />

-  Document generation

  <img src="/Users/Wei/Documents/NLP/NLP/transformer/result_DG.png" alt="result_DG" style="zoom:50%;" />

# 4. Drawbacks and variants of Transformers

## 4.1 Drawbacks

- **Quadratic compute in self-attention**: 
  - Computing all pairs of interactions means our computation grows **quadratically** with the sequence length! 
  - For recurrent models, it only grew linearly! 

- **Position representations**:
  - Are simple absolute indices the best we can do to represent position? 
  - Relative linear position attention [Shaw et al., 2018]
  - Dependency syntax-based position [Wang et al., 2019]



## 4.2 Quadratic computation

- The total number of operations grows as $O(T^2d)$, where T is the sequence length, d is the dimension

  ​	<img src="/Users/Wei/Documents/NLP/NLP/transformer/computation.png" alt="computation" style="zoom:50%;" />

- Bad when T is large (e.g. work on long documents, $T \geq 10,000$​​) 

- Can we build models like Transformers without paying the  $O(T^2)$​ all-pairs self-attention cost? 
  - For example, Linformer [Wang et al., 2020]
    - <img src="/Users/Wei/Documents/NLP/NLP/transformer/linformer.png" alt="linformer" style="zoom:50%;" />
  - For example, BigBird [Zaheer et al., 2021]
    - <img src="/Users/Wei/Documents/NLP/NLP/transformer/bigbird.png" alt="bigbird" style="zoom:50%;" />







Transformer  = 一个完全依赖 Attention 的模型，用 Positional Encoding 替代了 RNN 的循环结构。



## Attention

一句话中的词语存在相互依赖关系，比如指代关系、修饰关系等，所以单个词语是很难表达完整含义的。因此在 NLP 处理中需要有一种机制来刻画词语依赖，这个机制就是 **Attention**。

我们知道在神经网络中，每个单词都被映射成了一个嵌入向量，这个向量蕴含着一个单词的全部含义。那么 Attention 要做的就是把其他相关单词的信息融入到当前的嵌入向量中，让这个单词的嵌入向量随着模型层数的加深，表述地更加清晰、完整。 



self-attention:

you're re-expressing yourself in certain terms of a weighted combination of your entire neighborhood.

- attention is permutation invariant（因为用attention来记录信息，但attention与位置无关，所以要加position representation）
  - 增加了position representation 来maintain order
  - 

## Transformer Network

### Motivation 

<img src="/Users/Wei/Documents/NLP/NLP/seq2seq/motivation.png" alt="motivation" style="zoom:50%;" />

- 上面的模型都说**sequential** 的，即为了预测最终结果，需要一次预测一个；
- transformer可以并行计算



### Intuition : Attention + CNN

<img src="/Users/Wei/Documents/NLP/NLP/seq2seq/transformer_intuition.png" alt="transformer_intuition" style="zoom:50%;" />

### self-attention

<img src="/Users/Wei/Documents/NLP/NLP/seq2seq/self_attention_1.png" alt="self_attention_1" style="zoom:50%;" />

- query q: let you ask a question about the word 

  - $q^{<3>}$ = question at `I'Afrique` ,e.g. what's happending there

- key k: looks at all of the other words, and by the similarity to the query, 

  helps you figure out which words gives the most relevant answer to that question.  

  - $q^{<3>} \cdot k^{<1>}$ = how good is `Jane` as an answer to the question $q^{<3>}$

- value allows the representation to plug in how visite should be represented within A^3, within the representation of Africa. 

  - This allows you to come up with a representation for the word Africa that says this is Africa and someone is visiting Africa. 

- <img src="/Users/Wei/Documents/NLP/NLP/seq2seq/self_attention_2.png" alt="self_attention_2" style="zoom:50%;" />



### muti-head attention

- **Idea** is basically just a big for-loop over the self attention mechanism

<img src="/Users/Wei/Documents/NLP/NLP/seq2seq/multihead_attention_1.png" alt="multihead_attention_1" style="zoom:50%;" />

<img src="/Users/Wei/Documents/NLP/NLP/seq2seq/multihead_attention_2.png" alt="multihead_attention_2" style="zoom:50%;" />

- heads 可以并行计算

### Transformer



