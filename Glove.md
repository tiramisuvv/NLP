# 1. 共现矩阵 Co-occurrence Matrix 

## 1.1. 两种共现矩阵

### 1.1.1 Word- Document Matrix

**假设/猜想** 关联的单词会经常出现在同一个文章中。

**例子**🌰 

- "banks", "bonds", "stocks", "money", etc. 可能经常出现在同一个文章中；
-  "banks", "octopus", "banana", and "hockey"不大可能会连续出现。

**定义** $X  = \{X_{ij}\}$ 是一个word- document matrix：

1. 初始化X为一个$\mathbb{R}^{|V|\times M}$的0矩阵（$X_{ij} = 0$)，其中$|V|$是词汇量，M是文章数

2. 遍历所有文章，每当单词 $w_i$ 出现在文章j中，就$X_{ij}+= 1$ 

 **用途举例** general topics (all sports terms will have similar entries) leading to “Latent Semantic Analysis”

### 1.1.2 Window base co-occurence matrix

**Idea 💡** 计算每个单词在特定大小的窗口中出现的次数，得到$\mathbb{R}^{|V|\times |V|}$的共现矩阵

1. 与word2vec类似
2. 关注单词周围的某个window内的单词，获取一些语法和语义（syntactic and semantic）信息

**例子🌰**

窗口长度为1，数据为以下句子

- I like deep learning

- I like NLP

- I enjoy flying

  <img src="/Users/weiwang/Documents/NLP/glove/cooccirence_example.png" alt="cooccirence_example" style="zoom:50%;" />



**缺点**

1. 矩阵维度经常发生变化（新单词，新语料库的增加）
2. 矩阵稀疏（很多单词不会共现）
3. 高维度 $\approx 10^6 \times 10^6$
4. 基于SVD的方法计算复杂度高（$m\times n$ 的矩阵计算SVD的复杂度是 $O(mn^2)$)
5. 需要在$X$上加入一些技巧来处理词频的极端不平衡

**部分解决方法**

- 忽略功能次，如"the","he","has"等
- 使用ramp window，即根据文档中单词之间的距离对共现计数进行加权(不懂，TODO)
- 使用 Pearson correlation并将负计数设置为0，而不是原始计数



## 1.2 怎么定义共现向量 （co-occurence vectors）

### 1.2.1 传统方法 Dimensionality Reduction on X 

#### 1.2.1.1 Idea 💡 

- 在一个固定的低维的稠密向量中，保存***大部分***重要信息

#### 1.2.1.2 构造方法

使用SVD方法将共现矩阵X分解为 $U\Sigma V^T$，其中$\Sigma$是特征值矩阵，U，V是对应于行和列的正交基。

<img src="/Users/weiwang/Documents/NLP/glove/svd.png" alt="svd" style="zoom:50%;" />

通过取前k个最大的特征值，对X进行降维。

<img src="/Users/weiwang/Documents/NLP/glove/svd2.png" alt="svd2" style="zoom:50%;" />



### 1.2.2 Hacks on X

- 在原始计数矩阵上，用SVD效果不好！
- 按比例调整计数对效果有很大提升-- **Scaling the counts**
  - Problem: 功能词（如"the","he","has"等）出现频率太高，导致对语法有太多影响
    - $\log(\text{frequencies})$
    - $\min(x,t),  \text{ with } t = 100$ ：对高频词（频次>t）设置固定频次
    - 忽略功能词
- 使用ramp window，即基于在文档中词与词之间的距离给共现计数加上一个权值
- 使用 Pearson correlation并将负计数设置为0

**=> Idea 💡** 对计数进行处理是可以得到有效的词向量的

<img src="/Users/weiwang/Documents/NLP/glove/coals_model.png" alt="coals_model" style="zoom:50%;" />

⚠️**Interesting semantic patterns emerge in the vectors**:

语义向量基本上是***线性***的，虽然有一些摆动，但是基本是存在动词和动词实施者的方向。

如果能够建立这样的“线性关系”，那么可以得到好的类比



### 1.2.3 总结&对比

👈：基于计数的方法：使用整个矩阵的全局统计数据来直接估计

👉：基于预测的方法：定义概率分布并试图预测单词

<img src="/Users/weiwang/Documents/NLP/glove/count_baased_vs_prediction.png" alt="count_baased_vs_prediction" style="zoom:50%;" />

# 

# 2 Glove

## 2.1 Idea

目标：把上述两种方法结合起来，用NN+某种计数矩阵

**Crucial Insight** Ratio of co-occurrence probablilities can encode meaning components

<img src="/Users/weiwang/Documents/NLP/glove/co-occurrence_prob1.png" alt="co-occurrence_prob1" style="zoom:50%;" />



<img src="/Users/weiwang/Documents/NLP/glove/co-occurrence_prob2.png" alt="co-occurrence_prob2" style="zoom:50%;" />



重点是Difference between  co- occurrence probabilities

[TODO]**希望 Ratio of co- occurrence probabilities TO BE linear!**

Encoding meaning components in vector differences

## 2.2 模型

**Question**: How can we capture ratios of co-occurrence probabilities as linear meaning components in a word vector space

**Answer**: Use a **Log-bilinear model**:
$$
w_i\cdot w_j = \log P(i|j)
$$
这样， vector differences就是
$$
w_x\cdot (w_a- w_b) = \log \frac{P(x|a)}{P(x|b)}
$$


## 2.3 目标函数

[TODO]怎么得到的（ref https://zhuanlan.zhihu.com/p/60208480）
$$
J = \sum_{i, j = 1}^{V} f(X_{ij})\left(w_i^T \tilde{w}_j+b_i+\tilde{b}_j-\log X_{ij}\right)^2
$$

- Bias term 如果word common



## 2.4 模型结果

### 2.4.1 Nearest words

<img src="/Users/weiwang/Documents/NLP/glove/glove_result.png" alt="glove_result" style="zoom:50%;" />


### 2.4.2 Visualizations

#### 2.4.2.1 Women -- man

<img src="/Users/weiwang/Documents/NLP/glove/glove_visualization1.png" alt="glove_visualization1" style="zoom:50%;" />

#### 2.4.2.2 Company-CEO

<img src="/Users/weiwang/Documents/NLP/glove/glove_visualization2.png" alt="glove_visualization2" style="zoom:50%;" />

#### 2.4.2.3 Comparatives and Superlatives

<img src="/Users/weiwang/Documents/NLP/glove/glove_visualization3.png" alt="glove_visualization3" style="zoom:50%;" />
# 3. How to evaluate word vectors

## 3.1 概述

**Answer** Intrinsic vs. extrinsic 内在与外在

- Intrinsic:
  - Evaluation on a specific/intermediate **subtask**
  - Fast to compute 
  - Helps to understand that system 
  - Not clear if really helpful unless correlation to real task is established 
- Extrinsic: 
  - Evaluation on a real task
  - Can take a long time to compute accuracy
  - Unclear if the subsystem is the problem or its interaction or other subsystems 
  - If replacing exactly one subsystem with another improves accuracy -> Winning!

## 3.2 Intrinsic word vector evaluation

### 3.2.1 Word Vector Analogies
<img src="/Users/weiwang/Documents/NLP/glove/word_vector_analogies.png" alt="word_vector_analogies" style="zoom:50%;" />

dea 💡** Evaluate word vectors by how well their cosine distance after addition captures intuitive semantic and syntactic analogy questions （余弦距离能多好的捕捉到语义和句法类比问题）

<img src="/Users/weiwang/Documents/NLP/glove/word_vector_analogies2.png" alt="word_vector_analogies2" style="zoom:30%;" />

- 要从搜索中去掉input单词
- **Problem** 如果信息非线性，怎么办？[TODO]