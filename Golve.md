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

