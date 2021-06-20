
# 1. Introduction

## 1.1 定义

NLP = Natural language processing 


## 1.2 How do we represent the meaning of a word?

**meaning**

- 用一个词，词组表示的概念
- 一个人想用语音、符号等表达的想法
- 文章，艺术等作品中表达的思想

<img src="./word2vec/semantics_def.png" alt="semantics_def" style="zoom:30%;" />

## 1.3 How do we have usable meaning in a computer?

### 1.3.1 WordNet

#### Idea 💡 

- **用包含同义词，上位词(hypernyms)的词典**

#### 例子 🌰

![wordnet](./word2vec/wordnet.png)

#### 缺点 ❎

- 忽略词之间的细微差别

  - 'proficient' 不完全等同于 'good'

- 缺少单词的新含义,不可能持续更新

- 需要人工创造，调整

- 不能计算单词相似度

  
### 1.3.2 one-hot vector

#### Idea 💡 

在传统NLP里，把单词作为离散符合：用one-hot vector表示

#### Definition

用只有一位为1，其余都是0的，长度为 `|V|` 的向量表示，(其中`|V|` = 词汇量）。

#### 例子 🌰

```
motel = [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0]
hotel = [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0]
```

#### 缺点 ❎

- 不能表达词之间的相似性。
  -   任意两个one-hot 向量都垂直
-   向量维度很大，非常sparse

#### Solution ✅

- 用 WordNet 中的同义词来获得相似性（但不可行，比如WordNet中不可能收录所有同义词）  
- **Instead : learn to encode similarity in the vectors themselves**




## 1.4 Distributional semantics
#### Idea 💡 根据上下文推测单词含义
#### Definition

- **Distributional semantics** := A word’s meaning is given by the words that frequently appear close-by

#### 例子 🌰

![banking_meaning](./word2vec/banking_meaning.png)banking_meaning.jpg

- **context** = When a word w appears in a text, its ***context*** is the set of words that appear nearby (within a fixed-size window).

- If you can explain what context it's correct to use a certain word, versus in what context would be the wrong world to use. => understand the meaning of the word 

- ##### 注意⚠️ 这里假设：中心词c的含义只与上下文相关，i.e. 与远距离的单词无关。

  - 在实际中，不完全成立。比如人称代词可能指代的是几句甚至几段之前的人物。

## 1.5 语言模型(Language Models)
#### Idea 💡

语言模型是用来计算一个句子的概率的概率模型。比如*"The cat jumped over the puddle."* 这样完整且合理的句子应该有高的概率，而 *"stock boil fish is toy"* 有低概率。

对于任意一列n个单词，它的概率用 $P(w_1, w_2, ..., w_T)$ 表示。

一般的，根据Bayes 公式，
$$
P(w_1,...,w_T) = P(w_1)P(w_2|w_1)P(w_3|w_1,w_2)...P(w_T|w_1,...,w_{T-1})
$$
其中条件概率 $p(w1), p(w_2|w_1), ..., p(w_T|w_1,...,w_{T-1})$ 是模型的参数。对于一个长度为$T$ 的句子，需要计算 $T$  个参数。在词汇量为$|V|$ 的语料库，任意长度 T 的句子共用 $|V|^T$ 种可能，进一步就有 $T \cdot |V|^T$ 个参数。（这里忽略了重复参数，只看量级就好）。

### 1.5.1 Unigram 模型

假设单词之间完全独立，那么
$$
P(w_1,...,w_T) = \prod_{i = 1}^T P(w_i)
$$
显然，我们知道这个是不合理的，因为当前单词肯定是依赖于前一个单词的。

### 1.5.2 Bigram 模型

假设当前单词只依赖于它前一个单词，那么
$$
P(w_1,...,w_T) = \prod_{i = 2}^TP(w_i|w_{i-1})
$$

### 1.5.3 N-gram 模型

类似的，假设当前单词依赖于它前n-1个单词，称为N- gram 模型。类似的
$$
P(w_1, ..., w_T) = \prod_{i=n}^{T} P(w_i|w_{i-n+1},...,w_{i-1})
$$
进一步，根据大数定律，当语料库足够大时，
$$
P(w_i|w_{i-n+1}, ..., w_{i-1}) \approx \frac{\text{count}(w_{i-n+1}, ..., w_i)}{\text{count}(w_{i-n+1}, ..., w_{i-1})}
$$
以上面Bigram 为例 (n=2)，有
$$
P(w_i|w_{i-1}) \approx \frac{\text{count}(w_{i-1},w_i)}{\text{count}(w_{i-1})}
$$
不仅使得单个参数的统计变得更容易（统计时需要匹配的词串更短），也使得参数的总数变少了。

|     n      |   模型参数的数量   |
| :--------: | :----------------: |
| 1(unigram) |   $2\times 10^5$   |
| 2(bigram)  | $4\times 10^{10}$  |
| 3(trigram) | $8\times 10^{15}$  |
| 4(4-gram)  | $16\times 10^{20}$ |

上表给出了n-gram模型中模型参数数量随着的逐渐增大而变化的情况，其中假定词汇量为 $|V| = 200,000$(汉语的词汇量大致是这个量级)。实际应用中，最多采用n=3的trigram模型。

#### 平滑化

考虑两个问题（极端情况）：

- 如果 $\text{count}(w_{i-n+1}, ..., w_i) = 0$，是否认为 $P(w_i|w_{i-n+1}, ..., w_{i-1}) = 0$?

- 如果$\text{count}(w_{i-n+1}, ..., w_i) = \text{count}(w_{i-n+1}, ..., w_{i-1}) $, 是否认为 $P(w_i|w_{i-n+1}, ..., w_{i-1}) = 1$?



【TODO】平滑化

总结起来，n-gram模型是这样一种模型，其主要工作是在语料中统计各种词串出现的次数以及平滑化处理。概率值计算好之后就存储起来，下次需要计算一个句子的概率时，只需找到相关的概率参数，将它们连乘起来就好了。



### 1.5.4 目标函数

对于统计语言模型而言，利用最大似然，可把目标函数设为：
$$
\prod_{w \in C} P(w|\text{contex}(w))
$$
其中 C表示 语料库（Corpus)，Contex(w)表示单词w的上下文



# 2. Word Embedding 词嵌入


## 2.2 Word vector
### 2.2.1 定义

对于词典D里的任意单词w，都指定一个固定长度的向量$v(w) \in \mathbb{R}^{n}$，就称 $v(w)$ 是w的词向量，n为词向量的长度。 

- 这个向量被称为词向量（word vectors)，词嵌入（word embedding）或者 word representations

显然，上面提到的 one-hot vector 就是最简单的词向量。基于one-hot，我们希望能够得到满足下面条件的词向量：

- 长度 $n \ll |V|$ 
- 词向量是稠密的
- 词语之间的相似性可以通过词向量来体现 => w的词向量与它上下文中的单词的向量相似

### 2.2.2 例子 🌰

<img src="./word2vec/banking_vector.png" alt="banking_vector" style="zoom:50%;" />

banking_vector.jpg

进入计算，学习词向量的算法前，我们看两个词向量的可视化。（为了方便，下图把9维词向量投射到二维空间）

![word_vector_visualization](./word2vec/word_vector_visualization.png)word_vector_visualization.jpg

- 含义相近的词，有一定的聚集性，表现了词之间的相似性

这里的 **相似性**是指**“结构”相似性**：如果两个词出现在同样的上下文，我们就当做它们相似。如上图come 和 go，含义上是反义词，但在词向量空间很接近。

![word_vector_visualization2](./word2vec/word_vector_visualization2.png)

word_vector_visualization2.jpg

- 分别画出英语和西语的数字单词的词向量，同样发现有很高的相似性。



## 2.3 Word2vec 

### 2.3.1 Overview
Word2vec (Mikolov et al. 2013) 是一个学习词向量的框架。

#### Idea 💡
- 我们有一个大的语料库（大量文本）（a large corpus of text）
- 词汇表中的每个单词都由一个向量表示
- 遍历文本中的每个位置 t，都有对应的中心词（center word） c 和上下文（ context (“outside”)） o
- 通过单词 c 和 o 的词向量的相似度，来计算给定c出现o的概率（反之亦然）
- 通过不断调整词向量，最大化这个概率



"完形填空"

<img src="/Users/Wei/Library/Application Support/typora-user-images/Screen Shot 2021-06-20 at 8.55.21 AM.png" alt="Screen Shot 2021-06-20 at 8.55.21 AM" style="zoom:50%;" />

下图为$P(w_{t+j}|w_t)$的计算过程，window size = 2， 中心词 = into

![into_example](./word2vec/into_example.png)into_example.jpg

<img src="./word2vec/banking_example.png" alt="banking_example" style="zoom:70%;" />

banking_example



### 2.3.2 Details

Word2vec 包含：
- **两个学习词向量的模型** 
  
  - CBOW 通过上下文预测中心词
  - Skip-gram 通过中心词预测上下文

 cbow&skipgram.png.jpg

![cbow&skipgram](./word2vec/cbow&skipgram.png) 由图可见，两个模型都包含三层：**输入层**、**投影层**和**输出层**

- **两个训练方法** 降低模型学习过程中的运算量
  
  - negative sampling 负采样
  - hierarchical softmax

### 2.3.3 Continuous Bag of Words Model (CBOW)

#### 2.3.3.1 结构
1. simple CBOW
simple_cbow.jpg

2. Multi-Word Context Model
multi_cbow.jpg

cbow.jpg

**目标** 对每一个单词 w，我们要学习2个向量
- v (输入向量）当 w 在上下文中；
- u（输出向量）当 w 是中心词 

**例子** 🌰
有下面的句子：

*"The cat jumped over the puddle."*

希望通过上下文 {"The", "cat", "over", "the", "puddle"} 来预测/生成中心词 "jumped"

***In math*** 希望 通过{"The", "cat", ’over", "the’, "puddle"}的词向量得到向量 = "jumped"的词向量

**Notation for CBOW Model**
- `w_i` : Word i from vocabulary V
- `V \in R^{n\times |V|}: Input word matrix
- `v_i` : i-th column of V , the input vector representation of word `w_i`
- `U \in R^{|V|\times n}: Output word matrix
- `u_i` : i-th row of U , the output vector representation of word `w_i`

我们用 `w_{c-m}, ..., w_{c-1}, w_{c+1},...,w_{c+m}` 代表上下文， `w_c`代表中心词。假设已有所有词的one-hot vector表示。
Input:x, output:y

1. 生成上下文中的单词的 one-hot 向量: `x^{c-m}, ..., x^{c-1}, x^{c+1},...,x^{c+m} \in R^{|V|}`.
2. 得到上下文的词向量：`v_{c-m} = Vx^{c-m}, ..., v_{c-1} = Vx^{c-1}, v_{c+1} = Vx^{c+1},...,v_{c+m} = Vx^{c+m} \in R^{n}`
3. 计算平均值 `\hat{v} = \frac{v_{c-m} +...+ v_{c-1} + v_{c+1} + ...+v_{c+m}}{2m} \in R^{|V|}`

  1. 有的地方直接求和，不影响

4. 生成分数向量 `z = U\hat(v) \in R^{|V|}`

  - 这样，由于相似向量的点积大，所以为了得到高分，会向相似单词的词向量靠近。

6. 通过softmax函数把分数转化为概率 `\hat{y} = \text{softmax}(z) \in R^{|V|}`
7. 我们希望生成的概率`\hat{y}\in R^{|V|}` 与真实 `y\in R^{|V|}` 相匹配，注意这里`y` 也是one-hot。

#### 2.3.3.2 目标函数 Obejective function 
For each position 𝑡 = 1, ... , 𝑇, predict center word `w_c`, given the context words within a window of fixed size m `w_{c-m}, ..., w_{c-1}, w_{c+1},...,w_{c+m}`. 

Data likelihood:

``
L(\theta) = \Pi_{c = 1} ^T  P(w_c|w_{c-m}, ..., w_{c-1}, w_{c+1},...,w_{c+m})
``

``
L(\theta) = \Pi_{t = 1} ^T \Pi_{-m\leq j \leq m, j\neq 0} P(w_c|w_{c-m}, ..., w_{c-1}, w_{c+1},...,w_{c+m})
``


- `\theta` 包含所有需要优化的参数（i.e. 两个词向量矩阵）

The objective function `J_{\theta}` is the (average) negative log likelihood:

``
J = - \frac{1}{T}\log L(\theta) 
``

**Question: How to calculate `P(o|c)`

- `𝑃(𝑢_{𝑝𝑟𝑜𝑏𝑙𝑒𝑚𝑠} | 𝑣_{𝑖𝑛𝑡𝑜}) short for P(𝑝𝑟𝑜𝑏𝑙𝑒𝑚𝑠 | 𝑖𝑛𝑡𝑜 ; 𝑢_{𝑝𝑟𝑜𝑏𝑙𝑒𝑚𝑠}, 𝑣_{𝑖𝑛𝑡𝑜},\theta)


prob.jpg

#### 2.3.3.3 Gradiant descent and stochastic gradient descent


### 2.3.4 Skip-gram

![into_example](./word2vec/into_example.png)



### 2.3.4.1 目标函数  Obeject function 

对任意的位置 $𝑡 = 1, ... , 𝑇$, 给定中心词 w，预测它的上下文（window of fixed size m） 

Likelihood:
$$
L(\theta) = \prod_{t= 1}^T\prod_{-m\leq j\leq m, j\neq 0}P(w_{t+j}|w_t; \theta)
$$
其中用$\theta$ 表示所有（需要优化的）参数，那么目标函数 $J(\theta)$ 是
$$
J(\theta) = - \frac{1}{T}\log L(\theta) = -\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m, j\neq 0}\log P(w_{t+j}|w_t; \theta)
$$


为了最小化目标函数 $J(\theta)$ ，有下面问题：

- **Question：如何计算$P(w_{t+j}|w_t; \theta)$？**

- **Answer** 对每个单词w，使用两个词向量

  - $v_w$ ，当w是中心词

  - $u_w$，当w是上下文

  - 对于中心词 c 和上下文单词 o(output)，有：
    $$
    P(o|c) = \frac{\exp(u_o^Tv_c)}{\sum_{w\in V} \exp(u_w^Tv_c)}
    $$
  
  <img src="./word2vec/prob.png" alt="prob" style="zoom:50%;" />
  
   - Softmax function $\mathbb{R}^{n} \rightarrow (0, 1)^n$:
     $$
     \text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^n \exp(x_j)}
     $$
  
     	- 把任意的$x_i$ 映射到一个概率分布 $p_i$
     	- "max" :能够让vector中最大的数$x_{max}$ 被取到的概率非常大
     	- "soft" 对于小的值$x_i$ 仍然有概率

### 2.3.4.2 训练模型

#### 2.3.4.2.1 Gradiant descent

- 沿着梯度下降的方向优化参数<img src="./word2vec/gd.png" alt="gd" style="zoom:30%;" />(gd.jpg)

- **Idea** 对现在的参数 $\theta$ ，计算目标函数 $J(\theta)$ 的梯度，然后沿着负向梯度小步向下。（重复这个过程，直到结束）<img src="./word2vec/gr2.png" alt="gr2" style="zoom:30%;" />

  

- 
  $$
  \theta^{new} = \theta^{old} - \alpha \nabla_{\theta} J(\theta)
  $$

  $$
  \text{i.e.}\quad \theta^{new}_j = \theta^{old}_j - \alpha \frac{\partial}{\partial \theta} J(\theta) \quad \text{for any parameter } \theta_j
  $$

  

  

- 这里，我们的参数$\theta$ ：总共$|V|$ 单词，每个单词都包含两个d维的向量，所以<img src="./word2vec/theta.png" alt="theta" style="zoom:30%;" />(theta.jpg)
- 下面我们需要对**所有**向量求导

#### 2.3.4.2.2 求导细节

$$
J(\theta) = -\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m, j\neq 0}\log P(w_{t+j}|w_t; \theta)
\quad \text{with } P(o|c) = \frac{\exp(u_o^Tv_c)}{\sum_{w\in V} \exp(u_w^Tv_c)}
$$

$$
\begin{align}
\frac{\partial}{\partial v_c}\log  P(o|c) &=
\frac{\partial}{\partial v_c}\log \frac{\exp(u_o^Tv_c)}{\sum_{w\in V} \exp(u_w^Tv_c)} \\
&= \frac{\partial}{\partial v_c}(u_o^Tv_c)-\frac{\partial}{\partial v_c}\log {\sum_{w\in V} \exp(u_w^Tv_c)} \\
&= u_o - \frac{1}{{\sum_{w\in V} \exp(u_w^Tv_c)}}\cdot {\sum_{x\in V} \exp(u_x^Tv_c)}u_x.
\end{align}
$$

注意到 $J(\theta)$ 
$$
\begin{align}
\frac{\partial}{\partial v_c}\log  P(o|c) 
&= u_o - \sum_{x\in V} \frac{{\exp(u_x^Tv_c)}}{{\sum_{w\in V} \exp(u_w^Tv_c)}}\cdot u_x\\
&=u_o-\sum_{x\in V}P(x|c)u_X \\
&= \text{Actual context word} - \text{Expected context word} 
\end{align}
$$
注意到，$u_o$ 是上下文对应的词向量，而$\sum_{x\in V}P(x|c)u_X$  是以期望形式存在的，模型“预测“的结果



## 2.4 基于Hierarchical Softmax的模型

回忆CBOW和

## 2.5  基于Negative Sampling的模型

回忆
$$
J(\theta) = -\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m, j\neq 0}\log P(w_{t+j}|w_t; \theta)\quad \text{with } P(o|c) = \frac{\exp(u_o^Tv_c)}{\sum_{w\in V} \exp(u_w^Tv_c)}
$$
注意到分母导致每次计算目标函数 $J(\theta)$都需要遍历整个词典 V，计算量是$O(|V|)$ 级别的（millions)。一个简单的想法是我们不去直接遍历计算，我们预估这个和。那么类比上面的内容有三项需要更新：

- 目标函数 $J(\theta)$
- 导数 $\nabla_{\theta} J(\theta)$
- 更新规则

（下面以skip gram为例）

### 2.5.1 Idea 💡

这里我们改变预测相邻单词这一任务：

- （左图）是原始的，我们根据中心词 not 预测它的上下文 thou 

- （右图）是根据输入的（中心词c，单词w），判断它们是否是邻居（0表示“不是邻居”，1表示“邻居”）。
  - w 是上下文： label = 1
  - w 是随机生成的噪声：label = 0

<img src="./word2vec/neg_sampling1.png" alt="neg_sampling1" style="zoom:40%;" /><img src="./word2vec/neg_sampling2.png" alt="neg_sampling2" style="zoom:40%;" />

也就是我们需要的模型从神经网络改为逻辑回归模型——因此它变得更简单，计算速度更快。

**Main idea**: train binary logistic regressions for a true pair (center word and a word in its context window) versus several noise pairs (the center word paired with a random word)

这个方法需要我们更新数据集的结构：增加标签列（0或者1）显然，对于(中心词，上下文)，标签列都是1。

<img src="./word2vec/neg_sampling3.png" alt="neg_sampling3" style="zoom:75%;" />

为了让机器能够正确学习，这里我们需要引入负样本 \- 不是邻居的单词样本。下一个问题就是负样本要怎么选取？简单的说我们从词汇表中基于词频随机抽取单词

<img src="./word2vec/neg_sampling4.png" alt="neg_sampling4" style="zoom:75%;" />

neg_sampling1.jpg
neg_sampling2.jpg
neg_sampling3.jpg
neg_sampling4.jpg

### 2.5.2 目标函数 $J(\theta)$

考虑一对中心词c和另一个单词w的词对 (w,c)，

- 如果(w,c)来自于语料库，或者是 w是c的上下文，用$P(D=1|w,c)$表示；

- 如果(w,c)不在语料库中，用$P(D=0|w,c)$表示；

- 对$P(D=1|w,c)$ 用Sigmoid函数建模：
  $$
  P(D=1|w,c; \theta) = \sigma(u_c^Tv_w) = \frac{1}{1+\exp(-u_c^Tv_w)}
  $$

- 

显然我们希望：

- 如果(w,c)来自于语料库，$\max_{\theta}P(D=1|w,c;\theta) $； 

- 如果(w,c)不在语料库中，$\max_{\theta}P(D=0|w,c;\theta)$ ;

  

得到似然函数
$$
\begin{align}
L(\theta) &= \prod_{(w,c)\in D}P(D=1|w,c;\theta) \prod _{(w,c)\in \tilde{D}}P(D=0|w,c;\theta)\\
&= \prod_{(w,c)\in D}P(D=1|w,c;\theta) \prod _{(w,c)\in \tilde{D}}(1-P(D=1|w,c;\theta))\\
&= \prod_{(w,c)\in D}\frac{1}{1+\exp(-u_c^Tv_w)} \prod _{(w,c)\in \tilde{D}}(1-\frac{1}{1+\exp(-u_c^Tv_w)})\\
& = \prod_{(w,c)\in D}\frac{1}{1+\exp(-u_c^Tv_w)} \prod _{(w,c)\in \tilde{D}}\frac{1}{1+\exp(u_c^Tv_w)}\\
\end{align}
$$
其中 $\tilde{D}$是“假”的语料，

我们希望最小化如下目标函数
$$
\begin{align}
J  &= -\log L(\theta)\\
&= -\sum_{(w,c)\in D}\log \frac{1}{1+\exp(-u_c^Tv_w)} -\sum _{(w,c)\in \tilde{D}}\log\frac{1}{1+\exp(u_c^Tv_w)}
\end{align}
$$

- **Skip-gram** 对应给定的中心词向量$v_c$来观察上下文单词$w_{c-m+j}$的目标函数：
  $$
  -\log \sigma(u_{c-m+j}^Tv_c) -\sum _{k=1}^K\log\sigma(\tilde{u}_k^Tv_c)
  $$
  

- **CBOW** 对应给定的上下文向量$\hat{v}=\frac{v_{c-m}+v_{c-1}+...+v_{c+1}+v_{c+m}}{2m}$来观察中心词向量 $u_c$的目标函数：
  $$
  -\log \sigma(u_c^T\hat{v}) -\sum _{k=1}^K\log\sigma(\tilde{u}_k^T\hat{v})
  $$
  其中，$\{\tilde{u}_k|k=1,...,K\}$是从$P_n(w)$中抽取的K个负样本。

- 实际中发现效果最好的是3/4的Unigram模型：
  $$
  P(w) = U(w)^{3/4}/Z
  $$
  其中Z是正则化系数（没有特别说明的情况下，后续会始终用Z表示正则化系数）

- 这个指数$\alpha = 3/4$使得低频词更容易被抽到
  -  is: $0.9^{3/4} = 0.92$
  - constitution: $0.09^{3/4} = 0.16$
  - bombastic: $0.01^{3/4} = 0.032$

### 2.5.3 负采样介绍

**Question** 对于给定的词 w，如果生成w对应的负样本呢？

**Answer** 带权采样，且权重与单词出现频率相关

- 对于高频词，希望被选为负样本的概率比较大
- 反之，对于低频词，希望选中的概率就比较小。

设词典中的每一个词w都对应一个线段 len(w)，其长度为：
$$
len(w) = \frac{\text{count}(w)}{\sum_{u\in C}\text{count}(u)} = \text{w 在语料C中出现的频率}
$$
现在将这些线段首尾相连地拼接在一起，形成一个长度为1的单位线段。如果随机地往这个单位线段上打点，则其中长度越长的线段（对应高频词）被打中的概率就越大。

由此得到区间[0,1]上的一个非等距剖分，剖分点分别是$\{l_i\}_{i=1}^{N}$, 其中 $l_0 = 0, l_k = \sum_{j=1}^k len(w_j)$,$k=1,...,N$，$w_j$是词典里的第j个单词。记第i个剖分为 $I_i = \{l_{i-1}, l_i\}$。

另在区间[0,1]上取一个M等分的等距里剖分，剖分点记为$\{m_j\}_{j=0}^M$，其中$M\gg N$。

<img src="./word2vec/neg_sampling5.png" alt="neg_sampling5" style="zoom:40%;" />

定义如下映射：将内部剖分点$\{m_j\}_{j=1}^{M-1}$投射到$\{I_i\}$（上图红色虚线）
$$
\text{Table}(r) = w_k, \text{  if  } m_r \in I_k \text{ for } r = 1, 2,..., M-1
$$
每次生成一个[0, M-1]间的随机整数r，就是一个样负本 Table(r)

- 当对w进行负采样时，如果碰巧选到w自己该怎么办？跳过。

- 注意到这个过程其实就是Unigram U(w)
- Word2Vec源码中
  - 取指数$\alpha = 3/4$，即 $U(w)^{\alpha}$
  - 取 $M = 10^8$



### 2.5.4 其他

**窗口大小和负样本数量**

ref https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651669277&idx=2&sn=bc8f0590f9e340c1f1359982726c5a30&chksm=bd4c648e8a3bed9817f30c5a512e79fe0cc6fbc58544f97c857c30b120e76508fef37cae49bc&scene=0&xtrack=1#rd



## 其他问题



10）介绍下Hierarchical Softmax的计算过程，怎么把 Huffman 放到网络中的？参数是如何更新的？对词频低的和词频高的单词有什么影响？为什么？

Hierarchical Softmax利用了Huffman树依据词频建树，词频大的节点离根节点较近，词频低的节点离根节点较远，距离远参数数量就多，在训练的过程中，低频词的路径上的参数能够得到更多的训练，所以效果会更好。

（11）Word2Vec有哪些参数，有没有什么调参的建议？

- Skip-Gram 的速度比CBOW慢一点，小数据集中对低频次的效果更好；
- Sub-Sampling Frequent Words可以同时提高算法的速度和精度，Sample 建议取值为 ；
- Hierarchical Softmax对低词频的更友好；
- Negative Sampling对高词频更友好；
- 向量维度一般越高越好，但也不绝对；
- Window Size，Skip-Gram一般10左右，CBOW一般为5左右。

（12）Word2Vec有哪些局限性？

Word2Vec作为一个简单易用的算法，其也包含了很多局限性：

- Word2Vec只考虑到上下文信息，而忽略的全局信息；
- Word2Vec只考虑了上下文的共现性，而忽略的了彼此之间的顺序性；

Ref

1. https://mp.weixin.qq.com/s/zDneR1BU6xvt8cndEF4_Xw 
2. https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651669277&idx=2&sn=bc8f0590f9e340c1f1359982726c5a30&chksm=bd4c648e8a3bed9817f30c5a512e79fe0cc6fbc58544f97c857c30b120e76508fef37cae49bc&scene=0&xtrack=1#rd
