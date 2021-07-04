# Neural Networ & Backpropagation



<img src="/Users/Wei/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 5.37.57 PM.png" alt="Screen Shot 2021-07-03 at 5.37.57 PM" style="zoom:50%;" />

# Classification in traditional ML and NLP

<img src="/Users/Wei/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 5.42.28 PM.png" alt="Screen Shot 2021-07-03 at 5.42.28 PM" style="zoom:50%;" />

word window classification - named entity recognition 

Nll - negtive log likelihood

cross-entropy loss 更好用 【TODO】why

**Def**

True probability distribution p

Predicte probability distribution q

H(p,q )【TODO】 meaning?

【TODO】为什么logistic regression画出来 是linear 的？

word vector Representation learning 

$\theta$= W :weight + $v_w$:word vector

【Q】 在NN里训练word vector？



## Named Entity Recognition(NER)

Task：find the classify names in text:

<img src="/Users/Wei/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 5.44.49 PM.png" alt="Screen Shot 2021-07-03 at 5.44.49 PM" style="zoom:50%;" />

Why difficult 

Solution: Idea Window classification

Binary classification









## Gradient by hand - Matrix calculate 





1. n input &1 output $f(x) = f(x1, ..., x_n)$, $\frac{\part f}{\part x} = \left[\frac{\part f}{\part x_1},...,\frac{\part f}{\part x_n}\right]$
2. n input & m output $f(x) = [f_1(x1, ..., x_n),...,f_m(x1, ..., x_n)]$, ...
3. $\delta$ error signal = 偏导 above parameter = error signal coming above
4. shape convition 
5. 根据$\delta_{window}$ 每次更新word vector

**Questions** 是否用“pre-trained” word vectors？

**Answer** Almost always, yes!

![Screen Shot 2021-07-03 at 7.26.15 AM](/Users/Wei/Library/Application Support/typora-user-images/Screen Shot 2021-07-03 at 7.26.15 AM.png)

- small data set <= 几百k
- large dataset > 1M



### Computation Graph

<img src="/Users/Wei/Documents/NLP/NLP/NN/computation_graph.png" alt="computation_graph" style="zoom:50%;" />

- Node with one input & one output
- [Downstram gradient] = [upstream gradient] x [local graident]

<img src="/Users/Wei/Documents/NLP/NLP/NN/single_node_1in1out.png" alt="single_node" style="zoom:50%;" />

- Node with multiple input & one output 

  <img src="/Users/Wei/Documents/NLP/NLP/NN/single_node_mulin1out.png" alt="single_node_mulin1out" style="zoom:50%;" />

- 🌰

<img src="/Users/Wei/Documents/NLP/NLP/NN/backprog_example.png" alt="backprog_example" style="zoom:50%;" />

- Back prop = if you wiggle x a little bit, how big an effect does that have on the outcome of the whole thing

- 比如 x从$1\rightarrow 1.1$, 那么期待output 会变化 $\frac{\part f}{\part x}\cdot \Delta x = 2 \times 0.1 = +0.2$，实际output = (1.1+2) max(2, 0) = 6.2，变化为 $+0.2$

- 比如 y 从 $2\rightarrow 2.1$，

  - 期待output变化：$5\times 0.1 = 0.5$
  - 实际 $\Delta \text{output} = (1 + 2.1)\max(2.1, 0) - 6= 6.51 - 6 = 0.51$
  - 注意这里是预估变化，不是准确值

- 各个计算的back prop的含义

  - **plus +**  “distributes” the upstream gradient to each summand
  - **max** “routes” the upstream gradient
  - **Product *** “switches” the upstream gradient

- multiple output

  <img src="/Users/Wei/Documents/NLP/NLP/NN/multi_output.png" alt="multi_output" style="zoom:50%;" />

- Efficiency: compute all gradients at once

<img src="/Users/Wei/Documents/NLP/NLP/NN/Efficiency.png" alt="Efficiency" style="zoom:50%;" />

- Backprop
  1. 按照Topological sort排序
     1. 依赖在上

<img src="/Users/Wei/Documents/NLP/NLP/NN/backprop.png" alt="backprop" style="zoom:50%;" />

<img src="/Users/Wei/Documents/NLP/NLP/NN/AutomaticDifferentiation.png" alt="AutomaticDifferentiation" style="zoom:50%;" />

