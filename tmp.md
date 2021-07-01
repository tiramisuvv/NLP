input x: 一个句子，长度记为 $T_x$

- $x^{<t>}$ word at position t

$x^{(i)<t>}$ 第i个input的t位置，$T_{x}^{(i)}$ 对不同sample可能不同，如有的句子9个单词，有的15个单词



output y:

Vacabulary :30K-50K



# RNN

- 直接用NN

  - input,outputs 长度不同；
  - doesn't share features learned across different positions of text

  - a better model can reduce # of parameters 

- 结构 architectures 

  - $T_x = T_y$  

 <img src="/Users/Wei/Documents/NLP/NLP/rnn/structure.png" alt="structure" style="zoom:50%;" />

- 利用前面的信息

- 共享参数

- 缺点 只用前面信息，不用后面信息

  - He said 
  - He said
  - -》 Bidirectional RNN（BRNN）

  $W_{ax}$ 乘以x，用于计算a，比如$a^{<1>} = g(W_{aa}a^{<0>}+W_{ax}x^{<1>}+b_a)$

  [TODO] $T_x \neq T_y$ 是怎么做的-> different types

  ## Different types

  1. many-to-many
     1. $T_x = T_y$
     2. $T_x \neq T_y$
        1. machine translation 
        2. 
  2. many-to-one
     1. 🌰 setimant classification
  3. one-to-many
     1. music generation 
     2. 

  <img src="/Users/Wei/Documents/NLP/NLP/rnn/summary_types.png" alt="summary_types" style="zoom:50%;" />

  ## Backpropagation through time

