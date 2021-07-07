# Dependency Parsing

<img src="/Users/Wei/Library/Application Support/typora-user-images/Screen Shot 2021-07-06 at 10.39.24 AM.png" alt="Screen Shot 2021-07-06 at 10.39.24 AM" style="zoom:50%;" />

# Structure of Sentences

1. Phrase structure 词组结果

   ![Screen Shot 2021-07-06 at 10.43.14 AM](/Users/Wei/Library/Application Support/typora-user-images/Screen Shot 2021-07-06 at 10.43.14 AM.png)

- 抓住“结构”，与上下文/单词含义无关
  - Rule1: "Noun Phrase (NP)"
    - 比如 the cat，a dog -> NP =  determiner + noun
    - 再比如 the large cat, a barking dog,  -> NP = determiner + (adj) +noun
    - 进一步 the large cat in a crate, -> NP = determiner + (adj) +noun + prep
  - Rule 2: Preposition Phrase (PP) = Prep + NP
    - => The cat by the large crate on the large table by the door
  - Rule 3: verb phrase = Verb + PP

类似可以一层一层的，建一个长句子.

Phrase Structure Grammar tree

2. Denpendency structur

   <img src="/Users/weiwang/Documents/NLP/dependency_parsing/dependency_structure.png" alt="dependency_structure" style="zoom:50%;" />

   

   <img src="/Users/weiwang/Documents/NLP/dependency_parsing/why_need_sentence_structure.png" style="zoom:50%;" />

例子🌰 

San Jose cops kill man with knife.

- knife -> cops
- knife -> man 

Some ambiguity examples

1. 🌰 PP attachment ambiguities multiply 

<img src="/Users/weiwang/Documents/NLP/dependency_parsing/pp_amb.png" alt="pp_amb" style="zoom:50%;" />

2. 🌰 coordination scope ambiguity

   - [(Shuttle veteran and longtime NASA executive) Fred Gregory] appointed to board.

   - (Shuttle veteran) and (longtime NASA executive Fred Gregory) appointed to board.

3. 🌰 Adjectival Modifier Ambiguity

   - Students get first hand job experience 

4. 🌰 Verb Phrase(VP) attachment ambiguity



### Dependency Grammar and Dependency Structure

<img src="/Users/weiwang/Documents/NLP/dependency_parsing/dependency_tree.png" style="zoom:50%;" />

- 课程中只用arrow，不用type（nsubj,etc)
- the arrow connects a ***head*** (governor,superior, regent) with a ***dependent*** (modifier, inferior, subordinate)
- dependencies form a ***tree***(connected, acyclic, single-head)

## Treebanks

<img src="/Users/weiwang/Documents/NLP/dependency_parsing/treebank.png" alt="treebank" style="zoom:50%;" />

- 🎯goal of "universal dependency": have a uniform parallel system of dependency description which could be used for any human language 

<img src="/Users/weiwang/Documents/NLP/dependency_parsing/treebank2.png" alt="treebank2" style="zoom:50%;" />

* 与 “Grammar”相比的好处：telling what is the right structure for ambiguous sentences



A model to capture what is the right parse

<img src="/Users/weiwang/Documents/NLP/dependency_parsing/dependency_preference.png" style="zoom:50%;" />

<img src="/Users/weiwang/Documents/NLP/dependency_parsing/dependence_parsing.png" style="zoom:50%;" />

### Method : Transition-based parsing or deterministic dependency parsing

1. parsing的具体过程举例

<img src="/Users/weiwang/Documents/NLP/dependency_parsing/Transition-based parsing1.png" alt="Transition-based parsing1" style="zoom:50%;" /><img src="/Users/weiwang/Documents/NLP/dependency_parsing/Transition-based parsing2.png" alt="Transition-based parsing2" style="zoom:50%;" />

 

2. **Question** 如何决定每一步正确的action (shift, left arc, right arc, etc)
   1. 遍历所有可能 （指数级）
   2. dynamic programming（过去使用的方法）
   3. MaltParser 用ML classifier预测下一步的action
      - Each action is predicted by a discrimnatvie classifier (e.g. softmax classifier) over each legal move
        - Max of 3 uptyped choices (shift, left arc, right arc);
        - Max of |R| X 2 + 1 when typed
          - put labeds on the dependencies 
          - |R| different labels
        - Features: top of stack word, POS; first in buffer word, POS; etc
      - There is NO search (in the simplest form)

<img src="/Users/weiwang/Documents/NLP/dependency_parsing/malparser.png" alt="malparser" style="zoom:50%;" />

<img src="/Users/weiwang/Documents/NLP/dependency_parsing/conventioalfeaturerepresentation.png" alt="conventioalfeaturerepresentation" style="zoom:50%;" />

- logistic regression, SVM 等算法已经可以做的不错
- 下面会介绍Neural dependency parsing

## Evaluation

 <img src="/Users/weiwang/Documents/NLP/dependency_parsing/evaluation1.png" alt="evaluation1" style="zoom:50%;" />

- UAS = 忽略label (nsubj, root, etc)，只看arc的正确率 

- LAS = 包括label+arc 的正确率



## Why train a neural dependency parser？

<img src="/Users/weiwang/Documents/NLP/dependency_parsing/why_neural.png" alt="why_neural" style="zoom:50%;" />

<img src="/Users/weiwang/Documents/NLP/dependency_parsing/neural_parser.png" alt="neural_parser" style="zoom:50%;" />

<img src="/Users/weiwang/Documents/NLP/dependency_parsing/model_architecture.png" style="zoom:50%;" />

