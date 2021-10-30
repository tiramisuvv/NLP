# Contextual Word Representation

## 1. Motivation

>  “You shall know a word by the company it keeps” 
>
> ​                                                                                       --  J. R. Firth 1957: 11

=> motivated **word2vec**

(a summary of distributional semantics)

> “… the complete meaning of a word is always contextual, and no study of meaning apart from a complete context can be taken seriously.”   
>
> ​                                                                                        --  J. R. Firth 1935:

MEANING of word = the sense of the word inside a particular context of use (特定上下文中的单词含义  )



🌰 Consider I **record** the **record**: the two instances of **record** mean different things.

### 1.1 Before : Pretrained word embeddings

### 1.1.1 Structure

Each word has one representation.

1. Start with pretrained word embeddings (no context!) 
2. Learn how to incorporate context in an LSTM or Transformer while training on the task.

<img src="/Users/weiwang/Documents/NLP/contextual_representation/Circa2017.png" alt="Circa2017" style="zoom:50%;" />

### 1.1.2 Problems/Limitations

- **One embedding for each word** 

  - Always the same representation for a **word type** regardless of the context in which a **word token** occurs
    - We might want very fine-grained word sense disambiguation
  - We just have one representation for a word, but words have different **aspects**, including semantics, syntactic behavior, and register/connotations
    - e.g arrive vs arrival   

- Modeling

  - Most of the parameters in our network are **randomly initialized**!

- WANT = "context-specific representation of words"

- 🌰 在下面的NLM中， LSTM中的每个 h(i) 都是 word i的representation （学习了contexual 的meaning）

  <img src="/Users/weiwang/Documents/NLP/contextual_representation/contexual_eg.png" alt="contexual_eg" style="zoom:50%;" />

  - 

  

### 1.2 Now : Pretrained whold models

<img src="/Users/weiwang/Documents/NLP/contextual_representation/modernNLP.png" alt="modernNLP" style="zoom:50%;" />



- All (or almost all) parameters in NLP networks are initialized via **pretraining**. 
- Pretraining methods hide parts of the input from the model, and train the model to reconstruct those parts.



【核心想法】We jus took those context-specific representation of words, they'd be useful for doing othering things with them 

## 1.3 Pretrained vs random

![pre-trained_vs_randome](file:///Users/weiwang/Documents/NLP/contextual_representation/pre-trained_vs_randome.png?lastModify=1630730766)

## 2. Structure = Pretraining + Fine Tuning

<img src="/Users/weiwang/Documents/NLP/contextual_representation/structure.png" alt="structure" style="zoom:50%;" />

- **Step 1 ** Pretraining through language modeling:
  - Train a neural network to perform language modeling on a large amount of text. 
  - Save the network parameters.
  - Pretraining can improve NLP applications by serving as parameter initialization.
- **Step 2** Fine tune on your tasks

## 2.1 Three Type of Pretraining

<img src="/Users/weiwang/Documents/NLP/contextual_representation/type_of_pretrain.png" alt="type_of_pretrain" style="zoom:50%;" />







## 3. "Pre-ELMo" -- TagLM

- Idea: Want meaning of word in context, but standardly learn task RNN only on small task-labeled data (e.g., NER)
- Why don’t we do semi-supervised approach where we train NLM on large unlabeled corpus, rather than just word vectors?
- TagLM使用与上文无关的单词嵌入 + semi-supervised approach 的 RNN model 得到的 hidden states 作为特征输入

![TagLM](/Users/weiwang/Documents/NLP/contextual_representation/TagLM.png)



<img src="/Users/weiwang/Documents/NLP/contextual_representation/TagLM2.png" alt="TagLM2" style="zoom:50%;" />

- 把 bi-RNN 中训练得到的“representation” 再feed 进 supervised models

【some details】

Language model is trained on 800 million training words of “Billion word benchmark”

Language model observations

- An LM trained on supervised data does not help

- Having a bidirectional LM helps over only forward, by about 0.2

- Having a huge LM design (ppl 30) helps over a smaller model (ppl 48) by about 0.3

  Task-specific BiLSTM observations

• Using just the LM embeddings to predict isn’t great: 88.17 F1

• Well below just using an BiLSTM tagger on labeled data

- 上面的pre-train system 是fixed 的



## 4. ELMo

ELMo = Embeddings from Language Models

 <img src="/Users/weiwang/Documents/NLP/contextual_representation/elmo.png" alt="elmo" style="zoom:20%;" />



- Breakout version of **word token vectors** or **contextual word vectors**
- Learn word token vectors using long contexts not context windows (here, whole sentence, could be longer)
- Learn a deep Bi-NLM and use all its layers in prediction

- Train a bidirectional LM

- Aim at performant but not overly large LM:

  - Use 2 biLSTM layers

  - Use character CNN to build initial word representation (only)

    • 2048charn-gramfiltersand2highwaylayers,512dimprojection

  - User 4096 dim hidden/cell LSTM states with 512 dim

    projections to next input

  - Use a residual connection

  - Tie parameters of token input and output (softmax) and tie these between forward and backward LMs

  - P19-22

  - TagLM vs ELMo

    - TagLM 中只用到隐藏层的最后一层当作‘word represenatation’
    - ELMo 用到所有的（$ELMo_{k}^{task}$ 的公式）

  - 【Structure】

    ![ELMo_tag](/Users/weiwang/Documents/NLP/contextual_representation/ELMo_tag.png)

  - * **Step 1** run biLM to get representations for each word

      <img src="/Users/weiwang/Documents/NLP/contextual_representation/ELMo2.png" alt="ELMo2" style="zoom:50%;" />

      

    * **Step 2** let (whatever) end-task model use them

      * Freeze weights of ELMo for purposes of supervised model
      * Concatenate ELMo weights into task-specific model 
        * Details depend on task:
          * Concatenating into intermediate layer as forTagLM is typical
          * Can provide ELMo representations again when producing outputs, as in a question answering system

- **Highlight** ELMo representations can be used for almost **any** NLP taks

- ![ELMo_res](/Users/weiwang/Documents/NLP/contextual_representation/ELMo_res.png)

- The two biLSTM NLM layers have differentiated uses/meanings
  - Lower layer is better for lower-level syntax, etc
    - Part-of-speech tagging, syntactic dependencies, NER 
  - Higher layer is better for higher-level semantics
    - Sentiment, Semantic role labeling, question answering, SNLI

## 5. ULMfit and onward

ULMfit = Universal Language Model Fine-tuning

**IDEA** Transferring NLM knowledge



<img src="/Users/weiwang/Documents/NLP/contextual_representation/ULMfit.png" alt="ULMfit" style="zoom:50%;" />

- 在大型通用领域的无监督语料库上使用 biLM 训练
- 在目标任务数据上调整 LM
- 对特定任务将分类器进行微调

### Architecture

<img src="/Users/weiwang/Documents/NLP/contextual_representation/ULMfit2.png" alt="ULMfit2" style="zoom:50%;" />

【问题】固定了（b) 中的某些事情，在（c) 中重用，黑色表示freeze

- Use reasonable-size “1 GPU” language model not really huge one
- **A lot of care in LM fine-tuning**
  - Different per-layer learning rates
  - Slanted triangular learning rate (STLR) schedule 
- Gradual layer unfreezing and STLR when learning classifier
- Classify using concatenation [$h_T$,  maxpool($h$)  , meanpool($h$) ]

### Performance

**Text classifier error rates**

<img src="/Users/weiwang/Documents/NLP/contextual_representation/ULMfit_performance.png" alt="ULMfit_performance" style="zoom:50%;" />

**transfer learning**

![ULMfit_performance2](/Users/weiwang/Documents/NLP/contextual_representation/ULMfit_performance2.png)

 

## 6. Transformer architecture

<img src="/Users/weiwang/Documents/NLP/contextual_representation/transformer2.png" alt="transformer2" style="zoom:50%;" />

#### Empirical advantages of Transformer vs. LSTM:

- Self-attention == no locality bias 
  - Long-distance context has “equal opportunity” 
- Single multiplication per layer == efficiency on TPU 
  - Effective batch size is number of words, not sequences



+ <img src="/Users/weiwang/Documents/NLP/contextual_representation/transformer.png" alt="transformer" style="zoom:50%;" />
+ 后面都是Transformer base的，
+ Transformer的优点
  + (+) powerfull 
  + technically, allow the model scaling to much bigger sizes



## 7. BERT

BERT = Bidirectional Encoder Representations from Transformers

<img src="/Users/weiwang/Documents/NLP/contextual_representation/BERT.png" alt="BERT" style="zoom:50%;" />

 

### 7.1 Masked LM

**Problem**: Language models only use left context *or* right context, but language understanding is bidirectional.

- Why are LMs unidirectional?
- Reason 1: Directionality is needed to generate a well-formed probability distribution.
   • We don’t care about this.
- Reason 2: Words can “see themselves” in a bidirectional encoder.

<img src="/Users/weiwang/Documents/NLP/contextual_representation/bert_1.png" alt="bert_1" style="zoom:50%;" />

右边，第二层在读 `<s>` 时，由于输入是第一层的Bi-Directional 的结果，实际已经见过 `open` 

【GOAL】 Want: truly bidirectional information flow without leakage in a deep model 

**Solution**: Mask out *k* % of the input words, and then predict the masked words

- Objective  : Mad Libs style fill in the blank

  <img src="/Users/weiwang/Documents/NLP/contextual_representation/BERT_mask.png" alt="BERT_mask" style="zoom:50%;" />

- [-] cannot get as many predictions per sentence, only get 15% of words instead of 100% of words

- [+] be able to see both direction  

- They always use *k* = 15% 

  - Too little masking: Too expensive to train
  - Too much masking: Not enough context

**Problem**: Mask token never seen at fine-tuning 

**Solution**: 15% of the words to predict, but don’t replace with [MASK] 100% of the time. Instead: 

- 80% of the time, replace with [MASK]
  - 🌰  went to the store → went to the [MASK] 
- 10% of the time, replace random
  - 🌰  word went to the store → went to the running 
- 10% of the time, keep same
  - 🌰 went to the store → went to the store

### 7.2 Next Sentence Prediction

#### 7.2.1 Target： 

To learn *relationships* between sentences, predict whether Sentence B is actual sentence that proceeds Sentence A, or a random sentence

<img src="/Users/weiwang/Documents/NLP/contextual_representation/BERT_NSP.png" alt="BERT_NSP" style="zoom:50%;" />

#### 7.2.2 Input Representation

<img src="/Users/weiwang/Documents/NLP/contextual_representation/BERT_sentence_encoder.png" alt="BERT_sentence_encoder" style="zoom:50%;" />

### 7.3 VS ELMo & GPT 

<img src="/Users/weiwang/Documents/NLP/contextual_representation/BERT_architecture.png" alt="BERT_architecture" style="zoom:50%;" />

- ELMO :
  - LSTM based,  
  - 同时有left to right model 和 right to left model，**但是两个LM是完全独立训练的**
    - **不能**同时了解上下文
- GPT: 
  - transformer based, 
  - only see left context
- BERT:
  - transformer based
  - can see both left and right context

### 7.4 Architecture and Model Details

- Transformer encoder (as before)

- Self-attention ⇒ no locality bias

  -  Long-distance context has “equal opportunity”

- Single multiplication per layer ⇒ efficiency on GPU/TPU

  

- <u>Data</u> : Wikipedia (2.5B words) + BookCorpus (800M words)

- <u>Batch Size</u>: 131,072 words (1024 sequences * 128 length or 256 sequences * 512 length)

- <u>Training Time</u>: 1M steps (~40 epochs) ● Optimizer: AdamW, 1e-4 learning rate, linear decay

- Train 2 model sizes:

  - BERT-Base: 12-layer, 768-hidden, 12-head

  - BERT-Large: 24-layer, 1024-hidden, 16-head

- Trained on 4x4 or 8x8 TPU slice for 4 days



### 7.5 Fine-Tuning Procedure 

 Simply learn a classifier built on the top layer for each task that you fine tune for

<img src="/Users/weiwang/Documents/NLP/contextual_representation/BERT_fine_tune.png" alt="BERT_fine_tune" style="zoom:50%;" />

- 怎么Fine-Tuning？
  - 去掉Pre-training 中top-level prediction，并且用task 对应的prediction layer替代

<img src="/Users/weiwang/Documents/NLP/contextual_representation/BERT_fine_tune2.png" alt="BERT_fine_tune2" style="zoom:50%;" />

### 7.6 Performance

#### 7.6.1 On different tasks

<img src="/Users/weiwang/Documents/NLP/contextual_representation/BERT_performance.png" alt="BERT_performance" style="zoom:50%;" />

- 即时只有少量 data（比如RTE，2.5k）仍然有很好的效果



#### 7.6.2 Effect of Pre-training Task

【Questions】 

<img src="/Users/weiwang/Documents/NLP/contextual_representation/BERT_performance2.png" alt="BERT_performance2" style="zoom:50%;" />

#### 7.6.3 Effect of Directionality and Training Time

<img src="/Users/weiwang/Documents/NLP/contextual_representation/BERT_performance3.png" alt="BERT_performance3" style="zoom:50%;" />

#### 7.6.2 Effect of Model Size



<img src="/Users/weiwang/Documents/NLP/contextual_representation/BERT_performance4.png" alt="BERT_performance4" style="zoom:50%;" />



## 8.RoBERTa

RoBERTa = A Robustly Optimized BERT Pretraining Approach (Liu et al, University of Washington and Facebook, 2019)

"BERT was under trained" 

- Trained BERT for more epochs and/or on more data 

  - Showed that more epochs alone helps, even on same data 
  - More data also helps 

- Improved masking and pre-training data slightly

  <img src="/Users/weiwang/Documents/NLP/contextual_representation/RoBERTa.png" alt="RoBERTa" style="zoom:50%;" />

## 9. XLNet

XLNet: Generalized Autoregressive Pretraining for Language Understanding (Yang et al, CMU and Google, 2019)

### 9.1 Innovation #1: Relative position embeddings

[Questions] how to do?

- Sentence:  `John ate a hot dog`，
- Absolute attention: 
  - “How much should `dog` attend to `hot`(in any position), 
  - and how much should `dog` in position 4 attend to the word in position 3? (Or 508 attend to 507, …)” 
  - **Quadratic number of relationships** 
- Relative attention: 
  - “How much should `dog` attend to `hot` (in any position) and how much should `dog` attend to the previous word?”
  - **Good for long sentence**  

### 9.2 Innovation #2: Permutation Language Modeling

[Questions] how to do?

- In a left-to-right language model, every word is predicted based on all of the words to its left
- Instead: Randomly permute the order for every training sentence
- **Equivalent to masking, but many more predictions per sentence**
- Can be done efficiently with Transformers 

<img src="/Users/weiwang/Documents/NLP/contextual_representation/XLNet_permutation.png" alt="XLNet_permutation" style="zoom:50%;" />

- Permutation 目标：学习 "`学3`"

<img src="/Users/weiwang/Documents/NLP/contextual_representation/XLNet_Permutation1.png" alt="XLNet_Permutation1" style="zoom:30%;" /><img src="/Users/weiwang/Documents/NLP/contextual_representation/XLNet_permutation2.png" alt="XLNet_permutation2" style="zoom:30%;" />



Also used more data and bigger models, but showed that innovations improved on BERT even with same data and model size



### 9.3 Results

![XLNet_res](/Users/weiwang/Documents/NLP/contextual_representation/XLNet_res.png)

## 10. ALBERT

ALBERT =  A Lite BERT for Self-supervised Learning of Language Representations (Lan et al, Google and TTI Chicago, 2019) 

### 10.1 Innovation #1: Factorized embedding parameterization 

- Use small embedding size (e.g., 128) and then project it to Transformer hidden size (e.g., 1024) with parameter matrix
- <img src="/Users/weiwang/Documents/NLP/contextual_representation/ALBERT1.png" alt="ALBERT1" style="zoom:50%;" />

### 10.2 Innovation #2: Cross-layer parameter sharing 

- Share all parameters between Transformer layers

SOP(Sentence order prediction) used ub ALBERT



### 10.3 Results

<img src="/Users/weiwang/Documents/NLP/contextual_representation/ALBERT2.png" alt="ALBERT2" style="zoom:50%;" />



## 11. T5 - Comparison

T5 = Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al, Google, 2019)

- Ablated many aspects of pre-training: 
  - Model size
  - Amount of training data 
  - Domain/cleanness of training data 
  - Pre-training objective details (e.g., span length of masked text) 
  - Ensembling 
  - Finetuning recipe (e.g., only allowing certain layers to finetune)
  - Multi-task training
-  Conclusions:
  - Scaling up model size and amount of training data helps a lot
  - Best model is 11B parameters (BERT-Large is 330M), trained on 120B words of cleaned common crawl text
  - Exact masking/corruptions strategy doesn’t matter that much
  - Mostly negative results for better finetuning and multi-task strategies
- Results
  - <img src="/Users/weiwang/Documents/NLP/contextual_representation/T5.png" alt="T5" style="zoom:50%;" />

## 12. ELECTRA

ELECTRA = Efficiently Learning an Encoder to Classify Token Replacements Accurately

<img src="/Users/weiwang/Documents/NLP/contextual_representation/ELECTRA_architecture.png" alt="ELECTRA_architecture" style="zoom:50%;" />

- Train model to discriminate locally plausible text from real text 

<img src="/Users/weiwang/Documents/NLP/contextual_representation/ELECTRA_architecture2.png" alt="ELECTRA_architecture2" style="zoom:50%;" />

VS BERT

- BERT `mask` 是用完全random的单词替换；
- ELECTRA 是用 weak BERT 预测结果替换
- => 模型更好的学习词汇直接的细微差别 

**优点**：

- Predicting yes/not is easier than reconstruction.
- Every output position is used

### Performance 

FLOP

<img src="/Users/weiwang/Documents/NLP/contextual_representation/ELECTRA_performance.png" alt="ELECTRA_performance" style="zoom:50%;" />

<img src="/Users/weiwang/Documents/NLP/contextual_representation/ELECTRA_res.png" alt="ELECTRA_res" style="zoom:50%;" />



## Distillation

## DistilBert



先Bert+Fine tune，然后 distillate 可能优于 DistilBert + Finetune







## 2, Unknown words 

### Simplest and common solution:

- Train time: Vocab is {words occurring, say, ≥ 5 times} ∪ {<UNK>}

- Map **all** rarer (< 5) words to <UNK>, train a word vector for it

- Runtime: use <UNK> when out-of-vocabulary (OOV) words occur

- Problems:
   • No way to distinguish different UNK words, either for identity

  or meaning

  

### Solutions:

1. char-level models to build vectors! 

2. Especially in applications like question answering: Where it is important to match on word identity, even for words outside your word vector vocabulary

   -  Try these tips (from Dhingra, Liu, Salakhutdinov, Cohen 2017)

     - If the <UNK> word at test time appears in your unsupervised word embeddings, use that vector as is at test time.【definitely helps a lot】

     - Additionally, for other words, just assign them a random vector, adding them to your vocabulary【may help a little more】

   -  Collapsing things to word classes (like unknown number, capitalized thing, etc. and having an <UNK-class> for each

