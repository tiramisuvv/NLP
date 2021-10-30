# 1. Torchtext

 **Torchtext** 是一个非官方的、为Pytorch提供文本数据处理的库

## 1.1 主要组件

**torchtext主要包含的组件有：Field、Dataset和Iterator。**

### 1.1.1 Field: 数据预处理的配置信息

- 主要包含以下数据预处理的配置信息，比如指定分词方法，是否转成小写，起始字符，结束字符，补全字符以及词典等。

- 举例

  ```python
  # 定义两种Filed，分别用来处理文本和标签
  TEXT = Field(sequential=True, tokenize=tokenize, lower=True, fix_lengt=200)
  LABEL = Field(sequential=False, use_vocab=False)
  ```

### 1.1.2 Dataset：用于加载数据

- 继承自pytorch的Dataset，用于加载数据。

- 通过给TabularDataset 提供路径，格式，Field信息就可以方便的完成数据加载。

- 同时torchtext还提供预先构建的常用数据集的Dataset对象，可以直接加载使用，

- splits方法可以同时加载训练集，验证集和测试集。

- 举例

  ```python
  # 列名和对应的Field对象
  fields = [('id', None), ('label', label_field), ('text', text_field)]
  
  # TabularDataset：从 csv、tsv、json文件中读取数据并生成dataset
  train, valid, test = TabularDataset.splits(
      path=source_folder, 
      train='train.csv', 
      validation='valid.csv',
      test='test.csv', 
      format='CSV',
      fields=fields, 
      skip_header=True)
  
  ```

### 1.1.3 Iterator : 主要是数据输出的迭代器

- 输出batch用来分批次训练模型。

  ```python
  train_iter, val_iter = BucketIterator.split(
    (train,valid),
    batch_size=(8, 8),
    sort_key=lambda x: len(x.text),
    device=device, 
    sort_within_batch=False,
    repeat=False
  )
  test_iter =  Iterator(test, batch_size=8, device=device, train=False, shuffle=False, sort=False, sort_within_batch=False, repeat = False)
  ```

  

