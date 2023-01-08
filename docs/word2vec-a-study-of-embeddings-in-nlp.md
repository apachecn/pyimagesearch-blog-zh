# Word2Vec:自然语言处理中的嵌入研究

> 原文：<https://pyimagesearch.com/2022/07/11/word2vec-a-study-of-embeddings-in-nlp/>

* * *

## **目录**

* * *

## [**word 2 vec:NLP 中的嵌入研究**](#TOC)

[上周](https://pyimg.co/oa2kt)，我们看到了如何根据完整的语料库以受限的方式来表示文本，从而帮助计算机给单词赋予意义。我们的方法(单词袋)基于单词的频率，当输入文本变大时，需要复杂的计算。但是它所表达的思想值得探索。

随着今天的聚焦:Word2Vec，我们慢慢步入现代自然语言处理(NLP)的领域。

在本教程中，您将学习如何为文本实现 Word2Vec 方法。

作为一个小序言，Word2Vec 是我在自己的旅程中遇到的 NLP 中最简单但也是我最喜欢的主题之一。这个想法是如此简单，然而它在许多方面重新定义了 NLP 世界。让我们一起来探索和了解 Word2Vec 背后的本质。

本课是关于 NLP 101 的 4 部分系列的第 3 部分:

1.  [*自然语言处理入门*](https://pyimg.co/60xld)
2.  *[*介绍词袋(BoW)模型*](https://pyimg.co/oa2kt)*
3.  ****[word 2 vec:NLP 中的嵌入研究](https://pyimg.co/2fb0z)* (今日教程)***
4.  *【BagofWords 和 Word2Vec 的比较*

 ***要学习如何实现 Word2Vec，** ***只要坚持阅读。***

* * *

## [**word 2 vec:NLP 中的嵌入研究**](#TOC)

* * *

### [**word 2 vec**简介](#TOC)

让我们解决第一件事；Word2vec 这个名字是什么意思？

它正是你所想的(即，作为向量的词)。Word2Vec 本质上就是把你的文本语料库中的每一个单词都表达在一个 N 维空间(嵌入空间)中。单词在嵌入空间的每个维度中的权重为模型定义了它。

但是我们如何分配这些权重呢？向计算机教授语法和其他语义是一项艰巨的任务，这一点并不十分清楚，但表达每个单词的意思完全是另一回事。最重要的是，英语有几个词根据上下文有多种意思。那么这些权重是随机分配的吗(**表 1** )？

| 
 | **活着** | **财富** | **性别** |
| **男人** | one | -1 | -1 |
| **女王** | one | one | one |
| **框** | -1 | Zero | Zero |
| **表 1:** 嵌入。 |

信不信由你，答案就在最后一段本身。我们根据单词的**上下文**帮助定义单词的意思。

现在，如果这听起来让你困惑，让我们把它分解成更简单的术语。一个单词的上下文是由其相邻的单词定义的。因此，一个词的意义取决于它所联系的词。

如果您的文本语料库中有几个单词“read”与单词“book”出现在同一个句子中，Word2Vec 方法会自动将它们组合在一起。因此，这项技术完全依赖于一个好的数据集。

既然我们已经确定了 Word2Vec 的魔力在于单词联想，那么让我们更进一步，理解 Word2Vec 的两个子集。

* * *

### [**【CBOW】**](#TOC)

CBOW 是一种给定相邻单词，确定中心单词的技术。如果我们的输入句子是“我在看书。”，则窗口大小为 3 的输入对和标签将是:

*   `I`、`reading`，为标签`am`
*   `am`、`the`，为标签`reading`
*   `reading`、`book`，为标签`the`

看一看**图** **1** 。

假设我们在**图** **1** 中的输入句子是我们完整的输入文本。这使得我们的词汇量为 5，为了简单起见，我们假设有 3 个嵌入维度。

我们将考虑(`I`，`reading`)–(`am`)的输入标签对的例子。我们从`I`和`reading`(形状`1x5`)的一键编码开始，将这些编码与形状`5x3`的编码矩阵相乘。结果是一个`1x3`隐藏层。

这个隐藏层现在乘以一个`3x5`解码矩阵，给出我们对一个`1x5`形状的预测。这是比较实际的标签(`am`)一热编码相同的形状，以完成架构。

这里的星号是编码/解码矩阵。该损失影响这些矩阵在适应数据时的权重。矩阵提供了一个表达每个单词的有限空间。矩阵成为单词的矢量表示。

单个矩阵可以用于实现编码/解码两个目的。为了解开这两个任务之间的纠缠，我们将使用两个矩阵:当将单词视为相邻单词时，用上下文单词矩阵来表示单词；当将单词视为中心单词时，用中心单词矩阵来表示单词。

使用两个矩阵给每个单词两个不同的空间范围，同时给我们两个不同的视角来看待每个单词。

* * *

### [**Skip-Gram**](#TOC)

今天关注的第二种技术是跳格法。这里，给定中心词，我们必须预测它的相邻词。与 CBOW 完全相反，但效率更高。在此之前，让我们先了解一下什么是跳格。

假设我们给定的输入句子是“我正在看书”窗口大小为 3 时，相应的跳跃码对将是:

*   `am`，用于标签`I`和`reading`
*   `reading`，用于标签`am`和`the`
*   `the`，用于标签`reading`和`book`

我们来分析一下**图二**。

就像在 CBOW 中一样，让我们假设我们在**图 2** 中的输入句子是我们完整的输入文本。这使得我们的词汇量为 5，为了简单起见，我们假设有 3 个嵌入维度。

从编码矩阵开始，我们获取位于中心词索引处的向量(在本例中为`am`)。转置它，我们现在有了单词`am`的一个`3x1`向量表示(因为我们直接抓取了编码矩阵的一行，所以这个**不会是**的一次性编码)。

我们将这个向量表示乘以形状`5x3`的解码矩阵，得到形状`5x1`的预测输出。现在，这个向量基本上是整个词汇表的 softmax 表示，指向属于输入中心单词的相邻单词的索引。在这种情况下，输出应该指向`I`和`reading`的索引。

同样，为了更好地表示，我们将使用两种不同的矩阵。Skip-Gram 直观上比 CBOW 工作得更好，因为我们基于单个输入单词对几个单词进行分组，而在 CBOW 中，我们试图基于几个输入单词将一个单词关联起来。

对这两种技术有了基本的了解之后，让我们看看如何实现 CBOW 和 Skip-Gram。

* * *

### [**配置您的开发环境**](#TOC)

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
$ pip install tensorflow
```

**如果您需要帮助配置 OpenCV 的开发环境，我们*强烈推荐*阅读我们的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让您启动并运行。

* * *

### [**在配置开发环境时遇到了问题？**](#TOC)

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

* * *

### [**项目结构**](#TOC)

我们首先需要回顾我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

从这里，看一下目录结构:

```py
!tree .
.
├── data.txt
├── LICENSE
├── outputs
│   ├── loss_CBOW.png
│   ├── loss_skipgram.png
│   ├── output.txt
│   ├── TSNE_CBOW.png
│   └── tsne_skipgram.png
├── pyimagesearch
│   ├── config.py
│   ├── create_vocabulary.py
│   └── __init__.py
├── README.md
├── requirements.txt
├── train_CBOW.py
└── train_SkipGram.py

2 directories, 14 files
```

我们在主目录中有两个子目录，`outputs`和`pyimagesearch`。

目录包含了我们项目的所有结果和可视化。`pyimagesearch`目录包含几个脚本:

*   `config.py`:包含完整的配置管道。
*   帮助创建我们项目的词汇。
*   `__init__.py`:使`pyimagesearch`目录作为 python 包工作。

在父目录中，我们有:

*   `train_CBOW.py`:CBOW 架构培训脚本。
*   `train_SkipGram.py`:Skip-Gram 架构的训练脚本。
*   包含我们项目的训练数据。

* * *

### [**配置先决条件**](#TOC)

在`pyimagesearch`目录中，`config.py`脚本存放了我们项目的配置管道。

```py
# import the necessary packages
import os

# define the number of embedding dimensions
EMBEDDING_SIZE = 10

# define the window size and number of iterations
WINDOW_SIZE = 5
ITERATIONS = 1000

# define the path to the output directory
OUTPUT_PATH = "outputs"

# define the path to the skipgram outputs
SKIPGRAM_LOSS = os.path.join(OUTPUT_PATH, "loss_skipgram")
SKIPGRAM_TSNE = os.path.join(OUTPUT_PATH, "tsne_skipgram")

# define the path to the CBOW outputs
CBOW_LOSS = os.path.join(OUTPUT_PATH, "loss_cbow")
CBOW_TSNE = os.path.join(OUTPUT_PATH, "tsne_cbow")
```

在**第 5** 行，我们已经为你的嵌入矩阵定义了维数。接下来，我们定义上下文单词的窗口大小和迭代次数(**第 8 行和第 9 行**)。

输出路径在**行 12** 上定义，随后是 Skip-Gram 架构的损耗和 TSNE 图。我们对 CBOW 损失图和 TSNE 图做了同样的处理(**第 19 行和第 20 行**)。

* * *

### [**建筑词汇**](#TOC)

在我们的父目录中，有一个名为`data.txt`的文件，其中包含了我们将用来展示 Word2Vec 技术的文本。在这种情况下，我们使用的是关于诺贝尔奖获得者玛丽·居里的一段话。

然而，要应用 Word2Vec 算法，我们需要正确地处理数据。这涉及到符号化。`pyimagesearch`目录中的`create_vocabulary.py`脚本将帮助我们构建项目的词汇表。

```py
# import the necessary packages
import tensorflow as tf

def tokenize_data(data):
  # convert the data into tokens
  tokenizedText = tf.keras.preprocessing.text.text_to_word_sequence(
    input_text=data
  )

  # create and store the vocabulary of unique words along with the
  # size of the tokenized texts
  vocab = sorted(set(tokenizedText))
  tokenizedTextSize = len(tokenizedText)

  # return the vocabulary, size of the tokenized text, and the 
  # tokenized text
  return (vocab, tokenizedTextSize, tokenizedText) 
```

在**第 4 行**上，我们有一个名为`tokenize_data`的函数，它以文本数据为自变量。谢天谢地，由于`tensorflow`，我们可以使用`tf.keras.preprocessing.text.text_to_word_sequence`直接标记我们的数据。这将文本数据中的所有单词确认为标记。

现在，如果我们对标记化的文本进行排序，这将为我们提供词汇表(**第 12 行**)。这是因为令牌是基于字母顺序等范例创建的。所以我们最初的`tokenizedText`变量只有文本的标记化版本，没有排序的词汇。

在第 17 行**上，我们返回创建的词汇、标记化文本的大小以及标记化文本本身。**

* * *

### [**培训 CBOW 架构**](#TOC)

我们将从 CBOW 实现开始。在父目录中，`train_CBOW.py`脚本包含完整的 CBOW 实现。

```py
# USAGE
# python -W ignore train_CBOW.py

# set seed for reproducibility
import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from pyimagesearch import config
from pyimagesearch.create_vocabulary import tokenize_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# read the text data from the disk
print("[INFO] reading the data from the disk...")
with open("data.txt") as filePointer:
  lines = filePointer.readlines()
textData = "".join(lines)
```

首先，我们将首先获取数据，并将其转换成适合我们的`create_vocabulary`脚本的格式。在**的第 20-22 行**，我们打开`data.txt`文件并读取其中的行。但是，这将产生一个列表，其中包含单个字符串格式的所有行。

为了解决这个问题，我们简单地使用`"".join`函数，将所有的行连接成一个字符串。

```py
# tokenize the text data and store the vocabulary, the size of the
# tokenized text, and the tokenized text
(vocab, tokenizedTextSize, tokenizedText) = tokenize_data(
  data=textData
)

# map the vocab words to individual indices and map the indices to
# the words in vocab
vocabToIndex = {
  uniqueWord:index for (index, uniqueWord) in enumerate(vocab)
}
indexToVocab = np.array(vocab)

# convert the tokens into numbers
textAsInt = np.array([vocabToIndex[word] for word in tokenizedText])

# create the representational matrices as variable tensors
contextVectorMatrix =  tf.Variable(
  np.random.rand(tokenizedTextSize, config.EMBEDDING_SIZE)
)
centerVectorMatrix = tf.Variable(
  np.random.rand(tokenizedTextSize, config.EMBEDDING_SIZE)
)
```

有了正确格式的数据，我们现在可以在第 26-28 行的**上使用`tokenize_data`函数对数据进行标记。**

使用构建的词汇，我们可以将单词映射到它们的索引，并创建一个`vocabToIndex`字典，帮助我们将单词链接到它们的索引(**第 32-34 行**)。

类似地，我们在第 35 行的**上创建一个`indexToVocab`变量。这里的最后一步是在第 38** 行将我们的整个文本数据从 word 格式转换成 indices 格式。

现在，是时候创建我们的嵌入空间了。如前所述，我们将使用两个不同的矩阵，一个用于上下文单词表示，一个用于中心单词表示。每个单词都会用两个空格表示(**第 41-46 行**)。

```py
# initialize the optimizer and create an empty list to log the loss
optimizer = tf.optimizers.Adam()
lossList = list()

# loop over the training epochs
print("[INFO] Starting CBOW training...")
for iter in tqdm(range(config.ITERATIONS)):
  # initialize the loss per epoch
  lossPerEpoch = 0

  # the window for center vector prediction is created
  for start in range(tokenizedTextSize - config.WINDOW_SIZE):
    # generate the indices for the window
    indices = textAsInt[start:start + config.WINDOW_SIZE]
```

在**第 49 行**上，我们已经初始化了`Adam`优化器和一个列表来存储我们的每历元损失(**第 49 和 50 行**)。

现在，我们开始在历元上循环，并初始化第 54-56 行上的每历元损失变量。

随后在第 59-61 行上初始化用于 CBOW 考虑的窗口。

```py
   # initialize the gradient tape
    with tf.GradientTape() as tape:     
      # initialize the context vector
      combinedContext = 0

      # loop over the indices and grab the neighboring 
      # word representation from the embedding matrix
      for count,index in enumerate(indices):
        if count != config.WINDOW_SIZE // 2:
          combinedContext += contextVectorMatrix[index, :]

      # standardize the result according to the window size 
      combinedContext /= (config.WINDOW_SIZE-1)

      # calculate the center word embedding predictions
      output = tf.matmul(centerVectorMatrix, 
        tf.expand_dims(combinedContext, 1))

      # apply softmax loss and grab the relevant index
      softOut = tf.nn.softmax(output, axis=0)
      loss = softOut[indices[config.WINDOW_SIZE // 2]]

      # calculate the logarithmic loss
      logLoss = -tf.math.log(loss)

    # update the loss per epoch and apply 
    # the gradients to the embedding matrices
    lossPerEpoch += logLoss.numpy()
    grad = tape.gradient(
      logLoss, [contextVectorMatrix, centerVectorMatrix]
    )
    optimizer.apply_gradients(
      zip(grad, [contextVectorMatrix, centerVectorMatrix])
    )

  # update the loss list 
  lossList.append(lossPerEpoch)
```

对于梯度计算，我们在**线 64** 上初始化一个梯度带。在**行 66** 上，组合上下文向量变量被初始化。这将表示上下文向量的添加。

根据窗口循环遍历索引，我们从上下文向量矩阵中获取上下文向量表示(**行 70-72** )。

这是一个有趣的方法。你可能注意到了，我们走的是一条不同于之前解释的路线。我们可以直接从嵌入矩阵中获取这些索引，而不是将独热编码与嵌入空间相乘，因为这实质上意味着将独热编码(只有一个 1 的 0 的向量)与矩阵相乘。

我们得到了所有上下文向量的总和。我们根据第 75 行的**上考虑的上下文单词数(4)来标准化输出。**

接下来，该输出乘以第 78 和 79 行上的中心字嵌入空间。我们对这个输出应用 softmax，通过抓取属于中心单词标签的相关索引来确定损失(**第 82 行和第 83 行**)。

由于我们必须在该指数下最大化输出，我们计算第 86 行**的负对数损耗。**

基于损失，我们将梯度应用于我们创建的两个嵌入空间(**行 91-96** )。

一旦超出时期，我们就用每个时期计算的损失来更新损失列表(**行 99** )。

```py
# create output directory if it doesn't already exist
if not os.path.exists(config.OUTPUT_PATH):
  os.makedirs(config.OUTPUT_PATH)

# plot the loss for evaluation
print("[INFO] Plotting loss...")
plt.plot(lossList)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(config.CBOW_LOSS)
```

如果输出文件夹还不存在，我们将创建它(**行 102 和 103** )。每个历元的损失被标绘并保存在输出文件夹中(**第 106-110 行**)。

```py
# apply dimensional reductionality using tsne for the representation matrices
tsneEmbed = (
  TSNE(n_components=2)
  .fit_transform(centerVectorMatrix.numpy())
)
tsneDecode = (
  TSNE(n_components=2)
  .fit_transform(contextVectorMatrix.numpy())
)

# initialize a index counter
indexCount = 0 

# initialize the tsne figure
plt.figure(figsize=(25, 5))

# loop over the tsne embeddings and plot the corresponding words
print("[INFO] Plotting TSNE embeddings...")
for (word, embedding) in tsneDecode[:100]:
  plt.scatter(word, embedding)
  plt.annotate(indexToVocab[indexCount], (word, embedding))
  indexCount += 1
plt.savefig(config.CBOW_TSNE)
```

我们的嵌入空间已经准备好了，但是由于它们有许多维度，我们的肉眼将无法理解它们。解决这个问题的方法是降维。

维数约简是一种我们可以降低嵌入空间的维数，同时保持大部分重要信息(分离数据)完整的方法。在这种情况下，我们将应用[TSNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)(t-分布式随机邻居嵌入)。

使用 TSNE，我们将嵌入空间的维数减少到 2，从而使 2D 图成为可能(**行 113-120** )。基于索引，我们在 2D 空间中绘制单词(**第 129-134 行**)。

在我们检查结果之前，让我们看一下 Skip-Gram 实现。

* * *

### [**实现跳格架构**](#TOC)

正如我们前面所解释的，Skip-Gram 是一个输入得到多个输出的地方。让我们转到`train_SkipGram.py`脚本来完成我们的 Skip-Gram 实现。

```py
# USAGE
# python -W ignore train_skipgram.py

# set seed for reproducibility
import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from pyimagesearch import config
from pyimagesearch.create_vocabulary import tokenize_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# read the text data from the disk
print("[INFO] reading the data from the disk...")
with open("data.txt") as filePointer:
  lines = filePointer.readlines()
textData = "".join(lines)

# tokenize the text data and store the vocabulary, the size of the
# tokenized text, and the tokenized text
(vocab, tokenizedTextSize, tokenizedText) = tokenize_data(
  data=textData
)
```

最初的步骤类似于我们在 CBOW 实现中所做的。在将文本数据提供给`tokenize_data`函数之前，我们准备好文本数据，并获得词汇、文本大小和返回的标记化文本(**第 20-28 行**)。

```py
# map the vocab words to individual indices and map the indices to
# the words in vocab
vocabToIndex = {
  uniqueWord:index for (index, uniqueWord) in enumerate(vocab)
}
indexToVocab = np.array(vocab)

# convert the tokens into numbers
textAsInt = np.array([vocabToIndex[word] for word in tokenizedText])

# create the representational matrices as variable tensors
contextVectorMatrix =  tf.Variable(
  np.random.rand(tokenizedTextSize, config.EMBEDDING_SIZE)
)
centerVectorMatrix = tf.Variable(
  np.random.rand(tokenizedTextSize, config.EMBEDDING_SIZE)
)
```

下一步是创建到词汇和文本的索引映射，这是重复的(**第 32-38 行**)。然后我们创建两个嵌入空间，我们将在**第 41-46 行**使用。

```py
# initialize the optimizer and create an empty list to log the loss
optimizer = tf.optimizers.Adam()
lossList = list()

# loop over the training epochs
print("[INFO] Starting SkipGram training...")
for iter in tqdm(range(config.ITERATIONS)):
  # initialize the loss per epoch
  lossPerEpoch = 0

  # the window for center vector prediction is created
  for start in range(tokenizedTextSize - config.WINDOW_SIZE):
    # generate the indices for the window
    indices = textAsInt[start:start + config.WINDOW_SIZE]
```

在**第 49 行**上，我们已经初始化了`Adam`优化器和一个列表来存储我们的每历元损失(**第 49 和 50 行**)。

现在，我们开始在历元上循环，并初始化第 54-56 行上的每历元损失变量。

随后在**行 59-61** 上初始化窗口跳过程序考虑。

```py
    # initialize the gradient tape
    with tf.GradientTape() as tape: 
      # initialize the context loss
      loss = 0  

      # grab the center word vector and matrix multiply the
      # context embeddings with the center word vector
      centerVector = centerVectorMatrix[
        indices[config.WINDOW_SIZE // 2],
        :
      ]
      output = tf.matmul(
        contextVectorMatrix, tf.expand_dims(centerVector ,1)
      )

      # pass the output through a softmax function
      softmaxOutput = tf.nn.softmax(output, axis=0)

      # loop over the indices of the neighboring words and 
      # update the context loss w.r.t each neighbor
      for (count, index) in enumerate(indices):
        if count != config.WINDOW_SIZE // 2:
          loss += softmaxOutput[index]

      # calculate the logarithmic value of the loss
      logLoss = -tf.math.log(loss)

    # update the loss per epoch and apply the gradients to the
    # embedding matrices
    lossPerEpoch += logLoss.numpy()
    grad = tape.gradient(
      logLoss, [contextVectorMatrix, centerVectorMatrix]
    )
    optimizer.apply_gradients(
      zip(grad, [contextVectorMatrix, centerVectorMatrix])
    ) 

  # update the loss list
  lossList.append(lossPerEpoch)
```

在**行 64** 上，我们初始化一个梯度带，用于矩阵的梯度计算。这里，我们需要创建一个变量来存储每个上下文的损失(**第 66 行**)。

我们从中心词嵌入空间获取中心词表示，并将其乘以上下文向量矩阵(**行 70-76** )。

该输出通过 softmax，并且相关的上下文索引被采样用于损失计算(**行 79-85** )。

由于我们必须最大化相关指数，我们计算第 88 行**的负对数损耗。**

然后根据损失计算梯度，并应用于两个嵌入空间(**行 93-99** )。每个时期的损失然后被存储在**线 101** 上。

```py
# create output directory if it doesn't already exist
if not os.path.exists(config.OUTPUT_PATH):
  os.makedirs(config.OUTPUT_PATH)

# plot the loss for evaluation
print("[INFO] Plotting Loss...")
plt.plot(lossList)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(config.SKIPGRAM_LOSS)
```

如果输出文件夹还不存在，我们将创建它(**行 104 和 105** )。每个历元的损失被标绘并保存在输出文件夹中(**第 108-112 行**)。

```py
# apply dimensional reductionality using tsne for the
# representation matrices
tsneEmbed = (
  TSNE(n_components=2)
  .fit_transform(centerVectorMatrix.numpy())
)
tsneDecode = (
  TSNE(n_components=2)
  .fit_transform(contextVectorMatrix.numpy())
)

# initialize a index counter
indexCount = 0 

# initialize the tsne figure
plt.figure(figsize=(25, 5))

# loop over the tsne embeddings and plot the corresponding words
print("[INFO] Plotting TSNE Embeddings...")
for (word, embedding) in tsneEmbed[:100]:
  plt.scatter(word, embedding)
  plt.annotate(indexToVocab[indexCount], (word, embedding))
  indexCount += 1
plt.savefig(config.SKIPGRAM_TSNE)
```

如同对 CBOW 图所做的，我们应用维数约简并在 2D 空间中绘制单词(**行 114-137** )。

* * *

### [**可视化结果**](#TOC)

让我们来看看损失(**图 4 和图 5**

不要被损失值震惊。如果从分类的角度来看，每个窗口都有不同的标签。损失肯定会很大，但谢天谢地，它已经降到了一个相当可观的价值。

对于跳格损耗来说，想法是类似的，但是因为我们有一个输入的多个输出，损耗下降得更快。

再来看嵌入空间本身(**图 6 和图 7** )。

CBOW 嵌入空间相当分散，很少形成视觉联想。如果我们仔细观察，我们可能会发现类似的上下文单词，如组合在一起的年份等。，但一般来说，结果并不是那么好。

Skip-Gram 嵌入空间要好得多，因为我们可以看到已经形成的几个可视单词分组。这表明在我们特定的小数据集实例中，Skip-Gram 工作得更好。

* * *

* * *

## [**汇总**](#TOC)

Word2Vec 本质上是理解 NLP 中表征学习的一个重要里程碑。它提出的想法非常直观，为教会计算机如何理解单词的含义这一问题提供了有效的解决方案。

最棒的是，数据为我们做了大部分工作。这些关联是根据一个词与另一个词一起出现的频率形成的。虽然近年来出现了许多更好的算法，但嵌入空间的影响在所有这些算法中都可以找到。

* * *

### [**引用信息**](#TOC)

**Chakraborty，d .**“word 2 vec:自然语言处理中的嵌入研究”， *PyImageSearch* ，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva，R. Raha，A. Thanki 编辑。，2022 年，【https://pyimg.co/2fb0z 

```py
@incollection{Chakraborty_2022_Word2Vec,
  author = {Devjyoti Chakraborty},
  title = {{Word2Vec}: A Study of Embeddings in {NLP}},
  booktitle = {PyImageSearch},
  editor = {Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/2fb0z},
}
```

* * *

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****