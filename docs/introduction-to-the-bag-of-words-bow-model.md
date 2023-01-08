# 单词袋(BoW)模型简介

> 原文：<https://pyimagesearch.com/2022/07/04/introduction-to-the-bag-of-words-bow-model/>

* * *

## **目录**

* * *

## [**介绍词袋(BoW)模型**](#TOC)

基于文本数据创建统计模型总是比基于图像数据建模更复杂。图像数据包含可检测的模式，这可以帮助模型识别它们。文本数据中的模式更复杂，使用传统方法需要更多的计算。

上周，我们简单回顾了一下自然语言处理(NLP)的历史。今天，我们将学习计算机语言数据建模中最先使用的技术之一，单词袋(BoW)。

在本教程中，您将了解单词袋模型以及如何实现它。

本课是关于 NLP 101 的四部分系列的第二部分:

1.  [*自然语言处理入门*](https://pyimg.co/60xld)
2.  ****[袋字(弓)模型简介](https://pyimg.co/oa2kt)* (今日教程)***
3.  **Word2Vec:自然语言处理中的嵌入研究**
4.  *【BagofWords 和 Word2Vec 的比较*

 ***要学习如何** **实现词汇袋模型** **，** ***就继续阅读吧。***

* * *

## [**介绍词袋(BoW)模型**](#TOC)

* * *

### [**弓型简介**](#TOC)

单词袋模型是一种从文本数据中提取特征的简单方法。这个想法是将每个句子表示为一个单词包，不考虑语法和范例。句子中单词的出现为模型定义了句子的含义。

这可以被认为是表征学习的延伸，你在一个 N 维空间中表征句子。对于每一句话，模型将为每个维度分配一个权重。这将成为模型的句子标识。

让我们更深入地了解它的含义。看一下**图一**。

我们有两句话:“我有一只狗”和“你有一只猫。”首先，我们获取当前词汇表中的所有单词，并创建一个表示矩阵，其中每一列都对应一个单词，如图 1 所示。

我们的句子总共有 8 个单词，但是因为我们有 2 个单词(`have`、`a`)重复，所以总词汇量变成了 6。现在我们有 6 列代表词汇表中的每个单词。

现在，每个句子都表示为词汇表中所有单词的组合。例如，“我有一只狗”在词汇表中有 6 个单词中的 4 个，因此我们将打开现有单词的位，关闭句子中不存在的单词的位。

因此，如果 vocab 矩阵列按照`I`、`have`、`a`、`dog`、`you`和`cat`的顺序排列，那么第一句话(“我有一只狗”)表示变成`1,1,1,1,0,0`，而第二句话(“你有一只猫”)表示变成`0,1,1,0,1,1`。

这些表征成为让一个模型理解不同句子本质的关键。我们确实忽略了语法，但是因为这些句子是相对于完整的词汇来看的，所以每个句子都有独特的标记化表示，这有助于它们从其他句子中脱颖而出。

例如，第一个句子会很突出，因为它打开了`dog`和`I`位，而第二个句子打开了`cat`和`you`位。表示中的这些小变化有助于我们使用单词袋方法对文本数据建模。

这里，我们用逐位方法解释了 BoW。BoW 还可以配置为存储单词的出现频率，以便在模型训练期间进行额外的强化。

* * *

### [**鞠躬的利弊**](#TOC)

马上，我们看到了这种方法的一个主要问题。如果我们的输入数据很大，那就意味着词汇量也会增加。这反过来又使我们的表示矩阵变得更大，计算变得非常复杂。

另一个计算噩梦是在我们的矩阵(即稀疏矩阵)中包含许多 0。稀疏矩阵包含的信息较少，会浪费大量内存。

单词袋最大的缺点是完全不会学习语法和语义。表示矩阵中的标记化表示定义了一个句子，只有单词在句子中的出现/不出现才能将其与其他单词区分开来。

从更好的方面来看，单词袋方法以一种显著的方式强调了表征学习的好处。它简单直观的方法至少帮助我们解释了单词的组合对计算机意味着什么。

当然，这就给词汇袋的应用带来了问题。首先，这是向更复杂的表示学习示例(如 Word2Vec 和 Glove)迈出的一大步。由于它也呼应了“一键编码”表示的概念，词袋主要用于文本文档的特征生成。

既然我们已经掌握了单词袋的概念，那就让我们来实施它吧！

* * *

### [**配置您的开发环境**](#TOC)

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
$ pip install tensorflow
$ pip install numpy
```

**如果您需要帮助配置 OpenCV 的开发环境，我们*强烈推荐*阅读我们的** [***pip 安装 OpenCV* 指南**](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让您启动并运行。

* * *

### [**在配置开发环境时遇到了问题？**](#TOC)

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://www.pyimagesearch.com/pyimagesearch-university/)吧！

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
├── pyimagesearch
│   ├── bow.py
│   ├── config.py
│   ├── data_processing.py
│   ├── __init__.py
│   ├── model.py
│   └── tensorflow_wrapper.py
└── train.py

1 directory, 7 files
```

在`pyimagesearch`目录中，我们有:

*   将单词袋技术应用到句子中。
*   `config.py`:包含项目的配置管道。
*   `data_processing.py`:容纳数据处理实用程序。
*   `__init__.py`:将`pyimagesearch`目录变成 python 包。
*   `model.py`:包含一个小型的神经网络架构。
*   `tensorflow_wrapper.py`:包含用`tensorflow`工具包装的单词包方法。

在父目录中，我们有:

*   包含单词袋方法的训练脚本。

* * *

### [**配置先决条件**](#TOC)

在`pyimagesearch`目录中，我们有一个名为`config.py`的脚本，它包含了这个项目的配置管道。

```py
# define the data to be used
dataDict = {
    "sentence":[
        "Avengers is a great movie.",
        "I love Avengers it is great.",
        "Avengers is a bad movie.",
        "I hate Avengers.",
        "I didnt like the Avengers movie.",
        "I think Avengers is a bad movie.",
        "I love the movie.",
        "I think it is great."
    ],
    "sentiment":[
        "good",
        "good",
        "bad",
        "bad",
        "bad",
        "bad",
        "good",
        "good"
    ]
}

# define a list of stopwords
stopWrds = ["is", "a", "i", "it"] 

# define model training parameters
epochs = 30
batchSize = 10

# define number of dense units
denseUnits = 50
```

我们将使用 NumPy 和 TensorFlow 实现单词包，以比较这两种方法。在**的第 2-23 行**，我们有`dataDict`，一个包含我们输入数据的字典，这些数据被分成句子和它们相应的标签。

在第 26 行的**上，我们定义了一个停用词列表，由于 BoW 方法不关心语法和语义，我们将从数据中省略它。每个句子的独特单词越多，就越容易与其他句子区分开来。**

在**第 29-33 行**上，我们为 TensorFlow 模型定义了一些参数，如历元数、批量大小以及要添加到我们的小型神经网络中的密集单元数。配置管道到此结束。

* * *

### [**处理我们的数据为**弓近](#TOC)

作为深度学习的从业者，你我大概都忽略了，把日常项目中的很多东西想当然了。我们几乎在每个处理任务中都使用 TensorFlow/PyTorch 包装器，却忘记了是什么让这些包装器如此重要。

对于我们今天的项目，我们将自己编写一些预处理包装器。为此，让我们进入`data_processing.py`脚本。

```py
# import the necessary packages
import re

def preprocess(sentDf, stopWords, key="sentence"):
    # loop over all the sentences
    for num in range(len(sentDf[key])):
        # lowercase the string and remove punctuation
        sentence = sentDf[key][num]
        sentence = re.sub(
            r"[^a-zA-Z0-9]", " ", sentence.lower()
        ).split()

        # define a list for processed words
        newWords = list()

        # loop over the words in each sentence and filter out the
        # stopwords
        for word in sentence:
            if word not in stopWords:
                # append word if not a stopword    
                newWords.append(word)

        # replace sentence with the list of new words   
        sentDf[key][num] = newWords

    # return the preprocessed data
    return sentDf
```

在第 4 行的**上，我们有第一个帮助函数`preprocess`，它接受以下参数:**

*   `sentDf`:输入数据帧。
*   `stopWords`:要从数据中省略的单词列表。
*   `key`:进入输入数据帧相关部分的键。

我们循环遍历数据帧中所有可用的句子(**第 6-27 行**)，将单词小写，去掉标点符号，并省略停用词。

```py
def prepare_tokenizer(df, sentKey="sentence", outputKey="sentiment"):
    # counters for tokenizer indices
    wordCounter = 0
    labelCounter = 0

    # create placeholder dictionaries for tokenizer
    textDict = dict()
    labelDict = dict()

    # loop over the sentences
    for entry in df[sentKey]:
        # loop over each word and
        # check if encountered before
        for word in entry:
            if word not in textDict.keys():
                textDict[word] = wordCounter

                # update word counter if new
                # word is encountered
                wordCounter += 1

    # repeat same process for labels  
    for label in df[outputKey]:
        if label not in labelDict.keys():
            labelDict[label] = labelCounter
            labelCounter += 1

    # return the dictionaries 
    return (textDict, labelDict)
```

这个脚本中的第二个函数是`prepare_tokenizer` ( **第 29 行**)，它接受以下参数:

*   `df`:我们将从中创建我们的标记器的数据帧。
*   `sentKey`:从数据框中进入句子的键。
*   `outputKey`:从数据框中访问标签的键。

首先，我们为第 31 行和第 32 行的索引创建计数器。在**的第 35 行和第 36 行**，我们为分词器创建字典。

接下来，我们开始循环句子(**第 39 行**)，并将单词添加到我们的字典中。如果我们遇到一个我们已经见过的单词，我们忽略它。如果这个单词是新遇到的，它将被添加到词典中(**第 42-48 行**)。

我们对标签应用相同的过程(**第 51-54 行**)，结束`prepare_tokenizer`脚本。

* * *

### [**建立文字袋功能**](#TOC)

现在，我们将进入`bow.py`脚本，看看我们计算单词包的自定义函数。

```py
def calculate_bag_of_words(text, sentence):
    # create a dictionary for frequency check
    freqDict = dict.fromkeys(text, 0)

    # loop over the words in sentences
    for word in sentence:
        # update word frequency
        freqDict[word]=sentence.count(word)

    # return dictionary 
    return freqDict
```

函数`calculate_bag_of_words`接受词汇和句子作为它的参数(**第 1 行**)。接下来，我们在第三行创建一个字典来检查和存储单词的出现。

循环一个句子中的每个单词(**第 6 行**)，我们计算一个特定单词出现的次数并返回它(**第 8-11 行**)。

* * *

### [**张量流缠绕:另类**](#TOC)

到目前为止，我们已经看到了自己创建所有预处理功能是什么样子。如果你觉得太复杂，我们也将向你展示如何使用 TensorFlow 进行同样的过程。让我们进入`tensorflow_wrapper.py`。

```py
# import the necessary packages
from tensorflow.keras.preprocessing.text import Tokenizer 

def tensorflow_wrap(df):
    # create the tokenizer for sentences
    tokenizerSentence = Tokenizer()

    # create the tokenizer for labels
    tokenizerLabel = Tokenizer()

    # fit the tokenizer on the documents
    tokenizerSentence.fit_on_texts(df["sentence"])

    # fit the tokenizer on the labels
    tokenizerLabel.fit_on_texts(df["sentiment"])

    # create vectors using tensorflow
    encodedData = tokenizerSentence.texts_to_matrix(
        texts=df["sentence"], mode="count")

    # add label column
    labels = df["sentiment"]

    # correct label vectors
    for i in range(len(labels)):
        labels[i] = tokenizerLabel.word_index[labels[i]] - 1

    # return data and labels
    return (encodedData[:, 1:], labels.astype("float32"))
```

在脚本内部，我们有`tensorflow_wrap` ( **第 4 行**)函数，它接受数据帧作为参数。

在**的第 6-9 行**，我们分别为句子和标签初始化分词器。通过简单地使用`fit_on_texts`函数，我们已经完成了句子和标签的标记器的创建(**第 12-15 行**)。

使用另一个名为`texts_to_matrix`的函数来创建我们的编码，我们得到了经过处理的句子的矢量化格式(**第 18 行和第 19 行**)。

在第 22-26 行**上，我们创建标签，然后在第 29** 行**上返回编码和标签。**

```py
#import the necessary packages
import pyimagesearch.config as config
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def build_shallow_net():
    # define the model
    model = Sequential()
    model.add(Dense(config.denseUnits, input_dim=10, activation="relu"))
    model.add(Dense(config.denseUnits, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # compile the keras model
    model.compile(loss="binary_crossentropy", optimizer="adam",
        metrics=["accuracy"]
    )

    # return model
    return model
```

在第 6 行的**上，我们定义了`build_shallow_net`函数，它初始化了一个浅层神经网络。**

网络从一个`dense`层开始，其中输入数设置为 10。这是因为我们的文本语料库经过处理后的词汇量是 10。接下来是两个致密层，最后是乙状结肠激活的输出层(**第 8-11 行**)。

在**第 14 行和第 15 行**，我们用`binary_crossentropy`损失、`adam`优化器和`accuracy`作为度量来编译模型。

这样，我们的模型就可以使用了。

* * *

### [**训练弓模型**](#TOC)

现在是时候结合我们所有的模块，训练单词袋模型方法了。让我们进入`train.py`脚本。

```py
# USAGE
# python train.py

# import the necessary packages
from pyimagesearch import config
from pyimagesearch.model import build_shallow_net
from pyimagesearch.bow import calculate_bag_of_words
from pyimagesearch.data_processing import preprocess
from pyimagesearch.data_processing import prepare_tokenizer
from pyimagesearch.tensorflow_wrapper import tensorflow_wrap
import pandas as pd

# convert the input data dictionary to a pandas data frame
df = pd.DataFrame.from_dict(config.dataDict)

# preprocess the data frame and create data dictionaries
preprocessedDf = preprocess(sentDf=df, stopWords=config.stopWrds)
(textDict, labelDict) = prepare_tokenizer(df)

# create an empty list for vectors
freqList = list()

# build vectors from the sentences
for sentence in df["sentence"]:
    # create entries for each sentence and update the vector list   
    entryFreq = calculate_bag_of_words(text=textDict,
        sentence=sentence)
    freqList.append(entryFreq)
```

在**第 14 行**，我们将`config.py`中定义的输入数据字典转换成数据帧。然后使用`data_processing.py`创建的`preprocess`函数处理数据帧(**行 17** )。随后在**行 18** 上创建记号赋予器。

我们生成一个空列表来存储在**行 21** 出现的单词。循环每个句子，使用位于`bow.py` ( **第 24-28 行**)的`calculate_bag_of_words`函数计算单词的频率。

```py
# create an empty data frame for the vectors
finalDf = pd.DataFrame() 

# loop over the vectors and concat them
for vector in freqList:
    vector = pd.DataFrame([vector])
    finalDf = pd.concat([finalDf, vector], ignore_index=True)

# add label column to the final data frame
finalDf["label"] = df["sentiment"]

# convert label into corresponding vector
for i in range(len(finalDf["label"])):
    finalDf["label"][i] = labelDict[finalDf["label"][i]]

# initialize the vanilla model
shallowModel = build_shallow_net()
print("[Info] Compiling model...")

# fit the Keras model on the dataset
shallowModel.fit(
    finalDf.iloc[:,0:10],
    finalDf.iloc[:,10].astype("float32"),
    epochs=config.epochs,
    batch_size=config.batchSize
)
```

在**行 31** 上创建一个空数据帧来存储矢量化输入。`freqList`中的每个向量被添加到第 34-36 行的空数据帧中。

标签被添加到第 39 行**的数据帧中。但是由于标签仍然是字符串格式，我们在**的第 42 行和第 43 行**将它们转换成矢量格式。**

用于训练的普通模型在**行 46** 上初始化，我们继续在**行 50-55** 上拟合训练数据和标签。因为我们已经将标签列添加到数据帧中，所以我们可以使用`iloc`功能来分离数据和标签(`0:10`用于数据，`10`用于标签)。

```py
# create dataset using TensorFlow
trainX, trainY = tensorflow_wrap(df)

# initialize the new model for tf wrapped data
tensorflowModel = build_shallow_net()
print("[Info] Compiling model with tensorflow wrapped data...")

# fit the keras model on the tensorflow dataset
tensorflowModel.fit(
    trainX,
    trainY,
    epochs=config.epochs,
    batch_size=config.batchSize
)
```

现在我们进入`tensorflow`包装的数据。仅仅一行代码(**第 58 行**)就让我们得到了`trainX`(数据)和`trainY`(标签)。数据被放入名为`tensorflowModel` ( **第 61-70 行**)的不同模型中。

* * *

### [**了解培训指标**](#TOC)

需要记住的重要一点是，我们的数据集非常小，结果应该被认为是不确定的。然而，让我们来看看我们的训练精度。

```py
[INFO] Compiling model...
Epoch 1/30
1/1 [==============================] - 0s 495ms/step - loss: 0.7262 - accuracy: 0.5000
Epoch 2/30
1/1 [==============================] - 0s 10ms/step - loss: 0.7153 - accuracy: 0.5000
Epoch 3/30
1/1 [==============================] - 0s 10ms/step - loss: 0.7046 - accuracy: 0.5000
...
Epoch 27/30
1/1 [==============================] - 0s 7ms/step - loss: 0.4756 - accuracy: 1.0000
Epoch 28/30
1/1 [==============================] - 0s 5ms/step - loss: 0.4664 - accuracy: 1.0000
Epoch 29/30
1/1 [==============================] - 0s 10ms/step - loss: 0.4571 - accuracy: 1.0000
Epoch 30/30
1/1 [==============================] - 0s 5ms/step - loss: 0.4480 - accuracy: 1.0000
```

我们的普通模型在`30th`时期达到了`100%`的精度，这是给定数据集的大小所期望的。

```py
<keras.callbacks.History at 0x7f7bc5b5a110>
[Info] Compiling Model with Tensorflow wrapped data...
1/30
1/1 [==============================] - 1s 875ms/step - loss: 0.6842 - accuracy: 0.5000
Epoch 2/30
1/1 [==============================] - 0s 14ms/step - loss: 0.6750 - accuracy: 0.5000
Epoch 3/30
1/1 [==============================] - 0s 7ms/step - loss: 0.6660 - accuracy: 0.5000
...
Epoch 27/30
1/1 [==============================] - 0s 9ms/step - loss: 0.4730 - accuracy: 0.8750
Epoch 28/30
1/1 [==============================] - 0s 12ms/step - loss: 0.4646 - accuracy: 0.8750
Epoch 29/30
1/1 [==============================] - 0s 12ms/step - loss: 0.4561 - accuracy: 0.8750
Epoch 30/30
1/1 [==============================] - 0s 9ms/step - loss: 0.4475 - accuracy: 0.8750
<keras.callbacks.History at 0x7f7bc594c710>
```

由于数据集较小，数据包装模型在其最后一个时期似乎也达到了相当高的精度。

很明显，这两个模型都必须过度拟合。然而，要注意的重要一点是，当涉及到文本数据时，我们很可能希望我们的模型过度拟合训练数据以获得最佳结果。

如果您想知道为什么我们希望我们的模型过度拟合，那是因为当涉及到文本数据时，您的训练文本数据成为您的不容置疑的戒律。假设一个特定的单词在不同的句子中出现了多次。尽管如此，对于类似的上下文，您肯定希望您的模型能够理解并适应它，这样这个词的意思对模型来说就变得很清楚了。

正如我前面提到的，文本数据与图像数据有很大的不同。我们在进行过度拟合语句时考虑的一个假设是，训练数据将覆盖出现在不同上下文中的单词的几乎所有实例。

* * *

* * *

## [**汇总**](#TOC)

今天我们学习了单词袋(BoW)，这是自然语言处理(NLP)中表征学习的一个很好的介绍。我们本质上是按照一定的模式，将我们的数据重新打包成标记组。这些有助于模型对句子的意思有一个基本的理解。

自然语言处理的 BoW 方法在考虑上下文和含义的能力方面是有限的。自然，将句子表示为词汇出现在处理多义词和同音异义词时是无效的。

它无法解释句法依赖性和非标准文本，这表明 BoW 不是一个强大的算法。但是在 NLP 发展的背景下，这种技术开启了表征学习的许多后续推动，因此成为 NLP 历史的关键部分。

* * *

### [**引用信息**](#TOC)

**Chakraborty，D.** “单词袋(BoW)模型介绍”， *PyImageSearch* ，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva，R. Raha 和 A. Thanki 编辑。，2022 年，【https://pyimg.co/oa2kt 

```py
@incollection{Chakraborty_2022_BoW,
  author = {Devjyoti Chakraborty},
  title = {Introduction to the Bag-of-Words {(BoW)} Model},
  booktitle = {PyImageSearch},
  editor = {Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/oa2kt},
}
```

* * *

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****