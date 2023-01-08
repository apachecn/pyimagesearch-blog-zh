# 使用 TensorFlow 和 Keras 深入了解变形金刚:第 3 部分

> 原文：<https://pyimagesearch.com/2022/11/07/a-deep-dive-into-transformers-with-tensorflow-and-keras-part-3/>

* * *

## **目录**

* * *

## [**用 TensorFlow 和 Keras 深入了解变形金刚:第三部**](#TOC)

在本教程中，您将学习**如何在 TensorFlow 和 Keras** 中从头开始编写转换器架构。

本课是关于 NLP 104 的 3 部分系列的最后一课:

1.  [*用 TensorFlow 和 Keras 深入了解变形金刚:第 1 部分*](https://pyimg.co/8kdj1)
2.  [*用 TensorFlow 和 Keras 深入了解变形金刚:第二部*](https://pyimg.co/pzu1j)
3.  [***用 TensorFlow 和 Keras 深度潜入变形金刚:第三部***](https://pyimg.co/9nozd) **(今日教程)**

**要了解如何使用 TensorFlow 和 Keras 构建变压器架构，** ***继续阅读。***

* * *

## [**用 TensorFlow 和 Keras 深入了解变形金刚:第三部**](#TOC)

我们现在是变形金刚系列的第三部分，也是最后一部分。在[第一部分](https://t.co/gkq0EczVnI)中，我们了解了注意力从简单的前馈网络到现在的多头自我注意的演变。接下来，在[第二部分](https://t.co/taK51gSAPZ)中，我们将重点放在连接线上，除了注意力之外的各种组件，它们将架构连接在一起。

教程的这一部分将主要关注使用 TensorFlow 和 Keras 从头构建一个转换器，并将其应用于神经机器翻译的任务。对于代码，我们受到了关于变形金刚的官方 TensorFlow 博客文章的极大启发。

正如所讨论的，我们将了解如何构建每个组件，并最终将它们缝合在一起，以训练我们自己的 Transformer 模型。

* * *

### [**简介**](#TOC)

在前面的教程中，我们介绍了构建 Transformer 架构所需的每个组件和模块。在这篇博文中，我们将重新审视这些组件，看看如何使用 TensorFlow 和 Keras 构建这些模块。

然后，我们将布置训练管道和推理脚本，这是训练和测试整个 Transformer 架构所需要的。

这是一个拥抱面部空间的演示，展示了只在 **25** 个时期训练的模型。 ***这个空间的目的不是挑战谷歌翻译，而是展示用我们的代码训练你的模型并将其投入生产*** 是多么容易。

 <gradio-app space="pyimagesearch/nmt-transformer">* * *

### [**配置您的开发环境**](#TOC)

要遵循这个指南，您需要在系统上安装`tensorflow`和`tensorflow-text`。

幸运的是，TensorFlow 可以在 pip 上安装:

```py
$ pip install tensorflow==2.8.0
$ pip install tensorflow-text==2.8.0
```

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
$ tree .
.
├── inference.py
├── pyimagesearch
│   ├── attention.py
│   ├── config.py
│   ├── dataset.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── feed_forward.py
│   ├── __init__.py
│   ├── loss_accuracy.py
│   ├── positional_encoding.py
│   ├── rate_schedule.py
│   ├── transformer.py
│   └── translate.py
└── train.py

1 directory, 14 files
```

在`pyimagesearch`目录中，我们有以下内容:

*   `attention.py`:保存所有自定义注意模块
*   `config.py`:任务的配置文件
*   `dataset.py`:数据集管道的实用程序
*   `decoder.py`:解码器模块
*   `encoder.py`:编码器模块
*   `feed_forward.py`:点式前馈网络
*   `loss_accuracy.py`:保存训练模型所需的损失和准确性的代码片段
*   `positional_encoding.py`:模型的位置编码方案
*   `rate_schedule.py`:培训管道的学习率计划程序
*   `transformer.py`:变压器模块
*   训练和推理模型

在核心目录中，我们有两个脚本:

*   `train.py`:运行脚本来训练模型
*   `inference.py`:推理脚本

* * *

### [**配置**](#TOC)

在我们开始实现之前，让我们检查一下项目的配置。为此，我们将转到位于`pyimagesearch`目录中的`config.py`脚本。

```py
# define the dataset file
DATA_FNAME = "fra.txt"
```

在第 2 行的**上，我们定义了数据集文本文件。在我们的例子中，我们使用下载的`fra.txt`。**

```py
# define the batch size
BATCH_SIZE = 512
```

在第 5 行的**上，我们定义了数据集的批量大小。**

```py
# define the vocab size for the source and the target
# text vectorization layers
SOURCE_VOCAB_SIZE = 15_000
TARGET_VOCAB_SIZE = 15_000
```

在**的第 9 行和第 10 行**，我们定义了源和目标文本处理器的词汇量。这是让我们的文本矢量化层知道应该从所提供的数据集生成的词汇量所必需的。

```py
# define the maximum positions in the source and target dataset
MAX_POS_ENCODING = 2048
```

在**第 13** 行，我们定义了我们编码的最大长度。

```py
# define the number of layers for the encoder and the decoder
ENCODER_NUM_LAYERS = 6
DECODER_NUM_LAYERS = 6
```

在**第 16 行和第 17 行**，我们定义了变压器架构中编码器和解码器的层数。

```py
# define the dimensions of the model
D_MODEL = 512
```

变压器是一种各向同性的架构。这实质上意味着中间输出的维度在整个模型中不会改变。这需要定义一个静态模型维度。在**第 20 行**，我们定义整个模型的尺寸。

```py
# define the units of the point wise feed forward network
DFF = 2048
```

我们在**线 23** 上定义了点式前馈网络的中间尺寸。

```py
# define the number of heads and dropout rate
NUM_HEADS = 8
DROP_RATE = 0.1
```

多头关注层中的头数在**行 26** 中定义。辍学率在**第 27 行**指定。

```py
# define the number of epochs to train the transformer model
EPOCHS = 25
```

我们定义了在第 30 行的**上训练的时期数。**

```py
# define the output directory
OUTPUT_DIR = "output"
```

输出目录在**行 33** 上定义。

* * *

### [**数据集**](#TOC)

如前所述，我们需要一个包含源语言-目标语言句子对的数据集。为了配置和预处理这样的数据集，我们在`pyimagesearch`目录中准备了`dataset.py`脚本。

```py
# import the necessary packages
import random

import tensorflow as tf
import tensorflow_text as tf_text

# define a module level autotune
_AUTO = tf.data.AUTOTUNE
```

在第 8 行的**上，我们定义了模块级别`tf.data.AUTOTUNE`。**

```py
def load_data(fname):
    # open the file with utf-8 encoding
    with open(fname, "r", encoding="utf-8") as textFile:
        # the source and the target sentence is demarcated with tab,
        # iterate over each line and split the sentences to get
        # the individual source and target sentence pairs
        lines = textFile.readlines()
        pairs = [line.split("\t")[:-1] for line in lines]

        # randomly shuffle the pairs
        random.shuffle(pairs)

        # collect the source sentences and target sentences into
        # respective lists
        source = [src for src, _ in pairs]
        target = [trgt for _, trgt in pairs]

    # return the list of source and target sentences
    return (source, target)
```

在**第 11 行**上，我们定义了`load_data`函数，它从文本文件 fname 加载数据集。

接下来，在**第 13 行**，我们打开 utf-8 编码的文本文件，并使用`textFile`作为文件指针。

我们使用文件指针`textFile`从文件中读取行，如**第 17 行**所示。数据集中的源句子和目标句子用制表符分隔。在第 18 行的**上，我们迭代所有的源和目标句子对，用 split 方法将它们分开。**

在第 21 行的**上，我们随机打乱源和目标对，以调整数据管道。**

接下来，在第**行的第 25 行和第 26** 行，我们将源句子和目标句子收集到它们各自的列表中，稍后在第**行的第 29** 行返回。

```py
def splitting_dataset(source, target):
    # calculate the training and validation size
    trainSize = int(len(source) * 0.8)
    valSize = int(len(source) * 0.1)

    # split the inputs into train, val, and test
    (trainSource, trainTarget) = (source[:trainSize], target[:trainSize])
    (valSource, valTarget) = (
        source[trainSize : trainSize + valSize],
        target[trainSize : trainSize + valSize],
    )
    (testSource, testTarget) = (
        source[trainSize + valSize :],
        target[trainSize + valSize :],
    )

    # return the splits
    return (
        (trainSource, trainTarget),
        (valSource, valTarget),
        (testSource, testTarget),
    )
```

在**的第 32 行**，我们构建了`splitting_dataset`方法来将整个数据集分割成`train`、`validation`和`test`分割。

在**第 34 行和第 35 行**中，我们分别构建了 80%和 10%的列车大小和验证分割。

使用切片操作，我们将数据集分割成第 38-46 行上的各个分割。我们稍后返回第 49-53 行的**数据集分割。**

```py
def make_dataset(
    splits, batchSize, sourceTextProcessor, targetTextProcessor, train=False
):
    # build a TensorFlow dataset from the input and target
    (source, target) = splits
    dataset = tf.data.Dataset.from_tensor_slices((source, target))

    def prepare_batch(source, target):
        source = sourceTextProcessor(source)
        targetBuffer = targetTextProcessor(target)
        targetInput = targetBuffer[:, :-1]
        targetOutput = targetBuffer[:, 1:]
        return (source, targetInput), targetOutput

    # check if this is the training dataset, if so, shuffle, batch,
    # and prefetch it
    if train:
        dataset = (
            dataset.shuffle(dataset.cardinality().numpy())
            .batch(batchSize)
            .map(prepare_batch, _AUTO)
            .prefetch(_AUTO)
        )

    # otherwise, just batch the dataset
    else:
        dataset = dataset.batch(batchSize).map(prepare_batch, _AUTO).prefetch(_AUTO)

    # return the dataset
    return dataset
```

在**行 56** 上，我们构建了`make_dataset`函数，该函数为我们的训练管道构建了一个`tf.data.Dataset`。

在**第 60 行**，从提供的数据集分割中抓取源句子和目标句子。然后使用`tf.data.Dataset.from_tensor_slices()`函数将源和目标句子转换成`tf.data.Dataset`，如**行 61** 所示。

在**第 63-68 行**，我们定义了`prepare_batch`函数，它将作为`tf.data.Dataset`的映射函数。在**的第 64 行和第 65 行**，我们将源句子和目标句子分别传递到`sourceTextProcessor`和`targetTextProcessor`。`sourceTextProcessor`和`targetTextProcessor`为适配`tf.keras.layers.TextVectorization`层。这些层对字符串句子应用矢量化，并将它们转换为令牌 id。

在**行 66** 上，我们从开始到倒数第二个标记分割目标标记，作为目标输入。在**第 67 行**，我们从第二个令牌到最后一个令牌对目标令牌进行切片。这作为目标输出。右移一是为了在培训过程中实施教师强制。

在**第 68 行**，我们分别返回输入和目标。这里的输入是`source`和`targetInput`，而目标是`targetOuput`。这种格式适用于在培训时使用`model.fit()` API。

在**第 72-82 行**，我们构建数据集。在**第 85 行**，我们返回数据集。

```py
def tf_lower_and_split_punct(text):
    # split accented characters
    text = tf_text.normalize_utf8(text, "NFKD")
    text = tf.strings.lower(text)

    # keep space, a to z, and selected punctuations
    text = tf.strings.regex_replace(text, "[^ a-z.?!,]", "")

    # add spaces around punctuation
    text = tf.strings.regex_replace(text, "[.?!,]", r" \0 ")

    # strip whitespace and add [START] and [END] tokens
    text = tf.strings.strip(text)
    text = tf.strings.join(["[START]", text, "[END]"], separator=" ")

    # return the processed text
    return text
```

最后一个数据实用函数是`tf_lower_and_split_punct`，它接受任何一个句子作为参数(**第 88 行**)。我们从规范化句子开始，将它们转换成小写字母(分别是**第 90 行和第 91 行**)。

在第 94-97 行，我们去掉了句子中不必要的标点和字符。在**第 100 行**删除句子前的空格，然后在句子中添加开始和结束标记(**第 101 行**)。这些标记帮助模型理解何时开始或结束一个序列。

我们在第 104 行返回处理过的文本。

* * *

### [**注意**](#TOC)

在上一个教程中，我们学习了注意力的三种类型。总之，在构建变压器架构时，我们将注意以下三种类型:

*   [自我关注](https://pyimagesearch.com/2022/09/05/a-deep-dive-into-transformers-with-tensorflow-and-keras-part-1/#h4V5)
*   [交叉关注](https://pyimagesearch.com/2022/09/05/a-deep-dive-into-transformers-with-tensorflow-and-keras-part-1/#h4V4)
*   [多头自我关注](https://pyimagesearch.com/2022/09/05/a-deep-dive-into-transformers-with-tensorflow-and-keras-part-1/#h4V6)

我们在名为`attention.py`的`pyimagesearch`目录下的单个文件中构建这些不同类型的注意力。

```py
# import the necessary packages
import tensorflow as tf
from tensorflow.keras.layers import Add, Layer, LayerNormalization, MultiHeadAttention
```

在**第 2 行和第 3 行**，我们导入构建注意模块所需的必要包。

```py
class BaseAttention(Layer):
    """
    The base attention module. All the other attention modules will
    be subclassed from this module.
    """

    def __init__(self, **kwargs):
        # Note the use of kwargs here, it is used to initialize the
        # MultiHeadAttention layer for all the subclassed modules
        super().__init__()

        # initialize a multihead attention layer, layer normalization layer, and
        # an addition layer
        self.mha = MultiHeadAttention(**kwargs)
        self.layernorm = LayerNormalization()
        self.add = Add()
```

在第 6 行的**上，我们构建了父关注层，称为`BaseAttention`。所有其他具有特定任务的注意力模块都从该父层中被子类化。**

在**第 12 行**上，我们构建了层的构造器。在**第 15 行**，我们调用超级对象来构建图层。

在第 19-21 行、**、**上，我们初始化了一个`MultiHeadAttention`层、一个`LayerNormalization`层和一个`Add`层。这些是本教程后面指定的任何注意模块的基本层。

```py
class CrossAttention(BaseAttention):
    def call(self, x, context):
        # apply multihead attention to the query and the context inputs
        (attentionOutputs, attentionScores) = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True,
        )

        # store the attention scores that will be later visualized
        self.lastAttentionScores = attentionScores

        # apply residual connection and layer norm
        x = self.add([x, attentionOutputs])
        x = self.layernorm(x)

        # return the processed query
        return x
```

在第 24 行的**上，我们定义了`CrossAttention`层。该层是`BaseAttention`层的子类。这意味着该层已经有一个`MultiHeadAttention`、`LayerNormalization`和一个`Add`层。**

在第 25 行的**上，我们为该层构建`call`方法。该层接受`x`和`context`。在使用`CrossAttention`时，我们需要理解这里的`x`是查询，而`context`是稍后将构建键和值对的张量。**

在**第 27-32 行**，我们对输入应用了多头关注层。注意`query`、`key`和`value`术语是如何在**第 28-30 行**中使用的。

我们将注意力分数存储在第 35 行的**上。**

**第 38 行和第 39 行**是我们应用剩余连接和层标准化的地方。

我们在**线 42** 上返回处理后的输出。

```py
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        # apply self multihead attention
        attentionOutputs = self.mha(
            query=x,
            key=x,
            value=x,
        )

        # apply residual connection and layer norm
        x = self.add([x, attentionOutputs])
        x = self.layernorm(x)

        # return the processed query
        return x
```

我们在第 45 行的**上定义`GlobalSelfAttention`。**

**第 46 行**定义了该层的调用。这一层接受`x`。在**第 48-52 行**，我们将多头注意力应用于输入。请注意，`query`、`key`和`value`三个术语有相同的输入，`x`。这意味着我们在这一层使用多头自我关注。

在**行 55 和 56** 上，我们应用剩余连接和层标准化。处理后的输出在**线 59** 返回。

```py
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        # apply self multi head attention with causal masking (look-ahead-mask)
        attentionOutputs = self.mha(
            query=x,
            key=x,
            value=x,
            use_causal_mask=True,
        )

        # apply residual connection and layer norm
        x = self.add([x, attentionOutputs])
        x = self.layernorm(x)

        # return the processed query
        return x
```

我们在第 62 行的**上定义`CausalSelfAttention`。**

这一层类似于`GlobalSelfAttention`层，不同之处在于使用了因果蒙版。在**第 69 行**显示了因果掩码的用法。其他一切都保持不变。

* * *

### [**效用函数**](#TOC)

仅仅建立注意力模块是不够的。我们确实需要一些实用功能和模块来将所有东西缝合在一起。

我们需要的模块如下:

*   [位置编码](https://pyimagesearch.com/2022/09/26/a-deep-dive-into-transformers-with-tensorflow-and-keras-part-2/#h3Positional):由于我们知道自我关注层是排列不变的，我们需要某种方式将顺序信息注入到这些层中。在本节中，我们构建了一个嵌入层，它不仅负责标记的嵌入，还将位置信息注入到输入中。
*   [前馈网络](https://pyimagesearch.com/2022/09/26/a-deep-dive-into-transformers-with-tensorflow-and-keras-part-2/#h3FFN):变压器架构使用的前馈网络模块。
*   速率调度器:学习速率调度器，使架构学习得更好。
*   损失准确性:为了训练模型，我们需要建立屏蔽损失和准确性。损失将是目标函数，而准确度将是训练的度量。

* * *

#### [**位置编码**](#TOC)

如前一篇博文所示，为了构建位置编码，我们打开了`pyimagesearch`目录中的`positional_encoding.py`。

```py
# import the necessary packages
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Layer
```

从**第 2-4 行**，我们导入必要的包。

```py
def positional_encoding(length, depth):
    """
    Function to build the positional encoding as per the
    "Attention is all you need" paper.

    Args:
        length: The length of each sentence (target or source)
        depth: The depth of each token embedding
    """
    # divide the depth of the positional encoding into two for
    # sinusoidal and cosine embeddings
    depth = depth / 2

    # define the positions and depths as numpy arrays
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    # build the angle rates and radians
    angleRates = 1 / (10000**depths)
    angleRads = positions * angleRates

    # build the positional encoding, cast it to float32 and return it
    posEncoding = np.concatenate([np.sin(angleRads), np.cos(angleRads)], axis=-1)
    return tf.cast(posEncoding, dtype=tf.float32)
```

在第 7 行的**上，我们构建了`positional_encoding`函数。这个函数获取位置的长度和每次嵌入的深度。它计算由[瓦斯瓦尼等人](https://arxiv.org/abs/1706.03762)建议的位置编码。还可以看到编码的公式，如图**图 1** 所示。**

在**第 18 行**，我们将深度分成相等的两半，一部分用于正弦频率，另一部分用于余弦频率。从**第 21-26 行**，我们构建公式所需的`positions`、`depths`、`angleRates`和`angleRads`。

在**第 29 行**上，我们将正弦和余弦输出连接在一起，构建了完整的位置编码；`posEncoding`然后在**线 30** 返回。

```py
class PositionalEmbedding(Layer):
    def __init__(self, vocabSize, dModel, maximumPositionEncoding, **kwargs):
        """
        Args:
            vocabSize: The vocabulary size of the target or source dataset
            dModel: The dimension of the transformer model
            maximumPositionEncoding: The maximum length of a sentence in the dataset
        """
        super().__init__(**kwargs)

        # initialize an embedding layer
        self.embedding = Embedding(
            input_dim=vocabSize, output_dim=dModel, mask_zero=True
        )

        # initialize the positional encoding function
        self.posEncoding = positional_encoding(
            length=maximumPositionEncoding, depth=dModel
        )

        # define the dimensions of the model
        self.dModel = dModel

    def compute_mask(self, *args, **kwargs):
        # return the padding mask from the inputs
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        # get the length of the input sequence
        seqLen = tf.shape(x)[1]

        # embed the input and scale the embeddings
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dModel, tf.float32))

        # add the positional encoding with the scaled embeddings
        x += self.posEncoding[tf.newaxis, :seqLen, :]

        # return the encoded input
        return x
```

为我们模型中需要的定制层构建一个`tf.keras.layers.Layer`总是更好。`PositionalEmbedding`就是这样一层。我们在**第 33 行**定义自定义图层。

我们用一个`Embedding`和一个`positional_encoding`层初始化这个层，就像在**第 44-51 行**上所做的那样。我们还在**线 54** 上定义了模型的尺寸。

Keras 让我们为定制层公开一个`compute_mask`方法。我们在第 56 行定义了这个方法。有关填充和遮罩的更多信息，可以阅读官方 TensorFlow 指南。

`call`方法接受`x`作为它的输入(**行 60** )。输入首先被嵌入(**行 65** )，然后位置编码被添加到嵌入的输入(**行 69** )，最后在**行 72** 返回。

* * *

#### [**前馈**](#TOC)

为了构建前馈网络模块，如前一篇博文所示，我们打开了`pyimagesearch`目录中的`feed_forward.py`。

```py
# import the necessary packages
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Add, Dense, Dropout, Layer, LayerNormalization
```

在**行 2 和 3** 上，我们导入必要的包。

```py
class FeedForward(Layer):
    def __init__(self, dff, dModel, dropoutRate=0.1, **kwargs):
        """
        Args:
            dff: Intermediate dimension for the feed forward network
            dModel: The dimension of the transformer model
            dropOutRate: Rate for dropout layer
        """
        super().__init__(**kwargs)

        # initialize the sequential model of dense layers
        self.seq = Sequential(
            [
                Dense(units=dff, activation="relu"),
                Dense(units=dModel),
                Dropout(rate=dropoutRate),
            ]
        )

        # initialize the addition layer and layer normalization layer
        self.add = Add()
        self.layernorm = LayerNormalization()

    def call(self, x):
        # add the processed input and original input
        x = self.add([x, self.seq(x)])

        # apply layer norm on the residual and return it
        x = self.layernorm(x)
        return x
```

在**第 6 行**，我们定义了自定义图层`FeedForward`。该层由一个`tf.keras.Sequential`模块(**行 17-23** )、一个`Add`层(**行 26** )和一个`LayerNormalization`层(**行 27** )初始化。顺序模型有一个密集层和漏失层的堆栈。这就是我们进入变压器子层的前馈网络。

`call`方法(**第 29 行**)接受`x`作为其输入。输入通过顺序模型并与原始输入相加，作为**线 31** 上的剩余连接。处理后的子层输出然后通过**线 34** 上的`layernorm`层。

然后输出在**线 35** 上返回。

* * *

#### [**费率明细表**](#TOC)

为了构建学习率调度模块，我们打开了`pyimagesearch`目录中的`rate_schedule.py`文件。

```py
# import the necessary packages
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
```

在**第 2 行和第 3 行**，我们导入对费率表重要的必要包。

```py
class CustomSchedule(LearningRateSchedule):
    def __init__(self, dModel, warmupSteps=4000):
        super().__init__()

        # define the dmodel and warmup steps
        self.dModel = dModel
        self.dModel = tf.cast(self.dModel, tf.float32)
        self.warmupSteps = warmupSteps

    def __call__(self, step):
        # build the custom schedule logic
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmupSteps**-1.5)

        return tf.math.rsqrt(self.dModel) * tf.math.minimum(arg1, arg2)
```

在**第 6 行**上，我们构建了本文中实现的自定义`LearningRateSchedule`。我们把它命名为`CustomSchedule`(很有创意)。

在**第 7-13 行**，我们用必要的参数初始化模块。我们分别在**线 11** **和 13** 定义模型的尺寸和预热步骤的数量。

自定义时间表的逻辑如图**图 2** 所示。我们已经在 TensorFlow 的`__call__`方法中实现了相同的逻辑(来自**第 15-21 行**)。

* * *

#### [**损失准确度**](#TOC)

我们构建了在`pyimagesearch`目录下的`loss_accuracy.py`中定义指标的模块。

```py
# import the necessary packages
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
```

在**行 2 和 3** 上，我们导入必要的包。

```py
def masked_loss(label, prediction):
    # mask positions where the label is not equal to 0
    mask = label != 0

    # build the loss object and apply it to the labels
    lossObject = SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = lossObject(label, prediction)

    # mask the loss
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    # average the loss over the batch and return it
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss
```

在第 6 行的**上，我们构建了我们的`masked_loss`函数。它接受真实标签和来自我们模型的预测作为输入。**

我们首先在 8 号线的**上制作掩模。标签不等于 0 时，掩码无处不在。以`SparseCategoricalCrossentropy`作为我们的损失对象，我们计算不包括**线 11 和 12** 上的掩模的原始损失。**

然后将原始损耗与布尔掩码相乘，得到第 15 行**和第 16 行**的屏蔽损耗。在**第 19 行**上，我们对屏蔽损失进行平均，并在**第 20 行**上将其返还。

```py
def masked_accuracy(label, prediction):
    # mask positions where the label is not equal to 0
    mask = label != 0

    # get the argmax from the logits
    prediction = tf.argmax(prediction, axis=2)

    # cast the label into the prediction datatype
    label = tf.cast(label, dtype=prediction.dtype)

    # calculate the matches
    match = label == prediction
    match = match & mask

    # cast the match and masks
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    # average the match over the batch and return it
    match = tf.reduce_sum(match) / tf.reduce_sum(mask)
    return match
```

在第 23 行的**上，我们定义了自定义的`masked_accuracy`函数。这将是我们训练 transformer 模型时的自定义指标。**

在第 25 行的**上，我们构建布尔掩码。然后掩码在第 31** 行被转换为**预测的数据类型。**

**第 34 行和第 35 行**计算匹配(计算准确度所需的)，然后应用掩码以获得被屏蔽的匹配。

第 38 行和第 39 行用打字机打出火柴和面具。在**第 42 行**上，我们对屏蔽的匹配进行平均，并在**第 43 行**上将其返回。

* * *

### [**编码器**](#TOC)

在**图 3** 中，我们可以看到变压器架构中突出显示的编码器。如图**图 3** 所示，**编码器**是 N 个相同层的堆叠。每层由*两个*子层组成。

第一个是**多头自我关注机制**，第二个是简单的**位置式全连接前馈网络**。

[Vaswani 等人(2017)](https://arxiv.org/abs/1706.03762) 也在两个子层周围使用剩余连接和归一化操作。

我们在`pyimagesearch`目录中构建编码器模块，并将其命名为`encoder.py`。

```py
# import the necessary packages
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer

from .attention import GlobalSelfAttention
from .feed_forward import FeedForward
from .positional_encoding import PositionalEmbedding
```

在**第 2 行和第 7 行**，我们导入必要的包。

```py
class EncoderLayer(Layer):
    def __init__(self, dModel, numHeads, dff, dropOutRate=0.1, **kwargs):
        """
        Args:
            dModel: The dimension of the transformer module
            numHeads: Number of heads of the multi head attention module in the encoder layer
            dff: The intermediate dimension size in the feed forward network
            dropOutRate: The rate of dropout layer
        """
        super().__init__(**kwargs)

        # define the Global Self Attention layer
        self.globalSelfAttention = GlobalSelfAttention(
            num_heads=numHeads,
            key_dim=dModel // numHeads,
            dropout=dropOutRate,
        )

        # initialize the pointwise feed forward sublayer
        self.ffn = FeedForward(dff=dff, dModel=dModel, dropoutRate=dropOutRate)

    def call(self, x):
        # apply global self attention to the inputs
        x = self.globalSelfAttention(x)

        # apply feed forward network and return the outputs
        x = self.ffn(x)
        return x
```

编码器是编码器层的堆叠。在这里的**第 10 行**上，我们定义了保存两个子层的编码器层，即全局自关注(**第 22-26 行**)和前馈层(**第 29 行**)。

`call`的方法很简单。在**第 33 行**，我们将全局自我关注应用于编码器层的输入。在**线 36** 上，我们用逐点前馈网络处理相关输出。

编码器层的输出然后在**线 37** 上返回。

```py
class Encoder(Layer):
    def __init__(
        self,
        numLayers,
        dModel,
        numHeads,
        sourceVocabSize,
        maximumPositionEncoding,
        dff,
        dropOutRate=0.1,
        **kwargs
    ):
        """
        Args:
            numLayers: The number of encoder layers in the encoder
            dModel: The dimension of the transformer module
            numHeads: Number of heads of multihead attention layer in each encoder layer
            sourceVocabSize: The source vocabulary size
            maximumPositionEncoding: The maximum number of tokens in a sentence in the source dataset
            dff: The intermediate dimension of the feed forward network
            dropOutRate: The rate of dropout layer
        """
        super().__init__(**kwargs)

        # define the dimension of the model and the number of encoder layers
        self.dModel = dModel
        self.numLayers = numLayers

        # initialize the positional embedding layer
        self.positionalEmbedding = PositionalEmbedding(
            vocabSize=sourceVocabSize,
            dModel=dModel,
            maximumPositionEncoding=maximumPositionEncoding,
        )

        # define a stack of encoder layers
        self.encoderLayers = [
            EncoderLayer(
                dModel=dModel, dff=dff, numHeads=numHeads, dropOutRate=dropOutRate
            )
            for _ in range(numLayers)
        ]

        # initialize a dropout layer
        self.dropout = Dropout(rate=dropOutRate)

    def call(self, x):
        # apply positional embedding to the source token ids
        x = self.positionalEmbedding(x)

        # apply dropout to the embedded inputs
        x = self.dropout(x)

        # iterate over the stacks of encoder layer
        for encoderLayer in self.encoderLayers:
            x = encoderLayer(x=x)

        # return the output of the encoder
        return x
```

在第 40-51 行上，我们定义了我们的`Encoder`层。如上所述，编码器由一堆编码器层组成。为了使编码器自给自足，我们还在编码器内部添加了位置编码层。

在**行 65 和 66** 上，我们定义了编码器的尺寸和构建编码器的编码器层数。

**第 76-81 行**构建编码器层堆栈。在第 84 行的**处，我们初始化一个`Dropout`层来调整模型。**

该层的`call`方法接受`x`作为输入。首先，我们在输入上应用位置编码层，如**第 88 行**所示。然后嵌入被发送到**线 91** 上的脱落层。然后，处理后的输入在**行 94 和 95** 上的编码器层上迭代。然后，编码器的输出通过**线 98** 返回。

* * *

### [**解码器**](#TOC)

接下来，在**图 4** 中，我们可以看到变压器架构中突出显示的解码器。

除了每个编码器层中的两个子层之外，解码器还插入了第三个子层，该子层对编码器堆栈的输出执行多头关注。

解码器还具有残差连接和围绕三个子层的归一化操作。注意，解码器的第一个子层是一个**屏蔽的**多头关注层，而不是多头关注层。

我们在`pyimagesearch`内部构建解码器模块，并将其命名为`decoder.py`。

```py
# import the necessary packages
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Layer

from pyimagesearch.attention import CausalSelfAttention, CrossAttention

from .feed_forward import FeedForward
from .positional_encoding import PositionalEmbedding
```

在第 2-8 行上，我们导入必要的包。

```py
class DecoderLayer(Layer):
    def __init__(self, dModel, numHeads, dff, dropOutRate=0.1, **kwargs):
        """
        Args:
            dModel: The dimension of the transformer module
            numHeads: Number of heads of the multi head attention module in the encoder layer
            dff: The intermediate dimension size in the feed forward network
            dropOutRate: The rate of dropout layer
        """
        super().__init__(**kwargs)

        # initialize the causal attention module
        self.causalSelfAttention = CausalSelfAttention(
            num_heads=numHeads,
            key_dim=dModel // numHeads,
            dropout=dropOutRate,
        )

        # initialize the cross attention module
        self.crossAttention = CrossAttention(
            num_heads=numHeads,
            key_dim=dModel // numHeads,
            dropout=dropOutRate,
        )

        # initialize a feed forward network
        self.ffn = FeedForward(
            dff=dff,
            dModel=dModel,
            dropoutRate=dropOutRate,
        )

    def call(self, x, context):
        x = self.causalSelfAttention(x=x)
        x = self.crossAttention(x=x, context=context)

        # get the attention scores for plotting later
        self.lastAttentionScores = self.crossAttention.lastAttentionScores

        # apply feedforward network to the outputs and return it
        x = self.ffn(x)
        return x
```

解码器是单个解码器层的堆叠。在**第 11 行**，我们定义了自定义`DecoderLayer`。在**的第 23-27 行**，我们定义了`CausalSelfAttention`层。这一层是解码器层中的第一个子层。这为目标输入提供了因果掩蔽。

在**第 30-34 行**，我们定义了`CrossAttention`层。这将处理`CausalAttention`层的输出和`Encoder`输出。术语交叉来自解码器和编码器对此子层的输入。

在第 37-41 行上，我们定义了`FeedForward`层。

自定义层的`call`方法在**行 43** 定义。它接受`x`和`context`作为输入。在**行 44 和 45 上，**输入分别由因果层和交叉注意层处理。

注意力分数被缓存在**行 48** 上。之后，我们将前馈网络应用于**线 51** 上的处理输出。定制解码器层的输出然后在**线 52** 上返回。

```py
class Decoder(Layer):
    def __init__(
        self,
        numLayers,
        dModel,
        numHeads,
        targetVocabSize,
        maximumPositionEncoding,
        dff,
        dropOutRate=0.1,
        **kwargs
    ):
        """
        Args:
            numLayers: The number of encoder layers in the encoder
            dModel: The dimension of the transformer module
            numHeads: Number of heads of multihead attention layer in each encoder layer
            targetVocabSize: The target vocabulary size
            maximumPositionEncoding: The maximum number of tokens in a sentence in the source dataset
            dff: The intermediate dimension of the feed forward network
            dropOutRate: The rate of dropout layer
        """
        super().__init__(**kwargs)

        # define the dimension of the model and the number of decoder layers
        self.dModel = dModel
        self.numLayers = numLayers

        # initialize the positional embedding layer
        self.positionalEmbedding = PositionalEmbedding(
            vocabSize=targetVocabSize,
            dModel=dModel,
            maximumPositionEncoding=maximumPositionEncoding,
        )

        # define a stack of decoder layers
        self.decoderLayers = [
            DecoderLayer(
                dModel=dModel, dff=dff, numHeads=numHeads, dropOutRate=dropOutRate
            )
            for _ in range(numLayers)
        ]

        # initialize a dropout layer
        self.dropout = Dropout(rate=dropOutRate)

    def call(self, x, context):
        # apply positional embedding to the target token ids
        x = self.positionalEmbedding(x)

        # apply dropout to the embedded targets
        x = self.dropout(x)

        # iterate over the stacks of decoder layer
        for decoderLayer in self.decoderLayers:
            x = decoderLayer(x=x, context=context)

        # get the attention scores and cache it
        self.lastAttentionScores = self.decoderLayers[-1].lastAttentionScores

        # return the output of the decoder
        return x
```

我们在第 55-66 行的**上定义`Decoder`层。在**第 80 行和第 81 行**，我们定义了解码器模型的尺寸和解码器中使用的解码器层数。**

**第 84-88 行**定义了位置编码层。在**第 91-96 行**，我们为解码器定义了解码器层的堆栈。我们还定义了一个`Dropout`层在**线 99** 上。

`call`方法定义在**行 101** 上。它接受`x`和`context`作为输入。在**第 103** 行，我们首先将`x`令牌通过`positionalEmbedding`层进行嵌入。在**第 106** 行，我们将 dropout 应用于嵌入以调整模型。

我们迭代解码器层的堆栈，并将其应用于嵌入和上下文输入，如第**行 109 和 110** 所示。我们还缓存了第 113 行的**的最后关注分数。**

解码器的输出在**线 116** 上返回。

* * *

### [**变压器**](#TOC)

最后，所有的模块和组件都准备好构建整个 transformer 架构了。让我们看看图 5 中的**，我们可以看到整个架构。**

我们在`pyimagesearch`目录下的`transformer.py`中构建整个模块。

```py
# import the necessary packages
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Mean

from pyimagesearch.decoder import Decoder
from pyimagesearch.encoder import Encoder
```

**第 2-8 行**导入必要的包。

```py
class Transformer(Model):
    def __init__(
        self,
        encNumLayers,
        decNumLayers,
        dModel,
        numHeads,
        dff,
        sourceVocabSize,
        targetVocabSize,
        maximumPositionEncoding,
        dropOutRate=0.1,
        **kwargs
    ):
        """
        Args:
            encNumLayers: The number of encoder layers
            decNumLayers: The number of decoder layers
            dModel: The dimension of the transformer model
            numHeads: The number of multi head attention module for the encoder and decoder layers
            dff: The intermediate dimension of the feed forward network
            sourceVocabSize: The source vocabulary size
            targetVocabSize: The target vocabulary size
            maximumPositionEncoding: The maximum token length in the dataset
            dropOutRate: The rate of dropout layers
        """
        super().__init__(**kwargs)

        # initialize the encoder and the decoder layers
        self.encoder = Encoder(
            numLayers=encNumLayers,
            dModel=dModel,
            numHeads=numHeads,
            sourceVocabSize=sourceVocabSize,
            maximumPositionEncoding=maximumPositionEncoding,
            dff=dff,
            dropOutRate=dropOutRate,
        )
        self.decoder = Decoder(
            numLayers=decNumLayers,
            dModel=dModel,
            numHeads=numHeads,
            targetVocabSize=targetVocabSize,
            maximumPositionEncoding=maximumPositionEncoding,
            dff=dff,
            dropOutRate=dropOutRate,
        )

        # define the final layer of the transformer
        self.finalLayer = Dense(units=targetVocabSize)

    def call(self, inputs):
        # get the source and the target from the inputs
        (source, target) = inputs

        # get the encoded representation from the source inputs and the
        # decoded representation from the encoder outputs and target inputs
        encoderOutput = self.encoder(x=source)
        decoderOutput = self.decoder(x=target, context=encoderOutput)

        # apply a dense layer to the decoder output to formulate the logits
        logits = self.finalLayer(decoderOutput)

        # drop the keras mask, so it doesn't scale the losses/metrics.
        try:
            del logits._keras_mask
        except AttributeError:
            pass

        # return the final logits
        return logits
```

我们已经定义了`Decoder`和`Encoder`自定义层。是时候把所有东西放在一起，建立我们的变压器模型了。

注意我们是如何在第 11 行的**上定义一个名为 Transformer 的自定义`tf.keras.Model`。在**第 12-24 行**中提到了建造变压器所需的论据。**

从**第 40-57 行**，我们定义了编码器和解码器。在**第 60 行**，我们初始化计算逻辑的最终密集层。

模型的`call`方法在**行 62** 定义。输入是源令牌和目标令牌。我们先把 64 号线上的两个人分开。在第 68 行的**上，我们对源令牌应用编码器以获得编码器表示。接下来，在**行 69** 上，我们对目标令牌和编码器表示应用解码器。**

为了计算 logits，我们在解码器输出上应用最终的密集层，如第 72 行**所示。然后我们移除第 75-78** 行**上附加的 keras 遮罩。然后我们返回第 81 行**的**逻辑。**

* * *

### [**译者**](#TOC)

然而，我们还需要构建一些组件来训练和测试整个架构。第一个是翻译器模块，我们将需要它来执行神经机器翻译。

我们在`pyimagesearch`目录中构建翻译器模块，并将其命名为`translate.py`。

```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import StringLookup
```

在**行 1-3** 上，我们导入必要的包。

```py
class Translator(tf.Module):
    def __init__(
        self,
        sourceTextProcessor,
        targetTextProcessor,
        transformer,
        maxLength
    ):
        # initialize the source text processor
        self.sourceTextProcessor = sourceTextProcessor

        # initialize the target text processor and a string from
        # index string lookup layer for the target ids
        self.targetTextProcessor = targetTextProcessor
        self.targetStringFromIndex = StringLookup(
            vocabulary=targetTextProcessor.get_vocabulary(),
            mask_token="",
            invert=True
        )

        # initialize the pre-trained transformer model
        self.transformer = transformer

        self.maxLength = maxLength
```

`Transformer`模型，经过训练后，需要一个 API 来进行推断。我们需要一个定制的翻译器，它使用经过训练的 transformer 模型，并以人类可读的字符串给出结果。

在第 5 行的**上，我们定义了自定义的`tf.Module` names Translator，它将使用预先训练好的 Transformer 模型将源句子翻译成目标句子。在**第 14 行**，我们定义了源文本处理器。**

在第 18-23 行上，我们定义了目标文本处理器和一个字符串查找层。字符串查找对于从令牌 id 中获取字符串非常重要。

**第 26 行**定义了预应变变压器模型。**第 28 行**定义了翻译句子的最大长度。

```py
    def tokens_to_text(self, resultTokens):
        # decode the token from index to string
        resultTextTokens = self.targetStringFromIndex(resultTokens)

        # format the result text into a human readable format
        resultText = tf.strings.reduce_join(
            inputs=resultTextTokens, axis=1, separator=" "
        )
        resultText = tf.strings.strip(resultText)

        # return the result text
        return resultText
```

`tokens_to_text`方法是将令牌 id 转换成字符串所必需的。它接受`resultTokens`作为输入(**行 30** )。

在第 32 行的**上，我们将令牌从索引解码为字符串。这就是使用字符串查找层的地方。**第 35-38 行**负责连接字符串并去掉空格。这是将输出字符串转换成人类可读的句子所必需的。**

处理后的文本然后在**行 41** 上返回。

```py
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        # the input sentence is a string of source language
        # apply the source text processor on the list of source sentences
        sentence = self.sourceTextProcessor(sentence[tf.newaxis])

        encoderInput = sentence

        # apply the target text processor on an empty sentence
        # this will create the start and end tokens
        startEnd = self.targetTextProcessor([""])[0] # 0 index is to index the only batch

        # grab the start and end tokens individually
        startToken = startEnd[0][tf.newaxis]
        endToken = startEnd[1][tf.newaxis]

        # build the output array
        outputArray = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        outputArray = outputArray.write(index=0, value=startToken)

        # iterate over the maximum length and get the output ids
        for i in tf.range(self.maxLength):
            # transpose the output array stack
            output = tf.transpose(outputArray.stack())

            # get the predictions from the transformer and
            # grab the last predicted token
            predictions = self.transformer([encoderInput, output], training=False)
            predictions = predictions[:, -1:, :] # (bsz, 1, vocabSize)

            # get the predicted id from the predictions using argmax and
            # write the predicted id into the output array
            predictedId = tf.argmax(predictions, axis=-1)
            outputArray = outputArray.write(i+1, predictedId[0])

            # if the predicted id is the end token stop iteration
            if predictedId == endToken:
                break

        output = tf.transpose(outputArray.stack())
        text = self.tokens_to_text(output)

        return text
```

我们现在定义第 43 和 44 行的**翻译器的`__call__`方法。输入句子是一串源语言。我们在第 47 行**的**源句子列表上应用源文本处理器。**

编码器输入是符号化输入，如第 49 行**所示。在**的第 51-53 行**，我们在一个空句子上应用目标文本处理器，创建开始和结束标记。开始和结束标记在**线 56 和 57** 上被分开。**

我们在第 60 和 61 行的**上构建输出数组`tf.TensorArray`。我们现在迭代生成的令牌的最大数量，并从预训练的 Transformer 模型生成输出令牌 id(**第 64-80 行**)。在**行 66** 上，我们转置输出数组堆栈。在第 70 和 71 行**上，我们从转换器获得预测，并获取最后一个预测的令牌。

我们使用`tf.argmax`从预测中获得预测的 id，并将预测的 id 写入第 75 行**和第 76 行**的输出数组中。在**行 79 和 80** 上提供了停止迭代的条件。条件是预测标记应该与结束标记匹配。

然后，我们将`tokens_to_text`方法应用于输出数组，并在第**行第 82 行和第 83 行**获得字符串形式的结果文本。这个结果文本在第 85 行**返回。**

* * *

### [**训练**](#TOC)

我们组装所有部件来训练用于神经机器翻译任务的转换器架构。培训模块内置于`train.py`中。

```py
# USAGE
# python train.py

# setting seed for reproducibility
import sys
import tensorflow as tf

from pyimagesearch.loss_accuracy import masked_accuracy, masked_loss
from pyimagesearch.translate import Translator

tf.keras.utils.set_random_seed(42)
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.optimizers import Adam

from pyimagesearch import config
from pyimagesearch.dataset import (
    load_data,
    make_dataset,
    splitting_dataset,
    tf_lower_and_split_punct,
)
from pyimagesearch.rate_schedule import CustomSchedule
from pyimagesearch.transformer import Transformer
```

在第**行第 5-23** 行，我们定义了导入并为可重复性设置了随机种子。

```py
# load data from disk
print(f"[INFO] loading data from {config.DATA_FNAME}...")
(source, target) = load_data(fname=config.DATA_FNAME)
```

**第 26 行和第 27 行**使用`load_data`方法加载数据。

```py
# split the data into training, validation, and test set
print("[INFO] splitting the dataset into train, val, and test...")
(train, val, test) = splitting_dataset(source=source, target=target
```

一个数据集需要拆分成`train`、`val`和`test`。**第 30 行和第 31 行**正好有助于此。数据集被发送到`splitting_dataset`函数，该函数将其分割成相应的数据片段。

```py
# create source text processing layer and adapt on the training
# source sentences
print("[INFO] adapting the source text processor on the source dataset...")
sourceTextProcessor = TextVectorization(
    standardize=tf_lower_and_split_punct, max_tokens=config.SOURCE_VOCAB_SIZE
)
sourceTextProcessor.adapt(train[0])
```

**第 35-39 行**创建源文本处理器，一个`TextVectorization`层，并在源训练数据集上修改它。

```py
# create target text processing layer and adapt on the training
# target sentences
print("[INFO] adapting the target text processor on the target dataset...")
targetTextProcessor = TextVectorization(
    standardize=tf_lower_and_split_punct, max_tokens=config.TARGET_VOCAB_SIZE
)
targetTextProcessor.adapt(train[1])
```

**第 43-47 行**创建目标文本处理器，一个`TextVectorization`层，并适应目标训练数据集。

```py
# build the TensorFlow data datasets of the respective data splits
print("[INFO] building TensorFlow Data input pipeline...")
trainDs = make_dataset(
    splits=train,
    batchSize=config.BATCH_SIZE,
    train=True,
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor,
)
valDs = make_dataset(
    splits=val,
    batchSize=config.BATCH_SIZE,
    train=False,
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor,
)
testDs = make_dataset(
    splits=test,
    batchSize=config.BATCH_SIZE,
    train=False,
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor,
)
```

**第 50-71 行**使用`make_dataset`函数构建`tf.data.Dataset`管道。

```py
# build the transformer model
print("[INFO] building the transformer model...")
transformerModel = Transformer(
    encNumLayers=config.ENCODER_NUM_LAYERS,
    decNumLayers=config.DECODER_NUM_LAYERS,
    dModel=config.D_MODEL,
    numHeads=config.NUM_HEADS,
    dff=config.DFF,
    sourceVocabSize=config.SOURCE_VOCAB_SIZE,
    targetVocabSize=config.TARGET_VOCAB_SIZE,
    maximumPositionEncoding=config.MAX_POS_ENCODING,
    dropOutRate=config.DROP_RATE,
)
```

我们在第 74-85 行上构建我们的变压器模型。

```py
# compile the model
print("[INFO] compiling the transformer model...")
learningRate = CustomSchedule(dModel=config.D_MODEL)
optimizer = Adam(learning_rate=learningRate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
transformerModel.compile(
    loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy]
)
```

我们用自定义优化器编译模型，在第 88-93 行的上有`CustomSchedule`和`masked_loss`以及`masked_accuracy`函数。

```py
# fit the model on the training dataset
transformerModel.fit(
    trainDs,
    epochs=config.EPOCHS,
    validation_data=valDs,
)
```

使用`trainDs`我们在**行 96-100** 上拟合模型。这里我们使用 Keras 提供的高效优雅的`Model.fit` API。我们还通过向 fit 方法提供`valDs`来验证培训渠道。

```py
# infer on a sentence
translator = Translator(
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor,
    transformer=transformerModel,
    maxLength=50,
)

# serialize and save the translator
print("[INFO] serialize the inference translator to disk...")
tf.saved_model.save(
    obj=translator,
    export_dir="translator",
)
```

我们构建`Translator`进行推理，并在**的第 103-115 行**将其保存到磁盘。

以下是仅 25 个时期的训练脚本的输出。

```py
$ python train.py
[INFO] loading data from fra.txt...
[INFO] splitting the dataset into train, val, and test...
[INFO] adapting the source text processor on the source dataset...
[INFO] adapting the target text processor on the target dataset...
[INFO] building TensorFlow Data input pipeline...
[INFO] building the transformer model...
[INFO] compiling the transformer model...
Epoch 1/25
309/309 [==============================] - 85s 207ms/step - loss: 7.1164 - masked_accuracy: 0.2238 - val_loss: 4.8327 - val_masked_accuracy: 0.3452
Epoch 2/25
309/309 [==============================] - 61s 197ms/step - loss: 3.9636 - masked_accuracy: 0.4155 - val_loss: 3.0660 - val_masked_accuracy: 0.5020
.
.
.
Epoch 24/25
309/309 [==============================] - 61s 195ms/step - loss: 0.2388 - masked_accuracy: 0.9185 - val_loss: 1.0194 - val_masked_accuracy: 0.8032
Epoch 25/25
309/309 [==============================] - 61s 195ms/step - loss: 0.2276 - masked_accuracy: 0.9217 - val_loss: 1.0323 - val_masked_accuracy: 0.8036
[INFO] serialize the inference translator to disk...
```

* * *

### [**推论**](#TOC)

现在是有趣的部分。我们将测试我们的转换器执行机器翻译任务的能力。我们在`inference.py`中构建推理脚本。

```py
# USAGE
# python inference.py -s "input sentence"
```

我们在第 1 行**和第 2 行**定义推理脚本的用法。

```py
# import the necessary packages
import tensorflow_text as tf_text # this is a no op import important for op registry
import tensorflow as tf
import argparse
```

我们在第 5-7 行导入必要的包。

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sentence", required=True,
	help="input english sentence")
args = vars(ap.parse_args())
```

我们构造参数 parse 并解析第 10-13 行的参数。

```py
# convert the input english sentence to a constant tensor
sourceText = tf.constant(args["sentence"])
```

**第 16 行**将输入的源句子转换成`tf.Tensor`。这对翻译器来说很重要，因为它接受`tf.Tensor`而不是字符串。

```py
# load the translator model from disk
print("[INFO] loading the translator model from disk...")
translator = tf.saved_model.load("translator")
```

我们现在从磁盘的第 19 行和第 20 行加载保存的翻译模块。

```py
# perform inference and display the result
print("[INFO] translating english sentence to french...")
result = translator(sentence=sourceText)

translatedText = result.numpy()[0].decode()
print("[INFO] english sentence: {}".format(args["sentence"]))
print("[INFO] french translation: {}".format(translatedText))
```

在**第 23-28 行**，我们在翻译模块上执行推理，并在终端上显示结果。

以下输出显示了将英语句子翻译成法语的推理。

```py
$ python inference.py -s "i am hungry, let's get some food"

[INFO] loading the translator model from disk...
[INFO] translating english sentence to french...
[INFO] english sentence: i am hungry, let's get some food
[INFO] french translation: [START] jai faim , allons chercher de la nourriture . [END]
```

你可以通过这里的拥抱界面直接看到模特并与之互动:

 <gradio-app space="pyimagesearch/nmt-transformer">* * *

* * *

## [**汇总**](#TOC)

Transformer 博客帖子是 PyImageSearch 多个系列的高潮。我们从字母和单词(记号)开始，然后构建这些记号的表示。我们还使用这些表示来寻找记号之间的相似性，并将它们嵌入到高维空间中。

相同的嵌入还被传递到可以处理顺序数据的顺序模型(rnn)中。这些模型被用来构建语境，并巧妙地处理输入句子中对翻译输出句子有用的部分。这整个叙述跨越了多个博客帖子，我们非常感谢与我们一起踏上这一旅程的读者。

但是正如他们所说的，“每一个结束都是一个新的开始”，虽然 Transformer 架构和应用程序到 NLP 的旅程到此结束，但我们仍然有一些迫切的问题。

*   如何将此应用于图像？
*   我们如何扩展它？
*   我们能为各种形态制造变形金刚吗？

现在，这些问题需要自己的博文，有的需要自己的系列！那么，请告诉我们您希望我们接下来讨论的主题:

发推文 [@pyimagesearch](https://twitter.com/pyimagesearch) 或发电子邮件【ask.me@pyimagesearch.com 

* * *

### [**参考文献**](#TOC)

我们在整个系列中使用了以下参考资料:

*   [用变压器和 Keras 进行神经机器翻译](https://www.tensorflow.org/text/tutorials/transformer)
*   [第 13 讲:注意](https://www.youtube.com/watch?v=YAgjfMR9R_M)
*   [瓦斯瓦尼等人，2017，《注意力是你所需要的一切》](https://arxiv.org/pdf/1706.03762.pdf)
*   [马尼姆社区](https://www.manim.community/)

* * *

### [**引用信息**](#TOC)

A. R. Gosthipaty 和 R. Raha。“用 TensorFlow 和 Keras 深入研究变形金刚:第三部分”， *PyImageSearch* ，P. Chugh，S. Huot，K. Kidriavsteva，A. Thanki 编辑。，2022 年，【https://pyimg.co/9nozd 

```py
@incollection{ARG-RR_2022_DDTFK3,
  author = {Aritra Roy Gosthipaty and Ritwik Raha},
  title = {A Deep Dive into Transformers with {TensorFlow} and {K}eras: Part 3},
  booktitle = {PyImageSearch},
  editor = {Puneet Chugh and Susan Huot and Kseniia Kidriavsteva and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/9nozd},
}
```

* * *

* * *

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***</gradio-app>*</gradio-app>