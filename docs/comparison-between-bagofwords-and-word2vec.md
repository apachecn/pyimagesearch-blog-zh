# BagofWords 和 Word2Vec 的比较

> 原文：<https://pyimagesearch.com/2022/07/18/comparison-between-bagofwords-and-word2vec/>

* * *

## **目录**

* * *

## [**BagofWords 和 Word2Vec** 的比较](#TOC)

在过去的几周里，我们学习了重要的自然语言处理(NLP)技术，如单词袋和 Word2Vec。在某种形式上，两者都是 NLP 中表征学习的一部分。

总的来说，以一种让计算机理解文本的方式来表示特征确实有助于 NLP 的发展。但是上面提到的两种技术完全不同。这就引出了一个问题，是什么让我们选择一个而不是另一个？

在本教程中，您将对单词包和 Word2Vec 进行比较。

本课是关于 NLP 101 的 4 部分系列的最后一课:

1.  [*自然语言处理入门*](https://pyimg.co/60xld)
2.  *[*介绍词袋(BoW)模型*](https://pyimg.co/oa2kt)*
3.  *[*word 2 vec:NLP 中的嵌入研究*](https://pyimg.co/2fb0z)*
4.  *[***BagofWords 与 Word2Vec***](https://pyimg.co/txq23) **(今日教程)***

 *要知道单词袋和单词 2Vec， ***的区别，只要坚持阅读。***

* * *

## [**BagofWords 和 Word2Vec** 的比较](#TOC)

让我们简要回顾一下什么是 NLP 中的表征学习。向计算机教授文本数据是极其困难和复杂的。在本系列的第一篇博客文章中，我们回顾了自然语言处理的简史。

在那里，我们确定了在 NLP 中引入统计和表征学习是如何将 NLP 的总体进展向更积极的方向改变的。我们学习了单词袋(BOW)，这是一种源于表征学习的技术。接下来是更复杂、更全面的 Word2Vec 方法。

这两种技术都涉及到将我们的输入数据表达到一个表示(嵌入)空间中。我们发现的关联越多，我们对模型学习效果的验证就越多。

让我们更上一层楼，更深入地探究为什么这些技术既相似又截然不同。

* * *

### [**简述弓和字 2 vec**](#TOC)

单词包架构包括将每个输入句子转换成单词包。看一下**图一**。

这里的嵌入矩阵的列数等于总词汇表中的单词数。每个句子被表示为每个单词出现或不出现的组合。

例如，如果给定数据集的词汇大小为 300，大小为 5 的输入句子现在将变成大小为 300 的向量，其中 5 个出现的单词的位被打开，而 295 位被关闭。

Word2Vec 采用不同的方法来利用向量。在这里，我们考虑每个单词，而不是每个句子都被表示为实体。我们选择一个有限维的嵌入空间，其中每行代表词汇表中的一个单词。

在整个训练过程中，每个单词在每个维度上都有一定的值(或权重),代表其矢量形式。这些权重由每个单词的上下文(即相邻单词)确定。

因此，句子“天空是蓝色的。”和“蓝天很美。”意味着蓝色这个词会和我们嵌入空间的天空联系在一起。

这两种方法都很巧妙，并且各有千秋。但是让我们进一步仔细检查每个算法。

* * *

### [**配置您的开发环境**](#TOC)

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
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

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南** ***预配置*** **在您的网络浏览器中运行 Google Colab 的生态系统！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

* * *

### [**项目结构**](#TOC)

我们首先需要回顾我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

从这里，看一下目录结构:

```py
!tree .
.
├── datadf.csv
├── LICENSE
├── outputs
│   ├── loss_BOW.png
│   ├── loss_W2V.png
│   ├── terminal_outputs.txt
│   ├── TSNE_BOW.png
│   └── TSNE_W2V.png
├── pyimagesearch
│   ├── BOWmodel.py
│   ├── config.py
│   ├── data_processing.py
│   └── __init__.py
├── README.md
├── train_BOW.py
└── train_Word2Vec.py

2 directories, 14 files
```

我们有两个子目录:`outputs`和`pyimagesearch`。

在`outputs`目录中，我们有这个项目的所有结果和可视化。

在`pyimagesearch`目录中，我们有:

*   `BOWmodel.py`:包含了词袋的模型架构。
*   `config.py`:包含整个配置管道。
*   `data_processing.py`:包含几个数据处理工具的脚本。
*   `__init__.py`:使`pyimagesearch`目录的行为像一个 python 包。

在主目录中，我们有:

*   `train_BOW.py`:词袋架构训练脚本。
*   `train_Word2Vec.py`:word 2 vec 架构的训练脚本。
*   `datadf.csv`:我们项目的训练数据。

* * *

### [**配置先决条件**](#TOC)

在`pyimagesearch`目录中，`config.py`脚本包含了我们项目的整个配置管道。

```py
# import the necessary packages
import os

# define Bag-of-Words parameters
EPOCHS = 30

# define the Word2Vec parameters
EMBEDDING_SIZE = 2
ITERATIONS = 1000

# define the path to the output directory
OUTPUT_PATH = "outputs"

# define the path to the Bag-of-Words output
BOW_LOSS = os.path.join(OUTPUT_PATH, "loss_BOW")
BOW_TSNE = os.path.join(OUTPUT_PATH, "TSNE_BOW")

# define the path to the Word2vec output
W2V_LOSS = os.path.join(OUTPUT_PATH, "loss_W2V")
W2V_TSNE = os.path.join(OUTPUT_PATH, "TSNE_W2V")
```

在第 5 行的**上，我们定义了单词袋模型将被训练的时期的数量。**

在**第 8 行和第 9 行**，我们为 Word2Vec 模型定义参数，即 Word2Vec 模型将训练的嵌入维数和迭代次数。

接下来，定义`outputs`目录(**第 12 行**)，接着是损失和 TSNE 图的单独定义(**第 15-20 行**)。

* * *

### [**处理数据**](#TOC)

我们将继续讨论数据处理脚本`data_processing.py`。这个脚本包含了帮助我们管理数据的函数。

```py
# import the necessary packages
import re
import tensorflow as tf

def preprocess(sentDf, stopWords, key="sentence"):
	# loop over all the sentences
	for num in range(len(sentDf[key])):
		# strip the sentences off the stop-words
		newSent = ""
		for word in sentDf["sentence"].iloc[num].split():
			if word not in stopWords:
				newSent = newSent + " " + word

	# update the sentences
	sentDf["sentence"].iloc[num] = newSent

	# return the preprocessed data
	return(sentDf)
```

在第 5 行的**上，我们有第一个函数`preprocess`，它接受以下参数:**

*   `sentDf`:输入数据帧。
*   要从我们的数据集中省略的单词列表。
*   `key`:默认设置为`sentence`。它将用于访问数据帧的右列。

循环第 7 行**的句子，我们首先初始化一个空字符串来存储我们在第 9** 行**处理过的数据。现在，句子中的每个单词都被迭代(**第 10 行**)，停用词被省略。**

我们用第 15 行**上的新句子(没有停用词)更新数据帧。**

```py
def prepare_tokenizerBOW(df, topWords, sentKey="sentence", outputKey="sentiment"):
	# prepare separate tokenizers for the data and labels
	tokenizerData = tf.keras.preprocessing.text.Tokenizer(num_words=topWords,
			oov_token="<unk>",
			filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')
	tokenizerLabels = tf.keras.preprocessing.text.Tokenizer(num_words=5,
			oov_token="<unk>",
			filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')

	# fit the tokenizers on their respective data
	tokenizerData.fit_on_texts(df["sentence"])
	tokenizerLabels.fit_on_texts(df["sentiment"])

	# return the tokenizers
	return (tokenizerData, tokenizerLabels)
```

我们的下一个函数是第 20 行的**上的`prepare_tokenizerBOW`，它接受以下参数:**

*   `df`:输入数据帧去掉了停止字。
*   `topWords`:初始化 tensorflow 标记器所需的参数。
*   `sentKey`:从数据框中进入句子的键。
*   `outputKey`:从数据框中访问标签的键。

这个函数专门用于单词包体系结构，其中我们将为数据及其标签使用两个独立的标记化器。相应地，我们创建了两个记号赋予器，并把它们放在各自的文本中(**第 22-31 行**)。

```py
def prepare_tokenizerW2V(df, topWords, sentKey="sentence", outputKey="sentiment"):
	# prepare tokenizer for the Word2Vec data
	tokenizerWord2Vec = tf.keras.preprocessing.text.Tokenizer(num_words=topWords,
			oov_token="<unk>",
			filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~')

	# fit the tokenizer on the data
	tokenizerWord2Vec.fit_on_texts(df["sentence"])
	tokenizerWord2Vec.fit_on_texts(df["sentiment"])

	# return the tokenizer
	return (tokenizerWord2Vec)
```

这个脚本中的最后一个函数是第 36 行的**上的`prepare_tokenizerW2V`，它接受以下参数:**

*   `df`:输入数据帧去掉了停止字。
*   `topWords`:初始化 tensorflow 标记器所需的参数。
*   `sentKey`:从数据框中进入句子的键。
*   `outputKey`:从数据框中访问标签的键。

在第**行第 38-40** 行，我们已经为 Word2Vec 方法初始化了一个单独的标记化器，并将其安装到第**行第 43 和 44** 行的数据和标签上。因为这两种方法不同，所以我们使用单一的记号赋予器。

* * *

### [**创建词袋模型**](#TOC)

接下来，我们将定义单词袋模型的架构。让我们进入`BOWmodel.py`脚本。

```py
#import the necessary packages
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy

def build_shallow_net(inputDims, numClasses):
	# define the model
	model = Sequential()
	model.add(Dense(512, input_dim=inputDims, activation="relu"))
	model.add(Dense(128, activation="relu"))
	model.add(Dense(numClasses, activation="softmax"))

	# compile the keras model
	model.compile(loss=sparse_categorical_crossentropy, 
			optimizer="adam",
			metrics=["accuracy"]
	)

	# return model
	return model
```

在第 7 行的**上，我们有`build_shallow_Net`，它接受以下参数:**

*   `inputDims`:输入的维度等于词汇表中的字数。
*   `numClasses`:输出类的数量。

在**第 9-12 行**，我们定义了一个由两个密集层和最后一个`softmax`密集层组成的序列模型。因为我们处理的是小数据，所以像这样的简单模型就可以了。

在**的第 15-17 行**，我们用`sparse_categorical_crossentropy`损失和`adam`优化器编译模型，准确性作为我们的度量。

* * *

### [**训练词袋模型**](#TOC)

为了训练单词包架构，我们将进入`train_BOW.py`脚本。

```py
# USAGE
# python -W ignore train_BOW.py

# set seed for reproducibility
import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from pyimagesearch import config
from pyimagesearch.data_processing import preprocess
from pyimagesearch.BOWmodel import build_shallow_net
from pyimagesearch.data_processing import prepare_tokenizerBOW
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
import os

# prepare stop-words using the NLTK package
nltk.download("stopwords")
stopWords = nltk.corpus.stopwords.words("english")

# initialize the dataframe from csv format
dataDf = pd.read_csv("datadf.csv")

# preprocess the dataframe 
processedDf = preprocess(dataDf, stopWords)
```

这个脚本的第一步是创建`stopWords`列表。为此，我们将借助`nltk`包(**线 22 和 23** )。接下来，我们使用`csv`格式的输入文件初始化数据帧(**第 26 行**)。随后使用`preprocess`功能从输入句子中删除停用词(**第 29 行**)。

```py
# store the number of classification heads
numClasses = len(processedDf["sentiment"].unique())

# create the tokenizers for data and labels
(tokenizerData, tokenizerLabels) = prepare_tokenizerBOW(processedDf, topWords=106)

# create integer sequences of the data using tokenizer
trainSeqs = tokenizerData.texts_to_sequences(processedDf["sentence"])
trainLabels = tokenizerLabels.texts_to_sequences(processedDf["sentiment"])

# create the Bag-of-Words feature representation
encodedDocs = tokenizerData.texts_to_matrix(processedDf["sentence"].values, 
		mode="count"
)

# adjust the train label indices for training
trainLabels = np.array(trainLabels)
for num in range(len(trainLabels)):
	trainLabels[num] = trainLabels[num] - 1

# initialize the model for training
BOWModel = build_shallow_net(inputDims = tokenizerData.num_words-1,
		numClasses=numClasses
)

# fit the data into the model and store training details
history = BOWModel.fit(encodedDocs[:, 1:], 
		trainLabels.astype('float32'), 
		epochs=config.EPOCHS
)
```

在**线 32** 上，存储输出类别的数量。接下来，使用**行 35** 上的`prepare_tokenizerBOW`函数获得数据和标签的标记符。

现在我们可以使用第 38 行和第 39 行上的`texts_to_sequences`函数将我们的单词转换成整数序列。

使用`texts_to_matrix`函数，通过将`mode`参数设置为`count` ( **第 42-44 行**)，我们将输入文本转换为单词包表示。这将计算一个单词在一个句子中出现的次数，给我们句子和每个单词出现的向量表示。

在第**行第 47-49** 行，我们调整标签的索引用于训练。单词袋模型被初始化(**第 52-54 行**，并且该模型相应地在输入数据上被训练(**第 57-60 行**)。由于记号赋予器创建添加了未知单词记号作为它的第一个条目，我们已经考虑了除了从第 1 个索引开始而不是从第 0 个索引开始的所有单词。

```py
# create output directory if it doesn't already exist
if not os.path.exists(config.OUTPUT_PATH):
	os.makedirs(config.OUTPUT_PATH)

# plot the loss for BOW model
print("[INFO] Plotting loss...")
plt.plot(history.history["loss"])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(config.BOW_LOSS)

# get the weights for the first model layer
representationsBOW = BOWModel.get_weights()[0]

# apply dimensional reduction using TSNE
tsneEmbed = (TSNE(n_components=2)
		.fit_transform(representationsBOW)
)

# initialize a index counter 
indexCount = 1 

# initialize the tsne figure
plt.figure(figsize=(25, 5))

# loop over the tsne embeddings and plot the corresponding words
print("[INFO] Plotting TSNE embeddings...")
for (word, embedding) in tsneEmbed[:100]:
	plt.scatter(word, embedding)
	plt.annotate(tokenizerData.index_word[indexCount], (word, embedding))
	indexCount += 1
plt.savefig(config.BOW_TSNE)
```

在第 63 和 64 行上，我们创建了输出文件夹，如果它还不存在的话。

在**第 67-71 行**上，我们借助模型历史变量绘制了模型损失。

现在，我们要绘制单词袋表示空间。请注意，模型的第一层的输入维度等于单词数。如果我们假设每一列对应于数据集中的每个单词，则该层的权重可以被认为是我们的嵌入空间。

因此，在第 74 行**处抓取该层的权重，并应用 TSNE 嵌入进行降维(**行 77-79** )。我们继续为每个用于推理的单词绘制 TSNE 图。**

* * *

### [**训练 Word2Vec 模型**](#TOC)

现在我们将继续讨论`Word2Vec`模型。为了训练它，我们必须执行`train_Word2Vec.py`。

```py
# USAGE
# python -W ignore train_Word2Vec.py

# set seed for reproducibility
import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from pyimagesearch import config
from pyimagesearch.data_processing import preprocess
from pyimagesearch.data_processing import prepare_tokenizerW2V
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
import os

# prepare stop-words using the NLTK package
nltk.download("stopwords")
stopWords = nltk.corpus.stopwords.words("english")

# initialize the dataframe from csv format
dataDf = pd.read_csv("datadf.csv")

# preprocess the dataframe 
processedDf = preprocess(dataDf, stopWords)
```

正如单词包脚本中所做的，该脚本的第一步是创建`stopWords`列表。为此，我们将借助`nltk`包(**线 22 和 23** )。接下来，我们使用`csv`格式的输入文件初始化数据帧(**第 26 行**)。随后使用`preprocess`功能从输入句子中删除停用词(**第 29 行**)。

```py
# store the number of classification heads
numClasses = len(processedDf["sentiment"].unique())

# create the tokenizers for data and labels
(tokenizerData) = prepare_tokenizerW2V(processedDf, topWords=200)

# create integer sequences of the data using tokenizer
trainSeqs = tokenizerData.texts_to_sequences(processedDf["sentence"])
trainLabels = tokenizerData.texts_to_sequences(processedDf["sentiment"])

# create the representational matrices as variable tensors
contextVectorMatrix =  tf.Variable(
	np.random.rand(200, config.EMBEDDING_SIZE)
)
centerVectorMatrix = tf.Variable(
	np.random.rand(200, config.EMBEDDING_SIZE)
)

# initialize the optimizer and create an empty list to log the loss
optimizer = tf.optimizers.Adam()
lossList = list()
```

在**第 32 行**上，我们存储输出类的数量。接下来，在**行 35** 上创建覆盖数据和标签的单个记号赋予器。

使用记号赋予器的`texts_to_sequences`函数将单词序列转换成整数序列(**第 38 行和第 39 行**)。

对于 Word2Vec 架构，我们然后在第 42-47 行初始化上下文和中心单词矩阵。随后是`Adam`优化器和一个空的丢失列表初始化(**行 50 和 51** )。

```py
# loop over the training epochs
print("[INFO] Starting Word2Vec training...")
for iter in tqdm(range(config.ITERATIONS)):
	# initialize the loss per epoch
	lossPerEpoch = 0

	# loop over the indexes and labels
	for idxs,trgt in zip(trainSeqs, trainLabels):
		# convert label into integer
		trgt = trgt[0]

		# initialize the gradient tape
		with tf.GradientTape() as tape:	
			# initialize the combined context vector
			combinedContext = 0

			# update the combined context with each index
			for count,index in enumerate(idxs):
				combinedContext += contextVectorMatrix[index,:]

			# standardize the vector
			combinedContext /= len(idxs)

			# matrix multiply the center vector matrix 
			# with the combined context
			output = tf.matmul(centerVectorMatrix, 
				tf.expand_dims(combinedContext ,1))

			# calculate the softmax output and
			# grab the relevant index for loss calculation
			softOut = tf.nn.softmax(output, axis=0)
			loss = softOut[trgt]
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

在第**行第 55** 上，我们开始迭代训练时期，并初始化第**行第 57** 上的一个`lossPerEpoch`变量。

接下来，我们循环训练序列和标签的索引和标签(**行 60** )，并且首先将标签列表转换成单个变量(**行 62** )。

我们在**行 65** 初始化一个梯度带。句子索引用于从上下文矩阵中提取上下文向量，并且输出被加在一起，随后进行归一化(**第 67-74 行**)。

组合的上下文向量乘以中心向量矩阵，结果通过一个`softmax`函数(**第 78-83 行**)。抓取相关的中心词索引进行损失计算，并计算索引的负对数(**第 84 行和第 85 行**)。

一旦退出循环，`lossPerEpoch`变量就会被更新。梯度被应用于两个嵌入矩阵(**第 89-95 行**)。

最后，一旦一个时期结束，`lossPerEpoch`变量被添加到损失列表中(**第 98 行**)。

```py
# create output directory if it doesn't already exist
if not os.path.exists(config.OUTPUT_PATH):
	os.makedirs(config.OUTPUT_PATH)

# plot the loss for evaluation
print("[INFO] Plotting Loss...")
plt.plot(lossList)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(config.W2V_LOSS)

# apply dimensional reductionality using tsne for the
# representation matrices
tsneEmbed = (
	TSNE(n_components=2)
	.fit_transform(contextVectorMatrix.numpy())
)

# initialize a index counter
indexCount = 1 

# initialize the tsne figure
plt.figure(figsize=(25, 5))

# loop over the tsne embeddings and plot the corresponding words
print("[INFO] Plotting TSNE Embeddings...")
for (word, embedding) in tsneEmbed[:100]:
	if indexCount != 108:
		plt.scatter(word, embedding)
		plt.annotate(tokenizerData.index_word[indexCount], (word, embedding))
		indexCount += 1
plt.savefig(config.W2V_TSNE)
```

在**第 101 行和第 102 行**，如果输出文件夹还不存在，我们创建它。

在**第 105-109 行**上，我们绘制了 Word2Vec 模型的损耗。随后从嵌入矩阵创建 TSNE 图(**第 113-131 行**)。将对应于其指数的单词绘制在 TSNE 图上，以评估所形成的关联。

* * *

### [**训练结果和可视化**](#TOC)

让我们看看这两种体系结构的培训进展如何。

```py
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
Epoch 1/30
1/1 [==============================] - 0s 452ms/step - loss: 1.4033 - accuracy: 0.2333
Epoch 2/30
1/1 [==============================] - 0s 4ms/step - loss: 1.2637 - accuracy: 0.7000
Epoch 3/30
1/1 [==============================] - 0s 4ms/step - loss: 1.1494 - accuracy: 0.8667
...
Epoch 27/30
1/1 [==============================] - 0s 5ms/step - loss: 0.0439 - accuracy: 1.0000
Epoch 28/30
1/1 [==============================] - 0s 4ms/step - loss: 0.0374 - accuracy: 1.0000
Epoch 29/30
1/1 [==============================] - 0s 3ms/step - loss: 0.0320 - accuracy: 1.0000
Epoch 30/30
1/1 [==============================] - 0s 3ms/step - loss: 0.0275 - accuracy: 1.0000
[INFO] Plotting loss...
[INFO] Plotting TSNE embeddings...
```

使用我们的小输入数据集，词袋模型很快达到了 100%的准确性，并适合数据。然而，我们将根据 TSNE 地块做出最终评估。损失图可在图 3 的**中看到。**

损失下降得相当快。对于给定的数据集，我们的模型完美地过度拟合了它。

```py
[nltk_data] Downloading package stopwords to /root/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[INFO] Starting Word2Vec training...
100% 1000/1000 [04:52<00:00,  3.42it/s]
[INFO] Plotting Loss...
[INFO] Plotting TSNE Embeddings...
```

由于 Word2Vec 模型是我们直接构建的，我们将根据图 4 中的损失图来评估损失。

Word2Vec 的损失，虽然比单词袋更困难(因为有更多的标签)，但对于我们的小数据集来说，下降得很好。让我们看看 TSNE 的情节！

在**图 5 和图 6** 中，我们分别得到了单词袋和 Word2Vec 的 TSNE 图。

仔细观察(点击放大图片)，我们可以看到，虽然没有形成明确的分组，但有一些相似的上下文词或多或少彼此接近。例如，我们可以看到像“汉堡”和“比萨饼”这样的食物彼此靠近。

然而，我们必须记住，在词袋中，完整的句子被认为是输入。这就是为什么单词没有明确的分组。另一个原因可能是维度缩减导致了信息的丢失。添加另一个密集层使其更少依赖于我们正在考虑的层的权重。

在**图 6** 中，我们有 Word2Vec 的 TSNE 图。

很快，我们可以看到几个分组。您可以在代码的合作版本中放大这些图像，并检查分组。标准连续词袋和我们今天所做的区别在于，我们没有考虑句子中的窗口和中心词，而是为每个句子指定了一个明确的标签。

这使得矩阵在训练时更容易创建分组。这里的结果显然比袋的话。然而，如果你想得出自己的结论，别忘了在你自己的数据上尝试一下。

* * *

* * *

## [**汇总**](#TOC)

在今天的教程中，我们学习了单词袋和 Word2Vec 之间的根本区别。这两者都是 NLP 世界中的巨大垫脚石，但是理解为什么这两种技术都以自己的方式坚持使用嵌入是很重要的。

我们在一个小数据集上工作，以理解这两种方法分歧的本质。尽管弓形结构显示了较低的最终损失值，但 TSNE 曲线显示了显著的差异。这留下了一个有趣的结论，关于嵌入空间的理解。

回顾一下我们在本系列中的第二篇博客文章,我们会发现 Word2Vec 方法具有非常高的损耗值。尽管如此，可视化非常直观，显示了许多视觉分组。高损失的原因可以解释为基于每个句子出现的几个中心词的许多“标签”。

然而，这个问题在这里不再普遍，因为我们对每个句子都有固定的标签。所以很自然地，Word2Vec 方法显示了这个数据集的即时分组。

这可以在更大的数据集上进一步实验，以得出更明确的结论。然而，今天的结果质疑评估方法时使用的正确指标。

* * *

### [**引用信息**](#TOC)

**Chakraborty，D.** “词袋和 Word2Vec 之间的比较”， *PyImageSearch* ，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva，R. Raha 和 A. Thanki 编辑。，2022 年，【https://pyimg.co/txq23 

```py
@incollection{Chakraborty_2022_Comparison,
  author = {Devjyoti Chakraborty},
  title = {Comparison Between {BagofWords} and {Word2Vec}},
  booktitle = {PyImageSearch},
  editor = {Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/txq23},
}
```

* * *

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****