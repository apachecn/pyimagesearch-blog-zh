# 包含 Keras、TensorFlow 和深度学习的暹罗网络

> 原文：<https://pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/>

在本教程中，您将学习如何使用 Keras、TensorFlow 和深度学习来实现和训练暹罗网络。

本教程是我们关于暹罗网络基础的三部分系列的第二部分:

*   **第一部分:** *[用 Python](https://pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/)* 构建连体网络的图像对(上周的帖子)
*   **第二部分:** *用 Keras、TensorFlow 和深度学习训练暹罗网络*(本周教程)
*   **第三部分:** *使用连体网络比较图像*(下周教程)

使用我们的暹罗网络实施，我们将能够:

*   向我们的网络呈现两个输入图像。
*   网络将预测这两幅图像是否属于同一类(即*验证*)。
*   然后，我们将能够检查网络的置信度得分来确认验证。

暹罗网络的实际、真实使用案例包括人脸识别、签名验证、处方药识别、*等等！*

此外，暹罗网络可以用少得惊人的数据进行训练，使更高级的应用成为可能，如一次性学习和少量学习。

**要学习如何用 Keras 和 TenorFlow 实现和训练暹罗网络，*继续阅读。***

## **包含 Keras、TensorFlow 和深度学习的连体网络**

在本教程的第一部分，我们将讨论暹罗网络，它们如何工作，以及为什么您可能希望在自己的深度学习应用中使用它们。

从那里，您将学习如何配置您的开发环境，以便您可以按照本教程学习，并学习如何训练您自己的暹罗网络。

然后，我们将查看我们的项目目录结构，并实现一个配置文件，后面是三个助手函数:

1.  一种用于生成图像对的方法，以便我们可以训练我们的连体网络
2.  一个定制的 CNN 层，用于计算网络内向量*之间的欧几里德距离*
3.  用于将暹罗网络训练历史记录绘制到磁盘的实用程序

给定我们的辅助工具，我们将实现用于从磁盘加载 MNIST 数据集的训练脚本，并根据数据训练一个暹罗网络。

我们将讨论我们的结果来结束本教程。

### 什么是暹罗网络，它们是如何工作的？

[上周的教程](https://pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/)涵盖了暹罗网络的基础知识，它们是如何工作的，以及有哪些现实世界的应用适用于它们。**我将在这里对它们进行快速回顾，但我*强烈建议*你阅读上周的指南，以获得对暹罗网络的更深入的回顾。**

**本节顶部的图 1** 显示了暹罗网络的基本架构。你会立即注意到暹罗网络架构与大多数标准分类架构*不同*。

注意网络有两个输入端和*两个分支*(即“姐妹网络”)。这些姐妹网络彼此完全相同。组合两个子网络的输出，然后返回最终的输出相似性得分。

为了使这个概念更具体一些，让我们在上面的**图 1** 中进一步分解它:

*   **在*左侧*我们向暹罗模型展示了两个示例数字(来自 MNIST 数据集)。**我们的目标是确定这些数字是否属于*的同类*。
*   中间的*表示暹罗网络本身。**这两个子网络具有*相同的*架构和*相同的*参数，并且它们*镜像*彼此**——如果一个子网络中的权重被更新，则其他子网络中的权重也被更新。*
*   每个子网的输出是全连接(FC)层。我们通常计算这些输出之间的欧几里德距离，并通过 sigmoid 激活来馈送它们，以便我们可以确定两个输入图像有多相似。更接近“1”的 sigmoid 激活函数值意味着*更相似*，而更接近“0”的值表示“不太相似”

为了实际训练暹罗网络架构，我们可以利用许多损失函数，包括二元交叉熵、三元损失和对比损失。

后两个损失函数需要**图像三元组**(网络的三个输入图像)，这与我们今天使用的**图像对**(两个输入图像)不同。

今天我们将使用二进制交叉熵来训练我们的暹罗网络。在未来，我将涵盖中级/高级暹罗网络，包括图像三元组、三元组损失和对比损失——但现在，让我们先走后跑。

### **配置您的开发环境**

在这一系列关于暹罗网络的教程中，我们将使用 Keras 和 TensorFlow。我建议你现在就花时间配置你的深度学习开发环境。

我建议您按照这两个指南中的任何一个在您的系统上安装 TensorFlow 和 Keras(对于本指南，我建议您安装 **TensorFlow 2.3** ):

*   *[如何在 Ubuntu 上安装 tensor flow 2.0](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)*
*   *[如何在 macOS 上安装 tensor flow 2.0](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)*

这两个教程都将帮助您在一个方便的 Python 虚拟环境中，用这篇博文所需的所有软件来配置您的系统。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在你的 Windows、macOS 或 Linux 系统上运行代码*了吗？***

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

在我们可以训练我们的暹罗网络之前，我们首先需要回顾我们的项目目录结构。

首先使用本教程的 ***【下载】*** 部分下载源代码、预先训练好的暹罗网络模型等。

从那里，让我们来看看里面有什么:

```py
$ tree . --dirsfirst
.
├── output
│   ├── siamese_model
│   │   ├── variables
│   │   │   ├── variables.data-00000-of-00001
│   │   │   └── variables.index
│   │   └── saved_model.pb
│   └── plot.png
├── pyimagesearch
│   ├── config.py
│   ├── siamese_network.py
│   └── utils.py
└── train_siamese_network.py

2 directories, 6 files
```

查看完项目目录结构后，让我们继续创建配置文件。

***注:**与本教程相关的“下载”中包含的预训练`siamese_model`是使用 **TensorFlow 2.3 创建的。**我建议您使用 TensorFlow 2.3 来完成本指南。如果您希望使用 TensorFlow 的另一个版本，这完全没问题，但是您需要执行`train_siamese_network.py`来训练和序列化模型。当我们使用训练好的暹罗网络来比较图像时，你还需要为下周的教程保留这个模型。*

### **创建我们的暹罗网络配置文件**

我们的配置文件短小精悍。打开`config.py`，插入以下代码:

```py
# import the necessary packages
import os

# specify the shape of the inputs for our network
IMG_SHAPE = (28, 28, 1)

# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 100
```

**第 5 行**初始化我们输入的`IMG_SHAPE`空间维度。因为我们使用的是 MNIST 数字数据集，所以我们的图像是带有单一灰度通道的 *28×28* 像素。

然后我们定义我们的`BATCH_SIZE`和我们正在训练的纪元总数。

在我们自己的实验中，我们发现仅针对`10`个时期的训练产生了良好的结果，但是更长时间的训练产生了更高的准确性。如果你时间紧迫，或者如果你的机器没有 GPU，将`EPOCHS`升级到`10`仍然会产生好的结果。

接下来，让我们定义输出路径:

```py
# define the path to the base output directory
BASE_OUTPUT = "output"

# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
```

### **使用 Keras 和 TensorFlow 实施暹罗网络架构**

暹罗网络架构由*两个或更多*姐妹网络组成(在上面的**图 3** 中突出显示)。**本质上，姐妹网络是一个基本的卷积神经网络，它产生一个全连接(FC)层，有时称为*嵌入式*层。**

当我们开始构建暹罗网络架构本身时，我们将:

1.  实例化我们的姐妹网络
2.  创建一个`Lambda`层，用于计算姐妹网络输出之间的欧几里德距离
3.  创建具有单个节点和 sigmoid 激活函数的 FC 层

结果将是一个完全构建的连体网络。

但是在我们到达那里之前，我们首先需要实现我们的暹罗网络架构的姐妹网络组件。

打开项目目录结构中的`siamese_network.py`,让我们开始工作:

```py
# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
```

我们从第 2-8 行开始，导入我们需要的 Python 包。如果你以前曾经用 Keras/TensorFlow 训练过 CNN，这些输入对你来说应该都很标准。

如果你需要复习 CNN，我推荐你阅读我的 [Keras 教程](https://pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/)以及我的书 *[用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)。*

考虑到我们的导入，我们现在可以定义负责构建姐妹网络的`build_siamese_model`函数:

```py
def build_siamese_model(inputShape, embeddingDim=48):
	# specify the inputs for the feature extractor network
	inputs = Input(inputShape)

	# define the first set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dropout(0.3)(x)

	# second set of CONV => RELU => POOL => DROPOUT layers
	x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
	x = MaxPooling2D(pool_size=2)(x)
	x = Dropout(0.3)(x)
```

**第 12 行**初始化我们姐妹网络的输入空间维度。

从那里，**第 15-22 行**定义了两组`CONV => RELU => POOL`层集合。每个`CONV`层总共学习 64 个 *2×2* 滤镜。然后，我们应用一个 ReLU 激活函数，并使用一个 *2×2* 步距应用最大池。

我们现在可以完成构建姐妹网络架构了:

```py
	# prepare the final outputs
	pooledOutput = GlobalAveragePooling2D()(x)
	outputs = Dense(embeddingDim)(pooledOutput)

	# build the model
	model = Model(inputs, outputs)

	# return the model to the calling function
	return model
```

**第 25 行**将全局平均池应用于 *7x7x64* 卷(假设对网络的 *28×28* 输入)，产生 64-d 的输出

我们取这个`pooledOutput`，然后用指定的`embeddingDim`(**Line 26**)——**应用一个全连接层，这个** `Dense` **层作为姐妹网络的输出。**

**第 29 行**然后建立姐妹网络`Model`，然后返回给调用函数。

我在下面提供了该模型的摘要:

```py
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_3 (InputLayer)         [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 28, 28, 64)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 64)        16448     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 7, 7, 64)          0         
_________________________________________________________________
global_average_pooling2d (Gl (None, 64)                0         
_________________________________________________________________
dense (Dense)                (None, 48)                3120      
=================================================================
Total params: 19,888
Trainable params: 19,888
Non-trainable params: 0
_________________________________________________________________
```

下面是我们刚刚构建的模型的快速回顾:

*   每个姐妹网络将接受一个 *28x28x1* 输入。
*   然后，我们应用 CONV 层学习总共 64 个过滤器。使用 *2×2* 步幅应用最大池，以将空间维度减少到*14×14×64*。
*   应用了另一个 CONV 层(再次学习 64 个过滤器)和池层，将空间维度进一步减少到 *7x7x64。*
*   全局平均池用于将 *7x7x64* 卷平均到 64-d。
*   这个 64-d 池化输出被传递到具有 48 个节点的 FC 层。
*   48 维向量作为我们姐妹网络的输出。

在`train_siamese_network.py`脚本中，您将学习如何实例化我们姐妹网络的两个实例，然后完成暹罗网络架构本身的构建。

### **实现我们的配对生成、欧几里德距离和绘图历史实用函数**

```py
# import the necessary packages
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
```

我们从第 2-4 行的**开始，导入我们需要的 Python 包。**

```py
def make_pairs(images, labels):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	pairLabels = []

	# calculate the total number of classes present in the dataset
	# and then build a list of indexes for each class label that
	# provides the indexes for all examples with a given label
	numClasses = len(np.unique(labels))
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

	# loop over all images
	for idxA in range(len(images)):
		# grab the current image and label belonging to the current
		# iteration
		currentImage = images[idxA]
		label = labels[idxA]

		# randomly pick an image that belongs to the *same* class
		# label
		idxB = np.random.choice(idx[label])
		posImage = images[idxB]

		# prepare a positive pair and update the images and labels
		# lists, respectively
		pairImages.append([currentImage, posImage])
		pairLabels.append([1])

		# grab the indices for each of the class labels *not* equal to
		# the current label and randomly pick an image corresponding
		# to a label *not* equal to the current label
		negIdx = np.where(labels != label)[0]
		negImage = images[np.random.choice(negIdx)]

		# prepare a negative pair of images and update our lists
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])

	# return a 2-tuple of our image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))
```

我不打算对这个函数进行全面的回顾，因为我们在关于暹罗网络的[系列的第 1 部分*中已经详细介绍过了*；然而，高层次的要点是:](https://pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/)

1.  为了训练暹罗网络，我们需要*正*和*负*对
2.  一个**正对**是属于*同一*类的两个图像(即数字*“8”*的两个例子)
3.  一个**负对**是属于*不同*类的两个图像(即一个图像包含一个*【1】*，另一个图像包含一个*【3】*)
4.  `make_pairs`函数接受一组`images`和相关联的`labels`的输入，然后构建这些用于训练的正负图像对，并将它们返回给调用函数

关于`make_pairs`函数更详细的回顾，请参考我的教程 *[用 Python](https://pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/) 为暹罗网络构建图像对。*

我们的下一个函数`euclidean_distance`接受一个 2 元组的`vectors`，然后利用 Keras/TensorFlow 函数计算它们之间的欧氏距离:

```py
def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors

	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)

	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))
```

我们通过取平方差之和的平方根来舍入函数，得到欧几里德距离(**第 57 行**)。

请注意，我们使用 Keras/TensorFlow 函数来计算欧几里德距离，而不是使用 NumPy 或 SciPy。

这是为什么呢？

使用 NumPy 和 SciPy 内置的欧几里德距离函数不是更简单吗？

为什么要大费周章地用 Keras/TensorFlow 重新实现欧几里德距离呢？

一旦我们到达`train_siamese_network.py`脚本，**，原因将变得更加清楚，但是要点是为了构建我们的暹罗网络架构，我们需要能够计算暹罗架构本身内部的姐妹网络输出*之间的欧几里德距离。***

为了完成这项任务，我们将使用一个自定义的`Lambda`层，该层可用于在模型中嵌入任意 Keras/TensorFlow 函数(因此，Keras/TensorFlow 函数用于实现欧几里德距离)。

我们的最后一个函数`plot_training`，接受(1)来自调用`model.fit`的训练历史和(2)一个输出`plotPath`:

```py
def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)
```

给定我们的训练历史变量`H`，我们绘制我们的训练和验证损失和准确性。然后将输出图保存到磁盘的`plotPath`中。

### **使用 Keras 和 TensorFlow 创建我们的暹罗网络培训脚本**

我们现在准备实施我们的暹罗网络培训脚本！

在内部`train_siamese_network.py`我们将:

1.  从磁盘加载 MNIST 数据集
2.  构建我们的训练和测试图像对
3.  创建我们的`build_siamese_model`的两个实例作为我们的姐妹网络
4.  通过我们定制的`euclidean_distance`函数(使用一个`Lambda`层)用管道传输姐妹网络的输出，完成暹罗网络架构的构建
5.  对欧几里德距离的输出应用 sigmoid 激活
6.  在我们的图像对上训练暹罗网络架构

这听起来像是一个复杂的过程，但是我们能够用不到 60 行代码完成所有这些任务！

打开`train_siamese_network.py`，让我们开始工作:

```py
# import the necessary packages
from pyimagesearch.siamese_network import build_siamese_model
from pyimagesearch import config
from pyimagesearch import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.datasets import mnist
import numpy as np
```

**第 2-10 行**导入我们需要的 Python 包。值得注意的进口包括:

*   ``build_siamese_model`` :构建连体网络架构的姊妹网络组件
*   ``config`` :存储我们的训练配置
*   ``utils`` :保存我们的辅助函数实用程序，用于创建图像对，绘制训练历史，并使用 Keras/TensorFlow 函数计算欧几里德距离
*   ``Lambda`` :采用我们的欧几里德距离实现，并将其嵌入暹罗网络架构本身

导入工作完成后，我们可以继续从磁盘加载 MNIST 数据集，对其进行预处理，并构建图像对:

```py
# load MNIST dataset and scale the pixel values to the range of [0, 1]
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0

# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

# prepare the positive and negative pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
(pairTest, labelTest) = utils.make_pairs(testX, testY)
```

**第 14 行**从磁盘加载 MNIST 数字数据集。

然后，我们对 MNIST 图像进行预处理，将它们从范围*【0，255】*缩放到*【0，1】*(**第 15 行和第 16 行**)，然后添加通道维度(**第 19 行和第 20 行**)。

我们使用我们的`make_pairs`函数分别为我们的训练集和测试集创建正面和负面图像对(**第 24 行和第 25 行**)。如果你需要复习一下`make_pairs`功能，我建议你阅读本系列的[第 1 部分，其中详细介绍了图像对。](https://pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/)

**现在让我们构建我们的连体网络架构:**

```py
# configure the siamese network
print("[INFO] building siamese network...")
imgA = Input(shape=config.IMG_SHAPE)
imgB = Input(shape=config.IMG_SHAPE)
featureExtractor = build_siamese_model(config.IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
```

**第 29-33 行**创建我们的姐妹网络:

*   首先，我们创建两个输入，一个用于图像对中的每个图像(**行 29 和 30** )。
*   **31 号线**再搭建姊妹网架构，作为`featureExtractor`。
*   该对中的每个图像将通过`featureExtractor`，产生一个 48 维的特征向量(**第 32 行和第 33 行**)。由于在一对中有*两个*图像，因此我们有*两个* 48-d 特征向量。

也许你想知道为什么我们没有打两次电话？我们的架构中有两个姐妹网络，对吗？

好吧，请记住你上周学的内容:

> ***“这两个姐妹网络具有相同的架构和相同的参数，并且相互镜像** —如果一个子网中的权重被更新，则其他网络中的权重也会被更新。”*

因此，即使有两个姐妹网络*，我们实际上还是将它们实现为一个*单个*实例。本质上，这个单一的网络被视为一个特征提取器(因此我们将其命名为`featureExtractor`)。然后，当我们训练网络时，通过反向传播来更新网络的权重。*

现在，让我们完成暹罗网络架构的构建:

```py
# finally, construct the siamese network
distance = Lambda(utils.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)
```

**第 36 行**利用一个`Lambda`层来计算`featsA`和`featsB`网络之间的`euclidean_distance`(记住，这些值是通过姐妹网络特征提取器传递图像对中的每个图像的*输出*)。

然后，我们应用一个带有单个节点的`Dense`层，该节点应用了一个 sigmoid 激活函数。

这里使用了 sigmoid 激活函数，因为该函数的输出范围是*【0，1】。*更接近`0`的输出意味着图像对*不太相似*(因此来自*不同的*类)，而更接近`1`的值意味着它们*更相似*(并且更可能来自*相同的*类)。

**线 38** 然后构建连体网络`Model`。`inputs`由我们的图像对`imgA`和`imgB`组成。网络的`outputs`是乙状结肠激活。

既然我们的暹罗网络架构已经构建完毕，我们就可以继续训练它了:

```py
# compile the model
print("[INFO] compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])

# train the model
print("[INFO] training model...")
history = model.fit(
	[pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
	validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
	batch_size=config.BATCH_SIZE, 
	epochs=config.EPOCHS)
```

**第 42 行和第 43 行**使用二进制交叉熵作为我们的损失函数来编译我们的暹罗网络。

我们在这里使用二进制交叉熵，因为这本质上是一个两类分类问题——给定一对输入图像，我们试图确定这两幅图像有多相似，更具体地说，它们是否来自*相同的*或*不同的*类。

这里也可以使用更高级的损失函数，包括三重损失和对比损失。我将在 PyImageSearch 博客的未来系列中介绍如何使用这些损失函数，包括构造图像三元组(将介绍更高级的暹罗网络)。

**第 47-51 行**然后在图像对上训练暹罗网络。

一旦模型被训练，我们可以将它序列化到磁盘并绘制训练历史:

```py
# serialize the model to disk
print("[INFO] saving siamese model...")
model.save(config.MODEL_PATH)

# plot the training history
print("[INFO] plotting training history...")
utils.plot_training(history, config.PLOT_PATH)
```

恭喜你实现了我们的暹罗网络培训脚本！

### **使用 Keras 和 TensorFlow 训练我们的暹罗网络**

我们现在准备使用 Keras 和 TensorFlow 来训练我们的暹罗网络！确保使用本教程的 ***【下载】*** 部分下载源代码。

从那里，打开一个终端，并执行以下命令:

```py
$ python train_siamese_network.py
[INFO] loading MNIST dataset...
[INFO] preparing positive and negative pairs...
[INFO] building siamese network...
[INFO] training model...
Epoch 1/100
1875/1875 [==============================] - 11s 6ms/step - loss: 0.6210 - accuracy: 0.6469 - val_loss: 0.5511 - val_accuracy: 0.7541
Epoch 2/100
1875/1875 [==============================] - 11s 6ms/step - loss: 0.5433 - accuracy: 0.7335 - val_loss: 0.4749 - val_accuracy: 0.7911
Epoch 3/100
1875/1875 [==============================] - 11s 6ms/step - loss: 0.5014 - accuracy: 0.7589 - val_loss: 0.4418 - val_accuracy: 0.8040
Epoch 4/100
1875/1875 [==============================] - 11s 6ms/step - loss: 0.4788 - accuracy: 0.7717 - val_loss: 0.4125 - val_accuracy: 0.8173
Epoch 5/100
1875/1875 [==============================] - 11s 6ms/step - loss: 0.4581 - accuracy: 0.7847 - val_loss: 0.3882 - val_accuracy: 0.8331
...
Epoch 95/100
1875/1875 [==============================] - 11s 6ms/step - loss: 0.3335 - accuracy: 0.8565 - val_loss: 0.3076 - val_accuracy: 0.8630
Epoch 96/100
1875/1875 [==============================] - 11s 6ms/step - loss: 0.3326 - accuracy: 0.8564 - val_loss: 0.2821 - val_accuracy: 0.8764
Epoch 97/100
1875/1875 [==============================] - 11s 6ms/step - loss: 0.3333 - accuracy: 0.8566 - val_loss: 0.2807 - val_accuracy: 0.8773
Epoch 98/100
1875/1875 [==============================] - 11s 6ms/step - loss: 0.3335 - accuracy: 0.8554 - val_loss: 0.2717 - val_accuracy: 0.8836
Epoch 99/100
1875/1875 [==============================] - 11s 6ms/step - loss: 0.3307 - accuracy: 0.8578 - val_loss: 0.2793 - val_accuracy: 0.8784
Epoch 100/100
1875/1875 [==============================] - 11s 6ms/step - loss: 0.3329 - accuracy: 0.8567 - val_loss: 0.2751 - val_accuracy: 0.8810
[INFO] saving siamese model...
[INFO] plotting training history...
```

正如你所看到的，我们的模型在我们的验证集上获得了 **~88.10%的准确度**，**意味着 88%的时候，模型能够正确地确定两幅输入图像是否属于*相同的类别*。**

**上面的图 4** 显示了我们在 100 个时期的训练历史。我们的模型看起来相当稳定，鉴于我们的验证损失低于我们的训练损失，似乎我们可以通过“更努力地训练”来进一步提高准确性(“T2”，我在这里提到了这个)。

检查您的`output`目录，您现在应该看到一个名为`siamese_model`的目录:

```py
$ ls output/
plot.png		siamese_model
$ ls output/siamese_model/
saved_model.pb	variables
```

这个目录包含我们的序列化暹罗网络。下周你将学习如何利用这个训练好的模型，并使用它对输入图像进行预测— **敬请关注我们的暹罗网络介绍系列的最后部分；你不会想错过的！**

## **总结**

在本教程中，您学习了如何使用 Keras、TensorFlow 和深度学习来实现和训练暹罗网络。

我们在 MNIST 数据集上训练了我们的暹罗网络。我们的网络接受一对输入图像(数字),然后尝试确定这两个图像是否属于同一类。

例如，如果我们要向模型呈现两幅图像，每幅图像都包含一个*“9”*，那么暹罗网络将报告这两幅图像之间的*高相似度*，表明它们确实属于*相同的*类。

然而，如果我们提供两个图像，一个包含一个*“9”*，另一个包含一个*“2”*，那么网络应该报告*低相似度，假定这两个数字属于*单独的*类。*

为了方便起见，我们在这里使用了 MNIST 数据集，这样我们可以了解暹罗网络的基本原理；然而，这种相同类型的训练过程可以应用于面部识别、签名验证、处方药丸识别等。

下周，您将学习如何实际使用我们训练过的、序列化的暹罗网络模型，并使用它来进行相似性预测。

然后我会在未来的一系列文章中讨论更高级的暹罗网络，包括图像三联体、三联体损失和对比损失。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***