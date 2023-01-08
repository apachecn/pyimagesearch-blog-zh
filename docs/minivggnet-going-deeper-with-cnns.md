# MiniVGGNet:深入 CNN

> 原文：<https://pyimagesearch.com/2021/05/22/minivggnet-going-deeper-with-cnns/>

在我们之前的[教程](https://pyimagesearch.com/2021/05/22/lenet-recognizing-handwritten-digits/) 中，我们讨论了 LeNet，这是深度学习和计算机视觉文献中的一个开创性的卷积神经网络。VGGNet(有时简称为 VGG)是由 [Simonyan 和 Zisserman 在他们 2014 年的论文*中首次提出的，用于大规模图像识别的超深度学习卷积神经网络*](https://arxiv.org/abs/1409.1556) 。他们工作的主要贡献是证明了具有非常小(3 *×* 3)过滤器的架构可以被训练到越来越高的深度(16-19 层)，并在具有挑战性的 ImageNet 分类挑战中获得最先进的分类。

**要学习如何实现 VGGNet，*继续阅读。***

## **MiniVGGNet:深入 CNN**

以前，深度学习文献中的网络架构使用多种过滤器尺寸:

CNN 的第一层通常包括介于 7 *×* 7 ( [Krizhevsky、Sutskever 和 Hinton，2012](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) )和 11 *×* 11 ( [Sermanet 等人，2013](http://arxiv.org/abs/1312.6229) )之间的滤波器大小。从那里开始，过滤器尺寸逐渐减小到 5 *×* 5。最后，只有网络的最深层使用了 3 个*×3 个过滤器。*

VGGNet 的独特之处在于它在整个架构中使用了 3 个内核*×3 个内核**。可以说，这些小内核的使用帮助 VGGNet *将*推广到网络最初训练之外的分类问题。***

 **任何时候你看到一个完全由 3 个 *×* 3 个过滤器组成的网络架构，你可以放心，它的灵感来自 VGGNet。回顾 VGGNet 的*整个* 16 和 19 层变体对于这个卷积神经网络的介绍**来说太超前了。**

相反，我们将回顾 VGG 网络家族，并定义 CNN 必须展现出什么样的特征才能融入这个家族。在此基础上，我们将实现一个名为 *MiniVGGNet* 的较小版本的 VGGNet，它可以很容易地在您的系统上进行训练。这个实现还将演示如何使用两个重要的层— *批处理规范化* (BN)和*丢失*。

### **VGG 网络家族**

卷积神经网络的 VGG 家族可以由两个关键部分来表征:

1.  网络中的所有`CONV`层仅使用*3*3*×*3 过滤器。
2.  在应用`POOL`操作之前，堆叠*多个* `CONV => RELU`层集(其中连续`CONV => RELU`层的数量通常*增加*越深*越多*)。

在本教程中，我们将讨论 VGGNet 架构的一种变体，我称之为“MiniVGGNet ”,因为该网络比它的老大哥要浅得多。

### **配置您的开发环境**

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果你需要帮助为 OpenCV 配置开发环境，我*强烈推荐*阅读我的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让你启动并运行。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

#### **(迷你)VGGNet 架构**

在 ShallowNet 和 LeNet 中，我们应用了一系列的`CONV => RELU => POOL`层。然而，在 VGGNet 中，我们在应用单个`POOL`层之前堆叠多个 `CONV => RELU`层*。这样做允许网络在通过`POOL`操作对空间输入大小进行下采样之前，从`CONV`层中学习更多丰富的特征。*

总体来说，MiniVGGNet 由*两组*的`CONV => RELU => CONV => RELU => POOL`层组成，后面是一组`FC => RELU => FC => SOFTMAX`层。前两个`CONV`层将学习 32 个滤镜，每个大小为 3*×3*。接下来的两个`CONV`层将学习 64 个滤镜，同样，每个大小为 3 *×* 3。我们的`POOL`层将以 2 *×* 2 的步幅在 2 *×* 2 的窗口上执行最大池化。我们还将在激活后插入批量标准化层*，以及在`POOL`和`FC`层后插入丢弃层(`DO`)。*

网络架构本身详见**表 1** ，其中初始输入图像大小假设为 32 *×* 32 *×* 3。

同样，请注意批量归一化和丢弃层是如何根据我在[卷积神经网络(CNN)和层类型](https://pyimagesearch.com/2021/05/14/convolutional-neural-networks-cnns-and-layer-types/)中的*“经验法则】*包含在网络架构中的。应用批量标准化将有助于减少过度拟合的影响，并提高我们在 CIFAR-10 上的分类准确性。

### **实现 MiniVGGNet**

根据**表 1** 中对 MiniVGGNet 的描述，我们现在可以使用 Keras 实现网络架构。首先，在`pyimagesearch.nn.conv`子模块中添加一个名为`minivggnet.py`的新文件——这是我们编写 MiniVGGNet 实现的地方:

```py
--- pyimagesearch
|    |--- __init__.py
|    |--- nn
|    |    |--- __init__.py
...
|    |    |--- conv
|    |    |    |--- __init__.py
|    |    |    |--- lenet.py
|    |    |    |--- minivggnet.py
|    |    |    |--- shallownet.py
```

创建完`minivggnet.py`文件后，在您最喜欢的代码编辑器中打开它，然后我们开始工作:

```py
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
```

**第 2-10 行**从 Keras 库中导入我们需要的类。这些导入中的大部分你已经见过了，但我想让你注意一下`BatchNormalization` ( **第 3 行**)和`Dropout` ( **第 8 行**)——这些类将使我们能够对我们的网络架构应用批量标准化和丢弃。

就像我们对 ShallowNet 和 LeNet 的实现一样，我们将定义一个`build`方法，可以调用它来使用提供的`width`、`height`、`depth`和数量`classes`来构建架构:

```py
class MiniVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
```

**第 17 行**实例化了`Sequential`类，Keras 中顺序神经网络的构建块。然后我们初始化`inputShape`，假设我们正在使用信道最后的排序(**第 18 行**)。

**第 19 行**引入了一个我们之前没见过的变量`chanDim`，*通道维度的指标*。批量归一化在通道上操作，因此为了应用 BN，我们需要知道在哪一个轴上归一化。设置`chanDim = -1`意味着通道维度的索引*在输入形状中最后*(即通道最后排序)。但是，如果我们使用通道优先排序(**第 23-25 行**)，我们需要更新`inputShape`并设置`chanDim = 1`，因为通道维度现在是输入形状中的第一个条目。

MiniVGGNet 的第一层块定义如下:

```py
  		# first CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
```

在这里，我们可以看到我们的架构由`(CONV => RELU => BN) * 2 => POOL => DO`组成。**第 28 行**定义了一个有 32 个滤镜的`CONV`层，每个滤镜有一个 3 *×* 3 的滤镜大小。然后我们应用一个 ReLU 激活(**第 30 行**)，它立即被输入到一个`BatchNormalization`层(**第 31 行**)来使激活归零。

然而，我们没有应用一个`POOL`层来减少我们输入的空间维度，而是应用另一组`CONV => RELU => BN`——这允许我们的网络学习更丰富的特征，这是训练更深层次 CNN 时的常见做法。

在**线 35** 上，我们用的是尺寸为 2 *×* 2 的`MaxPooling2D`。由于我们没有*显式*设置步幅，Keras *隐式*假设我们的步幅等于最大池大小(即 2 *×* 2)。

然后，我们将`Dropout`应用于**第 36 行**，概率为 *p* = 0 *。* 25，暗示来自`POOL`层的一个节点在训练时会以 25%的概率随机断开与下一层的连接。我们应用辍学来帮助减少过度拟合的影响。你可以在单独的一课[中读到更多关于辍学的内容。然后，我们将第二层块添加到下面的 MiniVGGNet 中:](https://pyimagesearch.com/2021/05/14/convolutional-neural-networks-cnns-and-layer-types/)

```py
  		# second CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
```

上面的代码遵循与上面完全相同的*模式*；然而，现在我们正在学习两组 64 个过滤器(每组尺寸为 3 *×* 3)，而不是 32 个过滤器。同样，随着空间输入尺寸*在网络中更深处减小*，通常*增加*滤波器的数量。

接下来是我们的第一组(也是唯一一组)图层:

```py
  		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
```

我们的`FC`层有`512`个节点，后面是 ReLU 激活和 BN。我们还将在这里应用辍学，将概率增加到 50% —通常您会看到辍学与 *p* = 0 *。* 5 应用于`FC`层间。

最后，我们应用 softmax 分类器，并将网络架构返回给调用函数:

```py
  		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
```

现在我们已经实现了 MiniVGGNet 架构，让我们继续将它应用于 CIFAR-10。

### **cifar-10**上的 minivggnet

我们将遵循类似的模式训练 MiniVGGNet，就像我们在之前的教程 中为 LeNet 所做的那样，只是这次使用的是 CIFAR-10 数据集:

*   从磁盘加载 CIFAR-10 数据集。
*   实例化 MiniVGGNet 架构。
*   使用训练数据训练 MiniVGGNet。
*   使用测试数据评估网络性能。

要创建驱动程序脚本来训练 MiniVGGNet，请打开一个新文件，将其命名为`minivggnet_cifar10.py`，并插入以下代码:

```py
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import MiniVGGNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
```

**第 2 行**导入了`matplotlib`库，我们稍后将使用它来绘制我们的精度和损耗。我们需要将`matplotlib`后端设置为`Agg`，以指示创建一个*非交互*，它将被简单地保存到磁盘。取决于你默认的`matplotlib`后端是什么*和*你是否正在远程访问你的深度学习机器(例如，通过 SSH)，X11 会话可能会超时。如果发生这种情况，`matplotlib`会在试图显示你的身材时出错。相反，我们可以简单地将背景设置为`Agg`，并在完成网络训练后将图形写入磁盘。

**第 9-13 行**导入我们需要的 Python 包的剩余部分，所有这些你以前都见过——例外的是第 8 行**的`MiniVGGNet`，**,我们之前已经实现了。

接下来，让我们解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())
```

该脚本只需要一个命令行参数`--output`，即输出训练和损耗图的路径。

我们现在可以加载 CIFAR-10 数据集(预分割为训练和测试数据)，将像素缩放到范围`[0, 1]`，然后一次性编码标签:

```py
# load the training and testing data, then scale it into the
# range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck"]
```

让我们编译我们的模型并开始训练 MiniVGGNet:

```py
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=64, epochs=40, verbose=1)
```

我们将使用 SGD 作为我们的优化器，学习率为 *α* = 0 *。* 01 和 *γ* = 0 *的动量项。* 9。设置`nestrov=True`表示我们希望将 Nestrov 加速梯度应用于 SGD 优化器。

我们还没有见过的一个优化器术语是`decay`参数。这个论点是用来随着时间慢慢降低学习率的。衰减学习率有助于减少过拟合并获得更高的分类精度——学习率越小，权重更新就越小。`decay`的一个常见设置是将初始学习率除以总时期数——在这种情况下，我们将用 0.01 的初始学习率训练我们的网络总共 40 个时期，因此为`decay = 0.01 / 40`。

训练完成后，我们可以评估网络并显示一份格式良好的分类报告:

```py
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))
```

将我们的损耗和精度图保存到磁盘:

```py
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
```

在评估 MinIVGGNet 时，我进行了两个实验:

1.  一个*与*批量归一化。
2.  一个*没有*批量归一化。

让我们来看看这些结果，比较应用批处理规范化时网络性能是如何提高的。

#### **批量归一化**

要在 CIFAR-10 数据集上训练 MiniVGGNet，只需执行以下命令:

```py
$ python minivggnet_cifar10.py --output output/cifar10_minivggnet_with_bn.png
[INFO] loading CIFAR-10 data...
[INFO] compiling model...
[INFO] training network...
Train on 50000 samples, validate on 10000 samples
Epoch 1/40
23s - loss: 1.6001 - acc: 0.4691 - val_loss: 1.3851 - val_acc: 0.5234
Epoch 2/40
23s - loss: 1.1237 - acc: 0.6079 - val_loss: 1.1925 - val_acc: 0.6139
Epoch 3/40
23s - loss: 0.9680 - acc: 0.6610 - val_loss: 0.8761 - val_acc: 0.6909
...
Epoch 40/40
23s - loss: 0.2557 - acc: 0.9087 - val_loss: 0.5634 - val_acc: 0.8236
[INFO] evaluating network...
             precision    recall  f1-score   support

   airplane       0.88      0.81      0.85      1000
 automobile       0.93      0.89      0.91      1000
       bird       0.83      0.68      0.75      1000
        cat       0.69      0.65      0.67      1000
       deer       0.74      0.85      0.79      1000
        dog       0.72      0.77      0.74      1000
       frog       0.85      0.89      0.87      1000
      horse       0.85      0.87      0.86      1000
       ship       0.89      0.91      0.90      1000
      truck       0.88      0.91      0.90      1000

avg / total       0.83      0.82      0.82     10000
```

在我的 GPU 上，纪元相当快，23 秒。在我的 CPU 上，纪元要长得多，达到 171 秒。

训练完成后，我们可以看到 MiniVGGNet 在 CIFAR-10 数据集*上通过*批量归一化获得了 **83%** 的分类准确率——这一结果大大高于在 [**单独教程**](https://pyimagesearch.com/2021/05/22/a-gentle-guide-to-training-your-first-cnn-with-keras-and-tensorflow/) 中应用 ShallowNet 时的 60%准确率。因此，我们看到了更深层次的网络架构如何能够学习更丰富、更有区别的特征。

#### **无批量归一化**

回到`minivggnet.py`实现并注释掉*所有* `BatchNormalization`层，就像这样:

```py
		# first CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		#model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
```

一旦您注释掉了网络中的所有`BatchNormalization`层，请在 CIFAR-10 上重新训练 MiniVGGNet:

```py
$  python minivggnet_cifar10.py \
	--output output/cifar10_minivggnet_without_bn.png 
[INFO] loading CIFAR-10 data...
[INFO] compiling model...
[INFO] training network...
Train on 50000 samples, validate on 10000 samples
Epoch 1/40
13s - loss: 1.8055 - acc: 0.3426 - val_loss: 1.4872 - val_acc: 0.4573
Epoch 2/40
13s - loss: 1.4133 - acc: 0.4872 - val_loss: 1.3246 - val_acc: 0.5224
Epoch 3/40
13s - loss: 1.2162 - acc: 0.5628 - val_loss: 1.0807 - val_acc: 0.6139
...
Epoch 40/40
13s - loss: 0.2780 - acc: 0.8996 - val_loss: 0.6466 - val_acc: 0.7955
[INFO] evaluating network...
             precision    recall  f1-score   support

   airplane       0.83      0.80      0.82      1000
 automobile       0.90      0.89      0.90      1000
       bird       0.75      0.69      0.71      1000
        cat       0.64      0.57      0.61      1000
       deer       0.75      0.81      0.78      1000
        dog       0.69      0.72      0.70      1000
       frog       0.81      0.88      0.85      1000
      horse       0.85      0.83      0.84      1000
       ship       0.90      0.88      0.89      1000
      truck       0.84      0.89      0.86      1000

avg / total       0.79      0.80      0.79     10000
```

您将注意到的第一件事是，您的网络在没有批量标准化的情况下训练*更快*(13s 比 23s，减少了 43%)。然而，一旦网络完成训练，你会注意到一个较低的分类准确率 **79%** 。

当我们在**图 2** 中并排绘制 MiniVGGNet *与*批量归一化(*左*)和*与*批量归一化(*右*)时，我们可以看到批量归一化对训练过程的积极影响:

注意没有批量归一化的 MiniVGGNet 的损失*如何开始增加*超过第 30 个时期，表明网络对训练数据*过度拟合*。我们还可以清楚地看到，到第 25 个纪元时，验证精度已经相当饱和。

另一方面，带有批处理规范化的 MiniVGGNet 实现*更加稳定。虽然损失和准确性在 35 年后开始持平，但我们没有过度拟合得那么糟糕——这是我建议对您自己的网络架构应用批量标准化的众多原因之一。*

## **总结**

在本教程中，我们讨论了卷积神经网络的 VGG 家族。CNN 可以被认为是 VGG 网络，如果:

1.  它仅使用*3*×*3 过滤器，而不考虑网络深度。*
2.  在单个`POOL`操作的之前*应用了多个* `CONV => RELU`层，随着网络深度的增加，有时会有更多的`CONV => RELU`层堆叠在彼此之上。

然后我们实现了一个受 VGG 启发的网络，恰当地命名为`MiniVGGNet`。这个网络架构由两组`(CONV => RELU) * 2) => POOL`层和一组`FC => RELU => FC => SOFTMAX`层组成。我们还在每次激活后应用了批量标准化，并在每个池和完全连接层后应用了丢弃。为了评估 MiniVGGNet，我们使用了 CIFAR-10 数据集。

我们之前在 CIFAR-10 上最好的准确率只有 60%来自浅网( [**早前教程**](https://pyimagesearch.com/2021/05/22/a-gentle-guide-to-training-your-first-cnn-with-keras-and-tensorflow/) )。然而，使用 MiniVGGNet 我们能够将精确度一直提高到 ***83%*** 。

最后，我们检查了批处理规范化在深度学习和 CNN*中扮演的角色，使用*批处理规范化，MiniVGGNet 达到了 83%的分类准确率——但没有批处理规范化的*，准确率下降到 79%(我们也开始看到过度拟合的迹象)。*

因此，这里的要点是:

1.  批量标准化可以导致更快、更稳定的收敛和更高的精度。
2.  然而，这种优势是以训练时间为代价的——批量标准化将需要更多的“墙时间”来训练网络，尽管网络将在更少的时期内获得更高的精度。

也就是说，额外的培训时间往往超过了负面影响，我*强烈鼓励*您对自己的网络架构应用批量规范化。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******