# LeNet:识别手写数字

> 原文：<https://pyimagesearch.com/2021/05/22/lenet-recognizing-handwritten-digits/>

LeNet 架构是深度学习社区中的一项开创性工作，由 [LeCun 等人在其 1998 年的论文*基于梯度的学习应用于文档识别*](https://ieeexplore.ieee.org/document/726791) 中首次介绍。正如论文名所示，作者实现 LeNet 的动机主要是为了光学字符识别(OCR)。

LeNet 架构*简单*和*小*(就内存占用而言)，使其*成为教授 CNN 基础知识的完美*。

在本教程中，我们将试图重复类似于 LeCun 在 1998 年论文中的实验。我们将从回顾 LeNet 架构开始，然后使用 Keras 实现网络。最后，我们将在 MNIST 数据集上评估 LeNet 的手写数字识别性能。

**要了解更多 LeNet 架构，** ***继续阅读。***

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

## **LeNet 架构**

LeNet 架构(**图 2** )是第一个优秀的“真实世界”网络。这个网络很小，很容易理解——但是足够大，可以提供有趣的结果。

此外，LeNet + MNIST 的组合能够轻松地在 CPU 上运行，使初学者能够轻松地迈出深度学习和 CNN 的第一步。在许多方面，LeNet + MNIST 是应用于图像分类的深度学习的*“你好，世界”*。LeNet 架构由以下层组成，使用来自 **[卷积神经网络(CNN)的`CONV => ACT => POOL`模式和](https://pyimagesearch.com/2021/05/14/convolutional-neural-networks-cnns-and-layer-types/)** 层类型:

```py
INPUT => CONV => TANH => POOL => CONV => TANH => POOL =>
	FC => TANH => FC
```

注意 LeNet 架构如何使用 *tanh* 激活函数，而不是更流行的 *ReLU* 。早在 1998 年，ReLU 还没有在深度学习的背景下使用——更常见的是使用 *tanh* 或 *sigmoid* 作为激活函数。当今天实现 LeNet 时，用`RELU`替换`TANH`是很常见的。

**表 1** 总结了 LeNet 架构的参数。我们的*输入层*采用 28 行、28 列的输入图像，深度(即 MNIST 数据集中图像的尺寸)采用单通道(灰度)。然后我们学习 20 个滤镜，每个滤镜都是 5 *×* 5。`CONV`层之后是 ReLU 激活，然后是最大池，大小为 2 *×* 2，步幅为 2 *×* 2。

该架构的下一个模块遵循相同的模式，这次学习 50 个 5*×5*过滤器。随着实际空间输入维度*减少*，通常会看到网络更深层的`CONV`层*的数量增加*。

然后我们有两个`FC`层。第一个`FC`包含 500 个隐藏节点，随后是一个 ReLU 激活。最后的`FC`层控制输出类标签的数量(0-9；一个用于可能的十个数字中的每一个)。最后，我们应用 softmax 激活来获得类别概率。

### 实现 LeNet

给定**表 1** ，我们现在准备使用 Keras 库实现开创性的 LeNet 架构。首先在`pyimagesearch.nn.conv`子模块中添加一个名为`lenet.py`的新文件——这个文件将存储我们实际的 LeNet 实现:

```py
--- pyimagesearch
|    |--- __init__.py
|    |--- nn
|    |    |--- __init__.py
...
|    |    |--- conv
|    |    |    |--- __init__.py
|    |    |    |--- lenet.py
|    |    |    |--- shallownet.py
```

从那里，打开`lenet.py`，我们可以开始编码:

```py
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
```

**第 2-8 行**处理导入我们需要的 Python 包。这些导入与[之前的教程](https://pyimagesearch.com/2021/05/22/a-gentle-guide-to-training-your-first-cnn-with-keras-and-tensorflow/)中的 ShallowNet 实现完全相同，并且在使用 Keras 构建(几乎)任何 CNN 时，形成了必需导入的基本集合。

然后我们定义下面`LeNet`的`build`方法，用于实际构建网络架构:

```py
class LeNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
```

构建方法需要四个参数:

1.  输入图像的*宽度*。
2.  输入图像的*高度*。
3.  图像的*通道数*(深度)。
4.  分类任务中的编号*类别标签*。

在**行 14** 上初始化`Sequential`类，顺序网络的构建块一层接一层地顺序堆叠。然后我们初始化`inputShape`,就像使用“信道最后”排序一样。在我们的 Keras 配置被设置为使用“通道优先”排序的情况下，我们更新第 18 行**和第 19 行**上的`inputShape`。

第一组`CONV => RELU => POOL`层定义如下:

```py
  		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
```

我们的`CONV`层将学习 20 个过滤器，每个大小为 5 *×* 5。然后，我们应用一个 ReLU 激活函数，之后是一个 2 *×* 2 池，步长为 2 *×* 2，从而将输入卷大小减少了 75%。

然后应用另一组`CONV => RELU => POOL`层，这次学习 50 个过滤器而不是 20 个:

```py
  		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
```

然后可以展平输入体积，并且可以应用具有 500 个节点的完全连接的层:

```py
  		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))
```

接着是最终的 softmax 分类器:

```py
  		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
```

现在我们已经编写了 LeNet 架构，我们可以继续将其应用于 MNIST 数据集。

### MNIST 的莱内特

我们的下一步是创建一个驱动程序脚本，它负责:

1.  从磁盘加载 MNIST 数据集。
2.  实例化 LeNet 架构。
3.  培训 LeNet。
4.  评估网络性能。

为了在 MNIST 上训练和评估 LeNet，创建一个名为`lenet_mnist.py`的新文件，我们可以开始了:

```py
# import the necessary packages
from pyimagesearch.nn.conv import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
```

此时，我们的 Python 导入应该开始感觉非常标准，出现了一个明显的模式。在绝大多数机器学习的情况下，我们必须导入:

1.  一个*我们要训练的网络架构*。
2.  一个*优化器*来训练网络(在这种情况下，SGD)。
3.  一组便利函数，用于构造给定数据集的训练和测试拆分。
4.  计算分类报告的功能，这样我们就可以评估我们的分类器的性能。

同样，几乎所有的例子都将遵循这种导入模式，除此之外还有一些额外的类来帮助完成某些任务(比如预处理图像)。MNIST 数据集已经过预处理，因此我们可以通过以下函数调用进行加载:

```py
# grab the MNIST dataset (if this is your first time using this
# dataset then the 11MB download may take a minute)
print("[INFO] accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
```

**第 14 行**从磁盘加载 MNIST 数据集。如果这是您第一次调用`mnist.load_data()`函数，那么需要从 Keras 数据集存储库中下载 MNIST 数据集。MNIST 数据集被序列化为一个 11MB 的文件，因此根据您的互联网连接，下载可能需要几秒到几分钟的时间。

需要注意的是，`data`中的每个 MNIST 样本都由一个 28*×28*灰度图像的 784-d 矢量(即原始像素亮度)表示。因此，我们需要根据我们使用的是“信道优先”还是“信道最后”排序来重塑`data`矩阵:

```py
# if we are using "channels first" ordering, then reshape the
# design matrix such that the matrix is:
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
	trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
	testData = testData.reshape((testData.shape[0], 1, 28, 28))

# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
	trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
	testData = testData.reshape((testData.shape[0], 28, 28, 1))
```

如果我们正在执行“通道优先”排序(**第 20 行和第 21 行**)，那么`data`矩阵被重新整形，使得样本数是矩阵中的第一个条目，单个通道是第二个条目，随后是行数和列数(分别为 28 和 28)。否则，我们假设我们使用“信道最后”排序，在这种情况下，矩阵首先被重新整形为样本数、行数、列数，最后是信道数(**第 26 和 27 行**)。

既然我们的数据矩阵已经成形，我们可以将图像像素强度调整到范围`[0, 1]`:

```py
# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# convert the labels from integers to vectors
le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)
```

我们还将我们的类标签编码为一个热点向量，而不是单个整数值。例如，如果给定样本的类别标签是`3`，那么对标签进行一键编码的输出将是:

`[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`

注意向量中的所有条目都是零*，除了第四个索引的*现在被设置为 1(记住数字`0`是第一个索引，因此为什么第三个是第四个索引*)。现在的舞台是在 MNIST 训练 LeNet:*

```py
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainData, trainLabels,
	validation_data=(testData, testLabels), batch_size=128,
	epochs=20, verbose=1)
```

**第 40 行**用学习率`0.01`初始化我们的`SGD`优化器。`LeNet`本身在**第 41 行**被实例化，表明我们数据集中的所有输入图像都将是`28`像素宽，`28`像素高，深度为`1`。假设 MNIST 数据集中有 10 个类(每个数字对应一个类，0*-*9)，我们设置`classes=10`。

**第 42 行和第 43 行**使用交叉熵损失作为我们的损失函数来编译模型。**47-49 线**使用`128`的小批量在 MNIST 训练 LeNet 总共`20`个时期。

最后，我们可以评估我们网络的性能，并在下面的最终代码块中绘制损耗和精度随时间的变化图:

```py
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testData, batch_size=128)
print(classification_report(testLabels.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
```

我之前在一个 [**的前一课**](https://pyimagesearch.com/2021/05/22/a-gentle-guide-to-training-your-first-cnn-with-keras-and-tensorflow/) 中提到过这个事实，当评估 ShallowNet 时，但是要确保你理解当`model.predict`被调用时 **Line 53** 在做什么。对于`testX`中的每个样本，构建 128 个批量，然后通过网络进行分类。在所有测试数据点被分类后，返回`predictions`变量。

`predictions`变量实际上是一个形状为`(len(testX), 10)`的 NumPy 数组，表示我们现在有 10 个概率与`testX`中每个*数据点的*每个*类标签相关联。取**第 54-56 行**的`classification_report`中的`predictions.argmax(axis=1)`，找到*概率最大*的标签的索引(即最终输出分类)。给定来自网络的最终分类，我们可以将我们的*预测*类别标签与*基础事实*标签进行比较。*

要执行我们的脚本，只需发出以下命令:

```py
$ python lenet_mnist.py
```

然后，应该从磁盘下载和/或加载 MNIST 数据集，并开始训练:

```py
[INFO] accessing MNIST...
[INFO] compiling model...
[INFO] training network...
Train on 52500 samples, validate on 17500 samples
Epoch 1/20
3s - loss: 1.0970 - acc: 0.6976 - val_loss: 0.5348 - val_acc: 0.8228
...
Epoch 20/20
3s - loss: 0.0411 - acc: 0.9877 - val_loss: 0.0576 - val_acc: 0.9837
[INFO] evaluating network...
             precision    recall  f1-score   support

          0       0.99      0.99      0.99      1677
          1       0.99      0.99      0.99      1935
          2       0.99      0.98      0.99      1767
          3       0.99      0.97      0.98      1766
          4       1.00      0.98      0.99      1691
          5       0.99      0.98      0.98      1653
          6       0.99      0.99      0.99      1754
          7       0.98      0.99      0.99      1846
          8       0.94      0.99      0.97      1702
          9       0.98      0.98      0.98      1709

avg / total       0.98      0.98      0.98     17500
```

使用我的 Titan X GPU，我获得了三秒的时间。使用*仅仅*CPU，每个纪元的秒数跃升至 30 秒。训练完成后，我们可以看到 LeNet 获得了 **98%** 的分类精度，比使用标准前馈神经网络时的*92%有了巨大的提高。*

此外，在图 3 的**中查看我们的损耗和精度随时间的变化曲线，可以看出我们的网络表现良好。仅仅过了五个纪元，*已经*达到了 *≈* 96%的分类准确率。由于我们的学习率保持不变且不衰减，训练和验证数据的损失继续下降，只有少数轻微的“峰值”。在第二十个纪元结束时，我们的测试集达到了 98%的准确率。**

这张展示 MNIST LeNet 的损失和准确性的图可以说是我们正在寻找的*典型的*图:训练和验证损失和准确性(几乎)完全相互模拟，没有过度拟合的迹象。正如我们将看到的，通常*很难*获得这种表现如此良好的训练图，这表明我们的网络正在学习底层模式*而没有*过度拟合。

还有一个问题是，MNIST 数据集*经过了大量预处理*，不能代表我们在现实世界中会遇到的图像分类问题。研究人员倾向于使用 MNIST 数据集作为基准来评估新的分类算法。如果他们的方法不能获得 *>* 95%的分类准确率，那么要么是(1)算法的逻辑有缺陷，要么是(2)实现本身有缺陷。

尽管如此，将 LeNet 应用于 MNIST 是一种很好的方式，可以让你首次尝试将深度学习应用于图像分类问题，并模仿 LeCun 等人论文的结果。

## **总结**

在本教程中，我们探索了 LeNet 架构，该架构由 [LeCun 等人在其 1998 年的论文*基于梯度的学习应用于文档识别*](https://ieeexplore.ieee.org/document/726791) 中介绍。LeNet 是深度学习文献中的一项开创性工作——它彻底展示了如何训练神经网络以端到端的方式识别图像中的对象(即，不需要进行特征提取，网络能够从图像本身学习模式)。

尽管 LeNet 具有开创性，但以今天的标准来看，它仍被认为是一个“肤浅”的网络。只有四个可训练层(两个`CONV`层和两个`FC`层)，LeNet 的深度与当前最先进的架构如 VGG (16 和 19 层)和 ResNet (100+层)的深度相比相形见绌。

在我们的[下一个**教程**](https://pyimagesearch.com/2021/05/22/minivggnet-going-deeper-with-cnns/) 中，我们将讨论 VGGNet 架构的一个变种，我称之为*“迷你 VGGNet”*这种架构的变体使用了与 [Simonyan 和 Zisserman 的工作](https://arxiv.org/abs/1409.1556)完全相同的指导原则，但降低了深度，允许我们在更小的数据集上训练网络。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****