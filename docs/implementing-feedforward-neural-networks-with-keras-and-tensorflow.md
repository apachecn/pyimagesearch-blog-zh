# 用 Keras 和 TensorFlow 实现前馈神经网络

> 原文：<https://pyimagesearch.com/2021/05/06/implementing-feedforward-neural-networks-with-keras-and-tensorflow/>

既然我们已经在 *pure Python* 中实现了神经网络，那么让我们转到首选的实现方法——使用一个专用的(高度优化的)神经网络库，比如 Keras。

今天，我将讨论如何实现前馈多层网络，并将其应用于 MNIST 和 CIFAR-10 数据集。这些结果很难说是“最先进”，但有两个用途:

*   演示如何使用 Keras 库实现简单的神经网络。
*   使用标准神经网络获得基线，我们稍后将与卷积神经网络进行比较(注意 CNN 将*显著*优于我们之前的方法)。

### 赔偿

今天，我们将使用由 70，000 个数据点组成的*完整* MNIST 数据集(每位数 7，000 个示例)。每个数据点由一个 784 维向量表示，对应于 MNIST 数据集中的(展平的) *28×28* 图像。我们的目标是训练一个神经网络(使用 Keras)在这个数据集上获得 *> 90%* 的准确率。

我们将会发现，使用 Keras 来构建我们的网络架构比我们的纯 Python 版本要简单得多。事实上，实际的网络架构将只占用*四行代码*——本例中剩余的代码只涉及从磁盘加载数据、转换类标签，然后显示结果。

首先，打开一个新文件，将其命名为`keras_mnist.py`，并插入以下代码:

```py
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse
```

**第 2-11 行**导入我们需要的 Python 包。`LabelBinarizer`将用于将我们的*整数标签*一次性编码为*向量标签*。一键编码将分类标签从单个整数转换为向量。许多机器学习算法(包括神经网络)受益于这种类型的标签表示。我将在本节的后面更详细地讨论一键编码，并提供多个例子(包括使用`LabelBinarizer`)。

`classification_report`函数将给我们一个格式良好的报告，显示我们模型的总精度，以及每个数字的分类精度。

**第 4-6 行**导入必要的包，用 Keras 创建一个简单的前馈神经网络。`Sequential`类表示我们的网络将是前馈的，各层将按顺序添加到类*中，一层在另一层之上。**行 5** 上的`Dense`类是我们全连接层的实现。为了让我们的网络真正学习，我们需要应用`SGD` ( **行 6** )来优化网络的参数。最后，为了访问完整的 MNIST 数据集，我们需要在第 7 行的**上导入`mnist`辅助函数。***

让我们继续解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())
```

这里我们只需要一个开关`--output`，它是我们绘制的损耗和精度图保存到磁盘的路径。

接下来，让我们加载完整的 MNIST 数据集:

```py
# grab the MNIST dataset (if this is your first time using this
# dataset then the 11MB download may take a minute)
print("[INFO] accessing MNIST...")
((trainX, trainY), (testX, testY)) = mnist.load_data()

# each image in the MNIST dataset is represented as a 28x28x1
# image, but in order to apply a standard neural network we must
# first "flatten" the image to be simple list of 28x28=784 pixels
trainX = trainX.reshape((trainX.shape[0], 28 * 28 * 1))
testX = testX.reshape((testX.shape[0], 28 * 28 * 1))

# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
```

**第 22 行**从磁盘加载 MNIST 数据集。如果*以前从未*运行过这个功能，那么 MNIST 数据集将被下载并存储到你的本地机器上。一旦数据集被下载，它将被缓存到您的计算机中，并且不必再次下载。

MNIST 数据集中的每幅图像都表示为 *28×28×1* 像素图像。为了在图像数据上训练我们的神经网络，我们首先需要将 2D 图像展平成一个由 *28×28 = 784* 个值组成的平面列表(**第 27 行和第 28 行**)。

然后，我们通过将像素强度缩放到范围 *[0，1]* ，对第 31 行和第 32 行执行数据归一化。

给定训练和测试分割，我们现在可以编码我们的标签:

```py
# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
```

MNIST 数据集中的每个数据点都有一个范围为*【0，9】*的整数标签，MNIST 数据集中可能的十位数字中的每一位都有一个标签。值为`0`的标签表示对应的图像包含一个零数字。同样，值为`8`的标签表示相应的图像包含数字 8。

但是，我们首先需要将这些*整数标签*转换成*向量标签*，其中标签向量中的索引设置为`1`，否则设置为`0`(这个过程称为*一键编码*)。

例如，考虑标签`3`，我们希望对其进行二进制化/一键编码——标签`3`现在变成:

```py
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

请注意，只有数字 3 的索引被设置为 1，向量中的所有其他条目都被设置为零。精明的读者可能会奇怪为什么向量中的第四个条目*而不是第三个*条目*被更新？回想一下，标签中的第一个条目实际上是数字 0。因此，数字 3 的条目实际上是列表中的第四个索引。*

下面是第二个例子，这次标签`1`被二进制化:

```py
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
```

向量中的第二个条目被设置为 1(因为第一个条目对应于标签`0`)，而所有其他条目被设置为零。

我在下面的清单中包含了每个数字的独热编码表示，*0-9*:

```py
0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

这种编码可能看起来很乏味，但许多机器学习算法(包括神经网络)都受益于这种标签表示。幸运的是，大多数机器学习软件包提供了一种方法/功能来执行一键编码，消除了许多繁琐。

**第 35-37 行**简单地执行对训练集和测试集的输入*整数标签*进行一次性编码为*向量* *标签*的过程。

接下来，让我们定义我们的网络架构:

```py
# define the 784-256-128-10 architecture using Keras
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))
```

正如你所看到的，我们的网络是一个前馈架构，由第 40 行的**类实例化——这种架构意味着各层将堆叠在彼此之上，前一层的输出馈入下一层。**

**第 41 行**定义了网络中第一个完全连接的层。将`input_shape`设置为`784`，每个 MNIST 数据点的维数。然后，我们在这一层学习 256 个权重，并应用 sigmoid 激活函数。下一层(**第 42 行**)学习 128 个重量。最后，**行 43** 应用另一个全连接层，这次只学习 10 个权重，对应于十个(0-9)输出类。代替 sigmoid 激活，我们将使用 softmax 激活来获得每个预测的归一化类别概率。

让我们继续训练我们的网络:

```py
# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=100, batch_size=128)
```

在**第 47 行，**我们用学习率`0.01`(我们通常可以写成`1e-2`)初始化 SGD 优化器。我们将使用类别交叉熵损失函数作为我们的损失度量(**第 48 和 49 行**)。使用交叉熵损失函数也是我们必须将整数标签转换为向量标签的原因。

对**线 50 和 51** 上的`model`的`.fit`的调用启动了我们神经网络的训练。我们将提供训练数据和训练标签作为该方法的前两个参数。

然后可以提供`validation_data`，这是我们的测试分割。在*大多数*情况下，比如当你调整超参数或者决定一个模型架构时，你会希望你的验证集是一个*真的*验证集，而不是你的测试数据。在这种情况下，我们只是演示如何使用 Keras 从头开始训练神经网络，所以我们对我们的指导方针有点宽容。

我们将允许我们的网络一次使用 128 个数据点的批量来训练总共 100 个纪元。该方法返回一个字典`H`，我们将使用它在几个代码块中绘制网络随时间的损失/准确性。

一旦网络完成训练，我们将需要根据测试数据对其进行评估，以获得我们的最终分类:

```py
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in lb.classes_]))
```

对`model`的`.predict`方法的调用将返回`testX` ( **第 55 行**)中每个数据点的*的类标签概率。因此，如果您要检查`predictions` NumPy 数组，它将具有形状`(X, 10)`，因为在测试集中有 17500 个数据点和 10 个可能的类别标签(数字 0-9)。*

因此，给定行中的每个条目都是一个*概率*。为了确定具有最大概率的*的类，我们可以像在**行 56** 上一样简单地调用`.argmax(axis=1)`，这将给出具有最大概率的类标签的*索引*，并因此给出我们最终的输出分类。网络的最终输出分类被制成表格，然后最终分类报告在**行 56-58** 上显示给我们的控制台。*

我们的最终代码块处理随时间绘制训练损失、训练精度、验证损失和验证精度:

```py
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
```

然后根据`--output`命令行参数将该图保存到磁盘。要在 MNIST 上训练我们的全连接层网络，只需执行以下命令:

```py
$ python keras_mnist.py --output output/keras_mnist.png
[INFO] loading MNIST (full) dataset...
[INFO] training network...
Train on 52500 samples, validate on 17500 samples
Epoch 1/100
1s - loss: 2.2997 - acc: 0.1088 - val_loss: 2.2918 - val_acc: 0.1145
Epoch 2/100
1s - loss: 2.2866 - acc: 0.1133 - val_loss: 2.2796 - val_acc: 0.1233
Epoch 3/100
1s - loss: 2.2721 - acc: 0.1437 - val_loss: 2.2620 - val_acc: 0.1962
...
Epoch 98/100
1s - loss: 0.2811 - acc: 0.9199 - val_loss: 0.2857 - val_acc: 0.9153
Epoch 99/100
1s - loss: 0.2802 - acc: 0.9201 - val_loss: 0.2862 - val_acc: 0.9148
Epoch 100/100
1s - loss: 0.2792 - acc: 0.9204 - val_loss: 0.2844 - val_acc: 0.9160
[INFO] evaluating network...
             precision    recall  f1-score   support

        0.0       0.94      0.96      0.95      1726
        1.0       0.95      0.97      0.96      2004
        2.0       0.91      0.89      0.90      1747
        3.0       0.91      0.88      0.89      1828
        4.0       0.91      0.93      0.92      1686
        5.0       0.89      0.86      0.88      1581
        6.0       0.92      0.96      0.94      1700
        7.0       0.92      0.94      0.93      1814
        8.0       0.88      0.88      0.88      1679
        9.0       0.90      0.88      0.89      1735

avg / total       0.92      0.92      0.92     17500
```

如结果所示，我们获得了 *≈92%* 的准确度。此外，训练和验证曲线彼此匹配*几乎相同* ( **图 1** )，表明训练过程没有过度拟合或问题。

事实上，如果你不熟悉 MNIST 数据集，你可能会认为 92%的准确率是非常好的——这可能是在 20 年前。利用卷积神经网络，我们可以很容易地获得 *> 98%* 的准确率。目前最先进的方法甚至可以突破 99%的准确率。

虽然表面上看起来我们的(严格)全连接网络运行良好，但实际上我们可以做得更好。正如我们将在下一节中看到的，应用于更具挑战性的数据集的严格全连接网络在某些情况下比随机猜测好不了多少。

### **CIFAR-10**

当谈到计算机视觉和机器学习时，MNIST 数据集是“基准”数据集的经典定义，这种数据集太容易获得高精度的结果，并且不代表我们在现实世界中看到的图像。

对于一个更具挑战性的基准数据集，我们通常使用 CIFAR-10，这是一个由 60，000 张 *32×32* RGB 图像组成的集合，这意味着数据集中的每张图像都由 *32×32×3 = 3，072* 个整数表示。顾名思义，CIFAR-10 由 10 类组成，包括*飞机*、*汽车*、*鸟*、*猫*、*鹿*、*狗*、*蛙*、*马*、*船*、*卡车*。在图 2 的**中可以看到每个类别的 CIFAR-10 数据集样本。**

每类平均表示为每类 6，000 个图像。在 CIFAR-10 上训练和评估机器学习模型时，通常使用作者预定义的数据分割，并使用 50，000 张图像进行训练，10，000 张图像进行测试。

CIFAR-10 比 MNIST 数据集要硬得多。挑战来自于物体呈现方式的巨大差异。例如，我们不能再假设在给定的 *(x，y)*-坐标包含绿色像素的图像是一只青蛙。这个像素可以是包含鹿的森林的背景。或者它可以是绿色汽车或卡车的颜色。

这些假设与 MNIST 数据集形成鲜明对比，在后者中，网络可以学习关于像素强度空间分布的假设。例如， *1* 的前景像素的空间分布与 *0* 或 *5* 的前景像素的空间分布有很大不同。对象外观的这种变化使得应用一系列完全连接的层变得更加困难。正如我们将在本节的其余部分发现的，标准`FC`(全连接)层网络不适合这种类型的图像分类。

让我们开始吧。打开一个新文件，将其命名为`keras_cifar10.py`，并插入以下代码:

```py
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
```

**第 2-10 行**导入我们需要的 Python 包来构建我们完全连接的网络，与上一节的 MNIST 相同。例外情况是**第 7 行**上的特殊实用函数——由于 CIFAR-10 是一个如此常见的数据集，研究人员在其上对机器学习和深度学习算法进行基准测试，因此经常看到深度学习库提供简单的助手函数来*自动*从磁盘加载该数据集。

接下来，我们可以分析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())
```

我们需要的唯一命令行参数是`--output`，这是输出损耗/精度图的路径。

让我们继续加载 CIFAR-10 数据集:

```py
# load the training and testing data, scale it into the range [0, 1],
# then reshape the design matrix
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))
```

对**行 21** 上的`cifar10.load_data`的调用自动从磁盘加载 CIFAR-10 数据集，预先分割成训练和测试分割。如果这是您在*第一次*调用`cifar10.load_data`，那么该函数将为您获取并下载数据集。这个文件是 *≈170MB* ，所以在下载和解压的时候要有耐心。一旦文件被下载一次，它将被缓存在您的本地机器上，而不必再次下载。

**第 22 行和第 23 行**将 CIFAR-10 的数据类型从无符号 8 位整数转换为浮点，然后将数据缩放到范围 *[0，1]* 。**线 24 和 25** 负责*重塑*训练和测试数据的设计矩阵。回想一下，CIFAR-10 数据集中的每个图像都由一个 *32×32×3* 图像表示。

例如，`trainX`具有形状`(50000, 32, 32, 3)`，而`testX`具有形状`(10000, 32, 32, 3)`。如果我们要*将*这个图像展平成一个浮点值列表，那么列表中总共会有*32×32×3 = 3072*个条目。

为了展平训练和测试集中的每个图像，我们只需使用 NumPy 的`.reshape`函数。这个函数执行后，`trainX`现在有了`(50000, 3072)`的形状，而`testX`有了`(10000, 3072)`的形状。

既然已经从磁盘加载了 CIFAR-10 数据集，让我们再次将类标签整数二进制化为向量，然后初始化类标签的实际*名称*的列表:

```py
# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck"]
```

现在是时候定义网络架构了:

```py
# define the 3072-1024-512-10 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))
```

**第 37 行**实例化了`Sequential`类。然后，我们添加第一个`Dense`层，它的`input_shape`为`3072`，是设计矩阵中 3072 个展平像素值中每一个的节点——该层负责学习 1024 个权重。我们还将把过时的 sigmoid 换成 ReLU activation，希望能提高网络性能。

下一个完全连接的层(**行 39** )学习 512 个权重，而最后一层(**行 40** )学习对应于十个可能的输出分类的权重，以及 softmax 分类器，以获得每个分类的最终输出概率。

既然已经定义了网络的架构，我们就可以训练它了:

```py
# train the model using SGD
print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=100, batch_size=32)
```

我们将使用 SGD 优化器以`0.01`的学习速率训练网络，这是一个相当标准的初始选择。该网络将使用每批 32 个来训练总共 100 个时期。

一旦训练好网络，我们可以使用`classification_report`对其进行评估，以获得对模型性能的更详细的回顾:

```py
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))
```

最后，我们还将绘制一段时间内的损耗/精度图:

```py
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
```

要在 CIFAR-10 上训练我们的网络，请打开一个终端并执行以下命令:

```py
$ python keras_cifar10.py --output output/keras_cifar10.png
[INFO] training network...
Train on 50000 samples, validate on 10000 samples
Epoch 1/100
7s - loss: 1.8409 - acc: 0.3428 - val_loss: 1.6965 - val_acc: 0.4070
Epoch 2/100
7s - loss: 1.6537 - acc: 0.4160 - val_loss: 1.6561 - val_acc: 0.4163
Epoch 3/100
7s - loss: 1.5701 - acc: 0.4449 - val_loss: 1.6049 - val_acc: 0.4376
...
Epoch 98/100
7s - loss: 0.0292 - acc: 0.9969 - val_loss: 2.2477 - val_acc: 0.5712
Epoch 99/100
7s - loss: 0.0272 - acc: 0.9972 - val_loss: 2.2514 - val_acc: 0.5717
Epoch 100/100
7s - loss: 0.0252 - acc: 0.9976 - val_loss: 2.2492 - val_acc: 0.5739
[INFO] evaluating network...
             precision    recall  f1-score   support

   airplane       0.63      0.66      0.64      1000
 automobile       0.69      0.65      0.67      1000
       bird       0.48      0.43      0.45      1000
        cat       0.40      0.38      0.39      1000
       deer       0.52      0.51      0.51      1000
        dog       0.48      0.47      0.48      1000
       frog       0.64      0.63      0.64      1000
      horse       0.63      0.62      0.63      1000
       ship       0.64      0.74      0.69      1000
      truck       0.59      0.65      0.62      1000

avg / total       0.57      0.57      0.57     10000
```

查看输出，您可以看到我们的网络获得了 57%的准确率。检查我们的损失和准确性随时间的变化图(**图 3** )，我们可以看到我们的网络与过去的纪元 10 的过度拟合作斗争。亏损最初开始减少，稍微持平，然后飙升，再也没有下降。与此同时，培训损失也在不断下降。这种*减少*训练损失而*增加*的行为表明*极度过拟合*。

我们当然可以考虑进一步优化我们的超参数，特别是，尝试不同的学习速率，增加网络节点的深度和数量，但我们将为微薄的收益而战。

事实是，具有严格全连接层的基本前馈网络不适合挑战性的图像数据集。为此，我们需要一种更先进的方法:卷积神经网络。