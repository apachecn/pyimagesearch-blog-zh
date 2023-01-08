# 用 Keras 和 TensorFlow 训练你的第一个 CNN 的温和指南

> 原文：<https://pyimagesearch.com/2021/05/22/a-gentle-guide-to-training-your-first-cnn-with-keras-and-tensorflow/>

在本教程中，您将使用 Python 和 Keras 实现一个 CNN。我们将从快速回顾 Keras 配置开始，在构建和训练您自己的 CNN 时，您应该记住这些配置。

然后我们将实现 ShallowNet，顾名思义，它是一个非常浅的 CNN，只有一个层。然而，不要让这个网络的简单性欺骗了你——正如我们的结果将证明的那样，与许多其他方法相比，ShallowNet 能够在 CIFAR-10 和 Animals 数据集上获得更高的分类精度。

***注:*** 本教程要求你[下载动物数据集。](https://pis-datasets.s3.us-east-2.amazonaws.com/animals.zip)

## **Keras 配置和将图像转换为数组**

在我们实现`ShallowNet`之前，我们首先需要回顾一下`keras.json`配置文件，以及这个文件中的设置将如何影响您实现自己的 CNN。我们还将实现第二个名为`ImageToArrayPreprocessor`的图像预处理器，它接受输入图像，然后将其转换为 Keras 可以处理的 NumPy 数组。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **图像到数组预处理器**

正如我上面提到的，Keras 库提供了接受输入图像的`img_to_array`函数，然后根据我们的`image_data_format`设置对通道进行适当的排序。我们将把这个函数封装在一个名为`ImageToArrayPreprocessor`的新类中。创建一个具有特殊`preprocess`功能的类将允许我们创建预处理程序的“链”,以有效地准备用于训练和测试的图像。

为了创建我们的图像到数组预处理器，在`pyimagesearch`的`preprocessing`子模块中创建一个名为`imagetoarraypreprocessor.py`的新文件:

```py
|--- pyimagesearch
|    |--- __init__.py
|    |--- datasets
|    |    |--- __init__.py
|    |    |--- simpledatasetloader.py
|    |--- preprocessing
|    |    |--- __init__.py
|    |    |--- imagetoarraypreprocessor.py
|    |    |--- simplepreprocessor.py

```

从那里，打开文件并插入以下代码:

```py
# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
	def __init__(self, dataFormat=None):
		# store the image data format
		self.dataFormat = dataFormat

	def preprocess(self, image):
		# apply the Keras utility function that correctly rearranges
		# the dimensions of the image
		return img_to_array(image, data_format=self.dataFormat)
```

**第 2 行**从 Keras 导入`img_to_array`函数。

然后，我们在第 5-7 行的**中定义我们的`ImageToArrayPreprocessor`类的构造函数。构造函数接受一个名为`dataFormat`的可选参数。该值默认为`None`，表示应该使用`keras.json`内的设置。我们也可以显式地提供一个`channels_first`或`channels_last`字符串，但是最好让 Keras 根据配置文件选择使用哪个图像维度排序。**

最后，我们在第 9-12 行的**上有`preprocess`函数。这种方法:**

1.  接受一个`image`作为输入。
2.  在`image`上调用`img_to_array`，根据我们的配置文件/T2 的值对通道进行排序。
3.  返回一个通道正确排序的新 NumPy 数组。

定义一个*类*来处理这种类型的图像预处理，而不是简单地在每个图像上调用`img_to_array`的好处是，我们现在可以在从磁盘加载数据集时*将*预处理程序链接在一起。

例如，假设我们希望将所有输入图像的大小调整为 32*×32*像素的固定大小。为此，我们需要初始化一个`SimplePreprocessor`:

```py
sp = SimplePreprocessor(32, 32)
```

调整图像大小后，我们需要应用适当的通道排序——这可以使用上面的`ImageToArrayPreprocessor`来完成:

```py
iap = ImageToArrayPreprocessor()
```

现在，假设我们希望从磁盘加载一个图像数据集，并为训练准备数据集中的所有图像。使用`SimpleDatasetLoader`，我们的任务变得非常简单:

```py
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
```

注意我们的图像预处理程序是如何*链接*在一起的，并将按照*的顺序*应用。对于数据集中的每个图像，我们将首先应用`SimplePreprocessor`将其调整为 32 *×* 32 像素。一旦调整了图像的大小，就应用`ImageToArrayPreprocessor`来处理图像通道的排序。该图像处理管道可以在图 2 中可视化。

以这种方式将简单的预处理器链接在一起，其中每个预处理器负责*一个小任务*，这是一种建立可扩展的深度学习库的简单方法，该库专用于分类图像。

## **浅水网**

今天，我们将实现浅水网络架构。顾名思义，浅网架构只包含几层——整个网络架构可以概括为:`INPUT => CONV => RELU => FC`

这个简单的网络架构将允许我们使用 Keras 库实现卷积神经网络。在实现了 ShallowNet 之后，我将把它应用于动物和 CIFAR-10 数据集。正如我们的结果将展示的，CNN 能够*显著地胜过*许多其他的图像分类方法。

### **实施浅水网**

为了保持我们的`pyimagesearch`包的整洁，让我们在`nn`中创建一个新的子模块，命名为`conv`，我们所有的 CNN 实现都将位于其中:

```py
--- pyimagesearch
|    |--- __init__.py
|    |--- datasets
|    |--- nn
|    |    |--- __init__.py
...
|    |    |--- conv
|    |    |    |--- __init__.py
|    |    |    |--- shallownet.py
|    |--- preprocessing
```

在`conv`子模块中，创建一个名为`shallownet.py`的新文件来存储我们的 ShallowNet 架构实现。从那里，打开文件并插入以下代码:

```py
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
```

**第 2-7 行**导入我们需要的 Python 包。`Conv2D`类是卷积层的 Keras 实现。然后我们有了`Activation`类，顾名思义，它处理对输入应用激活函数。在将输入输入到`Dense`(即完全连接的)层之前，`Flatten`类获取我们的多维体积，并将其“展平”成 1D 数组。

在实现网络架构时，我更喜欢将它们定义在一个类中，以保持代码有组织——我们在这里也将这样做:

```py
class ShallowNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last"
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
```

在**第 9 行，**我们定义了`ShallowNet`类，然后在**第 11 行**定义了一个`build`方法。我们在本书中实现的每个 CNN 都将有一个`build`方法——这个函数将接受许多参数，构建网络架构，然后将其返回给调用函数。在这种情况下，我们的`build`方法需要四个参数:

*   `width`:将用于训练网络的输入图像的宽度(即矩阵中的列数)。
*   `height`:我们输入图像的高度(即矩阵中的行数)
*   `depth`:输入图像中的通道数。
*   `classes`:我们的网络要学习预测的总类数。对于动物，`classes=3`和对于 CIFAR-10，`classes=10`。

然后我们在**线 15** 上初始化`inputShape`到网络，假设“信道最后”排序。**第 18 行和第 19 行**检查 Keras 后端是否设置为“通道优先”，如果是，我们更新`inputShape`。通常的做法是为你建立的几乎每一个 CNN 都包含**行 15-19** ，从而确保你的网络将会工作，不管用户如何订购他们的图像频道。

现在我们的`inputShape`已经定义好了，我们可以开始构建浅水网络架构了:

```py
		# define the first (and only) CONV => RELU layer
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
```

在**第 22 行**，我们定义了第一个(也是唯一的)卷积层。该层将有 32 个滤镜( *K* )，每个滤镜是 3 个 *×* 3(即方形 *F×F* 滤镜)。我们将应用`same`填充来确保卷积运算的输出大小与输入匹配(使用`same`填充对于这个例子来说不是绝对必要的，但是现在开始形成是一个好习惯)。卷积后，我们在**线 24** 上应用一个 ReLU 激活。

让我们完成浅水网的构建:

```py
		# softmax classifier
		model.add(Flatten())
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
```

为了应用我们的全连接层，我们首先需要将多维表示展平为 1D 列表。展平操作由**线 27** 上的`Flatten`调用处理。然后，使用与输出类标签相同数量的节点创建一个`Dense`层(**第 28 行**)。**第 29 行**应用了一个 softmax 激活函数，它将给出每个类的类标签概率。浅网架构返回到**线 32** 上的调用函数。

现在已经定义了 ShallowNet，我们可以继续创建实际的“驱动程序脚本”来加载数据集，对其进行预处理，然后训练网络。我们将看两个利用浅水网的例子——动物和 CIFAR-10。

### **动物身上的浅网**

为了在 Animals 数据集上训练 ShallowNet，我们需要创建一个单独的 Python 文件。打开您最喜欢的 IDE，创建一个名为`shallownet_animals.py`的新文件，确保它与我们的`pyimagesearch`模块在同一个目录级别(或者您已经将`pyimagesearch`添加到 Python 解释器/IDE 在运行脚本时将检查的路径列表中)。

从那里，我们可以开始工作:

```py
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
```

**第 2-13 行**导入我们需要的 Python 包。这些导入中的大部分你已经在前面的例子中见过了，但是我想提醒你注意第 5-7 行的**，在这里我们导入了`ImageToArrayPreprocessor`、`SimplePreprocessor`和`SimpleDatasetLoader`——这些类将形成实际的*管道*，用于在图像通过我们的网络之前对它们进行处理。然后，我们将第 8 行的`ShallowNet`和第 9 行的`SGD`一起导入到**行，我们将使用随机梯度下降来训练浅水网。****

 **接下来，我们需要解析我们的命令行参数并获取我们的图像路径:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
```

我们的脚本在这里只需要一个开关`--dataset`，它是包含我们的动物数据集的目录的路径。**第 23 行**然后抓取动物体内所有 3000 张图片的文件路径。

还记得我说过如何创建一个管道来加载和处理我们的数据集吗？现在让我们看看这是如何做到的:

```py
# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0
```

**第 26 行**定义了用于将输入图像调整到 32*×32*像素的`SimplePreprocessor`。然后在**线 27** 上实例化`ImageToArrayPreprocessor`以处理信道排序。

我们在**第 31 行**将这些预处理器组合在一起，在那里我们初始化`SimpleDatasetLoader`。看一下构造函数的`preprocessors`参数——我们提供了一个*列表*,其中列出了将按照*顺序*应用的预处理程序。首先，给定的输入图像将被调整到 32*×32*像素。然后，调整大小后的图像将根据我们的`keras.json`配置文件按顺序排列其通道。**第 32 行**加载图像(应用预处理器)和分类标签。然后，我们将图像缩放到范围*【0，1】*。

既然已经加载了数据和标签，我们就可以执行我们的训练和测试分割，同时对标签进行一次性编码:

```py
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)
```

这里，我们将 75%的数据用于训练，25%用于测试。

下一步是实例化`ShallowNet`，然后训练网络本身:

```py
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=32, epochs=100, verbose=1)
```

我们使用学习率`0.005`初始化**行 46** 上的`SGD`优化器。`ShallowNet`架构在**行 47** 上被实例化，提供 32 像素的宽度和高度以及 3 的深度——这意味着我们的输入图像是 32*×32 像素，具有三个通道。由于 Animals 数据集有三个类标签，我们设置了`classes=3`。*

然后在**的第 48 和 49 行编译`model`，在这里我们将使用交叉熵作为我们的损失函数，SGD 作为我们的优化器。为了训练网络，我们在第 53 和 54 行**的**上调用`model`的`.fit`方法。`.fit`方法要求我们传递训练和测试数据。我们还将提供我们的测试数据，以便我们可以在每个时期后评估 ShallowNet 的性能。将使用 32 的小批量大小对网络进行 100 个时期的训练(这意味着将一次向网络呈现 32 个图像，并且将进行完整的向前和向后传递以更新网络的参数)。**

在训练我们的网络之后，我们可以评估它的性能:

```py
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=["cat", "dog", "panda"]))
```

为了获得测试数据的输出预测，我们调用了`model`的`.predict`。一份格式精美的分类报告显示在我们的屏幕上**第 59-61 行**。

我们的最终代码块处理*和*训练和测试数据的准确性和随时间的损失的绘图:

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
plt.show()
```

要在 Animals 数据集上训练 ShallowNet，只需执行以下命令:

```py
$ python shallownet_animals.py --dataset ../datasets/animals
```

训练应该非常快，因为网络非常浅，我们的图像数据集相对较小:

```py
[INFO] loading images...
[INFO] processed 500/3000
[INFO] processed 1000/3000
[INFO] processed 1500/3000
[INFO] processed 2000/3000
[INFO] processed 2500/3000
[INFO] processed 3000/3000
[INFO] compiling model...
[INFO] training network...
Train on 2250 samples, validate on 750 samples
Epoch 1/100
0s - loss: 1.0290 - acc: 0.4560 - val_loss: 0.9602 - val_acc: 0.5160
Epoch 2/100
0s - loss: 0.9289 - acc: 0.5431 - val_loss: 1.0345 - val_acc: 0.4933
...
Epoch 100/100
0s - loss: 0.3442 - acc: 0.8707 - val_loss: 0.6890 - val_acc: 0.6947
[INFO] evaluating network...
             precision    recall  f1-score   support

        cat       0.58      0.77      0.67       239
        dog       0.75      0.40      0.52       249
      panda       0.79      0.90      0.84       262

avg / total       0.71      0.69      0.68       750
```

由于训练数据量很小，epochs 非常快，在我的 CPU 和 GPU 上都不到一秒钟。

从上面的输出可以看出，在我们的测试数据上，ShallowNet 获得了 71%的分类准确率**，比我们之前使用简单前馈神经网络的 59%有了很大的提高。使用更先进的训练网络，以及更强大的架构，我们将能够进一步提高分类精度。**

 **图 3 中显示了损耗和精度随时间的变化。在 *x* 轴上，我们有我们的纪元编号，在 *y* 轴上，我们有我们的损耗和精度。检查该图，我们可以看到学习有点不稳定，在第 20 个纪元和第 60 个纪元左右损失大，这一结果可能是由于我们的学习率太高。

还要注意，训练和测试损失在超过第 30 个时期后严重偏离，这意味着我们的网络对训练数据*的建模过于接近*并且过度拟合。我们可以通过获取更多数据或应用数据扩充等技术来解决这个问题。

在纪元 60 左右，我们的测试精度饱和——我们无法超过 *≈* 70%的分类精度，同时我们的训练精度继续攀升至 85%以上。同样，收集更多的训练数据，应用数据扩充，并更加注意调整我们的学习速度，将有助于我们在未来改善我们的结果。

这里的关键点是，一个极其简单的卷积神经网络能够在动物数据集上获得 71%的分类准确率，而我们以前的最好成绩仅为 59%——这是超过 12%的改进！

### CIFAR-10 上的浅水网

让我们也将 ShallowNet 架构应用于 CIFAR-10 数据集，看看我们是否可以改进我们的结果。打开一个新文件，将其命名为`shallownet_cifar10.py`，并插入以下代码:

```py
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv import ShallowNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

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

**第 2-8 行**导入我们需要的 Python 包。然后，我们加载 CIFAR-10 数据集(预先分为训练集和测试集)，然后将图像像素强度缩放到范围*【0，1】*。由于 CIFAR-10 图像经过预处理，并且通道排序是在`cifar10.load_data`内部自动处理的*，我们不需要应用任何自定义预处理类。*

 *然后我们的标签被一次性编码成第 18-20 行上的向量。我们还在第 23 行和第 24 行的**上初始化了 CIFAR-10 数据集的标签名称。**

现在我们的数据准备好了，我们可以训练浅水网:

```py
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=32, epochs=40, verbose=1)
```

**第 28 行**用学习率`0.01`初始化 SGD 优化器。然后在**线 29** 上构建浅网，宽度为`32`，高度为`32`，深度为`3`(因为 CIFAR-10 图像有三个通道)。我们设置`classes=10`，因为顾名思义，在 CIFAR-10 数据集中有 10 个类。该模型在**线 30 和 31** 上编译，然后在 40 个时期内在**线 35 和 36** 上训练。

评估 ShallowNet 的方式与我们之前的动物数据集示例完全相同:

```py
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))
```

我们还将绘制一段时间内的损耗和精度，以便了解我们的网络性能如何:

```py
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
```

要在 CIFAR-10 上训练 ShallowNet，只需执行以下命令:

```py
$ python shallownet_cifar10.py
[INFO] loading CIFAR-10 data...
[INFO] compiling model...
[INFO] training network...
Train on 50000 samples, validate on 10000 samples
Epoch 1/40
5s - loss: 1.8087 - acc: 0.3653 - val_loss: 1.6558 - val_acc: 0.4282
Epoch 2/40
5s - loss: 1.5669 - acc: 0.4583 - val_loss: 1.4903 - val_acc: 0.4724
...
Epoch 40/40
5s - loss: 0.6768 - acc: 0.7685 - val_loss: 1.2418 - val_acc: 0.5890
[INFO] evaluating network...
             precision    recall  f1-score   support

   airplane       0.62      0.68      0.65      1000
 automobile       0.79      0.64      0.71      1000
       bird       0.43      0.46      0.44      1000
        cat       0.42      0.38      0.40      1000
       deer       0.52      0.51      0.52      1000
        dog       0.44      0.57      0.50      1000
       frog       0.74      0.61      0.67      1000
      horse       0.71      0.61      0.66      1000
       ship       0.65      0.77      0.70      1000
      truck       0.67      0.66      0.66      1000

avg / total       0.60      0.59      0.59     10000
```

同样，由于浅网络架构和相对较小的数据集，历元非常快。使用我的 GPU，我获得了 5 秒的历元，而我的 CPU 为每个历元花费了 22 秒。

在 40 个时期之后，对 ShallowNet 进行评估，我们发现它在测试集上获得了 **60%的准确度**，比之前使用简单神经网络获得的 57%的准确度有所提高。

更重要的是，在**图 4** 中绘制我们的损失和准确性让我们对训练过程有了一些了解，这表明我们的验证损失并没有飙升。我们的训练和测试损失/准确性开始偏离超过纪元 10。同样，这可以归因于较大的学习率，以及我们没有使用方法来帮助克服过度拟合(正则化参数、丢失、数据扩充等)的事实。).

由于低分辨率训练样本的数量有限，*也很容易*在 CIFAR-10 数据集上过度拟合。随着我们越来越习惯于构建和训练我们自己的定制卷积神经网络，我们将发现在 CIFAR-10 上提高分类精度的方法，同时减少过拟合。

## **总结**

在本教程中，我们实现了我们的第一个卷积神经网络架构 ShallowNet，并在 Animals 和 CIFAR-10 数据集上对其进行了训练。ShallowNet 获得了 71%的动物分类准确率，比我们以前使用简单前馈神经网络的最好成绩提高了 12%。

当应用于 CIFAR-10 时，ShallowNet 达到了 60%的准确率，比使用简单多层神经网络时的最高准确率提高了 57%(并且没有*显著的*过度拟合)。

ShallowNet 是一个*极其*简单的 CNN，仅使用*一个* `CONV`层——通过多组`CONV => RELU => POOL`操作训练更深的网络，可以获得更高的精度。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*********