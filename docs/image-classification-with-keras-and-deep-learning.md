# 基于 Keras 和深度学习的图像分类

> 原文：<https://pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/>

圣诞节在我心中占有特殊的位置。

不是因为我特别信仰宗教或精神。不是因为我喜欢寒冷的天气。当然也不是因为我喜欢蛋奶酒的味道(单是这种稠度就让我反胃)。

相反，因为我的父亲，圣诞节对我来说意义重大。

正如我在几周前的一篇帖子中提到的，我有一个特别艰难的童年。我家有很多精神疾病。我不得不在那样的环境中快速成长，有时我会错过作为一个孩子的纯真以及活在当下的时光。

但不知何故，在所有的挣扎中，我父亲让圣诞节成为幸福的灯塔。

也许我小时候最美好的回忆之一是在幼儿园的时候(5-6 岁)。我刚下公共汽车，手里拿着书包。

我正走在我们长长的、弯曲的车道上，在山脚下，我看到爸爸正在布置圣诞灯，这些圣诞灯后来装饰了我们的房子、灌木丛和树木，把我们的家变成了一个圣诞仙境。

我像火箭一样起飞，漫不经心地跑在车道上(只有孩子才能做到)，拉开拉链的冬衣在我身后翻滚，我一边跑一边喊着:

“等等我，爸爸！”

我不想错过装饰庆典。

在接下来的几个小时里，我父亲耐心地帮我解开打结的圣诞彩灯，把它们摆好，然后看着我随意地把彩灯扔向灌木丛和树木(比我的体型大很多倍)，毁掉了他孜孜不倦地设计的任何有条不紊、计划周密的装饰蓝图。

我说完后，他骄傲地笑了。他不需要任何言语。他的微笑表明我的装修是他见过的最好的。

这只是我爸爸为我准备的无数次特别圣诞节中的一个例子(不管家里可能还发生了什么)。

他可能甚至不知道他正在我的脑海中打造一个终身记忆——他只是想让我开心。

每年，当圣诞节来临的时候，我都会试着放慢脚步，减轻压力，享受一年中的时光。

没有我的父亲，我就不会有今天——我也肯定不会度过我的童年。

为了庆祝圣诞节，我想把这篇博客献给我的父亲。

即使你很忙，没有时间，或者根本不关心深度学习(今天教程的主题)，也要放慢脚步，读一读这篇博文，不为别的，只为我爸。

我希望你喜欢它。

## 基于 Keras 和深度学习的图像分类

***2020-05-13 更新:**此博文现已兼容 TensorFlow 2+!*

这篇博客是我们构建非圣诞老人深度学习分类器(即，可以识别圣诞老人是否在图像中的深度学习模型)的三部分系列中的第二部分:

1.  **第一部分:** [深度学习+谷歌图片获取训练数据](https://pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/)
2.  **第二部分:**使用深度学习训练圣诞老人/非圣诞老人检测器(本文)
3.  **第三部分:**将圣诞老人/非圣诞老人深度学习检测器部署到树莓派(下周帖子)

在本教程的第一部分，我们将检查我们的“圣诞老人”和“不是圣诞老人”数据集。

总之，这些图像将使我们能够使用 Python 和 Keras 训练一个卷积神经网络，以检测图像中是否有圣诞老人。

一旦我们探索了我们的训练图像，我们将继续训练[开创性的 LeNet 架构](https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)。我们将使用一个较小的网络架构，以确保没有昂贵的 GPU 的读者仍然可以跟随本教程。这也将确保初学者可以通过 Keras 和 Python 理解卷积神经网络的深度学习的基本原理。

最后，我们将在一系列图像上评估我们的*不是圣诞老人*模型，然后讨论我们方法的一些限制(以及如何进一步扩展它)。

### 我们的“圣诞老人”和“非圣诞老人”数据集

为了训练我们的非圣诞老人深度学习模型，我们需要两组图像:

*   图片*包含*圣诞老人(“Santa”)。
*   *不包含*圣诞老人(“非圣诞老人”)的图像。

上周，我们使用我们的 [Google Images hack](https://pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/) 快速获取深度学习网络的训练图像。

在这种情况下，我们可以看到使用该技术收集的包含圣诞老人的 461 幅图像的样本(**图 1** ，*左*)。

然后，我从 [UKBench 数据集](https://archive.org/details/ukbench)中随机抽取了 461 张不包含圣诞老人的图片(**图 1** ，*右*)，这是一组用于构建和评估基于内容的图像检索(CBIR)系统(即图像搜索引擎)的`~10,000`图片。

这两个图像集一起使用，将使我们能够训练我们的*而不是圣诞老人*深度学习模型。

### 配置您的开发环境

要针对本教程配置您的系统，我首先建议您遵循以下任一教程:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

这两个教程都将帮助您在一个方便的 Python 虚拟环境中，用这篇博文所需的所有软件来配置您的系统。

请注意 [PyImageSearch 不推荐也不支持 CV/DL 项目](https://pyimagesearch.com/faqs/single-faq/can-you-help-me-do-___-on-windows/)的窗口。

### 你的第一个带有卷积神经网络和 Keras 的图像分类器

LetNet 架构是卷积神经网络的优秀“第一图像分类器”。最初设计用于分类手写数字，我们可以很容易地将其扩展到其他类型的图像。

本教程旨在介绍使用深度学习、Keras 和 Python 进行图像分类，因此我不会讨论每一层的内部工作原理。如果你有兴趣深入研究深度学习，请看看我的书， [*用 Python 进行计算机视觉的深度学习*](https://pyimagesearch.com/deep-learning-computer-vision-python-book/) ，我在书中详细讨论了深度学习(并且有大量代码+实践，以及动手实现)。

让我们继续定义网络架构。打开一个新文件，将其命名为`lenet.py`，并插入以下代码:

***注意:**在运行代码之前，您需要使用本文的**“下载”**部分下载源代码+示例图片。出于完整性考虑，我在下面添加了代码，但是您需要确保您的目录结构与我的相匹配。*

```py
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

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

**第 2-8 行**处理导入我们需要的 Python 包。`Conv2D`类负责执行卷积。我们可以使用`MaxPooling2D`类进行最大池操作。顾名思义，`Activation`类应用了一个特定的激活函数。当我们准备好将`Flatten`我们的网络拓扑变成完全连通的时候，`Dense`层我们可以使用各自的类名。

在**行 10** 上定义了`LeNet`类，然后在**行 12** 上定义了`build`方法。每当我定义一个新的卷积神经网络架构时，我喜欢:

*   将它放在自己的类中(出于命名空间和组织的目的)
*   创建一个静态的`build`函数来构建架构本身

顾名思义,`build`方法有许多参数，下面我将逐一讨论:

*   `width`:我们输入图像的宽度
*   `height`:输入图像的高度
*   `depth`:我们输入图像中的通道数(`1`用于灰度单通道图像，`3`用于我们将在本教程中使用的标准 RGB 图像)
*   `classes`:我们想要识别的类的总数(在本例中是两个)

我们在第 14 行的**上定义我们的`model`。我们使用`Sequential`类，因为我们将*依次*添加层到`model`。**

**第 15 行**使用*通道最后一次*排序初始化我们的`inputShape`(tensor flow 的默认)。如果您正在使用 Theano(或任何其他假定*通道优先*排序的 Keras 后端)，**行 18 和 19** 正确更新`inputShape`。

现在我们已经初始化了我们的模型，我们可以开始向它添加层:

```py
		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

```

**第 21-25 行**创建了我们的第一组`CONV => RELU => POOL`图层。

`CONV`层会学习 20 个[卷积滤波器](https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/)，每个 *5×5* 。

然后，我们应用一个 ReLU 激活函数，接着是在 *x* 和 *y* 方向上的 *2×2* max-pooling，步幅为 2。为了可视化这个操作，考虑一个[滑动窗口](https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)，它在激活体积上“滑动”，在每个区域上取最大值操作，同时在水平和垂直方向上取两个像素的步长。

让我们定义第二组`CONV => RELU => POOL`层:

```py
		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

```

这次我们学习的是 *50 个卷积滤波器*，而不是之前层集中的 *20 个卷积滤波器*。我们在网络架构中经常会看到学习到的`CONV`过滤器数量 ***增加******更深*** 。

我们的最后一个代码块处理将卷展平为一组完全连接的层:

```py
		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model

```

在第 33 行的**上，我们将前面`MaxPooling2D`层的输出展平成一个矢量。该操作允许我们应用我们的密集/全连接层。**

我们的全连接层包含 500 个节点(**行 34** )，然后我们通过另一个非线性 ReLU 激活。

**第 38 行**定义了另一个全连接层，但这一层比较特殊——节点的数量等于`classes`的数量(即我们要识别的类)。

这个`Dense`层然后被输入到我们的 softmax 分类器中，该分类器将为每个类别产生*概率*。

最后， **Line 42** 将我们完全构建的深度学习+ Keras 图像分类器返回给调用函数。

### 用 Keras 训练我们的卷积神经网络图像分类器

让我们开始使用深度学习、Keras 和 Python 来训练我们的图像分类器。

***注:**务必向下滚动到**“下载”**部分，抓取代码+训练图像。这将使您能够跟随帖子，然后使用我们为您整理的数据集来训练您的图像分类器。* 

打开一个新文件，将其命名为`train_network.py`，并插入以下代码(或者简单地跟随代码下载):

```py
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

```

在**第 2-18 行**，我们导入所需的包。这些软件包使我们能够:

1.  从磁盘加载我们的图像数据集
2.  预处理图像
3.  实例化我们的卷积神经网络
4.  训练我们的图像分类器

注意，在**的第 3 行**上，我们将`matplotlib`后端设置为`"Agg"`，这样我们就可以在后台将绘图保存到磁盘上。如果您正在使用一个无头服务器来训练您的网络(例如 Azure、AWS 或其他云实例)，这一点很重要。

从那里，我们解析命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
  help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
  help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
  help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

```

这里我们有两个必需的命令行参数，`--dataset`和`--model`，以及一个到我们的准确度/损失图表的可选路径，`--plot`。

`--dataset`开关应该指向包含我们将在其上训练我们的图像分类器的图像的目录(即，“圣诞老人”和“非圣诞老人”图像)，而`--model`开关控制我们将在训练后保存我们的序列化图像分类器的位置。如果不指定`--plot`，则默认为该目录中的`plot.png`。

接下来，我们将设置一些训练变量，初始化列表，并收集图像路径:

```py
# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

```

在第 32-34 行上，我们定义了训练时期的数量、初始学习率和批量大小。

然后我们初始化数据和标签列表(**第 38 行和第 39 行**)。这些列表将负责存储我们从磁盘加载的图像以及它们各自的类标签。

从那里我们获取输入图像的路径，然后对它们进行洗牌(**第 42-44 行**)。

现在让我们对图像进行预处理:

```py
# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "santa" else 0
	labels.append(label)

```

这个循环简单地将每个图像加载并调整大小为固定的 *28×28* 像素(LeNet 所需的空间尺寸)，并将图像数组附加到`data`列表中(**第 49-52 行**)，然后从第 56-58 行**的`imagePath`中提取类`label`。**

我们能够执行此类标签提取，因为我们的数据集目录结构是按以下方式组织的:

```py
|--- images
|    |--- not_santa
|    |    |--- 00000000.jpg
|    |    |--- 00000001.jpg
...
|    |    |--- 00000460.jpg
|    |--- santa
|    |    |--- 00000000.jpg
|    |    |--- 00000001.jpg
...
|    |    |--- 00000460.jpg
|--- pyimagesearch
|    |--- __init__.py
|    |--- lenet.py
|    |    |--- __init__.py
|    |    |--- networks
|    |    |    |--- __init__.py
|    |    |    |--- lenet.py
|--- test_network.py
|--- train_network.py

```

因此，`imagePath`的一个例子是:

```py
images/santa/00000384.jpg
```

从`imagePath`中提取`label`后，结果是:

```py
santa
```

我更喜欢以这种方式组织深度学习图像数据集，因为它允许我们有效地组织我们的数据集并解析出类别标签，而不必使用单独的索引/查找文件。

接下来，我们将缩放图像并创建训练和测试分割:

```py
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

```

在**行 61** 上，我们通过将数据点从*【0，255】*(图像的最小和最大 RGB 值)缩放到范围*【0，1】*来进一步预处理我们的输入数据。

然后，我们使用 75%的图像进行训练，25%的图像进行测试，对数据进行训练/测试分割(**行 66 和 67** )。这是这种数据量的典型拆分。

我们还使用一键编码将标签转换成矢量——这是在第 70 行和第 71 行处理的。

随后，我们将执行一些数据扩充，使我们能够通过使用以下参数随机变换输入图像来生成“附加”训练数据:

```py
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

```

数据增强在我的新书《用 Python 进行计算机视觉的深度学习[](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)*的实践者包中有深入的介绍。*

 *实际上，第 74-76 行创建了一个图像生成器对象，它对我们的图像数据集执行随机旋转、移动、翻转、裁剪和剪切。这使得我们可以使用更小的数据集，但仍然可以获得高质量的结果。

让我们继续使用深度学习和 Keras 来训练我们的图像分类器。

```py
# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"], save_format="h5")

```

我们选择在这个项目中使用 LeNet 有两个原因:

1.  LeNet 是一个小型的卷积神经网络，初学者很容易理解
2.  我们可以很容易地在我们的圣诞老人/非圣诞老人数据集上训练 LeNet，而不必使用 GPU
3.  如果你想更深入地研究深度学习(包括 ResNet、GoogLeNet、SqueezeNet 等)请看看我的书， *[用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)。*

我们在第 80-83 行的**上构建 LeNet 模型和`Adam`优化器。由于这是一个两类分类问题，我们将使用二进制交叉熵作为我们的损失函数。如果您使用 *> 2* 类进行分类，请确保将`loss`替换为`categorical_crossentropy`。**

训练我们的网络在**行 87-89** 开始，在那里我们调用`model.fit`，提供我们的数据增强对象、训练/测试数据以及我们希望训练的时期数。

***2020-05-13 更新:**以前，TensorFlow/Keras 需要使用一种叫做`fit_generator`的方法来完成数据扩充。现在，`fit`方法也可以处理数据扩充，使代码更加一致。请务必查看我关于 [fit 和 fit 生成器](https://pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/)以及[数据扩充](https://pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/)的文章。*

**第 93 行**处理将模型序列化到磁盘，这样我们稍后就可以使用我们的图像分类*而不需要*重新训练它。

最后，让我们绘制结果，看看我们的深度学习图像分类器的表现如何:

```py
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

```

***2020-05-13 更新:**为了使该绘图片段与 TensorFlow 2+兼容，更新了`H.history`字典键，以完全拼出“精度”而没有“acc”(即`H.history["val_accuracy"]`和`H.history["accuracy"]`)。“val”没有拼成“validation”，这有点令人困惑；我们必须学会热爱 API 并与之共存，并永远记住这是一项正在进行的工作，世界各地的许多开发人员都为此做出了贡献。*

使用 matplotlib，我们构建我们的绘图，并使用包含路径+文件名的`--plot`命令行参数将绘图保存到磁盘。

为了训练*非圣诞老人*网络(在使用本博文的 ***【下载】*** 部分下载代码+图片之后)，打开一个终端并执行以下命令:

```py
$ python train_network.py --dataset images --model santa_not_santa.model
Using TensorFlow backend.
[INFO] loading images...
[INFO] compiling model...
[INFO] training network...
Train for 21 steps, validate on 231 samples
Epoch 1/25
 1/21 [>.............................] - ETA: 11s - loss: 0.6757 - accuracy: 0.7368
21/21 [==============================] - 1s 43ms/step - loss: 0.7833 - accuracy: 0.4947 - val_loss: 0.5988 - val_accuracy: 0.5022
Epoch 2/25
21/21 [==============================] - 0s 21ms/step - loss: 0.5619 - accuracy: 0.6783 - val_loss: 0.4819 - val_accuracy: 0.7143
Epoch 3/25
21/21 [==============================] - 0s 21ms/step - loss: 0.4472 - accuracy: 0.8194 - val_loss: 0.4558 - val_accuracy: 0.7879
...
Epoch 23/25
21/21 [==============================] - 0s 23ms/step - loss: 0.1123 - accuracy: 0.9575 - val_loss: 0.2152 - val_accuracy: 0.9394
Epoch 24/25
21/21 [==============================] - 0s 23ms/step - loss: 0.1206 - accuracy: 0.9484 - val_loss: 0.4427 - val_accuracy: 0.8615
Epoch 25/25
21/21 [==============================] - 1s 25ms/step - loss: 0.1448 - accuracy: 0.9469 - val_loss: 0.1682 - val_accuracy: 0.9524
[INFO] serializing network...
```

如您所见，网络训练了 25 个时期，我们实现了高精度( ***95.24%*** 测试精度)和跟随训练损耗的低损耗，如下面的图中所示:

### 评估我们的卷积神经网络图像分类器

下一步是在示例图像上评估我们的*非圣诞老人*模型*非*部分的训练/测试分割。

打开一个新文件，命名为`test_network.py`，让我们开始吧:

```py
# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

```

在**2-7 行**，我们导入我们需要的包。请特别注意`load_model`方法——该函数将使我们能够从磁盘加载我们的序列化卷积神经网络(即，我们刚刚在上一节中训练的网络)。

接下来，我们将解析我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

```

我们需要两个命令行参数:我们的`--model`和一个输入`--image`(即我们要分类的图像)。

从那里，我们将加载图像并对其进行预处理:

```py
# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

```

我们加载`image`并在**的第 18 行和第 19 行**上复制它。该副本允许我们稍后回忆原始图像，并在其上贴上我们的标签。

**第 22-25 行**处理将我们的图像缩放到范围*【0，1】*，将其转换为一个数组，并添加一个额外的维度(**第 22-25 行**)。

正如我在我的书《用 Python 进行计算机视觉的 [*深度学习*](https://pyimagesearch.com/deep-learning-computer-vision-python-book/) 中解释的那样，我们用 CNN 分批*训练/分类图像。假设通道最后排序，通过`np.expand_dims`向数组添加额外的维度允许我们的图像具有形状`(1, width, height, 3)`。*

 *如果我们忘记添加维度，当我们调用下面的`model.predict`时会导致错误。

从那里，我们将加载*而非圣诞老人*图像分类器模型，并进行预测:

```py
# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
(notSanta, santa) = model.predict(image)[0]

```

这个代码块非常简单明了，但是因为这是执行这个脚本的地方，所以让我们花一点时间来理解一下在这个引擎盖下发生了什么。

我们在**第 29 行**加载*非圣诞老人*模型，然后在**第 32 行**进行预测。

最后，我们将使用我们的预测在`orig`图像副本上绘图，并将其显示到屏幕上:

```py
# build the label
label = "Santa" if santa > notSanta else "Not Santa"
proba = santa if santa > notSanta else notSanta
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)

```

我们在第 35 行的**处建立标签(要么是“圣诞老人”，要么不是圣诞老人)，然后在第 36** 行的**处选择相应的概率值。**

在第 37**行**上使用了`label`和`proba`来构建标签文本以显示在图像上，如下图左上角所示。

我们将图像调整到标准宽度，以确保它们适合我们的屏幕，然后将标签文本放在图像上(**第 40-42 行**)。

最后，在**行 45** 上，我们显示输出图像，直到一个键被按下(**行 46** )。

让我们试试我们的*不是圣诞老人*深度学习网络:

```py
$ python test_network.py --model santa_not_santa.model \
	--image examples/santa_01.png

```

天哪！我们的软件认为这是好的老圣尼克，所以它真的必须是他！

让我们尝试另一个图像:

```py
$ python test_network.py --model santa_not_santa.model \
	--image examples/santa_02.png
```

圣诞老人被“不是圣诞老人”检测器正确地检测出来，看起来他很高兴能送来一些玩具！

现在，让我们对*不包含*圣诞老人的图像执行图像分类:

```py
$ python test_network.py --model santa_not_santa.model \
	--image examples/manhattan.png
```

***2020-06-03 更新:**曼哈顿地平线的图片不再包含在**“下载”中**更新这篇博文以支持 TensorFlow 2+导致了这张图片的错误分类。这个图保留在帖子中是为了遗留的演示目的，要知道你不会在* ***“下载”中找到它***

看起来外面太亮了，圣诞老人不可能飞过天空，在世界的这个地方(纽约市)送礼物——此时夜幕已经降临，他一定还在欧洲。

说到夜晚和平安夜，这里有一张寒冷夜空的图片:

```py
$ python test_network.py --model santa_not_santa.model \
	--image examples/night_sky.png
```

但对圣尼古拉斯来说肯定太早了。他也不在上面的图像中。

但是不要担心！

正如我下周将展示的，我们将能够发现他偷偷从烟囱下来，用树莓皮送礼物。

### 我们深度学习图像分类模型的局限性

我们的图像分类器有许多限制。

第一个是 *28×28* 像素图像相当小(LeNet 架构最初是为了识别手写数字，而不是照片中的物体)。

对于一些示例图像(其中圣诞老人已经很小了)，将输入图像的尺寸缩小到 *28×28* 像素有效地将圣诞老人缩小到只有 2-3 像素大小的微小红/白斑点。

在这些类型的情况下，我们的 LeNet 模型可能只是预测何时在我们的输入图像中有大量的红色和白色集中在一起(也可能是绿色，因为红色、绿色和白色是圣诞节的颜色)。

最先进的卷积神经网络通常接受最大维度为 200-300 像素的图像——这些更大的图像将帮助我们建立一个更强大的 Not Santa 分类器。然而，使用更大分辨率的图像还需要我们利用更深层次的网络架构，这反过来意味着我们需要收集额外的训练数据，并利用计算成本更高的训练过程。

这当然是可能的，但也超出了这篇博文的范围。

因此，如果你想改进我们的*而不是圣诞老人*应用程序，我建议你:

1.  收集额外的培训数据(理想情况下，5，000 多个“圣诞老人”图像示例)。
2.  在训练中使用更高分辨率的图像。我想象 64×64 像素会产生更高的精度。 *128×128* 像素可能是理想的(尽管我没有试过)。
3.  在培训中使用更深层次的网络架构。
4.  通读我的书， [*用 Python 进行计算机视觉的深度学习*](https://pyimagesearch.com/deep-learning-computer-vision-python-book/) ，我在其中更详细地讨论了在你自己的定制数据集上训练卷积神经网络。

尽管有这些限制，我还是对*而不是圣诞老人*应用的表现感到非常惊讶(我将在下周讨论)。我原以为会有相当数量的误报，但考虑到网络如此之小，它却异常强大。

## 摘要

在今天的博客文章中，你学习了如何在一系列包含“圣诞老人”和“非圣诞老人”的图像上训练[开创性的 LeNet 架构](https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)，我们的最终目标是构建一个类似于 HBO 的硅谷 [*非热狗*应用](https://www.engadget.com/2017/05/15/not-hotdog-app-hbo-silicon-valley/)的应用。

通过跟随我们之前关于通过谷歌图像收集深度学习图像的[的帖子，我们能够收集我们的“圣诞老人”数据集(`~460`图像)。](https://pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/)

“非圣诞老人”数据集是通过对 [UKBench 数据集](https://archive.org/details/ukbench)(其中没有包含圣诞老人的图像)进行采样而创建的。

然后，我们在一系列测试图像上评估了我们的网络——在每种情况下，我们的 *Not Santa* 模型都正确地对输入图像进行了分类。

在我们的下一篇博客文章中，我们将把我们训练好的卷积神经网络部署到 Raspberry Pi，以完成我们的 *Not Santa* 应用程序的构建。

## 现在怎么办？

现在，您已经学会了如何训练您的第一个卷积神经网络，我敢打赌，您对以下内容感兴趣:

*   掌握机器学习和神经网络的基础知识
*   更详细地研究深度学习
*   从头开始训练你自己的卷积神经网络

如果是这样，你会想看看我的新书， [*用 Python 进行计算机视觉的深度学习*](https://pyimagesearch.com/deep-learning-computer-vision-python-book/) 。

在这本书里，你会发现:

*   超级实用的**演练**
*   **实践教程**(有大量代码)
*   详细、全面的指南帮助您**从开创性的深度学习出版物中复制最先进的结果**。

**要了解更多关于我的新书(并开始你的深度学习掌握之旅)， [*只需点击这里*](https://pyimagesearch.com/deep-learning-computer-vision-python-book/) 。**

否则，请务必在下表中输入您的电子邮件地址，以便在 PyImageSearch 上发布新的深度学习帖子时得到通知。**