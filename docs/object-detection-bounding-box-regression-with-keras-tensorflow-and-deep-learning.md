# 对象检测:使用 Keras、TensorFlow 和深度学习的包围盒回归

> 原文：<https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/>

在本教程中，您将学习如何训练一个自定义深度学习模型，通过 Keras 和 TensorFlow 的边界框回归来执行对象检测。

今天的教程受到了我从 PyImageSearch 读者 Kyle 那里收到的一条消息的启发:

> *嗨，阿德里安，*
> 
> *非常感谢你的* [关于区域提议物体探测器的四部分系列教程。](https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/) *它帮助我了解了 R-CNN 物体探测器的基本工作原理。*
> 
> 但是我对“包围盒回归”这个术语有点困惑那是什么意思？包围盒回归是如何工作的？包围盒回归如何预测图像中对象的位置？

问得好，凯尔。

基本的 R-CNN 物体检测器[，比如我们在 PyImageSearch 博客](https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/)上提到的那些，依赖于**区域提议**生成器的概念。

这些区域提议算法(例如，选择性搜索)检查输入图像，然后识别潜在对象*可能*在哪里。请记住，他们完全不知道*是否*一个物体存在于一个给定的位置，只知道图像的区域*看起来很有趣*并需要进一步检查。

在 Girshick 等人的 R-CNN 的经典实现中，这些区域建议用于从预训练的 CNN(减去全连接层头)中提取输出特征，然后输入到 SVM 中进行最终分类。**在该实现中，来自区域提议的位置被视为*边界框*，而 SVM 为边界框区域产生了*类别标签*。**

本质上，最初的 R-CNN 架构实际上并没有“学习”如何检测边界框——它是*而不是*端到端可训练的(未来的迭代，如更快的 R-CNN，实际上*是*端到端可训练的)。

但这也带来了问题:

*   如果我们想训练一个端到端的物体探测器呢？
*   有没有可能构建一个可以输出边界框坐标的 CNN 架构，这样我们就可以真正地*训练*这个模型来做出更好的物体探测器预测？
*   如果是这样，我们如何着手训练这样一个模型？

所有这些问题的关键在于**边界框回归**的概念，这正是我们今天要讨论的内容。在本教程结束时，您将拥有一个端到端的可训练对象检测器，能够为图像中的对象生成*边界框预测*和*类别标签预测*。

**要了解如何使用 Keras、TensorFlow 和深度学习通过包围盒回归来执行对象检测，*请继续阅读。***

## **物体检测:使用 Keras、TensorFlow 和深度学习的包围盒回归**

在本教程的第一部分，我们将简要讨论包围盒回归的概念，以及如何使用它来训练端到端的对象检测器。

然后，我们将讨论用于训练边界框回归器的数据集。

从那里，我们将回顾项目的目录结构，以及一个简单的 Python 配置文件(因为我们的实现跨越多个文件)。给定我们的配置文件，我们将能够实现一个脚本，通过 Keras 和 TensorFlow 的包围盒回归来实际训练我们的对象检测模型。

训练好我们的模型后，我们将实现第二个 Python 脚本，这个脚本处理新输入图像上的*推理*(即，进行对象检测预测)。

我们开始吧！

### **什么是包围盒回归？**

我们可能都熟悉通过深度神经网络进行图像分类的概念。执行图像分类时，我们:

1.  向 CNN 展示输入图像
2.  向前通过 CNN
3.  输出一个包含 *N* 个元素的向量，其中 *N* 是类别标签的总数
4.  选择具有最大概率的类别标签作为我们最终预测的类别标签

**从根本上讲，我们可以把图像分类看作是预测一个*类标签。***

但不幸的是，这种类型的模型不能转化为对象检测。我们不可能为输入图像中的(x，y)坐标边界框的每个可能组合*构造一个类别标签。*

相反，我们需要依赖一种不同类型的机器学习模型，称为**回归。**与产生标签的分类不同，**回归使我们能够预测连续值。**

通常，回归模型适用于以下问题:

*   预测房屋价格([，我们在本教程中实际上已经做过](https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/)
*   预测股票市场
*   确定疾病在人群中传播的速度
*   *等。*

这里的要点是，回归模型的输出*不像分类模型那样局限于被离散化到“箱”中(记住，分类模型只能输出一个类标签，仅此而已)。*

相反，回归模型可以输出特定范围内的任何真实值。

通常，我们在训练期间将值的输出范围缩放到*【0，1】*，然后在预测之后将输出缩放回来(如果需要)。

**为了执行用于对象检测的包围盒回归，我们需要做的就是调整我们的网络架构:**

1.  在网络的顶端，放置一个具有四个神经元的全连接层，分别对应于左上和右下(x，y)坐标。
2.  给定四个神经元层，实现一个 sigmoid 激活函数，使得输出在范围 *[0，1]内返回。*
3.  对训练数据使用诸如均方误差或平均绝对误差的损失函数来训练模型，该训练数据包括(1)输入图像和(2)图像中对象的边界框。

在训练之后，我们可以向我们的包围盒回归器网络呈现输入图像。然后，我们的网络将执行向前传递，然后实际上*预测*对象的输出边界框坐标。

在本教程中，我们将通过包围盒回归为一个*单类*对象检测，但下周我们将把它扩展到*多类*对象检测。

### **我们的对象检测和包围盒回归数据集**

我们今天在这里使用的示例数据集是 CALTECH-101 数据集的子集，可用于训练对象检测模型。

具体来说，我们将使用由 **800 张图像**和图像中飞机的相应边界框坐标组成的**飞机**类。我已经在**图 2** 中包含了飞机示例图像的子集。

我们的目标是训练一个能够准确预测输入图像中飞机边界框坐标的目标检测器。

***注意:**没有必要从加州理工学院 101 的网站上下载完整的数据集。我已经在与本教程相关的**“下载”**部分包含了飞机图像的子集，包括一个 CSV 文件的边界框。*

### 配置您的开发环境

要针对本教程配置您的系统，我建议您遵循以下任一教程:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

这两个教程都将帮助您在一个方便的 Python 虚拟环境中，用这篇博文所需的所有软件来配置您的系统。

**这样说来，你是:**

*   时间紧迫？
*   在你雇主被行政锁定的笔记本电脑上学习？
*   想要跳过与包管理器、bash/ZSH 概要文件和虚拟环境的争论吗？
*   准备好立即运行代码了吗(并尽情地试验它)？

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！在您的浏览器 — *中访问运行在**谷歌的 Colab 生态系统*上的 PyImageSearch 教程 **Jupyter 笔记本**！****

### **项目结构**

去拿吧。本教程的 ***【下载】*** 部分的 zip 文件。在里面，您将找到数据子集以及我们的项目文件:

```py
$ tree --dirsfirst --filelimit 10
.
├── dataset
│   ├── images [800 entries]
│   └── airplanes.csv
├── output
│   ├── detector.h5
│   ├── plot.png
│   └── test_images.txt
├── pyimagesearch
│   ├── __init__.py
│   └── config.py
├── predict.py
└── train.py

4 directories, 8 files
```

### **创建我们的配置文件**

在实现边界框回归训练脚本之前，我们需要创建一个简单的 Python 配置文件，该文件将存储在训练和预测脚本中重用的变量，包括图像路径、模型路径等。

打开`config.py`文件，让我们看一看:

```py
# import the necessary packages
import os

# define the base path to the input dataset and then use it to derive
# the path to the images directory and annotation CSV file
BASE_PATH = "dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "airplanes.csv"])
```

```py
# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])
```

```py
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32
```

我们的深度学习超参数包括初始学习速率、时期数和批量大小。这些参数放在一个方便的地方，以便您可以跟踪您的实验输入和结果。

### **用 Keras 和 TensorFlow 实现我们的包围盒回归训练脚本**

实现了配置文件后，我们可以开始创建边界框回归训练脚本了。

该脚本将负责:

1.  从磁盘加载我们的飞机训练数据(即类标签和边界框坐标)
2.  从磁盘加载 VGG16(在 ImageNet 上预先训练)，从网络中移除完全连接的分类层头，并插入我们的边界框回归层头
3.  在我们的训练数据上微调包围盒回归图层头

我假设你已经习惯了[修改网络架构并对其进行微调](https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/)。

如果您对这个概念还不太熟悉，我建议您在继续之前阅读上面链接的文章。

边界框回归是一个最好通过代码解释的概念，所以打开项目目录中的`train.py`文件，让我们开始工作:

```py
# import the necessary packages
from pyimagesearch import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
```

我们的训练脚本从选择导入开始。其中包括:

*   ``config`` :我们在上一节开发的配置文件，由路径和超参数组成
*   ``VGG16``:CNN 架构作为我们微调方法的基础网络
*   ``tf.keras`` :从 TensorFlow/Keras 导入，包括图层类型、优化器和图像加载/预处理例程
*   ``train_test_split`` : Scikit-learn 的便利实用程序，用于将我们的网络分成训练和测试子集
*   ``matplotlib`` : Python 的事实绘图包
*   ``numpy`` : Python 的标准数值处理库
*   ``cv2`` : OpenCV

同样，您需要遵循*“配置您的开发环境”*一节，以确保您已经安装了所有必要的软件，或者选择在 Jupyter 笔记本中运行该脚本。

现在，我们的环境已经准备就绪，包也已导入，让我们来处理我们的数据:

```py
# load the contents of the CSV annotations file
print("[INFO] loading dataset...")
rows = open(config.ANNOTS_PATH).read().strip().split("\n")

# initialize the list of data (images), our target output predictions
# (bounding box coordinates), along with the filenames of the
# individual images
data = []
targets = []
filenames = []
```

这里，我们加载我们的边界框注释 CSV 数据(**第 19 行**)。文件中的每条记录都由一个图像文件名和与该图像关联的任何对象边界框组成。

然后我们进行三次列表初始化:

*   ``data`` :即将容纳我们所有的图像
*   很快就会拥有我们所有的预测和包围盒坐标
*   ``filenames`` :与实际图像相关联的文件名`data`

这是三个相互对应的独立列表。我们现在开始一个循环，试图从 CSV 数据填充列表:

```py
# loop over the rows
for row in rows:
	# break the row into the filename and bounding box coordinates
	row = row.split(",")
	(filename, startX, startY, endX, endY) = row
```

遍历 CSV 文件中的所有行(**第 29 行**)，我们的第一步是解开特定条目的`filename`和边界框坐标(**第 31 行和第 32 行**)。

为了对 CSV 数据有所了解，让我们来看一下内部情况:

```py
image_0001.jpg,49,30,349,137
image_0002.jpg,59,35,342,153
image_0003.jpg,47,36,331,135
image_0004.jpg,47,24,342,141
image_0005.jpg,48,18,339,146
image_0006.jpg,48,24,344,126
image_0007.jpg,49,23,344,122
image_0008.jpg,51,29,344,119
image_0009.jpg,50,29,344,137
image_0010.jpg,55,32,335,106
```

如您所见，每行包含五个元素:

1.  文件名
2.  起始*x*-坐标
3.  开始 *y* 坐标
4.  终点*x*-坐标
5.  终点*y*-坐标

这些正是我们脚本的第 32 行**解包到这个循环迭代的便利变量中的值。**

我们仍在循环中工作，接下来我们将加载一个图像:

```py
	# derive the path to the input image, load the image (in OpenCV
	# format), and grab its dimensions
	imagePath = os.path.sep.join([config.IMAGES_PATH, filename])
	image = cv2.imread(imagePath)
	(h, w) = image.shape[:2]

	# scale the bounding box coordinates relative to the spatial
	# dimensions of the input image
	startX = float(startX) / w
	startY = float(startY) / h
	endX = float(endX) / w
	endY = float(endY) / h
```

让我们结束我们的循环:

```py
	# load the image and preprocess it
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)

	# update our list of data, targets, and filenames
	data.append(image)
	targets.append((startX, startY, endX, endY))
	filenames.append(filename)
```

现在我们已经加载了数据，让我们为训练对其进行分区:

```py
# convert the data and targets to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(data, targets, filenames, test_size=0.10,
	random_state=42)

# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

# write the testing filenames to disk so that we can use then
# when evaluating/testing our bounding box regressor
print("[INFO] saving testing filenames...")
f = open(config.TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()
```

在这里我们:

```py
# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False

# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)

# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)
```

完成微调需要四个步骤:

1.  用预先训练好的 [ImageNet](http://www.image-net.org/) 砝码加载`VGG16`，砍掉*旧*全连接分类层头(**第 79、80 行**)。
2.  冻结 VGG16 网络体中的所有图层(**行 84** )。
3.  通过构建一个*新的*全连接图层头来执行网络手术，该图层头将输出对应于图像中对象的*左上*和*右下*边界框坐标的四个值(**行 87-95** )。
4.  通过将*新的*可训练头部(包围盒回归层)缝合到现有的冷冻体(**线 98** )上，完成网络手术。

现在让我们训练(即微调)我们新形成的野兽:

```py
# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=config.INIT_LR)
model.compile(loss="mse", optimizer=opt)
print(model.summary())

# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
	verbose=1)
```

```py
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")

# plot the model training history
N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
```

### **训练我们的基本包围盒回归器和对象检测器**

实现了包围盒回归网络后，让我们继续训练它。

首先使用本教程的 ***【下载】*** 部分下载源代码和示例飞机数据集。

从那里，打开一个终端，并执行以下命令:

```py
$ python train.py
[INFO] loading dataset...
[INFO] saving testing filenames...
```

我们的脚本从从磁盘加载我们的飞机数据集开始。

然后，我们构建我们的训练/测试分割，然后将测试集中的图像的文件名保存到磁盘上(这样我们就可以在以后使用我们训练过的网络进行预测时使用它们)。

从那里，我们的训练脚本输出具有边界框回归头的 VGG16 网络的模型摘要:

```py
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
dense (Dense)                (None, 128)               3211392
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256
_________________________________________________________________
dense_2 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 132
=================================================================
Total params: 17,936,548
Trainable params: 3,221,860
Non-trainable params: 14,714,688
```

注意`block5_pool (MaxPooling2D)` — **后面的图层，这些图层对应我们的包围盒回归图层头。**

训练时，这些层将学习如何预测边界框 *(x，y)*-图像中对象的坐标！

接下来是我们的实际培训流程:

```py
[INFO] training bounding box regressor...
Epoch 1/25
23/23 [==============================] - 37s 2s/step - loss: 0.0239 - val_loss: 0.0014
Epoch 2/25
23/23 [==============================] - 38s 2s/step - loss: 0.0014 - val_loss: 8.7668e-04
Epoch 3/25
23/23 [==============================] - 36s 2s/step - loss: 9.1919e-04 - val_loss: 7.5377e-04
Epoch 4/25
23/23 [==============================] - 37s 2s/step - loss: 7.1202e-04 - val_loss: 8.2668e-04
Epoch 5/25
23/23 [==============================] - 36s 2s/step - loss: 6.1626e-04 - val_loss: 6.4373e-04
...
Epoch 20/25
23/23 [==============================] - 37s 2s/step - loss: 6.9272e-05 - val_loss: 5.6152e-04
Epoch 21/25
23/23 [==============================] - 36s 2s/step - loss: 6.3215e-05 - val_loss: 5.4341e-04
Epoch 22/25
23/23 [==============================] - 37s 2s/step - loss: 5.7234e-05 - val_loss: 5.5000e-04
Epoch 23/25
23/23 [==============================] - 37s 2s/step - loss: 5.4265e-05 - val_loss: 5.5932e-04
Epoch 24/25
23/23 [==============================] - 37s 2s/step - loss: 4.5151e-05 - val_loss: 5.4348e-04
Epoch 25/25
23/23 [==============================] - 37s 2s/step - loss: 4.0826e-05 - val_loss: 5.3977e-04
[INFO] saving object detector model...
```

训练边界框回归器后，将生成以下训练历史图:

我们的对象检测模型以高损失开始，但是能够在训练过程中下降到较低损失的区域(即，模型学习如何做出更好的边界框预测)。

培训完成后，您的`output`目录应该包含以下文件:

```py
$ ls output/
detector.h5	plot.png	test_images.txt
```

`plot.png`文件包含我们的训练历史图，而`test_images.txt`包含我们的测试集中图像的文件名(我们将在本教程的后面对其进行预测)。

### **用 Keras 和 TensorFlow 实现我们的包围盒预测器**

此时，我们已经将边界框预测器序列化到磁盘上— **，但是我们如何使用该模型来检测输入图像中的对象呢？**

我们将在这一部分回答这个问题。

打开一个新文件，将其命名为`predict.py`，并插入以下代码:

```py
# import the necessary packages
from pyimagesearch import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os
```

至此，你应该认识到除了`imutils`(我的[计算机视觉便利包](https://github.com/jrosebr1/imutils))和潜在的`mimetypes`(内置于 Python 可以从文件名和 URL 中识别文件类型)。

让我们解析一下[命令行参数](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/):

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image/text file of image filenames")
args = vars(ap.parse_args())
```

```py
# determine the input file type, but assume that we're working with
# single input image
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]

# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:
	# load the filenames in our testing file and initialize our list
	# of image paths
	filenames = open(args["input"]).read().strip().split("\n")
	imagePaths = []

	# loop over the filenames
	for f in filenames:
		# construct the full path to the image filename and then
		# update our image paths list
		p = os.path.sep.join([config.IMAGES_PATH, f])
		imagePaths.append(p)
```

1.  **默认:**我们的`imagePaths`由来自`--input` ( **第 22 行**)的一条单独的图像路径组成。
2.  **文本文件:**如果**第 26 行**上的文本`filetype`的条件/检查符合`True`，那么我们**覆盖**并从`--input`文本文件(**第 29-37 行**)中的所有`filenames`(每行一个)填充我们的`imagePaths`。

给定一个或多个测试图像，让我们开始**用我们的深度学习 TensorFlow/Keras `model`执行包围盒回归**:

```py
# load our trained bounding box regressor from disk
print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)

# loop over the images that we'll be testing using our bounding box
# regression model
for imagePath in imagePaths:
	# load the input image (in Keras format) from disk and preprocess
	# it, scaling the pixel intensities to the range [0, 1]
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image) / 255.0
	image = np.expand_dims(image, axis=0)
```

在加载我们的`model` ( **第 41 行**)后，我们开始循环图像(**第 45 行**)。在里面，我们首先在*中加载并预处理图像，就像我们在*中做的一样。这包括:

*   将图像大小调整为 *224×224* 像素(**第 48 行**
*   转换为数组格式并将像素缩放到范围*【0，1】*(**第 49 行**)
*   添加批量维度(**第 50 行**)

从那里，我们可以**执行包围盒回归推理**并注释结果:

```py
	# make bounding box predictions on the input image
	preds = model.predict(image)[0]
	(startX, startY, endX, endY) = preds

	# load the input image (in OpenCV format), resize it such that it
	# fits on our screen, and grab its dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# scale the predicted bounding box coordinates based on the image
	# dimensions
	startX = int(startX * w)
	startY = int(startY * h)
	endX = int(endX * w)
	endY = int(endY * h)

	# draw the predicted bounding box on the image
	cv2.rectangle(image, (startX, startY), (endX, endY),
		(0, 255, 0), 2)

	# show the output image
	cv2.imshow("Output", image)
	cv2.waitKey(0)
```

**第 53 行**对输入图像进行边界框预测。注意`preds`包含了我们的边界框预测的 *(x，y)* 坐标；为了方便起见，我们通过**线 54** 解包这些值。

现在我们有了注释所需的一切。要在图像上标注边界框，我们只需:

*   用 OpenCV 从磁盘加载原始的`Image`并`resize`它，同时保持纵横比(**第 58 和 59 行**)
*   将预测边界框坐标从范围 *[0，1]* 缩放到范围 *[0，`w` ]* 和 *[0，`h` ]* 其中*`w`**`h`*为输入`image` ( **第 60-67 行**)的宽度和高度
*   绘制缩放的边界框(**第 70 行和第 71 行**)

最后，我们在屏幕上显示输出。按下一个键在循环中循环，一个接一个地显示结果，直到所有的测试图像都已完成(**行 74 和 75** )。

干得好！让我们在下一部分检查我们的结果。

### **使用 Keras 和 TensorFlow 的包围盒回归和对象检测结果**

我们现在准备好测试我们的包围盒回归对象检测模型了！

确保您已经使用本教程的 ***【下载】*** 部分下载了源代码、图像数据集和预训练的对象检测模型。

从这里开始，让我们尝试对单个输入图像应用对象检测:

```py
$ python predict.py --input datasimg/image_0697.jpg
[INFO] loading object detector...
```

正如你所看到的，我们的边界框回归器已经在输入图像中正确地定位了飞机，证明了我们的目标检测模型实际上*已经学会了*如何仅仅从输入图像中预测边界框坐标！

接下来，让我们通过提供到`test_images.txt`文件的路径作为`--input`命令行参数，将边界框回归器应用到测试集中的每个图像的*:*

```py
$ python predict.py --input output/test_images.txt
[INFO] loading object detector...
```

如**图 6** 所示，我们的目标检测模型在预测输入图像中飞机的位置方面做得非常好！

### **限制**

在这一点上，我们已经成功地训练了一个包围盒回归模型——但是这个架构的一个明显的限制是它只能预测一个*类的包围盒。*

**如果我们想要执行*多类物体检测*** 呢？我们不仅有“飞机”类，还有“摩托车”、“汽车”和“卡车”

用包围盒回归进行多类物体检测可能吗？

没错，我会在下周的教程中讲述这个话题。我们将学习多类对象检测如何需要改变包围盒回归架构(提示:我们 CNN 中的两个分支)并训练这样一个模型。 ***敬请期待！***

## **总结**

在本教程中，您学习了如何使用边界框回归训练端到端对象检测器。

为了完成这项任务，我们利用了 Keras 和 TensorFlow 深度学习库。

与仅输出类标签的分类模型不同，回归模型能够产生实值输出。

回归模型的典型应用包括预测房价、预测股票市场和预测疾病在一个地区的传播速度。

然而，回归模型并不局限于价格预测或疾病传播— *我们也可以将它们用于物体检测！*

诀窍是更新你的 CNN 架构:

1.  将具有四个神经元(左上和右下边界框坐标)的全连接层放置在网络的顶部
2.  在该层上放置一个 sigmoid 激活函数(使得输出值位于范围*【0，1】*)
3.  通过提供(1)输入图像和(2)图像中对象的目标边界框来训练模型
4.  随后，使用均方误差、平均绝对误差等来训练您的模型。

最终结果是一个端到端的可训练对象检测器，类似于我们今天构建的这个！

你会注意到我们的模型只能预测*一种*类型的类标签——我们如何扩展我们的实现来处理*多标签*？

这可能吗？

当然是这样——下周请继续关注本系列的第二部分！

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***