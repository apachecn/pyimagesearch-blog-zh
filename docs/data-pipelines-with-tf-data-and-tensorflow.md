# 使用 tf.data 和 TensorFlow 的数据管道

> 原文：<https://pyimagesearch.com/2021/06/21/data-pipelines-with-tf-data-and-tensorflow/>

在本教程中，您将学习如何使用`tf.data`和 TensorFlow 实现快速、高效的数据管道来训练神经网络。

本教程是我们关于`tf.data`模块的三部分系列的第二部分:

1.  [*温柔介绍 tf.data*](https://pyimagesearch.com/2021/06/14/a-gentle-introduction-to-tf-data-with-tensorflow/) (上周教程)
2.  *带有 tf.data 和 TensorFlow 的数据管道*(本帖)
3.  *用 tf.data 进行数据扩充*(下周教程)

上周，我们主要关注了 Keras 的`ImageDataGenerator`类与 TensorFlow v2 的`tf.data`类的基准测试— **如我们的结果所示，`tf.data`的图像数据效率提高了约 38 倍。**

但是，该教程没有解决以下问题:

*   如何使用 TensorFlow 函数有效地加载图像？
*   如何使用 TensorFlow 函数从文件路径中解析类标签？
*   什么是`@tf.function`装饰器，它如何用于数据加载和预处理？
*   我如何初始化三个独立的`tf.data`管道，一个用于训练，一个用于验证，一个用于测试？
*   最重要的是，*我如何使用`tf.data`管道训练神经网络？*

今天我们就在这里回答这些问题。

**要学习如何使用`tf.data`模块、** ***来训练一个神经网络，只需继续阅读。***

## **带有 tf.data 和 TensorFlow 的数据管道**

在本教程的第一部分，我们将讨论`tf.data`管道的效率，以及我们是否应该使用`tf.data`而不是 Keras 的经典`ImageDataGenerator`函数。

然后，我们将配置我们的开发环境，查看我们的项目目录结构，并讨论我们将在本教程中使用的图像数据集。

从那里我们可以实现我们的 Python 脚本，包括在磁盘上组织数据集的方法，初始化我们的`tf.data`管道，然后最终使用管道作为我们的数据生成器训练 CNN。

我们将通过比较我们的`tf.data`流水线速度和`ImageDataGenerator`来总结本指南，并确定哪一个获得了更快的吞吐率。

### **用 TensorFlow 构建数据管道，tf.data 和 ImageDataGenerator 哪个更好？**

就像计算机科学中的许多复杂问题一样，这个问题的答案是“视情况而定”

`ImageDataGenerator`类有很好的文档记录，易于使用，并且直接嵌入在 Keras API 中([如果你有兴趣的话](https://pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/)，我有一个关于使用 ImageDataGenerator 进行数据扩充的教程)。

另一方面,`tf.data`模块是一个较新的实现，旨在将 PyTorch 式的数据加载和处理引入 TensorFlow 库。

**当使用`tf.data`** ***创建数据管道时*** **是否需要更多的工作，所获得的加速是值得的。**

正如我上周在[演示的](https://pyimagesearch.com/2021/06/14/a-gentle-introduction-to-tf-data-with-tensorflow/)，使用`tf.data`可以使数据生成速度提高大约 38 倍。如果你需要纯粹的速度，并且愿意多写 20-30 行代码(取决于你想达到的高级程度)，那么*花*额外的前期时间来实现`tf.data`流水线是非常值得的。

### **配置您的开发环境**

这篇关于 tf.keras 数据管道的教程利用了 keras 和 TensorFlow。如果你打算遵循这个教程，我建议你花时间配置你的深度学习开发环境。

您可以利用这两个指南中的任何一个在您的系统上安装 TensorFlow 和 Keras:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

这两个教程都有助于在一个方便的 Python 虚拟环境中为您的系统配置这篇博客文章所需的所有软件。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

### **乳腺癌组织学数据集**

我们在今天的帖子中使用的数据集是浸润性导管癌(IDC)，这是所有乳腺癌中最常见的一种。长期使用 PyImageSearch 的读者会认识到，我们在使用 Keras 和深度学习教程的 [*乳腺癌分类中使用了这个数据集。*](https://pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/)

我们今天将在这里使用相同的数据集，以便我们可以比较`ImageDataGenerator`和`tf.data`的数据处理速度，从而我们可以确定哪种方法是用于训练深度神经网络的更快、更有效的数据管道。

该数据集最初是由 [Janowczyk 和 Madabhushi](https://www.ncbi.nlm.nih.gov/pubmed/27563488) 和 [Roa 等人](http://spie.org/Publications/Proceedings/Paper/10.1117/12.2043872)策划的，但在 [Kaggle 的网站](https://www.kaggle.com/paultimothymooney/breast-histopathology-images)上可以在公共领域获得。

总共有 277，524 个图像，每个图像都是 *50×50* 像素，属于两类:

*   **198，738 例阴性病例**(即无乳腺癌)
*   **78786 个阳性病例**(即表明在该贴片中发现了乳腺癌)

显然，类数据中存在不平衡，负数据点的数量是正数据点的两倍多，因此我们需要做一些类加权来处理这种偏差。

**图 3** 显示了正面和负面样本的例子——我们的目标是训练一个深度学习模型，能够辨别这两个类别之间的差异。

***注意:*** *在继续之前，我* ***强烈建议****[*阅读我之前的乳腺癌分类教程*](https://pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/)*——这不仅会让你更深入地回顾数据集，还会帮助你更好地理解/欣赏`ImageDataGenerator`管道和我们的`tf.data`管道之间的差异和字幕。**

 *### **项目结构**

首先，请务必访问本教程的 ***“下载”*** 部分以检索源代码。

从那里，解压缩归档文件:

```py
$ cd path/to/downloaded/zip
$ unzip tfdata-piplines.zip
```

现在您已经提取了文件，是时候将数据集放入目录结构中了。

继续创建以下目录:

```py
$ cd tfdata-pipelines
$ mkdir datasets
$ mkdir datasets/orig
```

然后，前往 Kaggle 的网站并登录。在那里，您可以单击以下链接将数据集下载到项目文件夹中:

[点击这里从 Kaggle](https://www.kaggle.com/paultimothymooney/breast-histopathology-images/download) 下载数据

***注:*** *你需要在 Kaggle 的网站上创建一个账户(如果你还没有账户的话)来下载数据集。*

请务必保存。`tfdata-pipelines/datasets/orig folder`中的 zip 文件。

现在回到您的终端，导航到您刚刚创建的目录，并解压缩数据:

```py
$ cd path/to/tfdata-pipelines/datasets/orig
$ unzip archive.zip -x "IDC_regular_ps50_idx5/*"
```

从那里，让我们回到项目目录，并使用 tree 命令来检查我们的项目结构:

```py
$ tree . --dirsfirst --filelimit 10
.
├── datasets
│   └── orig
│       ├── 10253
│       │   ├── 0
│       │   └── 1
│       ├── 10254
│       │   ├── 0
│       │   └── 1
│       ├── 10255
│       │   ├── 0
│       │   └── 1
...[omitting similar folders]
│       ├── 9381
│       │   ├── 0
│       │   └── 1
│       ├── 9382
│       │   ├── 0
│       │   └── 1
│       ├── 9383
│       │   ├── 0
│       │   └── 1
│       └── 7415_10564_bundle_archive.zip
├── build_dataset.py
├── plot.png
└── train_model.py

2 directories, 6 files
```

`datasets`目录，更具体地说，`datasets/orig`文件夹包含我们的原始组织病理学图像，我们将在这些图像上训练我们的 CNN。

在`pyimagesearch`模块中，我们有一个配置文件(用于存储重要的训练参数)，以及我们将要训练的 CNN，`CancerNet`。

`build_dataset.py`将重组`datasets/orig`的目录结构，这样我们就可以进行适当的训练、验证和测试分割。

然后，`train_model.py`脚本将使用`tf.data`在我们的数据集上训练`CancerNet`。

### **创建我们的配置文件**

在我们可以在磁盘上组织数据集并使用`tf.data`管道训练我们的模型之前，我们首先需要实现一个 Python 配置文件来存储重要的变量。这个配置文件在很大程度上是基于我以前的教程 [*乳腺癌分类与 Keras 和深度学习*](https://pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/) 所以我强烈建议你阅读该教程，如果你还没有。

否则，打开`pyimagesearch`模块中的`config.py`文件并插入以下代码:

```py
# import the necessary packages
import os

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = os.path.join("datasets", "orig")

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = os.path.join("datasets", "idc")
```

首先，我们将路径设置为从 Kaggle ( **第 5 行**)下载的*原始*输入目录。

在创建训练、测试和验证分割后，我们指定`BASE_PATH`存储图像文件的位置(**第 9 行**)。

使用`BASE_PATH`集合，我们可以分别定义我们的输出训练、验证和测试目录:

```py
# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
```

以下是我们的数据分割信息:

```py
# define the amount of data that will be used training
TRAIN_SPLIT = 0.8

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.1

# define input image spatial dimensions
IMAGE_SIZE = (48, 48)
```

我们将 80%的数据用于训练，20%用于测试。我们还将使用 10%的数据进行验证，这 10%将来自分配给培训的 80%份额。

此外，我们指出所有输入图像都将被调整大小，使其空间尺寸为 *48×48。*

最后，我们有几个重要的培训参数:

```py
# initialize our number of epochs, early stopping patience, initial
# learning rate, and batch size
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 5
INIT_LR = 1e-2
BS = 128
```

我们将允许我们的网络总共训练`40`个时期，但是如果在`5`个时期之后验证损失没有减少，我们将终止训练过程。

然后，我们定义我们的初始学习率和每批的大小。

### **在磁盘上实现我们的数据集管理器**

我们的乳腺癌图像数据集包含近 20 万幅图像，每幅图像都是 *50 *×* 50* 像素。如果我们试图一次将整个数据集存储在内存中，我们将需要将近 6GB 的 RAM。

大多数深度学习钻机可以轻松处理这么多数据；然而，一些笔记本电脑或台式机可能没有足够的内存，此外，本教程的全部目的是演示如何为驻留在磁盘上的数据创建有效的`tf.data`管道，所以让我们假设您的机器*不能*将整个数据集放入 RAM。

但是在我们能够构建我们的`tf.data`管道之前，我们首先需要实现`build_dataset.py`,它将获取我们的原始输入数据集并创建训练、测试和验证分割。

打开`build_dataset.py`脚本，让我们开始工作:

```py
# import the necessary packages
from pyimagesearch import config
from imutils import paths
import random
import shutil
import os

# grab the paths to all input images in the original input directory
# and shuffle them
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

# compute the training and testing split
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# we'll be using part of the training data for validation
i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# define the datasets that we'll be building
datasets = [
	("training", trainPaths, config.TRAIN_PATH),
	("validation", valPaths, config.VAL_PATH),
	("testing", testPaths, config.TEST_PATH)
]
```

我们从在第 2-6 行**导入我们需要的包开始。我们需要刚刚在上一节中实现的`config`脚本。`paths`模块允许我们获取数据集中所有图像的路径。`random`模块将被用来改变我们的图像路径。最后，`shutil`和`os`将用于将图像文件复制到它们的最终位置。**

**第 10-12 行**抓取我们`ORIG_INPUT_DATASET`中的所有`imagePaths`，然后按随机顺序洗牌。

使用我们的`TRAIN_SPLIT`和`VAL_SPLIT`参数，我们然后创建我们的训练、测试和验证分割(**第 15-22 行**)。

**第 25-29 行**创建一个名为`datasets`的列表。`datasets`中的每个条目由三个元素组成:

1.  拆分的名称
2.  与拆分相关联的图像路径
3.  该分割中的图像将存储到的目录的路径

我们现在可以循环我们的`datasets`:

```py
# loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
	# show which data split we are creating
	print("[INFO] building '{}' split".format(dType))

	# if the output base output directory does not exist, create it
	if not os.path.exists(baseOutput):
		print("[INFO] 'creating {}' directory".format(baseOutput))
		os.makedirs(baseOutput)

	# loop over the input image paths
	for inputPath in imagePaths:
		# extract the filename of the input image and extract the
		# class label ("0" for "negative" and "1" for "positive")
		filename = inputPath.split(os.path.sep)[-1]
		label = filename[-5:-4]

		# build the path to the label directory
		labelPath = os.path.sep.join([baseOutput, label])

		# if the label output directory does not exist, create it
		if not os.path.exists(labelPath):
			print("[INFO] 'creating {}' directory".format(labelPath))
			os.makedirs(labelPath)

		# construct the path to the destination image and then copy
		# the image itself
		p = os.path.sep.join([labelPath, filename])
		shutil.copy2(inputPath, p)
```

在我们的`for`循环中，我们:

1.  如果基础输出目录不存在，则创建它(**第 37-39 行**)
2.  在当前分割(**行 42** )中循环`imagePaths`
3.  提取图像`filename`和类`label` ( **第 45 和 46 行**)
4.  构建输出标签目录的路径(**第 49 行**)
5.  如果`labelPath`不存在，则创建它(**第 52-54 行**)
6.  将原始图像复制到最终的`{split_name}/{label_name}`子目录中

我们现在准备在磁盘上组织我们的数据集。

### **组织我们的图像数据集**

现在我们已经完成了数据拆分器的实现，让我们在数据集上运行它。

确保你们俩:

1.  转到本教程的 ***“下载”*** 部分检索源代码
2.  [从 Kaggle 下载乳腺组织病理学图像数据集](https://www.kaggle.com/paultimothymooney/breast-histopathology-images)

从那里，我们可以在磁盘上组织我们的数据集:

```py
$ python build_dataset.py
[INFO] building 'training' split
[INFO] 'creating datasets/idc/training' directory
[INFO] 'creating datasets/idc/training/0' directory
[INFO] 'creating datasets/idc/training/1' directory
[INFO] building 'validation' split
[INFO] 'creating datasets/idc/validation' directory
[INFO] 'creating datasets/idc/validation/0' directory
[INFO] 'creating datasets/idc/validation/1' directory
[INFO] building 'testing' split
[INFO] 'creating datasets/idc/testing' directory
[INFO] 'creating datasets/idc/testing/0' directory
```

现在让我们来看看`datasets/idc`子目录:

```py
$ tree --dirsfirst --filelimit 10
.
├── datasets
│   ├── idc
│   │   ├── training
│   │   │   ├── 0 [143065 entries]
│   │   │   └── 1 [56753 entries]
│   │   ├── validation
│   │   |   ├── 0 [15962 entries]
│   │   |   └── 1 [6239 entries]
│   │   └── testing
│   │       ├── 0 [39711 entries]
│   │       └── 1 [15794 entries]
│   └── orig [280 entries]
```

请注意我们是如何计算我们的培训、验证和测试分割的。

### **使用 tf.data 和 TensorFlow 数据管道实施我们的培训脚本**

准备好学习如何使用`tf.data`管道训练卷积神经网络了吗？

你当然是！(不然你也不会在这里了吧？)

打开项目目录结构中的`train_model.py`文件，让我们开始工作:

```py
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.cancernet import CancerNet
from pyimagesearch import config
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from tensorflow.data import AUTOTUNE
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os
```

**第 2 行和第 3 行**导入`matplotlib`，设置后台，以便在后台保存数字。

**第 6-16 行**导入我们所需的其余包，值得注意的包括:

*   `CancerNet`:我们将培训 CNN 的实施。我没有在这篇文章中介绍 CancerNet 的实现，因为我在之前的教程中已经广泛介绍过了， [*使用 Keras 的乳腺癌分类和深度学习*](https://pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/) 。
*   `config`:我们的配置文件
*   `Adagrad`:我们将用来训练`CancerNet`的优化器
*   `AUTOTUNE` : TensorFlow 的自动调整功能

因为我们正在构建一个`tf.data`管道来处理磁盘上的图像，所以我们需要定义一个函数来:

1.  从磁盘加载我们的图像
2.  从文件路径中解析类标签
3.  将图像和标签返回给调用函数

如果你读了上周关于 [*的教程，一个关于使用 TensorFlow*](https://pyimagesearch.com/2021/06/14/a-gentle-introduction-to-tf-data-with-tensorflow/) 的 tf.data 的温和介绍(如果你还没有，你现在应该停下来阅读那个指南)，那么下面的`load_images`函数应该看起来很熟悉:

```py
def load_images(imagePath):
	# read the image from disk, decode it, convert the data type to
	# floating point, and resize it
	image = tf.io.read_file(imagePath)
	image = tf.image.decode_png(image, channels=3)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image = tf.image.resize(image, config.IMAGE_SIZE)

	# parse the class label from the file path
	label = tf.strings.split(imagePath, os.path.sep)[-2]
	label = tf.strings.to_number(label, tf.int32)

	# return the image and the label
	return (image, label)
```

`load_images`函数需要一个参数`imagePath`，它是我们想要加载的输入图像的路径。然后，我们继续:

1.  从磁盘加载输入图像
2.  将其解码为无损 PNG 图像
3.  将其从无符号 8 位整数数据类型转换为 float32
4.  将图像大小调整为 *48 *×* 48* 像素

类似地，我们通过分割文件路径分隔符并提取适当的子目录名(存储类标签)来解析来自`imagePath`的类标签。

然后将`image`和`label`返回到调用函数。

***注意:*** *注意，我们没有使用 OpenCV 来加载我们的图像，而是使用 NumPy 来改变数据类型，或者内置 Python 函数来提取类标签，* ***我们改为使用 TensorFlow 函数。这是故意的！*** *尽可能使用 TensorFlow 的功能，让 TensorFlow 更好地优化数据管道。*

接下来，让我们定义一个将执行简单数据扩充的函数:

```py
@tf.function
def augment(image, label):
	# perform random horizontal and vertical flips
	image = tf.image.random_flip_up_down(image)
	image = tf.image.random_flip_left_right(image)

	# return the image and the label
	return (image, label)
```

`augment`函数需要两个参数——我们的输入`image`和类`label`。

我们随机执行水平和垂直翻转(**第 36 行和第 37 行**)，其结果返回给调用函数。

**注意`@tf.function`函数装饰器。**这个装饰器告诉 TensorFlow 使用我们的 Python 函数，并将其转换成 TensorFlow 可调用的图。图形转换允许 TensorFlow 对其进行优化，并使其*更快*——由于我们正在建立一个数据管道，速度是我们希望优化的。

现在让我们解析我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
```

我们有一个单独的(可选的)参数，`--plot`，它是显示我们的损失和准确性的训练图的路径。

```py
# grab all the training, validation, and testing dataset image paths
trainPaths = list(paths.list_images(config.TRAIN_PATH))
valPaths = list(paths.list_images(config.VAL_PATH))
testPaths = list(paths.list_images(config.TEST_PATH))

# calculate the total number of training images in each class and
# initialize a dictionary to store the class weights
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = {}

# loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]
```

**第 49-51 行**分别获取我们的训练、测试和验证目录中所有图像的路径。

然后，我们提取训练类标签，对它们进行一次性编码，并计算每个标签在分割中出现的次数(**第 55-57 行)**。我们执行该操作，因为我们知道在我们的数据集中存在严重的类别不平衡(参见上面的*“乳腺癌组织学数据集”*部分以获得关于类别不平衡的更多信息)。

**第 61 行和第 62 行**通过计算类别权重来说明类别不平衡，该权重将允许代表不足的标签在训练过程的权重更新阶段具有“更多权重”。

接下来，让我们定义我们的培训`tf.data`渠道:

```py
# build the training dataset and data input pipeline
trainDS = tf.data.Dataset.from_tensor_slices(trainPaths)
trainDS = (trainDS
	.shuffle(len(trainPaths))
	.map(load_images, num_parallel_calls=AUTOTUNE)
	.map(augment, num_parallel_calls=AUTOTUNE)
	.cache()
	.batch(config.BS)
	.prefetch(AUTOTUNE)
)
```

我们首先使用`from_tensor_slices`函数(**第 65 行**)创建一个`tf.data.Dataset`的实例。

`tf.data`管道本身由以下步骤组成:

1.  混洗训练集中的所有图像路径
2.  将图像载入缓冲区
3.  对加载的图像执行数据扩充
4.  缓存结果以供后续更快的读取
5.  创建一批数据
6.  允许`prefetch`在后台优化程序

***注:*** *如果你想知道更多关于如何构建`tf.data`管道的细节，请看我之前的教程* [*一个用 TensorFlow*](https://pyimagesearch.com/2021/06/14/a-gentle-introduction-to-tf-data-with-tensorflow/) *对 tf.data 的温和介绍。*

以类似的方式，我们可以构建我们的验证和测试`tf.data`管道:

```py
# build the validation dataset and data input pipeline
valDS = tf.data.Dataset.from_tensor_slices(valPaths)
valDS = (valDS
	.map(load_images, num_parallel_calls=AUTOTUNE)
	.cache()
	.batch(config.BS)
	.prefetch(AUTOTUNE)
)

# build the testing dataset and data input pipeline
testDS = tf.data.Dataset.from_tensor_slices(testPaths)
testDS = (testDS
	.map(load_images, num_parallel_calls=AUTOTUNE)
	.cache()
	.batch(config.BS)
	.prefetch(AUTOTUNE)
)
```

**76-82 线**建立我们的验证管道。请注意，我们在这里没有调用`shuffle`或`augment`，这是因为:

1.  验证数据不需要被打乱
2.  数据扩充不适用于验证数据

我们仍然使用`prefetch`,因为这允许我们在每个时期结束时优化评估例程。

类似地，我们在第 85-91 行的**上创建我们的测试`tf.data`管道。**

如果不考虑数据集初始化，我们会实例化我们的网络架构:

```py
# initialize our CancerNet model and compile it
model = CancerNet.build(width=48, height=48, depth=3,
	classes=1)
opt = Adagrad(lr=config.INIT_LR,
	decay=config.INIT_LR / config.NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
```

**第 94 和 95 行**初始化我们的`CancerNet`模型。我们在这里提供`classes=1`，尽管这是一个两类问题，因为我们在`CancerNet`中有一个 sigmoid 激活函数作为我们的最终层——sigmoid 激活将自然地处理这个两类问题。

然后，我们初始化优化器，并使用二进制交叉熵损失编译模型。

从那里，培训可以开始:

```py
# initialize an early stopping callback to prevent the model from
# overfitting
es = EarlyStopping(
	monitor="val_loss",
	patience=config.EARLY_STOPPING_PATIENCE,
	restore_best_weights=True)

# fit the model
H = model.fit(
	x=trainDS,
	validation_data=valDS,
	class_weight=classWeight,
	epochs=config.NUM_EPOCHS,
	callbacks=[es],
	verbose=1)

# evaluate the model on test set
(_, acc) = model.evaluate(testDS)
print("[INFO] test accuracy: {:.2f}%...".format(acc * 100))
```

**第 103-106 行**初始化我们的`EarlyStopping`标准。如果验证损失在总共`EARLY_STOPPING_PATIENCE`个时期后没有下降，那么我们将停止训练过程并保存我们的 CPU/GPU 周期。

在**行 109-115** 上调用`model.fit`开始训练过程。

**如果你以前曾经用 Keras/TensorFlow 训练过一个模型，那么这个函数调用应该看起来很熟悉，但是请注意，为了利用`tf.data`，我们需要做的就是通过我们的训练和测试`tf.data`管道对象—** ***，真的很容易！***

**第 118 和 119 行**在测试集上评估我们的模型。

最后一步是绘制我们的训练历史:

```py
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["accuracy"], label="train_acc")
plt.plot(H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
```

然后将结果图保存到磁盘。

### **TensorFlow 和 tf.data 管道结果**

我们现在准备使用我们定制的`tf.data`管道来训练一个卷积神经网络。

首先访问本教程的 ***【下载】*** 部分来检索源代码。从那里，您可以启动`train_model.py`脚本:

```py
$ python train_model.py 
1562/1562 [==============================] - 39s 23ms/step - loss: 0.6887 - accuracy: 0.7842 - val_loss: 0.4085 - val_accuracy: 0.8357
Epoch 2/40
1562/1562 [==============================] - 30s 19ms/step - loss: 0.5556 - accuracy: 0.8317 - val_loss: 0.5096 - val_accuracy: 0.7744
Epoch 3/40
1562/1562 [==============================] - 30s 19ms/step - loss: 0.5384 - accuracy: 0.8378 - val_loss: 0.5246 - val_accuracy: 0.7727
Epoch 4/40
1562/1562 [==============================] - 30s 19ms/step - loss: 0.5296 - accuracy: 0.8412 - val_loss: 0.5035 - val_accuracy: 0.7819
Epoch 5/40
1562/1562 [==============================] - 30s 19ms/step - loss: 0.5227 - accuracy: 0.8431 - val_loss: 0.5045 - val_accuracy: 0.7856
Epoch 6/40
1562/1562 [==============================] - 30s 19ms/step - loss: 0.5177 - accuracy: 0.8451 - val_loss: 0.4866 - val_accuracy: 0.7937
434/434 [==============================] - 5s 12ms/step - loss: 0.4184 - accuracy: 0.8333
[INFO] test accuracy: 83.33%...
```

正如你所看到的，我们在检测输入图像中的乳腺癌时获得了大约 83%的准确率。但是*更有趣的*是查看每个历元的速度——**注意，使用`tf.data`会导致历元花费大约 30 秒来完成。**

现在让我们将 epoch speed 与我们关于乳腺癌分类的[原始教程](https://pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/)进行比较，该教程利用了`ImageDataGenerator`功能:

```py
$ python train_model.py
Found 199818 images belonging to 2 classes.
Found 22201 images belonging to 2 classes.
Found 55505 images belonging to 2 classes.
Epoch 1/40
6244/6244 [==============================] - 142s 23ms/step - loss: 0.5954 - accuracy: 0.8211 - val_loss: 0.5407 - val_accuracy: 0.7796
Epoch 2/40
6244/6244 [==============================] - 135s 22ms/step - loss: 0.5520 - accuracy: 0.8333 - val_loss: 0.4786 - val_accuracy: 0.8097
Epoch 3/40
6244/6244 [==============================] - 133s 21ms/step - loss: 0.5423 - accuracy: 0.8358 - val_loss: 0.4532 - val_accuracy: 0.8202
...
Epoch 38/40
6244/6244 [==============================] - 133s 21ms/step - loss: 0.5248 - accuracy: 0.8408 - val_loss: 0.4269 - val_accuracy: 0.8300
Epoch 39/40
6244/6244 [==============================] - 133s 21ms/step - loss: 0.5254 - accuracy: 0.8415 - val_loss: 0.4199 - val_accuracy: 0.8318
Epoch 40/40
6244/6244 [==============================] - 133s 21ms/step - loss: 0.5244 - accuracy: 0.8422 - val_loss: 0.4219 - val_accuracy: 0.8314
```

**使用`ImageDataGenerator`作为我们的数据管道导致需要大约 133 秒来完成的时期。**

数字说明了一切——通过用`tf.data`、**、*替换我们的`ImageDataGenerator`呼叫，我们获得了大约 4.4 倍的加速！***

如果你愿意做一些额外的工作来编写一个定制的`tf.data`管道，那么性能的提升是非常值得的。

## **总结**

在本教程中，您学习了如何使用 TensorFlow 和`tf.data`模块实现自定义数据管道来训练深度神经网络。

在训练我们的网络时，当使用`tf.data`而不是`ImageDataGenerator`时，我们获得了**4.4 倍的加速比**。

缺点是它*要求我们编写额外的代码，包括实现两个自定义函数:*

1.  一个用来加载我们的图像数据，然后解析类标签
2.  另一个执行一些基本的数据扩充

说到数据扩充，下周我们将学习一些更高级的技术，在`tf.data`管道内执行数据扩充——请继续收听该教程，您不会想错过它的。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****