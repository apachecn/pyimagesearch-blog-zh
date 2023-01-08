# 使用 Keras、TensorFlow 和深度学习微调 ResNet

> 原文：<https://pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/>

![](img/3e381f5d6419a7889d42216a05ca27ec.png)

在本教程中，您将学习如何使用 Keras、TensorFlow 和深度学习来微调 ResNet。

几个月前，我在 Twitter 上发帖请求我的粉丝们帮助创建一个数据集，里面有**迷彩服和非迷彩服:**

这个数据集将被用于一个特别的项目，这个项目由 Victor Gevers 负责，他是 GDI 一个受人尊敬的道德黑客。基础，我正在努力(两个星期后会有更多的内容，届时我将公布我们所构建的内容的细节)。

两个 PyImageSearch 的读者，Julia Riede 和 Nitin Rai，不仅站出来帮忙 ***还打了一个本垒打！***

他们俩花了几天时间为每门课下载图片，整理文件，然后上传，这样我和 Victor 就可以用它们来训练一个模型了——**非常感谢你们，Julia 和 Nitin 没有你我们不可能成功！**

在我开始使用伪装与非伪装数据集的几天后，我收到了一封来自 PyImageSearch 读者 Lucas 的电子邮件:

> 嗨，阿德里安，我是图片搜索博客的忠实粉丝。这对我的本科项目帮助很大。
> 
> 我有个问题要问你:
> 
> **有没有关于如何微调 ResNet 的教程？**
> 
> 我看了你的档案，看起来你已经涉及了其他架构的微调(例如。VGGNet)但是我在 ResNet 上什么都找不到。过去几天，我一直试图用 Keras/TensorFlow 对 ResNet 进行微调，但总是出错。
> 
> 如果你能帮我，我将不胜感激。

我已经计划在迷彩与非迷彩服装数据集的基础上微调一个模型，所以帮助卢卡斯似乎是一个自然的选择。

**在本教程的剩余部分，您将:**

1.  探索创新的 ResNet 架构
2.  了解如何使用 Keras 和 TensorFlow 对其进行微调
3.  用于伪装与非伪装服装检测的微调 ResNet

两周后，我将向您展示我和 Victor 应用伪装检测的实际、真实的使用案例——这是一个很棒的故事，您不会想错过它的！

**要了解如何使用 Keras 和 TensorFlow 微调 ResNet，*继续阅读！***

## 使用 Keras、TensorFlow 和深度学习微调 ResNet

在本教程的第一部分，您将了解 ResNet 架构，包括我们如何使用 Keras 和 TensorFlow 微调 ResNet。

从那里，我们将详细讨论我们的迷彩服与普通服装图像数据集。

然后，我们将查看我们的项目目录结构，并继续:

1.  实现我们的配置文件
2.  创建一个 Python 脚本来构建/组织我们的图像数据集
3.  使用 Keras 和 TensorFlow 实现用于微调 ResNet 的第二个 Python 脚本
4.  执行训练脚本并在我们的数据集上微调 ResNet

我们开始吧！

### 什么是 ResNet？

ResNet 是由何等人在他们 2015 年的开创性论文 *[深度残差学习用于图像识别](https://arxiv.org/abs/1512.03385)* 中首次介绍的——该论文被引用了惊人的 43064 次！

2016 年的一篇后续论文， *[深度剩余网络](https://arxiv.org/abs/1603.05027)* 中的身份映射，进行了一系列的消融实验，玩弄了剩余模块中各种成分的包含、移除和排序，最终产生了 ResNet 的一个变种，即:

1.  更容易训练
2.  更能容忍超参数，包括正则化和初始学习率
3.  广义更好

ResNet 可以说是*最重要的*网络架构，因为:

*   Alex net——这在 2012 年重新点燃了研究人员对深度神经网络的兴趣
*   VGGNet——演示了如何仅使用 *3×3* 卷积成功训练更深层次的神经网络(2014 年)
*   Google net——介绍了初始模块/微架构(2014 年)

事实上，ResNet 采用的技术已经成功应用于非计算机视觉任务，包括音频分类和自然语言处理(NLP)！

### ResNet 是如何工作的？

***注:**以下部分改编自我的书第十二章[用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)(从业者捆绑)。*

何等人介绍的原始残差模块依靠的是*标识映射*的概念，即取模块的*原始输入*并将其加到一系列运算的*输出*的过程:

在模块的*顶端*，我们接受一个*输入*到模块(即网络中的前一层)。*右*分支是一个“线性捷径”——它将输入连接到模型底部的加法运算。然后，在剩余模块的*左*分支上，我们应用一系列卷积(两者都是 *3×3* )、激活和批量规格化。这是构造卷积神经网络时要遵循的标准模式。

但令 ResNet 有趣的是，他等人建议将 CONV、RELU 和 BN 层的*输出*加上*原始输入*。

**我们称这种加法为*身份映射*，因为输入(身份)被添加到一系列操作的输出中。**

这也是术语**残差**的使用方式——“残差”输入被添加到一系列层操作的输出中。输入和加法节点之间的连接称为**快捷方式。**

虽然传统的神经网络可以被视为学习函数 *y = f(x)* ，但残差层试图通过 *f(x) + id(x) = f(x) + x* 来逼近 *y* ，其中 *id(x)* 是恒等函数。

这些剩余层*从身份函数*开始，*随着网络的学习而进化*变得更加复杂。这种类型的剩余学习框架允许我们训练比以前提出的架构*更深*的网络。

**此外，由于输入包含在每个剩余模块中，结果是网络可以更快地学习*和更大的学习速率*。****

 *在 2015 年的原始论文中，何等人还包括了对原始残差模块的扩展，称为**瓶颈:**

在这里，我们可以看到相同的身份映射正在发生，只是现在剩余模块的左分支中的 CONV 层已被更新:

1.  我们使用了三个 CONV 层，而不是两个
2.  第一个和最后一个 CONV 层是 *1×1* 卷积
3.  在前两个 CONV 层中学习的过滤器数量是在最终 CONV 中学习的过滤器数量的 1/4

残差模块的这种变化充当了一种形式的*维度缩减*，从而减少了网络中的参数总数(并且这样做不会牺牲准确性)。这种形式的降维被称为**瓶颈。**

何等人 2016 年发表的关于深度剩余网络中*身份映射的论文进行了一系列的消融研究，玩转了剩余模块中各种组件的包含、移除和排序，最终产生了**预激活:**的概念*

在不涉及太多细节的情况下，预激活残差模块重新排列了卷积、批量归一化和激活的执行顺序。

原始剩余模块(具有瓶颈)接受一个输入(即 RELU 激活图)，然后在将该输出添加到原始输入并应用最终 RELU 激活之前应用一系列`(CONV => BN => RELU) * 2 => CONV => BN`。

他们 2016 年的研究表明，相反，应用一系列`(BN => RELU => CONV) * 3`会导致更高精度的模型，更容易训练。

我们称这种层排序方法为*预激活*，因为我们的 RELUs 和批处理规格化被放置在卷积之前*，这与在*卷积之后应用 RELUs 和批处理规格化*的典型方法形成对比。*

关于 ResNet 更完整的回顾，包括如何使用 Keras/TensorFlow 从头实现，一定要参考我的书， *[用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)* 。

### 如何用 Keras 和 TensorFlow 进行微调？

为了用 Keras 和 TensorFlow 微调 ResNet，我们需要使用*预训练的* ImageNet 权重从磁盘加载 ResNet，但*忽略了*全连接层头。

我们可以使用以下代码来实现这一点:

```py
>>> baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
```

检查``baseModel.summary()`` ，你会看到以下内容:

```py
...
conv5_block3_3_conv (Conv2D)    (None, 7, 7, 2048)   1050624     conv5_block3_2_relu[0][0]        
__________________________________________________________________________________________________
conv5_block3_3_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block3_3_conv[0][0]        
__________________________________________________________________________________________________
conv5_block3_add (Add)          (None, 7, 7, 2048)   0           conv5_block2_out[0][0]           
                                                                 conv5_block3_3_bn[0][0]          
__________________________________________________________________________________________________
conv5_block3_out (Activation)   (None, 7, 7, 2048)   0           conv5_block3_add[0][0]           
==================================================================================================
```

在这里，我们可以看到 ResNet 架构中的最后一层(同样，没有全连接的层头)是一个激活层，即 *7 x 7 x 2048。*

我们可以通过接受`baseModel.output`然后应用 *7×7* 平均池来构造一个*新的、刚刚初始化的*层头，然后是我们的全连接层:

```py
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)
```

在构造了`headModel`之后，我们只需要将它附加到 ResNet 模型的主体:

```py
model = Model(inputs=baseModel.input, outputs=headModel)
```

现在，如果我们看一下`model.summary()`，我们可以得出结论，我们已经成功地向 ResNet 添加了一个新的全连接层头，使架构适合微调:

```py
conv5_block3_3_conv (Conv2D)    (None, 7, 7, 2048)   1050624     conv5_block3_2_relu[0][0]        
__________________________________________________________________________________________________
conv5_block3_3_bn (BatchNormali (None, 7, 7, 2048)   8192        conv5_block3_3_conv[0][0]        
__________________________________________________________________________________________________
conv5_block3_add (Add)          (None, 7, 7, 2048)   0           conv5_block2_out[0][0]           
                                                                 conv5_block3_3_bn[0][0]          
__________________________________________________________________________________________________
conv5_block3_out (Activation)   (None, 7, 7, 2048)   0           conv5_block3_add[0][0]           
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 1, 1, 2048)   0           conv5_block3_out[0][0]           
__________________________________________________________________________________________________
flatten (Flatten)               (None, 2048)         0           average_pooling2d[0][0]          
__________________________________________________________________________________________________
dense (Dense)                   (None, 256)          524544      flatten[0][0]                    
__________________________________________________________________________________________________
dropout (Dropout)               (None, 256)          0           dense[0][0]                      
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 2)            514         dropout[0][0]                    
==================================================================================================
```

在本教程的剩余部分，我将为您提供一个使用 Keras 和 TensorFlow 微调 ResNet 的完整工作示例。

### 我们的迷彩服和普通服装数据集

在本教程中，我们将训练一个**迷彩服对普通服装**探测器。

我将在两周内讨论*为什么*我们要建立一个迷彩服检测器，但目前，让它作为一个如何用 Keras 和 TensorFlow 微调 ResNet 的独立示例。

我们在这里使用的数据集是由 PyImageSearch 的读者[朱莉娅·里德](https://github.com/jriede/ml-data/tree/master/camouflage-clothing)和[尼廷·拉伊](https://twitter.com/imneonizer/status/1225749289491554305)策划的。

数据集由两个类组成，每个类包含相同数量的图像:

*   `camouflage_clothes`:7949 张图片
*   `normal_clothes`:7949 张图片

在**图 6** 中可以看到每类图像的样本。

在本教程的剩余部分，您将学习如何微调 ResNet 来预测这两个类，您获得的知识将使您能够在自己的数据集上微调 ResNet。

### 下载我们的迷彩服和普通服装数据集

迷彩服与普通服装数据集可直接从 Kaggle 下载:

[https://www . ka ggle . com/imneonizer/normal-vs-迷彩服](https://www.kaggle.com/imneonizer/normal-vs-camouflage-clothes)

只需点击*“下载”*按钮(**图 7** )即可下载数据集的`.zip`档案。

### 项目结构

一定要从这篇博文的 ***“下载”*** 部分抓取并解压代码。让我们花点时间检查一下我们项目的组织结构:

```py
$ tree --dirsfirst --filelimit 10
.
├── 8k_normal_vs_camouflage_clothes_images
│   ├── camouflage_clothes [7949 entries]
│   └── normal_clothes [7949 entries]
├── pyimagesearch
│   ├── __init__.py
│   └── config.py
├── build_dataset.py
├── camo_detector.model
├── normal-vs-camouflage-clothes.zip
├── plot.png
└── train_camo_detector.py

4 directories, 7 files
```

如您所见，我已经将数据集(`normal-vs-camouflage-clothes.zip`)放在我们项目的根目录中，并提取了文件。其中的图像现在位于`8k_normal_vs_camouflage_clothes_images`目录中。

今天的`pyimagesearch`模块带有一个单独的 Python 配置文件(`config.py`)，其中包含了我们重要的路径和变量。我们将在下一节回顾这个文件。

我们的 Python 驱动程序脚本包括:

*   将我们的数据分成训练、测试和验证子目录
*   `train_camo_detector.py`:用 Python，TensorFlow/Keras，微调训练一个伪装分类器

### 我们的配置文件

在我们能够(1)构建我们的伪装与非伪装图像数据集和(2)在我们的图像数据集上微调 ResNet 之前，让我们首先创建一个简单的配置文件来存储我们所有重要的图像路径和变量。

打开项目中的`config.py`文件，并插入以下代码:

```py
# import the necessary packages
import os

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "8k_normal_vs_camouflage_clothes_images"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "camo_not_camo"

# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
```

模块导入允许我们直接在配置文件中构建动态路径。

我们现有的输入数据集路径应该放在**第 5 行**(此时您应该已经下载了 Kaggle 数据集)。

我们的新数据集目录的路径显示在第 9 行的**上，该目录将包含我们的训练、测试和验证分割。这条路径将由`build_dataset.py`脚本*创建*。**

每个类的三个子目录(我们有两个类)也将被创建(**第 12-14 行**)——到我们的训练、验证和测试数据集分割的路径。每个都将由我们数据集中的图像子集填充。

接下来，我们将定义分割百分比和分类:

```py
# define the amount of data that will be used training
TRAIN_SPLIT = 0.75

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.1

# define the names of the classes
CLASSES = ["camouflage_clothes", "normal_clothes"]
```

训练数据将由所有可用数据的 75%表示(**行 17** )，其中 10%将被标记为验证(**行 21** )。

我们的迷彩服和常服类别在**线 24** 处定义。

我们将总结一些超参数和我们的输出模型路径:

```py
# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 32
NUM_EPOCHS = 20

# define the path to the serialized output model after training
MODEL_PATH = "camo_detector.model"
```

在第 28-30 行**中设置初始学习率、批量和训练的时期数。**

微调后输出序列化的基于 ResNet 的伪装分类模型的路径将存储在**行 33** 定义的路径中。

### 实现我们的伪装数据集构建器脚本

实现了配置文件后，让我们继续创建数据集构建器，它将:

1.  将我们的数据集分别分成训练集、验证集和测试集
2.  在磁盘上组织我们的图像，这样我们就可以使用 Keras 的`ImageDataGenerator`类和相关的`flow_from_directory`函数来轻松地微调 ResNet

打开`build_dataset.py`，让我们开始吧:

```py
# import the necessary packages
from pyimagesearch import config
from imutils import paths
import random
import shutil
import os
```

我们从导入前一节中的`config`和`paths`模块开始，这将帮助我们找到磁盘上的图像文件。Python 中内置的三个模块将用于混合路径和创建目录/子目录。

让我们继续获取数据集中所有原始图像的路径:

```py
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

**第 25-29 行**定义了我们将在这个脚本的剩余部分构建的数据集分割。让我们继续:

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
		# extract the filename of the input image along with its
		# corresponding class label
		filename = inputPath.split(os.path.sep)[-1]
		label = inputPath.split(os.path.sep)[-2]

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

最后一个代码块处理将图像从原始位置复制到目标路径；在此过程中会创建目录和子目录。让我们更详细地回顾一下:

*   我们循环遍历每个`datasets`，如果它不存在就创建目录(**第 32-39 行**)
*   对于我们的每个`imagePaths`，我们继续:
    *   提取`filename`和`label`类(**行 45 和 46**
    *   构建标签目录的路径(**行 49** )并创建子目录，如果需要的话(**行 52-54** )
    *   将图像从源目录复制到目的地(**第 58 行和第 59 行**)

在下一节中，我们将相应地构建数据集。

### 构建伪装图像数据集

让我们现在建立和组织我们的图像伪装数据集。

确保你有:

1.  使用本教程的 ***【下载】*** 部分下载源代码
2.  按照上面的“*下载我们的迷彩服与普通服装数据集*”部分下载数据集

从那里，打开一个终端，并执行以下命令:

```py
$ python build_dataset.py
[INFO] building 'training' split
[INFO] 'creating camo_not_camo/training' directory
[INFO] 'creating camo_not_camo/training/normal_clothes' directory
[INFO] 'creating camo_not_camo/training/camouflage_clothes' directory
[INFO] building 'validation' split
[INFO] 'creating camo_not_camo/validation' directory
[INFO] 'creating camo_not_camo/validation/camouflage_clothes' directory
[INFO] 'creating camo_not_camo/validation/normal_clothes' directory
[INFO] building 'testing' split
[INFO] 'creating camo_not_camo/testing' directory
[INFO] 'creating camo_not_camo/testing/normal_clothes' directory
[INFO] 'creating camo_not_camo/testing/camouflage_clothes' directory
```

然后，您可以使用`tree`命令来检查`camo_not_camo`目录，以验证每个训练、测试和验证分割都已创建:

```py
$ tree camo_not_camo --filelimit 20
camo_not_camo
├── testing
│   ├── camouflage_clothes [2007 entries]
│   └── normal_clothes [1968 entries]
├── training
│   ├── camouflage_clothes [5339 entries]
│   └── normal_clothes [5392 entries]
└── validation
    ├── camouflage_clothes [603 entries]
    └── normal_clothes [589 entries]

9 directories, 0 files
```

### 使用 Keras 和 TensorFlow 实现我们的 ResNet 微调脚本

在磁盘上创建并正确组织数据集后，让我们学习如何使用 Keras 和 TensorFlow 微调 ResNet。

打开`train_camo_detector.py`文件，插入以下代码:

```py
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
```

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))
```

我们有一个单独的命令行参数`--plot`，它是一个图像文件的路径，该文件将包含我们的精度/损失训练曲线。我们的其他配置在我们之前查看的 Python 配置文件中。

**第 30-32 行**分别确定训练、验证和测试图像的总数。

接下来，我们将准备[数据扩充](https://pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/):

```py
# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=25,
	zoom_range=0.1,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.2,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()

# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean
```

数据扩充允许训练我们图像的时间突变，包括随机旋转、缩放、移位、剪切、翻转和均值减法。**第 35-42 行**用这些参数的选择初始化我们的训练数据扩充对象。类似地，**第 46 行**初始化确认/测试数据增加对象(它将仅用于均值减法)。

我们的两个数据扩充对象都被设置为实时执行均值减法(**第 51-53 行**)。

我们现在将从我们的数据扩充对象实例化三个 Python 生成器:

```py
# initialize the training generator
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=config.BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BS)
```

让我们加载我们的 ResNet50 分类模型，并为[微调](https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/)做准备:

```py
# load the ResNet-50 network, ensuring the head FC layer sets are left
# off
print("[INFO] preparing model...")
baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process
for layer in baseModel.layers:
	layer.trainable = False
```

微调的最后一步是确保我们 CNN 的基础权重被冻结(**行 103 和 104**)——我们只想训练(即微调)网络的负责人。

如果你需要温习微调的概念，请参考我的[微调文章](https://pyimagesearch.com/tag/fine-tuning/)，特别是 *[用 Keras 和深度学习](https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/)进行微调。*

我们现在已经准备好使用 TensorFlow、Keras 和深度学习来微调我们基于 ResNet 的伪装检测器:

```py
# compile the model
opt = Adam(lr=config.INIT_LR, decay=config.INIT_LR / config.NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the model
print("[INFO] training model...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // config.BS,
	validation_data=valGen,
	validation_steps=totalVal // config.BS,
	epochs=config.NUM_EPOCHS)
```

首先，我们用[学习率衰减](https://pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/)和 [Adam 优化器](https://pyimagesearch.com/2019/10/07/is-rectified-adam-actually-better-than-adam/)使用`"binary_crossentropy"`损失来编译我们的模型，因为这是两类问题(**第 107-109 行**)。如果您使用两类以上的数据进行训练，请确保将您的`loss`设置为`"categorical_crossentropy"`。

**第 113-118 行**然后使用我们的训练和验证数据生成器训练我们的模型。

培训完成后，我们将在测试集上评估我们的模型:

```py
# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // config.BS) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))

# serialize the model to disk
print("[INFO] saving model...")
model.save(config.MODEL_PATH, save_format="h5")
```

```py
# plot the training loss and accuracy
N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
```

一旦绘图生成， **Line 151** 将它保存到磁盘上由我们的`--plot`命令行参数指定的位置。

### 使用 Keras 和 TensorFlow 结果微调 ResNet

我们现在准备用 Keras 和 TensorFlow 对 ResNet 进行微调。

确保你有:

*   使用本教程的 ***【下载】*** 部分下载源代码
*   按照上面的“*下载我们的迷彩服与普通服装数据集”*部分下载数据集
*   执行`build_dataset.py`脚本，将数据集组织到项目目录结构中进行训练

从那里，打开一个终端，运行`train_camo_detector.py`脚本:

```py
$ python train_camo_detector.py
Found 10731 images belonging to 2 classes.
Found 1192 images belonging to 2 classes.
Found 3975 images belonging to 2 classes.
[INFO] preparing model...
[INFO] training model...
Epoch 1/20
335/335 [==============================] - 311s 929ms/step - loss: 0.1736 - accuracy: 0.9326 - val_loss: 0.1050 - val_accuracy: 0.9671
Epoch 2/20
335/335 [==============================] - 305s 912ms/step - loss: 0.0997 - accuracy: 0.9632 - val_loss: 0.1028 - val_accuracy: 0.9586
Epoch 3/20
335/335 [==============================] - 305s 910ms/step - loss: 0.0729 - accuracy: 0.9753 - val_loss: 0.0951 - val_accuracy: 0.9730
...
Epoch 18/20
335/335 [==============================] - 298s 890ms/step - loss: 0.0336 - accuracy: 0.9878 - val_loss: 0.0854 - val_accuracy: 0.9696
Epoch 19/20
335/335 [==============================] - 298s 891ms/step - loss: 0.0296 - accuracy: 0.9896 - val_loss: 0.0850 - val_accuracy: 0.9679
Epoch 20/20
335/335 [==============================] - 299s 894ms/step - loss: 0.0275 - accuracy: 0.9905 - val_loss: 0.0955 - val_accuracy: 0.9679
[INFO] evaluating network...
                   precision    recall  f1-score   support

    normal_clothes       0.95      0.99      0.97      2007
camouflage_clothes       0.99      0.95      0.97      1968

          accuracy                           0.97      3975
         macro avg       0.97      0.97      0.97      3975
      weighted avg       0.97      0.97      0.97      3975

[INFO] saving model...
```

在这里，你可以看到我们在普通服装与迷彩服检测器上获得了 **~97%的准确率**。

我们的训练图如下所示:

我们的培训损失比我们的验证损失减少得更快；此外，在训练接近尾声时，验证损失可能上升，这表明模型*可能*过度拟合。

未来的实验应该考虑对模型应用额外的正则化，以及收集额外的训练数据。

两周后，我将向您展示如何将这个经过微调的 ResNet 模型用于实际的应用程序中！

敬请关注帖子；你不会想错过的！

### 信用

没有以下条件，本教程是不可能完成的:

*   GDI 的维克多·格弗斯。基金会，他让我注意到了这个项目
*   Nitin Rai 负责策划普通服装与迷彩服的对比，并将数据集发布在 Kaggle 上
*   Julia Riede 策划了一个数据集的变体

此外，我要感谢[韩等人](https://www.researchgate.net/publication/322621180_Deep_neural_networks_show_an_equivalent_and_often_superior_performance_to_dermatologists_in_onychomycosis_diagnosis_Automatic_construction_of_onychomycosis_datasets_by_region-based_convolutional_deep_)在此图片标题中使用的 ResNet-152 可视化。

## 摘要

在本教程中，您学习了如何使用 Keras 和 TensorFlow 微调 ResNet。

微调是以下过程:

1.  采用预先训练的深度神经网络(在这种情况下，ResNet)
2.  从网络中移除全连接的层头
3.  在网络主体的顶部放置新的、刚刚初始化的层头
4.  可选地冻结身体中各层的重量
5.  训练模型，使用预先训练的权重作为起点来帮助模型更快地学习

使用微调，我们可以获得更高精度的模型，通常需要更少的工作、数据和训练时间。

作为一个实际应用，我们在迷彩服和非迷彩服的数据集上对 ResNet 进行了微调。

这个数据集是由 PyImageSearch 的读者 Julia Riede 和 T2 Nitin Rai 为我们策划和整理的——没有他们，这个教程以及我和 Victor Gevers 正在进行的项目就不可能完成！**如果你在网上看到朱莉娅和尼廷，请向他们致谢。**

在两周内，我将详细介绍我和 Victor Gevers 正在进行的项目，其中包括我们最近在 PyImageSearch 上涉及的以下主题:

*   人脸检测
*   年龄检测
*   从深度学习数据集中移除重复项
*   迷彩服与非迷彩服检测的模型微调

这是一个很棒的帖子，有非常真实的应用程序，用计算机视觉和深度学习让世界变得更美好——你不会想错过它的！

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****