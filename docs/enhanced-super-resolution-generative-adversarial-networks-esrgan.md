# 增强型超分辨率生成对抗网络

> 原文：<https://pyimagesearch.com/2022/06/13/enhanced-super-resolution-generative-adversarial-networks-esrgan/>

* * *

## **目录**

* * *

## [](#TOC)

 **上周我们学习了超分辨率甘斯。他们在实现超分辨率图像的更好清晰度方面做得非常好。但是，在超分辨率领域，甘斯的路走到尽头了吗？

深度学习的一个共同主题是，成长永不停止。因此，我们转向增强的超分辨率 gan。顾名思义，它对原来的 SRGAN 架构进行了许多更新，极大地提高了性能和可视化。

在本教程中，您将学习如何使用 tensorflow 实现 ESRGAN。

本课是关于 **GANs 201** 的 4 部分系列中的第 2 部分:

1.  [](https://pyimg.co/lgnrx)
2.  ******【增强超分辨率生成对抗网络(ESRGAN)* (本教程)*****
3.  ****用于图像到图像转换的像素 2 像素氮化镓****
4.  ****用于图像到图像转换的 CycleGAN】****

 *****要了解如何**实现 ESRGAN** ，*请继续阅读。***

* * *

### [**前言**](#TOC)

GANs 同时训练两个神经网络:鉴别器和生成器。生成器创建假图像，而鉴别器判断它们是真的还是假的。

SRGANs 将这一思想应用于图像超分辨率领域。发生器产生超分辨率图像，鉴别器判断真假。

* * *

### [**增强型超分辨率甘斯**](#TOC)

建立在 SRGANs 领导的基础上，ESRGAN 的主要目标是引入模型修改，从而提高训练效率并降低复杂性。

SRGANs 的简要概述:

*   将低分辨率图像作为输入提供给生成器，并将超分辨率图像作为输出。
*   将这些预测通过鉴别器，得到真实或虚假的品牌。
*   使用 VGG 网添加感知损失(像素方面)来增加我们预测的假图像的清晰度。

但是 ESRGANs 带来了哪些更新呢？

首先，发电机采取了一些主要措施来确保性能的提高:

*   **删除批量标准化层:**对 SRGAN 架构的简要回顾将显示批量标准化层在整个生成器架构中被广泛使用。由于提高了性能和降低了计算复杂性，ESRGANs 完全放弃了 BN 层的使用。

*   **Residual in Residual Dense Block:**标准残差块的升级，这种特殊的结构允许一个块中的所有层输出传递给后续层，如图**图 1** 所示。这里的直觉是，模型可以访问许多可供选择的特性，并确定其相关性。此外，ESRGAN 使用**残差缩放**来缩小残差输出，以防止不稳定。

*   对于鉴频器，主要增加的是**相对论损耗**。它估计真实图像比假预测图像更真实的概率**。自然地，将它作为损失函数加入使得模型能够克服相对论损失。**
*   **感知损失**有一点改变，使损失基于激活功能之前而不是激活功能之后的特征，如上周 [SRGAN 论文](https://pyimg.co/lgnrx)所示。
*   **总损失**现在是 GAN 损失、感知损失以及地面真实高分辨率和预测图像之间的像素距离的组合。

这些添加有助于显著改善结果。在我们的实现中，我们忠实于论文，并对传统的 SRGAN 进行了这些更新，以提高超分辨率结果。

ESRGAN 背后的核心理念不仅是提高结果，而且使流程更加高效。因此，本文并没有全面谴责批量规范化的使用。尽管如此，它指出，刮 BN 层的使用将有利于我们的特定任务，即使在最小的像素的相似性是必要的。

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

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

* * *

### [**项目结构**](#TOC)

我们首先需要回顾我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

从这里，看一下目录结构:

```py
!tree .
.
├── create_tfrecords.py
├── inference.py
├── outputs
├── pyimagesearch
│   ├── config.py
│   ├── data_preprocess.py
│   ├── __init__.py
│   ├── losses.py
│   ├── esrgan.py
│   ├── esrgan_training.py
│   ├── utils.py
│   └── vgg.py
└── train_esrgan.py

2 directories, 11 files
```

在`pyimagesearch`目录中，我们有:

*   `config.py`:包含完整项目的端到端配置管道。
*   `data_preprocess.py`:包含有助于数据处理的功能。
*   `__init__.py`:使目录像 python 包一样运行。
*   `losses.py`:初始化训练 ESRGAN 所需的损耗。
*   `esrgan.py`:包含 ESRGAN 架构。
*   `esrgan_training.py`:包含运行 ESRGAN 训练的训练类。
*   `utils.py`:包含附加实用程序
*   `vgg.py`:初始化用于感知损失计算的 VGG 模型。

在根目录中，我们有:

*   `create_tfrecords.py`:从我们将使用的数据集创建`TFRecords`。
*   `inference.py`:使用训练好的模型进行推理。
*   `train_srgan.py`:使用`esrgan.py`和`esrgan_training.py`脚本执行 ESRGAN 训练。

* * *

### [**配置先决条件**](#TOC)

位于`pyimagesearch`目录中的`config.py`脚本包含了整个项目所需的几个参数和路径。将配置变量分开是一个很好的编码实践。为此，让我们转向`config.py`剧本。

```py
# import the necessary packages
import os

# name of the TFDS dataset we will be using
DATASET = "div2k/bicubic_x4"

# define the shard size and batch size
SHARD_SIZE = 256
TRAIN_BATCH_SIZE = 64
INFER_BATCH_SIZE = 8

# dataset specs
HR_SHAPE = [96, 96, 3]
LR_SHAPE = [24, 24, 3]
SCALING_FACTOR = 4
```

我们首先在第 5 行引用我们项目的数据集。

`TFRecords`的`SHARD_SIZE`定义在**线 8** 上。随后是第 9 行**和第 10 行**上的`TRAIN_BATCH_SIZE`和`INFER_BATCH_SIZE`定义。

我们的高分辨率输出图像将具有尺寸`96 x 96 x 3`，而我们的输入低分辨率图像将具有尺寸`24 x 24 x 3` ( **行 13 和 14** )。相应地，在**线 15** 上`SCALING_FACTOR`被设置为`4`。

```py
# GAN model specs
FEATURE_MAPS = 64
RESIDUAL_BLOCKS = 16
LEAKY_ALPHA = 0.2
DISC_BLOCKS = 4
RESIDUAL_SCALAR = 0.2

# training specs
PRETRAIN_LR = 1e-4
FINETUNE_LR = 3e-5
PRETRAIN_EPOCHS = 1500
FINETUNE_EPOCHS = 1000
STEPS_PER_EPOCH = 10

# define the path to the dataset
BASE_DATA_PATH = "dataset"
DIV2K_PATH = os.path.join(BASE_DATA_PATH, "div2k")
```

正如我们对 SRGAN 所做的那样，该架构由剩余网络组成。首先，我们设置在`Conv2D`层使用的滤镜数量(**第 18 行**)。在**第 19 行**上，我们定义了剩余块的数量。我们的`ReLU`功能的`alpha`参数设置在第 20 行的**上。**

鉴别器架构将基于`DISC_BLOCKS` ( **第 21 行**)的值实现自动化。现在，我们为`RESIDUAL_SCALAR`定义一个值，这将有助于我们将剩余块输出缩放到各个级别，并保持训练过程稳定(**第 22 行**)。

现在，重复一下我们的 SRGAN 参数(学习率、时期等。)在第 25-29 行完成。我们将对我们的 GAN 进行预训练，然后对其进行全面训练以进行比较。出于这个原因，我们为预训练的 GAN 和完全训练的 GAN 定义了学习率和时期。

设置`BASE_DATA_PATH`来定义存储数据集的。`DIV2K_PATH`引用了`DIV2K`数据集(**第 32 和 33 行**)。 [`div2k`](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 数据集非常适合辅助图像超分辨率研究，因为它包含各种高分辨率图像。

```py
# define the path to the tfrecords for GPU training
GPU_BASE_TFR_PATH = "tfrecord"
GPU_DIV2K_TFR_TRAIN_PATH = os.path.join(GPU_BASE_TFR_PATH, "train")
GPU_DIV2K_TFR_TEST_PATH = os.path.join(GPU_BASE_TFR_PATH, "test")

# define the path to the tfrecords for TPU training
TPU_BASE_TFR_PATH = "gs://<PATH_TO_GCS_BUCKET>/tfrecord"
TPU_DIV2K_TFR_TRAIN_PATH = os.path.join(TPU_BASE_TFR_PATH, "train")
TPU_DIV2K_TFR_TEST_PATH = os.path.join(TPU_BASE_TFR_PATH, "test")

# path to our base output directory
BASE_OUTPUT_PATH = "outputs"

# GPU training ESRGAN model paths
GPU_PRETRAINED_GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH,
	"models", "pretrained_generator")
GPU_GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH, "models",
	"generator")

# TPU training ESRGAN model paths
TPU_OUTPUT_PATH = "gs://<PATH_TO_GCS_BUCKET>/outputs"
TPU_PRETRAINED_GENERATOR_MODEL = os.path.join(TPU_OUTPUT_PATH,
	"models", "pretrained_generator")
TPU_GENERATOR_MODEL = os.path.join(TPU_OUTPUT_PATH, "models",
	"generator")

# define the path to the inferred images and to the grid image
BASE_IMAGE_PATH = os.path.join(BASE_OUTPUT_PATH, "images")
GRID_IMAGE_PATH = os.path.join(BASE_OUTPUT_PATH, "grid.png")
```

因此，为了比较训练效率，我们将在 TPU 和 GPU 上训练 GAN。为此，我们必须为 GPU 训练和 TPU 训练分别创建引用数据和输出的路径。

在**的第 36-38 行**，我们为 GPU 训练定义了`TFRecords`。在**第 41-43 行**，我们为 TPU 训练定义了`TFRecords`。

现在，我们在**线 46** 上定义基本输出路径。接下来是经过 GPU 训练的 ESRGAN 发电机模型的参考路径(**第 49-52 行**)。我们对 TPU 培训的 ESRGAN 发电机模型做同样的事情(**线 55-59** )。

完成所有设置后，剩下的唯一任务是引用推断图像的路径(**行 62 和 63** )。

* * *

### [**实现数据处理实用程序**](#TOC)

训练 GANs 需要大量的计算能力和数据。为了确保我们有足够的数据，我们将采用几种数据扩充技术。让我们看看位于`pyimagesearch`目录中的`data_preprocess.py`脚本中的那些。

```py
# import the necessary packages
from tensorflow.io import FixedLenFeature
from tensorflow.io import parse_single_example
from tensorflow.io import parse_tensor
from tensorflow.image import flip_left_right
from tensorflow.image import rot90
import tensorflow as tf

# define AUTOTUNE object
AUTO = tf.data.AUTOTUNE

def random_crop(lrImage, hrImage, hrCropSize=96, scale=4):
	# calculate the low resolution image crop size and image shape
	lrCropSize = hrCropSize // scale
	lrImageShape = tf.shape(lrImage)[:2]

	# calculate the low resolution image width and height offsets
	lrW = tf.random.uniform(shape=(),
		maxval=lrImageShape[1] - lrCropSize + 1, dtype=tf.int32)
	lrH = tf.random.uniform(shape=(),
		maxval=lrImageShape[0] - lrCropSize + 1, dtype=tf.int32)

	# calculate the high resolution image width and height
	hrW = lrW * scale
	hrH = lrH * scale

	# crop the low and high resolution images
	lrImageCropped = tf.slice(lrImage, [lrH, lrW, 0], 
		[(lrCropSize), (lrCropSize), 3])
	hrImageCropped = tf.slice(hrImage, [hrH, hrW, 0],
		[(hrCropSize), (hrCropSize), 3])

	# return the cropped low and high resolution images
	return (lrImageCropped, hrImageCropped)
```

考虑到我们将在这个项目中使用的 TensorFlow 包装器的数量，为空间优化定义一个`tf.data.AUTOTUNE`对象是一个好主意。

我们定义的第一个数据增强函数是`random_crop` ( **第 12 行**)。它接受以下参数:

*   `lrImage`:低分辨率图像。
*   `hrImage`:高分辨率图像。
*   `hrCropSize`:用于低分辨率裁剪计算的高分辨率裁剪尺寸。
*   `scale`:我们用来计算低分辨率裁剪的因子。

然后计算左右宽度和高度偏移(**第 18-21 行**)。

为了计算相应的高分辨率值，我们简单地将低分辨率值乘以比例因子(**行 24 和 25** )。

使用这些值，我们裁剪出低分辨率图像及其对应的高分辨率图像，并返回它们(**第 28-34 行**)

```py
def get_center_crop(lrImage, hrImage, hrCropSize=96, scale=4):
	# calculate the low resolution image crop size and image shape
	lrCropSize = hrCropSize // scale
	lrImageShape = tf.shape(lrImage)[:2]

	# calculate the low resolution image width and height
	lrW = lrImageShape[1] // 2
	lrH = lrImageShape[0] // 2

	# calculate the high resolution image width and height
	hrW = lrW * scale
	hrH = lrH * scale

	# crop the low and high resolution images
	lrImageCropped = tf.slice(lrImage, [lrH - (lrCropSize // 2),
		lrW - (lrCropSize // 2), 0], [lrCropSize, lrCropSize, 3])
	hrImageCropped = tf.slice(hrImage, [hrH - (hrCropSize // 2),
		hrW - (hrCropSize // 2), 0], [hrCropSize, hrCropSize, 3])

	# return the cropped low and high resolution images
	return (lrImageCropped, hrImageCropped)
```

数据扩充实用程序中的下一行是`get_center_crop` ( **第 36 行**)，它接受以下参数:

*   `lrImage`:低分辨率图像。
*   `hrImage`:高分辨率图像。
*   `hrCropSize`:用于低分辨率裁剪计算的高分辨率裁剪尺寸。
*   `scale`:我们用来计算低分辨率裁剪的因子。

就像我们为之前的函数创建裁剪大小值一样，我们在第 38 行和第 39 行得到 lr 裁剪大小值和图像形状。

现在，要获得中心像素坐标，我们只需将低分辨率形状除以`2` ( **第 42 行和第 43 行**)。

为了获得相应的高分辨率中心点，将 lr 中心点乘以比例因子(**第 46 和 47 行**)。

```py
def random_flip(lrImage, hrImage):
	# calculate a random chance for flip
	flipProb = tf.random.uniform(shape=(), maxval=1)
	(lrImage, hrImage) = tf.cond(flipProb < 0.5,
		lambda: (lrImage, hrImage),
		lambda: (flip_left_right(lrImage), flip_left_right(hrImage)))

	# return the randomly flipped low and high resolution images
	return (lrImage, hrImage)
```

我们有`random_flip`功能来翻转**行 58** 上的图像。它接受低分辨率和高分辨率图像作为其参数。

基于使用`tf.random.uniform`的翻转概率值，我们翻转我们的图像并返回它们(**第 60-66 行**)。

```py
def random_rotate(lrImage, hrImage):
	# randomly generate the number of 90 degree rotations
	n = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)

	# rotate the low and high resolution images
	lrImage = rot90(lrImage, n)
	hrImage = rot90(hrImage, n)

	# return the randomly rotated images
	return (lrImage, hrImage)
```

在**行 68** 上，我们有另一个数据扩充函数叫做`random_rotate`，它接受低分辨率图像和高分辨率图像作为它的参数。

变量`n`生成一个值，这个值稍后将帮助应用于我们的图像集的旋转量(**第 70-77 行**)。

```py
def read_train_example(example):
	# get the feature template and  parse a single image according to
	# the feature template
	feature = {
		"lr": FixedLenFeature([], tf.string),
		"hr": FixedLenFeature([], tf.string),
	}
	example = parse_single_example(example, feature)

	# parse the low and high resolution images
	lrImage = parse_tensor(example["lr"], out_type=tf.uint8)
	hrImage = parse_tensor(example["hr"], out_type=tf.uint8)

	# perform data augmentation
	(lrImage, hrImage) = random_crop(lrImage, hrImage)
	(lrImage, hrImage) = random_flip(lrImage, hrImage)
	(lrImage, hrImage) = random_rotate(lrImage, hrImage)

	# reshape the low and high resolution images
	lrImage = tf.reshape(lrImage, (24, 24, 3))
	hrImage = tf.reshape(hrImage, (96, 96, 3))

	# return the low and high resolution images
	return (lrImage, hrImage)
```

数据扩充功能完成后，我们可以转到第 79 行**上的图像读取功能`read_train_example`。该函数在单个图像集(低分辨率和相应的高分辨率图像)中运行**

在第**行第 82-90** 行，我们创建一个 lr，hr 特征模板，并基于它解析示例集。

现在，我们在 lr-hr 集合上应用数据扩充函数(**第 93-95 行**)。然后，我们将 lr-hr 图像重塑回它们需要的尺寸**(第 98-102 行**)。

```py
def read_test_example(example):
	# get the feature template and  parse a single image according to
	# the feature template
	feature = {
		"lr": FixedLenFeature([], tf.string),
		"hr": FixedLenFeature([], tf.string),
	}
	example = parse_single_example(example, feature)

	# parse the low and high resolution images
	lrImage = parse_tensor(example["lr"], out_type=tf.uint8)
	hrImage = parse_tensor(example["hr"], out_type=tf.uint8)

	# center crop both low and high resolution image
	(lrImage, hrImage) = get_center_crop(lrImage, hrImage)

	# reshape the low and high resolution images
	lrImage = tf.reshape(lrImage, (24, 24, 3))
	hrImage = tf.reshape(hrImage, (96, 96, 3))

	# return the low and high resolution images
	return (lrImage, hrImage)
```

我们为推断图像创建一个类似于`read_train_example`的函数，称为`read_test_example`，它接受一个 lr-hr 图像集(**行 104** )。除了数据扩充过程(**第 107-125 行**)之外，重复前面函数中所做的一切。

```py
def load_dataset(filenames, batchSize, train=False):
	# get the TFRecords from the filenames
	dataset = tf.data.TFRecordDataset(filenames, 
		num_parallel_reads=AUTO)

	# check if this is the training dataset
	if train:
		# read the training examples
		dataset = dataset.map(read_train_example,
			num_parallel_calls=AUTO)
	# otherwise, we are working with the test dataset
	else:
		# read the test examples
		dataset = dataset.map(read_test_example,
			num_parallel_calls=AUTO)

	# batch and prefetch the data
	dataset = (dataset
		.shuffle(batchSize)
		.batch(batchSize)
		.repeat()
		.prefetch(AUTO)
	)

	# return the dataset
	return dataset
```

现在，总结一下，我们在**行 127** 上有了`load_dataset`函数。它接受文件名、批量大小和一个布尔变量，指示模式是训练还是推理。

在第 129 行和第 130 行上，我们从提供的文件名中得到`TFRecords`。如果模式设置为 train，我们将`read_train_example`函数映射到数据集。这样，它里面的所有条目都通过`read_train_example`函数传递(**第 133-136 行**)。

如果模式是推理，我们将`read_test_example`函数移至数据集(**第 138-141 行**)。

随着我们的数据集的创建，它现在被批处理、混洗并设置为自动预取(**第 144-152 行**)。

* * *

### [**实现 ESRGAN 架构**](#TOC)

我们的下一个目的地是位于`pyimagesearch`目录中的`esrgan.py`脚本。该脚本包含完整的 ESRGAN 架构。我们已经讨论了 ESRGAN 带来的变化，现在让我们一个一个地来看一下。

```py
# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Add
from tensorflow.nn import depth_to_space
from tensorflow.keras import Model
from tensorflow.keras import Input

class ESRGAN(object):
	@staticmethod
	def generator(scalingFactor, featureMaps, residualBlocks,
			leakyAlpha, residualScalar):
		# initialize the input layer
		inputs = Input((None, None, 3))
		xIn = Rescaling(scale=1.0/255, offset=0.0)(inputs)

		# pass the input through CONV => LeakyReLU block
		xIn = Conv2D(featureMaps, 9, padding="same")(xIn)
		xIn = LeakyReLU(leakyAlpha)(xIn)
```

为了简化工作流程，最好将 ESRGAN 定义为类模板(**第 14 行**)。

首先，我们算出发电机。第 16 行**上的函数`generator`作为我们的生成器定义，并接受以下参数:**

*   `scalingFactor`:输出图像缩放的决定因素。
*   `featureMaps`:卷积滤波器的数量。
*   `residualBlocks`:添加到架构中的剩余块数。
*   `leakyAlpha`:决定我们漏`ReLU`函数阈值的因子
*   `residualScalar`:保持残差块的输出成比例的值，使得训练稳定。

输入被初始化，像素被缩放到`0`和`1`的范围(**第 19 行和第 20 行**)。

处理后的输入然后通过一个`Conv2D`层，接着是一个`LeakyReLU`激活功能(**行 23 和 24** )。这些层的参数已经在`config.py`中定义过了。

```py
		# construct the residual in residual block
		x = Conv2D(featureMaps, 3, padding="same")(xIn)
		x1 = LeakyReLU(leakyAlpha)(x)
		x1 = Add()([xIn, x1])
		x = Conv2D(featureMaps, 3, padding="same")(x1)
		x2 = LeakyReLU(leakyAlpha)(x)
		x2 = Add()([x1, x2])
		x = Conv2D(featureMaps, 3, padding="same")(x2)
		x3 = LeakyReLU(leakyAlpha)(x)
		x3 = Add()([x2, x3])
		x = Conv2D(featureMaps, 3, padding="same")(x3)
		x4 = LeakyReLU(leakyAlpha)(x)
		x4 = Add()([x3, x4])
		x4 = Conv2D(featureMaps, 3, padding="same")(x4)
		xSkip = Add()([xIn, x4])

		# scale the residual outputs with a scalar between [0,1]
		xSkip = Lambda(lambda x: x * residualScalar)(xSkip)
```

正如我们之前提到的，ESRGAN 在残差块中使用残差。因此，接下来定义基块。

我们开始添加一个`Conv2D`和一个`LeakyReLU`层。由于块的性质，这种层组合的输出`x1`然后被添加到初始输入`x`。这被重复三次，在连接初始输入`xIn`和块输出`x4` ( **行 27-40** )的跳跃连接之前添加最后的`Conv2D`层。

现在，这和原来的论文有点偏离，所有的层都是相互连接的。互连的目的是确保模型在每一步都可以访问前面的特征。基于我们今天使用的任务和数据集，我们的方法足以给出我们想要的结果。

添加跳过连接后，使用**线 43** 上的`residualScalar`变量缩放输出。

```py
		# create a number of residual in residual blocks
		for blockId in range(residualBlocks-1):
			x = Conv2D(featureMaps, 3, padding="same")(xSkip)
			x1 = LeakyReLU(leakyAlpha)(x)
			x1 = Add()([xSkip, x1])
			x = Conv2D(featureMaps, 3, padding="same")(x1)
			x2 = LeakyReLU(leakyAlpha)(x)
			x2 = Add()([x1, x2])
			x = Conv2D(featureMaps, 3, padding="same")(x2)
			x3 = LeakyReLU(leakyAlpha)(x)
			x3 = Add()([x2, x3])
			x = Conv2D(featureMaps, 3, padding="same")(x3)
			x4 = LeakyReLU(leakyAlpha)(x)
			x4 = Add()([x3, x4])
			x4 = Conv2D(featureMaps, 3, padding="same")(x4)
			xSkip = Add()([xSkip, x4])
			xSkip = Lambda(lambda x: x * residualScalar)(xSkip)
```

现在，块重复是使用`for`循环的自动化。根据指定的剩余块数量，将添加块层(**第 46-61 行**)。

```py
		# process the residual output with a conv kernel
		x = Conv2D(featureMaps, 3, padding="same")(xSkip)
		x = Add()([xIn, x])

		# upscale the image with pixel shuffle
		x = Conv2D(featureMaps * (scalingFactor // 2), 3,
			padding="same")(x)
		x = tf.nn.depth_to_space(x, 2)
		x = LeakyReLU(leakyAlpha)(x)

		# upscale the image with pixel shuffle
		x = Conv2D(featureMaps, 3, padding="same")(x)
		x = tf.nn.depth_to_space(x, 2)
		x = LeakyReLU(leakyAlpha)(x)

		# get the output layer
		x = Conv2D(3, 9, padding="same", activation="tanh")(x)
		output = Rescaling(scale=127.5, offset=127.5)(x)

		# create the generator model
		generator = Model(inputs, output)

		# return the generator model
		return generator
```

最终的剩余输出在**行 64 和 65** 上加上另一个`Conv2D`层。

现在，升级过程在**线 68** 开始，这里`scalingFactor`变量开始起作用。随后是`depth_to_space`效用函数，通过相应地均匀减小通道尺寸来增加`featureMaps`的高度和宽度(**行 70** )。添加一个`LeakyReLU`激活功能来完成这个特定的层组合(**行 71** )。

在**行 73-76** 上重复相同的一组层。输出层是通过将`featureMaps`传递给另一个`Conv2D`层来实现的。注意这个卷积层有一个`tanh`激活函数，它将你的输入缩放到`-1`和`1`的范围。

因此，像素被重新调整到 0 到 255 的范围内。(**第 79 行和第 80 行**)。

随着**线 83** 上发电机的初始化，我们的 ESRGAN 发电机侧要求完成。

```py
	@staticmethod
	def discriminator(featureMaps, leakyAlpha, discBlocks):
		# initialize the input layer and process it with conv kernel
		inputs = Input((None, None, 3))
		x = Rescaling(scale=1.0/127.5, offset=-1)(inputs)
		x = Conv2D(featureMaps, 3, padding="same")(x)
		x = LeakyReLU(leakyAlpha)(x)

		# pass the output from previous layer through a CONV => BN =>
		# LeakyReLU block
		x = Conv2D(featureMaps, 3, padding="same")(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(leakyAlpha)(x)
```

正如我们所知，鉴别器的目标是接受图像作为输入，并输出一个单一的值，它表示图像是真的还是假的。

第 89 行**上的`discriminator`函数由以下参数定义:**

*   `featureMaps`:图层`Conv2D`的滤镜数量。
*   `leakyAlpha`:激活功能`LeakyReLU`所需的参数。
*   `discBlocks`:我们在架构中需要的鉴别器块的数量。

鉴频器的输入被初始化，像素被缩放到`-1`和`1`的范围(**行 91 和 92** )。

该架构从一个`Conv2D`层开始，接着是一个`LeakyReLU`激活层(**行 93 和 94** )。

尽管我们已经为生成器放弃了批处理规范化层，但我们将把它们用于鉴别器。下一组图层是一个`Conv` → `BN` → `LeakyReLU`的组合(**第 98-100 行**)。

```py
		# create a downsample conv kernel config
		downConvConf = {
			"strides": 2,
			"padding": "same",
		}

		# create a number of discriminator blocks
		for i in range(1, discBlocks):
			# first CONV => BN => LeakyReLU block
			x = Conv2D(featureMaps * (2 ** i), 3, **downConvConf)(x)
			x = BatchNormalization()(x)
			x = LeakyReLU(leakyAlpha)(x)

			# second CONV => BN => LeakyReLU block
			x = Conv2D(featureMaps * (2 ** i), 3, padding="same")(x)
			x = BatchNormalization()(x)
			x = LeakyReLU(leakyAlpha)(x)
```

在第**行第 103-106** 行，我们创建一个下采样卷积模板配置。然后在**线 109-118** 上的自动鉴别器模块中使用。

```py
		# process the feature maps with global average pooling
		x = GlobalAvgPool2D()(x)
		x = LeakyReLU(leakyAlpha)(x)

		# final FC layer with sigmoid activation function
		x = Dense(1, activation="sigmoid")(x)

		# create the discriminator model
		discriminator = Model(inputs, x)

		# return the discriminator model
		return discriminator
```

特征地图然后通过`GlobalAvgPool2D`层和另一个`LeakyReLU`激活层，之后最终的密集层给出我们的输出(**第 121-125 行**)。

鉴别器对象被初始化并在**行 128-131** 返回，鉴别器功能结束。

* * *

### [**为 ESRGAN** 建立训练管道](#TOC)

架构完成后，是时候转到位于`pyimagesearch`目录中的`esrgan_training.py`脚本了。

```py
# import the necessary packages
from tensorflow.keras import Model
from tensorflow import concat
from tensorflow import zeros
from tensorflow import ones
from tensorflow import GradientTape
from tensorflow.keras.activations import sigmoid
from tensorflow.math import reduce_mean
import tensorflow as tf

class ESRGANTraining(Model):
	def __init__(self, generator, discriminator, vgg, batchSize):
		# initialize the generator, discriminator, vgg model, and 
		# the global batch size
		super().__init__()
		self.generator = generator
		self.discriminator = discriminator
		self.vgg = vgg
		self.batchSize = batchSize
```

为了使事情变得更简单，完整的培训模块被打包在第 11 行上定义的类中。

自然地，第一个函数变成了`__init__`，它接受发生器模型、鉴别器模型、VGG 模型和批量规格(**第 12 行**)。

在这个函数中，我们为参数创建相应的类变量(**第 16-19 行**)。这些变量将在后面用于类函数。

```py
	def compile(self, gOptimizer, dOptimizer, bceLoss, mseLoss):
		super().compile()
		# initialize the optimizers for the generator 
		# and discriminator
		self.gOptimizer = gOptimizer
		self.dOptimizer = dOptimizer

		# initialize the loss functions
		self.bceLoss = bceLoss
		self.mseLoss = mseLoss
```

第 21 行**上的`compile`函数接收生成器和鉴别器优化器、二元交叉熵损失函数和均方损失函数。**

该函数初始化发生器和鉴别器的优化器和损失函数(**第 25-30 行**)。

```py
	def train_step(self, images):
		# grab the low and high resolution images
		(lrImages, hrImages) = images
		lrImages = tf.cast(lrImages, tf.float32)
		hrImages = tf.cast(hrImages, tf.float32)

		# generate super resolution images
		srImages = self.generator(lrImages)

		# combine them with real images
		combinedImages = concat([srImages, hrImages], axis=0)

		# assemble labels discriminating real from fake images where
		# label 0 is for predicted images and 1 is for original high
		# resolution images
		labels = concat(
			[zeros((self.batchSize, 1)), ones((self.batchSize, 1))],
			axis=0)
```

现在是我们定义培训程序的时候了。这在**行 32** 上定义的函数`train_step`中完成。这个函数接受图像作为它的参数。

我们将图像集解包成相应的低分辨率和高分辨率图像，并将它们转换成`float32`数据类型(**第 34-36 行**)。

在**行** **39** 处，我们从生成器中得到一批假的超分辨率图像。这些与真实的超分辨率图像连接，并且相应地创建标签(**行** **47-49** )。

```py
		# train the discriminator with relativistic error
		with GradientTape() as tape:
			# get the raw predictions and divide them into
			# raw fake and raw real predictions
			rawPreds = self.discriminator(combinedImages)
			rawFake = rawPreds[:self.batchSize]
			rawReal = rawPreds[self.batchSize:]

			# process the relative raw error and pass it through the
			# sigmoid activation function
			predFake = sigmoid(rawFake - reduce_mean(rawReal)) 
			predReal = sigmoid(rawReal - reduce_mean(rawFake))

			# concat the predictions and calculate the discriminator
			# loss
			predictions = concat([predFake, predReal], axis=0)
			dLoss = self.bceLoss(labels, predictions)

		# compute the gradients
		grads = tape.gradient(dLoss,
			self.discriminator.trainable_variables)

		# optimize the discriminator weights according to the
		# gradients computed
		self.dOptimizer.apply_gradients(
			zip(grads, self.discriminator.trainable_variables)
		)

		# generate misleading labels
		misleadingLabels = ones((self.batchSize, 1))
```

首先，我们将定义鉴别器训练。开始一个`GradientTape`，我们从我们的鉴别器得到对组合图像集的预测(**第 52-55 行**)。我们从这些预测中分离出假图像预测和真实图像预测，并得到两者的相对论误差。然后这些值通过一个 sigmoid 函数得到我们的最终输出值(**第 56-62 行**)。

预测再次被连接，并且通过使预测通过二元交叉熵损失来计算鉴别器损失(**行 66 和 67** )。

利用损失值，计算梯度，并相应地改变鉴别器权重(**第 70-77 行**)。

鉴别器训练结束后，我们现在生成生成器训练所需的误导标签(**第 80 行**)。

```py
		# train the generator (note that we should *not* update
		# the weights of the discriminator)
		with GradientTape() as tape:
			# generate fake images
			fakeImages = self.generator(lrImages)

			# calculate predictions
			rawPreds = self.discriminator(fakeImages)
			realPreds = self.discriminator(hrImages)
			relativisticPreds = rawPreds - reduce_mean(realPreds)
			predictions = sigmoid(relativisticPreds)

			# compute the discriminator predictions on the fake images
			# todo: try with logits
			#gLoss = self.bceLoss(misleadingLabels, predictions)
			gLoss = self.bceLoss(misleadingLabels, predictions)

			# compute the pixel loss
			pixelLoss = self.mseLoss(hrImages, fakeImages)

			# compute the normalized vgg outputs
			srVGG = tf.keras.applications.vgg19.preprocess_input(
				fakeImages)
			srVGG = self.vgg(srVGG) / 12.75
			hrVGG = tf.keras.applications.vgg19.preprocess_input(
				hrImages)
			hrVGG = self.vgg(hrVGG) / 12.75

			# compute the perceptual loss
			percLoss = self.mseLoss(hrVGG, srVGG)

			# compute the total GAN loss
			gTotalLoss = 5e-3 * gLoss + percLoss + 1e-2 * pixelLoss

		# compute the gradients
		grads = tape.gradient(gTotalLoss,
			self.generator.trainable_variables)

		# optimize the generator weights according to the gradients
		# calculated
		self.gOptimizer.apply_gradients(zip(grads,
			self.generator.trainable_variables)
		)

		# return the generator and discriminator losses
		return {"dLoss": dLoss, "gTotalLoss": gTotalLoss, 
			"gLoss": gLoss, "percLoss": percLoss, "pixelLoss": pixelLoss}
```

我们再次为生成器初始化一个`GradientTape`，并使用生成器生成假的超分辨率图像(**第 84-86 行**)。

计算伪超分辨率图像和真实超分辨率图像的预测，并计算相对论误差(**行 89-92** )。

预测被馈送到二进制交叉熵损失函数，同时使用均方误差损失函数计算像素损失(**行 97-100** )。

接下来，我们计算 VGG 输出和感知损失(**行 103-111** )。有了所有可用的损耗值，我们使用**行 114** 上的等式计算 GAN 总损耗。

计算并应用发电机梯度(**第 117-128 行**)。

* * *

### [**创建效用函数辅助甘训练**](#TOC)

我们使用了一些实用程序脚本来帮助我们的培训。第一个是保存我们在训练中使用的损失的脚本。为此，让我们转到`pyimagesearch`目录中的`losses.py`脚本。

```py
# import necessary packages
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import Reduction
from tensorflow import reduce_mean

class Losses:
	def __init__(self, numReplicas):
		self.numReplicas = numReplicas

	def bce_loss(self, real, pred):
		# compute binary cross entropy loss without reduction
		BCE = BinaryCrossentropy(reduction=Reduction.NONE)
		loss = BCE(real, pred)

		# compute reduced mean over the entire batch
		loss = reduce_mean(loss) * (1\. / self.numReplicas)

		# return reduced bce loss
		return
```

`__init__`函数定义了后续损失函数中使用的批量大小(**第 8 行和第 9 行**)。

损失被打包到第 7 行**的一个类中。我们定义的第一个损失是**线 11** 上的二元交叉熵损失。它接受真实标签和预测标签。**

二进制交叉熵损失对象定义在**行 13** ，损失计算在**行 14** 。然后在整批中调整损耗(**第 17 行**)。

```py
	def mse_loss(self, real, pred):
		# compute mean squared error loss without reduction
		MSE = MeanSquaredError(reduction=Reduction.NONE)
		loss = MSE(real, pred)

		# compute reduced mean over the entire batch
		loss = reduce_mean(loss) * (1\. / self.numReplicas)

		# return reduced mse loss
		return loss
```

下一个损失是在**行 22** 上定义的均方误差函数。一个均方误差损失对象被初始化，随后是整个批次的损失计算(**第 24-28 行**)。

我们的`losses.py`脚本到此结束。我们接下来进入`utils.py`脚本，它将帮助我们更好地评估 GAN 生成的图像。为此，接下来让我们进入`utils.py`剧本。

```py
# import the necessary packages
from . import config
from matplotlib.pyplot import subplots
from matplotlib.pyplot import savefig
from matplotlib.pyplot import title
from matplotlib.pyplot import xticks
from matplotlib.pyplot import yticks
from matplotlib.pyplot import show
from tensorflow.keras.preprocessing.image import array_to_img
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import os

# the following code snippet has been taken from:
# https://keras.io/examples/vision/super_resolution_sub_pixel
def zoom_into_images(image, imageTitle):
	# create a new figure with a default 111 subplot.
	(fig, ax) = subplots()
	im = ax.imshow(array_to_img(image[::-1]), origin="lower")

	title(imageTitle)
	# zoom-factor: 2.0, location: upper-left
	axins = zoomed_inset_axes(ax, 2, loc=2)
	axins.imshow(array_to_img(image[::-1]), origin="lower")

	# specify the limits.
	(x1, x2, y1, y2) = 20, 40, 20, 40
	# apply the x-limits.
	axins.set_xlim(x1, x2)
	# apply the y-limits.
	axins.set_ylim(y1, y2)

	# remove the xticks and yticks
	yticks(visible=False)
	xticks(visible=False)

	# make the line.
	mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")

	# build the image path and save it to disk
	imagePath = os.path.join(config.BASE_IMAGE_PATH,
		f"{imageTitle}.png")
	savefig(imagePath)

	# show the image
	show()
```

这个脚本在第 16 行的**处包含一个名为`zoom_into_images`的函数，它接受图像和图像标题作为参数。**

首先定义支线剧情，绘制图像(**第 18 行和第 19 行**)。在**第 21-24 行**，我们放大图像的左上区域，并再次绘制该部分。

该图的界限设置在第 27-31 行的**上。现在，我们移除 x 轴和 y 轴上的记号，并在原始绘图上插入线条(**第 34-38 行**)。**

绘制完图像后，我们保存图像并结束该功能(**第 41-43 行**)。

我们最后的实用程序脚本是`vgg.py`脚本，它初始化了我们感知损失的 VGG 模型。

```py
# import the necessary packages
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model

class VGG:
	@staticmethod
	def build():
		# initialize the pre-trained VGG19 model
		vgg = VGG19(input_shape=(None, None, 3), weights="imagenet",
			include_top=False)

		# slicing the VGG19 model till layer #20
		model = Model(vgg.input, vgg.layers[20].output)

		# return the sliced VGG19 model
		return model
```

我们在**5 号线**为 VGG 模型创建一个类。该函数包含一个名为`build`的单一函数，它简单地初始化一个预训练的 VGG-19 架构，并返回一个切片到第 20 层的 VGG 模型(**第 7-16 行**)。这就结束了`vgg.py`脚本。

* * *

### [**训练 ESR gan**](#TOC)

现在我们所有的积木都准备好了。我们只需要按照正确的顺序来执行它们，以便进行适当的 GAN 训练。为了实现这一点，我们进入`train_esrgan.py`脚本。

```py
# USAGE
# python train_esrgan.py --device gpu
# python train_esrgan.py --device tpu

# import tensorflow and fix the random seed for better reproducibility
import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from pyimagesearch.data_preprocess import load_dataset
from pyimagesearch.esrgan_training import ESRGANTraining
from pyimagesearch.esrgan import ESRGAN
from pyimagesearch.losses import Losses
from pyimagesearch.vgg import VGG
from pyimagesearch import config
from tensorflow import distribute
from tensorflow.config import experimental_connect_to_cluster
from tensorflow.tpu.experimental import initialize_tpu_system
from tensorflow.keras.optimizers import Adam
from tensorflow.io.gfile import glob
import argparse
import sys
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True, default="gpu",
	choices=["gpu", "tpu"], type=str,
	help="device to use for training (gpu or tpu)")
args = vars(ap.parse_args())
```

这里的第一个任务是定义一个参数解析器，以便用户可以选择是使用 TPU 还是 GPU 来完成 GAN 训练(**第 26-30 行**)。正如我们已经提到的，我们已经使用 TPU 和 GPU 训练了 GAN 来评估效率。

```py
# check if we are using TPU, if so, initialize the TPU strategy
if args["device"] == "tpu":
	# initialize the TPUs
	tpu = distribute.cluster_resolver.TPUClusterResolver() 
	experimental_connect_to_cluster(tpu)
	initialize_tpu_system(tpu)
	strategy = distribute.TPUStrategy(tpu)

	# ensure the user has entered a valid gcs bucket path
	if config.TPU_BASE_TFR_PATH == "gs://<PATH_TO_GCS_BUCKET>/tfrecord":
		print("[INFO] not a valid GCS Bucket path...")
		sys.exit(0)

	# set the train TFRecords, pretrained generator, and final
	# generator model paths to be used for TPU training
	tfrTrainPath = config.TPU_DIV2K_TFR_TRAIN_PATH
	pretrainedGenPath = config.TPU_PRETRAINED_GENERATOR_MODEL
	genPath = config.TPU_GENERATOR_MODEL

# otherwise, we are using multi/single GPU so initialize the mirrored
# strategy
elif args["device"] == "gpu":
	# define the multi-gpu strategy
	strategy = distribute.MirroredStrategy()

	# set the train TFRecords, pretrained generator, and final
	# generator model paths to be used for GPU training
	tfrTrainPath = config.GPU_DIV2K_TFR_TRAIN_PATH
	pretrainedGenPath = config.GPU_PRETRAINED_GENERATOR_MODEL
	genPath = config.GPU_GENERATOR_MODEL

# else, invalid argument was provided as input
else:
	# exit the program
	print("[INFO] please enter a valid device argument...")
	sys.exit(0)

# display the number of accelerators
print(f"[INFO] number of accelerators: {strategy.num_replicas_in_sync}...")
```

根据设备选择，我们必须初始化策略。首先，我们探索 TPU 选择的案例(**第 33 行**)。

为了恰当地利用 TPU 的能力，我们初始化了一个`TPUClusterResolver`来有效地利用资源。接下来，TPU 策略被初始化(**第 35-43 行**)。

定义了到 TPU 训练数据、预训练发生器和完全训练发生器的`TFRecords`路径(**第 47-49 行**)。

现在探讨第二种设备选择，即 GPU。对于 GPU，使用 GPU 镜像策略(**第 55 行**)，并定义 GPU 特定的`TFRecords`路径、预训练的生成器路径和完全训练的生成器路径(**第 59-61 行**)。

如果给出了任何其他选择，脚本会自己退出(**第 64-67 行**)。

```py
# grab train TFRecord filenames
print("[INFO] grabbing the train TFRecords...")
trainTfr = glob(tfrTrainPath +"/*.tfrec")

# build the div2k datasets from the TFRecords
print("[INFO] creating train and test dataset...")
trainDs = load_dataset(filenames=trainTfr, train=True,
	batchSize=config.TRAIN_BATCH_SIZE * strategy.num_replicas_in_sync)

# call the strategy scope context manager
with strategy.scope():
	# initialize our losses class object
	losses = Losses(numReplicas=strategy.num_replicas_in_sync)

	# initialize the generator, and compile it with Adam optimizer and
	# MSE loss
	generator = ESRGAN.generator(
		scalingFactor=config.SCALING_FACTOR,
		featureMaps=config.FEATURE_MAPS,
		residualBlocks=config.RESIDUAL_BLOCKS,
		leakyAlpha=config.LEAKY_ALPHA,
		residualScalar=config.RESIDUAL_SCALAR)
	generator.compile(optimizer=Adam(learning_rate=config.PRETRAIN_LR),
		loss=losses.mse_loss)

	# pretraining the generator
	print("[INFO] pretraining ESRGAN generator ...")
	generator.fit(trainDs, epochs=config.PRETRAIN_EPOCHS,
		steps_per_epoch=config.STEPS_PER_EPOCH)
```

我们获取`TFRecords`文件，然后使用`load_dataset`函数(**第 74-79 行**)创建一个训练数据集。

首先，我们将初始化预训练的生成器。为此，我们首先调用策略范围上下文管理器来初始化第 82-95 行上的损失和生成器。接着在**线 99 和 100** 上训练发电机。

```py
# check whether output model directory exists, if it doesn't, then
# create it
if args["device"] == "gpu" and not os.path.exists(config.BASE_OUTPUT_PATH):
	os.makedirs(config.BASE_OUTPUT_PATH)

# save the pretrained generator
print("[INFO] saving the pretrained generator...")
generator.save(pretrainedGenPath)

# call the strategy scope context manager
with strategy.scope():
	# initialize our losses class object
	losses = Losses(numReplicas=strategy.num_replicas_in_sync)

	# initialize the vgg network (for perceptual loss) and discriminator
	# network
	vgg = VGG.build()
	discriminator = ESRGAN.discriminator(
		featureMaps=config.FEATURE_MAPS,
		leakyAlpha=config.LEAKY_ALPHA,
		discBlocks=config.DISC_BLOCKS)

	# build the ESRGAN model and compile it
	esrgan = ESRGANTraining(
		generator=generator,
		discriminator=discriminator,
		vgg=vgg,
		batchSize=config.TRAIN_BATCH_SIZE)
	esrgan.compile(
		dOptimizer=Adam(learning_rate=config.FINETUNE_LR),
		gOptimizer=Adam(learning_rate=config.FINETUNE_LR),
		bceLoss=losses.bce_loss,
		mseLoss=losses.mse_loss,
	)

	# train the ESRGAN model
	print("[INFO] training ESRGAN...")
	esrgan.fit(trainDs, epochs=config.FINETUNE_EPOCHS,
		steps_per_epoch=config.STEPS_PER_EPOCH)

# save the ESRGAN generator
print("[INFO] saving ESRGAN generator to {}..."
	.format(genPath))
esrgan.generator.save(genPath)
```

如果设备被设置为 GPU，基本输出模型目录被初始化(如果还没有完成的话)(**行 104 和 105** )。预训练的发电机然后被保存到指定的路径(**行 109** )。

现在我们来看看训练有素的 ESRGAN。我们再次初始化策略范围上下文管理器，并初始化一个 loss 对象(**行 112-114** )。

感知损失所需的 VGG 模型被初始化，随后是 ESRGAN ( **行 118-129** )。然后用所需的优化器和损耗编译 es rgan(**第 130-135 行**)。

这里的最后一步是用训练数据拟合 ESRGAN，并让它训练(**行 139 和 140** )。

一旦完成，被训练的重量被保存在**线 145** 上的预定路径中。

* * *

### [**为 ESRGAN** 构建推理脚本](#TOC)

随着我们的 ESRGAN 培训的完成，我们现在可以评估我们的 ESRGAN 在结果方面表现如何。为此，让我们看看位于核心目录中的`inference.py`脚本。

```py
# USAGE
# python inference.py --device gpu
# python inference.py --device tpu

# import the necessary packages
from pyimagesearch.data_preprocess import load_dataset
from pyimagesearch.utils import zoom_into_images
from pyimagesearch import config
from tensorflow import distribute
from tensorflow.config import experimental_connect_to_cluster
from tensorflow.tpu.experimental import initialize_tpu_system
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.io.gfile import glob
from matplotlib.pyplot import subplots
import argparse
import sys
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--device", required=True, default="gpu",
	choices=["gpu", "tpu"], type=str,
	help="device to use for training (gpu or tpu)")
args = vars(ap.parse_args())
```

根据我们用于训练的设备，我们需要提供相同的设备来初始化模型并相应地加载所需的权重。为此，我们构建了一个参数解析器，它接受用户的设备选择(**第 21-25 行**)。

```py
# check if we are using TPU, if so, initialize the strategy
# accordingly
if args["device"] == "tpu":
	tpu = distribute.cluster_resolver.TPUClusterResolver()
	experimental_connect_to_cluster(tpu)
	initialize_tpu_system(tpu)
	strategy = distribute.TPUStrategy(tpu)

	# set the train TFRecords, pretrained generator, and final
	# generator model paths to be used for TPU training
	tfrTestPath = config.TPU_DIV2K_TFR_TEST_PATH
	pretrainedGenPath = config.TPU_PRETRAINED_GENERATOR_MODEL
	genPath = config.TPU_GENERATOR_MODEL

# otherwise, we are using multi/single GPU so initialize the mirrored
# strategy
elif args["device"] == "gpu":
	# define the multi-gpu strategy
	strategy = distribute.MirroredStrategy()

	# set the train TFRecords, pretrained generator, and final
	# generator model paths to be used for GPU training
	tfrTestPath = config.GPU_DIV2K_TFR_TEST_PATH
	pretrainedGenPath = config.GPU_PRETRAINED_GENERATOR_MODEL
	genPath = config.GPU_GENERATOR_MODEL

# else, invalid argument was provided as input
else:
	# exit the program
	print("[INFO] please enter a valid device argument...")
	sys.exit(0)
```

根据用户输入的选择，我们必须建立处理数据的策略。

第一个设备选择(TPU)是通过初始化`TPUClusterResolver`、策略和特定于 TPU 的输出路径来探索的，其方式与我们为训练脚本所做的相同(**第 29-33 行**)。

对于第二个选择(GPU)，我们重复与训练脚本相同的过程(**第 43-51 行**)。

如果给出了任何其他输入，脚本会自行退出(**第 54-57 行**)。

```py
# get the dataset
print("[INFO] loading the test dataset...")
testTfr = glob(tfrTestPath + "/*.tfrec")
testDs = load_dataset(testTfr, config.INFER_BATCH_SIZE, train=False)

# get the first batch of testing images
(lrImage, hrImage) = next(iter(testDs))

# call the strategy scope context manager
with strategy.scope():
    # load the ESRGAN trained models
    print("[INFO] loading the pre-trained and fully trained ESRGAN model...")
    esrganPreGen = load_model(pretrainedGenPath, compile=False)
    esrganGen = load_model(genPath, compile=False)

    # predict using ESRGAN
    print("[INFO] making predictions with pre-trained and fully trained ESRGAN model...")
    esrganPreGenPred = esrganPreGen.predict(lrImage)
    esrganGenPred = esrganGen.predict(lrImage)
```

出于测试目的，我们在**行 62** 上创建一个测试数据集。使用`next(iter())`，我们可以抓取一批图像集，我们在**行 65** 上将其解包。

接下来，预训练的 GAN 和完全训练的 ESRGAN 被初始化并加载到**线 71 和 72** 上。然后低分辨率图像通过这些 gan 进行预测(**线 76 和 77** )。

```py
# plot the respective predictions
print("[INFO] plotting the ESRGAN predictions...")
(fig, axes) = subplots(nrows=config.INFER_BATCH_SIZE, ncols=4,
	figsize=(50, 50))

# plot the predicted images from low res to high res
for (ax, lowRes, esrPreIm, esrGanIm, highRes) in zip(axes, lrImage,
		esrganPreGenPred, esrganGenPred, hrImage):
	# plot the low resolution image
	ax[0].imshow(array_to_img(lowRes))
	ax[0].set_title("Low Resolution Image")

	# plot the pretrained ESRGAN image
	ax[1].imshow(array_to_img(esrPreIm))
	ax[1].set_title("ESRGAN Pretrained")

	# plot the ESRGAN image
	ax[2].imshow(array_to_img(esrGanIm))
	ax[2].set_title("ESRGAN")

	# plot the high resolution image
	ax[3].imshow(array_to_img(highRes))
	ax[3].set_title("High Resolution Image")

# check whether output image directory exists, if it doesn't, then
# create it
if not os.path.exists(config.BASE_IMAGE_PATH):
	os.makedirs(config.BASE_IMAGE_PATH)

# serialize the results to disk
print("[INFO] saving the ESRGAN predictions to disk...")
fig.savefig(config.GRID_IMAGE_PATH)

# plot the zoomed in images
zoom_into_images(esrganPreGenPred[0], "ESRGAN Pretrained")
zoom_into_images(esrganGenPred[0], "ESRGAN")
```

为了可视化我们的结果，在第 81 行和第 82 行初始化子情节。然后，我们对批处理进行循环，并绘制低分辨率图像、预训练 GAN 输出、ESRGAN 输出和实际高分辨率图像进行比较(**第 85-101 行**)。

* * *

### [](#TOC)**的可视化效果**

 ****图 3 和图 4** 分别向我们展示了预训练的 ESRGAN 和完全训练的 ESRGAN 的最终预测图像。

这两个模型的输出在视觉上是无法区分的。然而，结果比上周的 SRGAN 输出要好，即使 ESRGAN 被训练的时期更少。

放大的补丁显示了 ESRGAN 实现的像素化信息的复杂清晰度，证明用于 SRGAN 增强的增强配方工作得相当好。

* * *

* * *

## [**汇总**](#TOC)

根据他们的结果，SRGANs 已经给人留下了深刻的印象。它优于现有的几种超分辨率算法。在其基础上，提出增强配方是整个深度学习社区非常赞赏的事情。

这些添加是经过深思熟虑的，结果一目了然。我们的 ESRGAN 取得了很大的成绩，尽管被训练了很少的时代。这非常符合 ESRGAN 优先关注效率的动机。

GANs 一直给我们留下深刻的印象，直到今天，新的域名都在使用 GANs。但是在今天的项目中，我们讨论了可以用来改善最终结果的方法。

* * *

### [**引用信息**](#TOC)

**Chakraborty，D.** “增强的超分辨率生成对抗网络(ESRGAN)， *PyImageSearch* ，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva，R. Raha，A. Thanki 编辑。，2022 年，【https://pyimg.co/jt2cb 

```py
@incollection{Chakraborty_2022_ESRGAN,
  author = {Devjyoti Chakraborty},
  title = {Enhanced Super-Resolution Generative Adversarial Networks {(ESRGAN)}},
  booktitle = {PyImageSearch},
  editor = {Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/jt2cb},
}
```

* * *

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***********