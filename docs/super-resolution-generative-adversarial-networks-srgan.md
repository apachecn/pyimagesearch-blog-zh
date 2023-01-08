# 超分辨率生成对抗网络

> 原文：<https://pyimagesearch.com/2022/06/06/super-resolution-generative-adversarial-networks-srgan/>

* * *

## **目录**

* * *

## [](#TOC)

 **超分辨率(SR)是以最小的信息失真将低分辨率图像上采样到更高的分辨率。自从研究人员获得了足够强大的机器来计算大量数据，超分辨率领域已经取得了重大进展，双三次调整大小，高效的亚像素网络等。

要理解超分辨率的重要性，环顾一下当今的技术世界就会自动解释它。从保存旧媒体材料(电影和连续剧)到增强显微镜的视野，超分辨率的影响是广泛而又极其明显的。稳健的超分辨率算法在当今世界极其重要。

由于生成对抗网络(GANs)的引入在深度学习领域掀起了风暴，所以结合 GAN 的超分辨率技术的引入只是时间问题。

今天我们将学习 SRGAN，这是一种巧妙的超分辨率技术，结合了 GANs 的概念和传统的 SR 方法。

在本教程中，您将学习如何实现 SRGAN。

本课是 4 部分系列 **GANs 201** 的第 1 部分。

1.  ***【超分辨率生成对抗网络(SRGAN)* (本教程)**
2.  *增强型超分辨率生成对抗网络(ESRGAN)*
3.  *用于图像到图像转换的像素 2 像素氮化镓*
4.  *用于图像到图像转换的 CycleGAN】*

**要学习如何实现 SRGANs，*继续阅读。***

* * *

## [](#TOC)

 **虽然 GANs 本身是一个革命性的概念，但其应用领域仍然是一个相当新的领域。在超分辨率中引入 GANs 并不像听起来那么简单。简单地在一个类似超分辨率的架构中添加 GANs 背后的数学将无法实现我们的目标。

SRGAN 的想法是通过结合有效的子像素网络的元素以及传统的 GAN 损失函数来构思的。在我们深入探讨这个问题之前，让我们先简要回顾一下生成敌对网络。

* * *

### [**甘斯**简介](#TOC)

用最常见的侦探和伪造者的例子(**图 1** )来最好地描述甘的想法。

伪造者试图制造出逼真的艺术品，而侦探则负责辨别真假艺术品。造假者是生产者，侦探是鉴别者。

理想的训练将生成欺骗鉴别器的数据，使其相信该数据属于训练数据集。

***注:*** *关于 GANs 最重要的直觉是，生成器产生的数据不一定是训练数据的复制品，但它必须* ***看起来*** *像它属于训练数据分布。*

* * *

### [**使用 GANs 的超分辨率**](#TOC)

SRGANs 保留了 GANs 的核心概念(即 min-max 函数)，使生成器和鉴别器通过相互对抗来一致学习。SRGAN 引入了一些自己独有的附加功能，这些功能是基于之前在该领域所做的研究。让我们先来看看 SRGAN 的完整架构(**图 2** )。

一些**要点**要注意:

*   生成器网络采用残差块，其思想是保持来自先前层的信息有效，并允许网络自适应地从更多特征中进行选择。
*   我们传递低分辨率图像，而不是添加随机噪声作为发生器输入。

**鉴别器网络**非常标准，其工作原理与普通 GAN 中的鉴别器一样。

SRGANs 中的**突出因素**是感知损失函数。虽然生成器和鉴别器将基于 GAN 架构进行训练，但 SRGANs 使用另一个损失函数的帮助来达到其目的地:感知/内容损失函数。

这个想法是，SRGAN 设计了一个损失函数，通过计算出感知相关的特征来达到它的目标。因此，不仅对抗性的损失有助于调整权重，内容损失也在发挥作用。

**内容损失**被定义为 VGG 损失，这意味着然后按像素比较预训练的 VGG 网络输出。真实图像 VGG 输出和伪图像 VGG 输出相似的唯一方式是当输入图像本身相似时。这背后的直觉是，逐像素比较将有助于实现超分辨率的核心目标。

当 GAN 损失和含量损失相结合时，结果确实是正的。我们生成的超分辨率图像非常清晰，能够反映高分辨率(hr)图像。

在我们的项目中，为了展示 SRGAN 的威力，我们将它与预训练的发生器和原始高分辨率图像进行比较。为了使我们的训练更有效，我们将把我们的数据转换成 [`TFRecords`](https://www.tensorflow.org/tutorials/load_data/tfrecord) 。

* * *

### [**配置您的开发环境**](#TOC)

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
$ pip install tensorflow
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
│   ├── srgan.py
│   ├── srgan_training.py
│   ├── utils.py
│   └── vgg.py
└── train_srgan.py

2 directories, 11 files
```

在`pyimagesearch`目录中，我们有:

*   ``config.py`` :包含完整项目的端到端配置管道，
*   ``data_preprocess.py`` :包含辅助数据处理的功能。
*   ``__init__.py`` :使目录的行为像一个 python 包。
*   ``losses.py`` :初始化训练 SRGAN 所需的损耗。
*   ``srgan.py`` :包含 SRGAN 架构。
*   ``srgan_training.py`` :包含运行 SRGAN 训练的训练类。
*   ``utils.py`` :包含附加实用程序
*   ``vgg.py`` :初始化用于感知损失计算的 VGG 模型。

在根目录中，我们有:

*   ``create_tfrecords.py`` :从我们将要使用的数据集创建``TFRecords`` 。
*   ``inference.py`` :使用训练好的模型进行推理。
*   ``train_srgan.py`` :使用``srgan.py`` 和``srgan_training.py`` 脚本执行 SRGAN 训练。

* * *

### [**创建配置管道**](#TOC)

在实施 SRGAN 时，有许多因素在起作用。为此，我们创建了一个在整个项目中使用的全局配置文件。让我们转到位于`pyimagesearch`目录中的`config.py`文件。

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

# GAN model specs
FEATURE_MAPS = 64
RESIDUAL_BLOCKS = 16
LEAKY_ALPHA = 0.2
DISC_BLOCKS = 4

# training specs
PRETRAIN_LR = 1e-4
FINETUNE_LR = 1e-5
PRETRAIN_EPOCHS = 2500
FINETUNE_EPOCHS = 2500
STEPS_PER_EPOCH = 10
```

在**第 5** 行，引用了`TFDS`数据集。我们将在我们的项目中使用 [`div2k`](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 数据集。该数据集的目的是帮助进行图像超分辨率研究，因为它包含各种高分辨率图像。

在第 8 行的**上，我们已经定义了创建`TFRecords`所需的分片大小。随后是第 9 和第 10** 行**的训练和推理批量。**

在**行 13** 上，定义高分辨率图像形状。这是我们的输出形状，图像将被放大。定义的下一个变量是低分辨率形状，它将作为我们的输入(**第 14 行**)。

相应地，比例因子被定义为`4` (96/24) ( **第 15 行**)。

在**第 18-21 行**中，定义了 GAN 型号规格。这些是:

*   `FEATURE_MAPS`:定义 CNN 的过滤器尺寸
*   `RESIDUAL_BLOCKS`:如前所述，生成器利用残差块，这定义了残差块的数量
*   `LEAKY_ALPHA`:为我们的激活函数值定义`alpha`参数
*   `DISC_BLOCKS`:定义鉴别器的块

然后我们定义训练参数(**第 24-28 行**)，包括:

*   `PRETRAIN_LR`:学习率定义预训练。
*   `FINETUNE_LR`:用于微调的学习率
*   `PRETRAIN_EPOCHS`:定义为预训练的时期数
*   `FINETUNE_EPOCHS`:用于微调的时期数
*   `STEPS_PER_EPOCH`:定义每个时期运行的步数

```py
# define the path to the dataset
BASE_DATA_PATH = "dataset"
DIV2K_PATH = os.path.join(BASE_DATA_PATH, "div2k")

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

# GPU training SRGAN model paths
GPU_PRETRAINED_GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH,
    "models", "pretrained_generator")
GPU_GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH, "models",
    "generator")

# TPU training SRGAN model paths
TPU_OUTPUT_PATH = "gs://<PATH_TO_GCS_BUCKET>/outputs"
TPU_PRETRAINED_GENERATOR_MODEL = os.path.join(TPU_OUTPUT_PATH,
    "models", "pretrained_generator")
TPU_GENERATOR_MODEL = os.path.join(TPU_OUTPUT_PATH, "models",
    "generator")

# define the path to the inferred images and to the grid image
BASE_IMAGE_PATH = os.path.join(BASE_OUTPUT_PATH, "images")
GRID_IMAGE_PATH = os.path.join(BASE_IMAGE_PATH, "grid.png")
```

在**第 31 行和第 32 行**，我们已经定义了存储数据集的引用路径。

由于我们将在 GPU 和 TPU 上训练我们的数据，我们已经为每个训练选择分别创建了`tfrecords`。如你所知，这个项目是数据密集型的。因此，需要将我们的数据转换到`tfrecords`以进行优化和更快的训练。

首先，我们为 GPU 训练数据定义了到`tfrecords`的路径(**第 35-37 行**)。接下来是参考 TPU 训练数据的`tfrecords`的定义(**第 40-42 行**)。

训练和推断数据路径完成后，我们在第 45 行的**上定义了全局输出目录。**

我们将比较预训练的主干和我们完全训练的发电机模型。我们在**行 48-51** 定义了 GPU 预训练生成器和普通生成器。

如前所述，由于我们也将在 TPU 上训练，我们在**线 54-58** 上定义单独的 TPU 训练输出和发生器路径。

最后，我们在**的第 61 行和第 62 行**添加推断图像子目录以及网格图像子目录，以结束我们的`config.py`脚本。

* * *

### [**搭建数据处理管道**](#TOC)

由于数据无疑是我们项目中最重要的一块拼图，我们必须建立一个广泛的数据处理管道来处理我们所有的需求，这涉及到几种数据扩充方法。为此，让我们移到`pyimagesearch`目录中的`data_preprocess.py`。

我们创建了一系列函数来帮助我们扩充数据集。

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

考虑到我们将在这个项目中使用的 TensorFlow 包装器的数量，为空间优化定义一个`tf.data.AUTOTUNE`对象是一个好方法。

我们定义的第一个函数是`random_crop` ( **第 12 行**)。它接受以下参数:

*   `lrImage`:低分辨率图像。
*   `hrImage`:高分辨率图像。
*   `hrCropSize`:用于低分辨率裁剪计算的高分辨率裁剪尺寸。
*   `scale`:我们用来计算低分辨率裁剪的因子。

使用`tf.random.uniform`，我们计算**行 18-21** 上的低分辨率(lr)宽度和高度偏移。

为了计算相应的高分辨率值，我们简单地将低分辨率值乘以比例因子(**行 24 和 25** )。

使用这些值，我们裁剪出低分辨率图像及其对应的高分辨率图像并返回它们(**第 28-34 行**)。

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

我们关注的下一个函数是`get_center_crop` ( **第 36 行**)，它接受以下参数:

*   `lrImage`:低分辨率图像
*   `hrImage`:高分辨率图像
*   `hrCropSize`:用于低分辨率裁剪计算的高分辨率裁剪尺寸
*   `scale`:我们用来计算低分辨率裁剪的因子

就像我们为之前的函数创建裁剪大小值一样，我们在第 38 行和第 39 行得到 lr 裁剪大小值和图像形状。

在**行 42 和 43** 上，我们将低分辨率形状除以 2，得到中心点。

为了获得相应的高分辨率中心点，将 lr 中心点乘以比例因子(**第 46 和 47 行**)。

在第**行第 50-53** 行，我们得到低分辨率和高分辨率图像的中心裁剪并返回它们。

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

在第 58 行的**上，我们有`random_flip`功能来翻转图像。它接受低分辨率和高分辨率图像作为其参数。**

基于使用`tf.random.uniform`的翻转概率值，我们翻转我们的图像并返回它们。

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

**行 68** 上的`random_rotate`功能将根据`tf.random.uniform` ( **行 70-77** )产生的值随机旋转一对高分辨率和低分辨率图像。

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
```

`read_train_example`函数接受一个示例图像集(一个 lr 和一个 hr 图像集)作为参数(**第 79 行**)。在**第 82-85 行**，我们创建一个特征模板。

我们从示例集中解析低分辨率和高分辨率图像(**第 86-90 行**)。

```py
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

使用我们之前创建的函数，我们将数据扩充应用于我们的示例图像集(**行 93-95** )。

一旦我们的图像被增强，我们就将图像重新整形为我们需要的输入和输出大小(**行 98 和 99** )。

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

我们创建一个类似的函数`read_test_example`，来读取一个推理图像集。从先前创建的`read_train_example`开始重复所有步骤。例外的是，由于这是为了我们的推断，我们不对数据进行任何扩充(**行 **104-125**** )。

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

我们的数据处理管道脚本中的最后一个函数是`load_dataset` ( **第 127 行**)，它接受以下参数:

*   `filenames`:正在考虑的文件的名称
*   `batchSize`:定义一次要考虑的批量
*   `train`:告诉我们模式是否设置为训练的布尔变量

在**的第 129** 行，我们使用`tf.data`从文件名中获取`TFRecords`。

如果模式设置为`train`，我们将`read_train_example`函数映射到我们的数据集(**第 133-136 行**)。这意味着数据集中的所有记录都通过该函数传递。

如果模式设置为其他，我们将`read_test_example`函数映射到我们的数据集(**第 138-141 行**)。

我们的最后一步是批处理和预取数据集(**行 144-149** )。

* * *

### [**实现 SRGAN 损耗功能**](#TOC)

尽管我们的实际损失计算将在以后进行，但是一个简单的实用脚本来存储损失函数将有助于更好地组织我们的项目。因为我们不需要写损失的数学方程(TensorFlow 会替我们写)，我们只需要调用所需的包。

为此，让我们转到位于`pyimagesearch`目录中的`losses.py`脚本。

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
        return loss
```

创建一个专门用于损失的完整类(**行 7** )。我们首先在**第 11 行**定义二元交叉熵损失函数，它接受真实值和预测值。

创建一个二元交叉熵对象，并计算损失(**第 13 行和第 14 行**)。然后计算整批的损失(**第 17 行**)。

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

接下来，定义**线 22** 上的均方误差损失函数。创建一个 MSE 对象，并计算损失(**第 24 和 25 行**)。

正如在前面的函数中所做的，我们计算整批的损失并返回它(**第 28-31 行**)。

* * *

### [**实现 SRGAN**](#TOC)

为了开始实现 SRGAN 架构，让我们转到位于`pyimagesearch`目录中的`srgan.py`。

```py
# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import Add
from tensorflow.nn import depth_to_space
from tensorflow.keras import Model
from tensorflow.keras import Input

class SRGAN(object):
    @staticmethod
    def generator(scalingFactor, featureMaps, residualBlocks):
        # initialize the input layer
        inputs = Input((None, None, 3))
        xIn = Rescaling(scale=(1.0 / 255.0), offset=0.0)(inputs)
```

为了包含我们的生成器和鉴别器，我们创建了一个名为`SRGAN` ( **第 14 行**)的类。

首先，我们在第 16 行上定义我们的生成器函数，它接受以下参数:

*   需要得到我们最终的升级输出。
*   `featureMaps`:决定我们想要的卷积滤波器的数量。
*   `residualBlocks`:决定我们想要的剩余连接块数。

在**的第 18 行和第 19 行**，我们定义了生成器的输入，并将像素重新缩放到`0`和`1`的范围。

```py
        # pass the input through CONV => PReLU block
        xIn = Conv2D(featureMaps, 9, padding="same")(xIn)
        xIn = PReLU(shared_axes=[1, 2])(xIn)

        # construct the "residual in residual" block
        x = Conv2D(featureMaps, 3, padding="same")(xIn)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(featureMaps, 3, padding="same")(x)
        x = BatchNormalization()(x)
        xSkip = Add()([xIn, x])
```

我们首先将输入通过`Conv2D`层和一个参数化 ReLU 层(**行 22 和 23** )。

接下来，我们建立一个基础剩余块网络。一个`Conv2D`层，一个批处理规范化层，接着是一个参数化 ReLU 层，`Conv2D`层，和另一个批处理规范化层(**第 26-30 行**)。

这里的最后一步是将我们的输入`xIn`与剩余块输出`x`相加，以完成剩余块网络(**行 31** )。

```py
        # create a number of residual blocks
        for _ in range(residualBlocks - 1):
            x = Conv2D(featureMaps, 3, padding="same")(xSkip)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(featureMaps, 3, padding="same")(x)
            x = BatchNormalization()(x)
            xSkip = Add()([xSkip, x])

        # get the last residual block without activation
        x = Conv2D(featureMaps, 3, padding="same")(xSkip)
        x = BatchNormalization()(x)
        x = Add()([xIn, x])
```

我们使用 for 循环来自动化剩余块条目，并且基本上重复基本剩余块的过程(**行 34-40** )。

一旦在循环之外，我们在添加跳过连接(**第 43-45 行**)之前添加最后的`Conv2D`和批处理规范化层。

```py
        # upscale the image with pixel shuffle
        x = Conv2D(featureMaps * (scalingFactor // 2), 3, padding="same")(x)
        x = depth_to_space(x, 2)
        x = PReLU(shared_axes=[1, 2])(x)

        # upscale the image with pixel shuffle
        x = Conv2D(featureMaps * scalingFactor, 3,
            padding="same")(x)
        x = depth_to_space(x, 2)
        x = PReLU(shared_axes=[1, 2])(x)

        # get the output and scale it from [-1, 1] to [0, 255] range
        x = Conv2D(3, 9, padding="same", activation="tanh")(x)
        x = Rescaling(scale=127.5, offset=127.5)(x)

        # create the generator model
        generator = Model(inputs, x)

        # return the generator
        return generator
```

我们的输入通过一个`Conv2D`层，在那里比例因子值开始起作用(**行 48** )。

`depth_to_space`函数是 TensorFlow 提供的一个漂亮的实用函数，它重新排列我们输入的像素，通过减少通道值来扩展它们的高度和宽度( **Line 49** )。

之后是参数`ReLU`函数，之后是重复的`Conv2D`、`depth_to_space`和另一个参数`ReLU`函数(**第 50-56 行**)。

注意**线 59** 上的`Conv2D`功能将`tanh`作为其激活功能。这意味着特征地图的值现在被缩放到`-1`和`1`的范围。我们使用`Rescaling` ( **第 60 行**)重新调整数值，并使像素回到 0 到 255 的范围内。

这就结束了我们的生成器，所以我们简单地初始化生成器并返回它(**第 63-66 行**)。

```py
    @staticmethod
    def discriminator(featureMaps, leakyAlpha, discBlocks):
        # initialize the input layer and process it with conv kernel
        inputs = Input((None, None, 3))
        x = Rescaling(scale=(1.0 / 127.5), offset=-1.0)(inputs)
        x = Conv2D(featureMaps, 3, padding="same")(x)

        # unlike the generator we use leaky relu in the discriminator
        x = LeakyReLU(leakyAlpha)(x)

        # pass the output from previous layer through a CONV => BN =>
        # LeakyReLU block
        x = Conv2D(featureMaps, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(leakyAlpha)(x)
```

我们继续讨论鉴别器，为此我们在第 69 行的**上定义了一个函数。它接受以下参数:**

*   `featureMaps`:决定一个`Conv2D`层内的滤镜数量。
*   `leakyAlpha`:提供给泄漏`ReLU`激活函数的值，
*   `discBlocks`:要添加到架构内部的鉴别器块的数量。

首先定义鉴频器的输入。然后，将像素重新缩放到范围`-1`到`1` ( **行 71 和 72** )。接下来是在第 73-76 条线**上的`Conv2D`层和`LeakyReLU`激活层。**

接下来，我们创建一个 3 层的组合:第`Conv2D`层，接着是批量标准化，最后是一个`LeakyReLU`函数(**第 80-82 行**)。你会发现这种组合重复了很多次。

```py
        # create a number of discriminator blocks
        for i in range(1, discBlocks):
            # first CONV => BN => LeakyReLU block
            x = Conv2D(featureMaps * (2 ** i), 3, strides=2,
                padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leakyAlpha)(x)

            # second CONV => BN => LeakyReLU block
            x = Conv2D(featureMaps * (2 ** i), 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leakyAlpha)(x)
```

基于之前的`discBlocks`值，我们开始一个循环，并不断添加鉴别器块。每个鉴别器块包含重复两次的`Conv2D` → `BatchNormalization` → `LeakyReLU`组合(**第 85-95 行**)。

```py
        # process the feature maps with global average pooling
        x = GlobalAvgPool2D()(x)
        x = LeakyReLU(leakyAlpha)(x)

        # final FC layer with sigmoid activation function
        x = Dense(1, activation="sigmoid")(x)

        # create the discriminator model
        discriminator = Model(inputs, x)

        # return the discriminator
        return discriminator
```

在循环之外，我们添加一个全局平均池层，然后是另一个`LeakyReLU`激活函数(**第 98 行和第 99 行**)。

由于鉴别器给我们关于输入真实性的信息，我们网络的最后一层是具有 sigmoid 激活函数的`Dense`层(**行 102** )。

* * *

### [**执行 SRGAN 训练脚本**](#TOC)

正如本博客开头所解释的，SRGAN 训练需要同时进行两次损失；VGG 含量的损失以及 GAN 的损失。让我们转到`pyimagesearch`目录中的`srgan_training`脚本。

```py
# import the necessary packages
from tensorflow.keras import Model
from tensorflow import GradientTape
from tensorflow import concat
from tensorflow import zeros
from tensorflow import ones
import tensorflow as tf

class SRGANTraining(Model):
    def __init__(self, generator, discriminator, vgg, batchSize):
        super().__init__()
        # initialize the generator, discriminator, vgg model, and 
        # the global batch size
        self.generator = generator
        self.discriminator = discriminator
        self.vgg = vgg
        self.batchSize = batchSize
```

为了让我们的生活更轻松，我们已经在**线 9** 创建了一个 SRGAN 培训课程。

这个类的`__init__`函数接受以下参数(**第 10 行**):

*   `generator`:SRGAN 的发电机
*   `discriminator`:SRGAN 的鉴别器
*   `vgg`:用于内容丢失的 VGG 网络
*   `batchSize`:训练时使用的批量

在第**行第 14-17** 行，我们简单地通过将类的生成器、鉴别器、VGG 和批处理大小值赋给参数来初始化它们。

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

在第 19 行的**上，我们定义了 SRGAN 的`compile`函数。它接受以下参数:**

*   `gOptimizer`:发电机优化器
*   `dOptimzer`:鉴别器的优化器
*   `bceLoss`:二元交叉熵损失
*   `mseLoss`:均方误差损失

该函数的其余部分初始化相应的类变量(**第 23-28 行**)。

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

在第 30 行的**上，我们定义了`train_step`，它接受训练图像作为它的参数。**

我们继续对图像进行解压缩，并将它们转换成`float`类型(**第 32-34 行**)。

将低分辨率图像通过发生器，我们获得我们的假超分辨率图像。这些与**线 40** 上的真实超分辨率图像相结合。

对于我们的鉴别器训练，我们必须为这组组合图像创建标签。生成器生成的假图像将有一个标签`0`，而真正的高分辨率图像将有一个标签`1` ( **第 45-47 行**)。

```py
        # train the discriminator
        with GradientTape() as tape:
            # get the discriminator predictions
            predictions = self.discriminator(combinedImages)

            # compute the loss
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

为了训练鉴别器，打开`GradientTape`进行反向传播(**线路 50** )。

组合图像集通过用于预测的鉴别器(**行 52** )。使用`bceLoss`，我们通过将它们与标签(**第 55 行**)进行比较来计算鉴频器损耗。

我们计算梯度并根据梯度优化权重(**行 58-65** )。

为了计算发电机重量，我们必须将发电机生成的图像标记为真实图像(**行 68** )。

```py
        # train the generator (note that we should *not* update the
        #  weights of the discriminator)!
        with GradientTape() as tape:
            # get fake images from the generator
            fakeImages = self.generator(lrImages)

            # get the prediction from the discriminator
            predictions = self.discriminator(fakeImages)

            # compute the adversarial loss
            gLoss = 1e-3 * self.bceLoss(misleadingLabels, predictions)

            # compute the normalized vgg outputs
            srVgg = tf.keras.applications.vgg19.preprocess_input(
                fakeImages)
            srVgg = self.vgg(srVgg) / 12.75
            hrVgg = tf.keras.applications.vgg19.preprocess_input(
                hrImages)
            hrVgg = self.vgg(hrVgg) / 12.75

            # compute the perceptual loss
            percLoss = self.mseLoss(hrVgg, srVgg)

            # calculate the total generator loss
            gTotalLoss = gLoss + percLoss

        # compute the gradients
        grads = tape.gradient(gTotalLoss,
            self.generator.trainable_variables)

        # optimize the generator weights with the computed gradients
        self.gOptimizer.apply_gradients(zip(grads,
            self.generator.trainable_variables)
        )

        # return the generator and discriminator losses
        return {"dLoss": dLoss, "gTotalLoss": gTotalLoss,
            "gLoss": gLoss, "percLoss": percLoss}
```

在**线 72** 上，我们为发电机启动另一个`GradientTape`。

在**行 74** 上，我们从通过生成器的低分辨率图像生成假的高分辨率图像。

这些假图像通过鉴别器得到我们的预测。将这些预测与误导标签进行比较，以在第 80 行第 80 行得到我们的二元交叉熵损失。

对于内容损失，我们通过 VGG 网传递假的超分辨率图像和实际的高分辨率图像，并使用我们的均方损失函数对它们进行比较(**第 83-91 行**)。

如前所述，发电机总损耗为发电损耗和容量损耗之和(**行 94** )。

接下来，我们计算生成器的梯度并应用它们(**第 97-103 行**)。

我们的 SRGAN 培训模块到此结束。

* * *

### [**实现最终的实用脚本**](#TOC)

正如您可以从`srgan_training`脚本中发现的，我们使用了一些助手脚本。在评估我们的产出之前，让我们快速浏览一遍。

首先，让我们转到位于`pyimagesearch`目录中的`vgg.py`脚本。

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

在里面，我们在第 5 行的**上定义了一个名为`VGG`的类。在**行第 7** 定义的构建函数使用`tensorflow`包来调用一个预训练的`VGG`模型，并用它来处理我们的内容丢失(**行第 9-16 行**)。**

在培训结束前，我们还有一个脚本要检查。为了帮助评估我们的输出图像，我们创建了一个放大脚本。为此，让我们转到位于`pyimagesearch`目录中的`utils.py`。

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

我们在第 16 行的**上实现我们的`zoom_into_images`函数。它接受图像和图像标题作为参数。**

在第 18 行**定义了多个图的子图。图像绘制在**线 19** 处。在**的第 21-24 行**，我们绘制了相同的图像，但是放大了一个特定的补丁以供参考。**

在**第 26-31 行**，我们指定图像的`x`和`y`坐标限制。

该函数的其余部分涉及一些修饰，如删除记号、添加线条，并将图像保存到我们的输出路径(**第 34-46 行**)。

* * *

### [**训练 SRGAN**](#TOC)

完成所有必需的脚本后，最后一步是执行培训流程。为此，让我们进入根目录中的`train_srgan.py`脚本。

```py
# USAGE
# python train_srgan.py --device tpu
# python train_srgan.py --device gpu

# import tensorflow and fix the random seed for better reproducibility
import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from pyimagesearch.data_preprocess import load_dataset
from pyimagesearch.srgan import SRGAN
from pyimagesearch.vgg import VGG
from pyimagesearch.srgan_training import SRGANTraining
from pyimagesearch import config
from pyimagesearch.losses import Losses
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

创建一个参数解析器来接收来自用户的设备选择输入(**行 26-30** )。

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
```

由于我们可以用 TPU 或 GPU 来训练 SRGAN，因此我们会做出相应的安排。

我们的第一种情况是，如果`device`被设置为`tpu`，我们初始化 TPU 训练的要求(**第 33-43 行**):

*   `TPUClusterResolver`:与集群管理系统(AWS、GCP 等)进行高效通信。)
*   为有效的 TPU 培训初始化正确的策略。

接下来，我们设置训练`TFRecords`路径、预训练生成器路径和最终生成器路径(**第 47-49 行**)。

如果`device`被设置为`gpu`，我们将训练策略设置为在多个 GPU 上镜像(**第 53-55 行**)。

定义了 GPU `TFRecords`路径、预训练生成器和最终生成器路径(**第 59-61 行**)。注意在每种情况下，这些变量是如何被排斥在案例本身之外的(例如，`GPU_GENERATOR_MODEL`或`TPU_GENERATOR_MODEL`)。

如果用户既没有给出`gpu`也没有给出`tpu`选择，程序简单地退出(**第 64-67 行**)。

```py
# display the number of accelerators
print("[INFO] number of accelerators: {}..."
    .format(strategy.num_replicas_in_sync))

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
    generator = SRGAN.generator(
        scalingFactor=config.SCALING_FACTOR,
        featureMaps=config.FEATURE_MAPS,
        residualBlocks=config.RESIDUAL_BLOCKS)
    generator.compile(
        optimizer=Adam(learning_rate=config.PRETRAIN_LR),
        loss=losses.mse_loss)

    # pretraining the generator
    print("[INFO] pretraining SRGAN generator...")
    generator.fit(trainDs, epochs=config.PRETRAIN_EPOCHS,
        steps_per_epoch=config.STEPS_PER_EPOCH)
```

在**的第 70-80 行**，我们从`TFRecords`构建`div2k`数据集。调用策略范围，我们从损失对象初始化损失函数(**第 83-85 行**)。

然后，使用我们的`config.py`脚本中的值初始化生成器，并用`Adam`优化器进行编译(**第 89-95 行**)。

为了获得更好的结果，我们在**线 99 和 100** 上对发电机网络进行了预处理。

```py
# check whether output model directory exists, if it doesn't, then
# create it
if args["device"] == "gpu" and not os.path.exists(config.BASE_OUTPUT_PATH):
    os.makedirs(config.BASE_OUTPUT_PATH)

# save the pretrained generator
print("[INFO] saving the SRGAN pretrained generator to {}..."
    .format(pretrainedGenPath))
generator.save(pretrainedGenPath)

# call the strategy scope context manager
with strategy.scope():
    # initialize our losses class object
    losses = Losses(numReplicas=strategy.num_replicas_in_sync)

    # initialize the vgg network (for perceptual loss) and discriminator
    # network
    vgg = VGG.build()
    discriminator = SRGAN.discriminator(
        featureMaps=config.FEATURE_MAPS, 
        leakyAlpha=config.LEAKY_ALPHA, discBlocks=config.DISC_BLOCKS)

    # build the SRGAN training model and compile it
    srgan = SRGANTraining(
        generator=generator,
        discriminator=discriminator,
        vgg=vgg,
        batchSize=config.TRAIN_BATCH_SIZE)
    srgan.compile(
        dOptimizer=Adam(learning_rate=config.FINETUNE_LR),
        gOptimizer=Adam(learning_rate=config.FINETUNE_LR),
        bceLoss=losses.bce_loss,
        mseLoss=losses.mse_loss,
    )

    # train the SRGAN model
    print("[INFO] training SRGAN...")
    srgan.fit(trainDs, epochs=config.FINETUNE_EPOCHS,
        steps_per_epoch=config.STEPS_PER_EPOCH)

# save the SRGAN generator
print("[INFO] saving SRGAN generator to {}...".format(genPath))
srgan.generator.save(genPath)
```

在**行 104 和 105** 上，我们检查我们的输出的输出路径是否存在。如果没有，我们创建一个。

一旦保存了预训练的生成器，我们就调用策略范围并再次初始化 loss 对象(**第 110-115 行**)。

由于我们将需要`VGG`网络来处理我们的内容丢失，我们初始化了一个`VGG`网络和鉴别器，后者的值在`config.py`脚本中(**第 119-122 行**)。

既然已经创建了生成器和鉴别器，我们就直接使用了`SRGANTraining`对象并编译了我们的 SRGAN 模型(**第 125-135 行**)。

初始化的 SRGAN 与启动训练的数据相匹配，然后保存训练的 SRGAN(**行 139-144** )

* * *

### [**为 SRGAN** 创建推理脚本](#TOC)

我们的训练结束了，让我们看看结果吧！为此，我们将转向`inference.py`脚本。

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

正如在训练脚本中所做的，我们需要创建另一个参数解析器，它从用户那里接受设备(TPU 或 GPU)的选择(**行 21-25** )。

```py
# check if we are using TPU, if so, initialize the strategy
# accordingly
if args["device"] == "tpu":
    # initialize the tpus
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

下一步同样与训练步骤相同。根据设备的选择，我们初始化集群(用于 TPU)、策略(用于 GPU 和 TPU)和路径，同时为脚本设置退出子句(**第 27-63 行**)。

```py
# get the dataset
print("[INFO] loading the test dataset...")
testTfr = glob(tfrTestPath + "/*.tfrec")
testDs = load_dataset(testTfr, config.INFER_BATCH_SIZE, train=False)

# get the first batch of testing images
(lrImage, hrImage) = next(iter(testDs))

# call the strategy scope context manager
with strategy.scope(): 
    # load the SRGAN trained models
    print("[INFO] loading the pre-trained and fully trained SRGAN model...")
    srganPreGen = load_model(pretrainedGenPath, compile=False)
    srganGen = load_model(genPath, compile=False)

    # predict using SRGAN
    print("[INFO] making predictions with pre-trained and fully trained SRGAN model...")
    srganPreGenPred = srganPreGen.predict(lrImage)
    srganGenPred = srganGen.predict(lrImage)
```

在**第 67 行和第 68 行**，我们获得了用于推理的数据集。

使用`next(iter())`，我们得到第一批测试图像(**第 71 行**)。接下来，预训练的 SRGAN 和完全训练的 SRGAN 模型权重被加载和初始化，第一个低分辨率图像通过它们(**第 74-83 行**)。

```py
# plot the respective predictions
print("[INFO] plotting the SRGAN predictions...")
(fig, axes) = subplots(nrows=config.INFER_BATCH_SIZE, ncols=4,
    figsize=(50, 50))

# plot the predicted images from low res to high res
for (ax, lowRes, srPreIm, srGanIm, highRes) in zip(axes, lrImage,
        srganPreGenPred, srganGenPred, hrImage):
    # plot the low resolution image
    ax[0].imshow(array_to_img(lowRes))
    ax[0].set_title("Low Resolution Image")

    # plot the pretrained SRGAN image
    ax[1].imshow(array_to_img(srPreIm))
    ax[1].set_title("SRGAN Pretrained")

    # plot the SRGAN image
    ax[2].imshow(array_to_img(srGanIm))
    ax[2].set_title("SRGAN")

    # plot the high resolution image
    ax[3].imshow(array_to_img(highRes))
    ax[3].set_title("High Resolution Image")
```

在第 87 和 88 行上，我们初始化子情节。然后，循环子图的列，我们绘制低分辨率图像、SRGAN 预训练结果、完全 SRGAN 超分辨率图像和原始高分辨率图像(**第 91-107 行**)。

```py
# check whether output image directory exists, if it doesn't, then
# create it
if not os.path.exists(config.BASE_IMAGE_PATH):
    os.makedirs(config.BASE_IMAGE_PATH)

# serialize the results to disk
print("[INFO] saving the SRGAN predictions to disk...")
fig.savefig(config.GRID_IMAGE_PATH)

# plot the zoomed in images
zoom_into_images(srganPreGenPred[0], "SRGAN Pretrained")
zoom_into_images(srganGenPred[0], "SRGAN")
```

如果输出图像的目录还不存在，我们就创建一个用于存储输出图像的目录(**行 111 和 112** )。

我们保存该图并绘制输出图像的放大版本(**行 116-120** )。

* * *

### [**SRGAN 的训练和可视化**](#TOC)

让我们来看看我们训练过的 SRGAN 的一些图像。**图 4-7** 显示了在 TPU 和 GPU 上训练的 SRGANs 的输出。

我们可以清楚地看到，经过完全训练的 SRGAN 输出比未经训练的 SRGAN 输出显示出更多的细节。

* * *

* * *

## [**汇总**](#TOC)

SRGANs 通过将传统 GAN 元素与旨在提高视觉性能的配方相结合，非常巧妙地实现了更好的图像超分辨率效果。

与以前的工作相比，输出的像素比较的简单增加给了我们明显更强的结果。然而，由于 GANs 本质上是试图重新创建数据，使其看起来像是属于训练分布，因此需要大量的计算能力来实现这一点。SRGANs 可以帮助你实现你的目标，但是问题是你必须准备好大量的计算能力。

* * *

### [**引用信息**](#TOC)

****Chakraborty，D.**** 【超分辨率生成对抗网络(SRGAN)】*PyImageSearch*，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva，R. Raha，A. Thanki 合编。，2022 年，[https://pyimg.co/lgnrx](https://pyimg.co/lgnrx)

```py
@incollection{Chakraborty_2022_SRGAN,
  author = {Devjyoti Chakraborty},
  title = {Super-Resolution Generative Adversarial Networks {(SRGAN)}},
  booktitle = {PyImageSearch},
  editor = {Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/lgnrx},
}
```

* * *

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！********