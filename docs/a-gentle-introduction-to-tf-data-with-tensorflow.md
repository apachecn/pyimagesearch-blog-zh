# 使用 TensorFlow 轻松介绍 tf.data

> 原文：<https://pyimagesearch.com/2021/06/14/a-gentle-introduction-to-tf-data-with-tensorflow/>

在本教程中，您将学习 TensorFlow 的`tf.data`模块的基础知识，该模块用于构建更快、更高效的深度学习数据管道。

这篇博文是我们关于`tf.data`的三部分系列的第一部分:

1.  *TF . data 的温和介绍*(本教程)
2.  *使用 tf.data 和 TensorFlow 的数据管道*(下周博文)
3.  *用 tf.data 进行数据扩充*(两周后的教程)

在我们开始之前，这里有一个你需要知道的快速分类:

*   `tf.data`模块允许我们在可重用的代码块中构建*复杂的*和*高效的*数据处理管道。
*   它非常容易使用
*   与使用通常用于训练 Keras 和 TensorFlow 模型的`ImageDataGenerator`对象相比，`tf.data`模块比**≈***快 38 倍。*
*   *我们可以很容易地用`tf.data`代替`ImageDataGenerator`通话*

 *只要有可能，你应该考虑交换掉`ImageDataGenerator`调用，转而使用`tf.data`——你的数据处理管道将*快得多*(因此，你的模型将训练得更快)。

**学习基本面`tf.data`进行深度学习，** ***保持阅读即可。***

## **使用 TensorFlow 对 tf.data 的简单介绍**

在本教程的第一部分，我们将讨论 TensorFlow 的`tf.data`模块是什么，包括它对于构建数据处理管道是否有效。

然后，我们将配置我们的开发环境，检查我们的项目目录结构，并讨论我们将用于基准测试`tf.data`和`ImageDataGenerator`的示例图像数据集。

为了比较`tf.data`和`ImageDataGenerator`的数据处理速度，我们今天将实现两个脚本:

1.  第一个将在处理一个完全符合内存中*的数据集时，比较`tf.data`和`ImageDataGenerator`*
2.  然后，当处理驻留在磁盘上图像时，我们将实现第二个东西来比较`tf.data`和`ImageDataGenerator`

最后，我们将运行每个脚本并比较我们的结果。

### **什么是“tf.data”模块？**

PyTorch 库的用户可能对`Dataset`和`DatasetLoader`类很熟悉——它们使得加载和预处理数据变得非常容易、高效和快速。

在 TensorFlow v2 之前，Keras 和 TensorFlow 用户必须:

1.  手动定义自己的数据加载功能
2.  利用 Keras 的`ImageDataGenerator`功能处理太大而无法装入内存的图像数据集和/或需要应用数据扩充时

两种解决方案都不是最好的。

手动实现自己的数据加载功能是一项艰苦的工作，可能会导致错误。`ImageDataGenerator`函数虽然是一个非常好的选择，但也不是最快的方法。

TensorFlow v2 API 经历了许多变化，可以说最大/最重要的变化之一是引入了`tf.data`模块。

引用 TensorFlow 文档:

> tf.data API 使您能够从简单的、可重用的部分构建复杂的输入管道。 *例如，图像模型的流水线可以从分布式文件系统中的文件聚集数据，对每个图像应用随机扰动，并将随机选择的图像合并成一批用于训练。文本模型的管道可能涉及从原始文本数据中提取符号，用查找表将它们转换为嵌入的标识符，以及将不同长度的序列批处理在一起。tf.data API 使得处理大量数据、读取不同的数据格式以及执行复杂的转换成为可能。*

使用`tf.data`处理数据现在*明显更容易*——正如我们将看到的，它也比依赖旧的`ImageDataGenerator`类更快更有效。

### **TF . data 构建数据管道效率更高吗？**

**简短的回答是** ***是的，*** **使用`tf.data`比使用`ImageDataGenerator`***明显更快更高效——正如本教程的结果将向您展示的那样，当处理内存中的数据集时，我们能够获得大约 6.1 倍的加速，当处理驻留在磁盘上的图像数据时，我们能够获得大约 38 倍的效率提升。*****

 *`tf.data`的“秘方”在于 TensorFlow 的多线程/多处理实现，更具体地说，是“自动调整”的概念。

训练神经网络，尤其是在大型数据集上，无非是生产者/消费者关系。

神经网络需要*消耗*数据，以便它可以从中学习(即，当前的一批数据)。然而，数据处理实用程序本身负责*向神经网络产生*该数据。

一个非常天真的单线程数据生成实现将创建一个数据批，将其发送到神经网络，等待它请求更多数据，然后(只有到那时)，创建一个新的数据批。这种实现的问题是数据处理器和神经网络都处于空闲状态，直到另一个完成其各自的工作。

相反，我们可以利用并行处理，这样任何组件都不会处于等待状态——数据处理器可以在内存中维护一个由 *N* 个批次组成的队列，同时网络在队列中训练下一个批次。

TensorFlow 的`tf.data`模块可以根据你的硬件/实验设计自动调整整个过程，从而使训练*明显更快*。

**总的来说，我建议尽可能用`tf.data`代替`ImageDataGenerator`通话。**

### **配置您的开发环境**

对 tf.data 的介绍使用了 Keras 和 TensorFlow。如果你打算遵循这个教程，我建议你花时间配置你的深度学习开发环境。

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

### **ka ggle 水果和蔬菜数据集**

在本教程中，我们将实现两个 Python 脚本:一个在适合内存的数据集中使用`tf.data`(例如，CIFAR-100 ),另一个脚本在图像数据集太大而不适合 RAM 的情况下使用`tf.data`(因此我们需要从磁盘中批处理数据)。

对于后一种实现，我们将利用 [Kaggle 水果和蔬菜数据集](https://www.kaggle.com/jorgebailon/fruits-vegetables)。

该数据集由 6，688 幅图像组成，属于七个大致平衡的类别:

1.  苹果(955 张图片)
2.  西兰花(946 张图片)
3.  葡萄(965 张图片)
4.  柠檬(944 张图片)
5.  芒果(934 张图片)
6.  橙色(970 张图片)
7.  草莓(976 张图片)

目标是训练一个模型，以便我们可以正确地对输入图像中的水果/蔬菜进行分类。

**为了方便起见，我在本教程的** ***【下载】*** **部分包含了 Kaggle 水果/蔬菜数据集。**

### **项目结构**

在我们继续之前，让我们首先回顾一下我们的项目目录结构。首先访问本指南的 ***“下载”*** 部分，检索源代码和 Python 脚本。

然后，您将看到以下目录结构:

```py
$ tree . --Ddirsfirst --filelimit 10
.
├── fruits
│   ├── apple [955 entries exceeds filelimit, not opening dir]
│   ├── broccoli [946 entries exceeds filelimit, not opening dir]
│   ├── grape [965 entries exceeds filelimit, not opening dir]
│   ├── lemon [944 entries exceeds filelimit, not opening dir]
│   ├── mango [934 entries exceeds filelimit, not opening dir]
│   ├── orange [970 entries exceeds filelimit, not opening dir]
│   └── strawberry [976 entries exceeds filelimit, not opening dir]
├── pyimagesearch
│   ├── __init__.py
│   └── helpers.py
├── reading_from_disk.py
└── reading_from_memory.py

9 directories, 4 files
```

`fruits`目录包含我们的图像数据集。

在`pyimagesearch`模块中，我们有一个单独的文件`helpers.py`，它包含一个单独的函数`benchmark`——这个函数将用于基准测试我们的`tf.data`和`ImageDataGenerator`对象。

最后，我们有两个 Python 脚本:

1.  `reading_from_memory.py`:对可以容纳*内存*的数据集进行数据处理速度基准测试
2.  `reading_from_disk.py`:针对太大而无法放入 RAM 的图像数据集，对我们的数据管道进行基准测试

现在让我们开始实现吧。

### **创建我们的基准计时函数**

在我们对`tf.data`和`ImageDataGenerator`进行基准测试之前，我们首先需要创建一个函数，它将:

1.  启动计时器
2.  要求`tf.data`或`ImageDataGenerator`生成总共 *N* 批数据
3.  停止计时器
4.  衡量这个过程需要多长时间

打开`pyimagesearch`模块中的`helpers.py`文件，现在我们将实现我们的`benchmark`功能:

```py
# import the necessary packages
import time

def benchmark(datasetGen, numSteps):
	# start our timer
	start = time.time()

	# loop over the provided number of steps
	for i in range(0, numSteps):
		# get the next batch of data (we don't do anything with the
		# data since we are just benchmarking)
		(images, labels) = next(datasetGen)

	# stop the timer
	end = time.time()

	# return the difference between end and start times
	return (end - start)
```

**第 2 行**导入了我们唯一需要的 Python 包`time`，它将允许我们在生成数据批次之前和之后抓取时间戳。

我们的`benchmark`功能定义在**行 4** 上。此方法需要两个参数:

1.  `datasetGen`:我们的数据集生成器，我们假设它是`tf.data.Dataset`或`ImageDataGenerator`的一个实例
2.  `numSteps`:要生成的数据批次总数

**线 6** 抢`start`时间。

然后我们继续在**线 9** 上循环所提供的批次数量。对`next`函数的调用(**第 12 行**)告诉`datasetGen`迭代器产生下一批数据。

我们在第 15 行**停止计时器，然后返回数据生成过程花费的时间。**

### **使用 tf.data 在内存中构建数据集**

随着我们的`benchmark`函数的实现，现在让我们创建一个脚本来比较`tf.data`和`ImageDataGenerator`的*内存中的*数据集(即，没有磁盘访问)。

根据定义，内存中的数据集可以放入系统的 RAM 中，这意味着不需要对磁盘进行昂贵的 I/O 操作。在这篇文章的后面，我们也将看看磁盘上的数据集，但是让我们先从更简单的方法开始。

打开项目目录中的`reading_from_memory.py`文件，让我们开始工作:

```py
# import the necessary packages
from pyimagesearch.helpers import benchmark
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar100
from tensorflow.data import AUTOTUNE
import tensorflow as tf
```

**第 2-6 行**导入我们需要的 Python 包，包括:

*   `benchmark`:我们在上一节中刚刚实现的基准测试功能
*   `ImageDataGenerator` : Keras 的数据生成迭代器
*   `cifar100`:CIFAR-100 数据集(即我们将用于基准测试的数据集)
*   `AUTOTUNE`:从基准测试的角度来看，这无疑是最重要的导入——使用 TensorFlow 的自动调优功能，我们可以在使用`tf.data`时获得大约 6.1 倍的加速
*   `tensorflow`:我们的 TensorFlow 库

现在让我们来关注一些重要的初始化:

```py
# initialize the batch size and number of steps
BS = 64
NUM_STEPS = 5000

# load the CIFAR-10 dataset from
print("[INFO] loading the cifar100 dataset...")
((trainX, trainY), (testX, testY)) = cifar100.load_data()
```

当对`tf.data`和`ImageDataGenerator`进行基准测试时，我们将使用`64`的批量。我们将要求每个方法生成`5000`批数据。然后我们将评估这两种方法，看看哪一种更快。

**第 14 行**从磁盘加载 CIFAR-100 数据集。我们将从这个数据集生成批量数据。

接下来，让我们实例化我们的`ImageDataGenerator`:

```py
# create a standard image generator object
print("[INFO] creating a ImageDataGenerator object...")
imageGen = ImageDataGenerator()
dataGen = imageGen.flow(
	x=trainX, y=trainY,
	batch_size=BS, shuffle=True)
```

**第 18 行**初始化一个“空”`ImageDataGenerator`(即不应用[数据增加](https://pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/)，只产生数据)。

然后我们通过调用`flow`方法在**的第 19-21 行**创建数据迭代器。我们将训练数据作为我们各自的`x`和`y`参数，连同我们的`batch_size`一起传入。`shuffle`参数设置为`True`，表示数据将被混洗。

现在，我们期待已久的时刻到来了——使用`tf.data`创建数据管道:

```py
# build a TensorFlow dataset from the training data
dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))

# build the data input pipeline
print("[INFO] creating a tf.data input pipeline..")
dataset = (dataset
	.shuffle(1024)
	.cache()
	.repeat()
	.batch(BS)
	.prefetch(AUTOTUNE)
)
```

**第 24 行**创建了我们的`dataset`，它是从`from_tensor_slices`函数返回的。该函数创建 TensorFlow 的`Dataset`对象的实例，其数据集的元素是单个数据点(即，在本例中，图像和类标签)。

**第 28-34 行**通过链接大量 TensorFlow 方法来创建数据集管道本身。让我们逐一分析:

1.  `shuffle`:用`1024`个数据点随机填充一个数据缓冲区，并随机打乱缓冲区中的数据。当数据被拉出缓冲区时(比如抓取下一批数据时)，TensorFlow 会自动重新填充缓冲区。
2.  `cache`:高效地缓存数据集，以便更快地进行后续读取。
3.  `repeat`:告诉`tf.data`继续循环我们的数据集(否则，一旦我们用完数据/一个时期结束，TensorFlow 就会出错)。
4.  `batch`:返回一批`BS`数据点(在本例中，该批中共有 64 个图像和类别标签。
5.  `prefetch` : **从效率角度来看流水线中最重要的功能** —告诉`tf.data`在处理*当前*数据的同时，在后台准备*更多的*数据

调用`prefetch`可以确保神经网络需要的下一批数据*始终可用*，并且网络不必等待数据生成过程返回数据，从而提高吞吐量和延迟。

当然，调用`prefetch`会导致额外的 RAM 被消耗(来保存额外的数据点)，但是额外的内存消耗*是值得的*，因为我们将有一个显著更高的吞吐率。

`AUTOTUNE`参数告诉 TensorFlow 构建我们的管道，然后进行优化，以便我们的 CPU 可以为管道中的每个参数预算时间。

完成初始化后，我们可以对我们的`ImageDataGenerator`进行基准测试:

```py
# benchmark the image data generator and display the number of data
# points generated, along with the time taken to perform the
# operation
totalTime = benchmark(dataGen, NUM_STEPS)
print("[INFO] ImageDataGenerator generated {} images in " \
	  " {:.2f} seconds...".format(
	BS * NUM_STEPS, totalTime))
```

**第 39 行**告诉`ImageDataGenerator`对象总共生成`NUM_STEPS`批数据。然后，我们在**的第 40-42 行**显示经过的时间。

类似地，我们对我们的`tf.data`对象做同样的事情:

```py
# create a dataset iterator, benchmark the tf.data pipeline, and
# display the number of data points generator along with the time taken
datasetGen = iter(dataset)
totalTime = benchmark(datasetGen, NUM_STEPS)
print("[INFO] tf.data generated {} images in {:.2f} seconds...".format(
	BS * NUM_STEPS, totalTime))
```

我们现在准备比较我们的`tf.data`管道和我们的`ImageDataGenerator`管道的性能。

### **比较内存数据集的 ImageDataGenerator 和 TF . data**

我们现在准备比较 Keras 的`ImageDataGenerator`类和 TensorFlow v2 的`tf.data`模块。

请务必访问本指南的 ***“下载”*** 部分以检索源代码。

从那里，打开一个终端并执行以下命令:

```py
$ python reading_from_memory.py
[INFO] loading the cifar100 dataset...
[INFO] creating a ImageDataGenerator object...
[INFO] creating a tf.data input pipeline..
[INFO] ImageDataGenerator generated 320000 images in 5.65 seconds...
[INFO] tf.data generated 320000 images in 0.92 seconds...
```

在这里，我们从磁盘加载 CIFAR-100 数据集。然后我们构造一个`ImageDataGenerator`对象和一个`tf.data`管道。

我们要求这两个类总共生成 5，000 个批处理。每批 64 幅图像导致生成总共 320，000 幅图像。

首先评估`ImageDatanGenerator`，在 5.65 秒内完成任务。然后`tf.data`模块运行，在 0.92 秒内完成同样的任务，**T3 加速≈6.1x！**

正如我们将看到的，当我们开始处理驻留在磁盘上的图像数据时，性能的提高甚至更加明显。

### **使用驻留在磁盘上的 tf.data 和数据集**

在上一节中，我们学习了如何为驻留在内存*中的数据集构建 TensorFlow 数据管道，但是驻留在磁盘上的数据集呢？*

使用`tf.data`管道是否也能提高磁盘数据集的性能？

剧透警告:

答案是*“是的，绝对是”* —正如我们将看到的，性能改进甚至更加显著。

打开项目目录中的`reading_from_disk.py`文件，让我们开始工作:

```py
# import the necessary packages
from pyimagesearch.helpers import benchmark
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.data import AUTOTUNE
from imutils import paths
import tensorflow as tf
import numpy as np
import argparse
import os
```

**第 2-9 行**导入我们需要的 Python 包。这些导入或多或少与我们之前的例子相同，唯一显著的不同是来自`imutils`的`paths`子模块——使用`paths`将允许我们获取驻留在磁盘上的图像数据集的文件路径。

对驻留在磁盘上的图像使用`tf.data`管道要求我们定义一个函数，该函数可以:

1.  接受输入图像路径
2.  从磁盘加载图像
3.  返回图像数据和类别标签(或者边界框坐标，例如，如果您正在进行对象检测)

**此外，从磁盘加载图像和解析类标签** ***时，我们应该使用*** ***TensorFlow 自带的函数*** **，而不是使用 OpenCV、NumPy 或 Python 的内置函数** **。**

使用 TensorFlow 的方法将允许`tf.data`进一步优化自己的管道，从而使其运行得更快。

现在让我们定义我们的`load_images`函数:

```py
def load_images(imagePath):
	# read the image from disk, decode it, resize it, and scale the
	# pixels intensities to the range [0, 1]
	image = tf.io.read_file(imagePath)
	image = tf.image.decode_png(image, channels=3)
	image = tf.image.resize(image, (96, 96)) / 255.0

	# grab the label and encode it
	label = tf.strings.split(imagePath, os.path.sep)[-2]
	oneHot = label == classNames
	encodedLabel = tf.argmax(oneHot)

	# return the image and the integer encoded label
	return (image, encodedLabel)
```

`load_images`方法接受一个参数`imagePath`，它是我们想要从磁盘加载的图像的路径。

然后，我们继续:

1.  从磁盘加载图像(**第 14 行**)
2.  将图像解码为具有 3 个通道的无损 PNG(**第 15 行**)
3.  将图像大小调整为 *96* *×* *96* 像素，将像素亮度从*【0，255】*缩放到*【0，1】*

此时，我们可以将`image`返回到`tf.data`管道；然而，我们仍然需要从`imagePath`中解析我们的类标签:

1.  **第 19 行**根据我们操作系统的文件路径分隔符分割`imagePath`字符串，并获取倒数第二个条目，在本例中，这是标签的子目录名称(参见本教程的*“项目结构”*部分，了解这是如何实现的)
2.  **第 20 行**使用我们的`classNames`(我们将在后面的脚本中定义)对`label`进行单热编码
3.  然后，我们在第 21 行的**处获取标签“热”的整数索引**

得到的 2 元组`image`和`encodedLabel`被返回到第 23 行**上的调用函数。**

现在已经定义了我们的`load_images`函数，让我们继续这个脚本的其余部分:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())
```

**第 27-30 行**解析我们的命令行参数。这里我们只需要一个参数，`--dataset`，它是驻留在磁盘上的水果和蔬菜数据集的路径。

这里，我们关注一些重要的初始化:

```py
# initialize batch size and number of steps
BS = 64
NUM_STEPS = 1000

# grab the list of images in our dataset directory and grab all
# unique class names
print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = np.array(sorted(os.listdir(args["dataset"])))
```

**第 33 行和第 34 行**定义了我们的批量大小(`64`)以及我们将在评估过程(`1000`)中生成的数据的批量。

**第 39 行**从我们的输入`--dataset`目录中抓取所有图像文件路径，而**第 40 行**从文件路径中提取类标签名。

现在，让我们创建我们的`tf.data`管道:

```py
# build the dataset and data input pipeline
print("[INFO] creating a tf.data input pipeline..")
dataset = tf.data.Dataset.from_tensor_slices(imagePaths)
dataset = (dataset
	.shuffle(1024)
	.map(load_images, num_parallel_calls=AUTOTUNE)
	.cache()
	.repeat()
	.batch(BS)
	.prefetch(AUTOTUNE)
)
```

我们再次调用了`from_tensor_slices`函数，但是这次传入了我们的`imagePaths`。这样做将创建一个`tf.data.Dataset`实例，其中数据集的元素是单独的文件路径。

然后我们定义管道本身(**第 45-52 行**)

1.  `shuffle`:从数据集中构建一个`1024`元素的缓冲区，并对其进行混洗。
2.  `map`:将`load_images`函数映射到批处理中的所有图像路径。**这一行代码负责从磁盘实际加载我们的输入图像，并解析类标签。**`AUTOTUNE`参数告诉 TensorFlow 自动优化这个函数调用，使其尽可能高效。
3.  `cache`:缓存结果，从而使后续的数据读取/访问更快。
4.  `repeat`:一旦到达数据集/时段的末尾，重复该过程。
5.  `batch`:返回一批数据。
6.  `prefetch`:在后台构建批量数据，从而提高吞吐率。

我们使用 Keras 的`ImageDataGenerator`对象创建一个类似的数据管道:

```py
# create a standard image generator object
print("[INFO] creating a ImageDataGenerator object...")
imageGen = ImageDataGenerator(rescale=1.0/255)
dataGen = imageGen.flow_from_directory(
	args["dataset"],
	target_size=(96, 96),
	batch_size=BS,
	class_mode="categorical",
	color_mode="rgb")
```

**第 56 行**创建`ImageDataGenerator`，告诉它从范围*【0，255】*到*【0，1】重新调整所有输入图像的像素强度。*

`flow_from_directory`函数告诉`dataGen`我们将从我们的输入`--dataset`目录中读取图像。从那里，每个图像被调整到 *96×96* 像素，并且返回`BS`大小的批次。

我们对下面的`ImageDataGenerator`进行了基准测试:

```py
# benchmark the image data generator and display the number of data
# points generated, along with the time taken to perform the
# operation
totalTime = benchmark(dataGen, NUM_STEPS)
print("[INFO] ImageDataGenerator generated {} images in " \
	  " {:.2f} seconds...".format(
	BS * NUM_STEPS, totalTime))
```

在这里，我们对我们的`tf.data`渠道进行了基准测试:

```py
# create a dataset iterator, benchmark the tf.data pipeline, and
# display the number of data points generated, along with the time
# taken
datasetGen = iter(dataset)
totalTime = benchmark(datasetGen, NUM_STEPS)
print("[INFO] tf.data generated {} images in {:.2f} seconds...".format(
	BS * NUM_STEPS, totalTime))
```

但是哪种方法更快呢？

我们将在下一节回答这个问题。

### **比较磁盘数据集的 ImageDataGenerator 和 TF . data**

让我们比较一下使用磁盘数据集时的`ImageDataGenerator`和`tf.data`。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例数据。

从那里，您可以执行`reading_from_disk.py`脚本:

```py
$ python reading_from_disk.py --dataset fruits
[INFO] loading image paths...
[INFO] creating a tf.data input pipeline..
[INFO] creating a ImageDataGenerator object...
Found 6688 images belonging to 7 classes.
[INFO] ImageDataGenerator generated 64000 images in 258.17 seconds...
[INFO] tf.data generated 64000 images in 6.81 seconds...
```

我们的`ImageDataGenerator`通过生成总共 1，000 个批次进行基准测试，每个批次中有 64 个图像，总共有 64，000 个图像。对于`ImageDataGenerator`来说，这个过程需要 258 秒多一点。

**`tf.data`模块执行同样的任务 6.81 秒——*****大幅度提升≈38x！***

只要有可能，我推荐使用`tf.data`而不是`ImageDataGenerator`。性能的提高简直是惊人的。

## **总结**

在本教程中，我们介绍了`tf.data`，一个用于高效数据加载和预处理的 TensorFlow 模块。

然后，我们将`tf.data`的性能与 Keras 中的原始`ImageDataGenerator`类进行了基准测试:

*   当处理内存数据集时，我们使用`tf.data`获得了 6.1 倍的吞吐率
*   当在磁盘上处理图像数据时，我们在使用`tf.data`时达到了惊人的**≈38 倍**的吞吐率

**数字说明了一切——如果您需要更快的数据加载和预处理，利用`tf.data`是值得的。**

下周我将带着一些使用`tf.data`训练神经网络的更高级的例子回来。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******