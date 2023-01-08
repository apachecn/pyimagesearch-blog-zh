# TFRecords 简介

> 原文：<https://pyimagesearch.com/2022/08/08/introduction-to-tfrecords/>

* * *

## **目录**

* * *

## [**TF records 简介**](#TOC)

在本教程中，您将了解 TensorFlow 的 TFRecords。

**要了解如何使用 TensorFlow 的 TFRecords 格式，** ***继续阅读。***

* * *

## [**TF records 简介**](#TOC)

* * *

### [**简介**](#TOC)

本教程的目标是作为一个**一站式**的目的地，让你了解关于 TFRecords 的一切。我们有目的地组织教程，使你对主题有更深的理解。它是为初学者设计的，我们希望你有**没有**关于这个主题的知识。

因此，没有任何进一步的拖延，让我们直接进入我们的教程。

* * *

### [**配置您的开发环境**](#TOC)

要遵循本指南，您需要在系统上安装 TensorFlow 和 TensorFlow 数据集库。

幸运的是，两者都是 pip 可安装的:

```py
$ pip install tensorflow
$ pip install tensorflow-datasets
```

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
.
├── create_tfrecords.py
├── example_tf_record.py
├── pyimagesearch
│   ├── advance_config.py
│   ├── config.py
│   ├── __init__.py
│   └── utils.py
├── serialization.py
└── single_tf_record.py

1 directory, 8 files
```

在`pyimagesearch`目录中，我们有:

*   `utils.py`:从磁盘加载和保存图像的实用程序。
*   `config.py`:单数据 tfrecord 示例的配置文件。
*   `advance_config.py`:[`div2k`](https://data.vision.ee.ethz.ch/cvl/DIV2K/)数据集示例的配置文件。

在核心目录中，我们有四个脚本:

*   `single_tf_record.py`:处理单个二进制记录的脚本，展示了如何将它保存为 TFRecord 格式。
*   `serialization.py`:解释数据序列化重要性的脚本。
*   `example_tf_record.py`:保存和加载单幅图像为 TFRecord 的脚本。
*   `create_tfrecords.py`:保存和加载整个 [`div2k`](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 数据集的脚本。

* * *

### [**什么是 TFRecords？**](#TOC)

TFRecord 是用于存储二进制记录序列的自定义 TensorFlow 格式。TFRecords 针对 TensorFlow 进行了高度优化，因此具有以下优势:

*   数据存储的有效形式
*   与其他类型的格式相比，读取速度更快

TFRecords 最重要的一个用例是当我们使用 TPU 训练一个模型时。TPU 超级强大，但要求与之交互的数据远程存储(通常，我们使用 Google 云存储)，这就是 TFRecords 的用武之地。当在 TPU 上训练模型时，我们以 TFRecord 格式远程存储数据集，因为这使得有效地保存数据和更容易地加载数据。

在这篇博文中，您将了解从如何构建基本 TF record 到用于训练 SRGAN 和 e SRGAN 模型的高级 TF record 的所有内容，这些内容将在以下博文中介绍:

*   [https://pyimagesearch . com/2022/06/06/super-resolution-generative-adversarial-networks-srgan/](https://pyimagesearch.com/2022/06/06/super-resolution-generative-adversarial-networks-srgan/)
*   [https://pyimagesearch . com/2022/06/13/enhanced-super-resolution-generative-adversarial-networks-ESR gan/](https://pyimagesearch.com/2022/06/13/enhanced-super-resolution-generative-adversarial-networks-esrgan/)

在我们开始之前，我们想提一下，这篇博客文章的灵感很大程度上来自于 Ryan Holbrook 关于 TFRecords 基础知识的 Kaggle 笔记本和 T2 tensor flow 关于 TFRecords 的指南。

* * *

### [**建立自己的 TFRecords**](#TOC)

* * *

#### [**建立 TFRecord**](#TOC)

先说简单的。我们将创建二进制记录(字节字符串)，然后使用 API 将它们保存到 TFRecord 中。这将使我们了解如何将大型数据集保存到 TFRecords 中。

```py
# USAGE
# python single_tf_record.py

# import the necessary packages
from pyimagesearch import config
from tensorflow.io import TFRecordWriter
from tensorflow.data import TFRecordDataset

# build a byte-string that will be our binary record
record = "12345"
binaryRecord = record.encode()

# print the original data and the encoded data
print(f"Original data: {record}")
print(f"Encoded data: {binaryRecord}")

# use the with context to initialize the record writer
with TFRecordWriter(config.TFRECORD_SINGLE_FNAME) as recordWriter:
	# write the binary record into the TFRecord
	recordWriter.write(binaryRecord)

# open the TFRecord file
with open(config.TFRECORD_SINGLE_FNAME, "rb") as filePointer:
	# print the binary record from the TFRecord
	print(f"Data from the TFRecord: {filePointer.read()}")

# build a dataset from the TFRecord and iterate over it to read
# the data in the decoded format
dataset = TFRecordDataset(config.TFRECORD_SINGLE_FNAME)
for element in dataset:
	# fetch the string from the binary record and then decode it
	element = element.numpy().decode()

	# print the decoded data
	print(f"Decoded data: {element}")
```

在**第 5-7 行**，我们导入我们需要的包。

让我们初始化想要存储为 TFRecord 的数据。在**的第 10 行**，我们构建了一个名为`record`的变量，用字符串`"12345"`初始化。接下来，在**的第 11 行**，我们将这个字符串编码成一个字节串。

这一点特别重要，因为 TFRecords 只能存储二进制记录，而字节串就是这样。

**行** **14 和** **15** 打印原始的和编码的字符串，以显示两者的区别。当我们查看脚本的输出时，我们将能够注意到差异。

**第 18-20 行**初始化一个`TFRecordWriter`。我们可以使用`write()` API 任意多次，只要我们想将二进制记录写入 TFRecord。注意`TFRecordWriter`是如何使用`with`上下文的。

在第 23-25 行上，我们打开 TFRecord 文件并检查其数据。

第 29 行对我们特别有用，因为我们现在可以从任何 TFRecord 构建一个`tf.data.Dataset`。

一旦我们有了数据集，我们就可以遍历它，按照我们自己的意愿使用数据。

你可以参考`tf.data`上的这篇[博文](https://pyimagesearch.com/2021/06/14/a-gentle-introduction-to-tf-data-with-tensorflow/)来温习一些基础知识。

让我们看看这个脚本的输出。

```py
$ python single_tf_record.py

Original data: 12345
Encoded data: b'12345'
Data from the TFRecord: b'\x05\x00\x00\x00\x00\x00\x00\x00\xea\xb2\x04>12345z\x1c\xed\xe8'
Decoded data: 12345
```

请注意以下事项:

*   原始数据和解码数据应该是相同的。
*   编码的数据只是原始数据的字节串。
*   TFRecord 中的数据是序列化的二进制记录。

* * *

#### [**序列化**](#TOC)

那么，什么是序列化二进制记录呢？

我们知道 TFRecords 存储一系列二进制记录。因此，我们首先需要学习如何将数据转换成二进制表示。稍后我们将在此基础上建立我们的直觉。

TensorFlow 有两个公共 API，负责在二进制记录中编码和解码数据。这两个公共 API 分别来自 [`tf.io.serialize_tensor`](https://www.tensorflow.org/api_docs/python/tf/io/serialize_tensor) 和 [`tf.io.parse_tensor`](https://www.tensorflow.org/api_docs/python/tf/io/parse_tensor) 。

让我们用这个例子来弄脏我们的手。

```py
# USAGE
# python serialization.py

# import the necessary packages
import tensorflow as tf
from tensorflow.io import serialize_tensor, parse_tensor

# build the original data
originalData = tf.constant(
	value=[1, 2, 3, 4],
	dtype=tf.dtypes.uint8
)

# serialize the data into binary records
serializedData = serialize_tensor(originalData)

# read the serialized data into the original format
parsedData = parse_tensor(
	serializedData,
	out_type=tf.dtypes.uint8
)

# print the original, encoded, and the decoded data
print(f"Original Data: {originalData}\n")
print(f"Encoded Data: {serializedData}\n")
print(f"Decoded Data: {parsedData}\n")
```

**行** **5 和** **6** 包含代码运行所需的导入。我们正在导入我们的盟友 TensorFlow 和两个用于**序列化**和**反序列化**的公共 API。

在**第 9-12 行**上，我们构建了一个`tf.constant`，它将作为我们的原始数据。该数据属于数据类型`tf.dtypes.uint8`。原始数据的**数据类型**很重要，因为在我们对二进制记录进行反序列化(解码)时需要用到它。

在**的第 15 行**，我们将`originalData`序列化为字节串(二进制表示)。

在**的第 18-21 行**，我们将`serializedData`反序列化为原始格式。注意我们需要如何使用参数`out_type`指定输出格式。这里我们提供与原始数据相同的数据类型(`tf.dtypes.uint8`)。

**第 2 行** **第 4 行** **-第 26 行**是帮助我们可视化流程的打印语句。我们来看看输出。

```py
$ python serialization.py

Original Data: [1 2 3 4]

Encoded Data: b'\x08\x04\x12\x04\x12\x02\x08\x04"\x04\x01\x02\x03\x04'

Decoded Data: [1 2 3 4]
```

从输出中可以明显看出，原始数据被序列化为一系列字节字符串，然后反序列化为原始数据。

**TF records from structured`tf.data`:**让我们稍微回顾一下我们刚刚讲过的内容。

我们知道二进制记录是如何存储在 TFRecord 中的；我们还讨论了将任何数据解析成二进制记录。现在，我们将深入研究将结构化数据集转换为 TFRecords 的过程。

在这一步中，我们将需要到目前为止我们已经讨论过的所有先决条件。

在我们处理整个数据集之前，让我们首先尝试理解如何处理数据集的单个实例。

数据集(结构化数据集)由单个实例组成。这些实例可以认为是一个 [`Example`](https://www.tensorflow.org/api_docs/python/tf/train/Example) 。让我们把一个图像和类名对看作一个`Example`。这个例子由两个单独的 [`Feature`](https://www.tensorflow.org/api_docs/python/tf/train/Feature) 合称为 [`Features`](https://www.tensorflow.org/api_docs/python/tf/train/Features) 。其中一个`Feature`是图像，另一个是类名。

从 TFRecords 上的 TensorFlow 官方指南中，如图**图 2** 所示，我们可以看到`tf.train.Feature`可以接受的不同数据类型。

这意味着该特性需要被序列化为上述列表中的一个，然后被打包成一个特性。

让我们通过下面的例子来看看如何做到这一点。

```py
# import the necessary packages
import tensorflow as tf
from pyimagesearch import config
from tensorflow.io import read_file
from tensorflow.image import decode_image, convert_image_dtype, resize
import matplotlib.pyplot as plt
import os

def load_image(pathToImage):
	# read the image from the path and decode the image
	image = read_file(pathToImage)
	image = decode_image(image, channels=3)

	# convert the image data type and resize it
	image = convert_image_dtype(image, tf.dtypes.float32)
	image = resize(image, (16, 16))

	# return the processed image
	return image

def save_image(image, saveImagePath, title=None):
	# show the image
	plt.imshow(image)

	# check if title is provided, if so, add the title to the plot
	if title:
		plt.title(title)

	# turn off the axis and save the plot to disk
	plt.axis("off")
	plt.savefig(saveImagePath)
```

在这个例子中，我们同时做很多事情。首先，让我们查看一下`utils.py`文件，了解一下预处理步骤。

在**2-7 行**，我们导入必要的包。接下来，我们在第 9-19 行的**上定义我们的`load_image`函数，它从磁盘读取一个图像，将其转换为 32 位浮点格式，将图像大小调整为 16×16，然后返回。**

接下来，我们在第 21-31 行的**上定义我们的`save_image`函数，它将图像和输出图像路径作为输入。在**第 23 行**上，我们显示图像，然后在**第 26 行和第 27 行**上设置情节标题。最后，我们将图像保存到磁盘的第 30 行和第 31 行。**

现在让我们看看如何从磁盘加载一个原始图像，并将其序列化为 TFRecord 格式。然后，我们将看到如何加载序列化的 TFRecord 并反序列化图像。

```py
# USAGE
# python example_tf_record.py

# import the necessary packages
from pyimagesearch import config
from pyimagesearch import utils
from tensorflow.keras.utils import get_file
from tensorflow.io import serialize_tensor
from tensorflow.io import parse_example
from tensorflow.io import parse_tensor
from tensorflow.io import TFRecordWriter
from tensorflow.io import FixedLenFeature
from tensorflow.train import BytesList
from tensorflow.train import Example
from tensorflow.train import Features
from tensorflow.train import Feature
from tensorflow.data import TFRecordDataset
import tensorflow as tf
import os
```

从**行** **5** **-19** ，我们导入所有需要的包。

```py
# a single instance of structured data will consist of an image and its
# corresponding class name
imagePath = get_file(
	config.IMAGE_FNAME,
	config.IMAGE_URL,
)
image = utils.load_image(pathToImage=imagePath)
class_name = config.IMAGE_CLASS

# check to see if the output folder exists, if not, build the output
# folder
if not os.path.exists(config.OUTPUT_PATH):
	os.makedirs(config.OUTPUT_PATH)

# save the resized image
utils.save_image(image=image, saveImagePath=config.RESIZED_IMAGE_PATH)

# build the image and the class name feature
imageFeature = Feature(
	bytes_list=BytesList(value=[
		# notice how we serialize the image
		serialize_tensor(image).numpy(),
	])
)
classNameFeature = Feature(
	bytes_list=BytesList(value=[
		class_name.encode(),
	])
)

# wrap the image and class feature with a features dictionary and then
# wrap the features into an example
features = Features(feature={
	"image": imageFeature,
	"class_name": classNameFeature,
})
example = Example(features=features)
```

在**的第 23-26 行**，我们从一个特定的 url 下载一个图像，并将图像保存到磁盘。接下来，在**的第 27 行**，我们使用`load_image`函数从磁盘加载图像作为`tf.Tensor`。最后，**第 28 行**指定了图像的类名。

图像和类名将作为我们的单个实例数据。我们现在需要将它们序列化并保存为单独的`Feature`。**第 39-49 行**负责序列化过程，并将图像和类名包装为`Feature`。

现在我们有了自己的个体`Feature`，我们需要将它包装成一个名为`Features`的集合。**第 53-56 行**构建一个`Features`，由一个`Feature`的字典组成。最后，**第 57 行**通过将`Features`包装成一个`Example`结束了我们的旅程。

```py
# serialize the entire example
serializedExample = example.SerializeToString()

# write the serialized example into a TFRecord
with TFRecordWriter(config.TFRECORD_EXAMPLE_FNAME) as recordWriter:
	recordWriter.write(serializedExample)

# build the feature schema and the TFRecord dataset
featureSchema = {
	"image": FixedLenFeature([], dtype=tf.string),
	"class_name": FixedLenFeature([], dtype=tf.string),
}
dataset = TFRecordDataset(config.TFRECORD_EXAMPLE_FNAME)

# iterate over the dataset
for element in dataset:
	# get the serialized example and parse it with the feature schema
	element = parse_example(element, featureSchema)

	# grab the serialized class name and the image
	className = element["class_name"].numpy().decode()
	image = parse_tensor(
		element["image"].numpy(),
		out_type=tf.dtypes.float32
	)

	# save the de-serialized image along with the class name
	utils.save_image(
		image=image,
		saveImagePath=config.DESERIALIZED_IMAGE_PATH,
		title=className
	)
```

在**第 60 行**上，我们可以使用`SerializeToString`函数直接序列化`Example`。接下来，我们从第 63 行和第 64 行的**上的序列化示例中直接构建 TFRecord。**

现在我们在第 67-70 行的**处构建一个特征示意图。该示意图将用于解析每个示例。**

如前所述，使用 TFRecords 构建`tf.data.Dataset`非常简单。在**第 71 行**，我们使用简单的 API `TFRecordDataset`构建数据集。

在第 74-90 行上，我们迭代数据集。**第 76 行**用于解析数据集的每个元素。请注意，我们在这里是如何使用特征原理图来解析示例的。在**的第 79-83 行**，我们获取了类名和图像的反序列化状态。最后，我们将图像保存到磁盘的第 86-90 行中。

* * *

### [**高级 TFRecord 生成**](#TOC)

现在让我们看看如何生成高级 TFRecords。在本节中，我们将使用`tfds`(代表`tensorflow_datasets`，即用型数据集的集合)加载 div2k 数据集，对其进行预处理，然后将预处理后的数据集序列化为 TFRecords。

```py
# USAGE
# python create_tfrecords.py

# import the necessary packages
from pyimagesearch import config
from tensorflow.io import serialize_tensor
from tensorflow.io import TFRecordWriter
from tensorflow.train import BytesList
from tensorflow.train import Feature
from tensorflow.train import Features
from tensorflow.train import Example
import tensorflow_datasets as tfds
import tensorflow as tf
import os

# define AUTOTUNE object
AUTO = tf.data.AUTOTUNE
```

从**第 5-14 行**，我们导入所有必需的包，包括我们的配置文件、tensorflow 数据集集合，以及将数据集序列化为 TFrecords 所需的其他 TensorFlow 子模块。在第 17 行的**上，我们定义了`tf.data.AUTOTUNE`用于优化目的。**

```py
def pre_process(element):
	# grab the low and high resolution images
	lrImage = element["lr"]
	hrImage = element["hr"]

	# convert the low and high resolution images from tensor to
	# serialized TensorProto proto
	lrByte = serialize_tensor(lrImage)
	hrByte = serialize_tensor(hrImage)

	# return the low and high resolution proto objects
	return (lrByte, hrByte)
```

在**的第 19-30 行**，我们定义了我们的`pre_process`函数，它接受一个由低分辨率和高分辨率图像组成的元素作为输入。在**第 21 和 22 行**，我们抓取低分辨率和高分辨率图像。在**第 26 行和第 27 行**，我们将低分辨率和高分辨率图像从 tensors 转换为序列化 TensorProto 类型。最后，在**第 30 行**，我们返回低分辨率和高分辨率图像。

```py
def create_dataset(dataDir, split, shardSize):
	# load the dataset, save it to disk, and preprocess it
	ds = tfds.load(config.DATASET, split=split, data_dir=dataDir)
	ds = (ds
		.map(pre_process, num_parallel_calls=AUTO)
		.batch(shardSize)
	)

	# return the dataset
	return ds
```

在**的第 32-41 行**，我们定义了我们的`create_dataset`函数，它将存储数据集、数据集分割和碎片大小的目录路径作为输入。在**第 34 行**，我们加载 div2k 数据集并将其存储在磁盘上。在**第 35-38 行**，我们预处理数据集和批量大小。最后，在**第 41 行**，我们返回 TensorFlow 数据集对象。

```py
def create_serialized_example(lrByte, hrByte):
	# create low and high resolution image byte list
	lrBytesList = BytesList(value=[lrByte])
	hrBytesList = BytesList(value=[hrByte])

	# build low and high resolution image feature from the byte list
	lrFeature = Feature(bytes_list=lrBytesList)
	hrFeature = Feature(bytes_list=hrBytesList)

	# build a low and high resolution image feature map
	featureMap = {
		"lr": lrFeature,
		"hr": hrFeature,
	}

	# build a collection of features, followed by building example
	# from features, and serializing the example
	features = Features(feature=featureMap)
	example = Example(features=features)
	serializedExample = example.SerializeToString()

	# return the serialized example
	return serializedExample
```

在**的第 43-65 行**，我们定义了我们的`create_serialized_example`函数，它以字节形式接受低分辨率和高分辨率图像作为输入。在**第 45 和 46 行**，我们创建了低分辨率和高分辨率图像字节列表对象。在**第 49-56 行**上，我们从字节列表构建低分辨率和高分辨率图像特征，并从低分辨率和高分辨率图像特征构建后续图像特征图。在**第 60-62 行**，我们从特征映射中构建一个特征集合，然后从特征中构建一个例子并序列化这个例子。最后，在**第 65 行**上，我们返回序列化的例子。

```py
def prepare_tfrecords(dataset, outputDir, name, printEvery=50):
	# check whether output directory exists
	if not os.path.exists(outputDir):
		os.makedirs(outputDir)

	# loop over the dataset and create TFRecords
	for (index, images) in enumerate(dataset):
		# get the shard size and build the filename
		shardSize = images[0].numpy().shape[0]
		tfrecName = f"{index:02d}-{shardSize}.tfrec"
		filename = outputDir + f"/{name}-" + tfrecName

		# write to the tfrecords
		with TFRecordWriter(filename) as outFile:
			# write shard size serialized examples to each TFRecord
			for i in range(shardSize):
				serializedExample = create_serialized_example(
					images[0].numpy()[i], images[1].numpy()[i])
				outFile.write(serializedExample)

			# print the progress to the user
			if index % printEvery == 0:
				print("[INFO] wrote file {} containing {} records..."
				.format(filename, shardSize))
```

在**第 67-90 行**上，我们定义了`prepare_tfrecords`函数，它主要以 TensorFlow 数据集和输出目录路径作为输入。在**第 69 行和第 70 行**，我们检查输出目录是否存在，如果不存在，我们就创建它。在第 73 行**，我们开始遍历数据集，抓取索引和图像。在第 75-77 行**上，我们设置了分片大小、输出 TFRecord 名称和输出 TFrecord 的路径。在**第 80-90 行**，我们打开一个空的 TFRecord，并开始向其中写入序列化的示例。

```py
# create training and validation dataset of the div2k images
print("[INFO] creating div2k training and testing dataset...")
trainDs = create_dataset(dataDir=config.DIV2K_PATH, split="train",
	shardSize=config.SHARD_SIZE)
testDs = create_dataset(dataDir=config.DIV2K_PATH, split="validation",
	shardSize=config.SHARD_SIZE)

# create training and testing TFRecords and write them to disk
print("[INFO] preparing and writing div2k TFRecords to disk...")
prepare_tfrecords(dataset=trainDs, name="train",
	outputDir=config.GPU_DIV2K_TFR_TRAIN_PATH)
prepare_tfrecords(dataset=testDs, name="test",
	outputDir=config.GPU_DIV2K_TFR_TEST_PATH)
```

在**第 93-97 行**，我们创建了 div2k 训练和测试 TensorFlow 数据集。从**第 100-104 行**，我们开始调用`prepare_tfrecords`函数来创建将保存在磁盘上的训练和测试 TFRecords。

* * *

* * *

## [**汇总**](#TOC)

在本教程中，您学习了什么是 TFRecords，以及如何使用 TensorFlow 生成 TFRecords 来训练深度神经网络。

我们首先从 TFRecords 的基础知识开始，学习如何使用它们来序列化数据。接下来，我们学习了如何使用 TFRecords 预处理和序列化像 div2k 这样的大型数据集。

TFRecord 格式的两个主要优点是，它帮助我们高效地存储数据集，并且我们获得了比从磁盘读取原始数据更快的 I/O 速度。

当我们用 TPU 训练深度神经网络时，TFRecords 是极其有益的。如果你对此感兴趣，那么一定要看看 [SRGAN](https://pyimg.co/lgnrx) 和 [ESRGAN](https://pyimg.co/jt2cb) 教程，它们涵盖了如何使用张量处理单元(TPU)和图形处理单元(GPU)来训练深度神经网络。

* * *

### [**参考**](#TOC)

对我们有帮助的所有教程列表:

*   [https://www . ka ggle . com/code/ryanholbrook/TF records-basics/notebook](https://www.kaggle.com/code/ryanholbrook/tfrecords-basics/notebook)
*   [https://www.tensorflow.org/tutorials/load_data/tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord)

* * *

### [**引用信息**](#TOC)

A. R. Gosthipaty 和 A. Thanki。“TF records 简介”， *PyImageSearch* ，D. Chakraborty，P. Chugh，S. Huot，K. Kidriavsteva，R. Raha 编辑。，2022 年，【https://pyimg.co/s5p1b 

```py
@incollection{ARG-AT_2022_TFRecords,
  author = {Aritra Roy Gosthipaty and Abhishek Thanki},
  title = {Introduction to TFRecords},
  booktitle = {PyImageSearch},
  editor = {Devjyoti Chakraborty and Puneet Chugh and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha},
  year = {2022},
  note = {https://pyimg.co/s5p1b},
}
```

* * *

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，** ***只需在下面的表格中输入您的电子邮件地址！****