# 使用 tf.data 和 TensorFlow 进行数据扩充

> 原文：<https://pyimagesearch.com/2021/06/28/data-augmentation-with-tf-data-and-tensorflow/>

在本教程中，您将学习使用 Keras 和 TensorFlow 将数据扩充合并到您的`tf.data`管道中的两种方法。

本教程是我们关于`tf.data`模块的三部分系列中的一部分:

1.  [*温柔介绍 TF . data*](https://pyimagesearch.com/2021/06/14/a-gentle-introduction-to-tf-data-with-tensorflow/)
2.  [*带有 tf.data 和 TensorFlow 的数据管道*](https://pyimagesearch.com/2021/06/21/data-pipelines-with-tf-data-and-tensorflow/)
3.  *用 tf.data 进行数据扩充*(今天的教程)

在这个系列中，我们已经发现了`tf.data`模块对于构建数据处理管道是多么的快速和高效。一旦建成，这些管道可以训练你的神经网络*比使用标准方法*快得多。

然而，我们还没有讨论的一个问题是:

> *我们如何在 tf.data 管道中应用数据增强？*

数据扩充是训练神经网络的一个重要方面，这些神经网络将部署在现实世界的场景中。通过应用数据扩充，我们可以提高我们的模型的能力，使其能够对未经训练的数据进行更好、更准确的预测。

TensorFlow 为我们提供了**两种方法**，我们可以使用这两种方法将数据增强应用于我们的`tf.data`管道:

1.  使用`Sequential`类和`preprocessing`模块构建一系列数据扩充操作，类似于 Keras 的`ImageDataGenerator`类
2.  应用`tf.image`功能手动创建数据扩充程序

第一种方法要简单得多，也更省力。第二种方法稍微复杂一些(通常是因为您需要阅读 TensorFlow 文档来找到您需要的确切函数)，但是允许对数据扩充过程进行更细粒度的控制。

在本教程中，你将学习如何使用*和`tf.data`两种*数据扩充程序。

**继续阅读，了解如何使用`tf.data`、**、**、*进行数据扩充。***

## **使用 tf.data 和 TensorFlow 进行数据扩充**

在本教程的第一部分中，我们将分解两种方法，您可以使用这两种方法通过`tf.data`处理管道进行数据扩充。

从那里，我们将配置我们的开发环境，并审查我们的项目目录结构。

今天我们将回顾两个 Python 脚本:

1.  第一个脚本将向您展示如何使用*层*应用数据扩充，而另一个脚本将使用 TensorFlow *操作*演示数据扩充
2.  我们的第二个脚本将使用数据增强和`tf.data`管道来训练一个深度神经网络

我们将讨论我们的结果来结束本教程。

### **用 tf.data 和 TensorFlow 进行数据扩充的两种方法**

本节介绍了使用 TensorFlow 和`tf.data`模块应用图像数据增强的两种方法。

### **使用层和“顺序”类的数据扩充**

通过使用 TensorFlow 的`preprocessing`模块和`Sequential`类，将数据扩充合并到`tf.data`管道中是最容易实现的。

**我们通常称这种方法为** ***【层数据扩充】*** **，因为我们用于数据扩充的`Sequential`类与我们用于实现顺序神经网络(例如 LeNet、VGGNet、AlexNet)的** ***类*** **相同。**

这个方法最好通过代码来解释:

```py
trainAug = Sequential([
	preprocessing.Rescaling(scale=1.0 / 255),
	preprocessing.RandomFlip("horizontal_and_vertical"),
	preprocessing.RandomZoom(
		height_factor=(-0.05, -0.15),
		width_factor=(-0.05, -0.15)),
	preprocessing.RandomRotation(0.3)
])
```

在这里，您可以看到我们正在构建一系列数据增强操作，包括:

*   随机水平和垂直翻转
*   随机缩放
*   随机旋转

然后，我们可以通过以下方式将数据扩充纳入我们的`tf.data`管道:

```py
trainDS = tf.data.Dataset.from_tensor_slices((trainX, trainLabels))
trainDS = (
	trainDS
	.shuffle(BATCH_SIZE * 100)
	.batch(BATCH_SIZE)
	.map(lambda x, y: (trainAug(x), y),
		 num_parallel_calls=tf.data.AUTOTUNE)
	.prefetch(tf.data.AUTOTUNE)
)
```

注意我们如何使用`map`函数在每一个输入图像上调用我们的`trainAug`管道。

在用`tf.data`应用数据增强时，我真的很喜欢这种方法。它非常容易使用，来自 Keras 的深度学习实践者会喜欢它与 Keras 的`ImageDataGenerator`类有多么相似。

另外，这些层还可以操作***内部的一个模型架构本身。如果你使用的是 GPU，这意味着 GPU 可以应用数据增强，而不是你的 CPU！**请注意，当使用只在您的 CPU 上运行的本地 TensorFlow 操作构建数据扩充时，这不是*而不是*的情况。*

 *### **使用张量流运算的数据增强**

我们可用于将数据扩充应用于`tf.data`管道的第二种方法是应用**张量流运算，**包括以下两种:

1.  图像处理功能*内置于`tf.image`模块内的*tensor flow 库中
2.  任何你想自己实现的自定义操作(使用 OpenCV、scikit-image、PIL/Pillow 等库。)

这种方法稍微复杂一点，因为它需要您手动实现数据扩充管道(相对于使用`preprocessing`模块中的类)，但是好处是您获得了更细粒度的控制(当然您可以实现任何您想要的定制操作)。

要使用 TensorFlow 操作应用数据扩充，我们首先需要定义一个接受输入图像的函数，然后应用我们的操作:

```py
def augment_using_ops(images, labels):
	images = tf.image.random_flip_left_right(images)
	images = tf.image.random_flip_up_down(images)
	images = tf.image.rot90(images)

	return (images, labels)
```

在这里，你可以看到我们是:

1.  随机水平翻转我们的图像
2.  随机垂直翻转图像
3.  应用随机的 90 度旋转

然后，将增强的图像返回给调用函数。

我们可以将这个数据扩充例程合并到我们的`tf.data`管道中，如下所示:

```py
ds = tf.data.Dataset.from_tensor_slices(imagePaths)
ds = (ds
	.shuffle(len(imagePaths), seed=42)
	.map(load_images, num_parallel_calls=AUTOTUNE)
	.cache()
	.batch(BATCH_SIZE)
	.map(augment_using_ops, num_parallel_calls=AUTOTUNE)
	.prefetch(tf.data.AUTOTUNE)
)
```

如您所见，这种数据扩充方法要求您对 TensorFlow 文档有更深入的了解，特别是`tf.image`模块，因为 TensorFlow 在那里实现其图像处理功能。

### 我应该对 tf.data 使用哪种数据扩充方法？

**对于大多数深度学习实践者来说，使用层和`Sequential`类来应用数据增强将会比** ***绰绰有余。***

TensorFlow 的`preprocessing`模块实现了您日常所需的绝大多数数据增强操作。

此外，`Sequential`类结合`preprocessing`模块更容易使用——熟悉 Keras 的`ImageDataGenerator`的深度学习实践者使用这种方法会感觉很舒服。

也就是说，如果您想要对您的数据扩充管道进行更细致的控制，或者如果您需要实施定制的数据扩充程序，您应该使用 TensorFlow 操作方法来应用数据扩充。

这个方法*需要更多的代码和 TensorFlow 文档知识(特别是`tf.image`中的函数)，但是如果你需要对你的数据扩充过程进行细粒度控制，你就不能打败这个方法。*

 *### **配置您的开发环境**

这篇关于 tf.keras 数据扩充的教程利用了 keras 和 TensorFlow。如果你打算遵循这个教程，我建议你花时间配置你的深度学习开发环境。

您可以利用这两个指南中的任何一个在您的系统上安装 TensorFlow 和 Keras:

*   [*如何在 Ubuntu 上安装 tensor flow 2*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

这两个教程都有助于在一个方便的 Python 虚拟环境中为您的系统配置这篇博客文章所需的所有软件。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **我们的示例数据集**

动物数据集中的图像属于三个不同的类别:狗、猫和熊猫，如图 4 中的**所示，每个类别有 1000 张示例图像。**

狗和猫的图像是从 [Kaggle 狗对猫挑战赛](http://pyimg.co/ogx37)中采集的，而熊猫的图像是从 [ImageNet 数据集](https://doi.org/10.1007/s11263-015-0816-y)中采集的。

我们的目标是训练一个卷积神经网络，可以正确识别这些物种中的每一个。

***注:*** *更多使用 Animals 数据集的例子，参考我的* [*k-NN 教程*](https://pyimagesearch.com/2021/04/17/your-first-image-classifier-using-k-nn-to-classify-images/) *和我的*[*Keras 入门教程*](https://pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/) *。*

### **项目结构**

在我们使用`tf.data`应用数据扩充之前，让我们首先检查我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索我们的 Python 脚本和示例数据集:

```py
$ tree . --dirsfirst --filelimit 10
.
├── dataset
│   └── animals
│       ├── cats [1000 entries exceeds filelimit, not opening dir]
│       ├── dogs [1000 entries exceeds filelimit, not opening dir]
│       └── panda [1000 entries exceeds filelimit, not opening dir]
├── data_aug_layers.png
├── data_aug_ops.png
├── load_and_visualize.py
├── no_data_aug.png
├── train_with_sequential_aug.py
└── training_plot.png

5 directories, 6 files
```

在`dataset/animals`目录中，我们有一个示例图像数据集，我们将对其应用数据扩充(我们在上一节中回顾了这个数据集)。

然后我们有两个 Python 脚本要实现:

1.  `load_and_visualize.py`:演示了如何使用(1)`Sequential`类和`preprocessing`模块以及(2) TensorFlow 操作来应用数据扩充。两种数据扩充程序的结果都将显示在我们的屏幕上，因此我们可以直观地验证该过程正在工作。
2.  `train_with_sequential_aug.py`:使用数据扩充和`tf.data`管道训练一个简单的 CNN。

运行这些脚本将产生以下输出:

*   `data_aug_layers.png`:应用层和`Sequential`类的数据扩充的输出
*   `data_aug_ops.png`:内置 TensorFlow 运算应用数据增强的输出可视化
*   `no_data_aug.png`:未应用*数据增强*的示例图像
*   `training_plot.png`:我们的培训和验证损失/准确性的图表

审查完我们的项目目录后，我们现在可以开始深入研究实现了！

### **用 tf.data 和 TensorFlow 实现数据扩充**

我们今天将在这里实现的第一个脚本将向您展示如何:

*   使用“层”和`Sequential`类执行数据扩充
*   使用内置张量流运算应用数据增强
*   或者，简单地说*而不是*应用数据扩充

使用`tf.data`，您将熟悉各种可用的数据扩充选项。然后，在本教程的后面，您将学习如何使用`tf.data`和数据增强来训练 CNN。

但是我们先走后跑吧。

打开项目目录结构中的`load_and_visualize.py`文件，让我们开始工作:

```py
# import the necessary packages
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.data import AUTOTUNE
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import os
```

**第 2-8 行**导入我们需要的 Python 包。**最重要的是，注意来自`layers.experimental`** 的`preprocessing`模块——`preprocessing`模块提供了我们使用 TensorFlow 的`Sequential`类执行数据扩充所需的函数。

虽然这个模块被称为`experimental`，但它已经在 TensorFlow API 中存在了近一年，所以可以肯定地说，这个模块绝不是“实验性的”(我想 TensorFlow 开发人员会在未来的某个时候重命名这个子模块)。

接下来，我们有我们的`load_images`函数:

```py
def load_images(imagePath):
	# read the image from disk, decode it, convert the data type to
	# floating point, and resize it
	image = tf.io.read_file(imagePath)
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	image = tf.image.resize(image, (156, 156))

	# parse the class label from the file path
	label = tf.strings.split(imagePath, os.path.sep)[-2]

	# return the image and the label
	return (image, label)
```

与本系列前面的教程一样，该函数负责:

1.  从磁盘加载我们的输入图像并预处理它(**第 13-16 行**)
2.  从文件路径中提取类标签(**行 19** )
3.  将`image`和`label`返回到调用函数(**第 22 行**

请注意，我们使用 TensorFlow 函数而不是 OpenCV 和 Python 函数来执行这些操作——我们使用 TensorFlow 函数，因此 TensorFlow 可以最大限度地优化我们的`tf.data`管道。

我们的下一个函数`augment_using_layers`，负责获取`Sequential`(使用`preprocessing`操作构建)的一个实例，然后应用它生成一组增强图像:

```py
def augment_using_layers(images, labels, aug):
	# pass a batch of images through our data augmentation pipeline
	# and return the augmented images
	images = aug(images)

	# return the image and the label
	return (images, labels)
```

我们的`augment_using_layers`函数接受三个必需的参数:

1.  数据批内的输入`images`
2.  对应的类`labels`
3.  我们的数据扩充(`aug`)对象，假设它是`Sequential`的一个实例

通过`aug`对象传递我们的输入`images`会导致随机扰动应用到图像上(**第 27 行**)。我们将在这个脚本的后面学习如何构造这个`aug`对象。

然后将增加的`images`和相应的`labels`返回给调用函数。

我们的最后一个函数`augment_using_ops`，使用`tf.image`模块中内置的 TensorFlow 函数应用数据扩充:

```py
def augment_using_ops(images, labels):
	# randomly flip the images horizontally, randomly flip the images
	# vertically, and rotate the images by 90 degrees in the counter
	# clockwise direction
	images = tf.image.random_flip_left_right(images)
	images = tf.image.random_flip_up_down(images)
	images = tf.image.rot90(images)

	# return the image and the label
	return (images, labels)
```

这个函数接受我们的数据批次`images`和`labels`。从这里开始，它适用于:

*   随机水平翻转
*   随机垂直翻转
*   90 度旋转(这实际上不是随机操作，而是与其他操作结合在一起，看起来像是随机操作)

同样，请注意，我们正在使用内置的 TensorFlow 函数构建这个数据扩充管道— **与使用`augment_using_layers`函数中的`Sequential`类和层方法相比，这种方法有什么优势？**

虽然前者实现起来要简单得多，但后者给了您对数据扩充管道更多的控制权。

根据您的数据扩充过程的先进程度，在`preprocessing`模块中可能没有实现您的管道。当这种情况发生时，您可以使用 TensorFlow 函数、OpenCV 方法和 NumPy 函数调用来实现您自己的定制方法。

从本质上来说，使用操作来应用数据扩充可以为您提供最细粒度的控制……但这也需要大量的工作，因为您需要(1)在 TensorFlow、OpenCV 或 NumPy 中找到适当的函数调用，或者(2)您需要手工实现这些方法。

现在我们已经实现了这两个函数，我们将看看如何使用它们来应用数据扩充。

让我们从解析命令行参数开始:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input images dataset")
ap.add_argument("-a", "--augment", type=bool, default=False,
	help="flag indicating whether or not augmentation will be applied")
ap.add_argument("-t", "--type", choices=["layers", "ops"], 
	help="method to be used to perform data augmentation")
args = vars(ap.parse_args())
```

我们有一个命令行参数，后跟两个可选参数:

1.  `--dataset`:我们要应用数据增强的图像的输入目录的路径
2.  `--augment`:一个布尔值，表示我们是否要对图像的输入目录应用数据扩充
3.  `--type`:我们将应用的数据增强类型(或者是`layers`或者是`ops`)

现在让我们为数据扩充准备我们的`tf.data`管道:

```py
# set the batch size
BATCH_SIZE = 8

# grabs all image paths
imagePaths = list(paths.list_images(args["dataset"]))

# build our dataset and data input pipeline
print("[INFO] loading the dataset...")
ds = tf.data.Dataset.from_tensor_slices(imagePaths)
ds = (ds
	.shuffle(len(imagePaths), seed=42)
	.map(load_images, num_parallel_calls=AUTOTUNE)
	.cache()
	.batch(BATCH_SIZE)
)
```

**第 54 行**设置我们的批量大小，而**第 57 行**获取我们的`--dataset`目录中所有输入图像的路径。

然后，我们开始在**线 61-67** 上建设`tf.data`管道，包括:

1.  打乱我们的图像
2.  在`imagePaths`列表中的每个输入图像上调用`load_images`
3.  缓存结果
4.  设置我们的批量

接下来，让我们检查是否应该应用数据扩充:

```py
# check if we should apply data augmentation
if args["augment"]:
	# check if we will be using layers to perform data augmentation
	if args["type"] == "layers":
		# initialize our sequential data augmentation pipeline
		aug = tf.keras.Sequential([
			preprocessing.RandomFlip("horizontal_and_vertical"),
			preprocessing.RandomZoom(
				height_factor=(-0.05, -0.15),
				width_factor=(-0.05, -0.15)),
			preprocessing.RandomRotation(0.3)
		])

		# add data augmentation to our data input pipeline
		ds = (ds
			.map(lambda x, y: augment_using_layers(x, y, aug),
				num_parallel_calls=AUTOTUNE)
		)

	# otherwise, we will be using TensorFlow image operations to
	# perform data augmentation
	else:
		# add data augmentation to our data input pipeline
		ds = (ds
			.map(augment_using_ops, num_parallel_calls=AUTOTUNE)
		)
```

**第 70 行**检查我们的`--augment`命令行参数是否表明我们是否应该应用数据扩充。

提供检查通道，**第 72 行**检查我们是否应用了**层/顺序**数据扩充。

使用`preprocessing`模块和`Sequential`类应用数据扩充在**行 74-80** 完成。顾名思义`Sequential`，你可以看到我们正在应用随机水平/垂直翻转、随机缩放和随机旋转，*一次一个，一个操作后面跟着下一个*(因此得名“顺序”)。

如果您以前使用过 Keras 和 TensorFlow，那么您会知道`Sequential`类也用于构建简单的神经网络，其中一个操作进入下一个操作。类似地，我们可以使用`Sequential`类构建一个数据扩充管道，其中一个扩充函数调用的输出是下一个的输入。

Keras 的`ImageDataGenerator`功能的用户在这里会有宾至如归的感觉，因为使用`preprocessing`和`Sequential`应用数据增强非常相似。

然后我们将我们的`aug`对象添加到**行 83-86** 上的`tf.data`管道中。注意我们如何将`map`函数与`lambda`函数一起使用，需要两个参数:

1.  `x`:输入图像
2.  `y`:图像的类别标签

然后，`augment_using_layers`函数应用实际的数据扩充。

否则，**第 90-94 行**处理我们使用**张量流运算执行数据扩充时的情况。**我们需要做的就是更新`tf.data`管道，为每批数据调用`augment_using_ops`。

现在，让我们最终确定我们的`tf.data`渠道:

```py
# complete our data input pipeline
ds = (ds
	.prefetch(AUTOTUNE)
)

# grab a batch of data from our dataset
batch = next(iter(ds))
```

用`AUTOTONE`参数调用`prefetch`优化了我们整个`tf.data`管道。

然后，我们使用我们的数据管道在**行 102 上生成数据`batch`(如果设置了`--augment`命令行参数，则可能应用了数据扩充)。**

这里的最后一步是可视化我们的输出:

```py
# initialize a figure
print("[INFO] visualizing the first batch of the dataset...")
title = "With data augmentation {}".format(
	"applied ({})".format(args["type"]) if args["augment"] else \
	"not applied")
fig = plt.figure(figsize=(BATCH_SIZE, BATCH_SIZE))
fig.suptitle(title)

# loop over the batch size
for i in range(0, BATCH_SIZE):
	# grab the image and label from the batch
	(image, label) = (batch[0][i], batch[1][i])

	# create a subplot and plot the image and label
	ax = plt.subplot(2, 4, i + 1)
	plt.imshow(image.numpy())
	plt.title(label.numpy().decode("UTF-8"))
	plt.axis("off")

# show the plot
plt.tight_layout()
plt.show()
```

**第 106-110 行**初始化一个 matplotlib 图形来显示我们的输出结果。

然后我们循环遍历`batch` ( **行 113** )内的每个图像/类标签，并继续:

*   提取图像和标签
*   在图上显示图像
*   将类别标签添加到图中

由此产生的图然后显示在我们的屏幕上。

### **用 tf.data 结果进行数据扩充**

我们现在已经准备好用`tf.data`来可视化应用数据扩充的输出！

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

从那里，执行以下命令:

```py
$ python load_and_visualize.py --dataset dataset/animals
[INFO] loading the dataset...
[INFO] visualizing the first batch of the dataset...
```

如你所见，我们在这里执行了**无数据扩充**——我们只是在屏幕上加载一组示例图像并显示它们。这个输出作为我们的基线，我们可以将接下来的两个输出进行比较。

**现在，让我们使用“层”方法**(即`preprocessing`模块和`Sequential`类)来应用数据扩充。

```py
$ python load_and_visualize.py --dataset dataset/animals \
	--aug 1 --type layers
[INFO] loading the dataset...
[INFO] visualizing the first batch of the dataset...
```

请注意我们是如何成功地将数据扩充应用于输入图像的——每个输入图像都被随机翻转、缩放和旋转。

**最后，让我们检查用于数据扩充的 TensorFlow 操作方法的输出**(即手动定义管道函数) **:**

```py
$ python load_and_visualize.py --dataset dataset/animals \
	--aug 1 --type ops
[INFO] loading the dataset...
[INFO] visualizing the first batch of the dataset...
```

我们的输出非常类似于**图** 5，从而证明我们已经能够成功地将数据增强整合到我们的`tf.data`管道中。

### **使用 tf.data 实施我们的数据扩充培训脚本**

在上一节中，我们学习了如何使用`tf.data`构建数据扩充管道；然而，我们*没有*使用我们的管道训练神经网络。本节将解决这个问题。

本教程结束时，您将能够开始将数据扩充应用到您自己的`tf.data`管道中。

在您的项目目录结构中打开`train_with_sequential.py`脚本，让我们开始工作:

```py
# import the necessary packages
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
```

**第 2-11 行**导入我们需要的 Python 包。我们将使用`Sequential`类来:

1.  建立一个简单、浅显的 CNN
2.  用`preprocessing`模块构建我们的数据扩充管道

然后，我们将在 CIFAR-10 数据集*上训练我们的 CNN，应用*数据增强。

接下来，我们有命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="training_plot.png",
	help="path to output training history plot")
args = vars(ap.parse_args())
```

确实只有一个参数是`--plot`，它是我们输出训练历史图的路径。

接下来，我们继续设置超参数并加载 CIFAR-10 数据集。

```py
# define training hyperparameters
BATCH_SIZE = 64
EPOCHS = 50

# load the CIFAR-10 dataset
print("[INFO] loading training data...")
((trainX, trainLabels), (textX, testLabels)) = cifar10.load_data()
```

现在我们已经准备好建立我们的数据扩充程序:

```py
# initialize our sequential data augmentation pipeline for training
trainAug = Sequential([
	preprocessing.Rescaling(scale=1.0 / 255),
	preprocessing.RandomFlip("horizontal_and_vertical"),
	preprocessing.RandomZoom(
		height_factor=(-0.05, -0.15),
		width_factor=(-0.05, -0.15)),
	preprocessing.RandomRotation(0.3)
])

# initialize a second data augmentation pipeline for testing (this
# one will only do pixel intensity rescaling
testAug = Sequential([
	preprocessing.Rescaling(scale=1.0 / 255)
])
```

**第 28-35 行**初始化我们的`trainAug`序列，包括:

1.  将我们的像素强度从范围*【0，255】*重新调整到*【0，1】*
2.  执行随机水平和垂直翻转
3.  随机缩放
4.  应用随机旋转

除了`Rescaling`之外，所有这些操作都是随机的，它只是我们构建到`Sequential`管道中的一个基本预处理操作。

然后我们在第 39-41 行的**中定义我们的`testAug`过程。**我们在这里执行的唯一操作是我们的*【0，1】*像素强度缩放。**我们** ***必须*** **在这里使用`Rescaling`类，因为我们的训练数据也被重新调整到范围*****【0，1】***——否则在训练我们的网络时会导致错误的输出。

处理好预处理和增强初始化后，让我们为我们的训练和测试数据构建一个`tf.data`管道:

```py
# prepare the training data pipeline (notice how the augmentation
# layers have been mapped)
trainDS = tf.data.Dataset.from_tensor_slices((trainX, trainLabels))
trainDS = (
	trainDS
	.shuffle(BATCH_SIZE * 100)
	.batch(BATCH_SIZE)
	.map(lambda x, y: (trainAug(x), y),
		 num_parallel_calls=tf.data.AUTOTUNE)
	.prefetch(tf.data.AUTOTUNE)
)

# create our testing data pipeline (notice this time that we are
# *not* apply data augmentation)
testDS = tf.data.Dataset.from_tensor_slices((textX, testLabels))
testDS = (
	testDS
	.batch(BATCH_SIZE)
	.map(lambda x, y: (testAug(x), y),
		num_parallel_calls=tf.data.AUTOTUNE)
	.prefetch(tf.data.AUTOTUNE)
)
```

**第 45-53 行**构建我们的训练数据集，包括洗牌、创建批处理和应用`trainAug`函数。

**第 57-64 行**对我们的测试集执行类似的操作，除了两个例外:

1.  我们不需要打乱数据进行评估
2.  我们的`testAug`对象*只有*执行重新缩放，而*没有*随机扰动

现在让我们实现一个基本的 CNN:

```py
# initialize the model as a super basic CNN with only a single CONV
# and RELU layer, followed by a FC and soft-max classifier
print("[INFO] initializing model...")
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same",
	input_shape=(32, 32, 3)))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation("softmax"))
```

这个 CNN 非常简单，仅由一个 CONV 层、一个 RELU 激活层、一个 FC 层和我们的 softmax 分类器组成。

然后我们继续使用我们的`tf.data`管道训练我们的 CNN:

```py
# compile the model
print("[INFO] compiling model...")
model.compile(loss="sparse_categorical_crossentropy",
	optimizer="sgd", metrics=["accuracy"])

# train the model
print("[INFO] training model...")
H = model.fit(
	trainDS,
	validation_data=testDS,
	epochs=EPOCHS)

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(testDS)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
```

一个简单的调用`model.fit`通过我们的`trainDS`和`testDS`使用我们的`tf.data`管道训练我们的模型，并应用数据增强。

训练完成后，我们在测试集上评估模型的性能。

我们的最终任务是生成一个训练历史图:

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

然后将结果图保存到磁盘上通过`--plot`命令行参数提供的文件路径中。

### **数据扩充和 tf.data 的训练结果**

我们现在准备通过`tf.data`管道使用数据增强来训练一个深度神经网络。

请务必访问本教程的 ***“下载”*** 部分来检索源代码。

从那里，您可以执行培训脚本:

```py
$ python train_with_sequential_aug.py
[INFO] loading training data...
[INFO] initializing model...
[INFO] compiling model...
[INFO] training model...
Epoch 1/50
782/782 [==============================] - 12s 12ms/step - loss: 2.1271 - accuracy: 0.2212 - val_loss: 2.0173 - val_accuracy: 0.2626
Epoch 2/50
782/782 [==============================] - 10s 13ms/step - loss: 1.9319 - accuracy: 0.3154 - val_loss: 1.9148 - val_accuracy: 0.3104
Epoch 3/50
782/782 [==============================] - 10s 13ms/step - loss: 1.8720 - accuracy: 0.3338 - val_loss: 1.8430 - val_accuracy: 0.3403
Epoch 4/50
782/782 [==============================] - 10s 13ms/step - loss: 1.8333 - accuracy: 0.3515 - val_loss: 1.8326 - val_accuracy: 0.3483
Epoch 5/50
782/782 [==============================] - 10s 13ms/step - loss: 1.8064 - accuracy: 0.3554 - val_loss: 1.9409 - val_accuracy: 0.3246
...
Epoch 45/50
782/782 [==============================] - 10s 13ms/step - loss: 1.5882 - accuracy: 0.4379 - val_loss: 1.7483 - val_accuracy: 0.3860
Epoch 46/50
782/782 [==============================] - 10s 13ms/step - loss: 1.5800 - accuracy: 0.4380 - val_loss: 1.6637 - val_accuracy: 0.4110
Epoch 47/50
782/782 [==============================] - 10s 13ms/step - loss: 1.5851 - accuracy: 0.4357 - val_loss: 1.7648 - val_accuracy: 0.3834
Epoch 48/50
782/782 [==============================] - 10s 13ms/step - loss: 1.5823 - accuracy: 0.4371 - val_loss: 1.7195 - val_accuracy: 0.4054
Epoch 49/50
782/782 [==============================] - 10s 13ms/step - loss: 1.5812 - accuracy: 0.4388 - val_loss: 1.6914 - val_accuracy: 0.4045
Epoch 50/50
782/782 [==============================] - 10s 13ms/step - loss: 1.5785 - accuracy: 0.4381 - val_loss: 1.7291 - val_accuracy: 0.3937
157/157 [==============================] - 0s 2ms/step - loss: 1.7291 - accuracy: 0.3937
[INFO] accuracy: 39.37%
```

由于我们的*非常*浅的神经网络(只有一个 CONV 层，后面跟着一个 FC 层)，我们在测试集上只获得了 39%的准确率——**准确率是** ***而不是*** **这是我们输出的重要收获。**

**相反，这里的关键要点是，我们已经能够成功地将数据增强应用到我们的培训渠道中。**你可以在*中替换任何你想要的*神经网络架构，我们的`tf.data`管道将*自动*将数据增强整合到其中。

作为对你的一个练习，我建议换掉我们超级简单的 CNN，尝试用诸如 [LeNet](https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/) 、 [MiniVGGNet](https://pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/) 或 [ResNet](https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/) 之类的架构来代替它。无论您选择哪种架构，我们的`tf.data`管道都将能够应用数据扩充，而无需您添加任何额外的代码(更重要的是，这个数据管道将比依赖旧的`ImageDataGenerator`类*高效得多*)。

## **总结**

在本教程中，您学习了如何使用`tf.data`和 TensorFlow 执行数据扩充。

使用`tf.data`应用数据扩充有两种方法:

1.  使用`preprocessing`模块和`Sequential`类进行数据扩充
2.  在`tf.image`模块中使用 TensorFlow 操作进行数据扩充(以及您想要使用 OpenCV、scikit-image 等实现的任何其他自定义图像处理例程。)

**对于大多数深度学习实践者来说，第一种方法就足够了。**最流行的数据扩充操作是已经在`preprocessing`模块中实现的*。*

类似地，使用`Sequential`类是在彼此之上应用一系列数据扩充操作的自然方式。Keras 的`ImageDataGenerator`类用户使用这种方法会有宾至如归的感觉。

**第二种方法主要针对那些需要对其数据增强管道进行更细粒度控制的深度学习实践者。**这种方法允许你利用`tf.image`中的图像处理功能以及你想使用的任何其他计算机视觉/图像处理库，包括 OpenCV、scikit-image、PIL/Pillow 等。

本质上，只要你能把你的图像作为一个 NumPy 数组处理，并作为一个张量返回，第二种方法对你来说是公平的。

***注:*** *如果只使用原生张量流运算，可以避免中间的 NumPy 数组表示，直接对张量流张量进行运算，这样会导致更快的增强。*

也就是说，从第一种方法开始——`preprocessing`模块和`Sequential`类是一种非常自然、易于使用的方法，用于通过`tf.data`应用数据扩充。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******