# 图像超分辨率

> 原文：<https://pyimagesearch.com/2022/02/14/image-super-resolution/>

在本教程中，您将学习使用图像超分辨率。

本课是关于超分辨率的 3 部分系列的一部分:

1.  [*OpenCV 超分辨率与深度学习*](https://pyimagesearch.com/2020/11/09/opencv-super-resolution-with-deep-learning/)
2.  *图像超分辨率*(本教程)
3.  [*用 TensorFlow、Keras、深度学习实现像素洗牌超分辨率*](https://pyimagesearch.com/2021/09/27/pixel-shuffle-super-resolution-with-tensorflow-keras-and-deep-learning/)

**要了解如何**使用图像超分辨率**，*继续阅读。***

## **图像超分辨率**

正如深度学习和卷积神经网络已经完全改变了通过深度学习方法产生的艺术景观，超分辨率算法也是如此。然而，值得注意的是，计算机视觉的超分辨率子领域已经得到了更严格的研究。以前的方法主要是基于示例的，并且倾向于:

1.  利用输入图像的内部相似性建立超分辨率输出([崔，2014](https://www.springerprofessional.de/en/deep-network-cascade-for-image-super-resolution/2245708))；[弗里德曼和法塔，2011](https://dl.acm.org/doi/10.1145/1944846.1944852)；[格拉斯纳等人，2009 年](https://www.wisdom.weizmann.ac.il/~vision/SingleImageSR.html)
2.  学习低分辨率到高分辨率的补丁([常等，2004](https://ieeexplore.ieee.org/document/1315043)；[弗里曼等人，2000 年](https://link.springer.com/article/10.1023/A:1026501619075)；【贾等著】2013 年
3.  使用稀疏编码的一些变体([杨等人，2010](https://ieeexplore.ieee.org/document/5466111) )

在本教程中，我们将实现[董等人(2016)](https://ieeexplore.ieee.org/document/7115171) 的工作。这种方法证明了以前的稀疏编码方法实际上等同于应用深度卷积神经网络——主要区别在于我们实现的方法更快，产生更好的结果，并且完全是端到端的。

虽然自 [Dong 等人在 2016](https://ieeexplore.ieee.org/document/7115171) 的工作(包括 [Johnson 等人(2016)](https://arxiv.org/abs/1603.08155) 关于将超分辨率框架化为风格转移的精彩论文)以来，已经有了许多超分辨率论文，但 Dong 等人的工作形成了许多其他超分辨率卷积神经网络(SRCNNs)建立的基础。

## **了解 SRCNNs**

SRCNNs 有许多重要的特征。下面列出了最重要的属性:

1.  **src nn 是全卷积**(不要和*全连接*混淆)。我们可以输入任何图像大小(假设宽度和高度可以平铺),并在其上运行 SRCNN。这使得 SRCNNs 非常快。
2.  我们是为了过滤器而训练，而不是为了准确性。在其他课程中，我们主要关注训练我们的 CNN，以在给定数据集上实现尽可能高的准确性。在这种情况下，我们关心的是 SRCNN 学习的实际*滤波器*，这将使我们能够放大图像——在训练数据集上获得的学习这些滤波器的实际精度是无关紧要的。
3.  **它们不需要解决使用优化问题。**在 SRCNN 已经学习了一组滤波器之后，它可以应用简单的正向传递来获得输出超分辨率图像。我们不必在每个图像的基础上优化损失函数来获得输出。
4.  它们完全是端到端的。同样，SRCNNs 完全是端到端的:向网络输入图像并获得更高分辨率的输出。没有中间步骤。一旦训练完成，我们就可以应用超分辨率了。

如上所述，我们的 SRCNN 的目标是学习一组过滤器，允许我们将低分辨率输入映射到高分辨率输出。因此，我们将构建两组图像补片，而不是实际的全分辨率图像:

1.  将作为网络输入的低分辨率面片
2.  将作为网络预测/重建的*目标*的高分辨率补丁

通过这种方式，我们的 SRCNN 将学习如何从低分辨率输入面片重建高分辨率面片。

实际上，这意味着我们:

1.  首先，需要构建低分辨率和高分辨率输入面片的数据集
2.  训练网络以学习将低分辨率补丁映射到它们的高分辨率对应物
3.  创建一个脚本，该脚本利用低分辨率图像的输入面片上的循环，通过网络传递它们，然后根据预测的面片创建输出高分辨率图像。

正如我们将在本教程后面看到的，构建 SRCNN 可以说比我们在其他课程中遇到的其他分类挑战更容易。

## **实施 src nn**

首先，我们将回顾这个项目的目录结构，包括任何需要的配置文件。接下来，我们将回顾一个用于构建低分辨率和高分辨率面片数据集的 Python 脚本。然后，我们将实现我们的 SRCNN 架构本身，并对其进行训练。最后，我们将利用训练好的模型将 SRCNNs 应用于输入图像。

### **配置您的开发环境**

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果您需要帮助配置 OpenCV 的开发环境，我们*强烈推荐*阅读我们的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让您启动并运行。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

我们首先需要回顾我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

### **开始项目**

让我们从回顾这个项目的目录结构开始。第一步是在`pyimagesearch`的`conv`子模块中创建一个名为`srcnn.py`的新文件——这就是我们的超分辨率卷积神经网络将要存在的地方:

```py
--- pyimagesearch
|    |--- __init__.py
...
|    |--- nn
|    |    |--- __init__.py
|    |    |--- conv
|    |    |    |--- __init__.py
|    |    |    |--- alexnet.py
|    |    |    |--- dcgan.py
...
|    |    |    |--- srcnn.py
...
```

从那里，我们有了项目本身的以下目录结构:

```py
--- super_resolution
|    |--- build_dataset.py
|    |--- config
|    |    |--- sr_config.py
|    |--- output/
|    |--- resize.py
|    |--- train.py
|    |--- uk_bench/
```

`sr_config.py`文件存储了我们需要的任何配置。然后我们将使用`build_dataset.py`来创建我们的低分辨率和高分辨率补丁用于训练。

***备注:*** [董等人](https://ieeexplore.ieee.org/document/7115171)将*面片*改为*子图*。这种澄清的尝试是为了避免关于面片的任何歧义(在一些计算机视觉文献中，这可能意味着重叠的 ROI)。我将交替使用这两个术语，因为我相信工作的上下文将定义一个补丁是否重叠——这两个术语都完全有效。**

从那里，我们有了实际训练我们网络的`train.py`脚本。最后，我们将实现`resize.py`来接受低分辨率输入图像并创建高分辨率输出。

`output`目录将存储

*   我们的 HDF5 训练图像集
*   输出模型本身

最后，`uk_bench`目录将包含我们正在学习模式的示例图像。

现在让我们来看看`sr_config.py`文件:

```py
# import the necessary packages
import os

# define the path to the input images we will be using to build the
# training crops
INPUT_IMAGES = "ukbench100"

# define the path to the temporary output directories
BASE_OUTPUT = "output"
IMAGES = os.path.sep.join([BASE_OUTPUT, "images"])
LABELS = os.path.sep.join([BASE_OUTPUT, "labels"])

# define the path to the HDF5 files
INPUTS_DB = os.path.sep.join([BASE_OUTPUT, "inputs.hdf5"])
OUTPUTS_DB = os.path.sep.join([BASE_OUTPUT, "outputs.hdf5"])

# define the path to the output model file and the plot file
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "srcnn.model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
```

**第 6 行**定义了`ukbench100`数据集的路径，它是更大的 [UKBench 数据集](https://archive.org/details/ukbench)的子集。 [Dong 等人](https://ieeexplore.ieee.org/document/7115171)对 91 幅图像数据集和 120 万幅 ImageNet 数据集进行了实验。复制在 ImageNet 上训练的 SRCNN 的工作超出了本课的范围，因此我们将坚持使用更接近 Dong 等人的第一个实验大小的图像数据集。

**第 9-11 行**构建了到临时输出目录的路径，我们将在那里存储我们的低分辨率和高分辨率子图像。给定低分辨率和高分辨率子图像，我们将从它们生成输出 HDF5 数据集(**第 14 行和第 15 行**)。**第 18 行和第 19 行**定义了输出模型文件的路径以及一个训练图。

让我们继续定义我们的配置:

```py
# initialize the batch size and number of epochs for training
BATCH_SIZE = 128
NUM_EPOCHS = 10

# initialize the scale (the factor in which we want to learn how to
# enlarge images by) along with the input width and height dimensions
# to our SRCNN
SCALE = 2.0
INPUT_DIM = 33

# the label size should be the output spatial dimensions of the SRCNN
# while our padding ensures we properly crop the label ROI
LABEL_SIZE = 21
PAD = int((INPUT_DIM - LABEL_SIZE) / 2.0)

# the stride controls the step size of our sliding window
STRIDE = 14
```

按照`NUM_EPOCHS`的定义，我们将只训练十个纪元。[董等人](https://ieeexplore.ieee.org/document/7115171)发现，长时间的训练实际上*会损害*的性能(这里的“性能”定义为输出超分辨率图像的质量)。十个历元应该足以让我们的 SRCNN 学习一组过滤器，以将我们的低分辨率补丁映射到它们的更高分辨率对应物。

`SCALE` ( **第 28 行**)定义了我们放大图像的因子——这里我们放大了`2x`，但你也可以放大`3x`或`4x`。

`INPUT_DIM`是我们的子窗口的空间宽度和高度(`33×33`像素)。我们的`LABEL_SIZE`是 SRCNN 的输出空间维度，而我们的`PAD`确保我们在构建数据集和对输入图像应用超分辨率时正确地裁剪标签 ROI。

最后，`STRIDE`在创建子图像时控制滑动窗口的步长。[董等人](https://ieeexplore.ieee.org/document/7115171)建议对于较小的图像数据集采用`14`像素的步距，对于较大的数据集采用`33`像素的步距。

在我们走得太远之前，你可能会困惑为什么`INPUT_DIM`比`LABEL_SIZE`大——这里的整个想法不就是构建更高分辨率的输出图像吗？当我们的神经网络的输出比输入的*小*时，这怎么可能呢？答案是双重的。

1.  在本课的前面，我们的 SRCNN 不包含零填充。使用零填充会引入边界伪影，这会降低输出图像的质量。由于我们没有使用零填充，我们的空间维度将在每一层之后自然减少。
2.  当将超分辨率应用于输入图像时(在训练之后)，我们实际上将通过因子`SCALE`增加*输入低分辨率图像的*——网络然后将较高比例的低分辨率图像转换为高分辨率输出图像。如果这个过程看起来很混乱，不要担心，本教程的剩余部分将有助于使它变得清晰。

### **构建数据集**

让我们继续为 SRCNN 构建我们的训练数据集。打开`build_dataset.py`并插入以下代码:

```py
# import the necessary packages
from pyimagesearch.io import HDF5DatasetWriter
from conf import sr_config as config
from imutils import paths
from PIL import Image
import numpy as np
import shutil
import random
import PIL
import cv2
import os

# if the output directories do not exist, create them
for p in [config.IMAGES, config.LABELS]:
	if not os.path.exists(p):
		os.makedirs(p)

# grab the image paths and initialize the total number of crops
# processed
print("[INFO] creating temporary images...")
imagePaths = list(paths.list_images(config.INPUT_IMAGES))
random.shuffle(imagePaths)
total = 0
```

**2-11 号线**办理我们的进口业务。注意我们是如何再次使用`HDF5DatasetWriter`类以 HDF5 格式将数据集写入磁盘的。我们的`sr_config`脚本被导入到**的第 3 行，**，这样我们就可以访问我们指定的值。

**第 14-16 行**创建临时输出目录来存储我们的子窗口。生成所有子窗口后，我们将它们添加到 HDF5 数据集，并删除临时目录。**第 21-23 行**然后获取输入图像的路径，并初始化一个计数器来计算生成的子窗口的数量`total`。

让我们遍历每一个图像路径:

```py
# loop over the image paths
for imagePath in imagePaths:
	# load the input image
	image = cv2.imread(imagePath)

	# grab the dimensions of the input image and crop the image such
	# that it tiles nicely when we generate the training data +
	# labels
	(h, w) = image.shape[:2]
	w -= int(w % config.SCALE)
	h -= int(h % config.SCALE)
	image = image[0:h, 0:w]

	# to generate our training images we first need to downscale the
	# image by the scale factor...and then upscale it back to the
	# original size -- this will process allows us to generate low
	# resolution inputs that we'll then learn to reconstruct the high
	# resolution versions from
	lowW = int(w * (1.0 / config.SCALE))
	lowH = int(h * (1.0 / config.SCALE))
	highW = int(lowW * (config.SCALE / 1.0))
	highH = int(lowH * (config.SCALE / 1.0))

	# perform the actual scaling
	scaled = np.array(Image.fromarray(image).resize((lowW, lowH),
		resample=PIL.Image.BICUBIC))
	scaled = np.array(Image.fromarray(scaled).resize((highW, highH),
		resample=PIL.Image.BICUBIC))
```

对于每个输入图像，我们首先从磁盘中加载它(**第 28 行**)，然后裁剪图像，使其在生成子窗口(**第 33-36 行**)时能很好地平铺。如果我们不采取这一步，我们的步幅大小将会不合适，我们将会在图像的空间维度之外裁剪出小块*。*

要为我们的 SRCNN 生成训练数据，我们需要:

1.  将原始输入图像缩小`SCALE`(对于`SCALE = 2.0`，我们将输入图像的尺寸减半)
2.  然后重新调整到原来的大小

该过程生成具有相同原始空间维度的低分辨率图像。我们将学习如何从这个低分辨率输入中*重建*一个高分辨率图像。

***备注:*** 我发现使用 OpenCV 的双线性插值(由[董等人](https://ieeexplore.ieee.org/document/7115171)推荐)产生的结果较差(它也引入了更多的代码来执行缩放)。我选择了 PIL/枕头的`.resize`功能，因为它更容易使用，效果也更好。

我们现在可以为输入和目标生成子窗口:

```py
	# slide a window from left-to-right and top-to-bottom
	for y in range(0, h - config.INPUT_DIM + 1, config.STRIDE):
		for x in range(0, w - config.INPUT_DIM + 1, config.STRIDE):
			# crop output the 'INPUT_DIM x INPUT_DIM' ROI from our
			# scaled image -- this ROI will serve as the input to our
			# network
			crop = scaled[y:y + config.INPUT_DIM,
				x:x + config.INPUT_DIM]

			# crop out the 'LABEL_SIZE x LABEL_SIZE' ROI from our
			# original image -- this ROI will be the target output
			# from our network
			target = image[
				y + config.PAD:y + config.PAD + config.LABEL_SIZE,
				x + config.PAD:x + config.PAD + config.LABEL_SIZE]

			# construct the crop and target output image paths
			cropPath = os.path.sep.join([config.IMAGES,
				"{}.png".format(total)])
			targetPath = os.path.sep.join([config.LABELS,
				"{}.png".format(total)])

			# write the images to disk
			cv2.imwrite(cropPath, crop)
			cv2.imwrite(targetPath, target)

			# increment the crop total
			total += 1
```

**第 55 行和第 56 行**在我们的图像上从左到右和从上到下滑动一个窗口。我们在第 60 行和第 61 行上裁剪`INPUT_DIM × INPUT_DIM`子窗口。这个`crop`就是从我们的`scaled`(即低分辨率图像)输入到我们神经网络的`33×33`。

我们还需要一个`target`供 SRCNN 预测(**第 66-68 行**)—`target`是 SRCNN 将试图重建的`LABEL_SIZE x LABEL_SIZE` ( `21×21`)输出。我们将`target`和`crop`写入磁盘的第 77 行和第 78 行。

最后一步是构建两个 HDF5 数据集，一个用于输入，另一个用于输出(即目标):

```py
# grab the paths to the images
print("[INFO] building HDF5 datasets...")
inputPaths = sorted(list(paths.list_images(config.IMAGES)))
outputPaths = sorted(list(paths.list_images(config.LABELS)))

# initialize the HDF5 datasets
inputWriter = HDF5DatasetWriter((len(inputPaths), config.INPUT_DIM,
	config.INPUT_DIM, 3), config.INPUTS_DB)
outputWriter = HDF5DatasetWriter((len(outputPaths),
	config.LABEL_SIZE, config.LABEL_SIZE, 3), config.OUTPUTS_DB)

# loop over the images
for (inputPath, outputPath) in zip(inputPaths, outputPaths):
	# load the two images and add them to their respective datasets
	inputImage = cv2.imread(inputPath)
	outputImage = cv2.imread(outputPath)
	inputWriter.add([inputImage], [-1])
	outputWriter.add([outputImage], [-1])

# close the HDF5 datasets
inputWriter.close()
outputWriter.close()

# delete the temporary output directories
print("[INFO] cleaning up...")
shutil.rmtree(config.IMAGES)
shutil.rmtree(config.LABELS)
```

重要的是要注意，类标签是不相关的(因此我指定了一个值`-1`)。“类标签”在技术上是输出子窗口，我们将尝试训练我们的 SRCNN 来重建它。我们将在`train.py`中编写一个定制的生成器来产生一个输入子窗口和目标输出子窗口的元组。

在我们的 HDF5 数据集生成之后，**行 108 和 109** 删除临时输出目录。

要生成数据集，请执行以下命令:

```py
$ python build_dataset.py 
[INFO] creating temporary images...
[INFO] building HDF5 datasets...
[INFO] cleaning up...
```

之后，您可以检查您的`BASE_OUTPUT`目录，找到`inputs.hdf5`和`outputs.hdf5`文件:

```py
$ ls ../datasets/ukbench/output/*.hdf5
inputs.hdf5		outputs.hdf5
```

### **Sr CNN 架构**

我们正在实现的 SRCNN 架构完全遵循[董等人](https://ieeexplore.ieee.org/document/7115171)的思路，使其易于实现。打开`srcnn.py`并插入以下代码:

```py
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K

class SRCNN:
	@staticmethod
	def build(width, height, depth):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# the entire SRCNN architecture consists of three CONV =>
		# RELU layers with *no* zero-padding
		model.add(Conv2D(64, (9, 9), kernel_initializer="he_normal",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(Conv2D(32, (1, 1), kernel_initializer="he_normal"))
		model.add(Activation("relu"))
		model.add(Conv2D(depth, (5, 5),
			kernel_initializer="he_normal"))
		model.add(Activation("relu"))

		# return the constructed network architecture
		return model
```

与其他架构相比，我们的 SRCNN 再简单不过了。整个架构仅由三层`CONV => RELU`组成，没有*零填充*(我们避免使用零填充，以确保我们不会在输出图像中引入任何边界伪影)。

我们的第一个`CONV`层学习`64`个滤镜，每个滤镜都是`9×9`。这个体积被送入第二个`CONV`层，在这里我们学习用于减少维度和学习局部特征的`32` `1×1`过滤器。最后的`CONV`层一共学习了`depth`个通道(对于 RGB 图像会是`3`，对于灰度会是`1`，每个通道都是`5×5`。

此网络体系结构有两个重要组成部分:

*   它小而紧凑，这意味着它将快速训练(记住，我们的目标不是在分类意义上获得更高的准确性——我们更感兴趣的是从网络中学习的*过滤器*，这将使我们能够执行超分辨率)。
*   它是完全卷积的，这使得它(1)再次更快，并且(2)我们可以接受任何输入图像大小，只要它能很好地平铺。

### **训练 SRCNN**

训练我们的 SRCNN 是一个相当简单的过程。打开`train.py`并插入以下代码:

```py
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from conf import sr_config as config
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import SRCNN
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
```

**2-11 号线**办理我们的进口业务。我们将需要`HDF5DatasetGenerator`来访问我们的序列化 HDF5 数据集以及我们的`SRCNN`实施。然而，需要稍加修改才能使用我们的 HDF5 数据集。

请记住，我们有两个 HDF5 数据集:输入子窗口和目标输出子窗口。我们的`HDF5DatasetGenerator`类旨在只处理一个 HDF5 文件，而不是两个。幸运的是，这很容易用我们的`super_res_generator`函数解决:

```py
def super_res_generator(inputDataGen, targetDataGen):
	# start an infinite loop for the training data
	while True:
		# grab the next input images and target outputs, discarding
		# the class labels (which are irrelevant)
		inputData = next(inputDataGen)[0]
		targetData = next(targetDataGen)[0]

		# yield a tuple of the input data and target data
		yield (inputData, targetData)
```

这个函数需要两个参数，`inputDataGen`和`targetDataGen`，它们都被假定为`HDF5DatasetGenerator`对象。

我们开始一个无限循环，它将继续循环我们在**行 15** 上的训练数据。对每个对象调用`next`(一个内置的 Python 函数，用于返回生成器中的下一项)会产生下一批集合。我们丢弃类标签(因为我们不需要它们)并返回一个由`inputData`和`targetData`组成的元组。

我们现在可以初始化我们的`HDF5DatasetGenerator`对象以及我们的模型和优化器:

```py
# initialize the input images and target output images generators
inputs = HDF5DatasetGenerator(config.INPUTS_DB, config.BATCH_SIZE)
targets = HDF5DatasetGenerator(config.OUTPUTS_DB, config.BATCH_SIZE)

# initialize the model and optimizer
print("[INFO] compiling model...")
opt = Adam(lr=0.001, decay=0.001 / config.NUM_EPOCHS)
model = SRCNN.build(width=config.INPUT_DIM, height=config.INPUT_DIM,
	depth=3)
model.compile(loss="mse", optimizer=opt)
```

当[董等人](https://ieeexplore.ieee.org/document/7115171)使用 RMSprop 时，我发现:

1.  使用`Adam`以较少的超参数调谐获得更好的结果
2.  一点点的学习速度衰减会产生更好、更稳定的训练

最后，注意我们将使用均方损失(MSE)而不是二元/分类交叉熵。

我们现在准备训练我们的模型:

```py
# train the model using our generators
H = model.fit_generator(
	super_res_generator(inputs.generator(), targets.generator()),
	steps_per_epoch=inputs.numImages // config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS, verbose=1)
```

我们的模型将被训练总共`NUM_EPOCHS` (10 个时期，根据我们的配置文件)。注意我们如何使用我们的`super_res_generator`分别从`inputs`和`targets`生成器联合产生训练批次。

我们的最后一个代码块处理将训练好的模型保存到磁盘、绘制损失以及关闭 HDF5 数据集:

```py
# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, config.NUM_EPOCHS), H.history["loss"],
	label="loss")
plt.title("Loss on super resolution training")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(config.PLOT_PATH)

# close the HDF5 datasets
inputs.close()
targets.close()
```

训练 SRCNN 架构就像执行以下命令一样简单:

```py
$ python train.py 
[INFO] compiling model...
Epoch 1/10
1100/1100 [==============================] - 14s - loss: 243.1207         
Epoch 2/10
1100/1100 [==============================] - 13s - loss: 59.0475      
...
Epoch 9/10
1100/1100 [==============================] - 13s - loss: 47.1672      
Epoch 10/10
1100/1100 [==============================] - 13s - loss: 44.7597      
[INFO] serializing model...
```

我们的模型现在已经训练好了，可以增加新输入图像的分辨率了！

### **使用 SRCNNs 提高图像分辨率**

我们现在准备实现`resize.py`，这个脚本负责从低分辨率输入图像构建高分辨率输出图像。打开`resize.py`并插入以下代码:

```py
# import the necessary packages
from conf import sr_config as config
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import argparse
import PIL
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-b", "--baseline", required=True,
	help="path to baseline image")
ap.add_argument("-o", "--output", required=True,
	help="path to output image")
args = vars(ap.parse_args())

# load the pre-trained model
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)
```

**第 11-18 行**解析我们的命令行参数——这个脚本需要三个开关:

*   我们输入的路径，一个我们想要放大的低分辨率图像。
*   `--baseline`:标准双线性插值后的输出基线图像——该图像将为我们提供一个基线，我们可以将其与我们的 SRCNN 结果进行比较。
*   `--output`:应用超分辨率后输出图像的路径。

**第 22 行**然后装盘后我们序列化的 SRCNN。

接下来，让我们准备一下要放大的图像:

```py
# load the input image, then grab the dimensions of the input image
# and crop the image such that it tiles nicely
print("[INFO] generating image...")
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
w -= int(w % config.SCALE)
h -= int(h % config.SCALE)
image = image[0:h, 0:w]

# resize the input image using bicubic interpolation then write the
# baseline image to disk
lowW = int(w * (1.0 / config.SCALE))
lowH = int(h * (1.0 / config.SCALE))
highW = int(lowW * (config.SCALE / 1.0))
highH = int(lowH * (config.SCALE / 1.0))
scaled = np.array(Image.fromarray(image).resize((lowW, lowH),
	resample=PIL.Image.BICUBIC))
scaled = np.array(Image.fromarray(scaled).resize((highW, highH),
	resample=PIL.Image.BICUBIC))
cv2.imwrite(args["baseline"], scaled)

# allocate memory for the output image
output = np.zeros(scaled.shape)
(h, w) = output.shape[:2]
```

我们首先从磁盘的**行 27** 载入图像。**第 28-31 行**裁剪我们的图像，当应用我们的滑动窗口并通过我们的 SRCNN 传递子图像时，它可以很好地平铺。**第 35-42 行**使用 PIL/皮鲁的`resize`函数对我们的输入`--image`应用标准双线性插值。

将我们的图像放大`SCALE`倍有两个目的:

1.  它为我们提供了使用传统图像处理进行标准升迁的基线。
2.  我们的 SRCNN 需要原始低分辨率图像的高分辨率输入——这张`scaled`图像就是为了这个目的。

最后，**行 46** 为我们的`output`图像分配内存。

我们现在可以应用我们的滑动窗口:

```py
# slide a window from left-to-right and top-to-bottom
for y in range(0, h - config.INPUT_DIM + 1, config.LABEL_SIZE):
	for x in range(0, w - config.INPUT_DIM + 1, config.LABEL_SIZE):
		# crop ROI from our scaled image
		crop = scaled[y:y + config.INPUT_DIM,
			x:x + config.INPUT_DIM].astype("float32")

		# make a prediction on the crop and store it in our output
		# image
		P = model.predict(np.expand_dims(crop, axis=0))
		P = P.reshape((config.LABEL_SIZE, config.LABEL_SIZE, 3))
		output[y + config.PAD:y + config.PAD + config.LABEL_SIZE,
			x + config.PAD:x + config.PAD + config.LABEL_SIZE] = P
```

对于沿途的每一站，在`LABEL_SIZE`步骤中，我们从`scaled`(**53 和 54 线**)中`crop`出子图像。`crop`的空间维度与我们 SRCNN 要求的输入维度相匹配。

然后我们获取`crop`子图像，并通过我们的 SRCNN 进行推理。`SRCNN`、`P`的输出具有空间维度`LABEL_SIZE x LABEL_SIZE x CHANNELS`，也就是`21×21×3`——然后我们将来自网络的高分辨率预测存储在`output`图像中。

***备注:*** 为了简单起见，我通过`.predict`方法一次处理一个子图像。为了实现更快的吞吐率，尤其是在 GPU 上，您将需要批处理`crop`子图像。这个动作可以通过维护一个批次的(`x,y`)-坐标列表来完成，该列表将批次中的每个样本映射到它们相应的`output`位置。

我们的最后一步是删除输出图像上由填充引起的任何黑色边框，然后将图像保存到磁盘:

```py
# remove any of the black borders in the output image caused by the
# padding, then clip any values that fall outside the range [0, 255]
output = output[config.PAD:h - ((h % config.INPUT_DIM) + config.PAD),
	config.PAD:w - ((w % config.INPUT_DIM) + config.PAD)]
output = np.clip(output, 0, 255).astype("uint8")

# write the output image to disk
cv2.imwrite(args["output"], output)
```

至此，我们完成了 SRCNN 管道的实现！在下一节中，我们将把`resize.py`应用到几个示例输入低分辨率图像，并将我们的 SRCNN 结果与传统图像处理进行比较。

## **超分辨率结果**

现在我们已经(1)训练了我们的 SRCNN 并且(2)实现了`resize.py`，我们准备好对输入图像应用超分辨率。打开一个 shell 并执行以下命令:

```py
$ python resize.py --image jemma.png --baseline baseline.png \
	--output output.png
[INFO] loading model...
[INFO] generating image...
```

**图 2** 包含了我们的输出图像。在*左侧*是我们希望提高分辨率的输入图像(`125×166`)。

然后，在*中间的*，我们通过标准双线性插值将输入图像分辨率增加`2x`到`250×332`。这张图片是我们的基线。请注意图像是如何低分辨率，模糊，并且一般来说，视觉上没有吸引力。

最后，在右边的*，我们有来自 SRCNN 的输出图像。在这里，我们可以看到我们再次将分辨率提高了`2x`，但这次图像明显不那么模糊，而且更加美观。*

 *我们还可以通过更高的倍数来提高我们的图像分辨率，前提是我们已经训练了我们的 SRCNN 这样做。

## **概要******

在本教程中，我们回顾了“超分辨率”的概念，然后实现了超分辨率卷积神经网络(SRCNN)。在训练我们的 SRCNN 之后，我们对输入图像应用了超分辨率。

我们的实现遵循了[董等人(2016)](https://ieeexplore.ieee.org/document/7115171) 的工作。尽管此后出现了许多超分辨率论文(并将继续出现)， [Dong et al.](https://ieeexplore.ieee.org/document/7115171) 的论文仍然是最容易理解和实现的论文之一，对于任何有兴趣研究超分辨率的人来说，这是一个极好的起点。

### **引用信息**

**罗斯布鲁克，A.** “图像超分辨率”， *PyImageSearch* ，2022，【https://pyimg.co/jia4g】T4

```py
@article{Rosebrock_2022_ISR,
  author = {Adrian Rosebrock},
  title = {Image Super Resolution},
  journal = {PyImageSearch},
  year = {2022},
  note = {https://pyimg.co/jia4g},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****