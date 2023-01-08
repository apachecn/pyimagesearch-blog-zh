# U-Net:在 PyTorch 中训练图像分割模型

> 原文：<https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/>

在今天的教程中，我们将研究图像分割，并基于流行的 U-Net 架构从头构建我们自己的分割模型。

本课是高级 PyTorch 技术 3 部分系列的最后一课:

1.  [*在 PyTorch*](https://pyimagesearch.com/2021/10/25/training-a-dcgan-in-pytorch/) 训练一个 DCGAN 周前的教程)
2.  [*在 PyTorch*](https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/) 中从头开始训练一个物体探测器(上周的课)
3.  *U-Net:在 PyTorch 中训练图像分割模型*(今天的教程)

计算机视觉社区已经设计了各种任务，例如图像分类、对象检测、定位等。，用于理解图像及其内容。这些任务让我们对对象类及其在图像中的位置有了更高层次的理解。

在图像分割中，我们更进一步，要求我们的模型将图像中的每个像素分类到它所代表的对象类别。这可以被视为像素级图像分类，并且是比简单的图像分类、检测或定位困难得多的任务。我们的模型必须自动确定图像中像素级的所有对象及其精确位置和边界。

因此，图像分割提供了对图像的复杂理解，并广泛用于医学成像、自动驾驶、机器人操纵等。

**要学习如何在 PyTorch 中训练一个基于 U-Net 的分割模型，** ***继续阅读。***

## **U-Net:在 PyTorch 中训练图像分割模型**

在本教程中，我们将学习图像分割，并在 PyTorch 中构建和训练一个分割模型。我们将重点介绍一个非常成功的架构，U-Net，它最初是为医学图像分割而提出的。此外，我们将了解 U-Net 模型的显著特征，这使得它成为图像分割任务的合适选择。

具体来说，我们将在本教程中详细讨论以下内容:

*   U-Net 的架构细节使其成为一个强大的细分模型
*   为我们的图像分割任务创建自定义 PyTorch 数据集
*   从头开始训练 U-Net 分段模型
*   用我们训练好的 U-Net 模型对新图像进行预测

### **U-Net 架构概述**

U-Net 架构(见**图 1** )遵循编码器-解码器级联结构，其中编码器逐渐将信息压缩成低维表示。然后，解码器将该信息解码回原始图像尺寸。由于这一点，该架构获得了一个整体的 U 形，这导致了名称 U-Net。

除此之外，U-Net 架构的一个显著特征是跳过连接(在图 1 中用灰色箭头表示)，这使得信息能够从编码器端流向解码器端，从而使模型能够做出更好的预测。

具体来说，随着我们的深入，编码器在更高的抽象层次上处理信息。这仅仅意味着在初始层，编码器的特征映射捕获关于对象纹理和边缘的低级细节，并且随着我们逐渐深入，特征捕获关于对象形状和类别的高级信息。

值得注意的是，要分割图像中的对象，低层和高层信息都很重要。例如，对象和边缘信息之间的纹理变化可以帮助确定各种对象的边界。另一方面，关于对象形状所属类别的高级信息可以帮助分割相应的像素，以校正它们所代表的对象类别。

因此，为了在预测期间使用这两条信息，U-Net 架构在编码器和解码器之间实现了跳跃连接。这使我们能够在编码器端从不同深度获取中间特征图信息，并在解码器端将其连接起来，以处理和促进更好的预测。

在本教程的后面，我们将更详细地研究 U-Net 模型，并在 PyTorch 中从头开始构建它。

### **我们的 TGS 盐分割数据集**

对于本教程，我们将使用 TGS 盐分割数据集。该数据集是作为 Kaggle 上的 [TGS 盐鉴定挑战赛](https://www.kaggle.com/c/tgs-salt-identification-challenge)的一部分推出的。

实际上，即使有人类专家的帮助，也很难从图像中准确地识别盐沉积的位置。因此，挑战要求参与者帮助专家从地球表面下的地震图像中精确地确定盐沉积的位置。这实际上很重要，因为对盐存在的不正确估计会导致公司在错误的地点设置钻探机进行开采，导致时间和资源的浪费。

我们使用这个数据集的一个子部分，它包括 4000 张大小为`101×101`像素的图像，取自地球上的不同位置。这里，每个像素对应于盐沉积或沉积物。除了图像，我们还提供了与图像相同尺寸的地面实况像素级分割掩模(见**图** 2)。

蒙版中的白色像素代表盐沉积，黑色像素代表沉积物。我们的目标是正确预测图像中对应于盐沉积的像素。因此，我们有一个二元分类问题，其中我们必须将每个像素分类到两个类别之一，类别 1:盐或类别 2:非盐(或者，换句话说，沉积物)。

### **配置您的开发环境**

要遵循本指南，您需要在系统上安装 PyTorch 深度学习库、matplotlib、OpenCV、imutils、scikit-learn 和 tqdm 软件包。

幸运的是，使用 pip 安装这些包非常容易:

```py
$ pip install torch torchvision
$ pip install matplotlib
$ pip install opencv-contrib-python
$ pip install imutils
$ pip install scikit-learn
$ pip install tqdm
```

**如果你在为 PyTorch 配置开发环境时需要帮助，我*强烈推荐*阅读** [**PyTorch 文档**](https://pytorch.org/get-started/locally/) — PyTorch 的文档很全面，可以让你快速上手。

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

从这里，看一下目录结构:

```py
.
├── dataset
│   └── train
├── output
├── pyimagesearch
│   ├── config.py
│   ├── dataset.py
│   └── model.py
├── predict.py
└── train.py
```

`dataset`文件夹存储我们将用于训练我们的细分模型的 TGS 盐细分数据集。

此外，我们将在`output`文件夹中存储我们的训练模型和训练损失图。

`pyimagesearch`文件夹中的`config.py` 文件存储了我们代码的参数、初始设置和配置。

另一方面，`dataset.py`文件由我们的定制分段数据集类组成，`model.py`文件包含我们的 U-Net 模型的定义。

最后，我们的模型训练和预测代码分别在`train.py`和`predict.py`文件中定义。

### **创建我们的配置文件**

我们从讨论`config.py`文件开始，它存储了教程中使用的配置和参数设置。

```py
# import the necessary packages
import torch
import os

# base path of the dataset
DATASET_PATH = os.path.join("dataset", "train")

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")

# define the test split
TEST_SPLIT = 0.15

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False
```

我们从进口**2 号线和**3 号线的必要包装开始。然后，我们在第 6 行的**上定义数据集(即`DATASET_PATH`)的路径，并在第 9**行和第 10** 行的`IMAGE_DATASET_PATH`和`MASK_DATASET_PATH`上定义数据集文件夹内图像和蒙版的路径。**

在**第 13 行**，我们定义了数据集的一部分，我们将为测试集保留。然后，在**行第 16** 处，我们定义了`DEVICE`参数，该参数根据可用性决定我们是使用 GPU 还是 CPU 来训练我们的分割模型。在这种情况下，我们使用支持 CUDA 的 GPU 设备，并且在第 19 行将`PIN_MEMORY`参数设置为`True`。

```py
# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 40
BATCH_SIZE = 64

# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tgs_salt.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
```

接下来，我们在第 23-25 行的**上定义`NUM_CHANNELS`、`NUM_CLASSES`和`NUM_LEVELS`参数，我们将在本教程稍后详细讨论。最后，在**第 29-31 行**，我们定义了初始学习率(即`INIT_LR`)、总时期数(即`NUM_EPOCHS`)和批量大小(即`BATCH_SIZE`)等训练参数。**

在**第 34 行和第 35 行**，我们还定义了输入图像的尺寸，我们的图像应该调整到这个尺寸，以便我们的模型处理它们。我们在第 38 行**上进一步定义了一个阈值参数，这将有助于我们在基于二元分类的分割任务中将像素分类为两类中的一类。**

最后，我们在第 41 行的**上定义到输出文件夹(即`BASE_OUTPUT`)的路径，并在第 45-47** 行的**上定义到输出文件夹内的训练模型权重、训练图和测试图像的对应路径。**

### **创建我们的自定义分段数据集类**

既然我们已经定义了初始配置和参数，我们就可以理解将用于分段数据集的自定义数据集类了。

让我们从项目目录的`pyimagesearch`文件夹中打开`dataset.py`文件。

```py
# import the necessary packages
from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms

	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)

	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]

		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.maskPaths[idx], 0)

		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)

		# return a tuple of the image and its mask
		return (image, mask)
```

我们首先从**行 2** 上的`torch.utils.data`模块导入`Dataset`类。这很重要，因为所有 PyTorch 数据集都必须从这个基本数据集类继承。此外，在**第 3 行**，我们导入了 OpenCV 包，这将使我们能够使用其图像处理功能。

我们现在准备定义我们自己的自定义分段数据集。每个 PyTorch 数据集都需要从`Dataset`类(**第 5 行**)继承，并且应该有一个 *`__len__`* ( **第 13-15 行**)和一个 *`__getitem__`* ( **第 17-34 行**)方法。下面我们将逐一讨论这些方法。

我们从定义我们的初始化器构造函数开始，也就是第 6-11 行的 *`__init__`* 方法。该方法将我们的数据集的图像路径列表(即`imagePaths`)、相应的地面真相遮罩(即`maskPaths`)以及我们想要应用于我们的输入图像(**行 6** )的变换集(即`transforms`)作为输入。

在**的第 9-11 行**，我们用输入到 *`__init__`* 构造函数的参数来初始化我们的`SegmentationDataset`类的属性。

接下来，我们定义 *`__len__`* 方法，该方法返回数据集中图像路径的总数，如**第 15 行**所示。

*`__getitem__`* 方法的任务是将一个索引作为输入(**第 17 行**)并从数据集中返回相应的样本。在**第 19** 行，我们简单地在输入图像路径列表中的`idx`索引处抓取图像路径。然后，我们使用 OpenCV 加载图像(**第 23 行**)。默认情况下，OpenCV 加载 BGR 格式的图像，我们将其转换为 RGB 格式，如**第 24 行**所示。我们还在第 25 行的**中加载了相应的灰度模式的地面实况分割蒙版。**

最后，我们检查想要应用于数据集图像的输入变换(**第 28 行**)，并分别在**第 30 行和第 31 行**用所需的变换来变换图像和蒙版。这是很重要的，因为我们希望我们的图像和地面真相面具对应，并有相同的维度。在**行第 34** 处，我们返回包含图像及其对应遮罩(即`(image, mask)`)的元组，如图所示。

这就完成了我们的定制分段数据集的定义。接下来，我们将讨论 U-Net 架构的实现。

### **在 PyTorch 中构建我们的 U-Net 模型**

是时候详细研究我们的 U-Net 模型架构，并在 PyTorch 中从头开始构建它了。

我们从项目目录的`pyimagesearch`文件夹中打开我们的`model.py`文件，然后开始。

```py
# import the necessary packages
from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
```

在**第 2-11 行**，我们从 PyTorch 导入必要的层、模块和激活函数，我们将使用它们来构建我们的模型。

总的来说，我们的 U-Net 模型将由一个`Encoder`类和一个`Decoder`类组成。编码器将逐渐减小空间维度来压缩信息。此外，它将增加通道的数量，即每个阶段的特征图的数量，使我们的模型能够捕捉我们图像中的不同细节或特征。另一方面，解码器将采用最终的编码器表示，并逐渐增加空间维度和减少通道数量，以最终输出与输入图像具有相同空间维度的分割掩模。

接下来，我们定义一个`Block`模块作为编码器和解码器架构的构建单元。值得注意的是，我们定义的所有模型或模型子部分都需要从 PyTorch `Module`类继承，该类是 PyTorch 中所有神经网络模块的父类。

```py
class Block(Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		# store the convolution and RELU layers
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)

	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		return self.conv2(self.relu(self.conv1(x)))
```

我们从在第 13-23 行的**上定义我们的`Block`类开始。该模块的功能是获取具有`inChannels`个通道的输入特征图，应用两个卷积运算并在它们之间进行 ReLU 激活，并返回具有`outChannels`个通道的输出特征图。**

*`__init__`* 构造器将两个参数`inChannels`和`outChannels` ( **第 14 行**)作为输入，这两个参数分别确定输入特征图和输出特征图中的通道数。

我们初始化两个卷积层(即`self.conv1`和`self.conv2`)以及第 17-19 行上的一个 ReLU 激活。在**第 21-23 行**上，我们定义了`forward`函数，该函数将我们的特征图`x`作为输入，应用`self.conv1 =>` **`self.relu`** `=> self.conv2`操作序列并返回输出特征图。

```py
class Encoder(Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		# store the encoder blocks and maxpooling layer
		self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)

	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		blockOutputs = []

		# loop through the encoder blocks
		for block in self.encBlocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)

		# return the list containing the intermediate outputs
		return blockOutputs
```

接下来，我们在第 25-47 行的**上定义我们的`Encoder`类。类构造器(即 *`__init__`* 方法)将通道维度(**行 26** )的元组(即`channels`)作为输入。请注意，第一个值表示我们的输入图像中的通道数量，随后的数字逐渐使通道尺寸加倍。**

我们首先在第 29-31 行的**上 PyTorch 的`ModuleList`功能的帮助下初始化编码器的块列表(即`self.encBlocks`)。每个`Block`获取前一个块的输入通道，并将输出特征图中的通道加倍。我们还初始化了一个`MaxPool2d()`层，它将特征图的空间维度(即高度和宽度)减少了 2 倍。**

最后，我们在第 34-47 行的**上为我们的编码器定义了`forward`函数。该功能将图像`x`作为输入，如**行 34** 所示。在第 36**行**上，我们初始化一个空的`blockOutputs`列表，存储来自编码器模块的中间输出。请注意，这将使我们能够稍后将这些输出传递给解码器，在那里可以用解码器特征映射来处理它们。**

在**第 39-44 行**上，我们循环通过编码器中的每个块，通过该块处理输入特征图(**第 42 行**，并将该块的输出添加到我们的`blockOutputs`列表中。然后，我们对我们的块输出应用最大池操作(**行 44** )。这是对编码器中的每个块进行的。

最后，我们在第 47 行的**返回我们的`blockOutputs`列表。**

```py
class Decoder(Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])

	def forward(self, x, encFeatures):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)

			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)

		# return the final decoder output
		return x

	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)

		# return the cropped features
		return encFeatures
```

现在我们定义我们的`Decoder`类(**第 50-87 行**)。类似于编码器定义，解码器 *`__init__`* 方法将信道维度(**行 51** )的元组(即`channels`)作为输入。请注意，与编码器端相比，这里的不同之处在于通道逐渐减少 2 倍，而不是增加。

我们初始化**线 55** 上的通道数。此外，在**第 56-58 行**，我们定义了一列上采样块(即`self.upconvs`)，它们使用`ConvTranspose2d`层以因子 2 对特征图的空间维度(即高度和宽度)进行上采样。此外，该层还将通道数量减少了 1/2。

最后，我们为解码器初始化一个类似于编码器端的块列表(即`self.dec_Blocks`)。

在**第 63-75 行**，我们定义了`forward`函数，它将我们的特征图`x`和来自编码器的中间输出列表(即`encFeatures`)作为输入。从**行 65** 开始，我们循环遍历多个通道，并执行以下操作:

*   首先，我们通过我们的第 *i* 上采样块(**行 67** )对解码器的输入(即`x`)进行上采样
*   由于我们必须(沿着通道维度)将来自编码器的第 *i* (即`encFeatures[i]`)个中间特征图与来自上采样块的当前输出`x`连接起来，我们需要确保`encFeatures[i]`和`x`的空间维度匹配。为了完成这个，我们在第 73 行使用了我们的`crop`函数。
*   接下来，我们沿着**行 74** 上的通道维度，将我们裁剪的编码器特征图(即`encFeat`)与我们当前的上采样特征图`x`连接起来
*   最后，我们将级联的输出通过我们的第 *i* 个解码器模块(**线 75**

循环完成后，我们在**线 78** 上返回最终的解码器输出。

在**第 80-87 行**，我们定义了我们的裁剪函数，该函数从编码器(即`encFeatures`)获取一个中间特征图，从解码器(即`x`)获取一个特征图输出，并在空间上将前者裁剪为后者的尺寸。

为此，我们首先获取第 83 行**上`x`的空间尺寸(即高度`H`和宽度`W`)。然后，我们使用`CenterCrop`函数(**行 84** )将`encFeatures`裁剪到空间维度`[H, W]`，并最终在**行 87** 上返回裁剪后的输出。**

既然我们已经定义了组成我们的 U-Net 模型的子模块，我们就准备构建我们的 U-Net 模型类。

```py
class UNet(Module):
	def __init__(self, encChannels=(3, 16, 32, 64),
		 decChannels=(64, 32, 16),
		 nbClasses=1, retainDim=True,
		 outSize=(config.INPUT_IMAGE_HEIGHT,  config.INPUT_IMAGE_WIDTH)):
		super().__init__()
		# initialize the encoder and decoder
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)

		# initialize the regression head and store the class variables
		self.head = Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize
```

我们首先定义 *`__init__`* 构造函数方法(**第 91-103 行**)。它将以下参数作为输入:

*   `encChannels`:元组定义了当我们的输入通过编码器时通道维度的逐渐增加。我们从 3 个通道(即 RGB)开始，然后将通道数量增加一倍。
*   `decChannels`:元组定义了当我们的输入通过解码器时，信道维度的逐渐减小。我们每走一步都将通道减少 2 倍。
*   `nbClasses`:这定义了我们必须对每个像素进行分类的分割类别的数量。这通常对应于我们的输出分割图中的通道数，其中每个类有一个通道。
    *   因为我们正在处理两个类(即，二进制分类)，所以我们保持单个通道并使用阈值进行分类，这将在后面讨论。
*   `retainDim`:表示是否要保留原来的输出尺寸。
*   `outSize`:决定输出分割图的空间尺寸。我们将其设置为与输入图像相同的尺寸(即(`config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH`))。

在第 97 和 98 行上，我们初始化我们的编码器和解码器网络。此外，我们初始化一个卷积头，稍后通过它将我们的解码器输出作为输入，并输出我们的具有`nbClasses`个通道的分割图(**行 101** )。

我们还在**行 102 和 103** 上初始化`self.retainDim`和`self.outSize`属性。

```py
def forward(self, x):
		# grab the features from the encoder
		encFeatures = self.encoder(x)

		# pass the encoder features through decoder making sure that
		# their dimensions are suited for concatenation
		decFeatures = self.decoder(encFeatures[::-1][0],
			encFeatures[::-1][1:])

		# pass the decoder features through the regression head to
		# obtain the segmentation mask
		map = self.head(decFeatures)

		# check to see if we are retaining the original output
		# dimensions and if so, then resize the output to match them
		if self.retainDim:
			map = F.interpolate(map, self.outSize)

		# return the segmentation map
		return map
```

最后，我们准备讨论我们的 U-Net 模型的`forward`功能(**第 105-124 行**)。

我们从通过编码器传递输入`x`开始。这将输出编码器特征图列表(即`encFeatures`，如**行 107** 所示。注意`encFeatures`列表包含从第一个编码器模块输出到最后一个的所有特征映射，如前所述。所以我们可以把这个列表中的特征图顺序倒过来:`encFeatures[::-1]`。

现在`encFeatures[::-1]`列表包含逆序的特征映射输出(即从最后一个到第一个编码器模块)。请注意，这一点很重要，因为在解码器端，我们将从最后一个编码器模块输出到第一个编码器模块输出开始利用编码器特征映射。

接下来，我们通过**线 111** 将最终编码器块的输出(即`encFeatures[::-1][0]`)和所有中间编码器块的特征映射输出(即`encFeatures[::-1][1:]`)传递给解码器。解码器的输出存储为`decFeatures`。

我们将解码器输出传递给我们的卷积头(**行 116** )以获得分段掩码。

最后，我们检查`self.retainDim`属性是否为`True` ( **第 120 行**)。如果是，我们将最终分割图插值到由`self.outSize` ( **行 121** )定义的输出尺寸。我们在第 124 行返回最终的分割图。

这就完成了我们的 U-Net 模型的实现。接下来，我们将了解细分渠道的培训程序。

### **训练我们的细分模型**

现在我们已经实现了数据集类和模型架构，我们准备在 PyTorch 中构建和训练我们的分段管道。让我们从项目目录中打开`train.py`文件。

具体来说，我们将详细了解以下内容:

*   构建数据加载管道
*   初始化模型和训练参数
*   定义训练循环
*   可视化训练和测试损失曲线

```py
# USAGE
# python train.py

# import the necessary packages
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
```

我们首先在**5 号线和 6 号线**上导入我们自定义的`SegmentationDataset`类和`UNet`模型。接下来，我们在**第 7 行**导入我们的配置文件。

由于我们的盐分割任务是像素级二进制分类问题，我们将使用二进制交叉熵损失来训练我们的模型。在**第 8 行**，我们从 PyTorch `nn`模块导入二值交叉熵损失函数(即`BCEWithLogitsLoss`)。除此之外，我们从 PyTorch `optim`模块导入了`Adam`优化器，我们将使用它来训练我们的网络( **Line 9** )。

接下来，在**的第 11 行**，我们从`sklearn`库中导入内置的`train_test_split`函数，使我们能够将数据集分成训练集和测试集。此外，我们从第 12 行的**上的`torchvision`导入`transforms`模块，对我们的输入图像应用图像变换。**

最后，我们导入其他有用的包来处理我们的文件系统，在训练过程中跟踪进度，为我们的训练过程计时，并在第**行第 13-18** 绘制损失曲线。

一旦我们导入了所有必需的包，我们将加载我们的数据并构建数据加载管道。

```py
# load the image and mask filepaths in a sorted manner
imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(imagePaths, maskPaths,
	test_size=config.TEST_SPLIT, random_state=42)

# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]

# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()
```

在**的第 21 行和第 22 行**，我们首先定义两个列表(即`imagePaths`和`maskPaths`)，分别存储所有图像的路径和它们对应的分割掩膜。

然后，我们在 scikit-learn 的第 26 行的`train_test_split`的帮助下，将我们的数据集划分为一个训练和测试集。注意，该函数将一系列列表(此处为`imagePaths`和`maskPaths`)作为输入，同时返回训练和测试集图像以及相应的训练和测试集掩码，我们在**行的第 30 和 31** 行对其进行解包。

我们将`testImages`列表中的路径存储在由**第 36 行**上的`config.TEST_PATHS`定义的测试文件夹路径中。

现在，我们已经准备好设置数据加载管道了。

```py
# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])

# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
	transforms=transforms)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")

# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())
```

我们首先定义加载输入图像时要应用的变换，并在第 41-44 行的`Compose`函数的帮助下合并它们。我们的转型包括:

*   它使我们能够将输入的图像转换成 PIL 图像格式。请注意，这是必要的，因为我们使用 OpenCV 在自定义数据集中加载图像，但是 PyTorch 希望输入图像样本是 PIL 格式的。
*   `Resize()`:允许我们将图像调整到我们的模型可以接受的特定输入尺寸(即`config.INPUT_IMAGE_HEIGHT`、`config.INPUT_IMAGE_WIDTH`)
*   `ToTensor()`:使我们能够将输入图像转换为 PyTorch 张量，并将输入的 PIL 图像从最初的`[0, 255]`转换为`[0, 1]`。

最后，我们将训练和测试图像以及相应的掩码传递给我们的定制`SegmentationDataset`，以在**第 47-50 行**上创建训练数据集(即`trainDS`)和测试数据集(即`testDS`)。请注意，我们可以简单地将**行 41** 上定义的转换传递给我们的自定义 PyTorch 数据集，以便在自动加载图像时应用这些转换。

我们现在可以借助`len()`方法打印出`trainDS`和`testDS`中的样本数，如**第 51 行和第 52 行**所示。

在**的第 55-60 行**，我们通过将我们的训练数据集和测试数据集传递给 Pytorch DataLoader 类，直接创建我们的训练数据加载器(即`trainLoader`)和测试数据加载器(即`testLoader`)。我们将`shuffle`参数`True`保存在训练数据加载器中，因为我们希望来自所有类的样本均匀地出现在一个批次中，这对于基于批次梯度的优化方法的最佳学习和收敛非常重要。

既然我们已经构建并定义了数据加载管道，我们将初始化我们的 U-Net 模型和训练参数。

```py
# initialize our UNet model
unet = UNet().to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}
```

我们首先在**行 63** 定义我们的`UNet()`模型。注意，`to()`函数将我们的`config.DEVICE`作为输入，并在提到的设备上注册我们的模型及其参数。

在**第 66 和 67 行**，我们定义了我们的损失函数和优化器，我们将用它来训练我们的分割模型。`Adam`优化器类将我们模型的参数(即`unet.parameters()`)和学习率(即`config.INIT_LR`)作为输入，我们将使用它们来训练我们的模型。

然后，我们定义迭代我们的整个训练和测试集所需的步骤数，即第 70 行**和第 71 行**上的`trainSteps`和`testSteps`。假设 dataloader 为我们的模型`config.BATCH_SIZE`提供了一次要处理的样本数，那么迭代整个数据集(即训练集或测试集)所需的步骤数可以通过将数据集中的样本总数除以批量大小来计算。

我们还在第 74 行的**上创建了一个空字典`H`，我们将使用它来跟踪我们的训练和测试损失历史。**

最后，我们已经准备好开始理解我们的训练循环。

```py
# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
	unet.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0

	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

		# perform a forward pass and calculate the training loss
		pred = unet(x)
		loss = lossFunc(pred, y)

		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()

		# add the loss to the total training loss so far
		totalTrainLoss += loss

	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()

		# loop over the validation set
		for (x, y) in testLoader:
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

			# make the predictions and calculate the validation loss
			pred = unet(x)
			totalTestLoss += lossFunc(pred, y)

	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgTestLoss = totalTestLoss / testSteps

	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTestLoss))

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))
```

为了给我们的训练过程计时，我们使用第 78 行的**函数。这个函数在被调用时输出时间。因此，我们可以在训练过程开始时调用一次，在结束时调用一次，然后减去两次输出，得到经过的时间。**

我们在训练循环中迭代`config.NUM_EPOCHS`，如第 79 行的**所示。在我们开始训练之前，将我们的模型设置为训练模式是很重要的，正如我们在**81**线上看到的。这将指导 PyTorch 引擎跟踪我们的计算和梯度，并构建一个计算图，以便稍后进行反向传播。**

我们初始化第 84 行**和第 85 行**上的变量`totalTrainLoss`和`totalTestLoss`，以跟踪我们在给定时期的损失。接下来，在**行 88** 上，我们迭代我们的`trainLoader`数据加载器，它一次提供一批样本。如**第 88-103 行**所示，训练循环包括以下步骤:

*   首先，在第 90 行的**上，我们将数据样本(即`x`和`y`)移动到由`config.DEVICE`定义的用于训练模型的设备上**
*   然后，我们将输入图像样本`x`通过**线 93** 上的`unet`模型，并获得输出预测(即`pred`
*   在**第 94 行**，我们计算模型预测`pred`和我们的地面实况标签`y`之间的损失
*   在第 98-100 行上，我们通过模型反向传播我们的损失并更新参数
    *   这是在三个简单步骤的帮助下完成的；我们从清除**线 98** 上先前步骤的所有累积梯度开始。接下来，我们在计算的损失函数上调用`backward`方法，如**第 99 行**所示。这将指导 PyTorch 计算我们的损失相对于计算图中涉及的所有变量的梯度。最后，我们调用`opt.step()`来更新我们的模型参数，如**行 100** 所示。
*   最后，**行 103** 使我们能够通过将该步骤的损失添加到`totalTrainLoss`变量来跟踪我们的训练损失，该变量累积所有样本的训练损失。

重复该过程，直到遍历所有数据集样本一次(即，完成一个时期)。

一旦我们处理了整个训练集，我们将希望在测试集上评估我们的模型。这很有帮助，因为它允许我们监控测试损失，并确保我们的模型不会过度适应训练集。

在测试集上评估我们的模型时，我们不跟踪梯度，因为我们不会学习或反向传播。因此，我们可以在`torch.no_grad()`的帮助下关闭梯度计算，并冻结模型权重，如**行 106** 所示。这将指示 PyTorch 引擎不计算和保存梯度，从而在评估期间节省内存和计算。

我们通过调用**行 108** 上的`eval()`函数将我们的模型设置为评估模式。然后，我们遍历测试集样本，并计算我们的模型对测试数据的预测(**第 116 行**)。然后将测试损失加到`totalTestLoss`中，该值累计整个测试集的测试损失。

然后，我们获得所有步骤的平均训练损失和测试损失，即第 120 和 121 行**、**上的`avgTrainLoss`和`avgTestLoss`，并将它们存储在第 124 和 125 行上的**中，存储到我们的字典`H`中，该字典是我们在开始时创建的，用于跟踪我们的损失。**

最后，我们打印当前的 epoch 统计数据，包括 128-130 线**上的列车和测试损失。这使我们结束了一个时期，包括在我们的训练集上的一个完整的训练周期和在我们的测试集上的评估。整个过程重复`config.NUM_EPOCHS`次，直到我们的模型收敛。**

在**的第 133 行和第 134 行**上，我们记下了我们训练循环的结束时间，并从`startTime`(我们在训练开始时初始化的)中减去`endTime`，以获得我们网络训练期间所用的总时间。

```py
# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)

# serialize the model to disk
torch.save(unet, config.MODEL_PATH)
```

接下来，我们使用 matplotlib 的 pyplot 包来可视化和保存我们在**行 138-146** 上的训练和测试损失曲线。我们可以简单地将损失历史字典`H`的`train_loss`和`test_loss`键传递给`plot`函数，如第**行第 140 和 141** 行所示。最后，我们设置我们的图的标题和图例(**行 142-145** )，并将我们的可视化保存在**行 146** 。

最后，在**第 149** 行，我们在`torch.save()`函数的帮助下保存我们训练好的 U-Net 模型的权重，该函数将我们训练好的`unet`模型和`config.MODEL_PATH`作为我们想要保存模型的输入。

一旦我们的模型被训练，我们将看到一个类似于图 4**所示的损失轨迹图。**注意到`train_loss`随着时间的推移逐渐减少并慢慢收敛。此外，我们看到`test_loss`也随着`train_loss`遵循相似的趋势和值而持续减少，这意味着我们的模型概括得很好，并且不会过度适应训练集。

### **使用我们训练的 U-Net 模型进行预测**

一旦我们训练并保存了我们的细分模型，我们就可以看到它的运行并将其用于细分任务。

从我们的项目目录中打开`predict.py`文件。

```py
# USAGE
# python predict.py

# import the necessary packages
from pyimagesearch import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)

	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")

	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()
```

我们一如既往地在第 5-10 行导入必要的包和模块。

为了使用我们的分割模型进行预测，我们将需要一个函数，该函数可以获取我们的训练模型和测试图像，预测输出分割掩码，并最终可视化输出预测。

为此，我们首先定义`prepare_plot`函数来帮助我们可视化我们的模型预测。

这个函数将一幅图像、它的真实遮罩和我们的模型预测的分割输出，即`origImage`、`origMask`和`predMask` ( **第 12 行**)作为输入，并创建一个单行三列的网格(**第 14 行**)来显示它们(**第 17-19 行**)。

最后，**行 22-24** 为我们的情节设置标题，在**行 27 和 28** 显示它们。

```py
def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()

	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0

		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (128, 128))
		orig = image.copy()

		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
			filename)

		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_HEIGHT))

```

接下来，我们定义我们的`make_prediction`函数(**第 31-77 行**)，它将把测试图像的路径和我们训练的分割模型作为输入，并绘制预测的输出。

由于我们仅使用我们训练过的模型进行预测，我们首先将我们的模型设置为`eval`模式，并分别关闭**线 33** 和**线 36** 上的 PyTorch 梯度计算。

在**第 39-41 行**上，我们使用 OpenCV ( **第 39 行**)从`imagePath`加载测试图像(即`image`)，将其转换为 RGB 格式(**第 40 行**)，并将其像素值从标准`[0-255]`归一化到我们的模型被训练处理的范围`[0, 1]`(**第 41 行**)。

然后在**第 44 行**将图像调整到我们的模型可以接受的标准图像尺寸。由于我们必须在将变量`image`传递给模型之前对其进行修改和处理，所以我们在**行 45** 上制作了一个额外的副本，并将其存储在`orig`变量中，我们将在以后使用它。

在第**行第 49-51** 行，我们获取测试图像的地面真相蒙版的路径，并将蒙版加载到第**行第 55** 行。注意，我们将遮罩调整到与输入图像相同的尺寸(**第 56 行和第 57 行**)。

现在，我们将我们的`image`处理成模型可以处理的格式。请注意，目前我们的`image`的形状是`[128, 128, 3]`。然而，我们的分割模型接受格式为`[batch_dimension, channel_dimension, height, width]`的四维输入。

```py
		# make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)

		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()

		# filter out the weak predictions and convert them to integers
		predMask = (predMask > config.THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)

		# prepare a plot for visualization
		prepare_plot(orig, gtMask, predMask)
```

在**第 62 行**上，我们转置图像将其转换为通道优先格式，即`[3, 128, 128]`，在**第 63 行**上，我们使用 numpy 的`expand_dims`函数添加额外的维度，将我们的图像转换为四维数组(即`[1, 3, 128, 128]`)。请注意，这里的第一维表示等于 1 的批处理维，因为我们一次处理一个测试图像。然后，我们在`torch.from_numpy()`函数的帮助下将我们的图像转换为 PyTorch 张量，并在 **Line 64** 的帮助下将其移动到我们的模型所在的设备上。

最后，在第**行第 68-70** 行，我们通过将测试图像传递给我们的模型并将输出预测保存为`predMask`来处理测试图像。然后我们应用 sigmoid 激活来得到我们在范围`[0, 1]`内的预测。如前所述，分割任务是一个分类问题，我们必须将像素分类到两个离散类中的一个。由于 sigmoid 在`[0, 1]`范围内输出连续值，我们使用**线 73** 上的`config.THRESHOLD`将输出二进制化，并分配像素，值等于`0`或`1`。这意味着任何大于阈值的都将被赋值`1`，而其他的将被赋值`0`。

由于阈值输出(即`(predMask > config.THRESHOLD)`)现在由值`0`或`1`组成，将其乘以`255`会使`predMask`中的最终像素值为`0`(即黑色像素值)或`255`(即白色像素值)。如前所述，白色像素对应于我们的模型检测到盐沉积的区域，黑色像素对应于不存在盐的区域。

我们借助于**行 77** 上的`prepare_plot`函数，绘制出我们的原始图像(即`orig`)、地面真实遮罩(即`gtMask`)和我们的预测输出(即`predMask`)。这就完成了我们的`make_prediction`函数的定义。

我们现在可以看到我们的模型在运行了。

```py
# load the image paths in our testing file and randomly select 10
# image paths
print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)

# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)

# iterate over the randomly selected test image paths
for path in imagePaths:
	# make predictions and visualize the results
	make_predictions(unet, path)
```

在**第 82 行和第 83 行**，我们打开存储测试图像路径的文件夹，随机抓取 10 个图像路径。**第 87 行**在`config.MODEL_PATH`点从保存的检查点加载我们 U 网的训练重量。

我们最后迭代我们随机选择的测试`imagePaths`，并在第 90-92 行**的`make_prediction`函数的帮助下预测输出。**

**图 5** 显示了我们的`make_prediction`函数的示例可视化输出。黄色区域代表 1 类:盐，深蓝色区域代表 2 类:非盐(沉积物)。

我们看到，在案例 1 和案例 2(即，分别为行 1 和行 2)中，我们的模型正确识别了包含盐沉积的大多数位置。然而，一些存在盐矿床的地区没有被确定。

然而，在情况 3(即，行 3)中，我们的模型已经将一些区域识别为没有盐的盐沉积(中间的黄色斑点)。这是一个假阳性，其中我们的模型错误地预测了阳性类别，即盐的存在，在地面真相中不存在盐的区域。

值得注意的是，实际上，从应用的角度来看，情况 3 中的预测具有误导性，并且比其他两种情况中的预测风险更大。这可能是因为在前两种情况下，如果专家在预测的黄色标记位置设置钻探机开采盐矿床，他们将成功发现盐矿床。然而，如果他们在假阳性预测的位置进行同样的操作(如案例 3 所示)，将会浪费时间和资源，因为在该位置不存在盐沉积。

### **学分**

[Aman Arora 的惊人文章](https://amaarora.github.io/2020/09/13/unet.html) 启发我们在`model.py`文件中实现 U-Net 模型。

## **总结**

在本教程中，我们学习了图像分割，并在 PyTorch 中从头开始构建了一个基于 U-Net 的图像分割管道。

具体来说，我们讨论了 U-Net 模型的架构细节和显著特征，使其成为图像分割的实际选择。

此外，我们还了解了如何在 PyTorch 中为手头的分割任务定义自己的自定义数据集。

最后，我们看到了如何在 PyTorch 中训练基于 U-Net 的分割管道，并使用训练好的模型实时预测测试图像。

学习完教程后，您将能够理解任何图像分割管道的内部工作原理，并在 PyTorch 中从头开始构建自己的分割模型。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****