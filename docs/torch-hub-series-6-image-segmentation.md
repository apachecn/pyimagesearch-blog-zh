# 火炬中心系列#6:图像分割

> 原文：<https://pyimagesearch.com/2022/01/24/torch-hub-series-6-image-segmentation/>

在本教程中，您将了解用于分段的全卷积网络(fcn)背后的概念。此外，我们将了解如何使用 Torch Hub 导入预训练的 FCN 模型，并在我们的项目中使用它来获得输入图像的实时分割输出。

本课是关于火炬中心的 6 部分系列的最后一部分:

1.  [*火炬中心系列# 1:*](https://pyimagesearch.com/2021/12/20/torch-hub-series-1-introduction-to-torch-hub/)火炬中心介绍
2.  [*火炬枢纽系列#2: VGG 和雷斯内特*](https://pyimagesearch.com/2021/12/27/torch-hub-series-2-vgg-and-resnet/)
3.  [*火炬轮毂系列#3: YOLO v5 和 SSD——物体检测模型*](https://pyimagesearch.com/2022/01/03/torch-hub-series-3-yolov5-and-ssd-models-on-object-detection/)
4.  [*火炬轮毂系列# 4:——甘模型*](https://pyimagesearch.com/2022/01/10/torch-hub-series-4-pgan-model-on-gan/)
5.  [*火炬中枢系列# 5:MiDaS——深度估计模型*](https://pyimagesearch.com/2022/01/17/torch-hub-series-5-midas-model-on-depth-estimation/)
6.  *火炬中枢系列#6:图像分割*(本教程)

**要了解完全卷积网络背后的概念并将其用于图像分割，*继续阅读。***

## **火炬中枢系列#6:图像分割**

### **话题描述**

在本系列的前几篇文章中，我们研究了不同的计算机视觉任务(例如，分类、定位、深度估计等。)，这使我们能够理解图像中的内容及其相关语义。此外，在过去的[教程](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/)中，我们对图像分割任务及其在理解像素级图像中复杂细节和信息的有用性有了深入的了解。此外，我们还研究了 UNET 等分割模型，这些模型利用跳跃连接等显著的架构特征来实时有效地分割图像。

在今天的教程中，我们将看看另一种分割图像的方法(即，使用全卷积网络(fcn))。**图 1** 展示了 FCN 的高层建筑。这些网络遵循一种训练范式，该范式使它们能够使用从除分割(例如，分类)之外的计算机视觉任务中学习到的特征来有效地分割图像。具体来说，我们将详细讨论以下内容:

*   为 FCN 模型提供动力的监督预培训范例
*   允许 FCN 模型处理任何大小的输入图像并有效计算分割输出的架构修改
*   从 Torch Hub 导入预训练的 FCN 细分模型，以便快速无缝地集成到我们的深度学习项目中
*   使用不同编码器的 FCN 模型实时分割图像，用于我们自己的深度学习应用

### **FCN 分割模型**

针对不同的计算机视觉任务(例如，分类、定位等)训练的深度学习模型。)努力从图像中提取相似的特征以理解图像内容，而不考虑手边的下游任务。从以下事实可以进一步理解这一点:只为对象分类而训练的模型的注意力图也可以指向图像中特定类别对象出现的位置，如在[之前的教程](https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/)中所见。这意味着分类模型具有关于全局对象类别以及它在图像中所处位置的信息。

FCN 分割模型旨在利用这一事实，并遵循一种基于重新利用已训练好的分类模型进行分割的方法。这需要对分类模型层进行仔细的设计和修改，以便将模型无缝地转换成分段管道。

从形式上看，FCN 方法大体上采用了两个步骤来实现这一目标。首先，它获取在图像分类任务上训练的现成模型(即 ResNet)。接下来，为了将其转换为分段模型，它用卷积层替换最后完全连接的层。注意，这可以通过简单地使用内核大小与输入特征映射维数相同的卷积层来实现(如**图 2** 所示)。

由于网络现在只由卷积层组成，而没有固定节点数的静态全连接层，因此它可以将任何维度的图像作为输入并进行处理。此外，我们在改进的分类模型的最终层之后添加去卷积层，以将特征图映射回输入图像的原始维度，从而获得具有与输入对应的像素的分割输出。

现在，我们已经了解了 FCN 背后的方法，让我们继续设置我们的项目目录，并查看我们预先培训的 FCN 模型的运行情况。

### **配置您的开发环境**

要遵循本指南，您需要在系统上安装 PyTorch 库、`torchvision`模块和`matplotlib`库。

幸运的是，使用 pip 很容易安装这些包:

```py
$ pip install torch torchvision
$ pip install matplotlib
```

**如果你需要帮助为 OpenCV 配置开发环境，我*强烈推荐*阅读我的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让你启动并运行。

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
│   ├── test_set
│   └── training_set
├── output
├── predict.py
└── pyimagesearch
    └── config.py
    └── utils.py
```

我们从理解项目目录的结构开始。具体来说，数据集文件夹存储了我们的狗和猫数据集的`test_set`和`training_set`图像。对于本教程，我们将使用`test_set`图像进行推理，并使用我们的 FCN 模型进行分割掩模预测。

像往常一样，输出文件夹存储输入图像的可视化和来自我们预训练的 FCN 模型的预测分割掩模。

此外，`predict.py`文件使我们能够从 Torch Hub 加载预训练的 FCN 分割模型，并将它们集成到我们的项目中，用于实时分割掩模预测和可视化。

最后，`pyimagesearch`文件夹中的`config.py`文件存储了我们代码的参数、初始设置和配置，`utils.py`文件定义了帮助函数，使我们能够有效地可视化我们的分段输出。

### **下载数据集**

在本系列前面的教程之后，我们将使用来自 Kaggle 的[狗&猫图像](https://www.kaggle.com/chetankv/dogs-cats-images)数据集。该数据集是作为狗与猫图像分类挑战的一部分引入的，由属于两个类别(即，狗和猫)的图像组成。训练集包括 8000 幅图像(即，每类 4000 幅图像)，测试集包括 2000 幅图像(即，每类 1000 幅图像)。

在本教程中，我们将使用来自数据集的测试集图像进行推理，并使用来自 Torch Hub 的预训练 FCN 模型生成分段掩码。猫狗数据集简洁易用。此外，它由深度学习社区中的分类模型训练的两个最常见的对象类(即，狗和猫图像)组成，这使得它成为本教程的合适选择。

### **创建配置文件**

我们从讨论`config.py`文件开始，它包含我们将在教程中使用的参数配置。

```py
# import the necessary packages
import os

# define gpu or cpu usage
DEVICE = "cpu"

# define the root directory followed by the test dataset paths
BASE_PATH = "dataset"
TEST_PATH = os.path.join(BASE_PATH, "test_set")

#define pre-trained model name and number of classes it was trained on
MODEL = ["fcn_resnet50", "fcn_resnet101"]
NUM_CLASSES = 21

# specify image size and batch size
IMAGE_SIZE = 224
BATCH_SIZE = 4

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the input image and output segmentation
# mask visualizations
SAVE_IMAGE_PATH = os.path.join(BASE_OUTPUT, "image_samples.png")
SEGMENTATION_OUTPUT_PATH = os.path.sep.join([BASE_OUTPUT,
	"segmentation_output.png"])
```

我们首先在**线 2** 上导入必要的包，其中包括用于文件处理功能的`os`模块。然后，在**第 5 行**上，我们定义了将用于计算的`DEVICE`。请注意，由于我们将使用来自 Torch Hub 的预训练 FCN 模型进行推理，我们将设备设置为 CPU，如图所示。

在**的第 8** 行，我们定义了`BASE_PATH`参数，它指向存储数据集的根文件夹的位置。此外，我们在**第 9 行**上定义了`TEST_PATH`参数，它指向我们的测试集在根数据集文件夹中的位置。

接下来，我们定义`MODEL`参数，它决定了我们将用来执行分割任务的 FCN 模型(**第 12 行**)。请注意，火炬中心为我们提供了访问具有不同分类主干的 FCN 模型的途径。例如，`fcn_resnet50`对应于具有 ResNet50 分类主干的预训练 FCN 模型，而`fcn_resnet101`对应于具有 ResNet101 分类主干的预训练 FCN 模型。

Torch Hub 上托管的 FCN 模型在 COCO train2017 的一个子集上进行预训练，在 Pascal 视觉对象类(VOC)数据集中存在 20 个类别。这使得包括背景类在内的类别总数为 21 个。在**第 13 行，**我们定义了火炬中心 FCN 模型预训练的总类别数(即 21)。

此外，我们定义我们将输入到我们的模型(**行 16** )的图像的空间维度(即`IMAGE_SIZE`)以及我们将用于加载我们的图像样本(**行 17** )的`BATCH_SIZE`。

最后，我们在**行 20** 上定义到输出文件夹(即`BASE_OUTPUT`)的路径，并在**行 24 和 25** 上定义用于存储输入图像可视化(即`SAVE_IMG_PATH`)和最终分割输出图(即`SEGMENTATION_OUTPUT_PATH`)的相应路径。

### **使用 FCN 模型的图像分割**

现在，我们已经定义了参数配置，我们准备设置项目管道，并查看预训练的 FCN 模型的运行情况。

正如在本系列前面的教程中所讨论的，Torch Hub 为访问模型提供了一个简单的 API。它可以导入预先训练好的模型，用于各种计算机视觉任务，从分类、定位、深度估计到生成建模。在这里，我们将更进一步，学习从 Torch Hub 导入和使用细分模型。

让我们从项目目录的 pyimagesearch 文件夹中打开`utils.py`文件，并从定义函数开始，这些函数将帮助我们从我们的 FCN 分割模型中绘制和可视化我们的分割任务预测。

```py
# import the necessary packages
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_segmentation_masks(allClassMask, images, numClasses,
	inverseTransforms, device):	
	# convert to boolean masks and batch dimension first format
	booleanMask  = (
		allClassMask == torch.arange(numClasses, device=device)[:, None, None, None])
	booleanMask = booleanMask.transpose(1,0)
	# initialize list to store our output masks
	outputMasks = []

	# loop over all images and corresponding boolean masks
	for image, mask in zip(images, booleanMask):
		# plot segmentation masks over input images
		outputMasks.append(
			draw_segmentation_masks(
				(inverseTransforms(image) * 255).to(torch.uint8),
				masks=mask,
				alpha=0.6
			)
		)

	# return segmentation plots
	return outputMasks
```

在**的第 2-6 行**上，我们首先导入必要的包，其中包括来自`torchvision.utils`的用于分割掩模可视化的`draw_segmentation_masks`函数(**第 2 行**)、来自`torchvision.transforms`的用于图像格式转换操作的功能模块(**第 3 行**)和`matplotlib`库(**第 4 行**)，以创建和可视化我们的图，我们将在后面详细讨论。最后，我们还导入了用于张量和数组操作的 Numpy 和 PyTorch 库(**第 5 行和第 6 行**)。

我们现在准备定义我们的助手函数来可视化来自我们的 FCN 分割模型的分割掩码输出。

我们定义了`visualize_segmentation_masks()`函数(**第 8-29 行**)，它绘制了输入图像上的预测遮罩，每个类像素用不同的颜色表示，我们将在后面看到。

该函数将按像素的类级分割掩码(即`allClassMask`)、我们想要分割的输入图像(即`images`)、我们的火炬中心 FCN 模型已被训练的类的总数(即`numClasses`)、将图像转换回非标准化形式所需的逆变换(即`inverseTransforms`)以及我们将用于计算的设备(即`device`)作为输入，如**第 8 行和第 9 行**所示。

在**第 11 行和第 12 行**，我们开始为数据集中的每个类创建一个布尔分段掩码，使用输入的类级分段掩码(即`allClassMask`)。这对应于布尔掩码的数量`numClasses`，存储在`booleanMask`变量(**行 11** )中。

使用条件语句`allClassMask == torch.arange(numClasses, device=device)[:, None, None, None])`创建布尔掩码。

语句的左侧(LHS)就是我们的类级分段掩码，维度为`[batch_size, height=IMG_SIZE,width=IMG_SIZE]`。另一方面，右侧(RHS)创建一个 dim 张量`[numClasses,1,1,1]`，其中条目`0, 1, 2, …, (numClasses-1)`按顺序排列。

为了匹配语句两边的维度，RHS 张量自动在空间上(即高度和宽度)传播到维度`[numClasses,batch_size,height=IMG_SIZE,width=IMG_SIZE]`。此外，LHS `allClassMask`在频道维度播放，有`numClasses`个频道，最终形状为`[numClasses,batch_size,height=IMG_SIZE,width=IMG_SIZE]`。

最后，执行条件语句，这给了我们`numClasses`个对应于每个类的布尔分段掩码。然后掩码被存储在`booleanMask`变量中。

注意**第 12 行**一开始形象化有点复杂。为了理解**12 号线**的工作原理，我们从一个简单的例子开始。我们将使用**图 4** 来更好地理解这个过程。

假设我们有一个具有三个类别(类别 0、类别 1 和类别 2)的分割任务，并且在我们的批处理中有一个输入图像(即`batch_size, bs=1`)。如图所示，我们有一个分段掩码`S`，它是我们的分段模型的预测输出，代表我们示例中第 12 行**的 LHS。注意`S`是一个逐像素的类级分割掩码，其中每个像素条目对应于该像素所属的类(即 0、1 或 2)。**

另一方面，所示的 RHS 是具有条目 0、1、2 的 dim `[numClasses = 3,1,1,1]`的张量。

如图所示，掩码`S`在信道维度上广播为`[numClasses=3,bs=1,height, width]`的形状。此外，RHS 张量在空间上传播(即，高度和宽度)以具有最终形状`[numClasses=3,bs=1,height, width]`。

注意，该语句的输出是对应于每个类的布尔分段掩码的数量。

我们现在继续解释`visualize_segmentation_masks()`函数。

在**第 13** 行，我们转置到从**第 12** 行输出的`booleanMask`，将其转换为形状为`[batch_size,numClasses, height=IMG_SIZE,width=IMG_SIZE]`的批量尺寸优先格式。

接下来，我们在第 15 行的**上创建一个空列表`outputMasks`来存储每幅图像的最终分割蒙版。**

在**第 18 行**上，我们开始遍历所有图像及其对应的布尔遮罩，如图所示。对于每个(`image`，`mask`)对，我们将输入`image`和`mask`传递给`draw_segmentation_masks()`函数(**第 21-26 行**)。

该函数在输入`image`上用不同的颜色覆盖每个类的布尔掩码。它还采用一个阿尔法参数，该参数是范围`[0, 1]`内的一个值，并且对应于当掩模覆盖在输入图像上时的透明度值(**行 24** )。

注意，`draw_segmentation_masks()`函数期望输入图像采用范围`[0, 255]`和`uint8`格式。为了实现这一点，我们使用`inverseTransforms`函数对我们的图像进行非规格化，并将其转换到范围`[0, 1]`，然后乘以 255，以获得范围`[0, 255]` ( **第 22 行**)中的最终像素值。此外，我们还按照函数的预期将图像的数据类型设置为`torch.uint8`。

最后，在**第 29** 行，我们返回输出分段掩码列表，如下所示。

现在我们已经定义了我们的可视化助手函数，我们准备设置我们的数据管道，并使用我们的 FCN 模型来完成图像分割的任务。

让我们打开`predict.py`文件开始吧。

```py
# USAGE
# python predict.py

# import the necessary packages
from pyimagesearch import config
from pyimagesearch import utils
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os
```

在**第 5-12 行，**我们导入必要的包和模块，包括我们的`config`文件(**第 5 行**)和来自`pyimagesearch`文件夹(**第 6 行**)的用于可视化助手功能的`utils`模块。

我们还从`torchvision.datasets`模块导入了`ImageFolder`类来创建我们的数据集(**第 7 行**)，导入了`save_image`函数来保存我们的可视化绘图(**第 8 行**)，导入了`torch.utils.data`的`DataLoader`类来访问 PyTorch 提供的特定于数据的功能，以建立我们的数据加载管道(**第 9 行**)。

最后，我们从`torchvision`导入`transforms`模块来应用图像转换，同时加载图像(**第 10 行**)以及 PyTorch 和 os 库，用于基于张量和文件处理的功能(**第 11 行和第 12 行**)。

```py
# create image transformations and inverse transformation
imageTransforms = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	)])
imageInverseTransforms = transforms.Normalize(
	mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
	std=[1/0.229, 1/0.224, 1/0.225]
)

# initialize dataset and dataloader
print("[INFO] creating data pipeline...")
testDs = ImageFolder(config.TEST_PATH, imageTransforms)
testLoader = DataLoader(testDs, shuffle=True,
	batch_size=config.BATCH_SIZE)

# load the pre-trained FCN segmentation model, flash the model to
# the device, and set it to evaluation mode
print("[INFO] loading FCN segmentation model from Torch Hub...")
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load("pytorch/vision:v0.10.0", config.MODEL[0],
	pretrained=True)
model.to(config.DEVICE)
model.eval()

# initialize iterator and grab a batch from the dataset
batchIter = iter(testLoader)
print("[INFO] getting the test data...")
batch = next(batchIter)

# unpack images and labels and move to device
(images, labels) = (batch[0], batch[1])
images = images.to(config.DEVICE)

# initialize a empty list to store images
imageList =[]

# loop over all images
for image in images:
	# add de-normalized images to the list
	imageList.append(
		(imageInverseTransforms(image) * 255).to(torch.uint8)
	)
```

现在我们已经导入了基本的包，是时候设置我们的图像转换了。

我们定义了加载输入图像时想要应用的变换，并在第 15-21 行的**函数的帮助下合并它们。我们的`imageTransforms`包括:**

*   `Resize()`:允许我们将图像调整到我们的模型可以接受的特定输入尺寸(即`config.IMAGE_SIZE`、`config.IMAGE_SIZE`)
*   `ToTensor()`:使我们能够将输入图像转换为 PyTorch 张量，并将输入的 PIL 图像从最初的`[0, 255]`转换为`[0, 1]`。
*   `Normalize()`:它有两个参数，即均值和标准差(即分别为`mean`和`std`)，使我们能够通过减去均值并除以给定的标准差来归一化图像。注意，我们使用 ImageNet 统计数据来标准化图像。

此外，在**的第 22-25 行**，我们定义了一个逆变换(即`imageInverseTransforms`)，它只是执行上面定义的规格化变换的相反操作。当我们想要将图像非规格化回范围`[0, 1]`以便可视化时，这将很方便。

我们现在准备使用 PyTorch 构建我们的数据加载管道。

在**第 29 行**，我们使用`ImageFolder`功能为我们的测试集图像创建 PyTorch 数据集，我们将使用它作为分割任务的输入。`ImageFolder`将测试集的路径(即`config.TEST_PATH`)和我们想要应用于图像的变换(即`imageTransforms`)作为输入。

在**第 30 行和第 31 行**，我们通过将我们的测试数据集(即`testDs`)传递给 PyTorch `DataLoader`类来创建我们的数据加载器(即`testLoader`)。我们将`shuffle`参数`True`保存在数据加载器中，因为我们想在每次运行脚本时处理一组不同的混洗图像。此外，我们使用`config.BATCH_SIZE`来定义`batch_size`参数，以确定数据加载器输出的单个批次中的图像数量。

既然我们已经构建并定义了数据加载管道，我们将从 Torch Hub 初始化我们预先训练的 FCN 细分模型。

在第 37 行的**上，我们使用`torch.hub.load`函数来加载我们预先训练好的 FCN 模型。请注意，该函数采用以下参数:**

*   模型存储的位置(即`pytorch/vision:v0.10.0`)
*   我们想要加载的模型的名称(即`config.MODEL[0]`对应于带有 ResNet50 编码器的 FCN 模型，或者`config.MODEL[1]`对应于带有 ResnNet101 编码器的 FCN 模型)
*   `pretrained`参数，当设置为 True 时，指示 Torch Hub API 下载所选模型的预训练权重并加载它们。

最后，我们使用`to()`函数将我们的模型传输到`config.DEVICE`，在提到的设备上注册我们的模型及其参数(**第 39 行**)。由于我们将使用我们预先训练的模型进行推理，我们在**行 40** 上将模型设置为`eval()`模式。

在完成我们的模型定义之后，现在是时候从我们的数据加载器访问样本，并查看我们的火炬中心 FCN 细分模型的运行情况。

我们首先使用第 43 行的**方法将`testLoader` iterable 转换为 python 迭代器。这允许我们在`next()`方法( **Line 45** )的帮助下简单地迭代我们的数据集批次，正如在[之前关于使用 PyTorch](https://pyimagesearch.com/2021/10/04/image-data-loaders-in-pytorch/) 加载数据的教程中详细讨论的。**

由于我们批次中的每个数据样本都是一个形式为`(images, labels)`的元组，我们在**第 48 行**解包测试图像(即`batch[0]`)和相应的标签(即`batch[1]`)。然后，我们将这批图像传输到我们模型的设备，由第**行 49** 上的`config.DEVICE`定义。

在**第 55-59 行**上，我们通过迭代`images`张量中的每个图像并将其转换为范围`[0, 255]`内的像素值来创建一个`imageList`。为了实现这一点，我们使用`inverseTransforms`函数对我们的图像进行非规格化，并将其转换到范围`[0, 1]`，然后乘以 255，以获得范围`[0, 255]`中的最终像素值。

此外，我们还将图像的数据类型设置为`torch.uint8`，这是我们的可视化函数所期望的。

```py
# create the output directory if not already exists
if not os.path.exists(config.BASE_OUTPUT):
	os.makedirs(config.BASE_OUTPUT)

# turn off auto grad
with torch.no_grad():
	# compute prediction from the model
	output = model(images)["out"]

# convert predictions to class probabilities
normalizedMasks = torch.nn.functional.softmax(output, dim=1)

# convert to pixel-wise class-level segmentation masks
classMask = normalizedMasks.argmax(1)

# visualize segmentation masks
outputMasks = utils.visualize_segmentation_masks(
	allClassMask=classMask,
	images=images,
	numClasses=config.NUM_CLASSES,
	inverseTransforms=imageInverseTransforms,
	device=config.DEVICE
)

# convert input images and output masks to tensors
inputImages = torch.stack(imageList)
generatedMasks = torch.stack(outputMasks)

# save input image visualizations and the mask visualization
print("[INFO] saving the image and mask visualization to disk...")
save_image(inputImages.float() / 255,
	config.SAVE_IMAGE_PATH, nrow=4, scale_each=True,
	normalize=True)
save_image(generatedMasks.float() / 255,
	config.SEGMENTATION_OUTPUT_PATH, nrow=4, scale_each=True,
	normalize=True)
```

现在是时候看看我们的预训练 FCN 模型的行动，并使用它来生成我们的批处理输入图像的分割掩模。

我们首先确保我们将存储分割预测的输出目录存在，如果不存在，我们创建它，如第**62 和 63** 行所示。

由于我们仅使用预训练的模型进行推断，我们指导 PyTorch 在`torch.no_grad()`的帮助下关闭梯度计算，如**第 66 行**所示。然后，我们将图像传递给我们预先训练好的 FCN 模型，并将输出图像存储在维度为`[batch_size, config.NUM_CLASSES,config.IMAGE_SIZE,config.IMAGE_SIZE]` ( **第 68 行**)的变量`output`中。

如第 71 行**所示，我们使用`softmax`函数将来自 FCN 模型的预测分割转换为分类概率。最后，在**第 74 行**上，我们通过在我们的`normalizedMasks`的类维度上使用`argmax()`函数为每个像素位置选择最可能的类来获得逐像素的类级分割掩模。**

然后，我们使用来自`utils`模块的`visualize_segmentation_masks()`函数来可视化我们的`classMask` ( **第 77-83 行**)并将输出分段掩码存储在`outputMasks`变量中。(**第 77 行**)。

在第**行第 86 和 87** 行，我们通过使用`torch.stack()`函数将各自列表中的条目进行堆栈，将输入图像列表`imageList`和最终列表`outputMasks`转换为张量。最后，我们使用来自`torchvision.utils`的`save_image`函数来保存我们的`inputImages`张量(**第 91-93 行**)和`generatedMasks`张量(**第 94-96 行**)。

请注意，我们将张量转换为`float()`，并通过用`save_image`函数将它们除以`255`来归一化它们。

此外，我们还注意到，`save_image`函数将我们想要保存图像的路径(即`config.SAVE_IMG_PATH`和`config.SEGMENTATION_OUTPUT_PATH`)、单行中显示的图像数量(即`nrow=4`)以及另外两个布尔参数(即`scale_each`和`normalize`)作为输入，它们对张量中的图像值进行缩放和归一化。

设置这些参数可确保图像在`save_image`功能要求的特定范围内标准化，以获得最佳的可视化结果。

**图 5** 在左侧示出了我们批次中的输入图像(即`inputImages`)的可视化，并且在右侧示出了来自我们预训练的 FCN 分割模型的对应预测分割掩模(即`generatedMasks`),用于四个不同的批次。

请注意，我们的 FCN 模型可以在所有情况下正确识别对应于类别猫(青色)和狗(绿色)的像素。此外，我们还注意到，即使一个对象的多个实例(比如，第二行第三幅图像中的猫)出现在一幅图像中，我们的模型也表现得相当好。

此外，我们观察到，在图像包含人形和狗/猫(第 4 行，第二幅图像)的情况下，我们的模型可以有效地分割我们的人(深蓝色)。

这可以主要归因于我们的 FCN 模型已经预先训练的 21 个类别，包括类别`cat`、`dog`和`person`。然而，假设我们想要分割不包括在 21 个类别中的对象。在这种情况下，我们可以使用来自 Torch Hub 的预训练权重来初始化 FCN 模型，并使用迁移学习范式来微调我们想要细分的新类别，如在[之前的帖子](https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/)中所讨论的。

## **总结**

在本教程中，我们研究了另一种图像分割方法，该方法依赖于利用从图像分类任务中学到的特征，并重新利用它们来增强分割预测。我们还了解了将预训练分类模型转换为用于分割的完全卷积网络模型所需的架构变化，这使我们能够高效地处理不同大小的输入图像，并实时输出准确的分割预测。

此外，我们使用 Torch Hub API 来导入预训练的 FCN 分割模型，并使用它来实时预测自定义输入图像的分割输出。

### **引用信息**

**钱德霍克，S.** “火炬中心系列#6:图像分割”， *PyImageSearch* ，2022，【https://pyimg.co/uk1oa】T4

```py
@article{Chandhok_2022_THS6,
  author = {Shivam Chandhok},
  title = {Torch Hub Series \#6: Image Segmentation},
  journal = {PyImageSearch},
  year = {2022},
  note = {https://pyimg.co/uk1oa},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！【T2****