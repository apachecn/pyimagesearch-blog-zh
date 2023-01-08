# Torch Hub 系列#3: YOLOv5 和 SSD——对象检测模型

> 原文：<https://pyimagesearch.com/2022/01/03/torch-hub-series-3-yolov5-and-ssd-models-on-object-detection/>

在我的童年，电影《间谍小子》是我最喜欢看的电视节目之一。看到我这个年龄的孩子使用未来的小工具来拯救世界并赢得胜利可能是一个常见的比喻，但看起来仍然很有趣。在像喷气背包和自动驾驶汽车这样的东西中，我最喜欢的是智能太阳镜，它可以识别你周围的物体和人(它还可以兼作双目望远镜)，有点像**图 1** 。

可以理解的是，这些小玩意在现实生活中的概念在当时很难理解。但是，现在已经到了 2022 年，一家自动驾驶汽车公司([特斯拉](https://www.tesla.com/))已经处于电机行业的顶端，从实时视频中检测物体简直易如反掌！

所以今天，除了理解一个年轻的我的狂热梦想，我们将看到 PyTorch Hub 如何使探索这些领域变得容易。

在本教程中，我们将学习 YOLOv5 和 SSD300 等模型背后的直觉，并使用 Torch Hub 驾驭它们的力量。

本课是关于火炬中心的 6 部分系列的第 3 部分:

1.  [*火炬中心系列# 1:*](https://pyimagesearch.com/2021/12/20/torch-hub-series-1-introduction-to-torch-hub/)火炬中心介绍
2.  [*火炬枢纽系列#2: VGG 和雷斯内特*](https://pyimagesearch.com/2021/12/27/torch-hub-series-2-vgg-and-resnet/)
3.  *火炬轮毂系列#3: YOLOv5 和 SSD*——*物体检测模型*(本教程)
4.  *火炬轮毂系列# 4:*—*甘上模*
5.  *火炬轮毂系列# 5:MiDaS*——*深度估计模型*
6.  *火炬中枢系列#6:图像分割*

要学习如何使用 YOLOv5 和 SSD300， ***继续阅读。***

## **火炬轮毂系列#3: YOLOv5** **和****SSD——物体检测模型**

### **物体检测一目了然**

乍一看，物体检测无疑是一个非常诱人的领域。让机器识别图像中某个对象的确切位置，让我相信我们离实现模仿人脑的梦想又近了一步。但即使我们把它放在一边，它在当今世界也有各种各样的重要用法。从人脸检测系统到帮助自动驾驶汽车安全导航，这样的例子不胜枚举。但是它到底是如何工作的呢？

实现物体检测的方法有很多，以机器学习为核心思想。例如，在这篇关于在 PyTorch 中从头开始训练一个物体检测器的[博客文章](https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/)中，我们有一个简单的架构，它接收图像作为输入并输出 5 样东西；检测到的对象的类别以及对象边界框的高度和宽度的起始值和结束值。

本质上，我们获取带注释的图像，并通过一个简单的输出大小为 5 的 CNN 传递它们。因此，就像机器学习中开发的每一个新事物一样，更复杂和错综复杂的算法随之而来，以改进它。

请注意，如果您考虑我在上一段中提到的方法，它可能只适用于检测单个对象的图像。然而，当多个物体在一张图像中时，这几乎肯定会遇到障碍。因此，为了解决这一问题以及效率等其他限制，我们转向 YOLO(v1)。

YOLO，或“[你只看一次](https://arxiv.org/abs/1506.02640)”(2015)，介绍了一种巧妙的方法来解决简单的 CNN 探测器的缺点。我们将每幅图像分割成一个 *S* × *S* 网格，得到每个细胞对应的物体位置。当然，有些单元格不会有任何对象，有些则会出现在多个单元格中。看一下**图二**。

了解完整图像中对象的中点、高度和宽度非常重要。然后，每个像元将输出一个概率值(像元中有一个对象的概率)、检测到的对象类以及像元特有的边界框值。

即使每个单元只能检测一个对象，多个单元的存在也会使约束无效。结果可以在**图 3** 中看到。

尽管取得了巨大的成果，但 YOLOv1 有一个重大缺陷；图像中对象的接近程度经常会使模型错过一些对象。自问世以来，已经出版了几个后继版本，如 YOLOv2、YOLOv3 和 YOLOv4，每一个都比前一个版本更好更快。这就把我们带到了今天的焦点之一，YOLOv5。

YOLOv5 的创造者 Glenn Jocher 决定不写论文，而是通过 GitHub 开源该模型。最初，这引起了很多关注，因为人们认为结果是不可重复的。然而，这种想法很快被打破了，今天，YOLOv5 是火炬中心展示区的官方最先进的模型之一。

要了解 YOLOv5 带来了哪些改进，我们还得回到 YOLOv2。其中，YOLOv2 引入了锚盒的概念。一系列预定的边界框、锚框具有特定的尺寸。根据训练数据集中的对象大小选择这些框，以捕捉要检测的各种对象类的比例和纵横比。网络预测对应于锚框而不是边界框本身的概率。

但在实践中，悬挂 YOLO 模型通常是在 COCO 数据集上训练的。这导致了一个问题，因为自定义数据集可能没有相同的锚框定义。YOLOv5 通过引入自动学习锚盒来解决这个问题。它还利用镶嵌增强，混合随机图像，使您的模型擅长识别较小比例的对象。

今天的第二个亮点是用于物体检测的固态硬盘或[单次多盒探测器](https://arxiv.org/abs/1512.02325)型号。SSD300 最初使用 VGG 主干进行娴熟的特征检测，并利用了 [Szegedy](https://arxiv.org/pdf/1412.1441.pdf) 在 MultiBox 上的工作，这是一种快速分类不可知边界框坐标建议的方法，启发了 SSD 的边界框回归算法。**图 4** 展示了 SSD 架构。

受 [inception-net](https://arxiv.org/pdf/1409.4842.pdf) 的启发，Szegedy 创建的多盒架构利用了多尺度卷积架构。Multibox 使用一系列正常的卷积和`1×1`过滤器(改变通道大小，但保持高度和宽度不变)来合并多尺度边界框和置信度预测模型。

SSD 以利用多尺度特征地图而不是单一特征地图进行检测而闻名。这允许更精细的检测和更精细的预测。使用这些特征图，生成了用于对象预测的锚框。

它问世时就已经超越了它的同胞，尤其是在速度上。今天我们将使用 Torch Hub 展示的 SSD，它使用 ResNet 而不是 VGG 网络作为主干。此外，其他一些变化，如根据现代卷积物体探测器论文的[速度/精度权衡移除一些层，也被应用到模型中。](https://arxiv.org/abs/1611.10012)

今天，我们将学习如何使用 Torch Hub 利用这些模型的功能，并使用我们的自定义数据集对它们进行测试！

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

从这里，看一下目录结构:

```py
!tree . 
.
├── output
│   ├── ssd_output
│   │   └── ssd_output.png
│   └── yolo_output
│       └── yolo_output.png
├── pyimagesearch
│   ├── config.py
│   ├── data_utils.py
├── ssd_inference.py
└── yolov5_inference.py
```

首先，我们有`output`目录，它将存放我们将从每个模型获得的输出。

在`pyimagesearch`目录中，我们有两个脚本:

*   这个脚本包含了项目的端到端配置管道
*   这个脚本包含了一些用于数据处理的帮助函数

在主目录中，我们有两个脚本:

*   `ssd_inference.py`:这个脚本包含自定义映像的 SSD 模型推断。
*   `yolov5_inference.py`:这个脚本包含了定制图像的 YOLOv5 模型推理。

### **下载数据集**

第一步是根据我们的需求配置数据集。像在之前的教程中一样，我们将使用来自 Kaggle 的[狗&猫图像](https://www.kaggle.com/chetankv/dogs-cats-images)数据集，因为它相对较小。

```py
$ mkdir ~/.kaggle
$ cp <path to your kaggle.json> ~/.kaggle/
$ chmod 600 ~/.kaggle/kaggle.json
$ kaggle datasets download -d chetankv/dogs-cats-images
$ unzip -qq dogs-cats-images.zip
$ rm -rf "/content/dog vs cat"
```

要使用数据集，您需要有自己独特的`kaggle.json`文件来连接到 Kaggle API ( **第 2 行**)。**线 3** 上的`chmod 600`命令给予用户读写文件的完全权限。

接下来是`kaggle datasets download`命令(**第 4 行**)允许你下载他们网站上的任何数据集。最后，解压文件并删除不必要的附加内容(**第 5 行和第 6 行**)。

让我们转到配置管道。

### **配置先决条件**

在`pyimagesearch`目录中，您会发现一个名为`config.py`的脚本。这个脚本将包含我们项目的完整的端到端配置管道。

```py
# import the necessary packages
import torch
import os

# define the root directory followed by the test dataset paths
BASE_PATH = "dataset"
TEST_PATH = os.path.join(BASE_PATH, "test_set")

# specify image size and batch size
IMAGE_SIZE = 300
PRED_BATCH_SIZE = 4

# specify threshold confidence value for ssd detections
THRESHOLD = 0.50

# determine the device type 
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# define paths to save output 
OUTPUT_PATH = "output"
SSD_OUTPUT = os.path.join(OUTPUT_PATH, "ssd_output")
YOLO_OUTPUT = os.path.join(OUTPUT_PATH, "yolo_output")
```

在第 6 行的**上，我们有指向数据集目录的`BASE_PATH`变量。由于我们将只使用模型来运行推理，我们将只需要测试集(**第 7 行**)。**

在**第 10 行**，我们有一个名为`IMAGE_SIZE`的变量，设置为`300`。这是 SSD 模型的一个要求，因为它是根据大小`300 x 300`图像训练的。预测批量大小设置为`4` ( **第 11 行**)，但是鼓励读者尝试不同的大小。

接下来，我们有一个名为`THRESHOLD`的变量，它将作为 SSD 模型结果的置信度值阈值，即只保留置信度值大于阈值的结果(**第 14 行**)。

建议您为今天的项目准备一个兼容 CUDA 的设备( **Line 17** )，但是由于我们不打算进行任何繁重的训练，CPU 应该也能正常工作。

最后，我们创建了路径来保存从模型推断中获得的输出(**第 20-22 行**)。

### **为数据管道创建辅助函数**

在我们看到运行中的模型之前，我们还有一项任务；为数据处理创建辅助函数。为此，转到位于`pyimagesearch`目录中的`data_utils.py`脚本。

```py
# import the necessary packages
from torch.utils.data import DataLoader

def get_dataloader(dataset, batchSize, shuffle=True):
	# create a dataloader and return it
	dataLoader= DataLoader(dataset, batch_size=batchSize,
		shuffle=shuffle)
	return dataLoader
```

`get_dataloader` ( **第 4 行)**函数接受数据集、批量大小和随机参数，返回一个`PyTorch Dataloader` ( **第 6 行和第 7 行**)实例。`Dataloader`实例解决了许多麻烦，这需要为巨大的数据集编写单独的定制生成器类。

```py
def normalize(image, mean=128, std=128):
    # normalize the SSD input and return it 
    image = (image * 256 - mean) / std
    return image
```

脚本中的第二个函数`normalize`，专门用于我们将发送到 SSD 模型的图像。它将`image`、平均值和标准偏差值作为输入，对它们进行归一化，并返回归一化图像(**第 10-13 行**)。

### **在 YOLOv5 上测试自定义图像**

先决条件得到满足后，我们的下一个目的地是`yolov5_inference.py`。我们将准备我们的自定义数据，并将其提供给 YOLO 模型。

```py
# import necessary packages
from pyimagesearch.data_utils import get_dataloader
import pyimagesearch.config as config
from torchvision.transforms import Compose, ToTensor, Resize
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import cv2
import os

# initialize test transform pipeline
testTransform = Compose([
	Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), ToTensor()])

# create the test dataset
testDataset = ImageFolder(config.TEST_PATH, testTransform)

# initialize the test data loader
testLoader = get_dataloader(testDataset, config.PRED_BATCH_SIZE)
```

首先，我们在第 16 行和第 17 行上创建一个 PyTorch 转换实例。使用 PyTorch 的另一个名为`ImageFolder`的恒星数据实用函数，我们可以直接创建一个 PyTorch 数据集实例(**第 20 行**)。然而，为了使这个函数工作，我们需要数据集与这个项目具有相同的格式。

一旦我们有了数据集，我们就将它传递给预先创建的`get_dataloader`函数，以获得一个类似 PyTorch Dataloader 实例的生成器(**第 23 行**)。

```py
# initialize the yolov5 using torch hub
yoloModel = torch.hub.load("ultralytics/yolov5", "yolov5s")

# initialize iterable variable
sweeper = iter(testLoader)

# initialize image 
imageInput = []

# grab a batch of test data
print("[INFO] getting the test data...")
batch = next(sweeper)
(images, _) = (batch[0], batch[1])

# send the images to the device
images = images.to(config.DEVICE) 
```

在**线 26** 上，使用焊炬集线器调用 YOLOv5。简单回顾一下，`torch.hub.load`函数将 GitHub 存储库和所需的入口点作为它的参数。入口点是函数的名称，在这个名称下，模型调用位于所需存储库的`hubconf.py`脚本中。

下一步对我们的项目至关重要。我们有很多方法可以从数据集中随机获取一批图像。然而，当我们处理越来越大的数据集时，依靠循环获取数据的效率会降低。

记住这一点，我们将使用一种比循环更有效的方法。我们可以选择使用第 29 行的**可迭代变量随机抓取数据。所以每次你运行第 36** 行上的命令，你会得到不同的一批数据。

在**行 40** 上，我们将抓取的数据加载到我们将用于计算的设备上。

```py
# loop over all the batch 
for index in range(0, config.PRED_BATCH_SIZE):
	# grab each image
	# rearrange dimensions to channel last and
	# append them to image list
	image = images[index]
	image = image.permute((1, 2, 0))
	imageInput.append(image.cpu().detach().numpy()*255.0)

# pass the image list through the model
print("[INFO] getting detections from the test data...")
results = yoloModel(imageInput, size=300)
```

在**第 43 行**，我们有一个循环，在这个循环中我们检查抓取的图像。然后，取每幅图像，我们重新排列维度，使它们成为通道最后的，并将结果添加到我们的`imageInput`列表中(**第 47-49 行**)。

接下来，我们将列表传递给 YOLOv5 模型实例(**第 53 行**)。

```py
# get random index value
randomIndex = random.randint(0,len(imageInput)-1)

# grab index result from results variable
imageIndex= results.pandas().xyxy[randomIndex]

# convert the bounding box values to integer
startX = int(imageIndex["xmin"][0])
startY = int(imageIndex["ymin"][0])
endX = int(imageIndex["xmax"][0])
endY = int(imageIndex["ymax"][0])

# draw the predicted bounding box and class label on the image
y = startY - 10 if startY - 10 > 10 else startY + 10
cv2.putText(imageInput[randomIndex], imageIndex["name"][0],
	(startX, y+10), cv2.FONT_HERSHEY_SIMPLEX,0.65, (0, 255, 0), 2)
cv2.rectangle(imageInput[randomIndex],
	(startX, startY), (endX, endY),(0, 255, 0), 2)

# check to see if the output directory already exists, if not
# make the output directory
if not os.path.exists(config.YOLO_OUTPUT):
    os.makedirs(config.YOLO_OUTPUT)

# show the output image and save it to path
plt.imshow(imageInput[randomIndex]/255.0)

# save plots to output directory
print("[INFO] saving the inference...")
outputFileName = os.path.join(config.YOLO_OUTPUT, "output.png")
plt.savefig(outputFileName)
```

第 56 行**上的`randomIndex`变量将作为我们选择的索引，同时访问我们将要显示的图像。使用其值，在**行 59** 上访问相应的边界框结果。**

我们在**行 62-65** 上分割图像的特定值(起始 X、起始 Y、结束 X 和结束 Y 坐标)。我们必须使用`imageIndex["Column Name"][0]`格式，因为`results.pandas().xyxy[randomIndex]`返回一个数据帧。假设在给定的图像中有一个检测，我们必须通过调用所需列的第零个索引来访问它的值。

使用这些值，我们分别使用`cv2.putText`和`cv2.rectangle`在第 69-72 行的**上绘制标签和边界框。给定坐标，这些函数将获取所需的图像并绘制出所需的必需品。**

最后，在使用`plt.imshow`绘制图像时，我们必须缩小数值(**第 80 行**)。

这就是 YOLOv5 在自定义图像上的结果！我们来看看**图 6-8** 中的一些结果。

正如我们从结果中看到的，预训练的 YOLOv5 模型在所有图像上定位得相当好。

### **在固态硬盘型号上测试定制映像**

对于 SSD 模型的推理，我们将遵循类似于 YOLOv5 推理脚本中的模式。

```py
# import the necessary packages
from pyimagesearch.data_utils import get_dataloader
from pyimagesearch.data_utils import normalize
from pyimagesearch import config
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import cv2
import os

# initialize test transform pipeline
testTransform = Compose([
	Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), ToTensor()])

# create the test dataset and initialize the test data loader
testDataset = ImageFolder(config.TEST_PATH, testTransform)
testLoader = get_dataloader(testDataset, config.PRED_BATCH_SIZE)

# initialize iterable variable
sweeper = iter(testLoader)

# list to store permuted images
imageInput = []
```

如前所述，我们在**的第 20 和 21 行创建 PyTorch 转换实例。**然后，使用`ImageFolder`实用函数，我们根据需要创建数据集实例，然后在第 24 和 25 行的**上创建`Dataloader`实例。**

可迭代变量`sweeper`在**线 28** 上初始化，以便于访问测试数据。接下来，为了存储我们将要预处理的图像，我们初始化一个名为`imageInput` ( **第 31 行**)的列表。

```py
# grab a batch of test data
print("[INFO] getting the test data...")
batch = next(sweeper)
(images, _ ) = (batch[0], batch[1])

# switch off autograd
with torch.no_grad():
	# send the images to the device
	images = images.to(config.DEVICE) 

	# loop over all the batch 
	for index in range(0, config.PRED_BATCH_SIZE):
		# grab the image, de-normalize it, scale the raw pixel
		# intensities to the range [0, 255], and change the channel
		# ordering from channels first tp channels last
		image = images[index]
		image = image.permute((1, 2, 0))
		imageInput.append(image.cpu().detach().numpy())
```

从 YOLOv5 推理脚本再次重复上面代码块中显示的过程。我们抓取一批数据(**第 35 行和第 36 行**)并循环遍历它们，将每个数据重新排列到最后一个通道，并将它们添加到我们的`imageInput`列表(**第 39-50 行**)。

```py
# call the required entry points
ssdModel = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub",
	"nvidia_ssd")
utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", 
	"nvidia_ssd_processing_utils")

# flash model to the device and set it to eval mode
ssdModel.to(config.DEVICE)
ssdModel.eval()

# new list for processed input
processedInput = []

# loop over images and preprocess them
for image in imageInput:
	image = normalize (image)
	processedInput.append(image)

# convert the preprocessed images into tensors
inputTensor = utils.prepare_tensor(processedInput)
```

在第 53 行和第 54 行上，我们使用`torch.hub.load`函数来:

*   通过调用相应的存储库和入口点名称来调用 SSD 模型
*   根据 SSD 模型的需要，调用一个额外的实用函数来帮助预处理输入图像。

然后将模型加载到我们正在使用的设备中，并设置为评估模式(**第 59 行和第 60 行**)。

在第 63 行的**上，我们创建了一个空列表来保存预处理后的输入。然后，循环遍历这些图像，我们对它们中的每一个进行归一化，并相应地添加它们(**第 66-68 行)**。最后，为了将预处理后的图像转换成所需的张量，我们使用了之前调用的效用函数(**第 71 行**)。**

```py
# turn off auto-grad
print("[INFO] getting detections from the test data...")
with torch.no_grad():
	# feed images to model
	detections = ssdModel(inputTensor)

# decode the results and filter them using the threshold
resultsPerInput = utils.decode_results(detections)
bestResults = [utils.pick_best(results,
	config.THRESHOLD) for results in resultsPerInput]
```

关闭自动梯度，将图像张量输入 SSD 模型(**第 75-77 行**)。

在第 80 行的**上，我们使用来自`utils`的另一个名为`decode_results`的函数来获得对应于每个输入图像的所有结果。现在，由于 SSD 为您提供了 8732 个检测，我们将使用之前在`config.py`脚本中设置的置信度阈值，只保留置信度超过 50%的检测(**第 81 行和第 82 行**)。**

这意味着`bestResults`列表包含边界框值、对象类别和置信度值，对应于它在给出检测输出时遇到的每个图像。这样，这个列表的索引将直接对应于我们的输入列表的索引。

```py
# get coco labels 
classesToLabels = utils.get_coco_object_dictionary()

# loop over the image batch
for image_idx in range(len(bestResults)):
	(fig, ax) = plt.subplots(1)

	# denormalize the image and plot the image
	image = processedInput[image_idx] / 2 + 0.5
	ax.imshow(image)

	# grab bbox, class, and confidence values
	(bboxes, classes, confidences) = bestResults[image_idx]
```

由于我们没有办法将类的整数结果解码成它们对应的标签，我们将借助来自`utils`的另一个函数`get_coco_object_dictionary` ( **第 85 行**)。

下一步是将结果与其对应的图像进行匹配，并在图像上绘制边界框。相应地，使用图像索引抓取相应的图像并将其反规格化(**第 88-93 行**)。

使用相同的索引，我们从结果中获取边界框结果、类名和置信度值(**第 96 行**)。

```py
	# loop over the detected bounding boxes
	for idx in range(len(bboxes)):
		# scale values up according to image size
		(left, bot, right, top) = bboxes[idx ] * 300

		# draw the bounding box on the image
		(x, y, w, h) = [val for val in [left, bot, right - left,
			top - bot]]
		rect = patches.Rectangle((x, y), w, h, linewidth=1,
			edgecolor="r", facecolor="none")
		ax.add_patch(rect)
		ax.text(x, y,
			"{} {:.0f}%".format(classesToLabels[classes[idx] - 1],
			confidences[idx] * 100),
			bbox=dict(facecolor="white", alpha=0.5))
```

由于单个图像可能有多个检测，我们创建一个循环并开始迭代可用的边界框(**行 99** )。边界框结果在 0 和 1 的范围内。所以在解包边界框值时，根据图像的高度和宽度缩放它们是很重要的(**第 101 行**)。

现在，SSD 模型输出左、下、右和上坐标，而不是 YOLOv5s 的起始 X、起始 Y、结束 X 和结束 Y 值。因此，我们必须计算起始 X、起始 Y、宽度和高度，以便在图像上绘制矩形(**行 104-107** )。

最后，我们在第 109-112 行的**函数的帮助下添加对象类名。**

```py
# check to see if the output directory already exists, if not
# make the output directory
if not os.path.exists(config.SSD_OUTPUT):
    os.makedirs(config.SSD_OUTPUT)

# save plots to output directory
print("[INFO] saving the inference...")
outputFileName = os.path.join(config.SSD_OUTPUT, "output.png")
plt.savefig(outputFileName)
```

我们通过将输出图像文件保存到我们之前设置的位置(**行 121 和 120** )来结束脚本，并绘制出我们的图像。

让我们看看脚本的运行情况！

在**图 9-11** 中，我们有 SSD 模型对来自自定义数据集的图像预测的边界框。

在**图 11** 中，SSD 模型设法找出了小狗，这是值得称赞的，它几乎将自己伪装成了它的父母。否则，SSD 模型在大多数图像上表现得相当好，置信度值告诉我们它对其预测的确信程度。

## **总结**

由于物体检测已经成为我们生活的一个主要部分，因此能够获得可以复制高水平研究/行业水平结果的模型对于学习人群来说是一个巨大的福音。

本教程再次展示了 Torch Hub 简单而出色的入口点调用系统，在这里，我们可以调用预训练的全能模型及其辅助助手函数来帮助我们更好地预处理数据。这整个过程的美妙之处在于，如果模型所有者决定将更改推送到他们的存储库中，而不是通过许多过程来更改托管数据，他们需要将他们的更改推送到他们的存储库本身。

随着处理托管数据的整个过程的简化，PyTorch 在与 GitHub 的整个合作过程中取得了成功。这样，即使是调用模型的用户也可以在存储库中了解更多信息，因为它必须是公开的。

我希望本教程可以作为在您的定制任务中使用这些模型的良好起点。指导读者尝试他们的图像或考虑使用这些模型来完成他们的定制任务！

### **引用信息**

**Chakraborty，D.** “火炬中心系列#3: YOLOv5 和 SSD-物体探测模型”， *PyImageSearch* ，2022 年，[https://PyImageSearch . com/2022/01/03/Torch-Hub-Series-3-yolov 5-SSD-物体探测模型/](https://pyimagesearch.com/2022/01/03/torch-hub-series-3-yolov5-and-ssd-models-on-object-detection/)

```py
@article{dev_2022_THS3,
  author = {Devjyoti Chakraborty},
  title = {{Torch Hub} Series \#3: {YOLOv5} and {SSD} — Models on Object Detection},
  journal = {PyImageSearch},
  year = {2022},
  note = {https://pyimagesearch.com/2022/01/03/torch-hub-series-3-yolov5-and-ssd-models-on-object-detection/},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****