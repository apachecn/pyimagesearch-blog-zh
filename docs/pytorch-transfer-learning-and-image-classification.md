# PyTorch:迁移学习和图像分类

> 原文：<https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/>

在本教程中，您将学习如何使用 PyTorch 深度学习库为图像分类执行迁移学习。

本教程是我们关于计算机视觉和深度学习从业者的中级 PyTorch 技术的 3 部分系列中的第 2 部分:

1.  [*py torch 中的图像数据加载器*](https://pyimagesearch.com/2021/10/04/image-data-loaders-in-pytorch/) (上周的教程)
2.  *PyTorch:迁移学习和图像分类*(本教程)
3.  *py torch 分布式培训简介*(下周博文)

如果您不熟悉 PyTorch 深度学习库，我们建议您阅读以下介绍性系列，以帮助您学习基础知识并熟悉 PyTorch 库:

*   [PyTorch:训练你的第一个卷积神经网络(CNN)](https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/)
*   [使用预训练网络的 PyTorch 图像分类](https://pyimagesearch.com/2021/07/26/pytorch-image-classification-with-pre-trained-networks/)
*   [使用预训练网络的 PyTorch 对象检测](https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/)

看完以上教程，你可以回到这里，用 PyTorch 学习迁移学习。

**学习如何用 PyTorch 进行图像分类的迁移学习，** ***继续阅读。***

## **PyTorch:迁移学习和图像分类**

在本教程的第一部分，我们将学习什么是迁移学习，包括 PyTorch 如何让我们进行迁移学习。

然后，我们将配置我们的开发环境，并检查我们的项目目录结构。

从这里，我们将实现几个 Python 脚本，包括:

*   存储重要变量的配置脚本
*   数据集加载器辅助函数
*   一个在磁盘上构建和组织数据集的脚本，这样 PyTorch 的`ImageFolder`和`DataLoader`类就可以很容易地被利用
*   通过特征提取执行基本迁移学习的驱动程序脚本
*   第二个驱动程序脚本通过用全新的、刚刚初始化的 FC 头替换预训练网络的全连接(FC)层头来执行微调
*   最终的脚本允许我们用训练好的模型进行推理

我们今天要复习的内容很多，所以让我们开始吧！

### **什么是迁移学习？**

从头开始训练一个卷积神经网络带来了许多挑战，最明显的是训练网络的数据量**和训练发生的时间量**。****

 ****迁移学习**是一种技术，它允许我们使用为某项任务训练的模型作为不同任务的机器学习模型的起点。

例如，假设在 ImageNet 数据集上为图像分类训练了一个模型。在这种情况下，我们可以采用这个模型，并“重新训练”它去识别它最初*从未*训练去识别的类！

想象一下，你会骑自行车，想骑摩托车。你骑自行车的经验——保持平衡、保持方向、转弯和刹车——将帮助你更快地学会骑摩托车。

这就是迁移学习在 CNN 中的作用。使用迁移学习，您可以通过冻结参数、更改输出层和微调权重来直接使用训练有素的模型。

**本质上，您可以简化整个训练过程，并在很短的时间内获得高精度的模型。**

### 如何用 PyTorch 进行迁移学习？

迁移学习有两种主要类型:

1.  **通过特征提取转移学习:**我们从预训练的网络中移除 FC 层头，并将其替换为 softmax 分类器。这种方法非常简单，因为它允许我们将预训练的 CNN 视为特征提取器，然后将这些特征通过逻辑回归分类器。
2.  **通过微调转移学习:**当应用微调时，我们再次从预训练的网络中移除 FC 层头，但这一次我们构建了一个*全新的、刚刚初始化的 FC 层头*，并将其置于网络的原始主体之上。CNN 主体中的权重被冻结，然后我们训练新的层头(通常具有非常小的学习率)。然后我们可以选择解冻网络的主体，并训练*整个*网络。

第一种方法更容易使用，因为涉及的代码更少，需要调整的参数也更少。然而，第二种方法往往更准确，导致模型更好地概括。

通过特征提取和微调的迁移学习都可以用 PyTorch 实现——我将在本教程的剩余部分向您展示如何实现。

### **配置您的开发环境**

为了遵循这个指南，你需要在你的机器上安装 OpenCV、`imutils`、`matplotlib`和`tqdm`。

幸运的是，所有这些都是 pip 可安装的:

```py
$ pip install opencv-contrib-python
$ pip install torch torchvision
$ pip install imutils matplotlib tqdm
```

**如果你需要帮助为 PyTorch 配置开发环境，我*强烈推荐*你** [**阅读 PyTorch 文档**](https://pytorch.org/get-started/locally/)——py torch 的文档很全面，会让你很快上手并运行。

如果你需要帮助安装 OpenCV，[一定要参考我的 *pip 安装 OpenCV* 教程](https://pyimagesearch.com/2021/07/26/pytorch-image-classification-with-pre-trained-networks/)。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **花卉照片数据集**

让我们来看看 Flowers 数据集，并可视化该数据集中的一些图像。**图 2** 展示了图像的外观。

我们将用于微调实验的数据集是由 TensorFlow 开发团队管理的花卉图像数据集[。](https://www.tensorflow.org/datasets/catalog/tf_flowers)

泰国数据集 3，670 张图片，属于五种不同的花卉品种:

1.  **雏菊:** 633 张图片
2.  **蒲公英:** 898 张图片
3.  **玫瑰:** 641 张图片
4.  **向日葵:** 699 张图片
5.  **郁金香:** 799 张图片

我们的工作是训练一个图像分类模型来识别这些花卉品种中的每一种。我们将通过 PyTorch 应用迁移学习来实现这一目标。

### **项目结构**

我们首先需要回顾我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

从这里，看一下目录结构:

```py
$ tree --dirsfirst --filelimit 10
.
├── flower_photos
│   ├── daisy [633 entries exceeds filelimit, not opening dir]
│   ├── dandelion [898 entries exceeds filelimit, not opening dir]
│   ├── roses [641 entries exceeds filelimit, not opening dir]
│   ├── sunflowers [699 entries exceeds filelimit, not opening dir]
│   ├── tulips [799 entries exceeds filelimit, not opening dir]
│   └── LICENSE.txt
├── output
│   ├── finetune.png
│   └── warmup.png
├── pyimagesearch
│   ├── config.py
│   └── create_dataloaders.py
├── build_dataset.py
├── feature_extraction_results.png
├── fine_tune.py
├── fine_tune_results.png
├── inference.py
└── train_feature_extraction.py
```

目录包含了我们的花卉图片集。

我们将在这个花卉数据集上训练我们的模型。然后,`output`目录将被我们的训练/验证图填充。

在`pyimagesearch`模块中，我们有两个 Python 文件:

1.  包含在我们的驱动脚本中使用的重要配置变量。
2.  `create_dataloaders.py`:实现`get_dataloader`助手函数，负责创建一个`DataLoader`实例来解析来自`flower_photos`目录的文件

我们有四个 Python 驱动脚本:

1.  `build_dataset.py`:取`flower_photos`目录，建立`dataset`目录。我们将创建特殊的子目录来存储我们的训练和验证分割，允许 PyTorch 的`ImageFolder`脚本解析目录并训练我们的模型。
2.  `train_feature_extraction.py`:通过特征提取进行迁移学习，将输出模型序列化到磁盘。
3.  `fine_tune.py`:通过微调进行迁移学习，并将模型保存到磁盘。
4.  `inference.py`:接受一个经过训练的 PyTorch 模型，并使用它对输入的花朵图像进行预测。

项目目录结构中的`.png`文件包含我们输出预测的可视化。

### **创建我们的配置文件**

在实施任何迁移学习脚本之前，我们首先需要创建配置文件。

这个配置文件将存储我们的驱动程序脚本中使用的重要变量和参数。我们不再在每个脚本中*重新定义它们，而是在这里简单地定义它们*一次*(从而使我们的代码更加清晰易读)。*

打开`pyimagesearch`模块中的`config.py`文件，插入以下代码:

```py
# import the necessary packages
import torch
import os

# define path to the original dataset and base path to the dataset
# splits
DATA_PATH = "flower_photos"
BASE_PATH = "dataset"

# define validation split and paths to separate train and validation
# splits
VAL_SPLIT = 0.1
TRAIN = os.path.join(BASE_PATH, "train")
VAL = os.path.join(BASE_PATH, "val")
```

**第 7 行**定义了`DATA_PATH`，我们输入`flower_photos`目录的路径。

然后我们设置`BASE_PATH`变量指向我们的`dataset`目录(**第 8 行**)。这个目录将通过我们的`build_dataset.py`脚本创建和填充。当我们运行迁移学习/推理脚本时，我们将从`BASE_PATH`目录中读取图像。

**第 12 行**将我们的验证分割设置为 10%，这意味着我们将 90%的数据用于训练，10%用于验证。

我们还在第 13 行**和第 14 行**定义了`TRAIN`和`VAL`子目录。一旦我们运行`build_dataset.py`，我们将在`dataset`中有两个子目录:

1.  `dataset/train`
2.  `dataset/val`

每个子目录将为五个花卉类别中的每一个类别存储其各自的图像。

我们将微调 ResNet 架构，在 ImageNet 数据集上进行预训练。这意味着我们必须为图像像素缩放设置一些重要的参数:

```py
# specify ImageNet mean and standard deviation and image size
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

**第 17 行和第 18 行**定义了 RGB 颜色空间中像素强度的平均值和标准偏差。

这些值是由研究人员在 ImageNet 数据集上训练他们的模型获得的。他们遍历 ImageNet 数据集中的所有图像，从磁盘加载它们，并计算 RGB 像素强度的平均值和标准偏差。

然后，在训练之前，将平均值和标准偏差值用于图像像素归一化。

即使我们没有使用 ImageNet 数据集进行迁移学习，我们*仍然*需要执行与 ResNet 接受培训时相同的预处理步骤；否则，模型将不能正确理解输入图像。

**第 19 行**将我们的输入`IMAGE_SIZE`设置为`224 × 224`像素。

`DEVICE`变量控制我们是使用 CPU 还是 GPU 进行训练。

接下来，我们有一些变量将用于特征提取和微调:

```py
# specify training hyperparameters
FEATURE_EXTRACTION_BATCH_SIZE = 256
FINETUNE_BATCH_SIZE = 64
PRED_BATCH_SIZE = 4
EPOCHS = 20
LR = 0.001
LR_FINETUNE = 0.0005
```

当执行特征提取时，我们将通过我们的网络分批传递图像`256` ( **线 25** )。

我们将使用`64` ( **第 26 行**)的图像批次，而不是通过微调来执行迁移学习。

当执行推理时(即通过`inference.py`脚本进行预测)，我们将使用`4`的批量大小。

最后，我们设置我们将训练模型的`EPOCHS`的数量、特征提取的学习速率和微调的学习速率。这些值是通过[运行简单的超参数调整实验](https://pyimagesearch.com/2021/05/17/introduction-to-hyperparameter-tuning-with-scikit-learn-and-python/)确定的。

我们将通过设置输出文件路径来结束我们的配置脚本:

```py
# define paths to store training plots and trained model
WARMUP_PLOT = os.path.join("output", "warmup.png")
FINETUNE_PLOT = os.path.join("output", "finetune.png")
WARMUP_MODEL = os.path.join("output", "warmup_model.pth")
FINETUNE_MODEL = os.path.join("output", "finetune_model.pth")
```

**第 33 行和第 34 行**为我们的输出训练历史和序列化模型设置文件路径，用于特征提取。

**第 35 行和第 36 行**也是如此，只是为了微调。

### **实现我们的数据加载器助手**

PyTorch 允许我们轻松地从存储在磁盘目录中的图像构建`DataLoader`对象。

***注:*** *如果你以前从未使用过 PyTorch 的`DataLoader`对象，我建议你阅读我们的*[*PyTorch 教程简介*](https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/) *，以及我们关于 py torch 图像数据加载器的指南。*

打开`pyimagesearch`模块内的`create_dataloaders.py`文件，我们开始吧:

```py
# import the necessary packages
from . import config
from torch.utils.data import DataLoader
from torchvision import datasets
import os
```

**第 2-5 行**导入我们需要的 Python 包，包括:

*   `config`:我们在上一节中创建的配置文件
*   `DataLoader` : PyTorch 的数据加载类，用于高效处理数据批处理
*   【PyTorch 的一个子模块，提供对`ImageFolder`类的访问，用于从磁盘上的输入目录中读取图像
*   `os`:用于确定 CPU 上核心/工作线程的数量，从而加快数据加载速度

在那里，我们定义了`get_dataloader`函数:

```py
def get_dataloader(rootDir, transforms, batchSize, shuffle=True):
	# create a dataset and use it to create a data loader
	ds = datasets.ImageFolder(root=rootDir,
		transform=transforms)
	loader = DataLoader(ds, batch_size=batchSize,
		shuffle=shuffle,
		num_workers=os.cpu_count(),
		pin_memory=True if config.DEVICE == "cuda" else False)

	# return a tuple of  the dataset and the data loader
	return (ds, loader)
```

该函数接受四个参数:

1.  `rootDir`:磁盘上包含我们数据集的输入目录的路径(即`dataset`目录)
2.  `transforms`:要执行的数据转换列表，包括预处理步骤和数据扩充
3.  `batchSize`:从`DataLoader`中产出的批次大小
4.  `shuffle`:是否打乱数据——我们将打乱数据进行训练，但*不会打乱*进行验证

**第 9 行和第 10 行**创建了我们的`ImageFolder`类，用于从`rootDir`中读取图像。这也是我们应用`transforms`的地方。

然后在第 11-14 行的**上创建`DataLoader`。在这里我们:**

*   传入我们的`ImageFolder`对象
*   设置批量大小
*   指示是否将执行随机播放
*   设置`num_workers`，这是我们机器上 CPUs 内核的数量
*   设置我们是否使用 GPU 内存

产生的`ImageFolder`和`DataLoader`实例被返回给**行 17** 上的调用函数。

### **创建我们的数据集组织脚本**

现在我们已经创建了配置文件并实现了`DataLoader`助手函数，让我们创建用于构建`dataset`目录的`build_dataset.py`脚本，以及`train`和`val`子目录。

打开项目目录结构中的`build_dataset.py`文件，插入以下代码:

```py
# USAGE
# python build_dataset.py

# import necessary packages
from pyimagesearch import config
from imutils import paths
import numpy as np
import shutil
import os
```

**第 5-9 行**导入我们需要的 Python 包。我们的进口产品包括:

*   `config`:我们的 Python 配置文件
*   `paths`:`imutils`的子模块，用于收集给定目录中图像的路径
*   `numpy`:数值数组处理
*   `shutil`:用于将文件从一个位置复制到另一个位置
*   `os`:用于在磁盘上创建目录的操作系统模块

接下来，我们有我们的`copy_images`函数:

```py
def copy_images(imagePaths, folder):
	# check if the destination folder exists and if not create it
	if not os.path.exists(folder):
		os.makedirs(folder)

	# loop over the image paths
	for path in imagePaths:
		# grab image name and its label from the path and create
		# a placeholder corresponding to the separate label folder
		imageName = path.split(os.path.sep)[-1]
		label = path.split(os.path.sep)[1]
		labelFolder = os.path.join(folder, label)

		# check to see if the label folder exists and if not create it
		if not os.path.exists(labelFolder):
			os.makedirs(labelFolder)

		# construct the destination image path and copy the current
		# image to it
		destination = os.path.join(labelFolder, imageName)
		shutil.copy(path, destination)
```

`copy_images`函数需要两个参数:

1.  `imagePaths`:给定输入目录下所有图像的路径
2.  `folder`:存储复制图像的输出基础目录(即`dataset`目录)

**第 13 行和第 14 行**快速检查一下`folder`目录是否存在。如果目录*不存在*，我们创建它。

从那里，我们循环所有的`imagePaths` ( **第 17 行**)。对于每个`path`，我们:

1.  抓取文件名(**第 20 行**)
2.  从图像路径中提取类别标签(**第 21 行**)
3.  构建基本输出目录(**第 22 行**)

如果`labelFolder`子目录尚不存在，我们在**的第 25 行和第 26 行**创建它。

从那里，我们构建到`destination`文件的路径(**第 30 行**)并复制它(**第 31 行**)。

现在让我们使用这个`copy_images`函数:

```py
# load all the image paths and randomly shuffle them
print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(config.DATA_PATH))
np.random.shuffle(imagePaths)

# generate training and validation paths
valPathsLen = int(len(imagePaths) * config.VAL_SPLIT)
trainPathsLen = len(imagePaths) - valPathsLen
trainPaths = imagePaths[:trainPathsLen]
valPaths = imagePaths[trainPathsLen:]

# copy the training and validation images to their respective
# directories
print("[INFO] copying training and validation images...")
copy_images(trainPaths, config.TRAIN)
copy_images(valPaths, config.VAL)
```

**第 35 行和第 36 行**从我们的输入`DATA_PATH`(即`flower_photos`目录)中读取所有的`imagePaths`，然后随机洗牌。

**第 39-42 行**根据我们的`VAL_SPLIT`百分比创建我们的培训和验证分割。

最后，我们使用`copy_images`函数将`trainPaths`和`valPaths`复制到它们各自的输出目录中(**第 47 行和第 48 行**)。

下一节将使这个过程更加清晰，包括为什么我们要费这么大劲以这种特定的方式组织数据集目录结构。

### **在磁盘上构建数据集**

我们现在准备构建数据集目录。请务必使用本教程的 ***“下载”*** 部分来访问源代码和示例图像。

从那里，打开一个 shell 并执行以下命令:

```py
$ python build_dataset.py
[INFO] loading image paths...
[INFO] copying training and validation images...
```

脚本执行后，您将看到一个新的`dataset`目录已经创建:

```py
$ tree dataset --dirsfirst --filelimit 10
dataset
├── train
│   ├── daisy [585 entries exceeds filelimit, not opening dir]
│   ├── dandelion [817 entries exceeds filelimit, not opening dir]
│   ├── roses [568 entries exceeds filelimit, not opening dir]
│   ├── sunflowers [624 entries exceeds filelimit, not opening dir]
│   └── tulips [709 entries exceeds filelimit, not opening dir]
└── val
    ├── daisy [48 entries exceeds filelimit, not opening dir]
    ├── dandelion [81 entries exceeds filelimit, not opening dir]
    ├── roses [73 entries exceeds filelimit, not opening dir]
    ├── sunflowers [75 entries exceeds filelimit, not opening dir]
    └── tulips [90 entries exceeds filelimit, not opening dir]
```

注意，`dataset`目录有两个子目录:

1.  `train`:包含五类中每一类的训练图像。
2.  `val`:存储五类中每一类的验证图像。

通过创建一个`train`和`val`目录，我们现在可以很容易地利用 PyTorch 的`ImageFolder`类来构建一个`DataLoader`，这样我们就可以微调我们的模型。

### **实现特征提取和迁移学习 PyTorch**

我们要实现的迁移学习的第一个方法是**特征提取。**

通过特征提取进行迁移学习的工作原理是:

1.  采用预先训练的 CNN(通常在 ImageNet 数据集上)
2.  从 CNN 移除 FC 层头
3.  将网络主体的输出视为具有空间维度`M × N × C`的任意特征提取器

从那以后，我们有两个选择:

1.  采用一个标准的逻辑回归分类器(如 scikit-learn 库中的分类器),并根据从每幅图像中提取的特征对其进行训练
2.  或者，更简单地说，在网络主体的顶部放置一个 softmax 分类器

这两种选择都是可行的，而且或多或少与另一种“相同”。

当提取的要素数据集适合计算机的 RAM 时，第一个选项非常有用。这样，您可以加载整个数据集，实例化您最喜欢的逻辑回归分类器模型的实例，然后训练它。

当你的数据集*太大*而不适合你的机器内存时，就会出现*问题*。当这种情况发生时，你可以使用类似于[在线学习的东西来训练你的逻辑回归分类器](https://pyimagesearch.com/2019/06/17/online-incremental-learning-with-keras-and-creme/)，但是这只是引入了另一组库和依赖。

相反，更简单的方法是利用 PyTorch 的能力，在提取的特征基础上创建一个类似逻辑回归的分类器，然后使用 PyTorch 函数训练它。这是我们今天要实施的方法。

打开项目目录结构中的`train_feature_extraction.py`文件，让我们开始吧:

```py
# USAGE
# python train_feature_extraction.py

# import the necessary packages
from pyimagesearch import config
from pyimagesearch import create_dataloaders
from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
```

**第 5-15 行**导入我们需要的 Python 包。值得注意的进口包括:

*   `config`:我们的 Python 配置文件
*   `create_dataloaders`:从我们的输入`dataset`目录中创建一个 PyTorch `DataLoader`的实例
*   `resnet50`:我们将使用的 ResNet 模型(在 ImageNet 数据集上进行了预训练)
*   `transforms`:允许我们定义一组预处理和/或数据扩充程序，这些程序将依次应用于输入图像
*   用于创建格式良好的进度条的 Python 库
*   `torch`和`nn`:包含 PyTorch 的神经网络类和函数

处理好我们的导入后，让我们继续定义我们的数据预处理和增强管道:

```py
# define augmentation pipelines
trainTansform = transforms.Compose([
	transforms.RandomResizedCrop(config.IMAGE_SIZE),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(90),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])
valTransform = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])
```

我们使用 PyTorch 的`transforms`子模块中的`Compose`函数构建数据处理/扩充步骤。

首先，我们创建一个`trainTransform`，给定一个输入图像，它将:

1.  随机调整图像大小并将其裁剪至`IMAGE_SIZE`尺寸
2.  随机执行水平翻转
3.  在`[-90, 90]`范围内随机旋转
4.  将生成的图像转换为 PyTorch 张量
5.  执行均值减法和缩放

然后我们有了我们的`valTransform`，它:

1.  将输入图像调整到`IMAGE_SIZE`尺寸
2.  将图像转换为 PyTorch 张量
3.  执行均值减法和缩放

注意，我们*不*在验证转换器内部执行数据扩充——没有必要为我们的验证数据执行数据扩充。

随着我们的训练和验证`Compose`对象的创建，让我们应用我们的`get_dataloader`函数:

```py
# create data loaders
(trainDS, trainLoader) = create_dataloaders.get_dataloader(config.TRAIN,
	transforms=trainTansform,
	batchSize=config.FEATURE_EXTRACTION_BATCH_SIZE)
(valDS, valLoader) = create_dataloaders.get_dataloader(config.VAL,
	transforms=valTransform,
	batchSize=config.FEATURE_EXTRACTION_BATCH_SIZE, shuffle=False)
```

**第 32-34 行**创建我们的训练数据加载器，而**第 35-37 行**创建我们的验证数据加载器。

这些加载器中的每一个都将分别从`dataset/train`和`dataset/val`目录中产生图像。

另外，注意我们*没有*对我们的验证数据进行混排(就像我们没有对验证数据进行数据扩充一样)。

现在，让我们通过特征提取为迁移学习准备 ResNet50 模型:

```py
# load up the ResNet50 model
model = resnet50(pretrained=True)

# since we are using the ResNet50 model as a feature extractor we set
# its parameters to non-trainable (by default they are trainable)
for param in model.parameters():
	param.requires_grad = False

# append a new classification top to our feature extractor and pop it
# on to the current device
modelOutputFeats = model.fc.in_features
model.fc = nn.Linear(modelOutputFeats, len(trainDS.classes))
model = model.to(config.DEVICE)
```

**第 40 行**从磁盘加载在 ImageNet 上预先训练的 ResNet。

由于我们将使用 ResNet 进行特征提取，因此在网络主体中不需要进行实际的“学习”，我们*冻结*网络主体中的所有层(**第 44 行和第 45 行**)。

在此基础上，我们创建了一个由单个 FC 层组成的新 FC 层头。实际上，当使用分类交叉熵损失进行训练时，这一层将充当我们的代理 softmax 分类器。

然后，这个新层被添加到网络主体中，而`model`本身被移动到我们的`DEVICE`(我们的 CPU 或 GPU)。

接下来，我们初始化损失函数和优化方法:

```py
# initialize loss function and optimizer (notice that we are only
# providing the parameters of the classification top to our optimizer)
lossFunc = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.fc.parameters(), lr=config.LR)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDS) // config.FEATURE_EXTRACTION_BATCH_SIZE
valSteps = len(valDS) // config.FEATURE_EXTRACTION_BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "train_acc": [], "val_loss": [],
	"val_acc": []}
```

我们将使用 Adam 优化器和分类交叉熵损失来训练我们的模型(**第 55 和 56 行**)。

我们还计算我们的模型将采取的步骤数，作为批量大小的函数，分别用于我们的训练集和测试集(**行 59 和 60** )。

现在，该训练模型了:

```py
# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.EPOCHS)):
	# set the model in training mode
	model.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0

	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0

	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = lossFunc(pred, y)

		# calculate the gradients
		loss.backward()

		# check if we are updating the model parameters and if so
		# update them, and zero out the previously accumulated gradients
		if (i + 2) % 2 == 0:
			opt.step()
			opt.zero_grad()

		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()
```

在第 69 行的**上，**我们循环我们想要的历元数。

对于`trainLoader`中的每一批数据，我们:

1.  将图像和类标签移动到我们的 CPU/GPU ( **Line 85** )。
2.  对数据进行预测(**行 88** )
3.  计算损失，计算梯度，更新模型权重，并将梯度归零(**第 89-98 行**)
4.  累计我们在该时期的总训练损失(**行 102** )
5.  计算正确预测的总数(**行 103 和 104** )

现在纪元已经完成，我们可以根据验证数据评估模型:

```py
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()

		# loop over the validation set
		for (x, y) in valLoader:
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

			# make the predictions and calculate the validation loss
			pred = model(x)
			totalValLoss += lossFunc(pred, y)

			# calculate the number of correct predictions
			valCorrect += (pred.argmax(1) == y).type(
				torch.float).sum().item()
```

请注意，我们关闭了亲笔签名，并将模型置于评估模式——这是使用 PyTorch 评估时的一个*要求*,所以不要忘记这样做！

从那里，我们循环遍历`valLoader`中的所有数据点，对它们进行预测，并计算我们的总损失和正确验证预测的数量。

以下代码块汇总了我们的训练/验证损失和准确性，更新了我们的训练历史，然后将损失/准确性信息打印到我们的终端:

```py
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps

	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDS)
	valCorrect = valCorrect / len(valDS)

	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_acc"].append(valCorrect)

	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
		avgValLoss, valCorrect))
```

我们的最终代码块绘制了我们的训练历史并将我们的模型序列化到磁盘:

```py
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.WARMUP_PLOT)

# serialize the model to disk
torch.save(model, config.WARMUP_MODEL)
```

在这个脚本执行之后，您将在您的`output`目录中找到一个名为`warmup_model.pth`的文件——这个文件是您的序列化 PyTorch 模型，它可以用于在`inference.py`脚本中进行预测。

### **具有特征提取的 PyTorch 迁移学习**

我们现在准备通过 PyTorch 的特征提取来执行迁移学习。

确保你有:

1.  使用本教程的 ***“下载”*** 部分访问源代码、示例图像等。
2.  执行了`build_dataset.py`脚本来创建我们的数据集目录结构

假设您已经完成了这两个步骤，您可以继续运行`train_feature_extraction.py`脚本:

```py
$ python train_feature_extraction.py
[INFO] training the network...
  0% 0/20 [00:00<?, ?it/s][INFO] EPOCH: 1/20
Train loss: 1.610827, Train accuracy: 0.4063
Val loss: 2.295713, Val accuracy: 0.6512
  5% 1/20 [00:17<05:24, 17.08s/it][INFO] EPOCH: 2/20
Train loss: 1.190757, Train accuracy: 0.6703
Val loss: 1.720566, Val accuracy: 0.7193
 10% 2/20 [00:33<05:05, 16.96s/it][INFO] EPOCH: 3/20
Train loss: 0.958189, Train accuracy: 0.7163
Val loss: 1.423687, Val accuracy: 0.8120
 15% 3/20 [00:50<04:47, 16.90s/it][INFO] EPOCH: 4/20
Train loss: 0.805547, Train accuracy: 0.7811
Val loss: 1.200151, Val accuracy: 0.7793
 20% 4/20 [01:07<04:31, 16.94s/it][INFO] EPOCH: 5/20
Train loss: 0.731831, Train accuracy: 0.7856
Val loss: 1.066768, Val accuracy: 0.8283
 25% 5/20 [01:24<04:14, 16.95s/it][INFO] EPOCH: 6/20
Train loss: 0.664001, Train accuracy: 0.8044
Val loss: 0.996960, Val accuracy: 0.8311
...
 75% 15/20 [04:13<01:24, 16.83s/it][INFO] EPOCH: 16/20
Train loss: 0.495064, Train accuracy: 0.8480
Val loss: 0.736332, Val accuracy: 0.8665
 80% 16/20 [04:30<01:07, 16.86s/it][INFO] EPOCH: 17/20
Train loss: 0.502294, Train accuracy: 0.8435
Val loss: 0.732066, Val accuracy: 0.8501
 85% 17/20 [04:46<00:50, 16.85s/it][INFO] EPOCH: 18/20
Train loss: 0.486568, Train accuracy: 0.8471
Val loss: 0.703661, Val accuracy: 0.8801
 90% 18/20 [05:03<00:33, 16.82s/it][INFO] EPOCH: 19/20
Train loss: 0.470880, Train accuracy: 0.8480
Val loss: 0.715560, Val accuracy: 0.8474
 95% 19/20 [05:20<00:16, 16.85s/it][INFO] EPOCH: 20/20
Train loss: 0.489092, Train accuracy: 0.8426
Val loss: 0.684679, Val accuracy: 0.8774
100% 20/20 [05:37<00:00, 16.86s/it]
[INFO] total time taken to train the model: 337.24s
```

总训练时间刚刚超过 5 分钟。我们得到了 **84.26%** 的训练准确率和 **87.74%** 的验证准确率。

**图 3** 显示了我们的训练历史。

对于我们在培训过程中投入的这么少的时间来说，还不算太坏！

### **用 PyTorch 微调 CNN**

到目前为止，在本教程中，您已经学习了如何通过特征提取来执行迁移学习。

这种方法在某些情况下工作得很好，但它的简单性有其缺点，即模型的准确性和概括能力都会受到影响。

**迁移学习的大部分形式适用** ***微调*** **，这就是本节的题目。**

与特征提取类似，我们首先从网络中移除 FC 层头，但这次我们创建了一个全新的层头，包含一组线性、ReLU 和 dropout 层，类似于您在现代最先进的 CNN 上看到的内容。

然后，我们执行以下操作的某种组合:

1.  冻结网络体中的所有层并训练层头
2.  冷冻所有层，训练层头，然后*解冻*身体并训练
3.  简单地让所有层解冻并一起训练它们

确切地说，你使用哪种方法是你自己进行的实验——一定要测量哪种方法给你带来的损失最小，准确度最高！

让我们学习如何通过 PyTorch 的迁移学习来应用微调。打开项目目录结构中的`fine_tune.py`文件，让我们开始吧:

```py
# USAGE
# python fine_tune.py

# import the necessary packages
from pyimagesearch import config
from pyimagesearch import create_dataloaders
from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
import time
import os
```

我们从第 5-17 行开始，导入我们需要的 Python 包。请注意，这些导入与我们之前的脚本基本上是*相同的*。

然后，我们定义我们的训练和验证转换，就像我们对特征提取所做的那样:

```py
# define augmentation pipelines
trainTansform = transforms.Compose([
	transforms.RandomResizedCrop(config.IMAGE_SIZE),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(90),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])
valTransform = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])
```

对于我们的数据加载器来说也是如此——它们以与特征提取完全相同的方式进行实例化:

```py
# create data loaders
(trainDS, trainLoader) = create_dataloaders.get_dataloader(config.TRAIN,
	transforms=trainTansform, batchSize=config.FINETUNE_BATCH_SIZE)
(valDS, valLoader) = create_dataloaders.get_dataloader(config.VAL,
	transforms=valTransform, batchSize=config.FINETUNE_BATCH_SIZE,
	shuffle=False)
```

***真正的变化*发生在我们从磁盘加载 ResNet 并修改架构本身的时候，所以让我们仔细检查这一部分:**

```py
# load up the ResNet50 model
model = resnet50(pretrained=True)
numFeatures = model.fc.in_features

# loop over the modules of the model and set the parameters of
# batch normalization modules as not trainable
for module, param in zip(model.modules(), model.parameters()):
	if isinstance(module, nn.BatchNorm2d):
		param.requires_grad = False

# define the network head and attach it to the model
headModel = nn.Sequential(
	nn.Linear(numFeatures, 512),
	nn.ReLU(),
	nn.Dropout(0.25),
	nn.Linear(512, 256),
	nn.ReLU(),
	nn.Dropout(0.5),
	nn.Linear(256, len(trainDS.classes))
)
model.fc = headModel

# append a new classification top to our feature extractor and pop it
# on to the current device
model = model.to(config.DEVICE)
```

**第 41 行**从磁盘加载我们的 ResNet 模型，并在 ImageNet 数据集上预先训练权重。

在这个特定的微调示例中，我们将构建一个新的 FC 层头，然后同时训练*FC 层头*和*网络主体。*

 *然而，我们首先需要密切关注网络架构中的批处理规范化层。这些图层具有特定的平均值和标准偏差值，这些值是最初在 ImageNet 数据集上训练网络时获得的。

我们*不想*在训练期间更新这些统计数据，所以我们在**的第 46-48 行冻结了`BatchNorm2d`的任何实例。**

**如果您在使用批量标准化的网络中执行微调，请确保在开始训练** **之前冻结这些层** ***！***

从那里，我们构造了新的`headModel`，它由一系列 FC = > RELU = >退出层(**第 51-59 行**)组成。

最终`Linear`层的输出是数据集中类的数量(**第 58 行**)。

最后，我们将新的`headModel`添加到网络中，从而替换旧的 FC 层头。

***注:*** *如果你想要更多关于迁移学习、特征提取、微调的细节，建议你阅读以下教程——*[*用 Keras 迁移学习和深度学习*](https://pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/)*；* [*微调与 Keras 和深度学习*](https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/)*；以及* [*Keras:深度学习*](https://pyimagesearch.com/2019/05/27/keras-feature-extraction-on-large-datasets-with-deep-learning/) *大数据集上的特征提取。*

完成“网络手术”后，我们可以继续实例化我们的损失函数和优化器:

```py
# initialize loss function and optimizer (notice that we are only
# providing the parameters of the classification top to our optimizer)
lossFunc = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=config.LR)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDS) // config.FINETUNE_BATCH_SIZE
valSteps = len(valDS) // config.FINETUNE_BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "train_acc": [], "val_loss": [],
	"val_acc": []}
```

从那里，我们开始我们的培训渠道:

```py
# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.EPOCHS)):
	# set the model in training mode
	model.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0

	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0

	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = lossFunc(pred, y)

		# calculate the gradients
		loss.backward()

		# check if we are updating the model parameters and if so
		# update them, and zero out the previously accumulated gradients
		if (i + 2) % 2 == 0:
			opt.step()
			opt.zero_grad()

		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()
```

在这一点上，微调我们模型的代码与特征提取方法的代码*相同*，所以您可以推迟到上一节来详细回顾代码。

培训完成后，我们可以进入新时代的验证阶段:

```py
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()

		# loop over the validation set
		for (x, y) in valLoader:
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

			# make the predictions and calculate the validation loss
			pred = model(x)
			totalValLoss += lossFunc(pred, y)

			# calculate the number of correct predictions
			valCorrect += (pred.argmax(1) == y).type(
				torch.float).sum().item()

	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps

	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDS)
	valCorrect = valCorrect / len(valDS)

	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_acc"].append(valCorrect)

	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
		avgValLoss, valCorrect))
```

验证完成后，我们绘制培训历史并将模型序列化到磁盘:

```py
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.FINETUNE_PLOT)

# serialize the model to disk
torch.save(model, config.FINETUNE_MODEL)
```

执行完`train_feature_extraction.py`脚本后，您将在您的`output`目录中找到一个名为`finetune_model.pth`的训练模型。

您可以使用此模型和`inference.py`对新图像进行预测。

### **PyTorch 微调结果**

现在让我们使用 PyTorch 进行微调。

同样，确保你有:

1.  使用本教程的 ***【下载】*** 部分下载源代码、数据集等。
2.  执行了`build_dataset.py`脚本来创建我们的`dataset`目录

从那里，您可以执行以下命令:

```py
$ python fine_tune.py
[INFO] training the network...
  0% 0/20 [00:00<?, ?it/s][INFO] EPOCH: 1/20
Train loss: 0.857740, Train accuracy: 0.6809
Val loss: 2.498850, Val accuracy: 0.6512
  5% 1/20 [00:18<05:55, 18.74s/it][INFO] EPOCH: 2/20
Train loss: 0.581107, Train accuracy: 0.7972
Val loss: 0.432770, Val accuracy: 0.8665
 10% 2/20 [00:38<05:40, 18.91s/it][INFO] EPOCH: 3/20
Train loss: 0.506620, Train accuracy: 0.8289
Val loss: 0.721634, Val accuracy: 0.8011
 15% 3/20 [00:57<05:26, 19.18s/it][INFO] EPOCH: 4/20
Train loss: 0.477470, Train accuracy: 0.8341
Val loss: 0.431005, Val accuracy: 0.8692
 20% 4/20 [01:17<05:10, 19.38s/it][INFO] EPOCH: 5/20
Train loss: 0.467796, Train accuracy: 0.8368
Val loss: 0.746030, Val accuracy: 0.8120
 25% 5/20 [01:37<04:53, 19.57s/it][INFO] EPOCH: 6/20
Train loss: 0.429070, Train accuracy: 0.8523
Val loss: 0.607376, Val accuracy: 0.8311
...
 75% 15/20 [04:51<01:36, 19.33s/it][INFO] EPOCH: 16/20
Train loss: 0.317167, Train accuracy: 0.8880
Val loss: 0.344129, Val accuracy: 0.9183
 80% 16/20 [05:11<01:17, 19.32s/it][INFO] EPOCH: 17/20
Train loss: 0.295942, Train accuracy: 0.9013
Val loss: 0.375650, Val accuracy: 0.8992
 85% 17/20 [05:30<00:58, 19.38s/it][INFO] EPOCH: 18/20
Train loss: 0.282065, Train accuracy: 0.9046
Val loss: 0.374338, Val accuracy: 0.8992
 90% 18/20 [05:49<00:38, 19.30s/it][INFO] EPOCH: 19/20
Train loss: 0.254787, Train accuracy: 0.9116
Val loss: 0.302762, Val accuracy: 0.9264
 95% 19/20 [06:08<00:19, 19.25s/it][INFO] EPOCH: 20/20
Train loss: 0.270875, Train accuracy: 0.9083
Val loss: 0.385452, Val accuracy: 0.9019
100% 20/20 [06:28<00:00, 19.41s/it]
[INFO] total time taken to train the model: 388.23s
```

由于我们的模型更加复杂(由于向网络主体添加了新的 FC 层头),培训现在需要`~6.5`分钟。

****然而，在图 4 中，我们获得了比我们的简单特征提取方法**** 更高的准确度(分别为 90.83%/90.19%对 84.26%/87.74%):

虽然执行微调确实需要更多的工作，但您通常会发现精确度更高，并且您的模型会更好地泛化。

### **实施我们的 PyTorch 预测脚本**

到目前为止，您已经学习了使用 PyTorch 应用迁移学习的两种方法:

1.  特征抽出
2.  微调

这两种方法都使模型获得了 80-90%的准确性…

但是我们如何使用这些模型进行预测呢？

答案是使用我们的`inference.py`脚本:

```py
# USAGE
# python inference.py --model output/warmup_model.pth
# python inference.py --model output/finetune_model.pth

# import the necessary packages
from pyimagesearch import config
from pyimagesearch import create_dataloaders
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import argparse
import torch
```

我们从一些导入开始我们的`inference.py`脚本，包括:

*   `config`:我们的配置文件
*   `create_dataloaders`:我们的助手实用程序从图像的输入目录(在本例中是我们的`dataset/val`目录)创建一个`DataLoader`对象
*   `transforms`:按顺序应用数据预处理
*   在屏幕上显示我们的输出图像和预测
*   `torch`和`nn`:我们的 PyTorch 绑定
*   `argparse`:解析任何命令行参数

说到命令行参数，现在让我们来解析它们:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
args = vars(ap.parse_args())
```

这里我们只需要一个参数，`--model`，它是驻留在磁盘上的经过训练的 PyTorch 模型的路径。

现在让我们为输入图像创建一个变换对象:

```py
# build our data pre-processing pipeline
testTransform = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])

# calculate the inverse mean and standard deviation
invMean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
invStd = [1/s for s in config.STD]

# define our de-normalization transform
deNormalize = transforms.Normalize(mean=invMean, std=invStd)
```

就像上一节中的验证转换器一样，我们在这里要做的就是:

1.  将我们的输入图像调整到`IMAGE_SIZE`尺寸
2.  将图像转换为 PyTorch 张量
3.  对输入图像应用平均缩放

然而，为了在屏幕上显示输出图像，我们实际上需要“反规格化”它们。**第 28 和 29 行**计算反均值和标准差，而**第 32 行**创建一个`deNormalize`变换。

使用`deNormalize`转换，我们将能够“撤销”`testTransform`，然后从我们的屏幕显示输出图像。

现在让我们为我们的`config.VAL`目录构建一个`DataLoader`:

```py
# initialize our test dataset and data loader
print("[INFO] loading the dataset...")
(testDS, testLoader) = create_dataloaders.get_dataloader(config.VAL,
	transforms=testTransform, batchSize=config.PRED_BATCH_SIZE,
	shuffle=True)
```

从那里，我们可以设置我们的目标计算设备并加载我们训练过的 PyTorch 模型:

```py
# check if we have a GPU available, if so, define the map location
# accordingly
if torch.cuda.is_available():
	map_location = lambda storage, loc: storage.cuda()

# otherwise, we will be using CPU to run our model
else:
	map_location = "cpu"

# load the model
print("[INFO] loading the model...")
model = torch.load(args["model"], map_location=map_location)

# move the model to the device and set it in evaluation mode
model.to(config.DEVICE)
model.eval()
```

**第 40-47 行**检查我们是否在使用我们的 CPU 或 GPU。

**第 51-55 行**继续:

1.  从磁盘加载我们训练过的 PyTorch 模式
2.  把它移到我们的目标`DEVICE`
3.  将模型置于评估模式

现在让我们从`testLoader`中随机抽取一组测试数据:

```py
# grab a batch of test data
batch = next(iter(testLoader))
(images, labels) = (batch[0], batch[1])

# initialize a figure
fig = plt.figure("Results", figsize=(10, 10))
```

最后，我们可以根据测试数据做出预测:

```py
# switch off autograd
with torch.no_grad():
	# send the images to the device
	images = images.to(config.DEVICE)

	# make the predictions
	print("[INFO] performing inference...")
	preds = model(images)

	# loop over all the batch
	for i in range(0, config.PRED_BATCH_SIZE):
		# initalize a subplot
		ax = plt.subplot(config.PRED_BATCH_SIZE, 1, i + 1)

		# grab the image, de-normalize it, scale the raw pixel
		# intensities to the range [0, 255], and change the channel
		# ordering from channels first tp channels last
		image = images[i]
		image = deNormalize(image).cpu().numpy()
		image = (image * 255).astype("uint8")
		image = image.transpose((1, 2, 0))

		# grab the ground truth label
		idx = labels[i].cpu().numpy()
		gtLabel = testDS.classes[idx]

		# grab the predicted label
		pred = preds[i].argmax().cpu().numpy()
		predLabel = testDS.classes[pred]

		# add the results and image to the plot
		info = "Ground Truth: {}, Predicted: {}".format(gtLabel,
			predLabel)
		plt.imshow(image)
		plt.title(info)
		plt.axis("off")

	# show the plot
	plt.tight_layout()
	plt.show()
```

**线 65** 关闭自动签名计算(将 PyTorch 模型置于评估模式时的要求)，而**线 67** 将`images`发送到适当的`DEVICE`。

**线 71** 使用我们训练过的`model`对`images`进行预测。

为了可视化这些预测，我们首先需要在第 74 行的**处对它们进行循环。在循环中，我们继续:**

1.  初始化一个子图来显示图像和预测(**第 76 行**
2.  通过“撤销”平均缩放和交换颜色通道顺序来反规格化图像(**第 81-84 行**)
3.  抓住*地面实况*标签(**第 87 行和第 88 行**)
4.  抓取*预测*标签(**行 91 和 92**
5.  将图像、地面实况和预测标签添加到绘图中(**第 95-99 行**)

然后输出的可视化显示在我们的屏幕上。

### **用我们训练过的 PyTorch 模型进行预测**

现在让我们使用我们的`inference.py`脚本和我们训练过的 PyTorch 模型进行预测。

前往本教程的 ***“下载”*** 部分，访问源代码、数据集等。，从那里，您可以执行以下命令:

```py
$ python inference.py --model output/finetune_model.pth
[INFO] loading the dataset...
[INFO] loading the model...
[INFO] performing inference...
```

你可以在**图 5** 中看到结果。

在这里，你可以看到我们已经正确地对我们的花图像进行了分类——最棒的是，由于迁移学习，我们能够以*很小的努力*获得如此高的准确度。

## **总结**

在本教程中，您学习了如何使用 PyTorch 执行迁移学习。

具体来说，我们讨论了两种类型的迁移学习:

1.  通过特征提取进行迁移学习
2.  通过微调转移学习

第一种方法通常更容易实现，需要的精力也更少。然而，它往往不如第二种方法准确。

我通常建议使用特征提取方法来获得基线精度。如果精度足以满足您的应用，那就太棒了！您已经完成了，您可以继续构建项目的其余部分。

但是，如果精确度*不够*，那么您应该进行微调，看看是否可以提高精确度。

无论是哪种情况，迁移学习，无论是通过特征提取还是微调，都会为你节省大量的时间和精力，而不是从头开始训练你的模型。

### **引用信息**

**Rosebrock，a .**“py torch:迁移学习和图像分类”， *PyImageSearch* ，2021，[https://PyImageSearch . com/2021/10/11/py torch-Transfer-Learning-and-Image-class ification/](https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/)

```py
@article{Rosebrock_2021_Transfer,
   author = {Adrian Rosebrock},
   title = {{PyTorch}: Transfer Learning and Image Classification},
   journal = {PyImageSearch},
   year = {2021},
   note = {https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/}, }
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*******