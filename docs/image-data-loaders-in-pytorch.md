# PyTorch 中的图像数据加载器

> 原文：<https://pyimagesearch.com/2021/10/04/image-data-loaders-in-pytorch/>

任何基于深度学习的系统的一个重要部分是构建数据加载管道，以便它可以与您的深度学习模型无缝集成。在本教程中，我们将了解 PyTorch 提供的数据加载功能的工作原理，并学习在我们自己的深度学习项目中有效地使用它们。

**要学习如何使用 PyTorch 数据集和数据加载器，** ***只需继续阅读。***

## **py torch 中的图像数据加载器**

我们将详细讨论以下内容:

*   如何将数据集重新构造为定型集和验证集
*   如何在 PyTorch 中加载数据集并利用内置的 PyTorch 数据扩充
*   如何设置 PyTorch 数据加载器以有效访问数据样本

### **我们的示例花卉数据集**

我们的目标是在 PyTorch `Dataset`和`DataLoader`类的帮助下创建一个基本的数据加载管道，使我们能够轻松高效地访问我们的数据样本，并将它们传递给我们的深度学习模型。

在本教程中，我们将使用由 5 种花卉组成的花卉数据集(见**图 1** ):

1.  郁金香
2.  雏菊
3.  蒲公英
4.  玫瑰
5.  向日葵

### **配置您的开发环境**

为了遵循这个指南，您需要在您的系统上安装 PyTorch 深度学习库、matplotlib、OpenCV 和 imutils 包。

幸运的是，使用 pip 安装这些包非常容易:

```py
$ pip install torch torchvision
$ pip install matplotlib
$ pip install opencv-contrib-python
$ pip install imutils
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

在我们继续之前，我们需要首先下载我们将在本教程中使用的 flowers 数据集。

从本教程的 ***“下载”*** 部分开始，访问源代码和示例 flowers 数据集。

从那里，解压缩档案文件，您应该会找到下面的项目目录:

```py
├── build_dataset.py
├── builtin_dataset.py
├── flower_photos
│   ├── daisy
│   ├── dandelion
│   ├── roses
│   ├── sunflowers
│   └── tulips
├── load_and_visualize.py
├── pyimagesearch
│   ├── config.py
│   ├── __init__.py
```

`build_dataset.py`脚本负责将数据集划分和组织成一个训练和验证集。

此外，`builtin_dataset.py`脚本显示了如何使用 PyTorch API 直接下载和加载一些常用的计算机视觉数据集，如 MNIST。

`flower_photos`文件夹包含我们将使用的数据集。它由 5 个子目录(即雏菊、蒲公英、玫瑰、向日葵、郁金香)组成，每个子目录包含相应花卉类别的图像。

另一方面，`load_and_visualize.py`脚本负责在 PyTorch `Dataset`和`DataLoader`类的帮助下加载和访问数据样本。

`pyimagesearch`文件夹中的`config.py` 文件存储我们代码的参数、初始设置、配置等信息。

请注意，在下载数据集之后，每个图像都将具有以下格式的路径，`folder_name/class_name/image_id.jpg`。例如，下面显示的是花卉数据集中一些图像的路径。

```py
flower_photos/dandelion/8981828144_4b66b4edb6_n.jpg
flower_photos/sunflowers/14244410747_22691ece4a_n.jpg
flower_photos/roses/1666341535_99c6f7509f_n.jpg
flower_photos/sunflowers/19519101829_46af0b4547_m.jpg
flower_photos/dandelion/2479491210_98e41c4e7d_m.jpg
flower_photos/sunflowers/3950020811_dab89bebc0_n.jpg
```

### **创建我们的配置文件**

首先，我们讨论存储教程中使用的配置和参数设置的`config.py`文件。

```py
# specify path to the flowers and mnist dataset
FLOWERS_DATASET_PATH = "flower_photos"
MNIST_DATASET_PATH = "mnist"

# specify the paths to our training and validation set 
TRAIN = "train"
VAL = "val"

# set the input height and width
INPUT_HEIGHT = 128
INPUT_WIDTH = 128

# set the batch size and validation data split
BATCH_SIZE = 8
VAL_SPLIT = 0.1
```

我们在第 2 行的**上定义了 flowers 数据集文件夹的路径，在第 3 行**上定义了 MNIST 数据集的路径。**除此之外，我们在**的第 6 行和第 7 行指定了训练和验证集文件夹的路径名。****

在**第 10 行和第 11 行，**我们定义了输入图像所需的高度和宽度，这将使我们能够在以后将输入调整到我们的模型可以接受的尺寸。

此外，我们在第 14 行和第 15 行定义了批量大小和作为验证集的数据集部分。

### **将数据集分割成训练和验证集**

在这里，我们讨论如何将 flowers 数据集重组为一个训练和验证集。

打开项目目录结构中的`build_dataset.py`文件，让我们开始吧。

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

我们从在第 5-9 行导入所需的包开始。

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
		label = path.split(os.path.sep)[-2]
		labelFolder = os.path.join(folder, label)

		# check to see if the label folder exists and if not create it
		if not os.path.exists(labelFolder):
			os.makedirs(labelFolder)

		# construct the destination image path and copy the current
		# image to it
		destination = os.path.join(labelFolder, imageName)
		shutil.copy(path, destination)
```

从第 11 行的**开始，我们定义了`copy_images`函数。该方法接受一个列表— `imagePaths`(即一组图像的路径)和一个目的地`folder`，并将输入的图像路径复制到目的地。**

当我们想要将一组图像路径复制到 training 或 validation 文件夹中时，这个函数会很方便。接下来，我们详细理解这个函数的每一行。

我们首先检查目标文件夹是否已经存在，如果不存在，我们在第 13 行和第 14 行创建它。

在第 17 行**，**上，我们循环输入`imagePaths`列表中的每条路径。

对于列表中的每个`path` (其形式为`root/class_label/image_id.jpg` ) :

*   我们将`image_id`和`class_label`分别存储在**的 20 行和 21 行**上。
*   在第 22 行的**上，我们在输入目的地定义了一个文件夹`labelFolder`来存储来自特定`class_label`的所有图像。我们在第 25 行和第 26 行**上创建`labelFolder`，如果它还不存在的话。
*   然后，我们为给定的`image_id` ( **行 30** )的图像构建目的地路径(在`labelFolder`内)，并将当前图像复制到其中(**行 31** )。

一旦我们定义了`copy_images`函数，我们就可以理解将数据集分成训练集和验证集所需的主要代码了。

```py
# load all the image paths and randomly shuffle them
print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(config.FLOWERS_DATASET_PATH))
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

**第 35 行**将花卉数据集中所有图像的路径加载到一个名为`imagePaths`的列表中。我们借助**第 36 行**上的`numpy`随机打乱图像路径，以确保训练集和验证集中的图像统一来自所有类别。

现在，我们定义了总图像的一部分，我们希望将其保留下来作为验证数据。

这由`config.VAL_SPLIT`定义。

一个常见的选择是留出 10-20%的数据进行验证。在**第 39 行**，我们将总图像路径的`confg.VAL_SPLIT`部分作为验证集长度(即`valPathsLen`)。

这是四舍五入到最接近的整数，因为我们希望图像的数量是一个整数。剩余的分数作为**线 40** 的车组长度(即`trainPathsLen`)。

在**第 41 行和第 42 行**，我们从`imagePaths`列表中获取训练路径和验证路径。

然后，我们将其传递给`copy_images`函数(如前所述),该函数接收列车路径和验证路径的列表，并将它们复制到由目标文件夹定义的`train`和`val`文件夹，即分别为`config.TRAIN`和`config.VAL`，如第**行第 47 和 48 行所示。**

这构建了我们的文件系统，如下所示。在这里，我们有单独的`train`和`val`文件夹，其中包括来自不同类的训练和验证图像，在它们各自的类文件夹中。

```py
├── train
│   ├── daisy
│   ├── dandelion
│   ├── roses
│   ├── sunflowers
│   └── tulips
└── val
    ├── daisy
    ├── dandelion
    ├── roses
    ├── sunflowers
    └── tulips
```

### **PyTorch 数据集和数据加载器**

既然我们已经将数据集划分为训练集和验证集，我们就可以使用 PyTorch 数据集和数据加载器来设置数据加载管道了。

PyTorch 数据集提供了加载和存储带有相应标签的数据样本的功能。除此之外，PyTorch 还有一个内置的`DataLoader`类，它在数据集周围包装了一个 iterable，使我们能够轻松地访问和迭代数据集中的数据样本。

让我们更深入一点，借助代码来理解数据加载器。基本上，我们的目标是在 PyTorch `Dataset`类的帮助下加载我们的训练集和 val 集，并在`DataLoader`类的帮助下访问样本。

打开项目目录中的`load_and_visualize.py`文件。

我们从导入所需的包开始。

```py
# USAGE
# python load_and_visualize.

# import necessary packages
from pyimagesearch import config
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
```

我们著名的进口产品(**第 6-9 行**)包括:

*   `ImageFolder`类:负责将图像从`train`和`val`文件夹加载到 PyTorch 数据集中
*   类:使我们能够在数据集周围包装一个 iterable，以便我们的深度学习模型可以有效地访问数据样本
*   一个内置的 PyTorch 类，提供了常见的图像转换
*   `matplotlib.pyplot`:用于绘制和可视化图像

现在，我们定义`visualize_batch`函数，稍后它将使我们能够绘制和可视化来自训练和验证批次的样本图像。

```py
def visualize_batch(batch, classes, dataset_type):
	# initialize a figure
	fig = plt.figure("{} batch".format(dataset_type),
		figsize=(config.BATCH_SIZE, config.BATCH_SIZE))

	# loop over the batch size
	for i in range(0, config.BATCH_SIZE):
		# create a subplot
		ax = plt.subplot(2, 4, i + 1)

		# grab the image, convert it from channels first ordering to
		# channels last ordering, and scale the raw pixel intensities
		# to the range [0, 255]
		image = batch[0][i].cpu().numpy()
		image = image.transpose((1, 2, 0))
		image = (image * 255.0).astype("uint8")

		# grab the label id and get the label from the classes list
		idx = batch[1][i]
		label = classes[idx]

		# show the image along with the label
		plt.imshow(image)
		plt.title(label)
		plt.axis("off")

	# show the plot
	plt.tight_layout()
	plt.show()
```

从**第 12 行**开始，`visualize_batch`函数将一批数据样本(即`batch`)、类别标签列表(即`classes`)和该批所属的`dataset_type`(即训练或验证)作为输入。

函数循环给定批次中的索引:

*   在第 *i* 个索引(**第 25 行**)处抓取图像
*   将其转换为通道最后格式，并将其缩放至常规图像像素范围`[0-255]` ( **第 26 行和第 27 行**)
*   获取第 *i* 个样本的整数标签，并借助列表`classes` ( **第 30 行和第 31 行**)将其映射到 flower 数据集的原始类标签
*   显示带有标签的图像(**行 34 和 35** )

在训练深度模型时，我们通常希望在训练集的图像上使用数据增强技术来提高模型的泛化能力。PyTorch 提供了常见的图像转换，可以在 transform 类的帮助下开箱即用。

我们现在将看看这是如何工作的，以及如何整合到我们的数据加载管道中。

```py
# initialize our data augmentation functions
resize = transforms.Resize(size=(config.INPUT_HEIGHT,
        config.INPUT_WIDTH))
hFlip = transforms.RandomHorizontalFlip(p=0.25)
vFlip = transforms.RandomVerticalFlip(p=0.25)
rotate = transforms.RandomRotation(degrees=15)
```

在第**行第 43-47** 行，我们定义了四种我们想要应用到图像的图像变换:

*   `resize`:这种变换使我们能够将图像调整到我们的深度模型可以接受的特定输入维度(即`config.INPUT_HEIGHT`、`config.INPUT_WIDTH`)
*   `hFlip`、`vFlip`:允许我们水平/垂直翻转图像。注意，它使用了一个参数`p`,该参数定义了将该变换应用于输入图像的概率。
*   `rotate` **:** 使我们能够将图像旋转给定的角度

请注意，PyTorch 除了上面提到的以外，还提供了许多其他的图像转换。

```py
# initialize our training and validation set data augmentation
# pipeline
trainTransforms = transforms.Compose([resize, hFlip, vFlip, rotate,
        transforms.ToTensor()])
valTransforms = transforms.Compose([resize, transforms.ToTensor()])
```

我们在`Compose`方法的帮助下合并变换，这样所有的变换都可以一个接一个地应用于我们的输入图像。注意，这里我们有另一个`To.Tensor()`转换，它简单地将所有输入图像转换成 PyTorch 张量。此外，该变换还将原来在`[0, 255]`范围内的输入 PIL 图像或`numpy.ndarray`转换为`[0, 1]`。

这里，我们为我们的训练和验证集定义了单独的转换，如**第 51-53 行**所示。这是因为我们通常不在验证或测试集上使用数据扩充，除了像`resize`和`ToTensor()`这样的转换，它们是将输入数据转换成我们的深度模型可以接受的格式所必需的。

既然我们已经设置了要应用的转换，我们就可以将图像加载到数据集中了。

PyTorch 提供了一个内置的`ImageFolder`功能，它接受一个根文件夹，并自动从给定的根目录获取数据样本来创建数据集。注意`ImageFolder`期望图像以如下格式排列:

```py
root/class_name_1/img_id.png
root/class_name_2/img_id.png
root/class_name_3/img_id.png
root/class_name_4/img_id.png
```

这允许它识别所有唯一的类名，并将它们映射到整数类标签。此外，`ImageFolder`还接受我们在加载图像时想要应用到输入图像的变换(如前所述)。

```py
# initialize the training and validation dataset
print("[INFO] loading the training and validation dataset...")
trainDataset = ImageFolder(root=config.TRAIN,
        transform=trainTransforms)
valDataset = ImageFolder(root=config.VAL, 
        transform=valTransforms)
print("[INFO] training dataset contains {} samples...".format(
        len(trainDataset)))
print("[INFO] validation dataset contains {} samples...".format(
        len(valDataset)))
```

在**的第 57-60 行**，我们使用`ImageFolder`分别为训练集和验证集创建 PyTorch 数据集。注意，每个 PyTorch 数据集都有一个`__len__`方法，使我们能够获得数据集中的样本数，如第 61-64 行的**所示。**

此外，每个数据集都有一个`__getitem__`方法，使我们能够直接索引样本并获取特定的数据点。

假设我们想检查数据集中第 *i* 个数据样本的类型。我们可以简单地将数据集索引为`trainDataset[i]`并访问数据点，它是一个元组。这是因为我们数据集中的每个数据样本都是一个格式为(`image, label`)的元组。

现在，我们准备为数据集创建一个数据加载器。

```py
# create training and validation set dataloaders
print("[INFO] creating training and validation set dataloaders...")
trainDataLoader = DataLoader(trainDataset, 
        batch_size=config.BATCH_SIZE, shuffle=True)
valDataLoader = DataLoader(valDataset, batch_size=config.BATCH_SIZE)
```

DataLoader 接受 PyTorch 数据集并输出 iterable，这样就可以轻松访问数据集中的数据样本。

在**第 68-70 行**，我们将训练和验证数据集传递给`DataLoader`类。

PyTorch 数据加载器接受一个`batch_size`,这样它就可以将数据集分成样本块。

然后，我们的深度模型可以并行处理每个块或批次中的样本。此外，我们还可以决定是否要在将样本传递到深度模型之前对其进行洗牌，这通常是基于批量梯度的优化方法的最佳学习和收敛所需要的。

**第 68-70 行**返回两个 iterables(即`trainDataLoader`和`valDataLoader`)。

```py
# grab a batch from both training and validation dataloader
trainBatch = next(iter(trainDataLoader))
valBatch = next(iter(valDataLoader))

# visualize the training and validation set batches
print("[INFO] visualizing training and validation batch...")
visualize_batch(trainBatch, trainDataset.classes, "train")
visualize_batch(valBatch, valDataset.classes, "val")
```

我们使用第 73 和 74 行的**所示的`iter()`方法将`trainDataLoader`和`valDataLoader` iterable 转换为 python 迭代器。这允许我们在`next()`方法的帮助下简单地迭代通过一批批的训练或验证。**

最后，我们在`visualize_batch`函数的帮助下可视化训练和验证批次。

**第 78 行和第 79 行**借助`visualize_batch`方法可视化`trainBatch`和`valBatch`，给出以下输出。**图 3 和图 4** 分别显示了来自训练和验证批次的样本图像。

### **内置数据集**

在 PyTorch 数据加载器教程的前几节中，我们学习了如何下载自定义数据集、构建数据集、将其作为 PyTorch 数据集加载，以及在数据加载器的帮助下访问其示例。

除此之外，PyTorch 还提供了一个简单的 API，可以用来直接从计算机视觉中一些常用的数据集下载和加载图像。这些数据集包括 MNIST、CIFAR-10、CIFAR-100、CelebA 等。

现在，我们将了解如何轻松访问这些数据集，并将其用于我们自己的项目。出于本教程的目的，我们使用 MNIST 数据集。

让我们首先打开项目目录中的`builtin_dataset.py`文件。

```py
# USAGE
# python builtin_dataset.py

# import necessary packages
from pyimagesearch import config
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
```

我们从在第 5-9 行**导入必要的包开始。**`torchvision.datasets`模块为我们提供了直接加载常用的、广泛使用的数据集的功能。

在第 6 行的**上，我们从这个模块导入了`MNIST`数据集。**

我们其他值得注意的导入包括 PyTorch `DataLoader`类(**第 7 行**)、torchvision 的 transforms 模块(**第 8 行**)和用于可视化的 matplotlib 库(**第 9 行**)。

```py
def visualize_batch(batch, classes, dataset_type):
	# initialize a figure
	fig = plt.figure("{} batch".format(dataset_type),
	figsize=(config.BATCH_SIZE, config.BATCH_SIZE))

	# loop over the batch size
	for i in range(0, config.BATCH_SIZE):
		# create a subplot
		ax = plt.subplot(2, 4, i + 1)

		# grab the image, convert it from channels first ordering to
		# channels last ordering, and scale the raw pixel intensities
		# to the range [0, 255]
		image = batch[0][i].cpu().numpy()
		image = image.transpose((1, 2, 0))
		image = (image * 255.0).astype("uint8")

		# grab the label id and get the label from the classes list
		idx = batch[1][i]
		label = classes[idx]

		# show the image along with the label
		plt.imshow(image[..., 0], cmap="gray")
		plt.title(label)
		plt.axis("off")

	# show the plot
	plt.tight_layout()
	plt.show()
```

接下来，我们定义了`visualize_batch`函数，帮助我们可视化一批样品。

这个函数类似于我们之前在`load_and_visualize.py`文件中定义的`visualize_batch`函数。这里唯一的不同是在**第 33 行**，这里我们以`cmap="gray"`模式绘制图像，因为 MNIST 由单通道灰度图像组成，这与花卉数据集中的 3 通道 RGB 图像形成对比。

```py
# define the transform
transform = transforms.Compose([transforms.ToTensor()])

# initialzie the training and validation dataset
print("[INFO] loading the training and validation dataset...")
trainDataset = MNIST(root=config.MNIST_DATASET_PATH, train=True,
	download=True, transform=transform)
valDataset = MNIST(root=config.MNIST_DATASET_PATH, train=False,
	download=True, transform=transform)
```

我们在第 42 行定义我们的变换。在**第 46-49 行**上，我们使用`torchvision.datasets`模块直接下载 MNIST 训练和验证集，并将其加载为 PyTorch 数据集`trainDataset`和`valDataset`。这里，我们需要提供以下论据:

*   `root`:我们要保存数据集的根目录
*   `train`:表示是要加载训练集(如果`train=True`)还是测试集(如果`train=False`)
*   `download`:负责自动下载数据集
*   `transforms`:应用于输入图像的图像变换

```py
# create training and validation set dataloaders
print("[INFO] creating training and validation set dataloaders...")
trainDataLoader = DataLoader(trainDataset, 
	batch_size=config.BATCH_SIZE, shuffle=True)
valDataLoader = DataLoader(valDataset, batch_size=config.BATCH_SIZE)

# grab a batch from both training and validation dataloader
trainBatch = next(iter(trainDataLoader))
valBatch = next(iter(valDataLoader))

# visualize the training set batch
print("[INFO] visualizing training batch...")
visualize_batch(trainBatch, trainDataset.classes, "train")

# visualize the validation set batch
print("[INFO] visualizing validation batch...")
visualize_batch(valBatch, valDataset.classes, "val")
```

我们为训练和验证数据集创建数据加载器，即第 53-55 行的`trainDataLoader`和`valDataLoader`。接下来，我们从训练和验证数据加载器中获取批次，并在**第 58-67 行**上可视化样本图像，如前所述。

现在您已经了解了如何将 PyTorch 数据加载器与内置的 PyTorch 数据集一起使用。

## **总结**

在本教程中，我们学习了如何在内置 PyTorch 功能的帮助下构建数据加载管道。具体来说，我们了解了 PyTorch 数据集和数据加载器如何高效地从我们的数据集中加载和访问数据样本。

我们的目标是通过将 flowers 数据集分为训练集和测试集来构建数据集，并在 PyTorch 数据集的帮助下加载数据样本。

我们讨论了在应用各种数据扩充时，如何使用 PyTorch 数据加载器有效地访问数据样本。

最后，我们还了解了一些常见的内置 PyTorch 数据集，这些数据集可以直接加载并用于我们的深度学习项目。

遵循教程后，我们建立了一个数据加载管道，可以无缝集成并用于训练手头的任何深度学习模型。恭喜你！

### **引用信息**

**Chandhok，s .**“py torch 中的图像数据加载器”， *PyImageSearch* ，2021 年，[https://PyImageSearch . com/2021/10/04/Image-Data-Loaders-in-py torch/](https://pyimagesearch.com/2021/10/04/image-data-loaders-in-pytorch/)

`@article{Chandhok_2021, author = {Shivam Chandhok}, title = {Image Data Loaders in {PyTorch}}, journal = {PyImageSearch}, year = {2021}, note = {https://pyimagesearch.com/2021/10/04/image-data-loaders-in-pytorch/} }`

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****