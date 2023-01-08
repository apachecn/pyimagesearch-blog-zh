# 在 PyTorch 中从头开始训练对象检测器

> 原文：<https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/>

在本教程中，您将学习如何使用 PyTorch 从头开始训练自定义对象检测器。

本课是高级 PyTorch 技术 3 部分系列的第 2 部分:

1.  [*在 PyTorch*](https://pyimagesearch.com/2021/10/25/training-a-dcgan-in-pytorch/) 训练一个 DCGAN(上周教程)
2.  *在 PyTorch 中从头开始训练一个物体探测器*(今天的教程)
3.  *U-Net:在 PyTorch 中训练图像分割模型*(下周博文)

从童年开始，人工智能(AI)的想法就吸引了我(就像其他孩子一样)。但是，当然，我对人工智能的概念与它的实际情况有很大的不同，毫无疑问是由于流行文化。直到我少年时代的末期，我坚信 AI 不受抑制的成长会导致 T-800(终结者中的终结者)之类的东西。幸运的是，使用**图 1** 可以更好地解释实际场景:

不过，不要误会我的意思。机器学习可能是一堆矩阵和微积分结合在一起，但我们可以用一个词来最好地描述这些东西的数量:*无限*。

我一直感兴趣的一个这样的应用是对象检测。注入图像数据来获取标签是一回事，但是让我们的模型知道标签在哪里呢？那是完全不同的一场球赛，就像一些间谍电影一样。这正是我们今天要经历的！

在今天的教程中，我们将学习如何在 PyTorch 中从头开始训练我们自己的物体检测器。这个博客将帮助你:

*   理解物体检测背后的直觉
*   了解构建您自己的对象检测器的逐步方法
*   了解如何微调参数以获得理想的结果

**要学习如何在 Pytorch 中从头开始训练物体检测器，*继续阅读。***

### **在 PyTorch 中从头开始训练物体检测器**

在今天强大的深度学习算法存在之前，物体检测是一个历史上广泛研究的领域。从 20 世纪 90 年代末到 21 世纪 20 年代初，提出了许多新的想法，这些想法至今仍被用作深度学习算法的基准。不幸的是，在那个时候，研究人员没有太多的计算能力可供他们支配，所以大多数这些技术都依赖于大量的额外数学来减少计算时间。谢天谢地，我们不会面临这个问题。

### **我们的目标检测方法**

我们先来了解一下物体检测背后的直觉。我们将要采用的方法非常类似于训练一个简单的分类器。分类器的权重不断变化，直到它为给定数据集输出正确的标签并减少损失。对于今天的任务，我们将做与*完全相同的*事情，除了我们的模型将输出 **5 个值**， **4 个是围绕我们对象的边界框**的 **坐标。**第 5 个值是被检测对象的标签**。注意图 2** 中**的架构。**

主模型将分为两个子集:回归器和分类器。前者将输出边界框的起始和结束坐标，而后者将输出对象标签。由这些**值**产生的组合损耗将用于我们的反向传播。很简单的开始方式，不是吗？

当然，多年来，一些强大的算法接管了物体检测领域，如 [R-CNN](https://arxiv.org/abs/1311.2524) 和 [YOLO](https://arxiv.org/abs/1506.02640) 。但是我们的方法将作为一个合理的起点，让你的头脑围绕物体检测背后的基本思想！

### 配置您的开发环境

要遵循这个指南，首先需要在系统中安装 PyTorch。要访问 PyTorch 自己的视觉计算模型，您还需要在您的系统中安装 Torchvision。对于一些数组和存储操作，我们使用了`numpy`。我们也使用`imutils`包进行数据处理。对于我们的情节，我们将使用`matplotlib`。为了更好地跟踪我们的模型训练，我们将使用`tqdm`，最后，我们的系统中需要 OpenCV！

幸运的是，上面提到的所有包都是 pip-installable！

```py
$ pip install opencv-contrib-python
$ pip install torch
$ pip install torchvision
$ pip install imutils
$ pip install matplotlib
$ pip install numpy
$ pip install tqdm
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

### 项目结构

我们首先需要回顾我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

从这里，看一下目录结构:

```py
!tree .
.
├── dataset.zip
├── output
│   ├── detector.pth
│   ├── le.pickle
│   ├── plots
│   │   └── training.png
│   └── test_paths.txt
├── predict.py
├── pyimagesearch
│   ├── bbox_regressor.py
│   ├── config.py
│   ├── custom_tensor_dataset.py
│   └── __init__.py
└── train.py
```

目录中的第一项是`dataset.zip`。这个 zip 文件包含完整的数据集(图像、标签和边界框)。在后面的章节中会有更多的介绍。

接下来，我们有了`output`目录。这个目录是我们所有保存的模型、结果和其他重要需求被转储的地方。

父目录中有两个脚本:

*   用于训练我们的目标探测器
*   `predict.py`:用于从我们的模型中进行推断，并查看运行中的对象检测器

最后，我们有最重要的目录，即`pyimagesearch`目录。它包含了 3 个非常重要的脚本。

*   `bbox_regressor.py`:容纳完整的物体检测器架构
*   `config.py`:包含端到端训练和推理管道的配置
*   `custom_tensor_dataset.py`:包含数据准备的自定义类

我们的项目目录审查到此结束。

### **配置物体检测的先决条件**

我们的第一个任务是配置我们将在整个项目中使用的几个超参数。为此，让我们跳到`pyimagesearch`文件夹并打开`config.py`脚本。

```py
# import the necessary packages
import torch
import os

# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = "dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output model, label encoder, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
```

我们首先定义几个路径，我们稍后会用到它们。然后在**第 7-12 行**，我们为数据集(图像和注释)和输出定义路径。接下来，我们为我们的检测器和标签编码器创建单独的路径，然后是我们的绘图和测试图像的路径(**第 16-19 行**)。

```py
# determine the current device and based on that set the pin memory
# flag
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32

# specify the loss weights
LABELS = 1.0
BBOX = 1.0
```

由于我们正在训练一个对象检测器，建议在 GPU 上训练，而不是在 CPU 上，因为计算更复杂。因此，如果我们的系统中有兼容 CUDA 的 GPU，我们就将 PyTorch 设备设置为 CUDA(**第 23 行和第 24 行**)。

当然，我们将在数据集准备期间使用 PyTorch 变换。因此，我们指定平均值和标准偏差值(**第 27 行和第 28 行**)。这三个值分别代表通道方向、宽度方向和高度方向的平均值和标准偏差。最后，我们为我们的模型初始化超参数，如学习速率、时期、批量大小和损失权重(**第 32-38 行**)。

### **创建自定义对象检测数据处理器**

让我们看看我们的数据目录。

```py
!tree . 
.
├── dataset
│   ├── annotations
│   └── images
│       ├── airplane
│       ├── face
│       └── motorcycle
```

数据集细分为两个文件夹:annotations(包含边界框起点和终点的 CSV 文件)和 images(进一步分为三个文件夹，每个文件夹代表我们今天将使用的类)。

因为我们将使用 PyTorch 自己的数据加载器，所以以数据加载器能够接受的方式预处理数据[是很重要的。`custom_tensor_dataset.py`脚本将完全做到这一点。](https://pytorch.org/docs/stable/data.html)

```py
# import the necessary packages
from torch.utils.data import Dataset

class CustomTensorDataset(Dataset):
	# initialize the constructor
	def __init__(self, tensors, transforms=None):
		self.tensors = tensors
		self.transforms = transforms
```

我们创建了一个定制类`CustomTensorDataset`，它继承自`torch.utils.data.Dataset`类(**第 4 行**)。这样，我们可以根据需要配置内部函数，同时保留`torch.utils.data.Dataset`类的核心属性。

在**的第 6-8 行**，构造函数`__init__`被创建。构造函数接受两个参数:

*   `tensors`:三个张量的元组，即图像、标签和边界框坐标。
*   `transforms`:将用于处理图像的`torchvision.transforms`实例。

```py
	def __getitem__(self, index):
		# grab the image, label, and its bounding box coordinates
		image = self.tensors[0][index]
		label = self.tensors[1][index]
		bbox = self.tensors[2][index]

		# transpose the image such that its channel dimension becomes
		# the leading one
		image = image.permute(2, 0, 1)

		# check to see if we have any image transformations to apply
		# and if so, apply them
		if self.transforms:
			image = self.transforms(image)

		# return a tuple of the images, labels, and bounding
		# box coordinates
		return (image, label, bbox)
```

因为我们使用的是自定义类，所以我们将覆盖父类(`Dataset`)的方法。因此，根据我们的需要改变了`__getitem__`方法。但是，首先，`tensor`元组被解包到它的组成元素中(**第 12-14 行**)。

图像张量最初的形式是`Height` × `Width` × `Channels`。然而，所有 PyTorch 模型都需要他们的输入成为**“通道优先”**相应地，`image.permute`方法重新排列图像张量(**行 18** )。

我们在第 22 行和第 23 行上为`torchvision.transforms`实例添加了一个检查。如果检查结果为`true`，图像将通过`transform`实例传递。此后，`__getitem__`方法返回图像、标签和边界框。

```py
	def __len__(self):
		# return the size of the dataset
		return self.tensors[0].size(0)
```

我们要覆盖的第二个方法是`__len__`方法。它返回图像数据集张量的大小(**第 29-31 行**)。这就结束了`custom_tensor_dataset.py`脚本。

### **构建异议检测架构**

谈到这个项目需要的模型，我们需要记住两件事。首先，为了避免额外的麻烦和有效的特征提取，我们将使用一个预先训练的模型作为基础模型。其次，基础模型将被分成两部分；盒式回归器和标签分类器。这两者都是独立的模型实体。

要记住的第二件事是，只有盒子回归器和标签分类器具有可训练的权重。预训练模型的重量将保持不变，如图**图 4** 所示。

考虑到这一点，让我们进入`bbox_regressor.py`！

```py
# import the necessary packages
from torch.nn import Dropout
from torch.nn import Identity
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential
from torch.nn import Sigmoid

class ObjectDetector(Module):
	def __init__(self, baseModel, numClasses):
		super(ObjectDetector, self).__init__()

		# initialize the base model and the number of classes
		self.baseModel = baseModel
		self.numClasses = numClasses
```

对于定制模型`ObjectDetector`，我们将使用`torch.nn.Module`作为父类(**第 10 行**)。对于构造函数`__init__`，有两个外部参数；基本型号和标签数量(**第 11-16 行**)。

```py
		# build the regressor head for outputting the bounding box
		# coordinates
		self.regressor = Sequential(
			Linear(baseModel.fc.in_features, 128),
			ReLU(),
			Linear(128, 64),
			ReLU(),
			Linear(64, 32),
			ReLU(),
			Linear(32, 4),
			Sigmoid()
		)
```

继续讨论回归变量，记住我们的最终目标是产生 4 个单独的值:起始 *x* 轴值，起始 *y* 轴值，结束 *x* 轴值，以及结束 *y* 轴值。第一个`Linear`层输入基本模型的全连接层，输出尺寸设置为 **128** ( **行 21** )。

接下来是几个`Linear`和`ReLU`层(**行 22-27** )，最后以输出 **4 值**的`Linear`层结束，接下来是`Sigmoid`层(**行 28** )。

```py
		# build the classifier head to predict the class labels
		self.classifier = Sequential(
			Linear(baseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, self.numClasses)
		)

		# set the classifier of our base model to produce outputs
		# from the last convolution block
		self.baseModel.fc = Identity()
```

下一步是对象标签的分类器。在回归器中，我们获取基本模型完全连接层的特征尺寸，并将其插入第一个`Linear`层(**第 33 行**)。接着重复`ReLU`、`Dropout`和`Linear`层(**行 34-40** )。`Dropout`层通常用于帮助扩展概化和防止过度拟合。

初始化的最后一步是将基本模型的全连接层变成一个`Identity`层，这意味着它将镜像其之前的卷积块产生的输出(**第 44 行**)。

```py
 	def forward(self, x):
		# pass the inputs through the base model and then obtain
		# predictions from two different branches of the network
		features = self.baseModel(x)
		bboxes = self.regressor(features)
		classLogits = self.classifier(features)

		# return the outputs as a tuple
		return (bboxes, classLogits)
```

接下来是`forward`步骤(**第 46 行**)。我们简单地将基本模型的输出通过回归器和分类器(**第 49-51 行**)。

这样，我们就完成了目标检测器的架构设计。

### **训练目标检测模型**

在我们看到物体探测器工作之前，只剩下最后一步了。因此，让我们跳到`train.py`脚本并训练模型！

```py
# USAGE
# python train.py

# import the necessary packages
from pyimagesearch.bbox_regressor import ObjectDetector
from pyimagesearch.custom_tensor_dataset import CustomTensorDataset
from pyimagesearch import config
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import time
import cv2
import os

# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []
```

导入必要的包后，我们为数据、标签、边界框和图像路径创建空列表(**第 29-32 行**)。

现在是进行一些数据预处理的时候了。

```py
# loop over all CSV files in the annotations directory
for csvPath in paths.list_files(config.ANNOTS_PATH, validExts=(".csv")):
	# load the contents of the current CSV annotations file
	rows = open(csvPath).read().strip().split("\n")

	# loop over the rows
	for row in rows:
		# break the row into the filename, bounding box coordinates,
		# and class label
		row = row.split(",")
		(filename, startX, startY, endX, endY, label) = row

		# derive the path to the input image, load the image (in
		# OpenCV format), and grab its dimensions
		imagePath = os.path.sep.join([config.IMAGES_PATH, label,
			filename])
		image = cv2.imread(imagePath)
		(h, w) = image.shape[:2]

		# scale the bounding box coordinates relative to the spatial
		# dimensions of the input image
		startX = float(startX) / w
		startY = float(startY) / h
		endX = float(endX) / w
		endY = float(endY) / h

		# load the image and preprocess it
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (224, 224))

		# update our list of data, class labels, bounding boxes, and
		# image paths
		data.append(image)
		labels.append(label)
		bboxes.append((startX, startY, endX, endY))
		imagePaths.append(imagePath)
```

在第 35 行的**处，我们开始遍历目录中所有可用的 CSV。打开 CSV，然后我们开始遍历这些行来分割数据(**第 37-44 行**)。**

在将行值分割成一组单独的值之后，我们首先挑选出图像路径(**行 48** )。然后，我们使用 OpenCV 读取图像并获得其高度和宽度(**第 50 行和第 51 行**)。

然后使用高度和宽度值将边界框坐标缩放到`0`和`1`的范围内(**第 55-58 行**)。

接下来，我们加载图像并做一些轻微的预处理(**第 61-63 行**)。

然后用解包后的值更新空列表，并且随着每次迭代的进行重复该过程(**行 67-70** )。

```py
# convert the data, class labels, bounding boxes, and image paths to
# NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)

# perform label encoding on the labels
le = LabelEncoder()
labels = le.fit_transform(labels)
```

为了更快地处理数据，列表被转换成`numpy`数组(**第 74-77 行**)。由于标签是字符串格式，我们使用 [scikit-learn 的`LabelEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) 将它们转换成各自的索引(**第 80 行和第 81 行**)。

```py
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, labels, bboxes, imagePaths,
	test_size=0.20, random_state=42)

# unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]
```

使用另一个方便的 scikit-learn 工具`train_test_split`，我们将数据分成训练集和测试集，保持一个`80-20`比率(**第 85 行和第 86 行**)。由于拆分将应用于传递给`train_test_split`函数的所有数组，我们可以使用简单的行切片将它们解包为元组(**第 89-92 行**)。

```py
# convert NumPy arrays to PyTorch tensors
(trainImages, testImages) = torch.tensor(trainImages),\
	torch.tensor(testImages)
(trainLabels, testLabels) = torch.tensor(trainLabels),\
	torch.tensor(testLabels)
(trainBBoxes, testBBoxes) = torch.tensor(trainBBoxes),\
	torch.tensor(testBBoxes)

# define normalization transforms
transforms = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])
```

解包的训练和测试数据、标签和边界框然后从 numpy 格式转换成 PyTorch 张量(**第 95-100 行**)。接下来，我们继续创建一个`torchvision.transforms`实例来轻松处理数据集(**第 103-107 行**)。这样，数据集也将使用`config.py`中定义的平均值和标准偏差值进行标准化。

```py
# convert NumPy arrays to PyTorch datasets
trainDS = CustomTensorDataset((trainImages, trainLabels, trainBBoxes),
	transforms=transforms)
testDS = CustomTensorDataset((testImages, testLabels, testBBoxes),
	transforms=transforms)
print("[INFO] total training samples: {}...".format(len(trainDS)))
print("[INFO] total test samples: {}...".format(len(testDS)))

# calculate steps per epoch for training and validation set
trainSteps = len(trainDS) // config.BATCH_SIZE
valSteps = len(testDS) // config.BATCH_SIZE

# create data loaders
trainLoader = DataLoader(trainDS, batch_size=config.BATCH_SIZE,
	shuffle=True, num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)
testLoader = DataLoader(testDS, batch_size=config.BATCH_SIZE,
	num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)
```

记住，在`custom_tensor_dataset.py`脚本中，我们创建了一个定制的`Dataset`类来满足我们的确切需求。截至目前，我们所需的实体只是张量。因此，为了将它们转换成 **PyTorch DataLoader** 接受的格式，我们创建了`CustomTensorDataset`类的训练和测试实例，将图像、标签和边界框作为参数传递(**第 110-113 行**)。

在**行 118 和 119** 上，使用数据集的长度和`config.py`中设置的批量值计算每个时期的步骤值。

最后，我们通过`DataLoader`传递`CustomTensorDataset`实例，并创建训练和测试数据加载器(**第 122-125 行**)。

```py
# write the testing image paths to disk so that we can use then
# when evaluating/testing our object detector
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()

# load the ResNet50 network
resnet = resnet50(pretrained=True)

# freeze all ResNet50 layers so they will *not* be updated during the
# training process
for param in resnet.parameters():
	param.requires_grad = False
```

因为我们稍后将使用测试映像路径进行评估，所以它被写入磁盘(**行 129-132** )。

对于我们架构中的基本模型，我们将使用一个预先训练好的 **resnet50** ( **Line 135** )。然而，如前所述，基本模型的重量将保持不变。因此，我们冻结了权重(**第 139 和 140 行**)。

```py
# create our custom object detector model and flash it to the current
# device
objectDetector = ObjectDetector(resnet, len(le.classes_))
objectDetector = objectDetector.to(config.DEVICE)

# define our loss functions
classLossFunc = CrossEntropyLoss()
bboxLossFunc = MSELoss()

# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(objectDetector.parameters(), lr=config.INIT_LR)
print(objectDetector)

# initialize a dictionary to store training history
H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [],
	 "val_class_acc": []}
```

模型先决条件完成后，我们创建我们的定制模型实例，并将其加载到当前设备中(**第 144 行和第 145 行**)。对于分类器损失，使用交叉熵损失，而对于箱式回归器，我们坚持均方误差损失(**第 148 和 149 行**)。在**行** **153** ，`Adam`被设置为目标探测器优化器。为了跟踪训练损失和其他度量，字典`H`在**行 157 和 158** 上被初始化。

```py
# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
	# set the model in training mode
	objectDetector.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0

	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0
```

对于训练速度评估，记录开始时间(**行 162** )。循环多个时期，我们首先将对象检测器设置为训练模式(**第 165 行**，并初始化正确预测的损失和数量(**第 168-174 行**)。

```py
	# loop over the training set
	for (images, labels, bboxes) in trainLoader:
		# send the input to the device
		(images, labels, bboxes) = (images.to(config.DEVICE),
			labels.to(config.DEVICE), bboxes.to(config.DEVICE))

		# perform a forward pass and calculate the training loss
		predictions = objectDetector(images)
		bboxLoss = bboxLossFunc(predictions[0], bboxes)
		classLoss = classLossFunc(predictions[1], labels)
		totalLoss = (config.BBOX * bboxLoss) + (config.LABELS * classLoss)

		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		opt.zero_grad()
		totalLoss.backward()
		opt.step()

		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += totalLoss
		trainCorrect += (predictions[1].argmax(1) == labels).type(
			torch.float).sum().item()
```

在列车数据加载器上循环，我们首先将图像、标签和边界框加载到正在使用的设备中(**行 179 和 180** )。接下来，我们将图像插入我们的对象检测器，并存储预测结果( **Line 183** )。最后，由于模型将给出两个预测(一个用于标签，一个用于边界框)，我们将它们索引出来并分别计算这些损失(**第 183-185 行**)。

这两个损失的组合值将作为架构的总损失。我们将在`config.py`中定义的边界框损失和标签损失的各自损失权重乘以损失，并将它们相加(**第 186 行**)。

在 PyTorch 的自动梯度功能的帮助下，我们简单地重置梯度，计算由于产生的损失而产生的权重，并基于当前步骤的梯度更新参数(**第 190-192 行**)。重置梯度是很重要的，因为`backward`函数一直在累积梯度。因为我们只想要当前步骤的梯度，所以`opt.zero_grad`清除了先前的值。

在第**行第 196-198** 行，我们更新损失值并修正预测。

```py
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		objectDetector.eval()

		# loop over the validation set
		for (images, labels, bboxes) in testLoader:
			# send the input to the device
			(images, labels, bboxes) = (images.to(config.DEVICE),
				labels.to(config.DEVICE), bboxes.to(config.DEVICE))

			# make the predictions and calculate the validation loss
			predictions = objectDetector(images)
			bboxLoss = bboxLossFunc(predictions[0], bboxes)
			classLoss = classLossFunc(predictions[1], labels)
			totalLoss = (config.BBOX * bboxLoss) + \
				(config.LABELS * classLoss)
			totalValLoss += totalLoss

			# calculate the number of correct predictions
			valCorrect += (predictions[1].argmax(1) == labels).type(
				torch.float).sum().item()
```

转到模型评估，我们将首先关闭自动渐变并切换到对象检测器的评估模式(**行 201-203** )。然后，循环测试数据，除了更新权重之外，我们将重复与训练中相同的过程(**第 212-214 行**)。

以与训练步骤相同的方式计算组合损失(**行 215 和 216** )。因此，总损失值和正确预测被更新(**第 217-221 行**)。

```py
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps

	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDS)
	valCorrect = valCorrect / len(testDS)

	# update our training history
	H["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_class_acc"].append(trainCorrect)
	H["total_val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_class_acc"].append(valCorrect)

	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
		avgValLoss, valCorrect))
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))
```

在一个时期之后，在**行 224 和 225** 上计算平均分批训练和测试损失。我们还使用正确预测的数量来计算历元的训练和测试精度(**行 228 和 229** )。

在计算之后，所有的值都记录在模型历史字典`H` ( **第 232-235 行**)中，同时计算结束时间以查看训练花费了多长时间以及退出循环之后(**第 243 行**)。

```py
# serialize the model to disk
print("[INFO] saving object detector model...")
torch.save(objectDetector, config.MODEL_PATH)

# serialize the label encoder to disk
print("[INFO] saving label encoder...")
f = open(config.LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["total_train_loss"], label="total_train_loss")
plt.plot(H["total_val_loss"], label="total_val_loss")
plt.plot(H["train_class_acc"], label="train_class_acc")
plt.plot(H["val_class_acc"], label="val_class_acc")
plt.title("Total Training Loss and Classification Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

# save the training plot
plotPath = os.path.sep.join([config.PLOTS_PATH, "training.png"])
plt.savefig(plotPath)
```

因为我们将使用对象检测器进行推理，所以我们将它保存到磁盘中( **Line 249** )。我们还保存了已创建的标签编码器，因此模式保持不变(**第 253-255 行**

为了评估模型训练，我们绘制了存储在模型历史字典`H` ( **第 258-271 行**)中的所有指标。

模型训练到此结束。接下来，我们来看看物体探测器训练的有多好！

### **评估物体探测训练**

由于模型的大部分重量保持不变，训练不会花很长时间。首先，让我们来看看一些训练时期。

```py
[INFO] training the network...
  0%|          | 0/20 [00:00<?,  
  5%|▌         | 1/20 [00:16<05:08, 16.21s/it][INFO] EPOCH: 1/20
Train loss: 0.874699, Train accuracy: 0.7608
Val loss: 0.360270, Val accuracy: 0.9902
 10%|█         | 2/20 [00:31<04:46, 15.89s/it][INFO] EPOCH: 2/20
Train loss: 0.186642, Train accuracy: 0.9834
Val loss: 0.052412, Val accuracy: 1.0000
 15%|█▌        | 3/20 [00:47<04:28, 15.77s/it][INFO] EPOCH: 3/20
Train loss: 0.066982, Train accuracy: 0.9883
...
 85%|████████▌ | 17/20 [04:27<00:47, 15.73s/it][INFO] EPOCH: 17/20
Train loss: 0.011934, Train accuracy: 0.9975
Val loss: 0.004053, Val accuracy: 1.0000
 90%|█████████ | 18/20 [04:43<00:31, 15.67s/it][INFO] EPOCH: 18/20
Train loss: 0.009135, Train accuracy: 0.9975
Val loss: 0.003720, Val accuracy: 1.0000
 95%|█████████▌| 19/20 [04:58<00:15, 15.66s/it][INFO] EPOCH: 19/20
Train loss: 0.009403, Train accuracy: 0.9982
Val loss: 0.003248, Val accuracy: 1.0000
100%|██████████| 20/20 [05:14<00:00, 15.73s/it][INFO] EPOCH: 20/20
Train loss: 0.006543, Train accuracy: 0.9994
Val loss: 0.003041, Val accuracy: 1.0000
[INFO] total time taken to train the model: 314.68s
```

我们看到，该模型在训练和验证时分别达到了惊人的精度 **0.9994** 和 **1.0000** 。让我们看看训练图上划时代的变化**图 5** ！

该模型在训练值和验证值方面都相当快地达到了饱和水平。现在是时候看看物体探测器的作用了！

### **从物体检测器得出推论**

这个旅程的最后一步是`predict.py`脚本。这里，我们将逐个循环测试图像，并用我们的预测值绘制边界框。

```py
# USAGE
# python predict.py --input datasimg/face/image_0131.jpg

# import the necessary packages
from pyimagesearch import config
from torchvision import transforms
import mimetypes
import argparse
import imutils
import pickle
import torch
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image/text file of image paths")
args = vars(ap.parse_args())
```

`argparse`模块用于编写用户友好的命令行界面命令。在**的第 15-18 行**，我们构建了一个参数解析器来帮助用户选择测试图像。

```py
# determine the input file type, but assume that we're working with
# single input image
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]

# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:
	# load the image paths in our testing file
	imagePaths = open(args["input"]).read().strip().split("\n")
```

我们按照参数解析的步骤来处理用户给出的任何类型的输入。在**行 22 和 23** 上，`imagePaths`变量被设置为处理单个输入图像，而在**行 27-29** 上，处理多个图像的事件。

```py
# load our object detector, set it evaluation mode, and label
# encoder from disk
print("[INFO] loading object detector...")
model = torch.load(config.MODEL_PATH).to(config.DEVICE)
model.eval()
le = pickle.loads(open(config.LE_PATH, "rb").read())

# define normalization transforms
transforms = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])
```

使用`train.py`脚本训练的模型被调用进行评估(**第 34 和 35 行**)。类似地，使用前述脚本存储的标签编码器被加载(**第 36 行**)。因为我们需要再次处理数据，所以创建了另一个`torchvision.transforms`实例，其参数与训练中使用的参数相同。

```py
# loop over the images that we'll be testing using our bounding box
# regression model
for imagePath in imagePaths:
	# load the image, copy it, swap its colors channels, resize it, and
	# bring its channel dimension forward
	image = cv2.imread(imagePath)
	orig = image.copy()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	image = image.transpose((2, 0, 1))

	# convert image to PyTorch tensor, normalize it, flash it to the
	# current device, and add a batch dimension
	image = torch.from_numpy(image)
	image = transforms(image).to(config.DEVICE)
	image = image.unsqueeze(0)
```

循环测试图像，我们读取图像并对其进行一些预处理(**第 50-54 行**)。这样做是因为我们的图像需要再次插入对象检测器。

我们继续将图像转换成张量，对其应用`torchvision.transforms`实例，并为其添加批处理维度(**第 58-60 行**)。我们的测试图像现在可以插入到对象检测器中了。

```py
	# predict the bounding box of the object along with the class
	# label
	(boxPreds, labelPreds) = model(image)
	(startX, startY, endX, endY) = boxPreds[0]

	# determine the class label with the largest predicted
	# probability
	labelPreds = torch.nn.Softmax(dim=-1)(labelPreds)
	i = labelPreds.argmax(dim=-1).cpu()
	label = le.inverse_transform(i)[0]
```

首先，从模型中获得预测(**行 64** )。我们继续从`boxPreds`变量(**第 65 行)**解包边界框值。

标签预测上的简单 softmax 函数将为我们提供对应于类的值的更好的图片。为此，我们在**69 线**上使用 PyTorch 自己的`torch.nn.Softmax`。用`argmax`隔离索引，我们将它插入标签编码器`le`，并使用`inverse_transform`(索引到值)来获得标签的名称(**第 69-71 行**)。

```py
	# resize the original image such that it fits on our screen, and
	# grab its dimensions
	orig = imutils.resize(orig, width=600)
	(h, w) = orig.shape[:2]

	# scale the predicted bounding box coordinates based on the image
	# dimensions
	startX = int(startX * w)
	startY = int(startY * h)
	endX = int(endX * w)
	endY = int(endY * h)

	# draw the predicted bounding box and class label on the image
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (0, 255, 0), 2)
	cv2.rectangle(orig, (startX, startY), (endX, endY),
		(0, 255, 0), 2)

	# show the output image 
	cv2.imshow("Output", orig)
	cv2.waitKey(0)
```

在**第 75 行**，我们已经调整了原始图像的大小以适应我们的屏幕。然后存储尺寸调整后的图像的高度和宽度，以基于图像缩放预测的边界框值(**第 76-83 行**)。之所以这样做，是因为我们在将注释与模型匹配之前，已经将它们缩小到了范围`0`和`1`。因此，出于显示目的，所有输出都必须放大。

显示边界框时，标签名称也将显示在它的顶部。为此，我们为第 86 行的**文本设置了 *y* 轴的值。使用 OpenCV 的`putText`函数，我们设置了显示在图像上的标签(**第 87 行和第 88 行**)。**

最后，我们使用 OpenCV 的`rectangle`方法在图像上创建边界框(**第 89 行和第 90 行**)。因为我们有起始 *x* 轴、起始 *y* 轴、结束 *x* 轴和结束 *y* 轴的值，所以很容易从它们创建一个矩形。这个矩形将包围我们的对象。

我们的推理脚本到此结束。让我们看看结果吧！

### **动作中的物体检测**

让我们看看我们的对象检测器的表现如何，使用来自每个类的一个图像。我们首先使用一个飞机的图像(**图 6** )，然后是一个人脸下的图像(**图 7** )，以及一个属于摩托车类的图像(**图 8** )。

事实证明，我们的模型的精度值没有说谎。我们的模型不仅正确地猜出了标签，而且生成的包围盒也几乎是完美的！

有了如此精确的检测和结果，我们都可以认为我们的小项目是成功的，不是吗？

### 摘要

在写这个物体检测教程的时候，回想起来我意识到了几件事。

老实说，我从来不喜欢在我的项目中使用预先训练好的模型。它会觉得我的工作不再是我的工作了。显然，这被证明是一个愚蠢的想法，事实上我的第一个一次性人脸分类器说我和我最好的朋友是同一个人(相信我，我们看起来一点也不相似)。

我认为本教程是一个很好的例子，说明了当你有一个训练有素的特征提取器时会发生什么。我们不仅节省了时间，而且最终的结果也是辉煌的。以**图 6 和图 8** 为例。预测的边界框具有最小的误差。

当然，这并不意味着没有改进的空间。在**图 7** 中，图像有许多元素，但物体探测器已经设法捕捉到物体的大致区域。然而，它可以更紧凑。我们强烈建议您修改参数，看看您的结果是否更好！

也就是说，物体检测在当今世界中扮演着重要的角色。自动交通、人脸检测、无人驾驶汽车只是物体检测蓬勃发展的现实世界应用中的一部分。每年，算法都被设计得更快更紧凑。我们已经达到了一个阶段，算法可以同时检测视频场景中的所有对象！我希望这篇教程激起了你对揭示这个领域复杂性的好奇心。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****