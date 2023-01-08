# PyTorch 分布式培训简介

> 原文：<https://pyimagesearch.com/2021/10/18/introduction-to-distributed-training-in-pytorch/>

在本教程中，您将学习使用 PyTorch 进行分布式培训的基础知识。

这是计算机视觉和深度学习从业者中级 PyTorch 技术 3 部分教程的最后一课:

*   [*py torch*](https://pyimagesearch.com/2021/10/04/image-data-loaders-in-pytorch/)中的图像数据加载器(第一课)
*   [*py torch:Tran*](https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/)*[sfer 学习与图像分类](https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/)* (上周教程)
*   *py torch 分布式培训简介*(今天的课程)

当我第一次了解 PyTorch 时，我对它相当冷漠。作为一个在深度学习期间一直使用 TensorFlow 的人，我还没有准备好离开 TensorFlow 创造的舒适区，尝试一些新的东西。

由于命运的安排，由于一些不可避免的情况，我不得不最终潜入 PyTorch。虽然说实话，我的开始很艰难。已经习惯于躲在 TensorFlow 的抽象后面，PyTorch 的冗长本质提醒了我为什么离开 Java 而选择 Python。

然而，过了一会儿，PyTorch 的美丽开始显露出来。它之所以更冗长，是因为它让你对自己的行为有更多的控制。PyTorch 让您对自己的每一步都有更明确的把握，给您更多的自由。也许 Java 也有同样的意图，但我永远不会知道，因为那艘船已经起航了！

分布式训练为您提供了几种方法来利用您拥有的每一点计算能力，并使您的模型训练更加有效。PyTorch 的一个显著特点是它支持分布式训练。

今天我们就来学习一下**数据并行**包，可以实现单机，多 GPU 并行。完成本教程后，读者将会:

*   清晰理解 PyTorch 的数据并行性
*   一种实现数据并行的设想
*   在遍历 PyTorch 的冗长代码时，对自己的目标有一个清晰的认识

**学习如何在 PyTorch 中使用数据并行训练，** ***继续阅读即可。***

## **py torch 分布式培训简介**

### **py torch 的数据并行训练是什么？**

想象一下，有一台配有 4 个 RTX 2060 图形处理器的计算机。你被赋予了一项任务，你必须处理几千兆字节的数据。小菜一碟，对吧？如果你没有办法将所有的计算能力结合在一起会怎么样？这将是非常令人沮丧的，就像我们有 10 亿美元，但每个月只能花 5 美元一样！

如果我们没有办法一起使用我们所有的资源，那将是不理想的。谢天谢地， **PyTorch** 有我们撑腰！**图 1** 展示了 PyTorch 如何以简单而高效的方式在单个系统中利用多个 GPU。

这被称为**数据并行**训练，在这里你使用一个带有多个 GPU 的主机系统来提高效率，同时处理大量的数据。

这个过程非常简单。一旦调用了`nn.DataParallel`,就会在每个 GPU 上创建单独的模型实例。然后，数据被分成相等的部分，每个模型实例一个。最后，每个实例创建自己的渐变，然后在所有可用的实例中进行平均和反向传播。

事不宜迟，让我们直接进入代码，看看分布式培训是如何运行的！

### **配置您的开发环境**

要遵循这个指南，首先需要在系统中安装 PyTorch。要访问 PyTorch 自己的视觉计算模型，您还需要在您的系统中安装 Torchvision。我们也在使用`imutils`包进行数据处理。最后，我们将使用`matplotlib`来绘制我们的结果！

幸运的是，上面提到的所有包都是 pip-installable！

```py
$ pip install torch
$ pip install torchvision
$ pip install imutils
$ pip install matplotlib
```

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

在进入项目之前，让我们回顾一下项目结构。

```py
$ tree -d .
.
├── distributed_inference.py
├── output
│   ├── food_classifier.pth
│   └── model_training.png
├── prepare_dataset.py
├── pyimagesearch
│   ├── config.py
│   ├── create_dataloaders.py
│   └── food_classifier.py
├── results.png
└── train_distributed.py

2 directories, 9 files
```

首先也是最重要的，是`pyimagesearch`目录。它包含:

*   `config.py`:包含在整个项目中使用的几个重要参数和路径
*   包含一个函数，可以帮助我们加载、处理和操作数据集
*   `food_classifier.py`:驻留在这个脚本中的主模型架构

我们将使用的其他脚本在父目录中。它们是:

*   `train_distributed.py`:定义数据流程，训练我们的模型
*   `distributed_inference.py`:将用于评估我们训练好的模型的个别测试数据

最后，我们有我们的`output`文件夹，它将存放所有其他脚本产生的所有结果(图、模型)。

### **配置先决条件**

为了开始我们的实现，让我们从`config.py`开始，这个脚本将包含端到端训练和推理管道的配置。这些值将在整个项目中使用。

```py
# import the necessary packages
import torch
import os

# define path to the original dataset
DATA_PATH = "Food-11"

# define base path to store our modified dataset
BASE_PATH = "dataset"

# define paths to separate train, validation, and test splits
TRAIN = os.path.join(BASE_PATH, "training")
VAL = os.path.join(BASE_PATH, "validation")
TEST = os.path.join(BASE_PATH, "evaluation") 
```

我们定义了一个到原始数据集的路径(**行 6** )和一个基本路径(**行 9** )来存储修改后的数据集。在**第 12-14 行**上，我们使用`os.path.join`函数为修改后的数据集定义了单独的训练、验证和测试路径。

```py
# initialize the list of class label names
CLASSES = ["Bread", "Dairy_product", "Dessert", "Egg", "Fried_food",
    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup",
    "Vegetable/Fruit"]

# specify ImageNet mean and standard deviation and image size
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
```

在**第 17-19 行**，我们定义了我们的目标类。我们正在选择 11 个类，我们的数据集将被分组到这些类中。在**第 22-24 行**上，我们为我们的 **ImageNet** 输入指定平均值、标准偏差和图像大小值。请注意，平均值和标准偏差各有 3 个值。每个值分别代表*通道方向、高度方向、*和*宽度方向*的平均值和标准偏差。图像尺寸被设置为``224 × 224`` 以匹配 **ImageNet** 模型的可接受的通用输入尺寸。

```py
# set the device to be used for training and evaluation
DEVICE = torch.device("cuda")

# specify training hyperparameters
LOCAL_BATCH_SIZE = 128
PRED_BATCH_SIZE = 4
EPOCHS = 20
LR = 0.0001

# define paths to store training plot and trained model
PLOT_PATH = os.path.join("output", "model_training.png")
MODEL_PATH = os.path.join("output", "food_classifier.pth")
```

由于今天的任务涉及演示用于训练的多个图形处理单元，我们将设置`torch.device`到`cuda` ( **第 27 行**)。`cuda`是 NVIDIA 开发的一个巧妙的应用编程接口(API)，使**CUDA**(**Compute Unified Device Architecture)**的 GPU 被允许用于通用处理。此外，由于 GPU 比 CPU 拥有更多的带宽和内核，因此它们在训练机器学习模型方面速度更快。

在**的第 30-33 行**，我们设置了几个超参数，如`LOCAL_BATCH_SIZE`(训练期间的批量)、`PRED_BATCH_SIZE`(推断期间的批量)、时期和学习率。然后，在**第 36 行和第 37 行**，我们定义路径来存储我们的训练图和训练模型。前者将评估它相对于模型度量的表现，而后者将被调用到推理模块。

对于我们的下一个任务，我们将进入`create_dataloaders.py`脚本。

```py
# import the necessary packages
from . import config
from torch.utils.data import DataLoader
from torchvision import datasets
import os

def get_dataloader(rootDir, transforms, bs, shuffle=True):
	# create a dataset and use it to create a data loader
	ds = datasets.ImageFolder(root=rootDir,
		transform=transforms)
	loader = DataLoader(ds, batch_size=bs, shuffle=shuffle,
		num_workers=os.cpu_count(),
		pin_memory=True if config.DEVICE == "cuda" else False)

	# return a tuple of the dataset and the data loader
	return (ds, loader)
```

在第 7 行的**上，我们定义了一个名为`get_dataloader`的函数，它将根目录、PyTorch 的转换实例和批处理大小作为外部参数。**

在**的第 9 行和第 10 行**，我们使用`torchvision.datasets.ImageFolder`来映射给定目录中的所有条目，以拥有`__getitem__`和`__len__`方法。这些方法在这里有非常重要的作用。

首先，它们有助于在从索引到数据样本的类似地图的结构中表示数据集。

其次，新映射的数据集现在可以通过一个`torch.utils.data.DataLoader`实例(**第 11-13 行**)传递，它可以并行加载多个数据样本。

最后，我们返回数据集和`DataLoader`实例(**第 16 行**)。

### **为分布式训练准备数据集**

对于今天的教程，我们使用的是 Food-11 数据集。如果你想快速下载 Food-11 数据集，请参考 Adrian 关于使用 Keras 创建的[微调模型的这篇精彩博文！](https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning)

尽管数据集已经有了一个训练、测试和验证分割，但我们将以一种更易于理解的方式来组织它。

在其原始形式中，数据集的格式如图**图 3** 所示:

每个文件名的格式都是`class_index_imageNumber.jpg`。例如，文件`0_10.jpg`指的是属于`Bread`标签的图像。来自所有类别的图像被分组在一起。在我们的自定义数据集中，我们将按标签排列图像，并将它们放在各自带有标签名称的文件夹中。因此，在数据准备之后，我们的数据集结构将看起来像**图 4** :

每个标签式文件夹将包含属于这些标签的相应图像。这样做是因为许多现代框架和函数在处理输入时更喜欢这样的文件夹结构。

所以，让我们跳进我们的`prepare_dataset.py`脚本，并编码出来！

```py
# USAGE
# python prepare_dataset.py

# import the necessary packages
from pyimagesearch import config
from imutils import paths
import shutil
import os

def copy_images(rootDir, destiDir):
	# get a list of the all the images present in the directory
	imagePaths = list(paths.list_images(rootDir))
	print(f"[INFO] total images found: {len(imagePaths)}...")
```

我们首先定义一个函数`copy_images` ( **第 10 行**)，它有两个参数:图像所在的根目录和自定义数据集将被复制到的目标目录。然后，在**的第 12 行**，我们使用`paths.list_images`函数生成根目录中所有图像的列表。这将在以后复制文件时使用。

```py
      # loop over the image paths
	for imagePath in imagePaths:
		# extract class label from the filename
		filename = imagePath.split(os.path.sep)[-1]
		label = config.CLASSES[int(filename.split("_")[0])].strip()

		# construct the path to the output directory
		dirPath = os.path.sep.join([destiDir, label])

		# if the output directory does not exist, create it
		if not os.path.exists(dirPath):
			os.makedirs(dirPath)

		# construct the path to the output image file and copy it
		p = os.path.sep.join([dirPath, filename])
		shutil.copy2(imagePath, p)
```

我们开始遍历第 16 行的**图像列表。首先，我们通过分隔前面的路径名(**第 18 行**)挑出文件的确切名称，然后我们通过`filename.split("_")[0])`识别文件的标签，并将其作为索引馈送给`config.CLASSES`。在第一次循环中，该函数创建目录路径(**第 25 行和第 26 行**)。最后，我们构建当前图像的路径，并使用 [`shutil`](https://docs.python.org/3/library/shutil.html) 包将图像复制到目标路径。**

```py
	# calculate the total number of images in the destination
	# directory and print it
	currentTotal = list(paths.list_images(destiDir))
	print(f"[INFO] total images copied to {destiDir}: "
		f"{len(currentTotal)}...")

# copy over the images to their respective directories
print("[INFO] copying images...")
copy_images(os.path.join(config.DATA_PATH, "training"), config.TRAIN)
copy_images(os.path.join(config.DATA_PATH, "validation"), config.VAL)
copy_images(os.path.join(config.DATA_PATH, "evaluation"), config.TEST)
```

我们对第 34 行和第 35 行进行健全性检查，看看是否所有的文件都被复制了。这就结束了`copy_images`功能。我们调用**行 40-42** 上的函数，并创建我们修改后的**训练、测试、**和**验证**数据集！

### **创建 PyTorch 分类器**

既然我们的数据集创建已经完成，是时候进入`food_classifier.py`脚本并定义我们的分类器了。

```py
# import the necessary packages
from torch.cuda.amp import autocast
from torch import nn

class FoodClassifier(nn.Module):
	def __init__(self, baseModel, numClasses):
		super(FoodClassifier, self).__init__()

		# initialize the base model and the classification layer
		self.baseModel = baseModel
		self.classifier = nn.Linear(baseModel.classifier.in_features,
			numClasses)

		# set the classifier of our base model to produce outputs
		# from the last convolution block
		self.baseModel.classifier = nn.Identity()
```

我们首先定义我们的自定义`nn.Module`类(**第 5 行**)。这通常是在架构更复杂时完成的，在定义我们的模型时允许更大的灵活性。在类内部，我们的第一项工作是定义`__init__`函数来初始化对象的状态。

第 7 行上的`super`方法将允许访问基类的方法。然后，在第**行第 10** 行，我们将基本模型初始化为构造函数(`__init__`)中传递的`baseModel`参数。然后我们创建一个单独的分类输出层(**第 11 行**)，带有 **11 个输出**，每个输出代表我们之前定义的一个类。最后，由于我们使用了自己的分类层，我们用`nn.Identity`替换了`baseModel`的内置分类层，这只是一个占位符层。因此，`baseModel`的内置分类器将正好反映其分类层之前的卷积模块的输出。

```py
	# we decorate the *forward()* method with *autocast()* to enable 
	# mixed-precision training in a distributed manner
	@autocast()
	def forward(self, x):
		# pass the inputs through the base model and then obtain the
		# classifier outputs
		features = self.baseModel(x)
		logits = self.classifier(features)

		# return the classifier outputs
		return logits
```

在**第 21 行**上，我们定义了自定义模型的`forward()`道次，但在此之前，我们用`@autocast()`修饰模型。这个 decorator 函数在训练期间支持混合精度，由于数据类型的智能分配，这实质上使您的训练更快。我已经把链接到了 **TensorFlow** 的一个博客，里面详细解释了混合精度。最后，在**第 24 行和第 25 行**上，我们获得`baseModel`输出，并将其通过自定义`classifier`层以获得最终输出。

### **使用分布式训练来训练 PyTorch 分类器**

我们的下一个目的地是`train_distributed.py`，在那里我们将进行模型训练，并学习如何使用多个 GPU！

```py
# USAGE
# python train_distributed.py

# import the necessary packages
from pyimagesearch.food_classifier import FoodClassifier
from pyimagesearch import config
from pyimagesearch import create_dataloaders
from sklearn.metrics import classification_report
from torchvision.models import densenet121
from torchvision import transforms
from tqdm import tqdm
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

# determine the number of GPUs we have
NUM_GPU = torch.cuda.device_count()
print(f"[INFO] number of GPUs found: {NUM_GPU}...")

# determine the batch size based on the number of GPUs
BATCH_SIZE = config.LOCAL_BATCH_SIZE * NUM_GPU
print(f"[INFO] using a batch size of {BATCH_SIZE}...")
```

`torch.cuda.device_count()`函数(**第 20 行**)将列出我们系统中兼容 CUDA 的 GPU 数量。这将用于确定我们的全局批量大小(**行 24** )，即`config.LOCAL_BATCH_SIZE * NUM_GPU`。这是因为如果我们的全局批量大小是``B`` ，并且我们有``N`` 兼容 CUDA 的 GPU，那么每个 GPU 都会处理批量大小`B/N`的数据。例如，对于全局批量``12`` 和``2`` 兼容 CUDA 的 GPU，每个 GPU 都会评估批量``6`` 的数据。

```py
# define augmentation pipelines
trainTansform = transforms.Compose([
	transforms.RandomResizedCrop(config.IMAGE_SIZE),
	transforms.RandomHorizontalFlip(),
	transforms.RandomRotation(90),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])
testTransform = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])

# create data loaders
(trainDS, trainLoader) = create_dataloaders.get_dataloader(config.TRAIN,
	transforms=trainTansform, bs=BATCH_SIZE)
(valDS, valLoader) = create_dataloaders.get_dataloader(config.VAL,
	transforms=testTransform, bs=BATCH_SIZE, shuffle=False)
(testDS, testLoader) = create_dataloaders.get_dataloader(config.TEST,
	transforms=testTransform, bs=BATCH_SIZE, shuffle=False)
```

接下来，我们使用 PyTorch 的一个非常方便的函数，称为`torchvision.transforms`。它不仅有助于构建复杂的转换管道，而且还赋予我们对选择使用的转换的许多控制权。

注意**第 28-34 行**，我们为我们的训练集图像使用了几个数据增强，如`RandomHorizontalFlip`、`RandomRotation`等。我们还使用此函数将均值和标准差归一化值添加到我们的数据集。

我们再次使用`torchvision.transforms`进行测试转换(**第 35-39 行**，但是我们没有添加额外的扩充。相反，我们通过在`create_dataloaders`脚本中创建的`get_dataloader`函数传递这些实例，并分别获得训练、验证和测试数据集和数据加载器(**第 42-47 行**)。

```py
# load up the DenseNet121 model
baseModel = densenet121(pretrained=True)

# loop over the modules of the model and if the module is batch norm,
# set it to non-trainable
for module, param in zip(baseModel.modules(), baseModel.parameters()):
	if isinstance(module, nn.BatchNorm2d):
		param.requires_grad = False

# initialize our custom model and flash it to the current device
model = FoodClassifier(baseModel, len(trainDS.classes))
model = model.to(config.DEVICE)
```

我们选择`densenet121`作为我们的基础模型来覆盖我们模型架构的大部分( **Line 50** )。然后我们在`densenet121`层上循环，并将`batch_norm`层设置为不可训练(行 54-56 )。这样做是为了避免由于批次大小不同而导致的批次标准化不稳定的问题。一旦完成，我们将`densenet121`发送给`FoodClassifier`类，并初始化我们的定制模型( **Line 59** )。最后，我们将模型加载到我们的设备上(**第 60 行**)。

```py
# if we have more than one GPU then parallelize the model
if NUM_GPU > 1:
	model = nn.DataParallel(model)

# initialize loss function, optimizer, and gradient scaler
lossFunc = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=config.LR * NUM_GPU)
scaler = torch.cuda.amp.GradScaler(enabled=True)

# initialize a learning-rate (LR) scheduler to decay the it by a factor
# of 0.1 after every 10 epochs
lrScheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDS) // BATCH_SIZE
valSteps = len(valDS) // BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "train_acc": [], "val_loss": [],
	"val_acc": []}
```

首先，我们使用一个条件语句来检查我们的系统是否适合 PyTorch 数据并行(**行 63 和 64** )。如果条件为真，我们通过`nn.DataParallel`模块传递我们的模型，并将我们的模型并行化。然后，在**第 67-69 行**，我们定义我们的损失函数，优化器，并创建一个 PyTorch 渐变缩放器实例。梯度缩放器是一个非常有用的工具，有助于将混合精度引入梯度计算。然后，我们初始化一个学习率调度器，使其每 10 个历元衰减一个因子的值(**行 73** )。

在**第 76 行和第 77 行**，我们为训练和验证批次计算每个时期的步骤。第 80**行和第 81** 行的`H`变量将是我们的训练历史字典，包含训练损失、训练精度、验证损失和验证精度等值。

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
	for (x, y) in trainLoader:
		with torch.cuda.amp.autocast(enabled=True):
			# send the input to the device
			(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

			# perform a forward pass and calculate the training loss
			pred = model(x)
			loss = lossFunc(pred, y)

		# calculate the gradients
		scaler.scale(loss).backward()
		scaler.step(opt)
		scaler.update()
		opt.zero_grad()

		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss.item()
		trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()

	# update our LR scheduler
	lrScheduler.step()
```

为了评估我们的模型训练的速度有多快，我们对训练过程进行计时( **Line 85** )。为了开始我们的模型训练，我们开始在**第 87 行**循环我们的纪元。我们首先将 PyTorch 定制模型设置为训练模式(**第 89 行**)，并初始化训练和验证损失以及正确预测(**第 92-98 行**)。

然后，我们使用 train dataloader ( **Line 101** )循环我们的训练集。一旦进入训练集循环，我们首先启用混合精度(**行 102** )并将输入(数据和标签)加载到 CUDA 设备(**行 104** )。最后，在**第 107 行和第 108 行**，我们让我们的模型执行向前传递，并使用我们的损失函数计算损失。

`scaler.scale(loss).backward`函数自动为我们计算梯度(**第 111 行**)，然后我们将其插入模型权重并更新模型(**第 111-113 行**)。最后，我们在完成一遍后使用`opt.zero_grad`重置梯度，因为`backward`函数不断累积梯度(我们每次只需要逐步梯度)。

**第 118-120 行**更新我们的损失并校正预测值，同时在一次完整的训练通过后更新我们的 LR 调度器(**第 123 行**)。

```py
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()

		# loop over the validation set
		for (x, y) in valLoader:
			with torch.cuda.amp.autocast(enabled=True):
				# send the input to the device
				(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

				# make the predictions and calculate the validation
				# loss
				pred = model(x)
				totalValLoss += lossFunc(pred, y).item()

			# calculate the number of correct predictions
			valCorrect += (pred.argmax(1) == y).type(
				torch.float).sum().item()

	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps

	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDS)
	valCorrect = valCorrect / len(valDS)
```

在我们的评估过程中，我们将使用`torch.no_grad`关闭 PyTorch 的自动渐变，并将我们的模型切换到评估模式(**第 126-128 行**)。然后，在训练步骤中，我们循环验证数据加载器，并在将数据加载到我们的 CUDA 设备之前启用混合精度(**第 131-134 行**)。接下来，我们获得验证数据集的预测，并更新验证损失值(**第 138 和 139 行**)。

一旦脱离循环，我们计算训练和验证损失和预测的分批平均值(**行 146-151** )。

```py
	# update our training history
	H["train_loss"].append(avgTrainLoss)
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss)
	H["val_acc"].append(valCorrect)

	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
		avgValLoss, valCorrect))

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))
```

在我们的纪元循环结束之前，我们将所有损失和预测值记录到我们的历史字典`H` ( **第 154-157 行**)中。

一旦在循环之外，我们使用**行 167** 上的`time.time()`函数记录时间，看看我们的模型执行得有多快。

```py
# evaluate the network
print("[INFO] evaluating network...")
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()

	# initialize a list to store our predictions
	preds = []

	# loop over the test set
	for (x, _) in testLoader:
		# send the input to the device
		x = x.to(config.DEVICE)

		# make the predictions and add them to the list
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())

# generate a classification report
print(classification_report(testDS.targets, preds,
	target_names=testDS.classes))
```

现在是时候在测试数据上测试我们新训练的模型了。再次关闭自动梯度计算，我们将模型设置为评估模式(**第 173-175 行**)。

接下来，我们在**行 178** 上初始化一个名为`preds`的空列表，它将存储测试数据的模型预测。最后，我们遵循同样的程序，将数据加载到我们的设备中，获得批量测试数据的预测，并将值存储在`preds`列表中(**第 181-187 行**)。

在几个方便的工具[](https://scikit-learn.org/stable/)**中，scikit-learn`classification_report`为我们提供了评估我们的模型的工具，其中`classification_report`提供了我们的模型给出的预测的完整的分类概述(**第 190 和 191 行**)。**

```py
[INFO] evaluating network...
               precision    recall  f1-score   support

        Bread       0.92      0.88      0.90       368
Dairy_product       0.87      0.84      0.86       148
      Dessert       0.87      0.92      0.89       500
          Egg       0.94      0.92      0.93       335
   Fried_food       0.95      0.91      0.93       287
         Meat       0.93      0.95      0.94       432
      Noodles       0.97      0.99      0.98       147
         Rice       0.99      0.95      0.97        96
      Seafood       0.95      0.93      0.94       303
         Soup       0.96      0.98      0.97       500
    Vegetable       0.96      0.97      0.96       231

     accuracy                           0.93      3347
    macro avg       0.94      0.93      0.93      3347
 weighted avg       0.93      0.93      0.93      3347
```

我们的模型的完整分类报告应该是这样的，让我们对我们的模型比其他模型预测得更好/更差的类别有一个全面的了解。

```py
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
plt.savefig(config.PLOT_PATH)

# serialize the model state to disk
torch.save(model.module.state_dict(), config.MODEL_PATH)
```

我们训练脚本的最后一步是绘制模型历史字典中的值(**行 194-204** )，并将模型状态保存在我们预定义的路径中(**行 207** )。

### **使用 PyTorch 进行分布式训练**

在执行训练脚本之前，我们需要运行`prepare_dataset.py`脚本。

```py
$ python prepare_dataset.py
[INFO] copying images...
[INFO] total images found: 9866...
[INFO] total images copied to dataset/training: 9866...
[INFO] total images found: 3430...
[INFO] total images copied to dataset/validation: 3430...
[INFO] total images found: 3347...
[INFO] total images copied to dataset/evaluation: 3347...
```

一旦这个脚本运行完毕，我们就可以继续执行`train_distributed.py`脚本了。

```py
$ python train_distributed.py
[INFO] number of GPUs found: 4...
[INFO] using a batch size of 512...
[INFO] training the network...
  0%|                                                                        | 0/20 [00:00<?, ?it/s][INFO] EPOCH: 1/20
Train loss: 1.267870, Train accuracy: 0.6176
Val loss: 0.838317, Val accuracy: 0.7586
  5%|███▏                                                            | 1/20 [00:37<11:47, 37.22s/it][INFO] EPOCH: 2/20
Train loss: 0.669389, Train accuracy: 0.7974
Val loss: 0.580541, Val accuracy: 0.8394
 10%|██████▍                                                         | 2/20 [01:03<09:16, 30.91s/it][INFO] EPOCH: 3/20
Train loss: 0.545763, Train accuracy: 0.8305
Val loss: 0.516144, Val accuracy: 0.8580
 15%|█████████▌                                                      | 3/20 [01:30<08:14, 29.10s/it][INFO] EPOCH: 4/20
Train loss: 0.472342, Train accuracy: 0.8547
Val loss: 0.482138, Val accuracy: 0.8682
...
 85%|█████████████████████████████████████████████████████▌         | 17/20 [07:40<01:19, 26.50s/it][INFO] EPOCH: 18/20
Train loss: 0.226185, Train accuracy: 0.9338
Val loss: 0.323659, Val accuracy: 0.9099
 90%|████████████████████████████████████████████████████████▋      | 18/20 [08:06<00:52, 26.32s/it][INFO] EPOCH: 19/20
Train loss: 0.227704, Train accuracy: 0.9331
Val loss: 0.313711, Val accuracy: 0.9140
 95%|███████████████████████████████████████████████████████████▊   | 19/20 [08:33<00:26, 26.46s/it][INFO] EPOCH: 20/20
Train loss: 0.228238, Train accuracy: 0.9332
Val loss: 0.318986, Val accuracy: 0.9105
100%|███████████████████████████████████████████████████████████████| 20/20 [09:00<00:00, 27.02s/it]
[INFO] total time taken to train the model: 540.37s
```

经过 20 个时期后，平均训练精度达到了 **0.9332** ，而验证精度达到了值得称赞的 **0.9105** 。让我们先来看看**图 5** 中的度量图！

通过观察整个过程中训练和验证指标的发展，我们可以有把握地说，我们的模型没有过度拟合。

### **数据分布式训练推理**

虽然我们已经在测试集上评估了模型，但是我们将创建一个单独的脚本`distributed_inference.py`，其中我们将逐个单独评估测试图像，而不是一次评估一整批。

```py
# USAGE
# python distributed_inference.py

# import the necessary packages
from pyimagesearch.food_classifier import FoodClassifier
from pyimagesearch import config
from pyimagesearch import create_dataloaders
from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
import torch

# determine the number of GPUs we have
NUM_GPU = torch.cuda.device_count()
print(f"[INFO] number of GPUs found: {NUM_GPU}...")

# determine the batch size based on the number of GPUs
BATCH_SIZE = config.PRED_BATCH_SIZE * NUM_GPU
print(f"[INFO] using a batch size of {BATCH_SIZE}...")

# define augmentation pipeline
testTransform = transforms.Compose([
	transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])
```

在初始化迭代器之前，我们设置了这些脚本的初始需求。这些包括设置由 CUDA GPUs 数量决定的批量大小(**第 15-19 行**)和为我们的测试数据集初始化一个`torchvision.transforms`实例(**第 23-27 行**)。

```py
# calculate the inverse mean and standard deviation
invMean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
invStd = [1/s for s in config.STD]

# define our denormalization transform
deNormalize = transforms.Normalize(mean=invMean, std=invStd)

# create test data loader
(testDS, testLoader) = create_dataloaders.get_dataloader(config.TEST,
	transforms=testTransform, bs=BATCH_SIZE, shuffle=True)

# load up the DenseNet121 model
baseModel = models.densenet121(pretrained=True)

# initialize our food classifier
model = FoodClassifier(baseModel, len(testDS.classes))

# load the model state
model.load_state_dict(torch.load(config.MODEL_PATH))
```

理解我们为什么计算第 30 和 31 行**的**反均值**和**反标准差**值很重要。这是因为我们的`torchvision.transforms`实例在数据集被插入模型之前对其进行了规范化。所以，为了把图像变回原来的形式，我们要预先计算这些值。我们很快就会看到这些是如何使用的！**

用这些值，我们创建一个`torchvision.transforms.Normalize`实例供以后使用(**第 34 行**)。接下来，我们在第 37 和 38 行的**上使用`create_dataloaders`方法创建我们的测试数据集和数据加载器。**

注意，我们已经在`train_distributed.py`中保存了训练好的模型状态。接下来，我们将像在训练脚本中一样初始化模型(**第 41-44 行**)，并使用`model.load_state_dict`函数将训练好的模型权重插入到初始化的模型(**第 47 行**)。

```py
# if we have more than one GPU then parallelize the model
if NUM_GPU > 1:
	model = nn.DataParallel(model)

# move the model to the device and set it in evaluation mode
model.to(config.DEVICE)
model.eval()

# grab a batch of test data
batch = next(iter(testLoader))
(images, labels) = (batch[0], batch[1])

# initialize a figure
fig = plt.figure("Results", figsize=(10, 10 * NUM_GPU))
```

我们使用`nn.DataParallel`重复并行化模型，并将模型设置为评估模式(**第 50-55 行**)。因为我们将处理单个数据点，所以我们不需要遍历整个测试数据集。相反，我们将使用`next(iter(loader))` ( **第 58 行和第 59 行**)抓取一批测试数据。您可以运行此功能(直到发生器用完批次)来随机化批次选择。

```py
# switch off autograd
with torch.no_grad():
	# send the images to the device
	images = images.to(config.DEVICE)

	# make the predictions
	preds = model(images)

	# loop over all the batch
	for i in range(0, BATCH_SIZE):
		# initialize a subplot
		ax = plt.subplot(BATCH_SIZE, 1, i + 1)

		# grab the image, de-normalize it, scale the raw pixel
		# intensities to the range [0, 255], and change the channel
		# ordering from channels first to channels last
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

同样，由于我们不打算改变模型的权重，我们关闭了自动渐变功能( **Line 65** )，并将测试图像刷新到我们的设备中。最后，在**第 70 行**，我们直接对批量图像进行模型预测。

在批处理的图像中循环，我们选择单个图像，反规格化它们，放大它们的值并改变它们的尺寸顺序(**第 80-83 行**)。如果我们显示图像，改变尺寸是必要的，因为 **PyTorch** 选择将其模块设计为接受通道优先输入。也就是说，我们刚从`torchvision.transforms`出来的图像现在是`Channels * Height * Width`。为了显示它，我们必须以`Height * Width * Channels`的形式重新排列维度。

我们使用图像的单独标签，通过使用`testDS.classes` ( **第 86 行和第 87 行**)来获得类的名称。接下来，我们得到单个图像的预测类别(**第 90 行和第 91 行**)。最后，我们比较单个图像的真实和预测标签(**行 94-98** )。

这就结束了我们的数据并行训练的推理脚本！

### **数据并行训练模型的 PyTorch 可视化**

让我们看看我们的推理脚本`distributed_inference.py`绘制的一些结果。

由于我们已经在推理脚本中采用了 4 个的**批次大小，我们的绘图将显示当前批次的图片。**

发送给我们的推理脚本的那批数据包含:牡蛎壳的图像(**图 6** )、薯条的图像(**图 7** )、包含肉的图像(**图 8** )和巧克力蛋糕的图像(**图 9** )。

这里我们看到 **3 个预测是正确的，共 4 个。**这一点，加上我们完整的测试集分数，告诉我们使用 PyTorch 的数据并行非常有效！

## **总结**

在今天的教程中，我们领略了 **PyTorch 的**大量分布式训练程序。就内部工作而言，`nn.DataParallel`可能不是其他分布式培训程序中最有效或最快的，但它肯定是一个很好的起点！它很容易理解，只需要一行代码就可以实现。正如我之前提到的，其他过程需要更多的代码，但是它们是为了更有效地处理事情而创建的。

`nn.DataParallel`的一些非常明显的问题是:

*   创建整个模型实例本身的冗余
*   当模型变得太大而不适合时无法工作
*   当可用的 GPU 不同时，无法自适应地调整训练

尤其是在处理大型架构时，模型并行是首选，在这种情况下，您可以在 GPU 之间划分模型层。

也就是说，如果你的系统中有多个 GPU，那么使用`nn.DataParallel`充分利用你的系统所能提供的每一点计算能力。

我希望本教程对你有足够的帮助，为你从整体上掌握分布式培训的好奇心铺平道路！

### **引用信息**

**Chakraborty，d .**“py torch 分布式培训简介”， *PyImageSearch* ，2021，[https://PyImageSearch . com/2021/10/18/Introduction-to-Distributed-Training-in-py torch/](https://pyimagesearch.com/2021/10/18/introduction-to-distributed-training-in-pytorch/)

`@article{Chakraborty_2021_Distributed, author = {Devjyoti Chakraborty}, title = {Introduction to Distributed Training in {PyTorch}}, journal = {PyImageSearch}, year = {2021}, note = {https://pyimagesearch.com/2021/10/18/introduction-to-distributed-training-in-pytorch/}, }`

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******