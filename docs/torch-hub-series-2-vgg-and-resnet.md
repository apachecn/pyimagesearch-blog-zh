# 火炬中心系列#2: VGG 和雷斯内特

> 原文：<https://pyimagesearch.com/2021/12/27/torch-hub-series-2-vgg-and-resnet/>

在[之前的教程](https://pyimagesearch.com/2021/12/20/torch-hub-series-1-introduction-to-torch-hub/)中，我们学习了火炬中心背后的精髓及其概念。然后，我们使用 Torch Hub 的复杂性发布了我们的模型，并通过它进行访问。但是，当我们的工作需要我们使用 Torch Hub 上众多功能强大的模型之一时，会发生什么呢？

在本教程中，我们将学习如何利用最常见的模型称为使用火炬枢纽的力量:VGG 和 ResNet 模型家族。我们将学习这些模型背后的核心思想，并针对我们选择的任务对它们进行微调。

本课是关于火炬中心的 6 部分系列的第 2 部分:

1.  [*火炬中心系列# 1:*](https://pyimagesearch.com/2021/12/20/torch-hub-series-1-introduction-to-torch-hub/)火炬中心介绍
2.  *火炬毂系列#2: VGG 和雷斯内特*(本教程)
3.  *火炬轮毂系列#3: YOLO v5 和 SSD*——*实物检测模型*
4.  *火炬轮毂系列# 4:*—*甘上模*
5.  *火炬轮毂系列# 5:MiDaS*——*深度估计模型*
6.  *火炬中枢系列#6:图像分割*

**要了解如何利用火炬枢纽来驾驭 VGG 网和雷斯网的力量，** ***继续阅读。***

## **火炬中心系列#2: VGG 和雷斯内特**

### **VGG 和雷斯内特**

说实话，在每一个深度学习爱好者的生活中，迁移学习总会发挥巨大的作用。我们并不总是拥有从零开始训练模型的必要硬件，尤其是在千兆字节的数据上。云环境确实让我们的生活变得更轻松，但它们的使用显然是有限的。

现在，你可能想知道我们是否必须在机器学习的旅程中尝试我们所学的一切。使用**图 1** 可以最好地解释这一点。

在机器学习领域，理论和实践是同等重要的。按照这种观念，硬件限制会严重影响你的机器学习之旅。谢天谢地，机器学习社区的好心人通过在互联网上上传预先训练好的模型权重来帮助我们绕过这些问题。这些模型在巨大的数据集上训练，使它们成为非常强大的特征提取器。

您不仅可以将这些模型用于您的任务，还可以将它们用作基准。现在，您一定想知道在特定数据集上训练的模型是否适用于您的问题所特有的任务。

这是一个非常合理的问题。但是想一想完整的场景。例如，假设您有一个在 ImageNet 上训练的模型(1400 万个图像和 20，000 个类)。在这种情况下，由于您的模型已经是一个熟练的特征提取器，因此针对类似的和更具体的图像分类对其进行微调将会给您带来好的结果。由于我们今天的任务是微调一个 VGG/雷斯网模型，我们将看到我们的模型从第一个纪元开始是多么熟练！

由于网上有大量预先训练好的模型权重，Torch Hub 可以识别所有可能出现的问题，并通过将整个过程浓缩到一行来解决它们。因此，您不仅可以在本地系统中加载 SOTA 模型，还可以选择是否需要对它们进行预训练。

事不宜迟，让我们继续本教程的先决条件。

### **配置您的开发环境**

要遵循这个指南，您需要在您的系统上安装 PyTorch 框架。

幸运的是，它是 pip 可安装的:

```py
$ pip install pytorch
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
$ tree .
.
├── inference.py
├── pyimagesearch
│   ├── classifier.py
│   ├── config.py
│   └── datautils.py
└── train.py

1 directory, 5 files
```

在`pyimagesearch`中，我们有 3 个脚本:

*   `classifier.py`:容纳项目的模型架构
*   `config.py`:包含项目的端到端配置管道
*   `datautils.py`:包含了我们将在项目中使用的两个数据实用函数

在父目录中，我们有两个脚本:

*   根据我们训练的模型权重进行推断
*   `train.py`:在所需数据集上训练模型

### **VGG 和 ResNet 架构概述**

论文[中介绍了 VGG16 架构的超深卷积网络用于大规模图像识别](https://arxiv.org/abs/1409.1556)它借鉴了 AlexNet 的核心思想，同时用多个`3×3`卷积滤波器取代了大尺寸的卷积滤波器。在图 3 中，我们可以看到完整的架构。

过多卷积滤波器的小滤波器尺寸和网络的深度架构胜过当时的许多基准模型。因此，直到今天，VGG16 仍被认为是 ImageNet 数据集的最新模型。

不幸的是，VGG16 有一些重大缺陷。首先，由于网络的性质，它有几个权重参数。这不仅使模型更重，而且增加了这些模型的推理时间。

理解了 VGG 篮网的局限性，我们继续关注他们的精神继承者；雷斯网。由何和介绍，ResNets 的想法背后的纯粹天才不仅在许多情况下超过了 Nets，而且他们的架构也使推理时间更快。

ResNets 背后的主要思想可以在**图 4** 中看到。

这种架构被称为“**剩余块**”正如您所看到的，一层的输出不仅会被提供给下一层，还会进行一次跳跃，并被提供给架构中的另一层。

现在，这个想法立刻消除了渐变消失的可能性。但是这里的主要思想是来自前面层的信息在后面的层中保持活跃。因此，精心制作的特征映射阵列在自适应地决定这些残余块层的输出中起作用。

ResNet 被证明是机器学习社区的一次巨大飞跃。ResNet 不仅在构思之初就超越了许多深度架构，而且还引入了一个全新的方向，告诉我们如何让深度架构变得更好。

有了这两个模型的基本思想之后，让我们开始编写代码吧！

### **熟悉我们的数据集**

对于今天的任务，我们将使用来自 Kaggle 的简单二元分类[狗&猫](https://www.kaggle.com/chetankv/dogs-cats-images)数据集。这个 217.78 MB 的数据集包含 10，000 幅猫和狗的图像，以 80-20 的训练与测试比率分割。训练集包含 4000 幅猫和 4000 幅狗的图像，而测试集分别包含 1000 幅猫和 1000 幅狗的图像。使用较小的数据集有两个原因:

*   微调我们的分类器将花费更少的时间
*   展示预训练模型适应具有较少数据的新数据集的速度

### **配置先决条件**

首先，让我们进入存储在`pyimagesearch`目录中的`config.py`脚本。该脚本将包含完整的训练和推理管道配置值。

```py
# import the necessary packages
import torch
import os

# define the parent data dir followed by the training and test paths
BASE_PATH = "dataset"
TRAIN_PATH = os.path.join(BASE_PATH, "training_set")
TEST_PATH = os.path.join(BASE_PATH, "test_set")

# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# specify training hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 128
PRED_BATCH_SIZE = 4
EPOCHS = 15
LR = 0.0001

# determine the device type 
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# define paths to store training plot and trained model
PLOT_PATH = os.path.join("output", "model_training.png")
MODEL_PATH = os.path.join("output", "model.pth")
```

我们首先在**第 6 行**初始化数据集的基本训练。然后，在**第 7 行和第 8 行**，我们使用`os.path.join`来指定数据集的训练和测试文件夹。

在第**行第 11 和 12** 行，我们指定了稍后创建数据集实例时所需的 ImageNet 平均值和标准偏差。这样做是因为模型是通过这些平均值和标准偏差值预处理的预训练数据，我们将尽可能使我们的当前数据与之前训练的数据相似。

接下来，我们为超参数赋值，如图像大小、批量大小、时期等。(**第 15-19 行**)并确定我们将训练我们模型的设备(**第 22 行**)。

我们通过指定存储训练图和训练模型权重的路径来结束我们的脚本(**行 25 和 26** )。

### **为我们的数据管道创建实用函数**

我们创建了一些函数来帮助我们处理数据管道，并在`datautils.py`脚本中对它们进行分组，以便更好地处理我们的数据。

```py
# import the necessary packages
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.datasets import Subset

def get_dataloader(dataset, batchSize, shuffle=True):
	# create a dataloader
	dl = DataLoader(dataset, batch_size=batchSize, shuffle=shuffle)

	# return the data loader
	return dl
```

我们的第一个效用函数是`get_dataloader`函数。它将数据集、批处理大小和一个布尔变量`shuffle`作为其参数(**第 6 行**)，并返回一个 PyTorch `dataloader`实例(**第 11 行**)。

```py
def train_val_split(dataset, valSplit=0.2):
	# grab the total size of the dataset
	totalSize = len(dataset)

	# perform training and validation split
	(trainIdx, valIdx) = train_test_split(list(range(totalSize)),
		test_size=valSplit)
	trainDataset = Subset(dataset, trainIdx)
	valDataset = Subset(dataset, valIdx)

	# return training and validation dataset
	return (trainDataset, valDataset)
```

接下来，我们创建一个名为`train_val_split`的函数，它接受数据集和一个验证分割百分比变量作为参数(**第 13 行**)。由于我们的数据集只有训练和测试目录，我们使用 PyTorch `dataset`子集特性将训练集分成训练和验证集。

我们首先使用`train_test_split`函数为我们的分割创建索引，然后将这些索引分配给子集(**第 20 行和第 21 行**)。该功能将返回训练和验证数据子集(**行 24** )。

### **为我们的任务**创建分类器

我们的下一个任务是为猫狗数据集创建一个分类器。请记住，我们不是从零开始训练我们的调用模型，而是对它进行微调。为此，我们将继续下一个脚本，即`classifier.py`。

```py
# import the necessary packages
from torch.nn import Linear
from torch.nn import Module

class Classifier(Module):
	def __init__(self, baseModel, numClasses, model):
		super().__init__()
		# initialize the base model 
		self.baseModel = baseModel

		# check if the base model is VGG, if so, initialize the FC
		# layer accordingly
		if model == "vgg":
			self.fc = Linear(baseModel.classifier[6].out_features, 
                                numClasses)

		# otherwise, the base model is of type ResNet so initialize
		# the FC layer accordingly
		else:
			self.fc = Linear(baseModel.fc.out_features, numClasses)
```

在我们的`Classifier`模块(**第 5 行**)中，构造函数接受以下参数:

*   因为我们将调用 VGG 或 ResNet 模型，我们已经覆盖了我们架构的大部分。我们将把调用的基础模型直接插入到我们的架构中的第 9 行上。
*   `numClasses`:将决定我们架构的输出节点。对于我们的任务，该值为 2。
*   一个字符串变量，它将告诉我们我们的基本模型是 VGG 还是 ResNet。因为我们必须为我们的任务创建一个单独的输出层，所以我们必须获取模型的最终线性层的输出。但是，每个模型都有不同的方法来访问最终的线性图层。因此，该`model`变量将有助于相应地选择特定于车型的方法(**第 14 行**和**第 20 行**)。

注意，对于 VGGnet，我们使用命令`baseModel.classifier[6].out_features`，而对于 ResNet，我们使用`baseModel.fc.out_features`。这是因为这些模型有不同的命名模块和层。所以我们必须使用不同的命令来访问每一层的最后一层。因此，`model`变量对于我们的代码工作非常重要。

```py
	def forward(self, x):
		# pass the inputs through the base model to get the features
		# and then pass the features through of fully connected layer
		# to get our output logits
		features = self.baseModel(x)
		logits = self.fc(features)

		# return the classifier outputs
		return logits
```

转到`forward`函数，我们简单地在**行 26** 上获得基本模型的输出，并通过我们最终的完全连接层(**行 27** )来获得模型输出。

### **训练我们的自定义分类器**

先决条件排除后，我们继续进行`train.py`。首先，我们将训练我们的分类器来区分猫和狗。

```py
# USAGE
# python train.py --model vgg
# python train.py --model resnet

# import the necessary packages
from pyimagesearch import config
from pyimagesearch.classifier import Classifier
from pyimagesearch.datautils import get_dataloader 
from pyimagesearch.datautils import train_val_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomRotation
from torchvision.transforms import Normalize
from torch.nn import CrossEntropyLoss
from torch.nn import Softmax
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import torch

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="vgg",
	choices=["vgg", "resnet"], help="name of the backbone model")
args = vars(ap.parse_args())
```

在**第 6-23 行**中，我们拥有培训模块所需的所有导入。不出所料，这是一个相当长的列表！

为了便于访问和选择，我们在第 26 行的**处创建了一个参数解析器，在第 27-29** 行的**处添加了参数模型选项(VGG 或雷斯尼)。**

接下来的一系列代码块是我们项目中非常重要的部分。例如，为了微调模型，我们通常冻结预训练模型的层。然而，在消融不同的场景时，我们注意到保持卷积层冻结，但是完全连接的层解冻以用于进一步的训练，这有助于我们的结果。

```py
# check if the name of the backbone model is VGG
if args["model"] == "vgg":
	# load VGG-11 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "vgg11",
		pretrained=True, skip_validation=True)

	# freeze the layers of the VGG-11 model
	for param in baseModel.features.parameters():
		param.requires_grad = False
```

从火炬中心调用的 VGG 模型架构(**线 34 和 35** )被分成几个子模块，以便于访问。卷积层被分组在一个名为`features`的模块下，而下面完全连接的层被分组在一个名为`classifier`的模块下。由于我们只需要冻结卷积层，我们直接访问第 38 行**上的参数，并通过将`requires_grad`设置为`False`来冻结它们，保持`classifier`模块层不变。**

```py
# otherwise, the backbone model we will be using is a ResNet
elif args["model"] == "resnet":
	# load ResNet 18 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "resnet18",
		pretrained=True, skip_validation=True)

	# define the last and the current layer of the model
	lastLayer = 8
	currentLayer = 1

	# loop over the child layers of the model
	for child in baseModel.children():

		# check if we haven't reached the last layer
		if currentLayer < lastLayer:
			# loop over the child layer's parameters and freeze them
			for param in child.parameters():
				param.requires_grad = False

		# otherwise, we have reached the last layers so break the loop
		else:
			break

		# increment the current layer
		currentLayer += 1   
```

在基本模型是 ResNet 的情况下，有几种方法可以解决这个问题。要记住的主要事情是，在 ResNet 中，我们只需要保持最后一个完全连接的层不冻结。相应地，在**第 48 行和第 49 行**，我们设置了最后一层和当前层索引。

在第 52 行**的 ResNet 的可用层上循环，我们冻结所有层，除了最后一层(**行 55-65** )。**

```py
# define the transform pipelines
trainTransform = Compose([
	RandomResizedCrop(config.IMAGE_SIZE),
	RandomHorizontalFlip(),
	RandomRotation(90),
	ToTensor(),
	Normalize(mean=config.MEAN, std=config.STD)
])

# create training dataset using ImageFolder
trainDataset = ImageFolder(config.TRAIN_PATH, trainTransform)
```

我们继续创建输入管道，从 PyTorch `transform`实例开始，它可以自动调整大小、规范化和增加数据，没有太多麻烦(**第 68-74 行**)。

我们通过使用另一个名为`ImageFolder`的 PyTorch 实用函数来完成它，该函数将自动创建输入和目标数据，前提是目录设置正确(**第 77 行**)。

```py
# create training and validation data split
(trainDataset, valDataset) = train_val_split(dataset=trainDataset)

# create training and validation data loaders
trainLoader = get_dataloader(trainDataset, config.BATCH_SIZE)
valLoader = get_dataloader(valDataset, config.BATCH_SIZE)
```

使用我们的`train_val_split`效用函数，我们将训练数据集分成一个训练和验证集(**第 80 行**)。接下来，我们使用来自`datautils.py`的`get_dataloader`实用函数来创建我们数据的 PyTorch `dataloader`实例(**第 83 行和第 84 行**)。这将允许我们以一种类似生成器的方式无缝地向模型提供数据。

```py
# build the custom model
model = Classifier(baseModel=baseModel.to(config.DEVICE),
	numClasses=2, model=args["model"])
model = model.to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = CrossEntropyLoss()
lossFunc.to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LR)

# initialize the softmax activation layer
softmax = Softmax()
```

继续我们的模型先决条件，我们创建我们的定制分类器并将其加载到我们的设备上(**第 87-89 行**)。

我们已经使用交叉熵作为我们今天任务的损失函数和 Adam 优化器(**第 92-94 行**)。此外，我们使用单独的`softmax`损失来帮助我们增加培训损失(**第 97 行**)。

```py
# calculate steps per epoch for training and validation set
trainSteps = len(trainDataset) // config.BATCH_SIZE
valSteps = len(valDataset) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {
	"trainLoss": [],
	"trainAcc": [],
	"valLoss": [],
	"valAcc": []
}
```

训练时期之前的最后一步是设置训练步骤和验证步骤值，然后创建一个存储所有训练历史的字典(**行 100-109** )。

```py
# loop over epochs
print("[INFO] training the network...")
for epoch in range(config.EPOCHS):
	# set the model in training mode
	model.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0

	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0
```

在训练循环中，我们首先将模型设置为训练模式( **Line 115** )。接下来，我们初始化训练损失、确认损失、训练和确认精度变量(**第 118-124 行**)。

```py
	# loop over the training set
	for (image, target) in tqdm(trainLoader):
		# send the input to the device
		(image, target) = (image.to(config.DEVICE),
			target.to(config.DEVICE))

		# perform a forward pass and calculate the training loss
		logits = model(image)
		loss = lossFunc(logits, target)

		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# add the loss to the total training loss so far, pass the
		# output logits through the softmax layer to get output
		# predictions, and calculate the number of correct predictions
		totalTrainLoss += loss.item()
		pred = softmax(logits)
		trainCorrect += (pred.argmax(dim=-1) == target).sum().item()
```

遍历完整的训练集，我们首先将数据和目标加载到设备中(**行 129 和 130** )。接下来，我们简单地通过模型传递数据并获得输出，然后将预测和目标插入我们的损失函数(**第 133 行和第 134 行**)，

**第 138-140 行**是标准 PyTorch 反向传播步骤，其中我们将梯度归零，执行反向传播，并更新权重。

接下来，我们将损失添加到我们的总训练损失中(**行 145** )，通过 softmax 传递模型输出以获得孤立的预测值，然后将其添加到`trainCorrect`变量中。

```py
	# switch off autograd
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()

		# loop over the validation set
		for (image, target) in tqdm(valLoader):
			# send the input to the device
			(image, target) = (image.to(config.DEVICE),
				target.to(config.DEVICE))

			# make the predictions and calculate the validation
			# loss
			logits = model(image)
			valLoss = lossFunc(logits, target)
			totalValLoss += valLoss.item()

			# pass the output logits through the softmax layer to get
			# output predictions, and calculate the number of correct
			# predictions
			pred = softmax(logits)
			valCorrect += (pred.argmax(dim=-1) == target).sum().item()
```

验证过程中涉及的大部分步骤与培训过程相同，除了以下几点:

*   模型被设置为评估模式(**行 152** )
*   权重没有更新

```py
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps

	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDataset)
	valCorrect = valCorrect / len(valDataset)

	# update our training history
	H["trainLoss"].append(avgTrainLoss)
	H["valLoss"].append(avgValLoss)
	H["trainAcc"].append(trainCorrect)
	H["valAcc"].append(valCorrect)

	# print the model training and validation information
	print(f"[INFO] EPOCH: {epoch + 1}/{config.EPOCHS}")
	print(f"Train loss: {avgTrainLoss:.6f}, Train accuracy: {trainCorrect:.4f}")
	print(f"Val loss: {avgValLoss:.6f}, Val accuracy: {valCorrect:.4f}")
```

在退出训练循环之前，我们计算平均损失(**行 173 和 174** )以及训练和验证精度(**行 177 和 178** )。

然后，我们继续将这些值添加到我们的训练历史字典中(**行 181-184** )。

```py
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["trainLoss"], label="train_loss")
plt.plot(H["valLoss"], label="val_loss")
plt.plot(H["trainAcc"], label="train_acc")
plt.plot(H["valAcc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)

# serialize the model state to disk
torch.save(model.module.state_dict(), config.MODEL_PATH)
```

在完成我们的训练脚本之前，我们绘制所有的训练字典变量(**行 192-201** )并保存图形(**行 202** )。

我们最后的任务是将模型权重保存在之前定义的路径中( **Line 205** )。

让我们看看每个时期的值是什么样的！

```py
[INFO] training the network...
100%|██████████| 50/50 [01:24<00:00,  1.68s/it]
100%|██████████| 13/13 [00:19<00:00,  1.48s/it]
[INFO] EPOCH: 1/15
Train loss: 0.289117, Train accuracy: 0.8669
Val loss: 0.217062, Val accuracy: 0.9119
100%|██████████| 50/50 [00:47<00:00,  1.05it/s]
100%|██████████| 13/13 [00:11<00:00,  1.10it/s]
[INFO] EPOCH: 2/15
Train loss: 0.212023, Train accuracy: 0.9039
Val loss: 0.223640, Val accuracy: 0.9025
100%|██████████| 50/50 [00:46<00:00,  1.07it/s]
100%|██████████| 13/13 [00:11<00:00,  1.15it/s]
[INFO] EPOCH: 3/15
...
Train loss: 0.139766, Train accuracy: 0.9358
Val loss: 0.187595, Val accuracy: 0.9194
100%|██████████| 50/50 [00:46<00:00,  1.07it/s]
100%|██████████| 13/13 [00:11<00:00,  1.15it/s]
[INFO] EPOCH: 13/15
Train loss: 0.134248, Train accuracy: 0.9425
Val loss: 0.146280, Val accuracy: 0.9437
100%|██████████| 50/50 [00:47<00:00,  1.05it/s]
100%|██████████| 13/13 [00:11<00:00,  1.12it/s]
[INFO] EPOCH: 14/15
Train loss: 0.132265, Train accuracy: 0.9428
Val loss: 0.162259, Val accuracy: 0.9319
100%|██████████| 50/50 [00:47<00:00,  1.05it/s]
100%|██████████| 13/13 [00:11<00:00,  1.16it/s]
[INFO] EPOCH: 15/15
Train loss: 0.138014, Train accuracy: 0.9409
Val loss: 0.153363, Val accuracy: 0.9313
```

我们预训练的模型精度在第一个历元就已经接近 **`90%`** 。到了第 **`13`个时期**，数值在大约`**~94%**`处饱和。从这个角度来看，在不同数据集上训练的预训练模型在它以前没有见过的数据集上以大约 **`86%`** 的精度开始。这就是它学会提取特征的程度。

在**图 5** 中绘制了指标的完整概述。

### **测试我们微调过的模型**

随着我们的模型准备就绪，我们将继续我们的推理脚本，`inference.py`。

```py
# USAGE
# python inference.py --model vgg
# python inference.py --model resnet

# import the necessary packages
from pyimagesearch import config
from pyimagesearch.classifier import Classifier
from pyimagesearch.datautils import get_dataloader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torchvision import transforms
from torch.nn import Softmax
from torch import nn
import matplotlib.pyplot as plt
import argparse
import torch
import tqdm

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="vgg",
	choices=["vgg", "resnet"], help="name of the backbone model")
args = vars(ap.parse_args())
```

因为我们必须在加载权重之前初始化我们的模型，所以我们需要正确的模型参数。为此，我们在第 23-26 行的**中创建了一个参数解析器。**

```py
# initialize test transform pipeline
testTransform = Compose([
	Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
	ToTensor(),
	Normalize(mean=config.MEAN, std=config.STD)
])

# calculate the inverse mean and standard deviation
invMean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
invStd = [1/s for s in config.STD]

# define our denormalization transform
deNormalize = transforms.Normalize(mean=invMean, std=invStd)

# create the test dataset
testDataset = ImageFolder(config.TEST_PATH, testTransform)

# initialize the test data loader
testLoader = get_dataloader(testDataset, config.PRED_BATCH_SIZE)
```

由于我们将在完整的测试数据集上计算我们的模型的准确性，我们在第 29-33 行的**上为我们的测试数据创建一个 PyTorch `transform`实例。**

因此，我们计算反平均值和反标准偏差值，我们用它们来创建一个`transforms.Normalize`实例(**第 36-40 行**)。这样做是因为数据在输入到模型之前经过了预处理。出于显示目的，我们必须将图像恢复到原始状态。

使用`ImageFolder`实用函数，我们创建我们的测试数据集实例，并将其提供给之前为测试`dataLoader`实例创建的`get_dataloader`函数(**第 43-46 行**)。

```py
# check if the name of the backbone model is VGG
if args["model"] == "vgg":
	# load VGG-11 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "vgg11",
		pretrained=True, skip_validation=True)

# otherwise, the backbone model we will be using is a ResNet
elif args["model"] == "resnet":
	# load ResNet 18 model
	baseModel = torch.hub.load("pytorch/vision:v0.10.0", "resnet18",
		pretrained=True, skip_validation=True)
```

如前所述，由于我们必须再次初始化模型，我们检查给定的模型参数，并相应地使用 Torch Hub 加载模型(**第 49-58 行**)。

```py
# build the custom model
model = Classifier(baseModel=baseModel.to(config.DEVICE),
	numClasses=2, vgg = False)
model = model.to(config.DEVICE)

# load the model state and initialize the loss function
model.load_state_dict(torch.load(config.MODEL_PATH))
lossFunc = nn.CrossEntropyLoss()
lossFunc.to(config.DEVICE)

# initialize test data loss
testCorrect = 0
totalTestLoss = 0
soft = Softmax()
```

在**第 61-66 行**上，我们初始化模型，将其存储在我们的设备上，并在模型训练期间加载先前获得的权重。

正如我们在`train.py`脚本中所做的，我们选择交叉熵作为我们的损失函数(**第 67 行**，并初始化测试损失和准确性(**第 71 行和第 72 行**)。

```py
# switch off autograd
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()

	# loop over the validation set
	for (image, target) in tqdm(testLoader):
		# send the input to the device
		(image, target) = (image.to(config.DEVICE), 
			target.to(config.DEVICE))

		# make the predictions and calculate the validation
		# loss
		logit = model(image)
		loss = lossFunc(logit, target)

		totalTestLoss += loss.item()

		# output logits through the softmax layer to get output
		# predictions, and calculate the number of correct predictions
		pred = soft(logit)
		testCorrect += (pred.argmax(dim=-1) == target).sum().item()
```

关闭自动梯度(**行 76** )，我们在**行 78 将模型设置为评估模式。**然后，在测试图像上循环，我们将它们提供给模型，并通过损失函数传递预测和目标(**第 81-89 行)**。

通过 softmax 函数(**第 95 和 96 行)**传递预测来计算精确度。

```py
# print test data accuracy		
print(testCorrect/len(testDataset))

# initialize iterable variable
sweeper = iter(testLoader)

# grab a batch of test data
batch = next(sweeper)
(images, labels) = (batch[0], batch[1])

# initialize a figure
fig = plt.figure("Results", figsize=(10, 10 ))
```

现在我们将看看测试数据的一些具体情况并显示它们。为此，我们在**行 102** 上初始化一个可迭代变量，并抓取一批数据(**行 105** )。

```py
# switch off autograd
with torch.no_grad():
	# send the images to the device
	images = images.to(config.DEVICE)

	# make the predictions
	preds = model(images)

	# loop over all the batch
	for i in range(0, config.PRED_BATCH_SIZE):
		# initialize a subplot
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
		gtLabel = testDataset.classes[idx]

		# grab the predicted label
		pred = preds[i].argmax().cpu().numpy()
		predLabel = testDataset.classes[pred]

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

我们再次关闭自动渐变，并对之前获取的一批数据进行预测(**第 112-117 行**)。

在批处理中循环，我们抓取单个图像，反规格化它们，重新缩放它们，并固定它们的尺寸以使它们可显示(**行 127-130** )。

基于当前正在考虑的图像，我们首先抓取它的地面真实标签(**行 133 和 134** )以及它们在**行 137 和 138** 上对应的预测标签，并相应地显示它们(**行 141-145** )。

### **微调模型的结果**

在整个测试数据集上，我们的 ResNet 支持的定制模型产生了 97.5%的准确率。在**图 6-9** 中，我们看到显示的一批数据，以及它们对应的基础事实和预测标签。

凭借 97.5%的准确度，您可以放心，这一性能水平不仅适用于该批次，还适用于所有*T2 批次。您可以重复运行`sweeper`变量来获得不同的数据批次，以便自己查看。*

## **总结**

今天的教程不仅展示了如何利用 Torch Hub 的模型库，还提醒了我们预先训练的模型在我们日常的机器学习活动中有多大的帮助。

想象一下，如果您必须为您选择的任何任务从头开始训练一个像 ResNet 这样的大型架构。这将需要更多的时间，而且肯定需要更多的纪元。至此，您肯定会欣赏 PyTorch Hub 背后的理念，即让使用这些最先进模型的整个过程更加高效。

从我们上一周的教程中离开的地方继续，我想强调 PyTorch Hub 仍然很粗糙，它仍然有很大的改进空间。当然，我们离完美的版本越来越近了！

### **引用信息**

**Chakraborty，D.** “火炬中心系列#2: VGG 和雷斯内特”， *PyImageSearch* ，2021，[https://PyImageSearch . com/2021/12/27/Torch-Hub-Series-2-vgg-and-ResNet/](https://pyimagesearch.com/2021/12/27/torch-hub-series-2-vgg-and-resnet/)

```py
@article{dev_2021_THS2,
  author = {Devjyoti Chakraborty},
  title = {{Torch Hub} Series \#2: {VGG} and {ResNet}},
  journal = {PyImageSearch},
  year = {2021},
  note = {https://pyimagesearch.com/2021/12/27/torch-hub-series-2-vgg-and-resnet/},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****