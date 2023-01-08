# PyTorch:训练你的第一个卷积神经网络(CNN)

> 原文：<https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/>

在本教程中，您将获得关于使用 PyTorch 深度学习库训练您的第一个卷积神经网络(CNN)的温和介绍。这个网络将能够识别手写的平假名字符。

今天的教程是 PyTorch 基础知识五部分系列的第三部分:

1.  [*py torch 是什么？*](https://pyimagesearch.com/2021/07/05/what-is-pytorch/)
2.  [*py torch 简介:用 PyTorch*](https://pyimagesearch.com/2021/07/12/intro-to-pytorch-training-your-first-neural-network-using-pytorch/) 训练你的第一个神经网络
3.  *PyTorch:训练你的第一个卷积神经网络*(今天的教程)
4.  *使用预训练网络的 PyTorch 图像分类*(下周教程)
5.  *使用预训练网络的 PyTorch 对象检测*

上周你学习了[如何使用 PyTorch 库](https://pyimagesearch.com/2021/07/12/intro-to-pytorch-training-your-first-neural-network-using-pytorch/)训练一个非常基本的前馈神经网络。该教程侧重于简单的数字数据。

今天，我们将进行下一步，学习如何使用 Kuzushiji-MNIST (KMNIST)数据集训练 CNN 识别手写平假名字符。

正如您将看到的，在图像数据集上训练 CNN 与在数字数据上训练基本的多层感知器(MLP)没有什么不同。我们仍然需要:

1.  定义我们的模型架构
2.  从磁盘加载我们的数据集
3.  循环我们的纪元和批次
4.  预测并计算我们的损失
5.  适当地调零我们的梯度，执行反向传播，并更新我们的模型参数

此外，这篇文章还会给你一些 PyTorch 的`DataLoader`实现的经验，这使得*处理数据集变得超级容易*——精通 PyTorch 的`DataLoader`是你作为深度学习实践者想要发展的一项关键技能(这是我在 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)专门开设了一门课程的主题)。

**要学习如何用 PyTorch 训练你的第一个 CNN，** ***继续阅读。***

## **PyTorch:训练你的第一个卷积神经网络(CNN)**

在本教程的剩余部分，您将学习如何使用 PyTorch 框架训练您的第一个 CNN。

我们将首先配置我们的开发环境来安装`torch`和`torchvision`，然后回顾我们的项目目录结构。

然后，我将向您展示包含平假名字符的 KMNIST 数据集(MNIST 数字数据集的替代物)。在本教程的后面，您将学习如何训练 CNN 识别 KMNIST 数据集中的每个平假名字符。

然后，我们将使用 PyTorch 实现三个 Python 脚本，包括 CNN 架构、训练脚本和用于对输入图像进行预测的最终脚本。

在本教程结束时，您将熟悉使用 PyTorch 训练 CNN 所需的步骤。

我们开始吧！

### **配置您的开发环境**

要遵循本指南，您需要在系统上安装 PyTorch、OpenCV 和 scikit-learn。

幸运的是，使用 pip 安装这三个都非常容易:

```py
$ pip install torch torchvision
$ pip install opencv-contrib-python
$ pip install scikit-learn
```

**如果您需要帮助配置 PyTorch 的开发环境，我*强烈推荐*您** [**阅读 PyTorch 文档**](https://pytorch.org/get-started/locally/)**——py torch 的文档非常全面，可以让您快速上手并运行。**

 **如果你需要帮助安装 OpenCV，[一定要参考我的 *pip 安装 OpenCV* 教程](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **kmn ist 数据集**

我们今天使用的数据集是库祖什基-MNIST 数据集，简称 KMNIST。该数据集旨在替代标准的 MNIST 数字识别数据集。

KMNIST 数据集由 70，000 幅图像及其对应的标注组成(60，000 幅用于训练，10，000 幅用于测试)。

KMNIST 数据集中总共有 10 个类(即 10 个平假名字符)，每个类都是均匀分布和表示的。我们的目标是训练一个 CNN，它可以准确地对这 10 个角色进行分类。

幸运的是，KMNIST 数据集内置于 PyTorch 中，这让我们非常容易使用！

### **项目结构**

在我们开始实现任何 PyTorch 代码之前，让我们先回顾一下我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，以检索源代码和预训练模型。

然后，您将看到以下目录结构:

```py
$ tree . --dirsfirst
.
├── output
│   ├── model.pth
│   └── plot.png
├── pyimagesearch
│   ├── __init__.py
│   └── lenet.py
├── predict.py
└── train.py

2 directories, 6 files
```

我们今天要复习三个 Python 脚本:

1.  著名的 LeNet 架构的 PyTorch 实现
2.  `train.py`:使用 PyTorch 在 KMNIST 数据集上训练 LeNet，然后将训练好的模型序列化到磁盘(即`model.pth`)
3.  从磁盘加载我们训练好的模型，对测试图像进行预测，并在屏幕上显示结果

一旦我们运行`train.py`，目录`output`将被填充`plot.png`(我们的训练/验证损失和准确性的图表)和`model.pth`(我们的训练模型文件)。

回顾了我们的项目目录结构后，我们可以继续用 PyTorch 实现我们的 CNN。

### **用 PyTorch 实现卷积神经网络(CNN)**

我们在这里用 PyTorch 实现的卷积神经网络(CNN)是开创性的 [LeNet 架构](https://pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)，由深度学习的创始人之一 Yann LeCunn 首先提出。

按照今天的标准，LeNet 是一个*非常浅的*神经网络，由以下几层组成:

`(CONV => RELU => POOL) * 2 => FC => RELU => FC => SOFTMAX`

正如您将看到的，我们将能够用 PyTorch 用 60 行代码(包括注释)实现 LeNet。

用 PyTorch 学习 CNN 的最好方法是实现一个，所以说，打开`pyimagesearch`模块中的`lenet.py`文件，让我们开始工作:

```py
# import the necessary packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
```

**第 2-8 行**导入我们需要的包。让我们逐一分析:

*   `Module`:我们没有使用`Sequential` PyTorch 类来实现 LeNet，而是子类化了`Module`对象，这样你就可以看到 PyTorch 是如何使用类来实现神经网络的
*   `Conv2d` : PyTorch 实现[卷积层](https://pyimagesearch.com/2021/05/14/convolution-and-cross-correlation-in-neural-networks/)
*   `Linear`:全连接层
*   `MaxPool2d`:应用 2D 最大池来减少输入体积的空间维度
*   `ReLU`:我方 ReLU [激活功能](https://pyimagesearch.com/2021/05/14/convolutional-neural-networks-cnns-and-layer-types/)
*   `LogSoftmax`:在构建 softmax 分类器时使用，以返回每个类别的预测概率
*   `flatten`:展平多维体的输出(例如，CONV 或池层)，以便我们可以对其应用完全连接的层

有了我们的导入，我们可以使用 PyTorch 实现我们的`LeNet`类:

```py
class LeNet(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(LeNet, self).__init__()

		# initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
			kernel_size=(5, 5))
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		# initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=20, out_channels=50,
			kernel_size=(5, 5))
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		# initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=800, out_features=500)
		self.relu3 = ReLU()

		# initialize our softmax classifier
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)
```

**第 10 行**定义了`LeNet`类。注意我们是如何子类化`Module`对象的——通过将我们的模型构建为一个类，我们可以很容易地:

*   重用变量
*   实现定制函数来生成子网/组件(在实现更复杂的网络时经常使用*，如 ResNet、Inception 等)。)*
*   *定义我们自己的`forward`传递函数*

 ***最棒的是，当定义正确时，PyTorch 可以自动*****应用其签名的模块来执行自动微分——反向传播由 PyTorch 库为我们处理！***

 *`LeNet`的构造函数接受两个变量:

1.  `numChannels`:输入图像的通道数(`1`表示灰度，`3`表示 RGB)
2.  `classes`:数据集中唯一类标签的总数

**第 13 行**调用父构造函数(即`Module`)，它执行许多 PyTorch 特定的操作。

从那里，我们开始定义实际的 LeNet 架构。

第 16-19 行初始化我们的第一组`CONV => RELU => POOL`层。我们的第一个 CONV 层一共学习了 20 个滤镜，每个滤镜都是 *5×5* 。然后应用一个 ReLU 激活函数，接着是一个 *2×2* max-pooling 层，步长为 *2×2* 以减少我们输入图像的空间维度。

然后我们在第 22-25 行的**上有了第二组`CONV => RELU => POOL`层。**我们将 CONV 层中学习到的滤镜数量增加到 50，但是保持 *5×5* 的内核大小。再次应用 ReLU 激活，然后是最大池。

接下来是我们的第一组也是唯一一组完全连接的层(**28 和 29** )。我们定义该层的输入数量(`800`)以及我们期望的输出节点数量(`500`)。FC 层之后是 ReLu 激活。

最后，我们应用我们的 softmax 分类器(**第 32 行和第 33 行**)。将`in_features`的数量设置为`500`，这是来自上一层的*输出*维度。然后我们应用`LogSoftmax`,这样我们可以在评估过程中获得预测的概率。

**理解这一点很重要，在这一点上我们所做的只是*初始化变量*。**这些变量本质上是占位符。PyTorch 完全不知道网络架构是什么，只知道一些变量存在于类定义中。

**为了构建网络架构本身(也就是说，哪一层是其他层的输入)，我们需要覆盖`Module`类的`forward`方法。**

`forward`功能有多种用途:

1.  它通过类的构造函数(即`__init__`)中定义的变量将层/子网连接在一起
2.  它定义了网络体系结构本身
3.  它允许模型向前传递，产生我们的输出预测
4.  此外，由于 PyTorch 的自动模块，它允许我们执行自动区分和更新我们的模型权重

现在让我们检查一下`forward`功能:

```py
	def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)

		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)

		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = flatten(x, 1)
		x = self.fc1(x)
		x = self.relu3(x)

		# pass the output to our softmax classifier to get our output
		# predictions
		x = self.fc2(x)
		output = self.logSoftmax(x)

		# return the output predictions
		return output
```

`forward`方法接受一个参数`x`，它是网络的一批输入数据。

然后，我们将我们的`conv1`、`relu1`和`maxpool1`层连接在一起，形成网络的第一个`CONV => RELU => POOL`层(**第 38-40 行**)。

在**线 44-46** 上执行类似的操作，这次构建第二组`CONV => RELU => POOL`层。

此时，变量`x`是一个多维张量；然而，为了创建我们完全连接的层，我们需要将这个张量“展平”成本质上相当于 1D 值列表的东西——第 50 行**上的`flatten`函数**为我们处理这个操作。

从那里，我们将`fc1`和`relu3`层连接到网络架构(**线 51 和 52** )，然后连接最后的`fc2`和`logSoftmax` ( **线 56 和 57** )。

然后将网络的`output`返回给调用函数。

**我想再次重申*在构造函数*中初始化变量相对于*在`forward`函数中构建网络本身的重要性:***

*   你的`Module`的构造器只初始化你的层类型。PyTorch 跟踪这些变量，但是它不知道这些层是如何相互连接的。
*   为了让 PyTorch 理解您正在构建的网络架构，您定义了`forward`函数。
*   在`forward`函数中，你获取在你的构造函数中初始化的变量并连接它们。
*   PyTorch 可以使用您的网络进行预测，并通过自动签名模块进行自动反向传播

祝贺你用 PyTorch 实现了你的第一个 CNN！

### **用 PyTorch 创建我们的 CNN 培训脚本**

随着 CNN 架构的实现，我们可以继续用 PyTorch 创建我们的训练脚本。

打开项目目录结构中的`train.py`文件，让我们开始工作:

```py
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time
```

**第 2 行和第 3 行**导入`matplotlib`并设置合适的后台引擎。

从那里，我们导入了许多著名的包:

*   `LeNet`:我们的 PyTorch 实现了上一节中的 LeNet CNN
*   `classification_report`:用于显示我们测试集的详细分类报告
*   `random_split`:从一组输入数据中构建随机训练/测试分割
*   PyTorch 的*棒极了的*数据加载工具，让我们可以毫不费力地建立数据管道来训练我们的 CNN
*   `ToTensor`:一个预处理功能，自动将输入数据转换成 PyTorch 张量
*   内置于 PyTorch 库中的 Kuzushiji-MNIST 数据集加载器
*   我们将用来训练神经网络的优化器
*   PyTorch 的神经网络实现

现在让我们解析我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True,
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
```

我们有两个命令行参数需要解析:

1.  `--model`:训练后输出序列化模型的路径(我们将这个模型保存到磁盘，这样我们就可以用它在我们的`predict.py`脚本中进行预测)
2.  `--plot`:输出训练历史图的路径

继续，我们现在有一些重要的初始化要处理:

```py
# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**第 29-31 行**设置我们的初始学习率、批量大小和训练的时期数，而**第 34 和 35 行**定义我们的训练和验证分割大小(75%的训练，25%的验证)。

**第 38 行**然后决定我们的`device`(即，我们将使用我们的 CPU 还是 GPU)。

让我们开始准备数据集:

```py
# load the KMNIST dataset
print("[INFO] loading the KMNIST dataset...")
trainData = KMNIST(root="data", train=True, download=True,
	transform=ToTensor())
testData = KMNIST(root="data", train=False, download=True,
	transform=ToTensor())

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = random_split(trainData,
	[numTrainSamples, numValSamples],
	generator=torch.Generator().manual_seed(42))
```

**第 42-45 行**使用 PyTorch 在`KMNIST`类中的构建加载 KMNIST 数据集。

对于我们的`trainData`，我们设置`train=True`，而我们的`testData`加载了`train=False`。当处理 PyTorch 库中内置的数据集时，这些布尔值很方便。

`download=True`标志表示 PyTorch 将自动下载 KMNIST 数据集并缓存到磁盘，如果我们以前没有下载过的话。

还要注意`transform`参数——这里我们可以应用一些数据转换(超出了本教程的范围，但很快会涉及到)。我们需要的唯一转换是将 PyTorch 加载的 NumPy 数组转换为张量数据类型。

加载了我们的训练和测试集后，我们在第 49-53 行上驱动我们的训练和验证集。使用 PyTorch 的`random_split`函数，我们可以很容易地拆分我们的数据。

我们现在有三组数据:

1.  培养
2.  确认
3.  测试

下一步是为每一个创建一个`DataLoader`:

```py
# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True,
	batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE
```

构建`DataLoader`对象是在**行的 56-59 行完成的。**我们只为`trainDataLoader`设置了`shuffle=True`，因为我们的验证和测试集不需要洗牌。

我们还导出每个时期的训练步骤和验证步骤的数量(**行 62 和 63** )。

此时，我们的数据已经为训练做好了准备；然而，我们还没有一个模型来训练！

现在让我们初始化 LeNet:

```py
# initialize the LeNet model
print("[INFO] initializing the LeNet model...")
model = LeNet(
	numChannels=1,
	classes=len(trainData.dataset.classes)).to(device)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()

# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()
```

**67-69 行**初始化我们的`model`。由于 KMNIST 数据集是灰度的，我们设置`numChannels=1`。我们可以通过调用我们的`trainData`的`dataset.classes`来轻松设置`classes`的数量。

我们还调用`to(device)`将`model`移动到我们的 CPU 或 GPU。

第 72 和 73 行初始化我们的优化器和损失函数。我们将使用 Adam 优化器进行训练，并将负对数似然用于我们的损失函数。

**当我们在模型定义中结合`nn.NLLoss`类和`LogSoftmax`类时，我们得到分类交叉熵损失(它是*等价于*训练一个具有输出`Linear`层和`nn.CrossEntropyLoss`损失的模型)。基本上，PyTorch 允许你以两种不同的方式实现分类交叉熵。**

习惯于看到这两种方法，因为一些深度学习实践者(几乎是任意地)更喜欢其中一种。

然后我们初始化`H`，我们的训练历史字典(**第 76-81 行**)。在每个时期之后，我们将用给定时期的训练损失、训练准确度、测试损失和测试准确度来更新该字典。

最后，我们启动一个计时器来测量训练需要多长时间( **Line 85** )。

至此，我们所有的初始化都已完成，所以是时候训练我们的模型了。

***注意:*** *请确保您已经阅读了本系列的前一篇教程，*[*py torch 简介:使用 PyTorch*](https://pyimagesearch.com/2021/07/12/intro-to-pytorch-training-your-first-neural-network-using-pytorch/) *训练您的第一个神经网络，因为我们将在该指南中学习概念。*

以下是我们的培训循环:

```py
# loop over our epochs
for e in range(0, EPOCHS):
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
	for (x, y) in trainDataLoader:
		# send the input to the device
		(x, y) = (x.to(device), y.to(device))

		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = lossFn(pred, y)

		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		opt.zero_grad()
		loss.backward()
		opt.step()

		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()
```

在**第 88 行，**我们循环我们想要的历元数。

然后，我们继续:

1.  将模型置于`train()`模式
2.  初始化当前时期的训练损失和验证损失
3.  初始化当前纪元的正确训练和验证预测的数量

**第 102 行**展示了使用 PyTorch 的`DataLoader`类的好处——我们所要做的就是在`DataLoader`对象上开始一个`for`循环。PyTorch *自动*产生一批训练数据。在引擎盖下，`DataLoader`也在洗牌我们的训练数据(如果我们做任何额外的预处理或数据扩充，它也会在这里发生)。

对于每一批数据(**行 104** )，我们进行正向传递，获得我们的预测，并计算损失(**行 107 和 108** )。

**接下来是** ***所有重要步骤*** **之:**

1.  调零我们的梯度
2.  执行反向传播
3.  更新我们模型的权重

**说真的，别忘了这一步！**未能按照正确的顺序完成这三个步骤*将会导致错误的训练结果。每当你用 PyTorch 编写一个训练循环时，我强烈建议你在做任何事情之前插入那三行代码*，这样你就能被提醒确保它们在正确的位置。**

 **我们通过更新我们的`totalTrainLoss`和`trainCorrect`簿记变量来包装代码块。

此时，我们已经循环了当前时期训练集中的所有批次的数据，现在我们可以在验证集上评估我们的模型:

```py
	# switch off autograd for evaluation
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()

		# loop over the validation set
		for (x, y) in valDataLoader:
			# send the input to the device
			(x, y) = (x.to(device), y.to(device))

			# make the predictions and calculate the validation loss
			pred = model(x)
			totalValLoss += lossFn(pred, y)

			# calculate the number of correct predictions
			valCorrect += (pred.argmax(1) == y).type(
				torch.float).sum().item()
```

在验证或测试集上评估 PyTorch 模型时，您需要首先:

1.  使用`torch.no_grad()`上下文关闭梯度跟踪和计算
2.  将模型置于`eval()`模式

从那里，你循环所有的验证`DataLoader` ( **第 128 行)，**将数据移动到正确的`device` ( **第 130 行**)，并使用数据进行预测(**第 133 行**)并计算你的损失(**第 134 行**)。

然后你可以得到正确预测的总数(**行 137 和 138** )。

我们通过计算一些统计数据来完善我们的训练循环:

```py
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps

	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDataLoader.dataset)
	valCorrect = valCorrect / len(valDataLoader.dataset)

	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_acc"].append(valCorrect)

	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
		avgValLoss, valCorrect))
```

**第 141 行和第 142 行**计算我们的平均训练和验证损失。**第 146 行和第 146** 行做同样的事情，但是为了我们的训练和验证准确性。

然后，我们获取这些值并更新我们的训练历史字典(**行 149-152** )。

最后，我们在我们的终端上显示训练损失、训练精度、验证损失和验证精度(**行 149-152** )。

我们快到了！

既然训练已经完成，我们需要在*测试集*上评估我们的模型(之前我们只使用了训练集和验证集):

```py
# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# we can now evaluate the network on the test set
print("[INFO] evaluating network...")

# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()

	# initialize a list to store our predictions
	preds = []

	# loop over the test set
	for (x, y) in testDataLoader:
		# send the input to the device
		x = x.to(device)

		# make the predictions and add them to the list
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())

# generate a classification report
print(classification_report(testData.targets.cpu().numpy(),
	np.array(preds), target_names=testData.classes))
```

**第 162-164 行**停止我们的训练计时器，显示训练花了多长时间。

然后，我们设置另一个`torch.no_grad()`上下文，并将我们的模型置于`eval()`模式下(**第 170 行和第 172 行**)。

评估由以下人员执行:

1.  初始化一个列表来存储我们的预测( **Line 175**
2.  在我们的`testDataLoader` ( **行 178** 上循环
3.  将当前一批数据发送到适当的设备(**行 180** )
4.  对当前一批数据进行预测( **Line 183**
5.  用来自模型的顶级预测更新我们的`preds`列表(**行 184** )

最后，我们展示一个详细的`classification_report`。

这里我们要做的最后一步是绘制我们的训练和验证历史，然后将我们的模型权重序列化到磁盘:

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
plt.savefig(args["plot"])

# serialize the model to disk
torch.save(model, args["model"])
```

**第 191-201 行**为我们的培训历史生成一个`matplotlib`图。

然后，我们调用`torch.save`将我们的 PyTorch 模型权重保存到磁盘，这样我们就可以从磁盘加载它们，并通过*单独的* Python 脚本进行预测。

总的来说，回顾这个脚本向您展示了 PyTorch 在训练循环中给予您的更多控制——这既是一件*好的*事情，也是一件*坏的*事情:

*   如果你想让*完全控制*训练循环，并且需要实现自定义程序，那么*是好的*
*   当你的训练循环很简单，一个等价于`model.fit`的 Keras/TensorFlow 就足够了，这就*糟糕了*

正如我在本系列的第一部分中提到的， [*什么是 PyTorch*](https://pyimagesearch.com/2021/07/05/what-is-pytorch/) *，*py torch 和 Keras/TensorFlow 都不比对方好，只是每个库有不同的注意事项和用例。

### 用 PyTorch 训练我们的 CNN

我们现在准备使用 PyTorch 来训练我们的 CNN。

请务必访问本教程的 ***“下载”*** 部分，以检索本指南的源代码。

在那里，您可以通过执行以下命令来训练您的 PyTorch CNN:

```py
$ python train.py --model output/model.pth --plot output/plot.png
[INFO] loading the KMNIST dataset...
[INFO] generating the train-val split...
[INFO] initializing the LeNet model...
[INFO] training the network...
[INFO] EPOCH: 1/10
Train loss: 0.362849, Train accuracy: 0.8874
Val loss: 0.135508, Val accuracy: 0.9605

[INFO] EPOCH: 2/10
Train loss: 0.095483, Train accuracy: 0.9707
Val loss: 0.091975, Val accuracy: 0.9733

[INFO] EPOCH: 3/10
Train loss: 0.055557, Train accuracy: 0.9827
Val loss: 0.087181, Val accuracy: 0.9755

[INFO] EPOCH: 4/10
Train loss: 0.037384, Train accuracy: 0.9882
Val loss: 0.070911, Val accuracy: 0.9806

[INFO] EPOCH: 5/10
Train loss: 0.023890, Train accuracy: 0.9930
Val loss: 0.068049, Val accuracy: 0.9812

[INFO] EPOCH: 6/10
Train loss: 0.022484, Train accuracy: 0.9930
Val loss: 0.075622, Val accuracy: 0.9816

[INFO] EPOCH: 7/10
Train loss: 0.013171, Train accuracy: 0.9960
Val loss: 0.077187, Val accuracy: 0.9822

[INFO] EPOCH: 8/10
Train loss: 0.010805, Train accuracy: 0.9966
Val loss: 0.107378, Val accuracy: 0.9764

[INFO] EPOCH: 9/10
Train loss: 0.011510, Train accuracy: 0.9960
Val loss: 0.076585, Val accuracy: 0.9829

[INFO] EPOCH: 10/10
Train loss: 0.009648, Train accuracy: 0.9967
Val loss: 0.082116, Val accuracy: 0.9823

[INFO] total time taken to train the model: 159.99s
[INFO] evaluating network...
              precision    recall  f1-score   support

           o       0.93      0.98      0.95      1000
          ki       0.96      0.95      0.96      1000
          su       0.96      0.90      0.93      1000
         tsu       0.95      0.97      0.96      1000
          na       0.94      0.94      0.94      1000
          ha       0.97      0.95      0.96      1000
          ma       0.94      0.96      0.95      1000
          ya       0.98      0.95      0.97      1000
          re       0.95      0.97      0.96      1000
          wo       0.97      0.96      0.97      1000

    accuracy                           0.95     10000
   macro avg       0.95      0.95      0.95     10000
weighted avg       0.95      0.95      0.95     10000
```

在我的 CPU 上训练 CNN 花了大约 160 秒。使用我的 GPU 训练时间下降到≈82 秒。

在最后一个时期结束时，我们获得了 99.67%的训练精度和 98.23%的验证精度。

当我们在我们的测试集上进行评估时，我们达到了 **≈95%的准确率**，考虑到平假名字符的*复杂性*和我们的浅层网络架构的*简单性*，这已经相当不错了(使用更深的网络，如 VGG 启发的模型或 ResNet-like 将允许我们获得更高的准确率，但对于使用 PyTorch 的 CNN 入门来说，这些模型更复杂)。

此外，如**图 4** 所示，我们的训练历史曲线是平滑的，表明很少/没有过度拟合发生。

在进入下一部分之前，看一下您的`output`目录:

```py
$ ls output/
model.pth	plot.png
```

注意`model.pth`文件——这是我们保存到磁盘上的经过训练的 PyTorch 模型。我们将从磁盘加载该模型，并在下一节中使用它进行预测。

### **实施我们的 PyTorch 预测脚本**

我们在此审查的最后一个脚本将向您展示如何使用保存到磁盘的 PyTorch 模型进行预测。

打开项目目录结构中的`predict.py`文件，我们将开始:

```py
# set the numpy seed for better reproducibility
import numpy as np
np.random.seed(42)

# import the necessary packages
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2
```

**第 2-13 行**导入我们需要的 Python 包。我们将 NumPy 随机种子设置在脚本的顶部，以便在不同的机器上有更好的可重复性。

然后我们导入:

*   `DataLoader`:用于加载我们的 KMNIST 测试数据
*   `Subset`:建立测试数据的子集
*   `ToTensor`:将我们的输入数据转换成 PyTorch 张量数据类型
*   内置于 PyTorch 库中的 Kuzushiji-MNIST 数据集加载器
*   我们的 OpenCV 绑定，我们将使用它进行基本的绘图并在屏幕上显示输出图像

接下来是我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to the trained PyTorch model")
args = vars(ap.parse_args())
```

这里我们只需要一个参数，`--model`，即保存到磁盘上的经过训练的 PyTorch 模型的路径。想必这个开关会指向`output/model.pth`。

继续，让我们设定我们的`device`:

```py
# set the device we will be using to test the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the KMNIST dataset and randomly grab 10 data points
print("[INFO] loading the KMNIST test dataset...")
testData = KMNIST(root="data", train=False, download=True,
	transform=ToTensor())
idxs = np.random.choice(range(0, len(testData)), size=(10,))
testData = Subset(testData, idxs)

# initialize the test data loader
testDataLoader = DataLoader(testData, batch_size=1)

# load the model and set it to evaluation mode
model = torch.load(args["model"]).to(device)
model.eval()
```

第 22 行决定了我们是在 CPU 还是 GPU 上执行推理。

然后，我们从 KMNIST 数据集中加载测试数据到**线 26 和 27** 。我们使用`Subset`类在**的第 28 行和第 29 行**上从这个数据集中随机抽取总共`10`个图像(这创建了一个完整测试数据的较小“视图”)。

创建一个`DataLoader`来通过第 32 行**上的模型传递我们的测试数据子集。**

然后，我们从磁盘的第 35 行的**上加载我们的序列化 PyTorch 模型，并将其传递给相应的`device`。**

最后，`model`被置于评估模式(**行 36** )。

现在让我们对测试集的一个样本进行预测:

```py
# switch off autograd
with torch.no_grad():
	# loop over the test set
	for (image, label) in testDataLoader:
		# grab the original image and ground truth label
		origImage = image.numpy().squeeze(axis=(0, 1))
		gtLabel = testData.dataset.classes[label.numpy()[0]]

		# send the input to the device and make predictions on it
		image = image.to(device)
		pred = model(image)

		# find the class label index with the largest corresponding
		# probability
		idx = pred.argmax(axis=1).cpu().numpy()[0]
		predLabel = testData.dataset.classes[idx]
```

**第 39 行**关闭梯度跟踪，而**第 41 行**在测试集中的所有图像上循环。

对于每个图像，我们:

1.  抓取当前图像并将其转换成一个 NumPy 数组(这样我们以后可以用 OpenCV 在上面绘图)
2.  提取地面实况分类标签
3.  将`image`发送到适当的`device`
4.  使用我们训练过的 LeNet 模型对当前`image`进行预测
5.  提取预测概率最高的类别标签

剩下的只是一点想象:

```py
		# convert the image from grayscale to RGB (so we can draw on
		# it) and resize it (so we can more easily see it on our
		# screen)
		origImage = np.dstack([origImage] * 3)
		origImage = imutils.resize(origImage, width=128)

		# draw the predicted class label on it
		color = (0, 255, 0) if gtLabel == predLabel else (0, 0, 255)
		cv2.putText(origImage, gtLabel, (2, 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

		# display the result in terminal and show the input image
		print("[INFO] ground truth label: {}, predicted label: {}".format(
			gtLabel, predLabel))
		cv2.imshow("image", origImage)
		cv2.waitKey(0)
```

KMNIST 数据集中的每个图像都是单通道灰度图像；但是，我们想使用 OpenCV 的`cv2.putText`函数在`image`上绘制预测的类标签和真实标签。

为了在灰度图像上绘制 RGB 颜色，我们首先需要通过将灰度图像在深度方向上总共堆叠三次来创建灰度图像的 RGB 表示( **Line 58** )。

此外，我们调整了`origImage`的大小，以便我们可以更容易地在屏幕上看到它(默认情况下，KMNIST 图像只有 *28×28* 像素，很难看到，尤其是在高分辨率显示器上)。

从那里，我们确定文本`color`并在输出图像上绘制标签。

我们通过在屏幕上显示输出`origImage`来结束脚本。

### **用我们训练过的 PyTorch 模型进行预测**

我们现在准备使用我们训练过的 PyTorch 模型进行预测！

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和预训练的 PyTorch 模型。

从那里，您可以执行`predict.py`脚本:

```py
$ python predict.py --model output/model.pth
[INFO] loading the KMNIST test dataset...
[INFO] Ground truth label: ki, Predicted label: ki
[INFO] Ground truth label: ki, Predicted label: ki
[INFO] Ground truth label: ki, Predicted label: ki
[INFO] Ground truth label: ha, Predicted label: ha
[INFO] Ground truth label: tsu, Predicted label: tsu
[INFO] Ground truth label: ya, Predicted label: ya
[INFO] Ground truth label: tsu, Predicted label: tsu
[INFO] Ground truth label: na, Predicted label: na
[INFO] Ground truth label: ki, Predicted label: ki
[INFO] Ground truth label: tsu, Predicted label: tsu
```

正如我们的输出所示，我们已经能够使用 PyTorch 模型成功地识别每个平假名字符。

## **总结**

在本教程中，您学习了如何使用 PyTorch 深度学习库训练您的第一个卷积神经网络(CNN)。

您还学习了如何:

1.  将我们训练过的 PyTorch 模型保存到磁盘
2.  在一个*单独的* Python 脚本中从磁盘加载它
3.  使用 PyTorch 模型对图像进行预测

这种在训练后保存模型，然后加载它并使用模型进行预测的顺序，是一个您应该熟悉的过程-作为 PyTorch 深度学习实践者，您将经常这样做。

说到从磁盘加载保存的 PyTorch 模型，下周你将学习如何使用预先训练的 PyTorch 来识别日常生活中经常遇到的 1000 个图像类。这些模型可以为您节省大量时间和麻烦——它们高度准确，不需要您手动训练它们。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！**********