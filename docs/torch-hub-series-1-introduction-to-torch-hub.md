# 火炬中心系列#1:火炬中心简介

> 原文：<https://pyimagesearch.com/2021/12/20/torch-hub-series-1-introduction-to-torch-hub/>

在本教程中，您将学习 PyTorch 火炬中心的基础知识。

本课是关于火炬中心的 6 部分系列的第 1 部分:

1.  *火炬轮毂系列#1:火炬轮毂简介*(本教程)
2.  *火炬中心系列#2: VGG 和雷斯内特*
3.  *火炬轮毂系列#3: YOLO v5 和 SSD——物体检测模型*
4.  *火炬轮毂系列# 4:PGAN——甘模型*
5.  *火炬轮毂系列# 5:MiDaS——深度估计模型*
6.  *火炬中枢系列#6:图像分割*

**要学习如何使用火炬中枢，** ***只要坚持阅读。***

## **火炬中心介绍**

那是 2020 年，我和我的朋友们夜以继日地完成我们最后一年的项目。像我们这一年的大多数学生一样，我们决定把它留到最后是个好主意。

这不是我们最明智的想法。接下来是永无休止的模型校准和训练之夜，烧穿千兆字节的云存储，并维护深度学习模型结果的记录。

我们为自己创造的环境不仅损害了我们的效率，还影响了我们的士气。由于我的其他队友的个人才华，我们设法完成了我们的项目。

回想起来，我意识到如果我们选择了一个更好的生态系统来工作，我们的工作会更有效率，也更令人愉快。

幸运的是，你不必犯和我一样的错误。

PyTorch 的创建者经常强调，这一计划背后的一个关键意图是弥合研究和生产之间的差距。PyTorch 现在在许多领域与它的同时代人站在一起，在研究和生产生态系统中被平等地利用。

他们实现这一目标的方法之一是通过火炬中心。火炬中心作为一个概念，是为了进一步扩展 PyTorch 作为一个基于生产的框架的可信度。在今天的教程中，我们将学习如何利用 Torch Hub 来存储和发布预先训练好的模型，以便广泛使用。

### 什么是火炬中心？

在计算机科学中，许多人认为研究和生产之间桥梁的一个关键拼图是可重复性。基于这一理念，PyTorch 推出了 Torch Hub，这是一个应用程序可编程接口(API ),它允许两个程序相互交互，并增强了工作流程，便于研究再现。

Torch Hub 允许您发布预先训练好的模型，以帮助研究共享和再现。利用 Torch Hub 的过程很简单，但是在继续之前，让我们配置系统的先决条件！

### **配置您的开发环境**

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果你需要帮助为 OpenCV 配置开发环境，我*强烈推荐*阅读我们的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让你启动并运行。

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

我们首先需要回顾我们的项目结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

在转到目录之前，我们先来看看**图 2** 中的项目结构。

今天，我们将使用两个目录。这有助于您更好地理解 Torch Hub 的使用。

子目录是我们初始化和训练模型的地方。在这里，我们将创建一个`hubconf.py`脚本。`hubconf.py`脚本包含名为`entry_points`的可调用函数。这些可调用的函数初始化并返回用户需要的模型。因此，这个脚本将把我们自己创建的模型连接到 Torch Hub。

在我们的主项目目录中，我们将使用`torch.hub.load`从 Torch Hub 加载我们的模型。在用预先训练的权重加载模型之后，我们将在一些样本数据上对其进行评估。

### 火炬中心概观

Torch Hub 已经托管了一系列用于各种任务的模型，如图 3 所示。

如你所见，Torch Hub 在其官方展示中总共接受了 42 个研究模型。每个模型属于以下一个或多个标签:音频、生成、自然语言处理(NLP)、可脚本化和视觉。这些模型也已经在广泛接受的基准数据集上进行了训练(例如， [Kinetics 400](https://deepmind.com/research/open-source/kinetics) 和 [COCO 2017](https://cocodataset.org/#home) )。

使用`torch.hub.load`函数很容易在您的项目中使用这些模型。让我们来看一个它是如何工作的例子。

我们将查看火炬中心的官方文件[usi](https://pyimagesearch.com/2021/10/25/training-a-dcgan-in-pytorch/)[ng a DCGAN](https://pyimagesearch.com/2021/10/25/training-a-dcgan-in-pytorch/)在 [fashion-gen](https://arxiv.org/abs/1806.08317) 上接受培训来生成一些图片。

(*如果你想了解更多关于 DCGANs 的信息，一定要看看这个* [*博客*](https://pyimagesearch.com/2021/10/25/training-a-dcgan-in-pytorch/) *。)*

```py
# USAGE
# python inference.py

# import the necessary packages
import matplotlib.pyplot as plt
import torchvision
import argparse
import torch

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-images", type=int, default=64,
	help="# of images you want the DCGAN to generate")
args = vars(ap.parse_args())

# check if gpu is available for use
useGpu = True if torch.cuda.is_available() else False

# load the DCGAN model
model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub", "DCGAN",
	pretrained=True, useGPU=useGpu)
```

在**的第 11-14 行**，我们创建了一个参数解析器，让用户可以更自由地选择生成图像的批量大小。

要使用[脸书研究所](https://research.fb.com/)预训练的 DCGAN 模型，我们只需要`torch.hub.load`函数，如**第 20 和 21 行所示。**这里的`torch.hub.load`函数接受以下参数:

*   `repo_or_dir`:如果`source`参数被设置为`github`，则格式为`repo_owner/repo_name:branch/tag_name`的存储库名称。否则，它将指向您的本地计算机中所需的路径。
*   `entry_point`:要在 torch hub 中发布模型，您需要在您的存储库/目录中有一个名为`hubconf.py`的脚本。在该脚本中，您将定义称为入口点的普通可调用函数。调用入口点以返回期望的模型。稍后你会在这篇博客中了解更多关于`entry_point`的内容。
*   `pretrained`和`useGpu`:这些属于这个函数的`*args`或参数旗帜。这些参数用于可调用模型。

现在，这不是火炬中心提供的唯一主要功能。您可以使用其他几个值得注意的函数，比如`torch.hub.list`来列出所有属于存储库的可用入口点(可调用函数),以及`torch.hub.help`来显示目标入口点的文档 docstring。

```py
# generate random noise to input to the generator
(noise, _) = model.buildNoiseData(args["num_images"])

# turn off autograd and feed the input noise to the model
with torch.no_grad():
	generatedImages = model.test(noise)

# reconfigure the dimensions of the images to make them channel 
# last and display the output
output = torchvision.utils.make_grid(generatedImages).permute(
	1, 2, 0).cpu().numpy()
plt.imshow(output)
plt.show()
```

在第 24 行的**上，我们使用一个名为`buildNoiseData`的被调用模型专有的函数来生成随机输入噪声，同时牢记输入大小。**

关闭自动渐变( **Line 27** )，我们通过给模型添加噪声来生成图像。

在绘制图像之前，我们对第 32-35 行的**图像进行了维度整形(由于 PyTorch 使用通道优先张量，我们需要使它们再次成为通道最后张量)。输出将类似于**图 4** 。**

瞧吧！这就是你使用预先训练的最先进的 DCGAN 模型所需要的一切。在火炬中心使用预先训练好的模型是*那*容易。然而，我们不会就此止步，不是吗？

打电话给一个预先训练好的模型来看看最新的最先进的研究表现如何是好的，但是当我们使用我们的研究产生最先进的结果时呢？为此，我们接下来将学习如何在 Torch Hub 上发布我们自己创建的模型。

### **在 PyTorch 车型上使用 Torch Hub**

让我们回到 2021 年 7 月 12 日，Adrian Rosebrock [发布了一篇博文，教你如何在 PyTorch 上构建一个简单的 2 层神经网络](https://pyimagesearch.com/2021/07/12/intro-to-pytorch-training-your-first-neural-network-using-pytorch/)。该博客教你定义自己的简单神经网络，并在用户生成的数据上训练和测试它们。

今天，我们将训练我们的简单神经网络，并使用 Torch Hub 发布它。我不会对代码进行全面剖析，因为已经有相关教程了。关于构建一个简单的神经网络的详细而精确的探究，请参考[这篇博客](https://pyimagesearch.com/2021/07/12/intro-to-pytorch-training-your-first-neural-network-using-pytorch/)。

### **构建简单的神经网络**

接下来，我们将检查代码的突出部分。为此，我们将进入子目录。首先，让我们在`mlp.py`中构建我们的简单神经网络！

```py
# import the necessary packages
from collections import OrderedDict
import torch.nn as nn

# define the model function
def get_training_model(inFeatures=4, hiddenDim=8, nbClasses=3):
	# construct a shallow, sequential neural network
	mlpModel = nn.Sequential(OrderedDict([
		("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
		("activation_1", nn.ReLU()),
		("output_layer", nn.Linear(hiddenDim, nbClasses))
	]))

	# return the sequential model
	return mlpModel
```

**行 6** 上的`get_training_model`函数接受参数(输入大小、隐藏层大小、输出类别)。在函数内部，我们使用`nn.Sequential`创建一个 2 层神经网络，由一个带有 ReLU activator 的隐藏层和一个输出层组成(**第 8-12 行**)。

### **训练神经网络**

我们不会使用任何外部数据集来训练模型。相反，我们将自己生成数据点。让我们进入`train.py`。

```py
# import the necessary packages
from pyimagesearch import mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch
import os

# define the path to store your model weights
MODEL_PATH = os.path.join("output", "model_wt.pth")

# data generator function
def next_batch(inputs, targets, batchSize):
    # loop over the dataset
	for i in range(0, inputs.shape[0], batchSize):
		# yield a tuple of the current batched data and labels
		yield (inputs[i:i + batchSize], targets[i:i + batchSize])

# specify our batch size, number of epochs, and learning rate
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-2

# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))
```

首先，我们在**第 11 行**创建一个路径来保存训练好的模型权重，稍后会用到。第 14-18 行**上的`next_batch`函数将作为我们项目的数据生成器，产生用于高效训练的批量数据。**

接下来，我们设置超参数(**第 21-23 行**)，如果有兼容的 GPU 可用，则将我们的`DEVICE`设置为`cuda`(**第 26 行**)。

```py
# generate a 3-class classification problem with 1000 data points,
# where each data point is a 4D feature vector
print("[INFO] preparing data...")
(X, y) = make_blobs(n_samples=1000, n_features=4, centers=3,
	cluster_std=2.5, random_state=95)

# create training and testing splits, and convert them to PyTorch
# tensors
(trainX, testX, trainY, testY) = train_test_split(X, y,
	test_size=0.15, random_state=95)
trainX = torch.from_numpy(trainX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()
```

在**的第 32 行和第 33 行**，我们使用`make_blobs`函数来模拟实际三类数据集的数据点。使用 [scikit-learn 的](https://scikit-learn.org/stable/) `train_test_split`函数，我们创建数据的训练和测试分割。

```py
# initialize our model and display its architecture
mlp = mlp.get_training_model().to(DEVICE)
print(mlp)

# initialize optimizer and loss function
opt = SGD(mlp.parameters(), lr=LR)
lossFunc = nn.CrossEntropyLoss()

# create a template to summarize current training progress
trainTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
```

在**第 45 行**上，我们从`mlp.py`模块调用`get_training_model`函数并初始化模型。

我们选择随机梯度下降作为优化器(**第 49 行**)，交叉熵损失作为损失函数(**第 50 行**)。

第 53 行**上的`trainTemplate`变量将作为字符串模板打印精度和损耗。**

```py
# loop through the epochs
for epoch in range(0, EPOCHS):
	# initialize tracker variables and set our model to trainable
	print("[INFO] epoch: {}...".format(epoch + 1))
	trainLoss = 0
	trainAcc = 0
	samples = 0
	mlp.train()

	# loop over the current batch of data
	for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):
		# flash data to the current device, run it through our
		# model, and calculate loss
		(batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
		predictions = mlp(batchX)
		loss = lossFunc(predictions, batchY.long())

		# zero the gradients accumulated from the previous steps,
		# perform backpropagation, and update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()

		# update training loss, accuracy, and the number of samples
		# visited
		trainLoss += loss.item() * batchY.size(0)
		trainAcc += (predictions.max(1)[1] == batchY).sum().item()
		samples += batchY.size(0)

	# display model progress on the current training batch
	trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
	print(trainTemplate.format(epoch + 1, (trainLoss / samples),
		(trainAcc / samples)))
```

循环训练时期，我们初始化损失(**行 59-61** )并将模型设置为训练模式(**行 62** )。

使用`next_batch`函数，我们遍历一批批训练数据(**第 65 行**)。在将它们加载到设备(**线 68** )后，在**线 69** 上获得数据批次的预测。这些预测然后被输入到损失函数中进行损失计算(**第 70 行**)。

使用`zero_grad` ( **线 74** )冲洗梯度，然后在**线 75** 上反向传播。最后，在**行 76** 上更新优化器参数。

对于每个时期，训练损失、精度和样本大小变量被升级(**行 80-82** )，并使用**行 85** 上的模板显示。

```py
	# initialize tracker variables for testing, then set our model to
	# evaluation mode
	testLoss = 0
	testAcc = 0
	samples = 0
	mlp.eval()

	# initialize a no-gradient context
	with torch.no_grad():
		# loop over the current batch of test data
		for (batchX, batchY) in next_batch(testX, testY, BATCH_SIZE):

			# flash the data to the current device
			(batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))

			# run data through our model and calculate loss
			predictions = mlp(batchX)
			loss = lossFunc(predictions, batchY.long())

			# update test loss, accuracy, and the number of
			# samples visited
			testLoss += loss.item() * batchY.size(0)
			testAcc += (predictions.max(1)[1] == batchY).sum().item()
			samples += batchY.size(0)

		# display model progress on the current test batch
		testTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
		print(testTemplate.format(epoch + 1, (testLoss / samples),
			(testAcc / samples)))
		print("")

# save model to the path for later use
torch.save(mlp.state_dict(), MODEL_PATH)
```

我们将模型设置为`eval`模式进行模型评估，并在训练阶段进行同样的操作，除了反向传播。

在**第 121** 行，我们有保存模型权重以备后用的最重要步骤。

让我们来评估我们的模型的划时代的性能！

```py
[INFO] training using cpu...
[INFO] preparing data...
Sequential(
  (hidden_layer_1): Linear(in_features=4, out_features=8, bias=True)
  (activation_1): ReLU()
  (output_layer): Linear(in_features=8, out_features=3, bias=True)
)
[INFO] epoch: 1...
epoch: 1 train loss: 0.798 train accuracy: 0.649
epoch: 1 test loss: 0.788 test accuracy: 0.613

[INFO] epoch: 2...
epoch: 2 train loss: 0.694 train accuracy: 0.665
epoch: 2 test loss: 0.717 test accuracy: 0.613

[INFO] epoch: 3...
epoch: 3 train loss: 0.635 train accuracy: 0.669
epoch: 3 test loss: 0.669 test accuracy: 0.613
...
[INFO] epoch: 7...
epoch: 7 train loss: 0.468 train accuracy: 0.693
epoch: 7 test loss: 0.457 test accuracy: 0.740

[INFO] epoch: 8...
epoch: 8 train loss: 0.385 train accuracy: 0.861
epoch: 8 test loss: 0.341 test accuracy: 0.973

[INFO] epoch: 9...
epoch: 9 train loss: 0.286 train accuracy: 0.980
epoch: 9 test loss: 0.237 test accuracy: 0.993

[INFO] epoch: 10...
epoch: 10 train loss: 0.211 train accuracy: 0.985
epoch: 10 test loss: 0.173 test accuracy: 0.993
```

因为我们是根据我们设定的范例生成的数据进行训练，所以我们的训练过程很顺利，最终达到了 0.985 的**训练精度。**

### **配置`hubconf.py`脚本**

模型训练完成后，我们的下一步是在 repo 中配置`hubconf.py`文件，使我们的模型可以通过 Torch Hub 访问。

```py
# import the necessary packages
import torch
from pyimagesearch import mlp

# define entry point/callable function 
# to initialize and return model
def custom_model():
	""" # This docstring shows up in hub.help()
	Initializes the MLP model instance
	Loads weights from path and
	returns the model
	"""
	# initialize the model
	# load weights from path
	# returns model
	model = mlp.get_training_model()
	model.load_state_dict(torch.load("model_wt.pth"))
	return model
```

如前所述，我们在 7 号线的**上创建了一个名为`custom_model`的入口点。在`entry_point`内部，我们从`mlp.py`模块(**第 16 行**)初始化简单的神经网络。接下来，我们加载之前保存的权重(**第 17 行**)。当前的设置使得这个函数将在您的项目目录中寻找模型权重。您可以在云平台上托管权重，并相应地选择路径。**

现在，我们将使用 Torch Hub 访问该模型，并在我们的数据上测试它。

### **用`torch.hub.load`来称呼我们的模型**

回到我们的主项目目录，让我们进入`hub_usage.py`脚本。

```py
# USAGE
# python hub_usage.py

# import the necessary packages
from pyimagesearch.data_gen import next_batch
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import argparse
import torch

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--batch-size", type=int, default=64,
	help="input batch size")
args = vars(ap.parse_args())
```

在导入必要的包之后，我们为用户创建一个参数解析器(**第 13-16 行**)来输入数据的批量大小。

```py
# load the model using torch hub
print("[INFO] loading the model using torch hub...")
model = torch.hub.load("cr0wley-zz/torch_hub_test:main",
	"custom_model")

# generate a 3-class classification problem with 1000 data points,
# where each data point is a 4D feature vector
print("[INFO] preparing data...")
(X, Y) = make_blobs(n_samples=1000, n_features=4, centers=3,
	cluster_std=2.5, random_state=95)

# create training and testing splits, and convert them to PyTorch
# tensors
(trainX, testX, trainY, testY) = train_test_split(X, Y,
	test_size=0.15, random_state=95)
trainX = torch.from_numpy(trainX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()
```

在**的第 20 行和第 21 行**，我们使用`torch.hub.load`来初始化我们自己的模型，就像前面我们加载 DCGAN 模型一样。模型已经初始化，权重已经根据子目录中的`hubconf.py`脚本中的入口点加载。正如你所注意到的，我们给子目录`github`作为参数。

现在，为了评估模型，我们将按照我们在模型训练期间创建的相同方式创建数据(**第 26 行和第 27 行**)，并使用`train_test_split`创建数据分割(**第 31-36 行**)。

```py
# initialize the neural network loss function
lossFunc = nn.CrossEntropyLoss()

# set device to cuda if available and initialize
# testing loss and accuracy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
testLoss = 0
testAcc = 0
samples = 0

# set model to eval and grab a batch of data
print("[INFO] setting the model in evaluation mode...")
model.eval()
(batchX, batchY) = next(next_batch(testX, testY, args["batch_size"]))
```

在**第 39 行**上，我们初始化交叉熵损失函数，如在模型训练期间所做的。我们继续初始化**第 44-46 行**的评估指标。

将模型设置为评估模式(**行 50** )，并抓取单批数据供模型评估(**行 51** )。

```py
# initialize a no-gradient context
with torch.no_grad():
	# load the data into device
	(batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))

	# pass the data through the model to get the output and calculate
	# loss
	predictions = model(batchX)
	loss = lossFunc(predictions, batchY.long())

	# update test loss, accuracy, and the number of
	# samples visited
	testLoss += loss.item() * batchY.size(0)
	testAcc += (predictions.max(1)[1] == batchY).sum().item()
	samples += batchY.size(0)
	print("[INFO] test loss: {:.3f}".format(testLoss / samples))
	print("[INFO] test accuracy: {:.3f}".format(testAcc / samples))
```

关闭自动梯度(**行 54** )，我们将该批数据加载到设备中，并将其输入到模型中(**行 56-60** )。`lossFunc`继续计算**线 61** 上的损耗。

在损失的帮助下，我们更新了第 66 行**上的精度变量，以及一些其他度量，如样本大小(**行 67** )。**

让我们看看这个模型的效果如何！

```py
[INFO] loading the model using torch hub...
[INFO] preparing data...
[INFO] setting the model in evaluation mode...
Using cache found in /root/.cache/torch/hub/cr0wley-zz_torch_hub_test_main
[INFO] test loss: 0.086
[INFO] test accuracy: 0.969
```

由于我们使用训练模型时使用的相同范例创建了我们的测试数据，因此它的表现与预期一致，测试精度为 0.969 。

### 摘要

在当今的研究领域，结果的再现是多么重要，这一点我怎么强调都不为过。特别是在机器学习方面，我们已经慢慢地达到了一个新的研究想法变得日益复杂的地步。在这种情况下，研究人员拥有一个平台来轻松地将他们的研究和结果公之于众，这是一个巨大的负担。

作为一名研究人员，当您已经有足够多的事情要担心时，拥有一个工具，使用一个脚本和几行代码来公开您的模型和结果，对我们来说是一个巨大的福音。当然，作为一个项目，随着时间的推移，火炬中心将更多地根据用户的需求进行发展。尽管如此，Torch Hub 的创建所倡导的生态系统将帮助未来几代人的机器学习爱好者。

### **引用信息**

Chakraborty，D. **“火炬中心系列#1:火炬中心简介”， *PyImageSearch* ，2021，[https://PyImageSearch . com/2021/12/20/Torch-Hub-Series-1-Introduction-to-Torch-Hub/](https://pyimagesearch.com/2021/12/20/torch-hub-series-1-introduction-to-torch-hub/)**

```py
@article{dev_2021_THS1,
   author = {Devjyoti Chakraborty},
   title = {{Torch Hub} Series \#1: Introduction to {Torch Hub}},
   journal = {PyImageSearch},
   year = {2021},
   note = {https://pyimagesearch.com/2021/12/20/torch-hub-series-1-introduction-to-torch-hub/},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****