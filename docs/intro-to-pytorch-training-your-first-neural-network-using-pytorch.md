# PyTorch 简介:使用 PyTorch 训练您的第一个神经网络

> 原文：<https://pyimagesearch.com/2021/07/12/intro-to-pytorch-training-your-first-neural-network-using-pytorch/>

在本教程中，您将学习如何使用 PyTorch 深度学习库来训练您的第一个神经网络。

本教程是 PyTorch 深度学习基础知识五部分系列的第二部分:

1.  [*py torch 是什么？*](https://pyimagesearch.com/2021/07/05/what-is-pytorch/)
2.  PyTorch 简介:使用 PyTorch 训练你的第一个神经网络(今天的教程)
3.  *PyTorch:训练你的第一个卷积神经网络*(下周教程)
4.  *使用预训练网络的 PyTorch 图像分类*
5.  *使用预训练网络的 PyTorch 对象检测*

本指南结束时，您将学会:

1.  如何用 PyTorch 定义一个基本的神经网络架构
2.  如何定义你的损失函数和优化器
3.  **如何正确地调零你的梯度，执行反向传播，并更新你的模型参数** —大多数刚接触 PyTorch 的深度学习实践者都会在这一步犯错误

**要学习如何用 PyTorch 训练你的第一个神经网络，** ***继续阅读。***

## **PyTorch 简介:使用 py torch 训练你的第一个神经网络**

在本指南中，您将熟悉 PyTorch 中的常见程序，包括:

1.  定义您的神经网络架构
2.  初始化优化器和损失函数
3.  循环你的训练次数
4.  在每个时期内循环数据批次
5.  对当前一批数据进行预测和计算损失
6.  归零你的梯度
7.  执行反向传播
8.  告诉优化器更新网络的梯度
9.  告诉 PyTorch 用 GPU 训练你的网络(当然，如果你的机器上有 GPU 的话)

我们将首先回顾我们的项目目录结构，然后配置我们的开发环境。

从这里，我们将实现两个 Python 脚本:

1.  第一个脚本将是我们简单的前馈神经网络架构，用 Python 和 PyTorch 库实现
2.  然后，第二个脚本将加载我们的示例数据集，并演示如何训练我们刚刚使用 PyTorch 实现的网络架构

随着我们的两个 Python 脚本的实现，我们将继续训练我们的网络。我们将讨论我们的结果来结束本教程。

我们开始吧！

### **配置您的开发环境**

要遵循本指南，您需要在系统上安装 PyTorch 深度学习库和 scikit-machine 学习包。

幸运的是，PyTorch 和 scikit-learn 都非常容易使用 pip 安装:

```py
$ pip install torch torchvision
$ pip install scikit-image
```

**如果您需要帮助配置 PyTorch 的开发环境，我*强烈推荐*您** [**阅读 PyTorch 文档**](https://pytorch.org/get-started/locally/)**——py torch 的文档非常全面，可以让您快速上手并运行。**

 **### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

要跟随本教程，请务必访问本指南的 ***“下载”*** 部分以检索源代码。

然后，您将看到下面的目录结构。

```py
$ tree . --dirsfirst
.
├── pyimagesearch
│   └── mlp.py
└── train.py

1 directory, 2 files
```

文件将存储我们的基本多层感知器(MLP)的实现。

然后我们将实现`train.py`，它将用于在一个示例数据集上训练我们的 MLP。

### **用 PyTorch 实现我们的神经网络**

您现在已经准备好用 PyTorch 实现您的第一个神经网络了！

这个网络是一个非常简单的前馈神经网络，称为**多层感知器(MLP)** (意味着它有一个或多个隐藏层)。在下周的教程中，您将学习如何构建更高级的神经网络架构。

要开始构建我们的 PyTorch 神经网络，打开项目目录结构的`pyimagesearch`模块中的`mlp.py`文件，让我们开始工作:

```py
# import the necessary packages
from collections import OrderedDict
import torch.nn as nn

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

**第 2 行和第 3 行**导入我们需要的 Python 包:

*   `OrderedDict`:一个字典对象，*记住对象被添加的顺序*——我们使用这个有序字典为网络中的每一层提供人类可读的名称
*   PyTorch 的神经网络实现

然后我们定义了接受三个参数的`get_training_model`函数( **Line 5** ):

1.  神经网络的输入节点数
2.  网络隐藏层中的节点数
3.  输出节点的数量(即输出预测的维度)

根据提供的默认值，您可以看到我们正在构建一个 4-8-3 神经网络，这意味着输入层有 4 个节点，隐藏层有 8 个节点，神经网络的输出将由 3 个值组成。

然后，通过首先初始化一个`nn.Sequential`对象(非常类似于 Keras/TensorFlow 的`Sequential`类)，在**第 7-11 行**上构建实际的神经网络架构。

在`Sequential`类中，我们构建了一个`OrderedDict`，其中字典中的每个条目都包含两个值:

1.  一个包含人类可读的层名称的字符串(当使用 PyTorch 调试神经网络架构时，这个名称非常有用)
2.  PyTorch 层定义本身

`Linear`类是我们的**全连接**层定义，这意味着该层中的每个输入连接到每个输出。`Linear`类接受两个必需的参数:

1.  层的**输入**的数量
2.  **输出的数量**

在**线 8** 上，我们定义了`hidden_layer_1`，它由一个接受`inFeatures` (4)输入然后产生`hiddenDim` (8)输出的全连接层组成。

从那里，我们应用一个 ReLU 激活函数(**第 9 行**)，接着是另一个`Linear`层，作为我们的输出(**第 10 行**)。

**注意，第二个`Linear`定义包含与*之前的* `Linear`层输出相同的*数量的输入——这不是偶然的！***前一层的输出尺寸*必须*匹配下一层的输入尺寸，否则 PyTorch 将出错(然后您将有一个相当繁琐的任务，自己调试层尺寸)。

PyTorch 在这方面*没有*宽容(相对于 Keras/TensorFlow)，所以*在指定你的图层尺寸时要格外小心*。

然后，将生成的 PyTorch 神经网络返回给调用函数。

### **创建我们的 PyTorch 培训脚本**

随着我们的神经网络架构的实现，我们可以继续使用 PyTorch 来训练模型。

为了完成这项任务，我们需要实施一个培训脚本，该脚本:

1.  创建我们的神经网络架构的一个实例
2.  构建我们的数据集
3.  确定我们是否在 GPU 上训练我们的模型
4.  定义一个训练循环(我们脚本中最难的部分)

打开`train.py`，让我们开始吧:

```py
# import the necessary packages
from pyimagesearch import mlp
from torch.optim import SGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import torch.nn as nn
import torch
```

**第 2-7 行**导入我们需要的 Python 包，包括:

*   我们对多层感知器架构的定义，在 PyTorch 中实现
*   `SGD`:我们将用来训练我们的模型的[随机梯度下降](https://pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/)优化器
*   `make_blobs`:建立示例数据的合成数据集
*   `train_test_split`:将我们的数据集分成训练和测试部分
*   `nn` : PyTorch 的神经网络功能
*   `torch`:基地 PyTorch 图书馆

当训练一个神经网络时，我们通过*批*数据来完成(正如你之前所学的)。下面的函数`next_batch`为我们的训练循环产生这样的批次:

```py
def next_batch(inputs, targets, batchSize):
	# loop over the dataset
	for i in range(0, inputs.shape[0], batchSize):
		# yield a tuple of the current batched data and labels
		yield (inputs[i:i + batchSize], targets[i:i + batchSize])
```

`next_batch`函数接受三个参数:

1.  `inputs`:我们对神经网络的输入数据
2.  `targets`:我们的目标输出值(即，我们希望我们的神经网络准确预测的值)
3.  `batchSize`:数据批量的大小

然后我们在`batchSize`块中循环输入数据(**第 11 行**)，并将它们交给调用函数(**第 13 行**)。

接下来，我们要处理一些重要的初始化:

```py
# specify our batch size, number of epochs, and learning rate
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-2

# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))
```

当使用 PyTorch 训练我们的神经网络时，我们将使用 64 的批量大小，训练 10 个时期，并使用 1e-2 的学习速率(**第 16-18 行**)。

我们将我们的训练设备(CPU 或 GPU)设置在**第 21 行。**GPU 当然会加快训练速度，但是*在这个例子中不是必需的*。

接下来，我们需要一个示例数据集来训练我们的神经网络。在本系列的下一篇教程中，我们将学习如何从磁盘加载图像并对图像数据训练神经网络，但现在，让我们使用 [scikit-learn 的 make_blobs 函数](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)为我们创建一个合成数据集:

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

**第 27 行和第 28 行**构建我们的数据集，包括:

1.  三级标签(`centers=3`)
2.  神经网络的四个总特征/输入(`n_features=4`)
3.  总共 1000 个数据点(`n_samples=1000`)

从本质上来说，`make_blobs`函数正在生成聚集数据点的高斯斑点。对于 2D 数据，`make_blobs`函数将创建如下所示的数据:

注意这里有三组数据。我们在做同样的事情，但是我们有四维而不是二维(这意味着我们不容易想象它)。

一旦我们的数据生成，我们应用`train_test_split`函数(**第 32 行和第 33 行**)来创建我们的训练分割，85%用于训练，15%用于评估。

从那里，训练和测试数据从 NumPy 数组转换为 PyTorch 张量，然后转换为浮点数据类型(**第 34-37 行**)。

现在让我们实例化我们的 PyTorch 神经网络架构:

```py
# initialize our model and display its architecture
mlp = mlp.get_training_model().to(DEVICE)
print(mlp)

# initialize optimizer and loss function
opt = SGD(mlp.parameters(), lr=LR)
lossFunc = nn.CrossEntropyLoss()
```

**第 40 行**初始化我们的 MLP，并将其推送到我们用于训练的任何东西`DEVICE`(CPU 或 GPU)。

**第 44 行**定义了我们的 SGD 优化器，它接受两个参数:

1.  通过简单调用`mlp.parameters()`获得的 MLP 模型参数
2.  学习率

最后，我们初始化我们的分类交叉熵损失函数，这是使用 *> 2* 类执行分类时使用的标准损失方法。

**我们现在到达最重要的代码块，训练循环。**与 Keras/TensorFlow 允许你简单地调用`model.fit`来训练你的模型不同，PyTorch *要求*你手工实现你的训练循环*。*

手动实现训练循环有好处也有坏处。

一方面，你可以*对训练过程进行完全的控制*,这使得实现定制训练循环变得更加容易。

但另一方面，手工实现训练循环需要更多的代码，最糟糕的是，这更容易搬起石头砸自己的脚(这对初露头角的深度学习实践者来说尤其如此)。

**我的建议:你会想要多次阅读对以下代码块*****的解释，以便你理解训练循环的复杂性。您将** ***尤其是*** **想要密切关注我们如何将梯度归零，执行反向传播，然后更新模型参数—** ***如果不按照正确的顺序执行，将会导致错误的结果！****

 *让我们回顾一下我们的培训循环:

```py
# create a template to summarize current training progress
trainTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"

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

**第 48 行**初始化`trainTemplate`，这是一个字符串，允许我们方便地显示纪元编号，以及每一步的损失和精度。

然后我们在第 51 行**上循环我们想要的训练时期数。**紧接着在这个`for`循环里面我们:

1.  显示纪元编号，这对调试很有用(**行 53** )
2.  初始化我们的训练损失和精确度(**第 54 行和第 55 行**)
3.  初始化训练循环当前迭代中使用的数据点总数(**行 56** )
4.  将 PyTorch 模型置于训练模式(**第 57 行**)

**调用 PyTorch 模型的`train()`方法需要在反向传播过程中更新模型参数。**

在我们的下一个代码块中，您将看到我们将模型置于`eval()`模式，这样我们就可以在测试集上评估损失和准确性。如果我们忘记在下一个训练循环的顶部调用`train()`，那么我们的模型参数将不会被更新。

外部的`for`循环(**第 51 行**)在我们的历元数上循环。**第 60 行**然后开始一个内部`for`循环，遍历训练集中的每一批。**几乎你用 PyTorch 编写的每一个训练程序都将包含一个*外循环*(在一定数量的时期内)和一个*内循环*(在数据批次内)。**

在内部循环(即批处理循环)中，我们继续:

1.  将`batchX`和`batchY`数据移动到我们的 CPU 或 GPU(取决于我们的`DEVICE`)
2.  通过神经系统传递`batchX`数据，并对其进行预测
3.  使用我们的损失函数，通过将输出`predictions`与我们的地面实况类标签进行比较来计算我们的损失

现在我们已经有了`loss`，我们可以更新我们的模型参数了— **这是 PyTorch 训练程序中最重要的一步，也是大多数初学者经常出错的一步。**

为了更新我们模型的参数，我们*必须按照*指定的*的确切顺序*调用**行 69-71** :

1.  `opt.zero_grad()`:将模型前一批次/步骤累积的梯度归零
2.  `loss.backward()`:执行反向传播
3.  `opt.step()`:基于反向传播的结果更新我们的神经网络中的权重

**再次强调，你** ***必须*** **应用归零渐变，执行一个向后的过程，然后按照我已经指出的** ***的确切顺序*** **更新模型参数。**

正如我提到的，PyTorch 让你*对你的训练循环有很多*的控制……但它也让*很容易*搬起石头砸自己的脚。每一个深度学习实践者，无论是深度学习领域的新手还是经验丰富的专家，都曾经搞砸过这些步骤。

最常见的错误是忘记将梯度归零。如果您不将梯度归零，那么您将在多个批次和多个时期累积梯度。这将打乱你的反向传播，并导致错误的体重更新。

**说真的，别把这些步骤搞砸了。把它们写在便利贴上，如果需要的话，可以把它们放在你的显示器上。**

在我们将权重更新到我们的模型后，我们在第 75-77 行**上计算我们的训练损失、训练精度和检查的样本数量(即，批中数据点的数量)。**

然后我们应用我们的`trainTemplate`来显示我们的纪元编号、训练损失和训练精度。请注意我们如何将损失和准确度除以批次中的样本总数，以获得平均值。

**此时，我们已经在一个时期的所有数据点上训练了我们的 PyTorch 模型——现在我们需要在我们的测试集上评估它:**

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
```

类似于我们如何初始化我们的训练损失、训练精度和一批中的样本数量，我们在第 86-88 行对我们的测试集做同样的事情。这里，我们初始化变量来存储我们的测试损失、测试精度和测试集中的样本数。

我们还将我们的模型放入第 89 行**的`eval()`模型中。**我们*被要求*在我们需要计算测试或验证集的损失/精确度时，将我们的模型置于评估模式。

**但是`eval()`模式实际上*做什么呢？*** 您将评估模式视为关闭特定图层功能的开关，例如停止应用丢弃，或允许应用批量归一化的累积状态。

**其次，你通常将`eval()`与`torch.no_grad()`上下文结合使用，**意味着在评估模式下关闭*(**第 92 行**)。*

从那里，我们循环测试集中的所有批次(**第 94 行**)，类似于我们在前面的代码块中循环训练批次的方式。

对于每一批(**行 96** )，我们使用我们的模型进行预测，然后计算损耗(**行 99 和 100** )。

然后我们更新我们的`testLoss`、`testAcc`和数量`samples` ( **行 104-106** )。

最后，我们在我们的终端(**行 109-112** )上显示我们的纪元编号、测试损失和测试精度。

**总的来说，我们训练循环的评估部分是** ***与训练部分非常相似的*** **，没有微小的** ***但是非常显著的*** **变化:**

1.  我们使用`eval()`将模型置于评估模式
2.  我们使用`torch.no_grad()`上下文来确保不执行毕业计算

从那里，我们可以使用我们的模型进行预测，并计算测试集的准确性/损失。

### **PyTorch 培训结果**

我们现在准备用 PyTorch 训练我们的神经网络！

请务必访问本教程的 ***“下载”*** 部分来检索源代码。

要启动 PyTorch 培训流程，只需执行`train.py`脚本:

```py
$ python train.py
[INFO] training on cuda...
[INFO] preparing data...
Sequential(
  (hidden_layer_1): Linear(in_features=4, out_features=8, bias=True)
  (activation_1): ReLU()
  (output_layer): Linear(in_features=8, out_features=3, bias=True)
)
[INFO] training in epoch: 1...
epoch: 1 train loss: 0.971 train accuracy: 0.580
epoch: 1 test loss: 0.737 test accuracy: 0.827

[INFO] training in epoch: 2...
epoch: 2 train loss: 0.644 train accuracy: 0.861
epoch: 2 test loss: 0.590 test accuracy: 0.893

[INFO] training in epoch: 3...
epoch: 3 train loss: 0.511 train accuracy: 0.916
epoch: 3 test loss: 0.495 test accuracy: 0.900

[INFO] training in epoch: 4...
epoch: 4 train loss: 0.425 train accuracy: 0.941
epoch: 4 test loss: 0.423 test accuracy: 0.933

[INFO] training in epoch: 5...
epoch: 5 train loss: 0.359 train accuracy: 0.961
epoch: 5 test loss: 0.364 test accuracy: 0.953

[INFO] training in epoch: 6...
epoch: 6 train loss: 0.302 train accuracy: 0.975
epoch: 6 test loss: 0.310 test accuracy: 0.960

[INFO] training in epoch: 7...
epoch: 7 train loss: 0.252 train accuracy: 0.984
epoch: 7 test loss: 0.259 test accuracy: 0.967

[INFO] training in epoch: 8...
epoch: 8 train loss: 0.209 train accuracy: 0.987
epoch: 8 test loss: 0.215 test accuracy: 0.980

[INFO] training in epoch: 9...
epoch: 9 train loss: 0.174 train accuracy: 0.988
epoch: 9 test loss: 0.180 test accuracy: 0.980

[INFO] training in epoch: 10...
epoch: 10 train loss: 0.147 train accuracy: 0.991
epoch: 10 test loss: 0.153 test accuracy: 0.980
```

我们的前几行输出显示了简单的 4-8-3 MLP 架构，这意味着有四个输入到神经网络，一个隐藏层有八个节点*，最后一个输出层有三个*节点*。*

然后我们训练我们的网络总共十个纪元。在训练过程结束时，我们在训练集上获得了 **99.1%的准确率，在测试集上获得了 **98%的准确率。****

因此，我们可以得出结论，我们的神经网络在做出准确预测方面做得很好。

恭喜你用 PyTorch 训练了你的第一个神经网络！

### 如何在我自己的自定义数据集上训练 PyTorch 模型？

本教程向您展示了如何在 scikit-learn 的`make_blobs`函数生成的示例数据集上训练 PyTorch 神经网络。

虽然这是学习 PyTorch 基础知识的一个很好的例子，但是从真实场景的角度来看，它并不十分有趣。

下周，您将学习如何在手写字符数据集上训练 PyTorch 模型，它有许多实际应用，包括手写识别、OCR、*等等！*

敬请关注下周的教程，了解更多关于 PyTorch 和图像分类的知识。

## **总结**

在本教程中，您学习了如何使用 PyTorch 深度学习库训练您的第一个神经网络。这个例子很简单，但是展示了 PyTorch 框架的基本原理。

**我认为 PyTorch 库的深度学习实践者最大的错误是忘记和/或混淆了以下步骤:**

1.  将先前步骤的梯度归零(`opt.zero_grad()`)
2.  执行反向传播(`loss.backward()`)
3.  更新模型参数(`opt.step()`)

**不按照这个顺序执行这些步骤*****在使用 PyTorch 时肯定会搬起石头砸自己的脚，更糟糕的是，如果你混淆了这些步骤，PyTorch 不会报错，** ***所以你可能甚至不知道自己打中了自己！****

 *PyTorch 库*超级强大，*但是你需要习惯这样一个事实:用 PyTorch 训练神经网络就像卸下你自行车的训练轮——如果你混淆了重要的步骤，没有安全网可以抓住你(不像 Keras/TensorFlow，它允许你将整个训练过程封装到一个单独的`model.fit`调用中)。

这并不是说 Keras/TensorFlow 比 PyTorch“更好”,这只是你需要知道的两个深度学习库之间的差异。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！********