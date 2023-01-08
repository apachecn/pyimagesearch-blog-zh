# 在 PyTorch 训练 DCGAN

> 原文：<https://pyimagesearch.com/2021/10/25/training-a-dcgan-in-pytorch/>

在本教程中，我们将学习如何使用 PyTorch 生成图像来训练我们的第一个 DCGAN 模型。

本课是高级 PyTorch 技术 3 部分系列的第 1 部分:

1.  在 PyTorch 中训练 DCGAN(今天的教程)
2.  *在 PyTorch 中从头开始训练物体探测器*(下周课)
3.  *U-Net:在 PyTorch 中训练图像分割模型*(两周内)

到 2014 年，机器学习的世界已经取得了相当大的进步。几个新概念(如关注和 R-CNN)被引入。然而，机器学习的想法已经成型，因此它们都以某种方式被期待。

那是直到 [](https://arxiv.org/abs/1406.2661) Ian Goodfellow 等人[写了一篇论文](https://arxiv.org/abs/1406.2661)改变了机器学习的世界；生成性对抗网络。

深度卷积网络在图像领域取得成功后，没过多久，这个想法就和 GANs 融合在一起了。于是，DCGANs 诞生了。

**学习如何使用 PyTorch 编写的 DCGAN 生成图像，** ***继续阅读。***

## **在 PyTorch 中训练 DCGAN**

教程的结构:

*   gan 和 DCGANs 介绍
*   了解 DCGAN 架构
*   PyTorch 实现和演练
*   关于下一步尝试什么的建议

### **生成性对抗网络**

GANs 的突出因素是它们能够生成真实的图像，类似于您可能使用的数据分布。

GANs 的概念简单而巧妙。让我们用一个简单的例子来理解这个概念(**图 1** )。

你最近报了一个艺术班，那里的艺术老师极其苛刻和严格。当你交第一幅画时，美术老师惊呆了。他威胁要把你开除，直到你能创作出一幅壮观的杰作。

不用说，你心烦意乱。这项任务非常困难，因为你还是个新手。唯一对你有利的是，你讨厌的艺术老师说，这幅杰作不必是他的收藏的直接复制品，但它必须看起来像是属于它们的。

你急切地开始改进你的作品。在接下来的几天里，你提交了几份试用版，每一份都比上一次好，但还不足以让你通过这次测试。

与此同时，你的美术老师也开始成为展示给他的画作的更好的评判者。只需一瞥，他就能说出你试图复制的艺术家和艺术品的名字。最后，清算日到了，你提交你的最终作品(**图 2** )。

你画了一幅非常好的画，你的美术老师把它放在他的收藏中。他夸你，收你为全日制学生(但到了那个时候，你意识到你已经不需要他了)。

GAN 的工作原理也是如此。“你”是试图生成模拟给定输入数据集的图像的生成者。而“美术老师”是鉴别者，他的工作是判断您生成的图像是否可以与输入数据集进行分组。上述示例与 GAN 的唯一区别在于，发生器和鉴别器都是从零开始一起训练的。

这些网络相互提供反馈，当我们训练 GAN 模型时，两者都得到了改善，我们得到了更好的输出图像质量。

关于在 Keras 和 Tensorflow 中实现 GAN 模型的完整教程，我推荐 Adrian Rosebrock 的教程。

#### **什么是 DCGANs？**

[拉德福德等人(2016)](https://arxiv.org/abs/1511.06434) 发表了一篇关于深度卷积生成对抗网络(DCGANs)的论文。

当时的 DCGANs 向我们展示了如何在没有监督的情况下有效地使用 GANs 的卷积技术来创建与我们的数据集中的图像非常相似的图像。

在这篇文章中，我将解释 DCGANs 及其关键研究介绍，并带您在 MNIST 数据集上完成相同的 PyTorch 实现。顺便说一下，这真的很酷，因为论文的合著者之一是 Soumith Chintala，PyTorch 的核心创建者！

#### **DCGANs 架构**

让我们深入了解一下架构:

**图 3** 包含了 DCGAN 中使用的发生器的架构，如文中所示。

如**图 3** 所示，我们将一个随机噪声向量作为输入，并将一幅完整的图像作为输出。让我们看看**图 4** 中的鉴别器架构。

鉴别器充当正常的确定性模型，其工作是将输入图像分类为真或假。

该论文的作者创建了一个不同的部分，解释他们的方法和香草甘之间的差异。

*   普通 GAN 的池层由分数阶卷积(在发生器的情况下)和阶卷积(在鉴别器的情况下)代替。对于前者，我绝对推荐塞巴斯蒂安·拉什卡的这个[视频教程](https://www.youtube.com/watch?v=ilkSwsggSNM)。分数步长卷积是标准向上扩展的替代方案，允许模型学习自己的空间表示，而不是使用不可训练的确定性池图层。
*   与传统 GANs 的第二个最重要的区别是，它排除了完全连接的层，而支持更深层次的架构。
*   第三， [Ioffe 和 Szegedy (2015)](https://arxiv.org/pdf/1502.03167.pdf) 强调了批量标准化的重要性，以确保更深网络中梯度的正确流动。
*   最后，拉德福德等人解释了 ReLU 和 leaky ReLU 在其架构中的使用，引用了有界函数的成功，以帮助更快地了解训练分布。

让我们看看这些因素是如何影响结果的！

### **配置您的开发环境**

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
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
!tree .
.
├── dcgan_mnist.py
├── output
│   ├── epoch_0002.png
│   ├── epoch_0004.png
│   ├── epoch_0006.png
│   ├── epoch_0008.png
│   ├── epoch_0010.png
│   ├── epoch_0012.png
│   ├── epoch_0014.png
│   ├── epoch_0016.png
│   ├── epoch_0018.png
│   └── epoch_0020.png
├── output.gif
└── pyimagesearch
    ├── dcgan.py
    └── __init__.py
```

在`pyimagesearch`目录中，我们有两个文件:

*   `dcgan.py`:包含完整的 DCGAN 架构
*   `__init__.py`:将`pyimagesearch`转换成 python 目录

在父目录中，我们有`dcgan_mnist.py`脚本，它将训练 DCGAN 并从中得出推论。

除此之外，我们还有`output`目录，它包含 DCGAN 生成器生成的图像的按时间顺序的可视化。最后，我们有`output.gif`，它包含转换成 gif 的可视化效果。

### **在 PyTorch 中实现 DCGAN**

我们的第一个任务是跳转到`pyimagesearch`目录并打开`dcgan.py`脚本。该脚本将包含完整的 DCGAN 架构。

```py
# import the necessary packages
from torch.nn import ConvTranspose2d
from torch.nn import BatchNorm2d
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import LeakyReLU
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import Sigmoid
from torch import flatten
from torch import nn

class Generator(nn.Module):
	def __init__(self, inputDim=100, outputChannels=1):
		super(Generator, self).__init__()

		# first set of CONVT => RELU => BN
		self.ct1 = ConvTranspose2d(in_channels=inputDim,
			out_channels=128, kernel_size=4, stride=2, padding=0,
			bias=False)
		self.relu1 = ReLU()
		self.batchNorm1 = BatchNorm2d(128)

		# second set of CONVT => RELU => BN
		self.ct2 = ConvTranspose2d(in_channels=128, out_channels=64,
					kernel_size=3, stride=2, padding=1, bias=False)
		self.relu2 = ReLU()
		self.batchNorm2 = BatchNorm2d(64)

		# last set of CONVT => RELU => BN
		self.ct3 = ConvTranspose2d(in_channels=64, out_channels=32,
					kernel_size=4, stride=2, padding=1, bias=False)
		self.relu3 = ReLU()
		self.batchNorm3 = BatchNorm2d(32)

		# apply another upsample and transposed convolution, but
		# this time output the TANH activation
		self.ct4 = ConvTranspose2d(in_channels=32,
			out_channels=outputChannels, kernel_size=4, stride=2,
			padding=1, bias=False)
		self.tanh = Tanh()
```

在这里，我们创建了生成器类(**第 13 行**)。在我们的`__init__`构造函数中，我们有 2 个**重要的事情**要记住(**第 14 行**):

*   `inputDim`:通过发生器的噪声矢量的输入大小。
*   `outputChannels`:输出图像的通道数。因为我们使用的是 **MNIST** 数据集，所以图像将是灰度的。因此它只有一个频道。

由于 PyTorch 的卷积不需要高度和宽度规格，所以除了通道尺寸之外，我们不必指定输出尺寸。然而，由于我们使用的是 MNIST 数据，我们需要一个大小为`1×28×28`的输出。

记住，生成器*将随机噪声建模成图像*。记住这一点，我们的下一个任务是定义生成器的层。我们将使用`CONVT`(转置卷积)、`ReLU`(整流线性单元)、`BN`(批量归一化)(**第 18-34 行**)。最终的转置卷积之后是一个`tanh`激活函数，将我们的输出像素值绑定到`1`到`-1` ( **行 38-41** )。

```py
	def forward(self, x):
		# pass the input through our first set of CONVT => RELU => BN
		# layers
		x = self.ct1(x)
		x = self.relu1(x)
		x = self.batchNorm1(x)

		# pass the output from previous layer through our second
		# CONVT => RELU => BN layer set
		x = self.ct2(x)
		x = self.relu2(x)
		x = self.batchNorm2(x)

		# pass the output from previous layer through our last set
		# of CONVT => RELU => BN layers
		x = self.ct3(x)
		x = self.relu3(x)
		x = self.batchNorm3(x)

		# pass the output from previous layer through CONVT2D => TANH
		# layers to get our output
		x = self.ct4(x)
		output = self.tanh(x)

		# return the output
		return output
```

在生成器的`forward`通道中，我们使用了三次`CONVT` = >、`ReLU` = > `BN`图案，而最后的`CONVT`层之后是`tanh`层(**行 46-65** )。

```py
class Discriminator(nn.Module):
	def __init__(self, depth, alpha=0.2):
		super(Discriminator, self).__init__()

		# first set of CONV => RELU layers
		self.conv1 = Conv2d(in_channels=depth, out_channels=32,
				kernel_size=4, stride=2, padding=1)
		self.leakyRelu1 = LeakyReLU(alpha, inplace=True)

		# second set of CONV => RELU layers
		self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=4,
				stride=2, padding=1)
		self.leakyRelu2 = LeakyReLU(alpha, inplace=True)

		# first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=3136, out_features=512)
		self.leakyRelu3 = LeakyReLU(alpha, inplace=True)

		# sigmoid layer outputting a single value
		self.fc2 = Linear(in_features=512, out_features=1)
		self.sigmoid = Sigmoid()
```

请记住，当生成器将随机噪声建模到图像中时，鉴别器*获取图像并输出单个值*(确定它是否属于输入分布)。

在鉴别器的构造函数`__init__`中，只有两个参数:

*   `depth`:决定输入图像的通道数
*   `alpha`:给予架构中使用的泄漏 ReLU 函数的值

我们初始化一组卷积层、漏 ReLU 层、两个线性层，然后是最终的 sigmoid 层(**行 75-90** )。这篇论文的作者提到，泄漏 ReLU 允许一些低于零的值的特性有助于鉴别器的结果。当然，最后的 sigmoid 层是将奇异输出值映射到 0 或 1。

```py
	def forward(self, x):
		# pass the input through first set of CONV => RELU layers
		x = self.conv1(x)
		x = self.leakyRelu1(x)

		# pass the output from the previous layer through our second
		# set of CONV => RELU layers
		x = self.conv2(x)
		x = self.leakyRelu2(x)

		# flatten the output from the previous layer and pass it
		# through our first (and only) set of FC => RELU layers
		x = flatten(x, 1)
		x = self.fc1(x)
		x = self.leakyRelu3(x)

		# pass the output from the previous layer through our sigmoid
		# layer outputting a single value
		x = self.fc2(x)
		output = self.sigmoid(x)

		# return the output
		return output
```

在鉴别器的`forward`通道中，我们首先添加一个卷积层和一个漏 ReLU 层，并再次重复该模式(**第 94-100 行**)。接下来是一个`flatten`层、一个全连接层和另一个泄漏 ReLU 层(**线 104-106** )。在最终的 sigmoid 层之前，我们添加另一个完全连接的层(**行 110 和 111** )。

至此，我们的 DCGAN 架构就完成了。

### **训练 DCGAN**

`dcgan_mnist.py`不仅包含 DCGAN 的训练过程，还将充当我们的推理脚本。

```py
# USAGE
# python dcgan_mnist.py --output output

# import the necessary packages
from pyimagesearch.dcgan import Generator
from pyimagesearch.dcgan import Discriminator
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from sklearn.utils import shuffle
from imutils import build_montages
from torch.optim import Adam
from torch.nn import BCELoss
from torch import nn
import numpy as np
import argparse
import torch
import cv2
import os

# custom weights initialization called on generator and discriminator
def weights_init(model):
	# get the class name
	classname = model.__class__.__name__

	# check if the classname contains the word "conv"
	if classname.find("Conv") != -1:
		# intialize the weights from normal distribution
		nn.init.normal_(model.weight.data, 0.0, 0.02)

	# otherwise, check if the name contains the word "BatcnNorm"
	elif classname.find("BatchNorm") != -1:
		# intialize the weights from normal distribution and set the
		# bias to 0
		nn.init.normal_(model.weight.data, 1.0, 0.02)
		nn.init.constant_(model.bias.data, 0)
```

在第 23-37 行上，我们定义了一个名为`weights_init`的函数。这里，我们根据遇到的层初始化自定义权重。稍后，在推断步骤中，我们将看到这改进了我们的训练损失值。

对于卷积层，我们将`0.0`和`0.02`作为该函数中的平均值和标准差。对于批量标准化图层，我们将偏差设置为`0`，并将`1.0`和`0.02`作为平均值和标准偏差值。这是论文作者提出的，并认为最适合理想的训练结果。

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=20,
	help="# epochs to train for")
ap.add_argument("-b", "--batch-size", type=int, default=128,
	help="batch size for training")
args = vars(ap.parse_args())

# store the epochs and batch size in convenience variables
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]
```

在第 40-47 行上，我们构建了一个扩展的参数解析器来解析用户设置的参数并添加默认值。

我们继续将`epochs`和`batch_size`参数存储在适当命名的变量中(**第 50 行和第 51 行**)。

```py
# set the device we will be using
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define data transforms
dataTransforms = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5), (0.5))]
)

# load the MNIST dataset and stack the training and testing data
# points so we have additional training data
print("[INFO] loading MNIST dataset...")
trainData = MNIST(root="data", train=True, download=True,
	transform=dataTransforms)
testData = MNIST(root="data", train=False, download=True,
	transform=dataTransforms)
data = torch.utils.data.ConcatDataset((trainData, testData))

# initialize our dataloader
dataloader = DataLoader(data, shuffle=True,
	batch_size=BATCH_SIZE)
```

由于 GAN 训练确实涉及更多的复杂性，如果有合适的 GPU 可用，我们将默认设备设置为`cuda`(**Line 54**)。

为了预处理我们的数据集，我们简单地在第 57-60 行的**上定义了一个`torchvision.transforms`实例，在这里我们将数据集转换成张量并将其归一化。**

PyTorch 托管了许多流行的数据集供即时使用。它省去了在本地系统中下载数据集的麻烦。因此，我们从先前从`torchvision.datasets` ( **第 65-69 行**)导入的 **MNIST** 包中准备训练和测试数据集实例。MNIST 数据集是一个流行的数据集，包含总共 70，000 个手写数字。

在连接训练和测试数据集(**第 69 行**)之后，我们创建一个 PyTorch `DataLoader`实例来自动处理输入数据管道(**第 72 行和第 73 行**)。

```py
# calculate steps per epoch
stepsPerEpoch = len(dataloader.dataset) // BATCH_SIZE

# build the generator, initialize it's weights, and flash it to the
# current device
print("[INFO] building generator...")
gen = Generator(inputDim=100, outputChannels=1)
gen.apply(weights_init)
gen.to(DEVICE)

# build the discriminator, initialize it's weights, and flash it to
# the current device
print("[INFO] building discriminator...")
disc = Discriminator(depth=1)
disc.apply(weights_init)
disc.to(DEVICE)

# initialize optimizer for both generator and discriminator
genOpt = Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999),
	weight_decay=0.0002 / NUM_EPOCHS)
discOpt = Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999),
	weight_decay=0.0002 / NUM_EPOCHS)

# initialize BCELoss function
criterion = BCELoss()
```

因为我们已经将`BATCH_SIZE`值提供给了`DataLoader`实例，所以我们在**行 76** 上计算每个时期的步数。

在**第 81-83 行**，我们初始化生成器，应用自定义权重初始化，并将其加载到我们当前的设备中。正如在`dcgan.py`中提到的，我们在初始化期间传递适当的参数。

类似地，在**第 87-90 行**，我们初始化鉴别器，应用自定义权重，并将其加载到我们当前的设备上。唯一传递的参数是`depth`(即输入图像通道)。

我们选择`Adam`作为生成器和鉴别器的优化器(**第 88-96 行**)，传递

*   **模型参数:**标准程序，因为模型权重将在每个时期后更新。
*   **学习率:**控制模型自适应的超参数。
*   **β衰变变量:**初始衰变率。
*   **权重衰减值，通过历元数进行调整:**一种正则化方法，增加一个小的惩罚来帮助模型更好地泛化。

最后，我们损失函数的二元交叉熵损失(**第 99 行**)。

```py
# randomly generate some benchmark noise so we can consistently
# visualize how the generative modeling is learning
print("[INFO] starting training...")
benchmarkNoise = torch.randn(256, 100, 1, 1, device=DEVICE)

# define real and fake label values
realLabel = 1
fakeLabel = 0

# loop over the epochs
for epoch in range(NUM_EPOCHS):
	# show epoch information and compute the number of batches per
	# epoch
	print("[INFO] starting epoch {} of {}...".format(epoch + 1,
		NUM_EPOCHS))

	# initialize current epoch loss for generator and discriminator
	epochLossG = 0
	epochLossD = 0
```

在**线 104** 上，我们使用`torch.randn`给发电机馈电，并在可视化发电机训练期间保持一致性。

对于鉴别器，真标签和假标签值被初始化(**行 107 和 108** )。

随着必需品的离开，我们开始在**行 111** 上的历元上循环，并初始化逐历元发生器和鉴别器损耗(**行 118 和 119** )。

```py
	for x in dataloader:
		# zero out the discriminator gradients
		disc.zero_grad()

		# grab the images and send them to the device
		images = x[0]
		images = images.to(DEVICE)

		# get the batch size and create a labels tensor
		bs =  images.size(0)
		labels = torch.full((bs,), realLabel, dtype=torch.float,
			device=DEVICE)

		# forward pass through discriminator
		output = disc(images).view(-1)

		# calculate the loss on all-real batch
		errorReal = criterion(output, labels)

		# calculate gradients by performing a backward pass
		errorReal.backward()
```

启动前，使用`zero_grad` ( **线 123** )冲洗电流梯度。

从`DataLoader`实例(**第 121 行**)中获取数据，我们首先倾向于鉴别器。我们将并发批次的所有图像发送至设备(**第 126 和 127 行**)。由于所有图像都来自数据集，因此它们被赋予了`realLabel` ( **第 131 和 132 行**)。

在**行 135** 上，使用图像执行鉴别器的一次正向通过，并计算误差(**行 138** )。

`backward`函数根据损失计算梯度(**行 141** )。

```py
		# randomly generate noise for the generator to predict on
		noise = torch.randn(bs, 100, 1, 1, device=DEVICE)

		# generate a fake image batch using the generator
		fake = gen(noise)
		labels.fill_(fakeLabel)

		# perform a forward pass through discriminator using fake
		# batch data
		output = disc(fake.detach()).view(-1)
		errorFake = criterion(output, labels)

		# calculate gradients by performing a backward pass
		errorFake.backward()

		# compute the error for discriminator and update it
		errorD = errorReal + errorFake
		discOpt.step()
```

现在，我们继续讨论发电机的输入。在**线 144** 上，基于发电机输入大小的随机噪声被产生并馈入发电机(**线 147** )。

由于生成器生成的所有图像都是假的，我们用`fakeLabel`值(**第 148 行**)替换标签张量的值。

在**行 152 和 153** 上，伪图像被馈送到鉴别器，并且计算伪预测的误差。

假图像产生的误差随后被送入`backward`函数进行梯度计算(**行 156** )。然后根据两组图像产生的总损失更新鉴别器(**第 159 和 160 行**)。

```py
		# set all generator gradients to zero
		gen.zero_grad()

		# update the labels as fake labels are real for the generator
		# and perform a forward pass  of fake data batch through the
		# discriminator
		labels.fill_(realLabel)
		output = disc(fake).view(-1)

		# calculate generator's loss based on output from
		# discriminator and calculate gradients for generator
		errorG = criterion(output, labels)
		errorG.backward()

		# update the generator
		genOpt.step()

		# add the current iteration loss of discriminator and
		# generator
		epochLossD += errorD
		epochLossG += errorG
```

继续发电机的训练，首先使用`zero_grad` ( **线 163** )冲洗梯度。

现在在**第 168-173 行**，我们做了一件非常有趣的事情:由于生成器必须尝试生成尽可能真实的图像，我们用`realLabel`值填充实际标签，并根据鉴别器对生成器生成的图像给出的预测来计算损失。生成器必须让鉴别器猜测其生成的图像是真实的。因此，这一步非常重要。

接下来，我们计算梯度(**行 174** )并更新生成器的权重(**行 177** )。

最后，我们更新发生器和鉴别器的总损耗值(**行 181 和 182** )。

```py
	# display training information to disk
	print("[INFO] Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(
		epochLossG / stepsPerEpoch, epochLossD / stepsPerEpoch))

	# check to see if we should visualize the output of the
	# generator model on our benchmark data
	if (epoch + 1) % 2 == 0:
		# set the generator in evaluation phase, make predictions on
		# the benchmark noise, scale it back to the range [0, 255],
		# and generate the montage
		gen.eval()
		images = gen(benchmarkNoise)
		images = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
		images = ((images * 127.5) + 127.5).astype("uint8")
		images = np.repeat(images, 3, axis=-1)
		vis = build_montages(images, (28, 28), (16, 16))[0]

		# build the output path and write the visualization to disk
		p = os.path.join(args["output"], "epoch_{}.png".format(
			str(epoch + 1).zfill(4)))
		cv2.imwrite(p, vis)

		# set the generator to training mode
		gen.train()
```

这段代码也将作为我们的训练可视化和推理片段。

对于某个纪元值，我们将生成器设置为评估模式(**行 190-194** )。

使用之前初始化的`benchmarkNoise`，我们让生成器产生图像(**行 195 和 196** )。然后首先对图像进行高度整形，并按比例放大到其原始像素值(**第 196 和 197 行**)。

使用一个名为`build_montages`的漂亮的`imutils`函数，我们显示每次调用期间生成的批处理图像(**第 198 和 199 行**)。`build_montages`函数接受以下参数:

*   形象
*   正在显示的每个图像的大小
*   将显示可视化效果的网格的大小

在**行 202-207** 上，我们定义了一个输出路径来保存可视化图像并将生成器设置回训练模式。

至此，我们完成了 DCGAN 培训！

### **DCGAN 训练结果和可视化**

让我们看看 DCGAN 在损耗方面的划时代表现。

```py
$ python dcgan_mnist.py --output output
[INFO] loading MNIST dataset...
[INFO] building generator...
[INFO] building discriminator...
[INFO] starting training...
[INFO] starting epoch 1 of 20...
[INFO] Generator Loss: 4.6538, Discriminator Loss: 0.3727
[INFO] starting epoch 2 of 20...
[INFO] Generator Loss: 1.5286, Discriminator Loss: 0.9514
[INFO] starting epoch 3 of 20...
[INFO] Generator Loss: 1.1312, Discriminator Loss: 1.1048
...
[INFO] Generator Loss: 1.0039, Discriminator Loss: 1.1748
[INFO] starting epoch 17 of 20...
[INFO] Generator Loss: 1.0216, Discriminator Loss: 1.1667
[INFO] starting epoch 18 of 20...
[INFO] Generator Loss: 1.0423, Discriminator Loss: 1.1521
[INFO] starting epoch 19 of 20...
[INFO] Generator Loss: 1.0604, Discriminator Loss: 1.1353
[INFO] starting epoch 20 of 20...
[INFO] Generator Loss: 1.0835, Discriminator Loss: 1.1242
```

现在，在没有初始化自定义权重的情况下重新进行整个训练过程*后，我们注意到损失值相对较高。因此，我们可以得出结论*自定义重量初始化确实有助于改善训练过程*。*

让我们看看我们的生成器在图 6-9 中的一些改进图像。

在**图 6** 中，我们可以看到，由于生成器刚刚开始训练，生成的图像几乎是胡言乱语。在**图 7** 中，我们可以看到随着图像慢慢成形，生成的图像略有改善。

在**图 8 和 9** 中，我们看到完整的图像正在形成，看起来就像是从 MNIST 数据集中提取出来的，这意味着我们的生成器学习得很好，最终生成了一些非常好的图像！

## **总结**

GANs 为机器学习打开了一扇全新的大门。我们不断看到 GANs 产生的许多新概念和许多旧概念以 GANs 为基础被复制。这个简单而巧妙的概念足够强大，在各自的领域胜过大多数其他算法。

在训练图中，我们已经看到，到 20 世纪本身，我们的 DCGAN 变得足够强大，可以产生完整和可区分的图像。但是我鼓励您将您在这里学到的编码技巧用于各种其他任务和数据集，并亲自体验 GANs 的魔力。

通过本教程，我试图使用 MNIST 数据集解释 gan 和 DCGANs 的基本本质。我希望这篇教程能帮助你认识到甘是多么的伟大！

### **引用信息**

Chakraborty，d .“在 PyTorch 训练一名 DCGAN”， *PyImageSearch* ，2021 年，[https://PyImageSearch . com/2021/10/25/Training-a-DCGAN-in-py torch/](https://pyimagesearch.com/2021/10/25/training-a-dcgan-in-pytorch/)

```py
@article{Chakraborty_2021_Training_DCGAN,
   author = {Devjyoti Chakraborty},
   title = {Training a {DCGAN} in {PyTorch}},
   journal = {PyImageSearch},
   year = {2021},
   note = {https://pyimagesearch.com/2021/10/25/training-a-dcgan-in-pytorch/},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****