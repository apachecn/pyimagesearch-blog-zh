# 带 Keras 和 TensorFlow 的 GANs

> 原文：<https://pyimagesearch.com/2020/11/16/gans-with-keras-and-tensorflow/>

在本教程中，你将学习如何使用 Keras 和 TensorFlow 实现生成式对抗网络(GANs)。

Goodfellow 等人在其 2014 年的论文 *[中首次引入了生成对抗网络](https://arxiv.org/abs/1406.2661)。*这些网络可用于生成合成(即伪造)图像，这些图像在感知上与真实可信的原始图像几乎相同。

为了生成合成图像，我们在训练期间使用了两个神经网络:

1.  一个**发生器**,它接受随机产生的噪声的输入向量，并产生一个输出“模仿”图像，该图像看起来与真实图像相似，如果不是完全相同的话
2.  一个**鉴别器**或**对手**，试图确定给定图像是“真实的”还是“伪造的”

通过同时训练这些网络，一个给另一个反馈，我们可以学习生成合成图像。

在本教程中，我们将实现拉德福德等人论文的一个变体， *[使用深度卷积生成对抗网络](https://arxiv.org/abs/1511.06434)* 的无监督表示学习——或者更简单地说， **DCGANs。**

我们会发现，训练 GANs 可能是一项众所周知的艰巨任务，因此我们将实施拉德福德等人和弗朗索瓦·乔莱(Keras 的创始人和谷歌深度学习科学家)推荐的一些最佳实践。

在本教程结束时，您将拥有一个功能完整的 GAN 实现。

**要学习如何用 Keras 和 TensorFlow 实现生成式对抗网络(GANs)，*继续阅读。***

## **具有 Keras 和 TensorFlow 的 GANs】**

***注:**本教程是我的书[用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)中的一章。如果你喜欢这篇文章，并想了解更多关于深度学习在计算机视觉中的应用，请务必阅读我的书* — *我毫不怀疑它将带你从深度学习初学者一路成为专家。*

在本教程的第一部分，我们将讨论什么是生成对立网络，包括它们与你以前见过的用于分类和回归的更“普通”的网络架构有何不同。

在此基础上，我们将讨论一般的 GAN 培训流程，包括您在培训自己的 GAN 时应该遵循的一些指南和最佳实践。

接下来，我们将回顾项目的目录结构，然后使用 Keras 和 TensorFlow 实现我们的 GAN 架构。

一旦我们的 GAN 被实现，我们将在时尚 MNIST 数据集上训练它，从而允许我们生成假的/合成的时尚服装图像。

最后，我们将讨论我们的结果来结束这个关于生成性敌对网络的教程。

### 什么是生成敌对网络？

对 GANs 的典型解释通常涉及两个人合谋伪造一套文件、复制一件艺术品或印刷假币的某种变体——假币印刷机是我个人最喜欢的，也是 Chollet 在工作中使用的。

在这个例子中，我们有两个人:

1.  杰克，假冒的打印机(*发生器*
2.  杰森(Jason)，美国财政部(负责美国印钞)的雇员，专门检测假币(即*鉴别器*)

杰克和杰森是青梅竹马，都是在波士顿的贫民区长大的，没有多少钱。经过艰苦的努力，杰森获得了大学奖学金，而杰克没有，随着时间的推移，他开始转向非法企业赚钱(在这种情况下，制造假币)。

杰克知道他不太擅长制造假币，但他觉得经过适当的训练，他可以复制出流通中还过得去的钞票。

一天，在感恩节假期里，杰森在当地一家酒吧喝了几杯后，无意中向杰克透露了他对自己的工作不满意。他报酬过低。他的老板既讨厌又恶毒，经常在其他员工面前大喊大叫，让杰森尴尬。杰森甚至想辞职。

杰克发现了一个机会，可以利用杰森在美国财政部的权限来设计一个复杂的伪造印刷计划。他们的阴谋是这样运作的:

1.  假钞印刷商杰克会印刷假钞，然后把假钞和真钱混在一起，然后给专家杰森看。
2.  杰森会对钞票进行分类，将每张钞票分为“假钞”或“真钞”，一路上向杰克提供反馈，告诉他如何改进伪造的印刷技术。

起初，杰克在印制伪钞方面做得很差。但随着时间的推移，在杰森的指导下，杰克最终提高到杰森不再能够分辨钞票之间的差异。在这个过程结束时，杰克和杰森都有了一叠叠可以骗过大多数人的假币。

### **甘的一般训练程序**

我们已经从类比的角度讨论了什么是甘，但是训练他们的实际*程序*是什么？大多数 GANs 都是通过六个步骤来训练的。

首先(步骤 1)，我们随机生成一个矢量(即噪声)。我们让这个噪声通过我们的生成器，它生成一个实际的图像(步骤 2)。然后，我们从我们的训练集中采样真实图像，并将它们与我们的合成图像混合(步骤 3)。

下一步(步骤 4)是使用这个混合集训练我们的鉴别器。鉴别器的目标是正确地将每个图像标记为“真”或“假”

接下来，我们将再次生成随机噪声，但这一次我们将有目的地将每个噪声向量标记为“真实图像”(步骤 5)。然后，我们将使用噪声向量和“真实图像”标签来训练 GAN，即使它们不是实际的真实图像(步骤 6)。

这一过程之所以有效是由于以下原因:

1.  我们在这个阶段冻结了鉴别器的权重，这意味着当我们更新生成器的权重时，鉴别器没有学习。
2.  我们试图“愚弄”鉴别者，使其无法判断哪些图像是真实的，哪些是合成的。来自鉴别器的反馈将允许生成器学习如何产生更真实的图像。

如果您对这个过程感到困惑，我会继续阅读本教程后面的实现——看到用 Python 实现的 GAN，然后进行解释，会更容易理解这个过程。

### **培训 GANs 时的指南和最佳实践**

众所周知，由于**不断演变的亏损格局**，gan 们很难训练。在我们算法的每次迭代中，我们:

1.  生成随机图像，然后训练鉴别器来正确区分这两者
2.  生成额外的合成图像，但这一次故意试图欺骗鉴别者
3.  基于鉴别器的反馈更新生成器的权重，从而允许我们生成更真实的图像

从这个过程中，你会注意到我们需要观察两个损耗:一个是鉴频器的损耗，另一个是发电机的损耗。由于发电机的损耗状况可以根据鉴频器的反馈而改变，*我们最终得到了一个动态系统。*

在训练 gan 时，我们的目标不是寻求最小损失值，而是在两者之间找到某种平衡(Chollet 2017)。

这种寻找平衡的概念在理论上可能是有意义的，但是一旦你尝试实现和训练你自己的 GANs，你会发现这是一个不简单的过程。

**在他们的[论文](https://arxiv.org/abs/1511.06434)中，拉德福德等人为更稳定的 GANs 推荐了以下架构指南:**

*   用交错卷积替换任何池层(参见[本教程](https://pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/)了解更多关于卷积和交错卷积的信息)。
*   在生成器和鉴别器中使用批处理规范化。
*   移除深层网络中的全连接层。
*   在发生器中使用 ReLU，除了最后一层，它将使用 *tanh。*
*   在鉴别器中使用泄漏 ReLU。

在他的书中，Francois Chollet 提供了更多关于训练 GANs 的建议:

1.  从*正态分布*(即高斯分布)而非*均匀分布中抽取随机向量。*
2.  将 dropout 添加到鉴别器中。
3.  训练鉴别器时，将噪声添加到类别标签中。
4.  要减少输出图像中的棋盘格像素伪像，在发生器和鉴别器中使用卷积或转置卷积时，请使用可被步幅整除的核大小。
5.  如果你的对抗性损失急剧上升，而你的鉴别器损失下降到零，尝试降低鉴别器的学习率，增加鉴别器的漏失。

请记住，这些都只是在许多情况下有效的*试探法*——我们将使用*拉德福德等人和乔莱建议的一些*技术，而不是*所有的*。

很可能，甚至很有可能，这里列出的技巧对你的 GANs 不起作用。现在花时间设定你的期望，当调整你的 GANs 的超参数时，与更基本的分类或回归任务相比，你可能会运行*数量级更多的实验*。

### **配置您的开发环境，使用 Keras 和 TensorFlow 训练 GANs】**

我们将使用 Keras 和 TensorFlow 来实现和培训我们的 gan。

我建议您按照这两个指南中的任何一个在您的系统上安装 TensorFlow 和 Keras:

*   *[如何在 Ubuntu 上安装 tensor flow 2.0](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)*
*   *[如何在 macOS 上安装 tensor flow 2.0](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)*

这两个教程都将帮助您在一个方便的 Python 虚拟环境中，用这篇博文所需的所有软件来配置您的系统。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在你的 Windows、macOS 或 Linux 系统上运行代码*了吗？***

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

现在我们已经了解了生成性对抗网络的基础，让我们回顾一下这个项目的目录结构。

确保您使用本教程的 ***【下载】*** 部分将源代码下载到我们的 GAN 项目:

```py
$ tree . --dirsfirst
.
├── output
│   ├── epoch_0001_output.png
│   ├── epoch_0001_step_00000.png
│   ├── epoch_0001_step_00025.png
...
│   ├── epoch_0050_step_00300.png
│   ├── epoch_0050_step_00400.png
│   └── epoch_0050_step_00500.png
├── pyimagesearch
│   ├── __init__.py
│   └── dcgan.py
└── dcgan_fashion_mnist.py

3 directories, 516 files
```

### **用 Keras 和 TensorFlow 实现我们的“生成器”**

现在我们已经回顾了我们的项目目录结构，让我们开始使用 Keras 和 TensorFlow 实现我们的生成式对抗网络。

在我们的项目目录结构中打开`dcgan.py`文件，让我们开始吧:

```py
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
```

**第 2-10 行**导入我们需要的 Python 包。所有这些类对你来说应该看起来相当熟悉，尤其是如果你已经读过我的 [*Keras 和 TensorFlow* 教程](https://pyimagesearch.com/category/keras-and-tensorflow/)或者我的书 [*用 Python 进行计算机视觉的深度学习。*](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)

唯一的例外可能是`Conv2DTranspose`类。转置卷积层，有时被称为*分数步长卷积*或(不正确的)*反卷积*，当我们需要在常规卷积的*相反方向*进行变换时使用。

我们的 GAN 的生成器将接受一个 *N* 维输入向量(即一个*列表*的数字，但是一个*体积*像一个图像)，然后将 *N* 维向量转换成一个输出图像。

这个过程意味着我们需要*重塑*，然后*在这个向量通过网络时将其放大*成一个体积——为了完成这种重塑和放大，我们需要转置卷积。

因此，我们可以将转置卷积视为实现以下目的的方法:

1.  接受来自网络中前一层的输入量
2.  产生比输入量大的输出量
3.  维护输入和输出之间的连接模式

本质上，我们的转置卷积层将重建我们的目标空间分辨率，并执行正常的卷积操作，利用花哨的零填充技术来确保满足我们的输出空间维度。

要了解转置卷积的更多信息，请查看 Theano 文档中的 [*卷积运算教程*](http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html)以及 Paul-Louis prve 的 *[深度学习中不同类型卷积的介绍](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)* 。

现在让我们开始实现我们的`DCGAN`类:

```py
class DCGAN:
	@staticmethod
	def build_generator(dim, depth, channels=1, inputDim=100,
		outputDim=512):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (dim, dim, depth)
		chanDim = -1
```

当我们在下一个代码块中定义网络主体时，这些参数的用法将变得更加清楚。

下面我们可以找到我们的发电机网络的主体:

```py
		# first set of FC => RELU => BN layers
		model.add(Dense(input_dim=inputDim, units=outputDim))
		model.add(Activation("relu"))
		model.add(BatchNormalization())

		# second set of FC => RELU => BN layers, this time preparing
		# the number of FC nodes to be reshaped into a volume
		model.add(Dense(dim * dim * depth))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
```

**第 23-25 行**定义了我们的第一组`FC => RELU => BN`层——应用批量标准化来稳定 GAN 训练是拉德福德等人的指导方针(参见上面的*“训练 GAN 时的指导方针和最佳实践”*部分)。

注意我们的`FC`层将如何拥有一个输入维度`inputDim`(随机生成的输入向量)，然后输出维度`outputDim`。通常`outputDim`会比`inputDim`大*。*

 ***第 29-31 行**应用第二组`FC => RELU => BN`层，但是这次我们准备`FC`层中的节点数等于`inputShape` ( **第 29 行**)中的单元数。尽管我们仍在使用扁平化的表示，但我们需要确保这个`FC`层的输出可以被整形为我们的目标卷 sze(即`inputShape`)。

实际的整形发生在下一个代码块中:

```py
		# reshape the output of the previous layer set, upsample +
		# apply a transposed convolution, RELU, and BN
		model.add(Reshape(inputShape))
		model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2),
			padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
```

在提供`inputShape`的同时调用`Reshape`允许我们从**行 29** 上的全连接层创建一个 3D 体积。**同样，这种整形仅在`FC`层中输出节点的数量与目标`inputShape`匹配的情况下才有可能。**

在训练你自己的 GANs 时，我们现在达成了一个重要的指导方针:

1.  为了*增加*空间分辨率，使用步长为 *> 1 的转置卷积。*
2.  要创建更深的 GAN *而不增加空间分辨率，您可以使用标准卷积或转置卷积(但保持跨距等于 1)。*

在这里，我们的转置卷积层正在学习`32`滤波器，每个滤波器都是 *5×5* ，同时应用一个 *2×2* 步幅——由于我们的步幅是 *> 1* ，我们可以增加我们的空间分辨率。

让我们应用另一个转置卷积:

```py
		# apply another upsample and transposed convolution, but
		# this time output the TANH activation
		model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2),
			padding="same"))
		model.add(Activation("tanh"))

		# return the generator model
		return model
```

**第 43 行和第 44 行**应用另一个转置卷积，再次增加空间分辨率，但是注意确保学习的滤波器数量等于`channels`的目标数量(灰度的`1`和 RGB 的`3`)。

然后，我们按照拉德福德等人的建议应用一个 *tanh* 激活函数。然后，模型返回到第**行 48 上的调用函数。**

### **了解我们 GAN 中的“发电机”**

假设`dim=7`、`depth=64`、`channels=1`、`inputDim=100`和`outputDim=512`(我们将在本教程稍后对我们的 GAN 进行时尚培训时使用)，我已经包括了以下模型摘要:

```py
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 512)               51712     
_________________________________________________________________
activation (Activation)      (None, 512)               0         
_________________________________________________________________
batch_normalization (BatchNo (None, 512)               2048      
_________________________________________________________________
dense_1 (Dense)              (None, 3136)              1608768   
_________________________________________________________________
activation_1 (Activation)    (None, 3136)              0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 3136)              12544     
_________________________________________________________________
reshape (Reshape)            (None, 7, 7, 64)          0         
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 14, 14, 32)        51232     
_________________________________________________________________
activation_2 (Activation)    (None, 14, 14, 32)        0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 14, 32)        128       
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 28, 28, 1)         801       
_________________________________________________________________
activation_3 (Activation)    (None, 28, 28, 1)         0        
================================================================= 
```

我们来分析一下这是怎么回事。

应用步长为 *2×2* 的转置卷积将我们的空间维度从 *7×7* 增加到 *14×14。*

第二个转置卷积(同样，步长为 *2×2* )使用单个通道将我们的空间维度分辨率从 *14×14* 增加到 *28×18* ，这是我们在时尚 MNIST 数据集中输入图像的*精确*维度。

**在实现自己的 GANs 时，确保输出体积的空间尺寸与输入图像的空间尺寸相匹配。**使用转置卷积增加生成器中体积的空间维度。我也推荐经常使用`model.summary()`来帮助你调试空间维度。

### **用 Keras 和 TensorFlow 实现我们的“鉴别器”**

鉴别器模型*实质上*更简单，类似于你可能在我的书或 PyImageSearch 博客的其他地方读过的基本 CNN 分类架构[。](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)

请记住，虽然生成器旨在*创建*合成图像，但鉴别器用于*对*进行分类，以确定任何给定的输入图像是*真实的*还是*虚假的。*

继续我们在`dcgan.py`中对`DCGAN`类的实现，现在让我们看看鉴别器:

```py
	@staticmethod
	def build_discriminator(width, height, depth, alpha=0.2):
		# initialize the model along with the input shape to be
		# "channels last"
		model = Sequential()
		inputShape = (height, width, depth)

		# first set of CONV => RELU layers
		model.add(Conv2D(32, (5, 5), padding="same", strides=(2, 2),
			input_shape=inputShape))
		model.add(LeakyReLU(alpha=alpha))

		# second set of CONV => RELU layers
		model.add(Conv2D(64, (5, 5), padding="same", strides=(2, 2)))
		model.add(LeakyReLU(alpha=alpha))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=alpha))

		# sigmoid layer outputting a single value
		model.add(Dense(1))
		model.add(Activation("sigmoid"))

		# return the discriminator model
		return model
```

正如我们所见，这个网络简单明了。我们先学习 32 个， *5×5* 滤镜，接着是第二个`CONV`层，这一层总共学习 64 个， *5×5* 滤镜。我们这里只有一个单独的`FC`层，这一层有`512`节点。

所有激活层都利用漏 ReLU 激活来稳定训练，*除了最后的激活函数为 sigmoid 的*。我们在这里使用一个 sigmoid 来捕捉输入图像是*真实的*还是*合成的概率。*

### **实施我们的 GAN 培训脚本**

现在，我们已经实现了我们的 DCGAN 架构，让我们在时尚 MNIST 数据集上训练它，以生成虚假的服装项目。训练过程结束时，我们将无法从*合成*图像中识别出*真实*图像。

在我们的项目目录结构中打开`dcgan_fashion_mnist.py`文件，让我们开始工作:

```py
# import the necessary packages
from pyimagesearch.dcgan import DCGAN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from sklearn.utils import shuffle
from imutils import build_montages
import numpy as np
import argparse
import cv2
import os
```

我们从导入所需的 Python 包开始。

让我们开始解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-e", "--epochs", type=int, default=50,
	help="# epochs to train for")
ap.add_argument("-b", "--batch-size", type=int, default=128,
	help="batch size for training")
args = vars(ap.parse_args())
```

对于这个脚本，我们只需要一个命令行参数`--output`，它是输出目录的路径，我们将在该目录中存储生成的图像的剪辑(从而允许我们可视化 GAN 训练过程)。

我们还可以(可选地)提供`--epochs`，训练的总时期数，以及`--batch-size`，用于在训练时控制批量大小。

现在让我们来关注一些重要的初始化:

```py
# store the epochs and batch size in convenience variables, then
# initialize our learning rate
NUM_EPOCHS = args["epochs"]
BATCH_SIZE = args["batch_size"]
INIT_LR = 2e-4
```

我们在第 26 和 27 行的**方便变量中存储了时期数和批量大小。**

我们还在第 28 行**上初始化我们的初始学习速率(`INIT_LR`)。**这个值是*通过大量实验和反复试验根据经验调整的*。如果您选择将这个 GAN 实现应用到您自己的数据集，您可能需要调整这个学习率。

我们现在可以从磁盘加载时尚 MNIST 数据集:

```py
# load the Fashion MNIST dataset and stack the training and testing
# data points so we have additional training data
print("[INFO] loading MNIST dataset...")
((trainX, _), (testX, _)) = fashion_mnist.load_data()
trainImages = np.concatenate([trainX, testX])

# add in an extra dimension for the channel and scale the images
# into the range [-1, 1] (which is the range of the tanh
# function)
trainImages = np.expand_dims(trainImages, axis=-1)
trainImages = (trainImages.astype("float") - 127.5) / 127.5
```

**第 33 行**从磁盘加载时尚 MNIST 数据集。这里我们忽略类标签，因为我们不需要它们——我们只对实际的像素数据感兴趣。

此外，不存在 gan 的“测试集”概念。训练 GAN 时，我们的目标不是最小损失或高精度。相反，我们寻求生成器和鉴别器之间的平衡。

为了帮助我们获得这种平衡，我们*结合*训练和测试图像(**第 34 行**)来为我们提供额外的训练数据。

**第 39 行和第 40 行**通过将像素强度缩放到范围*【0，1】*， *tanh* 激活函数的输出范围，来准备我们的训练数据。

现在让我们初始化我们的生成器和鉴别器:

```py
# build the generator
print("[INFO] building generator...")
gen = DCGAN.build_generator(7, 64, channels=1)

# build the discriminator
print("[INFO] building discriminator...")
disc = DCGAN.build_discriminator(28, 28, 1)
discOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS)
disc.compile(loss="binary_crossentropy", optimizer=discOpt)
```

**第 44 行**初始化发生器，该发生器将把输入随机向量转换成形状为 *7x7x64-* 通道图的体积。

**第 48-50 行**构建鉴别器，然后使用带有二进制交叉熵损失的 Adam 优化器编译它。

请记住，我们在这里使用的是*二元*交叉熵，因为我们的鉴别器有一个 sigmoid 激活函数，它将返回一个概率，指示输入图像是真实的还是伪造的。由于只有两个“类别标签”(真实与合成)，我们使用二进制交叉熵。

Adam 优化器的学习率和 beta 值是通过实验调整的。我发现 Adam 优化器的较低学习速率和 beta 值改善了时尚 MNIST 数据集上的 GAN 训练。应用学习率衰减也有助于稳定训练。

给定生成器和鉴别器，我们可以构建 GAN:

```py
# build the adversarial model by first setting the discriminator to
# *not* be trainable, then combine the generator and discriminator
# together
print("[INFO] building GAN...")
disc.trainable = False
ganInput = Input(shape=(100,))
ganOutput = disc(gen(ganInput))
gan = Model(ganInput, ganOutput)

# compile the GAN
ganOpt = Adam(lr=INIT_LR, beta_1=0.5, decay=INIT_LR / NUM_EPOCHS)
gan.compile(loss="binary_crossentropy", optimizer=discOpt)
```

实际的 GAN 由发生器和鉴别器组成；然而，我们首先需要*冻结*鉴别器权重(**第 56 行**)，然后我们才组合模型以形成我们的生成性对抗网络(**第 57-59 行**)。

这里我们可以看到,`gan`的输入将采用一个 100-d 的随机向量。该值将首先通过生成器，其输出将进入鉴别器——我们称之为“模型合成”，类似于我们在代数课上学到的“函数合成”。

**鉴别器权重在这一点上被冻结，因此来自鉴别器的反馈将使生成器能够*学习*如何生成更好的合成图像。**

**第 62 行和第 63 行**编译了`gan`。我再次使用 Adam 优化器，其超参数与鉴别器的优化器相同——这个过程适用于这些实验，但是您可能需要在自己的数据集和模型上调整这些值。

此外，我经常发现将 GAN 的学习速率设置为鉴频器的一半通常是一个好的起点。

在整个训练过程中，我们希望看到我们的 GAN 如何从随机噪声中进化出合成图像。为了完成这项任务，我们需要生成一些基准随机噪声来可视化训练过程:

```py
# randomly generate some benchmark noise so we can consistently
# visualize how the generative modeling is learning
print("[INFO] starting training...")
benchmarkNoise = np.random.uniform(-1, 1, size=(256, 100))

# loop over the epochs
for epoch in range(0, NUM_EPOCHS):
	# show epoch information and compute the number of batches per
	# epoch
	print("[INFO] starting epoch {} of {}...".format(epoch + 1,
		NUM_EPOCHS))
	batchesPerEpoch = int(trainImages.shape[0] / BATCH_SIZE)

	# loop over the batches
	for i in range(0, batchesPerEpoch):
		# initialize an (empty) output path
		p = None

		# select the next batch of images, then randomly generate
		# noise for the generator to predict on
		imageBatch = trainImages[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
		noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
```

**第 68 行**生成我们的`benchmarkNoise`。请注意，`benchmarkNoise`是从范围 *[-1，1]* 中的均匀分布生成的，这个范围与我们的 *tanh* 激活函数相同。**第 68 行**表示我们将生成 256 个合成图像，其中每个输入都以一个 100 维向量开始。

从第 71 行**开始，我们循环我们想要的历元数。**第 76 行**通过将训练图像的数量除以提供的批次大小来计算每个时期的批次数量。**

然后我们在第 79 行**上循环每一批。**

```py
		# generate images using the noise + generator model
		genImages = gen.predict(noise, verbose=0)

		# concatenate the *actual* images and the *generated* images,
		# construct class labels for the discriminator, and shuffle
		# the data
		X = np.concatenate((imageBatch, genImages))
		y = ([1] * BATCH_SIZE) + ([0] * BATCH_SIZE)
		y = np.reshape(y, (-1,))
		(X, y) = shuffle(X, y)

		# train the discriminator on the data
		discLoss = disc.train_on_batch(X, y)
```

**第 89 行**接受我们的输入`noise`，然后生成合成服装图像(`genImages`)。

给定我们生成的图像，我们需要训练鉴别器来识别真实图像和合成图像之间的差异。

为了完成这个任务，**线 94** 将当前的`imageBatch`和合成的`genImages`连接在一起。

然后我们需要在**行 95** 上建立我们的类别标签——每个*真实*图像将有一个类别标签`1`，而每个虚假图像将被标记为`0`。

然后在第 97 行**上联合混洗连接的训练数据，因此我们的真实和虚假图像不会一个接一个地相继出现(这将在我们的梯度更新阶段造成问题)。**

此外，我发现这种洗牌过程提高了鉴别器训练的稳定性。

**第 100 行**训练当前(混洗)批次的鉴别器。

我们训练过程的最后一步是训练`gan`本身:

```py
		# let's now train our generator via the adversarial model by
		# (1) generating random noise and (2) training the generator
		# with the discriminator weights frozen
		noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
		fakeLabels = [1] * BATCH_SIZE
		fakeLabels = np.reshape(fakeLabels, (-1,))
		ganLoss = gan.train_on_batch(noise, fakeLabels)
```

我们首先生成总共`BATCH_SIZE`个随机向量。然而，不像在我们之前的代码块中，我们很好地告诉我们的鉴别器什么是真的什么是假的，我们现在将试图通过将随机的`noise`标记为真实的图像来欺骗鉴别器。

来自鉴别器的反馈使我们能够实际训练发生器(记住，鉴别器权重对于该操作是冻结的)。

在训练 GAN 时，不仅看损失值很重要，而且你*也*需要检查你`benchmarkNoise`上`gan`的输出:

```py
		# check to see if this is the end of an epoch, and if so,
		# initialize the output path
		if i == batchesPerEpoch - 1:
			p = [args["output"], "epoch_{}_output.png".format(
				str(epoch + 1).zfill(4))]

		# otherwise, check to see if we should visualize the current
		# batch for the epoch
		else:
			# create more visualizations early in the training
			# process
			if epoch < 10 and i % 25 == 0:
				p = [args["output"], "epoch_{}_step_{}.png".format(
					str(epoch + 1).zfill(4), str(i).zfill(5))]

			# visualizations later in the training process are less
			# interesting
			elif epoch >= 10 and i % 100 == 0:
				p = [args["output"], "epoch_{}_step_{}.png".format(
					str(epoch + 1).zfill(4), str(i).zfill(5))]
```

如果我们已经到达了纪元的末尾，我们将构建路径`p`，到我们的输出可视化(**第 112-114 行**)。

否则，我发现在*之前的*步骤中比在*之后的*步骤(**第 118-129 行**)中更频繁地目视检查我们 GAN 的输出会有所帮助。

输出可视化将是完全随机的盐和胡椒噪声在开始，但应该很快开始开发输入数据的特征。这些特征可能看起来不真实，但不断演变的属性将向您展示网络实际上正在学习。

如果在 5-10 个时期之后，您的输出可视化仍然是盐和胡椒噪声，这可能是您需要调整您的超参数的信号，可能包括模型架构定义本身。

我们的最后一个代码块处理将合成图像可视化写入磁盘:

```py
		# check to see if we should visualize the output of the
		# generator model on our benchmark data
		if p is not None:
			# show loss information
			print("[INFO] Step {}_{}: discriminator_loss={:.6f}, "
				"adversarial_loss={:.6f}".format(epoch + 1, i,
					discLoss, ganLoss))

			# make predictions on the benchmark noise, scale it back
			# to the range [0, 255], and generate the montage
			images = gen.predict(benchmarkNoise)
			images = ((images * 127.5) + 127.5).astype("uint8")
			images = np.repeat(images, 3, axis=-1)
			vis = build_montages(images, (28, 28), (16, 16))[0]

			# write the visualization to disk
			p = os.path.sep.join(p)
			cv2.imwrite(p, vis)
```

**141 线**用我们的发电机从我们的`benchmarkNoise`产生`images`。然后，我们将图像数据从范围 *[-1，1]*(*tanh*激活函数的边界)缩放回范围*【0，255】*(**行 142** )。

因为我们正在生成单通道图像，所以我们将图像的灰度表示重复三次，以构建一个 3 通道 RGB 图像(**行 143** )。

`build_montages`函数生成一个 *16×16* 的网格，每个矢量中有一个 *28×28* 的图像。然后在**线 148** 上将剪辑写入磁盘。

### **用 Keras 和 TensorFlow 训练我们的 GAN**

为了在时尚 MNIST 数据集上训练我们的 GAN，请确保使用本教程的 ***【下载】*** 部分下载源代码。

从那里，打开一个终端，并执行以下命令:

```py
$ python dcgan_fashion_mnist.py --output output
[INFO] loading MNIST dataset...
[INFO] building generator...
[INFO] building discriminator...
[INFO] building GAN...
[INFO] starting training...
[INFO] starting epoch 1 of 50...
[INFO] Step 1_0: discriminator_loss=0.683195, adversarial_loss=0.577937
[INFO] Step 1_25: discriminator_loss=0.091885, adversarial_loss=0.007404
[INFO] Step 1_50: discriminator_loss=0.000986, adversarial_loss=0.000562
...
[INFO] starting epoch 50 of 50...
[INFO] Step 50_0: discriminator_loss=0.472731, adversarial_loss=1.194858
[INFO] Step 50_100: discriminator_loss=0.526521, adversarial_loss=1.816754
[INFO] Step 50_200: discriminator_loss=0.500521, adversarial_loss=1.561429
[INFO] Step 50_300: discriminator_loss=0.495300, adversarial_loss=0.963850
[INFO] Step 50_400: discriminator_loss=0.512699, adversarial_loss=0.858868
[INFO] Step 50_500: discriminator_loss=0.493293, adversarial_loss=0.963694
[INFO] Step 50_545: discriminator_loss=0.455144, adversarial_loss=1.128864
```

**图 5** 显示了我们的随机噪声向量(即`benchmarkNoise`在训练的不同时刻):

*   在开始训练 GAN 之前，*左上角的*包含 256 个(在一个 *8×8* 网格中)初始随机噪声向量。**我们可以清楚地看到这种噪音中没有规律。**甘还没学会什么时尚单品。
*   然而，在第二个纪元*(右上)*结束时，类似服装的结构开始出现。
*   到了第五纪元结束(*左下方*)，时尚物品的*明显*更加清晰。
*   当我们到达第 50 个纪元结束时(*右下)，*我们的时尚产品看起来是真实的。

**再次强调，重要的是要理解这些时尚单品是由随机噪声输入向量生成的——*它们完全是合成图像！***

## **总结**

在本教程中，我们讨论了生成敌对网络(GANs)。我们了解到 GANs 实际上由两个网络组成:

1.  负责生成假图像的**生成器**
2.  一个**鉴别器**，它试图从真实图像中识别出合成图像

通过同时训练这两个网络，我们可以学习生成非常逼真的输出图像。

然后我们实现了[深度卷积对抗网络(DC GAN)](https://arxiv.org/abs/1511.06434)，这是 [Goodfellow 等人最初的 GAN 实现的变体。](https://arxiv.org/abs/1406.2661)

使用我们的 DCGAN 实现，我们在时尚 MNIST 数据集上训练了生成器和鉴别器，产生了时尚物品的输出图像:

1.  *不是*训练集的一部分，是*完全合成的*
2.  看起来与时尚 MNIST 数据集中的任何图像几乎相同且无法区分

问题是训练 GANs 可能非常具有挑战性，比我们在 PyImageSearch 博客上讨论的任何其他架构或方法都更具挑战性。

众所周知，gan 很难训练的原因是由于*不断变化的亏损格局*——随着每一步，我们的亏损格局都会略有变化，因此一直在变化。

不断变化的损失状况与其他分类或回归任务形成鲜明对比，在这些任务中，损失状况是“固定的”且不变的。

当训练你自己的 GAN 时，你无疑必须仔细调整你的模型架构和相关的超参数——确保参考本教程顶部的*“训练 GAN 时的指导方针和最佳实践”*部分，以帮助你调整你的超参数并运行你自己的 GAN 实验。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****