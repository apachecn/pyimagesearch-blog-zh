# 基于预训练网络的 PyTorch 图像分类

> 原文：<https://pyimagesearch.com/2021/07/26/pytorch-image-classification-with-pre-trained-networks/>

在本教程中，您将学习如何使用 PyTorch 通过预训练的网络执行影像分类。利用这些网络，您只需几行代码就可以准确地对 1000 种常见的对象进行分类。

今天的教程是 PyTorch 基础知识五部分系列的第四部分:

1.  [*py torch 是什么？*](https://pyimagesearch.com/2021/07/05/what-is-pytorch/)
2.  [*py torch 简介:用 PyTorch*](https://pyimagesearch.com/2021/07/12/intro-to-pytorch-training-your-first-neural-network-using-pytorch/) 训练你的第一个神经网络
3.  [*PyTorch:训练你的第一个卷积神经网络*](https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/)
4.  *使用预训练网络的 PyTorch 图像分类*(今天的教程)
5.  8 月 2 日:用预先训练好的网络进行 PyTorch 物体检测(下周的教程)

在本教程的其余部分中，您将获得使用 PyTorch 对输入图像进行分类的经验，这些输入图像使用创新的、最先进的图像分类网络，包括 VGG、Inception、DenseNet 和 ResNet。

**学习如何用预先训练好的 PyTorch 网络进行图像分类，** ***继续阅读。***

## **使用预训练网络的 PyTorch 图像分类**

在本教程的第一部分，我们将讨论什么是预训练的图像分类网络，包括 PyTorch 库中内置的网络。

从那里，我们将配置我们的开发环境，并审查我们的项目目录结构。

然后，我将向您展示如何实现一个 Python 脚本，该脚本可以使用预先训练的 PyTorch 网络对输入图像进行准确分类。

我们将讨论我们的结果来结束本教程。

### **什么是预训练的图像分类网络？**

说到图像分类，没有比 [ImageNet](https://www.image-net.org/) 更出名的数据集/挑战了。**ImageNet 的目标是将输入图像准确地分类到一组 1000 个常见的对象类别中，计算机视觉系统将在日常生活中“看到”这些类别。**

大多数流行的深度学习框架，包括 PyTorch，Keras，TensorFlow，fast.ai 等，都包含了*预先训练好的*网络。这些是计算机视觉研究人员在 ImageNet 数据集上训练的高度准确、最先进的模型。

ImageNet 培训完成后，研究人员将他们的模型保存到磁盘上，然后免费发布给其他研究人员、学生和开发人员，供他们在自己的项目中学习和使用。

本教程将展示如何使用 PyTorch 通过以下先进的分类网络对输入图像进行分类:

*   VGG16
*   VGG19
*   开始
*   DenseNet
*   ResNet

我们开始吧！

### **配置您的开发环境**

要遵循这个指南，您需要在系统上安装 PyTorch 和 OpenCV。

幸运的是，PyTorch 和 OpenCV 都非常容易使用 pip 安装:

```py
$ pip install torch torchvision
$ pip install opencv-contrib-python
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

### **项目结构**

在我们用 PyTorch 实现图像分类之前，让我们先回顾一下我们的项目目录结构。

首先访问本指南的 ***“下载”*** 部分，检索源代码和示例图像。然后，您将看到下面的目录结构。

```py
$ tree . --dirsfirst
.
├── images
│   ├── bmw.png
│   ├── boat.png
│   ├── clint_eastwood.jpg
│   ├── jemma.png
│   ├── office.png
│   ├── scotch.png
│   ├── soccer_ball.jpg
│   └── tv.png
├── pyimagesearch
│   └── config.py
├── classify_image.py
└── ilsvrc2012_wordnet_lemmas.txt
```

在`pyimagesearch`模块中，我们有一个单独的文件`config.py`。该文件存储重要的配置，例如:

*   我们的输入图像尺寸
*   均值相减和缩放的均值和标准差
*   无论我们是否使用 GPU 进行训练
*   人类可读的 ImageNet 类标签的路径(即`ilsvrc2012_wordnet_lemmas.txt`)

我们的`classify_image.py`脚本将加载我们的`config`，然后使用 VGG16、VGG19、Inception、DenseNet 或 ResNet(取决于我们作为命令行参数提供的模型架构)对输入图像进行分类。

`images`目录包含了许多样本图像，我们将在其中应用这些图像分类网络。

### **创建我们的配置文件**

在实现我们的图像分类驱动程序脚本之前，让我们首先创建一个配置文件来存储重要的配置。

打开`pyimagesearch`模块中的`config.py`文件，插入以下代码:

```py
# import the necessary packages
import torch

# specify image dimension
IMAGE_SIZE = 224

# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# determine the device we will be using for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# specify path to the ImageNet labels
IN_LABELS = "ilsvrc2012_wordnet_lemmas.txt"
```

**第 5 行**定义了我们的输入图像空间维度，这意味着每幅图像在通过我们预先训练的 PyTorch 网络进行分类之前，将被调整到 *224×224* 像素。

***注:*** *在 ImageNet 数据集上训练的大多数网络接受 224×224 或 227×227 的图像。一些网络，尤其是全卷积网络，可以接受更大的图像尺寸。*

从那里，我们定义了我们的训练集的 RGB 像素强度的平均值和标准偏差(**行 8 和 9** )。在将输入图像通过我们的网络进行分类之前，我们首先通过减去平均值，然后除以标准偏差来缩放图像像素强度，这种预处理对于在 ImageNet 等大型多样化图像数据集上训练的 CNN 来说是典型的。

从那里，**行 12** 指定我们是使用我们的 CPU 还是 GPU 进行训练，而**行 15** 定义 ImageNet 类标签的输入文本文件的路径。

如果在您喜欢的文本编辑器中打开该文件，您将看到以下内容:

```py
tench, Tinca_tinca
goldfish, Carassius_auratus
...
bolete
ear, spike, capitulum
toilet_tissue, toilet_paper, bathroom_tissue
```

这个文本文件中的每一行都映射到我们的预训练 PyTorch 网络被训练来识别和分类的类标签的名称。

### **实现我们的图像分类脚本**

有了我们的配置文件，让我们继续实现我们的主要驱动程序脚本，使用我们预先训练的 PyTorch 网络对输入图像进行分类。

打开项目目录结构中的`classify_image.py`文件，让我们开始工作:

```py
# import the necessary packages
from pyimagesearch import config
from torchvision import models
import numpy as np
import argparse
import torch
import cv2
```

我们从第 2-7 行开始导入我们的 Python 包，包括:

*   `config`:我们在上一节中实现的配置文件
*   包含 PyTorch 的预训练神经网络
*   `numpy`:数值数组处理
*   `torch`:访问 PyTorch API
*   我们的 OpenCV 绑定

考虑到我们的导入，让我们定义一个函数来接受输入图像并对其进行预处理:

```py
def preprocess_image(image):
	# swap the color channels from BGR to RGB, resize it, and scale
	# the pixel values to [0, 1] range
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
	image = image.astype("float32") / 255.0

	# subtract ImageNet mean, divide by ImageNet standard deviation,
	# set "channels first" ordering, and add a batch dimension
	image -= config.MEAN
	image /= config.STD
	image = np.transpose(image, (2, 0, 1))
	image = np.expand_dims(image, 0)

	# return the preprocessed image
	return image
```

我们的 `preprocess_image`函数接受一个参数`image`，这是我们将要进行分类预处理的图像。

我们通过以下方式开始预处理操作:

1.  从 BGR 到 RGB 通道排序的交换(我们这里使用的预训练网络利用 RGB 通道排序，而 OpenCV 默认使用 BGR 排序)
2.  将我们的图像调整到固定尺寸(即 *224×224* )，忽略纵横比
3.  将我们的图像转换为浮点数据类型，然后将像素亮度缩放到范围*【0，1】*

从那里，我们执行第二组预处理操作:

1.  减去平均值(**行 18** )并除以标准偏差(**行 19** )
2.  将通道维度移动到数组(**第 20 行**)的*前面*，称为**通道优先排序**，是 PyTorch 期望的默认通道排序方式
3.  向数组添加批次维度(**第 21 行**)

经过预处理的`image`然后被返回给调用函数。

接下来，让我们解析我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="vgg16",
	choices=["vgg16", "vgg19", "inception", "densenet", "resnet"],
	help="name of pre-trained network to use")
args = vars(ap.parse_args())
```

我们有两个命令行参数要解析:

1.  `--image`:我们希望分类的输入图像的路径
2.  我们将使用预先训练好的 CNN 模型对图像进行分类

现在让我们定义一个`MODELS`字典，它将`--model`命令行参数的名称映射到其对应的 PyTorch 函数:

```py
# define a dictionary that maps model names to their classes
# inside torchvision
MODELS = {
	"vgg16": models.vgg16(pretrained=True),
	"vgg19": models.vgg19(pretrained=True),
	"inception": models.inception_v3(pretrained=True),
	"densenet": models.densenet121(pretrained=True),
	"resnet": models.resnet50(pretrained=True)
}

# load our the network weights from disk, flash it to the current
# device, and set it to evaluation mode
print("[INFO] loading {}...".format(args["model"]))
model = MODELS[args["model"]].to(config.DEVICE)
model.eval()
```

**第 37-43 行**创建我们的`MODELS`字典:

*   字典的*键*是模型的可读名称，通过`--model`命令行参数传递。
*   字典的*值*是相应的 PyTorch 函数，用于加载带有 ImageNet 上预先训练的权重的模型

通过 PyTorch，您将能够使用以下预训练模型对输入图像进行分类:

1.  VGG16
2.  VGG19
3.  开始
4.  DenseNet
5.  ResNet

指定`pretrained=True`标志指示 PyTorch 不仅加载模型架构定义，而且*还*下载模型的预训练 ImageNet 权重。

**Line 48** 然后加载模型和预训练的权重(如果你之前从未下载过模型权重，它们会自动为你下载和缓存),然后将模型设置为在你的 CPU 或 GPU 上运行，这取决于你在配置文件中的`DEVICE`。

**第 49 行**将我们的`model`置于评估模式，指示 PyTorch 处理特殊层，如退出和批量标准化，这与训练期间处理它们的方式不同。**在做出预测之前，将你的模型置于评估模式*是至关重要的*，所以别忘了这么做！**

既然我们的模型已经加载，我们需要一个输入图像——现在让我们来处理它:

```py
# load the image from disk, clone it (so we can draw on it later),
# and preprocess it
print("[INFO] loading image...")
image = cv2.imread(args["image"])
orig = image.copy()
image = preprocess_image(image)

# convert the preprocessed image to a torch tensor and flash it to
# the current device
image = torch.from_numpy(image)
image = image.to(config.DEVICE)

# load the preprocessed the ImageNet labels
print("[INFO] loading ImageNet labels...")
imagenetLabels = dict(enumerate(open(config.IN_LABELS)))
```

**线 54** 从磁盘加载我们的输入`image`。我们在**第 55 行**做了一份拷贝，这样我们就可以利用它来想象我们网络的最高预测。我们还利用第 56 行上的`preprocess_image`函数来执行尺寸调整和缩放。

**第 60 行**将我们的`image`从 NumPy 数组转换为 PyTorch 张量，而**第 61 行**将`image`移动到我们的设备(CPU 或 GPU)。

最后，**行 65** 从磁盘加载我们的输入 ImageNet 类标签。

我们现在准备使用我们的`model`对输入`image`进行预测:

```py
# classify the image and extract the predictions
print("[INFO] classifying image with '{}'...".format(args["model"]))
logits = model(image)
probabilities = torch.nn.Softmax(dim=-1)(logits)
sortedProba = torch.argsort(probabilities, dim=-1, descending=True)

# loop over the predictions and display the rank-5 predictions and
# corresponding probabilities to our terminal
for (i, idx) in enumerate(sortedProba[0, :5]):
	print("{}. {}: {:.2f}%".format
		(i, imagenetLabels[idx.item()].strip(),
		probabilities[0, idx.item()] * 100))
```

第 69 行执行我们网络的前向传递，产生网络的输出。

我们通过**第 70 行**上的`Softmax`函数传递这些，以获得`model`被训练的 1000 个可能的类标签中的每一个的预测概率。

**第 71 行**然后按照降序排列概率，在列表的*前端*概率较高。

然后，我们通过以下方式在第 75-78 行的**终端上显示前 5 个预测类别标签和相应的概率:**

*   循环前 5 个预测
*   使用我们的`imagenetLabels`字典查找类标签的名称
*   显示预测的概率

我们的最终代码块在输出图像上绘制了 top-1(即顶部预测标签):

```py
# draw the top prediction on the image and display the image to
# our screen
(label, prob) = (imagenetLabels[probabilities.argmax().item()],
	probabilities.max().item())
cv2.putText(orig, "Label: {}, {:.2f}%".format(label.strip(), prob * 100),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)
```

结果会显示在我们的屏幕上。

### **使用 PyTorch 结果的图像分类**

我们现在准备用 PyTorch 应用图像分类！

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

在那里，尝试使用以下命令对输入图像进行分类:

```py
$ python classify_image.py --image images/boat.png
[INFO] loading vgg16...
[INFO] loading image...
[INFO] loading ImageNet labels...
[INFO] classifying image with 'vgg16'...
0\. wreck: 99.99%
1\. seashore, coast, seacoast, sea-coast: 0.01%
2\. pirate, pirate_ship: 0.00%
3\. breakwater, groin, groyne, mole, bulwark, seawall, jetty: 0.00%
4\. sea_lion: 0.00%
```

杰克·斯派洛船长似乎被困在了海滩上！果然，VGG16 网络能够以 99.99%的概率将输入图像正确分类为“沉船”(即沉船)。

有趣的是,“海滨”是模型的第二大预测——这个预测也很准确，因为船在沙滩上。

让我们尝试一个不同的图像，这次使用 DenseNet 模型:

```py
$ python classify_image.py --image images/bmw.png --model densenet
[INFO] loading densenet...
[INFO] loading image...
[INFO] loading ImageNet labels...
[INFO] classifying image with 'densenet'...
0\. convertible: 96.61%
1\. sports_car, sport_car: 2.25%
2\. car_wheel: 0.45%
3\. beach_wagon, station_wagon, wagon, estate_car, beach_waggon, station_waggon, waggon: 0.22%
4\. racer, race_car, racing_car: 0.13%
```

来自 DenseNet 的顶级预测是“可转换的”，准确率为 96.61%。第二顶预测，“跑车”也准。

这张图片包含了 Jemma，我家的小猎犬:

```py
$ python classify_image.py --image images/jemma.png --model resnet
[INFO] loading resnet...
[INFO] loading image...
[INFO] loading ImageNet labels...
[INFO] classifying image with 'resnet'...
0\. beagle: 95.98%
1\. bluetick: 1.46%
2\. Walker_hound, Walker_foxhound: 1.11%
3\. English_foxhound: 0.45%
4\. maraca: 0.25%
```

这里我们使用 ResNet 架构对输入图像进行分类。Jemma 是一只“小猎犬”(狗的一种)，ResNet 以 95.98%的概率准确预测。

有趣的是，一只“蓝蜱”、“步行猎犬”和“英国猎狐犬”都是属于“猎犬”家族的狗——所有这些都是模型的合理预测。

让我们看看最后一个例子:

```py
$ python classify_image.py --image images/soccer_ball.jpg --model inception
[INFO] loading inception...
[INFO] loading image...
[INFO] loading ImageNet labels...
[INFO] classifying image with 'inception'...
0\. soccer_ball: 100.00%
1\. volleyball: 0.00%
2\. sea_urchin: 0.00%
3\. rugby_ball: 0.00%
4\. silky_terrier, Sydney_silky: 0.00%
```

我们的初始模型以 100%的概率正确地将输入图像分类为“足球”。

图像分类允许我们给输入图像分配一个或多个标签；然而，它没有告诉我们*任何关于*物体在图像*中的位置*的信息。

为了确定给定对象在输入图像中的位置，我们需要应用**对象检测:**

就像我们有用于图像分类的预训练网络一样，我们也有用于对象检测的预训练网络。下周您将学习如何使用 PyTorch 通过专门的对象检测网络来检测图像中的对象。

## **总结**

在本教程中，您学习了如何使用 PyTorch 执行影像分类。具体来说，我们利用了流行的预培训网络架构，包括:

*   VGG16
*   VGG19
*   开始
*   DenseNet
*   ResNet

这些模型是由负责发明和提出上面列出的新颖架构的研究人员训练的。训练完成后，这些研究人员将模型权重保存到磁盘上，然后发布给其他研究人员、学生和开发人员，供他们在自己的项目中学习和使用。

虽然模型可以免费使用，但请确保您检查了与它们相关的任何条款/条件，因为一些模型在商业应用中不能免费使用(通常人工智能领域的企业家通过训练模型本身而不是使用原作者提供的预训练权重来绕过这一限制)。

请继续关注下周的博文，在那里您将学习如何使用 PyTorch 执行对象检测。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******