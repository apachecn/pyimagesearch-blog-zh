# 使用 Python 构建暹罗网络的影像对

> 原文：<https://pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/>

在本教程中，您将学习如何为训练连体网络构建图像对。我们将使用 Python 实现我们的图像对生成器，这样无论您是否使用 TensorFlow、Keras、PyTorch 等，您都可以使用*相同的代码*。

本教程是暹罗网络介绍的第一部分:

*   **第一部分:** *用 Python 为暹罗网络构建图像对*(今天的帖子)
*   **第二部分:** *用 Keras、TensorFlow 和深度学习训练暹罗网络*(下周教程)
*   **第三部分:** *使用连体网络比较图像*(从现在起两周后的教程)

暹罗网络*是极其强大的*网络，负责*人脸识别、签名验证和处方药识别应用(仅举几例)的显著*增长。

事实上，如果你跟随我的关于 *[OpenCV 人脸识别](https://pyimagesearch.com/2018/09/24/opencv-face-recognition/)* 或 *[人脸识别与 OpenCV、Python 和深度学习](https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)* 的教程，你会看到这些帖子中使用的深度学习模型是连体网络！

深度学习模型如 FaceNet、VGGFace、dlib 的 ResNet 人脸识别模型都是连体网络的例子。

**此外，暹罗网络使更高级的训练程序成为可能，如一次性学习和少量学习** —与其他深度学习架构相比，暹罗网络需要*非常少的*训练示例才能有效。

今天我们将:

*   复习暹罗网络的基础知识
*   讨论图像对的概念
*   看看我们如何使用图像对来训练一个连体网络
*   实现 Python 代码以生成暹罗网络的图像对

下周我将向你展示如何实现和训练你自己的暹罗网络。最后，我们将建立图像三元组的概念，以及如何使用三元组损失和对比损失来训练更好、更准确的连体网络。

但是现在，让我们来理解图像对，这是实现基本暹罗网络时的一个基本要求。

**要了解如何为连体网络构建图像对，*继续阅读。***

## **使用 Python 构建暹罗网络的图像对**

在本教程的第一部分，我将提供暹罗网络的高级概述，包括:

*   它们是什么
*   我们为什么使用它们
*   何时使用它们
*   他们是如何被训练的

然后我们将讨论暹罗网络中“图像对”的概念，包括为什么在训练暹罗网络时构建图像对是一个*要求*。

从这里，我们将回顾我们的项目目录结构，然后实现一个 Python 脚本来生成图像对。无论您使用的是 Keras、TensorFlow、PyTorch 等，您都可以在自己的暹罗网络训练程序中使用此图像对生成功能。

最后，我们将通过回顾我们的结果来结束本教程。

### **暹罗网络的高级概述**

术语“连体双胞胎”，也称为“连体双胞胎”，是两个在子宫内结合的同卵双胞胎。这些双胞胎身体上彼此相连(即不能分开)，通常共享相同的器官，主要是下肠道、肝脏和泌尿道。

**就像连体双胞胎一样，*连体网络也是。***

转述[肖恩·本赫尔](https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942)，**暹罗网络是一类特殊的神经网络:**

*   暹罗网络包含两个(或更多)相同的子网。
*   这些子网具有*相同的*架构、参数和权重。
*   任何参数更新都会在两个子网中*镜像*，这意味着如果您更新了*一个*中的权重，那么*另一个*中的权重也会更新。

我们在执行验证、识别或识别任务时使用暹罗网络，最常见的例子是**人脸识别**和**签名验证。**

例如，假设我们的任务是检测签名伪造。不是训练一个分类模型来正确地*分类我们数据集中每个独特个体的*签名(这将需要*重要的*训练数据)，而是从我们的训练集中取两张图像，并询问神经网络这些签名是否来自*同一个*人，会怎么样？

*   如果两个签名相同，那么暹罗网络报告*“是”。*
*   否则，如果两个签名不相同，从而意味着潜在的伪造，暹罗网络报告*“否”。*

这是一个**验证任务**的例子(相对于分类、回归等)。)，虽然这听起来像是一个更难的问题，但在实践中它实际上变得*容易得多**—*我们需要*明显更少的*训练数据，并且我们的准确性实际上*通过使用暹罗网络而不是分类网络来提高*。

另一个额外的好处是，当我们的分类模型在进行分类时需要选择“以上都不是”时，我们不再需要一个“包罗万象”的类(这在实践中很容易出错)。相反，我们的暹罗网络优雅地处理了这个问题，报告说这两个签名是不同的。

请记住，暹罗网络体系结构不必关心传统意义上的分类，即从 *N* 个可能的类别中选择 1 个。更确切地说，暹罗网络只需要能够报告“相同”(属于同一类)或“不同”(属于不同类)。

下面是 Dey 等人 2017 年的出版物中使用的暹罗网络架构的可视化， *[SigNet:用于书写者独立离线签名验证的卷积暹罗网络](https://arxiv.org/abs/1707.02131) :*

在左边的*中，我们向图章模型提交了两个签名。我们的目标是确定这些签名是否属于同一个人。*

中间的*表示暹罗网络本身。**这两个子网络具有*相同的*架构和参数，并且*相互镜像***—如果一个子网络中的权重被更新，那么其他子网络中的权重也会被更新。*

这些子网络中的最终层通常(但不总是)是嵌入层，其中我们可以计算输出之间的欧几里德距离，并调整子网络的权重，以使它们输出正确的决策(是否属于同一类)。

右边的*显示了我们的损失函数，它结合了子网的输出，然后检查暹罗网络是否做出了正确的决定。*

训练暹罗网络时常用的损失函数包括:

*   二元交叉熵
*   三重损失
*   对比损失

你可能会惊讶地看到二进制交叉熵被列为训练暹罗网络的损失函数。

请这样想:

每个图像对或者是“相同的”(`1`)，意味着它们属于同一类，或者是“不同的”(`0`)，意味着它们属于不同的类。这自然有助于二进制交叉熵，因为只有两种可能的输出(尽管三重损失和对比损失往往明显优于标准二进制交叉熵)。

既然我们对暹罗网络有了一个较高层次的概述，现在让我们来讨论图像对的概念。

### **暹罗网络中“像对”的概念**

阅读完上一节后，您应该明白，连体网络由*两个相互镜像的子网*组成(即，当一个网络中的权重更新时，另一个网络中的权重也会更新)。

**由于有*两个子网络，*我们必须有*两个输入*到连体模型**(正如你在前面章节顶部的**图 2** 中看到的)。

当训练暹罗网络时，我们需要有**个正对**和**个负对:**

*   **正对:**属于同一类*的两幅图像(例如。同一个人的两个图像、同一签名的两个示例等。)*
*   **负对:**属于*不同*类的两幅图像(例如。不同人的两个图像、不同签名的两个示例等。)

当训练我们的暹罗网络时，我们随机抽取正对和负对的样本。这些对作为我们的训练数据，以便暹罗网络可以学习相似性。

在本教程的剩余部分，您将学习如何生成这样的图像对。在下周的教程中，您将学习如何定义暹罗网络架构，然后在我们的 pairs 数据集上训练暹罗模型。

### **配置您的开发环境**

在这一系列关于暹罗网络的教程中，我们将使用 Keras 和 TensorFlow，所以我建议你现在就花时间配置你的深度学习开发环境。

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

确保您使用了本教程的 ***【下载】*** 部分来下载源代码。从那里，让我们检查项目目录结构:

```py
$ tree . --dirsfirst
.
└── build_siamese_pairs.py

0 directories, 1 file
```

### **为暹罗网络实现我们的图像对生成器**

让我们开始为暹罗网络实现图像对生成。

打开`build_siamese_pairs.py`文件，插入以下代码:

```py
# import the necessary packages
from tensorflow.keras.datasets import mnist
from imutils import build_montages
import numpy as np
import cv2
```

**第 2-5 行**导入我们需要的 Python 包。

我们将使用 MNIST 数字数据集作为我们的样本数据集(为了方便起见)。也就是说，我们的`make_pairs`函数将与任何图像数据集的*一起工作，只要你提供两个单独的`image`和`labels`数组(你将在下一个代码块中学习如何做)。*

```py
def make_pairs(images, labels):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pairImages = []
	pairLabels = []
```

下一步是计算数据集中唯一类标签的总数:

```py
	# calculate the total number of classes present in the dataset
	# and then build a list of indexes for each class label that
	# provides the indexes for all examples with a given label
	numClasses = len(np.unique(labels))
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)]
```

**第 16 行**使用`np.unique`函数在我们的`labels`列表中查找所有唯一的类标签。取`np.unique`输出的`len`,得到数据集中唯一类标签的总数。在 MNIST 数据集的情况下，有 10 个唯一的类标签，对应于数字 *0-9。*

**第 17 行**然后使用 Python 数组理解为每个类标签构建一个索引列表。为了提高性能，我们在这里使用 Python 列表理解；然而，这段代码可能有点难以理解，所以让我们将它分解为一个专用的`for`循环，以及一些`print`语句:

```py
>>> for i in range(0, numClasses):
>>>	idxs = np.where(labels == i)[0]
>>>	print("{}: {} {}".format(i, len(idxs), idxs))
0: 5923 [    1    21    34 ... 59952 59972 59987]
1: 6742 [    3     6     8 ... 59979 59984 59994]
2: 5958 [    5    16    25 ... 59983 59985 59991]
3: 6131 [    7    10    12 ... 59978 59980 59996]
4: 5842 [    2     9    20 ... 59943 59951 59975]
5: 5421 [    0    11    35 ... 59968 59993 59997]
6: 5918 [   13    18    32 ... 59982 59986 59998]
7: 6265 [   15    29    38 ... 59963 59977 59988]
8: 5851 [   17    31    41 ... 59989 59995 59999]
9: 5949 [    4    19    22 ... 59973 59990 59992]
>>>
```

第 17 行构建了这个索引列表，但是以一种超级紧凑、高效的方式。

给定我们的`idx`循环列表，让我们现在开始产生我们的积极和消极对:

```py
	# loop over all images
	for idxA in range(len(images)):
		# grab the current image and label belonging to the current
		# iteration
		currentImage = images[idxA]
		label = labels[idxA]

		# randomly pick an image that belongs to the *same* class
		# label
		idxB = np.random.choice(idx[label])
		posImage = images[idxB]

		# prepare a positive pair and update the images and labels
		# lists, respectively
		pairImages.append([currentImage, posImage])
		pairLabels.append([1])
```

接下来，让我们生成我们的**负对:**

```py
		# grab the indices for each of the class labels *not* equal to
		# the current label and randomly pick an image corresponding
		# to a label *not* equal to the current label
		negIdx = np.where(labels != label)[0]
		negImage = images[np.random.choice(negIdx)]

		# prepare a negative pair of images and update our lists
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])

	# return a 2-tuple of our image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))
```

**第 39 行**抓取`labels` *不*等于当前`label`的所有索引。然后，我们随机选择这些指标中的一个作为我们的负面图像，`negImage` ( **Line 40** )。

```py
# load MNIST dataset and scale the pixel values to the range of [0, 1]
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()

# build the positive and negative image pairs
print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = make_pairs(trainX, trainY)
(pairTest, labelTest) = make_pairs(testX, testY)

# initialize the list of images that will be used when building our
# montage
images = []
```

**第 51 行**从磁盘加载 MNIST 训练和测试分割。

然后，我们在**行 55 和 56** 上生成训练和测试对。

```py
# loop over a sample of our training pairs
for i in np.random.choice(np.arange(0, len(pairTrain)), size=(49,)):
	# grab the current image pair and label
	imageA = pairTrain[i][0]
	imageB = pairTrain[i][1]
	label = labelTrain[i]

	# to make it easier to visualize the pairs and their positive or
	# negative annotations, we're going to "pad" the pair with four
	# pixels along the top, bottom, and right borders, respectively
	output = np.zeros((36, 60), dtype="uint8")
	pair = np.hstack([imageA, imageB])
	output[4:32, 0:56] = pair

	# set the text label for the pair along with what color we are
	# going to draw the pair in (green for a "positive" pair and
	# red for a "negative" pair)
	text = "neg" if label[0] == 0 else "pos"
	color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)

	# create a 3-channel RGB image from the grayscale pair, resize
	# it from 60x36 to 96x51 (so we can better see it), and then
	# draw what type of pair it is on the image
	vis = cv2.merge([output] * 3)
	vis = cv2.resize(vis, (96, 51), interpolation=cv2.INTER_LINEAR)
	cv2.putText(vis, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
		color, 2)

	# add the pair visualization to our list of output images
	images.append(vis)
```

这里的最后一步是构建我们的蒙太奇并将其显示到我们的屏幕上:

```py
# construct the montage for the images
montage = build_montages(images, (96, 51), (7, 7))[0]

# show the output montage
cv2.imshow("Siamese Image Pairs", montage)
cv2.waitKey(0)
```

第 94 行构建了一个 *7×7* 的蒙太奇，其中蒙太奇中的每个图像是 *96×51* 像素。

输出的连体图像对可视化显示在我们屏幕的第 97 和 98 行**。**

### **连体网络图像对生成结果**

我们现在准备运行我们的暹罗网络图像对生成脚本。确保使用本教程的 ***【下载】*** 部分下载源代码。

从那里，打开一个终端，并执行以下命令:

```py
$ python build_siamese_pairs.py
[INFO] loading MNIST dataset...
[INFO] preparing positive and negative pairs...
```

**图 5** 显示了我们的图像对生成脚本的输出。对于每一对图像，我们的脚本都将它们标记为*正对*(绿色)或*负对*(红色)。

例如，位于第一行第一列的对是一个*正对，*，因为两个数字都是 9。

然而，位于第一行第三列的数字对是一个*负对*，因为一个数字是“2”，而另一个是“0”。

在训练过程中，我们的暹罗网络将学习如何区分这两个数字。

一旦您了解如何以这种方式训练暹罗网络，您就可以替换 MNIST 数字数据集，并包含您自己的任何需要验证的数据集，包括:

*   **人脸识别:**给定两张包含人脸的独立图像，确定两张照片中的*是否是同一个人*。
*   **签名验证:**当呈现两个签名时，确定其中一个是否是伪造的。
*   **处方药丸识别:**给定两种处方药丸，确定它们是相同的药物还是不同的药物。

暹罗网络使所有这些应用成为可能— **下周我将向你展示如何训练你的第一个暹罗网络！**

## **总结**

在本教程中，您学习了如何使用 Python 编程语言为暹罗网络构建影像对。

我们的图像对生成实现是*库不可知的*，这意味着无论你的底层深度学习库是 Keras、TensorFlow、PyTorch 等，你都可以使用这个代码*。*

图像对生成是暹罗网络的一个基本方面。一个连体网络需要了解 ***同一类*【正对】**的两个图像和 ***不同类*【负对】的两个图像的区别。**

在训练过程中，我们可以更新我们网络的权重，这样它就可以区分相同类别的两幅图像和不同类别的两幅图像。

这听起来像是一个复杂的训练过程，但是正如我们下周将会看到的，它实际上是非常简单的(当然，一旦有人向你解释了它！).

敬请关注下周的培训暹罗网络教程，你不会想错过它。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***