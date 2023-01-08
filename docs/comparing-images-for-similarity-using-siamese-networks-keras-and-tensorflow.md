# 使用 siamese 网络、Keras 和 TensorFlow 比较图像的相似性

> 原文：<https://pyimagesearch.com/2020/12/07/comparing-images-for-similarity-using-siamese-networks-keras-and-tensorflow/>

在本教程中，您将学习如何使用 siamese 网络和 Keras/TensorFlow 深度学习库来比较两幅图像的相似性(以及它们是否属于相同或不同的类别)。

这篇博文是我们关于暹罗网络基础的三部分系列的第三部分:

*   **第一部分:** *[用 Python](https://pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/)* 构建连体网络的图像对(两周前的帖子)
*   **Part #2:** *[用 Keras、TensorFlow、深度学习](https://pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/)* 训练暹罗网络(上周教程)
*   **第三部分:** *使用连体网络比较图像*(本教程)

上周我们学习了如何训练我们的暹罗网络。我们的模型在我们的测试集上表现良好，正确地验证了两幅图像是属于相同的类别还是不同的类别。训练之后，我们将模型序列化到磁盘上。

上周的教程发布后不久，我收到了一封来自 PyImageSearch 读者 Scott 的电子邮件，他问:

> “你好，阿德里安——感谢暹罗网络上的这些指南。我听到有人在深度学习领域提到它们，但老实说，我从来没有真正确定它们是如何工作的，或者它们做了什么。这个系列确实帮助我消除了疑虑，甚至帮助了我的一个工作项目。
> 
> *我的问题是:*
> 
> ***我们如何利用训练好的暹罗网络，从训练和测试集之外的图像中对其进行预测？***
> 
> 这可能吗？

没错，斯科特。这正是我们今天要讨论的内容。

**要了解如何使用暹罗网络比较图像的相似性，*继续阅读。***

## **使用 siamese 网络、Keras 和 TensorFlow 比较图像的相似性**

在本教程的第一部分，我们将讨论如何使用一个经过*训练的*暹罗网络来预测两个图像对之间的相似性，更具体地说，这两个输入图像是属于*相同的*还是*不同的*类别。

然后，您将学习如何使用 Keras 和 TensorFlow 为暹罗网络配置开发环境。

一旦您的开发环境配置完毕，我们将检查我们的项目目录结构，然后使用我们的 siamese 网络实现一个 Python 脚本来比较图像的相似性。

我们将讨论我们的结果来结束本教程。

### **暹罗网络如何预测图像对之间的相似性？**

在上周的教程中，你学习了如何训练一个连体网络来验证两对数字是属于*相同的*还是*不同的*类。然后，我们在训练后将我们的连体模型序列化到磁盘上。

那么问题就变成了:

> “我们如何使用我们训练过的暹罗网络来预测两幅图像之间的**相似度**？”

答案是我们利用了我们的暹罗网络实现中的最后一层，即 **sigmoid 激活函数。**

sigmoid 激活函数有一个范围为*【0，1】*的输出，这意味着当我们向我们的暹罗网络呈现一个图像对时，该模型将输出一个值 *> = 0* 和 *< = 1。*

值`0`意味着两幅图像*完全不相似，*，而值`1`意味着两幅图像*非常相似。*

这种相似性的一个例子可以在本节顶部的图 1 中看到:

*   将一个*“7”*与一个*“0”*进行比较，相似性得分很低，只有 0.02。
*   然而，将一个*“0”*与另一个*“0”*进行比较，具有非常高的相似性得分 0.93。

一个好的经验法则是使用相似性临界值`0.5` (50%)作为阈值:

*   如果两个图像对的图像相似度为 *< = 0.5，*，则它们属于*不同的*类。
*   相反，如果对具有预测的相似性 *> 0.5，*，那么它们属于*相同的*类。

通过这种方式，您可以使用暹罗网络来(1)比较图像的相似性，以及(2)确定它们是否属于同一类别。

使用暹罗网络的实际使用案例包括:

*   **人脸识别:**给定两张包含人脸的独立图像，确定两张照片中的*是否是同一个人*。
*   **签名验证:**当呈现两个签名时，确定其中一个是否是伪造的。
*   **处方药标识:**给定两个处方药，确定是同一药物还是不同药物。

### **配置您的开发环境**

这一系列关于暹罗网络的教程利用了 Keras 和 TensorFlow。如果您打算继续学习本教程或本系列的前两部分，我建议您现在花时间配置您的深度学习开发环境。

您可以利用这两个指南中的任何一个在您的系统上安装 TensorFlow 和 Keras:

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

在我们深入本教程之前，让我们先花点时间回顾一下我们的项目目录结构。

首先确保使用本教程的 ***“下载”*** 部分下载源代码和示例图像。

接下来，让我们来看看这个项目:

```py
$ tree . --dirsfirst
.
├── examples
│   ├── image_01.png
...
│   └── image_13.png
├── output
│   ├── siamese_model
│   │   ├── variables
│   │   │   ├── variables.data-00000-of-00001
│   │   │   └── variables.index
│   │   └── saved_model.pb
│   └── plot.png
├── pyimagesearch
│   ├── config.py
│   ├── siamese_network.py
│   └── utils.py
├── test_siamese_network.py
└── train_siamese_network.py

4 directories, 21 files
```

在`examples`目录中，我们有许多示例数字:

我们将对这些数字进行采样，然后使用我们的暹罗网络比较它们的相似性。

`output`目录包含训练历史图(`plot.png`)和我们训练/序列化的暹罗网络模型(`siamese_model/`)。这两个文件都是在[上周关于训练你自己的定制暹罗网络模型的教程](https://pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/)中生成的— **确保你在继续之前阅读该教程，因为它是今天的*必读内容*！**

`pyimagesearch`模块包含三个 Python 文件:

1.  ``config.py`` :我们的配置文件，存储输出文件路径、训练配置等重要变量(包括图像输入尺寸、批量大小、时期等。)
2.  ``siamese_network.py`` :我们实施我们的暹罗网络架构
3.  ``utils.py`` :包含助手配置功能，用于生成图像对、计算欧几里德距离和绘制训练历史路径

`train_siamese_network.py`脚本:

1.  导入配置、暹罗网络实现和实用程序功能
2.  从磁盘加载 MNIST 数据集
3.  生成图像对
4.  创建我们的训练/测试数据集分割
5.  训练我们的暹罗网络
6.  将训练好的连体网络序列化到磁盘

**我*不会*今天讲述这四个脚本，因为我*已经*在[上周关于如何训练暹罗网络的教程](https://pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/)** 中讲述了它们。为了完整起见，我已经在今天的教程的项目目录结构中包含了这些文件，但是，要全面回顾这些文件，它们做什么，以及它们是如何工作的，请参考上周的教程。

最后，我们有今天教程的重点，`test_siamese_network.py`。

我们开始工作吧！

### **实现我们的暹罗网络图像相似性脚本**

我们现在准备使用 Keras 和 TensorFlow 实现图像相似性的连体网络。

首先，确保您使用本教程的 ***【下载】*** 部分下载源代码、示例图像和预训练的暹罗网络模型。

从那里，打开`test_siamese_network.py`，并跟随:

```py
# import the necessary packages
from pyimagesearch import config
from pyimagesearch import utils
from tensorflow.keras.models import load_model
from imutils.paths import list_images
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
```

我们首先导入我们需要的 Python 包(**第 2-9 行**)。值得注意的进口包括:

*   ``config`` :包含重要的配置，包括驻留在磁盘上的经过训练/序列化的暹罗网络模型的路径
*   ``utils`` :包含在我们的暹罗网络`Lambda`层中使用的`euclidean_distance`函数— **我们需要导入这个包来抑制任何关于从磁盘**加载 `UserWarnings` **层的** `Lambda`
*   ``load_model``:Keras/tensor flow 函数，用于从磁盘加载我们训练好的暹罗网络
*   ``list_images`` :抓取我们`examples`目录中所有图像的路径

让我们继续解析我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of testing images")
args = vars(ap.parse_args())
```

这里我们只需要一个参数，`--input`，它是我们在磁盘上的目录的路径，该目录包含我们想要比较相似性的图像。运行这个脚本时，我们将提供项目中`examples`目录的路径。

解析完命令行参数后，我们现在可以获取`--input`目录中的所有`testImagePaths`:

```py
# grab the test dataset image paths and then randomly generate a
# total of 10 image pairs
print("[INFO] loading test dataset...")
testImagePaths = list(list_images(args["input"]))
np.random.seed(42)
pairs = np.random.choice(testImagePaths, size=(10, 2))

# load the model from disk
print("[INFO] loading siamese model...")
model = load_model(config.MODEL_PATH)
```

**第 20 行**获取包含我们想要进行相似性比较的数字的所有示例图像的路径。**第 22 行**从这些`testImagePaths`中随机产生总共 10 对图像。

```py
# loop over all image pairs
for (i, (pathA, pathB)) in enumerate(pairs):
	# load both the images and convert them to grayscale
	imageA = cv2.imread(pathA, 0)
	imageB = cv2.imread(pathB, 0)

	# create a copy of both the images for visualization purpose
	origA = imageA.copy()
	origB = imageB.copy()

	# add channel a dimension to both the images
	imageA = np.expand_dims(imageA, axis=-1)
	imageB = np.expand_dims(imageB, axis=-1)

	# add a batch dimension to both images
	imageA = np.expand_dims(imageA, axis=0)
	imageB = np.expand_dims(imageB, axis=0)

	# scale the pixel values to the range of [0, 1]
	imageA = imageA / 255.0
	imageB = imageB / 255.0

	# use our siamese model to make predictions on the image pair,
	# indicating whether or not the images belong to the same class
	preds = model.predict([imageA, imageB])
	proba = preds[0][0]
```

**第 29 行**开始循环所有图像对。对于每个图像对，我们:

*   从磁盘加载两个图像(**行 31 和 32** )
*   克隆两个图像，以便我们稍后可以绘制/可视化它们(**第 35 行和第 36 行**)
*   添加一个通道尺寸(**行 39 和 40** )以及一个批次尺寸(**行 43 和 44** )
*   将像素亮度从范围*【0，255】*缩放到*【0，1】*，就像我们上周训练[暹罗网络](https://pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/) ( **第 47 行和第 48 行**)

一旦`imageA`和`imageB`被预处理，我们通过调用我们的暹罗网络模型上的`.predict`方法(**第 52 行**)来比较它们的相似性，从而得到两幅图像的概率/相似性分数(**第 53 行**)。

最后一步是在屏幕上显示图像对和相应的相似性得分:

```py
	# initialize the figure
	fig = plt.figure("Pair #{}".format(i + 1), figsize=(4, 2))
	plt.suptitle("Similarity: {:.2f}".format(proba))

	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(origA, cmap=plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(origB, cmap=plt.cm.gray)
	plt.axis("off")

	# show the plot
	plt.show()
```

**第 56 行和第 57 行**为该对创建一个`matplotlib`图形，并将相似性得分显示为图的标题。

**第 60-67 行**在图上画出图像对中的每一个图像，而**第 70 行**将输出显示到我们的屏幕上。

恭喜你实现了图像比较和相似性的暹罗网络！让我们在下一节看到我们努力的成果。

### **使用带有 Keras 和 TensorFlow 的暹罗网络的图像相似性结果**

我们现在准备使用我们的暹罗网络来比较图像的相似性！

在我们检查结果之前，请确保您:

1.  我已经阅读了我们之前关于训练连体网络的教程，所以你可以理解我们的连体网络模型是如何被训练和生成的
2.  使用本教程的 ***“下载”*** 部分下载源代码、预训练的暹罗网络和示例图像

从那里，打开一个终端，并执行以下命令:

```py
$ python test_siamese_network.py --input examples
[INFO] loading test dataset...
[INFO] loading siamese model...
```

***注:**你得到的是与`TypeError: ('Keyword argument not understood:', 'groups')`相关的错误吗？如果是这样，请记住本教程“下载”部分中包含的**预训练**模型是使用 **TensorFlow 2.3 训练的。**因此，在运行`test_siamese_network.py`时，您应该使用 TensorFlow 2.3。**如果您更喜欢使用 TensorFlow 的不同版本，只需运行** `train_siamese_network.py` **来训练模型并生成一个新的** `siamese_model` **序列化到磁盘。从那里你就可以无误地运行`test_siamese_network.py`。***

**上面的图 4** 显示了我们的图像相似性结果的蒙太奇。

对于第一个图像对，一个包含一个*“7”*，而另一个包含一个*“1”*——显然这不是同一个图像，相似度得分较低，为 42%。**我们的暹罗网络已经*正确地*将这些图像标记为属于*不同的*类。**

下一个图像对由两个*“0”*数字组成。我们的暹罗网络已经预测到*非常高的*相似性分数为 97%，**表明这两幅图像属于*相同的*类。**

您可以在图 4 的**中看到所有其他图像对的相同模式。具有高相似性分数的图像属于*相同*类，而具有低相似性分数的图像对属于*不同*类。**

由于我们使用 sigmoid 激活层作为我们的暹罗网络(其输出值在范围*【0，1】*)中的最后一层，一个好的经验法则是使用相似性截止值`0.5` (50%)作为阈值:

*   如果两个图像对的图像相似度为 *< = 0.5* ，那么它们属于*不同的*类。
*   相反，如果对的预测相似度为 *> 0.5* ，那么它们属于*相同的*类。

当使用暹罗网络计算图像相似性时，您可以在自己的项目中使用这条经验法则。

## **总结**

在本教程中，你学习了如何比较两幅图像的相似性，更具体地说，它们是属于同一个类别还是属于不同的类别。我们使用 siamese 网络以及 Keras 和 TensorFlow 深度学习库完成了这项任务。

这篇文章是我们介绍暹罗网络的三部分系列的最后一部分。为了便于参考，下面是该系列中每个指南的链接:

*   **第一部分: *[用 Python 为连体网络构建图像对](https://pyimagesearch.com/2020/11/23/building-image-pairs-for-siamese-networks-with-python/)***
*   **Part #2: *[用 Keras、TensorFlow、深度学习训练暹罗网络](https://pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/)***
*   **第三部分:** *使用暹罗网络、Keras 和 TensorFlow 比较图像的相似性*(本教程)

在不久的将来，我将报道更多关于暹罗网络的高级系列，包括:

*   图像三元组
*   对比损失
*   三重损失
*   用暹罗网络进行人脸识别
*   用连体网络进行一次性学习

敬请关注这些教程；你不想错过他们！

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***