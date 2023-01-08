# OCR:使用 OpenCV、Keras 和 TensorFlow 进行手写识别

> 原文：<https://pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow/>

在本教程中，您将学习如何使用 OpenCV、Keras 和 TensorFlow 执行 OCR 手写识别。

这篇文章是我们关于 Keras 和 TensorFlow 的光学字符识别的两部分系列文章的第二部分:

*   **Part 1:** *[用 Keras 和 TensorFlow](https://pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/)* 训练 OCR 模型(上周的帖子)
*   **第二部分:** *用 Keras 和 TensorFlow 进行基本的手写识别*(今天的帖子)

正如你将在下面看到的，**手写识别比使用特定字体/字符的传统 OCR 更难*。***

 *这个概念如此具有挑战性的原因是，与计算机字体不同，手写风格几乎有无限种变化。我们每个人都有自己独特的风格，这是 T2 特有的。

例如，我妻子的书法令人惊叹。她的笔迹不仅清晰可辨，而且风格独特，你会认为是专业书法家写的:

另一方面，我……我的笔迹看起来像有人用一只精神错乱的松鼠骗过了一个医生:

几乎无法辨认。我经常被那些至少看了我的笔迹的人问 2-3 个澄清问题，关于一个特定的单词或短语是什么。而且不止一次，我不得不承认我也看不懂它们。

说说尴尬！真的，他们让我离开小学真是个奇迹。

手写风格的这些变化给光学字符识别引擎带来了相当大的问题，这些引擎通常是在计算机字体上训练的，*而不是*手写字体。

更糟糕的是，字母可以相互“连接”和“接触”，这使得手写识别变得更加复杂，使得 OCR 算法难以区分它们，最终导致不正确的 OCR 结果。

手写识别可以说是 OCR 的“圣杯”。我们还没有到那一步，但是在深度学习的帮助下，我们正在取得巨大的进步。

今天的教程将作为手写识别的介绍。您将看到手写识别运行良好的示例，以及无法正确 OCR 手写字符的其他示例。我真的认为您会发现阅读本手写识别指南的其余部分很有价值。

**要了解如何使用 OpenCV、Keras 和 TensorFlow 进行手写识别，*继续阅读。***

## **OCR:使用 OpenCV、Keras 和 TensorFlow 进行手写识别**

在本教程的第一部分，我们将讨论手写识别以及它与“传统的”OCR 有何不同。

然后，我将简要回顾使用 Keras 和 TensorFlow 训练我们的识别模型的过程——在本教程中，我们将使用这个训练好的模型来 OCR 手写。

***注意:**如果你还没有阅读[上周的帖子](https://pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/)，我强烈建议你现在就阅读，然后再继续，因为这篇帖子概述了我们训练的用于 OCR 字母数字样本的模型。作为本教程的先决条件，你应该对上周的概念和脚本有一个牢固的理解。*

我们将回顾我们的项目结构，然后实现一个 Python 脚本来使用 OpenCV、Keras 和 TensorFlow 执行手写识别。

为了总结今天的 OCR 教程，我们将讨论我们的手写识别结果，包括哪些有效，哪些无效。

### **什么是手写识别？手写识别*和传统的 OCR*有什么不同？**

传统的 OCR 算法和技术假设我们使用某种固定的字体。在 20 世纪早期，这可能是微缩胶片使用的字体。

在 20 世纪 70 年代，专门为 OCR 算法开发了专门的字体*,从而使它们更加精确。*

到了 2000 年，我们可以使用电脑上预装的字体来自动生成训练数据，并使用这些字体来训练我们的 OCR 模型。

这些字体都有一些共同点:

1.  它们是以某种方式设计的。
2.  每个字符之间有一个*可预测的*和*假定的*空格(从而使分割更容易)。
3.  字体的风格更有利于 OCR。

从本质上来说，工程/计算机生成的字体使得 OCR *更加容易。*

不过，手写识别是一个完全不同的领域。考虑到变化的极端数量和字符如何经常重叠。每个人都有自己独特的写作风格。

字符可以被拉长、俯冲、倾斜、风格化、挤压、连接、微小、巨大等。(并以这些组合中的任何一种出现)。

数字化手写识别非常具有挑战性，而且离解决问题还很远，但是深度学习正在帮助我们提高手写识别的准确性。

### **手写识别——我们迄今为止所做的工作**

在[上周的教程](https://pyimagesearch.com/2020/08/17/ocr-with-keras-tensorflow-and-deep-learning/)中，我们使用 Keras 和 TensorFlow 训练了一个深度神经网络，既能识别 ***数字*** ( *0-9* )和 ***字母字符*** ( *A-Z* )。

为了训练我们的网络来识别这些字符集，我们利用了 MNIST 数字数据集以及 **NIST 特殊数据库 19** (用于 *A-Z* 字符)。

我们的模型在手写识别测试集上获得了 96%的准确率。

今天，我们将学习如何在我们自己的自定义图像中使用该模型进行手写识别。

### **配置您的 OCR 开发环境**

如果您尚未配置 TensorFlow 和上周教程中的相关库，我首先建议您遵循下面的相关教程:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

上面的教程将帮助您在一个方便的 Python 虚拟环境中，用这篇博客文章所需的所有软件来配置您的系统。

### **项目结构**

如果你还没有，去这篇博文的 ***“下载”*** 部分，获取今天教程的代码和数据集。

在里面，您会发现以下内容:

```py
$ tree --dirsfirst --filelimit 10
.
└── ocr-handwriting-recognition
    ├── images
    │   ├── hello_world.png
    │   ├── umbc_address.png
    │   └── umbc_zipcode.png
    ├── pyimagesearch
    │   ├── az_dataset
    │   │   ├── __init__.py
    │   │   └── helpers.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   └── resnet.py
    │   └── __init__.py
    ├── a_z_handwritten_data.csv
    ├── handwriting.model
    ├── ocr_handwriting.py
    ├── plot.png
    └── train_ocr_model.py

5 directories, 13 files
```

*   ``pyimagesearch`` 模块:
    *   包括用于 I/O 助手功能的子模块`az_dataset`和用于实现 ResNet 深度学习模型的子模块`models`
*   ``a_z_handwritten_data.csv`` :包含 Kaggle A-Z 数据集的 CSV 文件
*   ``train_ocr_model.py`` :上周的主 Python 驱动文件，我们用它来训练 ResNet 模型并显示我们的结果。我们的模型和训练图文件包括:
    *   我们在上周的教程中创建的自定义 OCR ResNet 模型
    *   ``plot.png`` :我们最近 OCR 训练运行的结果图
*   ``images/`` 子目录:包含三个 PNG 测试文件，供我们使用 Python 驱动程序脚本进行 OCR
*   ``ocr_handwriting.py`` :本周的主要 Python 脚本，我们将使用它来 OCR 我们的手写样本

### **使用 OpenCV、Keras 和 TensorFlow 实现我们的手写识别 OCR 脚本**

让我们打开`ocr_handwriting.py`并查看它，从导入和[命令行参数](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/)开始:

```py
# import the necessary packages
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained handwriting recognition model")
args = vars(ap.parse_args())
```

接下来，我们将加载我们在上周的教程中开发的自定义手写 OCR 模型:

```py
# load the handwriting OCR model
print("[INFO] loading handwriting OCR model...")
model = load_model(args["model"])
```

Keras 和 TensorFlow 的`load_model`实用程序使加载我们的序列化手写识别模型变得非常简单(**第 19 行**)。回想一下，我们的 OCR 模型使用 ResNet 深度学习架构来分类对应于数字 *0-9* 或字母 *A-Z* 的每个字符。

***注:**关于 ResNet CNN 架构的更多细节，请参考使用 Python 的[计算机视觉深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)从业者捆绑包。*

因为我们已经从磁盘加载了模型，所以让我们抓取图像，对其进行预处理，并找到角色轮廓:

```py
# load the input image from disk, convert it to grayscale, and blur
# it to reduce noise
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# perform edge detection, find contours in the edge map, and sort the
# resulting contours from left-to-right
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []
```

加载图像(**第 23 行**)后，我们将其转换为灰度(**第 24 行**)，然后应用高斯模糊降低噪点(**第 25 行**)。

我们的下一步将涉及一个大的轮廓处理循环。让我们更详细地分析一下，以便更容易理解:

```py
# loop over the contours
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)

	# filter out bounding boxes, ensuring they are neither too small
	# nor too large
	if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
		# extract the character and threshold it to make the character
		# appear as *white* (foreground) on a *black* background, then
		# grab the width and height of the thresholded image
		roi = gray[y:y + h, x:x + w]
		thresh = cv2.threshold(roi, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		(tH, tW) = thresh.shape

		# if the width is greater than the height, resize along the
		# width dimension
		if tW > tH:
			thresh = imutils.resize(thresh, width=32)

		# otherwise, resize along the height
		else:
			thresh = imutils.resize(thresh, height=32)
```

从第 40 行的**开始，我们循环每个轮廓并执行一系列的**四步:****

**步骤 1:** 选择适当大小的轮廓并提取它们:

*   **第 42 行**计算轮廓的边界框。
*   接下来，我们确保这些边界框的大小是合理的，并过滤掉那些过大或过小的边界框。
*   对于每个满足我们的尺寸标准的边界框，我们提取与字符(**行 50** )相关联的感兴趣区域(`roi`)。

**步骤 2:** 使用阈值算法清理图像，目标是在黑色背景上有白色字符:

*   对`roi` ( **行 51 和 52** )应用 Otsu 的二进制阈值法。这导致了由黑色背景上的白色字符组成的二进制图像。

**第三步:**将每个字符的大小调整为带边框的 *32×32* 像素图像:

*   根据宽度是否大于高度或者高度是否大于宽度，我们相应地调整阈值字符 ROI 的大小(**第 57-62 行**)。

但是等等！在我们继续从**第 40 行**开始的循环之前，我们需要填充这些 ROI 并将其添加到`chars`列表中:

```py
		# re-grab the image dimensions (now that its been resized)
		# and then determine how much we need to pad the width and
		# height such that our image will be 32x32
		(tH, tW) = thresh.shape
		dX = int(max(0, 32 - tW) / 2.0)
		dY = int(max(0, 32 - tH) / 2.0)

		# pad the image and force 32x32 dimensions
		padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
			left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
			value=(0, 0, 0))
		padded = cv2.resize(padded, (32, 32))

		# prepare the padded image for classification via our
		# handwriting OCR model
		padded = padded.astype("float32") / 255.0
		padded = np.expand_dims(padded, axis=-1)

		# update our list of characters that will be OCR'd
		chars.append((padded, (x, y, w, h)))
```

完成提取和准备的字符集后，我们可以**执行 OCR:**

```py
# extract the bounding box locations and padded characters
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")

# OCR the characters using our handwriting recognition model
preds = model.predict(chars)

# define the list of label names
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]
```

我们快完成了！是时候看看我们的劳动成果了。为了查看我们的手写识别结果是否符合我们的预期，让我们将它们可视化并显示出来:

```py
# loop over the predictions and bounding box locations together
for (pred, (x, y, w, h)) in zip(preds, boxes):
	# find the index of the label with the largest corresponding
	# probability, then extract the probability and label
	i = np.argmax(pred)
	prob = pred[i]
	label = labelNames[i]

	# draw the prediction on the image
	print("[INFO] {} - {:.2f}%".format(label, prob * 100))
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.putText(image, label, (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
```

总结起来，我们循环每个预测和相应的边界框(**行 98** )。

***注意:**如果你是使用 pip 安装方法安装 OpenCV 4.3.0 的 Ubuntu 用户，有一个 bug 会阻止我们使用`cv2.imshow`正确显示结果。解决方法是简单地在尺寸过小的显示框中单击鼠标，然后按下`q`键，重复几个周期，直到显示放大到合适的尺寸。*

恭喜你！您已经完成了对输入图像执行 OCR 的主要 Python 驱动程序文件。

让我们看看我们的结果。

### **手写识别 OCR 结果**

首先使用本教程的 ***“下载”*** 部分下载源代码、预训练的手写识别模型和示例图像。

打开终端并执行以下命令:

```py
$ python ocr_handwriting.py --model handwriting.model --image images/hello_world.png
[INFO] loading handwriting OCR model...
[INFO] H - 92.48%
[INFO] W - 54.50%
[INFO] E - 94.93%
[INFO] L - 97.58%
[INFO] 2 - 65.73%
[INFO] L - 96.56%
[INFO] R - 97.31%
[INFO] 0 - 37.92%
[INFO] L - 97.13%
[INFO] D - 97.83%
```

在这个例子中，我们试图对手写文本*“Hello World”进行 OCR。*

我们的手写识别模型在这里表现很好，但是犯了两个错误。

首先，它混淆了字母*【O】*和数字*【0】*(零)——这是一个可以理解的错误。

其次，也是更令人担忧的一点，手写识别模型混淆了“世界”中的*“O”*和*“2”。*

下一个例子包含我的母校的手写姓名和邮政编码，这所学校位于 UMBC 巴尔的摩县的马里兰大学:

```py
$ python ocr_handwriting.py --model handwriting.model --image images/umbc_zipcode.png 
[INFO] loading handwriting OCR model...
[INFO] U - 34.76%
[INFO] 2 - 97.88%
[INFO] M - 75.04%
[INFO] 7 - 51.22%
[INFO] B - 98.63%
[INFO] 2 - 99.35%
[INFO] C - 63.28%
[INFO] 5 - 66.17%
[INFO] 0 - 66.34%
```

我们的手写识别算法在这里几乎完美地执行了*。我们能够正确地识别*“UMBC”中的每一个手写字符；*然而，邮政编码被错误地识别了——我们的模型混淆了*“1”*和*“7”*。*

 *如果我们将去倾斜应用到我们的角色数据，我们也许能够改进我们的结果。

让我们检查最后一个例子。此图片包含 UMBC 的完整地址:

```py
$ python ocr_handwriting.py --model handwriting.model --image images/umbc_address.png 
[INFO] loading handwriting OCR model...
[INFO] B - 97.71%
[INFO] 1 - 95.41%
[INFO] 0 - 89.55%
[INFO] A - 87.94%
[INFO] L - 96.30%
[INFO] 0 - 71.02%
[INFO] 7 - 42.04%
[INFO] 2 - 27.84%
[INFO] 0 - 67.76%
[INFO] Q - 28.67%
[INFO] Q - 39.30%
[INFO] H - 86.53%
[INFO] Z - 61.18%
[INFO] R - 87.26%
[INFO] L - 91.07%
[INFO] E - 98.18%
[INFO] L - 84.20%
[INFO] 7 - 74.81%
[INFO] M - 74.32%
[INFO] U - 68.94%
[INFO] D - 92.87%
[INFO] P - 57.57%
[INFO] 2 - 99.66%
[INFO] C - 35.15%
[INFO] I - 67.39%
[INFO] 1 - 90.56%
[INFO] R - 65.40%
[INFO] 2 - 99.60%
[INFO] S - 42.27%
[INFO] O - 43.73%
```

这就是我们的手写识别模型 ***真正的*纠结的地方**。如你所见，在单词*、【山顶】、*、*、【巴尔的摩】、*和邮政编码中有多个错误。

鉴于我们的手写识别模型在训练和测试期间表现如此之好，难道我们不应该期待它在我们自己的自定义图像上也表现良好吗？

要回答这个问题，让我们进入下一部分。

### **局限性、缺点和下一步措施**

虽然我们的手写识别模型在我们的测试集上获得了 **96%的准确率，但我们在自己定制的图像上的手写识别准确率略低于此。**

最大的问题之一是，我们使用 MNIST(数字)和 NIST(字母字符)数据集的变体来训练我们的手写识别模型。

这些数据集虽然研究起来很有趣，但不一定会转化为现实世界的项目，因为图像已经为我们进行了预处理和清理— ***现实世界的角色并没有那么“干净”***

此外，我们的手写识别方法要求字符被单独分割。

对于某些字符来说，这可能是可能的，但是我们中的许多人(尤其是草书作者)在快速书写时会将字符联系起来。这使得我们的模型误以为一组*人物*实际上是一个*单个人物*，最终导致错误的结果。

最后，我们的模型架构有点过于简单。

虽然我们的手写识别模型在训练和测试集上表现良好，但该架构(结合训练数据集本身)不够健壮，不足以概括为“现成的”手写识别模型。

为了提高我们手写识别的准确性，我们应该研究长短期记忆网络(LSTMs)的进展，它可以自然地处理相连的字符。

我们将在未来关于 PyImageSearch 的教程中介绍如何使用 LSTMs，以及在我们即将出版的 OpenCV、Tesseract 和 Python 书的 [*OCR 中。*](https://pyimagesearch.com/ocr-with-opencv-tesseract-and-python/)

## 新书:OpenCV、Tesseract 和 Python 的 OCR

**光学字符识别(OCR)是一个简单的概念，但在实践中很难:**创建一个接受输入图像的软件，让该软件*自动识别图像中的文本*，然后将其转换为机器编码的文本(即“字符串”数据类型)。

尽管 OCR 是一个如此直观的概念，但它却难以置信的困难(T2)。计算机视觉领域已经存在了 50 多年(机械 OCR 机器可以追溯到 100 多年前)，但我们*仍然*没有“解决”OCR，创造出一个现成的 OCR 系统，几乎可以在任何情况下工作。

更糟糕的是，试图编写能够执行 OCR 的定制软件更加困难:

*   如果你是 OCR 领域的新手，像 Tesseract 这样的开源 OCR 包可能很难使用。
*   使用 Tesseract 获得高精度通常需要您知道要使用哪些选项、参数和配置— **不幸的是，没有多少高质量的 Tesseract 教程或在线书籍。**
*   OpenCV 和 scikit-image 等计算机视觉和图像处理库可以帮助您对图像进行预处理，以提高 OCR 的准确性……但是您使用的是哪些算法和技术呢？
*   深度学习几乎在计算机科学的每个领域都带来了前所未有的准确性。**OCR 应该使用哪些深度学习模型、图层类型和损失函数？**

如果你曾经发现自己很难将 OCR 应用到一个项目中，或者如果你只是对学习 OCR 感兴趣，我的新书， *[光学字符识别(OCR)、OpenCV 和宇宙魔方](https://pyimagesearch.com/ocr-with-opencv-tesseract-and-python/)* 就是为你准备的。

不管你目前在计算机视觉和 OCR 方面的经验水平如何，读完这本书后，你将拥有解决你自己的 OCR 项目所必需的知识。

如果您对 OCR 感兴趣，已经有了 OCR 项目的想法，或者您的公司需要它，**请点击下面的按钮预订您的副本**:

## **总结**

在本教程中，您学习了如何使用 Keras、TensorFlow 和 OpenCV 执行 OCR 手写识别。

我们的手写识别系统利用基本的计算机视觉和图像处理算法(边缘检测、轮廓和轮廓过滤)从输入图像中分割出字符。

从那里，我们通过我们训练的手写识别模型来识别每个字符。

我们的手写识别模型表现很好，但在某些情况下，结果还可以改进(理想情况下，使用更多的训练数据，这些数据代表我们希望识别的手写内容的*)——**训练数据的质量越高，我们的手写识别模型就越准确！***

其次，我们的手写识别管道没有处理字符可能被*连接*的情况，从而导致多个连接的字符被视为一个*单个*字符，从而混淆了我们的 OCR 模型。

处理相连的手写字符仍然是计算机视觉和 OCR 领域中一个开放的研究领域；然而，深度学习模型，特别是 LSTMs，已经显示出*在提高手写识别准确性方面的重大承诺*。

我将在以后的教程中介绍使用 LSTMs 进行更高级的手写识别。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****