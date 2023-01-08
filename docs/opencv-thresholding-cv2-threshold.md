# OpenCV 阈值(cv2.threshold)

> 原文：<https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/>

在本教程中，您将学习如何使用 OpenCV 和`cv2.threshold`函数来应用基本阈值和 Otsu 阈值。

阈值分割是[计算机视觉](https://pyimagesearch.com/)中最常见(也是最基本)的分割技术之一，它允许我们将图像的*前景*(即我们感兴趣的物体)与*背景*分开。

阈值有三种形式:

1.  我们有**简单的阈值处理**，我们手动提供参数来分割图像——这在受控的照明条件下非常有效，我们可以确保图像的前景和背景之间的高对比度。
2.  我们也有一些方法，比如试图更加动态化的 **Otsu 阈值法**和基于输入图像自动计算最佳阈值的*。*
3.  *最后，我们有**自适应阈值**，它不是试图使用单个值对图像*全局*进行阈值处理，而是将图像分成更小的块，并分别对这些块*和*分别进行阈值处理*和*。*

 *今天我们将讨论简单阈值和 Otsu 阈值。我们的下一个教程将详细介绍自适应阈值。

**要了解如何使用 [OpenCV](https://opencv.org/) 和`cv2.threshold`函数应用基本阈值和 Otsu 阈值，*请继续阅读。***

## **OpenCV 阈值处理(cv2.threshold )**

在本教程的第一部分，我们将讨论阈值的概念，以及阈值如何帮助我们使用 OpenCV 分割图像。

从那里，我们将配置我们的开发环境，并审查我们的项目目录结构。

然后，我将向您展示使用 OpenCV 对图像进行阈值处理的两种方法:

1.  **基本阈值**您必须手动提供阈值， *T*
2.  **Otsu 的阈值，**哪个*自动*决定阈值

作为一名计算机视觉从业者，理解这些方法是如何工作的非常重要。

让我们开始吧。

### 什么是阈值处理？

阈值处理是图像的二值化。一般来说，我们寻求将灰度图像转换成二进制图像，其中像素或者是 *0* 或者是 *255* 。

一个简单的阈值例子是选择一个阈值 *T* ，然后设置所有小于 *T* 到 *0* 的像素强度，以及所有大于 *T* 到 *255 的像素值。*这样，我们能够创建图像的二进制表示。

例如，看看下面的(灰度)PyImageSearch 徽标及其对应的阈值:

在左侧的*上，我们有原始的 PyImageSearch 徽标，它已被转换为灰度。在*右边*，我们有 PyImageSearch 徽标的阈值化二进制表示。*

为了构建这个阈值图像，我简单地设置我的阈值 *T=225。*这样，logo 中所有像素 *p* 其中 *p < T* 被设置为 *255* ，所有像素 *p > = T* 被设置为 *0* 。

通过执行这个阈值处理，我已经能够从背景中分割出 PyImageSearch 徽标。

通常，我们使用阈值来聚焦图像中特别感兴趣的对象或区域。在下一课的示例中，我们将使用阈值处理来检测图像中的硬币，分割 OpenCV 徽标的片段，并将车牌字母和字符从车牌本身中分离出来。

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

在我们使用 OpenCV 和`cv2.threshold`函数应用阈值之前，我们首先需要回顾我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

然后，您将看到以下目录结构:

```py
$ tree . --dirsfirst
.
├── images
│   ├── coins01.png
│   ├── coins02.png
│   └── opencv_logo.png
├── otsu_thresholding.py
└── simple_thresholding.py

1 directory, 5 files
```

我们今天要复习两个 Python 脚本:

1.  `simple_thresholding.py`:演示如何使用 OpenCV 应用阈值处理。在这里，我们手动*设置阈值，这样我们就可以从背景中分割出前景。*
2.  *`otsu_thresholding.py`:应用 Otsu 阈值法，阈值参数自动设置*。**

 *Otsu 阈值技术的好处是，我们不必手动设置阈值截止，Otsu 的方法会自动为我们完成。

在`images`目录中有许多演示图像，我们将对它们应用这些阈值脚本。

### **用 OpenCV 实现简单的阈值处理**

应用简单的阈值方法需要人工干预。我们必须指定一个阈值 *T* 。低于 *T* 的所有像素亮度被设置为 *255* 。并且所有大于 *T* 的像素亮度被设置为 *0* 。

我们也可以通过设置所有大于 *T* 到 *255* 的像素和所有小于 *T* 到 *0* 的像素强度来应用这种二进制化的逆过程。

让我们探索一些应用简单阈值方法的代码。打开项目目录结构中的`simple_thresholding.py`文件，插入以下代码:

```py
# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
args = vars(ap.parse_args())
```

我们从第 2 行和第 3 行开始，导入我们需要的 Python 包。然后我们在**的第 6-9 行**解析我们的命令行参数。

只需要一个命令行参数`--image`，它是我们希望应用阈值的输入图像的路径。

完成导入和命令行参数之后，让我们继续从磁盘加载映像并对其进行预处理:

```py
# load the image and display it
image = cv2.imread(args["image"])
cv2.imshow("Image", image)

# convert the image to grayscale and blur it slightly
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
```

**第 12 行和第 13 行**从磁盘加载我们的输入`image`并显示在我们的屏幕上。

然后，我们通过以下两种方式对图像进行预处理:

1.  将其转换为灰度
2.  应用 7×7 高斯模糊

应用高斯模糊有助于移除图像中我们不关心的一些高频边缘，并允许我们获得更“干净”的分割。

现在，让我们继续应用实际的阈值:

```py
# apply basic thresholding -- the first parameter is the image
# we want to threshold, the second value is is our threshold
# check; if a pixel value is greater than our threshold (in this
# case, 200), we set it to be *black, otherwise it is *white*
(T, threshInv) = cv2.threshold(blurred, 200, 255,
	cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Binary Inverse", threshInv)
```

图像模糊后，我们使用`cv2.threshold`函数计算**行 23 和 24** 上的阈值图像。此方法需要四个参数。

第一个是我们希望设定阈值的灰度图像。我们首先提供我们的`blurred`图像。

然后，我们*手动*提供我们的 *T* 阈值。我们使用一个值 *T=200* 。

我们的第三个参数是阈值处理期间应用的输出值。任何大于 *T* 的像素亮度 *p* 被设置为零，任何小于 *T* 的像素亮度 *p* 被设置为输出值:

在我们的例子中，任何大于 *200* 的像素值被设置为 *0* 。任何小于 *200* 的值被设置为 *255* 。

最后，我们必须提供一个阈值方法。我们使用`cv2.THRESH_BINARY_INV`方法，表示小于 *T* 的像素值 *p* 被设置为输出值(第三个参数)。

然后，`cv2.threshold`函数返回一个由两个值组成的元组:第一个值是阈值。在简单阈值的情况下，这个值是微不足道的，因为我们首先手动提供了 *T* 的值。但是在 Otsu 阈值的情况下，动态地为我们计算出 *T* ，有这个值就很好了。第二个返回值是阈值图像本身。

但是，如果我们想执行*反向*操作，就像这样:

如果想要将大于 *T* 的所有像素 *p* 设置为输出值呢？这可能吗？

当然啦！有两种方法可以做到。第一种方法是简单地对输出阈值图像进行位非运算。但是这增加了一行额外的代码。

相反，我们可以给`cv2.threshold`函数提供一个不同的标志:

```py
# using normal thresholding (rather than inverse thresholding)
(T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)
```

在第 28 行的**上，我们通过提供`cv2.THRESH_BINARY`来应用不同的阈值方法。**

在大多数情况下，你通常希望被分割的物体在*黑色*背景上呈现为*白色*，因此使用`cv2.THRESH_BINARY_INV`。但是如果你想让你的对象在*白色*背景上显示为*黑色*，一定要提供`cv2.THRESH_BINARY`标志。

我们要执行的最后一项任务是显示图像中的前景对象，隐藏其他所有内容。还记得我们讨论过的[图像遮罩](https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/)吗？这在这里会派上用场:

```py
# visualize only the masked regions in the image
masked = cv2.bitwise_and(image, image, mask=threshInv)
cv2.imshow("Output", masked)
cv2.waitKey(0)
```

在**第 32 行，**我们通过使用`cv2.bitwise_and`函数来执行屏蔽。我们提供我们的原始输入图像作为前两个参数，然后我们的反转阈值图像作为我们的遮罩。请记住，遮罩只考虑原始图像中遮罩大于零的像素。

### **简单阈值结果**

准备好查看使用 OpenCV 应用基本阈值的结果了吗？

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

从那里，您可以执行以下命令:

```py
$ python simple_thresholding.py --image images/coins01.png
```

在左上角的*，*是我们的原始输入图像。在右上角的*、*我们有使用逆阈值分割的图像，硬币在*黑*背景上显示为*白*。

类似地，在左下方的*上，我们翻转阈值方法，现在硬币在*白色*背景上显示为*黑色*。*

最后，*右下角的*应用我们的位与阈值遮罩，我们只剩下图像中的硬币(没有背景)。

让我们试试硬币的第二个图像:

```py
$ python simple_thresholding.py --image images/coins02.png
```

我们又一次成功地将图像的前景从背景中分割出来。

但是仔细看看，比较一下**图 5** 和**图 6** 的输出。你会注意到在**图 5** 中有一些硬币看起来有“洞”。这是因为阈值测试没有通过，因此我们不能在输出的阈值图像中包含硬币的那个区域。

然而，在**图 6** 中，你会注意到没有洞——表明分割(本质上)是完美的。

***注:*** *实事求是地说，这不是问题。真正重要的是我们能够获得硬币的轮廓。可以使用* [*形态学操作*](https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/) *或轮廓方法来填充阈值硬币遮罩内的这些小间隙。*

既然阈值处理在图 6 的**中表现完美，为什么它在图 5** 的**中表现得不那么完美呢？**

答案很简单:*光照条件。*

虽然很微妙，但这两张照片确实是在不同的光照条件下拍摄的。并且因为我们已经手动提供了阈值，所以不能保证在存在光照变化的情况下，该阈值 *T* 将从一个图像工作到下一个图像。

解决这个问题的一个方法是简单地为您想要阈值化的每个图像提供一个阈值 *T* 。但这是一个严重的问题，尤其是如果我们希望我们的系统是动态的 T2，并在各种光照条件下工作。

**解决方案是使用 Otsu 法和自适应阈值法等方法来帮助我们获得更好的结果。**

但是现在，让我们再看一个例子，在这个例子中，我们分割了 OpenCV 徽标的各个部分:

```py
$ python simple_thresholding.py --image images/opencv_logo.png
```

请注意，我们已经能够从输入图像中分割出 OpenCV 徽标的半圆以及“OpenCV”文本本身。虽然这可能看起来不太有趣，但是能够将图像分割成小块是一项非常有价值的技能。当我们深入轮廓并使用它们来量化和识别图像中的不同对象时，这将变得更加明显。

但是现在，让我们继续讨论一些更高级的阈值技术，在这些技术中，我们不必手动提供值 *T* 。

### **使用 OpenCV 实现 Otsu 阈值处理**

在前面关于简单阈值的部分中，我们需要手动*提供阈值 *T* 。对于受控照明条件下的简单图像，我们硬编码该值可能是可行的。*

 *但是在我们没有任何关于照明条件的先验知识的真实世界条件下，我们实际上使用 Otsu 的方法自动计算出最佳值 *T* 。

Otsu 的方法假设我们的图像包含两类像素:*背景*和*前景*。

**此外，Otsu 的方法假设我们的图像的像素强度的灰度直方图是** ***双峰*** **，这简单地意味着直方图是** ***两个峰值。***

例如，看看下面的处方药丸图像及其相关灰度直方图:

请注意直方图明显有两个峰值，第一个尖锐的峰值对应于图像的均匀背景色，而第二个峰值对应于药丸区域本身。

如果直方图的概念现在对你来说有点困惑，不要担心——我们将在我们的[图像直方图博客文章](https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/)中更详细地介绍它们。但目前只需理解直方图是一个简单的列表或一个像素值在图像中出现次数的“计数器”。

基于灰度直方图，Otsu 的方法然后计算最佳阈值 *T* ，使得背景和前景峰值之间的差异最小。

然而，Otsu 的方法不知道哪些像素属于前景，哪些像素属于背景，它只是试图最佳地分离直方图的峰值。

让我们来看看执行 Otsu 阈值处理的一些代码。打开项目目录结构中的`otsu_thresholding.py`文件，插入以下代码:

```py
# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
args = vars(ap.parse_args())
```

**第 2 行和第 3 行**导入我们需要的 Python 包，而**第 6-9 行**解析我们的命令行参数。

这里我们只需要一个开关`--image`，它是我们希望应用 Otsu 阈值的输入图像的路径。

我们现在可以加载和预处理我们的图像:

```py
# load the image and display it
image = cv2.imread(args["image"])
cv2.imshow("Image", image)

# convert the image to grayscale and blur it slightly
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
```

**第 12 行和第 13 行**从磁盘加载我们的图像并显示在我们的屏幕上。

然后，我们应用预处理，将图像转换为灰度，并对其进行模糊处理，以减少高频噪声。

现在让我们应用 Otsu 的阈值算法:

```py
# apply Otsu's automatic thresholding which automatically determines
# the best threshold value
(T, threshInv) = cv2.threshold(blurred, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Threshold", threshInv)
print("[INFO] otsu's thresholding value: {}".format(T))

# visualize only the masked regions in the image
masked = cv2.bitwise_and(image, image, mask=threshInv)
cv2.imshow("Output", masked)
cv2.waitKey(0)
```

应用 Otsu 的方法是在第 21 和 22 行的**上处理的，再次使用 OpenCV 的`cv2.threshold`函数。**

我们首先传递我们想要阈值化的(模糊的)图像。但是看看第二个参数——这应该是我们的阈值 *T* 。

那么我们为什么要把它设置为零呢？

**记住大津的方法是要** ***自动*** **为我们计算出** ***T*** **的最优值！从技术上讲，我们可以为这个参数指定我们想要的任何值；然而，我喜欢提供一个值`0`作为一种“无关紧要”的参数。**

第三个参数是阈值的输出值，前提是给定的像素通过了阈值测试。

最后一个论点是我们需要*特别注意的。之前，我们根据想要执行的阈值类型提供了值`cv2.THRESH_BINARY`或`cv2.THRESH_BINARY_INV`。*

但是现在我们传入了第二个标志，它与前面的方法进行了逻辑“或”运算。注意这个方法是`cv2.THRESH_OTSU`，明显对应大津的阈值法。

`cv2.threshold`函数将再次为我们返回 2 个值的元组:阈值 *T* 和阈值图像本身。

在前面的部分中，返回的值 *T* 是多余的和不相关的——我们已经知道了这个值 *T* ,因为我们必须手动提供它。

但是现在我们使用 Otsu 的方法进行自动阈值处理，这个值 *T* 变得很有趣——我们不知道 *T* 的最佳值是多少，因此我们使用 Otsu 的方法来计算它。**第 24 行**打印出由 Otsu 方法确定的 *T* 的值。

最后，我们在屏幕的第 28 和 29 行显示输出的阈值图像。

### **Otsu 阈值结果**

要查看 Otsu 方法的运行，请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

从那里，您可以执行以下命令:

```py
$ python otsu_thresholding.py --image images/coins01.png
[INFO] otsu's thresholding value: 191.0
```

很不错，对吧？我们甚至不需要提供我们的值*T*——Otsu 的方法自动为我们处理这个问题。我们仍然得到了一个很好的阈值图像作为输出。如果我们检查我们的终端，我们会看到 Otsu 的方法计算出了一个值 *T=191* :

所以基于我们的输入图像， *T* 的最优值是`191`；因此，任何大于`191`的像素 *p* 被设置为`0`，任何小于`191`的像素被设置为`255`(因为我们在上面的*“用 OpenCV 实现简单阈值处理”*部分提供了详细的`cv2.THRESH_BINARY_INV`标志)。

在我们继续下一个例子之前，让我们花一点时间来讨论术语“最优”的含义由 Otsu 的方法**返回的 *T* 的值在我们的图像的视觉研究中可能不是最优的**——我们可以清楚地看到阈值图像的硬币中的一些间隙和洞。但是这个值*是最佳的*，因为假设灰度像素值的双模式分布，它尽可能好地分割前景和背景**。**

如果灰度图像不遵循双模态分布，那么 Otsu 的方法仍将运行，但它可能不会给出我们想要的结果。在这种情况下，我们将不得不尝试自适应阈值，这将在我们的下一个教程。

无论如何，让我们尝试第二个图像:

```py
$ python otsu_thresholding.py --image images/coins02.png 
[INFO] otsu's thresholding value: 180.0
```

再次注意，Otsu 的方法很好地将前景和背景分离开来。而这次大津的方法已经确定了 *T* 的最优值为`180`。任何大于`180`的像素值被设置为`0`，任何小于`180`的像素值被设置为`255`(同样，假设取反阈值)。

如你所见，Otsu 的方法可以为我们节省大量猜测和检查 *T* 最佳值的时间。然而，有一些主要的缺点。

第一个是 Otsu 的方法假设输入图像的灰度像素强度为双峰分布。如果不是这样，那么 Otsu 的方法可以返回低于标准的结果。

其次，Otsu 法是一种全局阈值法。在光照条件是半稳定的情况下，我们想要分割的对象与背景有足够的对比度，我们可能能够摆脱 Otsu 的方法。

但是当光照条件不均匀时——比如当图像的不同部分比其他部分被照亮得更多时，我们会遇到一些严重的问题。在这种情况下，我们需要依靠**自适应阈值**(我们将在下周讨论)

## CV2 .阈值应用

在 cv2.threshold 的帮助下，您可以构建几个应用程序

*   图像分割应用程序，根据像素比较来分离对象。
*   背景去除，分离并去除前景和背景对象。
*   [光学字符识别](https://pyimagesearch.com/2021/08/09/what-is-optical-character-recognition-ocr/) (OCR)，通过阈值处理提高对比度和 OCR 准确度。
*   创建一个二进制图像，在这里设定阈值，并将图像转换为每个像素为 0(黑色)或 255(白色)的二进制图像。

更新:
2022 年 12 月 30 日更新教程内容和链接。

## **总结**

在这一课中，我们学习了所有关于阈值处理的知识:什么是阈值处理，我们为什么使用阈值处理，以及如何使用 OpenCV 和`cv2.threshold`函数来执行阈值处理。

我们从执行**简单阈值**开始，这需要我们手动提供一个值 *T* 来执行阈值。然而，我们很快意识到手动提供一个值 *T* 是非常繁琐的，需要我们硬编码这个值，这意味着这个方法不是在所有情况下都有效。

然后，我们继续使用 **Otsu 的阈值方法**，该方法*会自动*为我们计算 *T* 的最佳值，假设输入图像的灰度表示为双模态分布。

这里的问题是:( 1)我们的输入图像需要是双模态的，Otsu 的方法才能正确分割图像，以及(2) Otsu 的方法是一种全局阈值方法，这意味着我们至少需要对我们的照明条件进行一些适当的控制。

在我们的照明条件不理想的情况下，或者我们根本无法控制它们，我们需要**自适应阈值**(也称为局部阈值)。我们将在下一篇教程中讨论自适应阈值处理。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，** ***只需在下面的表格中输入您的电子邮件地址！*******