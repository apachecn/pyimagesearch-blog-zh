# OpenCV 形态学运算

> 原文：<https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/>

在本教程中，您将学习如何使用 OpenCV 应用形态学运算。

我们将讨论的形态学运算包括:

*   侵蚀
*   扩张
*   开始
*   关闭
*   形态梯度
*   布莱克有
*   大礼帽(也称为“白帽子”)

这些图像处理操作应用于灰度或二进制图像，并用于 OCR 算法的预处理、检测条形码、检测车牌等。

有时，巧妙使用形态学运算可以让你避免更复杂(计算成本更高)的机器学习和深度学习算法。

作为一个认真的计算机视觉从业者，你*需要*去理解形态学运算。

**要学习如何用 OpenCV 应用形态学运算，** ***继续阅读。***

## **OpenCV 形态学运算**

形态学操作是应用于二进制或灰度图像的简单变换。更具体地说，我们对图像内部的*形状*和*结构*应用形态学操作。

我们可以使用形态学操作来增加图像中物体的大小，以及减少它们的大小。我们还可以利用形态学操作来*闭合对象之间的*间隙，以及*打开*它们。

形态学操作使用**结构元素“探测”图像。**该结构元素定义了每个像素周围要检查的邻域。并且基于给定的操作和结构化元素的大小，我们能够调整我们的输出图像。

对结构化元素的这种解释可能听起来含糊不清——那是因为它确实如此。有许多不同的形态变换执行彼此“相反”的运算，正如加法是减法的“相反”,我们可以将腐蚀形态运算视为膨胀的“相反”。

如果这听起来令人困惑，不要担心——我们将回顾每种形态变换的许多例子，当你读完本教程时，你将对形态运算有一个清晰的认识。

### **为什么要学习形态学运算？**

形态学运算是我在图像处理中最喜欢涉及的话题之一。

这是为什么呢？

因为这些转变是如此强大。

我经常看到计算机视觉研究人员和开发人员试图解决一个问题，并立即投入到高级计算机视觉、机器学习和深度学习技术中。似乎一旦学会挥舞锤子，每个问题看起来都像钉子。

然而，有时使用不太先进的技术可以找到更“优雅”的解决方案。当然，这些技术可能不会漂浮在最新最先进算法的时髦词汇上，但它们可以完成工作。

例如，我曾经在 PyImageSearch 博客上写过一篇关于[检测图像中的条形码](https://pyimagesearch.com/2014/11/24/detecting-barcodes-images-python-opencv/)的文章。我没有使用任何花哨的技术。我没有使用任何机器学习。事实上，我能够检测图像中的条形码，只需要使用我们在本系列中已经讨论过的介绍性主题。

很疯狂，不是吗？

但是说真的，请注意这些转换——在您的计算机视觉职业生涯中，会有这样的时候，当您准备好解决一个问题时，却发现一个更优雅、更简单的解决方案可能已经存在了。很有可能，你会在形态学运算中找到那个优雅的解决方案。

让我们继续，从讨论使形态学操作成为可能的组件开始:*结构化元素。*

### **“结构化元素”的概念**

还记得我们关于[图像内核和卷积](https://pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/)的教程吗？

嗯，你可以(从概念上)把一个*结构化元素*想象成一种*内核*或者*遮罩*。然而，我们不是应用卷积，而是对像素执行简单的测试。

就像在图像内核中一样，对于图像中的每个像素，结构化元素从左到右和从上到下滑动。就像内核一样，结构化元素可以是任意大小的邻域。

例如，让我们看看下面中心像素红色的 4-邻域和 8-邻域:

这里，我们可以看到中心像素(即红色像素)位于邻域的中心:

*   4 邻域*(左)*将中心像素周围的区域定义为北、南、东、西的像素。
*   8-邻域*(右)*扩展了该区域，使其也包括拐角像素

这只是两个简单结构元素的例子。但我们也可以将它们做成任意的矩形或圆形结构——这完全取决于您的特定应用。

在 OpenCV 中，我们可以使用`cv2.getStructuringElement`函数或者 NumPy 本身来定义我们的结构化元素。就我个人而言，我更喜欢使用`cv2.getStructuringElement`函数，因为它给了你对返回元素更多的控制，但同样，这是个人的选择。

如果结构化元素的概念不完全清楚，那也没关系。在这节课中，我们将回顾许多例子。目前，要理解结构化元素的行为类似于内核或遮罩——但是我们不是将输入图像与我们的结构化元素进行卷积，而是只应用简单的像素测试。

现在我们对结构化元素有了基本的了解，让我们配置我们的开发环境，回顾项目目录结构，然后编写一些代码。

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

在我们开始用 OpenCV 实现形态学操作之前，让我们先回顾一下我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像:

```py
$ tree . --dirsfirst
.
├── car.png
├── morphological_hats.py
├── morphological_ops.py
├── pyimagesearch_logo.png
└── pyimagesearch_logo_noise.png

0 directories, 5 files
```

今天我们要复习两个 Python 脚本:

1.  `morphological_ops.py`:应用 OpenCV 的形态学操作，包括腐蚀、膨胀、打开、关闭和形态学渐变。
2.  `morphological_hats.py`:用 OpenCV 应用黑帽和礼帽/白帽操作。

三个。这两个脚本将使用我们的项目结构中包含的 png 图像来演示各种形态学操作。

### **侵蚀**

就像沿着河岸奔流的水侵蚀土壤一样，图像中的**侵蚀**会“侵蚀”前景对象，使其变小。简单地说，图像中靠近对象边界的像素将被丢弃，“侵蚀”掉。

腐蚀的工作原理是定义一个结构元素，然后在输入图像上从左到右和从上到下滑动这个结构元素。

只有当结构化元素内的所有像素都为 *> 0* 时，输入图像中的前景像素才会被保留**。否则，像素被设置为 *0* (即背景)。**

腐蚀对于移除图像中的小斑点或断开两个连接的对象非常有用。

我们可以通过使用`cv2.erode`函数来执行腐蚀。让我们打开一个新文件，命名为`morphological.py_ops.py`，并开始编码:

```py
# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
```

**第 2 行和第 3 行**导入`argparse`(用于命令行参数)和`cv2`(我们的 OpenCV 绑定)。

我们只有一个命令行参数需要解析，我们的输入`--image`将被腐蚀。

在本课的大多数示例中，我们将对 PyImageSearch 徽标应用形态学操作，如下所示:

正如我在本课前面提到的，我们通常(但不总是)对二进制图像应用形态学运算。正如我们将在本课稍后看到的，也有例外，特别是当使用*黑帽*和*白帽*操作符时，但目前，我们将假设我们正在处理一个二进制图像，其中背景像素是*黑*，前景像素是*白*。

让我们从磁盘加载我们的输入`--image`,然后应用一系列腐蚀:

```py
# load the image, convert it to grayscale, and display it to our
# screen
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)

# apply a series of erosions
for i in range(0, 3):
	eroded = cv2.erode(gray.copy(), None, iterations=i + 1)
	cv2.imshow("Eroded {} times".format(i + 1), eroded)
	cv2.waitKey(0)
```

**第 13 行**从磁盘加载我们的输入`image`，而**第 14 行**将其转换为灰度。由于我们的图像已经预先分割，我们现在正在处理一个二进制图像。

给定我们的标志图像，我们在线 **18-21** 上应用一系列腐蚀。`for`循环控制我们将要应用侵蚀的次数，或者说*迭代*。随着侵蚀次数的增加，前景 logo 会开始“侵蚀”消失。

我们通过调用`cv2.erode`函数在**第 19 行**上执行实际的侵蚀。这个函数有两个必需参数和第三个可选参数。

第一个参数是我们想要侵蚀的`image`——在本例中，它是我们的二进制图像(即 PyImageSearch 徽标)。

`cv2.erode`的第二个参数是结构化元素。如果这个值是`None`，那么将使用一个 *3×3* 结构元素，与我们上面看到的 8 邻域结构元素相同。当然，你也可以在这里提供你自己的定制结构元素来代替`None`。

最后一个参数是将要进行侵蚀的`iterations`的数量。随着迭代次数的增加，我们会看到越来越多的 PyImageSearch 徽标被蚕食。

最后，**第 20 行和第 21 行**向我们展示了我们被侵蚀的图像。

当您执行这个脚本时，您将看到我们侵蚀操作的以下输出:

在最顶端，我们有自己的原始图像。然后在图像下面，我们看到徽标分别被腐蚀了 1、2 和 3 次。**注意随着侵蚀迭代次数的增加，越来越多的徽标被侵蚀掉。**

同样，腐蚀对于从图像中移除小斑点或断开两个连接的组件最有用。记住这一点，看看 PyImageSearch 标志中的字母“p”。注意“p”的圆形区域是如何在两次腐蚀后从主干上断开的——这是一个断开图像两个相连部分的例子。

### **膨胀**

侵蚀的对立面是膨胀。就像腐蚀会侵蚀前景像素一样，膨胀会*增长*前景像素。

扩张*增加*前景对象的大小，对于将图像的破碎部分连接在一起特别有用。

就像腐蚀一样，膨胀也利用结构元素——如果结构元素中的任何*像素为 *> 0* ，则结构元素的中心像素 *p* 被设置为*白色*。*

 *我们使用`cv2.dilate`函数来应用扩展:

```py
# close all windows to cleanup the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)

# apply a series of dilations
for i in range(0, 3):
	dilated = cv2.dilate(gray.copy(), None, iterations=i + 1)
	cv2.imshow("Dilated {} times".format(i + 1), dilated)
	cv2.waitKey(0)
```

**第 24 行和第 25 行**简单地关闭所有打开的窗口，显示我们的原始图像，给我们一个新的开始。

然后第 28 行开始循环迭代次数，就像我们对 cv2.erode 函数所做的一样。

通过调用`cv2.dilate`函数在**行 29** 上执行实际的膨胀，其中实际的函数签名与`cv2.erode`的签名相同。

第一个参数是我们要扩张的`image`；第二个是我们的结构元素，当设置为`None`时，它是一个 *3×3* 8 邻域结构元素；最后一个参数是我们将要应用的膨胀数`iterations`。

我们的膨胀的输出可以在下面看到:

同样，在顶部*我们有我们的原始输入图像。在输入图像下方，我们的图像分别放大了 1、2 和 3 倍。*

与前景区域被慢慢侵蚀的侵蚀不同，膨胀实际上*增长了*我们的前景区域。

当连接一个物体的断裂部分时，膨胀特别有用——例如，看看底部的*图像，我们已经应用了 3 次迭代的膨胀。至此，**和*之间的空隙被全部的*和**字母连接起来。*

### **开启**

**一开口就是一个*****接着是一个** ***膨胀*** **。***

 *执行打开操作允许我们从图像中移除小斑点:首先应用腐蚀来移除小斑点，然后应用膨胀来重新生长原始对象的大小。

让我们看一些将开口应用于图像的示例代码:

```py
# close all windows to cleanup the screen, then initialize a list of
# of kernels sizes that will be applied to the image
cv2.destroyAllWindows()
cv2.imshow("Original", image)
kernelSizes = [(3, 3), (5, 5), (7, 7)]

# loop over the kernels sizes
for kernelSize in kernelSizes:
	# construct a rectangular kernel from the current size and then
	# apply an "opening" operation
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
	cv2.imshow("Opening: ({}, {})".format(
		kernelSize[0], kernelSize[1]), opening)
	cv2.waitKey(0)
```

**第 35 行和第 36 行**通过关闭所有打开的窗口并重新显示我们的原始图像来执行清理。

看看我们在第 37 行，`kernelSizes`定义的新变量。这个变量分别定义了我们将要应用的结构化元素的宽度和高度。

我们在**第 40 行**上循环这些`kernelSizes`，然后调用**第 43 行**上的`cv2.getStructuringElement`来构建我们的结构化元素。

`cv2.getStructuringElement`函数需要两个参数:第一个是我们想要的结构化元素的类型，第二个是结构化元素的大小(我们从第 40 行的**for 循环中获取)。**

我们传入一个值`cv2.MORPH_RECT`来表示我们想要一个矩形结构元素。但是你也可以传入一个值`cv2.MORPH_CROSS`来得到一个十字形结构元素(十字形就像一个 4 邻域结构元素，但是可以是任何大小)，或者传入一个`cv2.MORPH_ELLIPSE`来得到一个圆形结构元素。

具体使用哪种结构化元素取决于您的应用程序——我将把它作为一个练习留给读者，让他们来体验每一种结构化元素。

通过调用`cv2.morphologyEx`函数，在**线 42** 上执行实际的打开操作。这个函数在某种意义上是抽象的——它允许我们传递我们想要的任何形态学操作，后面是我们的内核/结构化元素。

`cv2.morphologyEx`的第一个必需参数是我们想要应用形态学操作的图像。第二个参数是形态学运算的实际*类型*——在本例中，它是一个*开*运算。最后一个必需的参数是我们正在使用的内核/结构化元素。

最后，**第 45-47 行**显示应用我们的打开的输出。

正如我上面提到的，打开操作允许我们移除图像中的小斑点。我继续给 PyImageSearch 徽标添加了一些 blobs(在我们的项目目录结构中的`pyimagesearch_logo_noise.png`):

当您将我们的开形态学操作应用于这个噪声图像时，您将收到以下输出:

请注意，当我们使用大小为 *5×5* 的内核时，小的随机斑点几乎完全消失了。当它到达一个大小为 *7×7* 的内核时，我们的打开操作不仅移除了所有的随机斑点，还在字母“p”和字母“a”上“打开”了洞。

### **关闭**

**与打开正好相反的是** ***关闭*** **。一个闭合是一个** ***膨胀*** **后跟一个** ***侵蚀*** **。**

顾名思义，闭合用于闭合对象内部的孔或者将组件连接在一起。

下面的代码块包含执行结束的代码:

```py
# close all windows to cleanup the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)

# loop over the kernels sizes again
for kernelSize in kernelSizes:
	# construct a rectangular kernel form the current size, but this
	# time apply a "closing" operation
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
	cv2.imshow("Closing: ({}, {})".format(
		kernelSize[0], kernelSize[1]), closing)
	cv2.waitKey(0)
```

执行结束操作也是通过调用`cv2.morphologyEx`来完成的，但是这次我们将通过指定`cv2.MORPH_CLOSE`标志来表明我们的形态学操作是一个结束操作。

我们将回到使用我们的原始图像(没有随机斑点)。随着结构化元素大小的增加，应用关闭操作的输出如下所示:

请注意关闭操作是如何开始弥合徽标中字母之间的间隙的。此外，像“e”、“s”和“a”这样的字母实际上是要填写的。

### **形态梯度**

**形态梯度是膨胀*和侵蚀*之间的差异。******用于确定图像中特定对象的轮廓:**

```py
# close all windows to cleanup the screen
cv2.destroyAllWindows()
cv2.imshow("Original", image)

# loop over the kernels a final time
for kernelSize in kernelSizes:
	# construct a rectangular kernel and apply a "morphological
	# gradient" operation to the image
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
	gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
	cv2.imshow("Gradient: ({}, {})".format(
		kernelSize[0], kernelSize[1]), gradient)
	cv2.waitKey(0)
```

需要注意的最重要的一行是**第 72 行**，在这里我们调用了`cv2.morphologyEx`——但是这次我们提供了`cv2.MORPH_GRADIENT`标志来表示我们想要应用形态渐变操作来显示我们的徽标的轮廓:

请注意在应用形态学梯度操作后，PyImageSearch 徽标的轮廓是如何清晰显示的。

### **礼帽/白帽和黑帽**

**一个** ***大礼帽*** **(也称** ***白礼帽*** **)形态学操作的区别是原始(灰度/单通道)** ***输入图像*** **和** ***开口*** **。**

大礼帽操作用于在**暗背景上显示图像的**亮区域**。**

到目前为止，我们只对二值图像应用了形态学运算。但是我们也可以将形态学操作应用于灰度图像。事实上，大礼帽/白礼帽和黑礼帽操作符都更适合灰度图像，而不是二值图像。

为了演示如何应用形态学运算，让我们看看下面的图像，我们的目标是检测汽车的牌照区域:

那么我们该如何着手做这件事呢？

嗯，看一下上面的例子图像，我们看到牌照是*亮的*，因为它是汽车本身的*暗背景*下的白色区域。寻找车牌区域的一个很好的起点是使用 top hat 操作符。

为了测试 top hat 操作符，创建一个新文件，将其命名为`morphological_hats.py`，并插入以下代码:

```py
# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
```

**第 2 行和第 3 行**导入我们需要的 Python 包，而**第 6-9 行**解析我们的命令行参数。我们只需要一个参数，`--image`，到我们的输入图像的路径(在我们的项目结构中我们假设它是`car.png`)。

让我们从磁盘加载我们的输入`--image`:

```py
# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# construct a rectangular kernel (13x5) and apply a blackhat
# operation which enables us to find dark regions on a light
# background
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
```

**第 12 行和第 13 行**从磁盘放置我们的输入`image`并将其转换为灰度，从而为我们的黑帽和白帽操作做准备。

**第 18 行**定义了一个宽度为 13 像素、高度为 5 像素的矩形结构元素。正如我在本课前面提到的，结构化元素可以是任意大小的。在本例中，我们应用了一个宽度几乎是高度 3 倍的矩形元素。

这是为什么呢？

因为车牌的宽度大约是高度的 3 倍！

通过对你想要在图像中检测的物体有一些基本的先验知识，我们可以构建结构元素来更好地帮助我们找到它们。

**第 19 行**应用黑帽运算符。

以类似的方式，我们也可以应用礼帽/白帽操作:

```py
# similarly, a tophat (also called a "whitehat") operation will
# enable us to find light regions on a dark background
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

# show the output images
cv2.imshow("Original", image)
cv2.imshow("Blackhat", blackhat)
cv2.imshow("Tophat", tophat)
cv2.waitKey(0)
```

要指定礼帽/白帽操作符而不是黑帽，我们只需将操作符的类型改为`cv2.MORPH_TOPHAT`。

下面您可以看到应用礼帽运算符的输出:

请注意*右侧*(即礼帽/白帽)区域是如何在*深色背景*的背景下*浅色*清晰地显示出来的——在这种情况下，我们可以清楚地看到汽车的牌照区域已经显露出来。

但也要注意，车牌字符本身没有包括在内。这是因为车牌字符在*浅色背景下是*深色*。*

为了帮助解决这个问题，我们可以应用一个黑帽运算符:

为了显示我们的车牌字符，您将首先通过 top hat 操作符分割出车牌本身，然后应用黑帽操作符(或阈值)来提取单个车牌字符(可能使用像轮廓检测这样的方法)。

### **运行我们的形态学操作演示**

要运行我们的形态学操作演示，请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

您可以使用以下命令执行`morphological_ops.py`脚本:

```py
$ python morphological_ops.py --image pyimagesearch_logo.png
```

使用以下命令可以启动`morphological_hats.py`脚本:

```py
$ python morphological_hats.py --image car.png
```

这些脚本的输出应该与我上面提供的图像和图形相匹配。

## **总结**

在本教程中，我们学习了形态学操作是应用于灰度或二进制图像的图像处理变换。这些操作需要一个**结构化元素**，用于定义操作所应用的像素邻域。

我们还回顾了您将在自己的应用程序中使用的最重要的形态学运算:

*   侵蚀
*   扩张
*   开始
*   关闭
*   形态梯度
*   Top hat/white hat
*   布莱克有

形态学运算通常用作更强大的计算机视觉解决方案的预处理步骤，如 OCR、自动车牌识别(ANPR)和条形码检测。

虽然这些技术很简单，但它们实际上非常强大，并且在预处理数据时非常有用。不要忽视他们。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！********