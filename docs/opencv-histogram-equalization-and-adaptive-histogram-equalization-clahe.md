# OpenCV 直方图均衡和自适应直方图均衡(CLAHE)

> 原文：<https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/>

在本教程中，您将学习使用 OpenCV 执行**直方图均衡化**和**自适应直方图均衡化。**

直方图均衡化是一种基本的图像处理技术，它通过更新图像直方图的像素强度分布来调整图像的全局对比度。这样做可以使低对比度区域在输出图像中获得更高的对比度。

本质上，直方图均衡的工作原理是:

1.  计算图像像素强度的直方图
2.  均匀地展开和分布最频繁的像素值(即直方图中具有最大计数的像素值)
3.  给出累积分布函数(CDF)的线性趋势

应用直方图均衡化的结果是具有更高全局对比度的图像。

我们可以通过应用一种称为对比度受限自适应直方图均衡化(CLAHE)的算法来进一步改善直方图均衡化，从而产生更高质量的输出图像。

除了摄影师使用直方图均衡化来校正曝光不足/曝光过度的图像之外，最广泛使用的直方图均衡化应用可以在医学领域中找到。

您通常会看到直方图均衡化应用于 X 射线扫描和 CT 扫描，以提高射线照片的对比度。这样做有助于医生和放射科医生更好地解读扫描结果，做出准确的诊断。

本教程结束时，您将能够使用 OpenCV 对图像成功应用基本直方图均衡和自适应直方图均衡。

**要学会用 OpenCV 使用直方图均衡化和自适应直方图均衡化，*继续阅读即可。***

## **OpenCV 直方图均衡和自适应直方图均衡(CLAHE)**

在本教程的第一部分，我们将讨论什么是直方图均衡化，以及我们如何使用 OpenCV 应用直方图均衡化。

从那里，我们将配置我们的开发环境，然后查看本指南的项目目录结构。

然后，我们将实现两个 Python 脚本:

1.  `simple_equalization.py`:使用 OpenCV 的`cv2.equalizeHist`函数执行基本直方图均衡。
2.  `adaptive_equalization.py`:使用 OpenCV 的`cv2.createCLAHE`方法进行自适应直方图均衡。

我们将在本指南的最后讨论我们的结果。

### **什么是直方图均衡化？**

直方图均衡化是一种基本的图像处理技术，可以提高图像的整体对比度。

应用直方图均衡化从计算输入灰度/单通道图像中像素强度的直方图开始:

请注意我们的直方图有许多峰值，表明有大量像素被归入相应的桶中。使用直方图均衡化，我们的目标是将这些像素分散到存储桶中，这样就不会有那么多像素被分入存储桶中。

从数学上来说，这意味着我们试图将线性趋势应用于我们的累积分布函数(CDF):

直方图均衡化应用前后可以在**图 3** 中看到:

请注意输入图像的对比度如何显著提高**，但代价是也提高了输入图像中*噪声*的对比度。**

这就提出了一个问题:

> 有没有可能在不增加噪点的同时提高图像对比度？

答案是*“是的”，*你只需要应用**自适应直方图均衡化。**

通过自适应直方图均衡化，我们将输入图像划分为一个 *M x N* 网格。然后，我们对网格中的每个单元应用均衡，从而产生更高质量的输出图像:

缺点是自适应直方图均衡从定义上来说在计算上更复杂(但是考虑到现代的硬件，两种实现仍然相当快)。

### **如何使用 OpenCV 进行直方图均衡化？**

OpenCV 通过以下两个函数实现了基本直方图均衡和自适应直方图均衡:

1.  `cv2.equalizeHist`
2.  `cv2.createCLAHE`

应用`cv2.equalizeHist`函数非常简单，只需将图像转换为灰度，然后对其调用`cv2.equalizeHist`:

```py
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
```

执行自适应直方图均衡要求我们:

1.  将输入图像转换为灰度/从中提取单个通道
2.  使用`cv2.createCLAHE`实例化 CLAHE 算法
3.  对 CLAHE 对象调用`.apply`方法以应用直方图均衡化

这比听起来容易得多，只需要几行代码:

```py
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized = clahe.apply(gray)
```

注意，我们向`cv2.createCLAHE`提供了两个参数:

1.  `clipLimit`:这是对比度限制的阈值
2.  `tileGridSize`:将输入图像分成*M×N*个小块，然后对每个局部小块应用直方图均衡化

在本指南的剩余部分，您将练习使用`cv2.equalizeHist`和`cv2.createCLAHE`。

### **配置您的开发环境**

要了解如何使用 OpenCV 应用直方图均衡，您需要安装 OpenCV 库。

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

 *那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

在我们用 OpenCV 实现直方图均衡化之前，让我们先回顾一下我们的项目目录结构。

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

从那里，检查项目目录结构:

```py
$ tree . --dirsfirst
.
├── images
│   ├── boston.png
│   ├── dog.png
│   └── moon.png
├── adaptive_equalization.py
└── simple_equalization.py

1 directory, 5 files
```

我们今天将讨论两个 Python 脚本:

1.  `simple_equalization.py`:使用 OpenCV 应用基本直方图均衡。
2.  `adaptive_equalization.py`:使用 CLAHE 算法执行自适应直方图均衡化。

我们的`images`目录包含我们将应用直方图均衡化的示例图像。

### **用 OpenCV 实现标准直方图均衡**

回顾了我们的项目目录结构后，让我们继续用 OpenCV 实现基本的直方图均衡化。

打开项目文件夹中的`simple_equalization.py`文件，让我们开始工作:

```py
# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to the input image")
args = vars(ap.parse_args())
```

**第 2 行和第 3 行**导入我们需要的 Python 包，而**第 6-9 行**解析我们的命令行参数。

这里我们只需要一个参数，`--image`，它是我们在磁盘上输入图像的路径，我们希望在这里应用直方图均衡化。

解析完命令行参数后，我们可以进入下一步:

```py
# load the input image from disk and convert it to grayscale
print("[INFO] loading input image...")
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply histogram equalization
print("[INFO] performing histogram equalization...")
equalized = cv2.equalizeHist(gray)
```

**第 13 行**从磁盘加载我们的`image`，而**第 14 行**将我们的图像从 RGB 转换成灰度。

**第 18 行**使用`cv2.equalizeHist`功能执行基本直方图均衡化。我们必须传入的唯一必需参数是灰度/单通道图像。

***注意:使用 OpenCV 进行直方图均衡化时，我们必须提供灰度/单通道图像。*** *如果我们试图传入一个多通道图像，OpenCV 会抛出一个错误。要对多通道图像执行直方图均衡化，您需要(1)将图像分割到其各自的通道中，(2)均衡化每个通道，以及(3)将通道合并在一起。*

最后一步是显示我们的输出图像:

```py
# show the original grayscale image and equalized image
cv2.imshow("Input", gray)
cv2.imshow("Histogram Equalization", equalized)
cv2.waitKey(0)
```

这里，我们显示输入的`gray`图像以及直方图均衡化的图像。

### **OpenCV 直方图均衡结果**

我们现在准备用 OpenCV 应用基本的直方图均衡化！

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

从那里，打开一个终端并执行以下命令:

```py
$ python simple_equalization.py --image images/moon.png
[INFO] loading input image...
[INFO] performing histogram equalization...
```

在顶部的*，*我们有月球的原始输入图像。*底部*显示应用直方图均衡化后的输出。请注意，我们提高了图像的整体对比度。

让我们试试另一张照片，这张曝光不足的照片:

```py
$ python simple_equalization.py --image images/dog.png
[INFO] loading input image...
[INFO] performing histogram equalization...
```

狗*(左)*因曝光不足而显得褪色。通过应用直方图均衡*(右)，*我们修正了这个效果，提高了狗的对比度。

下图突出显示了通过直方图均衡化进行全局对比度调整的局限性之一:

```py
$ python simple_equalization.py --image images/boston.png
[INFO] loading input image...
[INFO] performing histogram equalization...
```

左边*上的图片显示了几年前我和妻子在波士顿过圣诞节。由于相机上的自动调节，我们的脸相当黑，很难看到我们。*

通过应用直方图均衡化*(右)，*我们可以看到，不仅我们的脸是可见的，我们还可以看到坐在我们后面的另一对夫妇！如果没有直方图均衡，您可能会错过另一对。

然而，我们的产出并不完全令人满意。首先，壁炉里的火完全熄灭了。如果你研究我们的脸，特别是我的脸，你会看到我的前额现在完全被洗掉了。

为了改善我们的结果，我们需要应用自适应直方图均衡化。

### **用 OpenCV 实现自适应直方图均衡**

至此，我们已经看到了基本直方图均衡化的一些局限性。

自适应直方图均衡化虽然在计算上有点昂贵，但可以产生比简单直方图均衡化更好的结果。但是不要相信我的话——你应该自己看看结果。

打开项目目录结构中的`adaptive_equalization.py`文件，插入以下代码:

```py
# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to the input image")
ap.add_argument("-c", "--clip", type=float, default=2.0,
	help="threshold for contrast limiting")
ap.add_argument("-t", "--tile", type=int, default=8,
	help="tile grid size -- divides image into tile x time cells")
args = vars(ap.parse_args())
```

这里我们只需要两个导入，`argparse`用于命令行参数，`cv2`用于 OpenCV 绑定。

然后，我们有三个命令行参数，其中一个是必需的，另两个是可选的(但是在使用 CLAHE 进行实验时对调整和使用很有用):

1.  `--image`:我们的输入图像在磁盘上的路径，我们希望在这里应用直方图均衡。
2.  `--clip`:对比度限制的阈值。您通常希望将这个值留在`2-5`的范围内。如果你设置的值太大，那么实际上，你所做的是最大化局部对比度，这反过来会最大化噪声(这与你想要的相反)。相反，尽量将该值保持在最低水平。
3.  `--tile`:CLAHE 的平铺网格尺寸。从概念上讲，我们在这里做的是将输入图像分成`tile x tile`个单元，然后对每个单元应用直方图均衡化(使用 CLAHE 提供的附加功能)。
4.  现在让我们用 OpenCV 来应用 CLAHE:

```py
# load the input image from disk and convert it to grayscale
print("[INFO] loading input image...")
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
print("[INFO] applying CLAHE...")
clahe = cv2.createCLAHE(clipLimit=args["clip"],
	tileGridSize=(args["tile"], args["tile"]))
equalized = clahe.apply(gray)
```

**第 17 行和第 18 行**从磁盘加载我们的输入图像，并将其转换为灰度，就像我们对基本直方图均衡化所做的那样。

**第 22 和 23 行**通过`cv2.createCLAHE`函数初始化我们的`clahe`对象。这里，我们提供了通过命令行参数提供的`clipLimit`和`tileGridSize`。

对`.apply`方法的调用将自适应直方图均衡应用于`gray`图像。

最后一步是在屏幕上显示输出图像:

```py
# show the original grayscale image and CLAHE output image
cv2.imshow("Input", gray)
cv2.imshow("CLAHE", equalized)
cv2.waitKey(0)
```

这里，我们显示我们的输入图像`gray`以及来自 CLAHE 算法的输出图像`equalized`。

### **自适应直方图均衡结果**

现在让我们用 OpenCV 应用自适应直方图均衡化！

访问本教程的 ***“下载”*** 部分来检索源代码和示例图像。

从那里，打开一个 shell 并执行以下命令:

```py
$ python adaptive_equalization.py --image images/boston.png
[INFO] loading input image...
[INFO] applying CLAHE...
```

在左边的*，*是我们的原始输入图像。然后，我们在右侧的*应用自适应直方图均衡化——将这些结果与应用基本直方图均衡化的**图 4、**的结果进行比较。*

 *请注意自适应直方图均衡化如何提高了输入图像的对比度。我和我妻子更容易被看到。背景中那对曾经几乎看不见的夫妇可以被看到。额头上的神器比较少等等。

### **直方图均衡化建议**

当构建自己的图像处理管道并发现应该应用直方图均衡化时，我建议使用`cv2.equalizeHist`从简单的直方图均衡化开始。但是如果你发现结果很差，反而增加了输入图像的噪声，那么你应该通过`cv2.createCLAHE`尝试使用自适应直方图均衡化。

### **学分**

我感谢 Aruther Cotse(犹他大学)关于使用直方图进行图像处理的精彩报告。Cotse 的工作启发了这篇文章中的一些例子。

此外，我感谢维基百科关于[直方图均衡化](https://en.wikipedia.org/wiki/Histogram_equalization)页面的贡献者。如果您对直方图均衡化背后的更多数学细节感兴趣，请务必参考该页面。

示例`moon.png`图片来自[EarthSky](https://earthsky.org/space/moons-craters-earths-history-meteorite-bombardment)上的这篇文章，而`dog.png`图片来自[本页面](https://theonlinephotographer.typepad.com/the_online_photographer/2014/12/in-praise-of-low-contrast.html)。

## **总结**

在本教程中，您学习了如何使用 OpenCV 执行基本直方图均衡和自适应直方图均衡。

基本直方图均衡化旨在通过“分散”图像中常用的像素强度来提高图像的整体对比度。

虽然简单的直方图均衡易于应用且计算效率高，但问题*T2 在于它会增加噪声。本来可以轻易滤除的基本噪声现在进一步污染了信号(即我们想要处理的图像成分)。*

如果发生这种情况，我们可以应用**自适应直方图均衡**来获得更好的结果。

自适应直方图均衡化的工作原理是将图像划分为一个 *M x N* 网格，然后对每个网格局部应用直方图均衡化。结果是输出图像总体上具有更高的对比度，并且(理想地)噪声仍然被抑制。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****