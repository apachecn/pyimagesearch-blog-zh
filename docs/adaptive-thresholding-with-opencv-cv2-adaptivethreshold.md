# 使用 OpenCV ( cv2.adaptiveThreshold)进行自适应阈值处理

> 原文：<https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/>

在本教程中，您将了解自适应阈值以及如何使用 OpenCV 和`cv2.adaptiveThreshold`函数应用自适应阈值。

上周，我们学习了如何使用`cv2.threshold`函数应用基本阈值和 Otsu 阈值。

当应用基本阈值时，我们不得不*手动*提供阈值 *T* ，以分割我们的前景和背景。

假设我们的输入图像中的像素强度呈双峰分布，Otsu 的阈值方法可以*自动*确定 *T* 的最佳值。

然而，这两种方法都是**全局阈值技术**，这意味着使用相同的值 *T* 来测试*输入图像中的所有像素*，从而将它们分割成前景和背景。

这里的问题是只有一个值***可能不够。**由于光照条件、阴影等的变化。可能的情况是， *T* 的一个值将对输入图像的某一部分起作用，但在不同的部分将完全失效。*

 *我们可以利用**自适应阈值处理，而不是立即抛出我们的手，声称传统的计算机视觉和图像处理不会解决这个问题(从而立即跳到训练像 Mask R-CNN 或 U-Net 这样的深度神经分割网络)。**

顾名思义，自适应阈值处理一次考虑一小组相邻像素，为该特定局部区域计算 *T* ，然后执行分割。

根据您的项目，利用自适应阈值可以使您:

1.  获得比使用全局阈值方法更好的分割，例如基本阈值和 Otsu 阈值
2.  避免训练专用掩模 R-CNN 或 U-Net 分割网络的耗时且计算昂贵的过程

**要了解如何使用 OpenCV 和`cv2.adaptiveThreshold`函数、** ***执行自适应阈值处理，请继续阅读。***

## **使用 OpenCV ( cv2.adaptiveThreshold)的自适应阈值处理**

在本教程的第一部分，我们将讨论什么是自适应阈值处理，包括自适应阈值处理与我们到目前为止讨论的“正常”全局阈值处理方法有何不同。

从那里，我们将配置我们的开发环境，并审查我们的项目目录结构。

然后我将向您展示如何使用 OpenCV 和`cv2.adaptiveThreshold`函数实现自适应阈值处理。

我们将讨论我们的自适应阈值结果来结束本教程。

### 什么是自适应阈值处理？自适应阈值与“正常”阈值有何不同？

正如我们在本教程前面所讨论的，使用简单阈值方法的一个缺点是我们需要手动提供我们的阈值， *T* 。此外，找到一个好的 *T* 值可能需要许多手动实验和参数调整，这在大多数情况下是不实际的。

为了帮助我们自动确定 *T* 的值，我们利用了 Otsu 的方法。虽然 Otsu 的方法可以节省我们玩“猜测和检查”游戏的大量时间，但我们只剩下一个单一的值 *T* 来对整个图像进行阈值处理。

对于光线条件受控的简单图像，这通常不是问题。但是对于图像中光照不均匀的情况，只有单一的值 *T* 会严重损害我们的阈值处理性能。

**简单来说，只有一个值** ***T*** **可能不够。**

为了克服这个问题，我们可以使用自适应阈值处理，它考虑像素的小邻居，然后为每个邻居找到一个最佳阈值 *T* 。这种方法允许我们处理像素强度可能存在巨大范围的情况，并且 *T* 的最佳值可能因图像的不同部分而变化。

在自适应阈值处理中，有时称为局部阈值处理，我们的目标是统计检查给定像素的邻域中的像素强度值， *p* 。

作为所有自适应和局部阈值方法基础的一般假设是，图像的较小区域更可能具有近似均匀的照明。这意味着图像的局部区域将具有相似的照明，与图像整体相反，图像整体的每个区域可能具有显著不同的照明。

然而，为局部阈值选择像素邻域的大小是绝对关键的。

邻域必须*足够大*以覆盖足够的背景和前景像素，否则 *T* 的值将或多或少无关紧要。

但是如果我们使我们的邻域值*太大*，那么我们完全违背了图像的局部区域将具有近似均匀照明的假设。同样，如果我们提供非常大的邻域，那么我们的结果将看起来非常类似于使用简单阈值或 Otsu 方法的全局阈值。

实际上，调整邻域大小(通常)并不是一个困难的问题。您通常会发现，有很大范围的邻域大小可以为您提供足够的结果-这不像找到一个最佳值 *T* ，它可能会影响或破坏您的阈值输出。

### **自适应阈值处理的数学基础**

正如我上面提到的，我们在自适应阈值处理中的目标是统计地检查我们图像的局部区域，并为每个区域确定一个最佳值*T*——这就引出了一个问题:*我们使用哪个统计量来计算每个区域的阈值 T？*

通常的做法是使用每个区域中像素强度的算术平均值或高斯平均值(确实存在其他方法，但是算术平均值和高斯平均值是最流行的)。

在算术平均中，邻域中的每个像素对计算 *T* 的贡献相等。并且在高斯均值中，离区域的 *(x，y)*-坐标中心越远的像素值对 *T* 的整体计算贡献越小。

计算 *T* 的一般公式如下:

*T =均值(I[L])–C*

其中平均值是算术平均值或高斯平均值， *I [L]* 是图像的局部子区域， *I* ，并且 *C* 是我们可以用来微调阈值 *T* 的某个常数。

如果这一切听起来令人困惑，不要担心，我们将在本教程的后面获得使用自适应阈值的实践经验。

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

让我们从回顾我们的项目目录结构开始。

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像:

```py
$ tree . --dirsfirst
.
├── adaptive_thresholding.py
└── steve_jobs.png

0 directories, 2 files
```

今天我们要回顾一个 Python 脚本，`adaptive_thresholding.py`。

我们将把这个脚本应用到我们的示例图像`steve_jobs.png`，它将显示比较和对比的结果:

1.  基本全局阈值
2.  Otsu 全局阈值
3.  自适应阈值

我们开始吧！

### **使用 OpenCV 实施自适应阈值处理**

我们现在准备用 OpenCV 实现自适应阈值！

打开项目目录中的`adaptive_thresholding.py`文件，让我们开始工作:

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

**第 2 行和第 3 行**导入我们需要的 Python 包— `argparse`用于命令行参数，而`cv2`用于 OpenCV 绑定。

从那里我们解析我们的命令行参数。这里我们只需要一个参数，`--image`，它是我们想要阈值化的输入图像的路径。

现在让我们从磁盘加载我们的映像并对其进行预处理:

```py
# load the image and display it
image = cv2.imread(args["image"])
cv2.imshow("Image", image)

# convert the image to grayscale and blur it slightly
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
```

我们首先从磁盘加载我们的`image`,并在屏幕上显示原始图像。

在那里，我们对图像进行预处理，将它转换成灰度并用一个 *7 *×* 7* 内核进行模糊处理。应用高斯模糊有助于移除图像中我们不关心的一些高频边缘，并允许我们获得更“干净”的分割。

现在让我们应用带有硬编码阈值`T=230`的**基本阈值**:

```py
# apply simple thresholding with a hardcoded threshold value
(T, threshInv) = cv2.threshold(blurred, 230, 255,
	cv2.THRESH_BINARY_INV)
cv2.imshow("Simple Thresholding", threshInv)
cv2.waitKey(0)
```

在这里，我们应用基本的阈值处理并在屏幕上显示结果(如果你想了解简单阈值处理如何工作的更多细节，你可以阅读上周关于 [*OpenCV 阈值处理(cv2.threshold )*](https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/) 的教程)。

接下来，让我们应用 **Otsu 的阈值方法**，该方法*自动*计算我们的阈值参数`T`的最佳值，假设像素强度的双峰分布:

```py
# apply Otsu's automatic thresholding
(T, threshInv) = cv2.threshold(blurred, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Otsu Thresholding", threshInv)
cv2.waitKey(0)
```

现在，让我们使用平均阈值方法应用**自适应阈值**:

```py
# instead of manually specifying the threshold value, we can use
# adaptive thresholding to examine neighborhoods of pixels and
# adaptively threshold each neighborhood
thresh = cv2.adaptiveThreshold(blurred, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
cv2.imshow("Mean Adaptive Thresholding", thresh)
cv2.waitKey(0)
```

**第 34 行和第 35 行**使用 OpenCV 的`cv2.adaptiveThreshold`函数应用自适应阈值。

我们从传入模糊的输入图像开始。

第二个参数是输出阈值，就像简单的阈值处理和 Otsu 方法一样。

第三个论点是自适应阈值方法。这里我们提供一个值`cv2.ADAPTIVE_THRESH_MEAN_C`来表示我们正在使用局部像素邻域的算术平均值来计算我们的阈值`T`。

我们还可以提供一个值`cv2.ADAPTIVE_THRESH_GAUSSIAN_C`(我们接下来会这样做)来表示我们想要使用高斯平均值——您选择哪种方法完全取决于您的应用和情况，所以您会想要尝试这两种方法。

`cv2.adaptiveThreshold`的第四个值是阈值法，也与简单阈值法和 Otsu 阈值法一样。这里我们传入一个值`cv2.THRESH_BINARY_INV`来表示任何通过阈值测试的像素值都将有一个输出值`0`。否则，它的值将为`255`。

第五个参数是我们的像素邻域大小。这里你可以看到，我们将计算图像中每个 *21×21* 子区域的平均灰度像素强度值，以计算我们的阈值`T`。

`cv2.adaptiveThreshold`的最后一个参数是我上面提到的常数`C`——这个值只是让我们微调我们的阈值。

可能存在平均值本身不足以区分背景和前景的情况——因此，通过增加或减少一些值`C`,我们可以改善阈值的结果。同样，您为`C`使用的值完全取决于您的应用和情况，但是这个值很容易调整。

在这里，我们设置`C=10`。

最后，平均自适应阈值的输出显示在我们的屏幕上。

现在让我们来看看高斯版本的自适应阈值处理:

```py
# perform adaptive thresholding again, this time using a Gaussian
# weighting versus a simple mean to compute our local threshold
# value
thresh = cv2.adaptiveThreshold(blurred, 255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
cv2.imshow("Gaussian Adaptive Thresholding", thresh)
cv2.waitKey(0)
```

这一次，我们计算的是 21×21 T2 区域的加权高斯平均值，它赋予靠近窗口中心的像素更大的权重。然后我们设置`C=4`，这个值是我们在这个例子中根据经验调整的。

最后，高斯自适应阈值的输出显示在我们的屏幕上。

### **自适应阈值结果**

让我们把自适应阈值工作！

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

从那里，您可以执行`adaptive_thresholding.py`脚本:

```py
$ python adaptive_thresholding.py --image steve_jobs.png
```

在这里，你可以看到我们的输入图像，`steve_jobs.png`，这是苹果电脑公司史蒂夫·乔布斯的名片:

我们的目标是从*背景*(名片的其余部分)中分割出*前景*(苹果标志和文字)。

使用具有预设值 *T* 的简单阈值能够在某种程度上执行这种分割:

是的，苹果的标志和文字是前景的一部分，但我们也有很多噪声(这是不可取的)。

让我们看看 Otsu 阈值处理能做什么:

不幸的是，Otsu 的方法在这里失败了。所有的文本都在分割中丢失了，还有苹果标志的一部分。

幸运的是，我们有自适应阈值来拯救:

**图 6** 显示了平均自适应阈值的输出。

通过应用自适应阈值处理，我们可以对输入图像的*局部*区域进行阈值处理(而不是使用我们的阈值参数 *T* 的全局值)。这样做极大地改善了我们的前景和分割结果。

现在让我们看看高斯自适应阈值处理的输出:

这种方法提供了最好的结果。文本和大部分苹果标志一样是分段的。然后我们可以应用[形态学操作](https://pyimagesearch.com/2021/04/28/opencv-morphological-operations/)来清理最终的分割。

## **总结**

在本教程中，我们学习了自适应阈值和 OpenCV 的`cv2.adaptiveThresholding`功能。

与作为*全局*阈值化方法的[基本阈值化和 Otsu 阈值化](https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/)不同，自适应阈值化代替了对像素的*局部*邻域进行阈值化。

本质上，自适应阈值处理假设图像的局部区域将比图像整体具有更均匀的照明和光照。因此，为了获得更好的阈值处理结果，我们应该研究图像的子区域，并分别对它们进行阈值处理，以获得我们的最终输出图像。

自适应阈值处理往往会产生良好的结果，但在计算上比 Otsu 方法或简单阈值处理更昂贵，但在照明条件不均匀的情况下，自适应阈值处理是一种非常有用的工具。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，** ***只需在下面的表格中输入您的电子邮件地址！*****