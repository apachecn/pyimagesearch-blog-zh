# 使用 OpenCV 进行图像遮罩

> 原文：<https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/>

在本教程中，您将学习如何使用 OpenCV 遮罩图像。

我之前的指南讨论了[位运算](https://pyimagesearch.com/2021/01/19/opencv-bitwise-and-or-xor-and-not/)，这是一组在图像处理中大量使用的非常常见的技术。

正如我之前暗示的，我们可以使用位运算和掩码来构建非矩形的感兴趣区域。这使得我们能够从图像中提取出完全*任意形状的区域。*

**简单地说；蒙版让我们只关注图像中我们感兴趣的部分。**

例如，假设我们正在构建一个计算机视觉系统来识别人脸。我们感兴趣寻找和描述的唯一图像部分是图像中包含人脸的部分——我们根本不关心图像的其余内容。假如我们可以找到图像中的人脸，我们可以构造一个遮罩来只显示图像中的人脸。

你会遇到的另一个图像蒙版应用是阿尔法混合和透明(例如，在本指南中的 [*用 OpenCV*](https://pyimagesearch.com/2018/11/05/creating-gifs-with-opencv/) 创建 gif)。当使用 [OpenCV](https://opencv.org/) 对图像应用透明时，我们需要告诉 OpenCV 图像的哪些部分应该应用透明，哪些部分不应该应用— *遮罩允许我们进行区分。*

**要了解如何使用 OpenCV 执行图像遮罩，*请继续阅读。***

## **使用 OpenCV 进行图像遮蔽**

在本教程的第一部分，我们将配置我们的开发环境并回顾我们的项目结构。

然后，我们将使用 OpenCV 实现一个 Python 脚本来屏蔽图像。

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
*   准备好在您的 Windows、macOS 或 Linux 系统上运行代码*了吗*？

那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### 项目结构

使用 OpenCV 执行图像遮罩比您想象的要容易。但是在我们写任何代码之前，让我们先回顾一下我们的项目目录结构。

首先使用本指南的 ***“下载”*** 部分访问源代码和示例图像。

您的项目文件夹应该如下所示:

```py
$ tree . --dirsfirst
.
├── adrian.png
└── opencv_masking.py

0 directories, 2 files
```

我们的`opencv_masking.py`脚本将从磁盘加载输入的`adrian.png`图像。然后，我们将使用矩形和圆形遮罩分别从图像中提取身体和面部。

### **用 OpenCV 实现图像遮罩**

让我们来学习如何使用 OpenCV 应用图像遮罩！

打开项目目录结构中的`opencv_masking.py`文件，让我们开始工作:

```py
# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="adrian.png",
	help="path to the input image")
args = vars(ap.parse_args())
```

**第 2-4 行**导入我们需要的 Python 包。然后我们在**的第 7-10 行**解析我们的命令行参数。

这里我们只需要一个开关，`--image`，它是我们想要遮罩的图像的路径。我们继续将`--image`参数默认为项目目录中的`adrian.png`文件。

现在，让我们从磁盘加载这个映像并执行屏蔽:

```py
# load the original input image and display it to our screen
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# a mask is the same size as our image, but has only two pixel
# values, 0 and 255 -- pixels with a value of 0 (background) are
# ignored in the original image while mask pixels with a value of
# 255 (foreground) are allowed to be kept
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
cv2.imshow("Rectangular Mask", mask)

# apply our mask -- notice how only the person in the image is
# cropped out
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)
```

**第 13 行和第 14 行**从磁盘加载原始的`image`并显示在我们的屏幕上:

然后，我们构造一个 NumPy 数组，用零填充，其宽度和高度与第 20 行的原始图像相同。

正如我在之前关于 OpenCV 裁剪 *[图像的教程中提到的，我们可以使用](https://pyimagesearch.com/2021/01/19/crop-image-with-opencv/)[物体检测方法](https://pyimagesearch.com/category/object-detection/)自动检测图像中的物体/人物。不过，我们将暂时使用我们对示例图像的先验知识。*

我们*知道*我们想要提取的区域在图像的*左下角*。**第 21 行**在我们的蒙版上画了一个白色的矩形，对应着我们要从原始图像中提取的区域。

记得在我们的[位运算教程](https://pyimagesearch.com/2021/01/19/opencv-bitwise-and-or-xor-and-not/)中复习过`cv2.bitwise_and`函数吗？事实证明，这个函数在对图像应用蒙版时被广泛使用。

我们使用`cv2.bitwise_and`函数在**第 26 行**应用我们的蒙版。

前两个参数是`image`本身(即我们想要应用位运算的图像)。

然而，这个函数的重要部分是`mask`关键字。当提供时，当输入图像的像素值相等时，`bitwise_and`函数为`True`，并且掩码在每个 *(x，y)*-坐标处为非零(在这种情况下，仅是白色矩形的像素部分)。

应用我们的掩码后，我们在第 27 和 28 行显示输出，你可以在**图 3** 中看到:

使用我们的矩形遮罩，我们可以只提取图像中包含人的区域，而忽略其余部分。

让我们来看另一个例子，但是这次使用的是一个*非矩形*遮罩:

```py
# now, let's make a circular mask with a radius of 100 pixels and
# apply the mask again
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(mask, (145, 200), 100, 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask)

# show the output images
cv2.imshow("Circular Mask", mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)
```

在第 32 行的**，**上，我们重新初始化我们的`mask`，用零和与我们原始图像相同的尺寸填充。

然后，我们在我们的蒙版图像上画一个白色的圆圈，从我的脸的中心开始，半径为`100`像素。

然后再次使用`cv2.bitwise_and`功能，在**线 34** 上应用圆形掩模。

我们的圆形遮罩的结果可以在**图 4** 中看到:

在这里，我们可以看到我们的圆形遮罩显示在左边的*和右边的*上。与**图 3** 的输出不同，当我们提取一个矩形区域时，这一次，我们提取了一个圆形区域，它只对应于图像中我的脸。**

此外，我们可以使用这种方法从任意形状(矩形、圆形、直线、多边形等)的图像中提取区域。).

### **OpenCV 图像屏蔽结果**

要使用 OpenCV 执行图像遮罩，请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

从那里，打开一个 shell 并执行以下命令:

```py
$ python opencv_masking.py
```

您的屏蔽输出应该与我在上一节中的输出相匹配。

OpenCV 图像遮罩是处理图像的强大工具。它允许您将效果应用到单个图像，并创建一个全新的外观。

使用 OpenCV 图像遮罩，您可以选择性地修改颜色、对比度、变亮或变暗、添加或移除噪点，甚至从图像中擦除部分或对象。还可以添加文本和特殊效果，甚至将图像转换成不同的文件格式。

OpenCV 图像遮罩是一种轻松创建令人惊叹的视觉效果的好方法，可以帮助您:

*   装帧设计艺术
*   应用开发
*   机器人学
*   自治
*   计算机视觉研究主题

## **总结**

你来[学计算机视觉](https://pyimagesearch.com/)和基础蒙版，超！在本教程中，您学习了使用 OpenCV 进行遮罩的基础知识。

遮罩的关键点在于，它们允许我们将计算集中在感兴趣的图像区域。当我们探索机器学习、图像分类和对象检测等主题时，将我们的计算集中在我们感兴趣的区域会产生巨大的影响。

例如，让我们假设我们想要建立一个系统来对花的种类进行分类。

实际上，我们可能只对花瓣的颜色和纹理感兴趣来进行分类。但是，由于我们是在自然环境中拍摄的照片，我们的图像中也会有许多其他区域，包括地面上的灰尘、昆虫和其他花卉。我们如何量化和分类我们感兴趣的花？正如我们将看到的，答案是面具。

更新时间:2022 年 12 月 30 日，更新链接和内容。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***