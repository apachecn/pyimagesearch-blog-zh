# 使用 OpenCV 裁剪图像

> 原文：<https://pyimagesearch.com/2021/01/19/crop-image-with-opencv/>

在本教程中，您将学习如何使用 OpenCV 裁剪图像。

![](img/58270a8578784a561559106fd0cef220.png)

顾名思义，**裁剪是*选择*和*提取*感兴趣区域**(或简称 ROI)的行为，是图像中我们感兴趣的部分。

例如，在人脸检测应用程序中，我们可能希望从图像中裁剪出人脸。如果我们正在开发一个 Python 脚本来识别图像中的狗，我们可能希望在找到狗后从图像中裁剪掉它。

我们已经在教程中使用了裁剪， *[使用 OpenCV](https://pyimagesearch.com/2021/01/20/opencv-getting-and-setting-pixels/) ，*获取和设置像素，但是为了更加完整，我们将再次回顾它。

**要学习如何用 OpenCV 裁剪图像，*继续阅读。***

## **使用 OpenCV 的裁剪图像**

在本教程的第一部分，我们将讨论如何将 OpenCV 图像表示为 NumPy 数组。因为每个图像都是一个 NumPy 数组，所以我们可以利用 NumPy 数组切片来裁剪图像。

从那里，我们将配置我们的开发环境，并审查我们的项目目录结构。

然后我将演示用 OpenCV 裁剪图像是多么简单！

### **了解使用 OpenCV 和 NumPy 数组切片进行图像裁剪**

当我们裁剪图像时，我们希望删除图像中我们不感兴趣的外部部分。我们通常将这一过程称为选择我们的感兴趣区域，或者更简单地说，我们的 ROI。

我们可以通过使用 NumPy 数组切片来实现图像裁剪。

让我们首先用范围从*【0，24】:*的值初始化一个 NumPy 列表

```py
>>> import numpy as np
>>> I = np.arange(0, 25)
>>> I
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
       15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
>>> 
```

现在让我们把这个 1D 列表重塑成一个 2D 矩阵，假设它是一个图像:

```py
>>> I = I.reshape((5, 5))
>>> I
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
>>> 
```

现在，假设我要提取从 *x = 0* 、 *y = 0* 开始，到 *x = 2* 、 *y = 3* 结束的“像素”。使用以下代码可以实现这一点:

```py
>>> I[0:3, 0:2]
array([[ 0,  1],
       [ 5,  6],
       [10, 11]])
>>> 
```

注意我们是如何提取了三行( *y = 3* )和两列( *x = 2* )。

现在，让我们提取从 *x = 1* 、 *y = 3* 开始到 *x = 5* 和 *y = 5* 结束的像素:

```py
>>> I[3:5, 1:5]
array([[16, 17, 18, 19],
       [21, 22, 23, 24]])
>>>
```

这个结果提供了图像的最后两行，减去第一列。

你注意到这里的模式了吗？

将 NumPy 数组切片应用于图像时，我们使用以下语法提取 ROI:

`roi = image[startY:endY, startX:endX]`

`startY:endY`切片提供图像中我们的*行*(因为*y*-轴是我们的行数)，而`startX:endX`提供我们的*列*(因为*x*-轴是列数)。现在花一点时间说服你自己，上面的陈述是正确的。

但是如果你有点困惑，需要更多的说服力，不要担心！在本指南的后面，我将向您展示一些代码示例，使 OpenCV 的图像裁剪更加清晰和具体。

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

 *那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

在我们使用 OpenCV 实现图像裁剪之前，让我们先回顾一下我们的项目目录结构。

首先使用本指南的 ***“下载”*** 部分访问源代码和示例图像:

```py
$ tree . --dirsfirst
.
├── adrian.png
└── opencv_crop.py

0 directories, 2 files
```

今天我们只回顾一个 Python 脚本`opencv_crop.py`，它将从磁盘加载输入的`adrian.png`图像，然后使用 NumPy 数组切片从图像中裁剪出脸部和身体。

### **用 OpenCV 实现图像裁剪**

我们现在准备用 OpenCV 实现图像裁剪。

打开项目目录结构中的`opencv_crop.py`文件，插入以下代码:

```py
# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="adrian.png",
	help="path to the input image")
args = vars(ap.parse_args())
```

**第 2 行和第 3 行**导入我们需要的 Python 包，而**第 6-9 行**解析我们的命令行参数。

我们只需要一个命令行参数`--image`，它是我们希望裁剪的输入图像的路径。对于这个例子，我们将默认把`--image`切换到项目目录中的`adrian.png`文件。

接下来，让我们从磁盘加载我们的映像:

```py
# load the input image and display it to our screen
image = cv2.imread(args["image"])
cv2.imshow("Original", image)

# cropping an image with OpenCV is accomplished via simple NumPy
# array slices in startY:endY, startX:endX order -- here we are
# cropping the face from the image (these coordinates were
# determined using photo editing software such as Photoshop,
# GIMP, Paint, etc.)
face = image[85:250, 85:220]
cv2.imshow("Face", face)
cv2.waitKey(0)
```

**第 12 和 13 行**加载我们的原件`image`，然后显示在我们的屏幕上:

我们的目标是使用简单的裁剪方法从这个区域中提取出我的脸和身体。

我们通常会应用[物体检测技术](https://pyimagesearch.com/category/object-detection/)来检测图像中我的脸和身体。然而，由于我们仍然处于 OpenCV 教育课程的相对早期，我们将使用我们的*先验*图像知识和*手动*提供身体和面部所在的 NumPy 数组切片。

同样，我们当然可以使用[对象检测方法从图像中自动检测和提取人脸](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)*，但是暂时让事情简单一些。*

 *我们用一行代码从图像中提取我的脸(**第 20 行**)。

我们提供 NumPy 数组切片来提取图像的矩形区域，从 *(85，85)* 开始，到 *(220，250)* 结束。

我们为作物提供指数的顺序可能看起来违反直觉；但是，请记住 OpenCV 将图像表示为 NumPy 数组，首先是高度(行数)，其次是宽度(列数)。

为了执行我们的裁剪，NumPy 需要四个索引:

*   **起始 *y* :** 起始 *y* 坐标。在这种情况下，我们从 *y = 85* 开始。
*   **终点*y*终点 *y* 坐标。我们将在 y = 250 时结束收割。**
*   **起始 *x* :** 起始 *x* 切片坐标。我们在 *x = 85* 开始裁剪。
*   **结束*x*T3:结束*x*-切片的轴坐标。我们的切片结束于 *x = 220* 。**

我们可以看到下面裁剪我的脸的结果:

同样，我们可以从图像中裁剪出我的身体:

```py
# apply another image crop, this time extracting the body
body = image[90:450, 0:290]
cv2.imshow("Body", body)
cv2.waitKey(0)
```

裁剪我的身体是通过从原始图像的坐标 *(0，90)* 开始裁剪到 *(290，450)* 结束。

下面你可以看到 OpenCV 的裁剪输出:

虽然简单，裁剪是一个极其重要的技能，我们将在整个系列中使用。如果你仍然对裁剪感到不安，现在一定要花时间练习，磨练你的技能。从现在开始，裁剪将是一个假设的技能，你需要理解！

### **OpenCV 图像裁剪结果**

要使用 OpenCV 裁剪图像，请确保您已经进入本教程的 ***【下载】*** 部分，以访问源代码和示例图像。

从那里，打开一个 shell 并执行以下命令:

```py
$ python opencv_crop.py
```

您的裁剪输出应该与我在上一节中的输出相匹配。

## **总结**

在本教程中，您学习了如何使用 OpenCV 裁剪图像。因为 OpenCV 将图像表示为 NumPy 数组，所以裁剪就像将裁剪的开始和结束范围作为 NumPy 数组切片一样简单。

你需要做的就是记住下面的语法:

`cropped = image[startY:endY, startX:endX]`

只要记住提供起始和结束 *(x，y)*-坐标的顺序，用 OpenCV 裁剪图像就是轻而易举的事情！

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****