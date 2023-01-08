# OpenCV 加载图像(cv2.imread)

> 原文：<https://pyimagesearch.com/2021/01/20/opencv-load-image-cv2-imread/>

1.  从磁盘加载输入图像
2.  确定图像的宽度、高度和通道数
3.  将加载的图像显示到我们的屏幕上
4.  将图像作为不同的图像文件类型写回磁盘

在本指南结束时，您将会很好地理解如何使用 OpenCV 从磁盘加载图像。

**要了解如何使用 OpenCV 和** `cv2.imread` **，*从磁盘加载图像，请继续阅读。***

## **OpenCV 加载图像(cv2.imread)**

在本教程的第一部分，我们将讨论如何使用 OpenCV 和`cv2.imread`函数从磁盘加载图像。

从那里，您将学习如何配置您的开发环境来安装 OpenCV。

然后我们将回顾我们的项目目录结构，接着实现``load_image_opencv.py`` ，这是一个 Python 脚本，它将使用 OpenCV 和`cv2.imread`函数从磁盘加载输入图像。

我们将讨论我们的结果来结束本教程。

### 我们如何用 OpenCV 从磁盘加载图像？

`cv2.imread`函数接受一个参数，即图像在磁盘上的位置的路径:

```py
image = cv2.imread("path/to/image.png")
```

### **配置您的开发环境**

在使用 OpenCV 加载图像之前，您需要在系统上安装这个库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果你需要帮助配置 OpenCV 4.3+的开发环境，我*强烈推荐*阅读我的 [*pip 安装 OpenCV* 指南](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)**——它将在几分钟内让你启动并运行。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   准备好在您的 Windows、macOS 或 Linux 系统上运行代码*了吗*？

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

让我们回顾一下我们的项目结构。从 ***“下载”*** 部分获取源代码，解压缩内容，并导航到保存它的位置:

```py
$ tree . --dirsfirst
.
├── 30th_birthday.png
├── jurassic_park.png
├── load_image_opencv.py
└── newimage.jpg

0 directories, 4 files
```

现在让我们使用 OpenCV 实现我们的图像加载 Python 脚本！

### **实现我们的 OpenCV 图像加载脚本**

让我们开始学习如何使用 OpenCV 从磁盘加载输入图像。

创建一个名为`load_image_opencv.py`的 Python 脚本，并插入以下代码:

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

```py
# load the image from disk via "cv2.imread" and then grab the spatial
# dimensions, including width, height, and number of channels
image = cv2.imread(args["image"])
(h, w, c) = image.shape[:3]

# display the image width, height, and number of channels to our
# terminal
print("width: {} pixels".format(w))
print("height: {}  pixels".format(h))
print("channels: {}".format(c))
```

我们现在可以将图像尺寸(*宽度*、*高度*和*通道数量*)打印到终端进行查看(**第 18-20 行**)。

在未来的博客文章中，我们将讨论什么是图像通道，但现在请记住，彩色图像的通道数量将是三个，分别代表像素颜色的红色、绿色和蓝色(RGB)成分。

但是，如果我们不知道 OpenCV 是否正确地读取了图像，那么在内存中保存图像又有什么用呢？让我们在屏幕上显示图像进行验证:

```py
# show the image and wait for a keypress
cv2.imshow("Image", image)
cv2.waitKey(0)

# save the image back to disk (OpenCV handles converting image
# filetypes automatically)
cv2.imwrite("newimage.jpg", image)
```

现在，您应该能够将 OpenCV 应用于:

1.  从磁盘加载图像
2.  在屏幕上显示它
3.  将其写回磁盘

我们将在下一节回顾这些操作的一些结果。

### **OpenCV 图像加载结果**

现在是使用 OpenCV 从磁盘加载图像的时候了！

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

从那里，打开一个终端，并执行以下命令:

```py
$ python load_image_opencv.py --image 30th_birthday.png 
width: 720 pixels
height: 764  pixels
channels: 3
```

几年前，在我 30 岁生日的时候，我妻子租了一辆近似《侏罗纪公园》(我最喜欢的电影)的复制品，让我们开了一整天。

```py
$ ls
30th_birthday.png	jurassic_park.png	load_image_opencv.py	newimage.jpg
```

```py
$ python load_image_opencv.py --image jurassic_park.png 
width: 577 pixels
height: 433  pixels
channels: 3
```

继续本教程中的侏罗纪公园主题，这里我们有一张雷·阿诺(塞缪尔·L·杰克逊饰演)的照片。

该图像的宽度为 577 像素，高度为 433 像素，有三个通道。

说到最后一项，现在让我们试一试…

### 如果我们将一个无效的图像路径传递给“cv2.imread”会发生什么？

```py
$ python load_image_opencv.py --image path/does/not/exist.png
Traceback (most recent call last):
  File "load_image_opencv.py", line 17, in <module>
    (h, w, c) = image.shape[:3]
AttributeError: 'NoneType' object has no attribute 'shape'
```

在这里，我*特意*提供了一个**在我的磁盘上不存在**的镜像路径。

## **总结**

OpenCV 可以方便地读写各种图像文件格式(如 JPG、PNG、TIFF)。该库还简化了在屏幕上显示图像，并允许用户与打开的窗口进行交互。

如果 OpenCV 无法读取图像，您应该仔细检查输入的文件名是否正确，因为`cv2.imread`函数在失败时会返回一个``NoneType`` Python 对象。如果文件不存在或者 OpenCV 不支持图像格式，该函数将失败。

我们还根据底层 NumPy 数组形状的值将图像尺寸打印到终端(*宽度*、*高度*和*通道数量*)。然后，我们的脚本使用 JPG 格式将图像保存到磁盘，利用 OpenCV 的能力*自动将图像转换为期望的文件类型。*

在本系列的下一篇教程中，您将学习 OpenCV 图像基础知识，包括什么是像素、图像坐标系概述以及如何访问单个像素值。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***