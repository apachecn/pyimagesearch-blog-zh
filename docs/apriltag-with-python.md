# Python 的 AprilTag

> 原文：<https://pyimagesearch.com/2020/11/02/apriltag-with-python/>

在本教程中，您将学习如何使用 Python 和 OpenCV 库执行 AprilTag 检测。

AprilTags 是一种**基准标记。**基准点，或更简单的“标记”，是**参照物**，当图像或视频帧被捕获时，这些参照物被放置在摄像机的视野中。

然后，在后台运行的计算机视觉软件获取输入图像，检测基准标记，并基于标记的*类型*和标记在输入图像中所处的的*执行一些操作。*

AprilTags 是一种*特定的*类型的基准标记，**由一个以特定图案**生成的黑色正方形和白色前景组成(如本教程顶部的图所示)。

标记周围的黑色边框使计算机视觉和图像处理算法更容易在各种情况下检测 AprilTags，包括旋转、缩放、光照条件等的变化。

你可以从概念上认为 AprilTag 类似于 QR 码，这是一种可以使用计算机视觉算法检测的 2D 二进制模式。然而，AprilTag 只能存储 4-12 位数据，比 QR 码少几个数量级(典型的 QR 码最多可存储 3KB 数据)。

那么，为什么还要使用 AprilTags 呢？如果 AprilTags 保存的数据如此之少，为什么不直接使用二维码呢？

AprilTags 存储更少数据的事实实际上是一个*特性*，而不是一个*缺陷/限制。*套用[官方 AprilTag 文档](https://april.eecs.umich.edu/software/apriltag)，**由于 AprilTag 有效载荷如此之小，它们可以更容易*检测到*，更稳健*识别到*，并且在*更长的范围内不太难检测到。***

基本上，如果你想在 2D 条形码中存储数据，使用二维码。但是，如果您需要使用更容易在计算机视觉管道中检测到的标记，请使用 AprilTags。

诸如 AprilTags 的基准标记是许多计算机视觉系统的组成部分，包括但不限于:

*   摄像机标定
*   物体尺寸估计
*   测量相机和物体之间的距离
*   3D 定位
*   面向对象
*   机器人技术(即自主导航至特定标记)
*   *等。*

AprilTags 的主要优点之一是可以使用基本软件和打印机创建。只需在您的系统上生成 AprilTag，将其打印出来，并将其包含在您的图像处理管道中— **Python 库的存在是为了*自动*为您检测 April tag！**

在本教程的剩余部分，我将向您展示如何使用 Python 和 OpenCV 检测 AprilTags。

**要学习如何用 OpenCV 和 Python 检测 AprilTags，*继续阅读。***

## Python 的 AprilTag

在本教程的第一部分，我们将讨论什么是 AprilTags 和基准标记。然后我们将安装 [apriltag](https://pypi.org/project/apriltag/) ，我们将使用 Python 包来检测输入图像中的 apriltag。

接下来，我们将回顾我们的项目目录结构，然后实现用于检测和识别 AprilTags 的 Python 脚本。

我们将通过回顾我们的结果来结束本教程，包括讨论与 AprilTags 具体相关的一些限制(和挫折)。

### 什么是 AprilTags 和基准标记？

AprilTags 是一种基准标记。基准点是我们放置在摄像机视野中的特殊标记，这样它们很容易被识别。

例如，以下所有教程都使用基准标记来测量图像中某个对象的*大小*或特定对象之间的*距离*:

*   *[使用 Python 和 OpenCV](https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/)* 找到相机到物体/标记的距离
*   *[用 OpenCV](https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/) 测量图像中物体的尺寸*
*   *[用 OpenCV](https://pyimagesearch.com/2016/04/04/measuring-distance-between-objects-in-an-image-with-opencv/) 测量图像中物体间的距离*

成功实施这些项目的唯一可能是因为在摄像机的视野中放置了一个标记/参考对象。一旦我检测到物体，我就可以推导出其他物体的宽度和高度，因为我已经知道参考物体的尺寸。

**AprilTags 是一种*特殊的*类型的基准标记。**这些标记具有以下特性:

1.  它们是具有二进制值的正方形。
2.  背景是“黑色”
3.  前景是以“白色”显示的生成的图案
4.  图案周围有黑边，因此更容易被发现。
5.  它们几乎可以以任何尺寸生成。
6.  一旦生成，就可以打印出来并添加到您的应用程序中。

一旦在计算机视觉管道中检测到，AprilTags 可用于:

*   摄像机标定
*   3D 应用
*   猛击
*   机器人学
*   自主导航
*   物体尺寸测量
*   距离测量
*   面向对象
*   *…还有更多！*

使用基准的一个很好的例子是在一个大型的履行仓库(如亚马逊)，在那里你使用自动叉车。

你可以在地板上放置一个标签来定义叉车行驶的“车道”。可以在大货架上放置特定的标记，这样叉车就知道要拉下哪个板条箱。

标记甚至可以用于“紧急停机”,如果检测到“911”标记，叉车会自动停止、暂停操作并停机。

AprilTags 和密切相关的 ArUco tags 的用例数量惊人。在本教程中，我将讲述如何检测 AprilTags 的基础知识。PyImageSearch 博客上的后续教程将在此基础上构建，并向您展示如何使用它们实现真实世界的应用程序。

### 在系统上安装“April tag”Python 包

为了检测图像中的 AprilTag，我们首先需要安装一个 Python 包来促进 April tag 检测。

我们将使用的库是 [apriltag](https://pypi.org/project/apriltag/) ，幸运的是，它可以通过 pip 安装。

首先，确保你按照我的 [*pip 安装 opencv* 指南](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)在你的系统上安装 opencv。

如果您正在使用 Python 虚拟环境(这是我推荐的，因为这是 Python 的最佳实践)，请确保使用`workon`命令访问您的 Python 环境，然后将`apriltag`安装到该环境中:

```py
$ workon your_env_name
$ pip install apriltag
```

从那里，验证您可以将`cv2`(您的 OpenCV 绑定)和`apriltag`(您的 AprilTag 检测器库)导入到您的 Python shell 中:

```py
$ python
>>> import cv2
>>> import apriltag
>>> 
```

祝贺您在系统上安装了 OpenCV 和 AprilTag！

### 配置您的开发环境有问题吗？

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在你的 Windows、macOS 或 Linux 系统上运行代码*了吗？***

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！在你的浏览器中获得运行在 **Google Colab 生态系统上的 PyImageSearch 教程 **Jupyter 笔记本**！**无需安装。

最棒的是，这些笔记本可以在 Windows、macOS 和 Linux 上运行！

### 项目结构

在我们实现 Python 脚本来检测图像中的 AprilTags 之前，让我们先回顾一下我们的项目目录结构:

```py
$ tree . --dirsfirst
.
├── images
│   ├── example_01.png
│   └── example_02.png
└── detect_apriltag.py

1 directory, 3 files
```

### 用 Python 实现 AprilTag 检测

```py
# import the necessary packages
import apriltag
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image containing AprilTag")
args = vars(ap.parse_args())
```

接下来，让我们加载输入图像并对其进行预处理:

```py
# load the input image and convert it to grayscale
print("[INFO] loading image...")
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

**第 14 行**使用提供的`--image`路径从磁盘加载我们的输入图像。然后，我们将图像转换为*灰度*，这是 AprilTag 检测所需的唯一预处理步骤。

说到 AprilTag 检测，现在让我们继续执行检测步骤:

```py
# define the AprilTags detector options and then detect the AprilTags
# in the input image
print("[INFO] detecting AprilTags...")
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)
results = detector.detect(gray)
print("[INFO] {} total AprilTags detected".format(len(results)))
```

为了检测图像中的 AprilTag，我们首先需要指定`options`，更具体地说，是 **AprilTag 家族:**

AprilTags 中的**族**定义了 AprilTag 检测器将在输入图像中采用的标签集。标准/默认 AprilTag 系列为“tag 36h 11”；然而，AprilTags 总共有六个家庭:

1.  Tag36h11
2.  工作日标准 41h12
3.  工作日标准 52 小时 13 分
4.  TagCircle21h7
5.  TagCircle49h12
6.  TagCustom48h12

你可以在官方 AprilTag 网站上阅读更多关于 AprilTag 家族的信息，但在大多数情况下，你通常会使用“Tag36h11”。

**第 20 行**用默认的 AprilTag 家族`tag36h11`初始化我们的`options`。

这里的最后一步是遍历 AprilTags 并显示结果:

```py
# loop over the AprilTag detection results
for r in results:
	# extract the bounding box (x, y)-coordinates for the AprilTag
	# and convert each of the (x, y)-coordinate pairs to integers
	(ptA, ptB, ptC, ptD) = r.corners
	ptB = (int(ptB[0]), int(ptB[1]))
	ptC = (int(ptC[0]), int(ptC[1]))
	ptD = (int(ptD[0]), int(ptD[1]))
	ptA = (int(ptA[0]), int(ptA[1]))

	# draw the bounding box of the AprilTag detection
	cv2.line(image, ptA, ptB, (0, 255, 0), 2)
	cv2.line(image, ptB, ptC, (0, 255, 0), 2)
	cv2.line(image, ptC, ptD, (0, 255, 0), 2)
	cv2.line(image, ptD, ptA, (0, 255, 0), 2)

	# draw the center (x, y)-coordinates of the AprilTag
	(cX, cY) = (int(r.center[0]), int(r.center[1]))
	cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

	# draw the tag family on the image
	tagFamily = r.tag_family.decode("utf-8")
	cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	print("[INFO] tag family: {}".format(tagFamily))

# show the output image after AprilTag detection
cv2.imshow("Image", image)
cv2.waitKey(0)
```

我们开始在第 26 行的**上循环我们的 AprilTag 检测。**

每个 AprilTag 由一组`corners`指定。**第 29-33 行**提取 AprilTag 正方形的四个角，**第 36-39 行**在`image`上绘制 AprilTag 包围盒。

我们还计算 AprilTag 边界框的中心 *(x，y)*-坐标，然后画一个代表 AprilTag 中心的圆(**第 42 行和第 43 行**)。

我们将执行的最后一个注释是从结果对象中抓取检测到的`tagFamily`,然后将其绘制在输出图像上。

最后，我们通过显示 AprilTag 检测的结果来结束我们的 Python。

### AprilTag Python 检测结果

让我们来测试一下 Python AprilTag 检测器吧！确保使用本教程的 ***“下载”*** 部分下载源代码和示例图像。

从那里，打开一个终端，并执行以下命令:

```py
$ python detect_apriltag.py --image images/example_01.png 
[INFO] loading image...
[INFO] detecting AprilTags...
[INFO] 1 total AprilTags detected
[INFO] tag family: tag36h11
```

尽管 AprilTag 已经被旋转，我们仍然能够在输入图像中检测到它，从而证明 April tag 具有一定程度的鲁棒性，使它们更容易被检测到。

让我们试试另一张图片，这张图片有*多个* AprilTags:

```py
$ python detect_apriltag.py --image images/example_02.png 
[INFO] loading image...
[INFO] detecting AprilTags...
[INFO] 5 total AprilTags detected
[INFO] tag family: tag36h11
[INFO] tag family: tag36h11
[INFO] tag family: tag36h11
[INFO] tag family: tag36h11
[INFO] tag family: tag36h11
```

这里我们有一个无人驾驶的车队，每辆车上都有一个标签。**我们能够检测输入图像中的所有 AprilTag，*除了被其他机器人***部分遮挡的(这是有意义的——整个*April tag 必须在我们的视野中才能检测到它；遮挡给许多基准标记带来了一个大问题。*

 *当您需要在自己的输入图像中检测 AprilTags 时，请确保使用此代码作为起点！

### 局限和挫折

您可能已经注意到，我*没有*介绍如何手动生成您自己的 AprilTag 图像。这有两个原因:

1.  所有 AprilTag 家族中所有可能的 April tag 都可以从官方 AprilRobotics repo 下载。
2.  此外， [AprilTags repo 包含 Java 源代码，您可以用它来生成自己的标记。](https://github.com/AprilRobotics/apriltag-generation)
3.  如果你*真的*想深入兔子洞， [TagSLAM 库](https://berndpfrommer.github.io/tagslam_web/)包含一个特殊的 Python 脚本，可以用来生成标签——你可以在这里阅读关于这个脚本[的更多信息。](https://berndpfrommer.github.io/tagslam_web/making_tags/)

综上所述，我发现生成 AprilTags 是一件痛苦的事情。相反，我更喜欢使用 ArUco 标签，OpenCV 既可以使用它的`cv2.aruco`子模块 ***检测*** 和 ***生成*** 。

我将在 2020 年末/2021 年初的一个教程中向你展示如何使用`cv2.aruco`模块来检测*T2 的 AprilTags 和 ArUco 标签。一定要继续关注那个教程！*

### 信用

在本教程中，我们使用了来自其他网站的 AprilTags 的示例图像。我想花一点时间感谢官方 AprilTag 网站以及来自 [TagSLAM 文档](https://berndpfrommer.github.io/tagslam_web/making_tags/)的 [Bernd Pfrommer](http://pfrommer.us/) 提供的 April tag 示例。

## 摘要

在本教程中，您学习了 AprilTags，这是一组常用于机器人、校准和 3D 计算机视觉项目的基准标记。

我们在这些情况下使用 AprilTags(以及密切相关的 ArUco tags ),因为它们易于实时检测。存在库来检测几乎任何用于执行计算机视觉的编程语言中的 AprilTags 和 ArUco 标记，包括 Python、Java、C++等。

在我们的例子中，我们使用了[四月标签 Python 包](https://pypi.org/project/apriltag/)。这个包是 pip 可安装的，允许我们传递 OpenCV 加载的图像，使它在许多基于 Python 的计算机视觉管道中非常有效。

今年晚些时候/2021 年初，我将向您展示使用 AprilTags 和 ArUco 标记的真实项目，但我想现在介绍它们，以便您有机会熟悉它们。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****