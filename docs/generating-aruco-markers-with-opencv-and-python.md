# 用 OpenCV 和 Python 生成 ArUco 标记

> 原文：<https://pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/>

在本教程中，您将学习如何使用 OpenCV 和 Python 生成 ArUco 标记。

今天的博文是我们关于 ArUCo 标记和基准的三部分系列的第一部分:

1.  *用 OpenCV 和 Python 生成 ArUco 标记*(今天的帖子)
2.  *用 OpenCV 检测图像和视频中的 ArUco 标记*(下周教程)
3.  *使用 OpenCV* 自动确定 ArUco 标记类型(两周后的博客文章)

与 [AprilTags](https://pyimagesearch.com/2020/11/02/apriltag-with-python/) 类似，ArUco 标记是计算机视觉算法可以轻松检测到的 2D 二进制模式。

通常，我们将 AprilTags 和 ArUco 标记用于:

*   摄像机标定
*   物体尺寸估计
*   测量相机和物体之间的距离
*   3D 位置
*   面向对象
*   机器人和自主导航
*   *等。*

**与 AprilTags 相比，使用 ArUco 标记的主要优势包括:**

在这个 ArUco 标记的介绍系列中，您将学习如何生成它们，在图像和实时视频流中检测它们，甚至如何*自动*检测图像中 ArUco 标记的类型(即使您*不*知道正在使用哪种类型的标记)。

然后，在未来的 PyImageSearch 教程中，我们将利用这些知识，在我们自己的计算机视觉和图像处理管道中使用 ArUco 标记。

***更新于 2021 年 11 月 25 日，附有在白色背景上使用 ArUco 标记的说明。***

**要学习如何用 OpenCV 和 Python 生成 ArUco 标记，*继续阅读。***

## **用 OpenCV 和 Python 生成 ArUco 标记**

在本教程的第一部分，我们将讨论 ArUco 标记，包括它们是什么，以及为什么我们可能要在我们的计算机视觉和图像处理管道中使用它们。

然后我们将讨论如何使用 OpenCV 和 Python 生成 ArUco 标记。如果您不想编写代码来生成 ArUco 标记，我还会提供几个示例网站，它们会为您生成 ArUco 标记(尽管代码实现本身非常简单)。

从这里，我们将回顾我们的项目目录结构，然后实现一个名为`opencv_generate_aruco.py`的 Python 脚本，它将生成一个特定的 ArUco 映像，然后将其保存到磁盘。

我们将讨论我们的结果来结束本教程。

### 什么是 ArUco 标记？

在之前的教程中，我已经介绍了基准标记、AprilTags 和 ArUco 标记[的基础知识，所以我不打算在这里重复这些基础知识。](https://pyimagesearch.com/2020/11/02/apriltag-with-python/)

如果你是基准标记的新手，需要了解它们为什么重要，它们是如何工作的，或者我们什么时候想在计算机视觉/图像处理管道中使用它们，我建议你读一下我的 [AprilTag 教程](https://pyimagesearch.com/2020/11/02/apriltag-with-python/)。

从那里你应该回到这里，用 OpenCV 完成 ArUco 标记的教程。

### **如何用 OpenCV 和 Python 生成 ArUco 标记？**

**OpenCV 库通过其** `cv2.aruco.drawMarker` **函数内置了 ArUco 标记生成器。**

该函数的参数包括:

*   ``dictionary``:ArUco 字典，指定我们正在使用的标记类型
*   ``id`` :我们将要绘制的标记的 id(必须是 ArUco `dictionary`中的有效 ID)
*   ``sidePixels`` :我们将在其上绘制 ArUco 标记的(正方形)图像的像素大小
*   ``borderBits`` :边框的宽度和高度(像素)

然后，``drawMarker`` 函数返回绘制了 ArUco 标记的输出图像。

正如您将在本教程后面看到的，使用该函数在实践中相当简单。所需的步骤包括:

1.  选择您想要使用的 ArUco 词典
2.  指定您要抽取的 ArUco ID
3.  为输出 ArUco 图像分配内存(以像素为单位)
4.  使用``drawMarker`` 功能绘制 ArUco 标签
5.  画阿鲁科标记本身

也就是说，如果你不想写任何代码，你可以利用在线 ArUco 生成器。

### 有在线 ArUco 标记生成器吗？

如果你不想写一些代码，或者只是赶时间，你可以使用在线 ArUco 标记生成器。

我最喜欢的是奥列格·卡拉切夫的这张。

你要做的就是:

1.  选择您想要使用的 ArUco 词典
2.  输入标记 ID
3.  指定标记大小(以毫米为单位)

在那里，您可以将 ArUco 标记保存为 SVG 文件或 PDF，打印它，然后在您自己的 OpenCV 和计算机视觉应用程序中使用它。

### 什么是 ArUco 字典？

到目前为止，在本教程中，我已经提到了“阿鲁科字典”的概念，但是到底什么是阿鲁科字典呢？以及在 ArUco 生成和检测中起到什么作用？

**简单的回答是，ArUco 字典指定了我们正在生成和检测的 ArUco 标记的类型。如果没有字典，我们将无法生成和检测这些标记。**

想象你被绑架，蒙上眼睛，被带上飞机，然后被丢在世界上一个随机的国家。然后你得到一个笔记本，里面有你释放的秘密，但它是用你一生中从未见过的语言写的。

一个俘获者同情你，给你一本字典来帮助你翻译你在书中看到的内容。

使用字典，你能够翻译文件，揭示秘密，并完好无损地逃离你的生活。

但是如果没有那本字典，你将永远无法逃脱*。正如你需要那本字典来解释你逃跑的秘密一样，我们必须知道我们正在使用哪种 ArUco 标记来生成和检测它们。*

 *### **OpenCV 中 ArUco 字典的类型**

OpenCV 库中内置了 21 种不同的 ArUco 词典。我在下面的 Python 字典中列出了它们:

```py
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
```

这些字典中的大多数遵循特定的命名约定，`cv2.aruco.DICT_NxN_M`，大小为 *NxN* ，后跟一个整数值， *M* — ***，但是这些值是什么意思呢？***

*NxN* 值是 ArUco 标记的 2D *位大小*。例如，对于一个 *6×6* 标记，我们总共有 36 位。

网格大小后面的整数 *M* 指定可以用该字典生成的唯一 ArUco IDs 的总数。

为了使命名约定更加具体，请考虑以下示例:

那么，你如何决定使用哪一个 ArUco 标记字典呢？

1.  首先，考虑字典中需要多少个唯一值。只需要一小把马克笔？那么选择一个唯一值数量较少的字典。需要检测很多标志物？选择具有更多唯一 ID 值的字典。本质上，选择一本有你需要的最少数量的*身份证的字典——**不要拿超过你实际需要的。***
2.  **查看您的输入图像/视频分辨率大小。**请记住，您的网格尺寸越大，您的相机拍摄的 ArUco 标记就需要越大。如果你有一个*大的*网格，但是有一个*低分辨率的*输入，那么这个标记可能是不可检测的(或者可能被误读)。
3.  **考虑标记间距离。** OpenCV 的 ArUco 检测实现利用误差校正来提高标记检测的准确性和鲁棒性。误差校正依赖于*标记间距离的概念。* **较小的字典尺寸与较大的 *NxN* 标记尺寸增加了标记间的距离，从而使它们不容易出现错误读数。**

**ArUco 字典的理想设置包括:**

1.  需要生成和读取的少量唯一 ArUco IDs
2.  包含将被检测的 ArUco 标记的高质量图像输入
3.  更大的 *NxN* 网格尺寸，与少量唯一 ArUco IDs 相平衡，这样标记间的距离可用于纠正误读的标记

关于 ArUco 字典的更多细节，请务必参考 OpenCV 文档。

***注意:**我将通过说* ``ARUCO_DICT`` *变量中的最后几个条目表明我们也可以生成和检测 [AprilTags](https://pyimagesearch.com/2020/11/02/apriltag-with-python/) 来结束这一节！*

### **配置您的开发环境**

为了生成和检测 ArUco 标记，您需要安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果你需要帮助配置 OpenCV 的开发环境，我*强烈推荐*阅读我的 [*pip 安装 OpenCV 指南*](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)**——它将在几分钟内让你启动并运行。

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

在我们开始用 OpenCV 生成 ArUco 标记之前，让我们先回顾一下我们的项目目录结构。

使用本教程的 ***“下载”*** 部分将源代码和示例图像下载到本教程。从那里，让我们检查我们有什么:

```py
$ tree . --dirsfirst
.
├── tags
│   ├── DICT_5X5_100_id24.png
│   ├── DICT_5X5_100_id42.png
│   ├── DICT_5X5_100_id66.png
│   ├── DICT_5X5_100_id70.png
│   └── DICT_5X5_100_id87.png
└── opencv_generate_aruco.py

1 directory, 6 files
```

顾名思义，`opencv_generate_aruco.py`脚本用于生成 ArUco 标记。然后将生成的 ArUco 标记保存到任务的`tags/`目录中。

下周我们将学习如何真正地*检测*和*识别*这些(和其他)阿鲁科标记。

### **使用 OpenCV 和 Python 实现我们的 ArUco 标记生成脚本**

让我们学习如何用 OpenCV 生成 ArUco 标记。

打开项目目录结构中的`opencv_generate_aruco.py`文件，并插入以下代码:

```py
# import the necessary packages
import numpy as np
import argparse
import cv2
import sys
```

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output image containing ArUCo tag")
ap.add_argument("-i", "--id", type=int, required=True,
	help="ID of ArUCo tag to generate")
ap.add_argument("-t", "--type", type=str,
	default="DICT_ARUCO_ORIGINAL",
	help="type of ArUCo tag to generate")
args = vars(ap.parse_args())
```

```py
# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}
```

我在上面的*“OpenCV 中 ArUco 字典的类型”*一节中回顾了 ArUco 字典，所以如果您想了解关于这个代码块的更多解释，请务必参考那里。

定义了`ARUCO_DICT`映射后，现在让我们使用 OpenCV 加载 ArUco 字典:

```py
# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUCo tag of '{}' is not supported".format(
		args["type"]))
	sys.exit(0)

# load the ArUCo dictionary
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
```

```py
# allocate memory for the output ArUCo tag and then draw the ArUCo
# tag on the output image
print("[INFO] generating ArUCo tag type '{}' with ID '{}'".format(
	args["type"], args["id"]))
tag = np.zeros((300, 300, 1), dtype="uint8")
cv2.aruco.drawMarker(arucoDict, args["id"], 300, tag, 1)

# write the generated ArUCo tag to disk and then display it to our
# screen
cv2.imwrite(args["output"], tag)
cv2.imshow("ArUCo Tag", tag)
cv2.waitKey(0)
```

**第 57 行**为一张 *300x300x1* 灰度图像分配内存。我们在这里使用灰度，因为 ArUco 标签是一个*二进制*图像。

此外，您可以使用任何您想要的图像尺寸。我在这里硬编码了 300 个像素，但是同样，你可以根据自己的项目随意增减分辨率。

### **OpenCV ArUco 生成结果**

我们现在准备用 OpenCV 生成 ArUco 标记！

首先使用本教程的 ***【下载】*** 部分下载源代码和示例图像。

从那里，打开一个终端，并执行以下命令:

```py
$ python opencv_generate_aruco.py --id 24 --type DICT_5X5_100 \
	--output tags/DICT_5X5_100_id24.png
[INFO] generating ArUCo tag type 'DICT_5X5_100' with ID '24'
```

```py
$ python opencv_generate_aruco.py --id 42 --type DICT_5X5_100 \
	--output tags/DICT_5X5_100_id42.png
[INFO] generating ArUCo tag type 'DICT_5X5_100' with ID '42'
```

```py
$ python opencv_generate_aruco.py --id 66 --type DICT_5X5_100 \
	--output tags/DICT_5X5_100_id66.png
[INFO] generating ArUCo tag type 'DICT_5X5_100' with ID '66'
```

```py
$ python opencv_generate_aruco.py --id 87 --type DICT_5X5_100 \
	--output tags/DICT_5X5_100_id87.png
[INFO] generating ArUCo tag type 'DICT_5X5_100' with ID '87'
```

```py
$ python opencv_generate_aruco.py --id 70 --type DICT_5X5_100 \
	--output tags/DICT_5X5_100_id70.png
[INFO] generating ArUCo tag type 'DICT_5X5_100' with ID '70'
```

此时，我们已经生成了五个 ArUco 标记，这是我在下面创建的一个蒙太奇:

但那又怎样？马克笔放在我们的磁盘上没什么用。

**我们如何获取这些标记，然后*在图像和实时视频流中检测*？**

我将在下周的教程中讨论这个问题。

敬请关注。

**最后一点，如果你想自己创建 ArUco 标记，你需要将本教程中的 ArUco 标记放在白色背景上，以确保代码在下一篇博文中正常工作。**

如果你的 ArUco 马克笔没有白色背景，我不能保证它能正常工作。

## **总结**

在本教程中，您学习了如何使用 OpenCV 和 Python 生成 ArUco 标记。

在 OpenCV 中使用 ArUco 标签非常简单，因为 OpenCV 库中内置了方便的``cv2.aruco`` 子模块(即，您不需要任何额外的 Python 包或依赖项来检测 ArUco 标签)。

现在我们已经实际上*生成了*一些 ArUco 标签，下周我将向你展示如何获取生成的标签，并实际上*在图像和实时视频流中检测*它们。

在本系列教程结束时，您将拥有在自己的 OpenCV 项目中自信而成功地使用 ArUco 标签所必需的知识。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****