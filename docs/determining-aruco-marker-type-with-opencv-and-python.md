# 用 OpenCV 和 Python 确定 ArUco 标记类型

> 原文：<https://pyimagesearch.com/2020/12/28/determining-aruco-marker-type-with-opencv-and-python/>

在本教程中，您将学习如何使用 OpenCV 和 Python 自动确定 ArUco 标记类型/字典。

今天的教程是我们关于 ArUco 标记生成和检测的三部分系列的最后一部分:

1.  *[用 OpenCV 和 Python 生成 ArUco 标记](https://pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/)* (两周前的教程)
2.  *[用 OpenCV](https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/)* 检测图像和视频中的阿鲁科标记(上周的帖子)
3.  *用 OpenCV 和 Python 自动确定 ArUco 标记类型*(今日教程)

到目前为止，在这个系列中，我们已经学习了如何生成和检测 ArUco 标记；然而，这些方法依赖于这样一个事实，即我们*已经知道*使用了什么类型的 ArUco 字典来生成标记。

这就提出了一个问题:

> *如果你**不知道**用来生成标记的阿鲁科字典会怎么样？*

如果不知道使用的 ArUco 字典，你将无法在你的图像/视频中发现它们。

当这种情况发生时，你需要一种能够自动确定图像中 ArUco 标记类型的方法——这正是我今天将向你展示的方法。

**要了解如何使用 OpenCV 自动确定 ArUco 标记类型/字典，*继续阅读。***

## **使用 OpenCV 和 Python 确定 ArUco 标记类型**

在本教程的第一部分，您将了解各种 ArUco 标记和 AprilTags。

从那里，您将实现一个 Python 脚本，它可以自动检测图像或视频流中是否存在任何类型的 ArUco 字典，从而允许您可靠地检测 ArUco 标记*,即使您不知道是用什么 ArUco 字典生成它们！*

然后，我们将回顾我们的工作成果，并讨论接下来的步骤(*提示:*我们将从下周开始做一些增强现实)。

### **ArUco 和 AprilTag 标记的类型**

[两周前我们学习了如何*生成*阿鲁科标记](https://pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/)，然后上周我们学习了如何 [*在图像和视频中检测*它们](https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/) — **但是如果我们还不知道我们正在使用的阿鲁科字典会怎么样呢？**

当你开发一个计算机视觉应用程序，而你没有自己生成 ArUco 标记时，就会出现这种情况。相反，这些标记可能是由另一个人或组织生成的(或者，你可能只需要一个通用算法来检测图像或视频流中的*任何*阿鲁科类型)。

**当这样的情况出现时，你需要能够*自动*推断出 ArUco 字典的类型。**

在撰写本文时，OpenCV 文库可以检测 21 种不同类型的 AruCo/AprilTag 标记。

以下代码片段显示了分配给每种类型的标记字典的唯一变量标识符:

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

在本教程的剩余部分，您将学习如何自动检查输入图像中是否存在这些 ArUco 类型。

要了解更多关于这些 ArUco 类型的信息，请参考[这篇文章](https://pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/)。

### **配置您的开发环境**

为了生成和检测 ArUco 标记，您需要安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果你需要帮助为 OpenCV 配置开发环境，我*强烈推荐*阅读我的 [*pip 安装 opencv* 指南](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)**——它将在几分钟内让你启动并运行。

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

首先使用本教程的 ***【下载】*** 部分下载源代码和示例图像。

从那里，让我们检查我们项目的目录结构:

```py
$ tree . --dirsfirst
.
├── images
│   ├── example_01.png
│   ├── example_02.png
│   └── example_03.png
└── guess_aruco_type.py

1 directory, 4 files
```

当您的任务是在图像/视频流中查找 ArUco 标签，但不确定使用什么 ArUco 字典来生成这些标签时，这样的脚本非常有用。

### **实施我们的 ArUco/AprilTag 标记类型标识符**

我们将为我们的自动 AruCo/AprilTag 类型标识符实现的方法有点像黑客，但我的感觉是，*黑客*只是一个在实践中有效的*启发式*。

有时抛弃优雅，取而代之的是得到该死的解决方案是可以的——这个脚本就是这种情况的一个例子。

打开项目目录结构中的``guess_aruco_type.py`` 文件，插入以下代码:

```py
# import the necessary packages
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image containing ArUCo tag")
args = vars(ap.parse_args())
```

我们在第 2-4 行导入我们需要的命令行参数，然后解析我们的命令行参数。

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

我在之前的教程 *[中介绍了 ArUco 字典的类型，包括它们的命名约定，使用 OpenCV 和 Python](https://pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/)* 生成 ArUco 标记。

如果你想了解更多关于 ArUco 字典的信息，请参考那里；否则，**简单理解一下，这个字典列出了 OpenCV 可以检测到的所有可能的 ArUco 标签。**

我们将彻底遍历这个字典，为每个条目加载 ArUco 检测器，然后将检测器应用到我们的输入图像。

如果我们找到了特定的标记类型，那么我们就知道 ArUco 标记存在于图像中。

说到这里，我们现在来实现这个逻辑:

```py
# load the input image from disk and resize it
print("[INFO] loading image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

# loop over the types of ArUco dictionaries
for (arucoName, arucoDict) in ARUCO_DICT.items():
	# load the ArUCo dictionary, grab the ArUCo parameters, and
	# attempt to detect the markers for the current dictionary
	arucoDict = cv2.aruco.Dictionary_get(arucoDict)
	arucoParams = cv2.aruco.DetectorParameters_create()
	(corners, ids, rejected) = cv2.aruco.detectMarkers(
		image, arucoDict, parameters=arucoParams)

	# if at least one ArUco marker was detected display the ArUco
	# name to our terminal
	if len(corners) > 0:
		print("[INFO] detected {} markers for '{}'".format(
			len(corners), arucoName))
```

在这种情况下，我们将在图像中找到的标签数量以及 ArUco 字典的名称记录到我们的终端，这样我们就可以在运行脚本后进一步调查。

如我所说，这个脚本没有多少“优雅”——它是一个彻头彻尾的黑客。但是没关系。有时候，你所需要的只是一个好的黑客来解除你的障碍，让你继续你的项目。

### **ArUco 标记类型识别结果**

让我们使用 ArUco 标记类型标识符吧！

确保使用本教程的 ***【下载】*** 部分下载源代码和示例图片到本文。

从那里，弹出打开一个终端，并执行以下命令:

```py
$ python guess_aruco_type.py --image images/example_01.png
[INFO] loading image...
[INFO] detected 2 markers for 'DICT_5X5_50'
[INFO] detected 5 markers for 'DICT_5X5_100'
[INFO] detected 5 markers for 'DICT_5X5_250'
[INFO] detected 5 markers for 'DICT_5X5_1000'
```

该图像包含五个 ArUco 图像示例(我们在 ArUco 标记的本系列第 1 部分[中生成)。](https://pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/)

ArUco 标记属于 *5×5* 类，id 分别高达 50、100、250 或 1000。这些结果意味着:

1.  我们知道一个*事实*这些是 *5×5* 标记。
2.  我们知道在这个图像中检测到的标记的 id 为 *< 50。*
3.  但是，如果在其他图像中有更多的标记，我们可能会遇到值为 *> 50 的 ArUco *5×5* 标记。*
4.  如果我们使用的*只是*这张图片，那么假设`DICT_5X5_50`是安全的，但是如果我们有更多的图片，继续调查并找到*最小的* ArUco 字典，将所有唯一的 id 放入其中。

让我们尝试另一个示例图像:

```py
$ python guess_aruco_type.py --image images/example_02.png
[INFO] loading image...
[INFO] detected 1 markers for 'DICT_4X4_50'
[INFO] detected 1 markers for 'DICT_4X4_100'
[INFO] detected 1 markers for 'DICT_4X4_250'
[INFO] detected 1 markers for 'DICT_4X4_1000'
[INFO] detected 4 markers for 'DICT_ARUCO_ORIGINAL'
```

这里你可以看到一个包含 [Pantone 颜色匹配卡的示例图像。](https://www.pantone.com/pantone-color-match-card) OpenCV(错误地)认为这些标记*可能*属于 *4×4* 类，但是如果你放大示例图像，你会发现这不是真的，因为这些实际上是 *6×6* 标记，标记周围有一点额外的填充。

```py
$ python guess_aruco_type.py --image images/example_03.png
[INFO] loading image...
[INFO] detected 3 markers for 'DICT_APRILTAG_36h11'
```

在这里，OpenCV 可以推断出我们最有可能看到的是 AprilTags。

我希望你喜欢这一系列关于 ArUco 标记和 AprilTags 的教程！

在接下来的几周里，我们将开始研究 ArUco 标记的实际应用，包括如何将它们融入我们自己的计算机视觉和图像处理管道。

## **总结**

在本教程中，您学习了如何*自动*确定 ArUco 标记类型， ***即使您不知道最初使用的是什么 ArUco 字典！***

我们的方法有点麻烦，因为它要求我们遍历所有可能的 ArUco 字典，然后尝试在输入图像中检测特定的 ArUco 字典。

也就是说，我们的黑客技术*有效，*所以很难反驳。

请记住,“黑客”没有任何问题。正如我喜欢说的，hack 只是一种有效的启发式方法。

从下周开始，你将看到应用 ArUco 检测的真实世界示例，包括增强现实。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***