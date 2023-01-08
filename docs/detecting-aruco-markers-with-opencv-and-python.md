# 用 OpenCV 和 Python 检测 ArUco 标记

> 原文：<https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/>

在本教程中，您将学习如何使用 OpenCV 和 Python 检测图像和实时视频流中的 ArUco 标记。

这篇博文是我们关于 ArUco 标记和基准的三部分系列的第二部分:

1.  *[用 OpenCV 和 Python 生成 ArUco 标记](https://pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/)* (上周帖子)
2.  *使用 OpenCV 检测图像和视频中的 ArUco 标记*(今天的教程)
3.  *使用 OpenCV* 自动确定 ArUco 标记类型(下周发布)

上周我们学习了:

*   什么是阿鲁科字典
*   如何选择适合我们任务的 ArUco 字典
*   如何使用 OpenCV 生成 ArUco 标记
*   如何使用在线工具创建 ArUco 标记

今天我们将学习如何使用 OpenCV 实际检测 ArUco 标记。

**要了解如何使用 OpenCV 检测图像和实时视频中的 ArUco 标记，*请继续阅读。***

## **用 OpenCV 和 Python 检测 ArUco 标记**

从这里，我们将回顾我们的项目目录结构，并实现两个 Python 脚本:

1.  一个 Python 脚本来检测图像中的 ArUco 标记
2.  和另一个 Python 脚本来检测实时视频流中的 ArUco 标记

我们将用对我们结果的讨论来结束这篇关于使用 OpenCV 进行 ArUco 标记检测的教程。

### **OpenCV ArUCo 标记检测**

正如我在上周的教程中所讨论的，OpenCV 库带有内置的 ArUco 支持，既支持*生成* ArUco 标记，也支持*检测*它们。

### 了解*" cv2 . aruco . detect markers "*函数

我们可以用 3-4 行代码定义一个 ArUco 标记检测程序:

```py
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
	parameters=arucoParams)
```

### **配置您的开发环境**

为了生成和检测 ArUco 标记，您需要安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果你需要帮助配置 OpenCV 4.3+的开发环境，我*强烈推荐*阅读我的 [*pip 安装 opencv* 指南](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)**——它将在几分钟内让你启动并运行。

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

在我们学习如何检测图像中的 ArUco 标签之前，让我们先回顾一下我们的项目目录结构，这样您就可以很好地了解我们的项目是如何组织的，以及我们将使用哪些 Python 脚本。

首先使用本教程的 ***【下载】*** 部分下载源代码和示例图像。

从那里，我们可以检查项目目录:

```py
$ tree . --dirsfirst
.
├── images
│   ├── example_01.png
│   └── example_02.png
├── detect_aruco_image.py
└── detect_aruco_video.py

2 directories, 9 files
```

回顾了我们的项目目录结构后，我们可以继续用 OpenCV 实现 ArUco 标签检测了！

### **使用 OpenCV 在图像中检测 ArUco 标记**

准备好学习如何使用 OpenCV 检测图像中的 ArUco 标签了吗？

打开项目目录中的``detect_aruco_image.py`` 文件，让我们开始工作:

```py
# import the necessary packages
import argparse
import imutils
import cv2
import sys
```

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image containing ArUCo tag")
ap.add_argument("-t", "--type", type=str,
	default="DICT_ARUCO_ORIGINAL",
	help="type of ArUCo tag to detect")
args = vars(ap.parse_args())
```

**因此，您必须绝对确定用于*生成*ArUco 标签的类型与您用于*检测*阶段的类型相同。**

***注:**不知道你输入图片中的标签是用什么 ArUco 字典生成的？别担心，我会掩护你的。下周我将向你展示我个人收藏的一个 Python 脚本，当我不能识别一个给定的 ArUco 标签是什么类型时，我就使用它。这个脚本**自动识别**ArUco 标签类型。请继续关注下周的教程，在那里我将详细回顾它。*

接下来是我们的``ARUCO_DICT`` ，它列举了 OpenCV 支持的每个 ArUco 标签类型:

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

这个字典的*键*是一个人类可读的字符串(即 ArUco 标签类型的名称)。然后这个键映射到*值*，这是 OpenCV 对于 ArUco 标签类型的唯一标识符。

使用这个字典，我们可以获取输入的`--type`命令行参数，通过`ARUCO_DICT`传递它，然后获得 ArUco 标签类型的唯一标识符。

以下 Python shell 块显示了如何执行查找操作的简单示例:

```py
>>> print(args)
{'type': 'DICT_5X5_100'}
>>> arucoType = ARUCO_DICT[args["type"]]
>>> print(arucoType)
5
>>> 5 == cv2.aruco.DICT_5X5_100
True
>>> 
```

我介绍了 ArUco 字典的类型，包括它们的命名约定，在我之前的教程中， *[使用 OpenCV 和 Python](https://pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/)* 生成 ArUco 标记。

如果你想了解更多关于阿鲁科字典的信息，请参考那里；否则，**简单理解一下，这个字典列出了 OpenCV 可以检测到的所有可能的 ArUco 标签。**

接下来，让我们继续从磁盘加载输入图像:

```py
# load the input image from disk and resize it
print("[INFO] loading image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUCo tag of '{}' is not supported".format(
		args["type"]))
	sys.exit(0)

# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
	parameters=arucoParams)

```

```py
# verify *at least* one ArUco marker was detected
if len(corners) > 0:
	# flatten the ArUco IDs list
	ids = ids.flatten()

	# loop over the detected ArUCo corners
	for (markerCorner, markerID) in zip(corners, ids):
		# extract the marker corners (which are always returned in
		# top-left, top-right, bottom-right, and bottom-left order)
		corners = markerCorner.reshape((4, 2))
		(topLeft, topRight, bottomRight, bottomLeft) = corners

		# convert each of the (x, y)-coordinate pairs to integers
		topRight = (int(topRight[0]), int(topRight[1]))
		bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
		bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
		topLeft = (int(topLeft[0]), int(topLeft[1]))
```

```py
		# draw the bounding box of the ArUCo detection
		cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
		cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
		cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
		cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

		# compute and draw the center (x, y)-coordinates of the ArUco
		# marker
		cX = int((topLeft[0] + bottomRight[0]) / 2.0)
		cY = int((topLeft[1] + bottomRight[1]) / 2.0)
		cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

		# draw the ArUco marker ID on the image
		cv2.putText(image, str(markerID),
			(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 255, 0), 2)
		print("[INFO] ArUco marker ID: {}".format(markerID))

		# show the output image
		cv2.imshow("Image", image)
		cv2.waitKey(0)
```

最终的输出可视化显示在我们屏幕的第 98 和 99 行。

### **OpenCV ArUco 标记检测结果**

让我们使用 OpenCV ArUco 检测器吧！

使用本教程的 ***“下载”*** 部分下载源代码和示例图像。

从那里，您可以执行以下命令:

```py
$ python detect_aruco_image.py --image images/example_01.png --type DICT_5X5_100
[INFO] loading image...
[INFO] detecting 'DICT_5X5_100' tags...
[INFO] ArUco marker ID: 42
[INFO] ArUco marker ID: 24
[INFO] ArUco marker ID: 70
[INFO] ArUco marker ID: 66
[INFO] ArUco marker ID: 87
```

这张图片包含了我们在上周的博客文章中生成的 ArUco 标记。我把五个单独的 ArUco 标记中的每一个都放在一张图片中进行了剪辑。

如图 3 所示，我们已经能够正确地检测出每个 ArUco 标记并提取它们的 id。

让我们尝试一个不同的图像，这个图像包含 ArUco 标记*而不是我们生成的*:

```py
$ python detect_aruco_image.py --image images/example_02.png --type DICT_ARUCO_ORIGINAL
[INFO] loading image...
[INFO] detecting 'DICT_ARUCO_ORIGINAL' tags...
[INFO] ArUco marker ID: 241
[INFO] ArUco marker ID: 1007
[INFO] ArUco marker ID: 1001
[INFO] ArUco marker ID: 923
```

**图 4** 显示了我们的 OpenCV ArUco 检测器的结果。正如你所看到的，我已经在我的 [Pantone 配色卡](https://www.pantone.com/pantone-color-match-card)上检测到了四个 ArUco 标记(我们将在接下来的一些教程中使用它，所以习惯于看到它)。

查看上面脚本的命令行参数，您可能会想知道:

> *“嗨，阿德里安，你怎么知道用`DICT_ARUCO_ORIGINAL`而不用其他阿鲁科字典呢？”*

简短的回答是，我*没有* …至少最初没有。

实际上，我有一个“秘密武器”。我已经编写了一个 Python 脚本，它可以*自动*推断 ArUco 标记类型，*即使我不知道图像中的标记是什么类型。*

下周我会和你分享这个剧本，所以请留意它。

### **使用 OpenCV 检测实时视频流中的 ArUco 标记**

在上一节中，我们学习了如何检测图像中的 ArUco 标记…

**…但是有可能在*实时*视频流中检测出阿鲁科标记吗？**

答案是*是的，绝对是*——我将在这一节向你展示如何做到这一点。

打开项目目录结构中的`detect_aruco_video.py`文件，让我们开始工作:

```py
# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import sys
```

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
	default="DICT_ARUCO_ORIGINAL",
	help="type of ArUCo tag to detect")
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

请参考上面的*“使用 OpenCV 在图像中检测 ArUco 标记”*一节，了解此代码块的更多详细信息。

我们现在可以加载我们的 ArUco 字典:

```py
# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUCo tag of '{}' is not supported".format(
		args["type"]))
	sys.exit(0)

# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
```

```py
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 1000 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)

	# detect ArUco markers in the input frame
	(corners, ids, rejected) = cv2.aruco.detectMarkers(frame,
		arucoDict, parameters=arucoParams)
```

```py
	# verify *at least* one ArUco marker was detected
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()

		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned
			# in top-left, top-right, bottom-right, and bottom-left
			# order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))
```

```py
			# draw the bounding box of the ArUCo detection
			cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

			# compute and draw the center (x, y)-coordinates of the
			# ArUco marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

			# draw the ArUco marker ID on the frame
			cv2.putText(frame, str(markerID),
				(topLeft[0], topLeft[1] - 15),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
```

### **OpenCV ArUco 视频检测结果**

准备好将 ArUco 检测应用于实时视频流了吗？

首先使用本教程的 ***【下载】*** 部分下载源代码和示例图像。

在那里，打开一个 shell，并执行以下命令:

```py
$ python detect_aruco_video.py
```

如你所见，我很容易在实时视频中检测到阿鲁科标记。

## **总结**

在本教程中，您学习了如何使用 OpenCV 和 Python 检测图像和实时视频流中的 ArUco 标记。

使用 OpenCV 检测 ArUco 标记分为三个步骤:

1.  设置你正在使用的 ArUco 字典。
2.  定义 ArUco 检测器的参数(通常默认选项就足够了)。
3.  使用 OpenCV 的``cv2.aruco.detectMarkers`` 功能应用 ArUco 检测器。

**OpenCV 的 ArUco 标记是*极快的*，正如我们的结果所示，能够实时检测 ArUco 标记。**

在您自己的计算机视觉管道中使用 ArUco 标记时，请随意使用此代码作为起点。

然而，假设你正在开发一个计算机视觉项目来自动检测图像中的 ArUco 标记，*但是你不知道正在使用什么类型的标记，*因此，*你不能显式地设置 ArUco 标记字典*——那么你会怎么做？

如果你不知道正在使用的标记类型，如何检测 ArUco 标记？

我将在下周的博客文章中回答这个问题。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***