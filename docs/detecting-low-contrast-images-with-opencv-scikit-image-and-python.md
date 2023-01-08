# 用 OpenCV、scikit-image 和 Python 检测低对比度图像

> 原文：<https://pyimagesearch.com/2021/01/25/detecting-low-contrast-images-with-opencv-scikit-image-and-python/>

在本教程中，您将学习如何使用 OpenCV 和 scikit-image 检测低对比度图像。

每当我给渴望学习的学生讲授计算机视觉和图像处理的基础知识时，我教的第*件事之一就是:*

> *“为在**受控光照条件**下拍摄的图像编写代码，比在**没有保证的动态条件下容易得多。***

如果你能够控制环境，最重要的是，当你捕捉图像时，能够控制*光线*，那么编写处理图像的代码就越容易。

在受控的照明条件下，您可以对参数进行硬编码，包括:

*   模糊量
*   边缘检测界限
*   阈值限制
*   等等。

本质上，受控条件允许你利用你对环境的先验知识，然后编写代码来处理特定的环境，而不是试图处理每一个边缘情况或条件。

当然，控制你的环境和照明条件并不总是可能的…

…那你会怎么做？

你尝试过编码一个超级复杂的图像处理管道来处理每一个边缘情况吗？

嗯……你可以这么做——可能会浪费几周或几个月的时间，而且*仍然*可能无法捕捉到所有的边缘情况。

**或者，当低质量图像，特别是*低对比度图像*出现在您的管道中时，您可以改为*检测*。**

如果检测到低对比度图像，您可以丢弃图像或提醒用户在更好的照明条件下捕捉图像。

这样做将使您更容易开发图像处理管道(并减少您的麻烦)。

**要了解如何使用 OpenCV 和 scikit-image 检测低对比度图像，*请继续阅读。***

## **使用 OpenCV、scikit-image 和 Python 检测低对比度图像**

在本教程的第一部分，我们将讨论什么是低对比度图像，它们给计算机视觉/图像处理从业者带来的问题，以及我们如何以编程方式检测这些图像。

从那里，我们将配置我们的开发环境，并审查我们的项目目录结构。

回顾了我们的项目结构后，我们将继续编写两个 Python 脚本:

1.  一种用于检测静态图像中的低对比度
2.  另一个用于检测实时视频流中的低对比度帧

我们将讨论我们的结果来结束我们的教程。

### 低对比度图像/帧会产生什么问题？我们怎样才能发现它们？

低对比度图像的亮区和暗区差别很小，很难看出物体的边界和场景的背景从哪里开始。

在**图 1 *(左图)中显示了一个低对比度图像的示例。*** 这里你可以看到一个背景上的颜色匹配/校正卡。由于照明条件差(即光线不足)，卡片相对于背景的边界没有被很好地定义-就其本身而言，边缘检测算法，如 Canny 边缘检测器，可能难以检测卡片的边界，*尤其是*如果 Canny 边缘检测器参数是硬编码的。

**图 1 *(右)*** 显示“正常对比度”的示例图像。由于更好的照明条件，我们在这张图像中有更多的细节。请注意，颜色匹配卡的白色与背景形成了充分的对比——对于图像处理管道来说,*更容易检测颜色匹配卡的边缘(与右侧的*图像相比)。**

**每当你处理计算机视觉或图像处理问题时，*总是从捕捉图像/帧的环境开始。*** 你越能控制和保证光线条件，*就越容易*一次你就有时间编写代码来处理场景。

然而，有时你*无法*控制照明条件和你硬编码到管道中的任何参数(例如模糊尺寸、阈值限制、Canny 边缘检测参数等。)可能导致不正确/不可用的输出。

当这种情况不可避免地发生时，不要放弃。并且*当然*不会开始进入编码复杂图像处理管道的兔子洞去处理每一个边缘情况。

**相反，利用低对比度图像检测。**

使用低对比度图像检测，您可以以编程方式检测不足以用于图像处理管道的图像。

在本教程的剩余部分，您将学习如何检测静态场景和实时视频流中的低对比度图像。

我们将丢弃低对比度和不适合我们管道的图像/帧，同时只保留我们知道会产生有用结果的图像/帧。

本指南结束时，您将对低对比度图像检测有一个很好的了解，并且能够将其应用到您自己的项目中，从而使您自己的管道更容易开发，在生产中更稳定。

### **配置您的开发环境**

为了检测低对比度图像，您需要安装 [OpenCV 库](https://opencv.org/)以及 [scikit-image](https://scikit-image.org/) 。

幸运的是，这两个都是 pip 可安装的:

```py
$ pip install opencv-contrib-python
$ pip install scikit-image
```

**如果您需要帮助配置 OpenCV 和 scikit-image 的开发环境，我*强烈推荐*阅读我的 *[pip 安装 OpenCV 指南](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)***——它将在几分钟内让您启动并运行。

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

在我们深入了解本指南之前，让我们花点时间来检查一下我们的项目目录结构。

首先使用本教程的 ***“下载”*** 部分下载源代码、示例图像和示例视频:

```py
$ tree . --dirsfirst
.
├── examples
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
├── detect_low_contrast_image.py
├── detect_low_contrast_video.py
└── example_video.mp4

1 directory, 6 files
```

我们今天要复习两个 Python 脚本:

1.  `detect_low_contrast_image.py`:对静态*图像*(即`examples`目录下的图像)进行低对比度检测
2.  `detect_low_contrast_video.py`:对*实时视频流*(本例中为`example_video.mp4`)进行低对比度检测

当然，如果你认为合适的话，你可以用你自己的图像和视频文件/流来代替。

### **用 OpenCV 实现低对比度图像检测**

让我们学习如何使用 OpenCV 和 scikit-image 检测低对比度图像！

打开项目目录结构中的`detect_low_contrast_image.py`文件，并插入以下代码。

```py
# import the necessary packages
from skimage.exposure import is_low_contrast
from imutils.paths import list_images
import argparse
import imutils
import cv2
```

我们从第 2-6 行的**开始，导入我们需要的 Python 包。**

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-t", "--thresh", type=float, default=0.35,
	help="threshold for low contrast")
args = vars(ap.parse_args())
```

我们有两个命令行参数，第一个是必需的，第二个是可选的:

1.  `--input`:驻留在磁盘上的输入图像的路径
2.  ``--thresh`` :低对比度的阈值

我已经将`--thresh`参数设置为默认值`0.35`，这意味着“当亮度范围小于其数据类型全部范围的一小部分时” ( [官方 scikit-image 文档](https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.is_low_contrast))，图像将被视为低对比度*。*

本质上，这意味着如果亮度范围的 35%以下占据了数据类型的全部范围，则图像被认为是低对比度的。

为了使这成为一个具体的例子，考虑 OpenCV 中的图像由一个无符号的 8 位整数表示，该整数的取值范围为 *[0，255】。*如果像素强度分布占据的*小于*该*【0，255】*范围的 35%，则图像被视为低对比度。

```py
# grab the paths to the input images
imagePaths = sorted(list(list_images(args["input"])))

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# load the input image from disk, resize it, and convert it to
	# grayscale
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=450)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# blur the image slightly and perform edge detection
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 30, 150)

	# initialize the text and color to indicate that the input image
	# is *not* low contrast
	text = "Low contrast: No"
	color = (0, 255, 0)
```

1.  从磁盘加载`image`
2.  将其宽度调整为 450 像素
3.  将图像转换为灰度

从那里，我们应用模糊(以减少高频噪声)，然后应用 Canny 边缘检测器(**行 30 和 31** )来检测输入图像中的边缘。

```py
	# check to see if the image is low contrast
	if is_low_contrast(gray, fraction_threshold=args["thresh"]):
		# update the text and color
		text = "Low contrast: Yes"
		color = (0, 0, 255)

	# otherwise, the image is *not* low contrast, so we can continue
	# processing it
	else:
		# find contours in the edge map and find the largest one,
		# which we'll assume is the outline of our color correction
		# card
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)

		# draw the largest contour on the image
		cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

	# draw the text on the output image
	cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
		color, 2)

	# show the output image and edge map
	cv2.imshow("Image", image)
	cv2.imshow("Edge", edged)
	cv2.waitKey(0)
```

否则，图像是*而不是*低对比度，所以我们可以继续我们的图像处理流水线(**第 46-56 行**)。在这个代码块中，我们:

1.  在我们的边缘地图中寻找轮廓
2.  在我们的`cnts`列表中找到最大的轮廓(我们假设它将是我们在输入图像中的卡片)
3.  在图像上画出卡片的轮廓

最后，我们在`image`上绘制`text`，并在屏幕上显示`image`和边缘图。

### **低对比度图像检测结果**

现在让我们将低对比度图像检测应用于我们自己的图像！

首先使用本教程的 ***【下载】*** 部分下载源代码和示例图像:

```py
$ python detect_low_contrast_image.py --input examples
[INFO] processing image 1/3
[INFO] processing image 2/3
[INFO] processing image 3/3
```

我们这里的第一个图像被标记为“低对比度”。如你所见，将 Canny 边缘检测器应用于低对比度图像导致我们无法检测图像中的卡片轮廓。

如果我们试图进一步处理这个图像并检测卡片本身，我们最终会检测到其他轮廓。相反，通过应用低对比度检测，我们可以简单地忽略图像。

我们的第二个图像具有足够的对比度，因此，我们能够精确地计算边缘图并提取与卡片轮廓相关联的轮廓:

我们的最终图像也被标记为具有足够的对比度:

我们再次能够计算边缘图，执行轮廓检测，并提取与卡的轮廓相关联的轮廓。

### **在实时视频流中实现低对比度帧检测**

在本节中，您将学习如何使用 OpenCV 和 Python 在实时视频流中实现低对比度帧检测。

打开项目目录结构中的`detect_low_contrast_video.py`文件，让我们开始工作:

```py
# import the necessary packages
from skimage.exposure import is_low_contrast
import numpy as np
import argparse
import imutils
import cv2
```

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="optional path to video file")
ap.add_argument("-t", "--thresh", type=float, default=0.35,
	help="threshold for low contrast")
args = vars(ap.parse_args())
```

```py
# grab a pointer to the input video stream
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

# loop over frames from the video stream
while True:
	# read a frame from the video stream
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed then we've reached the end of
	# the video stream so exit the script
	if not grabbed:
		print("[INFO] no frame read from stream - exiting")
		break

	# resize the frame, convert it to grayscale, blur it, and then
	# perform edge detection
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 30, 150)

	# initialize the text and color to indicate that the current
	# frame is *not* low contrast
	text = "Low contrast: No"
	color = (0, 255, 0)
```

第 18 行实例化了一个指向我们视频流的指针。默认情况下，我们将使用网络摄像头；但是，如果你是一个视频文件，你可以提供`--input`命令行参数。

然后，我们在第 21 行的视频流中循环播放帧。在循环内部，我们:

1.  阅读下一篇`frame`
2.  检测我们是否到达了视频流的末尾，如果是，则从循环中`break`
3.  预处理帧，将其转换为灰度，模糊，并应用 Canny 边缘检测器

```py
	# check to see if the frame is low contrast, and if so, update
	# the text and color
	if is_low_contrast(gray, fraction_threshold=args["thresh"]):
		text = "Low contrast: Yes"
		color = (0, 0, 255)

	# otherwise, the frame is *not* low contrast, so we can continue
	# processing it
	else:
		# find contours in the edge map and find the largest one,
		# which we'll assume is the outline of our color correction
		# card
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)

		# draw the largest contour on the frame
		cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
```

```py
	# draw the text on the output frame
	cv2.putText(frame, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
		color, 2)

	# stack the output frame and edge map next to each other
	output = np.dstack([edged] * 3)
	output = np.hstack([frame, output])

	# show the output to our screen
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
```

### **实时检测低对比度帧**

我们现在可以检测实时视频流中的低对比度图像了！

使用本教程的 ***“下载”*** 部分下载源代码、示例图像和示例视频文件。

从那里，打开一个终端，并执行以下命令:

```py
$ python detect_low_contrast_video.py --input example_video.mp4
[INFO] accessing video stream...
[INFO] no frame read from stream - exiting
```

正如我们的输出所示，我们的低对比度帧检测器能够检测低对比度帧，*防止*它们进入我们图像处理管道的其余部分。

相反，具有足够对比度的*图像被允许继续。然后，我们对这些帧中的每一帧应用边缘检测，计算轮廓，并提取与颜色校正卡相关联的轮廓/轮廓。*

您可以以同样的方式在视频流中使用低对比度检测。

## **总结**

在本教程中，您学习了如何检测静态场景和实时视频流中的低对比度图像。我们使用 OpenCV 库和 scikit-image 包来开发我们的低对比度图像检测器。

虽然简单，但当用于计算机视觉和图像处理流水线时，这种方法会非常有效。

使用这种方法最简单的方法之一是向用户提供反馈。如果用户为您的应用程序提供了一个低对比度的图像，提醒他们并要求他们提供一个高质量的图像。

采用这种方法允许您对用于捕获图像的环境进行“保证”,这些图像最终会呈现给您的管道。此外，它有助于用户理解你的应用程序只能在特定的场景中使用，并且需要确保它们符合你的标准。

这里的要点是不要让你的图像处理管道过于复杂。当你可以保证光照条件和环境时，编写 OpenCV 代码就变得*容易得多*——尽你所能尝试执行这些标准。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***