# 视频流的光学字符识别

> 原文：<https://pyimagesearch.com/2022/03/07/ocring-video-streams/>

在本教程中，您将学习如何 OCR 视频流。

本课是关于使用 Python 进行光学字符识别的 4 部分系列的第 3 部分:

1.  *[多栏表格 OCR](https://pyimg.co/h18s2)*
2.  *[OpenCV 快速傅立叶变换(FFT)用于图像和视频流中的模糊检测](https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/)*
3.  *OCR 识别视频流*(本教程)
4.  *使用 OpenCV 和 GPU 提高文本检测速度*

## **对视频流进行光学字符识别**

在我们之前的教程中，你学习了如何使用快速傅立叶变换(FFT)来检测图像和文档中的模糊。使用这种方法，我们能够检测出模糊、低质量的图像，然后提醒用户应该尝试捕捉更高质量的版本，以便我们可以对其进行 OCR。

**记住，编写对*高质量*图像进行操作的计算机视觉代码总是比*低质量*图像更容易。**使用 FFT 模糊检测方法有助于确保只有更高质量的图像进入我们的管道。

然而，FFT 模糊检测器还有另一个用途，它可以用来从视频流中丢弃低质量的帧，否则无法进行 OCR。

由于光照条件的快速变化(例如，在明亮的晴天走进黑暗的房间)，摄像机镜头自动聚焦，或者最常见的*运动模糊，视频流自然会有低质量的帧。*

对这些帧进行光学字符识别几乎是不可能的。因此，我们可以简单地通过*检测到*帧模糊，忽略它，然后只对高质量的帧进行 OCR，而不是试图对视频流中的每个帧进行 OCR(这会导致低质量帧的无意义结果)？

这样的实现可能吗？

没错，我们将在本教程的剩余部分讲述如何将模糊检测器应用于 OCR 视频流。

在本教程中，您将:

*   了解如何对视频流进行 OCR
*   应用我们的 FFT 模糊检测器来检测和丢弃模糊、低质量的帧
*   构建一个显示视频流 OCR 阶段的输出可视化脚本
*   将所有部件放在一起，在视频流中完全实现 OCR

## **OCR 实时视频流**

在本教程的第一部分，我们将回顾我们的项目目录结构。

然后，我们将实现一个简单的视频编写器实用程序类。这个类将允许我们从一个*输入*视频中创建一个*输出*模糊检测和 OCR 结果的视频。

给定我们的视频编写器助手函数，然后我们将实现我们的驱动程序脚本来对视频流应用 OCR。

我们将讨论我们的结果来结束本教程。

**学习如何 OCR 视频流，** ***继续阅读。***

### **配置您的开发环境**

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果您需要帮助配置 OpenCV 的开发环境，我们*强烈推荐*阅读我们的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让您启动并运行。

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

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

让我们从查看视频 OCR 项目的目录结构开始:

```py
|-- pyimagesearch
|   |-- __init__.py
|   |-- helpers.py
|   |-- blur_detection
|   |   |-- __init__.py
|   |   |-- blur_detector.py
|   |-- video_ocr
|   |   |-- __init__.py
|   |   |-- visualization.py
|-- output
|   |-- ocr_video_output.avi
|-- video
|   |-- business_card.mp4
|-- ocr_video.py
```

在`pyimagesearch`模块中，我们有两个子模块:

1.  我们将用来帮助检测视频流中的模糊的子模块。
2.  包含一个助手函数，它将我们的视频 OCR 的输出作为一个单独的视频文件写到磁盘上。

`video`目录包含`business_card.mp4`，一个包含我们想要 OCR 的名片的视频。`output`目录包含在我们的输入视频上运行驱动脚本`ocr_video.py`的输出。

### **实现我们的视频编写器实用程序**

在我们实现我们的驱动程序脚本之前，我们首先需要实现一个基本的助手实用程序，它将允许我们将视频 OCR 脚本的输出作为一个单独的输出视频写入磁盘。

可视化脚本的输出示例可以在**图** **2** 中看到。请注意，输出有三个部分:

1.  检测到名片且注释模糊/不模糊的原始输入框*(上)*
2.  检测到文本的名片的*自顶向下*变换*(中)*
3.  从*自上而下*转换*(底部)*的 OCR 文本本身

在本节中，我们将实现一个助手实用函数来构建这样一个输出可视化。

请注意，这个视频编写器实用程序与 OCR 没有任何关系。相反，它只是一个简单的 Python 类，我实现它来编写视频 I/O。为了完整起见，我在本教程中只是回顾一下。

如果您发现自己很难跟上类的实现，*不要担心*——这不会影响您的 OCR 知识。也就是说，如果你想学习更多关于使用视频和 OpenCV 的知识，我推荐你跟随我们的*[*使用视频*](http://pyimg.co/s6wiy)*教程。**

 **现在让我们开始实现我们的视频作者实用程序。打开我们项目的`video_ocr`目录下的`visualization.py`文件，我们开始吧:

```py
# import the necessary packages
import numpy as np

class VideoOCROutputBuilder:
	def __init__(self, frame):
		# store the input frame dimensions
		self.maxW = frame.shape[1]
		self.maxH = frame.shape[0]
```

我们从定义我们的`VideoOCROutputBuilder`类开始。我们的构造函数只需要一个参数，我们的输入`frame`。然后我们将`frame`的宽度和高度分别存储为`maxW`和`maxH`。

考虑到我们的构造函数，让我们创建负责构造你在图 2 中看到的可视化的`build`方法。

```py
def build(self, frame, card=None, ocr=None):
		# grab the input frame dimensions and  initialize the card
		# image dimensions along with the OCR image dimensions
		(frameH, frameW) = frame.shape[:2]
		(cardW, cardH) = (0, 0)
		(ocrW, ocrH) = (0, 0)

		# if the card image is not empty, grab its dimensions
		if card is not None:
			(cardH, cardW) = card.shape[:2]

		# similarly, if the OCR image is not empty, grab its
		# dimensions
		if ocr is not None:
			(ocrH, ocrW) = ocr.shape[:2]
```

`build`方法接受三个参数，其中一个是必需的(另外两个是可选的):

1.  `frame`:来自视频的输入帧
2.  `card`:应用了*自上而下*透视变换后的名片，检测到名片上的文字
3.  `ocr`:OCR 识别的文本本身

**第 13 行**抓取输入`frame`的空间尺寸，而**第 14 和 15 行**初始化`card`和`ocr`图像的空间尺寸。

由于`card`和`ocr`都可能是`None`，我们不知道它们是否是有效图像。如果*是*，**行 18-24** 进行此项检查，如果通过，则抓取`card`和`ocr`的宽度和高度。

我们现在可以开始构建我们的`output`可视化:

```py
		# compute the spatial dimensions of the output frame
		outputW = max([frameW, cardW, ocrW])
		outputH = frameH + cardH + ocrH

		# update the max output spatial dimensions found thus far
		self.maxW = max(self.maxW, outputW)
		self.maxH = max(self.maxH, outputH)

		# allocate memory of the output image using our maximum
		# spatial dimensions
		output = np.zeros((self.maxH, self.maxW, 3), dtype="uint8")

		# set the frame in the output image
		output[0:frameH, 0:frameW] = frame
```

**第 27 行**通过找到穿过`frame`、`card`和`ocr`的`max`高度，计算出`output`可视化的最大*宽度*。**第 28 行**通过将所有三个高度相加来确定可视化的*高度*(我们做这个相加操作是因为这些图像需要*堆叠*在另一个之上)。

**第 31 行和第 32 行**用我们目前发现的最大宽度和高度值更新我们的`maxW`和`maxH`簿记变量。

给定我们最新更新的`maxW`和`maxH`，**第 36 行**使用我们目前发现的最大空间维度为我们的`output`图像分配内存。

随着`output`图像的初始化，我们将`frame`存储在`output` ( **第 39 行**)的顶部。

我们的下一个代码块处理将`card`和`ocr`图像添加到`output`帧:

```py
		# if the card is not empty, add it to the output image
		if card is not None:
			output[frameH:frameH + cardH, 0:cardW] = card

		# if the OCR result is not empty, add it to the output image
		if ocr is not None:
			output[
				frameH + cardH:frameH + cardH + ocrH,
				0:ocrW] = ocr

		# return the output visualization image
		return output
```

**第 42 行和第 43 行**验证一个有效的`card`图像已经被传递到函数中，如果是的话，我们将它添加到`output`图像中。**第 46-49 行**做同样的事情，只针对`ocr`图像。

最后，我们将`output`可视化返回给调用函数。

祝贺实现了我们的`VideoOCROutputBuilder`类！让我们在下一节中使用它！

### **实现我们的实时视频 OCR 脚本**

我们现在准备实现我们的`ocr_video.py`脚本。让我们开始工作:

```py
# import the necessary packages
from pyimagesearch.video_ocr import VideoOCROutputBuilder
from pyimagesearch.blur_detection import detect_blur_fft
from pyimagesearch.helpers import cleanup_text
from imutils.video import VideoStream
from imutils.perspective import four_point_transform
from pytesseract import Output
import pytesseract
import numpy as np
import argparse
import imutils
import time
import cv2
```

我们从第 2-13 行开始，导入我们需要的 Python 包。值得注意的进口包括:

*   我们的可视化构建器
*   我们的 FFT 模糊检测器
*   `cleanup_text`:用于清理 OCR 文本，剔除非 ASCII 字符，以便我们可以使用 OpenCV 的`cv2.putText`函数在输出图像上绘制 OCR 文本
*   `four_point_transform`:应用透视变换，这样我们就可以获得一张我们正在进行 OCR 的名片的*自上而下*/鸟瞰图
*   `pytesseract`:提供一个到 Tesseract OCR 引擎的接口

考虑到我们的导入，让我们继续我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video (webcam will be used otherwise)")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video")
ap.add_argument("-c", "--min-conf", type=int, default=50,
	help="minimum confidence value to filter weak text detection")
args = vars(ap.parse_args())
```

我们的脚本提供了三个命令行参数:

1.  `--input`:磁盘上可选输入视频文件的路径。如果没有提供视频文件，我们将使用我们的网络摄像头。
2.  `--output`:我们将要生成的可选输出视频文件的路径。
3.  `--min-conf`:用于过滤弱文本检测的最小置信度值。

现在我们可以继续初始化了:

```py
# initialize our video OCR output builder used to easily visualize
# output to our screen
outputBuilder = None

# initialize our output video writer along with the dimensions of the
# output frame
writer = None
outputW = None
outputH = None
```

**第 27 行**初始化我们的`outputBuilder`。这个对象将在我们的`while`循环的主体中被实例化，该循环从我们的视频流中访问帧(我们将在本教程的后面介绍)。

然后我们在第 31-33 行初始化输出视频写入器和输出视频的空间尺寸。

让我们继续访问我们的视频流:

```py
# create a named window for our output OCR visualization (a named
# window is required here so that we can automatically position it
# on our screen)
cv2.namedWindow("Output")

# initialize a Boolean used to indicate if either a webcam or input
# video is being used
webcam = not args.get("input", False)

# if a video path was not supplied, grab a reference to the webcam
if webcam:
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])
```

**第 38 行**为我们的输出可视化创建一个名为`Output`的命名窗口。我们*显式地*在这里创建了一个命名窗口，这样我们就可以使用 OpenCV 的`cv2.moveWindow`函数来移动屏幕上的窗口。我们需要执行这个移动操作，因为输出窗口的大小是动态的，它会随着输出的增加和缩小而增加。

**Line 42** 确定我们是否使用网络摄像头作为视频输入。如果是这样，**第 45-48 行**访问我们的网络摄像头视频流；否则，**行 51-53** 抓取一个指向驻留在磁盘上的视频的指针。

访问我们的视频流后，现在是开始循环播放帧的时候了:

```py
# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# a webcam or a video file
	orig = vs.read()
	orig = orig if webcam else orig[1]

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if not webcam and orig is None:
		break

	# resize the frame and compute the ratio of the *new* width to
	# the *old* width
	frame = imutils.resize(orig, width=600)
	ratio = orig.shape[1] / float(frame.shape[1])

	# if our video OCR output builder is None, initialize it
	if outputBuilder is None:
		outputBuilder = VideoOCROutputBuilder(frame)
```

**第 59 行和第 60 行**从视频流中读取原始(`orig`)帧。如果设置了`webcam`变量，并且`orig`帧为`None`，那么我们已经到达了视频文件的末尾，因此我们从循环中断开。

否则，**行 69 和 70** 将帧的宽度调整为 700 像素(这样处理起来更容易更快)，然后计算*新*宽度的`ratio`到*旧*宽度。在这个循环的后面，当我们将透视变换应用到原始的高分辨率帧时，我们将需要这个比率。

**第 73 和 74 行**使用调整后的`frame`初始化我们的`VideoOCROutputBuilder`。

接下来是更多的初始化，然后是模糊检测:

```py
	# initialize our card and OCR output ROIs
	card = None
	ocr = None

	# convert the frame to grayscale and detect if the frame is
	# considered blurry or not
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	(mean, blurry) = detect_blur_fft(gray, thresh=15)

	# draw whether or not the frame is blurry
	color = (0, 0, 255) if blurry else (0, 255, 0)
	text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
	text = text.format(mean)
	cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
		0.7, color, 2)
```

**第 77 和 78 行**初始化我们的`card`和`ocr`ROI。`card` ROI 将包含名片的*自顶向下*转换(如果在当前`frame`中找到名片)，而`ocr`将包含 OCR 处理的文本本身。

然后我们在**行 82 和 83** 上执行文本/文档模糊检测。我们首先将`frame`转换成灰度，然后应用我们的`detect_blur_fft`函数。

**第 86-90 行**画在`frame`上，表示当前帧是否模糊。

让我们继续我们的视频 OCR 管道:

```py
	# only continue to process the frame for OCR if the image is
	# *not* blurry
	if not blurry:
		# blur the grayscale image slightly and then perform edge
		# detection
		blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
		edged = cv2.Canny(blurred, 75, 200)

		# find contours in the edge map and sort them by size in
		# descending order, keeping only the largest ones
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

		# initialize a contour that corresponds to the business card
		# outline
		cardCnt = None

		# loop over the contours
		for c in cnts:
			# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)

			# if our approximated contour has four points, then we
			# can assume we have found the outline of the business
			# card
			if len(approx) == 4:
				cardCnt = approx
				break
```

在继续之前，我们检查以确认帧是*而不是*模糊的。假设检查通过，我们开始在输入帧中寻找名片，方法是用高斯核平滑帧，然后应用边缘检测(**第 97 和 98 行**)。

然后，我们将轮廓检测应用于边缘图，并按面积对轮廓进行排序，从最大到最小(**行 102-105** )。我们这里的假设是，名片将是输入帧中最大的 ROI，*也有四个顶点。*

为了确定我们是否找到了名片，我们遍历了第 112 行**上的最大轮廓。然后我们应用轮廓近似(**线 114 和 115** )并检查近似轮廓是否有四个点。**

假设轮廓有四个点，我们假设已经找到了我们的卡片轮廓，所以我们存储轮廓变量(`cardCnt`)，然后从循环(**第 120-122 行**)中存储`break`。

如果我们找到了我们的名片轮廓，我们现在尝试 OCR 它:

```py
		# ensure that the business card contour was found
		if cardCnt is not None:
			# draw the outline of the business card on the frame so
			# we visually verify that the card was detected correctly
			cv2.drawContours(frame, [cardCnt], -1, (0, 255, 0), 3)

			# apply a four-point perspective transform to the
			# *original* frame to obtain a top-down bird's-eye
			# view of the business card
			card = four_point_transform(orig,
				cardCnt.reshape(4, 2) * ratio)

			# allocate memory for our output OCR visualization
			ocr = np.zeros(card.shape, dtype="uint8")

			# swap channel ordering for the business card and OCR it
			rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
			results = pytesseract.image_to_data(rgb,
				output_type=Output.DICT)
```

**第 125 行**证实我们确实找到了我们的名片轮廓。然后我们通过 OpenCV 的`cv2.drawContours`函数在`frame`上绘制卡片轮廓。

接下来，我们通过使用我们的`four_point_transform`函数(**第 133 行和第 134 行**)对*原始的高分辨率图像*应用透视变换(这样我们可以更好地对其进行 OCR 识别)。我们还为我们的输出`ocr`可视化分配内存，在应用*自顶向下*转换后使用`card`的相同空间维度(**第 137 行**)。

**第 140-142 行**然后对名片应用文本检测和 OCR。

下一步是用 OCR 文本本身注释输出`ocr`可视化:

```py
			# loop over each of the individual text localizations
			for i in range(0, len(results["text"])):
				# extract the bounding box coordinates of the text
				# region from the current result
				x = results["left"][i]
				y = results["top"][i]
				w = results["width"][i]
				h = results["height"][i]

				# extract the OCR text itself along with the
				# confidence of the text localization
				text = results["text"][i]
				conf = int(results["conf"][i])

				# filter out weak confidence text localizations
				if conf > args["min_conf"]:
					# process the text by stripping out non-ASCII
					# characters
					text = cleanup_text(text)

					# if the cleaned up text is not empty, draw a
					# bounding box around the text along with the
					# text itself
					if len(text) > 0:
						cv2.rectangle(card, (x, y), (x + w, y + h),
							(0, 255, 0), 2)
						cv2.putText(ocr, text, (x, y - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5,
							(0, 0, 255), 1)
```

**第 145 行**循环所有文本检测。然后，我们继续:

*   抓取文本 ROI 的边界框坐标(**行 148-151** )
*   提取 OCR 文本及其相应的置信度/概率(**行 155 和 156** )
*   验证文本检测是否有足够的可信度，然后从文本中剔除非 ASCII 字符(**行 159-162** )
*   在`ocr`可视化上绘制 OCR 文本(**第 167-172 行**

本例中的其余代码块更侧重于簿记变量和输出:

```py
	# build our final video OCR output visualization
	output = outputBuilder.build(frame, card, ocr)

	# check if the video writer is None *and* an output video file
	# path was supplied
	if args["output"] is not None and writer is None:
		# grab the output frame dimensions and initialize our video
		# writer
		(outputH, outputW) = output.shape[:2]
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 27,
			(outputW, outputH), True)

	# if the writer is not None, we need to write the output video
	# OCR visualization to disk
	if writer is not None:
		# force resize the video OCR visualization to match the
		# dimensions of the output video
		outputFrame = cv2.resize(output, (outputW, outputH))
		writer.write(outputFrame)

	# show the output video OCR visualization
	cv2.imshow("Output", output)
	cv2.moveWindow("Output", 0, 0)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break
```

**第 175 行**使用我们的`VideoOCROutputBuilder`类的`.build`方法创建我们的`output`帧。

然后我们检查是否提供了一个`--output`视频文件路径，如果是，实例化我们的`cv2.VideoWriter`，这样我们就可以将`output`帧可视化写到磁盘上(**第 179-185 行**)。

类似地，如果`writer`对象已经被实例化，那么我们将输出帧写入磁盘(**第 189-193 行**)。

**第 196-202 行**向我们的屏幕显示`output`画面:

我们的最后一个代码块释放了视频指针:

```py
# if we are using a webcam, stop the camera video stream
if webcam:
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()
```

整体来看，这似乎是一个复杂的脚本。但是请记住，我们刚刚用不到 225 行代码(包括注释)实现了一个完整的视频 OCR 管道。仔细想想，代码并不多——而且这一切都是通过使用 OpenCV 和 Tesseract 实现的！

### **实时视频 OCR 结果**

我们现在准备好测试我们的视频 OCR 脚本了！打开终端并执行以下命令:

```py
$ python ocr_video.py --input video/business_card.mp4 --output output/ocr_video_output.avi
[INFO] opening video file...
```

**图 3** 显示了来自`output`目录中的`ocr_video_output.avi`文件的屏幕截图。

请注意左边的*部分，我们的脚本已经正确地检测到一个模糊的帧，并且没有对它进行 OCR。如果我们试图对这个帧进行 OCR，结果将会是无意义的，使最终用户感到困惑。*

相反，我们等待更高质量的帧*(右)*，然后对其进行 OCR。如您所见，通过等待更高质量的帧，我们能够正确地对名片进行 OCR。

如果你需要对视频流应用 OCR，我*强烈建议*使用某种低质量和高质量的帧检测器。**除非你** ***100%确信*** **视频是在理想的受控条件下拍摄的，并且每一帧都是高质量的，否则不要试图对视频流的每一帧进行 OCR。**

## **总结**

在本教程中，您学习了如何对视频流进行 OCR。然而，首先，我们需要检测模糊、低质量的帧，以便对视频流进行 OCR。

由于光照条件的快速变化、相机镜头自动对焦和运动模糊，视频自然会有低质量的帧。我们需要*检测*这些低质量的帧并丢弃它们，而不是尝试对这些低质量的帧进行 OCR，这最终会导致低 OCR 准确度(或者更糟，完全无意义的结果)。

检测低质量帧的一种简单方法是使用模糊检测。因此，我们利用 FFT 模糊检测器来处理视频流。结果是 OCR 管道能够在视频流上操作，同时仍然保持高精度。

我希望你喜欢这个教程！我希望你能把这种方法应用到你的项目中。

### **引用信息**

**Rosebrock，a .**“OCR ' ing Video Streams”， *PyImageSearch* ，D. Chakraborty，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva，R. Raha 和 A. Thanki 编辑。，2022 年，【https://pyimg.co/k43vd 

```py
@incollection{Rosebrock_2022_OCR_Video_Streams,
  author = {Adrian Rosebrock},
  title = {{OCR}’ing Video Streams},
  booktitle = {PyImageSearch},
  editor = {Devjyoti Chakraborty and Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/k43vd},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******