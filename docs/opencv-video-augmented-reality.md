# OpenCV 视频增强现实

> 原文：<https://pyimagesearch.com/2021/01/11/opencv-video-augmented-reality/>

在本教程中，您将学习如何使用 OpenCV 在视频流中执行实时增强现实。

上周我们用 OpenCV 讲述了增强现实的基础知识；然而，该教程只专注于将增强现实应用于*图像。*

这就提出了一个问题:

> *“有没有可能用 OpenCV 在实时视频中进行实时增强现实？”*

绝对是——本教程的其余部分将告诉你如何做。

**要了解如何使用 OpenCV 执行实时增强现实，*请继续阅读。***

## **OpenCV:实时视频增强现实**

在本教程的第一部分，您将了解 OpenCV 如何在实时视频流中促进增强现实。

从那里，我们将配置我们的开发环境，并审查我们的项目目录结构。

然后，我们将回顾两个 Python 脚本:

*   第一个将包含一个助手函数`find_and_warp`，它将接受输入图像，检测增强现实标记，然后将源图像扭曲到输入上。
*   第二个脚本将充当驱动程序脚本，并在实时视频流中使用我们的``find_and_warp`` 函数。

我们将通过讨论我们的实时增强现实结果来结束本教程。

我们开始吧！

### **如何利用 OpenCV 将增强现实应用于实时视频流？**

OpenCV 库存在的真正原因是为了促进实时图像处理。该库接受输入图像/帧，尽快处理它们，然后返回结果。

由于 OpenCV 适用于实时图像处理，我们也可以使用 OpenCV 来促进实时增强现实。

出于本教程的目的，我们将:

1.  访问我们的视频流
2.  [检测每个输入帧中的 ArUco 标记](https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/)
3.  获取一个源图像，并应用透视变换将源输入映射到帧上，从而创建我们的增强现实输出！

**为了让这个项目*更加*有趣，我们将利用*两个*视频流:**

1.  第一个视频流将充当我们进入真实世界的“眼睛”(即我们的摄像机所看到的)。
2.  然后，我们将从第二个视频流中读取帧，然后将它们转换为第一个视频流。

本教程结束时，您将拥有一个实时运行的全功能 OpenCV 增强现实项目！

### **配置您的开发环境**

为了使用 OpenCV 执行实时增强现实，您需要安装 OpenCV 库。

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
*   **准备好在你的 Windows、macOS 或 Linux 系统上运行代码*了吗？***

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

在我们可以用 OpenCV 实现实时增强现实之前，我们首先需要回顾一下我们的项目目录结构。

首先使用本教程的 ***【下载】*** 部分下载源代码和示例视频文件。

现在让我们看一下目录内容:

```py
$ tree . --dirsfirst
.
├── pyimagesearch
│   ├── __init__.py
│   └── augmented_reality.py
├── videos
│   └── jp_trailer_short.mp4
├── markers.pdf
└── opencv_ar_video.py

2 directories, 4 files
```

如果你没有自己的配色卡，也不用担心！在我们的项目目录结构中，您会看到我已经包含了`markers.pdf`，它是我自己的 Pantone 颜色匹配卡的扫描件:

### **实现我们的标记检测器/增强现实实用功能**

在我们使用 OpenCV 在实时视频流中实现增强现实之前，我们首先需要创建一个助手函数`find_and_warp`，顾名思义，它将:

1.  接受输入图像和源图像
2.  在输入图像上找到四个 ArUco 标签
3.  构建并应用单应矩阵将源图像扭曲到输入表面

此外，我们将包括处理所有四个 ArUco 参考点未被检测到时的逻辑(以及如何确保我们的输出中没有闪烁/断续)。

打开我们项目目录结构的`pyimagesearch`模块中的``augmented_reality.py`` 文件，让我们开始工作:

```py
# import the necessary packages
import numpy as np
import cv2

# initialize our cached reference points
CACHED_REF_PTS = None
```

由于光照条件、视点或运动模糊的变化，有时我们的四个参考 ArUco 标记*无法在给定的输入帧中检测到*。

当这种情况发生时，我们有两种行动方案:

1.  **从输出为空的函数返回。**这种方法的好处是简单且易于实施(逻辑上也很合理)。问题是，如果 ArUco 标签在帧#1 中被发现，在帧#2 中被错过，然后在帧#3 中再次被发现，那么它会产生“闪烁”效果。
2.  **退回到 ArUco 标记的先前已知位置。**这是缓存方式。它减少了闪烁，有助于创建无缝的增强现实体验，但如果参考标记快速移动，则效果可能会显得有点“滞后”

您决定使用哪种方法完全取决于您，但我个人喜欢缓存方法，因为它为增强现实创造了更好的用户体验。

完成了导入和变量初始化之后，让我们继续关注我们的``find_and_warp`` 函数。

```py
def find_and_warp(frame, source, cornerIDs, arucoDict, arucoParams,
	useCache=False):
	# grab a reference to our cached reference points
	global CACHED_REF_PTS

	# grab the width and height of the frame and source image,
	# respectively
	(imgH, imgW) = frame.shape[:2]
	(srcH, srcW) = source.shape[:2]
```

```py
	# detect AruCo markers in the input frame
	(corners, ids, rejected) = cv2.aruco.detectMarkers(
		frame, arucoDict, parameters=arucoParams)

	# if we *did not* find our four ArUco markers, initialize an
	# empty IDs list, otherwise flatten the ID list
	ids = np.array([]) if len(corners) != 4 else ids.flatten()

	# initialize our list of reference points
	refPts = []
```

```py
	# loop over the IDs of the ArUco markers in top-left, top-right,
	# bottom-right, and bottom-left order
	for i in cornerIDs:
		# grab the index of the corner with the current ID
		j = np.squeeze(np.where(ids == i))

		# if we receive an empty list instead of an integer index,
		# then we could not find the marker with the current ID
		if j.size == 0:
			continue

		# otherwise, append the corner (x, y)-coordinates to our list
		# of reference points
		corner = np.squeeze(corners[j])
		refPts.append(corner)
```

否则，我们将拐角 *(x，y)*-坐标添加到我们的参考列表中(**第 42 行和第 43 行**)。

但是如果我们找不到所有的四个参考点会怎么样呢？接下来会发生什么？

下一个代码块解决了这个问题:

```py
	# check to see if we failed to find the four ArUco markers
	if len(refPts) != 4:
		# if we are allowed to use cached reference points, fall
		# back on them
		if useCache and CACHED_REF_PTS is not None:
			refPts = CACHED_REF_PTS

		# otherwise, we cannot use the cache and/or there are no
		# previous cached reference points, so return early
		else:
			return None

	# if we are allowed to use cached reference points, then update
	# the cache with the current set
	if useCache:
		CACHED_REF_PTS = refPts
```

```py
	# unpack our ArUco reference points and use the reference points
	# to define the *destination* transform matrix, making sure the
	# points are specified in top-left, top-right, bottom-right, and
	# bottom-left order
	(refPtTL, refPtTR, refPtBR, refPtBL) = refPts
	dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
	dstMat = np.array(dstMat)

	# define the transform matrix for the *source* image in top-left,
	# top-right, bottom-right, and bottom-left order
	srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

	# compute the homography matrix and then warp the source image to
	# the destination based on the homography
	(H, _) = cv2.findHomography(srcMat, dstMat)
	warped = cv2.warpPerspective(source, H, (imgW, imgH))
```

上面的代码，以及这个函数的其余部分，基本上与上周的[相同，所以我将把这些代码块的详细讨论推迟到前面的指南。](https://pyimagesearch.com/2021/01/04/opencv-augmented-reality-ar/)

```py
	# construct a mask for the source image now that the perspective
	# warp has taken place (we'll need this mask to copy the source
	# image into the destination)
	mask = np.zeros((imgH, imgW), dtype="uint8")
	cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255),
		cv2.LINE_AA)

	# this step is optional, but to give the source image a black
	# border surrounding it when applied to the source image, you
	# can apply a dilation operation
	rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	mask = cv2.dilate(mask, rect, iterations=2)

	# create a three channel version of the mask by stacking it
	# depth-wise, such that we can copy the warped source image
	# into the input image
	maskScaled = mask.copy() / 255.0
	maskScaled = np.dstack([maskScaled] * 3)
```

**第 82-84 行**为一个``mask`` 分配内存，然后我们用*白色*填充前景，用*黑色*填充背景。

对第 89 和 90 行执行膨胀操作，以创建围绕源图像的黑色边框(可选，但出于美观目的看起来不错)。

然后，我们从范围*【0，255】*到*【0，1】*缩放我们的遮罩，然后在深度方向上堆叠它，得到一个 3 通道遮罩。

最后一步是使用`mask`将`warped`图像应用到输入表面:

```py
	# copy the warped source image into the input image by
	# (1) multiplying the warped image and masked together,
	# (2) then multiplying the original input image with the
	# mask (giving more weight to the input where there
	# *ARE NOT* masked pixels), and (3) adding the resulting
	# multiplications together
	warpedMultiplied = cv2.multiply(warped.astype("float"),
		maskScaled)
	imageMultiplied = cv2.multiply(frame.astype(float),
		1.0 - maskScaled)
	output = cv2.add(warpedMultiplied, imageMultiplied)
	output = output.astype("uint8")

	# return the output frame to the calling function
	return output
```

**行 104-109** 将``warped`` 图像复制到输出`frame`上，然后我们返回到**行 112 上的调用函数。**

有关实际单应矩阵构建、warp 变换和后处理任务的更多详细信息，请参考上周的指南。

### **创建我们的 OpenCV 视频增强现实驱动脚本**

```py
# import the necessary packages
from pyimagesearch.augmented_reality import find_and_warp
from imutils.video import VideoStream
from collections import deque
import argparse
import imutils
import time
import cv2
```

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input video file for augmented reality")
ap.add_argument("-c", "--cache", type=int, default=-1,
	help="whether or not to use reference points cache")
args = vars(ap.parse_args())
```

```py
# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] initializing marker detector...")
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters_create()

# initialize the video file stream
print("[INFO] accessing video stream...")
vf = cv2.VideoCapture(args["input"])

# initialize a queue to maintain the next frame from the video stream
Q = deque(maxlen=128)

# we need to have a frame in our queue to start our augmented reality
# pipeline, so read the next frame from our video file source and add
# it to our queue
(grabbed, source) = vf.read()
Q.appendleft(source)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

```

```py
# loop over the frames from the video stream
while len(Q) > 0:
	# grab the frame from our video stream and resize it
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	# attempt to find the ArUCo markers in the frame, and provided
	# they are found, take the current source image and warp it onto
	# input frame using our augmented reality technique
	warped = find_and_warp(
		frame, source,
		cornerIDs=(923, 1001, 241, 1007),
		arucoDict=arucoDict,
		arucoParams=arucoParams,
		useCache=args["cache"] > 0)
```

```py
	# if the warped frame is not None, then we know (1) we found the
	# four ArUCo markers and (2) the perspective warp was successfully
	# applied
	if warped is not None:
		# set the frame to the output augment reality frame and then
		# grab the next video file frame from our queue
		frame = warped
		source = Q.popleft()

	# for speed/efficiency, we can use a queue to keep the next video
	# frame queue ready for us -- the trick is to ensure the queue is
	# always (or nearly full)
	if len(Q) != Q.maxlen:
		# read the next frame from the video file stream
		(grabbed, nextFrame) = vf.read()

		# if the frame was read (meaning we are not at the end of the
		# video file stream), add the frame to our queue
		if grabbed:
			Q.append(nextFrame)
```

```py
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

### **利用 OpenCV 实现实时视频流中的增强现实**

准备好使用 OpenCV 在实时视频流中执行增强现实了吗？

首先使用本教程的 ***【下载】*** 部分下载源代码和示例视频。

从那里，打开一个终端，并执行以下命令:

```py
$ python opencv_ar_video.py --input videos/jp_trailer_short.mp4
[INFO] initializing marker detector...
[INFO] accessing video stream...
[INFO] starting video stream...
```

正如您从我的输出中看到的，我们是:

1.  从*读取帧*我的相机传感器以及*侏罗纪公园*驻留在磁盘上的预告片视频
2.  检测卡上的 ArUco 标签
3.  应用透视扭曲将视频帧从*侏罗纪公园*预告片转换到我的相机捕捉的真实世界环境中

此外，请注意，我们的增强现实应用程序是实时运行的！

但是，有一点问题…

**注意在输出帧中出现了*相当多的闪烁*——*为什么会这样？***

原因是 ArUco 标记检测并不完全“稳定”在一些帧中，所有四个标记都被检测到，而在其他帧中，它们没有被检测到。

一个理想的解决方案是确保所有四个标记都被检测到，但这并不是在所有情况下都能保证。

相反，我们可以依靠参考点缓存:

```py
$ python opencv_ar_video.py --input videos/jp_trailer_short.mp4 --cache 1
[INFO] initializing marker detector...
[INFO] accessing video stream...
[INFO] starting video stream...
```

使用参考点缓存，你现在可以看到我们的结果稍微好一点。当四个 ArUco 标记在*当前*帧中未被检测到时，我们退回到它们在*先前*帧中的位置，在那里四个标记都被检测到。

另一个潜在的解决方案是利用光流来帮助参考点跟踪(但这个主题超出了本教程的范围)。

## **总结**

在本教程中，您学习了如何使用 OpenCV 执行实时增强现实。

使用 OpenCV，我们能够访问我们的网络摄像头，[检测 ArUco 标签](https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/)，然后将输入图像/帧转换到我们的场景中，*，所有这些都是实时运行的！*

**然而，这种增强现实方法的最大缺点之一是它*要求*我们使用标记/基准，如 ArUco 标签、AprilTags 等。**

有一个活跃的增强现实研究领域叫做**无标记增强现实。**

有了无标记增强现实，我们*不需要*事先知道真实世界的环境，比如*有*驻留在我们视频流中的特定标记或物体。

无标记增强现实带来了更加美丽、身临其境的体验；然而，大多数无标记增强现实系统需要平坦的纹理/区域才能工作。

此外，无标记增强现实需要*明显*更复杂和计算成本更高的算法。

我们将在 PyImageSearch 博客的未来一组教程中介绍无标记增强现实。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***