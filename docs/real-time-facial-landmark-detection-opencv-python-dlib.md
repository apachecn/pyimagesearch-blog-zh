# 使用 OpenCV、Python 和 dlib 进行实时面部标志检测

> 原文：<https://pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/>

在过去的几周里，我们一直在讨论面部标志以及它们在计算机视觉和图像处理中的作用。

![](img/0d0ffae2c390dec2cb33dee104cdd83e.png)

我们已经开始学习如何在图像中检测面部标志。

然后我们发现如何[标记和注释*每个面部区域*](https://pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/) ，比如眼睛、眉毛、鼻子、嘴和下颌线。

今天，我们将扩展面部标志的实现，以在*实时视频流*中工作，为更多现实世界的应用铺平道路，包括下周关于*眨眼检测的教程。*

要了解如何实时检测视频流中的面部标志，请继续阅读。

## 使用 OpenCV、Python 和 dlib 进行实时面部标志检测

这篇博文的第一部分将利用 Python、OpenCV 和 dlib 实现视频流中的实时面部标志检测。

然后，我们将测试我们的实现，并使用它来检测视频中的面部标志。

### 视频流中的面部标志

让我们开始这个面部标志的例子。

打开一个新文件，将其命名为`video_facial_landmarks.py`，并插入以下代码:

```py
# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

```

**第 2-9 行**导入我们需要的 Python 包。

我们将使用 [imutils](https://github.com/jrosebr1/imutils) 的`face_utils`子模块，因此如果您还没有安装/升级到最新版本，请花一点时间现在就安装/升级:

```py
$ pip install --upgrade imutils

```

***注意:*** *如果您正在使用 Python 虚拟环境，请注意确保您正在正确的环境中安装/升级`imutils`。*

我们还将在`imutils`中使用`VideoStream`实现，允许你以更高效的*、*更快的线程方式访问你的网络摄像头/USB 摄像头/Raspberry Pi 摄像头模块。在这篇博文中，你可以读到更多关于`VideoStream`类以及它如何实现一个更高的框架。**

如果你想使用*视频文件*而不是*视频流*，一定要参考[这篇关于有效帧轮询的博文](https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/)来自一个预先录制的视频文件，用`FileVideoStream`代替`VideoStream`。

对于我们的面部标志实现，我们将使用 [dlib 库](http://dlib.net/)。你可以在本教程中学习如何在你的系统[上安装 dlib(如果你还没有这样做的话)。](https://pyimagesearch.com/2017/03/27/how-to-install-dlib/)

接下来，让我们解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

```

我们的脚本需要一个命令行参数，后跟第二个可选参数，每个参数的详细信息如下:

*   `--shape-predictor`:dlib 预训练的面部标志检测器的路径。使用这篇博文的 ***“下载”*** 部分下载代码+面部标志预测器文件的存档。
*   `--picamera`:可选命令行参数，此开关指示是否应该使用 Raspberry Pi 摄像头模块，而不是默认的网络摄像头/USB 摄像头。提供一个值 *> 0* 来使用你的树莓 Pi 相机。

既然已经解析了我们的命令行参数，我们需要初始化 dlib 的 [HOG +基于线性 SVM 的面部检测器](https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)，然后从磁盘加载面部标志预测器:

```py
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

```

下一个代码块只是处理初始化我们的`VideoStream`并允许相机传感器预热:

```py
# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

```

我们视频处理管道的核心可以在下面的`while`循环中找到:

```py
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

```

在第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第 31 行第

第 35 行从我们的视频流中抓取下一帧。

然后我们预处理这个帧，将它的宽度调整为 400 像素，并将其转换为灰度(**行 36 和 37** )。

在我们可以在我们的帧中检测面部标志之前，我们首先需要定位面部——这是通过返回边界框 *(x，y)* 的`detector`在**行 40** 上完成的——图像中每个面部的坐标。

既然我们已经在视频流中检测到了面部，下一步就是将面部标志预测器应用于每个面部 ROI:

```py
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

```

在第 43 行第 43 行第 43 行第 43 行第 43 行第 43 行第 43 行第 41 行，我们对每个检测到的人脸进行循环。

**第 47 行**将面部标志检测器应用到面部区域，返回一个`shape`对象，我们将其转换为一个 NumPy 数组(**第 48 行**)。

**第 52 行和第 53 行**然后在输出`frame`上画一系列圆圈，可视化每个面部标志。了解面部什么部位(即鼻子、眼睛、嘴巴等。)每个 *(x，y)*-坐标映射到，[请参考这篇博文](https://pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/)。

**第 56 和 57 行**显示输出`frame`到我们的屏幕。如果按下`q`键，我们从循环中断开并停止脚本(**第 60 行和第 61 行**)。

最后，**行 64 和 65** 做了一些清理工作:

```py
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

```

正如你所看到的，在*图像* 中检测面部标志的[与在*视频流*中检测面部标志的](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)[之间几乎没有区别——代码中的主要区别只是涉及设置我们的视频流指针，然后轮询视频流中的帧。](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)

检测面部标志的实际过程是*相同的，*不同于在*单一图像*中检测面部标志，我们现在检测在*系列帧中的面部标志。*

### 实时面部标志结果

要使用 OpenCV、Python 和 dlib 测试我们的实时面部标志检测器，请确保您使用这篇博客文章的 ***【下载】*** 部分下载代码、项目结构和面部标志预测器模型的档案。

如果您使用标准网络摄像头/USB 摄像头，您可以执行以下命令来启动视频面部标志预测器:

```py
$ python video_facial_landmarks.py \
	--shape-predictor shape_predictor_68_face_landmarks.dat

```

否则，如果您使用的是 Raspberry Pi，请确保将`--picamera 1`开关附加到命令中:

```py
$ python video_facial_landmarks.py \
	--shape-predictor shape_predictor_68_face_landmarks.dat \
	--picamera 1

```

这是一个简短的 GIF 输出，您可以看到面部标志已经成功地实时检测到我的面部:

![](img/0d0ffae2c390dec2cb33dee104cdd83e.png)

**Figure 1:** A short demo of real-time facial landmark detection with OpenCV, Python, an dlib.

我已经包括了一个完整的视频输出如下:

<https://www.youtube.com/embed/pD0gVP0aw3Q?feature=oembed>