# OpenCV 快速傅立叶变换(FFT ),用于图像和视频流中的模糊检测

> 原文：<https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/>

在本教程中，您将学习如何使用 OpenCV 和快速傅立叶变换(FFT)在图像和实时视频流中执行模糊检测。

今天的教程是我之前关于 OpenCV 模糊检测的博文的延伸。原始模糊检测方法:

*   依赖于计算拉普拉斯算子的方差
*   仅用一行代码就可以实现
*   使用起来非常简单吗

**缺点是拉普拉斯方法*需要大量的手动*调谐来定义图像被认为模糊与否的“阈值”。**如果你能控制你的照明条件、环境和图像捕捉过程，它会工作得很好——但如果不能，至少可以说，你会得到混合的结果。

我们今天要讨论的方法依赖于计算图像的快速傅立叶变换。它仍然需要一些手动调整，但正如我们将发现的，我们将涉及的 FFT 模糊检测器*比拉普拉斯方法的方差*更加鲁棒和可靠。

在本教程结束时，你将拥有一个全功能的 FFT 模糊检测器，可以应用于图像和视频流。

**要了解如何使用 OpenCV 和快速傅立叶变换(FFT)来执行模糊检测，*请继续阅读。***

**注:**博文更新于 2022 年 1 月 22 日。

## OpenCV 快速傅立叶变换(FFT)用于模糊检测

在本教程的第一部分，我们将简要讨论:

*   什么是模糊检测
*   为什么我们可能想要检测图像/视频流中的模糊
*   以及快速傅立叶变换如何让我们发现模糊。

在此基础上，我们将实现针对图像和实时视频的 FFT 模糊检测器。

我们将通过回顾 FFT 模糊检测器的结果来结束本教程。

### 什么是模糊检测，我们什么时候需要检测模糊？

**模糊检测，顾名思义，就是检测图像是否模糊的过程。**

模糊检测的可能应用包括:

*   自动图像质量分级
*   通过自动丢弃模糊/低质量的照片，帮助专业摄影师在照片拍摄过程中整理 100 到 1000 张照片
*   对实时视频流应用 OCR，但仅对非模糊帧应用昂贵的 OCR 计算

这里的关键要点是，为在理想条件下捕获的图像编写计算机视觉代码总是更容易。

不是试图处理图像质量非常差的边缘情况，而是简单地检测并丢弃质量差的图像(例如具有明显模糊的图像)。

这种模糊检测程序可以自动丢弃质量差的图像，或者简单地告诉最终用户*“嘿，伙计，再试一次。让我们在这里捕捉更好的图像。”*

请记住，计算机视觉应用程序应该是智能的，因此出现了术语*人工智能*——有时，“智能”可以只是检测输入数据的质量是否很差，而不是试图理解它。

### 什么是快速傅立叶变换(FFT)？

快速傅立叶变换是用于计算离散傅立叶变换的方便的数学算法。它用于将信号从一个域转换到另一个域。

FFT 在许多学科中都很有用，包括音乐、数学、科学和工程。例如，电气工程师，特别是处理无线、电源和音频信号的工程师，需要 FFT 计算将时间序列信号转换到频域，因为有些计算在频域中更容易完成。相反，可以使用 FFT 将频域信号转换回时域。

就计算机视觉而言，我们通常认为 FFT 是一种图像处理工具，它在两个领域中表示图像:

1.  傅立叶(即频率)域
2.  空间域

因此，FFT 以*实数*和*虚数*分量表示图像。

通过分析这些值，我们可以执行图像处理程序，如模糊、边缘检测、阈值处理、纹理分析和*是的，甚至模糊检测。*

回顾快速傅立叶变换的数学细节超出了这篇博客的范围，所以如果你有兴趣了解更多，我建议你阅读这篇关于 FFT 及其与图像处理的关系的文章。

对于有学术倾向的读者来说，[看看亚伦·博比克在佐治亚理工学院的计算机视觉课程](https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2014/slides/CS4495-Frequency.pdf)中精彩的幻灯片。

最后，傅立叶变换的维基百科页面更加详细地介绍了数学，包括它在非图像处理任务中的应用。

### 项目结构

首先使用本教程的 ***【下载】*** 部分下载源代码和示例图像。提取文件后，您将拥有一个如下组织的目录:

```py
$ tree --dirsfirst
.
├── images
│   ├── adrian_01.png
│   ├── adrian_02.png
│   ├── jemma.png
│   └── resume.png
├── pyimagesearch
│   ├── __init__.py
│   └── blur_detector.py
├── blur_detector_image.py
└── blur_detector_video.py

2 directories, 8 files
```

在下一节中，我们将实现基于 FFT 的模糊检测算法。

### 用 OpenCV 实现 FFT 模糊检测器

我们现在准备用 OpenCV 实现我们的快速傅立叶变换模糊检测器。

我们将要介绍的方法是基于刘等人 2008 年在发表的[之后的实现](https://github.com/whdcumt/BlurDetection)，[图像局部模糊检测和分类](http://www.cse.cuhk.edu.hk/leojia/all_final_papers/blur_detect_cvpr08.pdf)。

在我们的目录结构中打开`blur_detector.py`文件，并插入以下代码:

```py
# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np

def detect_blur_fft(image, size=60, thresh=10, vis=False):
	# grab the dimensions of the image and use the dimensions to
	# derive the center (x, y)-coordinates
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))
```

```py
	# compute the FFT to find the frequency transform, then shift
	# the zero frequency component (i.e., DC component located at
	# the top-left corner) to the center where it will be more
	# easy to analyze
	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)
```

这里，使用 NumPy 的内置算法，我们**计算 FFT** ( **第 15 行**)。

然后，我们将结果的零频率分量(DC 分量)移到中心，以便于分析(**第 16 行**)。

现在我们已经有了`image`的 FFT，让我们来看看设置了`vis`标志后的结果:

```py
	# check to see if we are visualizing our output
	if vis:
		# compute the magnitude spectrum of the transform
		magnitude = 20 * np.log(np.abs(fftShift))

		# display the original input image
		(fig, ax) = plt.subplots(1, 2, )
		ax[0].imshow(image, cmap="gray")
		ax[0].set_title("Input")
		ax[0].set_xticks([])
		ax[0].set_yticks([])

		# display the magnitude image
		ax[1].imshow(magnitude, cmap="gray")
		ax[1].set_title("Magnitude Spectrum")
		ax[1].set_xticks([])
		ax[1].set_yticks([])

		# show our plots
		plt.show()
```

```py
	# zero-out the center of the FFT shift (i.e., remove low
	# frequencies), apply the inverse shift such that the DC
	# component once again becomes the top-left, and then apply
	# the inverse FFT
	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)
```

在此，我们:

*   通过**线 43** 将 FFT 偏移的中心归零(即去除低频)
*   应用逆移位将 DC 组件放回左上角(**行 44** )
*   应用逆 FFT ( **行 45** )

从这里开始，我们还有三个步骤来确定我们的`image`是否模糊:

```py
	# compute the magnitude spectrum of the reconstructed image,
	# then compute the mean of the magnitude values
	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)

	# the image will be considered "blurry" if the mean value of the
	# magnitudes is less than the threshold value
	return (mean, mean <= thresh)
```

实现基于 FFT 的模糊检测算法做得很好。但是我们还没有完成。在下一节中，我们将把我们的算法应用于静态图像，以确保它按照我们的预期执行。

### 用 FFT 检测图像中的模糊

```py
# import the necessary packages
from pyimagesearch.blur_detector import detect_blur_fft
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path input image that we'll detect blur in")
ap.add_argument("-t", "--thresh", type=int, default=20,
	help="threshold for our blur detector to fire")
ap.add_argument("-v", "--vis", type=int, default=-1,
	help="whether or not we are visualizing intermediary steps")
ap.add_argument("-d", "--test", type=int, default=-1,
	help="whether or not we should progressively blur the image")
args = vars(ap.parse_args())
```

```py
# load the input image from disk, resize it, and convert it to
# grayscale
orig = cv2.imread(args["image"])
orig = imutils.resize(orig, width=500)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

# apply our blur detector using the FFT
(mean, blurry) = detect_blur_fft(gray, size=60,
	thresh=args["thresh"], vis=args["vis"] > 0)
```

```py
# draw on the image, indicating whether or not it is blurry
image = np.dstack([gray] * 3)
color = (0, 0, 255) if blurry else (0, 255, 0)
text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
text = text.format(mean)
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	color, 2)
print("[INFO] {}".format(text))

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
```

```py
# check to see if are going to test our FFT blurriness detector using
# various sizes of a Gaussian kernel
if args["test"] > 0:
	# loop over various blur radii
	for radius in range(1, 30, 2):
		# clone the original grayscale image
		image = gray.copy()

		# check to see if the kernel radius is greater than zero
		if radius > 0:
			# blur the input image by the supplied radius using a
			# Gaussian kernel
			image = cv2.GaussianBlur(image, (radius, radius), 0)

			# apply our blur detector using the FFT
			(mean, blurry) = detect_blur_fft(image, size=60,
				thresh=args["thresh"], vis=args["vis"] > 0)

			# draw on the image, indicating whether or not it is
			# blurry
			image = np.dstack([image] * 3)
			color = (0, 0, 255) if blurry else (0, 255, 0)
			text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
			text = text.format(mean)
			cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, color, 2)
			print("[INFO] Kernel: {}, Result: {}".format(radius, text))

		# show the image
		cv2.imshow("Test Image", image)
		cv2.waitKey(0)
```

### 图像结果中的 FFT 模糊检测

我们现在准备使用 OpenCV 和快速傅立叶变换来检测图像中的模糊。

首先确保使用本教程的 ***“下载”*** 部分下载源代码和示例图像。

从那里，打开一个终端，并执行以下命令:

```py
$ python blur_detector_image.py --image images/adrian_01.png
[INFO] Not Blurry (42.4630)
```

这里你可以看到我在锡安国家公园 — **的地铁里徒步[的输入图像，图像被正确地标记为*没有模糊。*](https://www.nps.gov/zion/planyourvisit/thesubway.htm)**

让我们尝试另一个图像，这是我家的狗 Jemma:

```py
$ python blur_detector_image.py --image images/jemma.png
[INFO] Blurry (12.4738)
```

```py
$ python blur_detector_image.py --image images/adrian_02.png --test 1
[INFO] Not Blurry (32.0934)
[INFO] Kernel: 1, Result: Not Blurry (32.0934)
[INFO] Kernel: 3, Result: Not Blurry (25.1770)
[INFO] Kernel: 5, Result: Not Blurry (20.5668)
[INFO] Kernel: 7, Result: Blurry (13.4830)
[INFO] Kernel: 9, Result: Blurry (7.8893)
[INFO] Kernel: 11, Result: Blurry (0.6506)
[INFO] Kernel: 13, Result: Blurry (-5.3609)
[INFO] Kernel: 15, Result: Blurry (-11.4612)
[INFO] Kernel: 17, Result: Blurry (-17.0109)
[INFO] Kernel: 19, Result: Blurry (-19.6464)
[INFO] Kernel: 21, Result: Blurry (-20.4758)
[INFO] Kernel: 23, Result: Blurry (-20.7365)
[INFO] Kernel: 25, Result: Blurry (-20.9362)
[INFO] Kernel: 27, Result: Blurry (-21.1911)
[INFO] Kernel: 29, Result: Blurry (-21.3853
```

如果您使用上面看到的`test`例程，您将应用一系列有意的模糊，并使用我们的快速傅立叶变换(FFT)方法来确定图像是否模糊。这个测试程序是有用的，因为它允许你调整你的模糊阈值参数。

我鼓励你自己去做，看看结果。欢迎在 Twitter @PyImageSearch 上与我们分享。

**在这里，你可以看到随着我们的图像变得*越来越模糊，*平均 FFT 幅度值*降低。***

我们的 FFT 模糊检测方法也可以应用于非自然场景图像。

例如，假设我们想要构建一个自动文档扫描仪应用程序——这样的计算机视觉项目应该自动拒绝模糊的图像。

然而，文档图像与自然场景图像有很大的不同，本质上对模糊更加敏感。

任何类型的模糊都会严重影响 OCR 准确度*。*

因此，我们应该*增加*我们的`--thresh`值(我还将包括`--vis`参数，这样我们可以直观地看到 FFT 幅度值是如何变化的):

```py
$ python blur_detector_image.py --image images/resume.png --thresh 27 --test 1 --vis 1
[INFO] Not Blurry (34.6735)
[INFO] Kernel: 1, Result: Not Blurry (34.6735)
[INFO] Kernel: 3, Result: Not Blurry (29.2539)
[INFO] Kernel: 5, Result: Blurry (26.2893)
[INFO] Kernel: 7, Result: Blurry (21.7390)
[INFO] Kernel: 9, Result: Blurry (18.3632)
[INFO] Kernel: 11, Result: Blurry (12.7235)
[INFO] Kernel: 13, Result: Blurry (9.1489)
[INFO] Kernel: 15, Result: Blurry (2.3377)
[INFO] Kernel: 17, Result: Blurry (-2.6372)
[INFO] Kernel: 19, Result: Blurry (-9.1908)
[INFO] Kernel: 21, Result: Blurry (-15.9808)
[INFO] Kernel: 23, Result: Blurry (-20.6240)
[INFO] Kernel: 25, Result: Blurry (-29.7478)
[INFO] Kernel: 27, Result: Blurry (-29.0728)
[INFO] Kernel: 29, Result: Blurry (-37.7561)
```

如果你运行这个脚本(你应该这样做)，你会看到我们的图像很快变得模糊不清，并且 OpenCV FFT 模糊检测器正确地将这些图像标记为模糊。

### 利用 OpenCV 和 FFT 检测视频中的模糊

到目前为止，我们已经将快速傅立叶变换模糊检测器应用于图像。

但是有没有可能将 FFT 模糊检测应用到视频流中呢？

整个过程也能在*实时*完成吗？

让我们找出答案——打开一个新文件，将其命名为`blur_detector_video.py`,并插入以下代码:

```py
# import the necessary packages
from imutils.video import VideoStream
from pyimagesearch.blur_detector import detect_blur_fft
import argparse
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--thresh", type=int, default=10,
	help="threshold for our blur detector to fire")
args = vars(ap.parse_args())
```

```py
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=500)

	# convert the frame to grayscale and detect blur in it
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	(mean, blurry) = detect_blur_fft(gray, size=60,
		thresh=args["thresh"], vis=False)
```

**第 17 行和第 18 行**初始化我们的网络摄像头流，让摄像头有时间预热。

从那里，我们开始在**线 21** 上的帧处理循环。在里面，我们抓取一帧并将其转换为灰度(**第 24-28 行**)，就像我们的单个图像模糊检测脚本一样。

```py
	# draw on the frame, indicating whether or not it is blurry
	color = (0, 0, 255) if blurry else (0, 255, 0)
	text = "Blurry ({:.4f})" if blurry else "Not Blurry ({:.4f})"
	text = text.format(mean)
	cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
		0.7, color, 2)

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

此时，我们的最后一个代码块应该看起来非常熟悉，因为这是我们第三次看到这些代码行。在这里我们:

*   标注*模糊*(红色`text`)或*不模糊*(绿色`text`)以及`mean`值(**第 33-37 行**)
*   显示结果(**第 40 行**)
*   如果按下`q`键，退出(**第 41-45 行**，并执行内务清理(**第 48 和 49 行**)

### 快速傅立叶变换视频模糊检测结果

我们现在准备好了解我们的 OpenCV FFT 模糊检测器是否可以应用于实时视频流。

确保使用本教程的 ***【下载】*** 部分下载源代码。

从那里，打开一个终端，并执行以下命令:

```py
$ python blur_detector_video.py
[INFO] starting video stream...
```

当我移动我的笔记本电脑时，**运动模糊**被引入画面。

如果我们正在实现一个计算机视觉系统来自动提取关键、重要的帧，或者创建一个自动视频 OCR 系统，我们会想要*丢弃*这些模糊的帧——**使用我们的 OpenCV FFT 模糊检测器，我们完全可以做到这一点！**

## 摘要

在今天的教程中，您学习了如何使用 OpenCV 的快速傅立叶变换(FFT)实现来执行图像和实时视频流中的模糊检测。

虽然不像拉普拉斯模糊检测器的[方差那么简单，但 FFT 模糊检测器更加稳定，在现实应用中往往能提供更好的模糊检测精度。](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)

问题是 FFT 方法*仍然需要我们手动设置阈值，特别是 FFT 幅度的平均值。*

一个*理想的*模糊检测器将能够检测图像和视频流*中的模糊，而不需要*这样的阈值。

为了完成这项任务，我们需要一点机器学习——我将在未来的教程中介绍自动模糊检测器。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***