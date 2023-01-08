# 利用 OpenCV 和 GPU 提高文本检测速度

> 原文：<https://pyimagesearch.com/2022/03/14/improving-text-detection-speed-with-opencv-and-gpus/>

在本教程中，您将学习使用 OpenCV 和 GPU 来提高文本检测速度。

本教程是关于使用 Python 进行 OCR 的 4 部分系列的最后一部分:

1.  [*多栏表格 OCR*](https://pyimg.co/h18s2)
2.  [*OpenCV 快速傅立叶变换(FFT)用于图像和视频流中的模糊检测*](https://pyimg.co/7emby)
3.  [*OCR 识别视频流*](https://pyimg.co/k43vd)
4.  *使用 OpenCV 和 GPU 提高文本检测速度*(本教程)

**学习如何用 OpenCV 和 GPU 提高文本检测速度，** ***继续阅读。***

## **使用 OpenCV 和 GPU 提高文本检测速度**

到目前为止，除了 EasyOCR 之外的所有内容都集中在我们的 CPU 上执行 OCR。但是如果我们可以在我们的 GPU 上应用 OCR 呢？由于许多最先进的文本检测和 OCR 模型都是基于深度学习的，难道这些模型不能在 GPU 上运行得更快更有效吗？

答案是*是*；他们绝对可以。

本教程将向您展示如何使用 NVIDIA GPU 在 OpenCV 的`dnn`(深度神经网络)模块上运行高效准确的场景文本检测器(EAST)模型。正如我们将看到的，我们的文本检测吞吐率接近三倍，从每秒`~23`帧(FPS)提高到惊人的`~97` FPS！

在本教程中，您将:

*   了解如何使用 OpenCV 的`dnn`模块在基于 NVIDIA CUDA 的 GPU 上运行深度神经网络
*   实现一个 Python 脚本来测试 CPU 和 GPU 上的文本检测速度
*   实现第二个 Python 脚本，这个脚本在实时视频流中执行文本检测
*   比较在 CPU 和 GPU 上运行文本检测的结果

### **通过 OpenCV 使用 GPU 进行 OCR**

本教程的第一部分包括回顾我们的项目目录结构。

然后，我们将实现一个 Python 脚本，在 CPU 和 GPU 上对运行的文本检测进行基准测试。我们将运行这个脚本，并测量在 GPU 上运行文本检测对我们的 FPS 吞吐率有多大影响。

一旦我们测量了我们的 FPS 增加，我们将实现第二个 Python 脚本，这一个，在实时视频流中执行文本检测。

我们将讨论我们的结果来结束本教程。

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

在我们可以用 GPU 应用文本检测之前，我们首先需要回顾我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

从这里，看一下目录结构:

```py
|-- pyimagesearch
|   |-- __init__.py
|   |-- east
|   |   |-- __init__.py
|   |   |-- east.py
|-- ../models
|   |-- east
|   |   |-- frozen_east_text_detection.pb
-- images
|   |-- car_wash.png
|-- text_detection_speed.py
|-- text_detection_video.py
```

在本教程中，我们将回顾两个 Python 脚本:

1.  `text_detection_speed.py`:使用我们的`images`目录中的`car_wash.png`图像，在 CPU 和 GPU 上测试文本检测速度。
2.  `text_detection_video.py`:演示如何在你的 GPU 上执行实时文本检测。

### **实施我们的 OCR GPU 基准测试脚本**

在使用我们的 GPU 实现实时视频流中的文本检测之前，让我们首先*测试*在我们的 CPU 上运行 EAST 检测模型与我们的 GPU 相比，我们获得了多少加速。

要找到答案，请打开我们项目目录中的`text_detection_speed.py`文件，让我们开始吧:

```py
# import the necessary packages
from pyimagesearch.east import EAST_OUTPUT_LAYERS
import numpy as np
import argparse
import time
import cv2
```

**第 2-6 行**处理导入我们需要的 Python 包。我们需要 EAST 模型的输出层(**行 2** )来获取文本检测输出。如果您需要复习这些输出值，请务必参考《使用 OpenCV、Tesseract 和 Python 的[*OCR:OCR 简介*](https://pyimagesearch.com/ocr-with-opencv-tesseract-and-python/) 一书。

接下来，我们有命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-e", "--east", required=True,
	help="path to input EAST text detector")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-t", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
ap.add_argument("-c", "--min-conf", type=float, default=0.5,
	help="minimum probability required to inspect a text region")
ap.add_argument("-n", "--nms-thresh", type=float, default=0.4,
	help="non-maximum suppression threshold")
ap.add_argument("-g", "--use-gpu", type=bool, default=False,
	help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())
```

`--image`命令行参数指定了我们将执行文本检测的输入图像的路径。

**第 12-21 行**然后指定东文本检测模型的命令行参数。

最后，我们有我们的`--use-gpu`命令行参数。默认情况下，我们将使用我们的 CPU。但是通过指定这个参数(并且假设我们有一个支持 CUDA 的 GPU 和 OpenCV 的`dnn`模块在 NVIDIA GPU 支持下编译)，我们可以使用我们的 GPU 进行文本检测推断。

考虑到我们的命令行参数，我们现在可以加载 EAST text 检测模型，并设置我们是使用 CPU 还是 GPU:

```py
# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# check if we are going to use GPU
if args["use_gpu"]:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# otherwise we are using our CPU
else:
	print("[INFO] using CPU for inference...")
```

**第 28 行**从磁盘加载我们的 EAST 文本检测模型。

**第 31-35 行**检查是否提供了`--use-gpu`命令行参数，如果提供了，则表明我们想要使用我们的支持 NVIDIA CUDA 的 GPU。

***注:*** *要使用你的 GPU 进行神经网络推理，你需要有 OpenCV 的`dnn`模块在 NVIDIA CUDA 支持下编译。OpenCV 的`dnn`模块没有通过 pip 安装的 NVIDIA 支持。相反，你需要用 GPU 支持显式编译 OpenCV。我们在 PyImageSearch* *的* [*教程中介绍了如何做到这一点。*](http://pyimg.co/jfftm)

接下来，让我们从磁盘加载示例图像:

```py
# load the input image and then set the new width and height values
# based on our command line arguments
image = cv2.imread(args["image"])
(newW, newH) = (args["width"], args["height"])

# construct a blob from the image, set the blob as input to the
# network, and initialize a list that records the amount of time
# each forward pass takes
print("[INFO] running timing trials...")
blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
timings = []
```

**第 43 行**从磁盘加载我们的输入`--image`，而**第 50 行和第 51 行**构造一个`blob`对象，这样我们就可以将它传递给东方文本检测模型。

**第 52 行**将我们的`blob`设置为东方网络的输入，而**第 53 行**初始化一个`timings`列表以测量推断需要多长时间。

当使用 GPU 进行推理时，与其余预测相比，您的第一个预测往往非常慢，原因是您的 GPU 尚未“预热”。因此，在您的 GPU 上进行测量时，您通常希望获得几次预测的平均值。

在下面的代码块中，我们对`500`试验执行文本检测，记录每个预测需要多长时间:

```py
# loop over 500 trials to obtain a good approximation to how long
# each forward pass will take
for i in range(0, 500):
	# time the forward pass
	start = time.time()
	(scores, geometry) = net.forward(EAST_OUTPUT_LAYERS)
	end = time.time()
	timings.append(end - start)

# show average timing information on text prediction
avg = np.mean(timings)
print("[INFO] avg. text detection took {:.6f} seconds".format(avg))
```

在所有试验完成后，我们计算`timings`的平均值，然后在终端上显示我们的平均文本检测时间。

### **速度测试:有无 GPU 的 OCR**

现在让我们在没有 GPU(即运行在 CPU 上)的情况下测量我们的 EAST text detection FPS 吞吐率*:*

```py
$ python text_detection_speed.py --image images/car_wash.png --east ../models/east/frozen_east_text_detection.pb
[INFO] loading EAST text detector...
[INFO] using CPU for inference...
[INFO] running timing trials...
[INFO] avg. text detection took 0.108568 seconds
```

我们的平均文本检测速度是`~0.1`秒，相当于`~9-10` FPS。在 CPU 上运行的深度学习模型对于许多应用来说是快速且足够的。

然而，就像 20 世纪 90 年代电视剧《T2 家装》中的蒂姆·泰勒(由《玩具总动员》中的蒂姆·艾伦饰演)，说的那样，*“更强大！”*

现在让我们来看看 GPU:

```py
$ python text_detection_speed.py --image images/car_wash.png --east ../models/east/frozen_east_text_detection.pb --use-gpu 1
[INFO] loading EAST text detector...
[INFO] setting preferable backend and target to CUDA...
[INFO] running timing trials...
[INFO] avg. text detection took 0.004763 seconds
```

使用 NVIDIA V100 GPU，我们的平均帧处理速率降至`~0.004`秒，这意味着我们现在可以处理`~250` FPS！**如你所见，使用你的 GPU 带来了*实质性的*差异！**

### **在 GPU 上对实时视频流进行 OCR**

准备好使用您的 GPU 实现我们的脚本来执行实时视频流中的文本检测了吗？

打开项目目录中的`text_detection_video.py`文件，让我们开始吧:

```py
# import the necessary packages
from pyimagesearch.east import EAST_OUTPUT_LAYERS
from pyimagesearch.east import decode_predictions
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
```

**第 2-10 行**导入我们需要的 Python 包。`EAST_OUTPUT_LAYERS`和`decode_predictions`函数来自我们在教程 [OpenCV 文本检测](https://pyimg.co/1vorv)中实现的东方文本检测器。如果您需要复习 EAST 检测模型，请务必复习该课程。

**第 4 行**导入我们的`VideoStream`来访问我们的网络摄像头，而**第 5 行**提供我们的`FPS`类来测量我们流水线的 FPS 吞吐率。

现在让我们继续我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-e", "--east", required=True,
	help="path to input EAST text detector")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-t", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
ap.add_argument("-c", "--min-conf", type=float, default=0.5,
	help="minimum probability required to inspect a text region")
ap.add_argument("-n", "--nms-thresh", type=float, default=0.4,
	help="non-maximum suppression threshold")
ap.add_argument("-g", "--use-gpu", type=bool, default=False,
	help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())
```

这些命令行参数与前面的命令行参数几乎相同。唯一的例外是我们用一个`--input`参数替换了`--image`命令行参数，该参数指定了磁盘上可选视频文件的路径(以防万一我们想要使用视频文件而不是我们的网络摄像头)。

接下来，我们进行一些初始化:

```py
# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)
```

在这里，我们初始化我们的原始框架的宽度和高度，东方模型的新框架尺寸，随后是*原始*和*新*尺寸之间的比率。

下一个代码块处理从磁盘加载 EAST text 检测模型，然后设置我们是使用 CPU 还是 GPU 进行推理:

```py
# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# check if we are going to use GPU
if args["use_gpu"]:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# otherwise we are using our CPU
else:
	print("[INFO] using CPU for inference...")
```

我们的文本检测模型需要帧来操作，所以下一个代码块访问我们的网络摄像头或驻留在磁盘上的视频文件，这取决于是否提供了`--input`命令行参数:

```py
# if a video path was not supplied, grab the reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["input"])

# start the FPS throughput estimator
fps = FPS().start()
```

**Line 62** 开始测量我们的 FPS 吞吐率，以了解我们的文本检测管道在一秒钟内可以处理的帧数。

现在让我们开始循环视频流中的帧:

```py
# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame, maintaining the aspect ratio
	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()

	# if our frame dimensions are None, we still need to compute the
	# ratio of old frame dimensions to new frame dimensions
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)
```

**第 68 和 69 行**从我们的网络摄像头或视频文件中读取下一个`frame`。

如果我们确实在处理一个视频文件，**行 72** 检查我们是否在视频的末尾，如果是，我们`break`退出循环。

**第 81-84 行**获取输入`frame`的空间尺寸，然后计算原始帧尺寸与 EAST 模型所需尺寸的比率。

现在我们已经有了这些维度，我们可以构造我们对东方文本检测器的输入:

```py
	# construct a blob from the image and then perform a forward pass
	# of the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(EAST_OUTPUT_LAYERS)

	# decode the predictions from OpenCV's EAST text detector and
	# then apply non-maximum suppression (NMS) to the rotated
	# bounding boxes
	(rects, confidences) = decode_predictions(scores, geometry,
		minConf=args["min_conf"])
	idxs = cv2.dnn.NMSBoxesRotated(rects, confidences,
		args["min_conf"], args["nms_thresh"])
```

**第 88-91 行**从输入`frame`构建`blob`。然后，我们将这个`blob`设置为我们的东文本检测`net`的输入。执行网络的前向传递，产生我们的原始文本检测。

然而，我们的原始文本检测在我们的当前状态下是不可用的，所以我们对它们调用`decode_predictions`，产生文本检测的边界框坐标以及相关概率的二元组(**行 96 和 97** )。

然后，我们应用非最大值抑制来抑制弱的、重叠的边界框(否则，每个检测将有*多个*边界框)。

如果您需要关于这个代码块的更多细节，包括如何实现`decode_predictions`函数，请务必查看 [OpenCV 文本检测](https://pyimg.co/1vorv)，在那里我将更详细地介绍东方文本检测器。

在非最大值抑制(NMS)之后，我们现在可以在每个边界框上循环:

```py
	# ensure that at least one text bounding box was found
	if len(idxs) > 0:
		# loop over the valid bounding box indexes after applying NMS
		for i in idxs.flatten():
			# compute the four corners of the bounding box, scale the
			# coordinates based on the respective ratios, and then
			# convert the box to an integer NumPy array
			box = cv2.boxPoints(rects[i])
			box[:, 0] *= rW
			box[:, 1] *= rH
			box = np.int0(box)

			# draw a rotated bounding box around the text
			cv2.polylines(orig, [box], True, (0, 255, 0), 2)

	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("Text Detection", orig)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break
```

第 102 行验证至少找到了一个文本边界框，如果是，我们在应用 NMS 后循环遍历保留的边界框的索引。

对于每个结果索引，我们计算文本 ROI 的边界框，将边界框 *(x，y)*-坐标缩放回`orig`输入帧尺寸，然后在`orig`帧上绘制边界框(**第 108-114 行**)。

**第 117 行**更新我们的 FPS 吞吐量估算器，而**第 120-125 行**在我们的屏幕上显示输出文本检测。

这里的最后一步是停止我们的 FPS 时间，估算吞吐率，并释放任何视频文件指针:

```py
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a webcam, release the pointer
if not args.get("input", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
```

**第 128-130 行**停止我们的 FPS 定时器并逼近我们的文本检测流水线的 FPS。然后，我们释放所有视频文件指针，关闭 OpenCV 打开的所有窗口。

### **GPU 和 OCR 结果**

这一部分需要在配有 GPU 的机器上本地执行。在 NVIDIA RTX 2070 超级 GPU(配有 i9 9900K 处理器)上运行`text_detection_video.py`脚本后，我获得了`~97` FPS:

```py
$ python text_detection_video.py --east ../models/east/frozen_east_text_detection.pb --use-gpu 1
[INFO] loading EAST text detector...
[INFO] setting preferable backend and target to CUDA...
[INFO] starting video stream...
[INFO] elapsed time: 74.71
[INFO] approx. FPS: 96.80
```

当我在不使用任何 GPU 的情况下运行同样的脚本*时，我达到了`~23`的 FPS，比上面的结果慢了`~77%`*。**

```py
$ python text_detection_video.py --east ../models/east/frozen_east_text_detection.pb
[INFO] loading EAST text detector...
[INFO] using CPU for inference...
[INFO] starting video stream...
[INFO] elapsed time: 68.59
[INFO] approx. FPS: 22.70
```

如你所见，使用你的 GPU 可以*显著*提高你的文本检测管道的吞吐速度！

## **总结**

在本教程中，您学习了如何使用 GPU 在实时视频流中执行文本检测。由于许多文本检测和 OCR 模型都是基于深度学习的，因此使用 GPU(而不是 CPU)可以极大地提高帧处理吞吐率。

使用我们的 CPU，我们能够处理`~22-23` FPS。然而，通过在 OpenCV 的`dnn`模块上运行 EAST 模型，我们可以达到`~97` FPS！

如果你有可用的 GPU，一定要考虑利用它——你将能够实时运行文本检测模型！

### **引用信息**

**Rosebrock，A.** “使用 OpenCV 和 GPU 提高文本检测速度”， *PyImageSearch* ，D. Chakraborty，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva，R. Raha，A. Thanki，eds .，2022 年，【https://pyimg.co/9wde6 

```py
@incollection{Rosebrock_2022_Improving_Text,
  author = {Adrian Rosebrock},
  title = {Improving Text Detection Speed with {OpenCV} and {GPUs}},
  booktitle = {PyImageSearch},
  editor = {Devjyoti Chakraborty and Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/9wde6},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****