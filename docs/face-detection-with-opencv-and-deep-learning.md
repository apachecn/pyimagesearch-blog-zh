# 基于 OpenCV 和深度学习的人脸检测

> 原文：<https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/>

最后更新于 2021 年 7 月 4 日。

今天，我将与您分享一个关于 OpenCV 库的鲜为人知的秘密:

**您可以使用库附带的预训练深度学习人脸检测器模型，通过 OpenCV 执行*快速、准确的*人脸检测。**

您可能已经知道 OpenCV 出厂时就带有预先训练好的 Haar 级联，可用于面部检测…

…但我敢打赌，你不知道从 OpenCV 3.3 开始就成为 OpenCV 一部分的**“隐藏”的基于深度学习的人脸检测器**。

在今天博文的剩余部分，我将讨论:

*   这个“隐藏的”深度学习人脸检测器在 OpenCV 库中的位置
*   如何使用 OpenCV 和深度学习在*图像* 中执行**人脸检测**
*   如何使用 OpenCV 和深度学习在*视频* 中执行**人脸检测**

正如我们将看到的，很容易将哈尔级联替换为更准确的深度学习人脸检测器。

**要了解更多关于 OpenCV 和深度学习的人脸检测，*继续阅读！***

*   【2021 年 7 月更新:包括了一个新的部分，关于您可能希望在您的项目中考虑的替代人脸检测方法。

将 OpenCV 的深度神经网络模块与 Caffe 模型一起使用时，您将需要两组文件:

*   **。定义*模型架构*的 prototxt** 文件(即层本身)
*   **。包含实际层的*权重*的 caffemodel** 文件

当使用使用 Caffe 训练的模型进行深度学习时，这两个文件都是必需的。

然而，您只能在 GitHub repo 中找到 prototxt 文件。

权重文件*不*包含在 OpenCV `samples`目录中，需要更多的挖掘才能找到它们…

### 我在哪里可以得到更准确的 OpenCV 人脸检测器？

为了您的方便，我把*和*都包括在内了:

1.  Caffe prototxt 文件
2.  和咖啡模型重量文件

**…在这篇博文的*“下载”*部分。**

要跳到下载部分，只需点击这里。

### OpenCV 深度学习人脸检测器是如何工作的？

OpenCV 的深度学习人脸检测器基于具有 ResNet 基础网络的单镜头检测器(SSD)框架(与您可能已经看到的通常使用 MobileNet 作为基础网络的其他 OpenCV SSDs 不同)。

对 SSD 和 ResNet 的全面回顾超出了这篇博客的范围，所以如果你有兴趣了解更多关于单次检测器的信息(包括如何训练你自己的定制深度学习对象检测器)，从 PyImageSearch 博客上的这篇文章[开始，然后看看我的书，](https://pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/)*[用 Python](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)* 进行计算机视觉的深度学习，其中包括深入的讨论和代码，使你能够训练你自己的对象检测器。

### 基于 OpenCV 和深度学习的图像人脸检测

在第一个例子中，我们将学习如何使用 OpenCV 对单个输入图像进行人脸检测。

在下一节中，我们将学习如何修改这段代码，并使用 OpenCV 将人脸检测应用于视频、视频流和网络摄像头。

打开一个新文件，将其命名为`detect_faces.py`，并插入以下代码:

```py
# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

```

这里我们导入我们需要的包(**第 2-4 行**)并解析命令行参数(**第 7-16 行**)。

我们有三个必需的参数:

*   `--image`:输入图像的路径。
*   【the Caffe prototxt 文件的路径。
*   `--model`:通往预训练的 Caffe 模型的道路。

可选参数`--confidence`可以覆盖默认阈值 0.5，如果您愿意的话。

从那里，让我们加载我们的模型，并从我们的图像创建一个斑点:

```py
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

```

首先，我们使用我们的`--prototxt`和`--model`文件路径加载我们的模型。我们把型号存为`net`(**20 线**)。

然后我们加载`image` ( **第 24 行**)，提取维度(**第 25 行**)，创建一个`blob` ( **第 26 行和第 27 行**)。

`dnn.blobFromImage`负责预处理，包括设置`blob`尺寸和标准化。如果你有兴趣了解更多关于`dnn.blobFromImage`功能的信息，我会在[这篇博文](https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)中详细回顾。

接下来，我们将应用面部检测:

```py
# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

```

为了检测人脸，我们将第 32 和 33 行**上的`blob`通过`net`。**

从这里开始，我们将在`detections`上循环，并在检测到的人脸周围绘制方框:

```py
# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)

```

我们开始在第 36 行**上循环检测。**

从那里，我们提取`confidence` ( **第 39 行**)并将其与置信度阈值(**第 43 行**)进行比较。我们执行此检查以过滤掉弱检测。

如果置信度满足最小阈值，我们继续画一个矩形，并沿着**行 46-56** 上检测的*概率*。

为此，我们首先计算边界框的 *(x，y)*——坐标(**第 46 行和第 47 行**)。

然后，我们建立我们的信心`text`字符串(**行 51** )，它包含检测的概率。

万一我们的`text`会偏离图像(例如当面部检测出现在图像的最顶端时)，我们将其下移 10 个像素(**第 52 行**)。

我们的脸矩形和信心`text`画在**线 53-56** 的`image`上。

从那里，我们再次循环进行后续的额外检测。如果没有剩余的`detections`，我们准备在屏幕上显示我们的输出`image`(**行 59 和 60** )。

#### 基于 OpenCV 结果的图像人脸检测

我们来试试 OpenCV 深度学习人脸检测器。

**确保你使用了这篇博文的*“下载”*部分来下载:**

*   这篇博文中使用的**源代码**
*   用于深度学习人脸检测的 **Caffe prototxt 文件**
*   用于深度学习人脸检测的 **Caffe 权重文件**
*   本帖中使用的**示例图片**

从那里，打开一个终端并执行以下命令:

```py
$ python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt \
	--model res10_300x300_ssd_iter_140000.caffemodel

```

上面的照片是我第一次去佛罗里达州的 Ybor 市时的照片，在那里，鸡可以在整个城市自由漫步。甚至有保护鸡的法律，我认为这很酷。即使我是在农村的农田里长大的，我仍然对看到一只公鸡过马路感到十分惊讶——这当然引发了许多*“小鸡为什么要过马路？”*笑话。

这里你可以看到我的脸以 74.30%的置信度被检测到，尽管我的脸有一个角度。OpenCV 的 Haar cascades 因丢失不在“直上”角度的面部而臭名昭著，但通过使用 OpenCV 的深度学习面部检测器，我们能够检测到我的面部。

现在我们来看看另一个例子是如何工作的，这次有三张脸:

```py
$ python detect_faces.py --image iron_chic.jpg --prototxt deploy.prototxt.txt \
	--model res10_300x300_ssd_iter_140000.caffemodel

```

这张照片是在佛罗里达州盖恩斯维尔拍摄的，当时我最喜欢的乐队之一在该地区一个很受欢迎的酒吧和音乐场所 Loosey's 结束了一场演出。在这里你可以看到我的未婚妻(*左*)、我(*中*)和乐队成员杰森(*右*)。

令我难以置信的是，OpenCV 可以检测到 Trisha 的脸，尽管在黑暗的场地中灯光条件和阴影投射在她的脸上(并且有 86.81%的概率！)

同样，这只是表明深度学习 OpenCV 人脸检测器比库附带的标准 Haar cascade 检测器好多少(在准确性方面)。

```py
# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

```

与上面相比，我们将需要导入三个额外的包:`VideoStream`、`imutils`和`time`。

如果您的虚拟环境中没有`imutils`，您可以通过以下方式安装:

```py
$ pip install imutils

```

我们的命令行参数大部分是相同的，除了这次我们没有一个`--image`路径参数。我们将使用网络摄像头的视频。

从那里，我们将加载我们的模型并初始化视频流:

```py
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

```

加载模型同上。

我们初始化一个`VideoStream`对象，指定索引为零的相机作为源(一般来说，这将是您的笔记本电脑的内置相机或您的台式机的第一个检测到的相机)。

这里有一些简短的注释:

*   **Raspberry Pi + picamera 用户**如果希望使用 Raspberry Pi 相机模块，可以将**第 25 行**替换为`vs = VideoStream(usePiCamera=True).start()`。
*   如果你要解析一个**视频文件**(而不是一个视频流)，用`VideoStream`类替换`FileVideoStream`。你可以在这篇博文中了解更多关于[filevodestream 类的信息。](https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/)

然后，我们让摄像机传感器预热 2 秒钟(**第 26 行**)。

从那里，我们循环遍历帧，并使用 OpenCV 计算人脸检测:

```py
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

```

这个块应该与上一节中的静态图像版本非常相似。

在这个模块中，我们从视频流中读取一个`frame`(**第 32 行**)，创建一个`blob` ( **第 37 行和第 38 行**)，并让`blob`通过深度神经`net`来获得人脸检测(**第 42 行和第 43 行**)。

我们现在可以循环检测，与置信度阈值进行比较，并在屏幕上绘制面部框+置信度值:

```py
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

```

要详细查看此代码块，请查看上一节，其中我们对静止的静态图像执行人脸检测。这里的代码几乎相同。

既然我们的 OpenCV 人脸检测已经完成，让我们在屏幕上显示该帧并等待按键:

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

我们在屏幕上显示`frame`，直到按下“q”键，这时我们`break`退出循环并执行清除。

#### 使用 OpenCV 结果在视频和网络摄像头中进行人脸检测

要试用 OpenCV 深度学习人脸检测器，请确保您使用这篇博客文章的 ***【下载】*** 部分来获取:

*   这篇博文中使用的**源代码**
*   用于深度学习人脸检测的 **Caffe prototxt 文件**
*   用于深度学习人脸检测的 **Caffe 权重文件**

下载完文件后，使用这个简单的命令就可以轻松运行带有网络摄像头的深度学习 OpenCV 人脸检测器:

```py
$ python detect_faces_video.py --prototxt deploy.prototxt.txt \
	--model res10_300x300_ssd_iter_140000.caffemodel

```

你可以在下面的视频中看到完整的视频演示，包括我的评论:

<https://www.youtube.com/embed/v_rZcmLVma8?feature=oembed>