# 基于 dlib 的多目标跟踪

> 原文：<https://pyimagesearch.com/2018/10/29/multi-object-tracking-with-dlib/>

![](img/6b83cfbc2426eb6777bac1fb4dd6ff23.png)

在本教程中，您将学习如何使用 dlib 库有效地跟踪实时视频中的多个对象。

到目前为止，在这个关于对象跟踪的系列中，我们已经学习了如何:

1.  [用 OpenCV](https://pyimagesearch.com/2018/07/30/opencv-object-tracking/) 追踪*单个*物体
2.  [利用 OpenCV](https://pyimagesearch.com/2018/08/06/tracking-multiple-objects-with-opencv/) 追踪*多个*物体
3.  [用 dlib](https://pyimagesearch.com/2018/10/22/object-tracking-with-dlib/) 进行单个物体跟踪
4.  [跟踪并统计进入商店的人数](https://pyimagesearch.com/2018/08/13/opencv-people-counter/)

我们当然可以用 dlib 跟踪多个物体；然而，为了获得可能的最佳性能，我们需要利用*多处理*，并将对象跟踪器*分布在我们处理器的多个内核上。*

正确利用多处理技术使我们的 dlib 多目标跟踪每秒帧数(FPS)吞吐率*提高了 45%以上！*

**要了解如何使用 dlib 跟踪多个对象，*继续阅读！***

## 基于 dlib 的多目标跟踪

在本指南的第一部分，我将演示如何实现一个简单、简单的 dlib 多对象跟踪脚本。这个程序将跟踪视频中的多个对象；然而，我们会注意到脚本运行得有点慢。

为了提高我们的 FPS 吞吐率，我将向您展示一个更快、更高效的 dlib 多对象跟踪器实现。

最后，我将讨论一些改进和建议，您可以用它们来增强我们的多对象跟踪实现。

### 项目结构

首先，请确保使用本教程的 ***“下载”*** 部分下载源代码和示例视频。

从那里，您可以使用`tree`命令来查看我们的项目结构:

```py
$ tree
.
├── mobilenet_ssd
│   ├── MobileNetSSD_deploy.caffemodel
│   └── MobileNetSSD_deploy.prototxt
├── multi_object_tracking_slow.py
├── multi_object_tracking_fast.py
├── race.mp4
├── race_output_slow.avi
└── race_output_fast.avi

1 directory, 7 files

```

目录包含我们的 MobileNet + SSD Caffe 模型文件，允许我们检测人(以及其他对象)。

今天我们将回顾两个 Python 脚本:

1.  dlib 多目标跟踪的简单“天真”方法。
2.  利用多重处理的先进、快速的方法。

剩下的三个文件是视频。我们有原始的`race.mp4`视频和两个处理过的输出视频。

### “天真”的 dlib 多目标跟踪实现

我们今天要讨论的第一个 dlib 多目标跟踪实现是“天真的”,因为它将:

1.  利用一个简单的跟踪对象列表。
2.  仅使用我们处理器的一个内核，按顺序更新每个跟踪器。

对于一些目标跟踪任务，这种实现将*多于足够的*；然而，为了优化我们的 FPS 吞吐率，我们应该将对象跟踪器分布在多个进程中。

在这一节中，我们将从简单的实现开始，然后在下一节中讨论更快的方法。

首先，打开`multi_object_tracking_slow.py`脚本并插入以下代码:

```py
# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import cv2

```

我们从在第 2-7 行导入必要的包和模块开始。最重要的是，我们将使用 dlib 和 OpenCV。我们还将使用我的 [imutils](https://github.com/jrosebr1/imutils) 便利功能包中的一些功能，例如每秒帧数计数器。

要安装 dlib，[请遵循本指南](https://pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/)。我也有一些 [OpenCV 安装教程](https://pyimagesearch.com/opencv-tutorials-resources-guides/)(甚至是最新的 OpenCV 4！).你甚至可以尝试最快的方法[通过 pip](https://pyimagesearch.com/2018/09/19/pip-install-opencv/) 在你的系统上安装 OpenCV。

要安装`imutils`，只需在终端中使用 pip:

```py
$ pip install --upgrade imutils

```

现在我们(a)已经安装了软件，并且(b)已经在脚本中放置了相关的导入语句，让我们解析我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

```

如果你不熟悉终端和命令行参数，请[阅读这篇文章](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/)。

我们的脚本在运行时处理以下命令行参数:

*   `--prototxt`:Caffe“deploy”proto txt 文件的路径。
*   `--model`:伴随 prototxt 的模型文件的路径。
*   `--video`:输入视频文件的路径。我们将在这段视频中使用 dlib 进行多目标跟踪。
*   `--output`:输出视频文件的可选路径。如果未指定路径，则不会将视频输出到磁盘。我建议输出到。avi 或. mp4 文件。
*   `--confidence`:对象检测置信度阈值`0.2`的可选覆盖。该值表示从对象检测器过滤弱检测的最小概率。

让我们定义该模型支持的`CLASSES`列表，并从磁盘加载我们的模型:

```py
# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

```

MobileNet SSD 预训练 Caffe 模型支持 20 个类和 1 个后台类。`CLASSES`以列表形式定义在**行 25-28** 上。

***注意:*** *如果您正在使用**“下载”**中提供的 Caffe 模型，请不要修改这个列表或类对象的排序。类似地，如果您碰巧装载了一个不同的模型，您将需要在这里定义模型支持的类(顺序很重要)。如果你对我们的物体探测器是如何工作的感到好奇，一定要参考这篇文章。*

对于今天的竞走例子，我们只关心`"person"`职业，但是你可以很容易地修改下面的**第 95 行**(将在本文后面讨论)来跟踪替代职业。

在**第 32 行**上，我们加载了我们预先训练好的对象检测器模型。我们将使用我们预先训练的 SSD 来检测视频中对象的存在。在那里，我们将创建一个 dlib 对象跟踪器来跟踪每个检测到的对象。

我们还需要执行一些初始化:

```py
# initialize the video stream and output video writer
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None

# initialize the list of object trackers and corresponding class
# labels
trackers = []
labels = []

# start the frames per second throughput estimator
fps = FPS().start()

```

在第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第 36 行第

随后，在**线 37** 上，我们的视频`writer`被初始化为`None`。在即将到来的`while`循环中，我们将更多地与视频`writer`合作。

现在让我们初始化第 41 和 42 行**上的`trackers`和`labels`列表。**

最后，我们在第 45 行启动每秒帧数计数器。

我们已经准备好开始处理我们的视频:

```py
# loop over frames from the video file stream
while True:
	# grab the next frame from the video file
	(grabbed, frame) = vs.read()

	# check to see if we have reached the end of the video file
	if frame is None:
		break

	# resize the frame for faster processing and then convert the
	# frame from BGR to RGB ordering (dlib needs RGB ordering)
	frame = imutils.resize(frame, width=600)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

```

在第 48 行的**上，我们开始在帧上循环，其中第 50** 行的**实际上抓取了`frame`本身。**

在第**行第 53 和 54** 行进行快速检查，看看我们是否已经到达视频文件的末尾，是否需要停止循环。

预处理发生在**线 58 和 59** 上。首先，`frame`被调整到`600`像素宽，保持纵横比。然后，`frame`转换为`rgb`颜色通道排序，以兼容 dlib(OpenCV 默认为 BGR，dlib 默认为 RGB)。

从那里我们在第 63-66 行的**上实例化视频`writer`(如果需要的话)。要了解更多关于用 OpenCV 将视频写入磁盘的信息，请查看我之前的博文。**

让我们开始**物体*探测*阶段:**

```py
	# if there are no object trackers we first need to detect objects
	# and then create a tracker for each object
	if len(trackers) == 0:
		# grab the frame dimensions and convert the frame to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

		# pass the blob through the network and obtain the detections
		# and predictions
		net.setInput(blob)
		detections = net.forward()

```

为了执行对象跟踪，我们必须首先执行对象检测，或者:

1.  **手动**，通过停止视频流并手动选择每个对象的边界框。
2.  **程序化**，使用经过训练的物体检测器检测物体的存在(这就是我们在这里做的)。

如果没有物体追踪器( **Line 70** )，那么我们知道我们还没有执行物体检测。

我们创建一个`blob`并通过 SSD 网络来检测**线 72-78** 上的对象。要了解`cv2.blobFromImage`功能，请务必[参考我在本文中撰写的](https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/)。

接下来，我们继续循环检测以找到属于`"person"`类的对象，因为我们的输入视频是一场人类竞走:

```py
		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])
				label = CLASSES[idx]

				# if the class label is not a person, ignore it
				if CLASSES[idx] != "person":
					continue

```

我们开始在**行 81** 上循环检测，在此我们:

1.  过滤掉弱检测(**行 88** )。
2.  确保每个检测都是一个`"person"` ( **行 91-96** )。当然，您可以删除这行代码，或者根据自己的过滤需要对其进行定制。

现在我们已经在帧中定位了每个`"person"`，让我们实例化我们的跟踪器并绘制我们的初始边界框+类标签:

```py
				# compute the (x, y)-coordinates of the bounding box
				# for the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# construct a dlib rectangle object from the bounding
				# box coordinates and start the correlation tracker
				t = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				t.start_track(rgb, rect)

				# update our set of trackers and corresponding class
				# labels
				labels.append(label)
				trackers.append(t)

				# grab the corresponding class label for the detection
				# and draw the bounding box
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(frame, label, (startX, startY - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

```

为了开始跟踪对象，我们:

*   计算每个*检测到的*对象的边界`box`(**行 100 和 101** )。
*   实例化边界框坐标并将其传递给*跟踪器* ( **第 105-107 行**)。边界框在这里尤其重要。我们需要为边界框创建一个`dlib.rectangle`并将其传递给`start_track`方法。从那里，dlib 可以开始跟踪该对象。
*   最后，我们用个人追踪器(**行 112** )填充`trackers`列表。

因此，在下一个代码块中，我们将处理已经建立跟踪器的情况，我们只需要更新位置。

在初始检测步骤中，我们还需要执行另外两项任务:

*   将类别标签附加到`labels`列表中(**第 111 行**)。如果你在追踪多种类型的物体(比如`"dog"` + `"person"`，你可能想知道每个物体的类型是什么。
*   在对象周围绘制每个边界框`rectangle`并在对象上方分类`label`(**第 116-119 行**)。

如果我们的检测列表的长度大于零，我们知道我们处于**对象*跟踪*阶段:**

```py
	# otherwise, we've already performed detection so let's track
	# multiple objects
	else:
		# loop over each of the trackers
		for (t, l) in zip(trackers, labels):
			# update the tracker and grab the position of the tracked
			# object
			t.update(rgb)
			pos = t.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# draw the bounding box from the correlation object tracker
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
			cv2.putText(frame, l, (startX, startY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

```

在物体*跟踪*阶段，我们循环所有`trackers`和**行 125** 上对应的`labels`。

然后我们进行到`update`每个物体位置(**第 128-129 行**)。为了更新位置，我们简单地传递`rgb`图像。

提取包围盒坐标后，我们可以为每个被跟踪的对象绘制一个包围盒`rectangle`和`label`(**第 138-141 行**)。

帧处理循环中的其余步骤包括写入输出视频(如有必要)并显示结果:

```py
	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

```

在这里我们:

*   如有必要，将`frame`写入视频(**第 144 和 145 行**)。
*   显示输出帧并捕捉按键(**第 148 和 149 行**)。如果按下`"q"`键(“退出”)，我们`break`退出循环。
*   最后，我们更新每秒帧数信息，用于基准测试(**第 156 行**)。

剩下的步骤是在终端中打印 FPS 吞吐量信息并释放指针:

```py
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()

```

为了结束，我们的`fps`统计数据被收集和打印(**第 159-161 行**)，视频`writer`被发布(**第 164 和 165 行**)，我们关闭所有窗口+发布视频流。

让我们来评估准确性和性能。

要跟随并运行这个脚本，请确保使用这篇博文的 ***“下载”*** 部分下载源代码+示例视频。

从那里，打开一个终端并执行以下命令:

```py
$ python multi_object_tracking_slow.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
	--video race.mp4 --output race_output_slow.avi
[INFO] loading model...
[INFO] starting video stream...
[INFO] elapsed time: 24.51
[INFO] approx. FPS: 13.87

```

看来我们的多目标追踪器起作用了！

但是如你所见，我们只获得了 **~13 FPS** 。

对于某些应用来说，这种 FPS 吞吐率可能已经足够了——然而，如果你需要更快的 FPS，我建议你看看下面的 T2 更高效的 dlib 多目标跟踪器。

其次，要明白跟踪精度并不完美。参考下面*“改进和建议”*部分的第三个建议，以及[阅读我关于 dlib 对象跟踪的第一篇文章](https://pyimagesearch.com/2018/10/22/object-tracking-with-dlib/)了解更多信息。

### *快速、高效的* dlib 多目标跟踪实现

如果您运行上一节中的 dlib 多对象跟踪脚本，同时打开系统的活动监视器，您会注意到处理器中只有*个*内核被利用。

为了加速我们的对象跟踪管道，我们可以利用 [Python 的多处理模块](https://docs.python.org/3.4/library/multiprocessing.html)，类似于[线程模块](https://docs.python.org/3.4/library/threading.html#module-threading)，但是用于产生*进程*而不是*线程*。

利用进程使我们的操作系统能够执行更好的进程调度，将进程映射到我们机器上的特定处理器核心(大多数现代操作系统能够以并行方式高效地调度使用大量 CPU 的进程)。

如果你是 Python 多重处理模块的新手，我建议你阅读 Sebastian Raschka 的这篇精彩介绍。

否则，打开`mutli_object_tracking_fast.py`并插入以下代码:

```py
# import the necessary packages
from imutils.video import FPS
import multiprocessing
import numpy as np
import argparse
import imutils
import dlib
import cv2

```

我们的包装是在**2-8 线**进口的。我们正在导入**线 3** 上的`multiprocessing`库。

我们将使用 Python `Process`类来生成一个新的进程——每个新进程都独立于原始进程*。*

 *为了生成这个流程，我们需要提供一个 Python 可以调用的函数，然后 Python 将获取并创建一个全新的流程+执行它:

```py
def start_tracker(box, label, rgb, inputQueue, outputQueue):
	# construct a dlib rectangle object from the bounding box
	# coordinates and then start the correlation tracker
	t = dlib.correlation_tracker()
	rect = dlib.rectangle(box[0], box[1], box[2], box[3])
	t.start_track(rgb, rect)

```

`start_tracker`的前三个参数包括:

*   我们将要跟踪的物体的边界框坐标，可能是由某种物体探测器返回的，不管是手动的还是编程的。
*   `label`:物体的人类可读标签。
*   一张 RGB 排序的图像，我们将用它来启动初始的 dlib 对象跟踪器。

请记住 Python 多重处理的工作方式——Python 将调用这个函数，然后创建一个全新的解释器来执行其中的代码。因此，每个`start_tracker`衍生的进程将独立于其父进程。为了与 Python 驱动脚本通信，我们需要利用[管道或队列](https://docs.python.org/3.4/library/multiprocessing.html#pipes-and-queues)。这两种类型的对象都是线程/进程安全的，使用锁和信号量来实现。

本质上，我们正在创建一个简单的生产者/消费者关系:

1.  我们的父进程将**产生**新帧，并将它们添加到特定对象跟踪器的队列中。
2.  然后，子进程**将使用**帧，应用对象跟踪，然后返回更新后的边界框坐标。

我决定在这篇文章中使用`Queue`对象；然而，请记住，如果您愿意，您可以使用一个`Pipe`——一定要参考 [Python 多处理文档](https://docs.python.org/3.4/library/multiprocessing.html),以获得关于这些对象的更多细节。

现在让我们开始一个无限循环，它将在这个过程中运行:

```py
	# loop indefinitely -- this function will be called as a daemon
	# process so we don't need to worry about joining it
	while True:
		# attempt to grab the next frame from the input queue
		rgb = inputQueue.get()

		# if there was an entry in our queue, process it
		if rgb is not None:
			# update the tracker and grab the position of the tracked
			# object
			t.update(rgb)
			pos = t.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the label + bounding box coordinates to the output
			# queue
			outputQueue.put((label, (startX, startY, endX, endY)))

```

我们在这里无限循环——这个函数将作为守护进程调用，所以我们不需要担心加入它。

首先，我们将尝试从第 21 行的**上的`inputQueue`抓取一个新的帧。**

如果帧不是空的，我们将抓取帧，然后`update`对象跟踪器，允许我们获得更新的边界框坐标(**第 24-34 行**)。

最后，我们将`label`和边界框写到`outputQueue`中，这样父进程就可以在我们脚本的主循环中使用它们(**第 38 行**)。

回到父进程，我们将解析我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

```

这个脚本的命令行参数与我们较慢的非多处理脚本完全相同。如果你需要复习这些论点，只需[点击这里](#clargs)。此外，如果你不熟悉 argparse 和命令行参数，请阅读我的帖子。

让我们初始化输入和输出队列:

```py
# initialize our lists of queues -- both input queue and output queue
# for *every* object that we will be tracking
inputQueues = []
outputQueues = []

```

这些队列将保存我们正在跟踪的对象。每个产生的进程将需要两个`Queue`对象:

1.  一个用来读取输入帧的
2.  另一个用来写入结果

下一个代码块与我们之前的脚本相同:

```py
# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and output video writer
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None

# start the frames per second throughput estimator
fps = FPS().start()

```

我们定义模型的`CLASSES`并加载模型本身(**第 61-68 行**)。请记住，这些`CLASSES`是静态的——我们的 MobileNet SSD 支持这些类，并且只支持这些类。如果你想检测+跟踪其他物体，你需要找到另一个预训练的模型或训练一个。此外，*这个列表的顺序很重要！*除非你喜欢混乱，否则不要改变列表的顺序！如果你想进一步了解物体探测器是如何工作的，我还建议你阅读本教程。

我们初始化我们的视频流对象，并将我们的视频`writer`对象设置为`None` ( **第 72 行和第 73 行**)。

我们的每秒帧数计算器在第 76 行**被实例化并启动。**

现在让我们开始循环视频流中的帧:

```py
# loop over frames from the video file stream
while True:
	# grab the next frame from the video file
	(grabbed, frame) = vs.read()

	# check to see if we have reached the end of the video file
	if frame is None:
		break

	# resize the frame for faster processing and then convert the
	# frame from BGR to RGB ordering (dlib needs RGB ordering)
	frame = imutils.resize(frame, width=600)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

```

上面的代码块再次与前面脚本中的代码块相同。确保根据需要参考上述中的[。](#while)

现在让我们来处理没有`inputQueues`的情况:

```py
	# if our list of queues is empty then we know we have yet to
	# create our first object tracker
	if len(inputQueues) == 0:
		# grab the frame dimensions and convert the frame to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

		# pass the blob through the network and obtain the detections
		# and predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])
				label = CLASSES[idx]

				# if the class label is not a person, ignore it
				if CLASSES[idx] != "person":
					continue

```

如果没有`inputQueues` ( **线 101** )，那么我们知道我们需要在对象*跟踪*之前应用对象*检测*。

我们在**行 103-109** 上应用对象*检测*，然后继续循环**行 112** 上的结果。我们获取我们的`confidence`值，并过滤掉**行 115-119** 上的弱`detections`。

如果我们的`confidence`满足由我们的命令行参数建立的阈值，我们考虑检测，但是我们通过类`label`进一步过滤它。在这种情况下，我们只寻找`"person"`对象(**第 122-127 行**)。

假设我们已经找到了一个`"person"`，我们将创建队列并生成跟踪流程:

```py
				# compute the (x, y)-coordinates of the bounding box
				# for the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				bb = (startX, startY, endX, endY)

				# create two brand new input and output queues,
				# respectively
				iq = multiprocessing.Queue()
				oq = multiprocessing.Queue()
				inputQueues.append(iq)
				outputQueues.append(oq)

				# spawn a daemon process for a new object tracker
				p = multiprocessing.Process(
					target=start_tracker,
					args=(bb, label, rgb, iq, oq))
				p.daemon = True
				p.start()

				# grab the corresponding class label for the detection
				# and draw the bounding box
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(frame, label, (startX, startY - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

```

我们首先计算第 131-133 行上的边界框坐标。

从那里我们创建两个新队列，`iq`和`oq` ( **行 137 和 138** )，分别将它们追加到`inputQueues`和`outputQueues`(**行 139 和 140** )。

从那里我们产生一个新的`start_tracker`进程，传递边界框、`label`、`rgb`图像和`iq` + `oq` ( **第 143-147 行**)。不要忘记[在这里阅读更多关于多重处理的内容](https://sebastianraschka.com/Articles/2014_multiprocessing.html)。

我们还画出被检测物体的包围盒`rectangle`和类`label` ( **第 151-154 行**)。

否则，我们已经执行了对象检测，因此我们需要将每个 dlib 对象跟踪器应用于该帧:

```py
	# otherwise, we've already performed detection so let's track
	# multiple objects
	else:
		# loop over each of our input ques and add the input RGB
		# frame to it, enabling us to update each of the respective
		# object trackers running in separate processes
		for iq in inputQueues:
			iq.put(rgb)

		# loop over each of the output queues
		for oq in outputQueues:
			# grab the updated bounding box coordinates for the
			# object -- the .get method is a blocking operation so
			# this will pause our execution until the respective
			# process finishes the tracking update
			(label, (startX, startY, endX, endY)) = oq.get()

			# draw the bounding box from the correlation object
			# tracker
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)
			cv2.putText(frame, label, (startX, startY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

```

循环遍历每个`inputQueues`，我们给它们添加`rgb`图像(**第 162 和 163 行**)。

然后我们循环遍历每个`outputQueues` ( **行 166** )，从每个独立的对象跟踪器(**行 171** )获得边界框坐标。最后，我们在第 175-178 行的**上画出边界框+关联的类`label`。**

让我们完成循环和脚本:

```py
	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()

```

如果需要的话，我们将帧写入输出视频，并将`frame`显示到屏幕上(**第 181-185 行**)。

如果按下`"q"`键，我们“退出”，退出循环(**第 186-190 行**)。

如果我们继续处理帧，我们的`fps`计算器在**行 193** 上更新，然后我们再次从`while`循环的开始开始处理。

否则，我们处理完帧，显示 FPS 吞吐量信息+释放指针并关闭窗口。

要执行这个脚本，请确保您使用帖子的 ***【下载】*** 部分下载源代码+示例视频。

从那里，打开一个终端并执行以下命令:

```py
$ python multi_object_tracking_fast.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
	--video race.mp4 --output race_output_fast.avi
[INFO] loading model...
[INFO] starting video stream...
[INFO] elapsed time: 14.01
[INFO] approx. FPS: 24.26

```

<https://www.youtube.com/embed/8PvvsPbSkUs?feature=oembed>*