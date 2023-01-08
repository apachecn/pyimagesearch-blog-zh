# OpenCV 社交距离检测器

> 原文：<https://pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/>

在本教程中，您将学习如何使用 OpenCV、深度学习和计算机视觉来实现新冠肺炎社交距离探测器。

今天的教程是受 PyImageSearch 读者 Min-Jun 的启发，他发邮件问:

> *嗨，阿德里安，*
> 
> 我已经看到计算机视觉社区的一些人实现了“社交距离探测器”，但我不确定他们是如何工作的。
> 
> 你会考虑就这个话题写一篇教程吗？
> 
> 谢谢你。

Min-Jun 是正确的——我在社交媒体上看到过许多社交距离检测器的实现，我最喜欢的是来自 reddit 用户 danlapko 和 T2 Rohit Kumar Srivastava 的实现。

今天，我将为你提供一个你自己的社交距离探测器的起点。然后，您可以在认为合适的时候扩展它，以开发您自己的项目。

**要了解如何用 OpenCV 实现社交距离检测器，*请继续阅读。***

## OpenCV 社交距离检测器

在本教程的第一部分，我们将简要讨论什么是社交距离，以及如何使用 OpenCV 和深度学习来实现社交距离检测器。

然后，我们将回顾我们的项目目录结构，包括:

1.  我们的配置文件用来保持我们的实现整洁
2.  我们的`detect_people`实用功能，使用 [YOLO 物体检测器](https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)检测视频流中的人
3.  我们的 Python 驱动程序脚本，它将所有部分粘合在一起，成为一个成熟的 OpenCV 社交距离检测器

我们将通过回顾结果来结束这篇文章，包括对局限性和未来改进的简短讨论。

### 什么是社交距离？

社会距离是一种用来控制传染病传播的方法。

顾名思义，社交距离意味着**人们应该在身体上远离彼此，**减少密切接触，从而减少传染病(如冠状病毒)的传播:

社交距离并不是一个新概念，可以追溯到五世纪([来源](https://en.wikipedia.org/wiki/Social_distancing))，甚至在《圣经》等宗教文献中也有提及:

> 那患了灾病的麻疯病人……他要独自居住；他的住处必在营外。—《利未记》13:46

社交距离可以说是防止疾病传播的最有效的非药物方法——根据定义，如果人们不在一起，他们就不能传播细菌。

### 将 OpenCV、计算机视觉和深度学习用于社交距离

我们可以使用 OpenCV、计算机视觉和深度学习来实现社交距离检测器。

构建社交距离检测器的步骤包括:

1.  应用**对象检测**来检测视频流中的所有人(仅*和*人)(参见本教程关于构建 [OpenCV 人物计数器](https://pyimagesearch.com/2018/08/13/opencv-people-counter/))
2.  **计算所有检测到的人之间的成对距离**
3.  基于这些距离，**检查任何两个人之间的距离是否小于 *N* 像素**

为了获得最精确的结果，你应该*通过内部/外部参数校准你的相机*，这样你就可以将*像素*映射到*可测单位。*

一种更简单(但不太准确)的替代方法是应用三角形相似性校准(如本教程中的[所述)。](https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/)

这两种方法都可以用来将像素映射到可测量的单位。

最后，如果你不想/不能应用相机校准，你仍然可以*利用社交距离探测器，但是你必须严格依赖像素距离，这不一定是准确的。*

为了简单起见，我们的 OpenCV 社交距离检测器实现将依赖于像素距离——我将把它留给读者作为一个练习，以便您在认为合适的时候扩展实现。

### 项目结构

一定要从这篇博文的 ***【下载】*** 部分抓取代码。从那里，提取文件，并使用`tree`命令查看我们的项目是如何组织的:

```py
$ tree --dirsfirst
.
├── pyimagesearch
│   ├── __init__.py
│   ├── detection.py
│   └── social_distancing_config.py
├── yolo-coco
│   ├── coco.names
│   ├── yolov3.cfg
│   └── yolov3.weights
├── output.avi
├── pedestrians.mp4
└── social_distance_detector.py

2 directories, 9 files
```

让我们在下一节深入研究 Python 配置文件。

### 我们的配置文件

为了帮助保持代码整洁有序，我们将使用一个配置文件来存储重要的变量。

现在让我们来看看它们——打开`pyimagesearch`模块中的`social_distancing_config.py`文件，看一看:

```py
# base path to YOLO directory
MODEL_PATH = "yolo-coco"

# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3
```

这里，我们有到 YOLO 对象检测模型的路径(**线 2** )。我们还定义了最小目标检测置信度和[非极大值抑制](https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/)阈值。

我们还要定义两个配置常数:

```py
# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = False

# define the minimum safe distance (in pixels) that two people can be
# from each other
MIN_DISTANCE = 50
```

第 10 行**上的`USE_GPU`布尔值表示您的支持 NVIDIA CUDA 的 GPU 是否将用于加速推理(要求 [OpenCV 的“dnn”模块安装有 NVIDIA GPU 支持](https://pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/))。**

**Line 14** 定义了人们为了遵守社交距离协议而必须保持的最小距离(以像素为单位)。

### 用 OpenCV 检测图像和视频流中的人物

```py
# import the necessary packages
from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
import cv2
```

```py
def detect_people(frame, net, ln, personIdx=0):
	# grab the dimensions of the frame and  initialize the list of
	# results
	(H, W) = frame.shape[:2]
	results = []
```

```py
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, centroids, and
	# confidences, respectively
	boxes = []
	centroids = []
	confidences = []
```

```py
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter detections by (1) ensuring that the object
			# detected was a person and (2) that the minimum
			# confidence is met
			if classID == personIdx and confidence > MIN_CONF:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# centroids, and confidences
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
```

```py
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	# return the list of results
	return results
```

### 利用 OpenCV 和深度学习实现社交距离检测器

```py
# import the necessary packages
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
```

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())
```

```py
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
```

在这里，我们加载我们的 load COCO 标签(**行 22 和 23** )以及定义我们的 YOLO 路径(**行 26 和 27** )。

使用 YOLO 路径，现在我们可以将模型加载到内存中:

```py
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

```py
# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
```

```py
# loop over the frames from the video stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# resize the frame and then detect people (and only people) in it
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln,
		personIdx=LABELS.index("person"))

	# initialize the set of indexes that violate the minimum social
	# distance
	violate = set()
```

```py
	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
				if D[i, j] < config.MIN_DISTANCE:
					# update our violation set with the indexes of
					# the centroid pairs
					violate.add(i)
					violate.add(j)
```

如果我们不能可视化结果，我们的应用程序会有什么乐趣呢？

我说，一点也不好玩！因此，让我们用矩形、圆形和文本来注释我们的框架:

```py
	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# if the index pair exists within the violation set, then
		# update the color
		if i in violate:
			color = (0, 0, 255)

		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	# draw the total number of social distancing violations on the
	# output frame
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
```

```py
	# check to see if the output frame should be displayed to our
	# screen
	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)
```

### OpenCV 社交距离探测器结果

我们现在准备测试我们的 OpenCV 社交距离探测器。

确保使用本教程的 ***【下载】*** 部分下载源代码和示例演示视频。

从那里，打开一个终端，并执行以下命令:

```py
$ time python social_distance_detector.py --input pedestrians.mp4  \
	--output output.avi --display 0
[INFO] loading YOLO from disk...
[INFO] accessing video stream...

real    3m43.120s
user    23m20.616s
sys     0m25.824s
```

在这里，你可以看到我能够在我的 CPU 上用 3m43s 处理整个视频，正如结果所示，**我们的社交距离检测器正在正确地标记违反社交距离规则的人。**

当前实现的问题是速度。我们基于 CPU 的社交距离探测器正在获得 **~2.3 FPS** ，这对于实时处理来说*太慢了*。

您可以通过以下方式获得更高的帧处理速率:( 1)利用支持 NVIDIA CUDA 的 GPU ;( 2)[利用 NVIDIA GPU 支持编译/安装 OpenCV 的“dnn”模块](https://pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/)。

如果你已经安装了 OpenCV 并支持 NVIDIA GPU，你需要做的就是在你的`social_distancing_config.py`文件中设置`USE_GPU = True`:

```py
# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = True
```

```py
$ time python social_distance_detector.py --input pedestrians.mp4 \
	--output output.avi --display 0
[INFO] loading YOLO from disk...
[INFO] setting preferable backend and target to CUDA...
[INFO] accessing video stream...

real    0m56.008s
user    1m15.772s
sys     0m7.036s
```

在这里，我们仅用 56 秒处理了整个视频，总计 **~9.38 FPS** ，这是一个 **307%的加速！**

### 局限性和未来的改进

正如本教程前面已经提到的，我们的社交距离检测器*没有*利用适当的相机校准，这意味着我们不能(容易地)将像素距离映射到实际的可测量单位(即米、英尺等)。).

**因此，改进社交距离探测器的第一步是利用适当的摄像机校准。**

这样做将产生更好的结果，并使您能够计算实际的可测量单位(而不是像素)。

**其次，你应该考虑应用自上而下的视角变换，**正如这个实现所做的:

从那里，您可以将距离计算应用于行人的俯视图，从而获得更好的距离近似值。

我的第三个建议是改进人员检测流程。

OpenCV 的 YOLO 实现非常慢*不是因为模型本身而是因为模型需要额外的后处理。*

为了进一步加速流水线，可以考虑利用运行在 GPU 上的单次检测器(SSD)——这将大大提高帧吞吐率*。*

最后，我想提一下，你会在网上看到许多社交距离探测器的实现——我今天在这里介绍的一个应该被视为你可以建立的*模板*和*起点*。

如果您想了解更多关于使用计算机视觉实现社交距离检测器的信息，请查看以下资源:

*   [自动社会距离测量](https://www.reddit.com/r/computervision/comments/gf4zhj/automatic_social_distance_measurement/)
*   [工作场所的社交距离](https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/)
*   [Rohit Kumar Srivastava 的社交距离实现](https://www.linkedin.com/feed/update/urn%3Ali%3Aactivity%3A6661455400346492928/)
*   [Venkatagiri Ramesh 的社交距离项目](https://www.linkedin.com/feed/update/urn%3Ali%3Aactivity%3A6655464103798157312/)
*   Mohan Morkel 的社交距离应用(我*认为*可能是基于 Venkatagiri Ramesh 的)

如果你已经实现了你自己的 OpenCV 社交距离项目，而我还没有链接到它，请接受我的道歉——现在有太多的实现让我无法跟踪。

## 摘要

在本教程中，您学习了如何使用 OpenCV、计算机视觉和深度学习来实现社交距离探测器。

我们的实施通过以下方式实现:

1.  使用 YOLO 对象检测器检测视频流中的人
2.  确定每个检测到的人的质心
3.  计算所有质心之间的成对距离
4.  检查是否有任何成对的距离相隔 *< N* 个像素，如果是，则表明这对人违反了社交距离规则

此外，通过使用支持 NVIDIA CUDA 的 GPU，以及在 NVIDIA GPU 支持下编译的 OpenCV 的`dnn`模块，我们的方法能够实时运行，使其可用作概念验证社交距离检测器。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***