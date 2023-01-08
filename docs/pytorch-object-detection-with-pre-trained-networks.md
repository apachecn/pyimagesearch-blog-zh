# 使用预训练网络的 PyTorch 对象检测

> 原文：<https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/>

在本教程中，您将学习如何使用 PyTorch 通过预训练的网络执行对象检测。利用预先训练的对象检测网络，您可以检测和识别您的计算机视觉应用程序将在日常生活中“看到”的 90 个常见对象。

今天的教程是 PyTorch 基础知识五部分系列的最后一部分:

1.  [*py torch 是什么？*](https://pyimagesearch.com/2021/07/05/what-is-pytorch/)
2.  [*py torch 简介:使用 PyTorch*](https://pyimagesearch.com/2021/07/12/intro-to-pytorch-training-your-first-neural-network-using-pytorch/) 训练你的第一个神经网络
3.  [*PyTorch:训练你的第一个卷积神经网络*](https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/)
4.  *[【py torch】用预先训练好的网络进行图像分类](https://pyimagesearch.com/2021/07/26/pytorch-image-classification-with-pre-trained-networks/)*
5.  *PyTorch 物体检测，带有预训练网络*(今天的教程)

在本教程的其余部分，您将获得使用 PyTorch 检测输入图像中的对象的经验，使用开创性的、最先进的图像分类网络，包括使用 ResNet 的更快 R-CNN、使用 MobileNet 的更快 R-CNN 和 RetinaNet。

**要学习如何用预先训练好的 PyTorch 网络进行物体检测，** ***继续阅读。***

## **使用预训练网络的 PyTorch 对象检测**

在本教程的第一部分，我们将讨论什么是预训练的对象检测网络，包括 PyTorch 库中内置了什么对象检测网络。

从那里，我们将配置我们的开发环境，并审查我们的项目目录结构。

今天我们将回顾两个 Python 脚本。第一个将在图像中执行**对象检测，而第二个将向您展示如何在视频流**中执行**实时对象检测(将需要 GPU 来获得实时性能)。**

最后，我们将讨论我们的结果来结束本教程。

### **什么是预训练的物体检测网络？**

正如 [ImageNet challenge](https://image-net.org/) 倾向于成为图像分类的*事实上的*标准一样， [COCO 数据集](https://cocodataset.org/#home)(上下文中的常见对象)倾向于成为对象检测基准的标准。

这个数据集包括你在日常生活中会看到的 90 多种常见物品。计算机视觉和深度学习研究人员在 COCO 数据集上开发、训练和评估最先进的对象检测网络。

大多数研究人员还将预训练的权重发布到他们的模型中，以便计算机视觉从业者可以轻松地将对象检测纳入他们自己的项目中。

本教程将展示如何使用 PyTorch 通过以下先进的分类网络来执行对象检测:

1.  具有 ResNet50 主干的更快的 R-CNN(更准确，但更慢)
2.  使用 MobileNet v3 主干的更快的 R-CNN(更快，但不太准确)
3.  具有 ResNet50 主干的 RetinaNet(速度和准确性之间的良好平衡)

准备好了吗？让我们开始吧。

### **配置您的开发环境**

要遵循这个指南，您需要在系统上安装 PyTorch 和 OpenCV。

幸运的是，PyTorch 和 OpenCV 都非常容易使用 pip 安装:

```py
$ pip install torch torchvision
$ pip install opencv-contrib-python
```

**如果您需要帮助配置 PyTorch 的开发环境，我*强烈推荐*您** [**阅读 PyTorch 文档**](https://pytorch.org/get-started/locally/)**——py torch 的文档非常全面，可以让您快速上手并运行。**

 **如果你需要帮助安装 OpenCV，[一定要参考我的 *pip 安装 OpenCV* 教程](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)。

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

在我们开始审查任何源代码之前，让我们首先审查我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

然后，您将看到以下目录结构:

```py
$ tree . --dirsfirst 
.
├── images
│   ├── example_01.jpg
│   ├── example_02.jpg
│   ├── example_03.jpg
│   ├── example_04.jpg
│   ├── example_05.jpg
│   └── example_06.jpg
├── coco_classes.pickle
├── detect_image.py
└── detect_realtime.py

1 directory, 9 files
```

在`images`目录中，您会发现许多我们将应用对象检测的示例图像。

`coco_classes.pickle`文件包含我们的 PyTorch 预训练对象检测网络被训练的类别标签的名称。

然后，我们要回顾两个 Python 脚本:

1.  `detect_image.py`:在静态图像中用 PyTorch 进行物体检测
2.  `detect_realtime.py`:将 PyTorch 对象检测应用于实时视频流

### **实现我们的 PyTorch 对象检测脚本**

在本节中，您将学习如何使用预先训练的 PyTorch 网络执行对象检测。

打开`detect_image.py`脚本并插入以下代码:

```py
# import the necessary packages
from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2
```

**第 2-7 行**导入我们需要的 Python 包。最重要的进口是来自`torchvision.models`的`detection`。`detection`模块包含 PyTorch 预先训练的物体探测器。

让我们继续解析我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
	choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
	help="name of the object detection model")
ap.add_argument("-l", "--labels", type=str, default="coco_classes.pickle",
	help="path to file containing list of categories in COCO dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
```

这里有许多命令行参数，包括:

*   `--image`:我们想要应用对象检测的输入图像的路径
*   我们将使用的 PyTorch 对象检测器的类型(更快的 R-CNN + ResNet，更快的 R-CNN + MobileNet，或 RetinaNet + ResNet)
*   `--labels`:COCO 标签文件的路径，包含人类可读的类标签
*   `--confidence`:过滤弱检测的最小预测概率

这里，我们有一些重要的初始化:

```py
# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
CLASSES = pickle.loads(open(args["labels"], "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
```

**第 23 行**设置我们将用于推理的`device`(CPU 或 GPU)。

然后，我们从磁盘加载我们的类标签(**行 27** )，并为每个唯一标签初始化一个随机颜色(**行 28** )。我们将在输出图像上绘制预测边界框和标签时使用这些颜色。

接下来，我们定义一个`MODELS`字典来将给定对象检测器的名称映射到其对应的 PyTorch 函数:

```py
# initialize a dictionary containing model name and its corresponding 
# torchvision function call
MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": detection.retinanet_resnet50_fpn
}

# load the model and set it to evaluation mode
model = MODELS[args["model"]](pretrained=True, progress=True,
	num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()
```

PyTorch 为我们提供了三种对象检测模型:

1.  具有 ResNet50 主干的更快的 R-CNN(更准确，但更慢)
2.  使用 MobileNet v3 主干的更快的 R-CNN(更快，但不太准确)
3.  具有 ResNet50 主干的 RetinaNet(速度和准确性之间的良好平衡)

然后，我们从磁盘加载`model`并将它发送到**行 39 和 40** 上的适当的`DEVICE`。我们传入许多关键参数，包括:

*   `pretrained`:告诉 PyTorch 在 COCO 数据集上加载带有预训练权重的模型架构
*   `progress=True`:如果模型尚未下载和缓存，则显示下载进度条
*   `num_classes`:唯一类的总数
*   `pretrained_backbone`:还向目标探测器提供主干网络

然后我们在第 41 行**将模型置于评估模式。**

加载完模型后，让我们继续为对象检测准备输入图像:

```py
# load the image from disk
image = cv2.imread(args["image"])
orig = image.copy()

# convert the image from BGR to RGB channel ordering and change the
# image from channels last to channels first ordering
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))

# add the batch dimension, scale the raw pixel intensities to the
# range [0, 1], and convert the image to a floating point tensor
image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)

# send the input to the device and pass the it through the network to
# get the detections and predictions
image = image.to(DEVICE)
detections = model(image)[0]
```

**第 44 行和第 45 行**从磁盘加载我们的输入`image`并克隆它，这样我们可以在稍后的脚本中在其上绘制边界框预测。

然后，我们通过以下方式对图像进行预处理:

1.  将颜色通道排序从 BGR 转换为 RGB(因为 PyTorch 模型是在 RGB 排序的图像上训练的)
2.  将颜色通道排序从“通道最后”(OpenCV 和 Keras/TensorFlow 默认)交换到“通道优先”(PyTorch 默认)
3.  添加批次维度
4.  从范围*【0，255】*到*【0，1】*缩放像素强度
5.  将图像从 NumPy 数组转换为浮点数据类型的张量

然后图像被移动到合适的设备上(**行 60** )。在这一点上，我们通过`model`传递`image`来获得我们的边界框预测。

现在让我们循环我们的边界框预测:

```py
# loop over the detections
for i in range(0, len(detections["boxes"])):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections["scores"][i]

	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# extract the index of the class label from the detections,
		# then compute the (x, y)-coordinates of the bounding box
		# for the object
		idx = int(detections["labels"][i])
		box = detections["boxes"][i].detach().cpu().numpy()
		(startX, startY, endX, endY) = box.astype("int")

		# display the prediction to our terminal
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		print("[INFO] {}".format(label))

		# draw the bounding box and label on the image
		cv2.rectangle(orig, (startX, startY), (endX, endY),
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(orig, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# show the output image
cv2.imshow("Output", orig)
cv2.waitKey(0)
```

**线路 64** 在来自网络的所有检测上循环。然后，我们获取与第 67 条线**上的检测相关联的`confidence`(即概率)。**

我们过滤掉不满足我们在第 71 行**上的最小置信度测试的弱检测。这样做有助于过滤掉误报检测。**

从那里，我们:

*   提取对应概率最大的类别标签的`idx`(**第 75 行**
*   获取边界框坐标并将其转换成整数(**第 76 行和第 77 行**)
*   将预测显示到我们的终端(**行 80 和 81** )
*   在我们的输出图像上画出预测的边界框和类标签(**行 84-88** )

我们通过显示上面画有边界框的输出图像来结束脚本。

### 【PyTorch 结果的物体检测

我们现在可以看到一些 PyTorch 物体检测结果了！

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

接下来，让我们应用对象检测:

```py
$ python detect_image.py --model frcnn-resnet  \
	--image images/example_01.jpg --labels coco_classes.pickle    
[INFO] car: 99.54%
[INFO] car: 99.18%
[INFO] person: 85.76%
```

我们在这里使用的对象检测器是一个具有 ResNet50 主干的更快的 R-CNN。由于网络的设计方式，更快的 R-CNN 往往非常擅长检测图像中的小物体——这一点可以从以下事实中得到证明:不仅在输入图像中检测到了每辆*汽车*，而且还检测到了其中一名*司机*(人眼几乎看不到他)。

这是另一个使用我们更快的 R-CNN 物体探测器的示例图像:

```py
$ python detect_image.py --model frcnn-resnet \
	--image images/example_06.jpg --labels coco_classes.pickle
[INFO] dog: 99.92%
[INFO] person: 99.90%
[INFO] chair: 99.42%
[INFO] tv: 98.22%
```

在这里，我们可以看到我们的输出对象检测是相当准确的。我们的模型在场景的前景中准确地检测到了我和 Jemma，一只小猎犬。它还检测背景中的电视和椅子。

让我们尝试最后一张图片，这是一个更复杂的场景，它真实地展示了更快的 R-CNN 模型在检测小物体方面有多好:

```py
$ python detect_image.py --model frcnn-resnet \
	--image images/example_05.jpg --labels coco_classes.pickle \
	--confidence 0.7
[INFO] horse: 99.88%
[INFO] person: 99.76%
[INFO] person: 99.09%
[INFO] dog: 93.22%
[INFO] person: 83.80%
[INFO] person: 81.58%
[INFO] truck: 71.33%
```

注意这里我们是如何手动指定我们的`--confidence`命令行参数`0.7`的，这意味着预测概率 *> 70%* 的物体检测将被认为是真阳性检测(如果你记得，`detect_image.py`脚本默认最小置信度为 90%)。

***注:*** *降低我们的默认置信度会让我们检测到更多的物体，但可能会以误报为代价。*

也就是说，正如图 5 的输出所示，我们的模型已经做出了高度准确的预测。我们不仅检测到了前景物体，如狗、马和马背上的人，还检测到了背景物体，包括背景中的卡车和多人。

为了获得更多使用 PyTorch 进行对象检测的经验，我建议您将`frcnn-mobilenet`和`retinanet`替换为`--model`命令行参数，然后比较输出结果。

### **用 PyTorch 实现实时物体检测**

在上一节中，您学习了如何在 PyTorch 中将对象检测应用于单个图像。本节将向您展示如何使用 PyTorch 将对象检测应用于视频流。

正如您将看到的，以前实现的大部分代码都可以重用，只需稍作修改。

打开项目目录结构中的`detect_realtime.py`脚本，让我们开始工作:

```py
# import the necessary packages
from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2
```

**第 2-11 行**导入我们需要的 Python 包。所有这些导入基本上与我们的`detect_image.py`脚本相同，但是有两个显著的增加:

1.  访问我们的网络摄像头
2.  `FPS`:测量我们的对象检测管道的大约每秒帧数吞吐率

接下来是我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
	choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
	help="name of the object detection model")
ap.add_argument("-l", "--labels", type=str, default="coco_classes.pickle",
	help="path to file containing list of categories in COCO dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
```

我们的第一个开关`--model`控制我们想要使用哪个 PyTorch 对象检测器。

`--labels`参数提供了 COCO 类文件的路径。

最后，`--confidence`开关允许我们提供一个最小的预测概率，以帮助过滤掉微弱的假阳性检测。

下一个代码块处理设置我们的推理设备(CPU 或 GPU)，以及加载我们的类标签:

```py
# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
CLASSES = pickle.loads(open(args["labels"], "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
```

**当在视频流中执行对象检测时，我** ***强烈推荐*** **使用 GPU——CPU 对于任何接近实时性能的东西来说都太慢了。**

然后我们定义我们的`MODELS`字典，就像前面的脚本一样:

```py
# initialize a dictionary containing model name and its corresponding 
# torchvision function call
MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": detection.retinanet_resnet50_fpn
}

# load the model and set it to evaluation mode
model = MODELS[args["model"]](pretrained=True, progress=True,
	num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()
```

**第 41-43 行**从磁盘加载 PyTorch 对象检测`model`并将其置于评估模式。

我们现在可以访问我们的网络摄像头了:

```py
# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
```

我们插入一个小的`sleep`语句，让我们的相机传感器预热。

对`FPS`的`start`方法的调用允许我们开始计算大约每秒帧数的吞吐率。

下一步是循环视频流中的帧:

```py
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	orig = frame.copy()

	# convert the frame from BGR to RGB channel ordering and change
	# the frame from channels last to channels first ordering
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = frame.transpose((2, 0, 1))

	# add a batch dimension, scale the raw pixel intensities to the
	# range [0, 1], and convert the frame to a floating point tensor
	frame = np.expand_dims(frame, axis=0)
	frame = frame / 255.0
	frame = torch.FloatTensor(frame)

	# send the input to the device and pass the it through the
	# network to get the detections and predictions
	frame = frame.to(DEVICE)
	detections = model(frame)[0]
```

**第 56-58 行**从视频流中读取一个`frame`，调整它的大小(输入帧越小，推断速度越快)，然后克隆它，以便我们以后可以在它上面绘图。

我们的预处理操作与之前的脚本相同:

*   从 BGR 转换到 RGB 通道排序
*   从“信道最后”切换到“信道优先”排序
*   添加批次维度
*   从范围*【0，255】*到*【0，1】*缩放帧中的像素亮度
*   将帧转换为浮点 PyTorch 张量

预处理后的`frame`然后被移动到适当的设备，之后进行预测(**第 73 行和第 74 行**)。

物体检测模型结果的处理与`predict_image.py`相同:

```py
	# loop over the detections
	for i in range(0, len(detections["boxes"])):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections["scores"][i]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# detections, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections["labels"][i])
			box = detections["boxes"][i].detach().cpu().numpy()
			(startX, startY, endX, endY) = box.astype("int")

			# draw the bounding box and label on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(orig, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(orig, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
```

最后，我们可以在窗口中显示输出帧:

```py
	# show the output frame
	cv2.imshow("Frame", orig)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
```

我们继续监控我们的 FPS，直到我们点击 OpenCV 打开的窗口并按下`q`键退出脚本，之后我们停止我们的 FPS 定时器并显示(1)脚本运行的时间和(2)大约每秒帧数的吞吐量信息。

### **PyTorch 实时物体检测结果**

让我们学习如何使用 PyTorch 对视频流应用对象检测。

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

从那里，您可以执行`detect_realtime.py`脚本:

```py
$ python detect_realtime.py --model frcnn-mobilenet \
	--labels coco_classes.pickle
[INFO] starting video stream...
[INFO] elapsed time: 56.47
[INFO] approx. FPS: 6.98
```

使用带有 MobileNet 背景的更快的 R-CNN 模型(速度最佳),我们实现了大约每秒 7 FPS。我们还没有达到真正的实时速度 *> 20 FPS* ，但是有了更快的 GPU 和更多的优化，我们可以轻松达到。

## **总结**

在本教程中，您学习了如何使用 PyTorch 和预训练的网络执行对象检测。您获得了在三个流行网络中应用对象检测的经验:

1.  使用 ResNet50 主干的更快的 R-CNN
2.  更快的 R-CNN 和 MobileNet 主干网
3.  具有 ResNet50 主干网的 RetinaNet

当涉及到准确性和检测小物体时，更快的 R-CNN 将表现得非常好。然而，这种准确性是有代价的——更快的 R-CNN 模型往往比单次检测器(SSD)和 YOLO 慢得多。

为了帮助加速更快的 R-CNN 架构，我们可以用更轻、更高效(但不太准确)的 MobileNet 主干网取代计算成本高昂的 ResNet 主干网。这样做会提高你的速度。

否则，RetinaNet 是速度和准确性之间的一个很好的折衷。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******