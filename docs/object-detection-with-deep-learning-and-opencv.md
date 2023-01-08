# 基于深度学习和 OpenCV 的物体检测

> 原文：<https://pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/>

最后更新于 2021 年 7 月 7 日。

几周前，我们学习了如何使用深度学习和 OpenCV 3.3 的深度神经网络(`dnn`)模块[对图像进行分类。](https://pyimagesearch.com/2017/08/21/deep-learning-with-opencv/)

虽然这篇原始的博客文章演示了我们如何将*一幅图像归类到 ImageNet 的 1000 个单独的类别标签中的一个，但是*无法*告诉我们 ***一个对象驻留在图像中的什么地方*** 。*

 *为了获得图像中对象的边界框 *(x，y)*-坐标，我们需要应用**对象检测**。

物体检测不仅能告诉我们**在图像中是什么，还能告诉我们**物体在哪里。****

 ****在今天博文的剩余部分，我们将讨论如何使用深度学习和 OpenCV 来应用对象检测。

*   【2021 年 7 月更新:增加了一个关于替代的基于深度学习的对象检测器的部分，包括关于如何从头训练 R-CNN 的文章，以及关于边界框回归的更多细节。

## 基于深度学习和 OpenCV 的物体检测

在今天关于使用深度学习进行物体检测的帖子的第一部分中，我们将讨论*单次检测器*和*移动探测器*。

当将这些方法结合在一起时，可以用于资源受限设备(包括 Raspberry Pi、智能手机等)上的超快速实时对象检测。)

从那里，我们将发现如何使用 OpenCV 的`dnn`模块来加载一个预先训练的对象检测网络。

这将使我们能够通过网络传递输入图像，并获得图像中每个对象的输出边界框 *(x，y)-* 坐标。

最后，我们将看看将 MobileNet 单镜头检测器应用于示例输入图像的结果。

在未来的博客文章中，我们将扩展我们的脚本来处理实时视频流。

### 用于目标检测的单次检测器

说到基于深度学习的对象检测，您可能会遇到三种主要的对象检测方法:

*   [更快的 R-CNNs](https://arxiv.org/abs/1506.01497) (任等，2015)
*   [你只看一次(YOLO)](https://arxiv.org/abs/1506.02640) (雷德蒙等人，2015)
*   【刘等，2015】

更快的 R-CNN 可能是使用深度学习进行对象检测的最“听说过”的方法；然而，这种技术可能很难理解(特别是对于深度学习的初学者)，很难实现，并且很难训练。

此外，即使使用“更快”的实现 R-CNN(其中“R”代表“区域提议”)，该算法也可能相当慢，大约为 7 FPS。

如果我们寻求纯粹的速度，那么我们倾向于使用 YOLO，因为这种算法更快，能够在 Titan X GPU 上处理 40-90 FPS。YOLO 的超快速变体甚至可以达到 155 FPS。

YOLO 的问题在于，它的精确度有待提高。

固态硬盘最初由谷歌开发，是两者之间的平衡。该算法比更快的 R-CNN 更简单(我认为在最初的开创性论文中解释得更好)。

我们还可以享受比任等人更快的 FPS 吞吐量。每秒 22-46 帧，这取决于我们使用的网络版本。固态硬盘也往往比 YOLO 更准确。要了解更多关于固态硬盘的知识，请参见[刘等](https://arxiv.org/abs/1512.02325)。

### MobileNets:高效(深度)神经网络

当构建对象检测网络时，我们通常使用现有的网络架构，如 VGG 或 ResNet，然后在对象检测管道内使用它。问题是这些网络架构可能非常大，大约 200-500MB。

诸如此类的网络架构由于其庞大的规模和由此产生的计算量而不适用于资源受限的设备。

相反，我们可以使用谷歌研究人员的另一篇论文 [MobileNets](https://arxiv.org/abs/1704.04861) (Howard et al .，2017)。我们称这些网络为“移动网络”,因为它们是为诸如智能手机等资源受限的设备而设计的。MobileNets 通过使用*深度方向可分离卷积*(上图**图 2** )与传统 CNN 不同。

深度方向可分离卷积背后的一般思想是将卷积分成两个阶段:

1.  一个 *3×3* 深度方向卷积。
2.  随后是 *1×1* 逐点卷积。

这允许我们实际上减少网络中的参数数量。

问题是我们牺牲了准确性——移动互联网通常不如它们的大兄弟们准确……

但是它们更加节约资源。

有关移动互联网的更多详情，请参见 [Howard 等人的](https://arxiv.org/abs/1704.04861)。

### 结合 MobileNets 和单次检测器，实现快速、高效的基于深度学习的对象检测

如果我们将 MobileNet 架构和单次检测器(SSD)框架相结合，我们将获得一种快速、高效的基于深度学习的对象检测方法。

我们将在这篇博文中使用的模型是 Howard 等人的[原始 TensorFlow 实现](https://github.com/Zehaos/MobileNet)的 Caffe 版本，并由川崎 305 训练([参见 GitHub](https://github.com/chuanqi305/MobileNet-SSD) )。

MobileNet SSD 首先在 [COCO 数据集](http://cocodataset.org/)(上下文中的常见对象)上进行训练，然后在 PASCAL VOC 上进行微调，达到 72.7% mAP(平均精度)。

因此，我们可以检测图像中的 20 个对象(背景类+1)，包括*飞机、自行车、鸟、船、瓶子、公共汽车、汽车、猫、椅子、牛、餐桌、狗、马、摩托车、人、盆栽植物、羊、沙发、火车、*和*电视监视器*。

### 基于深度学习的 OpenCV 物体检测

在本节中，我们将使用 OpenCV 中的 MobileNet SSD +深度神经网络(`dnn`)模块来构建我们的对象检测器。

我建议使用这篇博文底部的 ***【下载】*** 代码下载源代码+训练好的网络+示例图片，这样你就可以在你的机器上测试它们了。

让我们继续，开始使用 OpenCV 构建我们的深度学习对象检测器。

打开一个新文件，将其命名为`deep_learning_object_detection.py`，并插入以下代码:

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
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

```

在**的第 2-4 行**中，我们导入了这个脚本所需的包——`dnn`模块包含在`cv2`中，同样，假设您使用的是 *OpenCV 3.3* 。

然后，我们解析我们的命令行参数(**第 7-16 行**):

*   `--image`:输入图像的路径。
*   【the Caffe prototxt 文件的路径。
*   `--model`:预训练模型的路径。
*   `--confidence`:过滤弱检测的最小概率阈值。默认值为 20%。

同样，前三个参数的示例文件包含在这篇博文的 ***“下载”*** 部分。我强烈建议您从这里开始，同时也提供一些您自己的查询图片。

接下来，让我们初始化类标签和边界框颜色:

```py
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

```

**第 20-23 行**构建了一个名为`CLASSES`的列表，包含我们的标签。接下来是一个列表，`COLORS`，它包含了边界框相应的随机颜色(**第 24 行**)。

现在我们需要加载我们的模型:

```py
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

```

上面几行是不言自明的，我们只需打印一条消息并加载我们的`model` ( **第 27 行和第 28 行**)。

接下来，我们将加载我们的查询图像并准备我们的`blob`，我们将通过网络对其进行前馈:

```py
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
	(300, 300), 127.5)

```

注意这个块中的注释，我们加载我们的`image` ( **第 34 行**)，提取高度和宽度(**第 35 行**)，并从我们的图像(**第 36 行**)计算一个 *300 乘 300* 像素`blob`。

现在我们准备好做繁重的工作了——我们将让这个 blob 通过神经网络:

```py
# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

```

在**行 41 和 42** 上，我们设置网络的输入，并计算输入的前向传递，将结果存储为`detections`。根据您的型号和输入大小，计算向前传递和相关的检测可能需要一段时间，但对于这个示例，它在大多数 CPU 上相对较快。

让我们循环遍历我们的`detections`并确定*什么*和*物体在图像中的*位置:

```py
# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# display the prediction
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		print("[INFO] {}".format(label))
		cv2.rectangle(image, (startX, startY), (endX, endY),
			COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(image, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

```

我们首先循环我们的检测，记住在单个图像中可以检测多个对象。我们还对与每个检测相关联的置信度(即概率)进行检查。如果置信度足够高(即高于阈值)，那么我们将在终端中显示预测，并在带有文本和彩色边框的图像上绘制预测。让我们一行一行地分解它:

循环遍历我们的`detections`，首先我们提取`confidence`值(**第 48 行**)。

如果`confidence`高于我们的最小阈值(**行 52** )，我们提取类别标签索引(**行 56** )并计算被检测对象周围的边界框(**行 57** )。

然后，我们提取盒子的 *(x，y)*-坐标(**第 58 行**)，稍后我们将使用它来绘制矩形和显示文本。

接下来，我们构建一个文本`label`，包含`CLASS`名称和`confidence` ( **第 61 行**)。

使用标签，我们将它打印到终端(**行 62** )，然后使用我们之前提取的 *(x，y)*-坐标(**行 63 和 64** )在对象周围绘制一个彩色矩形。

一般来说，我们希望标签显示在矩形的上方，但是如果没有空间，我们将它显示在矩形顶部的正下方( **Line 65** )。

最后，我们使用刚刚计算的*y*-值(**第 66 和 67 行**)将彩色文本覆盖到`image`上。

剩下的唯一一步是显示结果:

```py
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)

```

我们将结果输出图像显示到屏幕上，直到按下一个键为止(**行 70 和 71** )。

### OpenCV 和深度学习对象检测结果

要下载代码+预先训练好的网络+示例图片，请务必使用这篇博文底部的 ***“下载”*** 部分。

从那里，解压缩归档文件并执行以下命令:

```py
$ python deep_learning_object_detection.py \
	--prototxt MobileNetSSD_deploy.prototxt.txt \
	--model MobileNetSSD_deploy.caffemodel --image images/example_01.jpg 
[INFO] loading model...
[INFO] computing object detections...
[INFO] loading model...
[INFO] computing object detections...
[INFO] car: 99.78%
[INFO] car: 99.25%

```

我们的第一个结果显示，识别和检测汽车的置信度接近 100%。

在这个示例中，我们使用基于深度学习的对象检测来检测飞机:

```py
$ python deep_learning_object_detection.py \
	--prototxt MobileNetSSD_deploy.prototxt.txt \
	--model MobileNetSSD_deploy.caffemodel --image images/example_02.jpg 
[INFO] loading model...
[INFO] computing object detections...
[INFO] loading model...
[INFO] computing object detections...
[INFO] aeroplane: 98.42%

```

深度学习检测和定位模糊对象的能力在下图中得到展示，我们看到一匹马(和它的骑手)跳过两侧有两株盆栽植物的围栏:

```py
$ python deep_learning_object_detection.py \
	--prototxt MobileNetSSD_deploy.prototxt.txt \
	--model MobileNetSSD_deploy.caffemodel --image images/example_03.jpg
[INFO] loading model...
[INFO] computing object detections...
[INFO] horse: 96.67%
[INFO] person: 92.58%
[INFO] pottedplant: 96.87%
[INFO] pottedplant: 34.42%

```

在本例中，我们可以看到啤酒瓶以令人印象深刻的 100%置信度被检测到:

```py
$ python deep_learning_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt \
	--model MobileNetSSD_deploy.caffemodel --image images/example_04.jpg 
[INFO] loading model...
[INFO] computing object detections...
[INFO] bottle: 100.00%

```

随后是另一个马图像，它也包含狗、汽车和人:

```py
$ python deep_learning_object_detection.py \
	--prototxt MobileNetSSD_deploy.prototxt.txt \
	--model MobileNetSSD_deploy.caffemodel --image images/example_05.jpg 
[INFO] loading model...
[INFO] computing object detections...
[INFO] car: 99.87%
[INFO] dog: 94.88%
[INFO] horse: 99.97%
[INFO] person: 99.88%

```

最后，一张我和杰玛的照片，一只家庭猎犬:

```py
$ python deep_learning_object_detection.py \
	--prototxt MobileNetSSD_deploy.prototxt.txt \
	--model MobileNetSSD_deploy.caffemodel --image images/example_06.jpg 
[INFO] loading model...
[INFO] computing object detections...
[INFO] dog: 95.88%
[INFO] person: 99.95%

```

不幸的是，此图像中的电视监视器无法识别，这可能是由于(1)我挡住了它，以及(2)电视周围的对比度差。也就是说，我们已经使用 OpenCV 的`dnn`模块展示了出色的对象检测结果。

### **替代深度学习对象检测器**

在这篇文章中，我们使用 OpenCV 和单镜头检测器(SSD)模型进行基于深度学习的对象检测。

然而，我们可以应用*个*深度学习对象检测器，包括:

*   [*YOLO 用 OpenCV 检测物体*](https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
*   [*【YOLO】和微小——YOLO 上的物体探测树莓派和 m ovidius NCS*](https://pyimagesearch.com/2020/01/27/yolo-and-tiny-yolo-object-detection-on-the-raspberry-pi-and-movidius-ncs/)
*   [*更快的 R-CNN 和 OpenCV*](https://learnopencv.com/faster-r-cnn-object-detection-with-pytorch/)
*   [*屏蔽 R-CNN 和 OpenCV*](https://pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/) (技术上是一种“实例分割”模型)
*   [*RetinaNet 天体探测器*](https://github.com/fizyr/keras-retinanet)

此外，如果您有兴趣了解如何训练自己的定制深度学习对象检测器，包括更深入地了解 R-CNN 系列对象检测器，请务必阅读这个由四部分组成的系列:

1.  [*用 Keras、TensorFlow、OpenCV*](https://pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/) 把任何 CNN 图像分类器变成物体检测器
2.  [*OpenCV 选择性搜索目标检测*](https://pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/)
3.  [*用 OpenCV、Keras 和 TensorFlow 进行区域提议对象检测*](https://pyimagesearch.com/2020/07/06/region-proposal-object-detection-with-opencv-keras-and-tensorflow/)
4.  [*R-CNN 物体检测用 Keras、TensorFlow、深度学习*](https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/)

从那里，我建议更详细地研究一下**边界框回归**的概念:

1.  [*物体检测:使用 Keras、TensorFlow 和深度学习的包围盒回归*](https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/)
2.  [*使用 Keras、TensorFlow 和深度学习的多类对象检测和包围盒回归*](https://pyimagesearch.com/2020/10/12/multi-class-object-detection-and-bounding-box-regression-with-keras-tensorflow-and-deep-learning/)

## 摘要

在今天的博文中，我们学习了如何使用深度学习和 OpenCV 执行对象检测。

具体来说，我们使用了 *MobileNets +单次检测器*和 OpenCV 3.3 的全新(全面改造)`dnn`模块来检测图像中的对象。

作为一个计算机视觉和深度学习社区，我们非常感谢 Aleksandr Rybnikov 的贡献，他是`dnn`模块的主要贡献者，使深度学习可以从 OpenCV 库中访问。你可以在这里找到亚历山大最初的 OpenCV 示例脚本[——为了这篇博文的目的，我修改了它。](https://github.com/opencv/opencv/blob/master/samples/dnn/mobilenet_ssd_python.py)

在未来的博客文章中，我将展示我们如何修改今天的教程，以处理实时视频流，从而使我们能够对视频执行基于深度学习的对象检测。我们一定会利用高效的帧 I/O 来提高我们整个流水线的 FPS。

**在 PyImageSearch 上发布未来的博客文章(如实时对象检测教程)时，我们会通知您，*只需在下面的表格中输入您的电子邮件地址*。*******