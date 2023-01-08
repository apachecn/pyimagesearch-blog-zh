# 使用 Keras、TensorFlow 和 OpenCV 将任何 CNN 图像分类器转换为对象检测器

> 原文：<https://pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/>

在本教程中，您将学习如何使用 Keras、TensorFlow 和 OpenCV 将*任何*预先训练的深度学习图像分类器转化为对象检测器。

**今天，我们开始一个关于深度学习和物体检测的四部分系列:**

*   **第一部分:** *用 Keras 和 TensorFlow 把任何深度学习图像分类器变成物体检测器*(今日帖)
*   **第二部分:** *OpenCV 对象检测选择性搜索*
*   **第三部分:** *使用 OpenCV、Keras 和 TensorFlow 进行目标检测的区域提议*
*   **第四部分:** *用 Keras 和 TensorFlow 进行 R-CNN 物体检测*

这一系列帖子的目标是更深入地了解基于深度学习的对象检测器是如何工作的，更具体地说:

1.  传统的计算机视觉对象检测算法如何与深度学习相结合
2.  端到端可训练物体检测器背后的动机是什么，以及与之相关的挑战是什么
3.  **最重要的是，开创性的更快的 R-CNN 架构是如何形成的(我们将在本系列中构建 R-CNN 架构的变体)**

今天，我们将从物体检测的基础开始，包括如何采用预先训练的图像分类器，并利用**图像金字塔**、**滑动窗口**和**非极大值抑制**来构建一个基本的物体检测器(想想猪+线性 SVM 启发)。

在接下来的几周里，我们将学习如何从头开始构建一个端到端的可训练网络。

但是今天，让我们从基础开始。

**要学习如何用 Keras 和 TensorFlow 将任何卷积神经网络图像分类器变成对象检测器，*继续阅读。***

## **使用 Keras、TensorFlow 和 OpenCV 将任何 CNN 图像分类器转变为对象检测器**

在本教程的第一部分，我们将讨论图像分类和目标检测任务之间的主要区别。

然后，我将向您展示如何用大约 200 行代码将任何为图像分类而训练的卷积神经网络变成一个对象检测器。

在此基础上，我们将使用 Keras、TensorFlow 和 OpenCV 实现必要的代码，将图像分类器转换为对象检测器。

最后，我们将回顾我们的工作结果，指出我们的实现中的一些问题和限制，包括我们如何改进这种方法。

### **图像分类与目标检测**

当执行**图像分类时，**给定一个输入图像，我们将其呈现给我们的神经网络，并且我们获得单个类别标签和与类别标签预测相关联的概率(**图 1** ，*左*)。

此类标签旨在表征整个图像的内容，或者至少是图像中最主要的可见内容。

**因此我们可以把图像分类看作:**

*   一个图像在
*   一类标签出来了

**对象检测，**另一方面，不仅通过边界框 *(x，y)*-坐标(**图 1，** *右*)告诉我们*图像中的*是什么(即类别标签)，还告诉我们*图像中的*位置。

**因此，物体检测算法允许我们:**

*   输入一幅图像
*   获得*多个*边界框和类标签作为输出

最核心的是，任何对象检测算法(无论传统的计算机视觉还是最先进的深度学习)都遵循相同的模式:

*   **1。输入:**我们希望对其应用对象检测的图像
*   **2。输出:**三个值，包括:
    *   2a。图像中每个对象的**边界框列表、**或(x，y)坐标
    *   2b。与每个边界框相关联的**类标签**
    *   2c。与每个边界框和类别标签相关联的**概率/置信度得分**

今天，您将看到这种模式的一个实例。

### **怎样才能把任何深度学习图像分类器变成物体检测器？**

在这一点上，你可能想知道:

> 嘿，阿德里安，如果我有一个为图像分类而训练的卷积神经网络，我到底要怎么用它来进行物体检测呢？
> 
> 根据你上面的解释，图像分类和目标检测似乎是根本不同的，需要两种不同类型的网络架构。

从本质上说，*是*正确的——物体检测*不需要专门的网络架构。*

任何读过关于更快的 R-CNN、单发探测器(SSDs)、YOLO、RetinaNet 等的论文的人。知道与传统的图像分类相比，目标检测网络更复杂，涉及面更广，需要多倍的数量级和更多的工作来实施。

也就是说，我们可以利用一种黑客技术将我们的 CNN 图像分类器变成一个物体检测器——秘方在于传统的计算机视觉 T2 算法。

回到基于深度学习的对象检测器之前，最先进的技术是使用 [HOG +线性 SVM](https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/) 来检测图像中的对象。

我们将借用 HOG +线性 SVM 的元素，将任何深度神经网络图像分类器转换为对象检测器。

**HOG+线性 SVM 的第一个关键要素是使用*图像金字塔。***

“图像金字塔”是图像的**多尺度表示**:

利用图像金字塔允许我们**在图像的不同比例(即大小)**(图 2 )下找到图像中的对象。

在金字塔的底部，我们有原始大小的原始图像(就宽度和高度而言)。

并且在每个后续层，图像被调整大小(二次采样)并且可选地被平滑(通常通过高斯模糊)。

图像被渐进地二次采样，直到满足某个停止标准，这通常是当已经达到最小尺寸并且不再需要进行二次采样时。

我们需要的第二个关键要素是*滑动窗口:*

顾名思义，滑动窗口是一个固定大小的矩形，在一幅图像中从左到右和从上到下滑动。(正如**图 3** 所示，我们的滑动窗口可以用来检测输入图像中的人脸)。

在窗口的每一站，我们将:

1.  提取 ROI
2.  通过我们的图像分类器(例如。、线性 SVM、CNN 等。)
3.  获得输出预测

**结合图像金字塔，滑动窗口允许我们在输入图像**的*不同位置*和*多尺度*定位物体

**我们需要的最后一个关键因素是*非极大值抑制。***

当执行对象检测时，我们的对象检测器通常会在图像中的对象周围产生多个重叠的边界框。

这种行为*完全正常*——它只是意味着随着滑动窗口接近图像，我们的分类器组件返回越来越大的肯定检测概率。

当然，多个边界框带来了一个问题——那里只有*一个*对象，我们需要以某种方式折叠/移除无关的边界框。

该问题的解决方案是应用非最大值抑制(NMS ),它折叠弱的、重叠的边界框，支持更有把握的边界框:

在左边的*，*我们有多个检测，而在右边的*，*我们有非最大值抑制的输出，其将多个边界框折叠成一个*单个*检测。

### **结合传统计算机视觉和深度学习构建物体检测器**

为了采用任何为*图像分类*而训练的卷积神经网络，并将其用于*物体检测*，我们将利用传统计算机视觉的三个关键要素:

1.  **图像金字塔:**以不同的比例/大小定位对象。
2.  **滑动窗口:**精确检测给定物体在图像中的位置*。*
3.  **非最大值抑制:**折叠弱的重叠包围盒。

我们算法的一般流程是:

*   **步骤#1:** 输入图像
*   **步骤#2:** 构建图像金字塔
*   **步骤#3:** 对于图像金字塔的每个比例，运行一个滑动窗口
    *   **步骤#3a:** 对于滑动窗口的每次停止，提取 ROI
    *   **步骤#3b:** 获取感兴趣区域，并将其通过我们最初为图像分类而训练的 CNN
    *   **步骤#3c:** 检查 CNN 的顶级类别标签的概率，并且如果满足最小置信度，则记录(1)类别标签和(2)滑动窗口的位置
*   **步骤#4:** 对边界框应用类别式非最大值抑制
*   **步骤#5:** 将结果返回给调用函数

这看起来似乎是一个复杂的过程，但是正如你将在这篇文章的剩余部分看到的，**我们可以用< 200 行代码实现整个物体检测过程！**

### **配置您的开发环境**

要针对本教程配置您的系统，我首先建议您遵循以下任一教程:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

这两个教程都将帮助您在一个方便的 Python 虚拟环境中，用这篇博文所需的所有软件来配置您的系统。

请注意 [PyImageSearch 不推荐也不支持 CV/DL 项目](https://pyimagesearch.com/faqs/single-faq/can-you-help-me-do-___-on-windows/)的窗口。

### **项目结构**

一旦你提取了。zip 从这篇博文的 ***【下载】*** 部分，你的目录将被组织如下:

```py
.
├── images
│   ├── hummingbird.jpg
│   ├── lawn_mower.jpg
│   └── stingray.jpg
├── pyimagesearch
│   ├── __init__.py
│   └── detection_helpers.py
└── detect_with_classifier.py

2 directories, 6 files
```

### **实现我们的图像金字塔和滑动窗口实用功能**

为了将我们的 CNN 图像分类器变成一个对象检测器，我们必须首先实现助手工具来构造滑动窗口和图像金字塔。

现在让我们实现这个助手函数——打开`pyimagesearch`模块中的`detection_helpers.py`文件，并插入以下代码:

```py
# import the necessary packages
import imutils

def sliding_window(image, step, ws):
	# slide a window across the image
	for y in range(0, image.shape[0] - ws[1], step):
		for x in range(0, image.shape[1] - ws[0], step):
			# yield the current window
			yield (x, y, image[y:y + ws[1], x:x + ws[0]])
```

```py
def image_pyramid(image, scale=1.5, minSize=(224, 224)):
	# yield the original image
	yield image

	# keep looping over the image pyramid
	while True:
		# compute the dimensions of the next image in the pyramid
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image
```

更多细节，请参考我的 *[用 Python 和 OpenCV](https://pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/)* 实现图像金字塔的文章，其中还包括一个替代的 scikit-image 图像金字塔实现，可能对你有用。

### **使用 Keras 和 TensorFlow 将预先训练的图像分类器转变为对象检测器**

```py
# import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from pyimagesearch.detection_helpers import sliding_window
from pyimagesearch.detection_helpers import image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2
```

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-s", "--size", type=str, default="(200, 150)",
	help="ROI size (in pixels)")
ap.add_argument("-c", "--min-conf", type=float, default=0.9,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--visualize", type=int, default=-1,
	help="whether or not to show extra visualizations for debugging")
args = vars(ap.parse_args())
```

```py
# initialize variables used for the object detection procedure
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224, 224)
```

理解上述每个常量控制什么，对于理解如何使用 Keras、TensorFlow 和 OpenCV 将图像分类器转变为对象检测器至关重要。在继续下一步之前，一定要在心里区分这些。

让我们加载我们的 ResNet 分类 CNN 并输入图像:

```py
# load our network weights from disk
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=True)

# load the input image from disk, resize it such that it has the
# has the supplied width, and then grab its dimensions
orig = cv2.imread(args["image"])
orig = imutils.resize(orig, width=WIDTH)
(H, W) = orig.shape[:2]
```

**36 线**加载 [ImageNet](http://www.image-net.org/) 上预训练的 ResNet。如果您选择使用不同的预训练分类器，您可以在这里为您的特定项目替换一个。要学习如何训练自己的分类器，建议你阅读 *[用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)* 。

```py
# initialize the image pyramid
pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)

# initialize two lists, one to hold the ROIs generated from the image
# pyramid and sliding window, and another list used to store the
# (x, y)-coordinates of where the ROI was in the original image
rois = []
locs = []

# time how long it takes to loop over the image pyramid layers and
# sliding window locations
start = time.time()
```

```py
# loop over the image pyramid
for image in pyramid:
	# determine the scale factor between the *original* image
	# dimensions and the *current* layer of the pyramid
	scale = W / float(image.shape[1])

	# for each layer of the image pyramid, loop over the sliding
	# window locations
	for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
		# scale the (x, y)-coordinates of the ROI with respect to the
		# *original* image dimensions
		x = int(x * scale)
		y = int(y * scale)
		w = int(ROI_SIZE[0] * scale)
		h = int(ROI_SIZE[1] * scale)

		# take the ROI and preprocess it so we can later classify
		# the region using Keras/TensorFlow
		roi = cv2.resize(roiOrig, INPUT_SIZE)
		roi = img_to_array(roi)
		roi = preprocess_input(roi)

		# update our list of ROIs and associated coordinates
		rois.append(roi)
		locs.append((x, y, x + w, y + h))
```

```py
		# check to see if we are visualizing each of the sliding
		# windows in the image pyramid
		if args["visualize"] > 0:
			# clone the original image and then draw a bounding box
			# surrounding the current region
			clone = orig.copy()
			cv2.rectangle(clone, (x, y), (x + w, y + h),
				(0, 255, 0), 2)

			# show the visualization and current ROI
			cv2.imshow("Visualization", clone)
			cv2.imshow("ROI", roiOrig)
			cv2.waitKey(0)
```

```py
# show how long it took to loop over the image pyramid layers and
# sliding window locations
end = time.time()
print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(
	end - start))

# convert the ROIs to a NumPy array
rois = np.array(rois, dtype="float32")

# classify each of the proposal ROIs using ResNet and then show how
# long the classifications took
print("[INFO] classifying ROIs...")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("[INFO] classifying ROIs took {:.5f} seconds".format(
	end - start))

# decode the predictions and initialize a dictionary which maps class
# labels (keys) to any ROIs associated with that label (values)
preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}
```

```py
# loop over the predictions
for (i, p) in enumerate(preds):
	# grab the prediction information for the current ROI
	(imagenetID, label, prob) = p[0]

	# filter out weak detections by ensuring the predicted probability
	# is greater than the minimum probability
	if prob >= args["min_conf"]:
		# grab the bounding box associated with the prediction and
		# convert the coordinates
		box = locs[i]

		# grab the list of predictions for the label and add the
		# bounding box and probability to the list
		L = labels.get(label, [])
		L.append((box, prob))
		labels[label] = L
```

```py
# loop over the labels for each of detected objects in the image
for label in labels.keys():
	# clone the original image so that we can draw on it
	print("[INFO] showing results for '{}'".format(label))
	clone = orig.copy()

	# loop over all bounding boxes for the current label
	for (box, prob) in labels[label]:
		# draw the bounding box on the image
		(startX, startY, endX, endY) = box
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)

	# show the results *before* applying non-maxima suppression, then
	# clone the image again so we can display the results *after*
	# applying non-maxima suppression
	cv2.imshow("Before", clone)
	clone = orig.copy()
```

```py
	# extract the bounding boxes and associated prediction
	# probabilities, then apply non-maxima suppression
	boxes = np.array([p[0] for p in labels[label]])
	proba = np.array([p[1] for p in labels[label]])
	boxes = non_max_suppression(boxes, proba)

	# loop over all bounding boxes that were kept after applying
	# non-maxima suppression
	for (startX, startY, endX, endY) in boxes:
		# draw the bounding box and label on the image
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.putText(clone, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

	# show the output after apply non-maxima suppression
	cv2.imshow("After", clone)
	cv2.waitKey(0)
```

为了应用 NMS，我们首先通过**行 159 和 160** 提取边界`boxes`和相关的预测概率(`proba`)。然后，我们将这些结果传递给我的 NMS 的[模拟](https://github.com/jrosebr1/imutils)实现(**第 161 行**)。关于非极大值抑制的更多细节，请务必参考我的博文。

应用 NMS 后，**行 165-171** 在“之后”图像上标注边界框矩形和标签。**第 174 和 175 行**显示结果，直到按下一个键，此时所有 GUI 窗口关闭，脚本退出。

干得好！在下一节中，我们将分析使用图像分类器进行对象检测的方法的结果。

### **使用 Keras 和 TensorFlow 对物体检测器结果进行图像分类**

在这一点上，我们准备看到我们努力工作的成果。

确保使用本教程的 ***【下载】*** 部分下载这篇博文的源代码和示例图片。

从那里，打开一个终端，并执行以下命令:

```py
$ python detect_with_classifier.py --image images/stingray.jpg --size "(300, 150)"
[INFO] loading network...
[INFO] looping over pyramid/windows took 0.19142 seconds
[INFO] classifying ROIs...
[INFO] classifying ROIs took 9.67027 seconds
[INFO] showing results for 'stingray'
```

在这里，您可以看到我输入了一个包含一个*“stingray”*的示例图像，在 ImageNet 上训练的 CNN 将能够识别该图像(因为 ImageNet 包含一个“stingray”类)。

**图 7 *(上)*** 显示了我们的目标检测程序的原始输出。

注意黄貂鱼周围有多个重叠的边界框。

应用非最大值抑制(**图 7，*底部*** )将边界框折叠成单个检测。

让我们尝试另一个图像，这是一只蜂鸟(同样，在 ImageNet 上训练的网络将能够识别):

```py
$ python detect_with_classifier.py --image images/hummingbird.jpg --size "(250, 250)"
[INFO] loading network...
[INFO] looping over pyramid/windows took 0.07845 seconds
[INFO] classifying ROIs...
[INFO] classifying ROIs took 4.07912 seconds
[INFO] showing results for 'hummingbird'
```

**图 8 *(上)*** 显示了我们检测程序的原始输出，而*底部*显示了应用非极大值抑制后的输出。

同样，我们的“图像分类器转物体探测器”程序在这里表现良好。

但是，现在让我们尝试一个示例图像，其中我们的对象检测算法没有以最佳方式执行:

```py
$ python detect_with_classifier.py --image images/lawn_mower.jpg --size "(200, 200)"
[INFO] loading network...
[INFO] looping over pyramid/windows took 0.13851 seconds
[INFO] classifying ROIs...
[INFO] classifying ROIs took 7.00178 seconds
[INFO] showing results for 'lawn_mower'
[INFO] showing results for 'half_track'
```

乍一看，这种方法似乎非常有效——我们能够在输入图像中定位*“割草机”*。

但是对于一辆*“半履带”*(一辆前面有普通车轮，后面有坦克状履带的军用车辆)，实际上有一个*秒*检测:

```py
$ python detect_with_classifier.py --image images/lawn_mower.jpg --size "(200, 200)" --min-conf 0.95
[INFO] loading network...
[INFO] looping over pyramid/windows took 0.13618 seconds
[INFO] classifying ROIs...
[INFO] classifying ROIs took 6.99953 seconds
[INFO] showing results for 'lawn_mower'
```

通过将最小置信度提高到 95%，我们过滤掉了不太有信心的*【半履带】*预测，只留下(正确的)*【割草机】*物体检测。

虽然我们将预先训练的图像分类器转变为对象检测器的过程并不*完美*，但它仍然可以用于某些情况，**特别是在受控环境中捕捉图像时。**

在本系列的剩余部分中，我们将学习如何改进我们的对象检测结果，并构建一个更加健壮的基于深度学习的对象检测器。

### **问题、限制和后续步骤**

如果您仔细检查我们的对象检测程序的结果，您会注意到一些关键的要点:

1.  **实际物体探测器是*慢*。**构建所有图像金字塔和滑动窗口位置需要大约 1/10 秒，这还不包括网络对所有感兴趣区域进行预测所需的时间(在 3 GHz CPU 上为 4-9 秒)！
2.  边界框的位置不一定准确。这种对象检测算法的最大问题是，我们检测的准确性取决于我们对图像金字塔比例、滑动窗口步长和 ROI 大小的选择。如果这些值中的任何一个是关闭的，那么我们的检测器将执行次优。
3.  网络不是端到端可训练的。原因基于深度学习的物体检测器如更快的 R-CNN、SSDs、YOLO 等。表现如此之好是因为它们是*端到端可训练的，*这意味着边界框预测中的任何误差都可以通过反向传播和更新网络的权重来变得更准确——因为我们使用的是具有固定权重的预训练图像分类器，所以我们不能通过网络反向传播误差项。

在这个由四部分组成的系列中，我们将研究如何解决这些问题，并构建一个类似于 R-CNN 网络家族的对象检测器。

## **总结**

在本教程中，您学习了如何使用 Keras、TensorFlow 和 OpenCV 将任何预先训练的深度学习图像分类器转化为对象检测器。

为了完成这项任务，我们将**深度学习**与**传统计算机视觉**算法结合起来:

*   为了检测在**不同比例**(即尺寸)的物体，我们利用了**图像金字塔、**，其获取我们的输入图像并对其重复下采样。
*   为了检测在**不同位置**的物体，我们使用了**滑动窗口**，它从左到右和从上到下在输入图像上滑动一个固定大小的窗口——在窗口的每一站，我们提取 ROI 并将其通过我们的图像分类器。
*   对象检测算法为图像中的对象产生多个重叠的边界框是很自然的；为了将这些重叠的边界框“折叠”成单个检测，我们应用了**非最大值抑制。**

我们拼凑的对象检测例程的最终结果相当合理，但是有两个主要问题:

1.  网络不是端到端可训练的。我们实际上并不是在“学习”探测物体；相反，我们只是采用感兴趣区域，并使用为图像分类而训练的 CNN 对它们进行分类。
2.  **物体检测结果慢得令人难以置信。**在我的英特尔至强 W 3 Ghz 处理器上，根据输入图像的分辨率，对单幅图像应用物体检测大约需要 4-9.5 秒。这种物体检测器不能实时应用。

为了解决这两个问题，下周，我们将开始探索从 R-CNN、快速 R-CNN 和更快 R-CNN 家族构建对象检测器所需的算法。

这将是一个伟大的系列教程，所以你不会想错过它们！

**要下载这篇文章的源代码(并在本系列的下一篇教程发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***