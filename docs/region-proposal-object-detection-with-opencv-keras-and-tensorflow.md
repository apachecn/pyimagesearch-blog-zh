# 基于 OpenCV、Keras 和 TensorFlow 的区域提议对象检测

> 原文：<https://pyimagesearch.com/2020/07/06/region-proposal-object-detection-with-opencv-keras-and-tensorflow/>

在本教程中，您将学习如何使用 OpenCV、Keras 和 TensorFlow 利用区域建议进行对象检测。

**今天的教程是我们关于深度学习和对象检测的 4 部分系列的第 3 部分:**

*   **Part 1:** *[用 Keras 和 TensorFlow](https://pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/)* 把任何深度学习图像分类器变成物体检测器
*   **第二部分:** *[OpenCV 选择性搜索对象检测](https://pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/)*
*   **第三部分:** *用 OpenCV、Keras 和 TensorFlow 进行物体检测的区域建议*(今天的教程)
*   **第四部分:** *用 Keras 和 TensorFlow 进行 R-CNN 物体检测*

在上周的教程中，我们学习了如何利用**选择性搜索**来*取代*使用边界框和滑动窗口进行物体检测的传统计算机视觉方法。

**但是问题仍然存在:我们如何获得区域提议(即，图像中*可能*包含感兴趣的对象的区域),然后实际上*将它们分类为*,以获得我们最终的对象检测？**

我们将在本教程中讨论这个问题。

**要了解如何使用 OpenCV、Keras 和 TensorFlow 使用区域建议执行对象检测，*请继续阅读。***

## **使用 OpenCV、Keras 和 TensorFlow 进行区域提议对象检测**

在本教程的第一部分，我们将讨论区域提议的概念，以及如何在基于深度学习的对象检测管道中使用它们。

然后，我们将使用 OpenCV、Keras 和 TensorFlow 实现区域提议对象检测。

我们将通过查看我们的区域提议对象检测结果来结束本教程。

### 什么是区域提议，它们如何用于对象检测？

我们在上周的教程 *[中讨论了区域提议的概念和选择性搜索算法【OpenCV 用于对象检测的选择性搜索](https://pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/)*——**我建议您在今天继续之前阅读一下该教程，**但要点是传统的计算机视觉对象检测算法依赖于图像金字塔和滑动窗口来定位图像中的对象，并且改变比例和位置:

图像金字塔和滑动窗口方法存在一些问题，但主要问题是:

1.  滑动窗口/图像金字塔非常慢
2.  它们对超参数选择很敏感(即金字塔尺度大小、ROI 大小和窗口步长)
3.  它们的计算效率很低

**区域提议算法寻求*取代*传统的图像金字塔和滑动窗口方法。**

这些算法:

1.  接受输入图像
2.  通过应用[超像素聚类算法](https://pyimagesearch.com/tag/superpixel/)对其进行过度分割
3.  基于五个分量(颜色相似性、纹理相似性、尺寸相似性、形状相似性/兼容性以及线性组合前述分数的最终元相似性)来合并超像素的片段

**最终结果是指示在图像中的什么地方*可能是*的物体的提议:**

请注意我是如何将图片上方句子中的*“could”*——**记住，区域提议算法*不知道*给定的区域*是否实际上*包含一个对象。**

相反，区域建议方法只是告诉我们:

> 嘿，这看起来像是输入图像的一个有趣区域。让我们应用计算量更大的分类器来确定这个区域中实际上有什么。

区域提议算法往往比图像金字塔和滑动窗口的传统对象检测技术更有效，因为:

*   检查的单个 ROI 更少
*   这比彻底检查输入图像的每个比例/位置要快
*   精确度的损失是最小的，如果有的话

在本教程的其余部分，您将学习如何实现区域建议对象检测。

### 配置您的开发环境

要针对本教程配置您的系统，我建议您遵循以下任一教程:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

这两个教程都将帮助您在一个方便的 Python 虚拟环境中，用这篇博文所需的所有软件来配置您的系统。

请注意 [PyImageSearch 不推荐也不支持 CV/DL 项目](https://pyimagesearch.com/faqs/single-faq/can-you-help-me-do-___-on-windows/)的窗口。

### **项目结构**

请务必从 ***“下载”*** 部分获取今天的文件，以便您可以跟随今天的教程:

```py
$ tree    
.
├── beagle.png
└── region_proposal_detection.py

0 directories, 2 files
```

如您所见，我们今天的项目布局非常简单，由一个 Python 脚本组成，对于今天的区域提议对象检测示例，这个脚本被恰当地命名为`region_proposal_detection.py`。

我还附上了我家的小猎犬杰玛的照片。我们将使用这张照片来测试我们的 OpenCV、Keras 和 TensorFlow 区域提议对象检测系统。

### **用 OpenCV、Keras 和 TensorFlow 实现区域提议对象检测**

让我们开始实现我们的区域提议对象检测器。

打开一个新文件，将其命名为`region_proposal_detection.py`，并插入以下代码:

```py
# import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
```

```py
def selective_search(image, method="fast"):
	# initialize OpenCV's selective search implementation and set the
	# input image
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)

	# check to see if we are using the *fast* but *less accurate* version
	# of selective search
	if method == "fast":
		ss.switchToSelectiveSearchFast()

	# otherwise we are using the *slower* but *more accurate* version
	else:
		ss.switchToSelectiveSearchQuality()

	# run selective search on the input image
	rects = ss.process()

	# return the region proposal bounding boxes
	return rects
```

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-m", "--method", type=str, default="fast",
	choices=["fast", "quality"],
	help="selective search method")
ap.add_argument("-c", "--conf", type=float, default=0.9,
	help="minimum probability to consider a classification/detection")
ap.add_argument("-f", "--filter", type=str, default=None,
	help="comma separated list of ImageNet labels to filter on")
args = vars(ap.parse_args())
```

```py
# grab the label filters command line argument
labelFilters = args["filter"]

# if the label filter is not empty, break it into a list
if labelFilters is not None:
	labelFilters = labelFilters.lower().split(",")
```

```py
# load ResNet from disk (with weights pre-trained on ImageNet)
print("[INFO] loading ResNet...")
model = ResNet50(weights="imagenet")

# load the input image from disk and grab its dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
```

```py
# run selective search on the input image
print("[INFO] performing selective search with '{}' method...".format(
	args["method"]))
rects = selective_search(image, method=args["method"])
print("[INFO] {} regions found by selective search".format(len(rects)))

# initialize the list of region proposals that we'll be classifying
# along with their associated bounding boxes
proposals = []
boxes = []
```

*   ``proposals`` :在**第 68 行**初始化，这个列表将保存来自我们的输入`--image`的足够大的预处理 ROI，我们将把这些 ROI 送入我们的 ResNet 分类器。
*   `boxes`:在**第 69 行**初始化，这个包围盒坐标列表对应于我们的`proposals`，并且类似于`rects`，有一个重要的区别:只包括足够大的区域。

```py
# loop over the region proposal bounding box coordinates generated by
# running selective search
for (x, y, w, h) in rects:
	# if the width or height of the region is less than 10% of the
	# image width or height, ignore it (i.e., filter out small
	# objects that are likely false-positives)
	if w / float(W) < 0.1 or h / float(H) < 0.1:
		continue

	# extract the region from the input image, convert it from BGR to
	# RGB channel ordering, and then resize it to 224x224 (the input
	# dimensions required by our pre-trained CNN)
	roi = image[y:y + h, x:x + w]
	roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	roi = cv2.resize(roi, (224, 224))

	# further preprocess by the ROI
	roi = img_to_array(roi)
	roi = preprocess_input(roi)

	# update our proposals and bounding boxes lists
	proposals.append(roi)
	boxes.append((x, y, w, h))
```

```py
# convert the proposals list into NumPy array and show its dimensions
proposals = np.array(proposals)
print("[INFO] proposal shape: {}".format(proposals.shape))

# classify each of the proposal ROIs using ResNet and then decode the
# predictions
print("[INFO] classifying proposals...")
preds = model.predict(proposals)
preds = imagenet_utils.decode_predictions(preds, top=1)

# initialize a dictionary which maps class labels (keys) to any
# bounding box associated with that label (values)
labels = {}
```

```py
# loop over the predictions
for (i, p) in enumerate(preds):
	# grab the prediction information for the current region proposal
	(imagenetID, label, prob) = p[0]

	# only if the label filters are not empty *and* the label does not
	# exist in the list, then ignore it
	if labelFilters is not None and label not in labelFilters:
		continue

	# filter out weak detections by ensuring the predicted probability
	# is greater than the minimum probability
	if prob >= args["conf"]:
		# grab the bounding box associated with the prediction and
		# convert the coordinates
		(x, y, w, h) = boxes[i]
		box = (x, y, x + w, y + h)

		# grab the list of predictions for the label and add the
		# bounding box + probability to the list
		L = labels.get(label, [])
		L.append((box, prob))
		labels[label] = L
```

```py
# loop over the labels for each of detected objects in the image
for label in labels.keys():
	# clone the original image so that we can draw on it
	print("[INFO] showing results for '{}'".format(label))
	clone = image.copy()

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
	clone = image.copy()
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

NMS 之前的*和 NMS* 之后的*都将保留在屏幕上，直到按下一个键(**第 170 行**)。*

### **使用 OpenCV、Keras 和 TensorFlow 的区域提议对象检测结果**

我们现在准备执行区域提议对象检测！

确保使用本教程的 ***“下载”*** 部分下载源代码和示例图像。

从那里，打开一个终端，并执行以下命令:

```py
$ python region_proposal_detection.py --image beagle.png
[INFO] loading ResNet...
[INFO] performing selective search with 'fast' method...
[INFO] 922 regions found by selective search
[INFO] proposal shape: (534, 224, 224, 3)
[INFO] classifying proposals...
[INFO] showing results for 'beagle'
[INFO] showing results for 'clog'
[INFO] showing results for 'quill'
[INFO] showing results for 'paper_towel'
```

最初，我们的结果看起来相当不错。

如果你看一下**图 3** ，你会看到在*左边*我们有*“小猎犬”*类(一种狗)的对象检测，在*右边*我们有应用非最大值抑制后的输出。

从输出可以看出，我家的小猎犬 Jemma 被正确检测到了！

然而，正如我们的其余结果所示，我们的模型还报告我们检测到了*“木屐”*(一种木鞋):

以及*【鹅毛笔】*(用羽毛制成的书写笔):

最后，一张*【纸巾】*:

看看每一类的 ROI，可以想象我们的 CNN 在做这些分类的时候是如何的混乱。

**但是我们实际上如何*去除*不正确的物体检测呢？**

这里的解决方案是，我们可以只过滤掉我们关心的检测。

例如，如果我正在构建一个“beagle detector”应用程序，我将提供`--filter beagle`命令行参数:

```py
$ python region_proposal_detection.py --image beagle.png --filter beagle
[INFO] loading ResNet...
[INFO] performing selective search with 'fast' method...
[INFO] 922 regions found by selective search
[INFO] proposal shape: (534, 224, 224, 3)
[INFO] classifying proposals...
[INFO] showing results for 'beagle'
```

在这种情况下，只找到*“beagle”*类(其余的都被丢弃)。

### **问题和局限**

正如我们的结果部分所展示的，我们的区域提议对象检测器“只是有点儿工作”——**当我们获得了*正确的*对象检测时，我们也获得了许多噪声。**

在下周的教程中，我将向你展示我们如何使用选择性搜索和区域建议来构建一个完整的 R-CNN 物体检测器管道，它比我们今天在这里讨论的方法更加精确。

## **总结**

在本教程中，您学习了如何使用 OpenCV、Keras 和 TensorFlow 执行区域提议对象检测。

使用区域建议进行对象检测是一个 4 步过程:

1.  **步骤#1:** 使用选择性搜索(区域提议算法)来生成输入图像的候选区域，其中*可能*包含感兴趣的对象。
2.  **步骤#2:** 获取这些区域，并通过预先训练的 CNN 对候选区域进行分类(同样，*可能*包含一个物体)。
3.  **步骤#3:** 应用非最大值抑制(NMS)来抑制弱的、重叠的边界框。
4.  步骤#4: 将最终的边界框返回给调用函数。

我们使用 OpenCV、Keras 和 TensorFlow 实现了上面的管道— *全部用了大约 150 行代码！*

但是，您会注意到，我们使用了一个在 ImageNet 数据集上经过*预训练*的网络。

这就引出了问题:

*   如果我们想在我们自己的自定义数据集上训练一个网络会怎么样？
*   我们如何使用选择性搜索来训练网络？
*   这将如何改变我们用于物体检测的推理代码？

我将在下周的教程中回答这些问题。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***