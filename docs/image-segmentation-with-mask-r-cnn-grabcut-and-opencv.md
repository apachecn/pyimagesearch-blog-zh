# 使用掩模 R-CNN、GrabCut 和 OpenCV 的图像分割

> 原文：<https://pyimagesearch.com/2020/09/28/image-segmentation-with-mask-r-cnn-grabcut-and-opencv/>

在本教程中，您将学习如何使用 Mask R-CNN、GrabCut 和 OpenCV 执行图像分割。

几个月前，你学习了[如何使用 GrabCut 算法从背景](https://pyimagesearch.com/2020/07/27/opencv-grabcut-foreground-segmentation-and-extraction/) [中分割出前景物体。](https://stackedit.io/app#) GrabCut 工作得相当好**，但是需要我们手动提供*，在输入图像中的*处，对象是**，这样 GrabCut 可以应用它的分割魔法。

另一方面，掩模 R-CNN 可以*自动*预测输入图像中每个对象的**边界框**和**逐像素分割掩模**。缺点是 Mask R-CNN 生成的遮罩并不总是“干净的”——通常会有一点背景“渗入”前景分割。

这提出了以下问题:

> *有没有可能把 Mask R-CNN 和 GrabCut 结合在一起*？
> 
> *我们是否可以使用 Mask R-CNN 计算初始分割，然后使用 GrabCut 进行细化？*

我们当然可以——本教程的其余部分将告诉你如何做。

**要了解如何使用 Mask R-CNN、GrabCut 和 OpenCV 执行图像分割，*继续阅读。***

## **使用掩膜 R-CNN、GrabCut 和 OpenCV 进行图像分割**

在本教程的第一部分，我们将讨论为什么我们可能要结合 GrabCut 与掩模 R-CNN 的图像分割。

在此基础上，我们将实现一个 Python 脚本:

1.  从磁盘加载输入图像
2.  为输入图像中的每个对象计算逐像素分割遮罩
3.  通过遮罩对对象应用 GrabCut 以改进图像分割

然后，我们将回顾一起应用 Mask R-CNN 和 GrabCut 的结果。

教程的*【概要】*涵盖了这种方法的一些局限性。

### **为什么要一起使用 GrabCut 和 Mask R-CNN 进行图像分割？**

Mask R-CNN 是一种最先进的深度神经网络架构，用于图像分割。使用 Mask R-CNN，我们可以*自动*计算图像中物体的像素级遮罩，允许我们从**背景中分割出**前景**。**

通过屏蔽 R-CNN 计算的屏蔽示例可以在本节顶部的图 1 的**中看到。**

*   在左上角的*，*我们有一个谷仓场景的输入图像。
*   屏蔽 R-CNN 检测到一匹*马*，然后自动计算其对应的分割屏蔽(*右上*)。
*   在*底部，*我们可以看到将计算出的遮罩应用到输入图像的结果——**注意马是如何被自动分割的。**

然而，掩模 R-CNN 的输出与完美的掩模相差*远*。我们可以看到背景(例如。马站立的场地上的泥土)正在“渗透”到前景中。

**我们在这里的目标是使用 GrabCut 改进这个遮罩，以获得更好的分割:**

在上图中，您可以看到使用 mask R-CNN 预测的 Mask 作为 GrabCut 种子应用 GrabCut 的输出。

**注意分割有多紧，*特别是在马的腿部周围。*** 不幸的是，我们现在已经失去了马的头顶以及它的蹄子。

同时使用 GrabCut 和 Mask R-CNN 可能会有所取舍。在某些情况下，它会工作得很好，而在其他情况下，它会让你的结果更糟。这完全取决于您的应用和您要分割的图像类型。

在今天的剩余教程中，我们将探索一起应用 Mask R-CNN 和 GrabCut 的结果。

### 配置您的开发环境

本教程只要求您在 Python 虚拟环境中安装 OpenCV。

对于大多数读者来说，最好的入门方式是遵循我的 *[pip install opencv](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)* 教程，该教程指导如何设置环境以及在 macOS、Ubuntu 或 Raspbian 上需要哪些 Python 包。

或者，如果你手头有一个支持 CUDA 的 GPU，你可以遵循我的 [OpenCV with CUDA 安装指南](https://pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/)。

### **项目结构**

继续从这篇博文的 ***【下载】*** 部分抓取代码并屏蔽 R-CNN 深度学习模型。一旦你提取了。zip，您将看到以下文件:

```py
$ tree --dirsfirst
.
├── mask-rcnn-coco
│   ├── colors.txt
│   ├── frozen_inference_graph.pb
│   ├── mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
│   └── object_detection_classes_coco.txt
├── example.jpg
└── mask_rcnn_grabcut.py

1 directory, 6 files
```

### **用掩模 R-CNN 和 GrabCut 实现图像分割**

让我们一起开始使用 OpenCV 实现 Mask R-CNN 和 GrabCut 进行图像分割。

打开一个新文件，将其命名为`mask_rcnn_grabcut.py`，并插入以下代码:

```py
# import the necessary packages
import numpy as rnp
import argparse
import imutils
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mask-rcnn", required=True,
	help="base path to mask-rcnn directory")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-u", "--use-gpu", type=bool, default=0,
	help="boolean indicating if CUDA GPU should be used")
ap.add_argument("-e", "--iter", type=int, default=10,
	help="# of GrabCut iterations (larger value => slower runtime)")
args = vars(ap.parse_args())
```

在导入必要的包(**第 2-6 行**)之后，我们定义我们的[命令行参数](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/) ( **第 9-22 行**):

```py
# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
```

```py
# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# check if we are going to use GPU
if args["use_gpu"]:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
```

**第 35-38 行**给出了模型配置和预训练权重的路径。我们的模型是基于张量流的。然而，如果需要的话，OpenCV 的 DNN 模块能够加载模型，并使用支持 CUDA 的 NVIDIA GPU 为推理做准备(**第 43-50 行**)。

既然我们的模型已经加载，我们也准备加载我们的图像并执行推理:

```py
# load our input image from disk and display it to our screen
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
cv2.imshow("Input", image)

# construct a blob from the input image and then perform a
# forward pass of the Mask R-CNN, giving us (1) the bounding box
# coordinates of the objects in the image along with (2) the
# pixel-wise segmentation for each specific object
blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)
(boxes, masks) = net.forward(["detection_out_final",
	"detection_masks"])
```

1.  `rcnnMask` : R-CNN 屏蔽
2.  ``rcnnOutput`` : R-CNN 屏蔽输出
3.  ``outputMask`` :基于来自我们的 mask R-CNN 的 Mask 近似值的 GrabCut mask(参考我们的[以前的 GrabCut 教程](https://pyimagesearch.com/2020/07/27/opencv-grabcut-foreground-segmentation-and-extraction/)的*“用 OpenCV 进行 GrabCut:用 masks 初始化”*部分)
4.  ``output`` : GrabCut + Mask R-CNN 屏蔽输出

请务必参考这个列表，这样您就可以跟踪剩余代码块的每个输出图像。

让我们开始循环检测:

```py
# loop over the number of detected objects
for i in range(0, boxes.shape[2]):
	# extract the class ID of the detection along with the
	# confidence (i.e., probability) associated with the
	# prediction
	classID = int(boxes[0, 0, i, 1])
	confidence = boxes[0, 0, i, 2]

	# filter out weak predictions by ensuring the detected
	# probability is greater than the minimum probability
	if confidence > args["confidence"]:
		# show the class label
		print("[INFO] showing output for '{}'...".format(
			LABELS[classID]))

		# scale the bounding box coordinates back relative to the
		# size of the image and then compute the width and the
		# height of the bounding box
		(H, W) = image.shape[:2]
		box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
		(startX, startY, endX, endY) = box.astype("int")
		boxW = endX - startX
		boxH = endY - startY
```

**第 67 行**开始我们的检测循环，此时我们继续:

*   抽出`classID`和`confidence` ( **线 71 和 72** )
*   基于我们的`--confidence`阈值(**第 76 行**)，过滤掉弱预测
*   根据图像的原始尺寸缩放边界框坐标(**行 84 和 85** )
*   提取边界框坐标，并确定该框的宽度和高度(**行 86-88** )

从这里开始，我们准备开始制作我们的 R-CNN 蒙版和蒙版图像:

```py
		# extract the pixel-wise segmentation for the object, resize
		# the mask such that it's the same dimensions as the bounding
		# box, and then finally threshold to create a *binary* mask
		mask = masks[i, classID]
		mask = cv2.resize(mask, (boxW, boxH),
			interpolation=cv2.INTER_CUBIC)
		mask = (mask > args["threshold"]).astype("uint8") * 255

		# allocate a memory for our output Mask R-CNN mask and store
		# the predicted Mask R-CNN mask in the GrabCut mask
		rcnnMask = np.zeros(image.shape[:2], dtype="uint8")
		rcnnMask[startY:endY, startX:endX] = mask

		# apply a bitwise AND to the input image to show the output
		# of applying the Mask R-CNN mask to the image
		rcnnOutput = cv2.bitwise_and(image, image, mask=rcnnMask)

		# show the output of the Mask R-CNN and bitwise AND operation
		cv2.imshow("R-CNN Mask", rcnnMask)
		cv2.imshow("R-CNN Output", rcnnOutput)
		cv2.waitKey(0)
```

```py
		# clone the Mask R-CNN mask (so we can use it when applying
		# GrabCut) and set any mask values greater than zero to be
		# "probable foreground" (otherwise they are "definite
		# background")
		gcMask = rcnnMask.copy()
		gcMask[gcMask > 0] = cv2.GC_PR_FGD
		gcMask[gcMask == 0] = cv2.GC_BGD

		# allocate memory for two arrays that the GrabCut algorithm
		# internally uses when segmenting the foreground from the
		# background and then apply GrabCut using the mask
		# segmentation method
		print("[INFO] applying GrabCut to '{}' ROI...".format(
			LABELS[classID]))
		fgModel = np.zeros((1, 65), dtype="float")
		bgModel = np.zeros((1, 65), dtype="float")
		(gcMask, bgModel, fgModel) = cv2.grabCut(image, gcMask,
			None, bgModel, fgModel, iterCount=args["iter"],
			mode=cv2.GC_INIT_WITH_MASK)
```

回想一下我之前的 GrabCut 教程,有两种使用 GrabCut 进行分割的方法:

1.  基于包围盒的
2.  基于遮罩的*(我们将要执行的方法)*

**第 116 行**克隆了`rcnnMask`，这样我们就可以在应用 GrabCut 时使用它。

然后我们设置“可能的前景”和“确定的背景”值(**第 117 和 118 行**)。我们还为 OpenCV 的 GrabCut 算法内部需要的前景和背景模型分配数组(**第 126 行和第 127 行**)。

从那里，我们用必要的参数(**第 128-130 行**)调用`cv2.grabCut`，包括我们初始化的掩码(我们掩码 R-CNN 的结果)。如果你需要重温 OpenCV 的 GrabCut 输入参数和 3 元组返回签名，我强烈推荐参考我的第一篇 GrabCut 博文中的*“OpenCV grab cut”*部分。

关于返回，我们只关心`gcMask`，我们将在接下来看到。

让我们继续并生成我们的**最后两个输出图像:**

```py
		# set all definite background and probable background pixels
		# to 0 while definite foreground and probable foreground
		# pixels are set to 1, then scale the mask from the range
		# [0, 1] to [0, 255]
		outputMask = np.where(
			(gcMask == cv2.GC_BGD) | (gcMask == cv2.GC_PR_BGD), 0, 1)
		outputMask = (outputMask * 255).astype("uint8")

		# apply a bitwise AND to the image using our mask generated
		# by GrabCut to generate our final output image
		output = cv2.bitwise_and(image, image, mask=outputMask)

		# show the output GrabCut mask as well as the output of
		# applying the GrabCut mask to the original input image
		cv2.imshow("GrabCut Mask", outputMask)
		cv2.imshow("Output", output)
		cv2.waitKey(0)
```

我们的**最后两个图像可视化**然后通过剩余的行显示。

在下一部分，我们将检查我们的结果。

### **屏蔽 R-CNN 和 GrabCut 图像分割结果**

我们现在准备应用掩模 R-CNN 和 GrabCut 进行图像分割。

确保您使用了本教程的 ***“下载”*** 部分来下载源代码、示例图像和预训练的屏蔽 R-CNN 权重。

作为参考，下面是我们将应用 GrabCut 和 Mask R-CNN 的输入图像:

打开终端，执行以下命令:

```py
$ python mask_rcnn_grabcut.py --mask-rcnn mask-rcnn-coco --image example.jpg
[INFO] loading Mask R-CNN from disk...
[INFO] showing output for 'horse'...
[INFO] applying GrabCut to 'horse' ROI...
[INFO] showing output for 'person'...
[INFO] applying GrabCut to 'person' ROI...
[INFO] showing output for 'dog'...
[INFO] applying GrabCut to 'dog' ROI...
[INFO] showing output for 'truck'...
[INFO] applying GrabCut to 'truck' ROI...
[INFO] showing output for 'person'...
[INFO] applying GrabCut to 'person' ROI...
```

现在让我们来看看每个单独的图像分割:

在这里，您可以看到 Mask R-CNN 在输入图像中检测到了一匹马。

然后，我们通过 GrabCut 传递该蒙版以改进蒙版，希望获得更好的图像分割。

虽然我们能够通过马的腿来移除背景，但不幸的是，它会切断马蹄和马的头顶。

现在让我们来看看如何分割坐在马背上的骑手:

这种分割比前一种好得多；但是，使用 GrabCut 后，人的头发会脱落。

以下是从输入图像中分割卡车的输出:

Mask R-CNN 在分割卡车方面做得非常好；然而，GrabCut 认为只有格栅，引擎盖和挡风玻璃在前景中，去除了其他部分。

下一个图像包含分割第二个人的可视化效果(远处栅栏旁边的那个人):

这是如何将 Mask R-CNN 和 GrabCut 成功结合用于图像分割的最佳范例之一。

注意我们有一个*明显更紧密的分割*——在应用 GrabCut 后，任何渗入前景的背景(如田野中的草)都被移除了。

最后，这是对狗应用 Mask R-CNN 和 GrabCut 的输出:

掩模 R-CNN 产生的掩模中仍然有大量的背景。

通过应用 GrabCut，可以删除该背景，但不幸的是，狗的头顶也随之丢失。

### **喜忧参半的结果、局限性和缺点**

在查看了本教程的混合结果后，您可能会奇怪为什么我还要写一篇关于一起使用 GrabCut 和 Mask R-CNN 的教程— **在许多情况下，似乎将 GrabCut 应用于 Mask R-CNN mask 实际上会使结果*更糟！***

虽然这是真的，但是*仍然有*种情况(例如**图 7** 中的第二个人物分割)，其中将 GrabCut 应用到遮罩 R-CNN 遮罩实际上*改进了*分割。

我使用了一个具有复杂前景/背景的图像来向您展示这种方法的局限性，但是复杂性较低的图像将获得更好的结果。

一个很好的例子是从输入图像中分割出服装来构建一个时尚搜索引擎。

实例分割网络如 Mask R-CNN，U-Net 等。可以预测每件衣服的位置和面具，从那里 GrabCut 可以提炼面具。

虽然将 Mask R-CNN 和 GrabCut 一起应用于图像分割时肯定会有混合的结果，但仍然值得进行一次实验，看看您的结果是否有所改善。

## **总结**

在本教程中，您学习了如何使用 Mask R-CNN、GrabCut 和 OpenCV 执行图像分割。

我们使用掩模 R-CNN 深度神经网络来计算图像中给定对象的初始前景分割掩模。

来自掩模 R-CNN 的掩模可以自动*计算*,但是通常具有“渗入”前景分割掩模的背景。为了解决这个问题，我们使用 GrabCut 来改进由 mask R-CNN 生成的 Mask。

在某些情况下，GrabCut 产生的图像分割比 Mask R-CNN 产生的原始掩模更好*。在其他情况下，产生的图像分割是 ***更差***——如果我们坚持使用由掩模 R-CNN 制作的掩模，情况会更好。*

 *最大的限制是，即使使用 Mask R-CNN 自动生成的 Mask/bounding box，GrabCut 仍然是一种迭代地需要手动注释来提供最佳结果的算法。由于我们没有手动向 GrabCut 提供提示和建议，因此不能进一步改进这些掩码。

我们是否使用过 Photoshop、GIMP 等图片编辑软件包？，那么我们将有一个漂亮的，易于使用的图形用户界面，这将允许我们提供提示，以 GrabCut 什么是前景，什么是背景。

你当然应该尝试使用 GrabCut 来改进你的屏蔽 R-CNN 屏蔽。在某些情况下，你会发现它工作完美，你会获得更高质量的图像分割。在其他情况下，你最好使用 R-CNN 面具。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****