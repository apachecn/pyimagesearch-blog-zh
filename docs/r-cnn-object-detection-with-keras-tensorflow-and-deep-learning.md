# 使用 Keras、TensorFlow 和深度学习的 R-CNN 对象检测

> 原文：<https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/>

在本教程中，您将学习如何使用 Keras、TensorFlow 和深度学习来构建 R-CNN 对象检测器。

**今天的教程是我们关于深度学习和物体检测的 4 部分系列的最后一部分:**

*   **第一部分:** *[用 Keras、TensorFlow、OpenCV](https://pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/)* 把任何 CNN 图像分类器变成物体检测器
*   **第二部分:** *[OpenCV 选择性搜索对象检测](https://pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/)*
*   **第三部分:** *[用 OpenCV、Keras 和 TensorFlow](https://pyimagesearch.com/2020/07/06/region-proposal-object-detection-with-opencv-keras-and-tensorflow/)* 进行物体检测的区域建议
*   **第四部分:** *用 Keras 和 TensorFlow 进行 R-CNN 物体检测*(今日教程)

上周，您学习了如何使用区域建议和选择性搜索来取代图像金字塔和滑动窗口的传统计算机视觉对象检测管道:

1.  使用选择性搜索，我们生成了候选区域(称为“建议”)，这些区域中的*可能会包含一个感兴趣的对象。*
2.  这些提议被传递给预先训练好的 CNN，以获得实际的分类。
3.  然后，我们通过应用置信度过滤和非极大值抑制来处理结果。

**我们的方法运行得很好，但也带来了一些问题:**

> 如果我们想在我们自己的定制数据集上训练一个对象检测网络会怎么样？
> 
> 我们如何使用选择性搜索来训练网络？
> 
> 使用选择性搜索将如何改变我们的物体检测推理脚本？

事实上，这些问题与 Girshick 等人在他们开创性的深度学习对象检测论文 *[中不得不考虑的问题相同，丰富的特征层次用于精确的对象检测和语义分割。](https://arxiv.org/abs/1311.2524)*

这些问题都将在今天的教程中得到回答——当你读完它时，你将拥有一个功能完整的 R-CNN，类似于 Girshick 等人实现的那个(但被简化了)!

**要了解如何使用 Keras 和 TensorFlow 构建 R-CNN 对象检测器，*继续阅读。***

## **使用 Keras、TensorFlow 和深度学习的 R-CNN 对象检测**

今天关于使用 Keras 和 TensorFlow 构建 R-CNN 对象检测器的教程是我们关于深度学习对象检测器系列中最长的教程。

我建议你相应地安排好你的时间——你可能需要 40 到 60 分钟来完整地阅读这篇教程。慢慢来，因为博文中有许多细节和细微差别(不要害怕阅读教程 2-3 遍，以确保你完全理解它)。

我们将从讨论使用 Keras 和 TensorFlow 实现 R-CNN 对象检测器所需的步骤开始我们的教程。

从那里，我们将回顾我们今天在这里使用的示例对象检测数据集。

接下来，我们将实现我们的配置文件以及一个助手实用函数，该函数用于通过 Union (IoU) 上的[交集来计算对象检测精度。](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

然后，我们将通过应用选择性搜索来构建我们的对象检测数据集。

选择性搜索，以及一点后处理逻辑，将使我们能够识别输入图像中的区域，这些区域中*包含*而*不包含*潜在的感兴趣对象。

我们将这些区域用作我们的训练数据，微调 MobileNet(在 ImageNet 上预先训练)来分类和识别我们数据集中的对象。

最后，我们将实现一个 Python 脚本，通过对输入图像应用选择性搜索，对选择性搜索生成的区域建议进行分类，然后将输出的 R-CNN 对象检测结果显示到我们的屏幕上，该脚本可用于推断/预测。

我们开始吧！

### **使用 Keras 和 TensorFlow 实现 R-CNN 对象检测器的步骤**

实现 R-CNN 对象检测器是一个有点复杂的多步骤过程。

**如果您还没有，请确保您已经阅读了本系列之前的教程，以确保您已经掌握了适当的知识和先决条件:**

1.  *[用 Keras、TensorFlow、OpenCV](https://pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/)* 把任何 CNN 图像分类器变成物体检测器
2.  *OpenCV 选择性搜索对象检测*
3.  *[使用 OpenCV、Keras 和 TensorFlow](https://pyimagesearch.com/2020/07/06/region-proposal-object-detection-with-opencv-keras-and-tensorflow/)* 进行对象检测的区域建议

我假设你对选择性搜索是如何工作的，如何在对象检测管道中使用区域建议，以及如何微调网络有一定的了解。

也就是说，下面你可以看到我们实施 R-CNN 目标检测器的 6 个步骤:

1.  **步骤#1:** 使用选择性搜索建立对象检测数据集
2.  **步骤#2:** 微调用于对象检测的分类网络(最初在 ImageNet 上训练)
3.  **步骤#3:** 创建一个对象检测推理脚本，该脚本利用选择性搜索来建议*可能*包含我们想要检测的对象的区域
4.  **步骤#4:** 使用我们的微调网络对通过选择性搜索提出的每个区域进行分类
5.  **步骤#5:** 应用非最大值抑制来抑制弱的重叠边界框
6.  **步骤#6:** 返回最终的物体检测结果

正如我前面已经提到的，本教程很复杂，涵盖了许多细微的细节。

**因此，如果您需要反复查看*****以确保您理解我们的 R-CNN 对象检测实现，请不要对自己太苛刻。***

 *记住这一点，让我们继续审查我们的 R-CNN 项目结构。

### **我们的目标检测数据集**

如图 2 所示，我们将训练一个 R-CNN 物体检测器来检测输入图像中的浣熊。

这个数据集包含 **200 张图像**和 **217 只浣熊**(一些图像包含不止一只浣熊)。

这个数据集最初是由受人尊敬的数据科学家 Dat Tran 策划的。

浣熊数据集的 GitHub 存储库可以在这里找到；然而，**为了方便起见，我将数据集包含在与本教程相关的*“下载”*中。**

如果您还没有，请确保使用这篇博客文章的 ***“下载”*** 部分来下载浣熊数据集和 Python 源代码，以便您能够完成本教程的其余部分。

### **配置您的开发环境**

要针对本教程配置您的系统，我建议您遵循以下任一教程:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

这两个教程都将帮助您在一个方便的 Python 虚拟环境中，用这篇博文所需的所有软件来配置您的系统。

请注意 [PyImageSearch 不推荐也不支持 CV/DL 项目](https://pyimagesearch.com/faqs/single-faq/can-you-help-me-do-___-on-windows/)的窗口。

### **项目结构**

如果您还没有，请使用 ***“下载”*** 部分来获取今天教程的代码和数据集。

在里面，您会发现以下内容:

```py
$ tree --dirsfirst --filelimit 10
.
├── dataset
│   ├── no_raccoon [2200 entries]
│   └── raccoon [1560 entries]
├── images
│   ├── raccoon_01.jpg
│   ├── raccoon_02.jpg
│   └── raccoon_03.jpg
├── pyimagesearch
│   ├── __init__.py
│   ├── config.py
│   ├── iou.py
│   └── nms.py
├── raccoons
│   ├── annotations [200 entries]
│   └── images [200 entries]
├── build_dataset.py
├── detect_object_rcnn.py
├── fine_tune_rcnn.py
├── label_encoder.pickle
├── plot.png
└── raccoon_detector.h5

8 directories, 13 files
```

### **实现我们的对象检测配置文件**

在我们深入项目之前，让我们首先实现一个存储关键常量和设置的配置文件，我们将在多个 Python 脚本中使用它。

打开`pyimagesearch`模块中的`config.py`文件，插入以下代码:

```py
# import the necessary packages
import os

# define the base path to the *original* input dataset and then use
# the base path to derive the image and annotations directories
ORIG_BASE_PATH = "raccoons"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations"])
```

我们首先在第**行的第 6-8** 行定义到原始浣熊数据集图像和对象检测注释(即，边界框信息)的路径。

接下来，我们定义即将构建的数据集的路径:

```py
# define the base path to the *new* dataset after running our dataset
# builder scripts and then use the base path to derive the paths to
# our output class label directories
BASE_PATH = "dataset"
POSITVE_PATH = os.path.sep.join([BASE_PATH, "raccoon"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_raccoon"])
```

```py
# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200
```

然后设置构建数据集时要使用的正负区域的最大数量:

```py
# define the maximum number of positive and negative images to be
# generated from each image
MAX_POSITIVE = 30
MAX_NEGATIVE = 10
```

最后，我们总结了特定于模型的常数:

```py
# initialize the input dimensions to the network
INPUT_DIMS = (224, 224)

# define the path to the output model and label binarizer
MODEL_PATH = "raccoon_detector.h5"
ENCODER_PATH = "label_encoder.pickle"

# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99
```

**第 28 行**为我们的分类网络(MobileNet，在 ImageNet 上预先训练)设置输入空间维度。

然后我们定义输出文件路径到我们的浣熊分类器和标签编码器(**第 31 和 32 行**)。

在推断过程中(用于滤除假阳性检测)阳性预测所需的最小概率在第 36 行**上设置为 99%。**

### **用并集交集(IoU)测量目标检测精度**

为了衡量我们的对象检测器在预测边界框方面做得有多好，我们将使用并集上的交集(IoU)度量。

IoU 方法计算*预测*边界框和*地面实况*边界框之间重叠面积与联合面积的比率:

检查这个等式，可以看到交集除以并集就是一个简单的比率:

*   在分子中，我们计算预测边界框和真实边界框之间的重叠区域的**。**
*   分母是并集的**区域，或者更简单地说，是由预测边界框和真实边界框*包围的区域。***
*   将重叠的面积除以并集的面积得到我们的最终分数—*上的交集(因此得名)。*

 *我们将使用 IoU 来衡量对象检测的准确性，包括给定的选择性搜索建议与真实边界框的重叠程度(这在我们为训练数据生成正面和负面示例时非常有用)。

如果你有兴趣了解更多关于 IoU 的知识，一定要参考我的教程， *[交集超过并集(IoU)用于对象检测。](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)*

否则，现在让我们简要回顾一下我们的 IoU 实现——打开`pyimagesearch`目录中的`iou.py`文件，并插入以下代码:

```py
def compute_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the intersection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou
```

`comptue_iou`函数接受两个参数`boxA`和`boxB`，这两个参数是我们试图计算联合交集(IoU)的基础事实和预测边界框。就我们的计算而言，参数的顺序并不重要。

在里面，我们开始计算右上角和左下角的 *(x，y)*——边界框的坐标(**第 3-6 行**)。

使用边界框坐标，我们计算边界框的交集(重叠区域)(**行 9** )。这个值是 IoU 公式的*分子*。

为了确定*分母*，我们需要导出预测边界框和真实边界框的面积(**第 13 行和第 14 行**)。

然后通过将交集面积(*分子*)除以两个边界框的并集面积(*分母*)，小心减去交集面积(否则交集面积将被加倍计算)，可以在**行 19** 上计算交集。

**第 22 行**返回 IoU 结果。

### **实现我们的对象检测数据集构建器脚本**

在我们可以创建我们的 R-CNN 对象检测器之前，**我们首先需要构建我们的数据集，**完成我们今天教程的六个步骤列表中的**步骤#1** 。

**我们的** `build_dataset.py` **剧本将:**

*   1.接受我们的输入`raccoons`数据集
*   2.遍历数据集中的所有图像
    *   2a。加载给定的输入图像
    *   2b。加载并解析输入图像中任何浣熊的边界框坐标
*   3.对输入图像运行选择性搜索
*   4.使用 IoU 来确定选择性搜索的哪些区域提议*与地面实况边界框*充分重叠，哪些*没有*
*   5.将区域建议保存为重叠(包含浣熊)或不重叠(不包含浣熊)

一旦我们的数据集建立起来，我们将能够进行**步骤# 2**——**微调一个对象检测网络**。

现在，我们已经在较高的层次上理解了数据集构建器，让我们来实现它。打开`build_dataset.py`文件，按照以下步骤操作:

```py
# import the necessary packages
from pyimagesearch.iou import compute_iou
from pyimagesearch import config
from bs4 import BeautifulSoup
from imutils import paths
import cv2
import os
```

除了我们的 IoU 和配置设置(**行 2 和 3** )，这个脚本还需要 [Beautifulsoup](https://www.crummy.com/software/BeautifulSoup/) 、 [imutils](https://github.com/jrosebr1/imutils) 和 [OpenCV](https://opencv.org/) 。如果您遵循了上面的*“配置您的开发环境”*一节，那么您的系统已经拥有了所有这些工具。

现在我们的导入已经完成，让我们创建两个空目录并构建一个包含所有浣熊图像的列表:

```py
# loop over the output positive and negative directories
for dirPath in (config.POSITVE_PATH, config.NEGATIVE_PATH):
	# if the output directory does not exist yet, create it
	if not os.path.exists(dirPath):
		os.makedirs(dirPath)

# grab all image paths in the input images directory
imagePaths = list(paths.list_images(config.ORIG_IMAGES))

# initialize the total number of positive and negative images we have
# saved to disk so far
totalPositive = 0
totalNegative = 0
```

我们的正面和负面目录将很快包含我们的*浣熊*或*无浣熊*图像。**第 10-13 行**创建这些目录，如果它们还不存在的话。

```py
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# show a progress report
	print("[INFO] processing image {}/{}...".format(i + 1,
		len(imagePaths)))

	# extract the filename from the file path and use it to derive
	# the path to the XML annotation file
	filename = imagePath.split(os.path.sep)[-1]
	filename = filename[:filename.rfind(".")]
	annotPath = os.path.sep.join([config.ORIG_ANNOTS,
		"{}.xml".format(filename)])

	# load the annotation file, build the soup, and initialize our
	# list of ground-truth bounding boxes
	contents = open(annotPath).read()
	soup = BeautifulSoup(contents, "html.parser")
	gtBoxes = []

	# extract the image dimensions
	w = int(soup.find("width").string)
	h = int(soup.find("height").string)
```

```py
	# loop over all 'object' elements
	for o in soup.find_all("object"):
		# extract the label and bounding box coordinates
		label = o.find("name").string
		xMin = int(o.find("xmin").string)
		yMin = int(o.find("ymin").string)
		xMax = int(o.find("xmax").string)
		yMax = int(o.find("ymax").string)

		# truncate any bounding box coordinates that may fall
		# outside the boundaries of the image
		xMin = max(0, xMin)
		yMin = max(0, yMin)
		xMax = min(w, xMax)
		yMax = min(h, yMax)

		# update our list of ground-truth bounding boxes
		gtBoxes.append((xMin, yMin, xMax, yMax))
```

```py
	# load the input image from disk
	image = cv2.imread(imagePath)

	# run selective search on the image and initialize our list of
	# proposed boxes
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)
	ss.switchToSelectiveSearchFast()
	rects = ss.process()
	proposedRects= []

	# loop over the rectangles generated by selective search
	for (x, y, w, h) in rects:
		# convert our bounding boxes from (x, y, w, h) to (startX,
		# startY, startX, endY)
		proposedRects.append((x, y, x + w, y + h))
```

```py
	# initialize counters used to count the number of positive and
	# negative ROIs saved thus far
	positiveROIs = 0
	negativeROIs = 0

	# loop over the maximum number of region proposals
	for proposedRect in proposedRects[:config.MAX_PROPOSALS]:
		# unpack the proposed rectangle bounding box
		(propStartX, propStartY, propEndX, propEndY) = proposedRect

		# loop over the ground-truth bounding boxes
		for gtBox in gtBoxes:
			# compute the intersection over union between the two
			# boxes and unpack the ground-truth bounding box
			iou = compute_iou(gtBox, proposedRect)
			(gtStartX, gtStartY, gtEndX, gtEndY) = gtBox

			# initialize the ROI and output path
			roi = None
			outputPath = None
```

我们在**行 84 和 85** 上初始化这两个计数器。

从第**行第 88** 开始，我们循环通过选择性搜索生成的区域建议(直到我们定义的最大建议数)。在内部，我们:

```py
			# check to see if the IOU is greater than 70% *and* that
			# we have not hit our positive count limit
			if iou > 0.7 and positiveROIs <= config.MAX_POSITIVE:
				# extract the ROI and then derive the output path to
				# the positive instance
				roi = image[propStartY:propEndY, propStartX:propEndX]
				filename = "{}.png".format(totalPositive)
				outputPath = os.path.sep.join([config.POSITVE_PATH,
					filename])

				# increment the positive counters
				positiveROIs += 1
				totalPositive += 1
```

假设这个特定区域通过了检查，以查看我们是否有 IoU > 70% *和*我们还没有达到当前图像的正面例子的极限(**行 105** ，我们简单地:

```py
			# determine if the proposed bounding box falls *within*
			# the ground-truth bounding box
			fullOverlap = propStartX >= gtStartX
			fullOverlap = fullOverlap and propStartY >= gtStartY
			fullOverlap = fullOverlap and propEndX <= gtEndX
			fullOverlap = fullOverlap and propEndY <= gtEndY
```

```py
			# check to see if there is not full overlap *and* the IoU
			# is less than 5% *and* we have not hit our negative
			# count limit
			if not fullOverlap and iou < 0.05 and \
				negativeROIs <= config.MAX_NEGATIVE:
				# extract the ROI and then derive the output path to
				# the negative instance
				roi = image[propStartY:propEndY, propStartX:propEndX]
				filename = "{}.png".format(totalNegative)
				outputPath = os.path.sep.join([config.NEGATIVE_PATH,
					filename])

				# increment the negative counters
				negativeROIs += 1
				totalNegative += 1
```

在这里，我们的条件(**第 127 行和第 128 行**)检查是否满足以下所有条件:

1.  有*没有*完全重叠
2.  欠条足够小
3.  没有超过我们对当前图像的反面例子数量的限制

```py
			# check to see if both the ROI and output path are valid
			if roi is not None and outputPath is not None:
				# resize the ROI to the input dimensions of the CNN
				# that we'll be fine-tuning, then write the ROI to
				# disk
				roi = cv2.resize(roi, config.INPUT_DIMS,
					interpolation=cv2.INTER_CUBIC)
				cv2.imwrite(outputPath, roi)
```

### **准备用于对象检测的图像数据集**

我们现在准备为 R-CNN 对象检测构建我们的图像数据集。

如果您还没有，请使用本教程的 ***“下载”*** 部分下载源代码和示例图像数据集。

从那里，打开一个终端，并执行以下命令:

```py
$ time python build_dataset.py
[INFO] processing image 1/200...
[INFO] processing image 2/200...
[INFO] processing image 3/200...
...
[INFO] processing image 198/200...
[INFO] processing image 199/200...
[INFO] processing image 200/200...

real	5m42.453s
user	6m50.769s
sys     1m23.245s
```

```py
$ ls -l dataset/raccoon/*.png | wc -l
    1560
$ ls -l dataset/no_raccoon/*.png | wc -l
    2200
```

下面是这两个类的示例:

从**图 6 *(左)*** 可以看出，“无浣熊”类具有通过选择性搜索生成的样本图像块，这些样本图像块*没有*与任何浣熊地面真实边界框明显重叠。

然后，在**图 6 *(右)*** 上，我们有了我们的“浣熊”级图像。

您会注意到，这些图像中的一些彼此相似，在某些情况下几乎是重复的——这实际上是预期的行为。

请记住，选择性搜索试图识别图像中*可能*包含潜在对象的区域。

因此，选择性搜索在相似区域多次发射是完全可行的。

您可以选择保留这些区域(如我所做的那样)，或者添加额外的逻辑来过滤掉明显重叠的区域(我将这作为一个练习留给您)。

### **使用 Keras 和 TensorFlow 微调对象检测网络**

有了通过前面两个部分(**步骤#1** )创建的数据集，我们现在准备好微调分类 CNN 来识别这两个类(**步骤#2** )。

当我们将这个分类器与选择性搜索相结合时，我们将能够构建我们的 R-CNN 对象检测器。

出于本教程的目的，我选择了 ***微调***MobileNet V2 CNN，它是在 1000 级 [ImageNet 数据集](http://www.image-net.org/)上预先训练的。如果您不熟悉迁移学习和微调的概念，我建议您仔细阅读:

*   *[使用 Keras 的迁移学习和深度学习](https://pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/)* (确保至少从头至尾阅读*“两种类型的迁移学习:特征提取和微调”*部分)
*   *[【Keras 微调】深度学习](https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/)* (我强烈推荐完整阅读本教程)

```py
# import the necessary packages
from pyimagesearch import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
```

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 5
BS = 32
```

`--plot` [命令行参数](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/)定义了我们的精度/损失图的路径(**第 27-30 行**)。

然后，我们建立训练超参数，包括我们的初始学习率、训练时期数和批量大小(**第 34-36 行**)。

加载我们的数据集很简单，因为我们已经在**步骤#1** 中完成了所有的艰苦工作:

```py
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class labels
print("[INFO] loading images...")
imagePaths = list(paths.list_images(config.BASE_PATH))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=config.INPUT_DIMS)
	image = img_to_array(image)
	image = preprocess_input(image)

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)
```

```py
# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
```

```py
# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False
```

```py
# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)
```

```py
# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
```

```py
# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(config.MODEL_PATH, save_format="h5")

# serialize the label encoder to disk
print("[INFO] saving label encoder...")
f = open(config.ENCODER_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()
```

```py
# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
```

使用 matplotlib，我们绘制精度和损耗曲线以供检查(**第 144-154 行**)。我们将得到的图形导出到包含在`--plot`命令行参数中的路径。

### **用 Keras 和 TensorFlow 训练我们的 R-CNN 目标检测网络**

我们现在准备微调我们的手机，这样我们就可以创建一个 R-CNN 对象检测器！

如果您还没有，请到本教程的 ***“下载”*** 部分下载源代码和样本数据集。

从那里，打开一个终端，并执行以下命令:

```py
$ time python fine_tune_rcnn.py
[INFO] loading images...
[INFO] compiling model...
[INFO] training head...
Train for 94 steps, validate on 752 samples
Train for 94 steps, validate on 752 samples
Epoch 1/5
94/94 [==============================] - 77s 817ms/step - loss: 0.3072 - accuracy: 0.8647 - val_loss: 0.1015 - val_accuracy: 0.9728
Epoch 2/5
94/94 [==============================] - 74s 789ms/step - loss: 0.1083 - accuracy: 0.9641 - val_loss: 0.0534 - val_accuracy: 0.9837
Epoch 3/5
94/94 [==============================] - 71s 756ms/step - loss: 0.0774 - accuracy: 0.9784 - val_loss: 0.0433 - val_accuracy: 0.9864
Epoch 4/5
94/94 [==============================] - 74s 784ms/step - loss: 0.0624 - accuracy: 0.9781 - val_loss: 0.0367 - val_accuracy: 0.9878
Epoch 5/5
94/94 [==============================] - 74s 791ms/step - loss: 0.0590 - accuracy: 0.9801 - val_loss: 0.0340 - val_accuracy: 0.9891
[INFO] evaluating network...
              precision    recall  f1-score   support

  no_raccoon       1.00      0.98      0.99       440
     raccoon       0.97      1.00      0.99       312

    accuracy                           0.99       752
   macro avg       0.99      0.99      0.99       752
weighted avg       0.99      0.99      0.99       752

[INFO] saving mask detector model...
[INFO] saving label encoder...

real	6m37.851s
user	31m43.701s
sys     33m53.058s
```

在我的 3Ghz 英特尔至强 W 处理器上微调 MobileNet 花费了大约 6m30 秒，正如您所看到的，我们获得了大约 99%的准确率。

正如我们的训练图所示，几乎没有过度拟合的迹象:

随着我们的 MobileNet 模型针对浣熊预测进行了微调，我们已经准备好将所有的部分放在一起，并创建我们的 R-CNN 对象检测管道！

### **将碎片放在一起:实现我们的 R-CNN 对象检测推理脚本**

到目前为止，我们已经完成了:

*   **步骤#1:** 使用选择性搜索建立对象检测数据集
*   **步骤#2:** 微调用于对象检测的分类网络(最初在 ImageNet 上训练)

在这一点上，我们将把我们训练好的模型用于在新图像上执行对象检测推断。

完成我们的对象检测推理脚本需要**步骤# 3–步骤#6** 。现在让我们回顾一下这些步骤:

*   **步骤#3:** 创建一个对象检测推理脚本，该脚本利用选择性搜索来建议*可能*包含我们想要检测的对象的区域
*   **步骤#4:** 使用我们的微调网络对通过选择性搜索提出的每个区域进行分类
*   **步骤#5:** 应用非最大值抑制来抑制弱的重叠边界框
*   **步骤#6:** 返回最终的物体检测结果

我们将进一步执行**步骤#6** 并显示结果，这样我们就可以直观地验证我们的系统正在工作。

现在让我们实现 R-CNN 对象检测管道—打开一个新文件，将其命名为`detect_object_rcnn.py`，并插入以下代码:

```py
# import the necessary packages
from pyimagesearch.nms import non_max_suppression
from pyimagesearch import config
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
```

```py
# load the our fine-tuned model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())

# load the input image from disk
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)

# run selective search on the image to generate bounding box proposal
# regions
print("[INFO] running selective search...")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()
```

```py
# initialize the list of region proposals that we'll be classifying
# along with their associated bounding boxes
proposals = []
boxes = []

# loop over the region proposal bounding box coordinates generated by
# running selective search
for (x, y, w, h) in rects[:config.MAX_PROPOSALS_INFER]:
	# extract the region from the input image, convert it from BGR to
	# RGB channel ordering, and then resize it to the required input
	# dimensions of our trained CNN
	roi = image[y:y + h, x:x + w]
	roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	roi = cv2.resize(roi, config.INPUT_DIMS,
		interpolation=cv2.INTER_CUBIC)

	# further preprocess the ROI
	roi = img_to_array(roi)
	roi = preprocess_input(roi)

	# update our proposals and bounding boxes lists
	proposals.append(roi)
	boxes.append((x, y, x + w, y + h))
```

```py
# convert the proposals and bounding boxes into NumPy arrays
proposals = np.array(proposals, dtype="float32")
boxes = np.array(boxes, dtype="int32")
print("[INFO] proposal shape: {}".format(proposals.shape))

# classify each of the proposal ROIs using fine-tuned model
print("[INFO] classifying proposals...")
proba = model.predict(proposals)
```

```py
# find the index of all predictions that are positive for the
# "raccoon" class
print("[INFO] applying NMS...")
labels = lb.classes_[np.argmax(proba, axis=1)]
idxs = np.where(labels == "raccoon")[0]

# use the indexes to extract all bounding boxes and associated class
# label probabilities associated with the "raccoon" class
boxes = boxes[idxs]
proba = proba[idxs][:, 1]

# further filter indexes by enforcing a minimum prediction
# probability be met
idxs = np.where(proba >= config.MIN_PROBA)
boxes = boxes[idxs]
proba = proba[idxs]
```

```py
# clone the original image so that we can draw on it
clone = image.copy()

# loop over the bounding boxes and associated probabilities
for (box, prob) in zip(boxes, proba):
	# draw the bounding box, label, and probability on the image
	(startX, startY, endX, endY) = box
	cv2.rectangle(clone, (startX, startY), (endX, endY),
		(0, 255, 0), 2)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	text= "Raccoon: {:.2f}%".format(prob * 100)
	cv2.putText(clone, text, (startX, y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# show the output after *before* running NMS
cv2.imshow("Before NMS", clone)
```

从那里，我们在 NMS 可视化(**线 101** )之前显示*。*

让我们应用 NMS，看看结果如何比较:

```py
# run non-maxima suppression on the bounding boxes
boxIdxs = non_max_suppression(boxes, proba)

# loop over the bounding box indexes
for i in boxIdxs:
	# draw the bounding box, label, and probability on the image
	(startX, startY, endX, endY) = boxes[i]
	cv2.rectangle(image, (startX, startY), (endX, endY),
		(0, 255, 0), 2)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	text= "Raccoon: {:.2f}%".format(proba[i] * 100)
	cv2.putText(image, text, (startX, y),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

# show the output image *after* running NMS
cv2.imshow("After NMS", image)
cv2.waitKey(0)
```

我们通过**线 104，**应用非最大值抑制(NMS ),有效地消除对象周围的重叠矩形。

从那里，**行 107-119** 绘制边界框、标签和概率，并在 NMS 结果后显示*，直到按下一个键。*

使用 TensorFlow/Keras、OpenCV 和 Python 实现您的基本 R-CNN 对象检测脚本非常棒。

### **使用 Keras 和 TensorFlow 的 R-CNN 物体检测结果**

至此，我们已经使用 Keras、TensorFlow 和 OpenCV 完全实现了一个基本的 R-CNN 对象检测管道。

你准备好看它的行动了吗？

首先使用本教程的 ***“下载”*** 部分下载源代码、示例数据集和预训练的 R-CNN 检测器。

从那里，您可以执行以下命令:

```py
$ python detect_object_rcnn.py --image images/raccoon_01.jpg
[INFO] loading model and label binarizer...
[INFO] running selective search...
[INFO] proposal shape: (200, 224, 224, 3)
[INFO] classifying proposals...
[INFO] applying NMS...
```

在这里，您可以看到在应用我们的 R-CNN 对象检测器后发现了两个浣熊包围盒:

通过应用非最大值抑制，我们可以抑制较弱的一个，留下一个正确的边界框:

让我们尝试另一个图像:

```py
$ python detect_object_rcnn.py --image images/raccoon_02.jpg
[INFO] loading model and label binarizer...
[INFO] running selective search...
[INFO] proposal shape: (200, 224, 224, 3)
[INFO] classifying proposals...
[INFO] applying NMS...
```

同样，这里我们有两个边界框:

对我们的 R-CNN 对象检测输出应用非最大值抑制留给我们最终的对象检测:

让我们看最后一个例子:

```py
$ python detect_object_rcnn.py --image images/raccoon_03.jpg
[INFO] loading model and label binarizer...
[INFO] running selective search...
[INFO] proposal shape: (200, 224, 224, 3)
[INFO] classifying proposals...
[INFO] applying NMS...
```

如您所见，只检测到一个边界框，因此 NMS 之前/之后的输出是相同的。

现在你知道了，构建一个简单的 R-CNN 物体探测器并不像看起来那么难！

我们能够使用 Keras、TensorFlow 和 OpenCV 在**仅 427 行代码、*包括*评论中构建一个简化的 R-CNN 对象检测管道！**

我希望当您开始构建自己的基本对象检测器时，可以使用这个管道。

## **总结**

在本教程中，您学习了如何使用 Keras、TensorFlow 和深度学习实现基本的 R-CNN 对象检测器。

我们的 R-CNN 对象检测器是 Girshick 等人在他们开创性的对象检测论文 *[的初始实验中可能已经创建的内容的精简、基本版本，丰富的特征层次用于精确的对象检测和语义分割。](https://arxiv.org/abs/1311.2524)*

我们实施的 R-CNN 对象检测管道是一个 6 步流程，包括:

1.  **步骤#1:** 使用选择性搜索建立对象检测数据集
2.  **步骤#2:** 微调用于对象检测的分类网络(最初在 ImageNet 上训练)
3.  **步骤#3:** 创建对象检测推理脚本，该脚本利用选择性搜索来建议*可能*包含我们想要检测的对象的区域
4.  **步骤#4:** 使用我们的微调网络对通过选择性搜索提出的每个区域进行分类
5.  **步骤#5:** 应用非最大值抑制来抑制弱的重叠边界框
6.  **步骤#6:** 返回最终的物体检测结果

总的来说，我们的 R-CNN 物体检测器表现相当不错！

我希望您可以使用这个实现作为您自己的对象检测项目的起点。

如果你想了解更多关于实现自己的定制深度学习对象检测器的信息，请务必参考我的书， *[使用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)* ，我在其中详细介绍了对象检测。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****