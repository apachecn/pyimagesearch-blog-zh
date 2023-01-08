# 用于对象检测的并集交集(IoU)

> 原文：<https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/>

最后更新于 2022 年 4 月 30 日

* * *

## **目录**

并集上的交集(IoU)用于评估对象检测的性能，方法是将基础真实边界框与预测边界框进行比较，IoU 是本教程的主题。

今天这篇博文的灵感来自我收到的一封电子邮件，来自罗切斯特大学的学生杰森。

Jason 有兴趣在他的最后一年项目中使用 [HOG +线性 SVM 框架](https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)构建一个定制的物体检测器。他非常了解构建目标探测器所需的步骤— *，但他不确定一旦训练完成，如何**评估他的探测器*** *的准确性。*

他的教授提到他应该用 ***交集超过并集【IoU】***的方法来进行评估，但杰森不确定如何实施。

我通过电子邮件帮助 Jason 解决了问题:

1.  描述什么是并集上的交集。
2.  解释*为什么*我们使用并集上的交集来评估对象检测器。
3.  向他提供一些来自我个人库的*示例 Python 代码，以在边界框上执行交集并运算。*

我的电子邮件真的帮助杰森完成了他最后一年的项目，我相信他会顺利通过。

考虑到这一点，我决定把我对 Jason 的回应变成一篇真正的博文，希望对你也有帮助。

***要了解如何使用交集超过并集评估指标来评估您自己的自定义对象检测器，请继续阅读。***

*   【2021 年 7 月更新:添加了关于 Union 实现的替代交集的部分，包括在训练深度神经网络对象检测器时可用作损失函数的 IoU 方法。
*   **更新 2022 年 4 月:**增加了 TOC，并把帖子链接到一个新的交集 over Union 教程。
*   【2022 年 12 月更新:删除了数据集的链接，因为该数据集不再公开，并刷新了内容。

* * *

## [用于物体检测的并集上的交集](#TOC)

在这篇博文的剩余部分，我将解释*什么是*联合评估指标的交集，以及*为什么*我们使用它。

我还将提供 Union 上交集的 Python 实现，您可以在评估自己的自定义对象检测器时使用它。

最后，我们将查看将交集运算评估指标应用于一组*地面实况*和*预测*边界框的一些*实际结果*。

* * *

### 什么是交集大于并集？

并集上的交集是一种评估度量，用于测量特定数据集上的对象检测器的准确性。我们经常在物体检测挑战中看到这种评估指标，例如广受欢迎的 [PASCAL VOC 挑战](http://host.robots.ox.ac.uk/pascal/VOC/)。

你通常会发现用于评估 [HOG +线性 SVM 物体检测器](https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)和卷积神经网络检测器(R-CNN，fast R-CNN，YOLO 等)性能的交集。);然而，请记住，用于生成预测的*实际算法**并不重要。***

并集上的交集仅仅是一个*评估度量*。任何提供预测边界框作为输出的算法都可以使用 IoU 进行评估。

更正式地说，为了将交集应用于并集来评估(任意)对象检测器，我们需要:

1.  *地面真实边界框*(即来自测试集的手动标记边界框，其指定了我们的对象在图像中的位置*)。*
2.  来自我们模型的*预测边界框*。

只要我们有这两组边界框，我们就可以在并集上应用交集。

下面，我提供了一个真实边界框与预测边界框的可视化示例:

在上图中，我们可以看到我们的对象检测器已经检测到图像中存在停车标志。

*预测的*包围盒以*红色*绘制，而*地面真实*(即手绘)包围盒以绿色绘制。

因此，计算并集上的交集可以通过以下方式确定:

检查这个等式，你可以看到交集除以并集仅仅是一个比率。

在分子中，我们计算在*预测*边界框和*地面真实*边界框之间的重叠区域的 ***。***

分母是联合 的 ***区域，或者更简单地说，由*和*预测边界框和真实边界框包围的区域。***

将重叠面积除以并集面积，得到我们的最终分数— *并集上的交集。*

* * *

### 你从哪里得到这些真实的例子？

在我们走得太远之前，您可能想知道地面真相的例子从何而来。我之前提到过这些图像是“手动标记的”，但这到底是什么意思呢？

你看，在训练自己的物体检测器(比如 [HOG +线性 SVM 方法](https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/))的时候，你需要一个数据集。该数据集应(至少)分为两组:

1.  一个*训练集*用于训练你的物体探测器。
2.  一个*测试装置*，用于评估您的物体探测器。

您可能还有一个*验证集*，用于[调整您的模型](https://pyimagesearch.com/2016/08/15/how-to-tune-hyperparameters-with-python-and-scikit-learn/)的超参数。

训练和测试集将包括:

1.  真实的图像本身。
2.  与图像中的对象相关联的*边界框*。边界框就是图像中物体的 *(x，y)*-坐标。

训练集和测试集的边界框由*手工标注*，因此我们称之为“地面真相”。

您的目标是获取训练图像+包围盒，构造一个对象检测器，然后在测试集上评估其性能。

**超过联合分数 *> 0.5* 的交集通常被认为是“好的”预测。**

* * *

### 为什么我们使用交集而不是并集？

如果你以前在职业生涯中执行过任何机器学习，特别是分类，你可能会习惯于*预测类别标签*，其中你的模型输出单个标签，要么是*正确的*要么是*不正确的。*

这种类型的二进制分类使得计算精度简单明了；然而，对于对象检测来说，这并不简单。

在所有现实中，*极不可能*我们预测的边界框的 *(x，y)*-坐标将会 ***精确地匹配****(x，y)*-地面真实边界框的坐标。

由于我们的模型的不同参数(图像金字塔比例、滑动窗口大小、特征提取方法等)。)，预测边界框和真实边界框之间的完全和完全匹配是不现实的。

正因为如此，我们需要定义一个评估标准，让*奖励*预测的边界框与实际情况有很大的重叠:

在上图中，我已经包括了优、劣交集的例子。

如您所见，与地面实况边界框重叠严重的预测边界框比重叠较少的预测边界框得分更高。这使得交集/并集成为评估自定义对象检测器的优秀指标。

我们并不关心 *(x，y)*-坐标的*精确*匹配，但是我们确实想要确保我们预测的边界框尽可能地匹配——并集上的交集能够考虑到这一点。

* * *

### [在 Python 中实现 Union 上的交集](#TOC)

既然我们已经理解了什么是并上交集以及为什么我们使用它来评估对象检测模型，那么让我们继续用 Python 来实现它。

不过，在我们开始编写任何代码之前，我想提供我们将使用的五个示例图像:

这些图像是 CALTECH-101 数据集的一部分，用于*图像分类*和*物体检测。*

该数据集是公开可用的，但截至 2022 年 12 月，它不再公开。

在 ***[PyImageSearch 大师课程](https://pyimagesearch.com/pyimagesearch-gurus/)*** 中，我演示了如何使用 HOG +线性 SVM 框架训练一个自定义对象检测器来检测图像中汽车的存在。

我从下面的自定义对象检测器中提供了真实边界框(绿色)和预测边界框(红色)的可视化效果:

给定这些边界框，我们的任务是定义联合上的交集度量，该度量可用于评估我们的预测有多“好(或坏)”。

也就是说，打开一个新文件，命名为`intersection_over_union.py`，让我们开始编码:

```py
# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

```

我们从导入所需的 Python 包开始。然后我们定义一个`Detection`对象，它将存储三个属性:

*   `image_path`:驻留在磁盘上的输入图像的路径。
*   `gt`:地面真实包围盒。
*   从我们的模型预测的边界框。

正如我们将在这个例子的后面看到的，我已经从我们各自的五个图像中获得了预测的边界框，并将它们硬编码到这个脚本中，以保持这个例子的简短和简洁。

对于 HOG +线性 SVM 物体检测框架的完整回顾，[请参考这篇博文](https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)。如果你有兴趣学习更多关于从头开始训练你自己的定制物体探测器的知识，一定要看看[的 PyImageSearch 大师课程](https://pyimagesearch.com/pyimagesearch-gurus/)。

让我们继续定义`bb_intersection_over_union`函数，顾名思义，它负责计算两个边界框之间的交集:

```py
def bb_intersection_over_union(boxA, boxB):
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
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

```

这个方法需要两个参数:`boxA`和`boxB`，它们被假定为我们的地面实况和预测边界框(这些参数被提供给`bb_intersection_over_union`的实际*顺序*并不重要)。

**第 11-14 行**确定了 *(x，y)*-相交矩形的坐标，然后我们用它来计算相交的面积(**第 17 行**)。

现在,`interArea`变量代表交集并集计算中的*分子*。

为了计算分母，我们首先需要导出预测边界框和真实边界框的面积(**行 21 和 22** )。

然后可以在第 27 行**上通过将交集面积除以两个边界框的并集面积来计算交集，注意从分母中减去交集面积(否则交集面积将被加倍计算)。**

最后，将 Union score 上的交集返回给第 30 行上的调用函数。

既然我们的联合交集方法已经完成，我们需要为我们的五个示例图像定义地面实况和预测边界框坐标:

```py
# define the list of example detections
examples = [
	Detection("image_0002.jpg", [39, 63, 203, 112], [54, 66, 198, 114]),
	Detection("image_0016.jpg", [49, 75, 203, 125], [42, 78, 186, 126]),
	Detection("image_0075.jpg", [31, 69, 201, 125], [18, 63, 235, 135]),
	Detection("image_0090.jpg", [50, 72, 197, 121], [54, 72, 198, 120]),
	Detection("image_0120.jpg", [35, 51, 196, 110], [36, 60, 180, 108])]

```

正如我上面提到的，为了保持这个例子简短，我从我的 HOG +线性 SVM 检测器中手动获得了预测的边界框坐标。这些预测的边界框(和相应的真实边界框)然后被*硬编码*到这个脚本中。

关于我如何训练这个精确物体探测器的更多信息，请参考 PyImageSearch 大师课程。

我们现在准备评估我们的预测:

```py
# loop over the example detections
for detection in examples:
	# load the image
	image = cv2.imread(detection.image_path)

	# draw the ground-truth bounding box along with the predicted
	# bounding box
	cv2.rectangle(image, tuple(detection.gt[:2]), 
		tuple(detection.gt[2:]), (0, 255, 0), 2)
	cv2.rectangle(image, tuple(detection.pred[:2]), 
		tuple(detection.pred[2:]), (0, 0, 255), 2)

	# compute the intersection over union and display it
	iou = bb_intersection_over_union(detection.gt, detection.pred)
	cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	print("{}: {:.4f}".format(detection.image_path, iou))

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)

```

在**的第 41 行**，我们开始循环遍历我们的每个`examples`(它们是`Detection`对象)。

对于它们中的每一个，我们在**行 43** 上从磁盘加载各自的`image`，然后绘制绿色的真实边界框(**行 47 和 48** )，接着是红色的预测边界框(**行 49 和 50** )。

通过传入地面实况和预测边界框，在**行 53** 上计算联合度量上的实际交集。

然后，我们在控制台后面的`image`上写交集并值。

最后，输出图像通过第 59 和 60 行显示在我们的屏幕上。

* * *

### [用并集上的交集将预测检测与实际情况进行比较](#TOC)

要查看联合度量的交集，请确保您已经使用本教程底部的 ***“下载”*** 部分下载了源代码+示例图片到这篇博客文章中。

解压缩归档文件后，执行以下命令:

```py
$ python intersection_over_union.py

```

我们的第一个示例图像在联合分数上有一个交集 *0.7980* ，表明两个边界框之间有明显的重叠:

下图也是如此，其交集超过联合分数 *0.7899* :

请注意实际边界框(绿色)比预测边界框(红色)宽。这是因为我们的对象检测器是使用 HOG +线性 SVM 框架定义的，这要求我们指定固定大小的滑动窗口(更不用说图像金字塔比例和 HOG 参数本身)。

真实边界框自然会与预测的边界框具有稍微不同的纵横比，但是如果联合分数的交集是 *> 0.5* 也没关系——正如我们所看到的，这仍然是一个很好的预测。

下一个示例演示了一个稍微“不太好”的预测，其中我们的预测边界框比地面实况边界框“紧密”得多:

其原因是因为我们的 HOG +线性 SVM 检测器可能无法在图像金字塔的较低层“找到”汽车，而是在图像小得多的金字塔顶部附近发射。

以下示例是一个*非常好的*检测，交集超过联合分数 *0.9472* :

请注意预测的边界框是如何与真实边界框几乎完美重叠的。

下面是计算并集交集的最后一个例子:

* * *

### 【Union 实现上的可选交集

本教程提供了 IoU 的 Python 和 NumPy 实现。但是，对于您的特定应用程序和项目，IoU 的其他实现可能更好。

例如，如果你正在使用 TensorFlow、Keras 或 PyTorch 等流行的库/框架训练深度学习模型，那么使用你的深度学习框架实现 IoU*应该会*提高算法的速度。

下面的列表提供了我建议的交集/并集的替代实现，包括在训练深度神经网络对象检测器时可以用作损失/度量函数的实现:

*   [TensorFlow 的 MeanIoU 函数](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU)，其计算对象检测结果样本的并集上的平均交集。
*   [TensorFlow 的 GIoULoss 损失度量](https://www.tensorflow.org/addons/api_docs/python/tfa/losses/GIoULoss)，最早由 Rezatofighi 等人在 [*的《并集上的广义交集:包围盒回归的一个度量和一个损失*](https://giou.stanford.edu/GIoU.pdf) 中引入，就像你训练一个神经网络使均方误差、交叉熵等最小化一样。这种方法充当插入替换损失函数，潜在地导致更高的对象检测精度。
*   IoU 的 PyTorch 实现(我没有测试或使用过)，但似乎对 PyTorch 社区有所帮助。
*   我们有一个很棒的使用 COCO 评估器的[平均精度(mAP)](https://pyimg.co/nwoka)教程，将带您了解如何使用交集/并集来评估 YOLO 性能。了解平均精度(mAP)的理论概念，并使用黄金标准 COCO 评估器评估 YOLOv4 检测器。

当然，您可以随时将我的 IoU Python/NumPy 实现转换成您自己的库、语言等。

黑客快乐！

* * *

## [总结](#TOC)

在这篇博文中，我讨论了用于评估对象检测器的 Union 上的*交集度量。该指标可用于评估*任何*物体检测器，前提是(1)模型为图像中的物体生成预测的 *(x，y)*-坐标【即边界框】，以及(2)您拥有数据集的真实边界框。*

通常，您会看到该指标用于评估 HOG +线性 SVM 和基于 CNN 的物体检测器。

要了解更多关于训练你自己的自定义对象检测器的信息，请参考这篇关于 [HOG +线性 SVM 框架](https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)的博客文章，以及 [***PyImageSearch 大师课程***](https://pyimagesearch.com/pyimagesearch-gurus/) ，在那里我演示了如何从头开始实现自定义对象检测器。如果你想深入研究，可以考虑通过我们的免费课程[学习计算机视觉](https://pyimagesearch.com/)。

***最后，在你离开之前，请务必在下面的表格中输入你的电子邮件地址，以便在将来发布 PyImageSearch 博客文章时得到通知——你不会想错过它们的！***