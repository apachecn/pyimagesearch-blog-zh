# 多栏表格 OCR

> 原文：<https://pyimagesearch.com/2022/02/28/multi-column-table-ocr/>

在本教程中，您将:

1.  发现一种将行和列关联在一起的技术
2.  了解如何检测图像中的文本/数据表
3.  从图像中提取检测到的表格
4.  对表格中的文本进行 OCR
5.  应用层次凝聚聚类(HAC)来关联行和列
6.  从 OCR 数据中构建一个熊猫

本教程是关于使用 Python 进行 OCR 的 4 部分系列的第一部分:

1.  *多栏表格 OCR* (本教程)
2.  *OpenCV 快速傅立叶变换(FFT)用于图像和视频流中的模糊检测*
3.  *对视频流进行光学字符识别*
4.  *使用 OpenCV 和 GPU 提高文本检测速度*

**学习如何 OCR 多栏表格，** ***继续阅读。***

## **多栏表格 OCR**

也许光学字符识别(OCR)更具挑战性的应用之一是如何成功地 OCR 多列数据(例如，电子表格、表格等)。).

从表面上看，对表格进行 OCR 似乎是一个更容易的问题，对吗？然而，考虑到文档有一个*保证结构*，难道我们不应该利用这个*先验*知识，然后对表中的每一列进行 OCR 吗？

大多数情况下，是的，会是这样。但不幸的是，我们有几个问题要解决:

1.  Tesseract 不太擅长多列 OCR，*尤其是*如果你的图像有噪声。
2.  您可能需要先*检测*图像中的表格，然后才能对其进行 OCR。
3.  你的 OCR 引擎(宇宙魔方，基于云的，等等。)可以正确地对文本进行 OCR，但是不能将文本关联到列/行中。

因此，尽管对多列数据进行 OCR 可能看起来是一项简单的任务，但它要困难得多，因为我们可能需要负责将文本关联到列和行中——正如您将看到的，这确实是我们实现中最复杂的部分。

好消息是，虽然对多列数据进行 OCR 肯定比其他 OCR 任务要求更高，但只要您将正确的算法和技术引入项目，这并不是一个“难题”。

在本教程中，您将学习一些 OCR 多列数据的提示和技巧，最重要的是，将文本的行/列关联在一起。

### **对多列数据进行光学字符识别**

在本教程的第一部分，我们将讨论我们的多列 OCR 算法的基本过程。这就是对多列数据进行 OCR 的*精确*算法。这是一个很好的起点，我们建议您在需要对表格进行 OCR 时使用它。

从那里，我们将回顾我们项目的目录结构。我们还将安装本教程所需的任何额外的 Python 包。

在我们的开发环境完全配置好之后，我们可以继续我们的实现了。我们将在这里花大部分时间讲述多列 OCR 算法的细节和内部工作原理。

我们将通过将我们的 Python 实现应用于以下各项来结束本课:

1.  检测图像中的文本表格
2.  提取表格
3.  对表格进行 OCR
4.  从表中构建一个 Pandas `DataFrame`来处理它，查询它，等等。

### **我们的多列 OCR 算法**

我们的多列 OCR 算法是一个多步骤的过程。首先，我们需要接受一个包含表格、电子表格等的输入图像。(**图**、 **1** 、*左*)。给定这个图像，然后我们需要提取表格本身(*右*)。

一旦我们有了表格，我们就可以应用 OCR 和文本本地化来为文本边界框生成 *(x，y)*-坐标。**获得这些包围盒坐标是至关重要的。**

为了将多列文本关联在一起，我们需要根据它们的起始 *x* 坐标对文本进行分组。

为什么开始 *x* 坐标？嗯，记住表格、电子表格等的结构。每一列的文本将具有几乎相同的起始*x*-坐标，因为它们属于*相同的*列(现在花一点时间来说服自己该陈述是正确的)。

因此，我们可以利用这些知识，然后用近似相同的 *x* 坐标将文本分组。

但是问题仍然存在，我们如何进行实际的分组？

答案是使用一种叫做层次凝聚聚类(HAC)的特殊聚类算法。如果你以前上过数据科学、机器学习或文本挖掘课程，那么你可能以前遇到过 HAC。

对 HAC 算法的全面回顾超出了本教程的范围。然而，总的想法是我们采用“自下而上”的方法，从我们的初始数据点(即文本边界框的 *x* 坐标)开始，作为单独的集群，每个集群仅包含一个观察值。

然后，我们计算坐标之间的距离，并开始以距离 *< T* 对观测值进行分组，其中 *T* 是预定义的阈值。

在 HAC 的每次迭代中，我们选择具有最小距离的两个聚类并将它们合并，同样，假设满足距离阈值要求。

我们继续聚类，直到不能形成其他聚类，通常是因为没有两个聚类落在我们的阈值要求内。

在多列 OCR 的情况下，将 HAC 应用于*x*-坐标值会产生具有相同或接近相同的*x*-坐标的聚类。由于我们假设属于同一列的文本将具有相似/接近相同的 *x* 坐标，我们可以将列关联在一起。

虽然这种多列 OCR 技术看起来很复杂，但它很简单，*尤其是*，因为我们将能够利用 scikit-learn 库中的 [`AgglomerativeClustering`实现。](http://pyimg.co/y3bic)

也就是说，如果您有兴趣了解更多关于机器学习和数据科学技术的知识，手动实现聚集聚类是一个很好的练习。在我的职业生涯中，我曾经不得不实施 HAC 3-4 次，主要是在我上大学的时候。

如果你想了解更多关于 HAC 的知识，我推荐 Cory Maklin 的 [scikit-learn 文档](http://pyimg.co/y3bic)和[优秀文章](http://pyimg.co/glaxm)。

对于更具理论性和数学动机的凝聚聚类处理，包括一般的聚类算法，我强烈推荐 Witten 等人的 [*【数据挖掘:实用的机器学习工具和技术】*(2011)](https://doi.org/https://doi.org/10.1016/B978-0-12-374856-0.00023-7)。

### **配置您的开发环境**

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果您需要帮助配置 OpenCV 的开发环境，我们*强烈推荐*阅读我们的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让您启动并运行。

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

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

接下来，让我们回顾一下我们的项目目录结构:

```py
|-- michael_jordan_stats.png
|-- multi_column_ocr.py
|-- results.csv
```

正如您所看到的，我们的项目目录结构非常简单——但是不要让这种简单性欺骗了您！

我们的`multi_column_ocr.py`脚本将接受输入图像`michael_jordan_stats.png`，检测数据表，提取它，然后通过 OCR 将它与行/列关联起来。

作为参考，我们的示例图像是迈克尔·乔丹棒球卡(**图** **3** )的扫描图，当时他在父亲去世后暂停篮球一年去打棒球。

我们的 Python 脚本可以对表格进行 OCR，解析出他的统计数据，然后将它们作为经过 OCR 处理的文本输出为 CSV 文件(`results.csv`)。

### **安装所需的软件包**

我们的 Python 脚本将向我们的终端显示一个格式良好的 OCR 文本表。我们仍然需要利用 [`tabulate` Python 包](https://pypi.org/project/tabulate/)来生成这个格式化的表格。

您可以使用以下命令安装`tabulate`:

```py
$ workon your_env_name # optional
$ pip install tabulate
```

如果您正在使用 Python 虚拟环境，不要忘记在安装`tabulate`之前使用`workon`命令(或等效命令)访问您的虚拟环境*(否则，您的`import`命令将失败)。*

同样，`tabulate`包仅用于*显示目的，*和*不会影响我们实际的多列 OCR 算法。如果你不想安装`tabulate`，那也没关系。您只需要在我们的 Python 脚本中注释掉利用它的 2-3 行代码。*

### **实现多列 OCR**

我们现在准备实现多列 OCR！打开项目目录结构中的`multi_column_ocr.py`文件，让我们开始工作:

```py
# import the necessary packages
from sklearn.cluster import AgglomerativeClustering
from pytesseract import Output
from tabulate import tabulate
import pandas as pd
import numpy as np
import pytesseract
import argparse
import imutils
import cv2
```

我们从导入所需的 Python 包开始。我们有几个以前没有(或者至少不经常)使用过的包，所以让我们回顾一下重要的包。

首先，我们有来自 scikit-learn 的`AgglomerativeClustering`实现。正如我前面所讨论的，我们将使用 HAC 将文本聚集成列。这个实现将允许我们这样做。

在执行 OCR 之后,`tabulate`包将允许我们在终端上打印一个格式良好的数据表。这个包是可选的，所以如果您不想安装它，只需在我们的实现中稍后注释掉`import`行和**行 178** 。

然后我们有了`pandas`库，这在数据科学家中很常见。在本课中，我们将使用`pandas`,因为它能够轻松地构建和管理表格数据。

其余的导入对您来说应该很熟悉，因为我们在整个课程中多次使用过它们。

让我们继续我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-o", "--output", required=True,
	help="path to output CSV file")
ap.add_argument("-c", "--min-conf", type=int, default=0,
	help="minimum confidence value to filter weak text detection")
ap.add_argument("-d", "--dist-thresh", type=float, default=25.0,
	help="distance threshold cutoff for clustering")
ap.add_argument("-s", "--min-size", type=int, default=2,
	help="minimum cluster size (i.e., # of entries in column)")
args = vars(ap.parse_args())
```

如你所见，我们有几个命令行参数。让我们逐一讨论一下:

*   `--image`:包含表格、电子表格等的输入图像的路径。，我们想要检测和 OCR
*   `--output`:包含我们提取的列数据的输出 CSV 文件的路径
*   `--min-conf`:用于过滤弱文本检测
*   `--dist-thresh`:应用 HAC 时的距离阈值截止(以像素为单位)；您可能需要为影像和数据集调整该值
*   `--min-size`:一个簇中被认为是一列的最小数据点数

这里最重要的命令行参数是`--dist-thresh`和`--min-size`。

当应用 HAC 时，我们需要使用距离阈值截止。如果您允许聚类无限期地继续，HAC 将在每次迭代中继续聚类，直到您最终得到一个包含所有数据点的聚类。

相反，当没有两个聚类的距离小于阈值时，应用距离阈值来*停止聚类过程*。

出于您的目的，您需要检查您正在处理的图像数据。如果你的表格数据在每行之间有大量的空白，那么你可能需要增加*`--dist-thresh`(**图 4** ，*左*)。*

 *否则，如果每行之间的空白减少，`--dist-thresh`会相应地*减少*(**图 4** ，*右*)。

**正确设置`--dist-thresh`是*最重要的*要对多列数据进行光学字符识别，一定要尝试不同的值。**

`--min-size`命令行参数也很重要。在我们的聚类算法的每次迭代中，HAC 将检查两个聚类，每个聚类可能包含*多个*数据点或仅仅一个*单个*数据点。如果两个簇之间的距离小于`--dist-thresh`，HAC 将合并它们。

然而，总会有异常值，远离表格的文本片段，或者只是图像中的噪声。如果 Tesseract 检测到该文本，那么 HAC 将尝试对其进行聚类。但是有没有一种方法可以防止这些集群出现在我们的结果中呢？

一种简单的方法是在 HAC 完成后检查给定集群中文本数据点的数量。

在这种情况下，我们设置了`--min-size=2`，这意味着如果一个聚类中有`≤2`个数据点，我们会将其视为异常值并忽略它。你可能需要为你的应用程序调整这个变量，但是我建议首先调整`--dist-thresh`。

考虑到我们的命令行参数，让我们开始我们的图像处理管道:

```py
# set a seed for our random number generator
np.random.seed(42)

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

**第 27 行**设置我们的伪随机数发生器的种子。我们这样做是为了为我们检测到的每一列文本生成颜色(这对可视化很有用)。

然后，我们从磁盘加载输入数据`--image`,并将其转换为灰度。

我们的下一个代码块检测我们的`image`中的大块文本，采用与我们的教程 [OCR 识别护照](https://pyimagesearch.com/2021/12/01/ocr-passports-with-opencv-and-tesseract/) *:* 相似的过程

```py
# initialize a rectangular kernel that is ~5x wider than it is tall,
# then smooth the image using a 3x3 Gaussian blur and then apply a
# blackhat morphological operator to find dark regions on a light
# background
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 11))
gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
grad = np.absolute(grad)
(minVal, maxVal) = (np.min(grad), np.max(grad))
grad = (grad - minVal) / (maxVal - minVal)
grad = (grad * 255).astype("uint8")

# apply a closing operation using the rectangular kernel to close
# gaps in between characters, apply Otsu's thresholding method, and
# finally a dilation operation to enlarge foreground regions
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
thresh = cv2.threshold(grad, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.dilate(thresh, None, iterations=3)
cv2.imshow("Thresh", thresh)
```

首先，我们构建一个大的矩形内核，它的宽度比高度大(**第 37 行**)。然后，我们对图像应用小的高斯模糊，并计算 blackhat 输出，显示亮背景上的暗区域(即亮背景上的暗文本)。

下一步是计算 blackhat 图像的梯度幅度表示，并将输出图像缩放回范围`[0, 255]` ( **第 43-47 行**)。

我们现在可以应用一个关闭操作(**线 52** )。**我们在这里使用我们的大矩形`kernel`来缩小表格中文本行之间的间隙。**

我们通过对图像进行阈值处理并应用一系列扩张来放大前景区域来完成流水线。

**图** **5** 显示该预处理流水线的输出。在*左边*，我们有我们的原始输入图像。这张图片是迈克尔·乔丹棒球卡的背面(当他离开 NBA 一年去打棒球的时候)。由于他们没有乔丹的任何棒球数据，他们在卡片的背面包括了他的篮球数据，尽管这是一张棒球卡(我知道这很奇怪，这也是为什么他的棒球卡是收藏家的物品)。

**我们的目标是 OCR 他的*“完整 NBA 记录”*表。**如果你看一下**图 5** *(右)*，你可以看到这个桌子是一个大的矩形斑点。

我们的下一个代码块处理检测和提取这个表:

```py
# find contours in the thresholded image and grab the largest one,
# which we will assume is the stats table
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
tableCnt = max(cnts, key=cv2.contourArea)

# compute the bounding box coordinates of the stats table and extract
# the table from the input image
(x, y, w, h) = cv2.boundingRect(tableCnt)
table = image[y:y + h, x:x + w]

# show the original input image and extracted table to our screen
cv2.imshow("Input", image)
cv2.imshow("Table", table)
```

首先，我们检测阈值图像中的所有轮廓(**行 60-62** )。然后我们根据轮廓的面积找到一个最大的轮廓(**线 63** )。

我们假设最大的轮廓是我们的桌子，我们可以从**图 5** *(右图)*中验证，事实确实如此。但是，您可能需要更新轮廓处理，以使用[其他方法](https://pyimagesearch.com/?s=contour+processing)为您的项目找到表格。这里介绍的方法当然不是万能的解决方案。

给定对应于表格的轮廓`tableCnt`，我们计算其边界框并提取`table`本身(**行 67 和 68** )。

提取的`table`然后通过**线 72** ( **图 6** )显示到我们的屏幕上。

现在我们有了统计数据`table`，让我们对其进行 OCR:

```py
# set the PSM mode to detect sparse text, and then localize text in
# the table
options = "--psm 6"
results = pytesseract.image_to_data(
	cv2.cvtColor(table, cv2.COLOR_BGR2RGB),
	config=options,
	output_type=Output.DICT)

# initialize a list to store the (x, y)-coordinates of the detected
# text along with the OCR'd text itself
coords = []
ocrText = []
```

**第 76-80 行**使用 Tesseract 来计算`table`中每段文本的边界框位置。

此外，注意我们在这里是如何使用`--psm 6`的，原因是当文本是单个统一的文本块时，这种特殊的页面分割模式工作得很好。

大多数数据表*和*将是统一的。他们将利用几乎相同的字体和字体大小。

如果`--psm 6`对你不太管用，你应该试试教程中涉及的其他 PSM 模式，**[**tesserac Page Segmentation Modes(PSMs)讲解:如何提高你的 OCR 准确率**](https://pyimg.co/irgus) *。*具体来说，我建议看一下关于检测稀疏文本的`--psm 11`——这种模式也适用于表格数据。**

 **在应用 OCR 文本检测之后，我们初始化两个列表:`coords`和`ocrText`。`coords`列表将存储文本边界框的 *(x，y)*-坐标，而`ocrText`将存储实际的 OCR 文本本身。

让我们继续循环我们的每个文本检测:

```py
# loop over each of the individual text localizations
for i in range(0, len(results["text"])):
	# extract the bounding box coordinates of the text region from
	# the current result
	x = results["left"][i]
	y = results["top"][i]
	w = results["width"][i]
	h = results["height"][i]

	# extract the OCR text itself along with the confidence of the
	# text localization
	text = results["text"][i]
	conf = int(results["conf"][i])

	# filter out weak confidence text localizations
	if conf > args["min_conf"]:
		# update our text bounding box coordinates and OCR'd text,
		# respectively
		coords.append((x, y, w, h))
		ocrText.append(text)
```

**行 91-94** 从检测到的文本区域提取边界框坐标，而**行 98 和 99** 提取 OCR 处理的文本本身，以及文本检测的置信度。

**第 102 行**过滤掉弱文本检测。如果`conf`小于我们的`--min-conf`，我们忽略文本检测。否则，我们分别更新我们的`coords`和`ocrText`列表。

我们现在可以进入项目的集群阶段:

```py
# extract all x-coordinates from the text bounding boxes, setting the
# y-coordinate value to zero
xCoords = [(c[0], 0) for c in coords]

# apply hierarchical agglomerative clustering to the coordinates
clustering = AgglomerativeClustering(
	n_clusters=None,
	affinity="manhattan",
	linkage="complete",
	distance_threshold=args["dist_thresh"])
clustering.fit(xCoords)

# initialize our list of sorted clusters
sortedClusters = []
```

**第 110 行**从文本边界框中提取所有的 *x* 坐标。注意我们是如何在这里形成一个合适的 *(x，y)-* 坐标元组，但是为每个条目设置`y = 0` 。为什么`y` 设置为`0`？

答案是双重的:

1.  首先，为了应用 HAC，我们需要一组输入向量(也称为“特征向量”)。我们的输入向量必须至少是二维的*，所以我们添加了一个包含值`0`的平凡维度。*
2.  其次，我们对 y 坐标值不感兴趣。我们只想聚集在 *x* 坐标位置上。具有相似的 *x* 坐标的文本片段很可能是表格中某一列的一部分。

从那里，**行 113-118** 应用层次凝聚聚类。我们将`n_clusters`设置为`None`,因为我们*不知道*我们想要生成多少个集群——相反，我们希望 HAC 自然地形成集群，并继续使用我们的`distance_ threshold`来创建集群。一旦没有两个聚类的距离小于`--dist-thresh`，我们就停止聚类处理。

另外，请注意，我们在这里使用的是曼哈顿距离函数。为什么不是其他的距离函数(例如欧几里德距离函数)？

虽然你可以(也应该)尝试我们的距离度量标准，但曼哈顿似乎是一个合适的选择。我们想要非常严格地要求 *x* 坐标彼此靠近。但是，我还是建议你尝试其他的距离方法。

关于 scikit-learn 的`AgglomerativeClustering`实现的更多细节，包括对参数和自变量的全面回顾，请务必参考 [scikit-learn 关于方法](http://pyimg.co/y3bic)的文档。

现在我们的聚类已经完成，让我们遍历每个独特的聚类:

```py
# loop over all clusters
for l in np.unique(clustering.labels_):
	# extract the indexes for the coordinates belonging to the
	# current cluster
	idxs = np.where(clustering.labels_ == l)[0]

	# verify that the cluster is sufficiently large
	if len(idxs) > args["min_size"]:
		# compute the average x-coordinate value of the cluster and
		# update our clusters list with the current label and the
		# average x-coordinate
		avg = np.average([coords[i][0] for i in idxs])
		sortedClusters.append((l, avg))

# sort the clusters by their average x-coordinate and initialize our
# data frame
sortedClusters.sort(key=lambda x: x[1])
df = pd.DataFrame()
```

**第 124 行**循环遍历所有唯一的集群标签。然后，我们使用 NumPy 来查找当前标签为`l` ( **第 127 行**)的所有数据点的索引，从而暗示它们都属于同一个集群。

**第 130 行**然后验证当前的集群中有超过`--min-size`个项目。

然后，我们计算集群内的 *x* 坐标的平均值，并用包含当前标签`l`和平均 *x* 坐标值的二元组更新我们的`sortedClusters`列表。

**第 139 行**根据我们的平均 *x* 坐标对我们的`sortedClusters`列表进行排序。我们执行这个排序操作，使得我们的集群在页面上从左到右排序*。*

最后，我们初始化一个空的`DataFrame`来存储我们的多列 OCR 结果。

现在让我们循环遍历排序后的集群:

```py
# loop over the clusters again, this time in sorted order
for (l, _) in sortedClusters:
	# extract the indexes for the coordinates belonging to the
	# current cluster
	idxs = np.where(clustering.labels_ == l)[0]

	# extract the y-coordinates from the elements in the current
	# cluster, then sort them from top-to-bottom
	yCoords = [coords[i][1] for i in idxs]
	sortedIdxs = idxs[np.argsort(yCoords)]

	# generate a random color for the cluster
	color = np.random.randint(0, 255, size=(3,), dtype="int")
	color = [int(c) for c in color]
```

**第 143 行**在我们的`sortedClusters`列表中的标签上循环。然后，我们找到属于当前聚类标签`l`的所有数据点的索引(`idxs`)。

使用这些索引，我们从集群中的所有文本片段中提取出 *y-* 坐标，并从*到* ( **第 150 行和第 151 行**)对它们进行排序。

我们还为当前列初始化了一个随机的`color`(这样我们就可以可视化哪些文本属于哪一列)。

现在，让我们遍历该列中的每一段文本:

```py
	# loop over the sorted indexes
	for i in sortedIdxs:
		# extract the text bounding box coordinates and draw the
		# bounding box surrounding the current element
		(x, y, w, h) = coords[i]
		cv2.rectangle(table, (x, y), (x + w, y + h), color, 2)

	# extract the OCR'd text for the current column, then construct
	# a data frame for the data where the first entry in our column
	# serves as the header
	cols = [ocrText[i].strip() for i in sortedIdxs]
	currentDF = pd.DataFrame({cols[0]: cols[1:]})

	# concatenate *original* data frame with the *current* data
	# frame (we do this to handle columns that may have a varying
	# number of rows)
	df = pd.concat([df, currentDF], axis=1)
```

对于每个`sortedIdx`，我们抓取边界框 *(x，y)*——文本块的坐标，并将其绘制在我们的`table` ( **第 158-162 行**)上。

然后我们抓取当前集群中所有的`ocrText`片段，从*到* ( **第 167 行**)排序。**这些文本代表表格中的一个唯一列(`cols`)。**

既然我们已经提取了当前列，我们为它创建一个单独的`DataFrame`，假设列中的第一个条目是标题，其余的是数据(**第 168 行**)。

最后，我们将原始数据帧`df`与新数据帧`currentDF` ( **第 173 行**)连接起来。我们执行这个连接操作来处理某些列比其他列有更多行的情况(自然地，由于表的设计，或者由于 Tesseract OCR 引擎在一行中缺少一段文本)。

至此，我们的表格 OCR 过程已经完成，我们只需要将表格保存到磁盘:

```py
# replace NaN values with an empty string and then show a nicely
# formatted version of our multi-column OCR'd text
df.fillna("", inplace=True)
print(tabulate(df, headers="keys", tablefmt="psql"))

# write our table to disk as a CSV file
print("[INFO] saving CSV file to disk...")
df.to_csv(args["output"], index=False)

# show the output image after performing multi-column OCR
cv2.imshow("Output", image)
cv2.waitKey(0)
```

如果一些列比其他列有更多的条目，那么空条目将被填充一个“非数字”(`NaN`)值。我们在第 177 行的**中用一个空字符串替换所有的`NaN`值。**

第 178 行向我们的终端显示了一个格式良好的表格，表明我们已经成功地对多列数据进行了 OCR。

然后，我们将数据帧作为 CSV 文件保存到磁盘的第**182 行。**

最后，我们在屏幕的第 185 行显示输出图像。

### **多列 OCR 结果**

我们现在可以看到我们的多列 OCR 脚本了！

打开终端并执行以下命令:

```py
$ python multi_column_ocr.py --image michael_jordan_stats.png --output results.csv
+----+---------+---------+-----+---------+-------+-------+-------+-------+-------+--------+
|    | Year    | CLUB    |   G |     FG% |   REB |   AST | STL   | BLK   |   PTS |   AVG. |
|----+---------+---------+-----+---------+-------+-------+-------+-------+-------+--------|
|  0 | 1984-85 | CHICAGO |  82 | 515     |   534 |   481 | 196   | 69    |  2313 |  282   |
|  1 | 1985-86 | CHICAGO |  18 |   0.457 |    64 |    53 | ar    | A     |   408 |  227   |
|  2 | 1986-87 | CHICAGO |  82 | 482     |   430 |   377 | 236   | 125   |  3041 |   37.1 |
|  3 | 1987-88 | CHICAGO |  82 | 535     |   449 |   485 | 259   | 131   |  2868 |   35   |
|  4 | 1988-89 | CHICAGO |  81 | 538     |   652 |   650 | 234   | 65    |  2633 |  325   |
|  5 | 1989-90 | CHICAGO |  82 | 526     |   565 |   519 | 227   | 54    |  2763 |   33.6 |
|  6 | TOTALS  |         | 427 | 516     |  2694 |  2565 | 1189  | 465   | 14016 |  328   |
+----+---------+---------+-----+---------+-------+-------+-------+-------+-------+--------+
[INFO] saving CSV file to disk...
```

**图** **7** *(上)*显示提取的表格。然后，我们应用层次凝聚聚类(HAC)对表格进行 OCR，得到底部的*图。*

注意我们是如何对文本列进行颜色编码的。具有相同边框颜色的文本属于*相同列*，表明我们的 HAC 方法成功地将文本列关联在一起。

我们的 OCR 表的文本版本可以在上面的终端输出中看到。我们的 OCR 结果在很大程度上是非常准确的，但也有一些值得注意的问题。

首先，字段 goal percentage ( `FG%`)除了一行之外都缺少小数点，可能是因为图像的分辨率太低，Tesseract 无法成功检测小数点。幸运的是，这很容易解决——使用基本的字符串解析/正则表达式来插入小数点*或*,只需将字符串转换为`float`数据类型。然后会自动添加小数点。

同样的问题可以在每场比赛的平均分数(`AVG.`)栏中找到——小数点又不见了。

不过，这个问题有点难以解决。如果我们简单地转换成一个`float`，文本`282`将会被错误地解析为`0.282`。相反，我们能做的是:

1.  检查绳子的长度
2.  如果字符串有*四个字符*，那么已经添加了小数点，所以不需要做进一步的工作
3.  否则，字符串有三个字符，所以在第二个和第三个字符之间插入一个小数点

任何时候，你都可以利用任何关于手头 OCR 任务的先验知识，这样你就有了更容易的 T2 时间。在这种情况下，我们的领域知识告诉我们哪些列应该有小数点，因此我们可以利用文本后处理试探法来进一步改进我们的 OCR 结果，即使 Tesseract 的性能没有达到最佳。

我们的 OCR 结果的最大问题可以在`STL`和`BLK`列中找到，这里的 OCR 结果完全不正确。我们对此无能为力，因为这是宇宙魔方结果的问题，是*而不是*我们的列分组算法。

由于本教程重点关注多列数据的 OCR 处理*特别是*，最重要的是，驱动多列数据的*算法*，我们在这里不打算花太多时间关注对 Tesseract 选项和配置的改进。

在我们的 Python 脚本被执行之后，我们有一个输出`results.csv`文件，其中包含我们的表，该表作为 CSV 文件被序列化到磁盘上。我们来看看它的内容:

```py
$ cat results.csv
Year,CLUB,G,FG%,REB,AST,STL,BLK,PTS,AVG.
1984-85,CHICAGO,82,515,534,481,196,69,2313,282
1985-86,CHICAGO,18,.457,64,53,ar,A,408,227
1986-87,CHICAGO,82,482,430,377,236,125,3041,37.1
1987-88,CHICAGO,82,535,449,485,259,131,2868,35.0
1988-89,CHICAGO,81,538,652,650,234,65,2633,325
1989-90,CHICAGO,82,526,565,519,227,54,2763,33.6
TOTALS,,427,516,2694,2565,1189,465,14016,328
```

如您所见，我们的 OCR 表已经作为 CSV 文件写入磁盘。您可以获取这个 CSV 文件，并使用数据挖掘技术对其进行处理。

## **总结**

在本教程中，您学习了如何使用 Tesseract OCR 引擎和分层凝聚聚类(HAC)对多列数据进行 OCR。

我们的多列 OCR 算法的工作原理是:

1.  使用梯度和形态学操作检测输入图像中的文本表格
2.  提取检测到的表
3.  使用 Tesseract(或等效物)定位表格中的文本并提取边界框 *(x，y)*-表格中文本的坐标
4.  将 HAC 应用于具有最大距离阈值截止的桌子的 *x* 坐标上的聚类

本质上，我们在这里所做的是用坐标 *x* 将文本本地化分组，这些坐标或者是*相同*或者是*彼此非常接近*。

这种方法为什么有效？

好吧，考虑电子表格、表格或任何其他具有多列的文档的结构。在这种情况下，每列中的数据将具有几乎相同的起始*x*-坐标，因为它们属于*相同的*列。因此，我们可以利用这些知识，然后用近似相同的 *x* 坐标将文本分组。

虽然这种方法很简单，但在实践中效果很好。

### **引用信息**

**罗斯布鲁克，A.** “多栏表格 OCR”， *PyImageSearch* ，2022，【https://pyimg.co/h18s2】T4

```py
@article{Rosebrock_2022_MCT_OCR,
  author = {Adrian Rosebrock},
  title = {Multi-Column Table {OCR}},
  journal = {PyImageSearch},
  year = {2022},
  note = {https://pyimg.co/h18s2},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*******