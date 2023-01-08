# OpenCV 连通分量标记和分析

> 原文：<https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/>

在本教程中，您将学习如何使用 OpenCV 执行连接组件标记和分析。具体来说，我们将重点介绍 OpenCV 最常用的连通分量标注函数`cv2.connectedComponentsWithStats`。

连通分量标记(也称为连通分量分析、斑点提取或区域标记)是[图论](https://en.wikipedia.org/wiki/Graph_theory)的算法应用，用于确定二进制图像中“斑点”状区域的连通性。

我们经常在使用轮廓的相同情况下使用连通分量分析；然而，连通分量标记通常可以为我们提供二值图像中斑点的更细粒度过滤。

使用轮廓分析时，我们经常受到轮廓层次的限制(即一个轮廓包含在另一个轮廓内)。有了连通分量分析，我们可以更容易地分割和分析这些结构。

连通分量分析的一个很好的例子是计算二进制(即，阈值化)牌照图像的连通分量，并基于它们的属性(例如，宽度、高度、面积、坚实度等)过滤斑点。).这正是我们今天要做的事情。

连通分量分析是添加到 OpenCV 工具带的另一个工具！

**要了解如何使用 OpenCV 执行连接组件标记和分析，*请继续阅读。***

## **OpenCV 连通分量标记和分析**

在本教程的第一部分，我们将回顾 OpenCV 提供的四个(是的，*四个*)函数来执行连通分量分析。这些函数中最受欢迎的是`cv2.connectedComponentsWithStats`。

从那里，我们将配置我们的开发环境，并审查我们的项目目录结构。

接下来，我们将实现两种形式的连通分量分析:

1.  第一种方法将演示如何使用 OpenCV 的连接组件分析功能，计算每个组件的统计数据，然后分别提取/可视化每个组件。
2.  第二种方法展示了一个连接组件分析的实际例子。我们对车牌进行阈值处理，然后使用连通分量分析来提取*和*车牌字符。

我们将在本指南的最后讨论我们的结果。

### **OpenCV 的连接组件函数**

OpenCV 提供了四个连通分量分析函数:

1.  `cv2.connectedComponents`
2.  `cv2.connectedComponentsWithStats`
3.  `cv2.connectedComponentsWithAlgorithm`
4.  `cv2.connectedComponentsWithStatsWithAlgorithm`

**最流行的方法是`cv2.connectedComponentsWithStats`，它返回以下信息:**

1.  连接组件的边界框
2.  组件的面积(像素)
3.  质心/中心 *(x，y)*-组件的坐标

第一种方法`cv2.connectedComponents`与第二种方法相同，只是*没有*返回上述统计信息。**在绝大多数情况下，你*会*需要统计数据，所以简单地用`cv2.connectedComponentsWithStats`代替是值得的。**

第三种方法`cv2.connectedComponentsWithAlgorithm`，实现了更快、更有效的连通分量分析算法。

如果 OpenCV 编译支持并行处理，那么`cv2.connectedComponentsWithAlgorithm`和`cv2.connectedComponentsWithStatsWithAlgorithm`将比前两个运行得更快。

但是一般来说，坚持使用`cv2.connectedComponentsWithStats`直到你对使用连接组件标签感到舒适为止。

### **配置您的开发环境**

要了解如何执行连接组件分析，您需要在计算机上安装 OpenCV:

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果你需要帮助为 OpenCV 配置开发环境，我*强烈推荐*阅读我的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让你启动并运行。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   准备好在您的 Windows、macOS 或 Linux 系统上运行代码*了吗*？

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

在我们用 OpenCV 实现连接组件分析之前，让我们先看一下我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像:

```py
$ tree . --dirsfirst
.
├── basic_connected_components.py
├── filtering_connected_components.py
└── license_plate.png

0 directories, 3 files
```

我们将对*应用连接成分分析，自动*从车牌中过滤出字符(`license_plate.png`)。

为了完成这项任务并了解有关连通分量分析的更多信息，我们将实现两个 Python 脚本:

1.  演示了如何应用连接组件标签，提取每个组件及其统计数据，并在我们的屏幕上显示它们。
2.  `filtering_connected_components.py`:应用连接组件分析，但通过检查每个组件的宽度、高度和面积(以像素为单位)来过滤掉非牌照字符。

### **用 OpenCV 实现基本连接组件**

让我们开始用 OpenCV 实现连通分量分析。

打开项目文件夹中的`basic_connected_components.py`文件，让我们开始工作:

```py
# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--connectivity", type=int, default=4,
	help="connectivity for connected component analysis")
args = vars(ap.parse_args())
```

**第 2 行和第 3 行**导入我们需要的 Python 包，而**第 6-11 行**解析我们的命令行参数。

我们有两个命令行参数:

1.  `--image`:我们的输入图像驻留在磁盘上的路径。
2.  `--connectivity`:或者`4`或者`8`连接(你可以参考[本页](https://stackoverflow.com/questions/7088678/4-connected-vs-8-connected-in-connected-component-labeling-what-is-are-the-meri)了解更多关于四对八连接的细节)。

让我们继续预处理我们的输入图像:

```py
# load the input image from disk, convert it to grayscale, and
# threshold it
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
```

**第 15-18 行**继续:

1.  从磁盘加载我们的输入`image`
2.  将其转换为灰度
3.  使用 [Otsu 的阈值方法](https://en.wikipedia.org/wiki/Otsu%27s_method)对其进行阈值处理

阈值处理后，我们的图像将如下所示:

请注意车牌字符是如何在黑色背景上显示为白色的。然而，在输入图像中也有一串*噪声*，其也作为前景出现。

**我们的目标是应用连通分量分析来过滤掉这些噪声区域，只给我们留下*和*车牌字符。**

但是在我们开始之前，让我们先来学习如何使用`cv2.connectedComponentsWithStats`函数:

```py
# apply connected component analysis to the thresholded image
output = cv2.connectedComponentsWithStats(
	thresh, args["connectivity"], cv2.CV_32S)
(numLabels, labels, stats, centroids) = output
```

对第 21 行和第 22 行上的`cv2.connectedComponentsWithStats`的调用使用 OpenCV 执行连通分量分析。我们在这里传入三个参数:

1.  二进制`thresh`图像
2.  `--connectivity`命令行参数
3.  数据类型(应该保留为`cv2.CV_32S`)

然后，`cv2.connectedComponentsWithStats`返回一个 4 元组:

1.  检测到的独特标签的总数(即总成分数)
2.  名为`labels`的遮罩与我们的输入图像`thresh`具有相同的空间维度。对于`labels`中的每个位置，我们都有一个整数 ID 值，对应于像素所属的连通分量。在本节的后面，您将学习如何过滤`labels`矩阵。
3.  `stats`:统计每个连接的组件，包括包围盒坐标和面积(以像素为单位)。
4.  每个相连组件的`centroids`(即中心) *(x，y)*-坐标。

现在让我们学习如何解析这些值:

```py
# loop over the number of unique connected component labels
for i in range(0, numLabels):
	# if this is the first component then we examine the
	# *background* (typically we would just ignore this
	# component in our loop)
	if i == 0:
		text = "examining component {}/{} (background)".format(
			i + 1, numLabels)

	# otherwise, we are examining an actual connected component
	else:
		text = "examining component {}/{}".format( i + 1, numLabels)

	# print a status message update for the current connected
	# component
	print("[INFO] {}".format(text))

	# extract the connected component statistics and centroid for
	# the current label
	x = stats[i, cv2.CC_STAT_LEFT]
	y = stats[i, cv2.CC_STAT_TOP]
	w = stats[i, cv2.CC_STAT_WIDTH]
	h = stats[i, cv2.CC_STAT_HEIGHT]
	area = stats[i, cv2.CC_STAT_AREA]
	(cX, cY) = centroids[i]
```

第 26 行遍历 OpenCV 返回的所有唯一连接组件的 id。

然后我们会遇到一个`if/else`语句:

*   第一个连接的组件，ID 为`0`，是*总是*的*背景*。我们通常会忽略背景，但是如果你需要它，请记住 ID `0`包含它。
*   否则，如果`i > 0`，那么我们知道这个组件更值得探索。

**第 44-49 行**向我们展示了如何解析我们的`stats`和`centroids`列表，允许我们提取:

1.  组件的起始`x`坐标
2.  组件的起始`y`坐标
3.  组件的宽度(`w`)
4.  组件的高度(`h`)
5.  质心 *(x，y)*-组件的坐标

现在让我们来看一下当前组件的边界框和质心:

```py
	# clone our original image (so we can draw on it) and then draw
	# a bounding box surrounding the connected component along with
	# a circle corresponding to the centroid
	output = image.copy()
	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
	cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
```

第 54 行创造了一个我们可以借鉴的`output`图像。然后，我们将组件的边界框绘制成绿色矩形(**线 55** )并将质心绘制成红色圆形(**线 56** )。

我们的最后一个代码块演示了如何为当前连接的组件创建一个遮罩:

```py
	# construct a mask for the current connected component by
	# finding a pixels in the labels array that have the current
	# connected component ID
	componentMask = (labels == i).astype("uint8") * 255

	# show our output image and connected component mask
	cv2.imshow("Output", output)
	cv2.imshow("Connected Component", componentMask)
	cv2.waitKey(0)
```

**第 61 行**首先找到`labels`中所有与当前组件 ID`i`相等的位置。然后，我们将结果转换为一个无符号的 8 位整数，对于背景，值为`0`，对于前景，值为`255`。

然后，`output`图像和`componentMask`显示在我们屏幕上的**64-66 行。**

### **OpenCV 连通分量分析结果**

我们现在准备好用 OpenCV 执行连接组件标记了！

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像:

```py
$ python basic_connected_components.py --image license_plate.png
[INFO] examining component 1/17 (background)
[INFO] examining component 2/17
[INFO] examining component 3/17
[INFO] examining component 4/17
[INFO] examining component 5/17
[INFO] examining component 6/17
[INFO] examining component 7/17
[INFO] examining component 8/17
[INFO] examining component 9/17
[INFO] examining component 10/17
[INFO] examining component 11/17
[INFO] examining component 12/17
[INFO] examining component 13/17
[INFO] examining component 14/17
[INFO] examining component 15/17
[INFO] examining component 16/17
[INFO] examining component 17/17
```

下面的动画展示了我在 17 个检测到的组件之间循环切换的过程:

第一个连通的组件其实就是我们的*后台*。我们通常*跳过*这个组件，因为背景并不经常需要。

然后显示其余的 16 个组件。对于每个组件，我们绘制了**边界框**(绿色矩形)和**质心/中心**(红色圆圈)。

您可能已经注意到，这些连接的组件中有一些是车牌字符，而其他的只是“噪音”

这就提出了一个问题:

> 有没有可能只检测**车牌字符的成分**？如果是这样，我们该怎么做？

我们将在下一节讨论这个问题。

### **如何用 OpenCV 过滤连通组件**

我们之前的代码示例演示了如何用 OpenCV 从*中提取*连接的组件，但是没有演示如何用*过滤*它们。

现在让我们来学习如何过滤连接的组件:

```py
# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--connectivity", type=int, default=4,
	help="connectivity for connected component analysis")
args = vars(ap.parse_args())
```

**第 2-4 行**导入我们需要的 Python 包，而**第 7-12 行**解析我们的命令行参数。

这些命令行参数*与我们之前脚本中的参数*相同，所以我建议你参考本教程的前面部分，以获得对它们的详细解释。

从那里，我们加载我们的图像，预处理它，并应用连接组件分析:

```py
# load the input image from disk, convert it to grayscale, and
# threshold it
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# apply connected component analysis to the thresholded image
output = cv2.connectedComponentsWithStats(
	thresh, args["connectivity"], cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

# initialize an output mask to store all characters parsed from
# the license plate
mask = np.zeros(gray.shape, dtype="uint8")
```

**第 16-19 行**加载我们的输入图像，并以与我们在之前的脚本中相同的方式对其进行预处理。然后，我们对第 22-24 行的**应用连通分量分析。**

**第 28 行**初始化一个输出`mask`来存储我们在执行连通分量分析后找到的所有牌照字符。

说到这里，我们现在来看一下每个独特的标签:

```py
# loop over the number of unique connected component labels, skipping
# over the first label (as label zero is the background)
for i in range(1, numLabels):
	# extract the connected component statistics for the current
	# label
	x = stats[i, cv2.CC_STAT_LEFT]
	y = stats[i, cv2.CC_STAT_TOP]
	w = stats[i, cv2.CC_STAT_WIDTH]
	h = stats[i, cv2.CC_STAT_HEIGHT]
	area = stats[i, cv2.CC_STAT_AREA]
```

注意，我们的`for`循环从 ID `1`开始，这意味着我们跳过了背景值`0`。

然后我们在第 35-39 行的**上提取当前连接组件的边界框坐标和`area`。**

我们现在准备过滤我们连接的组件:

```py
	# ensure the width, height, and area are all neither too small
	# nor too big
	keepWidth = w > 5 and w < 50
	keepHeight = h > 45 and h < 65
	keepArea = area > 500 and area < 1500

	# ensure the connected component we are examining passes all
	# three tests
	if all((keepWidth, keepHeight, keepArea)):
		# construct a mask for the current connected component and
		# then take the bitwise OR with the mask
		print("[INFO] keeping connected component '{}'".format(i))
		componentMask = (labels == i).astype("uint8") * 255
		mask = cv2.bitwise_or(mask, componentMask)
```

**第 43-45 行**展示了我们正在根据它们的宽度、高度和面积过滤连接的组件，丢弃那些*太小*或*太大的组件。*

***注:*** *想知道我是怎么想出这些值的？我使用了`print`语句来显示每个连接组件的宽度、高度和面积，同时将它们分别可视化到我的屏幕上。我记下了车牌字符的宽度、高度和面积，并找到了它们的最小/最大值，每端都有一点公差。对于您自己的应用程序，您也应该这样做。*

**第 49 行**验证`keepWidth`、`keepHeight`、`keepArea`都是`True`，暗示他们都通过了测试。

如果确实如此，我们计算当前标签 ID 的`componentMask`(就像我们在`basic_connected_components.py`脚本中所做的那样)并将牌照字符添加到我们的`mask`中。

最后，我们在屏幕上显示我们的输入`image`并输出牌照字符`mask`。

```py
# show the original input image and the mask for the license plate
# characters
cv2.imshow("Image", image)
cv2.imshow("Characters", mask)
cv2.waitKey(0)
```

正如我们将在下一节看到的，我们的`mask`将只包含牌照字符。

### **过滤连通分量结果**

让我们来学习如何使用 OpenCV 过滤连接的组件！

请务必访问本指南的 ***“下载”*** 部分，以检索源代码和示例图像—从那里，您可以执行以下命令:

```py
$ python filtering_connected_components.py --image license_plate.png
[INFO] keeping connected component 7
[INFO] keeping connected component 8
[INFO] keeping connected component 9
[INFO] keeping connected component 10
[INFO] keeping connected component 11
[INFO] keeping connected component 12
[INFO] keeping connected component 13
```

**图 5** 显示了过滤我们连接的组件的结果。在*顶部，*我们有包含牌照的原始输入图像。*底部*有过滤连接成分的结果，导致*只是*车牌字符本身。

如果我们正在构建一个自动牌照/车牌识别(ALPR/ANPR)系统，我们将获取这些字符，然后将它们传递给光学字符识别(OCR)算法进行识别。但这一切都取决于我们能否将字符二值化并提取出来，而连通分量分析使我们能够做到这一点！

## **总结**

在本教程中，您学习了如何执行连接的组件分析。

OpenCV 为我们提供了四个用于连通分量标记的函数:

1.  `cv2.connectedComponents`
2.  `cv2.connectedComponentsWithStats`
3.  `cv2.connectedComponentsWithAlgorithm`
4.  `cv2.connectedComponentsWithStatsWithAlgorithm()`

**其中最受欢迎的就是我们今天使用的`cv2.connectedComponentsWithStats`函数。**

当处理图像中的斑点状结构时，连通分量分析实际上可以*取代*轮廓检测、计算轮廓统计数据并过滤它们的过程。

连通分量分析是您工具箱中的一项便捷功能，因此请务必练习使用它。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***