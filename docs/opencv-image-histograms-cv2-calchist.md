# OpenCV 图像直方图(cv2.calcHist)

> 原文：<https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/>

在本教程中，您将学习如何使用 OpenCV 和`cv2.calcHist`函数计算图像直方图。

直方图在计算机视觉的几乎每个方面都很普遍。

我们使用灰度直方图进行阈值处理。我们使用直方图进行白平衡。我们使用颜色直方图来跟踪图像中的对象，例如使用 CamShift 算法。

我们使用颜色直方图作为特征，包括多维的颜色直方图。

在抽象意义上，我们使用图像梯度直方图来形成 HOG 和 SIFT 描述符。

甚至在图像搜索引擎和机器学习中使用的非常流行的视觉单词包表示也是直方图！

我敢肯定这不是你第一次在研究中碰到直方图。

那么，为什么直方图如此有用呢？

因为直方图捕捉了一组数据的*频率分布*。事实证明，检查这些频率分布是构建简单图像处理技术的一种非常好的方式……以及非常强大的机器学习算法。

在这篇博客文章中，你将会看到关于图像直方图的介绍，包括如何计算灰度和颜色直方图。在以后的博客文章中，我将介绍更高级的直方图技术。

**学习如何使用 OpenCV 和`cv2.calcHist`函数计算图像直方图，** ***继续阅读。***

## **OpenCV 图像直方图(cv2.calcHist )**

在本教程的第一部分，我们将讨论什么是图像直方图。从那以后，我将向您展示如何使用 OpenCV 和`cv2.calcHist`函数来计算图像直方图。

接下来，我们将配置我们的开发环境，并检查我们的项目目录结构。

然后，我们将实现三个 Python 脚本:

1.  一个用于计算灰度直方图
2.  另一个是计算颜色直方图
3.  以及演示如何只为输入图像的屏蔽区域计算直方图的最终脚本

我们开始吧！

### **什么是图像直方图？**

直方图表示图像中像素强度(彩色或灰度)的分布。它可以被可视化为一个图形(或绘图),给出强度(像素值)分布的高层次直觉。在本例中，我们将假设一个 RGB 颜色空间，因此这些像素值将在 *0* 到 *255* 的范围内。

绘制直方图时，*x*-轴充当我们的“仓”如果我们用`256`条构建一个直方图，那么我们可以有效地计算每个像素值出现的次数。

相比之下，如果我们只使用`2`(等间距)的面元，那么我们计算一个像素在*【0，128】*或*【128，255】*范围内的次数。

然后，在 *y* 轴上绘制归入 *x* 轴值的像素数。

让我们看一个示例图像来更清楚地说明这一点:

在**图 1** 中，我们绘制了一个直方图，沿 *x* 轴有 256 个面元，沿 *y* 轴有落入给定面元的像素百分比。检查直方图，注意有三个主峰。

直方图中的第一个峰值大约在 *x= *65** 处，我们看到像素数量出现了一个尖锐的峰值——很明显，图像中存在某种具有非常暗的值的对象。

然后，我们在直方图中看到一个缓慢上升的峰值，我们在大约 *x=100* 处开始上升，最后在大约 *x=150* 处结束下降。这个区域可能是指图像的背景区域。

最后，我们看到在范围 *x=150* 到 *x=175 内有大量的像素。*很难说这个区域到底是什么，但它肯定占据了图像的很大一部分。

***注:*** *我是故意* ***而不是*** *揭示我用来生成这个直方图的图像。我只是在展示我看柱状图时的思维过程。在不知道数据来源的情况下，能够解释和理解你正在查看的数据是一项很好的技能。*

通过简单地检查图像的直方图，您可以对对比度、亮度和强度分布有一个大致的了解。如果这个概念对你来说是新的或陌生的，不要担心——我们将在本课的后面检查更多像这样的例子。

### **使用 OpenCV 通过 cv2.calcHist 函数计算直方图**

让我们开始构建一些自己的直方图。

我们将使用`cv2.calcHist`函数来构建直方图。在我们进入任何代码示例之前，让我们快速回顾一下这个函数:

`cv2.calcHist(images, channels, mask, histSize, ranges)`

*   **`images` :** 这是我们想要计算直方图的图像。包装成列表:`[myImage]`。
*   **`channels` :** 一个索引列表，我们在其中指定想要计算直方图的通道的索引。要计算灰度图像的直方图，列表应该是`[0]`。为了计算所有三个红色、绿色和蓝色通道的直方图，通道列表应该是`[0, 1, 2]`。
*   **`mask` :** 还记得在我的 [*用 OpenCV* 向导进行图像蒙版中学习蒙版吗？](https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/)嗯，这里我们可以供应一个口罩。如果提供了屏蔽，将只为*屏蔽的像素*计算直方图。如果我们没有掩码或者不想应用掩码，我们可以只提供值`None`。
*   **`histSize` :** 这是我们在计算直方图时想要使用的箱数。同样，这是一个列表，我们为每个通道计算一个直方图。箱子的大小不必都一样。以下是每个通道 32 个仓的示例:`[32, 32, 32]`。
*   **`ranges` :** 可能的像素值范围。通常情况下，这是每个通道的 *[0，256]* (即*而不是*一个错别字—`cv2.calcHist`函数的结束范围是不包含的，因此您将希望提供 256 而不是 255 的值)，但是如果您使用 RGB 以外的颜色空间[如 HSV]，范围可能会有所不同。)

在接下来的小节中，您将获得使用 OpenCV 的`cv2.calcHist`函数计算图像直方图的实践经验。

### **配置您的开发环境**

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

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
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

让我们从回顾我们的项目目录结构开始。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。从那里，您将看到以下目录结构:

```py
$ tree . --dirsfirst
.
├── beach.png
├── color_histograms.py
├── grayscale_histogram.py
└── histogram_with_mask.py

0 directories, 4 files
```

我们今天要复习三个 Python 脚本:

1.  `grayscale_histogram.py`:演示了如何从一个输入的单通道灰度图像中计算像素强度直方图
2.  `color_histograms.py`:显示如何计算 1D(即“展平”)、2D 和 3D 颜色直方图
3.  `histogram_with_mask.py`:演示如何只为输入图像的*蒙版区域*计算直方图

我们的单个图像`beach.png`，作为这三个脚本的输入。

### **使用 OpenCV 创建灰度直方图**

让我们学习如何使用 OpenCV 计算灰度直方图。打开项目结构中的`grayscale_histogram.py`文件，我们将开始:

```py
# import the necessary packages
from matplotlib import pyplot as plt
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image")
args = vars(ap.parse_args())
```

**第 2-4 行**导入我们需要的 Python 包。我们将使用`matplotlib`的`pyplot`模块来绘制图像直方图，`argparse`用于命令行参数，`cv2`用于 OpenCV 绑定。

我们只有一个命令行参数需要解析，`--image`，它是驻留在磁盘上的输入图像的路径。

接下来，让我们从磁盘加载输入图像，并将其转换为灰度:

```py
# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

灰度转换完成后，我们可以使用`cv2.calcHist`函数来计算图像直方图:

```py
# compute a grayscale histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
```

继续将`cv2.calcHist`调用的参数与上面*“使用 OpenCV 计算 cv2.calcHist 函数的直方图”*一节中的函数文档进行匹配。

我们可以看到，我们的第一个参数是灰度图像。一幅灰度图像只有一个通道，所以我们为通道设置了一个值`[0]`。我们没有掩码，所以我们将掩码值设置为`None`。我们将在直方图中使用 256 个面元，可能的值范围从`0`到`255`。

计算出图像直方图后，我们在屏幕上显示灰度图像，并绘制出*未标准化的*图像直方图:

```py
# matplotlib expects RGB images so convert and then display the image
# with matplotlib
plt.figure()
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

# plot the histogram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
```

**非标准化直方图统计分布的原始频率。**考虑数一数袋子里不同颜色的 M & M 的数量。我们最终会得到每种颜色的整数计数。

另一方面，如果我想要每种颜色的百分比呢？

嗯，这很容易得到！我会简单地用每个整数除以袋子里 M&M 的总数。因此，我没有使用*原始频率直方图*，而是以*归一化*直方图结束，该直方图计算每种颜色的**百分比**。根据定义，归一化直方图的总和正好是`1`。

那么，我为什么更喜欢标准化的直方图而不是非标准化的直方图呢？

事实证明，是有的。让我们做一个小小的思维实验:

在这个思维实验中，我们想要比较两幅图像的直方图。这些图像在各个方面、形状和形式上都是相同的，只有一个例外。第一幅图像的尺寸*是第二幅图像的尺寸*的一半。

当我们去比较直方图时，虽然分布的形状看起来相同，但我们会注意到沿 *y* 轴的像素计数会显著不同。事实上，第一幅图像的 *y* 轴计数将是第二幅图像的*y*轴计数的一半。

这是为什么？

我们正在比较*原始频率计数*和*百分比计数！*

考虑到这一点，让我们看看如何归一化直方图，并获得每个像素的百分比计数:

```py
# normalize the histogram
hist /= hist.sum()

# plot the normalized histogram
plt.figure()
plt.title("Grayscale Histogram (Normalized)")
plt.xlabel("Bins")
plt.ylabel("% of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
```

直方图的归一化只需要一行代码，我们可以在第 34 行的**处看到:这里我们简单地将直方图中每个仓的原始频率计数除以计数的总和，这样我们得到的是每个仓的*百分比*，而不是每个仓的*原始计数*。**

然后我们在第 37-43 行**上绘制归一化直方图。**

### **灰度直方图结果**

我们现在可以用 OpenCV 计算灰度直方图了！

请务必访问本指南的 ***“下载”*** 部分，以检索源代码和示例图像。从那里，您可以执行以下命令:

```py
$ python grayscale_histogram.py --image beach.png
```

那么，我们如何解释这个直方图呢？

嗯，在 *x* 轴上绘制了面元 *(0-255)* 。并且 *y* 轴计算每个箱中的像素数量。大多数像素落在大约 *60* 到 *180 的范围内。*观察直方图的两个尾部，我们可以看到很少有像素落在范围*【0，50】*和*【200，255】*内——这意味着图像中很少有“黑”和“白”像素。

注意**图 4** 包含一个*非标准化*直方图，这意味着它包含面元内的*原始整数计数*。

如果我们想要的是 ***百分比计数*** (这样当所有的值加起来和`1`)，我们可以检查*归一化*直方图:

现在，容器计数表示为百分比**而不是原始计数。**

根据您的应用，您可能需要非标准化或标准化的图像直方图。在这一课中，我已经演示了如何计算两种类型的*，这样你就可以使用两种方法。*

### **使用 OpenCV 创建颜色直方图**

在上一节中，我们探讨了灰度直方图。现在让我们继续计算图像每个通道的直方图。

打开项目目录中的`color_histograms.py`文件，我们将开始:

```py
# import the necessary packages
from matplotlib import pyplot as plt
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image")
args = vars(ap.parse_args())
```

**第 2-5 行**导入我们需要的 Python 包。我们的导入与`grayscale_histogram.py`相同，但是有一个例外——我们现在正在导入`imutils`,它包含一个方便的`opencv2matplotlib`函数，用 matplotlib 处理 RGB 与 BGR 图像的显示。

然后我们在第 8-11 行解析我们的命令行参数。我们只需要一个参数`--image`，它是我们的输入图像在磁盘上的路径。

现在让我们计算三个直方图，输入 RGB 图像的每个通道一个直方图:

```py
# load the input image from disk
image = cv2.imread(args["image"])

# split the image into its respective channels, then initialize the
# tuple of channel names along with our figure for plotting
chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

# loop over the image channels
for (chan, color) in zip(chans, colors):
	# create a histogram for the current channel and plot it
	hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
	plt.plot(hist, color=color)
	plt.xlim([0, 256])
```

第 14 行从磁盘加载我们的图像。然后我们在第 18 行**将图像分割成各自的 BGR 通道。**

**第 19 行**定义了通道名称(字符串)列表，而**第 20-23 行**初始化我们的 matplotlib 图。

然后我们在**线 26** 上到达一个`for`回路。在这里，我们开始循环图像中的每个通道。

然后，对于每个通道，我们在第 28 行**上计算直方图。**代码与计算灰度图像直方图的代码相同；然而，我们对每个红色、绿色和蓝色通道都这样做，这使我们能够表征像素强度的分布。我们将直方图添加到第 29 行的**图上。**

现在让我们来看看 2D 直方图的计算。到目前为止，我们一次只计算了一个通道的直方图。现在我们转向多维直方图，一次考虑两个通道。

我喜欢用**和**这个词来解释多维直方图。

例如，我们可以问这样一个问题:

*   有多少像素的红色值为 10 **而蓝色值为 30**？
*   有多少像素的绿色值为 200 **红色值为 130**？

通过使用连接词**和**，我们能够构建多维直方图。

就这么简单。让我们来看一些代码来自动完成构建 2D 直方图的过程:

```py
# create a new figure and then plot a 2D color histogram for the
# green and blue channels
fig = plt.figure()
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and B")
plt.colorbar(p)

# plot a 2D color histogram for the green and red channels
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and R")
plt.colorbar(p)

# plot a 2D color histogram for blue and red channels
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for B and R")
plt.colorbar(p)

# finally, let's examine the dimensionality of one of the 2D
# histograms
print("2D histogram shape: {}, with {} values".format(
	hist.shape, hist.flatten().shape[0]))
```

是的，这是相当多的代码。但这只是因为我们正在为 RGB 通道的每种组合计算 2D 颜色直方图:红色和绿色，红色和蓝色，绿色和蓝色。

**既然我们正在处理多维直方图，我们需要记住我们正在使用的仓的数量。**在前面的例子中，我使用了 256 个箱子进行演示。

然而，如果我们为 2D 直方图中的每个维度使用 256 个面元，那么我们得到的直方图将具有 65，536 个单独的像素计数(因为 *256* × *256 = 65，536* )。这不仅浪费资源，而且不切实际。计算多维直方图时，大多数应用程序使用 8 到 64 个区间。正如**第 36 行和第 37 行**所示，我现在使用 32 个 bin，而不是 256 个。

通过检查`cv2.calcHist`函数的第一个参数，可以看出这段代码最重要的一点。这里我们看到我们正在传递两个通道的列表:绿色和蓝色通道。这就是全部了。

那么 2D 直方图是如何存储在 OpenCV 中的呢？这是 2D 数字阵列。由于我为每个通道使用了 32 个面元，现在我有了一个 *32* *×* *32* 直方图。

正如我们在运行这个脚本时将会看到的，我们的 2D 直方图将会有一个维度为*32**×**32 = 1024*(**行 60 和 61** )。

使用 2D 直方图一次考虑两个通道。但是如果我们想考虑所有三个 RGB 通道呢？你猜对了。我们现在要构建一个 3D 直方图:

```py
# our 2D histogram could only take into account 2 out of the 3
# channels in the image so now let's build a 3D color histogram
# (utilizing all channels) with 8 bins in each direction -- we
# can't plot the 3D histogram, but the theory is exactly like
# that of a 2D histogram, so we'll just show the shape of the
# histogram
hist = cv2.calcHist([image], [0, 1, 2],
	None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print("3D histogram shape: {}, with {} values".format(
	hist.shape, hist.flatten().shape[0]))
```

这里的代码非常简单——它只是上面的 2D 直方图代码的扩展。我们现在正在为每个 RGB 通道计算一个*8**×**8**×**8*直方图。我们无法将这个直方图可视化，但我们可以看到形状确实是具有`512`值的`(8, 8, 8)`。

最后，让我们在屏幕上显示我们的原始输入`image`:

```py
# display the original input image
plt.figure()
plt.axis("off")
plt.imshow(imutils.opencv2matplotlib(image))

# show our plots
plt.show()
```

`imutils`中的`opencv2matplotlib`便利功能用于将 BGR 图像转换为 RGB。如果你曾经使用过 Jupyter 笔记本，这个方便的方法也是很棒的。

### **颜色直方图结果**

我们现在可以用 OpenCV 计算颜色直方图了！

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。从那里，您可以执行`color_histograms.py`脚本:

```py
$ python color_histograms.py --image beach.png
```

在左边的*，你可以看到我们的输入图像包含一个宁静的海滩场景。在右边的*，我们有我们的“扁平”颜色直方图。**

 *我们看到在仓 *100* 周围的绿色直方图中有一个尖峰。我们看到图像中的大多数绿色像素都包含在*【85，200】*范围内——这些区域从海滩图像中的绿色植被和树木到浅绿色都是中等范围的。

在我们的图像中，我们还可以看到许多较亮的蓝色像素。考虑到我们既能看到清澈的大海*又能看到万里无云的蓝天*，这并不奇怪。

现在让我们来可视化我们的 2D 直方图:

第一个是绿色和蓝色通道的 2D 颜色直方图，第二个是绿色和红色的直方图，第三个是蓝色和红色的直方图。蓝色阴影代表*低*像素计数，而红色阴影代表*高*像素计数(即 2D 直方图中的峰值)。

我们倾向于在绿色和蓝色直方图中看到许多峰值，其中 *x=28* 和 *y=27* 。这个区域对应于植被和树木的绿色像素以及天空和海洋的蓝色像素。

此外，我们可以通过命令行输出来研究图像直方图的形状:

```py
2D histogram shape: (32, 32), with 1024 values
3D histogram shape: (8, 8, 8), with 512 values
```

每个 2D 直方图为 32×32。相乘，这意味着每个直方图是 *32×32 = 1024-d* ，意味着每个直方图由总共 1024 个值表示。

另一方面，我们的 3D 直方图是*8**×**8**×**8*，因此当相乘时，我们看到我们的 3D 图像直方图由 512 个值表示。

### **使用 OpenCV** 计算屏蔽区域的图像直方图

到目前为止，我们已经学会了如何计算输入图像的整体直方图？**但是如果你想只为输入图像的** ***特定区域*** **计算图像直方图呢？**

例如，您可能正在构建一个自动识别和匹配服装的计算机视觉应用程序。你首先要从图像中分割出服装。之后，你需要计算一个颜色直方图来量化衣服的颜色分布…但是你不想在计算中包括背景像素，这些像素不属于衣服本身。

那么，在这种情况下你会怎么做呢？

是否可以只为输入图像的特定*区域计算颜色直方图？*

你打赌它是。

在您的项目目录结构中打开`histogram_with_mask.py`,我将向您展示它是如何完成的:

```py
# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import cv2
```

**第 2-4 行**导入我们需要的 Python 包。我们将使用`matplotlib`进行绘图，使用 NumPy 进行数值数组处理，使用`cv2`进行 OpenCV 绑定。

现在让我们定义一个方便的函数`plot_histogram`，它将把我们的大多数 matplotlib 调用封装到一个简单易用的函数中:

```py
def plot_histogram(image, title, mask=None):
	# split the image into its respective channels, then initialize
	# the tuple of channel names along with our figure for plotting
	chans = cv2.split(image)
	colors = ("b", "g", "r")
	plt.figure()
	plt.title(title)
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")

	# loop over the image channels
	for (chan, color) in zip(chans, colors):
		# create a histogram for the current channel and plot it
		hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
		plt.plot(hist, color=color)
		plt.xlim([0, 256])
```

在**第 6 行**，我们定义`plot_histogram`。这个函数接受三个参数:一个`image`、我们绘图的`title`和一个`mask`。如果我们没有图像的蒙版，`mask`默认为`None`。

我们的`plot_histogram`函数的主体只是计算图像中每个通道的直方图并绘制出来，就像本节前面的例子一样；然而，请注意，我们现在正在将`mask`参数传递给`cv2.calcHist`。

**在事件中，我们** ***do*** **有一个输入蒙版，我们在这里传递它，这样 OpenCV 就知道只将** ***中被蒙版的像素*** **从输入`image`中包含到直方图构造中。**

有了`plot_histogram`函数定义，我们可以继续我们脚本的其余部分:

```py
# load the beach image and plot a histogram for it
image = cv2.imread("beach.png")
plot_histogram(image, "Histogram for Original Image")
cv2.imshow("Original", image)
```

我们首先在**行 24** 从磁盘加载我们的海滩图像，在**行 25** 将其显示在屏幕上，然后在**行 26** 为海滩图像的每个通道绘制颜色直方图。

请注意，我们这里传入的是*而不是*蒙版，所以我们计算的是图像的*整体*的颜色直方图。

现在，让我们来学习如何只为*图像的*蒙版区域计算颜色直方图:

```py
# construct a mask for our image; our mask will be *black* for regions
# we want to *ignore* and *white* for regions we want to *examine*
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (60, 290), (210, 390), 255, -1)
cv2.imshow("Mask", mask)

# display the masked region
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Applying the Mask", masked)

# compute a histogram for our image, but we'll only include pixels in
# the masked region
plot_histogram(image, "Histogram for Masked Image", mask=mask)

# show our plots
plt.show()
```

我们将我们的`mask`定义为一个 NumPy 数组，其宽度和高度与第 30 行上的**海滩图像相同。然后我们在第 31 条**线上画一个白色矩形，从点 *(60，210)* 到点 *(290，390)* 。****

这个矩形将作为我们的蒙版——直方图计算中只考虑原始图像中属于蒙版区域的像素。

最后，我们在屏幕上显示结果图(**第 43 行**)。

### **屏蔽直方图结果**

我们现在准备计算图像的掩蔽直方图。

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。从那里，您可以执行`histogram_with_mask.py`脚本:

```py
$ python histogram_with_mask.py
```

首先，我们显示原始输入图像及其相应的通道直方图，其中*未应用掩蔽*:

从这里，我们构建一个矩形蒙版，并使用按位 and 来可视化图像的蒙版区域:

最后，让我们将*整个*图像的直方图与仅从图像的*遮罩区域*计算的直方图进行比较:

在*左侧*，我们有*原始图像*的直方图，而在*右侧*，我们有*屏蔽图像*的直方图。

对于屏蔽图像，大多数红色像素落在范围*【10，25】*内，表明红色像素对我们的图像贡献很小。这是有道理的，因为我们的海洋和天空是蓝色的。

然后出现绿色像素，但这些像素朝向分布的较亮端，对应于绿色树叶和树木。

最后，我们的蓝色像素落在较亮的范围，显然是我们的蓝色海洋和天空。

**最重要的是，把我们的被遮罩的颜色直方图** ***(右)*** **与未被遮罩的颜色直方图** ***(左)*** **上图**。*注意颜色直方图的显著差异。*

通过利用遮罩，我们能够将计算仅应用于图像中我们感兴趣的特定区域——在本例中，我们只想检查蓝天和海洋的分布。

## **总结**

在本教程中，您学习了所有关于图像直方图的知识，以及如何使用 OpenCV 和`cv2.calcHist`函数来计算它们。

直方图非常简单，但却是非常强大的工具。它们广泛用于阈值处理、颜色校正，甚至图像特征！确保你很好地掌握了直方图，你将来肯定会用到它们。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****