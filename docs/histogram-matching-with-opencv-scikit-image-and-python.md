# 使用 OpenCV、scikit-image 和 Python 进行直方图匹配

> 原文：<https://pyimagesearch.com/2021/02/08/histogram-matching-with-opencv-scikit-image-and-python/>

在本教程中，您将学习如何使用 OpenCV 和 scikit-image 执行直方图匹配。

上周我们讨论了[直方图均衡化](https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/)，这是一种基本的图像处理技术，可以提高输入图像的对比度。

但是如果你想让两幅图像的对比度或颜色分布自动匹配，那该怎么办呢？

例如，假设我们有一个输入图像和一个参考图像。我们的目标是:

1.  计算每个图像的直方图
2.  取参考图像直方图
3.  使用参考直方图更新输入图像中的像素亮度值，使它们匹配

我们在这篇博文顶部的图中看到了结果。请注意输入图像是如何更新以匹配参考图像的颜色分布的。

当将图像处理流水线应用于在不同照明条件下捕获的图像时，直方图匹配是有益的，从而创建图像的“标准化”表示，*而不管它们是在什么照明条件下捕获的*(当然，对照明条件变化的程度设置合理的预期)。

今天我们将详细讨论直方图匹配。下周我将展示如何使用直方图均衡化进行色彩校正和色彩恒常性。

**要了解如何使用 OpenCV 和 scikit-image 执行直方图匹配，*继续阅读。***

## **使用 OpenCV、scikit-image 和 Python 进行直方图匹配**

在本教程的第一部分，我们将讨论直方图匹配，并使用 OpenCV 和 scikit-image 实现直方图匹配。

然后，我们将配置我们的开发环境，并检查我们的项目目录结构。

一旦我们理解了我们的项目结构，我们将实现一个 Python 脚本，它将:

1.  加载输入图像(即“源”图像)
2.  加载参考图像
3.  计算两幅图像的直方图
4.  获取输入图像并将其与参考图像进行匹配，从而将颜色/强度分布从参考图像转移到源图像中

我们将讨论我们的结果来结束本教程。

### **什么是直方图匹配？**

直方图匹配最好被认为是一种“转换”我们的目标是获取一幅输入图像(“源”)并更新其像素强度，使输入图像直方图的分布与参考图像的分布相匹配。

虽然输入图像的实际内容没有改变，但是*像素分布*改变，从而基于参考图像的分布来调整输入图像的亮度和对比度。

应用直方图匹配可以让我们获得有趣的美学效果(我们将在本教程的后面看到)。

此外，我们可以使用直方图匹配作为基本颜色校正/颜色恒常性的一种形式，允许我们建立更好、更可靠的图像处理管道*，而无需*利用复杂、计算昂贵的机器学习和深度学习算法。我们将在下周的指南中讨论这个概念。

### **OpenCV 和 scikit-image 如何用于直方图匹配？**

手工实现直方图匹配可能很痛苦，但对我们来说幸运的是，scikit-image 库已经有了一个`match_histograms`函数(你可以在这里找到[的文档)。](https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.match_histograms)

因此，应用直方图匹配就像用 OpenCV 的`cv2.imread`加载两幅图像，然后调用 scikit-image 的`match_histograms`函数一样简单:

```py
src = cv2.imread(args["source"])
ref = cv2.imread(args["reference"])
multi = True if src.shape[-1] > 1 else False
matched = exposure.match_histograms(src, ref, multichannel=multi)
```

我们将在本指南的后面更详细地介绍如何使用`match_histograms`函数。

### **配置您的开发环境**

要了解如何执行直方图匹配，您需要安装 OpenCV 和 scikit-image:

两者都可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
$ pip install scikit-image==0.18.1
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

### 项目结构

在我们可以用 OpenCV 和 scikit-image 实现直方图匹配之前，让我们首先使用我们的项目目录结构。

请务必访问本指南的 ***“下载”*** 部分，以检索源代码和示例图像。

从那里，看看我们的项目目录结构:

```py
$ tree . --dirsfirst
.
├── empire_state_cloudy.png
├── empire_state_sunset.png
└── match_histograms.py

0 directories, 3 files
```

我们今天只需要查看一个 Python 脚本`match_histograms.py`，它将加载`empire_state_cloud.png`(即*源*图像)和`empire_state_sunset.png`(即*引用*图像)。

然后，我们的脚本将应用直方图匹配将颜色分布从参考图像转移到源图像。

在这种情况下，我们可以将阴天拍摄的照片转换成美丽的日落！

### **用 OpenCV 和 scikit-image 实现直方图匹配**

准备好实现直方图匹配了吗？

太好了，我们开始吧。

打开`match_histograms.py`并插入以下代码:

```py
# import the necessary packages
from skimage import exposure
import matplotlib.pyplot as plt
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True,
	help="path to the input source image")
ap.add_argument("-r", "--reference", required=True,
	help="path to the input reference image")
args = vars(ap.parse_args())
```

我们从第 2-5 行开始，导入我们需要的 Python 包。

我们需要 scikit-image 的`exposure`库来计算图像直方图、累积分布函数，并应用直方图匹配。

我们将使用`matplotlib`来绘制直方图，这样我们可以在应用直方图匹配之前和之后可视化它们。

我们导入`argparse`用于命令行参数解析，导入`cv2`用于 OpenCV 绑定。

接下来是我们的命令行参数:

1.  `--source`:我们输入图像的路径。
2.  `--reference`:参考图像的路径。

**请记住，我们在这里做的是从*参考*获取颜色分布，然后将其传输到*源。*** 源图像本质上是将更新其颜色分布的图像。

现在让我们从磁盘加载源图像和参考图像:

```py
# load the source and reference images
print("[INFO] loading source and reference images...")
src = cv2.imread(args["source"])
ref = cv2.imread(args["reference"])

# determine if we are performing multichannel histogram matching
# and then perform histogram matching itself
print("[INFO] performing histogram matching...")
multi = True if src.shape[-1] > 1 else False
matched = exposure.match_histograms(src, ref, multichannel=multi)

# show the output images
cv2.imshow("Source", src)
cv2.imshow("Reference", ref)
cv2.imshow("Matched", matched)
cv2.waitKey(0)
```

**第 17 行和第 18 行**加载我们的`src`和`ref`图像。

加载这两幅图像后，我们可以对第 23 行**和第 24 行**进行直方图匹配。

直方图匹配可应用于单通道和多通道图像。**第 23 行**设置一个布尔值`multi`，这取决于我们使用的是多通道图像(`True`)还是单通道图像(`False`)。

从此，应用直方图匹配就像调用 scikit-image 的`exposure`子模块中的`match_histogram`函数一样简单。

从那里，**行 27-30** 显示我们的源，参考，并输出直方图`matched`图像到我们的屏幕上。

至此，我们在技术上已经完成了，但是为了充分理解直方图匹配的作用，让我们来检查一下`src`、`ref`和`matched`图像的颜色直方图:

```py
# construct a figure to display the histogram plots for each channel
# before and after histogram matching was applied
(fig, axs) =  plt.subplots(nrows=3, ncols=3, figsize=(8, 8))

# loop over our source image, reference image, and output matched
# image
for (i, image) in enumerate((src, ref, matched)):
	# convert the image from BGR to RGB channel ordering
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# loop over the names of the channels in RGB order
	for (j, color) in enumerate(("red", "green", "blue")):
		# compute a histogram for the current channel and plot it
		(hist, bins) = exposure.histogram(image[..., j],
			source_range="dtype")
		axs[j, i].plot(bins, hist / hist.max())

		# compute the cumulative distribution function for the
		# current channel and plot it
		(cdf, bins) = exposure.cumulative_distribution(image[..., j])
		axs[j, i].plot(bins, cdf)

		# set the y-axis label of the current plot to be the name
		# of the current color channel
		axs[j, 0].set_ylabel(color)
```

**第 34 行**创建一个`3 x 3`图形，分别显示`src`、`ref`和`matched`图像的红色、绿色和蓝色通道的直方图。

从那里，**行 38** 在我们的`src`、`ref`和`matched`图像上循环。然后我们将当前的`image`从 BGR 转换到 RGB 通道排序。

接下来是实际的绘图:

*   **第 45 和 46 行**计算当前`image`的当前通道的直方图
*   然后我们在第 47 行绘制直方图
*   类似地，**行 51 和 52** 计算当前通道的累积分布函数，然后将其绘制出来
*   **第 56 行**设置颜色的 *y* 轴标签

最后一步是显示绘图:

```py
# set the axes titles
axs[0, 0].set_title("Source")
axs[0, 1].set_title("Reference")
axs[0, 2].set_title("Matched")

# display the output plots
plt.tight_layout()
plt.show()
```

这里，我们设置每个轴的标题，然后在屏幕上显示直方图。

### **直方图匹配结果**

我们现在准备用 OpenCV 应用直方图匹配！

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

从那里，我们打开一个 shell 并执行以下命令:

```py
$ python match_histograms.py --source empire_state_cloudy.png \
	--reference empire_state_sunset.png
[INFO] loading source and reference images...
[INFO] performing histogram matching...
```

假设我们正在纽约市进行家庭度假，我们想要捕捉日落时帝国大厦的美丽照片。今天是我们假期的最后一天，我们的航班计划在午饭前起飞。

在去机场之前，我们很快拍了一张帝国大厦的照片——但这是一个多云、沉闷的日子。这与我们想要的天堂般的日落照片相去甚远。

我们该怎么办？

**答案是应用直方图匹配。**

在回家的飞机上，我们打开笔记本电脑开始工作:

*   我们从从磁盘*(左)*加载原始输入图像开始
*   然后我们打开谷歌图片，找到一张日落时帝国大厦的照片(*右)*
*   最后，我们应用直方图匹配将色彩强度分布从日落照片(参考图像)转移到我们的输入图像(源图像)

结果是它现在看起来像我们在白天拍摄的原始图像，现在看起来像是在黄昏时拍摄的(图 4)！

**图 5** 分别显示了图像的红色、绿色和蓝色通道的源图像、参考图像和匹配图像的直方图:

*“Source”*列显示了我们的输入源图像中像素强度的分布。*“参考”*栏显示了我们从磁盘加载的参考图像的分布。最后，*【匹配】*栏显示应用直方图匹配的输出。

**注意源像素强度是如何调整的，以匹配参考图像的分布！**那个操作，本质上就是直方图匹配。

虽然本教程从美学角度关注直方图匹配，但直方图匹配还有更重要的实际应用。

直方图匹配可用作图像处理管道中的归一化技术，作为一种颜色校正和颜色匹配的形式，从而使您能够获得一致的归一化图像表示，即使光照条件发生变化。

在下周的博客文章中，我将向您展示如何执行这种类型的规范化。

### **学分**

本教程的来源和参考图片分别来自[这篇文章](https://newyorkyimby.com/2020/09/empire-state-building-restoration-nearly-complete.html)和[这篇文章](https://www.great-towers.com/tower/empire-state-building)。感谢你让我有机会使用这些示例图像来教授直方图匹配。

此外，我想亲自感谢 scikit-image 库的开发人员和维护人员。我的直方图匹配实现基于 scikit-image 的[官方示例。](https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_histogram_matching.html)

我的主要贡献是演示了如何将 OpenCV 与 scikit-image 的`match_histograms`函数一起使用，并提供了对代码的更详细的讨论。

## **总结**

在本教程中，您学习了如何使用 OpenCV 和 scikit-image 执行直方图匹配。

直方图匹配是一种图像处理技术，它将像素强度分布从一幅图像(“参考”图像)转移到另一幅图像(“源”图像)。

虽然直方图匹配可以改善输入图像的美观，但它也可以用作归一化技术，其中我们“校正”输入图像，以使输入分布与参考分布*匹配，而不管光照条件的*变化。

执行这种标准化使我们作为计算机视觉从业者的生活更加轻松。如果我们可以安全地假设特定范围的光照条件，我们可以硬编码参数，包括 Canny 边缘检测阈值、高斯模糊大小等。

我将在下周的教程中向你展示如何执行这种类型的颜色匹配和规范化。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***