# OpenCV 用于目标检测的选择性搜索

> 原文：<https://pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/>

今天，您将学习如何使用 OpenCV 选择性搜索进行对象检测。

**今天的教程是我们关于深度学习和对象检测的 4 部分系列的第 2 部分:**

*   **Part 1:** *[用 Keras 和 TensorFlow](https://pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/)* 把任何深度学习图像分类器变成物体检测器
*   **第二部分:** *OpenCV 选择性搜索进行物体检测*(今日教程)
*   **第三部分:** *用 OpenCV、Keras 和 TensorFlow 进行物体检测的区域提议*(下周教程)
*   **第四部分:** *用 Keras 和 TensorFlow 进行 R-CNN 物体探测*(两周内出版)

选择性搜索(Selective Search)由 Uijlings 等人在其 2012 年的论文中首次提出，是计算机视觉、深度学习和对象检测研究的关键部分。

在他们的工作中，Uijlings 等人证明了:

1.  如何对图像进行 ***过分割*以自动识别图像中*可能*包含物体的位置**
2.  这种选择性搜索比穷尽式计算图像金字塔和滑动窗口(且不损失准确性)更高效
3.  *并且该选择性搜索可以用任何利用图像金字塔和滑动窗口的对象检测框架替换*

 ***自动区域提议算法，如选择性搜索，为 Girshick 等人的开创性 R-CNN 论文铺平了道路，该论文产生了高度准确的基于深度学习的对象检测器。

此外，选择性搜索和对象检测的研究使研究人员能够创建最先进的区域提议网络(RPN)组件，这些组件甚至比选择性搜索更准确和更高效*(参见 Girshick 等人 2015 年关于更快 R-CNN 的后续论文)。*

但是在我们进入 RPNs 之前，我们首先需要了解选择性搜索是如何工作的，包括我们如何使用 OpenCV 利用选择性搜索进行对象检测。

**要了解如何使用 OpenCV 的选择性搜索进行对象检测，*继续阅读。***

## **OpenCV 对象检测选择性搜索**

在本教程的第一部分，我们将讨论通过选择性搜索的区域建议的概念，以及它们如何有效地取代使用图像金字塔和滑动窗口来检测图像中的对象的传统方法。

从这里，我们将详细回顾选择性搜索算法，包括它如何通过以下方式过度分割图像:

1.  颜色相似性
2.  纹理相似性
3.  尺寸相似性
4.  形状相似性
5.  最终的元相似性，它是上述相似性度量的线性组合

然后，我将向您展示如何使用 OpenCV 实现选择性搜索。

### **区域提议与滑动窗口和图像金字塔**

在上周的教程中，你学习了如何通过应用图像金字塔和滑动窗口将任何图像分类器转变为对象检测器。

作为复习，**图像金字塔创建输入图像的多尺度表示，允许我们以多尺度/尺寸检测物体:**

**滑动窗口在图像金字塔的每一层上操作，从左到右和从上到下滑动，从而允许我们定位*给定物体在图像中的*位置:**

图像金字塔和滑动窗口方法存在许多问题，但两个主要问题是:

1.  慢得令人痛苦。 即使使用[循环优化](https://pyimagesearch.com/2017/08/28/fast-optimized-for-pixel-loops-with-opencv-and-python/)方法和多重处理，遍历每个图像金字塔层并通过滑动窗口检查图像中的每个位置在计算上也是昂贵的。
2.  **他们对参数选择很敏感。**您的图像金字塔比例和滑动窗口大小的不同值会导致*在阳性检测率、误检和漏检方面产生显著*不同的结果。

鉴于这些原因，计算机视觉研究人员已经开始研究创建自动区域提议生成器，以取代滑动窗口和图像金字塔。

总的想法是，区域提议算法应该检查图像，并试图找到图像中*可能*包含对象的区域(将区域提议视为[显著性检测](https://pyimagesearch.com/2018/07/16/opencv-saliency-detection/))。

区域提议算法应该:

1.  比滑动窗口和图像金字塔更快更有效
2.  准确检测图像中*可能*包含物体的区域
3.  将这些“候选提议”传递给下游分类器，以实际标记区域，从而完成对象检测框架

问题是，什么类型的区域提议算法可以用于对象检测？

### **什么是选择性搜索，选择性搜索如何用于物体检测？**

在 OpenCV 中实现的选择性搜索算法是由 Uijlings 等人在他们 2012 年的论文 *[中首次提出的，用于对象识别的选择性搜索](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)* 。

选择性搜索通过使用[超像素算法](https://pyimagesearch.com/tag/superpixel/)对图像进行过度分割来工作(代替 SLIC，Uijlings 等人使用来自 Felzenszwalb 和 Huttenlocher 2004 年论文 *[的 Felzenszwalb 方法，高效的基于图形的图像分割](http://cs.brown.edu/people/pfelzens/segment/)* )。

运行 Felzenszwalb 超像素算法的示例如下所示:

从那里开始，选择性搜索寻求将超像素合并在一起，找到图像中可能包含物体的区域。

选择性搜索基于五个关键相似性度量以分层方式合并超像素:

1.  **颜色相似度:**计算图像每个通道的 25-bin 直方图，将它们串联在一起，得到最终的描述符为 *25×3=75-d.* 任意两个区域的颜色相似度用[直方图相交距离](https://mpatacchiola.github.io/blog/2016/11/12/the-simplest-classifier-histogram-intersection.html)来度量。
2.  **纹理相似度:**对于纹理，选择性搜索提取每个通道 8 个方向的高斯导数(假设 3 通道图像)。这些方向用于计算每个通道的 10-bin 直方图，生成最终的纹理描述符，即 *8x10x=240-d.* 为了计算任何两个区域之间的纹理相似性，再次使用直方图相交。
3.  **大小相似性:**选择性搜索使用的大小相似性度量标准更倾向于将较小的区域更早*分组，而不是更晚*。*任何以前使用过层次凝聚聚类(HAC)算法的人都知道，HAC 容易使聚类达到临界质量，然后将它们接触到的*所有东西*结合起来。通过强制较小的区域更早地合并，我们可以帮助防止大量的集群吞噬所有较小的区域。*
4.  ***形状相似性/兼容性:**选择性搜索中形状相似性背后的思想是它们应该彼此*兼容*。如果两个区域“适合”彼此，则认为它们“兼容”(从而填补了我们区域提案生成中的空白)。此外，不接触的形状不应合并。*
5.  ***最终元相似性度量:**最终元相似性充当颜色相似性、纹理相似性、尺寸相似性和形状相似性/兼容性的线性组合。*

 *应用这些层次相似性度量的选择性搜索的结果可以在下图中看到:

在金字塔的底层，我们可以看到来自 Felzenszwalb 方法的原始过分割/超像素生成。

在中间层，我们可以看到区域被连接在一起，最终形成最终的提议集( *top* )。

如果你有兴趣了解更多关于选择性搜索的基础理论，我建议你参考以下资源:

*   *[高效的基于图的图像分割](http://cs.brown.edu/people/pfelzens/papers/seg-ijcv.pdf)* (Felzenszwalb 和 Huttenlocher，2004)
*   【乌伊林斯等，2012】
*   *[【选择性搜索进行物体检测(c++/Python)](https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/)*(Chandel/Mallick，2017)

### **选择性搜索生成区域，*非*类标签**

我看到的选择性搜索的一个常见误解是，读者错误地认为选择性搜索*取代了*整个对象检测框架，如 HOG +线性 SVM、R-CNN 等。

事实上，几周前，PyImageSearch 的读者 Hayden 发来了一封电子邮件，提出了完全相同的问题:

> *你好，Adrian，我正在用 OpenCV 使用选择性搜索来检测对象。*
> 
> 然而，选择性搜索只是返回边界框——我似乎想不出如何获得与这些边界框相关联的标签。

**所以，事情是这样的:**

1.  选择性搜索*是否生成图像中 ***可能*** 包含对象的区域。*
2.  *但是，选择性搜索*并不知道*在那个区域是什么(把它想成是[显著性检测](https://pyimagesearch.com/2018/07/16/opencv-saliency-detection/)的表亲)。***
3.  ***选择性搜索意味着用 ***取代*** 这种计算成本高、效率极低的方法，这种方法穷尽性地使用图像金字塔和滑动窗口来检查潜在物体的图像位置。***
4.  **通过使用选择性搜索，我们可以更有效地检查图像中*可能包含物体**的区域，然后将这些区域传递给 SVM、CNN 等。进行最终分类。*****

 ***如果你正在使用选择性搜索，只要记住**选择性搜索算法将*而不是*给你类别标签预测**——假设你的下游分类器将为你做那件事(下周博客文章的主题)**。**

但与此同时，让我们学习如何在我们自己的项目中使用 OpenCV 选择性搜索。

### **项目结构**

一定要抓住。本教程的压缩文件来自 ***【下载】*** 部分。一旦您提取了文件，您可以使用`tree`命令来查看里面的内容:

```py
$ tree
.
├── dog.jpg
└── selective_search.py

0 directories, 2 files
```

我们的项目非常简单，由一个 Python 脚本(`selective_search.py`)和一个测试图像(`dog.jpg`)组成。

在下一节中，我们将学习如何用 Python 和 OpenCV 实现我们的选择性搜索脚本。

### **用 OpenCV 和 Python 实现选择性搜索**

我们现在准备用 OpenCV 实现选择性搜索！

打开一个新文件，将其命名为`selective_search.py`，并插入以下代码:

```py
# import the necessary packages
import argparse
import random
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-m", "--method", type=str, default="fast",
	choices=["fast", "quality"],
	help="selective search method")
args = vars(ap.parse_args())
```

```py
# load the input image
image = cv2.imread(args["image"])

# initialize OpenCV's selective search implementation and set the
# input image
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

# check to see if we are using the *fast* but *less accurate* version
# of selective search
if args["method"] == "fast":
	print("[INFO] using *fast* selective search")
	ss.switchToSelectiveSearchFast()

# otherwise we are using the *slower* but *more accurate* version
else:
	print("[INFO] using *quality* selective search")
	ss.switchToSelectiveSearchQuality()
```

*   `"fast"`方法:`switchToSelectiveSearchFast`
*   `"quality"`方法:`switchToSelectiveSearchQuality`

通常，更快的方法将是合适的；但是，根据您的应用程序，您可能希望牺牲速度来获得更好的质量结果(稍后将详细介绍)。

让我们继续**使用我们的图像执行选择性搜索**:

```py
# run selective search on the input image
start = time.time()
rects = ss.process()
end = time.time()

# show how along selective search took to run along with the total
# number of returned region proposals
print("[INFO] selective search took {:.4f} seconds".format(end - start))
print("[INFO] {} total region proposals".format(len(rects)))
```

要运行选择性搜索，我们只需在我们的`ss`对象上调用`process`方法(**第 37 行**)。我们已经围绕这个调用设置了时间戳，所以我们可以感受一下算法有多快；**第 42 行**向我们的终端报告选择性搜索基准。

随后，**行 43** 告诉我们选择性搜索操作找到的区域建议的数量。

现在，如果我们不将结果可视化，找到我们的区域提案会有什么乐趣呢？毫无乐趣。最后，让我们在图像上绘制输出:

```py
# loop over the region proposals in chunks (so we can better
# visualize them)
for i in range(0, len(rects), 100):
	# clone the original image so we can draw on it
	output = image.copy()

	# loop over the current subset of region proposals
	for (x, y, w, h) in rects[i:i + 100]:
		# draw the region proposal bounding box on the image
		color = [random.randint(0, 255) for j in range(0, 3)]
		cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(0) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
```

### **OpenCV 选择性搜索结果**

我们现在已经准备好使用 OpenCV 对我们自己的图像进行选择性搜索。

首先使用这篇博文的 ***【下载】*** 部分下载源代码和示例图片。

从那里，打开一个终端，并执行以下命令:

```py
$ python selective_search.py --image dog.jpg 
[INFO] using *fast* selective search
[INFO] selective search took 1.0828 seconds
[INFO] 1219 total region proposals
```

在这里，你可以看到 OpenCV 的选择性搜索“快速模式”运行了大约 1 秒钟，生成了 1219 个边界框——**图 4** 中的可视化显示了我们在选择性搜索生成的每个区域上循环，并将它们可视化到我们的屏幕上。

如果您对这种可视化感到困惑，请考虑选择性搜索的最终目标:**到*用更有效的区域提议生成方法取代*传统的计算机视觉对象检测技术，如*滑动窗口*和*图像金字塔*。**

因此，选择性搜索将*而不是*告诉你*ROI 中的*是什么，但是它会告诉你 ROI“足够有趣”以传递给下游分类器(例如、SVM、CNN 等。)进行最终分类。

让我们对同一幅图像应用选择性搜索，但这一次，使用`--method quality`模式:

```py
$ python selective_search.py --image dog.jpg --method quality
[INFO] using *quality* selective search
[INFO] selective search took 3.7614 seconds
[INFO] 4712 total region proposals
```

“高质量”选择性搜索方法生成的区域提案增加了 286%，但运行时间也延长了 247%。

您是否应该使用“快速”或“高质量”模式取决于您的应用。

**在大多数情况下，“快速”选择性搜索就足够了，但是您可以选择使用“高质量”模式:**

1.  当执行推理并希望确保为下游分类器生成更多高质量区域时(当然，这意味着实时检测不是一个问题)
2.  当使用选择性搜索来生成训练数据时，从而确保生成更多的正区域和负区域供分类器学习

### **在哪里可以了解更多关于 OpenCV 的物体检测选择性搜索？**

在下周的教程中，您将学习如何:

1.  使用选择性搜索来生成对象检测建议区域
2.  取一个预先训练的 CNN，并对每个区域进行分类(丢弃任何低置信度/背景区域)
3.  应用非最大值抑制来返回我们的最终对象检测

在两周内，我们将使用选择性搜索来生成训练数据，然后微调 CNN 以通过区域提议来执行对象检测。

到目前为止，这是一个很棒的系列教程，你不想错过接下来的两个！

## **总结**

在本教程中，您学习了如何使用 OpenCV 执行选择性搜索来生成对象检测建议区域。

选择性搜索的工作原理是*通过基于五个关键要素组合区域来对图像*进行过度分割:

1.  颜色相似性
2.  纹理相似性
3.  尺寸相似性
4.  形状相似性
5.  **和最终相似性度量，其是上述四个相似性度量的线性组合**

值得注意的是，选择性搜索本身*并不*执行对象检测。

**相反，选择性搜索返回*可能*包含一个对象的建议区域。**

这里的想法是，我们用一个*更便宜、更高效的*选择性搜索来取代我们的*计算昂贵、效率极低的*滑动窗口和图像金字塔。

下周，我将向您展示如何通过选择性搜索生成建议区域，然后在它们之上运行图像分类器，**允许您创建一个基于深度学习的特定对象检测器！**

敬请关注下周的教程。

**要下载这篇文章的源代码(并在本系列的下一篇教程发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！**********