# 使用基本图像处理改进 OCR 结果

> 原文：<https://pyimagesearch.com/2021/11/22/improving-ocr-results-with-basic-image-processing/>

在我们的[之前的教程](https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/)中，您学习了如何通过提供适当的页面分割模式(PSM)来提高 Tesseract OCR 的准确性。PSM 允许您根据特定图像及其拍摄环境选择分割方法。

然而，有时改变 PSM 是不够的，相反，您需要使用一点计算机视觉和图像处理来清理图像*，然后再将图像*通过 Tesseract OCR 引擎。

**要了解如何使用基本的图像处理来改善 OCR 结果，** ***继续阅读。***

## **通过基本图像处理改善 OCR 结果**

确切地说*您使用哪种*图像处理算法或技术*在很大程度上取决于*您的具体情况、项目要求和输入图像；然而，尽管如此，**在进行光学字符识别之前，获得应用图像处理来清理图像的经验仍然很重要。**

本教程将为您提供这样一个例子。然后，您可以使用此示例作为起点，通过 OCR 的基本图像处理来清理图像。

## **学习目标**

在本教程中，您将:

1.  了解基本图像处理如何*显著*提高 Tesseract OCR 的准确性
2.  了解如何应用阈值、距离变换和形态学操作来清理图像
3.  比较应用我们的图像处理程序之前的 OCR 准确度和之后的 T2
4.  了解如何为您的特定应用构建图像处理管道

## **图像处理和镶嵌光学字符识别**

我们将从回顾我们的项目目录结构开始本教程。从这里，我们将看到一个示例图像，其中无论 PSM 如何，Tesseract OCR 都无法正确地对输入图像进行 OCR。然后，我们将应用一些图像处理和 OpenCV 来预处理和清理输入，让 Tesseract 成功地对图像进行 OCR。最后，我们将学习在哪里可以提高你的计算机视觉技能，这样你就可以制作有效的图像处理管道，就像本教程中的一样。让我们开始吧！

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
*   **准备好在你的 Windows、macOS 或 Linux 系统上运行代码*了吗？***

那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本电脑将在 Windows、macOS 和 Linux 上运行！

### **项目结构**

让我们从回顾我们的项目目录结构开始:

```py
|-- challenging_example.png
|-- process_image.py
```

这个项目涉及一个具有挑战性的例子图像，我收集的堆栈溢出(**图 2** )。`challenging_example.png`全力以赴不能与宇宙魔方一起工作(即使有宇宙魔方 v4 的深度学习能力)。为了解决这个问题，我们将开发一个图像处理脚本`process_image.py`,为使用 Tesseract 的成功 OCR 准备图像。

### **当魔方本身无法对图像进行光学字符识别时**

在本教程中，我们将对来自**图** **2** 的图像进行光学字符识别。作为人类，我们可以很容易地看到这张图像包含文本*“12-14，”*，但对于计算机来说，它提出了几个挑战，包括:

*   复杂的纹理背景
*   背景不一致——左侧*的*明显比右侧*的*亮*而右侧*则更暗
*   背景中的文本有一点倾斜，这可能会使前景文本与背景难以分割
*   图像顶部*有许多黑色斑点，这也增加了文本分割的难度*

为了演示在当前状态下分割该图像有多困难，让我们将 Tesseract 应用于原始图像:

```py
$ tesseract challenging_example.png stdout
Warning: Invalid resolution 0 dpi. Using 70 instead.

Estimating resolution as 169
```

使用默认的 PSM(`--psm 3`)；全自动页面分割，但没有 OSD)， ***Tesseract 完全无法对图像进行 OCR，返回空输出。***

如果你要试验来自[先前教程](https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/)的各种 PSM 设置，你会看到返回*任何输出*的唯一 PSM 之一是`--psm 8`(将图像视为一个单词):

```py
$ tesseract challenging_example.png stdout --psm 8
Warning: Invalid resolution 0 dpi. Using 70 instead.
 T2eti@ce
```

不幸的是，不管 PSM 如何，Tesseract 完全无法按原样 OCR 这个图像，要么什么都不返回，要么完全是乱码。那么，在这种情况下我们该怎么办呢？我们是否将该图像标记为“无法进行 OCR ”,然后继续下一个项目？

没那么快——我们需要的只是一点图像处理。

### **实现用于 OCR 的图像处理流水线**

在这一节中，我将向您展示一个使用 OpenCV 库的巧妙设计的图像处理管道如何帮助我们预处理和清理输入图像。结果将是更清晰的图像，Tesseract 可以正确地进行 OCR。

我通常不包括图像子步骤结果。我将在这里包括图像子步骤结果，因为我们将执行图像处理操作，改变图像在每个步骤中的外观。您将看到阈值处理、形态学操作等的子步骤图像结果。，所以你可以很容易地跟随。

如上所述，打开一个新文件，将其命名为`process_image.py`，并插入以下代码:

```py
# import the necessary packages
import numpy as np
import pytesseract
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
args = vars(ap.parse_args())
```

导入我们的包之后，包括用于管道的 OpenCV 和用于 OCR 的 PyTesseract，我们解析输入的`--image`命令行参数。

现在让我们深入研究图像处理管道:

```py
# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# threshold the image using Otsu's thresholding method
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow("Otsu", thresh)
```

这里，我们加载输入`--image`并将其转换为灰度(**第 15 行和第 16 行**)。使用我们的`gray`图像，然后我们应用 Otsu 的自动阈值算法，如图**图 3** ( **第 19 行和第 20 行**)。假设我们已经通过`cv2.THRESH_BINARY_INV`标志反转了二进制阈值，我们希望进行 OCR 的文本现在是白色的(前景)，我们开始看到部分背景被移除。

在我们的图像准备好进行 OCR 之前，我们还有许多路要走，所以让我们看看接下来会发生什么:

```py
# apply a distance transform which calculates the distance to the
# closest zero pixel for each pixel in the input image
dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

# normalize the distance transform such that the distances lie in
# the range [0, 1] and then convert the distance transform back to
# an unsigned 8-bit integer in the range [0, 255]
dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
dist = (dist * 255).astype("uint8")
cv2.imshow("Dist", dist)

# threshold the distance transform using Otsu's method
dist = cv2.threshold(dist, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Dist Otsu", dist)
```

**第 25 行**使用 *5 x 5* 的`maskSize`对我们的`thresh`图像进行距离变换——这种计算确定了输入图像中每个像素到最近的 0 像素(黑色)的距离。随后，我们将`dist`归一化并缩放到范围`[0, 255]` ( **行 30 和 31** )。距离变换开始显示数字本身，因为从前景像素到背景有一个更大的距离*。距离变换还有一个好处，就是可以清除图像背景中的大部分噪声。有关此转换的更多详细信息，请参考 [OpenCV 文档](https://docs.opencv.org/4.4.0/d7/d1b/group__imgproc__misc.html#%20ga8a0b7fdfcb7a13dde018988ba3a43042)。*

从那里，我们再次应用 Otsu 的阈值方法*，但是这次是应用到`dist`图(**第 35 和 36 行**)的结果显示在**图 4** 中。请注意，我们没有使用反向二进制阈值(我们已经丢弃了标志的`_INV`部分)，因为我们希望文本保留在前景中(白色)。*

 *让我们继续清理我们的前景:

```py
# apply an "opening" morphological operation to disconnect components
# in the image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
opening = cv2.morphologyEx(dist, cv2.MORPH_OPEN, kernel)
cv2.imshow("Opening", opening)
```

应用开放形态学操作(即，膨胀后腐蚀)断开连接的斑点并去除噪声(**行 41 和 42** )。**图 5** 展示了我们的打开操作有效地将*【1】*字符从图像*顶部*(*洋红色圆圈*)的斑点中断开。

此时，我们可以从图像中提取轮廓，并对其进行过滤，以仅显示*和*数字:

```py
# find contours in the opening image, then initialize the list of
# contours which belong to actual characters that we will be OCR'ing
cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
chars = []

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)

	# check if contour is at least 35px wide and 100px tall, and if
	# so, consider the contour a digit
	if w >= 35 and h >= 100:
		chars.append(c)
```

在二值图像中提取轮廓意味着我们要找到所有孤立的前景斑点。**第 47 行和第 48 行**找到所有轮廓(包括字符和噪声)。

在我们找到轮廓(`cnts`)的所有后，我们需要确定*哪些要丢弃*哪些要添加到我们的角色列表中。**第 53–60 行**在`cnts`上循环，过滤掉至少 35 像素宽和 100 像素高的轮廓。通过测试的人将被添加到`chars`列表中。

现在我们已经隔离了我们的角色轮廓，让我们清理一下周围的区域:

```py
# compute the convex hull of the characters
chars = np.vstack([chars[i] for i in range(0, len(chars))])
hull = cv2.convexHull(chars)

# allocate memory for the convex hull mask, draw the convex hull on
# the image, and then enlarge it via a dilation
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.drawContours(mask, [hull], -1, 255, -1)
mask = cv2.dilate(mask, None, iterations=2)
cv2.imshow("Mask", mask)

# take the bitwise of the opening image and the mask to reveal *just*
# the characters in the image
final = cv2.bitwise_and(opening, opening, mask=mask)
```

为了消除字符周围的所有斑点，我们:

*   计算将包围数字的所有*(**行 63 和 64** )的凸面`hull`*
*   为一个`mask` ( **行 68** )分配内存
*   画出数字的凸起`hull`(**行 69**
*   放大`mask` ( **线 70**

这些凸包遮蔽操作的效果在**图 6** ( *上*)中描绘。通过**线 75** 计算`opening`和`mask`之间的按位 AND 清理我们的`opening`图像，并产生我们的`final`图像，该图像仅由*数字*组成，没有背景噪声**图 6** *(底部*)。

这就是我们的图像处理流程——我们现在有了一个清晰的图像，可以很好地处理宇宙魔方。让我们执行 OCR 并显示结果:

```py
# OCR the input image using Tesseract
options = "--psm 8 -c tessedit_char_whitelist=0123456789"
text = pytesseract.image_to_string(final, config=options)
print(text)

# show the final output image
cv2.imshow("Final", final)
cv2.waitKey(0)
```

使用我们从早期教程**、**中获得的关于宇宙魔方选项的知识，我们构建我们的`options`设置(**第 78 行**):

*   [**PSM**](https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/) **:** *“把图像当成一个单词”*
*   [**白名单**](https://pyimagesearch.com/2021/09/06/whitelisting-and-blacklisting-characters-with-tesseract-and-python/) **:** 数字*0–9*是唯一会出现在结果中的字符(没有符号或字母)

用我们的设置对图像进行 OCR 后(**第 79 行**，我们在终端上显示`text`，并在屏幕上保持所有管道步骤图像(包括`final`图像)，直到按下一个键(**第 83 和 84 行**)。

### **基本图像处理和镶嵌 OCR 结果**

让我们测试一下我们的图像处理程序。打开终端并启动`process_image.py`脚本:

```py
$ python process_image.py --image challenging_example.png
1214
```

**成功！**通过使用一些基本的图像处理和 OpenCV 库，我们能够清理我们的输入图像，然后使用宇宙魔方*正确地 OCR 它，即使*宇宙魔方不能 OCR 原始输入图像！

## **总结**

在本教程中，您了解到基本的图像处理可能是使用 Tesseract OCR 引擎获得足够 OCR 准确度的一个*要求*。

虽然在[之前的教程](https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/)中介绍的页面分割模式(PSM)在应用 Tesseract 时*极其重要*，但有时 PSM 本身不足以对图像进行 OCR。**通常，当输入图像的背景复杂，并且 Tesseract 的基本分割方法无法正确地将前景文本从背景中分割出来时，就会出现这种情况。**当这种情况发生时，你需要开始应用计算机视觉和图像处理来清理输入图像。

并非所有用于 OCR 的图像处理管道都是相同的。对于这个特殊的例子，我们可以结合使用阈值、距离变换和形态学操作。但是，您的示例图像可能需要针对这些操作或不同的图像处理操作进行额外的参数调整！

在我们的下一个教程中，您将学习如何使用拼写检查算法进一步改善 OCR 结果。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****