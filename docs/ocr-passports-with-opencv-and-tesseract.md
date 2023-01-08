# 带 OpenCV 和宇宙魔方的 OCR 护照

> 原文：<https://pyimagesearch.com/2021/12/01/ocr-passports-with-opencv-and-tesseract/>

本课是关于 OCR 120 的 4 部分系列的第 4 部分:

1.  [Tesseract 页面分割模式(PSMs)讲解:如何提高你的 OCR 准确率](https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/)(2 周前的教程)
2.  [通过基本图像处理改善 OCR 结果](https://pyimagesearch.com/2021/11/22/improving-ocr-results-with-basic-image-processing/)(上周教程)
3.  [使用拼写检查来提高 Tesseract OCR 的准确性](https://pyimagesearch.com/2021/11/29/using-spellchecking-to-improve-tesseract-ocr-accuracy/)(之前的教程)
4.  带 OpenCV 和 Tesseract 的 OCR 护照(今天的教程)

**要学习如何使用 OpenCV 和 Tesseract 对护照进行光学字符识别，** ***继续阅读。***

## **带 OpenCV 和宇宙魔方的 OCR 护照**

到目前为止，在本课程中，我们一直依赖于 Tesseract OCR 引擎来*检测*输入图像中的文本。然而，正如我们在[之前的教程](https://pyimagesearch.com/2021/11/22/improving-ocr-results-with-basic-image-processing/)、**中发现的，有时宇宙魔方需要一点帮助*，在*之前我们实际上可以 OCR 文本。**

本教程将进一步探讨这一想法，展示计算机视觉和图像处理技术可以在复杂的输入图像中定位文本区域。一旦文本被定位，我们可以从输入图像中提取文本 ROI，然后使用 Tesseract 对其进行 OCR。

作为一个案例研究，我们将开发一个计算机视觉系统，它可以在扫描护照时自动定位机器可读区域。MRZ 包含护照持有人的姓名、护照号码、国籍、出生日期、性别和护照到期日期等信息。

通过自动识别该区域，我们可以帮助运输安全管理局(TSA)代理和移民官员更快地处理旅客，减少长队(更不用说排队等候的压力和焦虑)。

## **学习目标**

在本教程中，您将:

1.  了解如何使用图像处理技术和 OpenCV 库来本地化输入图像中的文本
2.  提取本地化的文本并用 Tesseract 进行 OCR
3.  构建一个示例 passport reader 项目，它可以自动检测、提取和 OCR 护照图像中的 MRZ

## **用图像处理在图像中寻找文本**

在本教程的第一部分，我们将简要回顾什么是护照 MRZ。从那以后，我将向您展示如何实现一个 Python 脚本来从输入图像中检测和提取 MRZ。一旦 MRZ 被提取出来，我们就可以用宇宙魔方来识别 MRZ。

### 什么是机器可读区域？

护照是一种旅行证件，看起来像一个小笔记本。此文件由您所在国家的政府签发，包含识别您个人身份的信息，包括您的姓名、地址等。

你通常在国际旅行时使用你的护照。一旦你到达目的地国家，移民官员会检查你的护照，确认你的身份，并在你的护照上盖你的到达日期。

在你的护照里，你会找到你的个人身份信息(**图 1** )。如果你看护照的*底部*，你会看到 2-3 行等宽字符。

1 类护照有三行，每行 30 个字符，而 3 类护照有两行，每行 44 个字符。

这些线被称为护照上的 MRZ。

MRZ 对您的个人识别信息进行编码，包括:

*   名字
*   护照号码
*   国籍
*   出生日期/年龄
*   性
*   护照到期日期

在电脑和核磁共振成像系统出现之前，美国运输安全管理局和移民官员必须检查你的护照，并繁琐地验证你的身份。这是一项耗时的任务，对官员来说很单调，对在长长的移民队伍中耐心等待的游客来说很沮丧。

MRZs 允许 TSA 代理快速扫描您的信息，验证您的身份，并使您能够更快地通过队列，从而减少队列长度(并减轻旅客和官员的压力)。

在本教程的其余部分，您将学习如何用 OpenCV 和 Tesseract 实现一个自动护照 MRZ 扫描仪。

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

我们首先需要回顾我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

在我们构建 MRZ 阅读器和扫描护照图像之前，让我们先回顾一下这个项目的目录结构:

```py
|-- passports
|   |-- passport_01.png
|   |-- passport_02.png
|-- ocr_passport.py
```

我们这里只有一个 Python 脚本，`ocr_passport.py`，顾名思义，它用于从磁盘加载护照图像并扫描它们。

在`passports`目录中，我们有两个图像，`passport_01.png`和`passport_02.png`——这些图像包含扫描的护照样本。我们的`ocr_passport.py`脚本将从磁盘加载这些图像，定位它们的 MRZ 地区，然后对它们进行 OCR。

### **在护照图像中定位 MRZs】**

让我们学习如何使用 OpenCV 和图像处理来定位护照图像的 MRZ。

打开项目目录结构中的`ocr_passport.py`文件，插入以下代码:

```py
# import the necessary packages
from imutils.contours import sort_contours
import numpy as np
import pytesseract
import argparse
import imutils
import sys
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
args = vars(ap.parse_args())
```

我们从第 2-8 行的**开始，导入我们需要的 Python 包。到本文的这一点，你应该开始觉得这些导入非常标准了。唯一的例外可能是第 2** 行**上的`sort_contours`导入——这个函数做什么？**

`sort_contours`函数将接受通过使用 OpenCV 的`cv2.findContours`函数找到的一组输入轮廓。然后，`sort_contours`将对这些轮廓进行水平排序(*从左到右*或*从右到左*)或垂直排序(*从上到下*或*从下到上*)。

我们执行这个排序操作，因为 OpenCV 的`cv2.findContours`不能保证轮廓的排序。我们需要*对它们进行显式排序*以访问护照图像底部*的*的 MRZ 行。执行该排序操作将使得检测 MRZ 区域*远*更容易(正如我们将在该实现中稍后看到的)。

**第 11-14 行**解析我们的命令行参数。这里只需要一个参数，即输入路径`--image`。

完成导入和命令行参数后，我们可以继续加载输入图像，并为 MRZ 检测做准备:

```py
# load the input image, convert it to grayscale, and grab its
# dimensions
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(H, W) = gray.shape

# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

# smooth the image using a 3x3 Gaussian blur and then apply a
# blackhat morpholigical operator to find dark regions on a light
# background
gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
cv2.imshow("Blackhat", blackhat)
```

**第 18 行和第 19 行**从磁盘加载我们的输入`image`，然后将其转换为灰度，这样我们就可以对其应用基本的图像处理例程(再次提醒，记住我们的目标是检测护照的 MRZ*，而*不必利用机器学习)。然后我们在第 20 行**获取输入图像的空间尺寸(宽度和高度)。**

**第 23 行和第 24 行**初始化了两个内核，我们稍后将在应用形态学操作时使用它们，特别是关闭操作。目前，请注意第一个内核是矩形的，宽度大约是高度的 3 倍。第二个内核是方形的。这些内核将允许我们填补 MRZ 字符之间的空白和 MRZ 线之间的空白。

在**行 29** 应用高斯模糊以减少高频噪声。然后，我们对第 30 行**的模糊灰度图像应用 blackhat 形态学操作。**

blackhat 算子用于显示亮背景(即护照背景)下的暗区域(即 MRZ 文本)。由于护照文本在浅色背景下总是黑色的(至少在这个数据集中)，所以 blackhat 操作是合适的。**图 3** 显示了应用 blackhat 算子的输出。

在**图 3** 中，*左侧*显示我们的原始输入图像，而*右侧*显示 blackhat 操作的输出。请注意，在此操作之后，文本是可见的，而大部分背景噪声已被移除。

MRZ 检测的下一步是使用 Scharr 算子计算 blackhat 图像的梯度幅度表示:

```py
# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
grad = np.absolute(grad)
(minVal, maxVal) = (np.min(grad), np.max(grad))
grad = (grad - minVal) / (maxVal - minVal)
grad = (grad * 255).astype("uint8")
cv2.imshow("Gradient", grad)
```

**第 35 行和第 36 行**计算沿着 blackhat 图像的 *x* 轴的 Scharr 梯度，显示图像中在亮背景下较暗的区域，并包含梯度中的垂直变化，例如 MRZ 文本区域。然后，我们使用最小/最大缩放(**第 37-39 行**)将该渐变图像缩放回范围`[0, 255]` 。生成的梯度图像随后显示在我们的屏幕上(**图 4** )。

下一步是尝试检测 MRZ 的实际*线*:

```py
# apply a closing operation using the rectangular kernel to close
# gaps in between letters -- then apply Otsu's thresholding method
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(grad, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Rect Close", thresh)

# perform another closing operation, this time using the square
# kernel to close gaps between lines of the MRZ, then perform a
# series of erosions to break apart connected components
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
thresh = cv2.erode(thresh, None, iterations=2)
cv2.imshow("Square Close", thresh)
```

首先，我们使用矩形内核应用一个关闭操作(**第 44-46 行**)。这种闭合操作是为了闭合 MRZ 字符之间的间隙。然后，我们使用 Otsu 方法应用阈值处理来自动对图像进行阈值处理(**图 4** )。正如我们所看到的，每条 MRZ 线都出现在我们的阈值图上。

然后，我们用正方形内核(**第 52 行**)进行闭合操作，来闭合实际线条之间的间隙。`sqKernel`是一个`21 x 21` 内核，它试图闭合线条之间的间隙，产生一个对应于 MRZ 的大矩形区域。

然后进行一系列腐蚀，以分离在闭合操作期间可能已经结合的连接部件(**线 53** )。这些侵蚀也有助于去除与 MRZ 无关的小斑点。

这些操作的结果可以在**图 4** 中看到。注意 MRZ 区域是如何在图像底部三分之一的*中成为一个大的矩形斑点。*

现在我们的 MRZ 区域可见了，让我们在`thresh`图像中找到轮廓——这个过程将允许我们检测和提取 MRZ 区域:

```py
# find contours in the thresholded image and sort them from bottom
# to top (since the MRZ will always be at the bottom of the passport)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="bottom-to-top")[0]

# initialize the bounding box associated with the MRZ
mrzBox = None
```

**第 58-61 行**检测阈值图像中的轮廓。然后我们从下到上对它们进行排序*。你可能会问，为什么*自下而上*？*

 *简单:MRZ 地区总是位于输入护照图像的底部三分之一处。我们使用这个*先验*知识来开发图像的结构。如果我们知道我们正在寻找一个大的矩形区域，*总是*出现在图像的*底部*， ***为什么不先搜索底部呢？***

每当应用图像处理操作时，总是要看看是否有一种方法可以利用你对问题的了解。不要让你的图像处理管道过于复杂。使用任何领域知识使问题变得简单。

第 64 行然后初始化`mrzBox`，与 MRZ 区域相关的边界框。

我们将尝试在下面的代码块中找到`mrzBox`:

```py
# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then derive the
	# how much of the image the bounding box occupies in terms of
	# both width and height
	(x, y, w, h) = cv2.boundingRect(c)
	percentWidth = w / float(W)
	percentHeight = h / float(H)

	# if the bounding box occupies > 80% width and > 4% height of the
	# image, then assume we have found the MRZ
	if percentWidth > 0.8 and percentHeight > 0.04:
		mrzBox = (x, y, w, h)
		break
```

我们在**线 67** 上的检测轮廓上开始一个循环。我们计算每个轮廓的边界框，然后确定边界框在图像中所占的百分比(**第 72 行和第 73 行**)。

我们计算边界框有多大(相对于原始输入图像),以过滤我们的轮廓。请记住，我们的 MRZ 是一个很大的矩形区域，几乎跨越护照的整个宽度。

因此，**行 77** 通过确保检测到的边界框跨越*至少*图像宽度的 80%以及高度的 4%来利用这一知识。假设当前边界框区域通过了这些测试，我们从循环中更新我们的`mrzBox`和`break`。

我们现在可以继续处理 MRZ 地区本身:

```py
# if the MRZ was not found, exit the script
if mrzBox is None:
	print("[INFO] MRZ could not be found")
	sys.exit(0)

# pad the bounding box since we applied erosions and now need to
# re-grow it
(x, y, w, h) = mrzBox
pX = int((x + w) * 0.03)
pY = int((y + h) * 0.03)
(x, y) = (x - pX, y - pY)
(w, h) = (w + (pX * 2), h + (pY * 2))

# extract the padded MRZ from the image
mrz = image[y:y + h, x:x + w]
```

**第 82-84 行**处理没有找到 MRZ 地区的情况——这里，我们退出脚本。如果*没有*包含护照的图像意外地通过脚本，或者如果护照图像质量低/噪音太大，我们的基本图像处理管道无法处理，就会发生这种情况。

假设我们*确实找到了 MRZ，下一步就是填充边界框区域。我们进行这种填充是因为我们在试图检测 MRZ 本身时应用了一系列腐蚀(回到**线 53** )。*

但是，我们需要填充这个区域，这样 MRZ 字符就不会碰到 ROI 的边界。如果字符接触到图像的边界，Tesseract 的 OCR 程序可能不准确。

**第 88 行**解包边界框坐标。然后我们在每个方向上填充 MRZ 地区 3%(**第 89-92 行**)。

一旦 MRZ 被填充，我们使用数组切片将其从图像中提取出来(**行 95** )。

提取 MRZ 后，最后一步是应用 Tesseract 对其进行 OCR:

```py
# OCR the MRZ region of interest using Tesseract, removing any
# occurrences of spaces
mrzText = pytesseract.image_to_string(mrz)
mrzText = mrzText.replace(" ", "")
print(mrzText)

# show the MRZ image
cv2.imshow("MRZ", mrz)
cv2.waitKey(0)
```

**Line 99** OCRs 识别护照上的 MRZ 地区。然后，我们显式地从 MRZ 文本(**第 100 行**)中删除任何空格，因为 Tesseract 可能在 OCR 过程中意外地引入了空格。

然后，我们通过在终端上显示 OCR 的`mrzText`并在屏幕上显示最终的`mrz` ROI 来完成我们的护照 OCR 实现。你可以在**图 5** 中看到结果。

### **文本斑点定位结果**

我们现在准备测试我们的文本本地化脚本。

打开终端并执行以下命令:

```py
$ python ocr_passport.py --image passports/passport_01.png
P<GBRJENNINGS<<PAUL<MICHAEL<<<<<<<<<<<<<<<<<
0123456784GBR5011025M0810050<<<<<<<<<<<<<<00
```

**图 6** *(左)*显示我们的原始输入图像，而**图 6** *(右)*显示通过我们的图像处理管道提取的 MRZ。我们的终端输出显示，我们已经使用宇宙魔方正确地识别了 MRZ 地区。

让我们尝试另一个护照图像，这是一个有三条 MRZ 线而不是两条的 Type-1 护照:

```py
$ python ocr_passport.py --image passports/passport_02.png
IDBEL590335801485120100200<<<<
8512017F0901015BEL<<<<<<<<<<<7
REINARTZ<<ULRIKE<KATLIA<E<<<<<<
```

如图 7 所示，我们检测了输入图像中的 MRZ，然后提取出来。然后，MRZ 被传递到 Tesseract 进行 OCR，我们的终端输出显示了结果。

然而，我们的 MRZ OCR 并不是 100%准确——注意在*“KATIA”中的*【T】*和*【I】*之间有一个*【L】**

为了获得更高的 OCR 准确性，我们应该考虑在护照中使用的字体上专门训练一个定制的 Tesseract 模型*，使 Tesseract 更容易识别这些字符。*

 *## **总结**

在本教程中，您学习了如何实现 OCR 系统，该系统能够本地化、提取和 OCR 护照 MRZ 中的文本。

当您构建自己的 OCR 应用程序时，不要盲目地将 Tesseract 扔向它们，看看什么能坚持下来。而是作为一个计算机视觉从业者仔细审视问题。

问问你自己:

*   我可以使用图像处理来定位图像中的文本，从而减少对 Tesseract 文本定位的依赖吗？
*   我可以使用 OpenCV 函数自动提取这些区域吗？
*   检测文本需要哪些图像处理步骤？

本教程中介绍的图像处理管道是您可以构建的文本本地化管道的*示例*。它将*而不是*在所有情况下都有效。尽管如此，计算梯度和使用形态学操作来填补文本中的空白将在数量惊人的应用中发挥作用。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******