# 自动识别收据和扫描件

> 原文：<https://pyimagesearch.com/2021/10/27/automatically-ocring-receipts-and-scans/>

在本教程中，您将学习如何使用 Tesseract 和 OpenCV 来构建自动收据扫描仪。我们将使用 OpenCV 来构建系统的实际图像处理组件，包括:

*   检测图像中的收据
*   找到收据的四个角
*   最后，应用透视变换以获得收据的*自上而下*的鸟瞰图

**要学习如何**自动 OCR 收据和扫描**，*继续阅读。***

## **自动识别收据和扫描件**

在那里，我们将使用 Tesseract 对收据本身进行 OCR，并逐行解析出每件商品，包括商品描述和价格。

如果你是一个企业主(像我一样)，需要向你的会计师报告你的费用，或者如果你的工作要求你一丝不苟地跟踪你的报销费用，那么你就会知道跟踪你的收据是多么令人沮丧、乏味和烦人。很难相信在这个时代，购物仍然通过一张很小很脆弱的纸被跟踪！

也许在未来，跟踪和报告我们的支出会变得不那么繁琐。但在此之前，收据扫描仪可以节省我们大量的时间，并避免手动编目购买的挫折。

本教程的收据扫描仪项目是构建成熟的收据扫描仪应用程序的起点。以本教程为起点，然后通过添加 GUI、将其与移动应用程序集成等方式对其进行扩展。

我们开始吧！

### **学习目标**

在本教程中，您将学习:

*   如何使用 OpenCV 检测、提取和转换输入图像中的收据
*   如何使用 Tesseract 逐行识别收据
*   查看选择正确的 Tesseract 页面分段模式(PSM)如何产生更好结果的真实应用

### **用 OpenCV 和 Tesseract 对收据进行 OCR 识别**

在本教程的第一部分，我们将回顾收据扫描仪项目的目录结构。

然后，我们将逐行检查我们的收据扫描器实现。最重要的是，我将向您展示在构建收据扫描仪时使用哪种 Tesseract PSM，以便您可以轻松地从收据中检测并提取每个**商品**和**价格**。

最后，我们将讨论我们的结果来结束本教程。

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

从这里，看一下目录结构:

```py
|-- scan_receipt.py
|-- whole_foods.png
```

我们今天只需要查看一个脚本`scan_receipt.py`，它将包含我们的收据扫描器实现。

这张照片是我去美国连锁杂货店全食超市时拍的收据。我们将使用我们的`scan_receipt.py`脚本来检测输入图像中的收据，然后从收据中提取每个商品和价格。

### **实施我们的收据扫描仪**

在我们开始实现收据扫描器之前，让我们先回顾一下将要实现的基本算法。然后，当显示包含收据的输入图像时，我们将:

1.  应用边缘检测以显示背景下收据的轮廓(这假设我们在背景和前景之间有足够的对比度；否则，我们将无法检测到收据)
2.  检测边缘图中的轮廓
3.  循环遍历所有轮廓，找到具有四个顶点的最大轮廓(因为收据是矩形的，并且将有四个角)
4.  应用透视变换，生成收据的*自上而下*鸟瞰图(需要提高 OCR 准确度)
5.  将带有`--psm 4`的 Tesseract OCR 引擎应用于收据的*自顶向下的*转换，允许我们逐行 OCR 收据
6.  使用正则表达式解析出商品名称和价格
7.  最后，在我们的终端上显示结果

这听起来像许多步骤，但是正如您将看到的，我们可以在不到 120 行代码(包括注释)内完成所有这些步骤。

说完这些，让我们深入到实现中。打开项目目录结构中的`scan_receipt.py`文件，让我们开始工作:

```py
# import the necessary packages
from imutils.perspective import four_point_transform
import pytesseract
import argparse
import imutils
import cv2
import re
```

我们从第 2-7 行的**导入我们需要的 Python 包开始。**这些进口商品主要包括:

*   `four_point_transform`:应用透视变换获得输入 ROI 的*自上而下*鸟瞰图。在[之前的教程](https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/)中，我们在获取数独板的*自上而下*视图时使用了这个函数(这样我们就可以自动解出谜题)——今天我们将在这里做同样的事情，只是用收据代替了数独谜题。
*   `pytesseract`:提供一个到 Tesseract OCR 引擎的接口。
*   我们的 OpenCV 绑定
*   `re` : [Python 的正则表达式包](https://docs.python.org/3/library/re.html)将允许我们轻松解析出收据每一行的商品名称和相关价格。

接下来，我们有命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input receipt image")
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())
```

我们的脚本需要一个命令行参数，后跟一个可选参数:

*   `--image`:输入图像的路径包含我们想要进行 OCR 的收据(在本例中为`whole_foods.png`)。您也可以在这里提供您的收据图像。
*   `--debug`:一个整数值，用来表示我们是否要通过我们的流水线显示调试图像，包括边缘检测、回执检测等的输出。

特别是，如果您在输入图像中找不到收据，可能是因为边缘检测过程未能检测到收据的边缘:这意味着您需要微调 Canny 边缘检测参数或使用不同的方法(例如，阈值处理、霍夫线变换等。).另一种可能性是轮廓近似步骤未能找到收据的四个角。

如果发生这些情况，为`--debug`命令行参数提供一个正值将会显示步骤的输出，允许您调试问题，调整参数/算法，然后继续。

接下来，让我们从磁盘加载我们的`--input`图像，并检查它的空间维度:

```py
# load the input image from disk, resize it, and compute the ratio
# of the *new* width to the *old* width
orig = cv2.imread(args["image"])
image = orig.copy()
image = imutils.resize(image, width=500)
ratio = orig.shape[1] / float(image.shape[1])
```

在这里，我们从磁盘加载我们的原始(`orig`)映像，然后制作一个克隆。我们需要克隆输入图像，这样我们就有了应用透视变换的*原始*图像。但是，我们可以应用我们实际的图像处理操作(即，边缘检测、轮廓检测等)。)到`image`。

我们将`image`的宽度调整为 500 像素(从而作为一种降噪方式)，然后计算*新*宽度到*旧*宽度的`ratio`。最后，这个`ratio`值将用于对`orig`图像应用透视变换。

现在让我们开始将我们的图像处理流水线应用到`image`上:

```py
# convert the image to grayscale, blur it slightly, and then apply
# edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
edged = cv2.Canny(blurred, 75, 200)

# check to see if we should show the output of our edge detection
# procedure
if args["debug"] > 0:
	cv2.imshow("Input", image)
	cv2.imshow("Edged", edged)
	cv2.waitKey(0)
```

这里，我们通过将图像转换为灰度来执行边缘检测，使用`5x5` 高斯内核模糊它(以减少噪声)，然后使用 Canny 边缘检测器应用边缘检测。

如果我们设置了`--debug`命令行参数，我们将在屏幕上显示输入图像和输出边缘图。

**图 2** 显示了我们的输入图像*(左)*，接着是我们的输出边缘图*(右)。*注意我们的边缘图*如何在输入图像中清晰地显示出*收据的轮廓。

给定我们的边缘图，让我们检测`edged`图像中的轮廓并处理它们:

```py
# find contours in the edge map and sort them by size in descending
# order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
```

注意**第 42 行，**这里我们根据面积(大小)从大到小对轮廓进行排序。**这个排序步骤很重要，因为我们假设输入图像*中有四个角*的*最大轮廓*就是我们的收据。**

排序步骤满足了我们的第一个需求。但是我们如何知道我们是否找到了一个有四个顶点的轮廓呢？

下面的代码块回答了这个问题:

```py
# initialize a contour that corresponds to the receipt outline
receiptCnt = None

# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if our approximated contour has four points, then we can
	# assume we have found the outline of the receipt
	if len(approx) == 4:
		receiptCnt = approx
		break

# if the receipt contour is empty then our script could not find the
# outline and we should be notified
if receiptCnt is None:
	raise Exception(("Could not find receipt outline. "
		"Try debugging your edge detection and contour steps."))
```

**Line 45** 初始化一个变量来存储与我们的收据相对应的轮廓。然后我们开始在**线 48** 上循环所有检测到的轮廓。

**线 50 和 51** 通过减少点数来逼近轮廓，从而简化形状。

**第 55-57 行**检查我们是否找到了一个有四个点的轮廓。如果是这样，我们可以有把握地假设我们已经找到了收据，因为这是具有四个顶点的*最大轮廓*。一旦我们找到轮廓，我们将它存储在循环的`receiptCnt`和`break`中。

**第 61-63 行**为我们的脚本提供了一种优雅的退出方式，如果我们的收据没有找到的话。通常，当脚本的边缘检测阶段出现问题时，就会发生这种情况。由于照明条件不足或者仅仅是收据和背景之间没有足够的对比度，边缘图可能由于其中有间隙或孔洞而被“破坏”。

发生这种情况时，轮廓检测过程不会将收据“视为”四角对象。相反，它看到一个奇怪的多边形对象，因此没有检测到收据。

如果发生这种情况，一定要使用`--debug`命令行参数来直观地检查你的边缘贴图的输出。

找到收据轮廓后，让我们对图像应用透视变换:

```py
# check to see if we should draw the contour of the receipt on the
# image and then display it to our screen
if args["debug"] > 0:
	output = image.copy()
	cv2.drawContours(output, [receiptCnt], -1, (0, 255, 0), 2)
	cv2.imshow("Receipt Outline", output)
	cv2.waitKey(0)

# apply a four-point perspective transform to the *original* image to
# obtain a top-down bird's-eye view of the receipt
receipt = four_point_transform(orig, receiptCnt.reshape(4, 2) * ratio)

# show transformed image
cv2.imshow("Receipt Transform", imutils.resize(receipt, width=500))
cv2.waitKey(0)
```

**第 67-71 行**在我们的`output`图像上概述了在调试模式下的收据。然后，我们在屏幕上显示输出图像，以验证收据被正确检测到(**图 3** ，*左*)。

在**第 75 行完成一个*自上而下*的收据鸟瞰图。**注意，我们将变换应用于更高分辨率的`orig`图像— *这是为什么呢？*

首先，变量`image`已经应用了边缘检测和轮廓处理。使用透视变换`image`然后进行 OCR 不会得到正确的结果；我们得到的只有噪音。

**相反，我们寻求*高分辨率*版本的收据。**因此，我们将透视变换应用于`orig`图像。为此，我们需要将我们的`receiptCnt` ( *x，y* )坐标乘以我们的`ratio`，从而将坐标缩放回`orig`空间维度。

为了验证我们已经计算了*自上而下*，原始图像的鸟瞰图，我们在屏幕上的**第 78 行和 79 行** ( **图 3** 、*右*)显示了高分辨率收据。

给定收据的*自上而下*视图，我们现在可以对其进行 OCR:

```py
# apply OCR to the receipt image by assuming column data, ensuring
# the text is *concatenated across the row* (additionally, for your
# own images you may need to apply additional processing to cleanup
# the image, including resizing, thresholding, etc.)
options = "--psm 4"
text = pytesseract.image_to_string(
	cv2.cvtColor(receipt, cv2.COLOR_BGR2RGB),
	config=options)

# show the raw output of the OCR process
print("[INFO] raw output:")
print("==================")
print(text)
print("\n")
```

**第 85-88 行**使用 Tesseract 来 OCR 收据，以`--psm 4`模式传递。使用`--psm 4`允许我们逐行 OCR 收据。每行将包括项目名称和项目价格。

**第 91-94 行**显示应用 OCR 后的原始数据`text`。

然而，*的问题*是宇宙魔方不知道收据上的*项目*是什么，而只是杂货店的名称、地址、电话号码和你通常在收据上找到的所有其他信息。

这就提出了一个问题——我们如何解析出我们不需要的信息，只留下商品名称和价格？

答案是利用正则表达式:

```py
# define a regular expression that will match line items that include
# a price component
pricePattern = r'([0-9]+\.[0-9]+)'

# show the output of filtering out *only* the line items in the
# receipt
print("[INFO] price line items:")
print("========================")

# loop over each of the line items in the OCR'd receipt
for row in text.split("\n"):
	# check to see if the price regular expression matches the current
	# row
	if re.search(pricePattern, row) is not None:
		print(row)
```

如果您以前从未使用过正则表达式，它们是一种特殊的工具，允许我们定义文本模式。正则表达式库(在 Python 中，这个库是`re`)然后将所有文本匹配到这个模式。

**第 98 行**定义了我们的`pricePattern`。该模式将匹配任意数量的数字`0-9`，后跟`.`字符(表示价格值中的小数分隔符)，再后跟任意数量的数字`0-9` *。*

例如，这个`pricePattern`将匹配文本`$9.75`，但不会匹配文本`7600`，因为文本`7600`不包含小数点分隔符。

如果你是正则表达式的新手或者只是需要复习一下，我建议你阅读下面这个由 [RealPython](http://pyimg.co/blheh) 撰写的系列。

**第 106 行**分割我们的原始 OCR'd `text`并允许我们单独循环每一行。

对于每一行，我们检查`row`是否与我们的`pricePattern` ( **行 109** )匹配。如果是这样，我们知道已经找到了包含商品和价格的行，所以我们将该行打印到我们的终端(**第 110 行**)。

祝贺您构建了您的第一个收据扫描仪 OCR 应用程序！

### **收据扫描仪和 OCR 结果**

现在我们已经实现了我们的`scan_receipt.py`脚本，让我们把它投入工作。打开终端并执行以下命令:

```py
$ python scan_receipt.py --image whole_foods.png
[INFO] raw output:
==================
WHOLE
FOODS

WHOLE FOODS MARKET - WESTPORT, CT 06880
399 POST RD WEST - (203) 227-6858

365 BACON LS NP 4.99

365 BACON LS NP 4.99

365 BACON LS NP 4.99

365 BACON LS NP 4.99
BROTH CHIC NP 4.18

FLOUR ALMOND NP 11.99

CHKN BRST BNLSS SK NP 18.80
HEAVY CREAM NP 3 7

BALSMC REDUCT NP 6.49

BEEF GRND 85/15 NP 5.04
JUICE COF CASHEW C NP 8.99
DOCS PINT ORGANIC NP 14.49
HNY ALMOND BUTTER NP 9.99
eee TAX .00 BAL 101.33
```

在我们的终端中可以看到 Tesseract OCR 引擎的原始输出。通过指定`--psm 4`，Tesseract 能够逐行对收据进行光学字符识别，捕获两个项目:

1.  名称/描述
2.  价格

但是，输出中有一堆其他的“噪音”，包括杂货店的名称、地址、电话号码等。我们如何解析这些信息，只给我们留下商品和它们的价格？

答案是使用正则表达式，该表达式过滤具有类似于价格的数值的行，这些正则表达式的输出如下所示:

```py
[INFO] price line items:
========================
365 BACON LS NP 4.99
365 BACON LS NP 4.99
365 BACON LS NP 4.99
365 BACON LS NP 4.99
BROTH CHIC NP 4.18
FLOUR ALMOND NP 11.99
CHKN BRST BNLSS SK NP 18.80
BALSMC REDUCT NP 6.49
BEEF GRND 85/15 NP 5.04
JUICE COF CASHEW C NP 8.99
DOCS PINT ORGANIC NP 14.49
HNY ALMOND BUTTER NP 9.99
eee TAX .00 BAL 101.33
```

通过使用正则表达式，我们只提取了商品和价格，包括最终的应付余额。

我们的收据扫描仪应用程序是一个重要的实现，它展示了如何将 OCR 与一些文本处理结合起来提取感兴趣的数据。有一个完整的计算机科学领域致力于文本处理，称为自然语言处理(NLP)。

就像计算机视觉是对编写能够理解图像内容的软件的高级研究一样，NLP 也试图做同样的事情，只是针对文本。根据您尝试使用计算机视觉和 OCR 构建的内容，您可能需要花几周到几个月的时间来熟悉 NLP，这些知识将更好地帮助您理解如何处理从 OCR 引擎返回的文本。

## **总结**

在本教程中，您学习了如何使用 OpenCV 和 Tesseract 实现一个基本的收据扫描器。我们的收据扫描仪实施需要基本的图像处理操作来检测收据，包括:

*   边缘检测
*   轮廓检测
*   使用弧长和近似值的轮廓滤波

从那里，我们使用 Tesseract，最重要的是，`--psm 4`，来 OCR 收据。通过使用`--psm 4`，我们从收据中一行一行地提取每个项目，包括项目名称和特定项目的成本。

我们的收据扫描仪最大的局限性是它需要:

1.  收据和背景之间有足够的对比
2.  收据的所有四个角在图像中都可见

如果这些情况不成立，我们的脚本将找不到收据。

### **引用信息**

**Rosebrock，A.** “自动 OCR 识别收据和扫描”， *PyImageSearch* ，2021，[https://PyImageSearch . com/2021/10/27/Automatically-OCR ing-Receipts-and-Scans/](https://pyimagesearch.com/2021/10/27/automatically-ocring-receipts-and-scans/)

`@article{Rosebrock_2021_Automatically, author = {Adrian Rosebrock}, title = {Automatically {OCR}’ing Receipts and Scans}, journal = {PyImageSearch}, year = {2021}, note = {https://pyimagesearch.com/2021/10/27/automatically-ocring-receipts-and-scans/}, }`

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****