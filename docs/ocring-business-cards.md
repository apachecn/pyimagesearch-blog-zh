# 用光学字符识别名片

> 原文：<https://pyimagesearch.com/2021/11/03/ocring-business-cards/>

在[之前的教程](https://pyimagesearch.com/2021/10/27/automatically-ocring-receipts-and-scans/)中，我们学习了如何通过以下方式自动 OCR 和扫描收据:

1.  检测输入图像中的收据
2.  应用透视变换以获得收据的*自上而下的*视图
3.  利用宇宙魔方来识别收据上的文字
4.  使用正则表达式提取价格数据

****要学习如何使用 Python 对名片进行光学字符识别，*继续阅读。*****

## **OCR 识别名片**

在本教程中，我们将使用一个非常相似的工作流程，但这次将其应用于名片 OCR。更具体地说，我们将学习如何从名片中提取姓名、头衔、电话号码和电子邮件地址。

然后，您将能够将这个实现扩展到您的项目中。

### **学习目标**

在本教程中，您将:

*   了解如何检测图像中的名片
*   对名片图像应用 OCR
*   利用正则表达式提取:
    *   名字
    *   职称
    *   电话号码
    *   电子邮件地址

### **名片 OCR**

在本教程的第一部分，我们将回顾我们的项目目录结构。然后，我们将实现一个简单而有效的 Python 脚本，允许我们对名片进行 OCR。

我们将通过讨论我们的结果以及后续步骤来结束本教程。

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
|-- larry_page.png
|-- ocr_business_card.py
|-- tony_stark.png
```

我们只需要回顾一个 Python 脚本，`ocr_business_card.py`。这个脚本将加载示例名片图像(即`larry_page.png`和`tony_stark.png`)，对它们进行 OCR，然后输出名片上的姓名、职务、电话号码和电子邮件地址。

最棒的是，我们将能够在 120 行代码(包括注释)内完成我们的目标！

### **实现名片 OCR**

我们现在准备实现我们的名片 OCR 脚本！首先，在我们的项目目录结构中打开`ocr_business_card.py`文件，并插入以下代码:

```py
# import the necessary packages
from imutils.perspective import four_point_transform
import pytesseract
import argparse
import imutils
import cv2
import re
```

我们这里的导入类似于之前关于 OCR 收据的教程中的导入。

我们需要我们的`four_point_transform`函数来获得一个*自顶向下的*，名片的鸟瞰图。获得此视图通常会产生更高的 OCR 准确性。

`pytesseract`包用于与 Tesseract OCR 引擎接口。然后我们有了 Python 的正则表达式库`re`，它将允许我们解析名片中的姓名、职位、电子邮件地址和电话号码。

导入工作完成后，我们可以转到命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="whether or not we are visualizing each step of the pipeline")
ap.add_argument("-c", "--min-conf", type=int, default=0,
	help="minimum confidence value to filter weak text detection")
args = vars(ap.parse_args())
```

我们的第一个命令行参数`--image`，是磁盘上输入图像的路径。我们假设该图像包含一张在前景和背景之间具有足够对比度的名片，确保我们可以成功地应用边缘检测和轮廓处理来提取名片。

然后我们有两个可选的命令行参数，`--debug`和`--min-conf`。`--debug`命令行参数用于指示我们是否正在调试我们的图像处理管道，并在我们的屏幕上显示更多处理过的图像(当您无法确定为什么会检测到名片时，这很有用)。

然后我们有了`--min-conf`，成功文本检测所需的最小置信度(在 0-100 的范围内)。您可以增加`--min-conf`来删除弱文本检测。

现在让我们从磁盘加载输入图像:

```py
# load the input image from disk, resize it, and compute the ratio
# of the *new* width to the *old* width
orig = cv2.imread(args["image"])
image = orig.copy()
image = imutils.resize(image, width=600)
ratio = orig.shape[1] / float(image.shape[1])
```

这里，我们从磁盘加载我们的输入`--image`，然后克隆它。我们把它做成一个克隆体，提取轮廓处理后的原高分辨率版名片。

然后，我们将`image`的宽度调整为 600 像素，然后计算*新的*宽度与*旧的*宽度的比率(当我们想要获得原始高分辨率名片的*自上而下的*视图时，这是一个要求)。

下面我们继续我们的图像处理流程。

```py
# convert the image to grayscale, blur it, and apply edge detection
# to reveal the outline of the business card
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)

# detect contours in the edge map, sort them by size (in descending
# order), and grab the largest contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# initialize a contour that corresponds to the business card outline
cardCnt = None
```

首先，我们采取我们的原始`image`然后将其转换为灰度，模糊，然后应用边缘检测，其结果可以在**图 2** 中看到。

请注意，名片的轮廓/边框在边缘图上是可见的。然而，假设在边缘图中有任何间隙。在这种情况下，名片将*而不是*通过我们的轮廓处理技术检测出来，所以你可能需要调整 Canny 边缘检测器的参数，或者在光线条件更好的环境中捕捉你的图像。

从那里，我们检测轮廓，并根据计算出的轮廓面积按降序(从大到小)对它们进行排序。我们在这里的假设是，名片轮廓将是最大的检测轮廓之一，因此这个操作。

我们还初始化`cardCnt` ( **Line 40** )，这是对应名片的轮廓。

现在让我们循环最大的等高线:

```py
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# if this is the first contour we've encountered that has four
	# vertices, then we can assume we've found the business card
	if len(approx) == 4:
		cardCnt = approx
		break

# if the business card contour is empty then our script could not
# find the  outline of the card, so raise an error
if cardCnt is None:
	raise Exception(("Could not find receipt outline. "
		"Try debugging your edge detection and contour steps."))
```

**第 45 和 46 行**执行轮廓近似。

如果我们的近似轮廓有四个顶点，那么我们可以假设我们找到了名片。如果发生这种情况，我们从循环中退出`break`，并更新我们的`cardCnt`。

如果我们到达了`for`循环的末尾，仍然没有找到有效的`cardCnt`，我们优雅地退出脚本。请记住，如果在图像中找不到名片，我们将无法处理它！

我们的下一个代码块处理显示一些调试图像以及获取名片的*自顶向下*视图:

```py
# check to see if we should draw the contour of the business card
# on the image and then display it to our screen
if args["debug"] > 0:
	output = image.copy()
	cv2.drawContours(output, [cardCnt], -1, (0, 255, 0), 2)
	cv2.imshow("Business Card Outline", output)
	cv2.waitKey(0)

# apply a four-point perspective transform to the *original* image to
# obtain a top-down bird's-eye view of the business card
card = four_point_transform(orig, cardCnt.reshape(4, 2) * ratio)

# show transformed image
cv2.imshow("Business Card Transform", card)
cv2.waitKey(0)
```

**第 62-66 行**检查我们是否处于`--debug`模式，如果是，我们在`output`图像上绘制名片的轮廓。

然后，我们将四点透视变换应用于原始的高分辨率图像，从而获得名片的俯视鸟瞰图，即 T2。

我们将`cardCnt`乘以我们计算的`ratio`,因为`cardCnt`是针对缩减的图像尺寸计算的。乘以`ratio`将`cardCnt`缩放回`orig`图像的尺寸。

然后，我们在屏幕上显示转换后的图像(**第 73 行和第 74 行**)。

有了我们获得的名片的*自上而下*视图，我们可以继续进行 OCR:

```py
# convert the business card from BGR to RGB channel ordering and then
# OCR it
rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
text = pytesseract.image_to_string(rgb)

# use regular expressions to parse out phone numbers and email
# addresses from the business card
phoneNums = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text)

# attempt to use regular expressions to parse out names/titles (not
# necessarily reliable)
nameExp = r"^[\w'\-,.][^0-9_!¡?÷?¿/\\+=@#$%ˆ&*(){}|~<>;:[\]]{2,}"
names = re.findall(nameExp, text)
```

**第 78 行和第 79 行** OCR 名片，产生`text`输出。

但问题是，我们如何从名片本身提取信息？答案是利用正则表达式。

**第 83 行和第 84 行**利用正则表达式从`text`中提取电话号码和电子邮件地址([瓦利亚，2020](https://dev.to/abhiwalia15/how-to-extract-email-phone-number-from-a-business-card-using-python-opencv-and-tesseractocr-1a6h) )，而**第 88 行和第 89 行**对姓名和职务做同样的处理( [*名和姓的正则表达式*，2020](https://stackoverflow.com/questions/2385701/regular-expression-for-first-and-last-name) )。

对正则表达式的回顾超出了本教程的范围，但是要点是它们可以用来匹配文本中的特定模式。

例如，电话号码由特定的数字模式组成，有时还包括破折号和括号。电子邮件地址也遵循一种模式，包括一个文本字符串，后跟一个“@”符号，然后是域名。

只要你能可靠地保证文本的模式，正则表达式就能很好地工作。也就是说，它们也不是完美的，所以如果你发现你的名片 OCR 准确率明显下降，你可能需要研究更高级的自然语言处理(NLP)算法。

这里的最后一步是向终端显示我们的输出:

```py
# show the phone numbers header
print("PHONE NUMBERS")
print("=============")

# loop over the detected phone numbers and print them to our terminal
for num in phoneNums:
	print(num.strip())

# show the email addresses header
print("\n")
print("EMAILS")
print("======")

# loop over the detected email addresses and print them to our
# terminal
for email in emails:
	print(email.strip())

# show the name/job title header
print("\n")
print("NAME/JOB TITLE")
print("==============")

# loop over the detected name/job titles and print them to our
# terminal
for name in names:
	print(name.strip())
```

这个最后的代码块循环遍历提取的电话号码(**第 96 和 97 行**)、电子邮件地址(**第 106 和 107 行**)和姓名/职务(**第 116 和 117 行**)，将它们显示到我们的终端。

当然，您可以提取这些信息，写入磁盘，保存到数据库，等等。不过，为了简单起见(并且不知道您的名片 OCR 项目规范)，我们将把它作为一个练习留给您，让您保存您认为合适的数据。

### **名片 OCR 结果**

我们现在准备将 OCR 应用于名片。打开终端并执行以下命令:

```py
$ python ocr_business_card.py --image tony_stark.png --debug 1
PHONE NUMBERS
=============
562-555-0100
562-555-0150

EMAILS
======

NAME/JOB TITLE
==============
Tony Stark
Chief Executive Officer

Stark Industries
```

**图 3** *(上)*展示了我们名片本地化的结果。请注意我们是如何在输入图像中正确检测出名片的。

从那里，**图 3** *(底部)*显示应用名片的透视变换的结果，因此产生了*自上而下的*，图像的鸟瞰图。

一旦我们有了图像的*自上而下*视图(通常需要获得更高的 OCR 准确度)，我们就可以应用 Tesseract 对其进行 OCR，其结果可以在上面的终端输出中看到。

请注意，我们的脚本已经成功提取了托尼·斯塔克名片上的两个电话号码。

没有报告电子邮件地址，因为名片上没有电子邮件地址。

然后，我们还会显示姓名和职位。有趣的是，我们可以成功地对所有文本进行 OCR，因为姓名文本比电话号码文本更失真。我们的透视变换有效地处理了所有的文本，即使当你离相机越远，扭曲的程度就越大。这就是透视变换的要点，也是它对我们的 OCR 的准确性如此重要的原因。

让我们尝试另一个例子图片，这是一个老拉里·佩奇(谷歌的联合创始人)的名片:

```py
$ python ocr_business_card.py --image larry_page.png --debug 1
PHONE NUMBERS
=============
650 330-0100
650 618-1499

EMAILS
======
larry@google.com

NAME/JOB TITLE
==============
Larry Page
CEO

Google
```

**图 4** *(上)*显示本地化页面名片的输出。下面的*按钮*显示了图像自上而下的*变换。*

这个*自顶向下的*转换通过 Tesseract OCR，产生 OCR 处理的文本作为输出。我们获取经过 OCR 处理的文本，应用正则表达式，从而获得上面的结果。

检查结果，您可以看到我们已经成功地从名片中提取了 Larry Page 的两个电话号码、电子邮件地址和姓名/职务。

## **总结**

在本教程中，您学习了如何构建一个基本的名片 OCR 系统。本质上，这个系统是我们收据扫描器的扩展，但是有不同的正则表达式和文本本地化策略。

如果您需要构建一个名片 OCR 系统，我建议您以本教程为起点，但是请记住，您可能希望利用更高级的文本后处理技术，例如真正的自然语言处理(NLP)算法，而不是正则表达式。

对于电子邮件地址和电话号码，正则表达式可以很好地工作*，但是对于姓名和职位可能无法获得高准确性。如果到了那个时候，你应该考虑尽可能地利用 NLP 来改善你的结果。*

 *### **引用信息**

**罗斯布鲁克，a .**“OCR 识别名片”， *PyImageSearch* ，2021，【https://pyimagesearch.com/2021/11/03/ocring-business-cards/】T4

`@article{Rosebrock_2021_OCR_BCards, author = {Adrian Rosebrock}, title = {{OCR}’ing Business Cards}, journal = {PyImageSearch}, year = {2021}, note = {https://pyimagesearch.com/2021/11/03/ocring-business-cards/}, }`

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，** ***只需在下面的表格中输入您的电子邮件地址！*****