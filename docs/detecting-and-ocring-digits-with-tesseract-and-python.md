# 用 Tesseract 和 Python 检测和识别数字

> 原文：<https://pyimagesearch.com/2021/08/30/detecting-and-ocring-digits-with-tesseract-and-python/>

在之前的教程中，我们实现了第一个 OCR 项目。我们看到，Tesseract 在一些图像上工作得很好，但在其他例子中却完全没有意义。成为一名成功的 OCR 从业者的一部分是了解当您看到 Tesseract 的这种乱码、无意义的输出时，这意味着(1)您的图像预处理技术和(2)您的 Tesseract OCR 选项是不正确的。

****要学习如何用宇宙魔方和 Python 检测和 OCR 数字，*只要坚持读下去。*****

## **用宇宙魔方和 Python 检测和识别数字**

宇宙魔方是一种工具，就像任何其他软件包一样。就像一个数据科学家不能简单地将数百万条客户购买记录导入 Microsoft Excel 并期望 Excel 自动识别购买模式*，**期望 Tesseract 找出你需要的东西并自动正确地输出它是不现实的。***

 *相反，如果您了解如何为手头的任务正确配置 Tesseract 会有所帮助。例如，假设你的任务是创建一个计算机视觉应用程序来自动 OCR 名片上的电话号码。

你将如何着手建设这样一个项目？你会尝试对整张*名片进行 OCR，然后结合使用正则表达式和后处理模式识别来解析出数字吗？*

或者，您会后退一步，检查宇宙魔方 OCR 引擎本身— **是否可以将宇宙魔方仅告知*****OCR 位数？***

 *原来是有的。这就是我们将在本教程中讨论的内容。

## **学习目标**

在本教程中，您将:

1.  获得从输入图像中识别数字的实践经验
2.  扩展我们以前的 OCR 脚本来处理数字识别
3.  了解如何将 Tesseract 配置为仅支持 OCR 数字
4.  通过`pytesseract`库将这个配置传递给 Tesseract

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

## **用魔方进行数字检测和识别**

在本教程的第一部分，我们将回顾数字检测和识别，包括现实世界中我们可能希望仅 OCR *数字的问题。*

从那里，我们将回顾我们的项目目录结构，我将向您展示如何使用 Tesseract 执行数字检测和识别。我们将以回顾我们的数字 OCR 结果来结束本教程。

### 什么是数字检测和识别？

顾名思义，数字识别是仅识别*数字，有意忽略其他字符的过程。数字识别经常应用于现实世界的 OCR 项目(可以在**图** **2** 中看到其蒙太奇)，包括:*

*   从名片中提取信息
*   构建智能水监控阅读器
*   银行支票和信用卡 OCR

我们的目标是“消除非数字字符的干扰”。我们将改为“激光输入”数字。幸运的是，一旦我们为 Tesseract 提供了正确的参数，完成这个数字识别任务就相对容易了。

### **项目结构**

让我们回顾一下该项目的目录结构:

```py
|-- apple_support.png
|-- ocr_digits.py
```

我们的项目由一个测试图像(`apple_support.png`)和我们的`ocr_digits.py` Python 脚本组成。该脚本接受图像和可选的“仅数字”设置，并相应地报告 OCR 结果。

### **使用 Tesseract 和 OpenCV 对数字进行 OCR 识别**

我们现在已经准备好用宇宙魔方来识别数字了。打开一个新文件，将其命名为`ocr_digits.py`，并插入以下代码:

```py
# import the necessary packages
import pytesseract
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-d", "--digits", type=int, default=1,
	help="whether or not *digits only* OCR will be performed")
args = vars(ap.parse_args())
```

如您所见，我们将 PyTesseract 包与 OpenCV 结合使用。处理完导入后，我们解析两个命令行参数:

*   `--image`:要进行 OCR 的图像的路径
*   `--digits`:一个标志，指示我们是否应该仅 OCR *位的*(通过`default`，该选项被设置为一个`True`布尔)

让我们继续加载我们的图像并执行 OCR:

```py
# load the input image, convert it from BGR to RGB channel ordering,
# and initialize our Tesseract OCR options as an empty string
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
options = ""

# check to see if *digit only* OCR should be performed, and if so,
# update our Tesseract OCR options
if args["digits"] > 0:
	options = "outputbase digits"

# OCR the input image using Tesseract
text = pytesseract.image_to_string(rgb, config=options)
print(text)
```

Tesseract 需要 RGB 颜色通道排序来执行 OCR。**第 16 和 17 行**加载输入`--image`并相应地交换颜色通道。

然后我们建立我们的宇宙魔方`options` ( **第 18–23 行**)。使用选项配置 Tesseract 允许对 Tesseract 执行 OCR 的方法进行更细粒度的控制。

目前，我们的`options`或者是空的(**第 18 行**或者是`outputbase digits`，表示我们将在输入图像上**仅 OCR 数字**(**第 22 行和第 23 行**)。

从那里，我们使用`image_to_string`函数调用，同时传递我们的`rgb`图像和我们的配置`options` ( **第 26 行**)。请注意，如果`--digits`命令行参数布尔值为`True`，我们将使用`config`参数并包括仅数字设置。

最后，我们在终端中显示 OCR `text`结果(**第 27 行**)。让我们看看这些结果是否符合我们的期望。

### **数字 OCR 结果**

我们现在已经准备好用宇宙魔方来识别数字了。

打开终端并执行以下命令:

```py
$ python ocr_digits.py --image apple_support.png
1-800-275-2273
```

作为对我们的`ocr_digits.py`脚本的输入，我们提供了一个类似名片的样本图像，其中包含文本*“苹果支持”，*以及相应的电话号码(**图 3** )。我们的脚本可以正确地识别电话号码，将其显示到我们的终端上，同时忽略*“Apple Support”*文本。

通过命令行或通过`image_to_string`函数使用 Tesseract 的一个问题是，很难准确调试*tessera CT 如何得到最终输出。*

一旦我们获得更多使用 Tesseract OCR 引擎的经验，我们将把注意力转向视觉调试，并最终通过置信度/概率分数过滤掉无关字符。目前，请注意我们提供给 Tesseract 的选项和配置，以实现我们的目标(即数字识别)。

如果您想 OCR *所有字符*(不仅限于数字)，您可以将`--digits`命令行参数设置为任意值 *≤0:*

```py
$ python ocr_digits.py --image apple_support.png --digits 0
a
Apple Support
1-800-275-2273
```

请注意*“Apple Support”*文本现在是如何与电话号码一起包含在 OCR 输出中的。**可是那个*****【a】*****在输出什么呢？这是从哪里来的？**

输出中的*“a”*是将苹果 logo*顶端*的叶子混淆为字母(**图 4** )。

## **总结**

在本教程中，您学习了如何将 Tesseract 和`pytesseract`配置为仅 OCR *的*数字。然后，我们使用 Python 脚本来处理数字的 OCR 识别。

你会想要密切注意我们提供给宇宙魔方的`config`和`options`。通常，能否成功地将 OCR 应用到一个 Tesseract 项目取决于提供正确的配置集。

在我们的下一个教程中，我们将通过学习如何将一组自定义字符列入白名单和黑名单来继续探索 Tesseract 选项。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******