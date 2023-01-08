# 使用 Tesseract 和 Python 校正文本方向

> 原文：<https://pyimagesearch.com/2022/01/31/correcting-text-orientation-with-tesseract-and-python/>

任何 OCR 系统的一个重要组成部分是**图像预处理**——你呈现给 OCR 引擎的输入图像质量越高，你的 OCR 输出就越好。要想在 OCR 中取得成功，你需要回顾一下可以说是*最重要的*预处理步骤:**文本定位。**

**要学习如何使用 Tesseract 和 Python 进行文本定位，** ***继续阅读。***

## 使用 Tesseract 和 Python 校正文本方向

文本方向是指图像中一段文本的旋转角度。如果文本被显著旋转，给定的单词、句子或段落对 OCR 引擎来说将看起来像乱码。OCR 引擎是智能的，但像人类一样，它们没有被训练成颠倒阅读！

因此，为 OCR 准备图像数据的第一个关键步骤是检测文本方向(如果有)，然后更正文本方向。从那里，您可以将校正后的图像呈现给您的 OCR 引擎(并理想地获得更高的 OCR 准确性)。

## **学习目标**

在本教程中，您将学习:

1.  方向和脚本检测(OSD)的概念
2.  如何用 Tesseract 检测文本脚本(即书写系统)
3.  如何使用 Tesseract 检测文本方向
4.  如何用 OpenCV 自动校正文本方向

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

## **什么是定向和脚本检测？**

在我们使用 Tesseract 自动检测和纠正文本方向之前，我们首先需要讨论一下**方向和脚本检测(OSD)的概念。**在自动检测和 OCR 文本时，Tesseract 有几种不同的模式可供您使用。其中一些模式对输入图像执行全面的 OCR，而其他模式输出元数据，如文本信息、方向等。(即您的 OSD 模式)。Tesseract 的 OSD 模式将为您提供两个输出值:

*   **文本方向:**输入图像中文本的估计旋转角度(以度为单位)。
*   **脚本:**图像中文本的预测“书写系统”。

**图 2** 显示了一个改变文本方向的例子。当在 OSD 模式下，Tesseract 将检测这些方向，并告诉我们如何纠正方向。

书写系统是一种交流信息的视觉方法，但不同于语音，书写系统还包括“存储”和“知识转移”的概念

当我们落笔时，我们使用的字符是书写系统的一部分。这些字符可以被我们和其他人阅读，从而从作者那里传递知识。此外，这种知识“储存”在纸上，这意味着如果我们死了，留在纸上的知识可以传授给其他人，他们可以阅读我们的手稿/书写系统。

**图 2** 还提供了各种文字和书写系统的例子，包括拉丁语(英语和其他语言中使用的文字)和 Abjad(其他语言中的希伯来语文字)。当置于 OSD 模式时，Tesseract 会自动检测输入图像中文本的书写系统。

如果你对脚本/书写系统的概念不熟悉，我强烈推荐阅读维基百科关于这个主题的优秀文章。这是一本很好的读物，它涵盖了书写系统的历史以及它们是如何演变的。

## **使用 Tesseract 检测和纠正文本方向**

现在我们已经了解了 OSD 的基础知识，让我们继续使用 Tesseract 检测和纠正文本方向。我们将从快速回顾我们的项目目录结构开始。从那以后，我将向您展示如何实现文本方向校正。我们将讨论我们的结果来结束本教程。

### **项目结构**

让我们深入了解一下这个项目的目录结构:

```py
|-- images
|   |-- normal.png
|   |-- rotated_180.png
|   |-- rotated_90_clockwise.png
|   |-- rotated_90_counter_clockwise.png
|   |-- rotated_90_counter_clockwise_hebrew.png
|-- detect_orientation.py
```

检测和纠正文本方向的所有代码都包含在`detect_orientation.py` Python 脚本中，并在不到 35 行代码中实现，包括注释。我们将使用项目文件夹中包含的一组`images/`来测试代码。

### **实现我们的文本定位和校正脚本**

让我们开始用 Tesseract 和 OpenCV 实现我们的文本方向修正器。

打开一个新文件，将其命名为`detect_orientation.py`，并插入以下代码:

```py
# import the necessary packages
from pytesseract import Output
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

一开始您可能没有意识到的一个导入是 PyTesseract 的`Output`类(【https://github.com/madmaze/pytesseract】的)。这个类简单地指定了四种数据类型，包括我们将利用的`DICT`。

我们唯一的命令行参数是要进行 OCR 的输入`--image`。现在让我们加载输入:

```py
# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to determine the text orientation
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)

# display the orientation information
print("[INFO] detected orientation: {}".format(
	results["orientation"]))
print("[INFO] rotate by {} degrees to correct".format(
	results["rotate"]))
print("[INFO] detected script: {}".format(results["script"]))
```

**第 16 行和第 17 行**加载我们的输入`--image`并交换颜色通道，以便它与 Tesseract 兼容。

从那里，我们将**方向和脚本检测(OSD)** 应用到`rgb`图像，同时指定我们的`output_type=Output.DICT` ( **第 18 行**)。然后我们在终端显示方向和脚本信息(包含在`results`字典中)，包括:

*   当前的方向
*   将图像旋转多少度以校正其方向
*   检测到的脚本类型，如拉丁文或阿拉伯文

根据这些信息，接下来，我们将更正文本方向:

```py
# rotate the image to correct the orientation
rotated = imutils.rotate_bound(image, angle=results["rotate"])

# show the original image and output image after orientation
# correction
cv2.imshow("Original", image)
cv2.imshow("Output", rotated)
cv2.waitKey(0)
```

使用我的 imutils `rotate_bound`方法(【http://pyimg.co/vebvn】T2)，我们旋转图像，确保整个图像在结果中保持完全可见(**第 28 行**)。如果我们使用 OpenCV 的通用`cv2.rotate`方法，图像的边角就会被剪掉。最后，我们显示原始图像和旋转图像，直到按下一个键(**第 32-34 行**)。

### **文本方向和校正结果**

我们现在准备应用文本 OSD！打开终端并执行以下命令:

```py
$ python detect_orientation.py --image images/normal.png
[INFO] detected orientation: 0
[INFO] rotate by 0 degrees to correct
[INFO] detected script: Latin
```

**图 3** 显示了我们的脚本和方向检测的结果。注意，输入图像已经*而不是*旋转，意味着方向是 0 。不需要旋转校正。然后，该脚本被检测为“拉丁文”

让我们试试另一张图片，这张图片带有旋转的文字:

```py
$ python detect_orientation.py --image images/rotated_90_clockwise.png
[INFO] detected orientation: 90
[INFO] rotate by 270 degrees to correct
[INFO] detected script: Latin
```

**图 4** 显示了包含旋转文本的原始输入图像。在 OSD 模式下使用 Tesseract，我们可以检测到输入图像中的文本方向为 90 —我们可以通过将图像旋转 270 (即*—*90*)来纠正这一方向。同样，检测到的脚本是拉丁文。*

 *我们将用最后一个非拉丁语文本的例子来结束本教程:

```py
$ python detect_orientation.py \
    --image images/rotated_90_counter_clockwise_hebrew.png
[INFO] detected orientation: 270
[INFO] rotate by 90 degrees to correct
[INFO] detected script: Hebrew
```

**图 5** 显示了我们输入的文本图像。然后，我们检测该脚本(希伯来语)，并通过将文本旋转 90来校正其方向。

正如你所看到的，宇宙魔方使文本 OSD 变得简单！

## **总结**

在本教程中，您学习了如何使用 Tesseract 的方向和脚本检测(OSD)模式执行自动文本方向检测和校正。

OSD 模式为我们提供图像中文本的元数据，包括**估计的文本方向**和**脚本/书写系统检测。**文本方向是指图像中文本的角度(以度为单位)。当执行 OCR 时，我们可以通过校正文本方向来获得更高的准确性。另一方面，文字检测指的是文本的书写系统，可以是拉丁语、汉字、阿拉伯语、希伯来语等。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****