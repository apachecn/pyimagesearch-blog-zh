# 使用拼写检查提高 Tesseract OCR 的准确性

> 原文：<https://pyimagesearch.com/2021/11/29/using-spellchecking-to-improve-tesseract-ocr-accuracy/>

在[之前的教程](https://pyimagesearch.com/2021/09/20/language-translation-and-ocr-with-tesseract-and-python/)中，您学习了如何使用`textblob`库和 Tesseract 自动 OCR 文本，然后将其翻译成不同的语言。本教程也将使用`textblob`，但这次是通过自动拼写检查 OCR 文本来提高 OCR 的准确性。

**要了解如何使用拼写检查对结果进行 OCR，请继续阅读。**

## **使用拼写检查来提高立方体 OCR 的准确性**

期望 ***任何*** **OCR 系统，即使是最先进的 OCR 引擎，做到 100%准确也是不现实的。这在实践中是不会发生的。不可避免的是，输入图像中的噪声、Tesseract 没有训练过的非标准字体或低于理想的图像质量都会导致 Tesseract 出错并错误地对一段文本进行 OCR。**

当这种情况发生时，您需要创建规则和试探法来提高输出 OCR 质量。你应该关注的第一个规则和启发是**自动拼写检查。**例如，如果您正在对一本书进行 OCR，您可以使用拼写检查来尝试在 OCR 过程后自动更正，从而创建更好、更准确的数字化文本版本。

## **学习目标**

在本教程中，您将:

1.  了解如何使用`textblob`包进行拼写检查
2.  包含不正确拼写的一段文本
3.  自动更正 OCR 文本的拼写

## **OCR 和拼写检查**

我们将从回顾我们的项目目录结构开始本教程。然后，我将向您展示如何实现一个 Python 脚本，该脚本可以自动对一段文本进行 OCR，然后使用`textblob`库对其进行拼写检查。一旦我们的脚本被实现，我们将把它应用到我们的示例图像。我们将讨论拼写检查的准确性，包括一些与自动拼写检查相关的限制和缺点，从而结束本教程。

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

我们的 OCR 拼写检查器的项目目录结构非常简单:

```py
|-- comic_spelling.png
|-- ocr_and_spellcheck.py
```

我们这里只有一个 Python 脚本，`ocr_and_spellcheck.py`。该脚本执行以下操作:

1.  从磁盘加载`comic_spelling.png`
2.  对图像中的文本进行 OCR
3.  对其应用拼写检查

通过应用拼写检查，我们将理想地能够提高我们脚本的 OCR 准确性，*不管*如果:

1.  输入图像中有不正确的拼写
2.  对字符进行不正确的 OCR 处理

### **实现我们的 OCR 拼写检查脚本**

让我们开始实现我们的 OCR 和拼写检查脚本。

打开一个新文件，将其命名为`ocr_and_spellcheck.py`，并插入以下代码:

```py
# import the necessary packages
from textblob import TextBlob
import pytesseract
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
args = vars(ap.parse_args())
```

**第 2-5 行**导入我们需要的 Python 包。你应该注意到`textblob`包的使用，我们在[的前一课](https://pyimagesearch.com/2021/09/20/language-translation-and-ocr-with-tesseract-and-python/)中利用它将 OCR 文本从一种语言翻译成另一种语言。我们将在本教程中使用`textblob`，**，但这次是为了它的自动拼写检查实现。**

**第 8-11 行**然后解析我们的命令行参数。我们只需要一个参数，`--image`,它是输入图像的路径:

接下来，我们可以从磁盘加载图像并对其进行 OCR:

```py
# load the input image and convert it from BGR to RGB channel
# ordering
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# use Tesseract to OCR the image
text = pytesseract.image_to_string(rgb)

# show the text *before* ocr-spellchecking has been applied
print("BEFORE SPELLCHECK")
print("=================")
print(text)
print("\n")
```

**第 15 行**使用提供的路径从磁盘加载我们的输入`image`。然后，我们将颜色通道排序从 BGR (OpenCV 的默认排序)交换到 RGB(这是 Tesseract 和`pytesseract`所期望的)。

一旦图像被加载，我们调用`image_to_string`来 OCR 图像。然后我们在屏幕上显示在拼写检查之前经过 OCR 处理的`text`*(**第 19-25 行**)。*

但是，可能会有拼写错误，例如用户在创建图像时拼写错误的文本，或者由于 Tesseract 错误地对一个或多个字符进行 OCR 而导致的“输入错误”——为了解决这个问题，我们需要利用`textblob`:

```py
# apply spell checking to the OCR'd text
tb = TextBlob(text)
corrected = tb.correct()

# show the text after ocr-spellchecking has been applied
print("AFTER SPELLCHECK")
print("================")
print(corrected)
```

**第 28 行**从 OCR 识别的文本中构造一个`TextBlob`。然后，我们通过`correct()`方法(**第 29 行**)应用自动拼写检查纠正。然后`corrected`文本(即拼写检查后的*)显示在终端上(**第 32-34 行**)。*

### **OCR 拼写检查结果**

我们现在准备对示例图像应用 OCR 拼写检查。

打开终端并执行以下命令:

```py
$ python ocr_and_spellcheck.py --image comic_spelling.png
BEFORE SPELLCHECK
=================
Why can't yu
spel corrctly?

AFTER SPELLCHECK
================
Why can't you
spell correctly?
```

**图 2** 显示了我们的示例图像(通过 [Explosm 漫画生成器](http://explosm.net/rcg)创建)，其中包括拼写错误的单词。使用 Tesseract，我们可以用 OCR 识别出有拼写错误的文本。

值得注意的是，这些拼写错误是*故意*引入的——在您的 OCR 应用程序中，这些拼写错误可能自然存在于您的输入图像中*或* Tesseract 可能会错误地 OCR 某些字符。

正如我们的输出所示，我们能够使用`textblob`来纠正这些拼写错误，正确地纠正单词*“Yu you”、“spel spel”、*和【T3”

### **局限性和缺点**

拼写检查算法的最大问题之一是**大多数拼写检查器需要** ***一些*** **人工干预才能准确。当我们犯了拼写错误时，我们的文字处理器会自动检测错误并提出候选修正——通常是拼写检查器认为我们应该拼写的两三个单词。除非我们十有八九拼错了一个单词，否则我们可以在拼写检查器建议的候选词中找到我们想要使用的单词。**

我们可以选择*移除*人工干预部分，而是允许拼写检查器使用它基于内部拼写检查算法认为最有可能的单词。我们冒着用在句子或段落的原始上下文中没有意义的单词替换只有小拼写错误的单词的风险。因此，在依赖*全自动*拼写检查器时，你应该小心谨慎。存在在输出的 OCR 文本中插入不正确的单词(相对于正确的单词，但是有小的拼写错误)的风险。

如果您发现拼写检查损害了 OCR 的准确性，您可能需要:

1.  除了包含在`textblob`库中的通用算法之外，寻找替代的拼写检查算法
2.  用基于启发的方法替换拼写检查(例如，正则表达式匹配)
3.  允许拼写错误存在，记住没有一个 OCR 系统是 100%准确的

## **总结**

在本教程中，您学习了如何通过应用自动拼写检查来改善 OCR 结果。**虽然我们的方法在我们的特定示例中运行良好，*但在其他情况下可能不太适用！请记住，拼写检查算法通常需要少量的人工干预。大多数拼写检查器会自动检查文档中的拼写错误，然后*会向用户建议*一个候选更正列表。由*人*做出最终的拼写检查决定。***

当我们去除人工干预的成分，转而允许拼写检查算法选择它认为最合适的纠正时，只有轻微拼写错误的单词将被替换为在句子的原始上下文中没有意义的单词。在您自己的 OCR 应用程序中，谨慎使用拼写检查，尤其是*自动*拼写检查——在某些情况下，它将有助于您的 OCR 准确性，但在其他情况下，它可能会损害准确性。

### **引用信息**

**罗斯布鲁克，**一**。**“使用拼写检查提高 Tesseract OCR 准确度”， *PyImageSearch* ，2021，[https://PyImageSearch . com/2021/11/29/Using-spell checking-to-improve-tessera CT-OCR-accuracy/](https://pyimagesearch.com/2021/11/29/using-spellchecking-to-improve-tesseract-ocr-accuracy/)

```py
@article{Rosebrock_2021_Spellchecking,
  author = {Adrian Rosebrock},
  title = {Using spellchecking to improve {T}esseract {OCR} accuracy},
  journal = {PyImageSearch},
  year = {2021},
  note = {https://pyimagesearch.com/2021/11/29/using-spellchecking-to-improve-tesseract-ocr-accuracy/},
}
```

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****