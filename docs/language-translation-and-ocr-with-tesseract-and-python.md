# 使用 Tesseract 和 Python 进行语言翻译和 OCR

> 原文：<https://pyimagesearch.com/2021/09/20/language-translation-and-ocr-with-tesseract-and-python/>

鉴于我们可以检测出文字的书写系统，这就提出了一个问题:

> *是否可以使用 OCR 和 Tesseract 将***文本从一种语言翻译成另一种语言？**

 ***要学习如何使用宇宙魔方和 Python 翻译语言，** ***继续阅读。***

## **使用 Tesseract 和 Python 进行语言翻译和 OCR**

简短的回答是*是的*，这是可能的——但是我们需要`textblob`库的一点帮助，这是一个流行的用于文本处理的 Python 包( [TextBlob:简化的文本处理](https://textblob.readthedocs.io/en/dev/))。本教程结束时，您将自动将 OCR 文本从一种语言翻译成另一种语言。

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

## **学习目标**

在本教程中，您将:

1.  了解如何使用`TextBlob` Python 包翻译文本
2.  实现一个 Python 脚本，对文本进行 ocr，然后进行翻译
3.  查看文本翻译的结果

## **OCR 和语言翻译**

在本教程的第一部分，我们将简要讨论`textblob`包以及如何用它来翻译文本。从那里，我们将回顾我们的项目目录结构，并实现我们的 OCR 和文本翻译 Python 脚本。我们将讨论我们的 OCR 和文本翻译结果来结束本教程。

### **用 TextBlob 将文本翻译成不同的语言**

为了将文本从一种语言翻译成另一种语言，我们将使用`textblob` Python 包([https://textblob.readthedocs.io/en/dev/](https://textblob.readthedocs.io/en/dev/))。如果您遵循了早期教程中[的开发环境配置说明，那么您应该已经在系统上安装了`textblob`。如果没有，可以用`pip`安装:](https://pyimagesearch.com/2021/08/09/what-is-optical-character-recognition-ocr/)

```py
$ pip install textblob
```

一旦安装了`textblob`，您应该运行以下命令来下载`textblob`用来自动分析文本的自然语言工具包(NLTK)语料库:

```py
$ python -m textblob.download_corpora
```

接下来，您应该通过打开 Python shell 来熟悉这个库:

```py
$ python
>>> from textblob import TextBlob
>>>
```

注意我们是如何导入`TextBlob`类的——这个类使我们能够自动分析一段文本的标签、名词短语，是的，甚至是语言翻译。一旦实例化，我们可以调用`TextBlob`类的`translate()`方法并执行自动文本翻译。现在让我们使用`TextBlob`来做这件事:UTF8ipxm

```py
>>> text = u"おはようございます。"
>>> tb = TextBlob(text)
>>> translated = tb.translate(to="en")
>>> print(translated)
Good morning.
>>>
```

请注意我是如何成功地将日语短语*“早上好”*翻译成英语的。

### **项目结构**

让我们首先回顾一下本教程的项目目录结构:

```py
|-- comic.png
|-- ocr_translate.py
```

我们的项目包括一个有趣的卡通形象，我用一个叫做 [Explosm](http://explosm.net) 的漫画工具制作的。我们基于`textblob`的 OCR 翻译器包含在`ocr_translate.py`脚本中。

### **实施我们的 OCR 和语言翻译脚本**

我们现在准备实现我们的 Python 脚本，它将自动 OCR 文本并将其翻译成我们选择的语言。在我们的项目目录结构中打开`ocr_translate.py`，并插入下面的代码:

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
ap.add_argument("-l", "--lang", type=str, default="es",
	help="language to translate OCR'd text to (default is Spanish)")
args = vars(ap.parse_args())
```

我们从导入开始，其中`TextBlob`是这个脚本中最值得注意的。从那里，我们深入到我们的命令行参数解析过程。我们有两个命令行参数:

*   `--image`:要进行 OCR 识别*和*翻译的输入图像的路径
*   `--lang`:将 OCR 文本翻译成的语言—默认情况下，是西班牙语(`es`)

使用`pytesseract`，我们将对输入图像进行 OCR:

```py
# load the input image and convert it from BGR to RGB channel
# ordering
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# use Tesseract to OCR the image, then replace newline characters
# with a single space
text = pytesseract.image_to_string(rgb)
text = text.replace("\n", " ")

# show the original OCR'd text
print("ORIGINAL")
print("========")
print(text)
print("")
```

在加载并将我们的`--image`转换为 RGB 格式后(**第 17 行和第 18 行**)，我们通过`pytesseract` ( **第 22 行**)通过 Tesseract 引擎发送它。我们的`textblob`包不知道如何处理`text`中出现的换行符，所以我们用空格`replace`它们(**第 23 行**)。

在打印出我们最初的 OCR'd `text`之后，我们将继续进行**将字符串翻译成我们想要的语言:**

```py
# translate the text to a different language
tb = TextBlob(text)
translated = tb.translate(to=args["lang"])

# show the translated text
print("TRANSLATED")
print("==========")
print(translated)
```

**第 32 行**构造一个`TextBlob`对象，将原来的`text`传递给构造函数。从那里，**第 33 行**将`tb`翻译成我们想要的`--lang`。最后，我们在终端中打印出`translated`结果(**第 36-38 行**)。

这就是全部了。请记住翻译引擎的复杂性。引擎盖下的`TextBlob`引擎类似于[谷歌翻译](https://translate.google.com)之类的服务，尽管功能可能没那么强大。当[谷歌翻译在 2000 年代中期](https://en.wikipedia.org/wiki/Google_Translate)问世时，它远没有今天这么完美和准确。有些人可能会认为谷歌翻译是黄金标准。根据您的 OCR 翻译需求，如果您发现`textblob`不适合您，您可以调用 Google Translate REST API。

### **OCR 语言翻译结果**

我们现在准备用 Tesseract 对输入图像进行 OCR，然后用`textblob`翻译文本。为了测试我们的自动 OCR 和翻译脚本，打开一个终端并执行图 2 ( *右*)所示的命令。在这里，我们的输入图像在左边的 T5 处，包含了英文感叹号，T7，“你告诉我学习 OCR 会很容易！”这张图片是使用 [Explosm](http://explosm.net) 漫画生成器生成的。正如我们的终端输出所示，我们成功地将文本翻译成了西班牙语、德语和阿拉伯语(一种从右向左的语言)。

一旦你使用了`textblob`包，文本的 OCR 识别和翻译就变得非常容易！

## **总结**

在本教程中，您学习了如何使用 Tesseract、Python 和`textblob`库自动 OCR 和翻译文本。使用`textblob`，翻译文本就像调用一个函数一样简单。

在我们的下一个教程中，您将学习如何使用 Tesseract 来自动*OCR 非英语语言，包括非拉丁语书写系统(例如，阿拉伯语、汉语等)。).*

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****