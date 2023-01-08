# 您使用 Tesseract 和 Python 的第一个 OCR 项目

> 原文：<https://pyimagesearch.com/2021/08/23/your-first-ocr-project-with-tesseract-and-python/>

我第一次使用 Tesseract 光学字符识别(OCR)引擎是在我读大学的时候。

![](img/7b528124d876dea491722d1007c6f190.png)

我正在上我的第一门计算机视觉课程。我们的教授希望我们为期末项目研究一个具有挑战性的计算机视觉主题，扩展现有的研究，然后就我们的工作写一篇正式的论文。我很难决定一个项目，所以我去见了教授，他是一名海军研究员，经常从事计算机视觉和机器学习的医学应用。他建议我研究*自动处方药识别*，这是一种在图像中自动识别处方药的过程。我考虑了一会儿这个问题，然后回答说:

> 你就不能用光学字符识别药片上的印记来识别它吗？

要学习如何在你的第一个项目中进行 OCR， ***继续阅读。***

## **您第一个使用 Tesseract 和 Python 的 OCR 项目**

我仍然记得我的教授脸上的表情。

他笑了，一个小傻笑出现在他的嘴角。知道了我将要遇到的问题，他用*回答道:“要是这么简单就好了。但你很快就会发现。”*

然后我回家，立即开始玩 Tesseract 库，阅读手册/文档，并试图通过命令行 OCR 一些示例图像。但我发现自己在挣扎。一些图像得到了正确的 OCR 识别，而另一些图像则完全没有意义。

为什么 OCR 这么难？为什么我如此挣扎？

我花了一个晚上，熬到深夜，继续用各种图像测试宇宙魔方——对我来说，我无法辨别宇宙魔方可以正确 OCR 的图像和它可能失败的图像之间的模式。这里发生了什么巫术？！

 ***不幸的是，我看到许多计算机视觉从业者在刚开始学习 OCR** 时也有这种感觉——也许你自己也有这种感觉:

1.  你在你的机器上安装宇宙魔方
2.  你可以通过谷歌搜索找到一些教程的基本例子
3.  这些示例返回正确的结果
4.  *…但是当你将同样的 OCR 技术应用于你的图像时，你会得到不正确的结果*

听起来熟悉吗？

问题是这些教程没有系统的教 OCR。他们会向你展示*如何*，但他们不会向你展示*为什么*——这是一条关键的信息，它允许你辨别 OCR 问题中的模式，允许你正确地解决它们。

在本教程中，您将构建您的第一个 OCR 项目。它将作为执行 OCR 所需的“基本”Python 脚本。在以后的文章中，我们将基于你在这里学到的东西。

在本教程结束时，您将对自己在项目中应用 OCR 的能力充满信心。

让我们开始吧。

## **学习目标**

在本教程中，您将:

1.  获得使用 Tesseract 对图像进行 OCR 的实践经验
2.  了解如何将`pytesseract`包导入到 Python 脚本中
3.  使用 OpenCV 从磁盘加载输入图像
4.  通过`pytesseract`库将图像传递到 Tesseract OCR 引擎
5.  在我们的终端上显示 OCR 文本结果

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

## **宇宙魔方入门**

在本教程的第一部分，我们将回顾这个项目的目录结构。从这里，我们将实现一个简单的 Python 脚本，它将:

1.  通过 OpenCV 从磁盘加载输入图像
2.  通过 Tesseract 和`pytesseract`对图像进行光学字符识别
3.  在屏幕上显示 OCR 文本

我们将讨论 OCR 处理的文本结果来结束本教程。

### **项目结构**

```py
|-- pyimagesearch_address.png
|-- steve_jobs.png
|-- whole_foods.png
|-- first_ocr.py
```

我们的第一个项目在组织方式上非常简单。在本教程的代码目录中，您会发现三个用于 OCR 测试的示例 PNG 图像和一个名为`first_ocr.py`的 Python 脚本。

让我们在下一节直接进入 Python 脚本。

### **带魔方的基本 OCR**

让我们从你的第一个宇宙魔方 OCR 项目开始吧！打开一个新文件，将其命名为`first_ocr.py`，并插入以下代码:

```py
# import the necessary packages
import pytesseract
import argparse
import cv2

# construct the argument parser and parse the arguments}
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
args = vars(ap.parse_args())
```

在这个脚本中，您将注意到的第一个 Python `import`是`pytesseract` ( [Python Tesseract](http://pyimg.co/7faq2) )，这是一个 Python 绑定，直接与您的系统上运行的 Tesseract OCR 应用程序绑定。`pytesseract`的力量在于我们能够与宇宙魔方交互，而不是依赖丑陋的`os.cmd`调用，这是在`pytesseract`存在之前我们需要做的。由于它的强大和易用性，我们将在本教程和以后的教程中使用`pytesseract`！

我们的脚本需要一个使用 Python 的`argparse`接口的命令行参数。当您执行这个示例脚本时，通过在终端中直接提供`--image`参数和图像文件路径值，Python 将动态加载您选择的图像。我在本教程的项目目录中提供了三个示例图像，您可以使用。我也强烈建议您通过这个 Python 示例脚本尝试使用 Tesseract 来 OCR 您的图像！

既然我们已经处理了导入和单独的命令行参数，让我们进入有趣的部分——用 Python 进行 OCR:

```py
# load the input image and convert it from BGR to RGB channel
# ordering}
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# use Tesseract to OCR the image
text = pytesseract.image_to_string(image)
print(text)
```

这里，**行 14 和 15** 从磁盘加载我们的输入`--image`并交换颜色通道排序。**宇宙魔方期望 RGB 格式的图像；**然而，OpenCV 以 BGR 顺序加载图像。这不是问题，因为我们可以使用 OpenCV 的`cv2.cvtColor`调用来修复它——只是要特别小心知道什么时候使用 RGB(红绿蓝)和 BGR(蓝绿红)。

**备注 1。** *我还想指出的是，很多时候当你在网上看到宇宙魔方的例子时，他们会使用`PIL`或`pillow`来加载图像。这些软件包以 RGB 格式加载图像，因此不需要转换步骤。*

最后， **Line 18** 对我们的输入 RGB `image`执行 OCR，并将结果作为字符串存储在`text`变量中。

假设`text`现在是一个字符串，我们可以将它传递给 Python 的内置`print`函数，并在我们的终端中查看结果(**第 19 行**)。未来的例子将解释如何用文本本身注释输入图像(即，使用 OpenCV 将`text`结果叠加在输入`--image`的副本上，并显示在屏幕上)。

我们完了！

等等，真的吗？

哦，对了，如果你没有注意到的话，使用 PyTesseract 的 OCR 就像一个简单的函数调用一样简单，前提是你已经按照正确的 RGB 顺序加载了图像。所以现在，让我们检查结果，看看它们是否符合我们的期望。

### **宇宙魔方 OCR 结果**

让我们测试一下新实现的 Tesseract OCR 脚本。打开您的终端，并执行以下命令:

```py
$ python first_ocr.py --image pyimagesearch_address.png
PyImageSearch
PO Box 17598 #17900
Baltimore, MD 21297
```

在**图 2** 中，您可以看到我们的输入图像，它包含 PyImageSearch 的地址，背景是灰色的，有点纹理。如命令和终端输出所示，Tesseract 和`pytesseract`都正确地对文本进行了 OCR。

让我们试试另一张图片，这是史蒂夫·乔布斯的旧名片:

```py
$ python first_ocr.py --image steve_jobs.png
Steven P. Jobs
Chairman of the Board

Apple Computer, Inc.

20525 Mariani Avenue, MS: 3K
Cupertino, California 95014
408 973-2121 or 996-1010.
```

图**图** **3** 中史蒂夫·乔布斯的名片被正确地进行了 OCR 识别*，尽管*输入图像给 OCR 识别扫描文档带来了一些常见的困难，包括:

*   纸张因老化而发黄
*   图像上的噪声，包括斑点
*   开始褪色的文字

尽管有这些挑战，宇宙魔方仍然能够正确地识别名片。但这回避了一个问题——**OCR*这么简单吗？*** 我们是不是只要打开一个 Python shell，导入`pytesseract`包，然后在一个输入图片上调用`image_to_string`？不幸的是，OCR 并没有那么简单(如果是的话，本教程就没有必要了)。作为一个例子，让我们将同样的`first_ocr.py`脚本应用于一张更具挑战性的全食收据照片:

```py
$ python first_ocr.py --image whole_foods.png
aie WESTPORT CT 06880

yHOLE FOODS MARKE
399 post RD WEST ~ ;

903) 227-6858

BACON LS NP

365
pacon LS N
```

图 4 中**的全食杂货店收据没有使用 Tesseract 正确识别。你可以看到宇宙魔方要吐出一堆乱码的废话。OCR 并不总是完美的。**

## **总结**

在本教程中，您使用 Tesseract OCR 引擎、`pytesseract`包(用于与 Tesseract OCR 引擎交互)和 OpenCV 库(用于从磁盘加载输入图像)创建了您的第一个 OCR 项目。

然后，我们将我们的基本 OCR 脚本应用于三个示例图像。我们的基本 OCR 脚本适用于前两个版本，但在最后一个版本中却非常困难。那么是什么原因呢？为什么宇宙魔方能够完美地识别前两个例子*，但是在第三张图片上*却完全失败*？秘密在于图像预处理步骤，以及底层的镶嵌模式和选项。*

 *祝贺您完成今天的教程，干得好！

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******