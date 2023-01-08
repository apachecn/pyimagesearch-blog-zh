# 使用 Tesseract 和 Python 将角色列入白名单和黑名单

> 原文：<https://pyimagesearch.com/2021/09/06/whitelisting-and-blacklisting-characters-with-tesseract-and-python/>

在我们的[之前的教程](https://pyimagesearch.com/2021/08/30/detecting-and-ocring-digits-with-tesseract-and-python/)中，你学习了如何从输入图像中只对个数字进行 OCR *。但是，如果您想对字符过滤过程进行更细粒度的控制，该怎么办呢？*

例如，在构建发票应用程序时，您可能希望不仅提取数字和字母，还提取特殊字符，如美元符号、小数点分隔符(即句点)和逗号。为了获得更细粒度的控制，我们可以应用**白名单**和**黑名单**，这正是本教程的主题。

## **学习目标**

在本教程中，您将了解到:

1.  白名单和黑名单的区别
2.  白名单和黑名单如何用于 OCR 问题
3.  如何使用 Tesseract 应用白名单和黑名单

要了解如何在进行 OCR 时加入白名单和黑名单， ***请继续阅读。***

## **用于 OCR 的白名单和黑名单字符**

在本教程的第一部分，我们将讨论白名单和黑名单之间的区别，这是在应用 OCR 和 Tesseract 时两种常见的字符过滤技术。从那里，我们将回顾我们的项目并实现一个可用于白名单/黑名单过滤的 Python 脚本。然后，我们将检查我们的字符过滤工作的结果。

### **什么是白名单和黑名单？**

作为白名单和黑名单如何工作的例子，让我们考虑一个为 Google 工作的系统管理员。谷歌是[*全球最受欢迎的网站*](https://en.wikipedia.org/wiki/List_of_most_popular_websites)*——几乎互联网上的每个人都使用谷歌——但随着它的受欢迎程度，邪恶的用户可能会试图攻击它，关闭它的服务器，或泄露用户数据。**系统管理员需要*将恶意行为的* IP 地址列入黑名单，同时允许所有其他有效的传入流量。***

 *现在，让我们假设同一个系统管理员需要配置一个开发服务器供 Google 内部使用和测试。**该系统管理员将需要阻止所有传入的 IP 地址** ***，除了谷歌开发者的*** ***白名单*** **IP 地址的** **。**

出于 OCR 目的的白名单和黑名单的概念是相同的。**白名单指定 OCR 引擎只允许*****识别的字符列表——如果一个字符不在白名单上，它*就不能*包含在输出的 OCR 结果中。***

 ***白名单的反义词是黑名单。**黑名单中指定的人物，** ***在任何情况下*** **，都不能列入输出。**

在本教程的其余部分，您将学习如何使用 Tesseract 应用白名单和黑名单。

### **项目结构**

让我们从回顾本教程的目录结构开始:

```py
|-- invoice.png
|-- pa_license_plate.png
|-- whitelist_blacklist.py
```

本教程将实现`whitelist_blacklist.py` Python 脚本，并使用两张图像——一张发票和一张牌照——进行测试。让我们深入研究代码。

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

### **用魔方**将角色列入白名单和黑名单

我们现在将学习如何使用 Tesseract OCR 引擎将字符列入白名单和黑名单。打开项目目录结构中的`whitelist_blacklist.py`文件，插入以下代码:

```py
# import the necessary packages
import pytesseract
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-w", "--whitelist", type=str, default="",
	help="list of characters to whitelist")
ap.add_argument("-b", "--blacklist", type=str, default="",
	help="list of characters to blacklist")
args = vars(ap.parse_args())
```

我们的导入没有什么特别之处——同样，我们使用的是 PyTesseract 和 OpenCV。白名单和黑名单功能通过基于字符串的配置选项内置到 PyTesseract 中。

我们的脚本接受一个输入`--image`路径。此外，它接受两个可选的命令行参数来直接从我们的终端驱动我们的白名单和黑名单功能:

*   `--whitelist`:作为我们的字符的字符串，可以传递给结果
*   `--blacklist`:必须*永不*包含在结果中的字符

`--whitelist`和`--blacklist`参数都有空字符串的`default`值，因此我们可以使用一个、两个或两个都不使用作为我们的 Tesseract OCR 配置的一部分。

接下来，让我们加载我们的图像并构建我们的宇宙魔方 OCR `options`:

```py
# load the input image, swap channel ordering, and initialize our
# Tesseract OCR options as an empty string
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
options = ""

# check to see if a set of whitelist characters has been provided,
# and if so, update our options string
if len(args["whitelist"]) > 0:
	options += "-c tessedit_char_whitelist={} ".format(
		args["whitelist"])

# check to see if a set of blacklist characters has been provided,
# and if so, update our options string
if len(args["blacklist"]) > 0:
	options += "-c tessedit_char_blacklist={}".format(
		args["blacklist"])
```

**第 18 行和第 19 行**以 RGB 格式加载我们的`--image`。我们的`options`变量首先被初始化为一个空字符串(**第 20 行**)。

从那里开始，如果`--whitelist`命令行参数至少有一个我们希望只允许 OCR 的字符，它将作为我们的`options` ( **第 24-26 行**)的一部分被附加到`-c tessedit_char_whitelist=`。

类似地，如果我们通过`--blacklist`参数将任何字符列入黑名单，那么`options`会被附加上`-c tessedit_char_blacklist=`，后跟任何在任何情况下*都不会出现在我们的结果中的字符(**第 30-32 行**)。*

 *同样，我们的`options`字符串可以包含一个、两个或者都不包含白名单/黑名单字符。

最后，我们对 PyTesseract 的`image_to_string`的调用执行 OCR:

```py
# OCR the input image using Tesseract
text = pytesseract.image_to_string(rgb, config=options)
print(text)
```

在我们对`image_to_string`的调用中，唯一新的参数是`config`参数(**第 35 行**)。请注意我们是如何传递连接在一起的宇宙魔方`options`的。OCR 字符的白名单和黑名单的结果通过脚本的最后一行打印出来。

### **白名单和黑名单以及镶嵌结果**

我们现在已经准备好用 Tesseract 应用白名单和黑名单了。打开终端并执行以下命令:

```py
$ python whitelist_blacklist.py --image pa_license_plate.png
PENNSYLVANIA

ZIW*4681

visitPA.com
```

正如终端输出所展示的，我们有一个宾夕法尼亚州的牌照(**图 2** )，除了车牌号码之间的星号(`*`)之外，所有内容都被正确地 OCR 识别了*——这个特殊的符号被错误地 OCR 识别了。利用一点领域知识，我们知道车牌不能包含一个`*`作为字符，所以一个简单的解决问题的方法是将**和`*`列入**的黑名单:*

```py
$ python whitelist_blacklist.py --image pa_license_plate.png \
    --blacklist "*#"
PENNSYLVANIA

ZIW4681

visitPA.com
```

如`--blacklist`命令行参数所示，我们将两个字符列入了黑名单:

*   `*`从上面
*   还有`#`符号(一旦你将`*`列入黑名单，宇宙魔方将试图将这个特殊符号标记为`#`，因此我们将*和*都列入黑名单)

**通过使用*****黑名单，我们的 OCR 结果现在正确了！***

 *让我们试试另一个例子，这是一张发票，包括发票号、签发日期和到期日:

```py
$ python whitelist_blacklist.py --image invoice.png
Invoice Number 1785439
Issue Date 2020-04-08
Due Date 2020-05-08

| DUE | $210.07
```

在图 3 的**中，Tesseract 已经能够正确识别发票的所有字段。有趣的是，它甚至将*底部*框的边缘确定为竖条(`|`)，这对多列数据可能很有用，但在这种情况下只是一个意外的巧合。**

现在假设我们只想过滤出*价格信息(即数字、美元符号和句点)，以及发票号和日期(数字和破折号):*

```py
$ python whitelist_blacklist.py --image invoice.png \
    --whitelist "0123456789.-"
1785439
2020-04-08
2020-05-08

210.07
```

结果正如我们所料！我们现在已经成功地使用了*白名单*来提取发票号、签发日期、到期日期和价格信息，同时丢弃其余信息。

如果需要，我们还可以*组合*白名单和黑名单:

```py
$ python whitelist_blacklist.py --image invoice.png \
	--whitelist "123456789.-" --blacklist "0"
1785439
22-4-8
22-5-8

21.7
```

这里，我们将数字、句点和破折号列入白名单，同时将数字`0`列入黑名单，正如我们的输出所示，我们有发票号、签发日期、到期日和价格，但是所有出现的`0`由于黑名单而被忽略。

当你对图像或文档结构有了先验知识后，你就可以使用白名单和黑名单作为简单而有效的方法来提高输出的 OCR 结果。当您试图提高项目的 OCR 准确性时，它们应该是您的第一站。

## **总结**

在本教程中，您学习了如何使用 Tesseract OCR 引擎应用白名单和黑名单字符过滤。

**白名单指定 OCR 引擎只允许*****识别的字符列表——如果一个字符不在白名单上，它*就不能*包含在输出的 OCR 结果中。白名单的反义词是黑名单。**黑名单指定** ***在任何情况下*** **都不能包含在输出中的字符。*****

 ***使用白名单和黑名单是一种简单而强大的技术，可以在 OCR 应用程序中使用。为了让白名单和黑名单起作用，您需要一个具有可靠模式或结构的文档或图像。例如，如果你正在构建一个基本的收据扫描软件，你可以写一个白名单，只有*允许数字、小数点、逗号和美元符号。*

如果你已经建立了一个自动车牌识别(ALPR)系统，你可能会注意到宇宙魔方变得“混乱”,并输出图像中不是的特殊字符。

在我们的下一个教程中，我们将继续构建我们的 Tesseract OCR 知识，这一次我们将注意力转向检测和校正文本方向，这是提高 OCR 准确性的一个重要的预处理步骤。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*************