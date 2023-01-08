# 非英语语言的 Tesseract OCR

> 原文：<https://pyimagesearch.com/2020/08/03/tesseract-ocr-for-non-english-languages/>

在本教程中，您将学习如何使用 Tesseract OCR 引擎对非英语语言进行 OCR。

如果你参考我在 PyImageSearch 博客上的[以前的光学字符识别(OCR)教程，你会注意到所有的 **OCR 文本都是用*英语*语言编写的。**](https://pyimagesearch.com/category/optical-character-recognition-ocr/)

但是如果你想对非英语文本进行 OCR 呢？

你需要采取什么步骤？

宇宙魔方如何处理非英语语言？

我们将在本教程中回答所有这些问题。

**要了解如何使用 Tesseract 对非英语语言的文本进行 OCR，请继续阅读。**

## **非英语语言的光学字符识别(OCR)**

在本教程的第一部分，您将学习如何为多种语言配置 Tesseract OCR 引擎，包括非英语语言。

然后，我将向您展示如何为 Tesseract 下载多个语言包，并验证它是否正常工作——我们将使用德语作为示例。

从那里，我们将配置 TextBlob 包，它将用于从一种语言翻译成另一种语言。

完成所有设置后，我们将为 Python 脚本实现项目结构，该脚本将:

1.  接受输入图像
2.  检测和 OCR 非英语语言的文本
3.  将 OCR 识别的文本从给定的输入语言翻译成英语
4.  将结果显示到我们的终端上

我们开始吧！

### **为多种语言配置 Tesseract OCR】**

在本节中，我们将为多种语言配置 Tesseract OCR。我们将一步一步地分解它，看看它在 macOS 和 Ubuntu 上是什么样子。

如果您尚未安装宇宙魔方:

*   我已经在我的博文 **[*OpenCV OCR 和使用 Tesseract 的文本识别中提供了安装*](https://pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/)***[*Tesseract OCR 引擎*](https://github.com/tesseract-ocr/tesseract)** 以及 **[pytesseract](https://github.com/madmaze/pytesseract)** (用于连接 tesseract 的 Python 绑定)的说明。***
*   按照该教程的*如何安装宇宙魔方 4* 部分的说明，确认您的宇宙魔方安装，然后回到这里学习如何为多种语言配置宇宙魔方。

从技术上讲，Tesseract *应该已经配置为处理多种语言，包括非英语语言；然而，根据我的经验，多语言支持可能有点不稳定。我们将回顾我的方法，它给出了一致的结果。*

如果您通过自制软件在 **macOS** 上安装了宇宙魔方，那么您的宇宙魔方语言包应该在`/usr/local/Cellar/tesseract/<version>/share/tessdata`中可用，其中`<version>`是您的宇宙魔方安装的版本号(您可以使用`tab`键自动完成以获得您机器上的完整路径)。

如果你在 **Ubuntu** 上运行，你的宇宙魔方语言包应该位于目录`/usr/share/tesseract-ocr/<version>/tessdata`中，其中`<version>`是你的宇宙魔方安装的版本号。

让我们用下面的**图 1、**所示的`ls`命令快速查看一下这个`tessdata`目录的内容，它对应于我的 macOS 上的 Homebrew 安装，用于英语语言配置。

*   ``eng.traineddata`` 是英语的语言包。
*   ``osd.traineddata`` 是与方位和脚本相关的特殊数据文件。
*   ``snum.traineddata`` 是宇宙魔方使用的内部序列号。
*   ``pdf.ttf`` 是一个支持 pdf 渲染的 True Type 格式字体文件。

### **下载语言包并添加到 Tesseract OCR**

宇宙魔方的第一个版本只支持英语。在第二个版本中增加了对法语、意大利语、德语、西班牙语、巴西葡萄牙语和荷兰语的支持。

在第三个版本中，对表意(符号)语言(如中文和日文)以及从右向左书写的语言(如阿拉伯语和希伯来语)的支持显著增加。

我们现在使用的第四个版本支持 100 多种语言，并且支持字符和符号。

***注:**第四个版本包含 Tesseract 的传统和更新、更准确的长期短期记忆(LSTM) OCR 引擎的训练模型。*

现在我们对支持的语言范围有了一个概念，让我们深入了解我发现的最*简单的*方法来配置 Tesseract 并释放这种巨大的多语言支持的力量:

1.  从 GitHub 手动下载 Tesseract 的语言包并安装。
2.  设置`TESSDATA_PREFIX`环境变量指向包含语言包的目录。

这里的第一步是克隆 Tesseract 的 GitHub `tessdata`存储库，它位于:

[https://github.com/tesseract-ocr/tessdata](https://github.com/tesseract-ocr/tessdata)

我们希望移动到我们希望成为本地`tessdata`目录的父目录的目录。然后，我们将简单地发出下面的`git`命令来`clone`将回购文件发送到我们的本地目录。

```py
$ git clone https://github.com/tesseract-ocr/tessdata
```

***注意:**注意，在撰写本文时，生成的* `tessdata` *目录将为 **~4.85GB** ，因此请确保您的硬盘上有足够的空间。*

第二步是设置环境变量`TESSDATA_PREFIX`指向包含语言包的目录。我们将把目录(`cd`)改为`tessdata`目录，并使用`pwd`命令确定*到该目录的完整系统路径*:

```py
$ cd tessdata/
$ pwd
/Users/adrianrosebrock/Desktop/tessdata
```

你的`tessdata`目录将有一个不同于我的路径，**所以确保你运行上面的命令来确定路径*特定的*到你的机器！**

从那里，您需要做的就是设置`TESSDATA_PREFIX`环境变量指向您的`tessdata`目录，从而允许 Tesseract 找到语言包。为此，只需执行以下命令:

```py
$ export TESSDATA_PREFIX=/Users/adrianrosebrock/Desktop/tessdata
```

同样，您的完整路径将与我的不同，所以要注意仔细检查和三次检查您的文件路径。

### **项目结构**

让我们回顾一下项目结构。

一旦您从本文的 ***【下载】*** 部分获取文件，您将看到以下目录结构:

```py
$ tree --dirsfirst --filelimit 10
.
├── images
│   ├── arabic.png
│   ├── german.png
│   ├── german_block.png
│   ├── swahili.png
│   └── vietnamese.png
└── ocr_non_english.py

1 directory, 6 files
```

`images/`子目录包含几个我们将用于 OCR 的 PNG 文件。标题指示将用于 OCR 的母语。

Python 文件`ocr_non_english.py`，位于我们的主目录中，是我们的驱动文件。它将 OCR 我们的母语文本，然后从母语翻译成英语。

### **验证 Tesseract 对非英语语言的支持**

此时，您应该已经将 Tesseract 正确配置为支持非英语语言，但是作为一项健全性检查，让我们通过使用`echo`命令来验证`TESSDATA_PREFIX`环境变量是否设置正确:

```py
$ echo $TESSDATA_PREFIX
/Users/adrianrosebrock/Desktop/tessdata
```

```py
$ tesseract german.png stdout -l deu
```

1.  检查`tessdata`目录。
2.  参考宇宙魔方文档，其中[列出了宇宙魔方支持](https://github.com/tesseract-ocr/tesseract/blob/master/doc/tesseract.1.asc#languages-and-scripts)的语言和对应的代码。
3.  [使用此网页](https://www.iban.com/country-codes)确定主要使用某种语言的国家代码。
4.  最后，如果你仍然不能得到正确的国家代码，使用一点 Google-foo，搜索你所在地区的三个字母的国家代码(在 Google 上搜索*宇宙魔方<语言名称>代码*也无妨)。

只要有一点耐心，再加上一些练习，你就可以用 Tesseract 对非英语语言的文本进行 OCR。

### **text blob 包的环境设置**

现在我们已经设置了 Tesseract 并添加了对非英语语言的支持，我们需要设置 TextBlob 包。

***注意:这一步假设你已经在 Python3 虚拟环境**中工作(例如* `$ workon cv` *其中* `cv` *是一个虚拟环境的名称——你的可能会不同)* **)。**

安装`textblob`只是一个快速命令:

```py
$ pip install textblob
```

伟大的工作设置您的环境依赖！

### **用非英语语言脚本实现我们的宇宙魔方**

我们现在已经准备好为非英语语言支持实现 Tesseract。让我们从下载小节回顾一下现有的`ocr_non_english.py`。

打开项目目录中的`ocr_non_english.py`文件，并插入以下代码:

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
ap.add_argument("-l", "--lang", required=True,
	help="language that Tesseract will use when OCR'ing")
ap.add_argument("-t", "--to", type=str, default="en",
	help="language that we'll be translating to")
ap.add_argument("-p", "--psm", type=int, default=13,
	help="Tesseract PSM mode")
args = vars(ap.parse_args())
```

**第 5 行**导入`TextBlob`，这是一个非常有用的处理文本数据的 Python 库。它可以执行各种自然语言处理任务，如标记词性。我们将使用它将 OCR 识别的外语文本翻译成英语。你可以在这里阅读更多关于 TextBlob 的内容:[https://textblob.readthedocs.io/en/dev/](https://textblob.readthedocs.io/en/dev/)

*   `--image`:要进行 OCR 的输入图像的路径。
*   ``--lang`` :宇宙魔方在 ORC 图像时使用的本地语言。
*   我们将把本地 OCR 文本翻译成的语言。
*   ``--psm`` :镶嵌的页面分割方式。我们的`default`是针对`13` **、**的页面分割模式，它将图像视为单行文本。对于我们今天的最后一个例子，我们将对一整块德语文本进行 OCR。对于这个完整的块，我们将使用一个页面分割模式`3`，这是一个没有方向和脚本检测(OSD)的全自动页面分割。

导入、便利函数和命令行`args`都准备好了，在循环遍历帧之前，我们只需要处理一些初始化:

```py
# load the input image and convert it from BGR to RGB channel
# ordering
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# OCR the image, supplying the country code as the language parameter
options = "-l {} --psm {}".format(args["lang"], args["psm"])
text = pytesseract.image_to_string(rgb, config=options)

# show the original OCR'd text
print("ORIGINAL")
print("========")
print(text)
print("")
```

在本节中，我们将从文件中加载图像，更改图像颜色通道的顺序，设置 Tesseract 的选项，并以图像的母语对图像执行光学字符识别。

*   Tesseract 用于 OCR 图像的本地语言(`-l`)。
*   页面分割模式选项(`-psm`)。这些对应于我们运行这个程序时在命令行上提供的输入参数。

接下来，我们将以本地语言显示 Tesseract 的 OCR 结果来结束这一部分(**第 32-35 行**):

```py
# translate the text into a different language
tb = TextBlob(text)
translated = tb.translate(to=args["to"])

# show the translated text
print("TRANSLATED")
print("==========")
print(translated)
```

既然我们已经用本地语言对文本进行了 OCR，我们将把文本从我们的`--lang`命令行参数指定的本地语言翻译成我们的`--to`命令行参数描述的输出语言。

我们使用`TextBlob` ( **第 38 行**)将文本抽象为 textblob。然后，我们使用`tb.tranlsate`翻译**第 39 行**的最终语言。我们最后打印翻译文本的结果(**第 42-44 行)**。现在，您有了一个完整的工作流程，其中包括对母语文本进行 OCR，并将其翻译成您想要的语言。

很好地实现了不同语言的 Tesseract 正如您所看到的，它相对简单。接下来，我们将确保我们的脚本和宇宙魔方正在所有的气缸上点火。

### **细分 OCR 和非英语语言结果**

是时候让非英语语言的宇宙魔方发挥作用了！

打开终端，从主项目目录执行以下命令:

```py
$ python ocr_non_english.py --image images/german.png --lang deu
ORIGINAL
========
Ich brauche ein Bier!

TRANSLATED
==========
I need a beer!
```

在图 3 的**中，**您可以看到一个带有文本*的输入图像，“我是 brauche ein Bier！”*这是德语中的*“我需要一杯啤酒！”*

```py
$ python ocr_non_english.py --image images/swahili.png --lang swa
ORIGINAL
========
Jina langu ni Adrian

TRANSLATED
==========
My name is Adrian
```

`--lang swa`标志表示我们想要 OCR 斯瓦希里语文本(**图 4** )。

宇宙魔方正确地 OCR 出了文本*“纪娜·顾岚·尼·阿德里安”，*翻译成英语就是*“我的名字是阿德里安。”*

此示例显示如何对越南语文本进行 OCR，越南语是一种不同于前面示例的脚本/书写系统:

```py
$ python ocr_non_english.py --image images/vietnamese.png --lang vie
ORIGINAL
========
Tôi mến bạn..

TRANSLATED
==========
I love you..
```

```py
$ python ocr_non_english.py --image images/arabic.png --lang ara
ORIGINAL
========
أنا أتحدث القليل من العربية فقط..

TRANSLATED
==========
I only speak a little Arabic ..
```

使用`--lang ara`标志，我们能够告诉 Tesseract 对阿拉伯文本进行 OCR。

在这里，我们可以看到，阿拉伯语“阿拉伯语”是“我只说一点阿拉伯语”。英语中“T1”是“roughly translates”，意思是“我是唯一一个阿拉伯语”是“T3”。

对于我们的最后一个示例，让我们对一大块德语文本进行 OCR:

```py
$ python ocr_non_english.py --image images/german_block.png --lang deu --psm 3
ORIGINAL
========
Erstes Kapitel

Gustav Aschenbach oder von Aschenbach, wie seit seinem fünfzigsten
Geburtstag amtlich sein Name lautete, hatte an einem
Frühlingsnachmittag des Jahres 19.., das unserem Kontinent monatelang
eine so gefahrdrohende Miene zeigte, von seiner Wohnung in der Prinz-
Regentenstraße zu München aus, allein einen weiteren Spaziergang
unternommen. Überreizt von der schwierigen und gefährlichen, eben
jetzt eine höchste Behutsamkeit, Umsicht, Eindringlichkeit und
Genauigkeit des Willens erfordernden Arbeit der Vormittagsstunden,
hatte der Schriftsteller dem Fortschwingen des produzierenden
Triebwerks in seinem Innern, jenem »motus animi continuus«, worin
nach Cicero das Wesen der Beredsamkeit besteht, auch nach der
Mittagsmahlzeit nicht Einhalt zu tun vermocht und den entlastenden
Schlummer nicht gefunden, der ihm, bei zunehmender Abnutzbarkeit
seiner Kräfte, einmal untertags so nötig war. So hatte er bald nach dem
Tee das Freie gesucht, in der Hoffnung, daß Luft und Bewegung ihn
wieder herstellen und ihm zu einem ersprießlichen Abend verhelfen
würden.

Es war Anfang Mai und, nach naßkalten Wochen, ein falscher
Hochsommer eingefallen. Der Englische Garten, obgleich nur erst zart
belaubt, war dumpfig wie im August und in der Nähe der Stadt voller
Wagen und Spaziergänger gewesen. Beim Aumeister, wohin stillere und
stillere Wege ihn geführt, hatte Aschenbach eine kleine Weile den
volkstümlich belebten Wirtsgarten überblickt, an dessen Rande einige
Droschken und Equipagen hielten, hatte von dort bei sinkender Sonne
seinen Heimweg außerhalb des Parks über die offene Flur genommen
und erwartete, da er sich müde fühlte und über Föhring Gewitter drohte,
am Nördlichen Friedhof die Tram, die ihn in gerader Linie zur Stadt
zurückbringen sollte. Zufällig fand er den Halteplatz und seine
Umgebung von Menschen leer. Weder auf der gepflasterten
Ungererstraße, deren Schienengeleise sich einsam gleißend gegen
Schwabing erstreckten, noch auf der Föhringer Chaussee war ein
Fuhrwerk zu sehen; hinter den Zäunen der Steinmetzereien, wo zu Kauf

TRANSLATED
==========
First chapter

Gustav Aschenbach or von Aschenbach, like since his fiftieth
Birthday officially his name was on one
Spring afternoon of the year 19 .. that our continent for months
showed such a threatening expression from his apartment in the Prince
Regentenstrasse to Munich, another walk alone
undertaken. Overexcited by the difficult and dangerous, just
now a very careful, careful, insistent and
Accuracy of the morning's work requiring will,
the writer had the swinging of the producing
Engine inside, that "motus animi continuus", in which
according to Cicero the essence of eloquence persists, even after the
Midday meal could not stop and the relieving
Slumber not found him, with increasing wear and tear
of his strength once was necessary during the day. So he had soon after
Tea sought the free, in the hope that air and movement would find him
restore it and help it to a profitable evening
would.

It was the beginning of May and, after wet and cold weeks, a wrong one
Midsummer occurred. The English Garden, although only tender
leafy, dull as in August and crowded near the city
Carriages and walkers. At the Aumeister, where quiet and
Aschenbach had walked the more quiet paths for a little while
overlooks a popular, lively pub garden, on the edge of which there are a few
Stops and equipages stopped from there when the sun was down
made his way home outside the park across the open corridor
and expected, since he felt tired and threatened thunderstorms over Foehring,
at the northern cemetery the tram that takes him in a straight line to the city
should bring back. By chance he found the stopping place and his
Environment of people empty. Neither on the paved
Ungererstrasse, the rail tracks of which glisten lonely against each other
Schwabing extended, was still on the Föhringer Chaussee
See wagon; behind the fences of the stonemasons where to buy
```

## 摘要

在这篇博文中，您了解了如何将 Tesseract 配置为 OCR 非英语语言。

大多数 Tesseract 安装会自然地处理多种语言，不需要额外的配置；但是，在某些情况下，您需要:

1.  手动下载宇宙魔方语言包
2.  设置`TESSDATA_PREFIX`环境变量来指向语言包
3.  请验证语言包目录是否正确

未能完成以上三个步骤可能会阻止您在非英语语言中使用 Tesseract，*所以请确保您严格遵循本教程中的步骤！*

如果你这样做，你不应该有任何非英语语言的 OCR 问题。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***