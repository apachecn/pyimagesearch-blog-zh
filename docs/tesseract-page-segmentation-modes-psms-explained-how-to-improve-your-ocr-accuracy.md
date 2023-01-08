# Tesseract 页面分割模式(PSMs)解释:如何提高您的 OCR 准确性

> 原文：<https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/>

大多数介绍 Tesseract 的教程都会向您提供在您的机器上安装和配置 Tesseract 的说明，提供一两个如何使用`tesseract`二进制文件的例子，然后可能会提供如何使用一个库(如`pytesseract` — **)将 Tesseract 与 Python 集成的例子。这些介绍教程的问题在于它们没有抓住页面分段模式(PSM)的重要性。**让我们用 OCR 找到靶心。

## **Tesseract 页面分割模式(PSMs)讲解:如何提高您的 OCR 准确度**

在阅读完这些指南后，计算机视觉/深度学习实践者得到的印象是，无论图像有多简单或多复杂，OCR 识别都像打开外壳、执行`tesseract`命令并提供输入图像的路径一样简单(即，没有额外的选项或配置)。

大多数情况下(对于复杂的图像，几乎*总是*)，宇宙魔方要么:

1.  无法光学字符识别(OCR) *图像中的任何*文本，返回空结果
2.  尝试对文本进行 OCR，但是完全不正确，返回无意义的结果

事实上，当我在大学开始使用 OCR 工具时，情况就是这样。我在网上阅读了一两个教程，浏览了文档，当我不能获得正确的 OCR 结果时，我很快变得沮丧。我完全不知道如何以及何时使用不同的选项。我甚至不知道有一半的选项是控制什么的，因为文档是如此的稀疏，没有提供具体的例子！

**我犯的错误，也可能是我看到的现在初露头角的 OCR 从业者犯的最大问题之一，是没有完全理解 Tesseract 的页面分割模式如何能够*****显著地影响 OCR 输出的准确性。***

 *当使用 Tesseract OCR 引擎时，*你绝对必须对 Tesseract 的 PSM*感到舒适*——没有它们，你很快就会变得沮丧，并且无法获得高 OCR 准确度。*

在本教程中，您将了解关于 Tesseract 的 14 种页面分割模式的所有信息，包括:

*   他们做什么
*   如何设置它们
*   何时使用它们(从而确保您能够正确地对输入图像进行 OCR)

让我们开始吧！

## **学习目标**

在本教程中，您将:

*   了解什么是页面分段模式(PSM)
*   了解选择 PSM 是如何区分正确的 OCR 结果和不正确的 OCR 结果的
*   查看 Tesseract OCR 引擎内置的 14 个 PSM
*   查看 14 种 PSM 的实际应用示例
*   使用这些 PSM 时，发现我的提示、建议和最佳实践

**要了解如何使用 PSM 提高 OCR 结果，** ***继续阅读。***

## **镶嵌页面分割模式**

在本教程的第一部分，我们将讨论什么是页面分割模式(PSM)，为什么它们很重要，以及它们如何*显著地*影响我们的 OCR 准确度。

在这里，我们将回顾本教程的项目目录结构，然后探究 Tesseract OCR 引擎内置的 14 个 PSM 中的每一个。

本教程将以讨论我的技巧、建议和使用 Tesseract 应用各种 PSM 时的最佳实践作为结束。

### **什么是页面分割模式？**

我看到初学 OCR 的从业者无法获得正确的 OCR 结果的首要原因是他们使用了不正确的页面分割模式。引用宇宙魔方文档，默认情况下，**宇宙魔方在分割输入图像** ( [*提高输出质量*](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html) )时，期望有一页*的文本*。

“一页文字”的假设非常重要。如果你正在对一本书的扫描章节进行 OCR，默认的 Tesseract PSM 可能会很适合你。但是如果你试图只对一个*单行*，一个*单字*，或者甚至一个*单字*进行 OCR，那么这个默认模式将会导致一个空字符串或者无意义的结果。

把宇宙魔方想象成你小时候长大的大哥哥。他真心实意地关心你，希望看到你快乐——但与此同时，他毫不犹豫地把你推倒在沙盒里，让你满嘴砂砾，也不会伸出援助之手让你重新站起来。

我的一部分认为这是一个用户体验(UX)的问题，可以由宇宙魔方开发团队来改善。包括一条短信说:

> *没有得到正确的 OCR 结果？尝试使用不同的页面分段模式。运行`tesseract --help-extra`可以看到所有的 PSM 模式。*

也许他们甚至可以链接到一个教程，用简单易懂的语言解释每个 PSM。从那里，最终用户可以更成功地将 Tesseract OCR 引擎应用到他们自己的项目中。

但是在那个时候到来之前，Tesseract 的页面分割模式，尽管是获得高 OCR 准确度的一个关键方面，对许多 OCR 新手来说还是一个谜。他们不知道它们是什么，如何使用它们，为什么它们很重要— **许多人甚至不知道在哪里可以找到各种页面分割模式！**

要列出 Tesseract 中的 14 个 PSM，只需向`tesseract`二进制文件提供`--help-psm`参数:

```py
$ tesseract --help-psm
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
       bypassing hacks that are Tesseract-specific.
```

然后，您可以通过为`--psm`参数提供相应的整数值来应用给定的 PSM。

例如，假设我们有一个名为`input.png`的输入图像，我们想要使用 PSM `7`，它用于 OCR 单行文本。因此，我们对`tesseract`的调用如下所示:

```py
$ tesseract input.png stdout --psm 7
```

在本教程的其余部分，我们将回顾 14 个宇宙魔方 PSM 的每一个。您将获得使用它们的实践经验，并且在完成本教程后，您会对使用 Tesseract OCR 引擎正确识别图像的能力更加有信心。

### **项目结构**

不像大多数教程，其中包括一个或多个 Python 脚本来回顾，本教程是为数不多的*不*利用 Python 的教程之一。相反，我们将使用`tesseract`二进制来探索每种页面分段模式。

请记住，本教程旨在了解 PSM 并获得使用它们的第一手经验。一旦你对它们有了很深的理解，这些知识就直接转移到 Python 上了。要在 Python 中设置 PSM，就像设置一个选项变量一样简单— *简单得不能再简单了，实际上只需要几次击键！*

因此，我们将首先从`tesseract`二进制开始。

说完这些，让我们看看我们的项目目录结构:

```py
|-- psm-0
|   |-- han_script.jpg
|   |-- normal.png
|   |-- rotated_90.png
|-- psm-1
|   |-- example.png
|-- psm-3
|   |-- example.png
|-- psm-4
|   |-- receipt.png
|-- psm-5
|   |-- receipt_rotated.png
|-- psm-6
|   |-- sherlock_holmes.png
|-- psm-7
|   |-- license_plate.png
|-- psm-8
|   |-- designer.png
|-- psm-9
|   |-- circle.png
|   |-- circular.png
|-- psm-10
|   |-- number.png
|-- psm-11
|   |-- website_menu.png
|-- psm-13
|   |-- the_old_engine.png
```

如您所见，我们有 13 个目录，每个目录中都有一个示例图像，它会突出显示何时使用该特定的 PSM。

但是等等 *…* 我之前在教程里不是说宇宙魔方有 *14* ，没有 *13* 吗，页面分割模式？如果是，为什么没有 14 个目录？

答案很简单——其中一个 PSM 没有在 Tesseract 中实现。它本质上只是未来潜在实现的占位符。

让我们开始探索使用 Tesseract 的页面分段模式吧！

### **PSM 0。仅定向和脚本检测**

`--psm 0`模式*并不*执行 OCR，至少在本书的上下文中我们是这样认为的。当我们想到 OCR 时，我们会想到一个软件，它能够定位输入图像中的字符，识别它们，然后将它们转换为机器编码的字符串。

方向和文字检测(OSD)检查输入图像，但 OSD 返回两个值，而不是返回实际的 OCR 文本:

1.  页面的方向，以度为单位，其中`angle = {0, 90, 180, 270}`
2.  文字的置信度(即图形符号/ [书写系统](https://en.wikipedia.org/wiki/Writing_system#General_properties))，如拉丁文、汉文、西里尔文等。

OSD 最好用一个例子来看。看一下**图 1** ，这里有三个示例图像。第一个是我第一本书 [*实用 Python 和 OpenCV*](https://pyimagesearch.com/practical-python-opencv/) 的一段文字。第二张是同一段文字，这次顺时针旋转了 90 度，最后的图像包含了汉文。

让我们从将`tesseract`应用到`normal.png`图像开始，该图像显示在**图** 1 中*左上角*处:

```py
$ tesseract normal.png stdout --psm 0
Page number: 0
Orientation in degrees: 0
Rotate: 0
Orientation confidence: 11.34
Script: Latin
Script confidence: 8.10
```

在这里，我们可以看到，宇宙魔方已经确定这个输入图像是未旋转的(即 0)，并且脚本被正确地检测为`Latin`。

现在让我们把同样的图像旋转 90 度，如图 1**(*右上*)所示:**

```py
$ tesseract rotated_90.png stdout --psm 0
Page number: 0
Orientation in degrees: 90
Rotate: 270
Orientation confidence: 5.49
Script: Latin
Script confidence: 4.76
```

宇宙魔方确定输入的图像已经旋转了 90，为了纠正图像，我们需要将其旋转 270。同样，脚本被正确地检测为`Latin`。

最后一个例子，我们现在将把 Tesseract OSD 应用于汉字图像(**图** 1，*底*):

```py
$ tesseract han_script.jpg stdout --psm 0
Page number: 0
Orientation in degrees: 0
Rotate: 0
Orientation confidence: 2.94
Script: Han
Script confidence: 1.43
```

注意脚本是如何被正确标记为`Han`的。

你可以把`--psm 0`模式想象成一种“元信息”模式，在这种模式下，宇宙魔方*只为你提供*输入图像的脚本和旋转——**当应用这种模式时，宇宙魔方*不会* OCR 实际文本并返回给你。**

如果你需要*只是*文本上的元信息，使用`--psm 0`是适合你的模式；然而，很多时候我们需要 OCR 文本本身，在这种情况下，您应该使用本教程中介绍的其他 PSM。

### **PSM 1。带 OSD 的自动页面分割**

关于`--psm 1`的 Tesseract 文档和示例并不完整，因此很难提供关于这种方法的详细研究和示例。我对`--psm 1`的理解是:

1.  应该执行 OCR 的自动页面分割
2.  并且应该在 OCR 过程中推断和利用 OSD 信息

然而，如果我们采用图 1 的**中的图像，并使用此模式将它们通过`tesseract`，您可以看到没有 OSD 信息:**

```py
$ tesseract example.png stdout --psm 1
Our last argument is how we want to approximate the
contour. We use cv2.CHAIN_APPROX_SIMPLE to compress
horizontal, vertical, and diagonal segments into their end-
points only. This saves both computation and memory. If
we wanted all the points along the contour, without com-
pression, we can pass in cv2\. CHAIN_APPROX_NONE; however,
be very sparing when using this function. Retrieving all
points along a contour is often unnecessary and is wasteful
of resources.
```

这个结果让我觉得 Tesseract 一定是在内部执行 OSD 但是没有返回给用户。根据我对`--psm 1`的实验和经验，我认为可能是`--psm 2`没有完全工作/实现。

简单地说:在我所有的实验中，我找不到一种情况是`--psm 1`获得了其他 PSM 不能获得的结果。如果以后我发现这样的情况，我会更新这一部分，并提供一个具体的例子。但在那之前，我认为不值得在你的项目中应用`--psm 1`。

### **PSM 2。自动页面分割，但没有 OSD 或 OCR**

在 Tesseract 中没有实现`--psm 2`模式。您可以通过运行`tesseract --help-psm`命令查看模式二的输出来验证这一点:

```py
$ tesseract --help-psm
Page segmentation modes:
...
  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
...
```

目前还不清楚 Tesseract 是否或何时会实现这种模式，但就目前而言，您可以放心地忽略它。

### **PSM 3。全自动页面分割，但没有 OSD**

**PSM 3 是宇宙魔方的默认行为。**如果你运行`tesseract`二进制而没有明确提供一个`--psm`，那么将使用一个`--psm 3`。

在这种模式下，宇宙魔方将:

1.  自动尝试对文本进行分段，将其视为包含多个单词、多行、多段等的文本“页面”。
2.  分割后，Tesseract 将对文本进行 OCR，并将其返回给您

然而，重要的是要注意，宇宙魔方*不会*执行任何方向/脚本检测。为了收集这些信息，您需要运行两次`tesseract`:

1.  一次用`--psm 0`模式收集 OSD 信息
2.  然后再次使用`--psm 3`对实际文本进行 OCR

以下示例显示了如何获取一段文本，并在两个单独的命令中同时应用 OSD 和 OCR:

```py
$ tesseract example.png stdout --psm 0
Page number: 0
Orientation in degrees: 0
Rotate: 0
Orientation confidence: 11.34
Script: Latin
Script confidence: 8.10

$ tesseract example.png stdout --psm 3
Our last argument is how we want to approximate the
contour. We use cv2.CHAIN_APPROX_SIMPLE to compress
horizontal, vertical, and diagonal segments into their end-
points only. This saves both computation and memory. If
we wanted all the points along the contour, without com-
pression, we can pass in cv2\. CHAIN_APPROX_NONE; however,
be very sparing when using this function. Retrieving all
points along a contour is often unnecessary and is wasteful
of resources.
```

同样，如果您只想要 OCR 处理的文本，可以跳过第一个命令。

### **PSM 4。假设一列可变大小的文本**

使用`--psm 4`的一个很好的例子是当您需要 OCR 列数据并要求文本按行连接时(例如，您将在电子表格、表格或收据中找到的数据)。

例如，考虑**图** 2，这是杂货店的收据。让我们尝试使用默认(`--psm 3`)模式对该图像进行 OCR:

```py
$ tesseract receipt.png stdout
ee

OLE
YOSDS:

cea eam

WHOLE FOODS MARKET - WESTPORT,CT 06880
399 POST RD WEST - (203) 227-6858

365
365
365
365

BACON LS
BACON LS
BACON LS
BACON LS

BROTH CHIC
FLOUR ALMOND
CHKN BRST BNLSS SK

HEAVY CREAM
BALSMC REDUCT

BEEF GRND 85/15
JUICE COF CASHEW C

DOCS PINT ORGANIC

HNY ALMOND BUTTER

wee TAX

.00

BAL

NP 4.99
NP 4.99
NP 4.99
NP 4.99
NP 2.19
NP 91.99
NP 18.80
NP 3.39
NP. 6.49
NP 5.04
ne £8.99
np £14.49
NP 9.99

101.33

aaa AAAATAT

ie
```

结果不太好。使用默认的`--psm 3`模式，Tesseract 无法推断我们正在查看列数据，并且同一行的文本应该*关联在一起。*

为了解决这个问题，我们可以使用`--psm 4`模式:

```py
$ tesseract receipt.png stdout --psm 4
WHOLE
FOODS.

cea eam

WHOLE FOODS MARKET - WESTPORT,CT 06880
399 POST RD WEST - (203) 227-6858

365 BACONLS NP 4.99

365 BACON LS NP 4.99

365 BACONLS NP 4.99

365  BACONLS NP 4,99
BROTH CHIC NP 2.19

FLOUR ALMOND NP 91.99

CHKN BRST BNLSS SK NP 18.80
HEAVY CREAM NP 3.39

BALSMC REDUCT NP 6.49

BEEF GRND 85/15 NP 6.04
JUICE COF CASHEW C NP £8.99
DOCS PINT ORGANIC NP 14,49
HNY ALMOND BUTTER NP 9,99
wee TAX = 00 BAL 101.33
```

如你所见，这里的结果要好得多。 Tesseract 能够理解文本应该按行进行分组，从而允许我们对收据中的项目进行 OCR。

### **PSM 5。假设一个单一的垂直对齐的文本块**

围绕`--psm 5`的文档有些混乱，因为它声明我们希望对垂直对齐的单个文本块进行 OCR。问题是“垂直对齐文本”的实际含义有点模糊(因为没有显示垂直对齐文本示例的 Tesseract 示例)。

对我来说，垂直对齐的文本要么放在页面顶部的*、页面中间的、页面底部的*。在图 3 的**中，**是顶部*对齐(*左*)、*中间*对齐(*中间*)和*底部*对齐(*右*)的文本示例。***

 *然而，在我自己的实验中，我发现`--psm 5`与`--psm 4`、*的工作方式相似，只是对于旋转的图像。*考虑**图 4、**图中，我们将一张收据顺时针旋转了 90 度来看看这个例子。

让我们首先应用默认的`--psm 3`:

```py
$ tesseract receipt_rotated.png stdout
WHOLE
FOODS.
(mM AR K E T)

WHOLE FOODS MARKET - WESTPORT,CT 06880
399 POST RD WEST - (203) 227-6858

365 BACON LS

365 BACON LS

365 BACON LS

365 BACON LS

BROTH CHIC

FLOUR ALMOND

CHKN BRST BNLSS SK

HEAVY CREAM

BALSMC REDUCT

BEEF GRND 85/15

JUICE COF CASHEW C

DOCS PINT ORGANIC

HNY ALMOND BUTTER
eee TAX  =.00 BAL

ee

NP 4.99
NP 4.99
NP 4,99
NP 4.99
NP 2.19
NP 1.99
NP 18.80
NP 3.39
NP 6.49
NP 8.04
NP £8.99
np "14.49
NP 9.99

101.33

aAnMAIATAAT AAA ATAT

ie
```

再说一次，我们在这里的结果并不好。虽然宇宙魔方可以纠正旋转，但我们没有收据的行元素。

为了解决这个问题，我们可以使用`--psm 5`:

```py
$ tesseract receipt_rotated.png stdout --psm 5
Cea a amD

WHOLE FOODS MARKET - WESTPORT, CT 06880

399 POST RD WEST - (203) 227-6858
* 365 BACONLS NP 4.99 F
* 365 BACON LS NP 4.99 F
* 365 BACONLS NP 4,99 F*
* 365  BACONLS NP 4.99 F
* BROTH CHIC NP 2.19 F
* FLOUR ALMOND NP 1.99 F
* CHKN BRST BNLSS SK NP 18.80 F
* HEAVY CREAM NP 3.39 F
* BALSMC REDUCT NP 6.49 F
* BEEF GRND 85/1§ NP {6.04 F
* JUICE COF CASHEW C NP [2.99 F
*, DOCS PINT ORGANIC NP "14.49 F
* HNY ALMOND BUTTER NP 9,99

wee TAX = 00 BAL 101.33
```

在旋转后的收据图像上，我们的 OCR 结果现在要好得多。

### **PSM 6。假设一个统一的文本块**

我喜欢用`--psm 6`来识别简单书籍的页面(例如，一本平装小说)。书中的页面倾向于在整本书中使用单一、一致的字体。同样，这些书遵循简单的页面结构，这对于 Tesseract 来说很容易解析和理解。

**这里的关键字是** ***统一文本，*** 的意思是文本是没有任何变化的单一字体。

下面显示了在默认`--psm 3`模式下，对夏洛克·福尔摩斯小说(**图** 5)中的单个统一文本块应用宇宙魔方的结果:

```py
$ tesseract sherlock_holmes.png stdout
CHAPTER ONE

we
Mr. Sherlock Holmes

M: Sherlock Holmes, who was usually very late in the morn-
ings, save upon those not infrequent occasions when he
was up all night, was seated at the breakfast table. I stood upon
the hearth-rug and picked up the stick which our visitor had left
behind him the night before. It was a fine, thick piece of wood,
bulbous-headed, of the sort which is known as a “Penang lawyer.”
Just under the head was a broad silver band nearly an inch across.
“To James Mortimer, M.R.C.S., from his friends of the C.C.H.,”
was engraved upon it, with the date “1884.” It was just such a
stick as the old-fashioned family practitioner used to carry--dig-
nified, solid, and reassuring.

“Well, Watson, what do you make of it2”

Holmes w:

sitting with his back to me, and I had given him no
sign of my occupation.

“How did you know what I was doing? I believe you have eyes in
the back of your head.”

“L have, at least, a well-polished, silver-plated coffee-pot in front
of me,” said he. “But, tell me, Watson, what do you make of our
visitor's stick? Since we have been so unfortunate as to miss him

and have no notion of his errand, this

accidental souvenir be-
comes of importance, Let me hear you reconstruct the man by an
examination of it.”
```

为了节省空间，我从上面的输出中删除了许多换行符。如果您在自己的系统中运行上面的命令，您会看到输出比文本中显示的更加混乱。

通过使用`--psm 6`模式，我们能够更好地对这一大块文本进行 OCR:

```py
$ tesseract sherlock_holmes.png stdout --psm 6
CHAPTER ONE
SS
Mr. Sherlock Holmes
M Sherlock Holmes, who was usually very late in the morn
ings, save upon those not infrequent occasions when he

was up all night, was seated at the breakfast table. I stood upon
the hearth-rug and picked up the stick which our visitor had left
behind him the night before. It was a fine, thick piece of wood,
bulbous-headed, of the sort which is known as a “Penang lawyer.”
Just under the head was a broad silver band nearly an inch across.
“To James Mortimer, M.R.C.S., from his friends of the C.C.H.,”
was engraved upon it, with the date “1884.” It was just such a
stick as the old-fashioned family practitioner used to carry--dig-
nified, solid, and reassuring.

“Well, Watson, what do you make of it2”

Holmes was sitting with his back to me, and I had given him no
sign of my occupation.

“How did you know what I was doing? I believe you have eyes in
the back of your head.”

“T have, at least, a well-polished, silver-plated coflee-pot in front
of me,” said he. “But, tell me, Watson, what do you make of our
visitor’s stick? Since we have been so unfortunate as to miss him
and have no notion of his errand, this accidental souvenir be-
comes of importance. Let me hear you reconstruct the man by an
examination of it.”

6
```

这个输出中的错误要少得多，从而演示了如何使用`--psm 6`来 OCR 统一文本块。

#### **PSM 7。将图像视为单个文本行**

当你处理一个*单行*的统一文本时，应该使用`--psm 7`模式。例如，让我们假设我们正在构建一个自动牌照/号码牌识别(ANPR)系统，并且需要对图**图** 6 中的牌照进行 OCR。

让我们从使用默认的`--psm 3`模式开始:

```py
$ tesseract license_plate.png stdout
Estimating resolution as 288
Empty page!!
Estimating resolution as 288
Empty page!!
```

默认的宇宙魔方模式停滞不前，完全无法识别车牌。

但是，如果我们使用`--psm 7`并告诉 Tesseract 将输入视为单行统一文本，我们就能够获得正确的结果:

```py
$ tesseract license_plate.png stdout --psm 7
MHOZDW8351
```

### **PSM 8。将图像视为一个单词**

如果你有一个统一文本的*单字*，你应该考虑使用`--psm 8`。一个典型的使用案例是:

1.  对图像应用文本检测
2.  在所有文本 ROI 上循环
3.  提取它们
4.  将每个单独的文本 ROI 通过 Tesseract 进行 OCR

例如，让我们考虑一下**图 7** ，这是一张店面的照片。我们可以尝试使用默认的`--psm 3`模式对该图像进行 OCR:

```py
$ tesseract designer.png stdout
MS atts
```

但不幸的是，我们得到的都是胡言乱语。

为了解决这个问题，我们可以使用`--psm 8`，告诉 Tesseract 绕过任何页面分割方法，而是将该图像视为一个单词:

```py
$ tesseract designer.png stdout --psm 8
Designer
```

果然，`--psm 8`能够解决问题！

此外，您可能会发现`--psm 7`和`--psm 8`可以互换使用的情况——两者的功能类似，因为我们分别在查看*单行*或*单字*。

### **PSM 9。将图像视为圆圈中的一个单词**

我已经玩了几个小时的`--psm 9`模式，真的，*我不知道它能做什么。*我已经搜索了谷歌并阅读了宇宙魔方文档，但是一无所获——我找不到一个关于圆形 PSM 打算做什么的*具体例子*。

对我来说，这个参数有两种解读方式(**图 8** ):

1.  文字其实是圆圈内的**(左)**
2.  文字被*包裹在*一个看不见的圆形/弧形区域*(右)*

对我来说，第二种选择似乎更有可能，但是无论我怎么努力，我都无法让这个参数起作用。我认为可以放心地假设这个参数很少使用——此外，实现可能会有一点问题。我建议尽量避免这种 PSM。

### **PSM 10。将图像视为单个字符**

当*已经从图像中提取出每个单独的字符*时，就应该将图像视为单个字符。

回到我们在 ANPR 的例子，假设您已经在输入图像中定位了车牌，然后提取了车牌上的每个字符*——然后您可以使用`--psm 10`将每个字符通过 Tesseract 进行 OCR。*

**图 9** 显示了数字`2`的一个例子。让我们试着用默认的`--psm 3`进行 OCR:

```py
$ tesseract number.png stdout
Estimating resolution as 1388
Empty page!!
Estimating resolution as 1388
Empty page!!
```

Tesseract 试图应用自动页面分割方法，但是由于没有实际的文本“页面”,默认的`--psm 3`失败并返回一个空字符串。

我们可以通过`--psm 10`将输入图像视为单个字符来解决这个问题:

```py
$ tesseract number.png stdout --psm 10
2
```

果然，`--psm 10`解决了这件事！

### **PSM 11。稀疏文本:找到尽可能多的文本，不分先后**

当图像中有大量文本需要提取时，检测稀疏文本会很有用。当使用这种模式时，你通常不关心文本的顺序/分组，而是关心文本本身的 T2。

如果您通过 OCR 在图像数据集中找到的所有文本来执行信息检索(即文本搜索引擎)，然后通过信息检索算法(tf-idf、倒排索引等)构建基于文本的搜索引擎，则此信息非常有用。).

**图 10** 显示了一个稀疏文本的例子。这里有我在 PyImageSearch 上 *[【入门】](http://pyimg.co/getstarted)* 页面的截图。该页面提供按流行的计算机视觉、深度学习和 OpenCV 主题分组的教程。

让我们尝试使用默认的`--psm 3`来 OCR 这个主题列表:

```py
$ tesseract website_menu.png stdout
How Do | Get Started?
Deep Learning
Face Applications

Optical Character Recognition (OCR)

Object Detection
Object Tracking

Instance Segmentation and Semantic

Segmentation

Embedded and lol Computer Vision

Computer Vision on the Raspberry Pi

Medical Computer Vision
Working with Video
Image Search Engines

Interviews, Case Studies, and Success Stories

My Books and Courses
```

虽然 Tesseract 可以对文本进行 OCR，但是有几个不正确的行分组和额外的空白。额外的空白和换行符是 Tesseract 的自动页面分割算法工作的结果——这里它试图推断文档结构，而实际上*没有文档结构。*

为了解决这个问题，我们可以用`--psm 11`将输入图像视为稀疏文本:

```py
$ tesseract website_menu.png stdout --psm 11
How Do | Get Started?

Deep Learning

Face Applications

Optical Character Recognition (OCR)

Object Detection

Object Tracking

Instance Segmentation and Semantic

Segmentation

Embedded and lol Computer Vision

Computer Vision on the Raspberry Pi

Medical Computer Vision

Working with Video

Image Search Engines

Interviews, Case Studies, and Success Stories

My Books and Courses
```

这次来自宇宙魔方的结果要好得多。

### **PSM 12。带 OSD 的稀疏文本**

`--psm 12`模式与`--psm 11`基本相同，但现在增加了 OSD(类似于`--psm 0`)。

也就是说，我在让这种模式正常工作时遇到了很多问题，并且找不到结果与`--psm 11`有意义不同的实际例子。

我觉得有必要说一下`--psm 12`的存在；然而，在实践中，如果您想要复制预期的`--psm 12`行为，您应该使用`--psm 0`(用于 OSD)后跟`--psm 11`(用于稀疏文本的 OCR)的组合。

### **PSM 13。原始行:将图像视为一个单独的文本行，绕过特定于宇宙魔方的 Hacks】**

有时，OSD、分段和其他内部特定于 Tesseract 的预处理技术会通过以下方式损害 OCR 性能:

1.  降低精确度
2.  根本没有检测到文本

通常，如果一段文本被紧密裁剪，该文本是计算机生成的/以某种方式风格化的，或者它是 Tesseract 可能无法自动识别的字体，就会发生这种情况。**当这种情况发生时，考虑将`--psm 13`作为“最后手段”**

要查看这种方法的实际应用，请考虑图 11 中的文本*“旧引擎”*以风格化的字体打印，类似于旧时代的报纸。

让我们尝试使用默认的`--psm 3`来 OCR 这个图像:

```py
$ tesseract the_old_engine.png stdout
Warning. Invalid resolution 0 dpi. Using 70 instead.
Estimating resolution as 491
Estimating resolution as 491
```

Tesseract 无法对图像进行 OCR，返回一个空字符串。

现在让我们使用`--psm 13`，绕过所有页面分割算法和镶嵌预处理函数，从而将图像视为一行原始文本:

```py
$ tesseract the_old_engine.png stdout --psm 13
Warning. Invalid resolution 0 dpi. Using 70 instead.
THE OLD ENGINE.
```

这一次我们能够正确地使用`--psm 13`对文本进行 OCR！

使用`--psm 13`有时有点麻烦，所以先尝试其他页面分割模式。

### 【Tips 的技巧、建议和最佳实践

**习惯宇宙魔方中的页面分割模式需要** ***练习*****——没有别的办法。我强烈建议你:**

1.  多次阅读本教程
2.  运行本教程文本中包含的示例
3.  然后开始用你自己的图像练习

不幸的是，Tesseract 没有包括太多关于 PSM 的文档，也没有容易参考的具体例子。本教程尽我所能为您提供尽可能多的关于 PSM 的信息，包括您何时想要使用每种 PSM 的实际例子。

也就是说，这里有一些提示和建议可以帮助您快速使用 PSM:

*   总是从默认的`--psm 3`开始，看看宇宙魔方吐出什么。在最好的情况下，OCR 结果是准确的，您就大功告成了。在最坏的情况下，你现在有一个基线要超越。
*   虽然我提到过`--psm 13`是一种“最后手段”的模式，但我还是建议第二次使用它。这个模式出奇的好，*尤其是*如果你已经预处理了你的图像和二值化了你的文本。如果`--psm 13`起作用，你可以停止或者将你的努力集中在模式 4-8 上，因为它们中的一个很可能会代替模式 13。
*   接下来，应用`--psm 0`验证旋转和脚本是否被正确检测。如果不是这样，期望 Tesseract 在不能正确检测旋转角度和脚本/书写系统的图像上表现良好是不合理的。
*   如果脚本和角度被正确检测，你需要遵循本教程中的指导方针。具体来说，您应该关注 PSM 4-8、10 和 11。避免 PSMs 1、2、9 和 12，除非您认为它们有特定的用例。
*   最后，您可能要考虑强制使用它。依次尝试 1-13 中的每种模式。这是一种“往墙上扔意大利面，看看会粘上什么”的黑客行为，但你有时会很幸运。

**如果你发现宇宙魔方没有给你想要的精度*****不管你用的是什么页面分割模式，不要惊慌或沮丧——这都是过程的一部分。OCR 部分是艺术，部分是科学。列奥纳多·达·芬奇并没有一开始就画蒙娜丽莎。这是一项需要练习才能获得的技能。***

 *我们只是触及了 OCR 的皮毛。未来的教程将进行更深入的探讨，并帮助您更好地磨练这种艺术。通过实践，如**图 12** 所示，您也将通过您的 OCR 项目击中靶心。

## **总结**

在本教程中，您学习了 Tesseract 的 14 种页面分段模式(PSM)。应用正确的 PSM 对于正确地对输入图像进行 OCR 是绝对关键的。

简而言之，你在 PSM 中的选择可能意味着一张精确 OCR 的图像与从宇宙魔方得到的*无结果*或*无意义结果*之间的差异。

Tesseract 中的 14 个 PSM 中的每一个都对您的输入图像做出假设，例如一个文本块(例如，扫描的一章)、一行文本(可能是一章中的一句话)，甚至是一个单词(例如，牌照/车牌)。

获得准确 OCR 结果的关键是:

1.  使用 OpenCV(或您选择的图像处理库)来清理您的输入图像，去除噪声，并可能从背景中分割文本
2.  应用 Tesseract，注意使用与任何预处理输出相对应的正确 PSM

例如，如果您正在构建一个自动车牌识别器(我们将在以后的教程中完成)，那么我们将利用 OpenCV 首先检测图像中的车牌。这可以使用图像处理技术或专用对象检测器来完成，例如 [HOG +线性 SVM](https://ieeexplore.ieee.org/document/1467360) 、[fast R-CNN](http://arxiv.org/abs/1506.01497)、 [SSD](http://arxiv.org/abs/1512.02325) 、 [YOLO](http://arxiv.org/abs/1506.02640) 等。

一旦我们检测到车牌，我们将从车牌中分割出字符，这样字符在黑色背景下显示为白色(前景)。

最后一步是将二进制化的车牌字符通过 Tesseract 进行 OCR。我们在 PSM 中的选择将是*正确的*和*错误的*结果之间的差异。

因为牌照可以被看作是“单行文本”或“单个单词”，所以我们想要尝试使用`--psm 7`或`--psm 8`。一个`--psm 13` *可能*也能工作，但是使用默认的(`--psm 3`)在这里不太可能工作，因为我们已经*已经*处理了我们的图像质量。

我强烈建议你花相当多的时间探索本教程中的所有例子，甚至回过头来再读一遍——从 Tesseract 的页面分割模式中可以获得如此多的知识。

从那里，开始应用各种 PSM 到你自己的图像。一路记录你的结果:

*   他们是你所期望的吗？
*   你得到正确的结果了吗？
*   Tesseract 是否未能对图像进行 OCR，返回空字符串？
*   宇宙魔方是否返回了完全无意义的结果？

你对 PSM 的实践越多，你获得的经验就越多，这将使你更容易正确地将 OCR 应用到你自己的项目中。*****