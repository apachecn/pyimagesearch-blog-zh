# 开始使用 EasyOCR 进行光学字符识别

> 原文：<https://pyimagesearch.com/2020/09/14/getting-started-with-easyocr-for-optical-character-recognition/>

在本教程中，您将学习如何使用 EasyOCR 包通过 Python 轻松执行光学字符识别和文本检测。

EasyOCR，顾名思义，是一个 Python 包，允许计算机视觉开发人员毫不费力地执行光学字符识别。

说到 OCR，EasyOCR 是迄今为止应用光学字符识别最直接的方式:

*   EasyOCR 包可以用一个简单的`pip`命令安装。
*   对 EasyOCR 包的依赖性很小，这使得配置 OCR 开发环境变得很容易。
*   一旦安装了 EasyOCR，只需要一个`import`语句就可以将包导入到您的项目中。
*   从那里，**你需要的只是两行代码来执行 OCR**——一行初始化`Reader`类，另一行通过`readtext`函数对图像进行 OCR。

听起来好得难以置信？

幸运的是，它不是——今天我将向您展示如何使用 EasyOCR 在您自己的项目中实现光学字符识别。

**要了解如何使用 EasyOCR 进行光学字符识别，*继续阅读。***

## **开始使用 EasyOCR 进行光学字符识别**

在本教程的第一部分，我们将简要讨论 EasyOCR 包。从那里，我们将配置我们的 OCR 开发环境并在我们的机器上安装 EasyOCR。

接下来，我们将实现一个简单的 Python 脚本，它通过 EasyOCR 包执行光学字符识别。您将直接看到实现 OCR(甚至是多种语言的 OCR 文本)是多么简单和直接。

我们将用 EasyOCR 结果的讨论来结束本教程。

### **什么是 EasyOCR 包？**

EasyOCR 包由一家专门从事光学字符识别服务的公司 [Jaided AI](https://jaided.ai/) 创建和维护。

EasyOCR 是使用 Python 和 PyTorch 库实现的。如果你有一个支持 CUDA 的 GPU，底层的 PyTorch 深度学习库可以大大加快你的文本检测和 OCR 速度*。*

截至本文撰写之时，EasyOCR 可以识别 58 种语言的文本，包括英语、德语、印地语、俄语、*等等！*easy ocr 的维护者计划在未来添加更多的语言。你可以在下一页找到 EasyOCR 支持的[语言的完整列表。](https://github.com/JaidedAI/EasyOCR#supported-languages)

目前，EasyOCR 仅支持对键入的文本进行 OCR。2020 年晚些时候，他们还计划发布一款手写识别模型。

### **如何在你的机器上安装 easy ocr**

要开始安装 EasyOCR，我的建议是遵循我的 *[pip install opencv](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)* 教程和**重要警告:**

**一定要在你的虚拟环境中安装**`opencv-python`***而不是*** `opencv-contrib-python` **。**此外，如果您在同一个环境中有这两个包，可能会导致意想不到的后果。如果你两个都安装了，pip 不太可能会抱怨，所以要注意检查一下`pip freeze`命令。

当然，这两个 OpenCV 包在前面提到的教程中都有讨论；一定要安装正确的。

并且*我的建议*是你在你的系统上为 EasyOCR 专用一个单独的 Python 虚拟环境( *[pip 安装 opencv](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)* 指南的**选项 B** )。

然而，尽管选项 B 建议将您的虚拟环境命名为`cv`，我还是建议将其命名为`easyocr`、`ocr_easy`，或者类似的名称。如果您看过我的个人系统，您会惊讶地发现，在任何给定的时间，我的系统上都有 10-20 个用于不同目的的虚拟环境，每个虚拟环境都有一个对我有意义的描述性名称。

您的安装步骤应该如下所示:

*   **步骤#1:** 安装 Python 3
*   **步骤#2:** 安装 pip
*   **第三步:**在你的*系统上安装`virtualenv`和`virtualenvwrapper`，*，包括按照指示编辑你的 Bash/ZSH 档案
*   **步骤#4:** 创建一个名为`easyocr`的 Python 3 虚拟环境(或者选择一个您自己选择的名称)，并使用`workon`命令确保它是活动的
*   **步骤#5:** 根据以下信息安装 OpenCV *和* EasyOCR

为了完成**步骤#1-#4，**确保**首先遵循**上面链接的安装指南。

当您准备好进行第五步时，只需执行以下操作:

```py
$ pip install opencv-python # NOTE: *not* opencv-contrib-python
$ pip install easyocr
```

```py
$ workon easyocr # replace `easyocr` with your custom environment name
$ pip freeze
certifi==2020.6.20
cycler==0.10.0
decorator==4.4.2
easyocr==1.1.7
future==0.18.2
imageio==2.9.0
kiwisolver==1.2.0
matplotlib==3.3.1
networkx==2.4
numpy==1.19.1
opencv-python==4.4.0.42
Pillow==7.2.0
pyparsing==2.4.7
python-bidi==0.4.2
python-dateutil==2.8.1
PyWavelets==1.1.1
scikit-image==0.17.2
scipy==1.5.2
six==1.15.0
tifffile==2020.8.13
torch==1.6.0
torchvision==0.7.0
```

请注意，安装了以下软件包:

*   `easyocr`
*   ``opencv-python``
*   `torch`和``torchvision``

还有一些其他的 EasyOCR 依赖项会自动为您安装。

**最重要的是，**正如我上面提到的，确保你的虚拟环境中安装了`opencv-python`和 ***而不是*** `opencv-contrib-python`。

如果你仔细按照我列出的步骤去做，你很快就可以开始工作了。一旦您的环境准备就绪，您就可以开始使用 EasyOCR 进行光学字符识别。

### **项目结构**

花点时间找到这篇博文的 ***【下载】*** 部分。在项目文件夹中，您会找到以下文件:

```py
$ tree --dirsfirst
.
├── images
│   ├── arabic_sign.jpg
│   ├── swedish_sign.jpg
│   └── turkish_sign.jpg
└── easy_ocr.py

1 directory, 4 files
```

今天的 EasyOCR 项目已经名副其实了。如你所见，我们有三个测试`images/`和一个 Python 驱动脚本`easy_ocr.py`。我们的驱动程序脚本接受任何输入图像和所需的 OCR 语言，以便很容易地完成工作，我们将在实现部分看到这一点。

### **使用 EasyOCR 进行光学字符识别**

配置了开发环境并检查了项目目录结构后，我们现在可以在 Python 脚本中使用 EasyOCR 包了！

打开项目目录结构中的`easy_ocr.py`文件，插入以下代码:

```py
# import the necessary packages
from easyocr import Reader
import argparse
import cv2
```

```py
def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()
```

如您所见，`cleanup_text`助手函数只是确保`text`字符串参数中的字符序数小于`128`，去掉任何其他字符。如果你对`128`的意义感到好奇，一定要看看任何标准的 [ASCII 字符表，比如这个](http://www.asciitable.com/)。

输入和便利实用程序准备就绪，现在让我们定义命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-l", "--langs", type=str, default="en",
	help="comma separated list of languages to OCR")
ap.add_argument("-g", "--gpu", type=int, default=-1,
	help="whether or not GPU should be used")
args = vars(ap.parse_args())
```

我们的脚本接受三个[命令行参数](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/):

*   `--image`:包含 OCR 文字的输入图像的路径。
*   ``--langs`` :逗号分隔的语言代码列表(无空格)。到`default`时，我们的脚本采用英语(`en`)。如果你想使用英国和法国模式，你可以通过`en,fr`。或者也许你想通过`es,pt,it`使用西班牙语、葡萄牙语和意大利语。请务必参考 EasyOCR 列出的[支持的语言](https://github.com/JaidedAI/EasyOCR#supported-languages)。
*   ``--gpu`` :是否要用 GPU。我们的`default`是`-1`，这意味着我们将使用我们的 CPU 而不是 GPU。如果您有支持 CUDA 的 GPU，启用此选项将允许更快的 OCR 结果。

给定我们的命令行参数，让我们**执行 OCR:**

```py
# break the input languages into a comma separated list
langs = args["langs"].split(",")
print("[INFO] OCR'ing with the following languages: {}".format(langs))

# load the input image from disk
image = cv2.imread(args["image"])

# OCR the input image using EasyOCR
print("[INFO] OCR'ing input image...")
reader = Reader(langs, gpu=args["gpu"] > 0)
results = reader.readtext(image)
```

```py
# loop over the results
for (bbox, text, prob) in results:
	# display the OCR'd text and associated probability
	print("[INFO] {:.4f}: {}".format(prob, text))

	# unpack the bounding box
	(tl, tr, br, bl) = bbox
	tl = (int(tl[0]), int(tl[1]))
	tr = (int(tr[0]), int(tr[1]))
	br = (int(br[0]), int(br[1]))
	bl = (int(bl[0]), int(bl[1]))

	# cleanup the text and draw the box surrounding the text along
	# with the OCR'd text itself
	text = cleanup_text(text)
	cv2.rectangle(image, tl, br, (0, 255, 0), 2)
	cv2.putText(image, text, (tl[0], tl[1] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
```

### **EasyOCR 结果**

我们现在可以看到 EasyOCR 库应用光学字符识别的结果了。

首先使用本教程的 ***【下载】*** 部分下载源代码和示例图像。

从那里，打开一个终端，并执行以下命令:

```py
$ python easy_ocr.py --image images/arabic_sign.jpg --langs en,ar
[INFO] OCR'ing with the following languages: ['en', 'ar']
[INFO] OCR'ing input image...
Using CPU. Note: This module is much faster with a GPU.
[INFO] 0.8129: خروج
[INFO] 0.7237: EXIT
```

EasyOCR 能够检测并正确识别输入图像中的英文和阿拉伯文文本。

***注意:**如果您是第一次使用 EasyOCR，您会看到在您的终端上打印出一个指示，说明 EasyOCR 正在“下载检测模型”下载文件时请耐心等待。一旦这些模型被缓存到您的系统中，您就可以无缝、快速地反复使用它们。*

让我们尝试另一个图像，这个图像包含一个瑞典标志:

```py
$ python easy_ocr.py --image images/swedish_sign.jpg --langs en,sv
[INFO] OCR'ing with the following languages: ['en', 'sv']
[INFO] OCR'ing input image...
Using CPU. Note: This module is much faster with a GPU.
[INFO] 0.7078: Fartkontrol
```

这里我们要求 EasyOCR 对英语(`en`)和瑞典语(`sv`)进行 OCR。

对于那些还不熟悉这个标志的人来说，*“Fartkontrol”*是瑞典人和丹麦人之间的一个小笑话。

直译过来，*Fartkontrol*在英文中的意思是*【速度控制】*(或者简称速度监控)。

但是当发音时，“*【屁控】*听起来像*“屁控”*——也许有人在控制自己的胀气方面有问题。在大学里，我有一个朋友在他们的浴室门上挂了一个瑞典*【法特控】*的标志——也许你不觉得这个笑话好笑，但每当我看到那个标志时，我都会暗自发笑(也许我只是一个不成熟的 8 岁小孩)。

最后一个例子，让我们看看土耳其的停车标志:

```py
$ python easy_ocr.py --image images/turkish_sign.jpg --langs en,tr
[INFO] OCR'ing with the following languages: ['en', 'tr']
[INFO] OCR'ing input image...
Using CPU. Note: This module is much faster with a GPU.
[INFO] 0.9741: DUR
```

我让 EasyOCR 对英语(`en`)和土耳其语(`tr`)文本进行 OCR，通过`--langs`命令行参数以逗号分隔列表的形式提供这些值。

EasyOCR 能够检测文本，*“DUR”，*，当从土耳其语翻译成英语时，该文本是*“停止”*

正如你所看到的，EasyOCR 名副其实——最后，一个易于使用的光学字符识别包！

此外，如果您有支持 CUDA 的 GPU，您可以通过提供`--gpu`命令行参数来获得*甚至更快的* OCR 结果，如下所示:

```py
$ python easy_ocr.py --image images/turkish_sign.jpg --langs en,tr --gpu 1
```

但是同样，您需要为 PyTorch 库配置一个 CUDA GPU(easy ocr 使用 PyTorch 深度学习库)。

## **总结**

在本教程中，您学习了如何使用 EasyOCR Python 包执行光学字符识别。

与 Tesseract OCR 引擎和 [pytesseract 包](https://github.com/madmaze/pytesseract)不同，如果您是光学字符识别领域的新手，使用它们可能会有点乏味，EasyOCR 包名副其实——**easy OCR 使 Python 的光学字符识别变得“简单”**

此外，EasyOCR 还有许多好处:

1.  您可以使用 GPU 来提高光学字符识别管道的速度。
2.  您可以使用 EasyOCR 同时对多种语言的文本进行 OCR。
3.  EasyOCR API 是 Pythonic 式的，使用起来简单直观。

我在我的书 *OCR with OpenCV、Tesseract 和 Python* 中介绍了 EasyOCR 如果您有兴趣了解更多关于光学字符识别的知识，一定要看看！

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***