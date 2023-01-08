# 使用 Microsoft 认知服务进行文本检测和 OCR

> 原文：<https://pyimagesearch.com/2022/03/28/text-detection-and-ocr-with-microsoft-cognitive-services/>

* * *

* * *

## **[微软认知服务的文本检测和 OCR](#TOC)**

本课是关于**文本检测和 OCR** 的 3 部分系列的一部分:

1.  *[使用亚马逊 Rekognition API 进行文本检测和 OCR](https://pyimg.co/po6tf)*
2.  ***微软认知服务的文本检测和 OCR*(今日教程)**
3.  *使用谷歌云视觉 API 进行文本检测和 OCR*

在本教程中，您将:

*   了解如何获取您的 MCS API 密钥
*   创建一个配置文件来存储您的订阅密钥和 API 端点 URL
*   实现 Python 脚本来调用 MCS OCR API
*   查看将 MCS OCR API 应用于样本图像的结果

**要了解文本检测和 OCR，*继续阅读。***

## **[微软认知服务的文本检测和 OCR](#TOC)**

* * *

在我们的[之前的教程](https://pyimg.co/po6tf)中，您学习了如何使用 Amazon Rekognition API 来 OCR 图像。使用 Amazon Rekognition API 最困难的部分是获取 API 密钥。然而，一旦你有了你的 API 密匙，事情就一帆风顺了。

本教程重点介绍一种不同的基于云的 API，称为微软认知服务(MCS)，是微软 Azure 的一部分。与 Amazon Rekognition API 一样，MCS 也具有很高的 OCR 准确性，但不幸的是，实现稍微复杂一些(微软的登录和管理仪表板也是如此)。

与 MCS 相比，我们更喜欢 Amazon Web Services(AWS)Rekognition API，无论是对于管理仪表板还是 API 本身。然而，如果你已经根深蒂固地融入了 MCS/Azure 生态系统，你应该考虑留在那里。MCS API 并没有*那么*难用(只是不像 Amazon Rekognition API 那么简单)。

* * *

### **[微软 OCR 认知服务](#TOC)**

我们将从如何获得 MCS API 密钥的回顾开始本教程。**您将需要这些 API 键来请求** **MCS API 对图像进行 OCR。**

一旦我们有了 API 密钥，我们将检查我们的项目目录结构，然后实现一个 Python 配置文件来存储我们的订阅密钥和 OCR API 端点 URL。

实现了配置文件后，我们将继续创建第二个 Python 脚本，这个脚本充当驱动程序脚本，它:

*   导入我们的配置文件
*   将输入图像加载到磁盘
*   将图像打包到 API 调用中
*   向 MCS OCR API 发出请求
*   检索结果
*   注释我们的输出图像
*   将 OCR 结果显示到我们的屏幕和终端上

让我们开始吧！

* * *

### **[获取您的微软认知服务密钥](#TOC)**

在继续其余部分之前，请确保按照此处所示的[说明获取 API 密钥。](http://pyimg.co/53rux)

* * *

### **[配置您的开发环境](#TOC)**

要遵循本指南，您需要在系统上安装 OpenCV 和 Azure 计算机视觉库。

幸运的是，两者都是 pip 可安装的:

```py
$ pip install opencv-contrib-python
$ pip install azure-cognitiveservices-vision-computervision
```

**如果您需要帮助配置 OpenCV 的开发环境，我们*强烈推荐*阅读我们的** [***pip 安装 OpenCV* 指南**](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让您启动并运行。

* * *

### **[在配置开发环境时遇到了问题？](#TOC)**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://www.pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

* * *

### **[项目结构](#TOC)**

我们首先需要回顾我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

我们的 MCS OCR API 的目录结构类似于前面教程中的 Amazon Rekognition API 项目的结构:

```py
|-- config
|   |-- __init__.py
│   |-- microsoft_cognitive_services.py
|-- images
|   |-- aircraft.png
|   |-- challenging.png
|   |-- park.png
|   |-- street_signs.png
|-- microsoft_ocr.py
```

在`config`中，我们有我们的`microsoft_cognitive_services.py`文件，它存储了我们的订阅密钥和端点 URL(即，我们提交图像的 API 的 URL)。

`microsoft_ocr.py`脚本将获取我们的订阅密钥和端点 URL，连接到 API，将我们的`images`目录中的图像提交给 OCR，并显示我们的屏幕结果。

* * *

### **[创建我们的配置文件](#TOC)**

确保您已按照 [**获取您的 Microsoft 认知服务密钥**](#h3_obtaining) 获取您的 MCS API 订阅密钥。从那里，打开`microsoft_cognitive_services.py`文件并更新您的`SUBSCRPTION_KEY`:

```py
# define our Microsoft Cognitive Services subscription key
SUBSCRIPTION_KEY = "YOUR_SUBSCRIPTION_KEY"

# define the ACS endpoint
ENDPOINT_URL = "YOUR_ENDPOINT_URL"
```

您应该用从 [**获取您的 Microsoft 认知服务密钥**](#h3_obtaining) 获得的订阅密钥替换字符串`"YOUR_SUBSCRPTION_KEY"`。

此外，确保你再次检查你的`ENDPOINT_URL`。在撰写本文时，端点 URL 指向最新版本的 MCS API 然而，随着微软发布新的 API 版本，这个端点 URL 可能会改变，因此值得仔细检查。

* * *

### **[实现微软认知服务 OCR 脚本](#TOC)**

现在，让我们学习如何向 MCS API 提交用于文本检测和 OCR 的图像。

打开项目目录结构中的`microsoft_ocr.py`脚本，插入以下代码:

```py
# import the necessary packages
from config import microsoft_cognitive_services as config
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import argparse
import time
import sys
import cv2
```

注意第 2 行的**我们导入我们的`microsoft_cognitive_services`配置来提供我们的订阅密钥和端点 URL。然后，我们将使用`azure`和`msrest` Python 包向 API 发送请求。**

接下来，让我们定义`draw_ocr_results`，这是一个帮助函数，用于用 OCR 处理的文本注释我们的输出图像:

```py
def draw_ocr_results(image, text, pts, color=(0, 255, 0)):
	# unpack the points list
	topLeft = pts[0]
	topRight = pts[1]
	bottomRight = pts[2]
	bottomLeft = pts[3]

	# draw the bounding box of the detected text
	cv2.line(image, topLeft, topRight, color, 2)
	cv2.line(image, topRight, bottomRight, color, 2)
	cv2.line(image, bottomRight, bottomLeft, color, 2)
	cv2.line(image, bottomLeft, topLeft, color, 2)

	# draw the text itself
	cv2.putText(image, text, (topLeft[0], topLeft[1] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

	# return the output image
	return image
```

我们的`draw_ocr_results`函数有四个参数:

1.  `image`:我们将要绘制的输入图像。
2.  `text`:OCR 识别的文本。
3.  `pts`:左上*、*右上*、*右下*、*左下(x，y)*-文本 ROI 的坐标*
4.  *`color`:我们在`image`上绘制的 BGR 颜色*

 ***第 13-16 行**解包我们的边界框坐标。从那里，**第 19-22 行**画出图像中文本周围的边界框。然后我们在第 25 和 26 行的**上绘制 OCR 文本本身。**

我们通过将输出`image`返回给调用函数来结束这个函数。

我们现在可以解析我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we'll submit to Microsoft OCR")
args = vars(ap.parse_args())

# load the input image from disk, both in a byte array and OpenCV
# format
imageData = open(args["image"], "rb").read()
image = cv2.imread(args["image"])
```

这里我们只需要一个参数，`--image`，它是磁盘上输入图像的路径。我们从磁盘中读取这个图像，两者都是二进制字节数组(这样我们就可以将它提交给 MCS API，然后再以 OpenCV/NumPy 格式提交(这样我们就可以利用/注释它)。

现在让我们构造一个对 MCS API 的请求:

```py
# initialize the client with endpoint URL and subscription key
client = ComputerVisionClient(config.ENDPOINT_URL,
	CognitiveServicesCredentials(config.SUBSCRIPTION_KEY))

# call the API with the image and get the raw data, grab the operation
# location from the response, and grab the operation ID from the
# operation location
response = client.read_in_stream(imageData, raw=True)
operationLocation = response.headers["Operation-Location"]
operationID = operationLocation.split("/")[-1]
```

**第 43 和 44 行**初始化 Azure 计算机视觉客户端。请注意，我们在这里提供了我们的`ENDPOINT_URL`和`SUBSCRIPTION_KEY`——现在是返回`microsoft_cognitive_services.py`并确保您已经正确插入了您的订阅密钥的好时机(否则，对 MCS API 的请求将会失败)。

然后，我们将图像提交给第 49-51 行的 MCS API 进行 OCR。

我们现在必须等待并轮询来自 MCS API 的结果:

```py
# continue to poll the Cognitive Services API for a response until
# we get a response
while True:
	# get the result
	results = client.get_read_result(operationID)

	# check if the status is not "not started" or "running", if so,
	# stop the polling operation
	if results.status.lower() not in ["notstarted", "running"]:
		break

	# sleep for a bit before we make another request to the API
	time.sleep(10)

# check to see if the request succeeded
if results.status == OperationStatusCodes.succeeded:
	print("[INFO] Microsoft Cognitive Services API request succeeded...")

# if the request failed, show an error message and exit
else:
	print("[INFO] Microsoft Cognitive Services API request failed")
	print("[INFO] Attempting to gracefully exit")
	sys.exit(0)
```

老实说，对结果进行投票并不是我最喜欢的使用 API 的方式。它需要更多的代码，有点乏味，而且如果程序员不小心将`break`恰当地从循环中取出，它可能容易出错。

当然，这种方法也有好处，包括保持连接，提交更大块的数据，以及以*批*返回结果，而不是一次*。*

不管怎样，这就是微软实现 API 的方式，所以我们必须遵守他们的规则。

**第 55 行**开始一个`while`循环，连续检查来自 MCS API 的响应(**第 57 行**)。

如果我们没有在``["notstarted", "running"]`` 列表中找到状态，我们可以从循环中安全地`break`并处理我们的结果(**行 61 和 62** )。

如果不满足上述条件，我们睡眠一小段时间，然后再次轮询( **Line 65** )。

检查请求是否成功，如果成功，我们就可以安全地继续处理我们的结果。否则，如果请求没有成功，那么我们就没有 OCR 结果可以显示(因为图像不能被处理)，然后我们优雅地从脚本中退出(**第 72-75 行**)。

假设我们的 OCR 请求成功，现在让我们处理结果:

```py
# make a copy of the input image for final output
final = image.copy()

# loop over the results
for result in results.analyze_result.read_results:
	# loop over the lines
	for line in result.lines:
		# extract the OCR'd line from Microsoft's API and unpack the
		# bounding box coordinates
		text = line.text
		box = list(map(int, line.bounding_box))
		(tlX, tlY, trX, trY, brX, brY, blX, blY) = box
		pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))

		# draw the output OCR line-by-line
		output = image.copy()
		output = draw_ocr_results(output, text, pts)
		final = draw_ocr_results(final, text, pts)

		# show the output OCR'd line
		print(text)
		cv2.imshow("Output", output)

# show the final output image
cv2.imshow("Final Output", final)
cv2.waitKey(0)
```

**第 78 行**用所有绘制的文本初始化我们的`final`输出图像。

我们开始遍历第 81 行**的所有`lines`OCR 文本。**在**第 83 行**上，我们开始遍历 OCR 文本的所有行。我们提取 OCR 处理过的`text`和每个`line`的边界框坐标，然后分别构造一个列表，包含*左上角*、*右上角*、*右下角*和*左下角*(**第 86-89 行**)。

然后我们在`output`和`final`图像上逐行绘制 OCR 文本(**第 92-94 行**)。我们在屏幕和终端上显示当前文本行(**第 97 行和第 98 行** ) —最终的输出图像，上面绘制了所有经过 OCR 处理的文本，显示在**第 101 行和第 102 行。**

* * *

### **[微软认知服务 OCR 结果](#TOC)**

现在让我们让 MCS OCR API 为我们工作。打开终端并执行以下命令:

```py
$ python microsoft_ocr.py --image images/aircraft.png
[INFO] making request to Microsoft Cognitive Services API...
WARNING!
LOW FLYING AND DEPARTING AIRCRAFT
BLAST CAN CAUSE PHYSICAL INJURY
```

**图 2** 显示了将 MCS OCR API 应用于我们的飞机警告标志的输出。如果你还记得，这是我们在[之前的教程](https://pyimg.co/po6tf)中应用[亚马逊识别 API](https://aws.amazon.com/rekognition/) 时使用的同一张图片。因此，我在本教程中包含了相同的图像，以演示 MCS OCR API 可以正确地对该图像进行 OCR。

让我们尝试一个不同的图像，这个图像包含几个具有挑战性的文本:

```py
$ python microsoft_ocr.py --image images/challenging.png
[INFO] making request to Microsoft Cognitive Services API...

LITTER
EMERGENCY
First
Eastern National
Bus Times
STOP
```

**图 3** 显示了将 MCS OCR API 应用于我们的输入图像的结果——正如我们所见，MCS 在图像 OCR 方面做了一件*出色的*工作。

在左边的*上，我们有第一张东部国家公共汽车时刻表的样本图像(即公共汽车到达的时间表)。文件打印时采用了光面处理(可能是为了防止水渍)。尽管如此，由于光泽，图像仍然有明显的反射，特别是在*“公交时间”*文本中。尽管如此，MCS OCR API 可以正确地对图像进行 OCR。*

在*中间*，*，*文本高度像素化且质量低下，但这并不影响 MCS OCR API！它能正确识别图像。

最后，*右边的*显示了一个垃圾桶，上面写着*“垃圾”*文本很小，由于图像质量很低，不眯着眼睛看很难。也就是说，MCS OCR API 仍然可以对文本进行 OCR(尽管垃圾桶底部的文本难以辨认——无论是人还是 API 都无法读取该文本)。

下一个样本图像包含一个国家公园标志，如图**图 4** 所示:

MCS OCR API 可以逐行 OCR 每个符号(**图 4** )。我们还可以为每条线计算旋转的文本边界框/多边形。

```py
$ python microsoft_ocr.py --image images/park.png
[INFO] making request to Microsoft Cognitive Services API...

PLEASE TAKE
NOTHING BUT
PICTURES
LEAVE NOTHING
BUT FOOT PRINTS
```

最后一个例子包含交通标志:

```py
$ python microsoft_ocr.py --image images/street_signs.png
[INFO] making request to Microsoft Cognitive Services API...

Old Town Rd
STOP
ALL WAY
```

**图 5** 显示我们可以正确识别站牌和街道名称牌上的每一段文字。

* * *

## **[汇总](#TOC)**

在本教程中，您了解了 Microsoft 认知服务(MCS) OCR API。尽管比 Amazon Rekognition API 稍难实现和使用，但微软认知服务 OCR API 证明了它非常健壮，能够在许多情况下 OCR 文本，*包括*低质量图像。

当处理低质量图像时，MCS API *闪亮*。通常，我建议你有计划地检测并丢弃低质量的图像(正如我们在[之前的教程](https://pyimg.co/7emby)中所做的)。然而，如果你发现自己处于*不得不*处理低质量图像的情况下，使用微软 Azure 认知服务 OCR API 可能是值得的。

* * *

### **[引用信息](#TOC)**

****罗斯布鲁克，A.**** “微软认知服务的文本检测和 OCR”， *PyImageSearch* ，D. Chakraborty，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva，R. Raha 和 A. Thanki 编辑。，2022 年，[https://pyimg.co/0r4mt](https://pyimg.co/0r4mt)

```py
@incollection{Rosebrock_2022_OCR_MCS,
  author = {Adrian Rosebrock},
  title = {Text Detection and {OCR} with {M}icrosoft Cognitive Services},
  booktitle = {PyImageSearch},
  editor = {Devjyoti Chakraborty and Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/0r4mt},
}
```

* * *

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****