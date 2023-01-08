# 使用 Google Cloud Vision API 进行文本检测和 OCR

> 原文：<https://pyimagesearch.com/2022/03/31/text-detection-and-ocr-with-google-cloud-vision-api/>

* * *

* * *

## **[使用谷歌云视觉 API 进行文本检测和 OCR](#TOC)**

在本课中，您将:

*   了解如何从 Google Cloud 管理面板获取您的 Google Cloud Vision API keys/JSON 配置文件
*   配置您的开发环境，以便与 Google Cloud Vision API 一起使用
*   实现用于向 Google Cloud Vision API 发出请求的 Python 脚本

本课是关于**文本检测和 OCR** 的 3 部分系列的最后一部分:

1.  [*使用亚马逊 Rekognition API 进行文本检测和 OCR*](https://pyimg.co/po6tf)
2.  [*微软认知服务的文本检测和 OCR*](https://pyimg.co/0r4mt)
3.  ***使用谷歌云视觉 API 进行文本检测和 OCR*(本教程)**

**要了解谷歌云视觉 API 的文本检测和 OCR，** ***继续阅读。***

* * *

## **[使用谷歌云视觉 API 进行文本检测和 OCR](#TOC)**

在今天的课程中，我们将了解 Google Cloud Vision API。代码方面，Google Cloud Vision API 很好用。不过，它要求我们使用他们的管理面板来生成一个客户端 JavaScript 对象符号(JSON)文件，该文件包含访问 Vision API 所需的所有信息。

我们对 JSON 文件有着复杂的感情。一方面，不必硬编码我们的私有和公共密钥是件好事。但是另一方面，使用管理面板来生成 JSON 文件本身也很麻烦。

实际上，这是一种“一个六个，另一个半打”的情况这并没有太大的区别(只是一些需要注意的事情)。

正如我们将发现的，谷歌云视觉 API，就像其他的一样，往往非常准确，在复杂图像的 OCR 方面做得很好。

让我们开始吧！

* * *

### **[用于 OCR 的谷歌云视觉 API](#TOC)**

在本课的第一部分，您将了解 Google Cloud Vision API，以及如何获取 API 密钥和生成 JSON 配置文件，以便使用 API 进行身份验证。

从那里，我们将确保您的开发环境正确配置了所需的 Python 包，以便与 Google Cloud Vision API 接口。

然后，我们将实现一个 Python 脚本，该脚本获取一个输入图像，将其打包到一个 API 请求中，并将其发送到用于 OCR 的 Google Cloud Vision API。

我们将通过讨论我们的结果来结束本课。

* * *

### **[获取您的谷歌云视觉 API 密钥](#TOC)**

* * *

#### **[先决条件](#TOC)**

使用 Google Cloud Vision API 只需要一个启用了计费的 Google Cloud 帐户。你可以在这里找到关于如何修改你的账单设置[的谷歌云指南。](http://pyimg.co/y0a0d)

* * *

#### **[启用 Google Cloud Vision API 并下载凭证的步骤](#TOC)**

你可以在我们的书里找到获取密钥的指南， [**OCR with OpenCV，Tesseract，和 Python**](https://pyimagesearch.com/ocr-with-opencv-tesseract-and-python/) 。

* * *

### **[为 Google Cloud Vision API 配置您的开发环境](#TOC)**

为了遵循这个指南，您需要在您的系统上安装 OpenCV 库和`google-cloud-vision` Python 包。

幸运的是，两者都是 pip 可安装的:

```py
$ pip install opencv-contrib-python
$ pip install --upgrade google-cloud-vision
```

如果您正在使用 Python 虚拟环境或 Anaconda 包管理器，请确保在运行上面的`pip` -install 命令之前，使用适当的命令访问您的 Python 环境*。否则，`google-cloud-vision`包将被安装在您的系统 Python 中，而不是您的 Python 环境中。*

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

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

让我们检查一下 Google Cloud Vision API OCR 项目的项目目录结构:

```py
|-- images
|   |-- aircraft.png
|   |-- challenging.png
|   |-- street_signs.png
|-- client_id.json
|-- google_ocr.py
```

我们将把我们的`google_ocr.py`脚本应用于`images`目录中的几个例子。

`client_id.json`文件提供了所有必要的凭证和认证信息。`google_ocr.py`脚本将加载这个文件，并将其提供给 Google Cloud Vision API 来执行 OCR。

* * *

### **[实现谷歌云视觉 API 脚本](#TOC)**

回顾了我们的项目目录结构后，我们可以继续实现`google_ocr.py`，Python 脚本负责:

1.  加载我们的`client_id.json`文件的内容
2.  连接到 Google Cloud Vision API
3.  将输入图像加载并提交给 API
4.  检索文本检测和 OCR 结果
5.  在屏幕上绘制和显示 OCR 文本

让我们开始吧:

```py
# import the necessary packages
from google.oauth2 import service_account
from google.cloud import vision
import argparse
import cv2
import io
```

**第 2-6 行**导入我们需要的 Python 包。注意，我们需要`service_account`来连接到 Google Cloud Vision API，而`vision`包包含负责 OCR 的`text_detection`函数。

接下来，我们有`draw_ocr_results`，一个用来注释输出图像的便利函数:

```py
def draw_ocr_results(image, text, rect, color=(0, 255, 0)):
	# unpacking the bounding box rectangle and draw a bounding box
	# surrounding the text along with the OCR'd text itself
	(startX, startY, endX, endY) = rect
	cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	cv2.putText(image, text, (startX, startY - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

	# return the output image
	return image
```

`draw_ocr_results`函数接受四个参数:

1.  `image`:我们正在绘制的输入图像
2.  `text`:OCR 识别的文本
3.  `rect`:文本 ROI 的边界框坐标
4.  `color`:绘制的边框和文字的颜色

**第 11 行**解包 *(x，y)*-我们文本 ROI 的坐标。我们使用这些坐标绘制一个包围文本的边界框，以及 OCR 文本本身(**第 12-14 行**)。

然后我们将`image`返回给调用函数。

让我们检查一下我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image that we'll submit to Google Vision API")
ap.add_argument("-c", "--client", required=True,
	help="path to input client ID JSON configuration file")
args = vars(ap.parse_args())
```

这里有两个命令行参数:

*   `--image`:输入图像的路径，我们将把它提交给 Google Cloud Vision API 进行 OCR。
*   `--client`:包含我们认证信息的客户端 ID JSON 文件(确保按照 **[获取您的 Google Cloud Vision API 密钥](#h3_Keys)** 部分生成该 JSON 文件)。

是时候连接到谷歌云视觉 API 了:

```py
# create the client interface to access the Google Cloud Vision API
credentials = service_account.Credentials.from_service_account_file(
	filename=args["client"],
	scopes=["https://www.googleapis.com/auth/cloud-platform"])
client = vision.ImageAnnotatorClient(credentials=credentials)

# load the input image as a raw binary file (this file will be
# submitted to the Google Cloud Vision API)
with io.open(args["image"], "rb") as f:
	byteImage = f.read()
```

**第 28-30 行**连接到 Google Cloud Vision API，提供磁盘上 JSON 认证文件的`--client`路径。**第 31 行**然后为所有图像处理/计算机视觉操作创建我们的`client`。

然后，我们将我们的输入`--image`作为一个字节数组(`byteImage`)从磁盘加载，提交给 Google Cloud Vision API。

现在让我们将`byteImage`提交给 API:

```py
# create an image object from the binary file and then make a request
# to the Google Cloud Vision API to OCR the input image
print("[INFO] making request to Google Cloud Vision API...")
image = vision.Image(content=byteImage)
response = client.text_detection(image=image)

# check to see if there was an error when making a request to the API
if response.error.message:
	raise Exception(
		"{}\nFor more info on errors, check:\n"
		"https://cloud.google.com/apis/design/errors".format(
			response.error.message))
```

**第 41 行**创建一个`Image`数据对象，然后提交给 Google Cloud Vision API 的`text_detection`函数(**第 42 行**)。

**第 45-49 行**检查我们的输入图像是否有错误，如果有，我们`raise`出错并退出脚本。

否则，我们可以处理 OCR 步骤的结果:

```py
# read the image again, this time in OpenCV format and make a copy of
# the input image for final output
image = cv2.imread(args["image"])
final = image.copy()

# loop over the Google Cloud Vision API OCR results
for text in response.text_annotations[1::]:
	# grab the OCR'd text and extract the bounding box coordinates of
	# the text region
	ocr = text.description
	startX = text.bounding_poly.vertices[0].x
	startY = text.bounding_poly.vertices[0].y
	endX = text.bounding_poly.vertices[1].x
	endY = text.bounding_poly.vertices[2].y

	# construct a bounding box rectangle from the box coordinates
	rect = (startX, startY, endX, endY)
```

**Line 53** 以 OpenCV/NumPy 数组格式从磁盘加载我们的输入图像(以便我们可以在上面绘图)。

**第 57 行**遍历来自 Google Cloud Vision API 响应的所有 OCR 结果`text`。**第 60 行**提取`ocr`文本本身，而**第 61-64 行**提取文本区域的边界框坐标。第 67 行然后从这些坐标构建一个矩形(`rect`)。

最后一步是在`output`和`final`图像上绘制 OCR 结果:

```py
	# draw the output OCR line-by-line
	output = image.copy()
	output = draw_ocr_results(output, ocr, rect)
	final = draw_ocr_results(final, ocr, rect)

	# show the output OCR'd line
	print(ocr)
	cv2.imshow("Output", output)
	cv2.waitKey(0)

# show the final output image
cv2.imshow("Final Output", final)
cv2.waitKey(0)
```

每一段 OCR 文本都显示在我们屏幕上的第 75-77 行**。带有*所有* OCR 文本的`final`图像显示在**行 80 和 81** 上。**

 *** * *

### **[谷歌云视觉 API OCR 结果](#TOC)**

现在让我们将 Google Cloud Vision API 投入使用吧！打开终端并执行以下命令:

```py
$ python google_ocr.py --image images/aircraft.png --client client_id.json
[INFO] making request to Google Cloud Vision API...
WARNING!
LOW
FLYING
AND
DEPARTING
AIRCRAFT
BLAST
CAN
CAUSE
PHYSICAL
INJURY
```

**图 2** 显示了将 Google Cloud Vision API 应用于我们的飞机图像的结果，我们一直在对所有三种云服务的 OCR 性能进行基准测试。与亚马逊 Rekognition API 和微软认知服务一样，谷歌云视觉 API 可以正确地对图像进行 OCR。

让我们尝试一个更具挑战性的图像，您可以在**图 3** 中看到:

```py
$ python google_ocr.py --image images/challenging.png --client client_id.json
[INFO] making request to Google Cloud Vision API...
LITTER
First
Eastern
National
Bus
Fimes
EMERGENCY
STOP
```

就像微软的认知服务 API 一样，谷歌云视觉 API 在我们具有挑战性的低质量图像上表现良好，这些图像具有像素化和低可读性(即使以人类的标准来看，更不用说机器了)。结果在**图 3** 中。

有趣的是，Google Cloud Vision API *确实*出错了，以为*【时代】*中的*【T】*是一个*【f】*

让我们看最后一张图片，这是一个街道标志:

```py
$ python google_ocr.py --image images/street_signs.png --client client_id.json
[INFO] making request to Google Cloud Vision API...
Old
Town
Rd
STOP
ALL
WAY
```

**图 4** 显示了将 Google Cloud Vision API 应用于我们的街道标志图像的输出。微软认知服务 API 对图像进行逐行 OCR，导致文本*“老城路”*和*“一路”*被 OCR 成单行。或者，Google Cloud Vision API 对文本进行逐字 ocr(Google Cloud Vision API 中的默认设置)。

* * *

* * *

## **[汇总](#TOC)**

在本课中，您学习了如何将基于云的 Google Cloud Vision API 用于 OCR。像我们在本书中讨论的其他基于云的 OCR APIs 一样，Google Cloud Vision API 可以不费吹灰之力获得高 OCR 准确度。当然，缺点是你需要一个互联网连接来利用这个 API。

当选择基于云的 API 时，我不会关注与 API 接口所需的 Python 代码量。**相反，考虑你正在使用的云平台的** ***整体生态系统*** **。**

假设您正在构建一个应用程序，要求您与 Amazon 简单存储服务(Amazon S3)进行数据存储。在这种情况下，使用 Amazon Rekognition API 更有意义。这使得你可以把所有东西都放在亚马逊的保护伞下。

另一方面，如果你正在使用谷歌云平台(GCP)实例来训练云中的深度学习模型，那么使用谷歌云视觉 API 更有意义。

这些都是构建应用程序时的设计和架构决策。假设您只是在“测试”这些 API 中的每一个。你不受这些考虑的约束。然而，如果你正在开发一个生产级的应用程序，那么*花时间*考虑每个云服务的权衡是非常值得的。你应该考虑*更多的*而不仅仅是 OCR 的准确性；考虑计算、存储等。，每个云平台提供的服务。

* * *

### **[引用信息](#TOC)**

**Rosebrock，A.** “使用谷歌云视觉 API 的文本检测和 OCR”， *PyImageSearch* ，D. Chakraborty，P. Chugh。A. R. Gosthipaty、S. Huot、K. Kidriavsteva、R. Raha 和 A. Thanki 编辑。，2022 年，【https://pyimg.co/evzxr 

```py
***@incollection{Rosebrock_2022_OCR_GCV,
  author = {Adrian Rosebrock},
  title = {Text Detection and {OCR} with {G}oogle Cloud Vision {API}},
  booktitle = {PyImageSearch},
  editor = {Devjyoti Chakraborty and Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},  year = {2022},
  note = {https://pyimg.co/evzxr},
}
```

* * *

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******