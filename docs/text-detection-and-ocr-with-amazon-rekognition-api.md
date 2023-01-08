# 使用 amazon 识别 api 进行文本检测和 ocr

> 原文：<https://pyimagesearch.com/2022/03/21/text-detection-and-ocr-with-amazon-rekognition-api/>

* * *

* * *

## **[使用亚马逊 Rekognition API 进行文本检测和 OCR](#TOC)**

在本教程中，您将:

*   了解 Amazon Rekognition API
*   了解如何将 amazon 识别 api 用于 ocr
*   获取您的亚马逊网络服务(AWS)认证密钥
*   安装 Amazon 的`boto3`包来与 OCR API 接口
*   实现一个与 Amazon Rekognition API 接口的 Python 脚本来 OCR 图像

本课是关于 ****文本检测和 OCR**** 的 3 部分系列的第 1 部分:

1.  ***使用亚马逊 Rekognition API 进行文本检测和 OCR*(今日教程)**
2.  *微软认知服务的文本检测和 OCR*
3.  *使用谷歌云视觉 API 进行文本检测和 OCR*

**要了解文本检测和 OCR，** ***继续阅读。***

* * *

## **[使用亚马逊 Rekognition API 进行文本检测和 OCR](#TOC)**

到目前为止，我们主要关注于使用 Tesseract OCR 引擎。然而，其他光学字符识别(OCR)引擎也是可用的，其中一些比 Tesseract 更精确，甚至在复杂、不受约束的条件下也能准确地识别文本。

通常，这些 OCR 引擎位于云中。许多是基于专利的。为了保持这些模型和相关数据集的私有性，公司并不自己分发模型，而是将它们放在 REST API 后面。

虽然这些模型确实比 Tesseract 更精确，但也有一些缺点，包括:

*   OCR 图像需要互联网连接，这对于大多数笔记本电脑/台式机来说不是问题，但如果您在边缘工作，互联网连接可能是不可能的
*   此外，如果您正在使用边缘设备，那么您可能不希望在网络连接上消耗功率
*   网络连接会带来延迟
*   OCR 结果需要更长时间，因为图像需要打包到 API 请求中并上传到 OCR API。API 将需要咀嚼图像并对其进行 OCR，然后最终将结果返回给客户端
*   由于 OCR 处理每幅图像的延迟和时间，这些 OCR APIs 能否实时运行是个疑问
*   它们是要花钱的(但是通常提供免费试用，或者每月有一定数量的 API 请求是免费的)

看着前面的列表，你可能想知道我们到底为什么要覆盖这些 APIs 好处是什么？

正如您将看到的，这里的主要好处是*准确性。*首先，考虑一下谷歌和微软运行各自的搜索引擎所获得的数据量。然后，考虑一下亚马逊每天从打印运输标签中产生的数据量*。*

 ***这些公司拥有令人难以置信的*****数量的图像数据——当他们根据自己的数据训练他们新颖、先进的 OCR 模型时，结果是一个** ***令人难以置信的*** **健壮而准确的 OCR 模型。***

 *在本教程中，您将学习如何使用 Amazon Rekognition API 来 OCR 图像。在即将到来的教程中，我们将涵盖微软 Azure 认知服务和谷歌云视觉 API。

* * *

### **[配置您的开发环境](#TOC)**

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install boto3
```

**如果您需要帮助配置 OpenCV 的开发环境，我们*强烈推荐*阅读我们的** [***pip 安装 OpenCV* 指南**](https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让您启动并运行。

* * *

### **[亚马逊用于 OCR 的识别 API](#TOC)**

本教程的第一部分将着重于获取您的 AWS 重新识别密钥。这些密钥将包括一个公共访问密钥和一个秘密密钥，类似于 SSH、SFTP 等。

然后，我们将向您展示如何安装用于 Python 的亚马逊 Web 服务(AWS)软件开发工具包(SDK)`boto3`。最后，我们将使用`boto3`包与 Amazon Rekognition OCR API 接口。

接下来，我们将实现我们的 Python 配置文件(它将存储我们的访问密钥、秘密密钥和 AWS 区域),然后创建我们的驱动程序脚本，用于:

1.  从磁盘加载输入图像
2.  将其打包成一个 API 请求
3.  将 api 请求发送到 aws 识别以进行 ocr
4.  从 API 调用中检索结果
5.  显示我们的 OCR 结果

我们将讨论我们的结果来结束本教程。

* * *

### **[获取您的 AWS 识别密钥](#TOC)**

您将需要从我们的伙伴网站获取更多信息，以获得有关获取您的 AWS 重新识别密钥的说明。你可以在这里找到说明[。](http://pyimg.co/vxd51)

* * *

### **[安装亚马逊的 Python 包](#TOC)**

为了与 Amazon Rekognition API 接口，我们需要使用`boto3`包:AWS SDK。幸运的是，`boto3`安装起来非常简单，只需要一个`pip` -install 命令:

```py
$ pip install boto3
```

如果您正在使用 Python 虚拟环境或 Anaconda 环境，请确保在运行上述命令之前，使用适当的命令访问您的 Python 环境*(否则，`boto3`将被安装在 Python 的系统安装中)。*

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

在使用 Amazon Rekognition API 执行文本检测和 OCR 之前，我们首先需要回顾一下我们的项目目录结构。

```py
|-- config
|   |-- __init__.py
|   |-- aws_config.py
|-- images
|   |-- aircraft.png
|   |-- challenging.png
|   |-- park.png
|   |-- street_signs.png
|-- amazon_ocr.py
```

`aws_config.py`文件包含我们的 AWS 访问密钥、秘密密钥和区域。在 **[获取 AWS 识别密钥](#AWS_Keys)** 一节中，您学习了如何获取这些值。

我们的`amazon_ocr.py`脚本将获取这个`aws_config`，连接到 Amazon Rekognition API，然后对我们的`images`目录中的每张图像执行文本检测和 OCR。

* * *

### **[创建我们的配置文件](#TOC)**

为了连接到 Amazon Rekognition API，我们首先需要提供我们的访问密钥、秘密密钥和区域。如果您尚未获得这些密钥，请转到 **[获取您的 AWS 重新识别密钥](#AWS_Keys)** 部分，并确保按照步骤操作并记下值。

之后，您可以回到这里，打开`aws_config.py`，并更新代码:

```py
# define our AWS Access Key, Secret Key, and Region
ACCESS_KEY = "YOUR_ACCESS_KEY"
SECRET_KEY = "YOUR_SECRET_KEY"
REGION = "YOUR_AWS_REGION"
```

共享我的 API 键会违反安全性，所以我在这里留了占位符值。确保用您的 API 密钥更新它们；否则，您将无法连接到 Amazon Rekognition API。

* * *

### **[实现亚马逊 Rekognition OCR 脚本](#TOC)**

实现了我们的`aws_config`之后，让我们转到`amazon_ocr.py`脚本，它负责:

1.  连接到亚马逊识别 api
2.  从磁盘载入输入图像
3.  在 API 请求中打包图像
4.  将包发送到 amazon recognition api 进行 ocr
5.  从 Amazon Rekognition API 获取 OCR 结果
6.  显示我们的输出文本检测和 OCR 结果

让我们开始实施:

```py
# import the necessary packages
from config import aws_config as config
import argparse
import boto3
import cv2
```

**第 2-5 行**导入我们需要的 Python 包。值得注意的是，我们需要我们的`aws_config`，以及`boto3`，它是亚马逊的 Python 包，来与他们的 API 接口。

现在让我们定义`draw_ocr_results`，这是一个简单的 Python 实用程序，用于从 Amazon Rekognition API 中提取输出 OCR 结果:

```py
def draw_ocr_results(image, text, poly, color=(0, 255, 0)):
   # unpack the bounding box, taking care to scale the coordinates
   # relative to the input image size
   (h, w) = image.shape[:2]
   tlX = int(poly[0]["X"] * w)
   tlY = int(poly[0]["Y"] * h)
   trX = int(poly[1]["X"] * w)
   trY = int(poly[1]["Y"] * h)
   brX = int(poly[2]["X"] * w)
   brY = int(poly[2]["Y"] * h)
   blX = int(poly[3]["X"] * w)
   blY = int(poly[3]["Y"] * h)
```

`draw_ocr_results`函数接受四个参数:

1.  我们正在其上绘制 OCR 文本的输入图像
2.  `text`:OCR 识别的文本本身
3.  `poly`:Amazon Rekognition API 返回的文本包围盒的多边形对象/坐标
4.  `color`:边框的颜色

**第 10 行**获取我们正在绘制的`image`的宽度和高度。**第 11-18 行**然后抓取文本 ROI 的边界框坐标，注意按宽度和高度缩放坐标。

我们为什么要执行这个缩放过程？

嗯，正如我们将在本教程后面发现的，Amazon Rekognition API 返回范围`[0, 1]` *中的边界框。*将边界框乘以原始图像的宽度和高度会使边界框回到原始图像的比例。

从那里，我们现在可以注释`image`:

```py
   # build a list of points and use it to construct each vertex
   # of the bounding box
   pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))
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

**第 22-26 行**建立对应于边界框每个顶点的点列表。给定顶点，**第 29-32 行**画出旋转文本的边界框。**第 35 行和第 36 行**绘制 OCR 文本本身。

然后，我们将带注释的输出`image`返回给调用函数。

定义了 helper 实用程序后，让我们继续讨论命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
   help="path to input image that we'll submit to AWS Rekognition")
ap.add_argument("-t", "--type", type=str, default="line",
   choices=["line", "word"],
   help="output text type (either 'line' or 'word')")
args = vars(ap.parse_args())
```

`--image`命令行参数对应于我们想要提交给 Amazon Rekognition OCR API 的输入图像的路径。

`--type`参数可以是`line`或`word`，表示我们是否希望 Amazon Rekognition API 返回分组为*行*或单个*单词的 OCR 结果。*

接下来，让我们连接到亚马逊网络服务:

```py
# connect to AWS so we can use the Amazon Rekognition OCR API
client = boto3.client(
   "rekognition",
   aws_access_key_id=config.ACCESS_KEY,
   aws_secret_access_key=config.SECRET_KEY,
   region_name=config.REGION)

# load the input image as a raw binary file and make a request to
# the Amazon Rekognition OCR API
print("[INFO] making request to AWS Rekognition API...")
image = open(args["image"], "rb").read()
response = client.detect_text(Image={"Bytes": image})

# grab the text detection results from the API and load the input
# image again, this time in OpenCV format
detections = response["TextDetections"]
image = cv2.imread(args["image"])

# make a copy of the input image for final output
final = image.copy()
```

**51-55 号线**连接到 AWS。在这里，我们提供访问密钥、秘密密钥和区域。

一旦连接好，我们就从磁盘中加载输入图像作为二进制对象(**第 60 行**)，然后通过调用`detect_text`函数并提供我们的`image`将它提交给 AWS。

调用`detect_text`会导致来自 Amazon Rekognition API 的`response`。我们接着抓取`TextDetections`结果(**第 65 行**)。

**第 66 行**以 OpenCV 格式从磁盘加载输入`--image`，而**第 69 行**克隆图像以在其上绘制。

我们现在可以在 Amazon Rekognition API 的文本检测边界框上循环:

```py
# loop over the text detection bounding boxes
for detection in detections:
   # extract the OCR'd text, text type, and bounding box coordinates
   text = detection["DetectedText"]
   textType = detection["Type"]
   poly = detection["Geometry"]["Polygon"]

   # only draw show the output of the OCR process if we are looking
   # at the correct text type
   if args["type"] == textType.lower():
      # draw the output OCR line-by-line
      output = image.copy()
      output = draw_ocr_results(output, text, poly)
      final = draw_ocr_results(final, text, poly)

      # show the output OCR'd line
      print(text)
      cv2.imshow("Output", output)
      cv2.waitKey(0)

# show the final output image
cv2.imshow("Final Output", final)
cv2.waitKey(0)
```

**第 72 行**遍历亚马逊的 OCR API 返回的所有`detections`。然后，我们提取经过 OCR 处理的`text`、`textType`(或者是 *"word"* 或者是 *"line"* )，以及经过 OCR 处理的文本的边界框坐标(**第 74 行和第 75 行**)。

**第 80 行**进行检查，以验证我们正在查看的是`word`还是`line` OCR 文本。如果当前的`textType`匹配我们的`--type`命令行参数，我们在`output`映像和我们的`final`克隆映像上调用我们的`draw_ocr_results`函数(**第 82-84 行**)。

**第 87-89 行**在我们的终端和屏幕上显示当前 OCR 识别的行或单词。这样，我们可以很容易地看到每一行或每一个单词，而不会使输出图像变得过于混乱。

最后，**行 92 和 93** 显示了在我们的屏幕上一次绘制*所有*文本的结果(为了可视化的目的)。

* * *

### **[亚马逊 Rekognition OCR 结果](#TOC)**

恭喜你实现了一个 Python 脚本来与 Amazon Rekognition 的 OCR API 接口！

让我们看看实际效果，首先逐行对整个图像进行光学字符识别:

```py
$ python amazon_ocr.py --image images/aircraft.png
[INFO] making request to AWS Rekognition API...
WARNING!
LOW FLYING AND DEPARTING AIRCRAFT
BLAST CAN CAUSE PHYSICAL INJURY
```

**图 2** 显示我们已经成功地逐行对输入的`aircraft.png`图像进行了 OCR，从而证明 Amazon Rekognition API 能够:

1.  定位输入图像中的每个文本块
2.  OCR 每个文本 ROI
3.  将文本块分组成行

但是如果我们想在*单词*级别而不是*行*级别获得我们的 OCR 结果呢？

这就像提供`--type`命令行参数一样简单:

```py
$ python amazon_ocr.py --image images/aircraft.png --type word
[INFO] making request to AWS Rekognition API...

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

正如我们的输出和**图 3** 所示，我们现在在*单词*级别上对文本进行 OCR。

我是 Amazon Rekognition 的 OCR API 的忠实粉丝。而 AWS、EC2 等。确实有一点学习曲线，好处是一旦你知道了，你就理解了亚马逊的整个网络服务生态系统，这使得开始集成不同的服务变得更加容易。

我也很欣赏亚马逊 Rekognition API 的入门超级简单。其他服务(例如 Google Cloud Vision API)让“第一次轻松获胜”变得更加困难。如果这是您第一次尝试使用云服务 API，那么在使用微软认知服务或谷歌云视觉 API 之前，一定要考虑先使用 Amazon Rekognition API。

* * *

* * *

## **[汇总](#TOC)**

在本教程中，您学习了如何创建 Amazon Rekognition 密钥，安装`boto3`，这是一个用于与 AWS 接口的 Python 包，以及实现一个 Python 脚本来调用 Amazon Rekognition API。

Python 脚本很简单，只需要不到 100 行代码就可以实现(包括注释)。

我们的 Amazon Rekognition OCR API 结果不仅是正确的，而且我们还可以在*行*和*字*级别解析结果，比 EAST text 检测模型和 Tesseract OCR 引擎提供的粒度更细(至少不需要微调几个选项)。

在下一课中，我们将了解用于 OCR 的 Microsoft 认知服务 API。

* * *

### **[引用信息](#TOC)**

****Rosebrock，A.**** “使用亚马逊 Rekognition API 的文本检测和 OCR”， *PyImageSearch* ，D. Chakraborty，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva，R. Raha，A. Thanki 编辑。，2022 年，[https://pyimg.co/po6tf](https://pyimg.co/po6tf)

```py
@incollection{Rosebrock_2022_OCR_Amazon_Rekognition_API,
  author = {Adrian Rosebrock},
  title = {Text Detection and OCR with Amazon Rekognition API},
  booktitle = {PyImageSearch},
  editor = {Devjyoti Chakraborty and Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha and Abhishek Thanki},
  year = {2022},
  note = {https://pyimg.co/po6tf},
}
```

* * *

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******