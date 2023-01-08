# 计算机视觉和深度学习的伦理应用——通过自动年龄和军事疲劳检测识别儿童兵

> 原文：<https://pyimagesearch.com/2020/05/11/an-ethical-application-of-computer-vision-and-deep-learning-identifying-child-soldiers-through-automatic-age-and-military-fatigue-detection/>

在本教程中，我们将学习如何应用计算机视觉、深度学习和 OpenCV，通过自动年龄检测和军事疲劳识别来识别潜在的儿童兵。

服兵役对我个人来说很重要，我认为这是值得尊敬和钦佩的。这正是为什么这个项目，利用技术来识别儿童兵，是我强烈感受到的东西——任何人都不应该被强迫服役，尤其是年幼的儿童。

你知道，在我的成长过程中，军队一直是我家庭的一个重要组成部分，尽管我没有亲自服役。

*   我的曾祖父在一战期间是一名步兵
*   我的祖父在军队里当了 25 年的厨师/面包师
*   我的父亲在美国军队服役了八年，在越南战争结束后研究传染病
*   我的堂兄高中一毕业就加入了海军陆战队，在光荣退伍前在阿富汗服役了两次

即使在我的直系亲属之外，军队仍然是我生活和社区的一部分。我在马里兰州的一个农村地区上的高中。高中毕业后没有多少机会，只有三条真正的道路:

1.  成为一名农民——许多人都这样做了，在各自的家庭农场工作，直到他们最终继承了它
2.  试着在大学里坚持下去——相当一部分人选择了这条路，但在这个过程中，他们或他们的家庭都背负了巨额债务
3.  **参军**——获得报酬，学习可以转移到现实工作中的实用技能，通过《退伍军人法案》支付高达 5 万美元的大学费用(也可以部分转移到你的配偶或孩子身上)，如果被部署，还可以获得额外的福利(当然，要冒着失去生命的风险)

如果你不想成为农民或从事农业，那就只剩下两个选择——上大学或参军。对一些人来说，大学没有任何意义。它不值得花费。

如果我没记错的话，在我高中毕业之前，我们班至少有 10 个孩子参了军，其中一些我认识，也和他们一起上过课。

像我这样长大的人，我对军人、*尤其是*那些为国效力的人、*无论他们来自哪个国家，都怀有无比的敬意。**你是否曾在美国、芬兰、日本等国服役。为你的国家服务是一件大事，我尊重所有这样做的人。***

我这辈子没有多少遗憾，但是真的，回头看看，没有服役就是其中之一。我真希望我在军队里呆了四年，每次回想起来，我仍然会感到后悔和内疚。

也就是说，选择不服役是我的选择。

我们中的大多数人都可以选择我们的服务——尽管当然有复杂的社会问题，如贫困或道德问题，超出了这篇博文的范围。然而，底线是小孩子永远不应该被强迫参军。

世界上有些地方，孩子们没有选择的权利。由于极端贫困、恐怖主义、政府宣传和/或操纵，儿童被迫参战。

这场战斗也不总是使用武器。在战争中，儿童可以被用作间谍/告密者、信使、人盾，甚至作为谈判的筹码。

无论是发射武器还是在大型游戏中充当棋子，儿童兵都会产生持久的健康影响，包括但不限于( [source](https://en.wikipedia.org/wiki/Children_in_the_military) ):

*   精神疾病(慢性压力、焦虑、PTSD 等。)
*   读写能力差
*   更高的贫困风险
*   成年人失业
*   酒精和药物滥用
*   自杀风险更高

儿童服兵役也不是一个新现象。这是一个古老的故事:

*   1212 年的儿童十字军因招募儿童而臭名昭著。一些人死了，但还有许多人被卖为奴隶
*   拿破仑征募儿童入伍
*   在第一次和第二次世界大战期间，儿童被利用
*   1973 年出版的非小说类书籍和 2001 年的电影《兵临城下》讲述了斯大林格勒战役的故事，更具体地说，是虚构的瓦西里·扎依采夫，一位著名的苏联狙击手。在那本书/电影中，还是个孩子的萨沙·菲利波夫被用作间谍和线人。萨沙通常会和纳粹交朋友，然后把信息反馈给苏联。萨沙后来被纳粹抓住并杀害
*   在现代，我们都太熟悉恐怖组织，如基地组织和 ISIS 招募弱势儿童加入他们的努力

虽然我们许多人都同意在战争中使用儿童是不可接受的，但儿童最终仍然参与战争的事实是一个更复杂的问题。当你的生命危在旦夕，当你的家人饥肠辘辘，当你周围的人奄奄一息，这就成了生死攸关的事情。

战斗或死亡。

这是一个令人悲伤的现实，但这是我们可以通过适当的教育来改善(并理想地解决)的事情，并慢慢地，逐渐地，让世界变得更安全，更美好。

与此同时，我们可以使用一点计算机视觉和深度学习来帮助识别潜在的儿童兵，无论是在战场上还是在他们接受教育/灌输的不太好的国家或组织中。

在我被介绍给来自 GDI.Foundation 的 Victor Gevers(一位受人尊敬的道德黑客)之后，今天的帖子是我过去几个月工作的高潮。

维克多和他的团队发现了教室面部识别软件的漏洞，该软件用于验证儿童是否出席。**在检查这些照片时，似乎这些孩子中的一些正在接受军事教育和训练**(也就是说，孩子们穿着军装和其他证据，这让我感到不舒服)。

我不打算讨论所涉及的政治、国家或组织的细节，这不是我的位置，这完全取决于 Victor 和他的团队如何处理这种特殊情况。

相反，我是来报告科学和算法的。

计算机视觉和深度学习领域理所当然地受到了一些应得的批评，因为它们允许强大的政府和组织创建“老大哥”般的警察国家，在那里一只警惕的眼睛总是存在。

也就是说，CV/DL 可以用来“观察观察者”总会有*和*组织和国家试图监视我们。我们可以反过来使用 CV/DL 作为一种问责形式，让他们对自己的行为负责。是的，如果应用得当，它可以用来拯救生命。

**要了解计算机视觉和深度学习的道德应用，特别是通过自动年龄和军事疲劳检测识别儿童兵，*继续阅读！***

## **计算机视觉和深度学习的道德应用——通过自动年龄和军事疲劳检测识别儿童兵**

在本教程的第一部分，我们将讨论我是如何参与这个项目的。

从这里，我们将看看在图像和视频流中识别潜在儿童兵的四个步骤。

一旦我们理解了我们的基本算法，我们将使用 Python、OpenCV 和 Keras/TensorFlow 实现我们的儿童士兵检测方法。

我们将通过检查我们的工作结果来结束本教程。

### **军队中的儿童、儿童兵和人权——我对这一事业的尝试**

我第一次参与这个项目是在一月中旬，当时我和来自 GDI.Foundation 的 Victor Gevers 联系

维克多和他的团队发现了用于课堂面部识别(即基于面部识别自动考勤的“智能考勤系统”)的软件中的数据泄露。

这次数据泄露暴露了数百万儿童的记录，包括身份证号码，GPS 定位，是的，*甚至是他们的面部照片。*

任何类型的数据泄露都令人担忧，但至少可以说，暴露儿童的泄露是非常严重的。

不幸的是，事情变得更糟了。

在检查泄漏的照片时，发现一群穿着军装的儿童。

这立刻引起了一些人的惊讶。

Victor 和我联系并简短地交换了电子邮件，讨论如何利用我的知识和专业技能来提供帮助。

Victor 和他的团队需要一种方法来自动检测人脸，确定他们的年龄，并确定此人是否穿着军装。

我同意了，条件是我可以在伦理上分享科学成果(而不是政治、国家或相关组织),作为一种教育形式，帮助其他人从现实世界的问题中学习。

有一些组织、联盟和个人比我更适合处理人道主义方面的问题——虽然我是 CV/DL 方面的专家，但我不是政治或人道主义方面的专家(尽管我尽最大努力教育自己并尽可能做出最好的决定)。

我希望你把这篇文章当作一种教育。这绝不是公开所涉及的国家或组织，我已经确保在本教程中不提供任何原始训练数据或示例图像。所有原始数据已被删除或适当匿名化。

### **我们如何利用计算机视觉和深度学习来检测和识别潜在的儿童兵？**

发现和识别潜在的儿童兵是一个分四步走的过程:

1.  **步骤# 1–人脸检测:**应用[人脸检测](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)在输入图像/视频流中定位人脸
2.  **步骤# 2——年龄检测:**利用[基于深度学习的年龄检测器](https://pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/)确定通过*步骤#1* 检测到的人的年龄
3.  **步骤# 3——军事疲劳检测:**将深度学习应用于[自动检测军装的伪装或其他迹象](https://pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/)
4.  **步骤# 4–组合结果:**从*步骤#2* 和*步骤#3* 中获取结果，以确定儿童是否可能穿着军装，根据原始图像的来源和上下文，哪个*可能*是儿童兵的指示

如果你已经注意到，在过去的 1-1.5 个月里，我已经*有目的地*在 PyImageSearch 博客上报道了这些话题，为这篇博客做准备。

我们将快速回顾下面四个步骤中的每一个，但是我建议您使用上面的链接来获得每个步骤的更多细节。

### **步骤#1:检测图像或视频流中的人脸**

在我们能够确定一个孩子是否在图像或视频流中之前，我们首先需要检测人脸。

**人脸检测是自动定位*人脸在图像中的*位置的过程。**

在本教程中，我们将使用 [OpenCV 的基于深度学习的人脸检测器](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)，但你也可以轻松地使用哈尔级联、HOG +线性 SVM 或任何其他人脸检测方法。

### **步骤#2:获取面部 ROI 并执行年龄检测**

一旦我们定位了图像/视频流中的每个人脸，我们就可以确定他们的年龄。

我们将使用 Levi 和 Hassner 在他们 2015 年的出版物 *[中训练的年龄检测器，使用卷积神经网络](https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf)进行年龄和性别分类。*

[这个年龄检测模型与 OpenCV 兼容，如本教程](https://pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/)所述。

### **步骤#3:训练一个迷彩/军用疲劳检测器，并将其应用于图像**

潜在儿童兵的标志可能是穿着军装，这通常包括某种类似迷彩的图案。

训练伪装探测器在之前的教程中有所介绍——我们今天将在这里使用训练过的模型。

### **步骤#4:综合模型结果，寻找 18 岁以下穿军装的儿童**

最后一步是将我们的*年龄探测器*和我们的*军用疲劳/伪装探测器的结果结合起来。*

如果我们(1)在照片中检测到一个 18 岁以下的人，并且(2)图像中似乎还有伪装，我们会将该图像记录到磁盘中以供进一步查看。

### **配置您的开发环境**

要针对本教程配置您的系统，我首先建议您遵循以下任一教程:

*   *[如何在 Ubuntu 上安装 tensor flow 2.0](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)*
*   *[如何在 macOS 上安装 tensor flow 2.0](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)*

这两个教程都将帮助你为你的系统配置这篇博文**所需的所有软件，只有一个例外**。您还需要通过以下方式将``progressbar2`` 软件包安装到您的虚拟环境中:

```py
$ workon dl4cv
$ pip install progressbar2
```

配置好系统后，您就可以继续学习本教程的其余部分了。

### **项目结构**

请务必从 ***“下载”*** 部分获取今天教程的文件。我们的项目组织如下:

```py
$ tree --dirsfirst
.
├── models
│   ├── age_detector
│   │   ├── age_deploy.prototxt
│   │   └── age_net.caffemodel
│   ├── camo_detector
│   │   └── camo_detector.model
│   └── face_detector
│       ├── deploy.prototxt
│       └── res10_300x300_ssd_iter_140000.caffemodel
├── output
│   ├── ages.csv
│   └── camo.csv
├── pyimagesearch
│   ├── __init__.py
│   ├── config.py
│   └── helpers.py
├── parse_results.py
└── process_dataset.py

6 directories, 12 files
```

`models/`目录包含我们每个预先训练的深度学习模型:

*   人脸检测器
*   年龄分类器
*   伪装分类器

***注意:**我不能像往常一样在本指南的**“下载”**部分提供本教程中使用的原始数据集。该数据集是敏感的，不能以任何方式分发。*

### **我们的配置文件**

在深入研究我们的实现之前，让我们首先定义一个简单的配置文件，分别存储我们的人脸检测器、年龄检测器和伪装检测器模型的文件路径。

打开项目目录结构中的`config.py`文件，并插入以下代码:

```py
# import the necessary packages
import os

# define the path to our face detector model
FACE_PROTOTXT = os.path.sep.join(["models", "face_detector",
	"deploy.prototxt"])
FACE_WEIGHTS = os.path.sep.join(["models", "face_detector",
	"res10_300x300_ssd_iter_140000.caffemodel"])

# define the path to our age detector model
AGE_PROTOTXT = os.path.sep.join(["models", "age_detector",
	"age_deploy.prototxt"])
AGE_WEIGHTS = os.path.sep.join(["models", "age_detector",
	"age_net.caffemodel"])

# define the path to our camo detector model
CAMO_MODEL = os.path.sep.join(["models", "camo_detector",
	"camo_detector.model"])
```

通过使我们的配置成为 Python 文件并使用``os`` 模块，我们能够直接构建与操作系统无关的路径。

我们的配置包含:

*   人脸检测器模型路径(**行 5-8**)；一定要看我的深度学习[人脸检测器教程](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
*   老化检测器路径(**第 11-14 行**)；花点时间阅读我的[深度学习年龄检测教程](https://pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/)，其中介绍了这个模型
*   伪装探测器模型路径(**行 17 和 18**)；用深度学习阅读全部关于[迷彩服分类](https://pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/)

定义了每一条路径后，我们就可以在下一节中在一个单独的 Python 文件中定义便利函数了。

### **人脸检测、年龄预测、伪装检测和人脸匿名化的便利功能**

为了完成这个项目，我们将使用以前教程中涉及的许多计算机视觉/深度学习技术，包括:

*   [人脸检测](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
*   [年龄检测](https://pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/)
*   [伪装检测](https://pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/)
*   [人脸匿名化](https://pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/)

现在让我们在儿童兵检测项目的中心位置为每种技术定义便利函数。

***注:**关于人脸检测、人脸匿名化、年龄检测、迷彩服检测的更详细回顾，请务必点击上面相应的链接。*

打开`pyimagesearch`模块中的`helpers.py`文件，在输入图像中插入以下用于**检测人脸和预测年龄**的代码:

```py
# import the necessary packages
import numpy as np
import cv2

def detect_and_predict_age(image, faceNet, ageNet, minConf=0.5):
	# define the list of age buckets our age detector will predict
	# and then initialize our results list
	AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
		"(38-43)", "(48-53)", "(60-100)"]
	results = []
```

我们的助手工具只需要 OpenCV 和 NumPy ( **第 2 行和第 3 行**)。

让我们继续执行*面部检测:*

```py
	# grab the dimensions of the image and then construct a blob
	# from it
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
```

```py
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > minConf:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the ROI of the face
			face = image[startY:endY, startX:endX]

			# ensure the face ROI is sufficiently large
			if face.shape[0] < 20 or face.shape[1] < 20:
				continue
```

**第 23-37 行**在`detections`上循环，确保高`confidence`，并提取一个`face` ROI，同时确保它足够大，原因有二:

*   首先，我们想过滤掉图像中的假阳性人脸检测
*   第二，年龄分类结果对于远离相机的面部(即，可察觉的小)将是不准确的

为了完成我们的*人脸检测和年龄预测*辅助工具，我们将*执行人脸预测:*

```py
			# construct a blob from *just* the face ROI
			faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
				(78.4263377603, 87.7689143744, 114.895847746),
				swapRB=False)

			# make predictions on the age and find the age bucket with
			# the largest corresponding probability
			ageNet.setInput(faceBlob)
			preds = ageNet.forward()
			i = preds[0].argmax()
			age = AGE_BUCKETS[i]
			ageConfidence = preds[0][i]

			# construct a dictionary consisting of both the face
			# bounding box location along with the age prediction,
			# then update our results list
			d = {
				"loc": (startX, startY, endX, endY),
				"age": (age, ageConfidence)
			}
			results.append(d)

	# return our results to the calling function
	return results
```

```py
def detect_camo(image, camoNet):
	# initialize (1) the class labels the camo detector can predict
	# and (2) the ImageNet means (in RGB order)
	CLASS_LABELS = ["camouflage_clothes", "normal_clothes"]
	MEANS = np.array([123.68, 116.779, 103.939], dtype="float32")

	# resize the image to 224x224 (ignoring aspect ratio), convert
	# the image from BGR to RGB ordering, and then add a batch
	# dimension to the volume
	image = cv2.resize(image, (224, 224))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = np.expand_dims(image, axis=0).astype("float32")

	# perform mean subtraction
	image -= MEANS

	# make predictions on the input image and find the class label
	# with the largest corresponding probability
	preds = camoNet.predict(image)[0]
	i = np.argmax(preds)

	# return the class label and corresponding probability
	return (CLASS_LABELS[i], preds[i])
```

```py
def anonymize_face_pixelate(image, blocks=3):
	# divide the input image into NxN blocks
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")

	# loop over the blocks in both the x and y direction
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			# compute the starting and ending (x, y)-coordinates
			# for the current block
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]

			# extract the ROI using NumPy array slicing, compute the
			# mean of the ROI, and then draw a rectangle with the
			# mean RGB values over the ROI in the original image
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(B, G, R), -1)

	# return the pixelated blurred image
	return image
```

对于面部匿名化，我们将使用像素化类型的面部模糊。这种方法通常是大多数人听到“面部模糊”时想到的——这与你在晚间新闻中看到的面部模糊类型相同，主要是因为它比更简单的高斯模糊(这确实有点“不和谐”)更“美观”。

### **使用 OpenCV 和 Keras/TensorFlow 实现我们的潜在儿童兵探测器**

配置文件和助手函数就绪后，让我们继续将它们应用于可能包含儿童兵的图像数据集。

打开`process_dataset.py`脚本，插入以下代码:

```py
# import the necessary packages
from pyimagesearch.helpers import detect_and_predict_age
from pyimagesearch.helpers import detect_camo
from pyimagesearch import config
from tensorflow.keras.models import load_model
from imutils import paths
import progressbar
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input directory of images to process")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory where CSV files will be stored")
args = vars(ap.parse_args())
```

```py
# initialize a dictionary that will store output file pointers for
# our age and camo predictions, respectively
FILES = {}

# loop over our two types of output predictions
for k in ("ages", "camo"):
	# construct the output file path for the CSV file, open a path to
	# the file pointer, and then store it in our files dictionary
	p = os.path.sep.join([args["output"], "{}.csv".format(k)])
	f = open(p, "w")
	FILES[k] = f
```

这两个文件指针都是打开的，以便在进程中写入。

此时，我们将初始化三个深度学习模型:

```py
# load our serialized face detector, age detector, and camo detector
# from disk
print("[INFO] loading trained models...")
faceNet = cv2.dnn.readNet(config.FACE_PROTOTXT, config.FACE_WEIGHTS)
ageNet = cv2.dnn.readNet(config.AGE_PROTOTXT, config.AGE_WEIGHTS)
camoNet = load_model(config.CAMO_MODEL)

# grab the paths to all images in our dataset
imagePaths = sorted(list(paths.list_images(args["dataset"])))
print("[INFO] processing {} images".format(len(imagePaths)))

# initialize the progress bar
widgets = ["Processing Images: ", progressbar.Percentage(), " ",
	progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
	widgets=widgets).start()
```

我们现在进入数据集处理脚本的核心。我们将开始循环所有图像以检测面部，预测年龄，并确定是否存在伪装:

```py
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# load the image from disk
	image = cv2.imread(imagePath)

	# if the image is 'None', then it could not be properly read from
	# disk (so we should just skip it)
	if image is None:
		continue

	# detect all faces in the input image and then predict their
	# perceived age based on the face ROI
	ageResults = detect_and_predict_age(image, faceNet, ageNet)

	# use our camo detection model to detect whether camouflage exists in
	# the image or not
	camoResults = detect_camo(image, camoNet)
```

```py
	# loop over the age detection results
	for r in ageResults:
		# the output row for the ages CSV consists of (1) the image
		# file path, (2) bounding box coordinates of the face, (3)
		# the predicted age, and (4) the corresponding probability
		# of the age prediction
		row = [imagePath, *r["loc"], r["age"][0], r["age"][1]]
		row = ",".join([str(x) for x in row])

		# write the row to the age prediction CSV file
		FILES["ages"].write("{}\n".format(row))
		FILES["ages"].flush()
```

```py
	# check to see if our camouflage predictor was triggered
	if camoResults[0] == "camouflage_clothes":
		# the output row for the camo CSV consists of (1) the image
		# file path and (2) the probability of the camo prediction
		row = [imagePath, camoResults[1]]
		row = ",".join([str(x) for x in row])

		# write the row to the camo prediction CSV file
		FILES["camo"].write("{}\n".format(row))
		FILES["camo"].flush()
```

```py
	# update the progress bar
	pbar.update(i)

# stop the progress bar
pbar.finish()
print("[INFO] cleaning up...")

# loop over the open file pointers and close them
for f in FILES.values():
	f.close()
```

第 92 行更新了我们的进度条，此时，我们将从循环的顶部开始处理数据集中的下一幅图像。

**第 95-100 行**停止进度条并关闭 CSV 文件指针。

很好地实现了数据集处理脚本。在下一节中，我们将让它发挥作用！

### **处理我们的潜在儿童兵数据集**

```py
$ time python process_dataset.py --dataset VictorGevers_Dataset --output output
[INFO] loading trained models...
[INFO] processing 56037 images
Processing Images: 100% |############################| Time:  1:49:48
[INFO] cleaning up...

real	109m53.034s
user	428m1.900s
sys   306m23.741s
```

该数据集由 Victor Gevers 提供(即在数据泄露期间获得的数据集)。

在我的 3 GHz 英特尔至强 W 处理器上，处理整个数据集花费了将近两个小时,**—*,如果有 GPU，速度会更快。***

 **当然，我 ***不能像往常一样*在指南的*部分提供本教程中使用的原始数据集****。该数据集是私有的、敏感的，并且不能以任何方式分发。*

 *脚本执行完毕后，我的`output`目录中有两个 CSV 文件:

```py
$ ls output/
ages.csv	camo.csv
```

以下是`ages.csv`的输出示例:

```py
$ tail output/ages.csv 
rBIABl3RztuAVy6gAAMSpLwFcC0051.png,661,1079,1081,1873,(48-53),0.6324904
rBIABl3RzuuAbzmlAAUsBPfvHNA217.png,546,122,1081,1014,(8-12),0.59567857
rBIABl3RzxKAaJEoAAdr1POcxbI556.png,4,189,105,349,(48-53),0.49577188
rBIABl3RzxmAM6nvAABRgKCu0g4069.png,104,76,317,346,(8-12),0.31842607
rBIABl3RzxmAM6nvAABRgKCu0g4069.png,236,246,449,523,(60-100),0.9929517
rBIABl3RzxqAbJZVAAA7VN0gGzg369.png,41,79,258,360,(38-43),0.63570714
rBIABl3RzxyABhCxAAav3PMc9eo739.png,632,512,1074,1419,(48-53),0.5355053
rBIABl3RzzOAZ-HuAAZQoGUjaiw399.png,354,56,1089,970,(60-100),0.48260492
rBIABl3RzzOAZ-HuAAZQoGUjaiw399.png,820,475,1540,1434,(4-6),0.6595153
rBIABl3RzzeAb1lkAAdmVBqVDho181.png,258,994,826,2542,(15-20),0.3086191
```

如您所见，每行包含:

1.  图像文件路径
2.  特定面的边界框坐标
3.  面部年龄范围预测和相关概率

下面是来自`camo.csv`的输出示例:

```py
$ tail output/camo.csv 
rBIABl3RY-2AYS0RAAaPGGXk-_A001.png,0.9579516
rBIABl3Ra4GAScPBAABEYEkNOcQ818.png,0.995684
rBIABl3Rb36AMT9WAABN7PoYIew817.png,0.99894327
rBIABl3Rby-AQv5MAAB8CPkzp58351.png,0.9577539
rBIABl3Re6OALgO5AABY5AH5hJc735.png,0.7973979
rBIABl3RvkuAXeryAABlfL8vLL4072.png,0.7121747
rBIABl3RwaOAFX21AABy6JNWkVY010.png,0.97816855
rBIABl3Rz-2AUOD0AAQ3eMMg8gg856.png,0.8256913
rBIABl3RztOAeFb1AABG-K96F_c092.png,0.50594944
rBIABl3RzxeAGI5XAAfg5J_Svmc027.png,0.98626024
```

该 CSV 文件包含的信息较少，仅包含:

1.  图像文件路径
2.  指示图像是否包含伪装的概率

我们现在有了年龄和伪装预测。

但是我们如何结合这些预测来确定一个特定的图像是否有潜在的儿童兵呢？

我将在下一节回答这个问题。

### **实现 Python 脚本来解析我们的检测结果**

您可以轻松地将儿童兵数据输出到另一个 CSV 文件，并将其提供给报告机构(如果您正在做这种工作的话)。

相反，我们要开发的脚本只是在可疑的儿童兵图像中匿名化人脸(即应用我们的像素化模糊方法)并在屏幕上显示结果。

现在让我们来看看`parse_results.py`:

```py
# import the necessary packages
from pyimagesearch.helpers import anonymize_face_pixelate
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--ages", required=True,
	help="path to input ages CSV file")
ap.add_argument("-c", "--camo", required=True,
	help="path to input camo CSV file")
args = vars(ap.parse_args())
```

```py
# load the contents of the ages and camo CSV files
ageRows = open(args["ages"]).read().strip().split("\n")
camoRows = open(args["camo"]).read().strip().split("\n")

# initialize two dictionaries, one to store the age results and the
# other to store the camo results, respectively
ages = {}
camo = {}
```

```py
# loop over the age rows
for row in ageRows:
	# parse the row
	row = row.split(",")
	imagePath = row[0]
	bbox = [int(x) for x in row[1:5]]
	age = row[5]
	ageProb = float(row[6])

	# construct a tuple that consists of the bounding box coordinates,
	# age, and age probability
	t = (bbox, age, ageProb)

	# update our ages dictionary to use the image path as the key and
	# the detection information as a tuple
	l = ages.get(imagePath, [])
	l.append(t)
	ages[imagePath] = l
```

```py
# loop over the camo rows
for row in camoRows:
	# parse the row
	row = row.split(",")
	imagePath = row[0]
	camoProb = float(row[1])

	# update our camo dictionary to use the image path as the key and
	# the camouflage probability as the value
	camo[imagePath] = camoProb
```

```py
# find all image paths that exist in *BOTH* the age dictionary and
# camo dictionary
inter = sorted(set(ages.keys()).intersection(camo.keys()))

# loop over all image paths in the intersection
for imagePath in inter:
	# load the input image and grab its dimensions
	image = cv2.imread(imagePath)
	(h, w) = image.shape[:2]

	# if the width is greater than the height, resize along the width
	# dimension
	if w > h:
		image = imutils.resize(image, width=600)

	# otherwise, resize the image along the height
	else:
		image = imutils.resize(image, height=600)

	# compute the resize ratio, which is the ratio between the *new*
	# image dimensions to the *old* image dimensions
	ratio = image.shape[1] / float(w)
```

计算*新的*图像尺寸和*旧的*图像尺寸(**第 76 行**)之间的比率允许我们在下一个代码块中缩放我们的面部边界框。

让我们循环一下这张图片的年龄预测:

```py
	# loop over the age predictions for this particular image
	for (bbox, age, ageProb) in ages[imagePath]:
		# extract the bounding box coordinates of the face detection
		bbox = [int(x) for x in np.array(bbox) * ratio]
		(startX, startY, endX, endY) = bbox

		# anonymize the face
		face = image[startY:endY, startX:endX]
		face = anonymize_face_pixelate(face, blocks=5)
		image[startY:endY, startX:endX] = face

		# set the color for the annotation to *green*
		color = (0, 255, 0)

		# override the color to *red* they are potential child soldier
		if age in ["(0-2)", "(4-6)", "(8-12)", "(15-20)"]:
			color = (0, 0,  255)

		# draw the bounding box of the face along with the associated
		# predicted age
		text = "{}: {:.2f}%".format(age, ageProb * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
```

让我们更进一步，在图像的左上角标注伪装的概率:

```py
	# draw the camouflage prediction probability on the image
	label = "camo: {:.2f}%".format(camo[imagePath] * 100)
	cv2.rectangle(image, (0, 0), (300, 40), (0, 0, 0), -1)
	cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
		0.8, (255, 255, 255), 2)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
```

### **结果:利用计算机视觉和深度学习带来好处**

我们现在准备结合年龄预测和伪装输出的结果来确定特定图像是否包含潜在的儿童兵。

为了执行这个脚本，我使用了以下命令:

```py
$ python parse_results.py --ages output/ages.csv --camo output/camo.csv
```

***注:**出于隐私考虑，即使使用了面部匿名技术，我也不愿意分享 Victor Gevers 提供给我的数据集中的原始图片。我在网上放了其他图片的样本来证明这个脚本运行正常。我希望你能理解和欣赏我为什么做这个决定。*

下面是一个包含潜在儿童兵的图像示例:

这是我们检测潜在儿童兵的方法的第二个图像:

这里的年龄预测有点偏差。

我估计这位年轻女士(参见**图 1** 中的[原图](https://theirworld.org/news/what-next-for-colombia-child-soldiers-after-ceasefire))大约在 12-16 岁之间；然而，我们的年龄预测模型预测 4-6-年龄预测模型的局限性在下面的*【概要】*部分中讨论。

## **总结**

在本教程中，您学习了计算机视觉和深度学习的道德应用——识别潜在的儿童兵。

为了完成这项任务，我们应用了:

1.  **年龄检测** —用于检测图像中人的年龄
2.  **迷彩/疲劳检测** —用于检测图像中是否有迷彩，表明此人可能穿着军装

我们的系统相当准确，但正如我在我的[年龄检测帖子](https://pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/)以及[伪装检测教程](https://pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/)中所讨论的，结果可以通过以下方式得到改善:

1.  用平衡数据集训练更准确的年龄检测器
2.  收集额外的儿童图像，以便更好地识别儿童的年龄段
3.  通过应用更积极的数据扩充和正则化技术来训练更精确的伪装检测器
4.  通过服装分割构建更好的军用疲劳/制服检测器

我希望你喜欢这篇教程——我也希望你不会觉得这篇文章的主题太令人沮丧。

计算机视觉和深度学习，就像几乎任何产品或科学一样，可以用于善或恶。尽你所能保持好的一面。世界是一个可怕的地方——让我们一起努力，创造一个更美好的世界。

**要下载这篇文章的源代码(包括预先训练的人脸检测器、年龄检测器和伪装检测器模型)，*只需在下面的表格中输入您的电子邮件地址！******