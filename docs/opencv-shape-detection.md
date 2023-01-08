# OpenCV 形状检测

> 原文：<https://pyimagesearch.com/2016/02/08/opencv-shape-detection/>

最后更新于 2021 年 7 月 7 日。

本教程是我们关于*形状检测和分析*的三部分系列的第二篇文章。

上周我们学习了如何使用 OpenCV 计算轮廓的[中心。](https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/)

今天，我们将利用轮廓属性来实际标记图像中的和*形状，就像这篇文章顶部的图一样。*

*   【2021 年 7 月更新:增加了新的章节，包括如何利用特征提取、机器学习和深度学习进行形状识别。

## OpenCV 形状检测

在开始本教程之前，让我们快速回顾一下我们的项目结构:

```py
|--- pyimagesearch
|    |--- __init__.py
|    |--- shapedetector.py
|--- detect_shapes.py
|--- shapes_and_colors.png

```

如您所见，我们定义了一个`pyimagesearch`模块。在这个模块中，我们有`shapedetector.py`，它将存储我们的`ShapeDetector`类的实现。

最后，我们有`detect_shapes.py`驱动程序脚本，我们将使用它从磁盘加载图像，分析它的形状，然后通过`ShapeDetector`类执行形状检测和识别。

在我们开始之前，确保您的系统上安装了 [imutils 包](https://github.com/jrosebr1/imutils)，这是一系列 OpenCV 便利函数，我们将在本教程的后面使用:

```py
$ pip install imutils

```

### 定义我们的形状检测器

构建形状检测器的第一步是编写一些代码来封装形状识别逻辑。

让我们继续定义我们的`ShapeDetector`。打开`shapedetector.py`文件并插入以下代码:

```py
# import the necessary packages
import cv2

class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)

```

**第 4 行**开始定义我们的`ShapeDetector`类。这里我们将跳过`__init__`构造函数，因为不需要初始化任何东西。

然后我们在**第 8 行**上有我们的`detect`方法，它只需要一个参数`c`，即我们试图识别的形状的轮廓(即轮廓)。

为了执行形状检测，我们将使用*轮廓逼近*。

顾名思义，轮廓近似是一种用减少的点集来减少曲线中点数的算法——因此有了术语*近似*。

这种算法通常被称为 [Ramer-Douglas-Peucker](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) 算法，或者简称为分裂合并算法。

轮廓近似是基于曲线可以由一系列短线段近似的假设。这导致了由原始曲线定义的点的子集组成的结果近似曲线。

轮廓近似实际上已经通过`cv2.approxPolyDP`方法在 OpenCV 中实现了。

为了执行轮廓近似，我们首先计算轮廓的周长(**线 11** )，然后构建实际的轮廓近似(**线 12** )。

第二个参数`cv2.approxPolyDP`的常用值通常在原始轮廓周长的 1-5%范围内。

***注:**有兴趣更深入地看看轮廓逼近吗？一定要去看看 [PyImageSearch 大师课程](https://pyimagesearch.com/pyimagesearch-gurus/)，在那里我详细讨论了计算机视觉和图像处理基础知识，比如轮廓和连通分量分析。*

给定我们的近似轮廓，我们可以继续执行形状检测:

```py
		# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"

		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"

		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"

		# return the name of the shape
		return shape

```

理解一个轮廓由一系列**顶点**组成是很重要的。我们可以检查列表中条目的数量来确定物体的形状。

例如，如果近似轮廓有*三个顶点*，那么它一定是一个三角形(**第 15 行和第 16 行**)。

如果一个轮廓有*个四顶点*，那么它一定是一个*正方形*或者是一个*矩形* ( **线 20** )。为了确定哪一个，我们计算形状的纵横比，纵横比就是轮廓边界框的宽度除以高度(**行 23 和 24** )。如果长宽比大约为 1.0，那么我们正在检查一个正方形(因为所有边的长度都大致相等)。否则，形状为矩形。

如果一个轮廓有*个五顶点*，我们可以把它标为*五边形* ( **第 31 和 32 线**)。

否则，通过排除过程(当然，在这个例子的上下文中)，我们可以假设我们正在检查的形状是一个*圆* ( **第 35 行和第 36 行**)。

最后，我们将识别出的形状返回给调用方法。

### 用 OpenCV 进行形状检测

既然已经定义了我们的`ShapeDetector`类，让我们创建`detect_shapes.py`驱动程序脚本:

```py
# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

```

我们从**线 2-5** 开始，导入我们需要的包。注意我们是如何从`pyimagesearch`的`shapedetector`子模块中导入`ShapeDetector`类的实现的。

**第 8-11 行**处理解析我们的命令行参数。这里我们只需要一个开关`--image`，它是我们想要处理的图像在磁盘上的路径。

接下来，让我们预处理我们的图像:

```py
# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

```

首先，我们在**行 15** 从磁盘加载我们的图像，并在**行 16** 调整它的大小。然后，我们在第 17 行的**处记录旧高度的`ratio`到新调整后的高度——我们将在本教程的后面部分找到我们这样做的确切原因。**

从那里，**行 21-23** 处理将调整大小的图像转换为灰度，平滑它以减少高频噪声，最后对它进行阈值处理以显示图像中的形状。

阈值处理后，我们的图像应该是这样的:

注意我们的图像是如何被二值化的——形状显示为白色前景的 T2 和黑色背景的 T4。

最后，我们在二进制图像中找到轮廓，根据 OpenCV 版本从`cv2.findContours` [中获取正确的元组值，最后初始化`ShapeDetector` ( **第 27-30 行**)。](https://pyimagesearch.com/2015/08/10/checking-your-opencv-version-using-python/)

最后一步是识别每个轮廓:

```py
# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 255, 255), 2)

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)

```

在第 33 行的**处，我们开始循环每个单独的轮廓。对于它们中的每一个，我们[计算轮廓的中心](https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/)，然后执行形状检测和标记。**

由于我们正在处理从*调整大小的图像*(而不是原始图像)中提取的轮廓，我们需要将轮廓和中心 *(x，y)*-坐标乘以我们的调整大小`ratio` ( **第 43-45 行**)。这将为我们提供原始图像的轮廓和质心的正确的 *(x，y)*-坐标。

最后，我们在图像上绘制轮廓和标记的形状(**第 44-48 行**)，然后显示我们的结果(**第 51 和 52 行**)。

要查看我们的形状检测器的运行情况，只需执行以下命令:

```py
$ python detect_shapes.py --image shapes_and_colors.png

```

从上面的动画中可以看到，我们的脚本分别遍历每个形状，对每个形状执行形状检测，然后在对象上绘制形状的名称。

### **使用特征提取和机器学习确定物体形状**

这篇文章演示了简单的轮廓属性，包括轮廓检测、轮廓近似和检查轮廓中的点数，如何用于识别图像中的形状。

然而，还有更先进的形状检测技术。**这些方法利用特征提取/图像描述符，并使用一系列数字(即“特征向量”)来量化图像中的形状。**

你应该研究的第一个方法是经典的[胡矩形符](https://pyimagesearch.com/2014/10/27/opencv-shape-descriptor-hu-moments-example/)。Hu 矩通过`cv2.HuMoments`函数内置到 OpenCV 库中。应用`cv2.HuMoments`的结果是用于量化图像中形状的七个数字的列表。

然后我们有 [Zernike moments](https://pyimagesearch.com/2014/04/07/building-pokedex-python-indexing-sprites-using-shape-descriptors-step-3-6/) 基于胡 moments 的研究和工作。应用 Zernike 矩的结果是用于量化图像中形状的 25 个数字的列表。Zernike 矩往往比 Hu 矩更强大，但可能需要一些手动参数调整(特别是矩的半径)。

### **可以用深度学习进行形状识别吗？**

**简而言之，是的，绝对的。**基于深度学习的模型 excel 以及物体和形状识别。如果您正在处理简单的形状，那么即使是浅层的 CNN 也可能胜过 Hu 矩、Zernike 矩和基于轮廓的形状识别方法——当然，前提是您有足够的数据来训练 CNN！

如果你有兴趣学习如何训练你自己的定制深度学习形状识别算法，请确保你参加了我在 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)的深度学习课程。

## 摘要

在今天的博文中，我们学习了如何使用 OpenCV 和 Python 进行形状检测。

为了实现这一点，我们利用*轮廓近似*，这是一个将曲线上的点数减少到更简单的*近似*版本的过程。

然后，基于这个轮廓近似值，我们检查了每个形状的顶点数。给定顶点数，我们能够准确地标记每个形状。

本课是形状检测和分析三部分系列的一部分。上周我们讲述了如何计算轮廓的[中心。今天我们讨论了 OpenCV 的形状检测。下周我们将讨论如何使用颜色通道统计 ***标记形状*** 的实际颜色。](https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/)

请务必在下面的表格中输入您的电子邮件地址，以便在下一篇帖子发布时得到通知— ***您不会想错过它的！***