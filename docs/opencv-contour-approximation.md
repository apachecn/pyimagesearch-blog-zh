# OpenCV 轮廓近似

> 原文：<https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/>

在本教程中，我们将了解 OpenCV 的轮廓近似的一步一步的实现和利用。

当我第一次偶然发现**轮廓逼近**的概念时，我想到的第一个问题是:**为什么是**？在我的机器学习及其相关领域的旅程中，我一直被教导数据就是一切。数据就是货币。你拥有的越多，你就越有可能成功。**图 1** 恰当地描述了这个场景。

因此，我不太理解逼近曲线数据点的概念。那不是更简单吗？那不会让我们丢失数据吗？我大吃一惊。

在今天的教程中，我们将学习 OpenCV 轮廓逼近，更准确地说，称为 Ramer–Douglas–peu cker 算法。您会惊讶地发现它对许多高优先级的实际应用程序是多么重要。

在这篇博客中，你将了解到:

*   什么是轮廓近似
*   理解轮廓近似所需的先决条件
*   如何实现轮廓逼近
*   轮廓逼近的一些实际应用

**学习如何实现轮廓逼近，** ***只需继续阅读。***

## **OpenCV 轮廓近似(cv2.approxPolyDP )**

让我们想象一个场景:你是一个自动驾驶机器人。你根据雷达(激光雷达、声纳等)收集的数据移动。).你必须不断地处理大量的数据，这些数据将被转换成你能理解的格式。然后，你会做出搬家所需的决定。在这种情况下，你最大的障碍是什么？是故意留在你路上的那块砖头吗？还是一条曲折的道路挡在你和目的地之间？

原来，简单的答案是…所有的人。想象一下，在给定时间内，你要获取多少数据来评估一种情况。原来*数据*一直在两面讨好。*数据*一直是我们的敌人吗？

虽然更多的数据确实给你的解决方案提供了更好的视角，但它也带来了计算复杂性和存储等问题。现在，你就像一个机器人，需要做出合理快速的决定来穿越你面前的路线。

这意味着简化复杂的数据将是您的首要任务。

假设你得到了一个你将要走的路线的俯视图。类似于**图 2。**

如果给定道路的确切宽度和其他参数，有一种方法可以简化这张地图，会怎么样？考虑到你的尺寸足够小，可以忽略多余的转弯(你可以直接走而不用沿着道路的确切曲线走的部分)，如果你可以删除一些多余的顶点，对你来说会更容易。类似**图 3** 的东西:

注意一些顶点是如何反复平滑的，从而产生一条更加线性的路线。很巧妙，不是吗？

这只是轮廓逼近在现实世界中的众多应用之一。在我们继续之前，让我们正式了解它是什么。

### **什么是轮廓逼近？**

轮廓近似法使用了**Ramer**–**Douglas**–**peu cker(RDP)**算法，旨在通过减少给定阈值的顶点来简化折线。通俗地说，我们选择一条曲线，减少它的顶点数，同时保留它的大部分形状。比如看一下**图 4** 。

这张来自维基百科 RDP 文章的信息丰富的 GIF 向我们展示了这种算法是如何工作的。我在这里给出算法的大概思路。

给定曲线的起点和终点，算法将首先找到与连接两个参考点的直线距离最大的顶点。姑且称之为`max_point`。如果`max_point`的距离小于阈值，我们会自动忽略起点和终点之间的所有顶点，并使曲线成为直线。

如果`max_point`位于阈值之外，我们将递归重复该算法，现在使`max_point`成为引用之一，并重复如图**图 4** 所示的检查过程。

注意某些顶点是如何被系统地消除的。最终，我们保留了大部分信息，但状态不太复杂。

这样一来，让我们看看如何使用 OpenCV 来利用 RDP 的力量！

### **配置您的开发环境**

为了遵循这个指南，您需要在您的系统上安装 OpenCV 库和``imutils`` 包。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
$ pip install imutils
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

### **项目结构**

让我们来看看项目结构。

```py
$ tree . --dirsfirst
.
├── opencv_contour_approx.py
└── shape.png

0 directories, 2 files
```

父目录包含一个脚本和一个图像:

*   `opencv_contour_approx.py`:项目中唯一需要的脚本包含了所有涉及到的编码。
*   我们将在其上测试轮廓近似的图像。

### **用 OpenCV 实现轮廓逼近**

在跳到轮廓近似之前，我们将通过一些先决条件来更好地理解整个过程。所以，事不宜迟，让我们跳进`opencv_contour_approx.py`开始编码吧！

```py
# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="shape.png",
	help="path to input image")
args = vars(ap.parse_args())
```

创建一个参数解析器实例是为了在用户选择想要修改的图像时给他们一个简单的命令行界面体验(**第 8-11 行**)。默认图像被设置为`shape.png`，该图像已经存在于目录中。然而，我们鼓励读者用他们自己的自定义图像来尝试这个实验！

```py
# load the image and display it
image = cv2.imread(args["image"])
cv2.imshow("Image", image)

# convert the image to grayscale and threshold it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 200, 255,
	cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
```

然后使用 OpenCV 的`imread`读取并显示作为参数提供的图像(**第 14 行和第 15 行**)。

图像将看起来像**图 6** :

由于我们将在图像中使用形状的边界，我们将图像从 RGB 转换为灰度(**线 **18**** )。一旦采用灰度格式，可以使用 OpenCV 的`threshold`函数(**行 **19-21**** )轻松分离出形状。结果见图 7**:**

注意，由于我们在**行 2 **0**** 上选择了`cv2.THRESH_BINARY_INV`作为参数，因此高亮度像素变为`0`，而周围的低亮度像素变为`255`。

```py
# find the largest contour in the threshold image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# draw the shape of the contour on the output image, compute the
# bounding box, and display the number of points in the contour
output = image.copy()
cv2.drawContours(output, [c], -1, (0, 255, 0), 3)
(x, y, w, h) = cv2.boundingRect(c)
text = "original, num_pts={}".format(len(c))
cv2.putText(output, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
	0.9, (0, 255, 0), 2)

# show the original contour image
print("[INFO] {}".format(text))
cv2.imshow("Original Contour", output)
cv2.waitKey(0)
```

使用 OpenCV 的`findContours`函数，我们可以挑出给定图像中所有可能的轮廓(取决于给定的参数)(**线 **24 和 25**T5)。我们使用了`RETR_EXTERNAL`参数，它只返回可用轮廓的单一表示。你可以在这里阅读更多相关信息[。](https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html)**

使用的另一个参数是`CHAIN_APPROX_SIMPLE`。这将删除单链线连接中的许多顶点，这些顶点本质上是多余的。

然后我们从轮廓数组中抓取最大的轮廓(这个轮廓属于形状)并在原始图像上跟踪它(**线 **26-36**** )。为此，我们使用 OpenCV 的`drawContours`函数。我们还使用`putText`函数在图像上书写。输出如**图 8** 所示:

现在，让我们演示一下轮廓近似可以做什么！

```py
# to demonstrate the impact of contour approximation, let's loop
# over a number of epsilon sizes
for eps in np.linspace(0.001, 0.05, 10):
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, eps * peri, True)

	# draw the approximated contour on the image
	output = image.copy()
	cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
	text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
	cv2.putText(output, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
		0.9, (0, 255, 0), 2)

	# show the approximated contour image
	print("[INFO] {}".format(text))
	cv2.imshow("Approximated Contour", output)
	cv2.waitKey(0)
```

如前所述，我们需要一个值`eps`，它将作为测量顶点的阈值。相应地，我们开始在一个范围内循环 epsilon 的(`eps`)值，以将其馈送给轮廓近似函数(**线**45**T5)。**

在**线 **47**** 上，使用`cv2.arcLength`计算轮廓的周长。然后我们使用`cv2.approxPolyDP`功能并启动轮廓近似过程(**线**48**)。`eps` × `peri`值作为近似精度，由于`eps`的递增性质，它将随着每个历元而变化。**

我们继续在每个时期追踪图像上的合成轮廓以评估结果(**线 **51-60**** )。

让我们看看结果！

### **轮廓近似结果**

在进行可视化之前，让我们看看轮廓近似如何影响这些值。

```py
$ python opencv_contour_approx.py
[INFO] original, num_pts=248
[INFO] eps=0.0010, num_pts=43
[INFO] eps=0.0064, num_pts=24
[INFO] eps=0.0119, num_pts=17
[INFO] eps=0.0173, num_pts=12
[INFO] eps=0.0228, num_pts=11
[INFO] eps=0.0282, num_pts=10
[INFO] eps=0.0337, num_pts=7
[INFO] eps=0.0391, num_pts=4
[INFO] eps=0.0446, num_pts=4
[INFO] eps=0.0500, num_pts=4
```

请注意，随着`eps`值的增加，轮廓中的点数不断减少。这表明近似法确实有效。请注意，在**的`eps`值为 0.0391** 时，点数开始饱和。让我们用可视化来更好地分析这一点。

### **轮廓近似可视化**

通过**图 9-12** ，我们记录了一些时期的轮廓演变。

注意曲线是如何逐渐变得越来越平滑的。随着阈值的增加，它变得越线性。当`eps`的值达到 **0.0500** 时，轮廓现在是一个完美的矩形，只有 **4 个**点。这就是拉默-道格拉斯-普克算法的威力。

### **学分**

受 [OpenCV 文档](https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html)启发的扭曲形状图像。

## **总结**

在这个数据驱动的世界中，简化数据同时保留大部分信息可能是最受欢迎的场景之一。今天，我们学习了如何使用 RDP 来简化我们的任务。它对矢量图形和机器人领域的贡献是巨大的。

RDP 还扩展到其他领域，如距离扫描，在那里它被用作去噪工具。我希望这篇教程能帮助你理解如何在工作中使用轮廓逼近。

### **引用信息**

**Chakraborty，**d .**“OpenCV 轮廓近似(cv2.approxPolyDP)，” *PyImageSearch* ，2021，[https://PyImageSearch . com/2021/10/06/OpenCV-Contour-Approximation/](https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/)**

`@article{dev2021opencv, author = {Devjyoti Chakraborty}, title = {Open{CV} Contour Approximation ( cv2.approx{PolyDP} )}, journal = {PyImageSearch}, year = {2021}, note = {https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/}, }`

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****