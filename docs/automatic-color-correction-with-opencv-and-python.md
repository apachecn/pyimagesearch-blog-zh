# 用 OpenCV 和 Python 实现自动色彩校正

> 原文：<https://pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/>

在本教程中，您将学习如何使用颜色匹配/平衡卡在 OpenCV 中执行自动颜色校正。

上周我们发现了如何执行[直方图匹配](https://pyimagesearch.com/2021/02/08/histogram-matching-with-opencv-scikit-image-and-python/)。使用直方图匹配，我们可以获取一幅图像的颜色分布，并将其与另一幅图像进行匹配。

色彩匹配的一个实际应用是通过**色彩恒常性进行基本的色彩校正。** 颜色恒常性的目标是正确感知物体的颜色*不管光源、光照等的差异*。(可以想象，说起来容易做起来难)。

摄影师和计算机视觉从业者可以通过使用颜色校正卡来帮助获得颜色恒常性，如下所示:

使用颜色校正/颜色恒常卡，我们可以:

1.  在输入图像中检测颜色校正卡
2.  计算卡片的直方图，它包含不同颜色、色调、阴影、黑色、白色和灰色的渐变颜色
3.  将色卡中的直方图匹配应用于另一幅图像，从而尝试实现颜色恒常性

在本教程中，我们将使用 OpenCV 构建一个颜色校正系统，将我们从之前的教程中学到的所有内容整合在一起:

1.  *[用 OpenCV 和 Python 检测 ArUco 标记](https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/)*
2.  *[【OpenCV 直方图均衡和自适应直方图均衡(CLAHE)](https://pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/)*
3.  *[直方图匹配用 OpenCV、scikit-image、Python](https://pyimagesearch.com/2021/02/08/histogram-matching-with-opencv-scikit-image-and-python/)*

完成本指南后，您将了解色彩校正卡如何与直方图匹配结合使用来构建基本色彩校正器的基本原理，*无论图像拍摄时的光照条件如何。*

**要了解如何使用 OpenCV 进行基本的色彩校正，*继续阅读。***

## **使用 OpenCV 和 Python 进行自动色彩校正**

在本教程的第一部分，我们将讨论什么是色彩校正和色彩恒常性，包括 OpenCV 如何促进自动色彩校正。

然后，我们将为这个项目配置我们的开发环境，并检查我们的项目目录结构。

准备好开发环境后，我们将实现一个 Python 脚本，利用 OpenCV 来执行颜色校正。

我们将讨论我们的结果来结束本教程。

### **什么是自动色彩校正？**

人类视觉系统受到照明和光源的显著影响。颜色恒常性是指对人类如何感知颜色的研究。

例如，看看维基百科关于[颜色恒常性](https://en.wikipedia.org/wiki/Color_constancy)的文章中的下图:

看这张卡片，似乎粉色阴影(左*第二张*)的*比*底部的粉色阴影* ***要强烈得多，但事实证明，它们是同一种颜色！****

这两张卡具有相同的 RGB 值。然而，我们人类的颜色感知系统会受到照片其余部分的色偏影响(即，在其顶部应用暖红色滤镜)。

如果我们试图使我们的图像处理环境正常化，这就产生了一点问题。正如我在之前关于 *[检测低对比度图像](https://pyimagesearch.com/2021/01/25/detecting-low-contrast-images-with-opencv-scikit-image-and-python/)* 的教程中所说:

> *为在* ***受控条件*** *下捕获的图像编写代码，要比在没有保证的* ***动态条件下容易得多。***

如果我们能够尽可能地控制我们的图像捕获环境，那么编写代码来分析和处理这些从受控环境中捕获的图像就会变得更加容易。

这么想吧。。。假设我们可以安全地假设一个环境的照明条件。在这种情况下，我们可以放弃昂贵的计算机视觉/深度学习算法，这些算法可以帮助我们在非理想条件下获得理想的结果。相反，我们利用基本的图像处理例程，允许我们硬编码参数，包括高斯模糊大小，Canny 边缘检测阈值等。

**本质上，有了*受控的环境，*我们可以摆脱基本的图像处理算法，而这些算法*更容易*实现。问题是我们需要对我们的照明条件做出安全的假设。色彩校正和白平衡有助于我们实现这一目标。**

我们可以帮助控制我们的环境的一种方法是应用色彩校正，即使照明条件有一点改变。

颜色检查卡是摄影师最喜欢的工具:

摄影师将这些卡片放入他们正在拍摄的场景中。然后，他们拍摄照片，调整他们的照明(同时仍然保持卡在相机的视野内)，然后继续拍摄，直到他们完成。

拍摄结束后，他们回到电脑前，将照片传输到他们的系统中，并使用工具，如 [Adobe Lightroom](https://www.adobe.com/products/photoshop-lightroom.html) 来实现整个拍摄过程中的颜色一致性([如果你感兴趣，这里有一个关于这个过程的教程](https://foodphotographyblog.com/what-is-a-color-checker-and-how-to-use-it/))。

当然，作为计算机视觉从业者，我们没有使用 Adobe Lightroom 的奢侈，我们也不想通过手动调整色彩平衡来启动/停止我们的管道——这违背了使用软件来自动化现实世界流程的整个目的。

相反，我们可以利用这些相同的颜色校正卡，加上一些[直方图匹配](https://pyimagesearch.com/2021/02/08/histogram-matching-with-opencv-scikit-image-and-python/)，我们可以构建一个能够执行颜色校正的系统。

在本指南的其余部分，您将利用直方图匹配和色彩校正卡(Pantone 的[)来执行*基本*色彩校正。](https://www.pantone.com/pantone-color-match-card)

### **潘通色彩校正卡**

在本教程中，我们将使用 [Pantone 的颜色匹配卡](https://www.pantone.com/pantone-color-match-card)。

该卡类似于摄影师使用的颜色校正卡，但 Pantone 使用它来帮助消费者将场景中感知的颜色与 Pantone 销售的油漆色调(与该颜色最相似)相匹配。

总的想法是:

1.  你把颜色校正卡放在你想要匹配的颜色上
2.  你在手机上打开 Pantone 的智能手机应用程序
3.  你给卡片拍了张照片
4.  该应用程序自动检测卡，执行颜色匹配，然后返回 Pantone 销售的最相似的色调

出于我们的目的，我们将严格使用该卡进行颜色校正(但您可以根据自己的需要轻松扩展它)。

### **配置您的开发环境**

要了解如何执行自动颜色校正，您需要安装 OpenCV 和 scikit-image:

两者都可以使用以下命令进行 pip 安装:

```py
$ pip install opencv-contrib-python
$ pip install scikit-image==0.18.1
```

**如果你需要帮助为 OpenCV 配置开发环境，我*强烈推荐*阅读我的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让你启动并运行。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   准备好在您的 Windows、macOS 或 Linux 系统上运行代码*了吗*？

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

虽然颜色匹配和颜色校正看起来是一个复杂的过程，但是我们会发现，我们能够用不到 100 行代码(包括注释)完成整个项目。

但是在我们开始编码之前，让我们先回顾一下我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像，然后查看文件夹:

```py
$ tree . --dirsfirst
.
├── examples
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
├── color_correction.py
└── reference.jpg

1 directory, 5 files
```

今天我们要回顾一个 Python 脚本，`color_correction.py`。该脚本将:

1.  加载我们的`reference.png`图像(包含我们的 Pantone 色彩校正卡)
2.  在`examples`目录中加载一个图像(我们将对其进行颜色校正以匹配`reference.png`)
3.  通过*中的 [ArUco 标记检测](https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/)检测配色卡*参考和输入图像
4.  应用直方图匹配来完善色彩校正过程

我们开始工作吧！

### **用 OpenCV 实现自动色彩校正**

我们现在准备用 OpenCV 和 Python 实现颜色校正。

打开项目目录结构中的`color_correction.py`文件，让我们开始工作:

```py
# import the necessary packages
from imutils.perspective import four_point_transform
from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2
import sys
```

我们从第 2-8 行的**开始，**导入我们需要的 Python 包。值得注意的包括:

*   `four_point_transform`:应用透视变换获得输入配色卡的*自上而下*鸟瞰图。参见[下面的教程](https://pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/)中使用该功能的示例。
*   `exposure`:包含 scikit-image 的直方图匹配功能。
*   `imutils`:我的一套使用 OpenCV 进行图像处理的便捷函数。
*   我们的 OpenCV 绑定。

处理好我们的导入后，我们可以继续定义`find_color_card`函数，这个方法负责在输入`image`中定位 Pantone 颜色匹配卡:

```py
def find_color_card(image):
	# load the ArUCo dictionary, grab the ArUCo parameters, and
	# detect the markers in the input image
	arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
	arucoParams = cv2.aruco.DetectorParameters_create()
	(corners, ids, rejected) = cv2.aruco.detectMarkers(image,
		arucoDict, parameters=arucoParams)
```

我们的`find_color_card`函数只需要一个参数`image`，它是(大概)包含我们的颜色匹配卡的图像。

从那里，**行 13-16** 执行 [ArUco 标记检测](https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/)以找到配色卡上的四个 ArUco 标记。

接下来，让我们按*左上*、*右上*、*右下*和*左下*的顺序排列四个 ArUco 标记(应用*自上而下*透视变换所需的顺序):

```py
	# try to extract the coordinates of the color correction card
	try:
		# otherwise, we've found the four ArUco markers, so we can
		# continue by flattening the ArUco IDs list
		ids = ids.flatten()

		# extract the top-left marker
		i = np.squeeze(np.where(ids == 923))
		topLeft = np.squeeze(corners[i])[0]

		# extract the top-right marker
		i = np.squeeze(np.where(ids == 1001))
		topRight = np.squeeze(corners[i])[1]

		# extract the bottom-right marker
		i = np.squeeze(np.where(ids == 241))
		bottomRight = np.squeeze(corners[i])[2]

		# extract the bottom-left marker
		i = np.squeeze(np.where(ids == 1007))
		bottomLeft = np.squeeze(corners[i])[3]

	# we could not find color correction card, so gracefully return
	except:
		return None
```

首先，我们将整个代码块包装在一个`try/except`块中。我们这样做只是为了防止使用`np.where`调用无法检测到所有四个标记。如果只有一个`np.where`调用失败，Python 将抛出一个错误。

我们的`try/except`块将捕获错误并返回`None`，暗示找不到颜色校正卡。

否则，**行 25-38** 按*左上*、*右上*、*右下*和*左下*的顺序提取每个单独的阿鲁科标记。

***注意:*** *你可能想知道我怎么知道每个标记的 id 是`923`、`1001`、`241`和`1007`？* [*这在我之前的一套 ArUco 标记检测教程中已经解决了。如果你还没有看过那本教程，一定要看一看。*](https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/)

假设我们找到了所有四个 ArUco 标记，我们现在可以应用透视变换:

```py
	# build our list of reference points and apply a perspective
	# transform to obtain a top-down, bird’s-eye view of the color
	# matching card
	cardCoords = np.array([topLeft, topRight,
		bottomRight, bottomLeft])
	card = four_point_transform(image, cardCoords)

	# return the color matching card to the calling function
	return card
```

**第 47-49 行**从我们的 ArUco 标记坐标构建一个 NumPy 数组，然后应用`four_point_transform`函数获得一个*自上而下*，颜色校正的鸟瞰图`card`。

将`card`的这个*自顶向下的*视图返回给调用函数。

实现了我们的`find_color_card`函数后，让我们继续解析命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reference", required=True,
	help="path to the input reference image")
ap.add_argument("-i", "--input", required=True,
	help="path to the input image to apply color correction to")
args = vars(ap.parse_args())
```

为了执行颜色匹配，我们需要两幅图像:

1.  到`--reference`图像的路径包含“理想”条件下的输入场景，我们希望将任何输入图像校正到该条件下。
2.  到`--input`图像的路径，我们假设它有不同的颜色分布，大概是由于光照条件的变化。

我们的目标是获取`--input`图像并执行颜色匹配，使其分布与`--reference`图像的分布相匹配。

但在此之前，我们需要从磁盘加载参考和源图像:

```py
# load the reference image and input images from disk
print("[INFO] loading images...")
ref = cv2.imread(args["reference"])
image = cv2.imread(args["input"])

# resize the reference and input images
ref = imutils.resize(ref, width=600)
image = imutils.resize(image, width=600)

# display the reference and input images to our screen
cv2.imshow("Reference", ref)
cv2.imshow("Input", image)
```

**第 64 行和第 65 行**从磁盘加载我们的输入图像，而**第 68 行和第 69 行**通过调整到 600 像素的宽度对它们进行预处理(以更快地处理图像)。

**第 72 行和第 73 行**然后将原始的`ref`和`image`显示到我们的屏幕上。

加载完我们的图像后，现在让我们将`find_color_card`函数应用于两幅图像:

```py
# find the color matching card in each image
print("[INFO] finding color matching cards...")
refCard = find_color_card(ref)
imageCard = find_color_card(image)

# if the color matching card is not found in either the reference
# image or the input image, gracefully exit
if refCard is None or imageCard is None:
	print("[INFO] could not find color matching card in both images")
	sys.exit(0)
```

**线 77 和 78** 试图在`ref`和`image`中定位配色卡。

如果我们在任一图像中都找不到颜色匹配卡，我们优雅地退出脚本(**第 82-84 行**)。

否则，我们可以安全地假设我们找到了颜色匹配卡，所以让我们应用颜色校正:

```py
# show the color matching card in the reference image and input image,
# respectively
cv2.imshow("Reference Color Card", refCard)
cv2.imshow("Input Color Card", imageCard)

# apply histogram matching from the color matching card in the
# reference image to the color matching card in the input image
print("[INFO] matching images...")
imageCard = exposure.match_histograms(imageCard, refCard,
	multichannel=True)

# show our input color matching card after histogram matching
cv2.imshow("Input Color Card After Matching", imageCard)
cv2.waitKey(0)
```

**第 88 和 89 行**将我们的`refCard`和`imageCard`显示到我们的屏幕上。

然后我们应用`match_histograms`函数将颜色分布从`refCard`转移到`imageCard`。

最后，直方图匹配后的输出`imageCard`、*、*显示在我们的屏幕上。这个新的`imageCard`现在包含了原`imageCard`的色彩校正版本。

### **自动色彩校正结果**

我们现在已经准备好用 OpenCV 进行自动色彩校正了！

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

从那里，您可以打开一个 shell 并执行以下命令:

```py
$ python color_correction.py --reference reference.jpg \
	--input examples/01.jpg
[INFO] loading images...
[INFO] finding color matching cards...
[INFO] matching images...
```

在左边的*，*是我们的参考图像。请注意我们是如何将颜色校正卡放在蓝绿色阴影上的。**我们这里的目标是确保蓝绿色的** **阴影在所有*输入图像中保持一致，*不管光照条件如何变化。****

现在，检查右边*的照片。这是我们的示例输入图像。**你可以看到，由于光照条件的原因，蓝绿色比参考图像中的蓝绿色略亮。***

怎样才能纠正这种表象？

答案是应用颜色校正:

在左边的*，*我们已经检测到参考图像中的色卡。*中间的*显示来自输入图像的色卡。**最后*右*显示配色后的输入色卡*。***

注意右边*的蓝绿色*与输入参考图像中的蓝绿色更加相似(即右边*的蓝绿色*比中间*的蓝绿色*更暗)。

让我们尝试另一个图像:

```py
$ python color_correction.py --reference reference.jpg \
	--input examples/02.jpg
[INFO] loading images...
[INFO] finding color matching cards...
[INFO] matching images...
```

同样，我们从我们的参考图像*(左)*和我们的输入图像*(右)*开始，我们试图对它们应用颜色校正。

下面是我们应用颜色匹配后的输出:

左边的*包含来自参考图像的配色卡，而中间的*显示来自输入图像(`02.jpg`)的配色卡。你可以看到中间*图中的蓝绿色明显比左边*图中的蓝绿色亮。****

 *通过应用颜色匹配和校正，我们可以校正这个视差*(右)*。注意左边的*和右边的*的蓝绿色更加相似。**

这是最后一个例子:

```py
$ python color_correction.py --reference reference.jpg \
	--input examples/03.jpg
[INFO] loading images...
[INFO] finding color matching cards...
[INFO] matching images...
```

这里的光照条件与前两者有明显的不同。左边*的图像*是我们的参考图像(在我的办公室拍摄)，而右边*的图像*是输入图像(在我的卧室拍摄)。

由于卧室中的窗户以及那天阳光是如何进入窗户的，配色卡的*右侧*有明显的阴影，从而使这更具挑战性(并展示了这种基本颜色校正方法的一些局限性)。

下面是通过直方图匹配应用颜色校正的输出:

左边的*图像是我们参考图像中的颜色匹配卡。然后我们就有了从输入图像中检测到的颜色校正卡(`03.jpg`)。*

应用直方图匹配产生右侧的*图像。虽然我们仍然有阴影，我们可以看到来自*中间*的较亮的蓝绿色已经被修正，以更相似地匹配来自参考图像的原始较暗的蓝绿色。*

## **总结**

在本教程中，您学习了如何使用 OpenCV 和 Python 执行基本的颜色校正。

我们通过以下方式实现了这一目标:

1.  将颜色校正卡放在我们相机的视野中
2.  拍摄现场照片
3.  使用 ArUco 标记检测来检测颜色校正卡
4.  应用直方图匹配将卡片的颜色分布转移到另一幅图像

综合起来，我们可以把这个过程看作是一个色彩校正过程(尽管是非常基本的)。

实现纯色恒常性，尤其是在没有标记/颜色校正卡的情况下实现纯色恒常性，仍然是一个活跃的研究领域，并且可能会持续很多年。但与此同时，我们可以利用直方图匹配和颜色匹配卡让我们朝着正确的方向前进。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****