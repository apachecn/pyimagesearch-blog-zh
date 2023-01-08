# 用 OpenCV 进行多模板匹配

> 原文：<https://pyimagesearch.com/2021/03/29/multi-template-matching-with-opencv/>

在本教程中，您将学习如何使用 OpenCV 执行多模板匹配。

上周您发现了如何利用 OpenCV 和`cv2.matchTemplate`函数进行基本的模板匹配。**这种方法的问题是它只能检测输入图像中模板的*一个实例*—*你不能执行多目标检测！***

我们只能检测到一个物体，因为我们使用了`cv2.minMaxLoc`函数来寻找具有*最大*归一化相关分数的*单一位置*。

为了执行多对象模板匹配，我们需要做的是:

1.  像平常一样应用`cv2.matchTemplate`功能
2.  找出模板匹配结果矩阵大于预设阈值分数的所有 *(x，y)*-坐标
3.  提取所有这些区域
4.  对它们应用非最大值抑制

应用以上四个步骤后，我们将能够在输入图像中检测多个模板。

**要学习如何用 OpenCV 进行多模板匹配，*继续阅读。***

## **OpenCV 多模板匹配**

在本教程的第一部分，我们将讨论基本模板匹配的问题，以及如何使用一些基本的计算机视觉和图像处理技术将其扩展到*多模板匹配*。

然后，我们将配置我们的开发环境，并检查我们的项目目录结构。

从那里，我们将使用 OpenCV 实现多模板匹配。

我们将讨论我们的结果来结束本教程。

### **基本模板匹配的问题**

正如我们在[上周的教程](https://pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/)中看到的，应用基本模板匹配只会导致*一个特定模板的*实例被匹配，如图**图 1** 所示。

我们的输入图像包含方块 8。虽然我们的模板包含钻石符号，但我们希望检测输入图像中的所有钻石。

然而，当使用基本模板匹配时，*多目标检测根本不可能。*

解决方案是从`cv2.matchTemplate`函数中过滤结果矩阵，然后应用非最大值抑制。

### **如何用 OpenCV 匹配** ***多个*** **模板？**

为了使用 OpenCV 和`cv2.matchTemplate`检测多个对象/模板，我们需要过滤由`cv2.matchTemplate`生成的`result`矩阵，如下所示:

```py
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
(yCoords, xCoords) = np.where(result >= args["threshold"])
```

调用`cv2.matchTemplate`会产生一个具有以下空间维度的`result`矩阵:

*   **宽度:** `image.shape[1] - template.shape[1] + 1`
*   **身高:** `image.shape[0] - template.shape[0] + 1`

然后，我们应用`np.where`函数来寻找归一化相关系数大于预设阈值的所有 *(x，y)——*坐标——**这个阈值步骤允许我们执行多模板匹配！**

最后一步，我们将在本教程稍后介绍，是应用[非最大值抑制](https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/)来过滤由`np.where`过滤生成的重叠边界框。

应用这些步骤后，我们的输出图像将如下所示:

请注意，我们已经检测到(几乎)输入图像中的所有钻石。

### **配置您的开发环境**

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果你需要帮助为 OpenCV 配置开发环境，我*强烈推荐*阅读我的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让你启动并运行。

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   准备好在您的 Windows、macOS 或 Linux 系统上运行代码*了吗*？

那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

让我们花点时间来检查一下我们的项目目录结构。请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

解压代码档案后，您会发现以下目录:

```py
$ tree . --dirsfirst
.
├── images
│   ├── 8_diamonds.png
│   └── diamonds_template.png
└── multi_template_matching.py

1 directory, 5 files
```

我们今天只回顾一个 Python 脚本`multi_template_matching.py`，它将使用我们的`images`目录中的输入图像执行多模板匹配。

### **用 OpenCV 实现多模板匹配**

上周我们学习了如何执行[模板匹配](https://pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/)。这种方法的问题在于，当模板的*多次出现*在输入图像中时失败——模板匹配将仅报告*一个*匹配的模板(即，具有最大相关分数的模板)。

我们将要讨论的 Python 脚本`multi_template_matching.py`，将扩展我们的基本模板匹配方法，并允许我们匹配*多个模板*。

让我们开始吧:

```py
# import the necessary pages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
```

**第 2-5 行**导入我们需要的 Python 包。最重要的是，我们需要本教程中的`non_max_suppression`函数，它执行[非最大值抑制](https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/) (NMS)。

应用多模板匹配将导致在我们的输入图像中对*的每个*物体进行*多次检测*。我们可以通过应用 NMS 来抑制弱的重叠边界框，从而解决这一问题。

从那里，我们解析我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image where we'll apply template matching")
ap.add_argument("-t", "--template", type=str, required=True,
	help="path to template image")
ap.add_argument("-b", "--threshold", type=float, default=0.8,
	help="threshold for multi-template matching")
args = vars(ap.parse_args())
```

我们有三个参数要解析，其中两个是必需的，第三个是可选的:

1.  `--image`:我们将应用多模板匹配的输入图像的路径。
2.  `--template`:模板图像的路径(即我们想要检测的物体的例子)。
3.  `--threshold`:用于 NMS 的阈值——范围*【0.8，0.95】*内的值通常效果最佳。

接下来，让我们从磁盘加载我们的`image`和`template`:

```py
# load the input image and template image from disk, then grab the
# template image spatial dimensions
print("[INFO] loading images...")
image = cv2.imread(args["image"])
template = cv2.imread(args["template"])
(tH, tW) = template.shape[:2]

# display the  image and template to our screen
cv2.imshow("Image", image)
cv2.imshow("Template", template)
```

**线 20 和 21** 从磁盘加载我们的`image`和`template`。我们在**第 22 行**上获取模板的空间维度，这样我们就可以用它们轻松地导出匹配对象的边界框坐标。

**第 25 行和第 26 行**将我们的`image`和`template`显示到我们的屏幕上。

下一步是执行模板匹配，就像我们上周做的一样:

```py
# convert both the image and template to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# perform template matching
print("[INFO] performing template matching...")
result = cv2.matchTemplate(imageGray, templateGray,
	cv2.TM_CCOEFF_NORMED)
```

**第 29 和 30 行**将我们的输入图像转换成灰度，而**第 34 和 35 行**执行模板匹配。

如果我们希望只检测模板的一个实例*，我们可以简单地调用`cv2.minMaxLoc`来找到具有最大归一化相关系数的 *(x，y)*-坐标。*

然而，由于我们想要检测*多个*物体，我们需要过滤我们的`result`矩阵并找到 ***所有的*** *(x，y)*-分数大于我们的`--threshold`的坐标:

```py
# find all locations in the result map where the matched value is
# greater than the threshold, then clone our original image so we
# can draw on it
(yCoords, xCoords) = np.where(result >= args["threshold"])
clone = image.copy()
print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))

# loop over our starting (x, y)-coordinates
for (x, y) in zip(xCoords, yCoords):
	# draw the bounding box on the image
	cv2.rectangle(clone, (x, y), (x + tW, y + tH),
		(255, 0, 0), 3)

# show our output image *before* applying non-maxima suppression
cv2.imshow("Before NMS", clone)
cv2.waitKey(0)
```

**第 40 行**使用`np.where`来查找所有 *(x，y)*-相关分数大于我们的`--threshold`命令行参数的坐标。

**第 42 行**显示在应用 NMS 之前*匹配位置的总数。*

从那里，我们循环所有匹配的 *(x，y)*-坐标，并在屏幕上绘制它们的边界框(**第 45-48 行**)。

如果我们在这里结束我们的实现，我们将会有一个问题——对`np.where`的调用将会返回 *(x，y)* 的所有位置的*,这些坐标都在我们的阈值之上。*

**很有可能多个位置指向*同一个*物体。**如果发生这种情况，我们基本上会多次报告同一个对象*，这是我们不惜一切代价想要避免的。*

 *解决方案是应用非最大值抑制:

```py
# initialize our list of rectangles
rects = []

# loop over the starting (x, y)-coordinates again
for (x, y) in zip(xCoords, yCoords):
	# update our list of rectangles
	rects.append((x, y, x + tW, y + tH))

# apply non-maxima suppression to the rectangles
pick = non_max_suppression(np.array(rects))
print("[INFO] {} matched locations *after* NMS".format(len(pick)))

# loop over the final bounding boxes
for (startX, startY, endX, endY) in pick:
	# draw the bounding box on the image
	cv2.rectangle(image, (startX, startY), (endX, endY),
		(255, 0, 0), 3)

# show the output image
cv2.imshow("After NMS", image)
cv2.waitKey(0)
```

**第 55 行**开始初始化我们的边界框列表`rects`。然后我们循环所有的 *(x，y)*-坐标，计算它们各自的边界框，然后更新`rects`列表。

在**行 63** 上应用非最大值抑制，抑制具有较低分数的重叠边界框，本质上将多个重叠检测压缩成单个检测。

最后，**行 67-70** 在我们最终的边界框上循环，并在我们的输出`image`上绘制它们。

### **多模板匹配结果**

我们现在准备用 OpenCV 应用多模板匹配！

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例图像。

从那里，打开一个终端并执行以下命令:

```py
$ python multi_template_matching.py --image images/8_diamonds.png \
	--template images/diamonds_template.png
[INFO] loading images...
[INFO] performing template matching...
[INFO] 601 matched locations *before* NMS
[INFO] 8 matched locations *after* NMS
```

**图 4** 显示我们的`diamonds_template.png` *(左)*`8_diamonds.png`图像*(右)。*我们的目标是检测*右*图像中的*所有*菱形符号。

在应用了`cv2.matchTemplate`函数之后，我们过滤得到的矩阵，找到归一化相关系数大于我们的`--threshold`参数的 *(x，y)-* 坐标。

这个过程产生总共 **601 个匹配的对象，**，我们在下面可视化:

查看**图 5** ，以及我们的终端输出，您可能会惊讶地发现我们有 601 个匹配的区域— *这怎么可能？！*方块 8 的牌上只有 8 颗方块(如果你用`8`数字本身来计算额外的方块，那就是 10 颗)——但那肯定加起来不到 601 个匹配！

这个现象是我在我的[非极大值抑制教程](https://pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/)中讨论的。对象检测算法类似于“热图”一个[滑动窗口](https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)越靠近图像中的一个物体，热图就变得“越来越热”。

然后，当我们使用`np.where`调用过滤这个热图时，我们最终得到了*所有*高于给定阈值的位置。**记住`np.where`函数*不知道*一幅图像中有多少物体——它只是告诉你哪里有*可能的*物体。**

这里的解决方案很简单，几乎所有的对象检测算法(包括基于高级深度学习的算法)都使用非极大值抑制(NMS)。

使用 NMS，我们检查相关系数得分，并抑制那些(1)重叠和(2)得分低于其周围邻居的得分。

应用 NMS 产生菱形符号的 8 个匹配位置:

卡片角上的*【8】*数字旁边的小钻石呢？为什么那些没有被检测出来？

这又回到了模板匹配的主要限制之一:

当您想要检测的对象在比例、旋转和视角方面开始不同时，模板匹配将会失败。

**由于钻石的尺寸比我们的模板*小*，标准的模板匹配程序将无法检测出它们。**

当这种情况发生时，你可以依靠[多尺度模板匹配](https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/)。或者，您可能需要考虑训练一个可以自然处理这些类型变化的对象检测器，例如大多数基于深度学习的对象检测器(例如，更快的 R-CNN、SSDs、YOLO 等)。).

### **演职员表和参考资料**

我要感谢在线教师[关于模板匹配的精彩文章](https://theailearner.com/2020/12/12/template-matching-using-opencv/)——我不能因为用扑克牌来演示模板匹配的想法而居功。这是他们的想法，而且是一个很好的想法。感谢他们想出了这个例子，我无耻地用在这里，谢谢。

此外，u/fireball_73 从 [Reddit 帖子](https://www.reddit.com/r/mildlyinfuriating/comments/91bqun/this_8_of_diamonds_card_has_10_diamonds_on_it/)中获得了方块 8 的图像。

## **总结**

在本教程中，您学习了如何使用 OpenCV 执行多模板匹配。

与只能检测输入图像中模板的单个实例的基本模板匹配不同，多模板匹配允许我们检测模板的多个实例*。*

应用多对象模板匹配是一个四步过程:

1.  像平常一样应用`cv2.matchTemplate`功能
2.  找出模板匹配结果矩阵大于预设阈值分数的所有 *(x，y)*-坐标
3.  提取所有这些区域
4.  对它们应用非最大值抑制

虽然这种方法可以处理多对象模板匹配，但它仍然容易受到模板匹配的其他限制—如果对象的比例、旋转或视角发生变化，模板匹配可能会失败。

你也许可以利用[多尺度模板匹配](https://pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/)(不同于*多模板*匹配)。不过，如果你到了那一步，你可能会想看看更先进的对象检测方法，如猪+线性 SVM，更快的 R-CNN，固态硬盘和 YOLO。

无论如何，模板匹配是超级快速、高效且易于实现的。因此，在执行模板匹配时，作为“第一步”是值得的(只是要事先意识到限制)。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****