# OpenCV 按位与、或、异或和非

> 原文：<https://pyimagesearch.com/2021/01/19/opencv-bitwise-and-or-xor-and-not/>

在本教程中，您将学习如何对 OpenCV 应用按位 AND、OR、XOR 和 NOT。

在我们之前关于使用 OpenCV 进行 [*裁剪的教程中，您学习了如何从图像中裁剪和提取感兴趣的区域(ROI)。*](https://pyimagesearch.com/2021/01/19/crop-image-with-opencv/)

在这个特殊的例子中，我们的 ROI 必须是矩形的。。。但是如果你想裁剪一个*非矩形*区域呢？

那你会怎么做？

答案是同时应用位运算和遮罩(我们将在关于使用 OpenCV 进行[图像遮罩的指南中讨论如何实现)。](https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/)

现在，我们将讨论基本的位运算——在下一篇博文中，我们将学习如何利用这些位运算来构造掩码。

**要学习如何用 OpenCV 应用位运算符，*继续阅读。***

## **OpenCV 按位与、或、异或、非**

在深入本教程之前，我假设您已经理解了四种基本的按位运算符:

1.  和
2.  运筹学
3.  异或(异或)
4.  不

如果你以前从未使用过位操作符，我建议你阅读一下来自 RealPython 的这篇优秀的(非常详细的)指南。

虽然你没有*也没有*来复习那本指南，但我发现理解对数字应用位运算符的基础的读者可以很快掌握对图像应用位运算符。

不管怎样，计算机视觉和图像处理是高度可视化的，我在本教程中精心制作了这些例子，以确保你理解如何用 OpenCV 将按位运算符应用于图像。

我们将从配置我们的开发环境开始，然后回顾我们的项目目录结构。

从这里，我们将实现一个 Python 脚本来执行 OpenCV 的 AND、OR、XOR 和 NOT 位操作符。

我们将以讨论我们的结果来结束本指南。

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

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

准备好学习如何使用 OpenCV 应用位运算符了吗？

太好了，我们开始吧。

请务必使用本指南的 ***【下载】*** 部分访问源代码，并从那里查看我们的项目目录结构:

```py
$ tree . --dirsfirst
.
└── opencv_bitwise.py

0 directories, 1 file
```

今天我们只回顾一个脚本`opencv_bitwise.py`，它将对示例图像应用 AND、OR、XOR 和 NOR 运算符。

在本指南结束时，您将会很好地理解如何在 OpenCV 中应用位操作符。

### **实现 OpenCV AND、OR、XOR 和 NOT 位运算符**

在这一节中，我们将回顾四种位运算:与、或、异或和非。虽然这四个操作非常基础和低级，但对图像处理至关重要——尤其是在本系列后面的蒙版处理中。

位运算以二进制方式运行，并表示为灰度图像。如果给定像素的值为零，则该像素被“关闭”,如果该像素的值大于零，则该像素被“打开”。

让我们继续，跳到一些代码中:

```py
# import the necessary packages
import numpy as np
import cv2

# draw a rectangle
rectangle = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv2.imshow("Rectangle", rectangle)

# draw a circle
circle = np.zeros((300, 300), dtype = "uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("Circle", circle)
```

对于代码导入的前几行，我们需要的包包括:NumPy 和我们的 OpenCV 绑定。

我们在**第 6 行**上将矩形图像初始化为一个 *300 x 300* 的 NumPy 数组。然后我们在图像的中心绘制一个 *250 x 250* 的白色矩形。

类似地，在**第 11 行，**我们初始化另一个图像来包含我们的圆，我们在**第 12 行**再次以图像的中间为中心，半径为 150 像素。

**图 2** 展示了我们的两种形状:

如果我们考虑这些输入图像，我们会看到它们只有两个像素强度值——要么像素是`0`(黑色)，要么像素大于零(白色)。**我们把只有两个像素亮度值的图像称为二值图像。**

另一种看待二进制图像的方式就像我们客厅里的开关。想象一下， *300 x 300* 图像中的每个像素都是一个灯开关。如果开关 ***关闭*** ，则像素值为零。但如果像素是 上的 ***，则具有大于零的值。***

在**图 2** 中，我们可以看到分别构成矩形和圆形的白色像素的像素值都为*开*，而周围像素的值为*关*。

在我们演示按位运算时，保持开/关的概念:

```py
# a bitwise 'AND' is only 'True' when both inputs have a value that
# is 'ON' -- in this case, the cv2.bitwise_and function examines
# every pixel in the rectangle and circle; if *BOTH* pixels have a
# value greater than zero then the pixel is turned 'ON' (i.e., 255)
# in the output image; otherwise, the output value is set to
# 'OFF' (i.e., 0)
bitwiseAnd = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND", bitwiseAnd)
cv2.waitKey(0)
```

正如我上面提到的，如果一个给定的像素的值大于零，它就被“打开”，如果它的值为零，它就被“关闭”。按位函数对这些二元条件进行运算。

为了利用位函数，我们假设(在大多数情况下)我们正在比较两个像素(唯一的例外是 NOT 函数)。我们将比较每个像素，然后构造我们的按位表示。

让我们快速回顾一下我们的二元运算:

*   **AND:** 当且仅当两个像素都大于零时，按位 AND 为真*。*
*   **OR:** 如果两个像素中的任何一个大于零，则按位 OR 为真*。*
*   **异或:**按位异或为真*当且仅当*两个像素中的一个大于零，*但不是两个都大于零。*
*   **NOT:** 按位 NOT 反转图像中的“开”和“关”像素。

在**第 21 行，**我们使用`cv2.bitwise_and`函数对矩形和圆形图像进行按位 AND 运算。如上所述，当且仅当两个像素都大于零时，按位 AND 运算才成立。我们的按位 AND 的输出可以在**图 3** 中看到:

我们可以看到我们的正方形的边缘丢失了——这是有意义的，因为我们的矩形没有圆形覆盖的面积大，因此两个像素都没有“打开”

现在让我们应用按位“或”运算:

```py
# a bitwise 'OR' examines every pixel in the two inputs, and if
# *EITHER* pixel in the rectangle or circle is greater than 0,
# then the output pixel has a value of 255, otherwise it is 0
bitwiseOr = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR", bitwiseOr)
cv2.waitKey(0)
```

我们使用`cv2.bitwise_or`函数对第 28 行的**进行按位或运算。如果两个像素中的任何一个大于零，则按位 OR 为真。看看图 4** 中**位或的输出:**

在这种情况下，我们的正方形和长方形已经合并。

接下来是按位异或:

```py
# the bitwise 'XOR' is identical to the 'OR' function, with one
# exception: the rectangle and circle are not allowed to *BOTH*
# have values greater than 0 (only one can be 0)
bitwiseXor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR", bitwiseXor)
cv2.waitKey(0)
```

我们使用`cv2.bitwise_xor`函数对**行 35** 进行逐位异或运算。

当且仅当两个像素中的一个大于零时，XOR 运算为真，但两个像素不能都大于零。

XOR 运算的输出显示在**图 5:**

在这里，我们看到正方形的中心被移除了。同样，这是有意义的，因为 XOR 运算不能使两个像素都大于零。

最后，我们得到了按位 NOT 函数:

```py
# finally, the bitwise 'NOT' inverts the values of the pixels;
# pixels with a value of 255 become 0, and pixels with a value of 0
# become 255
bitwiseNot = cv2.bitwise_not(circle)
cv2.imshow("NOT", bitwiseNot)
cv2.waitKey(0)
```

我们使用`cv2.bitwise_not`函数在**行 42** 上应用一个按位非运算。本质上，按位 NOT 函数翻转像素值。所有大于零的像素被设置为零，所有等于零的像素被设置为`255`:

注意我们的圆是如何被*反转*的——最初，圆是黑底白字，现在圆是白底黑字。

### **OpenCV 按位 AND、OR、XOR 和 NOT 结果**

要使用 OpenCV 执行位运算，请务必访问本教程的 ***“下载”*** 部分下载源代码。

从那里，打开一个 shell 并执行以下命令:

```py
$ python opencv_bitwise.py
```

您的输出应该与我在上一节中的输出相匹配。

## **总结**

在本教程中，您学习了如何使用 OpenCV 执行按位 AND、OR、XOR 和 NOT 运算。

虽然按位操作符本身似乎没什么用，但当你开始使用阿尔法混合和蒙版时，它们是必要的，我们将在另一篇博文中讨论这个概念。

在继续之前，请花时间练习并熟悉按位运算。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***