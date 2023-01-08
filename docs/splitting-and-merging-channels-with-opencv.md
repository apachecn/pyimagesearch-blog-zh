# 使用 OpenCV 拆分和合并通道

> 原文：<https://pyimagesearch.com/2021/01/23/splitting-and-merging-channels-with-opencv/>

在本教程中，您将学习如何使用 OpenCV 分割和合并通道。

我们知道，图像由三个部分表示:红色、绿色和蓝色通道。

虽然我们已经简要讨论了图像的灰度和二进制表示，但您可能想知道:

> 如何访问图像的每个单独的红色、绿色和蓝色通道？

由于 OpenCV 中的图像在内部表示为 NumPy 数组，因此可以通过多种方式来访问每个通道，这意味着可以通过多种方式来获取图像。然而，我们将关注您应该使用的两个主要方法:`cv2.split`和`cv2.merge`。

在本教程结束时，你会很好地理解如何使用`cv2.split`将图像分割成通道，并使用`cv2.merge`将各个通道合并在一起。

**要学习如何使用 OpenCV 拆分和合并频道，*继续阅读。***

## **使用 OpenCV 拆分和合并频道**

在本教程的第一部分，我们将配置我们的开发环境并回顾我们的项目结构。

然后，我们将实现一个 Python 脚本，它将:

1.  从磁盘加载输入图像
2.  将它分成各自的红色、绿色和蓝色通道
3.  将每个频道显示在我们的屏幕上，以便可视化
4.  将各个通道合并在一起，形成原始图像

我们开始吧！

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
*   **准备好在你的 Windows、macOS 或 Linux 系统上运行代码*了吗？***

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

让我们从回顾我们的项目目录结构开始。请务必使用本教程的 ***【下载】*** 部分下载源代码和示例图像:

```py
$ tree . --dirsfirst
.
├── adrian.png
├── opencv_channels.py
└── opencv_logo.png

0 directories, 3 files
```

在我们的项目中，您会看到我们有一个单独的 Python 脚本，`opencv_channels.py`，它将向我们展示:

1.  如何将输入图像(`adrian.png`和`opencv_logo.png`)分割成各自的红色、绿色和蓝色通道
2.  可视化每个 RGB 通道
3.  将 RGB 通道合并回原始图像

我们开始吧！

### **如何用 OpenCV 拆分和合并频道**

彩色图像由多个通道组成:红色、绿色和蓝色分量。我们已经看到，我们可以通过索引 NumPy 数组来访问这些组件。但是如果我们想把一幅图像分割成各个部分呢？

正如您将看到的，我们将利用`cv2.split`函数。

但是现在，让我们来看看**图 2 中的一个示例图像:**

这里，我们有(按外观顺序)红色、绿色、蓝色和我去佛罗里达旅行的原始图像。

但是给定这些表征，我们如何解释图像的不同通道呢？

让我们来看看*原始*图像(*右下*)中天空的颜色。请注意天空是如何略带蓝色的。当我们看蓝色通道的图像(*左下角*，我们看到蓝色通道在对应天空的区域非常亮。这是因为蓝色通道像素非常亮，表明它们对输出图像的贡献很大。

然后，看看我穿的黑色连帽衫。在图像的红色、绿色和蓝色通道中，我的黑色连帽衫非常暗，这表明这些通道对输出图像的连帽衫区域的贡献非常小(赋予它非常暗的黑色)。

当你单独调查每个通道*而不是作为一个整体*时，你可以想象每个通道对整体输出图像的贡献。执行此练习非常有帮助，尤其是在应用阈值处理和边缘检测等方法时，我们将在本模块的后面介绍这些方法。**

 **现在我们已经可视化了我们的通道，让我们检查一些代码来完成这个任务:

```py
# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="opencv_logo.png",
	help="path to the input image")
args = vars(ap.parse_args())
```

**第 2-4 行**导入我们需要的 Python 包。然后我们在**的第 7-10 行解析我们的命令行参数。**

这里我们只需要一个参数，`--image`，它指向驻留在磁盘上的输入图像。

现在让我们加载这个图像，并把它分成各自的通道:

```py
# load the input image and grab each channel -- note how OpenCV
# represents images as NumPy arrays with channels in Blue, Green,
# Red ordering rather than Red, Green, Blue
image = cv2.imread(args["image"])
(B, G, R) = cv2.split(image)

# show each channel individually
cv2.imshow("Red", R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)
cv2.waitKey(0)
```

**第 15 行**从磁盘加载我们的`image`。然后我们在**第 16 行**上通过调用`cv2.split`将其分成红色、绿色和蓝色通道分量。

通常，我们会想到 RGB 色彩空间中的图像——红色像素第一，绿色像素第二，蓝色像素第三。但是，OpenCV 按照相反的通道顺序将 RGB 图像存储为 NumPy 数组。它不是以 RGB 顺序存储图像，而是以 BGR 顺序存储图像。因此，我们以相反的顺序解包元组。

**第 19-22 行**分别显示每个通道，如图**图 2** 所示。

我们还可以使用`cv2.merge`功能将通道重新合并在一起:

```py
# merge the image back together again
merged = cv2.merge([B, G, R])
cv2.imshow("Merged", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

我们只是简单地指定我们的频道，还是按照 BGR 的顺序，然后`cv2.merge`为我们处理剩下的事情( **Line 25** )！

请注意我们是如何从每个单独的 RGB 通道重建原始输入图像的:

还有一个*秒*的方法来可视化每个通道的颜色贡献。在**图 3 中，**我们简单地检查了图像的单通道表示，它看起来像一个灰度图像。

但是，我们也可以将图像的颜色贡献想象为完整的 RGB 图像，如下所示:

使用这种方法，我们可以用“彩色”而不是“灰度”来可视化每个通道这是一种严格的可视化技术，我们不会在标准的计算机视觉或图像处理应用程序中使用。

但是，让我们研究一下代码，看看如何构造这种表示:

```py
# visualize each channel in color
zeros = np.zeros(image.shape[:2], dtype="uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)
```

为了显示通道的实际“颜色”，我们首先需要使用`cv2.split`分解图像。我们需要重建图像，但是这一次，除了当前通道之外，所有像素都为零。

在**第 31 行，**我们构造了一个 NumPy 的零数组，宽度和高度与最初的`image`相同。

然后，为了构建图像的红色通道表示，我们调用`cv2.merge`，为绿色和蓝色通道指定我们的`zeros`数组。

我们对第 33 行和第 34 行的其他通道采取了类似的方法。

您可以参考**图 5** 来查看这段代码的输出可视化。

### **通道拆分和合并结果**

要使用 OpenCV 拆分和合并通道，请务必使用本教程的 ***【下载】*** 部分下载源代码。

让我们执行我们的`opencv_channels.py`脚本来分割每个单独的通道并可视化它们:

```py
$ python opencv_channels.py
```

您可以参考上一节来查看脚本的输出。

如果您希望向`opencv_channels.py`脚本提供不同的图像，您需要做的就是提供`--image`命令行参数:

```py
$ python opencv_channels.py --image adrian.png
```

在这里，您可以看到我们已经将输入图像分割为相应的红色、绿色和蓝色通道分量:

这是每个频道的第二个可视化效果:

## **总结**

在本教程中，您学习了如何使用 OpenCV 以及`cv2.split`和`cv2.merge`函数来分割和合并图像通道。

虽然有 NumPy 函数可以用于拆分和合并，但是我强烈建议您使用`cv2.split`和`cv2.merge`函数——从代码的角度来看，它们更容易阅读和理解。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****