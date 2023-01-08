# OpenCV 和 Python 颜色检测

> 原文：<https://pyimagesearch.com/2014/08/04/opencv-python-color-detection/>

最后更新于 2021 年 7 月 9 日。

所以，我来了。乘坐美国国家铁路客运公司 158 次列车，在一次漫长的商务旅行后回家。

天气很热。空调几乎不工作了。一个婴儿就在我旁边尖叫，而陪同的母亲孤独地看着窗外，显然质疑生孩子是否是正确的人生决定。

更糟糕的是， ***无线网络无法工作。***

幸运的是，我带来了我的游戏机和口袋妖怪游戏集。

当我把我那可靠的蓝色版本放进我的游戏机时，我对自己说，与其第一千次与小茂战斗，也许我可以做一点计算机视觉。

老实说，能够只用颜色来分割每一个游戏卡带*难道不是很酷吗？*

给自己拿一杯水来对抗失灵的空调，拿一副耳塞来挡住哭闹的孩子。因为在这篇文章中，我将向你展示如何使用 OpenCV 和 Python 来执行颜色检测。

*   【2021 年 7 月更新:增加了如何使用颜色匹配卡和直方图匹配提高颜色检测准确度的新部分。

**OpenCV 和 Python 版本:**
这个例子将运行在 **Python 2.7/Python 3.4+** 和 **OpenCV 2.4.X/OpenCV 3.0+** 上。

# OpenCV 和 Python 颜色检测

让我们开始吧。

打开您最喜欢的编辑器，创建一个名为`detect_color.py`的文件:

```py
# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

```

我们将从在**2-4 号线**进口必要的包装开始。我们将使用 NumPy 进行数值处理，`argparse`解析我们的命令行参数，而`cv2`用于 OpenCV 绑定。

**第 7-9 行**然后解析我们的命令行参数。我们只需要一个开关`--image`，它是我们的映像驻留在磁盘上的路径。

然后，在**第 12 行**上，我们从磁盘上加载我们的映像。

现在，有趣的部分来了。

我们希望能够检测图像中的每一个 Game Boy 墨盒。这意味着我们必须识别图像中的**红色**、**蓝色**、**黄色**和**灰色**颜色。

让我们继续定义这个颜色列表:

```py
# define the list of boundaries
boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]

```

我们在这里所做的是在 RGB 颜色空间(或者更确切地说，BGR，因为 OpenCV 以逆序将图像表示为 NumPy 数组)中定义一个列表`boundaries`，其中列表中的每个条目都是一个具有两个值的元组:一个列表是下限*的*限制，一个列表是上限*的*限制。

例如，我们来看看元组`([17, 15, 100], [50, 56, 200])`。

这里，我们说的是，我们图像中所有具有 *R > = 100* 、 *B > = 15* 、 *G > = 17* 以及 *R < = 200* 、 *B < = 56* 和 *G < = 50* 的像素将被视为**红色**。

现在我们有了边界列表，我们可以使用`cv2.inRange`函数来执行实际的颜色检测。

让我们来看看:

```py
# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)

	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)

```

我们开始在第 23 行的**上循环我们的上下`boundaries`，然后在第 25 和 26** 行的**上将上下限值转换成 NumPy 数组。这两行看起来可以省略，但是当你使用 OpenCV Python 绑定时，OpenCV *期望*这些限制是 NumPy 数组。此外，由于这些是落在范围*【0，256】*内的像素值，我们可以使用无符号 8 位整数数据类型。**

要使用 OpenCV 执行实际的颜色检测，请看一下**第 29 行**，这里我们使用了`cv2.inRange`函数。

`cv2.inRange`函数需要三个参数:第一个是我们将要执行颜色检测的`image`，第二个是您想要检测的颜色的`lower`极限，第三个参数是您想要检测的颜色的`upper`极限。

调用`cv2.inRange`后，返回一个二进制掩码，其中白色像素(255)代表落入上下限范围内的像素，黑色像素(0)不属于。

*注意:我们正在 RGB 颜色空间中执行颜色检测。但是您也可以在 HSV 或 L*a*b*色彩空间中轻松做到这一点。你只需要调整你的上限和下限到各自的颜色空间。*

为了创建输出图像，我们在第 31 行应用蒙版。这一行简单地调用了`cv2.bitwise_and`，只显示了`image`中在`mask`中有相应白色(255)值的像素。

最后，我们的输出图像显示在**行 34 和 35** 上。

还不错。只有 35 行代码，绝大多数是导入、参数解析和注释。

让我们继续运行我们的脚本:

```py
$ python detect_color.py --image pokemon_games.png

```

如果您的环境配置正确(意味着您安装了带有 Python 绑定的 OpenCV)，您应该会看到如下输出图像:

如你所见，红色口袋妖怪子弹很容易被发现！

现在让我们试试蓝色的:

不，没问题！

黄色版本也有类似的故事:

最后，还发现了灰色 Game Boy 墨盒的轮廓:

# 通过色彩校正提高色彩检测的准确性

在本教程中，您学习了如何通过硬编码较低和较高的 RGB 颜色范围来执行颜色校正。

假设给你一个 1000 张图像的数据集，要求你找出 RGB 值分别在 *(17，15，100)* 和 *(50，56，200)*范围内的所有“红色”对象。

如果您的整个图像数据集是在受控的照明条件下拍摄的，并且每张图像都使用了*相同的照明*，那么这不会是一项困难的任务——您可以使用上面提到的硬编码 RGB 值。

但是…假设你的图像数据集*不是在受控的光线条件下拍摄的*。有些是用荧光灯拍摄的，有些是在阳光明媚的时候在户外拍摄的，有些是在黑暗沉闷的时候拍摄的。

[这里的重点是照明条件对输出像素值有巨大影响](https://pyimagesearch.com/2021/04/28/opencv-color-spaces-cv2-cvtcolor/)。

在不同的光照下，颜色看起来会有很大的不同，当这种情况发生时，您硬编码的较低和较高 RGB 范围将会失败。

一个潜在的解决方案是使用不同的颜色空间，这可以更好地模仿人类感知颜色的方式 HSV 和 L*a*b*颜色空间是很好的选择。

更好的选择是使用**颜色校正卡**。你将一张卡片(就像上面**图 5** 中的那张)放在包含我们正在捕捉的物体的场景中，然后你通过以下方式对所有这些图像进行后处理:

1.  检测颜色校正卡
2.  确定色块区域
3.  执行直方图匹配以将色彩空间从一幅图像转移到另一幅图像

通过这种方式，您可以确保所有图像的颜色一致，即使它们可能是在不同的光照条件下拍摄的。

要了解更多关于这种技术的信息，我建议阅读以下两个教程:

1.  [*直方图匹配用 OpenCV、scikit-image、Python*](https://pyimagesearch.com/2021/02/08/histogram-matching-with-opencv-scikit-image-and-python/)
2.  [*用 OpenCV 和 Python 自动校色*](https://pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/)

# 摘要

在这篇博文中，我展示了如何使用 OpenCV 和 Python 进行颜色检测。

要检测图像中的颜色，你需要做的第一件事就是为你的像素值定义上*上限*和下*下限*。

一旦定义了上限和下限，然后调用`cv2.inRange`方法返回一个掩码，指定哪些像素落在指定的上限和下限范围内。

最后，现在你有了蒙版，你可以用`cv2.bitwise_and`函数把它应用到你的图像上。

我的火车离家只有几站，所以我最好把这个帖子包起来。希望你觉得有用！

如果你有任何问题，一如既往，欢迎留言或给我发消息。