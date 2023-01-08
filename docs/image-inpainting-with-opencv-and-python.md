# 基于 OpenCV 和 Python 的图像修复

> 原文：<https://pyimagesearch.com/2020/05/18/image-inpainting-with-opencv-and-python/>

在本教程中，你将学习如何使用 OpenCV 和 Python 进行图像修复。

图像修复是**图像保护**和**图像修复**的一种形式，可以追溯到 18 世纪，当时意大利威尼斯公共图片修复主管[皮埃特罗·爱德华兹](https://it.wikipedia.org/wiki/Pietro_Edwards)应用这种科学方法修复和保护著名作品([来源](https://en.wikipedia.org/wiki/Inpainting))。

技术极大地促进了图像绘画，使我们能够:

*   恢复旧的、退化的照片
*   修复因损坏和老化而缺失区域的照片
*   从图像中遮罩并移除特定的对象(以一种美学上令人愉悦的方式)

今天，我们将看看 OpenCV“开箱即用”的两种图像修复算法

**学习如何使用 OpenCV 和 Python 进行图像修复，*继续阅读！***

## **利用 OpenCV 和 Python 进行图像修复**

在本教程的第一部分，你将学习 OpenCV 的修复算法。

从那里，我们将使用 OpenCV 的内置算法实现一个修复演示，然后将修复应用于一组图像。

最后，我们将查看结果并讨论后续步骤。

我也将坦率地说，这个教程是一个*介绍*修复包括它的基础知识，它是如何工作的，以及我们可以期待什么样的结果。

虽然本教程不一定在修复结果方面“开辟新天地”，但它是未来教程的必要先决条件，因为:

1.  它向您展示了如何使用 OpenCV 进行修复
2.  它为您提供了一个我们可以改进的基线
3.  它显示了传统修复算法所需的一些*手动输入*，深度学习方法现在可以*自动化*

### **OpenCV 的修复算法**

引用 [OpenCV 文档，](https://docs.opencv.org/master/df/d3d/tutorial_py_inpainting.html)Telea 方法:

> *…基于快速行进法。考虑图像中要修补的区域。算法从该区域的边界开始，进入该区域，首先逐渐填充边界中的所有内容。需要在要修复的邻域的像素周围有一个小的邻域。该像素被邻域中所有已知像素的归一化加权和所代替。砝码的选择是一件重要的事情。对那些靠近该点、靠近边界法线的像素和那些位于边界轮廓上的像素给予更大的权重。一旦像素被修复，它就使用快速行进方法移动到下一个最近的像素。FMM 确保已知像素附近的像素首先被修复，因此它就像一个手动启发式操作。*

第二种方法，纳维尔-斯托克斯方法，是基于流体动力学。

再次引用 OpenCV 文档:

> 该算法基于流体动力学并利用偏微分方程。基本原理是启发式的。它首先沿着边缘从已知区域行进到未知区域(因为边缘应该是连续的)。它继续等照度线(连接具有相同强度的点的线，就像轮廓连接具有相同海拔的点)，同时在修复区域的边界匹配梯度向量。为此，使用了流体动力学的一些方法。一旦它们被获得，颜色被填充以减少该区域的最小变化。

在本教程的剩余部分，您将学习如何使用 OpenCV 应用`cv2.INPAINT_TELEA`和`cv2.INPAINT_NS`方法。

### **如何使用 OpenCV 进行修复？**

当用 OpenCV 应用修补时，**我们需要提供两幅图像:**

1.  **我们希望修复和恢复的输入图像。**推测起来，这个图像以某种方式被“损坏”,我们需要应用修复算法来修复它
2.  **掩模图像，指示*图像中*损伤的位置。**该图像应具有与输入图像相同的空间尺寸(宽度和高度)。非零像素对应于应该修补(即，修复)的区域，而零像素被认为是“正常的”并且不需要修补

这些图像的示例可以在上面的**图 2** 中看到。

左边的*上的图像是我们的原始输入图像。请注意这张图片是如何陈旧、褪色、破损/撕裂的。*

右边*的图像是我们的蒙版图像。注意蒙版中的白色像素是如何标记输入图像中的损坏位置*(左图)。**

最后，在底部的*，*我们得到了用 OpenCV 修补后的输出图像。我们陈旧、褪色、受损的形象现在已经部分恢复了。

### **我们如何用 OpenCV 创建用于修补的蒙版？**

此时，最大的问题是:

> *“阿德里安，你是怎么创造出面具的？这是通过编程创建的吗？还是手动创建的？”*

对于上面的**图 2** (上一节)，我不得不*手动*创建蒙版。为此，我打开了 Photoshop ( [GIMP](https://www.gimp.org/) 或其他照片编辑/处理工具也可以)，然后使用*魔棒*工具和手动选择工具来选择图像的受损区域。

然后我用白色的*填充选区，将背景设为黑色，并将蒙版保存到磁盘。*

**这样做是一个手动的、繁琐的过程**——你*也许*能够使用图像处理技术，如阈值处理、边缘检测和轮廓来标记损坏的原因，以编程方式为你自己的图像定义蒙版，但实际上，可能会有某种人工干预。

**手动干预是使用 OpenCV 内置修复算法的*主要限制*之一。**

我在*“我们如何改进 OpenCV 修复结果？”中讨论了我们如何改进 OpenCV 的修复算法，包括基于深度学习的方法*一节。

### **项目结构**

滚动到本教程的 ***“下载”*** 部分，获取包含我们的代码和图像的`.zip`。这些文件组织如下:

```py
$ tree --dirsfirst
.
├── examples
│   ├── example01.png
│   ├── example02.png
│   ├── example03.png
│   ├── mask01.png
│   ├── mask02.png
│   └── mask03.png
└── opencv_inpainting.py

1 directory, 7 files
```

### **用 OpenCV 和 Python 实现修复**

```py
# import the necessary packages
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path input image on which we'll perform inpainting")
ap.add_argument("-m", "--mask", type=str, required=True,
	help="path input mask which corresponds to damaged areas")
ap.add_argument("-a", "--method", type=str, default="telea",
	choices=["telea", "ns"],
	help="inpainting algorithm to use")
ap.add_argument("-r", "--radius", type=int, default=3,
	help="inpainting radius")
args = vars(ap.parse_args())
```

```py
# initialize the inpainting algorithm to be the Telea et al. method
flags = cv2.INPAINT_TELEA

# check to see if we should be using the Navier-Stokes (i.e., Bertalmio
# et al.) method for inpainting
if args["method"] == "ns":
	flags = cv2.INPAINT_NS
```

```py
# load the (1) input image (i.e., the image we're going to perform
# inpainting on) and (2) the  mask which should have the same input
# dimensions as the input image -- zero pixels correspond to areas
# that *will not* be inpainted while non-zero pixels correspond to
# "damaged" areas that inpainting will try to correct
image = cv2.imread(args["image"])
mask = cv2.imread(args["mask"])
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
```

```py
# perform inpainting using OpenCV
output = cv2.inpaint(image, mask, args["radius"], flags=flags)
```

```py
# show the original input image, mask, and output image after
# applying inpainting
cv2.imshow("Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Output", output)
cv2.waitKey(0)
```

我们在屏幕上显示三幅图像:(1)我们的原始*受损*照片，(2)我们的`mask`突出显示受损区域，以及(3)修复(即修复)`output`照片。这些图像中的每一个都将保留在您的屏幕上，直到当其中一个 GUI 窗口成为焦点时按下任何键。

### **OpenCV 修复结果**

我们现在准备使用 OpenCV 应用修复。

确保您已经使用本教程的 ***【下载】*** 部分下载了源代码和示例图像。

从那里，打开一个终端，并执行以下命令:

```py
$ python opencv_inpainting.py --image examples/example01.png \
	--mask examples/mask01.png
```

在左边的*，*你可以看到我的狗 Janie 的原始输入图像，穿着一件超级朋克/ska 牛仔夹克。

我特意*给图片添加了文字*【阿德里安·乌兹在此】*，图片的蒙版显示在中间的*。**

```py
$ python opencv_inpainting.py --image examples/example02.png \
	--mask examples/mask02.png --method ns
```

在顶部，你可以看到一张旧照片，已经损坏。然后我手动为右边*的受损区域创建了一个蒙版(使用 Photoshop，如*“我们如何使用 OpenCV 创建修复蒙版？”*节)。*

*底部*显示了纳维尔-斯托克斯修复方法的输出。通过应用这种 OpenCV 修复方法，我们已经能够部分修复旧的、损坏的照片。

让我们尝试最后一张图片:

```py
$ python opencv_inpainting.py --image examples/example03.png \
	--mask examples/mask03.png
```

在左边的*，*我们有原始图像，而在右边的*，*我们有相应的遮罩。

请注意，该遮罩有两个我们将尝试“修复”的区域:

1.  右下角的水印
2.  圆形区域对应于其中一棵树

在这个例子中，我们将 OpenCV 修复视为一种从图像中移除对象的方法，其结果可以在 T2 底部看到。

不幸的是，结果没有我们希望的那么好。我们希望移除的树显示为圆形模糊，而水印也是模糊的。

这就引出了一个问题— ***我们能做些什么来改善我们的结果呢？***

### **如何提高 OpenCV 修复效果？**

OpenCV 内置修复算法的最大问题之一是它们需要*手动干预*，这意味着我们必须手动提供我们希望修复和恢复的被遮罩区域。

**手动供应面膜繁琐— *没有更好的方法吗？***

其实是有的。

使用基于深度学习的方法，包括完全卷积神经网络和生成对抗网络(GANs)，我们可以“学习修复”。

这些网络:

*   要求*零手动干预*
*   *可以生成自己的训练数据*
*   生成比传统计算机视觉修复算法更美观的结果

基于深度学习的修复算法超出了本教程的范围，但将在未来的博客文章中介绍。

## **总结**

这些方法是*传统的*计算机视觉算法，不依赖于深度学习，使它们易于高效利用。

然而，尽管这些算法很容易使用(因为它们是内置在 OpenCV 中的)，但它们在准确性方面还有很多不足之处。

更不用说，必须手动*提供蒙版图像，标记原始照片的损坏区域，这相当繁琐。*

在未来的教程中，我们将研究基于深度学习的修复算法——这些方法需要更多的计算，编码有点困难，但最终会产生更好的结果(另外，没有遮罩图像要求)。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****