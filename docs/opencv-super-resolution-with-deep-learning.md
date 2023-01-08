# 具有深度学习的 OpenCV 超分辨率

> 原文：<https://pyimagesearch.com/2020/11/09/opencv-super-resolution-with-deep-learning/>

在本教程中，您将学习如何使用 OpenCV 和深度学习在图像和实时视频流中执行超分辨率。

今天这篇博文的灵感来自于我收到的一封来自 PyImageSearch 阅读器 Hisham 的邮件:

> *"Hello Adrian, I have read your [book](https://pyimagesearch.com/deep-learning-computer-vision-python-book/) on deep learning of computer vision written in Python, and I have browsed the super-resolution you achieved with Keras and TensorFlow. Very useful, thank you.*
> 
> *I wonder:*
> 
> *Is there a pre-trained super-resolution model compatible with the `dnn` module of OpenCV?*
> 
> Can they work in real time?
> 
> If you have any suggestions, it will be a great help. "

你很幸运，希沙姆——这里有*个*超分辨率深度神经网络，它们都是:

1.  预先训练(意味着你不必自己在数据集上训练他们)
2.  与 OpenCV 兼容

**然而 OpenCV 的超分辨率功能，其实是“隐藏”在一个名为** `DnnSuperResImpl_create` **的晦涩函数中的一个名为** `dnn_superres` **的子模块里。**

这个函数需要一些解释才能使用，所以我决定写一个关于它的教程；这样每个人都可以学习如何使用 OpenCV 的超分辨率功能。

本教程结束时，你将能够使用 OpenCV 在*图像*和*实时视频流中执行超分辨率处理！*

**要了解如何使用 OpenCV 进行基于深度学习的超分辨率，*继续阅读。***

## 具有深度学习的 OpenCV 超分辨率

在本教程的第一部分，我们将讨论:

*   什么是超分辨率
*   为什么我们不能使用简单的最近邻、线性或双三次插值来大幅提高图像的分辨率
*   专业化的深度学习架构如何帮助我们实时实现超分辨率

接下来，我将向您展示如何使用这两种方法实现 OpenCV 超分辨率:

1.  形象
2.  实时视频分辨率

我们将讨论我们的结果来结束本教程。

### 什么是超分辨率？

超分辨率包含一组用于增强、增加和上采样输入图像分辨率的算法和技术。**更简单地说，取一幅输入图像，增加图像的宽度和高度，而质量下降最小(理想情况下为零)。**

说起来容易做起来难。

任何曾经在 Photoshop 或 GIMP 中打开一个小图像，然后试图调整其大小的人都知道，输出的图像最终看起来像素化了。

那是因为 Photoshop，GIMP，Image Magick，OpenCV(通过`cv2.resize`函数)等。所有都使用经典的插值技术和算法(例如最近邻插值、线性插值、双三次插值)来增加图像分辨率。

这些函数的“工作”方式是，呈现一个输入图像，调整图像的大小，然后将调整后的图像返回给调用函数…

**…然而，如果将空间维度*、*增加得太多，那么输出图像会出现像素化，会有伪像，总的来说，对人眼来说看起来“不美观”。**

例如，让我们考虑下图:

**在*顶端*我们有自己的原始图像。**红色矩形中突出显示的区域是我们希望提取并提高分辨率的区域(即，在不降低图像补片质量的情况下，调整到更大的宽度和高度)。

**在*底部*我们有应用双三次插值的输出，**用于增加输入图像大小的标准插值方法(当需要增加输入图像的空间维度时，我们通常在`cv2.resize`中使用)。

然而，花一点时间来注意在应用双三次插值后，图像补丁是如何像素化、模糊和不可读的。

这就提出了一个问题:

有没有更好的方法在不降低质量的情况下提高图像的分辨率？

答案是肯定的——而且这也不是魔法。通过应用新颖的深度学习架构，我们能够生成没有这些伪像的高分辨率图像:

同样，在*顶部*我们有我们的原始输入图像。在*中间*我们在应用双三次插值后有低质量的尺寸调整。**在*底部*我们有应用我们的超分辨率深度学习模型的输出。**

区别就像白天和黑夜。输出的深度神经网络超分辨率模型是清晰的，易于阅读，并且显示出*最小的*调整大小伪像的迹象。

在本教程的剩余部分，我将揭开这个“魔术”,并向您展示如何使用 OpenCV 执行超分辨率！

### OpenCV 超分辨率模型

在本教程中，我们将使用四个预先训练好的超分辨率模型。对模型架构、它们如何工作以及每个模型的培训过程的回顾超出了本指南的范围(因为我们只关注实现*和*)。

如果您想了解更多关于这些模型的信息，我在下面列出了它们的名称、实现和论文链接:

*   ***EDSR:** [单幅图像超分辨率增强深度残差网络](https://arxiv.org/abs/1707.02921) ( [实现](https://github.com/Saafke/EDSR_Tensorflow) )*
*   **ESPCN:** *[利用高效的亚像素卷积神经网络](https://arxiv.org/abs/1609.05158)* ( [实现](https://github.com/fannymonori/TF-ESPCN))实时单幅图像和视频超分辨率
*   **FSRCNN:** *[加速超分辨率卷积神经网络](https://arxiv.org/abs/1608.00367)* ( [实现](https://github.com/Saafke/FSRCNN_Tensorflow))
*   **LapSRN:** *[快速准确的图像超分辨率与深度拉普拉斯金字塔网络](https://arxiv.org/abs/1710.01992)* ( [实现](https://github.com/fannymonori/TF-LAPSRN))

非常感谢来自 BleedAI 的 Taha Anwar 整理了他的关于 OpenCV 超分辨率的指南，其中收集了很多信息——这对创作这篇文章非常有帮助。

### 使用 OpenCV 为超分辨率配置您的开发环境

**为了应用 OpenCV 超分辨率，您*必须*在您的系统上安装 OpenCV 4.3(或更高版本)。**虽然在 OpenCV 4.1.2 中用 C++实现了`dnn_superes`模块，但是 Python 绑定直到 OpenCV 4.3 才实现。

幸运的是，OpenCV 4.3+是 pip 安装的:

```py
$ pip install opencv-contrib-python
```

**如果你需要帮助配置 OpenCV 4.3+的开发环境，我*强烈推荐*阅读我的** ***[pip 安装 OpenCV 指南](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)***——它将在几分钟内让你启动并运行。

### 配置您的开发环境有问题吗？

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在你的 Windows、macOS 或 Linux 系统上运行代码*了吗？***

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### 项目结构

配置好开发环境后，让我们继续检查我们的项目目录结构:

```py
$ tree . --dirsfirst
.
├── examples
│   ├── adrian.png
│   ├── butterfly.png
│   ├── jurassic_park.png
│   └── zebra.png
├── models
│   ├── EDSR_x4.pb
│   ├── ESPCN_x4.pb
│   ├── FSRCNN_x3.pb
│   └── LapSRN_x8.pb
├── super_res_image.py
└── super_res_video.py

2 directories, 10 files
```

在这里，您可以看到我们今天要复习两个 Python 脚本:

1.  ``super_res_image.py`` :对从磁盘加载的图像执行 OpenCV 超分辨率
2.  ``super_res_video.py`` :将 OpenCV 的超分辨率应用于实时视频流

我们将在本文后面详细介绍这两个 Python 脚本的实现。

从那里，我们有四个超分辨率模型:

1.  ``EDSR_x4.pb`` :来自*单幅图像超分辨率增强深度残差网络*论文的模型— **将输入图像分辨率提高 4 倍**
2.  ``ESPCN_x4.pb`` :来自*的超分辨率模型，采用高效的亚像素卷积神经网络实现实时单幅图像和视频超分辨率*——**分辨率提升 4 倍**
3.  `FSRCNN_x3.pb`:来自*的模型加速超分辨率卷积神经网络*——**将图像分辨率提高 3 倍**
4.  ``LapSRN_x8.pb`` :来自*的超分辨率模型，深度拉普拉斯金字塔网络快速精确的图像超分辨率*——**将图像分辨率提高 8 倍**

最后，`examples`目录包含我们将应用 OpenCV 超分辨率的示例输入图像。

### 用图像实现 OpenCV 超分辨率

我们现在准备在图像中实现 OpenCV 超分辨率！

打开项目目录结构中的`super_res_image.py`文件，让我们开始工作:

```py
# import the necessary packages
import argparse
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to super resolution model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image we want to increase resolution of")
args = vars(ap.parse_args())
```

从那里，**第 8-13 行**解析我们的命令行参数。这里我们只需要两个命令行参数:

1.  `--model`:输入 OpenCV 超分辨率模型的路径
2.  ``--image`` :我们要应用超分辨率的输入图像的路径

给定我们的超分辨率模型路径，我们现在需要提取**模型名称**和**模型比例**(即，我们将增加图像分辨率的因子):

```py
# extract the model name and model scale from the file path
modelName = args["model"].split(os.path.sep)[-1].split("_")[0].lower()
modelScale = args["model"].split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])
```

**第 16 行**提取`modelName`，分别可以是`EDSR`、`ESPCN`、`FSRCNN`或`LapSRN`。**`modelName`***有*是这些型号名称之一；否则，** `dnn_superres` **模块和** `DnnSuperResImpl_create` **功能将不起作用。****

 **解析完模型名称和比例后，我们现在可以继续加载 OpenCV 超分辨率模型:

```py
# initialize OpenCV's super resolution DNN object, load the super
# resolution model from disk, and set the model name and scale
print("[INFO] loading super resolution model: {}".format(
	args["model"]))
print("[INFO] model name: {}".format(modelName))
print("[INFO] model scale: {}".format(modelScale))
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(args["model"])
sr.setModel(modelName, modelScale)
```

我们首先实例化`DnnSuperResImpl_create`的一个实例，这是我们实际的超分辨率对象。

对`readModel`的调用从磁盘加载我们的 OpenCV 超分辨率模型。

然后我们必须明确地调用`setModel`到*来设置`modelName`和`modelScale`。*

 *未能从磁盘读取模型或设置模型名称和比例将导致我们的超分辨率脚本出错或 segfaulting。

现在让我们用 OpenCV 执行超分辨率:

```py
# load the input image from disk and display its spatial dimensions
image = cv2.imread(args["image"])
print("[INFO] w: {}, h: {}".format(image.shape[1], image.shape[0]))

# use the super resolution model to upscale the image, timing how
# long it takes
start = time.time()
upscaled = sr.upsample(image)
end = time.time()
print("[INFO] super resolution took {:.6f} seconds".format(
	end - start))

# show the spatial dimensions of the super resolution image
print("[INFO] w: {}, h: {}".format(upscaled.shape[1],
	upscaled.shape[0]))
```

**第 31 和 32 行**从磁盘加载我们的输入`--image`并显示*原始*的宽度和高度。

从那里，**线 37** 调用`sr.upsample`，提供原始输入`image`。`upsample`函数，顾名思义，执行 OpenCV 超分辨率模型的前向传递，返回`upscaled`图像。

我们小心地测量超分辨率过程需要多长时间，然后在我们的终端上显示升级图像的新宽度和高度。

为了比较，让我们应用标准的双三次插值并计算它需要多长时间:

```py
# resize the image using standard bicubic interpolation
start = time.time()
bicubic = cv2.resize(image, (upscaled.shape[1], upscaled.shape[0]),
	interpolation=cv2.INTER_CUBIC)
end = time.time()
print("[INFO] bicubic interpolation took {:.6f} seconds".format(
	end - start))
```

双三次插值是用于提高图像分辨率的标准算法。这种方法在几乎*个*图像处理工具和库中实现，包括 Photoshop、GIMP、Image Magick、PIL/PIllow、OpenCV、Microsoft Word、Google Docs 等。— **如果一个软件需要处理图像，它*很可能*实现双三次插值。**

最后，让我们在屏幕上显示输出结果:

```py
# show the original input image, bicubic interpolation image, and
# super resolution deep learning output
cv2.imshow("Original", image)
cv2.imshow("Bicubic", bicubic)
cv2.imshow("Super Resolution", upscaled)
cv2.waitKey(0)
```

在这里，我们显示我们的原始输入`image`，调整后的图像`bicubic`，最后是我们的`upscaled`超分辨率图像。

我们将三个结果显示在屏幕上，这样我们可以很容易地比较结果。

### OpenCV 超分辨率结果

首先，确保您已经使用本教程的 ***“下载”*** 部分下载了源代码、示例图像和预训练的超分辨率模型。

从那里，打开一个终端，并执行以下命令:

```py
$ python super_res_image.py --model models/EDSR_x4.pb --image examples/adrian.png
[INFO] loading super resolution model: models/EDSR_x4.pb
[INFO] model name: edsr
[INFO] model scale: 4
[INFO] w: 100, h: 100
[INFO] super resolution took 1.183802 seconds
[INFO] w: 400, h: 400
[INFO] bicubic interpolation took 0.000565 seconds
```

在*顶部*我们有我们的原始输入图像。在*中间*我们应用了标准的双三次插值图像来增加图像的尺寸。最后，*底部*显示了 EDSR 超分辨率模型的输出(图像尺寸增加了 4 倍)。

如果你研究这两幅图像，你会发现超分辨率图像看起来“更平滑”特别是，看看我的前额区域。在双三次图像中，有很多像素化在进行— **,但是在超分辨率图像中，我的前额明显更平滑，像素化更少。**

EDSR 超分辨率模式的缺点是速度有点慢。标准的双三次插值可以以每秒 *> 1700* 帧的速率将 100×100 像素的图像增加到 400×400 像素。

另一方面，EDSR 需要一秒以上的时间来执行相同的上采样。因此，EDSR 不适合实时超分辨率(至少在没有 GPU 的情况下不适合)。

***注意:**这里所有的计时都是用 3 GHz 英特尔至强 W 处理器收集的。一个 GPU 被**而不是**使用。*

让我们试试另一张照片，这张是一只蝴蝶:

```py
$ python super_res_image.py --model models/ESPCN_x4.pb --image examples/butterfly.png
[INFO] loading super resolution model: models/ESPCN_x4.pb
[INFO] model name: espcn
[INFO] model scale: 4
[INFO] w: 400, h: 240
[INFO] super resolution took 0.073628 seconds
[INFO] w: 1600, h: 960
[INFO] bicubic interpolation took 0.000833 seconds
```

同样，在*顶部*我们有我们的原始输入图像。在应用标准的双三次插值后，我们得到了中间的*图像。在*底部*我们有应用 ESPCN 超分辨率模型的输出。*

你可以看到这两个超分辨率模型之间的差异的最好方法是研究蝴蝶的翅膀。请注意双三次插值方法看起来更加嘈杂和扭曲，而 ESPCN 输出图像明显更加平滑。

这里的好消息是，ESPCN 模型的速度明显快于 T1，能够在 CPU 上以 13 FPS 的速率将 400x240px 的图像上采样为 1600x960px 的图像。

下一个示例应用 FSRCNN 超分辨率模型:

```py
$ python super_res_image.py --model models/FSRCNN_x3.pb --image examples/jurassic_park.png
[INFO] loading super resolution model: models/FSRCNN_x3.pb
[INFO] model name: fsrcnn
[INFO] model scale: 3
[INFO] w: 350, h: 197
[INFO] super resolution took 0.082049 seconds
[INFO] w: 1050, h: 591
[INFO] bicubic interpolation took 0.001485 seconds
```

暂停一下，看看艾伦·格兰特的夹克(穿蓝色牛仔衬衫的那个人)。在双三次插值图像中，这件衬衫是颗粒状的。但是在 FSRCNN 输出中，封套要比 T1 平滑得多。

与 ESPCN 超分辨率模型类似，FSRCNN 仅用 0.08 秒对图像进行上采样(速率约为 12 FPS)。

最后，让我们看看 LapSRN 模型，它将输入图像分辨率提高了 8 倍:

```py
$ python super_res_image.py --model models/LapSRN_x8.pb --image examples/zebra.png
[INFO] loading super resolution model: models/LapSRN_x8.pb
[INFO] model name: lapsrn
[INFO] model scale: 8
[INFO] w: 400, h: 267
[INFO] super resolution took 4.759974 seconds
[INFO] w: 3200, h: 2136
[INFO] bicubic interpolation took 0.008516 seconds
```

也许不出所料，这种型号是最慢的，需要 4.5 秒才能将 400x267px 输入的分辨率提高到 3200x2136px 的输出。鉴于我们将空间分辨率提高了 8 倍，这一时序结果是有意义的。

也就是说，LapSRN 超分辨率模型的输出*非常棒。*看双三次插值输出*(中)*和 LapSRN 输出*(下)之间的斑马纹。*斑马身上的条纹清晰分明，不像双三次输出。

### 用 OpenCV 实现实时超分辨率

我们已经看到超分辨率应用于单个图像——**,但如何处理*实时*视频流呢？**

有可能实时执行 OpenCV 超分辨率吗？

答案是肯定的，这绝对是*可能的*——这正是我们的`super_res_video.py`脚本所做的。

***注:***`super_res_video.py`*的大部分脚本与我们的* `super_res_image.py` *脚本类似，所以我就不多花时间解释实时实现了。如果您需要更多的帮助来理解代码，请参考上一节“用图像实现 OpenCV 超分辨率”。*

让我们开始吧:

```py
# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to super resolution model")
args = vars(ap.parse_args())
```

**第 2-7 行**导入我们需要的 Python 包。除了我的 [imutils 库](https://github.com/jrosebr1/imutils)和来自它的 [VideoStream 实现之外，这些都与我们之前关于图像超分辨率的脚本几乎相同。](https://pyimagesearch.com/2016/01/04/unifying-picamera-and-cv2-videocapture-into-a-single-class-with-opencv/)

然后我们解析我们的命令行参数。只需要一个参数`--model`，它是我们的输入超分辨率模型的路径。

接下来，让我们提取模型名称和模型比例，然后从磁盘加载我们的 OpenCV 超分辨率模型:

```py
# extract the model name and model scale from the file path
modelName = args["model"].split(os.path.sep)[-1].split("_")[0].lower()
modelScale = args["model"].split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])

# initialize OpenCV's super resolution DNN object, load the super
# resolution model from disk, and set the model name and scale
print("[INFO] loading super resolution model: {}".format(
	args["model"]))
print("[INFO] model name: {}".format(modelName))
print("[INFO] model scale: {}".format(modelScale))
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(args["model"])
sr.setModel(modelName, modelScale)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
```

**第 16-18 行**从输入的`--model`文件路径中提取我们的`modelName`和`modelScale`。

使用这些信息，我们实例化我们的超分辨率(`sr`)对象，从磁盘加载模型，并设置模型名称和比例(**第 26-28 行**)。

然后我们初始化我们的`VideoStream`(这样我们可以从我们的网络摄像头读取帧)并允许摄像头传感器预热。

完成初始化后，我们现在可以从`VideoStream`开始循环遍历帧:

```py
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 300 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=300)

	# upscale the frame using the super resolution model and then
	# bicubic interpolation (so we can visually compare the two)
	upscaled = sr.upsample(frame)
	bicubic = cv2.resize(frame,
		(upscaled.shape[1], upscaled.shape[0]),
		interpolation=cv2.INTER_CUBIC)
```

**第 36 行**开始循环播放视频流中的帧。然后我们抓取下一个`frame`,调整它的宽度为 300 像素。

我们执行这个调整大小操作是为了可视化/示例的目的。回想一下，本教程的重点是用 OpenCV 应用超分辨率。因此，我们的示例应该显示如何获取低分辨率输入，然后生成高分辨率输出(这正是我们降低帧分辨率的原因)。

**第 44 行**使用我们的 OpenCV 分辨率模型调整输入`frame`的大小，得到`upscaled`图像。

**第 45-47 行**应用了基本的双三次插值，因此我们可以比较这两种方法。

我们的最后一个代码块将结果显示在屏幕上:

```py
# show the original frame, bicubic interpolation frame, and super
	# resolution frame

	cv2.imshow("Original", frame)
	cv2.imshow("Bicubic", bicubic)
	cv2.imshow("Super Resolution", upscaled)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
```

这里我们显示原始的`frame`、`bicubic`插值输出，以及来自我们的超分辨率模型的`upscaled`输出。

我们继续处理并在屏幕上显示帧，直到点击 OpenCV 打开的窗口并按下`q`,导致 Python 脚本退出。

最后，我们通过关闭 OpenCV 打开的所有窗口并停止我们的视频流来执行一点清理。

### 实时 OpenCV 超分辨率结果

现在让我们在实时视频流中应用 OpenCV 超级分辨率！

确保您已经使用本教程的 ***“下载”*** 部分下载了源代码、示例图像和预训练模型。

从那里，您可以打开一个终端并执行以下命令:

```py
$ python super_res_video.py --model models/FSRCNN_x3.pb
[INFO] loading super resolution model: models/FSRCNN_x3.pb
[INFO] model name: fsrcnn
[INFO] model scale: 3
[INFO] starting video stream...
```

在这里，您可以看到我能够在我的 CPU 上实时运行 FSRCNN 模型(不需要 GPU！).

此外，如果您将双三次插值的结果与超分辨率进行比较，您会发现超分辨率输出要干净得多。

### 建议

在一篇篇幅有限的博客文章中，很难展示超分辨率带给我们的所有微妙之处，因此我*强烈建议*下载代码/模型，并仔细研究输出。

## 摘要

在本教程中，您学习了如何在图像和实时视频流中实现 OpenCV 超分辨率。

诸如最近邻插值、线性插值和双三次插值之类的基本图像大小调整算法只能将输入图像的分辨率提高到一定程度，之后，图像质量会下降到图像看起来像素化的程度，并且一般来说，调整后的图像对人眼来说不美观。

深度学习超分辨率模型能够产生这些更高分辨率的图像，同时有助于防止这些像素化、伪像和令人不愉快的结果。

也就是说，你需要设定一个期望值，即不存在你在电视/电影中看到的神奇算法，它可以拍摄一张模糊的、缩略图大小的图像，并将其调整为海报，你可以打印出来并挂在墙上——这根本不可能。

也就是说，OpenCV 的超分辨率模块*可以*用于应用超分辨率。这是否适合您的管道是需要测试的:

1.  首先尝试使用`cv2.resize`和标准插值算法(以及调整大小需要多长时间)。
2.  然后，运行相同的操作，但是换成 OpenCV 的超分辨率模块(同样，计时调整大小需要多长时间)。

比较*输出*和*运行标准插值和 OpenCV 超分辨率花费的时间*。从那里，选择在输出图像的*质量*和调整大小发生的*时间*之间达到最佳平衡的调整大小模式。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！******