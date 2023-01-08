# 基于 dlib (HOG 和 CNN)的人脸检测

> 原文：<https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/>

在本教程中，您将学习如何使用 HOG +线性 SVM 和 CNN 通过 dlib 库执行人脸检测。

dlib 库可以说是人脸识别最常用的软件包之一。一个名为`face_recognition`的 Python 包将 dlib 的人脸识别功能封装到一个简单易用的 API 中。

***注:*** *如果你对使用 dlib 和`face_recognition`库进行人脸识别感兴趣，* [*参考本教程*](https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) *，在这里我会详细介绍这个话题。*

然而，我经常惊讶地听到读者不知道 dlib 包括两个内置在库中的人脸检测方法:

1.  精确且计算高效的 **HOG +线性 SVM 人脸检测器**。
2.  一个 **Max-Margin (MMOD) CNN 人脸检测器**，它既是*高度精确的*又是*非常健壮的*，能够从不同的视角、光照条件和遮挡情况下检测人脸。

最重要的是，MMOD 人脸检测器可以在 NVIDIA GPU 上运行，速度超快！

**要了解如何使用 dlib 的 HOG +线性 SVM 和 MMOD 人脸检测器，*继续阅读。***

## **用 dlib (HOG 和 CNN)进行人脸检测**

在本教程的第一部分，你会发现 dlib 的两个人脸检测功能，一个用于 HOG +线性 SVM 人脸检测器，另一个用于 MMOD CNN 人脸检测器。

从那里，我们将配置我们的开发环境，并审查我们的项目目录结构。

然后，我们将实现两个 Python 脚本:

1.  `hog_face_detection.py`:应用 dlib 的 HOG +线性 SVM 人脸检测器。
2.  `cnn_face_detection.py`:利用 dlib 的 MMOD CNN 人脸检测器。

然后，我们将在一组图像上运行这些面部检测器，并检查结果，注意在给定的情况下何时使用每个面部检测器。

我们开始吧！

### **Dlib 的人脸检测方法**

dlib 库提供了两个可用于人脸检测的函数:

1.  **猪+线性 SVM:** `dlib.get_frontal_face_detector()`
2.  **MMOD CNN:** `dlib.cnn_face_detection_model_v1(modelPath)`

`get_frontal_face_detector`函数不接受任何参数。对它的调用返回包含在 dlib 库中的预训练的 HOG +线性 SVM 人脸检测器。

Dlib 的 HOG +线性 SVM 人脸检测器快速高效。根据梯度方向直方图(HOG)描述符的工作原理，它对于旋转和视角的变化不是不变的。

对于更强大的面部检测，您可以使用 MMOD CNN 面部检测器，可通过`cnn_face_detection_model_v1`功能获得。这个方法接受一个参数`modelPath`，它是驻留在磁盘上的预先训练好的`mmod_human_face_detector.dat`文件的路径。

***注意:*** *我已经将`mmod_human_face_detector.dat`文件包含在本指南的* ***【下载】*** *部分，所以你不必去寻找它。*

在本教程的剩余部分，您将学习如何使用这两种 dlib 人脸检测方法。

### **配置您的开发环境**

为了遵循这个指南，您需要在您的系统上安装 OpenCV 库和 dlib。

幸运的是，您可以通过 pip 安装 OpenCV 和 dlib:

```py
$ pip install opencv-contrib-python
$ pip install dlib
```

**如果你需要帮助配置 OpenCV 和 dlib 的开发环境，我*强烈推荐*阅读以下两个教程:**

1.  [pip 安装 opencv](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)
2.  [如何安装 dlib](https://pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/)

### **在配置开发环境时遇到了问题？**

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   准备好在您的 Windows、macOS 或 Linux 系统上运行代码*了吗*？

那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

在使用 dlib 执行人脸检测之前，我们首先需要回顾一下我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

从这里，看一下目录结构:

```py
$ tree . --dirsfirst
.
├── images
│   ├── avengers.jpg
│   ├── concert.jpg
│   └── family.jpg
├── pyimagesearch
│   ├── __init__.py
│   └── helpers.py
├── cnn_face_detection.py
├── hog_face_detection.py
└── mmod_human_face_detector.dat
```

我们首先回顾两个 Python 脚本:

1.  `hog_face_detection.py`:使用 dlib 应用 HOG +线性 SVM 人脸检测。
2.  `cnn_face_detection.py`:通过从磁盘加载训练好的`mmod_human_face_detector.dat`模型，使用 dlib 进行基于深度学习的人脸检测。

我们的`helpers.py`文件包含一个 Python 函数`convert_and_trim_bb`，它将帮助我们:

1.  将 dlib 边界框转换为 OpenCV 边界框
2.  修剪超出输入图像边界的任何边界框坐标

`images`目录包含三张图像，我们将使用 dlib 对其进行人脸检测。我们可以将 HOG +线性 SVM 人脸检测方法与 MMOD CNN 人脸检测器进行比较。

### **创建我们的包围盒转换和裁剪函数**

OpenCV 和 dlib 以不同的方式表示边界框:

*   在 OpenCV 中，我们认为边界框是一个四元组:起始坐标 *x* ，起始坐标 *y* ，坐标，宽度和高度
*   Dlib 通过具有左、上、右和下属性的`rectangle`对象表示边界框

此外，dlib 返回的边界框可能会落在输入图像尺寸的边界之外(负值或图像宽度和高度之外的值)。

为了使使用 dlib 进行人脸检测更容易，让我们创建一个辅助函数来(1)将边界框坐标转换为标准 OpenCV 排序，以及(2)修剪图像范围之外的任何边界框坐标。

打开`pyimagesearch`模块中的`helpers.py`文件，让我们开始工作:

```py
def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()

	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])

	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY

	# return our bounding box coordinates
	return (startX, startY, w, h)
```

我们的`convert_and_trim_bb`函数需要两个参数:我们应用人脸检测的输入`image`和 dlib 返回的`rect`对象。

**第 4-7 行**提取边界框的起点和终点 *(x，y)*-坐标。

然后，我们确保边界框坐标落在第 11-14 行的输入`image`的宽度和高度内。

最后一步是计算边界框的宽度和高度(**第 17 行和第 18 行**)，然后以`startX`、`startY`、`w`和`h`的顺序返回边界框坐标的 4 元组。

### **用 dlib 实现 HOG +线性 SVM 人脸检测**

随着我们的`convert_and_trim_bb`助手工具的实现，我们可以继续使用 dlib 来执行 HOG +线性 SVM 人脸检测。

打开项目目录结构中的`hog_face_detection.py`文件，插入以下代码:

```py
# import the necessary packages
from pyimagesearch.helpers import convert_and_trim_bb
import argparse
import imutils
import time
import dlib
import cv2
```

**第 2-7 行**导入我们需要的 Python 包。注意，我们刚刚实现的`convert_and_trim_bb`函数被导入了。

当我们为 OpenCV 绑定导入`cv2`时，我们也导入了`dlib`，因此我们可以访问它的人脸检测功能。

接下来是我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-u", "--upsample", type=int, default=1,
	help="# of times to upsample")
args = vars(ap.parse_args())
```

我们有两个命令行参数要解析:

1.  `--image`:应用 HOG +线性 SVM 人脸检测的输入图像路径。
2.  `--upsample`:在应用面部检测之前对图像进行上采样的次数。

为了检测大输入图像中的小人脸，我们可能希望提高输入图像的分辨率，从而使较小的人脸看起来更大。这样做允许我们的[滑动窗口](https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)检测脸部。

向上采样的缺点是，它会在我们的[图像金字塔](https://pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/)中创建更多层，使检测过程更慢。

对于更快的面部检测，将`--upsample`值设置为`0`，这意味着*不执行*上采样(但是您有丢失面部检测的风险)。

接下来我们从磁盘加载 dlib 的 HOG +线性 SVM 人脸检测器:

```py
# load dlib's HOG + Linear SVM face detector
print("[INFO] loading HOG + Linear SVM face detector...")
detector = dlib.get_frontal_face_detector()

# load the input image from disk, resize it, and convert it from
# BGR to RGB channel ordering (which is what dlib expects)
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# perform face detection using dlib's face detector
start = time.time()
print("[INFO[ performing face detection with dlib...")
rects = detector(rgb, args["upsample"])
end = time.time()
print("[INFO] face detection took {:.4f} seconds".format(end - start))
```

对`dlib.get_frontal_face_detector()`的调用返回 dlib 的 HOG +线性 SVM 人脸检测器(**第 19 行**)。

然后，我们继续:

1.  从磁盘加载输入`image`
2.  调整图像大小(图像越小，HOG +线性 SVM 运行越快)
3.  将图像从 BGR 转换为 RGB 通道排序(dlib 需要 RGB 图像)

从那里，我们将我们的 HOG +线性 SVM 人脸检测器应用于**线 30** ，计时人脸检测过程需要多长时间。

现在让我们分析我们的边界框:

```py
# convert the resulting dlib rectangle objects to bounding boxes,
# then ensure the bounding boxes are all within the bounds of the
# input image
boxes = [convert_and_trim_bb(image, r) for r in rects]

# loop over the bounding boxes
for (x, y, w, h) in boxes:
	# draw the bounding box on our image
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
```

请记住，返回的`rects`列表需要一些工作——我们需要将 dlib `rectangle`对象解析成一个四元组，起始 *x* 坐标，起始 *y* 坐标，宽度和高度——这正是*第**行第 37** 所完成的*。

对于每个`rect`，我们调用我们的`convert_and_trim_bb`函数，确保(1)所有的边界框坐标都落在`image`的空间维度内，以及(2)我们返回的边界框是正确的 4 元组格式。

### **Dlib HOG +线性 SVM 人脸检测结果**

让我们来看看将我们的 dlib HOG +线性 SVM 人脸检测器应用于一组图像的结果。

请务必访问本教程的 ***“下载”*** 部分，以检索源代码、示例图像和预训练模型。

从那里，打开一个终端窗口并执行以下命令:

```py
$ python hog_face_detection.py --image images/family.jpg
[INFO] loading HOG + Linear SVM face detector...
[INFO[ performing face detection with dlib...
[INFO] face detection took 0.1062 seconds
```

**图 3** 显示了将 dlib 的 HOG +线性 SVM 人脸检测器应用于包含多张人脸的输入图像的结果。

面部检测过程花费了![\approx](img/efc44d3feeb891408f5038a6a5a7c926.png "\approx")

0.1 seconds, implying that we could process ![\approx](img/efc44d3feeb891408f5038a6a5a7c926.png "\approx")10 frames per second in a video stream scenario.

最重要的是，注意这四张脸都被正确地检测到了。

让我们尝试一个不同的图像:

```py
$ python hog_face_detection.py --image images/avengers.jpg 
[INFO] loading HOG + Linear SVM face detector...
[INFO[ performing face detection with dlib...
[INFO] face detection took 0.1425 seconds
```

几年前，当《复仇者联盟 4：终局之战》上映时，我和妻子决定装扮成电影《复仇者联盟》中的角色(抱歉，如果你没看过这部电影，但是拜托，已经两年了！)

注意我妻子的脸(呃，黑寡妇？)被检测到了，但显然，dlib 的 HOG +线性 SVM 人脸检测器不知道钢铁侠长什么样。

十有八九，我的脸没有被检测到，因为我的头稍微旋转了一下，不是相机的“正视图”。同样，HOG +线性 SVM 系列物体检测器*在旋转或视角变化的情况下表现不佳。*

让我们来看最后一张照片，这张照片上的人脸更加密集:

```py
$ python hog_face_detection.py --image images/concert.jpg 
[INFO] loading HOG + Linear SVM face detector...
[INFO[ performing face detection with dlib...
[INFO] face detection took 0.1069 seconds
```

早在 COVID 之前，就有这些叫做“音乐会”的东西乐队过去常常聚在一起，为人们演奏现场音乐以换取金钱。很难相信，我知道。

几年前，我的一群朋友聚在一起开音乐会。虽然这张照片上有八张脸，但只有六张被检测到。

正如我们将在本教程的后面看到的，我们可以使用 dlib 的 MMOD CNN 人脸检测器来提高人脸检测的准确性，并检测这张图像中的所有人脸。

### **用 dlib 实现 CNN 人脸检测**

到目前为止，我们已经了解了如何使用 dlib 的 HOG +线性 SVM 模型进行人脸检测。这种方法效果很好，但是使用 dlib 的 MMOD CNN 面部检测器要获得更高的准确度。

现在让我们来学习如何使用 dlib 的深度学习人脸检测器:

```py
# import the necessary packages
from pyimagesearch.helpers import convert_and_trim_bb
import argparse
import imutils
import time
import dlib
import cv2
```

我们这里的导入与我们之前的 HOG +线性 SVM 人脸检测脚本完全相同。

命令行参数类似，但是增加了一个参数(`--model`):

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-m", "--model", type=str,
	default="mmod_human_face_detector.dat",
	help="path to dlib's CNN face detector model")
ap.add_argument("-u", "--upsample", type=int, default=1,
	help="# of times to upsample")
args = vars(ap.parse_args())
```

这里有三个命令行参数:

1.  `--image`:驻留在磁盘上的输入图像的路径。
2.  `--model`:我们预先训练的 dlib MMOD CNN 人脸检测器。
3.  `--upsample`:在应用面部检测之前对图像进行上采样的次数。

考虑到我们的命令行参数，我们现在可以从磁盘加载 dlib 的深度学习人脸检测器:

```py
# load dlib's CNN face detector
print("[INFO] loading CNN face detector...")
detector = dlib.cnn_face_detection_model_v1(args["model"])

# load the input image from disk, resize it, and convert it from
# BGR to RGB channel ordering (which is what dlib expects)
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# perform face detection using dlib's face detector
start = time.time()
print("[INFO[ performing face detection with dlib...")
results = detector(rgb, args["upsample"])
end = time.time()
print("[INFO] face detection took {:.4f} seconds".format(end - start))
```

**第 22 行**通过调用`dlib.cnn_face_detection_model_v1`从磁盘加载`detector`。在这里，我们通过`--model`，这是经过训练的 dlib 人脸检测器所在的路径。

从那里，我们预处理我们的图像(**行 26-28** )，然后应用面部检测器(**行 33** )。

正如我们解析 HOG +线性 SVM 结果一样，我们在这里也需要这样做，但有一点需要注意:

```py
# convert the resulting dlib rectangle objects to bounding boxes,
# then ensure the bounding boxes are all within the bounds of the
# input image
boxes = [convert_and_trim_bb(image, r.rect) for r in results]

# loop over the bounding boxes
for (x, y, w, h) in boxes:
	# draw the bounding box on our image
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
```

Dlib 的 HOG +线性 SVM 探测器返回一个`rectangle`物体列表；然而，MMOD CNN 对象检测器返回一个结果对象列表，每个对象都有自己的矩形(因此我们在列表理解中使用`r.rect`)。否则，实现是相同的。

最后，我们在边界框上循环，并把它们画在我们的输出上`image`。

### **Dlib 的 CNN 人脸检测器结果**

让我们看看 dlib 的 MMOD CNN 人脸检测器如何与 HOG +线性 SVM 人脸检测器相媲美。

要跟进，请务必访问本指南的 ***“下载”*** 部分，以检索源代码、示例图像和预训练的 dlib 人脸检测器。

从那里，您可以打开一个终端并执行以下命令:

```py
$ python cnn_face_detection.py --image images/family.jpg 
[INFO] loading CNN face detector...
[INFO[ performing face detection with dlib...
[INFO] face detection took 2.3075 seconds
```

就像 HOG +线性 SVM 实现一样，dlib 的 MMOD CNN 人脸检测器可以正确检测输入图像中的所有四张人脸。

让我们尝试另一个图像:

```py
$ python cnn_face_detection.py --image images/avengers.jpg 
[INFO] loading CNN face detector...
[INFO[ performing face detection with dlib...
[INFO] face detection took 3.0468 seconds
```

此前，猪+线性 SVM 未能检测到我的脸在左边。但是通过使用 dlib 的深度学习人脸检测器，我们可以正确地检测出*两张*人脸。

让我们看最后一张图片:

```py
$ python cnn_face_detection.py --image images/concert.jpg 
[INFO] loading CNN face detector...
[INFO[ performing face detection with dlib...
[INFO] face detection took 2.2520 seconds
```

之前，使用 HOG +线性 SVM，我们只能检测到这张图像中八个人脸中的六个。但正如我们的输出所示，切换到 dlib 的深度学习人脸检测器会检测到所有八张人脸。

### **我应该使用哪种 dlib 人脸检测器？**

如果你使用的是 CPU 并且速度不是问题，使用 dlib 的 MMOD CNN 人脸检测器。它的*比 HOG +线性 SVM 人脸检测器的*更加准确和稳健。

此外，如果你有 GPU，那么毫无疑问*你应该使用 MMOD CNN 面部检测器——你将享受到*精确*面部检测的所有好处，以及能够实时运行的*速度*。*

假设你仅限于一个 CPU。在这种情况下，速度*是*的一个问题，你愿意容忍稍微低一点的准确性，那么就用 HOG +线性 SVM 吧——它仍然是一个准确的人脸检测器，而且*比 [OpenCV 的 Haar cascade 人脸检测器](https://pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/)更准确。*

## **总结**

在本教程中，您学习了如何使用 dlib 库执行人脸检测。

Dlib 提供了两种执行人脸检测的方法:

1.  **猪+线性 SVM:** `dlib.get_frontal_face_detector()`
2.  **MMOD CNN:** `dlib.cnn_face_detection_model_v1(modelPath)`

HOG +线性 SVM 人脸检测器将比 MMOD CNN 人脸检测器*快*，但也将*不太准确*，因为 HOG +线性 SVM 不容忍视角旋转的变化。

为了更鲁棒的人脸检测，使用 dlib 的 MMOD CNN 人脸检测器。这个模型需要更多的计算(因此更慢),但是对脸部旋转和视角的变化更加精确和稳定。

此外，如果你有一个 GPU，你可以在上面运行 dlib 的 MMOD CNN 人脸检测器，从而提高实时人脸检测速度。MMOD CNN 面部检测器结合 GPU 是天作之合——你既有深度神经网络的*准确性*，又有计算成本较低的模型的*速度*。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***