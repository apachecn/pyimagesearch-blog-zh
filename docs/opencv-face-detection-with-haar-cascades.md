# 基于 Haar 级联的 OpenCV 人脸检测

> 原文：<https://pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/>

在本教程中，您将学习如何使用 OpenCV 和 Haar 级联执行人脸检测。

这个指南，以及接下来的两个，受到了我从 PyImageSearch 阅读器 Angelos 收到的一封电子邮件的启发:

> *嗨，阿德里安，*
> 
> 过去三年来，我一直是 PyImageSearch 的忠实读者，感谢所有的博客帖子！
> 
> *我的公司做大量的人脸应用工作，包括人脸检测、识别等。*
> 
> 我们刚刚开始了一个使用嵌入式硬件的新项目。我没有奢侈地使用你们之前覆盖的 [*OpenCV 的深度学习人脸检测器*](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) *，它在我的设备上太慢了。*
> 
> 你建议我做什么？

首先，我建议 Angelos 研究一下协处理器，如 Movidius NCS 和 Google Coral USB 加速器。这些设备可以实时运行计算昂贵的基于深度学习的人脸检测器(包括 OpenCV 的深度学习人脸检测器)。

也就是说，我不确定这些协处理器是否是 Angelos 的一个选项。它们可能成本过高、需要太多的功率消耗等。

我想了想安杰洛的问题，然后翻了翻档案，看看我是否有可以帮助他的教程。

令我惊讶的是，我意识到我从来没有用 OpenCV 的 Haar cascades 编写过一个关于人脸检测的专门教程。

虽然我们可以通过深度学习人脸检测器获得*显著*更高的准确性和更鲁棒的人脸检测，但 OpenCV 的 Haar cascades 仍有其一席之地:

*   它们很轻
*   即使在资源受限的设备上，它们也是超级快的
*   哈尔级联模型很小(930 KB)

是的，哈尔级联有几个问题，即它们容易出现假阳性检测，并且不如它们的 HOG +线性 SVM、SSD、YOLO 等准确。，同行。**然而，它们仍然是有用和实用的，*尤其是在资源受限的设备上*。**

今天，您将学习如何使用 OpenCV 执行人脸检测。下周我们将讨论 OpenCV 中包含的其他 Haar 级联，即眼睛和嘴巴检测器。在两周内，你将学会如何使用 dlib 的 HOG +线性 SVM 人脸检测器和深度学习人脸检测器。

**要了解如何使用 OpenCV 和 Haar cascades 执行人脸检测，*继续阅读。***

## **利用哈尔级联进行 OpenCV 人脸检测**

在本教程的第一部分，我们将配置我们的开发环境，然后回顾我们的项目目录结构。

然后，我们将实现两个 Python 脚本:

1.  第一个将应用哈尔级联检测静态图像中的人脸
2.  第二个脚本将利用 OpenCV 的哈尔级联来检测实时视频流中的人脸

我们将讨论我们的结果来结束本教程，包括哈尔级联的局限性。

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

那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南已经过*预配置*，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

在我们学习如何用 OpenCV 的 Haar 级联应用人脸检测之前，让我们先回顾一下我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像:

```py
$ tree . --dirsfirst
.
├── images
│   ├── adrian_01.png
│   ├── adrian_02.png
│   └── messi.png
├── haar_face_detector.py
├── haarcascade_frontalface_default.xml
└── video_face_detector.py

1 directory, 6 files
```

我们今天要复习两个 Python 脚本:

1.  `haar_face_detector.py`:对输入图像应用 Haar 级联人脸检测。
2.  `video_face_detector.py`:利用哈尔级联进行实时人脸检测。

`haarcascade_frontalface_default.xml`文件是我们预先训练好的人脸检测器，由 OpenCV 库的开发者和维护者提供。

然后,`images`目录包含我们将应用哈尔级联的示例图像。

### **利用 OpenCV 和 Haar 级联实现人脸检测**

让我们开始用 OpenCV 和 Haar 级联实现人脸检测。

打开项目目录结构中的`haar_face_detector.py`文件，让我们开始工作:

```py
# import the necessary packages
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-c", "--cascade", type=str,
	default="haarcascade_frontalface_default.xml",
	help="path to haar cascade face detector")
args = vars(ap.parse_args())
```

**第 2-4 行**导入我们需要的 Python 包。我们将需要`argparse`进行命令行参数解析，`imutils`用于 OpenCV 便利函数，以及`cv2`用于 OpenCV 绑定。

**第 7-13 行**解析我们需要的命令行参数，包括:

1.  `--image`:我们要应用 Haar 级联人脸检测的输入图像的路径。
2.  `--cascade`:驻留在磁盘上的预训练 Haar 级联检测器的路径。

解析完命令行参数后，我们可以从磁盘加载 Haar cascade:

```py
# load the haar cascade face detector from
print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])

# load the input image from disk, resize it, and convert it to
# grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

对第 17 行**上`cv2.CascadeClassifier`的调用从磁盘加载我们的人脸检测器。**

然后，我们加载我们的输入`image`，调整它的大小，并将其转换为灰度(我们将 Haar 级联应用于灰度图像)。

最后一步是检测和注释:

```py
# detect faces in the input image using the haar cascade face
# detector
print("[INFO] performing face detection...")
rects = detector.detectMultiScale(gray, scaleFactor=1.05,
	minNeighbors=5, minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE)
print("[INFO] {} faces detected...".format(len(rects)))

# loop over the bounding boxes
for (x, y, w, h) in rects:
	# draw the face bounding box on the image
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
```

**第 28-30 行**然后在我们的输入`image`中检测实际的人脸，返回一个边界框列表，或者仅仅是开始和结束的 *(x，y)*-人脸在每个图像中的坐标。

让我们来看看这些参数的含义:

1.  `scaleFactor` : **每种图像比例下图像缩小多少。**该值用于创建[比例金字塔](https://pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/)。在图像中以多个尺度检测人脸(一些人脸可能更靠近前景，因此更大，其他人脸可能更小，并且在背景中，因此使用变化的尺度)。值`1.05`表示我们在金字塔的每一层将图像尺寸缩小 5%。
2.  `minNeighbors` : **每个窗口应该有多少个邻居，窗口中的区域才被认为是面部。**级联分类器将检测人脸周围的多个窗口。此参数控制需要检测多少个矩形(邻居)才能将窗口标记为面。
3.  `minSize` : **一组宽度和高度(以像素为单位)表示窗口的最小尺寸。**小于该尺寸的边界框被忽略。从`(30, 30)`开始并从那里进行微调是一个好主意。

最后，给定边界框列表，我们逐个循环，并在第 34-36 行的**上围绕面部画出边界框。**

### **哈尔卡斯克德人脸检测结果**

让我们来测试一下我们的 Haar cascade 人脸检测器吧！

首先访问本教程的 ***“下载”*** 部分，以检索源代码、示例图像和预训练的 Haar cascade 人脸检测器。

从那里，您可以打开一个 shell 并执行以下命令:

```py
$ python haar_face_detector.py --image images/messi.png
[INFO] loading face detector...
[INFO] performing face detection...
[INFO] 2 faces detected...
```

如图 2 所示，我们已经能够成功检测输入图像中的两张人脸。

让我们尝试另一个图像:

```py
$ python haar_face_detector.py --image images/adrian_01.png
[INFO] loading face detector...
[INFO] performing face detection...
[INFO] 1 faces detected...
```

果然，我的脸被检测到了。

下图提出了一点问题，并展示了哈尔级联的最大限制之一，即*假阳性检测:*

```py
$ python haar_face_detector.py --image images/adrian_02.png
[INFO] loading face detector...
[INFO] performing face detection...
[INFO] 2 faces detected...
```

虽然你可以看到我的脸被正确地检测到，但在图像的底部我们也有一个假阳性检测。

哈尔喀斯倾向于对你选择的`detectMultiScale`参数*非常敏感*。`scaleFactor`和`minNeighbors`是你最常调的。

当你以假阳性检测结束时(或者根本没有检测到人脸)，你应该返回到你的`detectMultiScale`功能，并尝试通过试错来调整参数。

例如，我们的*原始*对`detectMultiScale`的调用如下所示:

```py
rects = detector.detectMultiScale(gray, scaleFactor=1.05,
	minNeighbors=5, minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE)
```

通过实验，我发现当*通过将`minNeighbors`从`5`更新为`7`来消除假阳性*时，我仍然能够*检测到我的面部*:

```py
rects = detector.detectMultiScale(gray, scaleFactor=1.05,
	minNeighbors=7, minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE)
```

这样做之后，我们获得了正确的结果:

```py
$ python haar_face_detector.py --image images/adrian_02.png
[INFO] loading face detector...
[INFO] performing face detection...
[INFO] 1 faces detected...
```

这一更新之所以有效，是因为`minNeighbors`参数旨在帮助控制误报检测。

当应用人脸检测时，Haar cascades 在图像上从左到右和从上到下滑动窗口，计算 T2 积分图像。

当 Haar cascade 认为一张脸在一个区域中时，它将返回更高的置信度得分。如果在给定区域中有足够高的置信度得分，那么 Haar 级联将报告阳性检测。

通过增加`minNeighbors`,我们可以要求哈尔级联找到更多的邻居，从而消除我们在图 4**中看到的假阳性检测。**

上面的例子再次强调了哈尔级联的主要局限性。虽然它们速度很快，但您通过以下方式付出了代价:

1.  误报检测
2.  精确度较低(与 HOG +线性 SVM 和基于深度学习的人脸检测器相反)
3.  手动参数调谐

也就是说，在资源有限的环境中，你无法击败哈尔级联人脸检测的速度。

### **利用哈尔级联实现实时人脸检测**

我们之前的例子演示了如何将 Haar 级联的人脸检测应用于单个图像。

现在让我们了解如何在实时视频流中执行人脸检测:

```py
# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
```

**第 2-6 行**导入我们需要的 Python 包。类允许我们访问我们的网络摄像头。

我们只有一个命令行参数需要解析:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str,
	default="haarcascade_frontalface_default.xml",
	help="path to haar cascade face detector")
args = vars(ap.parse_args())
```

`--cascade`参数指向我们预先训练的驻留在磁盘上的 Haar cascade 人脸检测器。

然后，我们加载人脸检测器并初始化视频流:

```py
# load the haar cascade face detector from
print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
```

让我们开始从视频流中读取帧:

```py
# loop over the frames from the video stream
while True:
	# grab the frame from the video stream, resize it, and convert it
	# to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# perform face detection
	rects = detector.detectMultiScale(gray, scaleFactor=1.05,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
```

在我们的`while`循环中，我们:

1.  从相机中读取下一个`frame`
2.  将它的宽度调整为 500 像素(较小的帧处理速度更快)
3.  将框架转换为灰度

**第 33-35 行**然后使用我们的 Haar 级联执行人脸检测。

最后一步是在我们的`frame`上绘制检测到的人脸的边界框:

```py
	# loop over the bounding boxes
	for (x, y, w, h) in rects:
		# draw the face bounding box on the image
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
```

**第 38 行**循环遍历`rects`列表，包含:

1.  面部的起始`x`坐标
2.  面部的起始`y`坐标
3.  边界框的宽度(`w`)
4.  边界框的高度(`h`)

然后我们在屏幕上显示输出`frame`。

### **实时哈尔级联人脸检测结果**

我们现在已经准备好使用 OpenCV 实时应用人脸检测了！

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和预训练的 Haar cascade。

从那里，打开一个 shell 并执行以下命令:

```py
$ python video_face_detector.py
[INFO] loading face detector...
[INFO] starting video stream...
```

如你所见，我们的 Haar cascade 人脸检测器正在实时运行，没有任何问题！

如果您需要在嵌入式设备上获得实时人脸检测，*尤其是*，那么可以考虑利用 Haar 级联人脸检测器。

是的，它们不如更现代的面部检测器准确，是的，它们也容易出现假阳性检测，但好处是你会获得巨大的速度，并且你需要更少的计算能力。

否则，如果你在笔记本电脑/台式机上，或者你可以使用 Movidius NCS 或谷歌 Coral USB 加速器等协处理器，那么就使用[基于深度学习的人脸检测](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)。你将获得更高的准确度，并且仍然能够实时应用面部检测。

## **总结**

在本教程中，您学习了如何使用 OpenCV 和 Haar 级联执行人脸检测。

而哈尔级联*明显*不如它们的 HOG +线性 SVM、SSD、YOLO 等精确。，同行，他们是*速度非常快，轻巧*。这使得它们适合在嵌入式设备上使用，特别是在像 Movidius NCS 和 Google Coral USB 加速器这样的协处理器不可用的情况下。

下周我们将讨论其他 OpenCV 哈尔级联，包括眼睛和嘴巴探测器。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***