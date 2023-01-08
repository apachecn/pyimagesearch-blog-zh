# 人脸检测提示、建议和最佳实践

> 原文：<https://pyimagesearch.com/2021/04/26/face-detection-tips-suggestions-and-best-practices/>

在本教程中，您将学习我的技巧、建议和最佳实践，以使用 OpenCV 和 dlib 实现高人脸检测准确性。

我们已经在 PyImageSearch 博客上四次讨论了人脸检测:

1.  [利用 OpenCV 和 Haar 级联进行人脸检测](https://pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/)
2.  [利用 OpenCV 和深度神经网络(DNNs)进行人脸检测](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
3.  [使用 dlib 和 HOG +线性 SVM 算法进行人脸检测](https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/)
4.  [使用 dlib 和最大余量对象检测器(MMOD)进行人脸检测](https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/)

***注:*** *#3、#4 链接到同一个教程作为指南封面***HOG+线性 SVM 和 MMOD CNN 人脸检测器。**

 *今天，我们将比较和对比这些方法，让你知道什么时候应该使用每种方法，让你平衡速度、准确性和效率。

**要了解我的面部检测技巧、建议和最佳实践，*请继续阅读。***

## **人脸检测技巧、建议和最佳实践**

在本教程的第一部分，我们将回顾您在构建自己的计算机视觉管道时会遇到的四种主要人脸检测器，包括:

1.  OpenCV 和哈尔级联
2.  OpenCV 基于深度学习的人脸检测器
3.  Dlib 的 HOG +线性 SVM 实现
4.  Dlib 的 CNN 人脸检测器

然后我们将比较和对比这些方法。此外，我会给你每一个优点和缺点，以及我个人的建议，当你应该使用一个特定的人脸检测器。

在本教程的最后，我将推荐一种“默认的、通用的”人脸检测器，当你构建自己的需要人脸检测的计算机视觉项目时，这应该是你的“第一次尝试”。

### **您将在计算机视觉项目中经常使用的 4 种流行的人脸检测方法**

我们在 PyImageSearch 博客中介绍了四种主要的人脸检测方法:

1.  [OpenCV 和哈尔级联](https://pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/)
2.  [OpenCV 基于深度学习的人脸检测器](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
3.  [Dlib 的 HOG +线性 SVM 实现](https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/)
4.  [Dlib 的 CNN 人脸检测器](https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/)

***注:*** *#3、#4 链接到同一个教程作为指南封面***HOG+线性 SVM 和 MMOD CNN 人脸检测器。**

 *在继续之前，我建议你逐个查看这些帖子，以便更好地欣赏我们将要进行的比较/对比。

### **OpenCV 的 Haar cascade 人脸检测器的优缺点**

[OpenCV 的 Haar cascade 人脸检测器](https://pyimagesearch.com/2021/04/05/opencv-face-detection-with-haar-cascades/)是库自带的原装人脸检测器。它也是大多数人都熟悉的面部检测器。

**优点:**

*   *非常快*，能够超实时运行
*   低计算要求——*可以轻松地*在嵌入式、资源受限的设备上运行，如 Raspberry Pi (RPi)、NVIDIA Jetson Nano 和 Google Coral
*   小型号(400KB 多一点；作为参考，大多数深度神经网络将在 20-200MB 之间。

**缺点:**

*   非常容易出现假阳性检测
*   通常需要手动调谐到`detectMultiScale`功能
*   远不如其 HOG +线性 SVM 和基于深度学习的人脸检测技术准确

**我的建议:**当速度是您的主要关注点，并且您愿意牺牲一些准确性来获得实时性能时，请使用 Haar cascades。

如果您正在使用 RPi、Jetson Nano 或 Google Coral 等嵌入式设备，请考虑:

*   在 RPi 上使用 Movidius 神经计算棒(NCS)——这将允许您实时运行基于深度学习的面部检测器
*   阅读与您的设备相关的文档 Nano 和 Coral 有专门的推理引擎，可以实时运行深度神经网络

### **OpenCV 深度学习人脸检测器的利弊**

[OpenCV 的深度学习人脸检测器](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)是基于单镜头检测器(SSD)的小型 ResNet 主干，使其既有 ***的准确性*** 又有 ***的快速性。***

**优点:**

*   精确人脸检测器
*   利用现代深度学习算法
*   不需要调整参数
*   可以在现代笔记本电脑和台式机上实时运行
*   模型大小合理(刚刚超过 10MB)
*   依赖于 OpenCV 的`cv2.dnn`模块
*   通过使用 OpenVINO 和 Movidius NCS，可以在嵌入式设备上运行得更快

**缺点:**

*   比哈尔瀑布和 HOG +线性 SVM 更准确，但不如 dlib 的 CNN MMOD 人脸检测器准确
*   可能在训练集中存在无意识的偏见——可能无法像检测浅色皮肤的人那样准确地检测深色皮肤的人

**我的推荐:** OpenCV 的深度学习人脸检测器是你最好的“全能”检测器。使用起来非常简单，不需要额外的库，依靠 OpenCV 的`cv2.dnn`模块，这个模块被烘焙到 OpenCV 库中。

此外，如果您使用的是嵌入式设备，如 Raspberry Pi，您可以插入 Movidius NCS，并利用 OpenVINO 轻松获得实时性能。

也许这个模型最大的缺点是，我发现肤色较深的人的面部检测不如肤色较浅的人准确。这不一定是模型本身的问题，而是训练数据的问题——为了解决这个问题，我建议在更多样化的种族集上训练/微调面部检测器。

### **dlib 的 HOG +线性 SVM 人脸检测器的优劣**

[HOG +线性 SVM 算法](https://pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)是由 Dalal 和 Triggs 在他们 2005 年的开创性工作 [*中首次提出的，用于人体检测的梯度方向直方图*](https://hal.inria.fr/inria-00548512/document) 。

与哈尔级联类似，HOG +线性 SVM 依靠图像金字塔和滑动窗口来检测图像中的对象/人脸。

该算法是计算机视觉文献中的经典，至今仍在使用。

**优点:**

*   比哈尔瀑布更准确
*   比哈尔级联更稳定的检测(即需要调节的参数更少)
*   由 dlib 创建者和维护者 Davis King 专业实现
*   非常有据可查，包括计算机视觉文献中的 dlib 实现和 HOG +线性 SVM 框架

**缺点:**

*   仅适用于正面视图——轮廓面将*而非*被检测到，因为 HOG 描述符不能很好地容忍旋转或视角的变化
*   需要安装一个额外的库(dlib)——本质上不一定是问题，但是如果你使用的是 OpenCV，那么你会发现添加另一个库很麻烦
*   不如基于深度学习的人脸检测器准确
*   对于准确性，由于图像金字塔结构、滑动窗口和在窗口的每一站计算 HOG 特征，它实际上在计算上相当昂贵

**我的推荐:** HOG +线性 SVM 是*每个*计算机视觉从业者都应该了解的经典物体检测算法。也就是说，对于 HOG +线性 SVM 给你的精度来说，算法本身相当慢，尤其是当你把它与 OpenCV 的 SSD 人脸检测器进行比较时。

我倾向于在哈尔级联不够准确的地方使用 HOG +线性 SVM，但我不能承诺使用 OpenCV 的深度学习人脸检测器。

### **dlib 的 CNN 人脸检测器的利弊**

dlib 的创造者戴维斯·金(Davis King)根据他在 [max-margin 对象检测](https://arxiv.org/abs/1502.00046)方面的工作训练了一个 CNN 人脸检测器。这种方法*非常精确*，这要归功于算法本身的设计，以及 Davis 在管理训练集和训练模型时的细心。

也就是说，如果没有 GPU 加速，这个模型就无法实时运行。

**优点:**

*   ***y*****精确*** 人脸检测器*
*   *小型模型(小于 1MB)*
*   *专业实施和记录*

 ***缺点:**

*   需要安装额外的库(dlib)
*   代码更加冗长——如果使用 OpenCV，最终用户必须小心转换和修剪边界框坐标
*   没有 GPU 加速，无法实时运行
*   并非开箱即用，兼容 OpenVINO、Movidius NCS、NVIDIA Jetson Nano 或 Google Coral 的加速功能

**我的推荐:**离线批量处理人脸检测时，我倾向于使用 dlib 的 MMOD CNN 人脸检测器，意思是我可以设置好我的脚本，让它以批处理模式运行，而不用担心实时性能。

事实上，当我为人脸识别建立[训练集时，我经常在训练人脸识别器本身之前使用 dlib 的 CNN 人脸检测器来检测人脸。当我准备好部署我的人脸识别模型时，我通常会将 dlib 的 CNN 人脸检测器换成一个计算效率更高、可以实时运行的检测器(例如 OpenCV 的 CNN 人脸检测器)。](https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)

我唯一倾向于*而不是*使用 dlib 的 CNN 人脸检测器的地方是在我使用嵌入式设备的时候。这种模型不会在嵌入式设备上实时运行，它与嵌入式设备加速器(如 Movidius NCS)开箱即用。

**也就是说，** ***你就是打不过 dlib 的 MMOD CNN 的*** **，所以如果你需要准确的人脸检测，** ***就用这个模式吧。***

### **我个人对人脸检测的建议**

**说到一个好的，万能的人脸检测器，我建议用** [**OpenCV 的 DNN 人脸检测器**](https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/) **:**

*   它实现了*速度*和*精度*的良好平衡
*   作为一个基于深度学习的检测器，它比它的 Haar cascade 和 HOG +线性 SVM 同行更准确
*   它足够快，可以在 CPU 上实时运行
*   使用诸如 Movidius NCS 等 USB 设备可以进一步加速
*   不需要额外的库/包——对人脸检测器的支持通过`cv2.dnn`模块嵌入 OpenCV

也就是说，有时候你会想要使用上面提到的每一个面部检测器，所以一定要仔细阅读每一个章节。

## **总结**

在本教程中，您了解了我关于人脸检测的技巧、建议和最佳实践。

简而言之，它们是:

1.  **当速度是您的主要关注点时(例如，当您使用 Raspberry Pi 等嵌入式设备时)，请使用 OpenCV 的 Haar cascades。** Haar cascades 不像它们的 HOG +线性 SVM 和基于深度学习的同行那样准确，但它们在原始速度上弥补了这一点。请注意*肯定会有*，调用`detectMultiScale`时需要一些误报检测和参数调整。
2.  当哈尔级联不够准确时，使用 dlib 的 HOG +线性 SVM 检测器，但你不能承诺基于深度学习的人脸检测器的计算要求。HOG+线性 SVM 物体检测器是计算机视觉文献中的经典算法，至今仍然适用。dlib 库在实现它方面做了一件*了不起的工作*。请注意，在 CPU 上运行 HOG +线性 SVM 对于您的嵌入式设备来说可能太慢了。
3.  **当你需要超精确的人脸检测时，使用 dlib 的 CNN 人脸检测。**说到人脸检测的准确性，dlib 的 MMOD CNN 人脸检测器是*难以置信的准确。*也就是说，这是一种折衷——精度越高，运行时间越长。这种方法*不能*在笔记本电脑/台式机 CPU 上实时运行，即使有 GPU 加速，你也很难达到实时性能。我通常在离线批处理中使用这个人脸检测器，我不太关心人脸检测需要多长时间(相反，我想要的只是高精度)。
4.  **使用 OpenCV 的 DNN 人脸检测器作为一个很好的平衡。**作为一种基于深度学习的人脸检测器，这种方法是准确的——而且由于这是一个以 SSD 为骨干的浅层网络，它很容易在 CPU 上实时运行。此外，由于您可以将该模型与 OpenCV 的`cv2.dnn`模块一起使用，因此*也就是*意味着(1)您可以通过使用 GPU 来进一步提高速度，或者(2)在您的嵌入式设备上利用 Movidius NCS。

总的来说，OpenCV 的 DNN 人脸检测器应该是你应用人脸检测的“第一站”。根据 OpenCV DNN 人脸检测器的精确度，您可以尝试其他方法。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，** ***只需在下面的表格中输入您的电子邮件地址！******