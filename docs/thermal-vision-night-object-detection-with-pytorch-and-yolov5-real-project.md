# 热视觉:用 PyTorch 和 YOLOv5 探测夜间目标(真实项目)

> 原文：<https://pyimagesearch.com/2022/10/31/thermal-vision-night-object-detection-with-pytorch-and-yolov5-real-project/>

* * *

## **目录**

* * *

## [**热视觉:用 PyTorch 和 YOLOv5 进行夜间目标探测**](#TOC)

在今天的教程中，您将使用深度学习并结合 Python 和 OpenCV 来检测热图像中的对象。正如我们已经发现的，热感相机让我们在绝对黑暗的环境中也能看到东西，所以我们将学习如何在任何可见光条件下探测物体！

本课包括:

*   通过 PyTorch 和 YOLOv5 进行深度学习的对象检测
*   发现前视红外热启动器数据集
*   使用 PyTorch 和 YOLOv5 进行热目标检测

本教程是我们关于**红外视觉基础知识**的 4 部分课程的最后一部分:

1.  [*红外视觉介绍:近中远红外图像*](https://pyimg.co/oj6kb)
2.  [*热视觉:用 Python 和 OpenCV* 从图像中测量你的第一个温度 ](https://pyimg.co/mns3e)
3.  [*热视觉:带 Python 和 OpenCV 的发热探测器(入门项目)*](https://pyimg.co/6nxs0)
4.  [***热视觉:用 PyTorch 和 YOLOv5 进行夜间物体探测(真实项目)***](https://pyimg.co/p2zsm) **(今日教程)**

在本课结束时，您将学习如何使用热图像和深度学习以非常快速、简单和最新的方式检测不同的对象，仅使用四段代码！

**要了解如何利用 YOLOv5 使用您的自定义热成像数据集，** ***继续阅读*** **。**

* * *

## [**热视觉:用 PyTorch 和 YOLOv5 进行夜间目标探测**](#TOC)

* * *

### [**通过 PyTorch 和 YOLOv 进行深度学习的物体检测 5**](#TOC)

在我们的[上一篇教程](https://pyimg.co/6nxs0)中，我们介绍了如何在实际解决方案中应用使用 Python、OpenCV 和传统机器学习方法从热图像中测量的温度。

从这一点出发，并基于本课程中涵盖的所有内容，PyImageSearch 团队激发您的想象力，在任何热成像情况下脱颖而出，但之前会为您提供这种令人难以置信的组合的另一个强大而真实的例子:计算机视觉+热成像。

在这种情况下，我们将了解计算机如何在黑暗中实时区分不同的对象类别。

在开始本教程之前，为了更好地理解，我们鼓励你在 PyImageSearch 大学参加 Torch Hub 系列课程，或者获得一些 PyTorch 和深度学习的经验。和所有 PyImageSearch 大学课程一样，我们会一步一步涵盖各个方面。

正如在 [Torch Hub 系列#3 中所解释的:YOLOv5 和 SSD——关于对象检测的模型](https://pyimagesearch.com/2022/01/03/torch-hub-series-3-yolov5-and-ssd-models-on-object-detection/)，yolov 5——[——你只看一次](https://arxiv.org/abs/1506.02640) ( **图 1** ，2015)版本 5——是最强大的最先进的卷积神经网络模型之一的第五个版本。这种快速对象检测器模型通常在 [COCO 数据集](https://cocodataset.org/#home)上训练，这是一个开放访问的微软 RGB 成像数据库，由 33 万张图像、91 个对象类和 250 万个标记实例组成。

这种强大的组合使 YOLOv5 成为即使在我们的定制成像数据集中检测对象的完美模型。为了获得热物体检测器，我们将使用迁移学习(即，在专门为自动驾驶汽车解决方案收集的真实热成像数据集上训练 COCO 预训练的 YOLOv5 模型)。

* * *

### [**发现前视红外热启动器数据集**](#TOC)

我们将用来训练预训练 YOLOv5 模型的热成像数据集是[免费的 Teledyne FLIR ADAS 数据集](https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset)。

该数据库由 14，452 张灰度 8 和灰度 16 格式的热图像组成，据我们所知，这允许我们测量任何像素温度。用车载热感相机在加州的一些街道上拍摄的 14452 张灰度图像都被手工加上了边框，如图 2 所示**。我们将使用这些注释(标签+边界框)来检测该数据集中预定义的四个类别中的四个不同的对象类别:`car`、`person`、`bicycle`和`dog`。**

提供了一个带有 COCO 格式注释的 JSON 文件。为了简化本教程，我们给出了 YOLOv5 PyTorch 格式的注释。你可以找到一个`labels`文件夹，里面有每个 gray8 图像的单独注释。

我们还将数据集减少到 1，772 张图像:1000 张用于训练我们预先训练的 YOLOv5 模型，772 张用于验证它(即，大约 60-40%的训练-验证分离)。这些图像是从原始数据集的训练部分中选择的。

* * *

### [**利用 PyTorch 和 YOLOv 探测热物体 5**](#TOC)

一旦我们学会了到目前为止看到的所有概念…我们来玩吧！

* * *

### [**配置您的开发环境**](#TOC)

要遵循这个指南，您需要在您的系统上安装 OpenCV 库。

幸运的是，OpenCV 可以通过 pip 安装:

```py
$ pip install opencv-contrib-python
```

**如果您需要帮助配置 OpenCV 的开发环境，我们*强烈推荐*阅读我们的** [***pip 安装 OpenCV* 指南**](https://pyimagesearch.com/2018/09/19/pip-install-opencv/)——它将在几分钟内让您启动并运行。

* * *

### [**在配置开发环境时遇到了问题？**](#TOC)

说了这么多，你是:

*   时间紧迫？
*   了解你雇主的行政锁定系统？
*   想要跳过与命令行、包管理器和虚拟环境斗争的麻烦吗？
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

* * *

### [**项目结构**](#TOC)

我们首先需要回顾我们的项目目录结构。

首先访问本教程的 ***“下载”*** 部分，检索源代码和示例图像。

从这里，看一下目录结构:

```py
$ tree --dirsfirst
.
└── yolov5
    ├── data
    ├── models
    ├── utils
    ├── CONTRIBUTING.md
    ├── Dockerfile
    ├── LICENSE
    ├── ...
    └── val.py

1 directory, XX files
```

我们通过克隆官方的 YOLOv5 库来建立这个结构。

```py
# clone the yolov5 repository from GitHub and install some necessary packages (requirements.txt file)
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
%pip install -qr requirements.txt
```

参见**行 2 和 3** 上的代码。

注意，我们还安装了在`requirements.txt`文件(**第 4 行**)中指出的所需库:Matplotlib、NumPy、OpenCV、PyTorch 等。

在`yolov5`文件夹中，我们可以找到在我们的任何项目中使用 YOLOv5 所需的所有文件:

*   `data`:包含管理 COCO 等不同数据集所需的信息。
*   我们可以在另一种标记语言(YAML)格式中找到所有的 YOLOv5 CNN 结构，这是一种用于编程语言的对人类友好的数据序列化语言。
*   `utils`:包括一些必要的 Python 文件来管理训练、数据集、信息可视化和通用项目工具。

`yolov5`文件中的其余文件是必需的，但我们将只运行其中的两个:

*   是一个用来训练我们的模型的文件，它是我们上面克隆的存储库的一部分
*   `[detect.py](https://github.com/ultralytics/yolov5/blob/master/detect.py)`:是一个通过推断检测到的对象来测试我们的模型的文件，它也是我们上面克隆的存储库的一部分

`thermal_imaging_dataset`文件夹包括我们的 1，772 张灰度热成像图片。该文件夹包含图像(`thermal_imaging_dataset/images`)和标签(`thermal_imaging_dataset/labels`)，分别被分成训练集和验证集、`train`和`val`文件夹。

`thermal_imaging_video_test.mp4`是视频文件，我们将在其上测试我们的热目标检测模型。它包含 4，224 个以 30 帧/秒的速度获取的带有街道和高速公路场景的热帧。

```py
# import PyTorch and check versions
import torch
from yolov5 import utils
display = utils.notebook_init()
```

打开您的`yolov5.py`文件并导入所需的包(**第 7 行和第 8 行**)，如果您正在使用 Google Colab 上的 Jupyter 笔记本，请检查您的笔记本功能(**第 9 行**)。

检查您的环境是否包含 GPU ( **图 3** )，以便在合理的时间内成功运行我们的下一个培训流程。

#### [**预训练**](#TOC)

正如我们已经提到的，我们将使用迁移学习在我们的热成像数据集上训练我们的对象检测器模型，使用在 COCO 数据集上预先训练的 YOLOv5 CNN 架构作为起点。

为此，所选的经过训练的 YOLOv5 型号是 YOLOv5s 版本，因为它具有高速度精度性能。

* * *

#### [**训练**](#TOC)

在设置好环境并满足所有要求后，让我们来训练我们的预训练模型！

```py
# train pretrained YOLOv5s model on the custom thermal imaging dataset,
# basic parameters:
#  - image size (img): image size of the thermal dataset is 640 x 512, 640 passed
#  - batch size (batch): 16 by default, 16 passed
#  - epochs (epochs): number of epochs, 30 passed
#  - dataset (data): dataset in .yaml file format, custom thermal image dataset passed 
#  - pre-trained YOLOv5 model (weights): YOLOv5 model version, YOLOv5s (small version) passed
!python train.py --img 640 --batch 16 --epochs 30 --data thermal_image_dataset.yaml --weights yolov5s.pt
```

在**第 18 行**上，导入 PyTorch 和 YOLOv5 实用程序(**第 7-9 行**)后，我们通过指定以下参数运行`train.py`文件:

*   `img`:要通过我们模型的训练图像的图像大小。在我们的例子中，热图像有一个`640x512`分辨率，所以我们指定最大尺寸，640 像素。
*   `batch`:批量大小。我们设置了 16 个图像的批量大小。
*   `epochs`:训练时代。在一些测试之后，我们将 30 个时期确定为一个很好的迭代次数。
*   `data` : YAML 数据集文件。**图 4** 显示了我们的数据集文件。它指向 YOLOv5 数据集的结构，前面解释过:

    `thermal_imaging_datasimg/train`
    `thermal_imaging_dataset/labels/train`，

    用于训练，

    `thermal_imaging_datasimg/val`
    `thermal_imaging_dataset/labels/val`，

    用于验证。

    还表示班级的数量`nc: 4`，以及班级名称`names: ['bicycle', 'car', 'dog', 'person']`。

    这个 YAML 数据集文件应该位于`yolov5/data`中。
*   `weights`:在 COCO 数据集上计算预训练模型的权重，在我们的例子中是 YOLOv5s。`yolov5s.pt`文件是包含这些权重的预训练模型，位于`yolov5/models`。

这就是我们训练模型所需要的！

让我们看看结果吧！

在 0.279 小时内在 GPU NVIDIA Tesla T4 中完成 30 个纪元后，我们的模型已经学会检测类别`person`、`car`、`bicycle`和`dog`，达到平均 50.7%的平均精度，mAP (IoU = 0.5) = 0.507，如图**图 5** 所示。这意味着我们所有类的平均预测值为 50.7%，交集为 0.5(IoU，**图 6** )。

如图**图 6** 所示，当比较原始和预测时，并集上的交集(IoU)是边界框的右重叠。

因此，对于我们的`person`类，我们的模型平均正确地检测到 77.7%的情况，考虑到当有 50%或更高的边界框交集时的正确预测。

**图 7** 比较了两幅原始图像、它们的手绘边界框以及它们的预测结果。

尽管这超出了本教程的范围，但重要的是要注意我们的数据集是高度不平衡的，对于我们的`bicycle`和`dog`类，分别只有 280 和 31 个标签。这就是为什么我们分别得到 mAP `bicycle` (IoU = 0.5) = 0.456 和 mAP `dog` (IoU = 0.5) = 0.004。

最后，为了验证我们的结果，**图 8** 显示了在训练(*左上*)和验证(*左下*)过程中的分类损失，以及在 IoU 50% ( *中右*)时的平均精度，mAP (IoU = 0.5)，所有类别经过 30 个时期。

但是现在，让我们来测试我们的模型！

* * *

#### [**检测**](#TOC)

为此，我们将使用位于项目根的`thermal_imaging_video_test.mp4`，通过 Python 文件`detect.py`将它传递到我们的模型层。

```py
# test the trained model (night_object_detector.pt) on a thermal imaging video,
# parameters:
#  - trained model (weights): model trained in the previous step, night_object_detector.pt passed
#  - image size (img): frame size of the thermal video is 640 x 512, 640 passed
#  - confidence (conf): confidence threshold, only the inferences higher than this value will be shown, 0.35 passed
#  - video file (source): thermal imaging video, thermal_imaging_video.mp4 passed
!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.35 --source ../thermal_imaging_video.mp4
```

第 27 行显示了如何做。

我们通过指定以下参数来运行`detect.py`:

*   指向我们训练好的模型。在`best.pt`文件(`runs/train/exp/weights/best.pt`)中收集的计算重量。
*   `img`:将通过我们模型的测试图像的图像尺寸。在我们的例子中，视频中的热图像具有`640x512`分辨率，因此我们将最大尺寸指定为 640 像素。
*   `conf`:每次检测的置信度。该阈值建立了检测的概率水平，根据该水平，检测被认为是正确的，因此被显示。我们设定置信度为 35%。
*   `source`:测试模型的图像，在我们的例子中，是视频文件`thermal_imaging_video.mp4`。

来测试一下吧！

**图 9** 展示了我们良好结果的 GIF 图！

正如我们已经指出的，该视频的夜间物体检测已经获得了 35%的置信度。为了修改这个因素，我们应该检查图 10 中的**曲线，该曲线绘制了精度与置信度的关系。**

* * *

* * *

## [**汇总**](#TOC)

我们要感谢[超极本](https://github.com/ultralytics)的伟大工作。我们发现他们的`[train.py](https://github.com/ultralytics/yolov5/blob/master/train.py)`和`[detect.py](https://github.com/ultralytics/yolov5/blob/master/detect.py)`文件非常棒，所以我们把它们放在了这个帖子里。

在本教程中，我们学习了如何在任何光线条件下检测不同的物体，结合热视觉和深度学习，使用 CNN YOLOv5 架构和我们的自定义热成像数据集。

为此，我们发现了如何在 FLIR Thermal Starter 数据集上训练最先进的 YOLOv5 模型，该模型先前是使用 Microsoft COCO 数据集训练的。

尽管热图像与 COCO 数据集的常见 RGB 图像完全不同，但获得的出色性能和结果显示了 YOLOv5 模型的强大功能。

我们可以得出结论，人工智能如今经历了令人难以置信的有用范式。

本教程向您展示了如何在实际应用中应用热视觉和深度学习(例如，自动驾驶汽车)。如果你想了解这个令人敬畏的话题，请查看 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)的**自动驾驶汽车**课程。

PyImageSearch 团队希望您已经喜欢并深入理解了本**红外视觉基础**课程中教授的所有概念。

下节课再见！

* * *

### [**引用信息**](#TOC)

**Garcia-Martin，R.** “热视觉:用 PyTorch 和 YOLOv5 进行夜间目标探测”(真实项目)， *PyImageSearch* ，P. Chugh，A. R. Gosthipaty，S. Huot，K. Kidriavsteva 和 R. Raha 编辑。，2022 年，【https://pyimg.co/p2zsm 

```py
@incollection{RGM_2022_PYTYv5,
  author = {Raul Garcia-Martin},
  title = {Thermal Vision: Night Object Detection with {PyTorch} and {YOLOv5} (real project)},
  booktitle = {PyImageSearch},
  editor = {Puneet Chugh and Aritra Roy Gosthipaty and Susan Huot and Kseniia Kidriavsteva and Ritwik Raha},
  year = {2022},
  note = {https://pyimg.co/p2zsm},
}
```

* * *

* * *

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****