# 基于 OpenCV、Python 和深度学习的人脸识别

> 原文：<https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/>

最后更新于 2022 年 12 月 30 日，内容更新。

在今天的博文中，你将学习如何使用以下工具在图像和视频流中执行**人脸识别**:

*   OpenCV
*   计算机编程语言
*   深度学习

正如我们将看到的，我们今天在这里使用的基于深度学习的面部嵌入既(1) *高度准确*又(2)能够在*实时执行*。

**要用 [OpenCV](https://opencv.org/) 、 [Python](https://www.python.org/) 、[深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)、*了解更多关于人脸识别的知识，就继续阅读吧！***

*   【2021 年 7 月更新:增加了备选人脸识别方法部分，包括基于深度学习和非基于深度学习的方法。

## 基于 OpenCV、Python 和深度学习的人脸识别

在本教程中，您将学习如何使用 OpenCV、Python 和深度学习来执行面部识别。

我们将从简单讨论基于深度学习的面部识别如何工作开始，包括“深度度量学习”的概念

在那里，我将帮助您安装实际执行人脸识别所需的库。

最后，我们将实现静态图像和视频流的人脸识别。

我们将会发现，我们的人脸识别实现将能够实时运行。

### 理解深度学习人脸识别嵌入

那么，深度学习+人脸识别是如何工作的呢？

**秘密是一种叫做*深度度量学习的技术。***

如果你以前有深度学习的经验，你会知道我们通常训练一个网络来:

*   接受单个输入图像
*   并输出图像分类/标签

然而，深度度量学习是不同的。

我们不是试图输出单个标签(或者甚至是图像中对象的坐标/边界框)，而是输出一个实值特征向量。

对于 dlib 人脸识别网络，输出的特征向量是 **128-d** (即 128 个实数值的列表)，用于*量化人脸*。使用**三元组**来训练网络:

这里我们向网络提供三个图像:

*   这些图像中的两个是同一个人的示例面部。
*   第三张照片是从我们的数据集中随机选择的一张脸，与另外两张照片中的*不是*同一个人。

作为一个例子，让我们再次考虑图 1，其中我们提供了三张图片:一张查德·史密斯的图片和两张威尔·法瑞尔的图片。

我们的网络对人脸进行量化，为每个人脸构建 128 维嵌入(量化)。

从那里，总的想法是，我们将调整我们的神经网络的权重，这样两个威尔·费雷尔的 128-d 测量值将更接近彼此的 T2，而远离查德·史密斯的测量值。

我们用于人脸识别的网络架构是基于来自何等人的论文 [*【用于图像识别的深度残差学习】*](https://arxiv.org/abs/1512.03385)中的 ResNet-34，但是具有更少的层并且滤波器的数量减少了一半。

网络本身由 **[戴维斯·金](https://pyimagesearch.com/2017/03/13/an-interview-with-davis-king-creator-of-the-dlib-toolkit/)** 在大约 300 万张图像的数据集上训练。在[标记的野外人脸(LFW)](http://vis-www.cs.umass.edu/lfw/) 数据集上，该网络与其他最先进的方法进行了比较，达到了 **99.38%的准确率**。

戴维斯·金( [dlib](http://dlib.net/) 的创造者)和 **[亚当·盖特吉](https://adamgeitgey.com/)**([面部识别模块](https://github.com/ageitgey/face_recognition)的作者，我们很快就会用到)都写了关于基于深度学习的面部识别如何工作的详细文章:

*   [](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html)*(Davis)深度度量学习的高质量人脸识别*
*   *[深度学习的现代人脸识别](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)(亚当)*

 *我强烈建议你阅读上面的文章，了解更多关于深度学习面部嵌入如何工作的细节。

### 安装您的人脸识别库

为了使用 Python 和 OpenCV 执行人脸识别，我们需要安装两个额外的库:

*   [dlib](http://dlib.net/)
*   [人脸识别](https://github.com/ageitgey/face_recognition)

dlib 库由 [Davis King](https://pyimagesearch.com/2017/03/13/an-interview-with-davis-king-creator-of-the-dlib-toolkit/) 维护，包含我们的“深度度量学习”实现，该实现用于构建我们的人脸嵌入，用于实际的识别过程。

由[亚当·盖特基](https://adamgeitgey.com/)创建的`face_recognition`库将*包裹在* dlib 的面部识别功能周围，使其更容易使用。

[Learn from Adam Geitgey and Davis King at *PyImageConf* 2018](https://www.pyimageconf.com)

我假设您的系统上已经安装了 **OpenCV。如果没有，不用担心——只需访问我的 [OpenCV 安装教程](https://pyimagesearch.com/opencv-tutorials-resources-guides/)页面，并遵循适合您系统的指南。**

从那里，让我们安装`dlib`和`face_recognition`包。

***注意:*** *对于以下安装，如果您正在使用 Python 虚拟环境，请确保您处于其中。我强烈推荐使用虚拟环境来隔离您的项目——这是 Python 的最佳实践。如果你遵循了我的 OpenCV 安装指南(并安装了`virtualenv` + `virtualenvwrapper`，那么你可以在安装`dlib`和`face_recognition`之前使用`workon`命令。*

#### 安装`dlib` *没有* GPU 支持

如果你没有 GPU，你可以按照本指南使用 pip 通过[安装`dlib`:](https://pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/)

```py
$ workon # optional
$ pip install dlib

```

或者您可以从源代码编译:

```py
$ workon <your env name here> # optional
$ git clone https://github.com/davisking/dlib.git
$ cd dlib
$ mkdir build
$ cd build
$ cmake .. -DUSE_AVX_INSTRUCTIONS=1
$ cmake --build .
$ cd ..
$ python setup.py install --yes USE_AVX_INSTRUCTIONS

```

#### 使用 GPU 支持安装`dlib` *(可选)*

如果你*有*兼容 CUDA 的 GPU，你可以安装有 GPU 支持的`dlib`，让面部识别更快更有效。

为此，我建议从源代码安装`dlib`,因为您将对构建有更多的控制权:

```py
$ workon <your env name here> # optional
$ git clone https://github.com/davisking/dlib.git
$ cd dlib
$ mkdir build
$ cd build
$ cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
$ cmake --build .
$ cd ..
$ python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA

```

#### 安装`face_recognition`包

[人脸识别模块](https://github.com/ageitgey/face_recognition)可通过简单的 pip 命令安装:

```py
$ workon <your env name here> # optional
$ pip install face_recognition

```

#### 安装`imutils`

你还需要我的便利功能包。您可以通过 pip 将它安装在您的 Python 虚拟环境中:

```py
$ workon <your env name here> # optional
$ pip install imutils

```

### 我们的人脸识别数据集

由于 *[《侏罗纪公园》](https://www.imdb.com/title/tt0107290/)* (1993)是我一直以来最喜欢的电影，为了纪念 *[《侏罗纪世界:堕落王国》](https://www.imdb.com/title/tt4881806/)* (2018)于本周五在美国上映，我们将对电影中的角色样本应用人脸识别:

*   [爱伦·格兰特](http://jurassicpark.wikia.com/wiki/Alan_Grant)，*古生物学家* (22 张图片)
*   [克莱尔·迪林](http://jurassicpark.wikia.com/wiki/Claire_Dearing)，*公园运营经理* (53 张图片)
*   [爱丽·萨特勒](http://jurassicpark.wikia.com/wiki/Ellie_Sattler)，*古植物学家* (31 张图片)
*   [伊恩·马尔科姆](http://jurassicpark.wikia.com/wiki/Ian_Malcolm)，*数学家* (41 张图片)
*   [约翰·哈蒙德](http://jurassicpark.wikia.com/wiki/John_Hammond)，*商人/侏罗纪公园主人* (36 张图片)
*   [欧文·格雷迪](http://jurassicpark.wikia.com/wiki/Owen_Grady)，*恐龙研究员* (35 张图片)

这个数据集是在< 30 minutes using the method discussed in my *[如何(快速)构建深度学习图像数据集](https://pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/)* 教程*中构建的。*给定这个图像数据集，我们将:

*   为数据集中的每个面创建 128 维嵌入
*   使用这些嵌入来识别图像和视频流中的人物面部

### 人脸识别项目结构

通过检查`tree`命令的输出可以看到我们的项目结构:

```py
$ tree --filelimit 10 --dirsfirst
.
├── dataset
│   ├── alan_grant [22 entries]
│   ├── claire_dearing [53 entries]
│   ├── ellie_sattler [31 entries]
│   ├── ian_malcolm [41 entries]
│   ├── john_hammond [36 entries]
│   └── owen_grady [35 entries]
├── examples
│   ├── example_01.png
│   ├── example_02.png
│   └── example_03.png
├── output
│   └── lunch_scene_output.avi
├── videos
│   └── lunch_scene.mp4
├── search_bing_api.py
├── encode_faces.py
├── recognize_faces_image.py
├── recognize_faces_video.py
├── recognize_faces_video_file.py
└── encodings.pickle

10 directories, 11 files

```

我们的项目有 **4 个顶级目录:**

*   `dataset/`:包含六个角色的面部图像，根据他们各自的名字组织到子目录中。
*   `examples/`:在数据集中有三张*不是*的人脸图像进行测试。
*   `output/`:这是你可以存储处理过的人脸识别视频的地方。我会把我的一个留在文件夹里——原*侏罗纪公园*电影中的经典“午餐场景”。
*   `videos/`:输入的视频应该保存在这个文件夹中。这个文件夹还包含“午餐场景”视频，但它还没有经过我们的人脸识别系统。

我们在根目录下还有 6 个文件:

*   第一步是建立一个数据集(我已经为你完成了)。要了解如何使用 Bing API 用我的脚本构建数据集，只需查看[这篇博文](https://pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/)。
*   `encode_faces.py`:脸部的编码(128 维向量)是用这个脚本构建的。
*   `recognize_faces_image.py`:识别单幅图像中的人脸(基于数据集中的编码)。
*   `recognize_faces_video.py`:通过网络摄像头识别实时视频流中的人脸，并输出视频。
*   `recognize_faces_video_file.py`:识别驻留在磁盘上的视频文件中的人脸，并将处理后的视频输出到磁盘。我今天不会讨论这个文件，因为骨骼和视频流文件来自同一个骨骼。
*   `encodings.pickle`:面部识别编码通过`encode_faces.py`从你的数据集生成，然后序列化到磁盘。

在创建了一个图像数据集(用`search_bing_api.py`)之后，我们将运行`encode_faces.py`来构建嵌入。

在那里，我们将运行识别脚本来实际识别人脸。

### 使用 OpenCV 和深度学习对人脸进行编码

在我们能够识别图像和视频中的人脸之前，我们首先需要量化训练集中的人脸。请记住，我们在这里实际上不是在训练网络— **网络已经被*训练过*在`~3`百万张图像的数据集上创建 128 维嵌入**。

我们当然可以从零开始训练一个网络，甚至微调现有模型的权重，但这对许多项目来说很可能是矫枉过正。此外，你还需要大量*图片来从头开始训练网络。*

相反，使用预训练的网络，然后使用它为我们数据集中的 218 个人脸中的每一个构建 128-d 嵌入会更容易。

然后在分类的时候，我们可以用一个简单的 k-NN 模型+投票来做最终的人脸分类。这里也可以使用其他传统的机器学习模型。

为了构建我们的人脸嵌入，打开与这篇博文相关的 ***【下载】***`encode_faces.py`:

```py
# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

```

首先，我们需要导入所需的包。再次注意，这个脚本需要安装`imutils`、`face_recognition`和 OpenCV。向上滚动到“*安装您的面部识别库”*以确保您已经准备好在您的系统上运行这些库。

让我们用`argparse`来处理运行时处理的[命令行参数](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/):

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

```

如果你是 PyImageSearch 的新手，让我把你的注意力引向上面的代码块，随着你阅读我的博客文章越来越多，你会越来越熟悉它。我们使用`argparse`来解析命令行参数。当您在命令行中运行 Python 程序时，您可以在不离开终端的情况下向脚本提供附加信息。**第 10-17 行**不需要修改，因为它们解析来自终端的输入。如果这些命令行看起来不熟悉，可以看看我关于[命令行参数](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/)的博文。

让我们列出参数标志并进行讨论:

*   `--dataset`:我们数据集的路径(我们用上周博文的[方法#2 中描述的`search_bing_api.py`创建了一个数据集)。](https://pyimagesearch.com/2018/06/11/how-to-build-a-custom-face-recognition-dataset/)
*   我们的面部编码被写到这个参数指向的文件中。
*   `--detection-method`:在我们能够*编码图像中的*张脸之前，我们首先需要*检测*张脸。或者两种人脸检测方法包括`hog`或`cnn`。这两个标志是唯一适用于`--detection-method`的标志。

现在我们已经定义了参数，让我们获取数据集中文件的路径(并执行两次初始化):

```py
# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

```

**第 21 行**使用我们的输入数据集目录的路径来构建其中包含的所有`imagePaths`的列表。

我们还需要在循环之前初始化两个列表，分别是`knownEncodings`和`knownNames`。这两个列表将包含数据集中每个人的面部编码和相应的姓名(**第 24 行和第 25 行**)。

是时候开始循环我们的*侏罗纪公园*角色脸了！

```py
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from BGR (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

```

该循环将循环 218 次，对应于数据集中的 218 张人脸图像。我们将遍历第 28 行上每个图像的路径。

从那里，我们将从第 32 行的`imagePath`(我们的子目录被恰当地命名)中提取这个人的`name`。

然后让我们装载`image`，同时将`imagePath`传递到`cv2.imread` ( **线 36** )。

OpenCV 在 BGR 订购颜色通道，但是`dlib`实际上需要 RGB。`face_recognition`模块使用`dlib`，所以在我们继续之前，让我们交换一下**第 37 行**的颜色空间，命名新图像为`rgb`。

接下来，让我们本地化面部和计算机编码:

```py
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

```

这是剧本有趣的地方！

对于循环的每一次迭代，我们将检测一张脸(或者可能是多张脸，并假设在图像的多个位置都是同一个人——这一假设在您自己的图像中可能成立，也可能不成立，因此在这里要小心)。

例如，假设`rgb`包含一张(或多张)爱丽·萨特勒的脸。

**第 41 行和第 42 行**实际上找到/定位了她的脸部，从而得到一个脸部列表`boxes`。我们向`face_recognition.face_locations`方法传递两个参数:

*   `rgb`:我们的 RGB 图像。
*   `model`:或者是`cnn`或者是`hog`(这个值包含在与`"detection_method"`键相关的命令行参数字典中)。CNN 方法更准确，但速度较慢。HOG 更快，但不太准确。

然后，我们将把爱丽·萨特勒脸部的边界`boxes`转换成在**第 45 行**的 128 个数字的列表。这就是所谓的*将面部编码*成一个矢量，然后`face_recognition.face_encodings`方法为我们处理它。

从那里我们只需要将爱丽·萨特勒`encoding`和`name`添加到适当的列表中(`knownEncodings`和`knownNames`)。

我们将对数据集中的所有 218 幅图像继续这样做。

除非我们可以在另一个处理识别的脚本中使用`encodings`,否则编码图像的意义何在？

现在让我们来解决这个问题:

```py
# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
```

**第 56 行**用两个键`"encodings"`和`"names"`构造一个字典。

从那里开始**第 57-59 行**将名称和编码转储到磁盘，以备将来调用。

**我应该如何在终端中运行`encode_faces.py`脚本？**

要创建我们的面部嵌入，打开一个终端并执行以下命令:

```py
$ python encode_faces.py --dataset dataset --encodings encodings.pickle
[INFO] quantifying faces...
[INFO] processing image 1/218
[INFO] processing image 2/218
[INFO] processing image 3/218
...
[INFO] processing image 216/218
[INFO] processing image 217/218
[INFO] processing image 218/218
[INFO] serializing encodings...
$ ls -lh encodings*
-rw-r--r--@ 1 adrian  staff   234K May 29 13:03 encodings.pickle
```

正如您从我们的输出中看到的，我们现在有一个名为`encodings.pickle`的文件—该文件包含我们数据集中每个人脸的 128-d 人脸嵌入。

在我的 Titan X GPU 上，处理整个数据集需要一分多钟，但如果你使用 CPU， ***请准备好等待脚本完成！***

在我的 Macbook Pro(无 GPU)上，编码 218 张图像需要**21 分 20 秒**。

如果你有一个 GPU 和有 GPU 支持的编译 dlib，你应该期待更快的速度。

### 识别图像中的人脸

现在，我们已经为数据集中的每张图像创建了 128 维人脸嵌入，我们现在可以使用 OpenCV、Python 和深度学习来识别图像中的人脸。

打开`recognize_faces_image.py`并插入以下代码(或者更好的是，从这篇文章底部的**的“下载”部分获取与这篇博客文章相关的文件和图像数据，并跟随其后):**

```py
# import the necessary packages
import face_recognition
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

```

这个脚本在第 2-5 行只需要四次导入。模块将完成繁重的工作，OpenCV 将帮助我们加载、转换和显示处理后的图像。

我们将解析第 8-15 行**的三个命令行参数:**

*   `--encodings`:包含我们面部编码的 pickle 文件的路径。
*   这是正在接受面部识别的图像。
*   `--detection-method`:现在你应该对这个方法很熟悉了——根据你的系统的能力，我们将使用`hog`或`cnn`方法。对于速度，选择`hog`，对于精度，选择`cnn`。

***重要！**如果你是:*

1.  *在**CPU**T3 上运行人脸识别代码*
2.  ***或者**你用的是*
***   *…你会想要将`--detection-method`设置为`hog`，因为 CNN 人脸检测器(1)在没有 GPU 的情况下速度很慢，并且(Raspberry Pi 也没有足够的内存来运行 CNN。***

 **从这里，让我们加载预先计算的编码+人脸名称，然后为输入图像构造 128-d 人脸编码:

```py
# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the input image and convert it from BGR to RGB
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb,
	model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

# initialize the list of names for each face detected
names = []

```

**第 19 行**从磁盘加载我们的酸洗编码和面孔名称。在实际的人脸识别步骤中，我们将需要这些数据。

然后，在第 22 和 23 行的**上，我们加载并转换输入`image`到`rgb`颜色通道排序(就像我们在`encode_faces.py`脚本中所做的)。**

然后，我们继续检测输入图像中的所有人脸，并在第 29-31 行的**上计算它们的 128-d `encodings`(这些行看起来也应该很熟悉)。**

现在是为每个检测到的人脸初始化一个`names`列表的好时机——这个列表将在下一步中填充。

接下来，让我们循环面部表情`encodings`:

```py
# loop over the facial embeddings
for encoding in encodings:
	# attempt to match each face in the input image to our known
	# encodings
	matches = face_recognition.compare_faces(data["encodings"],
		encoding)
	name = "Unknown"

```

在第 37 行的**上，我们开始遍历从输入图像中计算出的人脸编码。**

***然后面部识别魔法就发生了！***

我们尝试使用`face_recognition.compare_faces` ( **第 40 行和第 41 行**)将输入图像(`encoding`)中的每个人脸与我们已知的编码数据集(保存在`data["encodings"]`)进行匹配。

这个函数返回一个由`True` / `False`值组成的列表，数据集中的每张图片对应一个值。对于我们的*侏罗纪公园*的例子，数据集中有 218 张图像，因此返回的列表将有 218 个布尔值。

在内部，`compare_faces`函数计算候选嵌入和数据集中所有人脸之间的欧几里德距离:

*   如果距离低于某个容差(容差越小，我们的面部识别系统将越严格)，那么我们返回`True`，**指示面部匹配。**
*   否则，如果距离高于公差阈值，我们返回`False`作为**面不匹配。**

本质上，我们正在利用一个“更奇特”的 k-NN 模型进行分类。更多细节请参考 [compare_faces 实现](https://github.com/ageitgey/face_recognition/blob/master/face_recognition/api.py#L213)。

`name`变量将最终保存这个人的姓名字符串——现在，我们把它保留为`"Unknown"`,以防没有“投票”(**第 42 行**)。

给定我们的`matches`列表，我们可以计算每个名字的“投票”数量(与每个名字相关联的`True`值的数量)，合计投票，并选择具有最多相应投票的人的名字:

```py
	# check to see if we have found a match
	if True in matches:
		# find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		# loop over the matched indexes and maintain a count for
		# each recognized face face
		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1

		# determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
		name = max(counts, key=counts.get)

	# update the list of names
	names.append(name)

```

如果`matches` ( **行 45** )中有`True`票，我们需要确定`matches`中这些`True`值所在的*索引*。我们在第 49 行**上做了同样的事情，在那里我们构造了一个简单的`matchedIdxs`列表，对于`example_01.png`可能是这样的:**

```py
(Pdb) matchedIdxs
[35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75]

```

然后我们初始化一个名为`counts`的字典，它将保存角色名作为*键*票数作为*值* ( **第 50 行**)。

从这里开始，让我们遍历`matchedIdxs`并设置与每个名字相关联的值，同时根据需要在`counts`中递增该值。对于伊恩·马尔科姆的高票，T2 字典可能是这样的:

```py
(Pdb) counts
{'ian_malcolm': 40}

```

回想一下，我们在数据集中只有 41 张 Ian 的照片，所以没有其他人投票的 40 分是非常高的。

**第 61 行**从`counts`中抽取票数最多的名字，在这种情况下，应该是`'ian_malcolm'`。

主面部编码循环的第二次迭代(因为在我们的示例图像中有两个人脸)为`counts`产生如下结果:

```py
(Pdb) counts
{'alan_grant': 5}

```

这绝对是一个较小的投票分数，但字典中只有一个名字，所以我们很可能找到了爱伦·格兰特。

***注:****PDB Python 调试器用于验证`counts`字典的值。PDB 的用法超出了这篇博客的范围；然而，你可以在 [Python 文档页面](https://docs.python.org/3/library/pdb.html)上找到如何使用它。*

如下面的**图 5** 所示，伊恩·马尔科姆和爱伦·格兰特都已被正确识别，因此脚本的这一部分运行良好。

让我们继续，循环每个人的边界框和标签名称，并出于可视化目的将它们绘制在输出图像上:

```py
# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

```

在第 67 行的**上，我们开始循环检测到的脸部边界`boxes`和预测的`names`。为了创建一个 iterable 对象，这样我们就可以很容易地遍历这些值，我们调用`zip(boxes, names)`产生元组，我们可以从中提取盒子坐标和名称。**

我们使用框坐标在第 69 行的**上画一个绿色矩形。**

我们还使用坐标来计算应该在哪里绘制人名文本(**行 70** )，然后实际将姓名文本放置在图像上(**行 71 和 72** )。如果人脸包围盒在图像的最顶端，我们需要将文本移动到包围盒顶端的下方(在**第 70 行**处理)，否则，文本会被截掉。

然后我们继续显示图像，直到按下一个键(**行 75 和 76** )。

**您应该如何运行面部识别 Python 脚本？**

使用您的终端，首先使用`workon`命令确保您处于各自的 Python 正确的虚拟环境中(当然，如果您使用的是虚拟环境)。

然后运行脚本，同时至少提供两个[命令行参数](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/)。如果选择使用 HoG 方法，也一定要通过`--detection-method hog`(否则会默认为深度学习检测器)。

让我们去吧！

要使用 OpenCV 和 Python 识别人脸，请打开您的终端并执行我们的脚本:

```py
$ python recognize_faces_image.py --encodings encodings.pickle \
	--image examples/example_01.png
[INFO] loading encodings...
[INFO] recognizing faces...

```

第二个人脸识别示例如下:

```py
$ python recognize_faces_image.py --encodings encodings.pickle \
	--image examples/example_03.png
[INFO] loading encodings...
[INFO] recognizing faces...

```

### 视频中的人脸识别

既然我们已经对*图像*应用了人脸识别，那么让我们也对*视频*(实时)应用人脸识别。

***重要性能提示:****CNN 人脸识别器只应在使用 GPU 的情况下实时使用(可以与 CPU 一起使用，但预计低于 0.5 FPS，这会导致视频断断续续)。或者(您正在使用 CPU)，您应该使用 HoG 方法(或者甚至 OpenCV Haar cascades，将在以后的博客文章中介绍)并期望足够的速度。*

下面的脚本与前面的`recognize_faces_image.py`脚本有许多相似之处。因此，我将轻松跳过我们已经介绍过的内容，只回顾视频组件，以便您了解正在发生的事情。

一旦你抓取了 ***【下载】*** ，打开`recognize_faces_video.py`并跟随:

```py
# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

```

我们在第 2-8 行**导入包，然后在第 11-20** 行**解析我们的命令行参数。**

我们有四个命令行参数，其中两个您应该从上面就知道了(`--encodings`和`--detection-method`)。另外两个论点是:

*   `--output`:输出视频的路径。
*   `--display`:指示脚本将帧显示到屏幕上的标志。值为`1`的显示器和值为`0`的显示器不会将输出帧显示到我们的屏幕上。

从那里我们将加载我们的编码并开始我们的`VideoStream`:

```py
# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

```

为了访问我们的相机，我们使用了来自 [imutils](https://github.com/jrosebr1/imutils/blob/master/imutils/video/videostream.py) 的`VideoStream`类。**第 29 行**开始播放。如果您的系统上有多个摄像头(如内置网络摄像头和外部 USB 摄像头)，您可以将`src=0`更改为`src=1`，以此类推。

稍后我们将可选地将处理过的视频帧写入磁盘，所以我们初始化`writer`到`None` ( **第 30 行**)。整整两秒钟的睡眠让我们的相机预热(**第 31 行**)。

从那里我们将开始一个`while`循环，并开始抓取和处理帧:

```py
# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

```

我们的循环从第 34 行的**开始，我们采取的第一步是从视频流中抓取一个`frame`(**第 36 行**)。**

以上代码块中剩余的**行 40-50** 与之前脚本中的行几乎相同，只是这是一个视频帧而不是静态图像。本质上，我们读取`frame`，进行预处理，然后检测人脸边界`boxes` +计算每个边界框的`encodings`。

接下来，让我们循环一下与我们刚刚找到的面部相关联的面部`encodings`:

```py
	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)

		# update the list of names
		names.append(name)

```

在这个代码块中，我们循环遍历每个`encodings`并尝试匹配人脸。如果找到匹配，我们计算数据集中每个名字的投票数。然后，我们提取最高票数，这就是与该脸相关联的名字。这些台词和我们之前看过的剧本*一模一样*，所以让我们继续。

在下一个块中，我们循环遍历已识别的人脸，并继续在人脸周围绘制一个框，并在人脸上方显示该人的姓名:

```py
	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

```

这些行也是相同的，所以让我们把重点放在与视频相关的代码上。

可选地，我们将把帧写入磁盘，所以让我们看看 [**如何使用 OpenCV**](https://pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/) 将视频写入磁盘:

```py
	# if the video writer is None *AND* we are supposed to write
	# the output video to disk initialize the writer
	if writer is None and args["output"] is not None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 20,
			(frame.shape[1], frame.shape[0]), True)

	# if the writer is not None, write the frame with recognized
	# faces to disk
	if writer is not None:
		writer.write(frame)

```

假设我们在命令行参数中提供了一个输出文件路径，并且我们还没有初始化一个视频`writer` ( **第 99 行**)，让我们继续初始化它。

在**行 100** ，我们初始化`VideoWriter_fourcc`。 [FourCC](https://www.fourcc.org/) 是一个 4 字符代码，在我们的例子中，我们将使用“MJPG”4 字符代码。

从那里，我们将把该对象连同我们的输出文件路径、每秒帧数目标和帧尺寸一起传递到`VideoWriter`(**第 101 行和第 102 行**)。

最后，如果`writer`存在，我们可以继续向磁盘写入一个帧(**第 106-107 行**)。

让我们来决定是否应该在屏幕上显示人脸识别视频帧:

```py
	# check to see if we are supposed to display the output frame to
	# the screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

```

如果设置了显示命令行参数，我们将继续显示该帧(**第 112 行**)并检查退出键(`"q"`)是否已被按下(**第 113-116 行**)，此时我们将`break`从循环中退出(**第 117 行**)。

最后，让我们履行家务职责:

```py
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()

```

在**第 120-125 行**中，我们清理并释放显示器、视频流和视频写入器。

你准备好看剧本了吗？

要使用 OpenCV 和 Python 演示**实时人脸识别，打开一个终端并执行以下命令:**

```py
$ python recognize_faces_video.py --encodings encodings.pickle \
	--output output/webcam_face_recognition_output.avi --display 1
[INFO] loading encodings...
[INFO] starting video stream...

```

下面您可以找到我录制的演示面部识别系统运行的输出示例视频:

<https://www.youtube.com/embed/dCKl4oGP69s?feature=oembed>*****