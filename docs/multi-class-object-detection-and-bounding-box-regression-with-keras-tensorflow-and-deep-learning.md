# 使用 Keras、TensorFlow 和深度学习的多类对象检测和包围盒回归

> 原文：<https://pyimagesearch.com/2020/10/12/multi-class-object-detection-and-bounding-box-regression-with-keras-tensorflow-and-deep-learning/>

在本教程中，您将学习如何通过 Keras 和 TensorFlow 深度学习库使用边界框回归来训练自定义多类对象检测器。

上周的教程涵盖了[如何使用包围盒回归](https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/)训练*单类*物体检测器。**今天，我们将扩展我们的包围盒回归方法来处理*多个类。***

为了用 Keras 和 TensorFlow 从头开始创建一个多类对象检测器，我们需要修改我们架构的网络头。操作顺序如下:

*   **步骤#1:** 取 VGG16(预先在 ImageNet 上训练好的)*取下*全连接(FC)层头
*   **步骤#2:** 构造一个带有两个分支的*新* FC 层头:
    *   **分支#1:** 一系列 FC 层，以具有(1)四个神经元的层结束，对应于预测边界框的左上和右下 *(x，y)*-( 2)sigmoid 激活函数，使得每四个神经元的输出位于范围*【0，1】*内。这个分支负责*包围盒*的预测。
    *   **分支#2:** 另一个 FC 层系列，但这一个在最后有一个 softmax 分类器。**这个分支负责*类标签*的预测。**
*   **步骤#3:** 将新的 FC 层头(带有两个分支)放在 VGG16 主体的顶部
*   **步骤#4:** 微调整个网络以进行端到端对象检测

结果将是一个卷积神经网络在您自己的*自定义数据集*上训练/微调，用于对象检测！

让我们开始吧。

**要了解如何使用 Keras/TensorFlow 通过包围盒回归来训练自定义多类对象检测器，*继续阅读。***

## **使用 Keras、TensorFlow 和深度学习进行多类对象检测和包围盒回归**

在本教程的第一部分，我们将简要讨论单类对象检测和多类对象检测之间的区别。

然后，我们将回顾我们将在其上训练多类对象检测器的数据集，以及我们项目的目录结构。

从这里，我们将实现两个 Python 脚本:

1.  一个是加载数据集，构建模型架构，然后训练多类对象检测器
2.  第二个脚本将从磁盘加载我们训练过的对象检测器，然后使用它对测试图像进行预测

这是一个更高级的教程，我认为以下教程是本指南的*先决条件*和*必读材料*:

1.  *[Keras，回归，CNN](https://pyimagesearch.com/2019/01/28/keras-regression-and-cnns/)*
2.  *[Keras:多输出多损耗](https://pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/)*
3.  *[用 Keras 和深度学习进行微调](https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/)*
4.  *[R-CNN 物体检测用 Keras、TensorFlow、深度学习](https://pyimagesearch.com/2020/07/13/r-cnn-object-detection-with-keras-tensorflow-and-deep-learning/)*
5.  *[物体检测:使用 Keras、TensorFlow、深度学习的包围盒回归](https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/)* (上周教程)

在继续之前，请务必阅读上述教程。

### **多类物体检测与单类物体检测有何不同？**

多类对象检测，顾名思义，意味着我们试图(1)检测输入图像中的*处的*对象，以及(2)预测*检测到的对象是什么*。

例如，下面的**图 1** 显示我们正在尝试检测属于*【飞机】**【人脸】*或*【摩托车】*类别的对象:

另一方面，单类对象检测是多类对象检测的简化形式，因为我们已经知道*对象是什么*(因为根据定义只有一个类，在这种情况下是*“飞机”*)，只需检测*对象在输入图像中的位置*就足够了:

与仅需要回归层头来预测边界框的单类对象检测器不同，多类对象检测器需要具有*个分支的全连接层头:*

*   **分支#1:** 一个回归层集合，就像在单类对象检测情况下一样
*   **分支#2:** 一个附加层集，这个层集带有一个 softmax 分类器，用于预测*类标签*

**一起使用，我们的多类物体检测器的单次向前通过将导致:**

1.  图像中对象的预测边界框坐标
2.  图像中对象的预测类别标签

今天，我将向您展示如何使用包围盒回归来训练您自己的自定义多类对象检测器。

### **我们的多类对象检测和包围盒回归数据集**

我们今天在这里使用的示例数据集是 CALTECH-101 数据集的子集，可用于训练对象检测模型。

具体来说，我们将使用以下类:

*   **飞机:** 800 张图片
*   **脸:** 435 张图片
*   **摩托车:** 798 张图片

总的来说，我们的数据集由 2033 个图像和它们相应的边界框 *(x，y)*-坐标组成。在这一部分的顶部，我已经在**图 3** 中包含了每个职业的可视化。

我们的目标是训练一个能够准确预测输入图像中的*飞机*、*人脸*和*摩托车*的包围盒坐标的物体检测器。

***注意:**没有必要从加州理工学院 101 的网站上下载完整的数据集。在与本教程相关的下载中，我包含了我们的样本数据集，包括一个边界框的 CSV 文件。*

### 配置您的开发环境

要针对本教程配置您的系统，我建议您遵循以下任一教程:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

这两个教程都将帮助您在一个方便的 Python 虚拟环境中，用这篇博文所需的所有软件来配置您的系统。

**这样说来，你是:**

*   时间紧迫？
*   在雇主管理锁定的笔记本电脑上学习？
*   想要跳过与包管理器、bash/ZSH 概要文件和虚拟环境的争论吗？
*   准备好运行代码*了吗*(并尽情体验它)？

那今天就加入 [PyImageSearch 加](https://pyimagesearch.com/pyimagesearch-plus/)吧！在你的浏览器——*无需安装* 中获取运行在**谷歌的 Colab 生态系统*上的 PyImageSearch 教程 **Jupyter 笔记本**。***

最棒的是，这些笔记本可以在 Windows、macOS 和 Linux 上运行！

### **项目结构**

去拿吧。本教程的 ***【下载】*** 部分的 zip 文件。在里面，您将找到数据子集以及我们的项目文件:

```py
$ tree --dirsfirst --filelimit 20
.
├── dataset
│   ├── annotations
│   │   ├── airplane.csv
│   │   ├── face.csv
│   │   └── motorcycle.csv
│   └── images
│       ├── airplane [800 entries]
│       ├── face [435 entries]
│       └── motorcycle [798 entries]
├── output
│   ├── plots
│   │   ├── accs.png
│   │   └── losses.png
│   ├── detector.h5
│   ├── lb.pickle
│   └── test_paths.txt
├── pyimagesearch
│   ├── __init__.py
│   └── config.py
├── predict.py
└── train.py

9 directories, 12 files
```

```py
$ head -n 10 face.csv 
image_0001.jpg,251,15,444,300,face
image_0002.jpg,106,31,296,310,face
image_0003.jpg,207,17,385,279,face
image_0004.jpg,102,55,303,328,face
image_0005.jpg,246,30,446,312,face
image_0006.jpg,248,22,440,298,face
image_0007.jpg,173,25,365,302,face
image_0008.jpg,227,47,429,333,face
image_0009.jpg,116,27,299,303,face
image_0010.jpg,121,34,314,302,face
```

如您所见，每行包含六个元素:

1.  文件名
2.  起始*x*-坐标
3.  开始 *y* 坐标
4.  终点*x*-坐标
5.  终点*y*-坐标
6.  类别标签

*   `detector.h5`文件是我们训练的多类包围盒回归器。
*   然后我们有了`lb.pickle`，一个序列化的标签二进制化器，我们用它一次性编码类标签，然后将预测的类标签转换为人类可读的字符串。
*   最后，`test_paths.txt`文件包含我们测试图像的文件名。

我们有三个 Python 脚本:

*   `config.py`:配置设置和变量文件。
*   ``train.py`` :我们的训练脚本，它将从磁盘加载我们的图像和注释，为边界框回归修改 VGG16 架构，为对象检测微调修改后的架构，最后用我们的序列化模型、训练历史图和测试图像文件名填充`output/`目录。
*   ``predict.py`` :使用我们训练过的物体检测器进行推理。这个脚本将加载我们的序列化模型和标签编码器，遍历我们的测试图像，然后对每个图像应用对象检测。

让我们从实现配置文件开始。

### **创建我们的配置文件**

在实现我们的训练脚本之前，让我们首先定义一个简单的配置文件来存储重要的变量(即输出文件路径和模型训练超参数)——**这个配置文件将在我们的两个 Python 脚本中使用。**

打开`pyimagesearch`模块中的`config.py`文件，让我们看看里面有什么:

```py
# import the necessary packages
import os

# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = "dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])
```

Python 的`os`模块( **Line 2** )允许我们在配置文件中构建动态路径。我们的前两条路径源自`BASE_PATH` ( **线 6** ):

*   ``IMAGES_PATH`` :加州理工学院 101 图像子集的路径
*   ``ANNOTS_PATH`` :包含 CSV 格式的包围盒标注的文件夹路径

接下来，我们有四个与输出文件相关联的路径:

```py
# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output model, label binarizer, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "lb.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
```

最后，让我们定义我们的标准深度学习超参数:

```py
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32
```

我们的学习速度、训练次数和批量大小是通过实验确定的。这些参数存在于我们方便的配置文件中，这样当您在这里时，您可以根据自己的需要以及任何输入/输出文件路径轻松地对它们进行调整。

### **使用 Keras 和 TensorFlow 实现我们的多类对象检测器训练脚本**

实现了配置文件后，现在让我们继续创建训练脚本，该脚本用于使用边界框回归来训练多类对象检测器。

打开项目目录中的`train.py`文件，插入以下代码:

```py
# import the necessary packages
from pyimagesearch import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
```

既然我们的包、文件和方法已经导入，让我们初始化几个列表:

```py
# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []
```

第 25-28 行初始化与我们的数据相关的四个空列表；这些列表将很快包括:

*   `data`:图像
*   ``labels`` :类别标签
*   ``bboxes`` :目标包围盒 *(x，y)*-坐标
*   ``imagePaths`` :驻留在磁盘上的图像的文件路径

现在我们的列表已经初始化，在接下来的三个代码块中，我们将准备数据并填充这些列表，以便它们可以作为多类边界框回归训练的输入:

```py
# loop over all CSV files in the annotations directory
for csvPath in paths.list_files(config.ANNOTS_PATH, validExts=(".csv")):
	# load the contents of the current CSV annotations file
	rows = open(csvPath).read().strip().split("\n")

	# loop over the rows
	for row in rows:
		# break the row into the filename, bounding box coordinates,
		# and class label
		row = row.split(",")
		(filename, startX, startY, endX, endY, label) = row
```

```py
$ head -n 5 dataset/annotations/*.csv
==> dataset/annotations/airplane.csv <==
image_0001.jpg,49,30,349,137,airplane
image_0002.jpg,59,35,342,153,airplane
image_0003.jpg,47,36,331,135,airplane
image_0004.jpg,47,24,342,141,airplane
image_0005.jpg,48,18,339,146,airplane

==> dataset/annotations/face.csv <==
image_0001.jpg,251,15,444,300,face
image_0002.jpg,106,31,296,310,face
image_0003.jpg,207,17,385,279,face
image_0004.jpg,102,55,303,328,face
image_0005.jpg,246,30,446,312,face

==> dataset/annotations/motorcycle.csv <==
image_0001.jpg,31,19,233,141,motorcycle
image_0002.jpg,32,15,232,142,motorcycle
image_0003.jpg,30,20,234,143,motorcycle
image_0004.jpg,30,15,231,132,motorcycle
image_0005.jpg,31,19,232,145,motorcycle

```

在我们的循环中，我们对逗号分隔的`row` ( **第 39 行和第 40 行**)进行解包，给出 CSV 中特定行的`filename`、 *(x，y)*-坐标和类`label`。

接下来让我们使用这些值:

```py
		# derive the path to the input image, load the image (in
		# OpenCV format), and grab its dimensions
		imagePath = os.path.sep.join([config.IMAGES_PATH, label,
			filename])
		image = cv2.imread(imagePath)
		(h, w) = image.shape[:2]

		# scale the bounding box coordinates relative to the spatial
		# dimensions of the input image
		startX = float(startX) / w
		startY = float(startY) / h
		endX = float(endX) / w
		endY = float(endY) / h
```

最后，让我们加载图像并进行预处理:

```py
		# load the image and preprocess it
		image = load_img(imagePath, target_size=(224, 224))
		image = img_to_array(image)

		# update our list of data, class labels, bounding boxes, and
		# image paths
		data.append(image)
		labels.append(label)
		bboxes.append((startX, startY, endX, endY))
		imagePaths.append(imagePath)
```

尽管我们的数据准备循环已经完成，但我们仍有一些预处理任务要处理:

```py
# convert the data, class labels, bounding boxes, and image paths to
# NumPy arrays, scaling the input pixel intensities from the range
# [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# only there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well
if len(lb.classes_) == 2:
	labels = to_categorical(labels)
```

如果你对一键编码不熟悉，请参考我的 *[Keras 教程:如何入门 Keras、深度学习和 Python](https://pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/)* 或我的书 *[用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)* 中的解释和例子。

让我们继续划分我们的数据分割:

```py
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, labels, bboxes, imagePaths,
	test_size=0.20, random_state=42)

# unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

# write the testing image paths to disk so that we can use then
# when evaluating/testing our object detector
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()
```

使用 scikit-learn 的实用程序，我们将数据分成 80%用于训练，20%用于测试(**第 86 行和第 87 行**)。`split`数据通过列表切片经由**行 90-93** 被进一步解包。

出于评估目的，我们将在*预测*脚本中使用我们的测试图像路径，所以现在是将它们以文本文件形式导出到磁盘的好时机(**第 98-100 行**)。

唷！这就是数据准备——正如你所看到的，为深度学习准备图像数据集可能是乏味的，但如果我们想成为成功的计算机视觉和深度学习实践者，这是没有办法的。

**现在到了换挡到*准备我们的[多输出(二分支)模型](https://pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/)用于多类包围盒回归的时候了。*** 当我们建造模型时，我们将为[微调](https://pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/)做准备。我的建议是在单独的窗口打开[上周的教程](https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/)，这样你就可以并排看到*单类*和*多类*包围盒回归的区别。

事不宜迟，让我们准备我们的模型:

```py
# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False

# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
```

**第 103 行和第 104 行**用在 ImageNet 数据集上预先训练的权重加载 VGG16 网络。我们省略了全连接的层头(`include_top=False`)，因为我们将构建一个*新的*层头，负责多输出预测(即，类标签*和*边界框位置)。

**行 108** 冻结 VGG16 网络的主体，这样在微调过程中权重将*而不是*被更新。

然后我们展平网络的输出，这样我们就可以构建我们的新层 had，并将其添加到网络主体中(**行 111 和 112** )。

说到构建新的层头，让我们现在就做:

```py
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid",
	name="bounding_box")(bboxHead)

# construct a second fully-connected layer head, this one to predict
# the class label
softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax",
	name="class_label")(softmaxHead)

# put together our model which accept an input image and then output
# bounding box coordinates and a class label
model = Model(
	inputs=vgg.input,
	outputs=(bboxHead, softmaxHead))
```

利用 [TensorFlow/Keras 的功能 API](https://pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/) 、**我们构建了两个全新的分支。**

1.  对应于预测边界框的左上角和右上角的 *(x，y)* 坐标的`4`神经元。
2.  然后，我们使用一个`sigmoid`函数来确保我们的输出预测值在范围*【0，1】*内(因为我们在数据预处理步骤中将目标/地面真实边界框坐标缩放到这个范围)。

新的两个分支图层头的可视化如下所示:

请注意图层头是如何附加到 VGG16 的主体上，然后分裂成类别标签预测的分支*(左)*以及边界框 *(x，y)*-坐标预测*(右)。*

如果你之前从未创建过多输出神经网络，我建议你看看我的教程 *[Keras:多输出多损失。](https://pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/)*

下一步是定义我们的损失并编译模型:

```py
# define a dictionary to set the loss methods -- categorical
# cross-entropy for the class label head and mean absolute error
# for the bounding box head
losses = {
	"class_label": "categorical_crossentropy",
	"bounding_box": "mean_squared_error",
}

# define a dictionary that specifies the weights per loss (both the
# class label and bounding box outputs will receive equal weight)
lossWeights = {
	"class_label": 1.0,
	"bounding_box": 1.0
}

# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(lr=config.INIT_LR)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())
```

**第 140 行**定义了一个字典来存储我们的损失方法。我们将使用分类交叉熵作为我们的类标签分支，使用均方误差作为我们的包围盒回归头。

优化器初始化后，我们编译模型并在终端上显示模型架构的摘要(**第 155 行和第 156 行**)——我们将在本教程稍后执行`train.py`脚本时查看模型摘要的输出。

接下来，我们需要再定义两个字典:

```py
# construct a dictionary for our target training outputs
trainTargets = {
	"class_label": trainLabels,
	"bounding_box": trainBBoxes
}

# construct a second dictionary, this one for our target testing
# outputs
testTargets = {
	"class_label": testLabels,
	"bounding_box": testBBoxes
}
```

我们现在准备训练我们的多类包围盒回归器:

```py
# train the network for bounding box regression and class label
# prediction
print("[INFO] training model...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
	verbose=1)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")

# serialize the label binarizer to disk
print("[INFO] saving label binarizer...")
f = open(config.LB_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()
```

我们将`LabelBinarizer`序列化，以便在运行我们的`predict.py`脚本时，我们可以将预测的类标签转换回人类可读的字符串。

现在让我们构建一个图来可视化我们的总损失、类别标签损失(分类交叉熵)和边界框回归损失(均方误差)。

```py
# plot the total loss, label loss, and bounding box loss
lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(N, H.history[l], label=l)
	ax[i].plot(N, H.history["val_" + l], label="val_" + l)
	ax[i].legend()

# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plotPath = os.path.sep.join([config.PLOTS_PATH, "losses.png"])
plt.savefig(plotPath)
plt.close()
```

第 193 行定义了我们每笔损失的名称。然后我们构建一个有三行的图，每一行代表各自的损失(**第 195 行**)。

**第 198 行**在每个损失名称上循环。对于每个损失，我们绘制训练和验证损失结果(**第 200-206 行)。**

一旦我们构建了损失图，我们就构建了输出损失文件的路径，然后将它保存到磁盘上(**第 209-212 行)。**

最后一步是规划我们的培训和验证准确性:

```py
# create a new figure for the accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"],
	label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"],
	label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

# save the accuracies plot
plotPath = os.path.sep.join([config.PLOTS_PATH, "accs.png"])
plt.savefig(plotPath)
```

**第 215-224 行**描绘了我们的训练和训练期间验证数据的准确性。然后我们在**的第 227 和 228 行将这个精度图序列化到磁盘上。**

### **为包围盒回归训练我们的多类对象检测器**

我们现在准备使用 Keras 和 TensorFlow 来训练我们的多类对象检测器。

首先使用本教程的 ***【下载】*** 部分下载源代码和数据集。

从那里，打开一个终端，并执行以下命令:

```py
$ python train.py
[INFO] loading dataset...
[INFO] saving testing image paths...
Model: "model"
_____________________________________________________
Layer (type)                    Output Shape         
=====================================================
input_1 (InputLayer)            [(None, 224, 224, 3) 
_____________________________________________________
block1_conv1 (Conv2D)           (None, 224, 224, 64) 
_____________________________________________________
block1_conv2 (Conv2D)           (None, 224, 224, 64) 
_____________________________________________________
block1_pool (MaxPooling2D)      (None, 112, 112, 64) 
_____________________________________________________
block2_conv1 (Conv2D)           (None, 112, 112, 128 
_____________________________________________________
block2_conv2 (Conv2D)           (None, 112, 112, 128 
_____________________________________________________
block2_pool (MaxPooling2D)      (None, 56, 56, 128)  
_____________________________________________________
block3_conv1 (Conv2D)           (None, 56, 56, 256)  
_____________________________________________________
block3_conv2 (Conv2D)           (None, 56, 56, 256)  
_____________________________________________________
block3_conv3 (Conv2D)           (None, 56, 56, 256)  
_____________________________________________________
block3_pool (MaxPooling2D)      (None, 28, 28, 256)  
_____________________________________________________
block4_conv1 (Conv2D)           (None, 28, 28, 512)  
_____________________________________________________
block4_conv2 (Conv2D)           (None, 28, 28, 512)  
_____________________________________________________
block4_conv3 (Conv2D)           (None, 28, 28, 512)  
_____________________________________________________
block4_pool (MaxPooling2D)      (None, 14, 14, 512)  
_____________________________________________________
block5_conv1 (Conv2D)           (None, 14, 14, 512)  
_____________________________________________________
block5_conv2 (Conv2D)           (None, 14, 14, 512)  
_____________________________________________________
block5_conv3 (Conv2D)           (None, 14, 14, 512)  
_____________________________________________________
block5_pool (MaxPooling2D)      (None, 7, 7, 512)    
_____________________________________________________
flatten (Flatten)               (None, 25088)        
_____________________________________________________
dense_3 (Dense)                 (None, 512)          
_____________________________________________________
dense (Dense)                   (None, 128)          
_____________________________________________________
dropout (Dropout)               (None, 512)          
_____________________________________________________
dense_1 (Dense)                 (None, 64)           
_____________________________________________________
dense_4 (Dense)                 (None, 512)          
_____________________________________________________
dense_2 (Dense)                 (None, 32)           
_____________________________________________________
dropout_1 (Dropout)             (None, 512)          
_____________________________________________________
bounding_box (Dense)            (None, 4)            
_____________________________________________________
class_label (Dense)             (None, 3)            
=====================================================
Total params: 31,046,311
Trainable params: 16,331,623
Non-trainable params: 14,714,688
_____________________________________________________
```

在这里，我们从磁盘加载数据集，然后构建我们的模型架构。

**注意，我们的架构在层头**中有*两个分支*——第一个分支预测*包围盒坐标*，第二个分支预测被检测对象的*类标签*(见上面的**图 4** )。

随着我们的数据集加载和模型的构建，让我们训练用于多类对象检测的网络:

```py
[INFO] training model...
Epoch 1/20
51/51 [==============================] - 255s 5s/step - loss: 0.0526 - bounding_box_loss: 0.0078 - class_label_loss: 0.0448 - bounding_box_accuracy: 0.7703 - class_label_accuracy: 0.9070 - val_loss: 0.0016 - val_bounding_box_loss: 0.0014 - val_class_label_loss: 2.4737e-04 - val_bounding_box_accuracy: 0.8793 - val_class_label_accuracy: 1.0000
Epoch 2/20
51/51 [==============================] - 232s 5s/step - loss: 0.0039 - bounding_box_loss: 0.0012 - class_label_loss: 0.0027 - bounding_box_accuracy: 0.8744 - class_label_accuracy: 0.9945 - val_loss: 0.0011 - val_bounding_box_loss: 9.5491e-04 - val_class_label_loss: 1.2260e-04 - val_bounding_box_accuracy: 0.8744 - val_class_label_accuracy: 1.0000
Epoch 3/20
51/51 [==============================] - 231s 5s/step - loss: 0.0023 - bounding_box_loss: 8.5802e-04 - class_label_loss: 0.0014 - bounding_box_accuracy: 0.8855 - class_label_accuracy: 0.9982 - val_loss: 0.0010 - val_bounding_box_loss: 8.6327e-04 - val_class_label_loss: 1.8589e-04 - val_bounding_box_accuracy: 0.8399 - val_class_label_accuracy: 1.0000
...
Epoch 18/20
51/51 [==============================] - 231s 5s/step - loss: 9.5600e-05 - bounding_box_loss: 8.2406e-05 - class_label_loss: 1.3194e-05 - bounding_box_accuracy: 0.9544 - class_label_accuracy: 1.0000 - val_loss: 6.7465e-04 - val_bounding_box_loss: 6.7077e-04 - val_class_label_loss: 3.8862e-06 - val_bounding_box_accuracy: 0.8941 - val_class_label_accuracy: 1.0000
Epoch 19/20
51/51 [==============================] - 231s 5s/step - loss: 1.0237e-04 - bounding_box_loss: 7.7677e-05 - class_label_loss: 2.4690e-05 - bounding_box_accuracy: 0.9520 - class_label_accuracy: 1.0000 - val_loss: 6.7227e-04 - val_bounding_box_loss: 6.6690e-04 - val_class_label_loss: 5.3710e-06 - val_bounding_box_accuracy: 0.8966 - val_class_label_accuracy: 1.0000
Epoch 20/20
51/51 [==============================] - 231s 5s/step - loss: 1.2749e-04 - bounding_box_loss: 7.3415e-05 - class_label_loss: 5.4076e-05 - bounding_box_accuracy: 0.9587 - class_label_accuracy: 1.0000 - val_loss: 7.2055e-04 - val_bounding_box_loss: 6.6672e-04 - val_class_label_loss: 5.3830e-05 - val_bounding_box_accuracy: 0.8941 - val_class_label_accuracy: 1.0000
[INFO] saving object detector model...
[INFO] saving label binarizer...
```

由于培训过程的输出非常冗长，所以很难直观地解析它，所以我提供了一些图表来帮助可视化正在发生的事情。

我们的第一个图是我们的**类标签精度:**

这里我们可以看到，我们的对象检测器以 100%的准确率正确地对训练和测试集中检测到的对象的标签进行分类。

**下一个图可视化了我们的三个损失成分:*类别标签损失、* *边界框损失、*和*总损失*(类别标签和边界框损失的组合):**

我们的总损失一开始很高，但是到大约第三个时期，训练和验证损失几乎相同。

到了第五个纪元，它们基本上是相同的。

经过第十(10)个时期，我们的训练损失开始低于我们的验证损失——我们可能开始过度拟合，这从边界框损失*(底部)*可以明显看出，这表明验证损失没有训练损失下降得多。

培训完成后，您的`output`目录中应该有以下文件:

```py
$ ls output/
detector.h5	lb.pickle	plots		test_paths.txt
```

### **用 Keras 和 TensorFlow 实现对象检测预测脚本**

我们的多类对象检测器现在已经被训练并序列化到磁盘，但我们仍然需要一种方法来获取这个模型，并使用它在输入图像上实际做出*预测*——我们的`predict.py`文件会处理这些。

```py
# import the necessary packages
from pyimagesearch import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os
```

现在让我们解析我们的命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input image/text file of image paths")
args = vars(ap.parse_args())
```

```py
# determine the input file type, but assume that we're working with
# single input image
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]

# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:
	# load the image paths in our testing file
	imagePaths = open(args["input"]).read().strip().split("\n")
```

1.  **默认:**我们的`imagePaths`由来自`--input` ( **第 23 行**)的一条单独的图像路径组成。
2.  **文本文件:**如果**行 27** 上的文本`filetype`的条件/检查符合`True`，那么我们 ***覆盖*** 并从`--input`文本文件(**行 29** )中的所有`filenames`(每行一个)填充我们的`imagePaths`。

现在让我们从磁盘加载我们的序列化多类边界框回归器和`LabelBinarizer`:

```py
# load our object detector and label binarizer from disk
print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.LB_PATH, "rb").read())
```

```py
# loop over the images that we'll be testing using our bounding box
# regression model
for imagePath in imagePaths:
	# load the input image (in Keras format) from disk and preprocess
	# it, scaling the pixel intensities to the range [0, 1]
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image) / 255.0
	image = np.expand_dims(image, axis=0)

	# predict the bounding box of the object along with the class
	# label
	(boxPreds, labelPreds) = model.predict(image)
	(startX, startY, endX, endY) = boxPreds[0]

	# determine the class label with the largest predicted
	# probability
	i = np.argmax(labelPreds, axis=1)
	label = lb.classes_[i][0]
```

**第 38 行**在所有图像路径上循环。**第 41-43 行**通过以下方式对每幅图像进行预处理:

1.  从磁盘加载输入图像，将其调整为 *224×224* 像素
2.  将其转换为 NumPy 数组，并将像素亮度缩放到范围*【0，1】*
3.  向图像添加批次维度

1.  边界框预测(`boxPreds`)
2.  和类标签预测(`labelPreds`)

我们提取第 48 行**上的边界框坐标。**

**第 52 行**确定对应概率最大的类标签，而**第 53 行**使用这个索引值从我们的`LabelBinarizer`中提取人类可读的类标签串。

最后一步是将边界框坐标缩放回图像的原始空间尺寸，然后注释我们的输出:

```py
	# load the input image (in OpenCV format), resize it such that it
	# fits on our screen, and grab its dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# scale the predicted bounding box coordinates based on the image
	# dimensions
	startX = int(startX * w)
	startY = int(startY * h)
	endX = int(endX * w)
	endY = int(endY * h)

	# draw the predicted bounding box and class label on the image
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (0, 255, 0), 2)
	cv2.rectangle(image, (startX, startY), (endX, endY),
		(0, 255, 0), 2)

	# show the output image
	cv2.imshow("Output", image)
	cv2.waitKey(0)
```

**第 57 行和第 58 行**从磁盘加载我们的输入图像，然后将其调整到 600 像素的宽度(因此保证图像适合我们的屏幕)。

在调整图像大小后，我们在第 59 行**获取它的空间尺寸(即宽度和高度)。**

请记住，我们的边界框回归模型返回范围为*【0，1】*的边界框坐标，但是我们的图像分别具有范围为*【0，`w`*和*【0，`h`】*的空间维度。

**因此，我们需要根据图像的空间尺寸**缩放预测的边界框坐标——我们在**的第 63-66 行完成。**

最后，我们通过绘制预测的边界框及其相应的类标签来注释我们的输出图像(**行 69-73** )。

该输出图像随后显示在我们的屏幕上(**行 76 和 77** )。按一个键循环循环，一个接一个地显示结果，直到所有的测试图像都用完。

实现我们的`predict.py`脚本干得漂亮！让我们在下一节中让它发挥作用。

### **使用包围盒回归检测多类对象**

我们现在准备测试我们的多类对象检测器！

确保您已经使用本教程的 ***“下载”*** 部分下载了源代码、示例图像和预训练模型。

从那里，打开一个终端，并执行以下命令:

```py
$ python predict.py --input datasimg/face/image_0131.jpg 
[INFO] loading object detector...
```

在这里，我们传入了一张*“脸”*的示例图像——我们的多类对象检测器已经正确地检测到了这张脸，并对它进行了标记。

让我们试试另一张图片，这张是*“摩托车”:*

```py
$ python predict.py --input datasimg/motorcycle/image_0026.jpg 
[INFO] loading object detector...
```

我们的多类对象检测器再次表现良好，正确地定位和标记图像中的摩托车。

这是最后一个例子，这是一架*“飞机”:*

```py
$ python predict.py --input datasimg/airplane/image_0002.jpg 
[INFO] loading object detector...
```

同样，我们的对象检测器的输出是正确的。

您还可以通过更新`--input`命令行参数来预测`output/test_images.txt`中的测试图像:

```py
$ python predict.py --input output/test_paths.txt 
[INFO] loading object detector...
```

在上面的**图 10** 中可以看到输出的剪辑，注意我们的物体探测器能够:

1.  检测目标在输入图像中的位置
2.  正确标注*被检测对象是什么*

您可以使用本教程中讨论的代码和方法作为起点，使用边界框回归和 Keras/TensorFlow 训练您自己的自定义多类对象检测器。

### **局限与不足**

本教程中使用的对象检测体系结构和训练过程的最大限制之一是，该模型只能预测*一组*边界框和类别标签。

**如果图像中有*个多个物体*，那么只会预测最有把握的一个。**

这是一个完全不同的问题，我们将在以后的教程中讨论。

## **总结**

在本教程中，您学习了如何使用边界框回归和 Keras/TensorFlow 深度学习库来训练自定义多类对象检测器。

单类对象检测器只需要*一个回归层头来预测边界框。另一方面，多类对象检测器需要具有两个分支*的全连接层头。**

第一个分支是回归层集，就像在单类对象检测架构中一样。第二个分支由 softmax 分类器组成，该分类器用于预测检测到的边界框的类别标签。

**一起使用，我们的多类物体检测器的单次向前通过将导致:**

1.  图像中对象的预测边界框坐标
2.  图像中对象的预测类别标签

我希望这篇教程能让你更好地理解边界框回归对于单对象和多对象用例的工作原理。请随意使用本指南作为训练您自己的自定义对象检测器的起点。

如果你需要额外的帮助来训练你自己的自定义对象检测器，一定要参考我的书 *[用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)* ，在那里我详细讨论了对象检测。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***