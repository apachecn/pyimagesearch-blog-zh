# 新冠肺炎:具有 OpenCV、Keras/TensorFlow 和深度学习的人脸面具检测器

> 原文：<https://pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/>

在本教程中，您将学习如何使用 OpenCV、Keras/TensorFlow 和深度学习来训练新冠肺炎面具检测器。

上个月，我写了一篇关于使用深度学习在 X 射线图像中检测新冠肺炎的博文。

读者真的很喜欢从该教程的及时、实际应用中学习，所以今天我们将看看计算机视觉的另一个与 COVID 相关的应用，**这是一个关于使用 OpenCV 和 Keras/TensorFlow 检测口罩的应用。**

我创作这篇教程的灵感来自于:

1.  收到无数来自 PyImageSearch 读者的请求，要求我写这样一篇博文
2.  看到其他人实现他们自己的解决方案(我最喜欢的是[般若班达里的](https://www.linkedin.com/feed/update/urn%3Ali%3Aactivity%3A6655711815361761280/)，我们将从今天开始构建)

如果部署正确，我们今天在这里建造的新冠肺炎面具探测器有可能用于帮助确保你和他人的安全(但我将把这留给医疗专业人员来决定，实施和在野外分发)。

**要了解如何使用 OpenCV、Keras/TensorFlow 和深度学习创建新冠肺炎面具检测器，*请继续阅读！***

## 新冠肺炎:具有 OpenCV、Keras/TensorFlow 和深度学习的人脸面具检测器

在本教程中，我们将讨论如何在我们的两阶段新冠肺炎面具检测器中使用计算机视觉，详细说明如何实现我们的[计算机视觉](https://pyimagesearch.com/)和[深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)流水线。

从那里，我们将审查数据集，我们将使用训练我们的自定义面具检测器。

然后，我将向您展示如何使用 Keras 和 TensorFlow 实现一个 Python 脚本，在我们的数据集上训练一个人脸遮罩检测器。

我们将使用这个 Python 脚本来训练一个面具检测器，并检查结果。

鉴于训练有素的新冠肺炎面具检测器，我们将继续实现两个额外的 Python 脚本用于:

1.  检测*图像中的新冠肺炎面具*
2.  检测*实时视频流*中的面具

我们将通过查看应用我们的面罩检测器的结果来结束这篇文章。

我还将提供一些进一步改进的额外建议。

### 两相新冠肺炎面罩检测器

为了训练一个定制的人脸面具检测器，我们需要将我们的项目分成两个不同的阶段，每个阶段都有各自的子步骤(如上面的图 1 所示):

1.  **训练:**在这里，我们将重点关注从磁盘加载我们的人脸面具检测数据集，在该数据集上训练模型(使用 Keras/TensorFlow)，然后将人脸面具检测器序列化到磁盘
2.  **部署:**一旦训练好了面具检测器，我们就可以继续加载面具检测器，执行面部检测，然后将每个面部分类为`with_mask`或`without_mask`

我们将在本教程的剩余部分详细回顾这些阶段中的每一个和相关联的子集，但同时，让我们看看将用于训练我们的新冠肺炎面具检测器的数据集。

### 我们的新冠肺炎面具检测数据集

我们今天在这里使用的数据集是由 PyImageSearch 阅读器 [Prajna Bhandary 创建的。](https://www.linkedin.com/feed/update/urn%3Ali%3Aactivity%3A6655711815361761280/)

该数据集由属于两类的**1376 幅图像**组成:

*   `with_mask` : 690 张图片
*   ``without_mask`` : 686 张图片

**我们的目标是训练一个定制的深度学习模型来检测一个人*是不是戴着面具的*或者*不是*。**

***注:**为了方便起见，我将般若创建的数据集包含在本教程的**“下载”**部分。*

### 我们的面罩数据集是如何创建的？

般若，像我一样，一直对世界的现状感到沮丧和压抑——每天都有成千上万的人死去，而对我们许多人来说，我们能做的(如果有的话)很少。

为了帮助她保持斗志，般若决定通过应用计算机视觉和深度学习来分散自己的注意力，以解决一个现实世界的问题:

*   最好的情况是——她可以用她的项目去帮助别人
*   最坏的情况——这给了她一个急需的精神解脱

不管怎样，这都是双赢！

作为程序员、开发人员和计算机视觉/深度学习从业者，我们可以*都*从般若的书中吸取一页——让你的技能成为你的分心和避风港。

为了创建这个数据集，般若有一个巧妙的解决方案:

1.  拍摄*正常的人脸图像*
2.  然后创建一个*定制的计算机视觉 Python 脚本*来给它们添加面具，**从而创建一个*人造的*(但仍然适用于现实世界)数据集**

一旦你把面部标志应用到问题上，这个方法实际上比听起来要容易得多。

面部标志允许我们自动推断面部结构的位置，包括:

*   眼睛
*   眉毛
*   鼻子
*   口
*   下颌的轮廓

为了使用面部标志来建立戴着面罩的面部数据集，我们需要首先从戴着面罩的人*而不是*的图像开始:

从那里，我们应用面部检测来计算图像中面部的边界框位置:

一旦我们知道*人脸在图像中的位置*，我们就可以提取人脸感兴趣区域(ROI):

从那里，我们应用面部标志，允许我们定位眼睛，鼻子，嘴等。：

接下来，我们需要一个面具的图像(背景透明)，如下图所示:

通过使用面部标志(即沿着下巴和鼻子的点)计算将放置面具的*，该面具将*自动*应用到面部。*

然后调整面具的大小并旋转，将其放在脸上:

然后，我们可以对所有输入图像重复这一过程，从而创建我们的人工人脸面具数据集:

***但是，使用这种方法人为创建数据集时，有一点需要注意！***

如果您使用一组图像来创建一个戴着面具的人的人工数据集，**您*不能*在您的训练集中“重用”没有面具的图像*——您仍然需要收集在人工生成过程中*而不是*使用的非人脸面具图像！***

如果您将用于生成人脸遮罩样本的原始图像作为非人脸遮罩样本包括在内，您的模型将变得严重偏颇，并且无法很好地进行概括。不惜一切代价，通过花时间收集没有面具的面孔的新例子来避免这种情况。

介绍如何使用面部标志将面具应用到脸上超出了本教程的范围，但如果你想了解更多，我建议:

1.  参考[般若的 GitHub 库](https://github.com/prajnasb/observations/tree/master/mask_classifier/Data_Generator)
2.  在 PyImageSearch 博客上阅读本教程，在那里我讨论了[如何使用面部标志自动将太阳镜应用到脸上](https://pyimagesearch.com/2018/11/05/creating-gifs-with-opencv/)

我的太阳镜帖子中的相同原理适用于构建人工人脸面具数据集——使用面部标志来推断面部结构，旋转和调整面具的大小，然后将其应用于图像。

### 项目结构

一旦您从本文的 ***【下载】*** 部分获取文件，您将看到以下目录结构:

```py
$ tree --dirsfirst --filelimit 10
.
├── dataset
│   ├── with_mask [690 entries]
│   └── without_mask [686 entries]
├── examples
│   ├── example_01.png
│   ├── example_02.png
│   └── example_03.png
├── face_detector
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── detect_mask_image.py
├── detect_mask_video.py
├── mask_detector.model
├── plot.png
└── train_mask_detector.py

5 directories, 10 files
```

*   ``train_mask_detector.py`` :接受我们的输入数据集，并对其进行微调，以创建我们的`mask_detector.model`。还会生成包含精度/损耗曲线的训练历史`plot.png`
*   `detect_mask_image.py`:在静态图像中执行人脸遮罩检测
*   这个脚本使用你的网络摄像头，对视频流中的每一帧进行面具检测

在接下来的两个部分中，我们将训练我们的面罩检测器。

### 使用 Keras 和 TensorFlow 实现我们的新冠肺炎面罩检测器训练脚本

现在我们已经回顾了我们的面具数据集，让我们学习如何使用 Keras 和 TensorFlow 来训练一个分类器来自动检测一个人是否戴着面具。

为了完成这项任务，我们将微调 [MobileNet V2 架构](https://arxiv.org/abs/1801.04381)，这是一种高效的架构，可应用于计算能力有限的嵌入式设备(例如、树莓派、谷歌珊瑚、英伟达 Jetson Nano 等。).

***注:**如果你的兴趣是嵌入式计算机视觉，一定要看看我的* [树莓 Pi for Computer Vision book](https://pyimagesearch.com/raspberry-pi-for-computer-vision/) *该书涵盖了使用计算有限的设备进行计算机视觉和深度学习。*

将我们的面罩检测器部署到嵌入式设备可以降低制造这种面罩检测系统的成本，因此我们选择使用这种架构。

我们开始吧！

打开目录结构中的`train_mask_detector.py`文件，插入以下代码:

```py
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
```

我们的训练脚本的导入对您来说可能看起来有些吓人，要么是因为有太多的导入，要么是因为您对深度学习不熟悉。如果你是新手，我会推荐你在继续之前阅读我的 Keras 教程和 T2 微调教程。

我们的一套`tensorflow.keras`进口允许:

*   数据扩充
*   加载 MobilNetV2 分类器(我们将使用预先训练的 [ImageNet](http://www.image-net.org/) 权重来微调该模型)
*   构建新的全连接(FC)磁头
*   预处理
*   加载图像数据

要安装必要的软件以便您可以使用这些导入，请确保遵循我的任一 Tensorflow 2.0+安装指南:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   *[如何在 macOS 上安装 tensor flow 2.0](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)*

让我们继续解析一些从终端启动脚本所需的[命令行参数](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/):

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())
```

我们的命令行参数包括:

*   `--dataset`:面和带遮罩的面的输入数据集的路径
*   `--plot`:输出训练历史图的路径，将使用`matplotlib`生成
*   `--model`:生成的序列化人脸面具分类模型的路径

我喜欢在一个地方定义我的深度学习超参数:

```py
# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
```

```py
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)
```

在这个街区，我们是:

*   抓取数据集中的所有`imagePaths`(**行 44** )
*   初始化`data`和`labels`列表(**第 45 和 46 行**
*   循环`imagePaths`并加载+预处理图像(**第 49-60 行**)。预处理步骤包括调整大小为 *224×224* 像素，转换为数组格式，以及将输入图像中的像素亮度缩放到范围 *[-1，1]* (通过`preprocess_input`便利函数)
*   将预处理后的`image`和关联的`label`分别追加到`data`和`labels`列表中(**第 59 行和第 60 行**
*   确保我们的训练数据是 NumPy 数组格式(**行 63 和 64** )

上面几行代码假设您的整个数据集足够小，可以放入内存。如果你的数据集大于你可用的内存，我建议使用 HDF5，这是我在 *[用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)* (从业者捆绑包第 9 章和第 10 章)中介绍的策略。

我们的数据准备工作还没有完成。接下来，我们将对我们的`labels`进行编码，对我们的数据集进行分区，并为[数据扩充](https://pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/)做准备:

```py
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
```

**第 67-69 行** one-hot 编码我们的类标签，这意味着我们的数据将采用以下格式:

```py
$ python  train_mask_detector.py --dataset  dataset 
[INFO] loading images...
-> (trainX, testX, trainY, testY) = train_test_split(data, labels,
(Pdb) labels[500:]
array([[1., 0.],
       [1., 0.],
       [1., 0.],
       ...,
       [0., 1.],
       [0., 1.],
       [0., 1.]], dtype=float32)
(Pdb)
```

```py
# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False
```

微调设置分为三个步骤:

1.  用预先训练好的 [ImageNet](http://www.image-net.org/) 权重加载 MobileNet，留下网络头(**行 88 和 89**
2.  构建一个新的 FC 头，并将其附加到基座上以代替旧的头(**第 93-102 行**)
3.  冻结网络的基本层(**行 106 和 107** )。这些基本层的权重在反向传播过程中不会更新，而头层权重*将*被调整。

微调是一种策略，我几乎总是推荐这种策略来建立基线模型，同时节省大量时间。要了解更多关于理论、目的和策略的信息，请参考我的[微调博文](https://pyimagesearch.com/tag/fine-tuning/)和 *[用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)* (从业者捆绑第五章)。

准备好数据和用于微调的模型架构后，我们现在准备编译和训练我们的面罩检测器网络:

```py
# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)
```

训练完成后，我们将在测试集上评估结果模型:

```py
# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")
```

我们的最后一步是绘制精度和损耗曲线:

```py
# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
```

一旦我们的绘图准备就绪，**第 152 行**使用`--plot`文件路径将数字保存到磁盘。

### 用 Keras/TensorFlow 训练新冠肺炎口罩检测器

我们现在准备使用 Keras、TensorFlow 和深度学习来训练我们的面罩检测器。

确保您已经使用本教程的 ***【下载】*** 部分下载源代码和面罩数据集。

从那里，打开一个终端，并执行以下命令:

```py
$ python train_mask_detector.py --dataset dataset
[INFO] loading images...
[INFO] compiling model...
[INFO] training head...
Train for 34 steps, validate on 276 samples
Epoch 1/20
34/34 [==============================] - 30s 885ms/step - loss: 0.6431 - accuracy: 0.6676 - val_loss: 0.3696 - val_accuracy: 0.8242
Epoch 2/20
34/34 [==============================] - 29s 853ms/step - loss: 0.3507 - accuracy: 0.8567 - val_loss: 0.1964 - val_accuracy: 0.9375
Epoch 3/20
34/34 [==============================] - 27s 800ms/step - loss: 0.2792 - accuracy: 0.8820 - val_loss: 0.1383 - val_accuracy: 0.9531
Epoch 4/20
34/34 [==============================] - 28s 814ms/step - loss: 0.2196 - accuracy: 0.9148 - val_loss: 0.1306 - val_accuracy: 0.9492
Epoch 5/20
34/34 [==============================] - 27s 792ms/step - loss: 0.2006 - accuracy: 0.9213 - val_loss: 0.0863 - val_accuracy: 0.9688
...
Epoch 16/20
34/34 [==============================] - 27s 801ms/step - loss: 0.0767 - accuracy: 0.9766 - val_loss: 0.0291 - val_accuracy: 0.9922
Epoch 17/20
34/34 [==============================] - 27s 795ms/step - loss: 0.1042 - accuracy: 0.9616 - val_loss: 0.0243 - val_accuracy: 1.0000
Epoch 18/20
34/34 [==============================] - 27s 796ms/step - loss: 0.0804 - accuracy: 0.9672 - val_loss: 0.0244 - val_accuracy: 0.9961
Epoch 19/20
34/34 [==============================] - 27s 793ms/step - loss: 0.0836 - accuracy: 0.9710 - val_loss: 0.0440 - val_accuracy: 0.9883
Epoch 20/20
34/34 [==============================] - 28s 838ms/step - loss: 0.0717 - accuracy: 0.9710 - val_loss: 0.0270 - val_accuracy: 0.9922
[INFO] evaluating network...
              precision    recall  f1-score   support

   with_mask       0.99      1.00      0.99       138
without_mask       1.00      0.99      0.99       138

    accuracy                           0.99       276
   macro avg       0.99      0.99      0.99       276
weighted avg       0.99      0.99      0.99       276
```

如你所见，我们在测试集上获得了 **~99%的准确率**。

查看**图 10** ，我们可以看到几乎没有过度拟合的迹象，验证损失*低于培训损失*(我在这篇博文中讨论了[的这一现象)。](https://pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/)

鉴于这些结果，我们希望我们的模型能够很好地推广到我们的训练和测试集之外的图像。

### 用 OpenCV 实现我们的新冠肺炎人脸面具检测器

既然我们的面罩检测器已经过培训，让我们来学习如何:

1.  从磁盘加载输入图像
2.  检测图像中的人脸
3.  应用我们的面罩检测器将面部分类为`with_mask`或`without_mask`

打开目录结构中的`detect_mask_image.py`文件，让我们开始吧:

```py
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
```

我们的驱动程序脚本需要三个 TensorFlow/Keras 导入来(1)加载我们的 MaskNet 模型和(2)预处理输入图像。

显示和图像操作需要 OpenCV。

下一步是解析[命令行参数](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/):

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
```

我们的四个命令行参数包括:

*   `--image`:包含用于推断的面部的输入图像的路径
*   `--face`:人脸检测器模型目录的路径(我们需要在分类之前定位人脸)
*   ``--model`` :我们之前在本教程中训练的面罩检测器模型的路径
*   `--confidence`:可选的概率阈值可以设置为覆盖 50%来过滤弱的人脸检测

接下来，我们将加载我们的人脸检测器和人脸面具分类器模型:

```py
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])
```

现在我们的深度学习模型已经在内存中，我们的下一步是加载和预处理输入图像:

```py
# load the input image from disk, clone it, and grab the image spatial
# dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# pass the blob through the network and obtain the face detections
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()
```

在从磁盘加载我们的`--image`(**第 37 行**)时，我们制作一个副本并获取帧尺寸，用于将来的缩放和显示目的(**第 38 和 39 行**)。

预处理由 [OpenCV 的 blobFromImage 函数](https://pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/) ( **第 42 行和第 43 行**)处理。如参数所示，我们将尺寸调整为 *300×300* 像素，并执行均值减法。

**行 47 和 48** 然后执行面部检测以定位*在图像中的位置*所有的面部。

一旦我们知道了每张脸的预测位置，我们将确保在提取人脸之前它们满足`--confidence`阈值:

```py
# loop over the detections
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with
	# the detection
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the confidence is
	# greater than the minimum confidence
	if confidence > args["confidence"]:
		# compute the (x, y)-coordinates of the bounding box for
		# the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# ensure the bounding boxes fall within the dimensions of
		# the frame
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
```

接下来，我们将通过我们的 MaskNet 模型运行面部 ROI:

```py
		# extract the face ROI, convert it from BGR to RGB channel
		# ordering, resize it to 224x224, and preprocess it
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# pass the face through the model to determine if the face
		# has a mask or not
		(mask, withoutMask) = model.predict(face)[0]
```

在这一部分，我们:

*   通过 NumPy 切片提取`face`ROI(**第 71 行**
*   像我们在培训期间一样预处理 ROI(**第 72-76 行**)
*   进行掩膜检测，预测`with_mask`或`without_mask` ( **行 80** )

从这里，我们将注释和显示结果！

```py
		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
```

### 用 OpenCV 实现图像中的新冠肺炎人脸检测

让我们把我们的新冠肺炎面罩探测器投入使用吧！

确保您已经使用本教程的 ***“下载”*** 部分下载了源代码、示例图像和预训练的面罩检测器。

从那里，打开一个终端，并执行以下命令:

```py
$ python detect_mask_image.py --image examples/example_01.png 
[INFO] loading face detector model...
[INFO] loading face mask detector model...
[INFO] computing face detections...
```

正如你所看到的，我们的面罩检测器正确地将这张图像标记为`Mask`。

让我们尝试另一个图像，这是一个戴着面具的人*而不是*:

```py
$ python detect_mask_image.py --image examples/example_02.png 
[INFO] loading face detector model...
[INFO] loading face mask detector model...
[INFO] computing face detections...
```

我们的面罩探测器已经正确预测了`No Mask`。

让我们尝试最后一张图片:

```py
$ python detect_mask_image.py --image examples/example_03.png 
[INFO] loading face detector model...
[INFO] loading face mask detector model...
[INFO] computing face detections...
```

这里发生了什么？

为什么我们能够*检测到*背景中两位男士的脸，并且*正确地为他们分类*戴面具/不戴面具，但是我们不能检测到前景中的女人？

我将在本教程后面的*“进一步改进的建议”*部分讨论这个问题的原因，**，但是要点是我们*太依赖*我们的两阶段过程。**

请记住，为了对一个人是否戴着面具进行分类，我们首先需要执行人脸检测— **如果没有找到人脸(这就是本图中发生的情况)，*则不能应用面具检测器！***

我们无法检测前景中的人脸的原因是:

1.  它被面具遮住了
2.  用于训练人脸检测器的数据集不包含戴口罩的人的示例图像

因此，如果人脸的大部分被遮挡，我们的人脸检测器将很可能无法检测到人脸。

在本教程的*“进一步改进的建议”*部分，我再次更详细地讨论了这个问题，包括如何提高我们的掩模检测器的精度。

### 用 OpenCV 实现实时视频流中的新冠肺炎人脸检测器

在这一点上，我们知道我们可以将人脸面具检测应用于静态图像— ***但是实时视频流呢？***

我们的新冠肺炎面罩检测器能够实时运行吗？

让我们找出答案。

打开目录结构中的`detect_mask_video.py`文件，插入以下代码:

```py
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
```

这个脚本的算法是相同的，但它是以这样一种方式拼凑起来的，以允许处理您的网络摄像头流的每一帧。

```py
def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []
```

通过在这里定义这个方便的函数，我们的帧处理循环在后面会更容易阅读。

此功能检测人脸，然后将我们的人脸遮罩分类器应用于每个人脸感兴趣区域。这样一个函数合并了我们的代码——如果您愿意，它甚至可以被移动到一个单独的 Python 文件中。

```py
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
```

在循环内部，我们过滤掉弱检测(**行 34-38** )并提取边界框，同时确保边界框坐标不会落在图像的边界之外(**行 41-47** )。

接下来，我们将面部 ROI 添加到两个相应的列表中:

```py
			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))
```

```py
	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
```

第 72 行向调用者返回我们的面部边界框位置和相应的遮罩/非遮罩预测。

接下来，我们将定义我们的[命令行参数](https://pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/):

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
```

我们的命令行参数包括:

```py
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
```

这里我们已经初始化了我们的:

*   人脸检测器
*   新冠肺炎面罩检测器
*   网络摄像头视频流

让我们继续循环流中的帧:

```py
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
```

我们开始在**行 103** 的帧上循环。在里面，我们从流中抓取`frame`并`resize`它(**第 106 和 107 行**)。

从那里，我们使用我们的便利工具；**111 线**检测并预测人们是否戴着口罩。

让我们后处理(即，注释)新冠肺炎面罩检测结果:

```py
	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
```

在我们对预测结果的循环中(从**行 115** 开始)，我们:

*   解包面部边界框并屏蔽/不屏蔽预测(**行 117 和 118**
*   确定`label`和`color` ( **线 122-126** )
*   注释`label`和面部边界框(**第 130-132 行**

最后，我们显示结果并执行清理:

```py
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

在`frame`显示之后，我们捕获`key`按压。如果用户按下`q`(退出)，我们`break`退出循环并执行内务处理。

使用 Python、OpenCV 和 TensorFlow/Keras 的深度学习实现您的实时人脸面具检测器非常棒！

### 用 OpenCV 实时检测新冠肺炎面具

要查看我们的实时新冠肺炎面罩检测器，请确保使用本教程的 ***【下载】*** 部分下载源代码和预先训练的面罩检测器模型。

然后，您可以使用以下命令在实时视频流中启动遮罩检测器:

```py
$ python detect_mask_video.py
[INFO] loading face detector model...
[INFO] loading face mask detector model...
[INFO] starting video stream...
```

在这里，你可以看到我们的面罩检测器能够实时运行(并且*在其预测中也是正确的*)。

### 改进建议

从上面的结果部分可以看出，我们的面罩检测器运行良好，尽管:

1.  训练数据有限
2.  人工生成的`with_mask`类(参见*“我们的面罩数据集是如何创建的？”*上节)。

为了进一步改进我们的面具检测模型，**你应该收集戴面具的人的*真实图像*(而不是人工生成的图像)。**

虽然我们的人工数据集在这种情况下工作得很好，但没有什么可以替代真实的东西。

**其次，你还应该收集面部图像，这些图像可能会“迷惑”我们的分类器，使其认为这个人戴着面具，而实际上他们并没有戴面具** —潜在的例子包括裹在脸上的衬衫、蒙在嘴上的大手帕等。

所有这些都是*可能被我们的面罩检测器*误认为面罩的例子。

**最后，你应该考虑训练一个专用的两类*物体检测器*，而不是一个简单的图像分类器。**

我们目前检测一个人是否戴着口罩的方法是一个两步过程:

1.  **步骤#1:** 执行面部检测
2.  **步骤#2:** 将我们的面具检测器应用到每张脸上

这种方法的问题是，根据定义，面具遮住了脸的一部分。如果足够多的人脸被遮挡，则无法检测到人脸，*因此，人脸遮罩检测器将不会被应用。*

首先，对象检测器将能够自然地检测戴着面具的人，否则由于太多的面部被遮挡，面部检测器不可能检测到戴着面具的人。

其次，这种方法将我们的计算机视觉管道减少到一个步骤——而不是应用人脸检测和*然后*我们的人脸面具检测器模型，我们需要做的只是应用对象检测器，在网络的单次前向传递中为我们提供人物`with_mask`和`without_mask`的边界框。

这种方法不仅计算效率更高，而且更加“优雅”和端到端。

## 摘要

在本教程中，您学习了如何使用 OpenCV、Keras/TensorFlow 和深度学习来创建新冠肺炎面具检测器。

为了创建我们的面具检测器，我们训练了一个两类模型，一类是戴面具的人*和不戴面具的人*。

我们在我们的*屏蔽/无屏蔽*数据集上微调了 MobileNetV2，并获得了一个准确率为 **~99%的分类器。**

然后，我们将这个人脸面具分类器应用到*图像*和*实时视频流*中，方法是:

1.  检测图像/视频中的人脸
2.  提取每一张人脸
3.  应用我们的面罩分类器

我们的面具检测器是准确的，而且由于我们使用了 MobileNetV2 架构，它也是*计算高效的*，使得它更容易部署到嵌入式系统([树莓派](https://www.raspberrypi.org/)、[谷歌珊瑚](https://coral.ai/)、[英伟达](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)等)。).

我希望你喜欢这个教程！

更新:【2022 年 12 月，更新链接和文字。

**要下载这篇文章的源代码(包括预先训练的新冠肺炎面罩检测器模型)，*只需在下面的表格中输入您的电子邮件地址！***