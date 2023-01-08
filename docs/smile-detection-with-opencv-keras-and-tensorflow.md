# 利用 OpenCV、Keras 和 TensorFlow 进行微笑检测

> 原文：<https://pyimagesearch.com/2021/07/14/smile-detection-with-opencv-keras-and-tensorflow/>

在本教程中，我们将构建一个完整的端到端应用程序，该应用程序可以使用深度学习和传统的计算机视觉技术实时检测视频流中的微笑。

为了完成这项任务，我们将在一个图像数据集上训练 LetNet 架构，该数据集包含了正在*微笑的*和没有微笑的*的人脸。一旦我们的网络经过训练，我们将创建一个单独的 Python 脚本——这个脚本将通过 OpenCV 的内置 Haar cascade 人脸检测器检测图像中的人脸，从图像中提取人脸感兴趣区域(ROI ),然后将 ROI 通过 LeNet 进行微笑检测。*

 ***要学习如何用 OpenCV、Keras 和 TensorFlow 检测微笑，** ***继续阅读。***

## **使用 OpenCV、Keras 和 TensorFlow 进行微笑检测**

当开发图像分类的真实应用程序时，您经常需要将传统的计算机视觉和图像处理技术与深度学习相结合。为了在学习和应用深度学习时取得成功，我尽了最大努力来确保本教程在算法、技术和你需要理解的库方面独立存在。

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
*   **准备好在您的 Windows、macOS 或 Linux 系统上运行代码*****？***

 *那今天就加入 [PyImageSearch 大学](https://pyimagesearch.com/pyimagesearch-university/)吧！

**获得本教程的 Jupyter 笔记本和其他 PyImageSearch 指南，这些指南是** ***预先配置的*** **，可以在您的网络浏览器中运行在 Google Colab 的生态系统上！**无需安装。

最棒的是，这些 Jupyter 笔记本可以在 Windows、macOS 和 Linux 上运行！

### **微笑数据集**

微笑数据集由脸的图像组成，这些脸或者是*微笑的*或者是*不微笑的* ( [罗马达，2010 年](https://github.com/hromi/SMILEsmileD))。数据集中总共有 13，165 幅灰度图像，每幅图像的大小为 64*×64*像素。

正如**图 2** 所展示的，这个数据集中的图像在脸部周围被*紧密裁剪*，这将使训练过程更容易，因为我们将能够直接从输入图像中学习“微笑”或“不微笑”的模式。

然而，近距离裁剪在测试过程中带来了一个问题——因为我们的输入图像不仅包含人脸，还包含图像的*背景*,我们首先需要*定位*图像中的人脸，并提取人脸 ROI，然后才能通过我们的网络进行检测。幸运的是，使用传统的计算机视觉方法，如 Haar cascades，这比听起来容易得多。

我们需要在 SMILES 数据集中处理的第二个问题是*类不平衡*。虽然数据集中有 13，165 幅图像，但是这些例子中有 9，475 幅是*没有微笑，*，而只有 3，690 幅属于*微笑*类。鉴于有超过 2 个*。* 5 *x* 把“不笑”的图片数量改为“笑”的例子，我们在设计训练程序时需要小心。

我们的网络可能*自然地*选择“不笑”的标签，因为(1)分布是不均匀的，以及(2)它有更多“不笑”的脸看起来像什么的例子。稍后，您将看到我们如何通过在训练期间计算每个职业的“权重”来消除职业不平衡。

### **训练微笑 CNN**

构建我们的微笑检测器的第一步是在微笑数据集上训练 CNN，以区分微笑和不微笑的脸。为了完成这个任务，让我们创建一个名为`train_model.py`的新文件。从那里，插入以下代码:

```py
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from pyimagesearch.nn.conv import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os
```

**第 2-14 行**导入我们需要的 Python 包。我们以前使用过所有的包，但是我想提醒你注意**第 7 行**，在那里我们导入了 LeNet**(**[LeNet Tutorial](https://pyimagesearch.com/2021/05/22/lenet-recognizing-handwritten-digits/))类——这是我们在创建微笑检测器时将使用的架构。

接下来，让我们解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())

# initialize the list of data and labels
data = []
labels = []
```

我们的脚本需要两个命令行参数，我在下面详细介绍了每个参数:

1.  `--dataset`:位于磁盘上的 SMILES 目录的路径。
2.  `--model`:训练完成后，序列化 LeNet 权重的保存路径。

我们现在准备从磁盘加载 SMILES 数据集，并将其存储在内存中:

```py
# loop over the input images
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = imutils.resize(image, width=28)
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-3]
	label = "smiling" if label == "positives" else "not_smiling"
	labels.append(label)
```

在第 29 行**，**上，我们遍历了`--dataset`输入目录中的所有图像。对于这些图像中的每一个，我们:

1.  从磁盘加载(**第 31 行**)。
2.  将其转换为灰度(**第 32 行**)。
3.  调整它的大小，使其有一个*像素的固定输入尺寸*(**第 33 行**)。
4.  将图像转换为与 Keras 及其通道排序兼容的数组(**第 34 行**)。
5.  将`image`添加到 LeNet 将接受培训的`data`列表中。

**第 39-41 行**处理从`imagePath`中提取类标签并更新`labels`列表。SMILES 数据集将*微笑的*人脸存储在`SMILES/positives/positives7`子目录中，而*不微笑的*人脸存储在`SMILES/negatives/negatives7`子目录中。

因此，给定图像的路径:

```py
SMILEs/positives/positives7/10007.jpg
```

我们可以通过在图像路径分隔符上拆分并抓取倒数第三个子目录:`positives`来提取类标签。事实上，这正是**线 39** 所要完成的。

既然我们的`data`和`labels`已经构造好了，我们可以将原始像素强度缩放到范围`[0, 1]`，然后对`labels`应用一键编码:

```py
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# convert the labels from integers to vectors
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 2)
```

我们的下一个代码块通过计算类权重来处理我们的数据不平衡问题:

```py
# calculate the total number of training images in each class and
# initialize a dictionary to store the class weights
classTotals = labels.sum(axis=0)
classWeight = dict()

# loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]
```

**第 53 行**计算每个类的例子总数。在这种情况下，`classTotals`将是一个数组:`[9475, 3690]`分别表示“不笑”和“笑”。

然后我们*缩放**第 57 行和第 58 行**上的这些总数*，以获得用于处理类不平衡的`classWeight`，产生数组:`[1, 2.56]`。这种加权意味着我们的网络将把每个“微笑”的实例视为 2.56 个“不微笑”的实例，并通过在看到“微笑”的实例时用更大的权重放大每个实例的损失来帮助解决类别不平衡问题。

既然我们已经计算了我们的类权重，我们可以继续将我们的数据划分为训练和测试部分，将 80%的数据用于训练，20%用于测试:

```py
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, stratify=labels, random_state=42)
```

最后，我们准备培训 LeNet:

```py
# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam",
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	class_weight=classWeight, batch_size=64, epochs=15, verbose=1)
```

**第 67 行**初始化将接受`28×28`单通道图像的 LeNet 架构。假设只有两个类(微笑和不微笑)，我们设置`classes=2`。

我们也将使用`binary_crossentropy`而不是`categorical_crossentropy`作为我们的损失函数。同样，分类交叉熵仅在类别数量超过两个时使用。

到目前为止，我们一直使用 SGD 优化器来训练我们的网络。在这里，我们将使用亚当([金玛和巴，2014](http://arxiv.org/abs/1412.6980) ) ( **第 68 行**)。

同样，优化器和相关参数通常被认为是在训练网络时需要调整的超参数。当我把这个例子放在一起时，我发现 Adam 的表现比 SGD 好得多。

**73 号线和 74 号线**使用我们提供的`classWeight`训练 LeNet 总共 15 个时期，以对抗等级不平衡。

一旦我们的网络经过训练，我们就可以对其进行评估，并将权重序列化到磁盘中:

```py
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])
```

我们还将为我们的网络构建一条学习曲线，以便我们可以直观地了解性能:

```py
# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
```

要训练我们的微笑检测器，请执行以下命令:

```py
$ python train_model.py --dataset ../datasets/SMILEsmileD \
	--model output/lenet.hdf5
[INFO] compiling model...
[INFO] training network...
Train on 10532 samples, validate on 2633 samples
Epoch 1/15
8s - loss: 0.3970 - acc: 0.8161 - val_loss: 0.2771 - val_acc: 0.8872
Epoch 2/15
8s - loss: 0.2572 - acc: 0.8919 - val_loss: 0.2620 - val_acc: 0.8899
Epoch 3/15
7s - loss: 0.2322 - acc: 0.9079 - val_loss: 0.2433 - val_acc: 0.9062
...
Epoch 15/15
8s - loss: 0.0791 - acc: 0.9716 - val_loss: 0.2148 - val_acc: 0.9351
[INFO] evaluating network...
             precision    recall  f1-score   support

not_smiling       0.95      0.97      0.96      1890
    smiling       0.91      0.86      0.88       743

avg / total       0.93      0.94      0.93      2633

[INFO] serializing network...
```

在 15 个时期之后，我们可以看到我们的网络正在获得 93%的分类准确度。**图 3** 描绘了我们的学习曲线:

过了第六个时期，我们的确认损失开始停滞——过了第 15 个时期的进一步训练将导致过度拟合。如果需要，我们可以通过使用更多的训练数据来提高微笑检测器的准确性，方法是:

1.  收集其他培训数据。
2.  应用*数据增强*来随机平移、旋转和移动我们的*现有的*训练集。

### **实时运行微笑有线电视新闻网**

既然我们已经训练了我们的模型，下一步是构建 Python 脚本来访问我们的网络摄像头/视频文件，并对每一帧应用微笑检测。为了完成这一步，打开一个新文件，将其命名为`detect_smile.py`，然后我们开始工作。

```py
# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
```

**第 2-7 行**导入我们需要的 Python 包。`img_to_array`函数将用于将视频流中的每个单独的帧转换成一个适当的通道有序数组。`load_model`函数将用于从磁盘加载我们训练好的 LeNet 模型的权重。

`detect_smile.py`脚本需要两个命令行参数，后跟第三个可选参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
args = vars(ap.parse_args())
```

第一个参数，`--cascade`是用于检测图像中人脸的 Haar 级联的路径。保罗·维奥拉(Paul Viola)和迈克尔·琼斯(Michael Jones)于 2001 年首次发表了他们的工作，详细描述了哈尔级联 *[使用简单特征的增强级联](https://ieeexplore.ieee.org/document/990517)* 进行快速对象检测。该出版物已成为计算机视觉文献中被引用最多的论文之一。

Haar 级联算法能够检测图像中的对象，而不管它们的位置和比例。也许最有趣的(也是与我们的应用相关的)是，探测器可以在现代硬件上实时运行。事实上，Viola 和 Jones 工作背后的动机是创造一个面部检测器。

第二个常见的行参数是`--model`，它指定了我们在磁盘上序列化 LeNet 权重的路径。我们的脚本将*默认*从内置/USB 摄像头读取帧；然而，如果我们想从文件中读取帧，我们可以通过可选的`--video`开关来指定文件。

在检测微笑之前，我们首先需要执行一些初始化:

```py
# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])
```

**第 20 行和第 21 行**分别加载 Haar 级联人脸检测器和预训练的 LeNet 模型。如果*没有*提供视频路径，我们就抓取一个指向我们网络摄像头的指针(**第 24 行和第 25 行**)。否则，我们打开一个指向磁盘上视频文件的指针(**第 28 行和第 29 行**)。

现在，我们已经到达了应用程序的主要处理管道:

```py
# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a frame, then we
	# have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame, convert it to grayscale, and then clone the
	# original frame so we can draw on it later in the program
	frame = imutils.resize(frame, width=300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frameClone = frame.copy()
```

**第 32 行**开始一个循环，这个循环将一直持续到(1)我们停止脚本或者(2)我们到达视频文件的结尾(假设应用了一个`--video`路径)。

**第 34 行**从视频流中抓取下一帧。如果`frame`不是`grabbed`，那么我们已经到达了视频文件的末尾。否则，我们通过调整`frame`的大小使其宽度为 300 像素(**行 43** )并将其转换为灰度(**行 44** )来预处理人脸检测。

`.detectMultiScale`方法处理检测边界框( *x，y*)—`frame`中面部的坐标:

```py
	# detect faces in the input frame, then clone the frame so that
	# we can draw on it
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
```

这里，我们传入我们的灰度图像，并指出对于一个被认为是人脸的给定区域，它的*必须*具有最小宽度`30×30`像素。`minNeighbors`属性有助于删除误报，而`scaleFactor`控制生成的图像金字塔([http://pyimg.co/rtped](http://pyimg.co/rtped))等级的数量。

同样，对用于物体检测的哈尔级联的详细回顾不在本教程的范围之内。

`.detectMultiScale`方法返回一个 4 元组的列表，这些元组组成了在`frame`中包围面部的*矩形*。列表中的前两个值是起始( *x，y* )坐标。在`rects`列表中的后两个值分别是边界框的宽度和高度。

我们循环下面的每组边界框:

```py
	# loop over the face bounding boxes
	for (fX, fY, fW, fH) in rects:
		# extract the ROI of the face from the grayscale image,
		# resize it to a fixed 28x28 pixels, and then prepare the
		# ROI for classification via the CNN
		roi = gray[fY:fY + fH, fX:fX + fW]
		roi = cv2.resize(roi, (28, 28))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)
```

对于每个边界框，我们使用 NumPy 数组切片来提取人脸 ROI ( **行 58** )。一旦我们有了 ROI，我们就对它进行预处理，并通过调整它的大小、缩放它、将其转换为 Keras 兼容的数组，并用一个额外的维度填充图像来准备通过 LeNet 进行分类(**第 59-62 行**)。

`roi`经过预处理后，可以通过 LeNet 进行分类:

```py
		# determine the probabilities of both "smiling" and "not
		# smiling", then set the label accordingly
		(notSmiling, smiling) = model.predict(roi)[0]
		label = "Smiling" if smiling > notSmiling else "Not Smiling"
```

对**行 66** 上的`.predict`的调用分别返回“不笑”和“笑”的*概率*。**第 67 行**根据哪个概率大来设置`label`。

一旦我们有了`label`，我们就可以画出它，以及相应的边框:

```py
		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frameClone, label, (fX, fY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
			(0, 0, 255), 2)
```

我们最后的代码块处理在屏幕上显示输出帧:

```py
	# show our detected faces along with smiling/not smiling labels
	cv2.imshow("Face", frameClone)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
```

如果按下`q`键，我们将退出脚本。

要使用您的网络摄像头运行`detect_smile.py`，请执行以下命令:

```py
$ python detect_smile.py --cascade haarcascade_frontalface_default.xml \
	--model output/lenet.hdf5 
```

如果您想使用视频文件，您可以更新您的命令以使用`--video`开关:

```py
$ python detect_smile.py --cascade haarcascade_frontalface_default.xml \
	--model output/lenet.hdf5 --video path/to/your/video.mov
```

我已经将微笑检测脚本的结果包含在**图 4** 中:

请注意 LeNet 是如何根据我的面部表情正确预测“微笑”或“不微笑”的。

## **总结**

在本教程中，我们学习了如何构建一个端到端的计算机视觉和深度学习应用程序来执行微笑检测。为此，我们首先在 SMILES 数据集上训练 LeNet 架构。由于 SMILES 数据集中的类不平衡，我们发现了如何计算类权重来帮助缓解这个问题。

训练完成后，我们在测试集上对 LeNet 进行了评估，发现该网络获得了相当高的 93%的分类准确率。通过收集更多的训练数据或者对现有的训练数据进行数据扩充，可以获得更高的分类精度。

然后，我们创建了一个 Python 脚本来从网络摄像头/视频文件中读取帧，检测人脸，然后应用我们预先训练好的网络。为了检测人脸，我们使用了 OpenCV 的哈尔级联。一旦检测到一张脸，就从画面中提取出来，然后通过 LeNet 来确定这个人是在笑还是没笑。作为一个整体，我们的微笑检测系统可以使用现代硬件在 CPU 上轻松地实时运行。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！*****