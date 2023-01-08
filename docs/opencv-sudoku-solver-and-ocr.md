# OpenCV 数独求解器和 OCR

> 原文：<https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/>

在本教程中，您将使用 OpenCV、深度学习和光学字符识别(OCR)创建一个自动数独解谜器。

我妻子是个超级数独迷。每次我们旅行，无论是从费城到奥尔巴尼的 45 分钟飞行，还是到加州的 6 小时洲际飞行，*她总是带着一个数独谜题。*

有趣的是，她更喜欢*印刷的*数独解谜书。她讨厌数码/智能手机应用程序版本，并拒绝玩它们。

我自己不是一个喜欢猜谜的人，但是有一次，我们坐在飞机上，我问:

> 如何知道你是否正确地解决了这个难题？书的后面有解决方案吗？还是你只是去做，希望它是正确的？

显然，这是一个愚蠢的问题，原因有二:

1.  **是的，那里的*就是*后面的一把解钥匙。**你需要做的就是翻到书的背面，找到谜题号码，然后看到答案。
2.  最重要的是，*她不会猜错谜题。*

然后她给我上了 20 分钟的课，告诉我她只能解决“4 级和 5 级难题”，接着是一堂关于“X 翼”和“Y 翼”技术的数独解谜课。我有计算机科学的博士学位，但所有这些我都不知道。

但对于那些没有像我一样嫁给数独大师的人来说，这确实提出了一个问题:

> *OpenCV 和 OCR 可以用来解决和检查数独谜题吗？*

如果数独谜题制造商*不需要*在书的背面打印答案，而是**提供一个应用程序让用户检查他们的谜题，**打印机可以把节省下来的钱装进口袋，或者免费打印额外的谜题。

数独拼图公司赚了更多的钱，最终用户也很高兴。似乎是双赢。

从我的角度来看，如果我出版一本数独教程，也许我能重新赢得我妻子的好感。

**要了解如何使用 OpenCV、深度学习和 OCR 构建自动数独解谜器，*请继续阅读。***

## **OpenCV 数独解算器和 OCR**

在本教程的第一部分，我们将讨论使用 OpenCV、深度学习和光学字符识别(OCR)技术构建数独解谜器所需的步骤。

在那里，您将配置您的开发环境，并确保安装了正确的库和包。

在我们编写任何代码之前，我们将首先回顾我们的项目目录结构，确保您知道在本教程的整个过程中将创建、修改和使用什么文件。

然后我将向您展示如何实现`SudokuNet`，这是一个基本的卷积神经网络(CNN)，将用于 OCR 数独拼图板上的数字。

然后，我们将使用 Keras 和 TensorFlow 训练该网络识别数字。

但是在我们真正能够*检查*和*解决*一个数独难题之前，我们首先需要定位*数独板在图像中的位置*——我们将实现助手函数和实用程序来帮助完成这项任务。

最后，我们将把所有的部分放在一起，实现我们完整的 OpenCV 数独解谜器。

### **如何用 OpenCV 和 OCR 解决数独难题**

用 OpenCV 创建一个自动数独解谜程序需要 6 个步骤:

*   **步骤#1:** 向我们的系统提供包含数独谜题的输入图像。
*   **步骤#2:** 在输入图像中的处定位*并提取棋盘。*
*   **步骤#3:** 给定棋盘，定位数独棋盘的每个独立单元(大多数标准数独谜题是一个 *9×9* 网格，所以我们需要定位这些单元)。
*   **步骤#4:** 判断单元格中是否有数字，如果有，就进行 OCR。
*   **步骤#5:** 应用数独难题解算器/检验器算法来验证难题。
*   **步骤#6:** 向用户显示输出结果。

这些步骤中的大部分可以使用 OpenCV 以及基本的计算机视觉和图像处理操作来完成。

最大的例外是第 4 步，我们需要应用 OCR。

OCR 的应用可能有点棘手，但我们有很多选择:

1.  使用 Tesseract OCR 引擎，这是开源 OCR 的事实上的标准
2.  利用基于云的 OCR APIs，如微软认知服务、亚马逊 Rekognition 或谷歌视觉 API
3.  训练我们自己的自定义 OCR 模型

所有这些都是完全有效的选择。然而，为了制作一个完整的端到端教程，**我决定使用深度学习来训练我们自己的自定义数独 OCR 模型。**

一定要系好安全带——这将是一次疯狂的旅行。

### **配置您的开发环境，使用 OpenCV 和 OCR 解决数独难题**

要针对本教程配置您的系统，我建议您遵循以下任一教程来建立您的基准系统并创建一个虚拟环境:

*   [*如何在 Ubuntu 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-ubuntu/)
*   [*如何在 macOS 上安装 tensor flow 2.0*](https://pyimagesearch.com/2019/12/09/how-to-install-tensorflow-2-0-on-macos/)

请注意 [PyImageSearch 不推荐也不支持 CV/DL 项目](https://pyimagesearch.com/faqs/single-faq/can-you-help-me-do-___-on-windows/)的窗口。

一旦您的环境启动并运行，您将需要本教程的另一个包。你需要安装 [py-sudoku](https://pypi.org/project/py-sudoku/) ，我们将使用这个库来帮助我们解决数独难题:

```py
$ pip install py-sudoku
```

### **项目结构**

花点时间从本教程的 ***“下载”*** 部分抓取今天的文件。从那里，提取归档文件，并检查内容:

```py
$ tree --dirsfirst 
.
├── output
│   └── digit_classifier.h5
├── pyimagesearch
│   ├── models
│   │   ├── __init__.py
│   │   └── Sudokunet.py
│   ├── Sudoku
│   │   ├── __init__.py
│   │   └── puzzle.py
│   └── __init__.py
├── solve_sudoku_puzzle.py
├── sudoku_puzzle.jpg
└── train_digit_classifier.py

4 directories, 9 files
```

和所有 CNN 一样，SudokuNet 需要用数据来训练。我们的`train_digit_classifier.py`脚本将在 [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)上训练一个数字 OCR 模型。

一旦 SudokuNet 训练成功，我们将部署它和我们的`solve_sudoku_puzzle.py`脚本来解决一个数独难题。

当你的系统工作时，你可以用这个应用给你的朋友留下深刻印象。或者更好的是，在飞机上愚弄他们，因为你解谜的速度可能比他们在你身后的座位上更快！别担心，我不会说出去的！

### SudokuNet:在 Keras 和 TensorFlow 中实现的数字 OCR 模型

每一个数独谜题都以一个 *NxN* 网格开始(通常是 *9×9* )，其中**一些单元格是*空白*** 而**其他单元格已经*包含一个数字。***

目标是使用关于*现有数字*到*的知识正确推断其他数字。*

但是在我们可以用 OpenCV 解决数独难题之前，我们首先需要实现一个神经网络架构，它将处理数独难题板上的 OCR 数字——给定这些信息，解决实际的难题将变得微不足道。

```py
# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
```

```py
class SudokuNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)
```

我们的`SudokuNet`类是在**的第 10-12 行**用一个静态方法(没有构造函数)定义的。`build`方法接受以下参数:

*   ``width``:MNIST 数字的宽度(`28`像素)
*   ``height``:MNIST 数字的高度(`28`像素)
*   `depth`:MNIST 数字图像通道(`1`灰度通道)
*   ``classes`` :数字位数 *0-9* ( `10`位数)

```py
		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(32, (5, 5), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		# first set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		# second set of FC => RELU layers
		model.add(Dense(64))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
```

如果您对 CNN 层和使用顺序 API 不熟悉，我建议您查看以下资源:

*   *[Keras 教程:如何入门 Keras、深度学习、Python](https://pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/)*
*   *[用 Python 进行计算机视觉的深度学习](https://pyimagesearch.com/deep-learning-computer-vision-python-book/)* (入门套装)

***注意:**作为题外话，我想在这里花点时间指出，例如，如果您正在构建一个 CNN 来分类 26 个大写英文字母加上 10 个数字(总共 36 个字符)，那么您肯定需要一个更深层次的 CNN(超出了本教程的范围，本教程主要关注数字，因为它们适用于数独)。我在书中讲述了如何用 OpenCV、Tesseract 和 Python 在**T4 数字和字母字符、 **OCR 上训练网络。*****

### **用 Keras 和 TensorFlow 实现我们的数独数字训练脚本**

```py
# import the necessary packages
from pyimagesearch.models import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model after training")
args = vars(ap.parse_args())
```

我们从少量导入开始我们的训练脚本。最值得注意的是，我们正在导入`SudokuNet`(在上一节中讨论过)和`mnist`数据集。手写数字的 MNIST 数据集内置在 TensorFlow/Keras' `datasets`模块中，将根据需要缓存到您的机器上。

```py
# initialize the initial learning rate, number of epochs to train
# for, and batch size
INIT_LR = 1e-3
EPOCHS = 10
BS = 128

# grab the MNIST dataset
print("[INFO] accessing MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

# add a channel (i.e., grayscale) dimension to the digits
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

# scale data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# convert the labels from integers to vectors
le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)
```

您可以在第 17-19 行上配置训练超参数。通过实验，我已经确定了学习率、训练次数和批量的适当设置。

***注:**高级用户可能希望查看我的 [Keras 学习率查找器](https://pyimagesearch.com/2019/08/05/keras-learning-rate-finder/)教程，以帮助自动找到最佳学习率。*

为了使用 MNIST 数字数据集，我们执行以下步骤:

*   将数据集加载到内存中(**第 23 行**)。这个数据集已经被分成训练和测试数据
*   给数字添加一个通道尺寸，表示它们是灰度级的(**第 30 行和第 31 行**)
*   将数据缩放到范围*【0，1】*(**第 30 行和第 31 行**)
*   一键编码标签(**第 34-36 行**)

```py
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR)
model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(
	trainData, trainLabels,
	validation_data=(testData, testLabels),
	batch_size=BS,
	epochs=EPOCHS,
	verbose=1)
```

```py
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testData)
print(classification_report(
	testLabels.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))

# serialize the model to disk
print("[INFO] serializing digit model...")
model.save(args["model"], save_format="h5")
```

### **用 Keras 和 TensorFlow 训练我们的数独数字识别器**

```py
$ python train_digit_classifier.py --model output/digit_classifier.h5
[INFO] accessing MNIST...
[INFO] compiling model...
[INFO] training network...
[INFO] training network...
Epoch 1/10
469/469 [==============================] - 22s 47ms/step - loss: 0.7311 - accuracy: 0.7530 - val_loss: 0.0989 - val_accuracy: 0.9706
Epoch 2/10
469/469 [==============================] - 22s 47ms/step - loss: 0.2742 - accuracy: 0.9168 - val_loss: 0.0595 - val_accuracy: 0.9815
Epoch 3/10
469/469 [==============================] - 21s 44ms/step - loss: 0.2083 - accuracy: 0.9372 - val_loss: 0.0452 - val_accuracy: 0.9854
...
Epoch 8/10
469/469 [==============================] - 22s 48ms/step - loss: 0.1178 - accuracy: 0.9668 - val_loss: 0.0312 - val_accuracy: 0.9893
Epoch 9/10
469/469 [==============================] - 22s 47ms/step - loss: 0.1100 - accuracy: 0.9675 - val_loss: 0.0347 - val_accuracy: 0.9889
Epoch 10/10
469/469 [==============================] - 22s 47ms/step - loss: 0.1005 - accuracy: 0.9700 - val_loss: 0.0392 - val_accuracy: 0.9889
[INFO] evaluating network...
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       980
           1       0.99      1.00      0.99      1135
           2       0.99      0.98      0.99      1032
           3       0.99      0.99      0.99      1010
           4       0.99      0.99      0.99       982
           5       0.98      0.99      0.98       892
           6       0.99      0.98      0.99       958
           7       0.98      1.00      0.99      1028
           8       1.00      0.98      0.99       974
           9       0.99      0.98      0.99      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000

[INFO] serializing digit model...
```

```py
$ ls -lh output
total 2824
-rw-r--r--@ 1 adrian  staff   1.4M Jun  7 07:38 digit_classifier.h5
```

这个`digit_classifier.h5`文件包含我们的 Keras/TensorFlow 模型，我们将在本教程的后面使用它来识别数独板上的数字。

这个模型非常小，可以部署到一个 [Raspberry Pi](https://pyimagesearch.com/category/raspberry-pi/) 甚至是一个[移动设备，比如运行 CoreML 框架的 iPhone](https://pyimagesearch.com/2018/04/23/running-keras-models-on-ios-with-coreml/)。

### **用 OpenCV 在图像中寻找数独拼图板**

至此，我们有了一个可以识别图像中数字的模型；然而，如果数字识别器不能在图像中找到数独拼图板，它就没什么用了。

例如，假设我们向系统展示了以下数独拼图板:

我们如何在图像中找到真正的数独拼图板呢？

一旦我们找到了谜题，我们如何识别每一个单独的细胞？

为了让我们的生活更轻松，我们将实现两个助手工具:

*   ``find_puzzle`` :从输入图像中定位并提取数独拼图板
*   ``extract_digit`` :检查数独拼图板上的每个单元格，并从单元格中提取数字(前提是有数字)

```py
# import the necessary packages
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_puzzle(image, debug=False):
	# convert the image to grayscale and blur it slightly
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 3)
```

*   ``image`` :一张数独拼图的照片。
*   ``debug`` :可选布尔值，表示是否显示中间步骤，以便您可以更好地可视化我们的计算机视觉管道中正在发生的事情。如果你遇到任何问题，我建议设置`debug=True`并使用你的计算机视觉知识来消除任何错误。

```py
	# apply adaptive thresholding and then invert the threshold map
	thresh = cv2.adaptiveThreshold(blurred, 255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	thresh = cv2.bitwise_not(thresh)

	# check to see if we are visualizing each step of the image
	# processing pipeline (in this case, thresholding)
	if debug:
		cv2.imshow("Puzzle Thresh", thresh)
		cv2.waitKey(0)
```

二进制自适应阈值操作允许我们将灰度像素锁定在*【0，255】*像素范围的两端。在这种情况下，我们都应用了二进制阈值，然后反转结果，如下面的**图 5** 所示:

```py
	# find contours in the thresholded image and sort them by size in
	# descending order
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	# initialize a contour that corresponds to the puzzle outline
	puzzleCnt = None

	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points, then we can
		# assume we have found the outline of the puzzle
		if len(approx) == 4:
			puzzleCnt = approx
			break
```

*   从**线 35** 开始循环所有轮廓
*   确定轮廓的周长(**线 37** )
*   [近似轮廓](https://docs.opencv.org/4.4.0/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c) ( **线 38** )
*   检查轮廓是否有四个顶点，如果有，标记为`puzzleCnt`，并且`break`退出循环(**第 42-44 行**

有可能数独网格的轮廓没有找到。既然如此，我们来举个`Exception`:

```py
	# if the puzzle contour is empty then our script could not find
	# the outline of the Sudoku puzzle so raise an error
	if puzzleCnt is None:
		raise Exception(("Could not find Sudoku puzzle outline. "
			"Try debugging your thresholding and contour steps."))

	# check to see if we are visualizing the outline of the detected
	# Sudoku puzzle
	if debug:
		# draw the contour of the puzzle on the image and then display
		# it to our screen for visualization/debugging purposes
		output = image.copy()
		cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
		cv2.imshow("Puzzle Outline", output)
		cv2.waitKey(0)
```

有了拼图的轮廓(手指交叉)，我们就可以对图像进行倾斜校正，从而获得拼图的俯视图:

```py
	# apply a four point perspective transform to both the original
	# image and grayscale image to obtain a top-down bird's eye view
	# of the puzzle
	puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
	warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

	# check to see if we are visualizing the perspective transform
	if debug:
		# show the output warped image (again, for debugging purposes)
		cv2.imshow("Puzzle Transform", puzzle)
		cv2.waitKey(0)

	# return a 2-tuple of puzzle in both RGB and grayscale
	return (puzzle, warped)
```

我们的`find_puzzle`返回签名由所有操作后的原始 RGB 图像和灰度图像的二元组组成，包括最终的四点透视变换。

到目前为止做得很好！

让我们继续朝着解决数独难题的方向前进。现在我们需要一种从数独谜题单元格中提取数字的方法，我们将在下一节中这样做。

### **用 OpenCV 从数独游戏中提取数字**

在上一节中，您学习了如何使用 OpenCV 从图像中检测和提取数独拼图板。

本节将向您展示如何检查数独棋盘中的每个单元格，检测单元格中是否有数字，如果有，提取数字。

继续上一节我们停止的地方，让我们再次打开`puzzle.py`文件并开始工作:

```py
def extract_digit(cell, debug=False):
	# apply automatic thresholding to the cell and then clear any
	# connected borders that touch the border of the cell
	thresh = cv2.threshold(cell, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh = clear_border(thresh)

	# check to see if we are visualizing the cell thresholding step
	if debug:
		cv2.imshow("Cell Thresh", thresh)
		cv2.waitKey(0)
```

在第 80-82 行上，我们的第一步是阈值化和清除任何接触单元格边界的前景像素(例如单元格分割线的任何线条标记)。该操作的结果可以通过**线 85-87** 显示。

让我们看看能否找到手指轮廓:

```py
	# find contours in the thresholded cell
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# if no contours were found than this is an empty cell
	if len(cnts) == 0:
		return None

	# otherwise, find the largest contour in the cell and create a
	# mask for the contour
	c = max(cnts, key=cv2.contourArea)
	mask = np.zeros(thresh.shape, dtype="uint8")
	cv2.drawContours(mask, [c], -1, 255, -1)
```

```py
	# compute the percentage of masked pixels relative to the total
	# area of the image
	(h, w) = thresh.shape
	percentFilled = cv2.countNonZero(mask) / float(w * h)

	# if less than 3% of the mask is filled then we are looking at
	# noise and can safely ignore the contour
	if percentFilled < 0.03:
		return None

	# apply the mask to the thresholded cell
	digit = cv2.bitwise_and(thresh, thresh, mask=mask)

	# check to see if we should visualize the masking step
	if debug:
		cv2.imshow("Digit", digit)
		cv2.waitKey(0)

	# return the digit to the calling function
	return digit
```

伟大的工作实现数字提取管道！

### **实现我们的 OpenCV 数独解谜器**

此时，我们配备了以下组件:

*   我们定制的**数独网模型**在 MNIST 数字数据集上训练，而**驻留在准备使用的磁盘**上
*   **表示提取数独拼图板**并应用透视变换
*   一个**管道来提取数独谜题的单个单元格**中的数字，或者忽略我们认为是噪音的数字
*   在我们的 Python 虚拟环境中安装了 **[py-sudoku 解谜器](https://pypi.org/project/py-sudoku/)**,这让我们不必手工设计算法，让我们可以专注于计算机视觉挑战

我们现在准备把每一部分放在一起构建一个 OpenCV 数独解算器！

打开`solve_sudoku_puzzle.py`文件，让我们完成数独求解器项目:

```py
# import the necessary packages
from pyimagesearch.sudoku import extract_digit
from pyimagesearch.sudoku import find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from Sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained digit classifier")
ap.add_argument("-i", "--image", required=True,
	help="path to input Sudoku puzzle image")
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())
```

与几乎所有 Python 脚本一样，我们选择了一些导入来开始这个聚会。

既然我们现在配备了导入和我们的`args`字典，让我们从磁盘加载我们的(1)数字分类器`model`和(2)输入`--image`:

```py
# load the digit classifier from disk
print("[INFO] loading digit classifier...")
model = load_model(args["model"])

# load the input image from disk and resize it
print("[INFO] processing image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
```

从那里，我们将找到我们的难题，并准备分离其中的细胞:

```py
# find the puzzle in the image and then
(puzzleImage, warped) = find_puzzle(image, debug=args["debug"] > 0)

# initialize our 9x9 Sudoku board
board = np.zeros((9, 9), dtype="int")

# a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
# infer the location of each cell by dividing the warped image
# into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

# initialize a list to store the (x, y)-coordinates of each cell
# location
cellLocs = []
```

```py
# loop over the grid locations
for y in range(0, 9):
	# initialize the current list of cell locations
	row = []

	for x in range(0, 9):
		# compute the starting and ending (x, y)-coordinates of the
		# current cell
		startX = x * stepX
		startY = y * stepY
		endX = (x + 1) * stepX
		endY = (y + 1) * stepY

		# add the (x, y)-coordinates to our cell locations list
		row.append((startX, startY, endX, endY))
```

考虑到数独游戏中的每个单元格，我们以嵌套的方式循环行(**第 48 行**)和列(**第 52 行**)。

在里面，我们使用我们的步长值来确定开始和结束 *(x，y)*—*当前单元格* ( **第 55-58 行**)的坐标。

```py
		# crop the cell from the warped transform image and then
		# extract the digit from the cell
		cell = warped[startY:endY, startX:endX]
		digit = extract_digit(cell, debug=args["debug"] > 0)

		# verify that the digit is not empty
		if digit is not None:
			# resize the cell to 28x28 pixels and then prepare the
			# cell for classification
			roi = cv2.resize(digit, (28, 28))
			roi = roi.astype("float") / 255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)

			# classify the digit and update the Sudoku board with the
			# prediction
			pred = model.predict(roi).argmax(axis=1)[0]
			board[y, x] = pred

	# add the row to our cell locations
	cellLocs.append(row)
```

```py
# construct a Sudoku puzzle from the board
print("[INFO] OCR'd Sudoku board:")
puzzle = Sudoku(3, 3, board=board.tolist())
puzzle.show()

# solve the Sudoku puzzle
print("[INFO] solving Sudoku puzzle...")
solution = puzzle.solve()
solution.show_full()
```

我们继续在终端上打印出解决的谜题( **Line 93** )

当然，如果我们不能在拼图图片上看到答案，这个项目会有什么乐趣呢？让我们现在就开始吧:

```py
# loop over the cell locations and board
for (cellRow, boardRow) in zip(cellLocs, solution.board):
	# loop over individual cell in the row
	for (box, digit) in zip(cellRow, boardRow):
		# unpack the cell coordinates
		startX, startY, endX, endY = box

		# compute the coordinates of where the digit will be drawn
		# on the output puzzle image
		textX = int((endX - startX) * 0.33)
		textY = int((endY - startY) * -0.2)
		textX += startX
		textY += endY

		# draw the result digit on the Sudoku puzzle image
		cv2.putText(puzzleImage, str(digit), (textX, textY),
			cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

# show the output image
cv2.imshow("Sudoku Result", puzzleImage)
cv2.waitKey(0)
```

要用解决方案编号来诠释我们的形象，我们只需:

*   在单元位置和电路板上循环(**第 96-98 行**)
*   解包单元格坐标(**第 100 行**)
*   计算将要绘制文本注释的坐标(**行 104-107** )
*   在我们的拼图板照片上画出每个输出数字(**行 110 和 111** )
*   显示我们解决的数独难题图像(**行 114** )直到按下任何键(**行 115** )

干得好！

让我们在下一部分启动我们的项目。你会对你的努力工作印象深刻的！

### **OpenCV 数独解谜器 OCR 结果**

我们现在准备好测试我们的 OpenV 数独解谜器了！

确保您使用本教程的 ***“下载”*** 部分下载源代码、经过训练的数字分类器和示例数独谜题图像。

从那里，打开一个终端，并执行以下命令:

```py
$ python solve_sudoku_puzzle.py --model output/digit_classifier.h5 \
	--image Sudoku_puzzle.jpg
[INFO] loading digit classifier...
[INFO] processing image...
[INFO] OCR'd Sudoku board:
+-------+-------+-------+
| 8     |   1   |     9 |
|   5   | 8   7 |   1   |
|     4 |   9   | 7     |
+-------+-------+-------+
|   6   | 7   1 |   2   |
| 5   8 |   6   | 1   7 |
|   1   | 5   2 |   9   |
+-------+-------+-------+
|     7 |   4   | 6     |
|   8   | 3   9 |   4   |
| 3     |   5   |     8 |
+-------+-------+-------+

[INFO] solving Sudoku puzzle...

---------------------------
9x9 (3x3) SUDOKU PUZZLE
Difficulty: SOLVED
---------------------------
+-------+-------+-------+
| 8 7 2 | 4 1 3 | 5 6 9 |
| 9 5 6 | 8 2 7 | 3 1 4 |
| 1 3 4 | 6 9 5 | 7 8 2 |
+-------+-------+-------+
| 4 6 9 | 7 3 1 | 8 2 5 |
| 5 2 8 | 9 6 4 | 1 3 7 |
| 7 1 3 | 5 8 2 | 4 9 6 |
+-------+-------+-------+
| 2 9 7 | 1 4 8 | 6 5 3 |
| 6 8 5 | 3 7 9 | 2 4 1 |
| 3 4 1 | 2 5 6 | 9 7 8 |
+-------+-------+-------+
```

**如你所见，我们已经使用 OpenCV、OCR 和深度学习成功解决了数独难题！**

现在，如果你是打赌型的，你可以挑战一个朋友或重要的人，看谁能在你的下一次跨大陆飞行中最快解决 10 个数独谜题！只是不要被抓到抓拍几张照片！

### **学分**

本教程的灵感来自于 Aakash Jhawar 和他的数独解谜器的第一部分和第二部分的[。](https://medium.com/@aakashjhawar/sudoku-solver-using-opencv-and-dl-part-1-490f08701179)

此外，你会注意到**我使用了与 Aakash 做的**相同的示例数独拼图板，不是出于懒惰，而是为了演示如何使用*不同的*计算机视觉和图像处理技术来解决*相同的*拼图。

我真的很喜欢 Aakash 的文章，并推荐 PyImageSearch 的读者也去看看(*尤其是*如果你想从头实现一个数独解算器，而不是使用`py-sudoku`库)。

## **总结**

在本教程中，您学习了如何使用 OpenCV、深度学习和 OCR 实现数独解谜器。

为了在图像中找到并定位数独拼图板，我们利用了 OpenCV 和基本的图像处理技术，包括模糊、阈值处理和轮廓处理等。

为了真正 OCR 数独板上的数字，我们使用 Keras 和 TensorFlow 训练了一个自定义数字识别模型。

将数独板定位器与我们的数字 OCR 模型结合起来，让我们能够快速解决实际的数独难题。

如果你有兴趣了解更多关于 OCR 的知识，我正在用 OpenCV、Tesseract 和 Python 编写一本名为*光学字符识别的新书。*

要了解关于这本书的更多信息，并订购您自己的书(加上预发行折扣和附加内容)，只需点击此处。

**否则，要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！***