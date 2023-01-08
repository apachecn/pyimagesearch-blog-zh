# 使用 scikit-learn ( GridSearchCV)调整网格搜索超参数

> 原文：<https://pyimagesearch.com/2021/05/24/grid-search-hyperparameter-tuning-with-scikit-learn-gridsearchcv/>

在本教程中，您将学习如何使用 scikit-learn 机器学习库和`GridSearchCV`类来网格搜索超参数。我们将把网格搜索应用于一个计算机视觉项目。

![](img/be3a4fc9593813e41df8ff948a9a2787.png)

这篇博客文章是我们关于超参数调优的四部分系列文章的第二部分:

1.  [*用 scikit-learn 和 Python*](https://pyimagesearch.com/2021/05/17/introduction-to-hyperparameter-tuning-with-scikit-learn-and-python/) 调优超参数简介(上周教程)
2.  *使用 scikit-learn ( GridSearchCV)进行网格搜索超参数调整*(今天的帖子)
3.  *使用 scikit-learn、Keras 和 TensorFlow 进行深度学习的超参数调整*(下周发布)
4.  *使用 Keras 调谐器和 TensorFlow 进行简单的超参数调谐*(从现在起两周后的教程)

上周，我们学习了如何将超参数调整到支持向量机(SVM ),该机器被训练用于预测海洋蜗牛的年龄。这是对超参数调整概念的一个很好的介绍，但是它没有演示如何将超参数调整应用到计算机视觉项目中。

今天，我们将建立一个计算机视觉系统来自动识别图像中物体的纹理 T2。我们将使用超参数调整来找到产生最高精度的最佳超参数集。

当您需要在自己的项目中调优超参数时，可以使用本文包含的代码作为起点。

**要学习如何用`GridSearchCV`和 scikit-learn、*网格搜索超参数，继续阅读。***

## **使用 scikit-learn ( GridSearchCV)调整网格搜索超参数**

在本教程的第一部分，我们将讨论:

1.  什么是网格搜索
2.  网格搜索如何应用于超参数调谐
3.  scikit-learn 机器学习库如何通过`GridSearchCV`类实现网格搜索

从那里，我们将配置我们的开发环境，并审查我们的项目目录结构。

然后，我将向您展示如何使用计算机视觉、机器学习和网格搜索超参数调整来调整纹理识别管道的参数，从而实现接近 100%纹理识别精度的系统。

本指南结束时，你将对如何将网格搜索应用于计算机视觉项目的超参数有深刻的理解。

### **什么是超参数网格搜索？**

网格搜索允许我们详尽地测试我们感兴趣的所有可能的超参数配置。

在本教程的后面，我们将调整支持向量机(SVM)的超参数，以获得高精度。SVM 的超参数包括:

1.  **核选择:**线性、多项式、径向基函数
2.  **严格(`C` ):** 典型值在`0.0001`到`1000`的范围内
3.  **内核特定参数:**次数(对于多项式)和伽玛(RBF)

例如，考虑以下可能的超参数列表:

```py
parameters = [
	{"kernel":
		["linear"],
		"C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]},
	{"kernel":
		["poly"],
		"degree": [2, 3, 4],
		"C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]},
	{"kernel":
		["rbf"],
		"gamma": ["auto", "scale"],
		"C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}
]
```

**网格搜索将** ***详尽地*** **测试这些超参数的所有可能组合，为每组训练一个 SVM。**网格搜索将报告最佳超参数(即最大化精度的参数)。

### **配置您的开发环境**

要遵循本指南，您需要在计算机上安装以下库:

*   [OpenCV](https://github.com/opencv/opencv)
*   [scikit-learn](https://scikit-learn.org/stable/)
*   [scikit-image](https://scikit-image.org/)
*   [imutils](https://github.com/jrosebr1/imutils)

幸运的是，这两个包都是 pip 可安装的:

```py
$ pip install opencv-contrib-python
$ pip install scikit-learn
$ pip install scikit-image
$ pip install imutils
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

### **我们的示例纹理数据集**

我们将创建一个计算机视觉和机器学习模型，能够自动识别图像中对象的纹理。

**我们将训练模型识别三种纹理:**

1.  砖
2.  大理石
3.  沙

每个类有 30 幅图像，数据集中总共有 90 幅图像。

我们现在的目标是:

1.  量化数据集中每个图像的纹理
2.  定义我们要搜索的超参数集
3.  使用网格搜索来调整超参数，并找到最大化纹理识别准确性的值

***注:*** *这个数据集是按照我关于* [*用 Google Images 创建图像数据集的教程创建的。*](https://pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/) *我已经在与本教程相关的* ***【下载】*** *中提供了示例纹理数据集。这样，您就不必自己重新创建数据集。*

### **项目结构**

在我们为超参数调优实现网格搜索之前，让我们花点时间回顾一下我们的项目目录结构。

从本教程的 ***【下载】*** 部分开始，访问源代码和示例纹理数据集。

从那里，解压缩档案文件，您应该会找到下面的项目目录:

```py
$ tree . --dirsfirst --filelimit 10
.
├── pyimagesearch
│   ├── __init__.py
│   └── localbinarypatterns.py
├── texture_dataset
│   ├── brick [30 entries exceeds filelimit, not opening dir]
│   ├── marble [30 entries exceeds filelimit, not opening dir]
│   └── sand [30 entries exceeds filelimit, not opening dir]
└── train_model.py

5 directories, 3 files
```

`texture_dataset`包含我们将在其中训练模型的数据集。我们有三个子目录，`brick`、`marble,`和`sand`，每个子目录有 30 张图片。

我们将使用[局部二进制模式(LBPs)](https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/) 来量化纹理数据集中每个图像的内容。LBP 图像描述符在`pyimagesearch`模块内的`localbinarypatterns.py`文件中实现。

`train_model.py`脚本负责:

1.  从磁盘加载`texture_dataset`中的所有图像
2.  使用 LBPs 量化每个图像
3.  在超参数空间上执行网格搜索，以确定优化精度的值

让我们开始实现我们的 Python 脚本。

### **我们的本地二进制模式(LBP)描述符**

我们今天要遵循的本地二进制模式实现来自我之前的教程。虽然为了完整起见，我在这里包含了完整的代码，但我将把对实现的详细回顾推迟到我以前的博客文章中。

说完了，打开你的项目目录结构的`pyimagesearch`模块中的`localbinarypatterns.py`文件，我们就可以开始了:

```py
# import the necessary packages
from skimage import feature
import numpy as np

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
```

**2 号线和 3 号线**导入我们需要的 Python 包。scikit-image 的`feature`子模块包含`local_binary_pattern`函数——该方法根据输入图像计算 LBPs。

接下来，我们定义我们的`describe`函数:

```py
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# return the histogram of Local Binary Patterns
		return hist
```

这个方法接受一个输入`image`(即我们想要计算 LBPs 的图像)和一个小的 epsilon 值。正如我们将看到的，当将结果 LBP 直方图归一化到范围 *[0，1]时，`eps`值防止被零除的错误。*

从那里，**行 15 和 16** 从输入图像计算均匀 LBPs。给定 LBP，然后我们使用 NumPy 构建每个 LBP 类型的直方图(**第 17-19 行**)。

然后将得到的直方图缩放到范围*【0，1】*(**第 22 行和第 23 行**)。

要更详细地回顾我们的 LBP 实现，请务必参考我的教程， [*使用 Python 的本地二进制模式& OpenCV。*](https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/)

### **为超参数调整实施网格搜索**

实现 LBP 图像描述符后，我们可以创建网格搜索超参数调优脚本。

打开项目目录中的`train_model.py`文件，我们将开始:

```py
# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from imutils import paths
import argparse
import time
import cv2
import os
```

**第 2-11 行**导入我们需要的 Python 包。我们的主要进口产品包括:

*   `LocalBinaryPatterns`:负责计算每个输入图像的 LBPs，从而量化纹理
*   `GridSearchCV` : scikit-learn 实现了超参数调谐的网格搜索
*   `SVC`:我们的支持向量机(SVM)用于分类(SVC)
*   `paths`:获取输入数据集目录中所有图像的路径
*   `time`:用于计时网格搜索需要多长时间

接下来，我们有命令行参数:

```py
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())
```

这里我们只有一个命令行参数，`--dataset`，它将指向驻留在磁盘上的`texture_dataset`。

让我们获取图像路径并初始化 LBP 描述符:

```py
# grab the image paths in the input dataset directory
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the local binary patterns descriptor along with
# the data and label lists
print("[INFO] extracting features...")
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []
```

**第 20 行**获取我们的`--dataset`目录中所有输入图像的路径。

然后我们初始化我们的`LocalBinaryPatterns`描述符，以及两个列表:

1.  `data`:存储从每个图像中提取的 LBP
2.  `labels`:包含特定图像的类别标签

现在让我们填充`data`和`labels`:

```py
# loop over the dataset of images
for imagePath in imagePaths:
	# load the image, convert it to grayscale, and quantify it
	# using LBPs
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	# extract the label from the image path, then update the
	# label and data lists
	labels.append(imagePath.split(os.path.sep)[-2])
	data.append(hist)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] constructing training/testing split...")
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	random_state=22, test_size=0.25)
```

在**第 30 行，**我们循环输入图像。

对于每幅图像，我们:

1.  从磁盘加载(**第 33 行**)
2.  将其转换为灰度(**第 34 行**)
3.  计算图像的 LBPs(**行 35** )

然后，我们用特定图像的类别标签更新我们的`labels`列表，并用计算出的 LBP 直方图更新我们的`data`列表。

***注:*** *困惑于我们如何从图像路径中确定类标签？回想一下在`texture_dataset`目录中，有三个子目录，分别对应三个纹理类:砖、大理石和沙子。由于给定图像的类标签包含在文件路径中，我们需要做的就是提取子目录名称，这正是第 39 行* *所做的。*

在运行网格搜索之前，我们首先需要定义要搜索的超参数:

```py
# construct the set of hyperparameters to tune
parameters = [
	{"kernel":
		["linear"],
		"C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]},
	{"kernel":
		["poly"],
		"degree": [2, 3, 4],
		"C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]},
	{"kernel":
		["rbf"],
		"gamma": ["auto", "scale"],
		"C": [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}
]
```

**第 49 行**定义了网格搜索将运行的`parameters`列表。如你所见，我们正在测试三种不同类型的 SVM 核:线性、多项式和径向基函数(RBF)。

每个内核都有自己的一组相关超参数要搜索。

支持向量机往往对超参数选择非常敏感；对于非线性内核来说,*尤其如此。如果我们想要高纹理分类精度，我们*需要*来获得这些正确的超参数选择。*

上面列出的值是您通常想要针对 SVM 和给定内核进行调优的值。

现在让我们在超参数空间上运行网格搜索:

```py
# tune the hyperparameters via a cross-validated grid search
print("[INFO] tuning hyperparameters via grid search")
grid = GridSearchCV(estimator=SVC(), param_grid=parameters, n_jobs=-1)
start = time.time()
grid.fit(trainX, trainY)
end = time.time()

# show the grid search information
print("[INFO] grid search took {:.2f} seconds".format(
	end - start))
print("[INFO] grid search best score: {:.2f}%".format(
	grid.best_score_ * 100))
print("[INFO] grid search best parameters: {}".format(
	grid.best_params_))
```

**第 65 行**初始化我们的`GridSearchCV`，它接受三个参数:

1.  `estimator`:我们正在调整的模型(在这种情况下，是支持向量机分类器)。
2.  `param_grid`:我们希望搜索的超参数空间(即我们的`parameters`列表)。
3.  `n_jobs`:要运行的并行作业的数量。值`-1`意味着将使用机器的所有处理器/内核，从而加快网格搜索过程。

**第 67 行**开始超参数空间的网格搜索。我们用`time()`函数包装`.fit`调用来测量超参数搜索空间需要多长时间。

**一旦网格搜索完成，我们将在终端上显示三条重要信息:**

1.  网格搜索花了多长时间
2.  我们在网格搜索中获得的最佳精度
3.  与我们最高精度模型相关的超参数

**从那里，我们对最佳模型进行全面评估:**

```py
# grab the best model and evaluate it
print("[INFO] evaluating...")
model = grid.best_estimator_
predictions = model.predict(testX)
print(classification_report(testY, predictions))
```

**行 80** 从网格搜索中抓取`best_estimator_`。**这是精确度最高的 SVM。**

***注:*** *超参数搜索完成后，scikit-learn 库总是用我们最高精度的模型填充`grid`的`best_estimator_`变量。*

**第 81 行**使用找到的最佳模型对我们的测试数据进行预测。然后我们在**的第 82 行显示一个完整的分类报告。**

### **计算机视觉项目结果网格搜索**

我们现在准备应用网格搜索来调整纹理识别系统的超参数。

请务必访问本教程的 ***“下载”*** 部分，以检索源代码和示例纹理数据集。

从那里，您可以执行`train_model.py`脚本:

```py
$ time python train_model.py --dataset texture_dataset
[INFO] extracting features...
[INFO] constructing training/testing split...
[INFO] tuning hyperparameters via grid search
[INFO] grid search took 1.17 seconds
[INFO] grid search best score: 86.81%
[INFO] grid search best parameters: {'C': 1000, 'degree': 3, 
	'kernel': 'poly'}
[INFO] evaluating...
              precision    recall  f1-score   support

       brick       1.00      1.00      1.00        10
      marble       1.00      1.00      1.00         5
        sand       1.00      1.00      1.00         8

    accuracy                           1.00        23
   macro avg       1.00      1.00      1.00        23
weighted avg       1.00      1.00      1.00        23

real	1m39.581s
user	1m45.836s
sys	0m2.896s
```

正如你所看到的，我们已经在测试集上获得了 100%的准确率，这意味着我们的 SVM 能够识别我们每一张图像中的纹理。

此外，运行调优脚本只需要 1 分 39 秒。

网格搜索在这里工作得很好，但正如我在上周的教程中提到的那样，随机搜索往往工作得一样好，并且需要更少的时间运行——搜索空间中的超参数越多，网格搜索需要的时间就越长(呈指数增长)。

为了说明这一点，下周，我将向您展示如何使用随机搜索来调整深度学习模型中的超参数。

## **总结**

在本教程中，您学习了如何使用网格搜索来自动调整机器学习模型的超参数。为了实现网格搜索，我们使用了 scikit-learn 库和`GridSearchCV`类。

我们的目标是训练一个计算机视觉模型，它可以自动识别图像中对象的纹理(砖、大理石或沙子)。

培训渠道本身包括:

1.  循环我们数据集中的所有图像
2.  使用[局部二元模式描述符](https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/)(一种常用于量化纹理的图像描述符)量化每幅图像的纹理
3.  使用网格搜索来探索支持向量机的超参数

在调整我们的 SVM 超参数后，我们在纹理识别数据集上获得了 100%的分类准确率。

**要下载这篇文章的源代码(并在未来教程在 PyImageSearch 上发布时得到通知)，*只需在下面的表格中输入您的电子邮件地址！****