# 你的第一个图像分类器:使用 k-NN 分类图像

> 原文：<https://pyimagesearch.com/2021/04/17/your-first-image-classifier-using-k-nn-to-classify-images/>

最近，我们花了相当多的时间讨论图像基础知识、学习类型，甚至是构建我们自己的图像分类器时可以遵循的四步管道。但是我们还没有建立我们自己的真正的图像分类器。

这将在本课中改变。我们将从构建几个助手工具开始，以便于预处理和从磁盘加载图像。从那里，我们将讨论 k-最近邻(k-NN)分类器，这是您第一次接触使用机器学习进行图像分类。事实上，这个算法*非常简单*以至于它根本不做任何实际的“学习”——然而它仍然是一个重要的算法，因此我们可以在未来的课程中了解神经网络如何从数据中学习。

最后，我们将应用 k-NN 算法来识别图像中的各种动物。

## **使用图像数据集**

在处理图像数据集时，我们首先必须考虑数据集的总大小(以字节为单位)。我们的数据集是否足够大，可以放入我们机器上的可用 RAM 中？我们可以像加载大型矩阵或数组一样加载数据集吗？还是数据集太大，超出了我们机器的内存，需要我们将数据集“分块”成段，一次只加载一部分？

对于较小的数据集，我们可以将它们加载到主存中，而不必担心内存管理；然而，对于更大的数据集，我们需要开发一些聪明的方法来有效地处理加载图像，以训练图像分类器(而不会耗尽内存)。

也就是说，在开始使用影像分类算法之前，您应该始终了解数据集的大小。正如我们将在本课的其余部分看到的，花时间组织、预处理和加载数据集是构建图像分类器的一个关键方面。

### **介绍“动物”数据集**

“Animals”数据集是我整理的一个简单的示例数据集，用于演示如何使用简单的机器学习技术以及高级深度学习算法来训练图像分类器。

动物数据集中的图像属于三个不同的类别:**狗**、**猫**和**熊猫**正如你在**图 1** 中看到的，每个类别有 1000 个示例图像。狗和猫的图像是从 [Kaggle 狗对猫挑战赛](http://pyimg.co/ogx37)中采集的，而熊猫的图像是从 [ImageNet 数据集](https://doi.org/10.1007/s11263-015-0816-y)中采集的。

动物数据集仅包含 3，000 张图像，可以很容易地放入我们机器的主内存中，这将使训练我们的模型更快，而不需要我们编写任何“开销代码”来管理否则无法放入内存的数据集。最重要的是，深度学习模型可以在 CPU 或 GPU 上的数据集上快速训练。无论你的硬件设置如何，你都可以使用这个数据集来学习机器学习和深度学习的基础知识。

我们在本课中的目标是利用 k-NN 分类器，尝试仅使用*原始像素强度*来识别图像中的每一个物种(即，不进行特征提取)。正如我们将看到的，原始像素亮度并不适合 k-NN 算法。尽管如此，这是一个重要的基准实验，因此我们可以理解为什么卷积神经网络能够在原始像素强度上获得如此高的精度，而传统的机器学习算法却无法做到这一点。

### **开始使用我们的深度学习工具包**

让我们开始定义我们工具包的项目结构:

```py
|--- pyimagesearch
```

如您所见，我们有一个名为`pyimagesearch`的模块。我们开发的所有代码都将存在于`pyimagesearch`模块中。出于本课的目的，我们需要定义两个子模块:

```py
|--- pyimagesearch
|    |--- __init__.py
|    |--- datasets
|    |    |--- __init__.py
|    |    |--- simpledatasetloader.py
|    |--- preprocessing
|    |    |--- __init__.py
|    |    |--- simplepreprocessor.py
```

`datasets`子模块将开始我们名为`SimpleDatasetLoader`的类的实现。我们将使用这个类从磁盘(可以放入主存)加载小的图像数据集，根据一组函数对数据集中的每个图像进行预处理，然后返回:

1.  图像(即原始像素强度)
2.  与每个图像关联的类别标签

然后我们有了`preprocessing`子模块。有许多预处理方法可以应用于我们的图像数据集，以提高分类精度，包括均值减法、随机面片采样或简单地将图像大小调整为固定大小。在这种情况下，我们的`SimplePreprocessor`类将执行后者——从磁盘加载一个图像，并将其调整为固定大小，忽略纵横比。在接下来的两节中，我们将手动实现`SimplePreprocessor`和`SimpleDatasetLoader`。

***备注:*** 当我们在课程中复习整个`pyimagesearch`模块进行深度学习时，我特意将`__init__.py`文件的解释留给读者作为练习。这些文件只包含快捷方式导入，与理解应用于图像分类的深度学习和机器学习技术无关。如果你是 Python 编程语言的新手，我建议你温习一下包导入的基础知识。

### **基本图像预处理器**

机器学习算法，如 k-NN，SVM，甚至卷积神经网络，都要求数据集中的所有图像都具有固定的特征向量大小。在图像的情况下，这个要求意味着我们的图像必须预处理和缩放，以具有相同的宽度和高度。

有许多方法可以实现这种调整大小和缩放，从尊重原始图像与缩放图像的纵横比的更高级的方法到忽略纵横比并简单地将宽度和高度压缩到所需尺寸的简单方法。确切地说，你应该使用哪种方法实际上取决于你的*变异因素*的复杂性——在某些情况下，忽略纵横比就可以了；在其他情况下，您可能希望保留纵横比。

让我们从基本的解决方案开始:构建一个调整图像大小的图像预处理器，忽略纵横比。打开`simplepreprocessor.py`，然后插入以下代码:

```py
# import the necessary packages
import cv2

class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		# resize the image to a fixed size, ignoring the aspect
		# ratio
		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)
```

**Line 2** 导入我们唯一需要的包，我们的 OpenCV 绑定。然后我们在第 5 行的**上定义了`SimplePreprocessor`类的构造函数。该构造函数需要两个参数，后跟第三个可选参数，每个参数的详细信息如下:**

*   `width`:调整大小后输入图像的目标宽度。
*   `height`:调整大小后输入图像的目标高度。
*   `inter`:可选参数，用于控制调整大小时使用哪种插值算法。

`preprocess`函数在**第 12 行**上定义，需要一个参数——我们想要预处理的输入`image`。

**第 15 行和第 16 行**通过将图像调整到固定的大小`width`和`height`来预处理图像，然后我们返回到调用函数。

同样，这个预处理器根据定义是非常基本的——我们所做的就是接受一个输入图像，将它调整到一个固定的尺寸，然后返回它。然而，当与下一节中的图像数据集加载器结合使用时，该预处理器将允许我们从磁盘快速加载和预处理数据集，使我们能够快速通过我们的图像分类管道，并转移到更重要的方面，*如训练我们的实际分类器*。

### **构建图像加载器**

既然我们的`SimplePreprocessor`已经定义好了，让我们继续讨论`SimpleDatasetLoader`:

```py
# import the necessary packages
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []
```

**第 2-4 行**导入我们需要的 Python 包:NumPy 用于数值处理，`cv2`用于 OpenCV 绑定，`os`用于提取图像路径中子目录的名称。

**第 7 行**定义了`SimpleDatasetLoader`的构造函数，在这里我们可以有选择地传入一列图像预处理程序(例如`SimplePreprocessor`)，它们可以顺序地应用于给定的输入图像。

将这些`preprocessors`指定为一个列表而不是单个值是很重要的——有时我们首先需要将图像的大小调整到一个固定的大小，然后执行某种缩放(比如均值减法),接着将图像数组转换为适合 Keras 的格式。这些预处理程序中的每一个都可以独立实现*，允许我们以一种高效的方式将它们依次应用到图像中。*

 *然后我们可以继续讨论`load`方法，这是`SimpleDatasetLoader`的核心:

```py
  	def load(self, imagePaths, verbose=-1):
		# initialize the list of features and labels
		data = []
		labels = []

		# loop over the input images
		for (i, imagePath) in enumerate(imagePaths):
			# load the image and extract the class label assuming
			# that our path has the following format:
			# /path/to/dataset/{class}/{image}.jpg
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]
```

我们的`load`方法需要一个参数— `imagePaths`，它是一个指定驻留在磁盘上的数据集中图像的文件路径的列表。我们还可以为`verbose`提供一个值。这个“详细级别”可以用来打印控制台的更新，允许我们监控`SimpleDatasetLoader`已经处理了多少图像。

**第 18 行和第 19 行**初始化我们的`data`列表(即图像本身)以及`labels`，图像的类别标签列表。

在**第 22 行**上，我们开始循环每个输入图像。对于这些图像中的每一个，我们从磁盘中加载它(**第 26 行**)并基于文件路径提取类标签(**第 27 行**)。我们假设我们的数据集是根据以下目录结构在磁盘上组织的:

```py
/dataset_name/class/image.jpg
```

`dataset_name`可以是数据集的任何名称，在本例中是`animals`。`class`应该是类标签的名称。对于我们的例子，`class`可以是`dog`、`cat`或`panda`。最后，`image.jpg`是实际图像本身的名称。

基于这种分层目录结构，我们可以保持数据集整洁有序。因此，假设`dog`子目录中的所有图像都是狗的例子是安全的。类似地，我们假设`panda`目录中的所有图像都包含熊猫的例子。

我们审查的几乎每个数据集都将遵循这种分层目录设计结构——我强烈鼓励你对自己的项目也这样做。

既然我们的映像已经从磁盘加载，我们可以对它进行预处理(如果需要的话):

```py
			# check to see if our preprocessors are not None
			if self.preprocessors is not None:
				# loop over the preprocessors and apply each to
				# the image
				for p in self.preprocessors:
					image = p.preprocess(image)

			# treat our processed image as a "feature vector"
			# by updating the data list followed by the labels
			data.append(image)
			labels.append(label)
```

**30 线**快速检查，确保我们的`preprocessors`不是`None`。如果检查通过，我们对第 33 行**上的每个`preprocessors`进行循环，并依次将它们应用于第 34** 行**上的图像——这个动作允许我们形成一个预处理程序链*,它可以应用于数据集中的每个图像。***

一旦图像经过预处理，我们就分别更新`data`和`label`列表(**第 38 行和第 39 行**)。

我们的最后一个代码块只是处理对控制台的打印更新，然后将一个由`data`和`labels`组成的二元组返回给调用函数:

```py
			# show an update every `verbose` images
			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1,
					len(imagePaths)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))
```

如您所见，我们的数据集加载器设计简单；然而，它让我们能够轻松地将任意数量的图像预处理程序应用到我们数据集中的每张图像上。这个数据集加载器的唯一警告是，它假设数据集中的所有图像都可以一次放入主存。

对于太大而不适合系统内存的数据集，我们需要设计一个更复杂的数据集加载器。

## k-NN:一个简单的分类器

k-最近邻分类器*是迄今为止*最简单的机器学习和图像分类算法。事实上，它的*非常简单*，它实际上并没有“学习”任何东西。相反，这种算法直接依赖于特征向量之间的距离(在我们的例子中，是图像的原始 RGB 像素强度)。

简单来说，k-NN 算法通过在 ***k*** *最接近的例子*中找到*最常见的类*来对未知数据点进行分类。在 *k* 个最接近的数据点中的每个数据点进行投票，如**图 2** 所示，票数最高的类别获胜。

为了让 k-NN 算法工作，它首先假设具有相似视觉内容的图像在一个 *n* 维空间中*靠近*。在这里，我们可以看到三类图像，分别标记为*狗*、*猫*和*熊猫*。在这个假想的例子中，我们沿着 x 轴*绘制了动物皮毛的“蓬松度”,沿着 y 轴*绘制了皮毛的“亮度”。在我们的 n 维空间中，每一个动物数据点都相对紧密地组合在一起。这意味着两个*猫*图像之间的距离*比一只*猫*和一只*狗*之间的距离*小得多。

然而，为了应用 k-NN 分类器，我们首先需要选择一个距离度量或相似性函数。常见的选择包括欧几里德距离(通常称为 L2 距离):

**【①**

 **但是，也可以使用其他距离度量，例如曼哈顿/城市街区(通常称为 L1 距离):

**②![d(\pmb{p},\pmb{q}) = \sum\limits_{i=1}^{N}|q_{i} - p_{i}|](img/3891f8d50c33186891f66a3897321a0b.png "d(\pmb{p},\pmb{q}) = \sum\limits_{i=1}^{N}|q_{i} - p_{i}|")**

 **实际上，您可以使用最适合您的数据(并给出最佳分类结果)的任何距离度量/相似性函数。然而，在本课的剩余部分，我们将使用最流行的距离度量:欧几里德距离。

### **一个工作过的 k-NN 例子**

至此，我们理解了 k-NN 算法的原理。我们知道它依赖于特征向量/图像之间的距离来进行分类。我们知道它需要一个距离/相似性函数来计算这些距离。

但是我们实际上如何*让*成为一个分类呢？要回答这个问题，我们来看看**图 3** 。这里我们有一个三种动物的数据集——*狗*、*猫*和*熊猫*——我们根据它们的*蓬松度*和*皮毛亮度*来绘制它们。

我们还插入了一个“未知动物”，我们试图只使用一个**单邻居**(即 *k* = 1)对其进行分类。在这种情况下，离输入图像最近的动物是狗数据点；因此我们的输入图像应该被归类为*狗*。

让我们尝试另一种“未知动物”，这次使用 *k* = 3 ( **图 4** )。我们在*前三名结果*中发现了*两只猫*和*一只熊猫*。由于*猫*类别的票数最多，我们将输入图像分类为*猫*。

我们可以针对不同的 *k* 值继续执行该过程，但是无论 *k* 变得多大或多小，原则都保持不变——在 *k* 个最接近的训练点中拥有最多票数的类别获胜，并被用作输入数据点的标签。

***备注:*** 在平局的情况下，k-NN 算法随机选择一个平局的类标签。

### **k-NN 超参数**

在运行 k-NN 算法时，我们关心两个明显的超参数。第一个很明显: *k* 的值。 *k* 的最优值是多少？如果它太小(例如， *k* = 1)，那么我们会提高效率，但容易受到噪声和离群数据点的影响。然而，如果 *k* 太大，那么我们就有*过度平滑*我们的分类结果并增加偏差的风险。

我们应该考虑的第二个参数是实际距离度量。欧氏距离是最佳选择吗？曼哈顿距离呢？

在下一节中，我们将在动物数据集上训练 k-NN 分类器，并在测试集上评估该模型。我鼓励您尝试不同的`k`值和不同的距离指标，注意性能的变化。

### **实现 k-NN**

本部分的目标是在动物数据集的*原始像素亮度*上训练一个 k-NN 分类器，并使用它对未知动物图像进行分类。

*   **步骤#1 —收集我们的数据集:**动物数据集由 3，000 张图像组成，每只狗、猫和熊猫分别有 1，000 张图像。每个图像都用 RGB 颜色空间表示。我们将对每张图片进行预处理，将其调整为 32×32 像素的*。考虑到三个 RGB 通道，调整后的图像尺寸意味着数据集中的每个图像都由*32×32×3 = 3072 个*整数表示。*
*   **步骤# 2——拆分数据集:**对于这个简单的例子，我们将使用*对数据进行两次*拆分。一部分用于培训，另一部分用于测试。我们将省略超参数调优的验证集，把它作为一个练习留给读者。
*   **步骤# 3-训练分类器:**我们的 k-NN 分类器将在训练集中的图像的原始像素强度上进行训练。
*   **步骤#4 —评估:**一旦我们的 k-NN 分类器被训练，我们就可以在测试集上评估性能。

让我们开始吧。打开一个新文件，将其命名为`knn.py`，并插入以下代码:

```py
# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

```

**第 2-9 行**导入我们需要的 Python 包。需要注意的最重要的是:

*   **第 2 行:**`KNeighborsClassifier`是我们对 k-NN 算法的实现，由 scikit-learn 库提供。
*   **第 3 行:** `LabelEncoder`，一个帮助实用程序，将表示为字符串的标签转换为整数，其中每个类标签有一个唯一的整数(应用机器学习时的常见做法)。
*   **第 4 行:**我们将导入`train_test_split`函数，这是一个方便的函数，用来帮助我们创建训练和测试分割。
*   **第 5 行:**`classification_report`函数是另一个实用函数，用于帮助我们评估分类器的性能，并将格式良好的结果表打印到控制台。

您还可以看到我们分别在**第 6 行和第 7 行**上导入的`SimplePreprocessor`和`SimpleDatasetLoader`的实现。

接下来，让我们解析我们的命令行参数:

```py
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())
```

我们的脚本需要一个命令行参数，后面跟着两个可选参数，每个参数如下所示:

*   `--dataset`:输入图像数据集在磁盘上的路径。
*   `--neighbors`:可选，使用 k-NN 算法时要应用的邻居数量 *k* 。
*   `--jobs`:可选，计算输入数据点与训练集之间的距离时要运行的并发作业的数量。值`-1`将使用处理器上所有可用的内核。

既然我们的命令行参数已经被解析，我们就可以获取数据集中图像的文件路径，然后加载并预处理它们(分类管道中的**步骤#1** ):

```py
# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
	data.nbytes / (1024 * 1024.0)))

```

**第 23 行**获取数据集中所有图像的文件路径。然后我们初始化我们的`SimplePreprocessor`,用来在**第 27 行**上将每个图像的大小调整为 *32×32* 像素。

在第 28 行的**上初始化`SimpleDatasetLoader`，为我们实例化的`SimplePreprocessor`提供一个参数(这意味着`sp`将被应用到数据集中的每个图像的*)。***

对第 29 行**上的`.load`的调用从磁盘加载了我们实际的图像数据集。这个方法返回一个二元组的`data`(每张图片调整到 *32×32* 像素)以及每张图片的`labels`。**

从磁盘加载我们的图像后，`data` NumPy 数组有一个`.shape`的`(3000, 32, 32, 3)`，表示数据集中有 3000 个图像，每个 *32×32* 像素有 3 个通道。

然而，为了应用 k-NN 算法，我们需要将我们的图像从 3D 表示“展平”为像素强度的单一列表。我们完成这个，**第 30 行**调用`data` NumPy 数组上的`.reshape`方法，将 *32×32×3* 图像展平成一个形状为`(3000, 3072)`的数组。实际的图像数据一点也没有改变——图像被简单地表示为 3000 个条目的列表，每个条目为 3072-dim(*32×32×3 = 3072*)。

为了演示在内存中存储这 3，000 个图像需要多少内存，**第 33 行和第 34 行**计算数组消耗的字节数，然后将该数字转换为兆字节。

接下来，让我们构建我们的培训和测试分割(管道中的**步骤#2** ):

```py
# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)
```

**第 37 行和第 38 行**将我们的`labels`(表示为字符串)转换成整数，其中我们每个类有*个唯一的整数*。这种转换允许我们将*猫*类映射到整数`0`，将*狗*类映射到整数`1`，将*熊猫*类映射到整数`2`。许多机器学习算法假设类别标签被编码为整数，所以我们养成执行这一步骤的习惯是很重要的。

计算我们的训练和测试分割是由第 42 行和第 43 行上的`train_test_split`函数处理的。这里，我们将我们的`data`和`labels`分成两个不同的集合:75%的数据用于训练，25%用于测试。

通常使用变量 *X* 来指代包含我们将用于训练和测试的数据点的数据集，而 *y* 指代类标签(您将在关于 *[参数化学习](https://pyimagesearch.com/2016/08/22/an-intro-to-linear-classification-with-python/)* 的课程中了解更多)。因此，我们使用变量`trainX`和`testX`分别指代*训练和测试示例*。变量`trainY`和`testY`是我们的*训练和测试标签*。你会在我们的课程*和*以及你可能阅读的其他机器学习书籍、课程和教程中看到这些常见的符号。

最后，我们能够创建我们的 k-NN 分类器并评估它(图像分类管道中的**步骤#3 和#4** ):

```py
# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),
	target_names=le.classes_))
```

**第 47 和 48 行**初始化`KNeighborsClassifier`类。对第 49**行**上的`.fit`方法的调用“训练”了分类器，尽管这里没有实际的“学习”在进行——k-NN 模型只是在内部存储了`trainX`和`trainY`数据，因此它可以通过计算输入数据和`trainX`数据之间的距离，在测试集上创建预测。

**第 50 行和第 51 行**使用`classification_report`函数评估我们的分类器。这里我们需要提供`testY`类标签，来自`model`的*预测类标签*，以及可选的类标签名称(例如，“狗”、“猫”、“熊猫”)。

### **k-NN 结果**

要运行我们的 k-NN 分类器，请执行以下命令:

```py
$ python knn.py --dataset ../datasets/animals
```

然后，您应该会看到类似以下内容的输出:

```py
[INFO] loading images...
[INFO] processed 500/3000
[INFO] processed 1000/3000
[INFO] processed 1500/3000
[INFO] processed 2000/3000
[INFO] processed 2500/3000
[INFO] processed 3000/3000
[INFO] features matrix: 8.8MB
[INFO] evaluating k-NN classifier...
              precision    recall  f1-score   support

        cats       0.37      0.52      0.43       239
        dogs       0.35      0.43      0.39       249
       panda       0.70      0.28      0.40       262

    accuracy                           0.41       750
   macro avg       0.47      0.41      0.41       750
weighted avg       0.48      0.41      0.40       750
```

请注意，我们的特征矩阵对于 3000 幅图像只消耗了 8.8MB 的内存，每幅图像的大小为 32×32×3(T1)，这个数据集可以很容易地(T2)存储在现代机器的内存中，没有任何问题。

评估我们的分类器，我们看到我们获得了 *48%* 的准确度——对于一个根本不做任何真正“学习”的分类器来说，这个准确度还不错，因为随机猜测正确答案的概率是 1 */* 3。

然而，检查每个类标签的准确性是很有趣的。“熊猫”类有 70%的时间被正确分类，可能是因为熊猫大部分是黑白的，因此这些图像在我们的 3 *，* 072-dim 空间中靠得更近。

狗和猫分别获得 35%和 37%的低得多的分类准确度。这些结果可以归因于这样一个事实，即狗和猫可能有非常相似的皮毛色调，它们皮毛的颜色不能用来区分它们。背景噪声(如后院的草地、动物休息的沙发的颜色等。)也可以“混淆”k-NN 算法，因为它不能学习这些物种之间的任何区别模式。这种混乱是 k-NN 算法的主要缺点之一:*虽然它很简单，但它也无法从数据中学习*。

### **k-NN 的利弊**

k-NN 算法的一个主要优点是它的实现和理解非常简单。此外，训练分类器绝对不需要时间，因为我们需要做的只是存储我们的数据点，以便以后计算到它们的距离并获得我们的最终分类。

然而，我们在分类时为这种简单付出了代价。对新的测试点进行分类需要与我们训练数据中的每一个数据点进行比较*，这会扩展 *O(N)* ，使得处理更大的数据集在计算上非常困难。*

我们可以通过使用**近似最近邻(ANN)** 算法(例如 [kd-trees](http://doi.acm.org/10.1145/361002.361007) 、 [FLANN](https://ieeexplore.ieee.org/document/6809191) 、随机投影( [Dasgupta，2000](http://dl.acm.org/citation.cfm?id=647234.719759))；[宾汉姆和曼尼拉，2001](http://doi.acm.org/10.1145/502512.502546)；[达斯古普塔和古普塔，2003](http://dx.doi.org/10.1002/rsa.10073))；等等。);然而，使用这些算法要求我们用空间/时间复杂度来换取最近邻算法的“正确性”，因为我们是在执行近似。也就是说，在许多情况下，使用 k-NN 算法的努力和较小的精度损失是非常值得的。这种行为与大多数机器学习算法(以及*所有*神经网络)形成对比，在这些算法中，我们花费大量时间预先训练我们的模型以获得高精度，反过来，在测试时有*非常快速的*分类。

最后，k-NN 算法更适合低维特征空间(图像不是)。高维特征空间中的距离通常是不直观的，您可以在[Pedro Domingos’(2012)的优秀论文](http://doi.acm.org/10.1145/2347736.2347755)中了解更多信息。

同样重要的是要注意，k-NN 算法实际上并没有“学习”任何东西——如果它犯了错误，它也不能让自己变得更聪明；它只是简单地依靠一个 *n* 维空间中的距离来进行分类。

考虑到这些缺点，为什么还要研究 k-NN 算法呢？原因是算法*简单*。这是容易理解的*。最重要的是，它给了我们一个基准*,我们可以用它来比较神经网络和卷积神经网络，因为我们将继续学习剩下的课程。**

 **## **总结**

在本课中，我们学习了如何构建一个简单的图像处理器并将图像数据集加载到内存中。然后我们讨论了*k-最近邻分类器*或者简称为 *k-NN* 。

k-NN 算法通过将未知数据点与训练集中的每个数据点进行比较来对未知数据点进行分类。使用距离函数或相似性度量来进行比较。然后从训练集中最 *k* 个相似的例子中，累计每个标签的“投票”数。最高票数的类别“胜出”,并被选为总体分类。

虽然简单直观，但 k-NN 算法有许多缺点。首先，它实际上并没有“学习”任何东西——如果算法出错，它没有办法为后面的分类“纠正”和“改进”自己。其次，在没有专门的数据结构的情况下，k-NN 算法随着数据点的数量而线性扩展，这不仅使其在高维中的使用具有实际挑战性，而且就其使用而言在理论上也是有问题的( [Domingos，2012](http://doi.acm.org/10.1145/2347736.2347755) )。现在我们已经使用 k-NN 算法获得了图像分类的基线，我们可以继续进行*参数化学习*，这是所有深度学习和神经网络建立的基础。使用参数化学习，我们实际上可以*从我们的输入数据中学习*，并发现潜在的模式。这个过程将使我们能够建立高精度的图像分类器，使 k-NN 的性能大打折扣。*******